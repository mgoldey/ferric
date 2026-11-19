//! Ab initio molecular dynamics front end (`run_aimd`); see `ferric-md`.
//!
//! [`ScfForceProvider`] adapts the SCF energy + analytic gradient evaluators
//! the geometry optimizers use (RHF/RKS, UHF/UKS, ROHF/ROKS) to the
//! [`ForceProvider`] trait, so [`ferric_md::run_md`] can propagate nuclei on
//! the Born-Oppenheimer surface. Each force evaluation moves a template
//! [`Molecule`] to the integrator's coordinates and runs one SCF; the
//! configuration goes through [`RhfConfig::for_geometry_task`] exactly as in
//! [`crate::optimize::optimize_geometry_with_scf_correction`], and the
//! orbitals of the previous step seed the next one through [`ScfWarmStart`]
//! when [`AimdConfig::scf_warm_start`] is set.
//!
//! The energy is conserved only as well as the SCF is converged: an
//! unconverged density makes the force inconsistent with the energy, which
//! shows up as a drift of the conserved quantity. A tight `density_conv` is
//! the caller's choice; the SCF defaults are not changed here.

use crate::optimize::{
    add_correction, compute_energy_and_gradient_rohf, compute_energy_and_gradient_uhf,
    compute_energy_gradient_and_result, require_converged,
};
use crate::result::{ScfResult, Spin};
use crate::rhf::RhfConfig;
use crate::warm_start::ScfWarmStart;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_md::{run_md, ForceProvider, MdConfig, MdObserver, MdResult, MdSystem};
use ndarray::Array2;

/// An additive energy/gradient correction evaluated after each SCF, with the
/// contract of [`crate::optimize::optimize_geometry_with_scf_correction`]:
/// `(energy, Option<gradient>)` in Hartree and Hartree/Bohr; an energy
/// without a gradient is an error and `(0.0, None)` adds nothing.
pub type ScfCorrection<'a> =
    dyn FnMut(&Molecule, &ScfResult) -> Result<(f64, Option<Array2<f64>>), FerricError> + 'a;

/// Settings of an ab initio MD run beyond the SCF configuration.
#[derive(Debug, Clone)]
pub struct AimdConfig {
    /// Integrator, thermostat, step count and output stride.
    pub md: MdConfig,
    /// Start each step's SCF from the previous step's converged orbitals.
    /// Default `true`.
    pub scf_warm_start: bool,
    /// Per-atom masses (u); `None` uses the standard atomic masses.
    pub masses_amu: Option<Vec<f64>>,
}

impl Default for AimdConfig {
    fn default() -> Self {
        Self {
            md: MdConfig::default(),
            scf_warm_start: true,
            masses_amu: None,
        }
    }
}

/// Result of [`run_aimd`].
#[derive(Debug, Clone)]
#[must_use]
pub struct AimdResult {
    /// The trajectory and energies.
    pub md: MdResult,
    /// The molecule at the final geometry.
    pub mol: Molecule,
    /// The SCF result of the last force evaluation (final geometry).
    pub last_scf: ScfResult,
    /// SCF iterations spent per force evaluation, in evaluation order
    /// (including the iterations of a failed warm attempt retried cold).
    pub scf_iterations: Vec<usize>,
    /// [`ScfWarmStart::summary`] of the run (`None` when warm start is off).
    pub warm_start_summary: Option<String>,
}

/// A [`ForceProvider`] that evaluates the SCF energy and analytic gradient.
pub struct ScfForceProvider<'a> {
    ctx: &'a ParallelContext,
    mol: Molecule,
    basis_name: &'a str,
    op: Operator,
    scf_config: RhfConfig,
    spin: Spin,
    warm: ScfWarmStart,
    correction: Option<&'a mut ScfCorrection<'a>>,
    scf_iterations: Vec<usize>,
    last_scf: Option<ScfResult>,
}

impl<'a> ScfForceProvider<'a> {
    /// A provider for `mol` (the template whose coordinates every evaluation
    /// overwrites). `scf_config` is passed through
    /// [`RhfConfig::for_geometry_task`]; `spin` selects the reference.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &'a ParallelContext,
        mol: &Molecule,
        basis_name: &'a str,
        op: Operator,
        scf_config: &RhfConfig,
        spin: Spin,
        scf_warm_start: bool,
        correction: Option<&'a mut ScfCorrection<'a>>,
    ) -> Self {
        Self {
            ctx,
            mol: mol.clone(),
            basis_name,
            op,
            scf_config: scf_config.for_geometry_task(),
            spin,
            warm: ScfWarmStart::new(scf_warm_start),
            correction,
            scf_iterations: Vec::new(),
            last_scf: None,
        }
    }

    /// The template molecule at the most recently evaluated geometry.
    pub fn molecule(&self) -> &Molecule {
        &self.mol
    }

    /// SCF iterations per evaluation so far.
    pub fn scf_iterations(&self) -> &[usize] {
        &self.scf_iterations
    }

    /// The SCF result of the most recent evaluation.
    pub fn last_scf(&self) -> Option<&ScfResult> {
        self.last_scf.as_ref()
    }

    /// [`ScfWarmStart::summary`] of the evaluations so far.
    pub fn warm_start_summary(&self) -> Option<String> {
        self.warm.summary()
    }

    fn set_coords(&mut self, coords: &Array2<f64>) -> Result<(), FerricError> {
        if coords.dim() != (self.mol.atoms.len(), 3) {
            return Err(FerricError::General(format!(
                "AIMD: coordinates have shape {:?} but the molecule has {} atoms",
                coords.dim(),
                self.mol.atoms.len()
            )));
        }
        for (atom, row) in self.mol.atoms.iter_mut().zip(coords.rows()) {
            atom.x = row[0];
            atom.y = row[1];
            atom.zpos = row[2];
        }
        Ok(())
    }

    fn scf_energy_gradient(&mut self) -> Result<(f64, Array2<f64>, ScfResult), FerricError> {
        let (ctx, mol, basis, op, cfg) = (
            self.ctx,
            &self.mol,
            self.basis_name,
            self.op,
            &self.scf_config,
        );
        let warm = &mut self.warm;
        match self.spin {
            Spin::Restricted => compute_energy_gradient_and_result(ctx, mol, basis, op, cfg, warm),
            Spin::Unrestricted => compute_energy_and_gradient_uhf(ctx, mol, basis, op, cfg, warm),
            Spin::RestrictedOpen => {
                compute_energy_and_gradient_rohf(ctx, mol, basis, op, cfg, warm)
            }
        }
    }

    /// SCF iterations of the latest evaluation, counting a failed warm
    /// attempt that was retried cold.
    fn latest_iterations(&self, scf: &ScfResult) -> usize {
        let wasted = self
            .warm
            .records()
            .last()
            .map_or(0, |r| r.wasted_iterations);
        scf.iterations + wasted
    }

    fn into_result(self, md: MdResult) -> Result<AimdResult, FerricError> {
        let warm_start_summary = self.warm.summary();
        let last_scf = self.last_scf.ok_or_else(|| {
            FerricError::General("AIMD: no force evaluation was performed".to_string())
        })?;
        Ok(AimdResult {
            md,
            mol: self.mol,
            last_scf,
            scf_iterations: self.scf_iterations,
            warm_start_summary,
        })
    }
}

/// Short method label: reference (`rhf`/`rks`, `uhf`/`uks`, `rohf`/`roks`),
/// the functional when there is one, and the basis.
fn method_label(spin: Spin, xc: Option<&str>, basis_name: &str) -> String {
    let reference = match (spin, xc.is_some()) {
        (Spin::Restricted, false) => "rhf",
        (Spin::Restricted, true) => "rks",
        (Spin::Unrestricted, false) => "uhf",
        (Spin::Unrestricted, true) => "uks",
        (Spin::RestrictedOpen, false) => "rohf",
        (Spin::RestrictedOpen, true) => "roks",
    };
    match xc {
        Some(xc) => format!("{reference}:{xc}/{basis_name}"),
        None => format!("{reference}/{basis_name}"),
    }
}

impl ForceProvider for ScfForceProvider<'_> {
    fn n_atoms(&self) -> usize {
        self.mol.atoms.len()
    }

    fn energy_gradient(&mut self, coords: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        self.set_coords(coords)?;
        let (e, g, scf) = self.scf_energy_gradient()?;
        require_converged(&scf)?;
        let (de, dg) = match self.correction.as_mut() {
            Some(correction) => correction(&self.mol, &scf)?,
            None => (0.0, None),
        };
        let out = add_correction(e, g, de, dg)?;
        let iterations = self.latest_iterations(&scf);
        self.scf_iterations.push(iterations);
        self.last_scf = Some(scf);
        Ok(out)
    }

    fn label(&self) -> String {
        method_label(self.spin, self.scf_config.xc.as_deref(), self.basis_name)
    }
}

/// Born-Oppenheimer molecular dynamics on the SCF surface of `mol`.
///
/// `spin` selects RHF/RKS ([`Spin::Restricted`]), UHF/UKS or ROHF/ROKS; a
/// functional in `scf_config.xc` makes it Kohn-Sham. `scf_config` goes through
/// [`RhfConfig::for_geometry_task`] like every geometry driver. `correction`
/// (e.g. dispersion) is added to every step's energy and gradient under the
/// rules of [`ScfCorrection`]. Masses come from [`AimdConfig::masses_amu`] or
/// the standard atomic masses. Every SCF must converge; otherwise the run
/// stops with the SCF error.
#[allow(clippy::too_many_arguments)]
pub fn run_aimd(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    scf_config: &RhfConfig,
    spin: Spin,
    config: &AimdConfig,
    correction: Option<&mut ScfCorrection<'_>>,
    observers: &mut [&mut dyn MdObserver],
) -> Result<AimdResult, FerricError> {
    let system = MdSystem::from_molecule(mol, config.masses_amu.as_deref())?;
    // Shorten the closure's lifetime bound to the provider's (an unsizing
    // coercion that does not happen through `Option` on its own).
    let correction = correction.map(|c| c as &mut ScfCorrection<'_>);
    let mut provider = ScfForceProvider::new(
        ctx,
        mol,
        basis_name,
        op,
        scf_config,
        spin,
        config.scf_warm_start,
        correction,
    );
    let md = run_md(&system, &mut provider, &config.md, observers)?;
    provider.into_result(md)
}
