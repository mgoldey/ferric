//! Molecule-to-energy drivers for CASSCF and SC-NEVPT2.
//!
//! Each driver runs the exact-J/K Coulomb RHF that starts the calculation
//! ([`rhf_start`](crate::driver::rhf_start)), then two-step CASSCF
//! ([`run_casscf`](crate::run_casscf)) or a CASSCF or CAS-CI reference plus
//! SC-NEVPT2 ([`nevpt2_on_rhf`](crate::driver::nevpt2_on_rhf)). The Python
//! bindings (`ferric.run_casscf`, `ferric.run_nevpt2`) and the CLI kinds
//! (`method.kind = "casscf"`, `"nevpt2"`) call these functions, so the same
//! input gives the same number on both surfaces.

use crate::{
    run_cas_ci, run_casscf, sc_nevpt2, CasCiConfig, CasScfConfig, Nevpt2Config, Nevpt2Reference,
    Nevpt2Result,
};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

/// Convergence settings of the RHF start; `None` keeps the [`RhfConfig`]
/// default.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ScfStart {
    /// SCF iteration cap.
    pub max_iter: Option<usize>,
    /// SCF energy-change threshold (Ha).
    pub energy_conv: Option<f64>,
    /// SCF density-change threshold.
    pub density_conv: Option<f64>,
}

/// The exact-J/K Coulomb RHF of `mol` in `prep` with Schwarz screening and
/// the [`RhfConfig`] defaults, `start`'s overrides and the memory budget
/// (`three_index_budget_bytes`; `None` is the auto-detected budget). An
/// unconverged SCF is a [`FerricError::ScfConvergence`].
pub fn rhf_start(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    start: ScfStart,
    memory_budget_bytes: Option<usize>,
) -> Result<ScfResult, FerricError> {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep)?;
    let mut cfg = RhfConfig {
        three_index_budget_bytes: memory_budget_bytes.unwrap_or(0),
        ..Default::default()
    };
    cfg.max_iter = start.max_iter.unwrap_or(cfg.max_iter);
    cfg.energy_conv = start.energy_conv.unwrap_or(cfg.energy_conv);
    cfg.density_conv = start.density_conv.unwrap_or(cfg.density_conv);
    let rhf = solve_rhf(ctx, mol, prep, op, &bounds, &cfg)?;
    if !rhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        });
    }
    Ok(rhf)
}

/// The doubly occupied core count `(nelec − n_α − n_β) / 2` that the active
/// electrons leave out of `nelec`. An odd or negative remainder is an error
/// whose text starts with `caller`.
pub fn default_n_core(
    nelec: usize,
    n_elec_active: (usize, usize),
    caller: &str,
) -> Result<usize, String> {
    let n_act_el = n_elec_active.0 + n_elec_active.1;
    if n_act_el > nelec || !(nelec - n_act_el).is_multiple_of(2) {
        return Err(format!(
            "{caller}: {n_act_el} active electrons do not leave an even core count out of {nelec}"
        ));
    }
    Ok((nelec - n_act_el) / 2)
}

/// The reference SC-NEVPT2 is computed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nevpt2On {
    /// Two-step CASSCF from the RHF orbitals.
    Casscf,
    /// CAS-CI on the canonical RHF orbitals.
    Casci,
}

impl Nevpt2On {
    /// `"casscf"` or `"casci"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Nevpt2On::Casscf => "casscf",
            Nevpt2On::Casci => "casci",
        }
    }

    /// Strict parse of `"casscf"` / `"casci"`; any other spelling is an
    /// error, never a default.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s {
            "casscf" => Ok(Nevpt2On::Casscf),
            "casci" => Ok(Nevpt2On::Casci),
            other => Err(format!(
                "reference must be \"casscf\" or \"casci\", got {other:?}"
            )),
        }
    }
}

/// SC-NEVPT2 on a converged reference, with the reference's own counters.
#[derive(Debug, Clone)]
pub struct Nevpt2Run {
    /// The SC-NEVPT2 result.
    pub result: Nevpt2Result,
    /// CASSCF macro iterations, or CAS-CI Davidson iterations.
    pub reference_iterations: usize,
    /// The CASSCF orbital-gradient norm; `None` on a CAS-CI reference.
    pub reference_grad_norm: Option<f64>,
}

/// The CAS-CI settings of the [`Nevpt2On::Casci`] reference: `cas`'s
/// partition, CI settings and budget, the active window starting at
/// `cas.n_core`, no RDMs.
pub fn casci_config_of(cas: &CasScfConfig) -> CasCiConfig {
    CasCiConfig {
        n_active: cas.n_active,
        n_elec_active: cas.n_elec_active,
        active_start: cas.n_core,
        conv_thresh: cas.ci_conv_thresh,
        max_subspace: cas.ci_max_subspace,
        max_iter: cas.ci_max_iter,
        memory_budget_bytes: cas.memory_budget_bytes,
        compute_rdms: false,
    }
}

/// SC-NEVPT2 on the reference `on` built from the converged RHF `rhf`:
/// [`run_casscf`] with `cas`, or [`run_cas_ci`] with [`casci_config_of`]
/// (`cas`'s orbital-optimization settings are then not read). The
/// SC-NEVPT2 settings are the [`Nevpt2Config`] defaults with `cas`'s memory
/// budget. An unconverged reference is an error.
pub fn nevpt2_on_rhf(
    mol: &Molecule,
    prep: &PreparedBasis,
    rhf: &ScfResult,
    on: Nevpt2On,
    cas: &CasScfConfig,
) -> Result<Nevpt2Run, FerricError> {
    let nev = Nevpt2Config {
        memory_budget_bytes: cas.memory_budget_bytes,
        ..Default::default()
    };
    match on {
        Nevpt2On::Casscf => {
            let r = run_casscf(mol, prep, rhf, cas)?;
            let reference = Nevpt2Reference::from_casscf(&r, cas)?;
            let result = sc_nevpt2(mol, prep, &reference, &nev)?;
            Ok(Nevpt2Run {
                result,
                reference_iterations: r.iterations,
                reference_grad_norm: Some(r.grad_norm),
            })
        }
        Nevpt2On::Casci => {
            let cfg = casci_config_of(cas);
            let ci = run_cas_ci(mol, prep, rhf, cfg)?;
            let reference = Nevpt2Reference::from_casci(&ci, rhf.mos_r(), &cfg)?;
            let result = sc_nevpt2(mol, prep, &reference, &nev)?;
            Ok(Nevpt2Run {
                result,
                reference_iterations: ci.iterations,
                reference_grad_norm: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_n_core_needs_an_even_remainder() {
        assert_eq!(default_n_core(10, (2, 2), "x"), Ok(3));
        assert_eq!(default_n_core(16, (5, 3), "x"), Ok(4));
        let e = default_n_core(10, (2, 1), "run_casscf").unwrap_err();
        assert_eq!(
            e,
            "run_casscf: 3 active electrons do not leave an even core count out of 10"
        );
        assert!(default_n_core(2, (2, 2), "x").is_err());
    }

    #[test]
    fn nevpt2_reference_parse_is_strict() {
        for on in [Nevpt2On::Casscf, Nevpt2On::Casci] {
            assert_eq!(Nevpt2On::parse_config_str(on.as_str()), Ok(on));
        }
        for bad in ["CASSCF", "rhf", ""] {
            let e = Nevpt2On::parse_config_str(bad).unwrap_err();
            assert!(e.starts_with("reference must be"), "{e}");
        }
    }

    #[test]
    fn casci_config_takes_the_partition_and_ci_settings() {
        let cas = CasScfConfig {
            n_active: 4,
            n_elec_active: (2, 2),
            n_core: 3,
            ci_conv_thresh: 1e-11,
            ci_max_subspace: 30,
            ci_max_iter: 77,
            memory_budget_bytes: Some(1 << 30),
            ..Default::default()
        };
        let c = casci_config_of(&cas);
        assert_eq!(
            c,
            CasCiConfig {
                n_active: 4,
                n_elec_active: (2, 2),
                active_start: 3,
                conv_thresh: 1e-11,
                max_subspace: 30,
                max_iter: 77,
                memory_budget_bytes: Some(1 << 30),
                compute_rdms: false,
            }
        );
    }
}
