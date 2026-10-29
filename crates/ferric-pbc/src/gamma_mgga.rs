//! Closed-shell Gamma-point meta-GGA nuclear forces and stress
//! (mgoldey/ferric#492), routed through the 1x1x1 k-point machinery.
//!
//! # Route
//!
//! The Gamma XC derivatives ([`crate::grad`], [`crate::stress`]) have no tau
//! term; the k-point ones ([`crate::kxcgrad`], [`crate::kxcstress`]) do. The
//! Gamma point IS the 1x1x1 Gamma-centred mesh: the Gamma density, Fock and
//! MOs are the `k = 0` ones (complex copies of the real matrices), and the
//! Gamma meta-GGA energy equals the 1x1x1 one (TPSS to 4.9e-15 Ha,
//! `tests/pbc_krks.rs`). So the Gamma meta-GGA force and stress are
//! [`crate::kxcgrad::kpoint_rks_gradient`] / [`crate::kxcstress::kpoint_rks_stress`]
//! on 1x1x1, fed the Gamma result, with plain (not range-separated) dense-AFT
//! kernels built here, exactly as [`crate::gamma_rsh`] routes the
//! range-separated hybrids.
//!
//! The k-point Fock guard rebuilds `h + J + V_xc` (meta-GGA: with `V^tau`)
//! from the Gamma density and compares it with the Gamma SCF's stored Fock
//! (bar [`crate::kgrad::FOCK_CONSISTENCY_BAR`]), so a Gamma SCF on another
//! grid, functional or J source (RS-GDF) is refused, not silently
//! differentiated. Because the derivative is the k-point code, the anchors
//! (`tests/pbc_gamma_mgga_derivs.rs`) are FD of the GAMMA SCF energy (the
//! independent construction: real lattice-summed AOs, the Gamma J and XC)
//! and, for a plain GGA, equality with the native Gamma force and stress.
//!
//! # Cost
//!
//! The 1x1x1 dense kernels are complex, `2 nao^4 x 16` bytes (4x the real
//! Gamma tensor, which stays resident in a binding), capped by
//! [`GammaMggaDerivConfig::kdense`]`.max_bytes`. A pure meta-GGA needs only
//! the J kernel, but [`KDenseAftEri::build`] always builds both J and K
//! (there is no J-only mode), so half of it is unused here; adding one is a
//! change to `kdense_aft`, left out.

use crate::dft::GammaRksConfig;
use crate::hcore::PeriodicHcoreConfig;
use crate::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use crate::kgrad::{KGradConfig, KGradJk};
use crate::kstress::{KStressConfig, KStressJk};
use crate::kxcgrad::{KRksGradConfig, KRksGradient, KXcGrid, KXcMutation};
use crate::kxcstress::{KRksStress, KRksStressConfig, KXcStressMutation};
use crate::lattice::Cell;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::result::ScfResult;

/// Settings of the routed Gamma meta-GGA derivatives.
#[derive(Clone, Copy, Default)]
pub struct GammaMggaDerivConfig {
    /// The 1x1x1 dense-AFT kernels (precision, hard cap, budget). Its
    /// `precision` must be the Gamma J's (the default is the default of
    /// [`crate::dense_aft::DenseAftEri::build`]'s callers); `rsh` must be
    /// `None` (range-separated hybrids route through [`crate::gamma_rsh`]).
    pub kdense: KDenseAftConfig,
    /// Force settings (budget, nucleus exponent).
    pub grad: KGradConfig,
    /// Stress settings (budget, nucleus exponent).
    pub stress: KStressConfig,
    /// TEST ONLY: a deliberate XC force defect.
    #[doc(hidden)]
    pub xc_grad_mutation: Option<KXcMutation>,
    /// TEST ONLY: a deliberate XC stress defect.
    #[doc(hidden)]
    pub xc_stress_mutation: Option<KXcStressMutation>,
}

impl GammaMggaDerivConfig {
    /// The config of a binding (Python, CLI): the 1x1x1 kernels at the
    /// default precision (the bindings' Gamma J precision), capped by the
    /// binding's dense-AFT cap `max_eri_bytes`; every derivative piece on the
    /// binding's memory budget.
    pub fn for_binding(max_eri_bytes: usize, budget_bytes: Option<usize>) -> Self {
        Self {
            kdense: KDenseAftConfig {
                max_bytes: max_eri_bytes,
                budget_bytes,
                ..Default::default()
            },
            grad: KGradConfig {
                budget_bytes,
                ..Default::default()
            },
            stress: KStressConfig {
                budget_bytes,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Closed-shell Gamma RKS nuclear gradient of a meta-GGA (SCAN, r2SCAN,
/// TPSS; any functional [`crate::kxcgrad::kpoint_rks_gradient`] accepts
/// except a range-separated hybrid), `scf` from [`crate::dft::gamma_rks`] on
/// the dense-AFT Gamma J, routed through the 1x1x1 k-point force (module doc).
///
/// * `hcore_cfg` — the hcore config of the energy (the 1x1x1 hcore is
///   rebuilt from it).
/// * `dft` — the SAME config the SCF ran with: functional, `grid` (the
///   atom-centred grid is rebuilt from it; a run on a caller-supplied grid is
///   not covered and is refused by the Fock guard), `xc.ao_threshold`,
///   `exxdiv`.
///
/// `grad` is `natoms x 3`, dE/dR in Hartree/Bohr per cell. Refuses an
/// open-shell result, a range-separated functional or kernel, and (Fock
/// guard) an SCF whose Fock the rebuilt one does not reproduce.
pub fn gamma_rks_mgga_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    scf: &ScfResult,
    dft: &GammaRksConfig,
    cfg: &GammaMggaDerivConfig,
) -> Result<KRksGradient, FerricError> {
    let who = "gamma_rks_mgga_gradient";
    let r = route(who, cell, prep, hcore_cfg, scf, dft, &cfg.kdense)?;
    let kcfg = KRksGradConfig {
        grad: cfg.grad,
        grid: KXcGrid::Atom(dft.grid.clone()),
        ao_threshold: dft.xc.ao_threshold,
        exxdiv: Some(dft.exxdiv),
        mutation: cfg.xc_grad_mutation,
        ..Default::default()
    };
    crate::kxcgrad::kpoint_rks_gradient(
        cell,
        prep,
        &r.mesh,
        hcore_cfg,
        &r.hk,
        KGradJk::Dense(&r.eri),
        &r.kscf,
        &dft.functional,
        &kcfg,
    )
}

/// Closed-shell Gamma RKS STRESS of a meta-GGA, routed through the 1x1x1
/// k-point stress exactly as [`gamma_rks_mgga_gradient`] routes the force:
/// same arguments, same refusals. `sigma` is `(1/Ω) dE/dε` (Hartree/Bohr³),
/// unsymmetrised (the atom-centred grid is not rotation invariant).
pub fn gamma_rks_mgga_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    scf: &ScfResult,
    dft: &GammaRksConfig,
    cfg: &GammaMggaDerivConfig,
) -> Result<KRksStress, FerricError> {
    let who = "gamma_rks_mgga_stress";
    let r = route(who, cell, prep, hcore_cfg, scf, dft, &cfg.kdense)?;
    let kcfg = KRksStressConfig {
        stress: cfg.stress,
        grid: KXcGrid::Atom(dft.grid.clone()),
        ao_threshold: dft.xc.ao_threshold,
        exxdiv: Some(dft.exxdiv),
        mutation: cfg.xc_stress_mutation,
        ..Default::default()
    };
    crate::kxcstress::kpoint_rks_stress(
        cell,
        prep,
        &r.mesh,
        hcore_cfg,
        &r.hk,
        KStressJk::Dense(&r.eri),
        &r.kscf,
        &dft.functional,
        &kcfg,
    )
}

/// The 1x1x1 inputs of the routed derivative.
struct Route {
    eri: KDenseAftEri,
    mesh: crate::kpts::KPointMesh,
    hk: crate::hcore::kpoint::PeriodicHcoreK,
    kscf: crate::kscf::KScfResult,
}

fn route(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    scf: &ScfResult,
    dft: &GammaRksConfig,
    kdense: &KDenseAftConfig,
) -> Result<Route, FerricError> {
    if kdense.rsh.is_some() || crate::rsh::is_range_separated(&dft.functional) {
        return Err(FerricError::General(format!(
            "{who}: {:?} / range-separated kernels: route range-separated hybrids through \
             gamma_rsh::gamma_rks_rsh_gradient / gamma_rks_rsh_stress",
            dft.functional
        )));
    }
    let (mesh, hk, kscf) = crate::gamma_rsh::one_point_inputs(
        who,
        "open-shell meta-GGA derivatives are not implemented",
        cell,
        prep,
        hcore_cfg,
        scf,
    )?;
    let eri = KDenseAftEri::build(cell, prep, &mesh, &hk.s, dft.exxdiv, kdense)?;
    Ok(Route {
        eri,
        mesh,
        hk,
        kscf,
    })
}

/// The refusal of the NATIVE Gamma force / stress entries ([`crate::grad`],
/// [`crate::stress`]) for a meta-GGA, pointing at this module for closed
/// shell (the native XC derivatives have no tau term) and at the 1x1x1
/// k-point UKS / ROKS (library) for open shell (the Gamma UKS / ROKS
/// meta-GGA is energy only).
pub(crate) fn refuse_native_mgga(who: &str, functional: &str) -> Result<(), FerricError> {
    if crate::dft::is_meta_gga_name(functional) {
        return Err(FerricError::General(format!(
            "{who}: {functional} is a meta-GGA; the Gamma XC derivatives here have no tau \
             term: closed shell, use gamma_mgga::gamma_rks_mgga_gradient / \
             gamma_rks_mgga_stress (the 1x1x1 k-point route, dense-AFT J); open shell (UKS / \
             ROKS), the Gamma meta-GGA is energy only and the k-point UKS on a 1x1x1 mesh \
             has the SCAN / r2SCAN force and stress (the k-point ROKS one too, library \
             only: kuxcgrad::kpoint_roks_gradient / kuxcstress::kpoint_roks_stress)"
        )));
    }
    Ok(())
}
