//! Ferric GW spike: G0W0, COHSEX, evGW₀, evGW, sc-COHSEX.
//!
//! Built on top of `ferric-rpa`'s PDEP eigenpotential basis as a low-rank
//! representation of the screened Coulomb interaction W(iω). See
//! `docs/superpowers/specs/2026-05-19-rpa-to-gw-spike-design.md`.
//!
//! Spike scope: closed-shell, HF reference, cc-pVDZ-scale, validation on H₂O.

pub mod method;
pub mod mo_b;
pub mod w_pdep;
pub mod pade;
pub mod sigma;
pub mod cohsex;
pub mod qp;

pub use method::GwMethod;

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::PdepRpaConfig;
use ferric_rpa::{run_pdep_rpa, PdepRpaResult};
use ferric_scf::ScfResult;
use ndarray::Array1;

/// Top-level GW configuration.
#[derive(Debug, Clone)]
pub struct GwConfig {
    pub method: GwMethod,
    /// Range of MOs (absolute indices into the full MO set) for which to
    /// compute quasiparticle energies. Defaults to {HOMO-2..LUMO+2}.
    pub qp_mos: Option<std::ops::Range<usize>>,
    /// Max evGW outer iterations (ignored for G0W0/COHSEX).
    pub max_ev_iter: usize,
    /// evGW convergence threshold on |Δε^QP|_max (Ha).
    pub ev_conv_thresh: f64,
    /// Number of Padé continued-fraction coefficients (must equal N_quad).
    /// 0 → use N_quad from PdepRpaConfig.
    pub pade_npts: usize,
    /// Newton-step damping for QP solver.
    pub qp_newton_damp: f64,
    /// Frozen core for both PDEP and GW (must match for self-consistency).
    pub frozen_core: usize,
}

impl Default for GwConfig {
    fn default() -> Self {
        Self {
            method: GwMethod::G0W0,
            qp_mos: None,
            max_ev_iter: 20,
            ev_conv_thresh: 1e-4,
            pade_npts: 0,
            qp_newton_damp: 1.0,
            frozen_core: 0,
        }
    }
}

/// Per-MO quasiparticle result.
#[derive(Debug)]
pub struct GwResult {
    /// MO indices for which QP energies were computed (absolute).
    pub mo_indices: Vec<usize>,
    /// Mean-field (input) orbital energies for those MOs.
    pub eps_mf: Array1<f64>,
    /// QP energies (final), Ha.
    pub eps_qp: Array1<f64>,
    /// Exchange self-energy Σ_x (diagonal MO), Ha.
    pub sigma_x: Array1<f64>,
    /// Correlation self-energy Σ_c evaluated at the converged QP energy, Ha.
    pub sigma_c: Array1<f64>,
    /// Z-factor (renormalization), dimensionless.
    pub z_factor: Array1<f64>,
    /// Iteration count for ev-loops (0 for non-iterative methods).
    pub n_ev_iter: usize,
    /// Underlying PDEP result (so callers can inspect W).
    pub pdep: PdepRpaResult,
}

/// Top-level dispatch.
pub fn run_gw(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    pdep_cfg: &PdepRpaConfig,
    gw_cfg: &GwConfig,
) -> Result<GwResult, FerricError> {
    if !matches!(rhf.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General(
            "ferric-gw: spike supports closed-shell (RHF) only".into(),
        ));
    }
    // 1. Run PDEP-RPA to get {λ_α(iω_k), V_α^dressed, B̃^P_ia}.
    let pdep = run_pdep_rpa(mol, obs, dfbs, op, rhf, pdep_cfg)?;

    // 2. Build dressed B̃ tensor over ALL (m,n) MO pairs needed for Σ.
    let mo_b = mo_b::build_full_b(mol, obs, dfbs, op, rhf, gw_cfg.frozen_core)?;

    // 3. Re-dress the eigenpotentials from physical → V^{-1/2}-dressed.
    //    eigenpotentials_phys = V^{-1/2} · V_dressed, so
    //    V_dressed = inv(V^{-1/2}) · eigenpotentials_phys.
    let (v_dressed, dress_dev) =
        w_pdep::redress_with_check(&mo_b.v_inv_sqrt, &pdep.eigenpotentials)?;
    eprintln!(
        "ferric-gw: redressed eigenpotentials, max |‖V_α‖² − 1| = {dress_dev:.3e}"
    );

    // 4. Decide which MOs to compute Σ for.
    let qp_range = gw_cfg.qp_mos.clone().unwrap_or_else(|| default_qp_range(mol, rhf));

    // 5. Dispatch by method.
    match gw_cfg.method {
        GwMethod::Cohsex => cohsex::run_cohsex(mol, rhf, &mo_b, &v_dressed, pdep, qp_range, gw_cfg),
        GwMethod::G0W0 => sigma::run_g0w0(mol, rhf, &mo_b, &v_dressed, pdep, qp_range, gw_cfg),
        GwMethod::EvGw0 => sigma::run_evgw0(
            mol, rhf, &mo_b, &v_dressed, pdep, qp_range, gw_cfg,
        ),
        GwMethod::EvGw => sigma::run_evgw(
            mol, obs, dfbs, op, rhf, pdep_cfg, &mo_b, pdep, qp_range, gw_cfg,
        ),
        GwMethod::ScCohsex => Err(FerricError::General(
            "sc-COHSEX not implemented in spike P0; see plan P2.".into(),
        )),
    }
}

fn default_qp_range(mol: &Molecule, rhf: &ScfResult) -> std::ops::Range<usize> {
    let nocc = (mol.nelec() as usize) / 2;
    let nmo = rhf.eps_r().len();
    let lo = nocc.saturating_sub(3);
    let hi = (nocc + 3).min(nmo);
    lo..hi
}
