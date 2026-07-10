//! Ferric GW spike: G0W0, COHSEX, evGW₀, evGW, sc-COHSEX.
//!
//! Built on top of `ferric-rpa`'s PDEP eigenpotential basis as a low-rank
//! representation of the screened Coulomb interaction W(iω). See
//! `docs/superpowers/specs/2026-05-19-rpa-to-gw-spike-design.md`.
//!
//! Spike scope: closed-shell, HF reference, cc-pVDZ-scale, validation on H₂O.

// Self-energy kernels intrinsically thread many quantities (G, W, orbital
// energies, frequency grid, QP indices); bundling them into structs would only
// move the boilerplate to the call sites. type_complexity likewise flags the
// multi-array return tuples, which are self-documenting and used once.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

pub mod method;
pub mod mo_b;
pub mod w_pdep;
pub mod pade;
pub mod sigma;
pub mod cohsex;
pub mod bse;
pub mod qp;
pub mod u_sigma;
pub mod u_cohsex;
pub mod vxc_mo;

pub use method::GwMethod;

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::PdepRpaConfig;
use ferric_rpa::{run_pdep_rpa, PdepRpaResult};
use ferric_scf::ScfResult;
use ndarray::Array1;

/// Clone a PDEP-RPA config with `need_inv_dielectric_freq` forced on. Every GW
/// method's Σ_c reads `PdepRpaResult.inv_dielectric_freq`, so the GW crate sets
/// the flag itself rather than trusting the external caller to have set it
/// (M9 memory gate: energy-only RPA runs default it off).
fn with_inv_dielectric(cfg: &PdepRpaConfig) -> PdepRpaConfig {
    let mut c = cfg.clone();
    c.need_inv_dielectric_freq = true;
    c
}

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
    /// Optional resident-bytes ceiling propagated into the underlying
    /// `PdepRpaConfig`/`RiMp2Config` transforms. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`].
    pub memory_budget_bytes: Option<usize>,
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
            memory_budget_bytes: None,
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
    /// Per-state QP Newton-solve convergence flag, aligned with `mo_indices`.
    /// `false` means the Newton iteration exhausted its step budget (or hit a
    /// near-pole `|f'|` bailout) without reaching its step-size tolerance —
    /// the corresponding `eps_qp`/`sigma_c`/`z_factor` entries are the last
    /// iterate, not a converged root. Always `true` for COHSEX (closed-form,
    /// no Newton solve). See `sigma::solve_qp_for_mo`.
    pub qp_converged: Vec<bool>,
    /// Iteration count for ev-loops (0 for non-iterative methods).
    pub n_ev_iter: usize,
    /// Whether the evGW/evGW₀ outer (eigenvalue self-consistency) loop met
    /// `ev_conv_thresh` within `max_ev_iter`. Always `true` for non-iterative
    /// methods (G0W0, COHSEX).
    pub outer_converged: bool,
    /// Underlying PDEP result (so callers can inspect W).
    pub pdep: PdepRpaResult,
}

impl GwResult {
    /// **Deprecated for G0W0 — pass `vxc_diag` to `run_gw` instead.** This applies
    /// Σ_x − v_xc to the QP energies *after* the QP solve, which is WRONG for a KS
    /// reference: Σ_c is energy-dependent and the shift (~−7 eV) moves the QP root
    /// by several eV, so Σ_c must be evaluated at the shifted energy. `run_gw` with
    /// `Some(vxc_diag)` now folds the shift into the QP self-consistency correctly.
    /// Retained only for the linearized/diagnostic case where the shift is small.
    pub fn apply_kohn_sham_correction(&mut self, vxc_diag: &Array1<f64>) {
        for (idx, &mo_abs) in self.mo_indices.iter().enumerate() {
            self.eps_qp[idx] += self.sigma_x[idx] - vxc_diag[mo_abs];
        }
    }
}

/// Spin-unrestricted GW result. Per-spin QP energies on a shared MO-index list.
///
/// The MO indices are *absolute* and shared between channels (so MO `i` here
/// refers to the i-th α MO and i-th β MO, which for UHF/UKS are different
/// orbitals — caller should be aware when interpreting). For HOMO-α vs
/// HOMO-β identification, use `eps_qp_a[idx]` vs `eps_qp_b[idx]` separately.
#[derive(Debug)]
pub struct UGwResult {
    pub mo_indices: Vec<usize>,
    pub eps_mf_a: Array1<f64>,
    pub eps_qp_a: Array1<f64>,
    pub sigma_x_a: Array1<f64>,
    pub sigma_c_a: Array1<f64>,
    pub z_factor_a: Array1<f64>,
    pub eps_mf_b: Array1<f64>,
    pub eps_qp_b: Array1<f64>,
    pub sigma_x_b: Array1<f64>,
    pub sigma_c_b: Array1<f64>,
    pub z_factor_b: Array1<f64>,
    /// Per-state QP Newton-solve convergence flags, aligned with `mo_indices`
    /// (α/β channels separately — see `GwResult::qp_converged` for the
    /// per-flag meaning). Always all-`true` for COHSEX.
    pub qp_converged_a: Vec<bool>,
    pub qp_converged_b: Vec<bool>,
    pub n_ev_iter: usize,
    /// Whether the U-evGW/U-evGW₀ outer loop met `ev_conv_thresh` within
    /// `max_ev_iter`. Always `true` for non-iterative methods.
    pub outer_converged: bool,
    pub pdep: PdepRpaResult,
}

impl UGwResult {
    /// Apply Σ_x − v_xc correction in place. Required when the reference is
    /// KS (UKS) rather than HF (UHF/ROHF). For each MO p, shift:
    ///   ε_qp_σ_p ← ε_qp_σ_p + (Σ_x_σ_p − v_xc_σ_p)
    /// where v_xc_σ_p are the diagonal v_xc matrix elements in MO basis.
    /// `vxc_diag_a/b` are absolute-MO-indexed (length nmo); only entries for
    /// `mo_indices` are read.
    pub fn apply_kohn_sham_correction(
        &mut self,
        vxc_diag_a: &Array1<f64>,
        vxc_diag_b: &Array1<f64>,
    ) {
        for (idx, &mo_abs) in self.mo_indices.iter().enumerate() {
            let d_a = self.sigma_x_a[idx] - vxc_diag_a[mo_abs];
            let d_b = self.sigma_x_b[idx] - vxc_diag_b[mo_abs];
            self.eps_qp_a[idx] += d_a;
            self.eps_qp_b[idx] += d_b;
        }
    }
}

/// Top-level dispatch — spin-unrestricted. Accepts UHF, ROHF, or UKS reference.
///
/// For UKS, the caller must apply the Σ_x − v_xc correction via
/// `UGwResult::apply_kohn_sham_correction` using `vxc_mo::vxc_diagonal_mo`
/// (we don't auto-apply since we don't carry the xc_name through).
pub fn run_u_gw(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    pdep_cfg: &PdepRpaConfig,
    gw_cfg: &GwConfig,
) -> Result<UGwResult, FerricError> {
    use ferric_rpa::run_u_pdep_rpa;
    if matches!(scf.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General(
            "run_u_gw: closed-shell ScfResult — use run_gw instead".into(),
        ));
    }

    // GW Σ_c requires the per-frequency inverse-dielectric stack; force it on
    // regardless of what the external caller left in `pdep_cfg` (M9 gate).
    let pdep_cfg = &with_inv_dielectric(pdep_cfg);
    let pdep = run_u_pdep_rpa(mol, obs, dfbs, op, scf, pdep_cfg)?;
    let (mo_b_a, mo_b_b) = mo_b::build_full_b_both_spins(mol, obs, dfbs, op, scf, gw_cfg.frozen_core)?;
    let (v_dressed, dress_dev) =
        w_pdep::redress_with_check(&mo_b_a.v_inv_sqrt, &pdep.eigenpotentials)?;
    eprintln!(
        "ferric-gw [U]: redressed eigenpotentials, max |‖V_α‖² − 1| = {dress_dev:.3e}"
    );

    let qp_range = gw_cfg.qp_mos.clone().unwrap_or_else(|| default_u_qp_range(mol, scf));

    let result = match gw_cfg.method {
        GwMethod::G0W0 => u_sigma::run_u_g0w0(&mo_b_a, &mo_b_b, pdep, qp_range, gw_cfg, &v_dressed),
        GwMethod::Cohsex => u_cohsex::run_u_cohsex(&mo_b_a, &mo_b_b, pdep, qp_range, gw_cfg, &v_dressed),
        GwMethod::EvGw0 => u_sigma::run_u_evgw0(&mo_b_a, &mo_b_b, pdep, qp_range, gw_cfg, &v_dressed),
        GwMethod::EvGw => u_sigma::run_u_evgw(
            mol, obs, dfbs, op, scf, pdep_cfg, &mo_b_a, &mo_b_b, pdep, qp_range, gw_cfg,
        ),
        GwMethod::ScCohsex => Err(FerricError::General(
            "U-sc-COHSEX not implemented; see plan P2.".into(),
        )),
    }?;
    if !result.outer_converged {
        eprintln!(
            "warning: U-{:?} eigenvalue self-consistency did NOT converge in {} \
             iterations (thresh {:.1e}); QP energies are the last sweep",
            gw_cfg.method, result.n_ev_iter, gw_cfg.ev_conv_thresh
        );
    }
    for (spin, flags) in [("alpha", &result.qp_converged_a), ("beta", &result.qp_converged_b)] {
        let bad: Vec<usize> = flags
            .iter()
            .enumerate()
            .filter(|(_, &c)| !c)
            .map(|(i, _)| result.mo_indices[i])
            .collect();
        if !bad.is_empty() {
            eprintln!(
                "warning: QP Newton solve did not converge for {spin} MO(s) {bad:?}; \
                 those QP energies are best-effort"
            );
        }
    }
    Ok(result)
}

fn default_u_qp_range(mol: &Molecule, scf: &ScfResult) -> std::ops::Range<usize> {
    let nelec = mol.nelec();
    let two_s = (mol.multiplicity as i32) - 1;
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;
    let nocc_max = nocc_a.max(nocc_b);
    let nmo = scf.eps_alpha.len();
    let lo = nocc_max.saturating_sub(3);
    let hi = (nocc_max + 3).min(nmo);
    lo..hi
}

/// Top-level dispatch.
/// `vxc_diag`: absolute-MO-indexed diagonal v_xc for a KS (RKS) reference. When
/// given, the QP equation includes Σ_x − v_xc *inside* the self-consistency
/// (correct for a KS reference; Σ_c is then evaluated at the shifted QP root).
/// `None` ⇒ HF reference (no shift). Use `vxc_mo::vxc_diagonal_mo` to build it.
/// Currently wired for `GwMethod::G0W0`; passing `Some(..)` with another method
/// is an error (evGW@KS not yet implemented).
pub fn run_gw(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    pdep_cfg: &PdepRpaConfig,
    gw_cfg: &GwConfig,
    vxc_diag: Option<&ndarray::Array1<f64>>,
) -> Result<GwResult, FerricError> {
    if vxc_diag.is_some() && !matches!(gw_cfg.method, GwMethod::G0W0) {
        return Err(FerricError::General(
            "run_gw: KS reference (vxc_diag) is only wired for G0W0; evGW@KS is not \
             yet implemented".into(),
        ));
    }
    if !matches!(rhf.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General(
            "ferric-gw: spike supports closed-shell (RHF) only".into(),
        ));
    }
    // 1. Run PDEP-RPA to get {λ_α(iω_k), V_α^dressed, B̃^P_ia}.
    //    GW Σ_c needs the inverse-dielectric stack — force the flag (M9 gate).
    let pdep_cfg = &with_inv_dielectric(pdep_cfg);
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
    let result = match gw_cfg.method {
        GwMethod::Cohsex => cohsex::run_cohsex(mol, rhf, &mo_b, &v_dressed, pdep, qp_range, gw_cfg),
        GwMethod::G0W0 => sigma::run_g0w0(mol, rhf, &mo_b, &v_dressed, pdep, qp_range, gw_cfg, vxc_diag),
        GwMethod::EvGw0 => sigma::run_evgw0(
            mol, rhf, &mo_b, &v_dressed, pdep, qp_range, gw_cfg,
        ),
        GwMethod::EvGw => sigma::run_evgw(
            mol, obs, dfbs, op, rhf, pdep_cfg, &mo_b, pdep, qp_range, gw_cfg,
        ),
        GwMethod::ScCohsex => Err(FerricError::General(
            "sc-COHSEX not implemented in spike P0; see plan P2.".into(),
        )),
    }?;
    warn_gw_unconverged(&result, gw_cfg);
    Ok(result)
}

/// Surface non-convergence (reliability audit: flags used to be silently
/// dropped — a stalled evGW or a Newton solve stuck at a Σc pole printed the
/// same table as a converged run).
fn warn_gw_unconverged(result: &GwResult, gw_cfg: &GwConfig) {
    if !result.outer_converged {
        eprintln!(
            "warning: {:?} eigenvalue self-consistency did NOT converge in {} \
             iterations (thresh {:.1e}); QP energies are the last sweep",
            gw_cfg.method, result.n_ev_iter, gw_cfg.ev_conv_thresh
        );
    }
    let bad: Vec<usize> = result
        .qp_converged
        .iter()
        .enumerate()
        .filter(|(_, &c)| !c)
        .map(|(i, _)| result.mo_indices[i])
        .collect();
    if !bad.is_empty() {
        eprintln!(
            "warning: QP Newton solve did not converge for MO(s) {bad:?} \
             (root near a Σc pole?); those QP energies are best-effort"
        );
    }
}

fn default_qp_range(mol: &Molecule, rhf: &ScfResult) -> std::ops::Range<usize> {
    let nocc = (mol.nelec() as usize) / 2;
    let nmo = rhf.eps_r().len();
    let lo = nocc.saturating_sub(3);
    let hi = (nocc + 3).min(nmo);
    lo..hi
}
