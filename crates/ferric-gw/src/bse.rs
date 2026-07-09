//! BSE (Bethe–Salpeter) singlet excitation energies, Tamm–Dancoff approximation.
//!
//! Solves the BSE-TDA eigenproblem `A·X = Ω·X` on the GW reference, where the
//! singlet TDA matrix in the particle–hole `(ia)` space is
//!
//! ```text
//!   A_{ia,jb} = (ε_a^QP − ε_i^QP) δ_ij δ_ab  +  2(ia|jb)_v  −  (ab|W|ij)
//! ```
//!
//! Conventions (verified against the COHSEX self-energy, `cohsex.rs:88`):
//!   * the **Hartree/coupling** term `2(ia|jb)` uses the BARE Coulomb v;
//!   * the **screened-exchange** term uses the FULL static screened interaction
//!     `(ab|W|ij) = Σ_α (1/λ_α(0)) M[α,a,b] M[α,i,j]`, i.e. weight `1/λ_α(0)`
//!     (the SEX weight `w_α + 1`), NOT the reduced `w_α = 1/λ_α − 1` used by the
//!     COH/Σ_c terms. This is the static-screening (Tamm–Dancoff) BSE kernel.
//!   * `M[α,p,q] = Σ_P (Ṽ_α)_P b̃^P_{pq}` is `project_b_into_pdep`, identical to
//!     the GW self-energy projection — reuses the validated dressed-basis path.
//!
//! The GW quasiparticle energies are taken from `run_gw` (G0W0@HF for the first
//! gate). Validation target: lowest singlet of H₂O / cc-pVDZ vs MOLGW BSE@G0W0.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::oneelectron;
use ferric_rpa::PdepRpaConfig;
use ferric_scf::ScfResult;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, Solve, UPLO};

use crate::cohsex::project_b_into_pdep;
use crate::method::GwMethod;
use crate::{mo_b, run_gw, w_pdep, GwConfig};

/// Result of a BSE-TDA singlet excitation calculation.
#[derive(Debug, Clone)]
pub struct BseResult {
    /// Singlet excitation energies Ω_n (Hartree), ascending.
    pub omega: Vec<f64>,
    /// Number of occupied / virtual orbitals in the BSE window (frozen-core aware).
    pub nocc: usize,
    pub nvir: usize,
    /// GW quasiparticle energies used for the diagonal (active block, Ha).
    pub eps_qp: Vec<f64>,
}

impl BseResult {
    /// Lowest singlet excitation energy in eV.
    pub fn lowest_ev(&self) -> f64 {
        self.omega[0] * 27.211_386_245_988
    }
}

/// Run a BSE-TDA singlet calculation on a closed-shell RHF reference.
///
/// Computes G0W0@HF quasiparticle energies for ALL MOs (so every particle–hole
/// pair has a real QP energy), builds the static-screened TDA matrix, and
/// returns its eigenvalues. `frozen_core` freezes the lowest `frozen_core`
/// occupied orbitals out of the (ia) space (and out of GW).
pub fn run_bse_tda(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    pdep_cfg: &PdepRpaConfig,
    frozen_core: usize,
) -> Result<BseResult, FerricError> {
    if !matches!(rhf.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General(
            "run_bse_tda: closed-shell (RHF) only".into(),
        ));
    }
    let nmo = rhf.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;

    // 1. GW@HF for ALL MOs so every (i,a) in the window has a QP energy.
    let gw_cfg = GwConfig {
        method: GwMethod::G0W0,
        qp_mos: Some(0..nmo),
        max_ev_iter: 0,
        ev_conv_thresh: 1e-4,
        pade_npts: 0,
        qp_newton_damp: 1.0,
        frozen_core,
        memory_budget_bytes: pdep_cfg.memory_budget_bytes,
    };
    let gw = run_gw(mol, obs, dfbs, op, rhf, pdep_cfg, &gw_cfg, None)?;

    // Map absolute MO index → QP energy.
    let mut eps_qp_full = rhf.eps_r().to_vec(); // fallback = HF (should be fully overwritten)
    for (k, &mo) in gw.mo_indices.iter().enumerate() {
        eps_qp_full[mo] = gw.eps_qp[k];
    }

    // 2. Active particle–hole window (frozen-core aware).
    let first_act = frozen_core;
    let nocc = nocc_total - frozen_core;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;
    if n == 0 {
        return Err(FerricError::General("run_bse_tda: empty (ia) space".into()));
    }

    // 3. Dressed B̃^P_{pq} over all MO pairs + projection M[α,p,q] = Σ_P Ṽ_α b̃^P_pq.
    let mob = mo_b::build_full_b(mol, obs, dfbs, op, rhf, frozen_core)?;
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mob.v_inv_sqrt, &gw.pdep.eigenpotentials)?;
    let m_proj = project_b_into_pdep(&mob, &v_dressed); // (M, n_act, n_act), local MO indices
    let m_modes = m_proj.shape()[0];

    // Reduced screening weight w_α = 1/λ_α(0) − 1; screened W = bare v + Σ w_α MM
    // (robust to an incomplete PDEP mode set — see run_bse_c6_ks for the rationale
    // and the basis-check diagnostic).
    let w_red: Vec<f64> = gw
        .pdep
        .eigenvalues_static
        .iter()
        .map(|&l| 1.0 / l - 1.0)
        .collect();
    assert_eq!(w_red.len(), m_modes);
    {
        let lmax = gw
            .pdep
            .eigenvalues_static
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
        let homo = eps_qp_full[nocc_total - 1];
        let lumo = eps_qp_full[nocc_total];
        eprintln!(
            "ferric BSE diag: λ_max(0)={lmax:.4}  GW gap={:.4} Ha ({:.3} eV)  IP={:.3} eV",
            lumo - homo,
            (lumo - homo) * 27.211_386_245_988,
            -homo * 27.211_386_245_988
        );
    }

    // Bare Coulomb (pq|rs) = Σ_P b̃^P_pq b̃^P_rs from the dressed RI tensor.
    // mob.b_full indices are LOCAL (relative to first_act). Occupied-active local
    // indices: 0..nocc ; virtual local indices: nocc..n_act.
    let b = &mob.b_full; // (naux, n_act, n_act)
    let naux = mob.naux;
    let bare = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for pp in 0..naux {
            acc += b[(pp, p, q)] * b[(pp, r, s)];
        }
        acc
    };
    let screened = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = bare(p, q, r, s); // bare v (complete RI basis)
        for alpha in 0..m_modes {
            acc += w_red[alpha] * m_proj[(alpha, p, q)] * m_proj[(alpha, r, s)];
        }
        acc
    };

    // 4. Assemble the singlet BSE-TDA matrix A (n × n), Hermitian.
    //    local occ i: 0..nocc   local vir a: nocc + a_loc, a_loc in 0..nvir
    //    A_{ia,jb} = (ε_a − ε_i) δ + 2(ia|jb)_v − (ab|W|ij)
    let mut a_mat = Array2::<f64>::zeros((n, n));
    for i in 0..nocc {
        let i_loc = i; // occupied-active local index
        let eps_i = eps_qp_full[first_act + i];
        for a in 0..nvir {
            let a_loc = nocc + a; // virtual local index in the active block
            let eps_a = eps_qp_full[nocc_total + a];
            let ia = i * nvir + a;
            for j in 0..nocc {
                let j_loc = j;
                for bb in 0..nvir {
                    let b_loc = nocc + bb;
                    let jb = j * nvir + bb;
                    let coul = bare(i_loc, a_loc, j_loc, b_loc); // (ia|jb)
                    let scr = screened(a_loc, b_loc, i_loc, j_loc); // (ab|W|ij)
                    a_mat[(ia, jb)] = 2.0 * coul - scr;
                }
            }
            a_mat[(ia, ia)] += eps_a - eps_i;
        }
    }

    // 5. Eigenvalues (Hermitian).
    let (evals, _) = a_mat
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("BSE-TDA eigh: {e}")))?;
    let mut omega: Vec<f64> = evals.to_vec();
    omega.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let eps_qp_act: Vec<f64> = (first_act..nmo).map(|p| eps_qp_full[p]).collect();
    Ok(BseResult {
        omega,
        nocc,
        nvir,
        eps_qp: eps_qp_act,
    })
}

/// CIS / TDHF-TDA cross-check: the SAME singlet-TDA assembly as `run_bse_tda`
/// but with the BARE Coulomb exchange (no screening, W → v) and HF orbital
/// energies (no GW). This isolates the kernel ASSEMBLY from the GW input:
///
/// ```text
///   A_{ia,jb} = (ε_a − ε_i)^{HF} δ + 2(ia|jb)_v − (ab|ij)_v
/// ```
///
/// If this reproduces an independent CIS-TDA (e.g. PySCF), the (ia)-space layout,
/// the 2v−exchange convention, and the integral contraction are all correct, and
/// any `run_bse_tda` discrepancy is attributable to the screening / GW gap, not
/// the assembly. No GW, no PDEP — pure HF integrals.
pub fn run_cis_tda(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
) -> Result<BseResult, FerricError> {
    if !matches!(rhf.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General("run_cis_tda: closed-shell (RHF) only".into()));
    }
    let nmo = rhf.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let eps = rhf.eps_r().to_vec();

    let first_act = frozen_core;
    let nocc = nocc_total - frozen_core;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;

    let mob = mo_b::build_full_b(mol, obs, dfbs, op, rhf, frozen_core)?;
    let b = &mob.b_full;
    let naux = mob.naux;
    let bare = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for pp in 0..naux {
            acc += b[(pp, p, q)] * b[(pp, r, s)];
        }
        acc
    };

    let mut a_mat = Array2::<f64>::zeros((n, n));
    for i in 0..nocc {
        let eps_i = eps[first_act + i];
        for a in 0..nvir {
            let a_loc = nocc + a;
            let eps_a = eps[nocc_total + a];
            let ia = i * nvir + a;
            for j in 0..nocc {
                for bb in 0..nvir {
                    let b_loc = nocc + bb;
                    let jb = j * nvir + bb;
                    let coul = bare(i, a_loc, j, b_loc); // (ia|jb)
                    let exch = bare(a_loc, b_loc, i, j); // (ab|ij)  bare
                    a_mat[(ia, jb)] = 2.0 * coul - exch;
                }
            }
            a_mat[(ia, ia)] += eps_a - eps_i;
        }
    }
    let (evals, _) = a_mat
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("CIS-TDA eigh: {e}")))?;
    let mut omega: Vec<f64> = evals.to_vec();
    omega.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let eps_act: Vec<f64> = (first_act..nmo).map(|p| eps[p]).collect();
    Ok(BseResult { omega, nocc, nvir, eps_qp: eps_act })
}

/// Result of a BSE dynamic-polarizability / C6 calculation (gate 2).
#[derive(Debug, Clone)]
pub struct BseC6Result {
    /// Molecular isotropic C6 (a.u.).
    pub c6: f64,
    /// Isotropic α(iω_k) at each Casimir–Polder grid point.
    pub alpha_iso: Vec<f64>,
    /// Static isotropic α (= α(iω=0)).
    pub alpha_static: f64,
    pub nocc: usize,
    pub nvir: usize,
}

/// BSE dynamic polarizability α(iω) and molecular C6 (gate 2 of the design ladder).
///
/// Uses the W-screened FULL (A±B) (not TDA) on the G0W0@HF reference and the
/// imaginary-frequency response identical in form to `dynamic_cphf_alpha_iw`:
///
/// ```text
///   (A+B)_W = Δε^GW δ + 4(ia|jb)_v − (ab|W|ij) − (ib|W|aj)
///   (A−B)_W = Δε^GW δ            + (ib|W|aj) − (ab|W|ij)
///   α(iω)   = 4 μᵀ (A−B)_W [ (A−B)_W (A+B)_W + ω² ]⁻¹ μ
///   C6      = (3/π) Σ_k w_k α_iso(iω_k)²        (Casimir–Polder)
/// ```
///
/// C6 is far less sensitive to the absolute GW gap than excitation energies (it
/// weights the whole α(iω) tail), so this is the meaningful dispersion target
/// even while the GW EA side is being tightened. `freqs`/`weights` are the
/// Casimir–Polder imaginary-frequency grid.
#[allow(clippy::too_many_arguments)]
pub fn run_bse_c6(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    pdep_cfg: &PdepRpaConfig,
    frozen_core: usize,
    freqs: &[f64],
    weights: &[f64],
) -> Result<BseC6Result, FerricError> {
    use std::f64::consts::PI;
    if !matches!(rhf.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General("run_bse_c6: closed-shell (RHF) only".into()));
    }
    let nmo = rhf.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let c = rhf.mos_r();

    // GW@HF for all MOs.
    let gw_cfg = GwConfig {
        method: GwMethod::G0W0,
        qp_mos: Some(0..nmo),
        max_ev_iter: 0,
        ev_conv_thresh: 1e-4,
        pade_npts: 0,
        qp_newton_damp: 1.0,
        frozen_core,
        memory_budget_bytes: pdep_cfg.memory_budget_bytes,
    };
    let gw = run_gw(mol, obs, dfbs, op, rhf, pdep_cfg, &gw_cfg, None)?;
    let mut eps_qp = rhf.eps_r().to_vec();
    for (k, &mo) in gw.mo_indices.iter().enumerate() {
        eps_qp[mo] = gw.eps_qp[k];
    }

    let first_act = frozen_core;
    let nocc = nocc_total - frozen_core;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;

    // Projected screened modes + bare integrals (same path as run_bse_tda).
    let mob = mo_b::build_full_b(mol, obs, dfbs, op, rhf, frozen_core)?;
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mob.v_inv_sqrt, &gw.pdep.eigenpotentials)?;
    let m_proj = project_b_into_pdep(&mob, &v_dressed);
    let m_modes = m_proj.shape()[0];
    // Reduced screening weight; screened W = bare v + Σ w_α MM (mode-set-robust,
    // see run_bse_c6_ks).
    let w_red: Vec<f64> = gw.pdep.eigenvalues_static.iter().map(|&l| 1.0 / l - 1.0).collect();
    let b = &mob.b_full;
    let naux = mob.naux;
    let bare = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for pp in 0..naux {
            acc += b[(pp, p, q)] * b[(pp, r, s)];
        }
        acc
    };
    let screened = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = bare(p, q, r, s); // bare v (complete RI basis)
        for alpha in 0..m_modes {
            acc += w_red[alpha] * m_proj[(alpha, p, q)] * m_proj[(alpha, r, s)];
        }
        acc
    };

    // Full screened (A±B) in (ia)-space. local occ i:0..nocc ; vir a: nocc+a.
    // (A+B) = Δε δ + 4(ia|jb)_v − (ab|W|ij) − (ib|W|aj)
    // (A−B) = Δε δ            + (ib|W|aj) − (ab|W|ij)
    let mut apb = Array2::<f64>::zeros((n, n));
    let mut amb = Array2::<f64>::zeros((n, n));
    for i in 0..nocc {
        let eps_i = eps_qp[first_act + i];
        for a in 0..nvir {
            let a_loc = nocc + a;
            let eps_a = eps_qp[nocc_total + a];
            let ia = i * nvir + a;
            for j in 0..nocc {
                for bb in 0..nvir {
                    let b_loc = nocc + bb;
                    let jb = j * nvir + bb;
                    let coul = bare(i, a_loc, j, b_loc); // (ia|jb)
                    let w_abij = screened(a_loc, b_loc, i, j); // (ab|W|ij)
                    let w_ibaj = screened(i, b_loc, a_loc, j); // (ib|W|aj)
                    apb[(ia, jb)] = 4.0 * coul - w_abij - w_ibaj;
                    amb[(ia, jb)] = w_ibaj - w_abij;
                }
            }
            apb[(ia, ia)] += eps_a - eps_i;
            amb[(ia, ia)] += eps_a - eps_i;
        }
    }

    // Dipole μ in (ia)-space (bare operator), per Cartesian axis.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let r_mo: [Array2<f64>; 3] = std::array::from_fn(|d| c.t().dot(&dip_ao[d]).dot(c));
    let mut mu: [Array1<f64>; 3] = std::array::from_fn(|_| Array1::zeros(n));
    for (d, m) in mu.iter_mut().enumerate() {
        for i in 0..nocc {
            for a in 0..nvir {
                m[i * nvir + a] = r_mo[d][(first_act + i, nocc_total + a)];
            }
        }
    }

    // α(iω) = 4 μᵀ (A−B)[(A−B)(A+B)+ω²]⁻¹ μ, isotropic average over axes.
    let mut alpha_iso = Vec::with_capacity(freqs.len());
    for &w in freqs {
        let mut sysm = amb.dot(&apb);
        for k in 0..n {
            sysm[(k, k)] += w * w;
        }
        let mut iso = 0.0;
        for axis in mu.iter() {
            let rhs = amb.dot(axis);
            let t = sysm
                .solve(&rhs)
                .map_err(|e| FerricError::Lapack(format!("BSE α(iω) solve: {e}")))?;
            iso += 4.0 * axis.dot(&t);
        }
        alpha_iso.push(iso / 3.0);
    }

    let c6 = 3.0 / PI * (0..freqs.len()).map(|k| weights[k] * alpha_iso[k] * alpha_iso[k]).sum::<f64>();
    let alpha_static = *alpha_iso.first().unwrap_or(&0.0);
    Ok(BseC6Result { c6, alpha_iso, alpha_static, nocc, nvir })
}

/// RPAx@KS spike: BSE-form screened-(A±B) α(iω)/C6 on a Kohn–Sham (e.g. PBE)
/// reference, using KS orbital energies DIRECTLY on the diagonal (no GW) and the
/// PDEP screening modes built from the SAME KS response. This isolates the
/// REFERENCE variable from gate 2: gate 2 was HF energies + HF-W (water C6 −64%);
/// if a PBE reference + PBE-W fixes the α deficit (α_static → ~DOSD), the
/// W-as-kernel dispersion lane is alive; if not, the screened kernel itself
/// under-polarizes and the lane is dead.
///
/// Same kernel as `run_bse_c6`:
/// ```text
///   (A+B) = Δε^KS δ + 4(ia|jb)_v − (ab|W|ij) − (ib|W|aj)
///   (A−B) = Δε^KS δ            + (ib|W|aj) − (ab|W|ij)
/// ```
/// with W from `run_pdep_rpa` on the KS reference. NB: this is RPAx-flavoured
/// TDDFT-with-screened-exchange, not a formally consistent GW@PBE BSE — the
/// cheap reference-isolation spike, deliberately.
/// `scissor` (Hartree) is added to every VIRTUAL orbital energy before assembling
/// the diagonal — a cheap proxy for the GW gap correction (KS gaps are too small;
/// the true GW gap is known exactly from run_gw). Pass 0.0 for plain KS. This
/// tests the α(iω)-falloff hypothesis: if widening the gap to the GW value fixes
/// the C6, the falloff is a gap problem (→ full GW@PBE worth building); if not,
/// it is intrinsic to the static-screened kernel.
#[allow(clippy::too_many_arguments)]
pub fn run_bse_c6_ks(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    ks: &ScfResult,
    pdep_cfg: &PdepRpaConfig,
    frozen_core: usize,
    freqs: &[f64],
    weights: &[f64],
    scissor: f64,
) -> Result<BseC6Result, FerricError> {
    use std::f64::consts::PI;
    if !matches!(ks.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General("run_bse_c6_ks: closed-shell only".into()));
    }
    let nmo = ks.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let c = ks.mos_r();
    let mut eps = ks.eps_r().to_vec(); // KS orbital energies, used on the diagonal
    for p in nocc_total..nmo {
        eps[p] += scissor; // scissor: shift virtuals up to the GW gap (cheap QP proxy)
    }

    // PDEP screening modes from the KS response.
    let pdep = ferric_rpa::run_pdep_rpa(mol, obs, dfbs, op, ks, pdep_cfg)?;
    {
        let lmax = pdep.eigenvalues_static.iter().cloned().fold(f64::MIN, f64::max);
        let homo = eps[nocc_total - 1];
        let lumo = eps[nocc_total];
        eprintln!(
            "RPAx@KS diag: λ_max(0)={lmax:.4}  n_modes={}  KS gap={:.3} eV",
            pdep.eigenvalues_static.len(),
            (lumo - homo) * 27.211_386_245_988
        );
    }

    let first_act = frozen_core;
    let nocc = nocc_total - frozen_core;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;

    let mob = mo_b::build_full_b(mol, obs, dfbs, op, ks, frozen_core)?;
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mob.v_inv_sqrt, &pdep.eigenpotentials)?;
    let m_proj = project_b_into_pdep(&mob, &v_dressed);
    let m_modes = m_proj.shape()[0];
    // Screened W = bare v + reduced-screening correction. Using the COMPLETE
    // b_full for the bare part and the reduced weight w_α = 1/λ_α − 1 for the
    // mode correction makes the screened integral robust to an INCOMPLETE PDEP
    // mode set (run_pdep_rpa drops near-unit modes, so eigenpotentials span only
    // M<naux dimensions; the naive Σ(1/λ)MM form then loses bare-exchange weight
    // — see the basis-check diagnostic). The reduced weight only multiplies the
    // genuine screening, which vanishes for the dropped λ≈1 modes.
    let w_red: Vec<f64> = pdep.eigenvalues_static.iter().map(|&l| 1.0 / l - 1.0).collect();
    let b = &mob.b_full;
    let naux = mob.naux;
    let bare = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for pp in 0..naux {
            acc += b[(pp, p, q)] * b[(pp, r, s)];
        }
        acc
    };
    let screened = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = bare(p, q, r, s); // bare v (complete RI basis)
        for alpha in 0..m_modes {
            acc += w_red[alpha] * m_proj[(alpha, p, q)] * m_proj[(alpha, r, s)];
        }
        acc
    };
    {
        // DIAGNOSTIC: compare bare vs screened exchange on the LUMO-LUMO element,
        // and the bare (HOMO a=0,a=0 | ...) — checks M-projection consistency vs PySCF.
        let a0 = nocc; // first virtual local index
        let i0 = nocc - 1; // HOMO local
        // BARE-LIMIT check on the off-diagonal pair coupling: Σ_α M[α,aa] M[α,ii]
        // (unit weights) MUST equal bare(aa|ii) if m_proj spans the full RI space.
        let mproj_bare_aaii: f64 =
            (0..m_modes).map(|al| m_proj[(al, a0, a0)] * m_proj[(al, i0, i0)]).sum();
        eprintln!(
            "RPAx@KS xch diag: bare(aa|ii)={:.5} W(aa|ii)={:.5}  bare(ai|ai)={:.5} W(ai|ai)={:.5}",
            bare(a0, a0, i0, i0),
            screened(a0, a0, i0, i0),
            bare(a0, i0, a0, i0),
            screened(a0, i0, a0, i0),
        );
        eprintln!(
            "RPAx@KS basis check: Σ_α M[aa]M[ii] (unit wt) = {:.5}  vs bare(aa|ii) {:.5}  (m_modes={}, naux={})",
            mproj_bare_aaii,
            bare(a0, a0, i0, i0),
            m_modes,
            naux,
        );
    }

    let mut apb = Array2::<f64>::zeros((n, n));
    let mut amb = Array2::<f64>::zeros((n, n));
    for i in 0..nocc {
        let eps_i = eps[first_act + i];
        for a in 0..nvir {
            let a_loc = nocc + a;
            let eps_a = eps[nocc_total + a];
            let ia = i * nvir + a;
            for j in 0..nocc {
                for bb in 0..nvir {
                    let b_loc = nocc + bb;
                    let jb = j * nvir + bb;
                    let coul = bare(i, a_loc, j, b_loc);
                    let w_abij = screened(a_loc, b_loc, i, j);
                    let w_ibaj = screened(i, b_loc, a_loc, j);
                    apb[(ia, jb)] = 4.0 * coul - w_abij - w_ibaj;
                    amb[(ia, jb)] = w_ibaj - w_abij;
                }
            }
            apb[(ia, ia)] += eps_a - eps_i;
            amb[(ia, ia)] += eps_a - eps_i;
        }
    }

    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let r_mo: [Array2<f64>; 3] = std::array::from_fn(|d| c.t().dot(&dip_ao[d]).dot(c));
    let mut mu: [Array1<f64>; 3] = std::array::from_fn(|_| Array1::zeros(n));
    for (d, m) in mu.iter_mut().enumerate() {
        for i in 0..nocc {
            for a in 0..nvir {
                m[i * nvir + a] = r_mo[d][(first_act + i, nocc_total + a)];
            }
        }
    }

    let mut alpha_iso = Vec::with_capacity(freqs.len());
    for &w in freqs {
        let mut sysm = amb.dot(&apb);
        for k in 0..n {
            sysm[(k, k)] += w * w;
        }
        let mut iso = 0.0;
        for axis in mu.iter() {
            let rhs = amb.dot(axis);
            let t = sysm
                .solve(&rhs)
                .map_err(|e| FerricError::Lapack(format!("RPAx α(iω) solve: {e}")))?;
            iso += 4.0 * axis.dot(&t);
        }
        alpha_iso.push(iso / 3.0);
    }
    let c6 = 3.0 / PI * (0..freqs.len()).map(|k| weights[k] * alpha_iso[k] * alpha_iso[k]).sum::<f64>();
    let alpha_static = *alpha_iso.first().unwrap_or(&0.0);
    Ok(BseC6Result { c6, alpha_iso, alpha_static, nocc, nvir })
}
