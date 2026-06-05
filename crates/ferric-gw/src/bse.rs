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
use ferric_rpa::PdepRpaConfig;
use ferric_scf::ScfResult;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};

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
    };
    let gw = run_gw(mol, obs, dfbs, op, rhf, pdep_cfg, &gw_cfg)?;

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

    // Full static screened weight 1/λ_α(0) for the screened-exchange term.
    let inv_lam: Vec<f64> = gw
        .pdep
        .eigenvalues_static
        .iter()
        .map(|&l| 1.0 / l)
        .collect();
    assert_eq!(inv_lam.len(), m_modes);
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
        let mut acc = 0.0;
        for alpha in 0..m_modes {
            acc += inv_lam[alpha] * m_proj[(alpha, p, q)] * m_proj[(alpha, r, s)];
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
