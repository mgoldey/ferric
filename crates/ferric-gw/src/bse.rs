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
//!
//! This module also holds the related TDHF/RPAx dense-(A±B) polarizability
//! family (`run_bse_c6`, `run_bse_c6_ks`, `run_rpax_static_polarizability`).
//! Of these, only `run_rpax_static_polarizability` is wired into the
//! CLI/Python surface — it is **static (ω=0) polarizability only**.
//! `run_bse_c6`/`run_bse_c6_ks` (dynamic α(iω) and C6) are deliberately
//! library-only: `docs/VALIDATION.md` records a validated negative result
//! (C6 from this exact kernel stays ~63% low regardless of the HOMO-LUMO
//! gap, worse than ferric's production dRPA/PDEP C6). See
//! `run_rpax_static_polarizability`'s doc for the full caveat.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_rpa::PdepRpaConfig;
use ferric_scf::ScfResult;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, Solve, UPLO};

use crate::cohsex::project_b_into_pdep;
use crate::method::GwMethod;
use crate::{mo_b, run_gw, w_pdep, GwConfig};

/// Fail-fast pre-flight guard for the dense particle–hole matrices these BSE/CIS
/// drivers build over the (ia) space of size `n = nocc·nvir`. `n_dense` is the
/// number of co-resident n×n f64 buffers at the peak: 1 for the single TDA
/// `a_mat` (:162, :243), 3 for the C6 drivers (apb + amb + per-frequency sysm,
/// :370-371/:409, :547-548/:584). Placed here so an M3/M5 restructure updates
/// the count in the same diff.
fn check_bse_dense_alloc(
    label: &str,
    n: usize,
    n_dense: usize,
    explicit_budget: Option<usize>,
) -> Result<(), FerricError> {
    let peak = n
        .saturating_mul(n)
        .saturating_mul(n_dense)
        .saturating_mul(8); // f64
    ferric_core::memory::check_alloc(
        &format!("{label} (dense (ia) matrix, n=nocc·nvir={n})"),
        peak,
        ferric_core::memory::resolve_budget_bytes(explicit_budget),
    )
}

/// Result of a BSE-TDA singlet excitation calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct BseResult {
    /// Singlet excitation energies Ω_n (Hartree), ascending.
    pub omega: Vec<f64>,
    /// Number of occupied / virtual orbitals in the BSE window (frozen-core aware).
    pub nocc: usize,
    pub nvir: usize,
    /// GW quasiparticle energies used for the diagonal (active block, Ha).
    pub eps_qp: Vec<f64>,
    /// Length-gauge oscillator strengths f_n (dimensionless), same ordering
    /// and length as `omega`. See `tda_oscillator_strengths` for the
    /// convention and its PySCF cross-check.
    pub oscillator_strength: Vec<f64>,
}

impl BseResult {
    /// Lowest singlet excitation energy in eV.
    pub fn lowest_ev(&self) -> f64 {
        self.omega[0] * 27.211_386_245_988
    }

    /// Oscillator strength of the lowest singlet.
    pub fn lowest_oscillator_strength(&self) -> f64 {
        self.oscillator_strength[0]
    }
}

impl std::fmt::Display for BseResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BSE-TDA: {} excitations, lowest {:.4} eV (f = {:.6})",
            self.omega.len(),
            self.lowest_ev(),
            self.lowest_oscillator_strength()
        )
    }
}

/// Length-gauge TDA/CIS oscillator strengths for every state in a solved
/// singlet particle–hole eigenproblem.
///
/// Standard closed-shell singlet TDA/CIS transition-dipole convention
/// (Szabo & Ostlund; Dreuw & Head-Gordon, Chem. Rev. 105, 4009 (2005) §2;
/// numerically cross-checked against PySCF's
/// `tdscf.rhf.TDA.oscillator_strength`, length gauge — see
/// `crates/ferric-gw/tests/bse_oscillator_strength.rs`):
///
/// ```text
///   <0|r|n> = sqrt(2) * sum_{ia} X_n(i,a) * <i|r|a>
///   f_n     = (2/3) * Omega_n * |<0|r|n>|^2
/// ```
///
/// `X_n(i,a)` is column `n` of the eigenvector matrix returned by the dense
/// `a_mat.eigh(...)` solve in [`run_bse_tda`]/[`run_cis_tda`] — LAPACK
/// `dsyev`/`dsyevd` normalizes each column to unit L2 norm
/// (`sum_{ia} X_n(i,a)^2 = 1`), which is the normalization this formula
/// assumes. The `sqrt(2)` factor is the closed-shell singlet spin-adaptation
/// factor (both spin channels contribute identically to a singlet
/// particle–hole excitation) — verified numerically against PySCF: PySCF's
/// internally-stored `xy` amplitude is `x1 * sqrt(0.5)` where `x1` is the
/// same unit-normalized eigenvector ferric's `eigh` returns, and
/// `_contract_multipole` contracts with an extra explicit factor of 2, i.e.
/// PySCF's effective prefactor is `2 * sqrt(0.5) = sqrt(2)` — matching this
/// implementation exactly (not just approximately) once the eigenvector
/// normalization convention is aligned.
///
/// `evecs` is `(n, n)` with `n = nocc*nvir`, row-major flat index
/// `ia = i*nvir + a` (LOCAL occ/vir indices within the active window, same
/// convention as `fill_row` in [`run_bse_tda`]/[`run_cis_tda`]). `mu_ao` are
/// the 3 AO dipole matrices about an arbitrary common origin (irrelevant —
/// occ/vir transition dipoles are origin-independent by orbital
/// orthogonality, also verified numerically). `mo_coeff` is the FULL
/// nbasis×nmo MO coefficient matrix; `first_act`/`nocc_total` locate the
/// active occ/vir columns within it.
fn tda_oscillator_strengths(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    mu_ao: &[Array2<f64>; 3],
    mo_coeff: &Array2<f64>,
    first_act: usize,
    nocc_total: usize,
    nocc: usize,
    nvir: usize,
) -> Vec<f64> {
    let n = nocc * nvir;
    debug_assert_eq!(evecs.shape(), &[n, n]);

    // Occ-virt dipole block <i|r|a> for i in the active occ window, a in the
    // active virt window (local indices), one Array2 per Cartesian axis.
    let orbo = mo_coeff.slice(ndarray::s![.., first_act..nocc_total]);
    let orbv = mo_coeff.slice(ndarray::s![.., nocc_total..(nocc_total + nvir)]);
    let dip_ia: [Array2<f64>; 3] = std::array::from_fn(|d| orbo.t().dot(&mu_ao[d]).dot(&orbv));

    let mut f = Vec::with_capacity(n);
    for state in 0..n {
        let x = evecs.column(state);
        let mut mu = [0.0_f64; 3];
        for (d, dip) in dip_ia.iter().enumerate() {
            let mut acc = 0.0;
            for i in 0..nocc {
                for a in 0..nvir {
                    acc += x[i * nvir + a] * dip[(i, a)];
                }
            }
            mu[d] = std::f64::consts::SQRT_2 * acc;
        }
        let mu2 = mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2];
        f.push((2.0 / 3.0) * evals[state] * mu2);
    }
    f
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
        verbose: pdep_cfg.verbose,
    };
    let gw = run_gw(mol, obs, dfbs, op, rhf, pdep_cfg, &gw_cfg, None)?;

    // Map absolute MO index → QP energy.
    let mut eps_qp_full = rhf.eps_r().to_vec(); // fallback = HF (should be fully overwritten)
    for (k, &mo) in gw.mo_indices.iter().enumerate() {
        eps_qp_full[mo] = gw.eps_qp[k];
    }

    // 2. Active particle–hole window (frozen-core aware).
    let first_act = frozen_core;
    let nocc = ferric_mp2::rimp2::active_occ(nocc_total, frozen_core)?;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;
    if n == 0 {
        return Err(FerricError::General("run_bse_tda: empty (ia) space".into()));
    }
    // Fail-fast on the dense TDA matrix a_mat (:162, one n×n f64 buffer).
    check_bse_dense_alloc("BSE-TDA", n, 1, pdep_cfg.memory_budget_bytes)?;

    // 3. Dressed B̃^P_{pq} over all MO pairs + projection M[α,p,q] = Σ_P Ṽ_α b̃^P_pq.
    let mob = mo_b::build_full_b(
        mol,
        obs,
        dfbs,
        op,
        rhf,
        frozen_core,
        pdep_cfg.memory_budget_bytes,
    )?;
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mob.v_inv_sqrt, &gw.pdep.eigenpotentials)?;
    let m_proj = project_b_into_pdep(&mob, &v_dressed, pdep_cfg.memory_budget_bytes)?; // (M, n_act, n_act), local MO indices
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
    //
    // Rows are independent: row `ia` is written exactly once, in full, by a
    // single (i,a) pair. Parallelize over the flat `ia` row axis of the SAME
    // preallocated matrix (order-preserving `par_chunks_mut`, no per-worker
    // copies, no reduction — bit-identical by construction). The `bare`/
    // `screened` closures are scalar contractions (no BLAS), so no
    // with_blas_threads guard is needed. Serial below PAR_ROWS_THRESHOLD.
    check_dense_response_alloc("BSE/TDA", n, 1, None)?;
    let mut a_mat = Array2::<f64>::zeros((n, n));
    let fill_row = |ia: usize, row: &mut [f64]| {
        let i = ia / nvir;
        let a = ia % nvir;
        let i_loc = i; // occupied-active local index
        let eps_i = eps_qp_full[first_act + i];
        let a_loc = nocc + a; // virtual local index in the active block
        let eps_a = eps_qp_full[nocc_total + a];
        for j in 0..nocc {
            let j_loc = j;
            for bb in 0..nvir {
                let b_loc = nocc + bb;
                let jb = j * nvir + bb;
                let coul = bare(i_loc, a_loc, j_loc, b_loc); // (ia|jb)
                let scr = screened(a_loc, b_loc, i_loc, j_loc); // (ab|W|ij)
                row[jb] = 2.0 * coul - scr;
            }
        }
        row[ia] += eps_a - eps_i;
    };
    const PAR_ROWS_THRESHOLD: usize = 8;
    if n >= PAR_ROWS_THRESHOLD {
        use rayon::prelude::*;
        a_mat
            .as_slice_mut()
            .expect("a_mat is contiguous (row-major default)")
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(ia, row)| fill_row(ia, row));
    } else {
        let flat = a_mat
            .as_slice_mut()
            .expect("a_mat is contiguous (row-major default)");
        for (ia, row) in flat.chunks_mut(n).enumerate() {
            fill_row(ia, row);
        }
    }

    // 5. Eigenvalues (Hermitian). The nov×nov dense eigendecomposition is the
    //    dominant serial cost of BSE-TDA — raise BLAS threads on it under the
    //    opt-in guard.
    //    Call-path proof: this eigh runs AFTER the row-parallel `a_mat` fill
    //    above (that par region has fully joined — `a_mat` is consumed here) and
    //    `run_bse_tda` is a top-level driver invoked only from serial callers
    //    (CLI/Python/tests), never from inside a rayon par_iter. So no enclosing
    //    parallel region; opt_in_blas_threads() self-guards to 1 from a rayon
    //    worker regardless, and defaults to 1 (bit-identical to today).
    let (evals, evecs) = with_blas_threads(opt_in_blas_threads(), || a_mat.eigh(UPLO::Upper))
        .map_err(|e| FerricError::Lapack(format!("BSE-TDA eigh: {e}")))?;
    // LAPACK dsyev/dsyevd (what ndarray-linalg's eigh wraps) returns
    // eigenvalues already ascending, with evecs columns in the matching
    // order — no re-sort needed, and re-sorting `omega` alone (as this used
    // to do) would have silently DEsynchronized it from `evecs`'s column
    // order once oscillator strengths needed both together. Assert the
    // ascending invariant instead of re-deriving it.
    debug_assert!(
        evals
            .as_slice()
            .unwrap()
            .windows(2)
            .all(|w| w[0] <= w[1] + 1e-9),
        "eigh eigenvalues expected ascending"
    );
    let omega: Vec<f64> = evals.to_vec();

    let mu_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let oscillator_strength = tda_oscillator_strengths(
        &evals,
        &evecs,
        &mu_ao,
        rhf.mos_r(),
        first_act,
        nocc_total,
        nocc,
        nvir,
    );

    let eps_qp_act: Vec<f64> = (first_act..nmo).map(|p| eps_qp_full[p]).collect();
    Ok(BseResult {
        omega,
        nocc,
        nvir,
        eps_qp: eps_qp_act,
        oscillator_strength,
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
///
/// Pre-flight for the dense BSE/TDHF response matrices.
///
/// These paths build dense `(n, n)` matrices with `n = nocc · nvir` and then
/// diagonalize them, so the resident peak is `n_mats` matrices plus the `eigh`
/// output (eigenvectors, another `n²`). `bse.rs` had no guard at all: the
/// matrices were allocated and the OOM killer decided.
///
/// Modest next to the other offenders in this crate family — 0.206 GB for the
/// two-matrix path at benzene/aug-cc-pVDZ (n = 3591) — but it grows as
/// `(nocc·nvir)²`, i.e. the fourth power of system size, so the headroom
/// disappears quickly. `dense O(nmo⁴)` is exactly what the module docs already
/// warn about; this makes the warning enforced.
fn check_dense_response_alloc(
    label: &str,
    n: usize,
    n_mats: usize,
    memory_budget_bytes: Option<usize>,
) -> Result<(), ferric_core::FerricError> {
    let per_mat = n.saturating_mul(n).saturating_mul(8);
    // +1 for the eigh eigenvector output, which is co-resident with the input.
    let bytes = per_mat.saturating_mul(n_mats.saturating_add(1));
    ferric_core::memory::check_alloc(
        &format!("{label} dense response matrices (n = nocc*nvir = {n}, {n_mats} matrices + eigh output)"),
        bytes,
        ferric_core::memory::resolve_budget_bytes(memory_budget_bytes),
    )
}

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
        return Err(FerricError::General(
            "run_cis_tda: closed-shell (RHF) only".into(),
        ));
    }
    let nmo = rhf.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let eps = rhf.eps_r().to_vec();

    let first_act = frozen_core;
    let nocc = ferric_mp2::rimp2::active_occ(nocc_total, frozen_core)?;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;
    // Fail-fast on the dense TDA matrix a_mat (:243, one n×n f64 buffer). No
    // config budget on this reference path, so resolve from env/auto.
    check_bse_dense_alloc("CIS-TDA", n, 1, None)?;

    // Same "no config budget on this reference path" reasoning as the guard
    // above: `run_cis_tda` takes no config struct, so the RI build resolves
    // from env/auto rather than from a caller ceiling.
    let mob = mo_b::build_full_b(mol, obs, dfbs, op, rhf, frozen_core, None)?;
    let b = &mob.b_full;
    let naux = mob.naux;
    let bare = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for pp in 0..naux {
            acc += b[(pp, p, q)] * b[(pp, r, s)];
        }
        acc
    };

    // Same row-independent structure as `run_bse_tda` (see the comment there):
    // row `ia` is written exactly once by a single (i,a) pair, so parallelize
    // over the flat `ia` axis with order-preserving `par_chunks_mut` into the
    // SAME preallocated matrix. No BLAS inside `bare`, so no
    // with_blas_threads guard needed. Serial below PAR_ROWS_THRESHOLD.
    check_dense_response_alloc("BSE/TDA", n, 1, None)?;
    let mut a_mat = Array2::<f64>::zeros((n, n));
    let fill_row = |ia: usize, row: &mut [f64]| {
        let i = ia / nvir;
        let a = ia % nvir;
        let eps_i = eps[first_act + i];
        let a_loc = nocc + a;
        let eps_a = eps[nocc_total + a];
        for j in 0..nocc {
            for bb in 0..nvir {
                let b_loc = nocc + bb;
                let jb = j * nvir + bb;
                let coul = bare(i, a_loc, j, b_loc); // (ia|jb)
                let exch = bare(a_loc, b_loc, i, j); // (ab|ij)  bare
                row[jb] = 2.0 * coul - exch;
            }
        }
        row[ia] += eps_a - eps_i;
    };
    const PAR_ROWS_THRESHOLD: usize = 8;
    if n >= PAR_ROWS_THRESHOLD {
        use rayon::prelude::*;
        a_mat
            .as_slice_mut()
            .expect("a_mat is contiguous (row-major default)")
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(ia, row)| fill_row(ia, row));
    } else {
        let flat = a_mat
            .as_slice_mut()
            .expect("a_mat is contiguous (row-major default)");
        for (ia, row) in flat.chunks_mut(n).enumerate() {
            fill_row(ia, row);
        }
    }
    // nov×nov Hermitian eigendecomposition — same BLAS-raise treatment and
    // call-path proof as run_bse_tda's eigh: it runs after the row-parallel
    // a_mat fill (joined; a_mat consumed here) and run_cis_tda is a top-level
    // serial driver, never called from a rayon par_iter. opt_in_blas_threads()
    // self-guards to 1 from a rayon worker and defaults to 1.
    let (evals, evecs) = with_blas_threads(opt_in_blas_threads(), || a_mat.eigh(UPLO::Upper))
        .map_err(|e| FerricError::Lapack(format!("CIS-TDA eigh: {e}")))?;
    // See run_bse_tda's identical comment: eigh already returns evals
    // ascending with evecs columns in matching order; don't re-sort omega
    // alone (that would desync it from evecs before oscillator strengths are
    // built from both).
    let omega: Vec<f64> = evals.to_vec();

    let mu_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let oscillator_strength = tda_oscillator_strengths(
        &evals,
        &evecs,
        &mu_ao,
        rhf.mos_r(),
        first_act,
        nocc_total,
        nocc,
        nvir,
    );

    let eps_act: Vec<f64> = (first_act..nmo).map(|p| eps[p]).collect();
    Ok(BseResult {
        omega,
        nocc,
        nvir,
        eps_qp: eps_act,
        oscillator_strength,
    })
}

/// Result of a BSE dynamic-polarizability / C6 calculation (gate 2).
#[derive(Debug, Clone)]
#[must_use]
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
        return Err(FerricError::General(
            "run_bse_c6: closed-shell (RHF) only".into(),
        ));
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
        verbose: pdep_cfg.verbose,
    };
    let gw = run_gw(mol, obs, dfbs, op, rhf, pdep_cfg, &gw_cfg, None)?;
    let mut eps_qp = rhf.eps_r().to_vec();
    for (k, &mo) in gw.mo_indices.iter().enumerate() {
        eps_qp[mo] = gw.eps_qp[k];
    }

    let first_act = frozen_core;
    let nocc = ferric_mp2::rimp2::active_occ(nocc_total, frozen_core)?;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;
    // Fail-fast on the dense (A±B) + per-frequency sysm buffers (:398-399, :437 —
    // 3 co-resident n×n f64 buffers).
    check_bse_dense_alloc("BSE-C6", n, 3, pdep_cfg.memory_budget_bytes)?;

    // Projected screened modes + bare integrals (same path as run_bse_tda).
    let mob = mo_b::build_full_b(
        mol,
        obs,
        dfbs,
        op,
        rhf,
        frozen_core,
        pdep_cfg.memory_budget_bytes,
    )?;
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mob.v_inv_sqrt, &gw.pdep.eigenpotentials)?;
    let m_proj = project_b_into_pdep(&mob, &v_dressed, pdep_cfg.memory_budget_bytes)?;
    let m_modes = m_proj.shape()[0];
    // Reduced screening weight; screened W = bare v + Σ w_α MM (mode-set-robust,
    // see run_bse_c6_ks).
    let w_red: Vec<f64> = gw
        .pdep
        .eigenvalues_static
        .iter()
        .map(|&l| 1.0 / l - 1.0)
        .collect();
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
    //
    // Row `ia` of BOTH apb and amb is written exactly once by a single (i,a)
    // pair — independent rows. Zip the two matrices' row-chunk iterators and
    // parallelize over the flat `ia` axis, filling the SAME two preallocated
    // matrices (order-preserving, no reduction, bit-identical by
    // construction). No BLAS inside `bare`/`screened`. Serial below
    // PAR_ROWS_THRESHOLD.
    check_dense_response_alloc("BSE/TDHF", n, 2, None)?;
    let mut apb = Array2::<f64>::zeros((n, n));
    let mut amb = Array2::<f64>::zeros((n, n));
    let fill_row = |ia: usize, apb_row: &mut [f64], amb_row: &mut [f64]| {
        let i = ia / nvir;
        let a = ia % nvir;
        let eps_i = eps_qp[first_act + i];
        let a_loc = nocc + a;
        let eps_a = eps_qp[nocc_total + a];
        for j in 0..nocc {
            for bb in 0..nvir {
                let b_loc = nocc + bb;
                let jb = j * nvir + bb;
                let coul = bare(i, a_loc, j, b_loc); // (ia|jb)
                let w_abij = screened(a_loc, b_loc, i, j); // (ab|W|ij)
                let w_ibaj = screened(i, b_loc, a_loc, j); // (ib|W|aj)
                apb_row[jb] = 4.0 * coul - w_abij - w_ibaj;
                amb_row[jb] = w_ibaj - w_abij;
            }
        }
        apb_row[ia] += eps_a - eps_i;
        amb_row[ia] += eps_a - eps_i;
    };
    const PAR_ROWS_THRESHOLD: usize = 8;
    if n >= PAR_ROWS_THRESHOLD {
        use rayon::prelude::*;
        let apb_flat = apb
            .as_slice_mut()
            .expect("apb is contiguous (row-major default)");
        let amb_flat = amb
            .as_slice_mut()
            .expect("amb is contiguous (row-major default)");
        apb_flat
            .par_chunks_mut(n)
            .zip(amb_flat.par_chunks_mut(n))
            .enumerate()
            .for_each(|(ia, (apb_row, amb_row))| fill_row(ia, apb_row, amb_row));
    } else {
        let apb_flat = apb
            .as_slice_mut()
            .expect("apb is contiguous (row-major default)");
        let amb_flat = amb
            .as_slice_mut()
            .expect("amb is contiguous (row-major default)");
        for (ia, (apb_row, amb_row)) in apb_flat
            .chunks_mut(n)
            .zip(amb_flat.chunks_mut(n))
            .enumerate()
        {
            fill_row(ia, apb_row, amb_row);
        }
    }

    // Dipole μ in (ia)-space (bare operator), per Cartesian axis.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
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
    // Frequencies are independent (each does a full n×n GEMM + solve) — par
    // over freqs, order-preserving collect (energy.rs pattern), BLAS pinned
    // to 1 inside the rayon region to avoid nested-thread oversubscription /
    // the dgetrf stack-overflow crash site.
    let alpha_iso: Vec<f64> = {
        use rayon::prelude::*;
        const PAR_FREQ_THRESHOLD: usize = 4;
        let compute_one = |&w: &f64| -> Result<f64, FerricError> {
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
            Ok(iso / 3.0)
        };
        if freqs.len() >= PAR_FREQ_THRESHOLD {
            with_blas_threads(1, || {
                freqs
                    .par_iter()
                    .map(compute_one)
                    .collect::<Result<Vec<f64>, FerricError>>()
            })?
        } else {
            freqs
                .iter()
                .map(compute_one)
                .collect::<Result<Vec<f64>, FerricError>>()?
        }
    };

    let c6 = 3.0 / PI
        * (0..freqs.len())
            .map(|k| weights[k] * alpha_iso[k] * alpha_iso[k])
            .sum::<f64>();
    let alpha_static = *alpha_iso.first().unwrap_or(&0.0);
    Ok(BseC6Result {
        c6,
        alpha_iso,
        alpha_static,
        nocc,
        nvir,
    })
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
        return Err(FerricError::General(
            "run_bse_c6_ks: closed-shell only".into(),
        ));
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
        let lmax = pdep
            .eigenvalues_static
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
        let homo = eps[nocc_total - 1];
        let lumo = eps[nocc_total];
        eprintln!(
            "RPAx@KS diag: λ_max(0)={lmax:.4}  n_modes={}  KS gap={:.3} eV",
            pdep.eigenvalues_static.len(),
            (lumo - homo) * 27.211_386_245_988
        );
    }

    let first_act = frozen_core;
    let nocc = ferric_mp2::rimp2::active_occ(nocc_total, frozen_core)?;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;
    // Fail-fast on the dense (A±B) + per-frequency sysm buffers (:575-576, :614 —
    // 3 co-resident n×n f64 buffers).
    check_bse_dense_alloc("BSE-C6 (KS)", n, 3, pdep_cfg.memory_budget_bytes)?;

    let mob = mo_b::build_full_b(
        mol,
        obs,
        dfbs,
        op,
        ks,
        frozen_core,
        pdep_cfg.memory_budget_bytes,
    )?;
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mob.v_inv_sqrt, &pdep.eigenpotentials)?;
    let m_proj = project_b_into_pdep(&mob, &v_dressed, pdep_cfg.memory_budget_bytes)?;
    let m_modes = m_proj.shape()[0];
    // Screened W = bare v + reduced-screening correction. Using the COMPLETE
    // b_full for the bare part and the reduced weight w_α = 1/λ_α − 1 for the
    // mode correction makes the screened integral robust to an INCOMPLETE PDEP
    // mode set (run_pdep_rpa drops near-unit modes, so eigenpotentials span only
    // M<naux dimensions; the naive Σ(1/λ)MM form then loses bare-exchange weight
    // — see the basis-check diagnostic). The reduced weight only multiplies the
    // genuine screening, which vanishes for the dropped λ≈1 modes.
    let w_red: Vec<f64> = pdep
        .eigenvalues_static
        .iter()
        .map(|&l| 1.0 / l - 1.0)
        .collect();
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
        let mproj_bare_aaii: f64 = (0..m_modes)
            .map(|al| m_proj[(al, a0, a0)] * m_proj[(al, i0, i0)])
            .sum();
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

    // Same row-independent structure as `run_bse_c6` (see the comment there):
    // row `ia` of both apb/amb written exactly once by a single (i,a) pair.
    check_dense_response_alloc("BSE/TDHF", n, 2, None)?;
    let mut apb = Array2::<f64>::zeros((n, n));
    let mut amb = Array2::<f64>::zeros((n, n));
    let fill_row = |ia: usize, apb_row: &mut [f64], amb_row: &mut [f64]| {
        let i = ia / nvir;
        let a = ia % nvir;
        let eps_i = eps[first_act + i];
        let a_loc = nocc + a;
        let eps_a = eps[nocc_total + a];
        for j in 0..nocc {
            for bb in 0..nvir {
                let b_loc = nocc + bb;
                let jb = j * nvir + bb;
                let coul = bare(i, a_loc, j, b_loc);
                let w_abij = screened(a_loc, b_loc, i, j);
                let w_ibaj = screened(i, b_loc, a_loc, j);
                apb_row[jb] = 4.0 * coul - w_abij - w_ibaj;
                amb_row[jb] = w_ibaj - w_abij;
            }
        }
        apb_row[ia] += eps_a - eps_i;
        amb_row[ia] += eps_a - eps_i;
    };
    const PAR_ROWS_THRESHOLD: usize = 8;
    if n >= PAR_ROWS_THRESHOLD {
        use rayon::prelude::*;
        let apb_flat = apb
            .as_slice_mut()
            .expect("apb is contiguous (row-major default)");
        let amb_flat = amb
            .as_slice_mut()
            .expect("amb is contiguous (row-major default)");
        apb_flat
            .par_chunks_mut(n)
            .zip(amb_flat.par_chunks_mut(n))
            .enumerate()
            .for_each(|(ia, (apb_row, amb_row))| fill_row(ia, apb_row, amb_row));
    } else {
        let apb_flat = apb
            .as_slice_mut()
            .expect("apb is contiguous (row-major default)");
        let amb_flat = amb
            .as_slice_mut()
            .expect("amb is contiguous (row-major default)");
        for (ia, (apb_row, amb_row)) in apb_flat
            .chunks_mut(n)
            .zip(amb_flat.chunks_mut(n))
            .enumerate()
        {
            fill_row(ia, apb_row, amb_row);
        }
    }

    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let r_mo: [Array2<f64>; 3] = std::array::from_fn(|d| c.t().dot(&dip_ao[d]).dot(c));
    let mut mu: [Array1<f64>; 3] = std::array::from_fn(|_| Array1::zeros(n));
    for (d, m) in mu.iter_mut().enumerate() {
        for i in 0..nocc {
            for a in 0..nvir {
                m[i * nvir + a] = r_mo[d][(first_act + i, nocc_total + a)];
            }
        }
    }

    // Frequencies independent — same par-over-freqs pattern as run_bse_c6.
    let alpha_iso: Vec<f64> = {
        use rayon::prelude::*;
        const PAR_FREQ_THRESHOLD: usize = 4;
        let compute_one = |&w: &f64| -> Result<f64, FerricError> {
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
            Ok(iso / 3.0)
        };
        if freqs.len() >= PAR_FREQ_THRESHOLD {
            with_blas_threads(1, || {
                freqs
                    .par_iter()
                    .map(compute_one)
                    .collect::<Result<Vec<f64>, FerricError>>()
            })?
        } else {
            freqs
                .iter()
                .map(compute_one)
                .collect::<Result<Vec<f64>, FerricError>>()?
        }
    };
    let c6 = 3.0 / PI
        * (0..freqs.len())
            .map(|k| weights[k] * alpha_iso[k] * alpha_iso[k])
            .sum::<f64>();
    let alpha_static = *alpha_iso.first().unwrap_or(&0.0);
    Ok(BseC6Result {
        c6,
        alpha_iso,
        alpha_static,
        nocc,
        nvir,
    })
}

/// Acceptance floor (a.u.) for a α diagonal element: `α_dd` must exceed this,
/// not merely be `> 0.0`.
///
/// A diagonal element of the static polarizability is a physical response
/// magnitude, strictly positive for any bound electronic system. Anything in
/// `(-∞, ALPHA_DIAGONAL_ZERO_TOL]` is REJECTED. The window around zero is not a
/// slack band — it is *extra* strictness: a diagonal of, say, `+1e-12` a.u. is
/// not a small polarizability, it is a degenerate solve, and accepting it just
/// because it clears a bare sign test would put the guard's pass condition at
/// the mercy of rounding. The sign of such a value is arbitrary.
///
/// The error message still distinguishes "vanishing" from "negative", because
/// the two have different likely causes (a collapsed/zero-response axis vs. the
/// known excitonic instability), but both are refusals.
///
/// Scale rationale: measured α diagonals on the systems this path is used for
/// are O(0.05)–O(20) a.u. (water/cc-pVDZ (−2.68, +14.60, +16.49); water/STO-3G
/// at scissor=0.36 has an α_xx of +0.051), and the dense ω=0 solve's own
/// residual is ~1e-10. 1e-8 a.u. sits many orders below the smallest physical
/// value observed and above solver noise.
pub const ALPHA_DIAGONAL_ZERO_TOL: f64 = 1e-8;

/// Honest-failure guard on a static polarizability tensor: every Cartesian
/// diagonal element must be strictly positive.
///
/// A negative α_dd says the molecule polarizes *against* an applied field along
/// axis d, which is physically impossible for a bound closed-shell system. On
/// the RPAx@KS path it is a real, understood signal about the INPUT — a genuine
/// BSE/TDDFT excitonic instability caused by pairing a strongly static-screened
/// exchange kernel with the SAME narrow, GW-uncorrected KS gap on the response
/// diagonal, which drives `(A−B)` (and sometimes even the TDA `A`) non-positive
/// -definite. It is NOT an assembly bug: the kernel was independently
/// re-derived and matches PySCF TDHF to 3e-4 in the bare-exchange limit (see
/// `crates/ferric-gw/tests/rpax_bare_ab_check.rs`), and the full root-cause
/// investigation is in `docs/rpax-negative-diagonal-investigation.md`.
///
/// Following the workspace "honest failure over silent-wrong" convention (see
/// CLAUDE.md Reliability Conventions: solver honesty, TS/MBD honesty), this
/// REFUSES rather than warning, clamping, or taking an absolute value: the
/// isotropic average can look perfectly reasonable while hiding an unphysical
/// tensor (water/cc-pVDZ: iso +9.47 a.u., within 2% of the DOSD reference,
/// while α_xx is −2.68), so a warning on stderr would be trivially missed by
/// any programmatic consumer and the number would be used as if valid.
///
/// The error message names the known remedies because they are established, not
/// speculative: a nonzero `scissor` (~0.3–0.4 Ha for small molecules) or a
/// diagonal sourced from real `run_gw` G0W0 QP energies both restore
/// positive-definiteness on every previously-failing system.
///
/// `context` is prefixed to the message so callers of different entry points
/// are distinguishable in a log.
pub fn check_alpha_diagonal_positive(
    context: &str,
    tensor: &[[f64; 3]; 3],
) -> Result<(), FerricError> {
    const AXES: [&str; 3] = ["x", "y", "z"];
    // `!(x > tol)` rather than `x <= tol` so NaN is REJECTED (NaN fails every
    // ordered comparison, so `x <= tol` would silently let it through).
    let bad: Vec<usize> = (0..3)
        .filter(|&d| !(tensor[d][d] > ALPHA_DIAGONAL_ZERO_TOL))
        .collect();
    if bad.is_empty() {
        return Ok(());
    }
    let detail = bad
        .iter()
        .map(|&d| {
            let v = tensor[d][d];
            let kind = if v.is_nan() {
                "NaN"
            } else if v.abs() < ALPHA_DIAGONAL_ZERO_TOL {
                "vanishing (|a| < 1e-8)"
            } else {
                "negative"
            };
            format!("alpha_{0}{0} = {v:+.6e} ({kind})", AXES[d])
        })
        .collect::<Vec<_>>()
        .join(", ");
    Err(FerricError::General(format!(
        "{context}: unphysical static polarizability -- every Cartesian diagonal element must be \
         strictly positive, but {detail}. A non-positive alpha_dd means the system responds \
         against the applied field along that axis, which cannot happen for a bound closed-shell \
         molecule. On the RPAx@KS path this is a genuine BSE/TDDFT excitonic instability from \
         pairing a static-screened exchange kernel with the same GW-uncorrected KS gap on the \
         response diagonal (NOT an assembly bug -- the kernel matches PySCF TDHF to 3e-4 in the \
         bare-exchange limit). REMEDY: re-run with a nonzero scissor (~0.3-0.4 Ha for small \
         molecules; [gw] scissor in the CLI TOML, scissor= in Python) to widen the diagonal \
         toward the true GW gap, or source the diagonal from real G0W0 quasiparticle energies. \
         Full diagonal: ({:+.6e}, {:+.6e}, {:+.6e}). Root cause: \
         docs/rpax-negative-diagonal-investigation.md",
        tensor[0][0], tensor[1][1], tensor[2][2]
    )))
}

/// Result of a static-only RPAx@KS polarizability calculation.
///
/// **Scope: static (ω=0) polarizability only.** This is deliberately narrow —
/// see the module-level and function-level docs on [`run_rpax_static_polarizability`]
/// for why the dynamic α(iω)/C6 variant (`run_bse_c6_ks`) is NOT exposed as a
/// production capability alongside this one.
#[derive(Debug, Clone)]
#[must_use]
pub struct RpaxStaticPolarizabilityResult {
    /// Cartesian α_ij(0) tensor, i,j ∈ {x,y,z}, in a.u. (e²·a₀²/E_h).
    pub tensor: [[f64; 3]; 3],
    /// Isotropic average (1/3) Tr α.
    pub iso: f64,
    pub nocc: usize,
    pub nvir: usize,
}

/// RPAx@KS **static** polarizability (ω=0 only) on a Kohn–Sham reference.
///
/// Same screened-(A±B) kernel as [`run_bse_c6_ks`] (KS orbital energies on the
/// diagonal, PDEP screening modes built from the SAME KS response — no GW),
/// but solves the CPHF-like linear response ONLY at ω=0 instead of looping
/// over a Casimir–Polder imaginary-frequency grid. That keeps this entry
/// point cheap (one dense linear solve per Cartesian axis, not one per
/// frequency point) and, more importantly, keeps it HONEST about what it
/// computes: static polarizability, nothing else.
///
/// ```text
///   (A+B) = Δε^KS δ + 4(ia|jb)_v − (ab|W|ij) − (ib|W|aj)
///   (A−B) = Δε^KS δ            + (ib|W|aj) − (ab|W|ij)
///   α_ij(0) = 4 μ_iᵀ (A−B) [(A−B)(A+B)]⁻¹ μ_j
/// ```
///
/// # Scope — READ BEFORE USING FOR ANYTHING BEYOND α(0)
///
/// This function and its CLI/Python wiring are **static polarizability
/// only**. Do **not** use this method, or extrapolate from its output, for
/// C6/dispersion coefficients. `docs/VALIDATION.md`'s "Correlation / response
/// (RPA, GW, BSE)" table records a validated NEGATIVE result for the dynamic
/// extension of this exact kernel: RPAx@PBE static α matches DOSD water
/// almost exactly (9.24 vs 9.64 a.u.), but the C6 built from α(iω) on the
/// same kernel stays ~63% low regardless of the HOMO-LUMO gap (a scissor-
/// shift scan from the KS gap to the true GW gap ruled out "just a gap
/// problem" — α(iω) itself falls off ~2× too fast at higher imaginary
/// frequency). That is a *worse*, mechanistically *different* C6 failure
/// than ferric's existing production dRPA/PDEP C6 pipeline (~−12 to −16%
/// deficit). The dynamic/C6 variant (`run_bse_c6_ks`) remains library-only
/// and unwired from the CLI/Python surface for exactly this reason.
///
/// `scissor` (Hartree) is added to every virtual orbital energy before
/// assembling the diagonal, matching `run_bse_c6_ks`'s knob (a cheap proxy
/// for widening the KS gap toward the true GW gap). Pass 0.0 for plain KS.
///
/// # Errors — unphysical α diagonal at `scissor = 0.0`
///
/// At the default `scissor = 0.0` this kernel can produce an α tensor with a
/// **negative diagonal element** even for small closed-shell molecules at
/// equilibrium geometry (water/cc-pVDZ, water/STO-3G, LiH/cc-pVDZ all do). That
/// is a real excitonic instability, not an assembly bug, and this function now
/// **returns `Err`** rather than handing the caller a physically impossible
/// tensor — see [`check_alpha_diagonal_positive`] for the full rationale and
/// the remedies (a nonzero `scissor` ~0.3–0.4 Ha, or a GW-corrected diagonal).
/// The refusal is deliberate: the isotropic average can look accurate while the
/// tensor is unphysical, so a silent return would be used as if valid.
#[allow(clippy::too_many_arguments)]
pub fn run_rpax_static_polarizability(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    ks: &ScfResult,
    pdep_cfg: &PdepRpaConfig,
    frozen_core: usize,
    scissor: f64,
) -> Result<RpaxStaticPolarizabilityResult, FerricError> {
    if !matches!(ks.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General(
            "run_rpax_static_polarizability: closed-shell only".into(),
        ));
    }
    let nmo = ks.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let c = ks.mos_r();
    let mut eps = ks.eps_r().to_vec();
    for p in nocc_total..nmo {
        eps[p] += scissor;
    }

    // PDEP screening modes from the KS response (same as run_bse_c6_ks).
    let pdep = ferric_rpa::run_pdep_rpa(mol, obs, dfbs, op, ks, pdep_cfg)?;

    let first_act = frozen_core;
    let nocc = ferric_mp2::rimp2::active_occ(nocc_total, frozen_core)?;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;
    // Fail-fast on the dense (A±B) buffers: apb + amb + sysm co-resident, same
    // 3 n×n f64 buffers as run_bse_c6_ks's per-frequency solve (here there's
    // only ONE "frequency", ω=0, so peak residency is identical).
    check_bse_dense_alloc(
        "RPAx static polarizability (KS)",
        n,
        3,
        pdep_cfg.memory_budget_bytes,
    )?;

    let mob = mo_b::build_full_b(
        mol,
        obs,
        dfbs,
        op,
        ks,
        frozen_core,
        pdep_cfg.memory_budget_bytes,
    )?;
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mob.v_inv_sqrt, &pdep.eigenpotentials)?;
    let m_proj = project_b_into_pdep(&mob, &v_dressed, pdep_cfg.memory_budget_bytes)?;
    let m_modes = m_proj.shape()[0];
    let w_red: Vec<f64> = pdep
        .eigenvalues_static
        .iter()
        .map(|&l| 1.0 / l - 1.0)
        .collect();
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
        let mut acc = bare(p, q, r, s);
        for alpha in 0..m_modes {
            acc += w_red[alpha] * m_proj[(alpha, p, q)] * m_proj[(alpha, r, s)];
        }
        acc
    };

    check_dense_response_alloc("BSE/TDHF", n, 2, None)?;
    let mut apb = Array2::<f64>::zeros((n, n));
    let mut amb = Array2::<f64>::zeros((n, n));
    let fill_row = |ia: usize, apb_row: &mut [f64], amb_row: &mut [f64]| {
        let i = ia / nvir;
        let a = ia % nvir;
        let eps_i = eps[first_act + i];
        let a_loc = nocc + a;
        let eps_a = eps[nocc_total + a];
        for j in 0..nocc {
            for bb in 0..nvir {
                let b_loc = nocc + bb;
                let jb = j * nvir + bb;
                let coul = bare(i, a_loc, j, b_loc);
                let w_abij = screened(a_loc, b_loc, i, j);
                let w_ibaj = screened(i, b_loc, a_loc, j);
                apb_row[jb] = 4.0 * coul - w_abij - w_ibaj;
                amb_row[jb] = w_ibaj - w_abij;
            }
        }
        apb_row[ia] += eps_a - eps_i;
        amb_row[ia] += eps_a - eps_i;
    };
    const PAR_ROWS_THRESHOLD: usize = 8;
    if n >= PAR_ROWS_THRESHOLD {
        use rayon::prelude::*;
        let apb_flat = apb
            .as_slice_mut()
            .expect("apb is contiguous (row-major default)");
        let amb_flat = amb
            .as_slice_mut()
            .expect("amb is contiguous (row-major default)");
        apb_flat
            .par_chunks_mut(n)
            .zip(amb_flat.par_chunks_mut(n))
            .enumerate()
            .for_each(|(ia, (apb_row, amb_row))| fill_row(ia, apb_row, amb_row));
    } else {
        let apb_flat = apb
            .as_slice_mut()
            .expect("apb is contiguous (row-major default)");
        let amb_flat = amb
            .as_slice_mut()
            .expect("amb is contiguous (row-major default)");
        for (ia, (apb_row, amb_row)) in apb_flat
            .chunks_mut(n)
            .zip(amb_flat.chunks_mut(n))
            .enumerate()
        {
            fill_row(ia, apb_row, amb_row);
        }
    }

    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let r_mo: [Array2<f64>; 3] = std::array::from_fn(|d| c.t().dot(&dip_ao[d]).dot(c));
    let mut mu: [Array1<f64>; 3] = std::array::from_fn(|_| Array1::zeros(n));
    for (d, m) in mu.iter_mut().enumerate() {
        for i in 0..nocc {
            for a in 0..nvir {
                m[i * nvir + a] = r_mo[d][(first_act + i, nocc_total + a)];
            }
        }
    }

    // ω=0 ONLY: a single dense solve, no frequency loop. sysm = (A−B)(A+B)
    // (the ω²·I shift vanishes at ω=0).
    let sysm = amb.dot(&apb);
    // α_ij(0) = 4 μ_iᵀ (A−B) sysm⁻¹ μ_j, full 3×3 tensor (not just the
    // isotropic average run_bse_c6_ks reports).
    let mut t: [Array1<f64>; 3] = std::array::from_fn(|_| Array1::zeros(n));
    for d in 0..3 {
        let rhs = amb.dot(&mu[d]);
        t[d] = sysm
            .solve(&rhs)
            .map_err(|e| FerricError::Lapack(format!("RPAx static α(0) solve: {e}")))?;
    }
    let mut tensor = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            tensor[i][j] = 4.0 * mu[i].dot(&t[j]);
        }
    }
    // HONEST FAILURE: refuse to hand back a physically impossible tensor. This
    // sits in the LIBRARY function, not in the CLI arm, so the CLI, the Python
    // binding (`run_tdhf_static_polarizability`) and any direct Rust caller are
    // all covered by the one check -- a CLI-only guard would leave the other
    // two silently wrong. See `check_alpha_diagonal_positive`.
    check_alpha_diagonal_positive("run_rpax_static_polarizability", &tensor)?;
    let iso = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    Ok(RpaxStaticPolarizabilityResult {
        tensor,
        iso,
        nocc,
        nvir,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    // FERRIC_MEM_BUDGET_GB is process-global; serialize env-mutating tests
    // (blas_threads.rs / ferric-core memory.rs pattern).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // ── alpha-diagonal positivity guard (unit level) ──────────────────────
    //
    // These are the CHEAP half of the guard's evidence. The expensive half --
    // that the guard actually fires on a real SCF-derived negative diagonal at
    // scissor=0.0 and stays quiet at scissor=0.36 -- lives in
    // tests/rpax_alpha_diagonal_guard.rs, which runs the full RPAx path.

    /// The guard must stay QUIET on a healthy tensor. Uses the MEASURED
    /// scissor-corrected water/cc-pVDZ diagonal, not a made-up one, so this
    /// pass condition is reachable for the right reason. Off-diagonals are
    /// deliberately nonzero and one is large+negative: an off-diagonal element
    /// of alpha is NOT sign-constrained and must not trip the guard.
    #[test]
    fn alpha_diagonal_guard_quiet_on_healthy_tensor() {
        let t = [
            [7.2461, -0.5, 0.1],
            [-0.5, 9.8842, -3.0],
            [0.1, -3.0, 8.6013],
        ];
        assert!(
            check_alpha_diagonal_positive("unit", &t).is_ok(),
            "guard fired on a physically fine tensor -- an always-firing guard is useless"
        );
    }

    /// The guard must FIRE on the measured water/cc-pVDZ scissor=0.0 diagonal,
    /// and the message must name the axis, the value AND the remedy.
    #[test]
    fn alpha_diagonal_guard_fires_on_measured_negative_water_tensor() {
        // Measured, docs/rpax-negative-diagonal-investigation.md step 1:
        // water/cc-pVDZ/PBE at scissor=0.0 -> diag (-2.6828, +14.5994, +16.4902),
        // iso +9.4690 (which on its own looks like a GOOD answer vs DOSD 9.64 --
        // exactly why a warning is not enough).
        let t = [
            [-2.6828, 0.0, 0.0],
            [0.0, 14.5994, 0.0],
            [0.0, 0.0, 16.4902],
        ];
        let err = check_alpha_diagonal_positive("unit-water", &t)
            .expect_err("guard failed to fire on a measured negative diagonal");
        let msg = err.to_string();
        assert!(msg.contains("unit-water"), "context missing: {msg}");
        assert!(msg.contains("alpha_xx"), "offending axis not named: {msg}");
        assert!(msg.contains("negative"), "not classified negative: {msg}");
        assert!(
            msg.contains("scissor"),
            "message must name the known remedy: {msg}"
        );
        // The healthy axes must NOT be reported as offenders.
        assert!(!msg.contains("alpha_yy"), "false positive on yy: {msg}");
        assert!(!msg.contains("alpha_zz"), "false positive on zz: {msg}");
    }

    /// Exactly-zero and near-zero diagonals are REJECTED too (strict
    /// positivity), but classified distinctly from clearly-negative ones.
    #[test]
    fn alpha_diagonal_guard_rejects_zero_and_near_zero() {
        for v in [0.0, 1e-12, -1e-12] {
            let t = [[v, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]];
            let msg = check_alpha_diagonal_positive("unit-zero", &t)
                .expect_err("near-zero diagonal must be rejected, not accepted")
                .to_string();
            assert!(
                msg.contains("vanishing"),
                "near-zero should be classified 'vanishing', got: {msg}"
            );
        }
        // Clearly below the floor on the negative side is classified "negative".
        let neg = [
            [-1.0 * (ALPHA_DIAGONAL_ZERO_TOL * 10.0), 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 0.0, 9.0],
        ];
        assert!(check_alpha_diagonal_positive("unit", &neg)
            .unwrap_err()
            .to_string()
            .contains("negative"));
        // ALPHA_DIAGONAL_ZERO_TOL is an ACCEPTANCE FLOOR, not just a message
        // split: a tiny POSITIVE diagonal is rejected too. A degenerate solve
        // must not pass merely by landing on the lucky side of zero.
        let tiny_pos = [
            [ALPHA_DIAGONAL_ZERO_TOL * 0.01, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 0.0, 9.0],
        ];
        assert!(
            check_alpha_diagonal_positive("unit", &tiny_pos).is_err(),
            "a +1e-10 a.u. diagonal is a degenerate solve, not a small polarizability"
        );
        // Just ABOVE the floor is accepted -- the floor must be crossable, or
        // the guard would reject everything and prove nothing.
        let just_above = [
            [ALPHA_DIAGONAL_ZERO_TOL * 100.0, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 0.0, 9.0],
        ];
        assert!(
            check_alpha_diagonal_positive("unit", &just_above).is_ok(),
            "values above the acceptance floor must pass"
        );
    }

    /// A NaN diagonal must be caught, not silently pass. `!(x > 0.0)` is used
    /// precisely so NaN is rejected; `x <= 0.0` would let it through.
    #[test]
    fn alpha_diagonal_guard_rejects_nan() {
        let t = [
            [f64::NAN, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 0.0, f64::NEG_INFINITY],
        ];
        let msg = check_alpha_diagonal_positive("unit-nan", &t)
            .expect_err("NaN diagonal must be rejected")
            .to_string();
        assert!(msg.contains("NaN"), "NaN not classified: {msg}");
        assert!(msg.contains("alpha_zz"), "-inf axis not caught: {msg}");
        // The NaN axis must be named EXPLICITLY. Without this, a comparison
        // written as `x <= tol` (which NaN fails, so NaN would slip through)
        // still passes the asserts above via the -inf axis alone -- a mutation
        // that survived exactly once during development.
        assert!(
            msg.contains("alpha_xx"),
            "NaN axis leaked through the comparison: {msg}"
        );
        // And NaN alone, with both other axes healthy, must still be refused.
        let only_nan = [[f64::NAN, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]];
        let m2 = check_alpha_diagonal_positive("unit-nan-only", &only_nan)
            .expect_err("a lone NaN diagonal must be rejected")
            .to_string();
        assert!(m2.contains("alpha_xx") && m2.contains("NaN"), "{m2}");
    }

    /// All three axes bad -> all three reported (not just the first).
    #[test]
    fn alpha_diagonal_guard_reports_every_offending_axis() {
        // Measured LiH/cc-pVDZ/PBE scissor=0.0: (-452.46, -452.46, -14.16).
        let t = [[-452.46, 0.0, 0.0], [0.0, -452.46, 0.0], [0.0, 0.0, -14.16]];
        let msg = check_alpha_diagonal_positive("unit-lih", &t)
            .unwrap_err()
            .to_string();
        for axis in ["alpha_xx", "alpha_yy", "alpha_zz"] {
            assert!(msg.contains(axis), "{axis} not reported: {msg}");
        }
    }

    #[test]
    fn cis_tda_fails_fast_under_tiny_env_budget() {
        // M2 size guard: the dense (ia)×(ia) TDA matrix must ERROR cleanly under
        // a tiny budget, before build_full_b / the a_mat allocation. RHF runs
        // BEFORE the env var is set so only the guarded path sees the tiny budget.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        std::env::set_var("FERRIC_MEM_BUDGET_GB", "0.0000001");
        let res = run_cis_tda(&mol, &obs, &dfbs, op, &rhf, 0);
        std::env::remove_var("FERRIC_MEM_BUDGET_GB");
        let err = res.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("CIS-TDA") && msg.contains("budget is"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn cis_tda_a_matrix_bit_identical_across_thread_counts() {
        // run_cis_tda reads the process-global FERRIC_MEM_BUDGET_GB
        // internally; hold ENV_LOCK so this can't observe
        // cis_tda_fails_fast_under_tiny_env_budget's temporary tiny-budget
        // mutation under cargo test's default parallelism (found 2026-07-18,
        // same class of bug as gto_eval.rs).
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // P4: the row-parallel A-matrix fill (par_chunks_mut over the flat
        // `ia` axis) must be bit-identical regardless of RAYON_NUM_THREADS —
        // each row is written by exactly one worker, no reduction, so thread
        // count must not perturb a single f64 bit. CIS-TDA is the cheapest
        // path that exercises the row-fill (no GW needed). H2/cc-pVDZ gives
        // nocc=1, nvir=9 → n=9 rows, above PAR_ROWS_THRESHOLD=8 so the
        // parallel branch actually runs at 4 threads.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

        let run_with_threads = |n: usize| -> BseResult {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap();
            pool.install(|| run_cis_tda(&mol, &obs, &dfbs, op, &rhf, 0).unwrap())
        };

        let r1 = run_with_threads(1);
        let r4 = run_with_threads(4);

        assert_eq!(r1.nocc, r4.nocc);
        assert_eq!(r1.nvir, r4.nvir);
        assert!(
            r1.nocc * r1.nvir >= 8,
            "test must exercise the parallel branch (n>=8), got n={}",
            r1.nocc * r1.nvir
        );
        assert_eq!(r1.omega.len(), r4.omega.len());
        for (k, (a, b)) in r1.omega.iter().zip(r4.omega.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "CIS-TDA eigenvalue {k} not bit-identical across thread counts: \
                 1-thread={a:.17e} (0x{:016x}), 4-thread={b:.17e} (0x{:016x})",
                a.to_bits(),
                b.to_bits(),
            );
        }
    }
}
