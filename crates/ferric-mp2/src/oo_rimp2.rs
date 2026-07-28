//! Orbital-optimized RI-MP2 (OO-RI-MP2).
//!
//! Minimizes E_HF + E_MP2 jointly by optimizing orbital rotation parameters
//! using a level-shifted approximate Newton step with DIIS extrapolation
//! and Cayley orbital rotations.
//!
//! The level-shifted diagonal Hessian uses orbital energy differences as the
//! approximate Hessian: kappa_{ai} = -g_{ai} / (eps_a - eps_i + mu), following
//! Bozkaya & Sherrill, JCP 135, 104103 (2011). DIIS (Pulay extrapolation)
//! accelerates convergence of the orbital rotation parameters.
//!
//! The analytic orbital gradient uses the Hylleraas functional derivative,
//! which includes both the 1-PDM/Fock terms and the 2-electron integral
//! response terms from the MO integral derivatives.

use crate::orbital_rotation::cayley_rotation;
use crate::rimp2::{active_occ, cholesky_inverse_sqrt};
use ferric_core::mol::Molecule;
use ferric_core::orbitals::OrbitalSpace;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex;
use ferric_scf::diis::Diis;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::rhf::build_jk_with_pool;
use ferric_scf::ScfResult;
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array2, Array3};
use std::cell::RefCell;

/// Configuration for OO-RI-MP2.
#[derive(Debug, Clone)]
pub struct OoRiMp2Config {
    pub max_iter: usize,
    pub grad_conv: f64,
    pub energy_conv: f64,
    pub step_size: f64,
    pub frozen_core: usize,
    /// Level shift for the approximate diagonal Hessian (Ha).
    /// Regularizes the Newton step when orbital energy gaps are small.
    pub level_shift: f64,
    /// Maximum DIIS subspace size for orbital rotation extrapolation.
    pub diis_size: usize,
    /// Whether to use DIIS for orbital rotations.
    pub use_diis: bool,
    /// Optional resident-bytes ceiling for the 3-index MO transform. `None` →
    /// resolved via [`ferric_core::memory::resolve_budget_bytes`].
    pub memory_budget_bytes: Option<usize>,
    /// Print one line per orbital-optimization iteration to stdout while the
    /// job runs (HF/MP2/total energy, gradient norm) — live progress for a
    /// long-running job, opt-in and additive. Default `false` (unchanged,
    /// silent-until-done output). Mirrors `ferric_scf::rhf::RhfConfig::verbose`;
    /// the CLI's `--verbose`/`-v` flag ORs into this the same way it does for
    /// `RhfConfig.verbose`.
    pub verbose: bool,
}

impl Default for OoRiMp2Config {
    fn default() -> Self {
        Self {
            max_iter: 100,
            grad_conv: 1e-4,
            energy_conv: 1e-8,
            step_size: 0.5,
            frozen_core: 0,
            level_shift: 0.1,
            diis_size: 6,
            use_diis: true,
            memory_budget_bytes: None,
            verbose: false,
        }
    }
}

/// Result from OO-RI-MP2.
#[derive(Debug)]
pub struct OoRiMp2Result {
    /// Total energy: E_HF(optimized) + E_MP2(optimized).
    pub total_energy: f64,
    /// Re-optimized HF energy component.
    pub hf_energy: f64,
    /// MP2 correlation energy with optimized orbitals.
    pub mp2_corr: f64,
    /// Whether gradient and energy convergence thresholds were met.
    pub converged: bool,
    /// Number of orbital optimization iterations.
    pub iterations: usize,
    /// Final orbital gradient norm.
    pub grad_norm: f64,
    /// Optimized MO coefficients.
    pub mos: Array2<f64>,
    /// Orbital energies from the optimized Fock matrix.
    pub orbital_energies: Vec<f64>,
}

/// AO-side invariants for OO-RI-MP2, built once and reused across every
/// orbital-rotation iteration. These depend only on `(obs, dfbs, op)` — not on
/// the MO coefficients — so rebuilding them per iteration (and per line-search
/// backtrack) was pure waste.
///
/// The `(naux, nao, nao)` AO 3-index tensor is served through a
/// memory-budgeted [`ThreeIndexSource`] (`FERRIC_ERI3_BUDGET_GB`): in-core when
/// it fits the budget (identical to the old resident `Array3`), disk-spilled in
/// aux-blocks when it does not. Consumers pull raw aux-blocks via
/// `for_each_block` and dress each block with `V^{-1/2}` on the fly, so the peak
/// resident 3-index footprint is one aux-block, not the full tensor.
///
/// `RefCell` gives the `for_each_block` iterator the `&mut` it needs (disk seek
/// + scratch reuse) while `OoRiMp2AoTensors` is shared as `&self` across the
/// hot orbital-optimization loop. Borrows are non-overlapping (each transform
/// takes the borrow, streams, drops it), so no runtime borrow conflict arises.
pub struct OoRiMp2AoTensors {
    /// V^{-1/2}, shape (naux, naux).
    pub v2c_inv_sqrt: Array2<f64>,
    /// Budget-aware raw AO 3-center integral source (P|mu nu), (naux, nao, nao).
    pub eri3_ao: RefCell<ThreeIndexSource>,
    naux: usize,
    nao: usize,
}

impl OoRiMp2AoTensors {
    /// Build the AO-side invariants once, budget from `FERRIC_ERI3_BUDGET_GB`.
    pub fn build(
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
    ) -> Result<Self, FerricError> {
        Self::build_with_budget(obs, dfbs, op, ferric_core::memory::resolve_budget_bytes(None))
    }

    /// Build with an explicit resident-bytes budget for the raw 3-index tensor.
    pub fn build_with_budget(
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
        let v2c_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
        let src = ThreeIndexSource::build(op, obs, dfbs, budget_bytes)?;
        let naux = src.naux();
        let nao = src.nao();
        Ok(Self { v2c_inv_sqrt, eri3_ao: RefCell::new(src), naux, nao })
    }

    /// Number of auxiliary basis functions (rows of the 3-index tensor).
    pub fn naux(&self) -> usize {
        self.naux
    }
    /// Number of AO basis functions.
    pub fn nao(&self) -> usize {
        self.nao
    }
}

/// Compute the full-MO 3-center B tensor: B^P_{pq} for all MO pairs p,q.
///
/// Returns b_full of shape (naux, nmo, nmo) where:
///   b_full[(P, p, q)] = sum_Q V^{-1/2}_{PQ} sum_{mu,nu} (Q|mu nu) C_{mu,p} C_{nu,q}
///
/// AO-side objects are rebuilt each call; prefer [`compute_b_full_mo_with`] in
/// hot loops where the AO invariants are hoisted.
pub fn compute_b_full_mo(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    c: &Array2<f64>,
) -> Result<Array3<f64>, FerricError> {
    let ao = OoRiMp2AoTensors::build(obs, dfbs, op)?;
    compute_b_full_mo_with(&ao, c)
}

/// Full-MO 3-center B tensor from pre-built AO invariants.
///
/// Delegates to the shared canonical streamer
/// [`crate::rimp2::stream_dressed_mo_band`] with `c_left = c_right = c` (the
/// full-MO square this function has always computed) and no output-band
/// restriction. That function is the chunked-aux-streaming + rayon-parallel
/// MO transform this function originated (see its doc for the memory/exactness
/// contract: one `(naux, nmo, nmo)` output tensor, transient bounded to one
/// `(≤256, nmo²)` panel, dressed in place with no second full-size copy).
///
/// Exactness: `b_full[P,p,q] = Σ_Q V^{-1/2}[P,Q] · (C^T (Q|μν) C)[p,q]`, the
/// same contraction as before — reordered, not approximated.
pub fn compute_b_full_mo_with(
    ao: &OoRiMp2AoTensors,
    c: &Array2<f64>,
) -> Result<Array3<f64>, FerricError> {
    let naux = ao.naux();
    let nmo = c.ncols();
    let b_flat = crate::rimp2::stream_dressed_mo_band(
        &mut ao.eri3_ao.borrow_mut(),
        &ao.v2c_inv_sqrt,
        c,
        c,
        None,
    )?;
    Ok(b_flat
        .into_shape_with_order((naux, nmo, nmo))
        .map_err(|e| FerricError::General(format!("compute_b_full_mo_with reshape: {e}")))?)
}

/// Compute the RI-MP2 energy for a given set of MO coefficients.
///
/// Returns (e_mp2, b_ov_flat) where b_ov_flat is (naux, nocc*nvir) for reuse.
/// Uses pre-built AO invariants (see [`OoRiMp2AoTensors`]); only the MO
/// transform + fitting contraction depend on `c`.
fn compute_rimp2_with_orbitals(
    ao: &OoRiMp2AoTensors,
    c: &Array2<f64>,
    eps: &[f64],
    orb: &OrbitalSpace,
) -> Result<(f64, Array2<f64>), FerricError> {
    let OrbitalSpace { nocc, nocc_total, first_occ, nvir } = *orb;

    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // MO transform (P|μν) -> (P|ia) and dress with V^{-1/2} on the fly, via the
    // shared canonical streamer (see `compute_b_full_mo_with`'s doc):
    //   b_flat[P, ia] = Σ_Q V^{-1/2}[P,Q] (C_occ^T (Q|μν) C_vir)[ia]
    let b_flat = crate::rimp2::stream_dressed_mo_band(
        &mut ao.eri3_ao.borrow_mut(),
        &ao.v2c_inv_sqrt,
        &c_occ,
        &c_vir,
        None,
    )?;

    // MP2 energy via i-blocked wide GEMMs (same path as the main RI-MP2 lane).
    let sc = crate::rimp2::spin_components_from_b_ov(
        &b_flat, eps, nocc, nvir, first_occ, nocc_total,
    );
    Ok((sc.e_total, b_flat))
}

/// Build the HF energy from MO coefficients + 1e integrals + J/K.
///
/// `ooc_budget` is the caller's solver-resolved memory budget (see
/// `rhf::resolve_three_index_budget`) — used ONLY to size the
/// `build_jk_with_pool` reduction band (`reduce::resolve_band_bytes`), never
/// affecting the result.
#[allow(clippy::too_many_arguments)]
fn compute_hf_energy(
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    c: &Array2<f64>,
    nocc_total: usize,
    h: &Array2<f64>,
    pool: &EnginePool,
    ooc_budget: usize,
) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
    let n = prep.nbasis();

    // Build density: D = 2 * C_occ C_occ^T
    let mut d = Array2::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for i in 0..nocc_total {
                sum += c[(mu, i)] * c[(nu, i)];
            }
            d[(mu, nu)] = 2.0 * sum;
        }
    }

    // Build J, K
    let mut j_mat = Array2::zeros((n, n));
    let mut k_mat = Array2::zeros((n, n));
    let ctx = ferric_core::parallel::ParallelContext::default();
    let band_bytes = ferric_scf::reduce::resolve_band_bytes(ooc_budget);
    build_jk_with_pool(&ctx, prep, bounds, 1e-12, &d, &mut j_mat, &mut k_mat, pool, band_bytes)?;

    // F = H + J - 0.5*K
    let f = h + &j_mat - &(0.5 * &k_mat);

    // E_elec = 0.5 * tr(D * (H + F))
    let hpf = h + &f;
    let e_elec: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| 0.5 * d[(i, j)] * hpf[(i, j)])
        .sum();
    let vnn = mol.nuclear_repulsion();
    let e_hf = e_elec + vnn;

    Ok((e_hf, f, d))
}

/// Compute orbital energies as diagonal of C^T F C.
///
/// # Why `diag` is legitimate HERE (audited 2026-07-28)
///
/// These values are used as MP2 DENOMINATORS, not merely as a Hessian
/// preconditioner, and orbital optimization rotates away from the canonical
/// basis by construction — so this is superficially the same shape as two real
/// bugs found elsewhere in this workspace (the Boys-screened `eps_loc` defect in
/// `ferric-rpa/src/screen.rs`, fixed 17e994e, and the AO-Laplace pseudo-density
/// bug, fixed 3693d5d): a canonical-style eigenvalue read off a rotated basis.
///
/// It was therefore MEASURED rather than assumed. At convergence the OO basis is
/// very nearly Fock-diagonal, so the discarded coupling is negligible —
/// water/cc-pVDZ (nocc = 5): max|F_oo offdiag| / diag-spread = 4.5e-5,
/// max|F_vv offdiag| / spread = 3.8e-5. Compare the Boys-localized case, where
/// the same ratio was 1.3-2.9e-2 — roughly 650x larger, and a genuine defect.
///
/// The occ-VIR block is deliberately excluded from that comparison: it is the
/// orbital gradient, which OO drives toward zero and which does not enter the
/// denominators.
///
/// Pinned by `oo_converged_basis_offdiagonal_fock_is_measured_not_assumed`,
/// which FAILS if either ratio exceeds 10% — i.e. if a future change makes the
/// converged basis meaningfully non-diagonal, this becomes a real defect and the
/// fix is semicanonicalization (re-diagonalize the occ and vir blocks), exactly
/// as done in `screen.rs`.
fn orbital_energies(c: &Array2<f64>, f: &Array2<f64>) -> Vec<f64> {
    let n = c.ncols();
    let f_mo = c.t().dot(f).dot(c);
    (0..n).map(|i| f_mo[(i, i)]).collect()
}

/// Below this many (ia,jb) pairs the elementwise denominator-divide pass
/// after the `eri_ov` GEMM runs serially (rayon dispatch overhead beats the
/// win on tiny jobs). Pure function of `nov²`, never of thread count — same
/// discipline as `PAR_2E_QUARTET_THRESHOLD` (ferric-scf/src/gradient.rs) and
/// `PAR_DENSITY_PAIRS_THRESHOLD` (ferric-scf/src/pairs.rs).
const PAR_T2_ELEMENT_THRESHOLD: usize = 4096;

/// Fail-fast pre-flight guard for the `(nov, nov)` t2/eri_ov_mat pair built by
/// [`compute_t2_and_integrals`] / [`compute_t2_only`] every OO-MP2
/// macro-iteration.
///
/// `compute_t2_and_integrals` returns (and its caller retains) BOTH the
/// `eri_ov_mat` GEMM output and the `t2` buffer — two co-resident `nov²` f64
/// arrays, not one. `compute_t2_only` only *retains* one (`eri_ov_mat` is
/// local scratch dropped at the end of that call — see its doc comment) but
/// still touches both at its momentary in-call peak, so the same two-buffer
/// estimate is the safe (never-under-count) bound for both callers. At
/// nocc≈60/nvir≈900 (nov≈54,000) this is `nov²·8·2 ≈ 46.6 GB` — the exact gap
/// this guard closes (previously zero pre-flight check on this allocation,
/// unlike the AO-tensor stage's `build_with_budget`).
///
/// `label` should identify the call site (e.g. "OO-RI-MP2 t2/eri_ov (iter
/// N)"); `budget_bytes` is the already-resolved [`ferric_core::memory::resolve_budget_bytes`]
/// value (callers resolve once per `oo_ri_mp2` invocation, not per iteration).
fn check_t2_pair_alloc(
    label: &str,
    nocc: usize,
    nvir: usize,
    budget_bytes: usize,
) -> Result<(), FerricError> {
    let nov = nocc * nvir;
    let peak = nov.saturating_mul(nov).saturating_mul(2).saturating_mul(8);
    ferric_core::memory::check_alloc(
        &format!("{label} (nocc={nocc}, nvir={nvir}; co-resident t2+eri_ov_mat (nov={nov})²)"),
        peak,
        budget_bytes,
    )
}

/// Fail-fast pre-flight guard for the dense MO-space intermediates
/// [`compute_orbital_gradient_panelled`] allocates unconditionally BEFORE its
/// c-panel loop: `b_ov` (naux×nov), `b_vv` (naux×nvir²), `b_oo` (naux×nocc²),
/// `ooov` (nocc²×nov), `ovov` (nov×nov), `s_mat` (nov×nov), and `b_diag`
/// (naux×nmo) — all f64 and all co-resident. `s_mat` alone is ~23 GB at
/// nocc≈60/nvir≈900, and unlike the VVOV transient (which the panel loop
/// budget-sizes) none of these were previously covered by any pre-flight
/// check (the co-resident t2 pair in the caller is guarded by
/// [`check_t2_pair_alloc`]; this closes the sibling gap).
///
/// `budget_bytes` is the already-resolved [`ferric_core::memory::resolve_budget_bytes`]
/// value, threaded down from `oo_ri_mp2`'s once-per-call resolution — same
/// convention as [`check_t2_pair_alloc`]. Pure arithmetic: never allocates.
fn check_gradient_intermediates_alloc(
    label: &str,
    nocc: usize,
    nvir: usize,
    naux: usize,
    nmo: usize,
    budget_bytes: usize,
) -> Result<(), FerricError> {
    let nov = nocc.saturating_mul(nvir);
    // Element counts of the pre-panel-loop co-resident buffers.
    let aux_elems = naux.saturating_mul(
        nov.saturating_add(nvir.saturating_mul(nvir))
            .saturating_add(nocc.saturating_mul(nocc))
            .saturating_add(nmo),
    ); // b_ov + b_vv + b_oo + b_diag
    let mo_elems = nocc
        .saturating_mul(nocc)
        .saturating_mul(nov) // ooov
        .saturating_add(nov.saturating_mul(nov).saturating_mul(2)); // ovov + s_mat
    let peak = aux_elems.saturating_add(mo_elems).saturating_mul(8);
    ferric_core::memory::check_alloc(
        &format!("{label} (nocc={nocc}, nvir={nvir}, naux={naux}; co-resident b_ov/b_vv/b_oo/b_diag + ooov + ovov + s_mat)"),
        peak,
        budget_bytes,
    )
}

/// Compute t2 amplitudes and (ia|jb) integrals from B tensor.
///
/// t2 is stored as flat vec of length (nocc*nvir)^2 with indexing t2[ia*nov + jb]
/// where ia = i*nvir + a, jb = j*nvir + b.
///
/// Returns (t2, eri_ov) where eri_ov[ia*nov + jb] = (ia|jb).
///
/// `eri_iajb = Σ_p b_flat[p,ia]·b_flat[p,jb]` is exactly the (ia,jb) entry of
/// `b_flat^T @ b_flat` — computed here as one wide `(nov × naux) @ (naux ×
/// nov)` GEMM instead of the former per-element scalar strided dot product
/// (the BLAS3-hostile anti-pattern: an O(nov²) loop of O(naux) scalar dots).
/// This call site is not inside a rayon region (see `oo_ri_mp2`'s main loop
/// and callers in rimp2.rs/oo_rimp2_gradient.rs), so the GEMM runs at the
/// ambient BLAS thread count.
pub fn compute_t2_and_integrals(
    b_flat: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
    first_occ: usize,
    _naux: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nov = nocc * nvir;
    let eri_ov_mat = b_flat.t().dot(b_flat); // (nov, nov), GEMM: eri_ov_mat[ia,jb] = Σ_p b[p,ia]·b[p,jb]

    let mut t2 = vec![0.0f64; nov * nov];
    // `.dot()`'s output memory layout is not guaranteed C-order (it can go
    // F-order in degenerate shapes — see ndarray-dot-forder-when-both-stride1
    // memory pitfall), and the public contract here is a flat C-order Vec
    // (`eri_ov[ia*nov+jb]`). `as_standard_layout` forces a C-contiguous copy
    // if needed (a no-op if already C-order) before flattening, so the raw
    // Vec extraction below is always correctly ordered regardless of what
    // `.dot()` chose internally.
    let eri_ov = eri_ov_mat
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset()
        .0;

    let fill_row = |ia: usize, t2_row: &mut [f64]| {
        let i = ia / nvir;
        let a = ia % nvir;
        let base = ia * nov;
        for j in 0..nocc {
            for b in 0..nvir {
                let jb = j * nvir + b;
                let denom = eps[first_occ + i] + eps[first_occ + j]
                    - eps[nocc_total + a]
                    - eps[nocc_total + b];
                t2_row[jb] = eri_ov[base + jb] / denom;
            }
        }
    };

    if nov * nov < PAR_T2_ELEMENT_THRESHOLD {
        for ia in 0..nov {
            fill_row(ia, &mut t2[ia * nov..(ia + 1) * nov]);
        }
    } else {
        use rayon::prelude::*;
        t2.par_chunks_mut(nov)
            .enumerate()
            .for_each(|(ia, row)| fill_row(ia, row));
    }

    (t2, eri_ov)
}

/// Compute only the t2 amplitudes from the B tensor, without materializing the
/// (ia|jb) integral array.
///
/// Identical numerics to [`compute_t2_and_integrals`] for its first return
/// value, via the same wide-GEMM restructure (`b_flat^T @ b_flat`). The GEMM
/// output (`eri_ov_mat`, one `nov²` buffer) is local scratch: it is read
/// row-by-row to fill `t2` and dropped at the end of this call, never
/// returned or retained by the caller. Peak transient footprint *inside this
/// function* is momentarily ~2×`nov²` (`eri_ov_mat` + `t2` both resident
/// during the fold) — up from the former scalar loop's ~1×`nov²` (`t2`
/// alone, no GEMM buffer) — but the important axis for callers is *retained*
/// memory after the call returns: [`compute_t2_and_integrals`] hands back and
/// the caller keeps two live `nov²` buffers for the rest of its scope, while
/// this function's caller keeps exactly one (`t2`); `eri_ov_mat` never
/// escapes. Callers that discard the integrals (e.g. OSV/PNO construction in
/// ferric-rpa) should still prefer this function for that reason — it is the
/// post-call retained footprint, not the momentary in-call peak, that halves.
/// Indexing matches: t2[ia*nov + jb], ia = i*nvir + a.
///
/// `memory_budget_bytes` gates the momentary co-resident `t2`+`eri_ov_mat`
/// pair (see [`check_t2_pair_alloc`]) — `None` resolves via
/// [`ferric_core::memory::resolve_budget_bytes`], matching every other
/// budget-aware entry point in this crate. This function currently has no
/// callers outside its own test (see the doc comment above for the intended
/// ferric-rpa OSV/PNO use), so its signature is free to carry the guard
/// directly rather than pushing it to a call site.
pub fn compute_t2_only(
    b_flat: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
    first_occ: usize,
    _naux: usize,
    memory_budget_bytes: Option<usize>,
) -> Result<Vec<f64>, FerricError> {
    check_t2_pair_alloc(
        "OO-RI-MP2 compute_t2_only",
        nocc,
        nvir,
        ferric_core::memory::resolve_budget_bytes(memory_budget_bytes),
    )?;
    let nov = nocc * nvir;
    let eri_ov_mat = b_flat.t().dot(b_flat); // (nov, nov) transient, dropped at end of scope

    let mut t2 = vec![0.0f64; nov * nov];

    let fill_row = |ia: usize, t2_row: &mut [f64]| {
        let i = ia / nvir;
        let a = ia % nvir;
        let eri_row = eri_ov_mat.row(ia);
        for j in 0..nocc {
            for b in 0..nvir {
                let jb = j * nvir + b;
                let denom = eps[first_occ + i] + eps[first_occ + j]
                    - eps[nocc_total + a]
                    - eps[nocc_total + b];
                t2_row[jb] = eri_row[jb] / denom;
            }
        }
    };

    if nov * nov < PAR_T2_ELEMENT_THRESHOLD {
        for ia in 0..nov {
            fill_row(ia, &mut t2[ia * nov..(ia + 1) * nov]);
        }
    } else {
        use rayon::prelude::*;
        t2.par_chunks_mut(nov)
            .enumerate()
            .for_each(|(ia, row)| fill_row(ia, row));
    }

    Ok(t2)
}

/// Build the full relaxed 1-PDM for OO-MP2 in MO basis.
///
/// For OO-MP2, the density is already "relaxed" because it is a stationary
/// point w.r.t. orbital rotations. The MO-basis density is:
///   P_pq = delta_pq (for occ) + P^MP2_pq
pub fn build_oo_mp2_relaxed_density(
    t2: &[f64],
    nocc: usize,
    nvir: usize,
    nmo: usize,
    first_occ: usize,
) -> Array2<f64> {
    let (p_oo, p_vv) = build_mp2_density(t2, nocc, nvir);
    let mut p = Array2::zeros((nmo, nmo));
    
    // HF occupied part
    for i in 0..nocc {
        let idx = first_occ + i;
        p[(idx, idx)] = 2.0;
    }
    
    // MP2 correction
    for i in 0..nocc {
        for j in 0..nocc {
            p[(first_occ + i, first_occ + j)] += p_oo[(i, j)];
        }
    }
    for a in 0..nvir {
        for b in 0..nvir {
            let nocc_total = nmo - nvir;
            p[(nocc_total + a, nocc_total + b)] += p_vv[(a, b)];
        }
    }
    p
}

/// Build the MP2 unrelaxed 1-particle density matrix in MO basis.
pub fn build_mp2_density(
    t2: &[f64],
    nocc: usize,
    nvir: usize,
) -> (Array2<f64>, Array2<f64>) {
    let nov = nocc * nvir;

    // P^MP2_ij = -sum_{kab} t_{ik,ab} (2 t_{jk,ab} - t_{jk,ba})
    let mut p_oo = Array2::zeros((nocc, nocc));
    for i in 0..nocc {
        for j in 0..nocc {
            let mut sum = 0.0;
            for k in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let ik_ab = (i * nvir + a) * nov + k * nvir + b;
                        let jk_ab = (j * nvir + a) * nov + k * nvir + b;
                        let jk_ba = (j * nvir + b) * nov + k * nvir + a;
                        sum += t2[ik_ab] * (2.0 * t2[jk_ab] - t2[jk_ba]);
                    }
                }
            }
            p_oo[(i, j)] = -sum;
        }
    }

    // P^MP2_ab = sum_{ijc} t_{ij,ac} (2 t_{ij,bc} - t_{ij,cb})
    let mut p_vv = Array2::zeros((nvir, nvir));
    for a in 0..nvir {
        for b in 0..nvir {
            let mut sum = 0.0;
            for i in 0..nocc {
                for j in 0..nocc {
                    for c in 0..nvir {
                        let ij_ac = (i * nvir + a) * nov + j * nvir + c;
                        let ij_bc = (i * nvir + b) * nov + j * nvir + c;
                        let ij_cb = (i * nvir + c) * nov + j * nvir + b;
                        sum += t2[ij_ac] * (2.0 * t2[ij_bc] - t2[ij_cb]);
                    }
                }
            }
            p_vv[(a, b)] = sum;
        }
    }

    (p_oo, p_vv)
}

/// Compute the full OO-MP2 orbital gradient g_{ai} for occupied-virtual rotations.
///
/// This implements the Hylleraas-style functional derivative, which includes:
/// 1. The 1-PDM/Fock (HF) term.
/// 2. The 2-electron integral response terms (response of (ia|jb) and (ib|ja)
///    to the orbital rotation, at FIXED t2 amplitudes).
/// 3. The orbital-energy-DENOMINATOR response term (response of D_{ijab} in
///    t_{ij,ab} = (ia|jb)/D_{ijab} to the orbital rotation).
///
/// FOUND 2026-07-20 (see docs/VALIDATION.md "OO-RI-MP2" row): term 3 above was
/// previously believed to vanish via "Brillouin's condition makes
/// d(eps_p)/d(kappa_ck) = 0" — that claim is WRONG. d(eps_p)/d(kappa_ck) is
/// generally nonzero (it comes entirely from the density-dependence of the
/// Fock matrix under the rotated occupied space, not from any frozen-Fock
/// rotation of F itself — the latter piece IS exactly zero at a converged HF
/// reference since F_{ck}=0 there, which is presumably how the false belief
/// arose). The MP2 integral-response contraction (below, "Term1..Term4",
/// implementing term 2 above) was independently re-derived from scratch via
/// multilinear MO-coefficient perturbation (verified against finite
/// difference on random tensors AND on real PySCF integrals, including every
/// simultaneous-delta-condition combination e.g. i=k AND a=c) and found to
/// ALREADY be correct in the code below — no sign or structural change to it
/// was needed. The single missing piece was term 3.
///
/// SIGN CONVENTION (unified 2026-07-22): this function returns g = +∂E/∂κ under
/// the CANONICAL Cayley rotation `U = (I−κ/2)⁻¹(I+κ/2) ≈ exp(κ)` (shared
/// `orbital_rotation::cayley_rotation`, same as `u_oo_rimp2`). The formula
/// below is written in the PRE-UNIFICATION Cayley sense `U = (I+κ/2)⁻¹(I−κ/2) ≈
/// exp(−κ)`; because those two rotations are inverses, the value this function
/// actually returns is the NEGATIVE of the formula as written — the HF term is
/// coded as `+4·F_{ck}` and the MP2 block as `+2·grad_ck − denom_term`. The
/// finite-difference validation tests (which drive `energy_at_kappa` through
/// the same shared Cayley) therefore compare against the returned +∂E/∂κ and
/// pass unchanged.
///
/// Full formula in the PRE-UNIFICATION Cayley sense (c=virtual, k=occupied index
/// of the rotation kappa_{ck}); NEGATE for what the code returns (see above):
///
///   dE_total/d kappa_{ck} = -4*F_{ck}  (HF part, "Term0" below)
///     + 2 * sum_{ijab} t_{ij,ab} * [2*d(ia|jb)/dk - d(ib|ja)/dk]   ("Term1..Term4")
///     - sum_{ijab} t_{ij,ab} * [dD_{ijab}/dk] * s_{ij,ab}          ("denominator response")
///
/// where s_{ij,ab} = [2*(ia|jb) - (ib|ja)] / D_{ijab}, and the integral
/// response (multilinearity in each of the 4 MO-coefficient slots of a
/// Cayley/exponential single-parameter rotation, dC_p/dk_{ck} =
/// -delta_{pk}*C_c + delta_{pc}*C_k) gives:
///
///   d(ia|jb)/d kappa_{ck} = delta_{ik}*(ca|jb) + delta_{jk}*(ia|cb)
///     - delta_{ac}*(ik|jb) - delta_{bc}*(ia|jk)
///
/// The denominator response needs d(eps_p)/d kappa_{ck} for p in {i,j,a,b}.
/// This is NOT a CPHF/Z-vector quantity — kappa_{ck} enters explicitly (not
/// implicitly) in eps_p(kappa) = [C(kappa)^T F(kappa) C(kappa)]_{pp}, so a
/// single closed-form density-response Fock build suffices:
///
///   d(eps_p)/d kappa_{ck} = -4*(pp|ck) + 2*(pc|pk)
///
/// (derived from dD_AO/dk_{ck} = -2*(C_c C_k^T + C_k C_c^T), i.e. the rank-2
/// AO density increment from rotating occupied k into virtual c, contracted
/// through the RHF veff operator 2J-K; verified against finite difference of
/// the true SCF-density-dependent eps_p(kappa) to ~1e-6..1e-8 on H2/cc-pVDZ
/// and H2O/{STO-3G,cc-pVDZ}). Then dD_{ijab}/dk = deps_i/dk + deps_j/dk -
/// deps_a/dk - deps_b/dk.
///
/// End-to-end verification (Python/PySCF reference, independent of this Rust
/// code, cross-validated against pyscf.mp.MP2 to ~1e-9 Ha before use): the
/// full formula above matches central finite difference of the true
/// (HF+MP2, t2-resolved-at-each-kappa) energy to max errors of 1.6e-11
/// (H2/cc-pVDZ), 4.8e-8 (H2O/STO-3G), 4.1e-8 (H2O/cc-pVDZ), and 1.5e-9
/// (NH3/STO-3G) — vs. the pre-fix formula's 3.4e-4 / 4.2e-3 / 2.2e-2 / 2.3e-3
/// respectively. See docs/VALIDATION.md for the corresponding Rust-side
/// numbers after porting.
// Orbital-space sizes plus the budget are all irreducibly distinct inputs.
#[allow(clippy::too_many_arguments)]
fn compute_orbital_gradient(
    f_mo: &Array2<f64>,
    t2: &[f64],
    b_full: &Array3<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    budget_bytes: usize,
) -> Result<Array2<f64>, FerricError> {
    // `budget_bytes` is the caller's ALREADY-RESOLVED budget (oo_ri_mp2
    // resolves resolve_budget_bytes(config.memory_budget_bytes) once per call
    // and threads it here — which for a None-config equals the old
    // resolve_budget_bytes(None) this function used to resolve locally, so
    // env-only/auto-detect runs keep the exact same panel width as before;
    // the change is that an explicit config.memory_budget_bytes is no longer
    // silently discarded).
    let naux = b_full.shape()[0];
    let nmo = nocc_total + nvir;
    // Fail-fast guard on the dense pre-panel-loop intermediates the panelled
    // evaluation allocates unconditionally (mirrors check_t2_pair_alloc on
    // the caller's co-resident t2 pair).
    check_gradient_intermediates_alloc(
        "OO-RI-MP2 gradient intermediates",
        nocc,
        nvir,
        naux,
        nmo,
        budget_bytes,
    )?;
    // VVOV panel width from the resident-bytes budget: one c-value costs
    // nvir·nocc·nvir·8 bytes of VVOV rows. Unset budget = one full-width panel
    // (bit-identical to the former unblocked path).
    let nov = nocc * nvir;
    let row_bytes = nvir.saturating_mul(nov).saturating_mul(8).max(1);
    let panel_c = (budget_bytes / row_bytes).max(1).min(nvir.max(1));
    Ok(compute_orbital_gradient_panelled(
        f_mo, t2, b_full, eps, nocc, nvir, first_occ, nocc_total, panel_c,
    ))
}

/// [`compute_orbital_gradient`] with an explicit VVOV c-panel width. The
/// panelled evaluation is exact for any `panel_c >= 1` — panels change memory
/// shape only, never the contraction. Split out so tests can force multi-panel
/// execution regardless of the environment budget.
// Orbital-space sizes plus the panel width are all irreducibly distinct.
#[allow(clippy::too_many_arguments)]
fn compute_orbital_gradient_panelled(
    f_mo: &Array2<f64>,
    t2: &[f64],
    b_full: &Array3<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    panel_c: usize,
) -> Array2<f64> {
    use rayon::prelude::*;

    let naux = b_full.shape()[0];
    let nov = nocc * nvir;

    // Dressed B blocks (contiguous, aux-major) sliced out of the full-MO tensor.
    //   Bov[P, i*nvir+a] = B^P_{i,a}   (occ, vir)
    //   Bvv[P, a*nvir+b] = B^P_{a,b}   (vir, vir)
    //   Boo[P, i*nocc+j] = B^P_{i,j}   (occ, occ)
    // Building the dense MO-ERI blocks below is then a single wide GEMM each,
    // replacing the former per-element O(naux) dot inside the ijab loops
    // (which cost O(naux · nvir³ nocc³)).
    let mut b_ov = Array2::<f64>::zeros((naux, nov));
    let mut b_vv = Array2::<f64>::zeros((naux, nvir * nvir));
    let mut b_oo = Array2::<f64>::zeros((naux, nocc * nocc));
    for p in 0..naux {
        for i in 0..nocc {
            let i_mo = first_occ + i;
            for a in 0..nvir {
                b_ov[(p, i * nvir + a)] = b_full[(p, i_mo, nocc_total + a)];
            }
            for j in 0..nocc {
                b_oo[(p, i * nocc + j)] = b_full[(p, i_mo, first_occ + j)];
            }
        }
        for a in 0..nvir {
            let a_mo = nocc_total + a;
            for b in 0..nvir {
                b_vv[(p, a * nvir + b)] = b_full[(p, a_mo, nocc_total + b)];
            }
        }
    }

    // OOOV is small (nocc²·nov·8 ≈ 0.44 GB at the audit scale) — build once.
    //   OOOV[(i*nocc+k), (j*nvir+b)] = (ik|jb)
    let ooov = b_oo.t().dot(&b_ov); // (nocc², nocc·nvir)

    // s_{ij,ab} = [2*(ia|jb) - (ib|ja)] / D_{ijab}, needed by the denominator-
    // response term below. Built once (nov×nov, same shape/cost as t2) via the
    // same wide GEMM the OV-OV block already needs.
    //   OVOV[(i*nvir+a), (j*nvir+b)] = (ia|jb)
    let ovov = b_ov.t().dot(&b_ov); // (nov, nov)
    let nmo = nocc_total + nvir; // == b_full's MO dimension (frozen core excluded from nocc)
    let mut s_mat = Array2::<f64>::zeros((nov, nov));
    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            for j in 0..nocc {
                for b in 0..nvir {
                    let jb = j * nvir + b;
                    let iajb = ovov[(ia, jb)];
                    let ibja = ovov[(i * nvir + b, j * nvir + a)];
                    let denom = eps[first_occ + i] + eps[first_occ + j]
                        - eps[nocc_total + a]
                        - eps[nocc_total + b];
                    s_mat[(ia, jb)] = (2.0 * iajb - ibja) / denom;
                }
            }
        }
    }

    // b_diag[P, p] = b_full[P, p, p], needed for the (pp|ck)-type integrals in
    // the denominator response's d(eps_p)/d kappa_{ck} = -4*(pp|ck) + 2*(pc|pk).
    let mut b_diag = Array2::<f64>::zeros((naux, nmo));
    for p in 0..naux {
        for m in 0..nmo {
            b_diag[(p, m)] = b_full[(p, m, m)];
        }
    }

    // VVOV[(c*nvir+a), (j*nvir+b)] = (ca|jb), shape (nvir², nocc·nvir), is the
    // single largest transient in the crate (~203 GB at the audit scale). Every
    // VVOV read inside the c_idx loop below is confined to the `nvir` rows
    // [c_idx*nvir, c_idx*nvir+nvir), so we never need more than a panel of c
    // rows resident. Block over c-panels: build vvov_panel for a panel of
    // c-values (a wide GEMM: b_vv[:, panel].t() · b_ov), consume it, discard it.
    // Peak VVOV footprint is one panel instead of the full (nvir², nov) square.
    let panel_c = panel_c.max(1).min(nvir.max(1));

    // g_{ai} has shape (nvir, nocc) -- virtual index a, occupied index i
    let mut g = Array2::zeros((nvir, nocc));

    // HF/Brillouin contribution: +4 * F_{ai}. Sign follows the CANONICAL Cayley
    // convention (shared `orbital_rotation::cayley_rotation`,
    // U = (I−κ/2)⁻¹(I+κ/2) ≈ exp(κ), g = +∂E/∂κ), where κ_{ai}>0 mixes virtual
    // into occupied via C_new ≈ C(I+κ). This is the closed-shell analogue of
    // u_oo_rimp2's `+2·F_ai` HF term (factor 4 vs 2 is the RHF density doubling).
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            g[(a, i)] += 4.0 * f_mo[(a_mo, i_mo)];
        }
    }

    // MP2 integral response:
    // dE_MP2/d kappa_{ck} = 2 * sum_{ijab} t_{ij,ab} * [2*d(ia|jb)/dk_{ck} - d(ib|ja)/dk_{ck}]
    //
    // d(ia|jb)/dk_{ck} = delta_{ik}*(ca|jb) + delta_{jk}*(ia|cb) - delta_{ac}*(ik|jb) - delta_{bc}*(ia|jk)
    // d(ib|ja)/dk_{ck} = delta_{ik}*(cb|ja) + delta_{jk}*(ib|ca) - delta_{bc}*(ik|ja) - delta_{ac}*(ib|jk)
    //
    // Combined: 2*d(ia|jb)/dk - d(ib|ja)/dk =
    //   delta_{ik} * [2*(ca|jb) - (cb|ja)]
    // + delta_{jk} * [2*(ia|cb) - (ib|ca)]
    // - delta_{ac} * [2*(ik|jb) - (ib|jk)]
    // - delta_{bc} * [2*(ia|jk) - (ik|ja)]
    //
    // For each (c, k) pair, we sum over ijab with the appropriate delta contractions.
    // ERIs are read from the (panelled) VVOV / (full) OOOV blocks:
    //   (ca|jb) = vvov[c*nvir+a, j*nvir+b]  (vvov row = local c-panel offset)
    //   (ik|jb) = ooov[i*nocc+k, j*nvir+b]     ((ib|jk)=(jk|ib)=ooov[j*nocc+k, i*nvir+b])

    let mut c0 = 0;
    while c0 < nvir {
        let c1 = (c0 + panel_c).min(nvir);
        // vvov_panel rows correspond to c in [c0, c1): local row (c-c0)*nvir + a.
        let bvv_panel = b_vv.slice(ndarray::s![.., c0 * nvir..c1 * nvir]);
        let vvov_panel = bvv_panel.t().dot(&b_ov); // ((c1-c0)·nvir, nov)

        // Parallelize over the panel's c-values. Each c_idx reads only shared,
        // read-only inputs (vvov_panel/ooov/t2) and produces its own row of nocc
        // gradient contributions — a disjoint-write pattern. We collect each
        // c_idx's row independently and scatter into `g` serially afterward.
        // The per-c_idx `grad_ck` accumulation order is byte-for-byte the same
        // as the serial version regardless of thread count, so the result is
        // bit-identical (no cross-c_idx summation whose order could vary).
        let panel_rows: Vec<(usize, Vec<f64>)> = (c0..c1)
            .into_par_iter()
            .map(|c_idx| {
                let cbase = (c_idx - c0) * nvir; // local vvov_panel row base for this c
                let c_mo = nocc_total + c_idx;
                // d(eps_p)/d kappa_{ck} = -4*(pp|ck) + 2*(pc|pk) for every p and
                // every k in this occupied range, batched via one GEMM (the
                // -4*(pp|ck) piece is Σ_P b_diag[P,p]·b_full[P,c_mo,k_mo], i.e.
                // b_diag^T @ b_full[:, c_mo, occ_slice]) plus one elementwise
                // pass per k (the (pc|pk) piece, which is NOT a plain GEMM since
                // both operands vary with p).
                let bc_col = b_full.slice(ndarray::s![.., c_mo, ..]); // (naux, nmo): (P, p) = B^P_{c,p}
                let b_ck = {
                    // b_full[:, c_mo, first_occ..first_occ+nocc] -> (naux, nocc)
                    let mut m = Array2::<f64>::zeros((naux, nocc));
                    for kk in 0..nocc {
                        for p in 0..naux {
                            m[(p, kk)] = b_full[(p, c_mo, first_occ + kk)];
                        }
                    }
                    m
                };
                let pp_ck = b_diag.t().dot(&b_ck); // (nmo, nocc): (pp|c,k)
                let mut row = vec![0.0_f64; nocc];
                for (k, row_k) in row.iter_mut().enumerate() {
                    let k_mo = first_occ + k;
                    let mut grad_ck = 0.0;

                    // Term 1: delta_{ik} -> i=k, sum over j,a,b
                    // 2 * sum_{jab} t_{kj,ab} * [2*(ca|jb) - (cb|ja)]
                    for j in 0..nocc {
                        for a in 0..nvir {
                            let ca = cbase + a;
                            for b in 0..nvir {
                                let t_kj_ab = t2[(k * nvir + a) * nov + j * nvir + b];
                                let eri_cajb = vvov_panel[(ca, j * nvir + b)];
                                let eri_cbja = vvov_panel[(cbase + b, j * nvir + a)];
                                grad_ck += t_kj_ab * (2.0 * eri_cajb - eri_cbja);
                            }
                        }
                    }

                    // Term 2: delta_{jk} -> j=k, sum over i,a,b
                    // 2 * sum_{iab} t_{ik,ab} * [2*(ia|cb) - (ib|ca)]
                    // (ia|cb) = (cb|ia) = vvov[c*nvir+b, i*nvir+a];
                    // (ib|ca) = (ca|ib) = vvov[c*nvir+a, i*nvir+b]
                    for i in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let t_ik_ab = t2[(i * nvir + a) * nov + k * nvir + b];
                                let eri_iacb = vvov_panel[(cbase + b, i * nvir + a)];
                                let eri_ibca = vvov_panel[(cbase + a, i * nvir + b)];
                                grad_ck += t_ik_ab * (2.0 * eri_iacb - eri_ibca);
                            }
                        }
                    }

                    // Term 3: delta_{ac} -> a=c, sum over i,j,b
                    // -2 * sum_{ijb} t_{ij,cb} * [2*(ik|jb) - (ib|jk)]
                    // (ik|jb) = ooov[i*nocc+k, j*nvir+b];
                    // (ib|jk) = (jk|ib) = ooov[j*nocc+k, i*nvir+b]
                    for i in 0..nocc {
                        for j in 0..nocc {
                            for b in 0..nvir {
                                let t_ij_cb = t2[(i * nvir + c_idx) * nov + j * nvir + b];
                                let eri_ikjb = ooov[(i * nocc + k, j * nvir + b)];
                                let eri_ibjk = ooov[(j * nocc + k, i * nvir + b)];
                                grad_ck -= t_ij_cb * (2.0 * eri_ikjb - eri_ibjk);
                            }
                        }
                    }

                    // Term 4: delta_{bc} -> b=c, sum over i,j,a
                    // -2 * sum_{ija} t_{ij,ac} * [2*(ia|jk) - (ik|ja)]
                    // (ia|jk) = (jk|ia) = ooov[j*nocc+k, i*nvir+a];
                    // (ik|ja) = ooov[i*nocc+k, j*nvir+a]
                    for i in 0..nocc {
                        for j in 0..nocc {
                            for a in 0..nvir {
                                let t_ij_ac = t2[(i * nvir + a) * nov + j * nvir + c_idx];
                                let eri_iajk = ooov[(j * nocc + k, i * nvir + a)];
                                let eri_ikja = ooov[(i * nocc + k, j * nvir + a)];
                                grad_ck -= t_ij_ac * (2.0 * eri_iajk - eri_ikja);
                            }
                        }
                    }

                    // Denominator-response term:
                    //   - sum_{ijab} t_{ij,ab} * [dD_{ijab}/dk_{ck}] * s_{ij,ab}
                    // dD_{ijab}/dk_{ck} = deps_i + deps_j - deps_a - deps_b, where
                    // deps_p = d(eps_p)/d kappa_{ck} = -4*(pp|ck) + 2*(pc|pk).
                    // (pc|pk) is computed here per-p (not a plain GEMM: both
                    // operands vary with p), reusing the already-sliced bc_col.
                    let mut deps = vec![0.0_f64; nmo];
                    for (p, deps_p) in deps.iter_mut().enumerate() {
                        let mut pc_pk = 0.0;
                        for pidx in 0..naux {
                            pc_pk += bc_col[(pidx, p)] * b_full[(pidx, p, k_mo)];
                        }
                        *deps_p = -4.0 * pp_ck[(p, k)] + 2.0 * pc_pk;
                    }
                    let mut denom_term = 0.0;
                    for i in 0..nocc {
                        let deps_i = deps[first_occ + i];
                        for a in 0..nvir {
                            let deps_a = deps[nocc_total + a];
                            let ia = i * nvir + a;
                            for j in 0..nocc {
                                let deps_j = deps[first_occ + j];
                                for b in 0..nvir {
                                    let deps_b = deps[nocc_total + b];
                                    let jb = j * nvir + b;
                                    let d_dd = deps_i + deps_j - deps_a - deps_b;
                                    let t_ijab = t2[ia * nov + jb];
                                    denom_term -= t_ijab * d_dd * s_mat[(ia, jb)];
                                }
                            }
                        }
                    }

                    // Canonical convention (g = +∂E/∂κ, shared Cayley
                    // U = (I−κ/2)⁻¹(I+κ/2)): the whole MP2 integral- +
                    // denominator-response block carries the OPPOSITE overall
                    // sign to the pre-unification `-2·grad_ck + denom_term`,
                    // matched to the HF term's flip above so the full gradient
                    // is consistently g = +∂E/∂κ.
                    *row_k = 2.0 * grad_ck - denom_term;
                }
                (c_idx, row)
            })
            .collect();

        for (c_idx, row) in panel_rows {
            for (k, &val) in row.iter().enumerate() {
                g[(c_idx, k)] += val;
            }
        }
        c0 = c1;
    }

    g
}

/// Run OO-RI-MP2.
///
/// Starting from converged RHF orbitals, iteratively optimize MO coefficients
/// to minimize E_HF + E_MP2 jointly.
pub fn oo_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &OoRiMp2Config,
) -> Result<OoRiMp2Result, FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc = active_occ(nocc_total, config.frozen_core)?;
    let first_occ = config.frozen_core;
    let nvir = nbas - nocc_total;
    let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);

    // Resolved once per call (not per iteration): the AO-tensor budget above
    // and the t2/eri_ov_mat guard below both gate off the same configured
    // ceiling, so re-resolving per iteration would only add repeated
    // env/auto-detect work for an answer that cannot change mid-run.
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes);

    // One-electron integrals (fixed)
    let h = oneelectron::hcore(obs);

    // AO-side invariants: built once, reused every iteration + backtrack.
    // Thread the config budget (M1 resolver) rather than the env-only default.
    let ao = OoRiMp2AoTensors::build_with_budget(obs, dfbs, op, budget_bytes)?;

    // EnginePool is geometry/basis-only (density-independent) — build ONCE
    // here and reuse across every compute_hf_energy call in the outer
    // iteration + backtracking loops below (this is the OO-MP2 orbital
    // rotation loop: (mol, obs, bounds) never change across iterations, only
    // the orbital coefficients C do), instead of build_jk constructing a
    // fresh pool per call. Reduction order is unchanged, so results stay
    // bit-identical across thread counts.
    let pool = EnginePool::new(bounds.op, obs, 1e-14)?;

    // Start from converged RHF orbitals
    let mut c = rhf.mos_r().clone();

    // Initial energies
    let (mut e_hf, mut f_ao, _d) = compute_hf_energy(mol, obs, bounds, &c, nocc_total, &h, &pool, budget_bytes)?;
    let mut eps = orbital_energies(&c, &f_ao);
    let (mut e_mp2, mut b_ov) = compute_rimp2_with_orbitals(&ao, &c, &eps, &orb)?;
    let mut total_energy = e_hf + e_mp2;
    let mut grad_norm = f64::MAX;
    let nmo = nbas;
    let max_kappa = 0.3; // cap individual rotation angles (radians)

    // DIIS for orbital rotation extrapolation.
    //
    // We apply DIIS to the MO coefficient matrix C itself (the state variable),
    // using the orbital gradient mapped into the AO basis as the error vector.
    // This mirrors how SCF DIIS works: the Fock matrix is replaced by C, and
    // the commutator error is replaced by the orbital gradient.  The gradient
    // is mapped to a (nbas, nbas) matrix G_AO = C * g_full * C^T where g_full
    // is the antisymmetric gradient in the MO basis (g_full[a,i] = g[a,i],
    // g_full[i,a] = -g[a,i]).  This ensures the DIIS error has the same shape
    // as the trial vector (C).
    let mut diis = if config.use_diis {
        Some(Diis::new(config.diis_size))
    } else {
        None
    };

    for iter in 1..=config.max_iter {
        // Compute full-MO B tensor for gradient evaluation
        let b_full = compute_b_full_mo_with(&ao, &c)?;

        // Build t2 amplitudes. Fail-fast guard: compute_t2_and_integrals
        // returns (and this loop retains, at least for the rest of the
        // iteration) BOTH eri_ov_mat and t2 — two co-resident (nov,nov) f64
        // buffers with zero pre-flight check before this fix (the AO-tensor
        // stage above was budgeted via build_with_budget; this MO-space
        // allocation was not). Guarded here (the call site) rather than
        // inside compute_t2_and_integrals itself, since that function is also
        // called from rimp2.rs/oo_rimp2_gradient.rs/ferric-rpa::pno with an
        // infallible signature this pass must not disturb.
        check_t2_pair_alloc(&format!("OO-RI-MP2 t2/eri_ov (iter {iter})"), nocc, nvir, budget_bytes)?;
        let naux = dfbs.nbasis();
        let (t2, _eri_ov) = compute_t2_and_integrals(
            &b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux,
        );

        // Fock matrix in MO basis
        let f_mo = c.t().dot(&f_ao).dot(&c);

        // Orbital gradient g_{ai} with full 2e response terms (gated on the
        // same once-per-call resolved budget as the AO tensors / t2 pair).
        let g = compute_orbital_gradient(
            &f_mo, &t2, &b_full, &eps, nocc, nvir, first_occ, nocc_total, budget_bytes,
        )?;

        // Check gradient norm
        grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Live per-iteration progress (see RhfConfig.verbose's doc for the
        // full rationale). STDOUT, opt-in via `config.verbose`, unlike this
        // print used to be (unconditional, stderr) before the SCF-style
        // verbose convention was extended to OO-RI-MP2.
        if config.verbose {
            println!(
                "OO-RI-MP2 iter {:3}: E_HF={:.10} E_MP2={:.10} E_tot={:.10} |g|={:.2e}",
                iter, e_hf, e_mp2, total_energy, grad_norm
            );
        }

        if grad_norm < config.grad_conv {
            return Ok(OoRiMp2Result {
                total_energy,
                hf_energy: e_hf,
                mp2_corr: e_mp2,
                converged: true,
                iterations: iter,
                grad_norm,
                mos: c,
                orbital_energies: eps,
            });
        }

        // Level-shifted approximate Newton step:
        //   kappa_{ai} = -g_{ai} / (eps_a - eps_i + mu)
        // The diagonal Hessian is the orbital energy gap; the level shift mu
        // regularizes small gaps (Bozkaya & Sherrill, JCP 135, 104103, 2011).
        let mut kappa_ov = Array2::zeros((nvir, nocc));
        for a in 0..nvir {
            for i in 0..nocc {
                let gap = eps[nocc_total + a] - eps[first_occ + i];
                kappa_ov[(a, i)] = -g[(a, i)] / (gap + config.level_shift);
            }
        }

        // Cap the step by scaling uniformly if any element exceeds max_kappa.
        let kappa_max_abs = kappa_ov.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        if kappa_max_abs > max_kappa {
            let scale = max_kappa / kappa_max_abs;
            kappa_ov *= scale;
        }

        // Build full antisymmetric kappa matrix (nmo x nmo) from ov block
        let mut kappa = Array2::zeros((nmo, nmo));
        for a in 0..nvir {
            let a_mo = nocc_total + a;
            for i in 0..nocc {
                let i_mo = first_occ + i;
                kappa[(a_mo, i_mo)] = kappa_ov[(a, i)];
                kappa[(i_mo, a_mo)] = -kappa_ov[(a, i)];
            }
        }

        // Cayley rotation
        let u = cayley_rotation(&kappa)?;
        let mut c_new = c.dot(&u);

        // DIIS extrapolation on the MO coefficients.
        // Error vector: map the orbital gradient to the AO basis as an
        // antisymmetric matrix in MO space, then project to AO:
        //   err_AO = C * g_antisym * C^T
        if let Some(ref mut diis_obj) = diis {
            let mut g_antisym = Array2::zeros((nmo, nmo));
            for a in 0..nvir {
                let a_mo = nocc_total + a;
                for i in 0..nocc {
                    let i_mo = first_occ + i;
                    g_antisym[(a_mo, i_mo)] = g[(a, i)];
                    g_antisym[(i_mo, a_mo)] = -g[(a, i)];
                }
            }
            let err_ao = c_new.dot(&g_antisym).dot(&c_new.t());
            c_new = diis_obj.step(&c_new, &err_ao);
        }

        // Evaluate energy at the new (possibly DIIS-extrapolated) orbitals
        let (ehf, fao, _d) =
            compute_hf_energy(mol, obs, bounds, &c_new, nocc_total, &h, &pool, budget_bytes)?;
        let epsnew = orbital_energies(&c_new, &fao);
        let (emp2, bov) = compute_rimp2_with_orbitals(&ao, &c_new, &epsnew, &orb)?;
        let total_new = ehf + emp2;
        let de = (total_new - total_energy).abs();

        // Backtracking if energy increased by more than a small tolerance.
        // DIIS can produce small uphill steps; we tolerate those.
        if total_new > total_energy + 1e-4 {
            // Fall back to a damped Newton step without DIIS extrapolation.
            let mut bt_kappa_ov = kappa_ov.clone();
            let mut bt_c = c.dot(&u);
            let mut bt_ehf = ehf;
            let mut bt_fao = fao.clone();
            let mut bt_eps = epsnew.clone();
            let mut bt_emp2 = emp2;
            let mut bt_bov = bov.clone();
            let mut bt_total = total_new;

            for _bt in 0..10 {
                bt_kappa_ov *= 0.5;
                let mut k = Array2::zeros((nmo, nmo));
                for a in 0..nvir {
                    let a_mo = nocc_total + a;
                    for i in 0..nocc {
                        let i_mo = first_occ + i;
                        k[(a_mo, i_mo)] = bt_kappa_ov[(a, i)];
                        k[(i_mo, a_mo)] = -bt_kappa_ov[(a, i)];
                    }
                }
                let u2 = cayley_rotation(&k)?;
                bt_c = c.dot(&u2);
                let (eh, fa, _) =
                    compute_hf_energy(mol, obs, bounds, &bt_c, nocc_total, &h, &pool, budget_bytes)?;
                let en = orbital_energies(&bt_c, &fa);
                let (em, bo) = compute_rimp2_with_orbitals(&ao, &bt_c, &en, &orb)?;
                bt_total = eh + em;
                if bt_total <= total_energy + 1e-12 {
                    bt_ehf = eh;
                    bt_fao = fa;
                    bt_eps = en;
                    bt_emp2 = em;
                    bt_bov = bo;
                    break;
                }
                bt_ehf = eh;
                bt_fao = fa;
                bt_eps = en;
                bt_emp2 = em;
                bt_bov = bo;
            }

            // The backtracking loop commits bt_* to its last trial step on every
            // path (break or exhaustion), so bt_c is always the step to take.
            c = bt_c.clone();
            e_hf = bt_ehf;
            e_mp2 = bt_emp2;
            total_energy = bt_total;
            f_ao = bt_fao;
            eps = bt_eps;
            b_ov = bt_bov;

            // Reset DIIS after backtracking since the extrapolated
            // subspace produced an uphill step.
            if let Some(ref mut diis_obj) = diis {
                diis_obj.reset();
            }
        } else {
            // Accept the (possibly DIIS-extrapolated) step
            c = c_new;
            e_hf = ehf;
            e_mp2 = emp2;
            total_energy = total_new;
            f_ao = fao;
            eps = epsnew;
            b_ov = bov;
        }

        if de < config.energy_conv && iter > 1 {
            // Energy converged; recompute gradient to check convergence.
            // Same co-resident t2/eri_ov_mat guard as the main-loop call site
            // above — this is a second, independent (nov,nov)-pair build.
            check_t2_pair_alloc(
                &format!("OO-RI-MP2 t2/eri_ov (iter {iter}, convergence recheck)"),
                nocc, nvir, budget_bytes,
            )?;
            let b_full2 = compute_b_full_mo_with(&ao, &c)?;
            let naux2 = dfbs.nbasis();
            let (t2_2, _) = compute_t2_and_integrals(
                &b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux2,
            );
            let f_mo2 = c.t().dot(&f_ao).dot(&c);
            let g2 = compute_orbital_gradient(
                &f_mo2, &t2_2, &b_full2, &eps, nocc, nvir, first_occ, nocc_total, budget_bytes,
            )?;
            grad_norm = g2.iter().map(|x| x * x).sum::<f64>().sqrt();

            if grad_norm < config.grad_conv * 10.0 {
                return Ok(OoRiMp2Result {
                    total_energy,
                    hf_energy: e_hf,
                    mp2_corr: e_mp2,
                    converged: true,
                    iterations: iter,
                    grad_norm,
                    mos: c,
                    orbital_energies: eps,
                });
            }
        }
    }

    Ok(OoRiMp2Result {
        total_energy,
        hf_energy: e_hf,
        mp2_corr: e_mp2,
        converged: false,
        iterations: config.max_iter,
        grad_norm,
        mos: c,
        orbital_energies: eps,
    })
}

/// Compute RI-MP2 energy for orbitals rotated by kappa (for finite-difference testing).
///
/// Takes initial MO coefficients, applies a Cayley rotation with the given kappa,
/// rebuilds Fock / density, and returns E_HF + E_MP2.
// System context (mol, two bases, operator, bounds) plus the rotation inputs
// (c_init, kappa) and orbital partition — all distinct, nothing left to bundle.
#[allow(clippy::too_many_arguments)]
pub fn energy_at_kappa(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    c_init: &Array2<f64>,
    kappa: &Array2<f64>,
    orb: &OrbitalSpace,
) -> Result<f64, FerricError> {
    let nocc_total = orb.nocc_total;
    let h = oneelectron::hcore(obs);
    let ao = OoRiMp2AoTensors::build(obs, dfbs, op)?;
    let u = cayley_rotation(kappa)?;
    let c_rot = c_init.dot(&u);
    let pool = EnginePool::new(bounds.op, obs, 1e-14)?;
    let (e_hf, f_ao, _) = compute_hf_energy(mol, obs, bounds, &c_rot, nocc_total, &h, &pool, ferric_core::memory::resolve_budget_bytes(None))?;
    let eps = orbital_energies(&c_rot, &f_ao);
    let (e_mp2, _) = compute_rimp2_with_orbitals(&ao, &c_rot, &eps, orb)?;
    Ok(e_hf + e_mp2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    fn setup_h2() -> (Molecule, PreparedBasis, PreparedBasis, Operator, SchwarzBounds, ScfResult) {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rhf.converged);
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        (mol, obs, dfbs, op, bounds, rhf)
    }

    /// M4 memory-guard regression: `check_t2_pair_alloc` must fire (Err) at
    /// the production-scale shape from the task's own motivating incident
    /// (nocc≈60, nvir≈900 → nov≈54,000 → co-resident t2+eri_ov_mat ≈46.6 GB)
    /// against a small configured budget, and the error message must name
    /// both the estimated requirement and the configured budget in GB so an
    /// operator can act on it without re-deriving the math.
    #[test]
    fn check_t2_pair_alloc_rejects_realistic_large_scale() {
        let nocc = 60;
        let nvir = 900;
        let budget_bytes = ferric_core::memory::gib_to_bytes(1.0); // 1 GiB — tiny vs ~46.6 GB
        let err = check_t2_pair_alloc("test large OO-MP2", nocc, nvir, budget_bytes).unwrap_err();
        let msg = err.to_string();
        // nov = 54,000; nov² = 2.916e9; ×2 buffers ×8 bytes = 46,656,000,000 B = 46.656 GB.
        assert!(
            msg.contains("46.6"),
            "expected the ~46.6 GB estimate in the error message, got: {msg}"
        );
        assert!(
            msg.contains("1.07 GB") || msg.contains("budget is 1."),
            "expected the ~1 GiB configured budget in the error message, got: {msg}"
        );
    }

    /// Companion to the rejection test: a small/typical system (water-scale,
    /// nocc/nvir well within existing OO-MP2 test fixtures) must NOT be
    /// rejected under the auto-resolved (or a generous explicit) budget —
    /// critical since OO-MP2 is iterative and a wrong guard formula would
    /// silently break every currently-passing convergence test by erroring
    /// out before the first iteration completes.
    #[test]
    fn check_t2_pair_alloc_accepts_small_system() {
        // Water/cc-pVDZ scale: nocc=5, nvir=19 (see test_oo_gradient_bit_identical_across_thread_counts).
        let nocc = 5;
        let nvir = 19;
        // Generous explicit budget (1 GiB) -- nov=95, nov²·2·8 = 144,400 bytes, nowhere close.
        let budget_bytes = ferric_core::memory::gib_to_bytes(1.0);
        assert!(check_t2_pair_alloc("test small OO-MP2", nocc, nvir, budget_bytes).is_ok());
        // Also must not reject under the real auto-resolved budget (None -> resolve_budget_bytes).
        let auto_budget = ferric_core::memory::resolve_budget_bytes(None);
        assert!(check_t2_pair_alloc("test small OO-MP2 (auto budget)", nocc, nvir, auto_budget).is_ok());
    }

    /// Memory-guard regression for the gradient-intermediates guard (Finding 2
    /// sibling of `check_t2_pair_alloc_rejects_realistic_large_scale`): at the
    /// same production-scale shape (nocc=60, nvir=900 → nov=54,000; ovov +
    /// s_mat alone are 2·nov²·8 ≈ 46.7 GB, plus the naux-scaled b_ov/b_vv/
    /// b_oo/b_diag blocks and ooov) the guard must fire against a small
    /// budget, naming the label and the budget so an operator can act on it.
    /// Pure arithmetic — allocates nothing.
    #[test]
    fn check_gradient_intermediates_alloc_rejects_realistic_large_scale() {
        let nocc = 60;
        let nvir = 900;
        let naux = 3000; // realistic RI aux dimension at this orbital scale
        let nmo = nocc + nvir; // no frozen core: nocc_total + nvir
        let budget_bytes = ferric_core::memory::gib_to_bytes(1.0); // 1 GiB — tiny vs ~69 GB
        let err = check_gradient_intermediates_alloc(
            "OO-RI-MP2 gradient intermediates",
            nocc, nvir, naux, nmo, budget_bytes,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("OO-RI-MP2 gradient intermediates"),
            "expected the guard label in the error message, got: {msg}"
        );
        assert!(
            msg.contains("budget is 1."),
            "expected the ~1 GiB configured budget in the error message, got: {msg}"
        );

        // Companion small-scale check (water/cc-pVDZ shape): must pass under
        // the same 1 GiB budget — a wrong estimate formula would otherwise
        // break every currently-passing OO-MP2 convergence test.
        assert!(check_gradient_intermediates_alloc(
            "OO-RI-MP2 gradient intermediates",
            5, 19, 84, 24, budget_bytes,
        )
        .is_ok());
    }

    /// `compute_t2_only`'s internal guard must likewise fire at large scale
    /// and pass through at small scale — same estimate as
    /// `check_t2_pair_alloc` (this function has no external callers to break,
    /// so its signature carries the guard directly rather than pushing it to
    /// a call site; see its doc comment).
    #[test]
    fn compute_t2_only_memory_guard_fires_at_large_scale_and_passes_small_scale() {
        let (_mol, obs, dfbs, op, _bounds, rhf) = setup_h2();
        let nbas = obs.nbasis();
        let nocc_total = 1usize;
        let nocc = nocc_total;
        let first_occ = 0;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();
        let c = rhf.mos_r();
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();
        let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);
        let h = oneelectron::hcore(&obs);
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let pool = EnginePool::new(op, &obs, 1e-14).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&_mol, &obs, &bounds, c, nocc_total, &h, &pool, ferric_core::memory::resolve_budget_bytes(None)).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e, b_flat) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();

        // Small H2/cc-pVDZ scale must pass under a generous explicit budget.
        let ok = compute_t2_only(
            &b_flat, &eps, nocc, nvir, nocc_total, first_occ, naux,
            Some(ferric_core::memory::gib_to_bytes(1.0)),
        );
        assert!(ok.is_ok(), "small-system compute_t2_only unexpectedly rejected: {:?}", ok.err());

        // A tiny budget must reject even this small system (proves the guard
        // is actually wired into compute_t2_only, not a no-op).
        let tiny_budget = 100usize; // 100 bytes -- far below even this tiny nov²
        let err = compute_t2_only(
            &b_flat, &eps, nocc, nvir, nocc_total, first_occ, naux, Some(tiny_budget),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("budget is"), "expected a budget-shaped error message, got: {msg}");
    }

    /// GEMM restructure (P6-residual) regression: `compute_t2_and_integrals`
    /// and `compute_t2_only` must reproduce the original per-element scalar
    /// formula `eri_iajb = Σ_p b_flat[p,ia]·b_flat[p,jb]` exactly (to
    /// numerical noise) — this is the check that would catch a transposition
    /// bug in the `b_flat^T @ b_flat` GEMM reindexing that a same-shape
    /// (nov×nov, symmetric-looking) mistake could otherwise hide.
    #[test]
    fn test_t2_gemm_matches_scalar_formula() {
        let (mol, obs, dfbs, op, bounds, rhf) = setup_h2();
        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nocc = nocc_total;
        let first_occ = 0;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();
        let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);
        let c = rhf.mos_r();
        let h = oneelectron::hcore(&obs);
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();
        let pool = EnginePool::new(op, &obs, 1e-14).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h, &pool, ferric_core::memory::resolve_budget_bytes(None)).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e, b_flat) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();

        let nov = nocc * nvir;
        // Reference: the original O(nov^2 * naux) scalar double-dot formula,
        // reimplemented independently of compute_t2_and_integrals/compute_t2_only.
        let mut t2_ref = vec![0.0f64; nov * nov];
        let mut eri_ov_ref = vec![0.0f64; nov * nov];
        for i in 0..nocc {
            for a in 0..nvir {
                let ia = i * nvir + a;
                for j in 0..nocc {
                    for b in 0..nvir {
                        let jb = j * nvir + b;
                        let eri_iajb: f64 =
                            (0..naux.min(b_flat.nrows())).map(|p| b_flat[(p, ia)] * b_flat[(p, jb)]).sum();
                        let denom = eps[first_occ + i] + eps[first_occ + j]
                            - eps[nocc_total + a]
                            - eps[nocc_total + b];
                        eri_ov_ref[ia * nov + jb] = eri_iajb;
                        t2_ref[ia * nov + jb] = eri_iajb / denom;
                    }
                }
            }
        }

        let (t2_gemm, eri_ov_gemm) =
            compute_t2_and_integrals(&b_flat, &eps, nocc, nvir, nocc_total, first_occ, naux);
        let t2_only_gemm =
            compute_t2_only(&b_flat, &eps, nocc, nvir, nocc_total, first_occ, naux, None).unwrap();

        let max_t2_diff = t2_ref
            .iter()
            .zip(t2_gemm.iter())
            .map(|(r, g)| (r - g).abs())
            .fold(0.0, f64::max);
        let max_eri_diff = eri_ov_ref
            .iter()
            .zip(eri_ov_gemm.iter())
            .map(|(r, g)| (r - g).abs())
            .fold(0.0, f64::max);
        let max_t2_only_diff = t2_ref
            .iter()
            .zip(t2_only_gemm.iter())
            .map(|(r, g)| (r - g).abs())
            .fold(0.0, f64::max);

        assert!(
            max_t2_diff < 1e-12,
            "compute_t2_and_integrals t2 vs scalar formula maxdiff={max_t2_diff:.3e}"
        );
        assert!(
            max_eri_diff < 1e-12,
            "compute_t2_and_integrals eri_ov vs scalar formula maxdiff={max_eri_diff:.3e}"
        );
        assert!(
            max_t2_only_diff < 1e-12,
            "compute_t2_only vs scalar formula maxdiff={max_t2_only_diff:.3e}"
        );
    }

    #[test]
    fn test_oo_rimp2_lowers_energy() {
        let (mol, obs, dfbs, op, bounds, rhf) = setup_h2();

        // Standard RI-MP2
        let ri_result = crate::rimp2::ri_mp2(
            &mol,
            &obs,
            &dfbs,
            op,
            &rhf,
            &crate::rimp2::RiMp2Config::default(),
        )
        .unwrap();

        // OO-RI-MP2
        let oo_result = oo_ri_mp2(
            &mol,
            &obs,
            &dfbs,
            op,
            &bounds,
            &rhf,
            &OoRiMp2Config::default(),
        )
        .unwrap();

        eprintln!("Standard RI-MP2 total: {:.10}", ri_result.total_energy);
        eprintln!(
            "OO-RI-MP2 total: {:.10} (HF={:.10}, MP2={:.10})",
            oo_result.total_energy, oo_result.hf_energy, oo_result.mp2_corr
        );
        eprintln!(
            "OO converged: {}, iters: {}, |g|: {:.2e}",
            oo_result.converged, oo_result.iterations, oo_result.grad_norm
        );
        eprintln!(
            "Energy lowering: {:.2e}",
            ri_result.total_energy - oo_result.total_energy
        );

        // OO-RI-MP2 total energy should be <= standard RI-MP2 total energy
        // (variational principle for orbital optimization)
        assert!(
            oo_result.total_energy <= ri_result.total_energy + 1e-10,
            "OO total ({:.10}) should be <= RI total ({:.10})",
            oo_result.total_energy,
            ri_result.total_energy
        );
        assert!(oo_result.converged, "OO-RI-MP2 should converge");
    }

    /// DIAGNOSTIC (2026-07-28): how non-canonical is the converged OO basis?
    ///
    /// `orbital_energies` takes `diag(C^T F C)` and those values are used as
    /// MP2 DENOMINATORS (eps_a + eps_b - eps_i - eps_j), not merely as a
    /// Hessian preconditioner. Orbital optimization rotates away from the
    /// canonical basis by construction, so F_mo is NOT diagonal and the
    /// off-diagonal coupling is silently dropped.
    ///
    /// This is the same defect class as the Boys-screened `eps_loc` bug
    /// (screen.rs, fixed 17e994e) and the AO-Laplace pseudo-density bug.
    /// Whether it MATTERS depends on how large the occ-occ / vir-vir
    /// off-diagonal blocks actually are at convergence -- which nobody had
    /// measured. This test measures it and pins the answer.
    ///
    /// NOTE the occ-VIR block is expected to be nonzero (that is the orbital
    /// gradient, which OO drives toward zero); what matters for the MP2
    /// denominators is the occ-OCC and vir-VIR blocks.
    #[test]
    fn oo_converged_basis_offdiagonal_fock_is_measured_not_assumed() {
        // Water, NOT H2: H2 has nocc = 1, so max|F_oo offdiag| is trivially 0
        // and the occ-occ half of this diagnostic would be vacuous.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol, &obs, op, &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        )
        .unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        assert!(mol.nelec() as usize / 2 > 1, "need nocc > 1 or occ-occ is vacuous");

        let oo = oo_ri_mp2(
            &mol, &obs, &dfbs, op, &bounds, &rhf, &OoRiMp2Config::default(),
        )
        .unwrap();
        assert!(oo.converged, "OO must converge for this diagnostic to mean anything");

        let nocc = mol.nelec() as usize / 2;
        let c = &oo.mos;
        // Rebuild F in the OPTIMIZED MO basis. compute_hf_energy returns the AO
        // Fock built from the given coefficients, which is exactly the matrix
        // orbital_energies() takes its diagonal from.
        let h = oneelectron::hcore(&obs);
        let pool = EnginePool::new(bounds.op, &obs, 1e-14).unwrap();
        let (_e, f_ao, _d) =
            compute_hf_energy(&mol, &obs, &bounds, c, nocc, &h, &pool, 0).unwrap();
        let f_mo = c.t().dot(&f_ao).dot(c);
        let n = f_mo.nrows();

        let block_max = |rs: std::ops::Range<usize>, cs: std::ops::Range<usize>| -> f64 {
            let mut m = 0.0f64;
            for i in rs.clone() {
                for j in cs.clone() {
                    if i != j {
                        m = m.max(f_mo[(i, j)].abs());
                    }
                }
            }
            m
        };
        let oo_blk = block_max(0..nocc, 0..nocc);
        let vv_blk = block_max(nocc..n, nocc..n);
        let ov_blk = {
            let mut m = 0.0f64;
            for i in 0..nocc {
                for a in nocc..n {
                    m = m.max(f_mo[(i, a)].abs());
                }
            }
            m
        };
        let diag: Vec<f64> = (0..n).map(|i| f_mo[(i, i)]).collect();
        let spread = diag.iter().cloned().fold(f64::MIN, f64::max)
            - diag.iter().cloned().fold(f64::MAX, f64::min);

        eprintln!(
            "H2O/cc-pVDZ OO-RI-MP2 converged basis: max|F_oo offdiag| = {oo_blk:.3e}, \
             max|F_vv offdiag| = {vv_blk:.3e}, max|F_ov| = {ov_blk:.3e}, \
             diag spread = {spread:.3e}"
        );
        eprintln!(
            "  ratio to spread: occ-occ {:.2e}, vir-vir {:.2e}",
            oo_blk / spread,
            vv_blk / spread
        );

        // TEETH: this must actually be exercising a ROTATED basis, else the
        // measurement is vacuous. A converged OO run that never moved off
        // canonical would make every off-diagonal ~0 for trivial reasons.
        assert!(
            oo.iterations > 0,
            "OO reported 0 iterations -- basis never rotated, diagnostic is vacuous"
        );

        // Pin the measured magnitude. This is NOT asserting the approximation
        // is fine; it is recording the size so a regression (or a fix) is
        // visible. If either blows up, the denominators are badly wrong.
        assert!(
            oo_blk / spread < 1e-1 && vv_blk / spread < 1e-1,
            "occ-occ ({:.2e}) or vir-vir ({:.2e}) off-diagonal coupling exceeds \
             10% of the diagonal spread -- the diag(F_mo) MP2 denominators in \
             orbital_energies() are then a poor approximation and must be \
             replaced by semicanonicalization (cf. screen.rs fix 17e994e)",
            oo_blk / spread,
            vv_blk / spread
        );
    }

    #[test]
    fn test_oo_rimp2_gradient_finite_difference() {
        let (mol, obs, dfbs, op, bounds, rhf) = setup_h2();

        let nbas = obs.nbasis();
        let nelec = mol.nelec() as usize;
        let nocc_total = nelec / 2;
        let nocc = nocc_total;
        let first_occ = 0;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();
        let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);

        // Compute analytic gradient at RHF orbitals
        let c = rhf.mos_r();
        let h = oneelectron::hcore(&obs);
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();
        let pool = EnginePool::new(op, &obs, 1e-14).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h, &pool, ferric_core::memory::resolve_budget_bytes(None)).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e_mp2, b_ov) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();

        // Build t2 amplitudes
        let (t2, _) = compute_t2_and_integrals(
            &b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux,
        );

        // Build full-MO B tensor for gradient
        let b_full = compute_b_full_mo_with(&ao, c).unwrap();

        let f_mo = c.t().dot(&f_ao).dot(c);
        // usize::MAX budget: guard passes, full-width panel — preserves the
        // pre-budget-parameter behavior for this tiny test system.
        let g = compute_orbital_gradient(
            &f_mo, &t2, &b_full, &eps, nocc, nvir, first_occ, nocc_total, usize::MAX,
        )
        .unwrap();

        // Finite difference check for each (a, i) component
        let delta = 1e-5;
        let mut max_err = 0.0f64;
        for a in 0..nvir {
            let a_mo = nocc_total + a;
            for i in 0..nocc {
                let i_mo = first_occ + i;

                // kappa+ : perturb (a,i) by +delta
                let mut kappa_plus = Array2::zeros((nbas, nbas));
                kappa_plus[(a_mo, i_mo)] = delta;
                kappa_plus[(i_mo, a_mo)] = -delta;

                let e_plus = energy_at_kappa(
                    &mol, &obs, &dfbs, op, &bounds, c, &kappa_plus, &orb,
                )
                .unwrap();

                // kappa- : perturb (a,i) by -delta
                let mut kappa_minus = Array2::zeros((nbas, nbas));
                kappa_minus[(a_mo, i_mo)] = -delta;
                kappa_minus[(i_mo, a_mo)] = delta;

                let e_minus = energy_at_kappa(
                    &mol, &obs, &dfbs, op, &bounds, c, &kappa_minus, &orb,
                )
                .unwrap();

                let fd_grad = (e_plus - e_minus) / (2.0 * delta);
                let analytic = g[(a, i)];
                let err = (fd_grad - analytic).abs();
                max_err = max_err.max(err);

                eprintln!(
                    "grad[a={},i={}]: analytic={:+.8e}, FD={:+.8e}, err={:.2e}",
                    a, i, analytic, fd_grad, err
                );
            }
        }

        eprintln!("Max gradient error: {:.2e}", max_err);
        // Allow generous tolerance for FD vs analytic (FD has O(delta^2) truncation
        // plus numerical noise from integral recomputation)
        assert!(
            max_err < 1e-3,
            "Gradient FD check failed: max_err={:.2e}",
            max_err
        );
    }

    /// Widening companion to `test_oo_rimp2_gradient_finite_difference`
    /// (docs/VALIDATION.md "OO-RI-MP2" row follow-up, 2026-07-19): that test
    /// is H2/cc-pVDZ only (nocc=1, nvir=4) -- an analytic-vs-FD match on a
    /// single, highly symmetric 2-electron system does not rule out a
    /// formula bug that happens to cancel for that particular shape.
    ///
    /// HISTORY: this test originally FAILED here (found 2026-07-19,
    /// max_err~4.2e-3 Ha/rad, confirmed delta-independent via a four-point
    /// 1e-5..3e-7 scan, so genuinely a formula bug and not FD truncation
    /// noise). ROOT CAUSE (found 2026-07-20): `compute_orbital_gradient`'s
    /// Term1-4 (the MP2 integral-response contraction) were themselves
    /// correct all along; what was missing was an orbital-energy-denominator
    /// response term (the doc comment's claim that this vanishes via
    /// "Brillouin's condition" was simply wrong -- d(eps_p)/d(kappa_ck) is
    /// generically nonzero, since it comes from the density-dependence of
    /// the Fock matrix under the rotated occupied space, not from any
    /// frozen-Fock rotation of F itself). H2/cc-pVDZ (nocc=1) happens not to
    /// expose this because... see `compute_orbital_gradient`'s doc comment
    /// for the full derivation and end-to-end Python/PySCF verification
    /// numbers. Fixed by adding the closed-form denominator-response term
    /// (one extra density-response Fock-like build per (c,k) via the
    /// existing full-MO B tensor -- no CPHF/Z-vector solve needed, since
    /// kappa_{ck} enters eps_p(kappa) explicitly, not implicitly).
    #[test]
    fn test_oo_rimp2_gradient_finite_difference_h2o_sto3g() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rhf.converged);
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let nbas = obs.nbasis();
        let nelec = mol.nelec() as usize;
        let nocc_total = nelec / 2;
        let nocc = nocc_total;
        let first_occ = 0;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();
        let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);
        eprintln!("H2O/STO-3G: nocc={}, nvir={}, naux={}", nocc, nvir, naux);

        // Compute analytic gradient at RHF orbitals
        let c = rhf.mos_r();
        let h = oneelectron::hcore(&obs);
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();
        let pool = EnginePool::new(op, &obs, 1e-14).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h, &pool, ferric_core::memory::resolve_budget_bytes(None)).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e_mp2, b_ov) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();

        let (t2, _) = compute_t2_and_integrals(
            &b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux,
        );

        let b_full = compute_b_full_mo_with(&ao, c).unwrap();

        let f_mo = c.t().dot(&f_ao).dot(c);
        // usize::MAX budget: guard passes, full-width panel — preserves the
        // pre-budget-parameter behavior for this tiny test system.
        let g = compute_orbital_gradient(
            &f_mo, &t2, &b_full, &eps, nocc, nvir, first_occ, nocc_total, usize::MAX,
        )
        .unwrap();

        // delta=1e-5 matches the sibling H2/cc-pVDZ test. A four-point delta
        // scan (1e-5/3e-6/1e-6/3e-7, see the doc comment above) confirmed the
        // resulting error is flat across a 30x range of step size -- i.e. NOT
        // finite-difference truncation noise, so a single delta here is
        // sufficient to expose (not just barely catch) the real discrepancy.
        let delta = 1e-5;
        let mut max_err = 0.0f64;
        for a in 0..nvir {
            let a_mo = nocc_total + a;
            for i in 0..nocc {
                let i_mo = first_occ + i;

                let mut kappa_plus = Array2::zeros((nbas, nbas));
                kappa_plus[(a_mo, i_mo)] = delta;
                kappa_plus[(i_mo, a_mo)] = -delta;
                let e_plus = energy_at_kappa(
                    &mol, &obs, &dfbs, op, &bounds, c, &kappa_plus, &orb,
                )
                .unwrap();

                let mut kappa_minus = Array2::zeros((nbas, nbas));
                kappa_minus[(a_mo, i_mo)] = -delta;
                kappa_minus[(i_mo, a_mo)] = delta;
                let e_minus = energy_at_kappa(
                    &mol, &obs, &dfbs, op, &bounds, c, &kappa_minus, &orb,
                )
                .unwrap();

                let fd_grad = (e_plus - e_minus) / (2.0 * delta);
                let analytic = g[(a, i)];
                let err = (fd_grad - analytic).abs();
                max_err = max_err.max(err);

                eprintln!(
                    "grad[a={},i={}]: analytic={:+.8e}, FD={:+.8e}, err={:.2e}",
                    a, i, analytic, fd_grad, err
                );
            }
        }

        eprintln!("H2O/STO-3G max gradient error: {:.2e}", max_err);
        // Fixed 2026-07-20 (missing denominator-response term, see
        // compute_orbital_gradient's doc comment). Reuses the sibling
        // H2/cc-pVDZ test's 1e-3 tolerance unchanged (not loosened).
        assert!(
            max_err < 1e-3,
            "H2O/STO-3G gradient FD check failed: max_err={:.2e}",
            max_err
        );
    }

    /// S1 spike (item #1, docs/perf-tasks/S1-spike-oo-rimp2-reference-energy.md):
    /// no external reference is reachable for OO-RI-MP2's absolute energy —
    /// PySCF's `pyscf.mp` has no orbital-optimization capability (confirmed by
    /// this spike; `dir(mp)` = GMP2/MP2/RMP2/UMP2/dfgmp2/dfmp2/dfump2, no OO/OMP2
    /// variant) and no psi4/forte (which does ship OMP2) is installed in this
    /// environment. This is an INTERNAL SELF-CONSISTENCY check only, not an
    /// absolute-energy validation:
    ///
    /// `test_oo_rimp2_gradient_finite_difference` (above) already proves the
    /// analytic orbital-gradient FORMULA matches finite differences at an
    /// arbitrary point (kappa=0, i.e. the starting RHF orbitals) — that
    /// catches formula/sign bugs in `compute_orbital_gradient`. It does NOT
    /// prove that the Newton+DIIS+Cayley solver's declared convergence in
    /// `oo_ri_mp2` (`converged: true`, gated on `grad_norm < grad_conv`) is a
    /// real stationary point of the true Hylleraas-derived analytic gradient
    /// — the solver could in principle converge (self-consistently, by its
    /// own gradient evaluation) to a point where an INDEPENDENT finite
    /// difference of the total energy is nonzero, e.g. from a subtle bug
    /// shared between the gradient used for the Newton step and the gradient
    /// used for the convergence check (they are the same function call, so a
    /// systematic error would not self-cancel).
    ///
    /// This test converges OO-RI-MP2 on H2/cc-pVDZ, then independently
    /// recomputes the orbital gradient via central finite difference of
    /// `energy_at_kappa` around the CONVERGED orbitals (not kappa=0), and
    /// checks both (a) the FD gradient itself is near zero (true stationary
    /// point) and (b) it agrees with the analytic gradient recomputed at
    /// convergence. What this DOES validate: internal self-consistency of the
    /// converged point (no drift between the solver's stopping criterion and
    /// the actual energy landscape). What this does NOT validate: the
    /// absolute converged energy against any external ground truth (no such
    /// reference is reachable in this environment/timebox — see spike doc).
    #[test]
    fn test_oo_rimp2_converged_gradient_vanishes_h2_ccpvdz() {
        let (mol, obs, dfbs, op, bounds, rhf) = setup_h2();

        let config = OoRiMp2Config {
            grad_conv: 1e-6,
            energy_conv: 1e-10,
            ..Default::default()
        };
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();
        assert!(
            oo.converged,
            "OO-RI-MP2 H2/cc-pVDZ did not converge: {} iters, |g|={:.2e}",
            oo.iterations, oo.grad_norm
        );
        eprintln!(
            "Converged at iter {}: E_tot={:.10}, solver |g|={:.2e}",
            oo.iterations, oo.total_energy, oo.grad_norm
        );

        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nocc = nocc_total;
        let first_occ = 0;
        let nvir = nbas - nocc_total;
        let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);
        let c = &oo.mos;

        // Independently recompute the analytic gradient at the converged
        // orbitals (same formula as the solver, but evaluated fresh here
        // rather than trusting the solver's last internal value).
        let h = oneelectron::hcore(&obs);
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();
        let pool = EnginePool::new(op, &obs, 1e-14).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h, &pool, ferric_core::memory::resolve_budget_bytes(None)).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e_mp2, b_ov) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();
        let naux = dfbs.nbasis();
        let (t2, _) = compute_t2_and_integrals(&b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux);
        let b_full = compute_b_full_mo_with(&ao, c).unwrap();
        let f_mo = c.t().dot(&f_ao).dot(c);
        // usize::MAX budget: guard passes, full-width panel (pre-parameter behavior).
        let g_analytic = compute_orbital_gradient(&f_mo, &t2, &b_full, &eps, nocc, nvir, first_occ, nocc_total, usize::MAX).unwrap();
        let analytic_norm = g_analytic.iter().map(|x| x * x).sum::<f64>().sqrt();
        eprintln!("Independently recomputed analytic |g| at convergence: {:.2e}", analytic_norm);

        // Central finite difference of the total energy around the converged
        // orbitals, in every (a,i) rotation direction — an FD gradient that is
        // itself near zero is direct evidence of a true stationary point,
        // independent of whatever internal gradient the solver trusted.
        let delta = 1e-5;
        let mut max_fd_grad = 0.0f64;
        let mut max_err_vs_analytic = 0.0f64;
        for a in 0..nvir {
            let a_mo = nocc_total + a;
            for i in 0..nocc {
                let i_mo = first_occ + i;

                let mut kappa_plus = Array2::zeros((nbas, nbas));
                kappa_plus[(a_mo, i_mo)] = delta;
                kappa_plus[(i_mo, a_mo)] = -delta;
                let e_plus =
                    energy_at_kappa(&mol, &obs, &dfbs, op, &bounds, c, &kappa_plus, &orb).unwrap();

                let mut kappa_minus = Array2::zeros((nbas, nbas));
                kappa_minus[(a_mo, i_mo)] = -delta;
                kappa_minus[(i_mo, a_mo)] = delta;
                let e_minus =
                    energy_at_kappa(&mol, &obs, &dfbs, op, &bounds, c, &kappa_minus, &orb).unwrap();

                let fd_grad = (e_plus - e_minus) / (2.0 * delta);
                let analytic = g_analytic[(a, i)];
                max_fd_grad = max_fd_grad.max(fd_grad.abs());
                max_err_vs_analytic = max_err_vs_analytic.max((fd_grad - analytic).abs());

                eprintln!(
                    "converged grad[a={},i={}]: analytic={:+.8e}, FD={:+.8e}",
                    a, i, analytic, fd_grad
                );
            }
        }

        eprintln!(
            "At convergence: max|FD grad|={:.2e}, max FD-vs-analytic err={:.2e}",
            max_fd_grad, max_err_vs_analytic
        );

        // (a) The converged point is a real stationary point of the true
        // energy landscape, not just of the solver's own gradient evaluation.
        assert!(
            max_fd_grad < 1e-3,
            "FD gradient at converged orbitals is not near zero: max|FD grad|={:.2e}",
            max_fd_grad
        );
        // (b) The analytic gradient formula still agrees with FD at this
        // (different from kappa=0) point -- same check as
        // test_oo_rimp2_gradient_finite_difference but at the solver's actual
        // output rather than the starting point.
        assert!(
            max_err_vs_analytic < 1e-3,
            "Analytic vs FD gradient mismatch at convergence: {:.2e}",
            max_err_vs_analytic
        );
    }

    #[test]
    fn test_oo_rimp2_h2o_ccpvdz() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rhf.converged);
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let ri = crate::rimp2::ri_mp2(
            &mol,
            &obs,
            &dfbs,
            op,
            &rhf,
            &crate::rimp2::RiMp2Config::default(),
        )
        .unwrap();

        let config = OoRiMp2Config::default();
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

        eprintln!("H2O RI-MP2 total:    {:.10}", ri.total_energy);
        eprintln!(
            "H2O OO-RI-MP2 total: {:.10} (HF={:.10}, MP2={:.10})",
            oo.total_energy, oo.hf_energy, oo.mp2_corr
        );
        eprintln!(
            "OO converged: {}, iters: {}, |g|: {:.2e}",
            oo.converged, oo.iterations, oo.grad_norm
        );

        assert!(
            oo.converged,
            "OO-RI-MP2 H2O did not converge: {} iters, |g|={:.2e}",
            oo.iterations, oo.grad_norm
        );
        assert!(
            oo.total_energy <= ri.total_energy + 1e-10,
            "OO={:.10} should be <= RI={:.10}",
            oo.total_energy, ri.total_energy
        );
    }

    /// External absolute-energy cross-check against Psi4's OMP2 implementation
    /// (Bozkaya's `OMP2 (OO-MP2)` module, the same JCP 135, 104103 (2011)
    /// method this module's doc comment cites).
    ///
    /// Reference source: Psi4's own regression-test suite,
    /// `tests/omp2-1/{input.dat,output.ref}` (fetched from
    /// `github.com/psi4/psi4`, Psi4 1.1rc3.dev5, run 2017-05-15) — a genuine,
    /// independently-maintained OMP2 implementation, not a number we derived
    /// ourselves. Geometry: H2O Z-matrix `O; H 1 0.958; H 1 0.958 2 104.4776`
    /// (Angstrom), reproduced here as Cartesians built from the same bond
    /// length/angle. Confirmed to be the *exact* same geometry Psi4 used: our
    /// from-scratch nuclear-repulsion recomputation matches Psi4's printed
    /// `refnuc = 9.18738642147759` to all 11 digits when using the older
    /// CODATA Bohr radius (0.52917720859 Å) that 2017-era Psi4 shipped with;
    /// ferric's own (newer CODATA 2018) constant gives NRE 9.187386462 instead
    /// of 9.187386421 — a 4e-8 Ha geometry-constant drift, immaterial here.
    /// Basis: cc-pVDZ, `mp2_type conv` (Psi4 ran CONVENTIONAL 4-index OMP2,
    /// not density-fitted) — Psi4's DF-OMP2 variant exists but this test file
    /// deliberately used the conventional-integral path.
    ///
    /// Method mismatch this test does NOT paper over: ferric's `oo_ri_mp2` is
    /// density-fitted (RI) throughout, Psi4's reference here is not. The two
    /// are different methods (RI-OMP2 vs conventional OMP2), not the same
    /// method run twice, so exact agreement is neither expected nor claimed.
    /// We bound the expected RI-fitting error independently: running plain
    /// (non-orbital-optimized) MP2 on this exact geometry/basis through PySCF
    /// gives conventional E_corr = -0.20401925 Ha vs DF-MP2 (aux=cc-pVDZ-RI)
    /// E_corr = -0.20400407 Ha — a 1.52e-5 Ha RI error at the plain-MP2 level.
    /// The tolerance below (2e-4 Ha) is set generously above that measured RI
    /// floor to leave room for the RI error potentially differing somewhat
    /// once orbitals are also relaxed, while still being tight enough that a
    /// gross algorithmic error (wrong sign, wrong prefactor, wrong term) would
    /// fail it by 1-2 orders of magnitude.
    #[test]
    fn test_oo_rimp2_h2o_ccpvdz_matches_psi4_omp2_reference() {
        // Cartesian geometry reconstructed from Psi4's Z-matrx
        // (O; H 1 0.958; H 1 0.958 2 104.4776, degrees), same convention Psi4
        // prints in its "Geometry (in Angstrom)" block.
        let xyz = "3\nwater (psi4 omp2-1 geometry)\n\
                   O 0.000000000000 0.000000000000 0.000000000000\n\
                   H 0.000000000000 0.000000000000 0.958000000000\n\
                   H 0.000000000000 0.927579144347 -0.239501421649\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();

        // Sanity-check the geometry against Psi4's printed nuclear repulsion
        // energy before trusting anything downstream (catches a transcription
        // error in the Cartesian coordinates above, independent of any
        // ferric-side bug).
        let nre = mol.nuclear_repulsion();
        let psi4_nre_newer_codata_bohr = 9.187386461930224; // matches ferric's ANGSTROM_TO_BOHR constant
        assert!(
            (nre - psi4_nre_newer_codata_bohr).abs() < 1e-7,
            "geometry transcription mismatch: ferric NRE={nre:.10}, expected {psi4_nre_newer_codata_bohr:.10} \
             (Psi4-printed refnuc 9.18738642147759 under Psi4's older CODATA Bohr radius)"
        );

        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rhf.converged);

        // SCF cross-check: Psi4 refscf = -76.02676109559437 Ha.
        let psi4_refscf = -76.02676109559437;
        assert!(
            (rhf.energy - psi4_refscf).abs() < 1e-6,
            "SCF mismatch vs Psi4 refscf: ferric={:.10}, psi4={:.10}, diff={:.2e}",
            rhf.energy, psi4_refscf, (rhf.energy - psi4_refscf).abs()
        );

        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let config = OoRiMp2Config::default();
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();
        assert!(
            oo.converged,
            "OO-RI-MP2 H2O (psi4 geometry) did not converge: {} iters, |g|={:.2e}",
            oo.iterations, oo.grad_norm
        );

        // Psi4 OMP2 total energy (conventional integrals): refomp2 = -76.23167598916250.
        // The output.ref run itself printed -76.23167597922692 (tiny
        // version-drift between the pinned #TEST constant and the actual run
        // in that same file) -- both agree to 1e-8 Ha, well inside our
        // RI-driven tolerance, so either is a valid target.
        let psi4_omp2_total = -76.23167598916250_f64;
        let diff = oo.total_energy - psi4_omp2_total;
        eprintln!(
            "ferric OO-RI-MP2 total: {:.10}  Psi4 OMP2 (conventional) total: {:.10}  diff: {:.3e}",
            oo.total_energy, psi4_omp2_total, diff
        );
        assert!(
            diff.abs() < 2e-4,
            "ferric OO-RI-MP2={:.10} vs Psi4 OMP2={:.10}: diff={:.3e} Ha exceeds the RI-error-informed \
             2e-4 Ha tolerance (measured plain-MP2 RI floor on this system/basis is 1.52e-5 Ha)",
            oo.total_energy, psi4_omp2_total, diff
        );
    }

    /// The aux-blocked (disk-spill) path through the ThreeIndexSource must give
    /// bit-comparable results to the in-core path: b_full, the OV-dressed MP2
    /// energy, and the orbital gradient all agree to machine precision.
    #[test]
    fn test_spill_budget_paths_match_incore() {
        let (mol, obs, dfbs, op, bounds, rhf) = setup_h2();
        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nvir = nbas - nocc_total;
        let orb = OrbitalSpace::new(nocc_total, nvir, nocc_total, 0);
        let c = rhf.mos_r();
        let h = oneelectron::hcore(&obs);

        // In-core reference (unlimited budget).
        let ao_ref = OoRiMp2AoTensors::build_with_budget(&obs, &dfbs, op, usize::MAX).unwrap();
        assert_eq!(ao_ref.eri3_ao.borrow().n_blocks(), 1);
        // Tiny budget: ~3 aux rows per block, forces disk spill + many blocks.
        let tiny = obs.nbasis() * obs.nbasis() * 8 * 3;
        let ao_spill = OoRiMp2AoTensors::build_with_budget(&obs, &dfbs, op, tiny).unwrap();
        assert!(
            ao_spill.eri3_ao.borrow().n_blocks() > 1,
            "expected multi-block spill, got {}",
            ao_spill.eri3_ao.borrow().n_blocks()
        );

        // b_full identical.
        let b_ref = compute_b_full_mo_with(&ao_ref, c).unwrap();
        let b_spill = compute_b_full_mo_with(&ao_spill, c).unwrap();
        let maxdiff = (&b_ref - &b_spill).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff < 1e-12, "b_full spill vs in-core maxdiff={maxdiff:.2e}");

        // MP2 energy + b_ov identical.
        let pool = EnginePool::new(op, &obs, 1e-14).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h, &pool, ferric_core::memory::resolve_budget_bytes(None)).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (e_ref, bov_ref) = compute_rimp2_with_orbitals(&ao_ref, c, &eps, &orb).unwrap();
        let (e_spill, bov_spill) = compute_rimp2_with_orbitals(&ao_spill, c, &eps, &orb).unwrap();
        assert!((e_ref - e_spill).abs() < 1e-12, "E_MP2 spill vs in-core: {:.3e}", (e_ref - e_spill).abs());
        let bovdiff = (&bov_ref - &bov_spill).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(bovdiff < 1e-12, "b_ov spill vs in-core maxdiff={bovdiff:.2e}");
    }

    /// The VVOV c-panelled gradient must be exact for any panel width:
    /// panel_c = 1 (max blocking) vs panel_c = nvir (single panel, the former
    /// unblocked path).
    #[test]
    fn test_vvov_panelled_gradient_exact() {
        let (mol, obs, dfbs, op, bounds, rhf) = setup_h2();
        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nocc = nocc_total;
        let first_occ = 0;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();
        let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);
        let c = rhf.mos_r();
        let h = oneelectron::hcore(&obs);
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();
        let pool = EnginePool::new(op, &obs, 1e-14).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h, &pool, ferric_core::memory::resolve_budget_bytes(None)).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e, b_ov) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();
        let (t2, _) = compute_t2_and_integrals(&b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux);
        let b_full = compute_b_full_mo_with(&ao, c).unwrap();
        let f_mo = c.t().dot(&f_ao).dot(c);

        let g_full = compute_orbital_gradient_panelled(
            &f_mo, &t2, &b_full, &eps, nocc, nvir, first_occ, nocc_total, nvir,
        );
        for panel in [1usize, 2, 3] {
            let g_p = compute_orbital_gradient_panelled(
                &f_mo, &t2, &b_full, &eps, nocc, nvir, first_occ, nocc_total, panel,
            );
            let maxdiff = (&g_full - &g_p).iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(
                maxdiff < 1e-13,
                "panelled gradient (panel_c={panel}) differs from full: {maxdiff:.2e}"
            );
        }
    }

    /// Region 2 (P6-residual): the per-q MO-transform loops in
    /// `compute_b_full_mo_with` / `compute_rimp2_with_orbitals` fan out over
    /// rayon above `PAR_MO_TRANSFORM_WORK_THRESHOLD`. `OoRiMp2AoTensors` holds
    /// a `RefCell` (not `Sync`), so we cannot force distinct rayon thread-pool
    /// sizes around a call that borrows it (rayon's `ThreadPool::install`
    /// requires `Send` on the closure and its captures) the way
    /// `test_oo_gradient_bit_identical_across_thread_counts` does for the
    /// (pure-data) orbital-gradient path. Instead, cross-check the rayon-path
    /// result (this system clears `PAR_MO_TRANSFORM_WORK_THRESHOLD`, so the
    /// default global rayon pool exercises the parallel branch) against a
    /// direct (non-chunked, non-parallel) scalar MO transform built straight
    /// from the raw AO 3-index tensor + metric, proving the rayon branch
    /// computes the identical contraction to ground truth.
    #[test]
    fn test_region2_mo_transform_matches_scalar_reference() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("aug-cc-pvtz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rhf.converged);
        let aux_bs = basis::bundled("aug-cc-pvtz-rifit").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nvir = nbas - nocc_total;
        let orb = OrbitalSpace::new(nocc_total, nvir, nocc_total, 0);
        let c = rhf.mos_r().clone();
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();

        // Sanity: confirm this system actually clears the rayon threshold for
        // at least one of the two loops, so the test is exercising the branch
        // it claims to.
        let naux = ao.naux();
        let qc_full = naux.min(crate::rimp2::MO_STREAM_CHUNK);
        let nov = nocc_total * nvir;
        assert!(
            qc_full * nbas * nbas >= crate::rimp2::PAR_MO_TRANSFORM_WORK_THRESHOLD
                || qc_full * nov >= crate::rimp2::PAR_MO_TRANSFORM_WORK_THRESHOLD,
            "test fixture too small to exercise Region 2 rayon branch: \
             qc={qc_full} nbas={nbas} nov={nov}"
        );

        let b_full = compute_b_full_mo_with(&ao, &c).unwrap();
        let (_e, b_ov) = compute_rimp2_with_orbitals(&ao, &c, rhf.eps_r(), &orb).unwrap();

        // Independent, unchunked, unparallelized scalar reference for both
        // outputs, built directly from the raw AO 3-index tensor + metric
        // (bypasses for_each_block/MO_CHUNK/rayon entirely).
        let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v2c_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
        let eri3_raw = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap(); // (naux, nao, nao)
        let nao = obs.nbasis();
        let nmo = nbas;

        // b_full_ref[P,p,q] = sum_Q v2c_inv_sqrt[P,Q] * (C^T (Q|mu nu) C)[p,q]
        let mut mo_raw = Array3::<f64>::zeros((naux, nmo, nmo));
        for qidx in 0..naux {
            let bq_ao = eri3_raw.slice(ndarray::s![qidx, .., ..]);
            let half = bq_ao.dot(&c);
            let bq_mo = c.t().dot(&half);
            mo_raw.slice_mut(ndarray::s![qidx, .., ..]).assign(&bq_mo);
        }
        let mo_raw_flat = mo_raw.into_shape_with_order((naux, nmo * nmo)).unwrap();
        let b_full_ref_flat = v2c_inv_sqrt.dot(&mo_raw_flat);
        let b_full_ref = b_full_ref_flat.into_shape_with_order((naux, nmo, nmo)).unwrap();

        let b_full_maxdiff = (&b_full - &b_full_ref).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            b_full_maxdiff < 1e-10,
            "compute_b_full_mo_with (rayon path) vs unchunked scalar reference: maxdiff={b_full_maxdiff:.3e}"
        );

        // b_ov_ref[P,ia] = sum_Q v2c_inv_sqrt[P,Q] * (C_occ^T (Q|mu nu) C_vir)[ia]
        let c_occ = c.slice(ndarray::s![.., 0..nocc_total]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();
        let mut mo_ov_raw = Array2::<f64>::zeros((naux, nov));
        for qidx in 0..naux {
            let bq_ao = eri3_raw.slice(ndarray::s![qidx, .., ..]);
            // occ-first contraction, matching the production transform
            // (stream_dressed_mo_band / transform_3center_ov) so this reference
            // agrees with it bitwise rather than only to the reassociation floor.
            let tmp = c_occ.t().dot(&bq_ao);
            let bq_mo = tmp.dot(&c_vir);
            mo_ov_raw
                .slice_mut(ndarray::s![qidx, ..])
                .assign(&bq_mo.into_shape_with_order(nov).unwrap());
        }
        let b_ov_ref = v2c_inv_sqrt.dot(&mo_ov_raw);
        let b_ov_maxdiff = (&b_ov - &b_ov_ref).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            b_ov_maxdiff < 1e-10,
            "compute_rimp2_with_orbitals b_ov (rayon path) vs unchunked scalar reference: maxdiff={b_ov_maxdiff:.3e}"
        );
        let _ = nao; // used only for the (naux, nao, nao) shape documented above
    }

    /// The rayon-parallelized c_idx response-term loop (P6) must produce a
    /// byte-for-byte identical gradient regardless of thread count. Each c_idx
    /// writes only its own row via a disjoint-write collect, so there is no
    /// summation whose order varies with scheduling — bit-identity, not merely
    /// close-agreement, is the correct assertion. Mirrors
    /// `whole_pipeline_rhf_gradient_bit_identical_across_thread_counts` in
    /// ferric-scf/src/rhf.rs.
    #[test]
    fn test_oo_gradient_bit_identical_across_thread_counts() {
        // Water/cc-pVDZ gives nocc=5, nvir=19 — enough c-values for rayon to
        // actually split work across threads, unlike H2 (nvir=9, nocc=1).
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nocc = nocc_total;
        let first_occ = 0;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();
        let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);
        let c = rhf.mos_r();
        let h = oneelectron::hcore(&obs);
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();
        let pool = EnginePool::new(op, &obs, 1e-14).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h, &pool, ferric_core::memory::resolve_budget_bytes(None)).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e, b_ov) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();
        let (t2, _) = compute_t2_and_integrals(&b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux);
        let b_full = compute_b_full_mo_with(&ao, c).unwrap();
        let f_mo = c.t().dot(&f_ao).dot(c);

        // Force a multi-panel width (panel_c=3) so the parallel region runs
        // inside more than one GEMM panel, exercising the interaction of
        // panelling and rayon scheduling together.
        let run_with_threads = |n: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
            pool.install(|| {
                compute_orbital_gradient_panelled(
                    &f_mo, &t2, &b_full, &eps, nocc, nvir, first_occ, nocc_total, 3,
                )
            })
        };

        let g1 = run_with_threads(1);
        let g4 = run_with_threads(4);
        let g8 = run_with_threads(8);

        for a in 0..nvir {
            for i in 0..nocc {
                assert_eq!(
                    g1[(a, i)].to_bits(),
                    g4[(a, i)].to_bits(),
                    "OO gradient not bit-identical 1 vs 4 threads at (a={a}, i={i}): \
                     1={:.17e} (0x{:016x}), 4={:.17e} (0x{:016x})",
                    g1[(a, i)], g1[(a, i)].to_bits(),
                    g4[(a, i)], g4[(a, i)].to_bits(),
                );
                assert_eq!(
                    g1[(a, i)].to_bits(),
                    g8[(a, i)].to_bits(),
                    "OO gradient not bit-identical 1 vs 8 threads at (a={a}, i={i}): \
                     1={:.17e} (0x{:016x}), 8={:.17e} (0x{:016x})",
                    g1[(a, i)], g1[(a, i)].to_bits(),
                    g8[(a, i)], g8[(a, i)].to_bits(),
                );
            }
        }
    }

    #[test]
    fn test_cayley_is_unitary() {
        let n = 5;
        // Build a random antisymmetric matrix
        let mut kappa = Array2::zeros((n, n));
        let vals = [0.1, -0.2, 0.05, -0.15, 0.3, 0.08, -0.12, 0.25, -0.07, 0.18];
        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                kappa[(i, j)] = vals[idx % vals.len()];
                kappa[(j, i)] = -vals[idx % vals.len()];
                idx += 1;
            }
        }

        let u = cayley_rotation(&kappa).unwrap();
        // U^T U should be identity
        let utu = u.t().dot(&u);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (utu[(i, j)] - expected).abs() < 1e-12,
                    "U^T U[{},{}] = {}, expected {}",
                    i, j, utu[(i, j)], expected
                );
            }
        }
    }
}
