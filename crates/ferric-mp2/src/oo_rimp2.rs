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
//!
//! # The functional (textbook OMP2, since F2 of the 2026-09-24 validation campaign)
//!
//! `E(C) = E_HF(C) + E_MP2^RI(C)`, where the MP2 part is evaluated in the
//! SEMICANONICAL frame of `C`: the active occ-occ and vir-vir blocks of
//! `CᵀF(C)C` are diagonalised first (see `semicanonical_rotation`). That is
//! the full-Fock non-canonical Hylleraas functional of Bozkaya's OMP2 (Psi4
//! `omp2`), and it is invariant to occ-occ and vir-vir rotations, so the
//! occ-vir stationarity the solver enforces IS full stationarity — the property
//! the no-Z-vector nuclear gradient in `oo_rimp2_gradient` relies on.
//!
//! The previous functional read the orbital energies off `diag(CᵀFC)` in
//! whatever frame the solver happened to be in. It was not invariant to
//! occ-occ / vir-vir rotations, which the solver never optimises, so its
//! converged point carried a residual occ-occ/vir-vir gradient of ~1e-2
//! (H2O and NH3 cc-pVDZ) and no envelope theorem applied to it. Measured
//! (`scripts/oo_mp2_stationarity_proto.py gap`) it differs from the textbook
//! value by +3.2e-5 Ha (H2O/cc-pVDZ) and −9.8e-5 Ha (NH3/cc-pVDZ); the
//! textbook functional with exact integrals reproduces Psi4 `omp2-1` to
//! 1.2e-9 Ha (`... psi4`).

use crate::orbital_rotation::cayley_rotation;
use crate::rimp2::{active_occ, cholesky_inverse_sqrt};
use ferric_core::external_potential::ExternalPotential;
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
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
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
#[derive(Debug, Clone)]
#[must_use]
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
    /// Optimized MO coefficients, in the SEMICANONICAL frame (active occ-occ
    /// and vir-vir blocks of `CᵀFC` diagonal). The energy is invariant to that
    /// choice of frame; `oo_rimp2_gradient::oo_ri_mp2_gradient` relies on it.
    pub mos: Array2<f64>,
    /// Diagonal of `CᵀFC` for `mos`: the MP2 denominators actually used.
    pub orbital_energies: Vec<f64>,
}

impl std::fmt::Display for OoRiMp2Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OO-RI-MP2 total: {:.10} Ha ({} iters, converged: {})",
            self.total_energy, self.iterations, self.converged
        )
    }
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

impl std::fmt::Debug for OoRiMp2AoTensors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OoRiMp2AoTensors")
            .field("naux", &self.naux)
            .field("nao", &self.nao)
            .finish_non_exhaustive()
    }
}

impl OoRiMp2AoTensors {
    /// Build the AO-side invariants once, budget from `FERRIC_ERI3_BUDGET_GB`.
    pub fn build(
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
    ) -> Result<Self, FerricError> {
        Self::build_with_budget(
            obs,
            dfbs,
            op,
            ferric_core::memory::resolve_budget_bytes(None),
        )
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
        Ok(Self {
            v2c_inv_sqrt,
            eri3_ao: RefCell::new(src),
            naux,
            nao,
        })
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
    b_flat
        .into_shape_with_order((naux, nmo, nmo))
        .map_err(|e| FerricError::General(format!("compute_b_full_mo_with reshape: {e}")))
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
    let OrbitalSpace {
        nocc,
        nocc_total,
        first_occ,
        nvir,
    } = *orb;

    let c_occ = c
        .slice(ndarray::s![.., first_occ..first_occ + nocc])
        .to_owned();
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
    let sc =
        crate::rimp2::spin_components_from_b_ov(&b_flat, eps, nocc, nvir, first_occ, nocc_total);
    Ok((sc.e_total, b_flat))
}

/// Build the HF energy from MO coefficients + 1e integrals + J/K.
///
/// `ooc_budget` is the caller's solver-resolved memory budget (see
/// `rhf::resolve_three_index_budget`) — used ONLY to size the
/// `build_jk_with_pool` reduction band (`reduce::resolve_band_bytes`), never
/// affecting the result.
///
/// `vnn` is the FULL classical nuclear-repulsion-like constant: plain
/// `mol.nuclear_repulsion()` in vacuum, or (when an external potential is
/// present) that PLUS `ext.charge_nuclear_energy(mol) +
/// ext.field_nuclear_energy(mol)` — the same three-term sum
/// `ferric_scf::driver::prepare_scf_env` folds into its own `vnn` once before
/// the SCF loop (see that module). Passing plain `mol.nuclear_repulsion()`
/// here when `ext` is Some silently drops the classical charge-nuclear/
/// field-nuclear terms — this was a bug (found together with the hcore-only
/// bug this function's caller, `oo_ri_mp2`, fixes): the ~1.6 Ha spurious
/// energy lowering it produced is NOT a small missing correction, because an
/// UNCONSTRAINED Cayley orbital rotation can otherwise "discover" that
/// increasing the (uncompensated, missing) attractive charge-electron
/// one-electron term without the offsetting classical repulsion is
/// variationally favorable, collapsing well past any physical stationary
/// point. `vnn` must be computed ONCE by the caller (it does not depend on
/// the orbitals `c`) and threaded into every `compute_hf_energy` call for a
/// given `oo_ri_mp2`/`energy_at_kappa` run — never recomputed per call from
/// `mol` alone once `ext` is `Some`.
#[allow(clippy::too_many_arguments)]
fn compute_hf_energy(
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    c: &Array2<f64>,
    nocc_total: usize,
    h: &Array2<f64>,
    vnn: f64,
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
    build_jk_with_pool(
        &ctx, prep, bounds, 1e-12, &d, &mut j_mat, &mut k_mat, pool, band_bytes,
    )?;

    // F = H + J - 0.5*K
    let f = h + &j_mat - &(0.5 * &k_mat);

    // E_elec = 0.5 * tr(D * (H + F))
    let hpf = h + &f;
    let e_elec: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| 0.5 * d[(i, j)] * hpf[(i, j)])
        .sum();
    let e_hf = e_elec + vnn;

    Ok((e_hf, f, d))
}

/// Diagonal of `CᵀFC`, used ONLY as the diagonal-Hessian preconditioner of the
/// Newton step (`kappa_ai = -g_ai / (eps_a - eps_i + mu)`), in the solver's own
/// orbital frame.
///
/// These are NOT the MP2 denominators. Those come from [`semicanonical_rotation`]
/// (eigenvalues of the active occ-occ and vir-vir Fock blocks). Reading
/// denominators off this diagonal in a rotated frame was defect F2, the same
/// class as the Boys-screened `eps_loc` bug in `ferric-rpa/src/screen.rs`
/// (fixed 17e994e). "Small" off-diagonal Fock elements (~1e-3 at convergence)
/// did not make it harmless: they came with a ~1e-2 occ-occ/vir-vir gradient.
fn orbital_energies(c: &Array2<f64>, f: &Array2<f64>) -> Vec<f64> {
    let n = c.ncols();
    let f_mo = c.t().dot(f).dot(c);
    (0..n).map(|i| f_mo[(i, i)]).collect()
}

/// Symmetric eigendecomposition of one diagonal block `[lo, hi)` of `m`,
/// written into the same block of `u` (eigenvectors) and `eps` (eigenvalues,
/// ascending). The block is symmetrised first so that roundoff asymmetry in
/// `CᵀFC` cannot leak into the eigensolve.
fn eigh_block_into(
    m: &Array2<f64>,
    lo: usize,
    hi: usize,
    u: &mut Array2<f64>,
    eps: &mut [f64],
) -> Result<(), FerricError> {
    use ndarray_linalg::{Eigh, UPLO};
    if hi <= lo {
        return Ok(());
    }
    let blk = m.slice(ndarray::s![lo..hi, lo..hi]);
    let sym = 0.5 * (&blk + &blk.t());
    let (w, v) = sym
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("OO-RI-MP2 semicanonical eigh: {e}")))?;
    u.slice_mut(ndarray::s![lo..hi, lo..hi]).assign(&v);
    for (k, wk) in w.iter().enumerate() {
        eps[lo + k] = *wk;
    }
    Ok(())
}

/// Rotation to the SEMICANONICAL frame of an MO Fock matrix `f_mo = CᵀFC`:
/// the active occupied block `[first_occ, nocc_total)` and the virtual block
/// `[nocc_total, nmo)` are diagonalised separately. Frozen-core rows/columns
/// are left untouched (identity block, `eps = diag`).
///
/// Returns `(u, eps)` with `u` block-diagonal (so `C·u` spans the same
/// occupied and virtual spaces as `C`) and `eps` the orbital energies of
/// `C·u`: eigenvalues on the rotated blocks, the plain diagonal on the core.
///
/// Evaluating MP2 with these denominators in the frame `C·u` IS the full-Fock
/// (non-canonical) Hylleraas functional, so the resulting energy is invariant
/// to occ-occ / vir-vir rotations of `C` — see the module doc.
pub(crate) fn semicanonical_rotation(
    f_mo: &Array2<f64>,
    first_occ: usize,
    nocc_total: usize,
) -> Result<(Array2<f64>, Vec<f64>), FerricError> {
    let nmo = f_mo.nrows();
    let mut u = Array2::<f64>::eye(nmo);
    let mut eps: Vec<f64> = (0..nmo).map(|p| f_mo[(p, p)]).collect();
    eigh_block_into(f_mo, first_occ, nocc_total, &mut u, &mut eps)?;
    eigh_block_into(f_mo, nocc_total, nmo, &mut u, &mut eps)?;
    Ok((u, eps))
}

/// Löwdin (symmetric) re-orthonormalisation `C (CᵀSC)^{-1/2}`: the orthonormal
/// set closest to `C`.
///
/// Needed after DIIS extrapolation. DIIS on the MO coefficients returns a
/// LINEAR COMBINATION of coefficient matrices, which is not orthonormal. The
/// energy of a non-orthonormal `C` (non-idempotent `D = 2C_oC_oᵀ`) is not a
/// determinant energy, and the analytic orbital gradient assumes
/// orthonormality. Measured with a faithful emulation of this solver
/// (`scripts/oo_mp2_stationarity_proto.py solve`): without this step the
/// converged `C` is off orthonormality by 4.9e-5 (H2/cc-pVDZ) to 2.4e-5
/// (NH3/6-31G), which moved the converged energy by up to 7.8e-5 Ha.
pub(crate) fn lowdin_orthonormalize(
    c: &Array2<f64>,
    s: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{Eigh, UPLO};
    let m = c.t().dot(s).dot(c);
    let sym = 0.5 * (&m + &m.t());
    let (w, v) = sym
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("OO-RI-MP2 Löwdin eigh: {e}")))?;
    let mut v_scaled = v.clone();
    for (k, wk) in w.iter().enumerate() {
        if wk.is_nan() || *wk <= 0.0 {
            return Err(FerricError::General(format!(
                "OO-RI-MP2 Löwdin: MO metric CᵀSC is not positive definite (eigenvalue {wk:.3e}); \
                 the DIIS-extrapolated orbitals are linearly dependent"
            )));
        }
        let f = 1.0 / wk.sqrt();
        v_scaled.column_mut(k).mapv_inplace(|x| x * f);
    }
    Ok(c.dot(&v_scaled.dot(&v.t())))
}

/// Everything the solver needs at one set of orbitals `c` (the solver's own
/// frame), evaluated once.
struct OoPoint {
    e_hf: f64,
    f_ao: Array2<f64>,
    /// Block-diagonal rotation to the semicanonical frame: `c_sc = c · u_sc`.
    u_sc: Array2<f64>,
    c_sc: Array2<f64>,
    /// Orbital energies of `c_sc` (the MP2 denominators).
    eps_sc: Vec<f64>,
    e_mp2: f64,
    /// `(naux, nocc·nvir)` dressed OV B tensor in the semicanonical frame.
    b_ov_sc: Array2<f64>,
}

impl OoPoint {
    fn total(&self) -> f64 {
        self.e_hf + self.e_mp2
    }
}

/// Evaluate the textbook OMP2 functional at `c`: HF energy and Fock matrix,
/// semicanonical frame, and RI-MP2 in that frame.
#[allow(clippy::too_many_arguments)]
fn evaluate_point(
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    c: &Array2<f64>,
    h: &Array2<f64>,
    vnn: f64,
    pool: &EnginePool,
    ooc_budget: usize,
    ao: &OoRiMp2AoTensors,
    orb: &OrbitalSpace,
) -> Result<OoPoint, FerricError> {
    let (e_hf, f_ao, _d) =
        compute_hf_energy(obs, bounds, c, orb.nocc_total, h, vnn, pool, ooc_budget)?;
    let f_mo = c.t().dot(&f_ao).dot(c);
    let (u_sc, eps_sc) = semicanonical_rotation(&f_mo, orb.first_occ, orb.nocc_total)?;
    let c_sc = c.dot(&u_sc);
    let (e_mp2, b_ov_sc) = compute_rimp2_with_orbitals(ao, &c_sc, &eps_sc, orb)?;
    Ok(OoPoint {
        e_hf,
        f_ao,
        u_sc,
        c_sc,
        eps_sc,
        e_mp2,
        b_ov_sc,
    })
}

/// Orbital gradient `g_ai = +dE/dkappa_ai` at `point`, in the SOLVER's frame
/// (the frame of the `c` that `point` was evaluated at).
///
/// The gradient is computed in the semicanonical frame (`c_sc = c·u`) and
/// rotated back. Because `E` is invariant to occ-occ/vir-vir rotations,
/// `E(c·U(kappa)) = E(c_sc·U(uᵀ kappa u))`, so `g = U_v · g_sc · U_oᵀ` with
/// `U_o`/`U_v` the active-occupied / virtual blocks of `u`. Rotating back keeps
/// the DIIS error vector and the Newton step in one continuous frame (the
/// semicanonical frame itself jumps: eigenvector signs and order are
/// arbitrary).
///
/// The `b_full`/`t2` allocation is guarded by the CALLER, with
/// `check_b_full_and_t2_alloc` and the pool charge, exactly as before this
/// helper existed.
#[allow(clippy::too_many_arguments)]
fn orbital_gradient_solver_frame(
    point: &OoPoint,
    ao: &OoRiMp2AoTensors,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    pool: &EnginePool,
    orb: &OrbitalSpace,
    budget_bytes: usize,
    rss_probe: bool,
) -> Result<Array2<f64>, FerricError> {
    let OrbitalSpace {
        nocc,
        nocc_total,
        first_occ,
        nvir,
    } = *orb;
    let b_full = compute_b_full_mo_with(ao, &point.c_sc)?;
    // Stage-seam RSS safety net (observational only): `b_full` and the
    // amplitudes are the two largest allocations of an iteration.
    if rss_probe {
        ferric_core::memory::warn_if_rss_over("OO-RI-MP2 b_full built", budget_bytes, 1.1);
    }
    let naux = b_full.shape()[0];
    let (t2, _eri_ov) = compute_t2_and_integrals(
        &point.b_ov_sc,
        &point.eps_sc,
        nocc,
        nvir,
        nocc_total,
        first_occ,
        naux,
    );
    let f_mo_sc = point.c_sc.t().dot(&point.f_ao).dot(&point.c_sc);
    let g_sc = compute_orbital_gradient(
        &point.c_sc,
        &f_mo_sc,
        &t2,
        &b_full,
        nocc,
        nvir,
        first_occ,
        nocc_total,
        obs,
        bounds,
        pool,
        budget_bytes,
    )?;
    let u_o = point.u_sc.slice(ndarray::s![
        first_occ..first_occ + nocc,
        first_occ..first_occ + nocc
    ]);
    let u_v = point.u_sc.slice(ndarray::s![nocc_total.., nocc_total..]);
    Ok(u_v.dot(&g_sc).dot(&u_o.t()))
}

/// Orbital-energy (Fock) response of the MP2 energy to an occ-vir rotation,
/// the term the full-Fock Hylleraas functional adds on top of the fixed-
/// denominator integral response.
///
/// With `W = dE_MP2/dF_pq` the unrelaxed MP2 density (`P_oo + P_ooᵀ`,
/// `P_vv + P_vvᵀ` from [`build_mp2_density`], zero elsewhere) and the rotation
/// `dC_k = C_c`, `dC_c = -C_k` (so `dD = 2(C_c C_kᵀ + C_k C_cᵀ)`):
///
/// ```text
///   Σ_pq W_pq dF_pq/dkappa_ck
///     = 2·[(F·W)_ck − (W·F)_ck]            (orbital-rotation part)
///     + 4·[Cᵀ G[C W Cᵀ] C]_ck,  G[X] = J[X] − ½K[X]   (density part)
/// ```
///
/// The density part is ONE exact 4-centre J/K build on the pseudo-density
/// `C W Cᵀ`. It must be exact, not RI, because the Fock matrix whose response
/// this is (the one in `compute_hf_energy`) is exact. The shipped code built it
/// per `(c,k)` from RI integrals (`-4(pp|ck)_RI + 2(pc|pk)_RI` on the diagonal
/// only). That RI mismatch alone left a κ-independent error of 1.2e-6 to
/// 3.8e-5 Ha/rad at the canonical RHF point. The rotation part was dropped
/// entirely: it vanishes only where `F_ov = 0`, i.e. at the RHF start, and
/// never at an OO iterate.
#[allow(clippy::too_many_arguments)]
fn fock_response_gradient(
    c: &Array2<f64>,
    f_mo: &Array2<f64>,
    t2: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    pool: &EnginePool,
    ooc_budget: usize,
) -> Result<Array2<f64>, FerricError> {
    let nmo = c.ncols();
    let n = c.nrows();
    let (p_oo, p_vv) = build_mp2_density(t2, nocc, nvir);
    let mut w = Array2::<f64>::zeros((nmo, nmo));
    for i in 0..nocc {
        for j in 0..nocc {
            w[(first_occ + i, first_occ + j)] = p_oo[(i, j)] + p_oo[(j, i)];
        }
    }
    for a in 0..nvir {
        for b in 0..nvir {
            w[(nocc_total + a, nocc_total + b)] = p_vv[(a, b)] + p_vv[(b, a)];
        }
    }
    let w_ao = c.dot(&w).dot(&c.t());
    let mut j_mat = Array2::<f64>::zeros((n, n));
    let mut k_mat = Array2::<f64>::zeros((n, n));
    let ctx = ferric_core::parallel::ParallelContext::default();
    let band_bytes = ferric_scf::reduce::resolve_band_bytes(ooc_budget);
    build_jk_with_pool(
        &ctx, obs, bounds, 1e-12, &w_ao, &mut j_mat, &mut k_mat, pool, band_bytes,
    )?;
    let g_ao = &j_mat - &(0.5 * &k_mat);
    let g_mo = c.t().dot(&g_ao).dot(c);
    let fw = f_mo.dot(&w);
    let wf = w.dot(f_mo);
    let mut g = Array2::<f64>::zeros((nvir, nocc));
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            g[(a, i)] = 4.0 * g_mo[(a_mo, i_mo)] + 2.0 * (fw[(a_mo, i_mo)] - wf[(a_mo, i_mo)]);
        }
    }
    Ok(g)
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

/// Fail-fast pre-flight guard for the **full-MO** `b_full` tensor
/// [`compute_b_full_mo_with`] builds each OO-MP2 macro-iteration, plus the
/// co-resident `(nov, nov)` t2/eri_ov pair that follows it.
///
/// # The gap this closes
///
/// `compute_b_full_mo_with` allocates a dense `(naux, nmo, nmo)` f64 tensor and
/// carries NO guard of its own. At `naux=2200`/`nmo=960` that is ~16 GB — on
/// its own larger than a typical budget. Two separate accounting mistakes let
/// it through:
///
/// * The caller's `check_t2_pair_alloc` ran AFTER `compute_b_full_mo_with`
///   returned, so even the check that did exist could not refuse the job; the
///   allocation had already happened.
/// * Neither guard charged `b_full` at all. `check_t2_pair_alloc` charges the
///   `nov²` pair, and `check_gradient_intermediates_alloc` charges the SLICES
///   the gradient extracts out of `b_full` (`b_ov`, `b_vv`, `b_oo`, `b_diag`) —
///   never the parent tensor those slices are cut from, which stays live
///   alongside them for the whole gradient evaluation.
///
/// This charges `b_full` and the t2 pair together, because they ARE co-resident:
/// `b_full` is built first and then passed to `compute_orbital_gradient` after
/// `t2` exists. Callers must invoke it BEFORE `compute_b_full_mo_with`.
///
/// `budget_bytes` is the already-resolved
/// [`ferric_core::memory::resolve_budget_bytes`] value, same convention as
/// [`check_t2_pair_alloc`]. Pure arithmetic: never allocates.
fn check_b_full_and_t2_alloc(
    label: &str,
    nocc: usize,
    nvir: usize,
    naux: usize,
    nmo: usize,
    budget_bytes: usize,
) -> Result<(), FerricError> {
    let nov = nocc.saturating_mul(nvir);
    let b_full = naux.saturating_mul(nmo).saturating_mul(nmo);
    let t2_pair = nov.saturating_mul(nov).saturating_mul(2);
    let peak = b_full.saturating_add(t2_pair).saturating_mul(8);
    ferric_core::memory::check_alloc(
        &format!(
            "{label} (nocc={nocc}, nvir={nvir}, naux={naux}, nmo={nmo}; \
             b_full (naux·nmo²) co-resident with t2+eri_ov_mat (nov={nov})²)"
        ),
        peak,
        budget_bytes,
    )
}

/// Element count of the dense intermediates [`compute_orbital_gradient`]
/// holds co-resident, shared by its pre-flight guard and its pool charge:
///
/// * [`compute_orbital_gradient_panelled`], before its c-panel loop: `b_ov`
///   (naux×nov), `b_vv` (naux×nvir²), `b_oo` (naux×nocc²), `ooov` (nocc²×nov).
/// * [`fock_response_gradient`]: `P_oo`/`P_vv`, and eight nmo×nmo planes (`W`,
///   `C W Cᵀ`, `J`, `K`, `G`, `CᵀGC`, `F·W`, `W·F`).
///
/// The former `ovov` and `s_mat` (nov×nov each, ~23 GB apiece at
/// nocc≈60/nvir≈900) and `b_diag` (naux×nmo) belonged to the removed RI
/// orbital-energy-response term (defect F2) and are gone.
fn gradient_intermediate_elems(nocc: usize, nvir: usize, naux: usize, nmo: usize) -> usize {
    let nov = nocc.saturating_mul(nvir);
    let aux_elems = naux.saturating_mul(
        nov.saturating_add(nvir.saturating_mul(nvir))
            .saturating_add(nocc.saturating_mul(nocc)),
    ); // b_ov + b_vv + b_oo
    let ooov = nocc.saturating_mul(nocc).saturating_mul(nov);
    let fock_resp = nmo
        .saturating_mul(nmo)
        .saturating_mul(8)
        .saturating_add(nocc.saturating_mul(nocc))
        .saturating_add(nvir.saturating_mul(nvir));
    aux_elems.saturating_add(ooov).saturating_add(fock_resp)
}

/// Fail-fast pre-flight guard for the dense MO-space intermediates
/// [`compute_orbital_gradient`] allocates unconditionally (see
/// [`gradient_intermediate_elems`]). Unlike the VVOV transient (which the
/// panel loop budget-sizes), none of these is blocked, so the budget check has
/// to happen up front. The co-resident t2 pair in the caller is guarded by
/// [`check_t2_pair_alloc`] / [`check_b_full_and_t2_alloc`].
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
    let peak = gradient_intermediate_elems(nocc, nvir, naux, nmo).saturating_mul(8);
    ferric_core::memory::check_alloc(
        &format!("{label} (nocc={nocc}, nvir={nvir}, naux={naux}, nmo={nmo}; co-resident b_ov/b_vv/b_oo + ooov + Fock-response planes)"),
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
/// pair (see `check_t2_pair_alloc`) — `None` resolves via
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
pub fn build_mp2_density(t2: &[f64], nocc: usize, nvir: usize) -> (Array2<f64>, Array2<f64>) {
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

/// Compute the full OO-MP2 orbital gradient `g_ai = +dE/dkappa_ai` for
/// occupied-virtual rotations, at `kappa = 0` around the orbitals `c`.
///
/// PRECONDITION: `c`, `f_mo = cᵀFc`, `t2` and `b_full` are all in the
/// SEMICANONICAL frame of `c` (active occ-occ and vir-vir blocks of `f_mo`
/// diagonal, `t2` built with those diagonal elements as denominators). Callers
/// in the solver go through [`orbital_gradient_solver_frame`], which
/// establishes this. In that frame the diagonal-denominator formula IS the
/// full-Fock Hylleraas functional, and the gradient below is exact for ANY
/// such point: canonical RHF or an arbitrary OO iterate with `F_ov != 0`.
///
/// Three pieces (sign: canonical Cayley `U = (I−κ/2)⁻¹(I+κ/2) ≈ exp(κ)`, so
/// `dC_k = C_c`, `dC_c = −C_k` for `kappa_ck`):
///
/// ```text
///   g_ck = 4 F_ck                                          (HF, "Term0")
///        + 2 Σ_ijab T̃_ij^ab d(ia|jb)/dkappa_ck              (integral response,
///              T̃ = 2t − t^swap; Term1..Term4 below)          fixed denominators)
///        + Σ_pq W_pq dF_pq/dkappa_ck                        (Fock response,
///                                                             fock_response_gradient)
/// ```
///
/// with `d(ia|jb)/dkappa_ck = δ_ik (ca|jb) + δ_jk (ia|cb) − δ_ac (ik|jb) − δ_bc (ia|jk)`
/// and `W` the unrelaxed MP2 density.
///
/// # History (defect F2, 2026-09-24)
///
/// The previous version replaced the Fock response by a DIAGONAL
/// orbital-energy response `Σ_p W_pp d(eps_p)/dkappa_ck`, with
/// `d(eps_p)/dkappa_ck = 4(pp|ck) − 2(pc|pk)` built from RI integrals. That
/// formula is exact only at the canonical RHF point, and only up to the RI
/// error of the J/K build. Measured against 4-point FD of its own
/// functional (`scripts/oo_mp2_stationarity_proto.py grad`, reference
/// `C_RHF·U(kappa)` with random antisymmetric kappa):
///
/// | system     | κ=0     | ‖κ‖=0.05 | ‖κ‖=0.2 |
/// |------------|---------|----------|---------|
/// | H2/cc-pVDZ | 4.5e-6  | 4.9e-4   | 2.0e-3  |
/// | H2O/STO-3G | 1.2e-6  | 4.6e-3   | 1.9e-2  |
/// | NH3/6-31G  | 3.8e-5  | 1.7e-3   | 7.2e-3  |
///
/// The κ-proportional part is exactly the missing `2(F·W − W·F)_ck` rotation
/// term. The κ=0 offset is the RI-vs-exact J/K mismatch. This formula: 1e-11
/// to 4e-11 at every κ. Since the solver re-references every
/// macro-iteration, "κ≠0" is its normal operating point. The solver
/// therefore stopped 2.2e-4 Ha/rad (FD) from the stationary point of its own
/// functional on H2/cc-pVDZ while reporting |g| = 2e-8.
// Orbital-space sizes plus the integral context and budget are all
// irreducibly distinct inputs.
#[allow(clippy::too_many_arguments)]
fn compute_orbital_gradient(
    c: &Array2<f64>,
    f_mo: &Array2<f64>,
    t2: &[f64],
    b_full: &Array3<f64>,
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    pool: &EnginePool,
    budget_bytes: usize,
) -> Result<Array2<f64>, FerricError> {
    // `budget_bytes` is the caller's ALREADY-RESOLVED budget (oo_ri_mp2
    // resolves resolve_budget_bytes(config.memory_budget_bytes) once per call
    // and threads it here).
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
    // DEBIT the shared pool for the same dense intermediates the guard above
    // projects, held for this function's body -- which is exactly how long
    // they are resident.
    //
    // This is the plane whose charge MATTERS most for composition: the caller
    // is still holding `b_full` (naux*nmo^2) and the t2 pair when this runs,
    // and its own guard compared THOSE against the whole ceiling. Now the two
    // sum in one ledger.
    //
    // HARD: the buffers are allocated unconditionally and the pre-flight
    // immediately above already errors on the same bytes. The one plane on
    // this path that IS blocked -- the VVOV c-panel -- is sized separately
    // from `budget_bytes` just below and is not part of this charge.
    let _grad_charge = crate::rimp2::charge_mo_side(
        "OO-RI-MP2 gradient intermediates (b_ov/b_vv/b_oo + ooov + Fock-response W/J/K)",
        gradient_intermediate_elems(nocc, nvir, naux, nmo).saturating_mul(8),
    )?;
    // VVOV panel width from the resident-bytes budget: one c-value costs
    // nvir·nocc·nvir·8 bytes of VVOV rows. Unset budget = one full-width panel
    // (bit-identical to the former unblocked path).
    //
    // `budget_bytes`, NOT the pool's `available_bytes()`: a width may depend on
    // the budget and the problem, never on the ledger (see
    // `rimp2::mo_stream_chunk_for`).
    let nov = nocc * nvir;
    let row_bytes = nvir.saturating_mul(nov).saturating_mul(8).max(1);
    let panel_c = (budget_bytes / row_bytes).max(1).min(nvir.max(1));
    let mut g = compute_orbital_gradient_panelled(
        f_mo, t2, b_full, nocc, nvir, first_occ, nocc_total, panel_c,
    );
    g += &fock_response_gradient(
        c,
        f_mo,
        t2,
        nocc,
        nvir,
        first_occ,
        nocc_total,
        obs,
        bounds,
        pool,
        budget_bytes,
    )?;
    Ok(g)
}

/// HF + fixed-denominator integral-response part of [`compute_orbital_gradient`]
/// (everything except the Fock response), with an explicit VVOV c-panel width.
/// The panelled evaluation is exact for any `panel_c >= 1`: panels change the
/// memory shape only, never the contraction. It is split out so tests can force
/// multi-panel execution regardless of the environment budget, and so the
/// thread-count bit-identity test can run it without a J/K build.
// Orbital-space sizes plus the panel width are all irreducibly distinct.
#[allow(clippy::too_many_arguments)]
fn compute_orbital_gradient_panelled(
    f_mo: &Array2<f64>,
    t2: &[f64],
    b_full: &Array3<f64>,
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

    // MP2 integral response (fixed denominators):
    // dE_MP2/d kappa_{ck} = 2 * sum_{ijab} t_{ij,ab} * [2*d(ia|jb)/dk_{ck} - d(ib|ja)/dk_{ck}]
    //
    // Written below in the PRE-UNIFICATION Cayley sense
    // (dC_p/dk_{ck} = -delta_{pk} C_c + delta_{pc} C_k); the accumulated
    // `grad_ck` is therefore the NEGATIVE of the canonical-convention value,
    // and each row is stored as `2 * grad_ck`, whose sign matches the HF term
    // above (see the `*row_k` assignment).
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
                let mut row = vec![0.0_f64; nocc];
                for (k, row_k) in row.iter_mut().enumerate() {
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

                    *row_k = 2.0 * grad_ck;
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
/// to minimize E_HF + E_MP2 jointly (the textbook full-Fock OMP2 functional,
/// see the module doc).
///
/// The returned `mos` are in the SEMICANONICAL frame (active occ-occ and
/// vir-vir Fock blocks diagonal) and `orbital_energies` are their diagonal
/// Fock elements, i.e. the MP2 denominators actually used. This is what
/// `oo_rimp2_gradient::oo_ri_mp2_gradient` expects.
pub fn oo_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &OoRiMp2Config,
    ext: Option<&ExternalPotential>,
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

    // One-electron integrals (fixed). `ext = None` is byte-for-byte identical
    // to the pre-fix `oneelectron::hcore(obs)` call (see
    // `hcore_with_external`'s doc comment) — this was a bug, not a design
    // choice: OO-MP2 rebuilt the bare hcore here and at `energy_at_kappa`,
    // silently dropping any external potential the caller's `RhfConfig` set,
    // even though `rhf` (the starting-orbital SCF result passed in) was
    // itself solved WITH the potential. Every downstream `compute_hf_energy`
    // call in this function reuses this one `h`.
    //
    // ECP: `hcore_ecp_with_external` adds V_ECP (zero work, byte-identical
    // result for an all-electron basis). With plain `hcore_with_external` an
    // ECP molecule got V_nuc built from the ECP-REDUCED charges (and V_nn
    // from the same charges) but no V_ECP, i.e. a different Hamiltonian from
    // the one `rhf` was solved with: HI/def2-SVP started 50.46 Ha
    // (= tr(D V_ECP)) below its own RHF energy at zero rotation.
    let h = oneelectron::hcore_ecp_with_external(obs, mol, obs.basis_set(), ext)?;
    // Classical constant: plain nuclear repulsion in vacuum, or (with `ext`)
    // PLUS the charge-nuclear/field-nuclear terms — see `compute_hf_energy`'s
    // doc comment for why this must be threaded explicitly rather than
    // recomputed from `mol` alone (a second, independent bug from the hcore
    // one: an uncompensated one-electron attraction with no offsetting
    // classical repulsion let the unconstrained Cayley rotation collapse to a
    // spuriously low, unphysical stationary point).
    let vnn = mol.nuclear_repulsion()
        + ext
            .map(|e| e.charge_nuclear_energy(mol) + e.field_nuclear_energy(mol))
            .unwrap_or(0.0);
    // AO overlap, for re-orthonormalising DIIS-extrapolated orbitals.
    let s_ao = oneelectron::overlap(obs);

    // AO-side invariants: built once, reused every iteration + backtrack.
    // Thread the config budget (M1 resolver) rather than the env-only default.
    let ao = OoRiMp2AoTensors::build_with_budget(obs, dfbs, op, budget_bytes)?;

    // EnginePool is geometry/basis-only (density-independent) — build ONCE
    // here and reuse across every J/K build in the outer iteration and
    // backtracking loops below (this is the OO-MP2 orbital rotation loop:
    // (mol, obs, bounds) never change across iterations, only the orbital
    // coefficients C do), instead of build_jk constructing a fresh pool per
    // call. Reduction order is unchanged, so results stay bit-identical across
    // thread counts.
    let pool = EnginePool::new(bounds.op, obs, 1e-14)?;

    // Start from converged RHF orbitals. `c` is the solver's own frame; it is
    // kept continuous across iterations (DIIS mixes successive `c`s), while
    // every energy and gradient is evaluated in the semicanonical frame of `c`
    // (see `evaluate_point` / `orbital_gradient_solver_frame`).
    let mut c = rhf.mos_r().clone();
    let mut point = evaluate_point(obs, bounds, &c, &h, vnn, &pool, budget_bytes, &ao, &orb)?;
    let mut grad_norm = f64::MAX;
    let nmo = nbas;
    let max_kappa = 0.3; // cap individual rotation angles (radians)

    let finish =
        |point: OoPoint, converged: bool, iterations: usize, grad_norm: f64| OoRiMp2Result {
            total_energy: point.total(),
            hf_energy: point.e_hf,
            mp2_corr: point.e_mp2,
            converged,
            iterations,
            grad_norm,
            mos: point.c_sc,
            orbital_energies: point.eps_sc,
        };

    // DIIS for orbital rotation extrapolation.
    //
    // We apply DIIS to the MO coefficient matrix C itself (the state variable),
    // using the orbital gradient mapped into the AO basis as the error vector.
    // This mirrors how SCF DIIS works: the Fock matrix is replaced by C, and
    // the commutator error is replaced by the orbital gradient.  The gradient
    // is mapped to a (nbas, nbas) matrix G_AO = C * g_full * C^T where g_full
    // is the antisymmetric gradient in the MO basis (g_full[a,i] = g[a,i],
    // g_full[i,a] = -g[a,i]).  This ensures the DIIS error has the same shape
    // as the trial vector (C). The extrapolated C is a linear combination and
    // is Löwdin re-orthonormalised before use (see `lowdin_orthonormalize`).
    let mut diis = if config.use_diis {
        Some(Diis::new(config.diis_size))
    } else {
        None
    };

    for iter in 1..=config.max_iter {
        // Fail-fast guard BEFORE the first large allocation of the iteration.
        //
        // Two things are charged together because they are co-resident: the
        // full-MO `b_full` tensor built inside the gradient helper
        // (naux·nmo², ~16 GB at naux=2200/nmo=960) and the pair of `(nov,nov)`
        // buffers `compute_t2_and_integrals` returns there. See
        // `check_b_full_and_t2_alloc`.
        let naux = dfbs.nbasis();
        check_b_full_and_t2_alloc(
            &format!("OO-RI-MP2 b_full/t2/eri_ov (iter {iter})"),
            nocc,
            nvir,
            naux,
            nmo,
            budget_bytes,
        )?;

        // DEBIT the shared pool for b_full + the t2 pair, before b_full
        // allocates.
        //
        // The two `check_*_alloc` guards on this path cover DISJOINT subsets
        // and their sum is never charged: this one takes `b_full + t2 +
        // eri_ov`, and `check_gradient_intermediates_alloc` (inside
        // `compute_orbital_gradient`) takes the gradient intermediates. The
        // second runs while every buffer the first covers is STILL RESIDENT,
        // so whichever asks second now sees only what the first left.
        //
        // The guard is scoped to ONE macro-iteration of the enclosing `loop`:
        // `b_full` is rebuilt every iteration, so a charge that did not
        // release would exhaust the pool on iteration two.
        //
        // HARD: `compute_b_full_mo_with` builds a dense (naux, nmo^2) tensor
        // with no blocked alternative on this path, and the pre-flight
        // immediately above already refuses the same bytes against the
        // ceiling. This changes WHAT it is measured against, not whether it
        // can refuse.
        let nov_iter = nocc.saturating_mul(nvir);
        let b_full_charge = crate::rimp2::charge_mo_side(
            "OO-RI-MP2 b_full + t2 + eri_ov",
            naux.saturating_mul(nmo)
                .saturating_mul(nmo)
                .saturating_add(nov_iter.saturating_mul(nov_iter).saturating_mul(2))
                .saturating_mul(8),
        )?;

        // Orbital gradient in the solver's frame (b_full, t2 and the full 2e +
        // Fock response are built and dropped inside).
        let g = orbital_gradient_solver_frame(
            &point,
            &ao,
            obs,
            bounds,
            &pool,
            &orb,
            budget_bytes,
            iter == 1,
        )?;
        drop(b_full_charge);

        // Check gradient norm
        grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Live per-iteration progress (see RhfConfig.verbose's doc for the
        // full rationale). STDOUT, opt-in via `config.verbose`.
        if config.verbose {
            println!(
                "OO-RI-MP2 iter {:3}: E_HF={:.10} E_MP2={:.10} E_tot={:.10} |g|={:.2e}",
                iter,
                point.e_hf,
                point.e_mp2,
                point.total(),
                grad_norm
            );
        }

        if grad_norm < config.grad_conv {
            return Ok(finish(point, true, iter, grad_norm));
        }

        // Level-shifted approximate Newton step:
        //   kappa_{ai} = -g_{ai} / (eps_a - eps_i + mu)
        // The diagonal Hessian is the orbital energy gap (in the solver's
        // frame, where `g` lives); the level shift mu regularizes small gaps
        // (Bozkaya & Sherrill, JCP 135, 104103, 2011).
        let eps_step = orbital_energies(&c, &point.f_ao);
        let mut kappa_ov = Array2::zeros((nvir, nocc));
        for a in 0..nvir {
            for i in 0..nocc {
                let gap = eps_step[nocc_total + a] - eps_step[first_occ + i];
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
            // A linear combination of orbital sets is not orthonormal.
            c_new = lowdin_orthonormalize(&c_new, &s_ao)?;
        }

        // Evaluate energy at the new (possibly DIIS-extrapolated) orbitals
        let trial = evaluate_point(obs, bounds, &c_new, &h, vnn, &pool, budget_bytes, &ao, &orb)?;
        let total_new = trial.total();
        let de = (total_new - point.total()).abs();

        // Backtracking if energy increased by more than a small tolerance.
        // DIIS can produce small uphill steps; we tolerate those.
        if total_new > point.total() + 1e-4 {
            // Fall back to a damped Newton step without DIIS extrapolation.
            let mut bt_kappa_ov = kappa_ov.clone();
            let mut bt_c = c.dot(&u);
            // The rejected trial seeds the backtrack; each step frees the
            // previous point BEFORE evaluating the next, so at most `point`
            // plus one backtrack amplitude set are resident.
            let mut bt_point = Some(trial);

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
                drop(bt_point.take());
                let p =
                    evaluate_point(obs, bounds, &bt_c, &h, vnn, &pool, budget_bytes, &ao, &orb)?;
                let accepted = p.total() <= point.total() + 1e-12;
                bt_point = Some(p);
                if accepted {
                    break;
                }
            }

            // The backtracking loop commits bt_* to its last trial step on every
            // path (break or exhaustion), so bt_c is always the step to take.
            c = bt_c;
            point = bt_point.expect("the backtracking loop evaluates at least one point");

            // Reset DIIS after backtracking since the extrapolated
            // subspace produced an uphill step.
            if let Some(ref mut diis_obj) = diis {
                diis_obj.reset();
            }
        } else {
            // Accept the (possibly DIIS-extrapolated) step
            c = c_new;
            point = trial;
        }

        if de < config.energy_conv && iter > 1 {
            // Energy converged; recompute gradient to check convergence.
            // Same co-resident b_full/t2/eri_ov guard as the main-loop call
            // site above — this is a second, independent build of both, and it
            // likewise must run BEFORE `compute_b_full_mo_with`.
            let naux2 = dfbs.nbasis();
            check_b_full_and_t2_alloc(
                &format!("OO-RI-MP2 b_full/t2/eri_ov (iter {iter}, convergence recheck)"),
                nocc,
                nvir,
                naux2,
                nmo,
                budget_bytes,
            )?;
            let g2 = orbital_gradient_solver_frame(
                &point,
                &ao,
                obs,
                bounds,
                &pool,
                &orb,
                budget_bytes,
                false,
            )?;
            grad_norm = g2.iter().map(|x| x * x).sum::<f64>().sqrt();

            if grad_norm < config.grad_conv * 10.0 {
                return Ok(finish(point, true, iter, grad_norm));
            }
        }
    }

    Ok(finish(point, false, config.max_iter, grad_norm))
}

/// Compute RI-MP2 energy for orbitals rotated by kappa (for finite-difference testing).
///
/// Takes initial MO coefficients, applies a Cayley rotation with the given kappa,
/// rebuilds Fock / density, and returns E_HF + E_MP2 — the same textbook OMP2
/// functional (semicanonical-frame MP2) `oo_ri_mp2` optimises. Vacuum only (no
/// external potential; V_ECP is included when the basis carries ECPs, as in
/// `oo_ri_mp2`).
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
    let h = oneelectron::hcore_ecp(obs, mol, obs.basis_set());
    let vnn = mol.nuclear_repulsion();
    let ao = OoRiMp2AoTensors::build(obs, dfbs, op)?;
    let u = cayley_rotation(kappa)?;
    let c_rot = c_init.dot(&u);
    let pool = EnginePool::new(bounds.op, obs, 1e-14)?;
    let point = evaluate_point(
        obs,
        bounds,
        &c_rot,
        &h,
        vnn,
        &pool,
        ferric_core::memory::resolve_budget_bytes(None),
        &ao,
        orb,
    )?;
    Ok(point.total())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    fn setup_h2() -> (
        Molecule,
        PreparedBasis,
        PreparedBasis,
        Operator,
        SchwarzBounds,
        ScfResult,
    ) {
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
        assert!(
            check_t2_pair_alloc("test small OO-MP2 (auto budget)", nocc, nvir, auto_budget).is_ok()
        );
    }

    /// Memory-guard regression for the gradient-intermediates guard (Finding 2
    /// sibling of `check_t2_pair_alloc_rejects_realistic_large_scale`): at the
    /// same production-scale shape (nocc=60, nvir=900 → nov=54,000; b_vv
    /// alone is naux·nvir²·8 ≈ 19 GB at naux=3000, plus b_ov/b_oo and ooov)
    /// the guard must fire against a small budget, naming the label and the
    /// budget so an operator can act on it.
    /// Pure arithmetic — allocates nothing.
    #[test]
    fn check_gradient_intermediates_alloc_rejects_realistic_large_scale() {
        let nocc = 60;
        let nvir = 900;
        let naux = 3000; // realistic RI aux dimension at this orbital scale
        let nmo = nocc + nvir; // no frozen core: nocc_total + nvir
        let budget_bytes = ferric_core::memory::gib_to_bytes(1.0); // 1 GiB — tiny vs ~22 GB
        let err = check_gradient_intermediates_alloc(
            "OO-RI-MP2 gradient intermediates",
            nocc,
            nvir,
            naux,
            nmo,
            budget_bytes,
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
            5,
            19,
            84,
            24,
            budget_bytes,
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
        let (_e_hf, f_ao, _) = compute_hf_energy(
            &obs,
            &bounds,
            c,
            nocc_total,
            &h,
            _mol.nuclear_repulsion(),
            &pool,
            ferric_core::memory::resolve_budget_bytes(None),
        )
        .unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e, b_flat) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();

        // Small H2/cc-pVDZ scale must pass under a generous explicit budget.
        let ok = compute_t2_only(
            &b_flat,
            &eps,
            nocc,
            nvir,
            nocc_total,
            first_occ,
            naux,
            Some(ferric_core::memory::gib_to_bytes(1.0)),
        );
        assert!(
            ok.is_ok(),
            "small-system compute_t2_only unexpectedly rejected: {:?}",
            ok.err()
        );

        // A tiny budget must reject even this small system (proves the guard
        // is actually wired into compute_t2_only, not a no-op).
        let tiny_budget = 100usize; // 100 bytes -- far below even this tiny nov²
        let err = compute_t2_only(
            &b_flat,
            &eps,
            nocc,
            nvir,
            nocc_total,
            first_occ,
            naux,
            Some(tiny_budget),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("budget is"),
            "expected a budget-shaped error message, got: {msg}"
        );
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
        let (_e_hf, f_ao, _) = compute_hf_energy(
            &obs,
            &bounds,
            c,
            nocc_total,
            &h,
            mol.nuclear_repulsion(),
            &pool,
            ferric_core::memory::resolve_budget_bytes(None),
        )
        .unwrap();
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
                        let eri_iajb: f64 = (0..naux.min(b_flat.nrows()))
                            .map(|p| b_flat[(p, ia)] * b_flat[(p, jb)])
                            .sum();
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
            None,
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

    // ------------------------------------------------------------------
    // Orbital-gradient / stationarity harness (defect F2, 2026-09-24).
    //
    // Reference numbers quoted below come from
    // `scripts/oo_mp2_stationarity_proto.py`, an independent numpy rebuild of
    // this functional on PySCF integrals (which reproduces Psi4 omp2-1 to
    // 1.2e-9 Ha with exact integrals, and reproduced this module's pre-fix
    // H2O/STO-3G κ=0 error, 1.24e-6, digit for digit).
    //
    // BAR DERIVATION (all FD tests below): 4-point central stencil, step
    // 1e-3, whose truncation error is O(h⁴) ~ 1e-12. Its roundoff is
    // (J/K screening noise ~1e-12 Ha) / h ~ 1e-9. The corrected formula
    // measured 1e-11..4e-11 against the same stencil in the prototype. The
    // bar is 1e-7: >= 100x the expected FD floor, and 12x below the SMALLEST
    // pre-fix error (1.24e-6, H2O/STO-3G at κ=0).
    // ------------------------------------------------------------------

    const FD_STEP: f64 = 1e-3;
    const ORB_GRAD_BAR: f64 = 1e-7;

    /// Pre-built context for orbital-gradient FD tests, so each FD energy
    /// evaluation reuses the AO 3-index tensor and engine pool.
    struct OrbFd {
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        op: Operator,
        bounds: SchwarzBounds,
        rhf: ScfResult,
        mol: Molecule,
        ao: OoRiMp2AoTensors,
        pool: EnginePool,
        h: Array2<f64>,
        orb: OrbitalSpace,
    }

    impl OrbFd {
        fn new(xyz: &str, basis_name: &str) -> Self {
            let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
            let bs = basis::bundled(basis_name).unwrap();
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
                    energy_conv: 1e-11,
                    density_conv: 1e-10,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(rhf.converged);
            let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
            let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
            let ao = OoRiMp2AoTensors::build_with_budget(&obs, &dfbs, op, usize::MAX).unwrap();
            let pool = EnginePool::new(op, &obs, 1e-14).unwrap();
            let h = oneelectron::hcore(&obs);
            let nbas = obs.nbasis();
            let nocc_total = mol.nelec() as usize / 2;
            let orb = OrbitalSpace::new(nocc_total, nbas - nocc_total, nocc_total, 0);
            OrbFd {
                obs,
                dfbs,
                op,
                bounds,
                rhf,
                mol,
                ao,
                pool,
                h,
                orb,
            }
        }

        fn point(&self, c: &Array2<f64>) -> OoPoint {
            evaluate_point(
                &self.obs,
                &self.bounds,
                c,
                &self.h,
                self.mol.nuclear_repulsion(),
                &self.pool,
                usize::MAX,
                &self.ao,
                &self.orb,
            )
            .unwrap()
        }

        /// The production gradient path, exactly as `oo_ri_mp2` calls it.
        fn analytic(&self, c: &Array2<f64>) -> Array2<f64> {
            orbital_gradient_solver_frame(
                &self.point(c),
                &self.ao,
                &self.obs,
                &self.bounds,
                &self.pool,
                &self.orb,
                usize::MAX,
                false,
            )
            .unwrap()
        }

        /// 4-point central FD of E(c·U(kappa)) in every occ-vir direction.
        fn fd(&self, c: &Array2<f64>) -> Array2<f64> {
            let OrbitalSpace {
                nocc,
                nvir,
                nocc_total,
                first_occ,
            } = self.orb;
            let n = c.ncols();
            let mut g = Array2::zeros((nvir, nocc));
            for a in 0..nvir {
                for i in 0..nocc {
                    let e_at = |s: f64| {
                        let mut k = Array2::zeros((n, n));
                        k[(nocc_total + a, first_occ + i)] = s * FD_STEP;
                        k[(first_occ + i, nocc_total + a)] = -s * FD_STEP;
                        self.point(&c.dot(&cayley_rotation(&k).unwrap())).total()
                    };
                    g[(a, i)] = (-e_at(2.0) + 8.0 * e_at(1.0) - 8.0 * e_at(-1.0) + e_at(-2.0))
                        / (12.0 * FD_STEP);
                }
            }
            g
        }
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        (a - b).iter().map(|v| v.abs()).fold(0.0, f64::max)
    }

    /// Deterministic dense antisymmetric kappa over ALL blocks (occ-occ,
    /// vir-vir and occ-vir) with Frobenius norm `norm`, from a 64-bit LCG. A
    /// full kappa makes the reference `C_RHF·U(kappa)` both non-Brillouin
    /// (F_ov != 0) and non-canonical in its occ-occ/vir-vir blocks, which is
    /// the state every OO macro-iteration after the first is in.
    fn pseudo_random_kappa(n: usize, norm: f64, seed: u64) -> Array2<f64> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let mut k = Array2::<f64>::zeros((n, n));
        for p in 0..n {
            for q in 0..p {
                let v = next();
                k[(p, q)] = v;
                k[(q, p)] = -v;
            }
        }
        let fro = k.iter().map(|v| v * v).sum::<f64>().sqrt();
        k * (norm / fro)
    }

    const H2O_XYZ: &str = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
    const NH3_XYZ: &str = "4\nammonia\nN 0.0000 0.0000 0.1162\nH 0.0000 0.9397 -0.2711\nH 0.8138 -0.4699 -0.2711\nH -0.8138 -0.4699 -0.2711\n";
    const H2_XYZ: &str = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";

    /// Analytic orbital gradient vs FD at `C_RHF·U(kappa)`, ‖kappa‖_F = `norm`.
    fn check_orbital_gradient_at(xyz: &str, basis_name: &str, norm: f64, label: &str) {
        let ctx = OrbFd::new(xyz, basis_name);
        let c0 = ctx.rhf.mos_r().clone();
        let c = if norm == 0.0 {
            c0
        } else {
            c0.dot(&cayley_rotation(&pseudo_random_kappa(c0.ncols(), norm, 7)).unwrap())
        };
        let an = ctx.analytic(&c);
        let fd = ctx.fd(&c);
        let err = max_abs_diff(&an, &fd);
        let gmax = an.iter().map(|v| v.abs()).fold(0.0, f64::max);
        eprintln!("{label}: ‖kappa‖={norm}: max|g|={gmax:.3e}, max|analytic − FD|={err:.3e}");
        // TEETH: at ‖kappa‖ > 0 the point must actually be off-stationary, or
        // agreement is vacuous.
        if norm > 0.0 {
            assert!(
                gmax > 1e-3,
                "{label}: reference point is already stationary (max|g|={gmax:.2e})"
            );
        }
        assert!(
            err < ORB_GRAD_BAR,
            "{label} ‖kappa‖={norm}: max|analytic − FD| = {err:.3e} ≥ {ORB_GRAD_BAR:e}"
        );
    }

    /// κ = 0 around canonical RHF, H2/cc-pVDZ (nocc = 1).
    ///
    /// MUTATION NOTE: the pre-2026-07-20 formula (no denominator response)
    /// missed here by 3.4e-4, and the 2026-07-20 RI diagonal response by
    /// 4.5e-6. Both passed the old 1e-3 bar; both fail this one.
    #[test]
    fn test_oo_rimp2_gradient_finite_difference() {
        check_orbital_gradient_at(H2_XYZ, "cc-pvdz", 0.0, "H2/cc-pVDZ");
    }

    /// κ = 0 around canonical RHF, H2O/STO-3G (nocc = 5).
    ///
    /// MUTATION NOTE: the 2026-07-20 RI diagonal orbital-energy response
    /// missed here by 1.24e-6 (measured in Rust AND reproduced by the
    /// prototype), which is 12x this bar. At κ = 0, F_ov = 0, so a missing
    /// `2(F·W − W·F)` rotation term is INVISIBLE to this test. That is why the
    /// κ ≠ 0 tests below exist.
    #[test]
    fn test_oo_rimp2_gradient_finite_difference_h2o_sto3g() {
        check_orbital_gradient_at(H2O_XYZ, "sto-3g", 0.0, "H2O/STO-3G");
    }

    /// κ ≠ 0: the reference `C_RHF·U(kappa)` has F_ov != 0 and non-canonical
    /// occ-occ/vir-vir blocks, which is the solver's normal operating point.
    ///
    /// MUTATION NOTE (prototype, shipped formula vs its own functional):
    /// H2O/STO-3G missed by 4.6e-3 at ‖κ‖=0.05 and 1.9e-2 at 0.2, growing
    /// linearly with ‖κ‖ and equal to the dropped rotation term.
    #[test]
    fn test_oo_rimp2_gradient_fd_off_reference_h2o_sto3g() {
        check_orbital_gradient_at(H2O_XYZ, "sto-3g", 0.05, "H2O/STO-3G");
        check_orbital_gradient_at(H2O_XYZ, "sto-3g", 0.2, "H2O/STO-3G");
    }

    /// κ ≠ 0 on a second, larger system (nocc = 5, nvir = 10).
    ///
    /// MUTATION NOTE (prototype): the shipped formula missed by 3.8e-5 at
    /// κ = 0 (RI-vs-exact J/K), 1.7e-3 at ‖κ‖=0.05 and 7.2e-3 at 0.2.
    #[test]
    fn test_oo_rimp2_gradient_fd_off_reference_nh3_631g() {
        check_orbital_gradient_at(NH3_XYZ, "6-31g", 0.0, "NH3/6-31G");
        check_orbital_gradient_at(NH3_XYZ, "6-31g", 0.05, "NH3/6-31G");
        check_orbital_gradient_at(NH3_XYZ, "6-31g", 0.2, "NH3/6-31G");
    }

    /// The solver's converged point must be a stationary point of the energy
    /// it reports, checked by FD that is independent of the solver's own
    /// gradient. The returned orbitals must also be orthonormal and
    /// semicanonical (the contract `oo_ri_mp2_gradient` relies on).
    ///
    /// BAR: grad_conv 1e-8, and the energy_conv early exit accepts
    /// |g| < 10·grad_conv, so max|g_ai| ≤ 1e-7 analytically, and FD agrees with
    /// analytic to ≤ ORB_GRAD_BAR. So max|g_FD| < 1e-6 has ≥ 5x margin.
    ///
    /// MUTATION NOTE: a faithful emulation of the shipped solver
    /// (`... solve`) converged H2/cc-pVDZ to analytic |g| = 2.2e-8 with FD
    /// 2.18e-4. That is the value `test_oo_rimp2_converged_gradient_vanishes_h2_ccpvdz`
    /// used to pass under its 1e-3 bar. The same emulation left CᵀSC − 1 at
    /// 4.9e-5 (H2) / 6.5e-7 (H2O/STO-3G) / 2.4e-5 (NH3/6-31G) through DIIS on C.
    fn check_converged_point(xyz: &str, basis_name: &str, label: &str) {
        let ctx = OrbFd::new(xyz, basis_name);
        let config = OoRiMp2Config {
            grad_conv: 1e-8,
            energy_conv: 1e-11,
            max_iter: 200,
            ..Default::default()
        };
        let oo = oo_ri_mp2(
            &ctx.mol,
            &ctx.obs,
            &ctx.dfbs,
            ctx.op,
            &ctx.bounds,
            &ctx.rhf,
            &config,
            None,
        )
        .unwrap();
        assert!(
            oo.converged,
            "{label}: OO-RI-MP2 did not converge ({} iters, |g|={:.2e})",
            oo.iterations, oo.grad_norm
        );
        let c = &oo.mos;
        let n = c.ncols();
        let OrbitalSpace { nocc_total, .. } = ctx.orb;

        // Orthonormal.
        let s = oneelectron::overlap(&ctx.obs);
        let ortho = max_abs_diff(&c.t().dot(&s).dot(c), &Array2::eye(n));
        // Semicanonical, with orbital_energies == diag(CᵀFC).
        let point = ctx.point(c);
        assert!((point.total() - oo.total_energy).abs() < 1e-10);
        let f_mo = c.t().dot(&point.f_ao).dot(c);
        let mut off_oo_vv = 0.0f64;
        let mut max_fov = 0.0f64;
        let mut eps_err = 0.0f64;
        for p in 0..n {
            eps_err = eps_err.max((f_mo[(p, p)] - oo.orbital_energies[p]).abs());
            for q in 0..n {
                let same_block = (p < nocc_total) == (q < nocc_total);
                if p != q && same_block {
                    off_oo_vv = off_oo_vv.max(f_mo[(p, q)].abs());
                } else if !same_block {
                    max_fov = max_fov.max(f_mo[(p, q)].abs());
                }
            }
        }
        let fd = ctx.fd(c);
        let an = ctx.analytic(c);
        let fd_max = fd.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let err = max_abs_diff(&an, &fd);
        eprintln!(
            "{label}: iters={} E={:.12} |CᵀSC−1|={ortho:.2e} max|F_oo/vv offdiag|={off_oo_vv:.2e} \
             max|F_ov|={max_fov:.2e} max|g_FD|={fd_max:.2e} max|an−FD|={err:.2e}",
            oo.iterations, oo.total_energy
        );
        assert!(
            ortho < 1e-10,
            "{label}: returned MOs not orthonormal: {ortho:.2e}"
        );
        assert!(
            off_oo_vv < 1e-8,
            "{label}: returned MOs not semicanonical: {off_oo_vv:.2e}"
        );
        assert!(
            eps_err < 1e-10,
            "{label}: orbital_energies != diag(CᵀFC): {eps_err:.2e}"
        );
        // TEETH: OO must have moved off the Brillouin point (F_ov = 0 at RHF).
        assert!(
            max_fov > 1e-4,
            "{label}: F_ov ~ 0 ({max_fov:.2e}); OO did not rotate"
        );
        assert!(
            fd_max < 1e-6,
            "{label}: FD gradient at convergence {fd_max:.3e} ≥ 1e-6"
        );
        assert!(
            err < ORB_GRAD_BAR,
            "{label}: analytic vs FD at convergence {err:.3e}"
        );
    }

    #[test]
    fn test_oo_rimp2_converged_gradient_vanishes_h2_ccpvdz() {
        check_converged_point(H2_XYZ, "cc-pvdz", "H2/cc-pVDZ");
    }

    #[test]
    fn test_oo_rimp2_converged_gradient_vanishes_h2o_sto3g() {
        check_converged_point(H2O_XYZ, "sto-3g", "H2O/STO-3G");
    }

    #[test]
    fn test_oo_rimp2_converged_gradient_vanishes_h2o_ccpvdz() {
        check_converged_point(H2O_XYZ, "cc-pvdz", "H2O/cc-pVDZ");
    }

    #[test]
    fn test_oo_rimp2_converged_gradient_vanishes_nh3_ccpvdz() {
        check_converged_point(NH3_XYZ, "cc-pvdz", "NH3/cc-pVDZ");
    }

    /// The energy is the textbook (full-Fock) OMP2 functional, so it must be
    /// INVARIANT to occ-occ and vir-vir rotations of the orbitals. The former
    /// diag(CᵀFC) functional was not, and its dE/dkappa in those directions was
    /// 5e-3..1.5e-2 at convergence (H2O, NH3 cc-pVDZ; `... gap`).
    ///
    /// MUTATION NOTE: evaluating MP2 with `orbital_energies(c, f)` (the old
    /// denominators) instead of the semicanonical ones changes the energy
    /// under the oo/vv rotation below by O(‖κ‖²·spread of F diag). At
    /// ‖κ‖=0.2 that is ≫ 1e-10.
    #[test]
    fn test_oo_energy_is_invariant_to_occ_occ_and_vir_vir_rotations() {
        let ctx = OrbFd::new(H2O_XYZ, "sto-3g");
        let c0 = ctx.rhf.mos_r().clone();
        let n = c0.ncols();
        let nocc = ctx.orb.nocc_total;
        // Start from an off-Brillouin point, then rotate within blocks only.
        let c = c0.dot(&cayley_rotation(&pseudo_random_kappa(n, 0.1, 3)).unwrap());
        let mut k_blk = pseudo_random_kappa(n, 0.2, 11);
        for p in 0..n {
            for q in 0..n {
                if (p < nocc) != (q < nocc) {
                    k_blk[(p, q)] = 0.0;
                }
            }
        }
        let e_a = ctx.point(&c).total();
        let e_b = ctx.point(&c.dot(&cayley_rotation(&k_blk).unwrap())).total();
        eprintln!(
            "E(C) = {e_a:.12}, E(C·U_oo/vv) = {e_b:.12}, diff {:.2e}",
            e_a - e_b
        );
        assert!(
            (e_a - e_b).abs() < 1e-10,
            "oo/vv rotation changed E by {:.3e}",
            e_a - e_b
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
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &config, None).unwrap();

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
            oo.total_energy,
            ri.total_energy
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
    ///
    /// Two assertions (defect F2, 2026-09-24):
    ///
    /// 1. The SAME method with the same aux basis, rebuilt independently in
    ///    `scripts/oo_mp2_stationarity_proto.py psi4` on PySCF integrals, gives
    ///    E = -76.2316624303 Ha. That rebuild with EXACT integrals reproduces
    ///    Psi4 to 1.2e-9 Ha. ferric must match the RI rebuild to 1e-6 (the
    ///    floor is ferric-vs-PySCF basis/constant drift, ~4e-8 in the NRE alone).
    /// 2. Vs Psi4 itself: the RI error of the rebuild is 1.36e-5 Ha, so the
    ///    bar is 3e-5.
    ///
    /// MUTATION NOTE: the pre-F2 solver (diag-Fock functional, non-orthonormal
    /// DIIS output, gradient exact only at the RHF point) landed 1.32e-4 from
    /// Psi4 and passed the former 2e-4 bar. It fails both bars above: the
    /// prototype's diag-functional value alone is 3.2e-5 from the textbook
    /// value.
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
            rhf.energy,
            psi4_refscf,
            (rhf.energy - psi4_refscf).abs()
        );

        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let config = OoRiMp2Config {
            grad_conv: 1e-7,
            energy_conv: 1e-11,
            max_iter: 200,
            ..Default::default()
        };
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &config, None).unwrap();
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
        let prototype_ri_total = -76.2316624303_f64;
        let d_proto = oo.total_energy - prototype_ri_total;
        eprintln!("  vs independent RI rebuild (same aux): {d_proto:.3e}");
        assert!(
            d_proto.abs() < 1e-6,
            "ferric OO-RI-MP2={:.10} vs independent RI-OMP2 rebuild={prototype_ri_total:.10}: \
             diff={d_proto:.3e} Ha",
            oo.total_energy
        );
        assert!(
            diff.abs() < 3e-5,
            "ferric OO-RI-MP2={:.10} vs Psi4 OMP2={:.10}: diff={:.3e} Ha exceeds 3e-5 \
             (the RI error of the same functional is 1.36e-5 Ha)",
            oo.total_energy,
            psi4_omp2_total,
            diff
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
        let maxdiff = (&b_ref - &b_spill)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(
            maxdiff < 1e-12,
            "b_full spill vs in-core maxdiff={maxdiff:.2e}"
        );

        // MP2 energy + b_ov identical.
        let pool = EnginePool::new(op, &obs, 1e-14).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(
            &obs,
            &bounds,
            c,
            nocc_total,
            &h,
            mol.nuclear_repulsion(),
            &pool,
            ferric_core::memory::resolve_budget_bytes(None),
        )
        .unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (e_ref, bov_ref) = compute_rimp2_with_orbitals(&ao_ref, c, &eps, &orb).unwrap();
        let (e_spill, bov_spill) = compute_rimp2_with_orbitals(&ao_spill, c, &eps, &orb).unwrap();
        assert!(
            (e_ref - e_spill).abs() < 1e-12,
            "E_MP2 spill vs in-core: {:.3e}",
            (e_ref - e_spill).abs()
        );
        let bovdiff = (&bov_ref - &bov_spill)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(
            bovdiff < 1e-12,
            "b_ov spill vs in-core maxdiff={bovdiff:.2e}"
        );
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
        let (_e_hf, f_ao, _) = compute_hf_energy(
            &obs,
            &bounds,
            c,
            nocc_total,
            &h,
            mol.nuclear_repulsion(),
            &pool,
            ferric_core::memory::resolve_budget_bytes(None),
        )
        .unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e, b_ov) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();
        let (t2, _) =
            compute_t2_and_integrals(&b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux);
        let b_full = compute_b_full_mo_with(&ao, c).unwrap();
        let f_mo = c.t().dot(&f_ao).dot(c);

        let g_full = compute_orbital_gradient_panelled(
            &f_mo, &t2, &b_full, nocc, nvir, first_occ, nocc_total, nvir,
        );
        for panel in [1usize, 2, 3] {
            let g_p = compute_orbital_gradient_panelled(
                &f_mo, &t2, &b_full, nocc, nvir, first_occ, nocc_total, panel,
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
        let b_full_ref = b_full_ref_flat
            .into_shape_with_order((naux, nmo, nmo))
            .unwrap();

        let b_full_maxdiff = (&b_full - &b_full_ref)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
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
        let b_ov_maxdiff = (&b_ov - &b_ov_ref)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
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
        let (_e_hf, f_ao, _) = compute_hf_energy(
            &obs,
            &bounds,
            c,
            nocc_total,
            &h,
            mol.nuclear_repulsion(),
            &pool,
            ferric_core::memory::resolve_budget_bytes(None),
        )
        .unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e, b_ov) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();
        let (t2, _) =
            compute_t2_and_integrals(&b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux);
        let b_full = compute_b_full_mo_with(&ao, c).unwrap();
        let f_mo = c.t().dot(&f_ao).dot(c);

        // Force a multi-panel width (panel_c=3) so the parallel region runs
        // inside more than one GEMM panel, exercising the interaction of
        // panelling and rayon scheduling together.
        let run_with_threads = |n: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap();
            pool.install(|| {
                compute_orbital_gradient_panelled(
                    &f_mo, &t2, &b_full, nocc, nvir, first_occ, nocc_total, 3,
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
                    g1[(a, i)],
                    g1[(a, i)].to_bits(),
                    g4[(a, i)],
                    g4[(a, i)].to_bits(),
                );
                assert_eq!(
                    g1[(a, i)].to_bits(),
                    g8[(a, i)].to_bits(),
                    "OO gradient not bit-identical 1 vs 8 threads at (a={a}, i={i}): \
                     1={:.17e} (0x{:016x}), 8={:.17e} (0x{:016x})",
                    g1[(a, i)],
                    g1[(a, i)].to_bits(),
                    g8[(a, i)],
                    g8[(a, i)].to_bits(),
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
                    i,
                    j,
                    utu[(i, j)],
                    expected
                );
            }
        }
    }
}
