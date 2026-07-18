//! Resolution-of-identity MP2 (RI-MP2 / density-fitted MP2).
//!
//! Approximates the 4-center ERIs using density fitting:
//! (ia|jb) ~ sum_P B^P_ia * B^P_jb, where B^P_ia = sum_Q V^{-1/2}_PQ (Q|ia).
//!
//! This reduces the MO integral transformation from O(N^5) to O(N^4) with
//! a controllable RI approximation error that is negligible for matched
//! auxiliary basis sets.

use crate::mo_transform::transform_3center_ov;
use ferric_core::mol::Molecule;
use ferric_core::orbitals::OrbitalSpace;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex;
use ferric_scf::ScfResult;
use ndarray::{Array2, Array3};
use ndarray_linalg::{Cholesky, Eigh, UPLO};

/// Configuration for RI-MP2.
#[derive(Debug, Clone, Default)]
pub struct RiMp2Config {
    pub frozen_core: usize,
    /// Optional resident-bytes ceiling for the 3-index MO transform. `None` →
    /// the unified resolver ([`ferric_core::memory::resolve_budget_bytes`]) picks
    /// it (env override > auto 0.8×RAM > 2 GiB). `Some(bytes)` forces the ceiling
    /// unless an env override wins.
    pub memory_budget_bytes: Option<usize>,
}

/// Number of active (correlated) occupied orbitals after freezing
/// `frozen_core`. Errors instead of underflowing when the freeze covers the
/// whole occupied space — `frozen_core` comes straight from user config.
/// (`frozen_core == 0` with zero occupied orbitals is allowed: an empty spin
/// channel, e.g. β of a hydrogen atom, is legitimate.)
pub fn active_occ(nocc_total: usize, frozen_core: usize) -> Result<usize, FerricError> {
    if frozen_core != 0 && frozen_core >= nocc_total {
        return Err(FerricError::General(format!(
            "frozen_core = {frozen_core} freezes all {nocc_total} occupied orbitals — nothing left to correlate"
        )));
    }
    Ok(nocc_total - frozen_core)
}

/// Results from an RI-MP2 calculation.
#[derive(Debug)]
pub struct RiMp2Result {
    /// MP2 correlation energy (always negative).
    pub mp2_corr: f64,
    /// Total energy: E_RHF + E_MP2.
    pub total_energy: f64,
}

/// Spin-component resolved MP2 correlation energy.
#[derive(Debug, Clone)]
pub struct SpinComponents {
    /// Opposite-spin correlation energy.
    pub e_os: f64,
    /// Same-spin correlation energy.
    pub e_ss: f64,
    /// Total: e_os + e_ss (equals standard MP2 correlation).
    pub e_total: f64,
}

/// Resident-bytes ceiling for the raw (P|μν) tensor during MO transforms.
///
/// Resolves via [`ferric_core::memory::resolve_budget_bytes`]: `explicit`
/// (from [`RiMp2Config::memory_budget_bytes`]) is honored unless an env override
/// (`FERRIC_MEM_BUDGET_GB` / legacy vars) wins; otherwise auto 0.8×RAM, then a
/// 2 GiB fallback. Passing `None` reproduces the pure-env/auto chain.
pub fn eri3_budget_bytes(explicit: Option<usize>) -> usize {
    ferric_core::memory::resolve_budget_bytes(explicit)
}

/// Build (P|ia) without materializing the full AO 3-index tensor: raw (P|μν)
/// is generated in aux-row blocks sized to `budget_bytes` and transformed to
/// MO immediately. Bit-identical to
/// `transform_3center_ov(&eri3_tensor(..), ..)`; peak transient memory is one
/// aux block instead of the naux·nao² tensor.
pub fn eri3_mo_ov_blocked(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    budget_bytes: usize,
) -> Result<Array3<f64>, FerricError> {
    let nao = obs.nbasis();
    let naux = dfbs.nbasis();
    let nocc = c_occ.ncols();
    let nvir = c_vir.ncols();
    let row_bytes = nao * nao * 8;
    let block_naux = (budget_bytes / row_bytes.max(1)).clamp(1, naux.max(1));
    if block_naux >= naux {
        let eri3_ao = threeindex::eri3_tensor(op, obs, dfbs)?;
        return Ok(transform_3center_ov(&eri3_ao, c_occ, c_vir));
    }
    let mut mo = Array3::<f64>::zeros((naux, nocc, nvir));
    let mut p0 = 0;
    while p0 < naux {
        let p1 = (p0 + block_naux).min(naux);
        let blk = threeindex::eri3_block(op, obs, dfbs, p0, p1)?;
        for (off, p) in (p0..p1).enumerate() {
            let bp_ao = blk.slice(ndarray::s![off, .., ..]);
            // same per-P GEMM order as transform_3center_ov (bitwise identical)
            let tmp = bp_ao.dot(c_vir);
            let bp_mo = c_occ.t().dot(&tmp);
            mo.slice_mut(ndarray::s![p, .., ..]).assign(&bp_mo);
        }
        p0 = p1;
    }
    Ok(mo)
}

/// Build a DRESSED MO 3-index block `B^P_{pq} = Σ_Q V^{-1/2}[P,Q] (c_left^T
/// (Q|μν) c_right)[pq]`, shape `(naux, nleft*nright)`, streaming raw AO
/// aux-blocks from `src` under its budget and dressing each block on the fly.
///
/// This is the general (occ/vir agnostic) form of the `eri3_mo_ov_blocked` +
/// `v_inv_sqrt.dot(..)` pair used by the energy path, mirroring M3's
/// `compute_b_full_mo_with` streaming idiom: the output is allocated once and
/// the metric GEMM accumulates in place (beta=1), so the peak transient is one
/// aux-block MO panel instead of the full `(naux, nao²)` AO tensor plus a
/// second full-size dressed copy. Exactness: the same contraction as
/// `v_inv_sqrt.dot(transform_3center(eri3_tensor(..), c_left, c_right))`,
/// reordered per aux-block, not approximated.
fn eri3_mo_block_dressed(
    src: &mut ThreeIndexSource,
    v_inv_sqrt: &Array2<f64>,
    c_left: &Array2<f64>,
    c_right: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let naux = src.naux();
    let nleft = c_left.ncols();
    let nright = c_right.ncols();
    let width = nleft * nright;
    let mut b_flat = Array2::<f64>::zeros((naux, width));
    src.for_each_block(|blk| {
        let qb = blk.data.shape()[0];
        // MO-transform this raw aux-block: mo[q, pq] = c_left^T (Q|μν) c_right.
        let mut mo_blk = Array2::<f64>::zeros((qb, width));
        for q in 0..qb {
            let bq_ao = blk.data.slice(ndarray::s![q, .., ..]);
            let tmp = bq_ao.dot(c_right); // (nao, nright)
            let bq_mo = c_left.t().dot(&tmp); // (nleft, nright)
            mo_blk
                .slice_mut(ndarray::s![q, ..])
                .assign(&bq_mo.into_shape_with_order(width).unwrap());
        }
        // Dress into every output aux row, accumulating in place (beta=1):
        //   b_flat[:, pq] += V^{-1/2}[:, Qblock] · mo_blk.
        let msub = v_inv_sqrt.slice(ndarray::s![.., blk.p0..blk.p0 + qb]);
        ndarray::linalg::general_mat_mul(1.0, &msub, &mo_blk, 1.0, &mut b_flat);
        Ok(())
    })?;
    Ok(b_flat)
}

/// Compute RI-MP2 with spin-component resolution.
///
/// Returns `(SpinComponents, B_flat)` where `B_flat` is the dressed 3-index tensor
/// `B^P_ia = sum_Q V^{-1/2}_PQ (Q|ia)` of shape `(naux, nocc*nvir)`.
///
/// The spin decomposition uses:
/// - Opposite-spin: `E_OS = sum_{ijab} (ia|jb)^2 / D_{ijab}`
/// - Same-spin: `E_SS = sum_{ijab} (ia|jb)[(ia|jb)-(ib|ja)] / D_{ijab}`
///
/// Note: `E_OS + E_SS = sum_{ijab} (ia|jb)[2(ia|jb)-(ib|ja)] / D_{ijab}` which is
/// the standard MP2 expression.
pub fn ri_mp2_spin_components(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &RiMp2Config,
) -> Result<(SpinComponents, Array2<f64>), FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc = active_occ(nocc_total, config.frozen_core)?;
    let first_occ = config.frozen_core;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();
    let eps = rhf.eps_r();
    let c = rhf.mos_r();

    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // (P|Q) metric and V^{-1/2}
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v2c_inv_sqrt = metric_inverse_sqrt(&v2c, op)?;

    // (P|mu nu) -> (P|ia), aux-blocked under FERRIC_ERI3_BUDGET_GB
    let eri3_mo = eri3_mo_ov_blocked(op, obs, dfbs, &c_occ, &c_vir, eri3_budget_bytes(config.memory_budget_bytes))?;

    // B_ia^P = sum_Q (P|Q)^{-1/2} (Q|ia). One dressing GEMM per top-level
    // call, outside any rayon region (the rayon fan-out in
    // spin_components_from_b_ov below hasn't started yet). Opt-in BLAS raise
    // via FERRIC_BLAS_THREADS (default 1, unchanged behavior).
    let eri3_flat = eri3_mo
        .into_shape_with_order((naux, nocc * nvir))
        .unwrap();
    let b_flat = with_blas_threads(opt_in_blas_threads(), || v2c_inv_sqrt.dot(&eri3_flat)); // (naux, nocc*nvir)

    let sc = spin_components_from_b_ov(
        &b_flat, eps, nocc, nvir, first_occ, nocc_total,
    );
    Ok((sc, b_flat))
}

/// Spin-component MP2 energy from a pre-built dressed `b_ov` (no integral
/// transform). Factored out of [`ri_mp2_spin_components`] so a caller that
/// already holds the intermediates (e.g. the fused coupled-rings RPA path) can
/// reuse them rather than rebuild the `(P|op|ia)` transform. `eps` is the full
/// orbital-energy slice `rhf.eps_r()`.
pub fn spin_components_from_b_ov(
    b_ov: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
) -> SpinComponents {
    // (ia|jb) comes from i-blocked wide GEMMs G_i = B_i^T·B (nvir x nocc*nvir)
    // instead of per-element strided dots over P: same FLOPs at BLAS3
    // throughput. The outer i-loop is near-embarrassingly parallel (each i owns
    // an independent G_i transient of nvir*nocc*nvir*8 bytes and a private
    // (e_os, e_ss) partial), so we fan it across rayon and tuple-reduce. BLAS
    // stays serial inside each closure via OPENBLAS_NUM_THREADS=1 — nested
    // BLAS threads under rayon is the documented dgetrf-crash footgun.
    use rayon::prelude::*;
    // Compute each i's (e_os_i, e_ss_i) partial in parallel, then collect into an
    // i-ordered Vec and sum SEQUENTIALLY. A rayon `reduce` combines partials in a
    // tree whose shape depends on the worker count, so floating-point non-associativity
    // makes the total vary with RAYON_NUM_THREADS (~µHa). Collect-then-serial-sum keeps
    // the parallel per-i compute but fixes the accumulation order to be thread-independent.
    let partials: Vec<(f64, f64)> = (0..nocc)
        .into_par_iter()
        .map(|i| {
            let b_i = b_ov.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
            let g_i = b_i.t().dot(b_ov); // (nvir, nocc*nvir); g_i[a, jb] = (ia|jb)
            let mut e_os_i = 0.0;
            let mut e_ss_i = 0.0;
            for j in 0..nocc {
                let e_ij = eps[first_occ + i] + eps[first_occ + j];
                for a in 0..nvir {
                    for b in 0..nvir {
                        let g_ab = g_i[(a, j * nvir + b)]; // (ia|jb)
                        let g_ba = g_i[(b, j * nvir + a)]; // (ib|ja)
                        let denom = e_ij - eps[nocc_total + a] - eps[nocc_total + b];
                        e_os_i += g_ab * g_ab / denom;
                        e_ss_i += g_ab * (g_ab - g_ba) / denom;
                    }
                }
            }
            (e_os_i, e_ss_i)
        })
        .collect();
    let mut e_os = 0.0;
    let mut e_ss = 0.0;
    for (e_os_i, e_ss_i) in partials {
        e_os += e_os_i;
        e_ss += e_ss_i;
    }
    SpinComponents { e_os, e_ss, e_total: e_os + e_ss }
}

/// Compute the RI-MP2 correlation energy.
///
/// Requires converged RHF orbitals, an orbital basis (`obs`), and a density-fitting
/// auxiliary basis (`dfbs`). The auxiliary basis should be matched to the orbital
/// basis (e.g., cc-pVDZ with cc-pVDZ-RI).
pub fn ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &RiMp2Config,
) -> Result<RiMp2Result, FerricError> {
    let (sc, _) = ri_mp2_spin_components(mol, obs, dfbs, op, rhf, config)?;
    Ok(RiMp2Result {
        mp2_corr: sc.e_total,
        total_energy: rhf.energy + sc.e_total,
    })
}

/// All intermediates needed by the analytical RI-MP2 gradient.
#[derive(Debug)]
pub struct Mp2Intermediates {
    pub t2: Vec<f64>,
    /// B^P_{ia}, shape (naux, nocc*nvir), occ-vir block
    pub b_ov: Array2<f64>,
    /// B^P_{ij}, shape (naux, nocc*nocc), occ-occ block. `None` when built via
    /// [`compute_mp2_intermediates_ov_only`] — the gradient/zvector pipeline
    /// never reads it; only the CPKS polarizability path does.
    pub b_oo: Option<Array2<f64>>,
    /// B^P_{ab}, shape (naux, nvir*nvir), vir-vir block. `None` when built via
    /// [`compute_mp2_intermediates_ov_only`]: at nvir≈860/naux≈2200 this block
    /// alone is ~13 GB, and holding it across the gradient/zvector pipeline was
    /// the M4-audit peak. Consumers (cpks_polar) must unwrap with a clear error.
    pub b_vv: Option<Array2<f64>>,
    /// V^{-1/2} matrix, shape (naux, naux)
    pub v_inv_sqrt: Array2<f64>,
    pub p_oo: Array2<f64>,
    pub p_vv: Array2<f64>,
    pub nocc: usize,
    pub nvir: usize,
    pub nocc_total: usize,
    pub first_occ: usize,
    pub naux: usize,
    pub e_mp2: f64,
}

impl Mp2Intermediates {
    /// The active occupied/virtual orbital partition for these intermediates.
    pub fn orbital_space(&self) -> OrbitalSpace {
        OrbitalSpace::new(self.nocc, self.nvir, self.nocc_total, self.first_occ)
    }

    /// Compute spin-component scaled P_oo density correction.
    pub fn p_oo_scs(&self, c_os: f64, c_ss: f64) -> Array2<f64> {
        // P_ij = -sum_{kab} t_{ik,ab} (2 t_{jk,ab} - t_{jk,ba})
        // For SCS, we scale the OS term by c_os and the SS term by c_ss.
        // Effective Γ_iajb = c_os * iajb + c_ss * (iajb - ibja)
        // Since t_ik,ab = (ia|kb) / D, we can effectively scale the whole P.
        // Actually, SCS-MP2 is equivalent to scaling the t2 amplitudes.
        // A simple way to get the SCS density: P_scs = c_os * P_os + c_ss * P_ss.
        // But our P_oo is already the sum. 
        // Standard MP2: P_total = P_OS + P_SS.
        // SCS-MP2: P_total = c_os * P_OS + c_ss * P_SS.
        // This requires computing OS and SS density parts separately.
        
        // For now, let's approximate by average scaling if c_os == c_ss.
        // Proper implementation requires splitting build_mp2_density into OS/SS.
        let scale = (c_os + c_ss) / 2.0; 
        &self.p_oo * scale
    }

    /// Compute spin-component scaled P_vv density correction.
    pub fn p_vv_scs(&self, c_os: f64, c_ss: f64) -> Array2<f64> {
        let scale = (c_os + c_ss) / 2.0;
        &self.p_vv * scale
    }
}

/// Compact RI-MO intermediates needed for RPA-family methods.
///
/// Holds only the occ-vir B tensor and V^{-1/2}, skipping the full MP2
/// amplitudes, occ-occ / vir-vir B blocks, and quadruple-loop MP2 energy
/// that `compute_mp2_intermediates` produces. For benzene/cc-pVDZ this
/// drops the setup cost from ~5 s to ~0.5 s.
#[derive(Debug)]
pub struct RpaIntermediates {
    pub b_ov: Array2<f64>,
    pub v_inv_sqrt: Array2<f64>,
    pub nocc: usize,
    pub nvir: usize,
    pub nocc_total: usize,
    pub first_occ: usize,
    pub naux: usize,
}

impl RpaIntermediates {
    /// The active occupied/virtual orbital partition for these intermediates.
    pub fn orbital_space(&self) -> OrbitalSpace {
        OrbitalSpace::new(self.nocc, self.nvir, self.nocc_total, self.first_occ)
    }
}

/// Per-spin RI-MO intermediates for U-RPA / U-MP2.
///
/// Closed-shell `compute_rpa_intermediates` builds one B_ov tensor and
/// uses spin counting (factor 4) in the dielectric. Open-shell wants
/// `Π = Π_α + Π_β` with `Π_σ = B_ov_σ · diag(2/Δε_σ) · B_ov_σ^T`. Caller
/// builds both α and β intermediates separately and the RPA driver sums
/// the channels at every Davidson matvec.
///
/// `is_alpha = true` selects the α MO set; `false` selects β. Both spins
/// share the same aux-basis metric `V^{-1/2}` (returned in the result),
/// but each has its own `b_ov` shape `(naux, nocc_σ · nvir_σ)`.
pub fn compute_rpa_intermediates_spin(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &RiMp2Config,
    is_alpha: bool,
) -> Result<RpaIntermediates, FerricError> {
    use ferric_scf::Spin;
    if matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "compute_rpa_intermediates_spin: use compute_rpa_intermediates for Restricted results".into(),
        ));
    }
    let nbas = obs.nbasis();
    let nelec_total = mol.nelec() as usize;
    // For Unrestricted with multiplicity M = 2S+1:
    //   nocc_α = (N + 2S)/2, nocc_β = (N - 2S)/2
    let two_s = mol.multiplicity as i32 - 1;
    let nocc_total = if is_alpha {
        ((nelec_total as i32 + two_s) / 2) as usize
    } else {
        ((nelec_total as i32 - two_s) / 2) as usize
    };
    // ROHF stores α MOs and uses them for both spin channels (the SOMO is
    // just unoccupied in β); only mos_alpha is present. Fall back to it
    // when caller requests β on a ROHF result.
    let c_full = if is_alpha || matches!(rhf.spin, Spin::RestrictedOpen) {
        rhf.mos_a()
    } else {
        rhf.mos_b()
    };

    let nocc = active_occ(nocc_total, config.frozen_core)?;
    let first_occ = config.frozen_core;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();

    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    // erf (long-range) metric is indefinite in a Coulomb aux basis → regularized
    // eigh V^{-1/2}; Cholesky for Coulomb/erfc. (RSH-RPA path.)
    let v_inv_sqrt = metric_inverse_sqrt(&v2c, op)?;

    let c_occ = c_full.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c_full.slice(ndarray::s![.., nocc_total..]).to_owned();

    let eri3_ov = eri3_mo_ov_blocked(op, obs, dfbs, &c_occ, &c_vir, eri3_budget_bytes(config.memory_budget_bytes))?;
    // Dressing GEMM, outside any rayon region (top-level driver call). Opt-in
    // BLAS raise via FERRIC_BLAS_THREADS (default 1, unchanged behavior).
    let b_ov = with_blas_threads(opt_in_blas_threads(), || {
        v_inv_sqrt.dot(&eri3_ov.into_shape_with_order((naux, nocc * nvir)).unwrap())
    });

    Ok(RpaIntermediates {
        b_ov, v_inv_sqrt,
        nocc, nvir, nocc_total, first_occ, naux,
    })
}

/// Build B^P_{ia} = V^{-1/2} (P|ia) plus V^{-1/2} for RPA. Skips the MP2
/// amplitude/energy/density work in `compute_mp2_intermediates`.
pub fn compute_rpa_intermediates(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &RiMp2Config,
) -> Result<RpaIntermediates, FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc = active_occ(nocc_total, config.frozen_core)?;
    let first_occ = config.frozen_core;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();
    let c = rhf.mos_r();

    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    // erf (long-range) metric is indefinite in a Coulomb aux basis → regularized
    // eigh V^{-1/2}; Cholesky for Coulomb/erfc. (RSH-RPA path.)
    let v_inv_sqrt = metric_inverse_sqrt(&v2c, op)?;

    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    let eri3_ov = eri3_mo_ov_blocked(op, obs, dfbs, &c_occ, &c_vir, eri3_budget_bytes(config.memory_budget_bytes))?;
    // Dressing GEMM, outside any rayon region (top-level driver call). Opt-in
    // BLAS raise via FERRIC_BLAS_THREADS (default 1, unchanged behavior).
    let b_ov = with_blas_threads(opt_in_blas_threads(), || {
        v_inv_sqrt.dot(&eri3_ov.into_shape_with_order((naux, nocc * nvir)).unwrap())
    });

    Ok(RpaIntermediates {
        b_ov, v_inv_sqrt,
        nocc, nvir, nocc_total, first_occ, naux,
    })
}

/// Compute all MP2 intermediates needed for the analytical gradient.
///
/// Builds B tensor blocks for occ-vir, occ-occ, and vir-vir MO pairs,
/// plus V^{-1/2}, t2 amplitudes, and unrelaxed density corrections.
pub fn compute_mp2_intermediates(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &RiMp2Config,
) -> Result<Mp2Intermediates, FerricError> {
    compute_mp2_intermediates_impl(mol, obs, dfbs, op, rhf, config, true)
}

/// [`compute_mp2_intermediates`] without the occ-occ and vir-vir B blocks
/// (`b_oo = b_vv = None`). The analytical-gradient pipeline
/// (`rimp2_gradient_analytical` → `solve_zvector` → 3c/2c derivative
/// contractions) reads only `t2`, `b_ov`, `v_inv_sqrt`, `p_oo`, `p_vv`, so the
/// (naux, nvir²) vir-vir block — the single largest resident of the old
/// intermediates (13 GB at nvir≈860/naux≈2200) — need never exist there.
/// Use the full builder only on paths that consume `b_oo`/`b_vv` (CPKS α).
pub fn compute_mp2_intermediates_ov_only(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &RiMp2Config,
) -> Result<Mp2Intermediates, FerricError> {
    compute_mp2_intermediates_impl(mol, obs, dfbs, op, rhf, config, false)
}

fn compute_mp2_intermediates_impl(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &RiMp2Config,
    with_oo_vv: bool,
) -> Result<Mp2Intermediates, FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc = active_occ(nocc_total, config.frozen_core)?;
    let first_occ = config.frozen_core;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();
    let c = rhf.mos_r();

    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;

    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // Budget-aware raw (P|μν) source: in-core when it fits, disk-spilled in
    // aux-blocks otherwise. The three MO blocks below each stream this source
    // and dress with V^{-1/2} on the fly (see eri3_mo_block_dressed), so the
    // peak transient is one aux-block MO panel — not the former dense
    // (naux, nao², 14.3 GB) AO tensor plus its three transformed copies.
    let mut src = ThreeIndexSource::build(
        op, obs, dfbs, eri3_budget_bytes(config.memory_budget_bytes),
    )?;

    // B^P_{ia} = V^{-1/2} (P|ia); optionally B^P_{ij} and B^P_{ab} (CPKS only —
    // the gradient pipeline never reads them, and b_vv is the 13 GB hog).
    let b_ov = eri3_mo_block_dressed(&mut src, &v_inv_sqrt, &c_occ, &c_vir)?;
    let (b_oo, b_vv) = if with_oo_vv {
        (
            Some(eri3_mo_block_dressed(&mut src, &v_inv_sqrt, &c_occ, &c_occ)?),
            Some(eri3_mo_block_dressed(&mut src, &v_inv_sqrt, &c_vir, &c_vir)?),
        )
    } else {
        (None, None)
    };

    // Energy via i-blocked wide GEMMs (BLAS3), same path as the main RI-MP2
    // lane — replaces the per-element O(naux) double-dot quadruple loop.
    let eps = rhf.eps_r();
    let sc = spin_components_from_b_ov(&b_ov, eps, nocc, nvir, first_occ, nocc_total);
    let (e_os, e_ss) = (sc.e_os, sc.e_ss);

    let (t2, _) = crate::oo_rimp2::compute_t2_and_integrals(
        &b_ov, rhf.eps_r(), nocc, nvir, nocc_total, first_occ, naux,
    );
    let (p_oo, p_vv) = crate::oo_rimp2::build_mp2_density(&t2, nocc, nvir);

    Ok(Mp2Intermediates {
        t2, b_ov, b_oo, b_vv, v_inv_sqrt, p_oo, p_vv,
        nocc, nvir, nocc_total, first_occ, naux,
        e_mp2: e_os + e_ss,
    })
}

/// Compute V^{-1/2} via Cholesky decomposition.
///
/// Given a positive-definite matrix V = L L^T, returns L^{-1} so that
/// L^{-1} V L^{-T} = I, i.e., L^{-1} acts as V^{-1/2}.
pub fn cholesky_inverse_sqrt(v: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    // One-time (naux, naux) setup factorization, called once per RI-MP2/RPA/GW
    // intermediates build, outside any rayon region. Opt-in BLAS raise via
    // FERRIC_BLAS_THREADS (default 1, unchanged behavior);
    // opt_in_blas_threads()'s rayon-worker self-guard also covers any caller
    // reached from inside a rayon pool. The forward-substitution triangular
    // solve below stays untouched (scalar, deliberately serial — see
    // docs/parallelism-gaps-2026-07-09.md's "deliberately serial" list).
    let l = with_blas_threads(opt_in_blas_threads(), || v.cholesky(UPLO::Lower))
        .map_err(|e| FerricError::Lapack(format!("Cholesky on (P|Q): {e}")))?;
    let n = l.nrows();
    // Forward-substitution to invert lower-triangular L
    let mut l_inv = Array2::zeros((n, n));
    for i in 0..n {
        l_inv[(i, i)] = 1.0 / l[(i, i)];
        for j in (0..i).rev() {
            let mut sum = 0.0;
            for k in j..i {
                sum += l[(i, k)] * l_inv[(k, j)];
            }
            l_inv[(i, j)] = -sum / l[(i, i)];
        }
    }
    // V^{-1/2} = L^{-1} (so that B = L^{-1} (Q|ia) and B^T B = (ia|P) V^{-1} (Q|jb))
    Ok(l_inv)
}

/// Compute a SYMMETRIC V^{-1/2} via regularized eigendecomposition with canonical
/// orthogonalization (drop modes with λ < `LINDEP_THRESH`).
///
/// Cholesky [`cholesky_inverse_sqrt`] fails (return_code≠0) when the 2-center
/// metric `(P|w(r₁₂)|Q)` is not positive-definite. That happens for the
/// LONG-RANGE `erf(ωr)/r` operator fitted in a Coulomb-optimized RI aux basis:
/// the smooth long-range kernel has almost no high-spatial-frequency content, so
/// the tight (high-exponent) aux functions produce many near-zero / slightly
/// negative eigenvalues under roundoff. This routine drops those null modes —
/// the same fix already used in `ferric_scf::df_k::DfK` and equivalent to
/// PySCF's `lindep` threshold in `df.aux_e2`.
///
/// Returns a SYMMETRIC `V^{-1/2} = U diag(λ^{-1/2}) Uᵀ`. Unlike the Cholesky
/// `L^{-1}` (lower-triangular), this is symmetric, but `Bᵀ B = (ia|P) V^{-1}
/// (Q|jb)` is identical because both satisfy `MᵀM = V^{-1}` — the RPA/MP2
/// intermediates contract `B = M (Q|ia)` and only `BᵀB` enters, so either factor
/// is valid. Use this for range-separated (erf) operators; Cholesky stays the
/// fast path for Coulomb/erfc (positive-definite).
pub fn eigh_inverse_sqrt(v: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = v.nrows();
    // One-time (naux, naux) setup factorization, outside any rayon region.
    // Same opt-in raise + rayon-worker self-guard as cholesky_inverse_sqrt
    // above.
    let (evals, evecs) = with_blas_threads(opt_in_blas_threads(), || v.eigh(UPLO::Upper))
        .map_err(|e| FerricError::Lapack(format!("eigh on (P|Q): {e}")))?;
    const LINDEP_THRESH: f64 = 1e-10;
    // A physical RI metric is Gram-PSD for any positive-definite kernel (erf,
    // erfc, Coulomb, terfc all are; verified for terfc via its 3D Fourier
    // transform k̂(q) > 0). Negative eigenvalues beyond eigensolver noise
    // therefore signal an UPSTREAM INTEGRAL BUG (e.g. the 2026-07 terfc shim
    // far-field table-domain bug) or a non-PD kernel — not something a drop
    // threshold can repair (the accompanying positive modes are contaminated
    // too). Warn loudly instead of silently regularizing.
    let lmax = evals[n - 1]; // eigh returns ascending order
    let max_neg = evals
        .iter()
        .filter(|&&e| e < 0.0)
        .fold(0.0_f64, |acc, &e| acc.max(-e));
    if max_neg > 1e-8 * lmax {
        eprintln!(
            "WARNING eigh_inverse_sqrt: (P|Q) metric INDEFINITE beyond noise \
             (max|λ_neg|={max_neg:.3e}, λ_max={lmax:.3e}, rel={:.1e}). \
             Indefinite metric = upstream integral bug or non-PD kernel; \
             downstream RI energies are untrustworthy.",
            max_neg / lmax
        );
    }
    let mut u_scaled = evecs.clone();
    for k in 0..n {
        if evals[k] < LINDEP_THRESH {
            for r in 0..n {
                u_scaled[(r, k)] = 0.0;
            }
        } else {
            let s = 1.0 / evals[k].sqrt();
            for r in 0..n {
                u_scaled[(r, k)] *= s;
            }
        }
    }
    Ok(with_blas_threads(opt_in_blas_threads(), || {
        u_scaled.dot(&evecs.t())
    }))
}

/// V^{-1/2} that auto-selects: regularized eigendecomposition for operators whose
/// 2-center metric can go numerically indefinite / rank-deficient, fast Cholesky
/// otherwise (Coulomb/erfc/Terfc/Yukawa, positive-definite). Centralizes
/// indefinite-metric handling for all RI paths.
///
/// Eigh path:
/// - `ErfCoulomb`: the long-range erf metric goes numerically indefinite in a
///   Coulomb-optimized aux basis (near-null modes from tight aux functions).
/// - `Terf`: the tempered LONG-range complement (terf + terfc = Coulomb; see
///   `Operator::terf`) plays the same algebraic role as `erf` — r0 → ∞ drives
///   terf → 0, exactly as ω → 0 does for erf — so its metric loses rank in the
///   same limit and needs the same regularized eigh branch, not Cholesky.
///
/// `Terfc` is deliberately on the CHOLESKY path: the terfc kernel is
/// positive-definite (3D Fourier transform k̂(q) > 0 for all q, r0), so its
/// Gram metric is PD and Cholesky must succeed. The apparent indefiniteness
/// that previously forced Terfc through eigh (dpotrf return_code=225 on
/// alkane_4/cc-pVDZ-RI at r0≈1.417 Bohr) was an upstream integral bug — the
/// shim skipped the terf subtraction for far-field primitives outside the
/// interpolation tables, leaving full-Coulomb contamination. With that fixed
/// (exact Poisson-series fallback in shim.cc), a Cholesky failure here is a
/// real regression signal and must stay loud, not be regularized away.
pub fn metric_inverse_sqrt(
    v: &Array2<f64>,
    op: ferric_integrals::operator::Operator,
) -> Result<Array2<f64>, FerricError> {
    use ferric_integrals::operator::OperatorKind;
    if matches!(op.kind, OperatorKind::ErfCoulomb | OperatorKind::Terf) {
        eigh_inverse_sqrt(v)
    } else {
        cholesky_inverse_sqrt(v)
    }
}

/// RI-MP2 correlation energy computed via the `einsum!` tensor framework.
///
/// Implements the same closed-shell RI-MP2 as [`ri_mp2_spin_components`] but
/// routes all 4-index contractions through `ferric_tensors::einsum!` for
/// demonstration and A/B testing.  Both functions use the same RI integrals
/// (same `b_flat` construction), so their energies should agree to near
/// machine precision (not just RI-approximation tolerance).
///
/// # Formula
/// Build `B_ov[P,i,a] = V^{-1/2}_{PQ}(Q|ia)`, then:
/// ```text
/// V[i,j,a,b]   = (ia|jb) = einsum("Pia,Pjb->iajb") permuted (i,a,j,b)->(i,j,a,b)
/// t[i,j,a,b]   = V[i,j,a,b] / (eps_i + eps_j - eps_a - eps_b)
/// e_os = sum_{ijab} t[i,j,a,b] * V[i,j,a,b]
/// e_ss = sum_{ijab} t[i,j,a,b] * (V[i,j,a,b] - V[i,j,b,a])
/// ```
pub fn ri_mp2_einsum(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &RiMp2Config,
) -> Result<SpinComponents, FerricError> {
    use ferric_tensors::{einsum, Axis, Tensor};
    use ndarray::IxDyn;

    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc = active_occ(nocc_total, config.frozen_core)?;
    let first_occ = config.frozen_core;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();
    let eps = rhf.eps_r();
    let c = rhf.mos_r();

    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // V^{-1/2} and AO 3-center integrals — identical to ri_mp2_spin_components
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_mo = eri3_mo_ov_blocked(op, obs, dfbs, &c_occ, &c_vir, eri3_budget_bytes(config.memory_budget_bytes))?; // (naux, nocc, nvir)

    // B^P_{ia} = V^{-1/2} (Q|ia); same b_flat as the scalar path. Outside any
    // rayon region (einsum! runs after this returns). Opt-in BLAS raise via
    // FERRIC_BLAS_THREADS (default 1, unchanged behavior).
    let flat = eri3_mo
        .into_shape_with_order((naux, nocc * nvir))
        .unwrap();
    let b_flat = with_blas_threads(opt_in_blas_threads(), || v_inv_sqrt.dot(&flat)); // (naux, nocc*nvir)
    let b_3d = b_flat
        .into_shape_with_order((naux, nocc, nvir))
        .unwrap()
        .into_dyn();

    // Wrap as Tensor<3> [Aux, O, V]
    let b_ov = Tensor::new(b_3d, [Axis::Aux, Axis::O, Axis::V]);

    // (ia|jb) in chemist notation: g[i,a,j,b]
    let g_iajb: ndarray::ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);

    // Permute (i,a,j,b) -> (i,j,a,b): axes [0,2,1,3]
    let v_arr = g_iajb
        .permuted_axes(IxDyn(&[0, 2, 1, 3]))
        .as_standard_layout()
        .into_owned(); // shape (nocc, nocc, nvir, nvir)

    // Build amplitude t[i,j,a,b] = V[i,j,a,b] / D_{ijab}
    // and accumulate e_os, e_ss with a denominator loop
    let mut t_arr = ndarray::ArrayD::zeros(IxDyn(&[nocc, nocc, nvir, nvir]));
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let d = eps[first_occ + i] + eps[first_occ + j]
                        - eps[nocc_total + a]
                        - eps[nocc_total + b];
                    t_arr[[i, j, a, b]] = v_arr[[i, j, a, b]] / d;
                }
            }
        }
    }

    // Wrap for einsum!
    let t_t = Tensor::new(t_arr, [Axis::O, Axis::O, Axis::V, Axis::V]);
    let v_t = Tensor::new(v_arr.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);

    // V - V.permuted([0,1,3,2]): (i,j,a,b) -> swap last two -> (ib|ja) term
    let v_swap = v_arr.clone()
        .permuted_axes(IxDyn(&[0, 1, 3, 2]))
        .as_standard_layout()
        .into_owned();
    let vmx_arr = &v_arr - &v_swap; // (ia|jb) - (ib|ja) = SS kernel
    let vmx_t = Tensor::new(vmx_arr, [Axis::O, Axis::O, Axis::V, Axis::V]);

    // e_os = sum t * V,  e_ss = sum t * (V - V_swap)
    let e_os: f64 = einsum!("ijab,ijab->", &t_t, &v_t);
    let e_ss: f64 = einsum!("ijab,ijab->", &t_t, &vmx_t);

    Ok(SpinComponents { e_os, e_ss, e_total: e_os + e_ss })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    fn run_ri_mp2(xyz: &str, basis_name: &str, aux_name: &str) -> (ScfResult, RiMp2Result) {
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
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        let aux_bs = basis::bundled(aux_name).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let mp2 = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();
        (rhf, mp2)
    }

    #[test]
    fn eri3_mo_ov_blocked_is_bit_identical_to_incore() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let nao = obs.nbasis();
        let c = ndarray::Array2::<f64>::eye(nao);
        let c_occ = c.slice(ndarray::s![.., ..5]).to_owned();
        let c_vir = c.slice(ndarray::s![.., 5..]).to_owned();

        let eri3_ao = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let reference = transform_3center_ov(&eri3_ao, &c_occ, &c_vir);
        // 1-byte budget forces single-aux-row blocking (max fragmentation)
        let blocked = eri3_mo_ov_blocked(op, &obs, &dfbs, &c_occ, &c_vir, 1).unwrap();

        assert_eq!(reference.shape(), blocked.shape());
        let maxdiff = reference
            .iter()
            .zip(blocked.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            maxdiff == 0.0,
            "blocked (P|ia) differs from in-core, maxdiff={maxdiff:e}"
        );
    }

    // NOTE: the FERRIC_ERI3_BUDGET_GB *wiring* is intentionally not tested
    // here — std::env::set_var is process-global and poisons the parallel
    // test harness (every concurrent test silently runs micro-blocked).
    // The blocked path itself is covered by
    // eri3_mo_ov_blocked_is_bit_identical_to_incore; the env wiring is
    // verified at the CLI level (water rs-mp2-rpa with and without the env
    // var must print identical energies).

    #[test]
    fn test_rimp2_h2o_ccpvdz() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let (rhf, mp2) = run_ri_mp2(xyz, "cc-pvdz", "cc-pvdz-ri");
        eprintln!(
            "H2O/cc-pVDZ RHF energy: {:.10}, RI-MP2 corr: {:.10}, total: {:.10}",
            rhf.energy, mp2.mp2_corr, mp2.total_energy
        );
        // PySCF RI-MP2 (cc-pvdz-ri): corr = -0.2040334729
        assert!(
            (mp2.mp2_corr - (-0.2040334729)).abs() < 1e-6,
            "RI-MP2 corr: got {:.10}, ref -0.2040334729",
            mp2.mp2_corr
        );
    }

    #[test]
    fn test_spin_components_sum_to_total() {
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let (sc, _) = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();
        eprintln!("SpinComponents: E_OS={:.10}, E_SS={:.10}, E_total={:.10}", sc.e_os, sc.e_ss, sc.e_total);
        assert!((sc.e_os + sc.e_ss - sc.e_total).abs() < 1e-15,
            "E_OS + E_SS = {} + {} = {} vs total {}", sc.e_os, sc.e_ss, sc.e_os + sc.e_ss, sc.e_total);
        // OS should be larger magnitude than SS for H2
        assert!(sc.e_os.abs() > sc.e_ss.abs(),
            "OS ({}) should dominate SS ({})", sc.e_os, sc.e_ss);
    }

    #[test]
    fn test_rimp2_h2_ccpvdz() {
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
        let (rhf, mp2) = run_ri_mp2(xyz, "cc-pvdz", "cc-pvdz-ri");
        eprintln!(
            "H2/cc-pVDZ RHF energy: {:.10}, RI-MP2 corr: {:.10}, total: {:.10}",
            rhf.energy, mp2.mp2_corr, mp2.total_energy
        );
        // PySCF canonical MP2: -0.0263715576 (RI should be close)
        assert!(
            (mp2.mp2_corr - (-0.0263715576)).abs() < 1e-4,
            "H2 RI-MP2 corr: {:.10}",
            mp2.mp2_corr
        );
    }

    #[test]
    fn ri_mp2_einsum_matches_scalar() {
        use ferric_core::parallel::ParallelContext;
        use ferric_scf::rhf::{solve_rhf, RhfConfig};
        use ferric_scf::screening::SchwarzBounds;

        let xyz = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = RiMp2Config { frozen_core: 0, memory_budget_bytes: None };

        let (sc_ref, _) = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let sc_ein = ri_mp2_einsum(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        assert!((sc_ein.e_os - sc_ref.e_os).abs() < 1e-9, "os {} vs {}", sc_ein.e_os, sc_ref.e_os);
        assert!((sc_ein.e_ss - sc_ref.e_ss).abs() < 1e-9, "ss {} vs {}", sc_ein.e_ss, sc_ref.e_ss);
        assert!((sc_ein.e_total - sc_ref.e_total).abs() < 1e-9, "tot {} vs {}", sc_ein.e_total, sc_ref.e_total);
    }

    #[test]
    fn frozen_core_exceeding_occupied_is_an_error() {
        // H2 has exactly 1 occupied orbital; frozen_core = 2 must come back
        // as a clean Err, not a usize underflow panic.
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let cfg = RiMp2Config { frozen_core: 2, memory_budget_bytes: None };
        let res = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg);
        assert!(res.is_err(), "frozen_core > nocc must be an error, got {res:?}");
        let cfg_all = RiMp2Config { frozen_core: 1, memory_budget_bytes: None };
        let res_all = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg_all);
        assert!(res_all.is_err(), "freezing every occupied orbital must be an error, got {res_all:?}");
    }

    #[test]
    fn test_mp2_intermediates() {
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let inter = compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();
        let mp2 = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();

        assert!((inter.e_mp2 - mp2.mp2_corr).abs() < 1e-12,
            "intermediates energy {} != ri_mp2 {}", inter.e_mp2, mp2.mp2_corr);

        for i in 0..inter.nocc {
            for j in 0..inter.nocc {
                assert!((inter.p_oo[(i,j)] - inter.p_oo[(j,i)]).abs() < 1e-12, "P_oo not symmetric");
            }
        }
        for a in 0..inter.nvir {
            for b in 0..inter.nvir {
                assert!((inter.p_vv[(a,b)] - inter.p_vv[(b,a)]).abs() < 1e-12, "P_vv not symmetric");
            }
        }

        let tr_oo: f64 = (0..inter.nocc).map(|i| inter.p_oo[(i,i)]).sum();
        let tr_vv: f64 = (0..inter.nvir).map(|a| inter.p_vv[(a,a)]).sum();
        assert!(tr_oo < 0.0, "tr(P_oo) should be negative: {}", tr_oo);
        assert!(tr_vv > 0.0, "tr(P_vv) should be positive: {}", tr_vv);
        assert!((tr_oo + tr_vv).abs() < 1e-10,
            "density not conserved: tr(P_oo)={} + tr(P_vv)={} = {}", tr_oo, tr_vv, tr_oo + tr_vv);
    }

    /// The terfc kernel is positive-definite (3D Fourier transform k̂(q) > 0), so
    /// (P|Q)_terfc must be PD and Cholesky must succeed — including the exact
    /// configuration that USED to fail (alkane_4/cc-pVDZ-RI at r0=0.75 Å,
    /// dpotrf return_code=225). The old failure was the shim's far-field
    /// table-domain bug (terf subtraction skipped for S > 20 primitives,
    /// leaving full-Coulomb contamination that made the metric spuriously
    /// indefinite). This test pins the integral fix at the metric level.
    /// Requires FERRIC_TERF_TABLE_DIR to point at the terfc interpolation tables.
    #[test]
    fn terfc_metric_positive_definite_alkane4() {
        if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
            eprintln!("skipping: FERRIC_TERF_TABLE_DIR not set");
            return;
        }
        let mol = Molecule::load_xyz(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/molecules/alkane_4.xyz"),
        )
        .unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        // r0 = 0.75 Angstrom -> Bohr; historically the worst case (most far-field
        // primitives beyond the table domain).
        let op = Operator::terfc(0.75 * 1.889_725_988_6);

        let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_inv_sqrt = metric_inverse_sqrt(&v2c, op).expect(
            "(P|Q)_terfc must be positive-definite (Cholesky); a dpotrf failure here \
             means far-field terfc integrals regressed",
        );

        // M V Mᵀ = I for any valid inverse-sqrt factor M (Cholesky L⁻¹ or
        // symmetric eigh form).
        let p = v_inv_sqrt.dot(&v2c).dot(&v_inv_sqrt.t());
        let n = p.nrows();
        let mut max_dev = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let target = if i == j { 1.0 } else { 0.0 };
                max_dev = max_dev.max((p[(i, j)] - target).abs());
            }
        }
        assert!(
            max_dev < 1e-8,
            "V^(-1/2) V V^(-T/2) != I: max deviation {max_dev:e}"
        );
    }

    /// PHYSICS regression for the terfc far-field integral fix (shim.cc
    /// table-domain bug: out-of-table primitives skipped the terf subtraction,
    /// leaving full-Coulomb contamination in far-field (P|Q) and (P|ia)).
    ///
    /// terfc(r,r₀)/r is a tempered SHORT-range Coulomb: smaller r₀ screens more,
    /// so |E_corr| must grow monotonically with r₀ and approach full-Coulomb
    /// correlation as r₀ → ∞:
    ///     |E(0.75Å)| < |E(1.05Å)| < |E(2.0Å)| < |E(Coulomb)|,
    ///     E(2.0Å)/E(Coulomb) > 0.95.
    ///
    /// Before the fix this failed catastrophically (alkane_4 E(0.75Å) = −1.289 Ha
    /// vs Coulomb −0.733 Ha — |ratio| 1.76, wrong side of Coulomb), and no eigh
    /// drop-threshold could repair it because the 3-index tensor was contaminated
    /// too. A projector-idempotency check passes even with garbage physics; this
    /// energy-ordering test is the discriminating one. Runs on the plain Cholesky
    /// metric path. Requires FERRIC_TERF_TABLE_DIR.
    #[test]
    fn terfc_ri_energy_monotone_in_r0_alkane4() {
        if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
            eprintln!("skipping: FERRIC_TERF_TABLE_DIR not set");
            return;
        }
        const A2B: f64 = 1.889_725_988_6;
        let mol = Molecule::load_xyz(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/molecules/alkane_4.xyz"
        ))
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let opc = Operator::coulomb();
        let bounds = SchwarzBounds::compute(opc, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            opc,
            &bounds,
            &RhfConfig { energy_conv: 1e-9, ..Default::default() },
        )
        .unwrap();
        let cfg = RiMp2Config::default();

        let e = |op: Operator| {
            ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg)
                .unwrap()
                .0
                .e_total
        };
        let e_coul = e(opc);
        let e075 = e(Operator::terfc(0.75 * A2B));
        let e105 = e(Operator::terfc(1.05 * A2B));
        let e20 = e(Operator::terfc(2.0 * A2B));

        eprintln!(
            "terfc alkane_4: E(0.75)={e075:.6} E(1.05)={e105:.6} E(2.0)={e20:.6} E(coul)={e_coul:.6}"
        );

        // Correlation energies are negative; compare magnitudes.
        assert!(
            e075.abs() < e105.abs(),
            "|E(0.75)|={} should be < |E(1.05)|={}",
            e075.abs(),
            e105.abs()
        );
        assert!(
            e105.abs() < e20.abs(),
            "|E(1.05)|={} should be < |E(2.0)|={}",
            e105.abs(),
            e20.abs()
        );
        assert!(
            e20.abs() < e_coul.abs(),
            "|E(2.0)|={} should be < |E(coulomb)|={}",
            e20.abs(),
            e_coul.abs()
        );
        assert!(
            e20 / e_coul > 0.95,
            "E(2.0)/E(coulomb)={} should exceed 0.95 (terfc→Coulomb as r0→∞)",
            e20 / e_coul
        );
    }
}
