//! GEMM building blocks shared by the closed-shell (`oo_rimp2`) and open-shell
//! (`u_rimp2`) OO-MP2 orbital gradients and the U-MP2 density.
//!
//! The integral-response part of the orbital gradient contracts amplitudes
//! with `(ca|jb)` and `(ik|jb)`. The `(ca|jb)` terms go through the RI
//! factorisation `(ca|jb) = Σ_P B^P_{ca} B^P_{jb}`:
//!
//! ```text
//!   Γ_k[P, x] = Σ_j Σ_y B^P_{jy} M_kj[x, y]          (pair_gamma_rows)
//!   G[c, k]   = Σ_{P,x} B^P_{cx} Γ_k[P, x]            (contract_vv_gamma)
//! ```
//!
//! so no `(nvir², nocc·nvir)` VVOV block is ever formed. The `(ik|jb)` terms
//! read the small `(nocc², nocc·nvir)` OOOV block.
//!
//! Every parallel region here runs one single-threaded BLAS call per task, and
//! every reduction across tasks is added in task order, so results do not
//! depend on the thread count. Slab widths are constants.

use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, Array3, ArrayView2, ArrayViewMut2, Axis, Zip};
use rayon::prelude::*;

/// Auxiliary functions per task in [`contract_vv_gamma`]. A constant, so the
/// summation order over `P` is the same on any thread count and budget.
pub(crate) const VV_CONTRACT_AUX_SLAB: usize = 32;

/// The MO blocks of one spin's dressed full-MO tensor `b_full[P, p, q]` that
/// the orbital gradient reads.
pub(crate) struct GradientB {
    /// `bov[P, i·nvir + a] = B^P_{ia}`, shape `(naux, nocc·nvir)`.
    pub bov: Array2<f64>,
    /// `boo[P, i·nocc + k] = B^P_{ik}`, shape `(naux, nocc²)`.
    pub boo: Array2<f64>,
    /// `bvv[c, P·nvir + x] = B^P_{cx}`, shape `(nvir, naux·nvir)`: the
    /// virtual `c` leads, so the contraction over `(P, x)` in
    /// [`contract_vv_gamma`] is a plain GEMM.
    pub bvv: Array2<f64>,
    /// Number of auxiliary functions.
    pub naux: usize,
    /// Active occupied orbitals.
    pub nocc: usize,
    /// Virtual orbitals.
    pub nvir: usize,
}

impl GradientB {
    /// Element count of `bov + boo + bvv`.
    pub(crate) fn elems(naux: usize, nocc: usize, nvir: usize) -> usize {
        naux.saturating_mul(
            nocc.saturating_mul(nvir)
                .saturating_add(nocc.saturating_mul(nocc))
                .saturating_add(nvir.saturating_mul(nvir)),
        )
    }

    /// Copy the blocks out of `b_full` (active occupied `first_occ..nocc_total`,
    /// virtual `nocc_total..`).
    pub(crate) fn extract(
        b_full: &Array3<f64>,
        nocc: usize,
        nvir: usize,
        first_occ: usize,
        nocc_total: usize,
    ) -> Self {
        let naux = b_full.shape()[0];
        let mut bov = Array2::<f64>::zeros((naux, nocc * nvir));
        let mut boo = Array2::<f64>::zeros((naux, nocc * nocc));
        let mut bvv = Array2::<f64>::zeros((nvir, naux * nvir));
        for p in 0..naux {
            let bp = b_full.index_axis(Axis(0), p);
            for i in 0..nocc {
                let row = bp.row(first_occ + i);
                bov.slice_mut(s![p, i * nvir..(i + 1) * nvir])
                    .assign(&row.slice(s![nocc_total..nocc_total + nvir]));
                boo.slice_mut(s![p, i * nocc..(i + 1) * nocc])
                    .assign(&row.slice(s![first_occ..first_occ + nocc]));
            }
            for c in 0..nvir {
                bvv.slice_mut(s![c, p * nvir..(p + 1) * nvir])
                    .assign(&bp.slice(s![nocc_total + c, nocc_total..nocc_total + nvir]));
            }
        }
        Self {
            bov,
            boo,
            bvv,
            naux,
            nocc,
            nvir,
        }
    }

    /// `B^P_{jy}` for one occupied `j`: `(naux, nvir)`.
    pub(crate) fn bov_occ(&self, j: usize) -> ArrayView2<'_, f64> {
        self.bov.slice(s![.., j * self.nvir..(j + 1) * self.nvir])
    }
}

/// `aᵀ · b` through `ferric_core::linalg::par_gemm` (fixed column blocks, so
/// the result does not depend on the thread count).
pub(crate) fn par_gemm_tn(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut c = Array2::<f64>::zeros((a.ncols(), b.ncols()));
    ferric_core::linalg::par_gemm(1.0, &a.t(), &b.view(), 0.0, &mut c);
    c
}

/// Run `task(n, acc)` for every `n` in `0..n_tasks` on the rayon pool, each on
/// its own zeroed `dim` accumulator at one BLAS thread, then add the
/// accumulators in task order. The result does not depend on the thread count.
///
/// The `n_tasks` accumulators are all held until the sum. They are charged to
/// the shared memory pool as `label` with the soft charge (they are small next
/// to the amplitudes the callers already hold, and their count is a function
/// of the problem only).
pub(crate) fn ordered_task_sum(
    label: &str,
    n_tasks: usize,
    dim: (usize, usize),
    task: impl Fn(usize, &mut Array2<f64>) + Sync,
) -> Array2<f64> {
    let _charge = crate::rimp2::charge_mo_side_soft(
        label,
        n_tasks
            .saturating_mul(dim.0)
            .saturating_mul(dim.1)
            .saturating_mul(8),
    );
    let parts: Vec<Array2<f64>> = ferric_integrals::blas_threads::with_blas_threads(1, || {
        (0..n_tasks)
            .into_par_iter()
            .map(|n| {
                let mut acc = Array2::<f64>::zeros(dim);
                task(n, &mut acc);
                acc
            })
            .collect()
    });
    let mut out = Array2::<f64>::zeros(dim);
    for p in &parts {
        out += p;
    }
    out
}

/// Soft pool charge for the per-worker `nvir × nvir` combination scratch of
/// [`pair_gamma_rows`] and [`pair_ooov_term`]. Soft because its size reads the
/// worker count (see `rimp2::charge_mo_side_soft`); no width reads it.
pub(crate) fn charge_pair_scratch(
    label: &str,
    nvir: usize,
) -> ferric_core::memory::pool::Reservation {
    crate::rimp2::charge_mo_side_soft(
        label,
        rayon::current_num_threads()
            .saturating_mul(nvir)
            .saturating_mul(nvir)
            .saturating_mul(8),
    )
}

/// `out = p·A − q·Aᵀ + p·Bᵀ − q·B` for square `nvir × nvir` blocks.
fn pair_combination(
    out: &mut Array2<f64>,
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    p: f64,
    q: f64,
) {
    Zip::from(out)
        .and(&a)
        .and(a.t())
        .and(b.t())
        .and(&b)
        .for_each(|o, &x, &xt, &yt, &y| *o = p * x - q * xt + p * yt - q * y);
}

/// Fill `gamma` (`nocc` rows of length `naux·nvir`, row `k` read as an
/// `(naux, nvir)` matrix) with
///
/// ```text
///   Γ_k = alpha · Σ_j B_j · M_kjᵀ,   M_kj = p·A_kj − q·A_kjᵀ + p·A_jkᵀ − q·A_jk
/// ```
///
/// where `A_kj = block(k, j)` is an `nvir × nvir` amplitude block and
/// `B_j = b.bov_occ(j)`. One task per `k`, the `j` sum in order inside it.
pub(crate) fn pair_gamma_rows<'t>(
    block: &(impl Fn(usize, usize) -> ArrayView2<'t, f64> + Sync),
    p: f64,
    q: f64,
    alpha: f64,
    b: &GradientB,
    mut gamma: ArrayViewMut2<'_, f64>,
) {
    let (nocc, nvir, naux) = (b.nocc, b.nvir, b.naux);
    assert_eq!(gamma.dim(), (nocc, naux * nvir), "pair_gamma_rows: gamma");
    ferric_integrals::blas_threads::with_blas_threads(1, || {
        gamma
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(k, row)| {
                let mut g = row
                    .into_shape_with_order((naux, nvir))
                    .expect("gamma row is contiguous");
                let mut m = Array2::<f64>::zeros((nvir, nvir));
                for j in 0..nocc {
                    pair_combination(&mut m, block(k, j), block(j, k), p, q);
                    general_mat_mul(alpha, &b.bov_occ(j), &m.t(), 1.0, &mut g);
                }
            });
    });
}

/// Fill `gamma` (rows of length `naux·nvir_out`, row `k` read as an
/// `(naux, nvir_out)` matrix) with `Γ_k = Σ_j B_j · block(k, j)ᵀ`, where
/// `block(k, j)` is `nvir_out × nvir_in` and `B_j = b_in.bov_occ(j)`
/// (`(naux, nvir_in)`). One task per `k`, the `j` sum in order inside it.
pub(crate) fn cross_gamma_rows<'t>(
    block: &(impl Fn(usize, usize) -> ArrayView2<'t, f64> + Sync),
    nvir_out: usize,
    b_in: &GradientB,
    mut gamma: ArrayViewMut2<'_, f64>,
) {
    let naux = b_in.naux;
    assert_eq!(gamma.ncols(), naux * nvir_out, "cross_gamma_rows: gamma");
    ferric_integrals::blas_threads::with_blas_threads(1, || {
        gamma
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(k, row)| {
                let mut g = row
                    .into_shape_with_order((naux, nvir_out))
                    .expect("gamma row is contiguous");
                for j in 0..b_in.nocc {
                    general_mat_mul(1.0, &b_in.bov_occ(j), &block(k, j).t(), 1.0, &mut g);
                }
            });
    });
}

/// `G[c, r] = Σ_{P,x} bvv[c, P·nvir + x] · gamma[r, P·nvir + x]`, shape
/// `(nvir, rows)`: slabs of [`VV_CONTRACT_AUX_SLAB`] auxiliary functions, one
/// pool task each, added in slab order.
pub(crate) fn contract_vv_gamma(
    label: &str,
    bvv: &Array2<f64>,
    gamma: &Array2<f64>,
    naux: usize,
    nvir: usize,
    aux_slab: usize,
) -> Array2<f64> {
    let rows = gamma.nrows();
    assert_eq!(bvv.dim(), (nvir, naux * nvir), "contract_vv_gamma: bvv");
    assert_eq!(gamma.ncols(), naux * nvir, "contract_vv_gamma: gamma");
    let aux_slab = aux_slab.max(1);
    let n_slabs = naux.div_ceil(aux_slab);
    ordered_task_sum(label, n_slabs, (nvir, rows), |n, acc| {
        let p0 = n * aux_slab;
        let p1 = (p0 + aux_slab).min(naux);
        let cols = p0 * nvir..p1 * nvir;
        let lhs = bvv.slice(s![.., cols.clone()]);
        let rhs = gamma.slice(s![.., cols]);
        general_mat_mul(1.0, &lhs, &rhs.t(), 0.0, acc);
    })
}

/// The occupied-occupied integral part of the pair-structured gradient terms,
/// as one task's contribution for occupied `i`:
///
/// ```text
///   acc[c, k] += alpha · Σ_j Σ_y N_ij[c, y] · ooov[i·nocc + k, j·nvir + y],
///   N_ij = p·A_ij − q·A_ji + p·A_jiᵀ − q·A_ijᵀ
/// ```
///
/// with `A_ij = block(i, j)` (`nvir × nvir`) and `ooov[(i,k),(j,y)] = (ik|jy)`.
// Two coefficient pairs plus the block source, the integrals and the target
// are all distinct inputs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pair_ooov_term<'t>(
    block: &(impl Fn(usize, usize) -> ArrayView2<'t, f64> + Sync),
    p: f64,
    q: f64,
    alpha: f64,
    ooov: &Array2<f64>,
    i: usize,
    nocc: usize,
    nvir: usize,
    acc: &mut ArrayViewMut2<'_, f64>,
) {
    let mut n = Array2::<f64>::zeros((nvir, nvir));
    let rows = i * nocc..(i + 1) * nocc;
    for j in 0..nocc {
        // N_ij = p·A_ij − q·A_ijᵀ + p·A_jiᵀ − q·A_ji, the same combination as
        // M_kj in `pair_gamma_rows` with (k, j) -> (i, j).
        pair_combination(&mut n, block(i, j), block(j, i), p, q);
        let o = ooov.slice(s![rows.clone(), j * nvir..(j + 1) * nvir]);
        general_mat_mul(alpha, &n, &o.t(), 1.0, acc);
    }
}
