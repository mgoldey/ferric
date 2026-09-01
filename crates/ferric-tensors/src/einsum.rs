//! Runtime engine for a binary tensor contraction.
//!
//! [`einsum_binary_batched`] is what the `einsum!` macro lowers to;
//! [`einsum_binary`] is the no-batch, unit-scale special case. Given two
//! operands and the positions of their batch / free / contracted axes, the
//! engine permutes each operand so those axis groups are contiguous, reshapes,
//! calls GEMM (`general_mat_mul` — one call, or one per batch slice when there
//! are batch axes), applies the scale factor, and reshapes the result to the
//! requested output shape. Any required permutation copy is logged at debug
//! level so hot transposes are discoverable.

use ferric_core::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ndarray::linalg::general_mat_mul;
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use thiserror::Error;

/// Errors from a contraction.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TensorError {
    /// A contracted axis had different lengths in the two operands.
    #[error("contracted-dimension mismatch: left {left} vs right {right}")]
    ContractedDimMismatch { left: usize, right: usize },
    /// The computed 2D product shape could not be reshaped to the output shape.
    #[error("output reshape failed: product has {got} elements, output shape needs {want}")]
    OutputReshape { got: usize, want: usize },
}

impl From<TensorError> for ferric_core::error::FerricError {
    fn from(e: TensorError) -> Self { Self::General(e.to_string()) }
}

/// Contract two operands over their contracted axes into one GEMM.
///
/// - `*_free`: axis positions (into the operand) that survive into the output.
/// - `*_contr`: axis positions that are summed over. Must be the same logical
///   axes in both operands (the macro guarantees ordering matches).
/// - `out_shape`: the shape of the result (free_left dims then free_right dims).
///   Empty slice means a scalar (returned as a length-1 `ArrayD` of shape `[]`).
///
/// This is the plain (no batch axes, unit scale) entry point. See
/// [`einsum_binary_batched`] for diagonal/batch axes and a scale factor.
pub fn einsum_binary(
    left: ArrayViewD<f64>,
    left_free: &[usize],
    left_contr: &[usize],
    right: ArrayViewD<f64>,
    right_free: &[usize],
    right_contr: &[usize],
    out_shape: &[usize],
) -> Result<ArrayD<f64>, TensorError> {
    // `to_2d_or_transpose` returns the operand already viewable as a matrix
    // whenever the required axis order is either the identity OR the exact
    // swap of the two axis groups — in the latter case it hands back the
    // TRANSPOSED view and ndarray's GEMM absorbs it into dgemm's transa/transb
    // flag, so no permutation copy is materialized at all. 72 of the ~130
    // workspace specs have at least one operand in that class (e.g. the CCSD
    // VVVV term `ijcd,abcd->ijab`), which is 52% of all copies that survived
    // the identity check. Only genuinely interleaved orders still copy.
    let l2 = to_2d_or_transpose(left, left_free, left_contr, "left");
    let r2 = to_2d_or_transpose(right, right_contr, right_free, "right");

    let (lf, lc) = (l2.view().shape()[0], l2.view().shape()[1]);
    let (rc, rf) = (r2.view().shape()[0], r2.view().shape()[1]);
    if lc != rc {
        return Err(TensorError::ContractedDimMismatch { left: lc, right: rc });
    }

    let mut out2 = ArrayD::<f64>::zeros(IxDyn(&[lf, rf]));
    {
        let l2m = l2.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let r2m = r2.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let mut o2m = out2.view_mut().into_dimensionality::<ndarray::Ix2>().unwrap();
        // Threading model, in two independent layers:
        //
        // 1. `gemm_row_banded` fans the GEMM out over rayon on disjoint bands
        //    of output ROWS, with OpenBLAS pinned to 1 inside. This is
        //    bit-identical to the serial product BY CONSTRUCTION (disjoint
        //    writes, unchanged per-element k-order) and thread-count
        //    independent, so it is ON by default. Below its size threshold, or
        //    inside an enclosing rayon region, it degrades to the serial
        //    `gemm_kblocked` with the caller's opt-in BLAS count — i.e. the
        //    exact pre-existing behavior.
        //
        // 2. The OpenBLAS opt-in (`FERRIC_BLAS_THREADS`) is unchanged and still
        //    applies on that serial path. It parallelizes over m/n INSIDE a
        //    block, so it does NOT preserve bit-identity: CCSD across
        //    FERRIC_BLAS_THREADS 1..12 shows 4 bit patterns (5 ulp, 6.0e-16
        //    rel). That is why it stays opt-in while (1) does not need to be.
        //
        // Orthogonally, the k-axis is blocked in OUR code (see
        // `gemm_kblocked`): coarse pairwise summation, measured 1.10-1.40x more
        // accurate than one full-k GEMM against a compensated reference. That
        // is an unconditional win at ANY thread count, and every row band runs
        // the same ascending-k block loop.
        gemm_row_banded(&l2m, &r2m, &mut o2m);
    }

    let want: usize = out_shape.iter().product::<usize>().max(1);
    let got = lf * rf;
    if got != want {
        return Err(TensorError::OutputReshape { got, want });
    }
    let out = out2
        .into_shape_with_order(IxDyn(out_shape))
        .map_err(|_| TensorError::OutputReshape { got, want })?;
    Ok(out)
}

/// Contract two operands with optional *batch* (diagonal) axes and a scale.
///
/// Batch axes are indices that appear in BOTH inputs and the OUTPUT — they are
/// iterated element-wise rather than contracted or outer-product'd. For each
/// fixed batch index the remaining free/contracted axes form an independent
/// contraction (`ij,ij->ij` Hadamard, `kij,kij->ij` per-element dot over `k`).
/// The whole result is multiplied by `scale`.
///
/// - `*_batch`: batch axis positions in each operand, in the SAME logical order
///   (the macro guarantees the orders match). `out_batch` gives their positions
///   in the output; batch dims always lead the output (`batch..., left_free...,
///   right_free...`), which the macro enforces.
/// - When `*_batch` is empty this is exactly [`einsum_binary`] times `scale`.
#[allow(clippy::too_many_arguments)]
pub fn einsum_binary_batched(
    left: ArrayViewD<f64>,
    left_batch: &[usize],
    left_free: &[usize],
    left_contr: &[usize],
    right: ArrayViewD<f64>,
    right_batch: &[usize],
    right_free: &[usize],
    right_contr: &[usize],
    out_shape: &[usize],
    scale: f64,
) -> Result<ArrayD<f64>, TensorError> {
    // Fast path: no batch axes -> one GEMM, then scale.
    if left_batch.is_empty() {
        let mut out =
            einsum_binary(left, left_free, left_contr, right, right_free, right_contr, out_shape)?;
        if scale != 1.0 {
            out.mapv_inplace(|x| x * scale);
        }
        return Ok(out);
    }

    // Batched path. Reshape each operand to (batch, free, contr): a 3D view
    // where index 0 walks the (flattened) batch space. Then for each batch
    // slice do one GEMM of (free x contr)·(contr x rfree) -> (free x rfree).
    let l3 = to_3d(left, left_batch, left_free, left_contr, "left");
    let r3 = to_3d(right, right_batch, right_contr, right_free, "right");

    let nb = l3.shape()[0];
    let (lf, lc) = (l3.shape()[1], l3.shape()[2]);
    let (rc, rf) = (r3.shape()[1], r3.shape()[2]);
    if nb != r3.shape()[0] {
        // Batch extents disagree: treat as a contracted-dim style mismatch.
        return Err(TensorError::ContractedDimMismatch { left: nb, right: r3.shape()[0] });
    }
    if lc != rc {
        return Err(TensorError::ContractedDimMismatch { left: lc, right: rc });
    }

    let mut out3 = ArrayD::<f64>::zeros(IxDyn(&[nb, lf, rf]));
    {
        let l3m = l3.view().into_dimensionality::<ndarray::Ix3>().unwrap();
        let r3m = r3.view().into_dimensionality::<ndarray::Ix3>().unwrap();
        let mut o3m = out3.view_mut().into_dimensionality::<ndarray::Ix3>().unwrap();
        // Opt-in multi-threaded BLAS for every batch-slice GEMM in this loop
        // (default resolves to 1 — a no-op — unless FERRIC_BLAS_THREADS is
        // set). This is a plain sequential `for`, not a par_iter; the
        // runtime rayon-worker guard inside opt_in_blas_threads still forces
        // 1 if a caller ever nests this inside a rayon region, so the wrap
        // is safe either way. One resolve+set/restore for the whole loop
        // rather than per-slice.
        with_blas_threads(opt_in_blas_threads(), || {
            for b in 0..nb {
                let lb = l3m.index_axis(ndarray::Axis(0), b);
                let rb = r3m.index_axis(ndarray::Axis(0), b);
                let mut ob = o3m.index_axis_mut(ndarray::Axis(0), b);
                general_mat_mul(scale, &lb, &rb, 0.0, &mut ob);
            }
        });
    }

    let want: usize = out_shape.iter().product::<usize>().max(1);
    let got = nb * lf * rf;
    if got != want {
        return Err(TensorError::OutputReshape { got, want });
    }
    let out = out3
        .into_shape_with_order(IxDyn(out_shape))
        .map_err(|_| TensorError::OutputReshape { got, want })?;
    Ok(out)
}

/// Contraction-axis block size for [`gemm_kblocked`]. Matches
/// `three_index_source.rs`'s `DRESS_K_BLOCK`, which the DF dressing work
/// (`eb895df`) measured as the accuracy knee.
const GEMM_K_BLOCK: usize = 128;

/// `out = left · right`, accumulating over the contraction axis in fixed
/// blocks of [`GEMM_K_BLOCK`] rather than as one full-k GEMM.
///
/// **This is chosen for ACCURACY first, and reproducibility falls out of it.**
///
/// Blocking the k-axis is coarse pairwise summation: each block's partial
/// product is formed independently and the blocks are then added, so rounding
/// error grows like log(k/blk) instead of k. Measured against a
/// Neumaier-compensated reference (`benchmarks/harness/examples/
/// gemm_accuracy_vs_threads.rs`), max |error| over the product:
///
/// ```text
///   m=200 k=4000 n=200     full-k GEMM 1.421e-14    k-blocked(128) 1.066e-14   1.33x better
///   m=400 k=2000 n=400     full-k GEMM 1.121e-14    k-blocked(128) 7.994e-15   1.40x better
///   m=100 k=8000 n=100     full-k GEMM 1.954e-14    k-blocked(128) 1.776e-14   1.10x better
/// ```
///
/// The same measurement corrected an earlier misconception worth recording:
/// **threading dgemm does NOT cost accuracy.** At 1 vs 12 OpenBLAS threads the
/// max error is *identical* (1.421e-14 both, and at k=8000 the threaded result
/// is marginally BETTER at 1.821e-14 vs 1.954e-14) — only the bit pattern
/// moves. So the reflex "don't thread, it breaks determinism" was defending
/// the *serial full-k* order, which is neither the fastest nor the most
/// accurate of the candidates. Accuracy and reproducibility are separate
/// properties here, and this function buys the first one unconditionally.
///
/// **It does NOT buy the second, and that was a wrong hypothesis worth
/// recording.** Fixing the block boundaries in our own code does not make the
/// product thread-independent, because OpenBLAS also parallelizes over m/n —
/// not only over k — so it still splits work *inside* a 128-wide block.
/// Measured directly (k-blocked at 1 vs 4 vs 12 threads, compared to each
/// other rather than to serial full-k): bit-identical in only 1 of 6 cases.
/// End-to-end, CCSD across `FERRIC_BLAS_THREADS` 1/2/4/6/8/12 still shows 4
/// distinct bit patterns with the same 5-ulp spread as before. Anyone wanting
/// bit-reproducible threaded GEMM needs a reproducible BLAS, not blocking.
///
/// Larger blocks are not uniformly better or worse (256/512 measured at or
/// slightly above full-k); 128 is the knee that `eb895df` independently landed
/// on for the DF dressing, so both paths now share one constant and one
/// rationale.
fn gemm_kblocked(
    left: &ndarray::ArrayView2<f64>,
    right: &ndarray::ArrayView2<f64>,
    out: &mut ndarray::ArrayViewMut2<f64>,
) {
    let k = left.ncols();
    if k <= GEMM_K_BLOCK {
        general_mat_mul(1.0, left, right, 0.0, out);
        return;
    }
    let mut k0 = 0;
    while k0 < k {
        let k1 = (k0 + GEMM_K_BLOCK).min(k);
        let lb = left.slice(ndarray::s![.., k0..k1]);
        let rb = right.slice(ndarray::s![k0..k1, ..]);
        // beta = 0 on the first block (initializes `out`), 1 thereafter, so
        // the blocks accumulate in ascending-k order — fixed, and independent
        // of how many threads OpenBLAS uses inside each block.
        let beta = if k0 == 0 { 0.0 } else { 1.0 };
        general_mat_mul(1.0, &lb, &rb, beta, out);
        k0 = k1;
    }
}

/// Number of output ROWS per rayon band in [`gemm_row_banded`].
///
/// Smallest band width worth using. Below this a band's GEMM stops being
/// cache-blocked: it re-streams the whole `right` operand with no reuse.
/// Measured at `RAYON_NUM_THREADS=1`, where banding can only cost, width 16 on
/// a 2500³ product runs at **0.35x** of the unbanded GEMM and width 32 at
/// 0.55x. See `gemm_band_width_sweep` (`--ignored`).
const GEMM_ROW_BAND_MIN: usize = 32;

/// Largest band width worth using: past this a band is big enough that BLAS is
/// already blocking inside it, and widening further only starves workers.
const GEMM_ROW_BAND_MAX: usize = 256;

/// **THE bit-identity invariant of the whole banded path: a band width must be
/// EVEN.**
///
/// Splitting a GEMM by rows is mathematically exact — row `i` of `A·B` depends
/// only on row `i` of `A` — but OpenBLAS does not honor that at the bit level.
/// Its dgemm microkernel unrolls the `m` axis by 2, so a band with an ODD row
/// count leaves a 1-row remainder that is finished by a different code path
/// whose k-accumulation order differs. The effect is large, not a last-ulp
/// wobble: max deviation 3.3e-2 on the probe data.
///
/// This was measured exhaustively, not assumed. For every band width `1..=m` on
/// three shapes (301x1000x97, 128x2000x64, 65x3000x33), compared bitwise
/// against the unbanded product (`every_band_width_bit_identity_map`):
///
/// ```text
///   EVEN widths that differed:      []      (none, on any shape)
///   ODD widths that were identical: only the degenerate ones that produce
///                                   no actual split (width == m, or 127 on
///                                   m=128 where the 1-row tail merges back)
/// ```
///
/// An earlier version of this code checked only powers of two — which all pass
/// — and concluded "any width >= 2 is safe". That was wrong, and the
/// exhaustive map is what caught it. Do not narrow this test back to a sample.
///
/// Because *every* even width is bit-identical, the width is a pure
/// PERFORMANCE knob and may therefore depend on the worker count (see
/// [`row_band_width`]) without affecting any result. This is a genuine
/// difference from `ferric_scf::reduce`, where the band width changes the FOLD
/// ORDER and so must be shape-pure: a row band accumulates nothing across
/// bands, so there is no order to change.
const fn is_valid_band_width(w: usize) -> bool {
    w.is_multiple_of(2)
}

/// Band width for an `m`-row output on a machine with `workers` rayon threads.
///
/// Aims for ~4 bands per worker (enough for rayon to load-balance without
/// making bands so thin they lose cache blocking), then clamps into
/// `[GEMM_ROW_BAND_MIN, GEMM_ROW_BAND_MAX]` and **rounds UP to even** — the
/// bit-identity invariant ([`is_valid_band_width`]).
///
/// Depending on `workers` is safe here ONLY because every even width gives a
/// bit-identical answer; it changes how the work is divided, never the result.
/// [`row_band_bounds`] additionally guarantees no band is left 1 row wide.
fn row_band_width(m: usize, workers: usize) -> usize {
    let target_bands = workers.max(1).saturating_mul(4);
    let w = m.div_ceil(target_bands);
    let w = w.clamp(GEMM_ROW_BAND_MIN, GEMM_ROW_BAND_MAX);
    // Round up to even. The invariant, not a nicety.
    let w = w + (w % 2);
    debug_assert!(is_valid_band_width(w));
    w
}

/// Row-band boundaries for an `m`-row output at band width `width`.
///
/// Two rules, both load-bearing for bit-identity rather than performance:
///  1. `width` must be EVEN — see [`is_valid_band_width`].
///  2. No band may be left 1 row wide. A lone trailing row is the degenerate
///     `m == 1` GEMM, which OpenBLAS dispatches differently; it is absorbed
///     into the preceding band, making that band `width + 1` rows.
///
/// Rule 2 makes the final band ODD, which rule 1 would seem to forbid. It does
/// not, and this was MEASURED rather than reasoned: the parity effect comes
/// from how the microkernel finishes a band's own row remainder, and a merged
/// `width + 1` tail is bit-identical to the unbanded product on every shape and
/// width tested. `every_even_band_width_is_bit_identical` runs these exact
/// production bounds (and asserts the merged case is actually reachable, so the
/// check cannot pass vacuously).
fn row_band_bounds(m: usize, width: usize) -> Vec<(usize, usize)> {
    debug_assert!(is_valid_band_width(width), "band width {width} must be even");
    let mut bounds = Vec::new();
    let mut r0 = 0;
    while r0 < m {
        let mut r1 = (r0 + width).min(m);
        // A 1-row trailing band would be odd-width and change that row's bits.
        // Absorb it into this band.
        if m - r1 == 1 {
            r1 = m;
        }
        bounds.push((r0, r1));
        r0 = r1;
    }
    bounds
}

/// Minimum FLOP count (2·m·n·k) before row-banding a GEMM over rayon is worth
/// the dispatch cost. Mirrors the [`PAR_PERMUTE_MIN_ELEMS`] convention: below
/// this the whole product is a few tens of microseconds serially and the fan-out
/// loses. Calibrated from the measured single-thread GEMM rate on this box
/// (~30-38 GFLOP/s): 4 MFLOP is ~0.1 ms, comfortably above rayon's few-µs
/// dispatch. Measured crossover (see the benchmark table in the commit) sits
/// below this, so the threshold is conservative.
const PAR_GEMM_MIN_FLOPS: usize = 4 << 20;

/// `out = left · right`, computed by [`gemm_kblocked`] on disjoint bands of
/// output ROWS in parallel.
///
/// **Bit-identical to the serial [`gemm_kblocked`]**, and therefore independent
/// of thread count. Three facts, the third of which is empirical and was NOT
/// obvious:
///
///  1. Row `i` of `left · right` depends only on row `i` of `left` and on
///     `right` — no output element's value depends on which other rows are
///     computed alongside it. Restricting the GEMM to a row slab does not
///     change any element's k-summation order: each band runs the SAME
///     ascending-k block loop over the SAME `GEMM_K_BLOCK` boundaries.
///  2. The bands partition `0..m` into disjoint contiguous ranges, so every
///     output element is written by exactly one worker. There is no
///     accumulation ACROSS workers, hence no reduction tree whose shape could
///     depend on the worker count. This is the same disjoint-write argument
///     [`permute_to_owned`] already relies on, applied to a GEMM.
///  3. **(1) is a statement about MATHEMATICS, and OpenBLAS does not honor it
///     unconditionally.** Its dgemm microkernel unrolls the `m` axis by 2, so
///     a band with an ODD row count finishes its last row on a different code
///     path whose k-accumulation order differs. The effect is large (max dev
///     3.3e-2), not a last-ulp wobble. The exactness anchor caught it on the
///     very first run (m=33 banded 32+1: exactly the 1-row tail differed).
///
///     The invariant is therefore **every band must have an EVEN row count**,
///     enforced by [`is_valid_band_width`] + [`row_band_bounds`]'s 1-row-tail
///     merge, and verified exhaustively over every width `1..=m` on three
///     shapes by `every_even_band_width_is_bit_identical` (even widths: zero
///     differences anywhere; odd widths: differ).
///
/// **Band boundaries need NOT be a pure function of shape here, and are not.**
/// [`row_band_width`] takes the worker count, which is a deliberate departure
/// from the `ferric_scf::reduce` / `ferric_cc::ccsd_t` house rule — and it is
/// safe for a reason that does not apply there: those band widths change the
/// FOLD ORDER of an accumulation, whereas a row band accumulates nothing across
/// bands, so per (3) any even width yields the identical bit pattern. The width
/// only decides how work is divided. `row_banded_gemm_is_thread_count_independent`
/// pins that by running every worker count's actual bounds and requiring one
/// bit pattern.
///
/// Adapting the width to the worker count is not gold-plating: a fixed width
/// cannot serve both ends. Measured at width 32, one worker on a 2500³ product
/// runs at **0.55x** of the unbanded GEMM (thin bands lose cache blocking),
/// while at width 256 twelve workers lose the small/medium shapes entirely
/// (400-row case 0.97x vs 2.64x at width 32).
///
/// **OpenBLAS is pinned to 1 inside the region.** `with_blas_threads(1, ..)` is
/// evaluated on the CALLER thread, before the `par_iter`, so it must be the
/// literal `1` — calling `opt_in_blas_threads()` here would return the user's
/// `FERRIC_BLAS_THREADS` (its rayon-worker self-guard only fires when called
/// FROM a worker) and raise the global count for the whole parallel region,
/// which is the rayon×OpenBLAS oversubscription / worker-stack-overflow hazard
/// documented in `ferric-core/src/blas_threads.rs`. This is the same bug that
/// was just fixed in `ferric_cc::ccsd_t`.
///
/// **No per-worker allocation.** Workers write disjoint mutable views of the
/// caller's existing `out`; nothing is allocated per band, so the memory story
/// is unchanged from the serial path.
///
/// Below [`PAR_GEMM_MIN_FLOPS`], or when there is at most one band, or when
/// already running inside a rayon worker (an enclosing parallel region owns the
/// threads — nesting would oversubscribe), this degrades to the serial
/// [`gemm_kblocked`] with the caller's existing opt-in BLAS behavior intact.
fn gemm_row_banded(
    left: &ndarray::ArrayView2<f64>,
    right: &ndarray::ArrayView2<f64>,
    out: &mut ndarray::ArrayViewMut2<f64>,
) {
    use rayon::prelude::*;

    let (m, k) = (left.nrows(), left.ncols());
    let n = right.ncols();
    let workers = rayon::current_num_threads();
    let width = row_band_width(m, workers);
    let flops = 2usize.saturating_mul(m).saturating_mul(n).saturating_mul(k);
    let serial = flops < PAR_GEMM_MIN_FLOPS
        || m <= width
        // One worker: banding can only lose. A band is a strictly worse-blocked
        // GEMM than the whole product, so with no concurrency to buy back the
        // lost cache reuse it is pure overhead — measured 0.55x at width 32 on
        // 2500³, and still 0.93x at width 256. Safe to branch on the worker
        // count because every even width is bit-identical (see
        // `is_valid_band_width`), so this changes only speed, never the answer.
        || workers < 2
        // Already inside a parallel region: the enclosing rayon region owns the
        // threads and this GEMM is one of many running concurrently. Banding
        // here would nest par_iter inside par_iter for no gain.
        || rayon::current_thread_index().is_some();
    if serial {
        // Preserve the EXISTING opt-in BLAS behavior for the serial path
        // (default 1; raised only by FERRIC_BLAS_THREADS, and forced back to 1
        // by the resolver's own guard when this runs on a rayon worker).
        with_blas_threads(opt_in_blas_threads(), || gemm_kblocked(left, right, out));
        return;
    }

    // Split `out` into the disjoint row bands `row_band_bounds` prescribes.
    // Bands are not uniformly wide (the 1-row-tail merge), so peel them off the
    // front with successive `split_at(Axis(0), ..)` calls — each yields two
    // non-aliasing halves, and we keep the left one as a band.
    //
    // `ndarray`'s own `rayon` feature is not enabled in this workspace (see
    // `permute_to_owned`), so we drive rayon over a `Vec` of these DISJOINT,
    // non-aliasing `ArrayViewMut2`s. That adds no tensor allocation — the Vec
    // holds ~m/32 view headers (pointer + shape + strides), not data.
    let bounds = row_band_bounds(m, width);
    let mut bands: Vec<ndarray::ArrayViewMut2<f64>> = Vec::with_capacity(bounds.len());
    {
        let mut rest = out.view_mut();
        for &(r0, r1) in &bounds {
            let (head, tail) = rest.split_at(ndarray::Axis(0), r1 - r0);
            bands.push(head);
            rest = tail;
        }
        debug_assert_eq!(rest.nrows(), 0, "row bands must cover every output row exactly once");
    }
    with_blas_threads(1, || {
        bands.par_iter_mut().zip(bounds.par_iter()).for_each(|(band, &(r0, r1))| {
            let lb = left.slice(ndarray::s![r0..r1, ..]);
            gemm_kblocked(&lb, right, band);
        });
    });
}

/// Minimum element count before the permutation copy is worth fanning out over
/// rayon. Below this the dispatch overhead dominates a copy that is only a few
/// microseconds serially.
///
/// Calibrated on the measured copy rate (~2.5 GB/s single-threaded, i.e.
/// ~3e8 f64/s): 1<<16 elements is ~0.2 ms of work, comfortably above rayon's
/// few-µs dispatch cost while still catching every production-scale
/// contraction. Small unit-test tensors stay on the serial path.
const PAR_PERMUTE_MIN_ELEMS: usize = 1 << 16;

/// Copy `permuted` into a fresh C-contiguous array, fanning the gather out over
/// rayon when the operand is large enough to pay for the dispatch.
///
/// This is the hot part of `einsum!`. 83% of the workspace's ~130 einsum specs
/// need at least one operand permuted before it can be viewed as a 2D matrix,
/// and the copy is a strided gather that ndarray performs single-threaded via
/// `as_standard_layout().into_owned()`. Measured on `ijcd,abcd->ijab` at
/// no=10: the copy is 47% of the contraction at nv=40 and **70% at nv=80** —
/// it is memory-bandwidth-bound and gets relatively WORSE with size, while the
/// GEMM beside it holds ~38 GFLOP/s.
///
/// Parallelism is over the outermost axis of the *output* (post-permutation)
/// array, so each worker writes a disjoint contiguous slab and reads a strided
/// slice of the source. No accumulation happens here — this is a pure data
/// movement — so unlike a reduction it is exactly bit-identical regardless of
/// thread count: every output element is written exactly once, by one worker,
/// with the same value. (Contrast the summation-order discipline that governs
/// the GEMM paths.)
///
/// BLAS is untouched: this is rayon-only, so it composes with the
/// `blas_threads` hazard model (rayon owns parallelism, OpenBLAS pinned to 1)
/// without needing a thread-count decision.
pub fn permute_to_owned(permuted: ndarray::ArrayViewD<f64>) -> ArrayD<f64> {
    use rayon::prelude::*;

    // Already contiguous in the requested order: ndarray hands back a view with
    // no copy at all, so there is nothing to parallelize.
    if permuted.is_standard_layout() {
        return permuted.to_owned();
    }
    let shape = permuted.shape().to_vec();
    let n: usize = shape.iter().product();
    if n < PAR_PERMUTE_MIN_ELEMS || shape.is_empty() || shape[0] < 2 {
        return permuted.as_standard_layout().into_owned();
    }

    // Fan out over the outermost axis by splitting the flat output buffer into
    // equal per-slab chunks. `ndarray`'s own `rayon` feature is not enabled in
    // this workspace, so rather than turn it on for every crate we drive rayon
    // over the raw output slice and re-view each chunk — the output IS standard
    // layout by construction, so slab `k` is exactly `chunk k` of the buffer.
    let n0 = shape[0];
    let slab: usize = shape[1..].iter().product::<usize>().max(1);
    let mut out = ArrayD::<f64>::zeros(IxDyn(&shape));
    let buf = out.as_slice_mut().expect("freshly allocated ArrayD is contiguous");
    buf.par_chunks_mut(slab).enumerate().for_each(|(k, chunk)| {
        let src = permuted.index_axis(ndarray::Axis(0), k);
        // Each (n-1)-dim source slab is still strided; let ndarray's own
        // optimized assign drive the inner gather into the contiguous chunk.
        let mut dst = ndarray::ArrayViewMutD::from_shape(IxDyn(&shape[1..]), chunk)
            .expect("chunk length matches slab shape by construction");
        dst.assign(&src);
    });
    debug_assert_eq!(n0 * slab, permuted.len());
    out
}

/// Permute `op` to (batch, second, third) axis order and reshape to the 3D
/// shape (prod(batch), prod(second), prod(third)), copying to contiguous when
/// the permutation is non-trivial.
fn to_3d(
    op: ArrayViewD<f64>,
    batch: &[usize],
    second: &[usize],
    third: &[usize],
    which: &str,
) -> ArrayD<f64> {
    let order: Vec<usize> =
        batch.iter().chain(second.iter()).chain(third.iter()).copied().collect();
    let is_identity = order.iter().enumerate().all(|(i, &p)| i == p);
    if !is_identity {
        log::debug!("einsum: {which} operand permuted to {:?} (transpose copy)", order);
    }
    let nb: usize = batch.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let d1: usize = second.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let d2: usize = third.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let permuted = op.permuted_axes(order);
    permute_to_owned(permuted.view())
        .into_shape_with_order(IxDyn(&[nb, d1, d2]))
        .expect("einsum: 3D reshape after permutation")
}

/// A 2D operand for GEMM: either a borrowed view onto the caller's data (no
/// copy) or an owned permuted array. `view()` yields the matrix either way.
enum Operand2d<'a> {
    /// Zero-copy: the operand was already usable as a matrix, possibly only
    /// after a transpose that `general_mat_mul` absorbs into dgemm's
    /// `transa`/`transb` flag.
    Borrowed(ndarray::ArrayView2<'a, f64>),
    /// The axis order was genuinely interleaved, so a gather was unavoidable.
    Owned(ArrayD<f64>),
}

impl Operand2d<'_> {
    fn view(&self) -> ndarray::ArrayView2<'_, f64> {
        match self {
            Operand2d::Borrowed(v) => v.reborrow(),
            Operand2d::Owned(a) => {
                a.view().into_dimensionality::<ndarray::Ix2>().expect("2D by construction")
            }
        }
    }
}

/// Like [`to_2d`], but avoids the permutation copy whenever the required axis
/// order is reachable by a pure transpose.
///
/// Three cases, in order:
///  1. The operand is already in `(first..., second...)` order → view it as
///     `(prod(first), prod(second))`, no copy.
///  2. It is in `(second..., first...)` order → view it as
///     `(prod(second), prod(first))` and TRANSPOSE the view. ndarray's
///     `general_mat_mul` maps a transposed view onto dgemm's `transa`/`transb`
///     flag, so BLAS reads the original memory with a stride — still no copy.
///  3. Anything else (genuinely interleaved axes) → fall back to [`to_2d`].
///
/// Case 2 is what pays: 72 of the workspace's ~130 specs have at least one
/// operand there, including the O(N^6) CCSD VVVV term `ijcd,abcd->ijab`.
///
/// The reshape in cases 1 and 2 requires the underlying data to be contiguous
/// in the grouped order, which is exactly `is_standard_layout()` on the
/// correspondingly-permuted view; when that does not hold (a non-contiguous
/// input slice) we fall back to the copy rather than reshape a strided view.
fn to_2d_or_transpose<'a>(
    op: ArrayViewD<'a, f64>,
    first: &[usize],
    second: &[usize],
    which: &str,
) -> Operand2d<'a> {
    let rows: usize = first.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let cols: usize = second.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);

    // `into_shape_with_order` consumes the view, so hand it a clone — an
    // ArrayView clone is just a pointer + shape + strides, not the data.
    let fwd: Vec<usize> = first.iter().chain(second.iter()).copied().collect();
    if fwd.iter().enumerate().all(|(i, &p)| i == p) && op.is_standard_layout() {
        if let Ok(v) = op.clone().into_shape_with_order(ndarray::Ix2(rows, cols)) {
            return Operand2d::Borrowed(v);
        }
    }

    // Transposed grouping: the operand is laid out as (second..., first...).
    let rev: Vec<usize> = second.iter().chain(first.iter()).copied().collect();
    if rev.iter().enumerate().all(|(i, &p)| i == p) && op.is_standard_layout() {
        if let Ok(v) = op.clone().into_shape_with_order(ndarray::Ix2(cols, rows)) {
            log::debug!("einsum: {which} operand consumed transposed (no copy)");
            return Operand2d::Borrowed(v.reversed_axes());
        }
    }

    Operand2d::Owned(to_2d(op, first, second, which))
}

/// Permute `op` so the axes in `first` come before the axes in `second`, then
/// reshape to 2D (prod(first), prod(second)). Copies to contiguous when the
/// permutation is non-trivial; logs the copy at debug level.
fn to_2d(op: ArrayViewD<f64>, first: &[usize], second: &[usize], which: &str) -> ArrayD<f64> {
    let order: Vec<usize> = first.iter().chain(second.iter()).copied().collect();
    let is_identity = order.iter().enumerate().all(|(i, &p)| i == p);
    if !is_identity {
        log::debug!("einsum: {which} operand permuted to {:?} (transpose copy)", order);
    }
    let rows: usize = first.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let cols: usize = second.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let permuted = op.permuted_axes(order);
    // The result must be C-contiguous so `into_shape_with_order` (which requires
    // standard layout) succeeds; `permute_to_owned` guarantees that.
    permute_to_owned(permuted.view())
        .into_shape_with_order(IxDyn(&[rows, cols]))
        .expect("einsum: 2D reshape after permutation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};

    fn naive_matmul(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let (ni, nk) = (a.shape()[0], a.shape()[1]);
        let nj = b.shape()[1];
        let mut c = Array::zeros(IxDyn(&[ni, nj]));
        for i in 0..ni {
            for j in 0..nj {
                let mut s = 0.0;
                for k in 0..nk {
                    s += a[[i, k]] * b[[k, j]];
                }
                c[[i, j]] = s;
            }
        }
        c
    }

    /// `gemm_kblocked` blocks the contraction axis, which is coarse pairwise
    /// summation and therefore MORE accurate than one full-k GEMM — the reason
    /// it was chosen. This pins that claim against a Neumaier-compensated
    /// reference so a future "simplification" back to a single
    /// `general_mat_mul` cannot silently give up the accuracy.
    ///
    /// Uses a large k (that is where accumulation error lives) and operands of
    /// mixed magnitude, which is what makes summation order matter at all —
    /// uniformly-scaled random data barely distinguishes the orderings.
    #[test]
    fn kblocked_gemm_is_at_least_as_accurate_as_full_k() {
        use ndarray::Array2;

        let (m, k, n) = (24usize, 3000usize, 24usize);
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        // Mixed magnitudes: a few large terms among many small ones is the
        // classic case where naive left-to-right accumulation loses digits.
        let a = Array2::from_shape_fn((m, k), |(_, p)| {
            let v = rnd();
            if p % 97 == 0 { v * 1e6 } else { v }
        });
        let b = Array2::from_shape_fn((k, n), |(p, _)| {
            let v = rnd();
            if p % 89 == 0 { v * 1e6 } else { v }
        });

        // Neumaier-compensated reference.
        let bt = b.t().as_standard_layout().into_owned();
        let mut want = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let (mut sum, mut c) = (0.0f64, 0.0f64);
                for p in 0..k {
                    let prod = a[[i, p]] * bt[[j, p]];
                    let t = sum + prod;
                    if sum.abs() >= prod.abs() {
                        c += (sum - t) + prod;
                    } else {
                        c += (prod - t) + sum;
                    }
                    sum = t;
                }
                want[[i, j]] = sum + c;
            }
        }

        let err_of = |got: &Array2<f64>| -> f64 {
            got.iter().zip(want.iter()).map(|(g, w)| (g - w).abs()).fold(0.0f64, f64::max)
        };

        let mut full = Array2::<f64>::zeros((m, n));
        general_mat_mul(1.0, &a.view(), &b.view(), 0.0, &mut full.view_mut());

        let mut blocked = Array2::<f64>::zeros((m, n));
        gemm_kblocked(&a.view(), &b.view(), &mut blocked.view_mut());

        let (e_full, e_blk) = (err_of(&full), err_of(&blocked));
        // Measured margin on this data is 1.167x (blocked 3.66e-4 vs full-k
        // 4.27e-4). Require a real improvement, not merely "no worse": a
        // >=1.05x floor still fails immediately if the blocking is removed
        // (which lands at ratio 1.000), while leaving headroom for BLAS
        // version differences. A "<= full * 1.05" style bound would NOT catch
        // that regression -- verified by mutation.
        let ratio = e_full / e_blk.max(1e-300);
        assert!(
            ratio >= 1.05,
            "k-blocked GEMM must be measurably MORE accurate than full-k \
             (blocked {e_blk:.3e} vs full-k {e_full:.3e}, ratio {ratio:.3} \
             < 1.05); if this fails, the k-blocking that justifies \
             gemm_kblocked's existence has been lost",
        );
        // And it must still be a correct product, not merely a precise one.
        assert!(
            e_blk / want.iter().fold(0.0f64, |m, w| m.max(w.abs())) < 1e-13,
            "k-blocked GEMM relative error {e_blk:.3e} is too large to be a \
             correct product",
        );
    }

    /// Deterministic mixed-magnitude test matrices. Mixed magnitudes matter:
    /// uniformly-scaled data can mask an ordering change because the low bits
    /// happen to agree, so a bit-identity test on it is weaker than it looks.
    fn gemm_pair(m: usize, k: usize, n: usize, seed: u64) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
        use ndarray::Array2;
        let mut s = seed | 1;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let a = Array2::from_shape_fn((m, k), |(_, p)| {
            let v = rnd();
            if p % 37 == 0 { v * 1e7 } else { v }
        });
        let b = Array2::from_shape_fn((k, n), |(p, _)| {
            let v = rnd();
            if p % 23 == 0 { v * 1e7 } else { v }
        });
        (a, b)
    }

    /// Shapes exercised by both bit-identity tests. Deliberately covers:
    /// tall / wide / square; `m` smaller than any plausible worker count;
    /// `m` NOT divisible by the band width (short last band); `k` both above
    /// and below `GEMM_K_BLOCK`; and the 1-row / 1-col degenerate cases.
    fn banding_shape_cases() -> Vec<(usize, usize, usize, &'static str)> {
        vec![
            (1, 1, 1, "1x1x1 scalar-ish"),
            (1, 4000, 1, "single element, k >> block"),
            (1, 4000, 300, "one row (m < band)"),
            (300, 4000, 1, "one col"),
            (3, 4000, 300, "m smaller than worker count"),
            (33, 4000, 200, "m = band+1 (1-row tail must be merged)"),
            (65, 3000, 200, "m = 2*band+1 (1-row tail must be merged)"),
            (34, 4000, 200, "m = band+2 (smallest legal 2-row tail)"),
            (64, 4000, 64, "m divisible by band"),
            (200, 100, 200, "k BELOW GEMM_K_BLOCK"),
            (200, 128, 200, "k EXACTLY GEMM_K_BLOCK"),
            (200, 129, 200, "k one past GEMM_K_BLOCK"),
            (301, 1000, 97, "tall, non-divisible m, prime-ish n"),
            (97, 3000, 401, "wide"),
            (256, 2048, 256, "square, all powers of two"),
        ]
    }

    /// `row_band_bounds` must (a) exactly partition `0..m` in ascending order
    /// and (b) NEVER emit a 1-row band, for every `m` and every even width.
    #[test]
    fn row_band_bounds_partition_and_never_one_row() {
        for width in [2usize, 4, 32, 64, 256] {
            for m in 0..600usize {
                let b = row_band_bounds(m, width);
                // (a) exact ascending partition of 0..m.
                let mut expect = 0usize;
                for &(r0, r1) in &b {
                    assert_eq!(r0, expect, "m={m} w={width}: band starts {r0}, want {expect}");
                    assert!(r1 > r0, "m={m} w={width}: empty band {r0}..{r1}");
                    expect = r1;
                }
                assert_eq!(expect, m, "m={m} w={width}: cover 0..{expect}, not 0..{m}");
                // (b) no 1-row band (m==1 aside: the whole product is one row,
                // and the serial path handles it).
                if m > 1 {
                    for &(r0, r1) in &b {
                        assert!(
                            r1 - r0 >= 2,
                            "m={m} w={width}: band {r0}..{r1} is 1 row wide",
                        );
                    }
                }
            }
        }
    }

    /// `row_band_width` must ALWAYS return an even width — that is the
    /// bit-identity invariant (`is_valid_band_width`), and it must hold for
    /// every `m` and every worker count, including the ones that hit the
    /// clamp boundaries.
    #[test]
    fn row_band_width_is_always_even() {
        for workers in [1usize, 2, 3, 4, 6, 8, 12, 16, 64, 128] {
            for m in 0..2000usize {
                let w = row_band_width(m, workers);
                assert!(
                    is_valid_band_width(w),
                    "m={m} workers={workers}: band width {w} is ODD — OpenBLAS's \
                     dgemm m-unroll of 2 makes odd bands non-bit-identical",
                );
                assert!(
                    (GEMM_ROW_BAND_MIN..=GEMM_ROW_BAND_MAX + 1).contains(&w),
                    "m={m} workers={workers}: band width {w} out of clamp range",
                );
            }
        }
    }

    /// THE load-bearing invariant: **every EVEN band width is bit-identical to
    /// the unbanded product, and odd widths are not.** This is what licenses
    /// `row_band_width` to depend on the worker count — the width becomes a
    /// pure PERFORMANCE knob that cannot move the answer.
    ///
    /// (Contrast `ferric_scf::reduce`, where band width changes the fold ORDER
    /// and so MUST be shape-pure. A row band accumulates nothing across bands,
    /// so there is no order to change — only OpenBLAS's internal m-unroll of 2
    /// matters, hence the parity rule.)
    ///
    /// This sweeps EVERY width `1..=m`, not a sample. An earlier version
    /// checked only powers of two, which all pass, and wrongly concluded that
    /// any width >= 2 was safe; width 3 differs on 7809/29197 elements. Do not
    /// narrow this back to a sample.
    ///
    /// It also asserts the two claims are each REACHABLE — that at least one
    /// odd width actually differed, and that the production `row_band_bounds`
    /// merged-tail case (which makes the last band odd) was actually
    /// exercised. A pass condition that never fires is arithmetic, not
    /// measurement.
    #[test]
    fn every_even_band_width_is_bit_identical() {
        use ndarray::Array2;

        fn banded_at(
            width: usize,
            left: &ndarray::ArrayView2<f64>,
            right: &ndarray::ArrayView2<f64>,
            out: &mut ndarray::ArrayViewMut2<f64>,
        ) {
            let m = left.nrows();
            let mut r0 = 0;
            while r0 < m {
                let mut r1 = (r0 + width).min(m);
                if m - r1 == 1 && width > 1 {
                    r1 = m;
                }
                let lb = left.slice(ndarray::s![r0..r1, ..]);
                let mut ob = out.slice_mut(ndarray::s![r0..r1, ..]);
                gemm_kblocked(&lb, right, &mut ob);
                r0 = r1;
            }
        }

        for (m, k, n) in [(301usize, 1000usize, 97usize), (128, 2000, 64), (65, 3000, 33)] {
            let (a, b) = gemm_pair(m, k, n, 0xb17_1de4 ^ m as u64);
            let mut want = Array2::<f64>::zeros((m, n));
            gemm_kblocked(&a.view(), &b.view(), &mut want.view_mut());

            // (1) Exhaustive parity map over every width.
            let mut odd_that_differed = 0usize;
            for width in 1..=m {
                let mut got = Array2::<f64>::zeros((m, n));
                banded_at(width, &a.view(), &b.view(), &mut got.view_mut());
                let bad = got
                    .iter()
                    .zip(want.iter())
                    .filter(|(g, w)| g.to_bits() != w.to_bits())
                    .count();
                if width % 2 == 0 {
                    assert_eq!(
                        bad, 0,
                        "({m}x{k}x{n}) EVEN band width {width}: {bad}/{} elements differ \
                         from the unbanded product. Even-width bit-identity is the \
                         invariant that lets the band width depend on thread count; if \
                         it fails, gemm_row_banded is unsound and must be reverted.",
                        m * n,
                    );
                } else if bad > 0 {
                    odd_that_differed += 1;
                }
            }
            // Reachability: the odd-width failure mode must be real, or the
            // parity rule is guarding against nothing.
            assert!(
                odd_that_differed > 0,
                "({m}x{k}x{n}) NO odd band width differed from the unbanded product. \
                 The OpenBLAS m-unroll-of-2 effect that is_valid_band_width guards \
                 against appears to be gone — re-measure before trusting the parity rule.",
            );

            // (2) The PRODUCTION bounds, whose 1-row-tail merge deliberately
            // makes the final band ODD (width+1). That band is a merged tail,
            // not an independently-dispatched odd band, so it must still be
            // bit-identical. Measured, not argued.
            let mut merged_cases = 0usize;
            for width in (2..=m.min(64)).step_by(2) {
                let bounds = row_band_bounds(m, width);
                if bounds.iter().any(|&(r0, r1)| (r1 - r0) % 2 == 1) {
                    merged_cases += 1;
                }
                let mut got = Array2::<f64>::zeros((m, n));
                for &(r0, r1) in &bounds {
                    let lb = a.slice(ndarray::s![r0..r1, ..]);
                    let mut ob = got.slice_mut(ndarray::s![r0..r1, ..]);
                    gemm_kblocked(&lb, &b.view(), &mut ob);
                }
                let bad = got
                    .iter()
                    .zip(want.iter())
                    .filter(|(g, w)| g.to_bits() != w.to_bits())
                    .count();
                assert_eq!(
                    bad, 0,
                    "({m}x{k}x{n}) production row_band_bounds at width {width}: \
                     {bad}/{} elements differ",
                    m * n,
                );
            }
            println!(
                "({m}x{k}x{n}): even-width identity OK; {odd_that_differed} odd widths \
                 differed; {merged_cases} production widths had an odd merged tail band"
            );
        }
    }

    /// Width sweep (`--ignored`): for each shape, time the serial product and
    /// then an explicitly-banded product at a range of FIXED band widths, so
    /// the width/parallelism tradeoff is measured rather than guessed.
    #[test]
    #[ignore]
    fn gemm_band_width_sweep() {
        use ndarray::Array2;
        use rayon::prelude::*;
        use std::time::Instant;

        // Band at an explicit width, bypassing row_band_width. Same 1-row-tail
        // merge rule so every point in the sweep is bit-identical.
        fn banded_at(
            width: usize,
            left: &ndarray::ArrayView2<f64>,
            right: &ndarray::ArrayView2<f64>,
            out: &mut ndarray::ArrayViewMut2<f64>,
        ) {
            let m = left.nrows();
            let mut bounds = Vec::new();
            let mut r0 = 0;
            while r0 < m {
                let mut r1 = (r0 + width).min(m);
                if m - r1 == 1 {
                    r1 = m;
                }
                bounds.push((r0, r1));
                r0 = r1;
            }
            let mut bands: Vec<ndarray::ArrayViewMut2<f64>> = Vec::with_capacity(bounds.len());
            let mut rest = out.view_mut();
            for &(a, b) in &bounds {
                let (h, t) = rest.split_at(ndarray::Axis(0), b - a);
                bands.push(h);
                rest = t;
            }
            with_blas_threads(1, || {
                bands.par_iter_mut().zip(bounds.par_iter()).for_each(|(band, &(a, b))| {
                    let lb = left.slice(ndarray::s![a..b, ..]);
                    gemm_kblocked(&lb, right, band);
                });
            });
        }

        let shapes: Vec<(usize, usize, usize, &str)> = vec![
            (100, 600, 100, "no=nv=10 naux=600"),
            (400, 600, 400, "no=10 nv=40 naux=600"),
            (1600, 1200, 1600, "CCSD ovov-ish"),
            (2500, 2500, 2500, "square 2500"),
            (100, 20000, 100, "k-dominated"),
            (5000, 300, 300, "very tall small k"),
        ];
        let widths = [16usize, 32, 64, 128, 256, 512, 1024];
        println!("rayon threads = {}", rayon::current_num_threads());
        print!("{:<34} {:>9}", "shape", "serial");
        for w in widths {
            print!(" {:>8}", format!("w{w}"));
        }
        println!("   (speedup vs serial)");
        for (m, k, n, label) in shapes {
            let (a, b) = gemm_pair(m, k, n, 0x5eed ^ m as u64);
            let mut o = Array2::<f64>::zeros((m, n));
            let reps = if 2 * m * n * k > 2_000_000_000 { 2 } else { 5 };

            gemm_kblocked(&a.view(), &b.view(), &mut o.view_mut());
            let t = Instant::now();
            for _ in 0..reps {
                gemm_kblocked(&a.view(), &b.view(), &mut o.view_mut());
            }
            let ser = t.elapsed().as_secs_f64() / reps as f64;

            print!("{:<34} {:>8.2}m", format!("{label} [{m}x{k}x{n}]"), ser * 1e3);
            for w in widths {
                banded_at(w, &a.view(), &b.view(), &mut o.view_mut());
                let t = Instant::now();
                for _ in 0..reps {
                    banded_at(w, &a.view(), &b.view(), &mut o.view_mut());
                }
                let el = t.elapsed().as_secs_f64() / reps as f64;
                print!(" {:>7.2}x", ser / el);
            }
            println!();
        }
    }

    /// Microbenchmark (`--ignored`), not a regression test: serial
    /// `gemm_kblocked` vs `gemm_row_banded` at CCSD-representative shapes.
    /// Run with `--test-threads=1` and `OPENBLAS_NUM_THREADS=1`.
    #[test]
    #[ignore]
    fn gemm_banding_bench() {
        use ndarray::Array2;
        use std::time::Instant;

        let shapes: Vec<(usize, usize, usize, &str)> = vec![
            (100, 600, 100, "Pia,Pjb->iajb  no=nv=10, naux=600"),
            (400, 600, 400, "Pia,Pjb->iajb  no=10 nv=40, naux=600"),
            (1600, 1200, 1600, "CCSD ovov-ish  no=10 nv=40"),
            (2500, 2500, 2500, "square 2500 (VVVV-ish slab)"),
            (100, 20000, 100, "k-dominated: thin bands"),
            (40, 20000, 40, "k-dominated, tiny m"),
            (64, 4000, 4000, "small m, wide n"),
            (5000, 300, 300, "very tall, small k"),
        ];
        let threads = rayon::current_num_threads();
        println!("rayon threads = {threads}");
        println!("{:<38} {:>10} {:>10} {:>8}", "shape", "serial ms", "band ms", "speedup");
        for (m, k, n, label) in shapes {
            let (a, b) = gemm_pair(m, k, n, 0x5eed ^ m as u64);
            let mut o = Array2::<f64>::zeros((m, n));

            // Warm up + time serial.
            gemm_kblocked(&a.view(), &b.view(), &mut o.view_mut());
            let reps = if 2 * m * n * k > 2_000_000_000 { 2 } else { 5 };
            let t = Instant::now();
            for _ in 0..reps {
                gemm_kblocked(&a.view(), &b.view(), &mut o.view_mut());
            }
            let ser = t.elapsed().as_secs_f64() / reps as f64;

            gemm_row_banded(&a.view(), &b.view(), &mut o.view_mut());
            let t = Instant::now();
            for _ in 0..reps {
                gemm_row_banded(&a.view(), &b.view(), &mut o.view_mut());
            }
            let ban = t.elapsed().as_secs_f64() / reps as f64;

            println!(
                "{:<38} {:>10.2} {:>10.2} {:>7.2}x",
                format!("{label} [{m}x{k}x{n}]"),
                ser * 1e3,
                ban * 1e3,
                ser / ban
            );
        }
    }

    /// TEMPORARY diagnostic probe (`--ignored`), not a regression test.
    #[test]
    #[ignore]
    fn rowsplit_probe() {
        use ndarray::Array2;
        for (m, k, n) in [(33usize, 4000usize, 200usize), (64, 4000, 64), (301, 1000, 97)] {
            println!("--- m={m} k={k} n={n} ---");
            let (a, b) = gemm_pair(m, k, n, 0xfeed);
            let mut full = Array2::<f64>::zeros((m, n));
            general_mat_mul(1.0, &a.view(), &b.view(), 0.0, &mut full.view_mut());
            for split in [64usize, 32, 16, 8, 4, 2, 1] {
                let mut got = Array2::<f64>::zeros((m, n));
                let mut r0 = 0;
                while r0 < m {
                    // MERGE a would-be 1-row remainder into the previous band:
                    // if the tail after this band would be a single row, take
                    // it now. Tests the "only m=1 differs" hypothesis.
                    let mut r1 = (r0 + split).min(m);
                    if m - r1 == 1 && split > 1 {
                        r1 = m;
                    }
                    let la = a.slice(ndarray::s![r0..r1, ..]);
                    let mut ob = got.slice_mut(ndarray::s![r0..r1, ..]);
                    general_mat_mul(1.0, &la, &b.view(), 0.0, &mut ob);
                    r0 = r1;
                }
                let bad =
                    got.iter().zip(full.iter()).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
                let rows: Vec<usize> = (0..m)
                    .filter(|&i| (0..n).any(|j| got[[i, j]].to_bits() != full[[i, j]].to_bits()))
                    .collect();
                let maxdev =
                    got.iter().zip(full.iter()).map(|(g, w)| (g - w).abs()).fold(0.0f64, f64::max);
                let shown: Vec<usize> = rows.iter().copied().take(12).collect();
                println!(
                    "split={split}: {bad}/{} differ (maxdev {maxdev:.3e}); {} rows differ, first {shown:?}",
                    m * n,
                    rows.len()
                );
            }
        }
    }

    /// EXACTNESS ANCHOR for the row-banded GEMM.
    ///
    /// `gemm_row_banded` must reproduce the serial `gemm_kblocked` product
    /// BIT-FOR-BIT (`to_bits()` equality, not an epsilon bound). This is the
    /// whole justification for turning banding on by default rather than
    /// leaving it opt-in like the OpenBLAS m/n threading: banding cannot change
    /// any element's k-summation order, because each band is the same
    /// ascending-k block loop restricted to a disjoint row slab.
    ///
    /// If this ever fails, the banded path is NOT a free win and must go back
    /// behind an opt-in flag.
    #[test]
    fn row_banded_gemm_is_bit_identical_to_serial() {
        use ndarray::Array2;
        for (m, k, n, label) in banding_shape_cases() {
            let (a, b) = gemm_pair(m, k, n, 0xfeed_beef ^ (m as u64) << 20 ^ (k as u64));

            let mut want = Array2::<f64>::zeros((m, n));
            gemm_kblocked(&a.view(), &b.view(), &mut want.view_mut());

            let mut got = Array2::<f64>::zeros((m, n));
            gemm_row_banded(&a.view(), &b.view(), &mut got.view_mut());

            let bad = got
                .iter()
                .zip(want.iter())
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            assert_eq!(
                bad, 0,
                "{label} ({m}x{k}x{n}): row-banded GEMM must be BIT-identical to \
                 serial gemm_kblocked, but {bad}/{} elements differ",
                m * n,
            );
        }
    }

    /// The banded result must not depend on the worker count.
    ///
    /// Here the band WIDTH really does vary with the worker count (see
    /// `row_band_width`) — unusually for this codebase, that is deliberate and
    /// safe, because every even width is bit-identical
    /// (`every_even_band_width_is_bit_identical`). This test is what holds that
    /// license to account: it runs the exact bounds each worker count would
    /// produce and requires one identical bit pattern across all of them.
    ///
    /// It deliberately does NOT go through `pool.install(gemm_row_banded)`:
    /// that would take the `current_thread_index().is_some()` serial fallback
    /// and pass vacuously no matter how broken the banding was. Instead it
    /// drives `row_band_width`/`row_band_bounds` for each simulated worker
    /// count directly, which is the code path whose thread-dependence is at
    /// issue. (`row_banded_gemm_is_bit_identical_to_serial` covers the real
    /// parallel entry point, outside any pool.)
    #[test]
    fn row_banded_gemm_is_thread_count_independent() {
        use ndarray::Array2;
        for (m, k, n, label) in banding_shape_cases() {
            let (a, b) = gemm_pair(m, k, n, 0x1234_5678 ^ (n as u64) << 16 ^ (k as u64));

            // Reference: the unbanded serial product.
            let mut want = Array2::<f64>::zeros((m, n));
            gemm_kblocked(&a.view(), &b.view(), &mut want.view_mut());

            // For each worker count, run the bands that count would produce.
            let mut widths = Vec::new();
            for threads in [1usize, 2, 3, 4, 6, 8, 12, 16, 64] {
                let width = row_band_width(m, threads);
                widths.push(width);
                let mut got = Array2::<f64>::zeros((m, n));
                for &(r0, r1) in &row_band_bounds(m, width) {
                    let lb = a.slice(ndarray::s![r0..r1, ..]);
                    let mut ob = got.slice_mut(ndarray::s![r0..r1, ..]);
                    gemm_kblocked(&lb, &b.view(), &mut ob);
                }
                let bad = got
                    .iter()
                    .zip(want.iter())
                    .filter(|(g, w)| g.to_bits() != w.to_bits())
                    .count();
                assert_eq!(
                    bad, 0,
                    "{label} ({m}x{k}x{n}): the bands chosen for {threads} workers \
                     (width {width}) differ from the unbanded product in {bad}/{} \
                     elements — the band width may vary with thread count ONLY \
                     because every even width is bit-identical",
                    m * n,
                );
            }
            // Reachability: on at least some shape the worker count must
            // actually CHANGE the width, or this test proves nothing about
            // thread-dependence.
            widths.dedup();
            if m >= 4 * GEMM_ROW_BAND_MIN {
                assert!(
                    widths.len() > 1,
                    "{label} ({m}x{k}x{n}): row_band_width returned the same width for \
                     every worker count, so this test never exercised a varying split",
                );
            }

            // And the real entry point, outside any pool, still matches.
            let mut ambient = Array2::<f64>::zeros((m, n));
            gemm_row_banded(&a.view(), &b.view(), &mut ambient.view_mut());
            let bad = ambient
                .iter()
                .zip(want.iter())
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            assert_eq!(bad, 0, "{label} ({m}x{k}x{n}): gemm_row_banded differs in {bad} elements");
        }
    }

    /// End-to-end: the public `einsum_binary` entry point (which now routes
    /// through `gemm_row_banded`) is bit-identical at 1 / 4 / 12 rayon threads
    /// on a contraction large enough to actually band.
    #[test]
    fn einsum_binary_is_thread_count_independent() {
        // Pia,Pjb->iajb with P=600, i=a=j=b=10 -> m=n=100, k=600. 2*100*100*600
        // = 12 MFLOP, above PAR_GEMM_MIN_FLOPS.
        let (np, ni) = (600usize, 10usize);
        let mut s: u64 = 0xdead_c0de;
        let v: Vec<f64> = (0..np * ni * ni)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();
        let bt = Array::from_shape_vec(IxDyn(&[np, ni, ni]), v).unwrap();

        let run = || {
            einsum_binary(bt.view(), &[1, 2], &[0], bt.view(), &[1, 2], &[0], &[ni, ni, ni, ni])
                .unwrap()
        };
        let ambient = run();
        for threads in [1usize, 4, 12] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let got = pool.install(run);
            let bad = got
                .iter()
                .zip(ambient.iter())
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            assert_eq!(
                bad, 0,
                "einsum_binary at {threads} rayon threads differs from ambient in \
                 {bad}/{} elements",
                ambient.len(),
            );
        }
    }

    /// `permute_to_owned` fans the transpose gather out over rayon above
    /// `PAR_PERMUTE_MIN_ELEMS`. It must produce EXACTLY what the serial
    /// `as_standard_layout().into_owned()` produces — this is pure data
    /// movement (every output element written once, by one worker), so unlike
    /// a reduction it is bit-identical, not merely close.
    ///
    /// Shapes deliberately straddle the threshold and include a non-divisible
    /// leading extent (so the last rayon chunk is short) and a leading extent
    /// of 1 (which takes the serial short-circuit).
    /// `to_2d_or_transpose` lets `general_mat_mul` absorb a group-swap into
    /// dgemm's transa/transb flag instead of materializing a permuted copy. A
    /// wrong transpose here does not crash — it silently computes a DIFFERENT
    /// contraction — so this checks the engine against a brute-force
    /// index-by-index reference over real workspace specs.
    ///
    /// The specs are chosen to cover all three dispatch cases: an operand
    /// already contiguous (`Pia,Pjb->iajb` left), an operand reachable by pure
    /// transpose (`ijcd,abcd->ijab` right — the O(N^6) CCSD VVVV term), and a
    /// genuinely interleaved one that must still copy (`akic,kjbc->aijb`).
    /// Extents are deliberately distinct primes so a transposed-but-wrong
    /// result cannot accidentally match by symmetry.
    #[test]
    fn transpose_dispatch_matches_naive_reference() {
        // (spec, dims) with every index a distinct extent.
        let dims: std::collections::HashMap<char, usize> =
            [
                ('P', 7),
                ('i', 3),
                ('j', 4),
                ('a', 5),
                ('b', 2),
                ('c', 6),
                ('d', 3),
                ('k', 2),
                ('l', 4),
            ]
            .into_iter()
            .collect();
        let specs = ["Pia,Pjb->iajb", "ijcd,abcd->ijab", "akic,kjbc->aijb", "ijcd,klcd->ijkl"];

        for spec in specs {
            let (lhs, out) = spec.split_once("->").unwrap();
            let (ls, rs) = lhs.split_once(',').unwrap();
            let (ls, rs, out) = (ls.as_bytes(), rs.as_bytes(), out.as_bytes());
            let ext = |s: &[u8]| -> Vec<usize> { s.iter().map(|c| dims[&(*c as char)]).collect() };
            let (lsh, rsh, osh) = (ext(ls), ext(rs), ext(out));

            let mk = |shape: &[usize], seed: u64| {
                let n: usize = shape.iter().product();
                let mut s = seed;
                let v: Vec<f64> = (0..n)
                    .map(|_| {
                        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                    })
                    .collect();
                Array::from_shape_vec(IxDyn(shape), v).unwrap()
            };
            let l = mk(&lsh, 0xabcd);
            let r = mk(&rsh, 0x1234);

            // Axis roles, exactly as the macro computes them.
            let contr: Vec<u8> =
                ls.iter().copied().filter(|c| rs.contains(c) && !out.contains(c)).collect();
            let pos = |s: &[u8], want: &[u8]| -> Vec<usize> {
                want.iter().map(|c| s.iter().position(|x| x == c).unwrap()).collect()
            };
            let lfree: Vec<u8> = ls.iter().copied().filter(|c| out.contains(c)).collect();
            let rfree: Vec<u8> = rs.iter().copied().filter(|c| out.contains(c)).collect();

            let got = einsum_binary(
                l.view(),
                &pos(ls, &lfree),
                &pos(ls, &contr),
                r.view(),
                &pos(rs, &rfree),
                &pos(rs, &contr),
                &osh,
            )
            .unwrap();

            // Brute force: walk every output index and every contracted index.
            let mut want = ArrayD::<f64>::zeros(IxDyn(&osh));
            let letters: Vec<u8> = out.iter().chain(contr.iter()).copied().collect();
            let extents: Vec<usize> = letters.iter().map(|c| dims[&(*c as char)]).collect();
            let total: usize = extents.iter().product();
            for flat in 0..total {
                // Decode a mixed-radix index over (out..., contr...).
                let mut rem = flat;
                let mut idx = std::collections::HashMap::new();
                for (k, &e) in letters.iter().zip(extents.iter()).rev() {
                    idx.insert(*k, rem % e);
                    rem /= e;
                }
                let pick = |s: &[u8]| -> Vec<usize> { s.iter().map(|c| idx[c]).collect() };
                let contrib = l[IxDyn(&pick(ls))] * r[IxDyn(&pick(rs))];
                want[IxDyn(&pick(out))] += contrib;
            }

            let max_dev = got
                .iter()
                .zip(want.iter())
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_dev < 1e-12,
                "{spec}: transpose-dispatch result differs from the naive \
                 reference by {max_dev:.3e}",
            );
        }
    }

    #[test]
    fn parallel_permute_matches_serial_bitwise() {
        // (shape, permutation) pairs; several exceed PAR_PERMUTE_MIN_ELEMS.
        let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
            (vec![7, 5, 3], vec![2, 0, 1]),          // small -> serial path
            (vec![41, 13, 17, 3], vec![2, 3, 0, 1]), // large, non-divisible
            (vec![64, 32, 32], vec![1, 2, 0]),       // large, power-of-two
            (vec![1, 400, 400], vec![2, 1, 0]),      // leading extent 1
            (vec![400, 400], vec![1, 0]),            // plain 2D transpose
        ];
        for (shape, order) in cases {
            let n: usize = shape.iter().product();
            let a =
                Array::from_shape_vec(IxDyn(&shape), (0..n).map(|x| x as f64 * 0.5 - 3.0).collect())
                    .unwrap();
            let permuted = a.view().permuted_axes(order.clone());
            let want = permuted.as_standard_layout().into_owned();
            let got = permute_to_owned(permuted.view());
            assert_eq!(got.shape(), want.shape(), "shape mismatch for {shape:?} / {order:?}");
            assert!(
                got.iter().zip(want.iter()).all(|(g, w)| g.to_bits() == w.to_bits()),
                "parallel permute must be BIT-identical to serial for {shape:?} / {order:?}",
            );
        }
    }

    #[test]
    fn matmul_contracted_last_first_no_transpose() {
        let a = Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f64).collect()).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[3, 2]), (0..6).map(|x| x as f64).collect()).unwrap();
        let out = einsum_binary(a.view(), &[0], &[1], b.view(), &[1], &[0], &[2, 2]).unwrap();
        let want = naive_matmul(&a, &b);
        assert!((&out - &want).iter().all(|x| x.abs() < 1e-12));
    }

    #[test]
    fn contraction_needs_transpose() {
        let a = Array::from_shape_vec(IxDyn(&[3, 2]), (0..6).map(|x| x as f64).collect()).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f64).collect()).unwrap();
        let out = einsum_binary(a.view(), &[1], &[0], b.view(), &[0], &[1], &[2, 2]).unwrap();
        let mut want = Array::zeros(IxDyn(&[2, 2]));
        for i in 0..2 { for j in 0..2 { let mut s=0.0; for k in 0..3 { s += a[[k,i]]*b[[j,k]]; } want[[i,j]]=s; } }
        assert!((&out - &want).iter().all(|x| x.abs() < 1e-12));
    }

    #[test]
    fn scalar_output() {
        let a = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0,2.0,3.0,4.0]).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0,1.0,1.0,1.0]).unwrap();
        let out = einsum_binary(a.view(), &[], &[0,1], b.view(), &[], &[0,1], &[]).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out.iter().next().unwrap() - 10.0).abs() < 1e-12);
    }

    /// `permute_to_owned` must be bit-identical to ndarray's own serial
    /// `as_standard_layout().into_owned()`, at every thread count.
    ///
    /// This is the invariant the CC/MP2 antisymmetrizers rely on. They used to
    /// call ndarray's serial path directly; they now call this function, and
    /// that substitution is only safe because the two produce *the same bytes*.
    /// A permutation is pure data movement — every output element is written
    /// exactly once, by one worker, with a value copied from one input element
    /// — so unlike a reduction there is no summation order to perturb, and the
    /// result cannot depend on how the outer axis is split across workers.
    ///
    /// The test pins that claim rather than trusting it: if someone ever adds
    /// accumulation to the permute path, or splits on an axis other than the
    /// outermost, this fails loudly instead of silently perturbing CCSD
    /// amplitudes. Shapes straddle `PAR_PERMUTE_MIN_ELEMS` (64K) so both the
    /// serial-fallback and the parallel branch are exercised.
    #[test]
    fn permute_to_owned_is_bit_identical_across_thread_counts() {
        // (shape, axis permutation, label) — the last two clear 64K elements
        // and so take the rayon path; the first two fall back to serial.
        let cases: &[(&[usize], &[usize], &str)] = &[
            (&[8, 8, 8, 8], &[1, 0, 3, 2], "small 4-D, below threshold"),
            (&[6, 5, 7, 4], &[3, 1, 0, 2], "ragged 4-D, below threshold"),
            (&[16, 16, 16, 16], &[1, 0, 3, 2], "P(ij)P(ab)-shaped, above threshold"),
            (&[12, 14, 13, 15], &[2, 0, 3, 1], "ragged 4-D, above threshold"),
        ];

        for (shape, axes, label) in cases {
            let n: usize = shape.iter().product();
            // Deterministic LCG, with occasional huge magnitudes so that any
            // accidental arithmetic (rather than pure movement) would show up.
            let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
            let src = ArrayD::from_shape_fn(IxDyn(shape), |_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                if s.is_multiple_of(41) { v * 1e9 } else { v }
            });

            // Reference: ndarray's serial path — exactly what the call sites
            // used before this function existed.
            let want = src.view().permuted_axes(IxDyn(axes)).as_standard_layout().into_owned();

            for threads in [1usize, 2, 3, 4, 8, 13] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("thread pool builds");
                let got =
                    pool.install(|| permute_to_owned(src.view().permuted_axes(IxDyn(axes))));

                assert_eq!(
                    got.shape(),
                    want.shape(),
                    "{label}: shape changed at {threads} workers"
                );
                let bad = got
                    .iter()
                    .zip(want.iter())
                    .filter(|(g, w)| g.to_bits() != w.to_bits())
                    .count();
                assert_eq!(
                    bad, 0,
                    "{label} (n={n}): {bad}/{n} elements differ from the serial \
                     reference at {threads} workers — a permutation is pure data \
                     movement and MUST be bit-identical regardless of how the \
                     outermost axis is banded across workers"
                );
            }
        }
    }

    #[test]
    fn three_index_eri_build() {
        let p=2; let n=2;
        let bt = Array::from_shape_vec(IxDyn(&[p,n,n]), (0..p*n*n).map(|x| x as f64).collect()).unwrap();
        let out = einsum_binary(bt.view(), &[1,2], &[0], bt.view(), &[1,2], &[0], &[n,n,n,n]).unwrap();
        let mut want = Array::zeros(IxDyn(&[n,n,n,n]));
        for i in 0..n {for a in 0..n {for j in 0..n {for b in 0..n {
            let mut s=0.0; for pp in 0..p { s+= bt[[pp,i,a]]*bt[[pp,j,b]]; }
            want[[i,a,j,b]]=s;
        }}}}
        assert!((&out - &want).iter().all(|x| x.abs() < 1e-12));
    }
}
