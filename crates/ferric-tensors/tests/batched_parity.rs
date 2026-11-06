//! The batched einsum path (`einsum_binary_batched` with batch axes) against
//! the definition it must reproduce: a LOOP of single unbatched contractions
//! (`einsum_binary`), one per batch index, each scaled — plus an independent
//! naive index loop so the two GEMM paths cannot agree on a shared mistake.
//!
//! Every batch-axis spec reaches this path: the `einsum!` macro and
//! `einsum_dyn` (the Python `ferric.einsum` binding) both lower to it.

use ferric_tensors::{einsum, einsum_binary, einsum_binary_batched, einsum_dyn};
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use std::collections::BTreeMap;

/// Deterministic, bounded, non-symmetric fill (a transposed axis cannot pass).
fn filled(shape: &[usize], seed: f64) -> ArrayD<f64> {
    let mut k = 0.0;
    ArrayD::from_shape_fn(IxDyn(shape), |_| {
        k += 1.0;
        (seed * k).sin() + 0.25 * (0.37 * seed * k).cos()
    })
}

/// Remove the axes `drop` from position list `ps` (positions shift down).
fn shift(ps: &[usize], drop: &[usize]) -> Vec<usize> {
    ps.iter()
        .map(|&p| p - drop.iter().filter(|&&d| d < p).count())
        .collect()
}

/// Fix the batch axes of `op` (positions `batch`, values `idx`) -> a view.
fn fix_batch<'a>(op: ArrayViewD<'a, f64>, batch: &[usize], idx: &[usize]) -> ArrayViewD<'a, f64> {
    let mut pairs: Vec<(usize, usize)> = batch.iter().copied().zip(idx.iter().copied()).collect();
    // Highest position first so earlier positions stay valid.
    pairs.sort_by_key(|a| std::cmp::Reverse(a.0));
    let mut v = op;
    for (ax, i) in pairs {
        v = v.index_axis_move(Axis(ax), i);
    }
    v
}

/// THE definition: one unbatched `einsum_binary` per batch index, times scale.
fn loop_of_unbatched(
    spec: &str,
    a: ArrayViewD<f64>,
    b: ArrayViewD<f64>,
    scale: f64,
) -> ArrayD<f64> {
    let s = ferric_einsum_spec::parse(spec).unwrap();
    let bext: Vec<usize> = s.left_batch.iter().map(|&p| a.shape()[p]).collect();
    let lf: Vec<usize> = s.left_free.iter().map(|&p| a.shape()[p]).collect();
    let rf: Vec<usize> = s.right_free.iter().map(|&p| b.shape()[p]).collect();
    let slice_shape: Vec<usize> = lf.iter().chain(&rf).copied().collect();
    let out_shape: Vec<usize> = bext.iter().chain(&slice_shape).copied().collect();
    let mut out = ArrayD::<f64>::zeros(IxDyn(&out_shape));
    let nb: usize = bext.iter().product();
    let (lfree, lcontr) = (
        shift(&s.left_free, &s.left_batch),
        shift(&s.left_contr, &s.left_batch),
    );
    let (rfree, rcontr) = (
        shift(&s.right_free, &s.right_batch),
        shift(&s.right_contr, &s.right_batch),
    );
    for flat in 0..nb {
        let mut rem = flat;
        let mut idx = vec![0; bext.len()];
        for (d, &n) in bext.iter().enumerate().rev() {
            idx[d] = rem % n;
            rem /= n;
        }
        let la = fix_batch(a.clone(), &s.left_batch, &idx);
        let rb = fix_batch(b.clone(), &s.right_batch, &idx);
        let mut slab = if la.is_empty() || rb.is_empty() {
            ArrayD::zeros(IxDyn(&slice_shape))
        } else {
            einsum_binary(la, &lfree, &lcontr, rb, &rfree, &rcontr, &slice_shape).unwrap()
        };
        slab.mapv_inplace(|x| x * scale);
        let mut dst = out.view_mut();
        for &i in &idx {
            dst = dst.index_axis_move(Axis(0), i);
        }
        dst.assign(&slab);
    }
    out
}

/// Independent reference: loop over every assignment of every index letter.
fn naive<'a>(
    spec: &str,
    a: ArrayViewD<'a, f64>,
    b: ArrayViewD<'a, f64>,
    scale: f64,
) -> ArrayD<f64> {
    let (ins, out) = spec.split_once("->").unwrap();
    let (la, lb) = ins.split_once(',').unwrap();
    let mut ext = BTreeMap::new();
    for (term, arr) in [(la, &a), (lb, &b)] {
        for (c, &n) in term.chars().zip(arr.shape()) {
            ext.insert(c, n);
        }
    }
    let letters: Vec<char> = ext.keys().copied().collect();
    let out_shape: Vec<usize> = out.chars().map(|c| ext[&c]).collect();
    let mut res = ArrayD::<f64>::zeros(IxDyn(&out_shape));
    let total: usize = letters.iter().map(|c| ext[c]).product();
    for flat in 0..total {
        let mut rem = flat;
        let mut val = BTreeMap::new();
        for c in letters.iter().rev() {
            val.insert(*c, rem % ext[c]);
            rem /= ext[c];
        }
        let idx = |t: &str| -> Vec<usize> { t.chars().map(|c| val[&c]).collect() };
        res[IxDyn(&idx(out))] += scale * a[IxDyn(&idx(la))] * b[IxDyn(&idx(lb))];
    }
    res
}

/// max|x-y| <= tol * max(1, max|y|).
fn assert_close(x: &ArrayD<f64>, y: &ArrayD<f64>, tol: f64, what: &str) {
    assert_eq!(x.shape(), y.shape(), "{what}: shape");
    let scale = y.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    let dev = x
        .iter()
        .zip(y.iter())
        .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
    assert!(
        dev <= tol * scale,
        "{what}: max dev {dev:e} > {tol:e} * {scale}"
    );
}

/// Calls `einsum_binary_batched` exactly as the `einsum!` macro emits it
/// (NOT via `einsum_dyn`, which has its own guards).
fn macro_style(spec: &str, a: ArrayViewD<f64>, b: ArrayViewD<f64>, scale: f64) -> ArrayD<f64> {
    let s = ferric_einsum_spec::parse(spec).unwrap();
    let out_shape: Vec<usize> = s
        .out_from_batch
        .iter()
        .map(|&p| a.shape()[p])
        .chain(s.out_from_left.iter().map(|&p| a.shape()[p]))
        .chain(s.out_from_right.iter().map(|&p| b.shape()[p]))
        .collect();
    einsum_binary_batched(
        a,
        &s.left_batch,
        &s.left_free,
        &s.left_contr,
        b,
        &s.right_batch,
        &s.right_free,
        &s.right_contr,
        &out_shape,
        scale,
    )
    .unwrap()
}

/// (spec, left shape, right shape). Contracted extents > GEMM_K_BLOCK (128)
/// are included on purpose: the unbatched path k-blocks there.
fn cases() -> Vec<(&'static str, Vec<usize>, Vec<usize>)> {
    vec![
        // per-element dot over k (batch = ij), the only batched spec in-tree
        ("kij,kij->ij", vec![5, 3, 4], vec![5, 3, 4]),
        ("kij,kij->ij", vec![300, 4, 3], vec![300, 4, 3]),
        // Hadamard: every slice is a 1x1x1 GEMM
        ("ij,ij->ij", vec![7, 9], vec![7, 9]),
        // genuine batched GEMMs, k below and above the k-block
        ("bij,bjk->bik", vec![3, 5, 7], vec![3, 7, 6]),
        ("bij,bjk->bik", vec![4, 9, 300], vec![4, 300, 11]),
        ("bij,bkj->bik", vec![3, 6, 260], vec![3, 5, 260]),
        // batch axis not leading in the left operand; interleaved groups
        ("ibj,bjk->bik", vec![6, 3, 140], vec![3, 140, 5]),
        // two batch axes, given in different orders in the two operands
        ("abij,bajk->abik", vec![2, 3, 4, 5], vec![3, 2, 5, 6]),
        // batch + outer product (no contracted axis)
        ("bi,bj->bij", vec![4, 5], vec![4, 6]),
        // batch size 1
        ("bij,bjk->bik", vec![1, 8, 200], vec![1, 200, 9]),
        // degenerate GEMM dims (the nvir=1 / column-major dot gotcha)
        ("bij,bjk->bik", vec![5, 1, 200], vec![5, 200, 1]),
        ("bij,bjk->bik", vec![5, 4, 1], vec![5, 1, 3]),
        ("bij,bjk->bik", vec![5, 1, 1], vec![5, 1, 1]),
        // above PAR_GEMM_MIN_FLOPS (4 Mi) in total but not per slice: the
        // rayon-over-batch branch (needs >= 2 rayon workers)
        ("bij,bjk->bik", vec![64, 40, 150], vec![64, 150, 40]),
        // each slice above it: per-slice gemm_row_banded
        ("bij,bjk->bik", vec![2, 120, 200], vec![2, 200, 120]),
    ]
}

#[test]
fn batched_matches_loop_of_unbatched_and_naive() {
    for (spec, sa, sb) in cases() {
        let (a, b) = (filled(&sa, 0.7), filled(&sb, 1.3));
        let got = einsum_dyn(spec, a.view(), b.view(), -0.5).unwrap();
        let mac = macro_style(spec, a.view(), b.view(), -0.5);
        let lp = loop_of_unbatched(spec, a.view(), b.view(), -0.5);
        let what = format!("{spec} {sa:?} {sb:?}");
        assert_close(&got, &lp, 1e-13, &format!("{what} dyn vs loop"));
        assert_close(&mac, &lp, 1e-13, &format!("{what} batched vs loop"));
        // The naive reference is ~1 µs per term; keep it to the small cases.
        let (ins, _) = spec.split_once("->").unwrap();
        let (la, lb) = ins.split_once(',').unwrap();
        let mut ext = BTreeMap::new();
        for (c, &n) in la.chars().zip(&sa).chain(lb.chars().zip(&sb)) {
            ext.insert(c, n);
        }
        if ext.values().product::<usize>() <= 1 << 20 {
            let nv = naive(spec, a.view(), b.view(), -0.5);
            assert_close(&lp, &nv, 1e-13, &format!("{what} loop vs naive"));
        }
    }
}

/// Batch-slice products must round exactly like the unbatched engine: one
/// k-blocked GEMM per slice, THEN the scale. A batched path that calls a
/// full-k GEMM or folds `scale` into alpha is within 1e-13 but not
/// bit-identical, and this is what pins the shared helper.
///
/// Excluded: slices with NO free extent and k > 1 (`kij,kij->ij`). Those go
/// through the vectorized batched dot, whose k-order is its own (dgemm's
/// 1 x k x 1 order is not reproducible from Rust); they are held to 1e-13 by
/// the test above. The k == 1 Hadamard (`ij,ij->ij`) stays in: exact.
#[test]
fn batched_is_bit_identical_to_loop_of_unbatched() {
    let mut checked = 0;
    for (spec, sa, sb) in cases() {
        let s = ferric_einsum_spec::parse(spec).unwrap();
        let ext = |sh: &[usize], ax: &[usize]| ax.iter().map(|&p| sh[p]).product::<usize>();
        let dot = ext(&sa, &s.left_free) == 1 && ext(&sb, &s.right_free) == 1;
        if dot && ext(&sa, &s.left_contr) > 1 {
            continue;
        }
        checked += 1;
        let (a, b) = (filled(&sa, 0.7), filled(&sb, 1.3));
        for scale in [1.0, -0.5, 1.0 / 3.0] {
            let mac = macro_style(spec, a.view(), b.view(), scale);
            let lp = loop_of_unbatched(spec, a.view(), b.view(), scale);
            let bad = mac
                .iter()
                .zip(lp.iter())
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count();
            assert_eq!(
                bad, 0,
                "{spec} {sa:?} {sb:?} scale={scale}: {bad} elements differ"
            );
        }
    }
    assert!(
        checked >= 10,
        "the exclusion must not empty the test ({checked} cases)"
    );
}

#[test]
fn non_contiguous_and_transposed_inputs() {
    // Left: a reversed-axes view (column-major); right: a step-2 slice.
    let base_a = filled(&[150, 6, 4], 0.9); // (j, i, b) -> reversed = (b, i, j)
    let a = base_a.view().reversed_axes();
    let base_b = filled(&[8, 150, 10], 1.7);
    let b = base_b.slice(ndarray::s![..;2, .., ..;2]).into_dyn();
    assert!(!a.is_standard_layout() && !b.is_standard_layout());
    let spec = "bij,bjk->bik";
    let (ac, bc) = (a.to_owned(), b.to_owned());
    let got = macro_style(spec, a.clone(), b.clone(), 0.75);
    let lp = loop_of_unbatched(spec, ac.view(), bc.view(), 0.75);
    assert_close(&got, &lp, 1e-13, "strided batched vs contiguous loop");
    assert_close(
        &got,
        &naive(spec, ac.view(), bc.view(), 0.75),
        1e-13,
        "vs naive",
    );
}

/// A zero-extent axis must give zeros (or an empty output), on the MACRO
/// path too — not only through `einsum_dyn`'s own guard.
#[test]
fn empty_axes_on_the_macro_path() {
    let empties: &[(&str, &[usize], &[usize])] = &[
        ("bij,bjk->bik", &[0, 3, 4], &[0, 4, 5]), // empty batch
        ("bij,bjk->bik", &[2, 0, 4], &[2, 4, 5]), // empty left free
        ("bij,bjk->bik", &[2, 3, 0], &[2, 0, 5]), // empty contraction -> zeros
        ("kij,kij->ij", &[0, 3, 4], &[0, 3, 4]),  // empty contraction, batched dot
        ("ij,jk->ik", &[3, 0], &[0, 4]),          // unbatched empty contraction
        ("ij,jk->ik", &[0, 2], &[2, 4]),          // unbatched empty free
    ];
    for &(spec, sa, sb) in empties {
        let (a, b) = (filled(sa, 0.3), filled(sb, 0.4));
        let got = macro_style(spec, a.view(), b.view(), 2.0);
        let want = naive(spec, a.view(), b.view(), 2.0);
        assert_eq!(got.shape(), want.shape(), "{spec} {sa:?} {sb:?}");
        assert!(
            got.iter().all(|&x| x == 0.0),
            "{spec} {sa:?} {sb:?}: nonzero"
        );
    }
    // And the literal macro.
    let a = ArrayD::<f64>::zeros(IxDyn(&[0, 3, 4]));
    let c: ArrayD<f64> = einsum!("kij,kij->ij", &a, &a, 0.5);
    assert_eq!(c.shape(), &[3, 4]);
    assert!(c.iter().all(|&x| x == 0.0));
}

/// An extent mismatch on a shared axis must still be an error when an operand
/// is empty (the empty-axis shortcut must not swallow it).
#[test]
fn empty_operand_does_not_hide_an_extent_mismatch() {
    let a = filled(&[0, 3, 4], 0.3);
    let b = filled(&[0, 5, 6], 0.4); // contracted j: 4 vs 5
    let r = einsum_binary_batched(
        a.view(),
        &[0],
        &[1],
        &[2],
        b.view(),
        &[0],
        &[2],
        &[1],
        &[0, 3, 6],
        1.0,
    );
    assert!(
        r.is_err(),
        "mismatched contracted extent with an empty batch must error"
    );
    let a = filled(&[2, 3, 4], 0.3);
    let b = filled(&[0, 4, 6], 0.4); // batch: 2 vs 0
    let r = einsum_binary_batched(
        a.view(),
        &[0],
        &[1],
        &[2],
        b.view(),
        &[0],
        &[2],
        &[1],
        &[2, 3, 6],
        1.0,
    );
    assert!(r.is_err(), "mismatched batch extent must error");
}

/// Timing: batched path vs a loop of unbatched contractions vs a hand-written
/// baseline. `--ignored --nocapture`, release, OPENBLAS_NUM_THREADS=1.
#[test]
#[ignore]
fn bench_batched_vs_unbatched() {
    use ndarray::linalg::general_mat_mul;
    use std::time::Instant;
    const REPS: usize = 5;
    fn stats(mut t: Vec<f64>) -> (f64, f64, f64) {
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (t[t.len() / 2], t[0], t[t.len() - 1])
    }
    fn time<F: FnMut() -> ArrayD<f64>>(mut f: F) -> ((f64, f64, f64), ArrayD<f64>) {
        let mut out = f(); // warm-up
        let mut ts = Vec::new();
        for _ in 0..REPS {
            let t0 = Instant::now();
            out = f();
            ts.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        (stats(ts), out)
    }
    type Hand = fn(&ArrayD<f64>, &ArrayD<f64>, f64) -> ArrayD<f64>;
    fn hadamard(a: &ArrayD<f64>, b: &ArrayD<f64>, s: f64) -> ArrayD<f64> {
        let mut o = a * b;
        o.mapv_inplace(|x| x * s);
        o
    }
    fn kdot(a: &ArrayD<f64>, b: &ArrayD<f64>, s: f64) -> ArrayD<f64> {
        let mut o = ArrayD::<f64>::zeros(IxDyn(&a.shape()[1..]));
        for k in 0..a.shape()[0] {
            ndarray::Zip::from(&mut o)
                .and(a.index_axis(Axis(0), k))
                .and(b.index_axis(Axis(0), k))
                .for_each(|o, &x, &y| *o += x * y);
        }
        o.mapv_inplace(|x| x * s);
        o
    }
    fn bgemm(a: &ArrayD<f64>, b: &ArrayD<f64>, s: f64) -> ArrayD<f64> {
        let (nb, m, n) = (a.shape()[0], a.shape()[1], b.shape()[2]);
        let a3 = a.view().into_dimensionality::<ndarray::Ix3>().unwrap();
        let b3 = b.view().into_dimensionality::<ndarray::Ix3>().unwrap();
        let mut o = ndarray::Array3::<f64>::zeros((nb, m, n));
        for i in 0..nb {
            let mut oi = o.index_axis_mut(Axis(0), i);
            general_mat_mul(
                s,
                &a3.index_axis(Axis(0), i),
                &b3.index_axis(Axis(0), i),
                0.0,
                &mut oi,
            );
        }
        o.into_dyn()
    }
    let cases: Vec<(&str, &str, Vec<usize>, Vec<usize>, Hand)> = vec![
        (
            "CC-denominator Hadamard no=10 nv=80",
            "ijab,ijab->ijab",
            vec![10, 10, 80, 80],
            vec![10, 10, 80, 80],
            hadamard,
        ),
        (
            "DF diagonal naux=400 n=100",
            "kij,kij->ij",
            vec![400, 100, 100],
            vec![400, 100, 100],
            kdot,
        ),
        (
            "per-occ (nv,naux)(naux,nv) 10x80x400x80",
            "bij,bjk->bik",
            vec![10, 80, 400],
            vec![10, 400, 80],
            bgemm,
        ),
        (
            "many small: 200 x 20^3",
            "bij,bjk->bik",
            vec![200, 20, 20],
            vec![200, 20, 20],
            bgemm,
        ),
        (
            "per-occ large 10x200x800x200",
            "bij,bjk->bik",
            vec![10, 200, 800],
            vec![10, 800, 200],
            bgemm,
        ),
    ];
    println!(
        "RAYON_NUM_THREADS={:?} OPENBLAS_NUM_THREADS={:?} rayon workers={}  ms: median [min,max] over {REPS}",
        std::env::var("RAYON_NUM_THREADS").ok(),
        std::env::var("OPENBLAS_NUM_THREADS").ok(),
        rayon::current_num_threads()
    );
    for (label, spec, sa, sb, hand) in cases {
        let (a, b) = (filled(&sa, 0.7), filled(&sb, 1.3));
        let (tb, ob) = time(|| einsum_dyn(spec, a.view(), b.view(), -0.5).unwrap());
        let (tl, ol) = time(|| loop_of_unbatched(spec, a.view(), b.view(), -0.5));
        let (th, oh) = time(|| hand(&a, &b, -0.5));
        assert_close(&ob, &ol, 1e-13, label);
        assert_close(&ob, &oh, 1e-13, label);
        println!(
            "{label:42} batched {:9.3} [{:.3},{:.3}]  loop {:9.3} [{:.3},{:.3}]  hand {:9.3} [{:.3},{:.3}]",
            tb.0, tb.1, tb.2, tl.0, tl.1, tl.2, th.0, th.1, th.2
        );
    }
}
