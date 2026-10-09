//! Microbenchmark for the RS-GDF exchange row blocking (`rsgdf.rs`,
//! `exchange_row_blocked`): `K = Σ_a B_a D B_aᵀ` on a synthetic `(naux, n²)`
//! B at a given size, for several fixed row-block sizes, plus a variant that
//! forms every `B_a D` as one full-height GEMM (parallel over aux rows) and
//! row-blocks only the second product. Prints wall time per build and the
//! number of elements whose bits differ from the frozen serial loop.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=6 \
//!   cargo run --release -p ferric-pbc --example exchange_block_bench -- 168 672
//! ```

// Links the BLAS backend the production code uses (`cblas_dgemm`).
extern crate ferric_pbc;

use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;
use std::time::Instant;

const AUX_CHUNK: usize = 32;
const MIN_TAIL: usize = 4;

fn synth(i: usize, seed: f64) -> f64 {
    ((i as f64 + 1.0) * 0.618_033_988_75 * seed).sin()
}

fn row_blocks(n: usize, bs: usize) -> Vec<std::ops::Range<usize>> {
    let mut r: Vec<_> = (0..n).step_by(bs).map(|r0| r0..(r0 + bs).min(n)).collect();
    if r.len() >= 2 && r[r.len() - 1].len() < MIN_TAIL {
        let tail = r.pop().unwrap();
        r.last_mut().unwrap().end = tail.end;
    }
    r
}

fn serial(b: &Array2<f64>, d: &Array2<f64>, n: usize) -> Array2<f64> {
    let mut k = Array2::<f64>::zeros((n, n));
    for row in b.rows() {
        let ba = row.into_shape_with_order((n, n)).unwrap();
        let tmp = ba.dot(d);
        general_mat_mul(1.0, &tmp, &ba.t(), 1.0, &mut k);
    }
    k
}

fn split_rows<'a>(
    mut v: ndarray::ArrayViewMut2<'a, f64>,
    ranges: &[std::ops::Range<usize>],
) -> Vec<ndarray::ArrayViewMut2<'a, f64>> {
    let mut out = Vec::new();
    for r in ranges.iter().rev() {
        let (head, tail) = v.split_at(ndarray::Axis(0), r.start);
        out.push(tail);
        v = head;
    }
    out.reverse();
    out
}

/// The production scheme (`exchange_row_blocked`) at block size `bs`.
fn blocked(b: &Array2<f64>, d: &Array2<f64>, n: usize, bs: usize) -> Array2<f64> {
    let ranges = row_blocks(n, bs);
    let mut k = Array2::<f64>::zeros((n, n));
    let mut parts = split_rows(k.view_mut(), &ranges);
    for c0 in (0..b.nrows()).step_by(AUX_CHUNK) {
        let c1 = (c0 + AUX_CHUNK).min(b.nrows());
        let bks: Vec<ArrayView2<f64>> = (c0..c1)
            .map(|a| b.row(a).into_shape_with_order((n, n)).unwrap())
            .collect();
        parts
            .par_iter_mut()
            .zip(ranges.par_iter())
            .for_each(|(kb, r)| {
                for bk in &bks {
                    let tmp = bk.slice(s![r.clone(), ..]).dot(d);
                    general_mat_mul(1.0, &tmp, &bk.t(), 1.0, kb);
                }
            });
    }
    k
}

/// `B_a D` as full-height GEMMs (the serial call, parallel over aux rows of
/// a round), then the second product row-blocked at `bs`.
fn full_first(b: &Array2<f64>, d: &Array2<f64>, n: usize, bs: usize) -> Array2<f64> {
    let ranges = row_blocks(n, bs);
    let mut k = Array2::<f64>::zeros((n, n));
    let mut parts = split_rows(k.view_mut(), &ranges);
    for c0 in (0..b.nrows()).step_by(AUX_CHUNK) {
        let c1 = (c0 + AUX_CHUNK).min(b.nrows());
        let bks: Vec<ArrayView2<f64>> = (c0..c1)
            .map(|a| b.row(a).into_shape_with_order((n, n)).unwrap())
            .collect();
        let tmps: Vec<Array2<f64>> = bks.par_iter().map(|bk| bk.dot(d)).collect();
        parts
            .par_iter_mut()
            .zip(ranges.par_iter())
            .for_each(|(kb, r)| {
                for (bk, tmp) in bks.iter().zip(&tmps) {
                    general_mat_mul(1.0, &tmp.slice(s![r.clone(), ..]), &bk.t(), 1.0, kb);
                }
            });
    }
    k
}

/// (a) `B_a D` full height per aux row (parallel over the round's aux
/// rows), then the second product over fixed 2-D output tiles `rb x cb`.
fn tiles2d(b: &Array2<f64>, d: &Array2<f64>, n: usize, rb: usize, cb: usize) -> Array2<f64> {
    let rows = row_blocks(n, rb);
    let cols = row_blocks(n, cb);
    let tiles: Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> = rows
        .iter()
        .flat_map(|r| cols.iter().map(move |c| (r.clone(), c.clone())))
        .collect();
    let mut acc: Vec<Array2<f64>> = tiles
        .iter()
        .map(|(r, c)| Array2::zeros((r.len(), c.len())))
        .collect();
    for c0 in (0..b.nrows()).step_by(AUX_CHUNK) {
        let c1 = (c0 + AUX_CHUNK).min(b.nrows());
        let bks: Vec<ArrayView2<f64>> = (c0..c1)
            .map(|a| b.row(a).into_shape_with_order((n, n)).unwrap())
            .collect();
        let tmps: Vec<Array2<f64>> = bks.par_iter().map(|bk| bk.dot(d)).collect();
        acc.par_iter_mut()
            .zip(tiles.par_iter())
            .for_each(|(kt, (r, c))| {
                for (bk, tmp) in bks.iter().zip(&tmps) {
                    general_mat_mul(
                        1.0,
                        &tmp.slice(s![r.clone(), ..]),
                        &bk.t().slice(s![.., c.clone()]),
                        1.0,
                        kt,
                    );
                }
            });
    }
    let mut k = Array2::<f64>::zeros((n, n));
    for ((r, c), kt) in tiles.iter().zip(&acc) {
        k.slice_mut(s![r.clone(), c.clone()]).assign(kt);
    }
    k
}

/// (b) aux rows in `ng` fixed contiguous groups; each group's partial K
/// with full-size serial GEMMs; partials summed in group order.
fn aux_groups(b: &Array2<f64>, d: &Array2<f64>, n: usize, ng: usize) -> Array2<f64> {
    let naux = b.nrows();
    let per = naux.div_ceil(ng);
    let parts: Vec<Array2<f64>> = (0..ng)
        .into_par_iter()
        .map(|g| {
            let mut k = Array2::<f64>::zeros((n, n));
            for a in g * per..((g + 1) * per).min(naux) {
                let ba = b.row(a).into_shape_with_order((n, n)).unwrap();
                let tmp = ba.dot(d);
                general_mat_mul(1.0, &tmp, &ba.t(), 1.0, &mut k);
            }
            k
        })
        .collect();
    let mut k = Array2::<f64>::zeros((n, n));
    for p in &parts {
        k += p;
    }
    k
}

fn diffs(a: &Array2<f64>, b: &Array2<f64>) -> (usize, f64) {
    let scale = b.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let mut cnt = 0;
    let mut err = 0.0_f64;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            cnt += 1;
            err = err.max((x - y).abs());
        }
    }
    (cnt, err / scale)
}

fn time<F: FnMut() -> Array2<f64>>(reps: usize, mut f: F) -> (f64, Array2<f64>) {
    let mut out = f();
    let t0 = Instant::now();
    for _ in 0..reps {
        out = f();
    }
    (t0.elapsed().as_secs_f64() / reps as f64, out)
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (n, naux) = (
        args.first().copied().unwrap_or(168),
        args.get(1).copied().unwrap_or(672),
    );
    let reps = args.get(2).copied().unwrap_or(3);
    let b = Array2::from_shape_fn((naux, n * n), |(a, i)| synth(a * n * n + i, 0.37));
    let d = Array2::from_shape_fn((n, n), |(i, j)| synth(i.min(j) * n + i.max(j), 1.7));
    println!(
        "n {n} naux {naux} rayon threads {} reps {reps}",
        rayon::current_num_threads()
    );
    let (ts, ks) = time(reps, || serial(&b, &d, n));
    println!("serial                      {:8.4} s/build", ts);
    for (rb, cb) in [
        (56usize, 84usize),
        (84, 56),
        (42, 84),
        (56, 56),
        (84, 42),
        (42, 56),
    ] {
        let nt = row_blocks(n, rb).len() * row_blocks(n, cb).len();
        let (t, k) = time(reps, || tiles2d(&b, &d, n, rb, cb));
        let (c, e) = diffs(&k, &ks);
        println!("tiles2d {rb:3}x{cb:3} ({nt:2} t) {t:8.4} s/build  x{:5.2}  bit-diffs {c:6} max rel {e:.1e}", ts / t);
    }
    for ng in [6usize, 12, 24, 48] {
        let (t, k) = time(reps, || aux_groups(&b, &d, n, ng));
        let (c, e) = diffs(&k, &ks);
        println!("aux-groups ng {ng:3}          {t:8.4} s/build  x{:5.2}  bit-diffs {c:6} max rel {e:.1e}", ts / t);
    }
    if std::env::var("ROWS").is_err() {
        return;
    }
    for bs in [16, 24, 32, 48, 56, 64, 84, 96, 168] {
        let nb = row_blocks(n, bs).len();
        let (t, k) = time(reps, || blocked(&b, &d, n, bs));
        let (c, e) = diffs(&k, &ks);
        println!("blocked    bs {bs:3} ({nb:2} blk) {t:8.4} s/build  x{:5.2}  bit-diffs {c:6} max rel {e:.1e}", ts / t);
        let (t, k) = time(reps, || full_first(&b, &d, n, bs));
        let (c, e) = diffs(&k, &ks);
        println!("full-first bs {bs:3} ({nb:2} blk) {t:8.4} s/build  x{:5.2}  bit-diffs {c:6} max rel {e:.1e}", ts / t);
    }
}
