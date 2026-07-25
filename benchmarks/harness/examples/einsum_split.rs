//! Where does `einsum!` time actually go — the permutation COPY or the GEMM?
//!
//! 83% of the workspace's ~130 einsum specs need at least one operand permuted
//! before it can be reshaped to 2D for GEMM, and `to_2d` does that with a
//! serial `as_standard_layout().into_owned()`. Only 2 specs have batch axes, so
//! the batched loop is NOT the hot path (contrary to the 2026-07-25 handoff's
//! next-step #2).
//!
//! This times a representative CCD contraction end-to-end via the public
//! `einsum_binary`, then times the same permutation copies alone, so the split
//! is unambiguous.
use std::time::Instant;

use ferric_tensors::einsum::einsum_binary;
use ndarray::{Array, ArrayD, IxDyn};

extern crate openblas_src as _;

fn rand_tensor(shape: &[usize], seed: u64) -> ArrayD<f64> {
    let n: usize = shape.iter().product();
    let mut s = seed;
    let v: Vec<f64> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect();
    Array::from_shape_vec(IxDyn(shape), v).unwrap()
}

/// The OLD serial permute+copy (`as_standard_layout().into_owned()`), kept as
/// the A-side reference. `to_2d` no longer does this — it goes through
/// `permute_to_owned`, which fans the gather out over rayon — so this measures
/// what the copy used to cost, not what it costs now.
fn permute_copy_serial(op: &ArrayD<f64>, order: Vec<usize>) -> usize {
    let permuted = op.view().permuted_axes(order);
    let owned = permuted.as_standard_layout().into_owned();
    owned.len()
}

/// The CURRENT parallel gather, mirroring `einsum.rs::permute_to_owned`.
fn permute_copy_parallel(op: &ArrayD<f64>, order: Vec<usize>) -> usize {
    use rayon::prelude::*;
    let permuted = op.view().permuted_axes(order);
    let shape = permuted.shape().to_vec();
    let slab: usize = shape[1..].iter().product::<usize>().max(1);
    let mut out = ArrayD::<f64>::zeros(IxDyn(&shape));
    let buf = out.as_slice_mut().unwrap();
    buf.par_chunks_mut(slab).enumerate().for_each(|(k, chunk)| {
        let src = permuted.index_axis(ndarray::Axis(0), k);
        let mut dst = ndarray::ArrayViewMutD::from_shape(IxDyn(&shape[1..]), chunk).unwrap();
        dst.assign(&src);
    });
    out.len()
}

fn main() {
    let no: usize = std::env::var("NO").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let nv: usize = std::env::var("NV").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    println!("spin-orbital CCD-ish shapes: no={no} nv={nv}");

    // "ijcd,abcd->ijab": right operand needs a permutation copy (abcd -> cdab).
    let t2 = rand_tensor(&[no, no, nv, nv], 0x1234);
    let vvvv = rand_tensor(&[nv, nv, nv, nv], 0x5678);
    println!(
        "t2 {:?} ({:.2} MB), vvvv {:?} ({:.2} MB)",
        t2.shape(),
        (t2.len() * 8) as f64 / 1e6,
        vvvv.shape(),
        (vvvv.len() * 8) as f64 / 1e6
    );

    let reps = 3;
    let mut t_full = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        // ijcd,abcd->ijab : left free [0,1] contr [2,3]; right free [0,1] contr [2,3]
        let out =
            einsum_binary(t2.view(), &[0, 1], &[2, 3], vvvv.view(), &[0, 1], &[2, 3], &[no, no, nv, nv])
                .unwrap();
        t_full = t_full.min(t.elapsed().as_secs_f64());
        std::hint::black_box(out.len());
    }

    // The right operand's to_2d call is `to_2d(right, right_contr, right_free)`
    // = permute abcd -> cdab, i.e. order [2,3,0,1]. The left is identity (and
    // `permute_to_owned` short-circuits identity to a plain to_owned()).
    let mut t_ser = f64::MAX;
    let mut t_par = f64::MAX;
    for _ in 0..reps {
        // Interleaved A/B — sequential runs drift with box/cache state.
        let t = Instant::now();
        std::hint::black_box(permute_copy_serial(&vvvv, vec![2, 3, 0, 1]));
        t_ser = t_ser.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        std::hint::black_box(permute_copy_parallel(&vvvv, vec![2, 3, 0, 1]));
        t_par = t_par.min(t.elapsed().as_secs_f64());
    }

    let gf = 2.0 * (no * no) as f64 * (nv * nv) as f64 * (nv * nv) as f64 / 1e9;
    let gemm = (t_full - t_par).max(0.0);
    println!("\nfull einsum_binary          {t_full:.3} s   (uses the PARALLEL copy)");
    println!("  permute copy, serial (old) {t_ser:.3} s");
    println!(
        "  permute copy, parallel     {t_par:.3} s   ({:.2}x, now {:.0}% of the contraction)",
        t_ser / t_par,
        100.0 * t_par / t_full
    );
    println!(
        "  => GEMM (by difference)    {gemm:.3} s   {:.1} GFLOP -> {:.1} GFLOP/s",
        gf,
        gf / gemm.max(1e-9)
    );
    println!(
        "\nprojected full-einsum time with the OLD serial copy: {:.3} s  =>  speedup {:.2}x",
        gemm + t_ser,
        (gemm + t_ser) / t_full
    );
}
