//! Is threaded OpenBLAS's GEMM *less accurate* than serial, or just different?
//!
//! Raising `FERRIC_BLAS_THREADS` makes CCSD ~1.6x faster but produces four
//! distinct energy bit patterns. The reflex is "threading breaks determinism,
//! so don't thread". But determinism only pins WHICHEVER order you have,
//! including a bad one — the DF dressing work (eb895df) found that k-blocking
//! is measurably MORE accurate than one full-k GEMM, because blocking is
//! coarse pairwise summation. So the real question is not "does the answer
//! move?" but "which order is closest to the truth?"
//!
//! This measures each ordering against a high-precision reference:
//!   - reference: exact-ish dot products accumulated in 128-bit (two-sum /
//!     Kahan-Babuska-Neumaier), which is ~machine-exact for these sizes
//!   - serial dgemm (1 thread)
//!   - threaded dgemm (N threads, k-axis split across threads)
//!   - explicit k-blocked serial dgemm (pairwise over k-chunks)
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-benchmarks \
//!     --example gemm_accuracy_vs_threads
use ndarray::{Array2, Axis};

extern crate openblas_src as _;

extern "C" {
    fn openblas_set_num_threads(n: i32);
}

/// Neumaier-compensated dot product: the accuracy reference.
fn dot_compensated(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64; // running compensation
    for (x, y) in a.iter().zip(b.iter()) {
        let p = x * y;
        let t = sum + p;
        if sum.abs() >= p.abs() {
            c += (sum - t) + p;
        } else {
            c += (p - t) + sum;
        }
        sum = t;
    }
    sum + c
}

fn reference(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    let bt = b.t().as_standard_layout().into_owned(); // (n, k) rows contiguous
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        let arow: Vec<f64> = (0..k).map(|p| a[[i, p]]).collect();
        for j in 0..n {
            let brow: Vec<f64> = (0..k).map(|p| bt[[j, p]]).collect();
            out[[i, j]] = dot_compensated(&arow, &brow);
        }
    }
    out
}

fn gemm(a: &Array2<f64>, b: &Array2<f64>, threads: i32) -> Array2<f64> {
    unsafe { openblas_set_num_threads(threads) };
    let out = a.dot(b);
    unsafe { openblas_set_num_threads(1) };
    out
}

/// Serial GEMM with the k-axis explicitly blocked: sum of per-block products,
/// i.e. coarse pairwise summation over k. This is the ordering eb895df found
/// to be MORE accurate than a single full-k GEMM.
fn gemm_kblocked(a: &Array2<f64>, b: &Array2<f64>, kblk: usize) -> Array2<f64> {
    let k = a.ncols();
    let mut out = Array2::<f64>::zeros((a.nrows(), b.ncols()));
    let mut k0 = 0;
    while k0 < k {
        let k1 = (k0 + kblk).min(k);
        let asub = a.slice(ndarray::s![.., k0..k1]);
        let bsub = b.slice(ndarray::s![k0..k1, ..]);
        out += &asub.dot(&bsub);
        k0 = k1;
    }
    out
}

fn err(got: &Array2<f64>, want: &Array2<f64>) -> (f64, f64) {
    let mut max_abs = 0.0f64;
    let mut denom = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        max_abs = max_abs.max((g - w).abs());
        denom = denom.max(w.abs());
    }
    (max_abs, max_abs / denom.max(1e-300))
}

fn main() {
    // Shapes chosen so k is LARGE — accumulation error grows with k, and k is
    // exactly the axis threaded OpenBLAS splits.
    let cases = [(200usize, 4000usize, 200usize), (400, 2000, 400), (100, 8000, 100)];

    for (m, k, n) in cases {
        let mut s: u64 = 0x243f_6a88_85a3_08d3;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let a = Array2::from_shape_fn((m, k), |_| rnd());
        let b = Array2::from_shape_fn((k, n), |_| rnd());

        println!("\n=== m={m} k={k} n={n} (k is the threaded/blocked axis) ===");
        let want = reference(&a, &b);

        let mut rows: Vec<(String, Array2<f64>)> = Vec::new();
        rows.push(("serial dgemm (1 thread)".into(), gemm(&a, &b, 1)));
        for t in [2, 4, 6, 8, 12] {
            rows.push((format!("threaded dgemm ({t} threads)"), gemm(&a, &b, t)));
        }
        for blk in [128usize, 256, 512] {
            rows.push((format!("serial k-blocked (blk={blk})"), gemm_kblocked(&a, &b, blk)));
        }
        // Does k-blocking make the result thread-INDEPENDENT? Only if OpenBLAS
        // never splits inside a block.
        for t in [4i32, 12] {
            unsafe { openblas_set_num_threads(t) };
            let g = gemm_kblocked(&a, &b, 128);
            unsafe { openblas_set_num_threads(1) };
            rows.push((format!("k-blocked(128) @ {t} threads"), g));
        }

        let base = rows[0].1.clone();
        println!("{:<32} {:>12} {:>12} {:>10}", "ordering", "max|err|", "rel err", "vs serial");
        for (name, got) in &rows {
            let (abs, rel) = err(got, &want);
            let bitsame = got.iter().zip(base.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
            println!(
                "{:<32} {:>12.3e} {:>12.3e} {:>10}",
                name,
                abs,
                rel,
                if bitsame { "identical" } else { "differs" }
            );
        }
        // The decisive check: is k-blocking thread-INDEPENDENT? Compare the
        // k-blocked results at different thread counts against EACH OTHER
        // (the column above compares against serial full-k, which cannot
        // answer this).
        unsafe { openblas_set_num_threads(1) };
        let kb1 = gemm_kblocked(&a, &b, 128);
        unsafe { openblas_set_num_threads(4) };
        let kb4 = gemm_kblocked(&a, &b, 128);
        unsafe { openblas_set_num_threads(12) };
        let kb12 = gemm_kblocked(&a, &b, 128);
        unsafe { openblas_set_num_threads(1) };
        let same4 = kb1.iter().zip(kb4.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
        let same12 = kb1.iter().zip(kb12.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
        println!(
            "  k-blocked(128) bit-identical across threads?  1-vs-4: {}   1-vs-12: {}",
            if same4 { "YES" } else { "NO" },
            if same12 { "YES" } else { "NO" }
        );
        let _ = want.index_axis(Axis(0), 0);
    }
}
