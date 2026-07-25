//! Probe the documented stack-overflow hazard in `lanczos.rs`: does a
//! multi-threaded OpenBLAS `eigh` on a naux-wide matrix overflow a worker stack?
//!
//! `lanczos_blas_threads()` pins BLAS to 1 thread and cites two reasons —
//! (1) stack overflow on a large `eigh`/QR, observed as aug-cc-pV{D,T}Z
//! PDEP-RPA tests aborting, and (2) reproducibility. Reason (1) is a crash and
//! is worth fixing if it is fixable; reason (2) is a preference. But the full
//! ferric-rpa suite now PASSES with `FERRIC_LANCZOS_BLAS_THREADS=12`, so this
//! isolates the mechanism directly at large naux.
//!
//! Runs the same dense `eigh` the solver does, at increasing sizes, on:
//!   - the main thread (large stack, ~8 MB)
//!   - a std::thread with an explicitly SMALL stack (mimics a rayon worker,
//!     whose default is 2 MB)
//!   - a rayon worker itself
//!
//! If threaded OpenBLAS allocates its workspace on the *caller's* stack, the
//! small-stack cases abort; if it heap-allocates (or uses its own pool's
//! stacks), they survive and the hazard is not what the comment describes.
use ndarray::Array2;
use ndarray_linalg::Eigh;

extern crate openblas_src as _;

extern "C" {
    fn openblas_set_num_threads(n: i32);
}

fn sym(n: usize) -> Array2<f64> {
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
        a[[i, i]] += n as f64; // keep it well-conditioned
    }
    a
}

fn run_eigh(n: usize, threads: i32, label: &str) {
    unsafe { openblas_set_num_threads(threads) };
    let a = sym(n);
    let (w, _v) = a.eigh(ndarray_linalg::UPLO::Lower).expect("eigh failed");
    unsafe { openblas_set_num_threads(1) };
    println!("  [{label}] n={n} threads={threads} OK (lambda_0={:.6})", w[0]);
}

fn main() {
    let threads: i32 =
        std::env::var("PROBE_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    // benzene/aug-cc-pVTZ has naux=763; go past it.
    let sizes: Vec<usize> = std::env::var("PROBE_SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![400, 763, 1200]);

    println!("main thread (default ~8 MB stack):");
    for &n in &sizes {
        run_eigh(n, threads, "main");
    }

    // A rayon worker's default stack is 2 MB. Mimic that explicitly.
    println!("\nstd::thread with a 2 MB stack (mimics a rayon worker):");
    for &n in &sizes {
        let h = std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(move || run_eigh(n, threads, "2MB-stack"))
            .expect("spawn");
        h.join().expect("2 MB stack thread aborted — THIS is the documented hazard");
    }

    // And inside an actual rayon worker. NOTE: the production resolver forces
    // 1 thread here via its `rayon::current_thread_index().is_some()` guard —
    // this deliberately bypasses that to test the underlying mechanism.
    println!("\ninside a rayon worker (guard bypassed on purpose):");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    pool.install(|| {
        use rayon::prelude::*;
        sizes.par_iter().for_each(|&n| run_eigh(n, threads, "rayon-worker"));
    });

    println!("\nAll probes completed without aborting.");
}
