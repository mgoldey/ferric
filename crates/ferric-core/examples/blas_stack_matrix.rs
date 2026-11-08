//! Crash-matrix reproducer for threaded OpenBLAS LAPACK off the main thread.
//!
//! Usage: `blas_stack_matrix <op> <n> <blas_threads> <ctx> [stack_mib] [concurrency]`
//!   op          : inv | getrf | eigh | cholesky | gemm
//!   ctx         : main | spawn | rayon | toggle (spawn + a thread flipping the count)
//!   stack_mib   : stack size for spawn/rayon threads (0 = the Rust default, 2 MiB)
//!   concurrency : number of concurrent callers (spawn/rayon only; default 1)
//!
//! Each combination runs in its own process, because a stack overflow aborts
//! the whole process. The reference result for every caller is computed first
//! on the main thread at 1 BLAS thread; then the BLAS count is raised and the
//! op is re-run in the requested context. The line `OK ... maxrel=<x>` reports
//! the largest relative deviation from that reference over all callers, so
//! silent corruption (not just a crash) is visible.
//!
//! `FERRIC_PRINT_BLAS_CONFIG=1` prints `openblas_get_config()` and the thread
//! counts OpenBLAS reports.
//!
//! This is the reproducer behind wiki/notes/blas-threads-2026-11-08; it is kept
//! so the crash matrix can be re-measured against a different OpenBLAS.

use ndarray::Array2;
use ndarray_linalg::{Cholesky, Determinant, Eigh, Inverse, UPLO};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

extern "C" {
    fn openblas_get_config() -> *const c_char;
    fn openblas_get_num_threads() -> c_int;
    fn openblas_get_num_procs() -> c_int;
    fn openblas_set_num_threads(n: c_int);
    fn openblas_get_parallel() -> c_int;
}

/// Deterministic pseudo-random SPD matrix: A = B Bᵀ + n I.
fn spd(n: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let b = Array2::from_shape_fn((n, n), |_| next());
    b.dot(&b.t()) + Array2::<f64>::eye(n) * n as f64
}

/// The op's result, flattened, for comparison against the reference.
fn run_op(op: &str, n: usize, seed: u64) -> Vec<f64> {
    let a = spd(n, seed);
    match op {
        "inv" => a.inv().expect("inv").into_raw_vec_and_offset().0,
        // sln_det is a bare dgetrf (no dgetri).
        "getrf" => {
            let (s, l) = a.sln_det().expect("getrf");
            vec![s, l]
        }
        "eigh" => a.eigh(UPLO::Lower).expect("eigh").0.to_vec(),
        "cholesky" => {
            a.cholesky(UPLO::Lower)
                .expect("chol")
                .into_raw_vec_and_offset()
                .0
        }
        "gemm" => a.dot(&a).into_raw_vec_and_offset().0,
        other => panic!("unknown op {other}"),
    }
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs() / scale))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if std::env::var("FERRIC_PRINT_BLAS_CONFIG").as_deref() == Ok("1") {
        let cfg = unsafe { CStr::from_ptr(openblas_get_config()) };
        println!(
            "config={:?} procs={} threads={} parallel={}",
            cfg,
            unsafe { openblas_get_num_procs() },
            unsafe { openblas_get_num_threads() },
            unsafe { openblas_get_parallel() }
        );
        if args.len() < 5 {
            return;
        }
    }
    let op = args[1].clone();
    let n: usize = args[2].parse().unwrap();
    let bt: c_int = args[3].parse().unwrap();
    let ctx = args[4].clone();
    let stack_mib: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(0);
    let conc: usize = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1);
    let callers = if ctx == "main" { 1 } else { conc.max(1) };

    // References at 1 BLAS thread on the main thread.
    unsafe { openblas_set_num_threads(1) };
    let refs: Vec<Vec<f64>> = (0..callers).map(|k| run_op(&op, n, 1 + k as u64)).collect();

    unsafe { openblas_set_num_threads(bt) };
    let t0 = std::time::Instant::now();
    let outs: Vec<Vec<f64>> = match ctx.as_str() {
        "main" => vec![run_op(&op, n, 1)],
        "spawn" => {
            let hs: Vec<_> = (0..callers)
                .map(|k| {
                    let op = op.clone();
                    let mut b = std::thread::Builder::new();
                    if stack_mib > 0 {
                        b = b.stack_size(stack_mib << 20);
                    }
                    b.spawn(move || run_op(&op, n, 1 + k as u64)).unwrap()
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        }
        "rayon" => {
            let mut b = rayon::ThreadPoolBuilder::new().num_threads(callers);
            if stack_mib > 0 {
                b = b.stack_size(stack_mib << 20);
            }
            let pool = b.build().unwrap();
            pool.install(|| {
                use rayon::prelude::*;
                (0..callers)
                    .into_par_iter()
                    .map(|k| run_op(&op, n, 1 + k as u64))
                    .collect()
            })
        }
        // Like spawn, plus one extra thread that flips the process-global
        // OpenBLAS count between 1 and `bt` for as long as the callers run:
        // the set/restore race of two concurrent `with_blas_threads` scopes.
        "toggle" => {
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let s2 = stop.clone();
            let toggler = std::thread::spawn(move || {
                let mut k = 0u64;
                while !s2.load(std::sync::atomic::Ordering::Relaxed) {
                    unsafe { openblas_set_num_threads(if k.is_multiple_of(2) { 1 } else { bt }) };
                    k += 1;
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                k
            });
            let hs: Vec<_> = (0..callers)
                .map(|k| {
                    let op = op.clone();
                    let mut b = std::thread::Builder::new();
                    if stack_mib > 0 {
                        b = b.stack_size(stack_mib << 20);
                    }
                    b.spawn(move || {
                        // Several rounds so the toggler lands mid-call.
                        let mut last = Vec::new();
                        for _ in 0..5 {
                            last = run_op(&op, n, 1 + k as u64);
                        }
                        last
                    })
                    .unwrap()
                })
                .collect();
            let r = hs.into_iter().map(|h| h.join().unwrap()).collect();
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            let flips = toggler.join().unwrap();
            eprintln!("toggler flips={flips}");
            r
        }
        other => panic!("unknown ctx {other}"),
    };
    let dt = t0.elapsed().as_secs_f64();
    let worst = outs
        .iter()
        .zip(&refs)
        .fold(0.0f64, |m, (o, r)| m.max(max_rel(o, r)));
    println!(
        "OK op={op} n={n} blas={bt} ctx={ctx} stack_mib={stack_mib} conc={callers} maxrel={worst:.2e} t={dt:.3}s"
    );
}
