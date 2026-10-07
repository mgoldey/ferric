//! Both sides of the FP64 offload decision on ONE box, same process, arms
//! interleaved (CPU, GPU, CPU, GPU, ...): OpenBLAS at 6 threads (physical cores;
//! 7-12 add nothing here) vs device GEMM INCLUDING H2D/D2H. Prints min and
//! median of 7 reps per cell. Run only when /proc/pressure/cpu `some avg10` is
//! <= 0.05 before the run, and a sampler thread (`ferric_benchmarks::quiet`)
//! finds no external CPU load during it (PSI after is informational: it includes
//! this harness's own load; any concurrent monitor such as a ps loop or htop
//! counts as external load); the sampler covers the shape loop (device setup is
//! before it); otherwise "NOT QUOTABLE: box contested" is printed.
//! Any concurrent monitor (ps loop, htop) counts as external load. Each shape also
//! checks GPU vs CPU agreement against a Higham bound and panics if exceeded.
//!
//! RAYON_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 \
//!   cargo run --release -p ferric-benchmarks --features gpu --example gpu_gemm_crossover
//!
//! FERRIC_XOVER_SMOKE=1 runs a tiny grid with 1 rep (wiring check only; the
//! numbers are not a measurement).
//!
//! Without `--features gpu` this builds to a stub that says so.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gpu_gemm_crossover needs --features gpu");
}

#[cfg(feature = "gpu")]
fn min_median(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.total_cmp(b));
    (v[0], v[v.len() / 2])
}

/// Higham bound on |gpu - cpu| for one entry of C = A*B (alpha=1, beta=0).
/// Each arm computes a length-k dot product with error <= gamma_n * (|A||B|)_ij,
/// gamma_n = n*u/(1-n*u), u = 2^-53. The device GEMM is k-blocked (block `kb`),
/// adding at most ceil(k/kb) accumulations into C, so n = k + ceil(k/kb) for it
/// and n = k for the CPU; by the triangle inequality the difference is bounded
/// by (gamma_k + gamma_n) * (|A||B|)_ij. This is conservative for FMA/blocked
/// summation orders (all are covered by the standard model).
#[cfg(feature = "gpu")]
fn higham_gamma(n: usize) -> f64 {
    let nu = n as f64 * f64::EPSILON / 2.0;
    nu / (1.0 - nu)
}

#[cfg(feature = "gpu")]
fn env_or_unset(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unset".to_string())
}

#[cfg(feature = "gpu")]
fn main() {
    use ferric_core::blas_threads::with_blas_threads;
    use ferric_core::gpu::config::derive_min_flops;
    use ferric_core::gpu::device::device;
    use ferric_core::gpu::gemm::{gemm_f64, offload_bytes};
    use ferric_core::gpu::pool::DevicePool;
    use ndarray::{linalg::general_mat_mul, Array2};
    use std::time::Instant;

    const KB: usize = 128;
    let smoke = std::env::var_os("FERRIC_XOVER_SMOKE").is_some();
    let reps: usize = if smoke { 1 } else { 7 };
    println!(
        "RAYON_NUM_THREADS={} OPENBLAS_NUM_THREADS={}",
        env_or_unset("RAYON_NUM_THREADS"),
        env_or_unset("OPENBLAS_NUM_THREADS")
    );
    let dev = device(0).expect("GPU 0");
    let pool = DevicePool::with_capacity_bytes(4 << 30);
    let psi_before = ferric_benchmarks::quiet::psi_cpu_some_avg10();
    println!("PSI cpu some avg10 before = {psi_before:.2} (must be <= 0.05 for a quotable run)");
    let sampler = ferric_benchmarks::quiet::Sampler::start();
    println!(
        "{:>6} {:>6} {:>6} {:>14} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "m", "k", "n", "flops", "cpu_min", "cpu_med", "gpu_min", "gpu_med", "min/min", "med/med"
    );
    let shapes: &[(usize, usize, usize)] = &[
        (128, 128, 128),
        (256, 256, 256),
        (384, 384, 384),
        (512, 512, 512),
        (768, 768, 768),
        (1024, 1024, 1024),
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (3072, 3072, 3072),
        (4096, 4096, 4096),
        (6144, 6144, 6144),
        (414, 558, 414),
        (3726, 414, 414),
        (414, 3726, 414),
        (393, 912, 5895),
        (1600, 128, 1600),
    ];
    let shapes = if smoke { &shapes[..2] } else { shapes };
    let mut table: Vec<(usize, f64, f64)> = Vec::new();
    for &(m, k, n) in shapes {
        assert!(offload_bytes(m, k, n) <= 4 << 30);
        let a = Array2::from_shape_fn((m, k), |(i, j)| {
            ((i * 31 + j * 17) % 97) as f64 / 97.0 - 0.5
        });
        let b = Array2::from_shape_fn((k, n), |(i, j)| {
            ((i * 13 + j * 29) % 89) as f64 / 89.0 - 0.5
        });
        let mut c_cpu = Array2::zeros((m, n));
        let mut c_gpu = Array2::zeros((m, n));
        // Warm-up (cuBLAS/context init, page-in) on both arms, untimed; each
        // arm into its own output so the arms can be compared.
        with_blas_threads(6, || general_mat_mul(1.0, &a, &b, 0.0, &mut c_cpu));
        gemm_f64(&dev, &pool, &a.view(), &b.view(), &mut c_gpu.view_mut(), KB).unwrap();
        // Agreement check (depth k; see higham_gamma). Fails loudly so a broken
        // device arm cannot look fast.
        let mut absprod = Array2::zeros((m, n));
        general_mat_mul(1.0, &a.mapv(f64::abs), &b.mapv(f64::abs), 0.0, &mut absprod);
        let g = higham_gamma(k) + higham_gamma(k + k.div_ceil(KB));
        for ((cc, gg), ap) in c_cpu.iter().zip(c_gpu.iter()).zip(absprod.iter()) {
            let bound = g * ap + f64::MIN_POSITIVE;
            assert!(
                (gg - cc).abs() <= bound,
                "GPU/CPU disagree at shape (m={m}, k={k}, n={n}): |{gg} - {cc}| > {bound:e}"
            );
        }
        let (mut cpu, mut gpu) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            // Thread-count save/set is outside the timed region.
            let dt = with_blas_threads(6, || {
                let t = Instant::now();
                general_mat_mul(1.0, &a, &b, 0.0, &mut c_cpu);
                t.elapsed().as_secs_f64()
            });
            cpu.push(dt);
            let t = Instant::now();
            gemm_f64(&dev, &pool, &a.view(), &b.view(), &mut c_gpu.view_mut(), KB).unwrap();
            gpu.push(t.elapsed().as_secs_f64());
        }
        let (cmin, cmed) = min_median(cpu);
        let (gmin, gmed) = min_median(gpu);
        table.push((2 * m * n * k, cmed, gmed));
        println!(
            "{m:>6} {k:>6} {n:>6} {:>14} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>8.2} {:>8.2}",
            2 * m * n * k,
            cmin * 1e3,
            cmed * 1e3,
            gmin * 1e3,
            gmed * 1e3,
            gmin / cmin,
            gmed / cmed
        );
    }
    let psi_after = ferric_benchmarks::quiet::psi_cpu_some_avg10();
    let summary = sampler.finish();
    if !ferric_benchmarks::quiet::print_report(psi_before, psi_after, &summary) {
        println!("NOT QUOTABLE: box contested");
    }
    println!(
        "candidate default (apply in step 1.3b): FERRIC_GPU_MIN_FLOPS = {}",
        derive_min_flops(&table)
    );
}
