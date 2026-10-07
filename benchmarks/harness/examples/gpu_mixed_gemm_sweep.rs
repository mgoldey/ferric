//! Panel-width sweep for the mixed-precision GEMM (plan §3.2 variants (b) and
//! (c), §3.6 CPU counterpart), ONE box, ONE process, arms interleaved per rep.
//! Accuracy (rms of e_ij = |got - ref| / (|A||B|)_ij against the CPU f64
//! product) is deterministic and quotable on any box; the TIMINGS are quotable
//! only when `/proc/pressure/cpu` `some avg10` is <= 0.05 before and after
//! (otherwise "NOT QUOTABLE: box contested" is printed).
//!
//! RAYON_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 \
//!   cargo run --release -p ferric-benchmarks --features gpu --example gpu_mixed_gemm_sweep
//!
//! `FERRIC_SWEEP_SMOKE=1`: first (small) shape only, 1 rep; a wiring check, not
//! a measurement; the rules ARE evaluated over its rows (restricted to the shape
//! it ran) so a naming mismatch fails the smoke run, but the verdicts mean nothing.
//!
//! Operands are in [0, 1) (positive: |A||B| = AB, so an f32 accumulator cannot
//! hide behind cancellation). Arm rows carry their own name, time and error.
//! The `gpu_split3_resident_128` arm leaves the sum of its three f64 results
//! to the host; that sum is OUTSIDE the timed region (stated in the table).
//!
//! Without `--features gpu` this builds to a stub that says so.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gpu_mixed_gemm_sweep needs --features gpu");
}

#[cfg(feature = "gpu")]
fn psi_cpu_some_avg10() -> f64 {
    std::fs::read_to_string("/proc/pressure/cpu")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("some"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kv| kv.strip_prefix("avg10="))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(f64::NAN)
}

#[cfg(feature = "gpu")]
fn min_median(v: &[f64]) -> (f64, f64) {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    (v[0], v[v.len() / 2])
}

#[cfg(feature = "gpu")]
fn env_or_unset(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unset".to_string())
}

#[cfg(feature = "gpu")]
fn lcg_positive(rows: usize, cols: usize, seed: u64) -> ndarray::Array2<f64> {
    let mut s = seed;
    ndarray::Array2::from_shape_fn((rows, cols), |_| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    })
}

#[cfg(feature = "gpu")]
fn rms_normalized(
    got: &ndarray::Array2<f64>,
    r: &ndarray::Array2<f64>,
    ab: &ndarray::Array2<f64>,
) -> f64 {
    let (mut sum2, mut n) = (0.0f64, 0usize);
    for ((g, rr), s) in got.iter().zip(r.iter()).zip(ab.iter()) {
        if *s > 0.0 {
            let e = (g - rr) / s;
            sum2 += e * e;
            n += 1;
        }
    }
    (sum2 / n.max(1) as f64).sqrt()
}

/// One timed arm: `run(true)` is the timed call (returns seconds, no output);
/// `run(false)` is an untimed call that also returns the f64 result for the
/// error figure.
#[cfg(feature = "gpu")]
struct Arm<'a> {
    name: String,
    run: Box<dyn FnMut(bool) -> (f64, Option<ndarray::Array2<f64>>) + 'a>,
    times: Vec<f64>,
    rms: f64,
}

#[cfg(feature = "gpu")]
fn main() {
    use ferric_core::blas_threads::with_blas_threads;
    use ferric_core::gpu::device::device;
    use ferric_core::gpu::gemm::{dev_left, dev_right, gemm_f64_dev};
    use ferric_core::gpu::mixed::{
        effective_k_panel, gemm_f32_f64acc, gemm_f32_f64acc_dev, mixed_offload_bytes,
        MIXED_K_PANEL_DEFAULT,
    };
    use ferric_core::gpu::mixed_host::gemm_f32_f64acc_host;
    use ferric_core::gpu::mixed_rules::{evaluate_rules, render, ArmRow, ShapeRows, PANELS};
    use ferric_core::gpu::pool::DevicePool;
    use ndarray::{linalg::general_mat_mul, Array2};
    use std::time::Instant;

    let smoke = std::env::var_os("FERRIC_SWEEP_SMOKE").is_some();
    let reps: usize = if smoke { 1 } else { 7 };
    println!(
        "RAYON_NUM_THREADS={} OPENBLAS_NUM_THREADS={} MIXED_K_PANEL_DEFAULT={} effective={}",
        env_or_unset("RAYON_NUM_THREADS"),
        env_or_unset("OPENBLAS_NUM_THREADS"),
        MIXED_K_PANEL_DEFAULT,
        effective_k_panel()
    );
    let psi_before = psi_cpu_some_avg10();
    println!("PSI cpu some avg10 before = {psi_before:.2} (must be <= 0.05 for quotable timings)");
    let dev_arc = device(0).expect("GPU 0");
    let dev = &*dev_arc;
    println!("device: {}", dev.info.name);
    let pool = DevicePool::with_capacity_bytes(3 << 30);
    let b_cpu = effective_k_panel();

    let all_shapes: [(usize, usize, usize); 4] = [
        (256, 8192, 256),
        (393, 912, 5895),
        (1024, 4096, 1024),
        (7921, 7921, 289),
    ];
    let shapes = if smoke {
        &all_shapes[..1]
    } else {
        &all_shapes[..]
    };

    // (shape, arm name -> (median_s, rms)) for the rules after the tables.
    let mut results: Vec<ShapeRows> = Vec::new();

    for &(m, k, n) in shapes {
        let a = lcg_positive(m, k, 1);
        let b = lcg_positive(k, n, 2);
        let a32: Array2<f32> = a.mapv(|x| x as f32);
        let b32: Array2<f32> = b.mapv(|x| x as f32);
        // Variant (c): x = hi + lo, hi = x as f32, lo = (x - hi) as f32.
        let a_hi: Array2<f32> = a32.clone();
        let b_hi: Array2<f32> = b32.clone();
        let a_lo: Array2<f32> = ndarray::Zip::from(&a)
            .and(&a_hi)
            .map_collect(|&x, &h| (x - f64::from(h)) as f32);
        let b_lo: Array2<f32> = ndarray::Zip::from(&b)
            .and(&b_hi)
            .map_collect(|&x, &h| (x - f64::from(h)) as f32);

        let mut r_cpu = Array2::zeros((m, n));
        with_blas_threads(6, || general_mat_mul(1.0, &a, &b, 0.0, &mut r_cpu));
        let ab = {
            // positive operands: |A||B| = AB
            r_cpu.clone()
        };

        let s = &dev.stream;
        let flat = |x: &Array2<f64>| x.as_slice().expect("standard layout").to_vec();
        let flat32 = |x: &Array2<f32>| x.as_slice().expect("standard layout").to_vec();
        let d_a64 = s.clone_htod(&flat(&a)).expect("H2D a64");
        let d_b64 = s.clone_htod(&flat(&b)).expect("H2D b64");
        let d_a32 = s.clone_htod(&flat32(&a32)).expect("H2D a32");
        let d_b32 = s.clone_htod(&flat32(&b32)).expect("H2D b32");
        let d_alo = s.clone_htod(&flat32(&a_lo)).expect("H2D a_lo");
        let d_blo = s.clone_htod(&flat32(&b_lo)).expect("H2D b_lo");
        let av = a.view();
        let bv = b.view();
        let a32v = a32.view();
        let b32v = b32.view();

        let mut arms: Vec<Arm> = Vec::new();

        // CPU f64 (6 physical cores).
        {
            let (a, b) = (&a, &b);
            let mut c = Array2::<f64>::zeros((m, n));
            arms.push(Arm {
                name: "cpu_f64".into(),
                run: Box::new(move |timed| {
                    let dt = with_blas_threads(6, || {
                        let t = Instant::now();
                        general_mat_mul(1.0, a, b, 0.0, &mut c);
                        t.elapsed().as_secs_f64()
                    });
                    (dt, if timed { None } else { Some(c.clone()) })
                }),
                times: vec![],
                rms: 0.0,
            });
        }
        // CPU mixed twin at the current panel width.
        {
            let mut c = Array2::<f64>::zeros((m, n));
            arms.push(Arm {
                name: format!("cpu_mixed_b{b_cpu}"),
                run: Box::new(move |timed| {
                    let dt = with_blas_threads(6, || {
                        let t = Instant::now();
                        gemm_f32_f64acc_host(&av, &bv, &mut c.view_mut(), b_cpu);
                        t.elapsed().as_secs_f64()
                    });
                    (dt, if timed { None } else { Some(c.clone()) })
                }),
                times: vec![],
                rms: 0.0,
            });
        }
        // GPU f64 resident.
        {
            let mut c = s.alloc_zeros::<f64>(m * n).expect("alloc c f64");
            let (d_a, d_b) = (&d_a64, &d_b64);
            arms.push(Arm {
                name: "gpu_f64_resident".into(),
                run: Box::new(move |timed| {
                    let lo = dev_left(d_a.slice(..), &av).expect("left layout");
                    let ro = dev_right(d_b.slice(..), &bv).expect("right layout");
                    let t = Instant::now();
                    gemm_f64_dev(dev, m, k, n, &lo, &ro, &mut c, 128).expect("gemm_f64_dev");
                    dev.stream.synchronize().expect("sync");
                    let dt = t.elapsed().as_secs_f64();
                    let out = (!timed).then(|| {
                        let host = dev.stream.clone_dtoh(&c).expect("D2H");
                        Array2::from_shape_vec((m, n), host).expect("shape")
                    });
                    (dt, out)
                }),
                times: vec![],
                rms: 0.0,
            });
        }
        // GPU mixed resident, one arm per panel width; "sgemm" is k_panel = k.
        let mut resident_panels: Vec<(String, usize)> = vec![("gpu_sgemm_resident".into(), k)];
        for &p in &PANELS {
            resident_panels.push((format!("gpu_mixed_resident_b{p}"), p));
        }
        for (name, p) in resident_panels {
            let mut c32 = s.alloc_zeros::<f32>(m * n).expect("alloc c32");
            let mut c64 = s.alloc_zeros::<f64>(m * n).expect("alloc c64");
            let (d_a, d_b) = (&d_a32, &d_b32);
            arms.push(Arm {
                name,
                run: Box::new(move |timed| {
                    let lo = dev_left(d_a.slice(..), &a32v).expect("left layout");
                    let ro = dev_right(d_b.slice(..), &b32v).expect("right layout");
                    let t = Instant::now();
                    gemm_f32_f64acc_dev(dev, m, k, n, &lo, &ro, &mut c32, &mut c64, p)
                        .expect("gemm_f32_f64acc_dev");
                    dev.stream.synchronize().expect("sync");
                    let dt = t.elapsed().as_secs_f64();
                    let out = (!timed).then(|| {
                        let host = dev.stream.clone_dtoh(&c64).expect("D2H");
                        Array2::from_shape_vec((m, n), host).expect("shape")
                    });
                    (dt, out)
                }),
                times: vec![],
                rms: 0.0,
            });
        }
        // GPU mixed per call (includes rounding, H2D and D2H).
        for p in [128usize, 512] {
            let mut c = Array2::<f64>::zeros((m, n));
            let pool = &pool;
            arms.push(Arm {
                name: format!("gpu_mixed_percall_{p}"),
                run: Box::new(move |timed| {
                    assert!(mixed_offload_bytes(m, k, n) <= 3 << 30);
                    let t = Instant::now();
                    gemm_f32_f64acc(dev, pool, &av, &bv, &mut c.view_mut(), p).expect("percall");
                    let dt = t.elapsed().as_secs_f64();
                    (dt, if timed { None } else { Some(c.clone()) })
                }),
                times: vec![],
                rms: 0.0,
            });
        }
        // Variant (c): three mixed products hi*hi + hi*lo + lo*hi, b = 128.
        // The host-side sum of the three f64 downloads is OUTSIDE the timed region.
        {
            let mut c32 = s.alloc_zeros::<f32>(m * n).expect("alloc c32");
            let mut c_hh = s.alloc_zeros::<f64>(m * n).expect("alloc hh");
            let mut c_hl = s.alloc_zeros::<f64>(m * n).expect("alloc hl");
            let mut c_lh = s.alloc_zeros::<f64>(m * n).expect("alloc lh");
            let (d_ahi, d_bhi, d_alo, d_blo) = (&d_a32, &d_b32, &d_alo, &d_blo);
            let (a_lo_v, b_lo_v) = (a_lo.view(), b_lo.view());
            arms.push(Arm {
                name: "gpu_split3_resident_128 (host sum untimed)".into(),
                run: Box::new(move |timed| {
                    let ah = dev_left(d_ahi.slice(..), &a32v).expect("layout");
                    let bh = dev_right(d_bhi.slice(..), &b32v).expect("layout");
                    let al = dev_left(d_alo.slice(..), &a_lo_v).expect("layout");
                    let bl = dev_right(d_blo.slice(..), &b_lo_v).expect("layout");
                    let t = Instant::now();
                    gemm_f32_f64acc_dev(dev, m, k, n, &ah, &bh, &mut c32, &mut c_hh, 128)
                        .expect("hh");
                    gemm_f32_f64acc_dev(dev, m, k, n, &ah, &bl, &mut c32, &mut c_hl, 128)
                        .expect("hl");
                    gemm_f32_f64acc_dev(dev, m, k, n, &al, &bh, &mut c32, &mut c_lh, 128)
                        .expect("lh");
                    dev.stream.synchronize().expect("sync");
                    let dt = t.elapsed().as_secs_f64();
                    let out = (!timed).then(|| {
                        let x = dev.stream.clone_dtoh(&c_hh).expect("D2H");
                        let y = dev.stream.clone_dtoh(&c_hl).expect("D2H");
                        let z = dev.stream.clone_dtoh(&c_lh).expect("D2H");
                        let sum: Vec<f64> = x
                            .iter()
                            .zip(&y)
                            .zip(&z)
                            .map(|((p, q), r)| p + q + r)
                            .collect();
                        Array2::from_shape_vec((m, n), sum).expect("shape")
                    });
                    (dt, out)
                }),
                times: vec![],
                rms: 0.0,
            });
        }

        // Warm-up + error figure (untimed), then interleaved timed reps.
        for arm in arms.iter_mut() {
            let (_, out) = (arm.run)(false);
            arm.rms = rms_normalized(&out.expect("output"), &r_cpu, &ab);
        }
        for _ in 0..reps {
            for arm in arms.iter_mut() {
                let (dt, _) = (arm.run)(true);
                arm.times.push(dt);
            }
        }

        println!(
            "\nshape (m,k,n) = ({m},{k},{n}), {} reps, flops = {}",
            reps,
            2 * m * n * k
        );
        println!(
            "{:<44} {:>9} {:>9} {:>10} {:>10} {:>11}",
            "arm", "min_ms", "med_ms", "min_GF/s", "med_GF/s", "rms_e"
        );
        let flops = (2 * m * n * k) as f64;
        let mut row = Vec::new();
        for arm in &arms {
            let (tmin, tmed) = min_median(&arm.times);
            println!(
                "{:<44} {:>9.3} {:>9.3} {:>10.1} {:>10.1} {:>11.3e}",
                arm.name,
                tmin * 1e3,
                tmed * 1e3,
                flops / tmin / 1e9,
                flops / tmed / 1e9,
                arm.rms
            );
            row.push(ArmRow {
                name: arm.name.clone(),
                median_s: tmed,
                rms: arm.rms,
            });
        }
        results.push(ShapeRows {
            shape: (m, k, n),
            rows: row,
        });
    }

    let psi_after = psi_cpu_some_avg10();
    println!("\nPSI cpu some avg10 after = {psi_after:.2}");
    let quiet = |p: f64| p.is_finite() && p <= 0.05;
    let quotable = quiet(psi_before) && quiet(psi_after);
    if !quotable {
        println!("NOT QUOTABLE: box contested (timings and every rule verdict below are wiring checks only)");
    }
    // The pre-registered rules live in ferric_core::gpu::mixed_rules (pure,
    // unit-tested). The smoke run evaluates them over its real rows, with the
    // panel rule restricted to the shapes it ran, so an arm-name mismatch
    // fails here instead of at the end of a full sweep.
    let rule_shapes: Vec<(usize, usize, usize)> = if smoke {
        shapes.to_vec()
    } else {
        vec![(393, 912, 5895), (256, 8192, 256)]
    };
    match evaluate_rules(&results, &rule_shapes) {
        Ok(outcome) => {
            for line in render(&outcome) {
                println!("{line}");
            }
        }
        Err(e) => panic!("rule evaluation failed: {e}"),
    }
    if smoke {
        println!("smoke run: rule verdicts above are wiring checks, not results");
    }
    if !quotable {
        println!("NOT QUOTABLE: box contested");
    }
}
