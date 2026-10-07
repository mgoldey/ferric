#![cfg(feature = "gpu")]
//! The f32-matrix / f64-vector GEMV (`gemv_f32mat_f64_dev`): both directions, the
//! derived per-element bound `eps = u32 + gamma_k(u64)(1+u32)` against the exact
//! (f64) matrix, the much tighter twin bound against the f32-rounded matrix held in
//! f64, a truncating-upload defect that the twin comparison must catch, accumulate
//! semantics, zero-size and geometry refusals, and determinism run to run.
//! References are plain sequential f64 sums, so each carries its own `gamma_k(u64)`
//! and the comparison bound adds it.
use cudarc::driver::CudaSlice;
use ferric_core::gpu::device::{device, Device, GpuError};
use ferric_core::gpu::gemv::{gemv_f32mat_f64_dev, mixed_gemv_error_factor, GemvSpec};
use ferric_core::gpu::mixed_host::{gamma, U32, U64};
use ferric_core::gpu::{probe, GpuStatus};
use std::sync::Arc;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn skip() -> bool {
    match probe(0) {
        GpuStatus::Ready(_) => false,
        o => {
            eprintln!("skipping: no CUDA device ({o:?})");
            assert!(
                std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
                "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
            );
            true
        }
    }
}

fn stream_of(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }
}

/// Round toward zero (the defect), as f32.
fn trunc32(x: f64) -> f32 {
    let r = x as f32;
    if f64::from(r).abs() > x.abs() {
        f32::from_bits(r.to_bits() - 1)
    } else {
        r
    }
}

struct Case {
    rows: usize,
    cols: usize,
    a: Vec<f64>,
    x: Vec<f64>,
}

fn case(rows: usize, cols: usize, transpose: bool, seed: u64) -> Case {
    let mut g = stream_of(seed);
    let a = (0..rows * cols).map(|_| g()).collect();
    let xl = if transpose { rows } else { cols };
    let x = (0..xl).map(|_| 3.0 * g()).collect();
    Case { rows, cols, a, x }
}

/// `y = A x` or `Aᵀ x` by a sequential f64 sum of the matrix `m`, plus `S = |m||x|`.
fn host(c: &Case, m: &[f64], transpose: bool) -> (Vec<f64>, Vec<f64>) {
    let n_out = if transpose { c.cols } else { c.rows };
    let (mut y, mut s) = (vec![0.0; n_out], vec![0.0; n_out]);
    for r in 0..c.rows {
        for k in 0..c.cols {
            let (o, xi) = if transpose { (k, r) } else { (r, k) };
            y[o] += m[r * c.cols + k] * c.x[xi];
            s[o] += (m[r * c.cols + k] * c.x[xi]).abs();
        }
    }
    (y, s)
}

fn run(
    dev: &Arc<Device>,
    c: &Case,
    transpose: bool,
    a32: &[f32],
    (y0, accumulate): (Option<&[f64]>, bool),
) -> Result<Vec<f64>, GpuError> {
    let st = &dev.stream;
    let d_a: CudaSlice<f32> = st.clone_htod(a32).unwrap();
    let d_x: CudaSlice<f64> = st.clone_htod(&c.x).unwrap();
    let n_out = if transpose { c.cols } else { c.rows };
    let mut d_y: CudaSlice<f64> = match y0 {
        Some(y) => st.clone_htod(y).unwrap(),
        None => st.alloc_zeros(n_out).unwrap(),
    };
    let spec = GemvSpec {
        rows: c.rows,
        cols: c.cols,
        transpose,
        accumulate,
    };
    gemv_f32mat_f64_dev(
        dev,
        spec,
        &d_a.slice(..),
        &d_x.slice(..),
        &mut d_y.slice_mut(..),
    )?;
    let mut out = vec![0.0; n_out];
    st.memcpy_dtoh(&d_y, &mut out).unwrap();
    st.synchronize().unwrap();
    Ok(out)
}

/// Worst `|got - want| / (eps S)` over the outputs.
fn worst(got: &[f64], want: &[f64], s: &[f64], eps: f64) -> f64 {
    got.iter()
        .zip(want)
        .zip(s)
        .map(|((g, w), s)| (g - w).abs() / (eps * s))
        .fold(0.0, f64::max)
}

const SHAPES: [(usize, usize); 6] = [(1, 1), (3, 5), (7, 200), (130, 33), (257, 1025), (1000, 31)];

#[test]
fn both_directions_obey_the_derived_bound_and_the_twin_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    for transpose in [false, true] {
        for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
            let c = case(rows, cols, transpose, 11 + i as u64);
            let a32: Vec<f32> = c.a.iter().map(|&v| v as f32).collect();
            let twin: Vec<f64> = a32.iter().map(|&v| f64::from(v)).collect();
            let got = run(&dev, &c, transpose, &a32, (None, false)).unwrap();
            let k = if transpose { rows } else { cols };
            // against the exact matrix: mixed factor + the sequential reference's own gamma_k
            let (want, s) = host(&c, &c.a, transpose);
            let eps = mixed_gemv_error_factor(k) + gamma(k, U64);
            let r_exact = worst(&got, &want, &s, eps);
            // against the f32-rounded matrix held in f64: only f64 accumulation on both sides
            let (want_t, s_t) = host(&c, &twin, transpose);
            let eps_t = 2.0 * gamma(k, U64) * (1.0 + U32);
            let r_twin = worst(&got, &want_t, &s_t, eps_t);
            eprintln!(
                "gemv t={transpose} {rows}x{cols}: exact ratio {r_exact:.3e}  twin ratio {r_twin:.3e}"
            );
            assert!(r_exact <= 1.0, "exact bound violated: {r_exact}");
            assert!(r_twin <= 1.0, "twin bound violated: {r_twin}");
            // determinism run to run
            let again = run(&dev, &c, transpose, &a32, (None, false)).unwrap();
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&again), "not deterministic");
        }
    }
}

/// The twin comparison has teeth: a matrix rounded toward zero instead of to
/// nearest differs from the twin by ~2^-23 relative per entry, orders of magnitude
/// above the f64 accumulation bound.
#[test]
fn a_truncated_matrix_is_far_outside_the_twin_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    for transpose in [false, true] {
        let c = case(257, 1025, transpose, 99);
        let a32: Vec<f32> = c.a.iter().map(|&v| v as f32).collect();
        let twin: Vec<f64> = a32.iter().map(|&v| f64::from(v)).collect();
        let t32: Vec<f32> = c.a.iter().map(|&v| trunc32(v)).collect();
        let got = run(&dev, &c, transpose, &t32, (None, false)).unwrap();
        let k = if transpose { c.rows } else { c.cols };
        let (want_t, s_t) = host(&c, &twin, transpose);
        let r = worst(&got, &want_t, &s_t, 2.0 * gamma(k, U64) * (1.0 + U32));
        eprintln!("truncated t={transpose}: twin ratio {r:.3e}");
        assert!(
            r > 1.0e3,
            "the defect is not visible to the twin bound: {r}"
        );
    }
}

#[test]
fn accumulate_adds_to_y_and_overwrite_ignores_it() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    for transpose in [false, true] {
        let c = case(37, 301, transpose, 5);
        let a32: Vec<f32> = c.a.iter().map(|&v| v as f32).collect();
        let twin: Vec<f64> = a32.iter().map(|&v| f64::from(v)).collect();
        let n_out = if transpose { c.cols } else { c.rows };
        let y0: Vec<f64> = (0..n_out).map(|i| 100.0 + i as f64).collect();
        let (ax, s) = host(&c, &twin, transpose);
        let got = run(&dev, &c, transpose, &a32, (Some(&y0), true)).unwrap();
        for i in 0..n_out {
            let want = y0[i] + ax[i];
            let tol = 4.0 * gamma(301, U64) * (s[i] + want.abs());
            assert!((got[i] - want).abs() <= tol, "i={i}: {} vs {want}", got[i]);
        }
        // beta = 0 into a poisoned output: the poison must not survive
        let fresh = run(&dev, &c, transpose, &a32, (Some(&y0), false)).unwrap();
        for i in 0..n_out {
            assert!((fresh[i] - ax[i]).abs() <= 2.0 * gamma(301, U64) * s[i]);
        }
    }
}

#[test]
fn zero_sizes_and_bad_lengths_are_typed_not_panics() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let st = &dev.stream;
    // rows = 0, transpose: y has `cols` elements and is zeroed (overwrite) or kept (accumulate)
    let d_a: CudaSlice<f32> = st.alloc_zeros(1).unwrap();
    let d_x: CudaSlice<f64> = st.alloc_zeros(1).unwrap();
    let mut d_y: CudaSlice<f64> = st.clone_htod(&[7.0, 7.0, 7.0]).unwrap();
    let zero_rows = |acc: bool| GemvSpec {
        rows: 0,
        cols: 3,
        transpose: true,
        accumulate: acc,
    };
    let empty_a = d_a.slice(0..0);
    let empty_x = d_x.slice(0..0);
    gemv_f32mat_f64_dev(
        &dev,
        zero_rows(true),
        &empty_a,
        &empty_x,
        &mut d_y.slice_mut(..),
    )
    .unwrap();
    let mut out = [0.0; 3];
    st.memcpy_dtoh(&d_y, &mut out).unwrap();
    st.synchronize().unwrap();
    assert_eq!(out, [7.0; 3], "accumulate over an empty product keeps y");
    gemv_f32mat_f64_dev(
        &dev,
        zero_rows(false),
        &empty_a,
        &empty_x,
        &mut d_y.slice_mut(..),
    )
    .unwrap();
    st.memcpy_dtoh(&d_y, &mut out).unwrap();
    st.synchronize().unwrap();
    assert_eq!(out, [0.0; 3], "overwrite over an empty product zeroes y");
    // wrong lengths
    let spec = GemvSpec {
        rows: 2,
        cols: 3,
        transpose: false,
        accumulate: false,
    };
    let a6: CudaSlice<f32> = st.alloc_zeros(6).unwrap();
    let x3: CudaSlice<f64> = st.alloc_zeros(3).unwrap();
    let mut y3: CudaSlice<f64> = st.alloc_zeros(3).unwrap();
    let e = gemv_f32mat_f64_dev(
        &dev,
        spec,
        &a6.slice(..),
        &x3.slice(..),
        &mut y3.slice_mut(..),
    );
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
}

#[cfg(feature = "test-seams")]
#[test]
fn an_unloadable_kernel_is_a_typed_kernel_error() {
    use ferric_core::gpu::device::FORCE_KERNEL_FAILURE;
    use std::sync::atomic::Ordering;
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let st = &dev.stream;
    let a: CudaSlice<f32> = st.alloc_zeros(6).unwrap();
    let x: CudaSlice<f64> = st.alloc_zeros(3).unwrap();
    let mut y: CudaSlice<f64> = st.alloc_zeros(2).unwrap();
    let spec = GemvSpec {
        rows: 2,
        cols: 3,
        transpose: false,
        accumulate: false,
    };
    FORCE_KERNEL_FAILURE.store(true, Ordering::SeqCst);
    let e = gemv_f32mat_f64_dev(&dev, spec, &a.slice(..), &x.slice(..), &mut y.slice_mut(..));
    FORCE_KERNEL_FAILURE.store(false, Ordering::SeqCst);
    assert!(matches!(e, Err(GpuError::Kernel(_))), "{e:?}");
    gemv_f32mat_f64_dev(&dev, spec, &a.slice(..), &x.slice(..), &mut y.slice_mut(..)).unwrap();
}
