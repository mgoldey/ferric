#![cfg(feature = "gpu")]
//! The strided-batched mixed GEMM (`gemm_f32_f64acc_strided_batched_dev`, the
//! kernel of the device XC pass 2) through its own entry point, in the XC
//! layout: left `A_b` compact `(m, k)` at stride `m·k`, right `X_bᵀ` the first
//! `k` columns of a row-major `(n, w)` AO block, transposed, at stride `n·w`.
//!
//! * Per element, every matrix of the batch: `|ĉ − c| ≤ (mixed_error_factor(k, b)
//!   + γ_k(u64))·(|A||X|ᵀ)_ij` (the per-matrix kernel's bound, 1.01 slack), and
//!   the output past `batch·m·n` keeps its NaN sentinel bit for bit.
//! * Cross-panel flush pin: panel sums +2^20, t_b, −2^20 (each exact in f32) give
//!   t_b exactly for every matrix b (t_b differs per matrix, so a stride error
//!   shows); summing the panels in f32 returns 0.
//! * Panel count: the return value is ⌈k/b⌉ for the whole batch (not per matrix).
//! * RMS(err/S) band against the host twin, measured on both sides; see
//!   `the_batched_kernel_sits_in_the_twins_rms_band_set_by_measured_mutants`.
//!
//! MUTATION CHECKS (each applied to `gemm_f32_f64acc_strided_batched_dev` one at
//! a time, confirmed with grep, then restored byte for byte; GTX 1080):
//! * panels summed in f32 (`beta = 1` after panel 0, one flush at the end):
//!   fails the flush pin, the RMS band (deep: 4.70 vs hi 2.16) and the
//!   per-element bound (b = 1 case: 1.23).
//! * panel width doubled (`kb = 2·k_panel`): fails the panel-count pin, the
//!   flush pin and the RMS band (XC shape: 2.07 vs hi 1.44).
//! * flush skipped on panel 0: fails the per-element bound (9.1e4), the flush
//!   pin and the band.
use ferric_core::gpu::batched::{dev_batched_left, dev_batched_right, BatchedDims};
use ferric_core::gpu::device::device;
use ferric_core::gpu::gemm::{dev_left, dev_right_padded};
use ferric_core::gpu::mixed::{gemm_f32_f64acc_dev_into, gemm_f32_f64acc_strided_batched_dev};
use ferric_core::gpu::mixed_host::{gamma, gemm_f32_f64acc_host, mixed_error_factor, U64};
use ferric_core::gpu::{probe, GpuStatus};
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, ArrayView2};

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// NaN with a payload: the output past the batch must keep these exact bits.
const SENTINEL: u64 = 0x7ff8_dead_beef_0002;

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

fn lcg(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    })
}

/// One XC-layout batch: `a[b]` is `(m, k)`, `x[b]` is the `(n, w)` AO block
/// whose first `k` columns are `X_bᵀ`'s rows.
struct Batch {
    a: Vec<Array2<f64>>,
    x: Vec<Array2<f64>>,
    k: usize,
}

impl Batch {
    fn random(
        m: usize,
        n: usize,
        k: usize,
        w: usize,
        batch: usize,
        seed: u64,
        signed: bool,
    ) -> Self {
        let shift = if signed { 0.5 } else { 0.0 };
        let a = (0..batch)
            .map(|b| lcg(m, k, seed + 2 * b as u64).mapv(|v| v - shift))
            .collect();
        let x = (0..batch)
            .map(|b| lcg(n, w, seed + 2 * b as u64 + 1).mapv(|v| v - shift))
            .collect();
        Self { a, x, k }
    }

    fn xt(&self, b: usize) -> ArrayView2<'_, f64> {
        self.x[b].slice(s![.., ..self.k]).reversed_axes()
    }

    fn dims(&self) -> BatchedDims {
        BatchedDims {
            m: self.a[0].nrows(),
            k: self.k,
            n: self.x[0].nrows(),
            batch: self.a.len(),
        }
    }

    /// f64 reference and S = |A||X|ᵀ of matrix `b`.
    fn reference(&self, b: usize) -> (Array2<f64>, Array2<f64>) {
        let (l, r) = (self.a[b].view(), self.xt(b));
        let mut c = Array2::zeros((l.nrows(), r.ncols()));
        general_mat_mul(1.0, &l, &r, 0.0, &mut c);
        let mut sab = Array2::zeros(c.dim());
        general_mat_mul(1.0, &l.mapv(f64::abs), &r.mapv(f64::abs), 0.0, &mut sab);
        (c, sab)
    }

    /// The host twin of matrix `b` at panel width `kp`.
    fn twin(&self, b: usize, kp: usize) -> Array2<f64> {
        let mut out = Array2::zeros((self.a[b].nrows(), self.x[b].nrows()));
        gemm_f32_f64acc_host(&self.a[b].view(), &self.xt(b), &mut out.view_mut(), kp);
        out
    }

    fn flat32(v: &[Array2<f64>]) -> Vec<f32> {
        v.iter()
            .flat_map(|m| m.iter().map(|&x| x as f32).collect::<Vec<_>>())
            .collect()
    }

    /// The batched device kernel: every matrix's result and the panel count.
    /// The output buffer is `extra` elements longer than the batch and starts
    /// as SENTINEL; the tail must come back untouched.
    fn device(&self, kp: usize, extra: usize) -> (Vec<Array2<f64>>, usize) {
        let d = self.dims();
        let dev = device(0).unwrap();
        let s = &dev.stream;
        let (m, n, w) = (d.m, d.n, self.x[0].ncols());
        let d_a = s.clone_htod(&Self::flat32(&self.a)).unwrap();
        let d_x = s.clone_htod(&Self::flat32(&self.x)).unwrap();
        let a0 = Array2::<()>::from_elem((m, d.k), ());
        let x0 = Array2::<()>::from_elem((n, w), ());
        let x0t = x0.slice(s![.., ..d.k]).reversed_axes();
        let l = dev_batched_left(d_a.slice(..), &a0.view(), d.batch, m * d.k).unwrap();
        let r = dev_batched_right(d_x.slice(..), &x0t, d.batch, n * w).unwrap();
        let total = d.batch * m * n;
        let mut c32 = s.alloc_zeros::<f32>(total + extra).unwrap();
        let mut c64 = s
            .clone_htod(&vec![f64::from_bits(SENTINEL); total + extra])
            .unwrap();
        let panels = gemm_f32_f64acc_strided_batched_dev(
            &dev,
            d,
            &l,
            &r,
            &mut c32.slice_mut(..),
            &mut c64.slice_mut(..),
            kp,
        )
        .unwrap();
        let host = s.clone_dtoh(&c64).unwrap();
        for (i, v) in host[total..].iter().enumerate() {
            assert_eq!(
                v.to_bits(),
                SENTINEL,
                "output written past the batch at +{i}"
            );
        }
        let out = (0..d.batch)
            .map(|b| {
                Array2::from_shape_vec((m, n), host[b * m * n..(b + 1) * m * n].to_vec()).unwrap()
            })
            .collect();
        (out, panels)
    }

    /// The per-matrix device kernel (the XC per-batch loop), for comparison.
    fn device_loop(&self, kp: usize) -> Vec<Array2<f64>> {
        let d = self.dims();
        let dev = device(0).unwrap();
        let s = &dev.stream;
        let (m, n, w) = (d.m, d.n, self.x[0].ncols());
        let x0 = Array2::<()>::from_elem((n, w), ());
        let x0t = x0.slice(s![.., ..d.k]).reversed_axes();
        let a0 = Array2::<()>::from_elem((m, d.k), ());
        (0..d.batch)
            .map(|b| {
                let d_a = s.clone_htod(&Self::flat32(&self.a[b..=b])).unwrap();
                let d_x = s.clone_htod(&Self::flat32(&self.x[b..=b])).unwrap();
                let l = dev_left(d_a.slice(..), &a0.view()).unwrap();
                let r = dev_right_padded(d_x.slice(..), &x0t).unwrap();
                let mut c32 = s.alloc_zeros::<f32>(m * n).unwrap();
                let mut c64 = s.alloc_zeros::<f64>(m * n).unwrap();
                gemm_f32_f64acc_dev_into(
                    &dev,
                    (m, d.k, n),
                    &l,
                    &r,
                    &mut c32.slice_mut(..),
                    &mut c64.slice_mut(..),
                    kp,
                )
                .unwrap();
                Array2::from_shape_vec((m, n), s.clone_dtoh(&c64).unwrap()).unwrap()
            })
            .collect()
    }
}

/// Running per-element statistics of e = |got − ref| / S.
#[derive(Default)]
struct Elem {
    sum2: f64,
    n: usize,
    worst: f64,
}

impl Elem {
    fn add(&mut self, got: &Array2<f64>, r: &Array2<f64>, sab: &Array2<f64>, bound: f64) {
        for ((g, rr), s) in got.iter().zip(r.iter()).zip(sab.iter()) {
            if *s > 0.0 {
                let e = (g - rr).abs() / s;
                self.worst = self.worst.max(e / bound);
                self.sum2 += e * e;
                self.n += 1;
            }
        }
    }

    fn rms(&self) -> f64 {
        (self.sum2 / self.n.max(1) as f64).sqrt()
    }
}

fn bound(k: usize, kp: usize) -> f64 {
    (mixed_error_factor(k, kp) + gamma(k, U64)) * 1.01
}

#[test]
fn every_matrix_of_a_batch_meets_the_per_element_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    // (m, n, k, w, batch, kp): the XC shape (np 16..64, k = NB <= 128, w = 4 NB),
    // k below / at / just above a panel, deep k, a batch of one, odd sizes.
    for &(m, n, k, w, batch, kp) in &[
        (64usize, 64usize, 128usize, 512usize, 24usize, 64usize),
        (48, 48, 100, 400, 7, 64),
        (16, 16, 64, 256, 5, 64),
        (16, 16, 65, 260, 5, 64),
        (32, 32, 7, 28, 3, 64),
        (33, 17, 300, 300, 4, 64),
        (32, 32, 1024, 1024, 3, 64),
        (40, 40, 128, 512, 1, 64),
        (24, 24, 129, 516, 6, 1),
    ] {
        let bt = Batch::random(m, n, k, w, batch, 101 + k as u64, true);
        let (got, _) = bt.device(kp, 37);
        let mut el = Elem::default();
        for (b, g) in got.iter().enumerate() {
            let (r, sab) = bt.reference(b);
            el.add(g, &r, &sab, bound(k, kp));
        }
        eprintln!(
            "{m}x{k}x{n} batch {batch} b={kp}: max e/bound {:.3e}",
            el.worst
        );
        assert!(
            el.worst <= 1.0,
            "{m}x{k}x{n} batch {batch} b={kp}: max e/bound {:e}",
            el.worst
        );
    }
}

/// Mutant emulation on the host: panels summed in f32, one f64 conversion at
/// the end (the flush lost between panels).
fn f32_across_panels(l: &ArrayView2<f64>, r: &ArrayView2<f64>, kp: usize) -> Array2<f64> {
    let (a32, b32) = (l.mapv(|x| x as f32), r.mapv(|x| x as f32));
    let k = l.ncols();
    let mut acc = Array2::<f32>::zeros((l.nrows(), r.ncols()));
    let mut k0 = 0;
    while k0 < k {
        let k1 = (k0 + kp).min(k);
        let mut c = Array2::<f32>::zeros(acc.dim());
        general_mat_mul(
            1.0f32,
            &a32.slice(s![.., k0..k1]),
            &b32.slice(s![k0..k1, ..]),
            0.0f32,
            &mut c,
        );
        acc += &c;
        k0 = k1;
    }
    acc.mapv(f64::from)
}

#[test]
fn the_f64_flush_across_panels_keeps_what_an_f32_sum_loses_in_every_matrix() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let (kp, batch, m, n) = (64usize, 5usize, 3usize, 2usize);
    let k = 3 * kp;
    let w = k + 9; // padded AO block, as in the XC layout
    let big = 1_048_576.0f64; // 2^20; ulp32(2^20) = 2^-3
                              // t_b = (b + 1)·2^-9: exact in f32, lost against 2^20 in an f32 sum.
    let t = |b: usize| (b + 1) as f64 * 2.0f64.powi(-9);
    let mut bt = Batch::random(m, n, k, w, batch, 7, true);
    for b in 0..batch {
        // A row 0 = ones; X row 0 holds one value per panel; everything else
        // in row 0 / column 0 zero, so (0,0) is exactly the three panel sums.
        bt.a[b].row_mut(0).fill(1.0);
        let mut x0 = bt.x[b].row_mut(0);
        x0.fill(0.0);
        x0[0] = big;
        x0[kp] = t(b);
        x0[2 * kp] = -big;
    }
    let (got, panels) = bt.device(kp, 0);
    assert_eq!(panels, 3);
    for (b, g) in got.iter().enumerate() {
        assert_eq!(
            g[(0, 0)].to_bits(),
            t(b).to_bits(),
            "matrix {b}: the f64 flush must keep the middle panel's {}",
            t(b)
        );
        // the construction has teeth: an f32 sum across the panels loses it
        let mutant = f32_across_panels(&bt.a[b].view(), &bt.xt(b), kp);
        assert_eq!(
            mutant[(0, 0)],
            0.0,
            "matrix {b}: the f32-sum emulation keeps it"
        );
    }
    // 2^24 + 1 over two panels: 16777217 in f64, 16777216 in f32.
    let k = 2 * kp;
    let mut bt = Batch::random(m, n, k, k, batch, 9, true);
    for b in 0..batch {
        bt.a[b].row_mut(0).fill(1.0);
        let mut x0 = bt.x[b].row_mut(0);
        x0.fill(0.0);
        x0[0] = 16_777_216.0;
        x0[kp] = 1.0;
    }
    let (got, panels) = bt.device(kp, 0);
    assert_eq!(panels, 2);
    for (b, g) in got.iter().enumerate() {
        assert_eq!(g[(0, 0)], 16_777_217.0, "matrix {b}");
    }
}

#[test]
fn the_batch_flushes_exactly_ceil_k_over_b_panels() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    for &(k, kp, batch) in &[
        (128usize, 64usize, 9usize),
        (100, 64, 9),
        (64, 64, 4),
        (65, 64, 4),
        (129, 64, 1),
        (300, 128, 3),
        (7, 1, 2),
        (5, 0, 2),
        (40, 1000, 3),
    ] {
        let bt = Batch::random(8, 8, k, k + 3, batch, 31, true);
        let (_, panels) = bt.device(kp, 0);
        assert_eq!(
            panels,
            k.div_ceil(kp.max(1)),
            "k={k} b={kp} batch={batch}: one count for the whole batch"
        );
    }
}

/// Per-element RMS(err/S) of the batched device kernel, the per-matrix device
/// kernel, the host twin and the defect emulations, on one batch.
fn rms_table(bt: &Batch, kp: usize) -> [f64; 7] {
    let k = bt.k;
    let bd = bound(k, kp);
    let (dev_b, _) = bt.device(kp, 0);
    let (dev_one, _) = bt.device(k, 0); // plain SGEMM: one panel, reachable via the API
    let dev_l = bt.device_loop(kp);
    let mut e: [Elem; 7] = Default::default();
    for b in 0..bt.a.len() {
        let (r, sab) = bt.reference(b);
        let (l, x) = (bt.a[b].view(), bt.xt(b));
        let storage_only = {
            let (l32, x32) = (l.mapv(|v| v as f32 as f64), x.mapv(|v| v as f32 as f64));
            l32.dot(&x32)
        };
        e[0].add(&dev_b[b], &r, &sab, bd);
        e[1].add(&dev_l[b], &r, &sab, bd);
        e[2].add(&bt.twin(b, kp), &r, &sab, bd);
        e[3].add(&dev_one[b], &r, &sab, bound(k, k));
        e[4].add(&f32_across_panels(&l, &x, kp), &r, &sab, f64::INFINITY);
        e[5].add(&storage_only, &r, &sab, bd);
        e[6].add(&bt.twin(b, k), &r, &sab, bound(k, k));
    }
    assert!(
        e[0].worst <= 1.0,
        "batched device element over its bound: {}",
        e[0].worst
    );
    e.map(|x| x.rms())
}

/// The RMS(err/S) band of the batched kernel against the host twin, with both
/// sides measured in the test (as in `ferric-mp2`'s `gpu_rimp2_mixed.rs`):
/// `lo = sqrt(storage-only / twin)`, `hi = sqrt(nearest upper defect / twin)`,
/// the geometric midpoints between the clean ratio 1 and each defect.
///
/// MEASURED (GTX 1080, sm_61, cuBLAS 12.0, positive operands, b = 64; RMS
/// ratios to the twin): the batched kernel 1.065 at the XC shape (64x128x64,
/// batch 64) and 1.18 deep (32x2048x32, batch 16), i.e. the twin's class: the
/// batched SGEMM sums a panel about as a sequential f32 sum does. The
/// per-matrix device kernel (the XC per-batch loop) reads 0.40-0.45: cuBLAS's
/// single-matrix SGEMM is ~2.5x more accurate inside a panel, which is why the
/// batched XC V_xc is a few times further from f64 than the loop's (max|dV|
/// 3-4x) without any defect. Defects: plain SGEMM (one panel, through this API)
/// 2.07 / 35.1; f32 sum across panels (host emulation) 1.075 / 4.65; f32
/// storage with an f64 GEMM 0.049 / 0.054.
///
/// MUTATION CHECK of the device kernel (temporarily applied to `mixed.rs`:
/// `beta = 1` after the first panel and one flush after the last, i.e. the
/// panels summed in f32): ratio 4.70 deep, outside `hi` = 2.16, so this test
/// fails; at the XC shape it reads 1.14, inside the band: with k = NB <= 128
/// (two panels) a lost flush adds one f32 rounding and no RMS band can see it.
/// That defect is pinned deterministically by
/// `the_f64_flush_across_panels_keeps_what_an_f32_sum_loses_in_every_matrix`
/// (which the same mutant fails; so does the per-element bound, but only in its
/// b = 1 case, 129 one-element panels: max e/bound 1.23). At the XC shape this
/// band catches a one-panel (plain SGEMM) kernel and a doubled panel width
/// (mutant `kb = 2·k_panel`: ratio 2.07 against `hi` = 1.44, which also fails
/// the panel-count pin and the flush pin).
/// The ratios are device- and library-dependent; re-derive on other hardware
/// by reading this test's `--nocapture` output.
#[test]
fn the_batched_kernel_sits_in_the_twins_rms_band_set_by_measured_mutants() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    // (m, n, k, w, batch, lost flush visible?)
    for &(m, n, k, w, batch, deep) in &[
        (64usize, 64usize, 128usize, 512usize, 64usize, false),
        (32, 32, 2048, 2048, 16, true),
    ] {
        let bt = Batch::random(m, n, k, w, batch, 977, false);
        let r = rms_table(&bt, 64);
        let t = r[2];
        let [dev, lp, _, one, xp, sto, twin_one] = r.map(|x| x / t);
        eprintln!(
            "{m}x{k}x{n} batch {batch}: RMS(err/S) device batched {:.3e}, twin {t:.3e} | ratios to the twin: batched {dev:.3}, per-matrix device {lp:.3}, plain SGEMM {one:.3} (twin {twin_one:.3}), f32 across panels {xp:.3}, f32 storage only {sto:.3}",
            r[0]
        );
        assert!(
            one > 1.0 && xp > 1.0 && sto < 1.0,
            "a defect is not on its side of the clean ratio"
        );
        let lo = sto.sqrt();
        let hi = if deep { one.min(xp) } else { one }.sqrt();
        assert!(
            lo < dev && dev < hi,
            "{m}x{k}x{n}: batched/twin RMS ratio {dev:.3} outside ({lo:.3}, {hi:.3})"
        );
    }
}
