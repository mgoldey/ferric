#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! Device-resident RI-J (`df_j_gpu`): agreement with the host passes to the
//! derived GEMV bounds, the counters (one upload per geometry, per-build bytes),
//! every decline to the host, and zero-size safety. Skips cleanly without a CUDA
//! device (`FERRIC_GPU_TESTS_REQUIRED=1` turns the skip into a failure).
use ferric_core::basis;
use ferric_core::gpu::device::{device, GpuError};
use ferric_core::gpu::{install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::DfJ;
use ferric_scf::df_j_gpu::{
    d_error_factor, j_error_factor, pair_len, resident_bytes, row_plan, DeviceDfJ,
    FORCE_BUILD_FAILURE, FORCE_HOST,
};
use ferric_scf::fock::JBuilder;
use ndarray::{Array2, ArrayView2};
use std::sync::atomic::Ordering;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ready() -> bool {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
        );
        return false;
    }
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            memory_gb: Some(1.0),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

fn alkane(nc: usize) -> Molecule {
    let mut xyz = String::new();
    let mut count = 0;
    for i in 0..nc {
        let x = 1.27 * i as f64;
        let y = if i % 2 == 0 { 0.0 } else { 0.5 };
        let s = if i % 2 == 0 { -1.0 } else { 1.0 };
        xyz.push_str(&format!(
            "C {x} {y} 0\nH {x} {} 0.9\nH {x} {} -0.9\n",
            y + s * 0.65,
            y + s * 0.65
        ));
        count += 3;
    }
    xyz.push_str("H -1 0 0\n");
    xyz.push_str(&format!(
        "H {} {} 0\n",
        1.27 * (nc - 1) as f64 + 1.0,
        0.5 * ((nc - 1) % 2) as f64
    ));
    count += 2;
    Molecule::parse_xyz(&format!("{count}\nalkane\n{xyz}"), 0, 1).unwrap()
}

struct Fixture {
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    n: usize,
    naux: usize,
}

fn fixture() -> Fixture {
    let mol = alkane(3);
    let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let (n, naux) = (obs.nbasis(), dfbs.nbasis());
    Fixture { obs, dfbs, n, naux }
}

impl Fixture {
    fn packed_budget(&self) -> usize {
        let unpacked = self.naux * self.n * self.n * 8;
        let packed = self.naux * (self.n * (self.n + 1) / 2) * 8;
        (unpacked + packed) / 2
    }
    fn dfj(&self, budget: usize) -> DfJ<'static> {
        DfJ::new(Operator::coulomb(), &self.obs, &self.dfbs, budget).unwrap()
    }
}

fn density(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| 0.01 * ((i * 7 + j * 3) % 11) as f64)
}

fn bits(m: &Array2<f64>) -> Vec<u64> {
    m.iter().map(|v| v.to_bits()).collect()
}

fn host_build(dfj: &mut DfJ, d: &Array2<f64>, n: usize) -> Array2<f64> {
    FORCE_HOST.store(true, Ordering::SeqCst);
    let mut j = Array2::zeros((n, n));
    dfj.build(d, &mut j).unwrap();
    FORCE_HOST.store(false, Ordering::SeqCst);
    j
}

fn weights(d: &Array2<f64>, n: usize) -> Vec<f64> {
    let mut w = Vec::new();
    for mu in 0..n {
        for nu in 0..mu {
            w.push(d[(mu, nu)] + d[(nu, mu)]);
        }
        w.push(d[(mu, mu)]);
    }
    w
}

/// Device passes against the host passes on the same packed tensor. Both sides
/// are classical GEMVs, so each is within its own `gamma_k (|A||x|)` of the exact
/// value and the difference is within the SUM of the two factors times `|A||x|`
/// (the host dot product has depth `pair`, the host reduction depth `naux`).
#[test]
fn device_passes_agree_with_the_host_to_the_derived_bounds() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let f = fixture();
    let dfj = f.dfj(f.packed_budget());
    let bp: ArrayView2<f64> = dfj.packed_incore_for_test().expect("packed tier");
    let (naux, pair) = bp.dim();
    assert_eq!(pair, pair_len(f.n).unwrap());
    let dev = device(0).unwrap();
    let mut dd = DeviceDfJ::upload_packed(&dev, &pool().unwrap(), &bp).unwrap();
    let w = weights(&density(f.n), f.n);
    let d_dev = dd.d_p(&w).unwrap();
    let eps_d = d_error_factor(pair) + d_error_factor(pair);
    for p in 0..naux {
        let (mut s, mut sabs) = (0.0f64, 0.0f64);
        for k in 0..pair {
            s += bp[(p, k)] * w[k];
            sabs += (bp[(p, k)] * w[k]).abs();
        }
        assert!(
            (d_dev[p] - s).abs() <= eps_d * sabs,
            "d_P[{p}]: {} vs {s}, bound {}",
            d_dev[p],
            eps_d * sabs
        );
    }
    let c: Vec<f64> = (0..naux)
        .map(|p| ((p * 13) % 17) as f64 / 17.0 - 0.4)
        .collect();
    let j_dev = dd.j_packed(&c).unwrap();
    let (rows, calls) = row_plan(naux, pair);
    let eps_j = j_error_factor(rows, calls) + j_error_factor(naux, 0);
    for k in 0..pair {
        let (mut s, mut sabs) = (0.0f64, 0.0f64);
        for p in 0..naux {
            s += bp[(p, k)] * c[p];
            sabs += (bp[(p, k)] * c[p]).abs();
        }
        assert!(
            (j_dev[k] - s).abs() <= eps_j * sabs,
            "J[{k}]: {} vs {s}, bound {}",
            j_dev[k],
            eps_j * sabs
        );
    }
}

/// End to end through `DfJ::build` on all three host tiers. The two J's differ by
/// the pass-1 error propagated through the metric solve, whose amplification is
/// cond(V)-dependent and not bounded here; this checks wiring (right tensor, right
/// unpack) at a scale far above the rounding bound and far below any wiring error.
#[test]
fn dfj_build_on_the_device_matches_the_host_on_every_tier() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let f = fixture();
    let d = density(f.n);
    let tiny = f.n * f.n * 8 * 40;
    for (name, budget) in [
        ("unpacked in-core", usize::MAX),
        ("packed in-core", f.packed_budget()),
        ("spilled", tiny),
    ] {
        let mut dfj = f.dfj(budget);
        let j_host = host_build(&mut dfj, &d, f.n);
        let a = stats();
        let mut j = Array2::zeros((f.n, f.n));
        dfj.build(&d, &mut j).unwrap();
        let b = stats();
        assert_eq!(
            b.dfj_device_builds,
            a.dfj_device_builds + 1,
            "{name}: not on the device"
        );
        assert!(dfj.device_resident_for_test());
        assert_eq!(j, j.t(), "{name}: device J must be exactly symmetric");
        let scale = j_host.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let diff = (&j - &j_host).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            diff <= 1e-9 * scale,
            "{name}: max|dJ| {diff:e} scale {scale:e}"
        );
    }
}

#[test]
fn the_tensor_goes_up_once_and_each_build_moves_the_derived_bytes() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let f = fixture();
    let d = density(f.n);
    let mut dfj = f.dfj(f.packed_budget());
    let pair = pair_len(f.n).unwrap();
    let per_build = 8 * (pair + f.naux);
    let (mut j1, mut j2) = (Array2::zeros((f.n, f.n)), Array2::zeros((f.n, f.n)));
    let a = stats();
    dfj.build(&d, &mut j1).unwrap();
    let b = stats();
    assert_eq!(b.resident_uploads, a.resident_uploads + 1);
    assert_eq!(
        (b.bytes_h2d - a.bytes_h2d) as usize,
        8 * f.naux * pair + per_build,
        "cold build: tensor once plus w and c"
    );
    assert_eq!((b.bytes_d2h - a.bytes_d2h) as usize, per_build);
    for _ in 0..3 {
        dfj.build(&d, &mut j2).unwrap();
    }
    let c = stats();
    assert_eq!(
        c.resident_uploads, b.resident_uploads,
        "the tensor must not be re-uploaded"
    );
    assert_eq!(c.dfj_device_builds, b.dfj_device_builds + 3);
    assert_eq!((c.bytes_h2d - b.bytes_h2d) as usize, 3 * per_build);
    assert_eq!((c.bytes_d2h - b.bytes_d2h) as usize, 3 * per_build);
    assert_eq!(
        bits(&j1),
        bits(&j2),
        "the device build is deterministic run to run"
    );
    drop(dfj);
    assert_eq!(
        pool().unwrap().available_bytes(),
        pool().unwrap().capacity_bytes()
    );
}

#[test]
fn force_host_is_the_host_path_bit_for_bit_and_touches_no_counter() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let f = fixture();
    let d = density(f.n);
    let mut dfj = f.dfj(f.packed_budget());
    let a = stats();
    let j1 = host_build(&mut dfj, &d, f.n);
    let j2 = host_build(&mut dfj, &d, f.n);
    assert_eq!(stats(), a, "FORCE_HOST must not move any counter");
    assert!(!dfj.device_resident_for_test());
    assert_eq!(bits(&j1), bits(&j2));
    // and it equals the host path of a DfJ that was declined (the same code)
    let hog = pool()
        .unwrap()
        .reserve("test hog", pool().unwrap().available_bytes())
        .unwrap();
    let mut declined = f.dfj(f.packed_budget());
    let mut j3 = Array2::zeros((f.n, f.n));
    declined.build(&d, &mut j3).unwrap();
    drop(hog);
    assert_eq!(bits(&j1), bits(&j3));
}

#[test]
fn a_pool_too_small_declines_once_runs_the_host_and_leaks_nothing() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let f = fixture();
    let d = density(f.n);
    let mut dfj = f.dfj(f.packed_budget());
    let j_host = host_build(&mut dfj, &d, f.n);
    let cap = pool().unwrap().capacity_bytes();
    assert_eq!(
        pool().unwrap().available_bytes(),
        cap,
        "another test left a reservation"
    );
    let pair = pair_len(f.n).unwrap();
    let need = resident_bytes(f.naux, pair).unwrap();
    // leave room for the small vectors but one byte short of the tensor
    let hog = pool()
        .unwrap()
        .reserve("test hog", cap - (need - 8 * f.naux * pair) - 1)
        .unwrap();
    let a = stats();
    let mut j = Array2::zeros((f.n, f.n));
    for _ in 0..3 {
        dfj.build(&d, &mut j).unwrap();
        assert_eq!(
            bits(&j),
            bits(&j_host),
            "a declined build is the host build"
        );
    }
    let b = stats();
    assert_eq!(b.dfj_declined, a.dfj_declined + 1, "declined once, sticky");
    assert_eq!(b.gemm_cpu_pool_full, a.gemm_cpu_pool_full + 1);
    assert_eq!(b.dfj_device_builds, a.dfj_device_builds);
    assert_eq!(b.resident_uploads, a.resident_uploads);
    assert_eq!(
        b.bytes_h2d, a.bytes_h2d,
        "a refused reservation moves no bytes"
    );
    drop(hog);
    assert_eq!(
        pool().unwrap().available_bytes(),
        cap,
        "the decline leaked a reservation"
    );
}

#[test]
fn a_failure_after_pass_one_declines_to_the_host_and_frees_the_device() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let f = fixture();
    let d = density(f.n);
    let mut dfj = f.dfj(f.packed_budget());
    let j_host = host_build(&mut dfj, &d, f.n);
    let a = stats();
    FORCE_BUILD_FAILURE.store(true, Ordering::SeqCst);
    let mut j = Array2::zeros((f.n, f.n));
    let r = dfj.build(&d, &mut j);
    FORCE_BUILD_FAILURE.store(false, Ordering::SeqCst);
    r.unwrap();
    let b = stats();
    assert_eq!(b.dfj_declined, a.dfj_declined + 1);
    assert_eq!(b.dfj_device_builds, a.dfj_device_builds);
    assert_eq!(bits(&j), bits(&j_host));
    assert!(!dfj.device_resident_for_test());
    assert_eq!(
        pool().unwrap().available_bytes(),
        pool().unwrap().capacity_bytes()
    );
}

#[test]
fn zero_sizes_and_overflow_are_typed_not_panics() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    for (naux, pair) in [(0usize, 0usize), (0, 6), (4, 0)] {
        let empty = Array2::<f64>::zeros((naux, pair));
        let mut dd = DeviceDfJ::upload_packed(&dev, &pool, &empty.view()).unwrap();
        assert_eq!(dd.d_p(&vec![0.0; pair]).unwrap(), vec![0.0; naux]);
        assert_eq!(dd.j_packed(&vec![0.0; naux]).unwrap(), vec![0.0; pair]);
    }
    assert_eq!(pair_len(0).unwrap(), 0);
    assert!(matches!(pair_len(usize::MAX), Err(GpuError::Layout(_))));
    assert!(matches!(
        resident_bytes(usize::MAX, 2),
        Err(GpuError::Layout(_))
    ));
    assert_eq!(row_plan(0, 5), (0, 0));
    assert_eq!(row_plan(5, 0), (0, 0));
    // wrong-length operands and out-of-order uploads are typed refusals
    let t = Array2::<f64>::ones((3, 6));
    let mut dd = DeviceDfJ::upload_packed(&dev, &pool, &t.view()).unwrap();
    assert!(matches!(dd.d_p(&[0.0; 5]), Err(GpuError::Layout(_))));
    assert!(matches!(dd.j_packed(&[0.0; 4]), Err(GpuError::Layout(_))));
    let mut half = DeviceDfJ::reserve(&dev, &pool, 3, 6).unwrap();
    assert!(matches!(
        half.write_rows(1, &[0.0; 6]),
        Err(GpuError::Layout(_))
    ));
    assert!(matches!(half.finish_upload(), Err(GpuError::Layout(_))));
    drop((dd, half));
    assert_eq!(pool.available_bytes(), pool.capacity_bytes());
}
