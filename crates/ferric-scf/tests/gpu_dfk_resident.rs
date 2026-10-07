#![cfg(feature = "gpu")]
//! Device DF-K occupied path vs the CPU path of the SAME `DfK` (same dressed B).
//!
//! The gate is derived, not tuned. With A_Pμi = (|B_P||C|)_μi and
//! S_μν = Σ_{P,i} A_Pμi A_Pνi, a path with stage-1 depth n and stage-2 depth
//! k_chunk + nchunks satisfies |K̂ − K| ≤ ε·S, ε = 2γ_n + γ_n² +
//! γ_{k_chunk+nchunks}(1+γ_n)² (u = 2⁻⁵³; module docs of `df_k_gpu`). Two
//! independently rounded K's therefore differ by ≤ (ε_dev + ε_cpu)·S; the CPU
//! side uses k_chunk = c_cpu·nocc with c_cpu = clamp(4096/n, 4, 64) and the safe
//! upper bound nchunks ≤ naux. A 1% allowance covers S's own rounding.
//!
//! Two measured sides (printed on every run; only the inequalities are asserted).
//! Defect absent: device-vs-CPU max |dK| / bound was 2e-3 (benzene/def2-SVP, nocc 21),
//! 6e-3 (water/cc-pVDZ), 9e-3 and 5e-3 (OH alpha and beta), 7e-3 (38 chunks vs 1),
//! 2e-3 (two-band sum). Defect present: B rounded through f32 gave 6e3 (benzene)
//! and 2e5 (water); a dropped last aux row gave 7e8 and 5e10. The bound is a
//! worst case, so the clean side sits two orders below it; the defect side sits
//! three to ten orders above it. Both mutants are ALSO asserted to exceed the
//! bound, so the gate cannot go blind. The same two defects were applied to the
//! production dispatcher (the `ROUND_B_TO_F32` seam forced on; the last row
//! dropped in `upload_band`) and this file's bound test failed for each.
//!
//! Source mutations of `df_k_gpu.rs` / `df_k.rs`, each applied alone and each made
//! at least one test of this file fail (restored afterwards): mirror lower to
//! upper instead of upper to lower (4 fail); SYRK beta always 1 (the alpha/beta
//! reuse test and the failure test fail; a single fresh build cannot see it);
//! every chunk reading aux rows 0..c (only the many-chunk test fails); no sticky
//! decline after a failed build (the once-only test fails); skipping the C_occ
//! upload when nocc equals the previous build's (the warm-build loop, fresh
//! coefficients at a fixed nocc, fails at ratio 3e12, and so does alpha-prime);
//! the DF-K hook moved
//! above the FORCE_DENSITY and nocc == 0 shortcuts (those two tests fail).
//!
//! What is pinned about reproducibility: the device K of one build is bit-identical
//! on rebuild within a process (alpha/beta test; 20 repeats identical on each of
//! two cards) and independent of RAYON_NUM_THREADS (child processes at 1 and 4
//! workers). It is NOT bit-identical to the CPU K (different summation order); the
//! contract against the CPU is the derived bound above.
//!
//! Also pins: exact bytes moved per build (8·n·nocc up, 8·n² down, B once);
//! run-to-run bit-identity of the device K; α/β reuse; FERRIC_DFK_FORCE_DENSITY
//! never reaches the card; an injected CUDA failure falls back once and stays;
//! a two-band sum under simulated ranks; independence from RAYON_NUM_THREADS.
use ferric_core::basis;
use ferric_core::gpu::{install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_core::parallel::{aux_band_for, ParallelContext};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_k::DfK;
use ferric_scf::df_k_gpu::{k_error_factor, DeviceDfK, FORCE_BUILD_FAILURE, FORCE_HOST};
use ferric_scf::fock::KBuilder;
use ndarray::{Array2, ArrayView2};
use std::sync::atomic::Ordering;
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());
/// Serializes the tests that set FERRIC_DFK_FORCE_DENSITY (process-global env).
static ENV: Mutex<()> = Mutex::new(());

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
            memory_gb: Some(4.0),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

fn dfk_for(xyz: &str, mult: usize, obs_name: &str) -> (DfK<'static>, usize) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let n = obs.nbasis();
    (
        DfK::new(Operator::coulomb(), &obs, &aux, usize::MAX).unwrap(),
        n,
    )
}

const WATER: &str = "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n";
const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

/// Deterministic mixed-sign occupied block (need not be orthonormal: K[C] is
/// the same algebra for any C).
fn c_occ(n: usize, nocc: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, nocc), |(mu, i)| {
        0.05 * (((mu * 5 + i * 3 + salt * 7) % 17) as f64 - 8.0)
    })
}

/// S_μν = Σ_P Σ_i (|B_P||C|)_μi (|B_P||C|)_νi from the flat dressed band.
fn s_matrix(flat: &ArrayView2<f64>, n: usize, c: &Array2<f64>) -> Array2<f64> {
    let abs_c = c.mapv(f64::abs);
    let mut s = Array2::<f64>::zeros((n, n));
    let all = flat.as_slice().expect("incore_flat is standard layout");
    for p in 0..flat.nrows() {
        let bp = ArrayView2::from_shape((n, n), &all[p * n * n..(p + 1) * n * n]).unwrap();
        let a = bp.mapv(f64::abs).dot(&abs_c);
        s += &a.dot(&a.t());
    }
    s
}

/// max over the upper triangle of |a − b| / (eps·S·1.01); asserts nothing.
fn max_ratio(a: &Array2<f64>, b: &Array2<f64>, s: &Array2<f64>, eps: f64) -> f64 {
    let n = a.nrows();
    let mut worst = 0.0f64;
    for r in 0..n {
        for c in r..n {
            let bound = eps * s[(r, c)] * 1.01 + f64::MIN_POSITIVE;
            worst = worst.max((a[(r, c)] - b[(r, c)]).abs() / bound);
        }
    }
    worst
}

fn cpu_k(dfk: &mut DfK, c: &Array2<f64>) -> Array2<f64> {
    FORCE_HOST.store(true, Ordering::SeqCst);
    let n = c.nrows();
    let mut k = Array2::zeros((n, n));
    dfk.build_from_occ(c, &mut k).unwrap();
    FORCE_HOST.store(false, Ordering::SeqCst);
    k
}

fn eps_cpu(n: usize, nocc: usize, naux: usize) -> f64 {
    k_error_factor(n, (4096 / n.max(1)).clamp(4, 64) * nocc, naux)
}

#[test]
fn k_matches_cpu_within_the_two_stage_bound_and_both_mutants_exceed_it() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    for (label, xyz, obs_name, nocc) in [
        ("water/cc-pVDZ", WATER.to_string(), "cc-pvdz", 5usize),
        (
            "benzene/def2-SVP",
            std::fs::read_to_string("../../testdata/molecules/benzene.xyz").unwrap(),
            "def2-svp",
            21,
        ),
    ] {
        let (mut dfk, n) = dfk_for(&xyz, 1, obs_name);
        let c = c_occ(n, nocc, 1);
        let flat = dfk
            .dressed_incore_flat_for_test()
            .expect("in core")
            .to_owned();
        let (naux, _) = flat.dim();
        let s = s_matrix(&flat.view(), n, &c);
        let k_cpu = cpu_k(&mut dfk, &c);

        let s0 = stats();
        let mut k_dev = Array2::zeros((n, n));
        dfk.build_from_occ(&c, &mut k_dev).unwrap();
        assert_eq!(
            stats().dfk_device_builds,
            s0.dfk_device_builds + 1,
            "{label}: the device path did not run"
        );

        // chunk plan of the production call: one chunk covers the whole band here
        let eps_dev = k_error_factor(n, naux * nocc, 1);
        let ratio = max_ratio(&k_dev, &k_cpu, &s, eps_dev + eps_cpu(n, nocc, naux));
        eprintln!("{label}: n={n} naux={naux} nocc={nocc}: max |K_dev-K_cpu|/bound = {ratio:.3e}");
        assert!(
            ratio <= 1.0,
            "{label}: device K outside the derived bound ({ratio:e})"
        );
        // symmetric by construction
        for r in 0..n {
            for cc in 0..r {
                assert_eq!(k_dev[(r, cc)], k_dev[(cc, r)]);
            }
        }

        // MUTANT 1: B rounded through f32 (the nearest subtle defect)
        let dev = ferric_core::gpu::device::device(0).unwrap();
        let rounded = flat.mapv(|x| x as f32 as f64);
        let mut m1 = DeviceDfK::upload(&dev, &pool().unwrap(), &rounded.view(), n).unwrap();
        let mut k1 = Array2::zeros((n, n));
        m1.build_from_occ(&c.view(), &mut k1).unwrap();
        let r1 = max_ratio(&k1, &k_cpu, &s, eps_dev + eps_cpu(n, nocc, naux));
        // MUTANT 2: the last aux row dropped
        let dropped = flat.slice(ndarray::s![..naux - 1, ..]).to_owned();
        let mut m2 = DeviceDfK::upload(&dev, &pool().unwrap(), &dropped.view(), n).unwrap();
        let mut k2 = Array2::zeros((n, n));
        m2.build_from_occ(&c.view(), &mut k2).unwrap();
        let r2 = max_ratio(&k2, &k_cpu, &s, eps_dev + eps_cpu(n, nocc, naux));
        eprintln!("{label}: f32-rounded B ratio {r1:.3e}; dropped aux row ratio {r2:.3e}");
        assert!(
            r1 > 1.0,
            "{label}: the f32 mutant must exceed the bound, got {r1:e}"
        );
        assert!(
            r2 > 1.0,
            "{label}: the dropped-row mutant must exceed the bound, got {r2:e}"
        );
    }
}

#[test]
fn b_is_uploaded_once_and_each_build_moves_exactly_the_documented_bytes() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut dfk, n) = dfk_for(WATER, 1, "cc-pvdz");
    let nocc = 5;
    let c = c_occ(n, nocc, 2);
    let band = dfk.dressed_incore_flat_for_test().unwrap().nrows();
    let mut k = Array2::zeros((n, n));
    let a = stats();
    dfk.build_from_occ(&c, &mut k).unwrap(); // cold: uploads B
    let b = stats();
    assert_eq!(b.resident_uploads, a.resident_uploads + 1);
    assert_eq!(
        b.bytes_h2d - a.bytes_h2d,
        (8 * band * n * n + 8 * n * nocc) as u64
    );
    assert_eq!(b.bytes_d2h - a.bytes_d2h, (8 * n * n) as u64);
    // Warm builds refresh C_occ at a FIXED nocc, as every SCF iteration does: a new
    // salt each time, and EACH build is checked against the CPU K.
    let flat = dfk.dressed_incore_flat_for_test().unwrap().to_owned();
    let mut warm = Vec::new();
    for i in 0..3 {
        let ci = c_occ(n, nocc, 10 + i);
        dfk.build_from_occ(&ci, &mut k).unwrap();
        warm.push((ci, k.clone()));
    }
    let w = stats();
    for (i, (ci, ki)) in warm.iter().enumerate() {
        let s = s_matrix(&flat.view(), n, ci);
        let k_cpu = cpu_k(&mut dfk, ci);
        let eps = k_error_factor(n, band * nocc, 1) + eps_cpu(n, nocc, band);
        let ratio = max_ratio(ki, &k_cpu, &s, eps);
        eprintln!("warm build {i}: ratio {ratio:.3e}");
        assert!(
            ratio <= 1.0,
            "warm build {i} (fresh C_occ, same nocc): {ratio:e}"
        );
    }
    assert_eq!(w.resident_uploads, b.resident_uploads, "B was re-uploaded");
    assert_eq!(w.bytes_h2d - b.bytes_h2d, 3 * (8 * n * nocc) as u64);
    assert_eq!(w.bytes_d2h - b.bytes_d2h, 3 * (8 * n * n) as u64);
    assert_eq!(w.dfk_device_builds, a.dfk_device_builds + 4);
    // the reservation is labelled
    assert!(pool()
        .unwrap()
        .occupancy_report()
        .contains("DF-K dressed B"));
}

#[test]
fn alpha_beta_reuse_does_not_accumulate_and_is_run_to_run_identical() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut dfk, n) = dfk_for(OH, 2, "cc-pvdz");
    let (ca, cb) = (c_occ(n, 5, 3), c_occ(n, 4, 4)); // α 5, β 4: the scratch is sized by α and reused
    let flat = dfk.dressed_incore_flat_for_test().unwrap().to_owned();
    let naux = flat.nrows();
    let (mut ka1, mut kb, mut ka2) = (
        Array2::zeros((n, n)),
        Array2::zeros((n, n)),
        Array2::zeros((n, n)),
    );
    dfk.build_from_occ(&ca, &mut ka1).unwrap();
    dfk.build_from_occ(&cb, &mut kb).unwrap();
    dfk.build_from_occ(&ca, &mut ka2).unwrap();
    let bits = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    // Measured: 20 repeats identical on each of two cards;
    // the claim is between two runs of the same path on one device, nothing more.
    assert_eq!(
        bits(&ka1),
        bits(&ka2),
        "α rebuilt after β differs: stale accumulator or non-determinism"
    );
    // alpha' = same nocc as alpha, different coefficients: the upload must be refreshed
    let ca2 = c_occ(n, 5, 11);
    let mut ka3 = Array2::zeros((n, n));
    dfk.build_from_occ(&ca2, &mut ka3).unwrap();
    assert_ne!(
        bits(&ka3),
        bits(&ka1),
        "alpha' gave alpha's K: a stale C_occ upload"
    );
    for (label, c, k) in [
        ("alpha", &ca, &ka1),
        ("beta", &cb, &kb),
        ("alpha-prime", &ca2, &ka3),
    ] {
        let nocc = c.ncols();
        let s = s_matrix(&flat.view(), n, c);
        let k_cpu = cpu_k(&mut dfk, c);
        let ratio = max_ratio(
            k,
            &k_cpu,
            &s,
            k_error_factor(n, naux * nocc, 1) + eps_cpu(n, nocc, naux),
        );
        eprintln!("{label}: ratio {ratio:.3e}");
        assert!(ratio <= 1.0, "{label}: {ratio:e}");
    }
}

#[test]
fn chunking_by_pool_budget_changes_only_the_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (dfk, n) = dfk_for(WATER, 1, "cc-pvdz");
    let (nocc, flat) = (
        5usize,
        dfk.dressed_incore_flat_for_test().unwrap().to_owned(),
    );
    let naux = flat.nrows();
    let c = c_occ(n, nocc, 5);
    let s = s_matrix(&flat.view(), n, &c);
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let big = pool().expect("installed pool");
    let mut whole = DeviceDfK::upload(&dev, &big, &flat.view(), n).unwrap();
    let mut tiny = DeviceDfK::upload(&dev, &big, &flat.view(), n)
        .unwrap()
        .with_scratch_bytes(3 * 8 * nocc * n + 5);
    let (mut k1, mut k2) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    let p1 = whole.build_from_occ(&c.view(), &mut k1).unwrap();
    let p2 = tiny.build_from_occ(&c.view(), &mut k2).unwrap();
    assert_eq!(p1.nchunks, 1);
    assert_eq!(p2.chunk, 3, "scratch for exactly 3 aux rows");
    assert_eq!(p2.nchunks, naux.div_ceil(3));
    let eps = k_error_factor(n, p1.chunk * nocc, p1.nchunks)
        + k_error_factor(n, p2.chunk * nocc, p2.nchunks);
    let ratio = max_ratio(&k1, &k2, &s, eps);
    eprintln!("1 chunk vs {} chunks: ratio {ratio:.3e}", p2.nchunks);
    assert!(ratio <= 1.0, "{ratio:e}");
}

#[test]
fn force_density_never_reaches_the_card() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let _e = ENV.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut dfk, n) = dfk_for(WATER, 1, "cc-pvdz");
    let c = c_occ(n, 5, 6);
    let a = stats();
    std::env::set_var("FERRIC_DFK_FORCE_DENSITY", "1");
    let mut k = Array2::zeros((n, n));
    dfk.build_from_occ(&c, &mut k).unwrap();
    std::env::remove_var("FERRIC_DFK_FORCE_DENSITY");
    let b = stats();
    assert_eq!(b.dfk_device_builds, a.dfk_device_builds);
    assert_eq!(
        b.resident_uploads, a.resident_uploads,
        "the density path must not upload B"
    );
    assert_eq!(b.bytes_h2d, a.bytes_h2d);
}

#[test]
fn device_build_with_zero_occupied_columns_is_exactly_zero() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut dfk, n) = dfk_for(WATER, 1, "cc-pvdz");
    let a = stats();
    let mut k = Array2::from_elem((n, n), 1.0);
    dfk.build_from_occ(&Array2::zeros((n, 0)), &mut k).unwrap();
    assert!(k.iter().all(|&v| v == 0.0));
    assert_eq!(
        stats().resident_uploads,
        a.resident_uploads,
        "an empty channel must not touch the card"
    );
}

#[test]
fn build_failure_after_upload_falls_back_once_and_stays_on_cpu() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut dfk, n) = dfk_for(WATER, 1, "cc-pvdz");
    let c = c_occ(n, 5, 7);
    let k_cpu = cpu_k(&mut dfk, &c);
    let cap = pool().unwrap().capacity_bytes();
    let before = pool().unwrap().available_bytes();
    let a = stats();
    FORCE_BUILD_FAILURE.store(true, Ordering::SeqCst);
    let mut k = Array2::zeros((n, n));
    dfk.build_from_occ(&c, &mut k).unwrap();
    FORCE_BUILD_FAILURE.store(false, Ordering::SeqCst);
    let b = stats();
    assert_eq!(b.gemm_cpu_cuda_error, a.gemm_cpu_cuda_error + 1);
    assert_eq!(b.dfk_declined, a.dfk_declined + 1);
    assert_eq!(b.dfk_device_builds, a.dfk_device_builds);
    assert_eq!(
        k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        k_cpu.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "fallback K must be the CPU K bit for bit"
    );
    // the decline released the resident tensor immediately (no leak), and is sticky
    assert_eq!(
        pool().unwrap().available_bytes(),
        before,
        "device memory not returned (capacity {cap})"
    );
    dfk.build_from_occ(&c, &mut k).unwrap();
    assert_eq!(
        stats().dfk_device_builds,
        a.dfk_device_builds,
        "a declined DfK must not retry the device"
    );
    assert_eq!(
        stats().gemm_cpu_cuda_error,
        b.gemm_cpu_cuda_error,
        "the failure is counted once, not per iteration"
    );

    // The device API itself: a failure after all chunk work leaves the caller's K untouched.
    let flat = dfk.dressed_incore_flat_for_test().unwrap().to_owned();
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let mut raw = DeviceDfK::upload(&dev, &pool().unwrap(), &flat.view(), n).unwrap();
    let mut sentinel = Array2::from_elem((n, n), 7.0);
    FORCE_BUILD_FAILURE.store(true, Ordering::SeqCst);
    let r = raw.build_from_occ(&c.view(), &mut sentinel);
    FORCE_BUILD_FAILURE.store(false, Ordering::SeqCst);
    assert!(r.is_err());
    assert!(
        sentinel.iter().all(|&v| v == 7.0),
        "a failed device build wrote into K"
    );
    // and the object is still usable afterwards (the accumulator is rebuilt with beta = 0)
    let mut again = Array2::zeros((n, n));
    raw.build_from_occ(&c.view(), &mut again).unwrap();
    let s = s_matrix(&flat.view(), n, &c);
    assert!(
        max_ratio(
            &again,
            &k_cpu,
            &s,
            k_error_factor(n, flat.nrows() * 5, 1) + eps_cpu(n, 5, flat.nrows())
        ) <= 1.0
    );
}

#[test]
fn two_simulated_ranks_upload_only_their_band_and_their_sum_is_the_full_k() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let (n, naux, nocc) = (obs.nbasis(), aux.nbasis(), 5usize);
    let op = Operator::coulomb();
    let c = c_occ(n, nocc, 8);
    let mut full = DfK::new(op, &obs, &aux, usize::MAX).unwrap();
    let flat = full.dressed_incore_flat_for_test().unwrap().to_owned();
    let s = s_matrix(&flat.view(), n, &c);
    let k_cpu = cpu_k(&mut full, &c);
    let mut sum = Array2::<f64>::zeros((n, n));
    let mut nch = 0usize;
    for rank in 0..2 {
        let ctx = ParallelContext::for_rank(rank, 2);
        let mut part = DfK::new_banded(op, &obs, &aux, usize::MAX, Some(&ctx)).unwrap();
        let (p0, p1) = aux_band_for(naux, rank, 2);
        let band_rows = part.dressed_incore_flat_for_test().unwrap().nrows();
        assert_eq!(band_rows, p1 - p0, "rank {rank} must hold only its band");
        let a = stats();
        let mut kp = Array2::zeros((n, n));
        part.build_from_occ(&c, &mut kp).unwrap();
        let b = stats();
        assert_eq!(
            b.dfk_device_builds,
            a.dfk_device_builds + 1,
            "rank {rank} did not use the device"
        );
        assert_eq!(
            b.bytes_h2d - a.bytes_h2d,
            (8 * band_rows * n * n + 8 * n * nocc) as u64,
            "rank {rank} uploaded more than its band"
        );
        nch += 1;
        sum += &kp;
    }
    // two partials summed: one extra accumulation, so depth k_chunk + (nchunks_0 + nchunks_1)
    let eps = k_error_factor(n, (naux.div_ceil(2) + 1) * nocc, nch) + eps_cpu(n, nocc, naux);
    let ratio = max_ratio(&sum, &k_cpu, &s, eps);
    eprintln!("two-band sum: ratio {ratio:.3e}");
    assert!(ratio <= 1.0, "{ratio:e}");
}

fn fnv(k: &Array2<f64>) -> u64 {
    k.iter().fold(0xcbf29ce484222325u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(0x100000001b3)
    })
}

/// Child half of the thread-independence test: print the rayon width and a hash of K.
#[test]
fn device_k_checksum_child() {
    if std::env::var_os("FERRIC_DFK_CHILD").is_none() {
        return;
    }
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        println!("CHILD skipped");
        return;
    }
    let (mut dfk, n) = dfk_for(WATER, 1, "cc-pvdz");
    let mut k = Array2::zeros((n, n));
    dfk.build_from_occ(&c_occ(n, 5, 9), &mut k).unwrap();
    println!(
        "CHILD threads={} hash={:016x}",
        rayon::current_num_threads(),
        fnv(&k)
    );
}

#[test]
fn device_k_is_independent_of_the_rayon_worker_count() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let run = |threads: &str| {
        let out = std::process::Command::new(&exe)
            .args([
                "--exact",
                "device_k_checksum_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("FERRIC_DFK_CHILD", "1")
            .env("RAYON_NUM_THREADS", threads)
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("FERRIC_GPU_TESTS_REQUIRED", "1")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        // libtest prints "test <name> ... " before the test body's own output, so the
        // result sits mid-line.
        let line = text
            .lines()
            .find_map(|l| l.find("CHILD threads=").map(|i| &l[i..]))
            .unwrap_or_else(|| panic!("child printed no result: {text}"));
        let parts: Vec<&str> = line.split_whitespace().collect();
        (parts[1].to_string(), parts[2].to_string())
    };
    let (t1, h1) = run("1");
    let (t4, h4) = run("4");
    assert_ne!(
        t1, t4,
        "the two arms ran with the same worker count ({t1}); the test would be vacuous"
    );
    assert_eq!(
        h1, h4,
        "device K depends on RAYON_NUM_THREADS ({t1} vs {t4})"
    );
}
