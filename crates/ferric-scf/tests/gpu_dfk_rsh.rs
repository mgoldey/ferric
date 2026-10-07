#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! Resident device DF-K under a range-separated hybrid (two fitters, one `DfK` for
//! `erfc(0.3)/r` and one for `erf(0.3)/r`, as `fock_assembly` builds them) and under
//! UKS (alpha and beta channels through each fitter).
//!
//! Each `DfK` owns its own `DeviceSlot`, so the second fitter is charged its own
//! resident tensor. The fit rule is `2·(resident + scratch) <= pool` for both on the
//! device; between `resident + scratch` and that, the first fitter is resident and
//! the second declines (`PoolFull`, sticky, counted once) and runs the CPU path.
//! Driver-level gap: these tests drive `DfK::build_from_occ` directly. The alpha/beta
//! routing and which fitter receives which K in `fock_assembly` /
//! `subtract_rsh_exchange` (both pub(crate)) are NOT exercised here; tracked for an
//! in-crate unit test.
//! The gate is the derived two-stage bound of `gpu_dfk_resident.rs`:
//! `|K_dev - K_host| <= (eps_dev + eps_cpu)·S`, with `S` from each fitter's own B.
//!
//! The process-wide pool is installed once (its capacity is fixed per process), at
//! the larger of the two tests' needs; each arm of the tight test then takes a hog
//! reservation so the AVAILABLE pool is exactly the arm's target. The asserted bar
//! is the derived bound, ratio <= 1: the clean side sits far below it and the f32-B
//! defect side far above, so it lies between them (the geometric midpoint of the two
//! sides would be looser than the proven bound and is not used).
//!
//! MEASURED (max |dK| / bound):
//!   arm                          clean side            f32-B defect side
//!   RSH water/cc-pVDZ, tight     SR 5.0e-3 (LR on CPU: 0)   SR 4.0e4
//!   RSH water/cc-pVDZ, doubled   SR 5.0e-3, LR 8.7e-3       SR 4.0e4, LR 1.7e5
//!   UKS O2/cc-pVDZ (12 builds)   worst 1.3e-2               worst 1.7e5
//! Largest clean 1.3e-2, smallest defect 4.0e4: the bar 1.0 (the derived bound) sits
//! between them, 80x above the clean side and 4e4x below the defect side (their
//! geometric midpoint is 22, above the bound and so not used). Identical on GPU 0
//! and GPU 1. Mutation: the f32-B seam forced on in the tight CLEAN arm fails the
//! test at the bound assertion (SR ratio 4.0e4 > 1).
use ferric_core::basis;
use ferric_core::gpu::stats::GpuStatsSnapshot;
use ferric_core::gpu::{install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_k::DfK;
use ferric_scf::df_k_gpu::{k_error_factor, resident_bytes, FORCE_HOST, ROUND_B_TO_F32};
use ferric_scf::fock::KBuilder;
use ndarray::{Array2, ArrayView2};
use std::sync::atomic::Ordering;
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());

const WATER: &str = "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n";
const O2: &str = "2\nO2\nO 0 0 0\nO 0 0 1.21\n";
const OMEGA: f64 = 0.3;

/// Sizes from the shell structure alone: (n, band, nelec).
fn dims(xyz: &str, mult: usize) -> (usize, usize, usize) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    (obs.nbasis(), aux.nbasis(), mol.nelec() as usize)
}

/// Device bytes one fitter holds once built: B, K accumulator, C_occ, one-chunk scratch.
fn one_fitter_bytes(n: usize, band: usize, nocc: usize) -> usize {
    resident_bytes(band, n).unwrap() + 8 * n * nocc + 8 * nocc * n * band
}

/// Tight target: the first fitter fits and half a resident tensor is left over, so
/// the second cannot. (`SCRATCH_BYTES_DEFAULT` is a CEILING on the scratch; the
/// scratch actually taken here is one chunk of 8·nocc·n·band bytes, far below it.)
fn tight_target() -> usize {
    let (n, band, ne) = dims(WATER, 1);
    one_fitter_bytes(n, band, ne / 2) + resident_bytes(band, n).unwrap() / 2
}

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
        let (n, band, ne) = dims(O2, 3);
        let nalpha = (ne + 3 - 1) / 2; // triplet: ne = 16 -> 9 alpha, 7 beta
        let uks = 2 * one_fitter_bytes(n, band, nalpha);
        let cap = (2 * tight_target()).max(uks + uks / 4);
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            memory_gb: Some(cap as f64 / 1e9),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

/// (erfc fitter, erf fitter, n) for one system.
fn pair(xyz: &str, mult: usize) -> (DfK<'static>, DfK<'static>, usize) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let n = obs.nbasis();
    let mk = |op| DfK::new(op, &obs, &aux, usize::MAX).unwrap();
    (mk(Operator::erfc(OMEGA)), mk(Operator::erf(OMEGA)), n)
}

fn c_occ(n: usize, nocc: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, nocc), |(mu, i)| {
        0.05 * (((mu * 5 + i * 3 + salt * 7) % 17) as f64 - 8.0)
    })
}

/// S_μν = Σ_P Σ_i (|B_P||C|)_μi (|B_P||C|)_νi from a fitter's flat dressed band.
fn s_matrix(dfk: &DfK, n: usize, c: &Array2<f64>) -> Array2<f64> {
    let flat = dfk.dressed_incore_flat_for_test().expect("in core");
    let abs_c = c.mapv(f64::abs);
    let all = flat.as_slice().expect("standard layout");
    let mut s = Array2::<f64>::zeros((n, n));
    for p in 0..flat.nrows() {
        let bp = ArrayView2::from_shape((n, n), &all[p * n * n..(p + 1) * n * n]).unwrap();
        let a = bp.mapv(f64::abs).dot(&abs_c);
        s += &a.dot(&a.t());
    }
    s
}

/// max over the upper triangle of |a − b| / (eps·S·1.01).
fn max_ratio(a: &Array2<f64>, b: &Array2<f64>, s: &Array2<f64>, bound_eps: f64) -> f64 {
    let n = a.nrows();
    let mut worst = 0.0f64;
    for r in 0..n {
        for c in r..n {
            let bound = bound_eps * s[(r, c)] * 1.01 + f64::MIN_POSITIVE;
            worst = worst.max((a[(r, c)] - b[(r, c)]).abs() / bound);
        }
    }
    worst
}

fn host_k(dfk: &mut DfK, c: &Array2<f64>) -> Array2<f64> {
    FORCE_HOST.store(true, Ordering::SeqCst);
    let n = c.nrows();
    let mut k = Array2::zeros((n, n));
    dfk.build_from_occ(c, &mut k).unwrap();
    FORCE_HOST.store(false, Ordering::SeqCst);
    k
}

/// ε_dev + ε_cpu for a single-chunk device build against the CPU path.
fn eps(n: usize, nocc: usize, band: usize) -> f64 {
    k_error_factor(n, band * nocc, 1)
        + k_error_factor(n, (4096 / n.max(1)).clamp(4, 64) * nocc, band)
}

/// Make the AVAILABLE pool exactly `target` bytes (a hog holds the rest).
fn available_exactly(target: usize) -> ferric_core::gpu::pool::DeviceReservation {
    let p = pool().unwrap();
    assert_eq!(
        p.available_bytes(),
        p.capacity_bytes(),
        "another test left a reservation behind"
    );
    assert!(target <= p.capacity_bytes(), "installed pool too small");
    p.reserve("test hog", p.capacity_bytes() - target).unwrap()
}

/// [declined, PoolFull fallbacks, resident uploads, device builds] between two snapshots.
fn delta(a: &GpuStatsSnapshot, b: &GpuStatsSnapshot) -> [u64; 4] {
    [
        b.dfk_declined - a.dfk_declined,
        b.gemm_cpu_pool_full - a.gemm_cpu_pool_full,
        b.resident_uploads - a.resident_uploads,
        b.dfk_device_builds - a.dfk_device_builds,
    ]
}

/// One RSH exchange (two fitters, same C_occ, two iterations) with the available pool
/// at `target`; returns the stats delta and the worst ratios [SR, LR].
fn rsh_arm(target: usize, round_b: bool) -> ([u64; 4], [f64; 2]) {
    let (mut sr, mut lr, n) = pair(WATER, 1);
    let nocc = 5;
    let band = sr.dressed_incore_flat_for_test().unwrap().nrows();
    assert_eq!(band, dims(WATER, 1).1, "sizing used the wrong band");
    let c = c_occ(n, nocc, 1);
    let (s_sr, s_lr) = (s_matrix(&sr, n, &c), s_matrix(&lr, n, &c));
    let (h_sr, h_lr) = (host_k(&mut sr, &c), host_k(&mut lr, &c));
    let hog = available_exactly(target);
    assert_eq!(
        pool().unwrap().available_bytes(),
        target,
        "hog left the wrong amount"
    );
    let a = stats();
    ROUND_B_TO_F32.store(round_b, Ordering::SeqCst);
    let (mut k_sr, mut k_lr) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    for _ in 0..2 {
        sr.build_from_occ(&c, &mut k_sr).unwrap();
        lr.build_from_occ(&c, &mut k_lr).unwrap();
    }
    ROUND_B_TO_F32.store(false, Ordering::SeqCst);
    let d = delta(&a, &stats());
    if d[0] > 0 {
        // one fitter is resident and the REAL pool refuses a second tensor of that size
        let p = pool().unwrap();
        let r = resident_bytes(band, n).unwrap();
        assert!(p.available_bytes() < r, "a second fitter would still fit");
        assert!(p.try_reserve("second fitter probe", r).is_none());
    }
    drop((sr, lr));
    drop(hog);
    let e = eps(n, nocc, band);
    (
        d,
        [
            max_ratio(&k_sr, &h_sr, &s_sr, e),
            max_ratio(&k_lr, &h_lr, &s_lr, e),
        ],
    )
}

/// Pool fits ONE fitter, not two: the erfc fitter is resident, the erf fitter declines
/// once with PoolFull and runs the CPU path; both K match the forced-host K inside the
/// two-stage bound. With the pool doubled both fitters are resident (two uploads, no
/// decline). The f32-B defect on each arm is asserted to exceed the bound, so the
/// gate cannot go blind. In the tight arm the K_LR comparison is CPU against CPU
/// (fallback output correctness only), not a device check.
#[test]
fn rsh_second_fitter_declines_cleanly_when_only_one_fits() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (n, band, _) = dims(WATER, 1);
    let tight = tight_target();
    let r1 = resident_bytes(band, n).unwrap();
    // "one fits, two do not" is checked against the real pool inside `rsh_arm`
    assert!(tight >= one_fitter_bytes(n, band, 5));
    eprintln!("water/cc-pVDZ: n {n} band {band}; tight pool {tight} B, resident {r1} B each");

    // one fitter fits: [declined, pool_full, uploads, device builds]
    let (d, [sr, lr]) = rsh_arm(tight, false);
    eprintln!("tight, clean:   {d:?} ratios SR {sr:.3e} LR {lr:.3e}");
    assert_eq!(
        d,
        [1, 1, 1, 2],
        "declined once (PoolFull), one upload, SR built twice"
    );
    assert!(
        sr <= 1.0 && lr <= 1.0,
        "K outside the derived bound: SR {sr:e} LR {lr:e}"
    );
    let (_, [sr_m, _]) = rsh_arm(tight, true);
    eprintln!("tight, f32-B:   SR ratio {sr_m:.3e} (LR is on the CPU)");
    assert!(
        sr_m > 1.0,
        "the f32-B mutant must exceed the bound, got {sr_m:e}"
    );

    // pool doubled: both resident
    let (d, [sr, lr]) = rsh_arm(2 * tight, false);
    eprintln!("doubled, clean: {d:?} ratios SR {sr:.3e} LR {lr:.3e}");
    assert_eq!(
        d,
        [0, 0, 2, 4],
        "no decline, two uploads, four device builds"
    );
    assert!(
        sr <= 1.0 && lr <= 1.0,
        "K outside the derived bound: SR {sr:e} LR {lr:e}"
    );
    let (_, [sr_m, lr_m]) = rsh_arm(2 * tight, true);
    eprintln!("doubled, f32-B: SR ratio {sr_m:.3e} LR ratio {lr_m:.3e}");
    assert!(
        sr_m > 1.0 && lr_m > 1.0,
        "both mutants must exceed the bound: {sr_m:e} {lr_m:e}"
    );
}

/// Three UKS iterations (triplet O2: 9 alpha, 7 beta columns) through two fitters.
/// Returns (stats after iteration 1, stats after iteration 3, worst ratio over all
/// 12 builds against the forced-host K).
fn uks_arm(round_b: bool) -> (GpuStatsSnapshot, GpuStatsSnapshot, f64) {
    let (mut sr, mut lr, n) = pair(O2, 3);
    let (na, nb) = (9usize, 7usize);
    let band = sr.dressed_incore_flat_for_test().unwrap().nrows();
    let mut snaps = vec![];
    let mut worst = 0.0f64;
    for it in 0..3 {
        let (ca, cb) = (c_occ(n, na, 3 * it + 1), c_occ(n, nb, 3 * it + 2));
        ROUND_B_TO_F32.store(round_b, Ordering::SeqCst);
        let mut ks = vec![];
        for dfk in [&mut sr, &mut lr] {
            for c in [&ca, &cb] {
                let mut k = Array2::zeros((n, n));
                dfk.build_from_occ(c, &mut k).unwrap();
                ks.push(k);
            }
        }
        ROUND_B_TO_F32.store(false, Ordering::SeqCst);
        snaps.push(stats());
        let cases = [(0, &ca), (0, &cb), (1, &ca), (1, &cb)];
        for (k, (which, c)) in ks.iter().zip(cases) {
            let dfk = if which == 0 { &mut sr } else { &mut lr };
            let s = s_matrix(dfk, n, c);
            let h = host_k(dfk, c);
            worst = worst.max(max_ratio(k, &h, &s, eps(n, c.ncols(), band)));
        }
    }
    (snaps[0], snaps[2], worst)
}

/// UKS triplet O2/cc-pVDZ with two fitters: the alpha and beta builds of a fitter
/// share its one resident B, so the run is exactly two uploads and four device builds
/// per iteration, and after the first iteration only C_occ goes up and K comes down.
#[test]
fn uks_alpha_and_beta_share_one_resident_b_per_fitter() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (n, _, _) = dims(O2, 3);
    let p = pool().unwrap();
    assert_eq!(p.available_bytes(), p.capacity_bytes());
    let before = stats();
    let (first, last, clean) = uks_arm(false);
    eprintln!("UKS O2/cc-pVDZ clean: worst ratio {clean:.3e}");
    assert_eq!(
        first.resident_uploads - before.resident_uploads,
        2,
        "one upload per fitter"
    );
    assert_eq!(
        last.resident_uploads, first.resident_uploads,
        "B was re-uploaded"
    );
    assert_eq!(first.dfk_device_builds - before.dfk_device_builds, 4);
    assert_eq!(
        last.dfk_device_builds - first.dfk_device_builds,
        8,
        "four per iteration"
    );
    assert_eq!(first.dfk_declined, before.dfk_declined, "a fitter declined");
    // two later iterations: per fitter alpha+beta C_occ up (8·n·(9+7)), 4 K down (8·n²)
    assert_eq!(
        last.bytes_h2d - first.bytes_h2d,
        2 * 2 * (8 * n * 16) as u64
    );
    assert_eq!(last.bytes_d2h - first.bytes_d2h, 2 * 4 * (8 * n * n) as u64);
    assert!(clean <= 1.0, "UKS K outside the derived bound: {clean:e}");
    let (_, _, defect) = uks_arm(true);
    eprintln!("UKS O2/cc-pVDZ f32-B: worst ratio {defect:.3e}");
    assert!(
        defect > 1.0,
        "the f32-B mutant must exceed the bound, got {defect:e}"
    );
}
