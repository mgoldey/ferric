#![cfg(feature = "gpu")]
//! Mixed `dfk-occ`: f32-resident dressed B and C_occ, f32 k-panels flushed into f64,
//! f64 SYRK. The kernel is NOT in `MixedKernelSet::SHIPPED`, so these tests enable
//! it through `GpuSettings::resolve_with_default(.., shipped + dfk-occ)` (the real
//! resolver, so the allowlist logic under test is the production one) and hand the
//! result to the `MIXED_SETTINGS_OVERRIDE` seam of `df_k_gpu`, which the DF-K
//! precision decision reads instead of the installed settings. Mode, device and
//! pool still come from the installed (auto, 4 GB) settings.
//!
//! The K gate is derived. With S_μν = Σ_{P,i} (|B_P||C|)_μi (|B_P||C|)_νi, the mixed
//! K differs from the exact one by at most `k_error_factor_mixed(n, b, k_chunk,
//! nchunks)·S` (stage 1: f32 operands, f32 panel sums of depth b, f64 flush over
//! ⌈n/b⌉ panels; stage 2: f64 SYRK), and the f64 host K by at most its own
//! `k_error_factor`; the compared quantity is |K_mixed − K_host| / (bound·S·1.01)
//! over the upper triangle, bound = ε_mixed(device) + ε_f64(host).
//!
//! MEASURED SIDES (GTX 1080, printed on every run; def2-universal-jkfit,
//! deterministic C_occ; "bound ratio" = max |K_mixed − K_host|/(bound·S·1.01)):
//!   system (nocc)            clean (b = 64)   (a) plain f32 acc.   (b) truncating upload
//!                            FERRIC_GPU_MIXED_K_PANEL = n          TRUNCATE_B_TO_F32
//!   water/cc-pVDZ (5)        6.8e-3           6.8e-3 (n < b: one panel anyway)   1.4e-2
//!   butane/cc-pVDZ (17)      1.4e-3           2.7e-3               3.1e-3
//!   octane/cc-pVDZ (33)      1.2e-3           1.7e-3               2.2e-3
//! The bound is a worst case: the clean side sits 2.5 decades below it and neither
//! defect reaches it (they are 1.4x to 2.4x the clean side), so the bound alone is
//! blind to both. The reason is that against the f64 host K the dominant term is
//! the f32 rounding of the OPERANDS, common to the clean and plain-accumulation
//! runs. The discriminating gate therefore compares against a TWIN: the f64 device
//! K of the f32-rounded operands, where the operand rounding cancels and only the
//! panel accumulation remains. RMS over the upper triangle of |K − K_twin|/(ε·S),
//! same ε as above (octane, n = 202, 4 panels at b = 64):
//!   clean 3.36e-5   (a) plain f32 accumulation 5.30e-5   (b) truncating upload 3.32e-4
//!   (butane: 4.12e-5, 5.17e-5, 3.42e-4; water: 9.0e-4, 9.0e-4 (one panel), 4.1e-3)
//! The asserted bar is the geometric midpoint to the NEARER defect (a):
//! sqrt(3.36e-5 · 5.30e-5) = 4.22e-5, with the clean side 1.26x below and (a) 1.26x
//! above it; (b) is 7.9x above. Plain accumulation is only 1.6x the clean side
//! because n/b = 4 panels: the random-walk depth ratio sqrt(202/64) = 1.8. The twin
//! bar is a measurement on this device, not a property of the algorithm; the
//! derived statement is the bound (clean <= 1, asserted on every fixture).
//!
//! Also pinned: resident bytes (4·band·n², half the f64 path), mixed panels per
//! build (band·⌈n/b⌉), the allowlist (precision = mixed with only rimp2-energy
//! leaves DF-K f64 and moves no mixed counter), the F32Range fallback (one f64
//! device upload, counted once, K bit-equal to the f64 device K), the f64 path's
//! K bits when mixed is not allowed, and the resolver's refusal of dfk-occ in
//! this build (not shipped).
use ferric_core::basis;
use ferric_core::gpu::config::{GpuSettings, GpuSettingsExplicit};
use ferric_core::gpu::device::FORCE_KERNEL_FAILURE;
use ferric_core::gpu::mixed::effective_k_panel;
use ferric_core::gpu::{
    install, pool, probe, stats, GpuMode, GpuStatus, MixedKernel, MixedKernelSet, Precision,
};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_k::DfK;
use ferric_scf::df_k_gpu::{
    k_error_factor, k_error_factor_mixed, DeviceDfK, DfkPrecision, FORCE_HOST,
    MIXED_SETTINGS_OVERRIDE, TRUNCATE_B_TO_F32,
};
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};
use ndarray::{Array2, ArrayView2};
use std::sync::atomic::Ordering;
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());

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

/// The build's shipped set plus dfk-occ: what the C3 commit will make real.
fn shipped_plus_dfk() -> MixedKernelSet {
    MixedKernelSet::SHIPPED.with(MixedKernel::DfkOcc)
}

fn resolve(
    kernels: Option<MixedKernelSet>,
    shipped: MixedKernelSet,
) -> Result<GpuSettings, String> {
    GpuSettings::resolve_with_default(
        GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            precision: Some(Precision::Mixed),
            mixed_kernels: kernels,
            memory_gb: Some(4.0),
            ..Default::default()
        },
        |_| None,
        Precision::F64,
        shipped,
    )
    .map(|(s, _)| s)
}

/// Installs the override for the guard's lifetime; clears it (and every seam) on drop.
struct Mixed;
impl Mixed {
    fn with(kernels: MixedKernelSet) -> Self {
        let s = resolve(Some(kernels), shipped_plus_dfk()).expect("resolve");
        *MIXED_SETTINGS_OVERRIDE.lock().unwrap() = Some(s);
        Mixed
    }
}
impl Drop for Mixed {
    fn drop(&mut self) {
        *MIXED_SETTINGS_OVERRIDE
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        TRUNCATE_B_TO_F32.store(false, Ordering::SeqCst);
        FORCE_HOST.store(false, Ordering::SeqCst);
        FORCE_KERNEL_FAILURE.store(false, Ordering::SeqCst);
        std::env::remove_var("FERRIC_GPU_MIXED_K_PANEL");
    }
}

fn dfk_for(xyz: &str, obs_name: &str) -> (DfK<'static>, usize) {
    let mult = if xyz.contains("OH") { 2 } else { 1 };
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

fn alkane(k: usize) -> String {
    std::fs::read_to_string(format!("../../testdata/molecules/alkane_{k}.xyz")).unwrap()
}

fn c_occ(n: usize, nocc: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, nocc), |(mu, i)| {
        0.05 * (((mu * 5 + i * 3 + salt * 7) % 17) as f64 - 8.0)
    })
}

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

/// RMS over the upper triangle of |a - b| / (eps·S) (the typical, not the worst, element).
fn rms_ratio(a: &Array2<f64>, b: &Array2<f64>, s: &Array2<f64>, eps: f64) -> f64 {
    let n = a.nrows();
    let (mut sum, mut cnt) = (0.0f64, 0usize);
    for r in 0..n {
        for c in r..n {
            let q = (a[(r, c)] - b[(r, c)]).abs() / (eps * s[(r, c)] + f64::MIN_POSITIVE);
            sum += q * q;
            cnt += 1;
        }
    }
    (sum / cnt as f64).sqrt()
}

/// Geometric midpoint of the measured clean (3.36e-5) and nearer defect (plain
/// f32 accumulation, 5.30e-5) twin RMS on octane/cc-pVDZ: sqrt(3.36e-5 · 5.30e-5).
const TWIN_RMS_BAR: f64 = 4.22e-5;

fn host_k(dfk: &mut DfK, c: &Array2<f64>) -> Array2<f64> {
    FORCE_HOST.store(true, Ordering::SeqCst);
    let n = c.nrows();
    let mut k = Array2::zeros((n, n));
    dfk.build_from_occ(c, &mut k).unwrap();
    FORCE_HOST.store(false, Ordering::SeqCst);
    k
}

fn eps_host(n: usize, nocc: usize, naux: usize) -> f64 {
    k_error_factor(n, (4096 / n.max(1)).clamp(4, 64) * nocc, naux)
}

fn bits(m: &Array2<f64>) -> Vec<u64> {
    m.iter().map(|v| v.to_bits()).collect()
}

/// Mixed device K of a FRESH DfK (the precision is fixed at its first upload).
fn dispatched_k(xyz: &str, obs: &str, c: &Array2<f64>) -> Array2<f64> {
    let (mut dfk, n) = dfk_for(xyz, obs);
    let mut k = Array2::zeros((n, n));
    dfk.build_from_occ(c, &mut k).unwrap();
    k
}

#[test]
fn mixed_k_is_inside_its_bound_and_two_mutants_leave_it() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    for (label, xyz, nocc) in [
        ("water/cc-pVDZ", WATER.to_string(), 5usize),
        ("butane/cc-pVDZ", alkane(4), 17),
        ("octane/cc-pVDZ", alkane(8), 33),
    ] {
        let (mut dfk0, n) = dfk_for(&xyz, "cc-pvdz");
        let c = c_occ(n, nocc, 1);
        let flat = dfk0.dressed_incore_flat_for_test().unwrap().to_owned();
        let naux = flat.nrows();
        let s = s_matrix(&flat.view(), n, &c);
        let k_host = host_k(&mut dfk0, &c);
        let b = effective_k_panel();
        let eps = k_error_factor_mixed(n, b, naux * nocc, 1) + eps_host(n, nocc, naux);

        let _m = Mixed::with(shipped_plus_dfk());
        let s0 = stats();
        let k_clean = dispatched_k(&xyz, "cc-pvdz", &c);
        let panels = stats().mixed_panels - s0.mixed_panels;
        assert_eq!(
            panels,
            (naux * n.div_ceil(b)) as u64,
            "{label}: the mixed path did not run"
        );
        let clean = max_ratio(&k_clean, &k_host, &s, eps);
        let rms_clean = rms_ratio(&k_clean, &k_host, &s, eps);

        std::env::set_var("FERRIC_GPU_MIXED_K_PANEL", n.to_string());
        let k_a = dispatched_k(&xyz, "cc-pvdz", &c);
        std::env::remove_var("FERRIC_GPU_MIXED_K_PANEL");
        let defect_a = max_ratio(&k_a, &k_host, &s, eps);
        let rms_a = rms_ratio(&k_a, &k_host, &s, eps);

        TRUNCATE_B_TO_F32.store(true, Ordering::SeqCst);
        let k_b = dispatched_k(&xyz, "cc-pvdz", &c);
        TRUNCATE_B_TO_F32.store(false, Ordering::SeqCst);
        let defect_b = max_ratio(&k_b, &k_host, &s, eps);
        let rms_b = rms_ratio(&k_b, &k_host, &s, eps);
        eprintln!("{label}: RMS clean {rms_clean:.4e}  (a) {rms_a:.4e}  (b) {rms_b:.4e}");
        // TWIN: the f64 device K of the f32-ROUNDED operands. Mixed minus twin is the
        // f32 panel accumulation alone (the operand rounding cancels).
        let dev = ferric_core::gpu::device::device(0).unwrap();
        let rb = ferric_core::gpu::mixed_host::round_trip_f32(&flat.view());
        let rc = ferric_core::gpu::mixed_host::round_trip_f32(&c.view());
        let mut tw = DeviceDfK::upload(&dev, &pool().unwrap(), &rb.view(), n).unwrap();
        let mut k_twin = Array2::zeros((n, n));
        tw.build_from_occ(&rc.view(), &mut k_twin).unwrap();
        drop(tw);
        let (tc, ta, tb) = (
            rms_ratio(&k_clean, &k_twin, &s, eps),
            rms_ratio(&k_a, &k_twin, &s, eps),
            rms_ratio(&k_b, &k_twin, &s, eps),
        );
        eprintln!("{label}: vs TWIN: RMS clean {tc:.4e} (a) {ta:.4e} (b) {tb:.4e}");
        if n > 2 * b {
            // Accumulation depth n = 202 > 3 panels: both defects are visible.
            assert!(tc < TWIN_RMS_BAR, "{label}: clean twin RMS {tc:e} >= bar");
            assert!(
                ta > TWIN_RMS_BAR,
                "{label}: plain f32 accumulation not caught ({ta:e})"
            );
            assert!(
                tb > TWIN_RMS_BAR,
                "{label}: truncating upload not caught ({tb:e})"
            );
        }

        eprintln!(
            "{label}: n={n} naux={naux} nocc={nocc} b={b} bound_factor={eps:.3e}: \
             clean {clean:.4e}  (a) plain f32 accumulation {defect_a:.4e}  (b) truncating upload {defect_b:.4e}"
        );
        assert!(
            clean <= 1.0,
            "{label}: mixed K outside its bound ({clean:e})"
        );
        for r in 0..n {
            for cc in 0..r {
                assert_eq!(k_clean[(r, cc)], k_clean[(cc, r)]);
            }
        }
    }
}

const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

/// Converged occupied orbitals (host RI-JK SCF, `def2-universal-jkfit` J and K):
/// restricted when `mult == 1` (beta = None), unrestricted otherwise.
fn scf_occ(xyz: &str, mult: usize, obs_name: &str) -> (Array2<f64>, Option<Array2<f64>>) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let aux = Some("def2-universal-jkfit".to_string());
    FORCE_HOST.store(true, Ordering::SeqCst);
    let res = if mult == 1 {
        let cfg = RhfConfig {
            energy_conv: 1e-3,
            density_conv: 1e-8,
            max_iter: 200,
            df_j_aux: aux.clone(),
            df_k_aux: aux,
            ..Default::default()
        };
        solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg)
    } else {
        let cfg = UhfConfig {
            energy_conv: 1e-3,
            density_conv: 1e-8,
            max_iter: 300,
            df_j_aux: aux.clone(),
            df_k_aux: aux,
            ..Default::default()
        };
        solve_uhf(&ctx, &mol, &obs, &bounds, &cfg)
    };
    FORCE_HOST.store(false, Ordering::SeqCst);
    let res = res.expect("SCF");
    let nelec = mol.nelec() as usize;
    let na = (nelec + mult - 1) / 2;
    let nb = nelec - na;
    let occ = |c: &Array2<f64>, k: usize| c.slice(ndarray::s![.., ..k]).to_owned();
    let ca = occ(&res.mos_alpha, na);
    // convention check: D_alpha = C_occ C_occ^T
    let d = ca.dot(&ca.t());
    let err = (&d - &res.density_alpha)
        .mapv(f64::abs)
        .fold(0.0f64, |a, &b| a.max(b));
    assert!(err < 1e-6, "MO column convention: |D - C C^T| = {err:e}");
    let cb = res.mos_beta.as_ref().map(|c| occ(c, nb));
    (ca, cb)
}

/// The f64 device K of the f32-ROUNDED operands: mixed minus this is the f32 panel
/// accumulation alone (the operand rounding is common to both).
fn twin_k(flat: &ArrayView2<f64>, c: &Array2<f64>, scratch: Option<usize>) -> Array2<f64> {
    let n = c.nrows();
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let rb = ferric_core::gpu::mixed_host::round_trip_f32(flat);
    let rc = ferric_core::gpu::mixed_host::round_trip_f32(&c.view());
    let mut tw = DeviceDfK::upload(&dev, &pool().unwrap(), &rb.view(), n).unwrap();
    if let Some(bytes) = scratch {
        tw = tw.with_scratch_bytes(bytes);
    }
    let mut k = Array2::zeros((n, n));
    tw.build_from_occ(&rc.view(), &mut k).unwrap();
    k
}

#[test]
fn mixed_alpha_beta_reuse_with_unequal_nocc_neither_panics_nor_accumulates() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // Converged UHF orbitals of OH: alpha 5, beta 4. The scratch is sized by alpha
    // and reused for beta (nocc_cap 5 > 4): every mixed copy must use nocc, not nocc_cap.
    let (ca, cb) = scf_occ(OH, 2, "cc-pvdz");
    let cb = cb.expect("UHF");
    assert_eq!((ca.ncols(), cb.ncols()), (5, 4));
    let (mut dfk0, n) = dfk_for(OH, "cc-pvdz");
    let flat = dfk0.dressed_incore_flat_for_test().unwrap().to_owned();
    let band = flat.nrows();
    let b = effective_k_panel();
    let hk = [host_k(&mut dfk0, &ca), host_k(&mut dfk0, &cb)];

    let _m = Mixed::with(shipped_plus_dfk());
    let (mut dfk, _) = dfk_for(OH, "cc-pvdz");
    let (mut ka1, mut kb, mut ka2) = (
        Array2::zeros((n, n)),
        Array2::zeros((n, n)),
        Array2::zeros((n, n)),
    );
    let s0 = stats();
    dfk.build_from_occ(&ca, &mut ka1).unwrap();
    dfk.build_from_occ(&cb, &mut kb).unwrap();
    dfk.build_from_occ(&ca, &mut ka2).unwrap();
    let s1 = stats();
    assert_eq!(
        s1.mixed_panels - s0.mixed_panels,
        (3 * band * n.div_ceil(b)) as u64,
        "three mixed builds"
    );
    assert_eq!(s1.mixed_fallback_f64, s0.mixed_fallback_f64);
    assert_eq!(s1.dfk_declined, s0.dfk_declined, "no decline");
    assert_eq!(
        bits(&ka1),
        bits(&ka2),
        "alpha rebuilt after beta differs: stale scratch or non-determinism"
    );
    for (label, c, k, host) in [("alpha", &ca, &ka1, &hk[0]), ("beta", &cb, &kb, &hk[1])] {
        let nocc = c.ncols();
        let s = s_matrix(&flat.view(), n, c);
        let eps = k_error_factor_mixed(n, b, band * nocc, 1) + eps_host(n, nocc, band);
        let clean = max_ratio(k, host, &s, eps);
        let twin = twin_k(&flat.view(), c, None);
        let rms = rms_ratio(k, &twin, &s, eps);
        eprintln!("OH {label}: nocc {nocc}: bound ratio {clean:.4e}; twin RMS {rms:.4e}");
        assert!(clean <= 1.0, "{label}: outside the bound ({clean:e})");
    }
}

#[test]
fn mixed_uploads_half_the_bytes_and_counts_panels() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let _m = Mixed::with(shipped_plus_dfk());
    let (mut dfk, n) = dfk_for(WATER, "cc-pvdz");
    let nocc = 5;
    let band = dfk.dressed_incore_flat_for_test().unwrap().nrows();
    let b = effective_k_panel();
    let c = c_occ(n, nocc, 2);
    let mut k = Array2::zeros((n, n));
    let a = stats();
    dfk.build_from_occ(&c, &mut k).unwrap(); // cold: uploads B (f32)
    let m1 = stats();
    assert_eq!(m1.resident_uploads, a.resident_uploads + 1);
    assert_eq!(
        m1.bytes_h2d - a.bytes_h2d,
        (4 * band * n * n + 4 * n * nocc) as u64,
        "resident f32 B plus the f32 C_occ"
    );
    assert_eq!(m1.bytes_d2h - a.bytes_d2h, (8 * n * n) as u64);
    assert_eq!(
        m1.mixed_panels - a.mixed_panels,
        (band * n.div_ceil(b)) as u64
    );
    assert_eq!(m1.gemm_mixed - a.gemm_mixed, band as u64);
    assert_eq!(m1.mixed_fallback_f64, a.mixed_fallback_f64);
    // a warm build: no re-upload, same panel count, 4·n·nocc up
    dfk.build_from_occ(&c_occ(n, nocc, 3), &mut k).unwrap();
    let m2 = stats();
    assert_eq!(m2.resident_uploads, m1.resident_uploads);
    assert_eq!(m2.bytes_h2d - m1.bytes_h2d, (4 * n * nocc) as u64);
    assert_eq!(
        m2.mixed_panels - m1.mixed_panels,
        (band * n.div_ceil(b)) as u64
    );
    let report = pool().unwrap().occupancy_report();
    assert!(report.contains("DF-K dressed B (f32)"), "{report}");
    assert!(!report.contains("DF-K dressed B\n"), "{report}");
}

#[test]
fn an_f32_range_violation_in_b_runs_the_f64_device_path_once_and_counts_it() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (dfk0, n) = dfk_for(WATER, "cc-pvdz");
    let mut flat = dfk0.dressed_incore_flat_for_test().unwrap().to_owned();
    let band = flat.nrows();
    flat[(band / 2, 7)] = 1e39; // finite in f64, beyond f32::MAX (3.4e38)
    let c = c_occ(n, 5, 4);
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let pool = pool().unwrap();

    let mut f64_dev = DeviceDfK::upload(&dev, &pool, &flat.view(), n).unwrap();
    let mut k_f64 = Array2::zeros((n, n));
    f64_dev.build_from_occ(&c.view(), &mut k_f64).unwrap();
    drop(f64_dev);

    let s0 = stats();
    let mut fb =
        DeviceDfK::upload_with_precision(&dev, &pool, &flat.view(), n, DfkPrecision::Mixed)
            .unwrap();
    let s1 = stats();
    assert!(!fb.is_mixed(), "an F32Range B must leave the f64 upload");
    assert_eq!(s1.mixed_fallback_f64 - s0.mixed_fallback_f64, 1);
    assert_eq!(
        s1.bytes_h2d - s0.bytes_h2d,
        (8 * band * n * n) as u64,
        "the f64 upload"
    );
    let (mut k1, mut k2) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    fb.build_from_occ(&c.view(), &mut k1).unwrap();
    fb.build_from_occ(&c.view(), &mut k2).unwrap();
    let s2 = stats();
    assert_eq!(s2.mixed_fallback_f64, s1.mixed_fallback_f64, "counted once");
    assert_eq!(s2.mixed_panels, s1.mixed_panels, "no mixed panel ran");
    assert_eq!(s2.dfk_declined, s0.dfk_declined, "never the CPU");
    assert_eq!(
        bits(&k1),
        bits(&k_f64),
        "K must equal the f64 device K bitwise"
    );
    assert_eq!(bits(&k2), bits(&k_f64));
}

#[test]
fn a_missing_flush_kernel_at_upload_runs_the_f64_device_path_and_counts_it() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (dfk0, n) = dfk_for(WATER, "cc-pvdz");
    let flat = dfk0.dressed_incore_flat_for_test().unwrap().to_owned();
    let band = flat.nrows();
    let (c, _) = scf_occ(WATER, 1, "cc-pvdz");
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let pool = pool().unwrap();
    let mut direct = DeviceDfK::upload(&dev, &pool, &flat.view(), n).unwrap();
    let mut k_f64 = Array2::zeros((n, n));
    direct.build_from_occ(&c.view(), &mut k_f64).unwrap();
    drop(direct);

    let _m = Mixed::with(shipped_plus_dfk());
    FORCE_KERNEL_FAILURE.store(true, Ordering::SeqCst);
    let s0 = stats();
    let mut fb =
        DeviceDfK::upload_with_precision(&dev, &pool, &flat.view(), n, DfkPrecision::Mixed)
            .unwrap();
    let s1 = stats();
    assert!(!fb.is_mixed());
    assert_eq!(s1.mixed_fallback_f64 - s0.mixed_fallback_f64, 1);
    assert_eq!(
        s1.bytes_h2d - s0.bytes_h2d,
        (8 * band * n * n) as u64,
        "no f32 bytes were uploaded or reserved"
    );
    let mut k = Array2::zeros((n, n));
    fb.build_from_occ(&c.view(), &mut k).unwrap();
    assert_eq!(bits(&k), bits(&k_f64), "K must equal the f64 device K");
    assert_eq!(stats().mixed_panels, s1.mixed_panels);
    assert_eq!(stats().dfk_declined, s0.dfk_declined, "never the CPU");
}

#[test]
fn a_flush_kernel_lost_after_a_mixed_upload_declines_sticky_and_leaves_k_alone() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut dfk0, n) = dfk_for(WATER, "cc-pvdz");
    let flat = dfk0.dressed_incore_flat_for_test().unwrap().to_owned();
    let (c, _) = scf_occ(WATER, 1, "cc-pvdz");
    let k_host = host_k(&mut dfk0, &c);
    let dev = ferric_core::gpu::device::device(0).unwrap();
    // (a) the device object: the build errors and k is untouched
    let _m = Mixed::with(shipped_plus_dfk());
    let mut d = DeviceDfK::upload_with_precision(
        &dev,
        &pool().unwrap(),
        &flat.view(),
        n,
        DfkPrecision::Mixed,
    )
    .unwrap();
    assert!(d.is_mixed());
    FORCE_KERNEL_FAILURE.store(true, Ordering::SeqCst);
    let mut k = Array2::from_elem((n, n), 7.0);
    let e = d.build_from_occ(&c.view(), &mut k).unwrap_err();
    assert!(
        matches!(e, ferric_core::gpu::device::GpuError::Kernel(_)),
        "{e:?}"
    );
    assert!(k.iter().all(|&v| v == 7.0), "k must be untouched on error");
    drop(d);
    FORCE_KERNEL_FAILURE.store(false, Ordering::SeqCst);

    // (b) the dispatcher: first build mixed, then the kernel is lost
    let (mut dfk, _) = dfk_for(WATER, "cc-pvdz");
    let s0 = stats();
    let mut k1 = Array2::zeros((n, n));
    dfk.build_from_occ(&c, &mut k1).unwrap();
    let s1 = stats();
    assert_eq!(s1.dfk_device_builds, s0.dfk_device_builds + 1);
    FORCE_KERNEL_FAILURE.store(true, Ordering::SeqCst);
    let mut k2 = Array2::zeros((n, n));
    dfk.build_from_occ(&c, &mut k2).unwrap(); // device fails, CPU answers
    FORCE_KERNEL_FAILURE.store(false, Ordering::SeqCst);
    let s2 = stats();
    assert_eq!(s2.dfk_declined, s1.dfk_declined + 1, "declined once");
    assert_eq!(s2.gemm_cpu_cuda_error, s1.gemm_cpu_cuda_error + 1);
    assert_eq!(
        s2.dfk_device_builds, s1.dfk_device_builds,
        "no device build"
    );
    assert_eq!(bits(&k2), bits(&k_host), "the CPU K is the host K");
    let mut k3 = Array2::zeros((n, n));
    dfk.build_from_occ(&c, &mut k3).unwrap(); // kernel is back, slot stays declined
    let s3 = stats();
    assert_eq!(
        s3.dfk_device_builds, s2.dfk_device_builds,
        "decline is sticky"
    );
    assert_eq!(s3.dfk_declined, s2.dfk_declined);
    assert_eq!(bits(&k3), bits(&k_host));
}

#[test]
fn a_c_occ_element_beyond_f32_declines_to_the_cpu_sticky_and_is_counted_by_reason() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut dfk0, n) = dfk_for(WATER, "cc-pvdz");
    let flat = dfk0.dressed_incore_flat_for_test().unwrap().to_owned();
    let (mut c, _) = scf_occ(WATER, 1, "cc-pvdz");
    c[(3, 1)] = 1e39; // finite in f64, beyond f32::MAX
    let k_host = host_k(&mut dfk0, &c);
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let _m = Mixed::with(shipped_plus_dfk());
    let mut d = DeviceDfK::upload_with_precision(
        &dev,
        &pool().unwrap(),
        &flat.view(),
        n,
        DfkPrecision::Mixed,
    )
    .unwrap();
    let mut k = Array2::from_elem((n, n), 7.0);
    let e = d.build_from_occ(&c.view(), &mut k).unwrap_err();
    assert!(
        matches!(e, ferric_core::gpu::device::GpuError::F32Range(_)),
        "{e:?}"
    );
    assert!(k.iter().all(|&v| v == 7.0), "k must be untouched on error");
    drop(d);

    let (mut dfk, _) = dfk_for(WATER, "cc-pvdz");
    let s0 = stats();
    let mut k1 = Array2::zeros((n, n));
    dfk.build_from_occ(&c, &mut k1).unwrap();
    let s1 = stats();
    assert_eq!(s1.dfk_declined, s0.dfk_declined + 1);
    assert_eq!(
        s1.gemm_cpu_f32_range,
        s0.gemm_cpu_f32_range + 1,
        "own reason"
    );
    assert_eq!(
        s1.gemm_cpu_cuda_error, s0.gemm_cpu_cuda_error,
        "not a CUDA error"
    );
    assert_eq!(bits(&k1), bits(&k_host), "never a wrong number: the CPU K");
    let mut k2 = Array2::zeros((n, n));
    dfk.build_from_occ(&c, &mut k2).unwrap();
    assert_eq!(stats().dfk_declined, s1.dfk_declined, "sticky");
}

#[test]
fn mixed_is_never_used_unless_the_allowlist_says_so() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // precision = mixed, but only rimp2-energy listed: DF-K must stay f64.
    let _m = Mixed::with(MixedKernelSet::EMPTY.with(MixedKernel::RiMp2Energy));
    let (mut dfk, n) = dfk_for(WATER, "cc-pvdz");
    let nocc = 5;
    let band = dfk.dressed_incore_flat_for_test().unwrap().nrows();
    let flat = dfk.dressed_incore_flat_for_test().unwrap().to_owned();
    let c = c_occ(n, nocc, 5);
    let mut k = Array2::zeros((n, n));
    let a = stats();
    dfk.build_from_occ(&c, &mut k).unwrap();
    let b = stats();
    assert_eq!(b.dfk_device_builds, a.dfk_device_builds + 1, "device ran");
    assert_eq!(
        b.bytes_h2d - a.bytes_h2d,
        (8 * band * n * n + 8 * n * nocc) as u64
    );
    assert_eq!(b.mixed_panels, a.mixed_panels);
    assert_eq!(b.gemm_mixed, a.gemm_mixed);
    assert_eq!(b.mixed_fallback_f64, a.mixed_fallback_f64);
    // and the K is exactly the f64 device K of the same B
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let mut direct = DeviceDfK::upload(&dev, &pool().unwrap(), &flat.view(), n).unwrap();
    let mut kd = Array2::zeros((n, n));
    direct.build_from_occ(&c.view(), &mut kd).unwrap();
    assert_eq!(bits(&k), bits(&kd));
}

#[test]
fn default_config_keeps_the_f64_path_bit_for_bit_and_moves_no_mixed_counter() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // no override: the installed settings (precision f64) decide
    let (mut dfk, n) = dfk_for(WATER, "cc-pvdz");
    let flat = dfk.dressed_incore_flat_for_test().unwrap().to_owned();
    let c = c_occ(n, 5, 6);
    let mut k = Array2::zeros((n, n));
    let a = stats();
    dfk.build_from_occ(&c, &mut k).unwrap();
    let b = stats();
    assert_eq!(
        (b.mixed_panels, b.gemm_mixed),
        (a.mixed_panels, a.gemm_mixed)
    );
    assert_eq!(b.mixed_fallback_f64, a.mixed_fallback_f64);
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let mut direct = DeviceDfK::upload(&dev, &pool().unwrap(), &flat.view(), n).unwrap();
    let mut kd = Array2::zeros((n, n));
    direct.build_from_occ(&c.view(), &mut kd).unwrap();
    assert_eq!(bits(&k), bits(&kd), "f64 dispatcher K != f64 device K");
    // (mode off is pinned by gpu_dfk_mode_off: K bits equal the forced-host twin)
}

#[test]
fn the_resolver_refuses_dfk_occ_in_this_build_and_a_kernel_list_with_f64() {
    // This build ships no dfk-occ row: naming it is a refusal, so no config can
    // reach the mixed DF-K path before the C3 commit.
    assert!(!MixedKernelSet::SHIPPED.contains(MixedKernel::DfkOcc));
    let e = resolve(
        Some(MixedKernelSet::EMPTY.with(MixedKernel::DfkOcc)),
        MixedKernelSet::SHIPPED,
    )
    .unwrap_err();
    assert!(e.contains("dfk-occ") && e.contains("not shipped"), "{e}");
    // mode off + kernel named + f64 precision is refused as well
    let e = GpuSettings::resolve_with_default(
        GpuSettingsExplicit {
            mixed_kernels: Some(MixedKernelSet::EMPTY.with(MixedKernel::DfkOcc)),
            ..Default::default()
        },
        |_| None,
        Precision::F64,
        MixedKernelSet::SHIPPED,
    )
    .unwrap_err();
    assert!(e.contains("f64"), "{e}");
}
