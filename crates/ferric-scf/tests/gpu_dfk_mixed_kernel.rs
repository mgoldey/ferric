#![cfg(feature = "gpu")]
//! Mixed `dfk-occ`: f32-resident dressed B and C_occ, f32 k-panels flushed into f64,
//! f64 SYRK. The kernel is NOT in `MixedKernelSet::SHIPPED`, so these tests enable
//! it through `GpuSettings::resolve_with_default(.., shipped + dfk-occ)` (the real
//! resolver, so the allowlist logic under test is the production one) and hand the
//! result to the `MIXED_SETTINGS_OVERRIDE` seam of `df_k_gpu`, which the DF-K
//! precision decision reads instead of the installed settings. Mode, device and
//! pool still come from the installed (auto, 4 GB) settings.
//!
//! WHAT EACH GATE CATCHES (the K bound is a worst case and is blind to subtle
//! defects: against the f64 host K the clean side sits 1e-2..1e-3 of the bound,
//! and the defects stay below it, because the f32 rounding of the OPERANDS
//! dominates the error).
//!  * `clean <= 1` against `k_error_factor_mixed` (+ the host f64 K's own factor)
//!    is the only DERIVED statement: |K_mixed - K_host| <= (eps_mixed + eps_host)*S
//!    with S = sum_{P,i} (|B_P||C|)(|B_P||C|)^T, on converged RI-JK orbitals.
//!  * `the_f64_flush_across_k_panels_is_pinned_exactly` (deterministic, device and
//!    order independent): n = b + 8, band 1, nocc 1, two single-term panels
//!    2^24 and 1: the f64 flush gives K = 2^48 + 2^25 + 1 bit for bit; summing the
//!    panels in f32 gives 2^48. Mutant FERRIC_GPU_MIXED_K_PANEL = n (one panel):
//!    K[nu*,nu*] = 2.81474976710656e14 against 281475010265089.
//!  * `c_occ_rounding_is_round_to_nearest_even_bit_for_bit_through_the_kernel`
//!    (B = I, nocc 1: K = c32 (x) c32, a 48-bit exact product): C_occ rounding is
//!    round-to-nearest-even; with the core bit-exact `upload_rounded` test this is
//!    the truncation guarantee for B as well.
//!  * the exact panel count band*ceil(n/b) (asserted on hexane n = 154 and octane
//!    n = 202, both > 2b) catches a changed panel width, b = 128 included: the
//!    RMS bar below does NOT. f32 accumulation inside a b = 64 wide panel is the
//!    design, not a defect.
//!  * the multi-chunk test (12 chunks of 10 aux rows): K equals the single-chunk
//!    mixed K to 1.5e-2 of the stage-2 bound; the mutant `p` for `p0 + p` in the
//!    chunk loop gives 1.2e14 of it (and 3.2e6 of the mixed bound against the host).
//!  * the OPERAND-CLASS gate (twin RMS): the twin is the f64 device K of the
//!    f32-rounded B and C; RMS over the upper triangle of |K - K_twin|/(eps*S).
//!    A MEASUREMENT, not a derived bound, identical on GPU 0 and GPU 1 (GTX 1080),
//!    converged RI-JK RHF orbitals, def2-universal-jkfit, b = 64:
//!    ```text
//!      system           n   nocc  clean      plain f32 acc (a)  truncating upload
//!      water           24    5    2.349e-3   2.349e-3 (1 panel)  6.950e-3
//!      butane         106   17    1.769e-4   1.825e-4            7.549e-4
//!      hexane         154   25    1.218e-4   1.702e-4            5.450e-4
//!      octane         202   33    1.280e-4   1.576e-4            4.903e-4
//!    ```
//!    The bar 2.9e-4 is the geometric midpoint of the largest clean value on the
//!    asserted systems (n > b: butane 1.769e-4) and the smallest truncating one
//!    (octane 4.903e-4): 1.6x either side. Plain f32 accumulation (a) is only
//!    1.0-1.4x the clean side and is NOT caught by this gate (the exact pin above
//!    is its guarantee). Water (n < b) is reported, not asserted.
//!  * open-shell: OH alpha 5 / beta 4 / alpha again on one mixed `DfK` (scratch
//!    sized by alpha, reused by beta): no panic, alpha bit-equal on rebuild, each
//!    spin inside its bound (2.8e-2, 2.3e-2). Mutant: copying the whole alpha-sized
//!    per-P panel panics in cudarc (`dst.len() >= src.len()`).
//!  * flush-kernel loss (`FORCE_KERNEL_FAILURE`) before the upload (f64 device
//!    upload, exactly 8*band*n^2 bytes, fallback +1, K bits equal the f64 device K)
//!    and after it (the build errors with `GpuError::Kernel`, k untouched, the slot
//!    declines sticky, the CPU K is the host K); a C_occ element beyond f32::MAX:
//!    `GpuError::F32Range`, k untouched, declined sticky, counted under
//!    `gemm_cpu_f32_range`.
//!
//! The twin-RMS bar and the clean ratios depend on cuBLAS SGEMM's internal order
//! on this device; only the bound, the exact pins and the counts are portable.
//!
//! Also pinned: resident bytes (4*band*n^2 + 4*n*nocc on the first build), mixed
//! panels per build, the allowlist (precision = mixed with only rimp2-energy
//! leaves DF-K f64 and moves no mixed counter), the f64 path's K bits when mixed
//! is not allowed, and the resolver's refusal of dfk-occ in this build.
use ferric_core::basis;
use ferric_core::gpu::config::{GpuSettings, GpuSettingsExplicit};
use ferric_core::gpu::device::FORCE_KERNEL_FAILURE;
use ferric_core::gpu::mixed::{effective_k_panel, MIXED_K_PANEL_DEFAULT};
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

/// Operand-class gate bar (twin RMS), a MEASUREMENT: see the docstring.
const TWIN_RMS_BAR: f64 = 2.9e-4;

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
fn mixed_k_is_inside_its_bound_and_the_operand_class_gate_separates_truncation() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    for (label, xyz) in [
        ("water/cc-pVDZ", WATER.to_string()),
        ("butane/cc-pVDZ", alkane(4)),
        ("hexane/cc-pVDZ", alkane(6)),
        ("octane/cc-pVDZ", alkane(8)),
    ] {
        let (c, _) = scf_occ(&xyz, 1, "cc-pvdz");
        let nocc = c.ncols();
        let (mut dfk0, n) = dfk_for(&xyz, "cc-pvdz");
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
            "{label}: the mixed path did not run (or the panel count is off)"
        );
        let clean = max_ratio(&k_clean, &k_host, &s, eps);

        TRUNCATE_B_TO_F32.store(true, Ordering::SeqCst);
        let k_b = dispatched_k(&xyz, "cc-pvdz", &c);
        TRUNCATE_B_TO_F32.store(false, Ordering::SeqCst);
        let k_twin = twin_k(&flat.view(), &c, None);
        let (tc, tb) = (
            rms_ratio(&k_clean, &k_twin, &s, eps),
            rms_ratio(&k_b, &k_twin, &s, eps),
        );
        // plain f32 accumulation across panels, reported (not asserted: the exact
        // cross-panel pin below is the guarantee)
        let prev = std::env::var("FERRIC_GPU_MIXED_K_PANEL").ok();
        std::env::set_var("FERRIC_GPU_MIXED_K_PANEL", n.to_string());
        let k_a = dispatched_k(&xyz, "cc-pvdz", &c);
        match prev {
            Some(v) => std::env::set_var("FERRIC_GPU_MIXED_K_PANEL", v),
            None => std::env::remove_var("FERRIC_GPU_MIXED_K_PANEL"),
        }
        let ta = rms_ratio(&k_a, &k_twin, &s, eps);
        eprintln!(
            "{label}: n={n} naux={naux} nocc={nocc} b={b} bound_factor={eps:.3e}: \
             bound ratio {clean:.4e}; twin RMS clean {tc:.4e} (a) plain f32 acc {ta:.4e} (b) truncating upload {tb:.4e}"
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
        if n > b {
            assert!(tc < TWIN_RMS_BAR, "{label}: clean twin RMS {tc:e} >= bar");
            assert!(
                tb > TWIN_RMS_BAR,
                "{label}: truncating upload not caught ({tb:e})"
            );
        }
    }
}

/// n = b + 8, band = 1, nocc = 1: C[0] = C[b] = 1, B_0[0,nu*] = 2^24, B_0[b,nu*] = 1.
/// Rows 0 and b fall in different k-panels, each panel sum has exactly one nonzero
/// term, so each panel is exact in f32; the f64 flush gives Y[nu*] = 2^24 + 1 exactly
/// and K[nu*,nu*] = (2^24+1)^2 = 2^48 + 2^25 + 1 (49 bits, exact in f64). Summing the
/// panels in f32 (in any order) rounds 2^24 + 1 to 2^24 and gives 2^48.
#[test]
fn the_f64_flush_across_k_panels_is_pinned_exactly() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let b = MIXED_K_PANEL_DEFAULT;
    let n = b + 8;
    let nu = 5usize;
    let mut flat = Array2::<f64>::zeros((1, n * n));
    let p24 = (1u64 << 24) as f64;
    flat[(0, nu)] = p24; // B_0[0, nu*]
    flat[(0, b * n + nu)] = 1.0; // B_0[b, nu*]
    flat[(0, nu * n)] = p24; // mirrors
    flat[(0, nu * n + b)] = 1.0;
    let mut c = Array2::<f64>::zeros((n, 1));
    c[(0, 0)] = 1.0;
    c[(b, 0)] = 1.0;
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let mut d = DeviceDfK::upload_with_precision(
        &dev,
        &pool().unwrap(),
        &flat.view(),
        n,
        DfkPrecision::Mixed,
    )
    .unwrap();
    assert!(d.is_mixed());
    let s0 = stats();
    let mut k = Array2::zeros((n, n));
    d.build_from_occ(&c.view(), &mut k).unwrap();
    let panels = stats().mixed_panels - s0.mixed_panels;
    assert_eq!(
        panels,
        n.div_ceil(effective_k_panel()) as u64,
        "k_panel override in force? the pin needs b = {b}"
    );
    let want = (1u64 << 48) + (1u64 << 25) + 1;
    assert_eq!(
        k[(nu, nu)].to_bits(),
        (want as f64).to_bits(),
        "K[nu*,nu*] = {:e}, want 2^48 + 2^25 + 1 = {want}",
        k[(nu, nu)]
    );
    let nonzero = k.iter().filter(|&&v| v != 0.0).count();
    assert_eq!(nonzero, 1, "only K[nu*,nu*] is nonzero");
}

/// B_0 = I, nocc = 1: Y = C32 exactly, K_mu,nu = c32_mu * c32_nu, a 48-bit product
/// exact in f64. Pins round-to-nearest-even of C_occ through the real kernel
/// (truncation would differ at 1 + 3*2^-24 and 1 + 2^-24 + 2^-40; the tie 1 + 2^-24
/// goes to the even neighbour 1.0 in both modes, so it pins ties-to-even only
/// against round-half-up).
#[test]
fn c_occ_rounding_is_round_to_nearest_even_bit_for_bit_through_the_kernel() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let vals = [
        1.0 + 3.0 * 2f64.powi(-24),
        1.0 + 2f64.powi(-24),
        1.0 + 2f64.powi(-24) + 2f64.powi(-40),
        -(1.0 + 3.0 * 2f64.powi(-24)),
        1.0 + 5.0 * 2f64.powi(-24), // tie between odd and even neighbours
        0.1,
        1.0 / 3.0,
        std::f64::consts::PI,
        -0.7,
        0.0,
        2f64.powi(-30) * 1.000_000_07,
        123.456_789_012_345,
    ];
    let n = vals.len();
    let mut flat = Array2::<f64>::zeros((1, n * n));
    for i in 0..n {
        flat[(0, i * n + i)] = 1.0;
    }
    let c = Array2::from_shape_fn((n, 1), |(i, _)| vals[i]);
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let mut d = DeviceDfK::upload_with_precision(
        &dev,
        &pool().unwrap(),
        &flat.view(),
        n,
        DfkPrecision::Mixed,
    )
    .unwrap();
    assert!(d.is_mixed());
    let mut k = Array2::zeros((n, n));
    d.build_from_occ(&c.view(), &mut k).unwrap();
    for i in 0..n {
        for j in 0..n {
            let want = (vals[i] as f32 as f64) * (vals[j] as f32 as f64);
            // 0 * (-x) is -0 in the product but SYRK writes +0: zero compares by value
            assert!(
                k[(i, j)] == want && (want == 0.0 || k[(i, j)].to_bits() == want.to_bits()),
                "K[{i},{j}] = {:e}, want {want:e}",
                k[(i, j)]
            );
        }
    }
    // the values really do distinguish the modes
    let trunc = |x: f64| {
        let r = x as f32;
        if f64::from(r).abs() > x.abs() {
            f64::from(f32::from_bits(r.to_bits() - 1))
        } else {
            f64::from(r)
        }
    };
    assert!(vals.iter().any(|&v| trunc(v) != v as f32 as f64));
}

/// Several chunks (a small scratch budget): every chunk must read ITS aux rows. A
/// `p`-for-`p0 + p` offset would make every chunk reuse rows 0..c.
#[test]
fn a_mixed_build_over_many_chunks_matches_the_single_chunk_build() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (c, _) = scf_occ(WATER, 1, "cc-pvdz");
    let nocc = c.ncols();
    let (mut dfk0, n) = dfk_for(WATER, "cc-pvdz");
    let flat = dfk0.dressed_incore_flat_for_test().unwrap().to_owned();
    let band = flat.nrows();
    let s = s_matrix(&flat.view(), n, &c);
    let k_host = host_k(&mut dfk0, &c);
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let pool = pool().unwrap();
    let mk = |bytes: Option<usize>| {
        let mut d =
            DeviceDfK::upload_with_precision(&dev, &pool, &flat.view(), n, DfkPrecision::Mixed)
                .unwrap();
        if let Some(b) = bytes {
            d = d.with_scratch_bytes(b);
        }
        let mut k = Array2::zeros((n, n));
        let plan = d.build_from_occ(&c.view(), &mut k).unwrap();
        (k, plan)
    };
    let (k1, p1) = mk(None);
    let (km, pm) = mk(Some(8 * nocc * n * 10)); // 10 aux rows per chunk
    assert_eq!(p1.nchunks, 1);
    assert_eq!(pm.chunk, 10);
    assert_eq!(pm.nchunks, band.div_ceil(10));
    assert!(pm.nchunks > 5);
    let b = effective_k_panel();
    // identical Y (same panels, same order): the K's differ by stage-2 order only
    let eps2 = k_error_factor(n, band * nocc, 1) + k_error_factor(n, pm.chunk * nocc, pm.nchunks);
    let d12 = max_ratio(&km, &k1, &s, eps2);
    let eps = k_error_factor_mixed(n, b, pm.chunk * nocc, pm.nchunks) + eps_host(n, nocc, band);
    let dh = max_ratio(&km, &k_host, &s, eps);
    eprintln!(
        "water: {} chunks: vs single-chunk mixed {d12:.3e} of the stage-2 bound; vs host {dh:.3e} of the mixed bound",
        pm.nchunks
    );
    assert!(d12 <= 1.0, "many-chunk K != single-chunk K ({d12:e})");
    assert!(dh <= 1.0);
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
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
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
