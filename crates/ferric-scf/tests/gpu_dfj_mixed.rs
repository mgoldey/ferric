#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! Mixed `dfj-pack`: the packed raw tensor `Bp` resident as f32 (nearest-rounded on
//! upload), both RI-J passes by the f32-matrix / f64-vector GEMV (f64 FMA
//! accumulation; `w`, `d_P`, `c`, `J` never rounded). The kernel is NOT in
//! `MixedKernelSet::SHIPPED`: the tests enable it through the real resolver
//! (`GpuSettings::resolve_with_default`, shipped + dfj-pack) and the
//! `MIXED_SETTINGS_OVERRIDE` seam of `df_k_gpu`, which every mixed allowlist decision
//! reads. Mode, device and pool still come from the installed (auto, 3.5 GB) settings.
//!
//! WHAT IS DERIVED AND WHAT IS MEASURED
//!  * DERIVED, per pass, elementwise (`df_j_gpu` module docs): against the f64
//!    tensor `|d_mixed - d| <= eps (|Bp||w|)` with `eps = u32 + gamma_k(u64)(1+u32)`,
//!    plus the f64 reference's own `gamma_k(u64)`; the same for `J = Bp^T c`.
//!    `per_pass_errors_are_inside_the_derived_bounds` asserts every ratio <= 1 and
//!    prints the measured worst ratio (water, C4, C6).
//!  * DERIVED given a MEASURED metric norm: `J = Bp^T V^-1 Bp w`, first order
//!    `dJ = dB^T c + Bp^T V^-1 (dB w)`; with `V = L L^T`, `X = L^-1 Bp` the second term
//!    is `X^T (L^-1 dd)`, `dd = dB w`, `|dd| <= u32 |Bp||w|`, so
//!    `||dJ||_2 <= eps (|| |Bp|^T|c| ||_2 + ||X||_2 ||V^-1||_2^(1/2) || |Bp||w| ||_2)`
//!    with `||X||_2^2 = lambda_max(Bp^T V^-1 Bp)`. `||V^-1||_2` and `||X||_2` are read
//!    off the real Cholesky solve by power iteration (MEASURED inputs); the end to
//!    end J figures (`end_to_end_j_*`) are MEASURED, not a theorem.
//!  * TWIN (sharp, device and order independent): `ROUND_B_TO_F32` uploads the f64
//!    tensor already rounded through f32, so the mixed device and the f64 twin hold
//!    the SAME numbers and differ only by f64 accumulation order (`~1e-16`). Round to
//!    nearest sits inside that bound; `TRUNCATE_B_TO_F32` (the mixed upload rounding
//!    toward zero) sits `~1e8` above it. Per pass and in the SCF energy.
//!
//! Also pinned: the pool charge (4 B/element, labelled), capacity (a pool too small
//! for the f64 tensor holds the f32 one), `F32Range` and kernel-load fallbacks to the
//! f64 device tensor (counted, never the CPU), `FORCE_HOST` bit identity, the
//! allowlist (`precision = mixed` without `dfj-pack` stays f64), tier independence,
//! determinism. The energy error map over alkanes C2..C8 is `#[ignore]`d (minutes).
use ferric_core::basis;
use ferric_core::gpu::config::{GpuSettings, GpuSettingsExplicit};
use ferric_core::gpu::device::{device, GpuError, FORCE_KERNEL_FAILURE};
use ferric_core::gpu::mixed_host::{gamma, U32, U64};
use ferric_core::gpu::{
    install, pool, probe, stats, GpuMode, GpuStatus, MixedKernel, MixedKernelSet, Precision,
};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_scf::df_j::DfJ;
use ferric_scf::df_j_gpu::{
    d_error_factor, d_error_factor_mixed, j_error_factor, j_error_factor_mixed, pair_len,
    resident_bytes, resident_bytes_with, row_plan, DeviceDfJ, DfjPrecision, FORCE_HOST,
    INJECT_F32_RANGE, ROUND_B_TO_F32, TRUNCATE_B_TO_F32,
};
use ferric_scf::df_k_gpu::MIXED_SETTINGS_OVERRIDE;
use ferric_scf::fock::JBuilder;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array1, Array2, ArrayView2};
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
            memory_gb: Some(3.5),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

/// The build's shipped set plus dfj-pack: what the ship commit will make real.
fn shipped_plus_dfj() -> MixedKernelSet {
    MixedKernelSet::SHIPPED.with(MixedKernel::DfjPack)
}

fn resolve(kernels: MixedKernelSet, shipped: MixedKernelSet) -> Result<GpuSettings, String> {
    GpuSettings::resolve_with_default(
        GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            precision: Some(Precision::Mixed),
            mixed_kernels: Some(kernels),
            memory_gb: Some(3.5),
            ..Default::default()
        },
        |_| None,
        Precision::F64,
        shipped,
    )
    .map(|(s, _)| s)
}

/// Resets the override and every seam on drop.
struct Seams;
impl Seams {
    /// `dfj-pack` allowed (through the resolver) for this guard's lifetime.
    fn allow_dfj() -> Self {
        Self::set_allow(true);
        Seams
    }
    /// Switch the `dfj-pack` allowlist on or off without ending the guard.
    fn set_allow(on: bool) {
        let s = on.then(|| resolve(shipped_plus_dfj(), shipped_plus_dfj()).expect("resolve"));
        *MIXED_SETTINGS_OVERRIDE.lock().unwrap() = s;
    }
    /// `precision = mixed` with only the SHIPPED kernels: dfj-pack NOT allowed.
    fn shipped_only() -> Self {
        let s = resolve(MixedKernelSet::SHIPPED, MixedKernelSet::SHIPPED).expect("resolve");
        *MIXED_SETTINGS_OVERRIDE.lock().unwrap() = Some(s);
        Seams
    }
    fn none() -> Self {
        *MIXED_SETTINGS_OVERRIDE.lock().unwrap() = None;
        Seams
    }
}
impl Drop for Seams {
    fn drop(&mut self) {
        *MIXED_SETTINGS_OVERRIDE
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        for f in [
            &FORCE_HOST,
            &ROUND_B_TO_F32,
            &TRUNCATE_B_TO_F32,
            &INJECT_F32_RANGE,
            &ferric_scf::df_j_gpu::FORCE_BUILD_FAILURE,
        ] {
            f.store(false, Ordering::SeqCst);
        }
        FORCE_KERNEL_FAILURE.store(false, Ordering::SeqCst);
    }
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

fn water() -> Molecule {
    Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap()
}

struct Fixture {
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    n: usize,
    naux: usize,
}

impl Fixture {
    fn new(mol: &Molecule, obs_name: &str) -> Self {
        let obs = PreparedBasis::new(mol, &basis::bundled(obs_name).unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let (n, naux) = (obs.nbasis(), dfbs.nbasis());
        Self { obs, dfbs, n, naux }
    }
    fn packed_budget(&self) -> usize {
        let unpacked = self.naux * self.n * self.n * 8;
        let packed = self.naux * (self.n * (self.n + 1) / 2) * 8;
        (unpacked + packed) / 2
    }
    /// The packed raw tensor `(naux, n(n+1)/2)` as the host holds it, streamed from a
    /// spilled source (every tier stores the same numbers bit for bit).
    fn packed_tensor(&self) -> Array2<f64> {
        let tiny = self.n * self.n * 8 * 40;
        let mut src = ThreeIndexSource::build_band(
            Operator::coulomb(),
            &self.obs,
            &self.dfbs,
            tiny,
            0,
            self.naux,
        )
        .unwrap();
        let pair = pair_len(self.n).unwrap();
        let mut out = Array2::<f64>::zeros((self.naux, pair));
        src.for_each_packed_block(1, |blk| {
            let rows = blk.data.nrows();
            out.slice_mut(ndarray::s![blk.p0..blk.p0 + rows, ..])
                .assign(&blk.data);
            Ok(())
        })
        .unwrap();
        out
    }
    fn dfj(&self, budget: usize) -> DfJ<'static> {
        DfJ::new(Operator::coulomb(), &self.obs, &self.dfbs, budget).unwrap()
    }
}

fn density(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| 0.01 * ((i * 7 + j * 3) % 11) as f64)
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

fn bits(m: &Array2<f64>) -> Vec<u64> {
    m.iter().map(|v| v.to_bits()).collect()
}

/// `y = Bp w` and `S = |Bp||w|` (sequential f64).
fn host_d(bp: &ArrayView2<f64>, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (naux, pair) = bp.dim();
    let (mut y, mut s) = (vec![0.0; naux], vec![0.0; naux]);
    for p in 0..naux {
        for k in 0..pair {
            let t = bp[(p, k)] * w[k];
            y[p] += t;
            s[p] += t.abs();
        }
    }
    (y, s)
}

/// `y = Bp^T c` and `S = |Bp|^T |c|` (sequential f64).
fn host_j(bp: &ArrayView2<f64>, c: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (naux, pair) = bp.dim();
    let (mut y, mut s) = (vec![0.0; pair], vec![0.0; pair]);
    for p in 0..naux {
        for k in 0..pair {
            let t = bp[(p, k)] * c[p];
            y[k] += t;
            s[k] += t.abs();
        }
    }
    (y, s)
}

/// Worst `|got - want| / (eps S)`.
fn worst(got: &[f64], want: &[f64], s: &[f64], eps: f64) -> f64 {
    got.iter()
        .zip(want)
        .zip(s)
        .map(|((g, w), s)| (g - w).abs() / (eps * s))
        .fold(0.0, f64::max)
}

fn upload(bp: &ArrayView2<f64>, precision: DfjPrecision) -> DeviceDfJ {
    let dev = device(0).unwrap();
    DeviceDfJ::upload_packed_with_precision(&dev, &pool().unwrap(), bp, precision).unwrap()
}

/// The per-pass statements for one system; returns the worst ratios
/// `(d_exact, j_exact, d_twin, j_twin, d_trunc, j_trunc)`.
fn per_pass(mol: &Molecule) -> [f64; 6] {
    let f = Fixture::new(mol, "def2-svp");
    let bp_owned = f.packed_tensor();
    let bp: ArrayView2<f64> = bp_owned.view();
    let (naux, pair) = bp.dim();
    assert_eq!(pair, pair_len(f.n).unwrap());
    let w = weights(&density(f.n), f.n);
    let c: Vec<f64> = (0..naux)
        .map(|p| ((p * 13) % 17) as f64 / 17.0 - 0.4)
        .collect();
    let bphat = bp.mapv(|x| f64::from(x as f32));
    // the f64 twin: same numbers as the mixed tensor, f64 storage
    ROUND_B_TO_F32.store(true, Ordering::SeqCst);
    let mut twin = upload(&bp, DfjPrecision::F64);
    ROUND_B_TO_F32.store(false, Ordering::SeqCst);
    let (d_tw, j_tw) = (twin.d_p(&w).unwrap(), twin.j_packed(&c).unwrap());
    let mut mixed = upload(&bp, DfjPrecision::Mixed);
    assert!(mixed.is_mixed());
    let (d_mx, j_mx) = (mixed.d_p(&w).unwrap(), mixed.j_packed(&c).unwrap());
    TRUNCATE_B_TO_F32.store(true, Ordering::SeqCst);
    let mut trunc = upload(&bp, DfjPrecision::Mixed);
    TRUNCATE_B_TO_F32.store(false, Ordering::SeqCst);
    let (d_tr, j_tr) = (trunc.d_p(&w).unwrap(), trunc.j_packed(&c).unwrap());

    let (d_ref, s_d) = host_d(&bp, &w);
    let (j_ref, s_j) = host_j(&bp, &c);
    let (_, s_dh) = host_d(&bphat.view(), &w);
    let (_, s_jh) = host_j(&bphat.view(), &c);
    let (rows, calls) = row_plan(naux, pair);
    let eps_d = d_error_factor_mixed(pair) + gamma(pair, U64);
    let eps_j = j_error_factor_mixed(naux) + gamma(naux, U64);
    // twin: f64 accumulation on both sides (mixed kernel + the cuBLAS twin)
    let eps_dt = gamma(pair, U64) * (1.0 + U32) + d_error_factor(pair);
    let eps_jt = gamma(naux, U64) * (1.0 + U32) + j_error_factor(rows, calls);
    [
        worst(&d_mx, &d_ref, &s_d, eps_d),
        worst(&j_mx, &j_ref, &s_j, eps_j),
        worst(&d_mx, &d_tw, &s_dh, eps_dt),
        worst(&j_mx, &j_tw, &s_jh, eps_jt),
        worst(&d_tr, &d_tw, &s_dh, eps_dt),
        worst(&j_tr, &j_tw, &s_jh, eps_jt),
    ]
}

#[test]
fn per_pass_errors_are_inside_the_derived_bounds() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let _s = Seams::none();
    for (name, mol) in [("water", water()), ("C4", alkane(4)), ("C6", alkane(6))] {
        let r = per_pass(&mol);
        eprintln!(
            "{name}: ratio to bound  exact d {:.3e} J {:.3e} | twin d {:.3e} J {:.3e} | truncated-vs-twin d {:.3e} J {:.3e}",
            r[0], r[1], r[2], r[3], r[4], r[5]
        );
        assert!(r[0] <= 1.0 && r[1] <= 1.0, "{name}: exact bound violated");
        assert!(r[2] <= 1.0 && r[3] <= 1.0, "{name}: twin bound violated");
        assert!(
            r[4] > 1.0e3 && r[5] > 1.0e3,
            "{name}: truncation is not visible to the twin bound"
        );
    }
}

/// Power iteration on `V^-1` through the real Cholesky solve: `||V^-1||_2`.
fn v_inv_norm(dfj: &DfJ, naux: usize) -> f64 {
    let mut v = Array1::from_shape_fn(naux, |i| 1.0 + 0.001 * (i % 7) as f64);
    let mut lam = 0.0;
    for _ in 0..60 {
        let nv = v.dot(&v).sqrt();
        v /= nv;
        let u = dfj.solve_metric_for_test(&v).unwrap();
        lam = v.dot(&u);
        v = u;
    }
    lam
}

/// `||X||_2` with `X = L^-1 Bp` (`V = L L^T`): `||X||_2^2 = lambda_max(Bp^T V^-1 Bp)`, the
/// largest eigenvalue of the fitted Coulomb operator on the packed pair space, by
/// power iteration through the real solve.
fn fit_operator_norm(dfj: &DfJ, bp: &ArrayView2<f64>) -> f64 {
    let pair = bp.ncols();
    let mut v: Vec<f64> = (0..pair).map(|i| 1.0 + 0.001 * (i % 7) as f64).collect();
    let mut lam = 0.0;
    for _ in 0..60 {
        let nv = l2(&v);
        v.iter_mut().for_each(|x| *x /= nv);
        let (d, _) = host_d(bp, &v);
        let c = dfj.solve_metric_for_test(&Array1::from(d)).unwrap();
        let (u, _) = host_j(bp, &c.to_vec());
        lam = v.iter().zip(&u).map(|(a, b)| a * b).sum::<f64>();
        v = u;
    }
    lam.sqrt()
}

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// End to end through `DfJ::build`: mixed device J against the host J, the derived
/// normwise bound (with the measured `||V^-1||`), and the twin.
fn end_to_end(name: &str, mol: &Molecule) {
    Seams::set_allow(true);
    let f = Fixture::new(mol, "def2-svp");
    let d = density(f.n);
    let mut dfj = f.dfj(usize::MAX);
    let bp = f.packed_tensor();
    let (naux, pair) = bp.dim();
    let w = weights(&d, f.n);
    let (d_ref, s_dw) = host_d(&bp.view(), &w);
    let c_ref = dfj
        .solve_metric_for_test(&Array1::from(d_ref.clone()))
        .unwrap();
    let c_slice = c_ref.to_vec();
    let (_, t1v) = host_j(&bp.view(), &c_slice);
    let (t1, bw) = (l2(&t1v), l2(&s_dw));
    let vinv = v_inv_norm(&dfj, naux);
    let x_norm = fit_operator_norm(&dfj, &bp.view());
    let n = f.n;

    FORCE_HOST.store(true, Ordering::SeqCst);
    let mut j_host = Array2::zeros((n, n));
    dfj.build(&d, &mut j_host).unwrap();
    FORCE_HOST.store(false, Ordering::SeqCst);

    let s0 = stats();
    let mut j_mix = Array2::zeros((n, n));
    dfj.build(&d, &mut j_mix).unwrap();
    let s1 = stats();
    assert_eq!(
        s1.dfj_device_builds,
        s0.dfj_device_builds + 1,
        "{name}: not on the device"
    );
    assert_eq!(
        (s1.bytes_h2d - s0.bytes_h2d) as usize,
        4 * naux * pair + 8 * (pair + naux),
        "{name}: the cold build must upload exactly 4 B per tensor element"
    );
    assert_eq!(j_mix, j_mix.t(), "{name}: J must be exactly symmetric");

    // packed lower triangle of dJ
    let mut dj = Vec::new();
    for mu in 0..n {
        for nu in 0..=mu {
            dj.push(j_mix[(mu, nu)] - j_host[(mu, nu)]);
        }
    }
    let dj_norm = l2(&dj);
    let eps = d_error_factor_mixed(pair).max(j_error_factor_mixed(naux))
        + 2.0 * gamma(pair.max(naux), U64);
    let t2 = x_norm * vinv.sqrt() * bw;
    let bound = 1.01 * eps * (t1 + t2);
    let jn = l2(&host_j(&bp.view(), &c_slice).0);
    eprintln!(
        "{name}: n {n} naux {naux} pair {pair}  ||V^-1|| {vinv:.3e}  ||dJ||2 {dj_norm:.3e} (rel {:.3e})  derived normwise bound {bound:.3e}  measured/bound {:.3e}  [T1 {t1:.3e}  T2 {t2:.3e}  ||X|| {x_norm:.3e}]",
        dj_norm / jn,
        dj_norm / bound
    );
    assert!(
        dj_norm <= bound,
        "{name}: measured exceeds the derived bound"
    );
    assert!(
        dj_norm > 0.0,
        "{name}: mixed J equals f64 J: the f32 tensor was not used"
    );

    // the twin (same numbers, f64 storage) agrees to the f64 floor; truncation does not
    drop(dfj);
    ROUND_B_TO_F32.store(true, Ordering::SeqCst);
    let mut dfj_tw = f.dfj(usize::MAX);
    let mut j_tw = Array2::zeros((n, n));
    Seams::set_allow(false); // twin arm: f64 device
    dfj_tw.build(&d, &mut j_tw).unwrap();
    ROUND_B_TO_F32.store(false, Ordering::SeqCst);
    let rel = |a: &Array2<f64>, b: &Array2<f64>| {
        (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()))
            / b.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    };
    let clean = rel(&j_mix, &j_tw);
    Seams::set_allow(true);
    TRUNCATE_B_TO_F32.store(true, Ordering::SeqCst);
    let mut dfj_tr = f.dfj(usize::MAX);
    let mut j_tr = Array2::zeros((n, n));
    dfj_tr.build(&d, &mut j_tr).unwrap();
    TRUNCATE_B_TO_F32.store(false, Ordering::SeqCst);
    let defect = rel(&j_tr, &j_tw);
    eprintln!(
        "{name}: max|dJ|/max|J| mixed-vs-twin {clean:.3e}  truncated-vs-twin {defect:.3e}  separation {:.3e}",
        defect / clean.max(1e-300)
    );
    assert!(
        defect > 1e3 * clean,
        "{name}: truncation not separated from the clean side"
    );
}

#[test]
fn end_to_end_j_mixed_vs_f64_water_c4_c6() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let _s = Seams::none();
    for (name, mol) in [("water", water()), ("C4", alkane(4)), ("C6", alkane(6))] {
        end_to_end(name, &mol);
    }
}

#[test]
fn mixed_j_is_tier_independent_deterministic_and_symmetric() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let _s = Seams::allow_dfj();
    let f = Fixture::new(&water(), "def2-svp");
    let d = density(f.n);
    let tiny = f.n * f.n * 8 * 40;
    let mut first: Option<Vec<u64>> = None;
    for (name, budget) in [
        ("unpacked in-core", usize::MAX),
        ("mid budget", f.packed_budget()),
        ("spilled", tiny),
    ] {
        let mut dfj = f.dfj(budget);
        let (mut j1, mut j2) = (Array2::zeros((f.n, f.n)), Array2::zeros((f.n, f.n)));
        dfj.build(&d, &mut j1).unwrap();
        dfj.build(&d, &mut j2).unwrap();
        assert!(dfj.device_resident_for_test(), "{name}");
        assert_eq!(bits(&j1), bits(&j2), "{name}: not deterministic");
        match &first {
            None => first = Some(bits(&j1)),
            Some(b) => assert_eq!(b, &bits(&j1), "{name}: differs from the first tier"),
        }
    }
}

#[test]
fn pool_charge_is_four_bytes_per_element_labelled_and_fits_where_f64_does_not() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let _s = Seams::none();
    let f = Fixture::new(&alkane(4), "def2-svp");
    let bp_owned = f.packed_tensor();
    let bp: ArrayView2<f64> = bp_owned.view();
    let (naux, pair) = bp.dim();
    let pl = pool().unwrap();
    let cap = pl.capacity_bytes();
    assert_eq!(pl.available_bytes(), cap, "another test left a reservation");
    let vecs = 8 * (2 * pair + 2 * naux);
    let a = stats();
    let mixed = upload(&bp, DfjPrecision::Mixed);
    let b = stats();
    assert_eq!(cap - pl.available_bytes(), 4 * naux * pair + vecs);
    assert_eq!(
        resident_bytes_with(naux, pair, DfjPrecision::Mixed).unwrap(),
        4 * naux * pair + vecs
    );
    assert_eq!(
        (b.bytes_h2d - a.bytes_h2d) as usize,
        4 * naux * pair,
        "the f32 tensor moves 4 B per element"
    );
    assert!(
        pl.occupancy_report().contains("DF-J raw B (packed, f32)"),
        "{}",
        pl.occupancy_report()
    );
    drop(mixed);
    assert_eq!(pl.available_bytes(), cap, "the charge is released on drop");
    let f64s = upload(&bp, DfjPrecision::F64);
    assert_eq!(cap - pl.available_bytes(), 8 * naux * pair + vecs);
    assert_eq!(resident_bytes(naux, pair).unwrap(), 8 * naux * pair + vecs);
    drop(f64s);

    // capacity: leave room for the f32 state and not one byte more than it needs
    let need = resident_bytes_with(naux, pair, DfjPrecision::Mixed).unwrap();
    let hog = pl.reserve("test hog", cap - need).unwrap();
    let dev = device(0).unwrap();
    let refused = DeviceDfJ::upload_packed(&dev, &pl, &bp);
    assert!(
        matches!(refused, Err(GpuError::PoolFull { .. })),
        "the f64 tensor must not fit"
    );
    assert_eq!(pl.available_bytes(), need, "the refusal leaked");
    let fits = upload(&bp, DfjPrecision::Mixed);
    assert_eq!(pl.available_bytes(), 0);
    drop((fits, hog));
    assert_eq!(pl.available_bytes(), cap);
}

#[test]
fn f32_range_overflow_is_a_typed_refusal_that_leaks_nothing() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let _s = Seams::none();
    let dev = device(0).unwrap();
    let pl = pool().unwrap();
    let mut t = Array2::<f64>::ones((4, 6));
    t[(2, 3)] = 1e300;
    let r = DeviceDfJ::upload_packed_with_precision(&dev, &pl, &t.view(), DfjPrecision::Mixed);
    assert!(
        matches!(r, Err(GpuError::F32Range(ref m)) if m.contains("DF-J raw B")),
        "{:?}",
        r.err()
    );
    assert_eq!(pl.available_bytes(), pl.capacity_bytes(), "leaked");
    // f64 residency holds the same tensor
    let ok = DeviceDfJ::upload_packed(&dev, &pl, &t.view()).unwrap();
    assert!(!ok.is_mixed());
}

/// The dispatcher's fallback: an `F32Range` refusal or a kernel that will not load
/// runs the f64 DEVICE tensor for that `DfJ` (counted once, never the CPU) and
/// the J is exactly the f64 device J.
#[test]
fn f32_range_and_kernel_loss_fall_back_to_the_f64_device_tensor_and_count_it() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let f = Fixture::new(&water(), "def2-svp");
    let d = density(f.n);
    let n = f.n;
    let build = |setup: &dyn Fn()| {
        let mut dfj = f.dfj(f.packed_budget());
        setup();
        let a = stats();
        let mut j = Array2::zeros((n, n));
        dfj.build(&d, &mut j).unwrap();
        let b = stats();
        (j, a, b, dfj.device_resident_for_test())
    };
    let (j64, ..) = {
        let _s = Seams::none();
        build(&|| {})
    };
    for (name, inject) in [("F32Range", 0usize), ("kernel load", 1)] {
        let _s = Seams::allow_dfj();
        let (j, a, b, resident) = build(&|| {
            if inject == 0 {
                INJECT_F32_RANGE.store(true, Ordering::SeqCst);
            } else {
                FORCE_KERNEL_FAILURE.store(true, Ordering::SeqCst);
            }
        });
        assert!(resident, "{name}: the J must stay on the device");
        assert_eq!(b.mixed_fallback_f64, a.mixed_fallback_f64 + 1, "{name}");
        assert_eq!(b.dfj_device_builds, a.dfj_device_builds + 1, "{name}");
        assert_eq!(b.dfj_declined, a.dfj_declined, "{name}: not a CPU decline");
        assert_eq!(
            b.gemm_cpu_f32_range, a.gemm_cpu_f32_range,
            "{name}: not a CPU fallback"
        );
        assert_eq!(bits(&j), bits(&j64), "{name}: must equal the f64 device J");
        assert_eq!(
            pool().unwrap().available_bytes(),
            pool().unwrap().capacity_bytes(),
            "{name}: the dropped DfJ must release the pool"
        );
    }
}

#[test]
fn force_host_is_the_host_path_and_precision_mixed_alone_does_not_turn_it_on() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let f = Fixture::new(&water(), "def2-svp");
    let d = density(f.n);
    let n = f.n;
    // host reference with the default settings
    let host = |_s: &Seams| {
        FORCE_HOST.store(true, Ordering::SeqCst);
        let mut dfj = f.dfj(f.packed_budget());
        let mut j = Array2::zeros((n, n));
        dfj.build(&d, &mut j).unwrap();
        FORCE_HOST.store(false, Ordering::SeqCst);
        j
    };
    let j_default = host(&Seams::none());
    let a = stats();
    let j_mixed_host = host(&Seams::allow_dfj());
    assert_eq!(stats(), a, "FORCE_HOST must not move any counter");
    assert_eq!(bits(&j_default), bits(&j_mixed_host));

    // precision = mixed with the shipped kernels only: the device tensor stays f64
    let _s = Seams::shipped_only();
    let mut dfj = f.dfj(usize::MAX);
    let mut j = Array2::zeros((n, n));
    let a = stats();
    dfj.build(&d, &mut j).unwrap();
    let b = stats();
    let (naux, pair) = (f.naux, pair_len(n).unwrap());
    assert_eq!(
        (b.bytes_h2d - a.bytes_h2d) as usize,
        8 * naux * pair + 8 * (pair + naux),
        "dfj-pack not in the allowlist: f64 residency (8 B per element)"
    );
    assert_eq!(b.mixed_fallback_f64, a.mixed_fallback_f64);
}

// ---------------------------------------------------------------- SCF energy

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    /// f64 J on the host (the reference)
    HostJ,
    /// mixed device J, nearest rounding
    Mixed,
    /// f64 device J of the f32-rounded tensor (the twin)
    Twin,
    /// mixed device J, truncating upload (the defect)
    Truncated,
}

/// (energy, iterations, converged)
type Run = (f64, usize, bool);

fn rhf(mol: &Molecule, prep: &PreparedBasis, arm: Arm) -> Run {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-3, // a loose "not still descending" bound; dP is the real gate
        density_conv: 1e-8,
        max_iter: 200,
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let _s = if matches!(arm, Arm::Mixed | Arm::Truncated) {
        Seams::allow_dfj()
    } else {
        Seams::none()
    };
    FORCE_HOST.store(arm == Arm::HostJ, Ordering::SeqCst);
    ROUND_B_TO_F32.store(arm == Arm::Twin, Ordering::SeqCst);
    TRUNCATE_B_TO_F32.store(arm == Arm::Truncated, Ordering::SeqCst);
    let r = solve_rhf(&ParallelContext::default(), mol, prep, op, &bounds, &cfg);
    match r {
        Ok(res) => (res.energy, res.iterations, res.converged),
        Err(FerricError::ScfConvergence { last_energy, .. }) => (last_energy, cfg.max_iter, false),
        Err(e) => panic!("unexpected SCF error: {e:?}"),
    }
}

fn prep(mol: &Molecule, name: &str) -> PreparedBasis {
    PreparedBasis::new(mol, &basis::bundled(name).unwrap()).unwrap()
}

/// Clean side: |E_mixed - E_twin| (same numbers, only f64 summation order differs).
/// Defect side: |E_truncated - E_twin|. `ENERGY_TOL` is the geometric midpoint of the
/// largest clean and smallest defect value over the asserted systems; the
/// separation is pre-registered at `MIN_SEPARATION`.
///
/// MEASURED (RHF, converged in 12-13 iterations on every arm, GTX 1080):
/// ```text
///   system               clean (mixed)   defect (truncated)   separation
///   water/cc-pVDZ        2.8e-14         3.0e-6               1.1e8
///   C2H6/def2-SVP        4.3e-14         8.8e-6               2.1e8
/// ```
/// sqrt(4.3e-14 * 3.0e-6) = 3.6e-10, written as 4e-10 (9e3 above the clean side,
/// 7e3 below the defect side).
const ENERGY_TOL: f64 = 4e-10;
const MIN_SEPARATION: f64 = 10.0;

fn two_sided(label: &str, mol: &Molecule, p: &PreparedBasis) -> (f64, f64) {
    let (e_tw, it_t, c_t) = rhf(mol, p, Arm::Twin);
    let s0 = stats();
    let (e_mx, it_m, c_m) = rhf(mol, p, Arm::Mixed);
    let s1 = stats();
    assert!(
        s1.dfj_device_builds > s0.dfj_device_builds,
        "{label}: the mixed arm never built J on the device"
    );
    assert_eq!(s1.mixed_fallback_f64, s0.mixed_fallback_f64, "{label}");
    let (e_tr, it_x, c_x) = rhf(mol, p, Arm::Truncated);
    let (clean, defect) = ((e_mx - e_tw).abs(), (e_tr - e_tw).abs());
    eprintln!(
        "{label}: twin {e_tw:.12} ({it_t} it, conv {c_t}); mixed {e_mx:.12} ({it_m} it, conv {c_m}); truncated {e_tr:.12} ({it_x} it, conv {c_x})"
    );
    eprintln!(
        "{label}: |dE| clean {clean:.3e}  defect {defect:.3e}  separation {:.3e}",
        defect / clean.max(1e-300)
    );
    assert!(c_t && c_m && c_x, "{label}: not converged");
    (clean, defect)
}

#[test]
fn scf_energy_two_sided_gate_clean_mixed_vs_truncating_defect() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let w = water();
    let c2 = alkane(2);
    let rows = [
        (
            "RHF water/cc-pVDZ",
            two_sided("RHF water/cc-pVDZ", &w, &prep(&w, "cc-pvdz")),
        ),
        (
            "RHF C2H6/def2-SVP",
            two_sided("RHF C2H6/def2-SVP", &c2, &prep(&c2, "def2-svp")),
        ),
    ];
    let max_clean = rows.iter().map(|r| r.1 .0).fold(0.0, f64::max);
    let min_defect = rows.iter().map(|r| r.1 .1).fold(f64::INFINITY, f64::min);
    eprintln!(
        "max clean {max_clean:.3e}  min defect {min_defect:.3e}  geometric midpoint {:.3e}",
        (max_clean.max(1e-300) * min_defect).sqrt()
    );
    for (label, (clean, defect)) in rows {
        assert!(
            clean <= ENERGY_TOL,
            "{label}: clean {clean:e} above the bar"
        );
        assert!(
            defect > ENERGY_TOL,
            "{label}: defect {defect:e} below the bar"
        );
        assert!(
            defect >= MIN_SEPARATION * clean,
            "{label}: separation {:.2} below the pre-registered {MIN_SEPARATION}",
            defect / clean.max(1e-300)
        );
    }
}

const HARTREE_TO_KCAL: f64 = 627.509_474_063;

/// OLS of y on x: (slope, standard error, degrees of freedom).
fn ols(x: &[f64], y: &[f64]) -> (f64, f64, usize) {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let slope = sxy / sxx;
    let rss: f64 = x
        .iter()
        .zip(y)
        .map(|(a, b)| {
            let r = b - my - slope * (a - mx);
            r * r
        })
        .sum();
    let df = x.len() - 2;
    (slope, (rss / df as f64 / sxx).sqrt(), df)
}

/// The error map (REPORT, no ship claim): |E_mixed - E_hostJ| for alkanes C2..C8
/// in Eh, kcal/mol and per atom, with the ln-ln slope against the atom count.
/// The ship bars are asserted per row only where they are bars of the plan:
/// |dE| <= 1.0e-3 Eh. The slope and the 1.8 uEh (Ochsenfeld) comparison are printed.
#[test]
#[ignore = "precondition: run serially on a quiet box; minutes of host integral work"]
fn energy_error_map_alkanes_c2_to_c8() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut ln_n, mut ln_e) = (Vec::new(), Vec::new());
    eprintln!("system   atoms   nbf   E_host(J f64)        dE (Eh)       dE (kcal/mol)   dE/atom (Eh)   dE/|E|        vs 1.8 uEh");
    for nc in [2usize, 4, 6, 8] {
        let mol = alkane(nc);
        let atoms = 3 * nc + 2;
        let p = prep(&mol, "def2-svp");
        let (e_ref, it_r, c_r) = rhf(&mol, &p, Arm::HostJ);
        let s0 = stats();
        let (e_mx, it_m, c_m) = rhf(&mol, &p, Arm::Mixed);
        let s1 = stats();
        assert!(c_r && c_m, "C{nc}: not converged");
        assert!(
            s1.dfj_device_builds > s0.dfj_device_builds,
            "C{nc}: not on device"
        );
        assert_eq!(
            s1.mixed_fallback_f64, s0.mixed_fallback_f64,
            "C{nc}: fell back"
        );
        let de = e_mx - e_ref;
        eprintln!(
            "C{nc}H{}   {atoms:>4}   {:>4}   {e_ref:.10}   {:+.3e}   {:+.3e}   {:+.3e}   {:+.3e}   {:.2}x   ({it_r}/{it_m} it)",
            2 * nc + 2,
            p.nbasis(),
            de,
            de * HARTREE_TO_KCAL,
            de / atoms as f64,
            de / e_ref.abs(),
            de.abs() / 1.8e-6
        );
        assert!(
            de.abs() <= 1.0e-3,
            "C{nc}: |dE| {:e} above the 1.0e-3 Eh bar",
            de.abs()
        );
        ln_n.push((atoms as f64).ln());
        ln_e.push(de.abs().max(1e-300).ln());
    }
    let (slope, se, df) = ols(&ln_n, &ln_e);
    eprintln!("ln|dE| vs ln(atoms): slope {slope:.3} +- {se:.3} (SE), df {df}");
    let per_atom: Vec<f64> = ln_n.iter().zip(&ln_e).map(|(n, e)| e - n).collect();
    let (slope_pa, se_pa, _) = ols(&ln_n, &per_atom);
    eprintln!("ln(|dE|/atoms) vs ln(atoms): slope {slope_pa:.3} +- {se_pa:.3} (SE), df {df}");
}
