//! The internal contraction rotation of the direct four-centre J/K builder
//! (`DirectJK`, `FERRIC_GC_ROTATE`; `ferric_integrals::contraction_rotation`).
//!
//! * Tolerance: rotated (both stages, and stage 1 alone) against unrotated
//!   J and K, closed and open shell, full and incremental, Schwarz and CSB
//!   screening: max |Δ| ≤ 1e-11 at the production `integral_thresh` (1e-12).
//!   SCF energies (RHF, UHF and ROHF, cc-pVDZ / cc-pVTZ / aug-cc-pVDZ, d and f
//!   shells; TRAH and Newton forced on) within 1e-11 Ha (measured at most
//!   1.2e-12, butane at default convergence). Through the entry points that
//!   follow the SCF's bounds: RHF, UHF, ROHF, RKS and UKS (PBE0) analytic
//!   gradients within 1e-10 Eh/bohr (measured at most 1.1e-11), the RHF
//!   Hessian (its CPHF) within 1e-9, stability eigenvalues within 1e-11, the
//!   KS Z-vector (see `off_config_reaches_the_consumers_bitwise`).
//! * The rule of who rotates: config `off` recorded on the bounds is
//!   bitwise the consumers on fresh bounds (`main`) and bitwise an unset
//!   config under `FERRIC_GC_ROTATE=off` (a child process with the variable
//!   set).
//!   The cases include a transition metal (Cu, cc-pVDZ), ECP bases
//!   (cc-pVDZ-PP on Cu and Ag), a ghost atom, a mixed per-atom basis and a
//!   Cartesian generally contracted basis.
//! * Rounding level: at threshold 0 the rotated build equals the dense
//!   `T J'(Tᵀ D T) Tᵀ` from the rotated basis's own unrotated walk to a few
//!   ulps of `‖T‖∞² max|X|`, so a transform defect of 1e-12 or more fails.
//! * Bitwise: a segmented basis (def2-SVP, 6-31G*, STO-3G) under `Auto` is
//!   the unrotated builder bit for bit; `Off` never rotates; the rotated
//!   builder is bitwise identical across thread counts; a symmetric density
//!   gives J and K as exactly symmetric as the unrotated builder's.
//!
//! The SCF tests choose the rotation through `RhfConfig::gc_rotation`, never
//! by setting `FERRIC_GC_ROTATE`, so no test mutates its own environment
//! (the variable is set only on child processes).

use ferric_core::basis;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::contraction_rotation::{RotatedBasis, RotationLevel};
use ferric_integrals::operator::Operator;
use ferric_scf::direct_jk::{DirectJK, GcRotate};
use ferric_scf::incore_jk::EriMode;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::{SchwarzBounds, ScreeningKind};
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const THRESH: f64 = 1e-12;
const JK_TOL: f64 = 1e-11;
const E_TOL: f64 = 1e-11;
/// Analytic gradients, rotated against unrotated (Eh/bohr).
const GRAD_TOL: f64 = 1e-10;

fn load(stem: &str, multiplicity: usize) -> Molecule {
    let text = std::fs::read_to_string(format!("../../testdata/molecules/{stem}.xyz"))
        .unwrap_or_else(|e| panic!("cannot read {stem}.xyz: {e}"));
    Molecule::parse_xyz(&text, 0, multiplicity)
        .unwrap_or_else(|e| panic!("cannot load {stem}.xyz: {e}"))
}

/// A symmetric test density with O(0.1) elements and structure that differs
/// between `seed`s (so α ≠ β in the open-shell builds).
fn density(n: usize, seed: usize) -> Array2<f64> {
    let d = Array2::from_shape_fn((n, n), |(i, j)| {
        let v = ((i * (7 + seed) + j * (3 + 2 * seed)) % (11 + seed)) as f64;
        0.02 * v * (-0.01 * (i as f64 - j as f64).abs()).exp() - 0.03
    });
    0.5 * (&d + &d.t())
}

fn max_abs(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    // NaN-safe: `f64::max` would skip a NaN difference and pass any bar.
    assert_eq!(a.dim(), b.dim());
    a.iter().zip(b.iter()).fold(0.0f64, |m, (x, y)| {
        let d = (x - y).abs();
        assert!(d.is_finite(), "non-finite difference: {x} vs {y}");
        m.max(d)
    })
}

/// The guard in [`max_abs`] trips: a NaN on either side panics instead of
/// reading as a zero difference.
#[test]
fn max_abs_refuses_a_nan_difference() {
    let a = Array2::from_elem((2, 2), 1.0);
    assert_eq!(max_abs(&a, &a), 0.0);
    for (i, bad) in [f64::NAN, f64::INFINITY].into_iter().enumerate() {
        let mut b = a.clone();
        b[(1, i)] = bad;
        for (x, y) in [(&a, &b), (&b, &a)] {
            let r = std::panic::catch_unwind(|| max_abs(x, y));
            let msg = r.expect_err("max_abs accepted a non-finite difference");
            let msg = msg
                .downcast_ref::<String>()
                .map(String::as_str)
                .unwrap_or_default();
            assert!(msg.contains("non-finite difference"), "{msg}");
        }
    }
}

fn bits_equal(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Closed-shell J, K from a fresh builder with rotation `mode`.
fn jk(
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
    mode: GcRotate,
) -> (Array2<f64>, Array2<f64>, Option<usize>) {
    let ctx = ParallelContext::default();
    let n = prep.nbasis();
    let mut djk = DirectJK::new(&ctx, prep, bounds, THRESH, usize::MAX).with_gc_rotation(mode);
    let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    djk.build(d, &mut j, &mut k).unwrap();
    (j, k, djk.gc_rotated_columns())
}

/// Open-shell J, K_α, K_β from a fresh builder with rotation `mode`.
fn jk_uhf(
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    mode: GcRotate,
) -> [Array2<f64>; 3] {
    let ctx = ParallelContext::default();
    let n = prep.nbasis();
    let mut djk = DirectJK::new(&ctx, prep, bounds, THRESH, usize::MAX).with_gc_rotation(mode);
    let mut out = [
        Array2::zeros((n, n)),
        Array2::zeros((n, n)),
        Array2::zeros((n, n)),
    ];
    let [j, ka, kb] = &mut out;
    djk.build_uhf(&(d_a + d_b), d_a, d_b, j, ka, kb).unwrap();
    out
}

fn parse(xyz: &str) -> Molecule {
    Molecule::parse_xyz(xyz, 0, 1).unwrap()
}

/// `bs` with every shell Cartesian.
fn cartesian(mut bs: BasisSet) -> BasisSet {
    for shells in bs.shells.values_mut() {
        for sh in shells {
            sh.pure = false;
        }
    }
    bs.name = format!("{} (cartesian)", bs.name);
    bs
}

/// `bs` with the shells of `z` taken from `other`.
fn mixed(mut bs: BasisSet, z: i32, other: &str) -> BasisSet {
    let o = basis::bundled(other).unwrap();
    bs.shells.insert(z, o.shells[&z].clone());
    bs.name = format!("{} + {other} on Z={z}", bs.name);
    bs
}

/// The J/K test systems: label, molecule, basis, screening.
fn jk_cases() -> Vec<(String, Molecule, BasisSet, ScreeningKind)> {
    let b = |n: &str| basis::bundled(n).unwrap();
    let mut v = Vec::new();
    for (stem, name, kind) in [
        ("water", "cc-pvdz", ScreeningKind::Schwarz),
        ("water", "cc-pvtz", ScreeningKind::Schwarz),
        ("c2h4", "cc-pvdz", ScreeningKind::Schwarz),
        ("c2h4", "cc-pvdz", ScreeningKind::Csb),
        ("c2h4", "aug-cc-pvdz", ScreeningKind::Schwarz),
        ("ch3oh", "cc-pvtz", ScreeningKind::Schwarz),
    ] {
        v.push((format!("{stem}/{name}"), load(stem, 1), b(name), kind));
    }
    let cuh = parse("2\nCuH\nCu 0 0 0\nH 0 0 1.46\n");
    v.push((
        "CuH/cc-pvdz".into(),
        cuh.clone(),
        b("cc-pvdz"),
        ScreeningKind::Schwarz,
    ));
    v.push((
        "CuH/cc-pvdz-pp (ECP)".into(),
        cuh,
        b("cc-pvdz-pp"),
        ScreeningKind::Schwarz,
    ));
    let agh = parse("2\nAgH\nAg 0 0 0\nH 0 0 1.62\n");
    v.push((
        "AgH/cc-pvdz-pp (ECP)".into(),
        agh,
        b("cc-pvdz-pp"),
        ScreeningKind::Schwarz,
    ));
    let ghost =
        parse("4\nwater + ghost\nO 0 0 0.1\nH 0 0.76 -0.47\nH 0 -0.76 -0.47\n@Ne 0 0 3.0\n");
    v.push((
        "water + @Ne/cc-pvdz (ghost)".into(),
        ghost,
        b("cc-pvdz"),
        ScreeningKind::Schwarz,
    ));
    v.push((
        "c2h4/cc-pvdz C, def2-svp H (mixed)".into(),
        load("c2h4", 1),
        mixed(b("cc-pvdz"), 1, "def2-svp"),
        ScreeningKind::Schwarz,
    ));
    v.push((
        "c2h4/cc-pvdz (cartesian)".into(),
        load("c2h4", 1),
        cartesian(b("cc-pvdz")),
        ScreeningKind::Schwarz,
    ));
    v
}

/// Rotated against unrotated J and K on generally contracted bases with
/// d and f shells, a transition metal, ECPs, a ghost atom, a mixed basis and
/// Cartesian shells; closed and open shell, both rotation levels.
#[test]
fn rotated_jk_matches_unrotated() {
    for (label, mol, bs, kind) in jk_cases() {
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds =
            SchwarzBounds::compute_for_screening(Operator::coulomb(), &prep, kind).unwrap();
        let n = prep.nbasis();
        let d = density(n, 0);
        let (j0, k0, c0) = jk(&prep, &bounds, &d, GcRotate::Off);
        assert_eq!(c0, Some(0));
        for mode in [GcRotate::Auto, GcRotate::Rot1] {
            let (j1, k1, c1) = jk(&prep, &bounds, &d, mode);
            assert!(c1.unwrap() > 0, "{label} {mode:?} rotated nothing");
            let (ej, ek) = (max_abs(&j1, &j0), max_abs(&k1, &k0));
            eprintln!(
                "{label} {kind:?} {mode:?} RHF: {} columns, max|dJ| {ej:.2e} max|dK| {ek:.2e}",
                c1.unwrap()
            );
            assert!(
                ej <= JK_TOL && ek <= JK_TOL,
                "{label} {mode:?}: {ej:e} {ek:e}"
            );
            // A symmetric density: the rotated J and K are as exactly
            // symmetric as the unrotated builder's.
            for (x0, x1, what) in [(&j0, &j1, "J"), (&k0, &k1, "K")] {
                if bits_equal(x0, &x0.t().to_owned()) {
                    assert!(
                        bits_equal(x1, &x1.t().to_owned()),
                        "{label} {mode:?} {what} not symmetric"
                    );
                }
            }
        }
        let (d_a, d_b) = (density(n, 1), density(n, 2));
        let ref_ = jk_uhf(&prep, &bounds, &d_a, &d_b, GcRotate::Off);
        let rot = jk_uhf(&prep, &bounds, &d_a, &d_b, GcRotate::Auto);
        for (x, (a, b)) in ["J", "Ka", "Kb"].iter().zip(rot.iter().zip(&ref_)) {
            let e = max_abs(a, b);
            eprintln!("{label} {kind:?} UHF: max|d{x}| {e:.2e}");
            assert!(e <= JK_TOL, "{label} UHF {x}: {e:e}");
        }
    }
}

/// `‖T‖∞` (largest absolute row sum).
fn norm_inf(t: &Array2<f64>) -> f64 {
    t.rows()
        .into_iter()
        .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0f64, f64::max)
}

/// Rounding-level anchor of the transforms: at threshold 0 (no screening),
/// the rotated build of J and K equals `T X' Tᵀ` with `X'` the unrotated walk
/// on the rotated basis at the densely formed `D' = Tᵀ D T`, to
/// `16 u ‖T‖∞² max|X|`: only the reassociation of the two small transforms
/// separates them. A transform defect of 1e-12 (a wrong `T` entry, a
/// one-sided or transposed pass, a dropped group) is far above that bar.
#[test]
fn rotated_transforms_match_dense_t_at_threshold_zero() {
    let cases = [
        (
            "water/cc-pvtz",
            load("water", 1),
            basis::bundled("cc-pvtz").unwrap(),
        ),
        (
            "CuH/cc-pvdz-pp",
            parse("2\nCuH\nCu 0 0 0\nH 0 0 1.46\n"),
            basis::bundled("cc-pvdz-pp").unwrap(),
        ),
    ];
    let ctx = ParallelContext::default();
    for (label, mol, bs) in cases {
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let n = prep.nbasis();
        let rb = RotatedBasis::detect_prepared(&prep, RotationLevel::Ogc)
            .unwrap()
            .unwrap();
        let t = rb.dense_t();
        let tn2 = norm_inf(&t).powi(2);
        let rbounds = SchwarzBounds::compute(Operator::coulomb(), &rb.prep).unwrap();
        for (seed, sym) in [(0, true), (5, false)] {
            let mut d = density(n, seed);
            if !sym {
                d[(0, n - 1)] += 0.3;
            }
            let mut rot = DirectJK::new(&ctx, &prep, &bounds, 0.0, usize::MAX)
                .with_gc_rotation(GcRotate::Auto);
            let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
            rot.build(&d, &mut j, &mut k).unwrap();
            let dp = t.t().dot(&d).dot(&t);
            let mut walk = DirectJK::new(&ctx, &rb.prep, &rbounds, 0.0, usize::MAX)
                .with_gc_rotation(GcRotate::Off);
            let (mut jp, mut kp) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
            walk.build(&dp, &mut jp, &mut kp).unwrap();
            for (what, x, xp) in [("J", &j, &jp), ("K", &k, &kp)] {
                let dense = t.dot(xp).dot(&t.t());
                assert!(
                    dense.iter().all(|v| v.is_finite()),
                    "{label} {what}: non-finite"
                );
                let scale = dense.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let bar = 16.0 * f64::EPSILON / 2.0 * tn2 * scale;
                let e = max_abs(x, &dense);
                eprintln!(
                    "{label} sym {sym} {what}: max|rotated - T X' Tᵀ| {e:.2e} (bar {bar:.2e}, \
                     ‖T‖∞² {tn2:.2}, max|X| {scale:.2e})"
                );
                assert!(e <= bar, "{label} {what}: {e:e} above {bar:e}");
                assert!(
                    bar < 1e-12,
                    "{label}: the bar {bar:e} cannot see a 1e-12 defect"
                );
            }
        }
    }
}

/// The incremental build on the rotated shells: `J(D1) + J(D2 − D1)`
/// against a fresh rotated `J(D2)`, and against the unrotated `J(D2)`.
#[test]
fn rotated_incremental_build_matches_full_build() {
    let mol = load("c2h4", 1);
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let n = prep.nbasis();
    let (d1, d2) = (density(n, 0), density(n, 3));
    let ctx = ParallelContext::default();
    let mut djk =
        DirectJK::new(&ctx, &prep, &bounds, THRESH, usize::MAX).with_gc_rotation(GcRotate::Auto);
    let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    djk.build(&d1, &mut j, &mut k).unwrap();
    djk.build_incremental(&(&d2 - &d1), &mut j, &mut k).unwrap();
    let (j2, k2, _) = jk(&prep, &bounds, &d2, GcRotate::Auto);
    let (j0, k0, _) = jk(&prep, &bounds, &d2, GcRotate::Off);
    for (what, a, b) in [
        ("J incr vs full", &j, &j2),
        ("K incr vs full", &k, &k2),
        ("J incr vs unrotated", &j, &j0),
        ("K incr vs unrotated", &k, &k0),
    ] {
        let e = max_abs(a, b);
        eprintln!("{what}: {e:.2e}");
        assert!(e <= JK_TOL, "{what}: {e:e}");
    }
}

/// A segmented basis has nothing to rotate: `Auto` is the unrotated builder
/// bit for bit (closed and open shell).
#[test]
fn segmented_basis_is_bitwise_unchanged() {
    let mol = load("c2h4", 1);
    for name in ["def2-svp", "6-31g*", "sto-3g"] {
        let prep = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let n = prep.nbasis();
        let d = density(n, 0);
        let (j0, k0, c0) = jk(&prep, &bounds, &d, GcRotate::Off);
        let (j1, k1, c1) = jk(&prep, &bounds, &d, GcRotate::Auto);
        assert_eq!((c0, c1), (Some(0), Some(0)), "{name}");
        assert!(bits_equal(&j0, &j1) && bits_equal(&k0, &k1), "{name}");
        let (d_a, d_b) = (density(n, 1), density(n, 2));
        let a = jk_uhf(&prep, &bounds, &d_a, &d_b, GcRotate::Off);
        let b = jk_uhf(&prep, &bounds, &d_a, &d_b, GcRotate::Auto);
        assert!(
            a.iter().zip(&b).all(|(x, y)| bits_equal(x, y)),
            "{name} UHF"
        );
    }
}

/// The rotated builder is bitwise identical across rayon thread counts.
#[test]
fn rotated_jk_is_bitwise_across_thread_counts() {
    let mol = load("c2h4", 1);
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let n = prep.nbasis();
    let (d_a, d_b) = (density(n, 1), density(n, 2));
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let (j, k, _) = jk(&prep, &bounds, &d_a, GcRotate::Auto);
            let u = jk_uhf(&prep, &bounds, &d_a, &d_b, GcRotate::Auto);
            (j, k, u)
        })
    };
    let (j1, k1, u1) = run(1);
    for threads in [2, 4] {
        let (j, k, u) = run(threads);
        assert!(
            bits_equal(&j, &j1) && bits_equal(&k, &k1),
            "{threads} threads"
        );
        assert!(
            u.iter().zip(&u1).all(|(x, y)| bits_equal(x, y)),
            "{threads} threads UHF"
        );
    }
}

/// The reference an SCF test runs.
#[derive(Debug, Clone, Copy)]
enum Kind {
    Rhf,
    Uhf,
    Rohf,
}

/// `(energy, iterations)` of an SCF on the direct route with
/// `RhfConfig::gc_rotation = Some(mode)`.
fn scf(mol: &Molecule, name: &str, mode: GcRotate, tight: bool, kind: Kind) -> (f64, usize) {
    let prep = PreparedBasis::new(mol, &basis::bundled(name).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let mut config = RhfConfig::default();
    if tight {
        config = config.with_energy_conv(1e-10).with_density_conv(1e-8);
    }
    config.eri = EriMode::Direct;
    let config = config.with_gc_rotation(mode);
    let op = Operator::coulomb();
    let res = match kind {
        Kind::Rhf => solve_rhf(&ctx, mol, &prep, op, &bounds, &config),
        Kind::Uhf => solve_uhf(&ctx, mol, &prep, &bounds, &config),
        Kind::Rohf => solve_rohf(&ctx, mol, &prep, op, &bounds, &config),
    };
    let res = res.unwrap();
    assert!(res.converged, "{name} {mode:?} did not converge");
    (res.energy, res.iterations)
}

fn compare_scf(stem: &str, multiplicity: usize, name: &str, tight: bool, kind: Kind) {
    let mol = load(stem, multiplicity);
    let (e0, it0) = scf(&mol, name, GcRotate::Off, tight, kind);
    for value in [GcRotate::Auto, GcRotate::Rot1] {
        let (e1, it1) = scf(&mol, name, value, tight, kind);
        let de = (e1 - e0).abs();
        eprintln!(
            "{stem}/{name} {kind:?} tight {tight} {value:?}: E {e1:.12} vs off {e0:.12} \
             ({:#x} vs {:#x}), |dE| {de:.2e}, iterations {it1} vs {it0}",
            e1.to_bits(),
            e0.to_bits()
        );
        assert!(de <= E_TOL, "{stem}/{name} {value:?}: |dE| = {de:e}");
    }
}

#[test]
fn rhf_energy_rotated_matches_unrotated() {
    compare_scf("water", 1, "cc-pvtz", true, Kind::Rhf);
    compare_scf("c2h4", 1, "cc-pvdz", true, Kind::Rhf);
    compare_scf("c2h4", 1, "aug-cc-pvdz", true, Kind::Rhf);
    // Default convergence: the incremental (ΔD) build runs on the rotated
    // shells.
    compare_scf("alkane_4", 1, "cc-pvdz", false, Kind::Rhf);
}

#[test]
fn uhf_energy_rotated_matches_unrotated() {
    compare_scf("oh", 2, "cc-pvdz", true, Kind::Uhf);
    compare_scf("nh2", 2, "cc-pvtz", true, Kind::Uhf);
}

#[test]
fn rohf_energy_rotated_matches_unrotated() {
    compare_scf("nh2", 2, "cc-pvdz", true, Kind::Rohf);
}

/// Fresh bounds for `prep` whose pooled builds run rotation `mode`
/// (`SchwarzBounds::gc`, the request an SCF records).
fn bounds_with(prep: &PreparedBasis, mode: GcRotate) -> SchwarzBounds {
    let b = SchwarzBounds::compute(Operator::coulomb(), prep).unwrap();
    b.gc.set_request(Some(mode));
    b
}

/// The pooled builds (`build_jk`, `build_jk_with_pool_homogeneous`) and the
/// single-matrix SCF builders (`DirectJ`, `DirectK`) rotate under `Auto` and
/// `Rot1`, and agree with the unrotated builds to `JK_TOL`; a rotated result
/// is not the unrotated one bit for bit (it did rotate).
#[test]
fn pooled_and_single_matrix_builds_rotate_and_match_unrotated() {
    use ferric_scf::direct_j::DirectJ;
    use ferric_scf::direct_k::DirectK;
    use ferric_scf::fock::{JBuilder, KBuilder};
    use ferric_scf::rhf::{build_jk, build_jk_with_pool_homogeneous};
    let ctx = ParallelContext::default();
    for (stem, name) in [("water", "cc-pvtz"), ("c2h4", "cc-pvdz")] {
        let mol = load(stem, 1);
        let prep = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
        let n = prep.nbasis();
        let d = density(n, 0);
        // A small response-like density: the homogeneous build rescales it.
        let dsmall = 3e-5 * &density(n, 4);
        let run = |mode: GcRotate| {
            let b = bounds_with(&prep, mode);
            let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
            build_jk(&ctx, &prep, &b, THRESH, &d, &mut j, &mut k).unwrap();
            let pool = ferric_scf::engine_pool::EnginePool::new(
                b.op,
                &prep,
                ferric_integrals::engine_pool::eri_precision(),
            )
            .unwrap();
            let (mut jh, mut kh) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
            build_jk_with_pool_homogeneous(
                &ctx,
                &prep,
                &b,
                THRESH,
                &dsmall,
                &mut jh,
                &mut kh,
                &pool,
                1 << 26,
            )
            .unwrap();
            let mut js = Array2::zeros((n, n));
            JBuilder::build(
                &mut DirectJ::new(&ctx, &prep, &b, THRESH, 1 << 30),
                &d,
                &mut js,
            )
            .unwrap();
            let mut ks = Array2::zeros((n, n));
            KBuilder::build(
                &mut DirectK::new(&ctx, &prep, &b, THRESH, 1 << 30),
                &d,
                &mut ks,
            )
            .unwrap();
            [j, k, jh, kh, js, ks]
        };
        let off = run(GcRotate::Off);
        for mode in [GcRotate::Auto, GcRotate::Rot1] {
            let rot = run(mode);
            let names = [
                "build_jk J",
                "build_jk K",
                "homog J",
                "homog K",
                "DirectJ",
                "DirectK",
            ];
            for ((what, a), b) in names.iter().zip(&rot).zip(&off) {
                // The homogeneous build's operand is 3e-5 of the others.
                let scale = if what.starts_with("homog") { 3e-5 } else { 1.0 };
                let e = max_abs(a, b);
                eprintln!("{stem}/{name} {mode:?} {what}: max|d| {e:.2e}");
                assert!(e <= JK_TOL * scale, "{stem}/{name} {mode:?} {what}: {e:e}");
                assert!(
                    !bits_equal(a, b),
                    "{stem}/{name} {mode:?} {what} did not rotate"
                );
            }
        }
    }
}

/// The pooled builds on a segmented basis are the unrotated builds bit for
/// bit under `Auto`, and the rotated pooled build is bitwise identical
/// across thread counts.
#[test]
fn pooled_builds_bitwise_segmented_and_across_threads() {
    use ferric_scf::rhf::build_jk;
    let mol = load("c2h4", 1);
    let ctx = ParallelContext::default();
    let build = |prep: &PreparedBasis, mode: GcRotate| {
        let n = prep.nbasis();
        let b = bounds_with(prep, mode);
        let d = density(n, 1);
        let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        build_jk(&ctx, prep, &b, THRESH, &d, &mut j, &mut k).unwrap();
        (j, k)
    };
    for name in ["def2-svp", "6-31g*", "sto-3g"] {
        let prep = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
        let (j0, k0) = build(&prep, GcRotate::Off);
        let (j1, k1) = build(&prep, GcRotate::Auto);
        assert!(bits_equal(&j0, &j1) && bits_equal(&k0, &k1), "{name}");
    }
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| build(&prep, GcRotate::Auto))
    };
    let (j1, k1) = run(1);
    for threads in [2, 4] {
        let (j, k) = run(threads);
        assert!(
            bits_equal(&j, &j1) && bits_equal(&k, &k1),
            "{threads} threads"
        );
    }
}

/// `(energy, store rotated?)` of a direct-route RHF on the in-core store.
fn incore_scf(mol: &Molecule, name: &str, mode: GcRotate) -> (f64, bool) {
    let prep = PreparedBasis::new(mol, &basis::bundled(name).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let mut config = RhfConfig::default()
        .with_energy_conv(1e-10)
        .with_density_conv(1e-8);
    config.eri = EriMode::Incore;
    let config = config.with_gc_rotation(mode);
    let res = solve_rhf(&ctx, mol, &prep, Operator::coulomb(), &bounds, &config).unwrap();
    assert!(res.converged);
    let store = ferric_scf::incore_jk::store_for(&bounds, &prep)
        .unwrap()
        .expect("the in-core store serves the SCF's bounds");
    (res.energy, store.rotated())
}

/// The in-core store holds the rotated integrals under `Auto` (none under
/// `Off`, and none for a segmented basis), and the SCF energy agrees with the
/// unrotated store's to `E_TOL`.
#[test]
fn incore_store_rotates_and_matches_unrotated() {
    let mol = load("c2h4", 1);
    let (e0, r0) = incore_scf(&mol, "cc-pvdz", GcRotate::Off);
    let (e1, r1) = incore_scf(&mol, "cc-pvdz", GcRotate::Auto);
    assert!(!r0 && r1);
    let de = (e1 - e0).abs();
    eprintln!("c2h4/cc-pvdz in-core: E {e1:.12} vs off {e0:.12}, |dE| {de:.2e}");
    assert!(de <= E_TOL, "|dE| = {de:e}");
    let (es0, _) = incore_scf(&mol, "def2-svp", GcRotate::Off);
    let (es1, rs1) = incore_scf(&mol, "def2-svp", GcRotate::Auto);
    assert!(!rs1);
    assert_eq!(es0.to_bits(), es1.to_bits(), "def2-svp in-core");
}

/// The post-SCF stability analyses (RHF internal and external, UHF
/// internal: the Newton operator's J/K, pooled builds on the SCF's bounds)
/// run the SCF's rotation; their lowest eigenvalues agree with the
/// unrotated ones.
#[test]
fn stability_eigenvalues_rotated_match_unrotated() {
    let ctx = ParallelContext::default();
    let cfg = |mode: GcRotate| {
        let mut c = RhfConfig::default()
            .with_energy_conv(1e-10)
            .with_density_conv(1e-8)
            .with_check_stability(true)
            .with_gc_rotation(mode);
        c.eri = EriMode::Direct;
        c
    };
    let lowest = |mol: &Molecule, uhf: bool, mode: GcRotate| {
        let prep = PreparedBasis::new(mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let res = if uhf {
            solve_uhf(&ctx, mol, &prep, &bounds, &cfg(mode))
        } else {
            solve_rhf(&ctx, mol, &prep, Operator::coulomb(), &bounds, &cfg(mode))
        }
        .unwrap();
        assert!(res.converged);
        let internal = res.stability.as_ref().expect("internal stability ran");
        let external = res.stability_external.as_ref().map(|s| s.lowest_eigenvalue);
        (internal.lowest_eigenvalue, external)
    };
    for (mol, uhf, label) in [
        (load("water", 1), false, "water RHF"),
        (load("oh", 2), true, "OH UHF"),
    ] {
        let (i0, x0) = lowest(&mol, uhf, GcRotate::Off);
        let (i1, x1) = lowest(&mol, uhf, GcRotate::Auto);
        eprintln!("{label}: internal {i1:.12} vs off {i0:.12}; external {x1:?} vs {x0:?}");
        // Measured: equal to 12 printed digits (internal), 4e-15
        // (external).
        assert!((i1 - i0).abs() <= 1e-11, "{label} internal: {i1} vs {i0}");
        if let (Some(a), Some(b)) = (x0, x1) {
            assert!((a - b).abs() <= 1e-11, "{label} external: {a} vs {b}");
        }
        assert_eq!(x0.is_some(), x1.is_some());
    }
}

/// The analytic gradient's four-centre derivative walk runs on the rotated
/// shells with the rotated densities; RHF and UHF gradients agree with the
/// unrotated ones to `GRAD_TOL`.
#[test]
fn gradient_rotated_matches_unrotated() {
    use ferric_scf::gradient::{rhf_gradient, uhf_gradient};
    let ctx = ParallelContext::default();
    let grad = |stem: &str, mult: usize, name: &str, mode: GcRotate| {
        let mol = load(stem, mult);
        let prep = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let mut c = RhfConfig::default()
            .with_energy_conv(1e-10)
            .with_density_conv(1e-8)
            .with_gc_rotation(mode);
        c.eri = EriMode::Direct;
        let op = Operator::coulomb();
        if mult == 1 {
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &c).unwrap();
            rhf_gradient(&mol, &prep, op, &bounds, &r, None).unwrap()
        } else {
            let r = solve_uhf(&ctx, &mol, &prep, &bounds, &c).unwrap();
            uhf_gradient(&mol, &prep, op, &bounds, &r, None).unwrap()
        }
    };
    for (stem, mult, name) in [
        ("c2h4", 1, "cc-pvdz"),
        ("water", 1, "cc-pvtz"),
        ("oh", 2, "cc-pvdz"),
    ] {
        let g0 = grad(stem, mult, name, GcRotate::Off);
        let g1 = grad(stem, mult, name, GcRotate::Auto);
        let e = max_abs(&g1, &g0);
        eprintln!("{stem}/{name} gradient: max|dG| {e:.2e} Eh/bohr");
        assert!(e <= GRAD_TOL, "{stem}/{name}: {e:e}");
        assert!(
            !bits_equal(&g1, &g0),
            "{stem}/{name}: the gradient did not rotate"
        );
    }
}

/// Rounding-level anchor of the rotated in-core store: at threshold 0 its
/// J and K equal `T X' Tᵀ`, with `X'` the same contraction on an unrotated
/// store of the rotated basis at the densely formed `D' = Tᵀ D T`, to
/// `16 u ‖T‖∞² max|X|` (only the two small transforms reassociate). The
/// rotated store is not a valid `abs_copy` source (refused).
#[test]
fn rotated_incore_store_matches_dense_t_at_threshold_zero() {
    use ferric_scf::incore_jk::{attach_for_scf, IncoreEri};
    let ctx = ParallelContext::default();
    for (label, mol, name) in [
        ("water/cc-pvtz", load("water", 1), "cc-pvtz"),
        ("c2h4/cc-pvdz", load("c2h4", 1), "cc-pvdz"),
    ] {
        let prep = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
        let n = prep.nbasis();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let mut config = RhfConfig::default().with_gc_rotation(GcRotate::Auto);
        config.eri = EriMode::Incore;
        let store = attach_for_scf(&ctx, &prep, &bounds, &config, usize::MAX / 4, (true, 0.0))
            .unwrap()
            .expect("eri = incore attaches a store");
        assert!(store.rotated(), "{label}: the store did not rotate");
        assert!(
            store.abs_copy().is_err(),
            "{label}: abs_copy of a rotated store"
        );
        let rb = RotatedBasis::detect_prepared(&prep, RotationLevel::Ogc)
            .unwrap()
            .unwrap();
        let t = rb.dense_t();
        let tn2 = norm_inf(&t).powi(2);
        let plan = IncoreEri::plan(&rb.prep, usize::MAX / 4).unwrap();
        let charge = plan.commit().unwrap();
        let inner = IncoreEri::build(&ctx, &rb.prep, Operator::coulomb(), &plan, charge).unwrap();
        assert!(!inner.rotated());
        for (seed, sym) in [(0, true), (5, false)] {
            let mut d = density(n, seed);
            if !sym {
                d[(0, n - 1)] += 0.3;
            }
            let (j, k) = store.jk_multi(&[&d], &[&d]).unwrap();
            let dp = t.t().dot(&d).dot(&t);
            let (jp, kp) = inner.jk_multi(&[&dp], &[&dp]).unwrap();
            for (what, x, xp) in [("J", &j[0], &jp[0]), ("K", &k[0], &kp[0])] {
                let dense = t.dot(xp).dot(&t.t());
                let scale = dense.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let bar = 16.0 * f64::EPSILON / 2.0 * tn2 * scale;
                let e = max_abs(x, &dense);
                eprintln!(
                    "{label} in-core sym {sym} {what}: max|store - T X' Tᵀ| {e:.2e} (bar {bar:.2e})"
                );
                assert!(e <= bar, "{label} {what}: {e:e} above {bar:e}");
                assert!(
                    bar < 1e-12,
                    "{label}: the bar {bar:e} cannot see a 1e-12 defect"
                );
            }
        }
        // A density of the wrong size is an error, not a panic.
        assert!(store
            .jk_multi(&[&Array2::zeros((n + 1, n + 1))], &[])
            .is_err());
    }
}

fn hash_bits<'a>(arrays: impl IntoIterator<Item = &'a Array2<f64>>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for a in arrays {
        a.dim().hash(&mut h);
        for v in a.iter() {
            v.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

/// Set in the environment of the child runs.
const CHILD: &str = "FERRIC_TEST_GC_ROTATION_CHILD";

/// Water / cc-pVDZ, closed-shell PBE0 on the direct route, SCF converged
/// unrotated (so every run starts from the same bits); then, on bounds that
/// carry `request` (`None`: fresh bounds, nothing recorded — the consumers
/// of `main`, where only the SCF's `DirectJK` rotated), the consumers that
/// follow the bounds: the analytic KS gradient, the KS Z-vector relaxation
/// gradient (`zvector_ks::ResponseFock`: Coulomb and exchange response
/// J/K, the Hessian products) and a pooled `build_jk`.
fn consumers(request: Option<Option<GcRotate>>) -> Vec<Array2<f64>> {
    use ferric_scf::ks_gradient::ks_gradient_closed;
    use ferric_scf::zvector_ks::relaxation_gradient_closed;
    let ctx = ParallelContext::default();
    let mol = load("water", 1);
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let scf_bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut cfg = RhfConfig::default()
        .with_energy_conv(1e-11)
        .with_density_conv(1e-9)
        .with_gc_rotation(GcRotate::Off);
    cfg.xc = Some("PBE0".into());
    // Four-centre J and K (a functional defaults to RI-JK).
    cfg.df_j_aux = Some(String::new());
    cfg.df_k_aux = Some(String::new());
    cfg.eri = EriMode::Direct;
    let res = solve_rhf(&ctx, &mol, &prep, op, &scf_bounds, &cfg).unwrap();
    assert!(res.converged);
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    if let Some(r) = request {
        bounds.gc.set_request(r);
    }
    let n = prep.nbasis();
    let g = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, "PBE0", &res, None).unwrap();
    let v = density(n, 3);
    let z =
        relaxation_gradient_closed(&ctx, &mol, &prep, &bs, op, &bounds, &cfg, &res, &v).unwrap();
    let d = density(n, 2);
    let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    ferric_scf::rhf::build_jk(&ctx, &prep, &bounds, THRESH, &d, &mut j, &mut k).unwrap();
    vec![g, z.gradient, z.z, j, k]
}

/// The blocker of the review: `gc_rotation = "off"` reaches the consumers.
/// Config `Off` recorded on the bounds is bitwise the consumers on fresh
/// bounds (`main`), and bitwise an unset config under
/// `FERRIC_GC_ROTATE=off` (a child process of this binary with the
/// variable set; no test mutates its own environment). `Auto` rotates
/// them (bits differ) and agrees: the KS gradient and the pooled J, K
/// within `GRAD_TOL`, `Z` within 1e-12 and the Z-vector gradient within
/// 1e-9 (measured differences printed).
#[test]
fn off_config_reaches_the_consumers_bitwise() {
    let off = consumers(Some(Some(GcRotate::Off)));
    let fresh = consumers(None);
    let names = ["KS gradient", "Z-vector gradient", "Z", "J", "K"];
    for ((what, a), b) in names.iter().zip(&off).zip(&fresh) {
        assert!(
            bits_equal(a, b),
            "{what}: off is not the fresh-bounds build"
        );
    }
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args(["child_prints_unset_config_bits", "--exact", "--nocapture"])
        .env(CHILD, "1")
        .env("FERRIC_GC_ROTATE", "off");
    let out = cmd.output().expect("spawn the child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "child failed: {stdout}");
    let child = stdout
        .lines()
        .find_map(|l| l.strip_prefix("GC_BITS "))
        .unwrap_or_else(|| panic!("no GC_BITS line: {stdout}"));
    assert_eq!(
        u64::from_str_radix(child.trim(), 16).unwrap(),
        hash_bits(&off),
        "config off is not the unset config under FERRIC_GC_ROTATE=off"
    );
    let auto = consumers(Some(Some(GcRotate::Auto)));
    for ((what, a), b) in names.iter().zip(&auto).zip(&off) {
        let e = max_abs(a, b);
        eprintln!("water/cc-pvdz PBE0 {what}: auto vs off max|d| {e:.2e}");
        assert!(!bits_equal(a, b), "{what}: auto did not rotate");
        // Measured: KS gradient 8.8e-13, Z-vector gradient 3.0e-10 (of a
        // gradient of order 1 for this arbitrary `V`), Z 4.9e-14, J and K
        // a few 1e-15.
        let bar = match *what {
            "Z-vector gradient" => 1e-9,
            "Z" => 1e-12,
            _ => GRAD_TOL,
        };
        assert!(e <= bar, "{what}: {e:e} above {bar:e}");
    }
}

/// Child entry point of `off_config_reaches_the_consumers_bitwise`: the
/// consumers on bounds that recorded an unset config, their bits printed.
/// A no-op outside the child.
#[test]
fn child_prints_unset_config_bits() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    println!("GC_BITS {:016x}", hash_bits(&consumers(Some(None))));
}

/// The KS gradients through their entry points (RKS and UKS with a hybrid,
/// so the scaled-K derivative wrappers run) and the ROHF gradient, on the
/// SCF's bounds: rotated against unrotated within 1e-10 Eh/bohr.
#[test]
fn ks_and_rohf_gradients_rotated_match_unrotated() {
    use ferric_scf::gradient::rohf_gradient;
    use ferric_scf::ks_gradient::{ks_gradient_closed, ks_gradient_uks};
    let ctx = ParallelContext::default();
    let run = |stem: &str, mult: usize, xc: Option<&str>, mode: GcRotate| {
        let mol = load(stem, mult);
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let mut c = RhfConfig::default()
            .with_energy_conv(1e-11)
            .with_density_conv(1e-9)
            .with_gc_rotation(mode);
        c.xc = xc.map(Into::into);
        // Four-centre J and K (a functional defaults to RI-JK).
        c.df_j_aux = Some(String::new());
        c.df_k_aux = Some(String::new());
        c.eri = EriMode::Direct;
        match (mult, xc) {
            (1, Some(x)) => {
                let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &c).unwrap();
                assert!(r.converged);
                ks_gradient_closed(&mol, &prep, &bs, op, &bounds, x, &r, None).unwrap()
            }
            (_, Some(x)) => {
                let r = solve_uhf(&ctx, &mol, &prep, &bounds, &c).unwrap();
                assert!(r.converged);
                ks_gradient_uks(&mol, &prep, &bs, op, &bounds, x, &r, None).unwrap()
            }
            (_, None) => {
                let r = solve_rohf(&ctx, &mol, &prep, op, &bounds, &c).unwrap();
                assert!(r.converged);
                rohf_gradient(&mol, &prep, op, &bounds, &r, None).unwrap()
            }
        }
    };
    for (label, stem, mult, xc) in [
        ("water RKS PBE0", "water", 1, Some("PBE0")),
        ("OH UKS PBE0", "oh", 2, Some("PBE0")),
        ("NH2 ROHF", "nh2", 2, None),
    ] {
        let g0 = run(stem, mult, xc, GcRotate::Off);
        let g1 = run(stem, mult, xc, GcRotate::Auto);
        let e = max_abs(&g1, &g0);
        eprintln!("{label} gradient: max|dG| {e:.2e} Eh/bohr");
        assert!(e <= GRAD_TOL, "{label}: {e:e}");
        assert!(
            !bits_equal(&g1, &g0),
            "{label}: the gradient did not rotate"
        );
    }
}

/// A CPHF consumer: the analytic RHF Hessian (its CPHF runs the
/// multi-density homogeneous pooled build on the SCF's bounds; the
/// second-derivative integrals stay unrotated). Rotated against unrotated
/// within 1e-9 Eh/bohr² (printed).
#[test]
fn hessian_cphf_rotated_matches_unrotated() {
    use ferric_scf::hessian::rhf_hessian;
    let ctx = ParallelContext::default();
    let mol = load("water", 1);
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let run = |mode: GcRotate| {
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let mut c = RhfConfig::default()
            .with_energy_conv(1e-11)
            .with_density_conv(1e-9)
            .with_gc_rotation(mode);
        c.eri = EriMode::Direct;
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &c).unwrap();
        assert!(r.converged);
        rhf_hessian(&ctx, &mol, &prep, op, &bounds, &r, &c).unwrap()
    };
    let h0 = run(GcRotate::Off);
    let h1 = run(GcRotate::Auto);
    let e = max_abs(&h1, &h0);
    eprintln!("water/cc-pvdz RHF Hessian: max|dH| {e:.2e} Eh/bohr^2");
    assert!(e <= 1e-9, "{e:e}");
    assert!(!bits_equal(&h1, &h0), "the Hessian did not rotate");
}

/// Forced second-order SCF: TRAH from the first iterations (RHF) and the
/// PCG Newton step (UHF) run their J/K on the rotated shells; energies
/// within `E_TOL` of the unrotated runs.
#[test]
fn second_order_scf_rotated_matches_unrotated() {
    let ctx = ParallelContext::default();
    let run = |stem: &str, mult: usize, mode: GcRotate| {
        let mol = load(stem, mult);
        let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let mut c = RhfConfig::default()
            .with_energy_conv(1e-11)
            .with_density_conv(1e-9)
            .with_gc_rotation(mode);
        c.eri = EriMode::Direct;
        let r = if mult == 1 {
            c.trah_trigger = Some(1.0);
            solve_rhf(&ctx, &mol, &prep, op, &bounds, &c)
        } else {
            c.newton_trigger = 1.0;
            solve_uhf(&ctx, &mol, &prep, &bounds, &c)
        }
        .unwrap();
        assert!(r.converged);
        (r.energy, r.iterations)
    };
    for (label, stem, mult) in [("water RHF TRAH", "water", 1), ("OH UHF Newton", "oh", 2)] {
        let (e0, i0) = run(stem, mult, GcRotate::Off);
        let (e1, i1) = run(stem, mult, GcRotate::Auto);
        let de = (e1 - e0).abs();
        eprintln!("{label}: |dE| {de:.2e}, iterations {i1} vs {i0}");
        assert!(de <= E_TOL, "{label}: |dE| = {de:e}");
    }
}

/// `attach_for_scf` reuses the bounds' store only when it was built on the
/// requested rotation: `off` on bounds holding a rotated store rebuilds an
/// unrotated one (and the other way round, and across stages), while a
/// repeat of the same request reuses the store.
#[test]
fn incore_store_reuse_follows_the_requested_rotation() {
    use ferric_scf::incore_jk::attach_for_scf;
    use std::sync::Arc;
    let ctx = ParallelContext::default();
    let mol = load("water", 1);
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let attach = |mode: GcRotate| {
        let mut config = RhfConfig::default().with_gc_rotation(mode);
        config.eri = EriMode::Incore;
        attach_for_scf(&ctx, &prep, &bounds, &config, usize::MAX / 4, (true, 0.0))
            .unwrap()
            .expect("eri = incore attaches a store")
    };
    let auto = attach(GcRotate::Auto);
    assert!(auto.rotated());
    assert!(
        Arc::ptr_eq(&auto, &attach(GcRotate::Auto)),
        "same request reused"
    );
    let off = attach(GcRotate::Off);
    assert!(!off.rotated(), "off served a rotated store");
    assert!(Arc::ptr_eq(&off, &attach(GcRotate::Off)));
    let auto2 = attach(GcRotate::Auto);
    assert!(auto2.rotated() && !Arc::ptr_eq(&auto2, &off));
    let rot1 = attach(GcRotate::Rot1);
    assert!(
        rot1.rotated() && !Arc::ptr_eq(&rot1, &auto2),
        "rot1 served auto's store"
    );
}

/// The whole rotated SCF is thread-count independent: RHF on the in-core
/// store (rotated ERIs, transforms in `coulomb` / `exchange`) and on the
/// direct route (rotated `DirectJK`, incremental builds), and a UHF on the
/// direct route, give the same energy and density bits on 1, 2 and 4
/// threads.
#[test]
fn rotated_scf_is_bitwise_across_thread_counts() {
    let run = |threads: usize, stem: &str, mult: usize, eri: EriMode| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let ctx = ParallelContext::default();
                let mol = load(stem, mult);
                let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
                let op = Operator::coulomb();
                let bounds = SchwarzBounds::compute(op, &prep).unwrap();
                let mut c = RhfConfig::default().with_gc_rotation(GcRotate::Auto);
                c.eri = eri;
                let r = if mult == 1 {
                    solve_rhf(&ctx, &mol, &prep, op, &bounds, &c)
                } else {
                    solve_uhf(&ctx, &mol, &prep, &bounds, &c)
                }
                .unwrap();
                assert!(r.converged);
                (r.energy, r.density_alpha.clone())
            })
    };
    for (label, stem, mult, eri) in [
        ("water RHF in-core", "water", 1, EriMode::Incore),
        ("water RHF direct", "water", 1, EriMode::Direct),
        ("OH UHF direct", "oh", 2, EriMode::Direct),
    ] {
        let (e1, d1) = run(1, stem, mult, eri);
        for threads in [2, 4] {
            let (e, d) = run(threads, stem, mult, eri);
            assert_eq!(
                e.to_bits(),
                e1.to_bits(),
                "{label}: {threads} threads energy"
            );
            assert!(bits_equal(&d, &d1), "{label}: {threads} threads density");
        }
    }
}
