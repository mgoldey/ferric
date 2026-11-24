//! The internal contraction rotation of the direct four-centre J/K builder
//! (`DirectJK`, `FERRIC_GC_ROTATE`; `ferric_integrals::contraction_rotation`).
//!
//! * Tolerance: rotated (both stages, and stage 1 alone) against unrotated
//!   J and K, closed and open shell, full and incremental, Schwarz and CSB
//!   screening: max |Δ| ≤ 1e-11 at the production `integral_thresh` (1e-12).
//!   SCF energies (RHF, UHF and ROHF, cc-pVDZ / cc-pVTZ / aug-cc-pVDZ, d and f
//!   shells) within 1e-10 Ha.
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
//! by setting `FERRIC_GC_ROTATE`, so no test mutates the environment.

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
const E_TOL: f64 = 1e-10;

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
    (a - b).iter().fold(0.0f64, |m, x| m.max(x.abs()))
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
