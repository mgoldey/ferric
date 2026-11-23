//! The internal contraction rotation of the direct four-centre J/K builder
//! (`DirectJK`, `FERRIC_GC_ROTATE`; `ferric_integrals::contraction_rotation`).
//!
//! * Tolerance: rotated (both stages, and stage 1 alone) against unrotated
//!   J and K, closed and open shell, full and incremental, Schwarz and CSB
//!   screening: max |Δ| ≤ 1e-11 at the production `integral_thresh` (1e-12).
//!   SCF energies (RHF, UHF and ROHF, cc-pVDZ / cc-pVTZ / aug-cc-pVDZ, d and f
//!   shells) within 1e-10 Ha.
//! * Bitwise: a segmented basis (def2-SVP, 6-31G*, STO-3G) under `Auto` is
//!   the unrotated builder bit for bit; `Off` never rotates; the rotated
//!   builder is bitwise identical across thread counts.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_jk::{DirectJK, GcRotate, GC_ROTATE_ENV};
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

/// Rotated against unrotated J and K on generally contracted bases with
/// d and f shells, closed and open shell, both rotation levels.
#[test]
fn rotated_jk_matches_unrotated() {
    let cases = [
        ("water", "cc-pvdz", ScreeningKind::Schwarz),
        ("water", "cc-pvtz", ScreeningKind::Schwarz),
        ("c2h4", "cc-pvdz", ScreeningKind::Schwarz),
        ("c2h4", "cc-pvdz", ScreeningKind::Csb),
        ("c2h4", "aug-cc-pvdz", ScreeningKind::Schwarz),
        ("ch3oh", "cc-pvtz", ScreeningKind::Schwarz),
    ];
    for (stem, name, kind) in cases {
        let mol = load(stem, 1);
        let prep = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
        let bounds =
            SchwarzBounds::compute_for_screening(Operator::coulomb(), &prep, kind).unwrap();
        let n = prep.nbasis();
        let d = density(n, 0);
        let (j0, k0, c0) = jk(&prep, &bounds, &d, GcRotate::Off);
        assert_eq!(c0, Some(0));
        for mode in [GcRotate::Auto, GcRotate::Rot1] {
            let (j1, k1, c1) = jk(&prep, &bounds, &d, mode);
            assert!(c1.unwrap() > 0, "{stem}/{name} {mode:?} rotated nothing");
            let (ej, ek) = (max_abs(&j1, &j0), max_abs(&k1, &k0));
            eprintln!("{stem}/{name} {kind:?} {mode:?} RHF: max|dJ| {ej:.2e} max|dK| {ek:.2e}");
            assert!(
                ej <= JK_TOL && ek <= JK_TOL,
                "{stem}/{name} {mode:?}: {ej:e} {ek:e}"
            );
        }
        let (d_a, d_b) = (density(n, 1), density(n, 2));
        let ref_ = jk_uhf(&prep, &bounds, &d_a, &d_b, GcRotate::Off);
        let rot = jk_uhf(&prep, &bounds, &d_a, &d_b, GcRotate::Auto);
        for (x, (a, b)) in ["J", "Ka", "Kb"].iter().zip(rot.iter().zip(&ref_)) {
            let e = max_abs(a, b);
            eprintln!("{stem}/{name} {kind:?} UHF: max|d{x}| {e:.2e}");
            assert!(e <= JK_TOL, "{stem}/{name} UHF {x}: {e:e}");
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

/// Serializes the tests that set `FERRIC_GC_ROTATE` (process-global) for the
/// span of an SCF.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The reference an SCF test runs.
#[derive(Debug, Clone, Copy)]
enum Kind {
    Rhf,
    Uhf,
    Rohf,
}

/// `(energy, iterations)` of an SCF on the direct route with
/// `FERRIC_GC_ROTATE=value`.
fn scf(mol: &Molecule, name: &str, value: &str, tight: bool, kind: Kind) -> (f64, usize) {
    let prep = PreparedBasis::new(mol, &basis::bundled(name).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let mut config = RhfConfig::default();
    if tight {
        config = config.with_energy_conv(1e-10).with_density_conv(1e-8);
    }
    config.eri = EriMode::Direct;
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(GC_ROTATE_ENV, value);
    let op = Operator::coulomb();
    let res = match kind {
        Kind::Rhf => solve_rhf(&ctx, mol, &prep, op, &bounds, &config),
        Kind::Uhf => solve_uhf(&ctx, mol, &prep, &bounds, &config),
        Kind::Rohf => solve_rohf(&ctx, mol, &prep, op, &bounds, &config),
    };
    std::env::remove_var(GC_ROTATE_ENV);
    let res = res.unwrap();
    assert!(res.converged, "{name} {value} did not converge");
    (res.energy, res.iterations)
}

fn compare_scf(stem: &str, multiplicity: usize, name: &str, tight: bool, kind: Kind) {
    let mol = load(stem, multiplicity);
    let (e0, it0) = scf(&mol, name, "off", tight, kind);
    for value in ["auto", "rot1"] {
        let (e1, it1) = scf(&mol, name, value, tight, kind);
        let de = (e1 - e0).abs();
        eprintln!(
            "{stem}/{name} {kind:?} tight {tight} {value}: E {e1:.12} vs off {e0:.12} \
             ({:#x} vs {:#x}), |dE| {de:.2e}, iterations {it1} vs {it0}",
            e1.to_bits(),
            e0.to_bits()
        );
        assert!(de <= E_TOL, "{stem}/{name} {value}: |dE| = {de:e}");
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
