//! The shared-primitive kernel of the rotated direct J/K walk
//! (`[scf] gc_kernel`, `ferric_scf::direct_jk::gc_kernel`).
//!
//! * Tolerance: J and K with the kernel on against the libint walk (kernel
//!   off), both rotated, at a fixed density: max |Δ| ≤ 1e-12 (closed and open
//!   shell, full and incremental, Schwarz and CSB screening, cc-pVDZ /
//!   aug-cc-pVDZ / cc-pVTZ, a Cartesian basis, a ghost atom, a mixed basis,
//!   an ECP basis). SCF energies (RHF, UHF, ROHF) within 1e-10 Ha.
//! * The probe (`gc_kernel::kernel_probe`): the kernel serves quartets with
//!   `auto` and none with `off`.
//! * Bitwise: `off` (explicit, or unset with the default) is the walk
//!   without the kernel; a segmented basis (def2-SVP, 6-31G*, STO-3G), the
//!   unrotated walk (`gc_rotation = "off"`) and the legacy digest are
//!   unchanged by `auto`; with `auto` J, K and whole-SCF energies are bitwise
//!   identical at 1, 2 and 4 threads.
//!
//! The setting is chosen through the bounds' record (what an SCF config
//! writes), never by setting `FERRIC_GC_KERNEL` in this process.

use ferric_core::basis;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_jk::gc_kernel::{kernel_probe, KernelCounts};
use ferric_scf::direct_jk::{DirectJK, GcKernel, GcRotate};
use ferric_scf::incore_jk::EriMode;
use ferric_scf::quartet_scatter::JkDigest;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::{SchwarzBounds, ScreeningKind};
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const THRESH: f64 = 1e-12;
const JK_TOL: f64 = 1e-12;
const E_TOL: f64 = 1e-10;

fn load(stem: &str, multiplicity: usize) -> Molecule {
    let text = std::fs::read_to_string(format!("../../testdata/molecules/{stem}.xyz"))
        .unwrap_or_else(|e| panic!("cannot read {stem}.xyz: {e}"));
    Molecule::parse_xyz(&text, 0, multiplicity).unwrap()
}

fn parse(xyz: &str) -> Molecule {
    Molecule::parse_xyz(xyz, 0, 1).unwrap()
}

/// A symmetric test density (the `gc_rotation` tests' recipe).
fn density(n: usize, seed: usize) -> Array2<f64> {
    let d = Array2::from_shape_fn((n, n), |(i, j)| {
        let v = ((i * (7 + seed) + j * (3 + 2 * seed)) % (11 + seed)) as f64;
        0.02 * v * (-0.01 * (i as f64 - j as f64).abs()).exp() - 0.03
    });
    0.5 * (&d + &d.t())
}

fn max_abs(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter().zip(b.iter()).fold(0.0f64, |m, (x, y)| {
        let d = (x - y).abs();
        assert!(d.is_finite(), "non-finite difference: {x} vs {y}");
        m.max(d)
    })
}

fn bits_equal(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Fresh bounds recording the kernel setting `kernel` (`None`: unrecorded)
/// and the digest `digest`.
fn bounds(
    prep: &PreparedBasis,
    kind: ScreeningKind,
    kernel: Option<GcKernel>,
    digest: Option<JkDigest>,
) -> SchwarzBounds {
    let b = SchwarzBounds::compute_for_screening(Operator::coulomb(), prep, kind).unwrap();
    if kernel.is_some() {
        b.gc_kernel.set(kernel);
    }
    b.digest.set(digest);
    b
}

/// Closed-shell J, K (full build of `d`, then an incremental `d2 − d` on
/// top when `d2` is given) from a fresh builder with rotation `rot`, and
/// the kernel counts the probe recorded.
fn jk(
    prep: &PreparedBasis,
    b: &SchwarzBounds,
    d: &Array2<f64>,
    d2: Option<&Array2<f64>>,
    rot: GcRotate,
) -> (Array2<f64>, Array2<f64>, Vec<KernelCounts>) {
    let ctx = ParallelContext::default();
    let n = prep.nbasis();
    let mut djk = DirectJK::new(&ctx, prep, b, THRESH, usize::MAX).with_gc_rotation(rot);
    let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    kernel_probe::start();
    djk.build(d, &mut j, &mut k).unwrap();
    if let Some(d2) = d2 {
        djk.build_incremental(&(d2 - d), &mut j, &mut k).unwrap();
    }
    (j, k, kernel_probe::take())
}

/// Open-shell J, K_α, K_β, and the probe's counts.
fn jk_uhf(
    prep: &PreparedBasis,
    b: &SchwarzBounds,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    rot: GcRotate,
) -> ([Array2<f64>; 3], Vec<KernelCounts>) {
    let ctx = ParallelContext::default();
    let n = prep.nbasis();
    let mut djk = DirectJK::new(&ctx, prep, b, THRESH, usize::MAX).with_gc_rotation(rot);
    let mut out = [
        Array2::zeros((n, n)),
        Array2::zeros((n, n)),
        Array2::zeros((n, n)),
    ];
    let [j, ka, kb] = &mut out;
    kernel_probe::start();
    djk.build_uhf(&(d_a + d_b), d_a, d_b, j, ka, kb).unwrap();
    (out, kernel_probe::take())
}

fn served(c: &[KernelCounts]) -> usize {
    c.iter().map(|x| x.quartets).sum()
}

fn cartesian(mut bs: BasisSet) -> BasisSet {
    for shells in bs.shells.values_mut() {
        for sh in shells {
            sh.pure = false;
        }
    }
    bs.name = format!("{} (cartesian)", bs.name);
    bs
}

fn mixed(mut bs: BasisSet, z: i32, other: &str) -> BasisSet {
    let o = basis::bundled(other).unwrap();
    bs.shells.insert(z, o.shells[&z].clone());
    bs.name = format!("{} + {other} on Z={z}", bs.name);
    bs
}

/// The J/K test systems: label, molecule, basis, screening.
fn cases() -> Vec<(String, Molecule, BasisSet, ScreeningKind)> {
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
    v.push((
        "AgH/cc-pvdz-pp (ECP)".into(),
        parse("2\nAgH\nAg 0 0 0\nH 0 0 1.62\n"),
        b("cc-pvdz-pp"),
        ScreeningKind::Schwarz,
    ));
    v.push((
        "water + @Ne/cc-pvdz (ghost)".into(),
        parse("4\nwater + ghost\nO 0 0 0.1\nH 0 0.76 -0.47\nH 0 -0.76 -0.47\n@Ne 0 0 3.0\n"),
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

/// Kernel on against the libint walk, both rotated: closed shell (full and
/// incremental) and open shell, within 1e-12; the probe sees the kernel
/// serve quartets with `auto` and none with `off`.
#[test]
fn kernel_jk_matches_the_libint_walk() {
    for (label, mol, bs, kind) in cases() {
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let n = prep.nbasis();
        let (d, d2) = (density(n, 0), density(n, 3));
        let off = bounds(&prep, kind, Some(GcKernel::Off), None);
        let on = bounds(&prep, kind, Some(GcKernel::Auto), None);
        let (j0, k0, c0) = jk(&prep, &off, &d, Some(&d2), GcRotate::Auto);
        let (j1, k1, c1) = jk(&prep, &on, &d, Some(&d2), GcRotate::Auto);
        assert_eq!(c0.len(), 2, "{label}: two rotated pair-digest builds");
        assert_eq!(served(&c0), 0, "{label}: off served quartets");
        assert!(served(&c1) > 0, "{label}: the kernel never ran");
        let (ej, ek) = (max_abs(&j1, &j0), max_abs(&k1, &k0));
        eprintln!(
            "{label} {kind:?} RHF: max|dJ| {ej:.2e} max|dK| {ek:.2e}; kernel {:?}",
            c1[0]
        );
        assert!(ej <= JK_TOL && ek <= JK_TOL, "{label}: {ej:e} {ek:e}");
        // Exactly symmetric, as the libint walk's.
        assert!(bits_equal(&j1, &j1.t().to_owned()), "{label} J");
        assert!(bits_equal(&k1, &k1.t().to_owned()), "{label} K");
        let (d_a, d_b) = (density(n, 1), density(n, 2));
        let (u0, c0) = jk_uhf(&prep, &off, &d_a, &d_b, GcRotate::Auto);
        let (u1, c1) = jk_uhf(&prep, &on, &d_a, &d_b, GcRotate::Auto);
        assert_eq!(served(&c0), 0, "{label} UHF");
        assert!(served(&c1) > 0, "{label} UHF: the kernel never ran");
        for (x, (a, b)) in ["J", "Ka", "Kb"].iter().zip(u1.iter().zip(&u0)) {
            let e = max_abs(a, b);
            eprintln!("{label} {kind:?} UHF: max|d{x}| {e:.2e}");
            assert!(e <= JK_TOL, "{label} UHF {x}: {e:e}");
        }
    }
}

/// `off`, explicit or the unset default, is the walk without the kernel bit
/// for bit; `auto` changes nothing where the kernel does not apply: a
/// segmented basis (no rotation), `gc_rotation = "off"` and the legacy
/// digest.
#[test]
fn off_and_inapplicable_cases_are_bitwise() {
    let mol = load("c2h4", 1);
    // Unset record with the default (`FERRIC_GC_KERNEL` unset in tests) and
    // an explicit `off` are one walk.
    if std::env::var_os("FERRIC_GC_KERNEL").is_none() {
        let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let n = prep.nbasis();
        let d = density(n, 0);
        let unset = bounds(&prep, ScreeningKind::Schwarz, None, None);
        let off = bounds(&prep, ScreeningKind::Schwarz, Some(GcKernel::Off), None);
        let (j0, k0, c0) = jk(&prep, &unset, &d, None, GcRotate::Auto);
        let (j1, k1, _) = jk(&prep, &off, &d, None, GcRotate::Auto);
        assert_eq!(served(&c0), 0);
        assert!(bits_equal(&j0, &j1) && bits_equal(&k0, &k1));
    }
    for name in ["def2-svp", "6-31g*", "sto-3g", "cc-pvdz"] {
        let prep = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
        let n = prep.nbasis();
        let d = density(n, 0);
        let (d_a, d_b) = (density(n, 1), density(n, 2));
        let off = bounds(&prep, ScreeningKind::Schwarz, Some(GcKernel::Off), None);
        let on = bounds(&prep, ScreeningKind::Schwarz, Some(GcKernel::Auto), None);
        // Segmented bases rotate nothing (so `Auto` rotation runs the
        // unrotated walk); cc-pVDZ is checked with the rotation off.
        let rot = if name == "cc-pvdz" {
            GcRotate::Off
        } else {
            GcRotate::Auto
        };
        let (j0, k0, _) = jk(&prep, &off, &d, None, rot);
        let (j1, k1, c1) = jk(&prep, &on, &d, None, rot);
        assert!(c1.is_empty(), "{name}: a rotated walk ran");
        assert!(bits_equal(&j0, &j1) && bits_equal(&k0, &k1), "{name}");
        let (u0, _) = jk_uhf(&prep, &off, &d_a, &d_b, rot);
        let (u1, c1) = jk_uhf(&prep, &on, &d_a, &d_b, rot);
        assert!(c1.is_empty(), "{name} UHF: a rotated walk ran");
        assert!(
            u0.iter().zip(&u1).all(|(x, y)| bits_equal(x, y)),
            "{name} UHF"
        );
    }
    // The legacy digest keeps libint, rotated or not.
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let d = density(prep.nbasis(), 0);
    let off = bounds(
        &prep,
        ScreeningKind::Schwarz,
        Some(GcKernel::Off),
        Some(JkDigest::Legacy),
    );
    let on = bounds(
        &prep,
        ScreeningKind::Schwarz,
        Some(GcKernel::Auto),
        Some(JkDigest::Legacy),
    );
    let (j0, k0, _) = jk(&prep, &off, &d, None, GcRotate::Auto);
    let (j1, k1, c1) = jk(&prep, &on, &d, None, GcRotate::Auto);
    assert!(
        c1.is_empty(),
        "legacy digest: the probe records pair builds only"
    );
    assert!(bits_equal(&j0, &j1) && bits_equal(&k0, &k1), "legacy");
}

/// With the kernel on, J and K (closed and open shell) are bitwise
/// identical at 1, 2 and 4 threads.
#[test]
fn kernel_jk_is_bitwise_across_thread_counts() {
    for (stem, name) in [("c2h4", "cc-pvdz"), ("water", "aug-cc-pvdz")] {
        let mol = load(stem, 1);
        let prep = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
        let on = bounds(&prep, ScreeningKind::Schwarz, Some(GcKernel::Auto), None);
        let n = prep.nbasis();
        let (d_a, d_b) = (density(n, 1), density(n, 2));
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let (j, k, _) = jk(&prep, &on, &d_a, Some(&d_b), GcRotate::Auto);
                let (u, _) = jk_uhf(&prep, &on, &d_a, &d_b, GcRotate::Auto);
                (j, k, u)
            })
        };
        let (j1, k1, u1) = run(1);
        for threads in [2, 4] {
            let (j, k, u) = run(threads);
            assert!(
                bits_equal(&j, &j1) && bits_equal(&k, &k1),
                "{stem}/{name} {threads} threads"
            );
            assert!(
                u.iter().zip(&u1).all(|(x, y)| bits_equal(x, y)),
                "{stem}/{name} {threads} threads UHF"
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    Rhf,
    Uhf,
    Rohf,
}

/// `(energy, iterations)` of a direct SCF with `gc_kernel = kernel`
/// (rotation `auto`).
fn scf(mol: &Molecule, name: &str, kernel: GcKernel, kind: Kind) -> (f64, usize) {
    let prep = PreparedBasis::new(mol, &basis::bundled(name).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let mut config = RhfConfig::default()
        .with_energy_conv(1e-10)
        .with_density_conv(1e-8)
        .with_gc_rotation(GcRotate::Auto);
    config.eri = EriMode::Direct;
    config.gc_kernel = Some(kernel);
    let op = Operator::coulomb();
    let res = match kind {
        Kind::Rhf => solve_rhf(&ctx, mol, &prep, op, &bounds, &config),
        Kind::Uhf => solve_uhf(&ctx, mol, &prep, &bounds, &config),
        Kind::Rohf => solve_rohf(&ctx, mol, &prep, op, &bounds, &config),
    };
    let res = res.unwrap();
    assert!(res.converged, "{name} {kernel:?} did not converge");
    (res.energy, res.iterations)
}

/// Whole-SCF energies with the kernel on against off: RHF, UHF, ROHF.
#[test]
fn scf_energies_match_the_libint_walk() {
    for (stem, mult, name, kind) in [
        ("water", 1, "cc-pvdz", Kind::Rhf),
        ("c2h4", 1, "aug-cc-pvdz", Kind::Rhf),
        ("water", 1, "cc-pvtz", Kind::Rhf),
        ("oh", 2, "cc-pvdz", Kind::Uhf),
        ("oh", 2, "aug-cc-pvdz", Kind::Rohf),
    ] {
        let mol = load(stem, mult);
        let (e0, it0) = scf(&mol, name, GcKernel::Off, kind);
        let (e1, it1) = scf(&mol, name, GcKernel::Auto, kind);
        let de = (e1 - e0).abs();
        eprintln!(
            "{stem}/{name} {kind:?}: E {e1:.12} vs off {e0:.12} ({:#x} vs {:#x}), |dE| {de:.2e}, \
             iterations {it1} vs {it0}",
            e1.to_bits(),
            e0.to_bits()
        );
        assert!(de <= E_TOL, "{stem}/{name} {kind:?}: |dE| = {de:e}");
    }
}

/// A whole SCF with the kernel on is bitwise identical at 1 and 4 threads.
#[test]
fn kernel_scf_is_bitwise_across_thread_counts() {
    let mol = load("ch3oh", 1);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| scf(&mol, "cc-pvdz", GcKernel::Auto, Kind::Rhf).0)
    };
    let e1 = run(1);
    for t in [2, 4] {
        let e = run(t);
        assert_eq!(e.to_bits(), e1.to_bits(), "{t} threads: {e} vs {e1}");
    }
}
