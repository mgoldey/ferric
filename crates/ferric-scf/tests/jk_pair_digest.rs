//! The bra-pair-local J/K digest (`[scf] jk_digest = "pair"`, the default)
//! against the 8-fold scatter it replaced (`"legacy"`).
//!
//! # What can move
//!
//! Both digests contract the same libint2 blocks of the same screened walk in
//! the same group partition; only the bracketing of each element's sum moves
//! (the pair digest accumulates bra-shell rows and forms `J = Jh + Jhᵀ`,
//! `K = Kb + Kbᵀ` or `Kb + Ktᵀ` once per build). So every element agrees to
//! rounding: here `|X_pair − X_legacy| ≤ max(1e-12, 1e-13 · max|X|)`, and SCF
//! energies within 1e-10 Ha.
//!
//! # That the pair digest ran
//!
//! A tolerance alone would pass if a builder silently fell back to the legacy
//! scatter. Every comparison therefore also reads the pair digest's probe
//! (`quartet_scatter::pair_probe`, recorded per finished build on the calling
//! thread): the pair run must record its builds, each with a positive count
//! of quartets digested by the pair kernel and on the expected kernel
//! (symmetric for a density symmetric to roundoff, general otherwise), and
//! the legacy run must record none.
//!
//! # Artifact hypothesis
//!
//! A dropped image or a wrong degenerate-case weight (the `½` on `Jh` for
//! `s1 = s2` or `s3 = s4`, the `½` on K for the self-paired quartet, the
//! `w = 1` of the general form), swapped bra rows, `D` in place of `Dᵀ` in
//! the general form, a skipped or untransposed `finish`, or the symmetric
//! kernel on a non-symmetric density moves elements by O(|X|), ten or more
//! orders above the bar.
//!
//! The bases cover the all-s path (STO-3G), Cartesian d (`6-31G*`, n = 6),
//! spherical d (cc-pVDZ, n = 5, the const-generic kernel) and f (cc-pVTZ,
//! n = 7, the runtime kernel); every walk includes `s1 = s2`, `s3 = s4` and
//! the self-paired quartet. The closed-shell, open-shell, incremental,
//! J-only, K-only and pooled builders are each compared, on symmetric and
//! non-symmetric densities, and the rotated (contraction-rotation) walk too.
//!
//! The config value reaching the consumers is tested in
//! [`legacy_config_reaches_the_consumers_bitwise`]: no test sets
//! `FERRIC_JK_DIGEST` in its own process (only on child processes).
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_j::DirectJ;
use ferric_scf::direct_jk::{DirectJK, GcRotate};
use ferric_scf::direct_k::DirectK;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::incore_jk::EriMode;
use ferric_scf::quartet_scatter::pair_probe::{self, PairBuild};
use ferric_scf::quartet_scatter::JkDigest;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const THRESH: f64 = 1e-12;

fn testdata(rel: &str) -> String {
    format!("{}/../../testdata/{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn load(xyz: &str, mult: usize) -> Molecule {
    let text = std::fs::read_to_string(testdata(&format!("molecules/{xyz}"))).unwrap();
    Molecule::parse_xyz(&text, 0, mult).unwrap()
}

fn setup(xyz: &str, basis_name: &str, mult: usize) -> PreparedBasis {
    let bs = basis::bundled(basis_name).unwrap();
    PreparedBasis::new(&load(xyz, mult), &bs).unwrap()
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A density with an exponential decay away from the diagonal (so the
/// density screen has something to prune, as for a real density), symmetric
/// or not.
fn density(n: usize, seed: u64, symmetric: bool) -> Array2<f64> {
    let mut s = seed;
    let mut d = Array2::from_shape_fn((n, n), |(i, j)| {
        lcg(&mut s) * (-0.05 * (i as f64 - j as f64).abs()).exp()
    });
    if symmetric {
        d = 0.5 * (&d + &d.t());
    }
    d
}

/// Bounds whose direct builds run `digest`.
fn bounds(prep: &PreparedBasis, digest: JkDigest) -> SchwarzBounds {
    let b = SchwarzBounds::compute(Operator::coulomb(), prep).unwrap();
    b.digest.set(Some(digest));
    b
}

/// Which pair kernel a run must have used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kernel {
    Symmetric,
    General,
}

impl Kernel {
    fn of(symmetric: bool) -> Self {
        if symmetric {
            Kernel::Symmetric
        } else {
            Kernel::General
        }
    }
}

/// A run's output with the pair-digest builds it recorded.
struct Ran<T> {
    out: T,
    builds: Vec<PairBuild>,
}

/// `f` with the pair probe on.
fn probed<T>(f: impl FnOnce() -> T) -> Ran<T> {
    pair_probe::start();
    let out = f();
    Ran {
        out,
        builds: pair_probe::take(),
    }
}

/// The pair run recorded `n_builds` builds, each on `kernel` with quartets
/// digested by the pair kernel; the legacy run recorded none.
fn assert_ran(
    what: &str,
    pair: &[PairBuild],
    legacy: &[PairBuild],
    kernel: Kernel,
    n_builds: usize,
) {
    assert!(
        legacy.is_empty(),
        "{what}: the legacy run went through the pair digest: {legacy:?}"
    );
    assert_eq!(
        pair.len(),
        n_builds,
        "{what}: pair-digest builds recorded {pair:?}"
    );
    for b in pair {
        assert!(
            b.quartets > 0,
            "{what}: the pair kernel digested nothing: {b:?}"
        );
        assert_eq!(
            b.general,
            kernel == Kernel::General,
            "{what}: expected the {kernel:?} kernel, got {b:?}"
        );
    }
}

/// Largest `|a − b|`, asserting it is under `max(1e-12, 1e-13 · max|b|)` and
/// that the comparison is not vacuous.
fn within(what: &str, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        scale > 1e-3,
        "{what}: max|X| = {scale:e}, a vacuous comparison"
    );
    let diff = a.iter().zip(b.iter()).fold(0.0f64, |m, (x, y)| {
        let d = (x - y).abs();
        assert!(d.is_finite(), "{what}: non-finite difference {x} vs {y}");
        m.max(d)
    });
    let tol = (1e-13 * scale).max(1e-12);
    assert!(
        diff <= tol,
        "{what}: max|pair - legacy| = {diff:e} above {tol:e} (max|X| = {scale:e})"
    );
    eprintln!("{what}: max|X| {scale:.3e}, max|pair - legacy| {diff:.2e}");
    diff
}

/// Pair against legacy for every matrix the two runs returned: within the
/// rounding bar, and the pair run really ran the pair digest (`n_builds`
/// builds on `kernel`) while the legacy run did not.
fn agree(
    what: &str,
    pair: &Ran<Vec<Array2<f64>>>,
    legacy: &Ran<Vec<Array2<f64>>>,
    kernel: Kernel,
    n_builds: usize,
) {
    assert_ran(what, &pair.builds, &legacy.builds, kernel, n_builds);
    assert_eq!(pair.out.len(), legacy.out.len());
    for (i, (p, l)) in pair.out.iter().zip(&legacy.out).enumerate() {
        within(&format!("{what} [{i}]"), p, l);
    }
}

fn zeros(n: usize) -> Array2<f64> {
    Array2::zeros((n, n))
}

/// `DirectJK::build` (unrotated) under `digest`: `[J, K]`.
fn closed(prep: &PreparedBasis, digest: JkDigest, d: &Array2<f64>) -> Ran<Vec<Array2<f64>>> {
    closed_at(prep, digest, d, THRESH)
}

/// [`closed`] at integral threshold `thresh`.
fn closed_at(
    prep: &PreparedBasis,
    digest: JkDigest,
    d: &Array2<f64>,
    thresh: f64,
) -> Ran<Vec<Array2<f64>>> {
    probed(|| {
        let n = prep.nbasis();
        let ctx = ParallelContext::default();
        let b = bounds(prep, digest);
        let mut djk =
            DirectJK::new(&ctx, prep, &b, thresh, usize::MAX / 4).with_gc_rotation(GcRotate::Off);
        let (mut j, mut k) = (zeros(n), zeros(n));
        djk.build(d, &mut j, &mut k).unwrap();
        vec![j, k]
    })
}

/// Closed shell, symmetric and non-symmetric densities.
fn check_closed(label: &str, prep: &PreparedBasis, seed: u64) {
    let n = prep.nbasis();
    for (sym, tag) in [(true, "sym"), (false, "nonsym")] {
        let d = density(n, seed, sym);
        let p = closed(prep, JkDigest::Pair, &d);
        let l = closed(prep, JkDigest::Legacy, &d);
        agree(&format!("{label} {tag} J, K"), &p, &l, Kernel::of(sym), 1);
        if sym {
            // K of a symmetric density comes out exactly symmetric.
            let kp = &p.out[1];
            assert!(
                kp == kp.t(),
                "{label}: pair K of a symmetric density is not symmetric"
            );
        }
    }
}

#[test]
fn pair_matches_legacy_sto3g_all_s_path() {
    check_closed("water/STO-3G", &setup("water.xyz", "sto-3g", 1), 1);
    // H2: two s shells, nearly every quartet degenerate (s1 = s2, s3 = s4,
    // self-paired).
    check_closed("H2/STO-3G", &setup("h2.xyz", "sto-3g", 1), 2);
}

#[test]
fn pair_matches_legacy_631gs_cartesian_d() {
    let prep = setup("water.xyz", "6-31g*", 1);
    assert!(
        prep.shell_dims().contains(&6),
        "6-31G* water must have a Cartesian d shell"
    );
    check_closed("water/6-31G*", &prep, 3);
}

#[test]
fn pair_matches_legacy_cc_pvdz() {
    check_closed("water/cc-pVDZ", &setup("water.xyz", "cc-pvdz", 1), 4);
    check_closed("benzene/cc-pVDZ", &setup("benzene.xyz", "cc-pvdz", 1), 5);
}

#[test]
fn pair_matches_legacy_cc_pvtz_f_shells() {
    let prep = setup("water.xyz", "cc-pvtz", 1);
    assert!(
        prep.shell_dims().contains(&7),
        "cc-pVTZ water must have an f shell"
    );
    check_closed("water/cc-pVTZ", &prep, 6);
}

/// A full build judges symmetry relative to the density itself: a purely
/// antisymmetric density of absolute size 1e-15 takes the general kernel
/// (and agrees with legacy relative to its own size), so no absolute floor
/// symmetrises a genuinely antisymmetric part away. (Integral threshold
/// 1e-30, so the density screen keeps the quartets of so small a density.)
#[test]
fn tiny_antisymmetric_density_takes_the_general_kernel() {
    let prep = setup("water.xyz", "cc-pvdz", 1);
    let n = prep.nbasis();
    let r = density(n, 9, false);
    let a = 0.5e-15 * (&r - &r.t());
    let p = closed_at(&prep, JkDigest::Pair, &a, 1e-30);
    let l = closed_at(&prep, JkDigest::Legacy, &a, 1e-30);
    assert_ran(
        "antisymmetric 1e-15",
        &p.builds,
        &l.builds,
        Kernel::General,
        1,
    );
    let scale = |x: &Array2<f64>| x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    // J of an antisymmetric density is zero; K is not.
    let (kp, kl) = (&p.out[1], &l.out[1]);
    assert!(scale(kl) > 1e-17, "vacuous: max|K| {:e}", scale(kl));
    let diff = kp.iter().zip(kl.iter()).fold(0.0f64, |m, (x, y)| {
        let d = (x - y).abs();
        assert!(
            d.is_finite(),
            "antisymmetric K: non-finite difference {x} vs {y}"
        );
        m.max(d)
    });
    assert!(
        diff <= 1e-13 * scale(kl),
        "antisymmetric K: max|pair - legacy| {diff:e}, max|K| {:e}",
        scale(kl)
    );
    // The symmetric kernel would have returned K of ½(D + Dᵀ) = 0.
    assert!(scale(kp) > 0.5 * scale(kl));
}

/// `DirectJK::build(D0)` then `build_incremental(ΔD)` on one builder under
/// `digest`: `[J, K]` after both.
fn closed_incremental(
    prep: &PreparedBasis,
    digest: JkDigest,
    d0: &Array2<f64>,
    dd: &Array2<f64>,
) -> Ran<Vec<Array2<f64>>> {
    probed(|| {
        let n = prep.nbasis();
        let ctx = ParallelContext::default();
        let b = bounds(prep, digest);
        let mut djk =
            DirectJK::new(&ctx, prep, &b, THRESH, usize::MAX / 4).with_gc_rotation(GcRotate::Off);
        let (mut j, mut k) = (zeros(n), zeros(n));
        djk.build(d0, &mut j, &mut k).unwrap();
        djk.build_incremental(dd, &mut j, &mut k).unwrap();
        vec![j, k]
    })
}

/// The incremental build, pair against legacy: a full build at `D0` then
/// `ΔD = 1e-7 · (symmetric)` with a 1e-17 skew (the rounding asymmetry a
/// difference of two O(1) densities carries). Judged against the full
/// density's scale, `ΔD` stays on the symmetric kernel (both builds
/// symmetric); a full build of the same `ΔD` alone takes the general one.
/// A non-symmetric `ΔD` takes the general kernel on the incremental build.
#[test]
fn pair_matches_legacy_incremental() {
    let prep = setup("water.xyz", "cc-pvdz", 1);
    let n = prep.nbasis();
    let d0 = density(n, 7, true);
    let mut dd = 1e-7 * &density(n, 8, true);
    dd[(0, 1)] += 1e-17;
    let p = closed_incremental(&prep, JkDigest::Pair, &d0, &dd);
    let l = closed_incremental(&prep, JkDigest::Legacy, &d0, &dd);
    agree("incremental ΔD (1e-17 skew)", &p, &l, Kernel::Symmetric, 2);
    let alone = closed(&prep, JkDigest::Pair, &dd);
    assert_eq!(alone.builds.len(), 1);
    assert!(
        alone.builds[0].general,
        "a full build of a 1e-7 ΔD with a 1e-17 skew must judge it on its own scale"
    );
    // A genuinely non-symmetric ΔD on the incremental path: general there.
    let mut skewed = dd.clone();
    skewed[(2, 9)] += 1e-9;
    let p = closed_incremental(&prep, JkDigest::Pair, &d0, &skewed);
    let l = closed_incremental(&prep, JkDigest::Legacy, &d0, &skewed);
    assert!(
        l.builds.is_empty(),
        "legacy incremental went through the pair digest"
    );
    assert!(p.builds.iter().all(|b| b.quartets > 0));
    let kinds: Vec<bool> = p.builds.iter().map(|b| b.general).collect();
    assert_eq!(
        kinds,
        vec![false, true],
        "full D0 symmetric, skewed ΔD general"
    );
    for (i, (x, y)) in p.out.iter().zip(&l.out).enumerate() {
        within(&format!("incremental skewed ΔD [{i}]"), x, y);
    }
}

/// `build_uhf(D0)` then `build_uhf_incremental(ΔD)` on one builder.
fn open_incremental(
    prep: &PreparedBasis,
    digest: JkDigest,
    d0: [&Array2<f64>; 2],
    dd: [&Array2<f64>; 2],
) -> Ran<Vec<Array2<f64>>> {
    probed(|| {
        let n = prep.nbasis();
        let ctx = ParallelContext::default();
        let b = bounds(prep, digest);
        let mut djk =
            DirectJK::new(&ctx, prep, &b, THRESH, usize::MAX / 4).with_gc_rotation(GcRotate::Off);
        let (mut j, mut ka, mut kb) = (zeros(n), zeros(n), zeros(n));
        djk.build_uhf(&(d0[0] + d0[1]), d0[0], d0[1], &mut j, &mut ka, &mut kb)
            .unwrap();
        djk.build_uhf_incremental(&(dd[0] + dd[1]), dd[0], dd[1], &mut j, &mut ka, &mut kb)
            .unwrap();
        vec![j, ka, kb]
    })
}

/// The open-shell incremental build, pair against legacy, with opposing
/// spin deltas carrying a 1e-17 skew: both builds on the symmetric kernel
/// (each spin judged against its own full density's scale).
#[test]
fn pair_matches_legacy_uhf_incremental() {
    let prep = setup("oh.xyz", "cc-pvdz", 2);
    let n = prep.nbasis();
    let (da, db) = (density(n, 61, true), density(n, 62, true));
    let mut dda = 1e-7 * &density(n, 63, true);
    dda[(1, 3)] += 1e-17;
    let mut ddb = -1.0 * &dda;
    ddb[(4, 2)] -= 1e-17;
    let p = open_incremental(&prep, JkDigest::Pair, [&da, &db], [&dda, &ddb]);
    let l = open_incremental(&prep, JkDigest::Legacy, [&da, &db], [&dda, &ddb]);
    agree("UHF incremental", &p, &l, Kernel::Symmetric, 2);
    assert!(p.builds.iter().all(|b| b.symmetric_inputs == [true, true]));
}

/// Open shell with α ≠ β: symmetric spins, then one non-symmetric spin (the
/// general kernel on both channels).
#[test]
fn pair_matches_legacy_open_shell() {
    let prep = setup("oh.xyz", "cc-pvdz", 2);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    for (sym, da, db) in [
        (true, density(n, 11, true), density(n, 12, true)),
        (false, density(n, 13, true), density(n, 14, false)),
    ] {
        let dt = &da + &db;
        let run = |digest| {
            probed(|| {
                let b = bounds(&prep, digest);
                let mut djk = DirectJK::new(&ctx, &prep, &b, THRESH, usize::MAX / 4)
                    .with_gc_rotation(GcRotate::Off);
                let (mut j, mut ka, mut kb) = (zeros(n), zeros(n), zeros(n));
                djk.build_uhf(&dt, &da, &db, &mut j, &mut ka, &mut kb)
                    .unwrap();
                vec![j, ka, kb]
            })
        };
        let p = run(JkDigest::Pair);
        agree(
            &format!("OH UHF sym={sym} J, Ka, Kb"),
            &p,
            &run(JkDigest::Legacy),
            Kernel::of(sym),
            1,
        );
        assert_eq!(p.builds[0].symmetric_inputs, [true, sym]);
    }
}

/// `DirectJ` (J only) and `DirectK` (K only), symmetric and not.
#[test]
fn pair_matches_legacy_j_only_and_k_only() {
    let prep = setup("water.xyz", "cc-pvdz", 1);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    for sym in [true, false] {
        let d = density(n, 21, sym);
        let j_of = |digest| {
            probed(|| {
                let b = bounds(&prep, digest);
                let mut x = zeros(n);
                JBuilder::build(
                    &mut DirectJ::new(&ctx, &prep, &b, THRESH, usize::MAX / 4),
                    &d,
                    &mut x,
                )
                .unwrap();
                vec![x]
            })
        };
        let k_of = |digest| {
            probed(|| {
                let b = bounds(&prep, digest);
                let mut x = zeros(n);
                KBuilder::build(
                    &mut DirectK::new(&ctx, &prep, &b, THRESH, usize::MAX / 4),
                    &d,
                    &mut x,
                )
                .unwrap();
                vec![x]
            })
        };
        // J has no exchange channel, so never the general kernel.
        let jp = j_of(JkDigest::Pair);
        agree(
            &format!("DirectJ sym={sym}"),
            &jp,
            &j_of(JkDigest::Legacy),
            Kernel::Symmetric,
            1,
        );
        assert!(jp.builds[0].coulomb && jp.builds[0].symmetric_inputs.is_empty());
        let kp = k_of(JkDigest::Pair);
        agree(
            &format!("DirectK sym={sym}"),
            &kp,
            &k_of(JkDigest::Legacy),
            Kernel::of(sym),
            1,
        );
        assert!(!kp.builds[0].coulomb);
    }
}

/// `rhf::build_jk` / `build_jk_with_pool` (the response, Newton and stability
/// builds).
#[test]
fn pair_matches_legacy_pooled_build() {
    let prep = setup("water.xyz", "6-31g*", 1);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    for sym in [true, false] {
        let d = density(n, 31, sym);
        let run = |digest| {
            probed(|| {
                let b = bounds(&prep, digest);
                let (mut j, mut k) = (zeros(n), zeros(n));
                ferric_scf::rhf::build_jk(&ctx, &prep, &b, THRESH, &d, &mut j, &mut k).unwrap();
                vec![j, k]
            })
        };
        agree(
            &format!("build_jk sym={sym} J, K"),
            &run(JkDigest::Pair),
            &run(JkDigest::Legacy),
            Kernel::of(sym),
            1,
        );
    }
}

/// The rotated walk (`gc_rotation = auto` on cc-pVDZ) needs no change of its
/// own: its rotated bounds share the parent's digest record, so the rotated
/// build runs the pair kernel under `pair` and not under `legacy`, and the
/// two agree to rounding.
#[test]
fn pair_matches_legacy_on_the_rotated_walk() {
    let prep = setup("water.xyz", "cc-pvdz", 1);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    let d = density(n, 41, true);
    let run = |digest| {
        probed(|| {
            let b = bounds(&prep, digest);
            let mut djk = DirectJK::new(&ctx, &prep, &b, THRESH, usize::MAX / 4)
                .with_gc_rotation(GcRotate::Auto);
            let (mut j, mut k) = (zeros(n), zeros(n));
            djk.build(&d, &mut j, &mut k).unwrap();
            assert!(djk.gc_rotated_columns().unwrap() > 0, "cc-pVDZ must rotate");
            vec![j, k]
        })
    };
    agree(
        "rotated J, K",
        &run(JkDigest::Pair),
        &run(JkDigest::Legacy),
        Kernel::Symmetric,
        1,
    );
}

/// `legacy` is reached through the bounds record and the pair digest is the
/// default: unrecorded bounds (no SCF, `FERRIC_JK_DIGEST` unset in the test
/// harness or set by the caller) agree with the variable's choice.
#[test]
fn unrecorded_bounds_follow_the_variable() {
    let prep = setup("water.xyz", "sto-3g", 1);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    let d = density(n, 51, true);
    let b = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    assert_eq!(b.digest.recorded(), None);
    let want = JkDigest::env_setting().unwrap();
    assert_eq!(b.digest.mode().unwrap(), want);
    let mut djk = DirectJK::new(&ctx, &prep, &b, THRESH, usize::MAX / 4);
    let (mut j, mut k) = (zeros(n), zeros(n));
    djk.build(&d, &mut j, &mut k).unwrap();
    let w = closed(&prep, want, &d);
    assert!(
        j == w.out[0] && k == w.out[1],
        "unrecorded bounds did not run {want}"
    );
}

/// Whole SCF on the direct route, pair against legacy: water / cc-pVDZ RHF
/// (closed-shell incremental builds on, as by default) and OH / 6-31G* UHF.
/// Energies within 1e-10 Ha. Every build of the pair SCF went through the
/// pair digest on the symmetric kernel (the incremental `ΔD` builds
/// included), and the legacy SCF never did.
#[test]
fn whole_scf_pair_matches_legacy() {
    let ctx = ParallelContext::default();
    for (label, xyz, basis_name, mult) in [
        ("water/cc-pVDZ RHF", "water.xyz", "cc-pvdz", 1),
        ("OH/6-31G* UHF", "oh.xyz", "6-31g*", 2),
    ] {
        let mol = load(xyz, mult);
        let prep = setup(xyz, basis_name, mult);
        let op = Operator::coulomb();
        let run = |digest: JkDigest| {
            probed(|| {
                let b = SchwarzBounds::compute(op, &prep).unwrap();
                let mut c = RhfConfig::default()
                    .with_energy_conv(1e-8)
                    .with_density_conv(1e-8);
                c.eri = EriMode::Direct;
                c.jk_digest = Some(digest);
                let r = if mult == 1 {
                    solve_rhf(&ctx, &mol, &prep, op, &b, &c).unwrap()
                } else {
                    ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &b, &c).unwrap()
                };
                assert!(r.converged, "{label} {digest}: not converged");
                (r.energy, r.iterations)
            })
        };
        let p = run(JkDigest::Pair);
        let l = run(JkDigest::Legacy);
        assert!(
            l.builds.is_empty(),
            "{label}: legacy SCF used the pair digest"
        );
        assert!(
            p.builds.len() >= p.out.1,
            "{label}: {} pair builds for {} iterations",
            p.builds.len(),
            p.out.1
        );
        for b in &p.builds {
            assert!(!b.general && b.quartets > 0, "{label}: {b:?}");
        }
        let de = (p.out.0 - l.out.0).abs();
        eprintln!(
            "{label}: E pair {:.12} legacy {:.12} |dE| {de:.2e} ({} pair builds)",
            p.out.0,
            l.out.0,
            p.builds.len()
        );
        assert!(de <= 1e-10, "{label}: |E_pair - E_legacy| = {de:e}");
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
const CHILD: &str = "FERRIC_TEST_JK_DIGEST_CHILD";

/// One consumer output: its name, the matrix, and whether it is in the MO
/// basis (its rows and columns then carry the SCF's orbital phases).
struct Output {
    name: String,
    x: Array2<f64>,
    mo: bool,
}

/// The consumers' outputs, with the pair-digest builds each consumer step
/// recorded (`(step, builds)`).
struct Consumers {
    out: Vec<Output>,
    builds: Vec<(String, usize)>,
}

/// An SCF whose config carries `digest`, then the consumers that follow
/// its bounds' record: water / cc-pVDZ PBE0 on the direct route (SCF,
/// analytic KS gradient, KS Z-vector relaxation gradient and `Z`, pooled
/// J/K), and water / 6-31G wB97X (its SCF and Z-vector: the
/// range-separated response builds make fresh bounds per operator, which
/// copy the record). Returned as matrices (energies as 1×1) for bitwise
/// comparison, with the pair-digest builds of each step.
fn consumers(digest: Option<JkDigest>) -> Consumers {
    use ferric_scf::ks_gradient::ks_gradient_closed;
    use ferric_scf::zvector_ks::relaxation_gradient_closed;
    let ctx = ParallelContext::default();
    let mol = load("water.xyz", 1);
    let op = Operator::coulomb();
    let mut out = Vec::new();
    let mut builds = Vec::new();
    let mut push = |name: String, x: Array2<f64>, mo: bool| out.push(Output { name, x, mo });
    for (xc, basis_name) in [("PBE0", "cc-pvdz"), ("HYB_GGA_XC_WB97X", "6-31g")] {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let mut cfg = RhfConfig::default()
            .with_energy_conv(1e-10)
            .with_density_conv(1e-8)
            .with_gc_rotation(GcRotate::Off);
        cfg.xc = Some(xc.into());
        // Four-centre J (a functional defaults to RI-JK); four-centre K for
        // PBE0 (the range-separated SCF builds its K by RI-K only, but its
        // Z-vector's response K is four-centre, on fresh bounds per
        // operator).
        cfg.df_j_aux = Some(String::new());
        if xc == "PBE0" {
            cfg.df_k_aux = Some(String::new());
        }
        cfg.eri = EriMode::Direct;
        cfg.jk_digest = digest;
        let res = probed(|| solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap());
        builds.push((format!("{xc} SCF"), res.builds.len()));
        let res = res.out;
        assert!(res.converged, "{xc}: SCF not converged");
        push(
            format!("{xc} energy"),
            Array2::from_elem((1, 1), res.energy),
            false,
        );
        push(format!("{xc} density"), res.density_alpha.clone(), false);
        let n = prep.nbasis();
        let v = density(n, 3, true);
        let z = probed(|| {
            relaxation_gradient_closed(&ctx, &mol, &prep, &bs, op, &bounds, &cfg, &res, &v).unwrap()
        });
        builds.push((format!("{xc} Z-vector"), z.builds.len()));
        push(format!("{xc} Z-vector gradient"), z.out.gradient, false);
        push(format!("{xc} Z"), z.out.z, true);
        if xc == "PBE0" {
            let g = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, xc, &res, None).unwrap();
            push(format!("{xc} KS gradient"), g, false);
            let d = density(n, 2, true);
            let (mut j, mut k) = (zeros(n), zeros(n));
            let pooled = probed(|| {
                ferric_scf::rhf::build_jk(&ctx, &prep, &bounds, THRESH, &d, &mut j, &mut k).unwrap()
            });
            builds.push((format!("{xc} pooled J/K"), pooled.builds.len()));
            push(format!("{xc} pooled J"), j, false);
            push(format!("{xc} pooled K"), k, false);
        }
    }
    Consumers { out, builds }
}

/// `p` with the phases of `l`'s MO rows and columns: `p ≈ R l C` with `R`,
/// `C` diagonal signs (an orbital's sign is arbitrary), read off the
/// elements above 1e-6 of `max|l|` by propagating from the largest one of
/// each connected component, then undone. A single sign-reversed element does not fit `R l C`, so it
/// still shows in the comparison.
fn align_phases(p: &Array2<f64>, l: &Array2<f64>) -> Array2<f64> {
    let (nr, nc) = l.dim();
    let big = l.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let sig = |i: usize, j: usize| l[(i, j)].abs() > 1e-6 * big;
    let (mut r, mut c) = (vec![0.0f64; nr], vec![0.0f64; nc]);
    // One connected component of significant elements at a time: seed its
    // row of the largest unassigned element, propagate, repeat.
    loop {
        let seed = (0..nr * nc)
            .filter(|&x| r[x / nc] == 0.0 && sig(x / nc, x % nc))
            .max_by(|&x, &y| {
                l[(x / nc, x % nc)]
                    .abs()
                    .total_cmp(&l[(y / nc, y % nc)].abs())
            });
        let Some(x) = seed else {
            break;
        };
        r[x / nc] = 1.0;
        loop {
            let mut changed = false;
            for i in 0..nr {
                for j in 0..nc {
                    if !sig(i, j) {
                        continue;
                    }
                    let s = (p[(i, j)] * l[(i, j)]).signum();
                    if r[i] != 0.0 && c[j] == 0.0 {
                        c[j] = s * r[i];
                        changed = true;
                    } else if c[j] != 0.0 && r[i] == 0.0 {
                        r[i] = s * c[j];
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
    let unit = |v: f64| if v == 0.0 { 1.0 } else { v };
    Array2::from_shape_fn((nr, nc), |(i, j)| p[(i, j)] * unit(r[i]) * unit(c[j]))
}

/// `align_phases` aligns every connected component of significant
/// elements, not only the largest element's, and leaves a sign-reversed
/// element that does not fit `R l C` visible.
#[test]
fn align_phases_aligns_every_component() {
    let l = Array2::from_diag(&ndarray::arr1(&[1.0, 0.5]));
    let p = Array2::from_diag(&ndarray::arr1(&[1.0, -0.5]));
    assert_eq!(align_phases(&p, &l), l);
    let l = ndarray::arr2(&[[1.0, 0.2, 0.0], [0.3, 0.4, 0.0], [0.0, 0.0, 0.7]]);
    // R l C with R = diag(1, −1, 1), C = diag(−1, 1, −1): two components.
    let phased = ndarray::arr2(&[[-1.0, 0.2, 0.0], [0.3, -0.4, 0.0], [0.0, 0.0, -0.7]]);
    assert_eq!(align_phases(&phased, &l), l);
    let one_flipped = ndarray::arr2(&[[1.0, -0.2, 0.0], [0.3, 0.4, 0.0], [0.0, 0.0, 0.7]]);
    assert!(
        align_phases(&one_flipped, &l) != l,
        "a single sign-reversed element must stay visible"
    );
}

/// The consumers' bits from a child process of this binary with an unset
/// config `jk_digest` and `FERRIC_JK_DIGEST=legacy`: `(hash, name)` per
/// output, in order.
fn child_bits() -> Vec<(u64, String)> {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["child_prints_unset_config_bits", "--exact", "--nocapture"])
        .env(CHILD, "1")
        .env("FERRIC_JK_DIGEST", "legacy")
        .output()
        .expect("spawn the child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "child failed: {stdout}");
    // One `JK_DIGEST_BITS <hash> <name>` line per output.
    stdout
        .lines()
        .filter_map(|l| l.strip_prefix("JK_DIGEST_BITS "))
        .map(|l| {
            let (h, name) = l.split_once(' ').expect("hash and name");
            (u64::from_str_radix(h, 16).unwrap(), name.trim().to_string())
        })
        .collect()
}

/// The config value reaches the J/K the consumers use: config `legacy` is
/// bitwise an unset config under `FERRIC_JK_DIGEST=legacy` (a child process
/// of this binary with the variable set; no test mutates its own
/// environment), over the SCF, the KS gradient, the Z-vector (PBE0 and the
/// range-separated wB97X) and pooled J/K, output by output. Each step that
/// runs direct builds (the SCFs, the Z-vectors, pooled J/K) went through
/// the pair digest under config `pair` and never under `legacy`, so a step
/// that dropped the setting is named. Config `pair` is not bitwise legacy,
/// and agrees with it within 1e-8 relative, signed (the Z-vector gradient,
/// the loosest, measured at about 2e-9; `Z` after aligning the MO phases).
#[test]
fn legacy_config_reaches_the_consumers_bitwise() {
    let legacy = consumers(Some(JkDigest::Legacy));
    let child = child_bits();
    assert_eq!(child.len(), legacy.out.len(), "child outputs: {child:?}");
    let differ: Vec<&str> = child
        .iter()
        .zip(&legacy.out)
        .filter(|((h, _), o)| *h != hash_bits([&o.x]))
        .map(|((_, name), _)| name.as_str())
        .collect();
    assert!(
        differ.is_empty(),
        "config legacy is not the unset config under FERRIC_JK_DIGEST=legacy: {differ:?}"
    );
    let pair = consumers(Some(JkDigest::Pair));
    for ((step, np), (_, nl)) in pair.builds.iter().zip(&legacy.builds) {
        eprintln!("{step}: {np} pair-digest builds under pair, {nl} under legacy");
        assert!(
            *np > 0,
            "{step}: config pair did not reach its direct builds"
        );
        assert_eq!(*nl, 0, "{step}: config legacy ran the pair digest");
    }
    assert_ne!(
        hash_bits(pair.out.iter().map(|o| &o.x)),
        hash_bits(legacy.out.iter().map(|o| &o.x)),
        "config pair is bitwise legacy: the switch did not reach the builds"
    );
    for (p, l) in pair.out.iter().zip(&legacy.out) {
        let name = &p.name;
        let px = if p.mo {
            align_phases(&p.x, &l.x)
        } else {
            p.x.clone()
        };
        let e = px.iter().zip(l.x.iter()).fold(0.0f64, |m, (x, y)| {
            let d = (x - y).abs();
            assert!(d.is_finite(), "{name}: non-finite difference {x} vs {y}");
            m.max(d)
        });
        let scale = l.x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        eprintln!("{name}: pair vs legacy max|d| {e:.2e} (max|X| {scale:.2e})");
        assert!(e <= 1e-8 * scale.max(1.0), "{name}: pair vs legacy {e:e}");
    }
}

/// Child entry point of `legacy_config_reaches_the_consumers_bitwise`
/// ([`child_bits`]): the consumers of an SCF whose config left `jk_digest`
/// unset, each output's bits printed. A no-op outside the child.
#[test]
fn child_prints_unset_config_bits() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    for o in consumers(None).out {
        println!("JK_DIGEST_BITS {:016x} {}", hash_bits([&o.x]), o.name);
    }
}

/// The open-shell guess Fock builds run the legacy digest whatever the
/// SCF's (`SchwarzBounds::for_guess`): on fitted J and K the guess's are
/// the only direct builds, so ROHF and UHF on config `pair` and `legacy`
/// are bitwise one run, and none of their builds went through the pair
/// digest. (The guess Fock of a linear open shell has a degenerate π pair
/// whose last bits pick the hole orientation; see `for_guess`.)
#[test]
fn open_shell_guess_builds_do_not_follow_the_digest() {
    let mol = load("oh.xyz", 2);
    let prep = setup("oh.xyz", "6-31g*", 2);
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    for rohf in [true, false] {
        let run = |digest: JkDigest| {
            probed(|| {
                let b = SchwarzBounds::compute(op, &prep).unwrap();
                let mut c = RhfConfig::default();
                c.df_j_aux = Some("def2-universal-jkfit".into());
                c.df_k_aux = Some("def2-universal-jkfit".into());
                c.jk_digest = Some(digest);
                let r = if rohf {
                    ferric_scf::rohf::solve_rohf(&ctx, &mol, &prep, op, &b, &c).unwrap()
                } else {
                    ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &b, &c).unwrap()
                };
                assert!(r.converged);
                (r.energy, r.density_total().to_owned())
            })
        };
        let (p, l) = (run(JkDigest::Pair), run(JkDigest::Legacy));
        let what = if rohf { "ROHF" } else { "UHF" };
        assert!(
            p.builds.is_empty() && l.builds.is_empty(),
            "{what}: a guess build ran the pair digest: {:?}",
            p.builds
        );
        assert_eq!(p.out.0.to_bits(), l.out.0.to_bits(), "{what} energy");
        assert!(p.out.1 == l.out.1, "{what}: densities differ");
    }
}

/// A density or output that is not `nbf × nbf` is an error at each public
/// direct builder's entry (`DirectJK`'s four, `DirectJ`, `DirectK`,
/// `LinkK`, `build_jk`, `build_jk_with_pool`), before the density screen
/// (which indexes it unchecked) or the kernel runs, under either digest;
/// never a panic.
#[test]
fn wrong_shape_is_an_error_at_every_builder_entry() {
    use ferric_scf::engine_pool::EnginePool;
    let prep = setup("water.xyz", "sto-3g", 1);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    let good = density(n, 61, true);
    let small = density(n - 1, 61, true);
    for digest in [JkDigest::Pair, JkDigest::Legacy] {
        let b = bounds(&prep, digest);
        let pool = EnginePool::new(b.op, &prep, 1e-14).unwrap();
        // (what, density, output dimension)
        for (what, d, out_n) in [("density", &small, n), ("output", &good, n + 1)] {
            let errs: Vec<(&str, Result<usize, ferric_core::FerricError>)> =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let o = || Array2::<f64>::zeros((out_n, out_n));
                    let (mut j, mut k, mut k2) = (o(), o(), o());
                    let mut djk = DirectJK::new(&ctx, &prep, &b, THRESH, usize::MAX / 4);
                    let mut dj = DirectJ::new(&ctx, &prep, &b, THRESH, usize::MAX / 4);
                    let mut dk = DirectK::new(&ctx, &prep, &b, THRESH, usize::MAX / 4);
                    // LinK keeps its own digest; it runs under each bounds
                    // all the same (its density pairs are built lazily in
                    // `build`, after the entry check).
                    let mut link = ferric_scf::link_k::LinkK::new(
                        &ctx,
                        &prep,
                        &b,
                        b.op,
                        THRESH,
                        usize::MAX / 4,
                    );
                    vec![
                        ("DirectJK::build", djk.build(d, &mut j, &mut k)),
                        (
                            "DirectJK::build_incremental",
                            djk.build_incremental(d, &mut j, &mut k),
                        ),
                        (
                            "DirectJK::build_uhf",
                            djk.build_uhf(d, d, d, &mut j, &mut k, &mut k2),
                        ),
                        (
                            "DirectJK::build_uhf_incremental",
                            djk.build_uhf_incremental(d, d, d, &mut j, &mut k, &mut k2),
                        ),
                        ("DirectJ", JBuilder::build(&mut dj, d, &mut j)),
                        ("DirectK", KBuilder::build(&mut dk, d, &mut k)),
                        ("LinkK", KBuilder::build(&mut link, d, &mut k)),
                        (
                            "build_jk",
                            ferric_scf::rhf::build_jk(&ctx, &prep, &b, THRESH, d, &mut j, &mut k),
                        ),
                        (
                            "build_jk_with_pool",
                            ferric_scf::rhf::build_jk_with_pool(
                                &ctx,
                                &prep,
                                &b,
                                THRESH,
                                d,
                                &mut j,
                                &mut k,
                                &pool,
                                1 << 20,
                            ),
                        ),
                    ]
                }))
                .unwrap_or_else(|_| panic!("{digest} {what}: a builder panicked"));
            for (who, r) in errs {
                let e = r.expect_err(&format!("{digest} {what}: {who} accepted a wrong shape"));
                assert!(e.to_string().contains("expected ("), "{who}: {e}");
            }
        }
    }
}
