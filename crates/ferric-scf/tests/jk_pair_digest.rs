//! The bra-pair-local J/K digest (`[scf] jk_digest = "pair"`, the default)
//! against the 8-fold scatter it replaced (`"legacy"`), at a fixed density.
//!
//! # What can move
//!
//! Both digests contract the same libint2 blocks of the same screened walk in
//! the same group partition; only the bracketing of each element's sum moves
//! (the pair digest accumulates bra-shell rows and forms `J = Jh + Jhᵀ`,
//! `K = Kb + Kbᵀ` or `Kb + Ktᵀ` once per build). So every element agrees to
//! rounding: here `|X_pair − X_legacy| ≤ max(1e-12, 1e-13 · max|X|)`.
//!
//! # Artifact hypothesis
//!
//! A dropped image or a wrong degenerate-case weight (the `½` on `Jh` for
//! `s1 = s2` or `s3 = s4`, the `½` on K for the self-paired quartet), swapped
//! bra rows, a skipped `finish`, or the symmetric kernel on a non-symmetric
//! density moves elements by O(|X|), ten or more orders above the bar. The
//! mutation runs that confirm each of these fails are recorded in the change
//! that introduced the digest.
//!
//! The bases cover the all-s path (STO-3G), Cartesian d (`6-31G*`, n = 6),
//! spherical d (cc-pVDZ, n = 5, the const-generic kernel) and f (cc-pVTZ,
//! n = 7, the runtime kernel); every walk includes `s1 = s2`, `s3 = s4` and
//! the self-paired quartet. The closed-shell, open-shell, J-only, K-only and
//! pooled builders are each compared, on symmetric and non-symmetric
//! densities, and the rotated (contraction-rotation) walk too.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_j::DirectJ;
use ferric_scf::direct_jk::{DirectJK, GcRotate};
use ferric_scf::direct_k::DirectK;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::quartet_scatter::JkDigest;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const THRESH: f64 = 1e-12;

fn testdata(rel: &str) -> String {
    format!("{}/../../testdata/{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn setup(xyz: &str, basis_name: &str, mult: usize) -> PreparedBasis {
    let text = std::fs::read_to_string(testdata(&format!("molecules/{xyz}"))).unwrap();
    let mol = Molecule::parse_xyz(&text, 0, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    PreparedBasis::new(&mol, &bs).unwrap()
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

/// Largest `|a − b|`, asserting it is under `max(1e-12, 1e-13 · max|b|)` and
/// that the comparison is not vacuous.
fn agree(what: &str, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        scale > 1e-3,
        "{what}: max|X| = {scale:e}, a vacuous comparison"
    );
    let diff = a
        .iter()
        .zip(b.iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    let tol = (1e-13 * scale).max(1e-12);
    assert!(
        diff <= tol,
        "{what}: max|pair - legacy| = {diff:e} above {tol:e} (max|X| = {scale:e})"
    );
    eprintln!("{what}: max|X| {scale:.3e}, max|pair - legacy| {diff:.2e}");
    diff
}

fn zeros(n: usize) -> Array2<f64> {
    Array2::zeros((n, n))
}

/// `DirectJK::build` (unrotated) under `digest`.
fn closed(prep: &PreparedBasis, digest: JkDigest, d: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    let b = bounds(prep, digest);
    let mut djk =
        DirectJK::new(&ctx, prep, &b, THRESH, usize::MAX / 4).with_gc_rotation(GcRotate::Off);
    let (mut j, mut k) = (zeros(n), zeros(n));
    djk.build(d, &mut j, &mut k).unwrap();
    (j, k)
}

/// Closed shell, symmetric and non-symmetric densities.
fn check_closed(label: &str, prep: &PreparedBasis, seed: u64) {
    let n = prep.nbasis();
    for (sym, tag) in [(true, "sym"), (false, "nonsym")] {
        let d = density(n, seed, sym);
        let (jp, kp) = closed(prep, JkDigest::Pair, &d);
        let (jl, kl) = closed(prep, JkDigest::Legacy, &d);
        agree(&format!("{label} {tag} J"), &jp, &jl);
        agree(&format!("{label} {tag} K"), &kp, &kl);
        if sym {
            // K of a symmetric density comes out exactly symmetric.
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

/// The incremental build's `ΔD`: small. Exactly symmetric, both digests
/// agree to rounding relative to its own size; with a roundoff-size skew it
/// still takes the symmetric kernel (the absolute floor of
/// `quartet_scatter::is_symmetric_to_roundoff`), which moves K by at most the
/// skew times a sum of integrals, far below the integral screen.
#[test]
fn pair_matches_legacy_on_a_small_delta_density() {
    let prep = setup("water.xyz", "cc-pvdz", 1);
    let n = prep.nbasis();
    let dd = 1e-7 * &density(n, 8, true);
    let scale = |x: &Array2<f64>| x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let max_diff = |p: &Array2<f64>, l: &Array2<f64>| {
        p.iter()
            .zip(l.iter())
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
    };
    let (jp, kp) = closed(&prep, JkDigest::Pair, &dd);
    let (jl, kl) = closed(&prep, JkDigest::Legacy, &dd);
    for (what, p, l) in [("J", &jp, &jl), ("K", &kp, &kl)] {
        let diff = max_diff(p, l);
        assert!(scale(l) > 1e-9, "ΔD {what}: vacuous");
        assert!(
            diff <= 1e-13 * scale(l),
            "ΔD {what}: max|pair - legacy| {diff:e}, max|X| {:e}",
            scale(l)
        );
    }
    // The same ΔD with a 1e-17 skew (the rounding asymmetry a difference of
    // two O(1) densities carries).
    let mut skewed = dd.clone();
    skewed[(0, 1)] += 1e-17;
    let (jp, kp) = closed(&prep, JkDigest::Pair, &skewed);
    let (jl, kl) = closed(&prep, JkDigest::Legacy, &skewed);
    for (what, p, l) in [("J", &jp, &jl), ("K", &kp, &kl)] {
        let diff = max_diff(p, l);
        assert!(
            diff <= 1e-15,
            "skewed ΔD {what}: max|pair - legacy| {diff:e}, max|X| {:e}",
            scale(l)
        );
        eprintln!(
            "skewed ΔD {what}: max|X| {:.3e}, max|pair - legacy| {diff:.2e}",
            scale(l)
        );
    }
}

/// Open shell with α ≠ β: symmetric spins, then one non-symmetric spin (the
/// general kernel on both channels).
#[test]
fn pair_matches_legacy_open_shell() {
    let prep = setup("oh.xyz", "cc-pvdz", 2);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    for (tag, da, db) in [
        ("sym", density(n, 11, true), density(n, 12, true)),
        ("nonsym", density(n, 13, true), density(n, 14, false)),
    ] {
        let dt = &da + &db;
        let run = |digest| {
            let b = bounds(&prep, digest);
            let mut djk = DirectJK::new(&ctx, &prep, &b, THRESH, usize::MAX / 4)
                .with_gc_rotation(GcRotate::Off);
            let (mut j, mut ka, mut kb) = (zeros(n), zeros(n), zeros(n));
            djk.build_uhf(&dt, &da, &db, &mut j, &mut ka, &mut kb)
                .unwrap();
            (j, ka, kb)
        };
        let (jp, kap, kbp) = run(JkDigest::Pair);
        let (jl, kal, kbl) = run(JkDigest::Legacy);
        agree(&format!("OH UHF {tag} J"), &jp, &jl);
        agree(&format!("OH UHF {tag} Ka"), &kap, &kal);
        agree(&format!("OH UHF {tag} Kb"), &kbp, &kbl);
    }
}

/// `DirectJ` (J only) and `DirectK` (K only), symmetric and not.
#[test]
fn pair_matches_legacy_j_only_and_k_only() {
    let prep = setup("water.xyz", "cc-pvdz", 1);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    for (sym, tag) in [(true, "sym"), (false, "nonsym")] {
        let d = density(n, 21, sym);
        let j_of = |digest| {
            let b = bounds(&prep, digest);
            let mut x = zeros(n);
            JBuilder::build(
                &mut DirectJ::new(&ctx, &prep, &b, THRESH, usize::MAX / 4),
                &d,
                &mut x,
            )
            .unwrap();
            x
        };
        let k_of = |digest| {
            let b = bounds(&prep, digest);
            let mut x = zeros(n);
            KBuilder::build(
                &mut DirectK::new(&ctx, &prep, &b, THRESH, usize::MAX / 4),
                &d,
                &mut x,
            )
            .unwrap();
            x
        };
        agree(
            &format!("DirectJ {tag}"),
            &j_of(JkDigest::Pair),
            &j_of(JkDigest::Legacy),
        );
        agree(
            &format!("DirectK {tag}"),
            &k_of(JkDigest::Pair),
            &k_of(JkDigest::Legacy),
        );
    }
}

/// `rhf::build_jk` / `build_jk_with_pool` (the response, Newton and stability
/// builds).
#[test]
fn pair_matches_legacy_pooled_build() {
    let prep = setup("water.xyz", "6-31g*", 1);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    for (sym, tag) in [(true, "sym"), (false, "nonsym")] {
        let d = density(n, 31, sym);
        let run = |digest| {
            let b = bounds(&prep, digest);
            let (mut j, mut k) = (zeros(n), zeros(n));
            ferric_scf::rhf::build_jk(&ctx, &prep, &b, THRESH, &d, &mut j, &mut k).unwrap();
            (j, k)
        };
        let (jp, kp) = run(JkDigest::Pair);
        let (jl, kl) = run(JkDigest::Legacy);
        agree(&format!("build_jk {tag} J"), &jp, &jl);
        agree(&format!("build_jk {tag} K"), &kp, &kl);
    }
}

/// The rotated walk (`gc_rotation = auto` on cc-pVDZ) needs no change of its
/// own: its rotated bounds share the parent's digest record, so the two
/// digests of the rotated build agree to rounding too, and the legacy
/// rotated build is not the pair one.
#[test]
fn pair_matches_legacy_on_the_rotated_walk() {
    let prep = setup("water.xyz", "cc-pvdz", 1);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    let d = density(n, 41, true);
    let run = |digest| {
        let b = bounds(&prep, digest);
        let mut djk =
            DirectJK::new(&ctx, &prep, &b, THRESH, usize::MAX / 4).with_gc_rotation(GcRotate::Auto);
        let (mut j, mut k) = (zeros(n), zeros(n));
        djk.build(&d, &mut j, &mut k).unwrap();
        assert!(djk.gc_rotated_columns().unwrap() > 0, "cc-pVDZ must rotate");
        (j, k)
    };
    let (jp, kp) = run(JkDigest::Pair);
    let (jl, kl) = run(JkDigest::Legacy);
    agree("rotated J", &jp, &jl);
    agree("rotated K", &kp, &kl);
    assert!(
        jp.iter()
            .zip(jl.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "the rotated walk ignored the digest record (pair and legacy are bitwise equal)"
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
    let (jw, kw) = closed(&prep, want, &d);
    assert!(j == jw && k == kw, "unrecorded bounds did not run {want}");
}
