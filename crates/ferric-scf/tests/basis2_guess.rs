//! Secondary-basis (Q-Chem BASIS2) SCF guess: `RhfConfig::guess_basis`.
//!
//! Order follows the experimental protocol: the exactness anchor (secondary
//! basis == target basis, so the projection is the identity and the guess is
//! the converged answer) comes first, then energy equivalence against the
//! default guess, the projection's own invariants, an independent PySCF
//! reference for the projected matrices, and the error contract.
//!
//! Artifact hypothesis for the anchor: if the projection or the guess
//! injection were wrong (wrong metric, wrong spin, dropped re-orthonormalisation),
//! the same-basis guess would NOT be the converged state and the target SCF
//! would need a normal iteration count rather than <= 2.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::basis2::{self as b2, GuessBasis, GuessProjection};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const WATER: &str = "3\n\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
const NH2: &str = "3\n\nN 0.0 0.0 0.1404\nH 0.0 0.8019 -0.4914\nH 0.0 -0.8019 -0.4914\n";
const O2: &str = "2\n\nO 0.0 0.0 0.0\nO 0.0 0.0 1.208\n";

#[derive(Clone, Copy, Debug)]
enum Kind {
    Rhf,
    Uhf,
    Rohf,
}

fn setup(xyz: &str, mult: usize, basis: &str) -> (Molecule, PreparedBasis, SchwarzBounds) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled(basis).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}

fn tight(xc: Option<&str>) -> RhfConfig {
    RhfConfig {
        density_conv: 1e-8,
        xc: xc.map(str::to_string),
        ..Default::default()
    }
}

fn solve(
    kind: Kind,
    mol: &Molecule,
    prep: &PreparedBasis,
    b: &SchwarzBounds,
    c: &RhfConfig,
) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    match kind {
        Kind::Rhf => solve_rhf(&ctx, mol, prep, op, b, c),
        Kind::Uhf => solve_uhf(&ctx, mol, prep, b, c),
        Kind::Rohf => solve_rohf(&ctx, mol, prep, op, b, c),
    }
    .unwrap()
}

fn with_guess(cfg: &RhfConfig, basis: &str, projection: &str) -> RhfConfig {
    RhfConfig {
        guess_basis: Some(GuessBasis::bundled(basis, projection).unwrap()),
        ..cfg.clone()
    }
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// ── 1. Exactness anchor: secondary basis == target basis ──────────────────

#[test]
fn projector_is_identity_for_the_same_basis() {
    let (mol, prep, _) = setup(WATER, 1, "6-31g");
    let (p, _) = b2::projector(&mol, &prep, &basis::bundled("6-31g").unwrap()).unwrap();
    let err = max_abs_diff(&p, &Array2::eye(prep.nbasis()));
    assert!(
        err < 1e-12,
        "P = S^-1 S for the same basis must be I, max dev {err:e}"
    );
}

fn same_basis_anchor(kind: Kind, xyz: &str, mult: usize, xc: Option<&str>) {
    let (mol, prep, bounds) = setup(xyz, mult, "6-31g");
    let base = tight(xc);
    let reference = solve(kind, &mol, &prep, &bounds, &base);
    for projection in GuessProjection::VALID {
        let r = solve(
            kind,
            &mol,
            &prep,
            &bounds,
            &with_guess(&base, "6-31g", projection),
        );
        let rep = r.guess_basis.as_ref().expect("report must be echoed");
        assert_eq!(rep.nbasis_guess, rep.nbasis_target);
        let de = (r.energy - rep.energy).abs();
        assert!(
            r.iterations <= 2,
            "{kind:?}/{projection}: same-basis guess is the converged state, yet the target \
             SCF took {} iterations",
            r.iterations
        );
        assert!(
            de < 1e-10,
            "{kind:?}/{projection}: E_target - E_secondary = {de:e}"
        );
        assert!(
            (r.energy - reference.energy).abs() < 1e-8,
            "{kind:?}/{projection}: {} vs default-guess {}",
            r.energy,
            reference.energy
        );
    }
}

#[test]
fn same_basis_guess_is_converged_rhf() {
    same_basis_anchor(Kind::Rhf, WATER, 1, None);
}

#[test]
fn same_basis_guess_is_converged_uhf() {
    same_basis_anchor(Kind::Uhf, NH2, 2, None);
}

#[test]
fn same_basis_guess_is_converged_rohf() {
    same_basis_anchor(Kind::Rohf, NH2, 2, None);
}

#[test]
fn same_basis_guess_is_converged_rks() {
    same_basis_anchor(Kind::Rhf, WATER, 1, Some("PBE"));
}

// ── 2. Final energy equals the default-guess energy ───────────────────────

fn energy_matches_default(
    kind: Kind,
    xyz: &str,
    mult: usize,
    small: &str,
    big: &str,
    xc: Option<&str>,
) {
    let (mol, prep, bounds) = setup(xyz, mult, big);
    let base = tight(xc);
    let reference = solve(kind, &mol, &prep, &bounds, &base);
    for projection in GuessProjection::VALID {
        let r = solve(
            kind,
            &mol,
            &prep,
            &bounds,
            &with_guess(&base, small, projection),
        );
        let rep = r.guess_basis.as_ref().unwrap();
        eprintln!(
            "{kind:?} {xc:?} {small}->{big} {projection}: default {} iters, basis2 {} iters \
             (+{} secondary)",
            reference.iterations, r.iterations, rep.iterations
        );
        let de = (r.energy - reference.energy).abs();
        assert!(de < 1e-8, "{kind:?}/{projection}: |dE| = {de:e}");
    }
}

#[test]
fn energy_matches_default_guess_rhf() {
    energy_matches_default(Kind::Rhf, WATER, 1, "6-31g", "cc-pvdz", None);
}

#[test]
fn energy_matches_default_guess_uhf() {
    energy_matches_default(Kind::Uhf, NH2, 2, "6-31g", "cc-pvdz", None);
}

#[test]
fn energy_matches_default_guess_rohf_triplet() {
    energy_matches_default(Kind::Rohf, O2, 3, "sto-3g", "6-31g", None);
}

#[test]
fn energy_matches_default_guess_rks() {
    energy_matches_default(Kind::Rhf, WATER, 1, "6-31g", "cc-pvdz", Some("PBE"));
}

#[test]
fn larger_secondary_basis_is_allowed() {
    energy_matches_default(Kind::Rhf, WATER, 1, "cc-pvdz", "6-31g", None);
}

// ── 3. Overlap projection invariants ──────────────────────────────────────

#[test]
fn overlap_projection_is_orthonormal_and_keeps_the_electron_count() {
    let (mol, prep_s, bounds_s) = setup(WATER, 1, "sto-3g");
    let small = solve(Kind::Rhf, &mol, &prep_s, &bounds_s, &tight(None));
    let nocc = 5;
    let c_small = small.mos_alpha.slice(ndarray::s![.., ..nocc]).to_owned();
    let target = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let (p, s_bb) = b2::projector(&mol, &target, &basis::bundled("sto-3g").unwrap()).unwrap();
    let c = b2::project_occupied(&p, &s_bb, &c_small).unwrap();
    let ortho = max_abs_diff(&c.t().dot(&s_bb).dot(&c), &Array2::eye(nocc));
    assert!(ortho < 1e-12, "C^T S_bb C deviates from I by {ortho:e}");
    let d = c.dot(&c.t()) * 2.0;
    let n_e: f64 = (d.dot(&s_bb)).diag().sum();
    assert!((n_e - 10.0).abs() < 1e-10, "tr(D S) = {n_e}");
}

// ── 4. Independent reference: PySCF project_dm_nr2nr / project_mo_nr2nr ──

fn json_matrix(v: &serde_json::Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone()).unwrap();
    let (n, m) = (rows.len(), rows[0].len());
    Array2::from_shape_vec((n, m), rows.into_iter().flatten().collect()).unwrap()
}

#[test]
fn projection_matches_pyscf() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/water_631g_ccpvdz_basis2_projection.json"
    );
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let small = basis::bundled("6-31g").unwrap();
    let target = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let prep_s = PreparedBasis::new(&mol, &small).unwrap();
    let (p, s_bb) = b2::projector(&mol, &target, &small).unwrap();
    let s_ss = ferric_integrals::oneelectron::overlap(&prep_s);
    let ov_b = max_abs_diff(&s_bb, &json_matrix(&r["s_target"]));
    let ov_s = max_abs_diff(&s_ss, &json_matrix(&r["s_small"]));
    assert!(
        ov_b < 1e-10 && ov_s < 1e-10,
        "AO order/basis data differ: {ov_b:e} {ov_s:e}"
    );
    let dm = b2::project_density(&p, &json_matrix(&r["dm_small"]));
    let mo = p.dot(&json_matrix(&r["c_occ_small"]));
    let e_dm = max_abs_diff(&dm, &json_matrix(&r["dm_projected"]));
    let e_mo = max_abs_diff(&mo, &json_matrix(&r["c_occ_projected"]));
    assert!(e_dm < 1e-10, "projected density vs PySCF: {e_dm:e}");
    assert!(e_mo < 1e-10, "projected MOs vs PySCF: {e_mo:e}");
}

// ── 5. Errors, never silent fallbacks ─────────────────────────────────────

#[test]
fn bad_projection_and_basis_names_are_errors() {
    assert!(GuessBasis::bundled("6-31g", "fop").is_err());
    assert!(GuessBasis::bundled("6-31g", "").is_err());
    assert!(GuessBasis::bundled("not-a-basis", "fock").is_err());
    assert_eq!(
        GuessProjection::parse_config_str("OVERLAP").unwrap(),
        GuessProjection::Overlap
    );
}

#[test]
fn missing_element_in_secondary_basis_is_an_error() {
    let (mol, prep, bounds) = setup(WATER, 1, "6-31g");
    let mut no_h: BasisSet = basis::bundled("sto-3g").unwrap();
    no_h.shells.remove(&1);
    let cfg = RhfConfig {
        guess_basis: Some(GuessBasis {
            basis: no_h,
            projection: GuessProjection::Fock,
        }),
        ..Default::default()
    };
    let e = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &cfg,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("guess_basis") && e.contains("Z=1"), "{e}");
}

#[test]
fn unconverged_secondary_scf_is_an_error_not_a_fallback() {
    let (mol, prep, bounds) = setup(NH2, 2, "6-31g");
    let cfg = RhfConfig {
        max_iter: 2,
        ..with_guess(&RhfConfig::default(), "sto-3g", "fock")
    };
    let e = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg)
        .unwrap_err()
        .to_string();
    assert!(e.contains("secondary SCF"), "{e}");
}

#[test]
fn symmetry_blocked_solve_refuses_guess_basis() {
    let (mol, prep, bounds) = setup(WATER, 1, "6-31g");
    let cfg = RhfConfig {
        symmetry: true,
        ..with_guess(&RhfConfig::default(), "sto-3g", "fock")
    };
    let r = ferric_scf::symmetric::solve_rhf_symmetric(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &cfg,
    );
    let Err(e) = r else {
        panic!("solve_rhf_symmetric must refuse guess_basis");
    };
    let e = e.to_string();
    assert!(e.contains("guess_basis"), "{e}");
}

#[test]
fn injected_path_refuses_guess_basis() {
    let cfg = with_guess(
        &RhfConfig {
            use_sad_guess: false,
            ..Default::default()
        },
        "sto-3g",
        "fock",
    );
    let e = ferric_scf::rhf::validate_injected(&cfg).unwrap_err();
    assert_eq!(e.field, "guess_basis");
}
