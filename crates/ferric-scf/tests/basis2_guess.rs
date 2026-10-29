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
        assert!(rep.converged && rep.used_as_guess && rep.unconverged_reason.is_none());
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

// ── 2. Same state as the default guess, on single-basin systems ───────────
//
// NOT a general guarantee. The guess can select WHICH stationary state the
// target SCF converges to, in either direction: NO UKS-PBE/aug-cc-pVDZ lands
// on a state 0.97 µHa above MINAO's, and He₂ at 0.3 Å RHF/aug-cc-pVTZ on one
// 7 mHa BELOW MINAO's. These systems were chosen because they have one basin
// in reach of both guesses, so any energy difference here is a defect in the
// projection, not state selection.

fn same_state_as_default(
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
fn single_basin_same_state_as_default_rhf() {
    same_state_as_default(Kind::Rhf, WATER, 1, "6-31g", "cc-pvdz", None);
}

#[test]
fn single_basin_same_state_as_default_uhf() {
    same_state_as_default(Kind::Uhf, NH2, 2, "6-31g", "cc-pvdz", None);
}

#[test]
fn single_basin_same_state_as_default_rohf_triplet() {
    same_state_as_default(Kind::Rohf, O2, 3, "sto-3g", "6-31g", None);
}

#[test]
fn single_basin_same_state_as_default_rks() {
    same_state_as_default(Kind::Rhf, WATER, 1, "6-31g", "cc-pvdz", Some("PBE"));
}

#[test]
fn larger_secondary_basis_is_allowed() {
    same_state_as_default(Kind::Rhf, WATER, 1, "cc-pvdz", "6-31g", None);
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

// ── 6. An unconverged secondary SCF is a guess, not a failure ─────────────

const CN: &str = "2\n\nC 0 0 0\nN 0 0 1.172\n";

/// CN UHF/aug-cc-pVDZ from an STO-3G secondary: the STO-3G UHF does not
/// converge in 200 iterations, while the target converges from the default
/// guess in 22. Before the fix the secondary's failure aborted the target
/// solve. Now the target starts from its default guess, converges to the
/// default-guess state, and the report says the secondary did not converge,
/// why, and that it did not seed the target. (Seeding from the secondary's
/// last iterate was measured and rejected: the target then fails too.)
#[test]
fn unconverged_secondary_still_seeds_a_converging_target() {
    let (mol, prep, bounds) = setup(CN, 2, "aug-cc-pvdz");
    let base = tight(None);
    let reference = solve(Kind::Uhf, &mol, &prep, &bounds, &base);
    let r = solve(
        Kind::Uhf,
        &mol,
        &prep,
        &bounds,
        &with_guess(&base, "sto-3g", "fock"),
    );
    let rep = r.guess_basis.as_ref().expect("report must be echoed");
    assert!(
        !rep.converged,
        "the STO-3G secondary SCF was expected to fail"
    );
    let why = rep.unconverged_reason.as_deref().unwrap_or_default();
    assert!(why.contains("iterations"), "reason: {why:?}");
    assert!(!rep.used_as_guess);
    assert!(r.converged);
    let de = (r.energy - reference.energy).abs();
    assert!(
        de < 1e-8,
        "{} vs default {} (|dE| = {de:e})",
        r.energy,
        reference.energy
    );
}

/// The convergence ladder with a secondary SCF that fails on rung 0 (the
/// rung's 2-iteration cap applies to the secondary too) still escalates:
/// rung 0's target fails, rung 1 converges from the carried density, and the
/// final result carries rung 0's secondary-basis report.
#[test]
fn ladder_escalates_past_a_failing_secondary() {
    let (mol, prep, bounds) = setup(WATER, 1, "cc-pvdz");
    let base = with_guess(&tight(None), "6-31g", "fock");
    let ladder = vec![
        ferric_scf::ladder::Rung {
            config: RhfConfig {
                max_iter: 2,
                ..base.clone()
            },
            restart: false,
        },
        ferric_scf::ladder::Rung {
            config: base.clone(),
            restart: false,
        },
    ];
    let lr = ferric_scf::ladder::solve_rhf_ladder(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &ladder,
    )
    .expect("a failing secondary SCF must not abort the ladder");
    assert!(
        lr.converged && lr.rung_reached == 1,
        "{:?}",
        lr.rung_outcomes
    );
    let rep = lr.result.guess_basis.as_ref().expect("rung 0's report");
    assert!(
        !rep.converged && !rep.used_as_guess && rep.iterations == 2,
        "{rep:?}"
    );
    let reference = solve(Kind::Rhf, &mol, &prep, &bounds, &tight(None));
    assert!((lr.result.energy - reference.energy).abs() < 1e-8);
}

// ── 7. The report survives every path that replaces the first result ──────

/// Stretched H₂ (2.5 Å) UHF singlet with the stability descent: the first
/// solve is a saddle and the descended state is re-solved from rotated MOs
/// (which outrank `guess_basis`), so the returned result is not the one the
/// secondary SCF seeded. Its report must still be there.
#[test]
fn report_survives_the_stability_descent() {
    let (mol, prep, bounds) = setup("2\n\nH 0 0 0\nH 0 0 2.5\n", 1, "cc-pvdz");
    let cfg = RhfConfig {
        check_stability: true,
        scf_stability_descent: true,
        ..with_guess(&tight(None), "6-31g", "fock")
    };
    let plain = RhfConfig {
        scf_stability_descent: false,
        ..cfg.clone()
    };
    let first = solve(Kind::Uhf, &mol, &prep, &bounds, &plain);
    let r = solve(Kind::Uhf, &mol, &prep, &bounds, &cfg);
    assert!(
        r.energy < first.energy - 1e-6,
        "the descent must have replaced the first solve for this test to mean anything \
         ({} vs {})",
        r.energy,
        first.energy
    );
    let rep = r.guess_basis.as_ref().expect("report lost by the descent");
    assert_eq!(rep.basis.to_ascii_lowercase(), "6-31g");
}

/// The DF-guess two-stage SCF: the secondary runs inside the DF pre-stage,
/// the exact stage starts from the DF density. The report must reach the
/// exact stage's result.
#[test]
fn report_survives_the_df_guess_exact_stage() {
    let (mol, prep, bounds) = setup(WATER, 1, "cc-pvdz");
    let cfg = with_guess(&tight(None), "6-31g", "fock");
    let before = b2::secondary_runs_on_this_thread();
    let dfg = ferric_scf::ladder::solve_rhf_with_df_guess(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &cfg,
        None,
    )
    .unwrap();
    assert_eq!(b2::secondary_runs_on_this_thread() - before, 1);
    assert!(dfg.result.converged);
    assert!(
        dfg.result.guess_basis.is_some(),
        "report lost by the exact stage"
    );
}

// ── 8. cDFT builds the guess once, not once per λ step ────────────────────

#[test]
fn cdft_runs_the_secondary_scf_once() {
    use ferric_dft::cdft::{Constraint, SpinChannel};
    let (mol, prep, bounds) = setup("2\n\nLi 0 0 0\nH 0 0 1.60\n", 1, "def2-svp");
    let bs = basis::bundled("def2-svp").unwrap();
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            spin: SpinChannel::Total,
            target: 2.2,
        }],
        cdft_lambda_tol: 1e-5,
        cdft_max_outer: 40,
        cdft_stability_descent: false,
        fractional_occ: false,
        ..with_guess(&RhfConfig::default(), "sto-3g", "fock")
    };
    let before = b2::secondary_runs_on_this_thread();
    let res = ferric_scf::cdft_driver::solve_cdft_uhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        &bs,
        &bounds,
        &cfg,
    )
    .unwrap();
    let runs = b2::secondary_runs_on_this_thread() - before;
    assert!(res.outer_iters > 1, "need several λ steps to mean anything");
    assert_eq!(
        runs, 1,
        "secondary SCF ran {runs} times over {} λ steps",
        res.outer_iters
    );
    assert!((res.populations[0] - 2.2).abs() < 1e-5);
    assert!(res.scf.guess_basis.is_some());
}

// ── 9. Guards, each with a test that fails without it ─────────────────────

/// An explicit starting density outranks `guess_basis`: the secondary SCF
/// does not run, and no report is attached.
#[test]
fn explicit_density_outranks_guess_basis() {
    let (mol, prep, bounds) = setup(WATER, 1, "6-31g");
    let reference = solve(Kind::Rhf, &mol, &prep, &bounds, &tight(None));
    let cfg = RhfConfig {
        init_guess_density: Some(reference.density_total.clone()),
        ..with_guess(&tight(None), "sto-3g", "fock")
    };
    let before = b2::secondary_runs_on_this_thread();
    let r = solve(Kind::Rhf, &mol, &prep, &bounds, &cfg);
    assert_eq!(b2::secondary_runs_on_this_thread(), before);
    assert!(r.guess_basis.is_none());
}

/// Caller-supplied UHF MOs outrank `guess_basis` the same way.
#[test]
fn explicit_mos_outrank_guess_basis() {
    let (mol, prep, bounds) = setup(NH2, 2, "6-31g");
    let reference = solve(Kind::Uhf, &mol, &prep, &bounds, &tight(None));
    let cb = reference.mos_beta.clone().unwrap();
    let cfg = with_guess(&tight(None), "sto-3g", "fock");
    let before = b2::secondary_runs_on_this_thread();
    let r = ferric_scf::uhf::solve_uhf_with_guess(
        &ParallelContext::default(),
        &mol,
        &prep,
        &bounds,
        &cfg,
        Some((&reference.mos_alpha, &cb)),
    )
    .unwrap();
    assert_eq!(b2::secondary_runs_on_this_thread(), before);
    assert!(r.guess_basis.is_none());
}

const HI: &str = "2\n\nH 0 0 0\nI 0 0 1.609\n";

fn hi_error(guess: BasisSet) -> String {
    let mut mol = Molecule::parse_xyz(HI, 0, 1).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    mol.apply_ecp(&bs);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = RhfConfig {
        guess_basis: Some(GuessBasis {
            basis: guess,
            projection: GuessProjection::Fock,
        }),
        ..Default::default()
    };
    solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &cfg,
    )
    .unwrap_err()
    .to_string()
}

/// def2-SVP puts a 28-electron ECP on iodine. A guess basis with iodine
/// functions but no ECP (def2-SVP with its ECPs stripped) would run a
/// secondary SCF with 28 more electrons: an ECP mismatch.
#[test]
fn ecp_core_mismatch_is_an_error() {
    let mut no_ecp = basis::bundled("def2-svp").unwrap();
    no_ecp.ecps.clear();
    let e = hi_error(no_ecp);
    assert!(e.contains("ECP core"), "{e}");
}

/// 6-31G has no iodine at all: that is a missing element, and must say so
/// rather than report an ECP mismatch.
#[test]
fn guess_basis_without_the_element_reports_the_missing_element() {
    let e = hi_error(basis::bundled("6-31g").unwrap());
    assert!(e.contains("I (Z=53)") && !e.contains("ECP"), "{e}");
}

/// The near-singular guard: two parallel columns make `Cᵀ S C` singular, and
/// Löwdin orthonormalisation would divide by ~0.
#[test]
fn lowdin_refuses_a_near_singular_occupied_block() {
    let s = Array2::<f64>::eye(3);
    let c = ndarray::array![[1.0, 1.0], [0.0, 1e-9], [0.0, 0.0]];
    let e = b2::lowdin_orthonormalize(&c, &s).unwrap_err().to_string();
    assert!(e.contains("near-singular"), "{e}");
    let ok = ndarray::array![[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
    assert!(b2::lowdin_orthonormalize(&ok, &s).is_ok());
}
