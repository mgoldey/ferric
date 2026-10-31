//! Real SCF results through the checkpoint file, and the measurement behind the
//! provisional orthonormality gate of the file guess.
//!
//! The format's own tests use synthetic matrices. This one holds what only a
//! solver can produce: a real `df_jk` route, ROHF's beta conventions, and the
//! size of `CᵀSC - I` at convergence. No solver reads the file here.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{hcore as hcore_matrix, overlap};
use ferric_integrals::operator::Operator;
use ferric_scf::checkpoint::{BuildStamp, Checkpoint, ReadMode, ScfIdentity};
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const WATER: &str = "3\n\nO 0 0 0.11779\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
const NH2: &str = "3\n\nN 0 0 0.1404\nH 0 0.8019 -0.4914\nH 0 -0.8019 -0.4914\n";

/// The provisional gate from the design: a file with same-geometry
/// `‖CᵀSC−I‖max` above this is refused as a guess.
const PROVISIONAL_GATE: f64 = 1e-8;

fn max_deviation(c: &Array2<f64>, s: &Array2<f64>) -> f64 {
    let metric = c.t().dot(s).dot(c);
    metric
        .indexed_iter()
        .map(|((i, j), v)| (v - f64::from(u8::from(i == j))).abs())
        .fold(0.0, f64::max)
}

fn build() -> BuildStamp {
    BuildStamp {
        version: "test".to_string(),
        commit: Some("abc".to_string()),
        dirty: Some(false),
    }
}

fn bits(a: &Array2<f64>) -> Vec<u64> {
    a.as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

struct Case {
    name: &'static str,
    xyz: &'static str,
    mult: usize,
    basis: &'static str,
    spin: Spin,
    config: RhfConfig,
}

fn solve(case: &Case, mol: &Molecule, prep: &PreparedBasis) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).unwrap();
    match case.spin {
        Spin::Restricted => solve_rhf(&ctx, mol, prep, op, &bounds, &case.config).unwrap(),
        Spin::Unrestricted => solve_uhf(&ctx, mol, prep, &bounds, &case.config).unwrap(),
        Spin::RestrictedOpen => solve_rohf(&ctx, mol, prep, op, &bounds, &case.config).unwrap(),
    }
}

fn tight() -> RhfConfig {
    RhfConfig {
        density_conv: 1e-8,
        ..Default::default()
    }
}

fn cases() -> Vec<Case> {
    let ri_pbe = RhfConfig {
        xc: Some("PBE".to_string()),
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        ..tight()
    };
    let case = |name, xyz, mult, basis, spin, config| Case {
        name,
        xyz,
        mult,
        basis,
        spin,
        config,
    };
    vec![
        case(
            "RHF cc-pvdz",
            WATER,
            1,
            "cc-pvdz",
            Spin::Restricted,
            tight(),
        ),
        case(
            "RHF aug-cc-pvtz",
            WATER,
            1,
            "aug-cc-pvtz",
            Spin::Restricted,
            tight(),
        ),
        case(
            "RI-J PBE def2-svp",
            WATER,
            1,
            "def2-svp",
            Spin::Restricted,
            ri_pbe,
        ),
        case(
            "UHF cc-pvdz",
            NH2,
            2,
            "cc-pvdz",
            Spin::Unrestricted,
            tight(),
        ),
        case(
            "ROHF cc-pvdz",
            NH2,
            2,
            "cc-pvdz",
            Spin::RestrictedOpen,
            tight(),
        ),
    ]
}

fn check_case(case: &Case, dir: &std::path::Path) {
    let name = case.name;
    let mol = Molecule::parse_xyz(case.xyz, 0, case.mult).unwrap();
    let basis = bundled(case.basis).unwrap();
    let prep = PreparedBasis::new(&mol, &basis).unwrap();
    let result = solve(case, &mol, &prep);
    let op = Operator::coulomb();
    let identity = ScfIdentity::new(&mol, &basis, case.spin, &op, &case.config).unwrap();
    let e1 = (&result.density_total * &hcore_matrix(&prep)).sum();
    let vnn = mol.nuclear_repulsion();
    let ck = Checkpoint::from_result(&result, e1, vnn, identity.clone(), build()).unwrap();
    let path = dir.join("real.npz");
    ck.write(&path).unwrap();
    let back = Checkpoint::read(&path).unwrap();

    // The real matrices and route survive, as does the real layout.
    let (o, state) = (&back.orbitals, back.final_state.as_ref().unwrap());
    assert_eq!(
        bits(&o.density_total),
        bits(&result.density_total),
        "{name}"
    );
    assert_eq!(bits(&o.mos_alpha), bits(&result.mos_alpha), "{name}");
    assert_eq!(
        o.mos_alpha.is_standard_layout(),
        result.mos_alpha.is_standard_layout(),
        "{name}"
    );
    assert_eq!(bits(&state.fock_alpha), bits(&result.fock_alpha), "{name}");
    assert_eq!(state.energy.to_bits(), result.energy.to_bits(), "{name}");
    assert_eq!(state.df_jk, result.df_jk, "{name}");
    assert_eq!(state.exit, result.exit, "{name}");

    // The same calculation is accepted to skip, on the same clean build.
    let skip = ReadMode::Skip {
        allow_other_build: false,
    };
    let verdict = back.check(&identity, &build(), skip).unwrap();
    assert!(verdict.warnings.is_empty() && !verdict.build_override_used);

    // Energy pieces are what the independent S/h check will compare later.
    assert!(state.one_electron_energy < 0.0, "{name}");

    let s = overlap(&prep);
    let alpha = max_deviation(&o.mos_alpha, &s);
    let beta = o.mos_beta.as_ref().map(|c| max_deviation(c, &s));
    println!(
        "ORTHO {name}: nao={} nmo={} iters={} exit={:?} alpha={alpha:.2e} beta={beta:?}",
        identity.nao,
        o.mos_alpha.ncols(),
        result.iterations,
        result.exit,
    );
    assert!(alpha < PROVISIONAL_GATE, "{name}: {alpha:e}");
    assert!(beta.unwrap_or(0.0) < PROVISIONAL_GATE, "{name}: {beta:?}");
}

#[test]
fn real_scf_results_round_trip_and_clear_the_provisional_orthonormality_gate() {
    let dir = std::env::temp_dir().join(format!("ferric-ckpt-real-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for case in cases() {
        check_case(&case, &dir);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two starting points, two converged states, on the same molecule and settings.
/// If the guess were path-only, skip could restore either for either, and every
/// rerun would repeat the wrong number exactly. So the identity must see it.
#[test]
fn a_different_guess_can_converge_to_a_different_state_and_identity_sees_it() {
    let n2_plus = "2\n\nN 0 0 0\nN 0 0 1.116\n";
    let mol = Molecule::parse_xyz(n2_plus, 1, 2).unwrap();
    let basis = bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &basis).unwrap();
    let case = |config: RhfConfig| Case {
        name: "N2+",
        xyz: n2_plus,
        mult: 2,
        basis: "6-31g",
        spin: Spin::Unrestricted,
        config,
    };
    let default = tight();
    let hcore = RhfConfig {
        use_sad_guess: false,
        ..tight()
    };
    let (from_default, from_hcore) = (
        solve(&case(default.clone()), &mol, &prep),
        solve(&case(hcore.clone()), &mol, &prep),
    );
    println!(
        "STATES N2+/6-31G: default {:.6} {:?}, hcore {:.6} {:?}",
        from_default.energy, from_default.exit, from_hcore.energy, from_hcore.exit
    );
    assert!(from_default.converged && from_hcore.converged);
    assert!((from_default.energy - from_hcore.energy).abs() > 1e-3);

    let op = Operator::coulomb();
    let spin = Spin::Unrestricted;
    let file_identity = ScfIdentity::new(&mol, &basis, spin, &op, &default).unwrap();
    let current = ScfIdentity::new(&mol, &basis, spin, &op, &hcore).unwrap();
    let e1 = (&from_default.density_total * &hcore_matrix(&prep)).sum();
    let vnn = mol.nuclear_repulsion();
    let ck = Checkpoint::from_result(&from_default, e1, vnn, file_identity, build()).unwrap();
    let skip = ReadMode::Skip {
        allow_other_build: false,
    };
    let err = ck.check(&current, &build(), skip).unwrap_err();
    assert!(err.to_string().contains("use_sad_guess"), "{err}");
    let guess = ck.check(&current, &build(), ReadMode::Guess).unwrap();
    assert_eq!(guess.warnings[0].name, "use_sad_guess");
}

#[test]
fn a_real_ri_scf_carries_a_route_and_an_exact_one_does_not() {
    let all = cases();
    let run = |case: &Case| {
        let mol = Molecule::parse_xyz(case.xyz, 0, case.mult).unwrap();
        let prep = PreparedBasis::new(&mol, &bundled(case.basis).unwrap()).unwrap();
        solve(case, &mol, &prep)
    };
    let exact = all.iter().find(|c| c.name == "RHF cc-pvdz").unwrap();
    let ri = all.iter().find(|c| c.name == "RI-J PBE def2-svp").unwrap();
    assert_eq!(run(exact).df_jk, None);
    let route = run(ri).df_jk.expect("RI-J must record its route");
    assert_eq!(route.j_aux.as_deref(), Some("def2-universal-jkfit"));
}
