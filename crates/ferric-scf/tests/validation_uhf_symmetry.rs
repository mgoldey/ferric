//! PySCF comparison for the point-group-blocked UHF and ROHF (issue #294,
//! stage 3a). References come from `scripts/validation/gen_uhf_symmetry.py`
//! (`testdata/reference/validation/uhf_symmetry/`); the fast anchors live in
//! `scf_open_symmetry.rs`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::symmetric_open::{solve_rohf_symmetric, solve_uhf_symmetric, IrrepSpinCount};
use ferric_scf::uhf::solve_uhf;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// system, xyz (under testdata/molecules), charge, multiplicity, bases.
const SYSTEMS: [(&str, &str, i32, usize, &[&str]); 6] = [
    ("nh2", "validation/nh2.xyz", 0, 2, &["sto-3g", "6-31g"]),
    ("ch3", "validation/ch3.xyz", 0, 2, &["sto-3g", "6-31g"]),
    ("o2", "o2.xyz", 0, 3, &["sto-3g", "6-31g"]),
    (
        "ch2_triplet",
        "validation/ch2_triplet.xyz",
        0,
        3,
        &["sto-3g", "6-31g"],
    ),
    ("no2", "validation/no2.xyz", 0, 2, &["6-31g"]),
    ("h2o_cation", "water.xyz", 1, 2, &["sto-3g", "6-31g"]),
];

fn load(rel: &str, charge: i32, mult: usize) -> Molecule {
    Molecule::load_xyz_with_charge(
        root()
            .join("testdata/molecules")
            .join(rel)
            .to_str()
            .unwrap(),
        charge,
        mult,
    )
    .unwrap()
}

fn reference(name: &str) -> serde_json::Value {
    let p = root()
        .join("testdata/reference/validation/uhf_symmetry")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn cfg(rohf: bool) -> RhfConfig {
    RhfConfig {
        symmetry: true,
        density_conv: 1e-10,
        energy_conv: 1e-12,
        max_iter: 300,
        rohf_occupation_guard: !rohf,
        ..RhfConfig::default()
    }
}

fn run(
    mol: &Molecule,
    basis_name: &str,
    rohf: bool,
    req: Option<&[IrrepSpinCount]>,
) -> (f64, String, Vec<IrrepSpinCount>) {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    if rohf {
        let r = solve_rohf_symmetric(&ctx, mol, &prep, op, &bounds, &cfg(true), req).unwrap();
        (r.energy(), r.group().to_string(), r.irrep_occupations())
    } else {
        let r = solve_uhf_symmetric(&ctx, mol, &prep, &bounds, &cfg(false), req).unwrap();
        (r.energy(), r.group().to_string(), r.irrep_occupations())
    }
}

fn pairs(v: &serde_json::Value) -> Vec<IrrepSpinCount> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, c)| {
            let a = c[0].as_u64().unwrap() as usize;
            let b = c[1].as_u64().unwrap() as usize;
            (k.clone(), (a, b))
        })
        .filter(|(_, (a, b))| a + b > 0)
        .collect()
}

/// Labels for C2v/Cs, multiset of (n_alpha, n_beta) for D2h (PySCF reports
/// Dooh and the detectors pick different axes) and for CH3 (D3h has three
/// equivalent C2v subgroups; which one is picked decides B1 versus B2).
fn assert_same_occupations(
    ours: &[IrrepSpinCount],
    theirs: &[IrrepSpinCount],
    group: &str,
    what: &str,
) {
    if group == "D2h" {
        let mut a: Vec<_> = ours.iter().map(|x| x.1).collect();
        let mut b: Vec<_> = theirs.iter().map(|x| x.1).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b, "{what}");
    } else {
        let mut a = ours.to_vec();
        let mut b = theirs.to_vec();
        a.sort();
        b.sort();
        assert_eq!(a, b, "{what}");
    }
}

fn compare(rohf: bool) {
    let kind = if rohf { "rohf" } else { "uhf" };
    let mut worst = 0.0f64;
    for (sys, xyz, charge, mult, bases) in SYSTEMS {
        for bas in bases {
            let want = reference(&format!("{sys}_{bas}.json"));
            let mol = load(xyz, charge, mult);
            let (e, group, occ) = run(&mol, bas, rohf, None);
            let w = &want[kind]["blocked"];
            let e_ref = w["energy"].as_f64().unwrap();
            worst = worst.max((e - e_ref).abs());
            assert!(
                (e - e_ref).abs() < 1e-8,
                "{kind} {sys}/{bas}: {e} vs {e_ref}"
            );
            let group = if sys == "ch3" { "D2h" } else { group.as_str() };
            assert_same_occupations(
                &occ,
                &pairs(&w["occ_per_irrep"]),
                group,
                &format!("{kind} {sys}/{bas}"),
            );
            if sys == "h2o_cation" {
                let x = &want[kind]["excited"];
                let req = pairs(&x["irrep_nelec"]);
                let (e, group, occ) = run(&mol, bas, rohf, Some(&req));
                let e_ref = x["energy"].as_f64().unwrap();
                worst = worst.max((e - e_ref).abs());
                assert!(
                    (e - e_ref).abs() < 1e-8,
                    "{kind} excited {bas}: {e} vs {e_ref}"
                );
                assert_same_occupations(&occ, &pairs(&x["occ_per_irrep"]), &group, "excited");
            }
        }
    }
    eprintln!("{kind}: max |E_ferric - E_pyscf| = {worst:e}");
}

#[test]
#[ignore = "validation: UHF symmetry"]
fn uhf_matches_pyscf_symmetry_references() {
    compare(false);
}

#[test]
#[ignore = "validation: UHF symmetry"]
fn rohf_matches_pyscf_symmetry_references() {
    compare(true);
}

#[test]
#[ignore = "validation: UHF symmetry"]
fn stretched_h2_negative_control_matches_pyscf() {
    let want = reference("h2_r2p0_6-31g.json");
    let mol = Molecule::load_xyz_with_charge(
        root()
            .join("testdata/molecules/validation/h2_r2p0.xyz")
            .to_str()
            .unwrap(),
        0,
        1,
    )
    .unwrap();
    let (e, _, occ) = run(&mol, "6-31g", false, None);
    let e_sym = want["blocked"]["energy"].as_f64().unwrap();
    assert!((e - e_sym).abs() < 1e-8, "{e} vs {e_sym}");
    assert_eq!(occ, vec![("Ag".to_string(), (1, 1))]);
    // The unblocked solution (stability descent) is the lower symmetry-broken one.
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let c = RhfConfig {
        scf_stability_descent: true,
        check_stability: true,
        density_conv: 1e-10,
        energy_conv: 1e-12,
        max_iter: 300,
        ..RhfConfig::default()
    };
    let broken = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &c).unwrap();
    let e_off = want["unblocked"]["energy"].as_f64().unwrap();
    assert!(
        (broken.energy - e_off).abs() < 1e-8,
        "{} vs {e_off}",
        broken.energy
    );
    assert!(broken.energy < e - 0.05);
}
