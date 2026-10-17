//! PySCF comparison for the point-group-blocked RHF (issue #294).
//!
//! References come from `scripts/validation/gen_scf_symmetry.py`
//! (`testdata/reference/validation/scf_symmetry/`). The fast anchors live in
//! `scf_symmetry.rs`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::symmetric::{solve_rhf_symmetric, SymmetricRhf};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load(name: &str) -> Molecule {
    Molecule::load_xyz(
        root()
            .join("testdata/molecules")
            .join(name)
            .to_str()
            .unwrap(),
    )
    .unwrap()
}

fn run_sym(mol: &Molecule, basis_name: &str, cfg: &RhfConfig) -> Result<SymmetricRhf, String> {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    solve_rhf_symmetric(&ParallelContext::default(), mol, &prep, op, &bounds, cfg)
        .map_err(|e| e.to_string())
}

fn cfg(symmetry: bool) -> RhfConfig {
    RhfConfig {
        symmetry,
        density_conv: 1e-10,
        energy_conv: 1e-12,
        max_iter: 200,
        ..RhfConfig::default()
    }
}

fn occupation(r: &SymmetricRhf, nocc: usize) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for l in &r.orbital_irreps()[..nocc] {
        *out.entry(l.clone()).or_insert(0) += 1;
    }
    out
}

fn nocc(mol: &Molecule) -> usize {
    (mol.nelec() / 2) as usize
}

fn water_excited_nelec() -> Vec<(String, usize)> {
    vec![("A1".into(), 8), ("B2".into(), 2), ("B1".into(), 0)]
}

fn reference(name: &str) -> serde_json::Value {
    let p = root()
        .join("testdata/reference/validation/scf_symmetry")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
#[ignore = "validation: SCF symmetry"]
fn matches_pyscf_symmetry_references() {
    let mut worst = 0.0f64;
    for (sys, xyz) in [
        ("h2o", "water.xyz"),
        ("nh3", "nh3.xyz"),
        ("c2h4", "c2h4.xyz"),
        ("benzene", "benzene.xyz"),
    ] {
        for bas in ["sto-3g", "6-31g"] {
            let want = reference(&format!("{sys}_{bas}.json"));
            let mol = load(xyz);
            let r = run_sym(&mol, bas, &cfg(true)).unwrap();
            let e = want["ground"]["energy"].as_f64().unwrap();
            worst = worst.max((r.energy() - e).abs());
            assert!(
                (r.energy() - e).abs() < 1e-8,
                "{sys}/{bas}: {} vs {e}",
                r.energy()
            );
            // Per-irrep occupations: labels for C2v/Cs, multiset for D2h.
            let ours = occupation(&r, nocc(&mol));
            let theirs: BTreeMap<String, usize> = want["ground"]["docc_per_irrep"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.replace('"', "''"), v.as_u64().unwrap() as usize))
                .collect();
            if r.group() == "D2h" {
                let mut a: Vec<_> = ours.values().copied().collect();
                let mut b: Vec<_> = theirs.values().copied().filter(|&n| n > 0).collect();
                a.sort();
                b.sort();
                assert_eq!(a, b, "{sys}/{bas}");
            } else {
                let theirs: BTreeMap<_, _> = theirs.into_iter().filter(|(_, n)| *n > 0).collect();
                assert_eq!(ours, theirs, "{sys}/{bas}");
            }
        }
        if sys == "h2o" {
            for bas in ["sto-3g", "6-31g"] {
                let want = reference(&format!("h2o_{bas}.json"));
                let mut c = cfg(true);
                c.irrep_nelec = Some(water_excited_nelec());
                let r = run_sym(&load("water.xyz"), bas, &c).unwrap();
                let e = want["excited"]["energy"].as_f64().unwrap();
                worst = worst.max((r.energy() - e).abs());
                assert!(
                    (r.energy() - e).abs() < 1e-8,
                    "excited {bas}: {} vs {e}",
                    r.energy()
                );
            }
        }
    }
    eprintln!("max |E_ferric - E_pyscf| = {worst:e}");
}
