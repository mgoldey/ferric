//! Construction anchors for `ferric_core::symmetry` (issue #294, stage 1):
//! every detected operation, represented on ferric's own AO basis, commutes
//! with the real S, T and V built by libint2, and preserves the S metric.
//! A wrong parity sign (any l), a wrong atom permutation or a wrong AO order
//! fails these — the pure-Rust unit tests in ferric-core cannot see them
//! because they never touch an integral.
//!
//! The PySCF test compares the detected group and the per-irrep AO counts
//! with `scripts/gen_pyscf_symmetry_refs.py` output.

use ferric_core::basis::{bundled, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::symmetry::{detect, AoRepresentation, SymmetryDetection, DEFAULT_SYMMETRY_TOL};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{kinetic, nuclear, overlap};
use ndarray::Array2;

fn molecule(angstrom_xyz: &str) -> Molecule {
    let n = angstrom_xyz.lines().count();
    Molecule::parse_xyz(&format!("{n}\n\n{angstrom_xyz}\n"), 0, 1).unwrap()
}

fn all_cartesian(bs: &BasisSet) -> BasisSet {
    let mut out = bs.clone();
    for shells in out.shells.values_mut() {
        shells.iter_mut().for_each(|s| s.pure = false);
    }
    out
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// Largest `|R X R^T - X|` over the operations and the three matrices, and
/// largest `|R^T S R - S|`.
fn commutation_errors(det: &SymmetryDetection, bs: &BasisSet) -> (f64, f64) {
    let prep = PreparedBasis::new(&det.molecule, bs).unwrap();
    let (s, t, v) = (overlap(&prep), kinetic(&prep), nuclear(&prep));
    let rep = AoRepresentation::new(det, &det.molecule, bs).unwrap();
    let (mut commute, mut metric) = (0.0f64, 0.0f64);
    for k in 0..det.ops.len() {
        let r = rep.dense(k);
        for x in [&s, &t, &v] {
            commute = commute.max(max_abs(&(r.dot(x).dot(&r.t()) - x)));
        }
        metric = metric.max(max_abs(&(r.t().dot(&s).dot(&r) - &s)));
    }
    (commute, metric)
}

const CASES: [(&str, &str); 7] = [
    ("water", "O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692"),
    ("nh3", "N 0 0 0.1\nH 0.94 0 -0.3\nH -0.47 0.814 -0.3\nH -0.47 -0.814 -0.3"),
    (
        "ethylene",
        "C 0 0 0.667\nC 0 0 -0.667\nH 0 0.92 1.24\nH 0 -0.92 1.24\nH 0 0.92 -1.24\nH 0 -0.92 -1.24",
    ),
    (
        "methane",
        "C 0 0 0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629",
    ),
    ("hf", "H 0 0 0\nF 0 0 0.917"),
    ("asym", "C 0 0 0\nO 1.2 0.1 0\nH -0.5 0.9 0.3\nF 0.2 -1.1 0.8"),
    ("h2", "H 0 0 0\nH 0 0 0.74"),
];

#[test]
fn operations_commute_with_s_t_v_and_preserve_the_overlap_metric() {
    // sto-3g: s,p; cc-pvdz: pure d; def2-svp: pure d + p; aug-cc-pvtz: pure f;
    // def2-qzvp (water only below): pure g.
    let bases = ["sto-3g", "6-31g", "cc-pvdz", "def2-svp", "aug-cc-pvtz"];
    let mut worst = (0.0f64, 0.0f64);
    for (name, geom) in CASES {
        let det = detect(&molecule(geom), DEFAULT_SYMMETRY_TOL).unwrap();
        for b in bases {
            let (c, m) = commutation_errors(&det, &bundled(b).unwrap());
            assert!(
                c < 1e-12 && m < 1e-12,
                "{name}/{b}: commute {c:.2e} metric {m:.2e}"
            );
            worst = (worst.0.max(c), worst.1.max(m));
        }
        let cart = all_cartesian(&bundled("cc-pvdz").unwrap());
        let (c, m) = commutation_errors(&det, &cart);
        assert!(
            c < 1e-12 && m < 1e-12,
            "{name}/cartesian cc-pvdz: commute {c:.2e} metric {m:.2e}"
        );
    }
    eprintln!(
        "worst commutation {:.2e}, worst metric {:.2e}",
        worst.0, worst.1
    );
}

#[test]
fn benzene_and_g_functions_commute() {
    let mut rows = Vec::new();
    for (el, r) in [("C", 1.397f64), ("H", 2.48)] {
        for i in 0..6 {
            let t = i as f64 * std::f64::consts::FRAC_PI_3;
            rows.push(format!("{el} {} {} 0", r * t.cos(), r * t.sin()));
        }
    }
    let det = detect(&molecule(&rows.join("\n")), DEFAULT_SYMMETRY_TOL).unwrap();
    assert_eq!(det.ops.len(), 8);
    let (c, m) = commutation_errors(&det, &bundled("cc-pvdz").unwrap());
    assert!(
        c < 1e-11 && m < 1e-11,
        "benzene commute {c:.2e} metric {m:.2e}"
    );
    let water = detect(&molecule(CASES[0].1), DEFAULT_SYMMETRY_TOL).unwrap();
    let (c, m) = commutation_errors(&water, &bundled("def2-qzvp").unwrap());
    assert!(
        c < 1e-12 && m < 1e-12,
        "water/def2-qzvp (g): commute {c:.2e} metric {m:.2e}"
    );
}

#[test]
fn a_wrongly_oriented_molecule_is_not_symmetric_in_its_input_frame() {
    // Anti-vacuity: the same operations built for the REORIENTED water are
    // NOT symmetries of the rigidly moved input geometry. Commutation is a
    // property of the returned frame, not something every matrix satisfies.
    let det = detect(&molecule(CASES[0].1), DEFAULT_SYMMETRY_TOL).unwrap();
    let mut moved = det.clone();
    for a in moved.molecule.atoms.iter_mut() {
        let (x, y) = (a.x, a.y);
        a.x = 0.8 * x - 0.6 * y;
        a.y = 0.6 * x + 0.8 * y;
    }
    let bs = bundled("cc-pvdz").unwrap();
    let (c, _) = commutation_errors(&moved, &bs);
    assert!(
        c > 1e-3,
        "rotated frame must violate commutation, got {c:.2e}"
    );
}

#[test]
fn point_group_and_irrep_dimensions_match_pyscf() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/symmetry_pointgroup.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(cases.len() >= 44);
    let mut labelled_checked = 0;
    let groups: std::collections::BTreeSet<_> = cases
        .iter()
        .map(|c| c["abelian_group"].as_str().unwrap())
        .collect();
    for g in ["C1", "Ci", "Cs", "C2", "C2v", "C2h", "D2", "D2h"] {
        assert!(groups.contains(g), "reference set lacks {g}");
    }
    for c in cases {
        let (name, basis) = (
            c["molecule"].as_str().unwrap(),
            c["basis"].as_str().unwrap(),
        );
        let cart = c["cartesian"].as_bool().unwrap();
        let tag = format!("{name}/{basis}/{}", if cart { "cart" } else { "pure" });
        let det = detect(
            &molecule(c["xyz_angstrom"].as_str().unwrap()),
            DEFAULT_SYMMETRY_TOL,
        )
        .unwrap();
        assert_eq!(
            det.group.name(),
            c["abelian_group"].as_str().unwrap(),
            "{tag}: group"
        );
        let bs = if cart {
            all_cartesian(&bundled(basis).unwrap())
        } else {
            bundled(basis).unwrap()
        };
        let rep = AoRepresentation::new(&det, &det.molecule, &bs).unwrap();
        assert_eq!(rep.nao() as u64, c["nao"].as_u64().unwrap(), "{tag}: nao");
        let mut dims: Vec<u64> = rep
            .irrep_dimensions()
            .unwrap()
            .iter()
            .map(|d| d.1 as u64)
            .collect();
        dims.retain(|&d| d > 0);
        dims.sort_unstable();
        let mut want: Vec<u64> = c["irrep_dimensions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect();
        want.retain(|&d| d > 0);
        want.sort_unstable();
        assert_eq!(dims, want, "{tag}: irrep dimensions");
        // Labelled counts. D2 / D2h axis labels (B1g, B2u, ...) depend on how
        // the axes are assigned, so only their dimension multiset is checked.
        if !matches!(det.group.name(), "D2" | "D2h") {
            let mut got: Vec<(String, u64)> = rep
                .irrep_dimensions()
                .unwrap()
                .iter()
                .filter(|d| d.1 > 0)
                .map(|(ir, d)| (ir.label.to_string(), *d as u64))
                .collect();
            let mut want: Vec<(String, u64)> = c["irreps"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(_, v)| v.as_u64().unwrap() > 0)
                .map(|(k, v)| (k.clone(), v.as_u64().unwrap()))
                .collect();
            got.sort();
            want.sort();
            assert_eq!(got, want, "{tag}: labelled irrep counts");
            labelled_checked += 1;
        }
    }
    assert!(
        labelled_checked >= 28,
        "labelled comparison must cover the low-symmetry cases"
    );
}
