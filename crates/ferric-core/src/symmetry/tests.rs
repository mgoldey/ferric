use super::*;
use crate::basis::bundled;
use crate::mol::{Atom, Molecule};

fn atom(z: i32, p: [f64; 3]) -> Atom {
    Atom {
        symbol: crate::elements::z_to_symbol(z).unwrap().to_string(),
        z,
        x: p[0],
        y: p[1],
        zpos: p[2],
        ghost: false,
        n_core_ecp: 0,
    }
}

fn mol_of(atoms: Vec<Atom>) -> Molecule {
    Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    }
}

fn xyz(text: &str, mult: usize) -> Molecule {
    Molecule::parse_xyz(text, 0, mult).unwrap()
}

/// Rigid motion applied to every test geometry so that detection has to
/// find the symmetry frame rather than read it off the axes.
fn rigid(p: [f64; 3]) -> [f64; 3] {
    let (ax, ang) = ([0.3f64, -0.5, 0.8], 0.9f64);
    let n = (ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]).sqrt();
    let k = [ax[0] / n, ax[1] / n, ax[2] / n];
    let (c, s) = (ang.cos(), ang.sin());
    let kxp = [
        k[1] * p[2] - k[2] * p[1],
        k[2] * p[0] - k[0] * p[2],
        k[0] * p[1] - k[1] * p[0],
    ];
    let kdp = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
    let mut o = [0.0; 3];
    for i in 0..3 {
        o[i] = p[i] * c + kxp[i] * s + k[i] * kdp * (1.0 - c) + [0.7, -1.3, 2.1][i];
    }
    o
}

/// Four generic points of distinct species, each expanded by every operation
/// in `ops`: a molecule whose point group is exactly that group.
fn generic_molecule(ops: &[usize]) -> Molecule {
    let seeds = [
        (1, [0.71, 0.43, 0.97]),
        (6, [1.13, -0.82, 0.37]),
        (7, [-0.58, 1.27, 0.66]),
        (8, [0.31, 0.52, -1.41]),
    ];
    let mut atoms = Vec::new();
    for (z, p) in seeds {
        let mut images: Vec<[f64; 3]> = Vec::new();
        for &g in ops {
            let s = OP_SIGNS[g];
            let q = [p[0] * s[0] as f64, p[1] * s[1] as f64, p[2] * s[2] as f64];
            if images
                .iter()
                .all(|r| (0..3).any(|a| (r[a] - q[a]).abs() > 1e-9))
            {
                images.push(q);
            }
        }
        atoms.extend(images.into_iter().map(|q| atom(z, rigid(q))));
    }
    mol_of(atoms)
}

const GROUP_OPS: [(AbelianGroup, &[usize]); 8] = [
    (AbelianGroup::C1, &[0]),
    (AbelianGroup::Ci, &[0, 4]),
    (AbelianGroup::Cs, &[0, 5]),
    (AbelianGroup::C2, &[0, 1]),
    (AbelianGroup::C2v, &[0, 1, 6, 7]),
    (AbelianGroup::C2h, &[0, 1, 4, 5]),
    (AbelianGroup::D2, &[0, 1, 2, 3]),
    (AbelianGroup::D2h, &[0, 1, 2, 3, 4, 5, 6, 7]),
];

#[test]
fn every_abelian_group_is_detected_in_its_canonical_frame() {
    for (group, ops) in GROUP_OPS {
        let det = detect(&generic_molecule(ops), DEFAULT_SYMMETRY_TOL).unwrap();
        assert_eq!(det.group, group, "{}", group.name());
        assert_eq!(det.ops, ops, "{}: canonical frame", group.name());
        assert!(det.max_symmetrization_shift < 1e-10, "{}", group.name());
    }
}

#[test]
fn detected_atom_permutations_are_consistent_with_the_frame_and_the_group_law() {
    for (group, ops) in GROUP_OPS {
        let det = detect(&generic_molecule(ops), DEFAULT_SYMMETRY_TOL).unwrap();
        let n = det.molecule.atoms.len();
        let pos = |i: usize| {
            let a = &det.molecule.atoms[i];
            [a.x, a.y, a.zpos]
        };
        for (k, &g) in det.ops.iter().enumerate() {
            let mut seen = vec![false; n];
            for i in 0..n {
                let j = det.atom_perms[k][i];
                seen[j] = true;
                assert_eq!(det.molecule.atoms[i].z, det.molecule.atoms[j].z);
                for ax in 0..3 {
                    let image = pos(i)[ax] * OP_SIGNS[g][ax] as f64;
                    assert!(
                        (image - pos(j)[ax]).abs() < 1e-12,
                        "{} {}",
                        group.name(),
                        OP_NAMES[g]
                    );
                }
            }
            assert!(seen.iter().all(|&s| s), "permutation is a bijection");
        }
        for (a, &g) in det.ops.iter().enumerate() {
            for (b, &h) in det.ops.iter().enumerate() {
                let gh = compose(g, h);
                let c = det
                    .ops
                    .iter()
                    .position(|&o| o == gh)
                    .expect("group closes under composition");
                for i in 0..n {
                    assert_eq!(
                        det.atom_perms[c][i],
                        det.atom_perms[a][det.atom_perms[b][i]]
                    );
                }
            }
        }
    }
}

#[test]
fn irrep_count_equals_group_order_and_characters_are_orthogonal() {
    for (_, ops) in GROUP_OPS {
        let ir = irreps(ops).unwrap();
        assert_eq!(ir.len(), ops.len());
        assert!(
            ir[0].characters.iter().all(|&c| c == 1),
            "totally symmetric first"
        );
        for (a, x) in ir.iter().enumerate() {
            for (b, y) in ir.iter().enumerate() {
                let ip: i32 = x
                    .characters
                    .iter()
                    .zip(&y.characters)
                    .map(|(p, q)| (*p * *q) as i32)
                    .sum();
                assert_eq!(ip, if a == b { ops.len() as i32 } else { 0 });
            }
        }
        let labels: Vec<_> = ir.iter().map(|r| r.label).collect();
        let mut uniq = labels.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), labels.len(), "labels are distinct: {labels:?}");
    }
    let c2v: Vec<_> = irreps(&[0, 1, 6, 7])
        .unwrap()
        .iter()
        .map(|r| r.label)
        .collect();
    assert_eq!(c2v, ["A1", "A2", "B1", "B2"]);
}

const WATER: &str = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
const NH3: &str = "4\n\nN 0 0 0.1\nH 0.94 0 -0.3\nH -0.47 0.8140 -0.3\nH -0.47 -0.8140 -0.3\n";
const ETHYLENE: &str = "6\n\nC 0 0 0.667\nC 0 0 -0.667\nH 0 0.92 1.24\nH 0 -0.92 1.24\nH 0 0.92 -1.24\nH 0 -0.92 -1.24\n";
const METHANE: &str = "5\n\nC 0 0 0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\n";
const H2: &str = "2\n\nH 0 0 0\nH 0 0 0.74\n";
const ASYM: &str = "4\n\nC 0 0 0\nO 1.2 0.1 0\nH -0.5 0.9 0.3\nF 0.2 -1.1 0.8\n";

fn benzene() -> Molecule {
    let mut s = String::from("12\n\n");
    for i in 0..6 {
        let t = i as f64 * std::f64::consts::FRAC_PI_3;
        s += &format!("C {} {} 0\n", 1.397 * t.cos(), 1.397 * t.sin());
    }
    for i in 0..6 {
        let t = i as f64 * std::f64::consts::FRAC_PI_3;
        s += &format!("H {} {} 0\n", 2.48 * t.cos(), 2.48 * t.sin());
    }
    xyz(&s, 1)
}

fn group_of(mol: &Molecule) -> AbelianGroup {
    detect(mol, DEFAULT_SYMMETRY_TOL).unwrap().group
}

#[test]
fn named_molecules_resolve_to_their_largest_abelian_subgroup() {
    assert_eq!(group_of(&xyz(WATER, 1)), AbelianGroup::C2v);
    assert_eq!(group_of(&xyz(NH3, 1)), AbelianGroup::Cs, "C3v -> Cs");
    assert_eq!(group_of(&xyz(ETHYLENE, 1)), AbelianGroup::D2h);
    assert_eq!(group_of(&benzene()), AbelianGroup::D2h, "D6h -> D2h");
    assert_eq!(group_of(&xyz(METHANE, 1)), AbelianGroup::D2, "Td -> D2");
    assert_eq!(group_of(&xyz(H2, 1)), AbelianGroup::D2h, "D-inf-h -> D2h");
    assert_eq!(group_of(&xyz(ASYM, 1)), AbelianGroup::C1);
    assert_eq!(group_of(&xyz("1\n\nNe 0 0 0\n", 1)), AbelianGroup::D2h);
}

#[test]
fn a_stretched_linear_molecule_keeps_d2h() {
    let far = format!("2\n\nH 0 0 0\nH 0 0 {}\n", 10000.0);
    assert_eq!(group_of(&xyz(&far, 1)), AbelianGroup::D2h);
}

#[test]
fn a_ghost_atom_is_a_different_species() {
    let ghost = "3\n\nO 0.0 0.0 0.1173\n@H 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
    assert_eq!(
        group_of(&Molecule::parse_xyz(ghost, 0, 2).unwrap()),
        AbelianGroup::Cs
    );
}

#[test]
fn a_distortion_above_tolerance_lowers_the_group_and_one_below_is_removed() {
    let water = xyz(WATER, 1);
    let bump = |d: f64| {
        let mut m = water.clone();
        m.atoms[1].y += d;
        m
    };
    let det = detect(&bump(1e-3), DEFAULT_SYMMETRY_TOL).unwrap();
    assert_eq!(
        det.group,
        AbelianGroup::Cs,
        "1e-3 Bohr distortion must break C2v"
    );
    assert_eq!(group_of(&bump(1e-2)), AbelianGroup::Cs);
    assert_eq!(
        detect(&bump(1e-3), 1e-2).unwrap().group,
        AbelianGroup::C2v,
        "tolerance is what decides"
    );
    let tiny = detect(&bump(2e-6), DEFAULT_SYMMETRY_TOL).unwrap();
    assert_eq!(tiny.group, AbelianGroup::C2v);
    assert!(tiny.max_symmetrization_shift > 1e-7 && tiny.max_symmetrization_shift < 4e-6);
}

#[test]
fn detection_rejects_a_bad_tolerance_and_an_empty_molecule() {
    assert!(detect(&xyz(WATER, 1), 0.0).is_err());
    assert!(detect(&xyz(WATER, 1), f64::NAN).is_err());
    assert!(detect(&mol_of(vec![]), 1e-4).is_err());
}

#[test]
fn input_frame_and_order_do_not_change_the_group() {
    let water = xyz(WATER, 1);
    let mut moved = water.clone();
    for a in moved.atoms.iter_mut() {
        let p = rigid([a.x, a.y, a.zpos]);
        (a.x, a.y, a.zpos) = (p[0], p[1], p[2]);
    }
    moved.atoms.reverse();
    let det = detect(&moved, DEFAULT_SYMMETRY_TOL).unwrap();
    assert_eq!(det.group, AbelianGroup::C2v);
    assert_eq!(det.ops, [0, 1, 6, 7]);
    // the symmetry frame puts the unique (O) atom on the z axis
    let o = det.molecule.atoms.iter().find(|a| a.z == 8).unwrap();
    assert!(o.x.abs() < 1e-12 && o.y.abs() < 1e-12);
}

#[test]
fn parity_table_matches_the_hand_derived_signs() {
    // pure p is (y, z, x)
    assert_eq!(ao_function_parity(1, 0, true), [0, 1, 0]);
    assert_eq!(ao_function_parity(1, 1, true), [0, 0, 1]);
    assert_eq!(ao_function_parity(1, 2, true), [1, 0, 0]);
    // pure d is (xy, yz, z2, xz, x2-y2)
    let d: Vec<_> = (0..5).map(|k| ao_function_parity(2, k, true)).collect();
    assert_eq!(d, [[1, 1, 0], [0, 1, 1], [0, 0, 0], [1, 0, 1], [0, 0, 0]]);
    // Cartesian d is (xx, xy, xz, yy, yz, zz)
    let c: Vec<_> = (0..6).map(|k| ao_function_parity(2, k, false)).collect();
    assert_eq!(
        c,
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 0]
        ]
    );
    // pure f: m=-3 (3x2y-y3) and m=3 (x3-3xy2)
    assert_eq!(ao_function_parity(3, 0, true), [0, 1, 0]);
    assert_eq!(ao_function_parity(3, 6, true), [1, 0, 0]);
}

#[test]
fn every_function_has_the_parity_of_its_degree() {
    for l in 0..=6 {
        for pure in [true, false] {
            for k in 0..crate::basis::num_functions(l, pure) {
                let p = ao_function_parity(l, k, pure);
                assert_eq!(
                    ((p[0] + p[1] + p[2]) as i32) % 2,
                    l % 2,
                    "l={l} k={k} pure={pure}"
                );
            }
        }
    }
}

fn all_cartesian(bs: &crate::basis::BasisSet) -> crate::basis::BasisSet {
    let mut out = bs.clone();
    for shells in out.shells.values_mut() {
        shells.iter_mut().for_each(|s| s.pure = false);
    }
    out
}

fn representation_cases() -> Vec<(&'static str, Molecule, crate::basis::BasisSet)> {
    let mut v = Vec::new();
    for (name, m) in [
        ("water", xyz(WATER, 1)),
        ("nh3", xyz(NH3, 1)),
        ("ethylene", xyz(ETHYLENE, 1)),
        ("benzene", benzene()),
        ("methane", xyz(METHANE, 1)),
        ("h2", xyz(H2, 1)),
        ("asym", xyz(ASYM, 1)),
    ] {
        for b in ["sto-3g", "cc-pvdz", "def2-svp", "aug-cc-pvtz"] {
            v.push((name, m.clone(), bundled(b).unwrap()));
        }
        v.push((name, m.clone(), all_cartesian(&bundled("cc-pvdz").unwrap())));
    }
    v.push(("water", xyz(WATER, 1), bundled("def2-qzvp").unwrap()));
    v
}

#[test]
fn ao_representation_is_an_orthogonal_representation_of_the_group() {
    for (name, mol, bs) in representation_cases() {
        let det = detect(&mol, DEFAULT_SYMMETRY_TOL).unwrap();
        let rep = AoRepresentation::new(&det, &bs).unwrap();
        let n = rep.nao();
        let eye = ndarray::Array2::<f64>::eye(n);
        for (a, &g) in det.ops.iter().enumerate() {
            let ra = rep.dense(a);
            assert_eq!(ra.t().dot(&ra), eye, "{name}: R orthogonal");
            for (b, &h) in det.ops.iter().enumerate() {
                let c = det.ops.iter().position(|&o| o == compose(g, h)).unwrap();
                assert_eq!(
                    ra.dot(&rep.dense(b)),
                    rep.dense(c),
                    "{name}: R(g)R(h)=R(gh)"
                );
            }
        }
    }
}

#[test]
fn irrep_dimensions_sum_to_nao_and_projectors_resolve_the_identity() {
    for (name, mol, bs) in representation_cases() {
        let det = detect(&mol, DEFAULT_SYMMETRY_TOL).unwrap();
        let rep = AoRepresentation::new(&det, &bs).unwrap();
        let dims = rep.irrep_dimensions().unwrap();
        assert_eq!(dims.iter().map(|d| d.1).sum::<usize>(), rep.nao(), "{name}");
        let mut total = ndarray::Array2::<f64>::zeros((rep.nao(), rep.nao()));
        for (ir, d) in &dims {
            let p = rep.projector(ir);
            assert!(
                (p.dot(&p) - &p).iter().all(|x| x.abs() < 1e-12),
                "{name} {}: idempotent",
                ir.label
            );
            assert!(
                (p.diag().sum() - *d as f64).abs() < 1e-12,
                "{name}: trace is the dimension"
            );
            let u = rep.so_basis(ir);
            assert!((u.t().dot(&u) - ndarray::Array2::<f64>::eye(*d))
                .iter()
                .all(|x| x.abs() < 1e-12));
            assert!(
                (u.dot(&u.t()) - &p).iter().all(|x| x.abs() < 1e-12),
                "{name}: P = U U^T"
            );
            total += &p;
        }
        assert!(
            (total - ndarray::Array2::<f64>::eye(rep.nao()))
                .iter()
                .all(|x| x.abs() < 1e-12),
            "{name}"
        );
    }
}

#[test]
fn water_cc_pvdz_irrep_dimensions_are_11_2_4_7() {
    let det = detect(&xyz(WATER, 1), DEFAULT_SYMMETRY_TOL).unwrap();
    let rep = AoRepresentation::new(&det, &bundled("cc-pvdz").unwrap()).unwrap();
    let mut d: Vec<usize> = rep
        .irrep_dimensions()
        .unwrap()
        .iter()
        .map(|x| x.1)
        .collect();
    d.sort_unstable();
    assert_eq!(d, [2, 4, 7, 11]);
}
