//! Point-group-blocked closed-shell RHF (issue #294, stage 2).
//!
//! Anchors, in the order they were written (before any measurement):
//! (a) C1 trivial limit: `symmetry = true` on a C1 molecule is BIT-identical to
//!     `symmetry = false`;
//! (b) blocked energy == unblocked energy to 1e-10 with the same per-irrep
//!     occupations, and the converged Fock and density commute with every
//!     operation;
//! (c) `irrep_nelec` reaches an excited occupation that aufbau does not;
//! (d) negative control: H2 at long range misses sigma_g^2 without symmetry.
//! The PySCF comparison is `#[ignore]`d ("validation: SCF symmetry"); its
//! references come from `scripts/validation/gen_scf_symmetry.py`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::symmetry::{detect, irreps, AoRepresentation, DEFAULT_SYMMETRY_TOL};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;
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

fn run_mol(mol: &Molecule, basis_name: &str, cfg: &RhfConfig) -> Result<ScfResult, String> {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, cfg).map_err(|e| e.to_string())
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

/// Occupied-orbital count per irrep label of a blocked result.
fn occupation(r: &ScfResult, nocc: usize) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for l in &r.orbital_irreps().unwrap()[..nocc] {
        *out.entry(l.clone()).or_insert(0) += 1;
    }
    out
}

fn nocc(mol: &Molecule) -> usize {
    (mol.nelec() / 2) as usize
}

// ---------------------------------------------------------------- anchor (a)

#[test]
fn c1_molecule_is_bit_identical_with_symmetry_on() {
    let mut mol = load("nh3.xyz");
    mol.atoms[1].x += 0.11;
    mol.atoms[1].y -= 0.07;
    mol.atoms[1].zpos += 0.05;
    let off = run_mol(&mol, "sto-3g", &cfg(false)).unwrap();
    let on = run_mol(&mol, "sto-3g", &cfg(true)).unwrap();
    assert_eq!(on.symmetry_group(), Some("C1"));
    assert!(on.symmetry.as_ref().unwrap().frame.is_none());
    assert_eq!(off.energy.to_bits(), on.energy.to_bits());
    assert_eq!(off.mos_alpha, on.mos_alpha);
    assert_eq!(off.eps_alpha, on.eps_alpha);
    assert_eq!(off.iterations, on.iterations);
    assert!(off.symmetry.is_none());
}

// ---------------------------------------------------------------- anchor (b)

struct Case {
    xyz: &'static str,
    group: &'static str,
}
const CASES: [Case; 4] = [
    Case {
        xyz: "water.xyz",
        group: "C2v",
    },
    Case {
        xyz: "nh3.xyz",
        group: "Cs",
    },
    Case {
        xyz: "c2h4.xyz",
        group: "D2h",
    },
    Case {
        xyz: "benzene.xyz",
        group: "D2h",
    },
];

/// `max |R X R^T - X|` over the operations, for an AO matrix X.
fn max_noncommutation(rep: &AoRepresentation, nops: usize, x: &Array2<f64>) -> f64 {
    (0..nops)
        .map(|k| {
            let r = rep.dense(k);
            (&r.dot(x).dot(&r.t()) - x)
                .iter()
                .fold(0.0f64, |m, v| m.max(v.abs()))
        })
        .fold(0.0, f64::max)
}

#[test]
fn blocked_matches_unblocked_and_fock_commutes() {
    for c in CASES {
        let mol = load(c.xyz);
        let on = run_mol(&mol, "sto-3g", &cfg(true)).unwrap();
        assert_eq!(on.symmetry_group(), Some(c.group), "{}", c.xyz);
        assert!(on.converged, "{}", c.xyz);
        // Unblocked reference in the SAME frame (the symmetric geometry).
        let frame = on.symmetry.as_ref().unwrap().frame.as_ref().unwrap();
        let off = run_mol(&frame.molecule, "sto-3g", &cfg(false)).unwrap();
        assert!(
            (on.energy - off.energy).abs() < 1e-10,
            "{}: blocked {} vs unblocked {}",
            c.xyz,
            on.energy,
            off.energy
        );
        // Same orbital energies, and the same per-irrep occupation: the
        // occupied-subspace weight in each irrep projector of the UNBLOCKED
        // run is integral and equals the blocked label count.
        for (a, b) in on.eps_alpha.iter().zip(&off.eps_alpha).take(nocc(&mol)) {
            assert!((a - b).abs() < 1e-8, "{}: eps {a} vs {b}", c.xyz);
        }
        let det = detect(&mol, DEFAULT_SYMMETRY_TOL).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let rep = AoRepresentation::new(&det, &det.molecule, &bs).unwrap();
        let prep = PreparedBasis::new(&det.molecule, &bs).unwrap();
        let s = ferric_integrals::oneelectron::overlap(&prep);
        let blocked_occ = occupation(&on, nocc(&mol));
        for ir in irreps(&det.ops).unwrap() {
            let p = rep.projector(&ir).unwrap();
            let n_h = 2.0 * (&p.dot(&off.density_alpha).dot(&s)).diag().sum();
            let expect = 2 * blocked_occ.get(ir.label).copied().unwrap_or(0);
            assert!(
                (n_h - expect as f64).abs() < 1e-6,
                "{} irrep {}: unblocked occupancy {n_h} vs blocked {expect}",
                c.xyz,
                ir.label
            );
        }
        // Commutation with every operation, converged Fock and density.
        let nops = det.ops.len();
        let ncomm = max_noncommutation(&rep, nops, &on.fock_alpha);
        let dcomm = max_noncommutation(&rep, nops, &on.density_total);
        assert!(ncomm < 1e-10, "{}: [R,F] = {ncomm:e}", c.xyz);
        assert!(dcomm < 1e-10, "{}: [R,D] = {dcomm:e}", c.xyz);
    }
}

// ---------------------------------------------------------------- anchor (c)

fn water_excited_nelec() -> Vec<(String, usize)> {
    vec![("A1".into(), 8), ("B2".into(), 2), ("B1".into(), 0)]
}

#[test]
fn irrep_nelec_reaches_the_excited_water_occupation() {
    let mol = load("water.xyz");
    let ground = run_mol(&mol, "sto-3g", &cfg(true)).unwrap();
    assert_eq!(
        occupation(&ground, 5),
        BTreeMap::from([("A1".into(), 3), ("B1".into(), 1), ("B2".into(), 1)])
    );
    let mut c = cfg(true);
    c.irrep_nelec = Some(water_excited_nelec());
    let ex = run_mol(&mol, "sto-3g", &c).unwrap();
    assert!(ex.converged);
    assert_eq!(
        occupation(&ex, 5),
        BTreeMap::from([("A1".into(), 4), ("B2".into(), 1)])
    );
    // The excited determinant is not the aufbau state, and it is higher.
    assert!(
        ex.energy - ground.energy > 0.5,
        "{}",
        ex.energy - ground.energy
    );
}

// ---------------------------------------------------------------- refusals

#[test]
fn unsupported_combinations_are_typed_errors() {
    let mol = load("water.xyz");
    let cases: Vec<(&str, Box<dyn Fn(&mut RhfConfig)>)> = vec![
        ("MOM", Box::new(|c| c.mom_after_iter = 3)),
        ("smearing", Box::new(|c| c.smearing_sigma = Some(0.01))),
        ("newton", Box::new(|c| c.newton_trigger = 1e-2)),
        ("trah", Box::new(|c| c.trah_trigger = Some(1e-2))),
        ("stability", Box::new(|c| c.check_stability = true)),
        ("xc", Box::new(|c| c.xc = Some("PBE".into()))),
        (
            "pcm",
            Box::new(|c| c.pcm = Some(ferric_pcm::PcmConfig::water())),
        ),
        (
            "external potential",
            Box::new(|c| {
                c.external_potential = Some(ferric_core::external_potential::ExternalPotential {
                    point_charges: vec![ferric_core::external_potential::PointCharge {
                        q: 0.1,
                        x: 5.0,
                        y: 0.0,
                        z: 0.0,
                    }],
                    ..Default::default()
                })
            }),
        ),
        (
            "guess density",
            Box::new(|c| c.init_guess_density = Some(Array2::zeros((7, 7)))),
        ),
    ];
    for (name, tweak) in cases {
        let mut c = cfg(true);
        tweak(&mut c);
        let err = run_mol(&mol, "sto-3g", &c).unwrap_err();
        assert!(err.contains("symmetry"), "{name}: {err}");
    }
    // irrep_nelec without symmetry, and bad requests.
    let mut c = cfg(false);
    c.irrep_nelec = Some(vec![("A1".into(), 10)]);
    assert!(run_mol(&mol, "sto-3g", &c).is_err());
    for bad in [
        vec![("Z9".to_string(), 2)],
        vec![("A1".to_string(), 3)],
        vec![("A1".to_string(), 12)],
        vec![("A1".to_string(), 2), ("A1".to_string(), 2)],
    ] {
        let mut c = cfg(true);
        c.irrep_nelec = Some(bad.clone());
        assert!(run_mol(&mol, "sto-3g", &c).is_err(), "{bad:?}");
    }
}

#[test]
fn gradient_refuses_a_symmetry_frame_result() {
    let mol = load("water.xyz");
    let r = run_mol(&mol, "sto-3g", &cfg(true)).unwrap();
    let e = r
        .require_input_frame("rhf_gradient")
        .unwrap_err()
        .to_string();
    assert!(e.contains("symmetry frame"), "{e}");
}

// ---------------------------------------------------------------- anchor (d)

fn h2(r_angstrom: f64) -> Molecule {
    let h = r_angstrom / 2.0;
    Molecule::parse_xyz(&format!("2\nh2\nH 0 0 {h}\nH 0 0 -{h}\n"), 0, 1).unwrap()
}

fn reference(name: &str) -> serde_json::Value {
    let p = root()
        .join("testdata/reference/validation/scf_symmetry")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn stretched_h2_needs_symmetry_to_reach_sigma_g_squared() {
    for (r, file) in [
        (50.0, "h2_r50_cc-pvdz.json"),
        (10000.0, "h2_r10000_cc-pvdz.json"),
    ] {
        let mol = h2(r);
        let want = reference(file)["sigma_g2"]["energy"].as_f64().unwrap();
        let mut on = cfg(true);
        on.irrep_nelec = Some(vec![("Ag".into(), 2)]);
        let blocked = run_mol(&mol, "cc-pvdz", &on).unwrap();
        assert!(
            (blocked.energy - want).abs() < 1e-7,
            "R={r}: {} vs {want}",
            blocked.energy
        );
        let plain = run_mol(&mol, "cc-pvdz", &cfg(false));
        eprintln!(
            "H2 R={r}: blocked {:.10} (ref {want:.10}); unblocked {:?}",
            blocked.energy,
            plain.as_ref().map(|p| (p.energy, p.converged))
        );
        match plain {
            Ok(p) => assert!(
                (p.energy - want).abs() > 1e-3,
                "R={r}: unblocked reached sigma_g^2 ({} vs {want}); the negative control fails",
                p.energy
            ),
            Err(_) => {} // non-convergence is also a miss
        }
    }
}

// ------------------------------------------------------- PySCF comparison

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
            let r = run_mol(&mol, bas, &cfg(true)).unwrap();
            let e = want["ground"]["energy"].as_f64().unwrap();
            worst = worst.max((r.energy - e).abs());
            assert!(
                (r.energy - e).abs() < 1e-8,
                "{sys}/{bas}: {} vs {e}",
                r.energy
            );
            // Per-irrep occupations: labels for C2v/Cs, multiset for D2h.
            let ours = occupation(&r, nocc(&mol));
            let theirs: BTreeMap<String, usize> = want["ground"]["docc_per_irrep"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.replace('"', "''"), v.as_u64().unwrap() as usize))
                .collect();
            if r.symmetry_group() == Some("D2h") {
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
                let r = run_mol(&load("water.xyz"), bas, &c).unwrap();
                let e = want["excited"]["energy"].as_f64().unwrap();
                worst = worst.max((r.energy - e).abs());
                assert!(
                    (r.energy - e).abs() < 1e-8,
                    "excited {bas}: {} vs {e}",
                    r.energy
                );
            }
        }
    }
    eprintln!("max |E_ferric - E_pyscf| = {worst:e}");
}
