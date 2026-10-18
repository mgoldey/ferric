//! Point-group-blocked open-shell UHF and ROHF (issue #294, stage 3a).
//!
//! Anchors, written before any measurement:
//! (a) C1 trivial limit: `symmetry = true` on a C1 molecule is BIT-identical to
//!     `symmetry = false` (UHF and ROHF);
//! (b) blocked energy == unblocked energy to 1e-10 with the same per-irrep
//!     (alpha, beta) occupations, also on a rotated+translated input;
//! (c) the converged F_alpha / F_beta (and the densities) commute with every
//!     operation;
//! (d) `irrep_nelec` reaches an excited occupation that aufbau does not;
//! (e) <S^2> is the same blocked and unblocked; negative control: H2 at 2 A
//!     has a LOWER symmetry-broken UHF solution that the blocked solver cannot
//!     reach (checked, not assumed).
//! The PySCF comparison is `validation_uhf_symmetry.rs` (ignored per-commit);
//! its references come from `scripts/validation/gen_uhf_symmetry.py`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::symmetry::{detect, irreps, AoRepresentation, DEFAULT_SYMMETRY_TOL};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::symmetric_open::{
    solve_rohf_symmetric, solve_uhf_symmetric, IrrepSpinCount, SymmetricRohf, SymmetricUhf,
};
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::Array2;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

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

fn setup(mol: &Molecule, basis_name: &str) -> (PreparedBasis, SchwarzBounds) {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (prep, bounds)
}

fn cfg(symmetry: bool, rohf: bool) -> RhfConfig {
    RhfConfig {
        symmetry,
        density_conv: 1e-10,
        energy_conv: 1e-12,
        max_iter: 300,
        // The F6 guard crosses irreps; the blocked ROHF refuses it.
        rohf_occupation_guard: !(symmetry && rohf),
        ..RhfConfig::default()
    }
}

fn uhf_off(mol: &Molecule, b: &str) -> ScfResult {
    let (prep, bounds) = setup(mol, b);
    solve_uhf(
        &ParallelContext::default(),
        mol,
        &prep,
        &bounds,
        &cfg(false, false),
    )
    .unwrap()
}

fn rohf_off(mol: &Molecule, b: &str) -> ScfResult {
    let (prep, bounds) = setup(mol, b);
    let op = Operator::coulomb();
    solve_rohf(
        &ParallelContext::default(),
        mol,
        &prep,
        op,
        &bounds,
        &cfg(false, true),
    )
    .unwrap()
}

fn uhf_on(mol: &Molecule, b: &str, req: Option<&[IrrepSpinCount]>) -> Result<SymmetricUhf, String> {
    let (prep, bounds) = setup(mol, b);
    solve_uhf_symmetric(
        &ParallelContext::default(),
        mol,
        &prep,
        &bounds,
        &cfg(true, false),
        req,
    )
    .map_err(|e| e.to_string())
}

fn rohf_on(
    mol: &Molecule,
    b: &str,
    req: Option<&[IrrepSpinCount]>,
) -> Result<SymmetricRohf, String> {
    let (prep, bounds) = setup(mol, b);
    let op = Operator::coulomb();
    solve_rohf_symmetric(
        &ParallelContext::default(),
        mol,
        &prep,
        op,
        &bounds,
        &cfg(true, true),
        req,
    )
    .map_err(|e| e.to_string())
}

fn counts(v: Vec<IrrepSpinCount>) -> BTreeMap<String, (usize, usize)> {
    v.into_iter().collect()
}

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

/// <S^2> = Sz(Sz+1) + nb - Tr(Pa S Pb S) from the AO spin densities.
fn s_squared(r: &ScfResult, s: &Array2<f64>, na: usize, nb: usize) -> f64 {
    let pa = &r.density_alpha;
    let pb = r.density_beta.as_ref().unwrap();
    let sz = 0.5 * (na as f64 - nb as f64);
    let m = pa.dot(s).dot(pb).dot(s);
    sz * (sz + 1.0) + nb as f64 - m.diag().sum()
}

/// Occupied weight of each irrep projector in an UNBLOCKED density, per spin.
fn projector_weights(
    mol: &Molecule,
    b: &str,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
) -> BTreeMap<String, (f64, f64)> {
    let det = detect(mol, DEFAULT_SYMMETRY_TOL).unwrap();
    let bs = basis::bundled(b).unwrap();
    let rep = AoRepresentation::new(&det, &det.molecule, &bs).unwrap();
    let prep = PreparedBasis::new(&det.molecule, &bs).unwrap();
    let s = ferric_integrals::oneelectron::overlap(&prep);
    let mut out = BTreeMap::new();
    for ir in irreps(&det.ops).unwrap() {
        let p = rep.projector(&ir).unwrap();
        let w = |d: &Array2<f64>| p.dot(d).dot(&s).diag().sum();
        out.insert(ir.label.to_string(), (w(d_a), w(d_b)));
    }
    out
}

fn assert_same_occupation(
    want: &BTreeMap<String, (f64, f64)>,
    got: &BTreeMap<String, (usize, usize)>,
    what: &str,
) {
    for (label, (wa, wb)) in want {
        let (ga, gb) = got.get(label).copied().unwrap_or((0, 0));
        assert!(
            (wa - ga as f64).abs() < 1e-6 && (wb - gb as f64).abs() < 1e-6,
            "{what} irrep {label}: unblocked ({wa:.4}, {wb:.4}) vs blocked ({ga}, {gb})"
        );
    }
}

// ---------------------------------------------------------------- anchor (a)

fn distorted(rel: &str, charge: i32, mult: usize) -> Molecule {
    let mut mol = load(rel, charge, mult);
    mol.atoms[1].x += 0.11;
    mol.atoms[1].y -= 0.07;
    mol.atoms[1].zpos += 0.05;
    mol.atoms[2].x -= 0.04;
    mol.atoms[2].zpos -= 0.06;
    mol
}

#[test]
fn c1_molecule_is_bit_identical_with_symmetry_on() {
    // Four atoms, generic displacements: three atoms are always planar (Cs).
    let mol = distorted("validation/ch3.xyz", 0, 2);
    let off = uhf_off(&mol, "sto-3g");
    let sym = uhf_on(&mol, "sto-3g", None).unwrap();
    assert_eq!(sym.group(), "C1");
    let on = sym.scf_in_frame();
    assert_eq!(off.energy.to_bits(), on.energy.to_bits());
    assert_eq!(off.mos_alpha, on.mos_alpha);
    assert_eq!(off.mos_beta, on.mos_beta);
    assert_eq!(off.iterations, on.iterations);

    // Like with like: the symmetric entry point runs with the F6 guard off
    // (guard-on is refused, see the refusal test), so compare against guard off.
    let (prep, bounds) = setup(&mol, "sto-3g");
    let guard_off = RhfConfig {
        rohf_occupation_guard: false,
        ..cfg(false, true)
    };
    let off = solve_rohf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &guard_off,
    )
    .unwrap();
    let sym = rohf_on(&mol, "sto-3g", None).unwrap();
    assert_eq!(sym.group(), "C1");
    assert_eq!(off.energy.to_bits(), sym.scf_in_frame().energy.to_bits());
    assert_eq!(off.mos_alpha, sym.scf_in_frame().mos_alpha);
    assert_eq!(off.iterations, sym.scf_in_frame().iterations);
}

// ---------------------------------------------------------------- anchors (b), (c), (e)

struct Case {
    xyz: &'static str,
    charge: i32,
    mult: usize,
    group: &'static str,
}
const CASES: [Case; 4] = [
    Case {
        xyz: "validation/nh2.xyz",
        charge: 0,
        mult: 2,
        group: "C2v",
    },
    Case {
        xyz: "validation/ch3.xyz",
        charge: 0,
        mult: 2,
        group: "C2v",
    },
    Case {
        xyz: "o2.xyz",
        charge: 0,
        mult: 3,
        group: "D2h",
    },
    Case {
        xyz: "validation/ch2_triplet.xyz",
        charge: 0,
        mult: 3,
        group: "C2v",
    },
];

#[test]
fn blocked_uhf_matches_unblocked_and_commutes() {
    for c in CASES {
        let mol = load(c.xyz, c.charge, c.mult);
        let sym = uhf_on(&mol, "sto-3g", None).unwrap();
        assert_eq!(sym.group(), c.group, "{}", c.xyz);
        let on = sym.scf_in_frame();
        assert!(on.converged, "{}", c.xyz);
        let fm = sym.molecule();
        let off = uhf_off(fm, "sto-3g");
        assert!(
            (on.energy - off.energy).abs() < 1e-10,
            "{}: blocked {} vs unblocked {}",
            c.xyz,
            on.energy,
            off.energy
        );
        let (na, nb) = ((fm.nelec() + fm.multiplicity as i32 - 1) / 2, 0);
        let nb = fm.nelec() - na + nb;
        let (prep, _) = setup(fm, "sto-3g");
        let s = ferric_integrals::oneelectron::overlap(&prep);
        let s2_on = s_squared(on, &s, na as usize, nb as usize);
        let s2_off = s_squared(&off, &s, na as usize, nb as usize);
        assert!(
            (s2_on - s2_off).abs() < 1e-8,
            "{}: <S2> {s2_on} vs {s2_off}",
            c.xyz
        );
        let w = projector_weights(
            &mol,
            "sto-3g",
            &off.density_alpha,
            off.density_beta.as_ref().unwrap(),
        );
        assert_same_occupation(&w, &counts(sym.irrep_occupations()), c.xyz);
        let det = detect(&mol, DEFAULT_SYMMETRY_TOL).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let rep = AoRepresentation::new(&det, &det.molecule, &bs).unwrap();
        let nops = det.ops.len();
        let fa = max_noncommutation(&rep, nops, &on.fock_alpha);
        let fb = max_noncommutation(&rep, nops, on.fock_beta.as_ref().unwrap());
        let dd = max_noncommutation(&rep, nops, &on.density_total);
        eprintln!(
            "UHF {}: |dE| = {:.1e}, [R,Fa] = {fa:.1e}, [R,Fb] = {fb:.1e}, [R,D] = {dd:.1e}, <S2> = {s2_on:.5}",
            c.xyz,
            (on.energy - off.energy).abs()
        );
        assert!(fa < 1e-10 && fb < 1e-10 && dd < 1e-10, "{}", c.xyz);
    }
}

#[test]
fn blocked_rohf_matches_unblocked_and_commutes() {
    for c in CASES {
        let mol = load(c.xyz, c.charge, c.mult);
        let sym = rohf_on(&mol, "sto-3g", None).unwrap();
        assert_eq!(sym.group(), c.group, "{}", c.xyz);
        let on = sym.scf_in_frame();
        assert!(on.converged, "{}", c.xyz);
        let off = rohf_off(sym.molecule(), "sto-3g");
        assert!(
            (on.energy - off.energy).abs() < 1e-10,
            "{}: blocked {} vs unblocked {}",
            c.xyz,
            on.energy,
            off.energy
        );
        let w = projector_weights(
            &mol,
            "sto-3g",
            &off.density_alpha,
            off.density_beta.as_ref().unwrap(),
        );
        assert_same_occupation(&w, &counts(sym.irrep_occupations()), c.xyz);
        let det = detect(&mol, DEFAULT_SYMMETRY_TOL).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let rep = AoRepresentation::new(&det, &det.molecule, &bs).unwrap();
        let nops = det.ops.len();
        let fa = max_noncommutation(&rep, nops, &on.fock_alpha);
        let dd = max_noncommutation(&rep, nops, &on.density_total);
        eprintln!(
            "ROHF {}: |dE| = {:.1e}, [R,F] = {fa:.1e}, [R,D] = {dd:.1e}",
            c.xyz,
            (on.energy - off.energy).abs()
        );
        assert!(fa < 1e-10 && dd < 1e-10, "{}", c.xyz);
    }
}

/// d functions (cc-pVDZ) exercise the pure-harmonic parity table that the
/// s/p-only STO-3G anchors cannot see (p shells are Cartesian-equivalent).
#[test]
fn d_functions_blocked_matches_unblocked() {
    for case in [&CASES[0], &CASES[2]] {
        let mol = load(case.xyz, case.charge, case.mult);
        let u = uhf_on(&mol, "cc-pvdz", None).unwrap();
        let e_off = uhf_off(u.molecule(), "cc-pvdz").energy;
        assert!(
            (u.energy() - e_off).abs() < 1e-9,
            "UHF {}: {} vs {e_off}",
            case.xyz,
            u.energy()
        );
        let r = rohf_on(&mol, "cc-pvdz", None).unwrap();
        let e_off = rohf_off(r.molecule(), "cc-pvdz").energy;
        assert!(
            (r.energy() - e_off).abs() < 1e-9,
            "ROHF {}: {} vs {e_off}",
            case.xyz,
            r.energy()
        );
    }
}

/// The input geometry is rotated and translated: detection must reorient, and
/// the blocked energy must equal the unblocked energy of the INPUT frame.
#[test]
fn rotated_translated_input_matches_unblocked_input_frame() {
    let (ax, ang) = ([1.0f64, 2.0, 3.0], 0.7f64);
    let n = (ax.iter().map(|v| v * v).sum::<f64>()).sqrt();
    let k = [ax[0] / n, ax[1] / n, ax[2] / n];
    let (c, s) = (ang.cos(), ang.sin());
    for case in [&CASES[0], &CASES[2]] {
        let mut mol = load(case.xyz, case.charge, case.mult);
        for a in &mut mol.atoms {
            let v = [a.x, a.y, a.zpos];
            let kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
            let cr = [
                k[1] * v[2] - k[2] * v[1],
                k[2] * v[0] - k[0] * v[2],
                k[0] * v[1] - k[1] * v[0],
            ];
            let r: Vec<f64> = (0..3)
                .map(|i| v[i] * c + cr[i] * s + k[i] * kv * (1.0 - c))
                .collect();
            a.x = r[0] + 1.3;
            a.y = r[1] - 0.8;
            a.zpos = r[2] + 2.1;
        }
        let u = uhf_on(&mol, "sto-3g", None).unwrap();
        assert_ne!(u.group(), "C1", "{}", case.xyz);
        assert!(u.max_symmetrization_shift() < 1e-4);
        let e_off = uhf_off(&mol, "sto-3g").energy;
        assert!(
            (u.energy() - e_off).abs() < 1e-9,
            "UHF {}: {} vs {e_off}",
            case.xyz,
            u.energy()
        );
        let r = rohf_on(&mol, "sto-3g", None).unwrap();
        let e_off = rohf_off(&mol, "sto-3g").energy;
        assert!(
            (r.energy() - e_off).abs() < 1e-9,
            "ROHF {}: {} vs {e_off}",
            case.xyz,
            r.energy()
        );
        eprintln!(
            "{}: shift {:.1e} Bohr, UHF dE {:.1e}",
            case.xyz,
            u.max_symmetrization_shift(),
            (u.energy() - uhf_off(&mol, "sto-3g").energy).abs()
        );
    }
}

// ---------------------------------------------------------------- anchor (d)

fn water_cation_excited() -> Vec<IrrepSpinCount> {
    vec![
        ("A1".into(), (3, 2)),
        ("B1".into(), (1, 1)),
        ("B2".into(), (1, 1)),
    ]
}

#[test]
fn irrep_nelec_reaches_the_excited_water_cation_state() {
    let mol = load("water.xyz", 1, 2);
    for rohf in [false, true] {
        let (ground, excited, ground_occ, excited_occ) = if rohf {
            let g = rohf_on(&mol, "sto-3g", None).unwrap();
            let e = rohf_on(&mol, "sto-3g", Some(&water_cation_excited())).unwrap();
            assert!(e.scf_in_frame().converged);
            (
                g.energy(),
                e.energy(),
                g.irrep_occupations(),
                e.irrep_occupations(),
            )
        } else {
            let g = uhf_on(&mol, "sto-3g", None).unwrap();
            let e = uhf_on(&mol, "sto-3g", Some(&water_cation_excited())).unwrap();
            assert!(e.scf_in_frame().converged);
            (
                g.energy(),
                e.energy(),
                g.irrep_occupations(),
                e.irrep_occupations(),
            )
        };
        assert_eq!(
            counts(ground_occ),
            BTreeMap::from([
                ("A1".into(), (3, 3)),
                ("B1".into(), (1, 0)),
                ("B2".into(), (1, 1))
            ]),
            "rohf={rohf}"
        );
        assert_eq!(
            counts(excited_occ),
            counts(water_cation_excited()),
            "rohf={rohf}"
        );
        // 2A1 lies ~2.3 eV above the 2B1 ground state of H2O+.
        assert!(excited - ground > 0.05, "rohf={rohf}: {}", excited - ground);
    }
}

// ---------------------------------------------------------------- negative control

#[test]
fn stretched_h2_unrestricted_solution_is_lower_than_the_symmetric_one() {
    let mol = Molecule::parse_xyz("2\nh2\nH 0 0 1.0\nH 0 0 -1.0\n", 0, 1).unwrap();
    let b = "6-31g";
    // The unblocked UHF needs its stability descent to leave the restricted
    // saddle (plain aufbau UHF stays on it, as the HOMO/LUMO mix relaxes back).
    let (prep0, bounds0) = setup(&mol, b);
    let descent = RhfConfig {
        scf_stability_descent: true,
        check_stability: true,
        ..cfg(false, false)
    };
    let broken = solve_uhf(
        &ParallelContext::default(),
        &mol,
        &prep0,
        &bounds0,
        &descent,
    )
    .unwrap();
    let sym = uhf_on(&mol, b, None).unwrap();
    // Symmetric: both spins sigma_g^2, equal to the restricted energy.
    assert_eq!(
        counts(sym.irrep_occupations()),
        BTreeMap::from([("Ag".into(), (1, 1))])
    );
    let (prep, bounds) = setup(&mol, b);
    let rhf = ferric_scf::rhf::solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &cfg(false, false),
    )
    .unwrap();
    assert!((sym.energy() - rhf.energy).abs() < 1e-9);
    eprintln!(
        "H2 2 A: broken {:.8} vs symmetric {:.8} (RHF {:.8})",
        broken.energy,
        sym.energy(),
        rhf.energy
    );
    assert!(
        broken.energy < sym.energy() - 0.05,
        "no lower symmetry-broken solution: {} vs {}",
        broken.energy,
        sym.energy()
    );
}

// ---------------------------------------------------------------- refusals

#[test]
fn unsupported_combinations_are_typed_errors() {
    let mol = load("validation/nh2.xyz", 0, 2);
    let tweaks: Vec<(&str, Box<dyn Fn(&mut RhfConfig)>)> = vec![
        ("MOM", Box::new(|c| c.mom_after_iter = 3)),
        ("smearing", Box::new(|c| c.smearing_sigma = Some(0.01))),
        ("newton", Box::new(|c| c.newton_trigger = 1e-2)),
        ("trah", Box::new(|c| c.trah_trigger = Some(1e-2))),
        ("stability", Box::new(|c| c.check_stability = true)),
        ("xc", Box::new(|c| c.xc = Some("PBE".into()))),
        ("aurora", Box::new(|c| c.aurora.enabled = true)),
        ("ah", Box::new(|c| c.ah_trigger = 1e-2)),
        ("fractional", Box::new(|c| c.fractional_occ = true)),
        ("descent", Box::new(|c| c.scf_stability_descent = true)),
        (
            "guess density",
            Box::new(|c| c.init_guess_density = Some(Array2::zeros((7, 7)))),
        ),
        (
            "cosmo",
            Box::new(|c| c.cosmo = Some(ferric_scf::CosmoConfig::default())),
        ),
        (
            "pcm",
            Box::new(|c| c.pcm = Some(ferric_pcm::PcmConfig::water())),
        ),
        (
            "polarizable",
            Box::new(|c| {
                c.polarizable = Some(ferric_scf::polarizable::PolarizableSites {
                    sites: vec![ferric_scf::polarizable::PolarizableSite {
                        x: 6.0,
                        y: 0.0,
                        z: 0.0,
                        alpha: 1.0,
                    }],
                    thole_a: None,
                    exclusions: vec![],
                    dipole_zeta: 1e6,
                    max_sites_dense: 100,
                })
            }),
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
            "constraints",
            Box::new(|c| {
                c.constraints = vec![ferric_dft::cdft::Constraint {
                    fragment: vec![0],
                    spin: ferric_dft::cdft::SpinChannel::Total,
                    target: 7.0,
                }]
            }),
        ),
    ];
    let (prep, bounds) = setup(&mol, "sto-3g");
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let mut survivors = Vec::new();
    for (name, tweak) in &tweaks {
        let mut c = cfg(true, true);
        tweak(&mut c);
        let u = solve_uhf_symmetric(&ctx, &mol, &prep, &bounds, &c, None);
        let r = solve_rohf_symmetric(&ctx, &mol, &prep, op, &bounds, &c, None);
        for (who, res) in [("uhf", u.err()), ("rohf", r.err())] {
            match res {
                Some(e) if e.to_string().contains("cannot be combined") => {}
                other => survivors.push(format!("{who} {name}: {other:?}")),
            }
        }
    }
    assert!(survivors.is_empty(), "NOT refused: {survivors:#?}");

    // The ROHF occupation guard crosses irreps.
    let mut guard = cfg(true, false);
    guard.rohf_occupation_guard = true;
    let e = solve_rohf_symmetric(&ctx, &mol, &prep, op, &bounds, &guard, None).unwrap_err();
    assert!(e.to_string().contains("rohf_occupation_guard"), "{e}");
    // symmetry off, and the closed-shell irrep_nelec field.
    let e = solve_uhf_symmetric(&ctx, &mol, &prep, &bounds, &cfg(false, false), None).unwrap_err();
    assert!(e.to_string().contains("symmetry = true"), "{e}");
    let mut c = cfg(true, false);
    c.irrep_nelec = Some(vec![("A1".into(), 4)]);
    let e = solve_uhf_symmetric(&ctx, &mol, &prep, &bounds, &c, None).unwrap_err();
    assert!(e.to_string().contains("(n_alpha, n_beta)"), "{e}");

    // solve_uhf / solve_rohf point at the symmetric entry points.
    let c = cfg(true, false);
    let e = solve_uhf(&ctx, &mol, &prep, &bounds, &c)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("solve_uhf") && e.contains("solve_uhf_symmetric"),
        "{e}"
    );
    let e = solve_rohf(&ctx, &mol, &prep, op, &bounds, &c)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("solve_rohf") && e.contains("solve_rohf_symmetric"),
        "{e}"
    );
    let mut only = cfg(false, false);
    only.irrep_nelec = Some(vec![("A1".into(), 4)]);
    assert!(solve_uhf(&ctx, &mol, &prep, &bounds, &only).is_err());
    assert!(solve_rohf(&ctx, &mol, &prep, op, &bounds, &only).is_err());
}

#[test]
fn bad_irrep_requests_are_errors() {
    let mol = load("validation/nh2.xyz", 0, 2); // (5, 4)
    let bad: Vec<(&str, Vec<IrrepSpinCount>)> = vec![
        ("unknown irrep", vec![("Z9".into(), (1, 1))]),
        (
            "listed twice",
            vec![("A1".into(), (1, 1)), ("A1".into(), (1, 1))],
        ),
        ("too many for the irrep", vec![("A2".into(), (9, 9))]),
        (
            "sum != molecule",
            vec![
                ("A1".into(), (3, 3)),
                ("A2".into(), (0, 0)),
                ("B1".into(), (1, 0)),
                ("B2".into(), (0, 0)),
            ],
        ),
        ("more alpha than the molecule", vec![("A1".into(), (6, 0))]),
    ];
    for (what, req) in &bad {
        assert!(uhf_on(&mol, "sto-3g", Some(req)).is_err(), "uhf {what}");
        assert!(rohf_on(&mol, "sto-3g", Some(req)).is_err(), "rohf {what}");
    }
    // Exceeds only the irrep's own dimension (2 <= the molecule's 5 alpha electrons).
    let small = vec![("A2".to_string(), (2, 0))];
    for e in [
        uhf_on(&mol, "sto-3g", Some(&small)).err().unwrap(),
        rohf_on(&mol, "sto-3g", Some(&small)).err().unwrap(),
    ] {
        assert!(e.contains("A2") && e.contains("basis functions"), "{e}");
    }
    // An empty request is an error on C1 and non-C1 alike.
    let c1 = distorted("validation/ch3.xyz", 0, 2);
    for m in [&mol, &c1] {
        let e = uhf_on(m, "sto-3g", Some(&[])).err().unwrap();
        assert!(e.contains("empty"), "{e}");
        let e = rohf_on(m, "sto-3g", Some(&[])).err().unwrap();
        assert!(e.contains("empty"), "{e}");
    }
    // ROHF cannot hold more beta than alpha in an irrep.
    let inverted = vec![("A1".into(), (2, 3)), ("B1".into(), (2, 0))];
    let e = rohf_on(&mol, "sto-3g", Some(&inverted)).unwrap_err();
    assert!(e.contains("n_alpha"), "{e}");
}
