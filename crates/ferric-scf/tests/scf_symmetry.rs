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
//! The PySCF comparison is `validation_scf_symmetry.rs` (ignored per-commit);
//! its references come from `scripts/validation/gen_scf_symmetry.py`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::symmetry::{detect, irreps, AoRepresentation, DEFAULT_SYMMETRY_TOL};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::symmetric::{solve_rhf_symmetric, SymmetricRhf};
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

/// Occupied-orbital count per irrep label of a blocked result.
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

// ---------------------------------------------------------------- anchor (a)

#[test]
fn c1_molecule_is_bit_identical_with_symmetry_on() {
    let mut mol = load("nh3.xyz");
    mol.atoms[1].x += 0.11;
    mol.atoms[1].y -= 0.07;
    mol.atoms[1].zpos += 0.05;
    let off = run_mol(&mol, "sto-3g", &cfg(false)).unwrap();
    let sym = run_sym(&mol, "sto-3g", &cfg(true)).unwrap();
    assert_eq!(sym.group(), "C1");
    let on = sym.scf_in_frame();
    assert_eq!(off.energy.to_bits(), on.energy.to_bits());
    assert_eq!(off.mos_alpha, on.mos_alpha);
    assert_eq!(off.eps_alpha, on.eps_alpha);
    assert_eq!(off.iterations, on.iterations);
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
        let sym = run_sym(&mol, "sto-3g", &cfg(true)).unwrap();
        assert_eq!(sym.group(), c.group, "{}", c.xyz);
        let on = sym.scf_in_frame();
        assert!(on.converged, "{}", c.xyz);
        // Unblocked reference in the SAME frame (the symmetric geometry).
        let off = run_mol(sym.molecule(), "sto-3g", &cfg(false)).unwrap();
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
        let blocked_occ = occupation(&sym, nocc(&mol));
        for ir in irreps(&det.ops).unwrap() {
            let p = rep.projector(&ir).unwrap();
            let n_h = 2.0 * p.dot(&off.density_alpha).dot(&s).diag().sum();
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
        eprintln!(
            "{}: |dE| = {:.1e}, max [R,F] = {ncomm:.1e}, max [R,D] = {dcomm:.1e}",
            c.xyz,
            (on.energy - off.energy).abs()
        );
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
    let ground = run_sym(&mol, "sto-3g", &cfg(true)).unwrap();
    assert_eq!(
        occupation(&ground, 5),
        BTreeMap::from([("A1".into(), 3), ("B1".into(), 1), ("B2".into(), 1)])
    );
    let mut c = cfg(true);
    c.irrep_nelec = Some(water_excited_nelec());
    let ex = run_sym(&mol, "sto-3g", &c).unwrap();
    assert!(ex.scf_in_frame().converged);
    assert_eq!(
        occupation(&ex, 5),
        BTreeMap::from([("A1".into(), 4), ("B2".into(), 1)])
    );
    // The excited determinant is not the aufbau state, and it is higher.
    assert!(
        ex.energy() - ground.energy() > 0.5,
        "{}",
        ex.energy() - ground.energy()
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
        ("aurora", Box::new(|c| c.aurora.enabled = true)),
        ("ah", Box::new(|c| c.ah_trigger = 1e-2)),
        (
            "fractional occupation",
            Box::new(|c| c.fractional_occ = true),
        ),
        (
            "stability descent",
            Box::new(|c| c.scf_stability_descent = true),
        ),
        (
            "constraints",
            Box::new(|c| {
                c.constraints = vec![ferric_dft::cdft::Constraint {
                    fragment: vec![0],
                    spin: ferric_dft::cdft::SpinChannel::Total,
                    target: 8.0,
                }]
            }),
        ),
        (
            "cosmo",
            Box::new(|c| c.cosmo = Some(ferric_scf::CosmoConfig::default())),
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
            "guess density",
            Box::new(|c| c.init_guess_density = Some(Array2::zeros((7, 7)))),
        ),
    ];
    let mut survivors = Vec::new();
    for (name, tweak) in cases {
        let mut c = cfg(true);
        tweak(&mut c);
        match run_sym(&mol, "sto-3g", &c) {
            Err(err) if err.contains("cannot be combined") => {}
            other => survivors.push(format!("{name}: {:?}", other.map(|r| r.energy()))),
        }
    }
    assert!(survivors.is_empty(), "NOT refused: {survivors:#?}");
    // irrep_nelec without symmetry, and bad requests.
    let mut c = cfg(false);
    c.irrep_nelec = Some(vec![("A1".into(), 10)]);
    assert!(run_sym(&mol, "sto-3g", &c).is_err());
    for bad in [
        vec![("Z9".to_string(), 2)],
        vec![("A1".to_string(), 3)],
        vec![("A1".to_string(), 12)],
        vec![("A1".to_string(), 2), ("A1".to_string(), 2)],
    ] {
        let mut c = cfg(true);
        c.irrep_nelec = Some(bad.clone());
        assert!(run_sym(&mol, "sto-3g", &c).is_err(), "{bad:?}");
    }
}

/// Every other solver that reads `RhfConfig` refuses the symmetry fields, and
/// the plain `solve_rhf` points at the typed entry point instead of returning a
/// frame-rotated `ScfResult` that other crates could misuse.
#[test]
fn other_solvers_refuse_symmetry() {
    let mol = load("water.xyz");
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let c = cfg(true);
    let mut only_nelec = cfg(false);
    only_nelec.irrep_nelec = Some(vec![("A1".into(), 10)]);
    let plain = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg(false)).unwrap();
    let errs: Vec<(&str, Option<String>)> = vec![
        (
            "solve_rhf",
            solve_rhf(&ctx, &mol, &prep, op, &bounds, &c)
                .err()
                .map(|e| e.to_string()),
        ),
        (
            "solve_rhf irrep_nelec only",
            solve_rhf(&ctx, &mol, &prep, op, &bounds, &only_nelec)
                .err()
                .map(|e| e.to_string()),
        ),
        (
            "solve_uhf",
            ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &c)
                .err()
                .map(|e| e.to_string()),
        ),
        (
            "solve_rohf",
            ferric_scf::rohf::solve_rohf(&ctx, &mol, &prep, op, &bounds, &c)
                .err()
                .map(|e| e.to_string()),
        ),
        (
            "rhf_hessian",
            ferric_scf::hessian::rhf_hessian(&ctx, &mol, &prep, op, &bounds, &plain, &c)
                .err()
                .map(|e| e.to_string()),
        ),
        (
            "validate_injected",
            ferric_scf::rhf::validate_injected(&c)
                .err()
                .map(|e| e.field.to_string()),
        ),
    ];
    let survivors: Vec<_> = errs
        .iter()
        .filter(|(_, e)| e.is_none())
        .map(|(n, _)| *n)
        .collect();
    assert!(survivors.is_empty(), "NOT refused: {survivors:?}");
    // Each refusal names its own entry point and the way out.
    for (i, who) in [
        (0, "solve_rhf"),
        (1, "solve_rhf"),
        (2, "solve_uhf"),
        (3, "solve_rohf"),
        (4, "rhf_hessian"),
    ] {
        let msg = errs[i].1.as_ref().unwrap();
        assert!(
            msg.contains(who) && msg.contains("solve_rhf_symmetric"),
            "{}: {msg}",
            errs[i].0
        );
    }
    assert_eq!(errs[5].1.as_deref(), Some("symmetry"));
}

/// The input geometry is rotated and translated: detection must reorient, and
/// the blocked energy must equal the unblocked energy of the input frame.
#[test]
fn rotated_translated_input_matches_unblocked_input_frame() {
    // Rodrigues rotation about (1, 2, 3)/|.| by 0.7 rad, then a shift.
    let (ax, ang) = ([1.0f64, 2.0, 3.0], 0.7f64);
    let n = (ax.iter().map(|v| v * v).sum::<f64>()).sqrt();
    let k = [ax[0] / n, ax[1] / n, ax[2] / n];
    let (c, s) = (ang.cos(), ang.sin());
    for xyz in ["water.xyz", "nh3.xyz", "c2h4.xyz"] {
        let mut mol = load(xyz);
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
        let sym = run_sym(&mol, "sto-3g", &cfg(true)).unwrap();
        assert_ne!(sym.group(), "C1", "{xyz}");
        assert!(sym.max_symmetrization_shift() < 1e-4, "{xyz}");
        let off = run_mol(&mol, "sto-3g", &cfg(false)).unwrap();
        assert!(
            (sym.energy() - off.energy).abs() < 1e-9,
            "{xyz}: {} vs {}",
            sym.energy(),
            off.energy
        );
    }
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
        let blocked = run_sym(&mol, "cc-pvdz", &on).unwrap();
        assert!(
            (blocked.energy() - want).abs() < 1e-9,
            "R={r}: {} vs {want}",
            blocked.energy()
        );
        let plain = run_mol(&mol, "cc-pvdz", &cfg(false));
        eprintln!(
            "H2 R={r}: blocked-ref = {:.2e}; unblocked {:?}",
            blocked.energy() - want,
            plain.as_ref().map(|p| (p.energy, p.converged))
        );
        // A failed (non-converged) unblocked run is also a miss.
        if let Ok(p) = plain {
            assert!(
                (p.energy - want).abs() > 1e-3,
                "R={r}: unblocked reached sigma_g^2 ({} vs {want}); the negative control fails",
                p.energy
            );
        }
    }
}

/// The unblocked control above must not CLAIM convergence at a non-stationary
/// state. Before the RHF orbital-gradient veto (`rhf::GradientVeto`), plain
/// Pulay settled ΔP/ΔE on a stretched-H2 state with `err_max` 0.156 and
/// reported `converged = true`. MEASURED with the veto: both runs end
/// non-converged at 200 iterations (either way not σ_g², the negative
/// control). The invariant pinned is the implication, not the outcome.
#[test]
fn stretched_h2_unblocked_plain_pulay_does_not_claim_a_nonstationary_state() {
    for r in [50.0, 10000.0] {
        let c = cfg(false);
        let p = run_mol(&h2(r), "cc-pvdz", &c).expect("non-convergence is Ok(converged=false)");
        let rep = p.convergence.unwrap();
        eprintln!(
            "H2 R={r}: E {:.10} converged={} iters={} err_max={:.3e} vetoes={:?}",
            p.energy, p.converged, p.iterations, rep.err_max, rep.gradient_vetoes
        );
        assert!(
            !p.converged || rep.err_max <= c.density_conv.sqrt().max(1e-3),
            "R={r}: reported converged at err_max {:.3e}",
            rep.err_max
        );
    }
}

/// A free-atom solve that reuses a symmetry config must not refuse (and so
/// silently fall back to Slater proatoms): the provider clears the fields.
#[test]
fn proatom_provider_ignores_symmetry_fields() {
    let bs = basis::bundled("sto-3g").unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let c = cfg(true);
    let provider = ferric_scf::properties::scf_proatom_provider(&ctx, &bs, op, &c);
    for z in [10, 8] {
        // Ne takes the RHF path, O (triplet) the UHF path; both refuse symmetry.
        assert!(provider(z, 0).is_some(), "Z={z}: SCF proatom fell back");
    }
}

/// Species identity includes the ECP core count: an ECP molecule must detect,
/// block and agree with the unblocked solve.
#[test]
fn ecp_molecule_blocked_matches_unblocked() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz("2\nHI\nI 0.3 0.2 0.1\nH 0.3 0.2 1.7\n", 0, 1).unwrap();
    mol.apply_ecp(&bs);
    assert!(mol.atoms[0].n_core_ecp > 0);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let sym = solve_rhf_symmetric(&ctx, &mol, &prep, op, &bounds, &cfg(true)).unwrap();
    assert_eq!(sym.group(), "C2v");
    let off = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg(false)).unwrap();
    assert!(
        (sym.energy() - off.energy).abs() < 1e-8,
        "{} vs {}",
        sym.energy(),
        off.energy
    );
}

/// CSB and CSAM screening requested by the caller must reach the blocked
/// solve. Screening only drops quartets below the threshold, so the energy
/// agrees with the unblocked solve under the same screen to within the
/// screening error either way; this guards against plain Schwarz being
/// substituted silently (see `symmetric::tests::frame_bounds_*` for the
/// flavour itself).
#[test]
fn csb_and_csam_screens_run_through_the_blocked_solve() {
    use ferric_scf::screening::ScreeningKind;
    let mol = load("water.xyz");
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    for kind in [ScreeningKind::Csb, ScreeningKind::Csam] {
        let bounds = SchwarzBounds::compute_for_screening(op, &prep, kind).unwrap();
        let sym = solve_rhf_symmetric(&ctx, &mol, &prep, op, &bounds, &cfg(true)).unwrap();
        let off = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg(false)).unwrap();
        assert!(
            (sym.energy() - off.energy).abs() < 1e-8,
            "{kind:?}: {} vs {}",
            sym.energy(),
            off.energy
        );
    }
}
