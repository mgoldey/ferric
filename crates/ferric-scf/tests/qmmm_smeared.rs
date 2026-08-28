//! Gaussian-smeared MM charge tests: analytic gradients vs finite difference
//! of ferric's own energy (Task A3), and cross-validation against PySCF's
//! `qmmm.mm_charge(radii=)` (Task A4).
//!
//! # Artifact hypothesis
//!
//! `smeared_charge_qm_gradient`/`smeared_site_forces` (in `gradient.rs`) read
//! the 9-block `compute_eri3_deriv` layout `[d/d(site), d/d(sh1), d/d(sh2)] ×
//! [x,y,z]`. If the block index were off by one group of 3 (e.g. reading
//! block 0 where block 1 was meant), the QM-atom gradient rows would either
//! be silently ZERO (block 0 belongs to the site, not any QM atom — a wrong
//! `sh2at` lookup would put mass on the wrong or a nonexistent atom row) or
//! would fail the translational-invariance check in `full_gradient_columns...`
//! below, since block 0 = −(block 1 + block 2) is an identity that a
//! misindexed contraction does not respect for a generic (non-symmetric)
//! shell pair. A sign error in the `−q_i/norm_i` prefactor would flip EVERY
//! component of EVERY test here uniformly — caught by the FD comparisons
//! (which have no shared arithmetic with the analytic path) rather than by
//! the translational-invariance check (which is sign-blind: −x + x = 0
//! regardless of the overall sign of x).

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge, SmearedCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{rhf_gradient, smeared_site_forces};
use ferric_scf::qmmm::{full_gradient, mm_forces, QmSelection, QmmmAtom, QmmmSystem};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_92;

fn water_bohr() -> Molecule {
    let r = 0.9572 * ANG2BOHR;
    let half = 104.52_f64.to_radians() / 2.0;
    let xyz = format!(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 {} {}\nH 0.0 {} {}\n",
        r * half.sin() / ANG2BOHR,
        r * half.cos() / ANG2BOHR,
        -r * half.sin() / ANG2BOHR,
        r * half.cos() / ANG2BOHR,
    );
    Molecule::parse_xyz(&xyz, 0, 1).unwrap()
}

fn sto3g_prep(mol: &Molecule) -> PreparedBasis {
    let bs = basis::bundled("sto-3g").unwrap();
    PreparedBasis::new(mol, &bs).unwrap()
}

fn scf_energy(mol: &Molecule, ext: Option<&ExternalPotential>) -> f64 {
    let prep = sto3g_prep(mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { external_potential: ext.cloned(), density_conv: 1e-11, ..Default::default() };
    let r = solve_rhf(&ctx, mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged);
    r.energy
}

/// One smeared site at (1.5, -2, 3) Bohr, width 1.0 Bohr, q = 0.6 — an
/// off-axis, non-round placement so every gradient component is live.
fn one_smeared_site() -> ExternalPotential {
    ExternalPotential {
        point_charges: vec![],
        smeared_charges: vec![SmearedCharge { q: 0.6, x: 1.5, y: -2.0, z: 3.0, width: 1.0 }],
        field: None,
    }
}

/// **Task A3, test 1**: analytic QM gradient vs central FD of the SCF energy,
/// water/STO-3G, one smeared site (width 1.0 Bohr) at (1.5, -2, 3).
#[test]
fn smeared_qm_gradient_matches_finite_difference() {
    let mol = water_bohr();
    let ext = one_smeared_site();
    let prep = sto3g_prep(&mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { external_potential: Some(ext.clone()), density_conv: 1e-11, ..Default::default() };
    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(result.converged);

    let analytic = rhf_gradient(&mol, &prep, op, &bounds, &result, Some(&ext)).unwrap();

    let h = 1e-3;
    let natoms = mol.atoms.len();
    let mut max_err = 0.0_f64;
    for a in 0..natoms {
        for c in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match c {
                0 => { mol_p.atoms[a].x += h; mol_m.atoms[a].x -= h; }
                1 => { mol_p.atoms[a].y += h; mol_m.atoms[a].y -= h; }
                _ => { mol_p.atoms[a].zpos += h; mol_m.atoms[a].zpos -= h; }
            }
            let e_p = scf_energy(&mol_p, Some(&ext));
            let e_m = scf_energy(&mol_m, Some(&ext));
            let fd = (e_p - e_m) / (2.0 * h);
            let err = (analytic[(a, c)] - fd).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-6,
                "QM gradient[{a}][{c}]: analytic {:.10e} vs FD {:.10e} (Δ {err:.3e})",
                analytic[(a, c)],
                fd
            );
        }
    }
    eprintln!("[qmmm-smeared] max|analytic - FD| on QM gradient: {max_err:.3e}");
}

/// **Task A3, test 2**: the force on the smeared site itself vs FD of the SCF
/// energy under a site displacement.
#[test]
fn smeared_site_force_matches_finite_difference() {
    let mol = water_bohr();
    let prep = sto3g_prep(&mol);

    let base = one_smeared_site();
    let site = base.smeared_charges[0];

    // Analytic: solve at the base geometry, then read the site force via
    // smeared_site_forces (dE/dR, so force = -that).
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { external_potential: Some(base.clone()), density_conv: 1e-11, ..Default::default() };
    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(result.converged);
    let d_total = result.density_total();
    let dedr = smeared_site_forces(&mol, &prep, d_total, &base.smeared_charges).unwrap();
    assert_eq!(dedr.len(), 1);

    let h = 1e-3;
    for (c, coord_name) in [(0, "x"), (1, "y"), (2, "z")] {
        let mut site_p = site;
        let mut site_m = site;
        match c {
            0 => { site_p.x += h; site_m.x -= h; }
            1 => { site_p.y += h; site_m.y -= h; }
            _ => { site_p.z += h; site_m.z -= h; }
        }
        let ext_p = ExternalPotential { point_charges: vec![], smeared_charges: vec![site_p], field: None };
        let ext_m = ExternalPotential { point_charges: vec![], smeared_charges: vec![site_m], field: None };
        let e_p = scf_energy(&mol, Some(&ext_p));
        let e_m = scf_energy(&mol, Some(&ext_m));
        let fd = (e_p - e_m) / (2.0 * h);
        let err = (dedr[0][c] - fd).abs();
        assert!(
            err < 1e-6,
            "site dE/dR[{coord_name}]: analytic {:.10e} vs FD {:.10e} (Δ {err:.3e})",
            dedr[0][c],
            fd
        );
    }
}

/// **Task A3, test 3**: `full_gradient` column sums vanish (translational
/// invariance) for a system with ONLY smeared sites — i.e. the QM-centre
/// contribution and the site force must sum to exactly cancel under a rigid
/// translation of the whole structure (both are derivatives of the same
/// translation-invariant energy).
#[test]
fn full_gradient_translational_invariance_smeared_only() {
    // Build a QmmmSystem: 3 QM atoms (water) + 2 MM sites, both smeared.
    let mut atoms: Vec<QmmmAtom> = water_atoms_full();
    atoms.push(QmmmAtom::new_smeared("X", 0, 1.5, -2.0, 3.0, 0.6, 1.0));
    atoms.push(QmmmAtom::new_smeared("X", 0, -3.0, 1.0, -4.0, -0.3, 0.8));

    let qm: Vec<usize> = (0..3).collect();
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(qm), 0, 1).unwrap();
    let mol = sys.to_qm_molecule();
    let ep = sys.to_external_potential().expect("smeared MM charges present");
    assert_eq!(ep.point_charges.len(), 0);
    assert_eq!(ep.smeared_charges.len(), 2);

    let prep = sto3g_prep(&mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { external_potential: Some(ep.clone()), density_conv: 1e-11, ..Default::default() };
    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(result.converged);

    let qm_grad = rhf_gradient(&mol, &prep, op, &bounds, &result, Some(&ep)).unwrap();
    let mm_f = mm_forces(&sys, &mol, &prep, result.density_total()).unwrap();
    assert_eq!(mm_f.len(), 2);
    let full = full_gradient(&sys, &qm_grad, &mm_f).unwrap();

    let mut col_sum = [0.0_f64; 3];
    for row in full.rows() {
        col_sum[0] += row[0];
        col_sum[1] += row[1];
        col_sum[2] += row[2];
    }
    for (k, s) in col_sum.iter().enumerate() {
        assert!(s.abs() < 1e-8, "column {k} sum = {s:.3e} (expected 0, translational invariance)");
    }
}

fn water_atoms_full() -> Vec<QmmmAtom> {
    let r = 0.9572 * ANG2BOHR;
    let half = 104.52_f64.to_radians() / 2.0;
    vec![
        QmmmAtom::new("O", 8, 0.0, 0.0, 0.0, 0.0),
        QmmmAtom::new("H", 1, 0.0, r * half.sin(), r * half.cos(), 0.0),
        QmmmAtom::new("H", 1, 0.0, -r * half.sin(), r * half.cos(), 0.0),
    ]
}

// ── Task A4: PySCF cross-validation ──

#[derive(Deserialize)]
struct RefAtom {
    symbol: String,
    xyz_bohr: [f64; 3],
}

#[derive(Deserialize)]
struct RefCharge {
    q: f64,
    xyz_bohr: [f64; 3],
}

#[derive(Deserialize)]
struct SmearedRef {
    atoms: Vec<RefAtom>,
    charge: i32,
    multiplicity: usize,
    mm_charges: Vec<RefCharge>,
    radii: Vec<f64>,
    energy: f64,
    energy_gas_phase: f64,
    qm_gradient: Vec<[f64; 3]>,
    mm_gradient: Vec<[f64; 3]>,
    converged: bool,
}

fn load(name: &str) -> SmearedRef {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference")
        .join(name);
    let r: SmearedRef = serde_json::from_str(&fs::read_to_string(&p).unwrap()).unwrap();
    assert!(r.converged, "reference {name} is not converged");
    r
}

fn z_of(symbol: &str) -> i32 {
    match symbol {
        "H" => 1,
        "O" => 8,
        s => panic!("unexpected element {s}"),
    }
}

fn check_smeared(name: &str) {
    let r = load(name);
    assert_eq!(r.mm_charges.len(), r.radii.len());

    let mut atoms: Vec<QmmmAtom> = r
        .atoms
        .iter()
        .map(|a| QmmmAtom::new(a.symbol.clone(), z_of(&a.symbol), a.xyz_bohr[0], a.xyz_bohr[1], a.xyz_bohr[2], 99.0))
        .collect();
    for (c, &width) in r.mm_charges.iter().zip(r.radii.iter()) {
        atoms.push(QmmmAtom::new_smeared("X", 0, c.xyz_bohr[0], c.xyz_bohr[1], c.xyz_bohr[2], c.q, width));
    }
    let qm: Vec<usize> = (0..r.atoms.len()).collect();
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(qm), r.charge, r.multiplicity).unwrap();
    let mol = sys.to_qm_molecule();
    assert_eq!(mol.atoms.len(), r.atoms.len());
    let ep = sys.to_external_potential().expect("reference has MM charges");
    assert_eq!(ep.point_charges.len(), 0, "all reference MM charges are smeared");
    assert_eq!(ep.smeared_charges.len(), r.mm_charges.len());

    let prep = sto3g_prep(&mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig { external_potential: Some(ep.clone()), density_conv: 1e-10, ..Default::default() };
    let gas_cfg = RhfConfig { density_conv: 1e-10, ..Default::default() };
    let scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let gas = solve_rhf(&ctx, &mol, &prep, op, &bounds, &gas_cfg).unwrap();
    assert!(scf.converged && gas.converged);

    let qm_grad = rhf_gradient(&mol, &prep, op, &bounds, &scf, cfg.external_potential.as_ref()).unwrap();
    let forces = mm_forces(&sys, &mol, &prep, scf.density_total()).unwrap();

    let de_gas = (gas.energy - r.energy_gas_phase).abs();
    assert!(de_gas < 5e-8, "{name}: gas-phase energy off by {de_gas:.3e}");

    let shift_f = scf.energy - gas.energy;
    let shift_p = r.energy - r.energy_gas_phase;
    assert!(
        (shift_f - shift_p).abs() < 1e-8,
        "{name}: embedding shift ferric {shift_f:.10e} vs PySCF {shift_p:.10e}"
    );

    assert_eq!(forces.len(), r.mm_gradient.len());
    for (i, (f, g)) in forces.iter().zip(r.mm_gradient.iter()).enumerate() {
        for k in 0..3 {
            let d = (f[k] + g[k]).abs();
            assert!(d < 1e-6, "{name}: MM force[{i}][{k}] ferric {} vs PySCF -grad {} (Δ {d:.3e})", f[k], -g[k]);
        }
    }

    assert_eq!(qm_grad.nrows(), r.qm_gradient.len());
    for (a, g) in r.qm_gradient.iter().enumerate() {
        for k in 0..3 {
            let d = (qm_grad[(a, k)] - g[k]).abs();
            assert!(
                d < 1e-6,
                "{name}: QM gradient[{a}][{k}] ferric {} vs PySCF {} (Δ {d:.3e})",
                qm_grad[(a, k)],
                g[k]
            );
        }
    }

    eprintln!("[qmmm-smeared-vs-pyscf] {name}: shift ferric {shift_f:+.8} / PySCF {shift_p:+.8}");
}

#[test]
fn smeared_single_site_matches_pyscf() {
    check_smeared("water_sto-3g_qmmm_smeared_r1.json");
}

#[test]
fn smeared_offaxis_distinct_widths_match_pyscf() {
    check_smeared("water_sto-3g_qmmm_smeared_offaxis.json");
}

/// Anchor: SiteBasis's tight-zeta test pins the normalisation constant in
/// isolation; this pins it through the FULL stack the plan asks for — a
/// `PointCharge` and a `width=1e-3` `SmearedCharge` at the same position must
/// give the same SCF energy and QM gradient to 1e-9 / 1e-8 respectively.
#[test]
fn tiny_width_scf_matches_point_charge_scf() {
    let mol = water_bohr();
    let (q, x, y, z) = (1.0, 0.0, 0.0, -6.0);

    let ext_point = ExternalPotential {
        point_charges: vec![PointCharge { q, x, y, z }],
        smeared_charges: vec![],
        field: None,
    };
    let ext_smeared = ExternalPotential {
        point_charges: vec![],
        smeared_charges: vec![SmearedCharge { q, x, y, z, width: 1e-3 }],
        field: None,
    };

    let prep = sto3g_prep(&mol);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg_point = RhfConfig { external_potential: Some(ext_point.clone()), density_conv: 1e-11, ..Default::default() };
    let cfg_smeared = RhfConfig { external_potential: Some(ext_smeared.clone()), density_conv: 1e-11, ..Default::default() };
    let r_point = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_point).unwrap();
    let r_smeared = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_smeared).unwrap();
    assert!(r_point.converged && r_smeared.converged);

    let de = (r_point.energy - r_smeared.energy).abs();
    assert!(de < 1e-9, "tiny-width SCF energy vs point-charge SCF energy differ by {de:.3e}");

    let g_point = rhf_gradient(&mol, &prep, op, &bounds, &r_point, Some(&ext_point)).unwrap();
    let g_smeared = rhf_gradient(&mol, &prep, op, &bounds, &r_smeared, Some(&ext_smeared)).unwrap();
    let max_diff: f64 = (&g_point - &g_smeared).iter().fold(0.0, |acc: f64, &v| acc.max(v.abs()));
    assert!(max_diff < 1e-8, "tiny-width gradient vs point-charge gradient differ by {max_diff:.3e}");
}

// NOTE: this file used to carry `empty_smeared_charges_gradient_is_bit_identical`
// here, which called `rhf_gradient` twice with the IDENTICAL `ext` argument
// and compared the two results — a tautology (any function that is even
// merely deterministic passes it; it could never fail even if the
// `is_empty()` guard in `oneelectron_gradient` were deleted entirely, since
// `smeared_charge_qm_gradient` itself independently short-circuits on an
// empty slice and returns a zero matrix, making the outer guard redundant
// rather than load-bearing). Removed by mutation-test: deleting the guard
// leaves this file's tests exactly as green as before, which is the
// signature of a test that checks nothing.
//
// The real, non-tautological claim -- "point charges plus an EXPLICIT EMPTY
// `smeared_charges` vec reproduces the pre-Lane-A point-charge-only
// gradient" -- is already covered by two independent anchors elsewhere,
// neither of which is a self-comparison:
//   - `oneelectron_gradient_external_charge_matches_finite_difference`
//     (crates/ferric-scf/src/gradient.rs) uses this EXACT `ExternalPotential`
//     shape (one point charge, `smeared_charges: Vec::new()`) and checks the
//     analytic gradient against a central finite difference of the SCF
//     energy -- an independent numerical computation, not a second call to
//     the same function.
//   - `hcore_with_external_empty_smeared_charges_matches_point_charge_only_path`
//     (crates/ferric-integrals/src/oneelectron.rs) checks the energy-level
//     (hcore) analogue by reconstructing the point-charge-only hcore via a
//     DIFFERENT code path (`kinetic(prep) + nuclear_with_external(prep, ext)`,
//     bypassing `hcore_with_external` entirely) and asserting bit-identity.
