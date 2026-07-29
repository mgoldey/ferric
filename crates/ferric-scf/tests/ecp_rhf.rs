//! Unit 5 + SCF validation: RHF on ECP-treated atoms/molecules (def2-SVP +
//! def2-ECP) must reproduce PySCF total energy to <=1e-5 Ha and HOMO orbital
//! energy to <=1 meV. PySCF uses its own analytic ECP; ferric uses libecpint —
//! cross-implementation agreement, not bit-identity.
//!
//! References from scripts/gw100/gen_ecp_scf_ref.py:
//!   testdata/reference/xe_def2svp_ecp_rhf.json
//!   testdata/reference/i2_def2svp_ecp_rhf.json

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use serde::Deserialize;

const HARTREE_TO_EV: f64 = 27.211_386_245_988;

#[derive(Deserialize)]
struct Ref {
    atom: String,
    nelectron: i32,
    e_tot: f64,
    homo_index: usize,
    e_homo: f64,
    #[serde(default)]
    charge: i32,
}

fn load_ref(name: &str) -> Ref {
    let path = format!(
        "{}/../../testdata/reference/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e}\nrun scripts/gw100/gen_ecp_scf_ref.py"));
    serde_json::from_str(&text).expect("parse reference JSON")
}

/// Build a Molecule from a PySCF-style "Sym x y z; Sym x y z" string in Bohr.
fn mol_from_bohr(atom_spec: &str, charge: i32) -> Molecule {
    let entries: Vec<&str> = atom_spec.split(';').collect();
    let mut xyz = format!("{}\nfrom_bohr\n", entries.len());
    for e in &entries {
        let f: Vec<&str> = e.split_whitespace().collect();
        // ferric parse_xyz reads Angstrom and multiplies by ANGSTROM_TO_BOHR.
        // We already have Bohr, so pre-divide to cancel the conversion.
        const A2B: f64 = 1.0 / 0.529_177_210_92;
        let x: f64 = f[1].parse::<f64>().unwrap() / A2B;
        let y: f64 = f[2].parse::<f64>().unwrap() / A2B;
        let z: f64 = f[3].parse::<f64>().unwrap() / A2B;
        xyz.push_str(&format!("{} {} {} {}\n", f[0], x, y, z));
    }
    Molecule::parse_xyz(&xyz, charge, 1).unwrap()
}

fn run_ecp_rhf(name: &str) {
    let r = load_ref(name);
    let mut mol = mol_from_bohr(&r.atom, r.charge);
    let bs = basis::bundled("def2-svp").unwrap();
    mol.apply_ecp(&bs);
    assert_eq!(mol.nelec(), r.nelectron, "{name}: electron count mismatch");

    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(res.converged, "{name}: ferric RHF did not converge");

    let de = (res.energy - r.e_tot).abs();
    let e_homo = res.eps_alpha[r.homo_index];
    let dhomo_ev = (e_homo - r.e_homo).abs() * HARTREE_TO_EV;
    eprintln!(
        "{name}: E_ferric={:.10} E_pyscf={:.10} dE={de:.2e} Ha | \
         eHOMO_ferric={:.6} eHOMO_pyscf={:.6} d={dhomo_ev:.4} meV*1e3?",
        res.energy, r.e_tot, e_homo, r.e_homo
    );
    eprintln!("  dHOMO = {:.4} meV", dhomo_ev * 1000.0);

    assert!(de < 1e-5, "{name}: |dE| = {de:.3e} Ha exceeds 1e-5");
    assert!(
        dhomo_ev * 1000.0 < 1.0,
        "{name}: |d eps_HOMO| = {:.4} meV exceeds 1 meV",
        dhomo_ev * 1000.0
    );
}

#[test]
fn ecp_rhf_xe_matches_pyscf() {
    run_ecp_rhf("xe_def2svp_ecp_rhf");
}

#[test]
fn ecp_rhf_i2_matches_pyscf() {
    run_ecp_rhf("i2_def2svp_ecp_rhf");
}

/// Rb⁺ (def2-SVP + def2 28-core ECP): the GW100 Rb₂ member runs on the ECP path,
/// not all-electron (def2 Rb is an ECP-valence basis). Closed-shell cation =
/// clean RHF target. nelec 37→9 via the ECP; must match PySCF −23.6625.
#[test]
fn ecp_rhf_rb_cation_matches_pyscf() {
    run_ecp_rhf("rb_cation_def2svp_ecp_rhf");
}

/// Total RHF energy at a given geometry, on the ECP path.
fn ecp_rhf_energy(mol: &Molecule, bs: &ferric_core::basis::BasisSet) -> f64 {
    let ctx = ParallelContext::default();
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        max_iter: 200,
        // Tight convergence: the FD reference differences two total energies, so
        // any SCF slop shows up divided by 2h and would swamp the comparison.
        density_conv: 1e-10,
        ..Default::default()
    };
    let res = solve_rhf(&ctx, mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged, "FD displaced-geometry SCF did not converge");
    res.energy
}

/// END-TO-END ECP GRADIENT: analytic `rhf_gradient` vs central finite difference
/// of the TOTAL ENERGY, on a system with a genuinely NONZERO gradient.
///
/// This is the exactness anchor for the whole ECP gradient stack — it is the only
/// check that exercises the two fixes together:
///   1. the `dV_ECP/dR` term (libecpint first derivatives, bound 2026-07-28);
///   2. the nuclear-repulsion derivative using `effective_z()` rather than the
///      bare `z`.
/// Each is individually capable of producing a plausible-looking but wrong
/// gradient, and only differencing the real energy catches either.
///
/// HI at 1.75 A is deliberately STRETCHED off equilibrium (~1.61 A) so dE/dR is
/// large and unambiguous — a symmetric or equilibrium geometry would give ~0 and
/// prove nothing, since a gradient that is wrong by a constant factor still
/// reads as zero there. The test asserts the gradient is non-negligible before
/// comparing, so it cannot silently degrade into a zero-vs-zero check.
///
/// H/I is also heteronuclear with only ONE ECP center, so the two atoms are
/// distinguishable: an atom-id misattribution in the derivative (libecpint
/// indexes by its own inferred centers, not ferric's atom order) shows up here
/// but would cancel in a homonuclear I2.
#[test]
fn ecp_rhf_gradient_matches_finite_difference_hi() {
    let ctx = ParallelContext::default();
    let bs = basis::bundled("def2-svp").unwrap();
    // Stretched HI (equilibrium ~1.61 A) -> genuinely nonzero dE/dR.
    let mut mol = Molecule::parse_xyz("2\n\nH 0.0 0.0 0.0\nI 0.0 0.0 1.75\n", 0, 1).unwrap();
    mol.apply_ecp(&bs);

    // TEETH: the ECP must actually be active, else this is an all-electron test.
    assert!(mol.atoms[0].n_core_ecp == 0, "H must have no ECP");
    assert!(
        mol.atoms[1].n_core_ecp > 0,
        "def2-svp must carry an ECP for I; without one this test is vacuous"
    );

    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        max_iter: 200,
        density_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "HI reference SCF did not converge");

    let grad = ferric_scf::gradient::rhf_gradient(&mol, &prep, op, &bounds, &rhf, None)
        .expect("ECP gradient must now be supported");

    // Central FD of the total energy. h = 1e-3 Bohr: large enough that SCF
    // noise (~1e-10 Ha) contributes only ~1e-7 Ha/Bohr after dividing by 2h,
    // small enough that O(h^2) truncation stays ~1e-6.
    let h = 1e-3;
    let mut fd = vec![[0.0f64; 3]; mol.atoms.len()];
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let displaced = |sign: f64| {
                let mut m = mol.clone();
                match c {
                    0 => m.atoms[a].x += sign * h,
                    1 => m.atoms[a].y += sign * h,
                    _ => m.atoms[a].zpos += sign * h,
                }
                ecp_rhf_energy(&m, &bs)
            };
            fd[a][c] = (displaced(1.0) - displaced(-1.0)) / (2.0 * h);
        }
    }

    let mut worst = 0.0f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            eprintln!(
                "HI grad atom {a} coord {c}: analytic {:+.9} FD {:+.9} diff {:+.2e}",
                grad[(a, c)],
                fd[a][c],
                grad[(a, c)] - fd[a][c]
            );
            worst = worst.max((grad[(a, c)] - fd[a][c]).abs());
        }
    }
    eprintln!("HI/def2-SVP ECP gradient: max |analytic - FD| = {worst:.3e} Ha/Bohr");

    // The gradient must actually be NONZERO, or the FD agreement below is
    // vacuous (0 == 0 tells us nothing about correctness).
    let max_component = grad.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        max_component > 1e-3,
        "stretched HI must have a clearly nonzero gradient; got max |g| = \
         {max_component:.3e}. A near-zero gradient makes this test vacuous."
    );

    assert!(
        worst < 1e-5,
        "ECP gradient disagrees with finite difference by {worst:.3e} Ha/Bohr"
    );

    // Translational invariance: rigidly shifting the molecule cannot change the
    // energy, so the gradient must sum to ~0 over atoms. Necessary but NOT
    // sufficient (it is blind to an atom-swap), which is why it follows the FD
    // check rather than replacing it.
    for c in 0..3 {
        let s: f64 = (0..mol.atoms.len()).map(|a| grad[(a, c)]).sum();
        eprintln!("HI gradient sum over atoms, coord {c}: {s:+.3e}");
        assert!(
            s.abs() < 1e-6,
            "translational invariance violated on coord {c}: sum = {s:.3e}"
        );
    }
}
