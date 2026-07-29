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

/// ECP gradients must ERROR, not return a silently wrong answer.
///
/// Before 2026-07-28 `rhf_gradient` accepted ECP molecules and produced a
/// gradient that was wrong two ways: the dV_ECP/dR term was missing entirely,
/// and the nuclear-repulsion derivative used the bare `z` while the ENERGY it
/// differentiates uses `effective_z()`. `ferric-cli` calls `mol.apply_ecp()`
/// unconditionally, so `task = "optimize"` on any Z >= 37 element optimized on
/// that wrong gradient with no error and no warning.
///
/// This is the coverage gap that let it survive: `ecp_rhf.rs` had zero gradient
/// tests. Asserting the ERROR (rather than a gradient value) is the point — the
/// repo convention is clean error over silent-wrong, and this pins the honest
/// behaviour until the libecpint derivative API is bound.
#[test]
fn ecp_gradient_errors_rather_than_returning_a_wrong_answer() {
    let ctx = ParallelContext::default();
    let bs = basis::bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz("1\n\nXe 0.0 0.0 0.0\n", 0, 1).unwrap();
    mol.apply_ecp(&bs);

    // TEETH: the ECP must actually be active, or this test proves nothing about
    // the ECP path — it would just be testing a plain all-electron Xe.
    assert!(
        mol.atoms.iter().any(|a| a.n_core_ecp > 0),
        "def2-svp must carry an ECP for Xe; without one this test is vacuous"
    );

    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();

    let err = ferric_scf::gradient::rhf_gradient(&mol, &prep, op, &bounds, &rhf, None)
        .expect_err("ECP gradient must be rejected, not silently computed");
    let msg = err.to_string();
    assert!(
        msg.contains("ECP"),
        "the error must name ECPs so the user knows why: got {msg}"
    );
}
