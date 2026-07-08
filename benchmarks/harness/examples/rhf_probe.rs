//! Minimal RHF-only probe for memory diagnosis: runs DF-J/DF-K RHF on one xyz
//! at cc-pVDZ with def2-universal-jkfit, honoring FERRIC_OOC_BUDGET_GB. Used to
//! isolate SCF memory from the RPA path in the c9_driver.
//!
//! Usage: cargo run --release -p ferric-scf --example rhf_probe -- <xyz>

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: rhf_probe <xyz>");
    let mol = Molecule::load_xyz(&path).expect("load xyz");
    eprintln!("[probe] {} atoms", mol.atoms.len());
    let obs_set = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        max_iter: 200,
        energy_conv: 1e-7,
        density_conv: 1e-6,
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        df_k_aux: Some("def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let t0 = Instant::now();
    eprintln!("[probe] entering solve_rhf...");
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("rhf");
    eprintln!("[probe] RHF E={:.8} t={:.1}s", rhf.energy, t0.elapsed().as_secs_f64());
}
