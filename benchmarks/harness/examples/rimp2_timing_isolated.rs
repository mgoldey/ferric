//! Isolated RI-MP2 timing: solves RHF once, then times ONLY the ri_mp2 call
//! (excluding RHF setup/SCF), to measure the occ-first half-transform fix's
//! real-world speedup without RHF wall-clock contaminating the number.
use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let mol = Molecule::load_xyz("testdata/molecules/benzene.xyz").unwrap();
    let obs_set = basis::bundled("aug-cc-pvtz").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    let rhf_cfg = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        max_iter: 100,
        energy_conv: 1e-8,
        density_conv: 1e-7,
        ..Default::default()
    };

    eprintln!("Solving RHF/aug-cc-pVTZ on benzene (not timed)...");
    let t_rhf = Instant::now();
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &rhf_cfg).unwrap();
    eprintln!(
        "RHF done: converged={} iters={} energy={:.10} Ha, wall={:.2}s",
        rhf.converged, rhf.iterations, rhf.energy, t_rhf.elapsed().as_secs_f64()
    );

    let dfbs_set = basis::bundled("aug-cc-pvtz-rifit").unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
    let mp2_cfg = RiMp2Config { frozen_core: 6, memory_budget_bytes: None, ..Default::default() };

    eprintln!("Timing ri_mp2 (isolated, RHF excluded)...");
    let t_mp2 = Instant::now();
    let result = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg).unwrap();
    let elapsed = t_mp2.elapsed().as_secs_f64();

    println!("RI-MP2 corr = {:.10} Ha", result.mp2_corr);
    println!("Total energy = {:.10} Ha", result.total_energy);
    println!("ISOLATED RI-MP2 wall-clock: {:.3}s", elapsed);
}
