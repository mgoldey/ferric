//! RI-MP2 correlation energy on top of an RHF reference.
//!
//! Computes RHF/cc-pVDZ then RI-MP2 with the cc-pVDZ-RI auxiliary basis
//! for water, and prints the MP2 correlation and total energies.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-cli --example api_rimp2

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let mol = Molecule::parse_xyz(
        "3\nwater\nO  0.000000  0.000000  0.117790\nH  0.000000  0.755453 -0.471161\nH  0.000000 -0.755453 -0.471161\n",
        0,
        1,
    )
    .expect("failed to parse XYZ");

    let obs_bs = basis::bundled("cc-pvdz").expect("orbital basis not found");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("orbital basis prep failed");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("Schwarz screening failed");
    let ctx = ParallelContext::default();

    // Step 1: converge the RHF reference
    let scf_cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).expect("SCF failed");
    println!("RHF energy:          {:.10} Ha", rhf.energy);

    // Step 2: RI-MP2 with a density-fitting auxiliary basis
    let aux_bs = basis::bundled("cc-pvdz-ri").expect("auxiliary basis not found");
    let dfbs = PreparedBasis::new(&mol, &aux_bs).expect("auxiliary basis prep failed");

    let mp2_cfg = RiMp2Config::default();
    let mp2 = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg).expect("RI-MP2 failed");

    println!("MP2 correlation:     {:.10} Ha", mp2.mp2_corr);
    println!("Total (RHF + MP2):   {:.10} Ha", mp2.total_energy);
}
