//! DFT (PBE) energy calculation via the Rust API.
//!
//! Runs Kohn-Sham DFT with the PBE functional on water/cc-pVDZ using
//! density-fitted Coulomb (RI-J) and the SCF ladder for robust convergence.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-cli --example api_dft

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ladder::{default_ladder_from, solve_rhf_ladder};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let mol = Molecule::parse_xyz(
        "3\nwater\nO  0.000000  0.000000  0.117790\nH  0.000000  0.755453 -0.471161\nH  0.000000 -0.755453 -0.471161\n",
        0,
        1,
    )
    .expect("failed to parse XYZ");

    let bs = basis::bundled("cc-pvdz").expect("basis set not found");
    let prep = PreparedBasis::new(&mol, &bs).expect("basis prep failed");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("Schwarz screening failed");
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };

    // The SCF ladder starts from a simpler functional and ramps up,
    // improving convergence robustness for DFT.
    let ladder = default_ladder_from(&cfg);
    let lr = solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ladder)
        .expect("KS-DFT failed");

    println!("PBE/cc-pVDZ energy:  {:.10} Ha", lr.result.energy);
    println!("Converged:           {}", lr.converged);
    println!("Ladder rung:         {}", lr.rung_reached);
    println!("Iterations (final):  {}", lr.result.iterations);
}
