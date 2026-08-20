//! Minimal RHF (restricted Hartree-Fock) energy calculation via the Rust API.
//!
//! Computes the RHF/cc-pVDZ energy of water and prints the result.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-cli --example api_rhf

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let mol = Molecule::parse_xyz(
        "3\nwater\nO  0.000000  0.000000  0.117790\nH  0.000000  0.755453 -0.471161\nH  0.000000 -0.755453 -0.471161\n",
        0, // charge
        1, // multiplicity
    )
    .expect("failed to parse XYZ");

    let bs = basis::bundled("cc-pvdz").expect("basis set not found");
    let prep = PreparedBasis::new(&mol, &bs).expect("basis prep failed");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("Schwarz screening failed");
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };

    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).expect("SCF failed");

    println!("RHF/cc-pVDZ energy:  {:.10} Ha", result.energy);
    println!("Converged:           {}", result.converged);
    println!("Iterations:          {}", result.iterations);
}
