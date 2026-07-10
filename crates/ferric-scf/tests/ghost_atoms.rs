//! CP-style ghost-atom RHF tests.
//!
//! Verifies that ghost atoms (XYZ `@` prefix):
//!   1. Contribute their element's basis functions (variational lowering)
//!   2. Do not contribute nuclear charge or electrons (far ghost = same energy)
//!   3. Gradient returns an error for ghost-containing molecules

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn rhf_energy(xyz: &str, basis_name: &str) -> f64 {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let config = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
    assert!(result.converged, "RHF did not converge for: {xyz}");
    result.energy
}

/// He alone (cc-pVDZ).
const HE_ALONE: &str = "1\nHe\nHe 0.0 0.0 0.0\n";

/// He + ghost He at 50 Å — far ghost contributes essentially nothing.
const HE_FAR_GHOST: &str = "2\nHe + far ghost He\nHe 0.0 0.0 0.0\n@He 0.0 0.0 50.0\n";

/// He + ghost He at 3 Å — nearby ghost makes the basis bigger (variational lowering).
const HE_NEAR_GHOST: &str = "2\nHe + near ghost He\nHe 0.0 0.0 0.0\n@He 0.0 0.0 3.0\n";

/// Far ghost: energy should equal He alone to 1e-9 Ha (the ghost at 50 Å barely
/// overlaps with He at the origin, so the extended basis adds almost nothing).
#[test]
fn test_ghost_far_equals_isolated_he() {
    let e_alone = rhf_energy(HE_ALONE, "cc-pvdz");
    let e_far = rhf_energy(HE_FAR_GHOST, "cc-pvdz");
    let diff = (e_alone - e_far).abs();
    eprintln!("He alone = {e_alone:.12}  He+far_ghost = {e_far:.12}  diff = {diff:.3e}");
    assert!(
        diff < 1e-9,
        "RHF(He + far ghost He) should equal RHF(He) to 1e-9 Ha, got diff = {diff:.3e}"
    );
}

/// Near ghost: energy must be lower than or equal to He alone (variational principle —
/// a larger basis can only lower the RHF energy).
#[test]
fn test_ghost_near_lowers_energy() {
    let e_alone = rhf_energy(HE_ALONE, "cc-pvdz");
    let e_near = rhf_energy(HE_NEAR_GHOST, "cc-pvdz");
    eprintln!("He alone = {e_alone:.12}  He+near_ghost = {e_near:.12}  CP lowering = {:.3e}", e_near - e_alone);
    assert!(
        e_near <= e_alone + 1e-12,
        "RHF(He + near ghost He) = {e_near:.12} must be ≤ RHF(He) = {e_alone:.12} (variational)"
    );
    // Sanity: the lowering should be physically meaningful (> 1e-12 Ha at 3 Å)
    let lowering = e_alone - e_near;
    eprintln!("CP lowering = {lowering:.3e} Ha");
    assert!(
        lowering > 1e-12,
        "Expected non-trivial CP lowering at 3 Å, got {lowering:.3e} Ha"
    );
}

/// Gradient guard: rhf_gradient should return Err on a ghost-containing molecule.
#[test]
fn test_gradient_errors_on_ghost() {
    use ferric_scf::gradient::rhf_gradient;

    let mol = Molecule::parse_xyz(HE_FAR_GHOST, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let config = RhfConfig { energy_conv: 1e-11, ..Default::default() };
    let ctx = ParallelContext::default();
    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

    let grad_result = rhf_gradient(&mol, &prep, op, &bounds, &result, None);
    assert!(
        grad_result.is_err(),
        "rhf_gradient should return Err for a molecule containing ghost atoms"
    );
    eprintln!("rhf_gradient ghost guard: {:?}", grad_result.err());
}
