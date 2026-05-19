use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_and_grad_on_points;
use ferric_dft::density_on_grid::eval_density_closed;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[test]
fn integral_of_rho_equals_nelec_h2o() {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n", 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
    let pts: Vec<[f64;3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();

    let dens = eval_density_closed(&rhf.density_total, &chi, &dchi);

    let n_e: f64 = grid.iter().zip(dens.rho.iter()).map(|(g, &r)| g.weight * r).sum();
    let err = (n_e - 10.0).abs();
    eprintln!("∫ρ dV = {n_e:.6} (expected 10.0), err = {err:.3e}");
    // (75, 110) is the "fine" production grid but ∫ρ on water at cc-pVDZ is
    // typically ~1e-3 accurate; 1e-2 is the safe assertion bound.
    assert!(err < 1e-2, "electron count off: {err:.3e}");

    // Check that σ is consistent with ∇ρ component-wise
    for g in 0..dens.rho.len() {
        let gx = dens.grad[(0, g)]; let gy = dens.grad[(1, g)]; let gz = dens.grad[(2, g)];
        let s = gx*gx + gy*gy + gz*gz;
        assert!((dens.sigma[g] - s).abs() < 1e-12 * (1.0 + s.abs()));
    }
}

#[test]
fn rho_is_nonnegative_h2o() {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n", 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
    let pts: Vec<[f64;3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
    let dens = eval_density_closed(&rhf.density_total, &chi, &dchi);

    // Variational ρ from a converged SCF must be non-negative everywhere
    // (numerically: allow tiny negatives at grid extremes within the
    // truncation error of finite basis).
    let min_rho = dens.rho.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(min_rho > -1e-10, "rho went substantially negative: min = {min_rho:.3e}");
}
