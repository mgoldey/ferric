//! A forced (closed-shell) UHF turns the β HOMO into the LUMO by 0.1 rad to
//! break spin symmetry. When the occupied orbitals fill every function the
//! overlap retains, the LUMO column is a zero pad, and the rotation would
//! only scale the HOMO by cos(0.1): the first β density would hold
//! cos²(0.1) = 0.990 electrons instead of 1.
//!
//! He with a ghost He on the same centre, STO-3G: two identical 1s
//! functions, so the overlap has rank 1 and the one α and one β electron
//! fill the retained space. The hook sees every density the solver builds;
//! the first β density must hold exactly one electron.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::overlap;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf_with_hook;
use ndarray::Array2;

#[test]
fn forced_uhf_first_beta_density_keeps_its_electron_at_full_retained_rank() {
    let mol = Molecule::parse_xyz("2\nHe + ghost\nHe 0 0 0\n@He 0 0 0\n", 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &bundled("sto-3g").unwrap()).unwrap();
    let s = overlap(&prep);
    assert_eq!(s.nrows(), 2);
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = RhfConfig {
        check_stability: false,
        scf_stability_descent: false,
        ..Default::default()
    };
    let mut electrons_b: Vec<f64> = Vec::new();
    let mut hook = |_d_a: &Array2<f64>, d_b: &Array2<f64>| {
        electrons_b.push((d_b * &s).sum());
        let n = d_b.nrows();
        Ok((0.0, Array2::zeros((n, n)), Array2::zeros((n, n))))
    };
    let r = solve_uhf_with_hook(
        &ParallelContext::default(),
        &mol,
        &prep,
        &bounds,
        &cfg,
        &mut hook,
    )
    .expect("UHF converges");
    assert!(r.converged);
    // tr(D_β S) of an S-orthonormal one-orbital density is 1 to round-off;
    // the zero-pad rotation would give cos²(0.1) = 0.990, 1e-2 away.
    let first = electrons_b[0];
    assert!(
        (first - 1.0).abs() < 1e-12,
        "first beta density holds {first} electrons (all: {electrons_b:?})"
    );
}
