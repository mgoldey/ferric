//! ROHF on a basis whose overlap is singular: a ghost atom on the same
//! centre duplicates every function of the real atom, so half of the
//! overlap eigenvalues are zero to round-off. The solve must drop those
//! modes (retained-rank orthogonalizer) rather than scale them by
//! `1/sqrt(λ)`, and it spans exactly the space of the deduplicated basis,
//! so its energy is the deduplicated ROHF energy.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::overlap;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ndarray_linalg::{Eigh, UPLO};

const ENERGY_CONV: f64 = 1e-11;

fn rohf_energy(xyz: &str, mult: usize, basis: &str) -> (f64, usize, f64) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &bundled(basis).unwrap()).unwrap();
    let s = overlap(&prep);
    let (evals, _) = s.eigh(UPLO::Upper).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: ENERGY_CONV,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    let r = solve_rohf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("ROHF failed: {e:?}"));
    assert!(r.converged);
    (r.energy, s.nrows(), evals[0])
}

#[test]
fn rohf_with_a_duplicated_basis_matches_the_deduplicated_solve() {
    // (label, real-only xyz, real + ghost xyz, multiplicity)
    let cases = [
        (
            "Li",
            "1\nLi\nLi 0 0 0\n",
            "2\nLi + ghost\nLi 0 0 0\n@Li 0 0 0\n",
            2,
        ),
        (
            "OH",
            "2\nOH\nO 0 0 0\nH 0 0 0.97\n",
            "3\nOH + ghost O\nO 0 0 0\nH 0 0 0.97\n@O 0 0 0\n",
            2,
        ),
    ];
    for (label, plain, ghost, mult) in cases {
        let (e_ref, n_ref, l_ref) = rohf_energy(plain, mult, "6-31g");
        let (e_dup, n_dup, l_dup) = rohf_energy(ghost, mult, "6-31g");
        assert!(n_dup > n_ref && l_ref > 1e-3, "{label}: setup");
        assert!(
            l_dup.abs() < 1e-12,
            "{label}: smallest S eigenvalue {l_dup:e}"
        );
        assert!(e_dup.is_finite(), "{label}: energy {e_dup}");
        let de = (e_dup - e_ref).abs();
        // Both solves span the same function space and stop at the same
        // criteria; each energy sits within energy_conv of its own
        // stationary point (density error enters at second order).
        assert!(de < 2.0 * ENERGY_CONV, "{label}: |E_dup - E_ref| = {de:e}");
    }
}
