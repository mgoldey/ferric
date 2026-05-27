//! cDFT weight-operator identities. These are exact (grid-tolerance) and need
//! no external reference: Σ_A w_A(r) = 1 implies Σ_A W^A = S, and the
//! per-atom populations of a density must sum to the electron count.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::cdft::{build_weight_matrix, population, SpinChannel};
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::overlap;
use ndarray::Array2;

fn lih() -> Molecule {
    // LiH ~1.60 Å along z. charge 0, singlet.
    Molecule::parse_xyz("2\nLiH\nLi 0 0 0\nH 0 0 1.60\n", 0, 1).unwrap()
}

#[test]
fn sum_of_atom_weights_equals_overlap() {
    let mol = lih();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let s = overlap(&prep);
    // Densest grid the Lebedev table supports (angular max = 302). The
    // partition-of-unity identity Σ_A w_A(r) = 1 is exact, so any residual here
    // is pure quadrature error. With diffuse def2-SVP AO *products* (no physical
    // density decay to help the quadrature) this plateaus at ~2e-5 even as the
    // radial grid is refined — it is the angular-quadrature floor of the 302-pt
    // Lebedev grid, not an error in the operator. (The operator's correctness is
    // pinned more tightly by atom_populations_sum_to_electron_count, which uses a
    // physical density and recovers N_e to ~1e-8 on this same grid.)
    let grid = build_atomic_grid(
        &mol,
        &AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
        },
    );
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&mol, &bs, &pts).unwrap();

    let n = s.nrows();
    let mut sum_w = Array2::<f64>::zeros((n, n));
    for a in 0..mol.atoms.len() {
        sum_w += &build_weight_matrix(&mol, &grid, &chi, &[a]);
    }
    let mut max_diff = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            max_diff = max_diff.max((sum_w[(i, j)] - s[(i, j)]).abs());
        }
    }
    // Quadrature floor of the 302-pt Lebedev grid for diffuse AO products.
    assert!(max_diff < 5e-5, "Σ_A W^A vs S max diff {max_diff:.3e}");
}

#[test]
fn atom_populations_sum_to_electron_count() {
    let mol = lih();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    // Use the 302-pt angular grid (Lebedev table max) so the quadrature is
    // converged; on a physical density this recovers N_e to ~1e-8.
    let grid = build_atomic_grid(
        &mol,
        &AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
        },
    );
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&mol, &bs, &pts).unwrap();

    // Build a real density via RHF so Tr is meaningful.
    let n = prep.nbasis();
    let op = ferric_integrals::operator::Operator::coulomb();
    let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
    let res = ferric_scf::rhf::solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &ferric_scf::rhf::RhfConfig::default(),
    )
    .unwrap();
    let d_total = res.density_total.clone(); // D_α + D_β
    let d_a = res.density_alpha.clone();
    let d_b = res.density_beta.clone().unwrap_or_else(|| d_a.clone());

    let mut n_sum = 0.0;
    for a in 0..mol.atoms.len() {
        let w = build_weight_matrix(&mol, &grid, &chi, &[a]);
        n_sum += population(&w, &d_a, &d_b, &SpinChannel::Total);
    }
    let nelec = mol.nelec() as f64;
    assert!((n_sum - nelec).abs() < 1e-6, "Σ pop {n_sum} vs nelec {nelec}");
    let _ = (n, d_total);
}
