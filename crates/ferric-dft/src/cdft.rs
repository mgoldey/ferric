//! Constrained-DFT weight operator and fragment populations.
//!
//! The constraint operator is a real-space-weighted overlap on the DFT
//! Becke–Lebedev grid:
//!
//!   W^C_μν = ∫ w_C(r) χ_μ(r) χ_ν(r) dr ≈ Σ_g weight_g · w_C(r_g) · χ_μg · χ_νg
//!
//! built with the same prescale+GEMM as the LDA Vxc (see vxc.rs). The fragment
//! weight is w_C(r) = Σ_{A∈C} w_A(r) with Becke fuzzy-cell weights. Because the
//! overlap is baked into W, a population is just N_C = Tr[W^C · D].

use crate::becke::becke_weights_all;
use crate::grid::GridPoint;
use ferric_core::mol::Molecule;
use ndarray::Array2;

/// Which population a constraint targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpinChannel {
    /// N_α + N_β (charge constraint).
    Total,
    /// N_α − N_β (spin constraint).
    SpinDiff,
}

/// One fragment population constraint.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Atom indices forming the fragment.
    pub fragment: Vec<usize>,
    pub spin: SpinChannel,
    /// Target population N_C^target (electrons).
    pub target: f64,
}

/// Build the cDFT weight operator W^C for `fragment` on the DFT grid.
///
/// `chi` is the (nbf, npts) AO-on-grid matrix (rows = AO index, cols = grid
/// point), as returned by `ao_grid::eval_basis_on_points`. `grid` must be the
/// same point set, in the same order, that `chi` was evaluated on.
pub fn build_weight_matrix(
    mol: &Molecule,
    grid: &[GridPoint],
    chi: &Array2<f64>,
    fragment: &[usize],
) -> Array2<f64> {
    let nbf = chi.nrows();
    let npts = chi.ncols();
    debug_assert_eq!(npts, grid.len());

    // Per-point fragment weight times quadrature weight: scale_g = w_g · w_C(r_g).
    let mut scale = vec![0.0_f64; npts];
    for (g, gp) in grid.iter().enumerate() {
        let w_atoms = becke_weights_all(mol, gp.xyz);
        let w_c: f64 = fragment.iter().map(|&a| w_atoms[a]).sum();
        scale[g] = gp.weight * w_c;
    }

    // Mirror vxc.rs: prescale chi columns, then chi_scaled · chiᵀ.
    let mut chi_scaled = chi.clone();
    for g in 0..npts {
        let s = scale[g];
        for mu in 0..nbf {
            chi_scaled[(mu, g)] *= s;
        }
    }
    let w: Array2<f64> = chi_scaled.dot(&chi.t());
    // Symmetrize (defends against grid asymmetry).
    0.5 * (&w + &w.t())
}

/// Fragment population N_C from a UHF/UKS density pair.
///
/// `Total` → Tr[W (D_α + D_β)], `SpinDiff` → Tr[W (D_α − D_β)].
pub fn population(
    w: &Array2<f64>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    spin: &SpinChannel,
) -> f64 {
    let n = w.nrows();
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..n {
            let d = match spin {
                SpinChannel::Total => d_a[(i, j)] + d_b[(i, j)],
                SpinChannel::SpinDiff => d_a[(i, j)] - d_b[(i, j)],
            };
            // Tr[W D] = Σ_ij W_ij D_ji ; W and D symmetric ⇒ Σ_ij W_ij D_ij.
            acc += w[(i, j)] * d;
        }
    }
    acc
}
