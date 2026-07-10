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
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ndarray::Array2;
use rayon::prelude::*;

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
    // Each point's Becke evaluation is independent; order-preserving parallel
    // map (bit-identical to the serial loop). Serial below the same
    // point-count threshold as grid.rs to keep tiny grids rayon-free.
    const PAR_WORK_THRESHOLD: usize = 2_000;
    let scale_for = |gp: &GridPoint| -> f64 {
        let w_atoms = becke_weights_all(mol, gp.xyz);
        let w_c: f64 = fragment.iter().map(|&a| w_atoms[a]).sum();
        gp.weight * w_c
    };
    let scale: Vec<f64> = if npts >= PAR_WORK_THRESHOLD {
        grid.par_iter().map(scale_for).collect()
    } else {
        grid.iter().map(scale_for).collect()
    };

    // Mirror vxc.rs: prescale chi columns, then chi_scaled · chiᵀ.
    let mut chi_scaled = chi.clone();
    for g in 0..npts {
        let s = scale[g];
        for mu in 0..nbf {
            chi_scaled[(mu, g)] *= s;
        }
    }
    // Digestion GEMM, after the rayon-gated scale_for map above has already
    // collected. Opt-in BLAS raise via FERRIC_BLAS_THREADS (default 1,
    // unchanged behavior); mirrors vxc.rs's semilocal_vxc_closed idiom.
    let w: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || chi_scaled.dot(&chi.t()));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{build_atomic_grid, AtomicGridConfig};
    use ferric_core::mol::{Atom, Molecule};

    #[test]
    fn parallel_weight_matrix_bit_identical_to_serial() {
        // 2 atoms * 25 radial * 50 angular = 2,500 points — above the 2,000
        // point PAR_WORK_THRESHOLD, so build_weight_matrix takes the rayon path.
        let mol = Molecule {
            atoms: vec![
                Atom { symbol: "H".into(), z: 1, x: 0.0, y: 0.0, zpos: 0.0, ghost: false, n_core_ecp: 0 },
                Atom { symbol: "H".into(), z: 1, x: 0.0, y: 0.0, zpos: 1.4, ghost: false, n_core_ecp: 0 },
            ],
            charge: 0,
            multiplicity: 1,
        };
        let grid = build_atomic_grid(&mol, &AtomicGridConfig { n_radial: 25, n_angular: 50 });
        let npts = grid.len();
        assert!(npts >= 2_000, "test must exercise the parallel path (npts={npts})");

        // Deterministic synthetic AO values (nbf = 2).
        let nbf = 2;
        let chi = Array2::from_shape_fn((nbf, npts), |(mu, g)| {
            0.1 * (mu as f64 + 1.0) + (0.001 * g as f64).sin()
        });
        let fragment = [0usize];

        let w_par = build_weight_matrix(&mol, &grid, &chi, &fragment);

        // Serial reference: pre-P2 per-point loop, verbatim, then the same
        // prescale + GEMM tail.
        let mut scale = vec![0.0_f64; npts];
        for (g, gp) in grid.iter().enumerate() {
            let w_atoms = becke_weights_all(&mol, gp.xyz);
            let w_c: f64 = fragment.iter().map(|&a| w_atoms[a]).sum();
            scale[g] = gp.weight * w_c;
        }
        let mut chi_scaled = chi.clone();
        for g in 0..npts {
            let s = scale[g];
            for mu in 0..nbf {
                chi_scaled[(mu, g)] *= s;
            }
        }
        let w: Array2<f64> = chi_scaled.dot(&chi.t());
        let w_ser = 0.5 * (&w + &w.t());

        for i in 0..nbf {
            for j in 0..nbf {
                assert_eq!(
                    w_par[(i, j)].to_bits(),
                    w_ser[(i, j)].to_bits(),
                    "W[{i},{j}] not bit-identical: par {} vs ser {}",
                    w_par[(i, j)],
                    w_ser[(i, j)]
                );
            }
        }
    }
}
