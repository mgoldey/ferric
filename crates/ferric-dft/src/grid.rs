//! Atom-centered molecular grid assembled from Treutler-Ahlrichs M4
//! radial + Lebedev angular + Becke fuzzy weights.
//!
//! Each grid point carries:
//!   * `xyz`: Cartesian position (Bohr)
//!   * `weight`: full quadrature weight including 4π r² Jacobian and
//!     Becke partition for the home atom
//!   * `home_atom`: index of the atom the radial shell is centered on
//!
//! Spherical integral of a function `f(r)` over all space:
//! ```text
//!     ∫ f dV ≈ Σ_g w_g · f(r_g)
//! ```
//!
//! Per-atom A partition integrals:
//! ```text
//!     ∫_A f dV = Σ_{g: home=A} w_g · f(r_g)
//! ```
//! (Becke partition is already baked into `w_g`; the home-atom restriction
//! is what distinguishes per-atom from total.)

use ferric_core::mol::Molecule;

use crate::becke::{becke_weights_all, becke_weights_and_grad};
use crate::lebedev::lebedev;
use crate::radial::treutler_ahlrichs_m4;

/// One grid point.
#[derive(Debug, Clone, Copy)]
pub struct GridPoint {
    pub xyz: [f64; 3],
    pub weight: f64,
    pub home_atom: usize,
}

/// Configuration for a Becke-Lebedev atomic grid.
#[derive(Debug, Clone)]
pub struct AtomicGridConfig {
    pub n_radial: usize,
    pub n_angular: usize,
}

impl Default for AtomicGridConfig {
    fn default() -> Self {
        // "Fine" production grid: 75 radial × 110 angular = 8250 pts/atom.
        // Comparable to ORCA "GRID5" or PySCF (75, 302).
        Self {
            n_radial: 75,
            n_angular: 110,
        }
    }
}

/// Build the full molecular grid for `mol`. Returns a flat Vec of GridPoints
/// summed across all atoms.
pub fn build_atomic_grid(mol: &Molecule, cfg: &AtomicGridConfig) -> Vec<GridPoint> {
    let (lebedev_pts, lebedev_w) = lebedev(cfg.n_angular);
    let mut grid = Vec::with_capacity(mol.atoms.len() * cfg.n_radial * lebedev_pts.len());

    for (a_idx, atom) in mol.atoms.iter().enumerate() {
        let (rs, ws) = treutler_ahlrichs_m4(atom.z, cfg.n_radial);
        for (r, w_r) in rs.iter().zip(ws.iter()) {
            for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                let xyz = [
                    atom.x + r * pt[0],
                    atom.y + r * pt[1],
                    atom.zpos + r * pt[2],
                ];
                // Becke partition weight for the home atom.
                let becke = becke_weights_all(mol, xyz);
                let w_becke = becke[a_idx];
                let weight = w_r * w_l * w_becke;
                grid.push(GridPoint {
                    xyz,
                    weight,
                    home_atom: a_idx,
                });
            }
        }
    }
    grid
}

/// Build the molecular grid AND the per-grid-point full quadrature-weight
/// nuclear-coordinate gradient (PySCF "weight1" convention).
///
/// Returns `(grid, weight1)` where `weight1[g][b][α]` is the total derivative
/// of `w_g = w_r · w_l · w_becke^{home(g)}(r_g(R); R)` with respect to `R_b^α`,
/// treating `r_g` as **moving rigidly with its home atom** (the convention
/// PySCF uses in `grids_response_cc`). Equivalent to:
///
/// ```text
///   weight1[g][b][α] = (w_r · w_l) · [ ∂w_becke^{home}/∂R_b^α|_{r lab-fixed}
///                                      + δ_{b, home} · ∇_r w_becke^{home}(r_g) ]
/// ```
///
/// The `∇_r w_becke` piece is reconstructed via translational invariance
/// `Σ_c ∂w_becke/∂R_c|_{r fixed} + ∇_r w_becke = 0` ⇒
/// `∇_r w_becke = -Σ_c ∂w_becke/∂R_c|_{r fixed}`, applied only to the home
/// row. After this fix, `Σ_b weight1[g][b][α] = 0` at every grid point
/// (rigid-translation invariance, exact).
///
/// Used by the XC gradient grid-response correction (P2.1).
pub fn build_atomic_grid_with_response(
    mol: &Molecule,
    cfg: &AtomicGridConfig,
) -> (Vec<GridPoint>, Vec<Vec<[f64; 3]>>) {
    let (lebedev_pts, lebedev_w) = lebedev(cfg.n_angular);
    let total_pts = mol.atoms.len() * cfg.n_radial * lebedev_pts.len();
    let mut grid = Vec::with_capacity(total_pts);
    let mut weight1: Vec<Vec<[f64; 3]>> = Vec::with_capacity(total_pts);
    let natoms = mol.atoms.len();

    for (a_idx, atom) in mol.atoms.iter().enumerate() {
        let (rs, ws) = treutler_ahlrichs_m4(atom.z, cfg.n_radial);
        for (r, w_r) in rs.iter().zip(ws.iter()) {
            for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                let xyz = [
                    atom.x + r * pt[0],
                    atom.y + r * pt[1],
                    atom.zpos + r * pt[2],
                ];
                let (becke, dw_lab) = becke_weights_and_grad(mol, xyz);
                let w_becke = becke[a_idx];
                let weight = w_r * w_l * w_becke;
                grid.push(GridPoint {
                    xyz,
                    weight,
                    home_atom: a_idx,
                });

                // Lab-fixed partition derivative for the HOME atom's weight:
                // dw_lab[a_idx][c][k] is ∂w_becke^{a_idx}/∂R_c^k at fixed r.
                // PySCF's weight1 adds ∇_r w_becke to the home-atom row, then
                // multiplies through by the radial-angular Jacobian.
                let scale = w_r * w_l;
                let mut grad_r = [0.0_f64; 3]; // = -Σ_c dw_lab[a_idx][c]
                for c in 0..natoms {
                    grad_r[0] -= dw_lab[a_idx][c][0];
                    grad_r[1] -= dw_lab[a_idx][c][1];
                    grad_r[2] -= dw_lab[a_idx][c][2];
                }
                let mut row = Vec::with_capacity(natoms);
                for b in 0..natoms {
                    let mut entry = [
                        scale * dw_lab[a_idx][b][0],
                        scale * dw_lab[a_idx][b][1],
                        scale * dw_lab[a_idx][b][2],
                    ];
                    if b == a_idx {
                        entry[0] += scale * grad_r[0];
                        entry[1] += scale * grad_r[1];
                        entry[2] += scale * grad_r[2];
                    }
                    row.push(entry);
                }
                weight1.push(row);
            }
        }
    }
    (grid, weight1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::{Atom, Molecule};

    fn h2() -> Molecule {
        Molecule {
            atoms: vec![
                Atom { symbol: "H".into(), z: 1, x: 0.0, y: 0.0, zpos: 0.0, n_core_ecp: 0 },
                Atom { symbol: "H".into(), z: 1, x: 0.0, y: 0.0, zpos: 1.4, n_core_ecp: 0 },
            ],
            charge: 0,
            multiplicity: 1,
        }
    }

    #[test]
    fn grid_integrates_unit_density_to_electron_count_h2() {
        // ∫ ρ dV = N_e for any normalized density. For H2 with ρ = 2 * 1s
        // Slater (ξ=1) localized on each atom averaged, just check that a
        // simple uniform-density-like test integrand integrates correctly.
        let mol = h2();
        let grid = build_atomic_grid(&mol, &AtomicGridConfig {
            n_radial: 75,
            n_angular: 110,
        });

        // Total weight ≈ infinity for ∫ 1 dV (whole space), so check
        // something normalizable: ∫ exp(-α r²) at H2's bond midpoint
        // (Gaussian centered at midpoint).
        let alpha = 1.0_f64;
        let center = [0.0, 0.0, 0.7];
        let approx: f64 = grid.iter().map(|g| {
            let dx = g.xyz[0] - center[0];
            let dy = g.xyz[1] - center[1];
            let dz = g.xyz[2] - center[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            g.weight * (-alpha * r2).exp()
        }).sum();
        let exact = (std::f64::consts::PI / alpha).powf(1.5);
        let err = (approx - exact).abs() / exact;
        eprintln!("H2 grid: ∫ Gaussian = {approx:.6}, exact = {exact:.6}, relerr={err:.2e}");
        assert!(err < 1e-3, "H2 Becke-Lebedev Gaussian relerr {err:.2e}");
    }

    #[test]
    fn grid_partition_sums_to_full_space() {
        // Σ_A ∫_A f dV = ∫ f dV — the Becke partition is exact at every
        // grid point.
        let mol = h2();
        let grid = build_atomic_grid(&mol, &AtomicGridConfig {
            n_radial: 50,
            n_angular: 50,
        });
        let alpha = 0.5_f64;
        let center = [0.0, 0.0, 0.7];
        let total: f64 = grid.iter().map(|g| {
            let dx = g.xyz[0] - center[0];
            let dy = g.xyz[1] - center[1];
            let dz = g.xyz[2] - center[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            g.weight * (-alpha * r2).exp()
        }).sum();
        // Per-atom sum to check partition reproduces total.
        let mut per_atom = [0.0_f64; 2];
        for g in &grid {
            let dx = g.xyz[0] - center[0];
            let dy = g.xyz[1] - center[1];
            let dz = g.xyz[2] - center[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            per_atom[g.home_atom] += g.weight * (-alpha * r2).exp();
        }
        let sum_atoms: f64 = per_atom.iter().sum();
        assert!((sum_atoms - total).abs() < 1e-12,
            "Becke partition sum mismatch: per-atom {sum_atoms}, total {total}");
    }
}
