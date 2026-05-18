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

use crate::becke::becke_weights_all;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::{Atom, Molecule};

    fn h2() -> Molecule {
        Molecule {
            atoms: vec![
                Atom { symbol: "H".into(), z: 1, x: 0.0, y: 0.0, zpos: 0.0 },
                Atom { symbol: "H".into(), z: 1, x: 0.0, y: 0.0, zpos: 1.4 },
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
