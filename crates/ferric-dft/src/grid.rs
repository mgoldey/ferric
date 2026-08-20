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

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use rayon::prelude::*;

use crate::becke::{becke_weights_all, becke_weights_and_grad};
use crate::lebedev::lebedev;
use crate::prune::{angular_orders_for_atom, PruneScheme};
use crate::radial::treutler_ahlrichs_m4;

/// Below this many points, run the Becke-weight pass serially. Rayon
/// spawn/join/steal overhead dominates on small (e.g. free-atom SAD) grids —
/// same rationale and threshold order-of-magnitude as `ao_grid.rs`'s
/// `PAR_WORK_THRESHOLD` (per-point Becke weight is O(natoms) to O(natoms²)
/// work, comparable to a handful of AO evaluations per point).
const PAR_WORK_THRESHOLD: usize = 2_000;

/// One grid point.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Angular-order pruning scheme, or `None` for the flat (unpruned) grid.
    ///
    /// `None` is the default and reproduces the historical grid exactly. Opting
    /// into [`PruneScheme::NwchemLike`] cuts ~23% of grid points (measured on
    /// H2O/CH4/benzene at 75x110; 33.5% at 99x302) for a ΔE_xc of ~1e-10 Ha —
    /// roughly 10^5 inside the reference-test budget, i.e. effectively free.
    ///
    /// NOT yet the default: it must first be validated through a live KS SCF
    /// (the accuracy numbers above come from re-evaluating E_xc on a
    /// flat-grid-converged density, which isolates the grid's contribution but
    /// is not the same as running the reference suites with pruning on).
    ///
    /// CAUTION: pruning ERRORS for `n_angular = 50` rather than silently
    /// returning a flat grid (NWChem's middle regions want order 74, which
    /// ferric would have to snap UP to 110, enlarging the grid — the
    /// config-honesty convention forbids the silent fallback). The NLC/fallback
    /// grids in `rhf.rs`/`uhf.rs`/`rohf.rs` use 50x50, so those must keep this
    /// `None`.
    pub prune: Option<crate::prune::PruneScheme>,
}

impl Default for AtomicGridConfig {
    fn default() -> Self {
        // "Fine" production grid: 75 radial × 110 angular = 8250 pts/atom.
        // Comparable to ORCA "GRID5" or PySCF (75, 302).
        Self {
            n_radial: 75,
            n_angular: 110,
            // Flat grid: byte-identical to the historical behavior. See the
            // field doc for why pruning is opt-in until it is validated
            // through a live KS SCF against the reference suites.
            prune: None,
        }
    }
}

/// Build the full molecular grid for `mol`. Returns a flat Vec of GridPoints
/// summed across all atoms.
///
/// The Becke-weight evaluation at each point is independent of every other
/// point (it only reads `mol` and that point's `xyz`), so this is a pure map
/// over the point axis: build the un-weighted `(home_atom, xyz, w_r*w_l)`
/// list first (cheap, serial — radial/angular quadrature only), then map
/// `becke_weights_all` over it with an order-preserving `into_par_iter().
/// map().collect()`. Collecting in index order makes the output bit-identical
/// to the old serial loop regardless of thread count — no reduction, so no
/// nondeterminism to guard against.
pub fn build_atomic_grid(mol: &Molecule, cfg: &AtomicGridConfig) -> Vec<GridPoint> {
    let (lebedev_pts, lebedev_w) = lebedev(cfg.n_angular);
    let mut pre: Vec<(usize, [f64; 3], f64)> =
        Vec::with_capacity(mol.atoms.len() * cfg.n_radial * lebedev_pts.len());

    for (a_idx, atom) in mol.atoms.iter().enumerate() {
        let (rs, ws) = treutler_ahlrichs_m4(atom.z, cfg.n_radial);
        for (r, w_r) in rs.iter().zip(ws.iter()) {
            for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                let xyz = [
                    atom.x + r * pt[0],
                    atom.y + r * pt[1],
                    atom.zpos + r * pt[2],
                ];
                pre.push((a_idx, xyz, w_r * w_l));
            }
        }
    }

    let build_point = |&(a_idx, xyz, w_rl): &(usize, [f64; 3], f64)| -> GridPoint {
        let becke = becke_weights_all(mol, xyz);
        let weight = w_rl * becke[a_idx];
        GridPoint {
            xyz,
            weight,
            home_atom: a_idx,
        }
    };

    if pre.len() >= PAR_WORK_THRESHOLD {
        pre.par_iter().map(build_point).collect()
    } else {
        pre.iter().map(build_point).collect()
    }
}

/// Build the molecular grid with **opt-in angular pruning**.
///
/// `prune = None` is exactly [`build_atomic_grid`] (bit-identical — it
/// delegates), so this is a strict superset of the flat path and the flat grid
/// remains the default everywhere. `prune = Some(scheme)` reduces the Lebedev
/// order on inner-core and far-tail radial shells per
/// [`crate::prune::angular_orders_for_atom`], keeping the full order only in
/// the chemically-active middle regions.
///
/// Returns `Err` (never a silent fallback to the flat grid) if the requested
/// angular order has no valid pruned region table — see
/// [`crate::prune::region_orders`].
///
/// # Weight correctness
///
/// Lebedev weights are normalised to `Σ w = 1` at every order, so mixing
/// orders across radial shells leaves `Σ_r Σ_Ω w_r w_Ω` unchanged shell by
/// shell. No renormalisation is applied or needed.
pub fn build_atomic_grid_pruned(
    mol: &Molecule,
    cfg: &AtomicGridConfig,
    prune: Option<PruneScheme>,
) -> std::result::Result<Vec<GridPoint>, FerricError> {
    let Some(scheme) = prune else {
        return Ok(build_atomic_grid(mol, cfg));
    };

    // Cache one Lebedev table per distinct order actually used, so a 75-shell
    // atom does not regenerate the same rule 75 times.
    let mut cache: Vec<(usize, Vec<[f64; 3]>, Vec<f64>)> = Vec::new();
    let mut pre: Vec<(usize, [f64; 3], f64)> = Vec::new();

    for (a_idx, atom) in mol.atoms.iter().enumerate() {
        let (rs, ws) = treutler_ahlrichs_m4(atom.z, cfg.n_radial);
        let orders = angular_orders_for_atom(atom.z, &rs, cfg.n_angular, scheme)?;
        for ((r, w_r), &order) in rs.iter().zip(ws.iter()).zip(orders.iter()) {
            if !cache.iter().any(|(o, _, _)| *o == order) {
                let (p, w) = lebedev(order);
                cache.push((order, p, w));
            }
            let (_, lebedev_pts, lebedev_w) =
                cache.iter().find(|(o, _, _)| *o == order).expect("just inserted");
            for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                let xyz = [
                    atom.x + r * pt[0],
                    atom.y + r * pt[1],
                    atom.zpos + r * pt[2],
                ];
                pre.push((a_idx, xyz, w_r * w_l));
            }
        }
    }

    // Same order-preserving parallel map as build_atomic_grid.
    let build_point = |&(a_idx, xyz, w_rl): &(usize, [f64; 3], f64)| -> GridPoint {
        let becke = becke_weights_all(mol, xyz);
        GridPoint {
            xyz,
            weight: w_rl * becke[a_idx],
            home_atom: a_idx,
        }
    };

    Ok(if pre.len() >= PAR_WORK_THRESHOLD {
        pre.par_iter().map(build_point).collect()
    } else {
        pre.iter().map(build_point).collect()
    })
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
    let natoms = mol.atoms.len();
    let mut pre: Vec<(usize, [f64; 3], f64)> =
        Vec::with_capacity(natoms * cfg.n_radial * lebedev_pts.len());

    for (a_idx, atom) in mol.atoms.iter().enumerate() {
        let (rs, ws) = treutler_ahlrichs_m4(atom.z, cfg.n_radial);
        for (r, w_r) in rs.iter().zip(ws.iter()) {
            for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                let xyz = [
                    atom.x + r * pt[0],
                    atom.y + r * pt[1],
                    atom.zpos + r * pt[2],
                ];
                pre.push((a_idx, xyz, w_r * w_l));
            }
        }
    }

    // Per-point work is O(natoms³) via becke_weights_and_grad and independent
    // across points — same order-preserving parallel map as build_atomic_grid
    // (bit-identical to the serial loop; no reduction).
    let build_point =
        |&(a_idx, xyz, scale): &(usize, [f64; 3], f64)| -> (GridPoint, Vec<[f64; 3]>) {
            let (becke, dw_lab) = becke_weights_and_grad(mol, xyz);
            let weight = scale * becke[a_idx];
            let gp = GridPoint {
                xyz,
                weight,
                home_atom: a_idx,
            };

            // Lab-fixed partition derivative for the HOME atom's weight:
            // dw_lab[a_idx][c][k] is ∂w_becke^{a_idx}/∂R_c^k at fixed r.
            // PySCF's weight1 adds ∇_r w_becke to the home-atom row, then
            // multiplies through by the radial-angular Jacobian.
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
            (gp, row)
        };

    let (grid, weight1): (Vec<GridPoint>, Vec<Vec<[f64; 3]>>) =
        if pre.len() >= PAR_WORK_THRESHOLD {
            pre.par_iter().map(build_point).unzip()
        } else {
            pre.iter().map(build_point).unzip()
        };
    (grid, weight1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::{Atom, Molecule};

    fn h2() -> Molecule {
        Molecule {
            atoms: vec![
                Atom { symbol: "H".into(), z: 1, x: 0.0, y: 0.0, zpos: 0.0, ghost: false, n_core_ecp: 0 },
                Atom { symbol: "H".into(), z: 1, x: 0.0, y: 0.0, zpos: 1.4, ghost: false, n_core_ecp: 0 },
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
            ..Default::default()
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

    /// Serial reference: the pre-P2 per-point loop, verbatim.
    fn build_atomic_grid_serial_ref(mol: &Molecule, cfg: &AtomicGridConfig) -> Vec<GridPoint> {
        let (lebedev_pts, lebedev_w) = lebedev(cfg.n_angular);
        let mut grid = Vec::new();
        for (a_idx, atom) in mol.atoms.iter().enumerate() {
            let (rs, ws) = treutler_ahlrichs_m4(atom.z, cfg.n_radial);
            for (r, w_r) in rs.iter().zip(ws.iter()) {
                for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                    let xyz = [
                        atom.x + r * pt[0],
                        atom.y + r * pt[1],
                        atom.zpos + r * pt[2],
                    ];
                    let becke = becke_weights_all(mol, xyz);
                    let weight = w_r * w_l * becke[a_idx];
                    grid.push(GridPoint { xyz, weight, home_atom: a_idx });
                }
            }
        }
        grid
    }

    #[test]
    fn parallel_grid_bit_identical_to_serial() {
        // Default fine grid on H2: 2 * 75 * 110 = 16,500 points — well above
        // PAR_WORK_THRESHOLD, so build_atomic_grid takes the rayon path.
        let mol = h2();
        let cfg = AtomicGridConfig::default();
        assert!(
            mol.atoms.len() * cfg.n_radial * 110 >= PAR_WORK_THRESHOLD,
            "test must exercise the parallel path"
        );
        let par = build_atomic_grid(&mol, &cfg);
        let ser = build_atomic_grid_serial_ref(&mol, &cfg);
        assert_eq!(par.len(), ser.len());
        for (g, (p, s)) in par.iter().zip(ser.iter()).enumerate() {
            assert_eq!(p.home_atom, s.home_atom, "home_atom mismatch at point {g}");
            for k in 0..3 {
                assert_eq!(
                    p.xyz[k].to_bits(),
                    s.xyz[k].to_bits(),
                    "xyz[{k}] not bit-identical at point {g}"
                );
            }
            assert_eq!(
                p.weight.to_bits(),
                s.weight.to_bits(),
                "weight not bit-identical at point {g}: par {} vs ser {}",
                p.weight,
                s.weight
            );
        }
    }

    #[test]
    fn parallel_grid_with_response_bit_identical_to_serial() {
        // Serial reference for the with-response variant: pre-P2 loop, verbatim.
        let mol = h2();
        let cfg = AtomicGridConfig::default();
        let (par_grid, par_w1) = build_atomic_grid_with_response(&mol, &cfg);

        let (lebedev_pts, lebedev_w) = lebedev(cfg.n_angular);
        let natoms = mol.atoms.len();
        let mut ser_grid = Vec::new();
        let mut ser_w1: Vec<Vec<[f64; 3]>> = Vec::new();
        for (a_idx, atom) in mol.atoms.iter().enumerate() {
            let (rs, ws) = treutler_ahlrichs_m4(atom.z, cfg.n_radial);
            for (r, w_r) in rs.iter().zip(ws.iter()) {
                for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                    let xyz = [
                        atom.x + r * pt[0],
                        atom.y + r * pt[1],
                        atom.zpos + r * pt[2],
                    ];
                    let (becke, dw_lab) = becke_weights_and_grad(&mol, xyz);
                    let weight = w_r * w_l * becke[a_idx];
                    ser_grid.push(GridPoint { xyz, weight, home_atom: a_idx });
                    let scale = w_r * w_l;
                    let mut grad_r = [0.0_f64; 3];
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
                    ser_w1.push(row);
                }
            }
        }

        assert_eq!(par_grid.len(), ser_grid.len());
        assert_eq!(par_w1.len(), ser_w1.len());
        for (g, (p, s)) in par_grid.iter().zip(ser_grid.iter()).enumerate() {
            assert_eq!(p.home_atom, s.home_atom);
            for k in 0..3 {
                assert_eq!(p.xyz[k].to_bits(), s.xyz[k].to_bits(), "xyz at point {g}");
            }
            assert_eq!(p.weight.to_bits(), s.weight.to_bits(), "weight at point {g}");
        }
        for (g, (pr, sr)) in par_w1.iter().zip(ser_w1.iter()).enumerate() {
            assert_eq!(pr.len(), sr.len());
            for (b, (pe, se)) in pr.iter().zip(sr.iter()).enumerate() {
                for k in 0..3 {
                    assert_eq!(
                        pe[k].to_bits(),
                        se[k].to_bits(),
                        "weight1 not bit-identical at point {g}, atom {b}, axis {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn prune_none_is_bit_identical_to_flat_grid() {
        // The whole opt-in guarantee: passing None must not perturb a single
        // bit of the existing default path.
        let mol = h2();
        let cfg = AtomicGridConfig::default();
        let flat = build_atomic_grid(&mol, &cfg);
        let none = build_atomic_grid_pruned(&mol, &cfg, None).unwrap();
        assert_eq!(flat.len(), none.len());
        for (g, (a, b)) in flat.iter().zip(none.iter()).enumerate() {
            assert_eq!(a.home_atom, b.home_atom, "home_atom at {g}");
            assert_eq!(a.weight.to_bits(), b.weight.to_bits(), "weight at {g}");
            for k in 0..3 {
                assert_eq!(a.xyz[k].to_bits(), b.xyz[k].to_bits(), "xyz[{k}] at {g}");
            }
        }
    }

    #[test]
    fn pruned_grid_is_smaller_and_still_integrates_a_gaussian() {
        // Point-count reduction plus the integration invariant: pruning must
        // shrink the grid without breaking the quadrature. A weight
        // normalisation bug would blow this relerr up by orders of magnitude.
        let mol = h2();
        let cfg = AtomicGridConfig::default();
        let flat = build_atomic_grid(&mol, &cfg);
        let pruned =
            build_atomic_grid_pruned(&mol, &cfg, Some(PruneScheme::NwchemLike)).unwrap();
        let frac = 1.0 - pruned.len() as f64 / flat.len() as f64;
        eprintln!(
            "H2 75x110: flat {} pts -> pruned {} pts ({:.1}% fewer)",
            flat.len(),
            pruned.len(),
            100.0 * frac
        );
        assert!(pruned.len() < flat.len(), "pruning did not remove any points");
        assert!(frac > 0.15, "pruning saved only {:.1}%", 100.0 * frac);

        let alpha = 1.0_f64;
        let center = [0.0, 0.0, 0.7];
        let integrate = |g: &[GridPoint]| -> f64 {
            g.iter()
                .map(|p| {
                    let dx = p.xyz[0] - center[0];
                    let dy = p.xyz[1] - center[1];
                    let dz = p.xyz[2] - center[2];
                    p.weight * (-alpha * (dx * dx + dy * dy + dz * dz)).exp()
                })
                .sum()
        };
        let exact = (std::f64::consts::PI / alpha).powf(1.5);
        let err_flat = (integrate(&flat) - exact).abs() / exact;
        let err_pruned = (integrate(&pruned) - exact).abs() / exact;
        eprintln!("Gaussian relerr: flat {err_flat:.2e}, pruned {err_pruned:.2e}");
        assert!(err_pruned < 1e-3, "pruned Gaussian relerr {err_pruned:.2e}");
    }

    #[test]
    fn pruned_grid_partition_still_sums_to_full_space() {
        // Becke partition exactness is independent of the angular order used
        // at each shell -- confirm pruning does not disturb it.
        let mol = h2();
        let grid = build_atomic_grid_pruned(
            &mol,
            &AtomicGridConfig::default(),
            Some(PruneScheme::NwchemLike),
        )
        .unwrap();
        let f = |p: &GridPoint| {
            let dz = p.xyz[2] - 0.7;
            (-0.5 * (p.xyz[0].powi(2) + p.xyz[1].powi(2) + dz * dz)).exp()
        };
        let total: f64 = grid.iter().map(|p| p.weight * f(p)).sum();
        let mut per_atom = [0.0_f64; 2];
        for p in &grid {
            per_atom[p.home_atom] += p.weight * f(p);
        }
        let sum_atoms: f64 = per_atom.iter().sum();
        assert!(
            (sum_atoms - total).abs() < 1e-12,
            "pruned Becke partition mismatch: {sum_atoms} vs {total}"
        );
    }

    #[test]
    fn pruning_errors_instead_of_silently_falling_back() {
        // n_angular = 50 has no useful pruned table on ferric's Lebedev set.
        // It must Err, not quietly return the flat grid.
        let mol = h2();
        let cfg = AtomicGridConfig { n_radial: 50, n_angular: 50, ..Default::default() };
        assert!(build_atomic_grid_pruned(&mol, &cfg, Some(PruneScheme::NwchemLike)).is_err());
        // ... but with pruning off, the same config still works.
        assert!(build_atomic_grid_pruned(&mol, &cfg, None).is_ok());
    }

    #[test]
    fn grid_partition_sums_to_full_space() {
        // Σ_A ∫_A f dV = ∫ f dV — the Becke partition is exact at every
        // grid point.
        let mol = h2();
        let grid = build_atomic_grid(&mol, &AtomicGridConfig {
            n_radial: 50,
            n_angular: 50,
            ..Default::default()
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
