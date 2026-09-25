//! Closed-shell RI-RPA nuclear gradient by central finite differences.
//!
//! The gradient is NOT analytic. For every Cartesian coordinate it displaces
//! the atom by ±h, solves a fresh exact-J/K RHF (`density_conv` 1e-9) and a
//! full `run_pdep_rpa` at each point, and differences the TOTAL energy
//! E_RHF + E_c^RPA: g = (E(+h) − E(−h)) / 2h. Orbital response and any
//! geometry dependence of the PDEP basis are therefore included exactly, at
//! the cost of 6·N_atoms full RHF + RPA calculations per gradient, with the
//! O(h²) truncation error of a 3-point stencil (at h = 5e-4 Bohr on distorted
//! H2O/NH3 cc-pVDZ that is ≤ 5.6e-8 Ha/Bohr against a converged 5-point FD;
//! see tests/validation_rpa_gradient.rs).
//!
//! An analytic path (3-centre ERI derivatives + Z-vector orbital response,
//! the same structure as the RI-MP2 gradient) does not exist.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

use crate::config::Chi0Backend;
use crate::{run_pdep_rpa, Chi0Sparsity, PdepRpaConfig, PdepRpaResult};

/// The closed-shell RI-RPA nuclear gradient at the reference geometry, by a
/// 3-point central finite difference of the TOTAL energy E_RHF + E_c^RPA.
///
/// Returns a `(n_atoms, 3)` array of `dE_total / dR` in Hartree/Bohr — the
/// full RPA gradient, NOT the correlation part alone (despite the name). Do
/// not add `rhf_gradient` to it: that counts the RHF gradient twice.
///
/// # Restrictions
///
/// * Closed-shell only (open-shell U-RPA is C8 scope).
/// * Dense χ₀ only — returns an error for `Chi0Backend::Laplace`.
/// * Dense PDEP only — returns an error for `Chi0Sparsity::BoysScreened`.
///
/// # Arguments
///
/// * `mol`, `obs_basis`, `aux_basis` — geometry and basis sets at the
///   reference point.  Re-passed as BasisSet (not PreparedBasis) because the
///   underlying machinery needs to rebuild prepared bases at displaced
///   geometries.
/// * `op` — the ERI operator (typically `Operator::coulomb()`).
/// * `rpa_config` — RPA configuration; truncation threshold and Davidson
///   parameters are honored at every displaced point.
/// * `h` — finite-difference step in Bohr.  Default-recommended: `5e-4`.
pub fn rpa_correlation_gradient(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    rpa_config: &PdepRpaConfig,
    h: f64,
) -> Result<Array2<f64>, FerricError> {
    if !matches!(rpa_config.chi0_backend, Chi0Backend::Dense) {
        return Err(FerricError::General(
            "rpa_correlation_gradient: only Chi0Backend::Dense is supported (Laplace path is energy-only; see C-grad scope)".into(),
        ));
    }
    if !matches!(rpa_config.chi0_sparsity, Chi0Sparsity::Dense) {
        return Err(FerricError::General(
            "rpa_correlation_gradient: only Chi0Sparsity::Dense is supported (BoysScreened gradient requires re-derivation of the Z-vector in the sparse representation; deferred)".into(),
        ));
    }

    let natoms = mol.atoms.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));

    for atom in 0..natoms {
        for coord in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            apply_displacement(&mut mol_p, atom, coord, h);
            apply_displacement(&mut mol_m, atom, coord, -h);
            let e_p = rpa_total_energy(&mol_p, obs_basis, aux_basis, op, rpa_config)?;
            let e_m = rpa_total_energy(&mol_m, obs_basis, aux_basis, op, rpa_config)?;
            grad[(atom, coord)] = (e_p - e_m) / (2.0 * h);
        }
    }

    Ok(grad)
}

fn apply_displacement(mol: &mut Molecule, atom: usize, coord: usize, h: f64) {
    match coord {
        0 => mol.atoms[atom].x += h,
        1 => mol.atoms[atom].y += h,
        _ => mol.atoms[atom].zpos += h,
    }
}

/// Re-solve RHF and re-run PDEP-RPA at a (possibly displaced) geometry and
/// return the TOTAL energy E_RHF + E_c^RPA, the quantity the FD gradient
/// differences.
fn rpa_total_energy(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    rpa_config: &PdepRpaConfig,
) -> Result<f64, FerricError> {
    let ctx = ParallelContext::default();
    let obs = PreparedBasis::new(mol, obs_basis)?;
    let dfbs = PreparedBasis::new(mol, aux_basis)?;
    let bounds = SchwarzBounds::compute(op, &obs)?;
    // Tighten SCF convergence so FD differences are not noise-limited. Under the
    // ΔP convergence gate the tight signal is density_conv (reachable, ~1e-9);
    // energy_conv is only a loose "not-descending" bound (floors above 1e-10
    // under DF noise), so it is left at the default rather than set to 1e-10 —
    // a tight energy_conv would hang the SCF at MaxIter. See rhf::scf_converged.
    let rhf_cfg = RhfConfig {
        density_conv: 1e-9,
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, mol, &obs, op, &bounds, &rhf_cfg)?;
    if !rhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        });
    }
    let r = run_pdep_rpa(mol, &obs, &dfbs, op, &rhf, rpa_config)?;
    Ok(rhf.energy + r.e_rpa)
}

/// The RPA total energy at the reference geometry and the full RPA nuclear
/// gradient (`rpa_correlation_gradient`, a finite difference of the total
/// energy). Used to drive geometry optimization.
pub fn total_rpa_gradient(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    rpa_config: &PdepRpaConfig,
    h: f64,
) -> Result<(f64, Array2<f64>), FerricError> {
    let ctx = ParallelContext::default();
    let obs = PreparedBasis::new(mol, obs_basis)?;
    let dfbs = PreparedBasis::new(mol, aux_basis)?;
    let bounds = SchwarzBounds::compute(op, &obs)?;
    // Tight SCF via density_conv (reachable under the ΔP gate); energy_conv left
    // at the loose default — a tight 1e-10 would hang at MaxIter. See above / gradient.rs:149.
    let rhf_cfg = RhfConfig {
        density_conv: 1e-9,
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, mol, &obs, op, &bounds, &rhf_cfg)?;
    if !rhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        });
    }
    let r: PdepRpaResult = run_pdep_rpa(mol, &obs, &dfbs, op, &rhf, rpa_config)?;
    let e_tot = rhf.energy + r.e_rpa;

    let grad = rpa_correlation_gradient(mol, obs_basis, aux_basis, op, rpa_config, h)?;
    Ok((e_tot, grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{QuadratureConfig, QuadratureScheme};
    use ferric_core::basis;

    fn h2o() -> Molecule {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        Molecule::parse_xyz(xyz, 0, 1).unwrap()
    }

    fn small_rpa_cfg() -> PdepRpaConfig {
        PdepRpaConfig {
            frozen_core: 0,
            trunc_thresh: 1e-4,
            eigensolver_conv_thresh: 1e-10,
            quadrature: QuadratureConfig {
                scheme: QuadratureScheme::GaussLegendre,
                n_points: 16,
                u0: 0.5,
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_rpa_gradient_translational_invariance() {
        // Sum of forces over all atoms must vanish in each Cartesian direction.
        // This is a property of any geometry-derived gradient regardless of accuracy.
        let mol = h2o();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let cfg = small_rpa_cfg();
        let grad = rpa_correlation_gradient(&mol, &obs_bs, &aux_bs, op, &cfg, 5e-4).unwrap();
        eprintln!("H2O/STO-3G RPA correlation gradient:");
        for (a, row) in grad.outer_iter().enumerate() {
            eprintln!(
                "  atom {a}: [{:+.6e}, {:+.6e}, {:+.6e}]",
                row[0], row[1], row[2]
            );
        }
        for c in 0..3 {
            let s: f64 = (0..3).map(|a| grad[(a, c)]).sum();
            assert!(s.abs() < 1e-6, "trans-invariance coord {c}: sum={s:.3e}");
        }
    }

    #[test]
    #[ignore] // ~1-2 min runtime
    fn test_rpa_gradient_h2o_ccpvdz_vs_fd() {
        // Self-consistency check: the FD gradient at h=5e-4 should
        // reproduce itself to round-off at h=2.5e-4 (Richardson convergence).
        let mol = h2o();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let cfg = small_rpa_cfg();
        let g1 = rpa_correlation_gradient(&mol, &obs_bs, &aux_bs, op, &cfg, 5e-4).unwrap();
        let g2 = rpa_correlation_gradient(&mol, &obs_bs, &aux_bs, op, &cfg, 2.5e-4).unwrap();
        eprintln!("=== H2O/cc-pVDZ RPA correlation gradient, FD step convergence ===");
        let mut max = 0.0_f64;
        for a in 0..3 {
            for c in 0..3 {
                let d = (g1[(a, c)] - g2[(a, c)]).abs();
                max = max.max(d);
                eprintln!(
                    "  atom={a} coord={c}: h=5e-4 {:+.8} h=2.5e-4 {:+.8} diff {:.2e}",
                    g1[(a, c)],
                    g2[(a, c)],
                    d
                );
            }
        }
        eprintln!("  max diff = {:.2e}", max);
        assert!(max < 5e-5, "FD convergence failed: {max:.2e}");
    }
}
