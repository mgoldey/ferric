//! RI-MP2 nuclear gradient via central finite differences.
//!
//! Computes dE_total/dR for E_total = E_HF + E_MP2(RI) by displacing each
//! nuclear coordinate by +/- delta and taking the central difference.
//!
//! This is O(6*N_atoms * cost_of_RI-MP2) but is exact (up to FD truncation)
//! and serves as the reference implementation for validating future analytical
//! gradient code.

use crate::rimp2::{ri_mp2, RiMp2Config};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Compute the total RI-MP2 energy (E_HF + E_MP2) for a given geometry.
fn total_energy(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
) -> Result<f64, FerricError> {
    let obs = PreparedBasis::new(mol, obs_basis)?;
    let bounds = SchwarzBounds::compute(op, &obs)?;
    let rhf_config = RhfConfig {
        energy_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(mol, &obs, op, &bounds, &rhf_config)?;
    let dfbs = PreparedBasis::new(mol, aux_basis)?;
    let mp2 = ri_mp2(mol, &obs, &dfbs, op, &rhf, mp2_config)?;
    Ok(mp2.total_energy)
}

/// Compute RI-MP2 nuclear gradient via central finite differences.
///
/// `delta` is in Bohr (molecular coordinates are in Bohr).
/// Returns a `(natoms, 3)` array of dE/dR per atom per Cartesian direction.
pub fn rimp2_gradient_fd(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    delta: f64,
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let mut grad = Array2::zeros((natoms, 3));
    for atom in 0..natoms {
        for coord in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match coord {
                0 => {
                    mol_p.atoms[atom].x += delta;
                    mol_m.atoms[atom].x -= delta;
                }
                1 => {
                    mol_p.atoms[atom].y += delta;
                    mol_m.atoms[atom].y -= delta;
                }
                _ => {
                    mol_p.atoms[atom].zpos += delta;
                    mol_m.atoms[atom].zpos -= delta;
                }
            }
            let e_p = total_energy(&mol_p, obs_basis, aux_basis, op, mp2_config)?;
            let e_m = total_energy(&mol_m, obs_basis, aux_basis, op, mp2_config)?;
            grad[(atom, coord)] = (e_p - e_m) / (2.0 * delta);
        }
    }
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    #[test]
    fn test_rimp2_gradient_fd_h2_symmetry() {
        // H2 along z-axis: gradient should be equal and opposite on the two atoms
        // and zero in x,y directions.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let config = RiMp2Config::default();
        let grad = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();

        eprintln!("H2 RI-MP2 gradient (FD, delta=1e-4):");
        for atom in 0..2 {
            eprintln!(
                "  atom {}: [{:+.10}, {:+.10}, {:+.10}]",
                atom,
                grad[(atom, 0)],
                grad[(atom, 1)],
                grad[(atom, 2)]
            );
        }

        // x and y gradients should be ~0
        for atom in 0..2 {
            for c in 0..2 {
                assert!(
                    grad[(atom, c)].abs() < 1e-8,
                    "atom={atom} coord={c}: {:.2e} should be ~0",
                    grad[(atom, c)]
                );
            }
        }

        // z gradients should be equal and opposite (translational invariance)
        assert!(
            (grad[(0, 2)] + grad[(1, 2)]).abs() < 1e-8,
            "z gradients not equal/opposite: {} vs {}",
            grad[(0, 2)],
            grad[(1, 2)]
        );

        // Should be nonzero (H2 at 0.74 A is near but not at equilibrium for MP2/cc-pVDZ)
        assert!(
            grad[(0, 2)].abs() > 1e-4,
            "z gradient too small: {}",
            grad[(0, 2)]
        );
    }

    #[test]
    fn test_rimp2_gradient_fd_consistency() {
        // Check that two different deltas give consistent results (FD convergence).
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let config = RiMp2Config::default();

        let g1 = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();
        let g2 = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 5e-5).unwrap();

        eprintln!("FD consistency check (H2 z-component):");
        eprintln!("  delta=1e-4: {:.10}", g1[(0, 2)]);
        eprintln!("  delta=5e-5: {:.10}", g2[(0, 2)]);
        eprintln!("  diff:       {:.2e}", (g1[(0, 2)] - g2[(0, 2)]).abs());

        // Central FD has O(delta^2) error, so halving delta should reduce error by ~4x.
        // Two independent runs should agree to ~1e-5 or better.
        assert!(
            (g1[(0, 2)] - g2[(0, 2)]).abs() < 1e-5,
            "FD inconsistent: delta=1e-4 gives {:.10}, delta=5e-5 gives {:.10}",
            g1[(0, 2)],
            g2[(0, 2)]
        );
    }

    #[test]
    fn test_rimp2_gradient_fd_h2o_translational_invariance() {
        // For any geometry the sum of forces over all atoms should be zero
        // (translational invariance / Newton's third law).
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let config = RiMp2Config::default();

        let grad = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();

        eprintln!("H2O/STO-3G RI-MP2 gradient (FD):");
        for atom in 0..3 {
            eprintln!(
                "  atom {}: [{:+.10}, {:+.10}, {:+.10}]",
                atom,
                grad[(atom, 0)],
                grad[(atom, 1)],
                grad[(atom, 2)]
            );
        }

        // Sum of gradients over all atoms should vanish for each coordinate.
        for c in 0..3 {
            let sum: f64 = (0..3).map(|a| grad[(a, c)]).sum();
            assert!(
                sum.abs() < 1e-6,
                "translational invariance violated: coord={c}, sum={:.2e}",
                sum
            );
        }
    }
}
