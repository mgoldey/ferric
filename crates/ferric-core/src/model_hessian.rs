//! Lindh model Hessian: an empirical force field used only to seed the
//! quasi-Newton geometry search.
//!
//! R. Lindh, A. Bernhardsson, G. Karlström and P.-Å. Malmqvist, Chem. Phys.
//! Lett. 241, 423 (1995), doi:10.1016/0009-2614(95)00646-L.
//!
//! Every pair, bend and torsion of the molecule contributes a harmonic term
//! whose force constant decays with the interatomic distances:
//!
//! ```text
//! rho_ij  = exp(alpha_ij (r_ref,ij^2 - r_ij^2))
//! k_ij    = k_r   rho_ij                     (stretch)
//! k_ijk   = k_phi rho_ij rho_jk              (bend, vertex j)
//! k_ijkl  = k_tau rho_ij rho_jk rho_kl       (torsion about j-k)
//! H_cart  = sum_q k_q b_q b_q^T,   b_q = dq/dx
//! ```
//!
//! with `alpha` and `r_ref` from the period of each atom (Table 1 of the
//! paper, rows H-He, Li-Ne, Na and heavier). Each term is a rank-one positive
//! semi-definite matrix built from a translation- and rotation-invariant
//! primitive, so `H_cart` is symmetric, positive semi-definite, and has the
//! rigid-body motions in its null space. Equations: `crates/ferric-scf/theory.md`,
//! "Lindh model Hessian"; proof `scripts/lindh_hessian_sympy.py`.
//!
//! # Terms that are dropped
//!
//! * Bends within 5 degrees of linear and torsions with such a bend
//!   ([`is_near_linear`](crate::internal_coords::is_near_linear)): their Wilson derivatives diverge. Dropping a
//!   positive semi-definite term keeps the sum positive semi-definite. The
//!   optimizer gives directions with no model curvature a floor value.
//! * Terms whose `rho` product is below [`LINDH_RHO_CUTOFF`](crate::model_hessian::LINDH_RHO_CUTOFF). At that weight a
//!   torsion's force constant is `5e-11` Ha/rad² and a stretch's `5e-9`
//!   Ha/Bohr², below anything that changes a step.

use crate::internal_coords::{is_near_linear, primitive_derivatives, Primitive};
use crate::mol::Molecule;
use crate::FerricError;
use ndarray::Array2;

/// Stretch force constant `k_r` (Hartree/Bohr²).
pub const LINDH_K_STRETCH: f64 = 0.45;
/// Bend force constant `k_phi` (Hartree/rad²).
pub const LINDH_K_BEND: f64 = 0.15;
/// Torsion force constant `k_tau` (Hartree/rad²).
pub const LINDH_K_TORSION: f64 = 0.005;

/// `alpha_ij` (Bohr⁻²) by period pair, Lindh et al. 1995 Table 1.
pub const LINDH_ALPHA: [[f64; 3]; 3] = [
    [1.0000, 0.3949, 0.3949],
    [0.3949, 0.2800, 0.2800],
    [0.3949, 0.2800, 0.2800],
];

/// `r_ref,ij` (Bohr) by period pair, Lindh et al. 1995 Table 1.
pub const LINDH_R_REF: [[f64; 3]; 3] = [[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]];

/// Terms whose `rho` product is below this are not added (see the module doc).
pub const LINDH_RHO_CUTOFF: f64 = 1e-8;

/// Lindh's period index: 0 for H-He, 1 for Li-Ne, 2 for Na and heavier.
#[must_use]
pub fn lindh_period(z: i32) -> usize {
    if z <= 2 {
        0
    } else if z <= 10 {
        1
    } else {
        2
    }
}

/// The distance weight `rho_ij` of atoms `i` and `j` of `mol`.
#[must_use]
pub fn lindh_rho(mol: &Molecule, i: usize, j: usize) -> f64 {
    let (a, b) = (&mol.atoms[i], &mol.atoms[j]);
    let (pa, pb) = (lindh_period(a.z), lindh_period(b.z));
    let r2 = (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.zpos - b.zpos).powi(2);
    let rref = LINDH_R_REF[pa][pb];
    (LINDH_ALPHA[pa][pb] * (rref * rref - r2)).exp()
}

/// Lindh force constant of one primitive at `mol`'s geometry (Hartree/Bohr²
/// for a stretch, Hartree/rad² for a bend or torsion).
#[must_use]
pub fn lindh_force_constant(mol: &Molecule, p: Primitive) -> f64 {
    match p {
        Primitive::Bond(i, j) => LINDH_K_STRETCH * lindh_rho(mol, i, j),
        Primitive::Angle(i, j, k) => LINDH_K_BEND * lindh_rho(mol, i, j) * lindh_rho(mol, j, k),
        Primitive::Dihedral(i, j, k, l) => {
            LINDH_K_TORSION * lindh_rho(mol, i, j) * lindh_rho(mol, j, k) * lindh_rho(mol, k, l)
        }
    }
}

/// The Lindh model Hessian in Cartesian coordinates, `(3N, 3N)`, Hartree/Bohr².
///
/// Sums every stretch `i<j`, every bend `i-j-k` (`i<k`) and every torsion
/// `i-j-k-l` (`j<k`, so each torsion is counted once), skipping the terms
/// listed in the module doc.
///
/// # Errors
///
/// Propagates a degenerate-primitive error (coincident atoms).
pub fn lindh_cartesian_hessian(mol: &Molecule) -> Result<Array2<f64>, FerricError> {
    let n = mol.atoms.len();
    let mut rho = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                rho[(i, j)] = lindh_rho(mol, i, j);
            }
        }
    }
    let rho_max = rho.iter().copied().fold(0.0f64, f64::max).max(1.0);
    // A pair below this weight cannot take part in a term above the cutoff,
    // whatever the other factors (each is at most `rho_max`).
    let pair_cut = LINDH_RHO_CUTOFF / (rho_max * rho_max);
    let nb: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i && rho[(i, j)] > pair_cut)
                .collect()
        })
        .collect();

    let mut h = Array2::<f64>::zeros((3 * n, 3 * n));
    for i in 0..n {
        for &j in nb[i].iter().filter(|&&j| j > i) {
            let w = rho[(i, j)];
            if w >= LINDH_RHO_CUTOFF {
                add_term(&mut h, mol, Primitive::Bond(i, j), LINDH_K_STRETCH * w)?;
            }
        }
    }
    for j in 0..n {
        for (a, &i) in nb[j].iter().enumerate() {
            for &k in &nb[j][a + 1..] {
                let w = rho[(i, j)] * rho[(j, k)];
                if w < LINDH_RHO_CUTOFF || is_near_linear(mol, i, j, k) {
                    continue;
                }
                add_term(&mut h, mol, Primitive::Angle(i, j, k), LINDH_K_BEND * w)?;
            }
        }
    }
    for j in 0..n {
        for &k in nb[j].iter().filter(|&&k| k > j) {
            let wjk = rho[(j, k)];
            for &i in nb[j].iter().filter(|&&i| i != k) {
                let wij = rho[(i, j)] * wjk;
                if wij * rho_max < LINDH_RHO_CUTOFF || is_near_linear(mol, i, j, k) {
                    continue;
                }
                for &l in nb[k].iter().filter(|&&l| l != j && l != i) {
                    let w = wij * rho[(k, l)];
                    if w < LINDH_RHO_CUTOFF || is_near_linear(mol, j, k, l) {
                        continue;
                    }
                    add_term(
                        &mut h,
                        mol,
                        Primitive::Dihedral(i, j, k, l),
                        LINDH_K_TORSION * w,
                    )?;
                }
            }
        }
    }
    Ok(h)
}

/// `h += k b bᵀ` for the primitive `p`.
fn add_term(h: &mut Array2<f64>, mol: &Molecule, p: Primitive, k: f64) -> Result<(), FerricError> {
    let (atoms, derivs) = primitive_derivatives(mol, p)?;
    for (a, da) in atoms.iter().zip(&derivs) {
        for (b, db) in atoms.iter().zip(&derivs) {
            for c in 0..3 {
                for d in 0..3 {
                    h[(3 * a + c, 3 * b + d)] += k * da[c] * db[d];
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water() -> Molecule {
        Molecule::parse_xyz(
            "3\nH2O\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap()
    }

    fn ethanol() -> Molecule {
        Molecule::parse_xyz(
            "9\nethanol\n\
             C -0.0478 -0.4252 0.0000\nC 1.2600 0.3599 0.0000\nO -1.1661 0.4545 0.0000\n\
             H -0.0951 -1.0675 0.8857\nH -0.0951 -1.0675 -0.8857\nH 2.1102 -0.3268 0.0000\n\
             H 1.3184 1.0000 0.8810\nH 1.3184 1.0000 -0.8810\nH -1.9867 -0.0507 0.0000\n",
            0,
            1,
        )
        .unwrap()
    }

    fn rigid_modes(mol: &Molecule) -> Vec<Vec<f64>> {
        let n = mol.atoms.len();
        let mut out = Vec::new();
        for c in 0..3 {
            let mut v = vec![0.0; 3 * n];
            for a in 0..n {
                v[3 * a + c] = 1.0;
            }
            out.push(v);
        }
        for axis in 0..3 {
            let mut v = vec![0.0; 3 * n];
            for (a, at) in mol.atoms.iter().enumerate() {
                let r = [at.x, at.y, at.zpos];
                // e_axis x r
                let (p, q) = ((axis + 1) % 3, (axis + 2) % 3);
                v[3 * a + q] += r[p];
                v[3 * a + p] -= r[q];
            }
            out.push(v);
        }
        out
    }

    /// Symmetric, positive semi-definite, and the six rigid-body motions are
    /// null vectors to rounding (each term is built from an invariant
    /// primitive, so `H t = 0` holds term by term).
    #[test]
    fn lindh_hessian_is_symmetric_psd_with_rigid_body_null_space() {
        use ndarray_linalg::{Eigh, UPLO};
        for mol in [water(), ethanol()] {
            let h = lindh_cartesian_hessian(&mol).unwrap();
            let asym = (&h - &h.t()).iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(asym < 1e-14, "asymmetry {asym:e}");
            let (w, _) = h.eigh(UPLO::Upper).unwrap();
            let scale = w.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(w[0] > -1e-12 * scale, "negative eigenvalue {:e}", w[0]);
            for t in rigid_modes(&mol) {
                let t = ndarray::Array1::from(t);
                let ht = h.dot(&t);
                let rel = ht.dot(&ht).sqrt() / (t.dot(&t).sqrt() * scale);
                assert!(rel < 1e-12, "rigid mode not null: |Ht|/|t||H| = {rel:e}");
            }
            // Exactly six eigenvalues below 1e-10 of the scale: the rigid
            // modes, and no accidental null directions in a bent molecule.
            let nzero = w.iter().filter(|v| v.abs() < 1e-10 * scale).count();
            assert_eq!(nzero, 6, "eigenvalues {w:?}");
        }
    }

    /// `rho = 1` at the reference distance, so a stretch there carries `k_r`.
    #[test]
    fn force_constant_is_k_r_at_the_reference_distance() {
        let r = LINDH_R_REF[1][1] * crate::units::BOHR_TO_ANGSTROM;
        let mol = Molecule::parse_xyz(&format!("2\nCC\nC 0 0 0\nC 0 0 {r}\n"), 0, 1).unwrap();
        let k = lindh_force_constant(&mol, Primitive::Bond(0, 1));
        assert!((k - LINDH_K_STRETCH).abs() < 1e-12, "k = {k}");
    }
}
