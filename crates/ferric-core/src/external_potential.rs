//! Classical external potentials: fixed point charges and a uniform
//! external electric field. See docs/superpowers/specs/2026-07-10-external-potentials-design.md.

use crate::mol::Molecule;
use ndarray::Array2;

/// A fixed classical point charge in Bohr / atomic units. Not a physical
/// atom: no basis shells, and `q` may be fractional (e.g. QM/MM partial
/// charges).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointCharge {
    pub q: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// A classical external environment: fixed point charges plus an optional
/// uniform electric field (atomic units, lab frame). `Default` is the empty
/// potential (no point charges, no field) — every consumer must treat this
/// as a true no-op.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExternalPotential {
    pub point_charges: Vec<PointCharge>,
    pub field: Option<[f64; 3]>,
}

impl ExternalPotential {
    /// Returns `true` if there are no point charges and no external field.
    pub fn is_empty(&self) -> bool {
        self.point_charges.is_empty() && self.field.is_none()
    }

    /// Classical charge-nuclear Coulomb energy: Σ_A Σ_i Z_A·q_i / |R_A − r_i|,
    /// summed over real (non-ghost) atoms A and external charges i.
    pub fn charge_nuclear_energy(&self, mol: &Molecule) -> f64 {
        let mut e = 0.0;
        for atom in &mol.atoms {
            if atom.ghost {
                continue;
            }
            let za = atom.effective_z() as f64;
            for pc in &self.point_charges {
                let dx = atom.x - pc.x;
                let dy = atom.y - pc.y;
                let dz = atom.zpos - pc.z;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                e += za * pc.q / r;
            }
        }
        e
    }

    /// Classical field-nuclear energy: -E·Σ_A Z_A·R_A over real atoms.
    pub fn field_nuclear_energy(&self, mol: &Molecule) -> f64 {
        let Some(field) = self.field else { return 0.0 };
        let mut e = 0.0;
        for atom in &mol.atoms {
            if atom.ghost {
                continue;
            }
            let za = atom.effective_z() as f64;
            e += -(field[0] * atom.x + field[1] * atom.y + field[2] * atom.zpos) * za;
        }
        e
    }

    /// dE/dR for the charge-nuclear term, QM-atom rows only (no rows for
    /// the external charges — they are fixed, not gradient variables).
    /// Sign/convention matches `gradient::oneelectron_gradient`'s nuclear
    /// repulsion loop: this returns dE/dR directly (not the force -dE/dR).
    pub fn charge_nuclear_gradient(&self, mol: &Molecule) -> Array2<f64> {
        let natoms = mol.atoms.len();
        let mut grad = Array2::zeros((natoms, 3));
        for (i, atom) in mol.atoms.iter().enumerate() {
            if atom.ghost {
                continue;
            }
            let za = atom.effective_z() as f64;
            for pc in &self.point_charges {
                let dx = atom.x - pc.x;
                let dy = atom.y - pc.y;
                let dz = atom.zpos - pc.z;
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt();
                // dE/dR_A = -Z_A*q*(R_A - R_C)/r^3 for E = Z*q/r.
                let f_over_r3 = -za * pc.q / (r * r2);
                grad[(i, 0)] += f_over_r3 * dx;
                grad[(i, 1)] += f_over_r3 * dy;
                grad[(i, 2)] += f_over_r3 * dz;
            }
        }
        grad
    }

    /// dE/dR for the field-nuclear term: constant -Z_A·E per real atom A
    /// (QM-atom rows only).
    pub fn field_nuclear_gradient(&self, mol: &Molecule) -> Array2<f64> {
        let natoms = mol.atoms.len();
        let mut grad = Array2::zeros((natoms, 3));
        let Some(field) = self.field else { return grad };
        for (i, atom) in mol.atoms.iter().enumerate() {
            if atom.ghost {
                continue;
            }
            let za = atom.effective_z() as f64;
            grad[(i, 0)] += -field[0] * za;
            grad[(i, 1)] += -field[1] * za;
            grad[(i, 2)] += -field[2] * za;
        }
        grad
    }
}

#[cfg(test)]
mod tests {
    use crate::mol::Molecule;

    fn single_h_atom() -> Molecule {
        // One H atom at the origin.
        Molecule::parse_xyz("1\nH\nH 0.0 0.0 0.0\n", 0, 2).unwrap()
    }

    #[test]
    fn charge_nuclear_energy_matches_coulomb_law() {
        let mol = single_h_atom();
        // Point charge q=2.0 at (0,0,5) Bohr... but parse_xyz converts Å->Bohr,
        // so place the charge directly in Bohr via the PointCharge struct
        // (ExternalPotential is always in Bohr/atomic units, independent of
        // the XYZ parser's unit conversion).
        let ep = super::ExternalPotential {
            point_charges: vec![super::PointCharge {
                q: 2.0,
                x: 0.0,
                y: 0.0,
                z: 5.0,
            }],
            field: None,
        };
        let e = ep.charge_nuclear_energy(&mol);
        // Z_H=1, q=2.0, r=5.0 Bohr => E = 1.0*2.0/5.0 = 0.4
        assert!((e - 0.4).abs() < 1e-12, "got {e}");
    }

    #[test]
    fn charge_nuclear_energy_ignores_ghost_atoms() {
        let mut mol = single_h_atom();
        mol.atoms[0].ghost = true;
        let ep = super::ExternalPotential {
            point_charges: vec![super::PointCharge {
                q: 2.0,
                x: 0.0,
                y: 0.0,
                z: 5.0,
            }],
            field: None,
        };
        assert_eq!(ep.charge_nuclear_energy(&mol), 0.0);
    }

    #[test]
    fn charge_nuclear_energy_empty_is_zero() {
        let mol = single_h_atom();
        let ep = super::ExternalPotential::default();
        assert_eq!(ep.charge_nuclear_energy(&mol), 0.0);
    }

    #[test]
    fn field_nuclear_energy_matches_hand_calc() {
        let mol = single_h_atom();
        let ep = super::ExternalPotential {
            point_charges: vec![],
            field: Some([0.0, 0.0, 0.01]),
        };
        // E = -E·Σ Z_A R_A = -(0.01)*(1.0*0.0) = 0.0 at the origin
        assert_eq!(ep.field_nuclear_energy(&mol), 0.0);
    }

    #[test]
    fn field_nuclear_energy_nonzero_offset() {
        // H atom displaced along z; field along z.
        let mol = Molecule::parse_xyz("1\nH\nH 0.0 0.0 1.0\n", 0, 2).unwrap();
        let ep = super::ExternalPotential {
            point_charges: vec![],
            field: Some([0.0, 0.0, 0.01]),
        };
        let e = ep.field_nuclear_energy(&mol);
        let z_bohr = 1.0 / 0.529_177_210_92; // parse_xyz converts Å -> Bohr
        let expected = -0.01 * z_bohr;
        assert!((e - expected).abs() < 1e-10, "got {e}, expected {expected}");
    }

    #[test]
    fn charge_nuclear_gradient_matches_coulomb_force() {
        let mol = single_h_atom();
        let ep = super::ExternalPotential {
            point_charges: vec![super::PointCharge {
                q: 2.0,
                x: 0.0,
                y: 0.0,
                z: 5.0,
            }],
            field: None,
        };
        let g = ep.charge_nuclear_gradient(&mol);
        assert_eq!(g.dim(), (1, 3));
        // F_z on the H atom = -dE/dz_H. E = Z*q/|R_H - R_C|; with R_H=(0,0,0),
        // R_C=(0,0,5): dE/dz_H = -Z*q*(z_H - z_C)/r^3 = -1*2*(-5)/125 = 0.08
        assert!((g[(0, 2)] - 0.08).abs() < 1e-10, "got {}", g[(0, 2)]);
        assert!(g[(0, 0)].abs() < 1e-12);
        assert!(g[(0, 1)].abs() < 1e-12);
    }

    #[test]
    fn field_nuclear_gradient_is_constant_force() {
        let mol = single_h_atom();
        let ep = super::ExternalPotential {
            point_charges: vec![],
            field: Some([0.0, 0.0, 0.01]),
        };
        let g = ep.field_nuclear_gradient(&mol);
        // dE/dz_H = -E_z * Z_H = -0.01; force convention in this codebase's
        // gradient.rs is dE/dR (not -dE/dR — see nuclear_repulsion_gradient
        // which accumulates +f_over_r3*dx directly as dE/dR), so we assert
        // on dE/dR here for consistency with oneelectron_gradient's convention.
        assert!((g[(0, 2)] - (-0.01)).abs() < 1e-12, "got {}", g[(0, 2)]);
    }
}
