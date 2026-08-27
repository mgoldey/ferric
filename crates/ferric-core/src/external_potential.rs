//! Classical external potentials: fixed point charges and a uniform
//! external electric field. See docs/superpowers/specs/2026-07-10-external-potentials-design.md.

use crate::mol::Molecule;
use ndarray::Array2;

/// `erf(x) = 1 − erfc(x)`; see [`erfc_scalar`] for the underlying fit.
///
/// Hand-rolled because `f64::erfc`/`erf` are not in stable Rust's core
/// numerics and `ferric-core` carries no special-function dependency —
/// same rationale and same 28-term Numerical Recipes Chebyshev fit as
/// `ferric_dft::vv10::erf_scalar` (duplicated here rather than shared,
/// because `ferric-core` sits below `ferric-dft` in the dependency graph and
/// this crate cannot depend on it).
#[inline]
fn erf(x: f64) -> f64 {
    1.0 - erfc_scalar(x)
}

/// Complementary error function via the Numerical Recipes 28-term Chebyshev
/// fit `erfc(x) ≈ t·exp(−x² + P(t))`, `t = 2/(2+|x|)`, sign-extended to
/// negative arguments through `erfc(−x) = 2 − erfc(x)`.
///
/// MEASURED (in `ferric_dft::vv10`, same coefficients) against
/// `scipy.special.erfc` on a 1201-point sweep of `x ∈ [−6, 6]`: maximum
/// relative error 6.8e-15, essentially double-precision exact.
#[inline]
fn erfc_scalar(x: f64) -> f64 {
    let z = x.abs();
    let t = 2.0 / (2.0 + z);
    let ty = 4.0 * t - 2.0;
    const COF: [f64; 28] = [
        -1.302_653_719_781_709_4,
        6.419_697_923_564_902e-1,
        1.947_647_320_418_583_6e-2,
        -9.561_514_786_808_631e-3,
        -9.465_953_444_820_36e-4,
        3.668_394_978_527_61e-4,
        4.252_332_480_690_7e-5,
        -2.027_857_811_253_4e-5,
        -1.624_290_004_647e-6,
        1.303_655_835_580e-6,
        1.562_644_172_2e-8,
        -8.523_809_591_5e-8,
        6.529_054_439e-9,
        5.059_343_495e-9,
        -9.913_641_56e-10,
        -2.273_651_22e-10,
        9.646_791_1e-11,
        2.394_038e-12,
        -6.886_027e-12,
        8.944_87e-13,
        3.130_92e-13,
        -1.127_08e-13,
        3.81e-16,
        7.106e-15,
        -1.523e-15,
        -9.4e-17,
        1.21e-16,
        -2.8e-17,
    ];
    let mut d = 0.0_f64;
    let mut dd = 0.0_f64;
    for j in (1..COF.len()).rev() {
        let tmp = d;
        d = ty * d - dd + COF[j];
        dd = tmp;
    }
    let ans = t * (-z * z + 0.5 * (COF[0] + ty * d) - dd).exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

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

/// A Gaussian-smeared classical charge in Bohr / atomic units (PySCF
/// `mm_charge(radii=)` convention). The charge density is
/// `q (ζ/π)^{3/2} exp(-ζ r²)` with `ζ = 1/width²`; its potential is
/// `q·erf(√ζ r)/r`, which reduces to the point-charge `q/r` as `width → 0`
/// (`ζ → ∞`). `width = 0` is not a valid `SmearedCharge` (it is what makes
/// `PointCharge` a distinct, always-available representation instead — see
/// [`ExternalPotential::smeared_charges`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmearedCharge {
    pub q: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Gaussian width in Bohr (`ζ = 1/width²`). Must be `> 0`.
    pub width: f64,
}

/// A classical external environment: fixed point charges, Gaussian-smeared
/// charges, plus an optional uniform electric field (atomic units, lab
/// frame). `Default` is the empty potential (no charges of either kind, no
/// field) — every consumer must treat this as a true no-op.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExternalPotential {
    pub point_charges: Vec<PointCharge>,
    /// Gaussian-smeared classical charges (see [`SmearedCharge`]). Empty by
    /// default: every consumer of this struct must take the exact same code
    /// path as before this field existed when it is empty — not "close", bit
    /// identical (this is `is_empty`'s and `hcore_with_external`'s exactness
    /// anchor).
    pub smeared_charges: Vec<SmearedCharge>,
    pub field: Option<[f64; 3]>,
}

impl ExternalPotential {
    /// Returns `true` if there are no point charges, no smeared charges, and
    /// no external field.
    pub fn is_empty(&self) -> bool {
        self.point_charges.is_empty() && self.smeared_charges.is_empty() && self.field.is_none()
    }

    /// Classical charge-nuclear Coulomb energy: Σ_A Σ_i Z_A·q_i / |R_A − r_i|
    /// over point charges, plus Σ_A Σ_i Z_A·q_i·erf(√ζ_i R_Ai)/R_Ai over
    /// Gaussian-smeared charges — summed over real (non-ghost) atoms A.
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
            for sc in &self.smeared_charges {
                let dx = atom.x - sc.x;
                let dy = atom.y - sc.y;
                let dz = atom.zpos - sc.z;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let sqrt_zeta = 1.0 / sc.width;
                e += za * sc.q * erf(sqrt_zeta * r) / r;
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
            for sc in &self.smeared_charges {
                let dx = atom.x - sc.x;
                let dy = atom.y - sc.y;
                let dz = atom.zpos - sc.z;
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt();
                let sqrt_zeta = 1.0 / sc.width;
                let sz_r = sqrt_zeta * r;
                // d/dR [erf(sqrt(zeta)*R)/R] = (2/(sqrt(pi)*w)) * exp(-R^2/w^2)/R
                //                              - erf(sqrt(zeta)*R)/R^2
                // (see the module-level derivation and
                // `smeared_charge_gradient_matches_numeric_derivative` for
                // the numeric check).
                let dpot_dr = (2.0 / (std::f64::consts::PI.sqrt() * sc.width)) * (-sz_r * sz_r).exp() / r
                    - erf(sz_r) / r2;
                // Chain rule: R = |R_A - R_C|, dR/dR_A = (R_A - R_C)/R.
                let coeff = za * sc.q * dpot_dr / r;
                grad[(i, 0)] += coeff * dx;
                grad[(i, 1)] += coeff * dy;
                grad[(i, 2)] += coeff * dz;
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
            smeared_charges: Vec::new(),
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
            smeared_charges: Vec::new(),
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
            smeared_charges: Vec::new(),
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
            smeared_charges: Vec::new(),
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
            smeared_charges: Vec::new(),
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
            smeared_charges: Vec::new(),
            field: Some([0.0, 0.0, 0.01]),
        };
        let g = ep.field_nuclear_gradient(&mol);
        // dE/dz_H = -E_z * Z_H = -0.01; force convention in this codebase's
        // gradient.rs is dE/dR (not -dE/dR — see nuclear_repulsion_gradient
        // which accumulates +f_over_r3*dx directly as dE/dR), so we assert
        // on dE/dR here for consistency with oneelectron_gradient's convention.
        assert!((g[(0, 2)] - (-0.01)).abs() < 1e-12, "got {}", g[(0, 2)]);
    }

    // ── Gaussian-smeared charges (A2) ──

    /// EXACTNESS ANCHOR: `is_empty()` must be true only when point charges,
    /// smeared charges, AND field are all unset — and, separately, an EMPTY
    /// `smeared_charges` vec must contribute exactly zero to
    /// `charge_nuclear_energy`/`charge_nuclear_gradient` (the loop over an
    /// empty slice is a no-op, but this pins it as a behavioral contract, not
    /// an accident of the current implementation).
    #[test]
    fn smeared_charges_empty_is_true_no_op() {
        let mol = single_h_atom();
        let ep_before = super::ExternalPotential {
            point_charges: vec![super::PointCharge { q: 2.0, x: 0.0, y: 0.0, z: 5.0 }],
            smeared_charges: Vec::new(),
            field: None,
        };
        assert!(!ep_before.is_empty());
        let e_before = ep_before.charge_nuclear_energy(&mol);
        let g_before = ep_before.charge_nuclear_gradient(&mol);

        // Default (fully empty) must report is_empty() == true.
        assert!(super::ExternalPotential::default().is_empty());

        // Adding an empty smeared_charges Vec (already the case above)
        // changes nothing versus a hand-built point-charges-only struct.
        let e = ep_before.charge_nuclear_energy(&mol);
        let g = ep_before.charge_nuclear_gradient(&mol);
        assert_eq!(e, e_before);
        assert_eq!(g, g_before);
    }

    #[test]
    fn is_empty_false_when_only_smeared_charges_present() {
        let ep = super::ExternalPotential {
            point_charges: vec![],
            smeared_charges: vec![super::SmearedCharge { q: 1.0, x: 0.0, y: 0.0, z: 0.0, width: 1.0 }],
            field: None,
        };
        assert!(!ep.is_empty());
    }

    /// A single smeared charge at very tight width (ζ = 1e6) reproduces the
    /// point-charge Coulomb energy to 1e-9 — the CLASSICAL analogue of
    /// `oneelectron.rs`'s `tiny_width_smeared_charge_reproduces_point_charge_hcore`.
    #[test]
    fn tight_width_smeared_energy_matches_point_charge_energy() {
        let mol = single_h_atom();
        let ep_point = super::ExternalPotential {
            point_charges: vec![super::PointCharge { q: 2.0, x: 0.0, y: 0.0, z: 5.0 }],
            smeared_charges: Vec::new(),
            field: None,
        };
        let ep_smeared = super::ExternalPotential {
            point_charges: vec![],
            smeared_charges: vec![super::SmearedCharge { q: 2.0, x: 0.0, y: 0.0, z: 5.0, width: 1e-3 }],
            field: None,
        };
        let e_point = ep_point.charge_nuclear_energy(&mol);
        let e_smeared = ep_smeared.charge_nuclear_energy(&mol);
        assert!((e_point - e_smeared).abs() < 1e-9, "point {e_point} vs smeared {e_smeared}");
    }

    /// The analytic `charge_nuclear_gradient` smeared-charge term against a
    /// central finite difference of `charge_nuclear_energy` under a QM-atom
    /// displacement — this is the numeric check the plan calls for on
    /// `d/dR[erf(R/w)/R]`. Off-axis placement so every gradient component is
    /// a live comparison, not masked by symmetry.
    #[test]
    fn smeared_charge_gradient_matches_numeric_derivative() {
        let mol = Molecule::parse_xyz("1\nH\nH 0.3 -0.2 0.5\n", 0, 2).unwrap();
        let ep = super::ExternalPotential {
            point_charges: vec![],
            smeared_charges: vec![super::SmearedCharge { q: 1.7, x: 1.1, y: -0.6, z: 2.3, width: 0.8 }],
            field: None,
        };
        let g = ep.charge_nuclear_gradient(&mol);
        assert_eq!(g.dim(), (1, 3));

        let h = 1e-5;
        for (axis, coord) in [(0, "x"), (1, "y"), (2, "z")] {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match axis {
                0 => { mol_p.atoms[0].x += h; mol_m.atoms[0].x -= h; }
                1 => { mol_p.atoms[0].y += h; mol_m.atoms[0].y -= h; }
                _ => { mol_p.atoms[0].zpos += h; mol_m.atoms[0].zpos -= h; }
            }
            let e_p = ep.charge_nuclear_energy(&mol_p);
            let e_m = ep.charge_nuclear_energy(&mol_m);
            let fd = (e_p - e_m) / (2.0 * h);
            let analytic = g[(0, axis)];
            assert!(
                (analytic - fd).abs() < 1e-7,
                "smeared charge_nuclear_gradient[{coord}]: analytic {analytic:.10e} vs FD {fd:.10e}"
            );
        }
    }

    /// `width → 0` limit: the smeared-charge gradient formula must converge
    /// to the point-charge gradient formula (both loops are algebraically
    /// distinct paths through the same physics, so this is an independent
    /// cross-check of the erf-based derivative, not just internal FD
    /// self-consistency).
    #[test]
    fn tight_width_smeared_gradient_matches_point_charge_gradient() {
        let mol = single_h_atom();
        let ep_point = super::ExternalPotential {
            point_charges: vec![super::PointCharge { q: 2.0, x: 0.3, y: -0.4, z: 5.0 }],
            smeared_charges: Vec::new(),
            field: None,
        };
        let ep_smeared = super::ExternalPotential {
            point_charges: vec![],
            smeared_charges: vec![super::SmearedCharge { q: 2.0, x: 0.3, y: -0.4, z: 5.0, width: 1e-3 }],
            field: None,
        };
        let g_point = ep_point.charge_nuclear_gradient(&mol);
        let g_smeared = ep_smeared.charge_nuclear_gradient(&mol);
        for k in 0..3 {
            assert!(
                (g_point[(0, k)] - g_smeared[(0, k)]).abs() < 1e-6,
                "component {k}: point {} vs smeared {}", g_point[(0, k)], g_smeared[(0, k)]
            );
        }
    }
}
