//! Molecular cavity construction: atom-centered spheres tessellated into
//! surface elements ("tesserae").
//!
//! # Simplification vs full GEPOL
//!
//! Production PCM codes (GEPOL-93/Pomelli-Tomasi, or the switching/Gaussian
//! "SWIG" scheme of Lange & Herbert, J. Chem. Phys. 133, 244111 (2010), used
//! by e.g. PySCF's `pcm.py`) place a smooth, atom-pair switching function on
//! *every* point of every sphere so a point's effective area varies
//! continuously as it approaches a neighboring sphere's boundary — this
//! removes cavity-surface discontinuities under nuclear motion (needed for
//! smooth gradients) and avoids the "hard trim" artifacts of naive
//! tessellation (spuriously large/small tesserae right at a trim boundary).
//!
//! This module instead uses a **much simpler, hard-cutoff tessellation**:
//!
//! 1. Each atom gets a sphere of radius `vdw_scale * bondi_radius(Z)`.
//! 2. Each sphere is covered with a fixed-order Lebedev point set (reused
//!    directly from `ferric_dft::lebedev`, which is already validated for
//!    DFT quadrature).
//! 3. A point is **kept** iff it does not fall strictly inside any *other*
//!    atom's sphere (`|point − R_B| >= R_B` for all B != A); otherwise it is
//!    **discarded** entirely (binary in/out, no switching function).
//! 4. Each surviving point's area is `4π·R_A² · w_i` (the Lebedev weight
//!    already sums to 1 over the *unpruned* sphere) — i.e. areas are NOT
//!    renormalized after pruning. This slightly undercounts total cavity
//!    area relative to a proper GEPOL run (missing area from the pruned
//!    points near a sphere-sphere intersection is simply dropped rather than
//!    redistributed to an added intersection "belt", which is what full
//!    GEPOL does with additional spheres or Voronoi-corrected tesserae).
//!
//! This is a **known, documented simplification**: it works well for
//! reasonably convex, non-overlapping-cavity molecules (small/medium
//! organics at standard bond lengths — the water/small-polar-solute
//! validation case this module targets) but will systematically
//! under-tessellate near deep concave pockets or heavily fused ring systems
//! where a real GEPOL run would add interstitial spheres. A future
//! refinement (shared with any COSMO cavity implementation) would add: (a) a
//! smooth switching function per tessera instead of the hard keep/discard
//! cut, and (b) added spheres at sphere-sphere-sphere intersection points
//! (GEPOL's "creation of new spheres" step) for buried/concave regions.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::lebedev;

/// One surface element ("tessera") of the cavity.
#[derive(Debug, Clone, Copy)]
pub struct Tessera {
    /// Position in Bohr (lab frame, same frame as `Molecule` coordinates).
    pub position: [f64; 3],
    /// Outward unit normal (points away from the sphere center, into the
    /// dielectric).
    pub normal: [f64; 3],
    /// Surface area of this tessera, in Bohr².
    pub area: f64,
    /// Radius (Bohr) of the parent sphere this tessera belongs to. Needed
    /// for the diagonal self-term of the D matrix.
    pub sphere_radius: f64,
    /// Index of the parent atom (sphere center) in `Molecule::atoms`.
    pub atom_index: usize,
}

/// Cavity construction knobs.
#[derive(Debug, Clone)]
pub struct CavityConfig {
    /// Multiplicative scale factor applied to each atom's Bondi radius
    /// (standard practice: continuum electrostatics needs a cavity somewhat
    /// larger than the bare vdW surface so the dielectric doesn't intrude
    /// into the electron density's exponential tail). 1.2 is the widely used
    /// default (Q-Chem, PySCF's `pcm.py` default `vdw_scale`).
    pub vdw_scale: f64,
    /// Lebedev order used to tessellate each atomic sphere. Must be one of
    /// the orders supported by `ferric_dft::lebedev` (6, 14, 26, 50, 110,
    /// 302). 110 is a reasonable default resolution for a first correct
    /// implementation (302 for higher accuracy, at higher cost).
    pub lebedev_order: usize,
    /// Skip ghost atoms (no nuclear charge / basis-only centers) when
    /// placing spheres — a ghost atom is not a physical part of the solute
    /// and should not carve out cavity surface. Default `true`.
    pub skip_ghost_atoms: bool,
}

impl Default for CavityConfig {
    fn default() -> Self {
        Self {
            vdw_scale: 1.2,
            lebedev_order: 110,
            skip_ghost_atoms: true,
        }
    }
}

/// Build the molecular cavity: one Lebedev-tessellated sphere per (real)
/// atom, with points buried inside a neighboring sphere discarded.
///
/// Returns `Err` if the resulting cavity has zero tesserae (e.g. an empty
/// molecule, or every point on every sphere buried — degenerate geometry)
/// since a zero-tessera cavity cannot support a PCM linear solve.
pub fn build_cavity(mol: &Molecule, cfg: &CavityConfig) -> Result<Vec<Tessera>, FerricError> {
    if mol.atoms.is_empty() {
        return Err(FerricError::General(
            "build_cavity: molecule has no atoms".into(),
        ));
    }
    let (dirs, weights) = lebedev::lebedev(cfg.lebedev_order);
    if dirs.is_empty() {
        return Err(FerricError::General(format!(
            "build_cavity: unsupported lebedev_order {}",
            cfg.lebedev_order
        )));
    }

    let centers: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let radii: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| crate::radii::bondi_radius_bohr(a.z) * cfg.vdw_scale)
        .collect();

    let mut tesserae = Vec::new();

    for (a_idx, atom) in mol.atoms.iter().enumerate() {
        if cfg.skip_ghost_atoms && atom.ghost {
            continue;
        }
        let r_a = radii[a_idx];
        if r_a <= 0.0 || !r_a.is_finite() {
            return Err(FerricError::General(format!(
                "build_cavity: non-positive/finite radius {r_a} for atom {a_idx} (Z={})",
                atom.z
            )));
        }
        let center = centers[a_idx];

        for (dir, &w) in dirs.iter().zip(weights.iter()) {
            let point = [
                center[0] + r_a * dir[0],
                center[1] + r_a * dir[1],
                center[2] + r_a * dir[2],
            ];

            // Discard if buried inside any OTHER real atom's sphere.
            let mut buried = false;
            for (b_idx, other) in mol.atoms.iter().enumerate() {
                if b_idx == a_idx {
                    continue;
                }
                if cfg.skip_ghost_atoms && other.ghost {
                    continue;
                }
                let r_b = radii[b_idx];
                let cb = centers[b_idx];
                let dx = point[0] - cb[0];
                let dy = point[1] - cb[1];
                let dz = point[2] - cb[2];
                let dist2 = dx * dx + dy * dy + dz * dz;
                if dist2 < r_b * r_b {
                    buried = true;
                    break;
                }
            }
            if buried {
                continue;
            }

            let area = 4.0 * std::f64::consts::PI * r_a * r_a * w;
            tesserae.push(Tessera {
                position: point,
                normal: *dir,
                area,
                sphere_radius: r_a,
                atom_index: a_idx,
            });
        }
    }

    if tesserae.is_empty() {
        return Err(FerricError::General(
            "build_cavity: zero surviving tesserae (every point on every sphere was buried — \
             degenerate/overlapping geometry, or a single-atom molecule with no exposed surface)"
                .into(),
        ));
    }

    Ok(tesserae)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::Molecule;

    fn water() -> Molecule {
        Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap()
    }

    #[test]
    fn water_cavity_has_positive_tesserae_and_area() {
        let mol = water();
        let cfg = CavityConfig::default();
        let tess = build_cavity(&mol, &cfg).unwrap();
        assert!(!tess.is_empty());
        let total_area: f64 = tess.iter().map(|t| t.area).sum();
        // Sanity: total surface area of a small-molecule cavity should be a
        // handful of tens of Bohr² (roughly consistent with vdW surface area
        // of water, ~150 Ų ≈ 53 Bohr²... loosely bounded here, not an exact
        // literature match, just a sanity range).
        assert!(total_area > 10.0 && total_area < 500.0, "total_area={total_area}");
    }

    #[test]
    fn every_tessera_normal_is_unit_length() {
        let mol = water();
        let cfg = CavityConfig::default();
        let tess = build_cavity(&mol, &cfg).unwrap();
        for t in &tess {
            let n2 = t.normal[0] * t.normal[0] + t.normal[1] * t.normal[1] + t.normal[2] * t.normal[2];
            assert!((n2 - 1.0).abs() < 1e-10, "normal not unit length: {n2}");
        }
    }

    #[test]
    fn single_atom_cavity_keeps_full_sphere() {
        let mol = Molecule::parse_xyz("1\nHe\nHe 0.0 0.0 0.0\n", 0, 1).unwrap();
        let cfg = CavityConfig {
            lebedev_order: 110,
            ..CavityConfig::default()
        };
        let tess = build_cavity(&mol, &cfg).unwrap();
        assert_eq!(tess.len(), 110, "single atom: no neighbor to bury any point");
        let r = crate::radii::bondi_radius_bohr(2) * cfg.vdw_scale;
        let total_area: f64 = tess.iter().map(|t| t.area).sum();
        let expected = 4.0 * std::f64::consts::PI * r * r;
        assert!((total_area - expected).abs() / expected < 1e-8);
    }

    #[test]
    fn ghost_atom_does_not_get_a_sphere() {
        // Water + a far ghost O: the ghost must not add cavity surface.
        let xyz = "4\nwater + ghost\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n@O 0.0 0.0 100.0\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let cfg = CavityConfig::default();
        let tess = build_cavity(&mol, &cfg).unwrap();
        assert!(tess.iter().all(|t| t.atom_index != 3), "ghost atom (index 3) must not own tesserae");
    }

    #[test]
    fn empty_molecule_is_an_error() {
        let mol = Molecule::parse_xyz("0\nempty\n", 0, 0);
        // parse_xyz may itself error on 0 atoms; if it doesn't, build_cavity must.
        if let Ok(mol) = mol {
            assert!(build_cavity(&mol, &CavityConfig::default()).is_err());
        }
    }
}
