//! Molecular cavity construction: atom-centered spheres tessellated into
//! surface elements ("tesserae").
//!
//! # SWIG smooth switching function (added 2026-07-19)
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
//! This module now ports that scheme (same technique already implemented
//! for `ferric_scf::cosmo::CosmoCavity::build` — see that function's doc for
//! the PySCF `gen_surface`/eq. 3.19-3.21 derivation this mirrors):
//!
//! 1. Each atom gets a sphere of radius `vdw_scale * bondi_radius(Z)`.
//! 2. Each sphere is covered with a fixed-order Lebedev point set (reused
//!    directly from `ferric_dft::lebedev`, which is already validated for
//!    DFT quadrature).
//! 3. A point's area is scaled by `prod_{B != A} h(d_AB)`, a smooth quintic
//!    switching weight (`h`=[`switch_h`]) that is 1 far outside every other
//!    sphere and smoothly goes to 0 deep inside one — rather than the old
//!    hard binary keep/discard cut. Points whose total switching weight
//!    underflows a numerical floor (matching PySCF's `w*swf > 1e-16` keep
//!    criterion) are dropped as an optimization only, not a physics cutoff.
//! 4. Each surviving point's area is `4π·R_A² · w_i · swf` (the Lebedev
//!    weight already sums to 1 over the *unpruned* sphere).
//!
//! **Measured effect (2026-07-19, see `docs/VALIDATION.md`'s PCM row for the
//! full numbers)**: UNLIKE the earlier null result on the sibling COSMO
//! crate (<0.5% shift there), this switching function is a real, substantial
//! lever on PCM — but the outcome is a genuine trade, not a clean win.
//! Methanol's error (previously ~3-4x too NEGATIVE) shrinks to under 2x and
//! flips sign (now too WEAK), and NH3 tightens from ~8% to ~1.7%; but both
//! water points (STO-3G and cc-pVDZ), previously ~0.3%/~0% agreement with
//! PySCF, loosen to ~9-10% off. Total cavity area barely changes and moves
//! in opposite directions per molecule (water grows slightly, methanol
//! shrinks slightly) — the mechanism is a redistribution of area among
//! tesserae (many small-area tesserae now exist near former hard-cut
//! boundaries, and the self-term `S_ii = ξ·√(4π/area)` diverges as
//! area→0), not a coverage fix. Kept in the tree because it demonstrably
//! changes the physics in a direction that helps the worst case (methanol)
//! and is a prerequisite for any future PCM analytic gradient (a hard cut
//! is non-differentiable under nuclear motion) — but it does not deliver
//! uniform improvement, and none of the four test systems individually
//! reach tight (<5%) agreement across the board.
//!
//! This remains a **documented simplification** relative to full GEPOL:
//! still no added interstitial spheres at sphere-sphere-sphere intersection
//! points (GEPOL's "creation of new spheres" step) for buried/concave
//! regions. The S/D point-charge-vs-Gaussian-smeared-charge boundary-element
//! formulation in `matrices.rs` (the lever the COSMO investigation
//! quantified as ~40% there) remains an untested candidate for the residual
//! gap on methanol and water/cc-pVDZ.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::lebedev;

/// SWIG (switching/Gaussian) transition-zone shape parameter (Lange &
/// Herbert, JCP 133, 244111 (2010), eq. 3.19): the quintic smoothstep
/// `h(x) = x^3 (10 - 15x + 6x^2)`, `h(x)=0` for `x<=0`, `h(x)=1` for `x>=1`.
/// Identical to `ferric_scf::cosmo::switch_h` (duplicated rather than shared
/// across crates — see `lib.rs`'s "Naming note for a future COSMO merge" for
/// why a dedup pass is deferred).
fn switch_h(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)
    }
}

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
    /// Gaussian charge-distribution width `xi_k` for this tessera (PySCF
    /// `pcm.py::gen_surface`'s `charge_exp`), set by the LOCAL Lebedev grid
    /// density: `xi_k = XI[ng] / (r_vdw * sqrt(w_k))`, `w_k` the tessera's
    /// *unnormalized* Lebedev weight (PySCF convention: sums to `4*pi` over a
    /// sphere). Used by [`crate::matrices::build_s_d`]'s Gaussian-smeared S/D
    /// formulation; not meaningful for anything else.
    pub charge_exp: f64,
    /// Raw switching-function value at this tessera's center (`swf` in
    /// PySCF) BEFORE folding into area — needed separately by the
    /// Gaussian-smeared diagonal self-terms of both S and D
    /// (`S_kk = xi_k * sqrt(2/pi) / switch_fun_k`).
    pub switch_fun: f64,
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
/// atom, with each point's area smoothly down-weighted near a neighboring
/// sphere's boundary by a SWIG-style quintic switching function (see module
/// doc) instead of a hard keep/discard cut.
///
/// Returns `Err` if the resulting cavity has zero tesserae (e.g. an empty
/// molecule, or every point on every sphere's switching weight underflows
/// the numerical floor — degenerate geometry) since a zero-tessera cavity
/// cannot support a PCM linear solve.
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
    let n_grid = dirs.len() as f64;

    let centers: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let radii: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| crate::radii::bondi_radius_bohr(a.z) * cfg.vdw_scale)
        .collect();

    // Per-atom switching-zone geometry (PySCF gen_surface / Lange & Herbert
    // eq. 3.19-3.21, ported verbatim from `ferric_scf::cosmo::CosmoCavity::build`):
    // the transition band width R_sw scales with the LOCAL Lebedev point
    // density (sqrt(14/N)), and R_in is the radius at which the switching
    // function starts to turn on.
    let r_sw: Vec<f64> = radii.iter().map(|&r| r * (14.0 / n_grid).sqrt()).collect();
    let r_in: Vec<f64> = radii
        .iter()
        .zip(r_sw.iter())
        .map(|(&r, &rsw)| {
            let ratio = r / rsw;
            let alpha = 0.5 + ratio - (ratio * ratio - 1.0 / 28.0).sqrt();
            r - alpha * rsw
        })
        .collect();

    let xi_prefactor = crate::matrices::gaussian_xi_table(cfg.lebedev_order)?;

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
            // PySCF's `w` is the Lebedev weight in ITS convention, which sums
            // to 4*pi over a sphere (ferric's `lebedev()` sums to 1); convert
            // once here so `xi_k = XI[ng] / (r_vdw * sqrt(w_pyscf))` matches
            // `gen_surface`'s `xi = XI[ng] / (r_vdw * w**0.5)` exactly.
            let w_pyscf = w * 4.0 * std::f64::consts::PI;
            let charge_exp = xi_prefactor / (r_a * w_pyscf.sqrt());

            // Smooth switching weight: product of h(d) over every OTHER
            // real atom's sphere (own-atom factor is exactly 1).
            let mut swf = 1.0_f64;
            for (b_idx, other) in mol.atoms.iter().enumerate() {
                if b_idx == a_idx {
                    continue;
                }
                if cfg.skip_ghost_atoms && other.ghost {
                    continue;
                }
                let cb = centers[b_idx];
                let dx = point[0] - cb[0];
                let dy = point[1] - cb[1];
                let dz = point[2] - cb[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let d = (dist - r_in[b_idx]) / r_sw[b_idx];
                // Numerical tidiness (matches PySCF): clamp sub-1e-8 diffs
                // to exactly 0 before calling switch_h.
                let d = if d.abs() < 1e-8 { 0.0 } else { d };
                swf *= switch_h(d);
                if swf == 0.0 {
                    break;
                }
            }
            // Numerical floor matching PySCF's `w*swf > 1e-16` keep
            // criterion — points below this contribute negligible area and
            // would otherwise bloat the tessera count for free.
            if w * swf <= 1e-16 {
                continue;
            }

            let area = 4.0 * std::f64::consts::PI * r_a * r_a * w * swf;
            tesserae.push(Tessera {
                position: point,
                normal: *dir,
                area,
                sphere_radius: r_a,
                atom_index: a_idx,
                charge_exp,
                switch_fun: swf,
            });
        }
    }

    if tesserae.is_empty() {
        return Err(FerricError::General(
            "build_cavity: zero surviving tesserae (every point on every sphere's switching \
             weight underflowed — degenerate/overlapping geometry, or a single-atom molecule \
             with no exposed surface)"
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
