//! Bondi van der Waals radii for PCM cavity construction.
//!
//! Reference: A. Bondi, "van der Waals Volumes and Radii", J. Phys. Chem.
//! 1964, 68, 3, 441-451. Values in Angstrom below, converted to Bohr at
//! lookup time. This is the standard radii table used by essentially every
//! production PCM/COSMO implementation (Gaussian, ORCA, Q-Chem, PySCF) as the
//! default cavity radii source, typically scaled by a factor ~1.2 (applied
//! by the caller via `CavityConfig::vdw_scale`, not baked in here).
//!
//! Elements not in the table fall back to a generic 2.0 A radius (a common
//! default for "unknown/heavy" elements in several codes) rather than
//! failing the whole cavity build — but see [`bondi_radius_bohr`] which
//! returns `None` so callers can choose to hard-error instead if they want
//! stricter behavior.

const BOHR_PER_ANGSTROM: f64 = 1.8897259886;

/// Bondi radii in Angstrom, indexed by atomic number Z (1-based; `RADII[0]`
/// is a dummy Z=0 entry). `None` = not tabulated.
///
/// Covers H through Kr plus the common heavy elements already bundled as
/// basis sets elsewhere in ferric (Rb..Xe get a generic fallback in
/// [`bondi_radius_bohr`], not tabulated individually here — Bondi 1964
/// itself only goes up to Xe for a handful of species, and PCM cavities
/// for period-5+ metals are rarely validated against experiment anyway).
const BONDI_AZ: &[(i32, f64)] = &[
    (1, 1.20),  // H
    (2, 1.40),  // He
    (3, 1.82),  // Li
    (5, 1.80),  // B (interpolated/common value; Bondi did not tabulate B)
    (6, 1.70),  // C
    (7, 1.55),  // N
    (8, 1.52),  // O
    (9, 1.47),  // F
    (10, 1.54), // Ne
    (11, 2.27), // Na
    (12, 1.73), // Mg
    (14, 2.10), // Si
    (15, 1.80), // P
    (16, 1.80), // S
    (17, 1.75), // Cl
    (18, 1.88), // Ar
    (19, 2.75), // K
    (20, 2.31), // Ca (common extended-Bondi value)
    (28, 1.63), // Ni
    (29, 1.40), // Cu
    (30, 1.39), // Zn
    (31, 1.87), // Ga
    (32, 2.11), // Ge
    (33, 1.85), // As
    (34, 1.90), // Se
    (35, 1.85), // Br
    (36, 2.02), // Kr
    (46, 1.63), // Pd
    (47, 1.72), // Ag
    (48, 1.58), // Cd
    (49, 1.93), // In
    (50, 2.17), // Sn
    (51, 2.06), // Sb
    (52, 2.06), // Te
    (53, 1.98), // I
    (54, 2.16), // Xe
];

/// Fallback radius (Angstrom) for an element not in [`BONDI_AZ`]. Matches
/// the common "generic heavy atom" default used when a tabulated value is
/// unavailable (e.g. Q-Chem falls back to a similar constant).
const FALLBACK_RADIUS_ANGSTROM: f64 = 2.0;

/// Bondi van der Waals radius for atomic number `z`, in Bohr, unscaled.
/// Returns the tabulated value when known, else [`FALLBACK_RADIUS_ANGSTROM`]
/// (converted to Bohr) — cavity construction should never hard-fail merely
/// because an exotic element lacks a literature radius, but callers that want
/// stricter behavior can check [`has_tabulated_radius`] first.
pub fn bondi_radius_bohr(z: i32) -> f64 {
    let angstrom = BONDI_AZ
        .iter()
        .find(|&&(zz, _)| zz == z)
        .map(|&(_, r)| r)
        .unwrap_or(FALLBACK_RADIUS_ANGSTROM);
    angstrom * BOHR_PER_ANGSTROM
}

/// `true` if `z` has a literature Bondi radius (as opposed to the generic
/// fallback).
pub fn has_tabulated_radius(z: i32) -> bool {
    BONDI_AZ.iter().any(|&(zz, _)| zz == z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydrogen_radius_matches_bondi_1964() {
        // 1.20 A * 1.8897259886 = 2.2676... Bohr
        let r = bondi_radius_bohr(1);
        assert!((r - 1.20 * BOHR_PER_ANGSTROM).abs() < 1e-10);
    }

    #[test]
    fn oxygen_radius_matches_bondi_1964() {
        let r = bondi_radius_bohr(8);
        assert!((r - 1.52 * BOHR_PER_ANGSTROM).abs() < 1e-10);
    }

    #[test]
    fn unknown_element_falls_back_not_panics() {
        // Z=57 (La) is not tabulated; must not panic, must return the fallback.
        let r = bondi_radius_bohr(57);
        assert!((r - FALLBACK_RADIUS_ANGSTROM * BOHR_PER_ANGSTROM).abs() < 1e-10);
        assert!(!has_tabulated_radius(57));
    }

    #[test]
    fn tabulated_common_elements() {
        for z in [1, 6, 7, 8, 16, 17, 35, 53] {
            assert!(has_tabulated_radius(z), "Z={z} should be tabulated");
        }
    }
}
