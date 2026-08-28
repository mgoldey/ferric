//! Unit conversions for `MmTopology::from_amber_units`.
//!
//! AMBER-style force-field parameters are conventionally quoted in kcal/mol,
//! Ångström, and degrees. Ferric works internally in atomic units (Hartree,
//! Bohr, radians), matching every other crate in the workspace. These
//! constants convert **once**, at construction time — nothing downstream of
//! [`crate::topology::MmTopology::from_amber_units`] ever sees AMBER units.
//!
//! The literals match the ones already used elsewhere in the workspace
//! (`crates/ferric-core/src/mol.rs`'s `ANGSTROM_TO_BOHR`), so a topology built
//! from AMBER units and one built directly in a.u. agree bit-for-bit on a
//! shared geometry.

/// 1 kcal/mol in Hartree (CODATA-consistent value used throughout ferric).
pub const KCAL_PER_MOL_TO_HARTREE: f64 = 1.0 / 627.509_474;

/// 1 Å in Bohr.
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_210_92;

/// Convert degrees to radians.
#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kcal_and_angstrom_constants_are_nonzero_and_sane() {
        assert!((KCAL_PER_MOL_TO_HARTREE - 1.0 / 627.509_474).abs() < 1e-18);
        assert!((ANGSTROM_TO_BOHR - 1.0 / 0.529_177_210_92).abs() < 1e-18);
    }

    #[test]
    fn deg_to_rad_matches_known_points() {
        assert!((deg_to_rad(180.0) - std::f64::consts::PI).abs() < 1e-14);
        assert!((deg_to_rad(0.0)).abs() < 1e-14);
        assert!((deg_to_rad(90.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-14);
    }
}
