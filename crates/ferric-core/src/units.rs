//! Length units: the ONE Ångström ↔ Bohr conversion every ferric crate uses.
//!
//! Ferric works in Bohr internally and takes Ångström at its surfaces (XYZ
//! geometry, attenuation radii r₀, range-separation ω in Å⁻¹, cavity and
//! Becke radii, covalent radii). All of them convert through
//! [`ANGSTROM_TO_BOHR`](crate::units::ANGSTROM_TO_BOHR) so that a length given in Å lands in the same frame as
//! the geometry.
//!
//! The Bohr radius is the CODATA 2010 value, 0.52917721092 Å. It is PySCF's
//! `pyscf.data.nist.BOHR` (= `pyscf.lib.param.BOHR`) and mctc-lib's `aatoau`
//! (DFT-D3), which most of ferric's validation references are generated with.
//!
//! `tests/units_literal_guard.rs` fails on any other Å ↔ Bohr float literal in
//! `crates/*/src` or `crates/*/examples`.

/// The Bohr radius in Ångström (CODATA 2010; PySCF `param.BOHR`).
pub const BOHR_RADIUS_ANGSTROM: f64 = 0.529_177_210_92;

/// 1 Å in Bohr: multiply an Ångström length by this to get Bohr.
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_RADIUS_ANGSTROM;

/// 1 Bohr in Å: multiply a Bohr length by this to get Ångström. Equally, an
/// inverse length in Å⁻¹ (e.g. a range-separation ω) times this is Bohr⁻¹.
pub const BOHR_TO_ANGSTROM: f64 = BOHR_RADIUS_ANGSTROM;

/// The Hartree in joules (CODATA 2010, the set the Bohr radius above comes from).
pub const HARTREE_JOULE: f64 = 4.359_744_34e-18;

/// 1 Hartree/Bohr^3 in GPa: multiply a stress (or pressure) in Hartree/Bohr^3
/// by this to get gigapascals. Built from [`BOHR_RADIUS_ANGSTROM`] so the
/// length unit cannot disagree with the geometry's.
pub const HARTREE_PER_BOHR3_TO_GPA: f64 = {
    let a0_m = BOHR_RADIUS_ANGSTROM * 1.0e-10;
    HARTREE_JOULE / (a0_m * a0_m * a0_m) * 1.0e-9
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol::Molecule;

    /// The bits of `1.0 / 0.529_177_210_92`, the factor XYZ geometry was parsed
    /// with before this module existed. Pinning the bits (not the value to a
    /// tolerance) is what guarantees no geometry moved.
    const PRIOR_GEOMETRY_FACTOR_BITS: u64 = 0x3ffe_3c51_75f6_75d0;

    #[test]
    fn constants_are_the_codata_2010_bohr_radius() {
        assert_eq!(ANGSTROM_TO_BOHR.to_bits(), PRIOR_GEOMETRY_FACTOR_BITS);
        assert_eq!(BOHR_TO_ANGSTROM.to_bits(), 0x3fe0_ef05_0bd6_139d);
        // Not inverted: 1 Å is about 1.89 Bohr, not 0.53.
        const { assert!(ANGSTROM_TO_BOHR > 1.889 && ANGSTROM_TO_BOHR < 1.890) };
        assert!((ANGSTROM_TO_BOHR * BOHR_TO_ANGSTROM - 1.0).abs() < 4.0 * f64::EPSILON);
    }

    /// Hartree/Bohr^3 -> GPa is about 29421 (the textbook 29421.0 GPa), and a
    /// stress converts to a pressure of the SAME magnitude (not inverted, not
    /// kbar or eV/A^3).
    #[test]
    fn hartree_per_bohr3_is_29421_gpa() {
        assert!((HARTREE_PER_BOHR3_TO_GPA - 29421.02).abs() < 0.05);
        // Cross-check from eV/A^3 = 160.2177 GPa, Hartree = 27.211386 eV.
        let from_ev = 27.211_386 / (BOHR_RADIUS_ANGSTROM.powi(3)) * 160.217_66;
        assert!((HARTREE_PER_BOHR3_TO_GPA / from_ev - 1.0).abs() < 1e-6);
    }

    /// XYZ coordinates are bit-identical to what the prior private factor in
    /// `mol.rs` produced, over a spread of magnitudes and signs.
    #[test]
    fn xyz_geometry_is_bit_identical_to_the_prior_factor() {
        let prior = f64::from_bits(PRIOR_GEOMETRY_FACTOR_BITS);
        let coords = [
            (0.0, 0.757, 0.587),
            (-1.234_567_89, 17.047, -0.117_79),
            (123.456, -98.765_432_1, 3.0e-5),
        ];
        let mut xyz = format!("{}\nbits\n", coords.len());
        for (x, y, z) in coords {
            xyz.push_str(&format!("H {x} {y} {z}\n"));
        }
        let mol = Molecule::parse_xyz(&xyz, 0, 1 + coords.len() % 2).unwrap();
        for (a, (x, y, z)) in mol.atoms.iter().zip(coords) {
            assert_eq!(a.x.to_bits(), (x * prior).to_bits());
            assert_eq!(a.y.to_bits(), (y * prior).to_bits());
            assert_eq!(a.zpos.to_bits(), (z * prior).to_bits());
        }
    }
}
