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
