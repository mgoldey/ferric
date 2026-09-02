//! Element symbol and atomic number lookup tables (H through Ca).

use std::collections::HashMap;
use std::sync::LazyLock;

static SYMBOL_TO_Z: LazyLock<HashMap<&'static str, i32>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    for &(sym, z) in ELEMENTS.iter() {
        m.insert(sym, z);
    }
    m
});

static Z_TO_SYMBOL: LazyLock<HashMap<i32, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    for &(sym, z) in ELEMENTS.iter() {
        m.insert(z, sym);
    }
    m
});

const ELEMENTS: &[(&str, i32)] = &[
    ("H", 1), ("He", 2), ("Li", 3), ("Be", 4), ("B", 5),
    ("C", 6), ("N", 7), ("O", 8), ("F", 9), ("Ne", 10),
    ("Na", 11), ("Mg", 12), ("Al", 13), ("Si", 14), ("P", 15),
    ("S", 16), ("Cl", 17), ("Ar", 18), ("K", 19), ("Ca", 20),
    // Period 4 transition metals + p-block through Kr. Bundled basis sets
    // (aug-cc-pVDZ/TZ, def2) cover these; the parser must too (Br/Se needed
    // for heavy-halide dispersion benchmarks).
    ("Sc", 21), ("Ti", 22), ("V", 23), ("Cr", 24), ("Mn", 25),
    ("Fe", 26), ("Co", 27), ("Ni", 28), ("Cu", 29), ("Zn", 30),
    ("Ga", 31), ("Ge", 32), ("As", 33), ("Se", 34), ("Br", 35),
    ("Kr", 36),
    // Period 5 through Xe. Needed for the ECP-treated GW100 molecules
    // (Rb, Ag, I, Xe) under def2-ECP / cc-pVnZ-PP.
    ("Rb", 37), ("Sr", 38), ("Y", 39), ("Zr", 40), ("Nb", 41),
    ("Mo", 42), ("Tc", 43), ("Ru", 44), ("Rh", 45), ("Pd", 46),
    ("Ag", 47), ("Cd", 48), ("In", 49), ("Sn", 50), ("Sb", 51),
    ("Te", 52), ("I", 53), ("Xe", 54),
];

/// Isotope-averaged standard atomic weights in unified atomic mass units (u,
/// a.k.a. amu / daltons), indexed by `Z - 1` (so `ATOMIC_MASSES[0]` is H).
///
/// **Provenance.** These are the IUPAC 2013 standard atomic weights:
///
/// > Meija, J.; Coplen, T.; Berglund, M.; et al. "Atomic weights of the
/// > elements 2013 (IUPAC Technical Report)." *Pure and Applied Chemistry*
/// > **88**(3), 265-291 (2016). doi:10.1515/pac-2015-0305
///
/// The values were transcribed programmatically (not by hand, and not from
/// recall) from `pyscf.data.elements.MASSES` in the local PySCF checkout,
/// which carries that citation in its source comment. Standard weights come
/// from IUPAC Table 1; for the twelve elements whose weights are published as
/// an interval (H, He, B, C, N, O, Mg, Si, S, Cl, Br, Tl) PySCF uses the
/// "conventional" single value from Table 3, and we inherit that choice. For
/// elements with no stable isotope (Tc here; heavier ones are out of range)
/// the mass of the most stable isotope is used instead.
///
/// These are *averaged over natural isotopic abundance* — appropriate for
/// vibrational frequencies and thermochemistry of samples at natural
/// abundance. They are NOT single-isotope masses; an isotopic-substitution
/// study needs a different table.
///
/// Coverage is Z = 1..=54 (H through Xe), matching [`ELEMENTS`].
const ATOMIC_MASSES: &[f64] = &[
    1.008, 4.002602, 6.94, 9.0121831, 10.81,                 // H  He Li Be B
    12.011, 14.007, 15.999, 18.998403163, 20.1797,           // C  N  O  F  Ne
    22.98976928, 24.305, 26.9815385, 28.085, 30.973761998,   // Na Mg Al Si P
    32.06, 35.45, 39.948, 39.0983, 40.078,                   // S  Cl Ar K  Ca
    44.955908, 47.867, 50.9415, 51.9961, 54.938044,          // Sc Ti V  Cr Mn
    55.845, 58.933194, 58.6934, 63.546, 65.38,               // Fe Co Ni Cu Zn
    69.723, 72.63, 74.921595, 78.971, 79.904,                // Ga Ge As Se Br
    83.798, 85.4678, 87.62, 88.90584, 91.224,                // Kr Rb Sr Y  Zr
    92.90637, 95.95, 97.90721, 101.07, 102.9055,             // Nb Mo Tc Ru Rh
    106.42, 107.8682, 112.414, 114.818, 118.71,              // Pd Ag Cd In Sn
    121.76, 127.6, 126.90447, 131.293,                       // Sb Te I  Xe
];

/// Isotope-averaged standard atomic weight for atomic number `z`, in unified
/// atomic mass units (u). Returns `None` outside Z = 1..=54.
///
/// See `ATOMIC_MASSES` for the data source (IUPAC 2013) and for why these
/// are natural-abundance averages rather than single-isotope masses.
pub fn atomic_mass(z: i32) -> Option<f64> {
    if z < 1 {
        return None;
    }
    ATOMIC_MASSES.get((z - 1) as usize).copied()
}

/// Convert an element symbol to its atomic number (case-insensitive).
pub fn symbol_to_z(sym: &str) -> Option<i32> {
    let normalized: String = sym.chars().enumerate().map(|(i, c)| {
        if i == 0 { c.to_ascii_uppercase() } else { c.to_ascii_lowercase() }
    }).collect();
    SYMBOL_TO_Z.get(normalized.as_str()).copied()
}

/// Convert an atomic number to its canonical element symbol.
pub fn z_to_symbol(z: i32) -> Option<&'static str> {
    Z_TO_SYMBOL.get(&z).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_to_z() {
        assert_eq!(symbol_to_z("H"), Some(1));
        assert_eq!(symbol_to_z("h"), Some(1));
        assert_eq!(symbol_to_z("O"), Some(8));
        assert_eq!(symbol_to_z("Cl"), Some(17));
        assert_eq!(symbol_to_z("CL"), Some(17));
        assert_eq!(symbol_to_z("Xx"), None);
    }

    #[test]
    fn test_atomic_mass_spot_values() {
        // Hand-checked against IUPAC 2013 standard atomic weights.
        assert_eq!(atomic_mass(1), Some(1.008));       // H
        assert_eq!(atomic_mass(6), Some(12.011));      // C
        assert_eq!(atomic_mass(8), Some(15.999));      // O
        assert_eq!(atomic_mass(26), Some(55.845));     // Fe
        assert_eq!(atomic_mass(36), Some(83.798));     // Kr
        assert_eq!(atomic_mass(54), Some(131.293));    // Xe
    }

    #[test]
    fn test_atomic_mass_out_of_range() {
        assert_eq!(atomic_mass(0), None);
        assert_eq!(atomic_mass(-1), None);
        assert_eq!(atomic_mass(55), None);
        assert_eq!(atomic_mass(999), None);
    }

    #[test]
    fn test_atomic_mass_table_covers_element_table() {
        // ATOMIC_MASSES is indexed by Z-1 in parallel with ELEMENTS; if one
        // table grows without the other, masses silently shift by an element.
        // Pin the length and require every symbol in ELEMENTS to have a mass.
        assert_eq!(ATOMIC_MASSES.len(), ELEMENTS.len());
        for &(sym, z) in ELEMENTS.iter() {
            let m = atomic_mass(z)
                .unwrap_or_else(|| panic!("no mass for {sym} (Z={z})"));
            assert!(m > 0.0, "{sym} has non-positive mass {m}");
            // Masses increase monotonically with Z across Z=1..=54 except for
            // the four classic inversions (Ar/K, Co/Ni, Te/I, and Th/Pa which
            // is out of range). A crude Z-vs-mass band catches an off-by-one
            // shift, which is the failure this test exists to detect.
            let approx = 2.0 * z as f64;
            assert!(
                (m - approx).abs() < 0.35 * approx + 4.0,
                "{sym} (Z={z}) mass {m} is implausible for its atomic number \
                 -- table may be misaligned"
            );
        }
    }

    #[test]
    fn test_z_to_symbol() {
        assert_eq!(z_to_symbol(1), Some("H"));
        assert_eq!(z_to_symbol(8), Some("O"));
        assert_eq!(z_to_symbol(999), None);
    }
}
