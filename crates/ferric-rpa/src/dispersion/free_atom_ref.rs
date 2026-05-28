//! Free-atom reference data for the Tkatchenko-Scheffler dispersion model.
//!
//! Values (atomic units) from Tkatchenko & Scheffler, PRL 102, 073005 (2009),
//! Table I (free-atom α and C6), and Chu & Dalgarno, JCP 121, 4083 (2004) for
//! the static polarizabilities. `vol_free` is the published free-atom effective
//! volume (a.u.); the TS volume ratio is v_eff / vol_free, where v_eff is the
//! in-molecule Becke/Hirshfeld effective volume.

/// Free-atom TS reference: `(alpha_free, c6_free, vol_free)` in a.u.
/// Indexed by atomic number `z` (1..=18 covered). Returns `None` outside the
/// table.
pub fn ts_free_atom(z: usize) -> Option<(f64, f64, f64)> {
    // alpha_free (a.u.), c6_free (a.u., homonuclear), vol_free (a.u.)
    let row = match z {
        1 => (4.500, 6.500, 8.451),    // H
        2 => (1.380, 1.460, 1.751),    // He
        3 => (164.200, 1387.000, 91.96), // Li
        4 => (38.000, 214.000, 61.50), // Be
        5 => (21.000, 99.500, 49.18),  // B
        6 => (12.000, 46.600, 24.670), // C
        7 => (7.400, 24.200, 17.001),  // N
        8 => (5.400, 15.600, 13.071),  // O
        9 => (3.800, 9.520, 9.500),    // F
        10 => (2.670, 6.380, 7.604),   // Ne
        11 => (162.700, 1556.000, 100.5), // Na
        12 => (71.000, 627.000, 91.0), // Mg
        13 => (60.000, 528.000, 86.0), // Al
        14 => (37.000, 305.000, 60.0), // Si
        15 => (25.000, 185.000, 49.0), // P
        16 => (19.600, 134.000, 41.50), // S
        17 => (15.000, 94.600, 34.50), // Cl
        18 => (11.100, 64.300, 28.90), // Ar
        _ => return None,
    };
    Some(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_free_atoms_present() {
        let (a_h, c6_h, _v_h) = ts_free_atom(1).unwrap();
        assert!((a_h - 4.5).abs() < 1e-9, "H alpha_free wrong: {a_h}");
        assert!((c6_h - 6.5).abs() < 1e-9, "H C6_free wrong: {c6_h}");
        let (a_c, c6_c, _v_c) = ts_free_atom(6).unwrap();
        assert!((a_c - 12.0).abs() < 1e-9);
        assert!((c6_c - 46.6).abs() < 1e-9);
        assert!(ts_free_atom(200).is_none(), "out-of-table should be None");
    }
}
