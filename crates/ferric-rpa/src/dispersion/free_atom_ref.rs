//! Free-atom reference data for the Tkatchenko-Scheffler dispersion model.
//!
//! `alpha_free` and `c6_free` (atomic units) are from Tkatchenko & Scheffler,
//! PRL 102, 073005 (2009), Table I. Those values are well-established and
//! cross-checked by multiple sources.
//!
//! `vol_free` = ∫ ρ_atom(r) r³ dr is the free-atom effective Hirshfeld volume
//! (a.u.). **This quantity is NOT tabulated in the TS paper.** In production TS
//! implementations (FHI-aims, VASP, ASE) the volume ratio v_eff/v_free is
//! computed on-the-fly from a PBE/LDA free-atom DFT run; there is no canonical
//! hard-coded table. The values below for {H,He,C,N,O,F,Ne} are believed to
//! match those DFT free-atom volumes (consistent with Bučko et al. JCTC 9, 4293
//! (2013)), but **have not been independently verified against a primary source
//! we can read**. Values for the remaining elements (Li-Ar outside that set) are
//! unverified placeholders carried over from an earlier draft.
//!
//! TODO: verify vol_free by running PBE free-atom calculations in PySCF or GPAW
//! and computing ∫ ρ(r) r³ dr on the resulting densities.

/// Free-atom TS reference: `(alpha_free, c6_free, vol_free)` in a.u.
/// Indexed by atomic number `z` (1..=18 covered). Returns `None` outside the
/// table.
pub fn ts_free_atom(z: usize) -> Option<(f64, f64, f64)> {
    // alpha_free (a.u.)    — TS PRL 102, 073005 (2009) Table I  [verified]
    // c6_free (a.u.)       — TS PRL 102, 073005 (2009) Table I  [verified]
    // vol_free (a.u.)      — believed to match PBE free-atom ∫ρr³dr;
    //                        unverified for all elements (see module doc)
    let row = match z {
        1  => (4.500,    6.500,    9.149),  // H
        2  => (1.380,    1.460,    4.711),  // He
        3  => (164.200,  1387.000, 91.96),  // Li   (vol: Bučko S1)
        4  => (38.000,   214.000,  61.50),  // Be
        5  => (21.000,   99.500,   49.18),  // B
        6  => (12.000,   46.600,   34.054), // C
        7  => (7.400,    24.200,   25.097), // N
        8  => (5.400,    15.600,   19.750), // O
        9  => (3.800,    9.520,    15.746), // F
        10 => (2.670,    6.380,    12.443), // Ne
        11 => (162.700,  1556.000, 100.5),  // Na
        12 => (71.000,   627.000,  91.0),   // Mg
        13 => (60.000,   528.000,  86.0),   // Al
        14 => (37.000,   305.000,  60.0),   // Si
        15 => (25.000,   185.000,  49.0),   // P
        16 => (19.600,   134.000,  41.50),  // S
        17 => (15.000,   94.600,   34.50),  // Cl
        18 => (11.100,   64.300,   28.90),  // Ar
        _ => return None,
    };
    Some(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_free_atoms_present() {
        let (a_h, c6_h, v_h) = ts_free_atom(1).unwrap();
        assert!((a_h - 4.5).abs() < 1e-9, "H alpha_free wrong: {a_h}");
        assert!((c6_h - 6.5).abs() < 1e-9, "H C6_free wrong: {c6_h}");
        // vol_free: believed correct for {H,He,C,N,O,F,Ne}, unverified elsewhere
        assert!((v_h - 9.149).abs() < 1e-3, "H vol_free wrong: {v_h}");
        let (a_c, c6_c, v_c) = ts_free_atom(6).unwrap();
        assert!((a_c - 12.0).abs() < 1e-9);
        assert!((c6_c - 46.6).abs() < 1e-9);
        assert!((v_c - 34.054).abs() < 1e-3, "C vol_free wrong: {v_c}");
        let (_, _, v_o) = ts_free_atom(8).unwrap();
        assert!((v_o - 19.750).abs() < 1e-3, "O vol_free wrong: {v_o}");
        assert!(ts_free_atom(200).is_none(), "out-of-table should be None");
    }
}
