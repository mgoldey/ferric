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
//! hard-coded table.
//!
//! Z=1..=18 `vol_free` was independently verified 2026-07-17 against ferric's
//! own free-atom UKS/RKS-PBE + Becke-volume pipeline (see
//! `crates/ferric-rpa/tests/free_atom_volumes_pbe.rs` and
//! `docs/vol-free-verification.md` for the full table + methodology). Result:
//! **H, He, Li, Be, B, C agree (<10%)** with the independently computed
//! value and are considered verified. **N, O, F, Ne, Na, Mg, Al, Si, P, S,
//! Cl, Ar disagree significantly (11%-98%, growing with Z)** with the
//! independently computed PBE/Becke value and are flagged suspect — do NOT
//! treat them as verified; see docs/vol-free-verification.md for the full
//! per-element comparison and the disagreement pattern. Per-element notes
//! below reflect this. No table value has been changed based on this
//! verification pass — disagreements are flagged for human review, not
//! silently corrected (this table feeds production TS C6 numbers).
//!
//! Z=19–54 α_free/C6_free are from Gould & Bučko JCTC 12, 3603 (2016) Table 2
//! (same Chu-Dalgarno lineage as TS-PRL Table I; cross-checks vs the Z≤18 rows
//! agree <5% — see docs/superpowers/specs/refs/source-crosscheck.md). Their
//! vol_free is None (no sourced fallback): the live free-atom SCF supplies the
//! volume, and None refuses rather than fabricating a denominator.

/// Free-atom TS reference: `(alpha_free, c6_free, vol_free)` in a.u.
/// Indexed by atomic number `z` (1..=54 covered). Returns `None` outside the
/// table.
pub fn ts_free_atom(z: usize) -> Option<(f64, f64, Option<f64>)> {
    // alpha_free (a.u.)    — TS PRL 102, 073005 (2009) Table I  [verified]
    // c6_free (a.u.)       — TS PRL 102, 073005 (2009) Table I  [verified]
    // vol_free (a.u.)      — Some(..) = sourced free-atom ∫ρr³dr fallback;
    //                        None = no sourced fallback (live SCF must supply it,
    //                        else the TS C6 refuses — no fabricated denominator).
    let row = match z {
        // vol_free verified 2026-07-17 vs ferric free-atom PBE/Becke pipeline,
        // <10% agreement (see docs/vol-free-verification.md).
        1  => (4.500,    6.500,    Some(9.149)),  // H    vol: verified (PBE/Becke -4.9%)
        2  => (1.380,    1.460,    Some(4.711)),  // He   vol: verified (PBE/Becke -7.5%)
        3  => (164.200,  1387.000, Some(91.96)),  // Li   vol: Bučko S1; verified (PBE/Becke -1.8%)
        4  => (38.000,   214.000,  Some(61.50)),  // Be   vol: verified (PBE/Becke -1.4%)
        5  => (21.000,   99.500,   Some(49.18)),  // B    vol: verified (PBE/Becke +1.7%)
        6  => (12.000,   46.600,   Some(34.054)), // C    vol: verified (PBE/Becke +7.8%)
        // vol_free FLAGGED SUSPECT 2026-07-17: disagrees >=10% with ferric's
        // free-atom PBE/Becke value (see docs/vol-free-verification.md). NOT
        // edited pending human review — do not treat as verified.
        7  => (7.400,    24.200,   Some(25.097)), // N    vol: SUSPECT (PBE/Becke +10.8%)
        8  => (5.400,    15.600,   Some(19.750)), // O    vol: SUSPECT (PBE/Becke +19.5%)
        9  => (3.800,    9.520,    Some(15.746)), // F    vol: SUSPECT (PBE/Becke +22.8%)
        10 => (2.670,    6.380,    Some(12.443)), // Ne   vol: SUSPECT (PBE/Becke +28.7%)
        11 => (162.700,  1556.000, Some(100.5)),  // Na   vol: SUSPECT (PBE/Becke +13.4%)
        12 => (71.000,   627.000,  Some(91.0)),   // Mg   vol: SUSPECT (PBE/Becke +15.9%)
        13 => (60.000,   528.000,  Some(86.0)),   // Al   vol: SUSPECT (PBE/Becke +42.5%)
        14 => (37.000,   305.000,  Some(60.0)),   // Si   vol: SUSPECT (PBE/Becke +73.1%)
        15 => (25.000,   185.000,  Some(49.0)),   // P    vol: SUSPECT (PBE/Becke +76.8%)
        16 => (19.600,   134.000,  Some(41.50)),  // S    vol: SUSPECT (PBE/Becke +84.9%)
        17 => (15.000,   94.600,   Some(34.50)),  // Cl   vol: SUSPECT (PBE/Becke +92.0%)
        18 => (11.100,   64.300,   Some(28.90)),  // Ar   vol: SUSPECT (PBE/Becke +98.5%)
        // Z=19–54: alpha_free/c6_free from Gould & Bučko JCTC 12, 3603 (2016)
        // Table 2 (neutral atoms), a.u. — see refs/gould-bucko-2016-table2-neutral.txt.
        // vol_free = None: no sourced free-atom volume for Z>18; the live free-atom
        // SCF (cli main.rs) supplies it, and None refuses if that path fails rather
        // than dividing by a fabricated volume. Chu04 C6 alt noted where spread >5%.
        19 => (290.0,  3910.0, None),  // K
        20 => (160.0,  2230.0, None),  // Ca
        21 => (123.0,  1570.0, None),  // Sc   Chu04 C6=1383 (~12%)
        22 => (102.0,  1200.0, None),  // Ti   Chu04 C6=1044 (~13%)
        23 => (87.3,   955.0,  None),  // V    Chu04 C6=832  (~13%)
        24 => (78.4,   709.0,  None),  // Cr   Chu04 C6=602  (~15%)
        25 => (66.8,   635.0,  None),  // Mn   Chu04 C6=552  (~13%)
        26 => (60.4,   548.0,  None),  // Fe   Chu04 C6=482  (~12%)
        27 => (53.9,   461.0,  None),  // Co   Chu04 C6=408  (~11%)
        28 => (48.4,   393.0,  None),  // Ni   Chu04 C6=373  (~5%)
        29 => (41.7,   264.0,  None),  // Cu
        30 => (38.4,   276.0,  None),  // Zn
        31 => (52.1,   456.0,  None),  // Ga
        32 => (40.2,   365.0,  None),  // Ge
        33 => (29.6,   260.0,  None),  // As   Chu04 C6=246  (~5%)
        34 => (26.2,   233.0,  None),  // Se   Chu04 C6=210  (~10%)
        35 => (21.6,   187.0,  None),  // Br   Chu04 C6=162  (~13%)
        36 => (16.8,   136.0,  None),  // Kr
        37 => (317.0,  4660.0, None),  // Rb
        38 => (198.0,  3230.0, None),  // Sr
        39 => (163.0,  2600.0, None),  // Y
        40 => (112.0,  1360.0, None),  // Zr
        41 => (97.9,   1140.0, None),  // Nb
        42 => (87.1,   1030.0, None),  // Mo
        43 => (79.6,   939.0,  None),  // Tc
        44 => (72.3,   809.0,  None),  // Ru
        45 => (66.4,   708.0,  None),  // Rh
        46 => (61.7,   628.0,  None),  // Pd   (Chu04/ASE 158 is Ruiz12 in-molecular, NOT free-atom)
        47 => (46.2,   341.0,  None),  // Ag
        48 => (46.7,   405.0,  None),  // Cd
        49 => (62.1,   643.0,  None),  // In
        50 => (60.0,   715.0,  None),  // Sn
        51 => (44.0,   504.0,  None),  // Sb
        52 => (40.0,   471.0,  None),  // Te   Chu04 C6=445  (~6%)
        53 => (33.6,   389.0,  None),  // I
        54 => (27.2,   302.0,  None),  // Xe
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
        // vol_free: Some(..) for Z≤18 (believed correct for {H,He,C,N,O,F,Ne})
        assert!((v_h.unwrap() - 9.149).abs() < 1e-3, "H vol_free wrong: {v_h:?}");
        let (a_c, c6_c, v_c) = ts_free_atom(6).unwrap();
        assert!((a_c - 12.0).abs() < 1e-9);
        assert!((c6_c - 46.6).abs() < 1e-9);
        assert!((v_c.unwrap() - 34.054).abs() < 1e-3, "C vol_free wrong: {v_c:?}");
        let (_, _, v_o) = ts_free_atom(8).unwrap();
        assert!((v_o.unwrap() - 19.750).abs() < 1e-3, "O vol_free wrong: {v_o:?}");
        assert!(ts_free_atom(200).is_none(), "out-of-table should be None");
    }

    #[test]
    fn heavy_z_rows_present_gould_bucko() {
        // Gould & Bučko JCTC 12, 3603 (2016) Table 2 (neutral atoms), a.u.
        let (a_ge, c6_ge, v_ge) = ts_free_atom(32).unwrap(); // Ge
        assert!((a_ge - 40.2).abs() < 1e-9, "Ge alpha: {a_ge}");
        assert!((c6_ge - 365.0).abs() < 1e-9, "Ge C6: {c6_ge}");
        assert!(v_ge.is_none(), "Ge vol_free must be None (no sourced fallback)");

        let (a_br, c6_br, v_br) = ts_free_atom(35).unwrap(); // Br
        assert!((a_br - 21.6).abs() < 1e-9, "Br alpha: {a_br}");
        assert!((c6_br - 187.0).abs() < 1e-9, "Br C6: {c6_br}");
        assert!(v_br.is_none(), "Br vol_free must be None");

        // Range endpoints present.
        assert!(ts_free_atom(19).is_some(), "K (Z=19) must be present");
        assert!(ts_free_atom(54).is_some(), "Xe (Z=54) must be present");
        // Still refuses genuinely-absent elements above the added range.
        assert!(ts_free_atom(55).is_none(), "Cs (Z=55) not added -> None");
    }
}
