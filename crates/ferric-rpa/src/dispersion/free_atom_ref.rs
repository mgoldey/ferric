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
//! hard-coded table. **ferric follows suit: `vol_free` is `None` for EVERY Z.**
//! The production TS C6 path (`ferric-cli`'s TS-C6 branch) supplies the
//! denominator from a live free-atom SCF run on the SAME integration scale
//! (same xc, same Hirshfeld quadrature) as the molecular volume — the only
//! ratio that is physically meaningful — and now hard-skips TS C6 with a
//! warning if that live SCF fails, rather than dividing by a hardcoded number.
//!
//! History (G7 verification 2026-07-17, docs/vol-free-verification.md;
//! G8 removal, docs/perf-tasks/G8-fix-vol-free-table.md): the Z=1..=18 rows
//! previously carried `Some(..)` hardcoded volumes. G7 checked all 18 against
//! ferric's own free-atom UKS/RKS-PBE + Becke-volume pipeline and found that
//! 12 of them (N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar) disagreed by
//! 11%–98% (growing with Z), and a `git log --follow` dig showed those 12
//! were never actually sourced (the original commit 007bfe8 attributed all
//! 18 to TS PRL Table I, which does not tabulate vol_free; the follow-up
//! f3ec1ca only touched {H,He,C,N,O,F,Ne}, citing a Bučko-2013 table later
//! downgraded to "could not be independently verified", commit 1907ec5). The
//! CLI already ran a scale-consistent live free-atom SCF as its PRIMARY path
//! and only used these table numbers as a last-resort, scale-mismatched
//! fallback. G8 removed that fallback entirely (extending this repo's
//! established no-silent-fallback / TS-MBD-honesty convention — see the 2026
//! -07-09 `ts_atom_params` / `ts_dynamic_polarizability` / `mbd_screen` hard
//! -error work) and, consequently, dropped the now-dead Z≤18 `Some(..)`
//! values to `None`. Nothing in the codebase reads a Z≤18 `vol_free` anymore
//! (the MBD path destructures it away; the CLI takes only the live-SCF value).
//!
//! Z=19–54 α_free/C6_free are from Gould & Bučko JCTC 12, 3603 (2016) Table 2
//! (same Chu-Dalgarno lineage as TS-PRL Table I; cross-checks vs the Z≤18 rows
//! agree <5% — see docs/superpowers/specs/refs/source-crosscheck.md). Their
//! vol_free is None (no sourced fallback), same as Z≤18 now: the live free-atom
//! SCF supplies the volume, and None refuses rather than fabricating a
//! denominator.

/// Free-atom TS reference: `(alpha_free, c6_free, vol_free)` in a.u.
/// Indexed by atomic number `z` (1..=54 covered). Returns `None` outside the
/// table.
pub fn ts_free_atom(z: usize) -> Option<(f64, f64, Option<f64>)> {
    // alpha_free (a.u.)    — TS PRL 102, 073005 (2009) Table I  [verified]
    // c6_free (a.u.)       — TS PRL 102, 073005 (2009) Table I  [verified]
    // vol_free (a.u.)      — None for EVERY Z: no sourced hardcoded free-atom
    //                        ∫ρr³dr. The live free-atom SCF (ferric-cli TS-C6
    //                        branch) supplies the volume on a scale consistent
    //                        with the molecular volume; if that SCF fails, TS C6
    //                        is skipped with a warning — no fabricated denominator.
    //                        (Z≤18 previously carried Some(..) values; G7 found 12
    //                        of 18 were never sourced and disagreed 11%–98% with a
    //                        live free-atom PBE/Becke calc, and G8 removed the
    //                        CLI's table fallback that read them — see the module
    //                        doc and docs/vol-free-verification.md.)
    let row = match z {
        1  => (4.500,    6.500,    None),  // H
        2  => (1.380,    1.460,    None),  // He
        3  => (164.200,  1387.000, None),  // Li
        4  => (38.000,   214.000,  None),  // Be
        5  => (21.000,   99.500,   None),  // B
        6  => (12.000,   46.600,   None),  // C
        7  => (7.400,    24.200,   None),  // N
        8  => (5.400,    15.600,   None),  // O
        9  => (3.800,    9.520,    None),  // F
        10 => (2.670,    6.380,    None),  // Ne
        11 => (162.700,  1556.000, None),  // Na
        12 => (71.000,   627.000,  None),  // Mg
        13 => (60.000,   528.000,  None),  // Al
        14 => (37.000,   305.000,  None),  // Si
        15 => (25.000,   185.000,  None),  // P
        16 => (19.600,   134.000,  None),  // S
        17 => (15.000,   94.600,   None),  // Cl
        18 => (11.100,   64.300,   None),  // Ar
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
        // vol_free is None for EVERY Z (G8): no sourced hardcoded free-atom
        // volume; the live free-atom SCF supplies the denominator and None
        // refuses rather than fabricating a scale-mismatched one.
        assert!(v_h.is_none(), "H vol_free must be None (no hardcoded fallback): {v_h:?}");
        let (a_c, c6_c, v_c) = ts_free_atom(6).unwrap();
        assert!((a_c - 12.0).abs() < 1e-9);
        assert!((c6_c - 46.6).abs() < 1e-9);
        assert!(v_c.is_none(), "C vol_free must be None: {v_c:?}");
        let (_, _, v_o) = ts_free_atom(8).unwrap();
        assert!(v_o.is_none(), "O vol_free must be None: {v_o:?}");
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
