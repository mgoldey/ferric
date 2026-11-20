//! Basis-aware default cutoff `r0` for terfc-attenuated MP2.
//!
//! MP2(terfc) replaces `1/r12` by `terfc(r12, r0)/r12`; the best `r0` depends
//! on the basis set because the attenuation cancels basis-set superposition
//! error. The table below holds the S66-trained optimal `r0` per basis set
//! without counterpoise correction. It is the single source read by the CLI
//! (`[mp2] att_r0`) and the Python `run_terfc_rimp2` / `run_rimp2(att_r0=)`.
//!
//! Source: M. Goldey and M. Head-Gordon, Chem. Phys. Lett. 608, 249 (2014),
//! "Convergence of attenuated second order Moller-Plesset perturbation theory
//! towards the complete basis set limit" (the chapter-6 article of M. Goldey's
//! dissertation, UC Berkeley), Table 6.2 (Dunning sets) and Table 6.4 (Pople
//! and Karlsruhe sets) of the dissertation. The calendar sets
//! (jun-/jul-/may-/apr-) of Table 6.2 are not bundled and so have no row.
//!
//! SCS-MP2(2terfc) has its own one-row table ([`SCS_2TERFC_TABLE`]): the
//! four-parameter model of dissertation chapter 5 (Eq. 5.6; J. Phys. Chem. B
//! 118, 6519 (2014)) was fitted in aug-cc-pVTZ only.
//!
//! Rows are keyed by any name `ferric_core::basis::bundled_stem` resolves, so
//! aliases (`6-31G(d)` for `6-31G*`) and case variants hit the same row. To
//! add a basis set, append a row; no other code changes.

/// One row of the default-`r0` table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerfcR0Row {
    /// A bundled basis name or alias; resolved through the basis registry.
    pub basis: &'static str,
    /// Optimal `r0` in Å.
    pub r0_angstrom: f64,
    /// Where the value comes from.
    pub source: &'static str,
}

const T62: &str = "Goldey & Head-Gordon, Chem. Phys. Lett. 608, 249 (2014); dissertation Table 6.2";
const T64: &str = "Goldey & Head-Gordon, Chem. Phys. Lett. 608, 249 (2014); dissertation Table 6.4";

/// The S66-trained optimal `r0` (Å, no counterpoise) per basis set.
pub const TERFC_R0_TABLE: &[TerfcR0Row] = &[
    TerfcR0Row {
        basis: "cc-pvdz",
        r0_angstrom: 1.55,
        source: T62,
    },
    TerfcR0Row {
        basis: "aug-cc-pvdz",
        r0_angstrom: 1.05,
        source: T62,
    },
    TerfcR0Row {
        basis: "cc-pvtz",
        r0_angstrom: 1.50,
        source: T62,
    },
    TerfcR0Row {
        basis: "aug-cc-pvtz",
        r0_angstrom: 1.35,
        source: T62,
    },
    TerfcR0Row {
        basis: "cc-pvqz",
        r0_angstrom: 1.55,
        source: T62,
    },
    TerfcR0Row {
        basis: "aug-cc-pvqz",
        r0_angstrom: 1.50,
        source: T62,
    },
    TerfcR0Row {
        basis: "6-31g*",
        r0_angstrom: 1.75,
        source: T64,
    },
    TerfcR0Row {
        basis: "6-31+g*",
        r0_angstrom: 1.45,
        source: T64,
    },
    TerfcR0Row {
        basis: "6-31++g**",
        r0_angstrom: 1.35,
        source: T64,
    },
    TerfcR0Row {
        basis: "6-311++g**",
        r0_angstrom: 1.35,
        source: T64,
    },
    TerfcR0Row {
        basis: "def2-svpd",
        r0_angstrom: 0.75,
        source: T64,
    },
    TerfcR0Row {
        basis: "def2-tzvpd",
        r0_angstrom: 1.30,
        source: T64,
    },
    TerfcR0Row {
        basis: "def2-tzvppd",
        r0_angstrom: 1.35,
        source: T64,
    },
];

/// The table row for `basis` (any registry name or alias, any case), if tabulated.
pub fn terfc_default_r0_row(basis: &str) -> Option<&'static TerfcR0Row> {
    let stem = ferric_core::basis::bundled_stem(basis)?;
    TERFC_R0_TABLE
        .iter()
        .find(|row| ferric_core::basis::bundled_stem(row.basis) == Some(stem))
}

/// The default terfc `r0` (Å) for `basis`, or `None` when it is not tabulated.
///
/// ```
/// assert_eq!(ferric_mp2::terfc_r0::terfc_default_r0_angstrom("aug-cc-pVTZ"), Some(1.35));
/// assert_eq!(ferric_mp2::terfc_r0::terfc_default_r0_angstrom("sto-3g"), None);
/// ```
pub fn terfc_default_r0_angstrom(basis: &str) -> Option<f64> {
    terfc_default_r0_row(basis).map(|r| r.r0_angstrom)
}

/// The `r0` a terfc run uses and, when it came from the table, the one line
/// the run prints naming the value and its source.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedTerfcR0 {
    /// `r0` in Å.
    pub r0_angstrom: f64,
    /// `Some(line)` when the value is the table default; `None` when explicit.
    pub note: Option<String>,
}

/// Resolve the terfc `r0`: an explicit value always wins; otherwise the table
/// value for `basis`; otherwise an error telling the user to set `r0`.
pub fn resolve_terfc_r0(basis: &str, explicit: Option<f64>) -> Result<ResolvedTerfcR0, String> {
    if let Some(r0) = explicit {
        return Ok(ResolvedTerfcR0 {
            r0_angstrom: r0,
            note: None,
        });
    }
    match terfc_default_r0_row(basis) {
        Some(row) => Ok(ResolvedTerfcR0 {
            r0_angstrom: row.r0_angstrom,
            note: Some(format!(
                "terfc r0 = {:.2} Å (default for {basis}; {})",
                row.r0_angstrom, row.source
            )),
        }),
        None => Err(format!(
            "terfc r0 has no tabulated default for basis \"{basis}\"; set r0 explicitly \
             (CLI [mp2] att_r0, Python r0= / att_r0=). Tabulated bases: {}",
            TERFC_R0_TABLE
                .iter()
                .map(|r| r.basis)
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// One row of the SCS-MP2(2terfc) parameter table: all four fitted parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scs2terfcRow {
    /// A bundled basis name or alias.
    pub basis: &'static str,
    /// Bonded attenuation distance r0(1), Å.
    pub r0_bonded_angstrom: f64,
    /// Non-bonded attenuation distance r0(2), Å.
    pub r0_nonbonded_angstrom: f64,
    /// Opposite-spin coefficient.
    pub c_os: f64,
    /// Same-spin coefficient.
    pub c_ss: f64,
    /// Where the values come from.
    pub source: &'static str,
}

/// SCS-MP2(2terfc) fits: aug-cc-pVTZ only (S66 + W4-11 training, no CP).
pub const SCS_2TERFC_TABLE: &[Scs2terfcRow] = &[Scs2terfcRow {
    basis: "aug-cc-pvtz",
    r0_bonded_angstrom: 0.75,
    r0_nonbonded_angstrom: 1.05,
    c_os: 1.27,
    c_ss: 4.05,
    source:
        "Goldey & Head-Gordon, J. Phys. Chem. B 118, 6519 (2014); dissertation Sec. 5.3, Eq. 5.6",
}];

/// The SCS-MP2(2terfc) row for `basis`, if it is tabulated.
pub fn scs_2terfc_row(basis: &str) -> Option<&'static Scs2terfcRow> {
    let stem = ferric_core::basis::bundled_stem(basis)?;
    SCS_2TERFC_TABLE
        .iter()
        .find(|row| ferric_core::basis::bundled_stem(row.basis) == Some(stem))
}

/// SCS-MP2(2terfc) parameters a run uses (Å for both r0).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedScs2terfc {
    /// r0(1), Å.
    pub r0_bonded_angstrom: f64,
    /// r0(2), Å.
    pub r0_nonbonded_angstrom: f64,
    /// Opposite-spin coefficient.
    pub c_os: f64,
    /// Same-spin coefficient.
    pub c_ss: f64,
    /// A line to print when any value came from the aTZ fit, else `None`.
    pub note: Option<String>,
}

/// Resolve the SCS-MP2(2terfc) parameters. Explicit values always win. A
/// tabulated basis fills any omitted value from its row. For an untabulated
/// basis both r0 must be explicit (an error otherwise); omitted c_os / c_ss
/// then take the aTZ-fit values and the note says so.
pub fn resolve_scs_2terfc(
    basis: &str,
    r0_bonded: Option<f64>,
    r0_nonbonded: Option<f64>,
    c_os: Option<f64>,
    c_ss: Option<f64>,
) -> Result<ResolvedScs2terfc, String> {
    let atz = &SCS_2TERFC_TABLE[0];
    let row = scs_2terfc_row(basis);
    let (r1, r2) = match (row, r0_bonded, r0_nonbonded) {
        (_, Some(a), Some(b)) => (a, b),
        (Some(r), a, b) => (
            a.unwrap_or(r.r0_bonded_angstrom),
            b.unwrap_or(r.r0_nonbonded_angstrom),
        ),
        (None, _, _) => {
            return Err(format!(
                "SCS-MP2(2terfc) is a four-parameter model fitted in aug-cc-pVTZ only \
                 (r0(1) = {} A, r0(2) = {} A, c_os = {}, c_ss = {}); basis \"{basis}\" has \
                 no fit. Set both r0_bonded and r0_nonbonded explicitly (A); c_os and c_ss \
                 are also aTZ-fit values, so set them explicitly too unless the aTZ \
                 coefficients are what you want",
                atz.r0_bonded_angstrom, atz.r0_nonbonded_angstrom, atz.c_os, atz.c_ss
            ))
        }
    };
    let used_table = row.is_some() && (r0_bonded.is_none() || r0_nonbonded.is_none());
    let cs_default = c_os.is_none() || c_ss.is_none();
    let note = (used_table || cs_default).then(|| {
        format!(
            "SCS-MP2(2terfc) parameters omitted by the user come from the aug-cc-pVTZ fit \
             ({}); basis is {basis}",
            atz.source
        )
    });
    Ok(ResolvedScs2terfc {
        r0_bonded_angstrom: r1,
        r0_nonbonded_angstrom: r2,
        c_os: c_os.unwrap_or(row.map_or(atz.c_os, |r| r.c_os)),
        c_ss: c_ss.unwrap_or(row.map_or(atz.c_ss, |r| r.c_ss)),
        note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_resolves_and_is_in_range() {
        for row in TERFC_R0_TABLE {
            assert!(
                ferric_core::basis::bundled_stem(row.basis).is_some(),
                "{} is not a bundled basis",
                row.basis
            );
            assert!((0.5..=2.0).contains(&row.r0_angstrom), "{}", row.basis);
            assert_eq!(terfc_default_r0_angstrom(row.basis), Some(row.r0_angstrom));
            assert!(!row.source.is_empty());
        }
    }

    #[test]
    fn no_two_rows_share_a_basis() {
        for (i, a) in TERFC_R0_TABLE.iter().enumerate() {
            for b in &TERFC_R0_TABLE[i + 1..] {
                assert_ne!(
                    ferric_core::basis::bundled_stem(a.basis),
                    ferric_core::basis::bundled_stem(b.basis)
                );
            }
        }
    }

    #[test]
    fn dissertation_values_are_pinned() {
        let want = [
            ("cc-pvdz", 1.55),
            ("aug-cc-pvdz", 1.05),
            ("cc-pvtz", 1.50),
            ("aug-cc-pvtz", 1.35),
            ("cc-pvqz", 1.55),
            ("aug-cc-pvqz", 1.50),
            ("6-31g*", 1.75),
            ("6-31+g*", 1.45),
            ("6-31++g**", 1.35),
            ("6-311++g**", 1.35),
            ("def2-svpd", 0.75),
            ("def2-tzvpd", 1.30),
            ("def2-tzvppd", 1.35),
        ];
        for (b, r0) in want {
            assert_eq!(terfc_default_r0_angstrom(b), Some(r0), "{b}");
        }
        assert_eq!(want.len(), TERFC_R0_TABLE.len());
    }

    #[test]
    fn aliases_and_case_hit_the_same_row() {
        assert_eq!(terfc_default_r0_angstrom("AUG-CC-PVTZ"), Some(1.35));
        assert_eq!(terfc_default_r0_angstrom("6-31G(d)"), Some(1.75));
        assert_eq!(terfc_default_r0_angstrom("6-311++G(d,p)"), Some(1.35));
    }

    #[test]
    fn untabulated_basis_errors_and_explicit_wins() {
        let e = resolve_terfc_r0("sto-3g", None).unwrap_err();
        assert!(e.contains("set r0 explicitly"), "{e}");
        let r = resolve_terfc_r0("sto-3g", Some(1.2)).unwrap();
        assert_eq!((r.r0_angstrom, r.note), (1.2, None));
        let r = resolve_terfc_r0("aug-cc-pvtz", Some(1.05)).unwrap();
        assert_eq!(r.r0_angstrom, 1.05);
        let r = resolve_terfc_r0("aug-cc-pvtz", None).unwrap();
        assert_eq!(r.r0_angstrom, 1.35);
        assert!(r.note.unwrap().contains("Table 6.2"));
    }

    #[test]
    fn scs_2terfc_table_is_the_atz_fit_only() {
        assert_eq!(SCS_2TERFC_TABLE.len(), 1);
        let r = scs_2terfc_row("AUG-CC-PVTZ").unwrap();
        assert_eq!(
            (
                r.r0_bonded_angstrom,
                r.r0_nonbonded_angstrom,
                r.c_os,
                r.c_ss
            ),
            (0.75, 1.05, 1.27, 4.05)
        );
        assert!(scs_2terfc_row("aug-cc-pvdz").is_none());
    }

    #[test]
    fn scs_2terfc_resolution_rules() {
        let d = resolve_scs_2terfc("aug-cc-pvtz", None, None, None, None).unwrap();
        assert_eq!(
            (
                d.r0_bonded_angstrom,
                d.r0_nonbonded_angstrom,
                d.c_os,
                d.c_ss
            ),
            (0.75, 1.05, 1.27, 4.05)
        );
        // untabulated, nothing explicit: hard error naming the aTZ fit
        let e = resolve_scs_2terfc("aug-cc-pvdz", None, None, None, None).unwrap_err();
        assert!(
            e.contains("aug-cc-pVTZ only") && e.contains("c_os and c_ss"),
            "{e}"
        );
        // untabulated, only one r0: still an error
        assert!(resolve_scs_2terfc("cc-pvdz", Some(0.8), None, None, None).is_err());
        // untabulated, both explicit: ok, aTZ coefficients flagged in the note
        let r = resolve_scs_2terfc("cc-pvdz", Some(0.8), Some(1.2), None, None).unwrap();
        assert_eq!((r.r0_bonded_angstrom, r.r0_nonbonded_angstrom), (0.8, 1.2));
        assert!(r.note.is_some());
        // everything explicit: no note
        let r = resolve_scs_2terfc("cc-pvdz", Some(0.8), Some(1.2), Some(1.0), Some(2.0)).unwrap();
        assert_eq!((r.c_os, r.c_ss, r.note), (1.0, 2.0, None));
    }
}
