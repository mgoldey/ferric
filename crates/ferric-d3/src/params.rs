//! Becke-Johnson damping parameters, per exchange-correlation functional.
//!
//! Every D3(BJ) number depends on four fitted parameters `(s6, s8, a1, a2)`
//! that are specific to the functional the correction is being added to.
//! Applying PBE's parameters to a B3LYP energy is not a small error: it is
//! using a correction fitted to a different amount of missing dispersion.
//!
//! Every row equals simple-dftd3 1.6.0's rational-damping table, and each
//! row's comment names the paper upstream cites for it. `tests/params_snapshot.rs`
//! asserts the whole table against a checked-in snapshot of upstream
//! (`testdata/d3bj_params_snapshot.json`, from
//! `scripts/gen_d3bj_param_snapshot.py`). Functionals with no published
//! D3(BJ) fit (M06-2X) are refused by name.

use ferric_core::FerricError;

/// The four Becke-Johnson rational-damping parameters, plus the scale `s9`
/// of the Axilrod-Teller-Muto three-body term.
///
/// ```text
/// E = - sum_{A<B} [ s6 C6/(R^6 + f^6) + s8 C8/(R^8 + f^8) ],  f = a1 R0 + a2
///     + s9 * E_ATM
/// ```
///
/// `s9 = 0.0` switches the three-body term off and reproduces the two-body
/// energy and gradient bit for bit; `s9 = 1.0` is "D3(BJ)-ATM". The three-body
/// damping does not use `a1`/`a2` (see the crate docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct D3Params {
    /// Scaling of the `C6/R^6` term. Almost always 1.0 for a non-double-hybrid.
    pub s6: f64,
    /// Scaling of the `C8/R^8` term.
    pub s8: f64,
    /// Slope of the BJ damping radius.
    pub a1: f64,
    /// Offset of the BJ damping radius, in Bohr.
    pub a2: f64,
    /// Scaling of the Axilrod-Teller-Muto three-body term. `0.0` = off.
    pub s9: f64,
}

/// D3(BJ) parameters for a named exchange-correlation functional.
///
/// The lookup is case-insensitive and tolerates the `-`/`_` spelling
/// differences between ferric's functional names and the D3 literature's.
///
/// The returned set has `s9 = 0.0` (three-body term OFF). That matches the
/// reference implementations for a named functional: simple-dftd3's
/// `load_rational_damping(method, atm=False)` (the Python
/// `RationalDampingParam(method=...)` default), its CLI without `--atm`, and
/// Grimme's original `dftd3` program without `-abc`. Set `s9 = 1.0` on the
/// result for D3(BJ)-ATM.
///
/// Returns an error -- never a default or a zeroed parameter set -- for a
/// functional that has no published D3(BJ) fit. Silently substituting some
/// other functional's parameters would produce a plausible-looking number that
/// is not the D3(BJ) correction for the functional the user asked for.
pub fn d3bj_params_for_functional(name: &str) -> Result<D3Params, FerricError> {
    let key = normalise(name);
    if let Some(reason) = REFUSED.iter().find(|(k, _)| *k == key).map(|(_, r)| r) {
        return Err(FerricError::General(format!(
            "no published D3(BJ) damping parameters for functional '{name}': {reason} \
             Refusing to invent or substitute a parameter set."
        )));
    }
    TABLE
        .iter()
        .find(|e| e.names.contains(&key.as_str()))
        .map(|e| D3Params {
            s6: e.s6,
            s8: e.s8,
            a1: e.a1,
            a2: e.a2,
            s9: 0.0,
        })
        .ok_or_else(|| {
            FerricError::General(format!(
                "no published D3(BJ) damping parameters for functional '{name}'. \
                 Supply them explicitly (s6/s8/a1/a2) or pick a functional with a \
                 published fit. Refusing to substitute another functional's \
                 parameters, which would silently change the correction."
            ))
        })
}

/// Every normalised name [`d3bj_params_for_functional`] resolves, aliases
/// included, in table order. `tests/params_snapshot.rs` compares each one to
/// a snapshot of simple-dftd3's table, so a name added here without a matching
/// upstream entry fails that test.
pub fn d3bj_tabulated_functionals() -> Vec<&'static str> {
    TABLE.iter().flat_map(|e| e.names.iter().copied()).collect()
}

/// Lowercase with `-`, `_` and spaces removed: the lookup key.
fn normalise(name: &str) -> String {
    name.to_ascii_lowercase().replace(['-', '_', ' '], "")
}

/// One row of the table: the normalised names that resolve to it and its
/// `(s6, s8, a1, a2)`.
struct Entry {
    names: &'static [&'static str],
    s6: f64,
    s8: f64,
    a1: f64,
    a2: f64,
}

const fn row(names: &'static [&'static str], s6: f64, s8: f64, a1: f64, a2: f64) -> Entry {
    Entry {
        names,
        s6,
        s8,
        a1,
        a2,
    }
}

/// The D3(BJ) table. Every row equals simple-dftd3 1.6.0's rational-damping
/// entry for the same functional (pinned by `tests/params_snapshot.rs`
/// against `testdata/d3bj_params_snapshot.json`, written by
/// `scripts/gen_d3bj_param_snapshot.py`). Sources are the ones upstream cites.
const TABLE: &[Entry] = &[
    // Grimme, Ehrlich, Goerigk, JCC 32, 1456 (2011), doi 10.1002/jcc.21759.
    row(&["blyp"], 1.0, 2.6996, 0.4298, 4.2359),
    row(&["bp86", "bp"], 1.0, 3.2822, 0.3946, 4.8516),
    row(&["b97d"], 1.0, 2.2609, 0.5545, 3.2297),
    row(&["pbe"], 1.0, 0.7875, 0.4289, 4.4407),
    row(&["pbe0"], 1.0, 1.2177, 0.4145, 4.8593),
    row(&["revpbe"], 1.0, 2.3550, 0.5238, 3.5016),
    row(&["b3lyp"], 1.0, 1.9889, 0.3981, 4.4211),
    row(&["tpss"], 1.0, 1.9435, 0.4535, 4.4752),
    row(&["tpss0"], 1.0, 1.2576, 0.3768, 4.5865),
    // Goerigk, Grimme, PCCP 13, 6670 (2011), doi 10.1039/c0cp02984j.
    row(&["b2plyp"], 0.64, 0.9147, 0.3065, 5.0570),
    row(&["bhlyp", "bhandhlyp"], 1.0, 1.0354, 0.2793, 4.9615),
    // Brandenburg, Bates, Sun, Perdew, PRB 94, 115144 (2016),
    // doi 10.1103/physrevb.94.115144.
    row(&["scan"], 1.0, 0.0000, 0.5380, 5.4200),
    // Ehlert, Huniar, Ning, Furness, Sun, Grimme, JCP 154, 061101 (2021),
    // doi 10.1063/5.0041008 (the r2SCAN-D4 paper's D3(BJ) fit).
    row(&["r2scan"], 1.0, 0.78981345, 0.49484001, 5.73083694),
    // omegaB97X-D3(BJ): Najibi, Goerigk, JCTC 14, 5725 (2018),
    // doi 10.1021/acs.jctc.8b00842. That paper fitted D3(BJ) to replace the
    // VV10 term of omegaB97X-V (Mardirossian, Head-Gordon, PCCP 16, 9904
    // (2014)), so this row belongs to the semilocal + exact-exchange part of
    // omegaB97X-V (libxc HYB_GGA_XC_WB97X_V without VV10), NOT to the 2008
    // omegaB97X of libxc HYB_GGA_XC_WB97X. Najibi, Goerigk, JCC 41, 2562
    // (2020), doi 10.1002/jcc.26411, cites the 2018 paper for omegaB97X-D3(BJ)
    // (its ref. 11). The numbers are simple-dftd3 1.6.0's `wb97x` d3.bj entry,
    // which upstream carries without a DOI ("TODO: find reference" in
    // src/dftd3/param.f90); its two neighbours from the same fit, `wb97m` and
    // `b97m`, carry 10.1021/acs.jctc.8b00842. The 2018 SI table itself was not
    // read here (paywalled), so the numbers rest on upstream.
    // Distinct from omegaB97X-D3 = omegaB97X-D3(0): Lin, Li, Mao, Chai, JCTC
    // 9, 263 (2013), doi 10.1021/ct300715s, a reparametrised functional fitted
    // with ZERO damping (upstream `wb97x` d3.zero: rs6 1.281, s8 1.0, rs8
    // 1.094), and from omegaB97X-D (Chai, Head-Gordon, PCCP 10, 6615 (2008)),
    // whose dispersion term is not D3 at all.
    row(&["wb97x"], 1.0, 0.2641, 0.0000, 5.4959),
];

/// Names refused BY NAME, with the reason, because the literature has no
/// D3(BJ) fit for them. A refused name must not also appear in [`TABLE`].
const REFUSED: &[(&str, &str)] = &[(
    "m062x",
    "M06-2X has a zero-damping D3(0) fit (Goerigk, Grimme, PCCP 13, 6670 (2011)) \
     but no rational-damping D3(BJ) fit; simple-dftd3 has no D3(BJ) entry for it.",
)];

#[cfg(test)]
mod tests {
    use super::*;

    /// Name normalisation must be genuinely case- and separator-insensitive,
    /// because ferric's own functional names ("PBE", "B3LYP") and the D3
    /// literature's ("b3-lyp") differ.
    #[test]
    fn lookup_is_case_and_separator_insensitive() {
        let want = d3bj_params_for_functional("b3lyp").unwrap();
        for spelling in ["B3LYP", "B3-LYP", "b3_lyp", "B3 LYP", "b3LyP"] {
            assert_eq!(
                d3bj_params_for_functional(spelling).unwrap(),
                want,
                "spelling {spelling:?} must resolve to the same parameters"
            );
        }
    }

    /// An unknown functional must ERROR, never fall back to a default.
    #[test]
    fn unknown_functional_errors() {
        let r = d3bj_params_for_functional("not-a-functional");
        assert!(r.is_err(), "unknown functional must error, got {r:?}");
        // ... and the message must name the functional, so the user can act.
        let msg = format!("{:?}", r.unwrap_err());
        assert!(
            msg.contains("not-a-functional"),
            "the error must name the functional it rejected: {msg}"
        );
    }

    /// M06-2X has no D3(BJ) fit anywhere in the literature: requesting it
    /// must error by name, and say why, in every spelling.
    #[test]
    fn m06_2x_is_refused_by_name() {
        for spelling in ["M06-2X", "m062x", "M06_2X"] {
            let r = d3bj_params_for_functional(spelling);
            let msg = format!("{:?}", r.expect_err("M06-2X must be refused"));
            assert!(msg.contains(spelling) && msg.contains("D3(0)"), "{msg}");
        }
    }

    /// A refused name must not also be tabulated (the refusal would shadow it).
    #[test]
    fn refused_names_are_not_tabulated() {
        let names = d3bj_tabulated_functionals();
        for (k, _) in REFUSED {
            assert!(!names.contains(k), "{k} is both refused and tabulated");
        }
    }

    /// Every tabulated entry must be physically sensible. A zeroed a2 (the BJ
    /// offset) would make the damping vanish and the energy diverge at short
    /// range, so this pins the table against a transcription slip.
    #[test]
    fn tabulated_parameters_are_physical() {
        for name in [
            "blyp", "bp86", "b97d", "pbe", "pbe0", "revpbe", "b3lyp", "tpss", "tpss0", "b2plyp",
            "bhlyp", "scan", "r2scan", "wb97x",
        ] {
            let p = d3bj_params_for_functional(name).unwrap();
            assert!(
                p.s6 > 0.0 && p.s6 <= 1.0,
                "{name}: s6 out of range: {}",
                p.s6
            );
            assert!(
                p.s8 >= 0.0 && p.s8 < 5.0,
                "{name}: s8 out of range: {}",
                p.s8
            );
            assert!(
                p.a1 >= 0.0 && p.a1 < 1.0,
                "{name}: a1 out of range: {}",
                p.a1
            );
            assert_eq!(
                p.s9, 0.0,
                "{name}: a named functional must load with ATM off"
            );
            assert!(
                p.a2 > 1.0 && p.a2 < 8.0,
                "{name}: a2 (BJ offset, Bohr) out of range: {}",
                p.a2
            );
        }
    }
}
