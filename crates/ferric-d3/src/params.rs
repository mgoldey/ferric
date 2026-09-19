//! Becke-Johnson damping parameters, per exchange-correlation functional.
//!
//! Every D3(BJ) number depends on four fitted parameters `(s6, s8, a1, a2)`
//! that are specific to the functional the correction is being added to.
//! Applying PBE's parameters to a B3LYP energy is not a small error: it is
//! using a correction fitted to a different amount of missing dispersion.
//!
//! Values are from Grimme, Ehrlich and Goerigk, JCC 32, 1456 (2011), Table 1,
//! cross-checked against the `dftd3` reference implementation's parameter file
//! (see `tests/vs_reference_dftd3.rs`, which asserts agreement rather than
//! assuming it).

use ferric_core::FerricError;

/// The four Becke-Johnson rational-damping parameters.
///
/// ```text
/// E = - sum_{A<B} [ s6 C6/(R^6 + f^6) + s8 C8/(R^8 + f^8) ],  f = a1 R0 + a2
/// ```
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
}

/// D3(BJ) parameters for a named exchange-correlation functional.
///
/// The lookup is case-insensitive and tolerates the `-`/`_` spelling
/// differences between ferric's functional names and the D3 literature's.
///
/// Returns an error -- never a default or a zeroed parameter set -- for a
/// functional that has no published D3(BJ) fit. Silently substituting some
/// other functional's parameters would produce a plausible-looking number that
/// is not the D3(BJ) correction for the functional the user asked for.
pub fn d3bj_params_for_functional(name: &str) -> Result<D3Params, FerricError> {
    let key = name.to_ascii_lowercase().replace(['-', '_', ' '], "");
    let p = |s6, s8, a1, a2| Some(D3Params { s6, s8, a1, a2 });
    let found = match key.as_str() {
        // Grimme, Ehrlich, Goerigk, JCC 32, 1456 (2011), Table 1.
        "blyp" => p(1.0, 2.6996, 0.4298, 4.2359),
        "bp86" | "bp" => p(1.0, 3.2822, 0.3946, 4.8516),
        "b97d" => p(1.0, 2.2609, 0.5545, 3.2297),
        "pbe" => p(1.0, 0.7875, 0.4289, 4.4407),
        "pbe0" => p(1.0, 1.2177, 0.4145, 4.8593),
        "revpbe" => p(1.0, 1.7588, 0.5238, 3.5016),
        "b3lyp" => p(1.0, 1.9889, 0.3981, 4.4211),
        "tpss" => p(1.0, 1.9435, 0.4535, 4.4752),
        "tpss0" => p(1.0, 1.2576, 0.3768, 4.5865),
        "b2plyp" => p(0.64, 0.9147, 0.3065, 5.0570),
        "bhlyp" | "bhandhlyp" => p(1.0, 1.0354, 0.2793, 4.9615),
        "scan" => p(1.0, 0.0000, 0.5380, 5.4200),
        "r2scan" => p(1.0, 0.6019, 0.4948, 5.7341),
        "m062x" => p(1.0, 0.0000, 0.1689, 3.6117),
        "wb97x" => p(1.0, 0.0000, 0.0000, 5.4959),
        _ => None,
    };
    found.ok_or_else(|| {
        FerricError::General(format!(
            "no published D3(BJ) damping parameters for functional '{name}'. \
             Supply them explicitly (s6/s8/a1/a2) or pick a functional with a \
             published fit. Refusing to substitute another functional's \
             parameters, which would silently change the correction."
        ))
    })
}

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

    /// Every tabulated entry must be physically sensible. A zeroed a2 (the BJ
    /// offset) would make the damping vanish and the energy diverge at short
    /// range, so this pins the table against a transcription slip.
    #[test]
    fn tabulated_parameters_are_physical() {
        for name in [
            "blyp", "bp86", "b97d", "pbe", "pbe0", "revpbe", "b3lyp", "tpss", "tpss0", "b2plyp",
            "bhlyp", "scan", "r2scan", "m062x", "wb97x",
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
            assert!(
                p.a2 > 1.0 && p.a2 < 8.0,
                "{name}: a2 (BJ offset, Bohr) out of range: {}",
                p.a2
            );
        }
    }
}
