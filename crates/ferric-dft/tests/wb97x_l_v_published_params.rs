//! ωB97X-L-V coefficients vs the PUBLISHED table that defines the functional.
//!
//! # Why this file exists
//!
//! Every other ωB97X-L-V test in the tree checks *plumbing*: that libxc exposes 18
//! external parameters in the expected order, that setting them changes the evaluated
//! energy, that they survive the rayon worker clone, that the UEG-constrained
//! coefficients are self-consistent with λ. All of those pass identically whether the
//! twelve fitted coefficients are the paper's or somebody else's — they never compare a
//! single fitted digit to the source.
//!
//! That gap is not academic. The functional's absolute total energies are published
//! NOWHERE (see `testdata/reference/wb97x_l_v_params.json`), so there is no energy to
//! regress against, and a mistranscribed coefficient would produce a perfectly
//! convergent SCF returning a silently wrong number for a *differently parameterized
//! functional*. The coefficient table is the only external anchor that exists.
//!
//! # Source
//!
//! Ransford & Carter-Fenk, *Phys. Chem. Chem. Phys.* **2026**, 28, 14428–14441,
//! doi:10.1039/D6CP00232C, Table 2 ("Final" column), p. 14433. Local copy at
//! `papers/wb97xlv.pdf`; digits and signs transcribed into
//! `testdata/reference/wb97x_l_v_params.json`, which records how the signs were
//! recovered from the PDF (plain `pdftotext` DROPS the minus glyph — see that file's
//! `_sign_provenance`, and do not "re-check" these signs with it).
//!
//! # Tolerance
//!
//! Exact equality, via a bit-for-bit comparison of the `f64` nearest each printed
//! decimal. These are transcribed literals, not computed quantities: there is no
//! arithmetic between the paper and the constant, so any nonzero difference is a typo
//! rather than numerical error. A float tolerance here would only hide the very defect
//! the test exists to catch.

use ferric_dft::libxc::{WB97X_L_V_EXT_PARAMS, WB97X_L_V_LAMBDA, WB97X_L_V_VV10};
use std::path::PathBuf;

fn published() -> serde_json::Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference/wb97x_l_v_params.json");
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {p:?}: {e}"));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parsing {p:?}: {e}"))
}

fn ferric_param(name: &str) -> f64 {
    WB97X_L_V_EXT_PARAMS
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("ferric has no ext param named {name}"))
        .1
}

fn as_f64(v: &serde_json::Value, key: &str) -> f64 {
    v.get(key)
        .unwrap_or_else(|| panic!("reference JSON missing key {key}"))
        .as_f64()
        .unwrap_or_else(|| panic!("reference key {key} is not a number"))
}

/// The twelve least-squares-optimized coefficients, digit for digit AND sign for sign.
///
/// The paper's names map onto libxc's as `cx_i -> _cx{i}`, `css_i -> _css{i}`,
/// `cab_i -> _cos{i}` (libxc spells opposite-spin "os", the paper "ab"; both mean the
/// αβ channel).
#[test]
fn fitted_coefficients_match_the_published_table() {
    let doc = published();
    let fit = &doc["least_squares_optimized_parameters_final"];

    // (paper key, libxc key)
    let pairs = [
        ("cx_1", "_cx1"),
        ("cx_2", "_cx2"),
        ("cx_3", "_cx3"),
        ("cx_4", "_cx4"),
        ("css_1", "_css1"),
        ("css_2", "_css2"),
        ("css_3", "_css3"),
        ("css_4", "_css4"),
        ("cab_1", "_cos1"),
        ("cab_2", "_cos2"),
        ("cab_3", "_cos3"),
        ("cab_4", "_cos4"),
    ];

    let mut wrong = Vec::new();
    for (paper_key, libxc_key) in pairs {
        let want = as_f64(fit, paper_key);
        let got = ferric_param(libxc_key);
        if got.to_bits() != want.to_bits() {
            let sign_only = got == -want;
            wrong.push(format!(
                "  {libxc_key:<6} (paper {paper_key:<6}): ferric {got:>9.3}  published {want:>9.3}\
                 {}",
                if sign_only { "   <-- SIGN FLIPPED" } else { "" }
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "{} of {} fitted coefficients disagree with Ransford & Carter-Fenk PCCP 2026, \
         28, 14428 Table 2 (Final column):\n{}\n\
         The published signs were recovered from papers/wb97xlv.pdf by two independent \
         methods (bbox x-offsets and the raw content stream); see _sign_provenance in \
         testdata/reference/wb97x_l_v_params.json. Plain `pdftotext` drops the minus \
         glyph and will wrongly report every value positive.",
        wrong.len(),
        pairs.len(),
        wrong.join("\n")
    );
}

/// The constrained (not fitted) coefficients, and the non-linear parameters.
///
/// `constrained_coefficients_satisfy_the_ueg_limit` in `ext_params.rs` already checks
/// these are internally consistent with λ. This checks the stronger thing: that λ
/// itself, and ω, b and C, are the values the paper actually chose. An internally
/// consistent set built on the wrong λ would satisfy that test and fail this one.
#[test]
fn fixed_parameters_match_the_published_table() {
    let doc = published();
    let fixed = &doc["fixed_parameters"];

    let lambda = as_f64(fixed, "lambda");
    assert_eq!(
        WB97X_L_V_LAMBDA.to_bits(),
        lambda.to_bits(),
        "lambda: ferric {WB97X_L_V_LAMBDA} vs published {lambda}"
    );

    for (paper_key, libxc_key) in [("cx_0", "_cx0"), ("css_0", "_css0"), ("cab_0", "_cos0")] {
        let want = as_f64(fixed, paper_key);
        let got = ferric_param(libxc_key);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{libxc_key} (paper {paper_key}): ferric {got} vs published {want}"
        );
    }

    let omega = as_f64(fixed, "omega");
    let got_omega = ferric_param("_omega");
    assert_eq!(
        got_omega.to_bits(),
        omega.to_bits(),
        "_omega: ferric {got_omega} vs published {omega} Bohr^-1"
    );

    let b = as_f64(fixed, "b");
    let c = as_f64(fixed, "C");
    assert_eq!(
        WB97X_L_V_VV10.b.to_bits(),
        b.to_bits(),
        "VV10 b: ferric {} vs published {b}",
        WB97X_L_V_VV10.b
    );
    assert_eq!(
        WB97X_L_V_VV10.c.to_bits(),
        c.to_bits(),
        "VV10 C: ferric {} vs published {c}",
        WB97X_L_V_VV10.c
    );
}

/// The CAM mixing ferric derives from eqn (27), against the JSON's record of it.
///
/// Unlike the coefficients above these two are not digits lifted from Table 2 — they
/// are ferric's reading of eqn (27) under libxc's CAM convention, and the JSON says so.
/// The test pins that reading so it cannot drift silently; it is NOT independent
/// evidence that the reading is right.
#[test]
fn cam_mixing_matches_the_recorded_reading_of_eqn_27() {
    let doc = published();
    let mix = &doc["exchange_mixing"];
    for (paper_key, libxc_key) in [("alpha", "_alpha"), ("beta", "_beta")] {
        let want = as_f64(mix, paper_key);
        let got = ferric_param(libxc_key);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{libxc_key}: ferric {got} vs recorded {want}"
        );
    }
    // And the consistency the paper does constrain: c_sr = alpha + beta = lambda.
    let c_sr = ferric_param("_alpha") + ferric_param("_beta");
    assert!(
        (c_sr - WB97X_L_V_LAMBDA).abs() < 1e-12,
        "c_sr = alpha + beta = {c_sr} != lambda {WB97X_L_V_LAMBDA}"
    );
}

/// Guard the guard: the reference file must actually carry the signs it claims.
///
/// If someone regenerates the JSON with bare `pdftotext` (which drops the minus glyph)
/// every fitted value would come back positive, the comparison above would be
/// trivially satisfiable by an all-positive constant table, and the real check would
/// evaporate with every test still green. Four coefficients whose published sign is
/// negative and whose magnitude is distinctive are pinned here directly.
#[test]
fn the_reference_file_retains_its_negative_signs() {
    let doc = published();
    let fit = &doc["least_squares_optimized_parameters_final"];
    for key in ["cx_3", "css_4", "cab_2", "cab_4"] {
        let v = as_f64(fit, key);
        assert!(
            v < 0.0,
            "{key} = {v} is not negative -- the reference table was probably regenerated \
             with plain `pdftotext`, which silently drops the PDF's minus glyph. See \
             _sign_provenance in the JSON."
        );
    }
    // ... and the positives really are positive, so a blanket sign flip also fails.
    for key in ["cx_2", "cx_4", "css_2", "cab_3"] {
        let v = as_f64(fit, key);
        assert!(v > 0.0, "{key} = {v} should be positive");
    }
}
