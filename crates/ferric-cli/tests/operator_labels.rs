//! The printed SR/LR operator names must follow the attenuator actually used.
//!
//! These were hardcoded "erfc"/"erf" in the rs-mp2-rpa arm and so were WRONG
//! for every terf-split run: `Attenuator::Terf` selects `terf`/`terfc`
//! (rs_mp2_rpa.rs), but the output said `E(SR-MP2, erfc)`. The header line
//! correctly said `[terf split]`, so the output contradicted itself.
//!
//! That mislabel was read downstream and an entire attMP2 analysis was
//! documented as "erfc, not terfc -- NOT comparable to published values",
//! which inverted the truth: it IS the published operator family. A
//! mislabelled number is worse than an unlabelled one.
//!
//! This is a source-level guard rather than an end-to-end run: exercising the
//! print path needs a converged SCF plus terfc interpolation tables, which is
//! far too heavy for a unit test. Checking that no hardcoded operator name
//! survives in these println!s catches the regression that actually happened.

use std::fs;

/// Named `main_rs` for what it checks (the CLI's dispatch/println! logic),
/// not for where the file is on disk any more: main() moved to lib.rs (was
/// main.rs) so ferric-python could depend on it as a library and call it
/// from a #[pyfunction] -- see crates/ferric-cli/Cargo.toml's [lib] section.
/// The function name and this test's own doc comment stay accurate to WHAT
/// is being guarded; only the file path underneath changed.
fn main_rs() -> String {
    fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"))
        .expect("read lib.rs")
}

/// The five rs-mp2-rpa component lines must interpolate `{sr_name}`/`{lr_name}`,
/// never a literal erf/erfc.
#[test]
fn rs_mp2_rpa_component_labels_are_not_hardcoded() {
    let src = main_rs();
    for bad in [
        r#""  E(SR-MP2, erfc)"#,
        r#""  E(LR-MP2, erf)"#,
        r#""  E(dMP2, erf)"#,
        r#""  E(dRPA, erf)"#,
        r#""  E(ΔdRPA, erfc)"#,
    ] {
        assert!(
            !src.contains(bad),
            "hardcoded operator label {bad:?} is back in main.rs. With \
             attenuator = \"terf\" the operators are terf/terfc, so this line \
             would print the wrong operator name for every terf-split run. \
             Interpolate sr_name/lr_name from rs_cfg.attenuator instead."
        );
    }
    assert!(
        src.contains("E(SR-MP2, {sr_name})") && src.contains("E(LR-MP2, {lr_name})"),
        "the attenuator-driven labels are missing; they must be derived from \
         rs_cfg.attenuator so the printed name matches the operator in use"
    );
}

/// Methods that are locked to one operator must SAY which, because a bare
/// "Attenuated RI-MP2" cannot be distinguished from a terfc run downstream.
#[test]
fn single_operator_methods_name_their_operator() {
    let src = main_rs();
    assert!(
        src.contains("Attenuated RI-MP2 (erfc)/"),
        "the att-rimp2 header must name erfc explicitly: attenuated_ri_mp2 \
         hardcodes Operator::erfc, while scs-mp2-2terfc and mp2-v use terfc"
    );
    assert!(
        src.contains("SCS-MP2(2terfc)/"),
        "the scs-mp2-2terfc header must name its operator"
    );
    assert!(
        src.contains("MP2-V({attenuator})/"),
        "the mp2-v header must interpolate its attenuator"
    );
}
