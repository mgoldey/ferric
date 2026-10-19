//! serde_json must parse f64 correctly rounded (the `float_roundtrip`
//! feature, enabled on the workspace dependency).
//!
//! Without it serde_json's fast path can land one ulp off the nearest double,
//! so a CLI test that compares a JSON-parsed energy to the in-process library
//! energy with `to_bits()` fails falsely. Only number-typed JSON values are
//! affected (test reference files, run-log energies read back by tests); the
//! basis-set and ECP loaders read numbers as strings and were never affected.
//!
//! Mutation ledger: removing `float_roundtrip` from the workspace
//! `serde_json` dependency makes `parses_the_cli_log_energy_correctly_rounded`
//! fail (serde_json yields -1.62246266888477, one ulp from the literal).

#[test]
fn parses_the_cli_log_energy_correctly_rounded() {
    // Exact text from a ferric-cli run log; the Rust literal is correctly
    // rounded by rustc, so it is the reference.
    let text = "-1.6224626688847699";
    let expected: f64 = -1.622_462_668_884_769_9;
    let parsed: f64 = serde_json::from_str(text).unwrap();
    assert_eq!(
        parsed.to_bits(),
        expected.to_bits(),
        "serde_json parsed {text} as {parsed:?}, expected {expected:?}"
    );
    // And str::parse (correctly rounded) agrees with the literal.
    assert_eq!(text.parse::<f64>().unwrap().to_bits(), expected.to_bits());
}

#[test]
fn round_trips_a_serialized_f64_bit_for_bit() {
    let x: f64 = -1.622_462_668_884_769_9;
    let s = serde_json::to_string(&x).unwrap();
    let back: f64 = serde_json::from_str(&s).unwrap();
    assert_eq!(back.to_bits(), x.to_bits(), "{s}");
}
