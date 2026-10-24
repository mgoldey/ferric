//! The build-time decision for the libint2 `boys.h` override (issue #234):
//! a stock header gets the override, anything else builds with a warning by
//! default, and is a build error under `FERRIC_REQUIRE_PATCHED_BOYS` unless
//! `FERRIC_ALLOW_UNPATCHED_BOYS` opts out. `build.rs` `include!`s the same
//! file, so these tests exercise the code the build runs.
//!
//! Mutation ledger (each mutant applied to a copy of
//! build_support/boys_policy.rs and this file compiled standalone with
//! rustc; the named tests failed, the unmutated copy passed 10/10):
//! - `if env.require && !env.allow_unpatched` -> `if env.require`:
//!   `allow_downgrades_the_refusal_to_a_warning` fails.
//! - `if env.require && ...` -> `if false && ...`: all three
//!   `require_refuses_*` tests fail.
//! - change the last digit of the 2.13.1 hash in `STOCK_BOYS_H_FNV1A64`:
//!   `stock_hashes_select_their_override` (and the message check in
//!   `require_refuses_an_unknown_header`) fail.
//! - swap the two hashes in `STOCK_BOYS_H_FNV1A64` (each version maps to the
//!   other's override): `stock_hashes_select_their_override` fails.
//! - `env_flag` accepting any set value (`value.is_some()`):
//!   `env_flag_parses_off_values_as_false` fails.
//! - Refuse message without the opt-out variable name:
//!   `require_refuses_an_unknown_header` fails.
//! - hash lookup moved BEFORE the `force_stock` check (a stock header then
//!   wins over FERRIC_LIBINT2_STOCK_BOYS):
//!   `forced_stock_wins_over_a_real_stock_header` fails.
//! - the LIBINT2_PREFIX remedy used for the forced-stock reason too:
//!   `forced_stock_wins_over_a_real_stock_header` fails.

#[allow(dead_code)]
mod policy {
    include!("../build_support/boys_policy.rs");
}

use policy::*;

const PATH: &str = "/prefix/include/libint2/boys.h";

/// The stock `boys.h` of `version`, rebuilt by reverse-applying the
/// committed `.patch` to the committed override (GNU `patch`). The README
/// pins the result's sha256; this test pins its FNV-1a 64 and runs it
/// through the same decision build.rs makes.
fn stock_boys_h(version: &str) -> Vec<u8> {
    let dir = format!(
        "{}/shim/libint2_overrides/{version}",
        env!("CARGO_MANIFEST_DIR")
    );
    let out = tempfile::NamedTempFile::new().unwrap();
    let status = std::process::Command::new("patch")
        .args(["-s", "-R", "-o"])
        .arg(out.path())
        .arg(format!("{dir}/libint2/boys.h"))
        .arg(format!("{dir}/libint2-{version}-gmeval-no-copy.patch"))
        .status()
        .expect("GNU patch is needed to rebuild the stock boys.h");
    assert!(status.success(), "patch -R failed for {version}");
    std::fs::read(out.path()).unwrap()
}

#[test]
fn fnv1a64_matches_the_reference_vectors() {
    // Published FNV-1a 64 test vectors.
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
}

#[test]
fn stock_hashes_select_their_override() {
    for version in ["2.7.2", "2.13.1"] {
        let stock = stock_boys_h(version);
        // Selected by default AND under the CI requirement.
        for require in [false, true] {
            let env = BoysEnv {
                require,
                ..BoysEnv::default()
            };
            assert_eq!(
                boys_outcome(PATH, Ok(&stock), env),
                BoysOutcome::Override(version),
                "{version} require={require}"
            );
        }
        // One flipped byte is no longer stock.
        let mut edited = stock.clone();
        edited[0] ^= 1;
        assert!(matches!(
            boys_outcome(PATH, Ok(&edited), BoysEnv::default()),
            BoysOutcome::Unpatched(_)
        ));
    }
}

#[test]
fn the_override_files_hash_to_no_stock_entry() {
    // A prefix that already carries ferric's patched boys.h must not be
    // mistaken for stock (the override would then shadow itself harmlessly,
    // but the table would be lying).
    for (version, _) in STOCK_BOYS_H_FNV1A64 {
        let path = format!(
            "{}/shim/libint2_overrides/{version}/libint2/boys.h",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(&path).expect("override file present");
        let h = fnv1a64(&bytes);
        assert!(
            STOCK_BOYS_H_FNV1A64.iter().all(|(_, s)| *s != h),
            "{path} hashes to a stock entry"
        );
    }
}

#[test]
fn unknown_header_warns_by_default() {
    let out = boys_outcome(PATH, Ok(b"not boys.h"), BoysEnv::default());
    let BoysOutcome::Unpatched(msg) = out else {
        panic!("expected a warning, got {out:?}");
    };
    assert!(msg.contains(PATH), "{msg}");
    assert!(msg.contains("NOT applied"), "{msg}");
}

#[test]
fn require_refuses_an_unknown_header() {
    let bytes = b"terf-patched boys.h";
    let env = BoysEnv {
        require: true,
        ..BoysEnv::default()
    };
    let BoysOutcome::Refuse(msg) = boys_outcome(PATH, Ok(bytes), env) else {
        panic!("expected a refusal");
    };
    // Names the hash, every known hash and the opt-out.
    assert!(msg.contains(&format!("{:#018x}", fnv1a64(bytes))), "{msg}");
    assert!(msg.contains("2.7.2 = 0x1d51ac52f2076ad8"), "{msg}");
    assert!(msg.contains("2.13.1 = 0x0f6f0c23194e18ad"), "{msg}");
    assert!(msg.contains(ALLOW_VAR), "{msg}");
    assert!(msg.contains(REQUIRE_VAR), "{msg}");
    assert!(msg.contains("LIBINT2_PREFIX"), "{msg}");
}

#[test]
fn require_refuses_an_unreadable_header() {
    let env = BoysEnv {
        require: true,
        ..BoysEnv::default()
    };
    let out = boys_outcome(PATH, Err("No such file".into()), env);
    assert!(matches!(out, BoysOutcome::Refuse(ref m) if m.contains("cannot read")));
}

#[test]
fn require_refuses_forced_stock() {
    let env = BoysEnv {
        require: true,
        force_stock: true,
        ..BoysEnv::default()
    };
    let out = boys_outcome(PATH, Ok(b"x"), env);
    assert!(matches!(out, BoysOutcome::Refuse(ref m) if m.contains(STOCK_VAR)));
}

#[test]
fn forced_stock_wins_over_a_real_stock_header() {
    // FERRIC_LIBINT2_STOCK_BOYS must suppress the override even when the
    // prefix's boys.h IS stock (the A/B case it exists for).
    let stock = stock_boys_h("2.13.1");
    let forced = BoysEnv {
        force_stock: true,
        ..BoysEnv::default()
    };
    let out = boys_outcome(PATH, Ok(&stock), forced);
    assert!(
        matches!(out, BoysOutcome::Unpatched(ref m) if m.contains(STOCK_VAR)),
        "{out:?}"
    );
    let required = BoysEnv {
        require: true,
        ..forced
    };
    let BoysOutcome::Refuse(msg) = boys_outcome(PATH, Ok(&stock), required) else {
        panic!("expected a refusal");
    };
    // The remedy fits the reason: unset the forcing variable, not "point
    // LIBINT2_PREFIX at a stock libint2" (it already is one).
    assert!(msg.contains(&format!("Unset {STOCK_VAR}")), "{msg}");
    assert!(msg.contains(ALLOW_VAR), "{msg}");
    assert!(!msg.contains("LIBINT2_PREFIX"), "{msg}");
}

#[test]
fn allow_downgrades_the_refusal_to_a_warning() {
    let env = BoysEnv {
        require: true,
        allow_unpatched: true,
        ..BoysEnv::default()
    };
    assert!(matches!(
        boys_outcome(PATH, Ok(b"x"), env),
        BoysOutcome::Unpatched(_)
    ));
}

#[test]
fn env_flag_parses_off_values_as_false() {
    for on in ["1", "true", "TRUE", "yes", "on", " 1 "] {
        assert!(env_flag(Some(on)), "{on:?}");
    }
    for off in ["0", "", "false", "no", "off"] {
        assert!(!env_flag(Some(off)), "{off:?}");
    }
    assert!(!env_flag(None));
}

/// The `boys.h` of a terf-patched 2.7.2 prefix: the stock 2.7.2 file (rebuilt
/// as above) with the committed copy of ferric's terf patch applied.
fn terf_boys_h() -> Vec<u8> {
    let stock = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(stock.path(), stock_boys_h("2.7.2")).unwrap();
    let out = tempfile::NamedTempFile::new().unwrap();
    let status = std::process::Command::new("patch")
        .args(["-s", "-o"])
        .arg(out.path())
        .arg(stock.path())
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shim/libint2_overrides/2.7.2/libint2-terf-boys.h.patch"
        ))
        .status()
        .expect("GNU patch is needed to rebuild the terf boys.h");
    assert!(status.success(), "patch failed for the terf boys.h");
    std::fs::read(out.path()).unwrap()
}

#[test]
fn terf_prefix_selects_the_2_7_2_override_with_the_terf_define() {
    // #234: the terf-patched prefix used to get no override at all (no
    // GmEval copy fix, no #433/#436/#443 fixes) and terf_gm_eval kept its
    // per-call copy. It now gets the 2.7.2 override with the terf evaluator.
    let terf = terf_boys_h();
    assert_ne!(
        terf,
        stock_boys_h("2.7.2"),
        "the terf patch changed nothing"
    );
    for require in [false, true] {
        let env = BoysEnv {
            require,
            ..BoysEnv::default()
        };
        assert_eq!(
            boys_outcome(PATH, Ok(&terf), env),
            BoysOutcome::OverrideTerf("2.7.2"),
            "require={require}"
        );
    }
    let mut edited = terf.clone();
    edited[0] ^= 1;
    assert!(matches!(
        boys_outcome(PATH, Ok(&edited), BoysEnv::default()),
        BoysOutcome::Unpatched(_)
    ));
    // FERRIC_LIBINT2_STOCK_BOYS still wins (A/B against the prefix's own header).
    let forced = BoysEnv {
        force_stock: true,
        ..BoysEnv::default()
    };
    assert!(matches!(
        boys_outcome(PATH, Ok(&terf), forced),
        BoysOutcome::Unpatched(_)
    ));
}

#[test]
fn terf_hashes_are_distinct_from_stock_and_override_hashes() {
    for (version, h) in TERF_BOYS_H_FNV1A64 {
        assert!(
            STOCK_BOYS_H_FNV1A64.iter().all(|(_, s)| *s != h),
            "terf {version} collides with a stock hash"
        );
        let path = format!(
            "{}/shim/libint2_overrides/{version}/libint2/boys.h",
            env!("CARGO_MANIFEST_DIR")
        );
        assert_ne!(fnv1a64(&std::fs::read(&path).unwrap()), h, "{path}");
    }
}

#[test]
fn terf_code_in_the_override_is_compiled_only_under_the_terf_define() {
    // A stock 2.7.2 prefix gets the same override WITHOUT TERF_DEFINE; any
    // terf code outside the guarded blocks would change the stock builds (and
    // fail to compile there: stock engine.h has no Operator::terf).
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shim/libint2_overrides/2.7.2/libint2/boys.h"
    );
    let text = std::fs::read_to_string(path).unwrap();
    let (mut depth, mut blocks, mut guarded_terf) = (0usize, 0usize, 0usize);
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with(&format!("#ifdef {TERF_DEFINE}")) {
            assert_eq!(depth, 0, "nested {TERF_DEFINE} block");
            depth = 1;
            blocks += 1;
            continue;
        }
        if depth > 0 {
            if t.starts_with("#if") {
                depth += 1;
            } else if t.starts_with("#endif") {
                depth -= 1;
            }
            guarded_terf += usize::from(line.contains("terf_gm_eval"));
            continue;
        }
        // Comments may name it (the file header does); code may not.
        let code = line.split("//").next().unwrap_or("");
        let comment = t.starts_with('*') || t.starts_with("/*");
        assert!(
            comment || !code.contains("terf_gm_eval"),
            "unguarded terf code in {path}: {line}"
        );
    }
    assert_eq!(depth, 0, "unterminated {TERF_DEFINE} block");
    // The include, the trait specialization and the evaluator itself.
    assert_eq!(blocks, 3, "expected 3 {TERF_DEFINE} blocks");
    assert!(guarded_terf >= 3, "the guarded blocks hold no terf_gm_eval");
}
