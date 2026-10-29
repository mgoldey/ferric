//! Frozen content of every (basis set, element) pair that origin/main shipped
//! before the bundled library was widened (PR #596).
//!
//! `frozen_fingerprints.txt` holds one FNV-1a 64 hash per (file stem,
//! element). Each hash covers a canonical text form of the element's loaded
//! shells: per shell `l`, `pure`, every exponent and every contraction
//! coefficient (Rust's round-trip `{:?}` of the f64, so one ulp shows), and
//! the element's ECP if it has one. The file was generated from origin/main's
//! JSON (see [`regenerate_frozen_fingerprints`]); the test loads the same
//! pairs through [`bundled`] today and fails on any difference. A re-fetch, a
//! re-pack, a loader change or a hand edit therefore cannot silently move a
//! number someone has already computed with a shipped set.
//!
//! New elements and new sets are not frozen here; they are validated against
//! PySCF in `crates/ferric-mp2/tests/basis_coverage_vs_pyscf.rs`.
//!
//! MUTATIONS (2026-10-29, each made the test fail and was reverted): one
//! exponent x1.01 on H in `sto-3g.json`, and one exponent x1.01 on Rb in
//! `def2-svp.json`.

use super::{bundled, load_bse_json, BasisSet, Shell, REGISTRY};
use crate::ecp::EcpDef;
use std::collections::BTreeMap;
use std::fmt::Write as _;

const FROZEN_FILE: &str = "src/basis/frozen_fingerprints.txt";

/// 64-bit FNV-1a: tiny, dependency-free and stable across Rust releases
/// (unlike `DefaultHasher`).
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Canonical text of one element: shells in load order, then the ECP.
fn canonical_text(shells: &[Shell], ecp: Option<&EcpDef>) -> String {
    let mut s = String::new();
    for sh in shells {
        let _ = writeln!(
            s,
            "l={} pure={} exp={:?} coef={:?}",
            sh.l, sh.pure, sh.exponents, sh.coefficients
        );
    }
    if let Some(e) = ecp {
        let _ = writeln!(s, "ecp n_core={}", e.n_core);
        for ch in &e.shells {
            let terms: Vec<(f64, i32, f64)> =
                ch.terms.iter().map(|t| (t.coef, t.r_exp, t.gexp)).collect();
            let _ = writeln!(s, "ecp l={} terms={terms:?}", ch.angular_momentum);
        }
    }
    s
}

/// Element -> hex hash for every element of `bs`.
fn fingerprints(bs: &BasisSet) -> BTreeMap<i32, String> {
    bs.shells
        .iter()
        .map(|(&z, shells)| {
            let text = canonical_text(shells, bs.ecp_for_element(z));
            (z, format!("{:016x}", fnv1a64(text.as_bytes())))
        })
        .collect()
}

fn primary_name(stem: &str) -> Option<&'static str> {
    REGISTRY
        .iter()
        .find(|(s, _)| *s == stem)
        .map(|(_, names)| names[0])
}

/// Parsed `frozen_fingerprints.txt`: stem -> (element -> hash).
fn frozen() -> BTreeMap<String, BTreeMap<i32, String>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FROZEN_FILE);
    let text = std::fs::read_to_string(&path).unwrap();
    let mut out: BTreeMap<String, BTreeMap<i32, String>> = BTreeMap::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 3, "bad line {line:?}");
        out.entry(f[0].to_string())
            .or_default()
            .insert(f[1].parse().unwrap(), f[2].to_string());
    }
    out
}

/// Every (set, element) origin/main shipped loads today with identical
/// content.
#[test]
fn origin_main_elements_are_frozen() {
    let mut changed = Vec::new();
    let mut n = 0usize;
    for (stem, want) in frozen() {
        let name = primary_name(&stem).unwrap_or_else(|| {
            panic!("{stem}.json shipped on origin/main is no longer registered")
        });
        let got = fingerprints(&bundled(name).unwrap());
        for (z, hash) in &want {
            n += 1;
            match got.get(z) {
                Some(h) if h == hash => {}
                Some(_) => changed.push(format!("{stem} Z={z}: content changed")),
                None => changed.push(format!("{stem} Z={z}: element dropped")),
            }
        }
    }
    // reachability: an empty or truncated file would pass vacuously
    assert!(n > 1500, "only {n} frozen (set, element) pairs");
    assert!(
        changed.is_empty(),
        "{} shipped (set, element) pairs changed:\n{}",
        changed.len(),
        changed.join("\n")
    );
}

/// Regenerate `frozen_fingerprints.txt` from a directory holding origin/main's
/// `bundled/*.json`. Run ONLY to freeze a new baseline on purpose:
///
/// ```text
/// git archive origin/main crates/ferric-core/src/basis/bundled | tar -x -C /tmp/om
/// FERRIC_FROZEN_SRC=/tmp/om/crates/ferric-core/src/basis/bundled \
/// FERRIC_FROZEN_REV=$(git rev-parse origin/main) \
///   cargo test -p ferric-core --lib regenerate_frozen_fingerprints -- --ignored
/// ```
#[test]
#[ignore = "writes the frozen baseline; run deliberately"]
fn regenerate_frozen_fingerprints() {
    let src = std::env::var("FERRIC_FROZEN_SRC").expect("FERRIC_FROZEN_SRC");
    let rev = std::env::var("FERRIC_FROZEN_REV").expect("FERRIC_FROZEN_REV");
    let mut paths: Vec<_> = std::fs::read_dir(&src)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter(|p| p.file_name().is_some_and(|f| f != "def2-ecp.json"))
        .collect();
    paths.sort();
    let mut out = format!(
        "# (file stem, Z, FNV-1a 64 of the element's canonical shell text) for every\n\
         # element origin/main {rev} shipped. Generated by\n\
         # basis::frozen_fingerprints::regenerate_frozen_fingerprints; do not edit.\n"
    );
    for p in &paths {
        let stem = p.file_stem().unwrap().to_string_lossy();
        let bs = load_bse_json(p.to_str().unwrap()).unwrap();
        for (z, h) in fingerprints(&bs) {
            let _ = writeln!(out, "{stem} {z} {h}");
        }
    }
    let dst = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FROZEN_FILE);
    std::fs::write(dst, out).unwrap();
}
