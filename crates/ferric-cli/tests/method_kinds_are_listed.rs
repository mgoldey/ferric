//! `SUPPORTED_METHOD_KINDS` must name exactly the `method.kind`s that `run()`
//! dispatches, and the unknown-kind error must list every one of them.
//!
//! The error message used to be a hand-written string beside a separate
//! `matches!` accept-list, and it drifted: `lmp2` and `lmp2-direct` were
//! accepted and dispatched but missing from the message, so a user who typo'd
//! `lmp2` was told it did not exist. The message is now derived from the list;
//! this test pins that, and pins the list against the dispatch arms by parsing
//! the source (running every kind would mean an SCF plus a correlated method
//! per kind).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Workspace root, resolved at RUN time -- see `ccsd_dispatch.rs` for why
/// `env!("CARGO_MANIFEST_DIR")` is wrong under `cargo nextest archive`.
fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("examples").is_dir() && p.join("testdata").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

/// Every double-quoted literal in `s`.
fn quoted(s: &str) -> Vec<String> {
    s.split('"').skip(1).step_by(2).map(str::to_string).collect()
}

#[test]
fn unknown_kind_message_lists_every_supported_kind() {
    let msg = ferric_cli::unsupported_method_message("not-a-method");
    assert!(msg.contains("\"not-a-method\""), "{msg}");
    // Match the QUOTED token: a bare substring check would let "rimp2" pass
    // on the strength of "oo-rimp2", and "lmp2" on "lmp2-direct".
    for kind in ferric_cli::SUPPORTED_METHOD_KINDS {
        assert!(
            msg.contains(&format!("\"{kind}\"")),
            "unknown-kind message omits {kind:?}: {msg}"
        );
    }
    // The two that drifted out of the old hand-written message.
    for kind in ["lmp2", "lmp2-direct"] {
        assert!(ferric_cli::SUPPORTED_METHOD_KINDS.contains(&kind));
        assert!(msg.contains(&format!("\"{kind}\"")), "{msg}");
    }
}

#[test]
fn supported_kinds_match_the_dispatch_arms_in_run() {
    let src = std::fs::read_to_string(workspace_root().join("crates/ferric-cli/src/lib.rs"))
        .expect("read ferric-cli/src/lib.rs");

    // The body of `run()`: from its signature to the next top-level `fn`.
    let start = src.find("pub fn run(args: Vec<String>)").expect("run() must exist");
    let len = src[start..]
        .find("\nfn ")
        .expect("a top-level fn must follow run()");
    let body = &src[start..start + len];

    let mut dispatched = BTreeSet::new();
    // Early-return branches: `if method == "uhf" {` and
    // `if matches!(method, "b2plyp" | "dsd-pbep86") {`.
    for line in body.lines() {
        let t = line.trim();
        if (t.starts_with("if method == \"") || t.starts_with("if matches!(method, "))
            && t.ends_with('{')
        {
            dispatched.extend(quoted(t));
        }
    }
    // The main dispatch `match method { "rhf" => run_rhf(...), ... _ => unreachable!() }`.
    let m = body
        .find("match method {\n        \"rhf\" => run_rhf(")
        .expect("the main `match method` dispatch must exist in run()");
    let arms_end = body[m..]
        .find("_ => unreachable!()")
        .expect("the dispatch must end in `_ => unreachable!()`");
    for line in body[m..m + arms_end].lines() {
        let t = line.trim();
        if t.starts_with('"') {
            if let Some(pat) = t.split("=>").next() {
                dispatched.extend(quoted(pat));
            }
        }
    }

    let declared: BTreeSet<String> = ferric_cli::SUPPORTED_METHOD_KINDS
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        declared.len(),
        ferric_cli::SUPPORTED_METHOD_KINDS.len(),
        "SUPPORTED_METHOD_KINDS has duplicates"
    );
    // Reachability: an empty scan would compare two empty sets and pass.
    assert!(dispatched.len() > 20, "scan found only {dispatched:?}");
    assert_eq!(
        declared, dispatched,
        "SUPPORTED_METHOD_KINDS (left) and the kinds run() dispatches (right) differ"
    );
}
