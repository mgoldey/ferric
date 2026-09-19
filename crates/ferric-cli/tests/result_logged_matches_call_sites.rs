//! `RESULT_LOGGED` must name exactly the methods that actually emit a `result`.
//!
//! The list drives whether a method emits `result_unlogged`, and its own
//! comment says "keep RESULT_LOGGED in step with the `rl.result(...)` call
//! sites; a name here that has no matching call site would claim coverage that
//! does not exist". Nothing enforced that.
//!
//! Both directions are wrong in different ways:
//!
//! * a name in the list with NO call site: the method emits neither `result`
//!   nor `result_unlogged`, so the log is SILENT -- and silence is exactly the
//!   ambiguity `result_unlogged` was added to remove ("unwired" vs "the run
//!   died before finishing").
//! * a call site with no name in the list: the method emits BOTH records, so a
//!   consumer sees a result and a claim that there is none.
//!
//! Checked by parsing the source rather than by running every method kind:
//! `mp3` alone is a full SCF plus an O(N^6) step, and the CLI has 20 kinds.
//! The parse is a real check because the two facts live in different places --
//! a list near the top of `run()` and call sites scattered through the
//! per-method functions -- which is what lets them drift.

use std::path::{Path, PathBuf};

/// Workspace root, resolved at RUN TIME -- see `ccsd_dispatch.rs` for why
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

#[test]
fn result_logged_names_exactly_the_methods_that_emit_a_result() {
    let src = std::fs::read_to_string(workspace_root().join("crates/ferric-cli/src/lib.rs"))
        .expect("read ferric-cli/src/lib.rs");

    // The declared list.
    let start = src
        .find("const RESULT_LOGGED: &[&str] = &[")
        .expect("RESULT_LOGGED must exist");
    let end = src[start..].find("];").expect("unterminated RESULT_LOGGED") + start;
    let mut declared: Vec<String> = src[start..end]
        .lines()
        .filter_map(|l| {
            let t = l.trim().trim_end_matches(',');
            t.strip_prefix('"')?.strip_suffix('"').map(str::to_string)
        })
        .collect();

    // The actual call sites: `rl.result(\n    "<kind>",`.
    let mut called: Vec<String> = Vec::new();
    let mut rest = src.as_str();
    while let Some(i) = rest.find("rl.result(") {
        rest = &rest[i + "rl.result(".len()..];
        // The kind is the first quoted string after the open paren.
        if let Some(q) = rest.find('"') {
            // ...but only if no `)` or `;` intervenes, which would mean this
            // call does not take a literal kind.
            if !rest[..q].contains(';') {
                if let Some(e) = rest[q + 1..].find('"') {
                    called.push(rest[q + 1..q + 1 + e].to_string());
                }
            }
        }
    }

    declared.sort();
    declared.dedup();
    called.sort();
    called.dedup();

    assert!(
        !declared.is_empty() && !called.is_empty(),
        "parse found nothing"
    );
    assert_eq!(
        declared, called,
        "RESULT_LOGGED and the rl.result(...) call sites disagree.\n  \
         declared: {declared:?}\n  called  : {called:?}\n\
         A name declared with no call site makes the log SILENT for that \
         method (neither result nor result_unlogged), which is the exact \
         ambiguity result_unlogged exists to remove. A call site not declared \
         makes the method emit BOTH."
    );
}
