//! CLI messages must not send readers to `docs/` paths.
//!
//! `docs/` was removed from the repository (its contents live in the
//! untracked project wiki), so a warning saying "see docs/VALIDATION.md"
//! pointed at a file no user has. The published documentation is the mdBook
//! under `site/src`; messages cite `site/src/reference/validation.md` and
//! `site/src/using/quickstart.md`, which this test also checks exist.
//!
//! Scope: non-comment source lines only. Code comments citing `docs/<note>.md`
//! are a historical record (the repo README explains the mapping to the wiki)
//! and are deliberately left alone.

use std::path::{Path, PathBuf};

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
fn no_message_cites_a_docs_path() {
    let root = workspace_root();
    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    for rel in [
        "crates/ferric-cli/src/lib.rs",
        "crates/ferric-cli/src/config.rs",
        "crates/ferric-cli/src/bin/ferric-batch.rs",
    ] {
        let src = std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        for (i, line) in src.lines().enumerate() {
            scanned += 1;
            if line.trim_start().starts_with("//") {
                continue;
            }
            if line.contains("docs/") || line.contains("CLAUDE.md") {
                offenders.push(format!("{rel}:{}: {}", i + 1, line.trim()));
            }
        }
    }
    assert!(scanned > 1000, "scanned only {scanned} lines");
    assert!(
        offenders.is_empty(),
        "user-facing strings cite unpublished paths (use site/src/...):\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn the_cited_pages_exist() {
    let root = workspace_root();
    for page in ["site/src/reference/validation.md", "site/src/using/quickstart.md"] {
        assert!(root.join(page).is_file(), "{page} is cited by CLI messages but missing");
    }
}
