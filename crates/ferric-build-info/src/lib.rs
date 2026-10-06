//! Which build of ferric is this?
//!
//! Every value is fixed at compile time by `build.rs`:
//!
//! - [`VERSION`]: the PEP 440 package version from the workspace
//!   `pyproject.toml`. A checkout carries a `.devN` version; only the release
//!   workflow, building from a `v*` tag, stamps the plain release version.
//! - [`COMMIT`]: the full git hash of `HEAD`, or `"unknown"` when the source
//!   was not a git checkout (an sdist, a vendored tree).
//! - [`DIRTY`]: whether any tracked file differed from `HEAD` (`git status
//!   --porcelain --untracked-files=no`); untracked files never count. `None`
//!   when it could not be determined.
//! - [`PROFILE`]: cargo's build profile (`release` or `debug`).
//!
//! Only the top-level artifacts (`ferric-cli`, `ferric-python`) depend on this
//! crate; see its `Cargo.toml` for why.

#[cfg(test)]
mod probe;

/// PEP 440 package version, or `"unknown"`.
pub const VERSION: &str = env!("FERRIC_BUILD_VERSION");
/// Full git commit hash of the source tree, or `"unknown"`.
pub const COMMIT: &str = env!("FERRIC_BUILD_COMMIT");
/// Cargo profile the build used.
pub const PROFILE: &str = env!("FERRIC_BUILD_PROFILE");
/// Whether the compiled sources differed from [`COMMIT`]; `None` if unknown.
pub const DIRTY: Option<bool> = match env!("FERRIC_BUILD_DIRTY").as_bytes() {
    b"true" => Some(true),
    b"false" => Some(false),
    _ => None,
};

/// `true` when the commit is known.
pub fn commit_known() -> bool {
    COMMIT != "unknown"
}

/// Short commit with a `-dirty` suffix when the build was dirty, e.g.
/// `396e0d61eded-dirty`; `None` when the commit is unknown.
pub fn short_commit() -> Option<String> {
    if !commit_known() {
        return None;
    }
    let short = &COMMIT[..12.min(COMMIT.len())];
    Some(match DIRTY {
        Some(true) => format!("{short}-dirty"),
        _ => short.to_string(),
    })
}

/// `"true"`, `"false"` or `"unknown"`.
pub fn dirty_str() -> &'static str {
    match DIRTY {
        Some(true) => "true",
        Some(false) => "false",
        None => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap()
            .to_path_buf()
    }

    fn run_git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    }

    /// The embedded commit is the HEAD of the checkout this test was built
    /// from, read here by an independent `git rev-parse`, and the embedded
    /// dirty flag matches the tree as it is now.
    #[test]
    fn embedded_commit_is_the_checkout_head() {
        let root = workspace_root();
        if !root.join(".git").exists() {
            assert_eq!(COMMIT, "unknown", "no .git, yet a commit was embedded");
            return;
        }
        assert_eq!(COMMIT, run_git(&root, &["rev-parse", "HEAD"]));
        assert_eq!(DIRTY, probe::probe_git(&root).dirty);
        assert!(
            DIRTY.is_some(),
            "a checkout must give a definite dirty flag"
        );
    }

    #[test]
    fn version_is_the_pyproject_project_version() {
        let text = std::fs::read_to_string(workspace_root().join("pyproject.toml")).unwrap();
        assert_eq!(Some(VERSION.to_string()), probe::pyproject_version(&text));
        assert!(
            !VERSION.contains('+'),
            "PyPI rejects local versions: {VERSION}"
        );
    }

    #[test]
    fn profile_is_a_cargo_profile() {
        assert!(matches!(PROFILE, "debug" | "release"), "{PROFILE}");
    }

    #[test]
    fn short_commit_carries_the_dirty_suffix() {
        let s = short_commit().expect("tests run from a checkout");
        assert!(COMMIT.starts_with(s.trim_end_matches("-dirty")));
        assert_eq!(s.ends_with("-dirty"), DIRTY == Some(true));
    }

    const H: &str = "0123456789abcdef0123456789abcdef01234567"; // pragma: allowlist secret
    fn st(commit: &str, dirty: Option<bool>) -> probe::GitState {
        probe::GitState {
            commit: commit.into(),
            dirty,
        }
    }

    #[test]
    fn release_gate_decisions() {
        assert!(
            probe::release_gate(true, &st(H, Some(false))).is_ok(),
            "release+clean"
        );
        assert!(probe::release_gate(true, &st(H, Some(true)))
            .unwrap_err()
            .contains("DIRTY"));
        assert!(
            probe::release_gate(true, &probe::GitState::unknown()).is_err(),
            "release+unknown"
        );
        assert!(
            probe::release_gate(true, &st(H, None)).is_err(),
            "release+unknown dirty"
        );
        assert!(
            probe::release_gate(false, &st(H, Some(true))).is_ok(),
            "dev+dirty"
        );
        assert!(
            probe::release_gate(false, &probe::GitState::unknown()).is_ok(),
            "dev+unknown"
        );
    }

    #[test]
    fn identity_decision_table() {
        let measured = || st("m".repeat(40).as_str(), Some(true));
        let boom = || -> probe::GitState { panic!("release must not run git") };
        // Release: env only, git never consulted.
        let g = probe::resolve_identity(true, Some(H), Some("0"), boom).unwrap();
        assert_eq!(g, st(H, Some(false)));
        let g = probe::resolve_identity(true, Some(H), Some("1"), boom).unwrap();
        assert_eq!(g.dirty, Some(true));
        // Release: missing or malformed values fail.
        assert!(probe::resolve_identity(true, None, Some("0"), boom).is_err());
        assert!(probe::resolve_identity(true, Some(H), None, boom).is_err());
        assert!(probe::resolve_identity(true, Some("abc"), Some("0"), boom).is_err());
        assert!(
            probe::resolve_identity(true, Some(&H.replace('0', "g")), Some("0"), boom).is_err()
        );
        assert!(probe::resolve_identity(true, Some(H), Some("false"), boom).is_err());
        assert!(probe::resolve_identity(true, Some(H), Some(""), boom).is_err());
        // Dev: env ignored, measurement wins (a dev build cannot be made to lie).
        let g = probe::resolve_identity(false, Some(H), Some("0"), measured).unwrap();
        assert_eq!(g, measured());
        let g = probe::resolve_identity(false, None, None, measured).unwrap();
        assert_eq!(g, measured());
    }

    #[test]
    fn pyproject_version_reads_only_the_project_table() {
        let t = "[build-system]\nversion = \"9\"\n[project]\nname = \"ferric\"\nversion = \"0.1.0.dev0\"\n[tool.x]\nversion = \"7\"\n";
        assert_eq!(probe::pyproject_version(t).as_deref(), Some("0.1.0.dev0"));
        assert_eq!(probe::pyproject_version("[tool]\nversion = \"1\"\n"), None);
        assert_eq!(
            probe::pyproject_version("[project]\nversion = \"\"\n"),
            None
        );
    }

    /// A scratch workspace with one crate, committed.
    fn scratch_repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("crates/a/src")).unwrap();
        std::fs::create_dir_all(root.join("crates/a/tests")).unwrap();
        std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        std::fs::write(
            root.join("pyproject.toml"),
            "[project]\nversion = \"1.0\"\n",
        )
        .unwrap();
        std::fs::write(root.join("README.md"), "readme\n").unwrap();
        std::fs::write(root.join("crates/a/Cargo.toml"), "[package]\n").unwrap();
        std::fs::write(root.join("crates/a/src/lib.rs"), "// v1\n").unwrap();
        std::fs::write(root.join("crates/a/tests/t.rs"), "// t\n").unwrap();
        run_git(root, &["init", "-q"]);
        run_git(root, &["add", "-A"]);
        run_git(
            root,
            &[
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-q",
                "-m",
                "init",
            ],
        );
        dir
    }

    #[test]
    fn clean_scratch_repo_reports_its_head_and_clean() {
        let d = scratch_repo();
        let g = probe::probe_git(d.path());
        assert_eq!(g.commit, run_git(d.path(), &["rev-parse", "HEAD"]));
        assert_eq!(g.dirty, Some(false));
    }

    #[test]
    fn editing_compiled_source_flips_dirty_and_reverting_clears_it() {
        let d = scratch_repo();
        let src = d.path().join("crates/a/src/lib.rs");
        std::fs::write(&src, "// v2\n").unwrap();
        assert_eq!(probe::probe_git(d.path()).dirty, Some(true));
        std::fs::write(&src, "// v1\n").unwrap();
        assert_eq!(probe::probe_git(d.path()).dirty, Some(false));
    }

    #[test]
    fn staged_is_dirty_but_untracked_is_not() {
        let d = scratch_repo();
        std::fs::write(d.path().join("crates/a/src/new.rs"), "// new\n").unwrap();
        assert_eq!(
            probe::probe_git(d.path()).dirty,
            Some(false),
            "untracked src file"
        );
        run_git(d.path(), &["add", "crates/a/src/new.rs"]);
        assert_eq!(
            probe::probe_git(d.path()).dirty,
            Some(true),
            "staged src file"
        );
    }

    #[test]
    fn version_edit_is_dirty() {
        let d = scratch_repo();
        std::fs::write(
            d.path().join("pyproject.toml"),
            "[project]\nversion = \"2.0\"\n",
        )
        .unwrap();
        assert_eq!(probe::probe_git(d.path()).dirty, Some(true));
    }

    #[test]
    fn any_tracked_edit_is_dirty_untracked_files_are_not() {
        let d = scratch_repo();
        std::fs::write(d.path().join("notes.txt"), "untracked\n").unwrap();
        assert_eq!(probe::probe_git(d.path()).dirty, Some(false));
        std::fs::write(d.path().join("README.md"), "changed\n").unwrap();
        assert_eq!(probe::probe_git(d.path()).dirty, Some(true));
    }

    #[test]
    fn no_repository_is_unknown_never_a_guess() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("crates/a/src")).unwrap();
        assert_eq!(probe::probe_git(d.path()), probe::GitState::unknown());
    }

    #[test]
    fn a_tree_nested_inside_another_repo_is_unknown() {
        let d = scratch_repo();
        let nested = d.path().join("vendor");
        std::fs::create_dir_all(nested.join("crates/b/src")).unwrap();
        assert_eq!(probe::probe_git(&nested), probe::GitState::unknown());
    }

    /// In a linked worktree `.git` is a FILE; HEAD, index and the branch ref
    /// must still resolve to real, existing paths.
    #[test]
    fn git_watch_files_resolve_in_a_linked_worktree() {
        let d = scratch_repo();
        let wt = tempfile::tempdir().unwrap();
        let wt_path = wt.path().join("wt");
        run_git(
            d.path(),
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "side",
                wt_path.to_str().unwrap(),
            ],
        );
        assert!(
            wt_path.join(".git").is_file(),
            ".git must be a file in a worktree"
        );
        let files = probe::git_watch_files(&wt_path);
        for name in ["HEAD", "index", "side"] {
            assert!(
                files.iter().any(|p| p.ends_with(name) && p.exists()),
                "{name} missing from {files:?}"
            );
        }
        assert_eq!(
            probe::probe_git(&wt_path).commit,
            run_git(&wt_path, &["rev-parse", "HEAD"])
        );
    }

    #[test]
    fn git_watch_files_include_head_index_and_the_branch_ref() {
        let d = scratch_repo();
        let files = probe::git_watch_files(d.path());
        assert!(files.iter().any(|p| p.ends_with("HEAD")), "{files:?}");
        assert!(files.iter().any(|p| p.ends_with("index")), "{files:?}");
        let branch = run_git(d.path(), &["symbolic-ref", "HEAD"]);
        assert!(files.iter().any(|p| p.ends_with(&branch)), "{files:?}");
    }
}
