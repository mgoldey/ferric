//! Build-time probes, shared by `build.rs` (which embeds their results) and the
//! crate's unit tests (which exercise them against throwaway git repositories).
//!
//! Everything here is a plain function of a workspace root path, so the dirty
//! logic can be tested by creating a scratch repo and editing files in it,
//! never by dirtying the real checkout.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Spelled-out value for "could not be determined". Never an empty string and
/// never a guess.
pub const UNKNOWN: &str = "unknown";

/// What git says about the source tree a build came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitState {
    /// Full `HEAD` hash, or [`UNKNOWN`].
    pub commit: String,
    /// Whether any TRACKED file differs from `HEAD` (staged or unstaged);
    /// untracked files never count. `None` when it could not be determined.
    pub dirty: Option<bool>,
}

impl GitState {
    pub fn unknown() -> Self {
        GitState {
            commit: UNKNOWN.to_string(),
            dirty: None,
        }
    }
}

/// The paths, relative to the workspace root, whose contents are compiled into
/// ferric: the root manifests, `pyproject.toml` (which carries the version),
/// and each crate's `Cargo.toml`, `build.rs`, `src/` and `shim/`.
///
/// The build script watches these (`rerun-if-changed`), so an edit to any
/// compiled source re-runs the probe and the dirty flag cannot go stale while
/// the code it describes changes. Tests, docs and scripts are absent: editing
/// them does not re-run the probe (the flag is whole-tree, so it refreshes at
/// the next compiled-source edit, commit or staging).
#[allow(dead_code)] // used by build.rs only; the lib compiles this file for tests
pub fn compiled_inputs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for f in ["Cargo.toml", "Cargo.lock", "pyproject.toml"] {
        if root.join(f).exists() {
            out.push(PathBuf::from(f));
        }
    }
    let mut crates: Vec<PathBuf> = std::fs::read_dir(root.join("crates"))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default();
    crates.sort();
    for dir in crates {
        let Some(name) = dir.file_name() else {
            continue;
        };
        for sub in ["Cargo.toml", "build.rs", "src", "shim"] {
            if dir.join(sub).exists() {
                out.push(Path::new("crates").join(name).join(sub));
            }
        }
    }
    out
}

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

/// `true` when `root` is itself the top level of a git work tree. A source
/// tree that merely sits INSIDE some other repository (an unpacked sdist in a
/// user's project, say) must not report that repository's commit.
fn is_toplevel(root: &Path) -> bool {
    let Some(top) = git(root, &["rev-parse", "--show-toplevel"]) else {
        return false;
    };
    match (
        std::fs::canonicalize(top.trim()),
        std::fs::canonicalize(root),
    ) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// Probe `root` for its commit and dirty state. Never panics; anything that
/// cannot be determined is reported as unknown.
pub fn probe_git(root: &Path) -> GitState {
    if !is_toplevel(root) {
        return GitState::unknown();
    }
    let Some(head) = git(root, &["rev-parse", "HEAD"]) else {
        return GitState::unknown();
    };
    let head = head.trim();
    let is_hash = matches!(head.len(), 40 | 64) && head.bytes().all(|b| b.is_ascii_hexdigit());
    if !is_hash {
        return GitState::unknown();
    }
    // Tracked files only: an untracked file (a scratch note, a stray build
    // artifact) is not part of what HEAD describes and must not mark the build
    // dirty. The whole tree is checked, not just compiled inputs.
    let dirty =
        git(root, &["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.trim().is_empty());
    GitState {
        commit: head.to_string(),
        dirty,
    }
}

/// Git files whose change means `HEAD` or the tracked state moved: `HEAD`
/// itself, the ref it points at (a commit on a branch changes only that file),
/// `index` (staging) and `packed-refs`. In a
/// worktree these live in different directories; `--git-path` resolves each.
/// Only existing files are returned, because cargo re-runs a build script on
/// EVERY build when a watched path does not exist.
pub fn git_watch_files(root: &Path) -> Vec<PathBuf> {
    let mut names = vec![
        "HEAD".to_string(),
        "index".to_string(),
        "packed-refs".to_string(),
    ];
    if let Some(r) = git(root, &["symbolic-ref", "-q", "HEAD"]) {
        names.push(r.trim().to_string());
    }
    names
        .iter()
        .filter_map(|n| git(root, &["rev-parse", "--git-path", n]))
        .map(|p| {
            let p = PathBuf::from(p.trim());
            if p.is_absolute() {
                p
            } else {
                root.join(p)
            }
        })
        .filter(|p| p.exists())
        .collect()
}

/// The `version` key of the `[project]` table in a `pyproject.toml`, which is
/// the one version maturin puts on the wheel (and the one the release
/// workflow stamps from the tag).
pub fn pyproject_version(text: &str) -> Option<String> {
    let mut in_project = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_project = line == "[project]";
            continue;
        }
        if !in_project {
            continue;
        }
        let Some(rest) = line.strip_prefix("version") else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let v = rest.trim().strip_prefix('"')?.split('"').next()?;
        if v.is_empty() {
            return None;
        }
        return Some(v.to_string());
    }
    None
}

/// Decide the build's git identity.
///
/// DEV build (`release == false`): `env_sha` / `env_dirty` are IGNORED and
/// `measure` (the real `git` probe) decides, so setting environment variables
/// can never make a dev build claim a commit it is not.
///
/// RELEASE build: ONLY the environment is used and `measure` is never called.
/// The release workflow computes the commit and the clean state on the host
/// BEFORE it stamps the tag's version into the tree, so by the time this runs
/// the tree is intentionally modified in exactly the verified version lines;
/// a fresh `git status` here would (correctly) say dirty and say nothing
/// useful. A missing or malformed value is an error: the sha must be 40 hex
/// digits, and the dirty flag exactly `0` or `1`.
pub fn resolve_identity(
    release: bool,
    env_sha: Option<&str>,
    env_dirty: Option<&str>,
    measure: impl FnOnce() -> GitState,
) -> Result<GitState, String> {
    if !release {
        return Ok(measure());
    }
    let sha = env_sha.ok_or("release build: FERRIC_GIT_SHA is not set")?;
    if !(sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit())) {
        return Err(format!(
            "release build: FERRIC_GIT_SHA must be 40 hex digits, got {sha:?}"
        ));
    }
    let dirty = match env_dirty {
        Some("0") => false,
        Some("1") => true,
        Some(o) => {
            return Err(format!(
                "release build: FERRIC_GIT_DIRTY must be 0 or 1, got {o:?}"
            ))
        }
        None => return Err("release build: FERRIC_GIT_DIRTY is not set".into()),
    };
    Ok(GitState {
        commit: sha.to_ascii_lowercase(),
        dirty: Some(dirty),
    })
}

/// The release gate. A release build (`FERRIC_RELEASE_BUILD=1`) must know its
/// commit and must be built from a clean tree; "unknown" is a failure, never
/// treated as clean. A dev build is always accepted (it reports whatever it
/// found).
pub fn release_gate(release: bool, git: &GitState) -> Result<(), String> {
    if !release {
        return Ok(());
    }
    if git.commit == UNKNOWN {
        return Err("release build, but the git commit is unknown; refusing to build a release wheel with no provenance".to_string());
    }
    match git.dirty {
        Some(false) => Ok(()),
        Some(true) => Err(format!(
            "release build from a DIRTY tree (tracked files differ from {}); commit or revert first",
            git.commit
        )),
        None => Err("release build, but the dirty state could not be determined; unknown is never treated as clean".to_string()),
    }
}
