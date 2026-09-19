//! Capture the git SHA at build time for the JSON run log's `git_sha` field.
//!
//! A run log that cannot say which commit produced a number is only half an
//! artifact. Resolving the SHA here, at build time, is the only honest place
//! for it: the binary may later run on another machine, far from the checkout,
//! long after the working tree moved on, so a runtime `git rev-parse` would
//! report whatever tree the process happens to sit in — which may have nothing
//! to do with the code that is running.
//!
//! Emits `FERRIC_GIT_SHA` only on success. `runlog` reads it with
//! `option_env!`, so a build outside a git checkout (a published crate, a
//! vendored source tarball, a Docker build with no `.git`) logs `null` rather
//! than a wrong or stale SHA.
//!
//! A dirty working tree is marked with a `-dirty` suffix, because "the number
//! came from commit abc1234" is false when uncommitted edits were compiled in.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Re-run when HEAD moves, so a rebuild after a commit does not bake in the
    // PREVIOUS commit's SHA. A stale SHA is worse than no SHA: it points a
    // reader at code that did not produce the number.
    //
    // This deliberately does NOT track the working tree's contents, so the
    // `-dirty` marker is only as fresh as the last time this script ran. That
    // is the honest trade: tracking every file would rebuild the crate on
    // every edit. `-dirty` therefore means "was dirty when this was built",
    // and its ABSENCE on a build older than your last edit is not a promise of
    // cleanliness -- which is why the SHA is provenance, not proof.
    //
    // ASK GIT where its directory is; do not guess "../../.git". In a git
    // WORKTREE -- which is how most work in this repo actually happens -- the
    // repo root's `.git` is a FILE containing "gitdir: <path>", not a
    // directory, so `../../.git/HEAD` does not exist and the rerun-if-changed
    // below was never emitted. The build script then did not re-run when HEAD
    // moved, and the binary baked in the PREVIOUS commit's SHA: exactly the
    // stale-SHA failure this file's own header calls worse than no SHA.
    //
    // Two directories, because they differ in a worktree:
    //   --absolute-git-dir  -> .../.git/worktrees/<name>, which holds THIS
    //                          worktree's HEAD
    //   --git-common-dir    -> .../.git, which holds packed-refs
    // In a normal checkout both are the same path and the duplicate is
    // harmless.
    for arg in ["--absolute-git-dir", "--git-common-dir"] {
        let Ok(out) = Command::new("git")
            .args(["rev-parse", "--path-format=absolute", arg])
            .output()
        else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        let Ok(dir) = String::from_utf8(out.stdout) else {
            continue;
        };
        let dir = std::path::Path::new(dir.trim());
        for f in ["HEAD", "ORIG_HEAD", "packed-refs"] {
            let p = dir.join(f);
            if p.exists() {
                println!("cargo:rerun-if-changed={}", p.display());
            }
        }
        // ...AND the ref HEAD points at.
        //
        // On a branch, `HEAD` holds the SYMBOLIC ref `ref: refs/heads/<name>`,
        // and that text does NOT change when you commit -- only the loose ref
        // file does. Watching HEAD alone therefore misses every commit on the
        // current branch, which is the common case, and the baked-in
        // FERRIC_GIT_SHA silently goes stale. (It DOES catch a branch switch,
        // which is presumably why this looked like it worked.)
        //
        // `git rev-parse --git-common-dir` is where refs live; in a WORKTREE
        // the per-worktree `--git-dir` has its own HEAD but shares refs with
        // the main checkout, so resolving the ref against the wrong one finds
        // nothing.
        if let Ok(head) = std::fs::read_to_string(dir.join("HEAD")) {
            if let Some(refname) = head.strip_prefix("ref: ").map(str::trim) {
                let common = std::process::Command::new("git")
                    .args(["rev-parse", "--git-common-dir"])
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| std::path::PathBuf::from(s.trim().to_string()))
                    .unwrap_or_else(|| dir.to_path_buf());
                let ref_path = common.join(refname);
                if ref_path.exists() {
                    println!("cargo:rerun-if-changed={}", ref_path.display());
                }
            }
        }
    }
    // Let a build system (a container build, a CI job with no .git) supply the
    // SHA directly. An explicitly provided value always wins over the probe.
    println!("cargo:rerun-if-env-changed=FERRIC_GIT_SHA");
    if let Ok(sha) = std::env::var("FERRIC_GIT_SHA") {
        if !sha.is_empty() {
            println!("cargo:rustc-env=FERRIC_GIT_SHA={sha}");
            return;
        }
    }
    if let Some(sha) = git_describe() {
        println!("cargo:rustc-env=FERRIC_GIT_SHA={sha}");
    }
}

/// `<short sha>` or `<short sha>-dirty`, or `None` if git is unavailable, this
/// is not a checkout, or the command fails for any reason. Never panics: a
/// missing SHA must not fail a build.
fn git_describe() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if sha.is_empty() {
        return None;
    }
    // `diff --quiet` exits 1 when the working tree differs from HEAD. A
    // failure to run it at all leaves the SHA unmarked rather than falsely
    // marking it dirty.
    let dirty = Command::new("git")
        .args(["diff", "--quiet", "HEAD"])
        .status()
        .map(|st| !st.success())
        .unwrap_or(false);
    Some(if dirty { format!("{sha}-dirty") } else { sha })
}
