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
    for p in ["../../.git/HEAD", "../../.git/ORIG_HEAD"] {
        if std::path::Path::new(p).exists() {
            println!("cargo:rerun-if-changed={p}");
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
