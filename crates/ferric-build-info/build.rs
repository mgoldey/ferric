//! Embed ferric's build identity: the package version, the git commit, whether
//! the compiled sources differed from that commit, and the cargo profile.
//!
//! Resolved here, at build time, because the binary may later run far from
//! its checkout (a symlinked `.so` in a venv, a copied wheel); a runtime
//! `git rev-parse` would describe whatever tree the process happens to sit
//! in, not the code that is running.

#[path = "src/probe.rs"]
mod probe;

use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
    // crates/ferric-build-info -> workspace root.
    let root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate lives at <root>/crates/ferric-build-info")
        .to_path_buf();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/probe.rs");
    // Editing any compiled source re-runs this script, so the dirty flag
    // cannot go stale while the code it describes changes underneath it.
    for p in probe::compiled_inputs(&root) {
        println!("cargo:rerun-if-changed={}", root.join(p).display());
    }
    for p in probe::git_watch_files(&root) {
        println!("cargo:rerun-if-changed={}", p.display());
    }

    let version = std::fs::read_to_string(root.join("pyproject.toml"))
        .ok()
        .and_then(|t| probe::pyproject_version(&t))
        .unwrap_or_else(|| {
            println!(
                "cargo:warning=ferric-build-info: no [project] version in {}; reporting \"{}\"",
                root.join("pyproject.toml").display(),
                probe::UNKNOWN
            );
            probe::UNKNOWN.to_string()
        });

    // Release builds (FERRIC_RELEASE_BUILD=1, set by the wheel workflow) take
    // their identity ONLY from FERRIC_GIT_SHA / FERRIC_GIT_DIRTY, which the
    // workflow computed on the host before stamping the tag's version; dev
    // builds ignore those variables and measure git themselves. See
    // probe::resolve_identity.
    for v in ["FERRIC_RELEASE_BUILD", "FERRIC_GIT_SHA", "FERRIC_GIT_DIRTY"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let release = std::env::var("FERRIC_RELEASE_BUILD").as_deref() == Ok("1");
    let git = probe::resolve_identity(
        release,
        std::env::var("FERRIC_GIT_SHA").ok().as_deref(),
        std::env::var("FERRIC_GIT_DIRTY").ok().as_deref(),
        || probe::probe_git(&root),
    )
    .unwrap_or_else(|msg| panic!("ferric-build-info: {msg}"));
    if let Err(msg) = probe::release_gate(release, &git) {
        panic!("ferric-build-info: {msg}");
    }
    let dirty = match git.dirty {
        Some(true) => "true",
        Some(false) => "false",
        None => probe::UNKNOWN,
    };
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| probe::UNKNOWN.to_string());

    println!("cargo:rustc-env=FERRIC_BUILD_VERSION={version}");
    println!("cargo:rustc-env=FERRIC_BUILD_COMMIT={}", git.commit);
    println!("cargo:rustc-env=FERRIC_BUILD_DIRTY={dirty}");
    println!("cargo:rustc-env=FERRIC_BUILD_PROFILE={profile}");
}
