//! The one lookup every ferric-cli integration test uses to find the
//! `ferric-cli` executable at RUN time.
//!
//! `env!("CARGO_BIN_EXE_ferric-cli")` is baked in when the test binary is
//! COMPILED. Under `cargo nextest archive` the binaries are built in one job
//! and run from an extracted copy in another, so that constant points at the
//! build job's target directory, which does not exist there. The lookup order:
//!
//! 1. the RUN-time `CARGO_BIN_EXE_ferric-cli` variable: cargo sets it for
//!    `cargo test`, and nextest sets it to the extracted (remapped) path when
//!    it runs an archive;
//! 2. `ferric-cli` next to the profile directory that holds the test binary,
//!    searching the test binary's ancestors, which covers both cargo layouts
//!    (`target/<profile>/deps/<test>` before Rust 1.100 and
//!    `target/<profile>/build/<pkg>/<hash>/out/<test>` since);
//! 3. the compile-time path, for a plain `cargo test` on a checkout that has
//!    not moved.
//!
//! A candidate must be an existing file, so a stale path never wins silently.
//! Included by each test with `#[path = "support/cli_bin.rs"] mod cli_bin;`
//! (`"../support/cli_bin.rs"` from a `<name>/mod.rs` test).

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Path to the `ferric-cli` executable; see the module docs for the order.
pub fn ferric_cli_bin() -> PathBuf {
    if let Some(p) = std::env::var_os("CARGO_BIN_EXE_ferric-cli").map(PathBuf::from) {
        if p.is_file() {
            return p;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(p) = sibling_in_ancestors(&exe) {
            return p;
        }
    }
    PathBuf::from(env!("CARGO_BIN_EXE_ferric-cli"))
}

/// `ferric-cli` in the nearest ancestor directory of `exe` that holds one.
/// Bounded to the `target/<profile>/build/<pkg>/<hash>/out` depth.
fn sibling_in_ancestors(exe: &Path) -> Option<PathBuf> {
    exe.ancestors()
        .skip(1)
        .take(6)
        .map(|dir| dir.join("ferric-cli"))
        .find(|p| p.is_file())
}
