//! Locations in the checkout, independent of the test's working directory.

use std::path::PathBuf;

/// The repository root (the directory holding the workspace `Cargo.toml`).
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// `<workspace>/testdata/<rel>`.
pub fn testdata(rel: &str) -> PathBuf {
    workspace_root().join("testdata").join(rel)
}
