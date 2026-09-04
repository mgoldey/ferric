//! `ferric-cli` binary -- thin wrapper over the `ferric-cli` library crate.
//!
//! `main.rs` moved to `lib.rs` so `ferric-python` can call `ferric_cli::main`
//! from a `#[pyfunction]` inside its own `.so` (the `ferric` console-script
//! entry point), without a second compiled binary in the wheel. This is now
//! the SAME shape `bin/ferric.rs` already used for its own aliasing, just one
//! layer up: `[[bin]]` targets are thin wrappers, the crate root is the
//! library.
fn main() {
    ferric_cli::main()
}
