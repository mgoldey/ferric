//! `ferric` — alias binary for `ferric-cli`, same code, second name.
//!
//! Both names are load-bearing: docs/examples invoke `ferric`, while
//! `ferric-batch` and the bisect scripts shell out to `ferric-cli`.
//!
//! Calls into the `ferric-cli` LIBRARY crate (formerly `main.rs`, moved to
//! `lib.rs` so `ferric-python`'s `#[pyfunction]` console-script entry point
//! can call the same `ferric_cli::main()` from inside its own `.so` -- see
//! crates/ferric-python/src/lib.rs for the pyfunction wrapper and
//! pyproject.toml's [project.scripts] for the console-script wiring). This
//! binary and `bin/ferric-cli.rs` are now both thin wrappers over the same
//! library, not one wrapping the other via a `#[path]` mod include.
fn main() {
    ferric_cli::main()
}
