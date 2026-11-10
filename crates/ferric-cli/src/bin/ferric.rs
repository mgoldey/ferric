//! `ferric` — alias binary for `ferric-cli`, same code, second name.
//!
//! Both names are in use: docs, examples and `ferric-batch` run `ferric`;
//! some scripts invoke `ferric-cli`.
//!
//! Calls into the `ferric-cli` LIBRARY crate (formerly `main.rs`, moved to
//! `lib.rs` so `ferric-python`'s `#[pyfunction]` console-script entry point
//! can call the same CLI logic (`ferric_cli::run`, with Python's `sys.argv`)
//! from inside its own `.so` -- see
//! crates/ferric-python/src/lib.rs for the pyfunction wrapper and
//! pyproject.toml's [project.scripts] for the console-script wiring). This
//! binary and `bin/ferric-cli.rs` are now both thin wrappers over the same
//! library, not one wrapping the other via a `#[path]` mod include.
fn main() {
    ferric_cli::main()
}
