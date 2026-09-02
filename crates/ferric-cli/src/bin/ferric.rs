//! `ferric` — alias binary for `ferric-cli`, same code, second name.
//!
//! Both names are load-bearing: docs/examples invoke `ferric`, while
//! `ferric-batch` and the bisect scripts shell out to `ferric-cli`.
//! A thin wrapper (rather than two `[[bin]]` targets sharing src/main.rs)
//! keeps cargo's duplicate-build-target warning away.
#[path = "../main.rs"]
mod cli;

fn main() {
    cli::main()
}
