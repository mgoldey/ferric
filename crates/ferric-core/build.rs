//! Link `libopenblas` for `ferric-core`'s own test/doc binaries.
//!
//! `ferric-core::blas_threads` (moved here from `ferric-integrals` so any
//! crate can reach it without a new dependency edge — see that module's doc)
//! calls the OpenBLAS runtime API (`openblas_get_num_threads` /
//! `openblas_set_num_threads`) directly via `extern "C"`. In every downstream
//! binary (ferric-cli, the Python extension, every integration-test binary)
//! this symbol is already resolved because `ferric-integrals`'s build script
//! emits `cargo:rustc-link-lib=dylib=openblas` (see that crate's `build.rs`)
//! and every such binary depends on `ferric-integrals` too. But `ferric-core`
//! itself has NO dependency on `ferric-integrals` (that dependency would be
//! backwards — `ferric-integrals` depends on `ferric-core`), so `ferric-core`'s
//! OWN `cargo test -p ferric-core` binary has nothing upstream emitting that
//! link directive and fails at link time. Mirror the same directive here,
//! scoped to this crate, so `ferric-core`'s unit tests link standalone too.
fn main() {
    println!("cargo:rustc-link-lib=dylib=openblas");
}
