fn main() {
    println!("cargo:rustc-link-lib=xc");
    // ferric-dft's own targets now call ndarray GEMMs (fxc/vxc BLAS3 paths),
    // so this crate's test binaries need a cblas provider directly — same
    // pattern as ferric-tensors/build.rs. Downstream crates already linked
    // openblas via ferric-integrals, which is why this was latent until now.
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rerun-if-changed=build.rs");
    // TODO: add `cargo:rerun-if-env-changed=LIBXC_DIR` (and LD_LIBRARY_PATH) so
    // that incremental builds re-link when the system libxc installation changes.
    // Currently omitted to keep the build script minimal; revisit before T5.
}
