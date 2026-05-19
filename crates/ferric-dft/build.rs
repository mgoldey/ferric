fn main() {
    println!("cargo:rustc-link-lib=xc");
    println!("cargo:rerun-if-changed=build.rs");
    // TODO: add `cargo:rerun-if-env-changed=LIBXC_DIR` (and LD_LIBRARY_PATH) so
    // that incremental builds re-link when the system libxc installation changes.
    // Currently omitted to keep the build script minimal; revisit before T5.
}
