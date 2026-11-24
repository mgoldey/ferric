fn main() {
    // libxc: `LIBXC_PREFIX` if set, else scripts/install-libxc.sh's default
    // prefix (~/.local/libxc-7.0.0) when it exists, else whatever the linker
    // finds on its default / `LIBRARY_PATH` / `-L` search path (the wheels
    // workflows put their own build there). The script rewrites the SONAME to
    // the absolute library path, so binaries linked here load that exact file
    // with no rpath or LD_LIBRARY_PATH. The linked version is checked at test
    // time (`libxc::tests::linked_libxc_is_the_pinned_major`), not here: the
    // header is not on a predictable path in every build environment.
    let prefix = std::env::var("LIBXC_PREFIX").ok().or_else(|| {
        let home = std::env::var("HOME").ok()?;
        let installer = format!("{home}/.local/libxc-7.0.0");
        std::path::Path::new(&format!("{installer}/include/xc_version.h"))
            .exists()
            .then_some(installer)
    });
    if let Some(prefix) = &prefix {
        println!("cargo:rustc-link-search=native={prefix}/lib");
        println!("cargo:rerun-if-changed={prefix}/include/xc_version.h");
    }
    println!("cargo:rustc-link-lib=xc");
    // ferric-dft's own targets now call ndarray GEMMs (fxc/vxc BLAS3 paths),
    // so this crate's test binaries need a cblas provider directly — same
    // pattern as ferric-tensors/build.rs. Downstream crates already linked
    // openblas via ferric-integrals, which is why this was latent until now.
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIBXC_PREFIX");
}
