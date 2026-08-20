//! Link configuration for the optional `xtb` feature.
//!
//! When the `xtb` feature is OFF (the default) this script emits nothing, so a
//! machine with no libxtb installed still builds the workspace. When it is ON we
//! add the same `$HOME/.local` prefix that ferric's other native dependencies
//! use (see `crates/ferric-integrals/build.rs`), plus the standard system dirs,
//! and link `libxtb` dynamically.
//!
//! Override the search prefix with `XTB_PREFIX=/some/where` (expects
//! `$XTB_PREFIX/lib/libxtb.so` and `$XTB_PREFIX/include/xtb.h`).

fn main() {
    println!("cargo:rerun-if-env-changed=XTB_PREFIX");

    if std::env::var_os("CARGO_FEATURE_XTB").is_none() {
        // Feature disabled: no native dependency at all.
        return;
    }

    let home = std::env::var("HOME")
        .expect("$HOME must be set to locate system libraries; set XTB_PREFIX to override");
    let prefix = std::env::var("XTB_PREFIX").unwrap_or_else(|_| format!("{home}/.local"));

    println!("cargo:rustc-link-search=native={prefix}/lib");
    println!("cargo:rustc-link-search=native={prefix}/lib64");
    // meson installs into a multiarch subdir on Debian/Ubuntu
    // (e.g. ~/.local/lib/x86_64-linux-gnu/libxtb.so).
    let multiarch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if multiarch == "x86_64" {
        println!("cargo:rustc-link-search=native={prefix}/lib/x86_64-linux-gnu");
    } else if multiarch == "aarch64" {
        println!("cargo:rustc-link-search=native={prefix}/lib/aarch64-linux-gnu");
    }
    println!("cargo:rustc-link-search=native=/usr/local/lib");

    // RUNTIME search path, not just link-time. Without this a built binary dies
    // with "libxtb.so.6: cannot open shared object file" unless the caller
    // exports LD_LIBRARY_PATH by hand -- meson installs libxtb into the
    // multiarch subdir (~/.local/lib/x86_64-linux-gnu), which is not on the
    // default loader path. Emitting rpath here makes `cargo test`/`cargo run`
    // work with no environment setup.
    for dir in [
        format!("{prefix}/lib"),
        format!("{prefix}/lib64"),
        format!("{prefix}/lib/{}-linux-gnu", std::env::consts::ARCH),
    ] {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    println!("cargo:rustc-link-lib=dylib=xtb");

    // libxtb is built with OpenMP (`-Dopenmp=true`); gfortran runtime and
    // libgomp come in via the shared library's own DT_NEEDED, so we do not
    // repeat them here.

    // Bake in the parameter-file directory discovered at build time. libxtb
    // reads GFN parameters from $XTBPATH when the environment is constructed;
    // without it, loading a Hamiltonian fails with "Parameter file ... not
    // found". Baking the build-time prefix means callers do not have to export
    // XTBPATH themselves, while an explicitly-set XTBPATH still wins at runtime.
    println!("cargo:rustc-env=FERRIC_XTB_PARAM_DIR={prefix}/share/xtb");
}
