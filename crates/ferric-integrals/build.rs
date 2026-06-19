use std::path::PathBuf;
use std::process::Command;

fn main() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/matt".to_string());
    let local_prefix = format!("{home}/.local");

    // --- libecpint: configure + build the vendored static library via CMake ---
    let (ecpint_lib_dir, ecpint_include_dirs) = build_libecpint();

    // --- libint2 shim (unchanged) ---
    cc::Build::new()
        .cpp(true)
        .file("shim/shim.cc")
        .include(format!("{local_prefix}/include"))
        .include(format!("{local_prefix}/include/libint2"))
        .include("/usr/local/include")
        .include("/usr/local/include/libint2")
        .include("/usr/include/eigen3")
        .flag("-std=c++17")
        .flag("-O2")
        .flag("-Wno-deprecated-declarations")
        .flag("-Wno-unused-parameter")
        .compile("ferric_shim");

    // --- ECP shim (new): wraps libecpint's ECPIntegrator into a C-ABI matrix call ---
    let mut ecp_build = cc::Build::new();
    ecp_build
        .cpp(true)
        .file("shim/ecp_shim.cc")
        .flag("-std=c++11")
        .flag("-O2")
        .flag("-Wno-deprecated-declarations")
        .flag("-Wno-unused-parameter");
    for inc in &ecpint_include_dirs {
        ecp_build.include(inc);
    }
    ecp_build.compile("ferric_ecp_shim");

    // libint2 + BLAS link
    println!("cargo:rustc-link-search=native={local_prefix}/lib");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=static=int2");
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // libecpint link (static): ecpint + its internal Faddeeva
    println!("cargo:rustc-link-search=native={}", ecpint_lib_dir.display());
    println!("cargo:rustc-link-lib=static=ecpint");
    println!("cargo:rustc-link-lib=static=Faddeeva");

    println!("cargo:rerun-if-changed=shim/shim.h");
    println!("cargo:rerun-if-changed=shim/shim.cc");
    println!("cargo:rerun-if-changed=shim/ecp_shim.h");
    println!("cargo:rerun-if-changed=shim/ecp_shim.cc");
    println!("cargo:rerun-if-changed=shim/libecpint/CMakeLists.txt");
    println!("cargo:rerun-if-changed=shim/libecpint/src");
    println!("cargo:rerun-if-changed=shim/libecpint/include");
}

/// Configure and build the vendored libecpint static library with CMake.
/// Returns (directory containing libecpint.a + libFaddeeva.a, include dirs for the shim).
fn build_libecpint() -> (PathBuf, Vec<PathBuf>) {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src_dir = manifest_dir.join("shim/libecpint");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("libecpint-build");
    std::fs::create_dir_all(&build_dir).expect("create libecpint build dir");

    // libecpint installs static archives flat into <build>/libecpint.a and
    // external/Faddeeva/libFaddeeva.a. We build in place (no install) and link
    // from the build tree.
    let ecpint_a = build_dir.join("src/libecpint.a");

    if !ecpint_a.exists() {
        // Configure.
        let status = Command::new("cmake")
            .current_dir(&build_dir)
            .arg(&src_dir)
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg("-DBUILD_SHARED_LIBS=OFF")
            .arg("-DLIBECPINT_USE_PUGIXML=OFF")
            .arg("-DLIBECPINT_BUILD_TESTS=OFF")
            .arg("-DLIBECPINT_BUILD_DOCS=OFF")
            .arg("-DLIBECPINT_MAX_L=5")
            .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
            .status()
            .expect("failed to run cmake configure for libecpint");
        assert!(status.success(), "libecpint cmake configure failed");

        // Build (just the ecpint static target and its Faddeeva dependency).
        let jobs = std::env::var("NUM_JOBS").unwrap_or_else(|_| "2".to_string());
        let status = Command::new("cmake")
            .current_dir(&build_dir)
            .arg("--build")
            .arg(".")
            .arg("--target")
            .arg("ecpint")
            .arg("-j")
            .arg(&jobs)
            .status()
            .expect("failed to build libecpint");
        assert!(status.success(), "libecpint build failed");
    }

    // The two static archives live in different subdirs of the build tree; copy
    // both next to each other so a single -L search dir resolves them.
    let lib_out = out_dir.join("libecpint-lib");
    std::fs::create_dir_all(&lib_out).expect("create libecpint lib dir");
    std::fs::copy(&ecpint_a, lib_out.join("libecpint.a")).expect("copy libecpint.a");
    let faddeeva_a = build_dir.join("external/Faddeeva/libFaddeeva.a");
    std::fs::copy(&faddeeva_a, lib_out.join("libFaddeeva.a")).expect("copy libFaddeeva.a");

    // The generated config.hpp lives in the build tree's include dir.
    let include_dirs = vec![
        src_dir.join("include"),
        src_dir.join("include/libecpint"),
        build_dir.join("include/libecpint"),
        build_dir.join("include"),
    ];
    (lib_out, include_dirs)
}
