use std::path::PathBuf;
use std::process::Command;

fn main() {
    let local_prefix = match std::env::var("LIBINT2_PREFIX") {
        Ok(p) => p,
        Err(_) => {
            let home = std::env::var("HOME")
                .expect("$HOME must be set to locate libint2; set LIBINT2_PREFIX to override");
            // scripts/install-libint.sh's default prefix first, then ~/.local
            // (a from-source libint2 install). Not the reverse: an old static
            // libint2.a in ~/.local would otherwise shadow a newer install.
            let installer = format!("{home}/.local/libint2-2.13.1");
            if std::path::Path::new(&format!("{installer}/include/libint2.hpp")).exists() {
                installer
            } else {
                format!("{home}/.local")
            }
        }
    };

    // ONE libint2 root for headers AND library. Stacking a second libint2
    // include directory behind the prefix let a missing or incomplete prefix
    // fall through silently to another version's headers: a conda build
    // compiled the shim against /usr/local's 2.7.2 headers
    // (LIBINT2_MAX_DERIV_ORDER undefined) and linked conda's 2.13.1 library,
    // and every SCF segfaulted.
    let libint_root = libint_root(&local_prefix);
    println!("cargo:warning=libint2 headers and library from {libint_root}");

    // --- libecpint: configure + build the vendored static library via CMake ---
    let (ecpint_lib_dir, ecpint_include_dirs) = build_libecpint();

    // --- libint2 header override (GmEval per-call copy fix) ---
    let boys_override = libint2_boys_override(&libint_root);

    // --- libint2 shim ---
    let mut shim_build = cc::Build::new();
    shim_build.cpp(true).file("shim/shim.cc");
    if let Some(dir) = &boys_override {
        // MUST precede every libint2 include dir: shim.cc reaches boys.h via
        // engine.impl.h's `#include <libint2/boys.h>`, and the first -I wins.
        shim_build.include(dir);
    }
    shim_build
        .include(format!("{libint_root}/include"))
        .include(format!("{libint_root}/include/libint2"))
        // Eigen under the libint2 prefix: a conda build (conda/recipe.yaml)
        // sets LIBINT2_PREFIX=$PREFIX, and conda-forge's `eigen` installs to
        // $PREFIX/include/eigen3. Listed before the system path so the
        // environment's Eigen wins over any system copy.
        .include(format!("{libint_root}/include/eigen3"))
        .include("/usr/include/eigen3")
        // STEP 4: opt in to the libint2-native terf operator. Requires a
        // libint2 patched with Operator::terf (see
        // wiki/perf-tasks/patches/libint2-terf-*.patch) selected via
        // LIBINT2_PREFIX. Off by default so stock libint2 still builds.
        .define(
            if std::env::var("FERRIC_LIBINT2_TERF").is_ok() {
                "FERRIC_LIBINT2_TERF"
            } else {
                "FERRIC_UNUSED_TERF"
            },
            None,
        )
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
    println!("cargo:rustc-link-search=native={libint_root}/lib");
    // A prefix with libint2.a (a from-source build, e.g. the old mpqc4
    // tarball) links statically. scripts/install-libint.sh installs the
    // conda-forge 2.13.1 build, which ships libint2.so only; that script sets
    // its SONAME to the absolute path, so binaries find it at run time without
    // an rpath.
    if std::path::Path::new(&format!("{libint_root}/lib/libint2.a")).exists() {
        println!("cargo:rustc-link-lib=static=int2");
    } else {
        println!("cargo:rustc-link-lib=dylib=int2");
    }
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // libecpint link (static): ecpint + its internal Faddeeva
    println!(
        "cargo:rustc-link-search=native={}",
        ecpint_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=static=ecpint");
    println!("cargo:rustc-link-lib=static=Faddeeva");

    println!("cargo:rerun-if-env-changed=LIBINT2_PREFIX");
    println!("cargo:rerun-if-changed={libint_root}/include/libint2/config.h");
    println!("cargo:rerun-if-env-changed=FERRIC_LIBINT2_STOCK_BOYS");
    for (version, _) in STOCK_BOYS_H_FNV1A64 {
        println!("cargo:rerun-if-changed=shim/libint2_overrides/{version}/libint2/boys.h");
    }
    println!("cargo:rerun-if-changed=shim/shim.h");
    println!("cargo:rerun-if-changed=shim/shim.cc");
    println!("cargo:rerun-if-changed=shim/ecp_shim.h");
    println!("cargo:rerun-if-changed=shim/ecp_shim.cc");
    println!("cargo:rerun-if-changed=shim/libecpint/CMakeLists.txt");
    println!("cargo:rerun-if-changed=shim/libecpint/src");
    println!("cargo:rerun-if-changed=shim/libecpint/include");
}

/// The single libint2 install the shim compiles and links against: the
/// configured prefix if it has libint2 headers, else `/usr/local` if it does.
/// Panics otherwise, naming both, so a missing install fails the build instead
/// of picking up a different version.
fn libint_root(prefix: &str) -> String {
    let has_headers =
        |root: &str| std::path::Path::new(&format!("{root}/include/libint2.hpp")).exists();
    if has_headers(prefix) {
        return prefix.to_string();
    }
    if std::env::var("LIBINT2_PREFIX").is_err() && has_headers("/usr/local") {
        return "/usr/local".to_string();
    }
    panic!(
        "libint2 headers not found: no {prefix}/include/libint2.hpp{}. \
         Set LIBINT2_PREFIX to a libint2 install (scripts/install-libint.sh).",
        if std::env::var("LIBINT2_PREFIX").is_err() {
            " and no /usr/local/include/libint2.hpp"
        } else {
            ""
        }
    );
}

/// FNV-1a 64 of each STOCK libint2 `include/libint2/boys.h` for which
/// `shim/libint2_overrides/<version>/libint2/boys.h` holds a patched copy.
/// - 2.7.2: mpqc4 tarball `libint-2.7.2-mpqc4.tgz`
///   (sha256 880592b9026352c871c96877a43788edd22edb0b1057ab5a73657c40b3a61505).
/// - 2.13.1: conda-forge `libint-2.13.1-h83f5b4b_0.conda` as installed by
///   scripts/install-libint.sh
///   (sha256 890ab0261a2be2517d3d4cedfd7d07e2a1c6be7b6f057dc7ea2ea418e17067c1).
const STOCK_BOYS_H_FNV1A64: [(&str, u64); 2] = [
    ("2.7.2", 0x1d51_ac52_f207_6ad8),
    ("2.13.1", 0x0f6f_0c23_194e_18ad),
];

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Decide whether to put `shim/libint2_overrides/<version>` first on the
/// shim's include path. Each such directory holds a patched copy of that
/// libint2 version's `boys.h` in which `GenericGmEval::eval` no longer
/// copy-constructs the erf/erfc evaluator per primitive quartet (see
/// shim/libint2_overrides/README.md).
///
/// The override is a whole-file replacement, so it is applied ONLY when the
/// boys.h under `libint_root` (the one root the shim compiles against) is
/// byte-identical to one of the stock files listed in `STOCK_BOYS_H_FNV1A64`,
/// and then the override made from THAT version is used. Any other libint2
/// (another version, the terf-patched prefix, a distro build) gets its own
/// header untouched plus a warning: silently shadowing a different version's
/// boys.h would be far worse than the slowdown.
/// `FERRIC_LIBINT2_STOCK_BOYS=1` forces the stock header (A/B runs).
fn libint2_boys_override(libint_root: &str) -> Option<PathBuf> {
    if std::env::var_os("FERRIC_LIBINT2_STOCK_BOYS").is_some() {
        println!(
            "cargo:warning=FERRIC_LIBINT2_STOCK_BOYS set: libint2 GmEval copy fix NOT applied"
        );
        return None;
    }
    let found = PathBuf::from(format!("{libint_root}/include/libint2/boys.h"));
    println!("cargo:rerun-if-changed={}", found.display());
    let bytes = match std::fs::read(&found) {
        Ok(b) => b,
        Err(e) => {
            println!(
                "cargo:warning=cannot read {} ({e}); libint2 GmEval copy fix NOT applied",
                found.display()
            );
            return None;
        }
    };
    let hash = fnv1a64(&bytes);
    let Some((version, _)) = STOCK_BOYS_H_FNV1A64.iter().find(|(_, h)| *h == hash) else {
        println!(
            "cargo:warning={} (FNV-1a 64 {hash:#018x}) is not a stock libint2 boys.h ferric has an override for ({}); libint2 GmEval copy fix NOT applied (erf/erfc integrals scale poorly across threads)",
            found.display(),
            STOCK_BOYS_H_FNV1A64
                .iter()
                .map(|(v, _)| *v)
                .collect::<Vec<_>>()
                .join(", ")
        );
        return None;
    };
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let dir = manifest_dir.join(format!("shim/libint2_overrides/{version}"));
    if !dir.join("libint2/boys.h").is_file() {
        // A stock hash with no override file is a repo-layout bug, not a
        // user environment problem: fail loudly rather than silently skip.
        panic!(
            "{} matches stock libint2 {version} but {} is missing",
            found.display(),
            dir.join("libint2/boys.h").display()
        );
    }
    println!("cargo:warning=libint2 {version} boys.h override applied (GmEval copy fix)");
    Some(dir)
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
