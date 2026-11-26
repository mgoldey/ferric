// build.rs for ferric-cuest.
//
// Feature `cuest` off (default): nothing to do; the crate is CPU-only.
//
// Feature `cuest` on:
//   1. Locate a cuEST install: include/cuest.h + lib/libcuest.so.
//      Search order: $CUEST_ROOT, $CONDA_PREFIX, /opt/nvidia/cuest,
//      /usr/local/cuda, /usr.
//   2. Locate the CUDA runtime (libcudart.so): the cuEST root's lib first
//      (conda installs both side by side), then $CUDA_PATH/$CUDA_HOME,
//      /usr/local/cuda, /usr/lib/x86_64-linux-gnu.
//   3. Compile csrc/abi_check.c against the installed cuest.h. That file
//      restates every prototype and enum value src/ffi.rs relies on; any
//      mismatch is a C compile error, so a wrong FFI never reaches a GPU.
//   4. Link cuest + cudart and add an rpath so tests/examples run without
//      LD_LIBRARY_PATH.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CUEST_ROOT");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=csrc/abi_check.c");
    if std::env::var("CARGO_FEATURE_CUEST").is_err() {
        return;
    }

    let root = find_cuest_root().unwrap_or_else(|| {
        panic!(
            "ferric-cuest: feature `cuest` is on but no cuEST install was found.\n\
             Looked for include/cuest.h + lib/libcuest.so under $CUEST_ROOT, $CONDA_PREFIX,\n\
             /opt/nvidia/cuest, /usr/local/cuda and /usr. Install cuEST 0.3.x (for example\n\
             `scripts/cuest/fetch_cuest.sh` or `conda install -c conda-forge libcuest-dev`)\n\
             and set CUEST_ROOT, or build without the `cuest` feature."
        )
    });
    let inc = root.join("include");
    let lib = root.join("lib");
    check_version(&inc.join("cuest.h"));

    let cudart_dir = find_cudart(&lib).unwrap_or_else(|| {
        panic!(
            "ferric-cuest: found cuEST at {} but no libcudart.so (CUDA runtime).\n\
             Set CUDA_PATH to a CUDA 12.x/13.x toolkit, or install cuda-cudart-dev into the conda env.",
            root.display()
        )
    });

    // Header-vs-FFI agreement check (no NVIDIA text is copied into the repo:
    // the header is read from the user's install at build time).
    cc::Build::new()
        .file("csrc/abi_check.c")
        .include(&inc)
        .flag_if_supported("-Werror=incompatible-pointer-types")
        .flag_if_supported("-Werror=int-conversion")
        .warnings_into_errors(true)
        .compile("ferric_cuest_abi_check");

    for dir in [&lib, &cudart_dir] {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=cuest");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-env=FERRIC_CUEST_ROOT={}", root.display());
}

fn has_cuest(root: &Path) -> bool {
    root.join("include/cuest.h").is_file() && root.join("lib/libcuest.so").exists()
}

fn find_cuest_root() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for var in ["CUEST_ROOT", "CONDA_PREFIX"] {
        if let Ok(r) = std::env::var(var) {
            roots.push(PathBuf::from(r));
        }
    }
    roots.extend(["/opt/nvidia/cuest", "/usr/local/cuda", "/usr"].map(PathBuf::from));
    roots.into_iter().find(|r| has_cuest(r))
}

fn find_cudart(cuest_lib: &Path) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = vec![cuest_lib.to_path_buf()];
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(r) = std::env::var(var) {
            dirs.push(PathBuf::from(&r).join("lib64"));
            dirs.push(PathBuf::from(&r).join("lib"));
        }
    }
    dirs.extend(
        [
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
        ]
        .map(PathBuf::from),
    );
    dirs.into_iter().find(|d| d.join("libcudart.so").exists())
}

/// ffi.rs is written against cuEST 0.3.x; the C API changed between 0.1,
/// 0.2 and 0.3. abi_check.c is the real guard; this gives a clearer message.
fn check_version(header: &Path) {
    let text = std::fs::read_to_string(header).unwrap_or_default();
    let grab = |name: &str| -> Option<u32> {
        text.lines()
            .find_map(|l| l.trim().strip_prefix(&format!("#define {name} ")))
            .and_then(|v| v.trim().parse().ok())
    };
    match (grab("CUEST_VER_MAJOR"), grab("CUEST_VER_MINOR")) {
        (Some(0), Some(3)) => {}
        (maj, min) => panic!(
            "ferric-cuest: {} reports cuEST version {maj:?}.{min:?}; this crate targets cuEST 0.3.x",
            header.display()
        ),
    }
}
