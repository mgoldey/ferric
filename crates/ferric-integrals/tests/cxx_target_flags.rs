//! The target-conditional C++ build choices of `build.rs`
//! (`build_support/cxx_target.rs`, `include!`d by both), checked for every
//! target on any host: the macOS and Windows branches cannot otherwise run on
//! the Linux box or Linux CI.
//!
//! Mutation ledger (each mutant applied to build_support/cxx_target.rs; the
//! named tests failed, the unmutated file passed):
//! - `msvc_abi` keyed on `os == "windows"` instead of `env == "msvc"`:
//!   `windows_gnu_is_not_msvc`, `cxx_runtime_per_target` and
//!   `non_msvc_targets_keep_the_historical_build_line` fail.
//! - `/EHsc` dropped from `shim_flags`: `msvc_turns_exceptions_on_in_both_shims` fails.
//! - `-O2` moved to the end of the non-MSVC `shim_flags`:
//!   `non_msvc_targets_keep_the_historical_build_line` fails.
//! - `cxx_runtime` returning `stdc++` on macOS: `cxx_runtime_per_target` fails.

#[allow(dead_code)]
mod cxx {
    include!("../build_support/cxx_target.rs");
}

use cxx::{CxxTarget, MSVC_UNISTD_STUB};

const LINUX: CxxTarget<'static> = CxxTarget {
    os: "linux",
    env: "gnu",
};
const MACOS: CxxTarget<'static> = CxxTarget {
    os: "macos",
    env: "",
};
const WIN_MSVC: CxxTarget<'static> = CxxTarget {
    os: "windows",
    env: "msvc",
};
const WIN_GNU: CxxTarget<'static> = CxxTarget {
    os: "windows",
    env: "gnu",
};

/// The literal flag lists build.rs passed before the target split: the Linux
/// (and every non-MSVC) build line must not change.
#[test]
fn non_msvc_targets_keep_the_historical_build_line() {
    for t in [LINUX, MACOS, WIN_GNU] {
        assert_eq!(
            t.shim_flags(),
            [
                "-std=c++17",
                "-O2",
                "-Wno-deprecated-declarations",
                "-Wno-unused-parameter"
            ],
            "{t:?}"
        );
        assert_eq!(
            t.ecp_shim_flags(),
            [
                "-std=c++11",
                "-O2",
                "-Wno-deprecated-declarations",
                "-Wno-unused-parameter"
            ],
            "{t:?}"
        );
        assert!(t.defines().is_empty(), "{t:?}");
        assert!(t.compat_headers().is_empty(), "{t:?}");
        assert_eq!(t.ecpint_archives(), ("libecpint.a", "libFaddeeva.a"));
        assert_eq!(t.cmake_generator(), None);
    }
}

#[test]
fn msvc_turns_exceptions_on_in_both_shims() {
    assert!(WIN_MSVC.shim_flags().contains(&"/EHsc"));
    assert!(WIN_MSVC.ecp_shim_flags().contains(&"/EHsc"));
    // libint2's C++ API needs C++17; MSVC defaults to C++14.
    assert!(WIN_MSVC.shim_flags().contains(&"/std:c++17"));
    assert!(WIN_MSVC.ecp_shim_flags().contains(&"/std:c++14"));
    // No GCC-style flag leaks into a cl.exe command line.
    for f in WIN_MSVC
        .shim_flags()
        .iter()
        .chain(WIN_MSVC.ecp_shim_flags())
    {
        assert!(f.starts_with('/'), "{f}");
    }
    assert_eq!(WIN_MSVC.defines(), ["_USE_MATH_DEFINES"]);
}

#[test]
fn windows_gnu_is_not_msvc() {
    assert!(!WIN_GNU.msvc_abi());
    assert!(WIN_MSVC.msvc_abi());
}

#[test]
fn cxx_runtime_per_target() {
    assert_eq!(LINUX.cxx_runtime(), Some("stdc++"));
    assert_eq!(WIN_GNU.cxx_runtime(), Some("stdc++"));
    assert_eq!(MACOS.cxx_runtime(), Some("c++"));
    assert_eq!(CxxTarget { os: "ios", env: "" }.cxx_runtime(), Some("c++"));
    assert_eq!(WIN_MSVC.cxx_runtime(), None);
}

#[test]
fn msvc_libecpint_archives_and_generator() {
    assert_eq!(WIN_MSVC.ecpint_archives(), ("ecpint.lib", "Faddeeva.lib"));
    assert_eq!(WIN_MSVC.cmake_generator(), Some("Ninja"));
}

#[test]
fn msvc_unistd_stub_supplies_what_basis_h_uses() {
    assert_eq!(WIN_MSVC.compat_headers(), [("unistd.h", MSVC_UNISTD_STUB)]);
    // libint2 2.7.2 basis.h calls ::stat and S_ISDIR.
    assert!(MSVC_UNISTD_STUB.contains("#include <sys/stat.h>"));
    assert!(MSVC_UNISTD_STUB.contains("#define S_ISDIR"));
    assert!(MSVC_UNISTD_STUB.starts_with("#pragma once"));
}

/// The static libint2 archive each target links (the Windows wheel's
/// export is `int2.lib`, which build.rs once linked as a dylib and reported
/// as shared), and the MSVC import-library ambiguity.
#[test]
fn libint_static_archive_per_target() {
    assert_eq!(LINUX.libint_static_archive(), "libint2.a");
    assert_eq!(MACOS.libint_static_archive(), "libint2.a");
    assert_eq!(WIN_GNU.libint_static_archive(), "libint2.a");
    assert_eq!(WIN_MSVC.libint_static_archive(), "int2.lib");
    // An archive alone is static everywhere.
    for t in [LINUX, MACOS, WIN_GNU, WIN_MSVC] {
        assert!(t.libint_is_static(true, false), "{t:?}");
        assert!(!t.libint_is_static(false, false), "{t:?}");
    }
    // MSVC: int2.lib next to bin/int2.dll is an import library.
    assert!(!WIN_MSVC.libint_is_static(true, true));
    // Elsewhere a stray DLL changes nothing (libint2.a is never an import lib).
    assert!(LINUX.libint_is_static(true, true));
}
