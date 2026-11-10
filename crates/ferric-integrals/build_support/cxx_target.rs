// Target-conditional choices of ferric-integrals' C++ build (the two shims,
// the C++ runtime, the vendored libecpint).
//
// Pure: no I/O, no environment reads. `build.rs` `include!`s this file and
// builds a `CxxTarget` from cargo's `CARGO_CFG_TARGET_OS` /
// `CARGO_CFG_TARGET_ENV`; `tests/cxx_target_flags.rs` includes the same file
// to check every target's choices on any host.
//
// Every non-MSVC target gets exactly the flags build.rs passed before this
// file existed (the Linux build line is unchanged); MSVC-ABI compilers (cl.exe
// or clang-cl, `target_env = "msvc"`) get their own spelling.

/// The compile target, as cargo names it to build scripts.
#[derive(Clone, Copy, Debug)]
pub struct CxxTarget<'a> {
    /// `CARGO_CFG_TARGET_OS`: "linux", "macos", "windows", ...
    pub os: &'a str,
    /// `CARGO_CFG_TARGET_ENV`: "gnu", "msvc", "" (macOS), ...
    pub env: &'a str,
}

/// `<unistd.h>` stand-in for MSVC targets (see [`CxxTarget::compat_headers`]).
pub const MSVC_UNISTD_STUB: &str = "\
#pragma once
/* ferric (crates/ferric-integrals/build.rs): MSVC has no <unistd.h>. libint2
 * 2.7.2's libint2/basis.h includes it unconditionally but uses only stat() and
 * S_ISDIR, which upstream (>= 2.8) takes from <io.h> + <sys/stat.h> under
 * _MSC_VER. This stub supplies the same. */
#include <io.h>
#include <sys/stat.h>
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif
";

impl CxxTarget<'_> {
    /// cl.exe / clang-cl command-line conventions and the MSVC runtime.
    pub fn msvc_abi(&self) -> bool {
        self.env == "msvc"
    }

    /// Flags for `shim/shim.cc` (includes the libint2 C++ API: C++17).
    ///
    /// MSVC: `/std:c++17` because MSVC defaults to C++14, and `/EHsc` because
    /// without it exceptions are off and the shim's FFI contract (catch every
    /// libint2 throw before the C ABI) cannot even compile.
    pub fn shim_flags(&self) -> &'static [&'static str] {
        if self.msvc_abi() {
            &["/std:c++17", "/O2", "/EHsc"]
        } else {
            &[
                "-std=c++17",
                "-O2",
                "-Wno-deprecated-declarations",
                "-Wno-unused-parameter",
            ]
        }
    }

    /// Flags for `shim/ecp_shim.cc` (libecpint, C++11). MSVC has no C++11
    /// mode; `/std:c++14` is its oldest.
    pub fn ecp_shim_flags(&self) -> &'static [&'static str] {
        if self.msvc_abi() {
            &["/std:c++14", "/O2", "/EHsc"]
        } else {
            &[
                "-std=c++11",
                "-O2",
                "-Wno-deprecated-declarations",
                "-Wno-unused-parameter",
            ]
        }
    }

    /// Preprocessor symbols for both shims. MSVC's <cmath> defines `M_PI`
    /// only under `_USE_MATH_DEFINES`.
    pub fn defines(&self) -> &'static [&'static str] {
        if self.msvc_abi() {
            &["_USE_MATH_DEFINES"]
        } else {
            &[]
        }
    }

    /// System headers to synthesise into an include directory searched AFTER
    /// every real one, as (file name, contents). Only MSVC needs any.
    pub fn compat_headers(&self) -> &'static [(&'static str, &'static str)] {
        if self.msvc_abi() {
            &[("unistd.h", MSVC_UNISTD_STUB)]
        } else {
            &[]
        }
    }

    /// The C++ standard library to link by name, if any: libstdc++ by default,
    /// libc++ on Apple targets (the macOS SDK has no libstdc++), none under
    /// MSVC (the compiler links its own runtime).
    pub fn cxx_runtime(&self) -> Option<&'static str> {
        if self.msvc_abi() {
            None
        } else if matches!(self.os, "macos" | "ios") {
            Some("c++")
        } else {
            Some("stdc++")
        }
    }

    /// File names of the libecpint and Faddeeva static archives CMake builds.
    pub fn ecpint_archives(&self) -> (&'static str, &'static str) {
        if self.msvc_abi() {
            ("ecpint.lib", "Faddeeva.lib")
        } else {
            ("libecpint.a", "libFaddeeva.a")
        }
    }

    /// CMake generator for libecpint, when the default will not do. Under
    /// MSVC the default (Visual Studio) generator is multi-config and puts the
    /// archives under `Release/`; single-config Ninja keeps them where the
    /// other targets have them (and needs `ninja` on PATH).
    pub fn cmake_generator(&self) -> Option<&'static str> {
        if self.msvc_abi() {
            Some("Ninja")
        } else {
            None
        }
    }
}
