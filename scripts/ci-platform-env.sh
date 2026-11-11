# shellcheck shell=bash
# Sourced (not executed) by every build/test step of
# .github/workflows/platform-tests.yml on macOS and Windows runners. Sets the
# toolchain and native-dependency environment that cargo, cc-rs, CMake and
# maturin need there. Linux CI never sources it.
#
# Inputs (environment):
#   DEPS_PREFIX   prefix holding OpenBLAS (+ lib/pkgconfig/openblas.pc) and
#                 Eigen (include/eigen3): the merged native-*-<platform>
#                 artifacts of native-deps-macwin.yml (scripts/wheels/README.md).
#   LIBS_PREFIX   prefix holding libint2 (include/libint2, lib/) and libxc.
#                 Defaults to DEPS_PREFIX.
#
# Why each setting exists (all measured on the 2026-11-09 wheels spike,
# wiki/notes/wheels-mac-win-2026-11-09/README.md):
#   - Windows: clang-cl, not cl.exe. cl.exe rejects the vendored libecpint's
#     C99 variable-length arrays (C2131); clang-cl is MSVC-ABI and accepts
#     them. CC/CXX reach both cc-rs (the shims) and CMake (libecpint).
#   - Windows: Git-for-Windows' /usr/bin/link (coreutils) shadows MSVC's
#     link.exe in a login bash, so the MSVC bin dir goes first on PATH and the
#     rustc linker is pinned.
#   - macOS: no OPENBLAS_CORETYPE (arm64 has one ISA; Haswell would be wrong).

: "${DEPS_PREFIX:?DEPS_PREFIX must name the OpenBLAS/Eigen prefix}"
LIBS_PREFIX="${LIBS_PREFIX:-$DEPS_PREFIX}"

case "${RUNNER_OS:-}" in
  Windows)
    DEPS_PREFIX="$(cygpath -m "$DEPS_PREFIX")"
    LIBS_PREFIX="$(cygpath -m "$LIBS_PREFIX")"
    MSVC_BIN="$(dirname "$(command -v cl)")"
    export PATH="$MSVC_BIN:/c/Program Files/LLVM/bin:$(cygpath -u "$LIBS_PREFIX")/bin:$(cygpath -u "$DEPS_PREFIX")/bin:$PATH"
    export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER="$MSVC_BIN/link.exe"
    export CC=clang-cl CXX=clang-cl
    ;;
  macOS)
    # The native deps (native-deps-macwin.yml) are built for macOS 11.0.
    export MACOSX_DEPLOYMENT_TARGET=11.0
    # Test binaries load the prefix's dylibs through @rpath; /bin/bash strips
    # DYLD_* (SIP), so this is set here, inside the shell, not in `env:`.
    export DYLD_FALLBACK_LIBRARY_PATH="$LIBS_PREFIX/lib:$DEPS_PREFIX/lib"
    ;;
  *)
    echo "ci-platform-env.sh: unexpected RUNNER_OS '${RUNNER_OS:-}'" >&2
    return 1
    ;;
esac

export LIBINT2_PREFIX="$LIBS_PREFIX"
export PKG_CONFIG_PATH="$DEPS_PREFIX/lib/pkgconfig"
export RUSTFLAGS="-L native=$LIBS_PREFIX/lib -L native=$DEPS_PREFIX/lib"
# Every non-Linux build must see a known boys.h (shim/libint2_overrides/), as
# Linux CI does; an unknown header is a build error, not a warning.
export FERRIC_REQUIRE_PATCHED_BOYS=1
