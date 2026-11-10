#!/usr/bin/env bash
# Build ferric's small libint2 export (static) for the macOS arm64 and Windows
# x86_64 wheels (.github/workflows/native-deps-macwin.yml). The Linux wheels
# build the same export inside the manylinux container (wheels.yml,
# build-native-deps); this is the non-Linux twin.
#
#   build-libint2.sh <install-prefix>
#
# Environment (set by the workflow): LIBINT_VERSION, LIBINT_URL, LIBINT_SHA256,
# EIGEN_PREFIX (a prefix holding Eigen 3.4's CMake config package, which
# libint2's CMake needs before it builds the C++ API). macOS: LLVM_AR (Apple's
# ar cannot read the @response file CMake writes for 16,838 objects).
# Windows: run in an MSVC developer environment with CC=CXX=clang-cl (cl.exe
# is not used anywhere: ferric's vendored libecpint needs clang-cl, and one
# compiler for every C++ object keeps the toolchain to one).
#
# This file is part of the native-deps cache key: editing it rebuilds libint2
# (~77 min on macos-15, ~91 min on windows-2025).
set -euo pipefail

PREFIX="$1"
: "${LIBINT_VERSION:?}" "${LIBINT_URL:?}" "${LIBINT_SHA256:?}" "${EIGEN_PREFIX:?}"
TMPBASE="${RUNNER_TEMP:-/tmp}"
if command -v cygpath >/dev/null; then TMPBASE="$(cygpath -u "$TMPBASE")"; fi
WORK="$TMPBASE/libint2-src"
rm -rf "$WORK" && mkdir -p "$WORK" "$PREFIX"
cd "$WORK"

curl -sSfL -o libint-small.tgz "$LIBINT_URL"
if command -v sha256sum >/dev/null; then SHA=(sha256sum); else SHA=(shasum -a 256); fi
echo "$LIBINT_SHA256  libint-small.tgz" | "${SHA[@]}" -c -
tar xzf libint-small.tgz

extra=()
case "$(uname -s)" in
  Darwin)
    : "${LLVM_AR:?macOS needs LLVM_AR}"
    # At the FIRST configure: a re-configure with -DCMAKE_AR did not change
    # the archive rule (spike run 42722710215).
    extra+=(-DCMAKE_AR="$LLVM_AR")
    ;;
esac

cmake -S "libint-${LIBINT_VERSION}" -B build -G Ninja \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_PREFIX_PATH="$EIGEN_PREFIX" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_BUILD_TYPE=Release "${extra[@]}"
grep -E '^CMAKE_(AR|C_COMPILER|CXX_COMPILER):' build/CMakeCache.txt
t0=$(date +%s)
cmake --build build > "$WORK/build.log" 2>&1 || {
  echo "::error::libint2 build failed; tail of the log follows"
  tail -60 "$WORK/build.log"
  exit 1
}
echo "libint2 compile: $(($(date +%s) - t0)) s"
cmake --install build > /dev/null

# share/ holds libint2's own basis library, which nothing reads; several of
# its file names contain '*', which upload-artifact refuses.
rm -rf "$PREFIX/share"
mkdir -p "$PREFIX/licenses"
for f in LICENSE COPYING COPYING.LESSER; do
  cp "libint-${LIBINT_VERSION}/$f" "$PREFIX/licenses/libint2-$f.txt"
done

# The static archive ferric-integrals/build.rs links (and nothing shared).
ls -l "$PREFIX/lib"
if [ -f "$PREFIX/lib/libint2.a" ] || [ -f "$PREFIX/lib/int2.lib" ]; then :; else
  echo "::error::no static libint2 archive under $PREFIX/lib"; exit 1
fi
grep -E 'LIBINT_MAX_AM|MAX_DERIV' "$PREFIX/include/libint2/config.h" | head -20
