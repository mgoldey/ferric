#!/usr/bin/env bash
# Build and install the libxc ferric links against (7.0.0, pinned with its
# tarball sha256 in scripts/wheels/native-pins.env) from upstream source.
#
#   scripts/install-libxc.sh [PREFIX] [--keep-soname] [--jobs N]
#                                             # default PREFIX: ~/.local/libxc-<version>
#   export LIBXC_PREFIX=<PREFIX>              # what crates/ferric-dft/build.rs reads
#
# crates/ferric-dft/build.rs also finds the default prefix by itself; a
# version bump updates the path there too.
#
# Why 7.0.0: it is the libxc PySCF 2.13 bundles (pyscf/lib/CMakeLists.txt
# fetches the same tarball), so ferric's PySCF reference JSONs and ferric
# itself evaluate the same functional code. libxc >= 6.1 is needed for the
# r2SCAN hybrids (HYB_MGGA_XC_R2SCAN0 / R2SCANH / R2SCAN50).
#
# The cmake options match PySCF's libxc build (no Fortran, third derivatives
# on, fourth off, Fermi-hole-curvature enforcement left at its default ON)
# and the wheels workflows' (Release, PIC, no test suite).
#
# The install's SONAME is rewritten to the library's absolute path, as
# scripts/install-libint.sh does for libint2: every executable, test binary
# and Python extension that links it records that path in DT_NEEDED and loads
# it without rpath or LD_LIBRARY_PATH. A system libxc (Ubuntu's libxc-dev,
# SONAME libxc.so.9 for 5.x) can then sit next to it without either being
# picked up by mistake. --keep-soname leaves the stock "libxc.so.15" (for a
# build that bundles the library, e.g. a wheel).
#
# Needs: curl, cmake (>= 3.5), a C compiler, patchelf. Build time is about
# 10 minutes on 6 cores (the Maple-generated kernels are large).
set -euo pipefail

# The pin (version + tarball sha256) is scripts/wheels/native-pins.env's, the
# one the macOS/Windows wheels build from, so every platform agrees.
pins="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wheels/native-pins.env"
pin() { sed -n "s/^$1=\([^[:space:]#]*\).*/\1/p" "$pins"; }
VERSION="$(pin LIBXC_VERSION)"
SHA256="$(pin LIBXC_SHA256)"
[ -n "$VERSION" ] && [ -n "$SHA256" ] \
  || { echo "install-libxc.sh: LIBXC_VERSION / LIBXC_SHA256 missing from $pins" >&2; exit 1; }
URL="https://gitlab.com/libxc/libxc/-/archive/${VERSION}/libxc-${VERSION}.tar.gz"

KEEP_SONAME=0
PREFIX=""
JOBS="$(nproc 2>/dev/null || echo 2)"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --keep-soname) KEEP_SONAME=1 ;;
    --jobs)
      [ "$#" -ge 2 ] || { echo "install-libxc.sh: --jobs needs a number" >&2; exit 2; }
      JOBS="$2"; shift ;;
    -*) echo "install-libxc.sh: unknown option $1" >&2; exit 2 ;;
    *)
      [ -z "$PREFIX" ] || { echo "install-libxc.sh: more than one PREFIX given" >&2; exit 2; }
      PREFIX="$1"
      ;;
  esac
  shift
done
PREFIX="${PREFIX:-$HOME/.local/libxc-${VERSION}}"

for tool in curl cmake patchelf; do
  command -v "$tool" >/dev/null || { echo "install-libxc.sh: missing $tool" >&2; exit 1; }
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

curl -fsSL --retry 3 -o "$work/libxc.tar.gz" "$URL"
echo "$SHA256  $work/libxc.tar.gz" | sha256sum -c --quiet -
tar -xzf "$work/libxc.tar.gz" -C "$work"

# CMAKE_POLICY_VERSION_MINIMUM: libxc 7.0.0 declares a cmake_minimum_required
# below 3.5, which CMake 4.x refuses outright (see wheels.yml).
cmake -S "$work/libxc-${VERSION}" -B "$work/build" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
  -DENABLE_FORTRAN=OFF -DDISABLE_KXC=OFF -DDISABLE_LXC=ON \
  > "$work/cmake.log" 2>&1 || { tail -40 "$work/cmake.log"; exit 1; }
cmake --build "$work/build" -j "$JOBS" > "$work/build.log" 2>&1 \
  || { tail -60 "$work/build.log"; exit 1; }
cmake --install "$work/build" > /dev/null

grep -q "#define XC_VERSION \"${VERSION}\"" "$PREFIX/include/xc_version.h" \
  || { echo "install-libxc.sh: $PREFIX/include/xc_version.h is not ${VERSION}" >&2; exit 1; }

# Resolve the versioned file the libxc.so symlink chain ends at, and give it
# its own absolute path as SONAME.
lib="$(readlink -f "$PREFIX/lib/libxc.so")"
[ -f "$lib" ] || { echo "install-libxc.sh: $PREFIX/lib/libxc.so missing after install" >&2; exit 1; }
if [ "$KEEP_SONAME" = 0 ]; then
  patchelf --set-soname "$lib" "$lib"
fi
echo "libxc ${VERSION} installed at $PREFIX (SONAME $(patchelf --print-soname "$lib"))"
