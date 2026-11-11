#!/usr/bin/env bash
# Build OpenBLAS and libxc from source, and stage Eigen's headers, for the
# macOS arm64 and Windows x86_64 wheels (.github/workflows/native-deps-macwin.yml).
#
#   build-blas-xc.sh <install-prefix>
#
# Environment (set by the workflow): OPENBLAS_VERSION, LIBXC_VERSION,
# EIGEN_PREFIX (where Eigen 3.4 is installed; its include/eigen3 is copied
# into the prefix, where ferric-integrals/build.rs looks for it). macOS:
# MACOSX_DEPLOYMENT_TARGET. Windows: an MSVC developer environment with
# CC=CXX=clang-cl.
#
# OpenBLAS, as on Linux (wheels.yml build-native-deps): ONE fixed kernel
# TARGET, no DYNAMIC_ARCH (the runtime dispatch blamed for the Zen 4
# garbage-numerics incident), pthreads, not OpenMP. Unlike Linux there is no
# Fortran compiler: LAPACK is OpenBLAS's C translation of reference LAPACK
# (C_LAPACK), so the wheel needs no libgfortran/libquadmath. NUM_THREADS=128
# fixes the maximum thread count (conda-forge's value; left unset it is the
# build runner's core count, 3 or 4, which capped the 0.1.0rc11 Linux wheel
# at 4 threads). The smoke job checks it (scripts/check_wheel_openblas_threads.py).
#   macOS arm64:     TARGET=VORTEX (Apple M1 and later)
#   Windows x86_64:  TARGET=HASWELL (AVX2), the Linux wheel's target
# OPENBLAS_VERSION differs per platform (native-deps-macwin.yml explains why).
#
# This file is part of the native-deps cache key: editing it rebuilds this part.
set -euo pipefail

PREFIX="$1"
: "${OPENBLAS_VERSION:?}" "${LIBXC_VERSION:?}" "${EIGEN_PREFIX:?}"
TMPBASE="${RUNNER_TEMP:-/tmp}"
if command -v cygpath >/dev/null; then TMPBASE="$(cygpath -u "$TMPBASE")"; fi
WORK="$TMPBASE/blas-xc-src"
rm -rf "$WORK" && mkdir -p "$WORK" "$PREFIX/include" "$PREFIX/licenses"
cd "$WORK"
OS="$(uname -s)"
case "$OS" in
  Darwin) NPROC=$(sysctl -n hw.ncpu) ;;
  *) NPROC="${NUMBER_OF_PROCESSORS:-4}" ;;
esac

# --- Eigen 3.4 headers (compiled into the shim; MPL-2.0) ---
cp -R "$EIGEN_PREFIX/include/eigen3" "$PREFIX/include/"
test -f "$PREFIX/include/eigen3/Eigen/Core"
curl -sSfL -o "$PREFIX/licenses/eigen-COPYING.MPL2.txt" \
  "https://gitlab.com/libeigen/eigen/-/raw/3.4.0/COPYING.MPL2"

# --- libxc: shared on macOS (delocate bundles it), static on Windows ---
curl -sSfL -o libxc.tgz "https://gitlab.com/libxc/libxc/-/archive/${LIBXC_VERSION}/libxc-${LIBXC_VERSION}.tar.gz"
# Two doc symlinks (ChangeLog, README) that Git-for-Windows tar cannot create.
tar --exclude="libxc-${LIBXC_VERSION}/ChangeLog" --exclude="libxc-${LIBXC_VERSION}/README" -xzf libxc.tgz
if [ "$OS" = Darwin ]; then XC_SHARED=ON; else XC_SHARED=OFF; fi
cmake -S "libxc-${LIBXC_VERSION}" -B build-xc -G Ninja \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DDISABLE_KXC=OFF \
  -DBUILD_SHARED_LIBS=$XC_SHARED
cmake --build build-xc > build-xc.log 2>&1 || { tail -60 build-xc.log; exit 1; }
cmake --install build-xc > /dev/null
cp "libxc-${LIBXC_VERSION}/COPYING" "$PREFIX/licenses/libxc-COPYING.txt"

# --- OpenBLAS ---
curl -sSfL -o openblas.tgz "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz"
tar xzf openblas.tgz
cp "OpenBLAS-${OPENBLAS_VERSION}/LICENSE" "$PREFIX/licenses/OpenBLAS-LICENSE.txt"
if [ "$OS" = Darwin ]; then
  (
    cd "OpenBLAS-${OPENBLAS_VERSION}"
    # `libs netlib shared`: `make`'s default also builds OpenBLAS's own test
    # programs, whose link line (-lto_library) Apple's ld rejects.
    make -j"$NPROC" libs netlib shared TARGET=VORTEX DYNAMIC_ARCH=0 USE_THREAD=1 USE_OPENMP=0 \
      NUM_THREADS=128 NOFORTRAN=1 C_LAPACK=1 NO_STATIC=1 > ../build-ob.log 2>&1 \
      || { tail -60 ../build-ob.log; exit 1; }
    make PREFIX="$PREFIX" NO_STATIC=1 install > /dev/null
  )
  # Relocatable install name: the prefix moves between jobs, and delocate
  # resolves @rpath through DYLD_LIBRARY_PATH.
  for lib in "$PREFIX"/lib/libopenblas*.dylib; do
    [ -L "$lib" ] || install_name_tool -id "@rpath/$(basename "$lib")" "$lib"
  done
  otool -D "$PREFIX"/lib/libopenblas.0.dylib "$PREFIX"/lib/libxc.*.dylib
else
  # OpenBLAS 0.3.27's ctest/CMakeLists.txt (configured unconditionally) has
  # `if (USE_OPENMP AND (${CMAKE_Fortran_COMPILER_ID} STREQUAL GNU) ...)`:
  # with no Fortran compiler the unquoted variable is empty and the if() does
  # not parse ("Unknown arguments specified"). Quote it.
  # (Later releases quote it already; patch only when the old form is there.)
  ct="OpenBLAS-${OPENBLAS_VERSION}/ctest/CMakeLists.txt"
  if grep -q '(${CMAKE_Fortran_COMPILER_ID} STREQUAL GNU)' "$ct"; then
    sed -i 's/(${CMAKE_Fortran_COMPILER_ID} STREQUAL GNU)/("${CMAKE_Fortran_COMPILER_ID}" STREQUAL GNU)/' "$ct"
  fi
  cmake -S "OpenBLAS-${OPENBLAS_VERSION}" -B build-ob -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
    -DTARGET=HASWELL -DDYNAMIC_ARCH=OFF -DUSE_THREAD=1 -DUSE_OPENMP=0 \
    -DNUM_THREADS=128 -DNOFORTRAN=1 -DC_LAPACK=ON -DBUILD_WITHOUT_LAPACK=OFF \
    -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DBUILD_TESTING=OFF
  cmake --build build-ob > build-ob.log 2>&1 || { tail -80 build-ob.log; exit 1; }
  cmake --install build-ob > /dev/null
fi

# A relocatable openblas.pc: openblas-src (the `system` feature) finds
# OpenBLAS through pkg-config, and the prefix is downloaded to a different
# path in every job that uses it.
pc="$PREFIX/lib/pkgconfig/openblas.pc"
test -f "$pc" || { echo "::error::no $pc"; find "$PREFIX" -name '*.pc'; exit 1; }
sed -e 's|^libdir=.*|libdir=${pcfiledir}/..|' \
    -e 's|^includedir=.*|includedir=${pcfiledir}/../../include|' \
    -e 's|^prefix=.*|prefix=${pcfiledir}/../..|' "$pc" > "$pc.new"
mv "$pc.new" "$pc"
cat "$pc"
find "$PREFIX" -name '*.pc' ! -name openblas.pc -print -delete
rm -rf "$PREFIX/share"

# Every BLAS/LAPACK entry point ferric calls (grep of the workspace, spike
# notes section 1), exported under the Fortran ABI name.
if [ "$OS" = Darwin ]; then
  exports="$(nm -gU "$PREFIX/lib/libopenblas.0.dylib" | awk '{print $3}' | sed 's/^_//')"
else
  dll="$(ls "$PREFIX"/bin/*openblas*.dll | head -1)"
  exports="$(MSYS_NO_PATHCONV=1 dumpbin /exports "$(cygpath -w "$dll")" | awk 'NF >= 4 {print $4}')"
fi
missing=0
for sym in dsyevd_ dsyev_ dgetrf_ dgetri_ dpotrf_ dpotrs_ dsyrk_ dgemm_ sgemm_ \
           cblas_dgemm cblas_sgemm openblas_set_num_threads openblas_get_num_threads; do
  # A here-string, not `printf | grep -q`: under pipefail grep -q exiting
  # early SIGPIPEs printf, which read as a missing symbol.
  grep -qx "$sym" <<< "$exports" || { echo "::error::OpenBLAS does not export $sym"; missing=1; }
done
[ "$missing" = 0 ] && echo "OpenBLAS exports every symbol ferric calls"
[ "$missing" = 0 ]
ls -l "$PREFIX/lib" "$PREFIX/bin" 2>/dev/null || true
