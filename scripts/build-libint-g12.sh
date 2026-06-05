#!/usr/bin/env bash
# Rebuild libint2 2.7.2 WITH the G12 (MP2-F12) integral class enabled.
#
# WHY: the bundled ~/.local libint is the mpqc4 *export* tarball, which was
# generated with G12 stripped (config.h: `#undef INCLUDE_G12`; src/ has 7560
# ERI files and ZERO g12/cgtg files). F12 needs the geminal kernels
# (cgtg, cgtg_x_coulomb, delcgtg2) which only exist if the libint GENERATOR
# is run with --enable-g12. This script bootstraps the generator from the
# upstream 2.7.2 source (NOT the -mpqc4 export), runs codegen with G12+T1G12,
# then builds and installs the resulting library.
#
# TWO-STAGE BUILD:
#   Stage 1 (this dir): configure + `make export` on the generator
#                       -> emits a NEW tarball libint-2.7.2.tgz WITH g12 source
#   Stage 2:            untar that, cmake/configure + make + install -> ~/.local
#
# COST (g12-max-am=4, t1g12 on): codegen ~hours, library compile ~hours,
# generated C++ a few GB. Plan for an overnight run. Disk: keep ~15-20 GB free.
#
# Flag names verified against libint v2.7.2 configure.ac and build_libint.cc:
#   --enable-g12=N           : N = max DERIVATIVE order of F12 energies. 0 = energies only.
#   --with-g12-max-am=4      : G12 integrals up to L=4 (g) -> covers cc-pVDZ AND
#                              def2-TZVP/aug-cc-pVTZ as the ORBITAL basis.
#                              (RI/JK aux at L=6 use the Coulomb engine, not G12.)
#   --enable-t1g12-support   : emit delcgtg2 / [Ti,G12] (the kinetic-cusp B term).
#                              Default ON with g12; stated explicitly here.
#   --with-max-am=6,4        : keep Coulomb-class AM identical to the current export
#                              so RI aux (L=6) and ERI derivatives are unchanged.
set -euo pipefail

VER=2.7.2
# IMPORTANT: the libint GENERATOR (src/bin/libint/build_libint.cc, autogen.sh,
# configure.ac) lives ONLY in the git repo. Every release .tgz -- including the
# plain v2.7.2 tarball AND the -mpqc4 one -- is a POST-codegen CMake *export*
# with a frozen feature set (G12 off). So we must git-clone the tag, not fetch
# a tarball. Verified: git repo has build_libint.cc + autogen.sh + configure.ac
# (all HTTP 200) and NO top-level CMakeLists.txt (404) -- the inverse of the
# tarball, which has CMakeLists.txt + pregenerated src/*.cc and no generator.
GIT_URL="https://github.com/evaleev/libint.git"
WORK="${HOME}/qc/libint-g12-build"
PREFIX="${HOME}/.local"            # same prefix ferric's build.rs already searches
JOBS="$(nproc)"

mkdir -p "${WORK}"
cd "${WORK}"

# --- clone the GENERATOR source at the v2.7.2 tag (autotools tree) ---
if [[ ! -d "libint-git/.git" ]]; then
  echo ">> cloning upstream libint generator at v${VER} (git, not a release tarball)"
  rm -rf libint-git
  git clone --depth 1 --branch "v${VER}" "${GIT_URL}" libint-git
fi
cd libint-git
# sanity: confirm we actually have the generator, fail loudly if not
test -f src/bin/libint/build_libint.cc || { echo "FATAL: no build_libint.cc -- not the generator tree"; exit 1; }
test -f autogen.sh || { echo "FATAL: no autogen.sh -- not the generator tree"; exit 1; }

# --- Stage 1: bootstrap + configure the generator with G12 on ---
./autogen.sh            # generates ./configure from configure.ac (needs autoconf/automake/libtool)

mkdir -p build && cd build
../configure \
  --enable-eri=1 \
  --enable-eri3=1 --enable-eri3-pure-sh \
  --enable-eri2=1 --enable-eri2-pure-sh \
  --enable-1body=1 \
  --enable-g12=0 \
  --with-g12-max-am=4 \
  --with-g12-opt-am=3 \
  --enable-t1g12-support \
  --with-max-am=6,4 \
  --with-opt-am=3 \
  --enable-generic-code \
  --enable-fma \
  --with-multipole-max-order=10 \
  CXX=g++ 'CXXFLAGS=-std=c++17 -O3 -march=native'

# Codegen + emit a NEW export tarball that CONTAINS the g12 source files.
echo ">> Stage 1: running G12 codegen (this is the long step) ..."
make export -j"${JOBS}"

# The export tarball lands in build/ as libint-<ver>.tgz (g12-enabled this time).
EXPORTED="$(ls -t libint-*.tgz | head -1)"
EXPORT_DIR="${WORK}/libint-git/build"
echo ">> Stage 1 done. G12-enabled export: ${EXPORT_DIR}/${EXPORTED}"

# --- Stage 2: build + install the generated library ---
cd "${WORK}"
rm -rf libint-export
mkdir -p libint-export && cd libint-export
tar xf "${EXPORT_DIR}/${EXPORTED}" --strip-components=1

mkdir -p bld && cd bld
cmake .. \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CXX_FLAGS="-std=c++17 -O3 -march=native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBINT2_INSTALL_BASISDIR="${PREFIX}/share/libint"
echo ">> Stage 2: compiling the G12-enabled library ..."
make -j"${JOBS}"
make install

echo
echo ">> DONE. Verify with:"
echo "   grep -E 'INCLUDE_G12|SUPPORT_T1G12' ${PREFIX}/include/libint2/config.h"
echo "   # expect: #define INCLUDE_G12 1   and   #define SUPPORT_T1G12 1"
echo "   ls ${PREFIX}/lib/libint2.a   # rebuilt, now with cgtg/delcgtg2 inside"
