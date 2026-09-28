#!/usr/bin/env bash
# Generate the libint2 export tarball the PyPI wheel builds against
# (.github/workflows/wheels.yml, LIBINT_URL / LIBINT_SHA256).
#
# Why a custom export: the wheel needs second derivatives (analytic RHF
# Hessian) and must stay under PyPI's 100 MB per-file limit. The stock mpqc4
# export has first derivatives only; conda-forge's full build is ~97 MB zipped
# by itself. The conda package links conda-forge's full libint (conda/).
#
# Contents, per integral class (max angular momentum per derivative order):
#   4-centre ERI     deriv 0,1,2 : l <= 6,6,3  (second derivatives up to f)
#   3-centre ERI     deriv 0,1   : l <= 6,6    (pure solid harmonics)
#   2-centre ERI     deriv 0,1   : l <= 6,6    (pure solid harmonics)
#   one-body         deriv 0,1,2 : l <= 6,4,3  (multipoles to order 2)
# Generic code + no unrolling keep the library small; --with-opt-am=3.
#
# Usage: scripts/generate-libint-small.sh [workdir]
# Takes ~20 min for code generation on 6 cores. Produces
# <workdir>/libint-2.7.2/gen-small/libint-2.7.2.tgz; upload it as the release
# asset named in wheels.yml and update LIBINT_SHA256 to its sha256sum.
set -euo pipefail

WORK=${1:-$PWD/libint-gen}
TAG=v2.7.2
mkdir -p "$WORK"
cd "$WORK"
if [ ! -d libint-2.7.2 ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/evaleev/libint.git libint-2.7.2
fi
cd libint-2.7.2
[ -x configure ] || ./autogen.sh
mkdir -p gen-small
cd gen-small
../configure --enable-eri=2 --enable-eri3=1 --enable-eri2=1 --enable-1body=2 \
  --with-max-am=6,4,3 --with-eri-max-am=6,6,3 --with-eri3-max-am=6,6 --with-eri2-max-am=6,6 \
  --with-opt-am=3 --enable-generic-code --disable-unrolling --with-multipole-max-order=2 \
  --enable-eri3-pure-sh --enable-eri2-pure-sh --disable-1body-property-derivs \
  CXX=g++ 'CXXFLAGS=-O1 -std=c++11'
make -j"$(nproc)" export
sha256sum libint-2.7.2.tgz
