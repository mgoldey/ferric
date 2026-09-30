#!/usr/bin/env bash
# Build and run the libint2 erf/erfc threading reproducer (mwe.cc).
# NOT YET RUN. Needs a quiet machine; takes ~1-2 minutes of CPU.
#
#   LIBINT_PREFIX=~/.local reference/pbc/libint-gmeval-mwe/run.sh [calls]              # 2.7.2 (static libint2.a)
#   LIBINT_PREFIX=~/.local/libint2-2.13.1 reference/pbc/libint-gmeval-mwe/run.sh [calls]  # 2.13.1 (conda libint2.so)
#
# Build stock and patched binaries (the patched one puts ferric's vendored
# boys.h first on the include path; the stock one uses the prefix's own):
#   mwe_stock    -- libint 2.7.2 headers as installed
#   mwe_patched  -- crates/ferric-integrals/shim/libint2_overrides/<version> first
# <version> is read from the prefix's libint2/config.h (LIBINT_VERSION), so the
# override always matches the headers it shadows.
# The two must print the SAME checksum for each operator (the patch does not
# change arithmetic).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PREFIX="${LIBINT_PREFIX:-$HOME/.local}"
CALLS="${1:-200000}"
NT=6
CXX="${CXX:-g++}"
FLAGS=(-std=c++17 -O2 -pthread -I/usr/include/eigen3)
# -lint2 resolves to libint2.a (2.7.2 mpqc4 build) or libint2.so (2.13.1 conda
# build; install-libint.sh sets an absolute SONAME, so no rpath/LD_LIBRARY_PATH).
LIBS=(-L"$PREFIX/lib" -lint2)
VERSION="$(sed -n 's/^#define LIBINT_VERSION "\(.*\)"$/\1/p' "$PREFIX/include/libint2/config.h")"
OVERRIDE="$ROOT/crates/ferric-integrals/shim/libint2_overrides/$VERSION"
[[ -f "$OVERRIDE/libint2/boys.h" ]] || { echo "no boys.h override for libint2 '$VERSION' at $OVERRIDE" >&2; exit 1; }

"$CXX" "${FLAGS[@]}" -I"$PREFIX/include" "$HERE/mwe.cc" "${LIBS[@]}" -o "$HERE/mwe_stock"
"$CXX" "${FLAGS[@]}" -I"$OVERRIDE" -I"$PREFIX/include" \
  "$HERE/mwe.cc" "${LIBS[@]}" -o "$HERE/mwe_patched"

for bin in mwe_stock mwe_patched; do
  echo "=== $bin"
  for op in coulomb erf erfc; do
    "$HERE/$bin" "$op" 1 "$CALLS"
    "$HERE/$bin" "$op" "$NT" "$CALLS"
    echo "--- $op: $NT separate 1-thread processes"
    for _ in $(seq "$NT"); do "$HERE/$bin" "$op" 1 "$CALLS" & done
    wait
  done
done
# Read-out: per-call slowdown = (6-thread mean ns/call) / (1-thread ns/call).
# Defect present => stock erf/erfc slow down several x with threads but ~1x as
# separate processes; coulomb ~1x either way; patched erf/erfc ~1x.
