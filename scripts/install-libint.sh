#!/usr/bin/env bash
# Install the libint2 build ferric links against: conda-forge libint 2.13.1
# (linux-64), a prebuilt shared library generated with second-derivative
# integrals (LIBINT2_MAX_DERIV_ORDER 2).
#
#   scripts/install-libint.sh [PREFIX] [--keep-soname] [--licenses-dir DIR]
#                                             # default PREFIX: ~/.local/libint2-2.13.1
#   export LIBINT2_PREFIX=<PREFIX>            # what crates/ferric-integrals/build.rs reads
#
# Angular-momentum limits of this build (config.h *_MAX_AM_LIST, by
# derivative order 0,1,2):
#   4-centre ERI 7,6,3   one-body 7,6,3   3- and 2-centre ERI 7,7,4
# so analytic second derivatives need basis functions up to f (4-centre) and
# auxiliary functions up to g.
#
# Why this package: upstream evaleev/libint only publishes first-derivative
# tarballs (mpqc4). conda-forge's libint-feedstock builds the Psi4 developers'
# generated source (loriab/libint release v0.1,
# libint-2.13.1-7-6-3-7-7-4_mm10f12ob2_0.tgz), which takes hours to compile;
# the prebuilt package installs in seconds.
#
# The package ships libint2.so only, with SONAME "libint2.so" and no path
# information a downstream binary could use. This script sets the SONAME to
# the library's absolute path, so every executable, test binary and Python
# extension that links it records that path in DT_NEEDED and loads it without
# rpath or LD_LIBRARY_PATH (cargo does not propagate a dependency's
# rustc-link-arg, so an rpath set in ferric-integrals' build.rs would not
# reach ferric-cli, ferric-python or other crates' tests).
#
# --keep-soname leaves the SONAME as the plain "libint2.so". The wheels
# workflow's large-libint wheel needs that: auditwheel finds the library
# through LD_LIBRARY_PATH and copies it into the wheel, and an absolute host
# path in the extension's DT_NEEDED would be wrong on every other machine.
# A binary linked against a --keep-soname install needs LD_LIBRARY_PATH (or
# an rpath) to load it. Without the flag the behaviour is unchanged.
#
# --licenses-dir DIR also copies the package's licence texts (info/licenses/
# in the package payload) to DIR/libint2-LICENSE.txt and
# DIR/libint2-COPYING.LESSER.txt (libint2 is LGPL-3.0-only). The wheels
# workflow attaches them to the GitHub release that carries the wheel which
# bundles this library.
#
# Needs: curl, python3, zstd, patchelf (patchelf only to print the SONAME
# under --keep-soname).
set -euo pipefail

KEEP_SONAME=0
PREFIX=""
LICENSES_DIR=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --keep-soname) KEEP_SONAME=1 ;;
    --licenses-dir)
      [ "$#" -ge 2 ] || { echo "install-libint.sh: --licenses-dir needs a directory" >&2; exit 2; }
      LICENSES_DIR="$2"; shift ;;
    -*) echo "install-libint.sh: unknown option $1" >&2; exit 2 ;;
    *)
      [ -z "$PREFIX" ] || { echo "install-libint.sh: more than one PREFIX given" >&2; exit 2; }
      PREFIX="$1"
      ;;
  esac
  shift
done
PREFIX="${PREFIX:-$HOME/.local/libint2-2.13.1}"
URL="https://conda.anaconda.org/conda-forge/linux-64/libint-2.13.1-h83f5b4b_0.conda"
SHA256="d5219aae3bdd9629c6cff471f129f3aebbaddb058d0d19fbbab12c28924a0535"  # pragma: allowlist secret

for tool in curl python3 zstd patchelf; do
  command -v "$tool" >/dev/null || { echo "install-libint.sh: missing $tool" >&2; exit 1; }
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

curl -fsSL --retry 3 -o "$work/libint.conda" "$URL"
echo "$SHA256  $work/libint.conda" | sha256sum -c --quiet -

# A .conda file is a zip holding pkg-*.tar.zst (the payload) and info-*.tar.zst.
python3 - "$work" <<'EOF'
import sys, zipfile
work = sys.argv[1]
with zipfile.ZipFile(f"{work}/libint.conda") as z:
    names = [n for n in z.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst")]
    assert len(names) == 1, names
    z.extract(names[0], work)
    open(f"{work}/payload", "w").write(names[0])
EOF
mkdir -p "$PREFIX"
zstd -dc "$work/$(cat "$work/payload")" | tar -x -C "$PREFIX"

# The licence texts ship inside the payload, at info/licenses/ (conda's
# location for a package's license_file entries), not in info-*.tar.zst.
if [ -n "$LICENSES_DIR" ]; then
  mkdir -p "$LICENSES_DIR"
  for f in LICENSE COPYING.LESSER; do
    [ -s "$PREFIX/info/licenses/$f" ] \
      || { echo "install-libint.sh: info/licenses/$f missing from the package" >&2; exit 1; }
    cp "$PREFIX/info/licenses/$f" "$LICENSES_DIR/libint2-$f.txt"
  done
  echo "libint2 licence texts copied to $LICENSES_DIR"
fi

lib="$(cd "$PREFIX/lib" && pwd)/libint2.so"
[ -f "$lib" ] || { echo "install-libint.sh: $lib missing after extraction" >&2; exit 1; }
if [ "$KEEP_SONAME" = 1 ]; then
  [ "$(patchelf --print-soname "$lib")" = "libint2.so" ] \
    || { echo "install-libint.sh: expected SONAME libint2.so in the package, got $(patchelf --print-soname "$lib")" >&2; exit 1; }
else
  patchelf --set-soname "$lib" "$lib"
fi

cfg="$PREFIX/include/libint2/config.h"
params="$PREFIX/include/libint2/libint2_params.h"
grep -q '# define LIBINT2_MAX_DERIV_ORDER 2' "$params" \
  || { echo "install-libint.sh: $params does not declare deriv order 2" >&2; exit 1; }
echo "libint2 2.13.1 installed at $PREFIX (SONAME $(patchelf --print-soname "$lib"))"
grep -E 'MAX_AM_LIST' "$cfg"
