#!/usr/bin/env bash
# Print the actions/cache key of one part of the macOS/Windows native deps:
#
#   native-cache-key.sh <libint2|blas-xc> <macos-arm64|windows-x86_64>
#
# THE single definition of these keys: native-deps-macwin.yml saves under
# them, and anything that reuses the deps (platform-tests.yml) restores with
# them. Run on the target runner, from the repository root, after
# ilammy/msvc-dev-cmd on Windows (the compiler's major version is part of the
# key). Diagnostics go to stderr; stdout is the key alone.
#
# Key = native-macwin-<cache version>-<part>-<platform>-<os>-<compiler major>
#       -<source pins>-script<sha256[:16] of scripts/wheels/build-<part>.sh>
set -euo pipefail
part="$1"
platform="$2"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# eval, not `. <(...)`: macOS's /bin/bash 3.2 sources a process substitution
# as an empty file, and the pins were silently missing there.
pins="$(grep -E '^[A-Z_0-9]+=' "$root/scripts/wheels/native-pins.env" | sed 's/[[:space:]]*#.*$//')"
eval "$pins"
for v in LIBINT_VERSION LIBINT_SHA256 OPENBLAS_VERSION_MACOS OPENBLAS_VERSION_WINDOWS \
         LIBXC_VERSION EIGEN_VERSION NATIVE_MACWIN_CACHE_VERSION; do
  [ -n "${!v:-}" ] || { echo "::error::native-cache-key.sh: $v missing from native-pins.env" >&2; exit 1; }
done

case "$(uname -s)" in
  Darwin)
    cc_ver="$(clang --version | head -1)"
    cc="appleclang$(printf '%s\n' "$cc_ver" | sed -n 's/.*clang version \([0-9]*\)\..*/\1/p')"
    os="macos$(sw_vers -productVersion | cut -d. -f1)-$(uname -m)"
    ob="$OPENBLAS_VERSION_MACOS"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    cc_ver="$(PATH="/c/Program Files/LLVM/bin:$PATH" clang-cl --version | head -1)"
    cc="clangcl$(printf '%s\n' "$cc_ver" | sed -n 's/.*clang version \([0-9]*\)\..*/\1/p')"
    os="$(cmd //c ver | tr -d '\r' | sed -n 's/.*Version \([0-9]*\.[0-9]*\.[0-9]*\).*/win\1/p')"
    ob="$OPENBLAS_VERSION_WINDOWS"
    ;;
  *) echo "native-cache-key.sh: unsupported OS $(uname -s)" >&2; exit 2 ;;
esac
echo "compiler: $cc_ver" >&2
case "$cc" in clangcl|appleclang) echo "::error::could not read the compiler version from: $cc_ver" >&2; exit 1 ;; esac
[ -n "$os" ] || { echo "::error::could not read the OS version" >&2; exit 1; }

case "$part" in
  libint2) src="libint${LIBINT_VERSION}-${LIBINT_SHA256}-eigen${EIGEN_VERSION}" ;;
  blas-xc) src="openblas${ob}-libxc${LIBXC_VERSION}-eigen${EIGEN_VERSION}" ;;
  *) echo "native-cache-key.sh: unknown part $part" >&2; exit 2 ;;
esac
script="$root/scripts/wheels/build-$part.sh"
# Hashed with CRs stripped: the key must not depend on the checkout's
# line-ending setting (a Windows checkout with core.autocrlf=true once gave the
# Windows keys a CRLF hash that an LF checkout could never reproduce).
hash="$(tr -d '\r' < "$script" | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-16)"
echo "native-macwin-${NATIVE_MACWIN_CACHE_VERSION}-${part}-${platform}-${os}-${cc}-${src}-script${hash}"
