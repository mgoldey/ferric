#!/usr/bin/env bash
# Install the cuEST 0.3.0 C library + headers into a prefix, without conda.
#
#   scripts/cuest/fetch_cuest.sh [cuda12|cuda13] [PREFIX]
#
# Defaults: cuda12, PREFIX=~/.local/cuest-0.3.0. Then build with
#   CUEST_ROOT=PREFIX cargo test -p ferric-cuest --features cuest ...
#
# Downloads the conda-forge packages libcuest and libcuest-dev 0.3.0.1
# (sha256-pinned below) and unpacks them. libcuest.so needs the matching
# CUDA toolkit's cudart, cublas, cublasLt, cusolver, cusparse, nvrtc and
# nvJitLink on the loader path (a system CUDA 12.x/13.x install, or use
# conda instead: `conda install -c conda-forge libcuest-dev cuda-version=12
# cuda-cudart-dev` and CUEST_ROOT=$CONDA_PREFIX).
#
# cuEST is NVIDIA proprietary software under its own license; this script
# only fetches it for local use, nothing is redistributed by ferric.
set -euo pipefail

flavor="${1:-cuda12}"
prefix="${2:-$HOME/.local/cuest-0.3.0}"
base="https://conda.anaconda.org/conda-forge/linux-64"
case "$flavor" in
  cuda12)
    build=hb7e823c_0
    sha_lib=b309cd54fdf2bc6e296b49f3a0eb6198333d908f6f99f4b9e47f427eab22f932
    sha_dev=40e34ec36f79efe2f610675e2d741dd4aeee7c567037f6fa51f1a7ea00b992f8 ;;
  cuda13)
    build=h345135a_0
    sha_lib=b766bf75603dc25162907b3ed8d29d72ebaebded067ee1a6afd8eb174628d1be
    sha_dev=0159f8dcf95b2207946a9f5bfd2892b2b9d8200685032e1ae8c29d1ad2eb4d85 ;;
  *) echo "usage: $0 [cuda12|cuda13] [PREFIX]" >&2; exit 2 ;;
esac

work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
for pkg in libcuest libcuest-dev; do
  f="$pkg-0.3.0.1-$build.conda"
  curl -fsSL -o "$work/$f" "$base/$f"
  want=$([[ $pkg == libcuest ]] && echo "$sha_lib" || echo "$sha_dev")
  echo "$want  $work/$f" | sha256sum -c --quiet -
  mkdir -p "$work/$pkg"
  (cd "$work/$pkg" && unzip -q "../$f" && tar --zstd -xf pkg-*.tar.zst)
done

mkdir -p "$prefix/include" "$prefix/lib"
cp -r "$work/libcuest-dev/include/." "$prefix/include/"
cp "$work/libcuest/lib/libcuest.so.0.3.0" "$prefix/lib/"
ln -sf libcuest.so.0.3.0 "$prefix/lib/libcuest.so.0"
ln -sf libcuest.so.0.3.0 "$prefix/lib/libcuest.so"
echo "cuEST 0.3.0 ($flavor) installed in $prefix"
if ldd "$prefix/lib/libcuest.so" | grep -q 'not found'; then
  echo "WARNING: missing CUDA libraries for libcuest.so:" >&2
  ldd "$prefix/lib/libcuest.so" | grep 'not found' >&2
fi
