#!/usr/bin/env bash
# Regenerate the committed PTX for the gpu kernels. Needs nvcc 12.x
# (13.x cannot emit compute_61). Commit each .cu and its .ptx together.
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=crates/ferric-core/src/gpu/kernels
nvcc --version | tail -1
for name in axpy_f32_to_f64 gemv_f32_f64; do
    src="$dir/$name.cu"
    out="$dir/$name.ptx"
    nvcc -arch=compute_61 -O3 -ptx -o "$out" "$src"
    grep -q '\.target sm_61' "$out"
    echo "wrote $out ($(wc -l < "$out") lines)"
done
grep -q '\.entry axpy_f32_to_f64' "$dir/axpy_f32_to_f64.ptx"
grep -q '\.entry gemv_f32_f64_n' "$dir/gemv_f32_f64.ptx"
grep -q '\.entry gemv_f32_f64_t' "$dir/gemv_f32_f64.ptx"
