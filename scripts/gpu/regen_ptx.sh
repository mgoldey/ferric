#!/usr/bin/env bash
# Regenerate the committed PTX for the gpu flush kernel. Needs nvcc 12.x
# (13.x cannot emit compute_61). Commit the .cu and .ptx together.
set -euo pipefail
cd "$(dirname "$0")/../.."
src=crates/ferric-core/src/gpu/kernels/axpy_f32_to_f64.cu
out=crates/ferric-core/src/gpu/kernels/axpy_f32_to_f64.ptx
nvcc --version | tail -1
nvcc -arch=compute_61 -O3 -ptx -o "$out" "$src"
grep -q '\.entry axpy_f32_to_f64' "$out"
grep -q '\.target sm_61' "$out"
echo "wrote $out ($(wc -l < "$out") lines)"
