// c64[i] += (double) c32[i], grid-stride. The f64 accumulation step of the
// k-panelled mixed GEMM (ferric_core::gpu::mixed). cuBLAS has no f32-in /
// f64-accumulate GEMM (cublasGemmEx: 64F compute needs 64F A/B/C), hence this.
// Compiled OFFLINE to PTX for compute_61 by scripts/gpu/regen_ptx.sh; the
// driver JIT-compiles the PTX for sm_61 and every later architecture. Never
// compiled at build or run time; CI needs neither nvcc nor libnvrtc.
extern "C" __global__ void axpy_f32_to_f64(double* c64, const float* c32,
                                           unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        c64[i] += (double)c32[i];
    }
}
