// out[i] = (float) in[i], IEEE round-to-nearest-even (__double2float_rn, the
// same rounding as Rust's `v as f32`), grid-stride. Sets *overflow = 1 when a
// finite input rounds to an infinity (|v| beyond f32::MAX), so the caller can
// refuse the copy as `GpuError::F32Range` like the host `round_to_f32`. Every
// writer stores the same value, so the unsynchronised store is benign.
// Compiled OFFLINE to PTX for compute_61 by scripts/gpu/regen_ptx.sh (no
// fast-math: subnormals are kept, not flushed to zero).
extern "C" __global__ void round_f64_to_f32(float* out, const double* in,
                                            unsigned long long n,
                                            unsigned int* overflow) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        const double v = in[i];
        const float r = __double2float_rn(v);
        if (isfinite(v) && isinf(r)) {
            *overflow = 1u;
        }
        out[i] = r;
    }
}
