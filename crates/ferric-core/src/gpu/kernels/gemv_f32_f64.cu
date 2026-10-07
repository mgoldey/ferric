// Mixed-storage GEMV for the device RI-J: the matrix A is f32 (resident, row-major
// rows x cols), the vector x and the result y are f64, and every product and sum is
// an f64 FMA. No panel flush is needed (the accumulator is already f64), and the
// vector is never rounded to f32. ferric_core::gpu::gemv::gemv_f32mat_f64_dev
// documents the error bound.
//
//   gemv_f32_f64_n:  y[r] (+)= sum_c A[r*cols + c] * x[c]      one block (128 threads) per row
//   gemv_f32_f64_t:  y[c] (+)= sum_r A[r*cols + c] * x[r]      32 columns x 16 row groups per block
//
// The summation order is fixed (strided partial sums, then a fixed-order reduction),
// so a result is deterministic run to run. Compiled OFFLINE to PTX for compute_61 by
// scripts/gpu/regen_ptx.sh; never compiled at build or run time.
typedef unsigned long long u64;

extern "C" __global__ void gemv_f32_f64_n(const float* __restrict__ a,
                                          const double* __restrict__ x,
                                          double* __restrict__ y, u64 rows, u64 cols,
                                          int accumulate) {
    __shared__ double part[128];
    for (u64 r = blockIdx.x; r < rows; r += gridDim.x) {
        const float* row = a + r * cols;
        double acc = 0.0;
        for (u64 c = threadIdx.x; c < cols; c += blockDim.x) {
            acc = fma((double)row[c], x[c], acc);
        }
        part[threadIdx.x] = acc;
        __syncthreads();
        for (unsigned s = 64; s > 0; s >>= 1) {
            if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[r] = accumulate ? y[r] + part[0] : part[0];
        __syncthreads();
    }
}

extern "C" __global__ void gemv_f32_f64_t(const float* __restrict__ a,
                                          const double* __restrict__ x,
                                          double* __restrict__ y, u64 rows, u64 cols,
                                          int accumulate) {
    __shared__ double part[16][32];
    const u64 col = (u64)blockIdx.x * 32 + threadIdx.x;
    double acc = 0.0;
    if (col < cols) {
        for (u64 r = threadIdx.y; r < rows; r += 16) {
            acc = fma((double)a[r * cols + col], x[r], acc);
        }
    }
    part[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.y == 0 && col < cols) {
        double s = part[0][threadIdx.x];
        for (int g = 1; g < 16; ++g) s += part[g][threadIdx.x];
        y[col] = accumulate ? y[col] + s : s;
    }
}
