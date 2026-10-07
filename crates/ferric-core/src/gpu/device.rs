//! One `CudaContext` + default stream + cuBLAS handle per device ordinal,
//! created once and shared. The pre-check keeps cudarc's "library not found"
//! PANIC (cudarc src/lib.rs `panic_no_lib_found`) from ever firing: we dlopen
//! the two libraries ourselves first and turn absence into an error.
use std::collections::HashMap;
#[cfg(feature = "test-seams")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::CudaBlas;
use cudarc::driver::{sys, CudaContext, CudaFunction, CudaStream};
use cudarc::nvrtc::Ptx;

use super::GpuInfo;

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum GpuError {
    #[error("CUDA driver library not loadable: {0}")]
    DriverLibrary(String),
    #[error("cuBLAS library not loadable: {0}")]
    BlasLibrary(String),
    #[error("CUDA device {ordinal} unusable: {detail}")]
    Device { ordinal: usize, detail: String },
    #[error("device memory pool refused {label}: {detail}")]
    PoolFull { label: String, detail: String },
    #[error("operand layout not offloadable: {0}")]
    Layout(String),
    #[error("cuBLAS/CUDA call failed: {0}")]
    Cuda(String),
    #[error("device kernel unavailable: {0}")]
    Kernel(String),
    /// An operand element is not finite after rounding to f32 (|x| > f32::MAX
    /// overflows to inf); the caller should run this GEMM in f64.
    #[error("operand not representable in f32: {0}")]
    F32Range(String),
}

pub struct Device {
    pub info: GpuInfo,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// cuBLAS handles are not safe for concurrent use; serialize callers.
    pub blas: Mutex<CudaBlas>,
    /// Flush kernel for the mixed GEMM, loaded from committed PTX on first use.
    /// `Err` is cached too (one JIT attempt per process).
    kernel_axpy: OnceLock<Result<CudaFunction, String>>,
}

/// Test seam: makes `axpy_f32_to_f64` report a load failure so callers' f64
/// fallback can be exercised on a healthy machine. Compiled only with the
/// `test-seams` feature.
#[cfg(feature = "test-seams")]
#[doc(hidden)]
pub static FORCE_KERNEL_FAILURE: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "test-seams")]
fn kernel_failure_forced() -> bool {
    FORCE_KERNEL_FAILURE.load(Ordering::Relaxed)
}

#[cfg(not(feature = "test-seams"))]
fn kernel_failure_forced() -> bool {
    false
}

fn precheck_libraries() -> Result<(), GpuError> {
    precheck_named(&["libcuda.so.1"], &["libcublas.so.12", "libcublas.so"])
}

/// dlopen the first loadable name of each list; absence becomes a typed error.
fn precheck_named(driver: &[&str], blas: &[&str]) -> Result<(), GpuError> {
    fn first_loadable(names: &[&str]) -> Result<(), String> {
        let mut last = String::from("no library names given");
        for n in names {
            // SAFETY: loading a system shared library runs its constructors; these
            // are NVIDIA's own driver/cuBLAS libraries, the same ones cudarc will dlopen.
            match unsafe { libloading::Library::new(n) } {
                Ok(_) => return Ok(()),
                Err(e) => last = e.to_string(),
            }
        }
        Err(last)
    }
    first_loadable(driver).map_err(GpuError::DriverLibrary)?;
    first_loadable(blas).map_err(GpuError::BlasLibrary)?;
    Ok(())
}

fn open(ordinal: usize) -> Result<Arc<Device>, GpuError> {
    precheck_libraries()?;
    let count = std::panic::catch_unwind(CudaContext::device_count)
        .map_err(|_| GpuError::DriverLibrary("cudarc panicked while loading the driver".into()))?
        .map_err(|e| GpuError::Device {
            ordinal,
            detail: format!("cuInit/device_count: {e:?}"),
        })?;
    if ordinal >= count.max(0) as usize {
        return Err(GpuError::Device {
            ordinal,
            detail: format!("only {count} CUDA device(s) present (ordinals 0..{count})"),
        });
    }
    let ctx = CudaContext::new(ordinal).map_err(|e| GpuError::Device {
        ordinal,
        detail: format!("{e:?}"),
    })?;
    let name = ctx.name().map_err(|e| GpuError::Device {
        ordinal,
        detail: format!("{e:?}"),
    })?;
    let cc_major = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| GpuError::Device {
            ordinal,
            detail: format!("{e:?}"),
        })?;
    let cc_minor = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| GpuError::Device {
            ordinal,
            detail: format!("{e:?}"),
        })?;
    let (free_bytes, total_bytes) = ctx.mem_get_info().map_err(|e| GpuError::Device {
        ordinal,
        detail: format!("{e:?}"),
    })?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())
        .map_err(|e| GpuError::Cuda(format!("cublasCreate: {e:?}")))?;
    Ok(Arc::new(Device {
        info: GpuInfo {
            ordinal,
            name,
            cc_major,
            cc_minor,
            free_bytes,
            total_bytes,
        },
        ctx,
        stream,
        blas: Mutex::new(blas),
        kernel_axpy: OnceLock::new(),
    }))
}

type Registry = Mutex<HashMap<usize, Result<Arc<Device>, GpuError>>>;

/// The shared device for `ordinal`, opened on first use. A failed open is
/// cached too, so a missing driver costs one attempt per process, not one per
/// GEMM.
pub fn device(ordinal: usize) -> Result<Arc<Device>, GpuError> {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    let reg = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = reg.lock().unwrap_or_else(|e| e.into_inner());
    map.entry(ordinal).or_insert_with(|| open(ordinal)).clone()
}

impl Device {
    /// The `c64 += (double) c32` flush kernel, JIT-loaded from the committed PTX.
    pub fn axpy_f32_to_f64(&self) -> Result<&CudaFunction, GpuError> {
        if kernel_failure_forced() {
            return Err(GpuError::Kernel(
                "forced by FORCE_KERNEL_FAILURE (test)".into(),
            ));
        }
        self.kernel_axpy
            .get_or_init(|| {
                let ptx = Ptx::from_src(include_str!("kernels/axpy_f32_to_f64.ptx"));
                let module = self
                    .ctx
                    .load_module(ptx)
                    .map_err(|e| format!("cuModuleLoadData(axpy_f32_to_f64.ptx): {e:?}"))?;
                module
                    .load_function("axpy_f32_to_f64")
                    .map_err(|e| format!("cuModuleGetFunction(axpy_f32_to_f64): {e:?}"))
            })
            .as_ref()
            .map_err(|e| GpuError::Kernel(e.clone()))
    }

    /// Current (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize), GpuError> {
        self.ctx
            .mem_get_info()
            .map_err(|e| GpuError::Cuda(format!("{e:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_driver_library_is_a_driver_error() {
        let e = precheck_named(&["libferric_no_such_driver.so.1"], &["libc.so.6"]).unwrap_err();
        assert!(
            matches!(e, GpuError::DriverLibrary(ref m) if !m.is_empty()),
            "{e:?}"
        );
    }

    #[test]
    fn absent_blas_library_is_a_blas_error() {
        // libc is always loadable, so only the cuBLAS slot fails.
        let e = precheck_named(&["libc.so.6"], &["libferric_no_such_blas.so"]).unwrap_err();
        assert!(
            matches!(e, GpuError::BlasLibrary(ref m) if !m.is_empty()),
            "{e:?}"
        );
    }
}
