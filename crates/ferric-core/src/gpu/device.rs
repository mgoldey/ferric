//! One `CudaContext` + default stream + cuBLAS handle per device ordinal,
//! created once and shared. The pre-check keeps cudarc's "library not found"
//! PANIC (cudarc src/lib.rs `panic_no_lib_found`) from ever firing: we dlopen
//! the two libraries ourselves first and turn absence into an error.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::CudaBlas;
use cudarc::driver::{sys, CudaContext, CudaStream};

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
}

pub struct Device {
    pub info: GpuInfo,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// cuBLAS handles are not safe for concurrent use; serialize callers.
    pub blas: Mutex<CudaBlas>,
}

fn precheck_libraries() -> Result<(), GpuError> {
    // SAFETY: loading a system shared library runs its constructors; these are
    // NVIDIA's own driver/cuBLAS libraries, the same ones cudarc will dlopen.
    unsafe {
        libloading::Library::new("libcuda.so.1")
            .map_err(|e| GpuError::DriverLibrary(e.to_string()))?;
        libloading::Library::new("libcublas.so.12")
            .or_else(|_| libloading::Library::new("libcublas.so"))
            .map_err(|e| GpuError::BlasLibrary(e.to_string()))?;
    }
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
    /// Current (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize), GpuError> {
        self.ctx
            .mem_get_info()
            .map_err(|e| GpuError::Cuda(format!("{e:?}")))
    }
}
