//! Minimal CUDA runtime helpers: device probing and an owned device buffer.

use crate::ffi;
use crate::{CuestError, Result};
use std::ffi::CStr;
use std::os::raw::c_void;

fn cuda_err(call: &'static str, code: ffi::cudaError_t) -> CuestError {
    // SAFETY: cudaGetErrorString returns a static NUL-terminated string.
    let msg = unsafe { CStr::from_ptr(ffi::cudaGetErrorString(code)) }
        .to_string_lossy()
        .into_owned();
    CuestError::Cuda { call, code, msg }
}

pub(crate) fn check(call: &'static str, code: ffi::cudaError_t) -> Result<()> {
    if code == ffi::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(cuda_err(call, code))
    }
}

/// What the CUDA runtime reports about device 0.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_count: i32,
    pub compute_capability: (i32, i32),
    pub runtime_version: i32,
    pub driver_version: i32,
}

impl DeviceInfo {
    pub fn probe() -> Result<Self> {
        let (mut n, mut major, mut minor, mut rt, mut drv) = (0, 0, 0, 0, 0);
        // SAFETY: out-pointers are valid locals.
        unsafe {
            check("cudaGetDeviceCount", ffi::cudaGetDeviceCount(&mut n))?;
            if n == 0 {
                return Err(CuestError::DeviceUnavailable("no CUDA device".into()));
            }
            check(
                "cudaDeviceGetAttribute",
                ffi::cudaDeviceGetAttribute(
                    &mut major,
                    ffi::CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR,
                    0,
                ),
            )?;
            check(
                "cudaDeviceGetAttribute",
                ffi::cudaDeviceGetAttribute(
                    &mut minor,
                    ffi::CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR,
                    0,
                ),
            )?;
            check("cudaRuntimeGetVersion", ffi::cudaRuntimeGetVersion(&mut rt))?;
            check("cudaDriverGetVersion", ffi::cudaDriverGetVersion(&mut drv))?;
        }
        Ok(Self {
            device_count: n,
            compute_capability: (major, minor),
            runtime_version: rt,
            driver_version: drv,
        })
    }
}

/// An owned device allocation, freed on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl DeviceBuffer {
    /// Allocate `bytes` on the device; a zero-byte request holds no allocation.
    pub fn new(bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        if bytes > 0 {
            // SAFETY: valid out-pointer.
            check("cudaMalloc", unsafe { ffi::cudaMalloc(&mut ptr, bytes) })?;
        }
        Ok(Self { ptr, bytes })
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn copy_from_host(&mut self, src: &[f64]) -> Result<()> {
        let n = std::mem::size_of_val(src);
        assert!(
            n <= self.bytes,
            "copy_from_host: {n} bytes into a {}-byte buffer",
            self.bytes
        );
        if n == 0 {
            return Ok(());
        }
        // SAFETY: both ranges are valid for n bytes.
        check("cudaMemcpy(H2D)", unsafe {
            ffi::cudaMemcpy(
                self.ptr,
                src.as_ptr().cast(),
                n,
                ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        })
    }

    /// Synchronize the device, then copy the buffer into `dst`.
    pub fn copy_to_host(&self, dst: &mut [f64]) -> Result<()> {
        let n = std::mem::size_of_val(dst);
        assert!(
            n <= self.bytes,
            "copy_to_host: {n} bytes from a {}-byte buffer",
            self.bytes
        );
        // cuEST runs on its own stream; wait for it before reading.
        // SAFETY: no arguments.
        check("cudaDeviceSynchronize", unsafe {
            ffi::cudaDeviceSynchronize()
        })?;
        if n == 0 {
            return Ok(());
        }
        // SAFETY: both ranges are valid for n bytes.
        check("cudaMemcpy(D2H)", unsafe {
            ffi::cudaMemcpy(
                dst.as_mut_ptr().cast(),
                self.ptr,
                n,
                ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from cudaMalloc and is freed once.
            unsafe {
                let _ = ffi::cudaFree(self.ptr);
            }
        }
    }
}
