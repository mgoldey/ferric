//! Owned cuEST objects (handle, parameter sets, workspaces) and status mapping.

use crate::cuda::DeviceBuffer;
use crate::ffi;
use crate::{CuestError, Result};
use std::os::raw::{c_int, c_void};

/// Turn a cuEST status into a `Result`, naming the call that produced it.
pub fn check(call: &'static str, status: ffi::cuestStatus_t) -> Result<()> {
    if status == ffi::CUEST_STATUS_SUCCESS {
        return Ok(());
    }
    let name = usize::try_from(status)
        .ok()
        .and_then(|i| ffi::STATUS_NAMES.get(i).copied())
        .unwrap_or("UNRECOGNIZED");
    Err(CuestError::Call { call, status, name })
}

/// An owned cuEST handle. Destroyed on drop.
pub struct CuestHandle {
    raw: ffi::cuestHandle_t,
}

impl CuestHandle {
    /// Create a handle with default parameters (cuEST allocates its own
    /// stream, cuBLAS and cuSOLVER handles).
    ///
    /// An unsupported GPU (cuEST needs compute capability 8.6+) comes back as
    /// [`CuestError::DeviceUnavailable`], so callers can skip rather than fail.
    pub fn new() -> Result<Self> {
        let params = Params::new(ffi::CUEST_HANDLE_PARAMETERS)?;
        let mut raw: ffi::cuestHandle_t = std::ptr::null_mut();
        // SAFETY: params is a live handle-parameters object; raw is a valid out-pointer.
        let st = unsafe { ffi::cuestCreate(params.raw, &mut raw) };
        if st == ffi::CUEST_STATUS_UNSUPPORTED_ARCHITECTURE {
            let cc = crate::cuda::DeviceInfo::probe()
                .map(|d| format!("sm_{}{}", d.compute_capability.0, d.compute_capability.1))
                .unwrap_or_else(|e| format!("unknown ({e})"));
            return Err(CuestError::DeviceUnavailable(format!(
                "cuestCreate returned UNSUPPORTED_ARCHITECTURE: device 0 is {cc}, cuEST needs sm_86 or newer"
            )));
        }
        check("cuestCreate", st)?;
        if raw.is_null() {
            return Err(CuestError::Call {
                call: "cuestCreate",
                status: 0,
                name: "SUCCESS_BUT_NULL_HANDLE",
            });
        }
        Ok(Self { raw })
    }

    pub fn raw(&self) -> ffi::cuestHandle_t {
        self.raw
    }

    /// cuEST library version as (major, minor, patch).
    pub fn version(&self) -> Result<(u32, u32, u32)> {
        let (mut a, mut b, mut c) = (0, 0, 0);
        // SAFETY: live handle, valid out-pointers.
        unsafe {
            check(
                "cuestGetMajorVersion",
                ffi::cuestGetMajorVersion(self.raw, &mut a),
            )?;
            check(
                "cuestGetMinorVersion",
                ffi::cuestGetMinorVersion(self.raw, &mut b),
            )?;
            check(
                "cuestGetPatchVersion",
                ffi::cuestGetPatchVersion(self.raw, &mut c),
            )?;
        }
        Ok((a, b, c))
    }
}

impl Drop for CuestHandle {
    fn drop(&mut self) {
        // SAFETY: raw came from cuestCreate and is destroyed once.
        unsafe {
            let _ = ffi::cuestDestroy(self.raw);
        }
    }
}

/// An owned `cuest*Parameters_t` of one parameter type. Destroyed on drop.
pub struct Params {
    pub(crate) raw: ffi::cuestParams_t,
    kind: c_int,
}

impl Params {
    pub fn new(kind: c_int) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        check("cuestParametersCreate", unsafe {
            ffi::cuestParametersCreate(kind, &mut raw)
        })?;
        Ok(Self { raw, kind })
    }

    /// Set an `int`-sized (enum) attribute.
    pub fn set_enum(&mut self, attribute: c_int, value: c_int) -> Result<()> {
        // SAFETY: value lives across the call; size matches its type.
        check("cuestParametersConfigure", unsafe {
            ffi::cuestParametersConfigure(
                self.kind,
                self.raw,
                attribute,
                (&value as *const c_int).cast::<c_void>(),
                std::mem::size_of::<c_int>(),
            )
        })
    }
}

impl Drop for Params {
    fn drop(&mut self) {
        // SAFETY: raw came from cuestParametersCreate with this kind.
        unsafe {
            let _ = ffi::cuestParametersDestroy(self.kind, self.raw);
        }
    }
}

/// Host + device buffers sized from a cuEST workspace descriptor.
pub struct Workspace {
    _host: Vec<u64>,
    _device: DeviceBuffer,
    pub(crate) raw: ffi::cuestWorkspace_t,
}

impl Workspace {
    /// A zero-byte request still gets a small real buffer on each side, so
    /// cuEST never sees a NULL buffer pointer (its docs promise nothing about
    /// NULL with size 0); the recorded sizes stay the requested ones.
    pub fn allocate(desc: &ffi::cuestWorkspaceDescriptor_t) -> Result<Self> {
        let mut host = vec![0u64; desc.hostBufferSizeInBytes.div_ceil(8).max(1)];
        let device = DeviceBuffer::new(desc.deviceBufferSizeInBytes.max(256))?;
        let raw = ffi::cuestWorkspace_t {
            hostBuffer: host.as_mut_ptr() as usize,
            hostBufferSizeInBytes: desc.hostBufferSizeInBytes,
            deviceBuffer: device.ptr() as usize,
            deviceBufferSizeInBytes: desc.deviceBufferSizeInBytes,
        };
        Ok(Self {
            _host: host,
            _device: device,
            raw,
        })
    }

    /// Bytes held on the device.
    pub fn device_bytes(&self) -> usize {
        self.raw.deviceBufferSizeInBytes
    }
}
