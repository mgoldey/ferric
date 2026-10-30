//! Reusable host staging buffers for device transfers.
//!
//! A transfer from or into memory the driver knows is page-locked runs at the
//! bus rate and asynchronously; one into fresh pageable memory first faults
//! every page in and then goes through the driver's bounce buffer (measured on
//! a GTX 1080: 1.3 GB/s device-to-host into fresh pageable memory, 6.8 GB/s
//! into reused pageable memory, 11.4 GB/s into page-locked memory). A
//! [`HostBuf`] is allocated once and handed out as plain slices, so the copies
//! go through the ordinary `memcpy_htod` / `memcpy_dtoh` calls.
//!
//! Page-locked memory is a scarce system resource: callers cap what they ask
//! for, fall back to [`HostBuf::heap`] on refusal, and drop the buffer with
//! the device state it serves.
use cudarc::driver::{DeviceRepr, PinnedHostSlice, ValidAsZeroBits};

use super::device::{Device, GpuError};

/// An element type a [`HostBuf`] can hold (`f32`, `f64`).
pub trait HostElem: DeviceRepr + ValidAsZeroBits + Copy + Default {}
impl HostElem for f32 {}
impl HostElem for f64 {}

/// One host staging buffer: page-locked, or ordinary heap memory.
pub enum HostBuf<T> {
    Pinned(PinnedHostSlice<T>),
    Heap(Vec<T>),
}

impl<T: HostElem> HostBuf<T> {
    /// `len` zeroed elements of page-locked memory (not write-combined, so host
    /// reads stay fast). `Err` when the driver refuses.
    pub fn pinned(dev: &Device, len: usize) -> Result<Self, GpuError> {
        let err = |e| GpuError::Cuda(format!("cuMemHostAlloc({len} elements): {e:?}"));
        // SAFETY: flags 0 = plain page-locked memory of `dev`'s context; it is
        // zeroed below before any slice of it is formed.
        let mut p = unsafe { dev.ctx.alloc_pinned_with_flags::<T>(len.max(1), 0) }.map_err(err)?;
        let ptr = p.as_mut_ptr().map_err(err)?;
        // SAFETY: `ptr` is the start of `p.len()` freshly allocated `T`s.
        unsafe { std::ptr::write_bytes(ptr, 0, p.len()) };
        Ok(HostBuf::Pinned(p))
    }

    /// `len` zeroed elements on the heap.
    pub fn heap(len: usize) -> Self {
        HostBuf::Heap(vec![T::default(); len])
    }

    pub fn is_pinned(&self) -> bool {
        matches!(self, HostBuf::Pinned(_))
    }

    pub fn len(&self) -> usize {
        match self {
            HostBuf::Pinned(p) => p.len(),
            HostBuf::Heap(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    /// The first `n` elements (a typed `Layout` error past the end).
    pub fn head(&self, n: usize) -> Result<&[T], GpuError> {
        let all = match self {
            HostBuf::Pinned(p) => p.as_slice().map_err(|e| pin_err("read", e))?,
            HostBuf::Heap(v) => v.as_slice(),
        };
        all.get(..n).ok_or_else(|| too_short(n, all.len()))
    }

    /// The first `n` elements, writable (a typed `Layout` error past the end).
    pub fn head_mut(&mut self, n: usize) -> Result<&mut [T], GpuError> {
        let all = match self {
            HostBuf::Pinned(p) => p.as_mut_slice().map_err(|e| pin_err("write", e))?,
            HostBuf::Heap(v) => v.as_mut_slice(),
        };
        let len = all.len();
        all.get_mut(..n).ok_or_else(|| too_short(n, len))
    }
}

fn pin_err(what: &str, e: impl std::fmt::Debug) -> GpuError {
    GpuError::Cuda(format!("page-locked host buffer ({what}): {e:?}"))
}

fn too_short(n: usize, len: usize) -> GpuError {
    GpuError::Layout(format!("host buffer holds {len} elements, {n} requested"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_buffers_hand_out_exact_heads_and_refuse_overruns() {
        let mut b = HostBuf::<f64>::heap(8);
        assert!(!b.is_pinned());
        assert_eq!((b.len(), b.bytes()), (8, 64));
        b.head_mut(3).unwrap().copy_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(b.head(4).unwrap(), &[1.0, 2.0, 3.0, 0.0]);
        assert!(matches!(b.head(9), Err(GpuError::Layout(_))));
        assert!(matches!(b.head_mut(9), Err(GpuError::Layout(_))));
    }
}
