//! A matrix that lives on the device for longer than one GEMM. Holds its own
//! pool reservation (drop-of-tensor is drop-of-charge, the same rule as
//! `ThreeIndexSource`), knows its shape, and nothing else: consumers build
//! cuBLAS operand descriptors over `buf()` themselves (see
//! `gemm::dev_left_padded` / `dev_right_padded`), because the first consumer
//! (RI-MP2) needs column sub-ranges of a row-major matrix whose leading
//! dimension is not the sub-range's width.
//!
//! All size arithmetic is checked: an element count whose byte size overflows
//! `usize` is a typed `GpuError::Layout`, never a wrap or a panic.
use cudarc::driver::CudaSlice;
use ndarray::ArrayView2;

use super::device::{Device, GpuError};
use super::mixed::round_to_f32;
use super::pool::{DevicePool, DeviceReservation};
use super::stats;

pub struct DeviceMatrix<T> {
    buf: CudaSlice<T>,
    rows: usize,
    cols: usize,
    bytes: usize,
    _lease: DeviceReservation,
}

impl<T> std::fmt::Debug for DeviceMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("bytes", &self.bytes)
            .finish()
    }
}

fn standard<'a>(m: &'a ArrayView2<f64>) -> Result<&'a [f64], GpuError> {
    if !m.is_standard_layout() {
        return Err(GpuError::Layout(
            "resident matrix must be row-major contiguous".into(),
        ));
    }
    m.as_slice()
        .ok_or_else(|| GpuError::Layout("resident matrix not contiguous".into()))
}

/// `elem_bytes · len`, or a typed refusal when it does not fit `usize`.
fn byte_len(len: usize, elem_bytes: usize) -> Result<usize, GpuError> {
    len.checked_mul(elem_bytes)
        .ok_or_else(|| GpuError::Layout(format!("{len} elements of {elem_bytes} B overflow usize")))
}

impl DeviceMatrix<f64> {
    /// Upload as-is. Charges `8·rows·cols` to `pool` under `label` first; a
    /// refusal moves no bytes.
    pub fn upload(
        dev: &Device,
        pool: &DevicePool,
        label: &str,
        m: &ArrayView2<f64>,
    ) -> Result<Self, GpuError> {
        let host = standard(m)?;
        let bytes = byte_len(host.len(), 8)?;
        let lease = pool.reserve(label, bytes)?;
        let buf = dev
            .stream
            .clone_htod(host)
            .map_err(|e| GpuError::Cuda(format!("H2D {label}: {e:?}")))?;
        stats::note_resident_upload(bytes);
        Ok(Self {
            buf,
            rows: m.nrows(),
            cols: m.ncols(),
            bytes,
            _lease: lease,
        })
    }
}

impl DeviceMatrix<f64> {
    /// A zeroed `rows × cols` device matrix (a GEMM result buffer that lives
    /// as long as its charge). Charges `8·rows·cols` first; not an upload, so
    /// it moves no bytes and is not counted as one.
    pub fn zeros(
        dev: &Device,
        pool: &DevicePool,
        label: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self, GpuError> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| GpuError::Layout(format!("{rows}x{cols} overflows usize")))?;
        let bytes = byte_len(len, 8)?;
        let lease = pool.reserve(label, bytes)?;
        let buf = dev
            .stream
            .alloc_zeros::<f64>(len)
            .map_err(|e| GpuError::Cuda(format!("alloc {label}: {e:?}")))?;
        Ok(Self {
            buf,
            rows,
            cols,
            bytes,
            _lease: lease,
        })
    }
}

impl DeviceMatrix<f32> {
    /// A zeroed `rows × cols` f32 device matrix (a panel scratch that lives as
    /// long as its charge). Charges `4·rows·cols` first; moves no bytes.
    pub fn zeros(
        dev: &Device,
        pool: &DevicePool,
        label: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self, GpuError> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| GpuError::Layout(format!("{rows}x{cols} overflows usize")))?;
        let bytes = byte_len(len, 4)?;
        let lease = pool.reserve(label, bytes)?;
        let buf = dev
            .stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| GpuError::Cuda(format!("alloc {label}: {e:?}")))?;
        Ok(Self {
            buf,
            rows,
            cols,
            bytes,
            _lease: lease,
        })
    }

    /// Upload `x as f32` (IEEE round-to-nearest-even). Charges `4·rows·cols`.
    /// A finite value that overflows f32 is refused (`GpuError::F32Range`)
    /// before the pool is touched.
    pub fn upload_rounded(
        dev: &Device,
        pool: &DevicePool,
        label: &str,
        m: &ArrayView2<f64>,
    ) -> Result<Self, GpuError> {
        let host = standard(m)?;
        let bytes = byte_len(host.len(), 4)?;
        let rounded = round_to_f32(label, host)?;
        let lease = pool.reserve(label, bytes)?;
        let buf = dev
            .stream
            .clone_htod(&rounded)
            .map_err(|e| GpuError::Cuda(format!("H2D {label}: {e:?}")))?;
        stats::note_resident_upload(bytes);
        Ok(Self {
            buf,
            rows: m.nrows(),
            cols: m.ncols(),
            bytes,
            _lease: lease,
        })
    }
}

impl<T> DeviceMatrix<T> {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn buf(&self) -> &CudaSlice<T> {
        &self.buf
    }
    /// Mutable access for GEMM result buffers ([`DeviceMatrix::zeros`]).
    pub fn buf_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.buf
    }
}
