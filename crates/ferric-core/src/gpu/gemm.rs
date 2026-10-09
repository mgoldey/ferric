//! Row-major f64 GEMM on the device. `out = left · right` is computed as the
//! column-major product `outᵀ = rightᵀ · leftᵀ`, so a row-major standard
//! operand is passed as-is with op N and ld = ncols, and a transposed-standard
//! view is passed with op T and ld = nrows. The k axis is blocked at the same
//! width as the CPU (`ferric_tensors::einsum::GEMM_K_BLOCK` = 128) in ascending
//! order, beta 0 then 1 — the same coarse-pairwise structure, NOT bit-identical
//! to the CPU (cuBLAS blocks inside each slab its own way). See the parity test
//! for the bound that IS guaranteed.
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, CudaView};
use ndarray::{ArrayView2, ArrayViewMut2};

use super::device::{Device, GpuError};
use super::pool::DevicePool;
use super::stats;

/// How cuBLAS should read one row-major operand.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    pub op: cublasOperation_t,
    pub ld: i32,
    /// Element offset per unit of k: `ld` when k runs along the stored
    /// column-major matrix's columns (op N), `1` when along its rows (op T).
    pub k_step: usize,
}

/// `Some` for the two layouts the device path accepts, `None` otherwise.
/// `k_is_cols` says whether the contraction index is this operand's COLUMN
/// axis in row-major terms (true for `left`, false for `right`).
pub(crate) fn col_major_desc<T>(v: &ArrayView2<T>, k_is_cols: bool) -> Option<Desc> {
    let (r, c) = (v.nrows(), v.ncols());
    if v.is_standard_layout() {
        // Row-major (r×c) memory read as column-major with ld = c is the
        // transpose (c×r) — exactly the factor the swapped product wants, so
        // op N. In that stored (c×r) matrix the row-major COLUMN axis is the
        // ROW axis, and the row-major ROW axis is the COLUMN axis. cuBLAS
        // sub-views: rows k0.. of a column-major matrix start at offset k0
        // (same ld); columns k0.. start at k0·ld. Hence k_step = 1 when k is
        // the row-major column axis (left operand), k_step = c (= ld) otherwise (right).
        let k_step = if k_is_cols { 1 } else { c };
        return Some(Desc {
            op: cublasOperation_t::CUBLAS_OP_N,
            ld: c as i32,
            k_step,
        });
    }
    if v.t().is_standard_layout() {
        // Memory is already column-major (r×c) with ld = r; op T supplies the
        // transpose. Here the row-major column axis IS the stored column axis
        // (offset k0·r per k) and the row-major row axis is the stored row
        // axis (offset k0).
        let k_step = if k_is_cols { r } else { 1 };
        return Some(Desc {
            op: cublasOperation_t::CUBLAS_OP_T,
            ld: r as i32,
            k_step,
        });
    }
    None
}

// Worked check of the four cases (the parity test pins them):
//   left  standard   (m×k, strides (k,1)):  stored col-major (k×m) ld=k, op N, k along rows    → k_step 1
//   right standard   (k×n, strides (n,1)):  stored col-major (n×k) ld=n, op N, k along columns → k_step n
//   left  transposed (m×k, strides (1,m)):  stored col-major (m×k) ld=m, op T, k along columns → k_step m
//   right transposed (k×n, strides (1,k)):  stored col-major (k×n) ld=k, op T, k along rows    → k_step 1

/// Bytes the pool is charged for one offload: both operands and the result.
pub fn offload_bytes(m: usize, k: usize, n: usize) -> usize {
    8 * (m * k + k * n + m * n)
}

/// Which side of `out = left · right` an operand is. The role fixes how the
/// contraction index k runs through the stored matrix, so it is never a free
/// parameter: the public constructors are [`dev_left`] / [`dev_right`] (exact
/// layouts) and [`dev_left_padded`] / [`dev_right_padded`] (sub-blocks with a
/// wider leading dimension); each fixes the role itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// m × k, k along the row-major COLUMN axis.
    Left,
    /// k × n, k along the row-major ROW axis.
    Right,
}

/// The plain numbers a device operand is made of, so the soundness check is a
/// pure function (unit-tested without a device).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OperandGeom {
    pub role: Role,
    pub op: cublasOperation_t,
    pub ld: i32,
    pub k_step: usize,
    pub rows: usize,
    pub cols: usize,
}

impl OperandGeom {
    /// Derive the geometry from the host twin's layout and the role; `k_step`
    /// comes from `col_major_desc(host, role == Left)`, never from the caller.
    pub(crate) fn derive<H>(
        role: Role,
        host: &ArrayView2<H>,
        len: usize,
    ) -> Result<Self, GpuError> {
        let d = col_major_desc(host, role == Role::Left).ok_or_else(|| {
            GpuError::Layout("operand is neither standard nor transposed-standard".into())
        })?;
        let (rows, cols) = (host.nrows(), host.ncols());
        if len != rows * cols {
            return Err(GpuError::Layout(format!(
                "device operand holds {len} elements but its {rows}x{cols} shape needs {}",
                rows * cols
            )));
        }
        Ok(Self {
            role,
            op: d.op,
            ld: d.ld,
            k_step: d.k_step,
            rows,
            cols,
        })
    }

    /// The (ld, k_step) a well-formed operand of this role, op and shape has.
    fn expected_ld_k_step(&self) -> (usize, usize) {
        let n_op = self.op == cublasOperation_t::CUBLAS_OP_N;
        match (self.role, n_op) {
            (Role::Left, true) => (self.cols, 1),
            (Role::Left, false) => (self.rows, self.rows),
            (Role::Right, true) => (self.cols, self.cols),
            (Role::Right, false) => (self.rows, 1),
        }
    }

    /// Like [`Self::derive`] for a host twin that is a sub-block of a larger
    /// row-major matrix: one axis has unit stride and the other a
    /// non-negative stride `ld` at least as long as the unit axis (a standard
    /// view with padded rows, or its transpose). `ld` becomes the cuBLAS
    /// leading dimension and `len` (the device view's length, counted from the
    /// block's first element) must cover the block's extent
    /// `ld·(outer − 1) + inner`. Anything else (overlapping rows, negative or
    /// no unit stride, an `ld` beyond cuBLAS's i32) is a typed `Layout` error.
    /// Total on zero-size views (nothing is read, only the shape is kept).
    pub(crate) fn derive_padded<H>(
        role: Role,
        host: &ArrayView2<H>,
        len: usize,
    ) -> Result<Self, GpuError> {
        let (rows, cols) = (host.nrows(), host.ncols());
        let (rs0, cs0) = (host.strides()[0], host.strides()[1]);
        // The stride of a length-1 axis is never stepped (ndarray stores 0 for
        // it, and a transposed view keeps whatever it had), so it must not
        // decide the layout: a single row (column) is N or T by the OTHER
        // axis' stride alone.
        let rs = if rows == 1 {
            if cs0 == 1 {
                isize::try_from(cols).unwrap_or(isize::MAX)
            } else {
                1
            }
        } else {
            rs0
        };
        let cs = if cols == 1 {
            if rs0 == 1 {
                isize::try_from(rows).unwrap_or(isize::MAX)
            } else {
                1
            }
        } else {
            cs0
        };
        let bad = || {
            GpuError::Layout(
                "padded operand needs a unit stride on one axis and a leading \
                 dimension at least as long as the other axis"
                    .into(),
            )
        };
        let long_enough = |stride: isize, axis_len: usize| {
            usize::try_from(stride).ok().filter(|&ld| ld >= axis_len)
        };
        // (op, ld, outer, inner): `outer` axes of `inner` contiguous elements.
        let (op, ld, outer, inner) = if rows == 0 || cols == 0 {
            (cublasOperation_t::CUBLAS_OP_N, cols, rows, cols)
        } else if let (1, Some(ld)) = (cs, long_enough(rs, cols)) {
            (cublasOperation_t::CUBLAS_OP_N, ld, rows, cols)
        } else if let (1, Some(ld)) = (rs, long_enough(cs, rows)) {
            (cublasOperation_t::CUBLAS_OP_T, ld, cols, rows)
        } else {
            return Err(bad());
        };
        let ld_i32 = i32::try_from(ld).map_err(|_| {
            GpuError::Layout(format!("leading dimension {ld} exceeds cuBLAS i32 range"))
        })?;
        let extent = if outer == 0 || inner == 0 {
            0
        } else {
            ld.saturating_mul(outer - 1).saturating_add(inner)
        };
        if len < extent {
            return Err(GpuError::Layout(format!(
                "device view holds {len} elements but the {rows}x{cols} block (ld {ld}) spans {extent}"
            )));
        }
        let k_step = match (role, op == cublasOperation_t::CUBLAS_OP_N) {
            (Role::Left, true) | (Role::Right, false) => 1,
            _ => ld,
        };
        Ok(Self {
            role,
            op,
            ld: ld_i32,
            k_step,
            rows,
            cols,
        })
    }

    /// `(ld, k_step)` is a well-formed descriptor for this role, op and shape:
    /// `ld` no smaller than the stored matrix's row count and `k_step` the
    /// unique value for that `ld` (1 when k runs along the stored rows, `ld`
    /// when along its columns). Exact (`ld` = minimum) for [`Self::derive`];
    /// larger for [`Self::derive_padded`]. The per-panel extent check then
    /// bounds every element read, whatever `ld` is.
    fn descriptor_ok(&self) -> bool {
        let (min_ld, _) = self.expected_ld_k_step();
        let Ok(ld) = usize::try_from(self.ld) else {
            return false;
        };
        let n_op = self.op == cublasOperation_t::CUBLAS_OP_N;
        let k_step = match (self.role, n_op) {
            (Role::Left, true) | (Role::Right, false) => 1,
            _ => ld,
        };
        ld >= min_ld && self.k_step == k_step
    }

    /// One past the last element cuBLAS reads for the panel [k0, k0 + kc), in
    /// the swapped column-major call (right is cuBLAS A with m_c = n, left is
    /// cuBLAS B with n_c = m).
    ///
    /// Total on all inputs: saturating arithmetic, so no argument (zero sizes
    /// included, where nothing is read and the caller skips this) can panic or wrap.
    fn panel_end(&self, m: usize, n: usize, k0: usize, kc: usize) -> usize {
        let ld = usize::try_from(self.ld).unwrap_or(usize::MAX);
        let n_op = self.op == cublasOperation_t::CUBLAS_OP_N;
        let strided = |count: usize, tail: usize| {
            ld.saturating_mul(count.saturating_sub(1))
                .saturating_add(tail)
        };
        let extent = match (self.role, n_op) {
            (Role::Right, true) => strided(kc, n),
            (Role::Right, false) => strided(n, kc),
            (Role::Left, true) => strided(m, kc),
            (Role::Left, false) => strided(kc, m),
        };
        k0.saturating_mul(self.k_step).saturating_add(extent)
    }
}

/// One operand already on the device, described the way cuBLAS reads it after
/// the row-major → column-major swap (`outᵀ = rightᵀ·leftᵀ`): `view` starts at
/// the operand's first element, `ld` is its leading dimension in the stored
/// column-major matrix, `op` says whether cuBLAS transposes it, and `k_step`
/// is the element offset per unit of k (see `col_major_desc`).
///
/// What is guaranteed: the fields are crate-private; the only public
/// constructors are [`dev_left`] / [`dev_right`] (exact layouts) and
/// [`dev_left_padded`] / [`dev_right_padded`] (sub-blocks of a larger
/// row-major matrix). They fix the role and derive (op, ld, k_step) from the
/// host twin's layout (the caller supplies no descriptor value); the exact
/// constructors require `view.len() == host.len()`, the padded ones a view
/// that reaches the block's extent. Before any cuBLAS call, the `*_dev` GEMMs
/// run `check_dev_geometry`: operand roles and shapes against `(m, k, n)`;
/// the descriptor against the role, op and shape (`ld` no smaller than the
/// stored matrix's row count, `ld > min` allowed for padded blocks, and
/// `k_step` the unique value for that `ld`: 1 when k runs along the stored
/// rows, `ld` when along its columns); the highest element any panel reads
/// (`k0·k_step` plus the cuBLAS read extent, for EVERY panel) against the view
/// length; output buffers against m·n. Violations are typed
/// `GpuError::Layout`, nothing is launched.
pub struct DevOperand<'a, T> {
    pub(crate) view: CudaView<'a, T>,
    pub(crate) geom: OperandGeom,
}

impl<'a, T> DevOperand<'a, T> {
    pub(crate) fn new<H>(
        view: CudaView<'a, T>,
        host: &ArrayView2<H>,
        role: Role,
    ) -> Result<Self, GpuError> {
        let geom = OperandGeom::derive(role, host, view.len())?;
        Ok(Self { view, geom })
    }
}

impl<'a, T> DevOperand<'a, T> {
    pub(crate) fn new_padded<H>(
        view: CudaView<'a, T>,
        host: &ArrayView2<H>,
        role: Role,
    ) -> Result<Self, GpuError> {
        let geom = OperandGeom::derive_padded(role, host, view.len())?;
        Ok(Self { view, geom })
    }
}

/// Resident LEFT operand (m × k) whose host twin `host` is standard or
/// transposed-standard; `view` holds exactly `host.len()` elements in the
/// host's memory order, otherwise `GpuError::Layout`.
pub fn dev_left<'a, T, H>(
    view: CudaView<'a, T>,
    host: &ArrayView2<H>,
) -> Result<DevOperand<'a, T>, GpuError> {
    DevOperand::new(view, host, Role::Left)
}

/// Resident RIGHT operand (k × n); see [`dev_left`].
pub fn dev_right<'a, T, H>(
    view: CudaView<'a, T>,
    host: &ArrayView2<H>,
) -> Result<DevOperand<'a, T>, GpuError> {
    DevOperand::new(view, host, Role::Right)
}

/// Resident LEFT operand that is a sub-block of a larger resident row-major
/// matrix (e.g. a column range, so the leading dimension is wider than the
/// block). `host` is the block as a view into the host twin of the resident
/// matrix (standard-with-padding or its transpose, see
/// `OperandGeom::derive_padded`); `view` starts at the block's first element
/// and must reach at least the block's extent, otherwise `GpuError::Layout`.
/// The descriptor is derived from `host`'s strides, never supplied by the
/// caller, and the `*_dev` GEMMs still check every panel's read extent
/// against `view.len()`.
pub fn dev_left_padded<'a, T, H>(
    view: CudaView<'a, T>,
    host: &ArrayView2<H>,
) -> Result<DevOperand<'a, T>, GpuError> {
    DevOperand::new_padded(view, host, Role::Left)
}

/// Resident RIGHT operand (k × n) that is a sub-block; see [`dev_left_padded`].
pub fn dev_right_padded<'a, T, H>(
    view: CudaView<'a, T>,
    host: &ArrayView2<H>,
) -> Result<DevOperand<'a, T>, GpuError> {
    DevOperand::new_padded(view, host, Role::Right)
}

/// Device-free soundness check for one `*_dev` call with k blocked at `kb`.
pub(crate) fn check_dev_geometry(
    m: usize,
    k: usize,
    n: usize,
    kb: usize,
    left: (&OperandGeom, usize),
    right: (&OperandGeom, usize),
    outputs: &[(&str, usize)],
) -> Result<(), GpuError> {
    if [m, k, n].iter().any(|&d| d > i32::MAX as usize) {
        return Err(GpuError::Layout(format!(
            "dimension exceeds cuBLAS i32 range: {m}x{k}x{n}"
        )));
    }
    // Nothing is read when any dimension is zero, so there is no extent to
    // check (and `m - 1`-style terms would be meaningless); roles, shapes and
    // descriptors are still validated.
    let reads_nothing = m == 0 || k == 0 || n == 0;
    for (name, (g, view_len), role, want) in [
        ("left", left, Role::Left, (m, k)),
        ("right", right, Role::Right, (k, n)),
    ] {
        if g.role != role {
            return Err(GpuError::Layout(format!(
                "{name} slot holds a {:?} operand",
                g.role
            )));
        }
        if (g.rows, g.cols) != want {
            return Err(GpuError::Layout(format!(
                "{name} operand is {}x{} but the product is {m}x{k}x{n}",
                g.rows, g.cols
            )));
        }
        if !g.descriptor_ok() {
            return Err(GpuError::Layout(format!(
                "{name} operand descriptor (ld {}, k_step {}) does not match its role, op and shape",
                g.ld, g.k_step
            )));
        }
        let kb = kb.max(1);
        let mut k0 = 0usize;
        while !reads_nothing && k0 < k {
            let kc = k.min(k0.saturating_add(kb)) - k0;
            let end = g.panel_end(m, n, k0, kc);
            if end > view_len {
                return Err(GpuError::Layout(format!(
                    "{name} operand panel {k0}..{} reads up to element {end} but the view holds {view_len}",
                    k0 + kc,
                )));
            }
            k0 += kc;
        }
    }
    let need = m
        .checked_mul(n)
        .ok_or_else(|| GpuError::Layout(format!("output size {m}x{n} overflows usize")))?;
    for &(name, len) in outputs {
        if len < need {
            return Err(GpuError::Layout(format!(
                "output buffer {name} holds {len} elements, {m}x{n} needs {need}"
            )));
        }
    }
    Ok(())
}

/// The k-blocked f64 loop on resident operands; `c` holds m·n f64 in the
/// column-major (n × m) layout that is row-major (m × n). Numerics unchanged:
/// `gemm_f64` is upload + this + download. Geometry and buffer lengths are
/// checked first (typed `GpuError::Layout`, nothing launched).
pub fn gemm_f64_dev(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    left: &DevOperand<'_, f64>,
    right: &DevOperand<'_, f64>,
    c: &mut CudaSlice<f64>,
    k_block: usize,
) -> Result<(), GpuError> {
    check_dev_geometry(
        m,
        k,
        n,
        k_block,
        (&left.geom, left.view.len()),
        (&right.geom, right.view.len()),
        &[("c", c.len())],
    )?;
    // Empty product: nothing to compute and nothing to write (same as the CPU
    // path and the mixed core). k == 0 with m, n > 0 is the zero matrix.
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        return dev
            .stream
            .memset_zeros(c)
            .map_err(|e| GpuError::Cuda(format!("memset c (k = 0): {e:?}")));
    }
    let (left_op, right_op) = (left.geom.op, right.geom.op);
    let (left_ld, right_ld) = (left.geom.ld, right.geom.ld);
    let (left_ks, right_ks) = (left.geom.k_step, right.geom.k_step);
    let cuda = |e: std::fmt::Arguments| GpuError::Cuda(e.to_string());
    let blas = dev.blas.lock().unwrap_or_else(|e| e.into_inner());
    let kb = k_block.max(1);
    let mut k0 = 0usize;
    while k0 < k {
        let k1 = (k0 + kb).min(k);
        let cfg = GemmConfig {
            transa: right_op,
            transb: left_op,
            m: n as i32,
            n: m as i32,
            k: (k1 - k0) as i32,
            alpha: 1.0,
            lda: right_ld,
            ldb: left_ld,
            beta: if k0 == 0 { 0.0 } else { 1.0 },
            ldc: n as i32,
        };
        let a_view = right.view.slice(k0 * right_ks..);
        let b_view = left.view.slice(k0 * left_ks..);
        // SAFETY: `check_dev_geometry` above proved, for every panel, that the
        // highest element each operand read touches lies inside its view
        // (descriptor derived from the role and shape, extents checked) and that
        // `c` holds AT LEAST m·n elements (it may hold more; cuBLAS writes only
        // m·n of them).
        unsafe { blas.gemm(cfg, &a_view, &b_view, c) }
            .map_err(|e| cuda(format_args!("cublasDgemm k-block {k0}..{k1}: {e:?}")))?;
        k0 = k1;
    }
    Ok(())
}

pub fn gemm_f64(
    dev: &Device,
    pool: &DevicePool,
    left: &ArrayView2<f64>,
    right: &ArrayView2<f64>,
    out: &mut ArrayViewMut2<f64>,
    k_block: usize,
) -> Result<(), GpuError> {
    let (m, k) = (left.nrows(), left.ncols());
    let n = right.ncols();
    if right.nrows() != k || out.nrows() != m || out.ncols() != n {
        return Err(GpuError::Layout(format!(
            "shape mismatch: ({m}x{k})·({}x{n}) -> ({}x{})",
            right.nrows(),
            out.nrows(),
            out.ncols()
        )));
    }
    if !out.is_standard_layout() {
        return Err(GpuError::Layout(
            "output must be row-major contiguous".into(),
        ));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    if [m, k, n].iter().any(|&d| d > i32::MAX as usize) {
        return Err(GpuError::Layout(format!(
            "dimension exceeds cuBLAS i32 range: {m}x{k}x{n}"
        )));
    }
    if k == 0 {
        out.fill(0.0);
        return Ok(());
    }
    if col_major_desc(left, true).is_none() {
        return Err(GpuError::Layout(
            "left operand is neither standard nor transposed-standard".into(),
        ));
    }
    if col_major_desc(right, false).is_none() {
        return Err(GpuError::Layout(
            "right operand is neither standard nor transposed-standard".into(),
        ));
    }
    let la = left
        .as_slice_memory_order()
        .ok_or_else(|| GpuError::Layout("left not contiguous".into()))?;
    let rb = right
        .as_slice_memory_order()
        .ok_or_else(|| GpuError::Layout("right not contiguous".into()))?;

    let bytes = offload_bytes(m, k, n);
    let _lease = pool.reserve("gemm_f64 operands+result", bytes)?;

    let s = &dev.stream;
    let cuda = |e: std::fmt::Arguments| GpuError::Cuda(e.to_string());
    let d_a: CudaSlice<f64> = s
        .clone_htod(la)
        .map_err(|e| cuda(format_args!("H2D left: {e:?}")))?;
    let d_b: CudaSlice<f64> = s
        .clone_htod(rb)
        .map_err(|e| cuda(format_args!("H2D right: {e:?}")))?;
    let mut d_c: CudaSlice<f64> = s
        .alloc_zeros(m * n)
        .map_err(|e| cuda(format_args!("alloc out: {e:?}")))?;

    let left_op = DevOperand::new(d_a.slice(..), left, Role::Left)?;
    let right_op = DevOperand::new(d_b.slice(..), right, Role::Right)?;
    gemm_f64_dev(dev, m, k, n, &left_op, &right_op, &mut d_c, k_block)?;
    let host = out.as_slice_mut().expect("standard layout checked above");
    s.memcpy_dtoh(&d_c, host)
        .map_err(|e| cuda(format_args!("D2H out: {e:?}")))?;
    s.synchronize()
        .map_err(|e| cuda(format_args!("sync: {e:?}")))?;
    stats::note_offloaded(8 * (m * k + k * n), 8 * m * n);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ShapeBuilder};

    // Pure descriptor logic: no device needed, runs in every CI configuration.
    #[test]
    fn descriptors_for_the_four_operand_cases() {
        let a = Array2::<f64>::zeros((5, 7)); // m=5, k=7 standard
        let d = col_major_desc(&a.view(), true).unwrap();
        assert_eq!(
            (d.op, d.ld, d.k_step),
            (cublasOperation_t::CUBLAS_OP_N, 7, 1)
        );
        let b = Array2::<f64>::zeros((7, 3)); // k=7, n=3 standard
        let d = col_major_desc(&b.view(), false).unwrap();
        assert_eq!(
            (d.op, d.ld, d.k_step),
            (cublasOperation_t::CUBLAS_OP_N, 3, 3)
        );
        let at = Array2::<f64>::zeros((7, 5)); // stored k x m, viewed .t() = m x k
        let d = col_major_desc(&at.t(), true).unwrap();
        assert_eq!(
            (d.op, d.ld, d.k_step),
            (cublasOperation_t::CUBLAS_OP_T, 5, 5)
        );
        let bt = Array2::<f64>::zeros((3, 7)); // stored n x k, viewed .t() = k x n
        let d = col_major_desc(&bt.t(), false).unwrap();
        assert_eq!(
            (d.op, d.ld, d.k_step),
            (cublasOperation_t::CUBLAS_OP_T, 7, 1)
        );
    }

    #[test]
    fn interleaved_view_has_no_descriptor() {
        let big = Array2::<f64>::zeros((8, 8));
        assert!(col_major_desc(&big.slice(ndarray::s![..4, ..4]), true).is_none());
    }

    #[test]
    fn negative_stride_view_has_no_descriptor() {
        let a = Array2::<f64>::zeros((8, 8));
        assert!(col_major_desc(&a.slice(ndarray::s![..;-1, ..]), true).is_none());
        assert!(col_major_desc(&a.slice(ndarray::s![.., ..;-1]), false).is_none());
    }

    /// Honest geometries for (m, k, n) in all four layout combinations.
    fn honest(m: usize, k: usize, n: usize) -> Vec<(&'static str, OperandGeom, OperandGeom)> {
        let l_std = Array2::<f64>::zeros((m, k));
        let l_t = Array2::<f64>::zeros((k, m));
        let r_std = Array2::<f64>::zeros((k, n));
        let r_t = Array2::<f64>::zeros((n, k));
        let l = |v: &ArrayView2<f64>| OperandGeom::derive(Role::Left, v, v.len()).unwrap();
        let r = |v: &ArrayView2<f64>| OperandGeom::derive(Role::Right, v, v.len()).unwrap();
        vec![
            ("std,std", l(&l_std.view()), r(&r_std.view())),
            ("T,std", l(&l_t.t()), r(&r_std.view())),
            ("std,T", l(&l_std.view()), r(&r_t.t())),
            ("T,T", l(&l_t.t()), r(&r_t.t())),
        ]
    }

    /// `check_dev_geometry` with each view exactly as long as its host shape.
    fn cdg(
        m: usize,
        k: usize,
        n: usize,
        kb: usize,
        l: &OperandGeom,
        r: &OperandGeom,
        outs: &[(&str, usize)],
    ) -> Result<(), GpuError> {
        check_dev_geometry(
            m,
            k,
            n,
            kb,
            (l, l.rows * l.cols),
            (r, r.rows * r.cols),
            outs,
        )
    }

    const SHAPES: [(usize, usize, usize); 9] = [
        (5, 7, 3),
        (1, 9, 4),
        (4, 9, 1),
        (1, 1, 1),
        (3, 1, 5),
        (4, 3, 6),
        (2, 300, 2),
        (7, 128, 7),
        (6, 129, 5),
    ];
    const PANELS: [usize; 5] = [0, 1, 3, 128, 1000];

    fn is_layout(e: &GpuError, needle: &str) -> bool {
        matches!(e, GpuError::Layout(s) if s.contains(needle))
    }

    #[test]
    fn honest_operands_pass_for_every_layout_shape_and_panel_width() {
        for &(m, k, n) in &SHAPES {
            for (label, l, r) in honest(m, k, n) {
                for kb in PANELS {
                    cdg(m, k, n, kb, &l, &r, &[("c", m * n)])
                        .unwrap_or_else(|e| panic!("{label} {m}x{k}x{n} kb={kb}: {e:?}"));
                }
            }
        }
    }

    #[test]
    fn the_extent_check_is_tight_one_element_short_of_any_operand_is_refused() {
        // The highest element any panel reads is the last element of the view,
        // so a view one element short must be refused for every layout.
        for &(m, k, n) in &SHAPES {
            for (label, l, r) in honest(m, k, n) {
                for kb in PANELS {
                    let (ll, rl) = (l.rows * l.cols, r.rows * r.cols);
                    let e =
                        check_dev_geometry(m, k, n, kb, (&l, ll - 1), (&r, rl), &[]).unwrap_err();
                    assert!(is_layout(&e, "reads up to"), "{label} L {m}x{k}x{n}: {e:?}");
                    let e =
                        check_dev_geometry(m, k, n, kb, (&l, ll), (&r, rl - 1), &[]).unwrap_err();
                    assert!(is_layout(&e, "reads up to"), "{label} R {m}x{k}x{n}: {e:?}");
                }
            }
        }
    }

    #[test]
    fn a_wrong_k_step_is_refused_even_when_it_would_stay_in_bounds() {
        // Right standard with k_step 1 instead of n reads in bounds and computes
        // the wrong product; the descriptor-vs-role check is what refuses it.
        for &(m, k, n) in &SHAPES {
            for (label, l, r) in honest(m, k, n) {
                for (bad_l, bad_r) in [
                    (
                        OperandGeom {
                            k_step: l.k_step + 1,
                            ..l
                        },
                        r,
                    ),
                    (
                        l,
                        OperandGeom {
                            k_step: r.k_step + 1,
                            ..r
                        },
                    ),
                    (
                        OperandGeom {
                            k_step: if l.k_step == 1 { l.cols } else { 1 },
                            ..l
                        },
                        r,
                    ),
                    (
                        l,
                        OperandGeom {
                            k_step: if r.k_step == 1 { r.cols } else { 1 },
                            ..r
                        },
                    ),
                ] {
                    if (bad_l.k_step, bad_r.k_step) == (l.k_step, r.k_step) {
                        continue; // degenerate shape: the swap is the identity
                    }
                    let e = cdg(m, k, n, 128, &bad_l, &bad_r, &[]).unwrap_err();
                    assert!(is_layout(&e, "descriptor"), "{label} {m}x{k}x{n}: {e:?}");
                }
            }
        }
    }

    #[test]
    fn a_left_built_as_right_and_a_right_built_as_left_are_refused() {
        let (m, k, n) = (4usize, 7usize, 3usize);
        for (label, l, r) in honest(m, k, n) {
            // roles swapped between the slots
            let e = cdg(m, k, n, 128, &r, &l, &[]).unwrap_err();
            assert!(is_layout(&e, "slot holds"), "{label}: {e:?}");
            // same role in both slots
            let e = cdg(m, k, n, 128, &l, &l, &[]).unwrap_err();
            assert!(is_layout(&e, "right slot"), "{label}: {e:?}");
        }
        // the host matrix of a left, derived under the Right role: shape check refuses
        let host = Array2::<f64>::zeros((m, k));
        let as_right = OperandGeom::derive(Role::Right, &host.view(), host.len()).unwrap();
        let (_, l, _) = honest(m, k, n).remove(0);
        let e = cdg(m, k, n, 128, &l, &as_right, &[]).unwrap_err();
        assert!(is_layout(&e, "right"), "{e:?}");
    }

    #[test]
    fn shapes_i32_range_and_output_buffers_are_checked_by_name() {
        let (_, l, r) = honest(4, 6, 5).remove(0);
        assert!(cdg(4, 6, 5, 8, &l, &r, &[("c", 20)]).is_ok());
        assert!(cdg(4, 6, 5, 8, &l, &r, &[("c", 99)]).is_ok());
        let e = cdg(4, 6, 5, 8, &l, &r, &[("c32", 20), ("c64", 19)]).unwrap_err();
        assert!(is_layout(&e, "c64"), "{e:?}");
        let e = cdg(4, 7, 5, 8, &l, &r, &[]).unwrap_err();
        assert!(is_layout(&e, "left"), "{e:?}");
        let e = cdg(4, 6, 6, 8, &l, &r, &[]).unwrap_err();
        assert!(is_layout(&e, "right"), "{e:?}");
        let big = i32::MAX as usize + 1;
        let e = cdg(big, 1, 1, 8, &l, &r, &[]).unwrap_err();
        assert!(is_layout(&e, "i32"), "{e:?}");
    }

    #[test]
    fn zero_sized_dimensions_never_panic_and_are_accepted_when_honest() {
        // m = 0, n = 0, k = 0 and all-zero, both roles, both ops (the T variants
        // are built by hand: an empty host array is always standard layout).
        let zero_shapes = [
            (0usize, 5usize, 3usize),
            (4, 5, 0),
            (4, 0, 3),
            (0, 0, 0),
            (0, 0, 5),
            (5, 0, 0),
            (0, 5, 0),
        ];
        for &(m, k, n) in &zero_shapes {
            for kb in PANELS {
                for (label, l, r) in honest(m, k, n) {
                    cdg(m, k, n, kb, &l, &r, &[("c", m * n)])
                        .unwrap_or_else(|e| panic!("{label} {m}x{k}x{n}: {e:?}"));
                }
                for op in [
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_T,
                ] {
                    for role in [Role::Left, Role::Right] {
                        let (rows, cols) = if role == Role::Left { (m, k) } else { (k, n) };
                        let mut g = OperandGeom {
                            role,
                            op,
                            ld: 0,
                            k_step: 0,
                            rows,
                            cols,
                        };
                        let (ld, ks) = g.expected_ld_k_step();
                        g.ld = ld as i32;
                        g.k_step = ks;
                        // the raw extent math is total too
                        for (mm, nn, k0, kc) in [(m, n, 0, 0), (0, 0, 0, 0), (m, n, 3, 1)] {
                            let _ = g.panel_end(mm, nn, k0, kc);
                        }
                    }
                }
            }
        }
        // a zero dimension does not excuse a wrong shape or role
        let (_, l, r) = honest(0, 5, 3).remove(0);
        let e = cdg(0, 6, 3, 8, &l, &r, &[]).unwrap_err();
        assert!(is_layout(&e, "left"), "{e:?}");
        let e = check_dev_geometry(0, 5, 3, 8, (&r, 0), (&l, 0), &[]).unwrap_err();
        assert!(is_layout(&e, "slot holds"), "{e:?}");
        // ... and output buffers are still sized against m*n
        let (_, l, r) = honest(2, 0, 3).remove(0);
        let e = cdg(2, 0, 3, 8, &l, &r, &[("c", 5)]).unwrap_err();
        assert!(is_layout(&e, "c holds 5"), "{e:?}");
        assert!(cdg(2, 0, 3, 8, &l, &r, &[("c", 6)]).is_ok());
    }

    #[test]
    fn a_view_of_the_wrong_length_never_derives_a_geometry() {
        let host = Array2::<f64>::zeros((3, 4));
        for len in [0usize, 11, 13] {
            let e = OperandGeom::derive(Role::Left, &host.view(), len).unwrap_err();
            assert!(is_layout(&e, "elements"), "{e:?}");
        }
    }

    // ---- padded (sub-block of a larger row-major matrix) operands ----------

    /// The RI-MP2 shapes: B is (naux x nov) row-major, nvir-wide column blocks.
    /// Returns the geometries of G_i = B_iᵀ·B_tail for block `i`.
    fn rimp2_geoms(
        b: &Array2<f64>,
        nvir: usize,
        i: usize,
        len: usize,
    ) -> Result<(OperandGeom, OperandGeom), GpuError> {
        let b_i = b.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
        let b_tail = b.slice(ndarray::s![.., i * nvir..]);
        Ok((
            OperandGeom::derive_padded(Role::Left, &b_i.t(), len)?,
            OperandGeom::derive_padded(Role::Right, &b_tail, len)?,
        ))
    }

    #[test]
    fn padded_rimp2_blocks_get_the_wide_leading_dimension() {
        let (naux, nov, nvir, i) = (7usize, 12usize, 3usize, 1usize);
        let b = Array2::<f64>::zeros((naux, nov));
        let off = i * nvir;
        let (l, r) = rimp2_geoms(&b, nvir, i, naux * nov - off).unwrap();
        assert_eq!(
            (l.op, l.ld, l.k_step, l.rows, l.cols),
            (cublasOperation_t::CUBLAS_OP_T, 12, 12, nvir, naux)
        );
        assert_eq!(
            (r.op, r.ld, r.k_step, r.rows, r.cols),
            (cublasOperation_t::CUBLAS_OP_N, 12, 12, naux, nov - off)
        );
        let (m, k, n) = (nvir, naux, nov - off);
        // a view from the block's first element to the end of B passes, for every panel width
        for kb in PANELS {
            let len = naux * nov - off;
            check_dev_geometry(m, k, n, kb, (&l, len), (&r, len), &[("c", m * n)]).unwrap();
        }
        // tight: the last element read is (k-1)·ld + width of the block on the
        // left (the tail on the right reaches the end of B)
        let last_left = (naux - 1) * nov + nvir;
        let e = check_dev_geometry(
            m,
            k,
            n,
            128,
            (&l, last_left - 1),
            (&r, naux * nov - off),
            &[],
        )
        .unwrap_err();
        assert!(is_layout(&e, "reads up to"), "{e:?}");
        check_dev_geometry(m, k, n, 128, (&l, last_left), (&r, naux * nov - off), &[]).unwrap();
        let last_right = (naux - 1) * nov + n;
        let e = check_dev_geometry(m, k, n, 128, (&l, last_left), (&r, last_right - 1), &[])
            .unwrap_err();
        assert!(is_layout(&e, "reads up to"), "{e:?}");
        check_dev_geometry(m, k, n, 128, (&l, last_left), (&r, last_right), &[]).unwrap();
    }

    #[test]
    fn padded_derivation_agrees_with_the_exact_one_on_unpadded_layouts() {
        for &(m, k, n) in &SHAPES {
            let l_std = Array2::<f64>::zeros((m, k));
            let l_t = Array2::<f64>::zeros((k, m));
            let r_std = Array2::<f64>::zeros((k, n));
            let r_t = Array2::<f64>::zeros((n, k));
            for (role, v) in [
                (Role::Left, l_std.view()),
                (Role::Left, l_t.t()),
                (Role::Right, r_std.view()),
                (Role::Right, r_t.t()),
            ] {
                let exact = OperandGeom::derive(role, &v, v.len()).unwrap();
                let padded = OperandGeom::derive_padded(role, &v, v.len()).unwrap();
                // A degenerate axis of length 1 admits two valid descriptions;
                // both must read the same elements, so compare the cuBLAS
                // read extents rather than the labels.
                assert_eq!(
                    exact.panel_end(m, n, 0, k),
                    padded.panel_end(m, n, 0, k),
                    "{role:?} {m}x{k}x{n}"
                );
                if v.nrows() > 1 && v.ncols() > 1 {
                    assert_eq!(exact, padded, "{role:?} {m}x{k}x{n}");
                }
            }
        }
    }

    #[test]
    fn padded_derivation_refuses_what_it_cannot_describe() {
        let data = vec![0.0f64; 64];
        let view = |shape: (usize, usize), strides: (usize, usize)| {
            ndarray::ArrayView2::from_shape(shape.strides(strides), &data[..])
        };
        // overlapping rows (stride < width): cuBLAS would read the wrong elements
        let v = view((3, 4), (2, 1)).unwrap();
        assert!(OperandGeom::derive_padded(Role::Left, &v, 64).is_err());
        // neither axis has unit stride
        let v = view((3, 4), (8, 2)).unwrap();
        assert!(OperandGeom::derive_padded(Role::Right, &v, 64).is_err());
        // negative stride (reversed rows)
        let a = Array2::<f64>::zeros((8, 8));
        let v = a.slice(ndarray::s![..;-1, ..4]);
        assert!(OperandGeom::derive_padded(Role::Left, &v, 64).is_err());
        // view shorter than the host extent: 3 rows of 4 at ld 10 need 24 elements
        let v = view((3, 4), (10, 1)).unwrap();
        assert!(OperandGeom::derive_padded(Role::Left, &v, 23).is_err());
        assert!(OperandGeom::derive_padded(Role::Left, &v, 24).is_ok());
        // ld beyond the cuBLAS i32 range (u8 elements: the zeroed 2 GiB is
        // lazily mapped and never touched)
        let ld = i32::MAX as usize + 1;
        let bytes = vec![0u8; ld + 1];
        let v = ndarray::ArrayView2::from_shape((2, 1).strides((ld, 1)), &bytes[..]).unwrap();
        let e = OperandGeom::derive_padded(Role::Left, &v, ld + 1).unwrap_err();
        assert!(is_layout(&e, "i32"), "{e:?}");
        // the stride of a length-1 axis is never stepped, so it cannot refuse a
        // view: a (1, 1) block with a huge or zero stride is an ordinary 1x1
        let big = [0.0f64; 1];
        for st in [(0usize, 1usize), (i32::MAX as usize + 1, 1), (1, 0)] {
            let v = ndarray::ArrayView2::from_shape((1, 1).strides(st), &big[..]).unwrap();
            let g = OperandGeom::derive_padded(Role::Left, &v, 1).unwrap();
            assert_eq!((g.ld, g.k_step, g.rows, g.cols), (1, 1, 1, 1), "{st:?}");
        }
    }

    #[test]
    fn padded_derivation_is_total_on_zero_sized_views() {
        let data = [0.0f64; 8];
        for shape in [(0usize, 5usize), (5, 0), (0, 0)] {
            for strides in [(5usize, 1usize), (1, 5), (0, 0), (usize::MAX / 4, 1)] {
                let v = ndarray::ArrayView2::from_shape(shape.strides(strides), &data[..]);
                if let Ok(v) = v {
                    for role in [Role::Left, Role::Right] {
                        let _ = OperandGeom::derive_padded(role, &v, 0);
                        let _ = OperandGeom::derive_padded(role, &v, 8);
                    }
                }
            }
        }
    }

    #[test]
    fn a_padded_descriptor_must_still_match_its_role_op_and_shape() {
        let b = Array2::<f64>::zeros((7, 12));
        let (l, r) = rimp2_geoms(&b, 3, 1, 7 * 12 - 3).unwrap();
        let (m, k, n) = (3usize, 7usize, 9usize);
        let len = 7 * 12 - 3;
        let ok = |l: &OperandGeom, r: &OperandGeom| {
            check_dev_geometry(m, k, n, 128, (l, len), (r, len), &[])
        };
        ok(&l, &r).unwrap();
        // ld below the stored matrix's row count
        let e = ok(&OperandGeom { ld: 2, ..l }, &r).unwrap_err();
        assert!(is_layout(&e, "descriptor"), "{e:?}");
        // k_step that disagrees with ld (reads in bounds, computes the wrong product)
        let e = ok(&OperandGeom { k_step: 3, ..l }, &r).unwrap_err();
        assert!(is_layout(&e, "descriptor"), "{e:?}");
        let e = ok(&l, &OperandGeom { k_step: 9, ..r }).unwrap_err();
        assert!(is_layout(&e, "descriptor"), "{e:?}");
        // a bigger ld with the matching k_step is a different, still well-formed, operand
        // that simply needs a longer view
        let wide_l = OperandGeom {
            ld: 20,
            k_step: 20,
            ..l
        };
        let e = ok(&wide_l, &r).unwrap_err();
        assert!(is_layout(&e, "reads up to"), "{e:?}");
    }

    /// One past the highest element a block view touches, counted from its
    /// first element (all strides non-negative here).
    fn block_extent(v: &ArrayView2<f64>) -> usize {
        if v.is_empty() {
            return 0;
        }
        let s = v.strides();
        (v.nrows() - 1) * s[0] as usize + (v.ncols() - 1) * s[1] as usize + 1
    }

    /// (label, left geometry, left extent, right geometry, right extent).
    type PaddedCase = (&'static str, OperandGeom, usize, OperandGeom, usize);

    /// Padded blocks (leading dimension `pad` elements wider than the block)
    /// for all four layout cases of an (m, k, n) product: the left as a
    /// standard block (Left-N) or a transposed block (Left-T), the right as a
    /// standard block (Right-N) or a transposed block (Right-T).
    fn padded_cases(m: usize, k: usize, n: usize, pad: usize) -> Vec<PaddedCase> {
        let l_std = Array2::<f64>::zeros((m, k + pad));
        let l_t = Array2::<f64>::zeros((k, m + pad));
        let r_std = Array2::<f64>::zeros((k, n + pad));
        let r_t = Array2::<f64>::zeros((n, k + pad));
        let lv_n = l_std.slice(ndarray::s![.., ..k]);
        let lv_t = l_t.slice(ndarray::s![.., ..m]);
        let rv_n = r_std.slice(ndarray::s![.., ..n]);
        let rv_t = r_t.slice(ndarray::s![.., ..k]);
        let left = |v: &ArrayView2<f64>| {
            let e = block_extent(v);
            (OperandGeom::derive_padded(Role::Left, v, e).unwrap(), e)
        };
        let right = |v: &ArrayView2<f64>| {
            let e = block_extent(v);
            (OperandGeom::derive_padded(Role::Right, v, e).unwrap(), e)
        };
        let (a, b) = (left(&lv_n), left(&lv_t.t()));
        let (c, d) = (right(&rv_n), right(&rv_t.t()));
        vec![
            ("Left-N,Right-N", a.0, a.1, c.0, c.1),
            ("Left-N,Right-T", a.0, a.1, d.0, d.1),
            ("Left-T,Right-N", b.0, b.1, c.0, c.1),
            ("Left-T,Right-T", b.0, b.1, d.0, d.1),
        ]
    }

    #[test]
    fn padded_blocks_of_every_layout_have_ld_above_the_minimum_and_pass_at_their_extent() {
        for &(m, k, n) in &SHAPES {
            for pad in [1usize, 5] {
                for (label, l, ll, r, rl) in padded_cases(m, k, n, pad) {
                    for kb in PANELS {
                        check_dev_geometry(m, k, n, kb, (&l, ll), (&r, rl), &[("c", m * n)])
                            .unwrap_or_else(|e| {
                                panic!("{label} {m}x{k}x{n} pad {pad} kb {kb}: {e:?}")
                            });
                    }
                    for g in [l, r] {
                        assert!(
                            g.ld as usize >= g.expected_ld_k_step().0,
                            "{label} {m}x{k}x{n}"
                        );
                    }
                }
            }
        }
        // all four layouts of one shape are strictly padded (ld > minimum)
        for (label, l, _, r, _) in padded_cases(4, 7, 5, 3) {
            for g in [l, r] {
                assert!(g.ld as usize > g.expected_ld_k_step().0, "{label}: {g:?}");
            }
        }
    }

    #[test]
    fn padded_extent_is_tight_in_both_directions_with_a_last_panel_past_zero() {
        // One element short of either operand is refused for every layout and
        // every panel width; kb in {1, 3} make the LAST panel start at
        // k0 > 0 (k >= 7 for the shapes that reach it), so the refusal there
        // comes from a later panel's read extent, not only the first.
        for &(m, k, n) in &SHAPES {
            for pad in [1usize, 5] {
                for (label, l, ll, r, rl) in padded_cases(m, k, n, pad) {
                    for kb in PANELS {
                        let ctx = format!("{label} {m}x{k}x{n} pad {pad} kb {kb}");
                        let e = check_dev_geometry(m, k, n, kb, (&l, ll - 1), (&r, rl), &[])
                            .unwrap_err();
                        assert!(is_layout(&e, "reads up to"), "L {ctx}: {e:?}");
                        let e = check_dev_geometry(m, k, n, kb, (&l, ll), (&r, rl - 1), &[])
                            .unwrap_err();
                        assert!(is_layout(&e, "reads up to"), "R {ctx}: {e:?}");
                        check_dev_geometry(m, k, n, kb, (&l, ll), (&r, rl), &[]).unwrap();
                    }
                }
            }
        }
        // the refusing panel really is the last one: with kb = 1 and kb = 3 the
        // message names a panel starting past 0 (the extent is reached only by
        // the last panel)
        let (m, k, n) = (4usize, 7usize, 5usize);
        for kb in [1usize, 3] {
            let last = (k - 1) / kb * kb;
            assert!(last > 0);
            for (label, l, ll, r, rl) in padded_cases(m, k, n, 3) {
                for (lv, rv) in [(ll - 1, rl), (ll, rl - 1)] {
                    let e = check_dev_geometry(m, k, n, kb, (&l, lv), (&r, rv), &[]).unwrap_err();
                    let GpuError::Layout(msg) = e else {
                        panic!("{label}")
                    };
                    assert!(
                        msg.contains(&format!("panel {last}..{k}")),
                        "{label} kb {kb}: refusal should come from the last panel: {msg}"
                    );
                }
            }
        }
    }

    #[test]
    fn padded_operands_with_k_equal_one_and_wrong_slots_are_handled() {
        // k = 1: one panel, k_step never multiplies a non-zero k0
        for (m, n) in [(1usize, 1usize), (4, 1), (1, 6), (3, 5)] {
            for pad in [1usize, 4] {
                for (label, l, ll, r, rl) in padded_cases(m, 1, n, pad) {
                    for kb in PANELS {
                        check_dev_geometry(m, 1, n, kb, (&l, ll), (&r, rl), &[("c", m * n)])
                            .unwrap_or_else(|e| panic!("{label} {m}x1x{n} kb {kb}: {e:?}"));
                        let e = check_dev_geometry(m, 1, n, kb, (&l, ll - 1), (&r, rl), &[])
                            .unwrap_err();
                        assert!(is_layout(&e, "reads up to"), "{label} {m}x1x{n}: {e:?}");
                    }
                }
            }
        }
        // wrong operand in a slot, and a padded operand of the wrong shape
        let (m, k, n) = (4usize, 7usize, 5usize);
        for (label, l, ll, r, rl) in padded_cases(m, k, n, 3) {
            let e = check_dev_geometry(m, k, n, 128, (&r, rl), (&l, ll), &[]).unwrap_err();
            assert!(is_layout(&e, "slot holds"), "{label}: {e:?}");
            let e = check_dev_geometry(m, k, n, 128, (&l, ll), (&l, ll), &[]).unwrap_err();
            assert!(is_layout(&e, "right slot"), "{label}: {e:?}");
            let e = check_dev_geometry(m + 1, k, n, 128, (&l, ll), (&r, rl), &[]).unwrap_err();
            assert!(is_layout(&e, "left operand is"), "{label}: {e:?}");
            let e = check_dev_geometry(m, k, n + 1, 128, (&l, ll), (&r, rl), &[]).unwrap_err();
            assert!(is_layout(&e, "right operand is"), "{label}: {e:?}");
        }
        // a left block built under the Right role sits in the right slot with
        // the wrong shape: refused by the shape check
        let big = Array2::<f64>::zeros((m, k + 3));
        let v = big.slice(ndarray::s![.., ..k]);
        let as_right = OperandGeom::derive_padded(Role::Right, &v, block_extent(&v)).unwrap();
        let (_, l, ll, _, _) = padded_cases(m, k, n, 3).remove(0);
        let e = check_dev_geometry(m, k, n, 128, (&l, ll), (&as_right, 1 << 20), &[]).unwrap_err();
        assert!(is_layout(&e, "right operand is"), "{e:?}");
    }

    #[test]
    fn offload_bytes_is_eight_times_the_three_operands() {
        assert_eq!(offload_bytes(2, 3, 5), 8 * (6 + 15 + 10));
    }
}
