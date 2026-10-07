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
/// parameter: [`dev_left`] / [`dev_right`] are the only public constructors.
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
    /// Elements in the device view.
    pub len: usize,
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
            len,
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

    /// One past the last element cuBLAS reads for the panel [k0, k0 + kc), in
    /// the swapped column-major call (right is cuBLAS A with m_c = n, left is
    /// cuBLAS B with n_c = m).
    fn panel_end(&self, m: usize, n: usize, k0: usize, kc: usize) -> usize {
        let ld = self.ld as usize;
        let n_op = self.op == cublasOperation_t::CUBLAS_OP_N;
        let extent = match (self.role, n_op) {
            (Role::Right, true) => ld * (kc - 1) + n,
            (Role::Right, false) => ld * (n - 1) + kc,
            (Role::Left, true) => ld * (m - 1) + kc,
            (Role::Left, false) => ld * (kc - 1) + m,
        };
        k0 * self.k_step + extent
    }
}

/// One operand already on the device, described the way cuBLAS reads it after
/// the row-major → column-major swap (`outᵀ = rightᵀ·leftᵀ`): `view` starts at
/// the operand's first element, `ld` is its leading dimension in the stored
/// column-major matrix, `op` says whether cuBLAS transposes it, and `k_step`
/// is the element offset per unit of k (see `col_major_desc`).
///
/// What is guaranteed: the fields are crate-private; the only public
/// constructors are [`dev_left`] and [`dev_right`], which fix the role and
/// derive (op, ld, k_step) from the host twin's layout (the caller supplies no
/// descriptor value) and require `view.len() == host.len()`. Before any cuBLAS
/// call, the `*_dev` GEMMs run `check_dev_geometry`: operand roles and shapes
/// against `(m, k, n)`, (ld, k_step) against the unique values of that role,
/// op and shape, and the highest element any panel reads (`k0·k_step` plus the
/// cuBLAS read extent, for EVERY panel) against the view length; output buffers
/// against m·n. Violations are typed `GpuError::Layout`, nothing is launched.
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

/// Device-free soundness check for one `*_dev` call with k blocked at `kb`.
pub(crate) fn check_dev_geometry(
    m: usize,
    k: usize,
    n: usize,
    kb: usize,
    left: &OperandGeom,
    right: &OperandGeom,
    outputs: &[(&str, usize)],
) -> Result<(), GpuError> {
    if [m, k, n].iter().any(|&d| d > i32::MAX as usize) {
        return Err(GpuError::Layout(format!(
            "dimension exceeds cuBLAS i32 range: {m}x{k}x{n}"
        )));
    }
    for (name, g, role, want) in [
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
        if (g.ld as usize, g.k_step) != g.expected_ld_k_step() {
            return Err(GpuError::Layout(format!(
                "{name} operand descriptor (ld {}, k_step {}) does not match its role, op and shape",
                g.ld, g.k_step
            )));
        }
        let kb = kb.max(1);
        let mut k0 = 0usize;
        while k0 < k {
            let kc = (k0 + kb).min(k) - k0;
            let end = g.panel_end(m, n, k0, kc);
            if end > g.len {
                return Err(GpuError::Layout(format!(
                    "{name} operand panel {k0}..{} reads up to element {end} but the view holds {}",
                    k0 + kc,
                    g.len
                )));
            }
            k0 += kc;
        }
    }
    for &(name, len) in outputs {
        if len < m * n {
            return Err(GpuError::Layout(format!(
                "output buffer {name} holds {len} elements, {m}x{n} needs {}",
                m * n
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
    check_dev_geometry(m, k, n, k_block, &left.geom, &right.geom, &[("c", c.len())])?;
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
        // SAFETY: shapes and leading dimensions come from the descriptors and
        // the buffers hold exactly those elements.
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
    use ndarray::Array2;

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
                    check_dev_geometry(m, k, n, kb, &l, &r, &[("c", m * n)])
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
                    let short_l = OperandGeom {
                        len: l.len - 1,
                        ..l
                    };
                    let e = check_dev_geometry(m, k, n, kb, &short_l, &r, &[]).unwrap_err();
                    assert!(is_layout(&e, "reads up to"), "{label} L {m}x{k}x{n}: {e:?}");
                    let short_r = OperandGeom {
                        len: r.len - 1,
                        ..r
                    };
                    let e = check_dev_geometry(m, k, n, kb, &l, &short_r, &[]).unwrap_err();
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
                    let e = check_dev_geometry(m, k, n, 128, &bad_l, &bad_r, &[]).unwrap_err();
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
            let e = check_dev_geometry(m, k, n, 128, &r, &l, &[]).unwrap_err();
            assert!(is_layout(&e, "slot holds"), "{label}: {e:?}");
            // same role in both slots
            let e = check_dev_geometry(m, k, n, 128, &l, &l, &[]).unwrap_err();
            assert!(is_layout(&e, "right slot"), "{label}: {e:?}");
        }
        // the host matrix of a left, derived under the Right role: shape check refuses
        let host = Array2::<f64>::zeros((m, k));
        let as_right = OperandGeom::derive(Role::Right, &host.view(), host.len()).unwrap();
        let (_, l, _) = honest(m, k, n).remove(0);
        let e = check_dev_geometry(m, k, n, 128, &l, &as_right, &[]).unwrap_err();
        assert!(is_layout(&e, "right"), "{e:?}");
    }

    #[test]
    fn shapes_i32_range_and_output_buffers_are_checked_by_name() {
        let (_, l, r) = honest(4, 6, 5).remove(0);
        assert!(check_dev_geometry(4, 6, 5, 8, &l, &r, &[("c", 20)]).is_ok());
        assert!(check_dev_geometry(4, 6, 5, 8, &l, &r, &[("c", 99)]).is_ok());
        let e = check_dev_geometry(4, 6, 5, 8, &l, &r, &[("c32", 20), ("c64", 19)]).unwrap_err();
        assert!(is_layout(&e, "c64"), "{e:?}");
        let e = check_dev_geometry(4, 7, 5, 8, &l, &r, &[]).unwrap_err();
        assert!(is_layout(&e, "left"), "{e:?}");
        let e = check_dev_geometry(4, 6, 6, 8, &l, &r, &[]).unwrap_err();
        assert!(is_layout(&e, "right"), "{e:?}");
        let big = i32::MAX as usize + 1;
        let e = check_dev_geometry(big, 1, 1, 8, &l, &r, &[]).unwrap_err();
        assert!(is_layout(&e, "i32"), "{e:?}");
    }

    #[test]
    fn a_view_of_the_wrong_length_never_derives_a_geometry() {
        let host = Array2::<f64>::zeros((3, 4));
        for len in [0usize, 11, 13] {
            let e = OperandGeom::derive(Role::Left, &host.view(), len).unwrap_err();
            assert!(is_layout(&e, "elements"), "{e:?}");
        }
    }

    #[test]
    fn offload_bytes_is_eight_times_the_three_operands() {
        assert_eq!(offload_bytes(2, 3, 5), 8 * (6 + 15 + 10));
    }
}
