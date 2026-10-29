//! Batched, basis-function-screened semilocal XC integration — the SCF path
//! behind [`crate::ks::KsXc`] / [`crate::ks::KsXcUks`].
//!
//! # Why this exists
//!
//! The dense path ([`crate::density_on_grid::eval_density_closed`] +
//! [`crate::vxc::semilocal_vxc_closed_scratch`]) holds χ and ∇χ as full
//! `(nbf, npts)` planes over the WHOLE grid and runs five `(nbf × npts × nbf)`
//! GEMMs per GGA iteration, on one BLAS thread, with no screening. Measured on
//! danuglipron (71 atoms, nbf 235, PBE/STO-3G, 585,750 points): XC was 88.7% of
//! SCF wall time at 49 s/iteration, one busy thread, 4.4 GB of χ/∇χ. A GTO
//! decays like `exp(-α r²)`, so over a spatially compact block of points most
//! basis functions are numerically zero: a numpy/PySCF prototype on the same
//! grid measured only 33% of the functions significant per 128-point block at
//! a 1e-10 cutoff (12% of the dense GEMM work, since the GEMMs are quadratic
//! in the active count).
//!
//! # Design
//!
//! 1. **Spatial batches** (`partition_points`). Becke points arrive atom by
//!    atom and shell by shell; a large-r angular shell is not compact. The grid
//!    is recursively bisected along the longest bounding-box axis into batches
//!    of at most `max_batch_pts` points. Each batch stores the ORIGINAL point
//!    indices, so weights and every per-point array stay in the grid's own
//!    order — the grid itself is never permuted.
//! 2. **Shell screening** (`screen_batch`). A shell is kept for a batch iff
//!    `max` over the batch's points and the shell's functions of
//!    `max(|χ|, |∂xχ|, |∂yχ|, |∂zχ|)` exceeds `screen_thresh`. The test uses
//!    the ACTUAL values (one full evaluation at construction), not an analytic
//!    bound, so there is no bound to get wrong: a dropped shell provably has
//!    every value and gradient component at or below the threshold on every
//!    point of that batch. `screen_thresh <= 0` keeps every shell — the
//!    exactness anchor.
//! 3. **Two passes per Fock build, parallel over batches.**
//!    * Pass 1 (density): per batch, `Φ = D_sub · [χ | ∂xχ | ∂yχ | ∂zχ]`
//!      (one GEMM; only the χ block unless meta-GGA), then ρ, ∇ρ, τ as
//!      point-local row reductions. Results are written back to the ORIGINAL
//!      point slots — disjoint per batch, so there is no reduction here.
//!    * The libxc kernel runs ONCE over the whole grid through
//!      `crate::vxc::closed_kernel` / `crate::vxc::polarized_kernel` —
//!      the very functions the dense path calls, so the two paths cannot drift.
//!      (Per-batch libxc calls from rayon workers would need per-worker libxc
//!      handles; the global call already parallelizes safely and is
//!      bit-identical to a serial call — see `libxc.rs::eval_chunked`.)
//!    * `E_xc` uses the dense path's own `deterministic_point_sum` over the
//!      original point order.
//!    * Pass 2 (potential): per batch, ONE GEMM `V_b = A · Bᵀ` with
//!      `B = [χ | ∂xχ | ∂yχ | ∂zχ]` and
//!      `A = [½sχ + Σ_a f_a ∂_aχ | ½t ∂xχ | ½t ∂yχ | ½t ∂zχ]`
//!      (`s = w v_ρ`, `f_a = 2 w v_σ ∇_aρ`, `t = ½ w v_τ`; the τ blocks only
//!      for meta-GGA), then `V_sub += V_b + V_bᵀ`. That single GEMM replaces
//!      the dense path's LDA GEMM + three per-axis GGA GEMMs (+ three τ GEMMs),
//!      and `V_b + V_bᵀ` reproduces the dense `½(V + Vᵀ)` exactly: the LDA
//!      and τ pieces are symmetric (so doubling ½ gives 1) and the GGA piece is
//!      `Σ_a M_aᵀ + M_a`.
//!
//! # Determinism (hard requirement)
//!
//! Every result is bit-identical across rayon thread counts:
//! * batch boundaries are a pure function of the point coordinates and
//!   `max_batch_pts` (total-order comparator with an index tie-break);
//! * inside a batch everything is serial (the AO evaluator here is serial,
//!   GEMMs run on the calling worker, and OpenBLAS is pinned to 1 thread for
//!   the whole Fock build before any worker starts). The one exception is the
//!   one-thread-pool serial mode (`Exec`), which is bit-identical to the
//!   parallel mode under the default `FERRIC_BLAS_THREADS`;
//! * pass-1 outputs are disjoint slots; the libxc call is chunk-bit-identical;
//!   `E_xc` uses the fixed-group `deterministic_point_sum`;
//! * pass-2 contributions are reduced in a FIXED grouping: batches are split
//!   into `n_groups(nbatches, nbf, nmats)` contiguous groups (a pure function
//!   of the problem shape, never of the thread count), each group accumulates
//!   its batches serially in ascending order into its own dense matrix, and
//!   the group matrices are summed in ascending group order. No per-thread
//!   accumulator exists anywhere.
//!
//! # Resident vs recompute
//!
//! Per-batch compact AO arrays `(nact, 4·nb)` are either held resident (the
//! default when they fit the memory budget) or re-evaluated each Fock build.
//! Recompute mode evaluates each batch's AO block TWICE per Fock build — once
//! for pass 1 and once for pass 2, because the global libxc call sits between
//! them — trading ~2 serial-per-batch AO evaluations for the resident memory.
//! Both modes call the same serial evaluator on the same (shell, point) pairs,
//! and every downstream operation sees identically-shaped inputs, so the two
//! modes are bit-identical by construction — the storage decision can move
//! memory and time, never an energy.

use std::borrow::Cow;

use ndarray::{s, Array1, Array2, ArrayView2};
use rayon::prelude::*;

use ferric_integrals::blas_threads::with_blas_threads;

use crate::ao_grid::{eval_shell_and_grad, GtoEvalError, LocatedShell};
use crate::density_on_grid::{DensityGrid, UksDensityGrid};
use crate::grid::GridPoint;
use crate::libxc::{FunctionalFamily, XcDef};
use crate::vxc::{closed_kernel, deterministic_point_sum, polarized_kernel, DENSITY_FLOOR};

/// Device offload of the per-batch GEMMs (feature `gpu`).
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::xc_grid_routes_to_device;
#[cfg(feature = "test-seams")]
pub use gpu::{FORCE_HOST, MIXED_VXC_HOST_TWIN, XC_GEMM_LOOP};

/// True when this process asked for V_xc in mixed precision and could run it:
/// `[gpu] mode = "on"`, `precision = "mixed"` with `xc-vxc` named in
/// `mixed_kernels` (it is opt-in, not in the default set), and a ready device
/// (or, in tests, the `MIXED_VXC_HOST_TWIN` seam). Always false without the
/// `gpu` feature.
pub(crate) fn mixed_vxc_requested() -> bool {
    #[cfg(feature = "gpu")]
    {
        gpu::mixed_vxc_requested()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// What one integration holds of the device: the locked device slot (`gpu`
/// builds; `None` when this integration runs on the CPU), or nothing.
#[cfg(feature = "gpu")]
type DevSession<'a> = Option<std::sync::MutexGuard<'a, gpu::DeviceSlot>>;
#[cfg(not(feature = "gpu"))]
type DevSession<'a> = std::marker::PhantomData<&'a ()>;

/// Default per-batch shell-screening threshold on `max(|χ|, |∇χ|)`.
///
/// Chosen by measurement (numpy/PySCF prototype of exactly this algorithm,
/// PBE, 75×110 Becke grid, energies at a FIXED density against the unscreened
/// batched sum):
///
/// ```text
///   system / basis              thresh  active  GEMM work  |dE_xc|   max|dV|
///   water / cc-pVDZ             1e-10   1.000   1.000      0         0
///   benzene / cc-pVDZ           1e-10   0.840   0.724      1.1e-13   2.8e-12
///   benzene / cc-pVDZ           1e-8    0.784   0.639      1.4e-10   1.5e-10
///   danuglipron / STO-3G (235)  1e-12   0.380   0.156      <1e-15    7.3e-14
///   danuglipron / STO-3G (235)  1e-10   0.331   0.120      4.0e-13   9.1e-12
///   danuglipron / STO-3G (235)  1e-8    0.277   0.086      4.1e-11   1.1e-9
/// ```
///
/// (danuglipron with a minao guess density, 128-point median batches.) At
/// 1e-10 the energy error is ~1e-13-1e-12 Ha, three or more orders below the
/// 1e-9 Ha acceptance bar and far below SCF convergence, while 1e-8 buys only
/// ~30% more GEMM savings for a 100× larger error. The error grows with
/// system size and basis diffuseness, which is why the default is not pushed
/// to the edge.
pub const DEFAULT_SCREEN_THRESH: f64 = 1e-10;

/// Default maximum points per spatial batch. The prototype measured a small
/// effect between 128 and 256 (active fraction 0.331 vs 0.341 on danuglipron);
/// 128 keeps GEMM panels comfortably cache-resident.
pub const DEFAULT_MAX_BATCH_PTS: usize = 128;

/// Target number of fixed reduction groups for the pass-2 V accumulation.
/// Enough for load balance on the box's cores; the actual count is a pure
/// function of the problem shape (see [`n_groups`]).
const TARGET_GROUPS: usize = 64;

/// Cap on the memory held by the per-group dense accumulators; the group count
/// shrinks (never below 1) for very large `nbf` to respect it.
const ACCUM_CAP_BYTES: usize = 512 << 20;

/// Where the per-batch compact AO arrays live between Fock builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AoStorage {
    /// Resident when they fit the memory budget, else recompute (default).
    Auto,
    /// Force resident (tests; still charged against a pool when installed).
    Resident,
    /// Force per-iteration recomputation.
    Recompute,
}

/// Internal knobs of the batched XC path. `Default` is the production setting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcBatchConfig {
    /// Shell kept iff `max(|χ|, |∇χ|)` over a batch exceeds this. `0.0` (or any
    /// value `<= 0`) keeps every shell: the exactness anchor.
    pub screen_thresh: f64,
    /// Maximum points per spatial batch (>= 1).
    pub max_batch_pts: usize,
    /// Resident / recompute policy for the compact AO arrays.
    pub storage: AoStorage,
}

impl Default for XcBatchConfig {
    fn default() -> Self {
        Self {
            screen_thresh: DEFAULT_SCREEN_THRESH,
            max_batch_pts: DEFAULT_MAX_BATCH_PTS,
            storage: AoStorage::Auto,
        }
    }
}

/// Summary of a screened grid, for tests and benchmarks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcScreeningStats {
    pub nbatches: usize,
    pub npts: usize,
    pub nbf: usize,
    /// Σ_b nact_b·nb_b / (nbf·npts): fraction of (function, point) pairs kept.
    pub active_fraction: f64,
    /// Σ_b nact_b²·nb_b / (nbf²·npts): fraction of the dense GEMM work kept.
    pub gemm_fraction: f64,
    /// True when the compact AO arrays are resident.
    pub resident: bool,
}

/// Owned copy of one located shell. [`LocatedShell`] borrows the `BasisSet`,
/// which a long-lived `KsXc` cannot hold alongside itself.
#[derive(Debug, Clone)]
pub(crate) struct OwnedShell {
    l: i32,
    pure: bool,
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
    center: [f64; 3],
    /// Number of basis functions in this shell.
    nfunc: usize,
    /// Global index of this shell's first basis function.
    offset: usize,
}

impl OwnedShell {
    fn located(&self) -> LocatedShell<'_> {
        LocatedShell {
            l: self.l,
            pure: self.pure,
            exponents: &self.exponents,
            coefficients: &self.coefficients,
            center: self.center,
        }
    }
}

/// Owned shells in `collect_shells` order, with global function offsets.
pub(crate) fn owned_shells(shells: &[LocatedShell]) -> Vec<OwnedShell> {
    let mut off = 0usize;
    shells
        .iter()
        .map(|sh| {
            let nfunc = ferric_core::basis::num_functions(sh.l, sh.pure);
            let o = OwnedShell {
                l: sh.l,
                pure: sh.pure,
                exponents: sh.exponents.to_vec(),
                coefficients: sh.coefficients.to_vec(),
                center: sh.center,
                nfunc,
                offset: off,
            };
            off += nfunc;
            o
        })
        .collect()
}

/// Partition point indices `0..xyz.len()` into spatially compact batches of at
/// most `max_pts` points each.
///
/// Recursive bisection along the longest bounding-box axis. The split index is
/// `max_pts · ceil(k/2)` with `k = ceil(n / max_pts)`, so leaves are full
/// `max_pts` batches except for a few remainders (a plain median split would
/// leave batches anywhere in `(max_pts/2, max_pts]`). The comparator is a
/// total order — coordinate by `f64::total_cmp`, ties by index — and
/// `select_nth_unstable_by` is a deterministic algorithm, so the partition is
/// a pure function of `(xyz, max_pts)`. Each batch's indices are returned in
/// ascending order. Every index appears in exactly one batch.
pub(crate) fn partition_points(xyz: &[[f64; 3]], max_pts: usize) -> Vec<Vec<u32>> {
    let max_pts = max_pts.max(1);
    assert!(
        xyz.len() <= u32::MAX as usize,
        "grid has more than u32::MAX points"
    );
    let mut idx: Vec<u32> = (0..xyz.len() as u32).collect();
    let mut out = Vec::with_capacity(xyz.len().div_ceil(max_pts));
    bisect(xyz, &mut idx, max_pts, &mut out);
    out
}

fn bisect(xyz: &[[f64; 3]], idx: &mut [u32], max_pts: usize, out: &mut Vec<Vec<u32>>) {
    let n = idx.len();
    if n == 0 {
        return;
    }
    if n <= max_pts {
        let mut v = idx.to_vec();
        v.sort_unstable();
        out.push(v);
        return;
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &i in idx.iter() {
        let p = xyz[i as usize];
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let mut axis = 0usize;
    for a in 1..3 {
        if hi[a] - lo[a] > hi[axis] - lo[axis] {
            axis = a;
        }
    }
    // k >= 2 here, and ceil(k/2) <= k-1, and n > max_pts·(k-1), so 0 < h < n.
    let k = n.div_ceil(max_pts);
    let h = max_pts * k.div_ceil(2);
    debug_assert!(h > 0 && h < n);
    idx.select_nth_unstable_by(h, |&a, &b| {
        xyz[a as usize][axis]
            .total_cmp(&xyz[b as usize][axis])
            .then(a.cmp(&b))
    });
    let (left, right) = idx.split_at_mut(h);
    bisect(xyz, left, max_pts, out);
    bisect(xyz, right, max_pts, out);
}

/// Indices of the shells significant on `pts`: `max(|χ|, |∂xχ|, |∂yχ|, |∂zχ|)`
/// over the points and the shell's functions strictly exceeds `thresh`.
/// `thresh <= 0` keeps every shell (no evaluation needed). Serial.
pub(crate) fn screen_batch(
    shells: &[OwnedShell],
    pts: &[[f64; 3]],
    thresh: f64,
) -> Result<Vec<u32>, GtoEvalError> {
    if thresh <= 0.0 {
        return Ok((0..shells.len() as u32).collect());
    }
    let mut keep = Vec::new();
    let mut buf = [0.0f64; 15];
    let mut gbuf: [[f64; 15]; 3] = [[0.0; 15]; 3];
    for (si, sh) in shells.iter().enumerate() {
        let loc = sh.located();
        let n = sh.nfunc;
        let mut m = 0.0_f64;
        for p in pts {
            buf.fill(0.0);
            for row in gbuf.iter_mut() {
                row.fill(0.0);
            }
            eval_shell_and_grad(
                &loc,
                p[0] - sh.center[0],
                p[1] - sh.center[1],
                p[2] - sh.center[2],
                &mut buf[..n],
                &mut gbuf,
            )?;
            for i in 0..n {
                m = m
                    .max(buf[i].abs())
                    .max(gbuf[0][i].abs())
                    .max(gbuf[1][i].abs())
                    .max(gbuf[2][i].abs());
            }
            if m > thresh {
                break;
            }
        }
        if m > thresh {
            keep.push(si as u32);
        }
    }
    Ok(keep)
}

/// Compact AO block for one batch: shape `(nact, 4·nb)`, row `r` = the r-th
/// active function (shells in `active` order, functions within a shell in
/// order), column blocks `[χ | ∂xχ | ∂yχ | ∂zχ]`, each `nb` wide. Serial.
///
/// Values are bit-identical to the dense
/// [`crate::ao_grid::eval_basis_and_grad_on_points_unchecked`]: same per-shell
/// function, same `p - center` offsets, same zeroed scratch.
pub(crate) fn eval_batch_ao(
    shells: &[OwnedShell],
    active: &[u32],
    nact: usize,
    pts: &[[f64; 3]],
) -> Result<Array2<f64>, GtoEvalError> {
    let nb = pts.len();
    let ld = 4 * nb;
    let mut ao = Array2::<f64>::zeros((nact, ld));
    let a = ao
        .as_slice_mut()
        .expect("freshly allocated Array2 is contiguous");
    let mut buf = [0.0f64; 15];
    let mut gbuf: [[f64; 15]; 3] = [[0.0; 15]; 3];
    let mut row0 = 0usize;
    for &si in active {
        let sh = &shells[si as usize];
        let loc = sh.located();
        let n = sh.nfunc;
        for (g, p) in pts.iter().enumerate() {
            buf.fill(0.0);
            for row in gbuf.iter_mut() {
                row.fill(0.0);
            }
            eval_shell_and_grad(
                &loc,
                p[0] - sh.center[0],
                p[1] - sh.center[1],
                p[2] - sh.center[2],
                &mut buf[..n],
                &mut gbuf,
            )?;
            for i in 0..n {
                let base = (row0 + i) * ld;
                a[base + g] = buf[i];
                a[base + nb + g] = gbuf[0][i];
                a[base + 2 * nb + g] = gbuf[1][i];
                a[base + 3 * nb + g] = gbuf[2][i];
            }
        }
        row0 += n;
    }
    debug_assert_eq!(row0, nact);
    Ok(ao)
}

/// `D[funcs, funcs]` as a dense `(nact, nact)` matrix.
fn gather_sub(d: &Array2<f64>, funcs: &[u32]) -> Array2<f64> {
    let n = funcs.len();
    Array2::from_shape_fn((n, n), |(i, j)| d[(funcs[i] as usize, funcs[j] as usize)])
}

/// How one spin density enters pass 1.
///
/// `Dense` is the full AO density matrix. `Factored` is `D = scale · C·Cᵀ` with
/// `C` the occupied MO block `(nbf, nocc)`: pass 1 then forms `D_sub·X` as
/// `scale · C_sub·(C_subᵀ·X)` — two GEMMs of `nact·nocc·ncols` each instead of
/// one of `nact²·ncols`, and a `(nact, nocc)` working matrix instead of
/// `(nact, nact)`.
#[derive(Clone, Copy)]
pub(crate) enum DensityInput<'a> {
    Dense(&'a Array2<f64>),
    Factored { c_occ: &'a Array2<f64>, scale: f64 },
}

impl DensityInput<'_> {
    fn nbf(&self) -> usize {
        match self {
            Self::Dense(d) => d.nrows(),
            Self::Factored { c_occ, .. } => c_occ.nrows(),
        }
    }

    fn is_square(&self) -> bool {
        match self {
            Self::Dense(d) => d.nrows() == d.ncols(),
            Self::Factored { .. } => true,
        }
    }

    fn gather(&self, funcs: &[u32]) -> SubDensity {
        match self {
            Self::Dense(d) => SubDensity::Dense(gather_sub(d, funcs)),
            Self::Factored { c_occ, scale } => SubDensity::Factored {
                csub: gather_rows(c_occ, funcs),
                scale: *scale,
            },
        }
    }
}

/// Per-batch gathered form of a [`DensityInput`].
enum SubDensity {
    Dense(Array2<f64>),
    Factored { csub: Array2<f64>, scale: f64 },
}

impl SubDensity {
    /// `D_sub · X` for the batch's `(nact, ncols)` AO columns.
    fn apply(&self, x: ArrayView2<f64>) -> Array2<f64> {
        match self {
            Self::Dense(dsub) => dsub.dot(&x),
            Self::Factored { csub, scale } => {
                let psi = csub.t().dot(&x);
                let mut phi = csub.dot(&psi);
                // `scale` is 1 or 2 (exact in binary): no rounding.
                phi *= *scale;
                phi
            }
        }
    }
}

/// `C[funcs, :]` as a dense `(nact, ncols)` matrix.
fn gather_rows(c: &Array2<f64>, funcs: &[u32]) -> Array2<f64> {
    let m = c.ncols();
    Array2::from_shape_fn((funcs.len(), m), |(i, j)| c[(funcs[i] as usize, j)])
}

/// Number of fixed reduction groups for the pass-2 accumulation: a pure
/// function of `(nbatches, nbf, nmats)` — NEVER of the rayon thread count —
/// so the V summation order is identical for every worker count.
pub(crate) fn n_groups(nbatches: usize, nbf: usize, nmats: usize) -> usize {
    let per = nbf
        .saturating_mul(nbf)
        .saturating_mul(8)
        .saturating_mul(nmats.max(1))
        .max(1);
    let cap = (ACCUM_CAP_BYTES / per).max(1);
    TARGET_GROUPS.min(cap).min(nbatches).max(1)
}

/// Contiguous `[b0, b1)` batch ranges for `ngroups` groups over `nbatches`.
fn group_ranges(nbatches: usize, ngroups: usize) -> Vec<(usize, usize)> {
    if nbatches == 0 {
        return Vec::new();
    }
    let size = nbatches.div_ceil(ngroups.max(1)).max(1);
    (0..nbatches.div_ceil(size))
        .map(|g| (g * size, ((g + 1) * size).min(nbatches)))
        .collect()
}

/// Add `V_b + V_bᵀ` into `acc[funcs, funcs]`. Both `(i,j)` and `(j,i)` receive
/// the same increment (fp `+` is commutative), so `acc` stays exactly
/// symmetric, like the dense path's final `½(V + Vᵀ)`.
fn scatter_sym(v: &[f64], funcs: &[u32], acc: &mut Array2<f64>) {
    scatter_sym_ld(v, funcs.len(), funcs, acc);
}

/// [`scatter_sym`] of a `V_b` whose rows are `ld ≥ nact` apart (the device's
/// zero-padded blocks); only the leading `nact × nact` block is read.
fn scatter_sym_ld(v: &[f64], ld: usize, funcs: &[u32], acc: &mut Array2<f64>) {
    let n = funcs.len();
    let nbf = acc.ncols();
    debug_assert!(n == 0 || v.len() >= (n - 1) * ld + n);
    let acc_s = acc.as_slice_mut().expect("accumulator is contiguous");
    for i in 0..n {
        let fi = funcs[i] as usize;
        for j in 0..n {
            let fj = funcs[j] as usize;
            acc_s[fi * nbf + fj] += v[i * ld + j] + v[j * ld + i];
        }
    }
}

/// Where a per-batch row-major block keeps its numbers: rows `ld` apart and
/// the components (χ, ∂xχ, ∂yχ, ∂zχ, or the matching factor columns) `cs`
/// columns apart. The CPU path is compact (`cs = nb`); the device pads.
#[derive(Clone, Copy, Debug)]
struct BlockLayout {
    ld: usize,
    cs: usize,
}

impl BlockLayout {
    fn compact(nb: usize, four: bool) -> Self {
        Self {
            ld: if four { 4 * nb } else { nb },
            cs: nb,
        }
    }
}

/// Per-batch pass-1 output (points in the batch's `idx` order).
struct BatchDensity {
    rho: Vec<f64>,
    grad: [Vec<f64>; 3],
    tau: Vec<f64>,
}

/// ρ, ∇ρ and (if `need_tau`) τ for one batch and one (sub-)density matrix.
///
/// `ao` is the batch's `(nact, 4nb)` block. Per point, every sum runs over μ
/// in ascending order; τ keeps the dense path's per-axis partial sums
/// (`((τx + τy) + τz)·½`).
fn batch_density_one(
    ao: &Array2<f64>,
    dsub: &SubDensity,
    nb: usize,
    need_tau: bool,
) -> BatchDensity {
    let ncols = if need_tau { 4 * nb } else { nb };
    if ao.nrows() == 0 {
        // No active function: every sum is empty (ρ = ∇ρ = τ = 0).
        return batch_density_from_phi(ao, &[], nb, need_tau);
    }
    // Φ = D_sub · [χ | (∂χ)] — ONE GEMM, BLAS threads set once by
    // `integrate_*_exec` (1 on rayon workers; see `Exec`).
    let phi = dsub.apply(ao.slice(s![.., ..ncols]));
    let phi = phi.as_standard_layout();
    batch_density_from_phi(ao, phi.as_slice().expect("standard layout"), nb, need_tau)
}

/// The point-local reductions of [`batch_density_one`] for a given row-major
/// `Φ` (`(nact, ncols)`, `ncols = 4nb` with τ, else `nb`), wherever the GEMM
/// that produced it ran.
fn batch_density_from_phi(ao: &Array2<f64>, p: &[f64], nb: usize, need_tau: bool) -> BatchDensity {
    debug_assert_eq!(p.len(), ao.nrows() * BlockLayout::compact(nb, need_tau).ld);
    batch_density_from_phi_in(ao, p, BlockLayout::compact(nb, need_tau), nb, need_tau)
}

/// [`batch_density_from_phi`] for a `Φ` laid out as `pl` (row `μ` at `μ·ld`,
/// component `c` at column `c·cs`); the same sums in the same order.
fn batch_density_from_phi_in(
    ao: &Array2<f64>,
    p: &[f64],
    pl: BlockLayout,
    nb: usize,
    need_tau: bool,
) -> BatchDensity {
    let nact = ao.nrows();
    let ld = 4 * nb;
    let ncols = if need_tau { 3 * pl.cs + nb } else { nb };
    debug_assert!(nact == 0 || p.len() >= (nact - 1) * pl.ld + ncols);
    let mut rho = vec![0.0_f64; nb];
    let mut gx = vec![0.0_f64; nb];
    let mut gy = vec![0.0_f64; nb];
    let mut gz = vec![0.0_f64; nb];
    let mut tau = Vec::new();
    let a = ao.as_slice().expect("batch AO block is contiguous");
    let tn = if need_tau { nb } else { 0 };
    let mut t = [vec![0.0_f64; tn], vec![0.0_f64; tn], vec![0.0_f64; tn]];
    for mu in 0..nact {
        let crow = &a[mu * ld..(mu + 1) * ld];
        let prow = &p[mu * pl.ld..mu * pl.ld + ncols];
        for g in 0..nb {
            let pv = prow[g];
            rho[g] += crow[g] * pv;
            gx[g] += pv * crow[nb + g];
            gy[g] += pv * crow[2 * nb + g];
            gz[g] += pv * crow[3 * nb + g];
        }
        if need_tau {
            for ax in 0..3 {
                let (off, poff) = ((1 + ax) * nb, (1 + ax) * pl.cs);
                let tax = &mut t[ax];
                for g in 0..nb {
                    tax[g] += crow[off + g] * prow[poff + g];
                }
            }
        }
    }
    for g in 0..nb {
        gx[g] *= 2.0;
        gy[g] *= 2.0;
        gz[g] *= 2.0;
    }
    if need_tau {
        tau = (0..nb)
            .map(|g| {
                let mut acc = 0.0_f64;
                acc += t[0][g];
                acc += t[1][g];
                acc += t[2][g];
                0.5 * acc
            })
            .collect();
    }
    BatchDensity {
        rho,
        grad: [gx, gy, gz],
        tau,
    }
}

/// Per-point pass-2 factors of one spin channel for one batch.
struct BatchFactors {
    /// `½ · w v_ρ` (gated on ρ).
    s_half: Vec<f64>,
    /// `f_a` for a = x, y, z (empty unless GGA/meta-GGA).
    f: [Vec<f64>; 3],
    /// `½ · (½ w v_τ)` (empty unless meta-GGA).
    t_half: Vec<f64>,
}

/// One `V_b = A · Bᵀ` for a batch and one spin channel, scattered as
/// `V_b + V_bᵀ` into `acc`.
fn batch_vxc_one(
    ao: &Array2<f64>,
    funcs: &[u32],
    nb: usize,
    fac: &BatchFactors,
    (has_gga, has_mgga, mixed): (bool, bool, bool),
    acc: &mut Array2<f64>,
) {
    let nact = ao.nrows();
    if nact == 0 {
        return;
    }
    let ncols = if has_mgga { 4 * nb } else { nb };
    let mut amat = Array2::<f64>::zeros((nact, ncols));
    fill_amat(
        ao,
        nb,
        fac,
        (has_gga, has_mgga),
        amat.as_slice_mut().expect("contiguous"),
    );
    let vb = vb_product(&amat, ao.slice(s![.., ..ncols]), mixed);
    let vb = vb.as_standard_layout();
    scatter_sym(vb.as_slice().expect("standard layout"), funcs, acc);
}

/// Process-wide count of V_xc passes that formed at least one batch product
/// with the mixed `xc-vxc` kernel: on the device (a pass with at least one
/// mixed wave) or, under the `MIXED_VXC_HOST_TWIN` test seam, with its host
/// twin. A pass the ladder approved but that ran f64 throughout (every wave
/// fell back, or the grid is declined) is not counted. Tests read deltas.
#[doc(hidden)]
pub static MIXED_VXC_BUILDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn note_mixed_vxc_pass() {
    MIXED_VXC_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// True when the CPU pass 2 forms mixed products with the host twin (the
/// `test-seams` seam); always false in a shipped build.
fn host_twin_active() -> bool {
    #[cfg(feature = "test-seams")]
    {
        gpu::mixed_vxc_host_twin()
    }
    #[cfg(not(feature = "test-seams"))]
    {
        false
    }
}

/// `V_b = A · Xᵀ` on the host. A `mixed` build under the `MIXED_VXC_HOST_TWIN`
/// test seam forms it with the host twin of the device's `xc-vxc` kernel
/// (operands rounded to f32, f32 panels summed in f64;
/// `ferric_core::gpu::mixed_host`), so the kernel's error map and its effect on
/// an SCF are measurable without a device. Otherwise (every shipped CPU build)
/// it is the f64 product whatever `mixed` says: mixed precision is device-only.
fn vb_product(amat: &Array2<f64>, x: ArrayView2<f64>, mixed: bool) -> Array2<f64> {
    #[cfg(feature = "test-seams")]
    if mixed && host_twin_active() {
        let mut out = Array2::<f64>::zeros((amat.nrows(), x.nrows()));
        ferric_core::gpu::mixed_host::gemm_f32_f64acc_host(
            &amat.view(),
            &x.t(),
            &mut out.view_mut(),
            ferric_core::gpu::mixed::effective_k_panel(),
        );
        return out;
    }
    let _ = mixed;
    amat.dot(&x.t())
}

/// Write the pass-2 left factor `A` of one batch (row-major `(nact, ncols)`,
/// `ncols = 4nb` for meta-GGA, else `nb`) into `am`.
fn fill_amat(ao: &Array2<f64>, nb: usize, fac: &BatchFactors, flags: (bool, bool), am: &mut [f64]) {
    let lay = BlockLayout::compact(nb, flags.1);
    debug_assert_eq!(am.len(), ao.nrows() * lay.ld);
    fill_amat_in(ao, nb, fac, flags, am, lay);
}

/// [`fill_amat`] into a block laid out as `lay` (row `μ` at `μ·ld`, factor
/// component `c` at column `c·cs`). Writes only the real entries; padding is
/// the caller's.
fn fill_amat_in(
    ao: &Array2<f64>,
    nb: usize,
    fac: &BatchFactors,
    (has_gga, has_mgga): (bool, bool),
    am: &mut [f64],
    lay: BlockLayout,
) {
    let nact = ao.nrows();
    let ld = 4 * nb;
    let ncols = if has_mgga { 3 * lay.cs + nb } else { nb };
    debug_assert!(nact == 0 || am.len() >= (nact - 1) * lay.ld + ncols);
    let a = ao.as_slice().expect("batch AO block is contiguous");
    for mu in 0..nact {
        let crow = &a[mu * ld..(mu + 1) * ld];
        let arow = &mut am[mu * lay.ld..mu * lay.ld + ncols];
        for g in 0..nb {
            let mut v = fac.s_half[g] * crow[g];
            if has_gga {
                v += fac.f[0][g] * crow[nb + g];
                v += fac.f[1][g] * crow[2 * nb + g];
                v += fac.f[2][g] * crow[3 * nb + g];
            }
            arow[g] = v;
        }
        if has_mgga {
            for ax in 0..3 {
                let (off, aoff) = ((1 + ax) * nb, (1 + ax) * lay.cs);
                for g in 0..nb {
                    arow[aoff + g] = fac.t_half[g] * crow[off + g];
                }
            }
        }
    }
}

/// How one Fock build is executed.
///
/// * **Parallel** (default): batches run on rayon workers, OpenBLAS pinned to
///   one thread for the whole call. Bit-identical across rayon thread counts.
/// * **Serial**: chosen only when the caller is NOT inside a rayon worker and
///   the rayon pool has exactly one thread — the documented
///   `OPENBLAS_NUM_THREADS>1 ⇒ RAYON_NUM_THREADS=1` configuration, where the
///   parallel mode would leave the build fully serial on one BLAS thread. The
///   batch loops then run on the CALLING thread (outside any rayon worker,
///   which is the call-path proof `blas_threads.rs` requires for a raise) with
///   `opt_in_blas_threads()` BLAS threads — the same opt-in the old dense path
///   used for its digestion GEMMs.
///
/// # What is and is not bit-identical
///
/// Serial and Parallel perform the same floating-point operations in the same
/// order: same batches, same fixed reduction groups, each group accumulated in
/// ascending batch order, groups summed in ascending order. With the default
/// `FERRIC_BLAS_THREADS` (unset ⇒ 1) the two are therefore bit-identical, so
/// the energy is bit-identical for EVERY rayon thread count (pinned by
/// `serial_and_parallel_execution_are_bit_identical`). With
/// `FERRIC_BLAS_THREADS=N>1` AND a one-thread rayon pool, the per-batch GEMMs
/// run on N OpenBLAS threads, whose internal work split is not guaranteed to
/// reproduce the single-thread bit pattern — the same caveat that already
/// applies to every other `FERRIC_BLAS_THREADS` site (see `einsum.rs`). The
/// per-batch GEMMs are small (`nact × 128 × nact`), so expect a modest gain
/// from the raise at best; the rayon-parallel mode is the fast path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Exec {
    serial: bool,
    blas: usize,
    /// Pass 2 may use the device's mixed-precision `xc-vxc` kernel (see
    /// `KsXc`'s precision ladder); ignored on the CPU, f64 by default.
    mixed_vxc: bool,
}

impl Exec {
    /// The mode for a call made from the current thread (see `Exec`).
    pub(crate) fn auto() -> Self {
        if rayon::current_thread_index().is_none() && rayon::current_num_threads() == 1 {
            Self {
                serial: true,
                blas: ferric_integrals::blas_threads::opt_in_blas_threads(),
                mixed_vxc: false,
            }
        } else {
            Self::parallel()
        }
    }
    pub(crate) fn parallel() -> Self {
        Self {
            serial: false,
            blas: 1,
            mixed_vxc: false,
        }
    }
    /// This mode with pass 2 allowed (or not) to run in mixed precision.
    pub(crate) fn with_mixed_vxc(self, mixed_vxc: bool) -> Self {
        Self { mixed_vxc, ..self }
    }
    #[cfg(test)]
    pub(crate) fn serial_single_blas() -> Self {
        Self {
            serial: true,
            blas: 1,
            mixed_vxc: false,
        }
    }
}

/// `(0..n).map(f)` collected in index order, serially or on rayon. Output
/// order (and each `f(i)`) is identical either way.
fn map_indexed<T, F>(serial: bool, n: usize, f: F) -> Result<Vec<T>, GtoEvalError>
where
    T: Send,
    F: Fn(usize) -> Result<T, GtoEvalError> + Sync + Send,
{
    if serial {
        (0..n).map(f).collect()
    } else {
        (0..n).into_par_iter().map(f).collect()
    }
}

/// One spatial batch: original point indices, active shells, active functions.
#[derive(Debug)]
pub(crate) struct XcBatch {
    idx: Vec<u32>,
    shells: Vec<u32>,
    funcs: Vec<u32>,
}

/// The screened, batched grid used by `KsXc`/`KsXcUks` for every Fock build.
#[derive(Debug)]
pub(crate) struct ScreenedGrid {
    nbf: usize,
    npts: usize,
    shells: Vec<OwnedShell>,
    batches: Vec<XcBatch>,
    /// Resident compact AO blocks, one per batch; `None` = recompute.
    ao: Option<Vec<Array2<f64>>>,
    max_batch_pts: usize,
    /// Device copy of the AO blocks and GEMM staging (feature `gpu`);
    /// `Untried` until the first integration asks for it. A `Mutex` because
    /// the integrations take `&self`; one lock per Fock build, uncontended.
    #[cfg(feature = "gpu")]
    device: std::sync::Mutex<gpu::DeviceSlot>,
}

impl ScreenedGrid {
    /// Partition `grid` and screen every batch (parallel over batches; each
    /// batch serial). Does NOT store AO values — see [`Self::materialize`].
    pub(crate) fn build(
        grid: &[GridPoint],
        shells: Vec<OwnedShell>,
        nbf: usize,
        cfg: &XcBatchConfig,
    ) -> Result<Self, GtoEvalError> {
        debug_assert_eq!(shells.iter().map(|s| s.nfunc).sum::<usize>(), nbf);
        // Evaluate every shell once, whatever the threshold. `screen_batch`
        // skips evaluation entirely at `thresh <= 0`, so without this an
        // unsupported shell (UnsupportedL) would first surface inside a Fock
        // build — where `add_xc` can only panic — instead of here, where it
        // is a proper constructor error. After this, per-shell evaluation
        // cannot fail later (the only error is a function of `l` alone).
        {
            let mut buf = [0.0f64; 15];
            let mut gbuf: [[f64; 15]; 3] = [[0.0; 15]; 3];
            for sh in &shells {
                eval_shell_and_grad(
                    &sh.located(),
                    0.5,
                    0.5,
                    0.5,
                    &mut buf[..sh.nfunc],
                    &mut gbuf,
                )?;
            }
        }
        let xyz: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let parts = partition_points(&xyz, cfg.max_batch_pts);
        let thresh = cfg.screen_thresh;
        let batches = parts
            .into_par_iter()
            .map(|idx| -> Result<XcBatch, GtoEvalError> {
                let pts: Vec<[f64; 3]> = idx.iter().map(|&i| xyz[i as usize]).collect();
                let kept = screen_batch(&shells, &pts, thresh)?;
                let funcs: Vec<u32> = kept
                    .iter()
                    .flat_map(|&s| {
                        let sh = &shells[s as usize];
                        (sh.offset..sh.offset + sh.nfunc).map(|f| f as u32)
                    })
                    .collect();
                Ok(XcBatch {
                    idx,
                    shells: kept,
                    funcs,
                })
            })
            .collect::<Result<Vec<_>, GtoEvalError>>()?;
        Ok(Self {
            nbf,
            npts: grid.len(),
            shells,
            batches,
            ao: None,
            max_batch_pts: cfg.max_batch_pts.max(1),
            #[cfg(feature = "gpu")]
            device: Default::default(),
        })
    }

    /// Bytes the resident compact AO blocks would occupy.
    pub(crate) fn resident_ao_bytes(&self) -> usize {
        self.batches
            .iter()
            .map(|b| b.funcs.len() * 4 * b.idx.len() * 8)
            .sum()
    }

    /// Bytes one Fock build allocates on top of the (optional) resident AO
    /// blocks: the O(npts) global density/kernel vectors, the per-group dense
    /// accumulators (all alive until the ordered final sum) plus the result,
    /// and per-worker batch scratch.
    ///
    /// Reads the rayon thread count for the worker-scratch term. That is safe
    /// here and ONLY here: this figure steers the resident-vs-recompute
    /// choice, and the two modes are bit-identical by construction, so no
    /// energy can depend on it. (Contrast `ks.rs`'s old `resolve_batch_size`,
    /// whose output fixed a summation order and therefore must not.)
    pub(crate) fn transient_bytes(&self, is_uks: bool) -> usize {
        let nmats = if is_uks { 2 } else { 1 };
        let mat = self.nbf * self.nbf * 8;
        let groups = n_groups(self.batches.len(), self.nbf, nmats);
        let accum = (groups + 1) * nmats * mat;
        // Global per-point doubles: closed = DensityGrid 5 + τ 1 + kernel
        // totals 4 + one functional's temporaries 4 + pass-1 batch copies 5
        // (+ slack) ≈ 20+6; UKS = UksDensityGrid 11 + τ 2 + interleaved
        // inputs 7 + totals 8 + temporaries 8 + pass-1 copies 10 ≈ 40+10.
        let per_pt = if is_uks { 50 } else { 26 };
        let companion = self.npts * per_pt * 8;
        // Per worker, bounded with nbf for nact: recomputed AO block (4 planes)
        // + Φ (4 planes, ×2 UKS) + A (4 planes) + D_sub/V_b (nbf² each, ×2 UKS).
        let nb = self.max_batch_pts;
        let worker = (4 + 4 * nmats + 4) * self.nbf * nb * 8 + 2 * nmats * mat;
        companion + accum + rayon::current_num_threads() * worker
    }

    /// Evaluate and keep every batch's compact AO block (parallel over
    /// batches, each batch serial).
    pub(crate) fn materialize(&mut self, grid: &[GridPoint]) -> Result<(), GtoEvalError> {
        let this = &*self;
        let ao = (0..this.batches.len())
            .into_par_iter()
            .map(|bi| this.eval_batch(bi, grid))
            .collect::<Result<Vec<_>, GtoEvalError>>()?;
        self.ao = Some(ao);
        Ok(())
    }

    fn eval_batch(&self, bi: usize, grid: &[GridPoint]) -> Result<Array2<f64>, GtoEvalError> {
        let b = &self.batches[bi];
        let pts: Vec<[f64; 3]> = b.idx.iter().map(|&i| grid[i as usize].xyz).collect();
        eval_batch_ao(&self.shells, &b.shells, b.funcs.len(), &pts)
    }

    fn batch_ao(
        &self,
        bi: usize,
        grid: &[GridPoint],
    ) -> Result<Cow<'_, Array2<f64>>, GtoEvalError> {
        match &self.ao {
            Some(v) => Ok(Cow::Borrowed(&v[bi])),
            None => Ok(Cow::Owned(self.eval_batch(bi, grid)?)),
        }
    }

    /// The device session for one integration: `None`/inert when this
    /// integration runs on the CPU (no `gpu` feature, mode off, no device,
    /// declined, called from a rayon worker). The first call uploads the AO
    /// blocks.
    fn device_session(&self, grid: &[GridPoint]) -> DevSession<'_> {
        #[cfg(feature = "gpu")]
        {
            gpu::session(self, grid)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = grid;
            std::marker::PhantomData
        }
    }

    /// True when the KS XC GEMMs of this grid run on the device (the AO
    /// blocks are uploaded and no failure has sent the grid back to the CPU).
    pub(crate) fn on_device(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            gpu::is_ready(self)
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Page-locked host staging bytes held for this grid's device waves (0
    /// without the `gpu` feature or off the device).
    pub(crate) fn pinned_staging_bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        {
            gpu::pinned_bytes(self)
        }
        #[cfg(not(feature = "gpu"))]
        {
            0
        }
    }

    /// True when the compact AO blocks are resident.
    pub(crate) fn is_resident(&self) -> bool {
        self.ao.is_some()
    }

    /// Maximum points per batch.
    pub(crate) fn max_batch_pts(&self) -> usize {
        self.max_batch_pts
    }

    /// Screening/batching summary.
    pub(crate) fn stats(&self) -> XcScreeningStats {
        let mut act = 0.0_f64;
        let mut gemm = 0.0_f64;
        for b in &self.batches {
            let na = b.funcs.len() as f64;
            let nb = b.idx.len() as f64;
            act += na * nb;
            gemm += na * na * nb;
        }
        let nbf = self.nbf as f64;
        let npts = (self.npts as f64).max(1.0);
        XcScreeningStats {
            nbatches: self.batches.len(),
            npts: self.npts,
            nbf: self.nbf,
            active_fraction: act / (nbf * npts).max(1.0),
            gemm_fraction: gemm / (nbf * nbf * npts).max(1.0),
            resident: self.is_resident(),
        }
    }

    /// Pass 1 for the (spin) densities `ds`: `out[i][bi]` is batch `bi` of
    /// density `i`. Runs the Φ GEMMs on the device when `dev` holds a ready
    /// session (see `gpu`), otherwise on the CPU; a device failure falls back
    /// to the CPU here, for this pass and every later one. Each batch's AO
    /// block is fetched once for all densities (one evaluation in recompute
    /// mode).
    fn density_pass(
        &self,
        grid: &[GridPoint],
        ds: &[DensityInput],
        need_tau: bool,
        serial: bool,
        dev: &mut DevSession<'_>,
    ) -> Result<Vec<Vec<BatchDensity>>, GtoEvalError> {
        #[cfg(feature = "gpu")]
        if let Some(parts) = gpu::density_pass(self, grid, ds, need_tau, serial, dev)? {
            return Ok(parts);
        }
        let _ = dev;
        let per_batch = map_indexed(serial, self.batches.len(), |bi| {
            let b = &self.batches[bi];
            let ao = self.batch_ao(bi, grid)?;
            Ok(ds
                .iter()
                .map(|d| batch_density_one(&ao, &d.gather(&b.funcs), b.idx.len(), need_tau))
                .collect::<Vec<_>>())
        })?;
        let mut out: Vec<Vec<BatchDensity>> = ds
            .iter()
            .map(|_| Vec::with_capacity(per_batch.len()))
            .collect();
        for parts in per_batch {
            for (o, p) in out.iter_mut().zip(parts) {
                o.push(p);
            }
        }
        Ok(out)
    }

    /// Pass 2: the `nmats` V matrices, `factors(batch, mat)` giving matrix
    /// `mat`'s per-point factors. Same fixed-group ordered reduction on either
    /// path; device dispatch and fallback as in [`Self::density_pass`].
    fn potential_pass<F>(
        &self,
        grid: &[GridPoint],
        nmats: usize,
        factors: &F,
        (has_gga, has_mgga): (bool, bool),
        exec: Exec,
        dev: &mut DevSession<'_>,
    ) -> Result<Vec<Array2<f64>>, GtoEvalError>
    where
        F: Fn(&XcBatch, usize) -> BatchFactors + Sync,
    {
        // Meta-GGA τ is on the never-f32 list: refused here whatever the caller says.
        let (serial, mixed) = (exec.serial, exec.mixed_vxc && !has_mgga);
        #[cfg(feature = "gpu")]
        if let Some(v) = gpu::potential_pass(
            self,
            grid,
            nmats,
            factors,
            (has_gga, has_mgga),
            exec.with_mixed_vxc(mixed),
            dev,
        )? {
            return Ok(v);
        }
        let _ = dev;
        if mixed && host_twin_active() {
            note_mixed_vxc_pass();
        }
        self.reduce_groups(serial, nmats, |bi, accs| {
            let b = &self.batches[bi];
            if b.funcs.is_empty() {
                return Ok(());
            }
            let nb = b.idx.len();
            let ao = self.batch_ao(bi, grid)?;
            for (mat, acc) in accs.iter_mut().enumerate() {
                let fac = factors(b, mat);
                batch_vxc_one(&ao, &b.funcs, nb, &fac, (has_gga, has_mgga, mixed), acc);
            }
            Ok(())
        })
    }

    /// Fixed-grouping, ordered reduction of per-batch V contributions.
    /// `body(bi, accs)` adds batch `bi` into the group's `nmats` accumulators.
    fn reduce_groups<F>(
        &self,
        serial: bool,
        nmats: usize,
        body: F,
    ) -> Result<Vec<Array2<f64>>, GtoEvalError>
    where
        F: Fn(usize, &mut [Array2<f64>]) -> Result<(), GtoEvalError> + Sync + Send,
    {
        let nbf = self.nbf;
        let ngroups = n_groups(self.batches.len(), nbf, nmats);
        let ranges = group_ranges(self.batches.len(), ngroups);
        let partials: Vec<Vec<Array2<f64>>> = map_indexed(serial, ranges.len(), |gi| {
            let (b0, b1) = ranges[gi];
            let mut accs: Vec<Array2<f64>> =
                (0..nmats).map(|_| Array2::zeros((nbf, nbf))).collect();
            for bi in b0..b1 {
                body(bi, &mut accs[..])?;
            }
            Ok(accs)
        })?;
        // Ascending group order — the one and only cross-group summation.
        let mut out: Vec<Array2<f64>> = (0..nmats).map(|_| Array2::zeros((nbf, nbf))).collect();
        for accs in &partials {
            for (o, a) in out.iter_mut().zip(accs) {
                *o += a;
            }
        }
        Ok(out)
    }

    /// Closed-shell `(E_xc, V_xc)` for the total density matrix `d`.
    ///
    /// The OpenBLAS thread count is set ONCE on the calling thread for the
    /// whole call (it is process-global, so setting it per GEMM inside workers
    /// would race): 1 in the rayon-parallel mode, where every GEMM runs on a
    /// worker and a multi-threaded OpenBLAS oversubscribes (and has crashed
    /// before, see `blas_threads.rs`); `opt_in_blas_threads()` in the serial
    /// mode. See `Exec` for when each is chosen and what is bit-identical
    /// (`Exec::auto()` is the production mode), and for `mixed_vxc`.
    pub(crate) fn integrate_closed_exec(
        &self,
        grid: &[GridPoint],
        d: DensityInput,
        xc: &XcDef,
        exec: Exec,
    ) -> Result<(f64, Array2<f64>), GtoEvalError> {
        with_blas_threads(exec.blas, || self.integrate_closed_inner(grid, d, xc, exec))
    }

    fn integrate_closed_inner(
        &self,
        grid: &[GridPoint],
        d: DensityInput,
        xc: &XcDef,
        exec: Exec,
    ) -> Result<(f64, Array2<f64>), GtoEvalError> {
        let serial = exec.serial;
        let npts = self.npts;
        assert_eq!(grid.len(), npts, "grid does not match the screened grid");
        assert!(d.nbf() == self.nbf && d.is_square(), "density matrix shape");
        let has_mgga = xc
            .funcs
            .iter()
            .any(|f| matches!(f.family(), FunctionalFamily::MetaGga));
        let has_gga = xc
            .funcs
            .iter()
            .any(|f| !matches!(f.family(), FunctionalFamily::Lda));

        let mut dev = self.device_session(grid);

        // ── Pass 1: density, parallel over batches, disjoint output slots.
        let parts = self
            .density_pass(grid, &[d], has_mgga, serial, &mut dev)?
            .pop()
            .expect("one density");
        let mut rho = Array1::<f64>::zeros(npts);
        let mut grad = Array2::<f64>::zeros((3, npts));
        let mut tau = Array1::<f64>::zeros(if has_mgga { npts } else { 0 });
        for (b, p) in self.batches.iter().zip(&parts) {
            for (j, &gi) in b.idx.iter().enumerate() {
                let g = gi as usize;
                rho[g] = p.rho[j];
                grad[(0, g)] = p.grad[0][j];
                grad[(1, g)] = p.grad[1][j];
                grad[(2, g)] = p.grad[2][j];
                if has_mgga {
                    tau[g] = p.tau[j];
                }
            }
        }
        drop(parts);
        let mut sigma = Array1::<f64>::zeros(npts);
        for g in 0..npts {
            let (gx, gy, gz) = (grad[(0, g)], grad[(1, g)], grad[(2, g)]);
            sigma[g] = gx * gx + gy * gy + gz * gz;
        }
        let dens = DensityGrid { rho, grad, sigma };

        // ── Kernel: one global libxc evaluation (shared with the dense path).
        let k = closed_kernel(&dens, if has_mgga { Some(&tau) } else { None }, xc);
        let e_xc = deterministic_point_sum(npts, |g| grid[g].weight * dens.rho[g] * k.exc[g]);

        // ── Pass 2: V, fixed-group ordered reduction.
        let factors = |b: &XcBatch, _mat: usize| -> BatchFactors {
            let nb = b.idx.len();
            let mut fac = BatchFactors {
                s_half: vec![0.0; nb],
                f: if has_gga {
                    [vec![0.0; nb], vec![0.0; nb], vec![0.0; nb]]
                } else {
                    [Vec::new(), Vec::new(), Vec::new()]
                },
                t_half: if has_mgga { vec![0.0; nb] } else { Vec::new() },
            };
            for (j, &gi) in b.idx.iter().enumerate() {
                let g = gi as usize;
                let r = dens.rho[g];
                if r > DENSITY_FLOOR {
                    let w = grid[g].weight;
                    // Same expressions as the dense path, then an exact ×½.
                    fac.s_half[j] = 0.5 * (w * k.vrho[g]);
                    if has_gga {
                        let vs = k.vsigma[g];
                        for ax in 0..3 {
                            fac.f[ax][j] = 2.0 * w * vs * dens.grad[(ax, g)];
                        }
                    }
                    if has_mgga {
                        fac.t_half[j] = 0.5 * (0.5 * w * k.vtau[g]);
                    }
                }
            }
            fac
        };
        let mut v = self.potential_pass(grid, 1, &factors, (has_gga, has_mgga), exec, &mut dev)?;
        Ok((e_xc, v.pop().expect("one matrix")))
    }

    /// Spin-polarized `(E_xc, V_α, V_β)` for the spin density matrices. BLAS
    /// threading and execution mode as in [`Self::integrate_closed_exec`].
    pub(crate) fn integrate_polarized_exec(
        &self,
        grid: &[GridPoint],
        d_a: DensityInput,
        d_b: DensityInput,
        xc: &XcDef,
        exec: Exec,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), GtoEvalError> {
        with_blas_threads(exec.blas, || {
            self.integrate_polarized_inner(grid, d_a, d_b, xc, exec)
        })
    }

    fn integrate_polarized_inner(
        &self,
        grid: &[GridPoint],
        d_a: DensityInput,
        d_b: DensityInput,
        xc: &XcDef,
        exec: Exec,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), GtoEvalError> {
        let serial = exec.serial;
        let npts = self.npts;
        assert_eq!(grid.len(), npts, "grid does not match the screened grid");
        assert!(
            d_a.nbf() == self.nbf && d_a.is_square(),
            "alpha density shape"
        );
        assert!(
            d_b.nbf() == self.nbf && d_b.is_square(),
            "beta density shape"
        );
        let has_mgga = xc
            .funcs
            .iter()
            .any(|f| matches!(f.family(), FunctionalFamily::MetaGga));
        let has_gga = xc
            .funcs
            .iter()
            .any(|f| !matches!(f.family(), FunctionalFamily::Lda));

        let mut dev = self.device_session(grid);

        // ── Pass 1.
        let mut parts = self.density_pass(grid, &[d_a, d_b], has_mgga, serial, &mut dev)?;
        let parts_b = parts.pop().expect("two densities");
        let parts_a = parts.pop().expect("two densities");
        let mut rho_a = Array1::<f64>::zeros(npts);
        let mut rho_b = Array1::<f64>::zeros(npts);
        let mut grad_a = Array2::<f64>::zeros((3, npts));
        let mut grad_b = Array2::<f64>::zeros((3, npts));
        let tn = if has_mgga { npts } else { 0 };
        let mut tau_a = Array1::<f64>::zeros(tn);
        let mut tau_b = Array1::<f64>::zeros(tn);
        for (b, (pa, pb)) in self.batches.iter().zip(parts_a.iter().zip(&parts_b)) {
            for (j, &gi) in b.idx.iter().enumerate() {
                let g = gi as usize;
                rho_a[g] = pa.rho[j];
                rho_b[g] = pb.rho[j];
                for ax in 0..3 {
                    grad_a[(ax, g)] = pa.grad[ax][j];
                    grad_b[(ax, g)] = pb.grad[ax][j];
                }
                if has_mgga {
                    tau_a[g] = pa.tau[j];
                    tau_b[g] = pb.tau[j];
                }
            }
        }
        drop((parts_a, parts_b));
        let mut sigma = Array2::<f64>::zeros((3, npts));
        for g in 0..npts {
            let (ax, ay, az) = (grad_a[(0, g)], grad_a[(1, g)], grad_a[(2, g)]);
            let (bx, by, bz) = (grad_b[(0, g)], grad_b[(1, g)], grad_b[(2, g)]);
            sigma[(0, g)] = ax * ax + ay * ay + az * az;
            sigma[(1, g)] = ax * bx + ay * by + az * bz;
            sigma[(2, g)] = bx * bx + by * by + bz * bz;
        }
        let dens = UksDensityGrid {
            rho_a,
            rho_b,
            grad_a,
            grad_b,
            sigma,
        };

        // ── Kernel.
        let k = polarized_kernel(
            &dens,
            if has_mgga {
                Some((&tau_a, &tau_b))
            } else {
                None
            },
            xc,
        );
        let e_xc = deterministic_point_sum(npts, |g| {
            grid[g].weight * (dens.rho_a[g] + dens.rho_b[g]) * k.exc[g]
        });

        // ── Pass 2. Spin σ gates on its own ρ_σ, exactly as the dense
        // `semilocal_vxc_polarized_scratch` `build` closure does.
        let factors = |b: &XcBatch, mat: usize| -> BatchFactors {
            let alpha = mat == 0;
            let nb = b.idx.len();
            let (rho_s, vrho_s, vs_self, vtau_s, g_self, g_cross) = if alpha {
                (
                    &dens.rho_a,
                    &k.vrho_a,
                    &k.vsigma_aa,
                    &k.vtau_a,
                    &dens.grad_a,
                    &dens.grad_b,
                )
            } else {
                (
                    &dens.rho_b,
                    &k.vrho_b,
                    &k.vsigma_bb,
                    &k.vtau_b,
                    &dens.grad_b,
                    &dens.grad_a,
                )
            };
            let mut fac = BatchFactors {
                s_half: vec![0.0; nb],
                f: if has_gga {
                    [vec![0.0; nb], vec![0.0; nb], vec![0.0; nb]]
                } else {
                    [Vec::new(), Vec::new(), Vec::new()]
                },
                t_half: if has_mgga { vec![0.0; nb] } else { Vec::new() },
            };
            for (j, &gi) in b.idx.iter().enumerate() {
                let g = gi as usize;
                if rho_s[g] > DENSITY_FLOOR {
                    let w = grid[g].weight;
                    fac.s_half[j] = 0.5 * (w * vrho_s[g]);
                    if has_gga {
                        let vs = vs_self[g];
                        let vc = k.vsigma_ab[g];
                        for ax in 0..3 {
                            let mut f = 2.0 * w * vs * g_self[(ax, g)];
                            f += w * vc * g_cross[(ax, g)];
                            fac.f[ax][j] = f;
                        }
                    }
                    if has_mgga {
                        fac.t_half[j] = 0.5 * (0.5 * w * vtau_s[g]);
                    }
                }
            }
            fac
        };
        let mut v = self.potential_pass(grid, 2, &factors, (has_gga, has_mgga), exec, &mut dev)?;
        let v_b = v.pop().expect("two matrices");
        let v_a = v.pop().expect("two matrices");
        Ok((e_xc, v_a, v_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_points(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut x = seed;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) * 20.0 - 10.0
        };
        (0..n).map(|_| [next(), next(), next()]).collect()
    }

    fn water_dimer_far() -> ferric_core::mol::Molecule {
        ferric_core::mol::Molecule::parse_xyz(
            "6\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\n\
             O 9.0 0.0 0.0\nH 9.0 0.0 0.96\nH 9.93 0.0 -0.24\n",
            0,
            1,
        )
        .unwrap()
    }

    /// THE SCREENING INVARIANT, tested directly rather than through an energy
    /// tolerance: for every batch and every shell, recompute
    /// `max over the batch's points and the shell's functions of
    /// max(|χ|, |∂xχ|, |∂yχ|, |∂zχ|)` with the independent dense evaluator
    /// (`eval_basis_and_grad_on_points_unchecked`, all shells, all points of
    /// the batch) and assert: kept ⇔ max > thresh, dropped ⇔ max <= thresh.
    /// Exact comparison — both evaluators produce bit-identical values.
    ///
    /// Catches (energy tolerances cannot — measured: screening on χ only moves
    /// E_xc by just 1.5e-13 on this fixture): screening on χ without ∇χ, on
    /// only one gradient axis, on the wrong batch's points (e.g. a partition
    /// index mix-up), on a subset of a shell's functions, `>=` vs `>` drift,
    /// and a batch whose `funcs` list disagrees with its `shells` list.
    ///
    /// Reachability (so it cannot pass vacuously): asserts that shells ARE
    /// dropped, and that some kept shells are significant ONLY through their
    /// gradient (χ max <= thresh < ∇χ max) — the prototype counted 6-13 such
    /// (batch, shell) pairs on these fixtures at 1e-10. Covers s/p (6-31G) and
    /// pure-d (cc-pVDZ) shells.
    #[test]
    fn every_dropped_shell_is_below_threshold_and_every_kept_one_above() {
        use crate::ao_grid::{collect_shells, eval_basis_and_grad_on_points_unchecked, nbasis};
        use crate::grid::{build_atomic_grid, AtomicGridConfig};
        let mol = water_dimer_far();
        let cfg_grid = AtomicGridConfig {
            n_radial: 40,
            n_angular: 110,
            ..Default::default()
        };
        let grid = build_atomic_grid(&mol, &cfg_grid);
        for basis in ["6-31g", "cc-pvdz"] {
            let bs = ferric_core::basis::bundled(basis).unwrap();
            let located = collect_shells(&mol, &bs).unwrap();
            let nbf = nbasis(&mol, &bs).unwrap();
            let thresh = DEFAULT_SCREEN_THRESH;
            let sg = ScreenedGrid::build(
                &grid,
                owned_shells(&located),
                nbf,
                &XcBatchConfig::default(),
            )
            .unwrap();
            if basis == "cc-pvdz" {
                assert!(
                    sg.shells.iter().any(|s| s.l == 2 && s.pure),
                    "cc-pVDZ fixture must contain pure d shells"
                );
            }
            let (mut n_dropped, mut n_grad_only, mut n_pairs) = (0usize, 0usize, 0usize);
            for b in &sg.batches {
                let pts: Vec<[f64; 3]> = b.idx.iter().map(|&i| grid[i as usize].xyz).collect();
                let (chi, dchi) =
                    eval_basis_and_grad_on_points_unchecked(&located, nbf, &pts).unwrap();
                let mut expect_funcs = Vec::new();
                for (si, sh) in sg.shells.iter().enumerate() {
                    let mut m_chi = 0.0_f64;
                    let mut m_all = 0.0_f64;
                    for f in sh.offset..sh.offset + sh.nfunc {
                        for g in 0..pts.len() {
                            m_chi = m_chi.max(chi[(f, g)].abs());
                            m_all = m_all
                                .max(chi[(f, g)].abs())
                                .max(dchi[(0, f, g)].abs())
                                .max(dchi[(1, f, g)].abs())
                                .max(dchi[(2, f, g)].abs());
                        }
                    }
                    let kept = b.shells.contains(&(si as u32));
                    n_pairs += 1;
                    if kept {
                        assert!(
                            m_all > thresh,
                            "{basis}: kept shell {si} has max {m_all:e} <= {thresh:e}"
                        );
                        expect_funcs.extend((sh.offset..sh.offset + sh.nfunc).map(|f| f as u32));
                        if m_chi <= thresh {
                            n_grad_only += 1;
                        }
                    } else {
                        assert!(
                            m_all <= thresh,
                            "{basis}: DROPPED shell {si} (l={}) has max(|chi|,|grad chi|) \
                             {m_all:e} > {thresh:e} (|chi| max {m_chi:e})",
                            sh.l
                        );
                        n_dropped += 1;
                    }
                }
                assert_eq!(
                    b.funcs, expect_funcs,
                    "{basis}: funcs must match the kept shells"
                );
            }
            eprintln!(
                "{basis}: {n_dropped}/{n_pairs} (batch, shell) pairs dropped, \
                 {n_grad_only} kept only through the gradient"
            );
            assert!(n_dropped > 0, "{basis}: fixture must drop shells");
            assert!(
                n_grad_only > 0,
                "{basis}: fixture must contain shells significant only through ∇χ, \
                 or a χ-only screen would pass this test"
            );
        }
    }

    /// The host twin of the device `xc-vxc` kernel (f32 operands, f32 panels of
    /// 64 summed in f64) against the f64 `V_b = A·Xᵀ` on real batches of a
    /// screened grid (water/cc-pVDZ AO values, GGA-shaped `A`; meta-GGAs never
    /// run mixed): every element inside the documented bound
    /// `|ΔV_ij| ≤ (ε_mixed(ncols, 64) + γ_ncols(u64))·(|A||X|ᵀ)_ij`, and the
    /// twin not identical to f64 (the comparison is live). The potential
    /// factors in `A` are synthetic sine waves, not functional output, so the
    /// printed worst ratio to the bound (0.10 when measured) describes these
    /// operands, not an SCF.
    #[test]
    fn mixed_vxc_host_twin_stays_inside_its_elementwise_bound() {
        use ferric_core::gpu::mixed_host::{gamma, gemm_f32_f64acc_host, mixed_error_factor, U64};
        let mol = ferric_core::mol::Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
            0,
            1,
        )
        .unwrap();
        let bs = ferric_core::basis::bundled("cc-pvdz").unwrap();
        let cfg = crate::grid::AtomicGridConfig {
            n_radial: 30,
            n_angular: 50,
            ..Default::default()
        };
        let grid = crate::grid::build_atomic_grid_pruned(&mol, &cfg, None).unwrap();
        let shells = crate::ao_grid::collect_shells(&mol, &bs).unwrap();
        let nbf = crate::ao_grid::nbasis(&mol, &bs).unwrap();
        let sg = ScreenedGrid::build(&grid, owned_shells(&shells), nbf, &XcBatchConfig::default())
            .unwrap();
        let mut worst = 0.0_f64;
        let mut changed = false;
        for bi in (0..sg.batches.len()).step_by(7) {
            let b = &sg.batches[bi];
            let (na, nb) = (b.funcs.len(), b.idx.len());
            if na == 0 {
                continue;
            }
            let ao = sg.eval_batch(bi, &grid).unwrap();
            for mgga in [false] {
                let wave = |j: usize, s: f64| ((j as f64 * 0.37 + s).sin()) * 0.3;
                let fac = BatchFactors {
                    s_half: (0..nb).map(|j| wave(j, 0.1)).collect(),
                    f: [
                        (0..nb).map(|j| wave(j, 1.0)).collect(),
                        (0..nb).map(|j| wave(j, 2.0)).collect(),
                        (0..nb).map(|j| wave(j, 3.0)).collect(),
                    ],
                    t_half: (0..nb).map(|j| wave(j, 4.0).abs()).collect(),
                };
                let nc = if mgga { 4 * nb } else { nb };
                let mut a = Array2::<f64>::zeros((na, nc));
                fill_amat(&ao, nb, &fac, (true, mgga), a.as_slice_mut().unwrap());
                let x = ao.slice(s![.., ..nc]);
                let exact = a.dot(&x.t());
                let mut twin = Array2::<f64>::zeros((na, na));
                gemm_f32_f64acc_host(&a.view(), &x.t(), &mut twin.view_mut(), 64);
                let mag = a.mapv(f64::abs).dot(&x.mapv(f64::abs).t());
                let eps = (mixed_error_factor(nc, 64) + gamma(nc, U64)) * 1.01;
                for ((t, e), m) in twin.iter().zip(exact.iter()).zip(mag.iter()) {
                    let err = (t - e).abs();
                    assert!(err <= eps * m, "batch {bi}: |dV| {err:e} > {:e}", eps * m);
                    if *m > 0.0 {
                        worst = worst.max(err / (eps * m));
                    }
                    changed |= t != e;
                }
            }
        }
        assert!(
            changed,
            "the twin reproduced f64 exactly: the comparison is not live"
        );
        eprintln!("worst |dV| / bound = {worst:.3e}");
    }

    /// Catches: the serial execution mode (one-thread rayon pool, BLAS opt-in)
    /// drifting from the parallel mode — different group boundaries, a
    /// different batch order, or a missed batch. With BLAS at 1 thread the
    /// two must be bit-identical (see `Exec`). Meta-GGA, UKS and closed shell,
    /// screening on, cc-pVDZ.
    #[test]
    fn serial_and_parallel_execution_are_bit_identical() {
        use crate::ao_grid::{collect_shells, nbasis};
        use crate::grid::{build_atomic_grid, AtomicGridConfig};
        let mol = water_dimer_far();
        let bs = ferric_core::basis::bundled("cc-pvdz").unwrap();
        let grid = build_atomic_grid(
            &mol,
            &AtomicGridConfig {
                n_radial: 30,
                n_angular: 50,
                ..Default::default()
            },
        );
        let located = collect_shells(&mol, &bs).unwrap();
        let nbf = nbasis(&mol, &bs).unwrap();
        let mut sg = ScreenedGrid::build(
            &grid,
            owned_shells(&located),
            nbf,
            &XcBatchConfig::default(),
        )
        .unwrap();
        sg.materialize(&grid).unwrap();
        // A symmetric, positive "density matrix": only the arithmetic path is
        // under test here, not the physics.
        let c = Array2::from_shape_fn((nbf, 6), |(i, j)| {
            ((i * 7 + j * 3) % 11) as f64 * 0.05 - 0.2
        });
        let d = 2.0 * c.dot(&c.t());
        let d_b = 0.5 * &d;
        for name in ["PBE", "SCAN"] {
            let xc1 = crate::libxc::xc_def_from_name(name).unwrap();
            let (e_p, v_p) = sg
                .integrate_closed_exec(&grid, DensityInput::Dense(&d), &xc1, Exec::parallel())
                .unwrap();
            let (e_s, v_s) = sg
                .integrate_closed_exec(
                    &grid,
                    DensityInput::Dense(&d),
                    &xc1,
                    Exec::serial_single_blas(),
                )
                .unwrap();
            assert_eq!(e_p.to_bits(), e_s.to_bits(), "{name} closed E");
            assert!(v_p
                .iter()
                .zip(v_s.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            let xc2 = crate::libxc::xc_def_from_name_nspin(name, 2).unwrap();
            let (e_p, a_p, b_p) = sg
                .integrate_polarized_exec(
                    &grid,
                    DensityInput::Dense(&d),
                    DensityInput::Dense(&d_b),
                    &xc2,
                    Exec::parallel(),
                )
                .unwrap();
            let (e_s, a_s, b_s) = sg
                .integrate_polarized_exec(
                    &grid,
                    DensityInput::Dense(&d),
                    DensityInput::Dense(&d_b),
                    &xc2,
                    Exec::serial_single_blas(),
                )
                .unwrap();
            assert_eq!(e_p.to_bits(), e_s.to_bits(), "{name} uks E");
            assert!(a_p
                .iter()
                .zip(a_s.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(b_p
                .iter()
                .zip(b_s.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    /// Catches: a partition that drops or duplicates a point (its weight would
    /// vanish from, or double in, every integral), or a batch above the cap.
    #[test]
    fn partition_covers_every_point_exactly_once_within_the_cap() {
        for &(n, cap) in &[
            (0usize, 128usize),
            (1, 128),
            (128, 128),
            (129, 128),
            (5000, 128),
            (5000, 37),
            (777, 1),
        ] {
            let pts = lcg_points(n, 7 + n as u64);
            let parts = partition_points(&pts, cap);
            let mut seen = vec![0u32; n];
            for p in &parts {
                assert!(
                    !p.is_empty() && p.len() <= cap,
                    "batch size {} cap {cap}",
                    p.len()
                );
                assert!(
                    p.windows(2).all(|w| w[0] < w[1]),
                    "batch indices must ascend"
                );
                for &i in p {
                    seen[i as usize] += 1;
                }
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "n={n} cap={cap}: not a partition"
            );
        }
    }

    /// Catches: batches that are not spatially compact (the whole point of
    /// sorting — Becke order interleaves distant points). On a uniform random
    /// cloud of side 20, a 128-point batch must have a bounding box far below
    /// the cloud's; unsorted contiguous chunks span the whole cloud.
    #[test]
    fn partition_batches_are_spatially_compact() {
        let pts = lcg_points(20_000, 3);
        let parts = partition_points(&pts, 128);
        let mean_diag: f64 = parts
            .iter()
            .map(|p| {
                let mut lo = [f64::INFINITY; 3];
                let mut hi = [f64::NEG_INFINITY; 3];
                for &i in p {
                    for a in 0..3 {
                        lo[a] = lo[a].min(pts[i as usize][a]);
                        hi[a] = hi[a].max(pts[i as usize][a]);
                    }
                }
                ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt()
            })
            .sum::<f64>()
            / parts.len() as f64;
        // Cloud diagonal ~34.6; 20000/128 ≈ 156 cells of volume 8000/156 ≈ 51
        // → edge ~3.7, diagonal ~6.4. Allow a generous 2x.
        assert!(mean_diag < 13.0, "mean batch diagonal {mean_diag}");
        // Most batches should be full (the split is at multiples of the cap).
        let full = parts.iter().filter(|p| p.len() == 128).count();
        assert!(
            full * 10 >= parts.len() * 8,
            "{full}/{} full batches",
            parts.len()
        );
    }

    /// Catches: a thread-count- or run-dependent partition (would reorder the
    /// V reduction and break bit-identity across runs).
    #[test]
    fn partition_is_deterministic_across_thread_counts() {
        let pts = lcg_points(10_000, 11);
        let run = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| partition_points(&pts, 128))
        };
        assert_eq!(run(1), run(4));
    }

    /// Catches: a group count that reads the thread pool (would change the V
    /// summation order with the worker count), and group ranges that skip or
    /// repeat a batch.
    #[test]
    fn groups_are_a_pure_function_of_shape_and_cover_all_batches() {
        let run = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| n_groups(4577, 235, 1))
        };
        assert_eq!(run(1), run(8));
        for &(nb, ng) in &[(1usize, 64usize), (63, 64), (64, 64), (4577, 64), (10, 3)] {
            let r = group_ranges(nb, ng);
            assert!(r.len() <= ng.max(1));
            let mut next = 0;
            for &(a, b) in &r {
                assert_eq!(a, next);
                assert!(b > a);
                next = b;
            }
            assert_eq!(next, nb);
        }
        // Huge nbf shrinks the group count (memory cap) but never below 1.
        assert_eq!(n_groups(1000, 100_000, 2), 1);
    }
}
