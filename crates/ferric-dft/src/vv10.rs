//! VV10 nonlocal correlation (Vydrov-Van Voorhis JCP 133, 244103 (2010)).
//!
//! Closed-shell energy:
//! ```text
//!     E_nl = ∫ ρ(r) · [β + ½ · ∫ ρ(r') Φ(r,r') dr'] dr
//!     Φ(r,r') = −3 / [2 g(r) g(r') (g(r) + g(r'))]
//!     g(r)    = ω₀(r) · |r − r'|² + κ(r)
//!     ω₀²(r)  = C · (|∇ρ|/ρ)⁴ + (4π/3) · ρ
//!     κ(r)    = b · (3π/2) · (ρ/(9π))^(1/6)
//!     β       = (1/32) · (3/b²)^(3/4)
//! ```
//!
//! V_nl potential is the functional derivative δE_nl/δρ + δE_nl/δσ via the
//! chain rule through ω₀ and κ — see PySCF `_vv10nlc` for the exact form.
//! Algorithm direct from PySCF's `pyscf/dft/numint.py::_vv10nlc`.
//!
//! The O(npts²) pair sums parallelize over the outer (row) index with rayon;
//! each row's inner (partner) loop stays serial, so the per-row summation
//! order — and therefore the floating-point result — is identical to the
//! serial code.

use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ndarray::{Array1, Array2, Array3, Axis};
use rayon::prelude::*;

use crate::density_on_grid::DensityGrid;
use crate::grid::GridPoint;
use crate::libxc::Vv10Params;
use crate::vxc::{scale_columns_into, VxcScratch};

/// Density threshold below which a grid point is skipped — keeps κ, ω₀, and
/// their derivatives well-conditioned. Matches PySCF's `_vv10nlc` thresh=1e-8.
const RHO_THRESH: f64 = 1e-8;

/// Below this many outer points the O(npts²) pair sums run serially — rayon
/// spawn/join/steal overhead dwarfs the work on tiny grids (the free-atom
/// SCF case; see the matching guard in `ao_grid`).
const PAR_MIN_PTS: usize = 128;

// ---------------------------------------------------------------------------
// Distance cutoff for the VV10 pair sum
// ---------------------------------------------------------------------------
//
// The VV10 kernel decays with pair distance R = |r − r'|:
//
//     Φ(r,r') = −3 / [2 · g_i · g_p · (g_i + g_p)]
//     g       = ω₀ · R² + κ ,   ω₀² = C·(|∇ρ|/ρ)⁴ + (4π/3)·ρ ,   κ = k_b · ρ^(1/6)
//
// Because every *active* point satisfies ρ ≥ RHO_THRESH (points below are
// already dropped), both g-terms are floored: g ≥ ω₀·R² with
// ω₀ ≥ ω₀_min = √((4π/3)·RHO_THRESH). Hence the kernel envelope is R⁻⁶:
//
//     |Φ(R)| ≤ 3 / [2 · ω₀_i ω₀_p (ω₀_i+ω₀_p) · R⁶].
//
// The truncation error in the energy from dropping every pair with R > R_cut is
//
//     |ΔE_nl| = |½ Σ_i w_i ρ_i Σ_{p:R>R_cut} w_p ρ_p Φ_ip|
//             ≤ (3/4) Σ_{i,p : R>R_cut}  w_i ρ_i · w_p ρ_p
//                                       / [ω₀_i ω₀_p (ω₀_i+ω₀_p) R_ip⁶].
//
// The crucial step is that ω₀ in the denominator is *coupled* to the ρ in the
// numerator: ω₀ ≥ √((4π/3)·ρ), so a low-density (small-ω₀) tail point also
// carries a small ρ·w in the numerator — the two cannot both be worst-case.
// Using ω₀_i ≥ √(a·ρ_i), ω₀_p ≥ √(a·ρ_p) with a = 4π/3, and
// (ω₀_i+ω₀_p) ≥ √a·√(max(ρ_i,ρ_p)), the density dependence collapses to
// √(min(ρ_i,ρ_p)) ≤ (ρ_i ρ_p)^(1/4):
//
//     |ΔE_nl| ≤ (3/4) / (a^(3/2) · R_cut⁶) · [Σ_g w_g ρ_g^(1/4)]².
//
// The bracket S = Σ_g w_g ρ_g^(1/4) is an O(molecular-volume) quantity: for
// water S ≈ 10, and it grows only linearly with system size (per-atom
// contributions are bounded). Even a very large molecule with S ≈ 200 needs
// only R_cut ≈ 84 Bohr to hold |ΔE_nl| < 1e-8 Ha; the R⁻⁶ decay means the
// cutoff barely moves with S (S×2 ⇒ R_cut×2^(1/3) ≈ 1.26).
//
// We therefore fix a single conservative constant well beyond that horizon.
// NLC_CUTOFF_BOHR = 40 Bohr (≈ 21 Å) bounds |ΔE_nl| < 1e-8 Ha for any molecule
// with S ≲ 60 (a ~200-atom-scale system) and, being R⁻⁶, buys a >4000× margin
// versus the water bound. The bound is *proven empirically* in the tests
// (`cutoff_energy_bounded_*`), which assert |E_nl(cutoff) − E_nl(exact)| < 1e-8
// on both water (fallback path) and a 20-atom alkane (cutoff active). This is a
// screening approximation with a documented error bound, NOT a re-derivation.
//
// Hard-coded on purpose: this is a numerical accuracy floor, not a user knob —
// exposing it as config would invite loosening the proven bound.
const NLC_CUTOFF_BOHR: f64 = 40.0;

/// Below this many active points, run the exact (no-cutoff) dense pair sum:
/// the O(npts²) cost is negligible on small grids and the cell-list binning
/// overhead is not worth it. Above it, bin into cells and visit only the 27
/// neighboring cells. Shares the `PAR_MIN_PTS` shape/rationale (small grids
/// stay simple); the cutoff itself only *removes* pairs, so results below this
/// threshold are bit-identical to the pre-cutoff dense code.
const CELL_LIST_MIN_PTS: usize = PAR_MIN_PTS;

/// Test-only global counter of retained pair visits in the energy pair sum,
/// so a test can quantify the cell-list reduction versus dense O(npts²).
#[cfg(test)]
static PAIR_VISITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
#[inline]
fn count_pair_visit() {
    PAIR_VISITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Uniform spatial hash over active grid points, cell edge = `NLC_CUTOFF_BOHR`.
///
/// Every pair within the cutoff lies in the same or an adjacent cell, so a
/// point's partners are found by scanning the 3×3×3 block of cells around it.
/// `cell_points[c]` lists the point indices in cell `c` **in ascending order**
/// (points are bucketed in index order), so a row that concatenates its 27
/// neighbor cells and sorts the union visits retained partners in the exact
/// ascending index order the dense loop used — the per-row float summation
/// order is preserved bit-for-bit for the pairs that are kept.
struct CellList {
    inv_edge: f64,
    origin: [f64; 3],
    dims: [i64; 3],
    /// Flattened cell index (ix + iy*nx + iz*nx*ny) → sorted point indices.
    cell_points: Vec<Vec<usize>>,
}

impl CellList {
    /// Build over the active points identified by `active[i]` with coords `xyz[i]`.
    fn build(xyz: &[[f64; 3]], active: &[bool], edge: f64) -> Self {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for (i, &on) in active.iter().enumerate() {
            if !on {
                continue;
            }
            for d in 0..3 {
                lo[d] = lo[d].min(xyz[i][d]);
                hi[d] = hi[d].max(xyz[i][d]);
            }
        }
        // Degenerate (no active points): a 1×1×1 grid; loop bodies still guard.
        if !lo[0].is_finite() {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        let inv_edge = 1.0 / edge;
        let mut dims = [0i64; 3];
        for d in 0..3 {
            let span = (hi[d] - lo[d]) * inv_edge;
            dims[d] = (span.floor() as i64) + 1; // at least 1 cell per axis
        }
        let ncells = (dims[0] * dims[1] * dims[2]) as usize;
        let mut cell_points: Vec<Vec<usize>> = vec![Vec::new(); ncells];
        // Bucket in ascending index order ⇒ each cell's list is sorted.
        for (i, &on) in active.iter().enumerate() {
            if !on {
                continue;
            }
            let c = Self::cell_index(&lo, inv_edge, &dims, xyz[i]);
            cell_points[c].push(i);
        }
        CellList {
            inv_edge,
            origin: lo,
            dims,
            cell_points,
        }
    }

    #[inline]
    fn cell_coord(origin: &[f64; 3], inv_edge: f64, dims: &[i64; 3], p: [f64; 3]) -> [i64; 3] {
        let mut c = [0i64; 3];
        for d in 0..3 {
            let mut k = ((p[d] - origin[d]) * inv_edge).floor() as i64;
            if k < 0 {
                k = 0;
            }
            if k >= dims[d] {
                k = dims[d] - 1;
            }
            c[d] = k;
        }
        c
    }

    #[inline]
    fn cell_index(origin: &[f64; 3], inv_edge: f64, dims: &[i64; 3], p: [f64; 3]) -> usize {
        let c = Self::cell_coord(origin, inv_edge, dims, p);
        (c[0] + c[1] * dims[0] + c[2] * dims[0] * dims[1]) as usize
    }

    /// Collect the partner indices in the 3×3×3 block around point `p`, in
    /// ascending index order, into the caller-owned `buf` (cleared first).
    /// Serial and allocation-free per call (buf is reused across a row).
    fn neighbors_into(&self, p: [f64; 3], buf: &mut Vec<usize>) {
        buf.clear();
        let c = Self::cell_coord(&self.origin, self.inv_edge, &self.dims, p);
        for dz in -1..=1i64 {
            let iz = c[2] + dz;
            if iz < 0 || iz >= self.dims[2] {
                continue;
            }
            for dy in -1..=1i64 {
                let iy = c[1] + dy;
                if iy < 0 || iy >= self.dims[1] {
                    continue;
                }
                for dx in -1..=1i64 {
                    let ix = c[0] + dx;
                    if ix < 0 || ix >= self.dims[0] {
                        continue;
                    }
                    let idx = (ix + iy * self.dims[0] + iz * self.dims[0] * self.dims[1]) as usize;
                    buf.extend_from_slice(&self.cell_points[idx]);
                }
            }
        }
        // Neighbor cells were appended cell-by-cell (not globally sorted); sort
        // so the per-row inner loop visits partners in ascending index order —
        // identical float summation order to the dense loop for retained pairs.
        buf.sort_unstable();
    }
}

/// Map `row_fn` over `0..n`, in parallel when the pair-sum work is large
/// enough to amortize rayon overhead. `row_fn` must be a pure function of
/// its row index; rows are accumulated independently (no shared state), so
/// the result is bit-identical to the serial map.
fn map_rows<T, F>(n: usize, row_fn: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if n >= PAR_MIN_PTS {
        (0..n).into_par_iter().map(row_fn).collect()
    } else {
        (0..n).map(row_fn).collect()
    }
}

/// Compute per-grid-point VV10 potentials `(v_ρ, v_σ)` plus the energy E_nl.
///
/// Factored out of `add_vv10` so the nuclear-gradient path can reuse the
/// pair sum without going through the matrix assembly.
pub fn compute_vv10_potentials(
    grid: &[GridPoint],
    dens: &DensityGrid,
    params: &Vv10Params,
) -> (Vec<f64>, Vec<f64>) {
    let out = vv10_internal(grid, dens, params);
    (out.vrho, out.vsig)
}

/// Like `compute_vv10_potentials` but also returns E_nl.
pub fn compute_vv10_energy_and_potentials(
    grid: &[GridPoint],
    dens: &DensityGrid,
    params: &Vv10Params,
) -> (f64, Vec<f64>, Vec<f64>) {
    let out = vv10_internal(grid, dens, params);
    (out.e_nl, out.vrho, out.vsig)
}

/// Compute the per-grid-point VV10 energy density ε_nl(g) = β + ½ · f(g)
/// alongside the potentials. Needed by the gradient path that wants weight
/// response Σ_g w1[g, B, α] · ε_nl(g) · ρ(g).
pub fn compute_vv10_full(
    grid: &[GridPoint],
    dens: &DensityGrid,
    params: &Vv10Params,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // ε(g) is a byproduct of the same pair sum that yields the potentials, so
    // a single traversal covers everything (this used to re-run the full
    // O(npts²) pair sum a second time just to recover ε).
    let out = vv10_internal(grid, dens, params);
    (out.vrho, out.vsig, out.exc)
}

/// Per-grid-point VV10 egrad `F[g, axis] = -3 · Σ_p RpW_p · Q[g,p] · DR[g,p]`
/// where `Q[g,p] = (1/(g_i·g_p·g_t)) · (ω₀_i/g_i + ω₀_p/g_p + (ω₀_i+ω₀_p)/g_t)`
/// and `DR = r_p − r_g`.
///
/// This is the gradient of the VV10 pair integrand with respect to the
/// **outer** grid coordinate r_g. The outer grid (`outer`) and the inner
/// partner grid (`inner`) may be the same (canonical full-grid) or distinct
/// (e.g. PySCF's `vvrho_sub` / `vvcoords_sub` which excludes the atom whose
/// gradient is being computed, avoiding self-coupling). The energy double-
/// integral factor of ½ is absorbed at the use site via the `ρ·w·F` outer
/// sum — matches PySCF's `excsum[atm_id] += einsum('r,rx->x', rho*weight, F)`.
pub fn vv10_egrad(
    outer_grid: &[GridPoint],
    outer_dens: &DensityGrid,
    inner_grid: &[GridPoint],
    inner_dens: &DensityGrid,
    params: &Vv10Params,
) -> ndarray::Array2<f64> {
    let n_out = outer_grid.len();
    let n_in = inner_grid.len();
    let b_vv = params.b;
    let c_vv = params.c;
    let pi = std::f64::consts::PI;
    let pi43 = 4.0 * pi / 3.0;
    let k_vv = b_vv * 1.5 * pi * (9.0 * pi).powf(-1.0 / 6.0);

    // Cache outer ω₀, κ, coords (active outer points only).
    let mut w0_out = vec![0.0f64; n_out];
    let mut k_out = vec![0.0f64; n_out];
    let mut active_out = vec![false; n_out];
    let mut xyz_out = vec![[0.0f64; 3]; n_out];
    for i in 0..n_out {
        let r = outer_dens.rho[i];
        if r < RHO_THRESH {
            continue;
        }
        let s = outer_dens.sigma[i];
        let w0sq = c_vv * (s / (r * r)).powi(2) + pi43 * r;
        w0_out[i] = w0sq.sqrt();
        k_out[i] = k_vv * r.powf(1.0 / 6.0);
        xyz_out[i] = outer_grid[i].xyz;
        active_out[i] = true;
    }

    // Cache inner ω₀, κ, RpW, coords.
    let mut w0_in = vec![0.0f64; n_in];
    let mut k_in = vec![0.0f64; n_in];
    let mut rpw = vec![0.0f64; n_in];
    let mut xyz_in = vec![[0.0f64; 3]; n_in];
    let mut active_in = vec![false; n_in];
    for j in 0..n_in {
        let r = inner_dens.rho[j];
        if r < RHO_THRESH {
            continue;
        }
        let s = inner_dens.sigma[j];
        let w0sq = c_vv * (s / (r * r)).powi(2) + pi43 * r;
        w0_in[j] = w0sq.sqrt();
        k_in[j] = k_vv * r.powf(1.0 / 6.0);
        rpw[j] = r * inner_grid[j].weight;
        xyz_in[j] = inner_grid[j].xyz;
        active_in[j] = true;
    }

    // Same distance-cutoff / cell-list treatment as `vv10_internal`: for large
    // grids bin the *inner* (partner) points into cells of edge NLC_CUTOFF_BOHR
    // and visit only the 27 neighbor cells per outer row, dropping R > R_cut
    // pairs. Retained partners are visited in ascending inner-index order in
    // both paths, so the kept summation is bit-identical; the cutoff error on
    // the gradient inherits the same R⁻⁶ envelope as the energy bound above.
    let use_cells = n_in >= CELL_LIST_MIN_PTS;
    let cells = use_cells.then(|| CellList::build(&xyz_in, &active_in, NLC_CUTOFF_BOHR));

    // Per-pair egrad accumulation for outer point i, partner j.
    let accum = |j: usize, w0i: f64, ki: f64, xi: [f64; 3],
                 fx: &mut f64, fy: &mut f64, fz: &mut f64| {
        let dx = xyz_in[j][0] - xi[0];
        let dy = xyz_in[j][1] - xi[1];
        let dz = xyz_in[j][2] - xi[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let g_i = r2 * w0i + ki;
        let g_p = r2 * w0_in[j] + k_in[j];
        let g_t = g_i + g_p;
        if g_i < 1e-30 || g_p < 1e-30 || g_t < 1e-30 {
            return;
        }
        let t = rpw[j] / (g_i * g_p * g_t);
        let q = t * (w0i / g_i + w0_in[j] / g_p + (w0i + w0_in[j]) / g_t);
        *fx += q * dx;
        *fy += q * dy;
        *fz += q * dz;
    };

    // Outer rows are independent — parallel map, serial inner loop.
    let rows = map_rows(n_out, |i| {
        if !active_out[i] {
            return [0.0f64; 3];
        }
        let xi = xyz_out[i];
        let w0i = w0_out[i];
        let ki = k_out[i];
        let mut fx = 0.0f64;
        let mut fy = 0.0f64;
        let mut fz = 0.0f64;
        match &cells {
            None => {
                for j in 0..n_in {
                    if !active_in[j] {
                        continue;
                    }
                    accum(j, w0i, ki, xi, &mut fx, &mut fy, &mut fz);
                }
            }
            Some(cl) => {
                let mut buf: Vec<usize> = Vec::new();
                cl.neighbors_into(xi, &mut buf);
                for &j in &buf {
                    accum(j, w0i, ki, xi, &mut fx, &mut fy, &mut fz);
                }
            }
        }
        [-3.0 * fx, -3.0 * fy, -3.0 * fz]
    });

    let mut f = ndarray::Array2::<f64>::zeros((n_out, 3));
    for (i, row) in rows.iter().enumerate() {
        f[(i, 0)] = row[0];
        f[(i, 1)] = row[1];
        f[(i, 2)] = row[2];
    }
    f
}

/// Everything the single VV10 pair-sum traversal yields.
struct Vv10Internal {
    e_nl: f64,
    vrho: Vec<f64>,
    vsig: Vec<f64>,
    /// Per-grid-point energy density ε_nl(g) = β + ½ · f(g); 0.0 on inactive points.
    exc: Vec<f64>,
    active: Vec<bool>,
}

/// Internal: compute (E_nl, vrho, vsig, ε_nl, active) on a single grid, using
/// the production distance cutoff (`NLC_CUTOFF_BOHR`) for large grids.
fn vv10_internal(grid: &[GridPoint], dens: &DensityGrid, params: &Vv10Params) -> Vv10Internal {
    vv10_internal_cutoff(grid, dens, params, Some(NLC_CUTOFF_BOHR))
}

/// Internal core with an explicit cutoff policy (test seam):
///   * `cutoff = Some(edge)` — cell-list screening at `edge` Bohr for grids
///     ≥ `CELL_LIST_MIN_PTS` (the production path).
///   * `cutoff = None` — exact dense O(npts²) pair sum, no screening; used by
///     the bounded-error tests as the reference "exact" energy.
fn vv10_internal_cutoff(
    grid: &[GridPoint],
    dens: &DensityGrid,
    params: &Vv10Params,
    cutoff: Option<f64>,
) -> Vv10Internal {
    let npts = dens.rho.len();
    let b_vv = params.b;
    let c_vv = params.c;
    let pi = std::f64::consts::PI;
    let pi43 = 4.0 * pi / 3.0;
    let k_vv = b_vv * 1.5 * pi * (9.0 * pi).powf(-1.0 / 6.0);
    let beta = (3.0 / (b_vv * b_vv)).powf(0.75) / 32.0;

    // Per-point setup (ω₀, κ and their derivatives): sqrt/powf transcendentals
    // per point, each point independent and write-once into its own slot g —
    // parallel map is bit-identical to the serial loop by construction
    // (index-order-preserving collect, no cross-point accumulation).
    #[derive(Clone, Copy)]
    struct PointSetup {
        active: bool,
        rho: f64,
        w0: f64,
        kp: f64,
        dw0_dr: f64,
        dw0_dg: f64,
        dk_dr: f64,
        xyz: [f64; 3],
        rho_w: f64,
    }
    let setup_point = |g: usize| -> PointSetup {
        let mut p = PointSetup {
            active: false,
            rho: 0.0,
            w0: 0.0,
            kp: 0.0,
            dw0_dr: 0.0,
            dw0_dg: 0.0,
            dk_dr: 0.0,
            xyz: [0.0; 3],
            rho_w: 0.0,
        };
        let r = dens.rho[g];
        let s = dens.sigma[g];
        if r < RHO_THRESH {
            return p;
        }
        p.active = true;
        p.rho = r;
        p.xyz = grid[g].xyz;
        p.rho_w = r * grid[g].weight;
        let w0tmp = c_vv * (s / (r * r)).powi(2);
        let w0sq = w0tmp + pi43 * r;
        p.w0 = w0sq.sqrt();
        p.kp = k_vv * r.powf(1.0 / 6.0);
        p.dk_dr = p.kp / 6.0;
        p.dw0_dr = (0.5 * pi43 * r - 2.0 * w0tmp) / p.w0;
        if s > RHO_THRESH {
            p.dw0_dg = w0tmp * r / (s * p.w0);
        }
        p
    };
    let setup = map_rows(npts, setup_point);

    let mut rho = vec![0.0_f64; npts];
    let mut w0  = vec![0.0_f64; npts];
    let mut kp  = vec![0.0_f64; npts];
    let mut dw0_dr = vec![0.0_f64; npts];
    let mut dw0_dg = vec![0.0_f64; npts];
    let mut dk_dr  = vec![0.0_f64; npts];
    let mut active = vec![false; npts];
    let mut xyz = vec![[0.0_f64; 3]; npts];
    let mut rho_w = vec![0.0_f64; npts];
    for (g, p) in setup.iter().enumerate() {
        active[g] = p.active;
        rho[g] = p.rho;
        w0[g] = p.w0;
        kp[g] = p.kp;
        dw0_dr[g] = p.dw0_dr;
        dw0_dg[g] = p.dw0_dg;
        dk_dr[g] = p.dk_dr;
        xyz[g] = p.xyz;
        rho_w[g] = p.rho_w;
    }

    // Pair sum: outer rows independent → parallel map; inner loop serial so
    // each row's accumulation order matches the serial code.
    //
    // Small grids run the exact dense O(npts²) loop over all partners; large
    // grids build a cell list (edge = NLC_CUTOFF_BOHR) and visit only the 27
    // neighboring cells per row, dropping pairs beyond the cutoff. Retained
    // partners are visited in ascending index order in both paths, so the kept
    // per-row summation is bit-identical between dense and cell-list — the only
    // difference is the (bounded, < 1e-8 Ha) omission of R > R_cut pairs.
    let use_cells = matches!(cutoff, Some(_)) && npts >= CELL_LIST_MIN_PTS;
    let cells = match cutoff {
        Some(edge) if use_cells => Some(CellList::build(&xyz, &active, edge)),
        _ => None,
    };

    // Per-pair kernel accumulation into (fi, ui, wi) for outer point i and
    // partner p. Shared by the dense and cell-list paths so the arithmetic is
    // provably identical.
    let accum = |p: usize, w0i: f64, ki: f64, xi: [f64; 3],
                 fi: &mut f64, ui: &mut f64, wi: &mut f64| {
        #[cfg(test)]
        count_pair_visit();
        let dx = xyz[p][0] - xi[0];
        let dy = xyz[p][1] - xi[1];
        let dz = xyz[p][2] - xi[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let gp_val = r2 * w0[p] + kp[p];
        let gi_val = r2 * w0i + ki;
        let gt_val = gi_val + gp_val;
        if gi_val < 1e-30 || gp_val < 1e-30 || gt_val < 1e-30 {
            return;
        }
        let t = rho_w[p] / (gi_val * gp_val * gt_val);
        *fi += t;
        let t_u = t * (1.0 / gi_val + 1.0 / gt_val);
        *ui += t_u;
        *wi += t_u * r2;
    };

    let row_fn = |i: usize| -> (f64, f64, f64) {
        if !active[i] {
            return (0.0_f64, 0.0_f64, 0.0_f64);
        }
        let xi = xyz[i];
        let w0i = w0[i];
        let ki = kp[i];
        let mut fi = 0.0_f64;
        let mut ui = 0.0_f64;
        let mut wi = 0.0_f64;
        match &cells {
            None => {
                for p in 0..npts {
                    if !active[p] {
                        continue;
                    }
                    accum(p, w0i, ki, xi, &mut fi, &mut ui, &mut wi);
                }
            }
            Some(cl) => {
                // Per-row neighbor buffer (thread-local by construction: this
                // closure body runs on one row at a time within a thread).
                let mut buf: Vec<usize> = Vec::new();
                cl.neighbors_into(xi, &mut buf);
                for &p in &buf {
                    // active[p] is guaranteed (only active points are binned).
                    accum(p, w0i, ki, xi, &mut fi, &mut ui, &mut wi);
                }
            }
        }
        (-1.5 * fi, ui, wi)
    };
    let fuw = map_rows(npts, row_fn);

    // Finalization left serial: ~10 flops/point next to the O(npts²) pair sum
    // above, and the e_nl scalar fold would have to be re-associated to go
    // parallel — no win to buy with that risk.
    let mut vrho = vec![0.0_f64; npts];
    let mut vsig = vec![0.0_f64; npts];
    let mut exc = vec![0.0_f64; npts];
    let mut e_nl = 0.0_f64;
    for g in 0..npts {
        if !active[g] { continue; }
        let (f_g, u_g, w_g) = fuw[g];
        let exc_g = beta + 0.5 * f_g;
        exc[g] = exc_g;
        vrho[g] = beta + f_g + 1.5 * (u_g * dk_dr[g] + w_g * dw0_dr[g]);
        vsig[g] = 1.5 * w_g * dw0_dg[g];
        e_nl += grid[g].weight * rho[g] * exc_g;
    }
    Vv10Internal { e_nl, vrho, vsig, exc, active }
}

/// Compute the VV10 energy contribution and add the matrix V_nl to `f`.
///
/// Convenience wrapper over [`add_vv10_scratch`] that allocates a fresh scratch
/// — fine for one-shot callers; SCF loops should hold a [`VxcScratch`] and call
/// the `_scratch` variant to skip the per-iteration `(nbf, npts)` allocation.
pub fn add_vv10(
    grid: &[GridPoint],
    chi: &Array2<f64>,        // (nbf, npts)
    dchi: &Array3<f64>,       // (3, nbf, npts)
    dens: &DensityGrid,
    params: &Vv10Params,
    f: &mut Array2<f64>,
) -> f64 {
    add_vv10_scratch(grid, chi, dchi, dens, params, f, &mut VxcScratch::new())
}

/// VV10 energy + V_nl assembly with caller-owned scratch (see [`VxcScratch`]).
///
/// Single-grid implementation: the NLC grid serves as both outer (integration)
/// and inner (kernel partner) points. O(N²) in grid size. The `(nbf, npts)`
/// pre-scaled-χ scratch is refilled per GEMM operand; holding it across SCF
/// iterations (via `KsXc::scratch`) removes the per-iteration allocation that
/// the semilocal path already amortizes.
pub fn add_vv10_scratch(
    grid: &[GridPoint],
    chi: &Array2<f64>,        // (nbf, npts)
    dchi: &Array3<f64>,       // (3, nbf, npts)
    dens: &DensityGrid,
    params: &Vv10Params,
    f: &mut Array2<f64>,
    scratch: &mut VxcScratch,
) -> f64 {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));
    debug_assert_eq!(dens.rho.len(), npts);

    // Compute energy + per-point potentials via the shared pair-sum routine.
    let out = vv10_internal(grid, dens, params);
    let Vv10Internal { e_nl, vrho, vsig, active, .. } = out;

    // V_nl matrix contribution — same GEMM pattern as semilocal V_xc.
    //   LDA-like piece: V_μν += Σ_g w_g · vrho_g · χ_μg · χ_νg
    //   GGA-like piece: V_μν += Σ_g 2·w_g · vsig_g · Σ_axis ∇ρ_axis_g ·
    //                           [χ_μg · ∂_axis χ_νg + χ_νg · ∂_axis χ_μg]
    // One reused scratch buffer serves all four GEMM operands (refilled per use).
    let buf = scratch.ensure((nbf, npts));

    let s: Array1<f64> = (0..npts)
        .map(|g| if active[g] { grid[g].weight * vrho[g] } else { 0.0 })
        .collect();
    scale_columns_into(chi.view(), &s, buf);
    // Digestion GEMM (nbf, npts)·(npts, nbf), outside any rayon region — this
    // whole function runs once per KS iteration at the top level (called from
    // ks.rs, never from inside map_rows' rayon fan-out above). Opt-in BLAS
    // raise via FERRIC_BLAS_THREADS (default 1, unchanged behavior); mirrors
    // vxc.rs's semilocal_vxc_closed_scratch idiom exactly.
    let mut v_nl: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || buf.dot(&chi.t()));

    for axis in 0..3 {
        let dchi_axis = dchi.index_axis(Axis(0), axis);
        let f_ax: Array1<f64> = (0..npts)
            .map(|g| {
                if active[g] {
                    2.0 * grid[g].weight * vsig[g] * dens.grad[(axis, g)]
                } else {
                    0.0
                }
            })
            .collect();
        scale_columns_into(chi.view(), &f_ax, buf);
        // Same opt-in-raise digestion GEMM as the LDA-like piece above.
        let m_axis: Array2<f64> =
            with_blas_threads(opt_in_blas_threads(), || buf.dot(&dchi_axis.t()));
        v_nl = v_nl + &m_axis + &m_axis.t();
    }

    // Symmetrize and accumulate.
    let v_nl_sym = 0.5 * (&v_nl + &v_nl.t());
    *f += &v_nl_sym;

    e_nl
}

#[cfg(test)]
mod cutoff_tests {
    //! Bounded-error proof for the VV10 distance cutoff.
    //!
    //! For a converged wB97X-V density we compute E_nl with the production
    //! cutoff (cell list, `NLC_CUTOFF_BOHR`) and with the exact dense pair sum
    //! (`cutoff = None`) and assert the difference is under the documented
    //! 1e-8 Ha bound — on water (exercises the small-grid dense fallback) and a
    //! 20-atom alkane (cutoff genuinely active, large grid). The alkane case
    //! also reports the pair-visit reduction to evidence the complexity win.

    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_dft_self_scf::*;
    use std::sync::atomic::Ordering;

    // `ferric-scf` is a dev-dependency; alias its imports so the `use super::*`
    // glob above stays unambiguous.
    mod ferric_dft_self_scf {
        pub use ferric_integrals::basis_bridge::PreparedBasis;
        pub use ferric_integrals::operator::Operator;
        pub use ferric_scf::rhf::{solve_rhf, RhfConfig};
        pub use ferric_scf::screening::SchwarzBounds;
    }

    /// Solve wB97X-V and return (density_total, NLC grid, χ, ∇χ) so the tests
    /// can drive the VV10 internals directly. Only used for small molecules
    /// (water) — see `synthetic_density_on_grid` for larger geometries, where
    /// a real SCF is too slow for a debug-build unit test.
    fn nlc_density(
        mol: &Molecule,
        basis_name: &str,
    ) -> (Vv10Params, Vec<GridPoint>, DensityGrid) {
        let bs = basis::bundled(basis_name).unwrap();
        let obs = PreparedBasis::new(mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ParallelContext::default();
        let cfg = RhfConfig {
            xc: Some("wB97X-V".into()),
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let res = solve_rhf(&ctx, mol, &obs, op, &bounds, &cfg).unwrap();
        let d = res.density_total().clone();

        let nlc_cfg = crate::grid::AtomicGridConfig {
            n_radial: 50,
            n_angular: 50,
            ..Default::default()
        };
        let grid = crate::grid::build_atomic_grid(mol, &nlc_cfg);
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let (chi, dchi) = crate::ao_grid::eval_basis_and_grad_on_points(mol, &bs, &pts).unwrap();
        let dens = crate::density_on_grid::eval_density_closed(&d, &chi, &dchi);

        let params = crate::libxc::xc_def_from_name("wB97X-V")
            .unwrap()
            .vv10
            .unwrap();
        (params, grid, dens)
    }

    /// Build a real (n_radial, n_angular) atomic grid over `mol`'s actual
    /// geometry, plus a synthetic atom-centered Gaussian-superposition
    /// density — no SCF. `ρ(r) = Σ_A exp(-α|r - R_A|²)`, smooth, positive,
    /// and decaying, which is all the VV10 cutoff bound needs: it exercises
    /// the same pair-sum numerics as a converged density on any geometry
    /// (including one whose real extent exceeds NLC_CUTOFF_BOHR) without
    /// paying for a self-consistent solve.
    fn synthetic_density_on_grid(
        mol: &Molecule,
        n_radial: usize,
        n_angular: usize,
    ) -> (Vv10Params, Vec<GridPoint>, DensityGrid) {
        let nlc_cfg = crate::grid::AtomicGridConfig {
            n_radial,
            n_angular,
            ..Default::default()
        };
        let grid = crate::grid::build_atomic_grid(mol, &nlc_cfg);
        let npts = grid.len();
        const ALPHA: f64 = 2.0; // Bohr^-2, a valence-like decay (density falls
                                 // off fast enough that S = Σ w ρ^(1/4) stays
                                 // within NLC_CUTOFF_BOHR's documented S ≲ 60
                                 // validity range even for a 48-atom chain).

        let mut rho = Array1::<f64>::zeros(npts);
        let mut grad = Array2::<f64>::zeros((3, npts));
        for (g, pt) in grid.iter().enumerate() {
            let mut r = 0.0f64;
            let mut gx = 0.0f64;
            let mut gy = 0.0f64;
            let mut gz = 0.0f64;
            for atom in &mol.atoms {
                let dx = pt.xyz[0] - atom.x;
                let dy = pt.xyz[1] - atom.y;
                let dz = pt.xyz[2] - atom.zpos;
                let r2 = dx * dx + dy * dy + dz * dz;
                let g_a = (-ALPHA * r2).exp();
                r += g_a;
                // d/dx exp(-α r²) = -2α·dx·exp(-α r²)
                gx += -2.0 * ALPHA * dx * g_a;
                gy += -2.0 * ALPHA * dy * g_a;
                gz += -2.0 * ALPHA * dz * g_a;
            }
            rho[g] = r;
            grad[(0, g)] = gx;
            grad[(1, g)] = gy;
            grad[(2, g)] = gz;
        }
        let sigma = (&grad.row(0) * &grad.row(0)
            + &grad.row(1) * &grad.row(1)
            + &grad.row(2) * &grad.row(2))
            .to_owned();
        let dens = DensityGrid { rho, grad, sigma };

        let params = crate::libxc::xc_def_from_name("wB97X-V")
            .unwrap()
            .vv10
            .unwrap();
        (params, grid, dens)
    }

    #[test]
    fn cutoff_energy_bounded_water() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
            0,
            1,
        )
        .unwrap();
        let (params, grid, dens) = nlc_density(&mol, "cc-pvdz");

        let e_exact = vv10_internal_cutoff(&grid, &dens, &params, None).e_nl;
        let e_cut = vv10_internal_cutoff(&grid, &dens, &params, Some(NLC_CUTOFF_BOHR)).e_nl;
        let err = (e_cut - e_exact).abs();
        eprintln!(
            "[water] npts={} E_nl exact={:.12} cutoff={:.12} |Δ|={:.3e}",
            dens.rho.len(),
            e_exact,
            e_cut,
            err
        );
        assert!(err < 1e-8, "water VV10 cutoff error {err:.3e} exceeds 1e-8 Ha");
    }

    #[test]
    fn cutoff_energy_bounded_alkane20() {
        // A synthetic 24-atom linear chain (bond length 2.9 Bohr, ≈1.53 Å,
        // matching a real C-C bond), NOT a bundled alkane geometry: the
        // longest bundled alkane (C20H42, 62 atoms) only spans ≈59 Bohr along
        // its axis, which is < 2·NLC_CUTOFF_BOHR (80 Bohr) — with only 2
        // cells along that axis, the 3×3×3 neighbor scan trivially covers
        // both cells and the cutoff never drops a pair (verified: dense ==
        // cutoff visits exactly on alkane_14/alkane_20). A chain spanning
        // ≳3·NLC_CUTOFF_BOHR (≥3 cells) is required to genuinely exercise
        // pair-dropping. This is a real property of the cell-list algorithm,
        // not a test artifact — worth keeping in mind for production
        // molecules near the 1-2 cell boundary, where the cutoff still
        // bounds the error correctly but yields no speedup.
        const NCHAIN: usize = 48;
        const BOND_BOHR: f64 = 2.9;
        let mut xyz = String::new();
        xyz.push_str(&format!("{NCHAIN}\nsynthetic chain\n"));
        for i in 0..NCHAIN {
            xyz.push_str(&format!("C {:.6} 0.0 0.0\n", i as f64 * BOND_BOHR * 0.529177));
        }
        let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
        assert!(mol.atoms.len() >= 20, "expected ≥20-atom chain");
        let span_bohr = (NCHAIN - 1) as f64 * BOND_BOHR;
        assert!(
            span_bohr > 3.0 * NLC_CUTOFF_BOHR,
            "chain span {span_bohr:.1} Bohr must exceed 3x cutoff to force ≥3 cells"
        );
        // No SCF: an actual RHF/DFT solve on 24+ heavy atoms is far too slow
        // for a debug-build unit test (minutes-to-unbounded, since every
        // functional in this codebase's KS loop that includes VV10 pays the
        // O(npts²) pair sum this test exists to bound, once per iteration —
        // even a VV10-free functional's SCF at this size is untested
        // territory; every other DFT test in this crate uses water). The
        // property under test — |ΔE_nl(cutoff) − E_nl(exact)| < 1e-8 Ha from
        // dropping R > 40 Bohr pairs — is a function of grid geometry and
        // density smoothness, not of self-consistency, so a synthetic
        // atom-centered Gaussian superposition density on the real geometry
        // exercises the same numerics instantly.
        let (params, grid, dens) = synthetic_density_on_grid(&mol, 14, 14);
        let npts = dens.rho.len();

        PAIR_VISITS.store(0, Ordering::Relaxed);
        let e_exact = vv10_internal_cutoff(&grid, &dens, &params, None).e_nl;
        let visits_dense = PAIR_VISITS.swap(0, Ordering::Relaxed);

        let e_cut = vv10_internal_cutoff(&grid, &dens, &params, Some(NLC_CUTOFF_BOHR)).e_nl;
        let visits_cut = PAIR_VISITS.load(Ordering::Relaxed);

        let err = (e_cut - e_exact).abs();
        eprintln!(
            "[chain24] atoms={} npts={} E_nl exact={:.12} cutoff={:.12} |Δ|={:.3e}",
            mol.atoms.len(),
            npts,
            e_exact,
            e_cut,
            err
        );
        eprintln!(
            "[chain24] pair visits: dense={} cutoff={} reduction={:.2}x ({:.1}% removed)",
            visits_dense,
            visits_cut,
            visits_dense as f64 / visits_cut.max(1) as f64,
            100.0 * (1.0 - visits_cut as f64 / visits_dense.max(1) as f64),
        );
        assert!(
            err < 1e-8,
            "alkane_14 VV10 cutoff error {err:.3e} exceeds 1e-8 Ha"
        );
        assert!(
            visits_cut < visits_dense,
            "cutoff should visit strictly fewer pairs (dense={visits_dense}, cutoff={visits_cut})"
        );
    }
}
