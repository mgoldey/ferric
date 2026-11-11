//! Spatially batched, shell-screened, rayon-parallel XC nuclear gradient — the
//! production path behind every `xc_gradient_*_from_density` entry point in
//! [`super`].
//!
//! # Why this exists
//!
//! The whole-grid gradient (`gga_closed_batch` and its siblings, still used
//! for the width-pinned [`super::xc_gradient_closed_gga_from_density_batched`]
//! hook) evaluates χ, ∇χ and ∇∇χ as thirteen dense `(nbf, npts)` planes over
//! the WHOLE grid, runs its four `D·χ`-class GEMMs on ONE BLAS thread, and makes
//! ~40 full-plane passes (fresh `scale_cols` copies, `row_dot`, strided
//! `col_dot`). MEASURED on sulfamethoxazole (28 atoms, PBE/def2-SVP, 6 rayon
//! threads): the XC term was 12-15 s of a 14.7 s gradient, against 1.7 s for
//! the SCF's whole V_xc build on the same grid, which already runs on
//! [`crate::xc_batch`]. This module puts the gradient on the same footing:
//!
//! 1. **Spatial batches** — [`crate::xc_batch::partition_points`] (the SCF's own
//!    bisection), at most `max_batch_pts` points each.
//! 2. **Shell screening per batch** — a shell is kept iff the largest of
//!    `|χ|`, `|∇χ|` and (for every path that contracts it) `|∇∇χ|` over the
//!    batch's points exceeds `screen_thresh`. The decision uses the ACTUAL
//!    values (the evaluation the batch needs anyway), behind a rigorous
//!    distance prefilter ([`crate::xc_batch::shell_screen_radii2_hess`]) that
//!    only skips shells which provably fail it. `screen_thresh <= 0` keeps
//!    every shell — the exactness anchor.
//! 3. **Compact per-batch algebra** — the batch's χ planes are `(nact, nb)`,
//!    `D` is gathered to `(nact, nact)`. The AO-derivative term and the
//!    grid-coordinate response contract the SAME per-(μ, g) integrand, one
//!    reduced over points (per basis function), one over functions (per
//!    point), so `fused_xc_reductions` builds it once and reduces it both
//!    ways, in place of the whole-grid path's `gga_ao_partials` /
//!    `mgga_tau_partials` row dots and its strided `hess_cols` /
//!    `tau_spatial_grad` column dots. Each term is the same expression over
//!    the same points; only the association of the sums differs (bound
//!    derived and tested in `fused_vs_legacy_tests`).
//! 4. **Parallel over batches, fixed-order reduction** — batches are split into
//!    `min(GRAD_TARGET_GROUPS, nbatches)` contiguous groups (a pure function of
//!    the grid), each group sums its batches in ascending order on one worker
//!    with its own libxc handles, and the group results are summed in
//!    ascending group order. No per-thread accumulator exists, so the gradient
//!    is bit-identical across rayon thread counts.
//!
//! The result is NOT bit-identical to the whole-grid path: every grid sum is
//! re-associated (per-batch partial sums, small GEMMs), and screening drops
//! terms below `screen_thresh`. Both effects are measured, not assumed, in
//! `crates/ferric-scf/tests/ks_gradient_screened.rs`.

use super::*;
use crate::ao_grid::{eval_shell_and_grad, eval_shell_grad_hess, GtoEvalError};
use crate::xc_batch::{
    gather_sub, group_ranges, owned_shells, partition_points, points_box, shell_screen_radii2,
    shell_screen_radii2_hess, OwnedShell, DEFAULT_MAX_BATCH_PTS, SCREEN_DIST2_MARGIN,
};

/// Default per-batch shell-screening threshold of the XC gradient, on
/// `max(|χ|, |∇χ|, |∇∇χ|)` — the SCF's [`crate::xc_batch::DEFAULT_SCREEN_THRESH`]
/// value, applied to the larger set of quantities the gradient contracts.
pub const DEFAULT_GRAD_SCREEN_THRESH: f64 = crate::xc_batch::DEFAULT_SCREEN_THRESH;

/// Number of fixed reduction groups (upper bound; never more than the batch
/// count). A pure function of the grid, never of the thread count.
const GRAD_TARGET_GROUPS: usize = 64;

/// Knobs of the screened gradient. `Default` is production.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradScreenConfig {
    /// Shell kept for a batch iff `max(|χ|, |∇χ|, |∇∇χ|)` over the batch exceeds
    /// this (`|∇∇χ|` only on paths that contract it). `<= 0` keeps every shell.
    pub screen_thresh: f64,
    /// Maximum points per spatial batch (>= 1).
    pub max_batch_pts: usize,
}

impl Default for GradScreenConfig {
    fn default() -> Self {
        Self {
            screen_thresh: DEFAULT_GRAD_SCREEN_THRESH,
            max_batch_pts: DEFAULT_MAX_BATCH_PTS,
        }
    }
}

/// Which closed-shell gradient formula a call uses — the same three-way
/// dispatch `ferric_scf::ks_gradient` makes on the functional family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ClosedKind {
    /// `xc_gradient_closed_lda_from_density`: χ + ∇χ only, no density floor.
    Lda,
    /// `xc_gradient_closed_gga_from_density` (accepts LDA components too).
    Gga,
    /// `xc_gradient_closed_mgga_from_density`.
    Mgga,
}

/// The density matrices of one call.
#[derive(Clone, Copy)]
pub(super) enum SpinDensity<'a> {
    Closed(&'a Array2<f64>, ClosedKind),
    /// UKS/ROKS `(D_α, D_β)`; `true` = meta-GGA.
    Open(&'a Array2<f64>, &'a Array2<f64>, bool),
}

impl SpinDensity<'_> {
    fn nbf(&self) -> usize {
        match self {
            Self::Closed(d, _) => d.nrows(),
            Self::Open(d, _, _) => d.nrows(),
        }
    }
    fn needs_hess(&self) -> bool {
        !matches!(self, Self::Closed(_, ClosedKind::Lda))
    }
    fn is_uks(&self) -> bool {
        matches!(self, Self::Open(..))
    }
}

/// One batch's compact AO block: the kept functions (global indices, ascending
/// by shell then function) and their values on the batch's points.
struct BatchAo {
    funcs: Vec<u32>,
    chi: Array2<f64>,
    dchi: Array3<f64>,
    /// Present iff the call contracts the AO Hessian.
    ddchi: Option<ndarray::Array4<f64>>,
}

/// Evaluate and screen one batch in a single pass. For every shell that
/// survives the distance prefilter the values are computed (the same per-shell
/// evaluator, the same `p - center` offsets and the same zeroed scratch as the
/// dense `eval_basis_*_on_points_unchecked`, so the kept rows are bit-identical
/// to the dense rows), and the shell is kept iff any of its values exceeds
/// `thresh` (always, for `thresh <= 0`). Serial.
fn eval_batch_ao_screened(
    shells: &[OwnedShell],
    radii2: &[f64],
    pts: &[[f64; 3]],
    thresh: f64,
    hess: bool,
) -> Result<BatchAo, GtoEvalError> {
    let nb = pts.len();
    let nplanes = if hess { 13 } else { 4 };
    let (lo, hi) = points_box(pts);
    // (shell index, values laid out [plane][function][point]).
    let mut kept: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut buf = [0.0f64; 15];
    let mut gbuf: [[f64; 15]; 3] = [[0.0; 15]; 3];
    let mut hbuf: [[f64; 15]; 9] = [[0.0; 15]; 9];
    for (si, sh) in shells.iter().enumerate() {
        if thresh > 0.0 {
            let mut d2 = 0.0_f64;
            for a in 0..3 {
                let e = (lo[a] - sh.center[a]).max(sh.center[a] - hi[a]).max(0.0);
                d2 += e * e;
            }
            if d2 > radii2[si] * (1.0 + SCREEN_DIST2_MARGIN) {
                continue;
            }
        }
        let n = sh.nfunc;
        let loc = sh.located();
        let mut blk = vec![0.0_f64; nplanes * n * nb];
        let mut mx = 0.0_f64;
        for (g, p) in pts.iter().enumerate() {
            buf.fill(0.0);
            for row in gbuf.iter_mut() {
                row.fill(0.0);
            }
            let (dx, dy, dz) = (
                p[0] - sh.center[0],
                p[1] - sh.center[1],
                p[2] - sh.center[2],
            );
            if hess {
                for row in hbuf.iter_mut() {
                    row.fill(0.0);
                }
                eval_shell_grad_hess(&loc, dx, dy, dz, &mut buf[..n], &mut gbuf, &mut hbuf)?;
            } else {
                eval_shell_and_grad(&loc, dx, dy, dz, &mut buf[..n], &mut gbuf)?;
            }
            for i in 0..n {
                blk[i * nb + g] = buf[i];
                mx = mx.max(buf[i].abs());
                for a in 0..3 {
                    let v = gbuf[a][i];
                    blk[((1 + a) * n + i) * nb + g] = v;
                    mx = mx.max(v.abs());
                }
                if hess {
                    for ab in 0..9 {
                        let v = hbuf[ab][i];
                        blk[((4 + ab) * n + i) * nb + g] = v;
                        mx = mx.max(v.abs());
                    }
                }
            }
        }
        if thresh <= 0.0 || mx > thresh {
            kept.push((si, blk));
        }
    }

    let nact: usize = kept.iter().map(|(si, _)| shells[*si].nfunc).sum();
    let mut funcs = Vec::with_capacity(nact);
    let mut chi = Array2::<f64>::zeros((nact, nb));
    let mut dchi = Array3::<f64>::zeros((3, nact, nb));
    let mut ddchi = hess.then(|| ndarray::Array4::<f64>::zeros((3, 3, nact, nb)));
    let mut row0 = 0usize;
    for (si, blk) in &kept {
        let sh = &shells[*si];
        let n = sh.nfunc;
        for i in 0..n {
            funcs.push((sh.offset + i) as u32);
            let r = row0 + i;
            let row = |plane: usize| &blk[(plane * n + i) * nb..(plane * n + i + 1) * nb];
            chi.row_mut(r)
                .as_slice_mut()
                .expect("contiguous")
                .copy_from_slice(row(0));
            for a in 0..3 {
                dchi.index_axis_mut(Axis(0), a)
                    .row_mut(r)
                    .as_slice_mut()
                    .expect("contiguous")
                    .copy_from_slice(row(1 + a));
            }
            if let Some(h) = ddchi.as_mut() {
                for a in 0..3 {
                    for b in 0..3 {
                        let mut ha = h.index_axis_mut(Axis(0), a);
                        let mut hab = ha.index_axis_mut(Axis(0), b);
                        hab.row_mut(r)
                            .as_slice_mut()
                            .expect("contiguous")
                            .copy_from_slice(row(4 + 3 * a + b));
                    }
                }
            }
        }
        row0 += n;
    }
    Ok(BatchAo {
        funcs,
        chi,
        dchi,
        ddchi,
    })
}

/// The per-point grid data one batch needs, gathered from the full grid.
struct BatchPoints<'a> {
    weights: Vec<f64>,
    home: Vec<usize>,
    /// `weight1` row of each point (absolute grid index already resolved).
    w1: Vec<&'a [[f64; 3]]>,
}

/// `M = D·X` for every plane `X` of `x` (`(k, nact, nb)` → `(k, nact, nb)`).
/// Plain `dot`: these run on rayon workers with OpenBLAS pinned to one thread
/// by the caller for the whole parallel region.
fn d_dot_planes(d: &Array2<f64>, x: &Array3<f64>) -> Array3<f64> {
    let (k, n, nb) = x.dim();
    let mut out = Array3::<f64>::zeros((k, n, nb));
    for p in 0..k {
        let prod = d.dot(&x.index_axis(Axis(0), p));
        out.index_axis_mut(Axis(0), p).assign(&prod);
    }
    out
}

/// ρ and 2·Σ_μ M_μ ∂χ_μ (= ∇ρ for a closed-shell or per-spin density) from
/// `m = D·χ`, accumulated μ-outer so every point's sum runs in ascending μ —
/// the per-point order of `eval_density_closed`.
fn rho_grad_from_m(
    chi: &Array2<f64>,
    dchi: &Array3<f64>,
    m: &Array2<f64>,
) -> (Vec<f64>, Array2<f64>) {
    let (nact, nb) = chi.dim();
    let mut rho = vec![0.0_f64; nb];
    let mut grad = Array2::<f64>::zeros((3, nb));
    for mu in 0..nact {
        let c = chi.row(mu);
        let p = m.row(mu);
        for g in 0..nb {
            rho[g] += c[g] * p[g];
        }
        for a in 0..3 {
            let da = dchi.index_axis(Axis(0), a);
            let dr = da.row(mu);
            let mut gr = grad.row_mut(a);
            for g in 0..nb {
                gr[g] += p[g] * dr[g];
            }
        }
    }
    grad.mapv_inplace(|v| 2.0 * v);
    (rho, grad)
}

/// τ = ½ Σ_axis Σ_μ ∂_axis χ_μ · (D ∂_axis χ)_μ, the `eval_tau_closed` sum.
fn tau_from_mdchi(dchi: &Array3<f64>, mdchi: &Array3<f64>) -> Vec<f64> {
    let (_, nact, nb) = dchi.dim();
    let mut per_axis = [vec![0.0_f64; nb], vec![0.0_f64; nb], vec![0.0_f64; nb]];
    for (axis, acc) in per_axis.iter_mut().enumerate() {
        let da = dchi.index_axis(Axis(0), axis);
        let pa = mdchi.index_axis(Axis(0), axis);
        for mu in 0..nact {
            let (dr, pr) = (da.row(mu), pa.row(mu));
            for g in 0..nb {
                acc[g] += dr[g] * pr[g];
            }
        }
    }
    (0..nb)
        .map(|g| 0.5 * (per_axis[0][g] + per_axis[1][g] + per_axis[2][g]))
        .collect()
}

/// Per-point coefficients of one density channel (the closed-shell density,
/// or one UKS spin), already multiplied by the quadrature weight and zeroed
/// below the density floor.
#[derive(Clone, Copy)]
struct Coeffs<'a> {
    /// `w·v_ρ` per point.
    t_rho: &'a [f64],
    /// `(3, nb)` GGA weight columns `c_b` (closed shell `w·2v_σ·∂_bρ`; UKS the
    /// mixed `w·(2v_σσ ∂_bρ_σ + v_σσ' ∂_bρ_σ')`).
    c: &'a Array2<f64>,
    /// `w·v_τ` per point, meta-GGA only.
    t_tau: Option<&'a [f64]>,
}

/// The two reductions of the fused integrand `X_axis[μ,g]` (see
/// [`fused_xc_reductions`]).
struct FusedSums {
    /// `rows[axis][μ] = Σ_g X_axis[μ,g]` — the AO-derivative partials, in the
    /// [`scatter_partials`] convention.
    rows: [ndarray::Array1<f64>; 3],
    /// `cols[axis][g] = Σ_μ X_axis[μ,g]` — half the grid-coordinate response.
    cols: [Vec<f64>; 3],
}

/// The XC-gradient integrand of one channel, built once per (μ, g) and reduced
/// both ways.
///
/// ```text
///   S[μ,g]      = t_ρ m_μ + Σ_b c_b (D∂_bχ)_μ
///   P_b[μ,g]    = c_b m_μ + ½ t_τ (D∂_bχ)_μ            (t_τ = 0 off meta-GGA)
///   X_a[μ,g]    = ∂_aχ_μ · S + Σ_b ∂²_{ab}χ_μ · P_b       (m = Dχ)
/// ```
///
/// **AO-derivative term** (χ_μ moves with atom(μ)): `Σ_g X_a[μ,g]` is exactly
/// `gga_ao_partials` + `mgga_tau_partials` (the latter already carries the ½
/// that the μ↔ν doubling leaves — see its derivation; it is not a second ½ on
/// τ), and [`scatter_partials`] adds `−2·rows` to atom(μ).
///
/// **Grid-coordinate response** (the point moves with its home atom): the old
/// per-point expression `w v_ρ ∂_aρ + w 2v_σ Σ_b ∂_bρ ∂²_{ab}ρ + w v_τ ∂_aτ`,
/// with `∂_aρ = 2Σ_μ m ∂_aχ`, `∂²_{ab}ρ = 2Σ_μ (∂_aχ D∂_bχ + m ∂²_{ab}χ)` and
/// `∂_aτ = Σ_μ Σ_b ∂²_{ab}χ D∂_bχ`, regroups term by term into
/// `2·Σ_μ X_a[μ,g]`, so home(g) gets `+2·cols`. (Closed shell: c_b = t_σ ∂_bρ
/// with t_σ = w 2v_σ. Below the floor every coefficient is zero, as the old
/// loop's `continue` was.) Only summation order differs from the separate
/// `hess_cols` + `gga_ao_partials` passes — the bound is derived in this
/// module's tests.
///
/// One pass, μ-outer with every plane read row-contiguously; scratch is
/// `O(nb)` (four rows), no `(nact, nb)` temporary. `ABS = true` (tests only)
/// evaluates the same expression with every factor replaced by its absolute
/// value — the `Σ|terms|` of the rounding bound.
fn fused_xc_reductions<const ABS: bool>(
    m: &Array2<f64>,
    mdchi: &Array3<f64>,
    dchi: &Array3<f64>,
    ddchi: &ndarray::Array4<f64>,
    co: Coeffs<'_>,
) -> FusedSums {
    let (nact, nb) = m.dim();
    let ab = |v: f64| if ABS { v.abs() } else { v };
    let m = m.as_standard_layout();
    let mdchi = mdchi.as_standard_layout();
    let dchi = dchi.as_standard_layout();
    let ddchi = ddchi.as_standard_layout();
    let c = co.c.as_standard_layout();
    let (m, mdchi, dchi, ddchi, c) = (
        m.as_slice().expect("standard layout"),
        mdchi.as_slice().expect("standard layout"),
        dchi.as_slice().expect("standard layout"),
        ddchi.as_slice().expect("standard layout"),
        c.as_slice().expect("standard layout"),
    );
    let row = |plane: usize, mu: usize| (plane * nact + mu) * nb;
    let t_rho: Vec<f64> = co.t_rho.iter().map(|&v| ab(v)).collect();
    let cb: [Vec<f64>; 3] =
        std::array::from_fn(|b| c[b * nb..(b + 1) * nb].iter().map(|&v| ab(v)).collect());
    let half_tau: Option<Vec<f64>> = co.t_tau.map(|t| t.iter().map(|&v| 0.5 * ab(v)).collect());

    let mut rows: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0_f64; nact]);
    let mut cols: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0_f64; nb]);
    let mut s = vec![0.0_f64; nb];
    let mut p: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0_f64; nb]);
    for mu in 0..nact {
        let mr = &m[mu * nb..(mu + 1) * nb];
        let md: [&[f64]; 3] = std::array::from_fn(|b| &mdchi[row(b, mu)..row(b, mu) + nb]);
        for g in 0..nb {
            let (mm, m0, m1, m2) = (ab(mr[g]), ab(md[0][g]), ab(md[1][g]), ab(md[2][g]));
            s[g] = t_rho[g] * mm + cb[0][g] * m0 + cb[1][g] * m1 + cb[2][g] * m2;
            p[0][g] = cb[0][g] * mm;
            p[1][g] = cb[1][g] * mm;
            p[2][g] = cb[2][g] * mm;
        }
        if let Some(ht) = half_tau.as_ref() {
            for (b, pb) in p.iter_mut().enumerate() {
                for g in 0..nb {
                    pb[g] += ht[g] * ab(md[b][g]);
                }
            }
        }
        for (a, (row_out, col)) in rows.iter_mut().zip(cols.iter_mut()).enumerate() {
            let da = &dchi[row(a, mu)..row(a, mu) + nb];
            let h: [&[f64]; 3] =
                std::array::from_fn(|b| &ddchi[row(3 * a + b, mu)..row(3 * a + b, mu) + nb]);
            let mut acc = 0.0_f64;
            for g in 0..nb {
                let x = ab(da[g]) * s[g]
                    + ab(h[0][g]) * p[0][g]
                    + ab(h[1][g]) * p[1][g]
                    + ab(h[2][g]) * p[2][g];
                acc += x;
                col[g] += x;
            }
            row_out[mu] = acc;
        }
    }
    FusedSums {
        rows: rows.map(ndarray::Array1::from_vec),
        cols,
    }
}

/// One point's coefficients (the scalars a [`Coeffs`] column holds at g).
#[derive(Clone, Copy)]
struct PointCoeffs {
    t_rho: f64,
    c: [f64; 3],
    t_tau: f64,
}

/// `Σ_μ X_a[μ,g]` at the single point `g` with coefficients `pc` — the
/// [`fused_xc_reductions`] column sum, for the rare UKS points whose grid term
/// and AO term disagree on the density floor (see `open_tail`). Column-strided,
/// so only for a handful of points.
fn fused_column<const ABS: bool>(
    g: usize,
    m: &Array2<f64>,
    mdchi: &Array3<f64>,
    dchi: &Array3<f64>,
    ddchi: &ndarray::Array4<f64>,
    pc: PointCoeffs,
) -> [f64; 3] {
    let ab = |v: f64| if ABS { v.abs() } else { v };
    let (t_rho, cb, ht) = (ab(pc.t_rho), pc.c.map(ab), 0.5 * ab(pc.t_tau));
    let mut out = [0.0_f64; 3];
    for mu in 0..m.nrows() {
        let mm = ab(m[(mu, g)]);
        let md: [f64; 3] = std::array::from_fn(|b| ab(mdchi[(b, mu, g)]));
        let s = t_rho * mm + cb[0] * md[0] + cb[1] * md[1] + cb[2] * md[2];
        let p: [f64; 3] = std::array::from_fn(|b| cb[b] * mm + ht * md[b]);
        for (a, o) in out.iter_mut().enumerate() {
            *o += ab(dchi[(a, mu, g)]) * s
                + ab(ddchi[(a, 0, mu, g)]) * p[0]
                + ab(ddchi[(a, 1, mu, g)]) * p[1]
                + ab(ddchi[(a, 2, mu, g)]) * p[2];
        }
    }
    out
}

/// `grad[atom(μ), axis] −= 2·rows[axis][μ]` ([`scatter_partials`]); with
/// `ABS` the bound's `+2·|rows|`.
fn scatter_rows<const ABS: bool>(
    rows: &[ndarray::Array1<f64>; 3],
    map: &[usize],
    grad: &mut Array2<f64>,
) {
    if ABS {
        for (axis, r) in rows.iter().enumerate() {
            for (mu, &v) in r.iter().enumerate() {
                grad[(map[mu], axis)] += 2.0 * v.abs();
            }
        }
    } else {
        scatter_partials(rows, map, grad);
    }
}

/// Weight response `Σ_g weight1[g][B] · f_g` for every atom B.
fn add_weight_response(grad: &mut Array2<f64>, w1: &[&[[f64; 3]]], f: &[f64]) {
    add_weight_response_k::<false>(grad, w1, f);
}

/// [`add_weight_response`]; `ABS` sums `|weight1|·|f|` (the bound's terms).
fn add_weight_response_k<const ABS: bool>(grad: &mut Array2<f64>, w1: &[&[[f64; 3]]], f: &[f64]) {
    let ab = |v: f64| if ABS { v.abs() } else { v };
    let natoms = grad.nrows();
    for (g, &fg) in f.iter().enumerate() {
        let row = w1[g];
        let fg = ab(fg);
        for b in 0..natoms {
            grad[(b, 0)] += ab(row[b][0]) * fg;
            grad[(b, 1)] += ab(row[b][1]) * fg;
            grad[(b, 2)] += ab(row[b][2]) * fg;
        }
    }
}

/// Which per-batch kernel a call runs. Production is always `Fused`; the other
/// two exist for the old-vs-new anchor in this module's tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BatchKernel {
    /// [`fused_xc_reductions`]: one integrand, two reductions.
    Fused,
    /// The separate `gga_ao_partials` / `mgga_tau_partials` + `hess_cols` /
    /// `tau_spatial_grad` passes this kernel replaced.
    #[cfg(test)]
    Legacy,
    /// `Σ|terms|` per gradient component (the fused expression in absolute
    /// values), for the rounding bound.
    #[cfg(test)]
    Budget,
}

/// The closed-shell GGA / meta-GGA tail after libxc: AO-derivative term,
/// weight response, grid-coordinate response — in the old pass order.
#[allow(clippy::too_many_arguments)]
fn closed_tail<const ABS: bool>(
    grad: &mut Array2<f64>,
    map: &[usize],
    ao: &BatchAo,
    bp: &BatchPoints<'_>,
    m: &Array2<f64>,
    mdchi: &Array3<f64>,
    co: Coeffs<'_>,
    f: &[f64],
) {
    let ddchi = ao.ddchi.as_ref().expect("GGA/mGGA evaluate the AO Hessian");
    let sums = fused_xc_reductions::<ABS>(m, mdchi, &ao.dchi, ddchi, co);
    scatter_rows::<ABS>(&sums.rows, map, grad);
    add_weight_response_k::<ABS>(grad, &bp.w1, f);
    for (g, &a) in bp.home.iter().enumerate() {
        for axis in 0..3 {
            grad[(a, axis)] += 2.0 * sums.cols[axis][g];
        }
    }
}

/// One UKS spin channel's batch quantities.
struct SpinChannel<'a> {
    m: &'a Array2<f64>,
    mdchi: &'a Array3<f64>,
    rho: &'a [f64],
    vrho: &'a [f64],
    vsig_same: &'a [f64],
    vsig_cross: &'a [f64],
    vtau: &'a [f64],
    grad_same: &'a Array2<f64>,
    grad_cross: &'a Array2<f64>,
}

impl SpinChannel<'_> {
    /// The unfloored coefficients at point `g` (`add_spin`'s formulas).
    fn point(&self, g: usize, w: f64) -> PointCoeffs {
        PointCoeffs {
            t_rho: w * self.vrho[g],
            c: std::array::from_fn(|b| {
                w * (2.0 * self.vsig_same[g] * self.grad_same[(b, g)]
                    + self.vsig_cross[g] * self.grad_cross[(b, g)])
            }),
            t_tau: w * self.vtau[g],
        }
    }
}

/// The UKS tail after libxc. The AO-derivative term floors each spin on its
/// OWN density; the grid term floors both spins on the TOTAL density (as the
/// old `uks_*_batch` code did). Where the two disagree for a spin — its own
/// density below the floor while the total is above, or the reverse — its
/// fused column is replaced by the column with the grid term's coefficients
/// ([`fused_column`]), so the result is the old expression exactly, not a
/// floor-shifted one.
#[allow(clippy::too_many_arguments)]
fn open_tail<const ABS: bool>(
    grad: &mut Array2<f64>,
    map: &[usize],
    ao: &BatchAo,
    bp: &BatchPoints<'_>,
    floor: crate::vxc::DensityFloor,
    mgga: bool,
    chans: [&SpinChannel<'_>; 2],
    f: &[f64],
) {
    let nb = bp.weights.len();
    let ddchi = ao.ddchi.as_ref().expect("UKS evaluates the AO Hessian");
    let above = |r: f64| crate::vxc::above_density_floor(r, floor);
    let sums: Vec<FusedSums> = chans
        .iter()
        .map(|ch| {
            let mut t_rho = vec![0.0_f64; nb];
            let mut t_tau = vec![0.0_f64; nb];
            let mut c = Array2::<f64>::zeros((3, nb));
            for g in 0..nb {
                if above(ch.rho[g]) {
                    let pc = ch.point(g, bp.weights[g]);
                    t_rho[g] = pc.t_rho;
                    t_tau[g] = pc.t_tau;
                    for b in 0..3 {
                        c[(b, g)] = pc.c[b];
                    }
                }
            }
            let co = Coeffs {
                t_rho: &t_rho,
                c: &c,
                t_tau: mgga.then_some(&t_tau[..]),
            };
            let s = fused_xc_reductions::<ABS>(ch.m, ch.mdchi, &ao.dchi, ddchi, co);
            scatter_rows::<ABS>(&s.rows, map, grad);
            s
        })
        .collect();
    add_weight_response_k::<ABS>(grad, &bp.w1, f);
    for g in 0..nb {
        let grid_on = above(chans[0].rho[g] + chans[1].rho[g]);
        let mut add = [0.0_f64; 3];
        for (ch, s) in chans.iter().zip(&sums) {
            match (grid_on, above(ch.rho[g])) {
                (true, true) => (0..3).for_each(|k| add[k] += s.cols[k][g]),
                (true, false) => {
                    let mut pc = ch.point(g, bp.weights[g]);
                    if !mgga {
                        pc.t_tau = 0.0;
                    }
                    let col = fused_column::<ABS>(g, ch.m, ch.mdchi, &ao.dchi, ddchi, pc);
                    (0..3).for_each(|k| add[k] += col[k]);
                }
                // Grid term off at this point: drop the AO-floored column.
                (false, _) => {}
            }
        }
        let a = bp.home[g];
        for (axis, v) in add.iter().enumerate() {
            grad[(a, axis)] += 2.0 * v;
        }
    }
}

/// One batch of the closed-shell gradient (LDA / GGA / meta-GGA): the terms of
/// `xc_gradient_closed_lda_from_density`, `gga_closed_batch` and
/// `mgga_closed_batch`, over this batch's points and kept functions.
#[allow(clippy::too_many_arguments)]
fn closed_batch(
    kernel: BatchKernel,
    xc: &XcDef,
    kind: ClosedKind,
    d: &Array2<f64>,
    map: &[usize],
    natoms: usize,
    ao: &BatchAo,
    bp: &BatchPoints<'_>,
) -> Array2<f64> {
    let nb = bp.weights.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));
    if ao.funcs.is_empty() {
        // No significant function: ρ, ∇ρ, τ and every AO term vanish here.
        // libxc still sees ρ = 0 in the dense path; its ε·ρ is then 0 as well.
        return grad;
    }
    let chi = &ao.chi;
    let dchi = &ao.dchi;
    let m = d.dot(chi);
    let mdchi = (kind != ClosedKind::Lda).then(|| d_dot_planes(d, dchi));
    let (rho, grad_rho) = rho_grad_from_m(chi, dchi, &m);
    let mut sigma = vec![0.0_f64; nb];
    for (g, s) in sigma.iter_mut().enumerate() {
        let (x, y, z) = (grad_rho[(0, g)], grad_rho[(1, g)], grad_rho[(2, g)]);
        *s = x * x + y * y + z * z;
    }
    let tau = (kind == ClosedKind::Mgga)
        .then(|| tau_from_mdchi(dchi, mdchi.as_ref().expect("mgga has mdchi")));

    let mut eps_total = vec![0.0_f64; nb];
    let mut vrho_total = vec![0.0_f64; nb];
    let mut vsigma_total = vec![0.0_f64; nb];
    let mut vtau_total = vec![0.0_f64; nb];
    for (i, func) in xc.funcs.iter().enumerate() {
        let w_i = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
        let mut exc = vec![0.0_f64; nb];
        let mut vrho = vec![0.0_f64; nb];
        match (kind, func.family()) {
            (_, FunctionalFamily::Lda) | (ClosedKind::Lda, _) => {
                func.eval_lda_unpolarized(&rho, &mut exc, &mut vrho);
            }
            (ClosedKind::Mgga, FunctionalFamily::MetaGga) => {
                let mut vsigma = vec![0.0_f64; nb];
                let mut vtau = vec![0.0_f64; nb];
                func.eval_mgga_unpolarized(
                    &rho,
                    &sigma,
                    tau.as_ref().expect("mgga has tau"),
                    &mut exc,
                    &mut vrho,
                    &mut vsigma,
                    &mut vtau,
                );
                for g in 0..nb {
                    vsigma_total[g] += w_i * vsigma[g];
                    vtau_total[g] += w_i * vtau[g];
                }
            }
            _ => {
                let mut vsigma = vec![0.0_f64; nb];
                func.eval_gga_unpolarized(&rho, &sigma, &mut exc, &mut vrho, &mut vsigma);
                for g in 0..nb {
                    vsigma_total[g] += w_i * vsigma[g];
                }
            }
        }
        for g in 0..nb {
            eps_total[g] += w_i * exc[g];
            vrho_total[g] += w_i * vrho[g];
        }
    }

    if kind == ClosedKind::Lda {
        // `xc_gradient_closed_lda`: t = w·v_ρ at every point (no floor).
        let t: Vec<f64> = (0..nb).map(|g| bp.weights[g] * vrho_total[g]).collect();
        let mt = scale_cols(&m.view(), &ArrayView1::from(&t[..]));
        let partials: [ndarray::Array1<f64>; 3] = std::array::from_fn(|axis| {
            row_dot_p(&mt.view(), &dchi.index_axis(Axis(0), axis), false)
        });
        scatter_partials(&partials, map, &mut grad);
        let f: Vec<f64> = (0..nb).map(|g| eps_total[g] * rho[g]).collect();
        add_weight_response(&mut grad, &bp.w1, &f);
        for g in 0..nb {
            let a = bp.home[g];
            let wv = bp.weights[g] * vrho_total[g];
            for k in 0..3 {
                grad[(a, k)] += wv * grad_rho[(k, g)];
            }
        }
        return grad;
    }

    let mdchi = mdchi.expect("GGA/mGGA have mdchi");
    let floor = crate::vxc::DensityFloor::for_xc(xc);
    let mut t_rho = vec![0.0_f64; nb];
    let mut t_sig = vec![0.0_f64; nb];
    let mut t_tau = vec![0.0_f64; nb];
    for g in 0..nb {
        if crate::vxc::above_density_floor(rho[g], floor) {
            t_rho[g] = bp.weights[g] * vrho_total[g];
            t_sig[g] = bp.weights[g] * 2.0 * vsigma_total[g];
            t_tau[g] = bp.weights[g] * vtau_total[g];
        }
    }
    let c = gga_weight_columns(&t_sig, &grad_rho);
    let f: Vec<f64> = (0..nb).map(|g| eps_total[g] * rho[g]).collect();
    let co = Coeffs {
        t_rho: &t_rho,
        c: &c,
        t_tau: (kind == ClosedKind::Mgga).then_some(&t_tau[..]),
    };
    match kernel {
        BatchKernel::Fused => closed_tail::<false>(&mut grad, map, ao, bp, &m, &mdchi, co, &f),
        #[cfg(test)]
        BatchKernel::Budget => closed_tail::<true>(&mut grad, map, ao, bp, &m, &mdchi, co, &f),
        #[cfg(test)]
        BatchKernel::Legacy => legacy::closed_tail(
            &mut grad,
            map,
            ao,
            bp,
            &m,
            &mdchi,
            co,
            &f,
            &grad_rho,
            &rho,
            floor,
            (&vrho_total, &vsigma_total, &vtau_total),
        ),
    }
    grad
}

/// One batch of the spin-polarized gradient (GGA or meta-GGA; LDA components
/// enter with `v_σ = 0`): the terms of `uks_gga_batch` / `uks_mgga_batch`.
#[allow(clippy::too_many_arguments)]
fn open_batch(
    kernel: BatchKernel,
    xc: &XcDef,
    mgga: bool,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    map: &[usize],
    natoms: usize,
    ao: &BatchAo,
    bp: &BatchPoints<'_>,
) -> Array2<f64> {
    let nb = bp.weights.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));
    if ao.funcs.is_empty() {
        return grad;
    }
    let chi = &ao.chi;
    let dchi = &ao.dchi;
    let m_a = d_a.dot(chi);
    let m_b = d_b.dot(chi);
    let mdchi_a = d_dot_planes(d_a, dchi);
    let mdchi_b = d_dot_planes(d_b, dchi);
    let (rho_a, grad_a) = rho_grad_from_m(chi, dchi, &m_a);
    let (rho_b, grad_b) = rho_grad_from_m(chi, dchi, &m_b);

    let mut rho_in = vec![0.0_f64; 2 * nb];
    let mut sigma_in = vec![0.0_f64; 3 * nb];
    for g in 0..nb {
        rho_in[2 * g] = rho_a[g];
        rho_in[2 * g + 1] = rho_b[g];
        let (ax, ay, az) = (grad_a[(0, g)], grad_a[(1, g)], grad_a[(2, g)]);
        let (bx, by, bz) = (grad_b[(0, g)], grad_b[(1, g)], grad_b[(2, g)]);
        sigma_in[3 * g] = ax * ax + ay * ay + az * az;
        sigma_in[3 * g + 1] = ax * bx + ay * by + az * bz;
        sigma_in[3 * g + 2] = bx * bx + by * by + bz * bz;
    }
    let tau_in = mgga.then(|| {
        let ta = tau_from_mdchi(dchi, &mdchi_a);
        let tb = tau_from_mdchi(dchi, &mdchi_b);
        let mut t = vec![0.0_f64; 2 * nb];
        for g in 0..nb {
            t[2 * g] = ta[g];
            t[2 * g + 1] = tb[g];
        }
        t
    });

    let mut eps_total = vec![0.0_f64; nb];
    let mut vrho_a = vec![0.0_f64; nb];
    let mut vrho_b = vec![0.0_f64; nb];
    let mut vsig_aa = vec![0.0_f64; nb];
    let mut vsig_ab = vec![0.0_f64; nb];
    let mut vsig_bb = vec![0.0_f64; nb];
    let mut vtau_a = vec![0.0_f64; nb];
    let mut vtau_b = vec![0.0_f64; nb];
    for (i, func) in xc.funcs.iter().enumerate() {
        let w_i = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
        let mut exc = vec![0.0_f64; nb];
        let mut vrho = vec![0.0_f64; 2 * nb];
        match func.family() {
            FunctionalFamily::Lda => func.eval_lda_polarized(&rho_in, &mut exc, &mut vrho),
            FunctionalFamily::MetaGga if mgga => {
                let mut vsig = vec![0.0_f64; 3 * nb];
                let mut vtau = vec![0.0_f64; 2 * nb];
                func.eval_mgga_polarized(
                    &rho_in,
                    &sigma_in,
                    tau_in.as_ref().expect("mgga has tau"),
                    &mut exc,
                    &mut vrho,
                    &mut vsig,
                    &mut vtau,
                );
                for g in 0..nb {
                    vsig_aa[g] += w_i * vsig[3 * g];
                    vsig_ab[g] += w_i * vsig[3 * g + 1];
                    vsig_bb[g] += w_i * vsig[3 * g + 2];
                    vtau_a[g] += w_i * vtau[2 * g];
                    vtau_b[g] += w_i * vtau[2 * g + 1];
                }
            }
            _ => {
                let mut vsig = vec![0.0_f64; 3 * nb];
                func.eval_gga_polarized(&rho_in, &sigma_in, &mut exc, &mut vrho, &mut vsig);
                for g in 0..nb {
                    vsig_aa[g] += w_i * vsig[3 * g];
                    vsig_ab[g] += w_i * vsig[3 * g + 1];
                    vsig_bb[g] += w_i * vsig[3 * g + 2];
                }
            }
        }
        for g in 0..nb {
            eps_total[g] += w_i * exc[g];
            vrho_a[g] += w_i * vrho[2 * g];
            vrho_b[g] += w_i * vrho[2 * g + 1];
        }
    }

    let floor = crate::vxc::DensityFloor::for_xc(xc);
    let f: Vec<f64> = (0..nb)
        .map(|g| eps_total[g] * (rho_a[g] + rho_b[g]))
        .collect();
    let ch_a = SpinChannel {
        m: &m_a,
        mdchi: &mdchi_a,
        rho: &rho_a,
        vrho: &vrho_a,
        vsig_same: &vsig_aa,
        vsig_cross: &vsig_ab,
        vtau: &vtau_a,
        grad_same: &grad_a,
        grad_cross: &grad_b,
    };
    let ch_b = SpinChannel {
        m: &m_b,
        mdchi: &mdchi_b,
        rho: &rho_b,
        vrho: &vrho_b,
        vsig_same: &vsig_bb,
        vsig_cross: &vsig_ab,
        vtau: &vtau_b,
        grad_same: &grad_b,
        grad_cross: &grad_a,
    };
    let chans = [&ch_a, &ch_b];
    match kernel {
        BatchKernel::Fused => open_tail::<false>(&mut grad, map, ao, bp, floor, mgga, chans, &f),
        #[cfg(test)]
        BatchKernel::Budget => open_tail::<true>(&mut grad, map, ao, bp, floor, mgga, chans, &f),
        #[cfg(test)]
        BatchKernel::Legacy => legacy::open_tail(&mut grad, map, ao, bp, floor, mgga, chans, &f),
    }
    grad
}

/// The screened XC nuclear gradient (see the module doc). `label` names the
/// memory-plan charge, as each whole-grid entry point did.
#[allow(clippy::too_many_arguments)]
pub(super) fn screened_xc_gradient(
    label: &'static str,
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    dens: SpinDensity<'_>,
    xc: &XcDef,
    grid_cfg: &AtomicGridConfig,
    shell_to_atom: &[usize],
    shell_offsets: &[usize],
    shell_dims: &[usize],
    cfg: GradScreenConfig,
) -> Result<Array2<f64>, KsGradError> {
    screened_xc_gradient_k(
        BatchKernel::Fused,
        label,
        mol,
        bs,
        dens,
        xc,
        grid_cfg,
        shell_to_atom,
        shell_offsets,
        shell_dims,
        cfg,
    )
}

/// [`screened_xc_gradient`] with the per-batch kernel chosen (tests compare
/// [`BatchKernel::Fused`] against the old passes through this).
#[allow(clippy::too_many_arguments)]
fn screened_xc_gradient_k(
    kernel: BatchKernel,
    label: &'static str,
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    dens: SpinDensity<'_>,
    xc: &XcDef,
    grid_cfg: &AtomicGridConfig,
    shell_to_atom: &[usize],
    shell_offsets: &[usize],
    shell_dims: &[usize],
    cfg: GradScreenConfig,
) -> Result<Array2<f64>, KsGradError> {
    let nbf = dens.nbf();
    let natoms = mol.atoms.len();
    let t = ferric_core::timing::Stage::start("xcg:grid+weight1");
    let (grid, weight1) = build_atomic_grid_with_response(mol, grid_cfg)?;
    t.end();
    let map = bf_to_atom(shell_to_atom, shell_offsets, shell_dims, nbf);
    let located = crate::ao_grid::collect_shells(mol, bs)?;
    let shells = owned_shells(&located);
    let hess = dens.needs_hess();
    let thresh = cfg.screen_thresh;
    let radii2 = if hess {
        shell_screen_radii2_hess(&shells, thresh)
    } else {
        shell_screen_radii2(&shells, thresh)
    };
    let xyz: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let batches = partition_points(&xyz, cfg.max_batch_pts.max(1));
    let ngroups = GRAD_TARGET_GROUPS.min(batches.len()).max(1);
    let ranges = group_ranges(batches.len(), ngroups);

    // Memory: the non-batchable grid + weight1, plus one batch's planes per
    // CONCURRENTLY LIVE group. Concurrency is a memory knob only — the group
    // partition (and therefore the result) never depends on it — so under a
    // crowded pool the gradient runs fewer groups at once rather than refusing,
    // and refuses only when one batch does not fit.
    let kind = if hess {
        crate::ao_grid::AoGridKind::ValueGradHess
    } else {
        crate::ao_grid::AoGridKind::ValueAndGrad
    };
    let mut live = rayon::current_num_threads().min(ranges.len()).max(1);
    let _charge = loop {
        let plan = xc_gradient_plan(
            label,
            nbf,
            live.saturating_mul(cfg.max_batch_pts.clamp(1, grid.len().max(1))),
            grid.len(),
            natoms,
            kind,
            dens.is_uks(),
            true,
        );
        match plan.commit() {
            Ok(c) => break c,
            Err(_) if live > 1 => live /= 2,
            Err(e) => return Err(e.into()),
        }
    };

    let group = |&(b0, b1): &(usize, usize)| -> Result<Array2<f64>, KsGradError> {
        let xc_w = xc.worker_copy();
        let mut acc = Array2::<f64>::zeros((natoms, 3));
        for idx in &batches[b0..b1] {
            let pts: Vec<[f64; 3]> = idx.iter().map(|&i| xyz[i as usize]).collect();
            let ao = eval_batch_ao_screened(&shells, &radii2, &pts, thresh, hess)?;
            let bp = BatchPoints {
                weights: idx.iter().map(|&i| grid[i as usize].weight).collect(),
                home: idx.iter().map(|&i| grid[i as usize].home_atom).collect(),
                w1: idx.iter().map(|&i| &weight1[i as usize][..]).collect(),
            };
            let sub_map: Vec<usize> = ao.funcs.iter().map(|&f| map[f as usize]).collect();
            let part = match dens {
                SpinDensity::Closed(d, ck) => {
                    let ds = gather_sub(d, &ao.funcs);
                    closed_batch(kernel, &xc_w, ck, &ds, &sub_map, natoms, &ao, &bp)
                }
                SpinDensity::Open(da, db, mgga) => {
                    let (dsa, dsb) = (gather_sub(da, &ao.funcs), gather_sub(db, &ao.funcs));
                    open_batch(kernel, &xc_w, mgga, &dsa, &dsb, &sub_map, natoms, &ao, &bp)
                }
            };
            acc += &part;
        }
        Ok(acc)
    };

    let t = ferric_core::timing::Stage::start("xcg:batches");
    // OpenBLAS pinned to one thread for the whole parallel region (each worker
    // runs its own small GEMMs), as `xc_batch` does for the SCF V_xc.
    let parts: Vec<Array2<f64>> = with_blas_threads(1, || -> Result<_, KsGradError> {
        if live >= rayon::current_num_threads() {
            ranges.par_iter().map(group).collect()
        } else {
            let mut out = Vec::with_capacity(ranges.len());
            for wave in ranges.chunks(live) {
                let r: Vec<Array2<f64>> = wave.par_iter().map(group).collect::<Result<_, _>>()?;
                out.extend(r);
            }
            Ok(out)
        }
    })?;
    t.end();

    let mut grad = Array2::<f64>::zeros((natoms, 3));
    for p in &parts {
        grad += p;
    }
    Ok(grad)
}

/// The per-batch passes [`fused_xc_reductions`] replaced, verbatim, kept as the
/// anchor of `fused_vs_legacy_tests` only.
#[cfg(test)]
mod legacy {
    use super::*;

    /// `hess_col[axis][b][g] = 2 Σ_μ [∂_axis χ_μ (D ∂_b χ)_μ + (Dχ)_μ ∂²_{axis,b} χ_μ]`
    /// = ∂²_{axis,b} ρ — the grid-coordinate response's second-derivative column.
    fn hess_cols(
        dchi: &Array3<f64>,
        ddchi: &ndarray::Array4<f64>,
        m: &Array2<f64>,
        mdchi: &Array3<f64>,
    ) -> [[ndarray::Array1<f64>; 3]; 3] {
        std::array::from_fn(|axis| {
            let dchi_axis = dchi.index_axis(Axis(0), axis);
            let ddchi_a = ddchi.index_axis(Axis(0), axis);
            std::array::from_fn(|b| {
                let mdchi_b = mdchi.index_axis(Axis(0), b);
                let ddchi_ab = ddchi_a.index_axis(Axis(0), b);
                let mut hb = col_dot_p(&dchi_axis, &mdchi_b, false);
                hb += &col_dot_p(&m.view(), &ddchi_ab, false);
                hb.mapv_inplace(|v| 2.0 * v);
                hb
            })
        })
    }

    /// The old closed-shell GGA / meta-GGA tail.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn closed_tail(
        grad: &mut Array2<f64>,
        map: &[usize],
        ao: &BatchAo,
        bp: &BatchPoints<'_>,
        m: &Array2<f64>,
        mdchi: &Array3<f64>,
        co: Coeffs<'_>,
        f: &[f64],
        grad_rho: &Array2<f64>,
        rho: &[f64],
        floor: crate::vxc::DensityFloor,
        (vrho_total, vsigma_total, vtau_total): (&[f64], &[f64], &[f64]),
    ) {
        let nb = bp.weights.len();
        let dchi = &ao.dchi;
        let ddchi = ao.ddchi.as_ref().expect("GGA/mGGA evaluate the AO Hessian");
        let partials = gga_ao_partials(m, mdchi, dchi, ddchi, co.c, co.t_rho, false);
        scatter_partials(&partials, map, grad);
        if let Some(t_tau) = co.t_tau {
            let tau_partials = mgga_tau_partials(mdchi, ddchi, t_tau, false);
            scatter_partials(&tau_partials, map, grad);
        }
        add_weight_response(grad, &bp.w1, f);
        let hcol = hess_cols(dchi, ddchi, m, mdchi);
        let dtau = co
            .t_tau
            .is_some()
            .then(|| tau_spatial_grad(mdchi, ddchi, false));
        for g in 0..nb {
            if !crate::vxc::above_density_floor(rho[g], floor) {
                continue;
            }
            let a = bp.home[g];
            let w = bp.weights[g];
            let (vr, vs, vt) = (vrho_total[g], vsigma_total[g], vtau_total[g]);
            for k in 0..3 {
                grad[(a, k)] += w * vr * grad_rho[(k, g)];
            }
            for axis in 0..3 {
                let mut sum_b = 0.0_f64;
                for b in 0..3 {
                    sum_b += grad_rho[(b, g)] * hcol[axis][b][g];
                }
                grad[(a, axis)] += w * 2.0 * vs * sum_b;
                if let Some(dt) = dtau.as_ref() {
                    grad[(a, axis)] += w * vt * dt[axis][g];
                }
            }
        }
    }

    /// The old UKS tail (`add_spin` per spin, then the total-floored grid loop).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn open_tail(
        grad: &mut Array2<f64>,
        map: &[usize],
        ao: &BatchAo,
        bp: &BatchPoints<'_>,
        floor: crate::vxc::DensityFloor,
        mgga: bool,
        chans: [&SpinChannel<'_>; 2],
        f: &[f64],
    ) {
        let nb = bp.weights.len();
        let dchi = &ao.dchi;
        let ddchi = ao.ddchi.as_ref().expect("UKS evaluates the AO Hessian");
        for ch in chans {
            let mut t_rho = vec![0.0_f64; nb];
            let mut t_tau = vec![0.0_f64; nb];
            let mut c = Array2::<f64>::zeros((3, nb));
            for g in 0..nb {
                if crate::vxc::above_density_floor(ch.rho[g], floor) {
                    let w = bp.weights[g];
                    t_rho[g] = w * ch.vrho[g];
                    t_tau[g] = w * ch.vtau[g];
                    for b in 0..3 {
                        c[(b, g)] = w
                            * (2.0 * ch.vsig_same[g] * ch.grad_same[(b, g)]
                                + ch.vsig_cross[g] * ch.grad_cross[(b, g)]);
                    }
                }
            }
            let partials = gga_ao_partials(ch.m, ch.mdchi, dchi, ddchi, &c, &t_rho, false);
            scatter_partials(&partials, map, grad);
            if mgga {
                let tau_partials = mgga_tau_partials(ch.mdchi, ddchi, &t_tau, false);
                scatter_partials(&tau_partials, map, grad);
            }
        }
        add_weight_response(grad, &bp.w1, f);

        let [a_ch, b_ch] = chans;
        let hess_a = hess_cols(dchi, ddchi, a_ch.m, a_ch.mdchi);
        let hess_b = hess_cols(dchi, ddchi, b_ch.m, b_ch.mdchi);
        let dtau = mgga.then(|| {
            (
                tau_spatial_grad(a_ch.mdchi, ddchi, false),
                tau_spatial_grad(b_ch.mdchi, ddchi, false),
            )
        });
        let (grad_a, grad_b) = (a_ch.grad_same, b_ch.grad_same);
        for g in 0..nb {
            if !crate::vxc::above_density_floor(a_ch.rho[g] + b_ch.rho[g], floor) {
                continue;
            }
            let a = bp.home[g];
            let w = bp.weights[g];
            let (vra, vrb) = (a_ch.vrho[g], b_ch.vrho[g]);
            let (vsaa, vsbb, vsab) = (a_ch.vsig_same[g], b_ch.vsig_same[g], a_ch.vsig_cross[g]);
            for axis in 0..3 {
                let rho_piece = vra * grad_a[(axis, g)] + vrb * grad_b[(axis, g)];
                let mut sig_piece = 0.0_f64;
                for b in 0..3 {
                    let gba = grad_a[(b, g)];
                    let gbb = grad_b[(b, g)];
                    let h_aa = hess_a[axis][b][g];
                    let h_bb = hess_b[axis][b][g];
                    sig_piece += 2.0 * vsaa * gba * h_aa
                        + 2.0 * vsbb * gbb * h_bb
                        + vsab * (gba * h_bb + gbb * h_aa);
                }
                let tau_piece = match dtau.as_ref() {
                    Some((da, db)) => a_ch.vtau[g] * da[axis][g] + b_ch.vtau[g] * db[axis][g],
                    None => 0.0,
                };
                grad[(a, axis)] += w * (rho_piece + sig_piece + tau_piece);
            }
        }
    }
}

/// The fused kernel against the separate passes it replaced, at a DERIVED
/// rounding bound.
///
/// # Hypotheses, stated before measuring
///
/// * Right: both kernels evaluate the same sum of the same products (the
///   identity is in [`fused_xc_reductions`]'s doc); only the association
///   differs. Each component is then within `2·γ_d·T` of the other, where `T`
///   is the component's `Σ|terms|` ([`BatchKernel::Budget`]: the fused
///   expression in absolute values, both reductions, plus the weight response)
///   and `d` bounds the depth of either evaluation tree: `P = 8` rounded
///   products per term (`w·2·v_σ·∂ρ·2·∂χ·Dχ` is the longest), the per-row
///   `nb ≤ DEFAULT_MAX_BATCH_PTS` fold plus its 7 (+3 τ) row-dot partials,
///   the per-column μ fold (`≤ nbf`) plus 3 `Σ_b` terms, the scatter into an
///   atom over μ (`≤ nbf`), the batch's grid loop (`≤ nb`), the batches of a
///   group (`≤ npts`, a loose cap) and the `≤ GRAD_TARGET_GROUPS` groups.
///   Expected observed: ~1e-15..1e-14 Ha/Bohr; the bound is ~1e-11 relative to
///   T.
/// * Broken (a reduction that drops the weight, a lost factor 2 or ½, a τ
///   term on the wrong axis, a floor applied to the wrong density): a miss of
///   a whole gradient term, 1e-6..1e-1 — orders above the bound.
///
/// Mutations seen to fail (applied, run, reverted; see the PR report):
/// grid scatter `2.0 *` → `1.0 *` in `closed_tail` fails the three water
/// cases (|Δ| ~0.3 vs bound ~5e-10), in `open_tail` both NH2 cases; the τ
/// `0.5 *` in `fused_xc_reductions` → `1.0 *` fails the two SCAN cases only;
/// the UKS `(true, false)` floor-mismatch arm emptied PASSES the NH2 cases
/// (sub-bound tail points) and fails `uks_floor_mismatch_points_match_legacy`.
#[cfg(test)]
mod fused_vs_legacy_tests {
    use super::*;
    use crate::grid::AtomicGridConfig;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    const WATER: &str = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
    const NH2: &str =
        "3\nNH2\nN 0.000000 0.000000 0.143500\nH 0.000000 0.801500 -0.502200\nH 0.000000 -0.801500 -0.502200\n";

    /// Product roundings per term (see the module hypotheses).
    const PRODUCT_ROUNDINGS: usize = 8;

    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * f64::EPSILON / 2.0;
        nu / (1.0 - nu)
    }

    /// Converge ferric's OWN SCF (never a foreign guess) and run all three
    /// kernels on its density; assert fused vs legacy at the derived bound.
    fn check(xyz: &str, mult: usize, xc_name: &str, kind: ClosedKind) {
        let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = ferric_scf::rhf::RhfConfig {
            xc: Some(xc_name.into()),
            density_conv: 1e-6,
            max_iter: 300,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let (d_a, d_b) = if mult == 1 {
            let r = ferric_scf::rhf::solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
            (r.density_total.clone(), None)
        } else {
            let r = ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
            (r.density_alpha.clone(), r.density_beta.clone())
        };
        let (xc, dens) = match d_b.as_ref() {
            None => (
                xc_def_from_name(xc_name).unwrap(),
                SpinDensity::Closed(&d_a, kind),
            ),
            Some(db) => (
                xc_def_from_name_nspin(xc_name, 2).unwrap(),
                SpinDensity::Open(&d_a, db, kind == ClosedKind::Mgga),
            ),
        };
        let grid = AtomicGridConfig::production();
        let run = |k: BatchKernel| {
            screened_xc_gradient_k(
                k,
                "fused-vs-legacy test",
                &mol,
                &bs,
                dens,
                &xc,
                &grid,
                prep.shell_to_atom(),
                prep.shell_offsets(),
                prep.shell_dims(),
                GradScreenConfig::default(),
            )
            .unwrap()
        };
        let (new, old, budget) = (
            run(BatchKernel::Fused),
            run(BatchKernel::Legacy),
            run(BatchKernel::Budget),
        );
        let npts = crate::grid::build_atomic_grid(&mol, &grid).len();
        let nbf = d_a.nrows();
        let depth = PRODUCT_ROUNDINGS
            + (DEFAULT_MAX_BATCH_PTS + 10)
            + (nbf + 3)
            + nbf
            + DEFAULT_MAX_BATCH_PTS
            + npts
            + GRAD_TARGET_GROUPS;
        let g = gamma(depth);
        let mut worst = 0.0_f64;
        let mut max_diff = 0.0_f64;
        for ((&n, &o), &t) in new.iter().zip(old.iter()).zip(budget.iter()) {
            assert!(t.is_finite() && t > 0.0, "Σ|terms| must be positive: {t}");
            let bound = 2.0 * g * t;
            let diff = (n - o).abs();
            max_diff = max_diff.max(diff);
            worst = worst.max(diff / bound);
            assert!(
                diff <= bound,
                "{xc_name} mult {mult}: fused {n:e} vs legacy {o:e}: |Δ| {diff:e} > bound {bound:e} \
                 (γ_{depth} = {g:e}, T = {t:e})"
            );
        }
        let max_bound = budget.iter().fold(0.0_f64, |a, &t| a.max(2.0 * g * t));
        eprintln!(
            "{xc_name} mult {mult}: max |Δ| {max_diff:.3e}, max bound {max_bound:.3e}, \
             worst |Δ|/bound {worst:.3e} (npts {npts}, depth {depth})"
        );
    }

    /// Deterministic values in [-1, 1).
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// The UKS floor-mismatch branches of `open_tail`, reached on purpose:
    /// point 0 has spin β below the floor while the total is above (the grid
    /// term must still carry β with its UNFLOORED coefficients); point 1 has
    /// spin α above the floor while the total is below (a negative β, so the
    /// grid term must DROP α's AO-floored column). A real SCF density reaches
    /// these points only in far tails where the terms are below the rounding
    /// bound (NH2/SCAN: emptying the branch passes `nh2_uks_scan`), so the
    /// branch is pinned here on synthetic O(1) data instead. Mutation: the
    /// `(true, false)` arm emptied fails this test.
    #[test]
    fn uks_floor_mismatch_points_match_legacy() {
        let (nact, nb, natoms) = (4usize, 6usize, 2usize);
        let mut seed = 7u64;
        let mut r = || lcg(&mut seed);
        let ao = BatchAo {
            funcs: (0..nact as u32).collect(),
            chi: Array2::from_shape_fn((nact, nb), |_| r()),
            dchi: Array3::from_shape_fn((3, nact, nb), |_| r()),
            ddchi: Some(ndarray::Array4::from_shape_fn((3, 3, nact, nb), |_| r())),
        };
        let w1_rows: Vec<Vec<[f64; 3]>> = (0..nb)
            .map(|_| (0..natoms).map(|_| [r(), r(), r()]).collect())
            .collect();
        let bp = BatchPoints {
            weights: (0..nb).map(|_| 0.5 + 0.5 * r().abs()).collect(),
            home: (0..nb).map(|g| g % natoms).collect(),
            w1: w1_rows.iter().map(|v| &v[..]).collect(),
        };
        let map: Vec<usize> = (0..nact).map(|mu| mu % natoms).collect();
        let floor = crate::vxc::DensityFloor::META_GGA;
        let mut rho_a: Vec<f64> = (0..nb).map(|_| 0.5).collect();
        let mut rho_b: Vec<f64> = (0..nb).map(|_| 0.3).collect();
        rho_b[0] = 1e-12; // β off, total on
        rho_a[1] = 2e-10; // α on ...
        rho_b[1] = -1.5e-10; // ... total off
        let planes = |r: &mut dyn FnMut() -> f64| Array3::from_shape_fn((3, nact, nb), |_| r());
        let (m_a, m_b) = (
            Array2::from_shape_fn((nact, nb), |_| r()),
            Array2::from_shape_fn((nact, nb), |_| r()),
        );
        let (md_a, md_b) = (planes(&mut r), planes(&mut r));
        // ∇ρ_σ must be the one built from m_σ (as `rho_grad_from_m` does in
        // production): the regrouping into X uses ∂_aρ = 2 Σ_μ m ∂_aχ.
        let (_, ga) = rho_grad_from_m(&ao.chi, &ao.dchi, &m_a);
        let (_, gb) = rho_grad_from_m(&ao.chi, &ao.dchi, &m_b);
        let v: Vec<Vec<f64>> = (0..7).map(|_| (0..nb).map(|_| r()).collect()).collect();
        let ch_a = SpinChannel {
            m: &m_a,
            mdchi: &md_a,
            rho: &rho_a,
            vrho: &v[0],
            vsig_same: &v[1],
            vsig_cross: &v[2],
            vtau: &v[3],
            grad_same: &ga,
            grad_cross: &gb,
        };
        let ch_b = SpinChannel {
            m: &m_b,
            mdchi: &md_b,
            rho: &rho_b,
            vrho: &v[4],
            vsig_same: &v[5],
            vsig_cross: &v[2],
            vtau: &v[6],
            grad_same: &gb,
            grad_cross: &ga,
        };
        let f: Vec<f64> = (0..nb).map(|_| r()).collect();
        for mgga in [false, true] {
            let run = |k: BatchKernel| {
                let mut g = Array2::<f64>::zeros((natoms, 3));
                let chans = [&ch_a, &ch_b];
                match k {
                    BatchKernel::Fused => {
                        open_tail::<false>(&mut g, &map, &ao, &bp, floor, mgga, chans, &f)
                    }
                    BatchKernel::Budget => {
                        open_tail::<true>(&mut g, &map, &ao, &bp, floor, mgga, chans, &f)
                    }
                    BatchKernel::Legacy => {
                        legacy::open_tail(&mut g, &map, &ao, &bp, floor, mgga, chans, &f)
                    }
                }
                g
            };
            let (new, old, t) = (
                run(BatchKernel::Fused),
                run(BatchKernel::Legacy),
                run(BatchKernel::Budget),
            );
            // Depth: products + nb fold + 10 partials + nact col fold + 3 +
            // nact scatter + nb grid loop + nb weight loop.
            let g = gamma(PRODUCT_ROUNDINGS + nb + 10 + nact + 3 + nact + 2 * nb);
            for ((&n, &o), &tt) in new.iter().zip(old.iter()).zip(t.iter()) {
                assert!(
                    (n - o).abs() <= 2.0 * g * tt,
                    "mgga {mgga}: fused {n:e} vs legacy {o:e}, bound {:e}",
                    2.0 * g * tt
                );
            }
        }
    }

    #[test]
    fn water_pbe() {
        check(WATER, 1, "PBE", ClosedKind::Gga);
    }

    #[test]
    fn water_b3lyp() {
        check(WATER, 1, "B3LYP", ClosedKind::Gga);
    }

    #[test]
    fn water_scan() {
        check(WATER, 1, "SCAN", ClosedKind::Mgga);
    }

    #[test]
    fn nh2_uks_pbe() {
        check(NH2, 2, "PBE", ClosedKind::Gga);
    }

    #[test]
    fn nh2_uks_scan() {
        check(NH2, 2, "SCAN", ClosedKind::Mgga);
    }
}
