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
//!    `D` is gathered to `(nact, nact)`, and the per-batch math is the
//!    whole-grid path's own helpers (`gga_ao_partials`, `mgga_tau_partials`,
//!    `tau_spatial_grad`, `gga_weight_columns`, `scatter_partials`) with their
//!    rayon fan-out switched off. Each term is the same expression over the
//!    same points; only the batching of the sums differs.
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

/// Weight response `Σ_g weight1[g][B] · f_g` for every atom B.
fn add_weight_response(grad: &mut Array2<f64>, w1: &[&[[f64; 3]]], f: &[f64]) {
    let natoms = grad.nrows();
    for (g, &fg) in f.iter().enumerate() {
        let row = w1[g];
        for b in 0..natoms {
            grad[(b, 0)] += row[b][0] * fg;
            grad[(b, 1)] += row[b][1] * fg;
            grad[(b, 2)] += row[b][2] * fg;
        }
    }
}

/// One batch of the closed-shell gradient (LDA / GGA / meta-GGA): the terms of
/// `xc_gradient_closed_lda_from_density`, `gga_closed_batch` and
/// `mgga_closed_batch`, over this batch's points and kept functions.
fn closed_batch(
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
    let ddchi = ao.ddchi.as_ref().expect("GGA/mGGA evaluate the AO Hessian");
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
    let partials = gga_ao_partials(&m, &mdchi, dchi, ddchi, &c, &t_rho, false);
    scatter_partials(&partials, map, &mut grad);
    if kind == ClosedKind::Mgga {
        let tau_partials = mgga_tau_partials(&mdchi, ddchi, &t_tau, false);
        scatter_partials(&tau_partials, map, &mut grad);
    }

    let f: Vec<f64> = (0..nb).map(|g| eps_total[g] * rho[g]).collect();
    add_weight_response(&mut grad, &bp.w1, &f);

    let hcol = hess_cols(dchi, ddchi, &m, &mdchi);
    let dtau = (kind == ClosedKind::Mgga).then(|| tau_spatial_grad(&mdchi, ddchi, false));
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
    grad
}

/// One batch of the spin-polarized gradient (GGA or meta-GGA; LDA components
/// enter with `v_σ = 0`): the terms of `uks_gga_batch` / `uks_mgga_batch`.
#[allow(clippy::too_many_arguments)]
fn open_batch(
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
    let ddchi = ao.ddchi.as_ref().expect("UKS evaluates the AO Hessian");
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
    // AO-derivative term per spin (`add_spin_contribution` / `add_spin`).
    #[allow(clippy::too_many_arguments)]
    let add_spin = |m_s: &Array2<f64>,
                    mdchi_s: &Array3<f64>,
                    vrho_s: &[f64],
                    vsig_same: &[f64],
                    vsig_cross: &[f64],
                    vtau_s: &[f64],
                    grad_same: &Array2<f64>,
                    grad_cross: &Array2<f64>,
                    rho_s: &[f64],
                    grad_out: &mut Array2<f64>| {
        let mut t_rho = vec![0.0_f64; nb];
        let mut t_tau = vec![0.0_f64; nb];
        let mut c = Array2::<f64>::zeros((3, nb));
        for g in 0..nb {
            if crate::vxc::above_density_floor(rho_s[g], floor) {
                let w = bp.weights[g];
                t_rho[g] = w * vrho_s[g];
                t_tau[g] = w * vtau_s[g];
                for b in 0..3 {
                    c[(b, g)] = w
                        * (2.0 * vsig_same[g] * grad_same[(b, g)]
                            + vsig_cross[g] * grad_cross[(b, g)]);
                }
            }
        }
        let partials = gga_ao_partials(m_s, mdchi_s, dchi, ddchi, &c, &t_rho, false);
        scatter_partials(&partials, map, grad_out);
        if mgga {
            let tau_partials = mgga_tau_partials(mdchi_s, ddchi, &t_tau, false);
            scatter_partials(&tau_partials, map, grad_out);
        }
    };
    add_spin(
        &m_a, &mdchi_a, &vrho_a, &vsig_aa, &vsig_ab, &vtau_a, &grad_a, &grad_b, &rho_a, &mut grad,
    );
    add_spin(
        &m_b, &mdchi_b, &vrho_b, &vsig_bb, &vsig_ab, &vtau_b, &grad_b, &grad_a, &rho_b, &mut grad,
    );

    let f: Vec<f64> = (0..nb)
        .map(|g| eps_total[g] * (rho_a[g] + rho_b[g]))
        .collect();
    add_weight_response(&mut grad, &bp.w1, &f);

    let hess_a = hess_cols(dchi, ddchi, &m_a, &mdchi_a);
    let hess_b = hess_cols(dchi, ddchi, &m_b, &mdchi_b);
    let dtau = mgga.then(|| {
        (
            tau_spatial_grad(&mdchi_a, ddchi, false),
            tau_spatial_grad(&mdchi_b, ddchi, false),
        )
    });
    for g in 0..nb {
        if !crate::vxc::above_density_floor(rho_a[g] + rho_b[g], floor) {
            continue;
        }
        let a = bp.home[g];
        let w = bp.weights[g];
        let (vra, vrb) = (vrho_a[g], vrho_b[g]);
        let (vsaa, vsbb, vsab) = (vsig_aa[g], vsig_bb[g], vsig_ab[g]);
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
                Some((da, db)) => vtau_a[g] * da[axis][g] + vtau_b[g] * db[axis][g],
                None => 0.0,
            };
            grad[(a, axis)] += w * (rho_piece + sig_piece + tau_piece);
        }
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
                    closed_batch(&xc_w, ck, &ds, &sub_map, natoms, &ao, &bp)
                }
                SpinDensity::Open(da, db, mgga) => {
                    let (dsa, dsb) = (gather_sub(da, &ao.funcs), gather_sub(db, &ao.funcs));
                    open_batch(&xc_w, mgga, &dsa, &dsb, &sub_map, natoms, &ao, &bp)
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
