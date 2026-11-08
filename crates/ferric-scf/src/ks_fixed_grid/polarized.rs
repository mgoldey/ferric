//! Spin-polarized (UKS) LDA and GGA exchange-correlation on the same grids as
//! [`FixedGridKs`], and its first and second nuclear derivatives: the XC half
//! of the analytic UKS Hessian (`crate::hessian::uks_moving_grid_hessian`).
//!
//! The energy is `E_xc = Σ_g W_g f(ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ)` with
//! `ρ_s = Σ D^s_μν χ_μ χ_ν`, `g^s = ∇ρ_s` and `σ_st = g^s·g^t`. Every per-spin
//! AO form (`ρ^s_x`, `g^s_x`, `ρ^s_xy`, `g^s_xy`, the AO-derivative rows of a
//! potential) is the closed-shell one of the parent module with `D → D^s`, and
//! is computed by the same functions on a per-spin view of the batch (the AO
//! planes are shared). What is new is the spin algebra (proved term by term in
//! `scripts/uks_xc_hessian_sympy.py`, against exact sympy derivatives with
//! `D^α ≠ D^β`):
//!
//! ```text
//! F^s  = W f_{ρs}                    G^α = W (2 f_σαα g^α + f_σαβ g^β)   (β: α ↔ β)
//! δσ_αα = 2 g^α·g1^α,  δσ_αβ = g^α·g1^β + g^β·g1^α,  δσ_ββ = 2 g^β·g1^β
//! δF^s = W [Σ_t f_{ρs ρt} ρ1^t + Σ_u f_{ρs σu} δσ_u]
//! δG^α = W [2 g^α δf_σαα + g^β δf_σαβ + 2 f_σαα g1^α + f_σαβ g1^β],
//!        δf_v = Σ_t f_{v ρt} ρ1^t + Σ_u f_{v σu} δσ_u                   (β: α ↔ β)
//! ∂E/∂x            = Σ_s F^s ρ^s_x + G^s·g^s_x
//! ∂²E/∂x∂y|_D     = Σ_s [ρ^s_x δF^s[y] + g^s_x·δG^s[y] + F^s ρ^s_xy + G^s·g^s_xy]
//! ∂V^s/∂x|_D       = V[δF^s[x], δG^s[x]] + (AO-derivative rows of V[F^s, G^s])
//! V1^s[D1^α, D1^β] = V[δF^s, δG^s] at ρ1^t = Σ D1^t χχ, g1^t = 2 Σ D1^t ∂χ χ
//! ```
//!
//! where `δF^s[y]` is the response to the perturbation `(ρ^t_y, g^t_y)` of BOTH
//! spins. The moving grid is the parent's construction unchanged: the
//! point-fixed `G_h` gains `W' (∂f)ᵀ + ∂f W'ᵀ + W'' f` with
//! `∂f = Σ_s f_{ρs} ρ^s_x + Gu^s·g^s_x` (`Gu^s = G^s/W`), the first-order Fock
//! of spin `s` gains `V[W'_x f_{ρs}, W'_x Gu^s]`, and the home rows are filled
//! by translation invariance.
//!
//! Density floor: each spin's potential pieces (`F^s`, `G^s` and their
//! responses) are gated on its own `ρ_s`, as `ferric_dft`'s polarized `V_xc`
//! gates `V^s`; the energy density `f` is not gated (nor is the production
//! energy).

use super::*;
use ferric_dft::libxc::xc_def_from_name_nspin;
use std::sync::Arc;

/// Deliberately broken spin-polarized constructions for the mutation tests;
/// `Default` is production.
#[derive(Debug, Clone, Copy)]
pub struct SpinMutants {
    /// Scale on the cross-spin second derivatives `f_{ρα ρβ}`, `f_{ρα σββ}`,
    /// `f_{ρβ σαα}` and `f_{σαα σββ}` (0 drops the αβ kernel block).
    pub cross_kernel: f64,
    /// Replace each spin's perturbation by the spin average before the kernel
    /// (the "total-density kernel": right only for spin-symmetric
    /// perturbations, so a singlet cannot detect it).
    pub total_kernel: bool,
    /// Build both spins' kernel rows from the α row (the closed-shell
    /// symmetric kernel applied to an open shell).
    pub alpha_rows_for_beta: bool,
}

impl Default for SpinMutants {
    fn default() -> Self {
        SpinMutants {
            cross_kernel: 1.0,
            total_kernel: false,
            alpha_rows_for_beta: false,
        }
    }
}

/// The variables of `f`: `ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ`.
const NVAR: usize = 5;

/// libxc derivatives of `f` summed over the component functionals, per point.
struct PolKernel {
    exc: Array1<f64>,
    /// `f_{ρs}`, `(2, n)`.
    vr: Array2<f64>,
    /// `f_{σαα}, f_{σαβ}, f_{σββ}`, `(3, n)`.
    vs: Array2<f64>,
    /// All second derivatives, symmetric, `(NVAR, NVAR, n)` (the σ rows and
    /// columns are zero for an LDA).
    k2: Array3<f64>,
    /// 1 where `ρ_s` is above the density floor, else 0, per spin.
    mask: [Array1<f64>; 2],
}

impl PolKernel {
    /// The points `c0..c1`.
    fn slice(&self, c0: usize, c1: usize) -> Self {
        PolKernel {
            exc: self
                .exc
                .slice(s![c0.min(self.exc.len())..c1.min(self.exc.len())])
                .to_owned(),
            vr: self.vr.slice(s![.., c0..c1]).to_owned(),
            vs: self.vs.slice(s![.., c0..c1]).to_owned(),
            k2: self.k2.slice(s![.., .., c0..c1]).to_owned(),
            mask: [0, 1].map(|s| self.mask[s].slice(s![c0..c1]).to_owned()),
        }
    }
}

/// A spin-polarized batch: the closed-shell [`Batch`] view of each spin
/// (shared AO planes; `p = P^s`, `rho = ρ_s`, `fw = F^s`, `grad = (Q^s, g^s,
/// G^s)`; its `k` is unused) and the polarized kernel.
struct PolBatch {
    spins: [Batch; 2],
    w: Array1<f64>,
    k: PolKernel,
    /// `ρ_α + ρ_β`.
    rho: Array1<f64>,
}

/// [`FixedGridKs`] evaluated spin-polarized: the same points, weights and
/// riding construction, the functional's `nspin = 2` definition.
pub struct PolarizedKs<'a> {
    grid: &'a FixedGridKs,
    xc: XcDef,
}

/// Pack index of `f_{ρs σu}` (libxc `v2rhosigma`).
fn rs_index(s: usize, u: usize) -> usize {
    3 * s + u
}

/// Pack index of `f_{σu σv}`, `u ≤ v` (libxc `v2sigma2`).
fn ss_index(u: usize, v: usize) -> usize {
    let (u, v) = (u.min(v), u.max(v));
    [[0, 1, 2], [1, 3, 4], [2, 4, 5]][u][v]
}

fn dot_rows(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
    (a * b).sum_axis(Axis(0))
}

impl FixedGridKs {
    /// This grid evaluated spin-polarized (the functional's `nspin = 2`
    /// definition).
    pub fn polarized(&self) -> Result<PolarizedKs<'_>, FerricError> {
        let xc = xc_def_from_name_nspin(&self.name, 2)
            .map_err(|e| FerricError::General(format!("spin-polarized XC grid: libxc: {e:?}")))?;
        Ok(PolarizedKs { grid: self, xc })
    }
}

impl<'a> PolarizedKs<'a> {
    /// The underlying grid.
    pub fn grid(&self) -> &'a FixedGridKs {
        self.grid
    }

    fn kernel(
        &self,
        rho: [&Array1<f64>; 2],
        sigma: Option<[Array1<f64>; 3]>,
        second: bool,
        m: &XcMutants,
    ) -> PolKernel {
        let n = rho[0].len();
        let gga = self.grid.gga;
        let mut rho_in = vec![0.0; 2 * n];
        let mut sig_in = vec![0.0; 3 * n];
        for g in 0..n {
            rho_in[2 * g] = rho[0][g];
            rho_in[2 * g + 1] = rho[1][g];
            if let Some(s) = &sigma {
                for u in 0..3 {
                    sig_in[3 * g + u] = s[u][g];
                }
            }
        }
        let mut exc = Array1::<f64>::zeros(n);
        let mut vr = Array2::<f64>::zeros((2, n));
        let mut vs = Array2::<f64>::zeros((3, n));
        let mut k2 = Array3::<f64>::zeros((NVAR, NVAR, n));
        for (i, f) in self.xc.funcs.iter().enumerate() {
            let wi = self.xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
            let mut e = vec![0.0; n];
            let mut v = vec![0.0; 2 * n];
            let mut vsig = vec![0.0; 3 * n];
            let mut frr = vec![0.0; 3 * n];
            let mut frs = vec![0.0; 6 * n];
            let mut fss = vec![0.0; 6 * n];
            if f.family() == FunctionalFamily::Lda {
                f.eval_lda_polarized(&rho_in, &mut e, &mut v);
                if second {
                    f.eval_lda_fxc_polarized(&rho_in, &mut frr);
                }
            } else {
                f.eval_gga_polarized(&rho_in, &sig_in, &mut e, &mut v, &mut vsig);
                if second {
                    f.eval_gga_fxc_polarized(&rho_in, &sig_in, &mut frr, &mut frs, &mut fss);
                }
            }
            for g in 0..n {
                exc[g] += wi * e[g];
                for s in 0..2 {
                    vr[(s, g)] += wi * v[2 * g + s];
                }
                for u in 0..3 {
                    vs[(u, g)] += wi * vsig[3 * g + u];
                }
                for s in 0..2 {
                    for t in 0..2 {
                        k2[(s, t, g)] += wi * frr[3 * g + s + t];
                    }
                    for u in 0..3 {
                        let x = wi * frs[6 * g + rs_index(s, u)];
                        k2[(s, 2 + u, g)] += x;
                        k2[(2 + u, s, g)] += x;
                    }
                }
                for u in 0..3 {
                    for w in 0..3 {
                        k2[(2 + u, 2 + w, g)] += wi * fss[6 * g + ss_index(u, w)];
                    }
                }
            }
        }
        // Mutants: the gradient-dependent derivatives and the cross-spin block.
        let gt = m.gradient_terms;
        if gt != 1.0 {
            vs *= gt;
            for a in 0..NVAR {
                for b in 0..NVAR {
                    if a >= 2 || b >= 2 {
                        k2.slice_mut(s![a, b, ..]).mapv_inplace(|x| x * gt);
                    }
                }
            }
        }
        let cross = m.spin.cross_kernel;
        if cross != 1.0 {
            // (ρα, ρβ), (ρα, σββ), (ρβ, σαα), (σαα, σββ)
            for (a, b) in [(0, 1), (0, 4), (1, 2), (2, 4)] {
                for (x, y) in [(a, b), (b, a)] {
                    k2.slice_mut(s![x, y, ..]).mapv_inplace(|v| v * cross);
                }
            }
        }
        if !gga {
            vs.fill(0.0);
            for a in 0..NVAR {
                for b in 0..NVAR {
                    if a >= 2 || b >= 2 {
                        k2.slice_mut(s![a, b, ..]).fill(0.0);
                    }
                }
            }
        }
        let floor = DensityFloor::for_xc(&self.xc);
        let mask =
            [0, 1].map(|s| rho[s].mapv(|r| f64::from(u8::from(above_density_floor(r, floor)))));
        PolKernel {
            exc,
            vr,
            vs,
            k2,
            mask,
        }
    }

    /// One batch at the spin densities `d`.
    fn batch(
        &self,
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
        range: (usize, usize),
        level: Level,
        second: bool,
        m: &XcMutants,
    ) -> Result<PolBatch, FerricError> {
        let grid = self.grid;
        let (g0, g1) = range;
        let level = match (grid.gga, level) {
            (true, l) => l.max(Level::Grad),
            (false, Level::Third) => Level::Hess,
            (false, l) => l,
        };
        let ao = Arc::new(eval_ao(geo, &grid.pts[g0..g1], level)?);
        let w = Array1::from(grid.w[g0..g1].to_vec());
        let p = d.map(|ds| ds.dot(&ao.chi));
        let rho = [0, 1].map(|s| (&ao.chi * &p[s]).sum_axis(Axis(0)));
        let gd = grid
            .gga
            .then(|| [0, 1].map(|s| FixedGridKs::gradient_densities(d[s], &ao, &p[s])));
        let sigma = gd.as_ref().map(|gd| {
            [
                dot_rows(&gd[0].1, &gd[0].1),
                dot_rows(&gd[0].1, &gd[1].1),
                dot_rows(&gd[1].1, &gd[1].1),
            ]
        });
        let k = self.kernel([&rho[0], &rho[1]], sigma, second, m);
        let fw = [0, 1].map(|s| &(&w * &k.vr.row(s)) * &k.mask[s]);
        let gws = gd
            .as_ref()
            .map(|gd| [0, 1].map(|s| &gu(&k, [&gd[0].1, &gd[1].1], s) * &w));
        let [pa, pb] = p;
        let [ra, rb] = rho.clone();
        let [fa, fb] = fw;
        let mut grads: [Option<GradPart>; 2] = [None, None];
        if let (Some([(qa, ga), (qb, gb)]), Some([gwa, gwb])) = (gd, gws) {
            grads = [
                Some(GradPart {
                    q: qa,
                    g: ga,
                    gw: gwa,
                }),
                Some(GradPart {
                    q: qb,
                    g: gb,
                    gw: gwb,
                }),
            ];
        }
        let [gpa, gpb] = grads;
        let n = g1 - g0;
        let view = |p, rho, fw, grad| Batch {
            ao: Arc::clone(&ao),
            p,
            rho,
            w: w.clone(),
            k: Kernel::zeros(n),
            fw,
            grad,
        };
        let spins = [view(pa, ra, fa, gpa), view(pb, rb, fb, gpb)];
        Ok(PolBatch {
            spins,
            w,
            k,
            rho: &rho[0] + &rho[1],
        })
    }

    /// `(E_xc, [V^α, V^β])` at the spin densities `d`.
    pub fn energy_vxc(
        &self,
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
    ) -> Result<(f64, [Array2<f64>; 2]), FerricError> {
        let mut e = 0.0;
        let mut v = [0, 1].map(|_| Array2::<f64>::zeros(d[0].dim()));
        for range in self.grid.ranges() {
            let b = self.batch(geo, d, range, Level::Value, false, &XcMutants::default())?;
            e += (&b.w * &b.rho * &b.k.exc).sum();
            for (s, vs) in v.iter_mut().enumerate() {
                let sb = &b.spins[s];
                *vs += &vmat(&sb.ao, &sb.fw, sb.grad.as_ref().map(|g| &g.gw));
            }
        }
        Ok((e, v.map(|m| 0.5 * (&m + &m.t()))))
    }

    /// The AO-derivative nuclear gradient of `E_xc` on the FIXED grid,
    /// `(natoms, 3)`.
    pub fn gradient(
        &self,
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
    ) -> Result<Array2<f64>, FerricError> {
        let mut grad = Array2::<f64>::zeros((geo.natoms(), 3));
        let level = if self.grid.gga {
            Level::Hess
        } else {
            Level::Grad
        };
        for range in self.grid.ranges() {
            let b = self.batch(geo, d, range, level, false, &XcMutants::default())?;
            for sb in &b.spins {
                let mut g = FixedGridKs::rho_derivative(geo, sb).dot(&sb.fw);
                if let Some(gp) = &sb.grad {
                    let gx = FixedGridKs::grad_rho_derivative(geo, sb);
                    for gam in 0..3 {
                        g += &gx.index_axis(Axis(1), gam).dot(&gp.gw.row(gam));
                    }
                }
                for x in 0..g.len() {
                    grad[(x / 3, x % 3)] += g[x];
                }
            }
        }
        Ok(grad)
    }

    /// `(ρ^s_x, g^s_x)` of both spins for every coordinate.
    fn spin_derivatives(
        geo: &XcGeometry<'_>,
        b: &PolBatch,
    ) -> [(Array2<f64>, Option<Array3<f64>>); 2] {
        [0, 1].map(|s| {
            let sb = &b.spins[s];
            (
                FixedGridKs::rho_derivative(geo, sb),
                sb.grad
                    .as_ref()
                    .map(|_| FixedGridKs::grad_rho_derivative(geo, sb)),
            )
        })
    }

    /// `(δF^s, δG^s)` of both spins for the perturbation of coordinate `x`.
    fn coordinate_response(
        b: &PolBatch,
        der: &[(Array2<f64>, Option<Array3<f64>>); 2],
        x: usize,
        m: &SpinMutants,
    ) -> [(Array1<f64>, Option<Array2<f64>>); 2] {
        let rho1 = [der[0].0.row(x).to_owned(), der[1].0.row(x).to_owned()];
        let g1 = match (&der[0].1, &der[1].1) {
            (Some(a), Some(bb)) => Some([
                a.index_axis(Axis(0), x).to_owned(),
                bb.index_axis(Axis(0), x).to_owned(),
            ]),
            _ => None,
        };
        let g = b.spins[0]
            .grad
            .as_ref()
            .zip(b.spins[1].grad.as_ref())
            .map(|(a, bb)| [&a.g, &bb.g]);
        pol_response(
            &b.k,
            &b.w,
            g,
            [&rho1[0], &rho1[1]],
            g1.as_ref().map(|g| [&g[0], &g[1]]),
            m,
        )
    }

    /// The fixed-grid skeleton of one batch (points fixed).
    fn batch_skeleton(
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
        b: &PolBatch,
        m: &XcMutants,
    ) -> Array2<f64> {
        let n3 = 3 * geo.natoms();
        let der = Self::spin_derivatives(geo, b);
        let mut h = Array2::<f64>::zeros((n3, n3));
        let npts = b.w.len();
        let mut df = [0, 1].map(|_| Array2::<f64>::zeros((n3, npts)));
        let mut dg = [0, 1].map(|s| {
            der[s]
                .1
                .as_ref()
                .map(|_| Array3::<f64>::zeros((n3, 3, npts)))
        });
        for x in 0..n3 {
            let r = Self::coordinate_response(b, &der, x, &m.spin);
            for (s, (f, g)) in r.into_iter().enumerate() {
                df[s].row_mut(x).assign(&f);
                if let (Some(dg), Some(g)) = (dg[s].as_mut(), g) {
                    dg.index_axis_mut(Axis(0), x).assign(&g);
                }
            }
        }
        for s in 0..2 {
            h += &der[s].0.dot(&df[s].t());
            if let (Some(gx), Some(dg)) = (&der[s].1, &dg[s]) {
                for gam in 0..3 {
                    h += &gx
                        .index_axis(Axis(1), gam)
                        .dot(&dg.index_axis(Axis(1), gam).t());
                }
            }
        }
        h *= m.skeleton_fxc;
        let mut second = Array2::<f64>::zeros((n3, n3));
        for s in 0..2 {
            FixedGridKs::add_same_centre(geo, &b.spins[s], &mut second);
            for (c, &(lo, hi)) in geo.ranges.iter().enumerate() {
                FixedGridKs::add_cross_centre(geo, d[s], &b.spins[s], (c, lo, hi), &mut second);
            }
        }
        h.scaled_add(m.skeleton_vxc, &second);
        h
    }

    /// `∂²E_xc/∂x∂y` at fixed spin densities on the FIXED grid, `(3N, 3N)`.
    pub fn skeleton_hessian(
        &self,
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
        m: &XcMutants,
    ) -> Result<Array2<f64>, FerricError> {
        let n3 = 3 * geo.natoms();
        let mut h = Array2::<f64>::zeros((n3, n3));
        for range in self.grid.ranges() {
            let b = self.batch(geo, d, range, Level::Third, true, m)?;
            h += &Self::batch_skeleton(geo, d, &b, m);
        }
        Ok(h)
    }

    /// `∂V^s/∂x` at fixed spin densities on the FIXED grid: `[α, β]`, each of
    /// length 3N.
    pub fn first_order_fock(
        &self,
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
        m: &XcMutants,
    ) -> Result<[Vec<Array2<f64>>; 2], FerricError> {
        let nbf = d[0].nrows();
        let n3 = 3 * geo.natoms();
        let mut v = [0, 1].map(|_| vec![Array2::<f64>::zeros((nbf, nbf)); n3]);
        for range in self.grid.ranges() {
            let b = self.batch(geo, d, range, Level::Hess, true, m)?;
            let der = Self::spin_derivatives(geo, &b);
            for x in 0..n3 {
                let r = Self::coordinate_response(&b, &der, x, &m.spin);
                for (s, (df, dg)) in r.iter().enumerate() {
                    v[s][x] += &(m.fock_fxc * vmat(&b.spins[s].ao, df, dg.as_ref()));
                }
            }
            for (s, vs) in v.iter_mut().enumerate() {
                FixedGridKs::add_ao_derivative_fock(geo, &b.spins[s], m.fock_vxc, None, vs);
            }
        }
        Ok(v)
    }

    /// Point-fixed `∂f/∂R_x` (unweighted), `(3N, npts)`.
    fn f_derivative(b: &PolBatch, der: &[(Array2<f64>, Option<Array3<f64>>); 2]) -> Array2<f64> {
        let mut df = Array2::<f64>::zeros(der[0].0.dim());
        let g = b.spins[0]
            .grad
            .as_ref()
            .zip(b.spins[1].grad.as_ref())
            .map(|(a, bb)| [&a.g, &bb.g]);
        for s in 0..2 {
            let fr = &b.k.vr.row(s) * &b.k.mask[s];
            df += &(&der[s].0 * &fr.insert_axis(Axis(0)));
            if let (Some(gx), Some(g)) = (&der[s].1, g) {
                let gus = &gu(&b.k, g, s);
                for gam in 0..3 {
                    df += &(&gx.index_axis(Axis(1), gam) * &gus.row(gam).insert_axis(Axis(0)));
                }
            }
        }
        df
    }

    /// The MOVING-grid skeleton (points riding their home atom, Becke weights
    /// differentiated), `(3N, 3N)`. Batches run in parallel and are added in
    /// batch order.
    pub fn skeleton_hessian_moving(
        &self,
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
        m: &XcMutants,
    ) -> Result<Array2<f64>, FerricError> {
        let grid = self.grid;
        let riding = grid.riding_for(geo)?;
        let mm = &m.moving;
        let n3 = 3 * geo.natoms();
        let ranges = grid.home_ranges(riding, mm.unsplit_batches);
        let mut h = Array2::<f64>::zeros((n3, n3));
        fold_batches(
            &ranges,
            2 * n3 * n3,
            |&(g0, g1, home)| {
                let b = self.batch(geo, d, (g0, g1), Level::Third, true, m)?;
                let mut gh = Self::batch_skeleton(geo, d, &b, m);
                let f = &b.rho * &b.k.exc;
                let wd = weight_derivatives(geo.mol, riding, (g0, g1), &grid.pts, home, Some(&f));
                if let Some(w2f) = &wd.hess_f {
                    gh.scaled_add(mm.weight_hess, w2f);
                }
                let der = Self::spin_derivatives(geo, &b);
                let df = Self::f_derivative(&b, &der);
                let mut riding_cross = None;
                if mm.riding_cross {
                    let c = riding_derivatives(&wd.grad, home).dot(&df.t());
                    riding_cross = Some(&c + &c.t());
                } else {
                    let c = wd.grad.dot(&df.t());
                    gh.scaled_add(mm.weight_cross, &(&c + &c.t()));
                }
                Ok((riding_cross, translation_fill(gh.view(), home, mm.fill)))
            },
            |(riding_cross, filled)| {
                if let Some(c) = riding_cross {
                    h.scaled_add(mm.weight_cross, &c);
                }
                h += &filled;
            },
        )?;
        Ok(h)
    }

    /// `∂V^s/∂x` at fixed spin densities on the MOVING grid: `[α, β]`, each of
    /// length 3N. Batches run in parallel and are added in batch order.
    pub fn first_order_fock_moving(
        &self,
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
        m: &XcMutants,
    ) -> Result<[Vec<Array2<f64>>; 2], FerricError> {
        let grid = self.grid;
        let riding = grid.riding_for(geo)?;
        let mm = &m.moving;
        let nbf = d[0].nrows();
        let n3 = 3 * geo.natoms();
        let ranges = grid.home_ranges(riding, mm.unsplit_batches);
        let zero = || [0, 1].map(|_| vec![Array2::<f64>::zeros((nbf, nbf)); n3]);
        let mut v = zero();
        fold_batches(
            &ranges,
            2 * n3 * nbf * nbf,
            |&(g0, g1, home)| {
                let mut part = zero();
                let b = self.batch(geo, d, (g0, g1), Level::Hess, true, m)?;
                let der = Self::spin_derivatives(geo, &b);
                let w1 = weight_derivatives(geo.mol, riding, (g0, g1), &grid.pts, home, None).grad;
                let g = b.spins[0]
                    .grad
                    .as_ref()
                    .zip(b.spins[1].grad.as_ref())
                    .map(|(a, bb)| [&a.g, &bb.g]);
                let gus = g.map(|g| [gu(&b.k, g, 0), gu(&b.k, g, 1)]);
                for x in (0..n3).filter(|x| x / 3 != home) {
                    let r = Self::coordinate_response(&b, &der, x, &m.spin);
                    let wx = w1.row(x).to_owned();
                    for (s, (df, dg)) in r.iter().enumerate() {
                        // The kernel term and the weight term in ONE V matrix
                        // (V is linear in its weights).
                        let wf = &(&wx * &b.k.vr.row(s)) * &b.k.mask[s];
                        let fw = m.fock_fxc * df + mm.fock_weight * &wf;
                        let gw = match (dg, gus.as_ref()) {
                            (Some(dg), Some(gu)) => {
                                let wg = &gu[s] * &wx.view().insert_axis(Axis(0));
                                Some(m.fock_fxc * dg + mm.fock_weight * &wg)
                            }
                            _ => None,
                        };
                        let c = vmat(&b.spins[s].ao, &fw, gw.as_ref());
                        part[s][x] += &c;
                        part[s][3 * home + x % 3].scaled_add(mm.fill, &c);
                    }
                }
                for (s, ps) in part.iter_mut().enumerate() {
                    FixedGridKs::add_ao_derivative_fock(
                        geo,
                        &b.spins[s],
                        m.fock_vxc,
                        Some((home, mm.fill)),
                        ps,
                    );
                }
                Ok(part)
            },
            |part| {
                for (vs, ps) in v.iter_mut().zip(&part) {
                    for (vx, px) in vs.iter_mut().zip(ps) {
                        *vx += px;
                    }
                }
            },
        )?;
        Ok(v)
    }

    /// [`Self::skeleton_hessian`] or [`Self::skeleton_hessian_moving`].
    pub fn skeleton_hessian_for(
        &self,
        mode: GridMode,
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
        m: &XcMutants,
    ) -> Result<Array2<f64>, FerricError> {
        match mode {
            GridMode::Fixed => self.skeleton_hessian(geo, d, m),
            GridMode::Moving => self.skeleton_hessian_moving(geo, d, m),
        }
    }

    /// [`Self::first_order_fock`] or [`Self::first_order_fock_moving`].
    pub fn first_order_fock_for(
        &self,
        mode: GridMode,
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
        m: &XcMutants,
    ) -> Result<[Vec<Array2<f64>>; 2], FerricError> {
        match mode {
            GridMode::Fixed => self.first_order_fock(geo, d, m),
            GridMode::Moving => self.first_order_fock_moving(geo, d, m),
        }
    }

    /// The spin-coupled kernel `(D1^α, D1^β) ↦ (V1^α, V1^β)` of the response,
    /// with the AOs and the functional's second derivatives at the converged
    /// spin densities kept resident. `factor` scales the response (1 in
    /// production); `m.gradient_terms` and `m.spin` are the mutants.
    pub fn response(
        &self,
        geo: &XcGeometry<'_>,
        d: [&Array2<f64>; 2],
        factor: f64,
        m: &XcMutants,
    ) -> Result<PolarizedXcResponse, FerricError> {
        let grid = self.grid;
        let nbf = d[0].nrows();
        let npts = grid.npts();
        grid.check_response_table(nbf)?;
        let mut chi = Array2::<f64>::zeros((nbf, npts));
        let mut dchi = grid.gga.then(|| Array3::<f64>::zeros((3, nbf, npts)));
        let mut g = grid
            .gga
            .then(|| [0, 1].map(|_| Array2::<f64>::zeros((3, npts))));
        let mut k = PolKernel {
            exc: Array1::zeros(0),
            vr: Array2::zeros((2, npts)),
            vs: Array2::zeros((3, npts)),
            k2: Array3::zeros((NVAR, NVAR, npts)),
            mask: [Array1::zeros(npts), Array1::zeros(npts)],
        };
        for (g0, g1) in grid.ranges() {
            let b = self.batch(geo, d, (g0, g1), Level::Value, true, m)?;
            let ao = &b.spins[0].ao;
            chi.slice_mut(s![.., g0..g1]).assign(&ao.chi);
            if let (Some(dst), Some(gd)) = (dchi.as_mut(), g.as_mut()) {
                dst.slice_mut(s![.., .., g0..g1]).assign(ao.dchi());
                for s in 0..2 {
                    let gp = b.spins[s].grad.as_ref().expect("gradient part");
                    gd[s].slice_mut(s![.., g0..g1]).assign(&gp.g);
                }
            }
            k.vs.slice_mut(s![.., g0..g1]).assign(&b.k.vs);
            k.k2.slice_mut(s![.., .., g0..g1]).assign(&b.k.k2);
            for s in 0..2 {
                k.mask[s].slice_mut(s![g0..g1]).assign(&b.k.mask[s]);
            }
        }
        Ok(PolarizedXcResponse {
            chi,
            dchi,
            w: Array1::from(grid.w.clone()),
            g,
            k,
            factor,
            spin: m.spin,
        })
    }
}

/// `Gu^s = 2 f_{σss} g^s + f_{σαβ} g^t` (unweighted, gated on `ρ_s`), `(3, n)`.
fn gu(k: &PolKernel, g: [&Array2<f64>; 2], s: usize) -> Array2<f64> {
    let t = 1 - s;
    let own = 2.0 * &(&k.vs.row(2 * s) * &k.mask[s]);
    let cross = &k.vs.row(1) * &k.mask[s];
    g[s] * &own.insert_axis(Axis(0)) + g[t] * &cross.insert_axis(Axis(0))
}

/// The polarized response `(δF^s, δG^s)` of both spins to a perturbation
/// `(ρ1^t, g1^t)` of both spins (see the module doc).
fn pol_response(
    k: &PolKernel,
    w: &Array1<f64>,
    g: Option<[&Array2<f64>; 2]>,
    rho1: [&Array1<f64>; 2],
    g1: Option<[&Array2<f64>; 2]>,
    m: &SpinMutants,
) -> [(Array1<f64>, Option<Array2<f64>>); 2] {
    let (rho1, g1): ([Array1<f64>; 2], Option<[Array2<f64>; 2]>) = if m.total_kernel {
        let r = 0.5 * &(rho1[0] + rho1[1]);
        let gg = g1.map(|g1| {
            let a = 0.5 * &(g1[0] + g1[1]);
            [a.clone(), a]
        });
        ([r.clone(), r], gg)
    } else {
        (rho1.map(|r| r.clone()), g1.map(|g1| g1.map(|x| x.clone())))
    };
    let mut pert: Vec<Array1<f64>> = vec![rho1[0].clone(), rho1[1].clone()];
    if let (Some(g), Some(g1)) = (g, g1.as_ref()) {
        pert.push(2.0 * &dot_rows(g[0], &g1[0]));
        pert.push(&dot_rows(g[0], &g1[1]) + &dot_rows(g[1], &g1[0]));
        pert.push(2.0 * &dot_rows(g[1], &g1[1]));
    }
    let dfv = |v: usize| -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(w.len());
        for (u, p) in pert.iter().enumerate() {
            out += &(&k.k2.slice(s![v, u, ..]) * p);
        }
        out
    };
    [0, 1].map(|s| {
        // Mutant: both spins take the α kernel rows.
        let (own_r, own_s) = if m.alpha_rows_for_beta {
            (0, 2)
        } else {
            (s, 2 + 2 * s)
        };
        let gate = &k.mask[s] * w;
        let df = &dfv(own_r) * &gate;
        let dg = match (g, g1.as_ref()) {
            (Some(g), Some(g1)) => {
                let t = 1 - s;
                let dss = dfv(own_s);
                let dab = dfv(3);
                let own = 2.0 * &k.vs.row(2 * s);
                let cross = k.vs.row(1);
                let mut out = g[s] * &(2.0 * &dss).insert_axis(Axis(0));
                out += &(g[t] * &dab.view().insert_axis(Axis(0)));
                out += &(&g1[s] * &own.view().insert_axis(Axis(0)));
                out += &(&g1[t] * &cross.insert_axis(Axis(0)));
                Some(out * gate.view().insert_axis(Axis(0)))
            }
            _ => None,
        };
        (df, dg)
    })
}

/// `(D1^α, D1^β) ↦ (V1^α, V1^β)` for symmetric response densities: the
/// spin-coupled XC kernel of the UKS response (CPKS).
pub struct PolarizedXcResponse {
    chi: Array2<f64>,
    dchi: Option<Array3<f64>>,
    w: Array1<f64>,
    g: Option<[Array2<f64>; 2]>,
    k: PolKernel,
    factor: f64,
    spin: SpinMutants,
}

impl PolarizedXcResponse {
    /// The XC potential responses of both spins to `(d1[0], d1[1])`: point
    /// chunks in parallel, added in chunk order.
    pub fn apply(&self, d1: [&Array2<f64>; 2]) -> [Array2<f64>; 2] {
        let nbf = self.chi.nrows();
        let mut v = [0, 1].map(|_| Array2::<f64>::zeros((nbf, nbf)));
        let chunks = response_chunks(self.chi.ncols());
        fold_batches(
            &chunks,
            2 * nbf * nbf,
            |&(c0, c1)| Ok(self.apply_chunk(d1, c0, c1)),
            |[pa, pb]| {
                v[0] += &pa;
                v[1] += &pb;
            },
        )
        .expect("the response kernel chunks are infallible");
        v
    }

    /// The contribution of the points `c0..c1`.
    fn apply_chunk(&self, d1: [&Array2<f64>; 2], c0: usize, c1: usize) -> [Array2<f64>; 2] {
        let chi = self.chi.slice(s![.., c0..c1]);
        let dchi = self.dchi.as_ref().map(|d| d.slice(s![.., .., c0..c1]));
        let pd = d1.map(|x| x.dot(&chi));
        let rho1 = [0, 1].map(|s| (&chi * &pd[s]).sum_axis(Axis(0)));
        let g1 = dchi.map(|dchi| [0, 1].map(|s| gradient_response(dchi, &pd[s])));
        let k = self.k.slice(c0, c1);
        let w = self.w.slice(s![c0..c1]).to_owned();
        let g = self
            .g
            .as_ref()
            .map(|g| [0, 1].map(|s| g[s].slice(s![.., c0..c1]).to_owned()));
        let r = pol_response(
            &k,
            &w,
            g.as_ref().map(|g| [&g[0], &g[1]]),
            [&rho1[0], &rho1[1]],
            g1.as_ref().map(|g| [&g[0], &g[1]]),
            &self.spin,
        );
        r.map(|(df, dg)| {
            let gw = dg.map(|dg| self.factor * dg);
            vmat_views(chi, dchi, &(self.factor * &df), gw.as_ref())
        })
    }
}
