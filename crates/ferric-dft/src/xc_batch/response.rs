//! The XC-kernel (f_xc) response δV_xc[δD] on the SCF's own screened, batched
//! grid — the Newton / second-order SCF Hessian-vector product's XC term.
//!
//! # Why
//!
//! The dense kernels in [`crate::fxc`] rebuild the grid and a full `(nbf,
//! npts)` χ/∇χ cache per Newton macro step, re-evaluate libxc's second
//! derivatives on every product, and back-project over the whole grid with
//! no screening, twice (one GEMM chain per spin) even for a closed shell.
//! Measured on C8H18/def2-TZVP PBE (wiki/notes/scf-second-order-cost-2026-11):
//! 3.6 s per product against ~0.2 s for a whole production V_xc build.
//!
//! # What
//!
//! [`ScreenedFxc`] borrows the `ScreenedGrid` (batches, kept shells, and
//! the resident or recomputed compact AO blocks) of the SCF's `KsXc` /
//! `KsXcUks`. Built once per Newton macro step at the reference density, it
//! stores per-point kernel data in batch order: libxc's second derivatives
//! are evaluated once there, never per product. Each product is then one
//! fused pass per batch: `Φ = δD_sub·[χ | ∂χ]`, the point-local linearised
//! potential, and one `V_b = A·Bᵀ` GEMM, reduced in the production fixed
//! group order (bit-identical across rayon thread counts).
//!
//! # Same mathematics
//!
//! The point-local algebra is `crate::fxc::linearize_point` — the kernel
//! the dense [`crate::fxc::MggaFxcKernel`] applies, equal to the GGA and LDA
//! kernels on τ-free functionals — or, for a restricted reference, its exact
//! one-spin regrouping `crate::fxc::restricted_coefficients`. The
//! back-projection is the production pass-2 GEMM: with the point factors
//! `s/2 = ½·w·u`, `f = w·w⃗` and `t/2 = ¼·w·t`, `V_b + V_bᵀ` is
//! `Σ_g w [u χχ + w⃗·∇(χχ) + ½ t ∇χ·∇χ]`, the dense kernels' symmetrised sum.
//!
//! Screening and density floor are the production V_xc's: a shell dropped
//! from a batch contributes neither to δρ there nor to δV, and a point is
//! live where the production V_xc is live (closed shell: total ρ above the
//! floor; polarized: each spin on its own ρ_σ). So this response is the EXACT
//! derivative of the screened V_xc the SCF builds; the dense kernels differ
//! from it by the screening error (and, for the restricted RKS Newton path,
//! by the ρ/2-vs-ρ gating band `(floor, 2·floor]`).

use ndarray::Array2;

use ferric_core::memory::pool::{reserve_global, Reservation};
use ferric_integrals::blas_threads::with_blas_threads;

use super::{
    batch_density_one, batch_vxc_one, BatchDensity, BatchFactors, DensityInput, Exec, KernelXc,
    ScreenedGrid,
};
use crate::ao_grid::GtoEvalError;
use crate::fxc::{
    linearize_point, point_hessians, restricted_coefficients, PointHessian, PointPerturbation,
};
use crate::grid::GridPoint;
use crate::libxc::{FunctionalFamily, XcDef};
use crate::vxc::{above_density_floor, DensityFloor};

/// Pool label for the per-point kernel data a [`ScreenedFxc`] holds.
const FXC_LABEL: &str = "KS-DFT f_xc response kernel (screened batches)";

/// Message for an AO evaluation failing after `ScreenedGrid::build` already
/// evaluated every shell once (it cannot: the only error depends on `l`).
const EVAL_INFALLIBLE: &str =
    "per-shell AO evaluation failed in the f_xc response, after the screened grid's \
     constructor evaluated every shell";

/// Restricted (closed-shell) per-point kernel data of one batch, quadrature
/// weight folded in: the `(δρ_α, g·∇δρ_α, δτ_α)` coefficients of u, of
/// `2δv_σαα + δv_σαβ` and of t, the ∇δρ_α coefficient, and g = ∇ρ_α.
/// Points below the density floor hold zeros.
struct RestrictedPoints {
    u: Vec<[f64; 3]>,
    s: Vec<[f64; 3]>,
    t: Vec<[f64; 3]>,
    gam: Vec<f64>,
    grad: Vec<[f64; 3]>,
}

/// Spin-polarized per-point kernel data of one batch.
struct PolarizedPoints {
    hess: Vec<PointHessian>,
    vsigma: Vec<[f64; 3]>,
    grad: Vec<[[f64; 3]; 2]>,
    wt: Vec<f64>,
    live: Vec<[bool; 2]>,
}

impl RestrictedPoints {
    /// No points: a batch with no kept shell, which every product skips.
    fn empty() -> Self {
        Self {
            u: Vec::new(),
            s: Vec::new(),
            t: Vec::new(),
            gam: Vec::new(),
            grad: Vec::new(),
        }
    }
}

impl PolarizedPoints {
    /// See [`RestrictedPoints::empty`].
    fn empty() -> Self {
        Self {
            hess: Vec::new(),
            vsigma: Vec::new(),
            grad: Vec::new(),
            wt: Vec::new(),
            live: Vec::new(),
        }
    }
}

enum Points {
    Restricted(Vec<RestrictedPoints>),
    Polarized(Vec<PolarizedPoints>),
}

/// The f_xc response at one reference density, on a borrowed screened grid.
/// See the module docs.
pub struct ScreenedFxc<'g> {
    screened: &'g ScreenedGrid,
    grid: &'g [GridPoint],
    has_gga: bool,
    has_mgga: bool,
    points: Points,
    /// Pool charge for `points`, held for this value's lifetime.
    _charge: Reservation,
}

impl std::fmt::Debug for ScreenedFxc<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScreenedFxc")
            .field("npts", &self.grid.len())
            .field("restricted", &matches!(self.points, Points::Restricted(_)))
            .field("has_gga", &self.has_gga)
            .field("has_mgga", &self.has_mgga)
            .finish()
    }
}

/// `(has_gga, has_mgga)` of a definition, as the production integrations
/// classify it.
fn families(xc: &XcDef) -> (bool, bool) {
    let has_mgga = xc
        .funcs
        .iter()
        .any(|f| matches!(f.family(), FunctionalFamily::MetaGga));
    let has_gga = xc
        .funcs
        .iter()
        .any(|f| !matches!(f.family(), FunctionalFamily::Lda));
    (has_gga, has_mgga)
}

/// Refuse a definition the kernel cannot differentiate: libxc would abort the
/// process on a meta-GGA without a second derivative.
fn check_fxc(xc: &XcDef) -> Result<(), String> {
    if let Some(f) = xc
        .funcs
        .iter()
        .find(|f| f.family() == FunctionalFamily::MetaGga && !f.has_fxc())
    {
        return Err(format!(
            "screened f_xc: libxc has no second derivative for {f:?}"
        ));
    }
    Ok(())
}

/// libxc-interleaved polarized inputs `(ρ 2/pt, σ 3/pt, τ 2/pt)` of one batch
/// from per-spin densities.
fn packed(a: &BatchDensity, b: &BatchDensity, mgga: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nb = a.rho.len();
    let mut rho = vec![0.0; 2 * nb];
    let mut sigma = vec![0.0; 3 * nb];
    let mut tau = vec![0.0; if mgga { 2 * nb } else { 0 }];
    for j in 0..nb {
        rho[2 * j] = a.rho[j];
        rho[2 * j + 1] = b.rho[j];
        let ga = [a.grad[0][j], a.grad[1][j], a.grad[2][j]];
        let gb = [b.grad[0][j], b.grad[1][j], b.grad[2][j]];
        sigma[3 * j] = ga[0] * ga[0] + ga[1] * ga[1] + ga[2] * ga[2];
        sigma[3 * j + 1] = ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2];
        sigma[3 * j + 2] = gb[0] * gb[0] + gb[1] * gb[1] + gb[2] * gb[2];
        if mgga {
            tau[2 * j] = a.tau[j];
            tau[2 * j + 1] = b.tau[j];
        }
    }
    (rho, sigma, tau)
}

fn grad_at(p: &BatchDensity, j: usize) -> [f64; 3] {
    [p.grad[0][j], p.grad[1][j], p.grad[2][j]]
}

/// The density with every entry exactly halved (×0.5 is exact in binary
/// barring underflow): ρ_α = ρ_β = ρ/2 of a closed shell.
fn halved(p: BatchDensity) -> BatchDensity {
    let h = |v: Vec<f64>| v.into_iter().map(|x| 0.5 * x).collect::<Vec<_>>();
    let [gx, gy, gz] = p.grad;
    BatchDensity {
        rho: h(p.rho),
        grad: [h(gx), h(gy), h(gz)],
        tau: h(p.tau),
    }
}

impl<'g> ScreenedFxc<'g> {
    /// Restricted reference: the closed-shell total density `d`
    /// (ρ_α = ρ_β = ρ/2). `xc_pol` is the functional resolved with
    /// `nspin = 2` (the kernel is libxc's polarized one at equal spins, as
    /// the dense RKS path has always used). Points are live where the
    /// closed-shell V_xc is: total ρ above the functional's floor.
    pub(crate) fn restricted(
        screened: &'g ScreenedGrid,
        grid: &'g [GridPoint],
        xc_pol: &XcDef,
        d: &Array2<f64>,
    ) -> Result<Self, String> {
        check_fxc(xc_pol)?;
        let (has_gga, has_mgga) = families(xc_pol);
        let floor = DensityFloor::for_xc(xc_pol);
        let per_pt = 8 * (3 + 3 + if has_mgga { 3 } else { 0 } + 1 + 3);
        let charge = reserve_global(FXC_LABEL, per_pt * grid.len()).map_err(|e| e.to_string())?;
        let exec = Exec::auto();
        let din = DensityInput::Dense(d);
        let pts = with_blas_threads(exec.blas, || {
            screened.reduce_groups_with(
                exec.serial,
                0,
                || KernelXc::for_group(xc_pol, exec.serial),
                |kxc, bi, _| -> Result<Result<RestrictedPoints, String>, GtoEvalError> {
                    let b = &screened.batches[bi];
                    let nb = b.idx.len();
                    if b.funcs.is_empty() {
                        // No kept shell: every product skips this batch, so
                        // it needs no kernel data (and no libxc call).
                        return Ok(Ok(RestrictedPoints::empty()));
                    }
                    let ao = screened.batch_ao(bi, grid)?;
                    let full = batch_density_one(&ao, &din.gather(&b.funcs), nb, has_mgga);
                    let live: Vec<bool> = full
                        .rho
                        .iter()
                        .map(|&r| above_density_floor(r, floor))
                        .collect();
                    let half = halved(full);
                    let (rho, sigma, tau) = packed(&half, &half, has_mgga);
                    let (hess, vsig) = match point_hessians(kxc.get(), &rho, &sigma, &tau) {
                        Ok(x) => x,
                        Err(e) => return Ok(Err(e)),
                    };
                    let mut out = RestrictedPoints {
                        u: vec![[0.0; 3]; nb],
                        s: vec![[0.0; 3]; nb],
                        t: vec![[0.0; 3]; if has_mgga { nb } else { 0 }],
                        gam: vec![0.0; nb],
                        grad: (0..nb).map(|j| grad_at(&half, j)).collect(),
                    };
                    for (j, &gi) in b.idx.iter().enumerate() {
                        if !live[j] {
                            continue;
                        }
                        let wt = grid[gi as usize].weight;
                        let (u, s, t, gam) = restricted_coefficients(&hess[j], &vsig[j], wt);
                        out.u[j] = u;
                        out.s[j] = s;
                        if has_mgga {
                            out.t[j] = t;
                        }
                        out.gam[j] = gam;
                    }
                    Ok(Ok(out))
                },
            )
        })
        .map_err(|e| format!("{e:?}"))?;
        let pts = pts.1.into_iter().collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            screened,
            grid,
            has_gga,
            has_mgga,
            points: Points::Restricted(pts),
            _charge: charge,
        })
    }

    /// Spin-polarized reference `(d_a, d_b)`; `xc_pol` resolved with
    /// `nspin = 2`. Spin σ is live where ρ_σ is above the floor, as the
    /// polarized V_xc gates it.
    pub(crate) fn polarized(
        screened: &'g ScreenedGrid,
        grid: &'g [GridPoint],
        xc_pol: &XcDef,
        (d_a, d_b): (&Array2<f64>, &Array2<f64>),
    ) -> Result<Self, String> {
        check_fxc(xc_pol)?;
        let (has_gga, has_mgga) = families(xc_pol);
        let floor = DensityFloor::for_xc(xc_pol);
        let per_pt = 8 * (std::mem::size_of::<PointHessian>() / 8 + 3 + 6 + 1) + 2;
        let charge = reserve_global(FXC_LABEL, per_pt * grid.len()).map_err(|e| e.to_string())?;
        let exec = Exec::auto();
        let (ia, ib) = (DensityInput::Dense(d_a), DensityInput::Dense(d_b));
        let pts = with_blas_threads(exec.blas, || {
            screened.reduce_groups_with(
                exec.serial,
                0,
                || KernelXc::for_group(xc_pol, exec.serial),
                |kxc, bi, _| -> Result<Result<PolarizedPoints, String>, GtoEvalError> {
                    let b = &screened.batches[bi];
                    let nb = b.idx.len();
                    if b.funcs.is_empty() {
                        // As in the restricted build: never read.
                        return Ok(Ok(PolarizedPoints::empty()));
                    }
                    let ao = screened.batch_ao(bi, grid)?;
                    let pa = batch_density_one(&ao, &ia.gather(&b.funcs), nb, has_mgga);
                    let pb = batch_density_one(&ao, &ib.gather(&b.funcs), nb, has_mgga);
                    let (rho, sigma, tau) = packed(&pa, &pb, has_mgga);
                    let (hess, vsigma) = match point_hessians(kxc.get(), &rho, &sigma, &tau) {
                        Ok(x) => x,
                        Err(e) => return Ok(Err(e)),
                    };
                    Ok(Ok(PolarizedPoints {
                        hess,
                        vsigma,
                        grad: (0..nb)
                            .map(|j| [grad_at(&pa, j), grad_at(&pb, j)])
                            .collect(),
                        wt: b.idx.iter().map(|&gi| grid[gi as usize].weight).collect(),
                        live: (0..nb)
                            .map(|j| {
                                [
                                    above_density_floor(pa.rho[j], floor),
                                    above_density_floor(pb.rho[j], floor),
                                ]
                            })
                            .collect(),
                    }))
                },
            )
        })
        .map_err(|e| format!("{e:?}"))?;
        let pts = pts.1.into_iter().collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            screened,
            grid,
            has_gga,
            has_mgga,
            points: Points::Polarized(pts),
            _charge: charge,
        })
    }

    /// True when built from a restricted reference.
    pub fn is_restricted(&self) -> bool {
        matches!(self.points, Points::Restricted(_))
    }

    /// δV_xc (one spin block) for the restricted perturbation
    /// δD_α = δD_β = `dd`. The closed-shell Fock carries V_xc once, so this
    /// is the whole restricted XC response. Panics on a polarized reference.
    pub fn apply_restricted(&self, dd: &Array2<f64>) -> Array2<f64> {
        let Points::Restricted(pts) = &self.points else {
            panic!("ScreenedFxc::apply_restricted on a polarized reference");
        };
        let (has_gga, has_mgga) = (self.has_gga, self.has_mgga);
        let din = DensityInput::Dense(dd);
        let exec = Exec::auto();
        let sg = self.screened;
        let mut v = with_blas_threads(exec.blas, || {
            sg.reduce_groups(exec.serial, 1, |bi, accs| {
                let b = &sg.batches[bi];
                if b.funcs.is_empty() {
                    return Ok(());
                }
                let nb = b.idx.len();
                let ao = sg.batch_ao(bi, self.grid)?;
                let p = batch_density_one(&ao, &din.gather(&b.funcs), nb, has_mgga);
                let c = &pts[bi];
                let mut fac = BatchFactors::zeros(nb, has_gga, has_mgga);
                for j in 0..nb {
                    let d = grad_at(&p, j);
                    let g = c.grad[j];
                    let (a, bb) = (p.rho[j], g[0] * d[0] + g[1] * d[1] + g[2] * d[2]);
                    let cc = if has_mgga { p.tau[j] } else { 0.0 };
                    let lin = |k: [f64; 3]| k[0] * a + k[1] * bb + k[2] * cc;
                    fac.s_half[j] = 0.5 * lin(c.u[j]);
                    if has_gga {
                        let s = lin(c.s[j]);
                        for ax in 0..3 {
                            fac.f[ax][j] = s * g[ax] + c.gam[j] * d[ax];
                        }
                    }
                    if has_mgga {
                        fac.t_half[j] = 0.25 * lin(c.t[j]);
                    }
                }
                batch_vxc_one(
                    &ao,
                    &b.funcs,
                    nb,
                    &fac,
                    (has_gga, has_mgga, false),
                    &mut accs[0],
                );
                Ok(())
            })
        })
        .expect(EVAL_INFALLIBLE);
        v.pop().expect("one matrix")
    }

    /// (δV_α, δV_β) for the perturbation (δD_α, δD_β). Panics on a
    /// restricted reference.
    pub fn apply_polarized(
        &self,
        dd_a: &Array2<f64>,
        dd_b: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let Points::Polarized(pts) = &self.points else {
            panic!("ScreenedFxc::apply_polarized on a restricted reference");
        };
        let (has_gga, has_mgga) = (self.has_gga, self.has_mgga);
        let (ia, ib) = (DensityInput::Dense(dd_a), DensityInput::Dense(dd_b));
        let exec = Exec::auto();
        let sg = self.screened;
        let mut v = with_blas_threads(exec.blas, || {
            sg.reduce_groups(exec.serial, 2, |bi, accs| {
                let b = &sg.batches[bi];
                if b.funcs.is_empty() {
                    return Ok(());
                }
                let nb = b.idx.len();
                let ao = sg.batch_ao(bi, self.grid)?;
                let pa = batch_density_one(&ao, &ia.gather(&b.funcs), nb, has_mgga);
                let pb = batch_density_one(&ao, &ib.gather(&b.funcs), nb, has_mgga);
                let c = &pts[bi];
                let mut fac = [
                    BatchFactors::zeros(nb, has_gga, has_mgga),
                    BatchFactors::zeros(nb, has_gga, has_mgga),
                ];
                for j in 0..nb {
                    let p = PointPerturbation {
                        grad: c.grad[j],
                        drho: [pa.rho[j], pb.rho[j]],
                        dgrad: [grad_at(&pa, j), grad_at(&pb, j)],
                        dtau: if has_mgga {
                            [pa.tau[j], pb.tau[j]]
                        } else {
                            [0.0; 2]
                        },
                    };
                    let (u, w, t) = linearize_point(&c.hess[j], &c.vsigma[j], &p);
                    let wt = c.wt[j];
                    for s in 0..2 {
                        if !c.live[j][s] {
                            continue;
                        }
                        fac[s].s_half[j] = 0.5 * (wt * u[s]);
                        if has_gga {
                            for ax in 0..3 {
                                fac[s].f[ax][j] = wt * w[s][ax];
                            }
                        }
                        if has_mgga {
                            fac[s].t_half[j] = 0.25 * (wt * t[s]);
                        }
                    }
                }
                let flags = (has_gga, has_mgga, false);
                let [fa, fb] = &fac;
                let [acc_a, acc_b] = accs else {
                    unreachable!("two accumulators")
                };
                batch_vxc_one(&ao, &b.funcs, nb, fa, flags, acc_a);
                batch_vxc_one(&ao, &b.funcs, nb, fb, flags, acc_b);
                Ok(())
            })
        })
        .expect(EVAL_INFALLIBLE);
        let vb = v.pop().expect("two matrices");
        let va = v.pop().expect("two matrices");
        (va, vb)
    }
}

#[cfg(test)]
impl ScreenedFxc<'_> {
    /// The restricted product with every operand replaced by its absolute
    /// value: AO blocks, `δD`, and the per-point coefficients. Each element is
    /// then the sum of the magnitudes of every term [`Self::apply_restricted`]
    /// adds (density sums, point algebra, quadrature), the `S` of a
    /// Higham-style `γ_m·S` rounding bound.
    fn apply_restricted_magnitude(&self, dd: &Array2<f64>) -> Array2<f64> {
        let Points::Restricted(pts) = &self.points else {
            panic!("restricted only");
        };
        let (has_gga, has_mgga) = (self.has_gga, self.has_mgga);
        let dd_abs = dd.mapv(f64::abs);
        let din = DensityInput::Dense(&dd_abs);
        let sg = self.screened;
        let abs3 = |x: [f64; 3]| [x[0].abs(), x[1].abs(), x[2].abs()];
        let mut v = sg
            .reduce_groups(true, 1, |bi, accs| {
                let b = &sg.batches[bi];
                if b.funcs.is_empty() {
                    return Ok(());
                }
                let nb = b.idx.len();
                let ao = sg.batch_ao(bi, self.grid)?.mapv(f64::abs);
                let p = batch_density_one(&ao, &din.gather(&b.funcs), nb, has_mgga);
                let c = &pts[bi];
                let mut fac = BatchFactors::zeros(nb, has_gga, has_mgga);
                for j in 0..nb {
                    let d = grad_at(&p, j);
                    let g = abs3(c.grad[j]);
                    let (a, bb) = (p.rho[j], g[0] * d[0] + g[1] * d[1] + g[2] * d[2]);
                    let cc = if has_mgga { p.tau[j] } else { 0.0 };
                    let lin = |k: [f64; 3]| {
                        let k = abs3(k);
                        k[0] * a + k[1] * bb + k[2] * cc
                    };
                    fac.s_half[j] = 0.5 * lin(c.u[j]);
                    if has_gga {
                        let s = lin(c.s[j]);
                        for ax in 0..3 {
                            fac.f[ax][j] = s * g[ax] + c.gam[j].abs() * d[ax];
                        }
                    }
                    if has_mgga {
                        fac.t_half[j] = 0.25 * lin(c.t[j]);
                    }
                }
                batch_vxc_one(
                    &ao,
                    &b.funcs,
                    nb,
                    &fac,
                    (has_gga, has_mgga, false),
                    &mut accs[0],
                );
                Ok(())
            })
            .expect(EVAL_INFALLIBLE);
        v.pop().expect("one matrix")
    }
}

#[cfg(test)]
impl ScreenedFxc<'_> {
    /// [`Self::apply_polarized`] on absolute values (AO blocks, both `δD_σ`,
    /// the per-point Hessians, `v_σ` and reference gradients): per spin, the
    /// summed magnitude of every term that product adds.
    fn apply_polarized_magnitude(
        &self,
        dd_a: &Array2<f64>,
        dd_b: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let Points::Polarized(pts) = &self.points else {
            panic!("polarized only");
        };
        let (has_gga, has_mgga) = (self.has_gga, self.has_mgga);
        let (aa, ab) = (dd_a.mapv(f64::abs), dd_b.mapv(f64::abs));
        let (ia, ib) = (DensityInput::Dense(&aa), DensityInput::Dense(&ab));
        let sg = self.screened;
        let abs3 = |x: [f64; 3]| [x[0].abs(), x[1].abs(), x[2].abs()];
        let mut v = sg
            .reduce_groups(true, 2, |bi, accs| {
                let b = &sg.batches[bi];
                if b.funcs.is_empty() {
                    return Ok(());
                }
                let nb = b.idx.len();
                let ao = sg.batch_ao(bi, self.grid)?.mapv(f64::abs);
                let pa = batch_density_one(&ao, &ia.gather(&b.funcs), nb, has_mgga);
                let pb = batch_density_one(&ao, &ib.gather(&b.funcs), nb, has_mgga);
                let c = &pts[bi];
                let mut fac = [
                    BatchFactors::zeros(nb, has_gga, has_mgga),
                    BatchFactors::zeros(nb, has_gga, has_mgga),
                ];
                for j in 0..nb {
                    let mut h = c.hess[j];
                    for row in h.iter_mut() {
                        for x in row.iter_mut() {
                            *x = x.abs();
                        }
                    }
                    let p = PointPerturbation {
                        grad: [abs3(c.grad[j][0]), abs3(c.grad[j][1])],
                        drho: [pa.rho[j], pb.rho[j]],
                        dgrad: [grad_at(&pa, j), grad_at(&pb, j)],
                        dtau: if has_mgga {
                            [pa.tau[j], pb.tau[j]]
                        } else {
                            [0.0; 2]
                        },
                    };
                    let (u, w, t) = linearize_point(&h, &abs3(c.vsigma[j]), &p);
                    let wt = c.wt[j];
                    for s in 0..2 {
                        fac[s].s_half[j] = 0.5 * (wt * u[s]);
                        if has_gga {
                            for ax in 0..3 {
                                fac[s].f[ax][j] = wt * w[s][ax];
                            }
                        }
                        if has_mgga {
                            fac[s].t_half[j] = 0.25 * (wt * t[s]);
                        }
                    }
                }
                let flags = (has_gga, has_mgga, false);
                let [acc_a, acc_b] = accs else {
                    unreachable!("two accumulators")
                };
                batch_vxc_one(&ao, &b.funcs, nb, &fac[0], flags, acc_a);
                batch_vxc_one(&ao, &b.funcs, nb, &fac[1], flags, acc_b);
                Ok(())
            })
            .expect(EVAL_INFALLIBLE);
        let vb = v.pop().expect("two matrices");
        let va = v.pop().expect("two matrices");
        (va, vb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fxc::MggaFxcKernel;
    use crate::grid::AtomicGridConfig;
    use crate::ks::{KsXc, KsXcUks};
    use crate::libxc::xc_def_from_name_nspin;
    use crate::xc_batch::{AoStorage, XcBatchConfig};
    use crate::xc_trait::{UksXcContribution, XcContribution};
    use ferric_core::basis::{self, BasisSet};
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    const WATER: &str = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n";
    const OH: &str = "2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";

    fn grid_cfg() -> AtomicGridConfig {
        AtomicGridConfig {
            n_radial: 50,
            n_angular: 110,
            ..Default::default()
        }
    }

    fn batch(thresh: f64) -> XcBatchConfig {
        XcBatchConfig {
            screen_thresh: thresh,
            storage: AoStorage::Resident,
            ..Default::default()
        }
    }

    /// ferric's own HF densities `(D_α, D_β)` for `xyz` at multiplicity
    /// `mult` in 6-31G (UHF; a closed shell gives D_α = D_β).
    fn hf_reference(xyz: &str, mult: usize) -> (Molecule, BasisSet, Array2<f64>, Array2<f64>) {
        let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
        let bs = basis::bundled("6-31g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds =
            ferric_scf::screening::SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let cfg = ferric_scf::uhf::UhfConfig {
            energy_conv: 1e-9,
            density_conv: 1e-7,
            ..Default::default()
        };
        let res =
            ferric_scf::uhf::solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg)
                .unwrap();
        let d_b = res.density_beta.clone().unwrap();
        (mol, bs, res.density_alpha, d_b)
    }

    /// A deterministic symmetric perturbation with entries in [-s, s].
    fn perturbation(n: usize, s: f64, seed: u64) -> Array2<f64> {
        let mut x = seed;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let m = Array2::from_shape_fn((n, n), |_| s * next());
        0.5 * (&m + &m.t())
    }

    fn max_abs(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0, |m, &x| m.max(x.abs()))
    }

    /// γ_m = m·u/(1 − m·u).
    fn gamma(m: usize) -> f64 {
        let u = f64::EPSILON / 2.0;
        m as f64 * u / (1.0 - m as f64 * u)
    }

    /// Exactness anchor: with screening off (`screen_thresh = 0`, every shell
    /// kept on every batch) the screened restricted response equals the
    /// dense [`MggaFxcKernel`] restricted response (the kernel the RKS Newton
    /// and TDDFT paths validated against FD of production V_xc), elementwise,
    /// within 2·γ_m·S: `S` = [`ScreenedFxc::apply_restricted_magnitude`] and
    /// m = npts + 4·nbf + 64: the quadrature and density sums (derived), plus
    /// 64 terms for the point algebra and the reference kernel, whose libxc
    /// inputs the two paths round differently. That 64 is NOT derived (libxc
    /// carries no rounding bound); it rests on the measured margin, err/bound
    /// 1.4e-5..1.2e-4 across these cases. LDA, GGA, global hybrid and τ-only
    /// meta-GGA, at ferric's own HF water density.
    ///
    /// Mutation (applied and reverted, 2026-11-11): dropping the cross-spin
    /// `H_i,ρβ` column in `restricted_coefficients` fails every functional
    /// (PBE: max|Δ| 2.6e-1 against a bound of 4e-12).
    #[test]
    fn unscreened_restricted_response_equals_dense_kernel() {
        let (mol, bs, d_a, _) = hf_reference(WATER, 1);
        let d = 2.0 * &d_a;
        let n = d.nrows();
        let dd = perturbation(n, 0.01, 0x9E37_79B9_7F4A_7C15);
        for name in ["LDA", "PBE", "B3LYP", "r2SCAN"] {
            let ks = KsXc::new_with_batch_config(
                &mol,
                &bs,
                name,
                &grid_cfg(),
                &grid_cfg(),
                None,
                None,
                batch(0.0),
            )
            .unwrap();
            let xc_pol = xc_def_from_name_nspin(name, 2).unwrap();
            let sf = ks.restricted_fxc(&xc_pol, &d).unwrap().unwrap();
            let got = sf.apply_restricted(&dd);
            let mag = sf.apply_restricted_magnitude(&dd);
            let dense = MggaFxcKernel::new(&mol, &bs, xc_pol, &grid_cfg()).unwrap();
            assert_eq!(dense.grid.len(), ks.grid.len(), "{name}: grids differ");
            let half = 0.5 * &d;
            let r = dense.reference_restricted(&half, &half).unwrap();
            let (want, _) = dense.apply_with_ref(&r, &dd, &dd);
            let g = gamma(ks.grid.len() + 4 * n + 64);
            let mut worst = 0.0_f64;
            for ((x, y), s) in got.iter().zip(&want).zip(&mag) {
                let bound = 2.0 * g * s;
                worst = worst.max((x - y).abs() / bound.max(f64::MIN_POSITIVE));
                assert!(
                    (x - y).abs() <= bound,
                    "{name}: |Δ| {:.3e} > bound {bound:.3e}",
                    (x - y).abs()
                );
            }
            eprintln!(
                "{name}: max|δV| {:.3e}, max|screened − dense| {:.3e}, worst err/bound {worst:.3e}",
                max_abs(&want),
                max_abs(&(&got - &want))
            );
        }
    }

    /// Exactness anchor, polarized: screening off, the screened polarized
    /// response equals the dense [`MggaFxcKernel`] (per-spin floor) on ferric's
    /// open-shell HF OH density with δD_α ≠ δD_β, elementwise within 2·γ_m·S_σ
    /// (`S_σ` from [`ScreenedFxc::apply_polarized_magnitude`], m as in the
    /// restricted anchor).
    #[test]
    fn unscreened_polarized_response_equals_dense_kernel() {
        let (mol, bs, d_a, d_b) = hf_reference(OH, 2);
        let n = d_a.nrows();
        let dd_a = perturbation(n, 0.01, 0x1234_5678_9ABC_DEF1);
        let dd_b = perturbation(n, 0.01, 0x0FED_CBA9_8765_4321);
        for name in ["LDA", "PBE", "B3LYP", "r2SCAN"] {
            let ks = KsXcUks::new_with_batch_config(
                &mol,
                &bs,
                name,
                &grid_cfg(),
                &grid_cfg(),
                None,
                None,
                batch(0.0),
            )
            .unwrap();
            let xc_pol = xc_def_from_name_nspin(name, 2).unwrap();
            let sf = ks.polarized_fxc(&xc_pol, &d_a, &d_b).unwrap().unwrap();
            let got = sf.apply_polarized(&dd_a, &dd_b);
            let mag = sf.apply_polarized_magnitude(&dd_a, &dd_b);
            let dense = MggaFxcKernel::new(&mol, &bs, xc_pol, &grid_cfg()).unwrap();
            let r = dense.reference(&d_a, &d_b).unwrap();
            let want = dense.apply_with_ref(&r, &dd_a, &dd_b);
            let g = gamma(ks.grid.len() + 4 * n + 64);
            let mut worst = 0.0_f64;
            for (x, y, m) in [(&got.0, &want.0, &mag.0), (&got.1, &want.1, &mag.1)] {
                for ((a, b), s) in x.iter().zip(y).zip(m) {
                    let bound = 2.0 * g * s;
                    worst = worst.max((a - b).abs() / bound.max(f64::MIN_POSITIVE));
                    assert!(
                        (a - b).abs() <= bound,
                        "{name}: |Δ| {:.3e} > bound {bound:.3e}",
                        (a - b).abs()
                    );
                }
            }
            eprintln!(
                "{name}: max|δV_α| {:.3e}, max|screened − dense| {:.3e}/{:.3e}, worst err/bound \
                 {worst:.3e}",
                max_abs(&want.0),
                max_abs(&(&got.0 - &want.0)),
                max_abs(&(&got.1 - &want.1))
            );
        }
    }
}
