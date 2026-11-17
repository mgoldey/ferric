//! Exact orbital Hessian of the CASSCF energy at fixed RDMs, and the
//! augmented-Hessian (AH) orbital step of the two-step optimizer.
//!
//! At orbitals `C` with the RDMs `γ`, `Γ` of the CI root held fixed, rotate
//! `C(ε) = C exp(ε X)`, `X` antisymmetric over the nonredundant pairs. Every
//! quantity of [`crate::casscf`] expressed in the rotated orbitals has the
//! frame derivative
//!
//! ```text
//!   dF^I = F^I X − X F^I + Cᵀ (J − ½K)[dD^c] C,   dD^c = 2 (CX)_c C_cᵀ + transpose
//!   dF^A = F^A X − X F^A + Cᵀ (J − ½K)[dD^a] C,   dD^a = (CX)_a γ C_aᵀ + transpose
//!   d(pu|vw) = Σ_r X_rp (ru|vw) + X_ru (pr|vw) + X_rv (pu|rw) + X_rw (pu|rv)
//!   dF_pi = 2 (dF^I + dF^A)_pi,   dF_pt = Σ_u dF^I_pu γ_ut + Σ_uvw d(pu|vw) Γ_tuvw
//! ```
//!
//! and `H_f x = pack 2 (dF − dFᵀ)` is the derivative of
//! [`CasScfProblem::gradient_at_fixed_rdms`](crate::casscf::CasScfProblem::gradient_at_fixed_rdms) along `x` in the rotated frame
//! ([`CasScfProblem::hessian_vector_frame`](crate::casscf::CasScfProblem::hessian_vector_frame)). The `(pr|vw)` and `(pu|rw)`
//! terms are contracted with `Γ` once per macro iteration into
//!
//! ```text
//!   W[(p,t),(r,s)] = Σ_vw (pr|vw) Γ_tsvw  +  Σ_uw (pu|rw) (Γ_tusw + Γ_tuws)
//! ```
//!
//! so that `Σ_uvw [X_ru (pr|vw) + X_rv (pu|rw) + X_rw (pu|rv)] Γ_tuvw =
//! Σ_rs W[(p,t),(r,s)] X_rs`; the first sum is the active-active coupling
//! term through `(pr|vw)`. With `exp(εX) exp(δA) = exp(εX + δA + ½εδ[X, A]
//! + …)`, the symmetric Hessian of `E(C exp(K))` at `K = 0` is
//! `H x = H_f x − ½ pack [G, X]`, `G = 2 (F − Fᵀ)` the full gradient matrix
//! ([`CasScfProblem::hessian_vector`](crate::casscf::CasScfProblem::hessian_vector)). Both density responses have the form
//! `dD = C (Y + Yᵀ) Cᵀ` with `Y` nonzero only in occupied columns
//! (`Y_ri = 2 X_ri` for core `i`; `Y_{r,t} = (X_a γ)_{rt}` for active `t`),
//! so in the MO basis
//!
//! ```text
//!   Cᵀ (J − ½K)[dD] C_occ = L vec(Y),
//!   L[(p,j),(r,k)] = 2 (pj|rk) − ½ (pr|jk) − ½ (pk|rj),   j, k occupied,
//! ```
//!
//! built once per macro iteration from `(pq|jk)`, `(pj|qk)` over the
//! occupied orbitals ([`HessianProducts::OccupiedKernel`](crate::casscf_newton::HessianProducts::OccupiedKernel), the default);
//! the active blocks of the same integrals give `W`. A product is then two
//! GEMVs of size `(nmo n_occ)²`. [`HessianProducts::AoJk`](crate::casscf_newton::HessianProducts::AoJk) takes one AO J/K
//! pass over the two densities per product instead. The Hessian omits the CI
//! response (two-step), so the step is exact Newton for the orbitals at
//! fixed CI.
//!
//! The step: the lowest eigenpair of `[[0, gᵀ], [g, H]]` by Davidson
//! (preconditioner `max(|H_diag|, hdiag_floor) − λ`, `H_diag` the approximate
//! diagonal of [`CasScfProblem::approx_hessian_diag`](crate::casscf::CasScfProblem::approx_hessian_diag)), `s = v/v_0`, stopped
//! at `‖(H − λ) s + g‖ <= η ‖g‖`, `η` set by [`CasScfConfig::ah_forcing`]
//! (`min(0.1, ‖g‖)` by default). The step is scaled to the trust
//! radius (2-norm) and to `max_step` (∞-norm). A step that raises the energy
//! by more than `1e-10` Ha is halved and evaluated again; the radius follows
//! the ratio of the actual to the predicted change `g·s + ½ s·Hs / β`.
//! Without the CI response the Hessian is stiffer than the relaxed one, so
//! each accepted step measures the curvature ratio `β` along itself
//! (`curvature_stretch`) and the next AH step is multiplied by `β`,
//! clamped to `[1, AH_MAX_STRETCH]`; the predicted change uses the same
//! `β`. Equations and measurements:
//! `book/src/17-multireference/05-casscf.md#casscf-proven-optimizers`; prototype
//! `scripts/prototypes/casscf_newton_proto.py`.

use crate::casscf::{CasScfConfig, CasScfPoint, CasScfProblem};
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ndarray::{s, Array1, Array2, ArrayView2};
use ndarray_linalg::{Eigh, UPLO};

/// Initial trust radius of the AH step (2-norm of `κ`, radians).
pub const AH_INITIAL_RADIUS: f64 = 0.4;
/// Largest trust radius the ratio test may grow to.
pub const AH_MAX_RADIUS: f64 = 1.0;
/// Davidson iterations (Hessian-vector products) per AH step.
pub const AH_MAX_DAVIDSON: usize = 40;
/// Largest factor an AH step is lengthened by (`curvature_stretch`).
pub const AH_MAX_STRETCH: f64 = 100.0;
/// Energy rise (Ha) above which a step is rejected and halved, the
/// threshold of the inverse-BFGS reset.
pub const AH_REJECT_RISE: f64 = 1e-10;

/// The fixed-RDM orbital Hessian at one set of orbitals: the matrices its
/// products need, built once per macro iteration
/// ([`CasScfProblem::orbital_hessian`](crate::casscf::CasScfProblem::orbital_hessian)).
pub struct OrbitalHessian {
    pub(crate) c: Array2<f64>,
    pub(crate) fi: Array2<f64>,
    fa: Array2<f64>,
    pub(crate) gmat: Array2<f64>,
    pub(crate) gamma: Array2<f64>,
    /// `Σ_uvw (pu|vw) Γ_tuvw`, `nmo x n_a`.
    two: Array2<f64>,
    /// `(pu|vw)`, `p` over all MOs, flat `[((p n_a + u) n_a + v) n_a + w]`
    /// (the CI-response products of [`crate::casscf_stability`]).
    pub(crate) paaa: Vec<f64>,
    /// `W[(p,t),(r,s)]`, `(nmo n_a) x (nmo n_a)`.
    w: Array2<f64>,
    /// With [`HessianProducts::OccupiedKernel`], the kernel
    /// `L[(p,j),(r,k)] = 2 (pj|rk) − ½ (pr|jk) − ½ (pk|rj)`, `j, k` over the
    /// occupied (core and active) orbitals, `(nmo n_occ) x (nmo n_occ)`.
    pub(crate) occ_kernel: Option<Array2<f64>>,
}

/// How a Hessian-vector product forms the response `Cᵀ (J − ½K)[dD] C` of
/// the core and active densities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianProducts {
    /// From the kernel `L` over occupied MO integrals, built once per macro
    /// iteration: `dD = C (Y + Yᵀ) Cᵀ` with `Y` nonzero in the occupied
    /// columns only, so `Cᵀ (J − ½K)[dD] C_occ = L vec(Y)`, two GEMVs of
    /// size `(nmo n_occ)²` per product.
    OccupiedKernel,
    /// One two-density AO J/K pass per product (the oracle of
    /// [`HessianProducts::OccupiedKernel`]).
    AoJk,
}

/// One AH step: the step `s`, `H s`, and the Davidson record.
#[derive(Debug, Clone)]
pub struct AhStep {
    /// Orbital step over [`CasScfProblem::pairs`](crate::casscf::CasScfProblem::pairs).
    pub step: Vec<f64>,
    /// `H s`.
    pub h_step: Vec<f64>,
    /// Hessian-vector products taken.
    pub products: usize,
    /// `‖(H − λ) s + g‖` at exit.
    pub residual: f64,
    /// The residual met the stopping criterion.
    pub converged: bool,
}

impl CasScfProblem<'_> {
    /// The fixed-RDM orbital Hessian at the point `pt` evaluated at `c`,
    /// with products from the occupied kernel
    /// ([`HessianProducts::OccupiedKernel`]).
    pub fn orbital_hessian(
        &self,
        c: &Array2<f64>,
        pt: &CasScfPoint,
    ) -> Result<OrbitalHessian, FerricError> {
        self.orbital_hessian_with(c, pt, HessianProducts::OccupiedKernel)
    }

    /// The fixed-RDM orbital Hessian at the point `pt` evaluated at `c`:
    /// `(pq|vw)` and `(pu|qv)` from the problem's integral route, contracted
    /// with `Γ` into `W` (module documentation), and the products formed as
    /// `products` says.
    pub fn orbital_hessian_with(
        &self,
        c: &Array2<f64>,
        pt: &CasScfPoint,
        products: HessianProducts,
    ) -> Result<OrbitalHessian, FerricError> {
        self.check_orbitals(c)?;
        let (nc, na, nmo) = (self.n_core, self.n_active, self.nmo);
        let (na2, na3) = (na * na, na * na * na);
        let rdms = pt.rdms();
        let (ppaa, papa, occ_kernel) = match products {
            HessianProducts::AoJk => {
                let (ppaa, papa) =
                    self.ints
                        .ppaa_papa(self.prep, c.view(), c.slice(s![.., nc..nc + na]))?;
                (ppaa, papa, None)
            }
            HessianProducts::OccupiedKernel => {
                let (ppoo, popo) =
                    self.ints
                        .ppaa_papa(self.prep, c.view(), c.slice(s![.., ..nc + na]))?;
                let kernel = occupied_kernel(&ppoo, &popo, nmo, nc + na);
                let (ppaa, papa) = active_blocks(&ppoo, &popo, nmo, nc, na);
                (ppaa, papa, Some(kernel))
            }
        };
        let dm2 = rdms.dm2.as_standard_layout().into_owned();
        let g = dm2
            .as_slice()
            .ok_or_else(|| FerricError::General("CASSCF Hessian: dm2 layout".to_string()))?;
        let gam = |t: usize, u: usize, v: usize, x: usize| g[((t * na + u) * na + v) * na + x];
        // two[p, t] = Σ_uvw (pu|vw) Γ_tuvw from the active rows u of (pq|vw).
        let mut two = Array2::<f64>::zeros((nmo, na));
        let mut paaa = Vec::with_capacity(nmo * na3);
        for p in 0..nmo {
            let row = &ppaa[(p * nmo + nc) * na2..(p * nmo + nc + na) * na2];
            paaa.extend_from_slice(row);
            for t in 0..na {
                two[(p, t)] = row
                    .iter()
                    .zip(&g[t * na3..(t + 1) * na3])
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
        // A[(p,r),(t,s)] = Σ_vw (pr|vw) Γ_tsvw.
        let g1 = Array2::from_shape_fn((na2, na2), |(vw, ts)| {
            gam(ts / na, ts % na, vw / na, vw % na)
        });
        let pp = ArrayView2::from_shape((nmo * nmo, na2), &ppaa)
            .map_err(|e| FerricError::General(format!("CASSCF (pq|vw) shape: {e}")))?;
        let a = pp.dot(&g1);
        drop(ppaa);
        // B[(p,r),(t,s)] = Σ_uw (pu|rw) (Γ_tusw + Γ_tuws), (pu|rw) permuted to [(p,r),(u,w)].
        let mut pr = Array2::<f64>::zeros((nmo * nmo, na2));
        for p in 0..nmo {
            for u in 0..na {
                for r in 0..nmo {
                    let src =
                        &papa[((p * na + u) * nmo + r) * na..((p * na + u) * nmo + r + 1) * na];
                    pr.slice_mut(s![p * nmo + r, u * na..(u + 1) * na])
                        .assign(&ndarray::ArrayView1::from(src));
                }
            }
        }
        drop(papa);
        let m = Array2::from_shape_fn((na2, na2), |(uw, ts)| {
            let (u, x, t, sx) = (uw / na, uw % na, ts / na, ts % na);
            gam(t, u, sx, x) + gam(t, u, x, sx)
        });
        let b = pr.dot(&m);
        drop(pr);
        let sum = &a + &b;
        drop((a, b));
        // W[(p,t),(r,s)] = sum[(p,r),(t,s)].
        let mut w = Array2::<f64>::zeros((nmo * na, nmo * na));
        for p in 0..nmo {
            for r in 0..nmo {
                let row = sum.row(p * nmo + r);
                for t in 0..na {
                    for sx in 0..na {
                        w[(p * na + t, r * na + sx)] = row[t * na + sx];
                    }
                }
            }
        }
        Ok(OrbitalHessian {
            c: c.clone(),
            fi: pt.f_inactive.clone(),
            fa: pt.f_active.clone(),
            gmat: pt.gradient_matrix.clone(),
            gamma: rdms.dm1.clone(),
            two,
            paaa,
            w,
            occ_kernel,
        })
    }

    /// Antisymmetric `X` from a vector over [`Self::pairs`].
    pub(crate) fn unpack(&self, x: &[f64]) -> Result<Array2<f64>, FerricError> {
        if x.len() != self.pairs.len() {
            return Err(FerricError::General(format!(
                "CASSCF: rotation vector has {} entries, expected {}",
                x.len(),
                self.pairs.len()
            )));
        }
        let mut k = Array2::<f64>::zeros((self.nmo, self.nmo));
        for (&(p, q), &v) in self.pairs.iter().zip(x) {
            k[(p, q)] = v;
            k[(q, p)] = -v;
        }
        Ok(k)
    }

    pub(crate) fn pack(&self, m: &Array2<f64>) -> Vec<f64> {
        self.pairs.iter().map(|&(p, q)| m[(p, q)]).collect()
    }

    /// `2 (dF − dFᵀ)` and `X` for the direction `x` (module documentation).
    fn frame_derivative(
        &self,
        h: &OrbitalHessian,
        x: &[f64],
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let (dg, xm, _) = self.frame_derivative_parts(h, x)?;
        Ok((dg, xm))
    }

    /// [`Self::frame_derivative`] with `dF^I` over the occupied columns
    /// (`nmo x n_occ`).
    pub(crate) fn frame_derivative_parts(
        &self,
        h: &OrbitalHessian,
        x: &[f64],
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), FerricError> {
        let (nc, na, nmo) = (self.n_core, self.n_active, self.nmo);
        let nocc = nc + na;
        let xm = self.unpack(x)?;
        let (gi, ga) = match &h.occ_kernel {
            Some(l) => occupied_response(l, &xm, &h.gamma, nc, na)?,
            None => self.ao_response(h, &xm)?,
        };
        let x_occ = xm.slice(s![.., ..nocc]);
        let dfi = h.fi.dot(&x_occ) - xm.dot(&h.fi.slice(s![.., ..nocc])) + gi;
        let dfa = h.fa.dot(&xm.slice(s![.., ..nc])) - xm.dot(&h.fa.slice(s![.., ..nc])) + ga;
        let mut df = Array2::<f64>::zeros((nmo, nmo));
        df.slice_mut(s![.., ..nc])
            .assign(&(2.0 * (&dfi.slice(s![.., ..nc]) + &dfa)));
        // Active columns: dF^I γ, the X_rp row rotation of Σ (pu|vw) Γ, and W X.
        let xa = xm.slice(s![.., nc..nocc]).as_standard_layout().into_owned();
        let xa_vec = Array1::from_iter(xa.iter().copied());
        let wx =
            h.w.dot(&xa_vec)
                .into_shape_with_order((nmo, na))
                .map_err(|e| FerricError::General(format!("CASSCF W X shape: {e}")))?;
        let act = dfi.slice(s![.., nc..nocc]).dot(&h.gamma) + xm.t().dot(&h.two) + wx;
        df.slice_mut(s![.., nc..nocc]).assign(&act);
        Ok((2.0 * (&df - &df.t()), xm, dfi))
    }

    /// `Cᵀ (J − ½K)[dD^c] C_occ` and `Cᵀ (J − ½K)[dD^a] C_c` by one AO J/K
    /// pass over the two densities ([`HessianProducts::AoJk`]).
    fn ao_response(
        &self,
        h: &OrbitalHessian,
        xm: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let (nc, nocc) = (self.n_core, self.n_core + self.n_active);
        let c = &h.c;
        let cx = c.dot(&xm.slice(s![.., ..nocc]));
        let cc = c.slice(s![.., ..nc]);
        let ca = c.slice(s![.., nc..nocc]);
        let half = 2.0 * cx.slice(s![.., ..nc]).dot(&cc.t());
        let ddc = &half + &half.t();
        let half = cx.slice(s![.., nc..nocc]).dot(&h.gamma).dot(&ca.t());
        let dda = &half + &half.t();
        let jk = self.ints.jk_multi(self.prep, &[ddc, dda])?;
        let gi_ao = &jk[0].0 - &(0.5 * &jk[0].1);
        let ga_ao = &jk[1].0 - &(0.5 * &jk[1].1);
        let gi = c.t().dot(&gi_ao).dot(&c.slice(s![.., ..nocc]));
        let ga = c.t().dot(&ga_ao).dot(&cc);
        Ok((gi, ga))
    }

    /// Frame derivative of [`Self::gradient_at_fixed_rdms`] along `x`:
    /// `d/dε g(C exp(ε X))` at `ε = 0`, each gradient in its own orbitals.
    pub fn hessian_vector_frame(
        &self,
        h: &OrbitalHessian,
        x: &[f64],
    ) -> Result<Vec<f64>, FerricError> {
        let (dg, _) = self.frame_derivative(h, x)?;
        Ok(self.pack(&dg))
    }

    /// `H x`, the symmetric fixed-RDM orbital Hessian of `E(C exp(K))` at
    /// `K = 0` times `x`.
    pub fn hessian_vector(&self, h: &OrbitalHessian, x: &[f64]) -> Result<Vec<f64>, FerricError> {
        let (dg, xm) = self.frame_derivative(h, x)?;
        let comm = h.gmat.dot(&xm) - xm.dot(&h.gmat);
        Ok(self.pack(&(dg - 0.5 * comm)))
    }
}

/// `L[(p,j),(r,k)] = 2 (pj|rk) − ½ (pr|jk) − ½ (pk|rj)` from `(pq|jk)` flat
/// `[((p n + q) n_o + j) n_o + k]` and `(pj|qk)` flat
/// `[((p n_o + j) n + q) n_o + k]` (module documentation).
fn occupied_kernel(ppoo: &[f64], popo: &[f64], nmo: usize, nocc: usize) -> Array2<f64> {
    let m = nmo * nocc;
    let mut l = Array2::<f64>::zeros((m, m));
    let popo_at =
        |p: usize, j: usize, r: usize, k: usize| popo[((p * nocc + j) * nmo + r) * nocc + k];
    let ppoo_at =
        |p: usize, r: usize, j: usize, k: usize| ppoo[((p * nmo + r) * nocc + j) * nocc + k];
    for p in 0..nmo {
        for j in 0..nocc {
            let mut row = l.row_mut(p * nocc + j);
            for r in 0..nmo {
                for k in 0..nocc {
                    row[r * nocc + k] = 2.0 * popo_at(p, j, r, k)
                        - 0.5 * ppoo_at(p, r, j, k)
                        - 0.5 * popo_at(p, k, r, j);
                }
            }
        }
    }
    l
}

/// The active blocks `(pq|vw)` and `(pu|qv)` of the occupied integrals of
/// [`occupied_kernel`], in the layouts of
/// `IntegralBackend::ppaa_papa`.
fn active_blocks(
    ppoo: &[f64],
    popo: &[f64],
    nmo: usize,
    nc: usize,
    na: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nocc = nc + na;
    let mut ppaa = Vec::with_capacity(nmo * nmo * na * na);
    for pq in 0..nmo * nmo {
        for v in 0..na {
            let base = (pq * nocc + nc + v) * nocc + nc;
            ppaa.extend_from_slice(&ppoo[base..base + na]);
        }
    }
    let mut papa = Vec::with_capacity(nmo * nmo * na * na);
    for p in 0..nmo {
        for u in 0..na {
            for q in 0..nmo {
                let base = ((p * nocc + nc + u) * nmo + q) * nocc + nc;
                papa.extend_from_slice(&popo[base..base + na]);
            }
        }
    }
    (ppaa, papa)
}

/// `Cᵀ (J − ½K)[dD^c] C_occ` (`nmo x n_occ`) and `Cᵀ (J − ½K)[dD^a] C_c`
/// (`nmo x n_c`) as `L vec(Y)` (module documentation): one GEMM of `L` with
/// the two columns `vec(Y^c)`, `vec(Y^a)`, at the serial-section BLAS
/// thread count.
pub(crate) fn occupied_response(
    l: &Array2<f64>,
    xm: &Array2<f64>,
    gamma: &Array2<f64>,
    nc: usize,
    na: usize,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let nmo = xm.nrows();
    let nocc = nc + na;
    let mut y = Array2::<f64>::zeros((nmo * nocc, 2));
    let ya = xm.slice(s![.., nc..nocc]).dot(gamma);
    for r in 0..nmo {
        for k in 0..nc {
            y[(r * nocc + k, 0)] = 2.0 * xm[(r, k)];
        }
        for t in 0..na {
            y[(r * nocc + nc + t, 1)] = ya[(r, t)];
        }
    }
    let ly = with_blas_threads(opt_in_blas_threads(), || l.dot(&y));
    let shape = |col: usize| {
        ly.column(col)
            .to_owned()
            .into_shape_with_order((nmo, nocc))
            .map_err(|e| FerricError::General(format!("CASSCF L vec(Y) shape: {e}")))
    };
    let gi = shape(0)?;
    let ga = shape(1)?.slice(s![.., ..nc]).to_owned();
    Ok((gi, ga))
}

/// Stretch of the next AH step. The fixed-RDM Hessian lacks the CI
/// response, which lowers the curvature: along the accepted step `s` the
/// energy change `ΔE = g·s + ½ s·Hs / β` gives the ratio
/// `β = s·Hs / (2 (ΔE − g·s))` of the model to the actual curvature, and the
/// next step is `β` times the AH step, clamped to `[1, AH_MAX_STRETCH]` (the
/// upper end when the actual curvature along `s` is not positive).
fn curvature_stretch(de: f64, gs: f64, shs: f64) -> f64 {
    let quad = de - gs;
    if quad > 0.0 && shs > 0.0 {
        (shs / (2.0 * quad)).clamp(1.0, AH_MAX_STRETCH)
    } else if shs > 0.0 {
        AH_MAX_STRETCH
    } else {
        1.0
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Lowest eigenpair of the augmented Hessian `[[0, gᵀ], [g, H]]` by Davidson
/// with the diagonal preconditioner `pre − λ` (`pre > 0`): returns the step
/// `s = v/v_0`, which solves `(H − λ) s = −g`, stopped when that residual is
/// at most `tol`. A vanishing leading component (a Hessian mode orthogonal
/// to `g` below the gradient-coupled root) gives the preconditioned steepest
/// descent step `−g/pre` instead, flagged unconverged.
pub fn ah_davidson(
    mut hop: impl FnMut(&[f64]) -> Result<Vec<f64>, FerricError>,
    g: &[f64],
    pre: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<AhStep, FerricError> {
    let n = g.len();
    let mut vs: Vec<Vec<f64>> = Vec::new();
    let mut hvs: Vec<Vec<f64>> = Vec::new();
    let mut v: Vec<f64> = g.iter().zip(pre).map(|(a, p)| -a / p).collect();
    let mut last: Option<AhStep> = None;
    for it in 1..=max_iter.max(1) {
        for _ in 0..2 {
            for u in &vs {
                let o = dot(u, &v);
                v.iter_mut().zip(u).for_each(|(a, b)| *a -= o * b);
            }
        }
        let nv = norm(&v);
        if !(nv > 1e-14) {
            break;
        }
        v.iter_mut().for_each(|a| *a /= nv);
        hvs.push(hop(&v)?);
        vs.push(std::mem::take(&mut v));
        let m = vs.len();
        let mut aug = Array2::<f64>::zeros((m + 1, m + 1));
        for i in 0..m {
            aug[(0, i + 1)] = dot(&vs[i], g);
            aug[(i + 1, 0)] = aug[(0, i + 1)];
            for j in 0..=i {
                let hij = 0.5 * (dot(&vs[i], &hvs[j]) + dot(&vs[j], &hvs[i]));
                aug[(i + 1, j + 1)] = hij;
                aug[(j + 1, i + 1)] = hij;
            }
        }
        let (w, z) = aug
            .eigh(UPLO::Lower)
            .map_err(|e| FerricError::General(format!("CASSCF AH: subspace eigh: {e}")))?;
        let lam = w[0];
        let z0 = z[(0, 0)];
        if z0.abs() < 1e-8 {
            let step: Vec<f64> = g.iter().zip(pre).map(|(a, p)| -a / p).collect();
            let h_step = hop(&step)?;
            return Ok(AhStep {
                step,
                h_step,
                products: it + 1,
                residual: f64::INFINITY,
                converged: false,
            });
        }
        let mut step = vec![0.0; n];
        let mut h_step = vec![0.0; n];
        for i in 0..m {
            let y = z[(i + 1, 0)] / z0;
            step.iter_mut().zip(&vs[i]).for_each(|(a, b)| *a += y * b);
            h_step
                .iter_mut()
                .zip(&hvs[i])
                .for_each(|(a, b)| *a += y * b);
        }
        let r: Vec<f64> = (0..n).map(|k| h_step[k] - lam * step[k] + g[k]).collect();
        let rn = norm(&r);
        let done = rn <= tol;
        last = Some(AhStep {
            step,
            h_step,
            products: it,
            residual: rn,
            converged: done,
        });
        if done {
            break;
        }
        v = r
            .iter()
            .zip(pre)
            .map(|(a, p)| {
                let d = p - lam;
                -a / if d.abs() < 1e-8 { 1e-8 } else { d }
            })
            .collect();
    }
    // A vanishing first vector (g = 0) gives the zero step.
    Ok(last.unwrap_or(AhStep {
        step: vec![0.0; n],
        h_step: vec![0.0; n],
        products: 0,
        residual: 0.0,
        converged: true,
    }))
}

/// The AH macro-iteration loop from `c0` (module documentation). Returns
/// the final point, its orbitals, the macro iterations (evaluations) taken
/// and whether both convergence criteria were met.
pub(crate) fn optimize_ah(
    problem: &CasScfProblem,
    c0: Array2<f64>,
    config: &CasScfConfig,
    acc: &mut PhaseAccumulator,
) -> Result<(CasScfPoint, Array2<f64>, usize, bool), FerricError> {
    let npair = problem.pairs().len();
    let mut c = c0;
    let mut pt = problem.evaluate_timed(&c, acc)?;
    let mut it = 1usize;
    let mut e_old: Option<f64> = None;
    let mut radius = AH_INITIAL_RADIUS;
    let mut alpha = 1.0f64;
    loop {
        let gn = norm(&pt.gradient);
        let done = npair == 0
            || e_old.is_some_and(|eo| {
                (pt.e_total - eo).abs() < config.energy_tol && gn < config.grad_tol
            });
        if done || it >= config.max_iter {
            return Ok((pt, c, it, done));
        }
        let hess = acc.time("orbital Hessian integrals", || {
            problem.orbital_hessian(&c, &pt)
        })?;
        let pre: Vec<f64> = problem
            .approx_hessian_diag(&pt)
            .into_iter()
            .map(|h| h.abs().max(config.hdiag_floor))
            .collect();
        let ah = ah_davidson(
            |v| {
                acc.time("Hessian-vector product", || {
                    problem.hessian_vector(&hess, v)
                })
            },
            &pt.gradient,
            &pre,
            config.ah_forcing.eta(gn) * gn,
            AH_MAX_DAVIDSON,
        )?;
        drop(hess);
        let (mut s, mut hs) = (ah.step, ah.h_step);
        s.iter_mut().for_each(|x| *x *= alpha);
        hs.iter_mut().for_each(|x| *x *= alpha);
        let sn = norm(&s);
        let smax = s.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let f = (radius / sn).min(config.max_step / smax).min(1.0);
        if f < 1.0 && f.is_finite() {
            s.iter_mut().for_each(|x| *x *= f);
            hs.iter_mut().for_each(|x| *x *= f);
        }
        let mut sn = norm(&s);
        let (c_new, pt_new, pred) = loop {
            let pred = dot(&pt.gradient, &s) + 0.5 * dot(&s, &hs) / alpha;
            let c_try = acc.time("orbital rotation", || problem.rotate(&c, &s))?;
            let pt_try = problem.evaluate_timed(&c_try, acc)?;
            it += 1;
            if pt_try.e_total - pt.e_total > AH_REJECT_RISE && it < config.max_iter {
                s.iter_mut().for_each(|x| *x *= 0.5);
                hs.iter_mut().for_each(|x| *x *= 0.5);
                sn *= 0.5;
                radius = sn;
                continue;
            }
            break (c_try, pt_try, pred);
        };
        let ratio = if pred < 0.0 {
            (pt_new.e_total - pt.e_total) / pred
        } else {
            0.0
        };
        alpha = curvature_stretch(
            pt_new.e_total - pt.e_total,
            dot(&pt.gradient, &s),
            dot(&s, &hs),
        );
        if ratio < 0.25 {
            radius = (0.5 * sn).max(1e-3);
        } else if ratio > 0.75 && sn >= 0.99 * radius {
            radius = (2.0 * radius).min(AH_MAX_RADIUS);
        }
        e_old = Some(pt.e_total);
        c = c_new;
        pt = pt_new;
    }
}
