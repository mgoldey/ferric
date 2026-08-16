//! Amplitude-threshold LinLCCD in the localized basis — the ladder member
//! of the single-threshold family (MP2: `ferric_mp2::lmp2_amplitude`;
//! rings/dRPA: `ferric_mp2::drpa_amplitude`; ladders: here).
//!
//! Built AFTER the proof notebook
//! `wiki/notebooks/13-amplitude-threshold-linlccd.ipynb`, which establishes
//! by derivation-by-verification (spin-orbital vs candidate spatial residual
//! on random symmetric inputs, 1e-14):
//!
//! ```text
//! R_iajb = (ia|jb) + F(T)_iajb + Σ_kl (ik|jl) T_kalb   [hh ladder]
//!                              + Σ_cd (ac|bd) T_icjd   [pp ladder]
//! E      = Σ (ia|jb) (2 T_iajb − T_ibja)
//! ```
//!
//! and that the system is linear with a symmetric PD operator (Fock part PD
//! for gapped systems; both ladder blocks are RI Gram matrices, hence PSD),
//! so the masked solve carries MP2-style Hylleraas protection — solver error
//! enters quadratically, only dropped integrals enter linearly.
//!
//! The canonical anchor is [`super::linlccd::linlccd`] — ferric's
//! spin-orbital einsum implementation (Carter-Fenk JPCA 2025), an
//! independent formulation sharing only the RI integrals. `LadderVariant`
//! is reused directly so each tier (DriversOnly ≡ RI-MP2 / Hh / Full)
//! anchors against its canonical counterpart.
//!
//! V1 SCOPE: dense masked CG over the (no·nv)² compound space (prototype
//! parity with the MP2/dRPA rigs); the OOOO/VVVV ladder blocks are kept
//! DENSE — thresholding them is future work, so no cost or scaling claim
//! attaches to this module. Closed-shell RHF, library-only. `Hh` never
//! forms the VVVV block, matching the canonical implementation's memory
//! shape.

use ndarray::Array2;

use ferric_core::basis::BasisSet;
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;

use crate::linlccd::LadderVariant;
use ferric_mp2::lmp2_amplitude::{
    assemble_localized, build_vvhv, check_vvhv, AmplitudeLmp2Config, VvHv,
};
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_vv};
use ferric_mp2::rimp2::metric_inverse_sqrt;

#[derive(Debug, Clone)]
pub struct AmplitudeLinLccdConfig {
    /// Threshold ε on |(ia|jb)| (Eq-8 symmetric test, swap-closed); 0 keeps
    /// everything (the exactness-anchor limit).
    pub eps: f64,
    pub frozen_core: usize,
    pub cg_rtol: f64,
    pub cg_max_iter: usize,
    pub eri3_budget_bytes: Option<usize>,
}

impl Default for AmplitudeLinLccdConfig {
    fn default() -> Self {
        Self { eps: 1e-4, frozen_core: 0, cg_rtol: 1e-11, cg_max_iter: 600, eri3_budget_bytes: None }
    }
}

#[derive(Debug)]
pub struct AmplitudeLinLccdResult {
    pub e_corr: f64,
    pub e_total: f64,
    pub keep_fraction: f64,
    pub cg_iterations: usize,
    pub cg_relres: f64,
    pub cg_converged: bool,
}

/// Amplitude-threshold LinLCCD with the VV-HV space built internally.
pub fn amplitude_linlccd(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLinLccdConfig,
    variant: LadderVariant,
) -> Result<AmplitudeLinLccdResult, FerricError> {
    let vvhv = build_vvhv(mol, obs, obs_bs, rhf)?;
    let nocc_total = (mol.nelec() as usize) / 2;
    let (dev_orth, dev_span) = check_vvhv(obs, rhf, nocc_total, &vvhv.c_vloc);
    if dev_orth > 1e-8 || dev_span > 1e-8 {
        return Err(FerricError::General(format!(
            "linlccd_amplitude: VV-HV construction check failed (orth {dev_orth:.2e}, span {dev_span:.2e})"
        )));
    }
    amplitude_linlccd_with_virtuals(mol, obs, dfbs, op, rhf, cfg, variant, &vvhv)
}

/// Mutation-test entry point.
#[allow(clippy::too_many_arguments)]
pub fn amplitude_linlccd_with_virtuals(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLinLccdConfig,
    variant: LadderVariant,
    vvhv: &VvHv,
) -> Result<AmplitudeLinLccdResult, FerricError> {
    let lcfg = AmplitudeLmp2Config {
        eps: cfg.eps,
        frozen_core: cfg.frozen_core,
        eri3_budget_bytes: cfg.eri3_budget_bytes,
        ..Default::default()
    };
    let lp = assemble_localized(mol, obs, dfbs, op, rhf, &lcfg, vvhv)?;
    let (no, nv) = (lp.no, lp.nv);
    let n = no * nv;
    let j2 = &lp.j_dense; // (ia|jb) over (i·nv+a, j·nv+b)

    // ladder blocks in the SAME localized basis (whitened RI Gram products)
    let (oo_mat, vv_mat) = {
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
        let vis = metric_inverse_sqrt(&v2c, op)?;
        let naux = eri3_ao.shape()[0];
        // hh: (ik|jl) as a (no², no²) matrix over row (i·no+j), col (k·no+l)
        let boo = transform_3center_oo(&eri3_ao, &lp.c_locc); // (naux, no, no)
        let boo = boo
            .into_shape_with_order((naux, no * no))
            .map_err(|e| FerricError::General(format!("linlccd_amplitude oo reshape: {e}")))?;
        let boo_t = vis.dot(&boo); // whitened, columns indexed (i,k)
        // (ik|jl) = Σ_P boo_t[P,(i,k)] boo_t[P,(j,l)]; we need row (i,j), col (k,l):
        let gram_oo = boo_t.t().dot(&boo_t); // rows (i,k), cols (j,l)
        let mut oo_mat = Array2::<f64>::zeros((no * no, no * no));
        for i in 0..no {
            for j in 0..no {
                for k in 0..no {
                    for l in 0..no {
                        oo_mat[(i * no + j, k * no + l)] = gram_oo[(i * no + k, j * no + l)];
                    }
                }
            }
        }
        let vv_mat = if variant.needs_vvvv_pub() {
            let bvv = transform_3center_vv(&eri3_ao, &vvhv.c_vloc); // (naux, nv, nv)
            let bvv = bvv
                .into_shape_with_order((naux, nv * nv))
                .map_err(|e| FerricError::General(format!("linlccd_amplitude vv reshape: {e}")))?;
            let bvv_t = vis.dot(&bvv);
            let gram_vv = bvv_t.t().dot(&bvv_t); // rows (a,c), cols (b,d)
            let mut m = Array2::<f64>::zeros((nv * nv, nv * nv));
            for a in 0..nv {
                for b in 0..nv {
                    for c in 0..nv {
                        for d in 0..nv {
                            // (ac|bd) at row (a·nv+b), col (c·nv+d)
                            m[(a * nv + b, c * nv + d)] = gram_vv[(a * nv + c, b * nv + d)];
                        }
                    }
                }
            }
            Some(m)
        } else {
            None
        };
        (oo_mat, vv_mat)
    };

    // Eq-8 swap-closed mask on (ia|jb)
    let mask: Vec<bool> = {
        let mut m = vec![false; n * n];
        for i in 0..no {
            for a in 0..nv {
                for j in 0..no {
                    for b in 0..nv {
                        let jd = j2[(i * nv + a, j * nv + b)].abs();
                        let kd = j2[(i * nv + b, j * nv + a)].abs();
                        if cfg.eps == 0.0 || jd > cfg.eps || kd > cfg.eps {
                            m[(i * nv + a) * n + (j * nv + b)] = true;
                        }
                    }
                }
            }
        }
        m
    };
    let kept = mask.iter().filter(|&&x| x).count();
    let apply_mask = |m: &mut Array2<f64>| {
        for (v, &keep) in m.iter_mut().zip(&mask) {
            if !keep {
                *v = 0.0;
            }
        }
    };

    // A(t) = F(t) + hh(t) [+ pp(t)], pattern-projected. All terms use the
    // (i·no+j, a·nv+b) "pair-matrix" layout for the ladder GEMMs.
    let to_pair = |t: &Array2<f64>| -> Array2<f64> {
        let mut p = Array2::<f64>::zeros((no * no, nv * nv));
        for i in 0..no {
            for a in 0..nv {
                for j in 0..no {
                    for b in 0..nv {
                        p[(i * no + j, a * nv + b)] = t[(i * nv + a, j * nv + b)];
                    }
                }
            }
        }
        p
    };
    let from_pair = |p: &Array2<f64>| -> Array2<f64> {
        let mut t = Array2::<f64>::zeros((n, n));
        for i in 0..no {
            for a in 0..nv {
                for j in 0..no {
                    for b in 0..nv {
                        t[(i * nv + a, j * nv + b)] = p[(i * no + j, a * nv + b)];
                    }
                }
            }
        }
        t
    };
    let aop = |t: &Array2<f64>| -> Array2<f64> {
        // Fock superoperator (same convention the notebook verified)
        let mut r = Array2::<f64>::zeros((n, n));
        for i in 0..no {
            for j in 0..no {
                let mut blk = Array2::<f64>::zeros((nv, nv));
                for a in 0..nv {
                    for b in 0..nv {
                        blk[(a, b)] = t[(i * nv + a, j * nv + b)];
                    }
                }
                let mut acc = lp.f_vv.dot(&blk);
                acc += &blk.dot(&lp.f_vv);
                for k in 0..no {
                    let fik = lp.f_oo[(i, k)];
                    let fkj = lp.f_oo[(k, j)];
                    for a in 0..nv {
                        for b in 0..nv {
                            if fik != 0.0 {
                                acc[(a, b)] -= fik * t[(k * nv + a, j * nv + b)];
                            }
                            if fkj != 0.0 {
                                acc[(a, b)] -= fkj * t[(i * nv + a, k * nv + b)];
                            }
                        }
                    }
                }
                for a in 0..nv {
                    for b in 0..nv {
                        r[(i * nv + a, j * nv + b)] = acc[(a, b)];
                    }
                }
            }
        }
        // ladders via pair-matrix GEMMs (gated per variant: DriversOnly
        // applies NEITHER — it must reproduce RI-MP2 exactly)
        if !matches!(variant, LadderVariant::DriversOnly) {
            let tp = to_pair(t);
            let mut lad = oo_mat.dot(&tp); // hh: Σ_kl (ik|jl) T_kalb
            if let Some(vv) = &vv_mat {
                lad += &tp.dot(&vv.t()); // pp: Σ_cd (ac|bd) T_icjd
            }
            r += &from_pair(&lad);
        }
        apply_mask(&mut r);
        r
    };

    // masked preconditioned CG on A t = −J
    let mut d2 = Array2::<f64>::zeros((n, n));
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    d2[(i * nv + a, j * nv + b)] = lp.f_vv[(a, a)] + lp.f_vv[(b, b)]
                        - lp.f_oo[(i, i)]
                        - lp.f_oo[(j, j)];
                }
            }
        }
    }
    if d2.iter().any(|&x| x <= 0.0) {
        return Err(FerricError::General(
            "linlccd_amplitude: non-positive denominator (not a gapped system?)".into(),
        ));
    }
    let mut rhs = j2.mapv(|x| -x);
    apply_mask(&mut rhs);
    let bnorm = rhs.iter().map(|x| x * x).sum::<f64>().sqrt();
    let dot = |x: &Array2<f64>, y: &Array2<f64>| -> f64 { (x * y).sum() };
    let mut t = Array2::<f64>::zeros((n, n));
    let mut r = rhs.clone();
    let mut z = &r / &d2;
    apply_mask(&mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut it = 0;
    let mut relres = 1.0;
    let mut converged = bnorm == 0.0;
    while it < cfg.cg_max_iter && !converged {
        it += 1;
        let ap = aop(&p);
        let alpha = rz / dot(&p, &ap);
        t.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        relres = dot(&r, &r).sqrt() / bnorm;
        if relres < cfg.cg_rtol {
            converged = true;
            break;
        }
        z = &r / &d2;
        apply_mask(&mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        p = &z + &(&p * beta);
        rz = rz_new;
    }
    if !converged {
        return Err(FerricError::General(format!(
            "linlccd_amplitude: CG failed to converge (relres {relres:.2e} after {it} iters)"
        )));
    }

    // E = Σ (ia|jb)(2 T_iajb − T_ibja) over the pattern (proof notebook §1)
    let mut e_corr = 0.0;
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    if !mask[(i * nv + a) * n + (j * nv + b)] {
                        continue;
                    }
                    let jv = j2[(i * nv + a, j * nv + b)];
                    e_corr += jv
                        * (2.0 * t[(i * nv + a, j * nv + b)] - t[(i * nv + b, j * nv + a)]);
                }
            }
        }
    }

    Ok(AmplitudeLinLccdResult {
        e_corr,
        e_total: rhf.energy + e_corr,
        keep_fraction: kept as f64 / (n * n) as f64,
        cg_iterations: it,
        cg_relres: relres,
        cg_converged: converged,
    })
}

impl LadderVariant {
    /// Public mirror of the crate-private `needs_vvvv` used by the canonical
    /// implementation, so the amplitude-threshold port keeps the same
    /// memory shape per variant.
    pub fn needs_vvvv_pub(self) -> bool {
        matches!(self, LadderVariant::Full)
    }
}
