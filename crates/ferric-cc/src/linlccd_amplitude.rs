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
    assemble_basis, assemble_ragged_direct, build_vvhv, check_vvhv, AmplitudeLmp2Config, VvHv,
};
use ferric_mp2::ragged::{apply_pattern, gather_into, matvec_indexed, solve_ragged_with};
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
    let lb = assemble_basis(mol, obs, dfbs, op, rhf, &lcfg, vvhv)?;
    let (no, nv) = (lb.no, lb.nv);
    // (ia|jb) directly onto ragged blocks — the dense (no·nv)² tensor is
    // never formed (the lmp2 direct-assembly path, scale 1.0)
    let (rg, _gated) = assemble_ragged_direct(mol, dfbs, op, &lb, cfg.eps, 1.0, None, None)?;

    // hh ladder coefficients (ik|jl): the OOOO block is no⁴ — tiny — via
    // the same whitened RI Gram as the canonical implementation
    let vis = metric_inverse_sqrt(&lb.v2c, op)?;
    let naux = lb.b_flat.nrows();
    let oo_g = {
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
        let boo = transform_3center_oo(&eri3_ao, &lb.c_locc)
            .into_shape_with_order((naux, no * no))
            .map_err(|e| FerricError::General(format!("linlccd_amplitude oo reshape: {e}")))?;
        let boo_t = vis.dot(&boo);
        boo_t.t().dot(&boo_t) // rows (i,k), cols (j,l): (ik|jl)
    };
    // pp ladder: whitened B over VV pairs, gathered per pair on demand —
    // the (nv²)² VVVV tensor is never formed
    let bvv_t: Option<Array2<f64>> = if variant.needs_vvvv_pub() {
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
        let bvv = transform_3center_vv(&eri3_ao, &vvhv.c_vloc)
            .into_shape_with_order((naux, nv * nv))
            .map_err(|e| FerricError::General(format!("linlccd_amplitude vv reshape: {e}")))?;
        Some(vis.dot(&bvv))
    } else {
        None
    };

    let apply_hh = !matches!(variant, LadderVariant::DriversOnly);
    let matvec = |t: &[Array2<f64>], flops: &mut u64| -> Vec<Array2<f64>> {
        let mut r = matvec_indexed(&rg, &lb.f_oo, t, flops); // F(t), pattern-projected
        if apply_hh {
            // hh: out_ij += Σ_(k,l) (ik|jl) · gather(T_kl)
            for (p_out, pb_out) in rg.pairs.iter().enumerate() {
                for (p_src, pb_src) in rg.pairs.iter().enumerate() {
                    let coeff = oo_g[(pb_out.i * no + pb_src.i, pb_out.j * no + pb_src.j)];
                    if coeff == 0.0 {
                        continue;
                    }
                    gather_into(&mut r[p_out], pb_src, pb_out, coeff, &t[p_src], flops);
                }
            }
        }
        if let Some(bvv) = &bvv_t {
            // pp (block-local): out_ij[a,b] += Σ_cd (ac|bd) T_ij[c,d], with
            // (ac|bd) = Σ_P bvv[P,(a,c)] bvv[P,(b,d)] gathered on the pair's
            // union domains — never a global VVVV
            for (p, pb) in rg.pairs.iter().enumerate() {
                let (da, db) = (&pb.da, &pb.db);
                let (nda, ndb) = (da.len(), db.len());
                // rows (a, c): a ∈ Da (output row), c ∈ Da (contraction)
                let mut ba = Array2::<f64>::zeros((naux, nda * nda));
                for (ra, &a) in da.iter().enumerate() {
                    for (rc, &c) in da.iter().enumerate() {
                        ba.column_mut(ra * nda + rc).assign(&bvv.column(a * nv + c));
                    }
                }
                let mut bb = Array2::<f64>::zeros((naux, ndb * ndb));
                for (cb, &b) in db.iter().enumerate() {
                    for (cd, &d) in db.iter().enumerate() {
                        bb.column_mut(cb * ndb + cd).assign(&bvv.column(b * nv + d));
                    }
                }
                let m = ba.t().dot(&bb); // rows (a,c), cols (b,d)
                *flops += (nda * nda * ndb * ndb * naux) as u64;
                let tp = &t[p];
                for ra in 0..nda {
                    for cb in 0..ndb {
                        let mut acc = 0.0;
                        for rc in 0..nda {
                            for cd in 0..ndb {
                                acc += m[(ra * nda + rc, cb * ndb + cd)] * tp[(rc, cd)];
                            }
                        }
                        r[p][(ra, cb)] += acc;
                    }
                }
            }
        }
        for (p, pb) in rg.pairs.iter().enumerate() {
            apply_pattern(&mut r[p], &pb.pat, pb.db.len());
        }
        r
    };

    let (t, it, relres, converged, _flops) =
        solve_ragged_with(&rg, cfg.cg_rtol, cfg.cg_max_iter, matvec);
    if !converged {
        return Err(FerricError::General(format!(
            "linlccd_amplitude(ragged): CG failed to converge (relres {relres:.2e} after {it} iters)"
        )));
    }

    // E = Σ (ia|jb)(2 T_iajb − T_ibja) over the pattern (proof notebook §1)
    let mut e_corr = 0.0;
    for (p, pb) in rg.pairs.iter().enumerate() {
        let nbb = pb.db.len();
        for (r_, &a) in pb.da.iter().enumerate() {
            for (c_, &b) in pb.db.iter().enumerate() {
                if !pb.pat[r_ * nbb + c_] {
                    continue;
                }
                let jv = pb.j_blk[(r_, c_)];
                let sr = pb.pos_da[b];
                let sc = pb.pos_db[a];
                let t_swap = if sr != usize::MAX && sc != usize::MAX { t[p][(sr, sc)] } else { 0.0 };
                e_corr += jv * (2.0 * t[p][(r_, c_)] - t_swap);
            }
        }
    }

    let n = no * nv;
    let kept: usize = rg.pairs.iter().map(|pb| pb.pat.iter().filter(|&&x| x).count()).sum();
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
