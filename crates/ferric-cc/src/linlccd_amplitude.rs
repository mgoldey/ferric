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
//! MEASURED CORRECTION to the notebook's PSD claim (2026-09-06,
//! proto_linlccd_pp_psd.py / WIKI-APPEND-linlccd-direct.md): in the
//! exchange pairing the ladder supermatrices are Σ_P Y_P ⊗ Y_P with Y_P
//! symmetric — NOT a manifest Gram. The exact-integral operator is PSD by
//! pointwise kernel positivity, but the RI projection breaks that
//! argument, and the RI blocks were MEASURED slightly indefinite for the
//! attenuated operator (erfc ω=1, alkane_4: λ_min = −6.9e-4) and
//! size-decreasing for Coulomb (+0.199 water → +0.045 C8). The OPERATIVE
//! CG license — global and direct paths alike — is that this bounded
//! indefiniteness stays orders below the Fock denominator floor (~2.4 Ha
//! on the measured systems), so the full operator remains SPD by Weyl.
//!
//! The canonical anchor is [`crate::linlccd::linlccd`] — ferric's
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
    assemble_basis, assemble_ragged_direct, build_vvhv, check_vvhv, localized_spaces,
    AmplitudeLmp2Config, VvHv,
};
use ferric_mp2::lmp2_direct::{
    assemble_boo_direct, assemble_pp_fitted_direct, assemble_ragged_direct_local, DirectConfig,
    DirectStats, OoGram, PpFitted,
};
use ferric_mp2::ragged::{apply_pattern, gather_into, matvec_indexed, solve_ragged_with, Ragged};
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_vv};
use ferric_mp2::rimp2::metric_inverse_sqrt;

/// Configuration for amplitude-threshold LinLCCD.
#[derive(Debug, Clone)]
pub struct AmplitudeLinLccdConfig {
    /// Threshold ε on |(ia|jb)| (Eq-8 symmetric test, swap-closed); 0 keeps
    /// everything (the exactness-anchor limit).
    pub eps: f64,
    pub frozen_core: usize,
    pub cg_rtol: f64,
    pub cg_max_iter: usize,
    pub eri3_budget_bytes: Option<usize>,
    /// Integral-free pair gate (see `lmp2_amplitude`); direct path only —
    /// gated pairs are never assembled, and the hh Gram's gated columns
    /// are exactly zero (the Gram/PSD license is unchanged). `None`
    /// (default) keeps every pair (the trivial-anchor limit).
    pub pair_gate_cal: Option<f64>,
}

impl Default for AmplitudeLinLccdConfig {
    fn default() -> Self {
        Self {
            eps: 1e-4,
            frozen_core: 0,
            cg_rtol: 1e-11,
            cg_max_iter: 600,
            eri3_budget_bytes: None,
            pair_gate_cal: None,
        }
    }
}

/// Result of an amplitude-threshold LinLCCD calculation.
#[derive(Debug, Clone)]
#[must_use]
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

    let oo = if matches!(variant, LadderVariant::DriversOnly) {
        None
    } else {
        Some(OoGram::dense(oo_g, no))
    };
    let pp = match &bvv_t {
        Some(bvv) => PpSource::GlobalWhitened { bvv_t: bvv, nv },
        None => PpSource::None,
    };
    let (e_corr, it, relres) = linlccd_masked_solve(&rg, &lb.f_oo, oo.as_ref(), pp, cfg)?;

    let n = no * nv;
    let kept: usize = rg.pairs.iter().map(|pb| pb.pat.iter().filter(|&&x| x).count()).sum();
    Ok(AmplitudeLinLccdResult {
        e_corr,
        e_total: rhf.energy + e_corr,
        keep_fraction: kept as f64 / (n * n) as f64,
        cg_iterations: it,
        cg_relres: relres,
        cg_converged: true,
    })
}

/// Masked ragged CG on the LinLCCD linear system — the UNCHANGED solver
/// shared by the global-B path ([`amplitude_linlccd_with_virtuals`]) and
/// the integral-direct path ([`amplitude_linlccd_direct_with_virtuals`]),
/// extracted verbatim (the `riccati_masked_solve` pattern from the dRPA
/// port): F(t) + optional hh gather (coefficients via [`OoGram`]) +
/// optional pp per-pair domain Grams from a whitened global Bvv, then the
/// Hylleraas energy on the pattern. Non-convergence is a hard error, so
/// `Ok` implies convergence.
/// pp-ladder source for the shared solver.
enum PpSource<'a> {
    /// No pp ladder (DriversOnly / Hh tiers).
    None,
    /// Licensed consistent Gram: per-pair gathers from ONE whitened global
    /// Bvv (naux, nv²) — provably-structured, the original path.
    GlobalWhitened { bvv_t: &'a Array2<f64>, nv: usize },
    /// Integral-direct per-pair domain-FITTED factors — the PSD contract
    /// here is the MEASURED one (see [`PpFitted`]'s doc for the license).
    Fitted(&'a [PpFitted]),
}

/// pp block application shared by both pp sources: given the pair's block
/// `m` (rows (a,c), cols (b,d)), accumulate `out[a,b] += Σ_cd m T[c,d]`.
fn apply_pp_block(
    rp: &mut Array2<f64>,
    m: &Array2<f64>,
    nda: usize,
    ndb: usize,
    tp: &Array2<f64>,
) {
    for ra in 0..nda {
        for cb in 0..ndb {
            let mut acc = 0.0;
            for rc in 0..nda {
                for cd in 0..ndb {
                    acc += m[(ra * nda + rc, cb * ndb + cd)] * tp[(rc, cd)];
                }
            }
            rp[(ra, cb)] += acc;
        }
    }
}

fn linlccd_masked_solve(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    oo: Option<&OoGram>,
    pp: PpSource<'_>,
    cfg: &AmplitudeLinLccdConfig,
) -> Result<(f64, usize, f64), FerricError> {
    let matvec = |t: &[Array2<f64>], flops: &mut u64| -> Vec<Array2<f64>> {
        use rayon::prelude::*;
        // F(t) is already rayon-parallel-over-output and deterministic
        // (matvec_indexed); the hh/pp ladder terms below are added to each
        // r[p] wholly inside ONE task keyed by the OUTPUT pair p, so every
        // write targets a distinct r[p] slot — deterministic under any
        // schedule, and the same per-block arithmetic order as the serial
        // loop (see the "deterministic is not bit-identical" convention:
        // disjoint writes only). Each task returns its own flop tally,
        // summed after the parallel region.
        let mut r = matvec_indexed(rg, f_oo, t, flops); // F(t), pattern-projected
        let ladder: Vec<(Array2<f64>, u64)> = rg
            .pairs
            .par_iter()
            .enumerate()
            .map(|(p, pb)| {
                let mut rp = Array2::<f64>::zeros((pb.da.len(), pb.db.len()));
                let mut fl = 0u64;
                // hh: out_ij += Σ_(k,l) (ik|jl) · gather(T_kl). For a FIXED
                // output (i,j), sum over source pairs (k,l) = (p_src.i,
                // p_src.j) with a nonzero OOOO coefficient. Iterate the
                // source pairs (same set the serial loop scanned) but write
                // only into THIS output's rp.
                if let Some(oo) = oo {
                    for (p_src, pb_src) in rg.pairs.iter().enumerate() {
                        let coeff = oo.coeff(pb.i, pb_src.i, pb.j, pb_src.j);
                        if coeff == 0.0 {
                            continue;
                        }
                        gather_into(&mut rp, pb_src, pb, coeff, &t[p_src], &mut fl);
                    }
                }
                match &pp {
                    PpSource::None => {}
                    PpSource::GlobalWhitened { bvv_t: bvv, nv } => {
                        let naux = bvv.nrows();
                        let (da, db) = (&pb.da, &pb.db);
                        let (nda, ndb) = (da.len(), db.len());
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
                        fl += (nda * nda * ndb * ndb * naux) as u64;
                        apply_pp_block(&mut rp, &m, nda, ndb, &t[p]);
                    }
                    PpSource::Fitted(factors) => {
                        let (nda, ndb) = (pb.da.len(), pb.db.len());
                        let f = &factors[p];
                        let m = f.a_rows.dot(&f.gb); // rows (a,c), cols (b,d)
                        fl += (nda * nda * ndb * ndb * f.gb.nrows()) as u64;
                        apply_pp_block(&mut rp, &m, nda, ndb, &t[p]);
                    }
                }
                (rp, fl)
            })
            .collect();
        for (p, (contrib, fl)) in ladder.into_iter().enumerate() {
            r[p] += &contrib;
            *flops += fl;
            apply_pattern(&mut r[p], &rg.pairs[p].pat, rg.pairs[p].db.len());
        }
        r
    };

    let (t, it, relres, converged, _flops) =
        solve_ragged_with(rg, cfg.cg_rtol, cfg.cg_max_iter, matvec);
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
    Ok((e_corr, it, relres))
}

/// Integral-direct amplitude-threshold LinLCCD — the direct-path sibling
/// of [`amplitude_linlccd`] (the `amplitude_drpa_direct` pattern): the
/// global (naux, no·nv) B, the (naux, nao²) AO `eri3_tensor`, and the N⁵
/// whitening GEMM are never formed. RHS (ia|jb) via per-occupied sparse
/// strips ([`assemble_ragged_direct_local`], scale = 1.0); the hh OOOO
/// block via the occ-occ direct pass ([`assemble_boo_direct`] — full aux
/// rows + GLOBAL whitening, so the hh matvec stays the exact Gram the
/// proof-notebook license requires). `Full` adds the pp ladder as
/// per-pair domain-FITTED factors ([`assemble_pp_fitted_direct`]) — that
/// block's SPD contract is the MEASURED one from the Phase-1 prototype
/// (proto_linlccd_pp_psd.py; GO verdict in WIKI-APPEND-linlccd-direct.md),
/// not a Gram identity: re-run the measurement before extending it to new
/// operators or regimes.
#[allow(clippy::too_many_arguments)]
pub fn amplitude_linlccd_direct(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLinLccdConfig,
    dcfg: &DirectConfig,
    variant: LadderVariant,
) -> Result<(AmplitudeLinLccdResult, DirectStats), FerricError> {
    let vvhv = build_vvhv(mol, obs, obs_bs, rhf)?;
    let nocc_total = (mol.nelec() as usize) / 2;
    let (dev_orth, dev_span) = check_vvhv(obs, rhf, nocc_total, &vvhv.c_vloc);
    if dev_orth > 1e-8 || dev_span > 1e-8 {
        return Err(FerricError::General(format!(
            "linlccd_amplitude(direct): VV-HV construction check failed (orth {dev_orth:.2e}, span {dev_span:.2e})"
        )));
    }
    amplitude_linlccd_direct_with_virtuals(mol, obs, dfbs, op, rhf, cfg, dcfg, variant, &vvhv)
}

/// [`amplitude_linlccd_direct`] with a caller-supplied virtual space (the
/// mutation-test entry point, mirroring the global path).
#[allow(clippy::too_many_arguments)]
pub fn amplitude_linlccd_direct_with_virtuals(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLinLccdConfig,
    dcfg: &DirectConfig,
    variant: LadderVariant,
    vvhv: &VvHv,
) -> Result<(AmplitudeLinLccdResult, DirectStats), FerricError> {
    let spaces = localized_spaces(mol, obs, rhf, cfg.frozen_core, vvhv)?;
    let (rg, _n_gated, dstats) = assemble_ragged_direct_local(
        mol,
        obs,
        dfbs,
        op,
        &spaces,
        cfg.eps,
        1.0,
        cfg.pair_gate_cal,
        dcfg,
    )?;
    let oo = if matches!(variant, LadderVariant::DriversOnly) {
        None
    } else {
        Some(assemble_boo_direct(
            obs,
            dfbs,
            op,
            &spaces,
            cfg.eps,
            cfg.pair_gate_cal,
            dcfg,
        )?)
    };
    // Full tier: per-pair domain-FITTED pp factors — ported on the
    // strength of the Phase-1 PSD measurement (GO verdict recorded in
    // WIKI-APPEND-linlccd-direct.md); see PpFitted's doc for the license.
    let pp_factors: Option<Vec<PpFitted>> = if variant.needs_vvvv_pub() {
        Some(assemble_pp_fitted_direct(mol, obs, dfbs, op, &spaces, &rg, dcfg)?)
    } else {
        None
    };
    let pp = match &pp_factors {
        Some(f) => PpSource::Fitted(f),
        None => PpSource::None,
    };
    let (e_corr, it, relres) = linlccd_masked_solve(&rg, &spaces.f_oo, oo.as_ref(), pp, cfg)?;
    let n = spaces.no * spaces.nv;
    let kept: usize = rg.pairs.iter().map(|pb| pb.pat.iter().filter(|&&x| x).count()).sum();
    Ok((
        AmplitudeLinLccdResult {
            e_corr,
            e_total: rhf.energy + e_corr,
            keep_fraction: kept as f64 / (n * n) as f64,
            cg_iterations: it,
            cg_relres: relres,
            cg_converged: true,
        },
        dstats,
    ))
}

impl LadderVariant {
    /// Public mirror of the crate-private `needs_vvvv` used by the canonical
    /// implementation, so the amplitude-threshold port keeps the same
    /// memory shape per variant.
    pub fn needs_vvvv_pub(self) -> bool {
        matches!(self, LadderVariant::Full)
    }
}
