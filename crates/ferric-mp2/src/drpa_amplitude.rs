//! Amplitude-threshold direct RPA (drCCD Riccati) in the localized basis —
//! Rust port of `scripts/amplitude_rpa_proto.py`, built AFTER the SymPy
//! proof notebook `wiki/notebooks/12-amplitude-threshold-drpa.ipynb`, which
//! establishes everything the anchors here rely on:
//!
//! 1. the drCCD Riccati root ≡ the plasmon-formula dRPA energy (exact
//!    symbolic scalar; 50-digit matrix identity) — so the canonical-plasmon
//!    anchor below tests the IMPLEMENTATION, never the formulation;
//! 2. the correlation energy is invariant under orthogonal rotations of the
//!    pair basis — the license for solving in the Boys-localized basis;
//! 3. dRPA is NON-VARIATIONAL: first order in both amplitude and integral
//!    perturbations (unlike MP2's Hylleraas stationarity), so the canonical
//!    reference is built on the exactly SEMICANONICALIZED recomputed Fock
//!    (the Python rig measured a 4e-7 Fock inconsistency producing a 1.7e-9
//!    anchor failure), and masked-solve energy errors are expected ~linear
//!    in ε, not quadratic.
//!
//! Spin-adapted closed-shell drCCD: `B_iajb = 2 (ia|jb)`,
//! `R(T) = B + F(T) + BT + TB + TBT = 0` with `F` the same non-canonical
//! Fock superoperator as `lmp2_amplitude`'s linear MP2 operator, and
//! `E_c = ½ Σ B ∘ T`. The mask is the Eq-8-style threshold on |B| (B is
//! symmetric, so the pattern is swap-closed automatically). Solver: damped
//! fixed point `T ← T − R(T)/D` projected onto the pattern each iteration
//! (the Python rig measured masking to REMOVE ring coupling — iteration
//! counts fall as ε loosens).
//!
//! V1 SCOPE: the masked solve is DENSE over the (no·nv)² compound space —
//! prototype parity (the Python rig is dense numpy too). The ragged
//! pair-block ring product (triple-domain intersections) is future work;
//! nothing here carries a cost or scaling claim. dRPA also has no
//! same-spin exchange diagrams by FORMULATION (see the SS section of
//! `wiki/amplitude-threshold-drpa.md`) — that is not a property of the
//! threshold. Closed-shell, library-only.

use ndarray::{s, Array2, Array4};

use ferric_core::basis::BasisSet;
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;

use crate::lmp2_amplitude::{
    assemble_basis, assemble_localized, assemble_ragged_direct, build_vvhv, check_vvhv,
    AmplitudeLmp2Config, VvHv,
};
use crate::ragged::{apply_pattern, matvec_indexed, ring_product};
use crate::rimp2::{
    active_occ, eri3_budget_bytes, eri3_mo_ov_blocked, metric_inverse_sqrt,
};
use ferric_integrals::threeindex::coulomb_metric_2c;

#[derive(Debug, Clone)]
pub struct AmplitudeDrpaConfig {
    /// Threshold ε on |B_iajb| = |2(ia|jb)| in the localized basis; 0 keeps
    /// everything (the exactness-anchor limit).
    pub eps: f64,
    pub frozen_core: usize,
    pub fp_rtol: f64,
    pub fp_max_iter: usize,
    pub eri3_budget_bytes: Option<usize>,
    /// Integral-free pair gate (see `lmp2_amplitude`); gated pairs are
    /// never assembled in the ragged path.
    pub pair_gate_cal: Option<f64>,
}

impl Default for AmplitudeDrpaConfig {
    fn default() -> Self {
        Self { eps: 1e-4, frozen_core: 0, fp_rtol: 1e-12, fp_max_iter: 500, eri3_budget_bytes: None, pair_gate_cal: None }
    }
}

#[derive(Debug)]
pub struct AmplitudeDrpaResult {
    pub e_corr: f64,
    pub e_total: f64,
    /// Canonical plasmon-formula dRPA on the exactly semicanonicalized
    /// recomputed Fock — the independent-construction reference (eigenvalue
    /// problem vs localized Riccati fixed point; shares only the RI
    /// integrals and the Fock operator).
    pub e_corr_plasmon_canonical: f64,
    pub keep_fraction: f64,
    pub pair_fraction: f64,
    pub iterations: usize,
    pub relres: f64,
    pub converged: bool,
}

/// Fock superoperator F(T) in the compound (ia),(jb) representation:
/// F(T)_iajb = Σ_c f_vv[a,c] T_icjb + T_iajc f_vv[c,b]
///           − Σ_k f_oo[i,k] T_kajb − T_iakb f_oo[k,j].
fn fock_super(t4: &Array4<f64>, f_oo: &Array2<f64>, f_vv: &Array2<f64>) -> Array4<f64> {
    let (no, nv, _, _) = t4.dim();
    let mut r = Array4::<f64>::zeros((no, nv, no, nv));
    for i in 0..no {
        for j in 0..no {
            let blk = t4.slice(s![i, .., j, ..]);
            let mut acc = f_vv.dot(&blk);
            acc += &blk.dot(f_vv);
            r.slice_mut(s![i, .., j, ..]).assign(&acc);
        }
    }
    for i in 0..no {
        for j in 0..no {
            // occupied couplings gather whole (k,·) blocks
            let mut acc = Array2::<f64>::zeros((nv, nv));
            for k in 0..no {
                let fik = f_oo[(i, k)];
                if fik != 0.0 {
                    acc.scaled_add(-fik, &t4.slice(s![k, .., j, ..]));
                }
                let fkj = f_oo[(k, j)];
                if fkj != 0.0 {
                    acc.scaled_add(-fkj, &t4.slice(s![i, .., k, ..]));
                }
            }
            let mut out = r.slice_mut(s![i, .., j, ..]);
            out += &acc;
        }
    }
    r
}

fn to4(m: &Array2<f64>, no: usize, nv: usize) -> Array4<f64> {
    m.clone().into_shape_with_order((no, nv, no, nv)).expect("compound reshape")
}

fn to2(m: &Array4<f64>) -> Array2<f64> {
    let (no, nv, _, _) = m.dim();
    m.clone().into_shape_with_order((no * nv, no * nv)).expect("compound reshape")
}

/// Amplitude-threshold dRPA with the VV-HV space built internally.
pub fn amplitude_drpa(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeDrpaConfig,
) -> Result<AmplitudeDrpaResult, FerricError> {
    let vvhv = build_vvhv(mol, obs, obs_bs, rhf)?;
    let nocc_total = (mol.nelec() as usize) / 2;
    let (dev_orth, dev_span) = check_vvhv(obs, rhf, nocc_total, &vvhv.c_vloc);
    if dev_orth > 1e-8 || dev_span > 1e-8 {
        return Err(FerricError::General(format!(
            "drpa_amplitude: VV-HV construction check failed (orth {dev_orth:.2e}, span {dev_span:.2e})"
        )));
    }
    amplitude_drpa_with_virtuals(mol, obs, dfbs, op, rhf, cfg, &vvhv)
}

/// Mutation-test / caller-supplied-virtuals entry point — RAGGED path
/// (direct assembly, no dense (no·nv)² object).
pub fn amplitude_drpa_with_virtuals(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeDrpaConfig,
    vvhv: &VvHv,
) -> Result<AmplitudeDrpaResult, FerricError> {
    let lcfg = AmplitudeLmp2Config {
        eps: cfg.eps,
        frozen_core: cfg.frozen_core,
        eri3_budget_bytes: cfg.eri3_budget_bytes,
        ..Default::default()
    };
    let lb = assemble_basis(mol, obs, dfbs, op, rhf, &lcfg, vvhv)?;
    // B = 2 (ia|jb), assembled directly onto ragged blocks (scale = 2.0;
    // the eps mask therefore acts on |B| exactly as the dense V1 did)
    let (rg, _gated) = assemble_ragged_direct(
        mol,
        dfbs,
        op,
        &lb,
        cfg.eps,
        2.0,
        cfg.pair_gate_cal,
        None,
    )?;
    let b_blocks: Vec<Array2<f64>> = rg.pairs.iter().map(|pb| pb.j_blk.clone()).collect();
    let bnorm = b_blocks.iter().map(|b| b.iter().map(|x| x * x).sum::<f64>()).sum::<f64>().sqrt();

    // damped fixed point T <- T - R(T)/D on the pattern, all ragged
    let mut t: Vec<Array2<f64>> = rg
        .pairs
        .iter()
        .map(|pb| Array2::<f64>::zeros((pb.da.len(), pb.db.len())))
        .collect();
    let mut relres = 1.0;
    let mut it = 0;
    let mut converged = bnorm == 0.0;
    let mut flops = 0u64;
    while it < cfg.fp_max_iter && !converged {
        it += 1;
        let f_t = matvec_indexed(&rg, &lb.f_oo, &t, &mut flops); // pattern-projected
        let bt = ring_product(&rg, &b_blocks, &t);
        let tb = ring_product(&rg, &t, &b_blocks);
        let tbt = ring_product(&rg, &t, &bt);
        let mut r2 = 0.0f64;
        let mut new_t = Vec::with_capacity(rg.pairs.len());
        for (p, pb) in rg.pairs.iter().enumerate() {
            let mut r = &b_blocks[p] + &f_t[p];
            r += &bt[p];
            r += &tb[p];
            r += &tbt[p];
            apply_pattern(&mut r, &pb.pat, pb.db.len());
            r2 += r.iter().map(|x| x * x).sum::<f64>();
            let mut tn = t[p].clone();
            tn -= &(&r / &pb.denom);
            apply_pattern(&mut tn, &pb.pat, pb.db.len());
            new_t.push(tn);
        }
        t = new_t;
        relres = r2.sqrt() / bnorm.max(1e-300);
        if relres < cfg.fp_rtol {
            converged = true;
        }
    }
    if !converged {
        return Err(FerricError::General(format!(
            "drpa_amplitude(ragged): Riccati fixed point failed to converge (relres {relres:.2e} after {it} iters)"
        )));
    }
    // E = 1/2 Σ B ∘ T on the pattern
    let mut e_corr = 0.0;
    for (p, pb) in rg.pairs.iter().enumerate() {
        let nbb = pb.db.len();
        for r in 0..pb.da.len() {
            for c in 0..nbb {
                if pb.pat[r * nbb + c] {
                    e_corr += 0.5 * b_blocks[p][(r, c)] * t[p][(r, c)];
                }
            }
        }
    }
    let e_ref = canonical_plasmon_drpa(mol, obs, dfbs, op, rhf, cfg)?;
    let n = lb.no * lb.nv;
    let kept: usize = rg.pairs.iter().map(|pb| pb.pat.iter().filter(|&&x| x).count()).sum();
    Ok(AmplitudeDrpaResult {
        e_corr,
        e_total: rhf.energy + e_corr,
        e_corr_plasmon_canonical: e_ref,
        keep_fraction: kept as f64 / (n * n) as f64,
        pair_fraction: rg.pairs.len() as f64 / (lb.no * lb.no) as f64,
        iterations: it,
        relres,
        converged,
    })
}

/// DENSE V1 solver (compound-matrix fixed point with FULL intermediates) —
/// retained as the independent cross-check for the ragged path below. The
/// two differ at finite ε: the ragged variant projects the ring
/// INTERMEDIATES onto the pattern as well (a slightly stronger truncation,
/// measured sub-dominant to the threshold error itself — see the tests);
/// at ε = 0 they are identical by construction.
pub fn amplitude_drpa_dense(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeDrpaConfig,
    vvhv: &VvHv,
) -> Result<AmplitudeDrpaResult, FerricError> {
    // reuse the LMP2 assembly (Boys occupieds, localized Fock blocks, RI J)
    let lcfg = AmplitudeLmp2Config {
        eps: cfg.eps,
        frozen_core: cfg.frozen_core,
        eri3_budget_bytes: cfg.eri3_budget_bytes,
        ..Default::default()
    };
    let lp = assemble_localized(mol, obs, dfbs, op, rhf, &lcfg, vvhv)?;
    let (no, nv) = (lp.no, lp.nv);
    let n = no * nv;
    let b2 = lp.j_dense.mapv(|x| 2.0 * x); // B = 2 (ia|jb)
    let mask: Vec<bool> = b2.iter().map(|&x| cfg.eps == 0.0 || x.abs() > cfg.eps).collect();
    let kept = mask.iter().filter(|&&m| m).count();

    // denominators from the localized Fock diagonals (positive for gapped)
    let mut d2 = Array2::<f64>::zeros((n, n));
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    let v = lp.f_vv[(a, a)] + lp.f_vv[(b, b)] - lp.f_oo[(i, i)] - lp.f_oo[(j, j)];
                    d2[(i * nv + a, j * nv + b)] = v;
                }
            }
        }
    }
    if d2.iter().any(|&x| x <= 0.0) {
        return Err(FerricError::General(
            "drpa_amplitude: non-positive denominator (not a gapped system?)".into(),
        ));
    }

    let apply_mask = |m: &mut Array2<f64>| {
        for (v, &keep) in m.iter_mut().zip(&mask) {
            if !keep {
                *v = 0.0;
            }
        }
    };
    let mut b_masked = b2.clone();
    apply_mask(&mut b_masked);
    let bnorm = b_masked.iter().map(|x| x * x).sum::<f64>().sqrt();

    // damped fixed point T <- T - R(T)/D on the pattern
    let mut t2 = Array2::<f64>::zeros((n, n));
    let mut relres = 1.0;
    let mut it = 0;
    let mut converged = false;
    while it < cfg.fp_max_iter {
        it += 1;
        let t4 = to4(&t2, no, nv);
        let f_t = to2(&fock_super(&t4, &lp.f_oo, &lp.f_vv));
        let bt = b2.dot(&t2);
        let tbt = t2.dot(&bt);
        let mut r = &b_masked + &f_t;
        r += &bt;
        r += &t2.dot(&b2);
        r += &tbt;
        apply_mask(&mut r);
        relres = r.iter().map(|x| x * x).sum::<f64>().sqrt() / bnorm.max(1e-300);
        if relres < cfg.fp_rtol {
            converged = true;
            break;
        }
        t2 -= &(&r / &d2);
        apply_mask(&mut t2);
    }
    if !converged {
        return Err(FerricError::General(format!(
            "drpa_amplitude: Riccati fixed point failed to converge (relres {relres:.2e} after {it} iters)"
        )));
    }
    let e_corr = 0.5 * b_masked.iter().zip(t2.iter()).map(|(b, t)| b * t).sum::<f64>();

    let e_ref = canonical_plasmon_drpa(mol, obs, dfbs, op, rhf, cfg)?;

    let pair_any = {
        let mut any = vec![false; no * no];
        for (idx, &m) in mask.iter().enumerate() {
            if m {
                let (row, col) = (idx / n, idx % n);
                any[(row / nv) * no + (col / nv)] = true;
            }
        }
        any.iter().filter(|&&x| x).count()
    };

    Ok(AmplitudeDrpaResult {
        e_corr,
        e_total: rhf.energy + e_corr,
        e_corr_plasmon_canonical: e_ref,
        keep_fraction: kept as f64 / (n * n) as f64,
        pair_fraction: pair_any as f64 / (no * no) as f64,
        iterations: it,
        relres,
        converged,
    })
}

/// Canonical plasmon-formula dRPA on the EXACTLY semicanonicalized
/// recomputed Fock: A = D + B, B = 2(ia|jb);
/// Ω_n² = eig[(A−B)(A+B)]; E = ½(ΣΩ − Tr A).
///
/// Semicanonicalization is load-bearing (proof notebook §3): dRPA is first
/// order in Fock inconsistencies, so the occupied/virtual blocks of the
/// CONVERGED Fock are re-diagonalized here rather than trusting
/// `rhf.eps_r()` to be its exact diagonal.
pub fn canonical_plasmon_drpa(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeDrpaConfig,
) -> Result<f64, FerricError> {
    use ndarray_linalg::{Eig, Eigh, UPLO};
    let nocc_total = (mol.nelec() as usize) / 2;
    let no = active_occ(nocc_total, cfg.frozen_core)?;
    let first = cfg.frozen_core;
    let nbas = obs.nbasis();
    let nvir = nbas - nocc_total;

    let f_ao = rhf.fock_r();
    let c_occ = rhf.mos_r().slice(s![.., first..nocc_total]).to_owned();
    let c_vir = rhf.mos_r().slice(s![.., nocc_total..]).to_owned();
    let semicanon = |c: &Array2<f64>| -> Result<(Array2<f64>, Vec<f64>), FerricError> {
        let f_blk = c.t().dot(&f_ao.dot(c));
        let (w, u) = f_blk
            .eigh(UPLO::Lower)
            .map_err(|e| FerricError::General(format!("drpa semicanonicalize: {e}")))?;
        Ok((c.dot(&u), w.to_vec()))
    };
    let (c_occ, e_occ) = semicanon(&c_occ)?;
    let (c_vir, e_vir) = semicanon(&c_vir)?;

    let budget = eri3_budget_bytes(cfg.eri3_budget_bytes);
    let b3 = eri3_mo_ov_blocked(op, obs, dfbs, &c_occ, &c_vir, budget)?;
    let naux = b3.shape()[0];
    let v = coulomb_metric_2c(op, dfbs)?;
    let vis = metric_inverse_sqrt(&v, op)?;
    let b_flat = b3
        .into_shape_with_order((naux, no * nvir))
        .map_err(|e| FerricError::General(format!("drpa reshape: {e}")))?;
    let btilde = vis.dot(&b_flat);
    let b_mat = btilde.t().dot(&btilde).mapv(|x| 2.0 * x); // 2 (ia|jb)

    let n = no * nvir;
    let mut a_mat = b_mat.clone();
    for i in 0..no {
        for a in 0..nvir {
            let idx = i * nvir + a;
            a_mat[(idx, idx)] += e_vir[a] - e_occ[i];
        }
    }
    let m = (&a_mat - &b_mat).dot(&(&a_mat + &b_mat));
    let (w, _) = m
        .eig()
        .map_err(|e| FerricError::General(format!("drpa plasmon eig: {e}")))?;
    let mut sum_omega = 0.0;
    for lam in w.iter() {
        if lam.im.abs() > 1e-8 * lam.re.abs().max(1.0) || lam.re < 0.0 {
            return Err(FerricError::General(format!(
                "drpa plasmon: non-real/negative Omega^2 = {lam} (RPA instability?)"
            )));
        }
        sum_omega += lam.re.sqrt();
    }
    let tr_a: f64 = (0..n).map(|k| a_mat[(k, k)]).sum();
    Ok(0.5 * (sum_omega - tr_a))
}
