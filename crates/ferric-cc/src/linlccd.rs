//! Linearized ladder coupled cluster doubles — LinLCCD and its hole–hole
//! approximation LinLCCD(hh).
//!
//! Method: Carter-Fenk, *J. Phys. Chem. A* **2025**, 129, 7251–7260
//! (doi:10.1021/acs.jpca.5c03203, `papers/linccd.pdf`).
//!
//! LinCCD (the paper's eq. 5) truncates coupled cluster at first order in `T̂₂`. It
//! diverges in small-gap systems. The paper's diagnosis is that the culprit is the
//! **exchange** part of the ring and crossed-ring contractions, not small denominators
//! per se. Removing ring/crossed-ring terms entirely leaves only driver + ladder
//! diagrams:
//!
//! ```text
//! LinLCCD (eq. 7):
//!   0 = <ij||ab> + (f_ac δ_bd + δ_ac f_bd) t_ij^cd
//!                - (f_ik δ_jl + δ_ik f_jl) t_kl^ab
//!                + ½ v_ij^kl t_kl^ab        (hh ladder)
//!                + ½ v_cd^ab t_ij^cd        (pp ladder)
//! ```
//!
//! Dropping the pp ladder — the O(n_o²n_v⁴) term responsible for CCD's O(N⁶) cost —
//! gives LinLCCD(hh) (eq. 14), which scales as **O(n_o⁴n_v²)**:
//!
//! ```text
//!   (f_ac δ_bd + δ_ac f_bd) t_ij^cd
//!     − (f_ik δ_jl + δ_ik f_jl + ½ v_ij^kl) t_kl^ab = −<ij||ab>
//! ```
//!
//! In a canonical basis the Fock terms collapse to the MP2 denominator, so this is
//! solved by Jacobi iteration + DIIS exactly as CCD is. The paper's eq. 15 recasts it as
//! `(ε_a + ε_b − η̃_ij) t̃_ij^ab = −ṽ_ij^ab`, showing *why* it is regular: the hh ladder
//! widens the effective gap in an **amplitude-independent** way. That dressed form is
//! interpretive; like the paper's own implementation we solve the equations
//! self-consistently rather than diagonalizing the n_o²×n_o² hh super-Fock matrix.
//!
//! LinLCCD(hh) is size-consistent, size-extensive, and orbital-invariant.
//!
//! Relative to [`super::ccd`], this module builds strictly fewer integral blocks: the
//! `(2n_v)⁴` VVVV tensor that dominates CCD's memory is never formed for
//! [`LadderVariant::Hh`], nor is OVVO. See
//! `docs/superpowers/specs/2026-07-26-linlccd-hh-design.md`.

use super::{CcConfig, CcResult};
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::cholesky_inverse_sqrt;
use ferric_mp2::spinorbital::{asym_oovv, asym_same, build_b};
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{ArrayD, IxDyn};

/// Which ladder diagrams to retain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LadderVariant {
    /// Driver terms only — no ladder contractions at all.
    ///
    /// These *are* the MP2 amplitude equations (paper eqs. 17–18), so this variant
    /// reproduces RI-MP2 exactly. It exists to isolate the ladder contractions during
    /// validation: it pins the integrals, denominators, and energy expression against
    /// already-validated RI-MP2, leaving a ladder error as the only remaining suspect.
    DriversOnly,
    /// Hole–hole ladder only — **LinLCCD(hh)**, eq. 14. O(n_o⁴n_v²).
    ///
    /// Never forms the VVVV block.
    Hh,
    /// Both hole–hole and particle–particle ladders — full **LinLCCD**, eq. 7.
    ///
    /// Restores the O(n_o²n_v⁴) pp ladder and the `(2n_v)⁴` VVVV block, so it carries
    /// CCD-like memory cost. Exact for two-electron systems, which LinLCCD(hh) is not.
    Full,
}

impl LadderVariant {
    fn needs_vvvv(self) -> bool {
        matches!(self, LadderVariant::Full)
    }
    fn needs_oooo(self) -> bool {
        matches!(self, LadderVariant::Hh | LadderVariant::Full)
    }
}

/// Solve the linearized ladder CCD amplitude equations.
///
/// `op` selects the two-electron operator, so the Coulomb-attenuated variant required by
/// the ωB97X-L-V double hybrid is a parameter rather than a separate code path: pass
/// [`Operator::erfc`] for short-range-only correlation.
///
/// Closed-shell RHF reference only. Open-shell requires ROHF semi-canonicalization,
/// which ferric does not yet have — see the design doc.
pub fn linlccd(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &CcConfig,
    variant: LadderVariant,
) -> Result<CcResult, FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let no = ferric_mp2::rimp2::active_occ(nocc_total, cfg.frozen_core)?;
    let first_occ = cfg.frozen_core;
    let nv = nbas - nocc_total;

    let no2 = 2 * no;
    let nv2 = 2 * nv;

    // Fail-fast size guard, expressed as a [`MemoryPlan`].
    //
    // # Why this changed shape
    //
    // Two things. First, the VVVV/OOOO conditioning below is DELIBERATE and is
    // preserved exactly: for Hh/DriversOnly those blocks are never built, and
    // guarding on a tensor we do not allocate would reject systems this method
    // runs comfortably (an over-estimating guard is also a bug). The plan
    // expresses that by only reserving a block when `variant.needs_*()` says it
    // is formed.
    //
    // Second, the old estimate omitted `eri3_ao` (`naux·nbas²`) entirely. That
    // tensor is unambiguously live across every MO-block build here — the
    // explicit `drop(eri3_ao)` further below exists precisely because it stays
    // resident until then, and each conditional `build_b` reads it. The sibling
    // `ccsd.rs` guard charges the same term, so this was a straight
    // inconsistency. Also newly charged: the full `v_oovv.clone()` into
    // `oovv_t`, and the `d`/`t`/`r` amplitude working set.
    //
    // `eri3_ao` is TRANSIENT (dead after `drop(eri3_ao)`, before the amplitude
    // loop's peak); the MO blocks and amplitudes are resident.
    let naux = dfbs.nbasis();
    let oovv_elems = no2.saturating_pow(2).saturating_mul(nv2.saturating_pow(2));
    let mut plan = MemoryPlan::resolve(
        cfg.memory_budget_bytes,
        format!("LinLCCD {variant:?} (no={no}, nv={nv} spatial)"),
    );
    plan.reserve(
        "eri3_ao (P|mn)",
        naux.saturating_mul(nbas).saturating_mul(nbas),
        Lifetime::Transient,
    );
    plan.reserve(
        "b_ov dressed RI block",
        naux.saturating_mul(no).saturating_mul(nv),
        Lifetime::Resident,
    );
    plan.reserve(
        "v_oovv <ij||ab> + oovv_t clone",
        oovv_elems.saturating_mul(2),
        Lifetime::Resident,
    );
    // `d`, `t`, the per-iteration `t.clone()`, `r` (seeded from a
    // `v_oovv.clone()`) and one `einsum!` output are the same shape and
    // co-resident inside the loop.
    plan.reserve(
        "d denominator + t/r/x amplitude working set (x5)",
        oovv_elems.saturating_mul(5),
        Lifetime::Resident,
    );
    if variant.needs_oooo() {
        plan.reserve(
            "b_oo dressed RI block",
            naux.saturating_mul(no).saturating_mul(no),
            Lifetime::Resident,
        );
        plan.reserve("v_oooo <ij||kl>", no2.saturating_pow(4), Lifetime::Resident);
        plan.reserve(
            "g_ijkl einsum! intermediate",
            no2.saturating_pow(4),
            Lifetime::Transient,
        );
    }
    if variant.needs_vvvv() {
        plan.reserve(
            "b_vv dressed RI block",
            naux.saturating_mul(nv).saturating_mul(nv),
            Lifetime::Resident,
        );
        plan.reserve("v_vvvv <ab||cd>", nv2.saturating_pow(4), Lifetime::Resident);
        plan.reserve(
            "g_abcd einsum! intermediate",
            nv2.saturating_pow(4),
            Lifetime::Transient,
        );
    }
    plan.check()?;

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + no]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // V^{-1/2} metric and AO 3-center integrals under the requested operator.
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;

    // Spatial dressed RI 3-index MO blocks. b_vv is built only when the pp ladder needs it.
    let b_ov = build_b(
        &transform_3center_ov(&eri3_ao, &c_occ, &c_vir),
        &v_inv_sqrt,
        Axis::O,
        Axis::V,
    );

    // --- Spin-orbital antisymmetrized integral blocks ---
    let v_oovv = {
        let g_iajb: ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);
        asym_oovv(&g_iajb, no, nv)
    };
    let oovv_t = Tensor::new(v_oovv.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);

    let oooo_t = if variant.needs_oooo() {
        let b_oo = build_b(&transform_3center_oo(&eri3_ao, &c_occ), &v_inv_sqrt, Axis::O, Axis::O);
        let g_ijkl: ArrayD<f64> = einsum!("Pij,Pkl->ijkl", &b_oo, &b_oo);
        Some(Tensor::new(asym_same(&g_ijkl, no), [Axis::O, Axis::O, Axis::O, Axis::O]))
    } else {
        None
    };

    let vvvv_t = if variant.needs_vvvv() {
        let b_vv = build_b(&transform_3center_vv(&eri3_ao, &c_vir), &v_inv_sqrt, Axis::V, Axis::V);
        let g_abcd: ArrayD<f64> = einsum!("Pab,Pcd->abcd", &b_vv, &b_vv);
        Some(Tensor::new(asym_same(&g_abcd, nv), [Axis::V, Axis::V, Axis::V, Axis::V]))
    } else {
        None
    };

    drop(eri3_ao);

    // Stage-seam RSS safety net: the transient `eri3_ao` has just been freed and
    // every resident MO block exists, so RSS here is directly comparable to the
    // plan's projected peak. Observational only — it never errors. This file's
    // guard had omitted `eri3_ao` outright before the plan rewrite above, which
    // is precisely the estimator-undershoot class this net exists to surface.
    ferric_core::memory::warn_if_rss_over(
        "LinLCCD MO blocks built",
        plan.budget_bytes(),
        1.1,
    );

    // --- Spin-orbital energies (even index = alpha, odd = beta) ---
    let mut eo = vec![0.0f64; no2];
    let mut ev = vec![0.0f64; nv2];
    for i in 0..no {
        eo[2 * i] = eps[first_occ + i];
        eo[2 * i + 1] = eps[first_occ + i];
    }
    for a in 0..nv {
        ev[2 * a] = eps[nocc_total + a];
        ev[2 * a + 1] = eps[nocc_total + a];
    }
    let mut d = ArrayD::zeros(IxDyn(&[no2, no2, nv2, nv2]));
    for i in 0..no2 {
        for j in 0..no2 {
            for a in 0..nv2 {
                for b in 0..nv2 {
                    d[[i, j, a, b]] = eo[i] + eo[j] - ev[a] - ev[b];
                }
            }
        }
    }

    // MP2 guess: t = <ij||ab> / D. For DriversOnly this is already the exact answer.
    let mut t = &v_oovv / &d;

    let dim = no2 * nv2;
    let mut diis = ferric_scf::diis::Diis::new(cfg.diis_subspace.max(1));
    let mut e_old = 0.0;

    // Honor cfg.energy_conv (ccd.rs hardcodes 1e-10 and ignores it).
    let conv = cfg.energy_conv;

    for iter in 0..cfg.max_iter {
        let t_t = Tensor::new(t.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);

        // Energy: E = 0.25 <ij||ab> t_ijab  (paper eq. 6).
        let e_corr: f64 = 0.25 * einsum!("ijab,ijab->", &oovv_t, &t_t);
        let d_e = (e_corr - e_old).abs();
        if iter > 0 && d_e < conv {
            let t2 = t.clone().into_dimensionality::<ndarray::Ix4>().unwrap();
            return Ok(CcResult { correlation_energy: e_corr, t1: None, t2 });
        }
        e_old = e_corr;

        // --- Residual R[i,j,a,b] ---
        // Sign/factor conventions inherited verbatim from the validated CCD residual
        // (ccd.rs) so that LinLCCD is strictly CCD-minus-terms.
        let mut r = v_oovv.clone();

        // hh ladder: 0.5 <kl||ij> t_klab
        if let Some(oooo) = &oooo_t {
            let x: ArrayD<f64> = einsum!("klij,klab->ijab", oooo, &t_t);
            r = r + 0.5 * x;
        }
        // pp ladder: 0.5 <ab||cd> t_ijcd   (Full only)
        if let Some(vvvv) = &vvvv_t {
            let x: ArrayD<f64> = einsum!("ijcd,abcd->ijab", &t_t, vvvv);
            r = r + 0.5 * x;
        }

        // Jacobi update; the increment is the DIIS error vector.
        let t_new = &r / &d;
        let err = &t_new - &t;

        let t_flat = t_new.view().into_shape_with_order((dim, dim)).unwrap().to_owned();
        let err_flat = err.view().into_shape_with_order((dim, dim)).unwrap().to_owned();
        let t_ext = diis.step(&t_flat, &err_flat);
        t = t_ext.into_shape_with_order(IxDyn(&[no2, no2, nv2, nv2])).unwrap();
    }

    Err(FerricError::Convergence(format!(
        "LinLCCD {variant:?} did not converge in {} iterations",
        cfg.max_iter
    )))
}
