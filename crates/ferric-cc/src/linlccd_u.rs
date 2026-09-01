//! Open-shell (unrestricted) LinLCCD(hh).
//!
//! Same amplitude equations as [`crate::linlccd`] — see that module for the physics —
//! but built from **per-spin** orbitals, so it accepts a UHF reference or a
//! semi-canonicalized ROHF one.
//!
//! # Why this needs its own module
//!
//! The restricted path's integral builders map spin-orbital index `p` to spatial
//! `p >> 1`, i.e. they assume one spatial set shared by both spins. That is false for
//! UHF and for semi-canonical ROHF, where α and β have genuinely different spatial
//! orbitals. This module uses [`ferric_mp2::spinorbital_u`], which takes the spatial
//! chemist blocks per spin combination.
//!
//! # ROHF references
//!
//! A raw ROHF `ScfResult` carries no `eps_beta`, and ferric's other open-shell code
//! falls back to α eigenvalues for both spins — the *effective* Roothaan Fock's, which
//! belong to neither spin. Rather than silently repeat that, this function REJECTS a
//! ROHF reference and directs the caller to
//! [`ferric_scf::semicanonical::semicanonicalize`], which produces genuine per-spin
//! orbitals and eigenvalues.

use crate::linlccd::LadderVariant;
use crate::{CcConfig, CcResult};
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mo_transform::{dress_3index, transform_3center_ov};
use ferric_mp2::rimp2::cholesky_inverse_sqrt;
use ferric_mp2::spinorbital_u::{interleaved_dims, u_asym_oovv, u_asym_same, SpinBlocks};
use ferric_scf::result::Spin;
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{Array3, ArrayD, IxDyn};

/// Contract two dressed 3-index blocks into a spatial chemist 4-index block:
/// `(pq|rs) = Σ_P B^P_pq B^P_rs`.
fn chemist(b1: &Array3<f64>, b2: &Array3<f64>) -> ArrayD<f64> {
    let t1 = Tensor::new(b1.clone().into_dyn(), [Axis::Aux, Axis::O, Axis::V]);
    let t2 = Tensor::new(b2.clone().into_dyn(), [Axis::Aux, Axis::O, Axis::V]);
    einsum!("Ppq,Prs->pqrs", &t1, &t2)
}

/// Solve the LinLCCD amplitude equations with an unrestricted reference.
///
/// `scf` must be `Spin::Unrestricted`. For a ROHF reference, semi-canonicalize first
/// (see the module docs).
pub fn u_linlccd(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    cfg: &CcConfig,
    variant: LadderVariant,
) -> Result<CcResult, FerricError> {
    match scf.spin {
        Spin::Unrestricted => {}
        Spin::RestrictedOpen => {
            return Err(FerricError::General(
                "u_linlccd: a raw ROHF reference has no per-spin orbital energies (its \
                 eps are the EFFECTIVE Roothaan Fock's, belonging to neither spin). \
                 Semi-canonicalize first: \
                 ferric_scf::semicanonical::semicanonicalize(..).to_unrestricted_result(..)"
                    .into(),
            ))
        }
        Spin::Restricted => {
            return Err(FerricError::General(
                "u_linlccd: restricted reference — use ferric_cc::linlccd::linlccd".into(),
            ))
        }
    }
    if !scf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: scf.iterations,
            last_energy: scf.energy,
        });
    }

    let nbas = obs.nbasis();
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_a_tot = ((nelec + two_s) / 2) as usize;
    let nocc_b_tot = ((nelec - two_s) / 2) as usize;
    let no_a = ferric_mp2::rimp2::active_occ(nocc_a_tot, cfg.frozen_core)?;
    let no_b = ferric_mp2::rimp2::active_occ(nocc_b_tot, cfg.frozen_core)?;
    let first_occ = cfg.frozen_core;
    let (nv_a, nv_b) = (nbas - nocc_a_tot, nbas - nocc_b_tot);

    let (no2, nv2) = interleaved_dims(no_a, no_b, nv_a, nv_b);

    // Fail-fast size guard, expressed as a [`MemoryPlan`].
    //
    // # Why this changed shape
    //
    // The old estimate was a bare `2·no2²·nv2²` and omitted two things.
    //
    // 1. `eri3_ao` (`naux·nbas²`) was never charged, although it is live across
    //    every `transform_3center_ov` below — the explicit `drop(eri3_ao)`
    //    further down exists because of exactly that. The sibling `ccsd.rs`
    //    guard charges the same term, so this was a straight inconsistency.
    // 2. The unrestricted path builds THREE spin blocks (`g_aa`, `g_ab`,
    //    `g_bb`) co-resident before `u_asym_*` folds them into one interleaved
    //    tensor, which the restricted sibling does not. Those spatial blocks
    //    are individually ~1/16 of the interleaved output, but three of them
    //    plus the output is a real transient peak the old figure never saw.
    //
    // The variant conditioning matches the allocation sites exactly: OOOO is
    // built only for Hh/Full, VVVV only for Full. Charging a block that is
    // never allocated would refuse jobs that fit.
    let naux = dfbs.nbasis();
    let oovv_elems = no2.saturating_pow(2).saturating_mul(nv2.saturating_pow(2));
    let mut plan = MemoryPlan::resolve(
        cfg.memory_budget_bytes,
        format!("U-LinLCCD {variant:?} (no_a={no_a}, no_b={no_b}, nv_a={nv_a}, nv_b={nv_b})"),
    );
    plan.reserve(
        "eri3_ao (P|mn)",
        naux.saturating_mul(nbas).saturating_mul(nbas),
        Lifetime::Transient,
    );
    plan.reserve(
        "b_ov_a/b_ov_b dressed RI blocks",
        naux.saturating_mul(
            no_a.saturating_mul(nv_a).saturating_add(no_b.saturating_mul(nv_b)),
        ),
        Lifetime::Resident,
    );
    plan.reserve(
        "v_oovv <ij||ab> + oovv_t clone",
        oovv_elems.saturating_mul(2),
        Lifetime::Resident,
    );
    // `d`, `t`, the per-iteration `t.clone()`, `r` and one `einsum!` output.
    plan.reserve(
        "d denominator + t/r/x amplitude working set (x5)",
        oovv_elems.saturating_mul(5),
        Lifetime::Resident,
    );
    // The three spatial (ia|jb) spin blocks held together while u_asym_oovv
    // builds the interleaved result above.
    plan.reserve(
        "g_ovov aa/ab/bb spin blocks",
        no_a.saturating_mul(nv_a)
            .saturating_pow(2)
            .saturating_add(no_a.saturating_mul(nv_a).saturating_mul(no_b).saturating_mul(nv_b))
            .saturating_add(no_b.saturating_mul(nv_b).saturating_pow(2)),
        Lifetime::Transient,
    );
    if matches!(variant, LadderVariant::Hh | LadderVariant::Full) {
        plan.reserve("v_oooo <ij||kl>", no2.saturating_pow(4), Lifetime::Resident);
        plan.reserve(
            "g_oooo aa/ab/bb spin blocks",
            no_a.saturating_pow(4)
                .saturating_add(no_a.saturating_pow(2).saturating_mul(no_b.saturating_pow(2)))
                .saturating_add(no_b.saturating_pow(4)),
            Lifetime::Transient,
        );
    }
    if matches!(variant, LadderVariant::Full) {
        plan.reserve("v_vvvv <ab||cd>", nv2.saturating_pow(4), Lifetime::Resident);
        plan.reserve(
            "g_vvvv aa/ab/bb spin blocks",
            nv_a.saturating_pow(4)
                .saturating_add(nv_a.saturating_pow(2).saturating_mul(nv_b.saturating_pow(2)))
                .saturating_add(nv_b.saturating_pow(4)),
            Lifetime::Transient,
        );
    }
    plan.check()?;

    let eps_a = &scf.eps_alpha;
    let eps_b = scf
        .eps_beta
        .as_ref()
        .ok_or_else(|| FerricError::General("unrestricted result carries no eps_beta".into()))?;
    let c_a = &scf.mos_alpha;
    let c_b = scf
        .mos_beta
        .as_ref()
        .ok_or_else(|| FerricError::General("unrestricted result carries no beta MOs".into()))?;

    let occ_a = c_a.slice(ndarray::s![.., first_occ..first_occ + no_a]).to_owned();
    let occ_b = c_b.slice(ndarray::s![.., first_occ..first_occ + no_b]).to_owned();
    let vir_a = c_a.slice(ndarray::s![.., nocc_a_tot..]).to_owned();
    let vir_b = c_b.slice(ndarray::s![.., nocc_b_tot..]).to_owned();

    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;

    // Per-spin dressed 3-index blocks.
    let b_ov_a = dress_3index(&transform_3center_ov(&eri3_ao, &occ_a, &vir_a), &v_inv_sqrt);
    let b_ov_b = dress_3index(&transform_3center_ov(&eri3_ao, &occ_b, &vir_b), &v_inv_sqrt);

    // oovv: three distinct spin combinations of (ia|jb).
    let g_ovov_aa = chemist(&b_ov_a, &b_ov_a);
    let g_ovov_ab = chemist(&b_ov_a, &b_ov_b);
    let g_ovov_bb = chemist(&b_ov_b, &b_ov_b);
    let v_oovv = u_asym_oovv(
        &SpinBlocks { aa: &g_ovov_aa, ab: &g_ovov_ab, bb: &g_ovov_bb },
        no_a,
        no_b,
        nv_a,
        nv_b,
    );

    let oooo_t = if matches!(variant, LadderVariant::Hh | LadderVariant::Full) {
        let b_oo_a = dress_3index(&transform_3center_ov(&eri3_ao, &occ_a, &occ_a), &v_inv_sqrt);
        let b_oo_b = dress_3index(&transform_3center_ov(&eri3_ao, &occ_b, &occ_b), &v_inv_sqrt);
        let g_aa = chemist(&b_oo_a, &b_oo_a);
        let g_ab = chemist(&b_oo_a, &b_oo_b);
        let g_bb = chemist(&b_oo_b, &b_oo_b);
        let v = u_asym_same(&SpinBlocks { aa: &g_aa, ab: &g_ab, bb: &g_bb }, no_a, no_b);
        Some(Tensor::new(v, [Axis::O, Axis::O, Axis::O, Axis::O]))
    } else {
        None
    };

    let vvvv_t = if matches!(variant, LadderVariant::Full) {
        let b_vv_a = dress_3index(&transform_3center_ov(&eri3_ao, &vir_a, &vir_a), &v_inv_sqrt);
        let b_vv_b = dress_3index(&transform_3center_ov(&eri3_ao, &vir_b, &vir_b), &v_inv_sqrt);
        let g_aa = chemist(&b_vv_a, &b_vv_a);
        let g_ab = chemist(&b_vv_a, &b_vv_b);
        let g_bb = chemist(&b_vv_b, &b_vv_b);
        let v = u_asym_same(&SpinBlocks { aa: &g_aa, ab: &g_ab, bb: &g_bb }, nv_a, nv_b);
        Some(Tensor::new(v, [Axis::V, Axis::V, Axis::V, Axis::V]))
    } else {
        None
    };
    drop(eri3_ao);

    let oovv_t = Tensor::new(v_oovv.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);

    // Stage-seam RSS safety net: the transient `eri3_ao` and the spatial spin
    // blocks are freed and every resident tensor exists, so RSS here is
    // directly comparable to the plan's projected peak. Observational only.
    ferric_core::memory::warn_if_rss_over(
        "U-LinLCCD MO blocks built",
        plan.budget_bytes(),
        1.1,
    );

    // Interleaved spin-orbital energies. Padded slots (where one spin has fewer
    // orbitals) get a large gap so any residual amplitude there is driven to zero;
    // their integrals are already zero, so this only avoids a 0/0.
    let big = 1.0e6;
    let mut eo = vec![big; no2];
    let mut ev = vec![-big; nv2];
    for i in 0..no_a {
        eo[2 * i] = eps_a[first_occ + i];
    }
    for i in 0..no_b {
        eo[2 * i + 1] = eps_b[first_occ + i];
    }
    for a in 0..nv_a {
        ev[2 * a] = eps_a[nocc_a_tot + a];
    }
    for a in 0..nv_b {
        ev[2 * a + 1] = eps_b[nocc_b_tot + a];
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

    let mut t = &v_oovv / &d;
    let dim = no2 * nv2;
    let mut diis = ferric_scf::diis::Diis::new(cfg.diis_subspace.max(1));
    let mut e_old = 0.0;

    for iter in 0..cfg.max_iter {
        let t_t = Tensor::new(t.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);
        let e_corr: f64 = 0.25 * einsum!("ijab,ijab->", &oovv_t, &t_t);
        if iter > 0 && (e_corr - e_old).abs() < cfg.energy_conv {
            let t2 = t.clone().into_dimensionality::<ndarray::Ix4>().unwrap();
            return Ok(CcResult { correlation_energy: e_corr, t1: None, t2 });
        }
        e_old = e_corr;

        let mut r = v_oovv.clone();
        if let Some(oooo) = &oooo_t {
            let x: ArrayD<f64> = einsum!("klij,klab->ijab", oooo, &t_t);
            r = r + 0.5 * x;
        }
        if let Some(vvvv) = &vvvv_t {
            let x: ArrayD<f64> = einsum!("ijcd,abcd->ijab", &t_t, vvvv);
            r = r + 0.5 * x;
        }

        let t_new = &r / &d;
        let err = &t_new - &t;
        let t_flat = t_new.view().into_shape_with_order((dim, dim)).unwrap().to_owned();
        let err_flat = err.view().into_shape_with_order((dim, dim)).unwrap().to_owned();
        t = diis
            .step(&t_flat, &err_flat)
            .into_shape_with_order(IxDyn(&[no2, no2, nv2, nv2]))
            .unwrap();
    }

    Err(FerricError::Convergence(format!(
        "U-LinLCCD {variant:?} did not converge in {} iterations",
        cfg.max_iter
    )))
}
