use super::{CcConfig, CcResult};
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::{active_occ, cholesky_inverse_sqrt};
use ferric_mp2::spinorbital::{asym_oovv, asym_ovvo, asym_same, build_b, transpose_b};
use ferric_tensors::{einsum, permute_to_owned, Axis, Tensor};
use ndarray::{ArrayD, IxDyn};

// Permutation antisymmetrizers P(ij)/P(ab)/P(ij)P(ab) on the i,j,a,b axes are
// shared with the CCSD residual builder — see `helpers.rs`.
use super::helpers::{p_ab, p_ij, p_ij_ab};

/// Compute the CCD correlation energy.
///
/// Delegates to [`ccd_spinorbital`], the complete spin-orbital CCD (all residual
/// terms, `einsum!` + DIIS). The previous closed-shell implementation was
/// incomplete (pp+hh ladders only) and converged only for H2; this is the
/// validated full residual that converges on real molecules.
pub fn ccd(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &CcConfig,
) -> Result<CcResult, FerricError> {
    ccd_spinorbital(mol, obs, dfbs, op, rhf, cfg)
}

/// Complete spin-orbital CCD (all residual terms) via `einsum!` + DIIS.
///
/// Amplitudes `t[i,j,a,b]` (spin-orbital, antisymmetric in ij and ab).
/// Residual (all integrals antisymmetrized `<pq||rs>`):
/// ```text
/// R = <ij||ab>
///   + 0.5 <ab||cd> t_ijcd                              (pp ladder)
///   + 0.5 <kl||ij> t_klab                              (hh ladder)
///   + P(ij)P(ab) <kb||cj> t_ikac                       (ring)
///   + 0.25 <kl||cd> t_ijcd t_klab                      (quad 1)
///   + P(ij)P(ab) 0.5 <kl||cd> t_ikac t_jlbd            (quad 2)
///   - 0.5 P(ab) <kl||cd> t_ijac t_klbd                 (quad 3)
///   - 0.5 P(ij) <kl||cd> t_ikab t_jlcd                 (quad 4)
/// ```
/// `t_new = R / D`, `D[i,j,a,b] = eps_i + eps_j - eps_a - eps_b`.
/// Energy `E = 0.25 <ij||ab> t_ijab`.
pub fn ccd_spinorbital(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &CcConfig,
) -> Result<CcResult, FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let no = active_occ(nocc_total, cfg.frozen_core)?;
    let first_occ = cfg.frozen_core;
    let nv = nbas - nocc_total; // spatial virtual

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + no]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // Fail-fast size guard, expressed as a [`MemoryPlan`].
    //
    // # Why this changed shape
    //
    // The previous guard charged ONLY `2·(2nv)⁴` — the VVVV ladder and its
    // `einsum!` intermediate. It never charged the dense AO 3-center tensor
    // `eri3_ao` (`naux·nbas²`), even though the sibling `ccsd.rs` guard DOES
    // charge exactly that term, and `eri3_ao` is live here across all four
    // `build_b` MO-block transforms below (it is the input to each). The
    // omission is not small: at `nbas=200`/`naux=800` it is 0.26 GB, and for a
    // small-virtual system it can dominate the VVVV term outright.
    //
    // Also previously uncharged and now declared: the antisymmetrized spatial
    // blocks `v_oovv` (plus its full clone into `oovv_t`), `v_oooo` and
    // `v_ovvo`, and the `d` denominator — all resident for the whole
    // amplitude loop.
    //
    // `eri3_ao` is TRANSIENT: it is dead once the `b_*` blocks are built,
    // before the `(2nv)⁴` ladder is formed. Charging it as resident would
    // over-estimate and refuse jobs that fit.
    let nv2 = 2 * nv;
    let no2 = 2 * no;
    let naux = dfbs.nbasis();
    let mut plan = MemoryPlan::resolve(
        cfg.memory_budget_bytes,
        format!("CCD (no={no}, nv={nv} spatial; {no2}o/{nv2}v spin-orbital)"),
    );
    plan.reserve(
        "eri3_ao (P|mn)",
        naux.saturating_mul(nbas).saturating_mul(nbas),
        Lifetime::Transient,
    );
    plan.reserve(
        "b_ov/b_vo/b_oo/b_vv dressed RI blocks",
        naux.saturating_mul(
            no.saturating_mul(nv)
                .saturating_mul(2)
                .saturating_add(no.saturating_pow(2))
                .saturating_add(nv.saturating_pow(2)),
        ),
        Lifetime::Resident,
    );
    plan.reserve("v_vvvv <ab||cd>", nv2.saturating_pow(4), Lifetime::Resident);
    plan.reserve(
        "g_abcd einsum! intermediate",
        nv2.saturating_pow(4),
        Lifetime::Transient,
    );
    plan.reserve(
        "v_oovv <ij||ab> + oovv_t clone",
        no2.saturating_pow(2)
            .saturating_mul(nv2.saturating_pow(2))
            .saturating_mul(2),
        Lifetime::Resident,
    );
    plan.reserve("v_oooo <ij||kl>", no2.saturating_pow(4), Lifetime::Resident);
    plan.reserve(
        "v_ovvo <ia||bj>",
        no2.saturating_pow(2).saturating_mul(nv2.saturating_pow(2)),
        Lifetime::Resident,
    );
    // `d`, `t`, the per-iteration `t.clone()` behind `t_t`, the residual `r`
    // (seeded from a `v_oovv.clone()`), and one `einsum!` output `x` live at a
    // time inside the loop — six same-shaped buffers, charged as one aggregate
    // rather than one reservation per name (they are the same shape, and
    // enumerating each temporary separately would be a second drift surface).
    plan.reserve(
        "d denominator + t/r/x amplitude working set (x6)",
        no2.saturating_pow(2)
            .saturating_mul(nv2.saturating_pow(2))
            .saturating_mul(6),
        Lifetime::Resident,
    );
    plan.check()?;

    // V^{-1/2} metric and AO 3-center integrals.
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;

    // Spatial dressed RI 3-index MO blocks.
    let b_ov = build_b(
        &transform_3center_ov(&eri3_ao, &c_occ, &c_vir),
        &v_inv_sqrt,
        Axis::O,
        Axis::V,
    );
    let b_oo = build_b(&transform_3center_oo(&eri3_ao, &c_occ), &v_inv_sqrt, Axis::O, Axis::O);
    let b_vv = build_b(&transform_3center_vv(&eri3_ao, &c_vir), &v_inv_sqrt, Axis::V, Axis::V);
    let b_vo = transpose_b(&b_ov);

    // --- Spin-orbital antisymmetrized integral blocks ---
    let v_oovv = {
        let g_iajb: ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);
        asym_oovv(&g_iajb, no, nv)
    };
    let v_vvvv = {
        let g_abcd: ArrayD<f64> = einsum!("Pab,Pcd->abcd", &b_vv, &b_vv);
        asym_same(&g_abcd, nv)
    };
    let v_oooo = {
        let g_ijkl: ArrayD<f64> = einsum!("Pij,Pkl->ijkl", &b_oo, &b_oo);
        asym_same(&g_ijkl, no)
    };
    let v_ovvo = {
        let g_kcbj: ArrayD<f64> = einsum!("Pkc,Pbj->kcbj", &b_ov, &b_vo);
        let g_kjbc: ArrayD<f64> = einsum!("Pkj,Pbc->kjbc", &b_oo, &b_vv);
        asym_ovvo(&g_kcbj, &g_kjbc, no, nv)
    };

    // Pre-wrap loop-invariant integral tensors.
    let oovv_t = Tensor::new(v_oovv.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);
    let vvvv_t = Tensor::new(v_vvvv, [Axis::V, Axis::V, Axis::V, Axis::V]);
    let oooo_t = Tensor::new(v_oooo, [Axis::O, Axis::O, Axis::O, Axis::O]);
    let ovvo_t = Tensor::new(v_ovvo, [Axis::O, Axis::V, Axis::V, Axis::O]);

    // Stage-seam RSS safety net: every block the plan declared resident is now
    // allocated, so this is the first point at which the pre-flight estimate
    // can be compared against reality. Observational only, never an error —
    // this file's guard had silently omitted `eri3_ao` entirely before the plan
    // rewrite above, which is exactly the undershoot class this net catches.
    ferric_core::memory::warn_if_rss_over(
        "CCD AO->MO transform complete",
        plan.budget_bytes(),
        1.1,
    );

    // --- Spin-orbital energies ---
    let no2 = 2 * no; // nv2 computed above for the size guard
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
    // Denominator D[i,j,a,b].
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

    // MP2 guess: t = <ij||ab> / D.
    let mut t = &v_oovv / &d;

    // DIIS on amplitudes flattened to (no2*nv2, no2*nv2).
    let dim = no2 * nv2;
    let mut diis = ferric_scf::diis::Diis::new(cfg.diis_subspace.max(1));
    let mut e_old = 0.0;

    for iter in 0..cfg.max_iter {
        let t_t = Tensor::new(t.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);

        // Energy: 0.25 <ij||ab> t_ijab.
        let e_corr: f64 = 0.25 * einsum!("ijab,ijab->", &oovv_t, &t_t);
        let d_e = (e_corr - e_old).abs();
        if iter > 0 && d_e < 1e-10 {
            // Reshape spin-orbital t (no2,no2,nv2,nv2) into Array4 stored (i,j,a,b).
            let t2 = t
                .clone()
                .into_dimensionality::<ndarray::Ix4>()
                .unwrap();
            println!(
                "spin-orbital CCD converged in {} iterations. E_corr = {:.10}",
                iter, e_corr
            );
            return Ok(CcResult { correlation_energy: e_corr, t1: None, t2 });
        }
        e_old = e_corr;

        // --- Residual R[i,j,a,b] ---
        // R0 = <ij||ab>
        let mut r = v_oovv.clone();

        // pp ladder: 0.5 <ab||cd> t_ijcd -> einsum('ijcd,abcd->ijab').
        {
            let x: ArrayD<f64> = einsum!("ijcd,abcd->ijab", &t_t, &vvvv_t);
            r = r + 0.5 * x;
        }
        // hh ladder: 0.5 <kl||ij> t_klab -> einsum('klij,klab->ijab').
        {
            let x: ArrayD<f64> = einsum!("klij,klab->ijab", &oooo_t, &t_t);
            r = r + 0.5 * x;
        }
        // ring: P(ij)P(ab) <kb||cj> t_ikac. Contract k,c -> left-free (b,j),
        // right-free (i,a): output 'bjia', then permute to 'ijab' = axes[2,1,0,3]
        // mapping (b,j,i,a)->(i,j,a,b): src order [b=0,j=1,i=2,a=3] -> want
        // [i,j,a,b] = src[2,1,3,0].
        {
            let bjia: ArrayD<f64> = einsum!("kbcj,ikac->bjia", &ovvo_t, &t_t);
            let x = permute_to_owned(bjia.view().permuted_axes(IxDyn(&[2, 1, 3, 0])));
            r = r + p_ij_ab(&x);
        }
        // quad 1: 0.25 <kl||cd> t_ijcd t_klab.
        // i1[i,j,k,l] = t_ijcd <kl||cd> ; then i1 . t_klab.
        {
            let i1: ArrayD<f64> = einsum!("ijcd,klcd->ijkl", &t_t, &oovv_t);
            let i1_t = Tensor::new(i1, [Axis::O, Axis::O, Axis::O, Axis::O]);
            let x: ArrayD<f64> = einsum!("ijkl,klab->ijab", &i1_t, &t_t);
            r = r + 0.25 * x;
        }
        // quad 2: P(ij)P(ab) 0.5 <kl||cd> t_ikac t_jlbd.
        // i2[i,a,l,d] = t_ikac <kl||cd> (contract k,c; left-free i,a; right-free l,d).
        // res[i,a,j,b] = i2[i,a,l,d] t_jlbd (contract l,d); permute iajb->ijab.
        {
            let i2: ArrayD<f64> = einsum!("ikac,klcd->iald", &t_t, &oovv_t);
            let i2_t = Tensor::new(i2, [Axis::O, Axis::V, Axis::O, Axis::V]);
            let iajb: ArrayD<f64> = einsum!("iald,jlbd->iajb", &i2_t, &t_t);
            let x = permute_to_owned(iajb.view().permuted_axes(IxDyn(&[0, 2, 1, 3])));
            r = r + 0.5 * p_ij_ab(&x);
        }
        // quad 3: -0.5 P(ab) <kl||cd> t_ijac t_klbd.
        // Y[c,b] = <kl||cd> t_klbd ; then x[i,j,a,b] = t_ijac Y[c,b].
        {
            let y: ArrayD<f64> = einsum!("klcd,klbd->cb", &oovv_t, &t_t);
            let y_t = Tensor::new(y, [Axis::V, Axis::V]);
            let x: ArrayD<f64> = einsum!("ijac,cb->ijab", &t_t, &y_t);
            r = r - 0.5 * p_ab(&x);
        }
        // quad 4: -0.5 P(ij) <kl||cd> t_ikab t_jlcd.
        // Z[k,j] = <kl||cd> t_jlcd ; then x[i,j,a,b] = t_ikab Z[k,j].
        {
            let z: ArrayD<f64> = einsum!("klcd,jlcd->kj", &oovv_t, &t_t);
            let z_t = Tensor::new(z, [Axis::O, Axis::O]);
            // contract k: left-free i,a,b; right-free j -> 'iabj'; permute to ijab.
            let iabj: ArrayD<f64> = einsum!("ikab,kj->iabj", &t_t, &z_t);
            let x = permute_to_owned(iabj.view().permuted_axes(IxDyn(&[0, 3, 1, 2])));
            r = r - 0.5 * p_ij(&x);
        }

        // Jacobi update t_new = R / D; increment is the DIIS error vector.
        let t_new = &r / &d;
        let err = &t_new - &t;

        let t_flat = t_new
            .view()
            .into_shape_with_order((dim, dim))
            .unwrap()
            .to_owned();
        let err_flat = err
            .view()
            .into_shape_with_order((dim, dim))
            .unwrap()
            .to_owned();
        let t_ext = diis.step(&t_flat, &err_flat);
        t = t_ext.into_shape_with_order(IxDyn(&[no2, no2, nv2, nv2])).unwrap();
    }

    Err(FerricError::Convergence("spin-orbital CCD did not converge".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_core::parallel::ParallelContext;

    #[test]
    fn test_ccd_h2_sto3g() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_set = basis::bundled("sto-3g").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::coulomb();
        
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf_config = RhfConfig::default();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_config).unwrap();
        
        let cc_cfg = CcConfig {
            frozen_core: 0,
            max_iter: 50,
            energy_conv: 1e-8,
            ..Default::default()
        };
        
        let result = ccd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();

        println!("CCD correlation energy: {:.10}", result.correlation_energy);
        // ccd() now delegates to the complete spin-orbital CCD. H2/STO-3G CCD
        // (= CCSD for 2 electrons) is -0.02052453; cc-pVDZ-RI aux adds ~5e-6 RI
        // error. The exact value is gated tightly by `ccd_so_h2_sto3g` below.
        assert!((result.correlation_energy - (-0.02052453)).abs() < 1e-4);
    }

    #[test]
    fn ccd_fails_fast_under_tiny_budget() {
        // M2 size guard: an explicit ~1 KB budget must ERROR before the VVVV
        // ladder allocation. Explicit config budget → no env var touched.
        // cc-pVDZ (nv2=18 → ~1.7 MB VVVV peak) so the tiny budget actually trips
        // (H2/STO-3G's VVVV is only 256 bytes — under even a 1 KB budget).
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cc_cfg = CcConfig {
            frozen_core: 0,
            memory_budget_bytes: Some(ferric_core::memory::gib_to_bytes(1e-6)),
            ..Default::default()
        };
        let err = match ccd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg) {
            Err(e) => e,
            Ok(_) => panic!("CCD should fail fast under tiny budget"),
        };
        let msg = err.to_string();
        assert!(msg.contains("CCD") && msg.contains("budget is"), "unexpected: {msg}");
        // The plan's breakdown must NAME the terms, not just report a total —
        // that breakdown is the whole reason `check_alloc` was replaced by a
        // `MemoryPlan` here (a bare "needs X GB" is what made the historical
        // incidents slow to diagnose).
        assert!(msg.contains("memory plan"), "no plan breakdown: {msg}");
        assert!(msg.contains("v_vvvv"), "breakdown must name the ladder: {msg}");
    }

    /// `eri3_ao` must be charged by the guard.
    ///
    /// The regression: this file's guard charged only the VVVV ladder and
    /// silently omitted the dense AO 3-center tensor, even though the sibling
    /// `ccsd.rs` guard charges exactly that term and `eri3_ao` is live across
    /// every `build_b` below the guard.
    ///
    /// The test does NOT re-derive the plan's arithmetic — doing so would
    /// reintroduce the second-estimator drift this pass removes. Instead it
    /// BISECTS for the smallest budget the guard accepts, then asserts that
    /// figure is larger than `eri3_ao` alone. `eri3_ao`'s size is read off the
    /// basis dimensions (`naux·nbas²·8`), which is a property of the INPUT, not
    /// a restatement of the plan.
    #[test]
    fn ccd_guard_charges_eri3_ao() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        // A deliberately LARGE aux basis: this makes eri3_ao the dominant term,
        // so a guard that omits it is caught by the bisection below.
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

        let accepts = |bytes: usize| -> bool {
            let cfg = CcConfig {
                frozen_core: 0,
                max_iter: 1,
                memory_budget_bytes: Some(bytes),
                ..Default::default()
            };
            // Only the guard's verdict matters: a budget error is a REFUSAL,
            // anything else (including a converged/unconverged energy) means
            // the guard let it through.
            match ccd(&mol, &obs, &dfbs, op, &rhf, &cfg) {
                Err(e) => !e.to_string().contains("budget is"),
                Ok(_) => true,
            }
        };

        let eri3_ao_bytes = dfbs.nbasis() * obs.nbasis() * obs.nbasis() * 8;

        // A budget of exactly `eri3_ao_bytes` cannot possibly hold `eri3_ao`
        // PLUS everything else this driver allocates, so a correct guard must
        // refuse it. The old guard, which never charged `eri3_ao` at all,
        // accepted it. One probe rather than a bisection: each probe is a full
        // CCD driver entry, so the cheap direct contradiction is preferred over
        // ~30 of them.
        assert!(
            !accepts(eri3_ao_bytes),
            "the guard accepted a budget of exactly eri3_ao's own size \
             ({eri3_ao_bytes} bytes) — that tensor is still uncharged"
        );
    }

    /// An AMPLE budget must still run. An over-estimating guard is also a bug:
    /// it refuses jobs that would have fit, and this pass ADDED terms to the
    /// estimate, so the over-count direction needs its own coverage.
    #[test]
    fn ccd_ample_budget_still_runs() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let base =
            CcConfig { frozen_core: 0, max_iter: 50, energy_conv: 1e-8, ..Default::default() };
        let unbudgeted = ccd(&mol, &obs, &dfbs, op, &rhf, &base).unwrap();
        let budgeted = ccd(
            &mol,
            &obs,
            &dfbs,
            op,
            &rhf,
            &CcConfig {
                memory_budget_bytes: Some(ferric_core::memory::gib_to_bytes(4.0)),
                ..base
            },
        )
        .expect("4 GiB must be ample for H2/cc-pVDZ CCD");
        // Guards and accounting only: the gate must not perturb the number it
        // gates, to the last bit.
        assert_eq!(
            budgeted.correlation_energy, unbudgeted.correlation_energy,
            "the memory guard changed the energy"
        );
    }

    #[test]
    fn ccd_so_h2_sto3g() {
        // ref: CCD(T1=0) = -0.02052453 (= CCSD for H2), exact-integral numpy value.
        // Use a large RI aux (def2-qzvpp-rifit) so the density-fitting error is
        // below the 1e-6 gate; cc-pvdz-ri leaves ~5e-6 RI error for STO-3G H2
        // (see ri_convergence_check_h2), which would falsely fail a correct
        // residual. The hard correctness gate is the exact reference + 1e-6 tol.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-8, ..Default::default() };
        let r = ccd_spinorbital(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        assert!((r.correlation_energy - (-0.02052453)).abs() < 1e-6, "got {:.8}", r.correlation_energy);
    }

    #[test]
    #[ignore]
    fn ri_convergence_check_h2() {
        for aux in ["cc-pvdz-ri", "def2-tzvpp-rifit", "def2-qzvpp-rifit", "aug-cc-pvtz-rifit"] {
            let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
            let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
            let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
            let op = Operator::coulomb();
            let ctx = ParallelContext::default();
            let bounds = SchwarzBounds::compute(op, &obs).unwrap();
            let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
            let cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-8, ..Default::default() };
            let r = ccd_spinorbital(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
            println!("aux={:20} E_corr={:.10} (ref -0.02052453, diff {:.2e})",
                aux, r.correlation_energy, r.correlation_energy - (-0.02052453));
        }
    }

    #[test]
    #[ignore]
    fn ri_convergence_check_h2o() {
        for aux in ["cc-pvdz-ri", "def2-tzvpp-rifit", "aug-cc-pvtz-rifit", "def2-qzvpp-rifit"] {
            let mol = Molecule::parse_xyz("3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n", 0, 1).unwrap();
            let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
            let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
            let op = Operator::coulomb();
            let ctx = ParallelContext::default();
            let bounds = SchwarzBounds::compute(op, &obs).unwrap();
            let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
            let cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-8, ..Default::default() };
            let r = ccd_spinorbital(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
            println!("aux={:20} E_corr={:.10} (ref -0.21259542, diff {:.2e})",
                aux, r.correlation_energy, r.correlation_energy - (-0.21259542));
        }
    }

    #[test]
    fn ccd_so_h2o_ccpvdz() {
        // ref: CCD(T1=0) = -0.21259542 (exact-integral value); RI vs exact -> 1e-4.
        // def2-qzvpp-rifit drives the RI error to ~3e-7 (cc-pvdz-ri leaves ~1.4e-4,
        // just over the gate — see ri_convergence_check_h2o); the residual physics
        // is exact (H2 nails the exact value at 1e-6).
        let mol = Molecule::parse_xyz("3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-8, ..Default::default() };
        let r = ccd_spinorbital(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        assert!((r.correlation_energy - (-0.21259542)).abs() < 1e-4, "got {:.8}", r.correlation_energy);
    }

    #[test]
    fn ccd_so_ch4_sto3g() {
        // Third molecule (widens past H2/H2O): CH4/STO-3G, Td symmetry.
        // ref: CCD(T1=0) = -0.07904596471422148, generated via PySCF
        // cc.CCSD with update_amps monkey-patched to zero t1 every
        // iteration (same T1=0 convention as the H2/H2O refs above).
        let mol = Molecule::parse_xyz(
            "5\nmethane Td\nC 0.0 0.0 0.0\nH 0.629118 0.629118 0.629118\n\
             H -0.629118 -0.629118 0.629118\nH -0.629118 0.629118 -0.629118\n\
             H 0.629118 -0.629118 -0.629118\n",
            0, 1,
        ).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-8, ..Default::default() };
        let r = ccd_spinorbital(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        assert!((r.correlation_energy - (-0.07904596471422148)).abs() < 1e-4, "got {:.8}", r.correlation_energy);
    }
}
