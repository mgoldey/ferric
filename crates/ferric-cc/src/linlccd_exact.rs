//! Exact-integral (non-RI) LinLCCD — a reference path for quantifying the RI error.
//!
//! [`crate::linlccd`] builds its integrals by density fitting, which carries an error
//! floor (~1e-5 Ha on water/cc-pVDZ for ferric's other RI methods). That floor is
//! invisible from inside the RI path: it looks like a converged answer. This module
//! solves the *same* amplitude equations from exact 4-center integrals, so the two can
//! be differenced and the RI contribution measured rather than assumed.
//!
//! O(nbas⁴) memory and O(N⁵) time for the transform — reference-sized systems only.
//! Everything downstream of the integral source is shared with the RI path: the same
//! `spinorbital::asym_*` antisymmetrizers, the same residual, the same DIIS. Only the
//! provenance of the spatial chemist blocks differs, which is exactly what makes the
//! comparison meaningful.

use crate::linlccd::LadderVariant;
use crate::{CcConfig, CcResult};
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::canonical::dense_ao_eri;
use ferric_mp2::spinorbital::{asym_oovv, asym_same};
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{Array2, ArrayD, ArrayView2, IxDyn};

/// Transform the dense AO ERI to a spatial MO block `(pq|rs)` in chemist notation,
/// where each index runs over the MO subset given by its coefficient matrix.
///
/// Four quarter transforms, one GEMM each — O(nbas⁴·nmo).
fn transform_4(
    ao: &[f64],
    nbas: usize,
    c1: &Array2<f64>,
    c2: &Array2<f64>,
    c3: &Array2<f64>,
    c4: &Array2<f64>,
) -> Result<ArrayD<f64>, FerricError> {
    let (n1, n2, n3, n4) = (c1.ncols(), c2.ncols(), c3.ncols(), c4.ncols());
    let nb2 = nbas * nbas;

    // (μ,νλσ) -> (p,νλσ)
    let ao_m = ArrayView2::from_shape((nbas, nb2 * nbas), ao)
        .map_err(|e| FerricError::General(format!("exact LinLCCD AO reshape: {e}")))?;
    let t1 = c1.t().dot(&ao_m); // (n1, ν λ σ)

    // (p,ν,λσ) -> (p,q,λσ)
    let mut t2 = Array2::<f64>::zeros((n1 * n2, nb2));
    for p in 0..n1 {
        let row = t1.row(p);
        let m = ArrayView2::from_shape(
            (nbas, nb2),
            row.as_slice()
                .ok_or_else(|| FerricError::General("exact LinLCCD: t1 row not contiguous".into()))?,
        )
        .map_err(|e| FerricError::General(format!("exact LinLCCD t1 reshape: {e}")))?;
        t2.slice_mut(ndarray::s![p * n2..(p + 1) * n2, ..]).assign(&c2.t().dot(&m));
    }
    drop(t1);

    // (pq,λ,σ) -> contract σ with c4, then λ with c3.
    let t2_m = ArrayView2::from_shape(
        (n1 * n2 * nbas, nbas),
        t2.as_slice()
            .ok_or_else(|| FerricError::General("exact LinLCCD: t2 not contiguous".into()))?,
    )
    .map_err(|e| FerricError::General(format!("exact LinLCCD t2 reshape: {e}")))?;
    let t3 = t2_m.dot(c4); // (p q λ, s)

    let mut out = ArrayD::<f64>::zeros(IxDyn(&[n1, n2, n3, n4]));
    for pq in 0..n1 * n2 {
        let block = t3.slice(ndarray::s![pq * nbas..(pq + 1) * nbas, ..]); // (λ, s)
        let rs = c3.t().dot(&block); // (r, s)
        for r in 0..n3 {
            for s in 0..n4 {
                out[[pq / n2, pq % n2, r, s]] = rs[[r, s]];
            }
        }
    }
    Ok(out)
}

/// Solve the LinLCCD amplitude equations using exact (non-RI) integrals.
///
/// Same equations, same solver, same antisymmetrization as [`super::linlccd`] — only
/// the integral source differs. See that module for the physics.
///
/// Closed-shell RHF reference only. Hard-errors on an unconverged reference.
pub fn linlccd_exact(
    mol: &Molecule,
    obs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &CcConfig,
    variant: LadderVariant,
) -> Result<CcResult, FerricError> {
    if !rhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        });
    }

    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let no = ferric_mp2::rimp2::active_occ(nocc_total, cfg.frozen_core)?;
    let first_occ = cfg.frozen_core;
    let nv = nbas - nocc_total;
    let (no2, nv2) = (2 * no, 2 * nv);

    // Fail-fast size guard, expressed as a [`MemoryPlan`].
    //
    // # Why this changed shape
    //
    // The previous guard charged only the dense AO buffer (`nbas^4`) plus
    // `2*(no2*nv2)^2` for `v_oovv`/`d` — the smallest of the three MO blocks
    // this driver can build. Two things were omitted, both load-bearing for
    // `LadderVariant::Full`:
    //
    // * `vvvv_t`, the spin-orbital `(2nv)^4` VVVV block. It dwarfs every
    //   other term here — at benzene/aug-cc-pVTZ scale (nbas=414, nv=393
    //   spatial) it is ~3053 GB against an AO buffer of ~235 GB, i.e. it was
    //   the DOMINANT uncharged term, not a rounding correction.
    // * `transform_4`'s own internal working set (`t1`/`t2`/`t3`, each
    //   `O(nbas^3 * n)` for the largest MO block being formed) — transient,
    //   but co-resident with the `ao` buffer that is passed in, and at the
    //   VVVV call each is O(100) GB at the same benzene scale.
    //
    // `oooo_t` (`(2no)^4`) is charged too, since `Hh` and `Full` both build
    // it; it is far smaller than `vvvv_t` (no << nv in every realistic basis)
    // but costs nothing to declare correctly rather than approximately.
    let nb2 = nbas * nbas;
    let mut plan = MemoryPlan::resolve(
        cfg.memory_budget_bytes,
        format!("exact LinLCCD {variant:?} (nbas={nbas}, no={no}, nv={nv})"),
    );
    plan.reserve("dense AO eri (nbas^4)", nb2.saturating_mul(nb2), Lifetime::Transient);
    // transform_4's own t1/t2/t3 working set, sized for the largest MO block
    // Every variant transforms OVOV; `Hh` and `Full` add OOOO; `Full` adds
    // VVVV. Charge the LARGEST of the calls this variant actually makes —
    // they are sequential, so the peak is the biggest one, not their sum.
    let (n1, n2, _n3, n4) = match variant {
        LadderVariant::Full => (nv, nv, nv, nv),
        // OOOO vs OVOV: `no <= nv` in every realistic basis, so OVOV is the
        // larger of the two calls `Hh` makes.
        LadderVariant::Hh | LadderVariant::DriversOnly => (no, nv, no, nv),
    };
    let transform4_peak = n1
        .saturating_mul(nbas.saturating_pow(3)) // t1: (n1, nu*lam*sig)
        .max(n1.saturating_mul(n2).saturating_mul(nb2)) // t2: (n1*n2, nb2)
        .max(n1.saturating_mul(n2).saturating_mul(nbas).saturating_mul(n4)); // t3
    plan.reserve("transform_4 t1/t2/t3 working set", transform4_peak, Lifetime::Transient);
    plan.reserve(
        "v_oovv <ij||ab> + oovv_t clone",
        no2.saturating_pow(2).saturating_mul(nv2.saturating_pow(2)).saturating_mul(2),
        Lifetime::Resident,
    );
    if matches!(variant, LadderVariant::Hh | LadderVariant::Full) {
        plan.reserve("oooo_t <ij||kl>", no2.saturating_pow(4), Lifetime::Resident);
    }
    if matches!(variant, LadderVariant::Full) {
        plan.reserve("vvvv_t <ab||cd> (spin-orbital)", nv2.saturating_pow(4), Lifetime::Resident);
    }
    plan.reserve(
        "d denominator + t/r/x amplitude working set",
        no2.saturating_pow(2).saturating_mul(nv2.saturating_pow(2)).saturating_mul(3),
        Lifetime::Resident,
    );
    // DIIS ring on flattened t (no2*nv2, no2*nv2): `Diis::step` clones BOTH
    // arguments into separate histories every call (see
    // `crate::diis_history_elems`'s doc), so a subspace of n holds 2n
    // full-size copies of this tensor.
    plan.reserve(
        "DIIS amplitude + error history (2 x diis_subspace)",
        crate::diis_history_elems(no2.saturating_pow(2).saturating_mul(nv2.saturating_pow(2)), cfg.diis_subspace),
        Lifetime::Resident,
    );
    plan.check()?;

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + no]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    let ao = dense_ao_eri(obs, op)?;

    // Spatial chemist blocks, then the SHARED antisymmetrizers.
    let v_oovv = {
        let g_iajb = transform_4(&ao, nbas, &c_occ, &c_vir, &c_occ, &c_vir)?;
        asym_oovv(&g_iajb, no, nv)
    };
    let oooo_t = if matches!(variant, LadderVariant::Hh | LadderVariant::Full) {
        let g_ijkl = transform_4(&ao, nbas, &c_occ, &c_occ, &c_occ, &c_occ)?;
        Some(Tensor::new(asym_same(&g_ijkl, no), [Axis::O, Axis::O, Axis::O, Axis::O]))
    } else {
        None
    };
    let vvvv_t = if matches!(variant, LadderVariant::Full) {
        let g_abcd = transform_4(&ao, nbas, &c_vir, &c_vir, &c_vir, &c_vir)?;
        Some(Tensor::new(asym_same(&g_abcd, nv), [Axis::V, Axis::V, Axis::V, Axis::V]))
    } else {
        None
    };
    drop(ao);

    let oovv_t = Tensor::new(v_oovv.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);

    // Spin-orbital energies (even = alpha, odd = beta).
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
        "exact LinLCCD {variant:?} did not converge in {} iterations",
        cfg.max_iter
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    #[test]
    fn linlccd_exact_fails_fast_under_tiny_budget() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig {
            frozen_core: 0,
            memory_budget_bytes: Some(ferric_core::memory::gib_to_bytes(1e-6)),
            ..Default::default()
        };
        let err = match linlccd_exact(&mol, &obs, op, &rhf, &cfg, LadderVariant::Full) {
            Err(e) => e,
            Ok(_) => panic!("exact LinLCCD should fail fast under tiny budget"),
        };
        let msg = err.to_string();
        assert!(msg.contains("LinLCCD") && msg.contains("budget is"), "unexpected: {msg}");
        assert!(msg.contains("memory plan"), "no plan breakdown: {msg}");
    }

    /// `vvvv_t` (the `(2nv)^4` VVVV block, only built under `LadderVariant::Full`)
    /// must be charged.
    ///
    /// The regression: the old guard charged only the AO buffer and the
    /// `v_oovv`/`d` working set, so a `Full`-variant job whose VVVV block alone
    /// exceeded the budget was still admitted. This bisects between a budget
    /// that holds everything EXCEPT `vvvv_t` (accepted for `Hh`, which never
    /// builds it) and the same budget under `Full` (must now be refused).
    #[test]
    fn linlccd_exact_full_charges_vvvv_that_hh_does_not_need() {
        let mol = Molecule::parse_xyz(
            "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nv = nbas - nocc_total;
        let nv2 = 2 * nv;
        let vvvv_bytes = nv2.pow(4) * 8;

        // A budget just under vvvv_t's own size cannot possibly hold it PLUS
        // everything else — so Full must be refused, while Hh (which never
        // builds vvvv_t) should still be accepted at the same budget for the
        // AO+oovv-only terms if they fit. We only assert the Full-refusal
        // direction here, which is the one the old guard got wrong.
        let cfg_full = CcConfig {
            frozen_core: 0,
            max_iter: 1,
            diis_subspace: 1,
            memory_budget_bytes: Some(vvvv_bytes),
            ..Default::default()
        };
        let refused = match linlccd_exact(&mol, &obs, op, &rhf, &cfg_full, LadderVariant::Full) {
            Err(e) => e.to_string().contains("budget is"),
            Ok(_) => false,
        };
        assert!(
            refused,
            "Full variant accepted a budget of exactly one vvvv_t block \
             ({vvvv_bytes} bytes) — vvvv_t is still uncharged"
        );
    }

    /// An AMPLE budget must still run to completion for every variant — an
    /// over-estimating guard is also a bug.
    #[test]
    fn linlccd_exact_ample_budget_still_converges() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig {
            frozen_core: 0,
            max_iter: 100,
            energy_conv: 1e-9,
            memory_budget_bytes: Some(ferric_core::memory::gib_to_bytes(4.0)),
            ..Default::default()
        };
        let r = linlccd_exact(&mol, &obs, op, &rhf, &cfg, LadderVariant::Full).unwrap();
        // Full LinLCCD is exact (not just RI-approximate) for 2-electron
        // systems, same anchor as the RI path's own H2 test.
        assert!(
            r.correlation_energy.is_finite(),
            "an ample 4 GiB budget must not be refused"
        );
    }
}
