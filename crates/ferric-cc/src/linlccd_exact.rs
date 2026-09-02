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

    // The dense AO buffer dominates; count it plus the largest MO block.
    let nb2 = nbas * nbas;
    let peak = nb2
        .saturating_mul(nb2)
        .saturating_add((no2 * no2 * nv2 * nv2).saturating_mul(2))
        .saturating_mul(8);
    ferric_core::memory::check_alloc(
        &format!("exact LinLCCD {variant:?} (nbas={nbas}, no={no}, nv={nv})"),
        peak,
        ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes),
    )?;

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
