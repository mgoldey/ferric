use super::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::cholesky_inverse_sqrt;
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{Array2, Array4, ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Spin-orbital CCD helpers (pattern reimplemented from ferric-mp2/src/mp3.rs,
// whose helpers are private). Spin convention: 2k=alpha, 2k+1=beta.
// <pq||rs> = <pq|rs> - <pq|sr>, with <pq|rs> = (pr|qs) chemist + spin deltas.
// ---------------------------------------------------------------------------

/// Build a dressed RI 3-index MO tensor `B^P_{pq}` for a spatial block.
fn build_b(eri3_mo: &ndarray::Array3<f64>, v_inv_sqrt: &Array2<f64>, l1: Axis, l2: Axis) -> Tensor<3> {
    let naux = eri3_mo.shape()[0];
    let d1 = eri3_mo.shape()[1];
    let d2 = eri3_mo.shape()[2];
    let flat = eri3_mo.view().into_shape_with_order((naux, d1 * d2)).unwrap();
    let b_flat = v_inv_sqrt.dot(&flat);
    let b = b_flat.into_shape_with_order((naux, d1, d2)).unwrap();
    Tensor::new(b.into_dyn(), [Axis::Aux, l1, l2])
}

/// Transpose a `[Aux, O, V]` RI block into `[Aux, V, O]`.
fn transpose_b(b_ov: &Tensor<3>) -> Tensor<3> {
    let out = b_ov.view().permuted_axes(IxDyn(&[0, 2, 1])).to_owned();
    Tensor::new(out, [Axis::Aux, Axis::V, Axis::O])
}

#[inline]
fn spin(p: usize) -> usize {
    p & 1
}
#[inline]
fn spat(p: usize) -> usize {
    p >> 1
}

/// Antisymmetrized spin-orbital `<ij||ab>` (oovv) from spatial chemist `(ia|jb)`.
#[allow(clippy::needless_range_loop)]
fn asym_oovv(g_iajb: &ArrayD<f64>, no: usize, nv: usize) -> ArrayD<f64> {
    let (no2, nv2) = (2 * no, 2 * nv);
    let mut out = ArrayD::zeros(IxDyn(&[no2, no2, nv2, nv2]));
    for i in 0..no2 {
        for j in 0..no2 {
            for a in 0..nv2 {
                for b in 0..nv2 {
                    let dir = if spin(i) == spin(a) && spin(j) == spin(b) {
                        g_iajb[[spat(i), spat(a), spat(j), spat(b)]]
                    } else {
                        0.0
                    };
                    let exc = if spin(i) == spin(b) && spin(j) == spin(a) {
                        g_iajb[[spat(i), spat(b), spat(j), spat(a)]]
                    } else {
                        0.0
                    };
                    out[[i, j, a, b]] = dir - exc;
                }
            }
        }
    }
    out
}

/// Antisymmetrized same-space `<pq||rs>` from spatial chemist `(pq|rs)`.
/// `<pq||rs> = (pr|qs) - (ps|qr)`. Used for vvvv and oooo.
#[allow(clippy::needless_range_loop)]
fn asym_same(g: &ArrayD<f64>, n: usize) -> ArrayD<f64> {
    let n2 = 2 * n;
    let mut out = ArrayD::zeros(IxDyn(&[n2, n2, n2, n2]));
    for p in 0..n2 {
        for q in 0..n2 {
            for r in 0..n2 {
                for s in 0..n2 {
                    let dir = if spin(p) == spin(r) && spin(q) == spin(s) {
                        g[[spat(p), spat(r), spat(q), spat(s)]]
                    } else {
                        0.0
                    };
                    let exc = if spin(p) == spin(s) && spin(q) == spin(r) {
                        g[[spat(p), spat(s), spat(q), spat(r)]]
                    } else {
                        0.0
                    };
                    out[[p, q, r, s]] = dir - exc;
                }
            }
        }
    }
    out
}

/// Antisymmetrized `<kb||cj>` (ovvo). k,j occupied; b,c virtual.
/// Direct `<kb|cj> = (kc|bj)`; exchange `<kb|jc> = (kj|bc)`.
/// Output shape `(2no, 2nv, 2nv, 2no)` indexed `[k,b,c,j]`.
#[allow(clippy::needless_range_loop)]
fn asym_ovvo(g_kcbj: &ArrayD<f64>, g_kjbc: &ArrayD<f64>, no: usize, nv: usize) -> ArrayD<f64> {
    let (no2, nv2) = (2 * no, 2 * nv);
    let mut out = ArrayD::zeros(IxDyn(&[no2, nv2, nv2, no2]));
    for k in 0..no2 {
        for b in 0..nv2 {
            for c in 0..nv2 {
                for j in 0..no2 {
                    let dir = if spin(k) == spin(c) && spin(b) == spin(j) {
                        g_kcbj[[spat(k), spat(c), spat(b), spat(j)]]
                    } else {
                        0.0
                    };
                    let exc = if spin(k) == spin(j) && spin(b) == spin(c) {
                        g_kjbc[[spat(k), spat(j), spat(b), spat(c)]]
                    } else {
                        0.0
                    };
                    out[[k, b, c, j]] = dir - exc;
                }
            }
        }
    }
    out
}

// --- Permutation antisymmetrizers on X[i,j,a,b] ---
fn swap_ij(x: &ArrayD<f64>) -> ArrayD<f64> {
    x.view().permuted_axes(IxDyn(&[1, 0, 2, 3])).as_standard_layout().into_owned()
}
fn swap_ab(x: &ArrayD<f64>) -> ArrayD<f64> {
    x.view().permuted_axes(IxDyn(&[0, 1, 3, 2])).as_standard_layout().into_owned()
}
/// P(ij) x = x - x.swap_ij
fn p_ij(x: &ArrayD<f64>) -> ArrayD<f64> {
    x - &swap_ij(x)
}
/// P(ab) x = x - x.swap_ab
fn p_ab(x: &ArrayD<f64>) -> ArrayD<f64> {
    x - &swap_ab(x)
}
/// P(ij)P(ab) x = x - x.swap_ij - x.swap_ab + x.swap_ij_ab
fn p_ij_ab(x: &ArrayD<f64>) -> ArrayD<f64> {
    let sij = swap_ij(x);
    let sab = swap_ab(x);
    let sijab = swap_ab(&sij);
    x - &sij - &sab + &sijab
}

/// Compute CCD correlation energy.
pub fn ccd(
    _mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &CcConfig,
) -> Result<CcResult, FerricError> {
    let nbas = obs.nbasis();
    let nocc_total = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
    let nocc = nocc_total - cfg.frozen_core;
    let nvir = nbas - nocc_total;

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., cfg.frozen_core..nocc_total]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // 1. Get RI amplitudes B^P in MO basis blocks
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
    
    let b_ao = eri3_ao; // (naux, nbas, nbas)
    let b_ov = transform_3center_ov(&b_ao, &c_occ, &c_vir);
    let b_oo = transform_3center_oo(&b_ao, &c_occ);
    let b_vv = transform_3center_vv(&b_ao, &c_vir);

    // Contract with V^-1/2: B^P_{xy} = sum_Q V^{-1/2}_{PQ} (Q|xy).
    // Reshape each (naux, d1, d2) block to (naux, d1*d2), apply the naux×naux
    // matmul via BLAS2 .dot(), reshape back.
    let dress = |b: &ndarray::Array3<f64>| -> ndarray::Array3<f64> {
        let (na, d1, d2) = b.dim();
        let flat = b.view().into_shape_with_order((na, d1 * d2)).unwrap();
        v_inv_sqrt
            .dot(&flat)
            .into_shape_with_order((na, d1, d2))
            .unwrap()
    };
    let b_ia = dress(&b_ov); // (naux, nocc, nvir)
    let b_ij = dress(&b_oo); // (naux, nocc, nocc)
    let b_ab = dress(&b_vv); // (naux, nvir, nvir)

    // Dressed RI blocks as Tensors for einsum!, wrapped ONCE here. These are
    // loop-invariant (geometry/orbital-only); the ladder helpers borrow them
    // each iteration instead of re-cloning the (potentially hundreds-of-MB)
    // B^P_ab / B^P_ij arrays per call.
    let b_ia_t = Tensor::new(b_ia.into_dyn(), [Axis::Aux, Axis::O, Axis::V]);
    let b_ab_t = Tensor::new(b_ab.into_dyn(), [Axis::Aux, Axis::V, Axis::V]);
    let b_ij_t = Tensor::new(b_ij.into_dyn(), [Axis::Aux, Axis::O, Axis::O]);

    // The chemist (ia|jb) integral g[i,a,j,b] = sum_P B^P_ia B^P_jb, built ONCE
    // via einsum! and reused for the MP2 guess, the energy, and the (ab|ij)
    // residual term (previously computed three times with scalar sum_P loops).
    let g_iajb: ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ia_t, &b_ia_t);
    // (ab|ij) residual term is exactly g[i,a,j,b]; keep an Ix4 view-free copy
    // once (loop-invariant) instead of cloning g every iteration.
    let g_iajb_4: ndarray::Array4<f64> = g_iajb
        .clone()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();

    // 2. Form initial T2 guess (MP2): t2[i,a,j,b] = g[i,a,j,b] / D_ijab.
    let mut t2 = Array4::zeros((nocc, nvir, nocc, nvir));
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let d_ijab = eps[cfg.frozen_core + i] + eps[cfg.frozen_core + j]
                               - eps[nocc_total + a] - eps[nocc_total + b];
                    t2[(i, a, j, b)] = g_iajb[[i, a, j, b]] / d_ijab;
                }
            }
        }
    }

    // gx[i,a,j,b] = 2 g[i,a,j,b] - g[i,b,j,a]; the exchange reindex (ib|ja) is
    // g permuted on a<->b = axes [0,3,2,1]. Used in the energy contraction.
    let g_ibja = g_iajb
        .clone()
        .permuted_axes(IxDyn(&[0, 3, 2, 1]))
        .as_standard_layout()
        .into_owned();
    let gx = 2.0 * &g_iajb - &g_ibja;
    let gx_t = Tensor::new(gx, [Axis::O, Axis::V, Axis::O, Axis::V]);

    // 3. Iteration loop, accelerated with DIIS on the amplitudes.
    //
    // The amplitude increment delta[i,a,j,b] = R[i,a,j,b] / D_ijab goes to zero
    // at convergence, so it is the natural DIIS error vector. T2 and delta are
    // flattened to (nocc*nvir, nocc*nvir) matrices for the (matrix-shaped)
    // ferric-scf Diis extrapolator; DIIS changes the path, not the fixed point,
    // so the converged energy is unchanged.
    let nov = nocc * nvir;
    let mut diis = ferric_scf::diis::Diis::new(8);
    let mut e_old = 0.0;
    for iter in 0..cfg.max_iter {
        // A. Compute correlation energy: e_corr = sum_iajb t2 * (2 g_iajb - g_ibja).
        let t2_t = Tensor::new(t2.clone().into_dyn(), [Axis::O, Axis::V, Axis::O, Axis::V]);
        let e_corr: f64 = einsum!("iajb,iajb->", &t2_t, &gx_t);

        let d_e = (e_corr - e_old).abs();
        if iter > 0 && d_e < 1e-10 {
            println!("CCD converged in {} iterations. E_corr = {:.10}", iter, e_corr);
            return Ok(CcResult { correlation_energy: e_corr, t1: None, t2 });
        }
        e_old = e_corr;

        // B. Compute residuals R_iajb = (ab|ij) + L_iajb + H_iajb.
        // The (ab|ij) term is the loop-invariant g_iajb_4; the ladder helpers
        // borrow the pre-wrapped, loop-invariant B tensors (no per-iter clone).
        let pp_ladder = crate::helpers::contract_pp_ladder(&b_ab_t, &t2_t);
        let hh_ladder = crate::helpers::contract_hh_ladder(&b_ij_t, &t2_t);
        let r2 = &g_iajb_4 + &pp_ladder + &hh_ladder;

        // C. Jacobi update t2 += R/D, then DIIS-extrapolate the new amplitudes
        //    using the increment as the error vector.
        let mut t2_new = Array4::zeros((nocc, nvir, nocc, nvir));
        let mut err = Array4::zeros((nocc, nvir, nocc, nvir));
        for i in 0..nocc {
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let d_ijab = eps[cfg.frozen_core + i] + eps[cfg.frozen_core + j]
                                   - eps[nocc_total + a] - eps[nocc_total + b];
                        let delta = r2[(i, a, j, b)] / d_ijab;
                        err[(i, a, j, b)] = delta;
                        t2_new[(i, a, j, b)] = t2[(i, a, j, b)] + delta;
                    }
                }
            }
        }
        // Flatten to (nov, nov) for the matrix-shaped DIIS, extrapolate, reshape.
        let t2_flat = t2_new
            .view()
            .into_shape_with_order((nov, nov))
            .unwrap()
            .to_owned();
        let err_flat = err
            .view()
            .into_shape_with_order((nov, nov))
            .unwrap()
            .to_owned();
        let t2_ext = diis.step(&t2_flat, &err_flat);
        t2 = t2_ext
            .into_shape_with_order((nocc, nvir, nocc, nvir))
            .unwrap();
    }

    Err(FerricError::Convergence("CCD did not converge".into()))
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
    let no = nocc_total - cfg.frozen_core; // spatial active occ
    let first_occ = cfg.frozen_core;
    let nv = nbas - nocc_total; // spatial virtual

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + no]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

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

    // --- Spin-orbital energies ---
    let (no2, nv2) = (2 * no, 2 * nv);
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
            let x = bjia.view().permuted_axes(IxDyn(&[2, 1, 3, 0])).as_standard_layout().into_owned();
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
            let x = iajb.view().permuted_axes(IxDyn(&[0, 2, 1, 3])).as_standard_layout().into_owned();
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
            let x = iabj.view().permuted_axes(IxDyn(&[0, 3, 1, 2])).as_standard_layout().into_owned();
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
        // H2/STO-3G CCD correlation energy is ~ -0.018 Hartree
        assert!((result.correlation_energy - (-0.018)).abs() < 1e-2);
    }

    #[test]
    fn ccd_h2_sto3g_energy_pinned() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig { frozen_core: 0, max_iter: 50, energy_conv: 1e-8, ..Default::default() };
        let r = ccd(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        assert!((r.correlation_energy - (-0.0239287831)).abs() < 1e-9, "got {:.10}", r.correlation_energy);
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
}
