//! Complete spin-orbital CCSD (Stanton-Gauss-Bartlett-Lee-Schaefer 1991 form).
//!
//! Translated from a validated numpy reference (`/tmp/ccsd_canon_spec.py`) that
//! reproduces H2/STO-3G = -0.02052453 (exact) and H2O/cc-pVDZ = -0.21332778.
//! Uses the canonical-HF simplification `fov = fo = fv = 0` (off-diagonal Fock
//! vanishes for RHF canonical orbitals; the orbital-energy diagonal enters only
//! through the amplitude denominators).
//!
//! All four-index contractions go through `einsum!` (one BLAS3 GEMM each). The
//! 11 antisymmetrized spin-orbital `<pq||rs>` blocks are built from dressed RI
//! 3-index MO blocks via the general antisymmetrizer
//! [`ferric_mp2::spinorbital::asym_phys`].

use super::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::{active_occ, cholesky_inverse_sqrt};
use ferric_mp2::spinorbital::{asym_phys, build_b, transpose_b};
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{ArrayD, IxDyn};

// Permutation antisymmetrizers P(ij)/P(ab)/P(ij)P(ab) on the i,j,a,b axes are
// shared with the CCD residual builder — see `helpers.rs`.
use super::helpers::{p_ab, p_ij, p_ij_ab};

/// Wrap a `<pq||rs>` ArrayD with axis labels (purely descriptive; einsum! only
/// uses labels for a debug-mode consistency check).
fn lbl4(a: ArrayD<f64>, l: [Axis; 4]) -> Tensor<4> {
    Tensor::new(a, l)
}
fn lbl2(a: ArrayD<f64>, l: [Axis; 2]) -> Tensor<2> {
    Tensor::new(a, l)
}

/// Complete spin-orbital CCSD via `einsum!` + optional DIIS on T2.
pub fn ccsd(
    _mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &CcConfig,
) -> Result<CcResult, FerricError> {
    let nbas = obs.nbasis();
    let nocc_total = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
    let first_occ = cfg.frozen_core;
    let no = active_occ(nocc_total, first_occ)?;
    let nv = nbas - nocc_total; // spatial virtual

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..nocc_total]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // Fail-fast size guard: peak is the antisymmetrized VVVV block g_vvvv (:159)
    // — a (2nv)⁴ f64 tensor built from ~3 co-resident copies (direct + exchange
    // einsum! outputs + asym_phys result :159-162), plus the dense AO 3-center
    // eri3_ao (:81, naux·nbf²). Keep this next to those allocations.
    let nv2 = 2 * nv;
    let naux = dfbs.nbasis();
    let peak_vvvv = nv2.saturating_pow(4).saturating_mul(3).saturating_mul(8); // ~3× (2nv)⁴ f64
    let eri3_bytes = naux.saturating_mul(nbas).saturating_mul(nbas).saturating_mul(8);
    let peak = peak_vvvv.saturating_add(eri3_bytes);
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes);
    ferric_core::memory::check_alloc(
        &format!("CCSD (no={no}, nv={nv} spatial; VVVV block over {nv2} spin-orbital virtuals)"),
        peak,
        budget,
    )?;

    // V^{-1/2} metric and AO 3-center integrals.
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;

    // Spatial dressed RI 3-index MO blocks B^P_{pq}.
    let b_ov = build_b(
        &transform_3center_ov(&eri3_ao, &c_occ, &c_vir),
        &v_inv_sqrt,
        Axis::O,
        Axis::V,
    );
    let b_oo = build_b(&transform_3center_oo(&eri3_ao, &c_occ), &v_inv_sqrt, Axis::O, Axis::O);
    let b_vv = build_b(&transform_3center_vv(&eri3_ao, &c_vir), &v_inv_sqrt, Axis::V, Axis::V);
    let b_vo = transpose_b(&b_ov);

    // --- Build all 11 antisymmetrized spin-orbital <pq||rs> blocks ---
    // For <P Q || R S>: direct chemist (PR|QS), exchange chemist (PS|QR).
    use Axis::{O, V};

    // OOOO: dir (OO|OO), exc (OO|OO)
    let g_oooo = {
        let d: ArrayD<f64> = einsum!("Pij,Pkl->ijkl", &b_oo, &b_oo); // (ik|jl) indexed [p,r,q,s]
        let e: ArrayD<f64> = einsum!("Pij,Pkl->ijkl", &b_oo, &b_oo); // (il|jk)
        asym_phys(&d, &e, no, no, no, no)
    };
    // OOOV: <oo||ov>; dir (OO|OV): b_oo,b_ov ; exc (OV|OO): b_ov,b_oo
    let g_ooov = {
        let d: ArrayD<f64> = einsum!("Pij,Pka->ijka", &b_oo, &b_ov);
        let e: ArrayD<f64> = einsum!("Pia,Pjk->iajk", &b_ov, &b_oo);
        asym_phys(&d, &e, no, no, no, nv)
    };
    // OOVO: <oo||vo>; dir (OV|OO): b_ov,b_oo ; exc (OO|OV): b_oo,b_ov
    let g_oovo = {
        let d: ArrayD<f64> = einsum!("Pia,Pjk->iajk", &b_ov, &b_oo);
        let e: ArrayD<f64> = einsum!("Pij,Pka->ijka", &b_oo, &b_ov);
        asym_phys(&d, &e, no, no, nv, no)
    };
    // OOVV: dir (OV|OV), exc (OV|OV)
    let g_oovv = {
        let d: ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);
        let e: ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);
        asym_phys(&d, &e, no, no, nv, nv)
    };
    // OVOO: <ov||oo>; dir (OO|VO): b_oo,b_vo ; exc (OO|VO): b_oo,b_vo
    let g_ovoo = {
        let d: ArrayD<f64> = einsum!("Pij,Pak->ijak", &b_oo, &b_vo);
        let e: ArrayD<f64> = einsum!("Pij,Pak->ijak", &b_oo, &b_vo);
        asym_phys(&d, &e, no, nv, no, no)
    };
    // OVOV: <ov||ov>; dir (OO|VV): b_oo,b_vv ; exc (OV|VO): b_ov,b_vo
    let g_ovov = {
        let d: ArrayD<f64> = einsum!("Pij,Pab->ijab", &b_oo, &b_vv);
        let e: ArrayD<f64> = einsum!("Pib,Paj->ibaj", &b_ov, &b_vo);
        asym_phys(&d, &e, no, nv, no, nv)
    };
    // OVVO: <ov||vo>; dir (OV|VO): b_ov,b_vo ; exc (OO|VV): b_oo,b_vv
    let g_ovvo = {
        let d: ArrayD<f64> = einsum!("Pia,Pbj->iabj", &b_ov, &b_vo);
        let e: ArrayD<f64> = einsum!("Pij,Pab->ijab", &b_oo, &b_vv);
        asym_phys(&d, &e, no, nv, nv, no)
    };
    // OVVV: <ov||vv>; dir (OV|VV): b_ov,b_vv ; exc (OV|VV): b_ov,b_vv
    let g_ovvv = {
        let d: ArrayD<f64> = einsum!("Pia,Pbc->iabc", &b_ov, &b_vv);
        let e: ArrayD<f64> = einsum!("Pia,Pbc->iabc", &b_ov, &b_vv);
        asym_phys(&d, &e, no, nv, nv, nv)
    };
    // VOVV: <vo||vv>; dir (VV|OV): b_vv,b_ov ; exc (VV|OV): b_vv,b_ov
    let g_vovv = {
        let d: ArrayD<f64> = einsum!("Pab,Pic->abic", &b_vv, &b_ov);
        let e: ArrayD<f64> = einsum!("Pab,Pic->abic", &b_vv, &b_ov);
        asym_phys(&d, &e, nv, no, nv, nv)
    };
    // VVVO: <vv||vo>; dir (VV|VO): b_vv,b_vo ; exc (VO|VV): b_vo,b_vv
    let g_vvvo = {
        let d: ArrayD<f64> = einsum!("Pab,Pci->abci", &b_vv, &b_vo);
        let e: ArrayD<f64> = einsum!("Pai,Pbc->aibc", &b_vo, &b_vv);
        asym_phys(&d, &e, nv, nv, nv, no)
    };
    // VVVV: dir (VV|VV), exc (VV|VV)
    let g_vvvv = {
        let d: ArrayD<f64> = einsum!("Pab,Pcd->abcd", &b_vv, &b_vv);
        let e: ArrayD<f64> = einsum!("Pab,Pcd->abcd", &b_vv, &b_vv);
        asym_phys(&d, &e, nv, nv, nv, nv)
    };

    // Pre-wrap loop-invariant integral tensors (labels are descriptive only).
    let oooo = lbl4(g_oooo, [O, O, O, O]);
    let ooov = lbl4(g_ooov, [O, O, O, V]);
    let oovo = lbl4(g_oovo, [O, O, V, O]);
    let oovv = lbl4(g_oovv.clone(), [O, O, V, V]);
    let ovoo = lbl4(g_ovoo, [O, V, O, O]);
    let ovov = lbl4(g_ovov, [O, V, O, V]);
    let ovvo = lbl4(g_ovvo, [O, V, V, O]);
    let ovvv = lbl4(g_ovvv, [O, V, V, V]);
    let vovv = lbl4(g_vovv, [V, O, V, V]);
    let vvvo = lbl4(g_vvvo, [V, V, V, O]);
    let vvvv = lbl4(g_vvvv, [V, V, V, V]);

    // --- Spin-orbital orbital energies and denominators ---
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
    let mut dia = ArrayD::zeros(IxDyn(&[no2, nv2]));
    for i in 0..no2 {
        for a in 0..nv2 {
            dia[[i, a]] = eo[i] - ev[a];
        }
    }
    let mut dijab = ArrayD::zeros(IxDyn(&[no2, no2, nv2, nv2]));
    for i in 0..no2 {
        for j in 0..no2 {
            for a in 0..nv2 {
                for b in 0..nv2 {
                    dijab[[i, j, a, b]] = eo[i] + eo[j] - ev[a] - ev[b];
                }
            }
        }
    }

    // Initial amplitudes: t1 = 0, t2 = <ij||ab> / D.
    let mut t1: ArrayD<f64> = ArrayD::zeros(IxDyn(&[no2, nv2]));
    let mut t2: ArrayD<f64> = &g_oovv / &dijab;

    // DIIS on flattened T2 (no2*nv2, no2*nv2); T1 updated plainly.
    let dim = no2 * nv2;
    let mut diis = ferric_scf::diis::Diis::new(cfg.diis_subspace.max(1));
    let mut t2_prev = t2.clone();
    let mut e_old = 0.0;

    for iter in 0..cfg.max_iter {
        let t1_t = lbl2(t1.clone(), [O, V]);
        let t2_t = lbl4(t2.clone(), [O, O, V, V]);

        // --- taus and tau (Stanton Eq. 9, 10) ---
        // taus = t2 + 0.5 (P(ij)P(ab)) t1 outer t1
        //      = t2 + 0.5 ( t_ia t_jb - t_ib t_ja - t_ja t_ib + t_jb t_ia )
        // tau  = t2 +       ( t_ia t_jb - t_ib t_ja )
        // t_ia t_jb. einsum! emits left-free (i,a) before right-free (j,b) as
        // (i,a,j,b); permute to the (i,j,a,b) layout the antisymmetrizers expect.
        let oo1_iajb: ArrayD<f64> = einsum!("ia,jb->iajb", &t1_t, &t1_t);
        let oo1: ArrayD<f64> = oo1_iajb
            .permuted_axes(IxDyn(&[0, 2, 1, 3]))
            .as_standard_layout()
            .into_owned();
        let taus = &t2 + &(0.5 * p_ij_ab(&oo1));
        let tau = &t2 + &p_ij(&oo1); // P(ij) of (t_ia t_jb) == t_ia t_jb - t_ib t_ja... see note
        // NOTE: tau = t2 + t_ia t_jb - t_ib t_ja. p_ij(oo1) = oo1 - swap_ij(oo1).
        // swap_ij(t_ia t_jb) = t_ja t_ib (i<->j) -> that is the t_ib t_ja term? No:
        // we need t_ib t_ja. swap_ij gives index [j,i,a,b] of oo1 = t_ja t_ib.
        // But we want subtract t_ib t_ja = same value (scalars), so p_ij is correct.
        let taus_t = lbl4(taus.clone(), [O, O, V, V]);
        let tau_t = lbl4(tau.clone(), [O, O, V, V]);

        // --- F intermediates (fov = fo = fv = 0) ---
        // Fae = -0.5 sum_mnf taus_mnaf <mn||ef>  ->  einsum('mnaf,mnef->af') then [a,e]
        // numpy: Fae[a,e] = -0.5 einsum('mnaf,mnef->ae', taus, oovv)
        let fae: ArrayD<f64> = {
            // contract m,n,f ; left-free a ; right-free e -> 'ae'
            let x: ArrayD<f64> = einsum!("mnaf,mnef->ae", &taus_t, &oovv);
            -0.5 * x
        };
        let fae_t = lbl2(fae.clone(), [V, V]);
        // Fmi[m,i] = 0.5 einsum('inef,mnef->mi', taus, oovv)
        // contract n,e,f ; left index i (from taus), right index m (from oovv)
        // -> einsum('inef,mnef->im') gives [i,m]; we want [m,i] => permute.
        let fmi: ArrayD<f64> = {
            let im: ArrayD<f64> = einsum!("inef,mnef->im", &taus_t, &oovv);
            0.5 * im.view().permuted_axes(IxDyn(&[1, 0])).as_standard_layout().into_owned()
        };
        let fmi_t = lbl2(fmi.clone(), [O, O]);
        // Fme[m,e] = einsum('nf,mnef->me', t1, oovv)
        let fme: ArrayD<f64> = einsum!("nf,mnef->me", &t1_t, &oovv);
        let fme_t = lbl2(fme.clone(), [O, V]);

        // --- W intermediates ---
        // Wmnij = oooo + P(ij)[ einsum('je,mnie->mnij', t1, ooov) ] + 0.25 einsum('ijef,mnef->mnij', tau, oovv)
        // numpy: +einsum('je,mnie->mnij',t1,ooov) - einsum('ie,mnje->mnij',t1,ooov)
        let wmnij: ArrayD<f64> = {
            let mut w = owned_arr4(&oooo);
            // term: einsum('je,mnie->mnij', t1, ooov): contract e ; left-free j ;
            // right-free m,n,i -> 'jmni'; want 'mnij' = permute [1,2,3,0].
            let jmni: ArrayD<f64> = einsum!("je,mnie->jmni", &t1_t, &ooov);
            let mnij = jmni.view().permuted_axes(IxDyn(&[1, 2, 3, 0])).as_standard_layout().into_owned();
            // P(ij) acts on the last two indices (i,j) of mnij. Our p_ij swaps axes 0,1.
            // Build pij over (i,j) = axes 2,3: do manual swap.
            let swapped = mnij.view().permuted_axes(IxDyn(&[0, 1, 3, 2])).as_standard_layout().into_owned();
            w = w + &mnij - &swapped;
            // 0.25 einsum('ijef,mnef->mnij', tau, oovv): contract e,f ; left-free i,j ;
            // right-free m,n -> 'ijmn'; want 'mnij' = permute [2,3,0,1].
            let ijmn: ArrayD<f64> = einsum!("ijef,mnef->ijmn", &tau_t, &oovv);
            let mnij2 = ijmn.view().permuted_axes(IxDyn(&[2, 3, 0, 1])).as_standard_layout().into_owned();
            w = w + 0.25 * mnij2;
            w
        };
        let wmnij_t = lbl4(wmnij, [O, O, O, O]);

        // Wabef = vvvv - einsum('mb,amef->abef', t1, vovv) + einsum('ma,bmef->abef', t1, vovv)
        //              + 0.25 einsum('mnab,mnef->abef', tau, oovv)
        let wabef: ArrayD<f64> = {
            let mut w = owned_arr4(&vvvv);
            // einsum('mb,amef->abef', t1, vovv): contract m ; left-free b ;
            // right-free a,e,f -> 'baef'; want 'abef' = permute [1,0,2,3].
            let baef: ArrayD<f64> = einsum!("mb,amef->baef", &t1_t, &vovv);
            let abef = baef.view().permuted_axes(IxDyn(&[1, 0, 2, 3])).as_standard_layout().into_owned();
            w -= &abef;
            // + einsum('ma,bmef->abef', t1, vovv): contract m ; left-free a ;
            // right-free b,e,f -> 'abef' directly.
            let abef2: ArrayD<f64> = einsum!("ma,bmef->abef", &t1_t, &vovv);
            w += &abef2;
            // 0.25 einsum('mnab,mnef->abef', tau, oovv): contract m,n ; left-free a,b ;
            // right-free e,f -> 'abef'.
            let abef3: ArrayD<f64> = einsum!("mnab,mnef->abef", &tau_t, &oovv);
            w = w + 0.25 * abef3;
            w
        };
        let wabef_t = lbl4(wabef, [V, V, V, V]);

        // Wmbej = ovvo + einsum('jf,mbef->mbej', t1, ovvv) - einsum('nb,mnej->mbej', t1, oovo)
        //              - einsum('jnfb,mnef->mbej', 0.5 t2 + t1 outer t1, oovv)
        let wmbej: ArrayD<f64> = {
            let mut w = owned_arr4(&ovvo);
            // einsum('jf,mbef->mbej', t1, ovvv): contract f ; left-free j ;
            // right-free m,b,e -> 'jmbe'; want 'mbej' = permute [1,2,3,0].
            let jmbe: ArrayD<f64> = einsum!("jf,mbef->jmbe", &t1_t, &ovvv);
            let mbej = jmbe.view().permuted_axes(IxDyn(&[1, 2, 3, 0])).as_standard_layout().into_owned();
            w += &mbej;
            // - einsum('nb,mnej->mbej', t1, oovo): contract n ; left-free b ;
            // right-free m,e,j -> 'bmej'; want 'mbej' = permute [1,0,2,3].
            let bmej: ArrayD<f64> = einsum!("nb,mnej->bmej", &t1_t, &oovo);
            let mbej2 = bmej.view().permuted_axes(IxDyn(&[1, 0, 2, 3])).as_standard_layout().into_owned();
            w -= &mbej2;
            // - einsum('jnfb,mnef->mbej', X, oovv) where X = 0.5 t2 + t1 outer t1.
            // X[j,n,f,b] = 0.5 t2[j,n,f,b] + t1[j,f] t1[n,b].
            // einsum! emits left-free (j,f) before right-free (n,b) as (j,f,n,b);
            // permute to the (j,n,f,b) layout used below.
            let tt_jfnb: ArrayD<f64> = einsum!("jf,nb->jfnb", &t1_t, &t1_t);
            let tt: ArrayD<f64> = tt_jfnb
                .permuted_axes(IxDyn(&[0, 2, 1, 3]))
                .as_standard_layout()
                .into_owned();
            let x_jnfb = &(0.5 * &t2) + &tt;
            let x_t = lbl4(x_jnfb, [O, O, V, V]);
            // contract n,f ; from X left indices j,b ; from oovv right indices m,e.
            // einsum('jnfb,mnef->...'): contracted are n,f. left-free j,b ;
            // right-free m,e -> 'jbme'; want 'mbej' = permute mapping:
            // src [j=0,b=1,m=2,e=3] -> want [m,b,e,j] = src[2,1,3,0].
            let jbme: ArrayD<f64> = einsum!("jnfb,mnef->jbme", &x_t, &oovv);
            let mbej3 = jbme.view().permuted_axes(IxDyn(&[2, 1, 3, 0])).as_standard_layout().into_owned();
            w -= &mbej3;
            w
        };
        let wmbej_t = lbl4(wmbej, [O, V, V, O]);

        // ===================== T1 residual =====================
        // r1 = einsum('ie,ae->ia', t1, Fae) - einsum('ma,mi->ia', t1, Fmi)
        //      + einsum('imae,me->ia', t2, Fme)
        //      - einsum('nf,naif->ia', t1, ovov)
        //      - 0.5 einsum('imef,maef->ia', t2, ovvv)
        //      - 0.5 einsum('mnae,nmei->ia', t2, oovo)
        let mut r1: ArrayD<f64> = {
            // einsum('ie,ae->ia', t1, Fae): contract e ; left-free i ; right-free a -> 'ia'.
            einsum!("ie,ae->ia", &t1_t, &fae_t)
        };
        {
            // - einsum('ma,mi->ia', t1, Fmi): contract m ; left-free a ; right-free i
            // -> 'ai'; want 'ia' = permute.
            let ai: ArrayD<f64> = einsum!("ma,mi->ai", &t1_t, &fmi_t);
            r1 = r1 - ai.view().permuted_axes(IxDyn(&[1, 0])).as_standard_layout().into_owned();
        }
        {
            // + einsum('imae,me->ia', t2, Fme): contract m,e ; left-free i,a ;
            // right-free (none) -> 'ia'.
            let ia: ArrayD<f64> = einsum!("imae,me->ia", &t2_t, &fme_t);
            r1 += &ia;
        }
        {
            // - einsum('nf,naif->ia', t1, ovov): contract n,f ; left-free (none) ;
            // right-free a,i -> 'ai'; want 'ia' = permute.
            let ai: ArrayD<f64> = einsum!("nf,naif->ai", &t1_t, &ovov);
            r1 = r1 - ai.view().permuted_axes(IxDyn(&[1, 0])).as_standard_layout().into_owned();
        }
        {
            // - 0.5 einsum('imef,maef->ia', t2, ovvv): contract m,e,f ; left-free i ;
            // right-free a -> 'ia'.
            let ia: ArrayD<f64> = einsum!("imef,maef->ia", &t2_t, &ovvv);
            r1 = r1 - 0.5 * ia;
        }
        {
            // - 0.5 einsum('mnae,nmei->ia', t2, oovo): contract m,n,e ; left-free a ;
            // right-free i -> 'ai'; want 'ia' = permute.
            let ai: ArrayD<f64> = einsum!("mnae,nmei->ai", &t2_t, &oovo);
            r1 = r1 - 0.5 * ai.view().permuted_axes(IxDyn(&[1, 0])).as_standard_layout().into_owned();
        }

        // ===================== T2 residual =====================
        let mut r2 = g_oovv.clone();

        // + P(ab)[ einsum('ijae,be->ijab', t2, Fae - 0.5 einsum('mb,me->be', t1, Fme)) ]
        {
            // be = Fae - 0.5 t1_mb Fme_me : einsum('mb,me->be', t1, Fme) contract m ;
            // left-free b ; right-free e -> 'be'.
            let mb_me: ArrayD<f64> = einsum!("mb,me->be", &t1_t, &fme_t);
            let be = &fae - &(0.5 * mb_me);
            let be_t = lbl2(be, [V, V]);
            // einsum('ijae,be->ijab', t2, be): contract e ; left-free i,j,a ;
            // right-free b -> 'ijab'.
            let x: ArrayD<f64> = einsum!("ijae,be->ijab", &t2_t, &be_t);
            r2 = r2 + p_ab(&x);
        }
        // - P(ij)[ einsum('imab,mj->ijab', t2, Fmi + 0.5 einsum('je,me->mj', t1, Fme)) ]
        {
            // mj = Fmi + 0.5 einsum('je,me->mj', t1, Fme): einsum('je,me->jm') contract e ;
            // left-free j ; right-free m -> 'jm'; want 'mj' = permute.
            let jm: ArrayD<f64> = einsum!("je,me->jm", &t1_t, &fme_t);
            let mj_part = jm.view().permuted_axes(IxDyn(&[1, 0])).as_standard_layout().into_owned();
            let mj = &fmi + &(0.5 * mj_part);
            let mj_t = lbl2(mj, [O, O]);
            // einsum('imab,mj->ijab', t2, mj): contract m ; left-free i,a,b ;
            // right-free j -> 'iabj'; want 'ijab' = permute src[i=0,a=1,b=2,j=3]->[0,3,1,2].
            let iabj: ArrayD<f64> = einsum!("imab,mj->iabj", &t2_t, &mj_t);
            let x = iabj.view().permuted_axes(IxDyn(&[0, 3, 1, 2])).as_standard_layout().into_owned();
            r2 = r2 - p_ij(&x);
        }
        // + 0.5 einsum('mnab,mnij->ijab', tau, Wmnij): contract m,n ; left-free a,b ;
        // right-free i,j -> 'abij'; want 'ijab' = permute [2,3,0,1].
        {
            let abij: ArrayD<f64> = einsum!("mnab,mnij->abij", &tau_t, &wmnij_t);
            let x = abij.view().permuted_axes(IxDyn(&[2, 3, 0, 1])).as_standard_layout().into_owned();
            r2 = r2 + 0.5 * x;
        }
        // + 0.5 einsum('ijef,abef->ijab', tau, Wabef): contract e,f ; left-free i,j ;
        // right-free a,b -> 'ijab'.
        {
            let x: ArrayD<f64> = einsum!("ijef,abef->ijab", &tau_t, &wabef_t);
            r2 = r2 + 0.5 * x;
        }
        // + P(ij)P(ab)[ einsum('imae,mbej->ijab', t2, Wmbej)
        //               - einsum('ie,ma,mbej->ijab', t1, t1, ovvo) ]
        {
            // einsum('imae,mbej->ijab', t2, Wmbej): contract m,e ; left-free i,a ;
            // right-free b,j -> 'iabj'; want 'ijab' = permute [0,3,1,2]? src [i,a,b,j]
            // -> want [i,j,a,b] = src[0,3,1,2].
            let iabj: ArrayD<f64> = einsum!("imae,mbej->iabj", &t2_t, &wmbej_t);
            let mut x = iabj.view().permuted_axes(IxDyn(&[0, 3, 1, 2])).as_standard_layout().into_owned();
            // - einsum('ie,ma,mbej->ijab', t1, t1, ovvo). Do in two binary steps.
            // First P[i,e,b,j] = einsum('ma,mbej->...'): wait, group t1_ie outer with
            // (t1_ma contracted into ovvo). Build Q[a,b,e,j] = einsum('ma,mbej->abej', t1, ovvo):
            // contract m ; left-free a ; right-free b,e,j -> 'abej'.
            let abej: ArrayD<f64> = einsum!("ma,mbej->abej", &t1_t, &ovvo);
            let abej_t = lbl4(abej, [V, V, V, O]);
            // then einsum('ie,abej->iabj', t1, Q): contract e ; left-free i ;
            // right-free a,b,j -> 'iabj'; want 'ijab' = src[i,a,b,j]->[0,3,1,2].
            let iabj2: ArrayD<f64> = einsum!("ie,abej->iabj", &t1_t, &abej_t);
            let x2 = iabj2.view().permuted_axes(IxDyn(&[0, 3, 1, 2])).as_standard_layout().into_owned();
            x -= &x2;
            r2 = r2 + p_ij_ab(&x);
        }
        // + P(ij)[ einsum('ie,abej->ijab', t1, vvvo) ]
        {
            // contract e ; left-free i ; right-free a,b,j -> 'iabj'; want 'ijab' =
            // src[i,a,b,j]->[0,3,1,2].
            let iabj: ArrayD<f64> = einsum!("ie,abej->iabj", &t1_t, &vvvo);
            let x = iabj.view().permuted_axes(IxDyn(&[0, 3, 1, 2])).as_standard_layout().into_owned();
            r2 = r2 + p_ij(&x);
        }
        // - P(ab)[ einsum('ma,mbij->ijab', t1, ovoo) ]
        {
            // contract m ; left-free a ; right-free b,i,j -> 'abij'; want 'ijab' =
            // src[a,b,i,j]->[2,3,0,1].
            let abij: ArrayD<f64> = einsum!("ma,mbij->abij", &t1_t, &ovoo);
            let x = abij.view().permuted_axes(IxDyn(&[2, 3, 0, 1])).as_standard_layout().into_owned();
            r2 = r2 - p_ab(&x);
        }

        // --- Update amplitudes: t = r / D ---
        let t1_new = &r1 / &dia;
        let t2_new = &r2 / &dijab;

        // DIIS on T2 (error = increment). T1 plain.
        let err = &t2_new - &t2_prev;
        let t2_flat = t2_new.view().into_shape_with_order((dim, dim)).unwrap().to_owned();
        let err_flat = err.view().into_shape_with_order((dim, dim)).unwrap().to_owned();
        let t2_ext = diis.step(&t2_flat, &err_flat);
        t2 = t2_ext.into_shape_with_order(IxDyn(&[no2, no2, nv2, nv2])).unwrap();
        t2_prev = t2.clone();
        t1 = t1_new;

        // --- Energy: 0.25 <ij||ab> t2 + 0.5 <ij||ab> t1_ia t1_jb ---
        let t1e = lbl2(t1.clone(), [O, V]);
        let t2e = lbl4(t2.clone(), [O, O, V, V]);
        let e2: f64 = 0.25 * einsum!("ijab,ijab->", &oovv, &t2e);
        // 0.5 einsum('ijab,ia,jb->', oovv, t1, t1): build Y[j,b] = einsum('ijab,ia->jb')
        // then einsum('jb,jb->').
        let jb: ArrayD<f64> = {
            // contract i,a ; left-free j,b ; right-free (none) -> 'jb'.
            einsum!("ijab,ia->jb", &oovv, &t1e)
        };
        let jb_t = lbl2(jb, [O, V]);
        let e1: f64 = 0.5 * einsum!("jb,jb->", &jb_t, &t1e);
        let e_corr = e2 + e1;

        let d_e = (e_corr - e_old).abs();
        if iter > 0 && d_e < cfg.energy_conv.min(1e-9) {
            let t2_out = t2.clone().into_dimensionality::<ndarray::Ix4>().unwrap();
            let t1_out = t1.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            println!(
                "spin-orbital CCSD converged in {} iterations. E_corr = {:.10}",
                iter, e_corr
            );
            return Ok(CcResult { correlation_energy: e_corr, t1: Some(t1_out), t2: t2_out });
        }
        e_old = e_corr;
    }

    Err(FerricError::Convergence("spin-orbital CCSD did not converge".into()))
}

/// Clone the underlying `ArrayD` out of a labeled `Tensor<4>` (used to seed a
/// mutable `W` intermediate from a loop-invariant integral block).
fn owned_arr4(t: &Tensor<4>) -> ArrayD<f64> {
    t.view().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    #[test]
    fn test_ccsd_h2_sto3g() {
        // Hard correctness gate: H2/STO-3G CCSD = -0.02052453 (exact-integral
        // numpy reference). Use def2-qzvpp-rifit so RI error is < 1e-6; cc-pvdz-ri
        // would leave ~5e-6 RI error (see ccd::ri_convergence_check_h2). Every
        // residual term and all 11 <pq||rs> blocks must be correct to hit this.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-9, ..Default::default() };
        let r = ccsd(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        println!("CCSD H2/STO-3G E_corr = {:.10}", r.correlation_energy);
        assert!(
            (r.correlation_energy - (-0.02052453)).abs() < 1e-6,
            "got {:.8}",
            r.correlation_energy
        );
    }

    #[test]
    fn ccsd_h2o_ccpvdz() {
        // ref CCSD(exact integrals) = -0.21332743. def2-qzvpp-rifit drives the RI
        // error below 1e-4 (cc-pvdz-ri leaves a larger DF error).
        let mol = Molecule::parse_xyz(
            "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-9, ..Default::default() };
        let r = ccsd(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        println!("CCSD H2O/cc-pVDZ E_corr = {:.10}", r.correlation_energy);
        assert!(
            (r.correlation_energy - (-0.21332743)).abs() < 1e-4,
            "got {:.8}",
            r.correlation_energy
        );
    }

    #[test]
    fn ccsd_ch4_sto3g() {
        // Third molecule (widens past H2/H2O): CH4/STO-3G, Td symmetry.
        // ref CCSD(exact integrals, PySCF cc.CCSD) = -0.07904929458457828.
        let mol = Molecule::parse_xyz(
            "5\nmethane Td\nC 0.0 0.0 0.0\nH 0.629118 0.629118 0.629118\n\
             H -0.629118 -0.629118 0.629118\nH -0.629118 0.629118 -0.629118\n\
             H 0.629118 -0.629118 -0.629118\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-9, ..Default::default() };
        let r = ccsd(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        println!("CCSD CH4/STO-3G E_corr = {:.10}", r.correlation_energy);
        assert!(
            (r.correlation_energy - (-0.07904929458457828)).abs() < 1e-4,
            "got {:.8}",
            r.correlation_energy
        );
    }

    #[test]
    fn ccsd_fails_fast_under_tiny_budget() {
        // M2 size guard: an explicit ~1 KB budget must ERROR before the VVVV /
        // eri3 allocations. Explicit config budget → no env var touched.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig {
            frozen_core: 0,
            memory_budget_bytes: Some(ferric_core::memory::gib_to_bytes(1e-6)),
            ..Default::default()
        };
        let err = match ccsd(&mol, &obs, &dfbs, op, &rhf, &cfg) {
            Err(e) => e,
            Ok(_) => panic!("CCSD should fail fast under tiny budget"),
        };
        let msg = err.to_string();
        assert!(msg.contains("CCSD") && msg.contains("budget is"), "unexpected: {msg}");
    }
}
