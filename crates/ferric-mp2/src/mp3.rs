//! Spin-orbital MP3 correlation energy via the `einsum!` tensor framework.
//!
//! Implements third-order Moller-Plesset perturbation theory in a spin-orbital
//! basis, with all 4-index contractions routed through the `ferric_tensors::einsum!`
//! macro (the integral builds AND the MP3 term contractions). The RI 3-index MO
//! tensors are built per spatial block (OO/OV/VV), expanded to spin orbitals, and
//! antisymmetrized into physicist `<pq||rs>` blocks.
//!
//! Validated against PySCF spin-orbital references (`testdata/reference/*_mp3.json`).

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{Array2, ArrayD, IxDyn};

/// Results from a spin-orbital MP3 calculation.
#[derive(Debug, Clone)]
pub struct Mp3Result {
    /// Reference (RHF) total energy.
    pub e_hf: f64,
    /// MP2 correlation energy.
    pub e_mp2: f64,
    /// MP3 correlation energy (third-order increment).
    pub e_mp3: f64,
    /// Total correlation energy: e_mp2 + e_mp3.
    pub e_corr: f64,
    /// Total energy: e_hf + e_corr.
    pub e_total: f64,
}

/// Build a dressed RI 3-index MO tensor `B^P_{pq}` for a spatial block.
///
/// `eri3_mo` is `(naux, d1, d2)` already MO-transformed; `v_inv_sqrt` is
/// `V^{-1/2}` `(naux, naux)`. Returns `Tensor<3>` labeled `[Aux, l1, l2]` with
/// `B^P_{pq} = sum_Q V^{-1/2}_{PQ} (Q|pq)`.
fn build_b(eri3_mo: &ndarray::Array3<f64>, v_inv_sqrt: &Array2<f64>, l1: Axis, l2: Axis) -> Tensor<3> {
    let naux = eri3_mo.shape()[0];
    let d1 = eri3_mo.shape()[1];
    let d2 = eri3_mo.shape()[2];
    let flat = eri3_mo
        .view()
        .into_shape_with_order((naux, d1 * d2))
        .unwrap();
    let b_flat = v_inv_sqrt.dot(&flat); // (naux, d1*d2)
    let b = b_flat.into_shape_with_order((naux, d1, d2)).unwrap();
    Tensor::new(b.into_dyn(), [Axis::Aux, l1, l2])
}

/// Spin of spin-orbital index `p`: 0=alpha (even), 1=beta (odd).
#[inline]
fn spin(p: usize) -> usize {
    p & 1
}
/// Spatial index of spin-orbital `p`.
#[inline]
fn spat(p: usize) -> usize {
    p >> 1
}

/// Antisymmetrized spin-orbital `<ij||ab>` (oovv) from spatial chemist `(ia|jb)`.
///
/// `g_iajb[i,a,j,b] = (ia|jb)`. Output shape `(2no, 2no, 2nv, 2nv)`.
#[allow(clippy::needless_range_loop)]
fn asym_oovv(g_iajb: &ArrayD<f64>, no: usize, nv: usize) -> ArrayD<f64> {
    let (no2, nv2) = (2 * no, 2 * nv);
    let mut out = ArrayD::zeros(IxDyn(&[no2, no2, nv2, nv2]));
    for i in 0..no2 {
        for j in 0..no2 {
            for a in 0..nv2 {
                for b in 0..nv2 {
                    // <ij|ab> = (ia|jb) with spin(i)=spin(a), spin(j)=spin(b)
                    let dir = if spin(i) == spin(a) && spin(j) == spin(b) {
                        g_iajb[[spat(i), spat(a), spat(j), spat(b)]]
                    } else {
                        0.0
                    };
                    // <ij|ba> = (ib|ja) with spin(i)=spin(b), spin(j)=spin(a)
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
///
/// `g[p,q,r,s] = (pq|rs)`, all four indices in the SAME spatial space of size
/// `n` spatial / `2n` spin. `<pq||rs> = (pr|qs) - (ps|qr)`, i.e. index into `g`
/// as `g[p,r,q,s]` (direct) and `g[p,s,q,r]` (exchange). Used for vvvv and oooo.
#[allow(clippy::needless_range_loop)]
fn asym_same(g: &ArrayD<f64>, n: usize) -> ArrayD<f64> {
    let n2 = 2 * n;
    let mut out = ArrayD::zeros(IxDyn(&[n2, n2, n2, n2]));
    for p in 0..n2 {
        for q in 0..n2 {
            for r in 0..n2 {
                for s in 0..n2 {
                    // <pq|rs> = (pr|qs), spin(p)=spin(r) & spin(q)=spin(s)
                    let dir = if spin(p) == spin(r) && spin(q) == spin(s) {
                        g[[spat(p), spat(r), spat(q), spat(s)]]
                    } else {
                        0.0
                    };
                    // <pq|sr> = (ps|qr), spin(p)=spin(s) & spin(q)=spin(r)
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

/// Antisymmetrized `<kb||cj>` (ovvo) for the particle-hole MP3 term.
///
/// k,j occupied; b,c virtual. Direct `<kb|cj> = (kc|bj)` from `g_kcbj[k,c,b,j]`;
/// exchange `<kb|jc> = (kj|bc)` from `g_kjbc[k,j,b,c]`.
/// Output shape `(2no, 2nv, 2nv, 2no)` indexed `[k,b,c,j]`.
#[allow(clippy::needless_range_loop)]
fn asym_ovvo(g_kcbj: &ArrayD<f64>, g_kjbc: &ArrayD<f64>, no: usize, nv: usize) -> ArrayD<f64> {
    let (no2, nv2) = (2 * no, 2 * nv);
    let mut out = ArrayD::zeros(IxDyn(&[no2, nv2, nv2, no2]));
    for k in 0..no2 {
        for b in 0..nv2 {
            for c in 0..nv2 {
                for j in 0..no2 {
                    // <kb|cj> = (kc|bj), spin(k)=spin(c) & spin(b)=spin(j)
                    let dir = if spin(k) == spin(c) && spin(b) == spin(j) {
                        g_kcbj[[spat(k), spat(c), spat(b), spat(j)]]
                    } else {
                        0.0
                    };
                    // <kb|jc> = (kj|bc), spin(k)=spin(j) & spin(b)=spin(c)
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

/// Compute the spin-orbital MP3 correlation energy via RI integrals.
///
/// All 4-index contractions go through `einsum!`. Returns [`Mp3Result`].
pub fn mp3_energy(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
) -> Result<Mp3Result, FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let no = nocc_total - frozen_core; // spatial active occupied
    let first_occ = frozen_core;
    let nv = nbas - nocc_total; // spatial virtual

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + no]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // V^{-1/2} metric and AO 3-center integrals.
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = crate::rimp2::cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = threeindex::eri3_tensor(op, obs, dfbs)?;

    // Spatial RI 3-index MO blocks.
    let b_ov = build_b(
        &crate::mo_transform::transform_3center_ov(&eri3_ao, &c_occ, &c_vir),
        &v_inv_sqrt,
        Axis::O,
        Axis::V,
    );
    let b_oo = build_b(
        &crate::mo_transform::transform_3center_oo(&eri3_ao, &c_occ),
        &v_inv_sqrt,
        Axis::O,
        Axis::O,
    );
    let b_vv = build_b(
        &crate::mo_transform::transform_3center_vv(&eri3_ao, &c_vir),
        &v_inv_sqrt,
        Axis::V,
        Axis::V,
    );

    // The vir-occ ordered RI block B^P_{bj}, needed for the ovvo (kc|bj) integral.
    let b_vo = transpose_b(&b_ov);

    // --- Spatial chemist 4-index blocks via einsum! ---
    // (ia|jb): g[i,a,j,b]
    let g_iajb: ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);
    // (ab|cd): g[a,b,c,d]
    let g_abcd: ArrayD<f64> = einsum!("Pab,Pcd->abcd", &b_vv, &b_vv);
    // (ij|kl): g[i,j,k,l]
    let g_ijkl: ArrayD<f64> = einsum!("Pij,Pkl->ijkl", &b_oo, &b_oo);
    // (kc|bj): g[k,c,b,j] — electron1=(k occ, c vir), electron2=(b vir, j occ)
    let g_kcbj: ArrayD<f64> = einsum!("Pkc,Pbj->kcbj", &b_ov, &b_vo);
    // (kj|bc): g[k,j,b,c]
    let g_kjbc: ArrayD<f64> = einsum!("Pkj,Pbc->kjbc", &b_oo, &b_vv);

    // --- Spin-orbital antisymmetrized integrals ---
    let asym_oovv_t = asym_oovv(&g_iajb, no, nv);
    let v_vvvv = asym_same(&g_abcd, nv);
    let v_oooo = asym_same(&g_ijkl, no);
    let v_ovvo = asym_ovvo(&g_kcbj, &g_kjbc, no, nv);

    // --- Spin-orbital energies and amplitudes ---
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

    // t[i,j,a,b] = <ij||ab> / (eo_i + eo_j - ev_a - ev_b)
    let mut t = ArrayD::zeros(IxDyn(&[no2, no2, nv2, nv2]));
    for i in 0..no2 {
        for j in 0..no2 {
            for a in 0..nv2 {
                for b in 0..nv2 {
                    let d = eo[i] + eo[j] - ev[a] - ev[b];
                    t[[i, j, a, b]] = asym_oovv_t[[i, j, a, b]] / d;
                }
            }
        }
    }

    let t_t = Tensor::new(t.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);
    let oovv_t = Tensor::new(asym_oovv_t.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);
    let vvvv_t = Tensor::new(v_vvvv, [Axis::V, Axis::V, Axis::V, Axis::V]);
    let oooo_t = Tensor::new(v_oooo, [Axis::O, Axis::O, Axis::O, Axis::O]);
    let ovvo_t = Tensor::new(v_ovvo, [Axis::O, Axis::V, Axis::V, Axis::O]);

    // e_mp2 = 0.25 * sum t * <ij||ab>
    let e_mp2: f64 = 0.25 * einsum!("ijab,ijab->", &t_t, &oovv_t);

    // e_pp = 0.125 * t_ijab <ab||cd> t_ijcd
    let x: ArrayD<f64> = einsum!("ijab,abcd->ijcd", &t_t, &vvvv_t);
    let x_t = Tensor::new(x, [Axis::O, Axis::O, Axis::V, Axis::V]);
    let e_pp: f64 = 0.125 * einsum!("ijcd,ijcd->", &x_t, &t_t);

    // e_hh = 0.125 * <kl||ij> t_ijab t_klab
    let y: ArrayD<f64> = einsum!("klij,ijab->klab", &oooo_t, &t_t);
    let y_t = Tensor::new(y, [Axis::O, Axis::O, Axis::V, Axis::V]);
    let e_hh: f64 = 0.125 * einsum!("klab,klab->", &y_t, &t_t);

    // e_ph = sum_{ijabkc} t[i,j,a,b] ovvo[k,b,c,j] t[i,k,a,c]
    // z[i,a,k,c] = sum_{j,b} t[i,j,a,b] ovvo[k,b,c,j]
    let z: ArrayD<f64> = einsum!("ijab,kbcj->iakc", &t_t, &ovvo_t);
    let z_t = Tensor::new(z, [Axis::O, Axis::V, Axis::O, Axis::V]);
    // t[i,k,a,c] in (i,a,k,c) order: permute storage (i,k,a,c)->(i,a,k,c) = axes [0,2,1,3]
    let t_iakc = t.clone().permuted_axes(IxDyn(&[0, 2, 1, 3])).to_owned();
    let t_iakc_t = Tensor::new(t_iakc, [Axis::O, Axis::V, Axis::O, Axis::V]);
    let e_ph: f64 = einsum!("iakc,iakc->", &z_t, &t_iakc_t);

    let e_mp3 = e_pp + e_hh + e_ph;
    let e_corr = e_mp2 + e_mp3;

    Ok(Mp3Result {
        e_hf: rhf.energy,
        e_mp2,
        e_mp3,
        e_corr,
        e_total: rhf.energy + e_corr,
    })
}

/// Transpose a `[Aux, O, V]` RI block into `[Aux, V, O]` (i.e. B^P_{ai} = B^P_{ia}).
fn transpose_b(b_ov: &Tensor<3>) -> Tensor<3> {
    let out = b_ov.view().permuted_axes(IxDyn(&[0, 2, 1])).to_owned();
    Tensor::new(out, [Axis::Aux, Axis::V, Axis::O])
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

    fn run_mp3(xyz: &str, obs_name: &str, dfbs_name: &str) -> Mp3Result {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled(obs_name).unwrap();
        let dfbs_bs = basis::bundled(dfbs_name).unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        mp3_energy(&mol, &obs, &dfbs, op, &rhf, 0).unwrap()
    }

    // Tolerances reflect the RI density-fitting error vs the exact-integral
    // PySCF spin-orbital references (cc-pVDZ-RI aux): ~2e-6 for H2/STO-3G and
    // ~1.5e-4 for H2O/cc-pVDZ. These are large enough to pass under RI yet tight
    // enough to catch any physics/sign/index regression in the MP3 terms.
    #[test]
    fn h2_sto3g_mp3_matches_reference() {
        let xyz = "2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";
        let r = run_mp3(xyz, "sto-3g", "cc-pvdz-ri");
        assert!((r.e_mp2 - (-0.0131380736)).abs() < 5e-6, "mp2 {}", r.e_mp2);
        assert!((r.e_mp3 - (-0.0048360726)).abs() < 5e-6, "mp3 {}", r.e_mp3);
    }

    #[test]
    fn h2o_ccpvdz_mp3_matches_reference() {
        let xyz = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
        let r = run_mp3(xyz, "cc-pvdz", "cc-pvdz-ri");
        assert!((r.e_mp2 - (-0.2040035637)).abs() < 5e-5, "mp2 {}", r.e_mp2);
        assert!((r.e_mp3 - (-0.0067894117)).abs() < 5e-4, "mp3 {}", r.e_mp3);
    }
}
