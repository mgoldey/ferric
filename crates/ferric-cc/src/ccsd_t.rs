//! CCSD(T) perturbative triples correction (spin-orbital).
//!
//! The non-iterative O(N^7) triples energy from the converged CCSD T1/T2.
//! Formula and conventions transcribed from PySCF `gccsd_t_slow.kernel`
//! (JCP 98, 8718 (1993)) and cross-checked in numpy against PySCF's RHF
//! `ccsd_t()` / GCCSD(T) — interleaved spin-orbital convention (2k=α, 2k+1=β),
//! reproducing H2O/cc-pVDZ (T) = -0.0030587091 to 7e-11. Uses the canonical-HF
//! simplification `fock[v,o]=0` (so the `fock·t2` piece of the disconnected
//! term vanishes), matching [`crate::ccsd::ccsd`].
//!
//! ```text
//! W = P(i/jk) P(a/bc) [ Σ_e t2_jkae <bc||ei>  −  Σ_m t2_imbc <ma||jk> ]   (connected)
//! V = P(i/jk) P(a/bc) [ t1_ia <bc||jk> ]                                  (disconnected)
//! t3c = W / D,   t3d = V / D,   D_ijkabc = ε_i+ε_j+ε_k − ε_a−ε_b−ε_c
//! E_(T) = (1/36) Σ_ijkabc (t3c + t3d) · D_ijkabc · t3c
//!       = (1/36) Σ_ijkabc (W + V) · W / D
//! ```
//!
//! MEMORY WARNING: this is the *dense* formulation — it materializes full 6D
//! `[2no, 2no, 2no, 2nv, 2nv, 2nv]` tensors. That is O((2no·2nv)³) storage:
//! ~0.4 GB for H2O/cc-pVDZ but ~1.8 TB for butane/cc-pVDZ. It is correct and
//! validated, but only runnable for very small systems. A production (T) must
//! loop over occupied triples i<j<k, forming one `[2nv,2nv,2nv]` block at a
//! time (~MB) — that rewrite is not done here.

use super::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::cholesky_inverse_sqrt;
use ferric_mp2::spinorbital::{asym_phys, build_b, transpose_b};
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{ArrayD, IxDyn};

/// P(a/bc) on the (…,a,b,c) axes 3,4,5: x − x.swap(a,b) − x.swap(a,c).
fn p_a_bc(x: &ArrayD<f64>) -> ArrayD<f64> {
    let s_ab = x.view().permuted_axes(IxDyn(&[0, 1, 2, 4, 3, 5])).as_standard_layout().into_owned();
    let s_ac = x.view().permuted_axes(IxDyn(&[0, 1, 2, 5, 4, 3])).as_standard_layout().into_owned();
    x - &s_ab - &s_ac
}
/// P(i/jk) on the (i,j,k,…) axes 0,1,2: x − x.swap(i,j) − x.swap(i,k).
fn p_i_jk(x: &ArrayD<f64>) -> ArrayD<f64> {
    let s_ij = x.view().permuted_axes(IxDyn(&[1, 0, 2, 3, 4, 5])).as_standard_layout().into_owned();
    let s_ik = x.view().permuted_axes(IxDyn(&[2, 1, 0, 3, 4, 5])).as_standard_layout().into_owned();
    x - &s_ij - &s_ik
}

/// Compute the (T) triples correction to CCSD.
///
/// Returns `E_(T)` (a negative number for typical closed-shell systems), to be
/// added to the CCSD correlation energy. Requires the CCSD T1 amplitudes.
pub fn ccsd_t(
    _mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cc: &CcResult,
    cfg: &CcConfig,
) -> Result<f64, FerricError> {
    let t1_spatial = cc
        .t1
        .as_ref()
        .ok_or_else(|| FerricError::General("CCSD(T) requires T1 amplitudes".into()))?;

    let nbas = obs.nbasis();
    let nocc_total = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
    let first_occ = cfg.frozen_core;
    let no = nocc_total - first_occ; // spatial active occ
    let nv = nbas - nocc_total; // spatial virtual
    let (no2, nv2) = (2 * no, 2 * nv);

    // No valid i<j<k triple unless there are ≥3 occupied spin-orbitals AND
    // ≥3 virtual spin-orbitals; (T) is identically zero otherwise (e.g. H2).
    if no2 < 3 || nv2 < 3 {
        return Ok(0.0);
    }

    // Fail-fast size guard: the dense (T) materializes several full 6D
    // [no2,no2,no2,nv2,nv2,nv2] tensors (d3 :133, t3c :162, t3d :170, plus the
    // einsum! term1/term2/t3d intermediates and p_a_bc/p_i_jk permutation copies
    // :39-49). Peak ≈ 8 co-resident 6D f64 buffers → 8·(no2·nv2)³·8 bytes. This
    // MUST stay next to those allocations; a butane input is ~1.8 TB here.
    let peak6d = (no2.saturating_mul(nv2))
        .saturating_pow(3)
        .saturating_mul(8) // ~8 co-resident 6D tensors
        .saturating_mul(8); // f64
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes);
    ferric_core::memory::check_alloc(
        &format!("CCSD(T) (no={no}, nv={nv} spatial; no2={no2}, nv2={nv2} spin-orbitals)"),
        peak6d,
        budget,
    )?;

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..nocc_total]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // Dressed RI 3-index MO blocks (same construction as the CCSD driver).
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
    let b_ov = build_b(&transform_3center_ov(&eri3_ao, &c_occ, &c_vir), &v_inv_sqrt, Axis::O, Axis::V);
    let b_oo = build_b(&transform_3center_oo(&eri3_ao, &c_occ), &v_inv_sqrt, Axis::O, Axis::O);
    let b_vv = build_b(&transform_3center_vv(&eri3_ao, &c_vir), &v_inv_sqrt, Axis::V, Axis::V);
    let b_vo = transpose_b(&b_ov);
    use Axis::{O, V};

    // --- Spin-orbital integral blocks needed by (T) ---
    // <bc||ei> (VVVO) indexed [b,c,e,i]: dir (be|ci)=b_vv,b_vo ; exc (bi|ce)=b_vo,b_vv
    let bcei = {
        let d: ArrayD<f64> = einsum!("Pbe,Pci->beci", &b_vv, &b_vo);
        let e: ArrayD<f64> = einsum!("Pbi,Pce->bice", &b_vo, &b_vv);
        asym_phys(&d, &e, nv, nv, nv, no)
    };
    // <ma||jk> (OVOO) indexed [m,a,j,k]: p=m,q=a,r=j,s=k. asym_phys wants g_dir
    // in chemist [p,r,q,s]=[m,j,a,k] (=(mj|ak)) and g_exc [p,s,q,r]=[m,k,a,j]
    // (=(mk|aj)); einsum! produces these layouts directly.
    let majk = {
        let d: ArrayD<f64> = einsum!("Pmj,Pak->mjak", &b_oo, &b_vo);
        let e: ArrayD<f64> = einsum!("Pmk,Paj->mkaj", &b_oo, &b_vo);
        asym_phys(&d, &e, no, nv, no, no)
    };
    // <bc||jk> (VVOO) indexed [b,c,j,k]: dir (bj|ck)=b_vo,b_vo ; exc (bk|cj)=b_vo,b_vo
    let bcjk = {
        let d: ArrayD<f64> = einsum!("Pbj,Pck->bjck", &b_vo, &b_vo);
        let e: ArrayD<f64> = einsum!("Pbk,Pcj->bkcj", &b_vo, &b_vo);
        asym_phys(&d, &e, nv, nv, no, no)
    };
    let bcei_t = Tensor::new(bcei, [V, V, V, O]);
    let majk_t = Tensor::new(majk, [O, V, O, O]);
    let bcjk_t = Tensor::new(bcjk, [V, V, O, O]);

    // --- Spin-orbital energies and the triples denominator D_ijkabc ---
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
    let mut d3 = ArrayD::zeros(IxDyn(&[no2, no2, no2, nv2, nv2, nv2]));
    for i in 0..no2 {
        for j in 0..no2 {
            for k in 0..no2 {
                for a in 0..nv2 {
                    for b in 0..nv2 {
                        for cc_ in 0..nv2 {
                            d3[[i, j, k, a, b, cc_]] =
                                eo[i] + eo[j] + eo[k] - ev[a] - ev[b] - ev[cc_];
                        }
                    }
                }
            }
        }
    }

    // --- T1 / T2 as labeled tensors ---
    let t1 = Tensor::new(t1_spatial.clone().into_dyn(), [O, V]);
    let t2 = Tensor::new(cc.t2.clone().into_dyn(), [O, O, V, V]);

    // --- Connected triples: t3c = einsum('jkae,bcei->ijkabc') - einsum('imbc,majk->ijkabc')
    // einsum! emits left-free then right-free. Term 1: left (j,k,a) free, contract e,
    // right free (b,c,i) -> 'jkabci'; want 'ijkabc' (i@5) => permute [5,0,1,2,3,4].
    let term1: ArrayD<f64> = einsum!("jkae,bcei->jkabci", &t2, &bcei_t);
    let term1 = term1.permuted_axes(IxDyn(&[5, 0, 1, 2, 3, 4])).as_standard_layout().into_owned();
    // Term 2: einsum('imbc,majk->ijkabc'): left (i,b,c) free, contract m, right free (a,j,k)
    // -> 'ibcajk'; want 'ijkabc' => src[i=0,b=1,c=2,a=3,j=4,k=5] -> [0,4,5,3,1,2].
    let term2: ArrayD<f64> = einsum!("imbc,majk->ibcajk", &t2, &majk_t);
    let term2 = term2.permuted_axes(IxDyn(&[0, 4, 5, 3, 1, 2])).as_standard_layout().into_owned();
    let mut t3c = &term1 - &term2;
    t3c = p_a_bc(&t3c);
    t3c = p_i_jk(&t3c);
    t3c = &t3c / &d3;

    // --- Disconnected triples: t3d = einsum('ia,bcjk->ijkabc'); canonical fock[v,o]=0.
    // left (i,a) free, contract (none), right free (b,c,j,k) -> 'iabcjk';
    // want 'ijkabc' => src[i=0,a=1,b=2,c=3,j=4,k=5] -> [0,4,5,1,2,3].
    let t3d: ArrayD<f64> = einsum!("ia,bcjk->iabcjk", &t1, &bcjk_t);
    let t3d = t3d.permuted_axes(IxDyn(&[0, 4, 5, 1, 2, 3])).as_standard_layout().into_owned();
    let mut t3d = p_a_bc(&t3d);
    t3d = p_i_jk(&t3d);
    t3d = &t3d / &d3;

    // --- Energy: (1/36) Σ (t3c + t3d) · D · t3c  (t3c,t3d already divided by D).
    let sum = &t3c + &t3d;
    let weighted = &sum * &d3;
    let et: f64 = (&weighted * &t3c).sum() / 36.0;
    Ok(et)
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
    use crate::ccsd::ccsd;

    #[test]
    fn test_ccsd_t_h2_sto3g_is_zero() {
        // H2/STO-3G has 1 occ + 1 vir spatial orbital, so there is no valid
        // i<j<k triple: E(T) is identically 0.0 (PySCF agrees to 1e-10). This
        // is the correct PHYSICAL answer, not stub behavior — it cannot
        // distinguish a working kernel from a stub, which is why the real gate
        // is the H2O test below.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cc_cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-9, ..Default::default() };
        let ccsd_res = ccsd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();
        let t_corr = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &ccsd_res, &cc_cfg).unwrap();
        assert!(t_corr.abs() < 1e-10, "H2 (T) should be 0, got {t_corr}");
    }

    #[test]
    fn test_ccsd_t_h2o_ccpvdz() {
        // Real correctness gate: H2O/cc-pVDZ (T) = -0.0030587091 (PySCF RHF
        // ccsd_t() and GCCSD(T), agree to 1e-9). Verified-exact spin-orbital
        // recipe (interleaved convention, drop canonical fock[v,o]); see the
        // numpy cross-check that reproduced this to 7e-11. def2-qzvpp-rifit
        // keeps the RI error below 1e-4.
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
        let cc_cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-9, ..Default::default() };
        let ccsd_res = ccsd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();
        let t_corr = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &ccsd_res, &cc_cfg).unwrap();
        println!("CCSD(T) H2O/cc-pVDZ (T) = {t_corr:.10}");
        assert!(
            (t_corr - (-0.0030587091)).abs() < 1e-4,
            "(T) = {t_corr:.10}, expected -0.0030587091"
        );
    }

    #[test]
    fn ccsd_t_fails_fast_under_tiny_budget() {
        // The M2 size guard must ERROR cleanly (not OOM-kill) before the dense 6D
        // allocation when the budget is tiny. Uses an EXPLICIT config budget so no
        // process-global env var is touched (explicit wins in resolve_budget_bytes).
        // We build a valid RHF + a dummy CcResult (t1/t2 shapes don't matter — the
        // guard fires before they are used numerically), then assert the error.
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

        // Dummy CCSD result — the guard runs before t1/t2 are consumed.
        let nbas = obs.nbasis();
        let nocc = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
        let nv = nbas - nocc;
        let dummy = CcResult {
            correlation_energy: 0.0,
            t1: Some(ndarray::Array2::<f64>::zeros((nocc, nv))),
            t2: ndarray::Array4::<f64>::zeros((nocc, nocc, nv, nv)),
        };
        // 1e-6 GiB ≈ 1 KB budget — far below the H2O/cc-pVDZ (T) peak.
        let cc_cfg = CcConfig {
            frozen_core: 0,
            memory_budget_bytes: Some(ferric_core::memory::gib_to_bytes(1e-6)),
            ..Default::default()
        };
        let err = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &dummy, &cc_cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("CCSD(T)"), "unexpected error: {msg}");
        assert!(msg.contains("budget is"), "unexpected error: {msg}");
    }
}
