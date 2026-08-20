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
use crate::rimp2::active_occ;
use crate::spinorbital::{asym_oovv, asym_ovvo, asym_same, build_b, transpose_b};
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{ArrayD, IxDyn};

/// Results from a spin-orbital MP3 calculation.
#[derive(Debug, Clone)]
#[must_use]
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

impl std::fmt::Display for Mp3Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MP3 total: {:.10} Ha (MP2: {:.10}, MP3: {:.10})",
            self.e_total, self.e_mp2, self.e_mp3)
    }
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
    let no = active_occ(nocc_total, frozen_core)?; // spatial active occupied
    let first_occ = frozen_core;
    let nv = nbas - nocc_total; // spatial virtual

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + no]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // Fail-fast size guard: peak is the spin-orbital VVVV block v_vvvv (:96) —
    // a (2nv)⁴ f64 tensor built from the einsum! g_abcd intermediate (:98) held
    // co-resident with the asym_same result → ~2× (2nv)⁴. No config budget on
    // this reference path. Keep next to that allocation.
    let nv2 = 2 * nv;
    let peak_vvvv = nv2.saturating_pow(4).saturating_mul(2).saturating_mul(8); // ~2× (2nv)⁴ f64
    ferric_core::memory::check_alloc(
        &format!("MP3 (no={no}, nv={nv} spatial; VVVV block over {nv2} spin-orbital virtuals)"),
        peak_vvvv,
        ferric_core::memory::resolve_budget_bytes(None),
    )?;

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

    // --- Spin-orbital antisymmetrized integrals ---
    // Each spatial chemist 4-index block is built via einsum!, immediately
    // consumed by its antisymmetrizer, and dropped at the end of its scope so
    // at most ONE spatial g block (e.g. the nvir^4 g_abcd) is alive at a time.
    let asym_oovv_t = {
        // (ia|jb): g[i,a,j,b]
        let g_iajb: ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);
        asym_oovv(&g_iajb, no, nv)
    };
    let v_vvvv = {
        // (ab|cd): g[a,b,c,d]
        let g_abcd: ArrayD<f64> = einsum!("Pab,Pcd->abcd", &b_vv, &b_vv);
        asym_same(&g_abcd, nv)
    };
    let v_oooo = {
        // (ij|kl): g[i,j,k,l]
        let g_ijkl: ArrayD<f64> = einsum!("Pij,Pkl->ijkl", &b_oo, &b_oo);
        asym_same(&g_ijkl, no)
    };
    let v_ovvo = {
        // (kc|bj): g[k,c,b,j] — electron1=(k occ, c vir), electron2=(b vir, j occ)
        let g_kcbj: ArrayD<f64> = einsum!("Pkc,Pbj->kcbj", &b_ov, &b_vo);
        // (kj|bc): g[k,j,b,c]
        let g_kjbc: ArrayD<f64> = einsum!("Pkj,Pbc->kjbc", &b_oo, &b_vv);
        asym_ovvo(&g_kcbj, &g_kjbc, no, nv)
    };

    // --- Spin-orbital energies and amplitudes ---
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

    // `t_t` needs an owned amplitude copy (used by every energy term); the
    // original `t` is kept owned so its LAST use (the t_iakc permute in e_ph)
    // can consume it without a clone.
    let t_t = Tensor::new(t.clone(), [Axis::O, Axis::O, Axis::V, Axis::V]);
    // The amplitude fill above already consumed asym_oovv_t by value reads, but
    // it is still needed for e_mp2 — move it into oovv_t (no clone).
    let oovv_t = Tensor::new(asym_oovv_t, [Axis::O, Axis::O, Axis::V, Axis::V]);

    // e_mp2 = 0.25 * sum t * <ij||ab>; oovv_t freed at end of scope.
    let e_mp2: f64 = {
        let v = 0.25 * einsum!("ijab,ijab->", &t_t, &oovv_t);
        drop(oovv_t);
        v
    };

    // e_pp = 0.125 * t_ijab <ab||cd> t_ijcd. Compute first so the (2nv)^4 hog
    // v_vvvv is moved into vvvv_t and freed (with x) as early as possible.
    let e_pp: f64 = {
        let vvvv_t = Tensor::new(v_vvvv, [Axis::V, Axis::V, Axis::V, Axis::V]);
        let x: ArrayD<f64> = einsum!("ijab,abcd->ijcd", &t_t, &vvvv_t);
        let x_t = Tensor::new(x, [Axis::O, Axis::O, Axis::V, Axis::V]);
        0.125 * einsum!("ijcd,ijcd->", &x_t, &t_t)
    };

    // e_hh = 0.125 * <kl||ij> t_ijab t_klab; v_oooo moved into oooo_t and freed.
    let e_hh: f64 = {
        let oooo_t = Tensor::new(v_oooo, [Axis::O, Axis::O, Axis::O, Axis::O]);
        let y: ArrayD<f64> = einsum!("klij,ijab->klab", &oooo_t, &t_t);
        let y_t = Tensor::new(y, [Axis::O, Axis::O, Axis::V, Axis::V]);
        0.125 * einsum!("klab,klab->", &y_t, &t_t)
    };

    // e_ph = sum_{ijabkc} t[i,j,a,b] ovvo[k,b,c,j] t[i,k,a,c]; v_ovvo moved in.
    let e_ph: f64 = {
        let ovvo_t = Tensor::new(v_ovvo, [Axis::O, Axis::V, Axis::V, Axis::O]);
        // z[i,a,k,c] = sum_{j,b} t[i,j,a,b] ovvo[k,b,c,j]
        let z: ArrayD<f64> = einsum!("ijab,kbcj->iakc", &t_t, &ovvo_t);
        let z_t = Tensor::new(z, [Axis::O, Axis::V, Axis::O, Axis::V]);
        // t[i,k,a,c] in (i,a,k,c) order: permute storage (i,k,a,c)->(i,a,k,c) =
        // axes [0,2,1,3]. This is the LAST use of `t`, so consume it (no clone).
        let t_iakc = t.permuted_axes(IxDyn(&[0, 2, 1, 3])).to_owned();
        let t_iakc_t = Tensor::new(t_iakc, [Axis::O, Axis::V, Axis::O, Axis::V]);
        einsum!("iakc,iakc->", &z_t, &t_iakc_t)
    };

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
        // mp3_energy reads the process-global FERRIC_MEM_BUDGET_GB internally;
        // hold ENV_LOCK so callers can't observe
        // mp3_fails_fast_under_tiny_env_budget's temporary tiny-budget
        // mutation under cargo test's default parallelism (found 2026-07-18,
        // same class of bug as gto_eval.rs).
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

    // FERRIC_MEM_BUDGET_GB is process-global; serialize env-mutating tests.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn mp3_fails_fast_under_tiny_env_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // cc-pVDZ (nv2=18 → ~1.7 MB VVVV peak) so the ~1 KB budget actually
        // trips (H2/STO-3G's VVVV is only 256 bytes — under even a 1 KB budget).
        let mol = Molecule::parse_xyz("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        std::env::set_var("FERRIC_MEM_BUDGET_GB", "0.000001");
        let res = mp3_energy(&mol, &obs, &dfbs, op, &rhf, 0);
        std::env::remove_var("FERRIC_MEM_BUDGET_GB");
        let err = res.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("MP3") && msg.contains("budget is"), "unexpected: {msg}");
    }

    #[test]
    fn mp3_frozen_core_exceeding_nocc_errors_not_panics() {
        // H2/STO-3G has nocc_total = 1; frozen_core = 2 must Err (not underflow
        // `nocc_total - frozen_core` as a usize and panic).
        // mp3_energy reads the process-global FERRIC_MEM_BUDGET_GB internally;
        // hold ENV_LOCK so this can't observe
        // mp3_fails_fast_under_tiny_env_budget's temporary tiny-budget
        // mutation under cargo test's default parallelism (found 2026-07-18,
        // same class of bug as gto_eval.rs).
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::parse_xyz("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let res = mp3_energy(&mol, &obs, &dfbs, op, &rhf, 2);
        assert!(res.is_err(), "expected Err for frozen_core > nocc_total, got {res:?}");
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
