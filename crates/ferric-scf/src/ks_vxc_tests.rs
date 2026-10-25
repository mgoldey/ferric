//! [`crate::rhf::ks_vxc_closed`] is EXACTLY the V_xc the RKS SCF put in its Fock.
//!
//! At a converged exit `solve_rhf` returns `fock_alpha = F[D]` and
//! `density_total = D` from the SAME iteration, where
//!
//!   F[D] = H + J[D] − ½·K_mix[D] + V_xc[D]
//!
//! with K_mix = c·K (global hybrid) or c_sr·K_SR + c_lr·K_LR (RSH, built only
//! through the SR/LR RI-K fitters). This module rebuilds H, J[D] and K_mix[D]
//! independently of the SCF loop, from the returned D, and requires
//!
//!   F − H − J[D] + ½·K_mix[D] == ks_vxc_closed(config, D).1
//!
//! LDA (no K), PBE0 (exact four-centre K) and wB97X-V (RSH ω + VV10, SR/LR
//! RI-K) are covered; wB97X-V is the first quantitative check of the V_xc
//! that the VV10 and RSH paths contribute.
//!
//! BAR (derived, not guessed). The two sides are the same arithmetic on the same
//! D up to (a) J/K screening differences between the SCF's incremental direct
//! build and a fresh one, bounded by `integral_thresh` = 1e-12 times a few
//! hundred quartets, and (b) the occupied-factored XC density pass the SCF uses
//! against the dense one here, which agree to rounding. So the residual is a
//! rounding residual, O(1e-15)·|F|; the bar `TOL` = 1e-11 leaves four orders of
//! headroom for screening and BLAS differences across machines. A wrong density
//! convention (V of D/2) or a missing V_xc is O(0.1–1).
//!
//! MEASURED (water/STO-3G, exact J, density_conv 1e-10), max-abs residual vs
//! max|V_xc|: LDA 5.6e-16 / 2.92, PBE0 4.4e-16 / 2.32, wB97X-V 1.3e-15 / 2.45.
//! Mutants: V[D/2] gives 0.59 (LDA) and 0.45 (wB97X-V); V[0] (= returning
//! zeros) gives max|V_xc| itself.

use crate::direct_jk::DirectJK;
use crate::fock::KBuilder;
use crate::rhf::{ks_vxc_closed, solve_rhf, RhfConfig};
use crate::screening::SchwarzBounds;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ndarray::Array2;

/// Explicit budget so the SCF never reads the process-global budget env var
/// (rhf.rs tests mutate it under their own lock).
const BUDGET: usize = 2 << 30;

/// See the module doc for the derivation.
const TOL: f64 = 1e-11;

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Returns (max|F − H − J + ½K_mix − V_xc|, max|V_xc|) for `xc` on water/STO-3G.
fn fock_residual(xc: &str, mutate_d: impl Fn(&Array2<f64>) -> Array2<f64>) -> (f64, f64) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let mut cfg = RhfConfig::default().with_xc(xc).with_density_conv(1e-10);
    cfg.energy_conv = 1e-10;
    cfg.three_index_budget_bytes = BUDGET;
    // Exact four-centre J (and K for a global hybrid): with `None` a functional
    // auto-defaults to RI-J / RI-K, which the rebuild below would not match.
    // RSH exchange exists only as SR/LR RI-K, so `df_k_aux` stays unset there
    // (the SCF and `build_rsh_dfk_pair(None)` then use the same default aux).
    let mix = ferric_dft::libxc::xc_def_from_name(xc).unwrap();
    cfg.df_j_aux = Some(String::new());
    if mix.cam.is_none() {
        cfg.df_k_aux = Some(String::new());
    }
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged, "{xc}: SCF did not converge");
    let d = &r.density_total;

    let h = oneelectron::hcore_ecp_with_external(&prep, &mol, &bs, None).unwrap();
    let n = h.nrows();
    let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    DirectJK::new(&ctx, &prep, &bounds, cfg.integral_thresh, BUDGET)
        .build(d, &mut j, &mut k)
        .unwrap();
    let half_kmix = if let Some(cam) = mix.cam {
        let (mut sr, mut lr) =
            crate::fock_assembly::build_rsh_dfk_pair(&ctx, &mol, &prep, None, cam.omega, BUDGET)
                .unwrap();
        let (mut ksr, mut klr) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        sr.build(d, &mut ksr).unwrap();
        lr.build(d, &mut klr).unwrap();
        0.5 * (cam.c_sr * &ksr + cam.c_lr * &klr)
    } else {
        0.5 * mix.b3lyp_mix.unwrap_or(0.0) * &k
    };
    let v_scf = &r.fock_alpha - &h - &j + &half_kmix;
    let (_, v) = ks_vxc_closed(&mol, &bs, &cfg, &mutate_d(d)).unwrap();
    (max_abs(&(&v_scf - &v)), max_abs(&v))
}

fn same(d: &Array2<f64>) -> Array2<f64> {
    d.clone()
}

#[test]
fn ks_vxc_closed_is_the_scf_fock_xc_term_lda() {
    let (res, vmax) = fock_residual("LDA", same);
    assert!(
        vmax > 0.1 && res < TOL,
        "LDA: residual {res:.3e}, max|V| {vmax:.3e}"
    );
}

#[test]
fn ks_vxc_closed_is_the_scf_fock_xc_term_pbe0() {
    let (res, vmax) = fock_residual("PBE0", same);
    assert!(
        vmax > 0.1 && res < TOL,
        "PBE0: residual {res:.3e}, max|V| {vmax:.3e}"
    );
}

#[test]
fn ks_vxc_closed_is_the_scf_fock_xc_term_wb97x_v() {
    let (res, vmax) = fock_residual("wB97X-V", same);
    assert!(
        vmax > 0.1 && res < TOL,
        "wB97X-V: residual {res:.3e}, max|V| {vmax:.3e}"
    );
}

/// Mutations: V evaluated for D/2 (a density-convention slip) and for D = 0
/// (V_xc[0] = 0, i.e. the old all-zeros placeholder) must both fail the bar.
#[test]
fn ks_vxc_closed_of_the_wrong_density_fails_the_fock_check() {
    for xc in ["LDA", "wB97X-V"] {
        let (res, _) = fock_residual(xc, |d| 0.5 * d);
        assert!(res > 1e-3, "{xc}: V[D/2] passed with residual {res:.3e}");
    }
    let (res, vmax) = fock_residual("LDA", |d| 0.0 * d);
    assert!(
        vmax == 0.0 && res > 1.0,
        "LDA: zeros passed with residual {res:.3e}"
    );
}
