//! Analytical nuclear gradient for closed-shell Kohn-Sham DFT.
//!
//! Composition:
//!
//! ```text
//!     ∇E_KS = ∇E_nn + ∇E_1e + ∇E_2e_scaled + ∇E_xc
//! ```
//!
//! Where:
//! - `∇E_nn + ∇E_1e` are identical to HF (nuclear repulsion + dT/dR, dV/dR,
//!   −W·dS/dR Pulay term). Reuses `oneelectron_gradient` from `gradient.rs`.
//! - `∇E_2e_scaled` uses the hybrid two-particle density
//!   `Γ_KS = ½·D·D − (c_K/4)·D·D` where `c_K = k_mix.sr` for plain hybrids
//!   (ω=0) and `c_K = 0` for pure DFT. Range-separated hybrids (ω>0, e.g.
//!   wB97X) split into c_SR·K[erfc(ω)] + c_LR·K[erf(ω)], each contributed via
//!   `twoelectron_k_gradient` with the corresponding operator.
//! - `∇E_xc` is the semilocal XC AO-basis-derivative contribution. Grid-weight
//!   derivatives ("grid response") are NOT included — this introduces an
//!   error of ~1e-5 Ha/Bohr at (75, 110) grids, suitable for geometry
//!   optimization but not high-accuracy gradient calculations.
//!
//! Supports LDA, GGA, plain hybrid-GGA (B3LYP), and range-separated hybrid-GGA
//! (wB97X-family) — semilocal piece + scaled exact-exchange piece + (when
//! the functional carries VV10, e.g. wB97X-V) the VV10 nonlocal-correlation
//! gradient via `ferric_dft::gradient::vv10_gradient_from_density`.
//!
//! GGA / hybrid-GGA require AO
//! Hessians for the v_σ-coupled term in ∇E_xc.

use crate::gradient::{
    build_energy_weighted_density, build_energy_weighted_density_uhf,
    oneelectron_gradient,
};
use crate::result::{ScfResult, Spin};
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::gradient::{
    xc_gradient_closed_gga_from_density, xc_gradient_closed_lda_from_density,
    xc_gradient_uks_from_density,
};
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::libxc::{xc_def_from_name, FunctionalFamily};
use ferric_dft::xc_trait::KMix;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ndarray::Array2;

/// Compute the closed-shell KS nuclear gradient.
///
/// Returns a `(natoms, 3)` array of dE/dR in atomic units.
///
/// Limitations (this round):
/// - LDA only (pure functionals: LDA, S+VWN). PBE / B3LYP / wB97X-V are
///   rejected with `UnsupportedFamily`.
/// - No grid response: Becke partition + radial weight derivatives are
///   neglected, giving ~1e-5 Ha/Bohr error.
/// - Range-separated hybrids not supported (would need erfc/erf 2e derivative
///   integrals contributing to ∇E_2e_scaled).
pub fn ks_gradient_closed(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
) -> Result<Array2<f64>, FerricError> {
    assert!(
        matches!(result.spin, Spin::Restricted),
        "ks_gradient_closed: ScfResult.spin must be Restricted"
    );

    // Validate functional family. LDA + GGA + plain HybridGga + RangeSepGga
    // are supported (semilocal piece). VV10 nonlocal correlation is NOT
    // included in the gradient — wB97X-V gradients drop the VV10 term and
    // therefore have an extra ~mHa/Bohr error vs the full PySCF gradient.
    let xc = xc_def_from_name(xc_name).map_err(|e| FerricError::General(format!("libxc: {e:?}")))?;
    let mut needs_gga = false;
    for f in &xc.funcs {
        match f.family() {
            FunctionalFamily::Lda => {}
            FunctionalFamily::Gga | FunctionalFamily::HybridGga
            | FunctionalFamily::RangeSepGga => needs_gga = true,
        }
    }
    // Pure DFT (LDA, PBE): no exact exchange, so sr=lr=0. KMix::default()
    // is HF (sr=lr=1) which would be wrong here.
    let k_mix: KMix = if let Some(cam) = xc.cam {
        KMix { sr: cam.c_sr, lr: cam.c_lr, omega: cam.omega }
    } else if let Some(mix) = xc.b3lyp_mix {
        KMix { sr: mix, lr: mix, omega: 0.0 }
    } else {
        KMix { sr: 0.0, lr: 0.0, omega: 0.0 }
    };
    let nocc = (mol.nelec() / 2) as usize;
    let w = build_energy_weighted_density(result, nocc);
    let d = result.density_r().clone();

    // 1e + nuclear repulsion gradient — identical to HF.
    let mut grad = oneelectron_gradient(mol, prep, &d, &w)?;

    // 2e gradient:
    //   * J piece always uses full Coulomb with Γ_J = 0.5·D·D
    //   * For plain hybrids (ω=0): single K with c_K = k_mix.sr
    //   * For RSH (ω>0): two K pieces with c_SR · K[erfc] + c_LR · K[erf]
    if k_mix.omega > 0.0 {
        // J piece (Coulomb, no K).
        grad += &twoelectron_gradient_scaled_k(prep, op, bounds, &d, 0.0)?;
        // K_SR (erfc(ω)) scaled by c_SR.
        grad += &twoelectron_k_gradient(
            prep, Operator::erfc(k_mix.omega), bounds, &d, k_mix.sr,
        )?;
        // K_LR (erf(ω)) scaled by c_LR.
        grad += &twoelectron_k_gradient(
            prep, Operator::erf(k_mix.omega), bounds, &d, k_mix.lr,
        )?;
    } else {
        // Plain hybrid or pure DFT: single Γ = 0.5·D·D − (c_K/4)·D·D.
        grad += &twoelectron_gradient_scaled_k(prep, op, bounds, &d, k_mix.sr)?;
    }

    // XC gradient: dispatch on functional family (LDA fast path, GGA needs Hessians).
    let grid_cfg = AtomicGridConfig::default();
    let xc_grad = if needs_gga {
        xc_gradient_closed_gga_from_density(
            mol, bs, &d, xc_name, &grid_cfg,
            prep.shell_to_atom(), prep.shell_offsets(), prep.shell_dims(),
        )
    } else {
        xc_gradient_closed_lda_from_density(
            mol, bs, &d, xc_name, &grid_cfg,
            prep.shell_to_atom(), prep.shell_offsets(), prep.shell_dims(),
        )
    }
    .map_err(|e| FerricError::General(format!("xc gradient: {e:?}")))?;
    grad += &xc_grad;

    // VV10 nonlocal-correlation gradient (only if the functional advertises it).
    if let Some(vv10_params) = xc.vv10 {
        let nlc_cfg = ferric_dft::grid::AtomicGridConfig { n_radial: 50, n_angular: 50 };
        let vv10_grad = ferric_dft::gradient::vv10_gradient_from_density(
            mol, bs, &d, &vv10_params, &nlc_cfg,
            prep.shell_to_atom(), prep.shell_offsets(), prep.shell_dims(),
        )
        .map_err(|e| FerricError::General(format!("vv10 gradient: {e:?}")))?;
        grad += &vv10_grad;
    }

    Ok(grad)
}

/// 2e K-only gradient with a chosen Coulomb-like operator (Coulomb, erfc, erf).
///
/// Used by the RSH gradient path: builds `Σ Γ_K · d(μν|λσ)/dR` with
/// `Γ_K = -(c_K/4) · D·D` (anti-symmetrized exchange piece). Caller is expected
/// to add this on top of a separate J-only gradient (from
/// `twoelectron_gradient_scaled_k` with `c_K=0`, Coulomb operator).
pub fn twoelectron_k_gradient(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
    c_k: f64,
) -> Result<Array2<f64>, FerricError> {
    let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();

    let mut grad = Array2::zeros((natoms, 3));
    let mut eng = Engine::new_2e_deriv(op, prep, 1e-14)?;
    let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let b12 = bounds.q[(s1, s2)];
            for s3 in 0..=s1 {
                let s4max = if s3 == s1 { s2 } else { s3 };
                for s4 in 0..=s4max {
                    let b34 = bounds.q[(s3, s4)];
                    if b12 * b34 * max_d < 1e-12 { continue; }
                    if let Some(dq) = eng.compute_eri_deriv_quartet(prep, s1, s2, s3, s4) {
                        let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                        let block_sz = n1 * n2 * n3 * n4;
                        let atoms = [sh2at[s1], sh2at[s2], sh2at[s3], sh2at[s4]];
                        let sym12 = s1 != s2;
                        let sym34 = s3 != s4;
                        let sym1234 = (s1, s2) != (s3, s4);
                        accum_2e_k_grad(
                            &mut grad, d, dq, block_sz, n1, n2, n3, n4,
                            offs[s1], offs[s2], offs[s3], offs[s4],
                            &atoms, sym12, sym34, sym1234, c_k,
                        );
                    }
                }
            }
        }
    }
    Ok(grad)
}

#[allow(clippy::too_many_arguments)]
fn accum_2e_k_grad(
    grad: &mut Array2<f64>,
    d: &Array2<f64>,
    dq: &[f64],
    block_sz: usize,
    n1: usize, n2: usize, n3: usize, n4: usize,
    o1: usize, o2: usize, o3: usize, o4: usize,
    atoms: &[usize; 4],
    sym12: bool, sym34: bool, sym1234: bool,
    c_k: f64,
) {
    let gamma_k = |mu, nu, la, sg| -> f64 {
        -0.25 * c_k * d[(mu, la)] * d[(nu, sg)]
    };
    for a in 0..n1 {
        for b in 0..n2 {
            for c in 0..n3 {
                for dd in 0..n4 {
                    let idx = ((a * n2 + b) * n3 + c) * n4 + dd;
                    let mu = o1 + a; let nu = o2 + b;
                    let la = o3 + c; let sg = o4 + dd;
                    let mut g = gamma_k(mu, nu, la, sg);
                    if sym12 { g += gamma_k(nu, mu, la, sg); }
                    if sym34 { g += gamma_k(mu, nu, sg, la); }
                    if sym12 && sym34 { g += gamma_k(nu, mu, sg, la); }
                    if sym1234 {
                        g += gamma_k(la, sg, mu, nu);
                        if sym12 { g += gamma_k(la, sg, nu, mu); }
                        if sym34 { g += gamma_k(sg, la, mu, nu); }
                        if sym12 && sym34 { g += gamma_k(sg, la, nu, mu); }
                    }
                    for center in 0..4 {
                        let atom = atoms[center];
                        for coord in 0..3 {
                            let dv = dq[(center * 3 + coord) * block_sz + idx];
                            grad[(atom, coord)] += g * dv;
                        }
                    }
                }
            }
        }
    }
}

/// 2e gradient with hybrid K scaling. Mirrors `twoelectron_gradient` from
/// `gradient.rs` but uses `Γ_KS = 0.5·D·D − (c_K/4)·D·D` instead of HF's
/// `0.5·D·D − 0.25·D·D`.
pub fn twoelectron_gradient_scaled_k(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
    c_k: f64,
) -> Result<Array2<f64>, FerricError> {
    let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();

    let mut grad = Array2::zeros((natoms, 3));
    let mut eng = Engine::new_2e_deriv(op, prep, 1e-14)?;
    let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let b12 = bounds.q[(s1, s2)];
            for s3 in 0..=s1 {
                let s4max = if s3 == s1 { s2 } else { s3 };
                for s4 in 0..=s4max {
                    let b34 = bounds.q[(s3, s4)];
                    if b12 * b34 * max_d < 1e-12 { continue; }
                    if let Some(dq) = eng.compute_eri_deriv_quartet(prep, s1, s2, s3, s4) {
                        let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                        let block_sz = n1 * n2 * n3 * n4;
                        let atoms = [sh2at[s1], sh2at[s2], sh2at[s3], sh2at[s4]];
                        let sym12 = s1 != s2;
                        let sym34 = s3 != s4;
                        let sym1234 = (s1, s2) != (s3, s4);
                        accum_2e_grad_scaled_k(
                            &mut grad, d, dq, block_sz, n1, n2, n3, n4,
                            offs[s1], offs[s2], offs[s3], offs[s4],
                            &atoms, sym12, sym34, sym1234, c_k,
                        );
                    }
                }
            }
        }
    }
    Ok(grad)
}

#[allow(clippy::too_many_arguments)]
fn accum_2e_grad_scaled_k(
    grad: &mut Array2<f64>,
    d: &Array2<f64>,
    dq: &[f64],
    block_sz: usize,
    n1: usize, n2: usize, n3: usize, n4: usize,
    o1: usize, o2: usize, o3: usize, o4: usize,
    atoms: &[usize; 4],
    sym12: bool, sym34: bool, sym1234: bool,
    c_k: f64,
) {
    let gamma_ks = |mu, nu, la, sg| -> f64 {
        0.5 * d[(mu, nu)] * d[(la, sg)] - 0.25 * c_k * d[(mu, la)] * d[(nu, sg)]
    };

    for a in 0..n1 {
        for b in 0..n2 {
            for c in 0..n3 {
                for dd in 0..n4 {
                    let idx = ((a * n2 + b) * n3 + c) * n4 + dd;
                    let mu = o1 + a;
                    let nu = o2 + b;
                    let la = o3 + c;
                    let sg = o4 + dd;

                    let mut g = gamma_ks(mu, nu, la, sg);
                    if sym12 { g += gamma_ks(nu, mu, la, sg); }
                    if sym34 { g += gamma_ks(mu, nu, sg, la); }
                    if sym12 && sym34 { g += gamma_ks(nu, mu, sg, la); }
                    if sym1234 {
                        g += gamma_ks(la, sg, mu, nu);
                        if sym12 { g += gamma_ks(la, sg, nu, mu); }
                        if sym34 { g += gamma_ks(sg, la, mu, nu); }
                        if sym12 && sym34 { g += gamma_ks(sg, la, nu, mu); }
                    }

                    for center in 0..4 {
                        let atom = atoms[center];
                        for coord in 0..3 {
                            let dv = dq[(center * 3 + coord) * block_sz + idx];
                            grad[(atom, coord)] += g * dv;
                        }
                    }
                }
            }
        }
    }
}

/// Spin-polarized (UKS) Kohn-Sham nuclear gradient.
///
/// Composition mirrors `ks_gradient_closed`:
///
/// ```text
///   ∇E_UKS = ∇E_nn + ∇E_1e + ∇E_2e_scaled_k + ∇E_xc[α,β] + ∇E_vv10[total]
/// ```
///
/// Limitations (this round):
/// - RSH (ω > 0) is rejected. UKS-RSH needs per-spin DfK_SR/DfK_LR derivative
///   integrals — same pattern as ks_gradient_closed's RSH path but doubled.
pub fn ks_gradient_uks(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
) -> Result<Array2<f64>, FerricError> {
    assert!(
        matches!(result.spin, Spin::Unrestricted),
        "ks_gradient_uks: ScfResult.spin must be Unrestricted"
    );

    let xc = ferric_dft::libxc::xc_def_from_name(xc_name)
        .map_err(|e| FerricError::General(format!("libxc: {e:?}")))?;
    let k_mix: KMix = if let Some(cam) = xc.cam {
        KMix { sr: cam.c_sr, lr: cam.c_lr, omega: cam.omega }
    } else if let Some(mix) = xc.b3lyp_mix {
        KMix { sr: mix, lr: mix, omega: 0.0 }
    } else {
        KMix { sr: 0.0, lr: 0.0, omega: 0.0 }
    };
    if k_mix.omega > 0.0 {
        return Err(FerricError::General(
            "ks_gradient_uks: range-separated UKS gradients not yet implemented".into(),
        ));
    }
    let c_k: f64 = k_mix.sr;

    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;

    let d_a = &result.density_alpha;
    let d_b = result
        .density_beta
        .as_ref()
        .expect("ks_gradient_uks: missing density_beta");
    let d_total = d_a + d_b;

    // 1e + nn gradient.
    let w = build_energy_weighted_density_uhf(result, nocc_a, nocc_b);
    let mut grad = oneelectron_gradient(mol, prep, &d_total, &w)?;

    // 2e gradient: J piece from D_total, K piece from per-spin densities,
    // scaled by c_K.
    grad += &twoelectron_gradient_uhf_scaled_k(
        prep, op, bounds, &d_total, d_a, d_b, c_k,
    )?;

    // XC gradient (polarized).
    let grid_cfg = AtomicGridConfig::default();
    let xc_grad = xc_gradient_uks_from_density(
        mol, bs, d_a, d_b, xc_name, &grid_cfg,
        prep.shell_to_atom(), prep.shell_offsets(), prep.shell_dims(),
    )
    .map_err(|e| FerricError::General(format!("uks xc gradient: {e:?}")))?;
    grad += &xc_grad;

    // VV10: function of ρ_tot, identical to the closed-shell formula.
    if let Some(vv10_params) = xc.vv10 {
        let nlc_cfg = ferric_dft::grid::AtomicGridConfig { n_radial: 50, n_angular: 50 };
        let vv10_grad = ferric_dft::gradient::vv10_gradient_from_density(
            mol, bs, &d_total, &vv10_params, &nlc_cfg,
            prep.shell_to_atom(), prep.shell_offsets(), prep.shell_dims(),
        )
        .map_err(|e| FerricError::General(format!("vv10 gradient: {e:?}")))?;
        grad += &vv10_grad;
    }

    Ok(grad)
}

/// Scaled-K UHF 2e gradient: same as `twoelectron_gradient_uhf` but the
/// exchange piece is multiplied by `c_K`. Used by UKS where hybrids include
/// a fraction of exact exchange.
///   Γ_UKS = 0.5·D·D − 0.5·c_K·(D_α·D_α + D_β·D_β)
pub fn twoelectron_gradient_uhf_scaled_k(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d_total: &Array2<f64>,
    d_alpha: &Array2<f64>,
    d_beta: &Array2<f64>,
    c_k: f64,
) -> Result<Array2<f64>, FerricError> {
    let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();

    let mut grad = Array2::zeros((natoms, 3));
    let mut eng = Engine::new_2e_deriv(op, prep, 1e-14)?;
    let max_d = d_total.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

    let gamma = |mu, nu, la, sg| -> f64 {
        0.5 * d_total[(mu, nu)] * d_total[(la, sg)]
            - 0.5 * c_k * (d_alpha[(mu, la)] * d_alpha[(nu, sg)]
                         + d_beta[(mu, la)]  * d_beta[(nu, sg)])
    };

    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let b12 = bounds.q[(s1, s2)];
            for s3 in 0..=s1 {
                let s4max = if s3 == s1 { s2 } else { s3 };
                for s4 in 0..=s4max {
                    let b34 = bounds.q[(s3, s4)];
                    if b12 * b34 * max_d < 1e-12 { continue; }
                    if let Some(dq) = eng.compute_eri_deriv_quartet(prep, s1, s2, s3, s4) {
                        let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                        let block_sz = n1 * n2 * n3 * n4;
                        let atoms = [sh2at[s1], sh2at[s2], sh2at[s3], sh2at[s4]];
                        let sym12 = s1 != s2;
                        let sym34 = s3 != s4;
                        let sym1234 = (s1, s2) != (s3, s4);
                        for a in 0..n1 {
                            for b in 0..n2 {
                                for c in 0..n3 {
                                    for dd in 0..n4 {
                                        let idx = ((a * n2 + b) * n3 + c) * n4 + dd;
                                        let mu = offs[s1] + a;
                                        let nu = offs[s2] + b;
                                        let la = offs[s3] + c;
                                        let sg = offs[s4] + dd;
                                        let mut g = gamma(mu, nu, la, sg);
                                        if sym12 { g += gamma(nu, mu, la, sg); }
                                        if sym34 { g += gamma(mu, nu, sg, la); }
                                        if sym12 && sym34 { g += gamma(nu, mu, sg, la); }
                                        if sym1234 {
                                            g += gamma(la, sg, mu, nu);
                                            if sym12 { g += gamma(la, sg, nu, mu); }
                                            if sym34 { g += gamma(sg, la, mu, nu); }
                                            if sym12 && sym34 { g += gamma(sg, la, nu, mu); }
                                        }
                                        for center in 0..4 {
                                            let atom = atoms[center];
                                            for coord in 0..3 {
                                                let dv = dq[(center * 3 + coord) * block_sz + idx];
                                                grad[(atom, coord)] += g * dv;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(grad)
}

/// Restricted-open-shell KS (ROKS) nuclear gradient.
///
/// Structure mirrors `ks_gradient_uks` but uses the ROHF energy-weighted
/// density convention (doubly-occ orbital weighted 2ε, singly-occ orbital
/// weighted 1ε) and the same UKS XC gradient via `xc_gradient_uks_from_density`.
/// The per-spin densities from a ROKS `ScfResult` already satisfy the
/// projector structure so the UKS XC path applies verbatim.
pub fn ks_gradient_roks(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
) -> Result<Array2<f64>, FerricError> {
    assert!(
        matches!(result.spin, Spin::RestrictedOpen),
        "ks_gradient_roks: ScfResult.spin must be RestrictedOpen"
    );

    let xc = ferric_dft::libxc::xc_def_from_name(xc_name)
        .map_err(|e| FerricError::General(format!("libxc: {e:?}")))?;
    let k_mix: KMix = if let Some(cam) = xc.cam {
        KMix { sr: cam.c_sr, lr: cam.c_lr, omega: cam.omega }
    } else if let Some(mix) = xc.b3lyp_mix {
        KMix { sr: mix, lr: mix, omega: 0.0 }
    } else {
        KMix { sr: 0.0, lr: 0.0, omega: 0.0 }
    };
    if k_mix.omega > 0.0 {
        return Err(FerricError::General(
            "ks_gradient_roks: range-separated ROKS gradients not yet implemented".into(),
        ));
    }
    let c_k: f64 = k_mix.sr;

    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_open = two_s as usize;
    let nocc_double = ((nelec - two_s) / 2) as usize;

    let d_a = &result.density_alpha;
    let d_b = result
        .density_beta
        .as_ref()
        .expect("ks_gradient_roks: missing density_beta");
    let d_total = d_a + d_b;

    // ROHF energy-weighted density: closed × 2ε, open × ε.
    let n = result.mos_alpha.nrows();
    let c_mo = &result.mos_alpha;
    let eps = &result.eps_alpha;
    let mut w = Array2::<f64>::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for i in 0..nocc_double {
                sum += 2.0 * eps[i] * c_mo[(mu, i)] * c_mo[(nu, i)];
            }
            for j in nocc_double..nocc_double + nocc_open {
                sum += eps[j] * c_mo[(mu, j)] * c_mo[(nu, j)];
            }
            w[(mu, nu)] = sum;
        }
    }

    let mut grad = oneelectron_gradient(mol, prep, &d_total, &w)?;

    // 2e: J from D_total, K_α/K_β scaled by c_K.
    grad += &twoelectron_gradient_uhf_scaled_k(
        prep, op, bounds, &d_total, d_a, d_b, c_k,
    )?;

    // XC piece via the polarized driver (ROKS densities already satisfy the
    // structure; UKS XC machinery applies as-is).
    let grid_cfg = AtomicGridConfig::default();
    let xc_grad = xc_gradient_uks_from_density(
        mol, bs, d_a, d_b, xc_name, &grid_cfg,
        prep.shell_to_atom(), prep.shell_offsets(), prep.shell_dims(),
    )
    .map_err(|e| FerricError::General(format!("roks xc gradient: {e:?}")))?;
    grad += &xc_grad;

    // VV10 (closed-shell-friendly, on total ρ).
    if let Some(vv10_params) = xc.vv10 {
        let nlc_cfg = ferric_dft::grid::AtomicGridConfig { n_radial: 50, n_angular: 50 };
        let vv10_grad = ferric_dft::gradient::vv10_gradient_from_density(
            mol, bs, &d_total, &vv10_params, &nlc_cfg,
            prep.shell_to_atom(), prep.shell_offsets(), prep.shell_dims(),
        )
        .map_err(|e| FerricError::General(format!("vv10 gradient: {e:?}")))?;
        grad += &vv10_grad;
    }

    Ok(grad)
}
