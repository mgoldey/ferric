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
//!   wB97X-V) require erfc/erf 2e derivative integrals split into two
//!   contributions — not supported in this round.
//! - `∇E_xc` is the semilocal XC AO-basis-derivative contribution. Grid-weight
//!   derivatives ("grid response") are NOT included — this introduces an
//!   error of ~1e-5 Ha/Bohr at (75, 110) grids, suitable for geometry
//!   optimization but not high-accuracy gradient calculations.
//!
//! Currently supports **LDA functionals only**. GGA / hybrid-GGA require AO
//! Hessians for the v_σ-coupled term in ∇E_xc.

use crate::gradient::{
    build_energy_weighted_density, oneelectron_gradient,
};
use crate::result::{ScfResult, Spin};
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::gradient::xc_gradient_closed_lda_from_density;
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

    // Validate functional family.
    let xc = xc_def_from_name(xc_name).map_err(|e| FerricError::General(format!("libxc: {e:?}")))?;
    for f in &xc.funcs {
        if !matches!(f.family(), FunctionalFamily::Lda) {
            return Err(FerricError::General(format!(
                "ks_gradient_closed: only LDA functionals supported; got {:?}",
                f.family()
            )));
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
    if k_mix.omega > 0.0 {
        return Err(FerricError::General(
            "ks_gradient_closed: range-separated hybrids not supported in this round".into(),
        ));
    }
    let c_k = k_mix.sr; // 0.0 for pure LDA

    let nocc = (mol.nelec() / 2) as usize;
    let w = build_energy_weighted_density(result, nocc);
    let d = result.density_r().clone();

    // 1e + nuclear repulsion gradient — identical to HF.
    let mut grad = oneelectron_gradient(mol, prep, &d, &w)?;

    // 2e gradient with hybrid K scaling: Γ = 0.5·D·D − (c_K/4)·D·D.
    grad += &twoelectron_gradient_scaled_k(prep, op, bounds, &d, c_k)?;

    // XC gradient (LDA path).
    let grid_cfg = AtomicGridConfig::default();
    let xc_grad = xc_gradient_closed_lda_from_density(
        mol,
        bs,
        &d,
        xc_name,
        &grid_cfg,
        prep.shell_to_atom(),
        prep.shell_offsets(),
        prep.shell_dims(),
    )
    .map_err(|e| FerricError::General(format!("xc gradient: {e:?}")))?;
    grad += &xc_grad;

    Ok(grad)
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
