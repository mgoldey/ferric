//! v_xc evaluation in MO basis for UKS-reference U-GW.
//!
//! At a KS reference, the QP equation reads
//!   ε^QP = ε_KS + (Σ_x^GW − v_xc^KS) + Σ_c
//! The "−v_xc" correction is needed because the KS Fock contains v_xc rather
//! than Σ_x. For UHF/ROHF references it is identically zero (Σ_x^GW = v_xc^HF).
//!
//! This module evaluates the diagonal v_xc matrix elements in MO basis on a
//! per-spin basis, given an xc functional name and the SCF result.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::{KsXc, KsXcUks};
use ferric_dft::xc_trait::{UksXcContribution, XcContribution};
use ferric_scf::{ScfResult, Spin};
use ndarray::Array1;

/// Diagonal v_xc matrix elements in MO basis for UKS reference.
/// Returns (vxc_diag_α, vxc_diag_β), each of length `nmo`.
///
/// Builds the AO-basis V_xc (+ V_nl/VV10) matrix via the same
/// `XcContribution`/`UksXcContribution::add_xc[_uks]` entry point the SCF
/// itself uses to build the KS Fock — this automatically stays consistent
/// with whichever grid-cache strategy `KsXc`/`KsXcUks` picked internally
/// (the full cached path, or the budget-triggered batched fallback), rather
/// than reaching into the AO/χ tensors directly (which are a private
/// implementation detail of `KsXc`/`KsXcUks`, not guaranteed to be
/// eagerly-materialized).
pub fn vxc_diagonal_mo(
    mol: &Molecule,
    bs: &BasisSet,
    xc_name: &str,
    scf: &ScfResult,
) -> Result<(Array1<f64>, Array1<f64>), FerricError> {
    if matches!(scf.spin, Spin::Restricted) {
        // Full closed-shell density: solve_rhf stores density_alpha = 0.5·D_full,
        // so the physical density matrix is 2·density_alpha (rhf.rs).
        let d_full = 2.0 * &scf.density_alpha;
        let main_grid = AtomicGridConfig::default();
        let nlc_grid = AtomicGridConfig { n_radial: 50, n_angular: 50, ..Default::default() };
        let ks_xc = KsXc::new(mol, bs, xc_name, &main_grid, &nlc_grid)
            .map_err(|e| FerricError::General(format!("KsXc::new: {e:?}")))?;
        let mut vxc_ao = ndarray::Array2::<f64>::zeros((d_full.nrows(), d_full.ncols()));
        let _e_xc_plus_nl = ks_xc.add_xc(&d_full, &mut vxc_ao);
        let diag = mo_diagonal(&vxc_ao, scf.mos_r());
        return Ok((diag.clone(), diag));
    }
    let d_a = &scf.density_alpha;
    let d_b = scf
        .density_beta
        .as_ref()
        .ok_or_else(|| FerricError::General("vxc_diagonal_mo: missing density_beta".into()))?;

    // Build the same grid + AO values KsXcUks uses for SCF — guarantees the
    // v_xc we subtract matches what entered the KS Fock.
    let main_grid = AtomicGridConfig::default();
    let nlc_grid = AtomicGridConfig { n_radial: 50, n_angular: 50, ..Default::default() };
    let ks_xc = KsXcUks::new(mol, bs, xc_name, &main_grid, &nlc_grid)
        .map_err(|e| FerricError::General(format!("KsXcUks::new: {e:?}")))?;

    let mut vxc_a = ndarray::Array2::<f64>::zeros((d_a.nrows(), d_a.ncols()));
    let mut vxc_b = ndarray::Array2::<f64>::zeros((d_b.nrows(), d_b.ncols()));
    let _e_xc_plus_nl = ks_xc.add_xc_uks(d_a, d_b, &mut vxc_a, &mut vxc_b);

    // Transform to MO basis and take the diagonal: (vxc_σ)_pp = c_σ_p^T v_xc^AO c_σ_p.
    let (mos_a_arr, mos_b_arr) = (scf.mos_a(), scf.mos_b());
    let diag_a = mo_diagonal(&vxc_a, mos_a_arr);
    let diag_b = mo_diagonal(&vxc_b, mos_b_arr);
    Ok((diag_a, diag_b))
}

fn mo_diagonal(ao: &ndarray::Array2<f64>, c: &ndarray::Array2<f64>) -> Array1<f64> {
    let nmo = c.ncols();
    let mut diag = Array1::<f64>::zeros(nmo);
    let ac = ao.dot(c);
    for p in 0..nmo {
        let mut acc = 0.0;
        for mu in 0..c.nrows() {
            acc += c[(mu, p)] * ac[(mu, p)];
        }
        diag[p] = acc;
    }
    diag
}
