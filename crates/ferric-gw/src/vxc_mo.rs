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
use ferric_dft::density_on_grid::{eval_density_closed, eval_density_uks};
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::{KsXc, KsXcUks};
use ferric_dft::vxc::{semilocal_vxc_closed, semilocal_vxc_polarized};
use ferric_scf::{ScfResult, Spin};
use ndarray::Array1;

/// Diagonal v_xc matrix elements in MO basis for UKS reference.
/// Returns (vxc_diag_α, vxc_diag_β), each of length `nmo`.
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
        let nlc_grid = AtomicGridConfig { n_radial: 50, n_angular: 50 };
        let ks_xc = KsXc::new(mol, bs, xc_name, &main_grid, &nlc_grid)
            .map_err(|e| FerricError::General(format!("KsXc::new: {e:?}")))?;
        let dens = eval_density_closed(&d_full, &ks_xc.chi, &ks_xc.dchi);
        // Meta-GGA needs τ; cheap and only computed when the functional is one.
        let tau = if ks_xc.xc.funcs.iter().any(|f| {
            matches!(f.family(), ferric_dft::libxc::FunctionalFamily::MetaGga)
        }) {
            Some(ferric_dft::density_on_grid::eval_tau_closed(&d_full, &ks_xc.dchi))
        } else {
            None
        };
        let (_e_xc, mut vxc_ao) = semilocal_vxc_closed(
            &ks_xc.grid, &ks_xc.chi, &ks_xc.dchi, &dens, tau.as_ref(), &ks_xc.xc,
        );
        // VV10 nonlocal piece is part of the KS Fock; subtract it too.
        if let (Some(g), Some(c), Some(dc), Some(params)) = (
            ks_xc.nlc_grid.as_ref(),
            ks_xc.nlc_chi.as_ref(),
            ks_xc.nlc_dchi.as_ref(),
            ks_xc.xc.vv10.as_ref(),
        ) {
            let dens_t = eval_density_closed(&d_full, c, dc);
            let mut v_nl = ndarray::Array2::<f64>::zeros(vxc_ao.dim());
            let _e_nl = ferric_dft::vv10::add_vv10(g, c, dc, &dens_t, params, &mut v_nl);
            vxc_ao += &v_nl;
        }
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
    let nlc_grid = AtomicGridConfig { n_radial: 50, n_angular: 50 };
    let ks_xc = KsXcUks::new(mol, bs, xc_name, &main_grid, &nlc_grid)
        .map_err(|e| FerricError::General(format!("KsXcUks::new: {e:?}")))?;

    let dens = eval_density_uks(d_a, d_b, &ks_xc.chi, &ks_xc.dchi);
    let tau = if ks_xc.xc.funcs.iter().any(|f| {
        matches!(f.family(), ferric_dft::libxc::FunctionalFamily::MetaGga)
    }) {
        Some(ferric_dft::density_on_grid::eval_tau_uks(d_a, d_b, &ks_xc.dchi))
    } else {
        None
    };
    let tau_ref = tau.as_ref().map(|(a, b)| (a, b));
    let (_e_xc, vxc_a_ao, vxc_b_ao) =
        semilocal_vxc_polarized(&ks_xc.grid, &ks_xc.chi, &ks_xc.dchi, &dens, tau_ref, &ks_xc.xc);

    // For VV10 the v_nl piece is the same for both spins (matches KsXcUks).
    // It's part of the KS Fock so we must subtract it too for the Σ_x − v_xc
    // correction to be consistent.
    let mut vxc_a = vxc_a_ao;
    let mut vxc_b = vxc_b_ao;
    if let (Some(g), Some(c), Some(dc), Some(params)) = (
        ks_xc.nlc_grid.as_ref(),
        ks_xc.nlc_chi.as_ref(),
        ks_xc.nlc_dchi.as_ref(),
        ks_xc.xc.vv10.as_ref(),
    ) {
        let d_total = d_a + d_b;
        let dens_total = ferric_dft::density_on_grid::eval_density_closed(&d_total, c, dc);
        let mut v_nl = ndarray::Array2::<f64>::zeros(vxc_a.dim());
        let _e_nl = ferric_dft::vv10::add_vv10(g, c, dc, &dens_total, params, &mut v_nl);
        vxc_a += &v_nl;
        vxc_b += &v_nl;
    }

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
