pub mod config;
pub mod davidson;
pub mod diagnostics;
pub mod energy;
pub mod quadrature;
pub mod sternheimer;

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{compute_mp2_intermediates, RiMp2Config};
use ferric_scf::rhf::RhfResult;
use ndarray::Array2;

pub use config::PdepRpaConfig;

/// Results from a PDEP-RPA calculation.
#[derive(Debug)]
pub struct PdepRpaResult {
    /// RPA correlation energy in Hartree.
    pub e_rpa: f64,
    /// Number of eigenpotentials retained after truncation.
    pub n_eigenpotentials: usize,
    /// Static dielectric eigenvalues λ_α(0), length M.
    pub eigenvalues_static: Vec<f64>,
    /// Imaginary-frequency quadrature points ω_k.
    pub quad_freqs: Vec<f64>,
    /// Quadrature weights w_k.
    pub quad_weights: Vec<f64>,
    /// λ_α(iω_k) tensor, shape (N_quad, M).
    pub eigenvalues_freq: Array2<f64>,
    /// RI-dRPA sanity-check energy (None unless run_diagnostics=true).
    pub e_rpa_dft_diag: Option<f64>,
}

/// Top-level PDEP-RPA energy calculation.
pub fn run_pdep_rpa(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &RhfResult,
    config: &PdepRpaConfig,
) -> Result<PdepRpaResult, FerricError> {
    // Step 1: Build MP2 intermediates to get B_ov and orbital energies.
    let mp2_cfg = RiMp2Config { frozen_core: config.frozen_core };
    let inter = compute_mp2_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;

    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let naux = inter.naux;
    let first_occ = inter.first_occ;
    let nocc_total = inter.nocc_total;

    // Step 2: Extract orbital energy slices.
    let eps_occ: Vec<f64> = rhf.orbital_energies[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = rhf.orbital_energies[nocc_total..nocc_total + nvir].to_vec();

    // Step 3: Run Davidson at ω=0.
    let max_vecs = if config.davidson_max_vecs == 0 {
        3 * naux
    } else {
        config.davidson_max_vecs
    };

    let b_ov_clone = b_ov.clone();
    let eps_occ_clone = eps_occ.clone();
    let eps_vir_clone = eps_vir.clone();

    let davidson_result = davidson::run_davidson_static(
        naux,
        move |v_mat: &Array2<f64>, omega: f64| {
            sternheimer::dielectric_matrix(v_mat, &b_ov_clone, &eps_occ_clone, &eps_vir_clone, omega)
        },
        config.davidson_conv_thresh,
        max_vecs,
        naux, // request all, truncate later
    )?;

    // Step 4: Truncate by departure from identity: keep eigenpotentials where
    // (λ_α(0) − 1) > trunc_thresh. The dielectric ε̃ = I + Π has eigenvalues ≥ 1,
    // so identity-modes (λ = 1) carry no RPA weight; only |λ−1| ≠ 0 modes matter.
    let n_keep = davidson_result
        .eigenvalues
        .iter()
        .filter(|&&lam| (lam - 1.0).abs() > config.trunc_thresh)
        .count();
    let n_keep = n_keep.max(1);

    let eigenvalues_static: Vec<f64> = davidson_result.eigenvalues[..n_keep].to_vec();
    let eigenvectors = davidson_result.eigenvectors.slice(ndarray::s![.., ..n_keep]).to_owned();

    // Step 5: Build quadrature grid.
    let (quad_freqs, quad_weights) = quadrature::build_quadrature(&config.quadrature);

    // Step 6: Evaluate λ_α(iω_k).
    let eigenvalues_freq = energy::eval_eigenvalues_at_frequencies(
        &eigenvectors,
        b_ov,
        &eps_occ,
        &eps_vir,
        &quad_freqs,
    );

    // Step 7: Integrate RPA correlation energy.
    let e_rpa = energy::rpa_correlation_energy(&quad_weights, &eigenvalues_freq);

    // Step 8: Diagnostic RI-dRPA energy (optional — full naux²×N_quad cost).
    let e_rpa_dft_diag = if config.run_diagnostics {
        Some(diagnostics::ri_drpa_energy(
            b_ov, &eps_occ, &eps_vir, &quad_freqs, &quad_weights,
        )?)
    } else {
        None
    };

    Ok(PdepRpaResult {
        e_rpa,
        n_eigenpotentials: n_keep,
        eigenvalues_static,
        quad_freqs,
        quad_weights,
        eigenvalues_freq,
        e_rpa_dft_diag,
    })
}
