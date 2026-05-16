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
use ndarray::{Array1, Array2};

pub use config::PdepRpaConfig;

/// Build an atom-localized seed for Davidson in the dressed aux basis.
///
/// For each atom A, constructs:
///   - 1 isotropic vector: uniform over all aux functions on atom A
///   - 3 directional vectors: linearly weighted by aux-function index modulo 3
///     (x/y/z surrogates) over aux functions on atom A
///
/// Total seed size: 4 * N_atoms vectors (before QR rank-reduction).
/// After QR, dependent columns are discarded; the result has at most
/// min(4*natoms, naux) orthonormal columns.
fn build_atom_seed(dfbs: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::QR;

    let naux = dfbs.nbasis();
    let natoms = dfbs.natoms();
    let shell_to_atom = dfbs.shell_to_atom();
    let shell_offsets = dfbs.shell_offsets();

    let n_seed_cols = 4 * natoms;
    let mut seed = Array2::<f64>::zeros((naux, n_seed_cols));

    for atom in 0..natoms {
        // Collect aux-function indices for this atom
        let mut aux_indices: Vec<usize> = Vec::new();
        for (sh, &a) in shell_to_atom.iter().enumerate() {
            if a == atom {
                for p in shell_offsets[sh]..shell_offsets[sh + 1] {
                    aux_indices.push(p);
                }
            }
        }
        if aux_indices.is_empty() {
            continue;
        }

        let n_on_atom = aux_indices.len() as f64;

        // Isotropic: 1/sqrt(n) for each aux function on this atom
        let inv_norm = n_on_atom.sqrt().recip();
        for &p in &aux_indices {
            seed[(p, 4 * atom)] = inv_norm;
        }

        // Three directional vectors: weight aux functions by index modulo 3
        // These provide differentiated projections across the aux space of each atom,
        // seeding the x/y/z directional response components.
        for dim in 0..3usize {
            let mut col: Array1<f64> = Array1::zeros(naux);
            for (k, &p) in aux_indices.iter().enumerate() {
                if k % 3 == dim {
                    col[p] = 1.0;
                }
            }
            let norm = col.dot(&col).sqrt();
            if norm > 1e-14 {
                col.mapv_inplace(|x| x / norm);
            }
            seed.column_mut(4 * atom + 1 + dim).assign(&col);
        }
    }

    // QR-orthonormalize: drops linearly dependent columns, keeps only rank(seed) vectors.
    let (q, _r) = seed.qr()
        .map_err(|e| FerricError::General(format!("atom seed QR failed: {e}")))?;
    Ok(q)
}

/// Results from a PDEP-RPA calculation.
#[derive(Debug)]
pub struct PdepRpaResult {
    /// RPA correlation energy in Hartree.
    pub e_rpa: f64,
    /// Number of eigenpotentials retained after truncation.
    pub n_eigenpotentials: usize,
    /// Static dielectric eigenvalues λ_α(0), length M.
    pub eigenvalues_static: Vec<f64>,
    /// PDEP eigenpotentials V_α expanded in the RI auxiliary basis (physical coefficients,
    /// after back-transforming from the V^{-1/2}-dressed Davidson basis).
    /// Shape (naux, M). Column α gives the c_α^P such that V_α(r) = Σ_P c_α^P χ_P(r).
    pub eigenpotentials: Array2<f64>,
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

    // Atom-localized seed gives 3·N_atoms scaling when n_desired ≪ naux (PDEP truncation
    // regime). When the caller asks for all naux modes (e.g. trunc_thresh = 0 for
    // apples-to-apples comparison with full RI-RPA), Davidson would have to grow the
    // subspace all the way back up — the identity seed converges faster in that case.
    //
    // Heuristic: when trunc_thresh > 0 AND naux > 4·N_atoms, use atom seed; otherwise
    // identity seed. This keeps the PDEP win for production runs without breaking the
    // full-basis verification path.
    let use_atom_seed =
        config.trunc_thresh > 0.0 && naux > 4 * dfbs.natoms();
    let davidson_result = if use_atom_seed {
        let seed = build_atom_seed(dfbs)?;
        davidson::run_davidson_seeded(
            seed,
            move |v_mat: &Array2<f64>, omega: f64| {
                sternheimer::dielectric_matrix(v_mat, &b_ov_clone, &eps_occ_clone, &eps_vir_clone, omega)
            },
            config.davidson_conv_thresh,
            max_vecs,
            naux,
        )?
    } else {
        davidson::run_davidson_static(
            naux,
            move |v_mat: &Array2<f64>, omega: f64| {
                sternheimer::dielectric_matrix(v_mat, &b_ov_clone, &eps_occ_clone, &eps_vir_clone, omega)
            },
            config.davidson_conv_thresh,
            max_vecs,
            naux,
        )?
    };

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

    // Back-transform from V^{-1/2}-dressed basis to physical aux-basis coefficients:
    // c_α (physical) = V^{-1/2} · V_α (dressed). Used for real-space cube export.
    let eigenpotentials_aux = inter.v_inv_sqrt.dot(&eigenvectors);

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
        eigenpotentials: eigenpotentials_aux,
        quad_freqs,
        quad_weights,
        eigenvalues_freq,
        e_rpa_dft_diag,
    })
}
