//! PDEP-RPA: projector-density-eigenpotential RPA correlation energy.
//!
//! # Threading note
//!
//! The hot path parallelizes over imaginary-frequency quadrature points with
//! rayon. Inside each task, OpenBLAS is invoked for GEMM/SYRK/eigh. On a
//! multi-core machine that gives a *product* of threads (rayon × OpenBLAS),
//! which oversubscribes and hurts wall-clock by 3-5×. For best performance
//! set `OPENBLAS_NUM_THREADS=1` (or `BLIS_NUM_THREADS=1`) so each rayon
//! worker gets a dedicated single-threaded BLAS call.

pub mod boys_localize;
pub mod config;
pub mod davidson;
pub mod diagnostics;
pub mod energy;
pub mod gradient;
pub mod lanczos;
pub mod optimize;
pub mod laplace_chi0;
pub mod properties;
pub mod quadrature;
pub mod screen;
pub mod seeds;
pub mod sternheimer;
pub mod sternheimer_sparse;

pub use lanczos::{run_lanczos_seeded, LanczosResult};

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};
use ferric_scf::ScfResult;
use ndarray::{Array1, Array2};

pub use config::{Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig};
pub use screen::{build_screened_bov, build_screened_bov_boys, ScreenedBov};

/// Allocating wrapper that dispatches the χ₀ kernel (Dense vs Laplace) based on
/// whether a `LaplaceQuadrature` is supplied.
///
/// This is the single seam through which the Davidson/Lanczos callbacks select
/// the backend. `eval_eigenvalues_at_frequencies` has its own (faster) in-place
/// path because it lives in the hot post-Davidson loop where rhs_scaled/out
/// scratch buffers are reused across frequencies.
fn dielectric_apply(
    v_mat: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    laplace: Option<&ferric_quadrature::LaplaceQuadrature>,
) -> Array2<f64> {
    match laplace {
        None => sternheimer::dielectric_matrix(v_mat, b_ov, eps_occ, eps_vir, omega),
        Some(q) => laplace_chi0::dielectric_matrix_laplace(
            v_mat, b_ov, eps_occ, eps_vir, omega, q,
        ),
    }
}

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

/// Build a Davidson/Lanczos seed from a Boys-screened B-tile representation.
///
/// For each Boys-localized occupied i_loc, build a seed column equal to the
/// sum of its tile columns (i.e., Σ_a B^P_{i_loc, a}) scattered back to the
/// full naux index via `p_lists[i_loc]`. Result has up to `nocc_loc` columns;
/// QR drops linearly dependent / zero columns.
///
/// This is the "Boys-as-seed" effect: starts the Krylov subspace from
/// localized-occupied → aux projections, which is where the dominant
/// dielectric eigenmodes live.
fn build_boys_screened_seed(sb: &ScreenedBov) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::QR;

    let naux = sb.naux;
    let nocc_loc = sb.n_occ_loc;
    let mut seed = Array2::<f64>::zeros((naux, nocc_loc));

    for i_loc in 0..nocc_loc {
        let p_list = &sb.p_lists[i_loc];
        let tile = &sb.tiles[i_loc];
        if p_list.is_empty() {
            continue;
        }
        // Sum over virtuals → length-m_i vector.
        let col_sum: Array1<f64> = tile.sum_axis(ndarray::Axis(1));
        for (slot, &p) in p_list.iter().enumerate() {
            seed[(p, i_loc)] = col_sum[slot];
        }
    }

    // QR-orthonormalize; drops null/dependent columns.
    let (q, _r) = seed
        .qr()
        .map_err(|e| FerricError::General(format!("Boys-screened seed QR failed: {e}")))?;
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
    rhf: &ScfResult,
    config: &PdepRpaConfig,
) -> Result<PdepRpaResult, FerricError> {
    // Step 1: Build RI-MO B^P_ia tensor and V^{-1/2}. RPA only needs the
    // occ-vir block; skip the full-MP2 amplitudes/density that the gradient
    // path requires.
    let mp2_cfg = RiMp2Config { frozen_core: config.frozen_core };
    let inter = compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;

    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let naux = inter.naux;
    let first_occ = inter.first_occ;
    let nocc_total = inter.nocc_total;

    // Step 2: Extract orbital energy slices.
    let eps_occ: Vec<f64> = rhf.eps_r()[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = rhf.eps_r()[nocc_total..nocc_total + nvir].to_vec();

    // Step 3: Run Davidson at ω=0.
    let max_vecs = if config.davidson_max_vecs == 0 {
        3 * naux
    } else {
        config.davidson_max_vecs
    };

    let b_ov_clone = b_ov.clone();
    let eps_occ_clone = eps_occ.clone();
    let eps_vir_clone = eps_vir.clone();

    // Build the Laplace quadrature once if the Laplace χ₀ backend was selected.
    // The same `(t_l, w_l)` are reused at every (ω, V) inside the Davidson loop
    // and in the post-Davidson `eval_eigenvalues_at_frequencies` path.
    let laplace_chi0_quad: Option<ferric_quadrature::LaplaceQuadrature> =
        match config.chi0_backend {
            Chi0Backend::Dense => None,
            Chi0Backend::Laplace { n_quad } => Some(
                laplace_chi0::build_laplace_for_gaps(&eps_occ_clone, &eps_vir_clone, n_quad),
            ),
        };
    let laplace_for_davidson = laplace_chi0_quad.clone();

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

    // Optional screened-tile representation (Boys-localized occupied,
    // per-orbital aux-row screening). Only used inside the Davidson/Lanczos
    // matvec closure when `chi0_sparsity = BoysScreened`. The post-Davidson
    // energy integration still uses the dense b_ov path for correctness.
    let screened_bov_opt: Option<ScreenedBov> = match config.chi0_sparsity {
        Chi0Sparsity::Dense => None,
        Chi0Sparsity::BoysScreened { thresh } => {
            let (sb, _boys) = screen::build_screened_bov_boys(
                mol,
                obs,
                dfbs,
                op,
                rhf,
                config.frozen_core,
                thresh,
            )?;
            Some(sb)
        }
    };

    let davidson_result = match (config.eigensolver, screened_bov_opt.as_ref()) {
        // --- Sparse Boys-screened path ---
        (Eigensolver::Davidson, Some(sb)) => {
            // Seed: localized-occupied → aux projection. Columns of the per-
            // orbital tile summed over virtuals and scattered back to full
            // naux via p_lists, then QR-orthonormalized. This is the "Boys
            // as seed" effect — falls out of the screened representation.
            let seed = build_boys_screened_seed(sb)?;
            let eps_vir_seed = eps_vir.clone();
            let sb_ref = sb;
            davidson::run_davidson_seeded(
                seed,
                |v_mat: &Array2<f64>, omega: f64| {
                    sternheimer_sparse::dielectric_matrix_screened(
                        v_mat, sb_ref, &eps_vir_seed, omega,
                    )
                },
                config.davidson_conv_thresh,
                max_vecs,
                naux,
            )?
        }
        (Eigensolver::Lanczos, Some(sb)) => {
            let seed = build_boys_screened_seed(sb)?;
            let block_size = seed.ncols().max(1);
            let max_iter = (max_vecs / block_size).max(8);
            let eps_vir_lz = eps_vir.clone();
            let sb_ref = sb;
            let matvec = |v: &Array2<f64>| -> Array2<f64> {
                sternheimer_sparse::dielectric_apply_screened(v, sb_ref, &eps_vir_lz, 0.0)
            };
            let lz = lanczos::run_lanczos_seeded(
                seed,
                matvec,
                naux,
                max_iter,
                config.davidson_conv_thresh,
            )?;
            davidson::DavidsonResult {
                eigenvalues: lz.eigenvalues,
                eigenvectors: lz.eigenvectors,
            }
        }
        // --- Dense legacy path ---
        (Eigensolver::Davidson, None) => {
            if use_atom_seed {
                let seed = build_atom_seed(dfbs)?;
                let laplace_q = laplace_for_davidson.clone();
                davidson::run_davidson_seeded(
                    seed,
                    move |v_mat: &Array2<f64>, omega: f64| {
                        dielectric_apply(
                            v_mat, &b_ov_clone, &eps_occ_clone, &eps_vir_clone, omega,
                            laplace_q.as_ref(),
                        )
                    },
                    config.davidson_conv_thresh,
                    max_vecs,
                    naux,
                )?
            } else {
                let laplace_q = laplace_for_davidson.clone();
                davidson::run_davidson_static(
                    naux,
                    move |v_mat: &Array2<f64>, omega: f64| {
                        dielectric_apply(
                            v_mat, &b_ov_clone, &eps_occ_clone, &eps_vir_clone, omega,
                            laplace_q.as_ref(),
                        )
                    },
                    config.davidson_conv_thresh,
                    max_vecs,
                    naux,
                )?
            }
        }
        (Eigensolver::Lanczos, None) => {
            let seed = if use_atom_seed {
                build_atom_seed(dfbs)?
            } else {
                Array2::eye(naux)
            };
            let block_size = seed.ncols().max(1);
            let max_iter = (max_vecs / block_size).max(8);
            let matvec = move |v: &Array2<f64>| -> Array2<f64> {
                sternheimer::dielectric_apply(
                    v, &b_ov_clone, &eps_occ_clone, &eps_vir_clone, 0.0,
                )
            };
            let lz = lanczos::run_lanczos_seeded(
                seed,
                matvec,
                naux,
                max_iter,
                config.davidson_conv_thresh,
            )?;
            davidson::DavidsonResult {
                eigenvalues: lz.eigenvalues,
                eigenvectors: lz.eigenvectors,
            }
        }
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

    // Step 6: Evaluate λ_α(iω_k). Dispatch to the Laplace-separable kernel
    // when the user opted in via `chi0_backend`.
    let eigenvalues_freq = match laplace_chi0_quad.as_ref() {
        None => energy::eval_eigenvalues_at_frequencies(
            &eigenvectors, b_ov, &eps_occ, &eps_vir, &quad_freqs,
        ),
        Some(q) => energy::eval_eigenvalues_at_frequencies_laplace(
            &eigenvectors, b_ov, &eps_occ, &eps_vir, &quad_freqs, q,
        ),
    };

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

/// Open-shell PDEP-RPA energy (UHF or ROHF reference).
///
/// Dispatches on `rhf.spin`:
/// * `Restricted`  — error (use [`run_pdep_rpa`] instead).
/// * `Unrestricted` / `RestrictedOpen` — builds per-spin B_ov_α and
///   B_ov_β, runs Davidson on the *summed* dielectric ε̃ = I + Π_α + Π_β,
///   integrates RPA correlation via the same trace-log quadrature.
///
/// First-land scope: Dense χ₀ backend, Davidson eigensolver, identity
/// seed. Boys-localized seeding, Lanczos, Laplace-separable χ₀, and
/// sparse screening are deferred to C8 — the open-shell merge with
/// the scaling stack.
pub fn run_u_pdep_rpa(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &PdepRpaConfig,
) -> Result<PdepRpaResult, FerricError> {
    use ferric_mp2::rimp2::compute_rpa_intermediates_spin;
    use ferric_scf::Spin;

    if matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "run_u_pdep_rpa: use run_pdep_rpa for closed-shell results".into(),
        ));
    }

    let mp2_cfg = RiMp2Config { frozen_core: config.frozen_core };
    let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, true)?;
    let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, false)?;
    let naux = inter_a.naux;
    debug_assert_eq!(inter_a.naux, inter_b.naux);

    let eps_occ_a: Vec<f64> =
        rhf.eps_a()[inter_a.first_occ..inter_a.first_occ + inter_a.nocc].to_vec();
    let eps_vir_a: Vec<f64> =
        rhf.eps_a()[inter_a.nocc_total..inter_a.nocc_total + inter_a.nvir].to_vec();
    // ROHF stores one set of orbital energies for both spins (Guest-Saunders
    // canonicalized α energies serve as β denominators too).
    let eps_b_full: &[f64] = if matches!(rhf.spin, Spin::RestrictedOpen) {
        rhf.eps_a()
    } else {
        rhf.eps_b()
    };
    let eps_occ_b: Vec<f64> =
        eps_b_full[inter_b.first_occ..inter_b.first_occ + inter_b.nocc].to_vec();
    let eps_vir_b: Vec<f64> =
        eps_b_full[inter_b.nocc_total..inter_b.nocc_total + inter_b.nvir].to_vec();

    let max_vecs = if config.davidson_max_vecs == 0 {
        3 * naux
    } else {
        config.davidson_max_vecs
    };

    // Eigensolve. Lanczos is preferred per [[ferric-rpa-status]]; Davidson
    // is kept as fallback to mirror the closed-shell dispatch.
    let b_a = inter_a.b_ov.clone();
    let b_b = inter_b.b_ov.clone();
    let ea_o = eps_occ_a.clone();
    let ea_v = eps_vir_a.clone();
    let eb_o = eps_occ_b.clone();
    let eb_v = eps_vir_b.clone();

    let davidson_result = match config.eigensolver {
        Eigensolver::Lanczos => {
            // Block Lanczos with identity seed. C8 will swap in Boys-localized
            // per-spin seeds; for first-land Dense+identity matches the
            // closed-shell trunc_thresh=0 verification path.
            let seed = Array2::eye(naux);
            let block_size = seed.ncols().max(1);
            let max_iter = (max_vecs / block_size).max(8);
            let matvec = move |v: &Array2<f64>| -> Array2<f64> {
                sternheimer::dielectric_apply_unrestricted(
                    v, &b_a, &ea_o, &ea_v, &b_b, &eb_o, &eb_v, 0.0,
                )
            };
            let lz = lanczos::run_lanczos_seeded(
                seed, matvec, naux, max_iter, config.davidson_conv_thresh,
            )?;
            davidson::DavidsonResult {
                eigenvalues: lz.eigenvalues,
                eigenvectors: lz.eigenvectors,
            }
        }
        Eigensolver::Davidson => davidson::run_davidson_static(
            naux,
            move |v_mat: &Array2<f64>, omega: f64| {
                sternheimer::dielectric_apply_unrestricted(
                    v_mat,
                    &b_a, &ea_o, &ea_v,
                    &b_b, &eb_o, &eb_v,
                    omega,
                )
            },
            config.davidson_conv_thresh,
            max_vecs,
            naux,
        )?,
    };

    let n_keep = davidson_result
        .eigenvalues
        .iter()
        .filter(|&&lam| (lam - 1.0).abs() > config.trunc_thresh)
        .count();
    let n_keep = n_keep.max(1);

    let eigenvalues_static: Vec<f64> = davidson_result.eigenvalues[..n_keep].to_vec();
    let eigenvectors = davidson_result.eigenvectors.slice(ndarray::s![.., ..n_keep]).to_owned();

    let eigenpotentials_aux = inter_a.v_inv_sqrt.dot(&eigenvectors);

    let (quad_freqs, quad_weights) = quadrature::build_quadrature(&config.quadrature);

    let eigenvalues_freq = energy::eval_eigenvalues_at_frequencies_unrestricted(
        &eigenvectors,
        &inter_a.b_ov, &eps_occ_a, &eps_vir_a,
        &inter_b.b_ov, &eps_occ_b, &eps_vir_b,
        &quad_freqs,
    );

    let e_rpa = energy::rpa_correlation_energy(&quad_weights, &eigenvalues_freq);

    Ok(PdepRpaResult {
        e_rpa,
        n_eigenpotentials: n_keep,
        eigenvalues_static,
        eigenpotentials: eigenpotentials_aux,
        quad_freqs,
        quad_weights,
        eigenvalues_freq,
        e_rpa_dft_diag: None,
    })
}
