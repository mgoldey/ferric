//! PDEP-RPA: projector-density-eigenpotential RPA correlation energy.
//!
//! # Lint policy
//!
//! Two clippy lints are allowed crate-wide because the patterns they flag are
//! deliberate in numerical/quantum-chemistry code:
//!
//! - `needless_range_loop`: tensor contractions and MO-index loops
//!   (`for i in 0..3 { for j in 0..3 { t[i][j] ... } }`) read more clearly with
//!   explicit indices than with `iter().enumerate().zip(...)`; the index *is*
//!   the physics (Cartesian component, orbital, atom).
//! - `excessive_precision`: reference tables (minimax quadrature nodes/weights,
//!   Lebedev grids) are transcribed at full source precision on purpose;
//!   trimming digits to f64's last representable place is lossy churn.
//!
//! # Threading note
//!
//! The hot path parallelizes over imaginary-frequency quadrature points with
//! rayon. Inside each task, OpenBLAS is invoked for GEMM/SYRK/eigh. On a
//! multi-core machine that gives a *product* of threads (rayon × OpenBLAS),
//! which oversubscribes and hurts wall-clock by 3-5×. For best performance
//! set `OPENBLAS_NUM_THREADS=1` (or `BLIS_NUM_THREADS=1`) so each rayon
//! worker gets a dedicated single-threaded BLAS call.
#![allow(clippy::needless_range_loop)] // tensor/MO-index loops read clearer with explicit indices
#![allow(clippy::excessive_precision)] // reference tables transcribed at full source precision

pub mod ao_rpa;
pub mod boys_localize;
pub mod budget;
pub mod channel;
pub mod config;
pub mod davidson;
pub mod diagnostics;
pub mod dispersion;
pub mod energy;
pub mod gradient;
pub mod lanczos;
pub mod optimize;
pub mod laplace_chi0;
#[cfg(feature = "mpi")]
pub mod mpi_rpa;
pub mod pno;
pub mod properties;
pub mod quadrature;
pub mod rs_mp2_rpa;
pub mod screen;
pub mod seeds;
pub mod sternheimer;
pub mod sternheimer_sparse;
pub mod timing;

pub use lanczos::{run_lanczos_full_rank, run_lanczos_seeded, LanczosResult};
pub use rs_mp2_rpa::{rs_mp2_lr_rpa, RsMp2RpaConfig, RsMp2RpaFormulation, RsMp2RpaResult};

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};
use ferric_scf::ScfResult;
use ndarray::{Array1, Array2};

pub use config::{Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig};
pub use dispersion::{
    casimir_polder_c6, pdep_dynamic_polarizability, ts_dynamic_polarizability, C6Result,
    DispersionPartition, DynamicPolarizability,
};
pub use screen::{build_screened_bov, build_screened_bov_boys, ScreenedBov};

/// Crate-wide serialization for tests that SET or transitively READ the
/// process-global `FERRIC_MEM_BUDGET_GB` / `FERRIC_LANCZOS_PANEL` / legacy
/// budget env vars. `ao_rpa`, `lanczos`, and `rs_mp2_rpa` each independently
/// mutate one of these (a private per-module lock cannot stop a cross-module
/// race — e.g. `lanczos_panel_width_honors_explicit_budget_argument` asserts
/// the NO-env-set default (256) via `lanczos::lanczos_panel_width`, which
/// transitively calls `ferric_core::memory::resolve_budget`; a concurrently-
/// running test elsewhere in this crate that `set_var`s `FERRIC_MEM_BUDGET_GB`
/// mid-flight flips that assertion — observed twice under parallel test
/// execution). Mirrors `ferric_dft::TEST_BUDGET_ENV_LOCK`'s exact pattern;
/// poisoning is tolerated via `into_inner` at use sites.
#[cfg(test)]
pub(crate) static TEST_BUDGET_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    /// Davidson eigenvectors in the V^{-1/2}-dressed RI basis, shape (naux, M).
    /// These are the raw Davidson/Lanczos output before back-transformation and
    /// are needed for the PDEP-truncated polarizability: dressed_Uᵀ · B̃ gives
    /// the correct projection onto the PDEP subspace.
    pub dressed_eigenvectors: Array2<f64>,
    /// Imaginary-frequency quadrature points ω_k.
    pub quad_freqs: Vec<f64>,
    /// Quadrature weights w_k.
    pub quad_weights: Vec<f64>,
    /// λ_α(iω_k) tensor, shape (N_quad, M).
    pub eigenvalues_freq: Array2<f64>,
    /// Per-frequency dynamic inverse-dielectric matrices W̃_d(iω_k) = ε̃_proj⁻¹ − I
    /// in the *fixed* PDEP eigenvector basis, length N_quad, each shape (M, M).
    ///
    /// `None` for the Laplace χ₀ path (not yet wired). Required by ferric-gw's
    /// Σ_c: the diagonal `eigenvalues_freq` weights are computed in a per-ω
    /// rotated eigenbasis and are NOT consistent with the static B̃ projection,
    /// so the GW self-energy must use these full matrices instead.
    pub inv_dielectric_freq: Option<Vec<Array2<f64>>>,
    /// RI-dRPA sanity-check energy (None unless run_diagnostics=true).
    pub e_rpa_dft_diag: Option<f64>,
    /// Whether the static-dielectric eigensolve (Davidson or Lanczos, per
    /// `config.eigensolver`) met its residual-norm convergence tolerance.
    /// `false` means the eigenvalues/eigenpotentials above are the
    /// eigensolver's best-effort Ritz pairs after exhausting its iteration
    /// budget, not verified eigenpairs — Davidson itself hard-errors instead
    /// of ever returning `false` here (see [`davidson::DavidsonResult`]), so
    /// in practice only the Lanczos path (the default eigensolver) can set
    /// this `false`. Callers that need a hard guarantee should check this
    /// explicitly; the CLI and Python bindings only warn.
    pub eigensolver_converged: bool,
}

/// Pre-flight peak-memory gate shared by [`run_pdep_rpa`] and the eigensolve
/// entry points. Cheap shape values only (nelec/nbasis accessors, no
/// ERI/GEMM work) so this runs before ANY large allocation. See `budget.rs`
/// for what the estimate covers; this is the closed-shell (`compute_rpa_
/// intermediates`) formula for nocc/nvir.
fn preflight_check_closed_shell(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    config: &PdepRpaConfig,
    label_prefix: &str,
) -> Result<(), FerricError> {
    use ferric_mp2::rimp2::active_occ;
    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let nocc_total = (mol.nelec() as usize) / 2;
    let nocc = active_occ(nocc_total, config.frozen_core)?;
    let nvir = nbas.saturating_sub(nocc_total);
    let n_workers = rayon::current_num_threads().max(1);
    let n_keep = naux; // trunc_thresh unknown pre-eigensolve; conservative upper bound
    let est = budget::estimate_peak_bytes(budget::PeakEstimateShape {
        naux, nocc, nvir,
        n_quad: config.quadrature.n_points,
        n_workers,
        n_keep,
        grid: None,
    });
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes);
    ferric_core::memory::check_alloc(
        &format!(
            "{label_prefix} preflight (naux={naux}, nocc={nocc}, nvir={nvir}, n_workers={n_workers})"
        ),
        est,
        budget_bytes,
    )
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
    preflight_check_closed_shell(mol, obs, dfbs, config, "PDEP-RPA")?;
    // Step 1: Build RI-MO B^P_ia tensor and V^{-1/2}. RPA only needs the
    // occ-vir block; skip the full-MP2 amplitudes/density that the gradient
    // path requires.
    let mp2_cfg = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes };
    let _t_setup = crate::timing::Stage::start("pdep:rpa_intermediates(ERI3+metric+MOtransform)");
    let inter = compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    _t_setup.end();
    run_pdep_rpa_from_intermediates(&inter, mol, obs, dfbs, op, rhf, config)
}

/// Output of the Davidson/Lanczos eigensolve stage (Steps 1-5 of
/// [`run_pdep_rpa_from_intermediates`]) — everything needed to enter the
/// imaginary-frequency quadrature loop (Steps 6-8), but not the
/// frequency-dependent results themselves.
///
/// Factored out so [`crate::mpi_rpa::run_pdep_rpa_mpi`] can share this
/// (replicated, NOT MPI-distributed — see that module's docs for why) setup
/// stage verbatim instead of re-deriving it or duplicating
/// `run_pdep_rpa_from_intermediates`'s eigensolver dispatch, and so the two
/// entry points cannot silently drift apart on the eigensolve logic.
pub(crate) struct EigensolveStage {
    pub eigenvectors: Array2<f64>,
    pub eigenvalues_static: Vec<f64>,
    pub eigenpotentials_aux: Array2<f64>,
    pub n_keep: usize,
    pub quad_freqs: Vec<f64>,
    pub quad_weights: Vec<f64>,
    pub eigensolver_converged: bool,
    /// `Some` only for `Chi0Backend::Laplace` — callers restricted to the
    /// Dense backend (e.g. `run_pdep_rpa_mpi`) can assume this is `None`.
    pub laplace_chi0_quad: Option<ferric_quadrature::LaplaceQuadrature>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_pdep_rpa_eigensolve(
    inter: &ferric_mp2::rimp2::RpaIntermediates,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &PdepRpaConfig,
) -> Result<EigensolveStage, FerricError> {
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
    // NOTE (M9 memory): the default subspace cap 3·naux bounds the Davidson
    // basis at naux·(3·naux)·8 bytes (≈346 MB at dimer/aTZ naux≈3800). This is
    // borderline but only pays on the Davidson path — Lanczos is the default
    // solver (config.eigensolver) and never grows a 3·naux basis. If Davidson
    // becomes the hot path at larger naux, gate this multiplier on
    // memory_budget_bytes / (naux·8) instead of the fixed 3×.
    let max_vecs = if config.eigensolver_max_vecs == 0 {
        3 * naux
    } else {
        config.eigensolver_max_vecs
    };

    // NOTE (2026-07-21 memory fix): these used to be `b_ov.clone()` /
    // `eps_occ.clone()` / `eps_vir.clone()` so the Davidson/Lanczos matvec
    // closures below could `move` owned copies into themselves. None of
    // run_davidson_seeded/run_davidson_static/run_lanczos_full_rank_budgeted
    // require `'static` (they call `matvec` synchronously within their own
    // call frame, never spawning a thread or storing the closure past the
    // call — confirmed by reading davidson.rs/lanczos.rs), and `inter` (the
    // source of `b_ov`) is borrowed for this whole function's body (it's
    // read again below at `inter.v_inv_sqrt.dot(...)`), so the closures can
    // safely capture plain references instead. At benzene aug-cc-pVQZ scale
    // (naux≈2976, nov≈61740) `b_ov.clone()` alone was ~1.4 GB — a pure
    // aliasing change, zero numerical effect (see `dense_legacy_path_clone_
    // elimination_is_bit_identical` regression test in this module).
    let b_ov_ref = b_ov;
    let eps_occ_ref = &eps_occ;
    let eps_vir_ref = &eps_vir;

    // Build the Laplace quadrature once if the Laplace χ₀ backend was selected.
    // The same `(t_l, w_l)` are reused at every (ω, V) inside the Davidson loop
    // and in the post-Davidson `eval_eigenvalues_at_frequencies` path.
    let laplace_chi0_quad: Option<ferric_quadrature::LaplaceQuadrature> =
        match config.chi0_backend {
            Chi0Backend::Dense => None,
            Chi0Backend::Laplace { n_quad } => Some(
                laplace_chi0::build_laplace_for_gaps(eps_occ_ref, eps_vir_ref, n_quad)?,
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
    // Resolve Auto → Dense/BoysScreened by atom count (boys-screening-crossover).
    let resolved_sparsity = config.chi0_sparsity.resolve(mol.atoms.len());
    if let Chi0Sparsity::Auto { boys_thresh, atom_cutoff, .. } = config.chi0_sparsity {
        let picked = match resolved_sparsity {
            Chi0Sparsity::BoysScreened { thresh, dist_cutoff } => {
                format!("BoysScreened{{{thresh:e}, dist_cutoff {dist_cutoff}}}")
            }
            _ => "Dense".to_string(),
        };
        let cmp = if mol.atoms.len() >= atom_cutoff { "≥" } else { "<" };
        eprintln!(
            "chi0_sparsity auto: {} atoms {cmp} cutoff {atom_cutoff} → {picked} (boys_thresh {boys_thresh:e})",
            mol.atoms.len()
        );
    }
    let _t_screen_build = crate::timing::Stage::start("pdep:boys_screen_build(localize+screened_3idx)");
    let screened_bov_opt: Option<ScreenedBov> = match resolved_sparsity {
        Chi0Sparsity::Dense => None,
        Chi0Sparsity::BoysScreened { thresh, dist_cutoff } => {
            let (sb, _boys) = screen::build_screened_bov_boys(
                mol,
                obs,
                dfbs,
                op,
                rhf,
                config.frozen_core,
                thresh,
                dist_cutoff,
            )?;
            Some(sb)
        }
        // `resolve` never returns Auto; this arm is unreachable but keeps the
        // match exhaustive without a catch-all that could hide a future variant.
        Chi0Sparsity::Auto { .. } => unreachable!("resolve() collapses Auto"),
    };
    _t_screen_build.end();

    let _t_eig = crate::timing::Stage::start("pdep:eigensolve(Davidson/Lanczos)");
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
                config.eigensolver_conv_thresh,
                max_vecs,
                naux,
                false,
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
                config.eigensolver_conv_thresh,
                config.verbose,
            )?;
            if !lz.converged {
                eprintln!(
                    "warning: Lanczos eigensolve did NOT converge (max Ritz residual \
                     {:.3e} > {:.3e} after {max_iter} block iterations); PDEP \
                     eigenpotentials and the RPA energy built on them are best-effort",
                    lz.max_resid, config.eigensolver_conv_thresh
                );
            }
            davidson::DavidsonResult {
                eigenvalues: lz.eigenvalues,
                eigenvectors: lz.eigenvectors,
                converged: lz.converged,
            }
        }
        // --- Dense legacy path ---
        (Eigensolver::Davidson, None) => {
            if use_atom_seed {
                let seed = build_atom_seed(dfbs)?;
                let laplace_q = laplace_for_davidson.as_ref();
                davidson::run_davidson_seeded(
                    seed,
                    |v_mat: &Array2<f64>, omega: f64| {
                        dielectric_apply(
                            v_mat, b_ov_ref, eps_occ_ref, eps_vir_ref, omega,
                            laplace_q,
                        )
                    },
                    config.eigensolver_conv_thresh,
                    max_vecs,
                    naux,
                    false,
                )?
            } else {
                let laplace_q = laplace_for_davidson.as_ref();
                davidson::run_davidson_static(
                    naux,
                    |v_mat: &Array2<f64>, omega: f64| {
                        dielectric_apply(
                            v_mat, b_ov_ref, eps_occ_ref, eps_vir_ref, omega,
                            laplace_q,
                        )
                    },
                    config.eigensolver_conv_thresh,
                    max_vecs,
                    naux,
                    false,
                )?
            }
        }
        (Eigensolver::Lanczos, None) => {
            // Full-rank identity seed. Block Lanczos with the naux-wide identity
            // seed collapses to a single dense eigh of A = ε̃(0) = I + Π (the
            // atom-localized seed is a Davidson-only optimization: Lanczos is
            // confined to the Krylov span of its seed, and a non-spanning atom
            // seed produced unphysical negative ghost Ritz values that broke FD
            // gradients). `run_lanczos_full_rank` reproduces that single-eigh
            // result exactly, but assembles A in column-panels so the matvec
            // never materializes the whole naux-wide block at once — the
            // memory-bound benzene/aTZ jobs (atz-benzene-rpa-memory-bound) drop
            // from concurrency-1 to concurrency-N per box. Eigenpairs match the
            // identity-seed Lanczos (hence Davidson) to LAPACK precision.
            let nov = nocc * nvir;
            let matvec = |v: &Array2<f64>| -> Array2<f64> {
                sternheimer::dielectric_apply(
                    v, b_ov_ref, eps_occ_ref, eps_vir_ref, 0.0,
                )
            };
            let lz = lanczos::run_lanczos_full_rank_budgeted(
                naux, nov, matvec, naux, config.memory_budget_bytes,
            )?;
            davidson::DavidsonResult {
                eigenvalues: lz.eigenvalues,
                eigenvectors: lz.eigenvectors,
                converged: lz.converged,
            }
        }
    };
    _t_eig.end();
    // Stage-seam RSS safety net (Item 3): observability only, never a hard
    // error. The Davidson/Lanczos eigensolve is the other large co-resident
    // allocation `budget.rs::estimate_peak_bytes` accounts for (the assembled
    // naux×naux dielectric + its eigh output) — check actual RSS right after
    // it fully drains, before the (usually much smaller) truncation/quadrature
    // stages below.
    ferric_core::memory::warn_if_rss_over(
        "PDEP-RPA eigensolve stage complete",
        ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes),
        1.1,
    );

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

    Ok(EigensolveStage {
        eigenvectors,
        eigenvalues_static,
        eigenpotentials_aux,
        n_keep,
        quad_freqs,
        quad_weights,
        eigensolver_converged: davidson_result.converged,
        laplace_chi0_quad,
    })
}

/// dRPA energy from PRE-BUILT RI intermediates, skipping the (P|op|ia) transform.
///
/// See the original doc comment (still accurate): this composes
/// [`run_pdep_rpa_eigensolve`] (Steps 1-5, replicated/local eigensolve) with
/// the serial imaginary-frequency quadrature evaluation (Steps 6-8). The
/// eigensolve helper is shared verbatim with
/// [`crate::mpi_rpa::run_pdep_rpa_mpi`], which instead routes Steps 6-8
/// through the MPI-distributed frequency evaluators — the two entry points
/// cannot silently diverge on the (replicated) Davidson/Lanczos setup because
/// they call the exact same function for it.
#[allow(clippy::too_many_arguments)]
pub fn run_pdep_rpa_from_intermediates(
    inter: &ferric_mp2::rimp2::RpaIntermediates,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &PdepRpaConfig,
) -> Result<PdepRpaResult, FerricError> {
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let first_occ = inter.first_occ;
    let nocc_total = inter.nocc_total;
    let eps_occ: Vec<f64> = rhf.eps_r()[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = rhf.eps_r()[nocc_total..nocc_total + nvir].to_vec();

    let stage = run_pdep_rpa_eigensolve(inter, mol, obs, dfbs, op, rhf, config)?;
    let eigenvectors = stage.eigenvectors;
    let quad_freqs = stage.quad_freqs;
    let quad_weights = stage.quad_weights;

    // Step 6: Evaluate λ_α(iω_k). Dispatch to the Laplace-separable kernel
    // when the user opted in via `chi0_backend`.
    let _t_quad = crate::timing::Stage::start("pdep:freq_quad(lambda+invdielectric)");
    // Energy-only fast path: the correlation energy needs only
    // Σ_α [ln λ_α + (1 − λ_α)] = ln det(ε) + tr(I − ε), which LU (dgetrf)
    // gives for ~half the FLOPs of the eigenvalues-only dsyevd. Taken when the
    // caller opts out of `eigenvalues_freq` (see `need_eigenvalues_freq`) AND
    // nothing else downstream needs a per-frequency diagonalization. The
    // Laplace-χ₀ backend has no log-det sibling, so it keeps the eigen path.
    let logdet_ok = !config.need_eigenvalues_freq
        && !config.need_inv_dielectric_freq
        && stage.laplace_chi0_quad.is_none();
    let trace_log_summands = if logdet_ok {
        Some(energy::eval_trace_log_summands_budgeted(
            &eigenvectors, b_ov, &eps_occ, &eps_vir, &quad_freqs, config.memory_budget_bytes,
        )?)
    } else {
        None
    };
    // Empty (0 × 0) when the log-det path ran: the field is then not meaningful
    // and the caller asked not to have it. Every consumer of `eigenvalues_freq`
    // must set `need_eigenvalues_freq: true` (the default).
    let eigenvalues_freq = if logdet_ok {
        Array2::<f64>::zeros((0, 0))
    } else {
        match stage.laplace_chi0_quad.as_ref() {
            None => energy::eval_eigenvalues_at_frequencies_budgeted(
                &eigenvectors, b_ov, &eps_occ, &eps_vir, &quad_freqs, config.memory_budget_bytes,
            )?,
            Some(q) => energy::eval_eigenvalues_at_frequencies_laplace(
                &eigenvectors, b_ov, &eps_occ, &eps_vir, &quad_freqs, q,
            )?,
        }
    };

    // Step 6b: Per-frequency full inverse-dielectric matrices in the PDEP basis
    // (for the GW self-energy; not needed by the RPA energy). Only the dense
    // (non-Laplace) χ₀ path is wired here. Gated on `need_inv_dielectric_freq`
    // so energy-only runs never materialize the nquad × M² stack (~1.85 GB at
    // dimer/aTZ scale) — GW/BSE/property callers set the flag (M9).
    let inv_dielectric_freq = match (config.need_inv_dielectric_freq, stage.laplace_chi0_quad.as_ref()) {
        (true, None) => Some(energy::eval_inv_dielectric_matrices(
            &eigenvectors, b_ov, &eps_occ, &eps_vir, &quad_freqs,
        )?),
        _ => None,
    };

    _t_quad.end();
    // Stage-seam RSS safety net (Item 3): observability only, never a hard
    // error. This is the stage the memory incident's post-mortem flagged as
    // budget-blind — the per-worker frequency-quadrature scratch scales with
    // rayon thread count (see energy.rs/sternheimer.rs's panelled fix above),
    // so this checks whether actual RSS stayed near the preflight estimate
    // once that stage has fully drained.
    ferric_core::memory::warn_if_rss_over(
        "PDEP-RPA freq_quad stage complete",
        ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes),
        1.1,
    );

    // Step 7: Integrate RPA correlation energy. Both branches evaluate the same
    // quantity; the log-det one just never formed the eigenvalues.
    let e_rpa = match trace_log_summands.as_ref() {
        Some(s) => energy::rpa_correlation_energy_from_summands(&quad_weights, s),
        None => energy::rpa_correlation_energy(&quad_weights, &eigenvalues_freq),
    };

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
        n_eigenpotentials: stage.n_keep,
        eigenvalues_static: stage.eigenvalues_static,
        eigenpotentials: stage.eigenpotentials_aux,
        dressed_eigenvectors: eigenvectors,
        quad_freqs,
        quad_weights,
        eigenvalues_freq,
        inv_dielectric_freq,
        e_rpa_dft_diag,
        eigensolver_converged: stage.eigensolver_converged,
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

    // Pre-flight peak-memory gate (M2-style fail-fast, see budget.rs). Cheap
    // shape values only. Open-shell replicates compute_rpa_intermediates_spin's
    // internal nocc_total/nocc/nvir formula for BOTH spins and estimates each
    // spin's RpaIntermediates build additively (they are NOT built
    // concurrently below — inter_a then inter_b, sequentially — but both
    // b_ov's are retained afterward for the spin-summed dielectric, so the
    // conservative sum-of-both-spins estimate matches what stays resident).
    {
        let nbas = obs.nbasis();
        let naux = dfbs.nbasis();
        let nelec_total = mol.nelec() as usize;
        let two_s = mol.multiplicity as i32 - 1;
        let nocc_total_a = ((nelec_total as i32 + two_s) / 2) as usize;
        let nocc_total_b = ((nelec_total as i32 - two_s) / 2) as usize;
        let nocc_a = ferric_mp2::rimp2::active_occ(nocc_total_a, config.frozen_core)?;
        let nocc_b = ferric_mp2::rimp2::active_occ(nocc_total_b, config.frozen_core)?;
        let nvir_a = nbas.saturating_sub(nocc_total_a);
        let nvir_b = nbas.saturating_sub(nocc_total_b);
        let n_workers = rayon::current_num_threads().max(1);
        let n_keep = naux;
        let est_a = budget::estimate_peak_bytes(budget::PeakEstimateShape {
            naux, nocc: nocc_a, nvir: nvir_a, n_quad: config.quadrature.n_points, n_workers, n_keep,
            grid: None,
        });
        let est_b = budget::estimate_peak_bytes(budget::PeakEstimateShape {
            naux, nocc: nocc_b, nvir: nvir_b, n_quad: config.quadrature.n_points, n_workers, n_keep,
            grid: None,
        });
        let est = est_a.saturating_add(est_b);
        let budget_bytes = ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes);
        ferric_core::memory::check_alloc(
            &format!(
                "U-PDEP-RPA preflight (naux={naux}, nocc_a={nocc_a}, nvir_a={nvir_a}, \
                 nocc_b={nocc_b}, nvir_b={nvir_b}, n_workers={n_workers})"
            ),
            est,
            budget_bytes,
        )?;
    }

    let mp2_cfg = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes };
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

    let max_vecs = if config.eigensolver_max_vecs == 0 {
        3 * naux
    } else {
        config.eigensolver_max_vecs
    };

    // Build per-spin Laplace quadratures if the Laplace χ₀ backend is
    // selected. Each spin has its own gap range so we keep two quadratures.
    let laplace_pair: Option<(ferric_quadrature::LaplaceQuadrature, ferric_quadrature::LaplaceQuadrature)> =
        match config.chi0_backend {
            Chi0Backend::Dense => None,
            Chi0Backend::Laplace { n_quad } => {
                let qa = laplace_chi0::build_laplace_for_gaps(&eps_occ_a, &eps_vir_a, n_quad)?;
                let qb = if eps_occ_b.is_empty() {
                    // Empty spin channel: build a degenerate quadrature; it
                    // never gets used in the per-spin accumulator (early-out).
                    qa.clone()
                } else {
                    laplace_chi0::build_laplace_for_gaps(&eps_occ_b, &eps_vir_b, n_quad)?
                };
                Some((qa, qb))
            }
        };

    // Eigensolve. Lanczos is preferred per [[ferric-rpa-status]]; Davidson
    // is kept as fallback to mirror the closed-shell dispatch.
    let b_a = inter_a.b_ov.clone();
    let b_b = inter_b.b_ov.clone();
    let ea_o = eps_occ_a.clone();
    let ea_v = eps_vir_a.clone();
    let eb_o = eps_occ_b.clone();
    let eb_v = eps_vir_b.clone();
    let lap_for_solver = laplace_pair.clone();

    let davidson_result = match (config.eigensolver, lap_for_solver) {
        (Eigensolver::Lanczos, lap_opt) => {
            // Full-rank identity seed → single dense eigh of the spin-summed
            // ε̃_U(0) = I + Π_α + Π_β, assembled in column-panels to cap the
            // matvec's transient footprint (see run_lanczos_full_rank; closed-
            // shell arm for the memory rationale). Eigenpairs match the prior
            // identity-seed Lanczos to LAPACK precision.
            let nov = inter_a.nocc * inter_a.nvir + inter_b.nocc * inter_b.nvir;
            let lap = lap_opt;
            let matvec = move |v: &Array2<f64>| -> Array2<f64> {
                let chan_a = channel::RpaChannel::new(&b_a, &ea_o, &ea_v);
                let chan_b = channel::RpaChannel::new(&b_b, &eb_o, &eb_v);
                match &lap {
                    None => sternheimer::dielectric_apply_unrestricted(v, &chan_a, &chan_b, 0.0),
                    Some((qa, qb)) => laplace_chi0::dielectric_matrix_laplace_unrestricted(
                        v, &chan_a, qa, &chan_b, qb, 0.0,
                    ),
                }
            };
            let lz = lanczos::run_lanczos_full_rank_budgeted(
                naux, nov, matvec, naux, config.memory_budget_bytes,
            )?;
            davidson::DavidsonResult {
                eigenvalues: lz.eigenvalues,
                eigenvectors: lz.eigenvectors,
                converged: lz.converged,
            }
        }
        (Eigensolver::Davidson, lap_opt) => {
            let lap = lap_opt;
            davidson::run_davidson_static(
                naux,
                move |v_mat: &Array2<f64>, omega: f64| {
                    let chan_a = channel::RpaChannel::new(&b_a, &ea_o, &ea_v);
                    let chan_b = channel::RpaChannel::new(&b_b, &eb_o, &eb_v);
                    match &lap {
                        None => sternheimer::dielectric_apply_unrestricted(v_mat, &chan_a, &chan_b, omega),
                        Some((qa, qb)) => laplace_chi0::dielectric_matrix_laplace_unrestricted(
                            v_mat, &chan_a, qa, &chan_b, qb, omega,
                        ),
                    }
                },
                config.eigensolver_conv_thresh,
                max_vecs,
                naux,
                false,
            )?
        }
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

    let freq_chan_a = channel::RpaChannel::new(&inter_a.b_ov, &eps_occ_a, &eps_vir_a);
    let freq_chan_b = channel::RpaChannel::new(&inter_b.b_ov, &eps_occ_b, &eps_vir_b);
    let eigenvalues_freq = match laplace_pair.as_ref() {
        None => energy::eval_eigenvalues_at_frequencies_unrestricted(
            &eigenvectors, &freq_chan_a, &freq_chan_b, &quad_freqs,
        )?,
        Some((qa, qb)) => energy::eval_eigenvalues_at_frequencies_laplace_unrestricted(
            &eigenvectors,
            &freq_chan_a, qa,
            &freq_chan_b, qb,
            &quad_freqs,
        )?,
    };

    let e_rpa = energy::rpa_correlation_energy(&quad_weights, &eigenvalues_freq);

    // Gated on `need_inv_dielectric_freq` (M9): only U-GW consumes the
    // nquad × M² inverse-dielectric stack; energy-only runs skip it.
    let inv_dielectric_freq = match (config.need_inv_dielectric_freq, laplace_pair.as_ref()) {
        (true, None) => Some(energy::eval_inv_dielectric_matrices_unrestricted(
            &eigenvectors, &freq_chan_a, &freq_chan_b, &quad_freqs,
        )?),
        _ => None,
    };

    Ok(PdepRpaResult {
        e_rpa,
        n_eigenpotentials: n_keep,
        eigenvalues_static,
        eigenpotentials: eigenpotentials_aux,
        dressed_eigenvectors: eigenvectors,
        quad_freqs,
        quad_weights,
        eigenvalues_freq,
        inv_dielectric_freq,
        e_rpa_dft_diag: None,
        eigensolver_converged: davidson_result.converged,
    })
}
