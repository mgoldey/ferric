//! MPI-distributed RPA imaginary-frequency quadrature (T10, RPA half).
//!
//! ## Why frequency-point round-robin, not aux-band striping (T8/T9's lever)
//!
//! T8 (`ferric-scf::df_j`/`df_k`) and T9 (`ferric-mp2::mpi_rimp2`) both
//! distribute by splitting the `naux`-sized auxiliary-basis axis of a B
//! tensor into contiguous bands, because in both cases the B tensor itself
//! (`(naux, nao²)` or `(naux, nocc·nvir)`) is the dominant memory footprint
//! and a genuine per-rank subset of its ROWS can be built and held.
//!
//! RPA's post-eigensolve frequency loop
//! ([`crate::energy::eval_eigenvalues_at_frequencies`] /
//! [`crate::energy::eval_inv_dielectric_matrices`]) does not have that shape.
//! Each quadrature point ω_k needs the FULL projection `y = Vᵀ·B_ov` (shape
//! `(m, nov)`, m = number of retained PDEP eigenpotentials) — there is no way
//! to hand one rank only some of `y`'s rows/columns and still evaluate a
//! single ω_k's dielectric build, because the eigh/inverse at each ω touches
//! the whole `(m×m)` projected matrix. Banding `y` (or the upstream `b_ov`)
//! the way T8/T9 band `B` would not shrink anything here — every rank would
//! still need the full tensor to process even one frequency. `b_ov` is
//! already fully assembled (replicated on every rank) by the time this
//! module's functions run, exactly as it was pre-T10 (see module docs on
//! [`run_pdep_rpa_mpi`] for what remains undistributed).
//!
//! What DOES have T8/T9's "disjoint, embarrassingly parallel, summed at the
//! end" shape is the **frequency axis itself**: `rayon` already parallelizes
//! `eval_eigenvalues_at_frequencies`/`eval_inv_dielectric_matrices` over
//! `quad_freqs` because every ω_k's dielectric build + eigh/inverse is fully
//! independent of every other ω_k. This is the same shell-quartet
//! round-robin idiom `direct_j.rs`/`direct_k.rs` use for MPI (`mpi.md`
//! Section 2), applied to frequency points instead of shell quartets: split
//! `0..n_quad` round-robin across ranks (`k % size == rank`), have each rank
//! run its OWN quadrature points through the same dielectric-build + eigh/inv
//! kernel the serial function uses (rayon still fills cores within a rank),
//! then `Allreduce(sum)` a zero-padded buffer (every non-owned row is 0.0) to
//! reassemble the full per-frequency arrays on every rank. This is a plain
//! LINEAR reduction — unlike RI-MP2's `(ia|jb)²` coupling, no cross-frequency
//! term ever needs a partial-rank result before it's assembled, so ONE
//! `Allreduce(sum)` (not DF-J's/RI-MP2's two-stage pattern) suffices.
//!
//! `inv_dielectric_freq` is `Vec<Array2<f64>>` (one M×M matrix per ω_k)
//! rather than a flat `Array2`; it is flattened into a zero-padded
//! `(n_quad, m*m)` buffer for the same Allreduce treatment, then unflattened
//! back into `Vec<Array2<f64>>`.
//!
//! ## What is NOT distributed here
//!
//! The Davidson/Lanczos eigensolve that produces `eigenvectors`
//! ([`crate::run_pdep_rpa_eigensolve`], Steps 1-5 of
//! [`crate::run_pdep_rpa_from_intermediates`]) is unchanged and still runs
//! identically (replicated) on every rank. It operates on the aux-basis
//! dielectric built from `b_ov`, which is itself still fully replicated per
//! rank (the RI 3-index transform, `compute_rpa_intermediates`, does not
//! take an MPI context) — exactly the state `docs/superpowers/mpi.md`
//! Section 4 described pre-T10. Distributing the eigensolve itself (or the
//! upstream RI transform) is a separate, larger lever that would need
//! Davidson/Lanczos restructured around a distributed matvec; this module
//! only distributes the compute-bound frequency loop that CONSUMES the
//! eigensolve's small `(naux × M)` output — that loop is what
//! `docs/superpowers/mpi.md`'s Section 4 named as remaining ("RPA and GW
//! (T10) are still undistributed").
//!
//! ## GW inherits the W-construction half of this for free
//!
//! `ferric-gw`'s Σ_c evaluation consumes `PdepRpaResult.inv_dielectric_freq`
//! (see `ferric-gw/src/sigma.rs`/`u_sigma.rs`). Since [`run_pdep_rpa_mpi`]
//! fills that field via [`eval_inv_dielectric_matrices_mpi`], any caller that
//! builds its `PdepRpaResult` via [`run_pdep_rpa_mpi`] instead of
//! [`crate::run_pdep_rpa`] gets an MPI-distributed W for free. The
//! GW-specific axis named in the T10 task brief — distributing the per-MO
//! `solve_qp_for_mo`/Σ_c(iω) Padé-continuation loop in `ferric-gw::sigma`
//! itself (currently rayon-parallel over MOs, not MPI-distributed) — is a
//! SEPARATE, NOT-yet-done piece; see the T10 report / `mpi.md` for what
//! remains.

#[cfg(feature = "mpi")]
mod inner {
    use crate::energy::dielectric_lapack_err;
    use crate::sternheimer::{build_scale_factors, dielectric_matrix_from_projection};
    use ferric_core::parallel::ParallelContext;
    use ferric_core::FerricError;
    use ferric_integrals::blas_threads::with_blas_threads;
    use ndarray::Array2;
    use ndarray_linalg::Inverse;
    use rayon::prelude::*;

    /// MPI-distributed variant of [`crate::energy::eval_eigenvalues_at_frequencies`].
    ///
    /// Splits `0..n_quad` round-robin across ranks (`k % size == rank`), runs
    /// this rank's own frequency subset through the same
    /// dielectric-projection + eigh kernel the serial function uses (rayon
    /// inside this rank's subset), then `Allreduce(sum)`s a zero-padded
    /// `(n_quad, m)` buffer so every rank ends up holding the full array —
    /// algebraically exact (each row is written by exactly one rank and zero
    /// everywhere else, so the sum reproduces the full matrix), not an
    /// approximation. At `ctx.size == 1` every `k % 1 == 0` is true, so every
    /// row is computed locally by this one rank and the Allreduce is a no-op
    /// sum of one nonzero contribution per row — byte-identical to
    /// [`crate::energy::eval_eigenvalues_at_frequencies`] (same GEMM/eigh
    /// call per frequency, same row assembly order).
    pub fn eval_eigenvalues_at_frequencies_mpi(
        ctx: &ParallelContext,
        eigenvectors: &Array2<f64>,
        b_ov: &Array2<f64>,
        eps_occ: &[f64],
        eps_vir: &[f64],
        quad_freqs: &[f64],
    ) -> Result<Array2<f64>, FerricError> {
        let n_quad = quad_freqs.len();
        let m = eigenvectors.ncols();
        let size = ctx.size.max(1);
        let rank = ctx.rank;

        // Frequency-independent projection, identical to the serial path.
        let y = eigenvectors.t().dot(b_ov);

        // This rank's owned frequency indices (round-robin, mirrors the
        // direct-JK shell-quartet convention: idx % size == rank).
        let my_idx: Vec<usize> = (0..n_quad).filter(|k| k % size == rank).collect();

        // rayon over just this rank's frequencies; BLAS pinned to 1 under the
        // rayon region (the per-frequency eigh must not nest OpenBLAS threads).
        let my_rows: Vec<(usize, Vec<f64>)> = with_blas_threads(1, || {
            my_idx
                .par_iter()
                .map(|&k| {
                    let omega = quad_freqs[k];
                    let scale = build_scale_factors(eps_occ, eps_vir, omega);
                    let eps_proj = dielectric_matrix_from_projection(&y, &scale);
                    // Eigenvalue-only divide-and-conquer solver (dsyevd_): only the
                    // spectrum is consumed (eigenvectors discarded), and eigenvalues
                    // are identical to `.eigh()` — so this is a safe, faster swap
                    // (validated in ferric_core::linalg tests). Runs once per
                    // quadrature point per rank.
                    let evals = ferric_core::linalg::eigvalsh_dc(
                        &eps_proj,
                        ferric_core::linalg::Uplo::Upper,
                    )
                    .map_err(|e| dielectric_lapack_err("dielectric eigh failed", e))?;
                    Ok((k, evals))
                })
                .collect::<Result<Vec<(usize, Vec<f64>)>, FerricError>>()
        })?;

        // Zero-padded local buffer: only this rank's owned rows are nonzero.
        let mut local = Array2::<f64>::zeros((n_quad, m));
        for (k, row) in &my_rows {
            for (alpha, &val) in row.iter().enumerate() {
                local[(*k, alpha)] = val;
            }
        }

        if let Some(world) = ctx.world() {
            use mpi::traits::CommunicatorCollectives;
            let mut global = Array2::<f64>::zeros((n_quad, m));
            world.all_reduce_into(
                local.as_slice().unwrap(),
                global.as_slice_mut().unwrap(),
                mpi::collective::SystemOperation::sum(),
            );
            Ok(global)
        } else {
            Ok(local)
        }
    }

    /// MPI-distributed variant of [`crate::energy::eval_inv_dielectric_matrices`].
    ///
    /// Same round-robin-over-frequency + zero-padded-Allreduce shape as
    /// [`eval_eigenvalues_at_frequencies_mpi`], applied to the M×M
    /// inverse-dielectric matrices instead of the M-length eigenvalue rows:
    /// each rank computes only its own ω_k's `W̃_d(iω_k)` matrix, flattens the
    /// `n_quad` matrices into one zero-padded `(n_quad, m*m)` buffer,
    /// `Allreduce(sum)`s it, then unflattens back into `Vec<Array2<f64>>`.
    pub fn eval_inv_dielectric_matrices_mpi(
        ctx: &ParallelContext,
        eigenvectors: &Array2<f64>,
        b_ov: &Array2<f64>,
        eps_occ: &[f64],
        eps_vir: &[f64],
        quad_freqs: &[f64],
    ) -> Result<Vec<Array2<f64>>, FerricError> {
        let n_quad = quad_freqs.len();
        let m = eigenvectors.ncols();
        let size = ctx.size.max(1);
        let rank = ctx.rank;

        let y = eigenvectors.t().dot(b_ov);
        let my_idx: Vec<usize> = (0..n_quad).filter(|k| k % size == rank).collect();

        let my_mats: Vec<(usize, Array2<f64>)> = with_blas_threads(1, || {
            my_idx
                .par_iter()
                .map(|&k| {
                    let omega = quad_freqs[k];
                    let scale = build_scale_factors(eps_occ, eps_vir, omega);
                    let eps_proj = dielectric_matrix_from_projection(&y, &scale);
                    let mut winv = eps_proj.inv().map_err(|e| {
                        dielectric_lapack_err("PDEP-basis dielectric inversion failed", e)
                    })?;
                    for d in 0..m {
                        winv[(d, d)] -= 1.0;
                    }
                    Ok((k, winv))
                })
                .collect::<Result<Vec<(usize, Array2<f64>)>, FerricError>>()
        })?;

        // Flatten into a zero-padded (n_quad, m*m) buffer for a single Allreduce.
        let mut local = Array2::<f64>::zeros((n_quad, m * m));
        for (k, mat) in &my_mats {
            let flat = mat.as_standard_layout();
            local.row_mut(*k).assign(&flat.to_shape(m * m).unwrap());
        }

        let flat_global = if let Some(world) = ctx.world() {
            use mpi::traits::CommunicatorCollectives;
            let mut global = Array2::<f64>::zeros((n_quad, m * m));
            world.all_reduce_into(
                local.as_slice().unwrap(),
                global.as_slice_mut().unwrap(),
                mpi::collective::SystemOperation::sum(),
            );
            global
        } else {
            local
        };

        // Unflatten back into per-frequency M×M matrices.
        let mut out = Vec::with_capacity(n_quad);
        for k in 0..n_quad {
            let row = flat_global.row(k).to_owned();
            let mat = Array2::from_shape_vec((m, m), row.to_vec())
                .map_err(|e| FerricError::General(format!("inv-dielectric unflatten failed: {e}")))?;
            out.push(mat);
        }
        Ok(out)
    }

    /// MPI-distributed top-level PDEP-RPA energy calculation.
    ///
    /// Identical to [`crate::run_pdep_rpa_from_intermediates`] through Step 5
    /// (RI transform, Davidson/Lanczos eigensolve, truncation) — shares
    /// [`crate::run_pdep_rpa_eigensolve`] verbatim, so those steps are NOT
    /// distributed here (see module docs) and run replicated on every rank,
    /// exactly as they do in the serial path. Only Steps 6/6b (the
    /// imaginary-frequency quadrature evaluation) are routed through
    /// [`eval_eigenvalues_at_frequencies_mpi`] /
    /// [`eval_inv_dielectric_matrices_mpi`] instead of their serial
    /// counterparts. Restricted to the `Chi0Backend::Dense` path (the
    /// production default) — the Laplace-separable χ₀ backend is not
    /// MPI-distributed by this module (mirrors T9's scoping to the paths T8
    /// already validated); errors out explicitly rather than silently
    /// falling back to a non-distributed Laplace evaluation.
    pub fn run_pdep_rpa_mpi(
        ctx: &ParallelContext,
        mol: &ferric_core::mol::Molecule,
        obs: &ferric_integrals::basis_bridge::PreparedBasis,
        dfbs: &ferric_integrals::basis_bridge::PreparedBasis,
        op: ferric_integrals::operator::Operator,
        rhf: &ferric_scf::ScfResult,
        config: &crate::config::PdepRpaConfig,
    ) -> Result<crate::PdepRpaResult, FerricError> {
        use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};

        if !matches!(config.chi0_backend, crate::config::Chi0Backend::Dense) {
            return Err(FerricError::General(
                "run_pdep_rpa_mpi: only Chi0Backend::Dense is MPI-distributed (T10 scope); \
                 use crate::run_pdep_rpa for the Laplace backend"
                    .into(),
            ));
        }

        let mp2_cfg = RiMp2Config {
            frozen_core: config.frozen_core,
            memory_budget_bytes: config.memory_budget_bytes,
            ..Default::default()
        };
        let inter = compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
        let b_ov = &inter.b_ov;
        let nocc = inter.nocc;
        let nvir = inter.nvir;
        let first_occ = inter.first_occ;
        let nocc_total = inter.nocc_total;
        let eps_occ: Vec<f64> = rhf.eps_r()[first_occ..first_occ + nocc].to_vec();
        let eps_vir: Vec<f64> = rhf.eps_r()[nocc_total..nocc_total + nvir].to_vec();

        // Steps 1-5: shared, replicated eigensolve (same function the serial
        // path uses).
        let stage = crate::run_pdep_rpa_eigensolve(&inter, mol, obs, dfbs, op, rhf, config)?;
        debug_assert!(
            stage.laplace_chi0_quad.is_none(),
            "chi0_backend already checked Dense above"
        );

        // Steps 6/6b: MPI-distributed frequency loop.
        let eigenvalues_freq = eval_eigenvalues_at_frequencies_mpi(
            ctx,
            &stage.eigenvectors,
            b_ov,
            &eps_occ,
            &eps_vir,
            &stage.quad_freqs,
        )?;

        let inv_dielectric_freq = if config.need_inv_dielectric_freq {
            Some(eval_inv_dielectric_matrices_mpi(
                ctx,
                &stage.eigenvectors,
                b_ov,
                &eps_occ,
                &eps_vir,
                &stage.quad_freqs,
            )?)
        } else {
            None
        };

        // Step 7: Integrate RPA correlation energy (same trace-log formula,
        // now fed by the MPI-assembled eigenvalues_freq).
        let e_rpa = crate::energy::rpa_correlation_energy(&stage.quad_weights, &eigenvalues_freq);

        // Step 8: Diagnostic RI-dRPA energy — not distributed (rarely used,
        // opt-in `run_diagnostics` flag); kept serial/replicated like the
        // rest of `run_pdep_rpa_from_intermediates`'s Step 8.
        let e_rpa_dft_diag = if config.run_diagnostics {
            Some(crate::diagnostics::ri_drpa_energy(
                b_ov, &eps_occ, &eps_vir, &stage.quad_freqs, &stage.quad_weights,
            )?)
        } else {
            None
        };

        Ok(crate::PdepRpaResult {
            e_rpa,
            n_eigenpotentials: stage.n_keep,
            eigenvalues_static: stage.eigenvalues_static,
            eigenpotentials: stage.eigenpotentials_aux,
            dressed_eigenvectors: stage.eigenvectors,
            quad_freqs: stage.quad_freqs,
            quad_weights: stage.quad_weights,
            eigenvalues_freq,
            inv_dielectric_freq,
            e_rpa_dft_diag,
            eigensolver_converged: stage.eigensolver_converged,
        })
    }
}

#[cfg(feature = "mpi")]
pub use inner::{eval_eigenvalues_at_frequencies_mpi, eval_inv_dielectric_matrices_mpi, run_pdep_rpa_mpi};
