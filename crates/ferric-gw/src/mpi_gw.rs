//! MPI-distributed per-MO QP self-energy loop (T10 remainder).
//!
//! ## Why MO-index round-robin, not aux-band striping (T8/T9's lever)
//!
//! T8/T9 band the `naux` axis of a large B tensor because a genuine per-rank
//! ROW SUBSET can be built and held — real memory reduction. GW's per-MO QP
//! loop has no such tensor: each MO's [`crate::sigma::solve_qp_for_mo`] call
//! needs the FULL `m_proj` (M × n_act × n_act) projection and the FULL
//! per-frequency inverse-dielectric stack `inv_diel_freq` (length N_quad, each
//! M × M) to sample Σ_c(iω_k) at its own Padé support nodes and Newton-solve
//! the QP root — there is no way to hand one rank only part of either tensor
//! and still evaluate even one MO's QP energy. What DOES split, disjointly
//! and independently, is the SET OF MOs itself: every `solve_qp_for_mo` call
//! is already independent per MO (no shared mutable state, no cross-MO
//! coupling in the QP equation) — the existing serial code already
//! parallelizes over `mo_indices` with rayon for exactly this reason. This
//! mirrors T10's RPA frequency-round-robin (`ferric-rpa::mpi_rpa`, itself
//! modeled on `direct_j.rs`/`direct_k.rs`'s shell-quartet round-robin,
//! `docs/superpowers/mpi.md` Section 2): split `0..mo_indices.len()`
//! round-robin across ranks (`local_idx % size == rank`), each rank calls
//! `solve_qp_for_mo` ONLY for its own MOs (each MO's Padé fit + 30-step
//! Newton solve is itself fully serial scalar work — no BLAS, no further
//! distribution axis inside a single MO's evaluation), then `Allreduce(sum)`
//! a zero-padded `(n_mo, 4)` buffer (`eps_qp`, `sigma_c`, `z_factor`,
//! `converged-as-0-or-1`) reassembles the full per-MO result on every rank —
//! a plain LINEAR reduction, no nonlinear coupling across MOs (unlike
//! RI-MP2's `(ia|jb)²` pattern), so ONE `Allreduce` suffices, mirroring T10's
//! single-reduction shape.
//!
//! `Σ_x` is NOT part of this distribution: [`crate::cohsex::sigma_x_diag`]
//! computes Σ_x for every ACTIVE MO in one O(n_act · naux) pass up front
//! (called once, not per-QP-MO), so it is already cheap and fully replicated
//! — nothing to distribute there.
//!
//! ## What is NOT distributed here
//!
//! * The upstream W construction (`PdepRpaResult`) — callers get that for
//!   free by building `pdep` via [`ferric_rpa::mpi_rpa::run_pdep_rpa_mpi`]
//!   instead of [`ferric_rpa::run_pdep_rpa`] (T10's own note in
//!   `docs/superpowers/mpi.md` Section 5).
//! * `m_proj` (the M × n_act × n_act projected B̃ tensor) and
//!   `inv_diel_freq` are still fully replicated on every rank — this module
//!   reduces WALL-CLOCK on the per-MO Newton/Padé loop but does NOT reduce
//!   per-rank resident memory the way T8/T9 do.
//! * Only G0W0's per-MO loop is wired here. evGW₀/evGW's outer
//!   eigenvalue-self-consistency loops (`sigma::run_evgw0`/`run_evgw`,
//!   `u_sigma::run_u_evgw0`/`run_u_evgw`) call the identical
//!   `solve_qp_for_mo` per outer iteration and could reuse this same
//!   round-robin+Allreduce helper, but are NOT wired to it in this pass — see
//!   the module-level report for why (partial-but-honest over
//!   complete-but-unverified, following T10's own precedent). Open-shell
//!   U-G0W0 (`u_sigma::run_u_g0w0`) is likewise not wired here; its QP loop
//!   has the identical independent-per-MO shape (twice, once per spin) and
//!   is a natural next increment reusing [`solve_qp_for_mos_mpi`] directly.

#[cfg(feature = "mpi")]
mod inner {
    use crate::sigma::solve_qp_for_mo;
    use ferric_core::parallel::ParallelContext;
    use ferric_core::FerricError;
    use ndarray::Array2;

    /// Per-MO QP result row: `(eps_qp, sigma_c, z_factor, newton_converged)`.
    pub type QpRow = (f64, f64, f64, bool);

    /// MPI-distributed variant of the `par_iter().map(solve_qp_for_mo)` loop
    /// used identically by [`crate::sigma::run_g0w0`],
    /// [`crate::sigma::run_evgw0`]/[`run_evgw`]'s inner per-iteration solve,
    /// and [`crate::u_sigma`]'s per-spin loops.
    ///
    /// `mo_locs[i]` and `eps_mf[i]` are the ALREADY-RESOLVED local MO index
    /// (`mo_abs - first_act`) and mean-field energy for the i-th entry of the
    /// (absolute) `mo_indices` list the caller is solving; `static_shifts[i]`
    /// is the per-MO Σ_x−v_xc shift (0.0 for HF references). Splits
    /// `0..mo_locs.len()` round-robin (`i % size == rank`), runs this rank's
    /// own subset through `solve_qp_for_mo` (rayon fills cores within the
    /// subset, matching the serial code's per-MO rayon fan-out), then
    /// `Allreduce(sum)`s a zero-padded `(n_mo, 4)` buffer so every rank ends
    /// up holding the full result — algebraically exact (each row is written
    /// by exactly one rank and zero everywhere else). At `ctx.size == 1`
    /// every `i % 1 == 0` is true, so every row is computed locally by this
    /// one rank in the SAME iteration order as `par_iter()` (`collect`
    /// preserves the source order regardless of thread scheduling) — the
    /// Allreduce sums one nonzero contribution per row, byte-identical to the
    /// serial per-MO loop.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_qp_for_mos_mpi(
        ctx: &ParallelContext,
        mo_locs: &[usize],
        eps_mf: &[f64],
        static_shifts: &[f64],
        m_proj: &ndarray::Array3<f64>,
        inv_diel_freq: &[Array2<f64>],
        quad_weights: &[f64],
        quad_freqs: &[f64],
        eps_prop: &[f64],
        pade_npts: usize,
        newton_damp: f64,
        ef: f64,
    ) -> Result<Vec<QpRow>, FerricError> {
        use rayon::prelude::*;

        let n_mo = mo_locs.len();
        debug_assert_eq!(eps_mf.len(), n_mo);
        debug_assert_eq!(static_shifts.len(), n_mo);
        let size = ctx.size.max(1);
        let rank = ctx.rank;

        // This rank's owned local indices into mo_locs/eps_mf/static_shifts
        // (round-robin, mirrors T10's frequency-index convention:
        // idx % size == rank).
        let my_idx: Vec<usize> = (0..n_mo).filter(|i| i % size == rank).collect();

        // Each MO's solve is independent scalar work (Padé fit + Newton) —
        // rayon over this rank's own subset only.
        let my_rows: Vec<(usize, QpRow)> = my_idx
            .par_iter()
            .map(|&i| {
                let row = solve_qp_for_mo(
                    mo_locs[i],
                    eps_mf[i],
                    m_proj,
                    inv_diel_freq,
                    quad_weights,
                    quad_freqs,
                    eps_prop,
                    pade_npts,
                    newton_damp,
                    ef,
                    static_shifts[i],
                )?;
                Ok((i, row))
            })
            .collect::<Result<Vec<(usize, QpRow)>, FerricError>>()?;

        // Zero-padded local buffer: only this rank's owned rows are nonzero.
        // Layout per row: [eps_qp, sigma_c, z_factor, converged_as_f64].
        let mut local = Array2::<f64>::zeros((n_mo, 4));
        for &(i, (eps_qp, sc, z, conv)) in &my_rows {
            local[(i, 0)] = eps_qp;
            local[(i, 1)] = sc;
            local[(i, 2)] = z;
            local[(i, 3)] = if conv { 1.0 } else { 0.0 };
        }

        let global = if let Some(world) = ctx.world() {
            use mpi::traits::CommunicatorCollectives;
            let mut global = Array2::<f64>::zeros((n_mo, 4));
            world.all_reduce_into(
                local.as_slice().unwrap(),
                global.as_slice_mut().unwrap(),
                mpi::collective::SystemOperation::sum(),
            );
            global
        } else {
            local
        };

        let mut out = Vec::with_capacity(n_mo);
        for i in 0..n_mo {
            let eps_qp = global[(i, 0)];
            let sc = global[(i, 1)];
            let z = global[(i, 2)];
            // Every row was written by exactly one rank as exactly 0.0 or
            // 1.0 and zero everywhere else, so the Allreduce sum reproduces
            // that same 0.0/1.0 exactly (no partial-owner ambiguity).
            let conv = global[(i, 3)] != 0.0;
            out.push((eps_qp, sc, z, conv));
        }
        Ok(out)
    }

    /// MPI-distributed G0W0: identical to [`crate::sigma::run_g0w0`] except
    /// the per-MO QP loop is routed through [`solve_qp_for_mos_mpi`] instead
    /// of a plain `par_iter()`. Everything upstream (PDEP-RPA, B̃ projection,
    /// redressing) is UNCHANGED and still replicated per rank — callers that
    /// also want W itself distributed should build `pdep` via
    /// [`ferric_rpa::mpi_rpa::run_pdep_rpa_mpi`] before calling this (see
    /// module docs).
    #[allow(clippy::too_many_arguments)]
    pub fn run_g0w0_mpi(
        ctx: &ParallelContext,
        mol: &ferric_core::mol::Molecule,
        rhf: &ferric_scf::ScfResult,
        mo_b: &crate::mo_b::MoB,
        v_dressed: &Array2<f64>,
        pdep: ferric_rpa::PdepRpaResult,
        qp_range: std::ops::Range<usize>,
        gw_cfg: &crate::GwConfig,
        vxc_diag: Option<&ndarray::Array1<f64>>,
    ) -> Result<crate::GwResult, FerricError> {
        use crate::cohsex::{project_b_into_pdep, sigma_x_diag};
        use crate::sigma::{fermi_level, warn_if_unconverged};
        use ndarray::Array1;

        let _ = (mol, rhf);
        let first_act = mo_b.first_act;
        let m_modes = v_dressed.ncols();
        if pdep.eigenvalues_freq.ncols() != m_modes {
            return Err(FerricError::General(
                "pdep eigenvalues_freq mode count does not match dressed eigenpotentials".into(),
            ));
        }

        let sigma_x_all = sigma_x_diag(mo_b);
        let m_proj = project_b_into_pdep(mo_b, v_dressed);
        let inv_diel_freq = pdep.inv_dielectric_freq.as_ref().ok_or_else(|| {
            FerricError::General(
                "PDEP result missing inv_dielectric_freq (GW requires the dense χ₀ path)".into(),
            )
        })?;
        let quad_freqs = pdep.quad_freqs.clone();
        let quad_weights = pdep.quad_weights.clone();
        let ef = fermi_level(&mo_b.eps_act, mo_b.n_occ_act);

        let mo_indices: Vec<usize> = qp_range.clone().collect();
        for &mo_abs in &mo_indices {
            if mo_abs < first_act {
                return Err(FerricError::General(format!(
                    "qp_mos index {mo_abs} is in the frozen-core block"
                )));
            }
        }
        let mo_locs: Vec<usize> = mo_indices.iter().map(|&mo_abs| mo_abs - first_act).collect();
        let eps_mf_locs: Vec<f64> = mo_locs.iter().map(|&m_loc| mo_b.eps_act[m_loc]).collect();
        let static_shifts: Vec<f64> = mo_indices
            .iter()
            .zip(mo_locs.iter())
            .map(|(&mo_abs, &m_loc)| {
                vxc_diag
                    .map(|v| sigma_x_all[m_loc] - v[mo_abs])
                    .unwrap_or(0.0)
            })
            .collect();

        let qp_rows = solve_qp_for_mos_mpi(
            ctx,
            &mo_locs,
            &eps_mf_locs,
            &static_shifts,
            &m_proj,
            inv_diel_freq,
            &quad_weights,
            &quad_freqs,
            &mo_b.eps_act,
            gw_cfg.pade_npts,
            gw_cfg.qp_newton_damp,
            ef,
        )?;

        let mut eps_qp = Array1::<f64>::zeros(mo_indices.len());
        let mut eps_mf = Array1::<f64>::zeros(mo_indices.len());
        let mut sx_out = Array1::<f64>::zeros(mo_indices.len());
        let mut sc_out = Array1::<f64>::zeros(mo_indices.len());
        let mut z_out = Array1::<f64>::ones(mo_indices.len());
        let mut qp_converged = vec![true; mo_indices.len()];
        for (idx, &m_loc) in mo_locs.iter().enumerate() {
            let (eps_qp_m, sc_final, z_renorm, converged) = qp_rows[idx];
            eps_mf[idx] = mo_b.eps_act[m_loc];
            sx_out[idx] = sigma_x_all[m_loc];
            eps_qp[idx] = eps_qp_m;
            sc_out[idx] = sc_final;
            z_out[idx] = z_renorm;
            qp_converged[idx] = converged;
        }
        warn_if_unconverged("G0W0 (mpi)", &mo_indices, &qp_converged);

        Ok(crate::GwResult {
            mo_indices,
            eps_mf,
            eps_qp,
            sigma_x: sx_out,
            sigma_c: sc_out,
            z_factor: z_out,
            qp_converged,
            n_ev_iter: 0,
            outer_converged: true,
            pdep,
        })
    }
}

#[cfg(feature = "mpi")]
pub use inner::{run_g0w0_mpi, solve_qp_for_mos_mpi, QpRow};
