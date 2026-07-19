//! MPI-distributed RI-MP2 (requires `mpi` feature).
//!
//! ## MPI aux-band striping (T9)
//!
//! Follows exactly the same aux-band-striping convention T8
//! (`ferric_scf::df_j`/`df_k`) established for the SCF Fock build, applied to
//! the RI-MP2 `B^P_{ia}` tensor and correlation-energy contraction:
//!
//! * **Aux-band ownership.** `naux` is split into contiguous, disjoint,
//!   balanced bands via [`ParallelContext::aux_band`]. Each rank builds and
//!   holds ONLY its own band of the dressed `B^P_{ia}` tensor
//!   ([`crate::rimp2::eri3_mo_block_dressed_band`], the same private helper
//!   `df_k.rs`'s `build_dressed_band` mirrors) — the resident B_ov footprint
//!   per rank is `(band) · nocc·nvir · 8` bytes ≈ (full tensor) / N, a real
//!   memory reduction, not just compute striping.
//! * **Why NOT a plain sum-over-P reduction (unlike DF-K's K).** DF-K's K =
//!   Σ_P B_P D B_Pᵀ is LINEAR in the aux band, so each rank's band-restricted
//!   partial K can be Allreduce-summed directly. RI-MP2's correlation energy
//!   is QUADRATIC in `(ia|jb) = Σ_P B^P_ia B^P_jb`: the energy needs
//!   `(ia|jb)²`, and `(Σ_band x_band)² ≠ Σ_band x_band²` — the full-P
//!   `(ia|jb)` must be assembled (summed over ALL P) before it can be
//!   squared. This is the same coupling shape DF-J hits with its `c_P = V⁻¹
//!   d_P` step: reduce an intermediate BEFORE the nonlinear stage.
//! * **Two-stage reduction, mirroring DF-J's two-Allreduce pattern.** For
//!   each occupied index `i`, `g_i[a, jb] = (ia|jb) = Σ_P B^P_ia B^P_jb` is
//!   itself a matrix product contracted over P — a LINEAR reduction over the
//!   aux axis. Each rank computes `g_i` restricted to its own aux band
//!   (`b_i_band^T · b_ov_band`, a genuine partial sum over P) and
//!   Allreduce-sums it once per `i` to assemble the full `g_i` on every rank
//!   (mirrors DF-J's Pass-1 `d_P` reduction: sum a linear intermediate before
//!   the quadratic/nonlinear step). Only THEN is `g_i` squared into the MP2
//!   denominator sum — never before the aux-P sum is complete, so this is
//!   algebraically exact, not an approximation.
//! * **Distributing the O(i) contraction itself.** Rather than have every
//!   rank redundantly loop over all `nocc` values of `i` (replicating FLOPs),
//!   the occupied index `i` is ALSO round-robin distributed across ranks
//!   (`i % size == rank`, mirroring the direct-JK shell-quartet
//!   round-robin convention in `direct_j.rs`/`direct_k.rs`): each rank
//!   computes only its own subset of `i`'s `(e_os_i, e_ss_i)` partials (after
//!   assembling that `i`'s full `g_i` via the aux-band Allreduce above), and a
//!   single final `Allreduce(sum)` over the two scalars yields the total
//!   correlation energy. This distributes BOTH the aux-band memory AND the
//!   O(nocc) contraction compute, while keeping the aux-P reduction exact.
//! * **1 rank / feature off.** The aux band is `[0, naux)` (full) and the `i`
//!   round-robin is trivially every `i` on the single rank, so every
//!   Allreduce is a no-op and this reproduces
//!   [`crate::rimp2::ri_mp2_spin_components`] byte-for-byte (same
//!   `eri3_mo_block_dressed_band` code path with a full band == the serial
//!   `eri3_mo_block_dressed`, and the same per-`i` `g_i = b_i^T · b_ov`
//!   contraction as [`crate::rimp2::spin_components_from_b_ov`], just walked
//!   sequentially instead of via rayon — floating-point `+` reduction order
//!   is preserved because both paths sum `i = 0, 1, ..., nocc-1` in ascending
//!   order into a single running total).
//!
//! `frozen_core` is handled by [`crate::rimp2::active_occ`] exactly as the
//! serial path does — it restricts the occupied index RANGE, which is
//! orthogonal to (and applied before) the aux-P band split.

#[cfg(feature = "mpi")]
mod inner {
    use crate::rimp2::{
        active_occ, eri3_budget_bytes, eri3_mo_block_dressed_band, metric_inverse_sqrt, RiMp2Config,
    };
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_core::FerricError;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_integrals::threeindex::coulomb_metric_2c;
    use ferric_integrals::three_index_source::ThreeIndexSource;
    use ferric_scf::ScfResult;
    use ndarray::Array2;

    /// Result of an MPI-distributed RI-MP2 run. Every rank returns the same
    /// (post-Allreduce) totals.
    #[derive(Debug)]
    pub struct MpiMp2Result {
        pub mp2_corr: f64,
        pub e_os: f64,
        pub e_ss: f64,
        pub total_energy: f64,
    }

    /// Compute RI-MP2 correlation energy, distributing the `B^P_{ia}` tensor
    /// across MPI ranks by aux-band striping (see module docs for the
    /// two-stage-reduction derivation). `ctx` selects the band via
    /// [`ParallelContext::aux_band`]; with `ctx.size == 1` (or the `mpi`
    /// feature simply unused at 1 rank) every reduction is a no-op and the
    /// result is byte-identical to [`crate::rimp2::ri_mp2_spin_components`].
    pub fn run_mpi_ri_mp2(
        ctx: &ParallelContext,
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        rhf: &ScfResult,
        config: &RiMp2Config,
    ) -> Result<MpiMp2Result, FerricError> {
        let nbas = obs.nbasis();
        let nelec = mol.nelec() as usize;
        let nocc_total = nelec / 2;
        let nocc = active_occ(nocc_total, config.frozen_core)?;
        let first_occ = config.frozen_core;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();
        let eps = rhf.eps_r();
        let c = rhf.mos_r();

        let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

        // (P|Q) metric and V^{-1/2}. Small (naux, naux); every rank builds and
        // holds the FULL metric identically (mirrors DfJ's full V^{-1}) — it
        // is not the memory hazard, B^P_ia is.
        let v2c = coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = metric_inverse_sqrt(&v2c, op)?;

        // This rank's aux band. [0, naux) at size==1 — identical to the serial
        // full-range path.
        let (p0, p1) = ctx.aux_band(naux);

        // Full raw AO source, budget-bounded / streamable (the dressing sum
        // over Q needs every Q regardless of this rank's P band) — same
        // requirement ThreeIndexSource::build_dressed_band imposes on DF-K.
        let mut raw = ThreeIndexSource::build(op, obs, dfbs, eri3_budget_bytes(config.memory_budget_bytes))?;

        // This rank's band of the dressed B^P_ia tensor: shape (band, nocc*nvir).
        // Real memory reduction — only [p0,p1) rows are ever resident here.
        let b_band = eri3_mo_block_dressed_band(&mut raw, &v_inv_sqrt, &c_occ, &c_vir, p0, p1)?;
        drop(raw);

        let world = ctx.world();
        let size = ctx.size.max(1);
        let rank = ctx.rank;

        // Two-stage reduction per occupied index i, round-robin distributed
        // across ranks (i % size == rank) so the O(nocc) contraction is ALSO
        // split, not just the aux band.
        //
        // Stage 1 (linear, aux-P): g_i_partial[a, jb] = Σ_{P in band}
        // B^P_ia B^P_jb — a genuine partial sum over the contracted aux axis,
        // computed from THIS rank's band of both operands. Allreduce-sum over
        // all ranks' bands (which tile [0,naux) disjointly) assembles the
        // FULL g_i on every rank — exactly mirroring DF-K's "band-restricted
        // partial, then Allreduce(sum)" but applied to the (nvir, nocc*nvir)
        // intermediate instead of the (nao,nao) K matrix.
        //
        // Stage 2 (quadratic, only after g_i is complete): square g_i into
        // the (e_os_i, e_ss_i) MP2 denominator sum. This can only happen
        // AFTER stage 1's Allreduce — squaring a partial-P g_i would be wrong
        // (see module docs: (Σx)² ≠ Σx²).
        let mut e_os = 0.0_f64;
        let mut e_ss = 0.0_f64;
        for i in 0..nocc {
            // g_i_partial = b_i_band^T · b_band  (nvir, nocc*nvir), summed
            // over only THIS rank's aux rows.
            let b_i_band = b_band.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
            let mut g_i = b_i_band.t().dot(&b_band); // partial sum over this rank's P band

            if let Some(world) = &world {
                use mpi::traits::CommunicatorCollectives;
                let mut g_i_global = Array2::<f64>::zeros(g_i.dim());
                world.all_reduce_into(
                    g_i.as_slice().unwrap(),
                    g_i_global.as_slice_mut().unwrap(),
                    mpi::collective::SystemOperation::sum(),
                );
                g_i = g_i_global;
            }

            // Round-robin: only the owning rank spends FLOPs squaring this i's
            // (now-complete) g_i into the energy sum. Every rank still paid
            // for the Allreduce above (needed so the owning rank has the full
            // g_i), but the O(nocc*nvir^2) inner double loop is NOT
            // replicated nocc times across ranks.
            if i % size == rank {
                let e_ij_i = eps[first_occ + i];
                for j in 0..nocc {
                    let e_ij = e_ij_i + eps[first_occ + j];
                    for a in 0..nvir {
                        for b in 0..nvir {
                            let g_ab = g_i[(a, j * nvir + b)]; // (ia|jb)
                            let g_ba = g_i[(b, j * nvir + a)]; // (ib|ja)
                            let denom = e_ij - eps[nocc_total + a] - eps[nocc_total + b];
                            e_os += g_ab * g_ab / denom;
                            e_ss += g_ab * (g_ab - g_ba) / denom;
                        }
                    }
                }
            }
        }

        // Final scalar reduction: sum each rank's round-robin energy share.
        if let Some(world) = &world {
            use mpi::traits::CommunicatorCollectives;
            let mut totals = [e_os, e_ss];
            let mut totals_global = [0.0_f64; 2];
            world.all_reduce_into(
                &totals[..],
                &mut totals_global[..],
                mpi::collective::SystemOperation::sum(),
            );
            totals = totals_global;
            e_os = totals[0];
            e_ss = totals[1];
        }

        let mp2_corr = e_os + e_ss;
        Ok(MpiMp2Result {
            mp2_corr,
            e_os,
            e_ss,
            total_energy: rhf.energy + mp2_corr,
        })
    }
}

#[cfg(feature = "mpi")]
pub use inner::*;
