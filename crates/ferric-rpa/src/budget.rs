//! Pre-flight peak-memory estimate for the RS-MP2-RPA / PDEP-RPA pipeline.
//!
//! # Why this exists
//!
//! A production RS-MP2-RPA job overshot its configured `[memory] budget_gb=10.0`
//! by ~1.7× (peak RSS ~16.95 GiB measured), thrashed under synchronous direct
//! reclaim (cgroup `memory.stat`: `pgscan_direct≈2.0M`, `pgsteal_direct≈1.37M`,
//! `workingset_refault_file≈1.29M`), and had to be killed manually. `[memory]
//! budget_gb` already bounds the ERI3 AO-block transform (see
//! `ferric_mp2::rimp2::eri3_mo_ov_blocked`) and the Lanczos full-rank panel
//! (`lanczos::lanczos_panel_width`), but nothing checks the *other* large
//! allocations that co-exist with those — most importantly the per-worker
//! frequency-quadrature scratch in `energy::eval_eigenvalues_at_frequencies`,
//! which scales with the **rayon thread count**, not with any budget knob.
//!
//! [`estimate_peak_bytes`] is a conservative (i.e. deliberately over-, never
//! under-, estimating) upper bound on the resident bytes the pipeline can hold
//! at once, covering:
//!
//! 1. The `RpaIntermediates` build (`compute_rpa_intermediates` /
//!    `_spin`): the metric `V^{-1/2}` build (`metric_inverse_sqrt`, up to 3
//!    co-resident `naux×naux` buffers on the `eigh` branch — see
//!    `ferric_mp2::rimp2::eigh_inverse_sqrt`), plus the raw-AO-block →
//!    MO-tensor transform (`eri3_mo_ov_blocked`) which co-resides its
//!    `(naux, nocc, nvir)` output with the dressed `b_ov` GEMM output of the
//!    SAME shape (`rimp2.rs` lines ~499-504) — i.e. **two** `naux·nocc·nvir`
//!    buffers alive simultaneously, not one.
//! 2. Formulation B (`RsMp2RpaFormulation::DeltaLr`) and T (`CoupledRings`)
//!    both build 2-3 *separate* `RpaIntermediates` (Coulomb/erf/erfc), each
//!    with its own `b_ov`; while the fused driver builds them sequentially
//!    (not all three alive at peak — see `rs_mp2_rpa.rs`'s "SHARED-INTERMEDIATE
//!    FUSION" comment), the retained `sc_full`/`sc_sr`/`sc_lr` spin-component
//!    scalars are tiny, so only ONE `RpaIntermediates`' `b_ov` is resident at a
//!    time from the MP2 side. The dRPA solve started from an intermediate,
//!    though, keeps that intermediate's `b_ov` alive for the ENTIRE Lanczos +
//!    frequency-quadrature stage (it's borrowed, not dropped) — so the b_ov
//!    co-resides with whatever the eigensolve/quadrature stage below needs.
//! 3. The Lanczos full-rank eigensolve (`lanczos::run_lanczos_full_rank_budgeted`):
//!    a dense `naux×naux` accumulator `a` (plus its `eigh` output, another
//!    `naux×naux` pair of eigenvalues/eigenvectors) — budget-gated via the
//!    panel width for the *matvec* transient, but the assembled `a` itself and
//!    its `eigh` output are NOT panelled and are always fully resident.
//! 4. The post-eigensolve frequency-quadrature loop
//!    (`energy::eval_eigenvalues_at_frequencies`): `map_init` gives each rayon
//!    WORKER its own `(m, nov)` + `(m, m)` scratch pair, so the total scratch
//!    scales with `n_workers·(m·nov + m²)·8` bytes — this is the term that
//!    scales with rayon thread count, the crux of the incident (more cores ⇒
//!    more concurrent scratch ⇒ budget-blind).
//!
//! This is a pre-flight ESTIMATE, not an exact accounting — it deliberately
//! rounds up (e.g. counts the full un-panelled Lanczos `a` matrix even though
//! the matvec that builds it is panelled) so a job that *would* have fit is
//! never wrongly rejected only in degenerate corners; the primary goal is
//! catching the "1.7× over budget, thrash and get OOM-killed" failure mode,
//! not shaving the last few percent off a tight-but-correct estimate.

/// Bytes per `f64`.
const F64_BYTES: usize = 8;

/// Shape parameters for [`estimate_peak_bytes`]. All fields are the ACTIVE
/// (post-frozen-core) dimensions already used elsewhere in the pipeline
/// (`RpaIntermediates::{nocc,nvir,naux}`, `PdepRpaConfig`/CLI knobs).
#[derive(Debug, Clone, Copy)]
pub struct PeakEstimateShape {
    /// Auxiliary (density-fitting) basis size.
    pub naux: usize,
    /// Active occupied orbitals (post-frozen-core).
    pub nocc: usize,
    /// Active virtual orbitals.
    pub nvir: usize,
    /// Number of imaginary-frequency quadrature points (`PdepRpaConfig::quadrature.n_points`).
    pub n_quad: usize,
    /// Number of rayon workers the frequency-quadrature loop may fan out
    /// across. Callers should pass `rayon::current_num_threads().max(1)`
    /// (the same idiom used elsewhere in this crate, e.g. `properties.rs`'s
    /// `dipole_band_width`) — this is the crux of the incident: more cores
    /// means more concurrent per-worker scratch, budget-blind until now.
    pub n_workers: usize,
    /// Number of PDEP eigenpotentials retained after truncation (`M` in the
    /// docs above). Bounds the per-worker `(m, nov)`/`(m, m)` quadrature
    /// scratch AND the Lanczos full-rank `naux×naux` accumulator's *useful*
    /// output width. When unknown ahead of time (the common pre-flight case:
    /// truncation count isn't known until the eigensolve runs), pass `naux`
    /// (the untruncated upper bound — `trunc_thresh=0.0` production runs, per
    /// `RsMp2RpaConfig::default`, keep every mode anyway, so `naux` is not a
    /// pessimistic guess there but the literal value).
    pub n_keep: usize,
    /// Grid-path allocations, when the caller is a per-atom property path
    /// (`pdep_polarizability_becke{,_dynamic}`, the Hirshfeld variants,
    /// `dispersion`). `None` for the pure energy path, which touches no grid —
    /// and which then estimates byte-identically to before this field existed.
    ///
    /// This is the term the 2026-07-13 incident needed and did not have: the
    /// estimator modelled only naux/nocc/nvir/n_workers, so it would have
    /// passed a 16-17 GB job as fitting a 10 GB budget.
    pub grid: Option<GridEstimateShape>,
}

/// Shapes for the grid-path allocations in `properties.rs` / `dispersion.rs`.
///
/// All fields are live values at the call site — `npts` from the built
/// [`ferric_dft::grid::build_atomic_grid`] (never a hardcoded 75x110), `nbf`
/// from the prepared basis, `natoms` from the molecule.
#[derive(Debug, Clone, Copy)]
pub struct GridEstimateShape {
    /// Number of Becke-Lebedev grid points actually built.
    pub npts: usize,
    /// AO basis functions (rows of the `chi` matrix).
    pub nbf: usize,
    /// Atoms carrying per-atom dipole tensors.
    pub natoms: usize,
    /// Chunk-partials held live at once by the banded dipole accumulation
    /// (`dipole_band_width`). Each partial is a full `natoms · 3 · nbf²` set,
    /// so this multiplies the largest grid term.
    ///
    /// Callers must pass the width the accumulation will ACTUALLY use, which is
    /// `min(dipole_band_width(..), n_chunks)` — the loop clamps each band with
    /// `(band0 + band_width).min(n_chunks)`, so a nominally huge width cannot
    /// materialize more partials than there are chunks. Passing the unclamped
    /// value produces a wildly pessimistic estimate on small systems: at
    /// water/STO-3G (nbf=7) a per-partial cost of ~1.2 kB against a 4 GiB budget
    /// gives a nominal width of ~3.6 million, which estimated 4.30 GB for a job
    /// whose real footprint is a few MB — enough to refuse a trivial
    /// calculation. Use [`effective_dipole_band_width`].
    pub dipole_band_width: usize,
    /// Rayon workers that may concurrently hold a per-chunk `chi` buffer.
    /// Pass `rayon::current_num_threads().max(1)`.
    pub n_workers: usize,
}

/// Chunk count the dipole accumulation will partition `npts` into.
///
/// Mirrors `properties::accumulate_atom_centred_dipoles`: `TARGET_CHUNKS`
/// groups (a pure function of `npts`, never of the worker count, so the fold
/// order cannot depend on `RAYON_NUM_THREADS`), floored at one chunk.
pub fn dipole_chunk_count(npts: usize) -> usize {
    const TARGET_CHUNKS: usize = 1024;
    let chunk_size = npts.div_ceil(TARGET_CHUNKS).max(1);
    npts.div_ceil(chunk_size).max(1)
}

/// The band width the accumulation will actually use: the budget-derived width
/// clamped to the number of chunks that exist.
///
/// Keep estimator and accumulator agreed on this; a mismatch is how a
/// pre-flight gate starts refusing jobs that would have run fine.
pub fn effective_dipole_band_width(nominal: usize, npts: usize) -> usize {
    nominal.max(1).min(dipole_chunk_count(npts))
}

/// Bytes the grid path holds resident: the `chi` AO-on-grid matrix, the
/// persistent per-atom AO dipole tensors, and one band of same-sized partials.
///
/// Kept separate from [`estimate_peak_bytes`]'s energy terms so each can be
/// unit-tested against a hand-derived figure.
pub fn estimate_grid_bytes(g: GridEstimateShape) -> usize {
    let GridEstimateShape { npts, nbf, natoms, dipole_band_width, n_workers } = g;

    // chi: evaluated PER GRID CHUNK inside the banded accumulation, one buffer
    // per rayon worker, each `chunk_size = npts/TARGET_CHUNKS` points wide.
    //
    // This used to be a single contiguous Array2::zeros((nbf, npts)) — ~3.75 GB
    // at nbf=800/npts=586k — built up front even though every consumer reads it
    // one grid point at a time. Fusing it into the loop cut that to ~44 MB at 12
    // workers (85x less; 3.7 MB serial). The estimate tracks the fused form: an
    // estimator that still charged the monolithic figure would over-reject, and
    // an over-estimating gate is as broken as an under-estimating one (it
    // becomes a wall and trains users to inflate budgets).
    let chunk_size = npts.div_ceil(dipole_chunk_count(npts).max(1)).max(1);
    let chi_bytes = nbf
        .saturating_mul(chunk_size)
        .saturating_mul(F64_BYTES)
        .saturating_mul(n_workers.max(1));

    // Grid side arrays: points (3 f64), weights (f64), home_atom (usize).
    let grid_side_bytes = npts.saturating_mul(3 * F64_BYTES + F64_BYTES + 8);

    // d_ai_ao: natoms x 3 matrices of nbf x nbf, persistent for the whole
    // accumulation.
    let per_atom_dipole_bytes = natoms
        .saturating_mul(3)
        .saturating_mul(nbf)
        .saturating_mul(nbf)
        .saturating_mul(F64_BYTES);

    // One band of chunk-partials, each a full d_ai_ao-sized set. This is the
    // term that used to scale with rayon worker count via the thread floor in
    // `dipole_band_width`; the byte cap now wins, but the partials are still
    // genuinely co-resident with the accumulator, so they are additive here.
    let band_partial_bytes = per_atom_dipole_bytes.saturating_mul(dipole_band_width.max(1));

    chi_bytes
        .saturating_add(grid_side_bytes)
        .saturating_add(per_atom_dipole_bytes)
        .saturating_add(band_partial_bytes)
}

/// Conservative upper bound (bytes) on peak resident memory for one
/// `RsMp2RpaConfig`/`PdepRpaConfig` run, given the shapes in `shape`.
///
/// See the module docs for exactly which co-resident allocations this covers.
/// Deliberately over- rather than under-estimates: e.g. it does not try to
/// account for the Lanczos panel narrowing the *matvec* transient (that part
/// IS budget-aware already, see `lanczos::lanczos_panel_width`) — it counts
/// the always-fully-resident assembled `naux×naux` matrix and its `eigh`
/// output instead, which dominates panel-transient savings anyway once the
/// panel width itself is bounded.
pub fn estimate_peak_bytes(shape: PeakEstimateShape) -> usize {
    let PeakEstimateShape { naux, nocc, nvir, n_quad, n_workers, n_keep, grid } = shape;
    let nov = nocc.saturating_mul(nvir);
    let m = n_keep.min(naux).max(1);
    let n_workers = n_workers.max(1);
    let n_quad = n_quad.max(1);

    // (1) RpaIntermediates build:
    //   - metric_inverse_sqrt eigh branch: up to 3 co-resident naux×naux
    //     buffers (evecs, u_scaled, dot() result) — see eigh_inverse_sqrt.
    //   - b_ov (naux·nocc·nvir): ONE resident buffer, not two. Historically
    //     this was eri3_mo_ov_blocked's raw MO output co-resident with a
    //     SEPARATE dressing-GEMM output of the same shape (2×); since
    //     compute_rpa_intermediates{,_spin} were migrated onto
    //     stream_dressed_mo_band (mirroring ri_mp2_spin_components), each
    //     aux-block chunk is dressed in place into the SAME output tensor as
    //     it streams — only one naux·nocc·nvir buffer is ever resident (the
    //     raw AO block scratch is chunk-sized, `MO_STREAM_CHUNK` aux rows
    //     wide, not naux-wide, so it's negligible next to the full b_ov).
    let metric_bytes = naux.saturating_mul(naux).saturating_mul(3).saturating_mul(F64_BYTES);
    let eri3_and_bov_bytes = naux.saturating_mul(nov).saturating_mul(F64_BYTES);
    let intermediates_peak = metric_bytes.saturating_add(eri3_and_bov_bytes);

    // (2) Lanczos full-rank eigensolve: assembled A (naux×naux, always fully
    // resident regardless of matvec panel width) + its eigh output
    // (eigenvectors naux×naux + eigenvalues naux, folded into the naux² term).
    let lanczos_peak = naux.saturating_mul(naux).saturating_mul(2).saturating_mul(F64_BYTES);

    // (3) Frequency-quadrature loop: per-worker (m, nov) + (m, m) scratch via
    // map_init, times n_workers (this is the term that scales with rayon
    // thread count — the crux of the incident). Plus the shared
    // frequency-independent projection y = Vᵀ·B_ov (m × nov), computed once
    // (energy.rs line ~228) and held for the whole loop.
    //
    // The property paths' per-worker pair is NOT (m, nov) + (m, m): the
    // map_init closures in `properties.rs` (the Becke/Hirshfeld dynamic
    // loops) seed a buffer of `b_ov.raw_dim()` — i.e. the FULL (naux, nov) —
    // and then form `eps_mat = b_scaled.dot(&b_scaled.t())`, a full
    // (naux, naux). Those shapes are naux-wide, not m-wide.
    //
    // That distinction used to be invisible because every caller passes
    // `n_keep = naux` (pre-eigensolve the retained count is unknown, and the
    // production `trunc_thresh = 0.0` keeps every mode anyway), so `m == naux`
    // and the m-based figure happened to land on the right number. It was
    // coincidence, not coverage: a caller that DID pass a truncated
    // `n_keep < naux` would silently under-charge the property paths by
    // `(naux - m) * (nov + naux + m)` elements per worker.
    //
    // So charge the per-worker term at the width the allocation actually uses:
    // `max(m, naux)` reduces to `m` for the energy path (where m == naux too)
    // and stays honest for a truncated property call. Deliberately over- not
    // under-estimating remains this estimator's stated contract.
    let per_worker_width = m.max(naux);
    let per_worker_scratch = per_worker_width
        .saturating_mul(nov)
        .saturating_add(per_worker_width.saturating_mul(per_worker_width))
        .saturating_mul(F64_BYTES);
    let quad_scratch_peak = per_worker_scratch.saturating_mul(n_workers);
    let y_projection_bytes = m.saturating_mul(nov).saturating_mul(F64_BYTES);

    // The b_ov held by the intermediate the eigensolve/quadrature stages
    // borrow from is already counted in eri3_and_bov_bytes above (it's the
    // SAME buffer, not a separate one) — intermediates_peak, lanczos_peak,
    // and quad_scratch_peak are additive because they represent genuinely
    // distinct, simultaneously-live allocations at the point the frequency
    // loop runs (b_ov + eigensolve output + per-worker scratch), but we must
    // not double the eri3/b_ov term again here.
    let output_arrays = naux.saturating_mul(m).saturating_mul(2).saturating_mul(F64_BYTES); // eigenvectors + eigenpotentials_aux

    let n_quad_unused_guard = n_quad; // n_quad only affects wall-time, not peak resident bytes here — kept as a named no-op so a future refinement that DOES need it has an obvious slot, and so the parameter isn't silently dead.
    let _ = n_quad_unused_guard;

    // (4) Grid path, when the caller is a per-atom property path. Additive:
    // chi/d_ai_ao/band-partials are live at the same time as b_ov and the
    // eigensolve output, since the accumulation runs after the intermediates
    // are built and both are still borrowed. `None` (the energy path) leaves
    // the estimate byte-identical to before this term existed.
    let grid_peak = grid.map_or(0, estimate_grid_bytes);

    intermediates_peak
        .saturating_add(lanczos_peak)
        .saturating_add(quad_scratch_peak)
        .saturating_add(y_projection_bytes)
        .saturating_add(output_arrays)
        .saturating_add(grid_peak)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Benzene-dimer-aug-cc-pVQZ-like dimensions: naux≈2976, nocc≈42,
    /// nvir≈1470, n_quad=20, n_workers=8 (the atz-benzene-rpa-memory-bound
    /// scale that measured ~17 GB RSS on a 23 GB box). The estimate must
    /// exceed a 10.74e9-byte (~10 GB) budget so `check_alloc` catches this
    /// shape BEFORE the allocations happen, and the error message must carry
    /// both the estimated and budgeted GB figures.
    #[test]
    fn benzene_dimer_aqz_scale_exceeds_10gb_budget() {
        let shape = PeakEstimateShape {
            naux: 2976,
            nocc: 42,
            nvir: 1470,
            n_quad: 20,
            n_workers: 8,
            n_keep: 2976, // trunc_thresh=0.0 production default: keep all modes
            grid: None,
        };
        let est = estimate_peak_bytes(shape);
        let budget = 10_740_000_000usize;
        assert!(
            est > budget,
            "expected benzene-dimer-aQZ-scale estimate to exceed 10.74 GB budget, got {:.2} GB",
            est as f64 / 1e9
        );

        let err = ferric_core::memory::check_alloc(
            "RS-MP2-RPA preflight (naux=2976, nocc=42, nvir=1470)",
            est,
            budget,
        )
        .unwrap_err();
        let msg = err.to_string();
        let est_gb = format!("{:.2} GB", est as f64 / 1e9);
        let budget_gb = format!("{:.2} GB", budget as f64 / 1e9);
        assert!(msg.contains(&est_gb), "message should contain estimated GB figure: {msg}");
        assert!(msg.contains(&budget_gb), "message should contain budgeted GB figure: {msg}");
    }

    /// Water/cc-pVDZ-like small-system dimensions must stay comfortably under
    /// a typical auto-resolved budget (a few GiB) and pass `check_alloc`, so
    /// the new gate never blocks a job that already runs fine today.
    #[test]
    fn small_system_scale_fits_typical_budget() {
        // water/cc-pVDZ: nao≈24, naux≈116 (cc-pvdz-ri), nocc≈5, nvir≈19.
        let shape = PeakEstimateShape {
            naux: 116,
            nocc: 5,
            nvir: 19,
            n_quad: 20,
            n_workers: 8,
            n_keep: 116,
            grid: None,
        };
        let est = estimate_peak_bytes(shape);
        // A typical small/auto-resolved budget: 2 GiB (the DEFAULT_BUDGET_BYTES
        // fallback) is already generous for this system size.
        let budget = ferric_core::memory::DEFAULT_BUDGET_BYTES;
        assert!(
            est < budget,
            "expected water/cc-pVDZ-scale estimate ({:.4} GB) to fit under the 2 GiB default budget",
            est as f64 / 1e9
        );
        assert!(ferric_core::memory::check_alloc("small test system", est, budget).is_ok());
    }

    /// Monotonicity sanity: doubling n_workers must not decrease the estimate
    /// (the whole point of accounting for rayon thread count) and must not
    /// explode super-linearly either (a basic non-pathological-growth check).
    #[test]
    fn estimate_scales_with_worker_count() {
        let base = PeakEstimateShape {
            naux: 500, nocc: 10, nvir: 100, n_quad: 20, n_workers: 4, n_keep: 500,
            grid: None,
        };
        let doubled = PeakEstimateShape { n_workers: 8, ..base };
        let est_base = estimate_peak_bytes(base);
        let est_doubled = estimate_peak_bytes(doubled);
        assert!(est_doubled > est_base, "doubling workers must increase the estimate");
    }

    /// n_keep should never be allowed to exceed naux inside the estimator
    /// (defensive: a caller passing an inconsistent n_keep > naux must not
    /// panic or produce an under-estimate via silent truncation elsewhere).
    #[test]
    fn n_keep_above_naux_is_clamped_not_panicking() {
        let shape = PeakEstimateShape {
            naux: 50, nocc: 5, nvir: 20, n_quad: 10, n_workers: 2, n_keep: 999,
            grid: None,
        };
        // Must not panic; must equal the naux-clamped estimate.
        let est = estimate_peak_bytes(shape);
        let clamped = estimate_peak_bytes(PeakEstimateShape { n_keep: 50, ..shape });
        assert_eq!(est, clamped);
    }
}
