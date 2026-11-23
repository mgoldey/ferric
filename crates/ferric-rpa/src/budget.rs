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
//! [`crate::budget::estimate_peak_bytes`] is a conservative (i.e. deliberately over-, never
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
    /// The caller needs `PdepRpaResult::inv_dielectric_freq`, i.e.
    /// `PdepRpaConfig::need_inv_dielectric_freq` is set.
    ///
    /// This is the ONLY circumstance under which [`PeakEstimateShape::n_quad`]
    /// is a peak-resident multiplier rather than a wall-time one, and getting
    /// that distinction right is the whole reason the field exists:
    ///
    /// * `false` (the energy path): quadrature points are processed per worker
    ///   via `map_init` and never retained, so `n_quad` genuinely drives only
    ///   wall time. `mwe_estimator_sees_grid.rs` documents this under "what is
    ///   NOT a defect" and says not to "fix" it — correctly, for that path.
    /// * `true`: `run_pdep_rpa` calls `energy::eval_inv_dielectric_matrices`,
    ///   which ends in `per_frequency(..)` whose `.collect()` RETAINS one owned
    ///   `(m, m)` matrix per frequency. `n_quad` then multiplies peak bytes.
    ///
    /// It is not a corner case: `ferric_gw::with_inv_dielectric` forces the
    /// flag on for EVERY GW method (Σ_c reads the stack), so every
    /// G0W0/evGW/COHSEX/BSE run was pre-flighted against an estimate that
    /// charged zero for it. At the benzene-dimer/aug-cc-pVTZ shape
    /// `ferric_rpa::lib`'s own comment quotes (naux = m = 2976, nov = 61,740,
    /// n_quad = 20, 8 workers) the omission was 1.42 GB of retained stack plus
    /// 11.76 GB of concurrent `y` clones.
    ///
    /// Energy-path callers pass `false` and estimate byte-identically to
    /// before this field existed.
    pub need_inv_dielectric: bool,
    /// AO basis size, for the `(naux, nao, nao)` three-index AO tensor that
    /// `ThreeIndexSource` holds resident while `b_ov` is streamed out of it.
    ///
    /// `0` contributes nothing, so a caller that has no meaningful `nao`
    /// estimates byte-identically to before this field existed.
    ///
    /// # Why the raw AO scratch is not "negligible"
    ///
    /// The raw AO scratch is "chunk-sized, `MO_STREAM_CHUNK` aux rows wide, not
    /// naux-wide, so it's negligible next to the full b_ov" for what the stream
    /// reads INTO, and not for what it reads FROM.
    /// `compute_rpa_intermediates` (rimp2.rs) does:
    ///
    /// ```text
    /// let budget_bytes = eri3_budget_bytes(config.memory_budget_bytes);
    /// let mut src = ThreeIndexSource::build(op, obs, dfbs, budget_bytes)?;
    /// let b_ov = stream_dressed_mo_band(&mut src, .., None)?;
    /// ```
    ///
    /// `src` is a SEPARATE full `(naux, nao, nao)` tensor, live for the whole
    /// streaming call, and `eri3_budget_bytes` is `resolve_budget_bytes(..)` —
    /// the WHOLE budget. `ThreeIndexSource::build`'s own gate is
    /// `if band * nao * nao * 8 <= budget_bytes { InCore }`, so it will admit
    /// an AO tensor occupying 100% of the same ceiling this estimate is then
    /// checked against.
    ///
    /// It is not a rounding term — it is LARGER than the `b_ov` that was being
    /// charged:
    ///
    /// ```text
    ///   system                naux   nao   AO tensor   charged b_ov   ratio
    ///   benzene/cc-pVDZ        420   114     0.04 GB       0.01 GB      6.7x
    ///   benzene/aug-cc-pVTZ   1512   414     2.07 GB       0.10 GB     20.8x
    ///   danuglipron/def2-SVP  2800   700    10.98 GB       1.23 GB      8.9x
    /// ```
    ///
    /// Charged at the IN-CORE size deliberately: that is the branch the budget
    /// permits, and a run that spills instead pays only a bounded band, so
    /// this is the conservative direction — the correct one for a pre-flight.
    pub nao: usize,
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
    let GridEstimateShape {
        npts,
        nbf,
        natoms,
        dipole_band_width,
        n_workers,
    } = g;

    // chi: evaluated PER GRID CHUNK inside the banded accumulation, one buffer
    // per rayon worker, each `chunk_size = npts/TARGET_CHUNKS` points wide.
    //
    // A single contiguous Array2::zeros((nbf, npts)) would be ~3.75 GB
    // at nbf=800/npts=586k, though every consumer reads it
    // one grid point at a time. Fusing it into the loop gives ~44 MB at 12
    // workers (85x less; 3.7 MB serial). The estimate tracks the fused form: an
    // estimator that charged the monolithic figure would over-reject, and
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
    // term that would scale with rayon worker count via the thread floor in
    // `dipole_band_width`; the byte cap wins, but the partials are still
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
/// The [`estimate_peak_bytes`] total, split into the part that has NO
/// streaming fallback and the part that does.
///
/// # Why this split exists (the hard/soft decision, measured not guessed)
///
/// The shared-pool brief's standing decision is: charge SOFT wherever a
/// batching/streaming fallback genuinely exists, and never refuse a job the
/// pre-migration tree completes. In this pipeline exactly one term has such a
/// fallback:
///
/// * **`quad_scratch`** — the per-worker `(m, nov) + (m, m)` frequency scratch.
///   [`crate::energy::eval_eigenvalues_at_frequencies_budgeted`] sizes it with
///   `quad_panel_width`, which already NARROWS the panel when the full width
///   does not fit and switches to `dielectric_matrix_from_projection_into_
///   panelled`. That is a real, already-implemented alternative code path, so
///   a `None` from `try_reserve` here has somewhere to go. It is also the term
///   that scales with the rayon thread count, i.e. the biggest and the most
///   over-estimated one.
///
/// Everything else is materialized in one shot with no narrower variant:
/// the `eigh`-branch metric buffers, the full `b_ov`, the resident
/// `(naux, nao, nao)` AO tensor, the assembled `naux x naux` Lanczos matrix
/// and its eigenvectors, the frequency-independent `y = V^T B_ov` projection,
/// the retained inverse-dielectric stack, and the grid planes. Those are HARD:
/// if they do not fit, failing fast with an occupancy breakdown is strictly
/// better than walking into the allocator.
///
/// `hard + soft + ao_tensor_bytes == estimate_peak_bytes(shape)` exactly,
/// which `hard_plus_soft_reconstructs_the_total` pins — a drift between the
/// split and the total would silently change what the gate admits.
///
/// # A soft charge must not starve the plane that asks next
///
/// Splitting hard from soft is necessary but not sufficient. The SOFT charge
/// also has to leave headroom for the HARD allocations that follow it, or an
/// optional plane can kill a mandatory one. `preflight_check_closed_shell`
/// returns, and then `compute_rpa_intermediates` hard-charges the AO tensor;
/// a bare `try_reserve(soft)` takes the scratch whenever it fits at that
/// instant and can leave nothing for it.
///
/// This is worker-dependent, so it hides on wide machines: the soft term
/// scales with `n_workers`, so at HIGH widths it fails on its own and the
/// fallback engages correctly, while at LOW widths it squeezes in and starves
/// the tensor. Measured (binary search for the smallest completing capacity,
/// water/cc-pVDZ) with a greedy soft charge:
///
/// ```text
///   workers   soft       hard+AO   hard+soft+AO   measured minimum
///   2          240_576   909_888      1_150_464   1_150_834  (starved)
///   4          481_152   909_888      1_391_040   1_391_076  (starved)
///   12       1_443_456   909_888      2_353_344     910_380  (panelled, OK)
/// ```
///
/// After the fix the minimum is `hard + AO` at all three. See the soft-charge
/// branch in `lib.rs::preflight_check_closed_shell` and the regression test
/// `the_soft_charge_leaves_room_for_the_mandatory_plane_that_asks_next`.
pub fn estimate_peak_split(shape: PeakEstimateShape) -> (usize, usize) {
    let total = estimate_peak_bytes(shape);
    let soft = quad_scratch_bytes(shape);
    let hard = total
        .saturating_sub(soft)
        .saturating_sub(ao_tensor_bytes(shape));
    (hard, soft)
}

/// The resident `(naux, nao, nao)` three-index AO tensor term of
/// [`estimate_peak_bytes`].
///
/// # Why this is SUBTRACTED from the pool charge but KEPT in the ceiling check
///
/// MEASURED, not assumed. `ferric_integrals::three_index_source::
/// ThreeIndexSource` ALREADY debits the shared pool for exactly these bytes,
/// under the label `"DF 3-index (P|mn) in-core"`, and holds the guard in its
/// own `_charge` field for the tensor's whole lifetime. Charging it again from
/// this crate's preflight double-counts it: the RPA gate reserves
/// `naux*nao^2*8`, `compute_rpa_intermediates` then builds the source, which
/// asks for the same bytes against a pool already holding them, and the second
/// ask self-refuses.
///
/// Measured on water/cc-pVDZ + cc-pvdz-ri (naux=84, nao=24) at a 1_137_360 B
/// capacity, with the AO term double-charged:
///
/// ```text
/// memory pool exhausted: "DF 3-index (P|mn) in-core" needs 0.000 GB but only
/// 0.000 GB of the 0.001 GB pool is free
///   memory pool: 0.001 GB outstanding of 0.001 GB
///          0.001 GB  PDEP-RPA preflight (...) [in-core]
/// ```
///
/// -- the incumbent in the breakdown IS this crate's own preflight, holding
/// the bytes the integrals layer is about to ask for. That is the exact
/// self-refusal the ksdft migration hit on its grid cache, for the same
/// reason. The term stays in `estimate_peak_bytes` (and therefore in the
/// `check_alloc` ceiling test) because that gate is a per-method PRE-FLIGHT
/// upper bound with no cross-crate ledger, and dropping it there would weaken
/// an existing, deliberately conservative guard.
pub fn ao_tensor_bytes(shape: PeakEstimateShape) -> usize {
    shape
        .naux
        .saturating_mul(shape.nao)
        .saturating_mul(shape.nao)
        .saturating_mul(F64_BYTES)
}

/// The per-worker frequency-quadrature scratch term of [`estimate_peak_bytes`].
///
/// Factored out (rather than recomputed at the call site) so the split above
/// and the total below cannot drift: both read this one function.
pub fn quad_scratch_bytes(shape: PeakEstimateShape) -> usize {
    let m = shape.n_keep.min(shape.naux).max(1);
    let nov = shape.nocc.saturating_mul(shape.nvir);
    let per_worker_width = m.max(shape.naux);
    per_worker_width
        .saturating_mul(nov)
        .saturating_add(per_worker_width.saturating_mul(per_worker_width))
        .saturating_mul(F64_BYTES)
        .saturating_mul(shape.n_workers.max(1))
}

/// [`quad_scratch_bytes`] at the nov-panel `width` the frequency loop
/// actually allocates (`energy::quad_panel_width`); `width >= nov` is the
/// full-width term itself.
pub fn quad_scratch_bytes_at_width(shape: PeakEstimateShape, width: usize) -> usize {
    let m = shape.n_keep.min(shape.naux).max(1);
    let nov = shape.nocc.saturating_mul(shape.nvir);
    let rows = m.max(shape.naux);
    let cols = width.min(nov);
    rows.saturating_mul(cols)
        .saturating_add(rows.saturating_mul(rows))
        .saturating_mul(F64_BYTES)
        .saturating_mul(shape.n_workers.max(1))
}

/// The planes only the static-eigenbasis route allocates: the assembled
/// `naux x naux` ε̃(0) and its eigenvectors (`2 naux²`), the projection
/// `y = Vᵀ·B_ov` (`m·nov`), and the eigenvector outputs (eigenvectors and
/// eigenpotentials, `2 naux·m`).
pub fn static_eigensolve_bytes(shape: PeakEstimateShape) -> usize {
    let naux = shape.naux;
    let m = shape.n_keep.min(naux).max(1);
    let nov = shape.nocc.saturating_mul(shape.nvir);
    let eigensolve = naux.saturating_mul(naux).saturating_mul(2);
    let y_projection = m.saturating_mul(nov);
    let outputs = naux.saturating_mul(m).saturating_mul(2);
    eigensolve
        .saturating_add(y_projection)
        .saturating_add(outputs)
        .saturating_mul(F64_BYTES)
}

/// Which planes a closed-shell energy run allocates (see `crate::energy_route`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyRoutePlanes {
    /// The PDEP eigensolve and the frequency loop in its eigenbasis.
    StaticEigenbasis,
    /// The frequency loop in the aux basis: no eigensolve planes, plus one
    /// `naux x naux` ε̃(0) for the keep-all Cholesky check.
    AuxBasisLogDet,
}

/// What the closed-shell preflight charges: the ceiling-check `total` and its
/// pool split (`hard` refuses, `soft` is the per-worker scratch, and the AO
/// tensor is `total - hard - soft`, charged by `ferric-integrals`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunCharge {
    /// Bytes the ceiling check compares to the budget.
    pub total: usize,
    /// Planes with no streaming fallback.
    pub hard: usize,
    /// The per-worker frequency scratch at the width the loop allocates.
    pub soft: usize,
}

/// The preflight charge of a closed-shell energy run on `planes`, with the
/// per-worker scratch at `panel_width` (the width the frequency loop will
/// allocate under the run's budget) rather than the full `nov`.
///
/// At `panel_width >= nov` on [`EnergyRoutePlanes::StaticEigenbasis`] this is
/// exactly [`estimate_peak_bytes`] / [`estimate_peak_split`]; a budgeted run
/// whose loop panels is charged the panel it allocates.
pub fn closed_shell_run_charge(
    shape: PeakEstimateShape,
    planes: EnergyRoutePlanes,
    panel_width: usize,
) -> RunCharge {
    let (static_hard, _) = estimate_peak_split(shape);
    let hard = match planes {
        EnergyRoutePlanes::StaticEigenbasis => static_hard,
        EnergyRoutePlanes::AuxBasisLogDet => static_hard
            .saturating_sub(static_eigensolve_bytes(shape))
            .saturating_add(
                shape
                    .naux
                    .saturating_mul(shape.naux)
                    .saturating_mul(F64_BYTES),
            ),
    };
    let soft = quad_scratch_bytes_at_width(shape, panel_width);
    RunCharge {
        total: hard
            .saturating_add(soft)
            .saturating_add(ao_tensor_bytes(shape)),
        hard,
        soft,
    }
}

/// What the Laplace-chi0 dielectric kernel
/// (`crate::laplace_chi0::dielectric_matrix_laplace_into`) allocates per
/// active frequency worker beyond the full-width loop scratch
/// ([`quad_scratch_bytes_at_width`] at `nov`): the larger of its two branches.
/// The Laplace branch keeps a copy of the unscaled projection (`m·nov`); the
/// high-frequency dense fallback (`sternheimer::dielectric_matrix`) builds the
/// projection and its scaled copy (`2 m·nov`) and a fresh `m²` matrix. Both
/// are charged with `m = naux`, as the loop scratch is.
pub fn laplace_kernel_extra_bytes(shape: PeakEstimateShape) -> usize {
    let m = shape.naux.max(shape.n_keep.min(shape.naux));
    let nov = shape.nocc.saturating_mul(shape.nvir);
    m.saturating_mul(nov)
        .saturating_mul(2)
        .saturating_add(m.saturating_mul(m))
        .saturating_mul(F64_BYTES)
        .saturating_mul(shape.n_workers.max(1))
}

/// [`ScanCharge`] with `extra` bytes more in the dRPA phase's per-worker
/// scratch (`drpa_soft` and the total).
pub fn with_drpa_scratch(base: ScanCharge, extra: usize) -> ScanCharge {
    let drpa_soft = base.drpa_soft.saturating_add(extra);
    let build = base.ao_resident.saturating_add(base.stream_scratch);
    let drpa = base.drpa_hard.saturating_add(drpa_soft);
    ScanCharge {
        total: base
            .persistent
            .saturating_add(build.max(base.spin_scratch).max(drpa)),
        drpa_soft,
        ..base
    }
}

/// Planes the rs-mp2-rpa scan holds for its whole run: the metric buffers
/// (`3 naux²`, as in [`estimate_peak_bytes`]) and one `b_ov` (`naux·nov`).
/// The driver keeps one `RpaIntermediates` at a time, so one `b_ov`.
pub fn scan_persistent_bytes(shape: PeakEstimateShape) -> usize {
    let nov = shape.nocc.saturating_mul(shape.nvir);
    shape
        .naux
        .saturating_mul(shape.naux)
        .saturating_mul(3)
        .saturating_add(shape.naux.saturating_mul(nov))
        .saturating_mul(F64_BYTES)
}

/// What the rs-mp2-rpa preflight charges, phase by phase.
///
/// Every operator of a scan (Coulomb, then the long- and short-range pair of
/// each ω) runs the same three phases on top of [`scan_persistent_bytes`],
/// and the AO tensor of one phase is dropped before the next starts:
///
/// ```text
///   build   AO tensor at its tier + the stream chunk   (compute_rpa_intermediates)
///   spin    per-worker g_i, nocc·nvir² each            (spin_components_from_b_ov)
///   drpa    static eigensolve planes + freq scratch     (run_pdep_rpa_from_intermediates)
/// ```
///
/// All operators share one shape, so the charge is the same for every ω and
/// both formulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanCharge {
    /// The ceiling-check figure: `persistent + max(build, spin, drpa)`.
    pub total: usize,
    /// [`scan_persistent_bytes`].
    pub persistent: usize,
    /// The budget handed to `ThreeIndexSource::build`: what is left after the
    /// persistent planes and the stream chunk. The AO tier is chosen at it.
    pub ao_budget: usize,
    /// The tier the AO tensor takes at `ao_budget`.
    pub ao_tier: ferric_integrals::three_index_source::jk_storage::Tier,
    /// The AO tensor's resident peak at that tier.
    pub ao_resident: usize,
    /// The `mo_blk` chunk of the stream.
    pub stream_scratch: usize,
    /// The spin-component phase: `n_workers · nocc · nvir² · 8`.
    pub spin_scratch: usize,
    /// The dRPA planes with no fallback ([`static_eigensolve_bytes`]).
    pub drpa_hard: usize,
    /// The dRPA per-worker frequency scratch at the loop's panel width.
    pub drpa_soft: usize,
}

/// [`ScanCharge`] for `shape` under `budget_bytes`, with the frequency
/// scratch at `panel_width` and the stream chunk at `stream_scratch` bytes.
///
/// The AO tier is planned with `ferric_integrals::three_index_source::
/// planned_residency`, which is the decision `ThreeIndexSource::build` makes,
/// so the charge follows the tier the build takes: the whole tensor in core,
/// the packed triangle, or two spill blocks.
pub fn scan_charge(
    shape: PeakEstimateShape,
    budget_bytes: usize,
    panel_width: usize,
    stream_scratch: usize,
) -> ScanCharge {
    let persistent = scan_persistent_bytes(shape);
    let ao_budget = budget_bytes
        .saturating_sub(persistent)
        .saturating_sub(stream_scratch);
    let (ao_tier, ao_resident) =
        ferric_integrals::three_index_source::planned_residency(shape.nao, shape.naux, ao_budget);
    let spin_scratch = shape
        .nocc
        .saturating_mul(shape.nvir)
        .saturating_mul(shape.nvir)
        .saturating_mul(F64_BYTES)
        .saturating_mul(shape.n_workers.max(1));
    let drpa_hard = static_eigensolve_bytes(shape);
    let drpa_soft = quad_scratch_bytes_at_width(shape, panel_width);
    let build = ao_resident.saturating_add(stream_scratch);
    let drpa = drpa_hard.saturating_add(drpa_soft);
    ScanCharge {
        total: persistent.saturating_add(build.max(spin_scratch).max(drpa)),
        persistent,
        ao_budget,
        ao_tier,
        ao_resident,
        stream_scratch,
        spin_scratch,
        drpa_hard,
        drpa_soft,
    }
}

/// [`ScanCharge`] of the open-shell rs-mp2-rpa driver: [`scan_charge`]'s
/// three phases on top of TWO co-resident spin sets.
///
/// Per operator the alpha intermediates are built, then the beta ones with
/// alpha still resident, and both live through the spin and dRPA phases.
/// So:
///
/// ```text
///   persistent  scan_persistent_bytes(alpha) + scan_persistent_bytes(beta)
///   build       AO tensor at its tier (once: `shape_b.nao` is 0) + stream
///   spin        the larger spin's n_workers · nocc · nvir² (the U-MP2 and
///               D_σσ kernels run one spin at a time)
///   drpa        static_eigensolve_bytes and the frequency scratch of each
///               spin, summed
/// ```
///
/// The two eigensolve terms each count the `2 naux²` ε̃(0) planes, which the
/// spin-summed eigensolve allocates once, so the dRPA phase over-counts by
/// `2 naux² + 2 naux·m` doubles. That is the over-estimating direction. On a
/// closed-shell shape pair (`shape_a == shape_b` but for `nao`) the
/// persistent term is twice the closed-shell one, as it must be: two `b_ov`
/// are resident where the closed-shell driver holds one.
pub fn scan_charge_unrestricted(
    (shape_a, shape_b): (PeakEstimateShape, PeakEstimateShape),
    budget_bytes: usize,
    (panel_a, panel_b): (usize, usize),
    stream_scratch: usize,
) -> ScanCharge {
    let persistent = scan_persistent_bytes(shape_a).saturating_add(scan_persistent_bytes(shape_b));
    let ao_budget = budget_bytes
        .saturating_sub(persistent)
        .saturating_sub(stream_scratch);
    let nao = shape_a.nao.max(shape_b.nao);
    let (ao_tier, ao_resident) =
        ferric_integrals::three_index_source::planned_residency(nao, shape_a.naux, ao_budget);
    let spin = |s: PeakEstimateShape| {
        s.nocc
            .saturating_mul(s.nvir)
            .saturating_mul(s.nvir)
            .saturating_mul(F64_BYTES)
            .saturating_mul(s.n_workers.max(1))
    };
    let spin_scratch = spin(shape_a).max(spin(shape_b));
    let drpa_hard =
        static_eigensolve_bytes(shape_a).saturating_add(static_eigensolve_bytes(shape_b));
    let drpa_soft = quad_scratch_bytes_at_width(shape_a, panel_a)
        .saturating_add(quad_scratch_bytes_at_width(shape_b, panel_b));
    let build = ao_resident.saturating_add(stream_scratch);
    let drpa = drpa_hard.saturating_add(drpa_soft);
    ScanCharge {
        total: persistent.saturating_add(build.max(spin_scratch).max(drpa)),
        persistent,
        ao_budget,
        ao_tier,
        ao_resident,
        stream_scratch,
        spin_scratch,
        drpa_hard,
        drpa_soft,
    }
}

/// Bytes a closed-shell rs-mp2-rpa scan holds through its eigenbasis dRPA
/// phase besides the frequency scratch: the persistent planes
/// ([`scan_persistent_bytes`]), the Coulomb `b_ov` a terf scan keeps
/// (`held_b_ov_bytes`, 0 otherwise) and the static eigensolve planes
/// ([`static_eigensolve_bytes`]).
pub fn scan_dielectric_resident_bytes(shape: PeakEstimateShape, held_b_ov_bytes: usize) -> usize {
    scan_persistent_bytes(shape)
        .saturating_add(held_b_ov_bytes)
        .saturating_add(static_eigensolve_bytes(shape))
}

/// What a frequency loop's per-worker scratch may use: what `budget` leaves
/// beside the `resident` planes (everything the caller charges besides that
/// scratch, a projection the loop holds included), so that
/// `resident + scratch ≤ budget`. `None` (no set budget) runs the loop at full
/// width. A `resident` at or over the budget is
/// [`PanelAllowance::NoRoom`](crate::energy::PanelAllowance::NoRoom): the
/// narrowest panel, which a pre-flight charges and the rs-mp2-rpa driver
/// refuses before any loop runs ([`check_scan_resident`]).
///
/// Sizing the panel against the whole budget and then adding the resident
/// planes on top put every panelled run's demand at `budget + resident`, so
/// the ceiling check refused runs the narrower panel would have fitted.
pub fn panel_budget_beside(
    budget: Option<usize>,
    resident: usize,
) -> crate::energy::PanelAllowance {
    use crate::energy::PanelAllowance;
    match budget {
        None => PanelAllowance::Unbudgeted,
        Some(b) if resident >= b => PanelAllowance::NoRoom,
        Some(b) => PanelAllowance::Scratch(b - resident),
    }
}

/// The planes a closed-shell or open-shell scan holds through its dRPA phase,
/// for [`check_scan_resident`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanResident {
    /// Aux metric + the current operator's `b_ov` (per spin).
    pub persistent: usize,
    /// The Coulomb `b_ov` a terf scan keeps for every point (0 otherwise).
    pub held_coulomb: usize,
    /// The static eigensolve planes.
    pub eigensolve: usize,
}

/// The refusal when the planes a scan holds through its dRPA phase
/// ([`ScanResident`]) leave nothing of `budget_bytes` for a frequency panel.
pub fn check_scan_resident(
    label: &str,
    planes: ScanResident,
    budget_bytes: usize,
) -> Result<(), ferric_core::error::FerricError> {
    let ScanResident {
        persistent,
        held_coulomb,
        eigensolve,
    } = planes;
    let kept = persistent.saturating_add(held_coulomb);
    let resident = kept.saturating_add(eigensolve);
    if resident < budget_bytes {
        return Ok(());
    }
    let gb = |b: usize| b as f64 / 1e9;
    let what = if kept >= budget_bytes {
        "the persistent planes alone"
    } else {
        "the persistent planes and the dRPA eigensolve planes"
    };
    let held = if held_coulomb > 0 {
        format!(
            " + the Coulomb b_ov a terf scan holds for every point {:.3} GB",
            gb(held_coulomb)
        )
    } else {
        String::new()
    };
    Err(ferric_core::error::FerricError::MemoryExhausted(format!(
        "{label}: {what} exceed the memory budget: persistent planes (aux metric + b_ov) \
         {:.3} GB{held} + static eigensolve planes {:.3} GB = {:.3} GB against a budget of \
         {:.3} GB, which leaves no room for the frequency panel (it is sized against the \
         budget minus these planes). Raise [memory] budget_gb / FERRIC_MEM_BUDGET_GB, or \
         shrink the auxiliary basis or the active space (frozen_core)",
        gb(persistent),
        gb(eigensolve),
        gb(resident),
        gb(budget_bytes),
    )))
}

/// The per-worker scratch of the signed log-det dRPA
/// (`crate::energy::eval_signed_trace_log_summands`) at panel width `width`:
/// a `2 naux × width` panel and the `(2 naux)²` dielectric.
pub fn signed_quad_scratch_bytes(naux: usize, width: usize, n_workers: usize) -> usize {
    let m = naux.saturating_mul(2);
    m.saturating_mul(width)
        .saturating_add(m.saturating_mul(m))
        .saturating_mul(F64_BYTES)
        .saturating_mul(n_workers.max(1))
}

/// A [`ScanCharge`] adjusted for the terf long-range fit (Coulomb RI minus
/// terfc RI, `ferric_mp2::rimp2::SignedBOv`).
///
/// The Coulomb `b_ov` of each spin (`held_b_ov_bytes` in all) stays resident
/// for the whole scan and joins the persistent planes, which shrinks the
/// budget left for the AO tensor, so its tier is re-planned. The long-range
/// dRPA runs the signed log-det, whose scratch (`signed_drpa_bytes`, from
/// [`signed_quad_scratch_bytes`]) is its own phase: it forms no static
/// eigenbasis, so the eigensolve planes are not live beside it, and the dRPA
/// phase is the larger of `drpa_hard + drpa_soft` and the signed scratch.
/// `drpa_soft` becomes the larger of the two scratches (the pool's soft
/// reservation around either loop). The terf three-index build itself is
/// gone, so no phase is added.
pub fn with_signed_lr(
    base: ScanCharge,
    (nao, naux): (usize, usize),
    budget_bytes: usize,
    held_b_ov_bytes: usize,
    signed_drpa_bytes: usize,
) -> ScanCharge {
    let persistent = base.persistent.saturating_add(held_b_ov_bytes);
    let ao_budget = budget_bytes
        .saturating_sub(persistent)
        .saturating_sub(base.stream_scratch);
    let (ao_tier, ao_resident) =
        ferric_integrals::three_index_source::planned_residency(nao, naux, ao_budget);
    let drpa_soft = base.drpa_soft.max(signed_drpa_bytes);
    let build = ao_resident.saturating_add(base.stream_scratch);
    let drpa = base
        .drpa_hard
        .saturating_add(base.drpa_soft)
        .max(signed_drpa_bytes);
    ScanCharge {
        total: persistent.saturating_add(build.max(base.spin_scratch).max(drpa)),
        persistent,
        ao_budget,
        ao_tier,
        ao_resident,
        drpa_soft,
        ..base
    }
}

/// Upper-bound estimate, in bytes, of the peak resident memory of a PDEP-RPA
/// run with the given shape.
///
/// Sums the terms live at the frequency loop: the RI intermediates (three
/// `naux²` metric buffers, the `naux·nocc·nvir` `b_ov`, and the
/// `naux·nao²` AO three-index tensor), the full-rank Lanczos eigensolve
/// (`2·naux²`), the per-worker quadrature scratch
/// ([`quad_scratch_bytes`]), the `y` projection, the eigenvector outputs,
/// the retained inverse-dielectric stack and its per-worker clones when
/// `need_inv_dielectric` is set, and the per-atom grid term when `grid` is
/// `Some`. All arithmetic saturates. The contract is to over- rather than
/// under-estimate.
pub fn estimate_peak_bytes(shape: PeakEstimateShape) -> usize {
    let PeakEstimateShape {
        naux,
        nocc,
        nvir,
        n_quad,
        n_workers,
        n_keep,
        grid,
        need_inv_dielectric,
        nao,
    } = shape;
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
    //     raw AO block SCRATCH is chunk-sized, `MO_STREAM_CHUNK` aux rows
    //     wide, not naux-wide, so it's negligible next to the full b_ov).
    //
    //     CAVEAT: that is the buffer the stream reads
    //     INTO. The `ThreeIndexSource` it reads FROM is a separate full
    //     `(naux, nao, nao)` tensor, live across the whole streaming call and
    //     admitted in-core at up to 100% of this very budget. It is charged
    //     below as `ao_tensor_bytes` — see `PeakEstimateShape::nao`.
    let metric_bytes = naux
        .saturating_mul(naux)
        .saturating_mul(3)
        .saturating_mul(F64_BYTES);
    let eri3_and_bov_bytes = naux.saturating_mul(nov).saturating_mul(F64_BYTES);
    // The resident three-index AO tensor. `nao == 0` (a caller that does not
    // supply it) contributes nothing, keeping such callers byte-identical.
    let ao_tensor_bytes = naux
        .saturating_mul(nao)
        .saturating_mul(nao)
        .saturating_mul(F64_BYTES);
    let intermediates_peak = metric_bytes
        .saturating_add(eri3_and_bov_bytes)
        .saturating_add(ao_tensor_bytes);

    // (2) The static eigensolve and what it leaves resident for the loop
    // ([`static_eigensolve_bytes`]): the assembled ε̃(0) and its eigenvectors,
    // the projection y = Vᵀ·B_ov, and the eigenvector outputs.
    let lanczos_peak = static_eigensolve_bytes(shape);

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
    // That distinction is invisible when every caller passes
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

    // The b_ov held by the intermediate the eigensolve/quadrature stages
    // borrow from is already counted in eri3_and_bov_bytes above (it's the
    // SAME buffer, not a separate one) — intermediates_peak, lanczos_peak,
    // and quad_scratch_peak are additive because they represent genuinely
    // distinct, simultaneously-live allocations at the point the frequency
    // loop runs (b_ov + eigensolve output + per-worker scratch), but we must
    // not double the eri3/b_ov term again here.

    // (3b) The per-frequency inverse-dielectric stack, when the caller keeps it.
    //
    // This is the slot the old `n_quad_unused_guard` no-op reserved. Its
    // comment said "n_quad only affects wall-time, not peak resident bytes
    // here" — true on the energy path, where `per_frequency`'s results are
    // consumed and dropped, and FALSE under `need_inv_dielectric_freq`, where
    // `eval_inv_dielectric_matrices` collects them into a retained
    // `Vec<Array2>`. `ferric_gw::with_inv_dielectric` sets that flag for every
    // GW method, so the common GW path was the one being mis-estimated.
    //
    // Two terms, both measured at the benzene-dimer/aTZ shape:
    //   * the retained stack, n_quad owned (m, m) matrices          (1.42 GB)
    //   * the concurrent `y.clone()` inside
    //     `dielectric_matrix_from_projection`, one (m, nov) per live
    //     worker — saturating at n_workers, since rayon runs at most
    //     that many closures at once                               (11.76 GB)
    //
    // WHY `clones` STAYS IN THE **HARD** HALF (checked 2026-09-18).
    //
    // The ferric-gw migration binary-searched the smallest completing pool
    // capacity and found it worker-DEPENDENT, with the whole spread traceable
    // to this term and none of it to ferric-gw:
    //
    //     workers   min completing   delta vs w=1   min(n_quad,w) x 6_720
    //     1             899_808              0            6_720 x 1
    //     2             906_528          6_720           13_440 (+1 unit)
    //     4             919_968         20_160           26_880 (+3 units)
    //     12            946_848         47_040           53_760 (+7 units)
    //     24            946_848         47_040          saturated at w=8
    //
    // It saturates at w = 8 = n_quad, which is this expression's `min` and
    // nothing else's. The natural reaction is "per-worker scratch belongs in
    // the SOFT half" -- and that would be WRONG here. A soft charge is only
    // honest if its `None` branch really streams. The in-place
    // `dielectric_matrix_from_projection_into` exists, but every call site is
    // inside `#[cfg(test)]`; the production `eval_inv_dielectric_matrices`
    // path still clones. Softening this would declare a fallback that does
    // not exist, and the run would allocate the bytes anyway after the gate
    // waved it through -- the exact overcommit the pool was built to stop.
    //
    // So: HARD until the in-place variant is actually wired into the
    // production path. Wiring it is the real fix and would shrink the term
    // rather than reclassify it.
    //
    // The clone term is the larger of the two and is charged rather than
    // designed away here on purpose: a scratch-reusing
    // `dielectric_matrix_from_projection_into` already exists beside the
    // cloning variant, so removing the clone is a real follow-up — but an
    // estimator must charge what the code allocates TODAY, not what a future
    // refactor might. Under-charging is how a gate approves a job that then
    // gets OOM-killed.
    let inv_dielectric_bytes = if need_inv_dielectric {
        let stack = n_quad
            .saturating_mul(m)
            .saturating_mul(m)
            .saturating_mul(F64_BYTES);
        let clones = n_quad
            .min(n_workers)
            .saturating_mul(m)
            .saturating_mul(nov)
            .saturating_mul(F64_BYTES);
        stack.saturating_add(clones)
    } else {
        0
    };

    // (4) Grid path, when the caller is a per-atom property path. Additive:
    // chi/d_ai_ao/band-partials are live at the same time as b_ov and the
    // eigensolve output, since the accumulation runs after the intermediates
    // are built and both are still borrowed. `None` (the energy path) leaves
    // the estimate byte-identical to before this term existed.
    let grid_peak = grid.map_or(0, estimate_grid_bytes);

    intermediates_peak
        .saturating_add(lanczos_peak)
        .saturating_add(quad_scratch_peak)
        .saturating_add(grid_peak)
        .saturating_add(inv_dielectric_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_charge_shape() -> PeakEstimateShape {
        PeakEstimateShape {
            naux: 700,
            nocc: 25,
            nvir: 169,
            n_quad: 20,
            n_workers: 6,
            n_keep: 700,
            grid: None,
            need_inv_dielectric: false,
            nao: 202,
        }
    }

    /// At full panel width the static route's charge IS the estimate the
    /// ceiling check used before the panel width was threaded through, and a
    /// narrower panel lowers only the per-worker scratch. Mutation-tested:
    /// charging `quad_scratch_bytes(shape)` (full width) in
    /// `closed_shell_run_charge` fails the second assertion.
    #[test]
    fn the_run_charge_follows_the_panel_width_the_loop_allocates() {
        let shape = run_charge_shape();
        let nov = shape.nocc * shape.nvir;
        let full = closed_shell_run_charge(shape, EnergyRoutePlanes::StaticEigenbasis, nov);
        assert_eq!(full.total, estimate_peak_bytes(shape));
        let narrow = closed_shell_run_charge(shape, EnergyRoutePlanes::StaticEigenbasis, 64);
        assert_eq!(narrow.hard, full.hard);
        assert_eq!(narrow.soft, quad_scratch_bytes_at_width(shape, 64));
        assert!(narrow.total < full.total);
        assert_eq!(full.total - narrow.total, full.soft - narrow.soft);
    }

    /// The aux route drops the static eigensolve planes and keeps one
    /// `naux x naux` matrix; everything else is the static route's charge.
    #[test]
    fn the_aux_route_charges_no_static_eigensolve() {
        let shape = run_charge_shape();
        let nov = shape.nocc * shape.nvir;
        let stat = closed_shell_run_charge(shape, EnergyRoutePlanes::StaticEigenbasis, nov);
        let aux = closed_shell_run_charge(shape, EnergyRoutePlanes::AuxBasisLogDet, nov);
        let square = shape.naux * shape.naux * F64_BYTES;
        assert_eq!(
            stat.hard - aux.hard,
            static_eigensolve_bytes(shape) - square
        );
        assert_eq!(aux.soft, stat.soft);
    }

    /// The hard/soft split must reconstruct the total EXACTLY. If it drifts,
    /// the pool gate admits a different number of bytes than the ceiling
    /// check approved, silently. Mutation-tested: changing `quad_scratch_bytes`
    /// to use `n_workers` instead of `n_workers.max(1)`, or dropping the
    /// `m.max(naux)` widening, fails this.
    #[test]
    fn hard_plus_soft_reconstructs_the_total() {
        for shape in [
            PeakEstimateShape {
                naux: 2976,
                nocc: 42,
                nvir: 1470,
                n_quad: 20,
                n_workers: 8,
                n_keep: 2976,
                grid: None,
                need_inv_dielectric: false,
                nao: 414,
            },
            PeakEstimateShape {
                naux: 116,
                nocc: 5,
                nvir: 19,
                n_quad: 8,
                n_workers: 1,
                n_keep: 60,
                grid: Some(GridEstimateShape {
                    npts: 12_000,
                    nbf: 24,
                    natoms: 3,
                    dipole_band_width: 4,
                    n_workers: 1,
                }),
                need_inv_dielectric: true,
                nao: 24,
            },
        ] {
            let (hard, soft) = estimate_peak_split(shape);
            assert_eq!(
                hard.saturating_add(soft)
                    .saturating_add(ao_tensor_bytes(shape)),
                estimate_peak_bytes(shape),
                "hard + soft + the integrals-owned AO tensor must reconstruct \
                 the total exactly -- a drift here means the pool gate debits \
                 a different number than the ceiling check approved"
            );
            assert!(
                ao_tensor_bytes(shape) > 0,
                "REACHABILITY: both fixtures set nao > 0, so the term this \
                 split deliberately EXCLUDES must be nonzero -- otherwise the \
                 exclusion is untested"
            );
            assert!(soft > 0, "the quadrature scratch is never zero-sized");
            assert!(
                hard > 0,
                "the hard remainder must not be swallowed by the soft term"
            );
        }
    }

    /// Benzene-dimer-aug-cc-pVQZ-like dimensions: naux≈2976, nocc≈42,
    /// nvir≈1470, n_quad=20, n_workers=8 (the atz-benzene-rpa-memory-bound
    /// scale that measured ~17 GB RSS on a 23 GB box). The estimate must
    /// exceed a 10.74e9-byte (~10 GB) budget so `check_alloc` catches this
    /// shape BEFORE the allocations happen, and the error message must carry
    /// both the estimated and budgeted GB figures.
    #[test]
    fn benzene_dimer_aqz_scale_exceeds_10gb_budget() {
        let shape = PeakEstimateShape {
            // AO tensor not modelled by this case.
            nao: 0,
            need_inv_dielectric: false,
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
        assert!(
            msg.contains(&est_gb),
            "message should contain estimated GB figure: {msg}"
        );
        assert!(
            msg.contains(&budget_gb),
            "message should contain budgeted GB figure: {msg}"
        );
    }

    /// Water/cc-pVDZ-like small-system dimensions must stay comfortably under
    /// a typical auto-resolved budget (a few GiB) and pass `check_alloc`, so
    /// the new gate never blocks a job that already runs fine today.
    #[test]
    fn small_system_scale_fits_typical_budget() {
        // water/cc-pVDZ: nao≈24, naux≈116 (cc-pvdz-ri), nocc≈5, nvir≈19.
        let shape = PeakEstimateShape {
            // AO tensor not modelled by this case.
            nao: 0,
            need_inv_dielectric: false,
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
            // AO tensor not modelled by this case.
            nao: 0,
            need_inv_dielectric: false,
            naux: 500,
            nocc: 10,
            nvir: 100,
            n_quad: 20,
            n_workers: 4,
            n_keep: 500,
            grid: None,
        };
        let doubled = PeakEstimateShape {
            n_workers: 8,
            ..base
        };
        let est_base = estimate_peak_bytes(base);
        let est_doubled = estimate_peak_bytes(doubled);
        assert!(
            est_doubled > est_base,
            "doubling workers must increase the estimate"
        );
    }

    /// n_keep should never be allowed to exceed naux inside the estimator
    /// (defensive: a caller passing an inconsistent n_keep > naux must not
    /// panic or produce an under-estimate via silent truncation elsewhere).
    #[test]
    fn n_keep_above_naux_is_clamped_not_panicking() {
        let shape = PeakEstimateShape {
            // AO tensor not modelled by this case.
            nao: 0,
            need_inv_dielectric: false,
            naux: 50,
            nocc: 5,
            nvir: 20,
            n_quad: 10,
            n_workers: 2,
            n_keep: 999,
            grid: None,
        };
        // Must not panic; must equal the naux-clamped estimate.
        let est = estimate_peak_bytes(shape);
        let clamped = estimate_peak_bytes(PeakEstimateShape {
            n_keep: 50,
            ..shape
        });
        assert_eq!(est, clamped);
    }

    /// The open-shell scan charge holds two spin sets: on a singlet shape
    /// pair (β carrying no AO tensor) every per-spin term doubles against the
    /// closed-shell charge, the AO tier and the spin scratch do not, and a
    /// larger β spin raises the total. Charging only the α set (the closed-
    /// shell `scan_charge` on `shape_a`) fails the first and last assertions.
    #[test]
    fn the_open_shell_scan_charge_holds_both_spins() {
        let a = run_charge_shape();
        let b = PeakEstimateShape { nao: 0, ..a };
        let budget = 64usize << 30;
        let nov = a.nocc * a.nvir;
        let cs = scan_charge(a, budget, nov, 1 << 20);
        let u = scan_charge_unrestricted((a, b), budget, (nov, nov), 1 << 20);
        assert_eq!(u.persistent, 2 * cs.persistent);
        assert_eq!(u.drpa_hard, 2 * cs.drpa_hard);
        assert_eq!(u.drpa_soft, 2 * cs.drpa_soft);
        assert_eq!(u.spin_scratch, cs.spin_scratch);
        assert!(u.total > cs.total);
        let bigger = PeakEstimateShape {
            nocc: a.nocc + 5,
            ..b
        };
        let u2 = scan_charge_unrestricted((a, bigger), budget, (nov, nov), 1 << 20);
        assert!(u2.total > u.total && u2.persistent > u.persistent);
    }
}
