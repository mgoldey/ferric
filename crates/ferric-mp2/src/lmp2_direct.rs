//! Integral-direct assembly for the amplitude-threshold local MP2 family —
//! the missing piece named by the wall-clock measurement in
//! `wiki/amplitude-threshold-lmp2.md` §20–21: the global 3-index tensor
//! (`eri3_mo_ov_blocked`, `(naux, no·nv)`), its N⁴ whitening `B̃ = V^{-1/2}B`,
//! and the N⁵ all-pairs Gram product `J = B̃ᵀB̃` behind
//! `assemble_ragged_direct_aux` are never formed here.
//!
//! Pipeline (paper's Alg-2 shape, per-atom batched):
//! 1. Integral-free R⁻⁶ pair gate (shared `pair_gate_keep`) → surviving
//!    pairs and per-occupied partner lists.
//! 2. Distance/magnitude-linked maps: i → aux-shell fit domain `D_i`
//!    (`aux_radius_bohr`, the SAME rule as `domain_fit_pair`), i → virtual
//!    domain `V_i` (`virt_radius_bohr` on dipole centroids), and AO-shell
//!    supports for occupied and virtual columns (`ao_tail` on per-shell
//!    max |C| — truncation ≡ zeroing small coefficients).
//! 3. Per-atom batches (occupieds grouped by nearest atom to their Boys
//!    centroid): evaluate only the (μν|P) shell triples the batch needs,
//!    half-transform into per-occupied sparse strips
//!    `B_i[P ∈ D̃_i, a ∈ Ṽ_i]` (UNWHITENED, same values as the global B),
//!    where D̃_i/Ṽ_i are unions over i's surviving partners — every pair
//!    (i,j) then reads its `D_ij = D_i ∪ D_j` rows and `C_ij = V_i ∪ V_j`
//!    columns from the two strips.
//! 4. Domain-restricted 2-center metric: only (P|Q) shell pairs inside some
//!    D̃_i² are evaluated — exact for the domain-local fit, which never
//!    consults a metric element outside a pair domain.
//! 5. Per-pair domain-local same-kernel fit `J_ij = A_i^T V_DD⁻¹ A_j`
//!    (identical formulation to `domain_fit_pair`) → Eq-8 mask →
//!    `PairBlock`s. Solve and energy are the unchanged ragged machinery.
//!    Evaluated without an inverse: `L = chol(V_DD)` once per pair-domain
//!    group, `W_k = L⁻¹ A_k` once per distinct occupied in the group, and
//!    `J_ij = W_iᵀ W_j` per pair.
//!
//! # Exactness anchor (written BEFORE any sweep — Experimental Protocol)
//!
//! Trivial maps (`virt_radius_bohr = None`, `ao_tail = 0`, huge
//! `aux_radius_bohr`, no gate, ε = 0) make every map a no-op; the energy
//! must reproduce the existing global-B domain-fit path (same radius) and
//! the canonical `ri_mp2` to CG tolerance. Enforced in
//! `tests/lmp2_direct.rs`, with one mutation arm per map.
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! If the locality is real: at FIXED radii/thresholds the extra error vs
//! the eps-only path stays flat as the alkane grows, and strip sizes
//! saturate past the onset. If a map is mis-constructed (the
//! `docs/ao-laplace-locality-saturation.md` §5 failure class — mixed index
//! sets), exactness will instead demand radii that track the molecular
//! DIAMETER. These predictions differ, so the sweep can distinguish them.

use ferric_scf::PreparedAux;
use ndarray::{s, Array2, Array3, ArrayView2, Axis, ShapeBuilder};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;

use crate::lmp2_amplitude::{
    checked_vvhv_with, localized_spaces_lambda, pair_gate_keep, solve_and_count,
    AmplitudeLmp2Config, AmplitudeLmp2Result, LocalizedSpaces, StageTimings, VvHv,
};
pub use crate::lmp2_pair_skip::PairSkip;
use crate::ragged::{pair_blocks_screened, PairBlock, Ragged};
use crate::rimp2::{ri_mp2, RiMp2Config};

/// κ of the Fock-propagated screen on the integral-direct path. Kept as the
/// name the direct-path code and tests use; it is
/// [`crate::lmp2_amplitude::FOCK_SCREEN_KAPPA_DEFAULT`], which the CLI and
/// Python switch on by default on the in-core path too.
pub const DIRECT_FOCK_SCREEN_KAPPA: f64 = crate::lmp2_amplitude::FOCK_SCREEN_KAPPA_DEFAULT;

/// Production default of [`DirectConfig::aux_radius_bohr`] (Bohr) on the
/// user-facing integral-direct entry points. The CLI `[local]` parser and
/// Python `run_rimp2(..., integral_direct=True)` both read the defaults from
/// [`DirectConfig::production`], the one place they are written down.
pub const DIRECT_DEFAULT_AUX_RADIUS_BOHR: f64 = 10.0;
/// Production default of [`DirectConfig::virt_radius_bohr`] (Bohr). Raised
/// from 12 with `ao_tail` 1e-3 → 1e-4 after the ε = 0 map ablation on folded
/// C20H42 conformers (book chapter 13, options table).
pub const DIRECT_DEFAULT_VIRT_RADIUS_BOHR: f64 = 20.0;
/// Production default of [`DirectConfig::ao_tail`].
pub const DIRECT_DEFAULT_AO_TAIL: f64 = 1e-4;
/// Production default of [`DirectConfig::schwarz_skip`].
pub const DIRECT_DEFAULT_SCHWARZ_SKIP: f64 = 1e-5;
/// Default of [`DirectConfig::batch_merge`] when the user picks the
/// [`Eri3Pass::Batches`] pass (the production pass is [`Eri3Pass::AuxMajor`],
/// which has no batches and runs at `batch_merge = 1`).
pub const DIRECT_DEFAULT_BATCH_MERGE: usize = 4;
/// Default of [`DirectConfig::scratch_budget_bytes`]: 2 GiB.
pub const DIRECT_DEFAULT_SCRATCH_BUDGET_BYTES: usize = 2usize << 30;
/// Default of [`DirectConfig::slab_budget_bytes`]: 64 MiB per node. Measured
/// (release, 1 rank x 6 threads unbound, 6-31G / cc-pVDZ-RI, eps 1e-4,
/// production maps, two runs at the deciding caps): the alkane_20 stage-3
/// wall did not depend on the cap (4.9-5.9 s from 1 MiB to 2 GiB, 4.9/5.3 s
/// at 64 MiB, 5.6/5.9 s at 2 GiB), while the stage-3 peak resident set fell
/// from 2.3 GB at 2 GiB to 0.35 GB at 64 MiB.
pub const DIRECT_DEFAULT_SLAB_BUDGET_BYTES: usize = 64usize << 20;

/// Locality knobs of the integral-direct path. Every knob has a trivial
/// limit in which it is a no-op (the exactness-anchor configuration).
#[derive(Debug, Clone)]
pub struct DirectConfig {
    /// Aux fit-domain radius (Bohr): pair (i,j) fits in aux functions
    /// within this radius of EITHER Boys centroid — the same rule as
    /// `AmplitudeLmp2Config::fit_radius_bohr`. Huge (≥1e5) = global fit.
    pub aux_radius_bohr: f64,
    /// Virtual domain radius (Bohr) on dipole centroids: pair (i,j)'s
    /// candidate virtuals are those within this radius of either occupied
    /// centroid. `None` = all virtuals (trivial limit).
    pub virt_radius_bohr: Option<f64>,
    /// AO-support threshold: an obs shell enters an orbital's support iff
    /// its max |C| entry is ≥ this. 0.0 keeps every shell (trivial limit).
    /// Truncation is exactly "zero the small coefficients", applied
    /// symmetrically to both strips of a pair.
    pub ao_tail: f64,
    /// Shell-pair Schwarz skip: batch triples whose Cauchy–Schwarz bound
    /// √max(P|P) · Q(μν) (with Q(μν) = √(μν|μν), both under the same
    /// operator) falls below this are never evaluated (their T entries
    /// stay zero) — a rigorous upper-bound cut on the raw 3-index entries
    /// feeding the strips, anchored by the conservativeness test. 0.0
    /// evaluates everything (trivial limit).
    pub schwarz_skip: f64,
    /// Nearest-atom batches merged per integral pass. 1 = one batch per
    /// atom (the anchor limit). Per-atom batches re-evaluate shared shell
    /// triples ~5-6× (measured on C20-C48 maps,
    /// `scripts/queue/proto_batch_amortization.py`); merging 4 batches
    /// along the molecule's long axis cuts evaluations to ~0.40× at the
    /// cost of larger scratch slabs. Energies agree with k=1 to the
    /// float-reassociation floor (anchored ≤1e-10), not bit-identically —
    /// the per-occupied transform runs over the larger batch union.
    pub batch_merge: usize,
    /// GLOBAL scratch cap (bytes) of the stage-3 aux-major pass (it sizes the
    /// tasks in flight), the κ screen's own-domain `W` store (kept only when
    /// it fits), and the per-group scratch of the fitted-pair assembly. Every
    /// user splits it over its concurrent workers, so the co-resident scratch
    /// never exceeds it (the ferric-batch N×-overcommit lesson: a per-worker
    /// budget is a budget on nothing). It also bounds
    /// [`Self::slab_budget_bytes`] from above.
    pub scratch_budget_bytes: usize,
    /// Integral-slab scratch cap (bytes) of the stage-3 batch pass
    /// ([`Eri3Pass::Batches`]), per node and shared across the batch workers:
    /// each worker slabs under `cap / n_workers`. Under a rank stripe
    /// ([`Self::mpi`]) each rank takes `1 / (ranks on this node)` of it, so
    /// the node's slabs do not multiply by the rank count (see
    /// `batch_slab_cap`). It is further capped by
    /// [`Self::scratch_budget_bytes`] and by what the memory budget leaves
    /// beside the strips. It changes how many slabs a batch is cut into, and
    /// the strips only at round-off (the virtual GEMM's row count follows the
    /// slab). Default
    /// [`DIRECT_DEFAULT_SLAB_BUDGET_BYTES`].
    pub slab_budget_bytes: usize,
    /// ε-linked Schwarz virtual-candidate screen (prototype-validated GO,
    /// WIKI-APPEND-eps-linked-maps.md): with `Some(kappa)` and ε > 0, the
    /// pair candidate set C_ij = V_i ∪ V_j keeps virtual `a` iff
    /// scale·q_ia·qmax_j ≥ kappa·ε or scale·q_ja·qmax_i ≥ kappa·ε, with
    /// q_ia = √(b_ia^T V_{D_i D_i}⁻¹ b_ia) the strip-local fitted
    /// diagonal (Cauchy–Schwarz-exact on a global Gram fit; measured
    /// escape-free against the domain-local fit at kappa ≤ 1, both
    /// operators, C8). `None` — or ε = 0 — is the trivial limit (no
    /// refinement). kappa = 1 is the conservative candidate; larger kappa
    /// trades bounded sub-dominant error for smaller pair blocks
    /// (measured C8/erfc/1e-3: kappa=3 cut |C_ij| 40% at +2.9e-6 Ha,
    /// 1600× below the ε truncation).
    pub virt_schwarz_kappa: Option<f64>,
    /// Stats flag: split the integral-fill timer into libint evaluation and
    /// slab scatter (`t_eri3_eval_thread_s` / `t_eri3_scatter_thread_s`).
    /// Splitting buffers each chunk's blocks (an extra copy), so it is OFF
    /// by default; off, only their sum `t_eri3_fill_thread_s` is timed and
    /// the two split timers are NaN. Never changes a result.
    pub split_eri3_timers: bool,
    /// How stage 3 walks the (P|μν) triples. See [`Eri3Pass`].
    pub eri3_pass: Eri3Pass,
    /// MPI context to stripe the assembly over. `None`, or a one-rank
    /// context, is the serial path exactly. With `size > 1` the domain
    /// groups are ordered along the molecule and cut into one contiguous run
    /// per rank; a rank builds strips only for the occupieds in its pairs,
    /// each covering only the partners it has there, and the metric blocks
    /// those need. The ranks then exchange their pair blocks (the
    /// `ragged_transport` module), so every rank returns the full ragged
    /// system and runs the same solve.
    ///
    /// What is distributed: the strip build, the metric and the pair fits,
    /// in time and in memory. What is not: every rank ends up holding all
    /// pair blocks and the solver's vectors, so the solve's footprint does
    /// not shrink with the rank count.
    ///
    /// A strip built over fewer partners goes through differently shaped
    /// GEMMs, so a striped run agrees with the serial one to rounding, not
    /// bit for bit. The ranks must hold identical localized spaces: the
    /// owner of each group is computed on every rank from them. Stage-3 and
    /// stage-5 counters, timers and strip sizes describe this rank's share.
    pub mpi: Option<ferric_core::parallel::ParallelContext>,
    /// Stage-5 provable-empty pair skip ([`crate::lmp2_pair_skip`]): a pair
    /// whose fitted retention test is bounded below ε without its own fit is
    /// not fitted. Energies are unchanged (bitwise: a skipped pair is one
    /// whose blocks the mask would drop, and the groups' whitening is the
    /// same). [`PairSkip::Off`] is the anchor limit; production is
    /// [`PairSkip::Auto`] (on for erfc / terfc, off for Coulomb, where the
    /// bound never reaches a pair).
    pub pair_skip: PairSkip,
}

impl DirectConfig {
    /// The locality maps the user-facing integral-direct entry points run
    /// when no map key is given: aux radius 10 Bohr, virtual radius 20 Bohr,
    /// AO tail 1e-4, Schwarz skip 1e-5, the aux-major stage-3 pass
    /// ([`Eri3Pass::AuxMajor`], `batch_merge = 1`), no ε-linked virtual
    /// screen. Measured against the batches pass at merge 4 (release, 6
    /// cores, 6-31G / cc-pVDZ-RI, eps 1e-4, energies identical to 12
    /// digits): stage 3 1.35 -> 0.27 s (C12), 5.17 -> 1.19 s (C20),
    /// 15.6 -> 5.6 s (C32); shell triples 11.2M -> 1.2M, 41.8M -> 3.4M,
    /// 122M -> 8.9M; peak RSS unchanged. ([`Default`] is instead the exactness-anchor configuration,
    /// every map at or near its no-op limit.) The pair gate, the Fock screen
    /// (on, κ = [`DIRECT_FOCK_SCREEN_KAPPA`]) and λ-Boys live on
    /// [`AmplitudeLmp2Config`], not here.
    pub fn production() -> Self {
        Self {
            aux_radius_bohr: DIRECT_DEFAULT_AUX_RADIUS_BOHR,
            virt_radius_bohr: Some(DIRECT_DEFAULT_VIRT_RADIUS_BOHR),
            ao_tail: DIRECT_DEFAULT_AO_TAIL,
            schwarz_skip: DIRECT_DEFAULT_SCHWARZ_SKIP,
            batch_merge: 1,
            eri3_pass: Eri3Pass::AuxMajor,
            virt_schwarz_kappa: None,
            pair_skip: PairSkip::Auto,
            ..Self::default()
        }
    }

    /// The ONE place a user-facing (pass, batch_merge) pair is resolved, for
    /// the CLI and Python alike. `pass`: `None` = the production pass
    /// (aux-major), else [`Eri3Pass::parse_config_str`]. `batch_merge`:
    /// `None` = the pass's default (1 for aux-major, [`DIRECT_DEFAULT_BATCH_MERGE`]
    /// for batches). An explicit `batch_merge > 1` with aux-major is an
    /// error, as is `batch_merge = 0`.
    pub fn resolve_pass(
        pass: Option<&str>,
        batch_merge: Option<usize>,
    ) -> Result<(Eri3Pass, usize), String> {
        let pass = match pass {
            None => Self::production().eri3_pass,
            Some(s) => Eri3Pass::parse_config_str(s)?,
        };
        if batch_merge == Some(0) {
            return Err("batch_merge must be >= 1".to_string());
        }
        match (pass, batch_merge) {
            (Eri3Pass::AuxMajor, Some(m)) if m > 1 => Err(format!(
                "batch_merge = {m} needs the batches pass: the aux-major pass \
                 (the default) has no batches; set eri3_pass = \"batches\" or drop batch_merge"
            )),
            (Eri3Pass::AuxMajor, _) => Ok((pass, 1)),
            (Eri3Pass::Batches, m) => Ok((pass, m.unwrap_or(DIRECT_DEFAULT_BATCH_MERGE))),
        }
    }
}

/// Stage-3 integral pass. Both build the same strips to round-off: every strip
/// element sums the same integrals with the same coefficients and the same
/// Schwarz cut. They differ in how often a triple is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Eri3Pass {
    /// Occupied batches, each with its own union slab
    /// ([`DirectConfig::batch_merge`]). A triple needed by several batches is
    /// evaluated once per batch.
    #[default]
    Batches,
    /// One task per aux shell, over the union of the supports of the occupieds
    /// that need that shell: each needed triple is evaluated once. Rejects
    /// `batch_merge > 1`, which has no meaning here. The production pass.
    AuxMajor,
}

impl Eri3Pass {
    /// Strict parser for the user-facing spelling: `"aux-major"` or
    /// `"batches"`; anything else (case, underscores, empty) is an error.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s {
            "aux-major" => Ok(Self::AuxMajor),
            "batches" => Ok(Self::Batches),
            other => Err(format!(
                "unknown eri3_pass {other:?}: expected \"aux-major\" or \"batches\""
            )),
        }
    }

    /// The user-facing spelling ([`Self::parse_config_str`] round-trips it).
    pub fn as_config_str(self) -> &'static str {
        match self {
            Self::AuxMajor => "aux-major",
            Self::Batches => "batches",
        }
    }
}

impl Default for DirectConfig {
    fn default() -> Self {
        Self {
            aux_radius_bohr: 10.0,
            virt_radius_bohr: None,
            ao_tail: 0.0,
            schwarz_skip: 0.0,
            batch_merge: 1,
            scratch_budget_bytes: DIRECT_DEFAULT_SCRATCH_BUDGET_BYTES,
            slab_budget_bytes: DIRECT_DEFAULT_SLAB_BUDGET_BYTES,
            virt_schwarz_kappa: None,
            split_eri3_timers: false,
            eri3_pass: Eri3Pass::Batches,
            mpi: None,
            pair_skip: PairSkip::Off,
        }
    }
}

/// Counters + stage timings of one direct assembly — the honest-measurement
/// surface: strip saturation is the locality claim, so it is reported, not
/// inferred.
#[derive(Debug, Clone, Default)]
pub struct DirectStats {
    /// Mean per-occupied strip aux rows (functions).
    pub strip_rows_mean: f64,
    /// Maximum per-occupied strip aux rows (functions).
    pub strip_rows_max: usize,
    /// Mean per-occupied strip virtual columns.
    pub strip_cols_mean: f64,
    /// Maximum per-occupied strip virtual columns.
    pub strip_cols_max: usize,
    /// Total bytes held by all strips (the peak 3-index working set).
    pub strip_bytes_total: usize,
    /// (P|μν) shell triples actually evaluated (recompute across batches
    /// included — the honest integral-work counter).
    pub n_eri3_shell_triples: u64,
    /// Batch triples skipped by the shell-pair Schwarz cut (0 when
    /// `schwarz_skip` is 0.0).
    pub n_eri3_skipped: u64,
    /// (P|Q) metric shell pairs evaluated (of nshdf·(nshdf+1)/2 total).
    pub n_metric_shell_pairs: usize,
    /// Mean pair virtual-candidate columns |C_ij| after the Schwarz screen,
    /// over unique surviving pairs (the V_i ∪ V_j union sizes when
    /// `virt_schwarz_kappa` is off).
    pub virt_cand_mean: f64,
    /// Maximum of the same |C_ij| over unique surviving pairs.
    pub virt_cand_max: usize,
    /// Stages 1+2 wall seconds: pair gate, per-occupied distance/support maps
    /// and integral batches (integral-free).
    pub t_maps_s: f64,
    /// Stage-3 wall seconds (all batch workers).
    pub t_eri3_s: f64,
    /// Stage-3 split, wall seconds measured on each batch worker's thread
    /// and summed over workers (thread-seconds; NOT CPU time — a descheduled
    /// worker still accrues): the integral fill (libint evaluation +
    /// scatter into the slab) and the half-transform into the strips. Their
    /// sum is below the workers' total by the per-batch setup. This field is
    /// the fill.
    pub t_eri3_fill_thread_s: f64,
    /// The fill split into evaluation and scatter: only with
    /// [`DirectConfig::split_eri3_timers`] (NaN otherwise). The clock is read
    /// per 64-shell-pair chunk, not per triple.
    pub t_eri3_eval_thread_s: f64,
    /// The scatter half of the fill split (NaN unless
    /// [`DirectConfig::split_eri3_timers`]).
    pub t_eri3_scatter_thread_s: f64,
    /// Half-transform thread-seconds, summed over workers (see
    /// `t_eri3_fill_thread_s`).
    pub t_eri3_transform_thread_s: f64,
    /// The serial prefix of `t_eri3_s`, wall seconds: the construction check
    /// and the shell-pair / aux Schwarz factors, before any batch worker runs.
    pub t_eri3_setup_s: f64,
    /// Integral slabs filled, over all batches.
    pub n_eri3_slabs: usize,
    /// Largest single stage-3 slab with every buffer its transform holds
    /// beside it (the occupied half-transform, one occupied's selected rows
    /// and their product, the index maps), measured on the buffers, bytes
    /// (batch pass; 0 on the aux-major pass). Up to one such slab per batch
    /// worker is resident at once.
    pub eri3_slab_bytes_max: usize,
    /// Half-transform floating-point operations actually done
    /// (`Σ_i Σ_{P∈slab} 2·nsf·nnf + 2·nnf·|V_i|`, batch-union axes), and what
    /// the same contraction would cost restricted to i's OWN occupied and
    /// virtual AO support (a counter for the sparsity the transform does not
    /// exploit; batch pass only). On the aux-major pass `transform_flops` counts
    /// its own contraction and the second counter stays 0.
    pub transform_flops: f64,
    /// Half-transform flops restricted to each occupied's own AO support (see
    /// `transform_flops`; 0 on the aux-major pass).
    pub transform_flops_own_support: f64,
    /// Integrals libint wrote, and shell triples it screened to nothing. Only
    /// with [`DirectConfig::split_eri3_timers`] (0 otherwise).
    pub n_eri3_integrals: u64,
    /// Shell triples libint screened to nothing (0 unless
    /// [`DirectConfig::split_eri3_timers`]).
    pub n_eri3_null_blocks: u64,
    /// [`Eri3Pass::AuxMajor`]: largest one-task scratch (slab + half-transformed
    /// slab + occupied coefficients), and how many tasks the scratch budget lets
    /// run at once. 0 on the batch pass.
    pub aux_major_task_bytes_max: usize,
    /// [`Eri3Pass::AuxMajor`]: tasks the scratch budget lets run at once
    /// (0 on the batch pass).
    pub aux_major_concurrency: usize,
    /// [`Eri3Pass::AuxMajor`]: the most tasks actually observed in flight at once
    /// (0 on the batch pass). Never above `aux_major_concurrency`.
    pub aux_major_in_flight_peak: usize,
    /// [`Eri3Pass::AuxMajor`]: bytes of the per-occupied virtual coefficient
    /// blocks, resident for all of stage 3 (0 on the batch pass).
    pub aux_major_coeff_bytes: usize,
    /// Integral batches run (aux-shell tasks on the aux-major pass), and the max/mean wall seconds of one batch
    /// worker (the load-balance counter: max ≫ mean is a tail imbalance).
    pub n_batches: usize,
    /// Longest wall seconds of one batch worker.
    pub batch_work_max_s: f64,
    /// Mean wall seconds of one batch worker.
    pub batch_work_mean_s: f64,
    /// Wall seconds evaluating the needed (P|Q) metric shell pairs.
    pub t_metric_s: f64,
    /// Stage-5 wall seconds: the optional Schwarz virtual-candidate screen,
    /// per-pair-domain fits and ragged assembly.
    pub t_pairs_s: f64,
    /// Stage-5 pair-domain groups (one V_DD Cholesky each).
    pub n_pair_groups: usize,
    /// Stage-5 thread-seconds summed over this rank's domain groups, by part:
    /// `[gather V_DD, Cholesky, whiten (gather + one triangular solve), pair
    /// products J_ij = W_iᵀW_j + the Eq-8 mask and Fock screen + PairBlock]`.
    /// Their sum over the worker count is the group phase's wall time when
    /// the groups balance.
    pub pair_fit_thread_s: [f64; 4],
    /// Longest wall seconds of one stage-5 domain group (the load-balance
    /// counter: compare with `group_work_mean_s`).
    pub group_work_max_s: f64,
    /// Mean wall seconds of one stage-5 domain group.
    pub group_work_mean_s: f64,
    /// Mean distinct occupieds over a group's pairs (each is whitened once
    /// per group, `W_k = L⁻¹ A_k`, instead of once per pair).
    pub n_distinct_occ_per_group: f64,
    /// Stage-5 whitenings by triangular solve, counted per occupied right-
    /// hand side (not per LAPACK call: a group batches its occupieds into
    /// one solve): one per occupied for the κ screen when it is on, plus
    /// one per (group, distinct occupied) not reused from it, so that
    /// `n_trsm + n_w_reused` is every (group, occupied) whitening.
    pub n_trsm: usize,
    /// (group, occupied) whitenings taken from the κ screen's own-domain
    /// `W` (pair (i,i)'s group has domain exactly D_i) instead of a solve.
    pub n_w_reused: usize,
    /// Bytes of the κ screen's own-domain whitened strips at full width,
    /// `8·Σ_i |D_i|·|cols_i|` — the store's PEAK, held at the end of the
    /// screen. It is pre-flighted against [`DirectConfig::scratch_budget_bytes`]:
    /// over budget, no `W_i` is kept (q is still computed; reuse is off and
    /// every group solves). 0 with the screen off.
    pub own_w_bytes: usize,
    /// Bytes of that store actually held through the group loop: each kept
    /// `W_i` is trimmed right after the screen to the columns its reuse can
    /// read (the candidate union of i's pairs in the group whose domain is
    /// exactly D_i), and dropped if no such group exists. 0 when not kept.
    pub own_w_bytes_held: usize,
    /// Largest single-group scratch, bytes: `8·(d² + Σ_k d·|C_k|)` (the
    /// Cholesky factor, computed in V_DD's buffer, and one whitened `W_k` per
    /// distinct occupied k over its candidate union C_k). Each rayon worker holds one group's
    /// scratch at a time, so the co-resident group scratch is at most
    /// (worker threads) × this; it is NOT under `scratch_budget_bytes`.
    pub group_scratch_bytes_max: usize,
    /// Boys localization + localized-space setup (`localized_spaces`),
    /// wall seconds; set by the driver (0 when the assembly is called
    /// directly).
    pub t_loc_s: f64,
    /// The integral-free memory pre-flight of this assembly (see
    /// [`DirectMemoryEstimate`]).
    pub memory: DirectMemoryEstimate,
    /// Stage-5 domain groups allowed in flight at once: the worker count when
    /// ungated (whatever the number of groups), else the widest cap
    /// `min(len, workers)` of the budget-sized waves (see `plan_group_wave`),
    /// so a gated run with one group reports 1. With more than one group, 1
    /// means the budget serialised stage 5. 0 before stage 5.
    pub group_concurrency: usize,
    /// The largest per-group scratch bound the waves are sized with
    /// (`group_scratch_bound`; ≥ the measured `group_scratch_bytes_max`).
    pub group_scratch_bound_bytes: usize,
    /// Stage-5 groups measured in flight at once (peak); 0 before stage 5.
    pub group_in_flight_peak: usize,
    /// Stage-5 waves: 1 when ungated or when every group fits one wave (a
    /// single group included), more when the budget split the groups; 0
    /// before stage 5.
    pub n_group_waves: usize,
    /// The pair-skip mode this run used ([`PairSkip::resolve`]d for the
    /// operator: `Off`, `On` or `Audit`).
    pub pair_skip: PairSkip,
    /// Off-diagonal pairs whose bound is finite (`γ < 1`); 0 when off.
    pub n_pair_skip_bounded: usize,
    /// Pairs the bound proves empty: skipped under `On`, still fitted under
    /// `Audit`.
    pub n_pairs_proved_empty: usize,
    /// Domain groups not fitted because every pair in them was skipped.
    pub n_pair_groups_skipped: usize,
    /// Fitted pairs whose Eq-8 mask (and Fock screen) kept nothing.
    pub n_pairs_empty: usize,
    /// Wall seconds of the skip proof: the own-domain eigenvalue floors, the
    /// strip and metric norm tables and the per-pair bounds (0 when off).
    pub t_pair_skip_s: f64,
    /// `Audit`: bounded pairs compared with their fit.
    pub pair_skip_audit_n: usize,
    /// `Audit`: largest `max|g| / bound` over bounded fitted pairs (must be
    /// ≤ 1; 0 unless audit).
    pub pair_skip_audit_max_ratio: f64,
    /// `Audit`: largest `max(|g|, |g̃|/κ) / gate bound` (must be ≤ 1; 0
    /// unless audit).
    pub pair_skip_audit_gate_max_ratio: f64,
    /// `Audit`: pairs proved empty whose fit kept a block (must be 0).
    pub pair_skip_audit_violations: usize,
}

/// Byte accounting of the direct LMP2 path, computed from the locality maps
/// BEFORE any integral is evaluated.
///
/// With `n_ij = |V_i ∪ V_j|` the candidate width of surviving pair `(i,j)`
/// (each `i ≠ j` pair stores two blocks, `(i,j)` and `(j,i)`), `nv` the
/// virtual count and `S` the strips:
///
/// ```text
/// R  = Σ_blocks [ 8·(4 n²) + n² + 8·(2n + 2nv) + sizeof(PairBlock) ]   pair blocks
/// V  = Σ_blocks [ 8·n² + sizeof(Array2) ]                              one PCG vector
/// S  = Σ_i [ 8·rows_i·cols_i + 8·(rows_i + naux + cols_i + nv) ]       strips
/// assembly peak = max(S + scratch, S + metric + R)
/// solve peak    = R + PCG_RESIDENT_VECTORS · V                           (gather plan: only
///                                                                        what the budget has left)
/// ```
///
/// At ε = 0 the Eq-8 mask keeps every candidate, so `n_ij` is each block's
/// exact width and the estimate is exact (`exact = true`). At ε > 0 the mask
/// drops rows and columns, so it is an upper bound; the assembly then charges
/// each block's exact bytes as it is built instead of refusing on the bound.
#[derive(Debug, Clone, Copy, Default)]
pub struct DirectMemoryEstimate {
    /// Per-occupied unwhitened strips, `S`.
    pub strips_bytes: usize,
    /// Domain-restricted 2-center metric blocks.
    pub metric_bytes: usize,
    /// Stage-3 scratch cap actually used by this process: on the batch pass
    /// the slab cap ([`DirectConfig::slab_budget_bytes`] over the ranks on
    /// the node), on the aux-major pass `scratch_budget_bytes`; either one
    /// lowered to what the budget leaves beside `S`.
    pub stage3_scratch_bytes: usize,
    /// The ragged pair system, `R`.
    pub pair_block_bytes: usize,
    /// ONE amplitude-shaped PCG vector, `V`.
    pub pcg_vector_bytes: usize,
    /// Whether `R` and `V` are exact (ε = 0) rather than candidate bounds.
    pub exact: bool,
}

impl DirectMemoryEstimate {
    /// Stage 3 (strips + slab scratch) or stage 5 (strips + metric + pair
    /// blocks), whichever is larger.
    pub fn assembly_peak(&self) -> usize {
        (self.strips_bytes + self.stage3_scratch_bytes)
            .max(self.strips_bytes + self.metric_bytes + self.pair_block_bytes)
    }

    /// The ragged system plus the PCG iteration vectors.
    pub fn solve_peak(&self) -> usize {
        self.pair_block_bytes + crate::ragged::PCG_RESIDENT_VECTORS * self.pcg_vector_bytes
    }

    /// The larger of the assembly and solve peaks.
    pub fn peak(&self) -> usize {
        self.assembly_peak().max(self.solve_peak())
    }

    /// The peak without the stage-3 slab scratch (which the budget blocks):
    /// strips + metric + pair blocks during the fits, or the solve. This is
    /// what the gate compares with the budget.
    pub fn resident_peak(&self) -> usize {
        (self.strips_bytes + self.metric_bytes + self.pair_block_bytes).max(self.solve_peak())
    }
}

/// Bytes one [`PairBlock`] of shape `na × nb` over `nv` virtuals holds: its
/// [`PairBlock::heap_bytes`] at exact capacities plus its slot in
/// `Ragged::pairs`.
fn block_bytes(na: usize, nb: usize, nv: usize) -> usize {
    let (us, f) = (std::mem::size_of::<usize>(), std::mem::size_of::<f64>());
    std::mem::size_of::<PairBlock>()
        + us * (na + nb + 2 * nv)
        + na * nb
        + f * (na * na + nb * nb + 2 * na * nb)
}

/// Bytes of one amplitude-shaped block (as [`crate::ragged::amplitude_vector_bytes`]).
fn amp_block_bytes(na: usize, nb: usize) -> usize {
    std::mem::size_of::<Array2<f64>>() + std::mem::size_of::<f64>() * na * nb
}

/// The memory budget the assembly is gated against: the LMP2 solve that
/// consumes it must fit, so the pre-flight refuses an exact over-budget
/// estimate up front and stage 5 charges every built block.
#[derive(Debug, Clone, Copy)]
struct SolveGate {
    budget_bytes: usize,
}

/// The stage-3 slab scratch under a budget: the configured cap, lowered to
/// what the budget leaves beside the strips (at least one byte, which slabs
/// one aux shell at a time).
fn stage3_scratch(configured: usize, budget: usize, strips: usize) -> usize {
    configured.min(budget.saturating_sub(strips).max(1))
}

/// Ranks of a `world`-rank stripe that share this node, given the launcher's
/// node-local rank count (`None`: no recognised launcher variable). An
/// unrecognised launcher counts every rank as local: that over-divides the
/// slab budget on a multi-node run (more, smaller slabs) rather than
/// over-committing the node.
fn ranks_sharing_node(world: usize, launcher_local: Option<usize>) -> usize {
    if world <= 1 {
        return 1;
    }
    launcher_local.map_or(world, |l| l.clamp(1, world))
}

/// The launcher's node-local rank count, `None` when no launcher variable
/// parses as a positive count (unset, or Slurm's `"4,2"`): the world-size
/// fallback of [`ranks_sharing_node`] then applies, never a guessed 1.
fn launcher_local_ranks() -> Option<usize> {
    ferric_core::parallel::local_ranks_hint_with(|k| std::env::var(k).ok())
}

/// This process's batch-pass slab cap: [`DirectConfig::slab_budget_bytes`]
/// over the ranks sharing the node (1 without a multi-rank stripe), never
/// above `scratch` (the stage-3 scratch after the memory gate), at least one
/// byte (one aux shell per slab).
fn batch_slab_cap(dcfg: &DirectConfig, scratch: usize) -> usize {
    let ranks = dcfg
        .mpi
        .map_or(1, |c| ranks_sharing_node(c.size, launcher_local_ranks()));
    (dcfg.slab_budget_bytes / ranks).min(scratch).max(1)
}

/// The typed refusal: names the estimate, the budget and the knobs.
fn memory_refusal(est: &DirectMemoryEstimate, budget: usize, lead: &str) -> FerricError {
    let gb = |b: usize| b as f64 / 1e9;
    FerricError::MemoryExhausted(format!(
        "lmp2_direct: {lead}: resident peak {:.2} GB (pair blocks {:.2} GB + {} PCG amplitude vectors \
         {:.2} GB; strips {:.2} GB, metric {:.2} GB) is over the memory budget of {:.2} GB. \
         The pair blocks and PCG vectors stay resident for the whole solve and are not \
         blocked. Raise [memory] budget_gb / FERRIC_MEM_BUDGET_GB, or shrink the pair blocks: \
         a larger [local] eps, a smaller virt_radius, or the pair gate (gate_cal)",
        gb(est.resident_peak()),
        gb(est.pair_block_bytes),
        crate::ragged::PCG_RESIDENT_VECTORS,
        gb(crate::ragged::PCG_RESIDENT_VECTORS * est.pcg_vector_bytes),
        gb(est.strips_bytes),
        gb(est.metric_bytes),
        gb(budget),
    ))
}

/// The integral-free pre-flight (see [`DirectMemoryEstimate`]); the stage-3
/// scratch is left at 0 for the caller to fill.
fn preflight_memory(
    maps: &LocalityMaps,
    need: &[(usize, usize)],
    dfbs: &PreparedBasis,
    no: usize,
    nv: usize,
    eps: f64,
) -> DirectMemoryEstimate {
    let us = std::mem::size_of::<usize>();
    let dims_df = dfbs.shell_dims();
    let naux = dfbs.nbasis();
    let strips_bytes = (0..no)
        .filter(|&i| maps.stripe.as_ref().is_none_or(|s| s.needed[i]))
        .map(|i| {
            let rows: usize = maps.aux_ext[i].iter().map(|&sh| dims_df[sh]).sum();
            let cols = maps.virt_ext[i].len();
            std::mem::size_of::<Strip>() + 8 * rows * cols + us * (rows + naux + cols + nv)
        })
        .sum();
    let metric_bytes = need
        .iter()
        .map(|&(sp, sq)| 8 * dims_df[sp] * dims_df[sq])
        .sum::<usize>()
        + 4 * dfbs.nshells() * dfbs.nshells();
    let (mut pair_block_bytes, mut pcg_vector_bytes) = (0usize, 0usize);
    for i in 0..no {
        for j in i..no {
            if !maps.keep[i * no + j] {
                continue;
            }
            let n = sorted_union([
                maps.virt_dom[i].iter().copied(),
                maps.virt_dom[j].iter().copied(),
            ])
            .len();
            if n == 0 {
                continue;
            }
            let blocks = if i == j { 1 } else { 2 };
            pair_block_bytes += blocks * block_bytes(n, n, nv);
            pcg_vector_bytes += blocks * amp_block_bytes(n, n);
        }
    }
    DirectMemoryEstimate {
        strips_bytes,
        metric_bytes,
        stage3_scratch_bytes: 0,
        pair_block_bytes,
        pcg_vector_bytes,
        exact: eps == 0.0,
    }
}

/// One batch worker's stage-3 timers (thread-seconds).
#[derive(Debug, Clone, Copy, Default)]
struct BatchTimers {
    fill_s: f64,
    eval_s: f64,
    scatter_s: f64,
    transform_s: f64,
    total_s: f64,
    n_slabs: usize,
    transform_flops: f64,
    transform_flops_own_support: f64,
    n_integrals: u64,
    n_null_blocks: u64,
    /// Largest live slab this worker held, measured on the buffers
    /// themselves ([`SlabDims::live_bytes`] bounds it).
    slab_bytes_max: usize,
}

/// One batch's output: its strips, evaluated/skipped triple counts and
/// timers.
type BatchOut = (Vec<(usize, Strip)>, u64, u64, BatchTimers);

/// One occupied's sparse unwhitened RI strip: `b[(r, c)] = (P_r | i a_c)`.
struct Strip {
    /// Global aux FUNCTION ids of the rows (sorted).
    rows: Vec<usize>,
    /// naux-length inverse map (usize::MAX = absent).
    row_pos: Vec<usize>,
    /// Global virtual ids of the columns (sorted).
    cols: Vec<usize>,
    /// nv-length inverse map (usize::MAX = absent).
    col_pos: Vec<usize>,
    b: Array2<f64>,
}

impl Strip {
    /// The placeholder for an occupied this rank does not build under a
    /// rank stripe. Its inverse maps are empty, so any read of it panics on
    /// the index instead of returning a zero integral.
    fn absent() -> Self {
        Self {
            rows: Vec::new(),
            row_pos: Vec::new(),
            cols: Vec::new(),
            col_pos: Vec::new(),
            b: Array2::zeros((0, 0)),
        }
    }
}

fn sorted_union(lists: impl IntoIterator<Item = impl IntoIterator<Item = usize>>) -> Vec<usize> {
    let set: BTreeSet<usize> = lists.into_iter().flatten().collect();
    set.into_iter().collect()
}

/// `FERRIC_MEM_TRACE` line at a stage boundary: current and peak RSS beside
/// the byte counts this module accounts for there. Informational only.
pub(crate) fn mem_trace(stage: &str, fields: &[(&str, usize)]) {
    if !ferric_core::memory::trace_enabled() {
        return;
    }
    let gb = |b: Option<usize>| b.map_or("n/a".to_string(), |b| format!("{:.3}", b as f64 / 1e9));
    let mut line = format!(
        "[ferric-mem] lmp2-direct {stage} rss={} GB peak={} GB",
        gb(ferric_core::memory::read_own_rss_bytes()),
        gb(ferric_core::memory::sys::own_peak_rss_bytes())
    );
    for (k, v) in fields {
        line.push_str(&format!(" {k}={:.3}", *v as f64 / 1e9));
    }
    eprintln!("{line}");
}

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Per-shell max |C| support selection for one coefficient column.
fn supp_shells(prep: &PreparedBasis, col: ndarray::ArrayView1<f64>, tail: f64) -> Vec<usize> {
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    (0..prep.nshells())
        .filter(|&sh| {
            let mut m = 0.0f64;
            for f in 0..dims[sh] {
                m = m.max(col[offs[sh] + f].abs());
            }
            m >= tail
        })
        .collect()
}

/// The evaluated `(P|Q)` metric shell-pair blocks, keyed `(sp ≥ sq)` through
/// a dense `nsh_df²` id table (was a `HashMap<(usize, usize), Array2>`).
struct MetricBlocks {
    nsh: usize,
    id: Vec<u32>,
    blocks: Vec<Array2<f64>>,
}

impl MetricBlocks {
    /// Heap bytes of the evaluated blocks and the shell-pair index.
    fn bytes(&self) -> usize {
        8 * self.blocks.iter().map(Array2::len).sum::<usize>() + 4 * self.id.len()
    }

    /// Block `(sp, sq)` with `sp ≥ sq`; `None` when it was never evaluated.
    fn get(&self, sp: usize, sq: usize) -> Option<&Array2<f64>> {
        let k = self.id[sp * self.nsh + sq];
        (k != u32::MAX).then(|| &self.blocks[k as usize])
    }
}

/// The metric shell pairs `(sp ≥ sq)` inside some `D̃_i²`, in lexicographic
/// `(sp, sq)` order (the order the former `BTreeSet` produced), marked in a
/// lower-triangle bitmap instead of a tree. Each `aux_ext` list is sorted.
fn metric_need_pairs(aux_ext: &[Vec<usize>], nsh: usize) -> Vec<(usize, usize)> {
    let mut mark = vec![false; nsh * nsh];
    for shells in aux_ext {
        for (x, &sp) in shells.iter().enumerate() {
            for &sq in &shells[..=x] {
                mark[sp * nsh + sq] = true;
            }
        }
    }
    let mut need = Vec::new();
    for sp in 0..nsh {
        for sq in 0..=sp {
            if mark[sp * nsh + sq] {
                need.push((sp, sq));
            }
        }
    }
    need
}

/// Evaluate the needed metric blocks (rayon, one 2-center engine per
/// worker) into a [`MetricBlocks`] store.
fn eval_metric_blocks(
    op: Operator,
    dfbs: &PreparedBasis,
    need: &[(usize, usize)],
) -> Result<MetricBlocks, FerricError> {
    use rayon::prelude::*;
    let dims_df = dfbs.shell_dims();
    let nsh = dfbs.nshells();
    Engine::new_2center(op, dfbs, 1e-14)?;
    let blocks: Vec<Array2<f64>> = need
        .par_iter()
        .map_init(
            || Engine::new_2center(op, dfbs, 1e-14).expect("2-center engine (pre-validated)"),
            |eng, &(sp, sq)| {
                let (np, nq) = (dims_df[sp], dims_df[sq]);
                let vals = eng.compute_eri2(dfbs, sp, sq);
                let mut blk = Array2::<f64>::zeros((np, nq));
                for p in 0..np {
                    for q in 0..nq {
                        blk[(p, q)] = vals[p * nq + q];
                    }
                }
                blk
            },
        )
        .collect();
    let mut id = vec![u32::MAX; nsh * nsh];
    for (k, &(sp, sq)) in need.iter().enumerate() {
        id[sp * nsh + sq] = u32::try_from(k).map_err(|_| {
            FerricError::General("lmp2_direct: more than 2^32 metric shell pairs".to_string())
        })?;
    }
    Ok(MetricBlocks { nsh, id, blocks })
}

/// Gather the dense V_DD metric block for a sorted aux-shell list from the
/// sparse per-shell-pair block map (upper-triangle keys, sp >= sq).
///
/// Column-major: V_DD is symmetric, so this is the same matrix, and LAPACK
/// then factors it in place ([`chol_lower`]) and solves against the factor
/// ([`whiten`]) with no transposed copy of either.
fn gather_vdd(
    dsh: &[usize],
    dims_df: &[usize],
    metric_blocks: &MetricBlocks,
    ctx: &str,
) -> Result<Array2<f64>, FerricError> {
    let d: usize = dsh.iter().map(|&sh| dims_df[sh]).sum();
    let mut loc_off = Vec::with_capacity(dsh.len());
    let mut acc = 0usize;
    for &sh in dsh {
        loc_off.push(acc);
        acc += dims_df[sh];
    }
    // Raw column-major buffer: the blocks are tiny (one aux shell pair
    // each, ~10⁵ per domain), so a per-block ndarray slice costs more than
    // its copy. Each lower block (bx ≥ by) is written once into column
    // block `by` and once, mirrored, into column block `bx`, both with
    // contiguous inner runs.
    let mut buf = vec![0.0f64; d * d];
    for (bx, &sp) in dsh.iter().enumerate() {
        for (by, &sq) in dsh.iter().enumerate().take(bx + 1) {
            let (key, transposed) = if sp >= sq {
                ((sp, sq), false)
            } else {
                ((sq, sp), true)
            };
            let blk = metric_blocks.get(key.0, key.1).ok_or_else(|| {
                FerricError::General(format!(
                    "lmp2_direct: metric block ({sp},{sq}) missing for {ctx}"
                ))
            })?;
            let at = (dims_df[sp], dims_df[sq], loc_off[bx], loc_off[by]);
            copy_metric_block(&mut buf, d, blk, transposed, at, bx != by);
        }
    }
    Ok(Array2::from_shape_vec((d, d).f(), buf).expect("d·d buffer"))
}

/// Copy one aux-shell-pair metric block into the column-major `d × d`
/// buffer at `V[r0 + p, c0 + q]`, `at = (dp, dq, r0, c0)` (`blk` is stored
/// `(dq, dp)` when `transposed`), and with `mirror` its transpose at
/// `V[c0 + q, r0 + p]`. Both inner loops run down a column.
fn copy_metric_block(
    buf: &mut [f64],
    d: usize,
    blk: &Array2<f64>,
    transposed: bool,
    at: (usize, usize, usize, usize),
    mirror: bool,
) {
    let (dp, dq, r0, c0) = at;
    let blk = blk.as_standard_layout();
    let b = blk.as_slice().expect("standard layout");
    let val = |p: usize, q: usize| {
        if transposed {
            b[q * dp + p]
        } else {
            b[p * dq + q]
        }
    };
    for q in 0..dq {
        let col = (c0 + q) * d + r0;
        for p in 0..dp {
            buf[col + p] = val(p, q);
        }
    }
    if mirror {
        for p in 0..dp {
            let col = (r0 + p) * d + c0;
            for q in 0..dq {
                buf[col + q] = val(p, q);
            }
        }
    }
}

/// Function ids of a sorted shell list.
fn shell_funcs(prep: &PreparedBasis, shells: &[usize]) -> Vec<usize> {
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut out = Vec::new();
    for &sh in shells {
        out.extend(offs[sh]..offs[sh] + dims[sh]);
    }
    out
}

/// `L` with `V = L Lᵀ` (lower Cholesky factor) of a gathered `V_DD`,
/// factored in place in `vdd`'s own buffer and layout.
fn chol_lower(vdd: Array2<f64>, what: &str) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{CholeskyInto, UPLO};
    vdd.cholesky_into(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("lmp2_direct V_DD Cholesky ({what}): {e}")))
}

/// `W = L⁻¹ A` (one triangular solve with many right-hand sides), in place
/// in `a`. With `l` and `a` both column-major (as [`gather_vdd`] and
/// [`gather_strip`] build them) LAPACK reads and writes them directly; a
/// row-major operand would be transposed into a copy first.
fn whiten(l: &Array2<f64>, a: Array2<f64>, what: &str) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{Diag, SolveTriangularInto, UPLO};
    l.solve_triangular_into(UPLO::Lower, Diag::NonUnit, a)
        .map_err(|e| FerricError::General(format!("lmp2_direct triangular solve ({what}): {e}")))
}

/// Rows `dfuncs` of occupied `k`'s strip, over the global virtuals `cols`
/// (`None`: every strip column, in strip order), as a column-major matrix
/// ready for [`whiten`]. See [`gather_strip_into`].
fn gather_strip(
    strips: &[Strip],
    k: usize,
    dfuncs: &[usize],
    cols: Option<&[usize]>,
    what: &str,
) -> Result<Array2<f64>, FerricError> {
    let ncols = cols.map_or(strips[k].cols.len(), <[usize]>::len);
    let mut a = Array2::<f64>::zeros((dfuncs.len(), ncols).f());
    gather_strip_into(strips, k, dfuncs, cols, what, a.view_mut())?;
    Ok(a)
}

/// [`gather_strip`] into a caller-owned `dfuncs.len() × |cols|` block (a
/// column range of a group's batched right-hand side). A row or column
/// missing from the strip is a construction bug (the extended domains must
/// cover every pair domain and candidate set) and errors loudly.
fn gather_strip_into(
    strips: &[Strip],
    k: usize,
    dfuncs: &[usize],
    cols: Option<&[usize]>,
    what: &str,
    mut out: ndarray::ArrayViewMut2<f64>,
) -> Result<(), FerricError> {
    let st = &strips[k];
    let mut srs = Vec::with_capacity(dfuncs.len());
    for &gp in dfuncs {
        let sr = st.row_pos[gp];
        if sr == usize::MAX {
            return Err(FerricError::General(format!(
                "lmp2_direct: strip {k} missing aux row {gp} for {what} — extended domain does \
                 not cover the fit domain (construction bug)"
            )));
        }
        srs.push(sr);
    }
    let scs: Vec<usize> = match cols {
        None => (0..st.cols.len()).collect(),
        Some(cols) => {
            let mut scs = Vec::with_capacity(cols.len());
            for &av in cols {
                let sc = st.col_pos[av];
                if sc == usize::MAX {
                    return Err(FerricError::General(format!(
                        "lmp2_direct: strip {k} missing virtual {av} for {what}"
                    )));
                }
                scs.push(sc);
            }
            scs
        }
    };
    for (mut col, &sc) in out.columns_mut().into_iter().zip(&scs) {
        for (x, &sr) in col.iter_mut().zip(&srs) {
            *x = st.b[(sr, sc)];
        }
    }
    Ok(())
}

/// The κ-screen inputs from one whitening per occupied over its OWN
/// distance domain D_i (stage 5a): `W_i = L_i⁻¹ B_i[D_i, all strip cols]`
/// with `L_i = chol(V_{D_i D_i})`, so `q_ia = √scale · ‖W_i[:, a]‖ =
/// √(scale · b_iaᵀ V⁻¹ b_ia)` with no explicit inverse. Returns
/// `(q per i, qmax per i, W_i per i)`. With `keep_w`, `W_i` is kept because
/// pair (i,i)'s domain group has domain exactly D_i: [`fit_domain_group`]
/// reuses its columns there instead of re-solving. The caller pre-flights
/// the full store (`8·Σ_i |D_i|·|cols_i|`) against the scratch budget and
/// passes `keep_w = false` over it; otherwise each `W_i` is dropped as soon
/// as its q is taken (one worker's `W` at a time). Under a rank stripe
/// `needed` marks the occupieds with a strip here; the others get an empty
/// q and `qmax = 0`. `floor` ([`FloorMode`]) adds each occupied's
/// [`OwnFloor`] (the pair skip's eigenvalue floor, from `L_i⁻¹`), or computes
/// only that.
#[allow(clippy::too_many_arguments)]
fn whiten_own_domains(
    no: usize,
    scale: f64,
    aux_dom: &[Vec<usize>],
    strips: &[Strip],
    dims_df: &[usize],
    metric_blocks: &MetricBlocks,
    dfbs: &PreparedBasis,
    aux_radius_bohr: f64,
    keep_w: bool,
    needed: Option<&[bool]>,
    floor: FloorMode,
) -> Result<OwnWhitened, FerricError> {
    use rayon::prelude::*;
    let sq = scale.abs().sqrt();
    let inputs = OwnInputs {
        aux_dom,
        strips,
        dims_df,
        metric_blocks,
        dfbs,
        aux_radius_bohr,
    };
    let per_i: Vec<Result<OwnPerI, FerricError>> = (0..no)
        .into_par_iter()
        .map(|i| {
            if needed.is_some_and(|n| !n[i]) {
                // no strip on this rank, and none of its pairs asks for q_i
                return Ok((Vec::new(), 0.0, keep_w.then(|| Array2::zeros((0, 0))), None));
            }
            whiten_one_own(&inputs, i, sq, keep_w, floor)
        })
        .collect();
    let mut out = OwnWhitened {
        q: Vec::with_capacity(no),
        qmax: Vec::with_capacity(no),
        w: keep_w.then(|| Vec::with_capacity(no)),
        floor: Vec::with_capacity(if floor == FloorMode::No { 0 } else { no }),
    };
    for r in per_i {
        let (q, qm, w, f) = r?;
        out.q.push(q);
        out.qmax.push(qm);
        if let (Some(ws), Some(w)) = (out.w.as_mut(), w) {
            ws.push(w);
        }
        if floor != FloorMode::No {
            out.floor.push(f);
        }
    }
    Ok(out)
}

/// Whether [`whiten_own_domains`] also computes the pair skip's
/// [`OwnFloor`]: `No`, `Also` (with q), or `Only` (q left empty, `qmax = 0`;
/// for occupieds whose q an earlier pass already holds).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FloorMode {
    No,
    Also,
    Only,
}

/// One occupied's own-domain whitening ([`whiten_own_domains`]): `(q_i,
/// max q_i, W_i if kept, its [`OwnFloor`] if asked)`.
type OwnPerI = (Vec<f64>, f64, Option<Array2<f64>>, Option<OwnFloor>);

/// The shared inputs of [`whiten_one_own`].
struct OwnInputs<'a> {
    aux_dom: &'a [Vec<usize>],
    strips: &'a [Strip],
    dims_df: &'a [usize],
    metric_blocks: &'a MetricBlocks,
    dfbs: &'a PreparedBasis,
    aux_radius_bohr: f64,
}

/// Occupied `i` of [`whiten_own_domains`]: `L_i = chol(V_{D_i D_i})`,
/// `W_i = L_i⁻¹ B_i[D_i, :]`, `q_ia = sq·‖W_i[:, a]‖`, and per `floor` the
/// pair skip's [`OwnFloor`] from the same `L_i`.
fn whiten_one_own(
    w: &OwnInputs,
    i: usize,
    sq: f64,
    keep_w: bool,
    floor: FloorMode,
) -> Result<OwnPerI, FerricError> {
    let dom = &w.aux_dom[i];
    let dfuncs = shell_funcs(w.dfbs, dom);
    if dfuncs.is_empty() {
        return Err(FerricError::General(format!(
            "lmp2_direct: empty aux domain for occupied {i} at radius {} \
             Bohr (virt_schwarz_kappa needs a nonempty D_i)",
            w.aux_radius_bohr
        )));
    }
    let what = format!("occupied {i} Schwarz diagonal");
    let vdd = gather_vdd(dom, w.dims_df, w.metric_blocks, &what)?;
    let vfloor = (floor != FloorMode::No).then(|| vdd.clone());
    let l = chol_lower(vdd, &what)?;
    let own = vfloor
        .map(|v| own_floor(&v, &l, (dom, w.dims_df, w.dfbs.nshells()), &what))
        .transpose()?;
    if floor == FloorMode::Only {
        return Ok((Vec::new(), 0.0, None, own));
    }
    let wi = whiten(&l, gather_strip(w.strips, i, &dfuncs, None, &what)?, &what)?;
    let q: Vec<f64> = wi
        .columns()
        .into_iter()
        .map(|c| sq * c.dot(&c).sqrt())
        .collect();
    let qm = q.iter().copied().fold(0.0f64, f64::max);
    Ok((q, qm, keep_w.then_some(wi), own))
}

/// One occupied's own-domain metric data for the pair skip
/// ([`crate::lmp2_pair_skip`]): `λ ≤ λ_min(V_{D_i D_i})` (`None`: too ill
/// conditioned to trust the computed inverse), the inflation that bounds the
/// exact own-domain norm by the computed one
/// ([`crate::lmp2_pair_skip::norm_inflation`]), an upper bound on
/// `tr V_{D_i D_i}` and the domain's function count.
#[derive(Debug, Clone)]
struct OwnFloor {
    lam: Option<f64>,
    qinfl: f64,
    trace: f64,
    d: usize,
    /// Per GLOBAL aux shell `P`: the computed `‖L̂⁻¹[:, P]‖_F` times `qinfl`
    /// for `P` in the domain, NaN outside it, so that `‖V^{-1/2}x‖ ≤ Σ_P c_P
    /// ‖x_P‖`: the shell-wise bound [`pair_g_bound`] compares with
    /// `‖x‖/√λ`.
    col: Vec<f64>,
}

/// [`OwnFloor`] from the own-domain metric `v` and its Cholesky factor:
/// `L⁻¹` by one triangular solve against the identity, `λ` the larger of
/// the trace floor `eigen_floor(‖L̂⁻¹‖_F², tr V, d)` and the shift-verified
/// floor (`crate::lmp2_pair_skip::verified_floor`), both proved below
/// `λ_min(V)` with worst-case rounding. No floor (`lam = None`) when neither
/// exists or the computed inverse is not trusted (no inflation).
fn own_floor(
    v: &Array2<f64>,
    l: &Array2<f64>,
    (dom, dims_df, nsh): (&[usize], &[usize], usize),
    what: &str,
) -> Result<OwnFloor, FerricError> {
    use crate::lmp2_pair_skip::gamma_n;
    use crate::lmp2_pair_skip::{eigen_floor, norm_inflation, verified_floor};
    let d = l.nrows();
    // the rounded diagonal sum, inflated to bound tr V from above
    let trace = v.diag().sum() * (1.0 + gamma_n(d));
    let mut eye = Array2::<f64>::zeros((d, d).f());
    eye.diag_mut().fill(1.0);
    let linv = whiten(l, eye, what)?;
    let fro2: f64 = linv.iter().map(|x| x * x).sum();
    let lam = match (eigen_floor(fro2, trace, d), verified_floor(v, &linv)) {
        (Some(t), Some(s)) => Some(t.max(s)),
        (t, s) => t.or(s),
    };
    let qinfl = lam.and_then(|lam| norm_inflation(fro2, trace, lam, d));
    let (lam, qinfl) = match qinfl {
        Some(q) => (lam, q),
        None => (None, f64::NAN),
    };
    let mut col = vec![f64::NAN; nsh];
    let mut c0 = 0usize;
    for &sh in dom {
        let c1 = c0 + dims_df[sh];
        let f2: f64 = linv
            .slice(ndarray::s![.., c0..c1])
            .iter()
            .map(|x| x * x)
            .sum();
        col[sh] = qinfl * f2.sqrt();
        c0 = c1;
    }
    Ok(OwnFloor {
        lam,
        qinfl,
        trace,
        d,
        col,
    })
}

/// Bytes of the full own-domain `W` store [`whiten_own_domains`] would keep:
/// `8·Σ_i |D_i|·|cols_i|` (the pre-flight of [`DirectStats::own_w_bytes`]).
fn own_w_store_bytes(aux_dom: &[Vec<usize>], strips: &[Strip], dims_df: &[usize]) -> usize {
    aux_dom
        .iter()
        .zip(strips)
        .map(|(dom, st)| 8 * dom.iter().map(|&sh| dims_df[sh]).sum::<usize>() * st.cols.len())
        .sum()
}

/// [`whiten_own_domains`]' output: the κ-screen magnitudes and (when kept)
/// the own-domain whitened strips they came from, in strip-column order.
struct OwnWhitened {
    q: Vec<Vec<f64>>,
    qmax: Vec<f64>,
    w: Option<Vec<Array2<f64>>>,
    /// Per occupied (empty unless requested): the pair skip's [`OwnFloor`],
    /// `None` for an occupied without a strip on this rank.
    floor: Vec<Option<OwnFloor>>,
}

/// Trim the kept own-domain `W_i` (strip-column order) to the columns its
/// reuse can read — the candidate union of i's pairs inside the group whose
/// domain is exactly D_i, the same set [`whiten_group_occupieds`] asks for
/// — and drop `W_i` where no such group exists. Trimming runs one `i` at a
/// time, so the store never grows past its full-width peak.
fn trim_own_w(
    own: Vec<Array2<f64>>,
    unique: &[(usize, usize)],
    pair_dom: &[Vec<usize>],
    pair_cands: &[Vec<usize>],
    aux_dom: &[Vec<usize>],
    strips: &[Strip],
) -> Vec<Option<GroupW>> {
    let mut want: Vec<Option<BTreeSet<usize>>> = vec![None; own.len()];
    for (px, &(i, j)) in unique.iter().enumerate() {
        if pair_cands[px].is_empty() {
            continue;
        }
        for k in [i, j] {
            if aux_dom[k] == pair_dom[px] {
                want[k]
                    .get_or_insert_with(BTreeSet::new)
                    .extend(pair_cands[px].iter().copied());
            }
        }
    }
    own.into_iter()
        .zip(want)
        .enumerate()
        .map(|(k, (w, cols))| {
            let cols: Vec<usize> = cols?.into_iter().collect();
            let pos: Vec<usize> = cols.iter().map(|&a| strips[k].col_pos[a]).collect();
            assert!(
                !pos.contains(&usize::MAX),
                "lmp2_direct: strip {k} missing a candidate virtual (construction bug)"
            );
            Some(GroupW {
                w: w.select(Axis(1), &pos),
                cols,
            })
        })
        .collect()
}

/// Per-slot output of one domain group's pair fits.
type SlotOut = Vec<(usize, Vec<PairBlock>)>;

/// One domain group's pair fits and counters, or its failure.
type GroupResult = Result<(SlotOut, GroupCounts), FerricError>;

/// One domain group's work counters (summed into [`DirectStats`]).
#[derive(Debug, Clone, Copy, Default)]
struct GroupCounts {
    /// Distinct occupieds over the group's pairs.
    n_occ: usize,
    /// Triangular solves run (one per distinct occupied not reused).
    n_trsm: usize,
    /// Distinct occupieds whose `W` came from the own-domain κ whitening.
    n_reused: usize,
    /// This group's scratch bytes, `8·(d² + Σ_k d·|C_k|)`.
    scratch_bytes: usize,
    /// Wall seconds of this group's parts, in [`DirectStats::pair_fit_thread_s`]
    /// order.
    t: [f64; 4],
    /// Every pair of the group was proved empty: nothing was fitted.
    skipped: bool,
}

/// Borrowed context for [`fit_domain_group`] — everything stage 5's group
/// workers read but never write. Extracted from the assembly driver
/// (complexity ceiling).
struct PairFitCtx<'a> {
    unique: &'a [(usize, usize)],
    pair_cands: &'a [Vec<usize>],
    strips: &'a [Strip],
    dims_df: &'a [usize],
    metric_blocks: &'a MetricBlocks,
    f_vv: &'a Array2<f64>,
    fo: &'a [f64],
    fv: &'a [f64],
    eps: f64,
    scale: f64,
    /// The optional Fock-propagated gate (`AmplitudeLmp2Config::fock_screen`).
    fock_screen: Option<f64>,
    nv: usize,
    aux_radius_bohr: f64,
    dfbs: &'a PreparedBasis,
    /// Each occupied's distance domain D_i (shell list), for the reuse test.
    aux_dom: &'a [Vec<usize>],
    /// Own-domain whitened strips from the κ screen, trimmed to their
    /// reusable columns (`None` when the screen is off or over budget;
    /// `Some(None)` entries where no group has domain exactly D_i).
    own_w: Option<&'a [Option<GroupW>]>,
    /// Per unique pair: proved empty by the pair skip, so not fitted (`None`:
    /// the skip is off or auditing).
    skip: Option<&'a [bool]>,
    /// [`PairSkip::Audit`]: every fitted pair's `(index, max|g|, max gate)`
    /// (scaled), for the comparison with its bound.
    audit: Option<&'a Mutex<Vec<(usize, f64, f64)>>>,
    /// Test-only mutant: in the group whose first pair index is `.0`,
    /// scale the first pair's i-occupied `W` by `1 + .1`.
    #[cfg(test)]
    w_mutant: Option<(usize, f64)>,
}

/// One whitened occupied inside a group: `w` over the sorted global
/// virtuals `cols` (the union of its pairs' candidate sets in the group).
struct GroupW {
    cols: Vec<usize>,
    w: Array2<f64>,
}

impl GroupW {
    /// Columns of `w` for the sorted subset `cij ⊆ cols`.
    fn select(&self, cij: &[usize]) -> Array2<f64> {
        self.w.select(Axis(1), &col_positions(&self.cols, cij, 0))
    }
}

/// Positions of the sorted subset `cij ⊆ cols` inside `cols`, shifted by
/// `off` (the first column of `cols` in the matrix that holds it).
fn col_positions(cols: &[usize], cij: &[usize], off: usize) -> Vec<usize> {
    cij.iter()
        .map(|a| {
            off + cols
                .binary_search(a)
                .expect("C_ij inside the group column union (construction)")
        })
        .collect()
}

/// Where one whitened occupied of a group lives.
enum WSrc {
    /// Columns `off..off + |cols|` of [`GroupWs::solved`].
    Solved(usize),
    /// Taken from the κ screen's own-domain `W` (no solve).
    Reused(Array2<f64>),
}

/// The whitened occupieds `W_k` of one domain group, each over the sorted
/// union `C_k` of its pairs' candidate sets. The solved ones sit side by
/// side in ONE matrix, the result of a single multi-right-hand-side
/// triangular solve against the group's factor (one BLAS-3 call per group
/// instead of one per occupied; the columns do not interact, so this is
/// the per-occupied solve up to the solver's blocking).
struct GroupWs {
    solved: Array2<f64>,
    at: HashMap<usize, (Vec<usize>, WSrc)>,
}

impl GroupWs {
    /// `W_k` over the sorted subset `cij ⊆ C_k`.
    fn select(&self, k: usize, cij: &[usize]) -> Array2<f64> {
        let (cols, src) = &self.at[&k];
        match src {
            WSrc::Solved(off) => self.solved.select(Axis(1), &col_positions(cols, cij, *off)),
            WSrc::Reused(w) => w.select(Axis(1), &col_positions(cols, cij, 0)),
        }
    }

    /// Bytes held: the solved matrix and every reused `W_k`.
    fn bytes(&self) -> usize {
        let reused: usize = self
            .at
            .values()
            .map(|(_, src)| match src {
                WSrc::Reused(w) => w.len(),
                WSrc::Solved(_) => 0,
            })
            .sum();
        8 * (self.solved.len() + reused)
    }
}

/// The Eq-8-masked blocks of pair (i,j) — and of (j,i) when i ≠ j — from
/// the fitted `g = J_ij` over the candidate set `cij`.
fn mask_pair(
    ctx: &PairFitCtx<'_>,
    i: usize,
    j: usize,
    mut g: Array2<f64>,
    cij: &[usize],
) -> Vec<PairBlock> {
    if ctx.scale != 1.0 {
        g.mapv_inplace(|x| ctx.scale * x);
    }
    pair_blocks_screened(
        i,
        j,
        &g,
        cij,
        ctx.nv,
        ctx.f_vv,
        ctx.fo,
        ctx.fv,
        ctx.eps,
        ctx.fock_screen,
    )
}

/// Whitened strips `W_k = L⁻¹ A_k[D, C_k]` for every distinct occupied `k`
/// of a group, over the union `C_k` of its pairs' candidate sets: every
/// `A_k` is gathered into its column range of one column-major right-hand
/// side, whitened by one triangular solve. A `k` whose own domain D_k IS the
/// group domain takes the columns of its κ-screen `W` instead.
fn whiten_group_occupieds(
    ctx: &PairFitCtx<'_>,
    dsh: &[usize],
    dfuncs: &[usize],
    l: &Array2<f64>,
    pxs: &[usize],
) -> Result<(GroupWs, GroupCounts), FerricError> {
    let want = group_cand_unions(pxs, ctx.unique, ctx.pair_cands);
    let mut counts = GroupCounts::default();
    let mut at = HashMap::with_capacity(want.len());
    let mut to_solve: Vec<(usize, Vec<usize>)> = Vec::new();
    for (k, cols) in want {
        let cols: Vec<usize> = cols.into_iter().collect();
        counts.n_occ += 1;
        let reuse = match ctx.own_w {
            Some(own) if ctx.aux_dom[k] == dsh => own[k].as_ref(),
            _ => None,
        };
        match reuse {
            Some(gw) => {
                counts.n_reused += 1;
                let w = gw.select(&cols);
                at.insert(k, (cols, WSrc::Reused(w)));
            }
            None => to_solve.push((k, cols)),
        }
    }
    counts.n_trsm += to_solve.len();
    let solved = solve_group_rhs(ctx, dfuncs, l, pxs.len(), to_solve, &mut at)?;
    #[allow(unused_mut)]
    let mut ws = GroupWs { solved, at };
    counts.scratch_bytes += ws.bytes();
    #[cfg(test)]
    apply_w_mutant(ctx, pxs, &mut ws);
    Ok((ws, counts))
}

/// Each distinct occupied `k` of a group's pairs → the union `C_k` of its
/// pairs' (nonempty) candidate sets.
fn group_cand_unions(
    pxs: &[usize],
    unique: &[(usize, usize)],
    pair_cands: &[Vec<usize>],
) -> BTreeMap<usize, BTreeSet<usize>> {
    let mut want: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for &px in pxs.iter().filter(|&&px| !pair_cands[px].is_empty()) {
        let (i, j) = unique[px];
        for k in [i, j] {
            want.entry(k)
                .or_default()
                .extend(pair_cands[px].iter().copied());
        }
    }
    want
}

/// The group's batched whitening: each `A_k[D, C_k]` of `to_solve` gathered
/// into its column range of one column-major right-hand side (recorded in
/// `at` as [`WSrc::Solved`]), then one triangular solve against `l` in place.
fn solve_group_rhs(
    ctx: &PairFitCtx<'_>,
    dfuncs: &[usize],
    l: &Array2<f64>,
    n_pairs: usize,
    to_solve: Vec<(usize, Vec<usize>)>,
    at: &mut HashMap<usize, (Vec<usize>, WSrc)>,
) -> Result<Array2<f64>, FerricError> {
    let width: usize = to_solve.iter().map(|(_, c)| c.len()).sum();
    let mut rhs = Array2::<f64>::zeros((dfuncs.len(), width).f());
    let mut off = 0usize;
    for (k, cols) in to_solve {
        let what = format!("occupied {k} in a group of {n_pairs} pairs");
        let block = rhs.slice_mut(s![.., off..off + cols.len()]);
        gather_strip_into(ctx.strips, k, dfuncs, Some(&cols), &what, block)?;
        let n = cols.len();
        at.insert(k, (cols, WSrc::Solved(off)));
        off += n;
    }
    match width {
        0 => Ok(rhs),
        _ => whiten(l, rhs, &format!("a group of {n_pairs} pairs")),
    }
}

/// Test-only `W` mutant (see `PairFitCtx::w_mutant`): in the group whose
/// first pair is `first`, scale that pair's i-occupied `W` by `1 + rel`.
#[cfg(test)]
fn apply_w_mutant(ctx: &PairFitCtx<'_>, pxs: &[usize], ws: &mut GroupWs) {
    let Some((first, rel)) = ctx.w_mutant else {
        return;
    };
    if pxs[0] != first {
        return;
    }
    match ws.at.get_mut(&ctx.unique[first].0) {
        Some((cols, WSrc::Solved(off))) => ws
            .solved
            .slice_mut(s![.., *off..*off + cols.len()])
            .mapv_inplace(|x| x * (1.0 + rel)),
        Some((_, WSrc::Reused(w))) => w.mapv_inplace(|x| x * (1.0 + rel)),
        None => {}
    }
}

/// One domain group's pair fits: `L = chol(V_DD)` once, `W_k = L⁻¹ A_k`
/// once per distinct occupied, then `J_ij = W_iᵀ W_j` per pair and the
/// Eq-8 mask. Same fit as `A_iᵀ V_DD⁻¹ A_j` without the explicit inverse
/// (the pre-Stage-C path survives as the test oracle
/// `fit_domain_group_inverse_oracle`).
fn fit_domain_group(
    ctx: &PairFitCtx<'_>,
    dsh: &[usize],
    pxs: &[usize],
) -> Result<(SlotOut, GroupCounts), FerricError> {
    let (i0, j0) = ctx.unique[pxs[0]];
    let dfuncs = shell_funcs(ctx.dfbs, dsh);
    if dfuncs.is_empty() {
        return Err(FerricError::General(format!(
            "lmp2_direct: empty aux domain for pair ({i0},{j0}) at radius {} Bohr",
            ctx.aux_radius_bohr
        )));
    }
    let what = format!("group of pair ({i0},{j0})");
    let mut clock = Instant::now();
    let vdd = gather_vdd(dsh, ctx.dims_df, ctx.metric_blocks, &what)?;
    let t_gather = lap(&mut clock);
    let l = chol_lower(vdd, &what)?;
    let t_chol = lap(&mut clock);
    let (ws, mut counts) = whiten_group_occupieds(ctx, dsh, &dfuncs, &l, pxs)?;
    counts.t = [t_gather, t_chol, lap(&mut clock), 0.0];
    counts.scratch_bytes += 8 * l.len();
    let mut out_group: SlotOut = Vec::with_capacity(pxs.len());
    for &px in pxs {
        let (i, j) = ctx.unique[px];
        let cij = &ctx.pair_cands[px];
        if cij.is_empty() || ctx.skip.is_some_and(|sk| sk[px]) {
            out_group.push((px, Vec::new()));
            continue;
        }
        let w_i = ws.select(i, cij);
        let g = if i == j {
            w_i.t().dot(&w_i)
        } else {
            w_i.t().dot(&ws.select(j, cij))
        };
        if let Some(audit) = ctx.audit {
            record_audit(ctx, audit, px, &g, cij);
        }
        out_group.push((px, mask_pair(ctx, i, j, g, cij)));
    }
    counts.t[3] = lap(&mut clock);
    Ok((out_group, counts))
}

/// [`PairSkip::Audit`]: record pair `px`'s scaled `max|g|` and the largest
/// value its retention test reads (`max(|g|, |g̃|/κ)` with the Fock screen,
/// else `max|g|`).
fn record_audit(
    ctx: &PairFitCtx<'_>,
    audit: &Mutex<Vec<(usize, f64, f64)>>,
    px: usize,
    g: &Array2<f64>,
    cij: &[usize],
) {
    let (i, j) = ctx.unique[px];
    let gs = g.mapv(|x| ctx.scale * x);
    let gmax = gs.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let gate_max = match ctx.fock_screen {
        Some(kappa) if ctx.eps > 0.0 => crate::ragged::fock_dressed_gate(
            &gs, cij, ctx.f_vv, ctx.fo[i], ctx.fo[j], ctx.fv, kappa,
        )
        .iter()
        .fold(0.0f64, |m, x| m.max(x.abs())),
        _ => gmax,
    };
    if let Ok(mut v) = audit.lock() {
        v.push((px, gmax, gate_max));
    }
}

/// Stage 5b's output ([`prove_empty_pairs`]).
struct SkipProof {
    /// Per unique pair: proved empty.
    empty: Vec<bool>,
    /// Per unique pair: the scaled bound on `max|g|` (NaN: none).
    g_bound: Vec<f64>,
    /// The dressing the proof used: `Some((κ, w_max))` with the Fock screen.
    fock: Option<(f64, f64)>,
}

/// `‖block‖_F²` of every evaluated metric shell pair, `(sp ≥ sq)` through a
/// dense `nsh²` table; NaN where the block was never evaluated (a bound that
/// reads one is not finite, so nothing is proved from it).
struct MetricFro2 {
    nsh: usize,
    fro2: Vec<f64>,
}

impl MetricFro2 {
    fn new(mb: &MetricBlocks) -> Self {
        let fro2 = mb
            .id
            .iter()
            .map(|&k| match k {
                u32::MAX => f64::NAN,
                k => mb.blocks[k as usize].iter().map(|x| x * x).sum(),
            })
            .collect();
        Self { nsh: mb.nsh, fro2 }
    }

    fn get(&self, p: usize, q: usize) -> f64 {
        let (a, b) = if p >= q { (p, q) } else { (q, p) };
        self.fro2[a * self.nsh + b]
    }
}

/// Per aux shell `P` of occupied `i`'s strip, `max_a Σ_{p∈P} B_i[p, a]²` over
/// every strip column (a superset of each pair's candidates); NaN for a
/// shell the strip has no rows of.
fn row_shell_maxsq(st: &Strip, func_shell: &[usize], nsh: usize) -> Vec<f64> {
    let mut t = vec![f64::NAN; nsh];
    let ncols = st.cols.len();
    let mut r0 = 0usize;
    while r0 < st.rows.len() {
        let sh = func_shell[st.rows[r0]];
        let mut r1 = r0 + 1;
        while r1 < st.rows.len() && func_shell[st.rows[r1]] == sh {
            r1 += 1;
        }
        let mut m = 0.0f64;
        for c in 0..ncols {
            let s: f64 = (r0..r1).map(|r| st.b[(r, c)] * st.b[(r, c)]).sum();
            m = m.max(s);
        }
        t[sh] = m;
        r0 = r1;
    }
    t
}

/// What [`skip_prefilter`] reads: data stage 5 already holds.
struct PrefilterIn<'a> {
    aux_dom: &'a [Vec<usize>],
    dims_df: &'a [usize],
    fro2: &'a MetricFro2,
    t: &'a [Vec<f64>],
    scale: f64,
}

/// The pair skip's cost-only prefilter: per unique pair, whether
/// [`pair_g_bound`] is worth evaluating under `On`. It never decides a skip;
/// a pair it leaves out is fitted as before. Its cut is a lower bound on the
/// γ-free terms `|scale|·(u_A v_A + u_B v_B)` that [`pair_g_bound`] compares
/// with the same `cutoff`, from the strip-norm table `t` and the metric
/// shell-block norms. With `f_P = ‖V_PP‖_F²` and `F_i = ‖V_{D_iD_i}‖_F ≥
/// λ_max`, the own-domain norms obey `u_A ≥ √(max_{P∈D_i} t_i[P] / F_i)`
/// (likewise `v_B` with `j`). Every diagonal element of `V_PP` is at most
/// `√f_P`, so `λ ≤ λ_min ≤ m = min_P √f_P` and `‖L⁻¹[:, P]‖_F² = Σ_{k∈P}
/// (V⁻¹)_kk ≥ dim_P/√f_P`; hence the cross norm `v_A ≥ min(√(Σ_{P∈D_i}
/// t_j[P] / m_i), Σ_{P∈D_i} √(dim_P t_j[P] / √f_P))`, one term per
/// alternative of [`pair_g_bound`], and `u_B` likewise over `B` with `m_j`.
/// With `stage2 = Some((kept, qmax))`, only the pairs stage 1 kept are
/// re-tested and the own norms are the computed `qmax` (at most `u_A`,
/// since the inflation is ≥ 1). A pair whose lower bound (times 0.999
/// against the rounding of these sums) reaches the cutoff fails the cutoff
/// in [`pair_g_bound`] too (book `#integral-direct-pair-skip`).
fn skip_prefilter(
    pf: &PrefilterIn,
    unique: &[(usize, usize)],
    mine: Option<&[bool]>,
    cutoff: f64,
    stage2: Option<(&[bool], &[f64])>,
) -> Vec<bool> {
    use rayon::prelude::*;
    let no = pf.aux_dom.len();
    // per occupied: lower bound on its own-domain norm, and m_i
    let own: Vec<(f64, f64)> = (0..no)
        .into_par_iter()
        .map(|i| {
            let (ti, dom) = (&pf.t[i], &pf.aux_dom[i]);
            if ti.is_empty() {
                return (f64::NAN, f64::NAN);
            }
            let m = dom
                .iter()
                .map(|&p| pf.fro2.get(p, p).sqrt())
                .fold(f64::INFINITY, f64::min);
            let u = match stage2 {
                Some((_, qmax)) => qmax[i],
                None => {
                    let tq = dom.iter().map(|&p| ti[p]).fold(0.0f64, f64::max);
                    let vf2: f64 = dom
                        .iter()
                        .map(|&p| dom.iter().map(|&q| pf.fro2.get(p, q)).sum::<f64>())
                        .sum();
                    (tq / vf2.sqrt()).sqrt()
                }
            };
            (u, m)
        })
        .collect();
    // lower bound on the cross norm of strip `x` over `shells`, whitened by
    // the occupied whose diagonal floor bound is `m`
    let cross = |x: &[f64], shells: &mut dyn Iterator<Item = usize>, m: f64| -> f64 {
        let (mut s, mut w) = (0.0f64, 0.0f64);
        for p in shells {
            s += x[p];
            w += (pf.dims_df[p] as f64 * x[p] / pf.fro2.get(p, p).sqrt()).sqrt();
        }
        (s / m).sqrt().min(w)
    };
    let sc = pf.scale.abs();
    unique
        .par_iter()
        .enumerate()
        .map(|(px, &(i, j))| {
            if i == j
                || mine.is_some_and(|m| !m[px])
                || pf.t[i].is_empty()
                || pf.t[j].is_empty()
                || stage2.is_some_and(|(kept, _)| !kept[px])
            {
                return false;
            }
            let (di, dj) = (&pf.aux_dom[i], &pf.aux_dom[j]);
            let ((u_a, m_i), (v_b, m_j)) = (own[i], own[j]);
            let v_a = cross(&pf.t[j], &mut di.iter().copied(), m_i);
            let mut b = dj.iter().copied().filter(|q| di.binary_search(q).is_err());
            let u_b = cross(&pf.t[i], &mut b, m_j);
            // NaN (a block or a strip row missing) fails the comparison too
            0.999 * sc * (u_a * v_a + u_b * v_b) < cutoff
        })
        .collect()
}

/// The occupieds the pairs marked in `a` read.
fn occupieds_of(no: usize, unique: &[(usize, usize)], a: &[bool]) -> Vec<bool> {
    let mut f = vec![false; no];
    for (&(i, j), _) in unique.iter().zip(a).filter(|(_, &x)| x) {
        f[i] = true;
        f[j] = true;
    }
    f
}

/// Test-only replacement of the prefilter's cut ([`PrefilterMutant`]).
#[cfg(test)]
fn prefilter_mutate(
    a: Vec<bool>,
    unique: &[(usize, usize)],
    mine: Option<&[bool]>,
    m: PrefilterMutant,
) -> Vec<bool> {
    let ok = |px: usize| unique[px].0 != unique[px].1 && mine.is_none_or(|v| v[px]);
    a.into_iter()
        .enumerate()
        .map(|(px, x)| match m {
            PrefilterMutant::Normal => x,
            PrefilterMutant::RejectAll => false,
            PrefilterMutant::Invert => !x && ok(px),
            PrefilterMutant::AttemptAll => ok(px),
        })
        .collect()
}

/// What every pair bound of [`pair_g_bound`] reads.
struct SkipTables<'a> {
    aux_dom: &'a [Vec<usize>],
    dims_df: &'a [usize],
    own: &'a OwnWhitened,
    t: &'a [Vec<f64>],
    fro2: &'a MetricFro2,
    /// Test-only weakened bound: scales every factor and γ.
    #[cfg(test)]
    mutant: f64,
    /// Test-only mutant: drop the γ-free terms `u_A v_A + u_B v_B` from both
    /// the `On` cutoff and the bound.
    #[cfg(test)]
    drop_free: bool,
}

/// The scaled upper bound on `max_ab |g_ab|` of pair `(i, j)`, `i ≠ j`
/// ([`crate::lmp2_pair_skip::fitted_bound`] with `A = D_i`,
/// `B = D_j \ D_i`, plus the floating-point allowance), or `None` when
/// nothing is proved. With `cutoff`, returns `None` as soon as the
/// γ-free terms alone reach it (the pair cannot be proved empty).
fn pair_g_bound(
    tb: &SkipTables<'_>,
    i: usize,
    j: usize,
    scale: f64,
    cutoff: Option<f64>,
) -> Option<f64> {
    use crate::lmp2_pair_skip::{fitted_bound, gamma_n, roundoff_allowance, PairNorms};
    let (fi, fj) = (tb.own.floor[i].as_ref()?, tb.own.floor[j].as_ref()?);
    let (li, lj) = (fi.lam?, fj.lam?);
    let (di, dj) = (&tb.aux_dom[i], &tb.aux_dom[j]);
    let b: Vec<usize> = dj
        .iter()
        .copied()
        .filter(|sh| di.binary_search(sh).is_err())
        .collect();
    let d = fi.d + b.iter().map(|&q| tb.dims_df[q]).sum::<usize>();
    // every sum below has at most d² + 4 rounded terms: inflate the norms
    // built from them (u_B, v_A, γ) by 1 + γ_{d²+4}
    let fsum = 1.0 + gamma_n(d * d + 4);
    let sa: f64 = di.iter().map(|&p| tb.t[j][p]).sum();
    let sb: f64 = b.iter().map(|&p| tb.t[i][p]).sum();
    // computed own-domain norms, inflated to bound the exact ones
    let (u_a, v_b) = (tb.own.qmax[i] * fi.qinfl, tb.own.qmax[j] * fj.qinfl);
    // ‖V_BB^{-1/2} a_B‖ ≤ ‖L_j⁻¹ (a_B, 0)‖ (Schur monotonicity), bounded
    // either by ‖a_B‖/√λ_j or shell by shell; likewise β_A with L_i
    let shellwise = |c: &[f64], t: &[f64], dom: &[usize]| -> f64 {
        dom.iter().map(|&p| c[p] * t[p].sqrt()).sum()
    };
    let u_b = fsum * (sb / lj).sqrt().min(shellwise(&fj.col, &tb.t[i], &b));
    let v_a = fsum * (sa / li).sqrt().min(shellwise(&fi.col, &tb.t[j], di));
    let sc = scale.abs();
    #[cfg(test)]
    let sc = sc * tb.mutant;
    // the γ-free terms (a test mutant drops them here AND in the bound)
    let free = u_a * v_a + u_b * v_b;
    #[cfg(test)]
    let free = if tb.drop_free { 0.0 } else { free };
    if cutoff.is_some_and(|c| sc * free >= c) {
        return None;
    }
    // ‖V_{P,B}‖_F² per shell P of A, and ‖V_{A,Q}‖_F² per shell Q of B
    let row2: Vec<f64> = di
        .iter()
        .map(|&p| b.iter().map(|&q| tb.fro2.get(p, q)).sum::<f64>())
        .collect();
    let vab2: f64 = row2.iter().sum();
    let col2: Vec<f64> = b
        .iter()
        .map(|&q| di.iter().map(|&p| tb.fro2.get(p, q)).sum::<f64>())
        .collect();
    // γ = ‖V_AA^{-1/2} V_AB V_BB^{-1/2}‖₂ ≤ ‖V_AB‖_F/√(λ_iλ_j), or one side
    // whitened shell by shell (the other side by its floor)
    let side_a: f64 = di
        .iter()
        .zip(&row2)
        .map(|(&p, r)| fi.col[p] * r.sqrt())
        .sum();
    let side_b: f64 = b
        .iter()
        .zip(&col2)
        .map(|(&q, c)| fj.col[q] * c.sqrt())
        .sum();
    let gamma = fsum
        * (vab2 / (li * lj))
            .sqrt()
            .min(side_a / lj.sqrt())
            .min(side_b / li.sqrt());
    #[cfg(test)]
    let gamma = gamma * tb.mutant;
    let n = PairNorms {
        u_a,
        u_b,
        v_a,
        v_b,
        gamma,
    };
    let g = fitted_bound(&n)?;
    #[cfg(test)]
    let g = if tb.drop_free {
        g - (u_a * v_a + u_b * v_b)
    } else {
        g
    };
    // tr V_DD ≤ tr V_{D_iD_i} + tr V_{D_jD_j}, since B ⊆ D_j
    let ro = roundoff_allowance(d, fi.trace + fj.trace, li.min(lj), &n);
    // relative slack for the rounding of the norms and of the dressed gate
    Some(sc * (g + ro) * (1.0 + 1e-9)).filter(|x| x.is_finite())
}

/// The retention test [`prove_empty_pairs`] proves pairs against: the
/// fitted-integral scale and `ε`, the Fock screen's `κ` (`None`: plain Eq-8
/// mask) and the occupied and virtual orbital energies.
#[derive(Clone, Copy)]
struct SkipGate<'a> {
    scale: f64,
    eps: f64,
    fock_screen: Option<f64>,
    fo: &'a [f64],
    fv: &'a [f64],
}

/// Peak bytes [`prove_empty_pairs`] allocates beside the resident strips,
/// metric and held `W`: the per-shell-pair metric norms (`8·nsh²`), per
/// occupied the strip row-shell table and the floor's column norms (`16·nsh`),
/// per pair its bound and verdict, and per concurrent worker the
/// largest own-domain workspace (`V_DD`, its copy, `L`, `L⁻¹` and the
/// shifted test matrix, `5·d²`, plus the gathered and whitened strip,
/// `2·d·cols`).
fn pair_proof_bytes(pre: &StripsAndMetric, dims_df: &[usize], n_pairs: usize) -> usize {
    let (no, nsh) = (pre.aux_dom.len(), pre.metric_blocks.nsh);
    let worker = pre
        .aux_dom
        .iter()
        .zip(&pre.strips)
        .map(|(dom, st)| {
            let d: usize = dom.iter().map(|&sh| dims_df[sh]).sum();
            8 * (5 * d * d + 2 * d * st.cols.len())
        })
        .max()
        .unwrap_or(0);
    let workers = rayon::current_num_threads().clamp(1, no.max(1));
    8 * nsh * nsh + 16 * no * nsh + 9 * n_pairs + workers * worker
}

/// Stage 5b, the provable-empty pair skip ([`crate::lmp2_pair_skip`]): the
/// own-domain fitted norms and eigenvalue floors (one Cholesky, one inverse
/// and one whitening per occupied), the per-shell strip and metric norm
/// tables, then one bound per off-diagonal pair of this rank. `None` when
/// the skip is off, at ε = 0, or when the Fock screen is on and its
/// denominators admit no bound.
fn prove_empty_pairs(
    pre: &StripsAndMetric,
    dfbs: &PreparedBasis,
    spaces: &LocalizedSpaces,
    unique: &[(usize, usize)],
    mine: Option<&[bool]>,
    gate: SkipGate<'_>,
    aux_radius_bohr: f64,
) -> Result<Option<SkipProof>, FerricError> {
    use crate::lmp2_pair_skip::{gate_bound, proved_empty, unit_fock_weight};
    use rayon::prelude::*;
    let SkipGate {
        scale,
        eps,
        fock_screen,
        fo,
        fv,
    } = gate;
    if !pre.pair_skip.evaluates() || !(eps > 0.0) {
        return Ok(None);
    }
    let fock = match fock_screen {
        None => None,
        Some(k) => match unit_fock_weight(&spaces.f_vv, fo, fv) {
            Some(w) => Some((k, w)),
            None => return Ok(None),
        },
    };
    let (no, nsh) = (spaces.no, dfbs.nshells());
    let dims_df = dfbs.shell_dims();
    let needed = pre.stripe.as_ref().map(|s| s.needed.as_slice());
    let mut func_shell = vec![0usize; dfbs.nbasis()];
    for (sh, (&off, &dim)) in dfbs.shell_offsets().iter().zip(dims_df).enumerate() {
        func_shell[off..off + dim].fill(sh);
    }
    let t: Vec<Vec<f64>> = (0..no)
        .into_par_iter()
        .map(|i| match needed {
            Some(n) if !n[i] => Vec::new(),
            _ => row_shell_maxsq(&pre.strips[i], &func_shell, nsh),
        })
        .collect();
    // Under `On`, a pair whose γ-free terms already reach ε through the
    // dressing cannot be proved empty: stop before the metric sum.
    let cutoff = (pre.pair_skip == PairSkip::On).then(|| eps / gate_bound(1.0, fock));
    // Under `On`, the cost-only prefilter ([`skip_prefilter`]): a pair whose
    // lower bound on the γ-free terms already reaches the cutoff is not
    // attempted, and an occupied no attempted pair reads gets no floor (its
    // pairs are fitted as before). Stage 1 uses norm tables only; stage 2
    // adds the computed own-domain norms of the occupieds stage 1 keeps.
    let fro2 = MetricFro2::new(&pre.metric_blocks);
    let whiten_for = |need: Option<&[bool]>, floor: FloorMode| {
        whiten_own_domains(
            no,
            1.0,
            &pre.aux_dom,
            &pre.strips,
            dims_df,
            &pre.metric_blocks,
            dfbs,
            aux_radius_bohr,
            false,
            need,
            floor,
        )
    };
    let attempt: Option<(Vec<bool>, Vec<bool>, OwnWhitened)> = match cutoff {
        None => None,
        Some(c) => {
            let pf = PrefilterIn {
                aux_dom: &pre.aux_dom,
                dims_df,
                fro2: &fro2,
                t: &t,
                scale,
            };
            let a1 = skip_prefilter(&pf, unique, mine, c, None);
            let had_q = occupieds_of(no, unique, &a1);
            let q = whiten_for(Some(&had_q), FloorMode::No)?;
            let a2 = skip_prefilter(&pf, unique, mine, c, Some((&a1, &q.qmax)));
            #[cfg(test)]
            let a2 = prefilter_mutate(a2, unique, mine, pre.prefilter_mutant);
            Some((a2, had_q, q))
        }
    };
    let own = match &attempt {
        None => whiten_for(needed, FloorMode::Also)?,
        Some((a, had_q, q)) => {
            // floors for the attempted pairs' occupieds, reusing stage 2's
            // q; an occupied without one (only a test mutant attempts such
            // a pair) gets it now, never a zero qmax
            let want = occupieds_of(no, unique, a);
            let only: Vec<bool> = want.iter().zip(had_q).map(|(&w, &h)| w && h).collect();
            let rest: Vec<bool> = want.iter().zip(had_q).map(|(&w, &h)| w && !h).collect();
            let mut own = whiten_for(Some(&only), FloorMode::Only)?;
            let fresh = whiten_for(Some(&rest), FloorMode::Also)?;
            own.qmax.clone_from(&q.qmax);
            for i in (0..no).filter(|&i| rest[i]) {
                own.qmax[i] = fresh.qmax[i];
                own.floor[i] = fresh.floor[i].clone();
            }
            own
        }
    };
    let attempt = attempt.map(|(a, _, _)| a);
    let tb = SkipTables {
        aux_dom: &pre.aux_dom,
        dims_df,
        own: &own,
        t: &t,
        fro2: &fro2,
        #[cfg(test)]
        mutant: pre.bound_mutant,
        #[cfg(test)]
        drop_free: pre.drop_free_mutant,
    };
    let g_bound: Vec<f64> = unique
        .par_iter()
        .enumerate()
        .map(|(px, &(i, j))| {
            if i == j || mine.is_some_and(|m| !m[px]) || attempt.as_ref().is_some_and(|a| !a[px]) {
                return f64::NAN;
            }
            pair_g_bound(&tb, i, j, scale, cutoff).unwrap_or(f64::NAN)
        })
        .collect();
    let empty = g_bound
        .iter()
        .map(|&g| proved_empty(g, eps, fock))
        .collect();
    Ok(Some(SkipProof {
        empty,
        g_bound,
        fock,
    }))
}

/// Seconds since `clock`, which restarts.
fn lap(clock: &mut Instant) -> f64 {
    let t = clock.elapsed().as_secs_f64();
    *clock = Instant::now();
    t
}

/// Cauchy–Schwarz factors for the stage-3 batch triple cut, same-op
/// kernel: |(P|μν)| ≤ √(P|P) · Q(μν), with the aux factor per aux SHELL
/// (√ of the max (P|P) diagonal over the shell — conservative at shell
/// granularity; dropping it makes the cut non-conservative wherever
/// √(P|P) > 1, found in review before any merge). `None` when the cut is
/// off. Schwarz bounds exist for Coulomb/erf/erfc/terfc only, so the error
/// names the `schwarz_skip = 0.0` fix; terfc factors carry the table-error
/// inflation of `schwarz::bound_inflation`.
///
/// The returned pair factor is `Q(μν) · w(μν)`, where `w` bounds the
/// orbital coefficients the triple is transformed with (see
/// [`coefficient_pair_weights`]): the cut then drops a shell triple only
/// when each of its FUNCTION triples contributes below `schwarz_skip` to
/// every TRANSFORMED entry `(P|ia)`, not just when its raw value is small.
/// It is a per-function-triple bound: the dropped shell block's summed
/// contribution to one `(P|ia)` can reach `n_μ n_ν` times it (the shell
/// sizes), and nothing bounds the sum over all dropped triples; the cut's
/// error is calibrated by measurement, as the raw cut's was. A bound on the raw `(P|μν)`
/// alone is not a bound on what it feeds: in a near-linearly-dependent
/// basis the localized virtuals carry |C| ~ 10² (aug-cc-pVDZ n-C12H26:
/// max |C_v| 119), and the raw cut at 1e-5 moved E_corr by -91 mHa, to
/// 79 mHa below canonical RI-MP2 (C10: -7.3 mHa; measured 2026-11-25,
/// `production_schwarz_cut_is_harmless_in_a_diffuse_basis`).
fn schwarz_triple_factors(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    spaces: &LocalizedSpaces,
    schwarz_skip: f64,
) -> Result<Option<(Array2<f64>, Vec<f64>)>, FerricError> {
    if schwarz_skip <= 0.0 {
        return Ok(None);
    }
    let sb = ferric_scf::screening::SchwarzBounds::compute(op, obs).map_err(|e| {
        FerricError::General(format!(
            "lmp2_direct: schwarz_skip = {schwarz_skip} requires shell-pair Schwarz bounds, \
             unavailable for operator {:?} ({e}); set schwarz_skip = 0.0 for this operator",
            op.kind
        ))
    })?;
    // The aux factor is the library's validated per-shell table (floored,
    // terfc-inflated), not a private copy.
    let qp_shell = ferric_integrals::schwarz::schwarz3_aux(op, dfbs)?;
    let w = coefficient_pair_weights(obs, &spaces.c_locc, &spaces.c_vloc);
    Ok(Some((sb.q * w, qp_shell)))
}

/// Largest |C_μk| over the functions μ of each obs shell and every column k.
fn shell_coeff_max(prep: &PreparedBasis, c: &Array2<f64>) -> Vec<f64> {
    let (dims, offs) = (prep.shell_dims(), prep.shell_offsets());
    (0..prep.nshells())
        .map(|sh| {
            c.slice(s![offs[sh]..offs[sh] + dims[sh], ..])
                .iter()
                .fold(0.0f64, |m, x| m.max(x.abs()))
        })
        .collect()
}

/// `w(μν) = max(1, cᵒ_μ cᵛ_ν + cᵒ_ν cᵛ_μ)` per shell pair, with `cᵒ`/`cᵛ`
/// the per-shell max |C| of the active occupieds / the virtuals: one stored
/// triple `(P|μν)` enters `(P|ia) = Σ_μν C_μi C_νa (P|μν)` through both
/// orientations, `(C_μi C_νa + C_νi C_μa)(P|μν)`, so
/// `√(P|P)·Q(μν)·w(μν) < skip` bounds each dropped FUNCTION triple's
/// contribution to every transformed entry by `skip` (the shell block sums
/// up to `n_μ n_ν` of them; `schwarz_cut_bounds_the_transformed_entries`
/// checks the function-level claim). The floor at 1 keeps the cut never
/// LOOSER than the raw-value cut it replaces: without it the small
/// coefficients of a compact basis drop more triples (C20/6-31G moved
/// +2.7 µHa from the no-cut energy, vs -0.2 µHa for the raw cut, same RHF),
/// because the per-triple bound says nothing about the sum over the triples
/// dropped. The raw cut's calibration (C12-C48 6-31G) therefore still holds
/// wherever every `w ≤ 1`.
fn coefficient_pair_weights(
    obs: &PreparedBasis,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
) -> Array2<f64> {
    let (co, cv) = (shell_coeff_max(obs, c_occ), shell_coeff_max(obs, c_vir));
    let n = obs.nshells();
    Array2::from_shape_fn((n, n), |(a, b)| (co[a] * cv[b] + co[b] * cv[a]).max(1.0))
}

/// Borrowed, read-only context of the stage-3 batch workers. Extracted from
/// the assembly driver (complexity ceiling); the per-batch arithmetic is
/// IDENTICAL to the pre-extraction inline loop.
struct BatchCtx<'a> {
    op: Operator,
    obs: &'a PreparedBasis,
    dfbs: &'a PreparedBasis,
    spaces: &'a LocalizedSpaces,
    aux_ext: &'a [Vec<usize>],
    virt_ext: &'a [Vec<usize>],
    occ_supp: &'a [Vec<usize>],
    virt_supp: &'a [Vec<usize>],
    /// Shell-pair Schwarz factors (Q(μν), per-aux-shell √max(P|P)) when the
    /// triple cut is on.
    qpair: Option<&'a (Array2<f64>, Vec<f64>)>,
    schwarz_skip: f64,
    /// Per-worker share of the process's batch-pass slab cap
    /// ([`batch_slab_cap`]), bytes.
    worker_budget: usize,
    /// [`DirectConfig::split_eri3_timers`].
    split_timers: bool,
}

/// One batch's compact AO axes: the occupied-support functions `s` and
/// the virtual-support functions `n`, with their shell sets and nao-length
/// inverse maps.
struct BatchAxes {
    u_shells: Vec<usize>,
    in_s: Vec<bool>,
    in_n: Vec<bool>,
    s_pos: Vec<usize>,
    n_pos: Vec<usize>,
    nsf: usize,
    nnf: usize,
}

impl BatchAxes {
    fn new(ctx: &BatchCtx<'_>, is: &[usize], batch_virts: &[usize]) -> Self {
        let obs = ctx.obs;
        let nao = obs.nbasis();
        let s_shells = sorted_union(is.iter().map(|&i| ctx.occ_supp[i].iter().copied()));
        let n_shells = sorted_union(
            batch_virts
                .iter()
                .map(|&a| ctx.virt_supp[a].iter().copied()),
        );
        let s_funcs = shell_funcs(obs, &s_shells);
        let n_funcs = shell_funcs(obs, &n_shells);
        let mut s_pos = vec![usize::MAX; nao];
        for (k, &f) in s_funcs.iter().enumerate() {
            s_pos[f] = k;
        }
        let mut n_pos = vec![usize::MAX; nao];
        for (k, &f) in n_funcs.iter().enumerate() {
            n_pos[f] = k;
        }
        let u_shells = sorted_union([s_shells.iter().copied(), n_shells.iter().copied()]);
        let mut in_s = vec![false; obs.nshells()];
        for &sh in &s_shells {
            in_s[sh] = true;
        }
        let mut in_n = vec![false; obs.nshells()];
        for &sh in &n_shells {
            in_n[sh] = true;
        }
        Self {
            u_shells,
            in_s,
            in_n,
            s_pos,
            n_pos,
            nsf: s_funcs.len(),
            nnf: n_funcs.len(),
        }
    }
}

/// Per-occupied support-masked coefficients on the batch's compact axes:
/// the occupied columns as the rows of one `(|is|, nsf)` matrix, and each
/// occupied's (nnf, |virt_ext_i|) virtual block.
fn batch_coefficients(
    ctx: &BatchCtx<'_>,
    ax: &BatchAxes,
    is: &[usize],
) -> (Array2<f64>, Vec<Array2<f64>>) {
    let dims_obs = ctx.obs.shell_dims();
    let offs_obs = ctx.obs.shell_offsets();
    let mut c_occ = Array2::<f64>::zeros((is.len(), ax.nsf));
    for (k, &i) in is.iter().enumerate() {
        for &sh in &ctx.occ_supp[i] {
            for f in 0..dims_obs[sh] {
                let g = offs_obs[sh] + f;
                c_occ[(k, ax.s_pos[g])] = ctx.spaces.c_locc[(g, i)];
            }
        }
    }
    let c_virt: Vec<Array2<f64>> = is
        .iter()
        .map(|&i| {
            let cols = &ctx.virt_ext[i];
            let mut cv = Array2::<f64>::zeros((ax.nnf, cols.len()));
            for (k, &a) in cols.iter().enumerate() {
                for &sh in &ctx.virt_supp[a] {
                    for f in 0..dims_obs[sh] {
                        let g = offs_obs[sh] + f;
                        cv[(ax.n_pos[g], k)] = ctx.spaces.c_vloc[(g, a)];
                    }
                }
            }
            cv
        })
        .collect();
    (c_occ, c_virt)
}

/// Zeroed strips (rows = i's extended aux functions, cols = i's extended
/// virtuals) for the batch's occupieds.
fn alloc_strips(ctx: &BatchCtx<'_>, is: &[usize]) -> Vec<Strip> {
    let naux = ctx.dfbs.nbasis();
    let nv = ctx.spaces.nv;
    is.iter()
        .map(|&i| {
            let rows = shell_funcs(ctx.dfbs, &ctx.aux_ext[i]);
            let mut row_pos = vec![usize::MAX; naux];
            for (k, &f) in rows.iter().enumerate() {
                row_pos[f] = k;
            }
            let cols = ctx.virt_ext[i].clone();
            let mut col_pos = vec![usize::MAX; nv];
            for (k, &a) in cols.iter().enumerate() {
                col_pos[a] = k;
            }
            let b = Array2::<f64>::zeros((rows.len(), cols.len()));
            Strip {
                rows,
                row_pos,
                cols,
                col_pos,
                b,
            }
        })
        .collect()
}

/// One (μ, ν) shell pair a batch needs: the shells, which slab orientations
/// it feeds (μ∈s,ν∈n and/or ν∈s,μ∈n), and its Schwarz factor `Q(μν)`
/// (0 when the triple cut is off).
#[derive(Debug, Clone, Copy)]
struct NeedPair {
    ua: usize,
    ub: usize,
    need_ab: bool,
    need_ba: bool,
    q: f64,
}

/// Timer granularity of [`fill_slab`]: the clock is read once per chunk of
/// this many shell pairs (twice when the eval/scatter split is on).
const FILL_CHUNK: usize = 64;

/// The batch's needed shell pairs, built ONCE per batch (was: every
/// `u_shells²` pair re-tested per aux shell). With the triple cut on they
/// are stably sorted by `Q(μν)` descending, so for each aux shell the
/// surviving pairs are a prefix found by binary search.
fn batch_pairs(ctx: &BatchCtx<'_>, ax: &BatchAxes) -> Vec<NeedPair> {
    let mut pairs = Vec::new();
    for (ux, &ua) in ax.u_shells.iter().enumerate() {
        for &ub in &ax.u_shells[..=ux] {
            let need_ab = ax.in_s[ua] && ax.in_n[ub];
            let need_ba = ax.in_s[ub] && ax.in_n[ua];
            if need_ab || need_ba {
                let q = ctx.qpair.map_or(0.0, |(q, _)| q[(ua, ub)]);
                pairs.push(NeedPair {
                    ua,
                    ub,
                    need_ab,
                    need_ba,
                    q,
                });
            }
        }
    }
    if ctx.qpair.is_some() {
        pairs.sort_by(|a, b| b.q.partial_cmp(&a.q).expect("NaN Schwarz factor"));
    }
    pairs
}

/// Write one (P|μν) block of aux shell `sp` into `t_slab[(pl, s, n)]` (both
/// orientations the pair needs).
fn scatter_block(
    ctx: &BatchCtx<'_>,
    ax: &BatchAxes,
    pr: &NeedPair,
    sp: usize,
    block: &[f64],
    p_local: &[usize],
    t_slab: &mut Array3<f64>,
) {
    let dims_obs = ctx.obs.shell_dims();
    let offs_obs = ctx.obs.shell_offsets();
    let n_p = ctx.dfbs.shell_dims()[sp];
    let off_p = ctx.dfbs.shell_offsets()[sp];
    let (n1, n2) = (dims_obs[pr.ua], dims_obs[pr.ub]);
    for p in 0..n_p {
        let pl = p_local[off_p + p];
        for fa in 0..n1 {
            let ga = offs_obs[pr.ua] + fa;
            for fb in 0..n2 {
                let gb = offs_obs[pr.ub] + fb;
                let val = block[(p * n1 + fa) * n2 + fb];
                if pr.need_ab {
                    t_slab[(pl, ax.s_pos[ga], ax.n_pos[gb])] = val;
                }
                if pr.need_ba {
                    t_slab[(pl, ax.s_pos[gb], ax.n_pos[ga])] = val;
                }
            }
        }
    }
}

/// Evaluate one slab's (P|μν) triples into `t_slab[(pl, s, n)]`. Returns
/// (evaluated, skipped). Per aux shell, the pairs passing the Schwarz cut
/// `√(P|P)·Q(μν) ≥ schwarz_skip` are the prefix of the Q-sorted list (the
/// SAME product and comparison as the unsorted test, so the same triples
/// are evaluated and the same ones counted as skipped). Every slab entry is
/// written only by its own shell pair, so evaluation order cannot change
/// the slab. The fill timer reads the clock once per [`FILL_CHUNK`] pairs;
/// with `ctx.split_timers` each chunk's blocks are buffered so evaluation
/// and scatter are timed apart (an extra copy, hence opt-in).
#[allow(clippy::too_many_arguments)]
fn fill_slab(
    ctx: &BatchCtx<'_>,
    ax: &BatchAxes,
    pairs: &[NeedPair],
    eng: &mut Engine,
    slab: &[usize],
    p_local: &[usize],
    t_slab: &mut Array3<f64>,
    tm: &mut BatchTimers,
) -> (u64, u64) {
    let (obs, dfbs) = (ctx.obs, ctx.dfbs);
    let (mut n_triples, mut n_skipped) = (0u64, 0u64);
    let mut buf: Vec<f64> = Vec::new();
    let mut meta: Vec<(NeedPair, usize)> = Vec::new();
    for &sp in slab {
        let kept = match ctx.qpair {
            Some((_, qp_shell)) => {
                let qp = qp_shell[sp];
                pairs.partition_point(|pr| qp * pr.q >= ctx.schwarz_skip)
            }
            None => pairs.len(),
        };
        n_skipped += (pairs.len() - kept) as u64;
        n_triples += kept as u64;
        for chunk in pairs[..kept].chunks(FILL_CHUNK) {
            let te = Instant::now();
            if !ctx.split_timers {
                for pr in chunk {
                    if let Some(block) = eng.compute_eri3(obs, dfbs, sp, pr.ua, pr.ub) {
                        scatter_block(ctx, ax, pr, sp, block, p_local, t_slab);
                    }
                }
                tm.fill_s += te.elapsed().as_secs_f64();
                continue;
            }
            buf.clear();
            meta.clear();
            for pr in chunk {
                if let Some(block) = eng.compute_eri3(obs, dfbs, sp, pr.ua, pr.ub) {
                    meta.push((*pr, buf.len()));
                    buf.extend_from_slice(block);
                } else {
                    tm.n_null_blocks += 1;
                }
            }
            let ts = Instant::now();
            for (pr, off) in &meta {
                scatter_block(ctx, ax, pr, sp, &buf[*off..], p_local, t_slab);
            }
            let tf = Instant::now();
            tm.n_integrals += buf.len() as u64;
            tm.eval_s += (ts - te).as_secs_f64();
            tm.scatter_s += (tf - ts).as_secs_f64();
            tm.fill_s += (tf - te).as_secs_f64();
        }
    }
    (n_triples, n_skipped)
}

/// The slab-fill kernel signature (a seam so tests can run the frozen
/// pre-B4 scan through the same batch worker).
type FillFn = fn(
    &BatchCtx<'_>,
    &BatchAxes,
    &[NeedPair],
    &mut Engine,
    &[usize],
    &[usize],
    &mut Array3<f64>,
    &mut BatchTimers,
) -> (u64, u64);

/// Half-transform one slab's rows into each batch occupied's strip:
/// row P of strip i is `c_occ_iᵀ · T_P · C_virt_i`. Two BLAS3 steps: per aux
/// function one GEMM over every batch occupied, `H_P = C_occ · T_P`
/// ((|is|, nsf)·(nsf, nnf)), so each `T_P` is read once rather than once per
/// occupied; then per occupied one GEMM of the `H` rows it needs with its
/// virtual block. An `H` row an occupied does not need
/// (its aux shell is outside that occupied's `aux_ext`) is computed and
/// dropped: cheaper than the per-occupied GEMVs it replaces.
fn transform_slab(
    ctx: &BatchCtx<'_>,
    is: &[usize],
    strips: &mut [Strip],
    (c_occ, c_virt): (&Array2<f64>, &[Array2<f64>]),
    p_local: &[usize],
    t_slab: &Array3<f64>,
) -> usize {
    let h = occ_half_transform(c_occ, t_slab);
    let mut per_occ_peak = 0usize;
    for (k, &i) in is.iter().enumerate() {
        let (pls, gps) = slab_rows(ctx, i, p_local);
        if pls.is_empty() {
            continue;
        }
        let g = h.index_axis(Axis(1), k).select(Axis(0), &pls);
        let rows = g.dot(&c_virt[k]);
        per_occ_peak = per_occ_peak.max(8 * (g.len() + rows.len() + pls.len() + gps.len()));
        assign_strip_rows(&mut strips[k], &gps, &rows);
    }
    8 * h.len() + per_occ_peak
}

/// The dimensions a batch slab's live bytes depend on: the batch's AO axes
/// (`nsf`, `nnf`), its occupied count, the widest virtual block of any of
/// them (`max_cols`) and the aux basis size (the slab's `p_local` map).
#[derive(Debug, Clone, Copy)]
struct SlabDims {
    nsf: usize,
    nnf: usize,
    n_occ: usize,
    max_cols: usize,
    naux: usize,
}

impl SlabDims {
    /// Bytes a batch slab of `nps` aux functions holds at its peak, while
    /// one occupied's rows are transformed: the `p_local` map (`naux`), the
    /// triples `t_slab` (`nps·nsf·nnf`), the occupied half-transform `H` of
    /// [`occ_half_transform`] (`nps·|is|·nnf`) and, for that occupied, its
    /// selected `H` rows (`≤ nps·nnf`), their product with its virtual block
    /// (`≤ nps·max_cols`) and the two row-index lists (`≤ 2·nps`). The slab
    /// is sized by this, so all of them stay inside the worker's budget.
    fn live_bytes(&self, nps: usize) -> usize {
        8 * self.naux + nps * self.row_bytes()
    }

    /// Bytes each slab aux function adds to [`Self::live_bytes`].
    fn row_bytes(&self) -> usize {
        8 * (self.nnf * (self.nsf + self.n_occ) + self.nnf + self.max_cols + 2)
    }

    /// Most aux functions a slab may hold under `budget` (at least one).
    fn max_funcs(&self, budget: usize) -> usize {
        (budget.saturating_sub(self.live_bytes(0)) / self.row_bytes()).max(1)
    }
}

/// `H[(pl, k, n)] = Σ_s c_occ[(k, s)] · t_slab[(pl, s, n)]`: one GEMM per aux
/// function, so each `H_P` is a contiguous `(|is|, nnf)` block.
fn occ_half_transform(c_occ: &Array2<f64>, t_slab: &Array3<f64>) -> Array3<f64> {
    let (nps, _, nnf) = t_slab.dim();
    let mut h = Array3::<f64>::zeros((nps, c_occ.nrows(), nnf));
    for (t_p, mut h_p) in t_slab.outer_iter().zip(h.outer_iter_mut()) {
        ndarray::linalg::general_mat_mul(1.0, c_occ, &t_p, 0.0, &mut h_p);
    }
    h
}

/// Occupied `i`'s aux functions present in this slab: their slab positions and
/// global ids, in `aux_ext[i]` order.
fn slab_rows(ctx: &BatchCtx<'_>, i: usize, p_local: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let dims_df = ctx.dfbs.shell_dims();
    let offs_df = ctx.dfbs.shell_offsets();
    let gps: Vec<usize> = ctx.aux_ext[i]
        .iter()
        .filter(|&&sp| p_local[offs_df[sp]] != usize::MAX) // shell in this slab
        .flat_map(|&sp| offs_df[sp]..offs_df[sp] + dims_df[sp])
        .collect();
    let pls = gps.iter().map(|&gp| p_local[gp]).collect();
    (pls, gps)
}

/// Store half-transformed rows in a strip: `rows` row `j` is the strip row of
/// global aux function `gps[j]`.
fn assign_strip_rows(strip: &mut Strip, gps: &[usize], rows: &Array2<f64>) {
    for (row, &gp) in rows.outer_iter().zip(gps) {
        let r = strip.row_pos[gp];
        strip.b.row_mut(r).assign(&row);
    }
}

/// End of the slab starting at `start`: the longest run of shells whose
/// function count stays within `max_funcs` (at least one shell).
fn slab_end(p_shells: &[usize], dims_df: &[usize], start: usize, max_funcs: usize) -> usize {
    let mut end = start;
    let mut nfuncs = 0usize;
    while end < p_shells.len() {
        let add = dims_df[p_shells[end]];
        if nfuncs > 0 && nfuncs + add > max_funcs {
            break;
        }
        nfuncs += add;
        end += 1;
    }
    end
}

/// Half-transform flops of one batch per aux function, per occupied: on the
/// batch-union axes as coded, and on i's own AO support (stats only).
fn transform_flop_rates(ctx: &BatchCtx<'_>, ax: &BatchAxes, is: &[usize]) -> Vec<(f64, f64)> {
    let dims = ctx.obs.shell_dims();
    let nfuncs = |shells: &[usize]| shells.iter().map(|&sh| dims[sh]).sum::<usize>() as f64;
    is.iter()
        .map(|&i| {
            let ncols = ctx.virt_ext[i].len() as f64;
            let own_n = sorted_union(
                ctx.virt_ext[i]
                    .iter()
                    .map(|&a| ctx.virt_supp[a].iter().copied()),
            );
            let (ns, nn) = (nfuncs(&ctx.occ_supp[i]), nfuncs(&own_n));
            let (bs, bn) = (ax.nsf as f64, ax.nnf as f64);
            (
                2.0 * bs * bn + 2.0 * bn * ncols,
                2.0 * ns * nn + 2.0 * nn * ncols,
            )
        })
        .collect()
}

/// One stage-3 batch worker: evaluate only the shell triples the batch's
/// occupieds need, slab by slab under the per-worker scratch share, and
/// half-transform into their strips.
fn eval_batch(ctx: &BatchCtx<'_>, is: &[usize], fill: FillFn) -> BatchOut {
    let t_start = Instant::now();
    let (obs, dfbs) = (ctx.obs, ctx.dfbs);
    let naux = dfbs.nbasis();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();
    let mut eng =
        Engine::new_3center(ctx.op, obs, dfbs, 1e-14).expect("3-center engine (pre-validated)");
    let mut tm = BatchTimers::default();
    let (mut n_triples, mut n_skipped) = (0u64, 0u64);
    let p_shells = sorted_union(is.iter().map(|&i| ctx.aux_ext[i].iter().copied()));
    let batch_virts = sorted_union(is.iter().map(|&i| ctx.virt_ext[i].iter().copied()));
    let ax = BatchAxes::new(ctx, is, &batch_virts);
    let (c_occ, c_virt) = batch_coefficients(ctx, &ax, is);
    let mut strips = alloc_strips(ctx, is);
    let pairs = batch_pairs(ctx, &ax);
    for (k, (full, own)) in transform_flop_rates(ctx, &ax, is).into_iter().enumerate() {
        let n_rows = strips[k].rows.len() as f64;
        tm.transform_flops += full * n_rows;
        tm.transform_flops_own_support += own * n_rows;
    }
    let dims = SlabDims {
        nsf: ax.nsf.max(1),
        nnf: ax.nnf.max(1),
        n_occ: is.len(),
        max_cols: c_virt.iter().map(|cv| cv.ncols()).max().unwrap_or(0),
        naux,
    };
    let max_funcs = dims.max_funcs(ctx.worker_budget);
    let mut slab_start = 0usize;
    while slab_start < p_shells.len() {
        let end = slab_end(&p_shells, dims_df, slab_start, max_funcs);
        let slab = &p_shells[slab_start..end];
        let mut p_local = vec![usize::MAX; naux];
        let mut nps = 0usize;
        for &sp in slab {
            for f in 0..dims_df[sp] {
                p_local[offs_df[sp] + f] = nps;
                nps += 1;
            }
        }
        let mut t_slab = Array3::<f64>::zeros((nps, ax.nsf, ax.nnf));
        let (nt, ns) = fill(
            ctx,
            &ax,
            &pairs,
            &mut eng,
            slab,
            &p_local,
            &mut t_slab,
            &mut tm,
        );
        n_triples += nt;
        n_skipped += ns;
        tm.n_slabs += 1;
        let tt = Instant::now();
        let transform_bytes =
            transform_slab(ctx, is, &mut strips, (&c_occ, &c_virt), &p_local, &t_slab);
        tm.transform_s += tt.elapsed().as_secs_f64();
        let live = 8 * (p_local.len() + t_slab.len()) + transform_bytes;
        tm.slab_bytes_max = tm.slab_bytes_max.max(live);
        slab_start = end;
    }
    tm.total_s = t_start.elapsed().as_secs_f64();
    (
        is.iter().copied().zip(strips).collect::<Vec<_>>(),
        n_triples,
        n_skipped,
        tm,
    )
}

/// One occupied's coefficients on its OWN AO support, built once for the whole
/// aux-major pass: the support mask per column is the same as
/// [`batch_coefficients`]'s.
struct OwnCoeffs {
    /// (shell, function within the shell, coefficient) of the occupied column.
    occ: Vec<(usize, usize, f64)>,
    /// Shells the virtual block covers.
    n_shells: Vec<usize>,
    /// Their functions, and the `(|n_funcs|, |virt_ext_i|)` virtual block.
    n_funcs: Vec<usize>,
    c_virt: Array2<f64>,
}

fn own_coeffs(ctx: &BatchCtx<'_>, i: usize) -> OwnCoeffs {
    let obs = ctx.obs;
    let (dims, offs) = (obs.shell_dims(), obs.shell_offsets());
    let occ = ctx.occ_supp[i]
        .iter()
        .flat_map(|&sh| (0..dims[sh]).map(move |f| (sh, f)))
        .map(|(sh, f)| (sh, f, ctx.spaces.c_locc[(offs[sh] + f, i)]))
        .collect();
    let cols = &ctx.virt_ext[i];
    let n_shells = sorted_union(cols.iter().map(|&a| ctx.virt_supp[a].iter().copied()));
    let n_funcs = shell_funcs(obs, &n_shells);
    let mut pos = vec![usize::MAX; obs.nbasis()];
    for (k, &g) in n_funcs.iter().enumerate() {
        pos[g] = k;
    }
    let mut c_virt = Array2::<f64>::zeros((n_funcs.len(), cols.len()));
    for (k, &a) in cols.iter().enumerate() {
        for &sh in &ctx.virt_supp[a] {
            for f in 0..dims[sh] {
                let g = offs[sh] + f;
                c_virt[(pos[g], k)] = ctx.spaces.c_vloc[(g, a)];
            }
        }
    }
    OwnCoeffs {
        occ,
        n_shells,
        n_funcs,
        c_virt,
    }
}

/// One aux shell's compact axes: the union of the supports of the occupieds
/// that need it. A shell's functions are consecutive on both axes.
struct AuxAxes {
    s_shells: Vec<usize>,
    n_shells: Vec<usize>,
    nsf: usize,
    nnf: usize,
}

impl AuxAxes {
    fn new(ctx: &BatchCtx<'_>, own: &[OwnCoeffs], is: &[usize]) -> Self {
        let dims = ctx.obs.shell_dims();
        let s_shells = sorted_union(is.iter().map(|&i| ctx.occ_supp[i].iter().copied()));
        let n_shells = sorted_union(is.iter().map(|&i| own[i].n_shells.iter().copied()));
        let nsf = s_shells.iter().map(|&sh| dims[sh]).sum();
        let nnf = n_shells.iter().map(|&sh| dims[sh]).sum();
        Self {
            s_shells,
            n_shells,
            nsf,
            nnf,
        }
    }

    /// Slab, half-transformed slab and occupied-coefficient bytes of one task.
    fn task_bytes(&self, np: usize, nocc: usize) -> usize {
        8 * (np * self.nsf * self.nnf + nocc * np * self.nnf + nocc * self.nsf)
    }
}

/// A worker's reusable per-task storage. The slab is kept between tasks so a
/// task's memset replaces a fresh mapping's page faults.
struct AuxWorker {
    eng: Engine,
    slab: Vec<f64>,
    /// Per shell: start on the `s` and `n` axes (usize::MAX = not on the axis).
    s_start: Vec<usize>,
    n_start: Vec<usize>,
    buf: Vec<f64>,
    meta: Vec<(NeedPair, usize)>,
}

/// Record each axis shell's start position; [`clear_starts`] undoes it.
fn set_starts(start: &mut [usize], shells: &[usize], dims: &[usize]) {
    let mut at = 0;
    for &sh in shells {
        start[sh] = at;
        at += dims[sh];
    }
}

fn clear_starts(start: &mut [usize], shells: &[usize]) {
    for &sh in shells {
        start[sh] = usize::MAX;
    }
}

/// Write one (P|μν) block into the slab, laid out `[s, (p, n)]`, both
/// orientations the pair needs.
fn scatter_aux_block(
    slab: &mut [f64],
    nnf: usize,
    np: usize,
    (n1, n2): (usize, usize),
    (ab, ba): (Option<(usize, usize)>, Option<(usize, usize)>),
    block: &[f64],
) {
    let s_stride = np * nnf;
    for p in 0..np {
        let plane = &block[p * n1 * n2..(p + 1) * n1 * n2];
        if let Some((s0, n0)) = ab {
            for (fa, row) in plane.chunks_exact(n2).enumerate() {
                let d = (s0 + fa) * s_stride + p * nnf + n0;
                slab[d..d + n2].copy_from_slice(row);
            }
        }
        if let Some((s0, n0)) = ba {
            for fb in 0..n2 {
                let d = (s0 + fb) * s_stride + p * nnf + n0;
                for fa in 0..n1 {
                    slab[d + fa] = plane[fa * n2 + fb];
                }
            }
        }
    }
}

/// The needed shell pairs of aux shell `sp` that survive the Schwarz cut, plus
/// the count cut. Same product and comparison as the batch pass.
fn aux_pairs(ctx: &BatchCtx<'_>, ax: &AuxAxes, w: &AuxWorker, sp: usize) -> (Vec<NeedPair>, u64) {
    let u_shells = sorted_union([ax.s_shells.iter().copied(), ax.n_shells.iter().copied()]);
    let qp = ctx.qpair.map(|(q, qp_shell)| (q, qp_shell[sp]));
    let mut kept = Vec::new();
    let mut cut = 0u64;
    for (ux, &ua) in u_shells.iter().enumerate() {
        for &ub in &u_shells[..=ux] {
            let need_ab = w.s_start[ua] != usize::MAX && w.n_start[ub] != usize::MAX;
            let need_ba = w.s_start[ub] != usize::MAX && w.n_start[ua] != usize::MAX;
            if !(need_ab || need_ba) {
                continue;
            }
            if let Some((q, qp)) = qp {
                if qp * q[(ua, ub)] < ctx.schwarz_skip {
                    cut += 1;
                    continue;
                }
            }
            kept.push(NeedPair {
                ua,
                ub,
                need_ab,
                need_ba,
                q: 0.0,
            });
        }
    }
    (kept, cut)
}

/// Evaluate the kept pairs of `sp` into the worker's slab.
fn fill_aux_slab(
    ctx: &BatchCtx<'_>,
    ax: &AuxAxes,
    w: &mut AuxWorker,
    pairs: &[NeedPair],
    sp: usize,
    tm: &mut BatchTimers,
) {
    let (obs, dfbs) = (ctx.obs, ctx.dfbs);
    let dims = obs.shell_dims();
    let np = dfbs.shell_dims()[sp];
    let AuxWorker {
        eng,
        slab,
        s_start,
        n_start,
        buf,
        meta,
    } = w;
    let place = |pr: &NeedPair| {
        (
            pr.need_ab.then(|| (s_start[pr.ua], n_start[pr.ub])),
            pr.need_ba.then(|| (s_start[pr.ub], n_start[pr.ua])),
        )
    };
    for chunk in pairs.chunks(FILL_CHUNK) {
        let te = Instant::now();
        if !ctx.split_timers {
            for pr in chunk {
                if let Some(block) = eng.compute_eri3(obs, dfbs, sp, pr.ua, pr.ub) {
                    let shape = (dims[pr.ua], dims[pr.ub]);
                    scatter_aux_block(slab, ax.nnf, np, shape, place(pr), block);
                }
            }
            tm.fill_s += te.elapsed().as_secs_f64();
            continue;
        }
        buf.clear();
        meta.clear();
        for pr in chunk {
            if let Some(block) = eng.compute_eri3(obs, dfbs, sp, pr.ua, pr.ub) {
                meta.push((*pr, buf.len()));
                buf.extend_from_slice(block);
            } else {
                tm.n_null_blocks += 1;
            }
        }
        let ts = Instant::now();
        for (pr, off) in meta.iter() {
            let shape = (dims[pr.ua], dims[pr.ub]);
            scatter_aux_block(slab, ax.nnf, np, shape, place(pr), &buf[*off..]);
        }
        let tf = Instant::now();
        tm.n_integrals += buf.len() as u64;
        tm.eval_s += (ts - te).as_secs_f64();
        tm.scatter_s += (tf - ts).as_secs_f64();
        tm.fill_s += (tf - te).as_secs_f64();
    }
}

/// Half-transform `sp`'s slab into the strip rows of its occupieds: one wide
/// GEMM over the occupieds, `H = C_occ · T`, then per occupied the block on its
/// own virtual support times its virtual coefficients. Rows of different aux
/// shells are disjoint, so the lock order cannot change any strip.
fn transform_aux_slab(
    ctx: &BatchCtx<'_>,
    ax: &AuxAxes,
    w: &AuxWorker,
    (own, strips): (&[OwnCoeffs], &[Mutex<Strip>]),
    sp: usize,
    is: &[usize],
    tm: &mut BatchTimers,
) {
    let dims = ctx.obs.shell_dims();
    let np = ctx.dfbs.shell_dims()[sp];
    let off_p = ctx.dfbs.shell_offsets()[sp];
    let mut c_occ = Array2::<f64>::zeros((is.len(), ax.nsf));
    for (k, &i) in is.iter().enumerate() {
        for &(sh, f, c) in &own[i].occ {
            c_occ[(k, w.s_start[sh] + f)] = c;
        }
    }
    let t = ArrayView2::from_shape((ax.nsf, np * ax.nnf), &w.slab[..ax.nsf * np * ax.nnf])
        .expect("aux slab shape");
    let h = c_occ.dot(&t);
    let gps: Vec<usize> = (off_p..off_p + np).collect();
    for (k, &i) in is.iter().enumerate() {
        let o = &own[i];
        let mut g_block = Array2::<f64>::zeros((np, o.n_funcs.len()));
        let mut col = 0;
        for &sh in &o.n_shells {
            let n0 = w.n_start[sh];
            for f in 0..dims[sh] {
                for p in 0..np {
                    g_block[(p, col)] = h[(k, p * ax.nnf + n0 + f)];
                }
                col += 1;
            }
        }
        let rows = g_block.dot(&o.c_virt);
        let mut strip = strips[i].lock().expect("strip lock");
        assign_strip_rows(&mut strip, &gps, &rows);
    }
    tm.transform_flops += 2.0 * (is.len() * ax.nsf * np * ax.nnf) as f64
        + is.iter()
            .map(|&i| 2.0 * (np * own[i].n_funcs.len() * own[i].c_virt.ncols()) as f64)
            .sum::<f64>();
}

/// One aux-shell task. Timers and counters land in the returned [`BatchOut`];
/// its strip list stays empty because the rows went straight to the strips.
#[allow(clippy::too_many_arguments)]
fn aux_task(
    ctx: &BatchCtx<'_>,
    w: &mut AuxWorker,
    own: &[OwnCoeffs],
    strips: &[Mutex<Strip>],
    sp: usize,
    is: &[usize],
    ax: &AuxAxes,
) -> BatchOut {
    let t_start = Instant::now();
    let dims = ctx.obs.shell_dims();
    let np = ctx.dfbs.shell_dims()[sp];
    let mut tm = BatchTimers {
        n_slabs: 1,
        ..Default::default()
    };
    set_starts(&mut w.s_start, &ax.s_shells, dims);
    set_starts(&mut w.n_start, &ax.n_shells, dims);
    let (pairs, cut) = aux_pairs(ctx, ax, w, sp);
    let len = ax.nsf * np * ax.nnf;
    w.slab.clear();
    w.slab.resize(len, 0.0);
    fill_aux_slab(ctx, ax, w, &pairs, sp, &mut tm);
    let tt = Instant::now();
    transform_aux_slab(ctx, ax, w, (own, strips), sp, is, &mut tm);
    tm.transform_s += tt.elapsed().as_secs_f64();
    clear_starts(&mut w.s_start, &ax.s_shells);
    clear_starts(&mut w.n_start, &ax.n_shells);
    tm.total_s = t_start.elapsed().as_secs_f64();
    (Vec::new(), pairs.len() as u64, cut, tm)
}

/// [`Eri3Pass::AuxMajor`]: one evaluation per needed triple. The budget bounds
/// the co-resident task scratch by limiting how many tasks run at once, so it
/// sets concurrency only and no arithmetic depends on it or on the thread
/// count. A single aux shell cannot be split: one over-budget task still runs
/// alone, the same floor as the batch pass's one-shell slab.
fn aux_major_pass(
    ctx: &BatchCtx<'_>,
    budget_bytes: usize,
    stats: &mut DirectStats,
) -> Vec<BatchOut> {
    use rayon::prelude::*;
    let dfbs = ctx.dfbs;
    let no = ctx.spaces.no;
    let own: Vec<OwnCoeffs> = (0..no)
        .into_par_iter()
        .map(|i| own_coeffs(ctx, i))
        .collect();
    stats.aux_major_coeff_bytes = own.iter().map(|o| 8 * o.c_virt.len()).sum();
    let mut owners: Vec<Vec<usize>> = vec![Vec::new(); dfbs.nshells()];
    for (i, shells) in ctx.aux_ext.iter().enumerate() {
        for &sp in shells {
            owners[sp].push(i);
        }
    }
    let tasks: Vec<(usize, &[usize], AuxAxes)> = owners
        .iter()
        .enumerate()
        .filter(|(_, is)| !is.is_empty())
        .map(|(sp, is)| (sp, is.as_slice(), AuxAxes::new(ctx, &own, is)))
        .collect();
    let dims_df = dfbs.shell_dims();
    let task_max = tasks
        .iter()
        .map(|(sp, is, ax)| ax.task_bytes(dims_df[*sp], is.len()))
        .max()
        .unwrap_or(0);
    let concurrent = (budget_bytes / task_max.max(1)).clamp(1, rayon::current_num_threads().max(1));
    stats.aux_major_task_bytes_max = task_max;
    stats.aux_major_concurrency = concurrent;
    let all: Vec<usize> = (0..no).collect();
    let strips: Vec<Mutex<Strip>> = alloc_strips(ctx, &all)
        .into_iter()
        .map(Mutex::new)
        .collect();
    let nsh = ctx.obs.nshells();
    let mut out: Vec<BatchOut> = Vec::with_capacity(tasks.len() + 1);
    let (in_flight, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
    // Fewer than a full pool only when the budget says so: a chunk at a time
    // keeps at most `concurrent` slabs alive.
    let chunk = if concurrent >= rayon::current_num_threads() {
        tasks.len().max(1)
    } else {
        concurrent
    };
    for group in tasks.chunks(chunk) {
        let done: Vec<BatchOut> = group
            .par_iter()
            .map_init(
                || AuxWorker {
                    eng: Engine::new_3center(ctx.op, ctx.obs, dfbs, 1e-14)
                        .expect("3-center engine (pre-validated)"),
                    slab: Vec::new(),
                    s_start: vec![usize::MAX; nsh],
                    n_start: vec![usize::MAX; nsh],
                    buf: Vec::new(),
                    meta: Vec::new(),
                },
                |w, (sp, is, ax)| {
                    let now = in_flight.fetch_add(1, Ordering::Relaxed) + 1;
                    peak.fetch_max(now, Ordering::Relaxed);
                    let done = aux_task(ctx, w, &own, &strips, *sp, is, ax);
                    in_flight.fetch_sub(1, Ordering::Relaxed);
                    done
                },
            )
            .collect();
        out.extend(done);
    }
    stats.aux_major_in_flight_peak = peak.into_inner();
    let built: Vec<(usize, Strip)> = strips
        .into_iter()
        .enumerate()
        .map(|(i, m)| (i, m.into_inner().expect("strip lock")))
        .collect();
    match out.first_mut() {
        Some(first) => first.0 = built,
        None => out.push((built, 0, 0, BatchTimers::default())),
    }
    out
}

/// Stage 1+2 products (integral-free): the pair gate, the per-occupied
/// distance/support maps, their partner-extended unions and the integral
/// batches. Extracted from the assembly driver (complexity ceiling); the
/// construction is unchanged.
struct LocalityMaps {
    keep: Vec<bool>,
    n_gated: usize,
    aux_dom: Vec<Vec<usize>>,
    virt_dom: Vec<Vec<usize>>,
    occ_supp: Vec<Vec<usize>>,
    virt_supp: Vec<Vec<usize>>,
    aux_ext: Vec<Vec<usize>>,
    virt_ext: Vec<Vec<usize>>,
    batches: Vec<(usize, Vec<usize>)>,
    /// The rank stripe ([`DirectConfig::mpi`] with more than one rank): the
    /// extended sets and batches above then cover this rank's pairs only.
    stripe: Option<RankStripe>,
}

fn centers(m: &Array2<f64>) -> Vec<[f64; 3]> {
    m.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect()
}

/// The Cartesian axis along which `pts` spread most (largest variance; the
/// lowest axis on a tie).
fn spread_axis(pts: &[[f64; 3]]) -> usize {
    let n = pts.len().max(1) as f64;
    let mut best = (0usize, -1.0f64);
    for x in 0..3 {
        let mean = pts.iter().map(|a| a[x]).sum::<f64>() / n;
        let var = pts.iter().map(|a| (a[x] - mean).powi(2)).sum::<f64>();
        if var > best.1 {
            best = (x, var);
        }
    }
    best.0
}

/// The surviving pairs `(i ≤ j)` in stage-5 order (i-major).
fn kept_pairs(keep: &[bool], no: usize) -> Vec<(usize, usize)> {
    (0..no)
        .flat_map(|i| (i..no).map(move |j| (i, j)))
        .filter(|&(i, j)| keep[i * no + j])
        .collect()
}

/// Each pair's fit domain `D_ij = D_i ∪ D_j` (aux shells).
fn pair_domains(unique: &[(usize, usize)], aux_dom: &[Vec<usize>]) -> Vec<Vec<usize>> {
    unique
        .iter()
        .map(|&(i, j)| sorted_union([aux_dom[i].iter().copied(), aux_dom[j].iter().copied()]))
        .collect()
}

/// One rank's share of the pair fits under [`DirectConfig::mpi`].
#[derive(Debug, Clone)]
struct RankStripe {
    ctx: ferric_core::parallel::ParallelContext,
    /// Per surviving pair ([`kept_pairs`] order): this rank fits it.
    mine: Vec<bool>,
    /// Per occupied: it is in a pair this rank fits, so this rank builds its
    /// strip.
    needed: Vec<bool>,
}

/// The rank that fits each domain group. Groups are ordered by where their
/// first pair sits along the molecule (`pos` = each occupied's place along
/// [`spread_axis`]; a pair sorts by its lower then its upper occupied) and
/// that order is cut into `size` runs of about equal `weight`. A rank's pairs
/// are then a contiguous run, so the partners an occupied has on one rank sit
/// close together and its strip there covers only their domains.
fn group_owners(
    group_list: &[(Vec<usize>, Vec<usize>)],
    unique: &[(usize, usize)],
    pos: &[usize],
    weight: &[f64],
    size: usize,
) -> Vec<usize> {
    let keys: Vec<(usize, usize)> = group_list
        .iter()
        .map(|(_, pxs)| {
            pxs.iter()
                .map(|&px| {
                    let (i, j) = unique[px];
                    (pos[i].min(pos[j]), pos[i].max(pos[j]))
                })
                .min()
                .expect("a domain group has at least one pair")
        })
        .collect();
    let mut order: Vec<usize> = (0..group_list.len()).collect();
    order.sort_unstable_by_key(|&g| (keys[g], g));
    let total: f64 = weight.iter().sum();
    let mut owner = vec![0usize; group_list.len()];
    let mut before = 0.0f64;
    for g in order {
        if total > 0.0 {
            let mid = before + 0.5 * weight[g];
            owner[g] = ((mid / total * size as f64) as usize).min(size - 1);
        }
        before += weight[g];
    }
    owner
}

/// Rank `ctx.rank`'s stripe of the pair fits, and each occupied's partners
/// on this rank (itself and every occupied it shares one of this rank's
/// pairs with; empty for an occupied this rank never fits). Integral-free and
/// the same arithmetic on every rank, so the ranks agree on the owners
/// without talking. A group's weight is a flop proxy for its fit: `d²·(d +
/// Σ_pairs |C_ij|)` for a domain of `d` functions.
fn rank_stripe(
    ctx: ferric_core::parallel::ParallelContext,
    keep: &[bool],
    aux_dom: &[Vec<usize>],
    virt_dom: &[Vec<usize>],
    occ_xyz: &[[f64; 3]],
    dims_df: &[usize],
) -> (RankStripe, Vec<Vec<usize>>) {
    let no = occ_xyz.len();
    let unique = kept_pairs(keep, no);
    let group_list = group_by_domain(pair_domains(&unique, aux_dom));
    let axis = spread_axis(occ_xyz);
    let mut along: Vec<usize> = (0..no).collect();
    along.sort_unstable_by(|&a, &b| {
        occ_xyz[a][axis]
            .partial_cmp(&occ_xyz[b][axis])
            .expect("NaN occupied centroid")
            .then(a.cmp(&b))
    });
    let mut pos = vec![0usize; no];
    for (p, &i) in along.iter().enumerate() {
        pos[i] = p;
    }
    let weight: Vec<f64> = group_list
        .iter()
        .map(|(dsh, pxs)| {
            let d: usize = dsh.iter().map(|&sh| dims_df[sh]).sum();
            let n: usize = pxs
                .iter()
                .map(|&px| {
                    let (i, j) = unique[px];
                    sorted_union([virt_dom[i].iter().copied(), virt_dom[j].iter().copied()]).len()
                })
                .sum();
            (d * d) as f64 * (d + n) as f64
        })
        .collect();
    let owner = group_owners(&group_list, &unique, &pos, &weight, ctx.size);
    let mut mine = vec![false; unique.len()];
    let mut partners: Vec<Vec<usize>> = vec![Vec::new(); no];
    for ((_, pxs), &r) in group_list.iter().zip(&owner) {
        if r != ctx.rank {
            continue;
        }
        for &px in pxs {
            mine[px] = true;
            let (i, j) = unique[px];
            partners[i].extend([i, j]);
            partners[j].extend([i, j]);
        }
    }
    for p in &mut partners {
        p.sort_unstable();
        p.dedup();
    }
    let needed = partners.iter().map(|p| !p.is_empty()).collect();
    (RankStripe { ctx, mine, needed }, partners)
}

/// Occupieds grouped by the atom nearest their Boys centroid; with
/// `batch_merge > 1`, ordered along the axis of maximal atomic spread and
/// merged in consecutive runs (amortizes the shell triples neighboring
/// batches re-evaluate — see `DirectConfig::batch_merge`).
fn integral_batches(
    atom_xyz: &[[f64; 3]],
    occ_xyz: &[[f64; 3]],
    batch_merge: usize,
) -> Vec<(usize, Vec<usize>)> {
    let mut batch_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, oc) in occ_xyz.iter().enumerate() {
        let (amin, _) = atom_xyz
            .iter()
            .enumerate()
            .map(|(ai, &axyz)| (ai, dist2(axyz, *oc)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).expect("NaN centroid distance"))
            .expect("molecule has no atoms");
        batch_map.entry(amin).or_default().push(i);
    }
    let mut batches: Vec<(usize, Vec<usize>)> = batch_map.into_iter().collect();
    batches.sort_unstable_by_key(|(a, _)| *a);
    if batch_merge <= 1 {
        return batches;
    }
    let axis = spread_axis(atom_xyz);
    batches.sort_unstable_by(|(a1, _), (a2, _)| {
        atom_xyz[*a1][axis]
            .partial_cmp(&atom_xyz[*a2][axis])
            .expect("NaN atom coordinate")
            .then(a1.cmp(a2))
    });
    batches
        .chunks(batch_merge)
        .map(|ch| {
            (
                ch[0].0,
                ch.iter().flat_map(|(_, is)| is.iter().copied()).collect(),
            )
        })
        .collect()
}

/// The rank stripe (`None` without [`DirectConfig::mpi`] or on one rank) and
/// each occupied's partners: every surviving partner, or under a stripe the
/// ones it shares one of this rank's pairs with ([`rank_stripe`]), so its
/// strip covers those pairs only.
fn stripe_and_partners(
    dcfg: &DirectConfig,
    keep: &[bool],
    aux_dom: &[Vec<usize>],
    virt_dom: &[Vec<usize>],
    occ_xyz: &[[f64; 3]],
    dfbs: &PreparedBasis,
) -> (Option<RankStripe>, Vec<Vec<usize>>) {
    let no = occ_xyz.len();
    match dcfg.mpi.filter(|c| c.size > 1) {
        None => (
            None,
            (0..no)
                .map(|i| (0..no).filter(|&j| keep[i * no + j]).collect())
                .collect(),
        ),
        Some(c) => {
            let (s, p) = rank_stripe(c, keep, aux_dom, virt_dom, occ_xyz, dfbs.shell_dims());
            (Some(s), p)
        }
    }
}

/// `batches` without the occupieds a rank stripe leaves to other ranks.
fn own_batches(
    mut batches: Vec<(usize, Vec<usize>)>,
    stripe: Option<&RankStripe>,
) -> Vec<(usize, Vec<usize>)> {
    if let Some(s) = stripe {
        batches.retain_mut(|(_, is)| {
            is.retain(|&i| s.needed[i]);
            !is.is_empty()
        });
    }
    batches
}

fn build_locality_maps(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    spaces: &LocalizedSpaces,
    eps: f64,
    pair_gate_cal: Option<f64>,
    dcfg: &DirectConfig,
) -> LocalityMaps {
    let (no, nv) = (spaces.no, spaces.nv);
    let (keep, n_gated) = match pair_gate_cal {
        Some(cal) => pair_gate_keep(&spaces.occ_centers, &spaces.occ_spreads, no, eps, cal),
        None => (vec![true; no * no], 0),
    };
    let atom_xyz: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let occ_xyz = centers(&spaces.occ_centers);
    let virt_xyz = centers(&spaces.virt_centers);
    let df_shell_atom = dfbs.shell_to_atom();
    let r_aux2 = dcfg.aux_radius_bohr * dcfg.aux_radius_bohr;
    let aux_dom: Vec<Vec<usize>> = occ_xyz
        .iter()
        .map(|oc| {
            (0..dfbs.nshells())
                .filter(|&sp| dist2(atom_xyz[df_shell_atom[sp]], *oc) <= r_aux2)
                .collect()
        })
        .collect();
    let virt_dom: Vec<Vec<usize>> = match dcfg.virt_radius_bohr {
        None => (0..no).map(|_| (0..nv).collect()).collect(),
        Some(rv) => occ_xyz
            .iter()
            .map(|oc| {
                (0..nv)
                    .filter(|&a| dist2(virt_xyz[a], *oc) <= rv * rv)
                    .collect()
            })
            .collect(),
    };
    let occ_supp = (0..no)
        .map(|i| supp_shells(obs, spaces.c_locc.column(i), dcfg.ao_tail))
        .collect();
    let virt_supp = (0..nv)
        .map(|a| supp_shells(obs, spaces.c_vloc.column(a), dcfg.ao_tail))
        .collect();
    let (stripe, partners) = stripe_and_partners(dcfg, &keep, &aux_dom, &virt_dom, &occ_xyz, dfbs);
    // extended per-occupied unions over surviving partners (pair sets are
    // D_ij = D_i ∪ D_j and C_ij = V_i ∪ V_j, both ⊆ the extended sets)
    let aux_ext = (0..no)
        .map(|i| sorted_union(partners[i].iter().map(|&j| aux_dom[j].iter().copied())))
        .collect();
    let virt_ext = (0..no)
        .map(|i| sorted_union(partners[i].iter().map(|&j| virt_dom[j].iter().copied())))
        .collect();
    let batches = own_batches(
        integral_batches(&atom_xyz, &occ_xyz, dcfg.batch_merge),
        stripe.as_ref(),
    );
    LocalityMaps {
        keep,
        n_gated,
        aux_dom,
        virt_dom,
        occ_supp,
        virt_supp,
        aux_ext,
        virt_ext,
        batches,
        stripe,
    }
}

/// Integral-direct ragged assembly: never forms a global 3-index tensor or
/// a global metric factorization. Returns
/// `(ragged, gated_unique_pairs, stats)`. See the module doc for the
/// pipeline and the anchor contract.
#[allow(clippy::too_many_arguments)]
pub fn assemble_ragged_direct_local(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    spaces: &LocalizedSpaces,
    eps: f64,
    scale: f64,
    pair_gate_cal: Option<f64>,
    dcfg: &DirectConfig,
) -> Result<(Ragged, usize, DirectStats), FerricError> {
    assemble_gated(
        mol,
        obs,
        dfbs,
        op,
        spaces,
        eps,
        scale,
        pair_gate_cal,
        dcfg,
        None,
        None,
    )
}

/// [`assemble_ragged_direct_local`] with the optional Fock-propagated gate
/// on every pair's candidate square (`AmplitudeLmp2Config::fock_screen`,
/// [`crate::ragged::fock_dressed_gate`]). `None` is
/// [`assemble_ragged_direct_local`] exactly. Ungated against the memory
/// budget, like [`assemble_ragged_direct_local`].
#[allow(clippy::too_many_arguments)]
pub fn assemble_ragged_direct_local_screened(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    spaces: &LocalizedSpaces,
    eps: f64,
    scale: f64,
    pair_gate_cal: Option<f64>,
    dcfg: &DirectConfig,
    fock_screen: Option<f64>,
) -> Result<(Ragged, usize, DirectStats), FerricError> {
    assemble_gated(
        mol,
        obs,
        dfbs,
        op,
        spaces,
        eps,
        scale,
        pair_gate_cal,
        dcfg,
        None,
        fock_screen,
    )
}

/// [`assemble_ragged_direct_local_screened`] gated against the memory
/// budget of the LMP2 solve it feeds (`gate = None`: ungated, the
/// historical behaviour; `fock_screen = None`: the plain Eq-8 mask).
#[allow(clippy::too_many_arguments)]
fn assemble_gated(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    spaces: &LocalizedSpaces,
    eps: f64,
    scale: f64,
    pair_gate_cal: Option<f64>,
    dcfg: &DirectConfig,
    gate: Option<SolveGate>,
    fock_screen: Option<f64>,
) -> Result<(Ragged, usize, DirectStats), FerricError> {
    let mut stats = DirectStats::default();
    let pre = strips_and_metric(
        mol,
        obs,
        dfbs,
        op,
        spaces,
        eps,
        pair_gate_cal,
        dcfg,
        gate,
        &mut stats,
    )?;
    let t0 = Instant::now();
    let input = PairFitInput {
        pre: &pre,
        dfbs,
        spaces,
        eps,
        scale,
        dcfg,
        path: FitPath::Cholesky,
        gate,
        fock_screen,
    };
    let (slots, _) = fit_pair_slots(input, &mut stats)?;
    mem_trace(
        "fits",
        &[(
            "own_blocks",
            slots
                .iter()
                .flatten()
                .flatten()
                .map(PairBlock::heap_bytes)
                .sum(),
        )],
    );
    // The fits were the last readers of the strips and the metric: drop them
    // before the ranks exchange blocks, so a rank never holds its strips
    // beside the whole ragged system.
    let (n_gated, ctx) = (pre.n_gated, pre.stripe.as_ref().map(|s| s.ctx));
    drop(pre);
    let ragged = gather_ragged(slots, ctx, gate, 0, t0, &mut stats)?;
    Ok((ragged, n_gated, stats))
}

/// What stages 1–4 hand to stage 5: the gate, the distance domains, the
/// unwhitened strips and the domain-restricted metric.
struct StripsAndMetric {
    keep: Vec<bool>,
    n_gated: usize,
    aux_dom: Vec<Vec<usize>>,
    virt_dom: Vec<Vec<usize>>,
    strips: Vec<Strip>,
    metric_blocks: MetricBlocks,
    /// The rank stripe: `strips` then holds [`Strip::absent`] for every
    /// occupied outside `stripe.needed`, and the built ones cover this
    /// rank's pairs only.
    stripe: Option<RankStripe>,
    /// [`DirectConfig::pair_skip`] resolved for the operator (never `Auto`).
    pair_skip: PairSkip,
    /// Test-only mutant: every pair bound is multiplied by this (1 = the
    /// real bound; below 1 weakens it, which the anchor and audit must see).
    #[cfg(test)]
    bound_mutant: f64,
    /// Test-only mutant: drop the γ-free terms from the cutoff and the bound
    /// alike ([`SkipTables`]).
    #[cfg(test)]
    drop_free_mutant: bool,
    /// Test-only prefilter mutant ([`skip_prefilter`]): `Normal`, or one
    /// that attempts nothing, inverts the cut, or attempts every pair.
    #[cfg(test)]
    prefilter_mutant: PrefilterMutant,
}

/// Test-only replacements for [`skip_prefilter`]'s cut.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefilterMutant {
    Normal,
    RejectAll,
    Invert,
    AttemptAll,
}

/// Stages 1–4 of [`assemble_ragged_direct_local`]: gate and maps
/// (integral-free), the batched 3-index evaluation + half-transform into
/// per-occupied strips, and the domain-restricted 2-center metric. Fills
/// the stage-1–4 counters and timers of `stats`.
#[allow(clippy::too_many_arguments)]
fn strips_and_metric(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    spaces: &LocalizedSpaces,
    eps: f64,
    pair_gate_cal: Option<f64>,
    dcfg: &DirectConfig,
    gate: Option<SolveGate>,
    stats: &mut DirectStats,
) -> Result<StripsAndMetric, FerricError> {
    use rayon::prelude::*;
    let no = spaces.no;
    let pair_skip = dcfg
        .pair_skip
        .resolve(&op)
        .map_err(|e| FerricError::General(format!("lmp2_direct: {e}")))?;
    let t0 = Instant::now();
    let maps = build_locality_maps(mol, obs, dfbs, spaces, eps, pair_gate_cal, dcfg);
    let need = metric_need_pairs(&maps.aux_ext, dfbs.nshells());
    let mut est = preflight_memory(&maps, &need, dfbs, no, spaces.nv, eps);
    let mut scratch_budget_bytes = dcfg.scratch_budget_bytes;
    if let Some(g) = gate {
        // Stage 3 holds the strips beside the slab scratch: the scratch gets
        // what the budget leaves. Strips depend on the slab partition only at
        // round-off (the per-occupied virtual GEMM's row count follows the
        // slab), so this only changes how many slabs run.
        scratch_budget_bytes =
            stage3_scratch(scratch_budget_bytes, g.budget_bytes, est.strips_bytes);
    }
    let slab_cap = batch_slab_cap(dcfg, scratch_budget_bytes);
    est.stage3_scratch_bytes = match dcfg.eri3_pass {
        Eri3Pass::Batches => slab_cap,
        Eri3Pass::AuxMajor => scratch_budget_bytes,
    };
    stats.memory = est;
    mem_trace(
        "preflight",
        &[
            ("strips", est.strips_bytes),
            ("metric", est.metric_bytes),
            ("pair_blocks", est.pair_block_bytes),
            ("pcg_vector", est.pcg_vector_bytes),
            ("resident_peak", est.resident_peak()),
        ],
    );
    if let Some(g) = gate {
        if est.strips_bytes + est.metric_bytes > g.budget_bytes
            || (est.exact && est.resident_peak() > g.budget_bytes)
        {
            return Err(memory_refusal(
                &est,
                g.budget_bytes,
                "refused before any integral",
            ));
        }
    }
    let dcfg = &DirectConfig {
        scratch_budget_bytes,
        ..dcfg.clone()
    };
    let LocalityMaps {
        keep,
        n_gated,
        aux_dom,
        virt_dom,
        occ_supp,
        virt_supp,
        aux_ext,
        virt_ext,
        batches,
        stripe,
    } = maps;
    stats.t_maps_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    if dcfg.eri3_pass == Eri3Pass::AuxMajor && dcfg.batch_merge > 1 {
        return Err(FerricError::General(format!(
            "lmp2_direct: eri3_pass = AuxMajor has no batches, so batch_merge = {} means nothing; \
             set batch_merge = 1",
            dcfg.batch_merge
        )));
    }
    Engine::new_3center(op, obs, dfbs, 1e-14)?; // surface construction errors serially
    let qpair = schwarz_triple_factors(op, obs, dfbs, spaces, dcfg.schwarz_skip)?;
    stats.t_eri3_setup_s = t0.elapsed().as_secs_f64();
    let n_workers = rayon::current_num_threads()
        .max(1)
        .min(batches.len().max(1));
    let bctx = BatchCtx {
        op,
        obs,
        dfbs,
        spaces,
        aux_ext: &aux_ext,
        virt_ext: &virt_ext,
        occ_supp: &occ_supp,
        virt_supp: &virt_supp,
        qpair: qpair.as_ref(),
        schwarz_skip: dcfg.schwarz_skip,
        worker_budget: slab_cap / n_workers,
        split_timers: dcfg.split_eri3_timers,
    };
    let batch_out: Vec<BatchOut> = match dcfg.eri3_pass {
        Eri3Pass::Batches => batches
            .par_iter()
            .map(|(_, is)| eval_batch(&bctx, is, fill_slab))
            .collect(),
        Eri3Pass::AuxMajor => aux_major_pass(&bctx, dcfg.scratch_budget_bytes, stats),
    };
    let needed = stripe.as_ref().map(|s| s.needed.as_slice());
    let strips = collect_strips(no, batch_out, dcfg.split_eri3_timers, needed, stats)?;
    stats.t_eri3_s = t0.elapsed().as_secs_f64();
    mem_trace(
        "stage3",
        &[
            ("strips", stats.strip_bytes_total),
            ("scratch_cap", stats.memory.stage3_scratch_bytes),
            ("slab_max", stats.eri3_slab_bytes_max),
            ("workers_x_slab_max", n_workers * stats.eri3_slab_bytes_max),
        ],
    );

    let t0 = Instant::now();
    stats.n_metric_shell_pairs = need.len();
    let metric_blocks = eval_metric_blocks(op, dfbs, &need)?;
    stats.t_metric_s = t0.elapsed().as_secs_f64();
    mem_trace(
        "strips+metric",
        &[
            ("strips", stats.strip_bytes_total),
            ("metric", metric_blocks.bytes()),
        ],
    );
    Ok(StripsAndMetric {
        keep,
        n_gated,
        aux_dom,
        virt_dom,
        strips,
        metric_blocks,
        stripe,
        pair_skip,
        #[cfg(test)]
        bound_mutant: 1.0,
        #[cfg(test)]
        drop_free_mutant: false,
        #[cfg(test)]
        prefilter_mutant: PrefilterMutant::Normal,
    })
}

/// Batch outputs → one strip per occupied (an occupied no batch built is a
/// construction bug), summing the stage-3 counters, timers and strip sizes.
/// Under a rank stripe `needed` marks the occupieds this rank builds; the
/// others get [`Strip::absent`], and the row/column means are over the
/// built strips.
fn collect_strips(
    no: usize,
    batch_out: Vec<BatchOut>,
    split_timers: bool,
    needed: Option<&[bool]>,
    stats: &mut DirectStats,
) -> Result<Vec<Strip>, FerricError> {
    stats.n_batches = batch_out.len();
    if !split_timers {
        stats.t_eri3_eval_thread_s = f64::NAN;
        stats.t_eri3_scatter_thread_s = f64::NAN;
    }
    let mut strips: Vec<Option<Strip>> = (0..no).map(|_| None).collect();
    for (batch_strips, n_triples, n_skipped, tm) in batch_out {
        stats.n_eri3_shell_triples += n_triples;
        stats.n_eri3_skipped += n_skipped;
        stats.t_eri3_fill_thread_s += tm.fill_s;
        stats.t_eri3_eval_thread_s += tm.eval_s;
        stats.t_eri3_scatter_thread_s += tm.scatter_s;
        stats.t_eri3_transform_thread_s += tm.transform_s;
        stats.n_eri3_slabs += tm.n_slabs;
        stats.transform_flops += tm.transform_flops;
        stats.transform_flops_own_support += tm.transform_flops_own_support;
        stats.n_eri3_integrals += tm.n_integrals;
        stats.n_eri3_null_blocks += tm.n_null_blocks;
        stats.eri3_slab_bytes_max = stats.eri3_slab_bytes_max.max(tm.slab_bytes_max);
        stats.batch_work_max_s = stats.batch_work_max_s.max(tm.total_s);
        stats.batch_work_mean_s += tm.total_s;
        for (i, st) in batch_strips {
            strips[i] = Some(st);
        }
    }
    if stats.n_batches > 0 {
        stats.batch_work_mean_s /= stats.n_batches as f64;
    }
    let strips = fill_absent_strips(strips, needed)?;
    let n_built = needed.map_or(no, |n| n.iter().filter(|&&b| b).count());
    for st in &strips {
        stats.strip_rows_mean += st.rows.len() as f64;
        stats.strip_rows_max = stats.strip_rows_max.max(st.rows.len());
        stats.strip_cols_mean += st.cols.len() as f64;
        stats.strip_cols_max = stats.strip_cols_max.max(st.cols.len());
        stats.strip_bytes_total += st.b.len() * 8;
    }
    if n_built > 0 {
        stats.strip_rows_mean /= n_built as f64;
        stats.strip_cols_mean /= n_built as f64;
    }
    Ok(strips)
}

/// One strip per occupied: the built one, [`Strip::absent`] for an occupied
/// the rank stripe leaves to other ranks, an error for any other gap.
fn fill_absent_strips(
    strips: Vec<Option<Strip>>,
    needed: Option<&[bool]>,
) -> Result<Vec<Strip>, FerricError> {
    strips
        .into_iter()
        .enumerate()
        .map(|(i, s)| match s {
            Some(s) => Ok(s),
            None if needed.is_some_and(|n| !n[i]) => Ok(Strip::absent()),
            None => Err(FerricError::General(format!(
                "lmp2_direct: no strip built for i={i}"
            ))),
        })
        .collect()
}

/// Everything stage 5 reads: what [`fit_pairs`] is called with, bundled for
/// [`fit_pair_slots`].
#[derive(Clone, Copy)]
struct PairFitInput<'a> {
    pre: &'a StripsAndMetric,
    dfbs: &'a PreparedBasis,
    spaces: &'a LocalizedSpaces,
    eps: f64,
    scale: f64,
    dcfg: &'a DirectConfig,
    path: FitPath,
    gate: Option<SolveGate>,
    fock_screen: Option<f64>,
}

/// Which stage-5 fit runs: the production Cholesky/triangular-solve path,
/// or (tests only) the frozen pre-Stage-C explicit-inverse oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FitPath {
    Cholesky,
    #[cfg(test)]
    InverseOracle,
    /// The Cholesky path with one group's `W` perturbed (see
    /// `PairFitCtx::w_mutant`) — the per-pair error check's mutant.
    #[cfg(test)]
    MutantW(usize, f64),
}

impl FitPath {
    /// Whether this path whitens by triangular solves (counted in `n_trsm`).
    fn triangular(self) -> bool {
        match self {
            FitPath::Cholesky => true,
            #[cfg(test)]
            FitPath::InverseOracle => false,
            #[cfg(test)]
            FitPath::MutantW(..) => true,
        }
    }
}

/// Stage 5a, the ε-linked Schwarz virtual-candidate screen
/// (WIKI-APPEND-eps-linked-maps.md). `q_ia = √(b_iaᵀ V_{D_i D_i}⁻¹ b_ia)`
/// from the strip rows over i's OWN distance domain D_i (inside the strip's
/// extended rows, since i is its own partner); `a` stays in C_ij iff
/// `scale·q_ia·qmax_j ≥ κε` or `scale·q_ja·qmax_i ≥ κε` — both Eq-8
/// orientations, Cauchy–Schwarz-exact on a global Gram fit and measured
/// escape-free against the domain-local fit at κ ≤ 1. Off (ε = 0 or no κ):
/// `None`. Also returns the own-domain whitened strips for reuse.
/// The own-domain `W` store is pre-flighted: kept only when its full width
/// fits [`DirectConfig::scratch_budget_bytes`] (recorded in
/// `stats.own_w_bytes`).
#[allow(clippy::type_complexity)]
fn schwarz_screen(
    pre: &StripsAndMetric,
    dfbs: &PreparedBasis,
    no: usize,
    scale: f64,
    dcfg: &DirectConfig,
    path: FitPath,
    stats: &mut DirectStats,
) -> Result<(Option<(Vec<Vec<f64>>, Vec<f64>)>, Option<Vec<Array2<f64>>>), FerricError> {
    let (dims_df, mb, r) = (dfbs.shell_dims(), &pre.metric_blocks, dcfg.aux_radius_bohr);
    match path {
        #[cfg(test)]
        FitPath::MutantW(..) => {
            schwarz_screen(pre, dfbs, no, scale, dcfg, FitPath::Cholesky, stats)
        }
        FitPath::Cholesky => {
            stats.own_w_bytes = own_w_store_bytes(&pre.aux_dom, &pre.strips, dims_df);
            let keep = stats.own_w_bytes <= dcfg.scratch_budget_bytes;
            let (aux_dom, strips) = (&pre.aux_dom, &pre.strips);
            let needed = pre.stripe.as_ref().map(|s| s.needed.as_slice());
            let nf = FloorMode::No;
            let o = whiten_own_domains(
                no, scale, aux_dom, strips, dims_df, mb, dfbs, r, keep, needed, nf,
            )?;
            Ok((Some((o.q, o.qmax)), o.w))
        }
        #[cfg(test)]
        FitPath::InverseOracle => {
            let q = inverse_oracle::schwarz_virtual_q_inverse_oracle(
                no,
                scale,
                &pre.aux_dom,
                &pre.strips,
                dims_df,
                mb,
                dfbs,
                r,
            )?;
            Ok((Some(q), None))
        }
    }
}

/// Per-pair candidate sets C_ij = V_i ∪ V_j, screened by `qvirt` when the
/// κ screen is on. The screen only ever SHRINKS the union, so the
/// strip-coverage invariant is untouched; `a ∈ V_i ∪ V_j` lies in both
/// strips' extended columns (i, j are mutual partners), so a missing
/// column position is a construction bug and panics loudly. Under a rank
/// stripe `mine` marks this rank's pairs; the others get an empty set and
/// their strips are not read.
fn pair_candidates(
    unique: &[(usize, usize)],
    virt_dom: &[Vec<usize>],
    strips: &[Strip],
    qvirt: Option<&(Vec<Vec<f64>>, Vec<f64>)>,
    kappa_eps: Option<f64>,
    mine: Option<&[bool]>,
) -> Vec<Vec<usize>> {
    unique
        .iter()
        .enumerate()
        .map(|(px, &(i, j))| {
            if mine.is_some_and(|m| !m[px]) {
                return Vec::new();
            }
            let cij = sorted_union([virt_dom[i].iter().copied(), virt_dom[j].iter().copied()]);
            match (qvirt, kappa_eps) {
                (Some((q, qmax)), Some(th)) => cij
                    .into_iter()
                    .filter(|&a| {
                        let qia = q[i][strips[i].col_pos[a]];
                        let qja = q[j][strips[j].col_pos[a]];
                        qia * qmax[j] >= th || qja * qmax[i] >= th
                    })
                    .collect(),
                _ => cij,
            }
        })
        .collect()
}

/// Upper bound on one domain group's scratch in [`fit_domain_group`],
/// counted from its allocations. Held for the whole group: the Cholesky
/// factor, computed in `V_DD`'s own buffer (`d²`), and one whitened `W_k`
/// per distinct occupied over its candidate union (`Σ_k d·|C_k|`). The
/// whitening has no transient: every `A_k` is gathered straight into its
/// column range of one column-major right-hand side, which the triangular
/// solve overwrites with the `W_k` ([`whiten_group_occupieds`]). On top of
/// that, the pair fits' transient: the widest pair's two column selections,
/// `J_ij` and its transpose (`2·d·n + 2·n²`), and with the Fock-propagated
/// screen on
///   (`screen_on`) the gate scratch of
///   [`crate::ragged::pair_blocks_screened`] (`F'`, `t⁰`, `F't⁰`, the `t⁰F'`
///   temporary, the gate and its transpose,
///   [`FOCK_SCREEN_SCRATCH_MATRICES`]` · n²`, all six counted although at
///   most four are live at once).
///
/// The pair blocks a group builds are not scratch: they are bounded by
/// [`group_block_bound`] and charged at their exact bytes once built.
fn group_scratch_bound(
    dfbs: &PreparedBasis,
    dsh: &[usize],
    pxs: &[usize],
    unique: &[(usize, usize)],
    pair_cands: &[Vec<usize>],
    screen_on: bool,
) -> usize {
    let dims = dfbs.shell_dims();
    let d: usize = dsh.iter().map(|&sh| dims[sh]).sum();
    let n_max = pxs
        .iter()
        .map(|&px| pair_cands[px].len())
        .max()
        .unwrap_or(0);
    let want = group_cand_unions(pxs, unique, pair_cands);
    let w: usize = want.values().map(|c| d * c.len()).sum();
    let screen = if screen_on {
        FOCK_SCREEN_SCRATCH_MATRICES * n_max * n_max
    } else {
        0
    };
    let fits = 2 * d * n_max + 2 * n_max * n_max + screen;
    8 * (d * d + w + fits)
}

/// Upper bound on the pair blocks one domain group builds: each pair's
/// block (and its mirror when `i ≠ j`) at the full candidate width
/// `n = |C_ij|`, [`block_bytes`] each — the Eq-8 mask only drops rows and
/// columns, so the built blocks are never larger.
fn group_block_bound(
    pxs: &[usize],
    unique: &[(usize, usize)],
    pair_cands: &[Vec<usize>],
    nv: usize,
) -> usize {
    pxs.iter()
        .map(|&px| {
            let n = pair_cands[px].len();
            let (i, j) = unique[px];
            match n {
                0 => 0,
                _ if i == j => block_bytes(n, n, nv),
                _ => 2 * block_bytes(n, n, nv),
            }
        })
        .sum()
}

/// One stage-5 wave: groups `start..end` of the group list, run together
/// with at most `workers` of them in flight, and the scratch reserved for
/// them (`min(len, workers)` × the wave's largest group scratch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GroupWave {
    end: usize,
    scratch: usize,
    /// The wave stopped because the next group did not fit, not because the
    /// list ran out.
    budget_limited: bool,
}

/// The next wave from `start`: as many groups as fit in `avail` bytes,
/// charging each group's block bound (`blocks[g]`) — its blocks stay
/// resident — and `min(len, workers)` copies of the wave's largest scratch
/// bound (`scratch[g]`), since any `workers` of the wave's groups can run at
/// once. Always at least one group, so the schedule makes progress; a lone
/// group that does not fit is left to the charge-as-built gate, which
/// refuses it on its exact bytes.
fn plan_group_wave(
    start: usize,
    scratch: &[usize],
    blocks: &[usize],
    workers: usize,
    avail: usize,
) -> GroupWave {
    let workers = workers.max(1);
    let need = |len: usize, s_max: usize, b_sum: usize| {
        b_sum.saturating_add(len.min(workers).saturating_mul(s_max))
    };
    let (mut s_max, mut b_sum) = (scratch[start], blocks[start]);
    let mut end = start + 1;
    while end < scratch.len() {
        let (s, b) = (s_max.max(scratch[end]), b_sum.saturating_add(blocks[end]));
        if need(end + 1 - start, s, b) > avail {
            return GroupWave {
                end,
                scratch: (end - start).min(workers).saturating_mul(s_max),
                budget_limited: true,
            };
        }
        (s_max, b_sum) = (s, b);
        end += 1;
    }
    GroupWave {
        end,
        scratch: (end - start).min(workers).saturating_mul(s_max),
        budget_limited: false,
    }
}

/// Candidate-square matrices the Fock-propagated gate holds per pair
/// (see [`group_scratch_bound`]).
const FOCK_SCREEN_SCRATCH_MATRICES: usize = 6;

/// Stage-5 charge-as-built gate: every group's blocks are charged at their
/// EXACT bytes (pair block + PCG vectors) the moment they exist, so at ε > 0,
/// where the pre-flight is only a bound, the assembly stops at the first
/// group that takes the stage-5 resident set (strips + metric + own-domain
/// `W` + the running wave's group scratch + blocks so far) or the solve's
/// (blocks + PCG vectors) over the budget. Groups not yet started are
/// skipped once it trips.
struct BlockCharge {
    budget_bytes: usize,
    base_bytes: usize,
    est: DirectMemoryEstimate,
    /// The running wave's reserved group scratch ([`GroupWave::scratch`]).
    scratch: AtomicUsize,
    ragged: AtomicUsize,
    vectors: AtomicUsize,
    tripped: Mutex<Option<String>>,
}

impl BlockCharge {
    fn is_tripped(&self) -> bool {
        match self.tripped.lock() {
            Ok(t) => t.is_some(),
            Err(_) => true,
        }
    }

    fn charge(&self, out: &SlotOut) {
        let (mut r, mut v) = (0usize, 0usize);
        for pb in out.iter().flat_map(|(_, blocks)| blocks) {
            r += std::mem::size_of::<PairBlock>() + pb.heap_bytes();
            v += amp_block_bytes(pb.da.len(), pb.db.len());
        }
        let r_tot = self.ragged.fetch_add(r, Ordering::Relaxed) + r;
        let v_tot = self.vectors.fetch_add(v, Ordering::Relaxed) + v;
        let solve = r_tot + crate::ragged::PCG_RESIDENT_VECTORS * v_tot;
        let base = self.base_bytes + self.scratch.load(Ordering::Relaxed);
        let held = (base + r_tot).max(solve);
        if held <= self.budget_bytes {
            return;
        }
        let gb = |b: usize| b as f64 / 1e9;
        if let Ok(mut t) = self.tripped.lock() {
            t.get_or_insert_with(|| {
                format!(
                    "lmp2_direct: refused during the pair fits: the blocks built so far already \
                     need {:.2} GB (pair blocks {:.2} GB + {} PCG amplitude vectors {:.2} GB; \
                     strips + metric + group scratch {:.2} GB), over the memory budget of \
                     {:.2} GB; the whole system needs at most {:.2} GB (candidate-domain \
                     bound). The pair blocks \
                     and PCG vectors stay resident for the whole solve and are not blocked. \
                     Raise [memory] budget_gb / FERRIC_MEM_BUDGET_GB, or shrink the pair \
                     blocks: a larger [local] eps, a smaller virt_radius, or the pair gate \
                     (gate_cal)",
                    gb(held),
                    gb(r_tot),
                    crate::ragged::PCG_RESIDENT_VECTORS,
                    gb(crate::ragged::PCG_RESIDENT_VECTORS * v_tot),
                    gb(base),
                    gb(self.budget_bytes),
                    gb(self.est.resident_peak()),
                )
            });
        }
    }
}

/// Stage 5 of [`assemble_ragged_direct_local`]: per-pair domain fits + the
/// Eq-8 mask, GROUPED BY PAIR DOMAIN. The C20–C48 maps put 6–9× more
/// surviving pairs than distinct pair domains
/// (`scripts/queue/proto_batch_amortization.py`), so the V_DD Cholesky is
/// done once per group and each distinct occupied is whitened once per
/// group ([`fit_domain_group`]); per-group work is held per worker and
/// dropped after its group. Results are written back into original-pair
/// order slots, so the assembled pair order is the ungrouped loop's.
///
/// Returns the per-pair slots THIS rank fitted (all of them without a rank
/// stripe) and the resident bytes they sit beside (strips, metric, held `W`).
fn fit_pair_slots(
    input: PairFitInput<'_>,
    stats: &mut DirectStats,
) -> Result<(Vec<crate::ragged_transport::Slot>, usize), FerricError> {
    use rayon::prelude::*;
    let PairFitInput {
        pre,
        dfbs,
        spaces,
        eps,
        scale,
        dcfg,
        path,
        gate,
        fock_screen,
    } = input;
    let (no, nv) = (spaces.no, spaces.nv);
    let fo: Vec<f64> = (0..no).map(|i| spaces.f_oo[(i, i)]).collect();
    let fv: Vec<f64> = (0..nv).map(|a| spaces.f_vv[(a, a)]).collect();
    let unique = kept_pairs(&pre.keep, no);
    let mine = pre.stripe.as_ref().map(|s| s.mine.as_slice());
    let kappa_eps: Option<f64> = match dcfg.virt_schwarz_kappa {
        Some(k) if eps > 0.0 => Some(k * eps),
        _ => None,
    };
    if kappa_eps.is_some() && fock_screen.is_some() {
        // The κ candidate screen bounds the BARE integrals (the Eq-8 mask);
        // the Fock-dressed gate can keep a virtual it drops, and its Fock
        // sums would run over the screened set only (book ch. 13,
        // §amplitude-lmp2-fock-screen-candidates).
        return Err(FerricError::General(
            "lmp2_direct: virt_schwarz_kappa is a Schwarz bound on the plain Eq-8 mask and \
             can drop virtuals the Fock screen keeps; set fock_screen = false to use it, \
             or leave virt_schwarz_kappa unset"
                .into(),
        ));
    }
    let (qvirt, own_w) = match kappa_eps {
        None => (None, None),
        Some(_) => schwarz_screen(pre, dfbs, no, scale, dcfg, path, stats)?,
    };
    if qvirt.is_some() && path.triangular() {
        stats.n_trsm += no;
    }
    let pair_cands = pair_candidates(
        &unique,
        &pre.virt_dom,
        &pre.strips,
        qvirt.as_ref(),
        kappa_eps,
        mine,
    );
    for c in &pair_cands {
        stats.virt_cand_mean += c.len() as f64;
        stats.virt_cand_max = stats.virt_cand_max.max(c.len());
    }
    let n_fitted = mine.map_or(unique.len(), |m| m.iter().filter(|&&b| b).count());
    if n_fitted > 0 {
        stats.virt_cand_mean /= n_fitted as f64;
    }

    let t_skip = Instant::now();
    stats.pair_skip = pre.pair_skip;
    // The proof's tables and workers must fit what the budget leaves beside
    // the resident set; if not, every pair is fitted (the result of `Off`).
    let proof_fits = gate.is_none_or(|g| {
        let held = stats.memory.strips_bytes
            + stats.memory.metric_bytes
            + own_w.iter().flatten().map(|w| 8 * w.len()).sum::<usize>();
        held + pair_proof_bytes(pre, dfbs.shell_dims(), unique.len()) <= g.budget_bytes
    });
    if !proof_fits && pre.pair_skip.evaluates() && eps > 0.0 {
        eprintln!(
            "lmp2_direct: pair-skip proof does not fit the memory budget beside the strips; \
             fitting every pair"
        );
    }
    let proof = match path.triangular() && proof_fits {
        true => prove_empty_pairs(
            pre,
            dfbs,
            spaces,
            &unique,
            mine,
            SkipGate {
                scale,
                eps,
                fock_screen,
                fo: &fo,
                fv: &fv,
            },
            dcfg.aux_radius_bohr,
        )
        .unwrap_or_else(|e| {
            // The skip is optional and rank-local: an inconclusive proof
            // fits every pair (the result `Off` gives), so this rank still
            // reaches the gather collectives the other ranks wait in.
            eprintln!("lmp2_direct: pair-skip proof inconclusive ({e}); fitting every pair");
            None
        }),
        false => None,
    };
    if let Some(p) = &proof {
        stats.n_pair_skip_bounded = p.g_bound.iter().filter(|g| g.is_finite()).count();
        stats.n_pairs_proved_empty = p.empty.iter().filter(|&&e| e).count();
    }
    let skip: Option<&[bool]> = proof
        .as_ref()
        .filter(|_| pre.pair_skip == PairSkip::On)
        .map(|p| p.empty.as_slice());
    let audit_out =
        (pre.pair_skip == PairSkip::Audit && proof.is_some()).then(|| Mutex::new(Vec::new()));
    stats.t_pair_skip_s = t_skip.elapsed().as_secs_f64();

    let pair_dom = pair_domains(&unique, &pre.aux_dom);
    let own_w: Option<Vec<Option<GroupW>>> = own_w.map(|w| {
        trim_own_w(
            w,
            &unique,
            &pair_dom,
            &pair_cands,
            &pre.aux_dom,
            &pre.strips,
        )
    });
    stats.own_w_bytes_held = own_w
        .iter()
        .flatten()
        .flatten()
        .map(|g| 8 * g.w.len())
        .sum();
    let group_list = own_groups(group_by_domain(pair_dom), mine);
    let ctx = PairFitCtx {
        unique: &unique,
        pair_cands: &pair_cands,
        strips: &pre.strips,
        dims_df: dfbs.shell_dims(),
        metric_blocks: &pre.metric_blocks,
        f_vv: &spaces.f_vv,
        fo: &fo,
        fv: &fv,
        eps,
        scale,
        fock_screen,
        nv,
        aux_radius_bohr: dcfg.aux_radius_bohr,
        dfbs,
        aux_dom: &pre.aux_dom,
        own_w: own_w.as_deref(),
        skip,
        audit: audit_out.as_ref(),
        #[cfg(test)]
        w_mutant: match path {
            FitPath::MutantW(first, rel) => Some((first, rel)),
            _ => None,
        },
    };
    let base_bytes = stats.memory.strips_bytes + stats.memory.metric_bytes + stats.own_w_bytes_held;
    // The screen is inert at ε = 0 (`pair_blocks_screened` skips it there).
    let screen_on = fock_screen.is_some() && eps > 0.0;
    let workers = rayon::current_num_threads().max(1);
    let scratch: Vec<usize> = group_list
        .iter()
        .map(|(dsh, pxs)| group_scratch_bound(dfbs, dsh, pxs, &unique, &pair_cands, screen_on))
        .collect();
    let blocks: Vec<usize> = group_list
        .iter()
        .map(|(_, pxs)| group_block_bound(pxs, &unique, &pair_cands, nv))
        .collect();
    stats.group_scratch_bound_bytes = scratch.iter().copied().max().unwrap_or(0);
    let charge = gate.map(|g| BlockCharge {
        budget_bytes: g.budget_bytes,
        base_bytes,
        est: stats.memory,
        scratch: AtomicUsize::new(0),
        ragged: AtomicUsize::new(0),
        vectors: AtomicUsize::new(0),
        tripped: Mutex::new(None),
    });
    let (in_flight, in_flight_peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let run_group = |(dsh, pxs): &(Vec<usize>, Vec<usize>)| -> GroupResult {
        if charge.as_ref().is_some_and(BlockCharge::is_tripped) {
            return Ok((Vec::new(), GroupCounts::default()));
        }
        if skip.is_some_and(|sk| pxs.iter().all(|&px| sk[px])) {
            // every pair proved empty: no Cholesky, no whitening, no fit
            let out = pxs.iter().map(|&px| (px, Vec::new())).collect();
            let counts = GroupCounts {
                skipped: true,
                ..GroupCounts::default()
            };
            return Ok((out, counts));
        }
        let now = in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        in_flight_peak.fetch_max(now, Ordering::Relaxed);
        let out = match path {
            FitPath::Cholesky => fit_domain_group(&ctx, dsh, pxs),
            #[cfg(test)]
            FitPath::MutantW(..) => fit_domain_group(&ctx, dsh, pxs),
            #[cfg(test)]
            FitPath::InverseOracle => {
                inverse_oracle::fit_domain_group_inverse_oracle(&ctx, dsh, pxs)
                    .map(|s| (s, GroupCounts::default()))
            }
        };
        in_flight.fetch_sub(1, Ordering::Relaxed);
        let out = out?;
        if let Some(c) = &charge {
            c.charge(&out.0);
        }
        Ok(out)
    };
    // Groups run in waves sized against what the budget has left beside the
    // resident set and the blocks built so far (exact bytes), charging the
    // wave's own blocks at their candidate bound and `min(len, workers)`
    // copies of its largest group scratch. The whole system's candidate
    // bound is never reserved up front: it is loose at ε > 0 (the mask drops
    // rows and columns), and reserving it serialised runs whose blocks fit.
    // Groups are independent and write their own slots, so the schedule
    // changes the order of work, never a block.
    let (group_out, concurrent, n_waves) = match &charge {
        None => (group_list.par_iter().map(&run_group).collect(), workers, 1),
        Some(c) => {
            let mut out: Vec<GroupResult> = Vec::with_capacity(group_list.len());
            let (mut start, mut concurrent, mut n_waves) = (0usize, 0usize, 0usize);
            let mut limited = false;
            while start < group_list.len() && !c.is_tripped() {
                let held = base_bytes + c.ragged.load(Ordering::Relaxed);
                let avail = c.budget_bytes.saturating_sub(held);
                let wave = plan_group_wave(start, &scratch, &blocks, workers, avail);
                // the widest wave the budget let run
                concurrent = concurrent.max((wave.end - start).min(workers));
                limited |= wave.budget_limited;
                c.scratch.store(wave.scratch, Ordering::Relaxed);
                let wave_groups = &group_list[start..wave.end];
                out.extend(wave_groups.par_iter().map(&run_group).collect::<Vec<_>>());
                n_waves += 1;
                start = wave.end;
            }
            if limited && concurrent == 1 && workers > 1 {
                ferric_core::memory::warn_width_collapsed(
                    "LMP2 pair-fit groups",
                    1,
                    workers,
                    c.budget_bytes.saturating_sub(base_bytes),
                    "stage-5 domain groups ran one at a time beside the pair blocks (same \
                     blocks, serial)",
                );
            }
            (out, concurrent, n_waves)
        }
    };
    stats.group_concurrency = concurrent;
    stats.n_group_waves = n_waves;
    stats.group_in_flight_peak = in_flight_peak.into_inner();
    if let Some(msg) = charge.and_then(|c| c.tripped.into_inner().ok().flatten()) {
        return Err(FerricError::MemoryExhausted(msg));
    }
    let slots = slots_from_groups(unique.len(), group_out, stats)?;
    stats.n_pairs_empty = slots
        .iter()
        .enumerate()
        .filter(|(px, slot)| {
            !skip.is_some_and(|sk| sk[*px]) && slot.as_ref().is_some_and(|b| b.is_empty())
        })
        .count();
    if let (Some(p), Some(a)) = (&proof, audit_out) {
        let rows = a.into_inner().unwrap_or_default();
        audit_pair_skip(p, &rows, &slots, &unique, spaces, stats);
    }
    Ok((slots, base_bytes))
}

/// [`PairSkip::Audit`]: compare every bounded fitted pair with its bound
/// (`DirectStats::pair_skip_audit_*`); with `FERRIC_LMP2_PAIR_SKIP_DUMP` set,
/// one `PAIRSKIP i j R g_bound gate_bound max|g| max_gate kept proved` line
/// per fitted pair on stderr (the measurement's raw data).
fn audit_pair_skip(
    p: &SkipProof,
    rows: &[(usize, f64, f64)],
    slots: &[crate::ragged_transport::Slot],
    unique: &[(usize, usize)],
    spaces: &LocalizedSpaces,
    stats: &mut DirectStats,
) {
    let dump = std::env::var_os("FERRIC_LMP2_PAIR_SKIP_DUMP").is_some();
    let (mut ratio, mut gate_ratio) = (0.0f64, 0.0f64);
    let (mut n, mut violations) = (0usize, 0usize);
    for &(px, gmax, gate_max) in rows {
        let kept = slots[px].as_ref().is_some_and(|b| !b.is_empty());
        let gb = p.g_bound[px];
        let gate_b = crate::lmp2_pair_skip::gate_bound(gb, p.fock);
        if gb.is_finite() {
            n += 1;
            ratio = ratio.max(gmax / gb);
            gate_ratio = gate_ratio.max(gate_max / gate_b);
        }
        if p.empty[px] && kept {
            violations += 1;
        }
        if dump {
            let (i, j) = unique[px];
            let r: f64 = (0..3)
                .map(|x| (spaces.occ_centers[(i, x)] - spaces.occ_centers[(j, x)]).powi(2))
                .sum::<f64>()
                .sqrt();
            eprintln!(
                "PAIRSKIP {i} {j} {r:.6} {gb:.6e} {gate_b:.6e} {gmax:.6e} {gate_max:.6e} {} {}",
                u8::from(kept),
                u8::from(p.empty[px])
            );
        }
    }
    stats.pair_skip_audit_n = n;
    stats.pair_skip_audit_max_ratio = ratio;
    stats.pair_skip_audit_gate_max_ratio = gate_ratio;
    stats.pair_skip_audit_violations = violations;
}

/// Stage 5 after the fits: the slots become the ragged system. Under a rank
/// stripe the ranks first exchange block skeletons (index lists and masks),
/// and the system comes back as this rank's share
/// ([`crate::ragged::RaggedShare`]): its own blocks whole, the others
/// hollow. `beside_bytes` is what is still resident beside this rank's own
/// blocks; with a gate, the exchange buffer must fit in what the budget has
/// left.
fn gather_ragged(
    mut slots: Vec<crate::ragged_transport::Slot>,
    stripe: Option<ferric_core::parallel::ParallelContext>,
    gate: Option<SolveGate>,
    beside_bytes: usize,
    t0: Instant,
    stats: &mut DirectStats,
) -> Result<Ragged, FerricError> {
    let owned_slots: Vec<bool> = slots.iter().map(Option::is_some).collect();
    if let Some(c) = stripe {
        let avail = gate.map(|g| {
            let own: usize = slots
                .iter()
                .flatten()
                .flatten()
                .map(PairBlock::heap_bytes)
                .sum();
            g.budget_bytes.saturating_sub(beside_bytes + own)
        });
        crate::ragged_transport::gather_slots(&c, &mut slots, avail)?;
    }
    let ragged = ragged_from_slots(slots, stripe.map(|c| (c, owned_slots.as_slice())))?;
    stats.t_pairs_s = t0.elapsed().as_secs_f64();
    mem_trace(
        "pairs",
        &[
            ("ragged", ragged.heap_bytes()),
            ("group_scratch_max", stats.group_scratch_bytes_max),
            ("group_scratch_bound", stats.group_scratch_bound_bytes),
            ("own_w_held", stats.own_w_bytes_held),
        ],
    );
    Ok(ragged)
}

/// Stage 5 on borrowed stage-1–4 products (tests fit the same `pre` more
/// than once): [`fit_pair_slots`] then [`gather_ragged`], with the strips
/// and the metric still resident during the exchange. The driver
/// ([`assemble_gated`]) drops them first.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fit_pairs(
    pre: &StripsAndMetric,
    dfbs: &PreparedBasis,
    spaces: &LocalizedSpaces,
    eps: f64,
    scale: f64,
    dcfg: &DirectConfig,
    path: FitPath,
    gate: Option<SolveGate>,
    fock_screen: Option<f64>,
    stats: &mut DirectStats,
) -> Result<Ragged, FerricError> {
    let t0 = Instant::now();
    let input = PairFitInput {
        pre,
        dfbs,
        spaces,
        eps,
        scale,
        dcfg,
        path,
        gate,
        fock_screen,
    };
    let (slots, base_bytes) = fit_pair_slots(input, stats)?;
    let ctx = pre.stripe.as_ref().map(|s| s.ctx);
    gather_ragged(slots, ctx, gate, base_bytes, t0, stats)
}

/// Pairs grouped by their pair domain D_ij (shell list), groups ordered by
/// their first pair index (so the group order is deterministic).
fn group_by_domain(pair_dom: Vec<Vec<usize>>) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut dom_groups: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for (px, dsh) in pair_dom.into_iter().enumerate() {
        dom_groups.entry(dsh).or_default().push(px);
    }
    let mut group_list: Vec<(Vec<usize>, Vec<usize>)> = dom_groups.into_iter().collect();
    group_list.sort_unstable_by_key(|(_, pxs)| pxs[0]);
    group_list
}

/// Group outputs → the ragged pair list in original-pair order (an unfilled
/// slot is a grouping bug), summing the per-group counters into `stats`.
/// The domain groups this rank fits: every group without a stripe, else the
/// groups of its own pairs ([`rank_stripe`] gives a whole group to one rank,
/// so a group's first pair decides).
fn own_groups(
    group_list: Vec<(Vec<usize>, Vec<usize>)>,
    mine: Option<&[bool]>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    match mine {
        None => group_list,
        Some(m) => group_list
            .into_iter()
            .filter(|(_, pxs)| m[pxs[0]])
            .collect(),
    }
}

/// Per-unique-pair slots filled from the groups this rank fitted, with the
/// group counters summed into `stats`. A slot of a group another rank owns
/// stays `None`.
fn slots_from_groups(
    n_unique: usize,
    group_out: Vec<Result<(SlotOut, GroupCounts), FerricError>>,
    stats: &mut DirectStats,
) -> Result<Vec<crate::ragged_transport::Slot>, FerricError> {
    stats.n_pair_groups = group_out.len();
    stats.n_pair_groups_skipped = 0;
    let mut n_occ = 0usize;
    let mut slots: Vec<crate::ragged_transport::Slot> = (0..n_unique).map(|_| None).collect();
    for r in group_out {
        let (slot_out, counts) = r?;
        if counts.skipped {
            stats.n_pair_groups -= 1;
            stats.n_pair_groups_skipped += 1;
        }
        n_occ += counts.n_occ;
        stats.n_trsm += counts.n_trsm;
        stats.n_w_reused += counts.n_reused;
        stats.group_scratch_bytes_max = stats.group_scratch_bytes_max.max(counts.scratch_bytes);
        for (acc, t) in stats.pair_fit_thread_s.iter_mut().zip(counts.t) {
            *acc += t;
        }
        let wall: f64 = counts.t.iter().sum();
        stats.group_work_max_s = stats.group_work_max_s.max(wall);
        stats.group_work_mean_s += wall;
        for (px, blocks) in slot_out {
            slots[px] = Some(blocks);
        }
    }
    if stats.n_pair_groups > 0 {
        stats.group_work_mean_s /= stats.n_pair_groups as f64;
        stats.n_distinct_occ_per_group = n_occ as f64 / stats.n_pair_groups as f64;
    }
    Ok(slots)
}

/// The ragged system of fully filled slots, in slot order. An empty slot is
/// an error: a grouping bug, or a striped run whose skeletons were not
/// gathered (a rank-pinned context with no MPI world behind it). `share`
/// names the MPI context and the slots this rank fitted; the system is then
/// that rank's share.
fn ragged_from_slots(
    slots: Vec<crate::ragged_transport::Slot>,
    share: Option<(ferric_core::parallel::ParallelContext, &[bool])>,
) -> Result<Ragged, FerricError> {
    let mut pairs: Vec<PairBlock> = Vec::new();
    let mut owned: Vec<bool> = Vec::new();
    for (px, s) in slots.into_iter().enumerate() {
        let blocks = s.ok_or_else(|| {
            FerricError::General(format!(
                "lmp2_direct: pair slot {px} never filled (grouping bug, or a rank stripe \
                 without an MPI world to gather over)"
            ))
        })?;
        owned.resize(owned.len() + blocks.len(), share.is_none_or(|(_, o)| o[px]));
        pairs.extend(blocks);
    }
    let mut ragged = Ragged::from_pairs(pairs);
    ragged.share = share.map(|(ctx, _)| crate::ragged::RaggedShare { ctx, owned });
    Ok(ragged)
}

/// Compact whitened occ-occ Gram `(ik|jl)` for the LinLCCD hh ladder —
/// the occ-occ strip extension of the integral-direct path (the deferral
/// note's PSD-safe block: WIKI-APPEND-drpa-direct.md).
///
/// Storage is over the gate-surviving unordered `(i,k)` columns only
/// (`col_of` maps `i*no+k` to a compact column, `usize::MAX` = gated).
/// A gated column is EXACTLY ZERO in the whitened Boo, so the Gram stays
/// a Gram of one consistent whitened B — the CG-license structure is
/// unchanged; the dropped-coupling energy error is a measured quantity
/// (sub-dominance tests), never an assumed one.
#[derive(Debug, Clone)]
pub struct OoGram {
    g: Array2<f64>,
    col_of: Vec<usize>,
    no: usize,
}

impl OoGram {
    /// Wrap a DENSE `(no², no²)` `(ik|jl)` matrix (the global path's
    /// `transform_3center_oo` + whitened Gram) — identical lookups, so the
    /// shared solver reproduces the pre-refactor energies bit for bit.
    pub fn dense(g: Array2<f64>, no: usize) -> Self {
        let col_of = (0..no * no).collect();
        Self { g, col_of, no }
    }

    /// hh coefficient `(ik|jl)`; 0.0 when either column is gated out.
    #[inline]
    pub fn coeff(&self, i: usize, k: usize, j: usize, l: usize) -> f64 {
        let a = self.col_of[i * self.no + k];
        if a == usize::MAX {
            return 0.0;
        }
        let b = self.col_of[j * self.no + l];
        if b == usize::MAX {
            return 0.0;
        }
        self.g[(a, b)]
    }

    /// Surviving columns (the honesty counter for the gate restriction).
    pub fn ncols(&self) -> usize {
        self.g.nrows()
    }
}

/// Occ-occ half-transform of one aux slab into the surviving columns of
/// `boo`: per aux function, `M = C_occᵀ · T_p · C_occ`, then the compact
/// `(i,k)` entries are read out.
fn boo_transform_slab(
    boo: &mut Array2<f64>,
    t_slab: &Array3<f64>,
    p_funcs: &[usize],
    c_m: &Array2<f64>,
    cols: &[(usize, usize)],
) {
    for (pl, &gp) in p_funcs.iter().enumerate() {
        let tv = t_slab.index_axis(Axis(0), pl);
        let t1 = c_m.t().dot(&tv); // (no, nuf)
        let m = t1.dot(c_m); // (no, no)
        for (cx, &(i, k)) in cols.iter().enumerate() {
            boo[(gp, cx)] = m[(i, k)];
        }
    }
}

/// Integral-direct occ-occ RI Gram for the hh ladder: evaluates `(P|μν)`
/// only over occupied AO supports (never the full AO `eri3_tensor`),
/// half-transforms into unwhitened columns `(P|ik)` for the gate-surviving
/// unordered pairs, then whitens with the GLOBAL metric — full aux rows,
/// so the resulting `(ik|jl)` equals the global path's object exactly at
/// `ao_tail = 0` (reassociation floor), and the hh matvec stays the Gram
/// of ONE consistent whitened B (the PSD license of proof notebook 13).
///
/// The naux² metric IS formed here (like the global path); what is never
/// formed is the `(naux, nao²)` AO tensor and the `(naux, no·nv)` B.
/// Serial integral pass — the occ-occ volume is ~(occ-support)² per aux
/// shell, small next to the main ov strip pass; parallelize only if a
/// measurement says it matters.
#[allow(clippy::too_many_arguments)]
pub fn assemble_boo_direct(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    spaces: &LocalizedSpaces,
    eps: f64,
    pair_gate_cal: Option<f64>,
    dcfg: &DirectConfig,
) -> Result<OoGram, FerricError> {
    let no = spaces.no;
    let naux = dfbs.nbasis();
    let nao = obs.nbasis();
    let (keep, _) = match pair_gate_cal {
        Some(cal) => pair_gate_keep(&spaces.occ_centers, &spaces.occ_spreads, no, eps, cal),
        None => (vec![true; no * no], 0),
    };
    let mut col_of = vec![usize::MAX; no * no];
    let mut cols: Vec<(usize, usize)> = Vec::new();
    for i in 0..no {
        for k in i..no {
            if keep[i * no + k] {
                col_of[i * no + k] = cols.len();
                col_of[k * no + i] = cols.len();
                cols.push((i, k));
            }
        }
    }
    // occupied AO supports (ao_tail ≡ zeroing small coefficients, the same
    // rule and trivial limit as the ov strips)
    let occ_supp: Vec<Vec<usize>> = (0..no)
        .map(|i| supp_shells(obs, spaces.c_locc.column(i), dcfg.ao_tail))
        .collect();
    let u_shells = sorted_union(occ_supp.iter().map(|s| s.iter().copied()));
    let u_funcs = shell_funcs(obs, &u_shells);
    let nuf = u_funcs.len();
    let mut u_pos = vec![usize::MAX; nao];
    for (k, &f) in u_funcs.iter().enumerate() {
        u_pos[f] = k;
    }
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();
    // support-masked occupied coefficients on the compact AO axis
    let mut c_m = Array2::<f64>::zeros((nuf, no));
    for (i, supp) in occ_supp.iter().enumerate() {
        for &sh in supp {
            for f in 0..dims_obs[sh] {
                let g = offs_obs[sh] + f;
                c_m[(u_pos[g], i)] = spaces.c_locc[(g, i)];
            }
        }
    }
    // slab ALL aux shells under the scratch budget (full aux rows — the
    // global whitening below is exact, not domain-truncated)
    let mut boo = Array2::<f64>::zeros((naux, cols.len()));
    let mut eng = Engine::new_3center(op, obs, dfbs, 1e-14)?;
    let per_func = nuf.max(1) * nuf.max(1) * 8;
    let max_funcs = (dcfg.scratch_budget_bytes / per_func).max(1);
    let nsh_df = dfbs.nshells();
    let mut slab_start = 0usize;
    while slab_start < nsh_df {
        let mut slab_end = slab_start;
        let mut nfuncs = 0usize;
        while slab_end < nsh_df {
            let add = dims_df[slab_end];
            if nfuncs > 0 && nfuncs + add > max_funcs {
                break;
            }
            nfuncs += add;
            slab_end += 1;
        }
        let slab: Vec<usize> = (slab_start..slab_end).collect();
        let p_funcs = shell_funcs(dfbs, &slab);
        let mut p_local = vec![usize::MAX; naux];
        for (k, &f) in p_funcs.iter().enumerate() {
            p_local[f] = k;
        }
        let mut t_slab = Array3::<f64>::zeros((p_funcs.len(), nuf, nuf));
        for &sp in &slab {
            for (ux, &ua) in u_shells.iter().enumerate() {
                for &ub in &u_shells[..=ux] {
                    if let Some(block) = eng.compute_eri3(obs, dfbs, sp, ua, ub) {
                        let (n1, n2) = (dims_obs[ua], dims_obs[ub]);
                        for p in 0..dims_df[sp] {
                            let pl = p_local[offs_df[sp] + p];
                            for fa in 0..n1 {
                                let ga = offs_obs[ua] + fa;
                                for fb in 0..n2 {
                                    let gb = offs_obs[ub] + fb;
                                    let val = block[(p * n1 + fa) * n2 + fb];
                                    t_slab[(pl, u_pos[ga], u_pos[gb])] = val;
                                    t_slab[(pl, u_pos[gb], u_pos[ga])] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
        boo_transform_slab(&mut boo, &t_slab, &p_funcs, &c_m, &cols);
        slab_start = slab_end;
    }
    // GLOBAL whitening: (ik|jl) = Boõᵀ Boõ, one consistent whitened B —
    // gated columns are exactly zero, so this stays an exact Gram.
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let vis = crate::rimp2::metric_inverse_sqrt(&v2c, op)?;
    let boo_t = vis.dot(&boo);
    let g = boo_t.t().dot(&boo_t);
    Ok(OoGram { g, col_of, no })
}

/// Per-pair FITTED pp-ladder factors for the integral-direct LinLCCD Full
/// tier: the pair's block, rows (a,c) ∈ Da², cols (b,d) ∈ Db², is the
/// domain-local same-kernel fit
/// `(ac|bd)_fit = Σ_{P,Q ∈ D_ij} (ac|P) [V_DD⁻¹]_PQ (Q|bd)`,
/// held factored — `m = a_rows · gb` is formed per solver iteration (the
/// same per-iteration GEMM shape as the licensed whitened-Bvv gather).
///
/// PSD LICENSE (empirical, measured — NOT an algebraic Gram identity):
/// one V_DD per pair domain means these blocks are NOT submatrices of one
/// consistent whitened Gram; the ragged-CG SPD contract for them was
/// MEASURED, not derived — `scripts/queue/proto_linlccd_pp_psd.py`
/// (anchors A1-A4, mutation arms M1-M2, radius/eps/system sweeps recorded
/// in WIKI-APPEND-linlccd-direct.md / wiki amplitude-threshold-drpa).
/// The measurement also showed the LICENSED (global whitened Gram) block
/// is itself slightly indefinite under erfc — the operative license for
/// both constructions is bounded indefiniteness ≪ the Fock denominator
/// floor (per-arm margins in the note). Do not extend this construction
/// to new operators or regimes without re-running that measurement.
#[derive(Debug, Clone)]
pub struct PpFitted {
    /// `(|Da|², d)` — rows (a,c), unwhitened `(ac|P ∈ D_ij)`.
    pub a_rows: Array2<f64>,
    /// `(d, |Db|²)` — `V_DD⁻¹ · Abᵀ`, cols (b,d).
    pub gb: Array2<f64>,
}

/// Gathered V_DD for one aux-shell domain, evaluated shell pair by shell
/// pair (no global metric).
fn vdd_for_domain(
    dfbs: &PreparedBasis,
    op: Operator,
    dsh: &[usize],
    d: usize,
) -> Result<Array2<f64>, FerricError> {
    let dims_df = dfbs.shell_dims();
    let mut eng = Engine::new_2center(op, dfbs, 1e-14)?;
    let mut loc_off = Vec::with_capacity(dsh.len());
    let mut acc = 0usize;
    for &sh in dsh {
        loc_off.push(acc);
        acc += dims_df[sh];
    }
    let mut vdd = Array2::<f64>::zeros((d, d));
    for (bx, &sp) in dsh.iter().enumerate() {
        for (by, &sq) in dsh.iter().enumerate().take(bx + 1) {
            let vals = eng.compute_eri2(dfbs, sp, sq);
            for p in 0..dims_df[sp] {
                for q in 0..dims_df[sq] {
                    let v = vals[p * dims_df[sq] + q];
                    vdd[(loc_off[bx] + p, loc_off[by] + q)] = v;
                    vdd[(loc_off[by] + q, loc_off[bx] + p)] = v;
                }
            }
        }
    }
    Ok(vdd)
}

/// One domain group's `(ac|P)` tensor over the group's union virtuals:
/// `avv[(pl, u1, u2)]` with `pl` local to `dfuncs`, evaluated slab by slab
/// over the domain's aux shells under the scratch budget share.
#[allow(clippy::too_many_arguments)]
fn avv_for_group(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    spaces: &LocalizedSpaces,
    dsh: &[usize],
    uvirt: &[usize],
    ao_tail: f64,
    budget_bytes: usize,
) -> Result<Array3<f64>, FerricError> {
    let nao = obs.nbasis();
    let naux = dfbs.nbasis();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();
    let nu = uvirt.len();
    // AO supports of the union virtuals (same ao_tail rule as the strips)
    let v_supp: Vec<Vec<usize>> = uvirt
        .iter()
        .map(|&a| supp_shells(obs, spaces.c_vloc.column(a), ao_tail))
        .collect();
    let v_shells = sorted_union(v_supp.iter().map(|s| s.iter().copied()));
    let v_funcs = shell_funcs(obs, &v_shells);
    let nvf = v_funcs.len();
    let mut v_pos = vec![usize::MAX; nao];
    for (k, &f) in v_funcs.iter().enumerate() {
        v_pos[f] = k;
    }
    let mut c_v = Array2::<f64>::zeros((nvf, nu));
    for (ux, (&a, supp)) in uvirt.iter().zip(&v_supp).enumerate() {
        for &sh in supp {
            for f in 0..dims_obs[sh] {
                let g = offs_obs[sh] + f;
                c_v[(v_pos[g], ux)] = spaces.c_vloc[(g, a)];
            }
        }
    }
    let dfuncs = shell_funcs(dfbs, dsh);
    let d = dfuncs.len();
    let mut p_loc = vec![usize::MAX; naux];
    for (k, &f) in dfuncs.iter().enumerate() {
        p_loc[f] = k;
    }
    let mut avv = Array3::<f64>::zeros((d, nu, nu));
    let mut eng = Engine::new_3center(op, obs, dfbs, 1e-14)?;
    let per_func = nvf.max(1) * nvf.max(1) * 8;
    let max_funcs = (budget_bytes / per_func).max(1);
    let mut slab_start = 0usize;
    while slab_start < dsh.len() {
        let mut slab_end = slab_start;
        let mut nfuncs = 0usize;
        while slab_end < dsh.len() {
            let add = dims_df[dsh[slab_end]];
            if nfuncs > 0 && nfuncs + add > max_funcs {
                break;
            }
            nfuncs += add;
            slab_end += 1;
        }
        let slab = &dsh[slab_start..slab_end];
        let mut sl_loc = vec![usize::MAX; naux];
        let mut nps = 0usize;
        for &sp in slab {
            for f in 0..dims_df[sp] {
                sl_loc[offs_df[sp] + f] = nps;
                nps += 1;
            }
        }
        let mut t_slab = Array3::<f64>::zeros((nps, nvf, nvf));
        for &sp in slab {
            for (ux, &ua) in v_shells.iter().enumerate() {
                for &ub in &v_shells[..=ux] {
                    if let Some(block) = eng.compute_eri3(obs, dfbs, sp, ua, ub) {
                        let (n1, n2) = (dims_obs[ua], dims_obs[ub]);
                        for p in 0..dims_df[sp] {
                            let pl = sl_loc[offs_df[sp] + p];
                            for fa in 0..n1 {
                                let ga = offs_obs[ua] + fa;
                                for fb in 0..n2 {
                                    let gb = offs_obs[ub] + fb;
                                    let val = block[(p * n1 + fa) * n2 + fb];
                                    t_slab[(pl, v_pos[ga], v_pos[gb])] = val;
                                    t_slab[(pl, v_pos[gb], v_pos[ga])] = val;
                                }
                            }
                        }
                    }
                }
            }
            for f in 0..dims_df[sp] {
                let gp = offs_df[sp] + f;
                let tv = t_slab.index_axis(Axis(0), sl_loc[gp]);
                let m = c_v.t().dot(&tv).dot(&c_v); // (nu, nu)
                avv.index_axis_mut(Axis(0), p_loc[gp]).assign(&m);
            }
        }
        slab_start = slab_end;
    }
    Ok(avv)
}

/// Assemble the FITTED pp-ladder factors for every pair of an
/// already-assembled ragged system (pattern-derived Da/Db), grouped by
/// pair aux domain exactly like stage 5 of
/// [`assemble_ragged_direct_local`]. See [`PpFitted`] for the license.
pub fn assemble_pp_fitted_direct(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    spaces: &LocalizedSpaces,
    rg: &Ragged,
    dcfg: &DirectConfig,
) -> Result<Vec<PpFitted>, FerricError> {
    use ndarray_linalg::InverseC;
    use rayon::prelude::*;
    let no = spaces.no;
    let atom_xyz: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let occ_xyz: Vec<[f64; 3]> = (0..no)
        .map(|i| {
            [
                spaces.occ_centers[(i, 0)],
                spaces.occ_centers[(i, 1)],
                spaces.occ_centers[(i, 2)],
            ]
        })
        .collect();
    let df_shell_atom = dfbs.shell_to_atom();
    let r_aux2 = dcfg.aux_radius_bohr * dcfg.aux_radius_bohr;
    let aux_dom: Vec<Vec<usize>> = (0..no)
        .map(|i| {
            (0..dfbs.nshells())
                .filter(|&sp| dist2(atom_xyz[df_shell_atom[sp]], occ_xyz[i]) <= r_aux2)
                .collect()
        })
        .collect();
    let mut dom_groups: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for (px, pb) in rg.pairs.iter().enumerate() {
        let dsh = sorted_union([aux_dom[pb.i].iter().copied(), aux_dom[pb.j].iter().copied()]);
        dom_groups.entry(dsh).or_default().push(px);
    }
    let mut group_list: Vec<(Vec<usize>, Vec<usize>)> = dom_groups.into_iter().collect();
    group_list.sort_unstable_by_key(|(_, pxs)| pxs[0]);
    Engine::new_3center(op, obs, dfbs, 1e-14)?; // surface construction errors serially
    let n_workers = rayon::current_num_threads()
        .max(1)
        .min(group_list.len().max(1));
    let budget_share = (dcfg.scratch_budget_bytes / n_workers).max(1);
    let group_out: Vec<Result<Vec<(usize, PpFitted)>, FerricError>> = group_list
        .par_iter()
        .map(|(dsh, pxs)| {
            let (i0, j0) = (rg.pairs[pxs[0]].i, rg.pairs[pxs[0]].j);
            let dfuncs = shell_funcs(dfbs, dsh);
            let d = dfuncs.len();
            if d == 0 {
                return Err(FerricError::General(format!(
                    "pp_fitted_direct: empty aux domain for pair ({i0},{j0}) at radius {} Bohr",
                    dcfg.aux_radius_bohr
                )));
            }
            let vdd = vdd_for_domain(dfbs, op, dsh, d)?;
            let vdd_inv = vdd.invc().map_err(|e| {
                FerricError::General(format!(
                    "pp_fitted_direct V_DD Cholesky inverse (group of pair ({i0},{j0})): {e}"
                ))
            })?;
            let uvirt = sorted_union(pxs.iter().map(|&px| {
                let pb = &rg.pairs[px];
                pb.da
                    .iter()
                    .copied()
                    .chain(pb.db.iter().copied())
                    .collect::<Vec<_>>()
            }));
            let mut u_pos = vec![usize::MAX; spaces.nv];
            for (k, &a) in uvirt.iter().enumerate() {
                u_pos[a] = k;
            }
            let avv = avv_for_group(
                obs,
                dfbs,
                op,
                spaces,
                dsh,
                &uvirt,
                dcfg.ao_tail,
                budget_share,
            )?;
            let mut out = Vec::with_capacity(pxs.len());
            for &px in pxs {
                let pb = &rg.pairs[px];
                let gather = |idx: &[usize]| {
                    let n = idx.len();
                    let mut a = Array2::<f64>::zeros((n * n, d));
                    for (ra, &va) in idx.iter().enumerate() {
                        for (rc, &vc) in idx.iter().enumerate() {
                            for r in 0..d {
                                a[(ra * n + rc, r)] = avv[(r, u_pos[va], u_pos[vc])];
                            }
                        }
                    }
                    a
                };
                let a_rows = gather(&pb.da);
                let ab = gather(&pb.db);
                let gb = vdd_inv.dot(&ab.t());
                out.push((px, PpFitted { a_rows, gb }));
            }
            Ok(out)
        })
        .collect();
    let mut slots: Vec<Option<PpFitted>> = (0..rg.pairs.len()).map(|_| None).collect();
    for r in group_out {
        for (px, f) in r? {
            slots[px] = Some(f);
        }
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(px, s)| {
            s.ok_or_else(|| {
                FerricError::General(format!("pp_fitted_direct: pair slot {px} never filled"))
            })
        })
        .collect()
}

/// [`amplitude_lmp2_direct`] on the system `aux` was prepared on (see [`PreparedAux`]):
/// `rhf` is checked against that system first
/// ([`PreparedAux::check_reference`]).
/// `op` is the correlation operator (the system's Coulomb operator
/// unless the caller attenuates it).
/// The reference must be a closed-shell restricted (RHF / RKS) result
/// ([`PreparedAux::check_restricted_reference`]).
pub fn amplitude_lmp2_direct_on(
    aux: &PreparedAux<'_>,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
    dcfg: &DirectConfig,
) -> Result<(AmplitudeLmp2Result, DirectStats), FerricError> {
    let sys = aux.system();
    aux.check_restricted_reference(rhf)?;
    amplitude_lmp2_direct(
        sys.mol(),
        sys.basis(),
        sys.basis_set(),
        aux.basis(),
        op,
        rhf,
        cfg,
        dcfg,
    )
}

/// Integral-direct amplitude-threshold LMP2 driver — the direct-path
/// sibling of [`crate::lmp2_amplitude::amplitude_lmp2`]. `cfg.fit_radius_bohr`
/// and `cfg.aux_tail_frac` are ignored (the direct path's locality lives in
/// `dcfg`); everything else means the same thing.
///
/// `cfg.compute_reference` (default false) opts in to the canonical
/// `ri_mp2` reference, which DOES form the global `(naux, no·nv)` tensor
/// this path otherwise never builds — leave it off for a reduced-cost run.
#[allow(clippy::too_many_arguments)]
pub fn amplitude_lmp2_direct(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
    dcfg: &DirectConfig,
) -> Result<(AmplitudeLmp2Result, DirectStats), FerricError> {
    crate::lmp2_amplitude::require_hf_reference(rhf, "lmp2_direct")?;
    mem_trace("entry", &[]);
    let (vvhv, t_spaces_s, report) =
        checked_vvhv_with(mol, obs, obs_bs, rhf, "lmp2_direct", &cfg.virtual_space)?;
    mem_trace("vvhv", &[]);
    let (mut r, st) =
        amplitude_lmp2_direct_with_virtuals(mol, obs, dfbs, op, rhf, cfg, dcfg, &vvhv)?;
    r.timings.t_spaces_s = t_spaces_s;
    r.virtual_space = report;
    eprintln!("lmp2_direct: {}", r.virtual_space.log_line(&r.timings));
    Ok((r, st))
}

/// [`amplitude_lmp2_direct`] with a caller-supplied virtual space (the
/// mutation-test entry point, mirroring the global path).
#[allow(clippy::too_many_arguments)]
pub fn amplitude_lmp2_direct_with_virtuals(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
    dcfg: &DirectConfig,
    vvhv: &VvHv,
) -> Result<(AmplitudeLmp2Result, DirectStats), FerricError> {
    crate::lmp2_amplitude::require_hf_reference(rhf, "lmp2_direct")?;
    let t0 = Instant::now();
    let spaces =
        localized_spaces_lambda(mol, obs, rhf, cfg.frozen_core, vvhv, cfg.lambda_boys_spec())?;
    let t_loc_s = t0.elapsed().as_secs_f64();
    mem_trace("localized", &[]);
    let (no, nv) = (spaces.no, spaces.nv);
    let budget_bytes = crate::rimp2::eri3_budget_bytes(cfg.eri3_budget_bytes);
    let (rg, n_pairs_gated, mut dstats) = assemble_gated(
        mol,
        obs,
        dfbs,
        op,
        &spaces,
        cfg.eps,
        1.0,
        cfg.pair_gate_cal,
        dcfg,
        Some(SolveGate { budget_bytes }),
        cfg.fock_screen,
    )?;
    // The exact solve footprint, now that the blocks exist (the charge gate
    // already bounded it; this is the authority the solve runs under).
    let resident = crate::ragged::solve_resident_bytes(&rg, cfg.cg_max_iter);
    if resident > budget_bytes {
        let est = DirectMemoryEstimate {
            pair_block_bytes: rg.heap_bytes(),
            pcg_vector_bytes: crate::ragged::amplitude_vector_bytes(&rg),
            exact: true,
            ..dstats.memory
        };
        return Err(memory_refusal(
            &est,
            budget_bytes,
            "refused before the solve",
        ));
    }
    let t_assembly_s = t0.elapsed().as_secs_f64();
    dstats.t_loc_s = t_loc_s;

    let t0 = Instant::now();
    let e_ref = if cfg.compute_reference {
        ri_mp2(
            mol,
            obs,
            dfbs,
            op,
            rhf,
            &RiMp2Config {
                frozen_core: cfg.frozen_core,
                memory_budget_bytes: cfg.eri3_budget_bytes,
                ..Default::default()
            },
        )?
        .mp2_corr
    } else {
        f64::NAN
    };
    let t_reference_s = if cfg.compute_reference {
        t0.elapsed().as_secs_f64()
    } else {
        0.0
    };

    mem_trace(
        "pre-solve",
        &[
            ("ragged", rg.heap_bytes()),
            (
                "pcg_vectors",
                crate::ragged::PCG_RESIDENT_VECTORS * crate::ragged::amplitude_vector_bytes(&rg),
            ),
        ],
    );
    let t0 = Instant::now();
    let mut r = solve_and_count(
        &rg,
        &spaces.f_oo,
        &spaces.f_vv,
        rhf.energy,
        cfg,
        vvhv,
        no,
        nv,
        "lmp2_direct",
    )?;
    mem_trace("solved", &[]);
    r.e_corr_canonical_ri = e_ref;
    r.n_pairs_gated = n_pairs_gated;
    r.aux_dom_mean = dstats.strip_rows_mean;
    r.aux_dom_max = dstats.strip_rows_max;
    r.lambda_boys = spaces.lambda_boys;
    r.timings = StageTimings {
        t_spaces_s: 0.0,
        t_loc_s,
        t_assembly_s,
        t_solve_s: t0.elapsed().as_secs_f64(),
        t_reference_s,
    };
    Ok((r, dstats))
}

#[cfg(test)]
mod stage_b_tests {
    use super::*;
    use crate::lmp2_amplitude::checked_vvhv;
    use crate::lmp2_amplitude::localized_spaces;
    use ferric_core::parallel::ParallelContext;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    /// The pre-Stage-B metric pair set: a BTreeSet over every i's extended
    /// list — the oracle of the bitmap version.
    fn metric_need_oracle(aux_ext: &[Vec<usize>]) -> Vec<(usize, usize)> {
        let mut need = std::collections::BTreeSet::new();
        for shells in aux_ext {
            for (x, &sp) in shells.iter().enumerate() {
                for &sq in &shells[..=x] {
                    need.insert((sp, sq));
                }
            }
        }
        need.into_iter().collect()
    }

    /// B2 anchor: the bitmap pair list is exactly the BTreeSet oracle's
    /// (same pairs, same order, so the same `n_metric_shell_pairs` and the
    /// same evaluation order) on overlapping random sorted domains, with
    /// diagonal and boundary shells present.
    #[test]
    fn metric_bitmap_matches_btreeset_oracle() {
        let nsh = 37;
        let mut seed = 11u64;
        let mut rnd = |m: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize % m
        };
        let aux_ext: Vec<Vec<usize>> = (0..9)
            .map(|_| {
                let lo = rnd(nsh);
                let len = 1 + rnd(nsh - lo);
                let mut v: Vec<usize> = (lo..lo + len).filter(|_| rnd(4) > 0).collect();
                v.push(nsh - 1);
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        let want = metric_need_oracle(&aux_ext);
        assert_eq!(metric_need_pairs(&aux_ext, nsh), want);
        assert!(want.iter().any(|&(p, q)| p == q) && want.len() > 50);
    }

    /// The pre-B4 slab fill: every `u_shells²` pair re-tested per aux shell,
    /// in loop order (the Stage-A loop) — the oracle of the sorted list.
    #[allow(clippy::too_many_arguments)]
    fn fill_slab_unsorted_oracle(
        ctx: &BatchCtx<'_>,
        ax: &BatchAxes,
        _pairs: &[NeedPair],
        eng: &mut Engine,
        slab: &[usize],
        p_local: &[usize],
        t_slab: &mut Array3<f64>,
        _tm: &mut BatchTimers,
    ) -> (u64, u64) {
        let (obs, dfbs) = (ctx.obs, ctx.dfbs);
        let (dims_obs, offs_obs) = (obs.shell_dims(), obs.shell_offsets());
        let (dims_df, offs_df) = (dfbs.shell_dims(), dfbs.shell_offsets());
        let (mut n_triples, mut n_skipped) = (0u64, 0u64);
        for &sp in slab {
            for (ux, &ua) in ax.u_shells.iter().enumerate() {
                for &ub in &ax.u_shells[..=ux] {
                    let need_ab = ax.in_s[ua] && ax.in_n[ub];
                    let need_ba = ax.in_s[ub] && ax.in_n[ua];
                    if !(need_ab || need_ba) {
                        continue;
                    }
                    if let Some((q, qp_shell)) = ctx.qpair {
                        if qp_shell[sp] * q[(ua, ub)] < ctx.schwarz_skip {
                            n_skipped += 1;
                            continue;
                        }
                    }
                    n_triples += 1;
                    let Some(block) = eng.compute_eri3(obs, dfbs, sp, ua, ub) else {
                        continue;
                    };
                    let (n1, n2) = (dims_obs[ua], dims_obs[ub]);
                    for p in 0..dims_df[sp] {
                        let pl = p_local[offs_df[sp] + p];
                        for fa in 0..n1 {
                            let ga = offs_obs[ua] + fa;
                            for fb in 0..n2 {
                                let gb = offs_obs[ub] + fb;
                                let val = block[(p * n1 + fa) * n2 + fb];
                                if need_ab {
                                    t_slab[(pl, ax.s_pos[ga], ax.n_pos[gb])] = val;
                                }
                                if need_ba {
                                    t_slab[(pl, ax.s_pos[gb], ax.n_pos[ga])] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
        (n_triples, n_skipped)
    }

    struct AuxCase {
        mol: Molecule,
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        spaces: LocalizedSpaces,
    }

    fn aux_case(nc: usize) -> AuxCase {
        let mol = Molecule::load_xyz(&format!(
            "{}/../../testdata/molecules/alkane_{nc}.xyz",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let obs_bs = ferric_core::basis::bundled("6-31g").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
        let cfg = RhfConfig::default().with_energy_conv(1e-10);
        let rhf = solve_rhf(
            &ParallelContext::default(),
            &mol,
            &obs,
            Operator::coulomb(),
            &bounds,
            &cfg,
        )
        .unwrap();
        let (vvhv, _) = checked_vvhv(&mol, &obs, &obs_bs, &rhf, "test").unwrap();
        let spaces = localized_spaces(&mol, &obs, &rhf, nc, &vvhv).unwrap();
        AuxCase {
            mol,
            obs,
            dfbs,
            spaces,
        }
    }

    fn aux_strips(
        c: &AuxCase,
        op: Operator,
        cal: f64,
        dcfg: &DirectConfig,
    ) -> (Vec<Strip>, DirectStats) {
        let mut st = DirectStats::default();
        let pre = strips_and_metric(
            &c.mol,
            &c.obs,
            &c.dfbs,
            op,
            &c.spaces,
            1e-4,
            Some(cal),
            dcfg,
            None,
            &mut st,
        )
        .unwrap();
        (pre.strips, st)
    }

    fn production(pass: Eri3Pass, batch_merge: usize) -> DirectConfig {
        DirectConfig {
            aux_radius_bohr: 10.0,
            virt_radius_bohr: Some(12.0),
            ao_tail: 1e-3,
            schwarz_skip: 1e-5,
            batch_merge,
            eri3_pass: pass,
            ..Default::default()
        }
    }

    /// Largest strip-element difference over the largest element.
    fn strip_rel_diff(a: &[Strip], b: &[Strip]) -> f64 {
        let (mut d, mut m) = (0f64, 0f64);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert_eq!((&x.rows, &x.cols), (&y.rows, &y.cols));
            for (u, v) in x.b.iter().zip(y.b.iter()) {
                d = d.max((u - v).abs());
                m = m.max(u.abs());
            }
        }
        d / m
    }

    /// Independent construction: the aux-major strips equal the batch pass's
    /// element by element, at production maps, for both operators and both
    /// sizes, while evaluating strictly fewer triples than merged batches.
    #[test]
    fn aux_major_strips_match_batches() {
        for nc in [4, 8] {
            let c = aux_case(nc);
            for (op, cal) in [(Operator::coulomb(), 0.7), (Operator::erfc(1.0), 0.02)] {
                let (a, sa) = aux_strips(&c, op, cal, &production(Eri3Pass::AuxMajor, 1));
                let (b, sb) = aux_strips(&c, op, cal, &production(Eri3Pass::Batches, 4));
                let rel = strip_rel_diff(&a, &b);
                eprintln!(
                    "C{nc} {:?}: rel {rel:.2e}, triples aux {} batches {}",
                    op.kind, sa.n_eri3_shell_triples, sb.n_eri3_shell_triples
                );
                assert!(rel <= 1e-13, "C{nc} strips differ: {rel:e}");
                assert!(sa.n_eri3_shell_triples < sb.n_eri3_shell_triples);
                assert_eq!(sa.n_batches, sa.n_eri3_slabs);
            }
        }
    }

    /// The results depend on neither the thread count nor the budget: both set
    /// concurrency only, so the strips are bitwise equal. A budget of one byte
    /// must drop to one task at a time; an ample one must not.
    #[test]
    fn aux_major_strips_ignore_threads_and_budget() {
        let c = aux_case(4);
        let op = Operator::coulomb();
        let run = |threads: usize, budget: usize| {
            let dcfg = DirectConfig {
                scratch_budget_bytes: budget,
                ..production(Eri3Pass::AuxMajor, 1)
            };
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| aux_strips(&c, op, 0.7, &dcfg))
        };
        let ample = 1usize << 32;
        let (base, st) = run(4, ample);
        assert_eq!(st.aux_major_concurrency, 4);
        assert!(st.aux_major_task_bytes_max > 0 && st.aux_major_coeff_bytes > 0);
        for (threads, budget, want) in [(1, ample, 1), (4, 1, 1), (3, ample, 3)] {
            let (s, st) = run(threads, budget);
            assert_eq!(st.aux_major_concurrency, want);
            assert_eq!(strip_rel_diff(&base, &s), 0.0, "t={threads} b={budget}");
        }
    }

    /// The batch slab is sized with every buffer beside it (PR #716
    /// review): 2 MiB per worker, `nsf = nnf = 64`, 16 occupieds, 64
    /// virtual columns. Triples and `H` alone let 51 functions in and leave
    /// 8,192 B, while one occupied's selected `H` rows over all 51 need
    /// 26,112 B; counted in full, the slab and all its buffers fit.
    #[test]
    fn batch_slab_and_its_buffers_fit_the_worker_budget() {
        let budget = 2usize << 20;
        let d = SlabDims {
            nsf: 64,
            nnf: 64,
            n_occ: 16,
            max_cols: 64,
            naux: 1000,
        };
        let funcs = d.max_funcs(budget);
        assert!(funcs > 1, "the example must block");
        assert!(d.live_bytes(funcs) <= budget);
        assert!(d.live_bytes(funcs + 1) > budget);
        // the triples + H sizing of the earlier code: 51 functions, and the
        // selected rows alone overrun what it left
        let old_row = 8 * d.nnf * (d.nsf + d.n_occ);
        let old = budget / old_row;
        assert_eq!(old, 51);
        assert_eq!(budget - old * old_row, 8192);
        assert_eq!(8 * old * d.nnf, 26_112);
        assert!(d.live_bytes(old) > budget);
    }

    /// The batch pass's strips do not depend on the slab cap beyond rounding:
    /// an ample cap (one slab per batch), the production default, a cap a few
    /// slabs wide and a one-byte cap (one aux shell per slab) give the same
    /// strips, through strictly more slabs as the cap shrinks. Not bitwise:
    /// the virtual GEMM's row count follows the slab, so BLAS blocks the
    /// k-sums differently (measured here: at most a few 1e-16 relative,
    /// printed below); 1e-13 is the aux-major-vs-batches bar of
    /// `aux_major_strips_match_batches`, and a misplaced row or a lost slab
    /// is O(1). The scratch cap bounds the slab cap from above, so a
    /// one-byte scratch cap is a one-byte slab cap too.
    #[test]
    fn batch_strips_ignore_the_slab_cap() {
        let c = aux_case(4);
        let op = Operator::coulomb();
        let run = |slab: usize, scratch: usize| {
            let dcfg = DirectConfig {
                slab_budget_bytes: slab,
                scratch_budget_bytes: scratch,
                ..production(Eri3Pass::Batches, 4)
            };
            aux_strips(&c, op, 0.7, &dcfg)
        };
        let ample = 1usize << 40;
        let (base, s_ample) = run(ample, ample);
        assert_eq!(s_ample.n_eri3_slabs, s_ample.n_batches);
        let workers = rayon::current_num_threads().min(s_ample.n_batches).max(1);
        let few = workers * s_ample.eri3_slab_bytes_max / 3;
        let mut prev = s_ample.n_eri3_slabs;
        for (slab, scratch) in [
            (DIRECT_DEFAULT_SLAB_BUDGET_BYTES, ample),
            (few, ample),
            (1, ample),
            (ample, 1),
        ] {
            let (s, st) = run(slab, scratch);
            let rel = strip_rel_diff(&base, &s);
            eprintln!(
                "slab {slab} scratch {scratch}: {} slabs, rel {rel:.2e}",
                st.n_eri3_slabs
            );
            assert!(rel <= 1e-13, "slab {slab} scratch {scratch}: rel {rel:e}");
            assert!(
                st.n_eri3_slabs >= prev,
                "slab {slab}: {} slabs",
                st.n_eri3_slabs
            );
            if slab == few || slab == 1 {
                assert!(
                    st.n_eri3_slabs > prev,
                    "slab {slab}: blocking never engaged"
                );
            }
            prev = st.n_eri3_slabs;
        }
    }

    /// The node's slab budget is split over the ranks sharing the node: the
    /// launcher's local count when it sets one, every rank otherwise.
    #[test]
    fn slab_budget_splits_over_ranks_on_the_node() {
        assert_eq!(ranks_sharing_node(1, None), 1);
        assert_eq!(ranks_sharing_node(1, Some(4)), 1);
        assert_eq!(ranks_sharing_node(4, None), 4);
        assert_eq!(ranks_sharing_node(4, Some(2)), 2);
        assert_eq!(ranks_sharing_node(4, Some(8)), 4);
        assert_eq!(ranks_sharing_node(4, Some(0)), 1);
        let dcfg = |size: usize| DirectConfig {
            slab_budget_bytes: 1200,
            mpi: Some(ParallelContext::for_rank(0, size)),
            ..DirectConfig::default()
        };
        // one rank, or no context: the whole budget, capped by the scratch
        assert_eq!(batch_slab_cap(&dcfg(1), usize::MAX), 1200);
        assert_eq!(
            batch_slab_cap(
                &DirectConfig {
                    mpi: None,
                    ..dcfg(1)
                },
                500
            ),
            500
        );
        let local = launcher_local_ranks();
        assert_eq!(
            batch_slab_cap(&dcfg(3), usize::MAX),
            1200 / ranks_sharing_node(3, local)
        );
    }

    /// The budget limits the tasks really in flight, not just the number the
    /// pass computes. The ample run must exceed one task, or this test could not
    /// see a limit being ignored.
    #[test]
    fn aux_major_budget_limits_tasks_in_flight() {
        let c = aux_case(4);
        let op = Operator::coulomb();
        let run = |budget: usize| {
            let dcfg = DirectConfig {
                scratch_budget_bytes: budget,
                ..production(Eri3Pass::AuxMajor, 1)
            };
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap()
                .install(|| aux_strips(&c, op, 0.7, &dcfg))
                .1
        };
        let ample = run(1usize << 32);
        assert_eq!(ample.aux_major_concurrency, 4);
        assert!(ample.aux_major_in_flight_peak > 1 && ample.aux_major_in_flight_peak <= 4);
        let one = run(1);
        assert_eq!(one.aux_major_in_flight_peak, 1);
        let two = run(2 * ample.aux_major_task_bytes_max);
        assert_eq!(two.aux_major_concurrency, 2);
        assert!(
            two.aux_major_in_flight_peak <= 2,
            "{}",
            two.aux_major_in_flight_peak
        );
    }

    #[test]
    fn eri3_pass_parser_is_strict() {
        assert_eq!(
            Eri3Pass::parse_config_str("aux-major"),
            Ok(Eri3Pass::AuxMajor)
        );
        assert_eq!(Eri3Pass::parse_config_str("batches"), Ok(Eri3Pass::Batches));
        for bad in ["", "Aux-Major", "aux_major", "auxmajor", "batch", "BATCHES"] {
            assert!(Eri3Pass::parse_config_str(bad).is_err(), "{bad:?}");
        }
        for p in [Eri3Pass::AuxMajor, Eri3Pass::Batches] {
            assert_eq!(Eri3Pass::parse_config_str(p.as_config_str()), Ok(p));
        }
    }

    #[test]
    fn production_is_aux_major_and_resolution_follows_it() {
        let p = DirectConfig::production();
        assert_eq!((p.eri3_pass, p.batch_merge), (Eri3Pass::AuxMajor, 1));
        // The exactness-anchor config is unchanged.
        let d = DirectConfig::default();
        assert_eq!((d.eri3_pass, d.batch_merge), (Eri3Pass::Batches, 1));
        assert_eq!(
            DirectConfig::resolve_pass(None, None),
            Ok((Eri3Pass::AuxMajor, 1))
        );
        assert_eq!(
            DirectConfig::resolve_pass(Some("aux-major"), Some(1)),
            Ok((Eri3Pass::AuxMajor, 1))
        );
        assert_eq!(
            DirectConfig::resolve_pass(Some("batches"), None),
            Ok((Eri3Pass::Batches, DIRECT_DEFAULT_BATCH_MERGE))
        );
        assert_eq!(DIRECT_DEFAULT_BATCH_MERGE, 4);
        assert_eq!(
            DirectConfig::resolve_pass(Some("batches"), Some(2)),
            Ok((Eri3Pass::Batches, 2))
        );
        assert!(DirectConfig::resolve_pass(Some("bogus"), None).is_err());
        assert!(DirectConfig::resolve_pass(Some("batches"), Some(0)).is_err());
    }

    #[test]
    fn resolve_pass_rejects_batch_merge_with_aux_major() {
        for pass in [None, Some("aux-major")] {
            let e = DirectConfig::resolve_pass(pass, Some(4)).unwrap_err();
            assert!(e.contains("eri3_pass = \"batches\""), "{e}");
        }
    }

    #[test]
    fn aux_major_rejects_batch_merge() {
        let c = aux_case(2);
        let mut st = DirectStats::default();
        let r = strips_and_metric(
            &c.mol,
            &c.obs,
            &c.dfbs,
            Operator::coulomb(),
            &c.spaces,
            1e-4,
            Some(0.7),
            &production(Eri3Pass::AuxMajor, 2),
            None,
            &mut st,
        );
        assert!(r.is_err());
    }

    /// Integral-work counters for the aux-major idea (plan stage F0): what the
    /// shipped merged batches evaluate, how many of those triples are distinct,
    /// and what one evaluation per aux shell over the union of its occupieds'
    /// supports would evaluate and transform. Integral-free. The batch row is
    /// the check on the counter: it must reproduce the bench's triples and
    /// transform flops. FERRIC_LMP2_COUNT_SIZES selects the alkanes.
    #[test]
    #[ignore = "counter probe: one SCF per size, minutes"]
    fn aux_major_counter_probe() {
        let sizes = std::env::var("FERRIC_LMP2_COUNT_SIZES").unwrap_or_else(|_| "32,48".into());
        for nc in sizes.split(',').map(|t| t.parse::<usize>().unwrap()) {
            let mol = Molecule::load_xyz(&format!(
                "{}/../../testdata/molecules/alkane_{nc}.xyz",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let obs_bs = ferric_core::basis::bundled("6-31g").unwrap();
            let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
            let dfbs =
                PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz-ri").unwrap())
                    .unwrap();
            let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
            let cfg = RhfConfig::default().with_energy_conv(1e-10);
            let rhf = solve_rhf(
                &ParallelContext::default(),
                &mol,
                &obs,
                Operator::coulomb(),
                &bounds,
                &cfg,
            )
            .unwrap();
            let (vvhv, _) = checked_vvhv(&mol, &obs, &obs_bs, &rhf, "probe").unwrap();
            let spaces = localized_spaces(&mol, &obs, &rhf, nc, &vvhv).unwrap();
            let dcfg = DirectConfig {
                aux_radius_bohr: 10.0,
                virt_radius_bohr: Some(12.0),
                ao_tail: 1e-3,
                schwarz_skip: 1e-5,
                batch_merge: 4,
                ..Default::default()
            };
            for (name, op, cal) in [
                ("coul", Operator::coulomb(), 0.7),
                ("erfc1", Operator::erfc(1.0), 0.02),
            ] {
                let qpair = schwarz_triple_factors(op, &obs, &dfbs, &spaces, dcfg.schwarz_skip)
                    .unwrap()
                    .unwrap();
                for eps in [1e-3, 1e-4] {
                    let maps =
                        build_locality_maps(&mol, &obs, &dfbs, &spaces, eps, Some(cal), &dcfg);
                    count_row(
                        nc,
                        name,
                        eps,
                        &obs,
                        &dfbs,
                        &spaces,
                        &maps,
                        &qpair,
                        dcfg.schwarz_skip,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn count_row(
        nc: usize,
        name: &str,
        eps: f64,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        spaces: &LocalizedSpaces,
        maps: &LocalityMaps,
        qpair: &(Array2<f64>, Vec<f64>),
        skip: f64,
    ) {
        let (q, qp) = qpair;
        let (dobs, ddf) = (obs.shell_dims(), dfbs.shell_dims());
        let (nsh, nshdf) = (obs.nshells(), dfbs.nshells());
        let funcs = |shs: &[usize]| shs.iter().map(|&s| dobs[s]).sum::<usize>() as f64;
        let mark = |shs: &[usize]| {
            let mut v = vec![false; nsh];
            shs.iter().for_each(|&s| v[s] = true);
            v
        };
        let own_n: Vec<Vec<usize>> = maps
            .virt_ext
            .iter()
            .map(|vs| sorted_union(vs.iter().map(|&a| maps.virt_supp[a].iter().copied())))
            .collect();
        let npairs = nsh * (nsh + 1) / 2;
        let mut seen = vec![false; npairs * nshdf];
        // triples, integrals, (for each map) in order
        let (mut t_b, mut i_b, mut t_u, mut i_u, mut fl_b) = (0u64, 0f64, 0u64, 0f64, 0f64);
        for (_, is) in &maps.batches {
            let ss = sorted_union(is.iter().map(|&i| maps.occ_supp[i].iter().copied()));
            let nn = sorted_union(is.iter().map(|&i| own_n[i].iter().copied()));
            let ps = sorted_union(is.iter().map(|&i| maps.aux_ext[i].iter().copied()));
            let (ins, inn) = (mark(&ss), mark(&nn));
            let (nsf, nnf) = (funcs(&ss), funcs(&nn));
            for &i in is {
                let rows: usize = maps.aux_ext[i].iter().map(|&p| ddf[p]).sum();
                fl_b += rows as f64 * (2.0 * nsf * nnf + 2.0 * nnf * maps.virt_ext[i].len() as f64);
            }
            for ua in 0..nsh {
                for ub in 0..=ua {
                    if !((ins[ua] && inn[ub]) || (ins[ub] && inn[ua])) {
                        continue;
                    }
                    let id = ua * (ua + 1) / 2 + ub;
                    for &sp in &ps {
                        if qp[sp] * q[(ua, ub)] >= skip {
                            let w = (ddf[sp] * dobs[ua] * dobs[ub]) as f64;
                            t_b += 1;
                            i_b += w;
                            if !seen[id * nshdf + sp] {
                                seen[id * nshdf + sp] = true;
                                t_u += 1;
                                i_u += w;
                            }
                        }
                    }
                }
            }
        }
        let (mut t_p, mut i_p, mut fl_p, mut fl_po) = (0u64, 0f64, 0f64, 0f64);
        let (mut mx_s, mut mx_n) = (0f64, 0f64);
        for sp in 0..nshdf {
            let is: Vec<usize> = (0..spaces.no)
                .filter(|&i| maps.aux_ext[i].binary_search(&sp).is_ok())
                .collect();
            if is.is_empty() {
                continue;
            }
            let ss = sorted_union(is.iter().map(|&i| maps.occ_supp[i].iter().copied()));
            let nn = sorted_union(is.iter().map(|&i| own_n[i].iter().copied()));
            let (ins, inn) = (mark(&ss), mark(&nn));
            let (nsf, nnf) = (funcs(&ss), funcs(&nn));
            mx_s = mx_s.max(nsf);
            mx_n = mx_n.max(nnf);
            for &i in &is {
                let nv = maps.virt_ext[i].len() as f64;
                fl_p += ddf[sp] as f64 * (2.0 * nsf * nnf + 2.0 * nnf * nv);
                let (ni, nni) = (funcs(&maps.occ_supp[i]), funcs(&own_n[i]));
                fl_po += ddf[sp] as f64 * (2.0 * ni * nni + 2.0 * nni * nv);
            }
            for ua in 0..nsh {
                for ub in 0..=ua {
                    if ((ins[ua] && inn[ub]) || (ins[ub] && inn[ua]))
                        && qp[sp] * q[(ua, ub)] >= skip
                    {
                        t_p += 1;
                        i_p += (ddf[sp] * dobs[ua] * dobs[ub]) as f64;
                    }
                }
            }
        }
        println!(
            "COUNT c={nc} op={name} eps={eps:.0e} batches={} batch_Mt={:.2} batch_Mi={:.0} uniq_Mt={:.2} \
             uniq_Mi={:.0} pmajor_Mt={:.2} pmajor_Mi={:.0} batch_tG={:.1} pmajor_tG={:.1} pmajor_own_tG={:.1} \
             pmajor_max_nsf={mx_s} pmajor_max_nnf={mx_n}",
            maps.batches.len(),
            t_b as f64 / 1e6,
            i_b / 1e6,
            t_u as f64 / 1e6,
            i_u / 1e6,
            t_p as f64 / 1e6,
            i_p / 1e6,
            fl_b / 1e9,
            fl_p / 1e9,
            fl_po / 1e9,
        );
    }

    /// REGRESSION (2026-11-25, aug-cc-pVDZ n-C12H26 91 mHa below canonical):
    /// the production Schwarz cut must bound what a dropped triple does to
    /// the TRANSFORMED entries, not only its raw value. For every triple the
    /// production factors drop, each function pair's contribution to any
    /// `(P|ia)`, `|(P|μν)|·(|C_μi||C_νa| + |C_νi||C_μa|)` with
    /// function-level (not shell-level) coefficient maxima, must stay below
    /// the skip, and no triple the raw-value cut keeps may be dropped (the
    /// weight's floor at 1). A cut on `|(P|μν)|` alone fails the first
    /// wherever |C| > 1
    /// (mutation: drop the coefficient weight in `schwarz_triple_factors`).
    /// n-C4H10 / aug-cc-pVDZ keeps it CI-sized (water is too compact: the
    /// cut drops nothing there); its diffuse virtuals have max |C| ~ 69.
    #[test]
    fn schwarz_cut_bounds_the_transformed_entries() {
        let mol = Molecule::load_xyz(&format!(
            "{}/../../testdata/molecules/alkane_4.xyz",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let obs_bs = ferric_core::basis::bundled("aug-cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(
            &mol,
            &ferric_core::basis::bundled("aug-cc-pvdz-ri").unwrap(),
        )
        .unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let cfg = RhfConfig::default().with_energy_conv(1e-10);
        let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
        let (vvhv, _) = checked_vvhv(&mol, &obs, &obs_bs, &rhf, "test").unwrap();
        let spaces = localized_spaces(&mol, &obs, &rhf, 4, &vvhv).unwrap();
        let fmax = |c: &Array2<f64>| -> Vec<f64> {
            c.rows()
                .into_iter()
                .map(|r| r.iter().fold(0.0f64, |m, x| m.max(x.abs())))
                .collect()
        };
        let (co, cv) = (fmax(&spaces.c_locc), fmax(&spaces.c_vloc));
        // The production skip, and a looser one: the bound must hold at any
        // skip, and at C4 the production skip drops only nearby triples (the
        // raw-value cut passes there; it broke at C10-C12, where the large
        // coefficients sit on distant diffuse shells).
        for skip in [DIRECT_DEFAULT_SCHWARZ_SKIP, 1e-3] {
            let (q, qp) = schwarz_triple_factors(op, &obs, &dfbs, &spaces, skip)
                .unwrap()
                .unwrap();
            let (n_dropped, n_looser, worst) = dropped_transformed_worst(
                &obs,
                &dfbs,
                op,
                &CutFactors {
                    q_raw: &bounds.q,
                    q: &q,
                    qp: &qp,
                    co: &co,
                    cv: &cv,
                },
                skip,
            );
            eprintln!(
                "transformed Schwarz bound: skip {skip:.0e}: {n_dropped} triples dropped, worst \
                 dropped contribution {worst:.3e}; max |C_v| {:.2}",
                cv.iter().fold(0.0f64, |m, &x| m.max(x))
            );
            assert!(
                n_dropped > 100,
                "cut dropped only {n_dropped} triples — test vacuous"
            );
            assert_eq!(
                n_looser, 0,
                "{n_looser} triples dropped that the raw-value cut keeps (weight floor lost)"
            );
            assert!(
                worst <= skip,
                "a dropped triple moves a transformed (P|ia) by {worst:.3e} > skip {skip:.0e}"
            );
        }
    }

    /// For the triples `(qp, q)` drop at `skip`: (count, count the raw-value
    /// factors `q_raw` keep, worst function-level transformed contribution
    /// `|(P|μν)|·(cᵒ_μ cᵛ_ν + cᵒ_ν cᵛ_μ)`).
    /// The cut factors `dropped_transformed_worst` compares: the old raw
    /// pair factors, the new weighted ones, the aux factors, and the
    /// per-function largest occupied / virtual coefficients.
    struct CutFactors<'a> {
        q_raw: &'a Array2<f64>,
        q: &'a Array2<f64>,
        qp: &'a [f64],
        co: &'a [f64],
        cv: &'a [f64],
    }

    fn dropped_transformed_worst(
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        f: &CutFactors,
        skip: f64,
    ) -> (usize, usize, f64) {
        let CutFactors {
            q_raw,
            q,
            qp,
            co,
            cv,
        } = *f;
        let (dobs, oobs) = (obs.shell_dims(), obs.shell_offsets());
        let mut eng = Engine::new_3center(op, obs, dfbs, 1e-14).unwrap();
        let (mut n_dropped, mut n_looser, mut worst) = (0usize, 0usize, 0.0f64);
        for sp in 0..dfbs.nshells() {
            let np = dfbs.shell_dims()[sp];
            for ua in 0..obs.nshells() {
                for ub in 0..=ua {
                    if qp[sp] * q[(ua, ub)] >= skip {
                        continue;
                    }
                    if qp[sp] * q_raw[(ua, ub)] >= skip {
                        n_looser += 1;
                    }
                    n_dropped += 1;
                    let Some(block) = eng.compute_eri3(obs, dfbs, sp, ua, ub) else {
                        continue;
                    };
                    let (n1, n2) = (dobs[ua], dobs[ub]);
                    for p in 0..np {
                        for fa in 0..n1 {
                            let mu = oobs[ua] + fa;
                            for fb in 0..n2 {
                                let nu = oobs[ub] + fb;
                                let w = co[mu] * cv[nu] + co[nu] * cv[mu];
                                worst = worst.max(block[(p * n1 + fa) * n2 + fb].abs() * w);
                            }
                        }
                    }
                }
            }
        }
        (n_dropped, n_looser, worst)
    }

    /// B4 anchor: on C4/6-31G with production maps (gate on, Schwarz cut
    /// 1e-5, merged batches, a scratch share small enough to force several
    /// slabs per batch), the Q-sorted pair list with the binary-searched
    /// cut evaluates exactly the oracle's triples (same evaluated and
    /// skipped counts per batch) and builds bitwise-identical strips.
    /// Butane's (6-31G, cc-pVDZ-RI, Coulomb) stage-3 batch inputs: the
    /// locality maps (merge 2) and the Schwarz factors of the triple cut.
    struct BatchFixture {
        c: AuxCase,
        maps: LocalityMaps,
        qpair: Option<(Array2<f64>, Vec<f64>)>,
        schwarz_skip: f64,
    }

    impl BatchFixture {
        fn butane() -> Self {
            let c = aux_case(4);
            let dcfg = DirectConfig {
                aux_radius_bohr: 10.0,
                virt_radius_bohr: Some(12.0),
                ao_tail: 1e-3,
                schwarz_skip: 1e-5,
                batch_merge: 2,
                ..Default::default()
            };
            let maps =
                build_locality_maps(&c.mol, &c.obs, &c.dfbs, &c.spaces, 1e-4, Some(0.7), &dcfg);
            let op = Operator::coulomb();
            let qpair =
                schwarz_triple_factors(op, &c.obs, &c.dfbs, &c.spaces, dcfg.schwarz_skip).unwrap();
            Self {
                c,
                maps,
                qpair,
                schwarz_skip: dcfg.schwarz_skip,
            }
        }

        fn ctx(&self, worker_budget: usize) -> BatchCtx<'_> {
            BatchCtx {
                op: Operator::coulomb(),
                obs: &self.c.obs,
                dfbs: &self.c.dfbs,
                spaces: &self.c.spaces,
                aux_ext: &self.maps.aux_ext,
                virt_ext: &self.maps.virt_ext,
                occ_supp: &self.maps.occ_supp,
                virt_supp: &self.maps.virt_supp,
                qpair: self.qpair.as_ref(),
                schwarz_skip: self.schwarz_skip,
                worker_budget,
                split_timers: false,
            }
        }
    }

    /// The batch worker's MEASURED peak (every buffer live while a slab is
    /// transformed: `p_local`, the triples, `H`, one occupied's selected
    /// rows, their product and its index lists) stays inside the worker
    /// budget (PR #716 review). Each batch gets the budget at which the
    /// earlier sizing, triples and `H` only, would take every aux function
    /// in one slab: exactly full, so the rows, product and maps beside it
    /// overran. The control is that this budget still blocks the batch.
    #[test]
    fn batch_transform_live_bytes_fit_the_worker_budget() {
        let fx = BatchFixture::butane();
        let dims_df = fx.c.dfbs.shell_dims();
        let mut n_blocked = 0usize;
        for (_, is) in &fx.maps.batches {
            let probe = fx.ctx(usize::MAX);
            let virts = sorted_union(is.iter().map(|&i| probe.virt_ext[i].iter().copied()));
            let ax = BatchAxes::new(&probe, is, &virts);
            let p_shells = sorted_union(is.iter().map(|&i| probe.aux_ext[i].iter().copied()));
            let nps: usize = p_shells.iter().map(|&sp| dims_df[sp]).sum();
            let budget = 8 * nps * ax.nnf * (ax.nsf + is.len());
            let (_, _, _, tm) = eval_batch(&fx.ctx(budget), is, fill_slab);
            assert!(
                tm.slab_bytes_max <= budget,
                "batch {is:?}: live {} B over the {budget} B worker budget",
                tm.slab_bytes_max
            );
            n_blocked += usize::from(tm.n_slabs > 1);
        }
        assert!(n_blocked > 0, "no batch blocked: the test is vacuous");
    }

    #[test]
    fn sorted_pair_list_matches_unsorted_scan() {
        let fx = BatchFixture::butane();
        let maps = &fx.maps;
        let ctx = fx.ctx(2 << 20);
        let (mut tot_trip, mut tot_skip) = (0u64, 0u64);
        for (_, is) in &maps.batches {
            let (sa, ta, ka, _) = eval_batch(&ctx, is, fill_slab);
            let (sb, tb, kb, _) = eval_batch(&ctx, is, fill_slab_unsorted_oracle);
            assert_eq!((ta, ka), (tb, kb), "triple/skip counters differ");
            for ((ia, a), (ib, b)) in sa.iter().zip(&sb) {
                assert_eq!(ia, ib);
                assert_eq!((&a.rows, &a.cols), (&b.rows, &b.cols));
                assert!(
                    a.b.iter()
                        .zip(b.b.iter())
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "strip {ia} not bitwise identical"
                );
            }
            tot_trip += ta;
            tot_skip += ka;
        }
        eprintln!("sorted pair list: {tot_trip} triples, {tot_skip} skipped");
        assert!(
            tot_skip > 0 && tot_trip > 0,
            "cut inactive: test is vacuous"
        );
    }
}

/// The pre-Stage-C stage-5 kernels, FROZEN as the oracle of the
/// Cholesky/triangular-solve path (the `OoGram::dense` pattern): an
/// explicit `V_DD⁻¹` per group and per κ-screen occupied, then a per-pair
/// `A_iᵀ (V⁻¹ A_j)`. Do not edit — the Stage-C tests measure the
/// production path against exactly this arithmetic.
#[cfg(test)]
mod inverse_oracle {
    use super::*;
    use crate::ragged::pair_block_from_g_cand;

    /// Frozen `schwarz_virtual_q` (pre-Stage-C).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn schwarz_virtual_q_inverse_oracle(
        no: usize,
        scale: f64,
        aux_dom: &[Vec<usize>],
        strips: &[Strip],
        dims_df: &[usize],
        metric_blocks: &MetricBlocks,
        dfbs: &PreparedBasis,
        aux_radius_bohr: f64,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), FerricError> {
        use ndarray_linalg::InverseC;
        use rayon::prelude::*;
        let sq = scale.abs().sqrt();
        let per_i: Vec<Result<(Vec<f64>, f64), FerricError>> = (0..no)
            .into_par_iter()
            .map(|i| {
                let dfuncs = shell_funcs(dfbs, &aux_dom[i]);
                if dfuncs.is_empty() {
                    return Err(FerricError::General(format!(
                        "lmp2_direct: empty aux domain for occupied {i} at radius {aux_radius_bohr} \
                         Bohr (virt_schwarz_kappa needs a nonempty D_i)"
                    )));
                }
                let vdd = gather_vdd(
                    &aux_dom[i],
                    dims_df,
                    metric_blocks,
                    &format!("occupied {i} Schwarz diagonal"),
                )?;
                let vdd_inv = vdd.invc().map_err(|e| {
                    FerricError::General(format!(
                        "lmp2_direct V_DD Cholesky inverse (Schwarz diagonal, occupied {i}): {e}"
                    ))
                })?;
                let st = &strips[i];
                let mut b = Array2::<f64>::zeros((dfuncs.len(), st.cols.len()));
                for (r, &gp) in dfuncs.iter().enumerate() {
                    let sr = st.row_pos[gp];
                    if sr == usize::MAX {
                        return Err(FerricError::General(format!(
                            "lmp2_direct: strip {i} missing aux row {gp} for the Schwarz diagonal — \
                             D_i not covered by the extended domain (construction bug)"
                        )));
                    }
                    b.row_mut(r).assign(&st.b.row(sr));
                }
                let x = vdd_inv.dot(&b); // (d, ncols)
                let mut q = vec![0.0f64; st.cols.len()];
                let mut qm = 0.0f64;
                for (c, qc) in q.iter_mut().enumerate() {
                    let v = sq * b.column(c).dot(&x.column(c)).max(0.0).sqrt();
                    *qc = v;
                    qm = qm.max(v);
                }
                Ok((q, qm))
            })
            .collect();
        let mut qs = Vec::with_capacity(no);
        let mut qmax = Vec::with_capacity(no);
        for r in per_i {
            let (q, qm) = r?;
            qs.push(q);
            qmax.push(qm);
        }
        Ok((qs, qmax))
    }

    /// Frozen `fit_domain_group` (pre-Stage-C): V_DD Cholesky inverse ONCE
    /// per group, then every pair gathers its strip blocks, fits, and
    /// Eq-8-masks.
    pub(super) fn fit_domain_group_inverse_oracle(
        ctx: &PairFitCtx<'_>,
        dsh: &[usize],
        pxs: &[usize],
    ) -> Result<SlotOut, FerricError> {
        use ndarray_linalg::InverseC;
        let (i0, j0) = ctx.unique[pxs[0]];
        let dfuncs = shell_funcs(ctx.dfbs, dsh);
        if dfuncs.is_empty() {
            return Err(FerricError::General(format!(
                "lmp2_direct: empty aux domain for pair ({i0},{j0}) at radius {} Bohr",
                ctx.aux_radius_bohr
            )));
        }
        let d = dfuncs.len();
        // V_DD gathered block-wise from the sparse metric, ONCE per group
        let vdd = gather_vdd(
            dsh,
            ctx.dims_df,
            ctx.metric_blocks,
            &format!("pair ({i0},{j0})"),
        )?;
        let vdd_inv = vdd.invc().map_err(|e| {
            FerricError::General(format!(
                "lmp2_direct V_DD Cholesky inverse (group of pair ({i0},{j0})): {e}"
            ))
        })?;
        let mut out_group: SlotOut = Vec::with_capacity(pxs.len());
        for &px in pxs {
            let (i, j) = ctx.unique[px];
            let cij = &ctx.pair_cands[px];
            if cij.is_empty() {
                out_group.push((px, Vec::new()));
                continue;
            }
            // gather A_i, A_j from the strips (rows D_ij, cols C_ij)
            let gather = |k: usize| -> Result<Array2<f64>, FerricError> {
                let st = &ctx.strips[k];
                let mut a = Array2::<f64>::zeros((d, cij.len()));
                for (r, &gp) in dfuncs.iter().enumerate() {
                    let sr = st.row_pos[gp];
                    if sr == usize::MAX {
                        return Err(FerricError::General(format!(
                            "lmp2_direct: strip {k} missing aux row {gp} for pair ({i},{j}) — \
                             extended domain does not cover the pair domain (construction bug)"
                        )));
                    }
                    for (c, &av) in cij.iter().enumerate() {
                        let sc = st.col_pos[av];
                        if sc == usize::MAX {
                            return Err(FerricError::General(format!(
                                "lmp2_direct: strip {k} missing virtual {av} for pair ({i},{j})"
                            )));
                        }
                        a[(r, c)] = st.b[(sr, sc)];
                    }
                }
                Ok(a)
            };
            let a_i = gather(i)?;
            let a_j = gather(j)?;
            let mut g = a_i.t().dot(&vdd_inv.dot(&a_j));
            if ctx.scale != 1.0 {
                g.mapv_inplace(|x| ctx.scale * x);
            }
            let mut out = Vec::with_capacity(2);
            if let Some(pb) =
                pair_block_from_g_cand(i, j, &g, cij, ctx.nv, ctx.f_vv, ctx.fo, ctx.fv, ctx.eps)
            {
                out.push(pb);
            }
            if i != j {
                let gt = g.t().to_owned();
                if let Some(pb) = pair_block_from_g_cand(
                    j, i, &gt, cij, ctx.nv, ctx.f_vv, ctx.fo, ctx.fv, ctx.eps,
                ) {
                    out.push(pb);
                }
            }
            out_group.push((px, out));
        }
        Ok(out_group)
    }
}

#[cfg(test)]
mod stage_c_tests {
    use super::*;
    use crate::lmp2_amplitude::localized_spaces;
    use crate::lmp2_amplitude::{checked_vvhv, solve_and_count};
    use ferric_core::parallel::ParallelContext;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    /// RHF + localized spaces of an n-alkane / 6-31G / cc-pVDZ-RI, carbon
    /// cores frozen (plus the RHF energy and virtual space, for solves).
    struct Setup {
        mol: Molecule,
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        spaces: LocalizedSpaces,
        e_hf: f64,
        vvhv: VvHv,
    }

    fn setup(xyz: &str, fc: usize) -> Setup {
        let mol = Molecule::load_xyz(&format!(
            "{}/../../testdata/molecules/{xyz}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let obs_bs = ferric_core::basis::bundled("6-31g").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
        let cfg = RhfConfig::default().with_energy_conv(1e-10);
        let rhf = solve_rhf(
            &ParallelContext::default(),
            &mol,
            &obs,
            Operator::coulomb(),
            &bounds,
            &cfg,
        )
        .unwrap();
        let (vvhv, _) = checked_vvhv(&mol, &obs, &obs_bs, &rhf, "test").unwrap();
        let spaces = localized_spaces(&mol, &obs, &rhf, fc, &vvhv).unwrap();
        Setup {
            mol,
            obs,
            dfbs,
            spaces,
            e_hf: rhf.energy,
            vvhv,
        }
    }

    /// Production maps (Schwarz cut, k = 4 batches) with screen `kappa`.
    fn production_maps(kappa: Option<f64>) -> DirectConfig {
        DirectConfig {
            aux_radius_bohr: 10.0,
            virt_radius_bohr: Some(12.0),
            ao_tail: 1e-3,
            schwarz_skip: 1e-5,
            batch_merge: 4,
            virt_schwarz_kappa: kappa,
            ..Default::default()
        }
    }

    /// Stages 1–4 at ε with the calibrated gate `cal`.
    fn pre_of(su: &Setup, op: Operator, eps: f64, cal: f64) -> StripsAndMetric {
        pre_with(su, op, eps, Some(cal), &production_maps(None)).0
    }

    /// Stages 1–4 at ε under `dcfg` (gate `cal`, `None` = every pair), with
    /// their counters.
    fn pre_with(
        su: &Setup,
        op: Operator,
        eps: f64,
        cal: Option<f64>,
        dcfg: &DirectConfig,
    ) -> (StripsAndMetric, DirectStats) {
        let mut st = DirectStats::default();
        let pre = strips_and_metric(
            &su.mol, &su.obs, &su.dfbs, op, &su.spaces, eps, cal, dcfg, None, &mut st,
        )
        .unwrap();
        (pre, st)
    }

    fn fit(
        su: &Setup,
        pre: &StripsAndMetric,
        eps: f64,
        dcfg: &DirectConfig,
        path: FitPath,
    ) -> (Ragged, DirectStats) {
        let mut st = DirectStats::default();
        let r = fit_pairs(
            pre, &su.dfbs, &su.spaces, eps, 1.0, dcfg, path, None, None, &mut st,
        )
        .unwrap();
        (r, st)
    }

    /// The slots one simulated rank fits from its own strips (no gather: a
    /// rank-pinned context has no MPI world).
    fn rank_slots(
        su: &Setup,
        pre: &StripsAndMetric,
        eps: f64,
        dcfg: &DirectConfig,
    ) -> Vec<crate::ragged_transport::Slot> {
        let mut st = DirectStats::default();
        let input = PairFitInput {
            pre,
            dfbs: &su.dfbs,
            spaces: &su.spaces,
            eps,
            scale: 1.0,
            dcfg,
            path: FitPath::Cholesky,
            gate: None,
            fock_screen: None,
        };
        fit_pair_slots(input, &mut st).unwrap().0
    }

    /// Largest elementwise difference between two blocks' arrays, after
    /// requiring the same pair, index lists and mask.
    fn block_diff(a: &PairBlock, b: &PairBlock) -> f64 {
        assert_eq!((a.i, a.j), (b.i, b.j));
        assert_eq!((&a.da, &a.db), (&b.da, &b.db), "pair ({}, {})", a.i, a.j);
        assert_eq!((&a.pos_da, &a.pos_db), (&b.pos_da, &b.pos_db));
        assert_eq!(a.pat, b.pat, "pair ({}, {}) mask", a.i, a.j);
        [
            (&a.fvv_aa, &b.fvv_aa),
            (&a.fvv_bb, &b.fvv_bb),
            (&a.j_blk, &b.j_blk),
            (&a.denom, &b.denom),
        ]
        .iter()
        .flat_map(|(x, y)| x.iter().zip(y.iter()).map(|(p, q)| (p - q).abs()))
        .fold(0.0, f64::max)
    }

    /// The rank stripe partitions the assembly. For 2, 3 and 5 simulated
    /// ranks, each building only its own strips: every slot is fitted by
    /// exactly one rank, the union of the ranks' slots is the serial ragged
    /// system (same index lists and masks, values to rounding: a strip over
    /// fewer partners goes through differently shaped GEMMs), and no rank
    /// holds more strip bytes than the serial run. One rank is the serial
    /// path bit for bit. The gate is off, as in the CLI default, so every
    /// occupied pairs with every other and a strip shrinks only because it
    /// covers this rank's partners.
    #[test]
    fn rank_stripes_union_to_the_serial_blocks() {
        use ferric_core::parallel::ParallelContext;
        // octane: long enough that the pair domains differ, so there are
        // several domain groups to stripe (butane has one).
        let su = setup("alkane_8.xyz", 8);
        let (eps, op) = (1e-4, Operator::coulomb());
        let dcfg = production_maps(None);
        let (pre, st_pre) = pre_with(&su, op, eps, None, &dcfg);
        let (serial, st) = fit(&su, &pre, eps, &dcfg, FitPath::Cholesky);
        assert!(
            st.n_pair_groups >= 5,
            "need >= 5 groups, got {}",
            st.n_pair_groups
        );
        let one = DirectConfig {
            mpi: Some(ParallelContext::for_rank(0, 1)),
            ..dcfg.clone()
        };
        let (pre_one, _) = pre_with(&su, op, eps, None, &one);
        assert!(pre_one.stripe.is_none(), "one rank is the serial path");
        let (one_rank, _) = fit(&su, &pre_one, eps, &one, FitPath::Cholesky);
        for (a, b) in one_rank.pairs.iter().zip(&serial.pairs) {
            assert_eq!(block_diff(a, b), 0.0);
        }
        let scale = serial
            .pairs
            .iter()
            .flat_map(|pb| pb.j_blk.iter())
            .fold(0.0f64, |m, v| m.max(v.abs()));
        for size in [2usize, 3, 5] {
            let mut strip_bytes = Vec::new();
            let per_rank: Vec<_> = (0..size)
                .map(|rank| {
                    let striped = DirectConfig {
                        mpi: Some(ParallelContext::for_rank(rank, size)),
                        ..dcfg.clone()
                    };
                    let (pre_r, st_r) = pre_with(&su, op, eps, None, &striped);
                    strip_bytes.push(st_r.strip_bytes_total);
                    rank_slots(&su, &pre_r, eps, &striped)
                })
                .collect();
            let n_slots = per_rank[0].len();
            let mut union: Vec<PairBlock> = Vec::new();
            for px in 0..n_slots {
                let owners: Vec<usize> = (0..size).filter(|&r| per_rank[r][px].is_some()).collect();
                assert_eq!(owners.len(), 1, "size {size}: slot {px} owners {owners:?}");
                union.extend(per_rank[owners[0]][px].clone().unwrap());
            }
            assert_eq!(union.len(), serial.pairs.len(), "size {size}");
            let worst = union
                .iter()
                .zip(&serial.pairs)
                .map(|(a, b)| block_diff(a, b))
                .fold(0.0, f64::max);
            eprintln!(
                "[stripe] size {size}: max block diff {worst:.2e} (scale {scale:.2e}); strip \
                 bytes per rank {strip_bytes:?} vs serial {}",
                st_pre.strip_bytes_total
            );
            assert!(worst <= 1e-11 * scale.max(1.0), "size {size}: {worst:.3e}");
            // No rank did all of the work, and more than one did some. (A
            // group is never split, and most of octane's pairs share one
            // domain, so at 5 ranks some get no group.)
            let working = per_rank
                .iter()
                .filter(|s| s.iter().flatten().count() > 0)
                .count();
            assert!(
                working >= 2,
                "size {size}: {working} rank(s) fitted anything"
            );
            for (rank, s) in per_rank.iter().enumerate() {
                let n = s.iter().flatten().count();
                assert!(n < n_slots, "size {size}: rank {rank} fitted everything");
                assert!(
                    strip_bytes[rank] <= st_pre.strip_bytes_total,
                    "size {size}: rank {rank} holds {} strip bytes, serial {}",
                    strip_bytes[rank],
                    st_pre.strip_bytes_total
                );
            }
            assert!(
                strip_bytes.iter().any(|&b| b < st_pre.strip_bytes_total),
                "size {size}: no rank's strips shrank"
            );
        }
    }

    /// A rank-pinned context has no MPI world to gather over, so the full
    /// driver must refuse rather than return a partial ragged system.
    #[test]
    fn a_stripe_without_a_world_is_refused() {
        use ferric_core::parallel::ParallelContext;
        let su = setup("alkane_8.xyz", 8);
        let eps = 1e-4;
        let striped = DirectConfig {
            mpi: Some(ParallelContext::for_rank(0, 2)),
            ..production_maps(None)
        };
        let (pre, _) = pre_with(&su, Operator::coulomb(), eps, Some(0.7), &striped);
        let mut st = DirectStats::default();
        let err = fit_pairs(
            &pre,
            &su.dfbs,
            &su.spaces,
            eps,
            1.0,
            &striped,
            FitPath::Cholesky,
            None,
            None,
            &mut st,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("never filled"), "{err}");
    }

    /// The trivial limit of the split solve: a share that owns every block
    /// (one rank, no MPI world) runs the same PCG as no share at all, so the
    /// amplitudes, the energy and the true residual are bit for bit the
    /// plain indexed solve's.
    #[test]
    fn a_share_owning_every_block_is_the_plain_solve() {
        use crate::lmp2_amplitude::hylleraas_energy_and_residual;
        use crate::ragged::{solve_ragged_budgeted_with, MatvecPath, RaggedShare};
        use ferric_core::parallel::ParallelContext;
        let su = setup("alkane_4.xyz", 4);
        let eps = 1e-4;
        let pre = pre_of(&su, Operator::coulomb(), eps, 0.7);
        let dcfg = production_maps(None);
        let (plain, _) = fit(&su, &pre, eps, &dcfg, FitPath::Cholesky);
        let mut shared = plain.clone();
        shared.share = Some(RaggedShare {
            ctx: ParallelContext::for_rank(0, 1),
            owned: vec![true; shared.pairs.len()],
        });
        let f_oo = &su.spaces.f_oo;
        // budget 0: no gather plan, so both run the indexed matvec
        let a = solve_ragged_budgeted_with(&plain, f_oo, 1e-9, 200, 0, 0.0);
        let b = solve_ragged_budgeted_with(&shared, f_oo, 1e-9, 200, 0, 0.0);
        assert_eq!(a.decision.path, MatvecPath::Indexed);
        assert_eq!(b.decision.path, MatvecPath::Indexed);
        assert!(a.solve.converged && a.solve.iterations > 3);
        assert_eq!(a.solve.iterations, b.solve.iterations);
        for (x, y) in a.solve.t.iter().zip(&b.solve.t) {
            assert_eq!(x, y);
        }
        let ea = hylleraas_energy_and_residual(&plain, f_oo, None, &a.solve.t).unwrap();
        let eb = hylleraas_energy_and_residual(&shared, f_oo, None, &b.solve.t).unwrap();
        assert_eq!(ea.0.to_bits(), eb.0.to_bits());
        assert_eq!(ea.1.to_bits(), eb.1.to_bits());
        assert!(ea.0 < -0.1, "a real correlation energy, got {}", ea.0);
    }

    /// The planned counterpart: with an ample budget a share owning every
    /// block builds the plan (of the same size) and solves bit for bit as
    /// the plain system, Hylleraas energy and true residual included.
    #[test]
    fn a_share_owning_every_block_is_the_plain_planned_solve() {
        use crate::lmp2_amplitude::hylleraas_energy_and_residual;
        use crate::ragged::{solve_ragged_budgeted_with, MatvecPath, RaggedShare};
        use ferric_core::parallel::ParallelContext;
        let su = setup("alkane_4.xyz", 4);
        let eps = 1e-4;
        let pre = pre_of(&su, Operator::coulomb(), eps, 0.7);
        let dcfg = production_maps(None);
        let (plain, _) = fit(&su, &pre, eps, &dcfg, FitPath::Cholesky);
        let mut shared = plain.clone();
        shared.share = Some(RaggedShare {
            ctx: ParallelContext::for_rank(0, 1),
            owned: vec![true; shared.pairs.len()],
        });
        let f_oo = &su.spaces.f_oo;
        let a = solve_ragged_budgeted_with(&plain, f_oo, 1e-9, 200, usize::MAX, 0.0);
        let b = solve_ragged_budgeted_with(&shared, f_oo, 1e-9, 200, usize::MAX, 0.0);
        assert_eq!(a.decision.path, MatvecPath::Planned);
        assert_eq!(b.decision.path, MatvecPath::Planned);
        assert_eq!(a.decision.plan_bytes, b.decision.plan_bytes);
        assert!(a.solve.converged && a.solve.iterations > 3);
        assert_eq!(a.solve.iterations, b.solve.iterations);
        assert_eq!(a.solve.relres.to_bits(), b.solve.relres.to_bits());
        for (x, y) in a.solve.t.iter().zip(&b.solve.t) {
            assert!(x.iter().zip(y).all(|(u, v)| u.to_bits() == v.to_bits()));
        }
        let ea = hylleraas_energy_and_residual(&plain, f_oo, a.plan(), &a.solve.t).unwrap();
        let eb = hylleraas_energy_and_residual(&shared, f_oo, b.plan(), &b.solve.t).unwrap();
        assert_eq!(ea.0.to_bits(), eb.0.to_bits());
        assert_eq!(ea.1.to_bits(), eb.1.to_bits());
        assert!(ea.0 < -0.1, "a real correlation energy, got {}", ea.0);
    }

    /// A chain of `no` occupieds one Bohr apart whose aux domain is the
    /// `2·reach + 1` shells around it (one function per shell), every pair
    /// kept: the integral-free inputs of [`rank_stripe`].
    fn chain(no: usize, reach: usize) -> (Vec<bool>, Vec<Vec<usize>>, Vec<[f64; 3]>) {
        let dom = (0..no)
            .map(|i| (i.saturating_sub(reach)..(i + reach + 1).min(no)).collect())
            .collect();
        let xyz = (0..no).map(|i| [0.0, i as f64, 0.0]).collect();
        (vec![true; no * no], dom, xyz)
    }

    /// The stripe plan on a synthetic chain: the ranks' pairs partition the
    /// surviving pairs, an occupied is needed exactly when it is in one of
    /// the rank's pairs, its partners are exactly the occupieds it shares
    /// those pairs with (and itself), and the pair work is split within a
    /// factor of two of even.
    #[test]
    fn rank_stripe_partitions_pairs_and_names_their_partners() {
        use ferric_core::parallel::ParallelContext;
        let (no, reach) = (40usize, 3usize);
        let (keep, dom, xyz) = chain(no, reach);
        let dims = vec![1usize; no];
        let unique = kept_pairs(&keep, no);
        for size in [2usize, 3, 4, 7] {
            let mut owners = vec![0usize; unique.len()];
            for rank in 0..size {
                let ctx = ParallelContext::for_rank(rank, size);
                let (s, partners) = rank_stripe(ctx, &keep, &dom, &dom, &xyz, &dims);
                let mut want: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); no];
                for (px, &(i, j)) in unique.iter().enumerate() {
                    if s.mine[px] {
                        owners[px] += 1;
                        want[i].extend([i, j]);
                        want[j].extend([i, j]);
                    }
                }
                for i in 0..no {
                    assert_eq!(
                        s.needed[i],
                        !want[i].is_empty(),
                        "size {size} rank {rank} i {i}"
                    );
                    let got: BTreeSet<usize> = partners[i].iter().copied().collect();
                    assert_eq!(got, want[i], "size {size} rank {rank} i {i}");
                }
                let share = s.mine.iter().filter(|&&m| m).count() as f64 / unique.len() as f64;
                assert!(
                    share > 0.5 / size as f64 && share < 2.0 / size as f64,
                    "size {size} rank {rank}: {share:.3} of the pairs"
                );
            }
            assert!(owners.iter().all(|&n| n == 1), "size {size}: {owners:?}");
        }
    }

    /// Why the cut follows the molecule: on the chain, the partner-domain
    /// rows a rank's strips need (`Σ_i |∪_{j ∈ partners(i)} D_j|`, the strip
    /// row count) fall with the rank count, while an every-`size`-th-group
    /// split leaves each rank needing most of them. The bounds are the
    /// measured fractions (0.60, 0.40, 0.30 at 2, 4, 8 ranks) with margin:
    /// the rank at the low end of the chain holds a strip for every occupied.
    #[test]
    fn a_contiguous_stripe_shrinks_the_strip_rows_a_modulo_one_does_not() {
        use ferric_core::parallel::ParallelContext;
        let (no, reach) = (60usize, 3usize);
        let (keep, dom, xyz) = chain(no, reach);
        let dims = vec![1usize; no];
        let rows = |partners: &[Vec<usize>]| -> usize {
            partners
                .iter()
                .map(|p| sorted_union(p.iter().map(|&j| dom[j].iter().copied())).len())
                .sum()
        };
        let serial = rows(&vec![(0..no).collect::<Vec<usize>>(); no]);
        let unique = kept_pairs(&keep, no);
        let groups = group_by_domain(pair_domains(&unique, &dom));
        for (size, bound) in [(2usize, 0.65), (4, 0.45), (8, 0.35)] {
            let worst = (0..size)
                .map(|rank| {
                    let ctx = ParallelContext::for_rank(rank, size);
                    rows(&rank_stripe(ctx, &keep, &dom, &dom, &xyz, &dims).1)
                })
                .max()
                .unwrap();
            let modulo = (0..size)
                .map(|rank| {
                    let mut partners: Vec<Vec<usize>> = vec![Vec::new(); no];
                    for (_, pxs) in groups.iter().skip(rank).step_by(size) {
                        for &px in pxs {
                            let (i, j) = unique[px];
                            partners[i].extend([i, j]);
                            partners[j].extend([i, j]);
                        }
                    }
                    rows(&partners)
                })
                .max()
                .unwrap();
            let (c, m) = (worst as f64 / serial as f64, modulo as f64 / serial as f64);
            eprintln!("[stripe rows] size {size}: contiguous {c:.3}, modulo {m:.3} of serial");
            assert!(c < bound, "size {size}: contiguous {c:.3}");
            assert!(m > 0.75, "size {size}: modulo {m:.3}");
        }
    }

    /// The surviving pairs (i ≤ j) in stage-5 order and their domains D_ij.
    fn unique_pairs(pre: &StripsAndMetric, no: usize) -> (Vec<(usize, usize)>, Vec<Vec<usize>>) {
        let unique: Vec<(usize, usize)> = (0..no)
            .flat_map(|i| (i..no).map(move |j| (i, j)))
            .filter(|&(i, j)| pre.keep[i * no + j])
            .collect();
        let doms = unique
            .iter()
            .map(|&(i, j)| {
                sorted_union([
                    pre.aux_dom[i].iter().copied(),
                    pre.aux_dom[j].iter().copied(),
                ])
            })
            .collect();
        (unique, doms)
    }

    /// 2-norm condition number κ(V_DD) of one domain.
    fn domain_condition(pre: &StripsAndMetric, dfbs: &PreparedBasis, dsh: &[usize]) -> f64 {
        use ndarray_linalg::{EighInto, UPLO};
        let v = gather_vdd(dsh, dfbs.shell_dims(), &pre.metric_blocks, "test").unwrap();
        let (w, _) = v.eigh_into(UPLO::Lower).unwrap();
        w[w.len() - 1] / w[0]
    }

    /// κ(V_{D_ij}) for every pair-domain group, keyed by (i, j) with i ≤ j.
    fn pair_conditions(
        pre: &StripsAndMetric,
        dfbs: &PreparedBasis,
        no: usize,
    ) -> HashMap<(usize, usize), f64> {
        let (unique, doms) = unique_pairs(pre, no);
        let mut by_dom: HashMap<Vec<usize>, f64> = HashMap::new();
        unique
            .iter()
            .zip(doms)
            .map(|(&ij, dsh)| {
                let k = *by_dom
                    .entry(dsh.clone())
                    .or_insert_with(|| domain_condition(pre, dfbs, &dsh));
                (ij, k)
            })
            .collect()
    }

    /// Per assembled block: (i, j, max|ΔJ|, max|J_oracle|); panics if the
    /// two assemblies differ in structure (pair order, retained axes or
    /// Eq-8 pattern).
    fn block_errors(new: &Ragged, old: &Ragged) -> Vec<(usize, usize, f64, f64)> {
        assert_eq!(new.pairs.len(), old.pairs.len(), "pair count differs");
        new.pairs
            .iter()
            .zip(&old.pairs)
            .map(|(a, b)| {
                assert_eq!((a.i, a.j), (b.i, b.j), "pair order differs");
                assert_eq!((&a.da, &a.db), (&b.da, &b.db), "retained axes differ");
                assert_eq!(a.pat, b.pat, "Eq-8 pattern differs for ({},{})", a.i, a.j);
                let (mut d, mut j) = (0.0f64, 0.0f64);
                for (x, y) in a.j_blk.iter().zip(b.j_blk.iter()) {
                    d = d.max((x - y).abs());
                    j = j.max(y.abs());
                }
                (a.i, a.j, d, j)
            })
            .collect()
    }

    /// GLOBAL check: max|ΔJ| / max|J| over all blocks ≤ min(κ_max·u, 1e-10).
    /// Returns (relative error, bar).
    fn global_check(errs: &[(usize, usize, f64, f64)], kmax: f64) -> (f64, f64) {
        let d = errs.iter().map(|e| e.2).fold(0.0f64, f64::max);
        let j = errs.iter().map(|e| e.3).fold(0.0f64, f64::max);
        (d / j, (kmax * f64::EPSILON).min(1e-10))
    }

    /// The per-pair bar's constant c in ‖ΔJ_ij‖_max/‖J_ij‖_max ≤ c·κ(V_{D_ij})·u.
    /// Measured worst (2026-10-30): 0.22 / 0.41 (C4 / C8 Coulomb), 0.57 /
    /// 1.28 (C4 / C8 erfc) — a small-|J| block's rounding scales with
    /// ‖W_i‖‖W_j‖, not |J_ij|, so the ratio can pass 1. c = 8 leaves 6×
    /// margin; the 1e-6 one-group mutant scores ~2.8e3.
    const PER_PAIR_C: f64 = 8.0;

    /// PER-PAIR check: worst ‖ΔJ_ij‖_max / (‖J_ij‖_max · κ(V_{D_ij}) · u)
    /// over the blocks, each against its OWN group's κ (a block J_ji is
    /// the transpose of J_ij, same domain). Empty blocks are skipped.
    fn per_pair_worst(
        errs: &[(usize, usize, f64, f64)],
        kappas: &HashMap<(usize, usize), f64>,
    ) -> (f64, (usize, usize)) {
        let mut worst = (0.0f64, (0, 0));
        for &(i, j, d, jm) in errs {
            if jm == 0.0 {
                continue;
            }
            let k = kappas[&(i.min(j), i.max(j))];
            let r = d / jm / (k * f64::EPSILON);
            if r > worst.0 {
                worst = (r, (i, j));
            }
        }
        worst
    }

    /// The stage-5 counters recomputed INDEPENDENTLY of `fit_pairs`' own
    /// tallies: groups from the pair domains, distinct occupieds per group
    /// from the pairs with a nonempty candidate set, and reuse where an
    /// occupied's own domain IS the group domain (κ screen on, store kept).
    /// Returns (groups, Σ distinct occupieds, Σ reusable).
    fn recount(
        pre: &StripsAndMetric,
        dfbs: &PreparedBasis,
        no: usize,
        eps: f64,
        kappa: Option<f64>,
    ) -> (usize, usize, usize) {
        let (unique, doms) = unique_pairs(pre, no);
        let qvirt = kappa.map(|_| {
            let o = whiten_own_domains(
                no,
                1.0,
                &pre.aux_dom,
                &pre.strips,
                dfbs.shell_dims(),
                &pre.metric_blocks,
                dfbs,
                10.0,
                false,
                None,
                FloorMode::No,
            )
            .unwrap();
            (o.q, o.qmax)
        });
        let cands = pair_candidates(
            &unique,
            &pre.virt_dom,
            &pre.strips,
            qvirt.as_ref(),
            kappa.map(|k| k * eps),
            None,
        );
        let mut occ: HashMap<&[usize], BTreeSet<usize>> = HashMap::new();
        for (px, &(i, j)) in unique.iter().enumerate() {
            let set = occ.entry(&doms[px]).or_default();
            if !cands[px].is_empty() {
                set.insert(i);
                set.insert(j);
            }
        }
        let n_occ = occ.values().map(BTreeSet::len).sum();
        let n_reuse = if kappa.is_some() {
            occ.iter()
                .map(|(dsh, ks)| {
                    ks.iter()
                        .filter(|&&k| pre.aux_dom[k].as_slice() == *dsh)
                        .count()
                })
                .sum()
        } else {
            0
        };
        (occ.len(), n_occ, n_reuse)
    }

    /// STAGE C ANCHOR: the Cholesky/triangular-solve pair fits (with the
    /// κ-screen whitening reused for pair (i,i)'s group) reproduce the
    /// frozen explicit-inverse path with the same pairs, retained axes and
    /// Eq-8 pattern, on C4 and C8, Coulomb and erfc, κ off and κ = 1, by
    /// TWO bars: GLOBAL, max|ΔJ| ≤ min(κ_max(V_DD)·u, 1e-10)·max|J|, and
    /// PER PAIR, ‖ΔJ_ij‖_max ≤ c·κ(V_{D_ij})·u·‖J_ij‖_max with each group's
    /// own κ (the global bar alone cannot see an error confined to a block
    /// whose |J| is ≪ max|J|; `per_pair_check_catches_one_group_w_error`).
    /// With κ on, q agrees to κ(V_DD)·u·max q. The counters are checked
    /// against an independent recount from the pair domains and candidates.
    ///
    /// The bar is the conditioning floor, not 1e-12: two correct f64 routes
    /// differ by O(κ·u). Measured (2026-10-30): Coulomb C4/C8 1.8e-12 /
    /// 2.9e-12 relative at κ(V_DD) = 9.9e5 / 1.7e6; erfc 2.3e-13 / 6.5e-13
    /// at κ = 4.2e4 / 4.8e4 — 0.8–6% of κ·u. A wrong W is an O(1) miss.
    #[ignore = "tier slow (>=60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
    #[test]
    fn pair_fit_cholesky_matches_inverse_oracle() {
        let (mut worst, mut worst_pp) = (0.0f64, 0.0f64);
        let eps = 1e-4;
        for (xyz, fc) in [("alkane_4.xyz", 4usize), ("alkane_8.xyz", 8)] {
            let su = setup(xyz, fc);
            let no = su.spaces.no;
            for (opn, op, cal) in [
                ("coul", Operator::coulomb(), 0.7),
                ("erfc", Operator::erfc(1.0), 0.02),
            ] {
                let pre = pre_of(&su, op, eps, cal);
                let kappas = pair_conditions(&pre, &su.dfbs, no);
                let kmax = kappas.values().copied().fold(0.0f64, f64::max);
                for kappa in [None, Some(1.0)] {
                    let dcfg = production_maps(kappa);
                    let (new, s_new) = fit(&su, &pre, eps, &dcfg, FitPath::Cholesky);
                    let (old, s_old) = fit(&su, &pre, eps, &dcfg, FitPath::InverseOracle);
                    let errs = block_errors(&new, &old);
                    let (rel, bar) = global_check(&errs, kmax);
                    let (pp, at) = per_pair_worst(&errs, &kappas);
                    worst = worst.max(rel);
                    worst_pp = worst_pp.max(pp);
                    eprintln!(
                        "STAGE-C {xyz} {opn} kappa={kappa:?}: pairs {} groups {} trsm {} \
                         reused {} occ/group {:.2} cand {:.1}/{} global rel {rel:.3e} \
                         (bar {bar:.3e}) per-pair worst {pp:.3e} of kappa_ij*u at {at:?}; \
                         own W {} B full, {} B held; group scratch max {} B",
                        new.pairs.len(),
                        s_new.n_pair_groups,
                        s_new.n_trsm,
                        s_new.n_w_reused,
                        s_new.n_distinct_occ_per_group,
                        s_new.virt_cand_mean,
                        s_new.virt_cand_max,
                        s_new.own_w_bytes,
                        s_new.own_w_bytes_held,
                        s_new.group_scratch_bytes_max,
                    );
                    assert!(
                        rel <= bar,
                        "blocks differ beyond kappa(V_DD)*u: rel {rel:.3e}"
                    );
                    assert!(
                        pp <= PER_PAIR_C,
                        "pair {at:?} differs beyond {PER_PAIR_C}*kappa(V_Dij)*u: {pp:.3e}"
                    );
                    assert_eq!(s_new.virt_cand_max, s_old.virt_cand_max);
                    assert!((s_new.virt_cand_mean - s_old.virt_cand_mean).abs() < 1e-12);

                    let (groups, n_occ, n_reuse) = recount(&pre, &su.dfbs, no, eps, kappa);
                    let own = usize::from(kappa.is_some()) * no;
                    assert_eq!(s_new.n_pair_groups, groups, "group count");
                    assert_eq!(s_new.n_w_reused, n_reuse, "reused whitenings");
                    assert_eq!(s_new.n_trsm, own + n_occ - n_reuse, "triangular solves");
                    assert!(
                        (s_new.n_distinct_occ_per_group - n_occ as f64 / groups as f64).abs()
                            < 1e-12
                    );
                    assert_eq!(kappa.is_some(), n_reuse > 0, "reuse arm inert");
                    assert!(s_new.group_scratch_bytes_max > 0);
                    if kappa.is_some() {
                        assert!(s_new.own_w_bytes_held > 0);
                        assert!(s_new.own_w_bytes_held <= s_new.own_w_bytes);
                        let o = whiten_own_domains(
                            no,
                            1.0,
                            &pre.aux_dom,
                            &pre.strips,
                            su.dfbs.shell_dims(),
                            &pre.metric_blocks,
                            &su.dfbs,
                            dcfg.aux_radius_bohr,
                            false,
                            None,
                            FloorMode::No,
                        )
                        .unwrap();
                        let (q_old, _) = inverse_oracle::schwarz_virtual_q_inverse_oracle(
                            no,
                            1.0,
                            &pre.aux_dom,
                            &pre.strips,
                            su.dfbs.shell_dims(),
                            &pre.metric_blocks,
                            &su.dfbs,
                            dcfg.aux_radius_bohr,
                        )
                        .unwrap();
                        let qmax = o.qmax.iter().copied().fold(0.0f64, f64::max);
                        let dq =
                            o.q.iter()
                                .flatten()
                                .zip(q_old.iter().flatten())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f64, f64::max);
                        eprintln!("STAGE-C {xyz} {opn} q: max|dq| {dq:.3e} max q {qmax:.3e}");
                        let ku = kmax * f64::EPSILON;
                        assert!(dq <= ku * qmax, "q differs: {dq:.3e} vs max q {qmax:.3e}");
                    } else {
                        assert_eq!((s_new.own_w_bytes, s_new.own_w_bytes_held), (0, 0));
                    }
                }
            }
        }
        eprintln!("STAGE-C worst block error {worst:.3e} (global), {worst_pp:.3e} (per pair, of kappa_ij*u)");
    }

    /// Over the scratch budget the own-domain store is not kept: reuse is
    /// off, every group solves, and the blocks are UNCHANGED (C4 Coulomb).
    #[test]
    fn own_w_store_over_budget_falls_back_to_solving() {
        let su = setup("alkane_4.xyz", 4);
        let pre = pre_of(&su, Operator::coulomb(), 1e-4, 0.7);
        let (kept, s_kept) = fit(
            &su,
            &pre,
            1e-4,
            &production_maps(Some(1.0)),
            FitPath::Cholesky,
        );
        let tight = DirectConfig {
            scratch_budget_bytes: s_kept.own_w_bytes - 1,
            ..production_maps(Some(1.0))
        };
        let (fell, s_fell) = fit(&su, &pre, 1e-4, &tight, FitPath::Cholesky);
        eprintln!(
            "STAGE-C budget: kept reuse {} trsm {} | over budget reuse {} trsm {} held {}",
            s_kept.n_w_reused,
            s_kept.n_trsm,
            s_fell.n_w_reused,
            s_fell.n_trsm,
            s_fell.own_w_bytes_held
        );
        assert!(s_kept.n_w_reused > 0);
        assert_eq!((s_fell.n_w_reused, s_fell.own_w_bytes_held), (0, 0));
        assert_eq!(s_fell.n_trsm, s_kept.n_trsm + s_kept.n_w_reused);
        let errs = block_errors(&fell, &kept);
        let (d, j) = errs
            .iter()
            .fold((0.0f64, 0.0f64), |(d, j), e| (d.max(e.2), j.max(e.3)));
        assert!(d <= 1e-10 * j, "fallback changed the blocks: {d:.3e}");
    }

    /// MUTANT for the per-pair bar: scale ONE off-diagonal group's whitened
    /// `W` (its first pair's i-occupied) by 1 + δ, picking the group whose
    /// affected blocks have the smallest |J|. At δ = 1e-6 both bars catch it
    /// (measured); at δ = 1e-8 the global bar (normalized by max|J| over ALL
    /// blocks) MISSES it and only the per-pair bar catches it — the reason
    /// the per-pair bar exists.
    /// C8 (C4's 10-Bohr domains collapse to ONE group, which has no
    /// off-diagonal first pair).
    #[test]
    fn per_pair_check_catches_one_group_w_error() {
        let su = setup("alkane_8.xyz", 8);
        let no = su.spaces.no;
        let eps = 1e-4;
        let pre = pre_of(&su, Operator::coulomb(), eps, 0.7);
        let kappas = pair_conditions(&pre, &su.dfbs, no);
        let kmax = kappas.values().copied().fold(0.0f64, f64::max);
        let dcfg = production_maps(None);
        let (old, _) = fit(&su, &pre, eps, &dcfg, FitPath::InverseOracle);
        let jmax_of = |i: usize, j: usize| {
            old.pairs
                .iter()
                .filter(|p| (p.i.min(p.j), p.i.max(p.j)) == (i, j))
                .flat_map(|p| p.j_blk.iter())
                .fold(0.0f64, |m, x| m.max(x.abs()))
        };
        let (unique, doms) = unique_pairs(&pre, no);
        let groups = group_by_domain(doms);
        // off-diagonal first pair; affected blocks = its group's pairs with k
        let (first, jm) = groups
            .iter()
            .filter(|(_, pxs)| unique[pxs[0]].0 != unique[pxs[0]].1)
            .map(|(_, pxs)| {
                let k = unique[pxs[0]].0;
                let jm = pxs
                    .iter()
                    .filter(|&&px| unique[px].0 == k || unique[px].1 == k)
                    .map(|&px| jmax_of(unique[px].0, unique[px].1))
                    .fold(0.0f64, f64::max);
                (pxs[0], jm)
            })
            .filter(|&(_, jm)| jm > 0.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("an off-diagonal group");
        let global_max = old
            .pairs
            .iter()
            .flat_map(|p| p.j_blk.iter())
            .fold(0.0f64, |m, x| m.max(x.abs()));
        // (global passes?, per-pair passes?) at a relative W error `rel`
        let verdicts = |rel_w: f64| {
            let (bad, _) = fit(&su, &pre, eps, &dcfg, FitPath::MutantW(first, rel_w));
            let errs = block_errors(&bad, &old);
            let (rel, bar) = global_check(&errs, kmax);
            let (pp, at) = per_pair_worst(&errs, &kappas);
            eprintln!(
                "STAGE-C mutant dW/W {rel_w:.0e}: group of pair {:?}, its max|J| {jm:.3e} vs \
                 global {global_max:.3e}; global rel {rel:.3e} (bar {bar:.3e}) per-pair \
                 {pp:.3e} of kappa_ij*u at {at:?} (bar {PER_PAIR_C})",
                unique[first],
            );
            (rel <= bar, pp <= PER_PAIR_C)
        };
        // 1e-6 (the review's mutant): BOTH bars catch it at C8 — the
        // smallest-|J| off-diagonal group still has max|J| ≈ 5e-3 of the global.
        assert_eq!(verdicts(1e-6), (false, false), "1e-6 W error not caught");
        // 1e-8: below the global bar's reach (1e-10 / 5e-3 ≈ 2e-8), so ONLY
        // the per-pair bar catches it.
        assert_eq!(verdicts(1e-8), (true, false), "per-pair bar not stronger");
    }

    /// Energies: the Cholesky pair fits and the frozen explicit-inverse
    /// oracle, each solved at the production rtol, agree to ≤1e-10 Ha (C8,
    /// Coulomb, ε = 1e-4, κ screen on — the reuse arm live).
    #[test]
    fn pair_fit_energy_matches_inverse_oracle_c8() {
        let su = setup("alkane_8.xyz", 8);
        let eps = 1e-4;
        let pre = pre_of(&su, Operator::coulomb(), eps, 0.7);
        let dcfg = production_maps(Some(1.0));
        let cfg = AmplitudeLmp2Config {
            eps,
            frozen_core: 8,
            ..Default::default()
        };
        let (no, nv) = (su.spaces.no, su.spaces.nv);
        let e = |path| {
            let (rg, st) = fit(&su, &pre, eps, &dcfg, path);
            let r = solve_and_count(
                &rg,
                &su.spaces.f_oo,
                &su.spaces.f_vv,
                su.e_hf,
                &cfg,
                &su.vvhv,
                no,
                nv,
                "t",
            )
            .unwrap();
            (r.e_corr, st.n_w_reused)
        };
        let (e_new, reused) = e(FitPath::Cholesky);
        let (e_old, _) = e(FitPath::InverseOracle);
        eprintln!(
            "STAGE-C C8 energy: chol {e_new:.12} oracle {e_old:.12} dE {:.3e} (reused {reused})",
            e_new - e_old
        );
        assert!(reused > 0);
        assert!((e_new - e_old).abs() <= 1e-10, "dE {:.3e}", e_new - e_old);
    }

    /// Stage 5 of `pre` under its pair-skip mode, with the Fock screen on.
    fn fit_skip(
        su: &Setup,
        pre: &StripsAndMetric,
        eps: f64,
        dcfg: &DirectConfig,
    ) -> (Ragged, DirectStats) {
        let mut st = DirectStats::default();
        let r = fit_pairs(
            pre,
            &su.dfbs,
            &su.spaces,
            eps,
            1.0,
            dcfg,
            FitPath::Cholesky,
            None,
            Some(DIRECT_FOCK_SCREEN_KAPPA),
            &mut st,
        )
        .unwrap();
        (r, st)
    }

    /// Bit-for-bit equality of two ragged systems: pair identity, index
    /// lists and masks, then every array's shape and the `to_bits` of every
    /// element, so `-0.0` against `0.0` or a NaN differs.
    fn same_ragged(a: &Ragged, b: &Ragged) -> bool {
        let same_bits = |x: &Array2<f64>, y: &Array2<f64>| {
            x.dim() == y.dim() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        };
        a.pairs.len() == b.pairs.len()
            && a.pairs.iter().zip(&b.pairs).all(|(x, y)| {
                (x.i, x.j, &x.da, &x.db, &x.pos_da, &x.pos_db, &x.pat)
                    == (y.i, y.j, &y.da, &y.db, &y.pos_da, &y.pos_db, &y.pat)
                    && same_bits(&x.fvv_aa, &y.fvv_aa)
                    && same_bits(&x.fvv_bb, &y.fvv_bb)
                    && same_bits(&x.j_blk, &y.j_blk)
                    && same_bits(&x.denom, &y.denom)
            })
    }

    /// The pair skip under an attenuated kernel (exactness anchor + rigor
    /// audit, CLAUDE.md protocol). Octane, short aux radius so the bound
    /// reaches pairs at this size, gate off (every pair fitted), Fock screen
    /// on. `On` skips pairs AND builds the bit-identical ragged system of
    /// `Off`; `Audit` finds every fitted pair below its bound (ratio ≤ 1) and
    /// no proved-empty pair with a block; the weakened-bound mutant breaks
    /// both.
    #[test]
    fn pair_skip_is_bitwise_and_its_audit_holds_under_erfc() {
        let su = setup("alkane_8.xyz", 8);
        // octane is short (pair distances up to ~19 Bohr), so the anchor runs
        // at a loose eps where the bound reaches pairs; production reach is
        // measured on C20-C48 (wiki/notes/lmp2-attenuated-pair-skip-2026-11-24)
        for (op, eps) in [(Operator::erfc(2.0), 1e-2), (Operator::erfc(1.0), 1e-2)] {
            let dcfg = DirectConfig {
                aux_radius_bohr: 4.0,
                ..production_maps(None)
            };
            let (mut pre, _) = pre_with(&su, op, eps, None, &dcfg);
            pre.pair_skip = PairSkip::Off;
            let (r_off, s_off) = fit_skip(&su, &pre, eps, &dcfg);
            pre.pair_skip = PairSkip::Audit;
            let (r_aud, s_aud) = fit_skip(&su, &pre, eps, &dcfg);
            pre.pair_skip = PairSkip::On;
            let (r_on, s_on) = fit_skip(&su, &pre, eps, &dcfg);
            let w = op.omega;
            assert!(
                same_ragged(&r_off, &r_aud),
                "w={w}: audit changed the system"
            );
            assert!(same_ragged(&r_off, &r_on), "w={w}: skip changed the system");
            assert!(
                s_on.n_pairs_proved_empty > 0,
                "w={w}: the skip reaches no pair (vacuous)"
            );
            // the prefilter (On only) prunes nothing the proof would take
            assert_eq!(s_on.n_pairs_proved_empty, s_aud.n_pairs_proved_empty);
            assert!(
                s_on.n_pair_skip_bounded < s_aud.n_pair_skip_bounded,
                "w={w}: the prefilter pruned no attempt ({} vs {})",
                s_on.n_pair_skip_bounded,
                s_aud.n_pair_skip_bounded
            );
            // prefilter mutants: it only removes attempts, so whatever it
            // decides, the system keeps its bits and nothing new is proved
            for m in [
                PrefilterMutant::RejectAll,
                PrefilterMutant::Invert,
                PrefilterMutant::AttemptAll,
            ] {
                pre.prefilter_mutant = m;
                let (r_m, s_m) = fit_skip(&su, &pre, eps, &dcfg);
                assert!(same_ragged(&r_off, &r_m), "w={w}: {m:?} changed the system");
                assert!(s_m.n_pairs_proved_empty <= s_aud.n_pairs_proved_empty);
                match m {
                    PrefilterMutant::RejectAll => assert_eq!(s_m.n_pairs_proved_empty, 0),
                    PrefilterMutant::Invert => assert_eq!(
                        s_m.n_pairs_proved_empty, 0,
                        "w={w}: a pair the cut rejects was proved (the cut is not a lower bound)"
                    ),
                    _ => assert_eq!(s_m.n_pairs_proved_empty, s_aud.n_pairs_proved_empty),
                }
            }
            pre.prefilter_mutant = PrefilterMutant::Normal;
            // a mutant that weakens the bound CONSISTENTLY: drop the γ-free
            // terms u_A v_A + u_B v_B from the On cutoff and from the bound.
            // What is left (the γ terms) sits ~1e5 below the fit on the
            // nearest bounded pairs (R ≈ 2.3 Bohr, max|g| ~ 5e-3, mutant
            // bound ~ 1e-7), so at eps 1e-5 it "proves" kept pairs empty:
            // On must change the system and the audit must count them
            let eps_m = 1e-5;
            let (mut pm, _) = pre_with(&su, op, eps_m, None, &dcfg);
            pm.pair_skip = PairSkip::Off;
            let (r0, _) = fit_skip(&su, &pm, eps_m, &dcfg);
            pm.pair_skip = PairSkip::Audit;
            let (_, s0) = fit_skip(&su, &pm, eps_m, &dcfg);
            assert_eq!(s0.pair_skip_audit_violations, 0, "w={w} eps 1e-5");
            assert!(s0.pair_skip_audit_max_ratio <= 1.0, "w={w} eps 1e-5");
            pm.drop_free_mutant = true;
            pm.prefilter_mutant = PrefilterMutant::AttemptAll;
            pm.pair_skip = PairSkip::On;
            let (rm, _) = fit_skip(&su, &pm, eps_m, &dcfg);
            pm.pair_skip = PairSkip::Audit;
            let (_, sm) = fit_skip(&su, &pm, eps_m, &dcfg);
            let differs = !same_ragged(&r0, &rm);
            eprintln!(
                "w={w}: drop-free mutant: ragged differs {differs}, audit violations {}, \
                 max ratio {:.3e} (unmutated {:.3e})",
                sm.pair_skip_audit_violations,
                sm.pair_skip_audit_max_ratio,
                s0.pair_skip_audit_max_ratio
            );
            assert!(
                differs && sm.pair_skip_audit_violations > 0,
                "w={w}: the drop-free mutant was not caught"
            );
            assert!(sm.pair_skip_audit_max_ratio > 1.0);
            assert!(
                s_on.n_pairs_proved_empty <= s_off.n_pairs_empty,
                "w={w}: more proved than empty"
            );
            assert!(s_aud.pair_skip_audit_n > 0);
            assert_eq!(s_aud.pair_skip_audit_violations, 0, "w={w}");
            assert!(
                s_aud.pair_skip_audit_max_ratio <= 1.0,
                "w={w}: {}",
                s_aud.pair_skip_audit_max_ratio
            );
            assert!(s_aud.pair_skip_audit_gate_max_ratio <= 1.0, "w={w}");
            assert!(s_aud.pair_skip_audit_max_ratio > 0.0);
            // mutant: every factor and gamma 1e6 too small. The audit must
            // see computed elements above the bound; the anchor breaks only
            // if a nonempty pair is bounded at all (octane's close pairs
            // have gamma >= 1 or an unbounded floor, so it may not)
            pre.bound_mutant = 1e-6;
            pre.pair_skip = PairSkip::On;
            let (r_mut, _) = fit_skip(&su, &pre, eps, &dcfg);
            let anchor_caught = !same_ragged(&r_off, &r_mut);
            pre.pair_skip = PairSkip::Audit;
            let (_, s_mut) = fit_skip(&su, &pre, eps, &dcfg);
            assert!(
                s_mut.pair_skip_audit_max_ratio > 1.0,
                "w={w}: weakened bound not caught by the audit (anchor caught: {anchor_caught})"
            );
            assert!(
                !anchor_caught || s_mut.pair_skip_audit_violations > 0,
                "w={w}: the anchor broke but the audit saw no violation"
            );
            pre.bound_mutant = 1.0;
        }
    }

    /// A proof that does not fit the memory budget beside the resident set
    /// is not attempted: the run fits every pair and builds `Off`'s system
    /// bit for bit. With the same budget plus the proof's own bytes the
    /// proof runs (the control: the refusal is the budget, not the setup).
    #[test]
    fn pair_skip_proof_over_the_budget_fits_every_pair() {
        let su = setup("alkane_8.xyz", 8);
        let (op, eps) = (Operator::erfc(2.0), 1e-2);
        let dcfg = DirectConfig {
            aux_radius_bohr: 4.0,
            ..production_maps(None)
        };
        let (mut pre, _) = pre_with(&su, op, eps, None, &dcfg);
        pre.pair_skip = PairSkip::Off;
        let (r_off, s_off) = fit_skip(&su, &pre, eps, &dcfg);
        pre.pair_skip = PairSkip::On;
        let n_pairs = kept_pairs(&pre.keep, su.spaces.no).len();
        let proof = pair_proof_bytes(&pre, su.dfbs.shell_dims(), n_pairs);
        // fit_skip's stats carry no stage-3 estimate, so nothing is resident
        // in the gate's accounting and the proof's own bytes decide
        let held = s_off.memory.strips_bytes + s_off.memory.metric_bytes;
        assert_eq!(held, 0);
        let blocks: usize = r_off
            .pairs
            .iter()
            .map(|b| 8 * (b.fvv_aa.len() + b.fvv_bb.len() + b.j_blk.len() + b.denom.len()))
            .sum();
        let room = 1usize << 30;
        eprintln!(
            "proof {proof} B, blocks {blocks} B, group scratch {} B",
            s_off.group_scratch_bound_bytes
        );
        let run = |budget_bytes: usize| {
            let mut st = DirectStats::default();
            let gate = Some(SolveGate { budget_bytes });
            let r = fit_pairs(
                &pre,
                &su.dfbs,
                &su.spaces,
                eps,
                1.0,
                &dcfg,
                FitPath::Cholesky,
                gate,
                Some(DIRECT_FOCK_SCREEN_KAPPA),
                &mut st,
            )
            .unwrap();
            (r, st)
        };
        let (r_fit, s_fit) = run(held + proof + room);
        assert!(
            s_fit.n_pairs_proved_empty > 0,
            "control: the proof did not run"
        );
        assert!(same_ragged(&r_off, &r_fit));
        let (r_over, s_over) = run(held + proof - 1);
        assert_eq!(
            s_over.n_pair_skip_bounded, 0,
            "the proof ran over the budget"
        );
        assert!(
            same_ragged(&r_off, &r_over),
            "the refused proof changed the system"
        );
    }

    /// Coulomb: the bound still holds (audit) but proves nothing at octane.
    #[test]
    fn pair_skip_audit_holds_for_coulomb() {
        let su = setup("alkane_8.xyz", 8);
        let eps = 1e-4;
        let dcfg = DirectConfig {
            aux_radius_bohr: 4.0,
            ..production_maps(None)
        };
        let (mut pre, _) = pre_with(&su, Operator::coulomb(), eps, None, &dcfg);
        pre.pair_skip = PairSkip::Audit;
        let (_, st) = fit_skip(&su, &pre, eps, &dcfg);
        assert_eq!(st.pair_skip_audit_violations, 0);
        assert!(
            st.pair_skip_audit_max_ratio <= 1.0,
            "{}",
            st.pair_skip_audit_max_ratio
        );
    }
}

/// The memory gate of the direct LMP2 path: the pre-flight is exact at ε = 0,
/// an over-budget run refuses with a typed error (up front at ε = 0, during the
/// pair fits at ε > 0), an ample budget is bit-identical to the ungated
/// assembly, and the stage-3 slab blocking the budget drives leaves the energy
/// unchanged. Every test runs on a pinned 2-worker pool.
#[cfg(test)]
mod memory_gate_tests {
    use super::*;
    use crate::lmp2_amplitude::{checked_vvhv, solve_and_count};
    use crate::ragged::amplitude_vector_bytes;
    use ferric_core::parallel::ParallelContext;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    struct Case {
        mol: Molecule,
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        rhf: ScfResult,
        vvhv: VvHv,
        nc: usize,
    }

    fn case(nc: usize) -> Case {
        let mol = Molecule::load_xyz(&format!(
            "{}/../../testdata/molecules/alkane_{nc}.xyz",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let obs_bs = ferric_core::basis::bundled("6-31g").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
        let rcfg = RhfConfig::default().with_energy_conv(1e-10);
        let rhf = solve_rhf(
            &ParallelContext::default(),
            &mol,
            &obs,
            Operator::coulomb(),
            &bounds,
            &rcfg,
        )
        .unwrap();
        let (vvhv, _) = checked_vvhv(&mol, &obs, &obs_bs, &rhf, "test").unwrap();
        Case {
            mol,
            obs,
            dfbs,
            rhf,
            vvhv,
            nc,
        }
    }

    /// Production-like maps with a virtual radius short enough that the
    /// candidate sets differ between pairs (so the ε > 0 bound is loose).
    fn maps() -> DirectConfig {
        DirectConfig {
            aux_radius_bohr: 10.0,
            virt_radius_bohr: Some(8.0),
            ao_tail: 1e-3,
            schwarz_skip: 1e-5,
            ..Default::default()
        }
    }

    /// [`maps`] with a 1 MiB stage-3 slab cap. Under a gate the slab cap is
    /// also bounded by what the budget leaves beside the strips, and slab
    /// blocking changes the strips at round-off; a cap below every budget
    /// these tests use keeps the stage-3 blocking, and so the bits, the same
    /// across budgets, so a bit-identity check isolates the stage-5 schedule.
    fn fixed_slab_maps() -> DirectConfig {
        DirectConfig {
            slab_budget_bytes: 1 << 20,
            ..maps()
        }
    }

    fn lcfg(c: &Case, eps: f64, budget: usize) -> AmplitudeLmp2Config {
        AmplitudeLmp2Config {
            eps,
            frozen_core: c.nc,
            eri3_budget_bytes: Some(budget),
            ..Default::default()
        }
    }

    fn pinned<T: Send>(f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(f)
    }

    fn run(
        c: &Case,
        eps: f64,
        budget: usize,
        dcfg: &DirectConfig,
    ) -> Result<(AmplitudeLmp2Result, DirectStats), FerricError> {
        pinned(|| {
            amplitude_lmp2_direct_with_virtuals(
                &c.mol,
                &c.obs,
                &c.dfbs,
                Operator::coulomb(),
                &c.rhf,
                &lcfg(c, eps, budget),
                dcfg,
                &c.vvhv,
            )
        })
    }

    /// The ungated assembly (the historical code path) and its ragged system.
    fn ungated(c: &Case, eps: f64) -> (Ragged, f64) {
        pinned(|| {
            // the production localization (λ-Boys at the config's λ)
            let lam = lcfg(c, eps, AMPLE).lambda_boys_spec();
            let spaces =
                localized_spaces_lambda(&c.mol, &c.obs, &c.rhf, c.nc, &c.vvhv, lam).unwrap();
            let (rg, _, _) = assemble_ragged_direct_local(
                &c.mol,
                &c.obs,
                &c.dfbs,
                Operator::coulomb(),
                &spaces,
                eps,
                1.0,
                None,
                &maps(),
            )
            .unwrap();
            let r = solve_and_count(
                &rg,
                &spaces.f_oo,
                &spaces.f_vv,
                c.rhf.energy,
                &lcfg(c, eps, AMPLE),
                &c.vvhv,
                spaces.no,
                spaces.nv,
                "test",
            )
            .unwrap();
            (rg, r.e_corr)
        })
    }

    const AMPLE: usize = 1 << 40;

    fn exhausted(r: Result<(AmplitudeLmp2Result, DirectStats), FerricError>) -> String {
        match r {
            Err(FerricError::MemoryExhausted(m)) => m,
            Err(e) => panic!("expected MemoryExhausted, got {e}"),
            Ok((r, _)) => panic!("expected MemoryExhausted, ran to E = {}", r.e_corr),
        }
    }

    /// ε = 0: the pre-flight's PCG vector is the solve's to the byte and its
    /// pair-block bytes are the ragged system's to within the index maps and
    /// Vec capacity slack; an ample budget is bit-identical to the ungated
    /// assembly.
    #[test]
    fn eps_zero_estimate_is_exact_and_ample_budget_is_inert() {
        let c = case(6);
        let (rg, e_ungated) = ungated(&c, 0.0);
        let (r, st) = run(&c, 0.0, AMPLE, &maps()).unwrap();
        let est = st.memory;
        assert!(est.exact);
        assert_eq!(est.pcg_vector_bytes, amplitude_vector_bytes(&rg));
        let heap = rg.heap_bytes();
        let rel = (heap as f64 - est.pair_block_bytes as f64).abs() / heap as f64;
        eprintln!(
            "C6 eps=0: est pair blocks {} vs heap {heap} (rel {rel:.2e}), vector {}, peak {}",
            est.pair_block_bytes,
            est.pcg_vector_bytes,
            est.resident_peak()
        );
        assert!(rel < 0.02, "pair-block estimate off by {rel:.3e}");
        assert_eq!(r.e_corr.to_bits(), e_ungated.to_bits());
    }

    /// ε > 0 under an ample budget: bit-identical to the ungated assembly.
    #[test]
    fn finite_eps_ample_budget_is_bit_identical_to_ungated() {
        let c = case(6);
        let (_, e_ungated) = ungated(&c, 1e-3);
        let (r, _) = run(&c, 1e-3, AMPLE, &maps()).unwrap();
        assert_eq!(r.e_corr.to_bits(), e_ungated.to_bits());
    }

    /// ε = 0 one byte under the exact estimate refuses before any integral;
    /// 2% over it (the estimate's slack against the heap count) still runs.
    #[test]
    fn eps_zero_over_budget_refuses_up_front() {
        let c = case(6);
        let (r0, st) = run(&c, 0.0, AMPLE, &maps()).unwrap();
        let peak = st.memory.resident_peak();
        let msg = exhausted(run(&c, 0.0, peak - 1, &maps()));
        assert!(msg.contains("before any integral"), "{msg}");
        assert!(msg.contains("budget_gb") && msg.contains("eps"), "{msg}");
        let (r1, _) = run(&c, 0.0, peak + peak / 50, &maps()).unwrap();
        assert_eq!(r1.e_corr.to_bits(), r0.e_corr.to_bits());
    }

    /// ε > 0: the candidate bound is loose, so the pre-flight must NOT refuse
    /// on it (a budget at the true footprint runs) and the charge-as-built
    /// gate must stop a budget under the true footprint.
    #[test]
    fn finite_eps_charge_gate_refuses_only_the_true_overshoot() {
        let c = case(6);
        let (r0, st) = run(&c, 1e-3, AMPLE, &fixed_slab_maps()).unwrap();
        let est = st.memory;
        assert!(!est.exact);
        let base = est.strips_bytes + est.metric_bytes + st.own_w_bytes_held;
        // Everything the gate charges, at its true size: an upper bound on
        // both the stage-5 resident set and the solve's.
        let truth = base + st.group_scratch_bound_bytes + r0.solve_resident_bytes;
        assert!(
            est.resident_peak() > truth,
            "the bound {} must be loose against the true footprint {truth} for this test \
             to reach the charge gate",
            est.resident_peak()
        );
        let (r1, s1) = run(&c, 1e-3, truth, &fixed_slab_maps()).unwrap();
        assert_eq!(r1.e_corr.to_bits(), r0.e_corr.to_bits());
        // The candidate bound does not fit beside two groups' scratch: stage
        // 5 ran in budget-sized waves, with the same blocks.
        assert_eq!(st.group_concurrency, 2);
        assert_eq!(st.n_group_waves, 1);
        assert!(
            s1.n_group_waves >= 2,
            "group blocking never engaged: {s1:?}"
        );
        let solve_only = r0.solve_resident_bytes;
        let tight = base.max(solve_only / 2);
        let msg = exhausted(run(&c, 1e-3, tight, &fixed_slab_maps()));
        assert!(msg.contains("during the pair fits"), "{msg}");
    }

    /// A wave takes groups while their block bounds plus `min(len, workers)`
    /// copies of the largest scratch fit, always at least one group, and says
    /// whether the budget (not the list's end) stopped it.
    #[test]
    fn plan_group_wave_packs_groups_into_the_budget() {
        let scratch = [10, 30, 20, 10];
        let blocks = [100, 100, 100, 100];
        // everything: 400 + 2·30
        let all = plan_group_wave(0, &scratch, &blocks, 2, 460);
        assert_eq!(
            all,
            GroupWave {
                end: 4,
                scratch: 60,
                budget_limited: false
            }
        );
        // two groups: 200 + 2·30 = 260 fits, three (300 + 60) does not
        let two = plan_group_wave(0, &scratch, &blocks, 2, 300);
        assert_eq!(
            two,
            GroupWave {
                end: 2,
                scratch: 60,
                budget_limited: true
            }
        );
        // nothing fits: still one group, its own scratch reserved
        let one = plan_group_wave(1, &scratch, &blocks, 4, 0);
        assert_eq!(
            one,
            GroupWave {
                end: 2,
                scratch: 30,
                budget_limited: true
            }
        );
        // the last group alone
        let last = plan_group_wave(3, &scratch, &blocks, 4, 0);
        assert_eq!(
            last,
            GroupWave {
                end: 4,
                scratch: 10,
                budget_limited: false
            }
        );
    }

    /// ε > 0 with the budget at the strips + metric + candidate bound of the
    /// pair blocks + one group's scratch bound. The previous schedule
    /// reserved the whole candidate bound up front, which leaves room for one
    /// group's scratch and ran stage 5 one group at a time (the C32 audit: a
    /// 13.2 GB bound under the default budget). The waves reserve only their
    /// own groups' block bounds beside the blocks actually built, so both
    /// workers run groups, with the same blocks.
    #[test]
    fn finite_eps_groups_run_concurrently_under_the_candidate_bound() {
        let c = case(6);
        let (r0, st) = run(&c, 1e-3, AMPLE, &fixed_slab_maps()).unwrap();
        assert_eq!(st.group_concurrency, 2);
        assert_eq!(st.n_group_waves, 1);
        let est = st.memory;
        assert!(!est.exact);
        let base = est.strips_bytes + est.metric_bytes + st.own_w_bytes_held;
        let g = st.group_scratch_bound_bytes;
        let budget = base + est.pair_block_bytes + g;
        // what a schedule that reserves the whole bound would have left
        assert!(budget - base - est.pair_block_bytes < 2 * g);
        let (r1, s1) = run(&c, 1e-3, budget, &fixed_slab_maps()).unwrap();
        assert_eq!(r1.e_corr.to_bits(), r0.e_corr.to_bits());
        // The bound does not fit beside the scratch, so the schedule ran in
        // waves, and none of them was cut to one group.
        assert!(s1.n_group_waves >= 2, "{s1:?}");
        assert_eq!(s1.group_concurrency, 2, "{s1:?}");
        assert!((1..=2).contains(&s1.group_in_flight_peak), "{s1:?}");
    }

    /// The stage-3 slab blocking: a one-byte scratch cap (one aux shell per
    /// slab) gives the energy of the default cap, through more slabs.
    #[test]
    fn stage3_slab_blocking_matches_unblocked() {
        let c = case(6);
        let (r0, s0) = run(&c, 1e-3, AMPLE, &maps()).unwrap();
        let tiny = DirectConfig {
            scratch_budget_bytes: 1,
            ..maps()
        };
        let (r1, s1) = run(&c, 1e-3, AMPLE, &tiny).unwrap();
        eprintln!(
            "slabs: default {} vs tiny {}",
            s0.n_eri3_slabs, s1.n_eri3_slabs
        );
        assert!(s1.n_eri3_slabs > s0.n_eri3_slabs, "blocking never engaged");
        assert!(
            (r1.e_corr - r0.e_corr).abs() <= 1e-10,
            "blocked {:.15} vs unblocked {:.15}",
            r1.e_corr,
            r0.e_corr
        );
    }

    /// The budget lowers the stage-3 scratch to what it leaves beside the
    /// strips, and never raises it.
    #[test]
    fn stage3_scratch_follows_the_budget() {
        assert_eq!(stage3_scratch(2 << 30, AMPLE, 1000), 2 << 30);
        assert_eq!(stage3_scratch(2 << 30, 5000, 1000), 4000);
        assert_eq!(stage3_scratch(2 << 30, 500, 1000), 1);
    }
}
