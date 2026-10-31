//! Integral-direct assembly for the amplitude-threshold local MP2 family —
//! the missing piece named by the wall-clock measurement in
//! `wiki/amplitude-threshold-lmp2.md` §20–21: the global 3-index tensor
//! (`eri3_mo_ov_blocked`, `(naux, no·nv)`) and the N⁵ whitening GEMM behind
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

use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
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
    checked_vvhv, localized_spaces, pair_gate_keep, solve_and_count, AmplitudeLmp2Config,
    AmplitudeLmp2Result, LocalizedSpaces, StageTimings, VvHv,
};
use crate::ragged::{pair_block_from_g_cand, PairBlock, Ragged};
use crate::rimp2::{ri_mp2, RiMp2Config};

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
    /// GLOBAL integral-slab scratch cap (bytes), shared across all
    /// concurrently-running batch workers — each worker slabs under
    /// budget/n_workers, so the co-resident scratch never exceeds this
    /// (the ferric-batch N×-overcommit lesson: a per-worker budget is a
    /// budget on nothing).
    pub scratch_budget_bytes: usize,
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
    /// `batch_merge > 1`, which has no meaning here.
    AuxMajor,
}

impl Default for DirectConfig {
    fn default() -> Self {
        Self {
            aux_radius_bohr: 10.0,
            virt_radius_bohr: None,
            ao_tail: 0.0,
            schwarz_skip: 0.0,
            batch_merge: 1,
            scratch_budget_bytes: 2usize << 30,
            virt_schwarz_kappa: None,
            split_eri3_timers: false,
            eri3_pass: Eri3Pass::Batches,
        }
    }
}

/// Counters + stage timings of one direct assembly — the honest-measurement
/// surface: strip saturation is the locality claim, so it is reported, not
/// inferred.
#[derive(Debug, Clone, Default)]
pub struct DirectStats {
    /// Per-occupied strip aux rows (functions), mean/max.
    pub strip_rows_mean: f64,
    pub strip_rows_max: usize,
    /// Per-occupied strip virtual columns, mean/max.
    pub strip_cols_mean: f64,
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
    /// Pair virtual-candidate columns |C_ij| after the Schwarz screen
    /// (mean/max over unique surviving pairs; equals the V_i ∪ V_j union
    /// sizes when `virt_schwarz_kappa` is off).
    pub virt_cand_mean: f64,
    pub virt_cand_max: usize,
    pub t_maps_s: f64,
    /// Stage-3 wall seconds (all batch workers).
    pub t_eri3_s: f64,
    /// Stage-3 split, wall seconds measured on each batch worker's thread
    /// and summed over workers (thread-seconds; NOT CPU time — a descheduled
    /// worker still accrues): the integral fill (libint evaluation +
    /// scatter into the slab) and the half-transform into the strips. Their
    /// sum is below the workers' total by the per-batch setup.
    pub t_eri3_fill_thread_s: f64,
    /// The fill split into evaluation and scatter: only with
    /// [`DirectConfig::split_eri3_timers`] (NaN otherwise). The clock is read
    /// per 64-shell-pair chunk, not per triple.
    pub t_eri3_eval_thread_s: f64,
    pub t_eri3_scatter_thread_s: f64,
    pub t_eri3_transform_thread_s: f64,
    /// The serial prefix of `t_eri3_s`, wall seconds: the construction check
    /// and the shell-pair / aux Schwarz factors, before any batch worker runs.
    pub t_eri3_setup_s: f64,
    /// Integral slabs filled, over all batches.
    pub n_eri3_slabs: usize,
    /// Half-transform floating-point operations actually done
    /// (`Σ_i Σ_{P∈slab} 2·nsf·nnf + 2·nnf·|V_i|`, batch-union axes), and what
    /// the same contraction would cost restricted to i's OWN occupied and
    /// virtual AO support (a counter for the sparsity the transform does not
    /// exploit; batch pass only). On the aux-major pass `transform_flops` counts
    /// its own contraction and the second counter stays 0.
    pub transform_flops: f64,
    pub transform_flops_own_support: f64,
    /// Integrals libint wrote, and shell triples it screened to nothing. Only
    /// with [`DirectConfig::split_eri3_timers`] (0 otherwise).
    pub n_eri3_integrals: u64,
    pub n_eri3_null_blocks: u64,
    /// [`Eri3Pass::AuxMajor`]: largest one-task scratch (slab + half-transformed
    /// slab + occupied coefficients), and how many tasks the scratch budget lets
    /// run at once. 0 on the batch pass.
    pub aux_major_task_bytes_max: usize,
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
    pub batch_work_max_s: f64,
    pub batch_work_mean_s: f64,
    pub t_metric_s: f64,
    pub t_pairs_s: f64,
    /// Stage-5 pair-domain groups (one V_DD Cholesky each).
    pub n_pair_groups: usize,
    /// Mean distinct occupieds over a group's pairs (each is whitened once
    /// per group, `W_k = L⁻¹ A_k`, instead of once per pair).
    pub n_distinct_occ_per_group: f64,
    /// Stage-5 triangular solves (multi-RHS): one per occupied for the κ
    /// screen when it is on, plus one per (group, distinct occupied) not
    /// reused from it.
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
    /// Largest single-group scratch, bytes: `8·(2d² + Σ_k d·|C_k|)` (V_DD,
    /// its Cholesky factor, and one whitened `W_k` per distinct occupied k
    /// over its candidate union C_k). Each rayon worker holds one group's
    /// scratch at a time, so the co-resident group scratch is at most
    /// (worker threads) × this; it is NOT under `scratch_budget_bytes`.
    pub group_scratch_bytes_max: usize,
    /// Boys localization + localized-space setup (`localized_spaces`),
    /// wall seconds; set by the driver (0 when the assembly is called
    /// directly).
    pub t_loc_s: f64,
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

fn sorted_union(lists: impl IntoIterator<Item = impl IntoIterator<Item = usize>>) -> Vec<usize> {
    let set: BTreeSet<usize> = lists.into_iter().flatten().collect();
    set.into_iter().collect()
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
fn gather_vdd(
    dsh: &[usize],
    dims_df: &[usize],
    metric_blocks: &MetricBlocks,
    ctx: &str,
) -> Result<Array2<f64>, FerricError> {
    let d: usize = dsh.iter().map(|&sh| dims_df[sh]).sum();
    let mut vdd = Array2::<f64>::zeros((d, d));
    let mut loc_off = Vec::with_capacity(dsh.len());
    let mut acc = 0usize;
    for &sh in dsh {
        loc_off.push(acc);
        acc += dims_df[sh];
    }
    for (bx, &sp) in dsh.iter().enumerate() {
        for (by, &sq) in dsh.iter().enumerate() {
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
            for p in 0..dims_df[sp] {
                for q in 0..dims_df[sq] {
                    let v = if transposed { blk[(q, p)] } else { blk[(p, q)] };
                    vdd[(loc_off[bx] + p, loc_off[by] + q)] = v;
                }
            }
        }
    }
    Ok(vdd)
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

/// `L` with `V = L Lᵀ` (lower Cholesky factor) of a gathered `V_DD`.
fn chol_lower(vdd: &Array2<f64>, what: &str) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{Cholesky, UPLO};
    vdd.cholesky(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("lmp2_direct V_DD Cholesky ({what}): {e}")))
}

/// `W = L⁻¹ A` (one triangular solve with many right-hand sides).
fn whiten(l: &Array2<f64>, a: &Array2<f64>, what: &str) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{Diag, SolveTriangular, UPLO};
    l.solve_triangular(UPLO::Lower, Diag::NonUnit, a)
        .map_err(|e| FerricError::General(format!("lmp2_direct triangular solve ({what}): {e}")))
}

/// Rows `dfuncs` of occupied `k`'s strip, over the global virtuals `cols`
/// (`None`: every strip column, in strip order). A row or column missing
/// from the strip is a construction bug (the extended domains must cover
/// every pair domain and candidate set) and errors loudly.
fn gather_strip(
    strips: &[Strip],
    k: usize,
    dfuncs: &[usize],
    cols: Option<&[usize]>,
    what: &str,
) -> Result<Array2<f64>, FerricError> {
    let st = &strips[k];
    let ncols = cols.map_or(st.cols.len(), <[usize]>::len);
    let mut a = Array2::<f64>::zeros((dfuncs.len(), ncols));
    for (r, &gp) in dfuncs.iter().enumerate() {
        let sr = st.row_pos[gp];
        if sr == usize::MAX {
            return Err(FerricError::General(format!(
                "lmp2_direct: strip {k} missing aux row {gp} for {what} — extended domain does \
                 not cover the fit domain (construction bug)"
            )));
        }
        match cols {
            None => a.row_mut(r).assign(&st.b.row(sr)),
            Some(cols) => {
                for (c, &av) in cols.iter().enumerate() {
                    let sc = st.col_pos[av];
                    if sc == usize::MAX {
                        return Err(FerricError::General(format!(
                            "lmp2_direct: strip {k} missing virtual {av} for {what}"
                        )));
                    }
                    a[(r, c)] = st.b[(sr, sc)];
                }
            }
        }
    }
    Ok(a)
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
/// as its q is taken (one worker's `W` at a time).
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
) -> Result<OwnWhitened, FerricError> {
    use rayon::prelude::*;
    let sq = scale.abs().sqrt();
    type PerI = (Vec<f64>, f64, Option<Array2<f64>>);
    let per_i: Vec<Result<PerI, FerricError>> = (0..no)
        .into_par_iter()
        .map(|i| {
            let dfuncs = shell_funcs(dfbs, &aux_dom[i]);
            if dfuncs.is_empty() {
                return Err(FerricError::General(format!(
                    "lmp2_direct: empty aux domain for occupied {i} at radius {aux_radius_bohr} \
                     Bohr (virt_schwarz_kappa needs a nonempty D_i)"
                )));
            }
            let what = format!("occupied {i} Schwarz diagonal");
            let vdd = gather_vdd(&aux_dom[i], dims_df, metric_blocks, &what)?;
            let l = chol_lower(&vdd, &what)?;
            let w = whiten(&l, &gather_strip(strips, i, &dfuncs, None, &what)?, &what)?;
            let q: Vec<f64> = w
                .columns()
                .into_iter()
                .map(|c| sq * c.dot(&c).sqrt())
                .collect();
            let qm = q.iter().copied().fold(0.0f64, f64::max);
            Ok((q, qm, keep_w.then_some(w)))
        })
        .collect();
    let mut out = OwnWhitened {
        q: Vec::with_capacity(no),
        qmax: Vec::with_capacity(no),
        w: keep_w.then(|| Vec::with_capacity(no)),
    };
    for r in per_i {
        let (q, qm, w) = r?;
        out.q.push(q);
        out.qmax.push(qm);
        if let (Some(ws), Some(w)) = (out.w.as_mut(), w) {
            ws.push(w);
        }
    }
    Ok(out)
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

/// One domain group's work counters (summed into [`DirectStats`]).
#[derive(Debug, Clone, Copy, Default)]
struct GroupCounts {
    /// Distinct occupieds over the group's pairs.
    n_occ: usize,
    /// Triangular solves run (one per distinct occupied not reused).
    n_trsm: usize,
    /// Distinct occupieds whose `W` came from the own-domain κ whitening.
    n_reused: usize,
    /// This group's scratch bytes, `8·(2d² + Σ_k d·|C_k|)`.
    scratch_bytes: usize,
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
    nv: usize,
    aux_radius_bohr: f64,
    dfbs: &'a PreparedBasis,
    /// Each occupied's distance domain D_i (shell list), for the reuse test.
    aux_dom: &'a [Vec<usize>],
    /// Own-domain whitened strips from the κ screen, trimmed to their
    /// reusable columns (`None` when the screen is off or over budget;
    /// `Some(None)` entries where no group has domain exactly D_i).
    own_w: Option<&'a [Option<GroupW>]>,
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
        let pos: Vec<usize> = cij
            .iter()
            .map(|a| {
                self.cols
                    .binary_search(a)
                    .expect("C_ij inside the group column union (construction)")
            })
            .collect();
        self.w.select(Axis(1), &pos)
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
    let mut out = Vec::with_capacity(2);
    if let Some(pb) =
        pair_block_from_g_cand(i, j, &g, cij, ctx.nv, ctx.f_vv, ctx.fo, ctx.fv, ctx.eps)
    {
        out.push(pb);
    }
    if i != j {
        let gt = g.t().to_owned();
        if let Some(pb) =
            pair_block_from_g_cand(j, i, &gt, cij, ctx.nv, ctx.f_vv, ctx.fo, ctx.fv, ctx.eps)
        {
            out.push(pb);
        }
    }
    out
}

/// Whitened strips `W_k = L⁻¹ A_k[D, C_k]` for every distinct occupied `k`
/// of a group, over the union `C_k` of its pairs' candidate sets. A `k`
/// whose own domain D_k IS the group domain takes the columns of its
/// κ-screen `W` instead of a new solve.
fn whiten_group_occupieds(
    ctx: &PairFitCtx<'_>,
    dsh: &[usize],
    dfuncs: &[usize],
    l: &Array2<f64>,
    pxs: &[usize],
) -> Result<(HashMap<usize, GroupW>, GroupCounts), FerricError> {
    let mut want: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for &px in pxs.iter().filter(|&&px| !ctx.pair_cands[px].is_empty()) {
        let (i, j) = ctx.unique[px];
        for k in [i, j] {
            want.entry(k)
                .or_default()
                .extend(ctx.pair_cands[px].iter().copied());
        }
    }
    let mut counts = GroupCounts::default();
    let mut ws = HashMap::with_capacity(want.len());
    for (k, cols) in want {
        let cols: Vec<usize> = cols.into_iter().collect();
        counts.n_occ += 1;
        let reuse = match ctx.own_w {
            Some(own) if ctx.aux_dom[k] == dsh => own[k].as_ref(),
            _ => None,
        };
        let w = match reuse {
            Some(gw) => {
                counts.n_reused += 1;
                gw.select(&cols)
            }
            None => {
                counts.n_trsm += 1;
                let what = format!("occupied {k} in a group of {} pairs", pxs.len());
                whiten(
                    l,
                    &gather_strip(ctx.strips, k, dfuncs, Some(&cols), &what)?,
                    &what,
                )?
            }
        };
        counts.scratch_bytes += 8 * w.len();
        ws.insert(k, GroupW { cols, w });
    }
    #[cfg(test)]
    if let Some((first, rel)) = ctx.w_mutant {
        if pxs[0] == first {
            let k = ctx.unique[first].0;
            if let Some(gw) = ws.get_mut(&k) {
                gw.w.mapv_inplace(|x| x * (1.0 + rel));
            }
        }
    }
    Ok((ws, counts))
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
    let vdd = gather_vdd(dsh, ctx.dims_df, ctx.metric_blocks, &what)?;
    let l = chol_lower(&vdd, &what)?;
    let (ws, mut counts) = whiten_group_occupieds(ctx, dsh, &dfuncs, &l, pxs)?;
    counts.scratch_bytes += 8 * (vdd.len() + l.len());
    let mut out_group: SlotOut = Vec::with_capacity(pxs.len());
    for &px in pxs {
        let (i, j) = ctx.unique[px];
        let cij = &ctx.pair_cands[px];
        if cij.is_empty() {
            out_group.push((px, Vec::new()));
            continue;
        }
        let w_i = ws[&i].select(cij);
        let g = if i == j {
            w_i.t().dot(&w_i)
        } else {
            w_i.t().dot(&ws[&j].select(cij))
        };
        out_group.push((px, mask_pair(ctx, i, j, g, cij)));
    }
    Ok((out_group, counts))
}

/// Cauchy–Schwarz factors for the stage-3 batch triple cut, same-op
/// kernel: |(P|μν)| ≤ √(P|P) · Q(μν), with the aux factor per aux SHELL
/// (√ of the max (P|P) diagonal over the shell — conservative at shell
/// granularity; dropping it makes the cut non-conservative wherever
/// √(P|P) > 1, found in review before any merge). `None` when the cut is
/// off. Schwarz bounds exist for Coulomb/erf/erfc/terfc only, so the error
/// names the `schwarz_skip = 0.0` fix; terfc factors carry the table-error
/// inflation of `schwarz::bound_inflation`.
fn schwarz_triple_factors(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
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
    Ok(Some((sb.q, qp_shell)))
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
    /// Per-worker share of the GLOBAL scratch budget, bytes.
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
/// the occupied column on `s`, and the (nnf, |virt_ext_i|) virtual block.
fn batch_coefficients(
    ctx: &BatchCtx<'_>,
    ax: &BatchAxes,
    is: &[usize],
) -> (Vec<Array1<f64>>, Vec<Array2<f64>>) {
    let dims_obs = ctx.obs.shell_dims();
    let offs_obs = ctx.obs.shell_offsets();
    let c_occ: Vec<Array1<f64>> = is
        .iter()
        .map(|&i| {
            let mut c = Array1::<f64>::zeros(ax.nsf);
            for &sh in &ctx.occ_supp[i] {
                for f in 0..dims_obs[sh] {
                    let g = offs_obs[sh] + f;
                    c[ax.s_pos[g]] = ctx.spaces.c_locc[(g, i)];
                }
            }
            c
        })
        .collect();
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
/// row P of strip i is `c_occ_iᵀ · T_P · C_virt_i`.
#[allow(clippy::too_many_arguments)]
fn transform_slab(
    ctx: &BatchCtx<'_>,
    is: &[usize],
    strips: &mut [Strip],
    c_occ: &[Array1<f64>],
    c_virt: &[Array2<f64>],
    p_local: &[usize],
    t_slab: &Array3<f64>,
) {
    let dims_df = ctx.dfbs.shell_dims();
    let offs_df = ctx.dfbs.shell_offsets();
    for (k, &i) in is.iter().enumerate() {
        let strip = &mut strips[k];
        let ci = &c_occ[k];
        let cv = &c_virt[k];
        for &sp in &ctx.aux_ext[i] {
            let p0 = offs_df[sp];
            if p_local[p0] == usize::MAX {
                continue; // shell not in this slab
            }
            for f in 0..dims_df[sp] {
                let gp = p0 + f;
                let pl = p_local[gp];
                let tv = t_slab.index_axis(Axis(0), pl);
                let t1 = ci.dot(&tv); // (nnf)
                let row = t1.dot(cv); // (ncols_i)
                let r = strip.row_pos[gp];
                strip.b.row_mut(r).assign(&row);
            }
        }
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
    let per_func = ax.nsf.max(1) * ax.nnf.max(1) * 8;
    let max_funcs = (ctx.worker_budget / per_func).max(1);
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
        transform_slab(ctx, is, &mut strips, &c_occ, &c_virt, &p_local, &t_slab);
        tm.transform_s += tt.elapsed().as_secs_f64();
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
        for p in 0..np {
            let r = strip.row_pos[off_p + p];
            strip.b.row_mut(r).assign(&rows.row(p));
        }
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
}

fn centers(m: &Array2<f64>) -> Vec<[f64; 3]> {
    m.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect()
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
    let axis = {
        let n = atom_xyz.len().max(1) as f64;
        let mut best = (0usize, -1.0f64);
        for x in 0..3 {
            let mean = atom_xyz.iter().map(|a| a[x]).sum::<f64>() / n;
            let var = atom_xyz.iter().map(|a| (a[x] - mean).powi(2)).sum::<f64>();
            if var > best.1 {
                best = (x, var);
            }
        }
        best.0
    };
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
    let partners: Vec<Vec<usize>> = (0..no)
        .map(|i| (0..no).filter(|&j| keep[i * no + j]).collect())
        .collect();
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
    // extended per-occupied unions over surviving partners (pair sets are
    // D_ij = D_i ∪ D_j and C_ij = V_i ∪ V_j, both ⊆ the extended sets)
    let aux_ext = (0..no)
        .map(|i| sorted_union(partners[i].iter().map(|&j| aux_dom[j].iter().copied())))
        .collect();
    let virt_ext = (0..no)
        .map(|i| sorted_union(partners[i].iter().map(|&j| virt_dom[j].iter().copied())))
        .collect();
    let batches = integral_batches(&atom_xyz, &occ_xyz, dcfg.batch_merge);
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
        &mut stats,
    )?;
    let ragged = fit_pairs(
        &pre,
        dfbs,
        spaces,
        eps,
        scale,
        dcfg,
        FitPath::Cholesky,
        &mut stats,
    )?;
    Ok((ragged, pre.n_gated, stats))
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
    stats: &mut DirectStats,
) -> Result<StripsAndMetric, FerricError> {
    use rayon::prelude::*;
    let no = spaces.no;
    let t0 = Instant::now();
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
    } = build_locality_maps(mol, obs, dfbs, spaces, eps, pair_gate_cal, dcfg);
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
    let qpair = schwarz_triple_factors(op, obs, dfbs, dcfg.schwarz_skip)?;
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
        worker_budget: dcfg.scratch_budget_bytes / n_workers,
        split_timers: dcfg.split_eri3_timers,
    };
    let batch_out: Vec<BatchOut> = match dcfg.eri3_pass {
        Eri3Pass::Batches => batches
            .par_iter()
            .map(|(_, is)| eval_batch(&bctx, is, fill_slab))
            .collect(),
        Eri3Pass::AuxMajor => aux_major_pass(&bctx, dcfg.scratch_budget_bytes, stats),
    };
    let strips = collect_strips(no, batch_out, dcfg.split_eri3_timers, stats)?;
    stats.t_eri3_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let need = metric_need_pairs(&aux_ext, dfbs.nshells());
    stats.n_metric_shell_pairs = need.len();
    let metric_blocks = eval_metric_blocks(op, dfbs, &need)?;
    stats.t_metric_s = t0.elapsed().as_secs_f64();
    Ok(StripsAndMetric {
        keep,
        n_gated,
        aux_dom,
        virt_dom,
        strips,
        metric_blocks,
    })
}

/// Batch outputs → one strip per occupied (an occupied no batch built is a
/// construction bug), summing the stage-3 counters, timers and strip sizes.
fn collect_strips(
    no: usize,
    batch_out: Vec<BatchOut>,
    split_timers: bool,
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
        stats.batch_work_max_s = stats.batch_work_max_s.max(tm.total_s);
        stats.batch_work_mean_s += tm.total_s;
        for (i, st) in batch_strips {
            strips[i] = Some(st);
        }
    }
    if stats.n_batches > 0 {
        stats.batch_work_mean_s /= stats.n_batches as f64;
    }
    let strips: Vec<Strip> = strips
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            s.ok_or_else(|| FerricError::General(format!("lmp2_direct: no strip built for i={i}")))
        })
        .collect::<Result<_, _>>()?;
    for st in &strips {
        stats.strip_rows_mean += st.rows.len() as f64;
        stats.strip_rows_max = stats.strip_rows_max.max(st.rows.len());
        stats.strip_cols_mean += st.cols.len() as f64;
        stats.strip_cols_max = stats.strip_cols_max.max(st.cols.len());
        stats.strip_bytes_total += st.b.len() * 8;
    }
    if no > 0 {
        stats.strip_rows_mean /= no as f64;
        stats.strip_cols_mean /= no as f64;
    }
    Ok(strips)
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
            let o = whiten_own_domains(no, scale, aux_dom, strips, dims_df, mb, dfbs, r, keep)?;
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
/// column position is a construction bug and panics loudly.
fn pair_candidates(
    unique: &[(usize, usize)],
    virt_dom: &[Vec<usize>],
    strips: &[Strip],
    qvirt: Option<&(Vec<Vec<f64>>, Vec<f64>)>,
    kappa_eps: Option<f64>,
) -> Vec<Vec<usize>> {
    unique
        .iter()
        .map(|&(i, j)| {
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

/// Stage 5 of [`assemble_ragged_direct_local`]: per-pair domain fits + the
/// Eq-8 mask, GROUPED BY PAIR DOMAIN. The C20–C48 maps put 6–9× more
/// surviving pairs than distinct pair domains
/// (`scripts/queue/proto_batch_amortization.py`), so the V_DD Cholesky is
/// done once per group and each distinct occupied is whitened once per
/// group ([`fit_domain_group`]); per-group work is held per worker and
/// dropped after its group. Results are written back into original-pair
/// order slots, so the assembled pair order is the ungrouped loop's.
#[allow(clippy::too_many_arguments)]
fn fit_pairs(
    pre: &StripsAndMetric,
    dfbs: &PreparedBasis,
    spaces: &LocalizedSpaces,
    eps: f64,
    scale: f64,
    dcfg: &DirectConfig,
    path: FitPath,
    stats: &mut DirectStats,
) -> Result<Ragged, FerricError> {
    use rayon::prelude::*;
    let (no, nv) = (spaces.no, spaces.nv);
    let t0 = Instant::now();
    let fo: Vec<f64> = (0..no).map(|i| spaces.f_oo[(i, i)]).collect();
    let fv: Vec<f64> = (0..nv).map(|a| spaces.f_vv[(a, a)]).collect();
    let unique: Vec<(usize, usize)> = (0..no)
        .flat_map(|i| (i..no).map(move |j| (i, j)))
        .filter(|&(i, j)| pre.keep[i * no + j])
        .collect();
    let kappa_eps: Option<f64> = match dcfg.virt_schwarz_kappa {
        Some(k) if eps > 0.0 => Some(k * eps),
        _ => None,
    };
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
    );
    for c in &pair_cands {
        stats.virt_cand_mean += c.len() as f64;
        stats.virt_cand_max = stats.virt_cand_max.max(c.len());
    }
    if !pair_cands.is_empty() {
        stats.virt_cand_mean /= pair_cands.len() as f64;
    }

    let pair_dom: Vec<Vec<usize>> = unique
        .iter()
        .map(|&(i, j)| {
            sorted_union([
                pre.aux_dom[i].iter().copied(),
                pre.aux_dom[j].iter().copied(),
            ])
        })
        .collect();
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
    let group_list = group_by_domain(pair_dom);
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
        nv,
        aux_radius_bohr: dcfg.aux_radius_bohr,
        dfbs,
        aux_dom: &pre.aux_dom,
        own_w: own_w.as_deref(),
        #[cfg(test)]
        w_mutant: match path {
            FitPath::MutantW(first, rel) => Some((first, rel)),
            _ => None,
        },
    };
    let group_out: Vec<Result<(SlotOut, GroupCounts), FerricError>> = group_list
        .par_iter()
        .map(|(dsh, pxs)| match path {
            FitPath::Cholesky => fit_domain_group(&ctx, dsh, pxs),
            #[cfg(test)]
            FitPath::MutantW(..) => fit_domain_group(&ctx, dsh, pxs),
            #[cfg(test)]
            FitPath::InverseOracle => {
                inverse_oracle::fit_domain_group_inverse_oracle(&ctx, dsh, pxs)
                    .map(|s| (s, GroupCounts::default()))
            }
        })
        .collect();
    let ragged = ragged_from_groups(unique.len(), group_out, stats)?;
    stats.t_pairs_s = t0.elapsed().as_secs_f64();
    Ok(ragged)
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
fn ragged_from_groups(
    n_unique: usize,
    group_out: Vec<Result<(SlotOut, GroupCounts), FerricError>>,
    stats: &mut DirectStats,
) -> Result<Ragged, FerricError> {
    stats.n_pair_groups = group_out.len();
    let mut n_occ = 0usize;
    let mut slots: Vec<Option<Vec<PairBlock>>> = (0..n_unique).map(|_| None).collect();
    for r in group_out {
        let (slot_out, counts) = r?;
        n_occ += counts.n_occ;
        stats.n_trsm += counts.n_trsm;
        stats.n_w_reused += counts.n_reused;
        stats.group_scratch_bytes_max = stats.group_scratch_bytes_max.max(counts.scratch_bytes);
        for (px, blocks) in slot_out {
            slots[px] = Some(blocks);
        }
    }
    if stats.n_pair_groups > 0 {
        stats.n_distinct_occ_per_group = n_occ as f64 / stats.n_pair_groups as f64;
    }
    let mut pairs: Vec<PairBlock> = Vec::new();
    for (px, s) in slots.into_iter().enumerate() {
        pairs.extend(s.ok_or_else(|| {
            FerricError::General(format!(
                "lmp2_direct: pair slot {px} never filled (grouping bug)"
            ))
        })?);
    }
    let mut by_i: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut by_j: HashMap<usize, Vec<usize>> = HashMap::new();
    for (pidx, pb) in pairs.iter().enumerate() {
        by_i.entry(pb.i).or_default().push(pidx);
        by_j.entry(pb.j).or_default().push(pidx);
    }
    Ok(Ragged { pairs, by_i, by_j })
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

/// Integral-direct amplitude-threshold LMP2 driver — the direct-path
/// sibling of [`crate::lmp2_amplitude::amplitude_lmp2`]. `cfg.fit_radius_bohr`
/// and `cfg.aux_tail_frac` are ignored (the direct path's locality lives in
/// `dcfg`); everything else means the same thing.
///
/// `cfg.compute_reference` (default false) opts in to the canonical
/// `ri_mp2` reference, which DOES form the global `(naux, no·nv)` tensor
/// this path otherwise never builds — leave it off for a reduced-cost run.
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
    let (vvhv, t_spaces_s) = checked_vvhv(mol, obs, obs_bs, rhf, "lmp2_direct")?;
    let (mut r, st) =
        amplitude_lmp2_direct_with_virtuals(mol, obs, dfbs, op, rhf, cfg, dcfg, &vvhv)?;
    r.timings.t_spaces_s = t_spaces_s;
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
    let t0 = Instant::now();
    let spaces = localized_spaces(mol, obs, rhf, cfg.frozen_core, vvhv)?;
    let t_loc_s = t0.elapsed().as_secs_f64();
    let (no, nv) = (spaces.no, spaces.nv);
    let (rg, n_pairs_gated, mut dstats) = assemble_ragged_direct_local(
        mol,
        obs,
        dfbs,
        op,
        &spaces,
        cfg.eps,
        1.0,
        cfg.pair_gate_cal,
        dcfg,
    )?;
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
    r.e_corr_canonical_ri = e_ref;
    r.n_pairs_gated = n_pairs_gated;
    r.aux_dom_mean = dstats.strip_rows_mean;
    r.aux_dom_max = dstats.strip_rows_max;
    r.timings = StageTimings {
        t_spaces_s: 0.0,
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
        let cfg = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
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
            let cfg = RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            };
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
                let qpair = schwarz_triple_factors(op, &obs, &dfbs, dcfg.schwarz_skip)
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

    /// B4 anchor: on C4/6-31G with production maps (gate on, Schwarz cut
    /// 1e-5, merged batches, a scratch share small enough to force several
    /// slabs per batch), the Q-sorted pair list with the binary-searched
    /// cut evaluates exactly the oracle's triples (same evaluated and
    /// skipped counts per batch) and builds bitwise-identical strips.
    #[test]
    fn sorted_pair_list_matches_unsorted_scan() {
        let mol = Molecule::load_xyz(&format!(
            "{}/../../testdata/molecules/alkane_4.xyz",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let obs_bs = ferric_core::basis::bundled("6-31g").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let cfg = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
        let (vvhv, _) = checked_vvhv(&mol, &obs, &obs_bs, &rhf, "test").unwrap();
        let spaces = localized_spaces(&mol, &obs, &rhf, 4, &vvhv).unwrap();
        let dcfg = DirectConfig {
            aux_radius_bohr: 10.0,
            virt_radius_bohr: Some(12.0),
            ao_tail: 1e-3,
            schwarz_skip: 1e-5,
            batch_merge: 2,
            ..Default::default()
        };
        let maps = build_locality_maps(&mol, &obs, &dfbs, &spaces, 1e-4, Some(0.7), &dcfg);
        let qpair = schwarz_triple_factors(op, &obs, &dfbs, dcfg.schwarz_skip).unwrap();
        let ctx = BatchCtx {
            op,
            obs: &obs,
            dfbs: &dfbs,
            spaces: &spaces,
            aux_ext: &maps.aux_ext,
            virt_ext: &maps.virt_ext,
            occ_supp: &maps.occ_supp,
            virt_supp: &maps.virt_supp,
            qpair: qpair.as_ref(),
            schwarz_skip: dcfg.schwarz_skip,
            worker_budget: 2 << 20,
            split_timers: false,
        };
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
        let cfg = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
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
        let mut st = DirectStats::default();
        strips_and_metric(
            &su.mol,
            &su.obs,
            &su.dfbs,
            op,
            &su.spaces,
            eps,
            Some(cal),
            &production_maps(None),
            &mut st,
        )
        .unwrap()
    }

    fn fit(
        su: &Setup,
        pre: &StripsAndMetric,
        eps: f64,
        dcfg: &DirectConfig,
        path: FitPath,
    ) -> (Ragged, DirectStats) {
        let mut st = DirectStats::default();
        let r = fit_pairs(pre, &su.dfbs, &su.spaces, eps, 1.0, dcfg, path, &mut st).unwrap();
        (r, st)
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
}
