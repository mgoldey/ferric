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

use ndarray::{Array1, Array2, Array3, Axis};
use std::collections::{BTreeSet, HashMap};
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
    /// Integral batches run, and the max/mean wall seconds of one batch
    /// worker (the load-balance counter: max ≫ mean is a tail imbalance).
    pub n_batches: usize,
    pub batch_work_max_s: f64,
    pub batch_work_mean_s: f64,
    pub t_metric_s: f64,
    pub t_pairs_s: f64,
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

/// Per-occupied Schwarz magnitudes q_ia = √(b_ia^T V_{D_i D_i}⁻¹ b_ia)
/// over each i's OWN distance domain, from strip-local rows — the
/// `virt_schwarz_kappa` screen's inputs (stage 5a). Returns (q per i,
/// qmax per i). Extracted from the assembly driver (complexity ceiling).
#[allow(clippy::too_many_arguments)]
fn schwarz_virtual_q(
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

/// Per-slot output of one domain group's pair fits.
type SlotOut = Vec<(usize, Vec<PairBlock>)>;

/// Borrowed context for [`fit_domain_group`] — everything stage 5's group
/// workers read but never write. Extracted from the assembly driver
/// (complexity ceiling); the per-pair arithmetic is IDENTICAL to the
/// pre-extraction inline loop, so blocks and energies are unchanged.
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
}

/// One domain group's pair fits: V_DD Cholesky inverse ONCE, then every
/// pair in the group gathers its strip blocks, fits, and Eq-8-masks.
fn fit_domain_group(
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
            if let Some(pb) =
                pair_block_from_g_cand(j, i, &gt, cij, ctx.nv, ctx.f_vv, ctx.fo, ctx.fv, ctx.eps)
            {
                out.push(pb);
            }
        }
        out_group.push((px, out));
    }
    Ok(out_group)
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
                }
            }
            let ts = Instant::now();
            for (pr, off) in &meta {
                scatter_block(ctx, ax, pr, sp, &buf[*off..], p_local, t_slab);
            }
            let tf = Instant::now();
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
    use rayon::prelude::*;
    let (no, nv) = (spaces.no, spaces.nv);
    let mut stats = DirectStats::default();

    // ---- stage 1+2: gate and maps (integral-free) ----
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

    // ---- stage 3: batched integral evaluation + half-transform ----
    let t0 = Instant::now();
    Engine::new_3center(op, obs, dfbs, 1e-14)?; // surface construction errors serially
    let qpair = schwarz_triple_factors(op, obs, dfbs, dcfg.schwarz_skip)?;
    let dims_df = dfbs.shell_dims();
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
    let batch_out: Vec<BatchOut> = batches
        .par_iter()
        .map(|(_, is)| eval_batch(&bctx, is, fill_slab))
        .collect();

    stats.n_batches = batch_out.len();
    if !dcfg.split_eri3_timers {
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
    stats.t_eri3_s = t0.elapsed().as_secs_f64();

    // ---- stage 4: domain-restricted 2-center metric ----
    let t0 = Instant::now();
    let need = metric_need_pairs(&aux_ext, dfbs.nshells());
    stats.n_metric_shell_pairs = need.len();
    let metric_blocks = eval_metric_blocks(op, dfbs, &need)?;
    stats.t_metric_s = t0.elapsed().as_secs_f64();

    // ---- stage 5: per-pair domain fits + Eq-8 mask, GROUPED BY DOMAIN ----
    // The C20-C48 maps put 6-9× more surviving pairs than DISTINCT pair
    // domains (`scripts/queue/proto_batch_amortization.py`), so the d³
    // V_DD Cholesky inverse is hoisted to once per group. The inverse is
    // held per worker and dropped after its group (no cache memory);
    // results are written back into original-pair-order slots, and the
    // per-pair arithmetic is unchanged, so the assembled blocks — and the
    // energies — are bit-identical to the ungrouped loop.
    let t0 = Instant::now();
    let fo: Vec<f64> = (0..no).map(|i| spaces.f_oo[(i, i)]).collect();
    let fv: Vec<f64> = (0..nv).map(|a| spaces.f_vv[(a, a)]).collect();
    let unique: Vec<(usize, usize)> = (0..no)
        .flat_map(|i| (i..no).map(move |j| (i, j)))
        .filter(|&(i, j)| keep[i * no + j])
        .collect();

    // ---- stage 5a: ε-linked Schwarz virtual candidates (prototype GO,
    // WIKI-APPEND-eps-linked-maps.md). q_ia = √(b_ia^T V_{D_i D_i}⁻¹ b_ia)
    // from the strip rows over i's OWN distance domain D_i (⊆ the strip's
    // extended rows since i is its own partner); a stays in C_ij iff
    // scale·q_ia·qmax_j ≥ κ·ε or scale·q_ja·qmax_i ≥ κ·ε — both Eq-8
    // orientations, Cauchy–Schwarz-exact on a global Gram fit and measured
    // escape-free vs the domain-local fit at κ ≤ 1. ε = 0 or None: no-op.
    let kappa_eps: Option<f64> = match dcfg.virt_schwarz_kappa {
        Some(k) if eps > 0.0 => Some(k * eps),
        _ => None,
    };
    let qvirt: Option<(Vec<Vec<f64>>, Vec<f64>)> = match kappa_eps {
        None => None,
        Some(_) => Some(schwarz_virtual_q(
            no,
            scale,
            &aux_dom,
            &strips,
            dims_df,
            &metric_blocks,
            dfbs,
            dcfg.aux_radius_bohr,
        )?),
    };
    // per-pair candidate sets C_ij (screened union; the screen only ever
    // SHRINKS the union, so the strip-coverage invariant is untouched)
    let pair_cands: Vec<Vec<usize>> = unique
        .iter()
        .map(|&(i, j)| {
            let cij = sorted_union([virt_dom[i].iter().copied(), virt_dom[j].iter().copied()]);
            match (&qvirt, kappa_eps) {
                (Some((q, qmax)), Some(th)) => cij
                    .into_iter()
                    .filter(|&a| {
                        // a ∈ virt_dom[i] ∪ virt_dom[j] ⊆ virt_ext of BOTH
                        // strips (i, j are mutual partners), so both col
                        // positions exist; usize::MAX would be a
                        // construction bug and panics loudly here.
                        let qia = q[i][strips[i].col_pos[a]];
                        let qja = q[j][strips[j].col_pos[a]];
                        qia * qmax[j] >= th || qja * qmax[i] >= th
                    })
                    .collect(),
                _ => cij,
            }
        })
        .collect();
    for c in &pair_cands {
        stats.virt_cand_mean += c.len() as f64;
        stats.virt_cand_max = stats.virt_cand_max.max(c.len());
    }
    if !pair_cands.is_empty() {
        stats.virt_cand_mean /= pair_cands.len() as f64;
    }

    let mut dom_groups: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for (px, &(i, j)) in unique.iter().enumerate() {
        let dsh = sorted_union([aux_dom[i].iter().copied(), aux_dom[j].iter().copied()]);
        dom_groups.entry(dsh).or_default().push(px);
    }
    let mut group_list: Vec<(Vec<usize>, Vec<usize>)> = dom_groups.into_iter().collect();
    group_list.sort_unstable_by_key(|(_, pxs)| pxs[0]);
    let ctx = PairFitCtx {
        unique: &unique,
        pair_cands: &pair_cands,
        strips: &strips,
        dims_df,
        metric_blocks: &metric_blocks,
        f_vv: &spaces.f_vv,
        fo: &fo,
        fv: &fv,
        eps,
        scale,
        nv,
        aux_radius_bohr: dcfg.aux_radius_bohr,
        dfbs,
    };
    let group_out: Vec<Result<SlotOut, FerricError>> = group_list
        .par_iter()
        .map(|(dsh, pxs)| fit_domain_group(&ctx, dsh, pxs))
        .collect();
    let mut slots: Vec<Option<Vec<PairBlock>>> = (0..unique.len()).map(|_| None).collect();
    for r in group_out {
        for (px, blocks) in r? {
            slots[px] = Some(blocks);
        }
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
    stats.t_pairs_s = t0.elapsed().as_secs_f64();
    Ok((Ragged { pairs, by_i, by_j }, n_gated, stats))
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
