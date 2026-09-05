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
    build_vvhv, check_vvhv, hylleraas_energy, localized_spaces, pair_gate_keep,
    AmplitudeLmp2Config, AmplitudeLmp2Result, LocalizedSpaces, StageTimings, VvHv,
};
use crate::ragged::{pair_block_from_g_cand, solve_ragged, PairBlock, Ragged};
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
    /// GLOBAL integral-slab scratch cap (bytes), shared across all
    /// concurrently-running batch workers — each worker slabs under
    /// budget/n_workers, so the co-resident scratch never exceeds this
    /// (the ferric-batch N×-overcommit lesson: a per-worker budget is a
    /// budget on nothing).
    pub scratch_budget_bytes: usize,
}

impl Default for DirectConfig {
    fn default() -> Self {
        Self {
            aux_radius_bohr: 10.0,
            virt_radius_bohr: None,
            ao_tail: 0.0,
            schwarz_skip: 0.0,
            scratch_budget_bytes: 2usize << 30,
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
    pub t_maps_s: f64,
    pub t_eri3_s: f64,
    pub t_metric_s: f64,
    pub t_pairs_s: f64,
}

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
    let naux = dfbs.nbasis();
    let nao = obs.nbasis();
    let mut stats = DirectStats::default();

    // ---- stage 1+2: gate and maps (integral-free) ----
    let t0 = Instant::now();
    let (keep, n_gated) = match pair_gate_cal {
        Some(cal) => pair_gate_keep(&spaces.occ_centers, &spaces.occ_spreads, no, eps, cal),
        None => (vec![true; no * no], 0),
    };
    let partners: Vec<Vec<usize>> =
        (0..no).map(|i| (0..no).filter(|&j| keep[i * no + j]).collect()).collect();

    let atom_xyz: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let occ_xyz: Vec<[f64; 3]> = (0..no)
        .map(|i| [spaces.occ_centers[(i, 0)], spaces.occ_centers[(i, 1)], spaces.occ_centers[(i, 2)]])
        .collect();
    let virt_xyz: Vec<[f64; 3]> = (0..nv)
        .map(|a| [spaces.virt_centers[(a, 0)], spaces.virt_centers[(a, 1)], spaces.virt_centers[(a, 2)]])
        .collect();

    let nsh_df = dfbs.nshells();
    let df_shell_atom = dfbs.shell_to_atom();
    let r_aux2 = dcfg.aux_radius_bohr * dcfg.aux_radius_bohr;
    let aux_dom: Vec<Vec<usize>> = (0..no)
        .map(|i| {
            (0..nsh_df)
                .filter(|&sp| dist2(atom_xyz[df_shell_atom[sp]], occ_xyz[i]) <= r_aux2)
                .collect()
        })
        .collect();
    let virt_dom: Vec<Vec<usize>> = match dcfg.virt_radius_bohr {
        None => (0..no).map(|_| (0..nv).collect()).collect(),
        Some(rv) => {
            let rv2 = rv * rv;
            (0..no)
                .map(|i| (0..nv).filter(|&a| dist2(virt_xyz[a], occ_xyz[i]) <= rv2).collect())
                .collect()
        }
    };
    let occ_supp: Vec<Vec<usize>> =
        (0..no).map(|i| supp_shells(obs, spaces.c_locc.column(i), dcfg.ao_tail)).collect();
    let virt_supp: Vec<Vec<usize>> =
        (0..nv).map(|a| supp_shells(obs, spaces.c_vloc.column(a), dcfg.ao_tail)).collect();

    // extended per-occupied unions over surviving partners (pair sets are
    // D_ij = D_i ∪ D_j and C_ij = V_i ∪ V_j, both ⊆ the extended sets)
    let aux_ext: Vec<Vec<usize>> = (0..no)
        .map(|i| sorted_union(partners[i].iter().map(|&j| aux_dom[j].iter().copied())))
        .collect();
    let virt_ext: Vec<Vec<usize>> = (0..no)
        .map(|i| sorted_union(partners[i].iter().map(|&j| virt_dom[j].iter().copied())))
        .collect();

    // per-atom batches: occupieds grouped by nearest atom to their centroid
    let mut batch_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..no {
        let (amin, _) = atom_xyz
            .iter()
            .enumerate()
            .map(|(ai, &axyz)| (ai, dist2(axyz, occ_xyz[i])))
            .min_by(|a, b| a.1.partial_cmp(&b.1).expect("NaN centroid distance"))
            .expect("molecule has no atoms");
        batch_map.entry(amin).or_default().push(i);
    }
    let mut batches: Vec<(usize, Vec<usize>)> = batch_map.into_iter().collect();
    batches.sort_unstable_by_key(|(a, _)| *a);
    stats.t_maps_s = t0.elapsed().as_secs_f64();

    // ---- stage 3: batched integral evaluation + half-transform ----
    let t0 = Instant::now();
    Engine::new_3center(op, obs, dfbs, 1e-14)?; // surface construction errors serially
    // Cauchy–Schwarz factors for the batch triple cut, same-op kernel:
    // |(P|μν)| ≤ √(P|P) · Q(μν). The aux factor is per aux SHELL
    // (√ of the max (P|P) diagonal over the shell — conservative at shell
    // granularity); dropping it would make the cut non-conservative
    // wherever √(P|P) > 1 (found in review before any merge).
    // schwarz() supports Coulomb/erf/erfc only — name the knob in the error
    // so a terfc/table-engine run knows the fix is schwarz_skip = 0.0.
    let qpair: Option<(Array2<f64>, Vec<f64>)> = if dcfg.schwarz_skip > 0.0 {
        let sb = ferric_scf::screening::SchwarzBounds::compute(op, obs).map_err(|e| {
            FerricError::General(format!(
                "lmp2_direct: schwarz_skip = {} requires shell-pair Schwarz bounds, \
                 unavailable for operator {:?} ({e}); set schwarz_skip = 0.0 for this operator",
                dcfg.schwarz_skip, op.kind
            ))
        })?;
        let mut eng2 = Engine::new_2center(op, dfbs, 1e-14)?;
        let dims_df = dfbs.shell_dims();
        let qp_shell: Vec<f64> = (0..dfbs.nshells())
            .map(|sp| {
                let np = dims_df[sp];
                let vals = eng2.compute_eri2(dfbs, sp, sp);
                let mut m = 0.0f64;
                for p in 0..np {
                    m = m.max(vals[p * np + p].abs());
                }
                m.sqrt()
            })
            .collect();
        Some((sb.q, qp_shell))
    } else {
        None
    };
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();

    let batch_out: Vec<(Vec<(usize, Strip)>, u64, u64)> = batches
        .par_iter()
        .map(|(_, is)| {
            let mut eng =
                Engine::new_3center(op, obs, dfbs, 1e-14).expect("3-center engine (pre-validated)");
            let mut n_triples = 0u64;
            let mut n_skipped = 0u64;
            // union sets for the batch
            let p_shells = sorted_union(is.iter().map(|&i| aux_ext[i].iter().copied()));
            let s_shells = sorted_union(is.iter().map(|&i| occ_supp[i].iter().copied()));
            let batch_virts = sorted_union(is.iter().map(|&i| virt_ext[i].iter().copied()));
            let n_shells =
                sorted_union(batch_virts.iter().map(|&a| virt_supp[a].iter().copied()));
            let s_funcs = shell_funcs(obs, &s_shells);
            let n_funcs = shell_funcs(obs, &n_shells);
            let (nsf, nnf) = (s_funcs.len(), n_funcs.len());
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

            // per-occupied masked coefficients on the batch's compact axes
            let c_occ_masked: Vec<Array1<f64>> = is
                .iter()
                .map(|&i| {
                    let mut c = Array1::<f64>::zeros(nsf);
                    for &sh in &occ_supp[i] {
                        for f in 0..dims_obs[sh] {
                            let g = offs_obs[sh] + f;
                            c[s_pos[g]] = spaces.c_locc[(g, i)];
                        }
                    }
                    c
                })
                .collect();
            let c_virt_masked: Vec<Array2<f64>> = is
                .iter()
                .map(|&i| {
                    let cols = &virt_ext[i];
                    let mut cv = Array2::<f64>::zeros((nnf, cols.len()));
                    for (k, &a) in cols.iter().enumerate() {
                        for &sh in &virt_supp[a] {
                            for f in 0..dims_obs[sh] {
                                let g = offs_obs[sh] + f;
                                cv[(n_pos[g], k)] = spaces.c_vloc[(g, a)];
                            }
                        }
                    }
                    cv
                })
                .collect();

            // strip allocation
            let mut strips: Vec<Strip> = is
                .iter()
                .map(|&i| {
                    let rows = shell_funcs(dfbs, &aux_ext[i]);
                    let mut row_pos = vec![usize::MAX; naux];
                    for (k, &f) in rows.iter().enumerate() {
                        row_pos[f] = k;
                    }
                    let cols = virt_ext[i].clone();
                    let mut col_pos = vec![usize::MAX; nv];
                    for (k, &a) in cols.iter().enumerate() {
                        col_pos[a] = k;
                    }
                    let b = Array2::<f64>::zeros((rows.len(), cols.len()));
                    Strip { rows, row_pos, cols, col_pos, b }
                })
                .collect();

            // slab the batch's aux shells under the per-worker share of the
            // GLOBAL scratch budget (workers run concurrently under rayon)
            let n_workers = rayon::current_num_threads().max(1).min(batches.len().max(1));
            let per_func = nsf.max(1) * nnf.max(1) * 8;
            let max_funcs = ((dcfg.scratch_budget_bytes / n_workers) / per_func).max(1);
            let mut slab_start = 0usize;
            while slab_start < p_shells.len() {
                let mut slab_end = slab_start;
                let mut nfuncs = 0usize;
                while slab_end < p_shells.len() {
                    let add = dims_df[p_shells[slab_end]];
                    if nfuncs > 0 && nfuncs + add > max_funcs {
                        break;
                    }
                    nfuncs += add;
                    slab_end += 1;
                }
                let slab = &p_shells[slab_start..slab_end];
                let mut p_local = vec![usize::MAX; naux];
                let mut nps = 0usize;
                for &sp in slab {
                    for f in 0..dims_df[sp] {
                        p_local[offs_df[sp] + f] = nps;
                        nps += 1;
                    }
                }
                let mut t_slab = Array3::<f64>::zeros((nps, nsf, nnf));
                for &sp in slab {
                    for (ux, &ua) in u_shells.iter().enumerate() {
                        for &ub in &u_shells[..=ux] {
                            let need_ab = in_s[ua] && in_n[ub];
                            let need_ba = in_s[ub] && in_n[ua];
                            if !(need_ab || need_ba) {
                                continue;
                            }
                            if let Some((q, qp_shell)) = &qpair {
                                if qp_shell[sp] * q[(ua, ub)] < dcfg.schwarz_skip {
                                    n_skipped += 1;
                                    continue;
                                }
                            }
                            n_triples += 1;
                            if let Some(block) = eng.compute_eri3(obs, dfbs, sp, ua, ub) {
                                let (n1, n2) = (dims_obs[ua], dims_obs[ub]);
                                for p in 0..dims_df[sp] {
                                    let pl = p_local[offs_df[sp] + p];
                                    for fa in 0..n1 {
                                        let ga = offs_obs[ua] + fa;
                                        for fb in 0..n2 {
                                            let gb = offs_obs[ub] + fb;
                                            let val = block[(p * n1 + fa) * n2 + fb];
                                            if need_ab {
                                                t_slab[(pl, s_pos[ga], n_pos[gb])] = val;
                                            }
                                            if need_ba {
                                                t_slab[(pl, s_pos[gb], n_pos[ga])] = val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // transform the slab rows into each occupied's strip
                for (k, &i) in is.iter().enumerate() {
                    let strip = &mut strips[k];
                    let ci = &c_occ_masked[k];
                    let cv = &c_virt_masked[k];
                    for &sp in &aux_ext[i] {
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
                slab_start = slab_end;
            }
            (is.iter().copied().zip(strips).collect::<Vec<_>>(), n_triples, n_skipped)
        })
        .collect();

    let mut strips: Vec<Option<Strip>> = (0..no).map(|_| None).collect();
    for (batch_strips, n_triples, n_skipped) in batch_out {
        stats.n_eri3_shell_triples += n_triples;
        stats.n_eri3_skipped += n_skipped;
        for (i, st) in batch_strips {
            strips[i] = Some(st);
        }
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
    let mut need: BTreeSet<(usize, usize)> = BTreeSet::new();
    for shells in &aux_ext {
        for (x, &sp) in shells.iter().enumerate() {
            for &sq in &shells[..=x] {
                need.insert((sp, sq)); // sp >= sq (sorted list)
            }
        }
    }
    let need: Vec<(usize, usize)> = need.into_iter().collect();
    stats.n_metric_shell_pairs = need.len();
    Engine::new_2center(op, dfbs, 1e-14)?;
    let metric_blocks: HashMap<(usize, usize), Array2<f64>> = need
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
                ((sp, sq), blk)
            },
        )
        .collect::<Vec<_>>()
        .into_iter()
        .collect();
    stats.t_metric_s = t0.elapsed().as_secs_f64();

    // ---- stage 5: per-pair domain fits + Eq-8 mask ----
    let t0 = Instant::now();
    let fo: Vec<f64> = (0..no).map(|i| spaces.f_oo[(i, i)]).collect();
    let fv: Vec<f64> = (0..nv).map(|a| spaces.f_vv[(a, a)]).collect();
    let unique: Vec<(usize, usize)> = (0..no)
        .flat_map(|i| (i..no).map(move |j| (i, j)))
        .filter(|&(i, j)| keep[i * no + j])
        .collect();
    let per_pair: Vec<Result<Vec<PairBlock>, FerricError>> = unique
        .par_iter()
        .map(|&(i, j)| {
            use ndarray_linalg::InverseC;
            let dsh = sorted_union([aux_dom[i].iter().copied(), aux_dom[j].iter().copied()]);
            let dfuncs = shell_funcs(dfbs, &dsh);
            if dfuncs.is_empty() {
                return Err(FerricError::General(format!(
                    "lmp2_direct: empty aux domain for pair ({i},{j}) at radius {} Bohr",
                    dcfg.aux_radius_bohr
                )));
            }
            let cij = sorted_union([virt_dom[i].iter().copied(), virt_dom[j].iter().copied()]);
            if cij.is_empty() {
                return Ok(Vec::new());
            }
            let d = dfuncs.len();
            // V_DD gathered block-wise from the sparse metric
            let mut vdd = Array2::<f64>::zeros((d, d));
            let mut loc_off = Vec::with_capacity(dsh.len());
            let mut acc = 0usize;
            for &sh in &dsh {
                loc_off.push(acc);
                acc += dims_df[sh];
            }
            for (bx, &sp) in dsh.iter().enumerate() {
                for (by, &sq) in dsh.iter().enumerate() {
                    let (key, transposed) = if sp >= sq { ((sp, sq), false) } else { ((sq, sp), true) };
                    let blk = metric_blocks.get(&key).ok_or_else(|| {
                        FerricError::General(format!(
                            "lmp2_direct: metric block ({sp},{sq}) missing for pair ({i},{j})"
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
            let vdd_inv = vdd.invc().map_err(|e| {
                FerricError::General(format!("lmp2_direct V_DD Cholesky inverse ({i},{j}): {e}"))
            })?;
            // gather A_i, A_j from the strips (rows D_ij, cols C_ij)
            let gather = |k: usize| -> Result<Array2<f64>, FerricError> {
                let st = &strips[k];
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
            if scale != 1.0 {
                g.mapv_inplace(|x| scale * x);
            }
            let mut out = Vec::with_capacity(2);
            if let Some(pb) =
                pair_block_from_g_cand(i, j, &g, &cij, nv, &spaces.f_vv, &fo, &fv, eps)
            {
                out.push(pb);
            }
            if i != j {
                let gt = g.t().to_owned();
                if let Some(pb) =
                    pair_block_from_g_cand(j, i, &gt, &cij, nv, &spaces.f_vv, &fo, &fv, eps)
                {
                    out.push(pb);
                }
            }
            Ok(out)
        })
        .collect();
    let mut pairs: Vec<PairBlock> = Vec::new();
    for r in per_pair {
        pairs.extend(r?);
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

/// Integral-direct amplitude-threshold LMP2 driver — the direct-path
/// sibling of [`crate::lmp2_amplitude::amplitude_lmp2`]. `cfg.fit_radius_bohr`
/// and `cfg.aux_tail_frac` are ignored (the direct path's locality lives in
/// `dcfg`); everything else means the same thing.
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
    let vvhv = build_vvhv(mol, obs, obs_bs, rhf)?;
    let nocc_total = (mol.nelec() as usize) / 2;
    let (dev_orth, dev_span) = check_vvhv(obs, rhf, nocc_total, &vvhv.c_vloc);
    if dev_orth > 1e-8 || dev_span > 1e-8 {
        return Err(FerricError::General(format!(
            "lmp2_direct: VV-HV construction check failed (orth {dev_orth:.2e}, span {dev_span:.2e})"
        )));
    }
    amplitude_lmp2_direct_with_virtuals(mol, obs, dfbs, op, rhf, cfg, dcfg, &vvhv)
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
    let (no, nv) = (spaces.no, spaces.nv);
    let (rg, n_pairs_gated, dstats) = assemble_ragged_direct_local(
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
    let t_reference_s = if cfg.compute_reference { t0.elapsed().as_secs_f64() } else { 0.0 };

    let t0 = Instant::now();
    let (t, iters, relres, converged, flops_mv) =
        solve_ragged(&rg, &spaces.f_oo, cfg.cg_rtol, cfg.cg_max_iter);
    if !converged {
        return Err(FerricError::General(format!(
            "lmp2_direct: ragged CG failed to converge (relres {relres:.2e} after {iters} iters)"
        )));
    }
    let e_corr = hylleraas_energy(&rg, &t);

    let total_el = (no * nv) as u64 * (no * nv) as u64;
    let kept: u64 =
        rg.pairs.iter().map(|pb| pb.pat.iter().filter(|&&x| x).count() as u64).sum();
    let dom: Vec<usize> = {
        let mut per_i: Vec<Vec<bool>> = vec![vec![false; nv]; no];
        for pb in &rg.pairs {
            for &a in &pb.da {
                per_i[pb.i][a] = true;
            }
        }
        per_i.iter().map(|v| v.iter().filter(|&&x| x).count()).collect()
    };
    let dom_max = dom.iter().copied().max().unwrap_or(0);
    let dom_mean = if no > 0 { dom.iter().sum::<usize>() as f64 / no as f64 } else { 0.0 };
    let dense_flops =
        2 * ((no * no) as u64 * (nv as u64).pow(3) + (no as u64).pow(3) * (nv * nv) as u64);
    let t_solve_s = t0.elapsed().as_secs_f64();

    Ok((
        AmplitudeLmp2Result {
            e_corr,
            e_total: rhf.energy + e_corr,
            e_corr_canonical_ri: e_ref,
            keep_fraction: kept as f64 / total_el as f64,
            pair_fraction: rg.pairs.len() as f64 / (no * no) as f64,
            dom_mean,
            dom_max,
            cg_iterations: iters,
            cg_relres: relres,
            cg_converged: converged,
            n_valence_virt: vvhv.n_valence,
            n_hard_virt: vvhv.n_hard,
            ragged_flops_per_matvec: flops_mv,
            dense_flops_per_matvec: dense_flops,
            n_pairs_gated,
            aux_dom_mean: dstats.strip_rows_mean,
            aux_dom_max: dstats.strip_rows_max,
            timings: StageTimings { t_spaces_s: 0.0, t_assembly_s, t_solve_s, t_reference_s },
        },
        dstats,
    ))
}
