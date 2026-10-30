//! Shared ragged pair-block machinery for the amplitude-threshold family
//! (LMP2 linear solve; dRPA ring products; LinLCCD ladders). Extracted from
//! `lmp2_amplitude` when the dRPA/LinLCCD ports moved onto the direct
//! assembly path — the block layout, Eq-8 pattern conventions, and the
//! solver are documented there and in wiki/amplitude-threshold-lmp2.md.

use ndarray::Array2;
use std::collections::HashMap;

/// One pair's ragged domain block: indices, sparsity pattern, and local integrals.
#[derive(Debug, Clone)]
pub struct PairBlock {
    /// Occupied index i.
    pub i: usize,
    /// Occupied index j.
    pub j: usize,
    /// Domain-A virtual indices (global, sorted).
    pub da: Vec<usize>,
    /// Domain-B virtual indices (global, sorted).
    pub db: Vec<usize>,
    /// row-major (da.len() × db.len()) pattern
    pub pat: Vec<bool>,
    /// Virtual-virtual Fock block for domain A (da × da).
    pub fvv_aa: Array2<f64>,
    /// Virtual-virtual Fock block for domain B (db × db).
    pub fvv_bb: Array2<f64>,
    /// Domain-local exchange integrals (da × db).
    pub j_blk: Array2<f64>,
    /// Orbital-energy denominators (da × db).
    pub denom: Array2<f64>,
    /// Positions of da entries within the full virtual space.
    pub pos_da: Vec<usize>,
    /// Positions of db entries within the full virtual space.
    pub pos_db: Vec<usize>,
}

/// Collection of ragged pair blocks with bidirectional index maps.
#[derive(Debug, Clone)]
pub struct Ragged {
    /// All pair blocks, in construction order.
    pub pairs: Vec<PairBlock>,
    /// Pairs that share occupied index i → indices into `pairs`.
    pub by_i: HashMap<usize, Vec<usize>>,
    /// Pairs that share occupied index j → indices into `pairs`.
    pub by_j: HashMap<usize, Vec<usize>>,
}

/// Build one pair's ragged block over a CANDIDATE index subset: `g` is
/// (cand.len(), cand.len()) with local indices mapping to the global
/// virtual indices `cand[..]`. Every Eq-8-retained element is guaranteed
/// inside the candidate square when `cand` comes from the Schwarz screen
/// (the bound never underestimates), so this is exact, not approximate.
#[allow(clippy::too_many_arguments)]
pub fn pair_block_from_g_cand(
    i: usize,
    j: usize,
    g: &Array2<f64>,
    cand: &[usize],
    nv_full: usize,
    f_vv: &Array2<f64>,
    fo: &[f64],
    fv: &[f64],
    eps: f64,
) -> Option<PairBlock> {
    pair_block_from_g_cand_gated(i, j, g, g, cand, nv_full, f_vv, fo, fv, eps)
}

/// [`pair_block_from_g_cand`] with the Eq-8 test applied to a SEPARATE
/// gate quantity `gate` (same shape as `g`): element (a,b) is retained iff
/// `|gate[a,b]| > eps || |gate[b,a]| > eps` (or `eps == 0`), while the
/// retained integral values still come from `g`. `gate = g` is exactly
/// [`pair_block_from_g_cand`] (it delegates here with the same reads, so
/// every molecular caller is byte-identical — pinned by
/// `gated_with_gate_equal_to_g_is_bitwise_the_original`).
///
/// Seam for ferric-pbc's Gamma LMP2, where the raw integral carries a
/// distance-independent uniform-field term and a future gate may threshold
/// on the integral MINUS that term (reference/pbc/FINDINGS.md Iteration 5,
/// recommendation 6). No such gate is implemented here.
#[allow(clippy::too_many_arguments)]
pub fn pair_block_from_g_cand_gated(
    i: usize,
    j: usize,
    g: &Array2<f64>,
    gate: &Array2<f64>,
    cand: &[usize],
    nv_full: usize,
    f_vv: &Array2<f64>,
    fo: &[f64],
    fv: &[f64],
    eps: f64,
) -> Option<PairBlock> {
    assert_eq!(
        g.dim(),
        gate.dim(),
        "pair_block_from_g_cand_gated: gate shape differs from g"
    );
    let nv = cand.len();
    let _ = nv_full;
    let mut any_a = vec![false; nv];
    let mut any_b = vec![false; nv];
    let mut any = false;
    for a in 0..nv {
        for b in 0..nv {
            let jd = gate[(a, b)].abs();
            let kd = gate[(b, a)].abs();
            if eps == 0.0 || jd > eps || kd > eps {
                any_a[a] = true;
                any_b[b] = true;
                any = true;
            }
        }
    }
    if !any {
        return None;
    }
    // local (candidate-space) retained axes -> GLOBAL virtual indices
    let da_loc: Vec<usize> = (0..nv).filter(|&a| any_a[a]).collect();
    let db_loc: Vec<usize> = (0..nv).filter(|&b| any_b[b]).collect();
    let da: Vec<usize> = da_loc.iter().map(|&a| cand[a]).collect();
    let db: Vec<usize> = db_loc.iter().map(|&b| cand[b]).collect();
    let (na, nb) = (da.len(), db.len());
    let mut pat = vec![false; na * nb];
    let mut j_blk = Array2::<f64>::zeros((na, nb));
    let mut denom = Array2::<f64>::zeros((na, nb));
    for (r, &al) in da_loc.iter().enumerate() {
        for (c, &bl) in db_loc.iter().enumerate() {
            let jd = g[(al, bl)];
            let (gj, gk) = (gate[(al, bl)], gate[(bl, al)]);
            if eps == 0.0 || gj.abs() > eps || gk.abs() > eps {
                pat[r * nb + c] = true;
                j_blk[(r, c)] = jd;
            }
            denom[(r, c)] = fv[da[r]] + fv[db[c]] - fo[i] - fo[j];
        }
    }
    let mut fvv_aa = Array2::<f64>::zeros((na, na));
    for (r, &a) in da.iter().enumerate() {
        for (c, &a2) in da.iter().enumerate() {
            fvv_aa[(r, c)] = f_vv[(a, a2)];
        }
    }
    let mut fvv_bb = Array2::<f64>::zeros((nb, nb));
    for (r, &b) in db.iter().enumerate() {
        for (c, &b2) in db.iter().enumerate() {
            fvv_bb[(r, c)] = f_vv[(b, b2)];
        }
    }
    let nv_glob = fv.len();
    let mut pos_da = vec![usize::MAX; nv_glob];
    for (k, &a) in da.iter().enumerate() {
        pos_da[a] = k;
    }
    let mut pos_db = vec![usize::MAX; nv_glob];
    for (k, &b) in db.iter().enumerate() {
        pos_db[b] = k;
    }
    Some(PairBlock {
        i,
        j,
        da,
        db,
        pat,
        fvv_aa,
        fvv_bb,
        j_blk,
        denom,
        pos_da,
        pos_db,
    })
}

/// Zero out elements of `x` that lie outside the sparsity pattern `pat`.
pub fn apply_pattern(x: &mut Array2<f64>, pat: &[bool], nb: usize) {
    for (r, mut row) in x.rows_mut().into_iter().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            if !pat[r * nb + c] {
                *v = 0.0;
            }
        }
    }
}

/// Element-wise dot product summed over a slice of block matrices.
pub fn dot_blocks(x: &[Array2<f64>], y: &[Array2<f64>]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a * b).sum()).sum()
}

/// Ragged PCG on the fixed pattern: solve P A P t = −P J.
pub fn solve_ragged(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    rtol: f64,
    max_iter: usize,
) -> (Vec<Array2<f64>>, usize, f64, bool, u64) {
    let s = solve_ragged_traced(rg, f_oo, rtol, max_iter);
    (s.t, s.iterations, s.relres, s.converged, s.flops_per_matvec)
}

/// [`solve_ragged`] with a caller-supplied pattern-projected SPD matvec —
/// the entry point for operators that EXTEND the Fock superoperator
/// (LinLCCD's hh/pp ladders). The matvec MUST be symmetric positive
/// definite on the pattern and MUST pattern-project its output.
pub fn solve_ragged_with<F>(
    rg: &Ragged,
    rtol: f64,
    max_iter: usize,
    matvec: F,
) -> (Vec<Array2<f64>>, usize, f64, bool, u64)
where
    F: Fn(&[Array2<f64>], &mut u64) -> Vec<Array2<f64>>,
{
    let s = solve_ragged_with_trace(rg, rtol, max_iter, matvec);
    (s.t, s.iterations, s.relres, s.converged, s.flops_per_matvec)
}

/// The CG step coefficients of one ragged PCG solve: `alphas[k]` is the
/// step length of iteration k, `betas[k]` the direction update that
/// followed it (absent after the converging step). These are the Lanczos
/// coefficients of the PRECONDITIONED operator, so they give its Ritz
/// spectrum for free (see [`lanczos_kappa`]).
#[derive(Debug, Clone, Default)]
pub struct CgTrace {
    pub alphas: Vec<f64>,
    pub betas: Vec<f64>,
}

/// Everything one ragged PCG solve returns.
#[derive(Debug, Clone)]
pub struct RaggedSolve {
    pub t: Vec<Array2<f64>>,
    pub iterations: usize,
    pub relres: f64,
    pub converged: bool,
    pub flops_per_matvec: u64,
    pub trace: CgTrace,
}

/// [`solve_ragged`] that also returns the CG coefficient trace. The Fock
/// matvec runs through a [`SolvePlan`] built once per solve (bitwise
/// [`matvec_indexed`]).
pub fn solve_ragged_traced(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    rtol: f64,
    max_iter: usize,
) -> RaggedSolve {
    let plan = SolvePlan::new(rg, f_oo);
    pcg_core(rg, rtol, max_iter, |p, ap, flops| {
        matvec_planned(rg, &plan, p, ap, flops)
    })
}

/// Condition-number estimate `λmax/λmin` of the diagonally PRECONDITIONED
/// operator from the CG coefficients: the Lanczos tridiagonal
/// `T[k,k] = 1/α_k + β_{k-1}/α_{k-1}`, `T[k,k+1] = √β_k/α_k` has the Ritz
/// values of the Krylov space as eigenvalues. Ritz values lie inside the
/// spectrum, so this is a LOWER bound on the true κ (tight once CG has
/// resolved the extremes). NaN when no iteration ran or when the smallest
/// Ritz value is not positive (a non-SPD operator).
pub fn lanczos_kappa(trace: &CgTrace) -> f64 {
    use ndarray_linalg::Eigh;
    let k = trace.alphas.len();
    if k == 0 || trace.betas.len() + 1 < k {
        return f64::NAN;
    }
    let (a, b) = (&trace.alphas, &trace.betas);
    let mut tri = Array2::<f64>::zeros((k, k));
    for j in 0..k {
        tri[(j, j)] = 1.0 / a[j] + if j > 0 { b[j - 1] / a[j - 1] } else { 0.0 };
        if j + 1 < k {
            let off = b[j].sqrt() / a[j];
            tri[(j, j + 1)] = off;
            tri[(j + 1, j)] = off;
        }
    }
    let Ok((w, _)) = tri.eigh(ndarray_linalg::UPLO::Lower) else {
        return f64::NAN;
    };
    let (lo, hi) = (w[0], w[k - 1]);
    if lo > 0.0 {
        hi / lo
    } else {
        f64::NAN
    }
}

/// The PCG loop behind every ragged solve, recording the CG coefficients.
pub fn solve_ragged_with_trace<F>(rg: &Ragged, rtol: f64, max_iter: usize, matvec: F) -> RaggedSolve
where
    F: Fn(&[Array2<f64>], &mut u64) -> Vec<Array2<f64>>,
{
    pcg_core(rg, rtol, max_iter, |p, ap, flops| {
        for (slot, v) in ap.iter_mut().zip(matvec(p, flops)) {
            *slot = v;
        }
    })
}

/// The shared PCG loop on the fixed pattern, solving `P A P t = −P J` with
/// the diagonal (denominator) preconditioner. `matvec_into(p, ap, flops)`
/// overwrites every block of `ap` with `A·p`. The iteration vectors `z`,
/// `p`, `ap` are allocated once and updated in place; every update keeps
/// the per-element expression of the original allocating loop
/// (`z = r / denom`, `p = z + β·p`), and the reductions still go through
/// [`dot_blocks`], so the iterates are bitwise those of that loop.
fn pcg_core<F>(rg: &Ragged, rtol: f64, max_iter: usize, mut matvec_into: F) -> RaggedSolve
where
    F: FnMut(&[Array2<f64>], &mut [Array2<f64>], &mut u64),
{
    use ndarray::Zip;
    let zeros: Vec<Array2<f64>> = rg
        .pairs
        .iter()
        .map(|pb| Array2::<f64>::zeros((pb.da.len(), pb.db.len())))
        .collect();
    let mut r: Vec<Array2<f64>> = rg
        .pairs
        .iter()
        .map(|pb| {
            let mut b = -pb.j_blk.clone();
            apply_pattern(&mut b, &pb.pat, pb.db.len());
            b
        })
        .collect();
    let bnorm = dot_blocks(&r, &r).sqrt();
    let mut trace = CgTrace::default();
    if bnorm == 0.0 {
        return RaggedSolve {
            t: zeros,
            iterations: 0,
            relres: 0.0,
            converged: true,
            flops_per_matvec: 0,
            trace,
        };
    }
    let precond_into = |r: &[Array2<f64>], z: &mut [Array2<f64>]| {
        for ((zb, rb), pb) in z.iter_mut().zip(r).zip(&rg.pairs) {
            Zip::from(zb)
                .and(rb)
                .and(&pb.denom)
                .for_each(|zv, &rv, &dv| *zv = rv / dv);
        }
    };
    let mut flops_total = 0u64;
    let mut t = zeros.clone();
    let mut z = zeros.clone();
    precond_into(&r, &mut z);
    let mut p = z.clone();
    let mut ap = zeros;
    let mut rz = dot_blocks(&r, &z);
    let (mut it, mut relres, mut converged) = (0, 1.0, false);
    while it < max_iter {
        it += 1;
        matvec_into(&p, &mut ap, &mut flops_total);
        let alpha = rz / dot_blocks(&p, &ap);
        trace.alphas.push(alpha);
        for k in 0..p.len() {
            t[k].scaled_add(alpha, &p[k]);
            r[k].scaled_add(-alpha, &ap[k]);
        }
        relres = dot_blocks(&r, &r).sqrt() / bnorm;
        if relres < rtol {
            converged = true;
            break;
        }
        precond_into(&r, &mut z);
        let rz_new = dot_blocks(&r, &z);
        let beta = rz_new / rz;
        trace.betas.push(beta);
        for (pk, zk) in p.iter_mut().zip(&z) {
            Zip::from(pk)
                .and(zk)
                .for_each(|pv, &zv| *pv = zv + beta * *pv);
        }
        rz = rz_new;
    }
    let flops_per_mv = if it > 0 { flops_total / it as u64 } else { 0 };
    RaggedSolve {
        t,
        iterations: it,
        relres,
        converged,
        flops_per_matvec: flops_per_mv,
        trace,
    }
}

/// Index-passing form of the ragged matvec (the one actually used).
///
/// Rayon-parallel over output pairs (2026-08-17: the serial version was
/// the measured dominant solver cost — 62 of 108 s on alkane_12/eps=1e-4
/// while the rayon ring product took 29 s). Each block is computed wholly
/// inside one task with the SAME per-block arithmetic order as the serial
/// version (fvv GEMMs, by_j edges, by_i edges, pattern projection), so
/// the output is bit-identical and deterministic under any schedule.
pub fn matvec_indexed(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    t: &[Array2<f64>],
    flops: &mut u64,
) -> Vec<Array2<f64>> {
    use rayon::prelude::*;
    let results: Vec<(Array2<f64>, u64)> = rg
        .pairs
        .par_iter()
        .enumerate()
        .map(|(p_idx, pb)| {
            let (na, nb) = (pb.da.len(), pb.db.len());
            let mut fl = (na * na * nb + na * nb * nb) as u64;
            let mut r = pb.fvv_aa.dot(&t[p_idx]);
            r += &t[p_idx].dot(&pb.fvv_bb);
            for &q in &rg.by_j[&pb.j] {
                let qb = &rg.pairs[q];
                let f = f_oo[(pb.i, qb.i)];
                if f == 0.0 {
                    continue;
                }
                gather_into(&mut r, qb, pb, -f, &t[q], &mut fl);
            }
            for &q in &rg.by_i[&pb.i] {
                let qb = &rg.pairs[q];
                let f = f_oo[(qb.j, pb.j)];
                if f == 0.0 {
                    continue;
                }
                gather_into(&mut r, qb, pb, -f, &t[q], &mut fl);
            }
            apply_pattern(&mut r, &pb.pat, nb);
            (r, fl)
        })
        .collect();
    let mut out = Vec::with_capacity(results.len());
    for (r, fl) in results {
        *flops += fl;
        out.push(r);
    }
    out
}

/// One Fock-coupling edge of an output pair's matvec, pre-resolved from the
/// pattern: source pair `q`, coefficient `-f_oo[..]`, and the
/// `(destination, source)` flat-index pairs [`gather_into`] would visit, in
/// its order.
#[derive(Debug, Clone)]
struct PlanEdge {
    q: usize,
    coeff: f64,
    idx: Vec<(u32, u32)>,
}

fn flat_u32(k: usize) -> u32 {
    u32::try_from(k).expect("pair block larger than 2^32 elements")
}

/// The hoisted gather plan of [`matvec_indexed`] for a FIXED pattern and
/// `f_oo`: every edge's index intersections (recomputed on every call and
/// every CG iteration by [`gather_into`]) resolved once. Edges with a zero
/// Fock coupling are dropped at build time, exactly as the indexed matvec
/// skips them; the edge order is by_j then by_i, as there.
#[derive(Debug, Clone)]
pub struct SolvePlan {
    edges: Vec<Vec<PlanEdge>>,
}

impl SolvePlan {
    pub fn new(rg: &Ragged, f_oo: &Array2<f64>) -> Self {
        let edge = |dst: &PairBlock, q: usize, f: f64| -> PlanEdge {
            let src = &rg.pairs[q];
            let nb_src = src.db.len();
            let nb = dst.db.len();
            let mut idx = Vec::new();
            for (r, &a) in dst.da.iter().enumerate() {
                let sr = src.pos_da[a];
                if sr == usize::MAX {
                    continue;
                }
                for (c, &b) in dst.db.iter().enumerate() {
                    let sc = src.pos_db[b];
                    if sc == usize::MAX {
                        continue;
                    }
                    idx.push((flat_u32(r * nb + c), flat_u32(sr * nb_src + sc)));
                }
            }
            PlanEdge { q, coeff: -f, idx }
        };
        let edges = rg
            .pairs
            .iter()
            .map(|pb| {
                let mut out = Vec::new();
                for &q in &rg.by_j[&pb.j] {
                    let f = f_oo[(pb.i, rg.pairs[q].i)];
                    if f != 0.0 {
                        out.push(edge(pb, q, f));
                    }
                }
                for &q in &rg.by_i[&pb.i] {
                    let f = f_oo[(rg.pairs[q].j, pb.j)];
                    if f != 0.0 {
                        out.push(edge(pb, q, f));
                    }
                }
                out
            })
            .collect();
        Self { edges }
    }
}

/// [`matvec_indexed`] through a [`SolvePlan`], writing into `out` (one
/// block per pair, any prior contents replaced). Per block: the same two
/// Fvv GEMMs, the same edge order and per-element `+= coeff·t`, the same
/// pattern projection — so the result is bitwise [`matvec_indexed`]'s.
pub fn matvec_planned(
    rg: &Ragged,
    plan: &SolvePlan,
    t: &[Array2<f64>],
    out: &mut [Array2<f64>],
    flops: &mut u64,
) {
    use rayon::prelude::*;
    let fl: u64 = out
        .par_iter_mut()
        .enumerate()
        .map(|(p_idx, slot)| {
            let pb = &rg.pairs[p_idx];
            let (na, nb) = (pb.da.len(), pb.db.len());
            let mut fl = (na * na * nb + na * nb * nb) as u64;
            let mut r = pb.fvv_aa.dot(&t[p_idx]);
            r += &t[p_idx].dot(&pb.fvv_bb);
            if !r.is_standard_layout() {
                r = r.as_standard_layout().into_owned();
            }
            let rs = r.as_slice_mut().expect("standard layout");
            for e in &plan.edges[p_idx] {
                let src_std = t[e.q].as_standard_layout();
                let src = src_std.as_slice().expect("standard layout");
                for &(d, s) in &e.idx {
                    rs[d as usize] += e.coeff * src[s as usize];
                }
                fl += e.idx.len() as u64;
            }
            apply_pattern(&mut r, &pb.pat, nb);
            *slot = r;
            fl
        })
        .sum();
    *flops += fl;
}

/// Gather and accumulate a sub-block of the ring product into `out`.
pub fn gather_into(
    out: &mut Array2<f64>,
    src: &PairBlock,
    dst: &PairBlock,
    coeff: f64,
    tq: &Array2<f64>,
    flops: &mut u64,
) {
    let mut n = 0u64;
    for (r, &a) in dst.da.iter().enumerate() {
        let sr = src.pos_da[a];
        if sr == usize::MAX {
            continue;
        }
        for (c, &b) in dst.db.iter().enumerate() {
            let sc = src.pos_db[b];
            if sc == usize::MAX {
                continue;
            }
            out[(r, c)] += coeff * tq[(sr, sc)];
            n += 1;
        }
    }
    *flops += n;
}

/// One (i,k,j) contributing triple to an output pair's ring-product sum,
/// pre-resolved from the pattern alone (no operand data): the contraction
/// set `Db_ik ∩ Da_kj` and the output row/col intersections, each stored as
/// `(destination_index, source_index)` pairs exactly like the per-call
/// computation in [`ring_product`] — see that function's doc for the
/// definitions. `panel_x` is `x`'s `(rows.len(), cset.len())` sub-block,
/// GATHERED ONCE here because the caller promises `x` is constant across
/// the whole solve (see [`RingPlan::new`]).
#[derive(Debug)]
struct RingTriple {
    p_kj: usize,
    cset: Vec<(usize, usize)>,
    rows: Vec<(usize, usize)>,
    cols: Vec<(usize, usize)>,
    panel_x: Array2<f64>,
}

/// Pre-resolved ring-product plan for a FIXED pattern (`rg`) and a FIXED
/// first operand `x` — built once per solve, reused every fixed-point
/// iteration. Amortizes two things that [`ring_product`] recomputes on
/// every call even though neither depends on the varying operand `y`:
///
/// 1. the plan/intersection bookkeeping (rows/cset/cols index pairs) —
///    measured 8.4% of ring_product's serial wall (2026-08-17 profile);
/// 2. `x`'s sub-block gathers (the `xa` panel in [`ring_product`]) — valid
///    to cache ONLY because the dRPA fixed point's `ring_product(B, T)`
///    call uses the same `B` (the integral blocks) as `x` on every
///    iteration; `T` (the `y` operand) changes every iteration and is
///    still gathered fresh in [`ring_product_planned`].
///
/// Per-iteration work with a plan is therefore: gather `y`'s sub-block +
/// one GEMM + scatter-add, per triple — the gather-39%/gemm-41% panel
/// batching attempt (2026-08-17, feat/drpa-ragged-blas3) that zero-padded
/// BOTH operands into one big GEMM measured 24% SLOWER (FLOP inflation
/// from padding beat the plan-hoist saving); this does NOT repeat that:
/// no padding, one GEMM per triple exactly as before, only the constant
/// side's gather and the index bookkeeping are hoisted out of the loop.
#[derive(Debug)]
pub struct RingPlan {
    /// same length/order as `rg.pairs`; `dims[out_p]` is that output
    /// pair's `(na, nb)` block shape (needed to emit the right-shaped
    /// zero block even when `triples[out_p]` is empty).
    dims: Vec<(usize, usize)>,
    /// `triples[out_p]` are the (i,k,j) contributions to output pair
    /// `out_p`.
    triples: Vec<Vec<RingTriple>>,
}

/// The pattern-only index sets of one (i,k,j) ring-product triple:
/// `(p_ik, p_kj, cset, rows, cols)` exactly as [`RingTriple`] stores them.
type TripleIndex = (
    usize,
    usize,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
);

/// Every contributing triple of output pair `out_pb`, resolved from the
/// pattern alone. Shared by [`RingPlan::new`] (which then gathers the
/// constant operand's panels) and [`RingPlan::bytes_for`] (which only sizes
/// them), so the size a caller budgets for and the plan that is built cannot
/// disagree.
fn triple_index_sets(
    rg: &Ragged,
    pair_index: &HashMap<(usize, usize), usize>,
    out_pb: &PairBlock,
) -> Vec<TripleIndex> {
    let (i, j) = (out_pb.i, out_pb.j);
    let mut out = Vec::new();
    let Some(iks) = rg.by_i.get(&i) else {
        return out;
    };
    for &p_ik in iks {
        let ik = &rg.pairs[p_ik];
        let k = ik.j;
        let Some(&p_kj) = pair_index.get(&(k, j)) else {
            continue;
        };
        let kj = &rg.pairs[p_kj];
        let cset: Vec<(usize, usize)> = ik
            .db
            .iter()
            .enumerate()
            .filter_map(|(cx, &c)| {
                let ry = kj.pos_da[c];
                (ry != usize::MAX).then_some((cx, ry))
            })
            .collect();
        if cset.is_empty() {
            continue;
        }
        let rows: Vec<(usize, usize)> = out_pb
            .da
            .iter()
            .enumerate()
            .filter_map(|(r, &a)| {
                let rx = ik.pos_da[a];
                (rx != usize::MAX).then_some((r, rx))
            })
            .collect();
        let cols: Vec<(usize, usize)> = out_pb
            .db
            .iter()
            .enumerate()
            .filter_map(|(c, &b)| {
                let cy = kj.pos_db[b];
                (cy != usize::MAX).then_some((c, cy))
            })
            .collect();
        if rows.is_empty() || cols.is_empty() {
            continue;
        }
        out.push((p_ik, p_kj, cset, rows, cols));
    }
    out
}

fn pair_index_of(rg: &Ragged) -> HashMap<(usize, usize), usize> {
    rg.pairs
        .iter()
        .enumerate()
        .map(|(p, pb)| ((pb.i, pb.j), p))
        .collect()
}

impl RingPlan {
    /// Bytes a [`RingPlan`] built on `rg` holds: every gathered `panel_x`
    /// (`rows × cset` f64) plus its three `(usize, usize)` index lists.
    ///
    /// This is the dominant allocation of the dRPA Riccati solve when the
    /// pattern is full: at ε = 0 every output pair has `no` triples of
    /// `nv × nv` panels, i.e. `no` times the size of B itself. Computed from
    /// the same index sets [`RingPlan::new`] uses, without gathering
    /// anything.
    pub fn bytes_for(rg: &Ragged) -> usize {
        let pair_index = pair_index_of(rg);
        let idx = std::mem::size_of::<(usize, usize)>();
        rg.pairs
            .iter()
            .map(|out_pb| {
                triple_index_sets(rg, &pair_index, out_pb)
                    .iter()
                    .map(|(_, _, cset, rows, cols)| {
                        rows.len() * cset.len() * 8 + (cset.len() + rows.len() + cols.len()) * idx
                    })
                    .sum::<usize>()
            })
            .sum()
    }

    /// Build the plan for ring products with a CONSTANT first operand `x`
    /// (caller's responsibility: `x` must not change for the lifetime of
    /// this plan — the dRPA solver holds `x = b_blocks`, assembled once
    /// before the fixed-point loop starts).
    pub fn new(rg: &Ragged, x: &[Array2<f64>]) -> RingPlan {
        let pair_index = pair_index_of(rg);
        let dims: Vec<(usize, usize)> = rg
            .pairs
            .iter()
            .map(|pb| (pb.da.len(), pb.db.len()))
            .collect();
        let triples: Vec<Vec<RingTriple>> = rg
            .pairs
            .iter()
            .map(|out_pb| {
                triple_index_sets(rg, &pair_index, out_pb)
                    .into_iter()
                    .map(|(p_ik, p_kj, cset, rows, cols)| {
                        let xs = &x[p_ik];
                        let mut panel_x = Array2::<f64>::zeros((rows.len(), cset.len()));
                        for (rr, &(_, rx)) in rows.iter().enumerate() {
                            for (cc, &(cx, _)) in cset.iter().enumerate() {
                                panel_x[(rr, cc)] = xs[(rx, cx)];
                            }
                        }
                        RingTriple {
                            p_kj,
                            cset,
                            rows,
                            cols,
                            panel_x,
                        }
                    })
                    .collect()
            })
            .collect();
        RingPlan { dims, triples }
    }
}

/// [`ring_product`] using a pre-built [`RingPlan`]: skips the plan
/// bookkeeping and the constant operand's gathers, both hoisted into the
/// plan at construction. `y` is the varying operand (gathered fresh here
/// every call) and must be indexed identically to the `rg` the plan was
/// built from (same pair order/count) — same contract as [`ring_product`].
pub fn ring_product_planned(plan: &RingPlan, y: &[Array2<f64>]) -> Vec<Array2<f64>> {
    use rayon::prelude::*;
    plan.triples
        .par_iter()
        .zip(&plan.dims)
        .map(|(triples, &(na, nb))| {
            let mut acc = Array2::<f64>::zeros((na, nb));
            for t in triples {
                accumulate_triple(&mut acc, t, y);
            }
            acc
        })
        .collect()
}

#[inline]
fn accumulate_triple(acc: &mut Array2<f64>, t: &RingTriple, y: &[Array2<f64>]) {
    let ys = &y[t.p_kj];
    let mut yb = Array2::<f64>::zeros((t.cset.len(), t.cols.len()));
    for (rr, &(_, ry)) in t.cset.iter().enumerate() {
        for (cc, &(_, cy)) in t.cols.iter().enumerate() {
            yb[(rr, cc)] = ys[(ry, cy)];
        }
    }
    let prod = t.panel_x.dot(&yb);
    for (rr, &(r, _)) in t.rows.iter().enumerate() {
        for (cc, &(c, _)) in t.cols.iter().enumerate() {
            acc[(r, c)] += prod[(rr, cc)];
        }
    }
}

/// Ragged pair-space ring product: `(X·Y)_ij = Σ_k X_ik Y_kj`, projected
/// onto the OUTPUT pair's pattern — the contraction the drCCD ring terms
/// (BT, TB, TBT) need, with every factor living on the SAME ragged
/// structure `rg` (dRPA's B and T share the Eq-8 pattern by construction).
///
/// Per output pair (i, j) and middle occupied k with both (i,k) and (k,j)
/// blocks present: the contraction index runs over Db_ik ∩ Da_kj, output
/// rows over Da_ij ∩ Da_ik, output columns over Db_ij ∩ Db_kj — three
/// intersections resolved through the O(1) position maps, then ONE
/// domain-sized GEMM per (i,k,j) triple and a scatter-add. Cost scales
/// with present pairs × domain³ instead of the dense (no·nv)³.
///
/// Rayon-parallel over output pairs (each writes only its own block —
/// deterministic by construction). The result is NOT pattern-masked here;
/// callers project with [`apply_pattern`] when required.
pub fn ring_product(rg: &Ragged, x: &[Array2<f64>], y: &[Array2<f64>]) -> Vec<Array2<f64>> {
    use rayon::prelude::*;
    let pair_index: HashMap<(usize, usize), usize> = rg
        .pairs
        .iter()
        .enumerate()
        .map(|(p, pb)| ((pb.i, pb.j), p))
        .collect();
    rg.pairs
        .par_iter()
        .map(|out_pb| {
            let (i, j) = (out_pb.i, out_pb.j);
            let (na, nb) = (out_pb.da.len(), out_pb.db.len());
            let mut acc = Array2::<f64>::zeros((na, nb));
            // middle occupieds: pairs (i, k) present via by_i[i]
            if let Some(iks) = rg.by_i.get(&i) {
                for &p_ik in iks {
                    let ik = &rg.pairs[p_ik];
                    let k = ik.j;
                    let Some(&p_kj) = pair_index.get(&(k, j)) else {
                        continue;
                    };
                    let kj = &rg.pairs[p_kj];
                    // contraction set: Db_ik ∩ Da_kj (global virtual ids)
                    let cset: Vec<(usize, usize)> = ik
                        .db
                        .iter()
                        .enumerate()
                        .filter_map(|(cx, &c)| {
                            let ry = kj.pos_da[c];
                            (ry != usize::MAX).then_some((cx, ry))
                        })
                        .collect();
                    if cset.is_empty() {
                        continue;
                    }
                    // output rows: Da_ij ∩ Da_ik; output cols: Db_ij ∩ Db_kj
                    let rows: Vec<(usize, usize)> = out_pb
                        .da
                        .iter()
                        .enumerate()
                        .filter_map(|(r, &a)| {
                            let rx = ik.pos_da[a];
                            (rx != usize::MAX).then_some((r, rx))
                        })
                        .collect();
                    let cols: Vec<(usize, usize)> = out_pb
                        .db
                        .iter()
                        .enumerate()
                        .filter_map(|(c, &b)| {
                            let cy = kj.pos_db[b];
                            (cy != usize::MAX).then_some((c, cy))
                        })
                        .collect();
                    if rows.is_empty() || cols.is_empty() {
                        continue;
                    }
                    // gather domain-sized sub-blocks and GEMM
                    let xs = &x[p_ik];
                    let ys = &y[p_kj];
                    let mut xa = Array2::<f64>::zeros((rows.len(), cset.len()));
                    for (rr, &(_, rx)) in rows.iter().enumerate() {
                        for (cc, &(cx, _)) in cset.iter().enumerate() {
                            xa[(rr, cc)] = xs[(rx, cx)];
                        }
                    }
                    let mut yb = Array2::<f64>::zeros((cset.len(), cols.len()));
                    for (rr, &(_, ry)) in cset.iter().enumerate() {
                        for (cc, &(_, cy)) in cols.iter().enumerate() {
                            yb[(rr, cc)] = ys[(ry, cy)];
                        }
                    }
                    let prod = xa.dot(&yb);
                    for (rr, &(r, _)) in rows.iter().enumerate() {
                        for (cc, &(c, _)) in cols.iter().enumerate() {
                            acc[(r, c)] += prod[(rr, cc)];
                        }
                    }
                }
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One (1 × m) pair block with a full pattern and the given denominators.
    fn diag_block(denom: &[f64]) -> Ragged {
        let m = denom.len();
        let j_blk = Array2::from_shape_fn((1, m), |(_, c)| 0.3 + 0.1 * c as f64);
        let pb = PairBlock {
            i: 0,
            j: 0,
            da: vec![0],
            db: (0..m).collect(),
            pat: vec![true; m],
            fvv_aa: Array2::zeros((1, 1)),
            fvv_bb: Array2::zeros((m, m)),
            j_blk,
            denom: Array2::from_shape_vec((1, m), denom.to_vec()).unwrap(),
            pos_da: vec![0],
            pos_db: (0..m).collect(),
        };
        Ragged {
            pairs: vec![pb],
            by_i: HashMap::from([(0, vec![0])]),
            by_j: HashMap::from([(0, vec![0])]),
        }
    }

    /// The CG-coefficient κ estimate on an operator with a KNOWN spectrum:
    /// A = diag(1..m) with a unit preconditioner is plain CG, which resolves
    /// all m distinct eigenvalues in m steps, so κ_est must be m; with the
    /// exact diagonal as preconditioner the preconditioned operator is the
    /// identity (κ = 1, one step).
    #[test]
    fn lanczos_kappa_recovers_a_known_spectrum() {
        let m = 6;
        let d: Vec<f64> = (1..=m).map(|x| x as f64).collect();
        let matvec = |p: &[Array2<f64>], _: &mut u64| -> Vec<Array2<f64>> {
            vec![Array2::from_shape_fn((1, m), |(_, c)| p[0][(0, c)] * d[c])]
        };
        let plain = solve_ragged_with_trace(&diag_block(&[1.0; 6]), 1e-13, 50, matvec);
        assert!(plain.converged);
        let k = lanczos_kappa(&plain.trace);
        assert!(
            (k - m as f64).abs() < 1e-8 * m as f64,
            "unit preconditioner: kappa {k} != {m}"
        );
        let pre = solve_ragged_with_trace(&diag_block(&d), 1e-13, 50, matvec);
        assert_eq!(pre.iterations, 1);
        let k1 = lanczos_kappa(&pre.trace);
        assert!((k1 - 1.0).abs() < 1e-12, "exact preconditioner: kappa {k1}");
        assert!(lanczos_kappa(&CgTrace::default()).is_nan());
    }

    /// Byte-identity guard for the ferric-pbc seam: with `gate = g` the gated
    /// builder IS the original (every field bitwise), on a block with
    /// elements straddling eps in both orientations.
    #[test]
    fn gated_with_gate_equal_to_g_is_bitwise_the_original() {
        let nv = 5;
        let mut seed = 99u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let g = Array2::from_shape_fn((nv, nv), |_| 1e-3 * rnd());
        let f_vv = Array2::from_shape_fn(
            (nv, nv),
            |(a, b)| if a == b { 1.0 + a as f64 } else { 0.01 },
        );
        let fo = vec![-1.0, -0.7];
        let fv: Vec<f64> = (0..nv).map(|a| 1.0 + a as f64).collect();
        let cand: Vec<usize> = (0..nv).collect();
        for eps in [0.0, 1e-4, 2e-4, 1.0] {
            let a = pair_block_from_g_cand(0, 1, &g, &cand, nv, &f_vv, &fo, &fv, eps);
            let b = pair_block_from_g_cand_gated(0, 1, &g, &g, &cand, nv, &f_vv, &fo, &fv, eps);
            match (a, b) {
                (None, None) => assert_eq!(eps, 1.0),
                (Some(a), Some(b)) => {
                    assert_eq!(
                        (a.i, a.j, &a.da, &a.db, &a.pat),
                        (b.i, b.j, &b.da, &b.db, &b.pat)
                    );
                    assert_eq!((&a.pos_da, &a.pos_db), (&b.pos_da, &b.pos_db));
                    for (x, y) in [
                        (&a.j_blk, &b.j_blk),
                        (&a.denom, &b.denom),
                        (&a.fvv_aa, &b.fvv_aa),
                        (&a.fvv_bb, &b.fvv_bb),
                    ] {
                        assert!(x
                            .iter()
                            .zip(y.iter())
                            .all(|(p, q)| p.to_bits() == q.to_bits()));
                    }
                }
                _ => panic!("gated and original disagree on emptiness at eps {eps}"),
            }
        }
        // And the gate is really consulted: an all-zero gate at eps > 0
        // retains nothing although g has elements above eps.
        let zero = Array2::<f64>::zeros((nv, nv));
        assert!(
            pair_block_from_g_cand_gated(0, 1, &g, &zero, &cand, nv, &f_vv, &fo, &fv, 1e-4)
                .is_none()
        );
    }
}
