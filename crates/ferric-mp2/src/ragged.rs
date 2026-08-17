//! Shared ragged pair-block machinery for the amplitude-threshold family
//! (LMP2 linear solve; dRPA ring products; LinLCCD ladders). Extracted from
//! `lmp2_amplitude` when the dRPA/LinLCCD ports moved onto the direct
//! assembly path — the block layout, Eq-8 pattern conventions, and the
//! solver are documented there and in wiki/amplitude-threshold-lmp2.md.

use ndarray::Array2;
use std::collections::HashMap;

pub struct PairBlock {
    pub i: usize,
    pub j: usize,
    pub da: Vec<usize>,
    pub db: Vec<usize>,
    /// row-major (da.len() × db.len()) pattern
    pub pat: Vec<bool>,
    pub fvv_aa: Array2<f64>,
    pub fvv_bb: Array2<f64>,
    pub j_blk: Array2<f64>,
    pub denom: Array2<f64>,
    /// inverse maps: pos_da[v] = column of v in da, usize::MAX if absent
    pub pos_da: Vec<usize>,
    pub pos_db: Vec<usize>,
}

pub struct Ragged {
    pub pairs: Vec<PairBlock>,
    pub by_i: HashMap<usize, Vec<usize>>,
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
    let nv = cand.len();
    let _ = nv_full;
    let mut any_a = vec![false; nv];
    let mut any_b = vec![false; nv];
    let mut any = false;
    for a in 0..nv {
        for b in 0..nv {
            let jd = g[(a, b)].abs();
            let kd = g[(b, a)].abs();
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
            let kd = g[(bl, al)];
            if eps == 0.0 || jd.abs() > eps || kd.abs() > eps {
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
    Some(PairBlock { i, j, da, db, pat, fvv_aa, fvv_bb, j_blk, denom, pos_da, pos_db })
}


pub fn apply_pattern(x: &mut Array2<f64>, pat: &[bool], nb: usize) {
    for (r, mut row) in x.rows_mut().into_iter().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            if !pat[r * nb + c] {
                *v = 0.0;
            }
        }
    }
}

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
    solve_ragged_with(rg, rtol, max_iter, |t, flops| matvec_indexed(rg, f_oo, t, flops))
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
    let npair = rg.pairs.len();
    let zeros: Vec<Array2<f64>> = rg
        .pairs
        .iter()
        .map(|pb| Array2::<f64>::zeros((pb.da.len(), pb.db.len())))
        .collect();
    let mut rhs: Vec<Array2<f64>> = Vec::with_capacity(npair);
    for pb in &rg.pairs {
        let mut b = -pb.j_blk.clone();
        apply_pattern(&mut b, &pb.pat, pb.db.len());
        rhs.push(b);
    }
    let bnorm = dot_blocks(&rhs, &rhs).sqrt();
    if bnorm == 0.0 {
        return (zeros, 0, 0.0, true, 0);
    }
    let precond = |r: &[Array2<f64>]| -> Vec<Array2<f64>> {
        r.iter()
            .zip(&rg.pairs)
            .map(|(rb, pb)| rb / &pb.denom)
            .collect()
    };
    let mut flops_total = 0u64;
    let mut t = zeros.clone();
    let mut r = rhs.clone();
    let mut z = precond(&r);
    let mut p: Vec<Array2<f64>> = z.to_vec();
    let mut rz = dot_blocks(&r, &z);
    let mut it = 0;
    let mut relres = 1.0;
    let mut converged = false;
    while it < max_iter {
        it += 1;
        let ap = matvec(&p, &mut flops_total);
        let alpha = rz / dot_blocks(&p, &ap);
        for k in 0..npair {
            t[k].scaled_add(alpha, &p[k]);
            r[k].scaled_add(-alpha, &ap[k]);
        }
        relres = dot_blocks(&r, &r).sqrt() / bnorm;
        if relres < rtol {
            converged = true;
            break;
        }
        z = precond(&r);
        let rz_new = dot_blocks(&r, &z);
        let beta = rz_new / rz;
        for k in 0..npair {
            let mut pk = z[k].clone();
            pk.scaled_add(beta, &p[k]);
            p[k] = pk;
        }
        rz = rz_new;
    }
    let flops_per_mv = if it > 0 { flops_total / it as u64 } else { 0 };
    (t, it, relres, converged, flops_per_mv)
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
pub fn ring_product(
    rg: &Ragged,
    x: &[Array2<f64>],
    y: &[Array2<f64>],
) -> Vec<Array2<f64>> {
    use rayon::prelude::*;
    let pair_index: HashMap<(usize, usize), usize> =
        rg.pairs.iter().enumerate().map(|(p, pb)| ((pb.i, pb.j), p)).collect();
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
                    let Some(&p_kj) = pair_index.get(&(k, j)) else { continue };
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
