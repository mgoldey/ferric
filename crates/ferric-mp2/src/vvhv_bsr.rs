//! Atom-group block-sparse matrices for the local VV-HV build
//! ([`crate::vvhv_local`]).
//!
//! A [`Partition`] splits an index space (AOs, minimal functions, candidates,
//! orbitals) into groups; a [`Bsr`] stores the dense blocks of a matrix
//! between a row partition and a column partition, one row of blocks per row
//! group. A block whose Frobenius norm is below the product's threshold is
//! dropped (`tau = 0` drops nothing, so every operation is then exact up to
//! summation order: the trivial limit of the local build).

use std::sync::Arc;

use ndarray::{linalg::general_mat_mul, Array2};
use rayon::prelude::*;

use ferric_core::error::FerricError;

/// A partition of `0..n` into groups. Indices inside a group are ascending.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Partition {
    pub groups: Vec<Vec<usize>>,
    /// Global index → (group, position in group).
    pub place: Vec<(usize, usize)>,
}

impl Partition {
    /// Groups from per-index labels in `0..ngroups` (empty groups are kept so
    /// two partitions built from the same group set share group ids).
    pub fn from_labels(labels: &[usize], ngroups: usize) -> Arc<Self> {
        let mut groups = vec![Vec::new(); ngroups];
        let mut place = Vec::with_capacity(labels.len());
        for (i, &g) in labels.iter().enumerate() {
            place.push((g, groups[g].len()));
            groups[g].push(i);
        }
        Arc::new(Self { groups, place })
    }

    pub fn n(&self) -> usize {
        self.place.len()
    }

    pub fn ngroups(&self) -> usize {
        self.groups.len()
    }
}

/// Block counters a truncated operation adds to.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BlockCount {
    pub kept: u64,
    pub dropped: u64,
    pub products: usize,
}

impl BlockCount {
    pub fn add(&mut self, o: BlockCount) {
        self.kept += o.kept;
        self.dropped += o.dropped;
        self.products += o.products;
    }
}

/// Block-sparse matrix: `blk[gi]` = the kept blocks of row group `gi`, as
/// `(column group, dense block)` sorted by column group.
#[derive(Debug, Clone)]
pub(crate) struct Bsr {
    pub rows: Arc<Partition>,
    pub cols: Arc<Partition>,
    pub blk: Vec<Vec<(usize, Array2<f64>)>>,
}

fn frob(b: &Array2<f64>) -> f64 {
    b.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Keep `b` iff its Frobenius norm is >= `tau` (`tau = 0` keeps everything).
fn keep(b: &Array2<f64>, tau: f64) -> bool {
    tau <= 0.0 || frob(b) >= tau
}

fn same(a: &Arc<Partition>, b: &Arc<Partition>) -> bool {
    Arc::ptr_eq(a, b) || **a == **b
}

impl Bsr {
    pub fn zeros(rows: Arc<Partition>, cols: Arc<Partition>) -> Self {
        let blk = vec![Vec::new(); rows.ngroups()];
        Self { rows, cols, blk }
    }

    pub fn identity(p: Arc<Partition>) -> Self {
        let mut m = Self::zeros(p.clone(), p.clone());
        for (g, idx) in p.groups.iter().enumerate() {
            if !idx.is_empty() {
                m.blk[g].push((g, Array2::eye(idx.len())));
            }
        }
        m
    }

    /// Blocks of a dense matrix, dropping those below `tau`.
    pub fn from_dense(
        d: &Array2<f64>,
        rows: Arc<Partition>,
        cols: Arc<Partition>,
        tau: f64,
        cnt: &mut BlockCount,
    ) -> Self {
        assert_eq!(d.dim(), (rows.n(), cols.n()), "Bsr::from_dense shape");
        let blk: Vec<(Vec<(usize, Array2<f64>)>, u64, u64)> = rows
            .groups
            .par_iter()
            .map(|ri| {
                let (mut out, mut k, mut dr) = (Vec::new(), 0u64, 0u64);
                if ri.is_empty() {
                    return (out, k, dr);
                }
                for (gj, cj) in cols.groups.iter().enumerate() {
                    if cj.is_empty() {
                        continue;
                    }
                    let b = Array2::from_shape_fn((ri.len(), cj.len()), |(r, c)| d[(ri[r], cj[c])]);
                    if keep(&b, tau) {
                        out.push((gj, b));
                        k += 1;
                    } else {
                        dr += 1;
                    }
                }
                (out, k, dr)
            })
            .collect();
        let mut m = Self::zeros(rows, cols);
        for (g, (b, k, dr)) in blk.into_iter().enumerate() {
            m.blk[g] = b;
            cnt.kept += k;
            cnt.dropped += dr;
        }
        m
    }

    /// Drop the blocks below `tau` (counted as dropped; the rest as kept).
    pub fn drop_below(mut self, tau: f64, cnt: &mut BlockCount) -> Self {
        for row in &mut self.blk {
            let before = row.len() as u64;
            row.retain(|(_, b)| keep(b, tau));
            cnt.kept += row.len() as u64;
            cnt.dropped += before - row.len() as u64;
        }
        self
    }

    pub fn to_dense(&self) -> Array2<f64> {
        let mut d = Array2::<f64>::zeros((self.rows.n(), self.cols.n()));
        for (gi, row) in self.blk.iter().enumerate() {
            let ri = &self.rows.groups[gi];
            for (gj, b) in row {
                let cj = &self.cols.groups[*gj];
                for (r, &ir) in ri.iter().enumerate() {
                    for (c, &ic) in cj.iter().enumerate() {
                        d[(ir, ic)] = b[(r, c)];
                    }
                }
            }
        }
        d
    }

    pub fn nnz_blocks(&self) -> usize {
        self.blk.iter().map(Vec::len).sum()
    }

    /// Mean and max of (kept blocks in a block row) / (column groups).
    pub fn row_fill(&self) -> (f64, f64) {
        let ncg = self
            .cols
            .groups
            .iter()
            .filter(|g| !g.is_empty())
            .count()
            .max(1) as f64;
        let rows: Vec<f64> = self
            .blk
            .iter()
            .zip(&self.rows.groups)
            .filter(|(_, g)| !g.is_empty())
            .map(|(r, _)| r.len() as f64 / ncg)
            .collect();
        let n = rows.len().max(1) as f64;
        (
            rows.iter().sum::<f64>() / n,
            rows.iter().copied().fold(0.0, f64::max),
        )
    }

    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols.clone(), self.rows.clone());
        for (gi, row) in self.blk.iter().enumerate() {
            for (gj, b) in row {
                t.blk[*gj].push((gi, b.t().to_owned()));
            }
        }
        for row in &mut t.blk {
            row.sort_by_key(|(g, _)| *g);
        }
        t
    }

    /// `alpha·self + beta·other` (block union), dropping blocks below `tau`.
    pub fn axpby(
        &self,
        alpha: f64,
        other: &Bsr,
        beta: f64,
        tau: f64,
        cnt: &mut BlockCount,
    ) -> Self {
        assert!(same(&self.rows, &other.rows) && same(&self.cols, &other.cols));
        let mut out = Self::zeros(self.rows.clone(), self.cols.clone());
        for gi in 0..self.blk.len() {
            let (a, b) = (&self.blk[gi], &other.blk[gi]);
            let (mut i, mut j) = (0, 0);
            let mut row = Vec::with_capacity(a.len().max(b.len()));
            while i < a.len() || j < b.len() {
                let ga = a.get(i).map_or(usize::MAX, |x| x.0);
                let gb = b.get(j).map_or(usize::MAX, |x| x.0);
                let (g, blk) = if ga == gb {
                    let r = (&a[i].1 * alpha) + (&b[j].1 * beta);
                    i += 1;
                    j += 1;
                    (ga, r)
                } else if ga < gb {
                    i += 1;
                    (ga, &a[i - 1].1 * alpha)
                } else {
                    j += 1;
                    (gb, &b[j - 1].1 * beta)
                };
                if keep(&blk, tau) {
                    row.push((g, blk));
                    cnt.kept += 1;
                } else {
                    cnt.dropped += 1;
                }
            }
            out.blk[gi] = row;
        }
        out
    }

    /// `self · other`, dropping result blocks below `tau`. Parallel over row
    /// groups; each result block accumulates in column-group order of the
    /// inner index, so the result is deterministic.
    pub fn mul(&self, other: &Bsr, tau: f64, cnt: &mut BlockCount) -> Self {
        assert!(same(&self.cols, &other.rows), "Bsr::mul inner partition");
        let ncg = other.cols.ngroups();
        let rows: Vec<(Vec<(usize, Array2<f64>)>, u64, u64)> = self
            .blk
            .par_iter()
            .enumerate()
            .map(|(gi, arow)| {
                let nr = self.rows.groups[gi].len();
                let mut acc: Vec<Option<Array2<f64>>> = vec![None; ncg];
                for (gk, a) in arow {
                    for (gj, b) in &other.blk[*gk] {
                        let c = acc[*gj].get_or_insert_with(|| Array2::zeros((nr, b.ncols())));
                        general_mat_mul(1.0, a, b, 1.0, c);
                    }
                }
                let (mut out, mut k, mut dr) = (Vec::new(), 0u64, 0u64);
                for (gj, c) in acc.into_iter().enumerate() {
                    if let Some(c) = c {
                        if keep(&c, tau) {
                            out.push((gj, c));
                            k += 1;
                        } else {
                            dr += 1;
                        }
                    }
                }
                (out, k, dr)
            })
            .collect();
        let mut m = Self::zeros(self.rows.clone(), other.cols.clone());
        for (g, (b, k, dr)) in rows.into_iter().enumerate() {
            m.blk[g] = b;
            cnt.kept += k;
            cnt.dropped += dr;
        }
        cnt.products += 1;
        m
    }

    /// `diag(selfᵀ · other)`: per column, the dot of the two columns.
    pub fn col_dots(&self, other: &Bsr) -> Vec<f64> {
        assert!(same(&self.rows, &other.rows) && same(&self.cols, &other.cols));
        let mut out = vec![0.0; self.cols.n()];
        for gi in 0..self.blk.len() {
            let (a, b) = (&self.blk[gi], &other.blk[gi]);
            let mut j = 0;
            for (ga, ba) in a {
                while j < b.len() && b[j].0 < *ga {
                    j += 1;
                }
                if j < b.len() && b[j].0 == *ga {
                    let bb = &b[j].1;
                    let cj = &self.cols.groups[*ga];
                    for (c, &ic) in cj.iter().enumerate() {
                        let mut s = 0.0;
                        for r in 0..ba.nrows() {
                            s += ba[(r, c)] * bb[(r, c)];
                        }
                        out[ic] += s;
                    }
                }
            }
        }
        out
    }

    /// Scale column `c` by `f[c]`.
    pub fn scale_cols(&mut self, f: &[f64]) {
        let cols = self.cols.clone();
        for row in &mut self.blk {
            for (gj, b) in row.iter_mut() {
                for (c, &ic) in cols.groups[*gj].iter().enumerate() {
                    b.column_mut(c).mapv_inplace(|x| x * f[ic]);
                }
            }
        }
    }

    /// Scale row `r` by `f[r]`.
    pub fn scale_rows(&mut self, f: &[f64]) {
        let rows = self.rows.clone();
        for (gi, row) in self.blk.iter_mut().enumerate() {
            for (_, b) in row.iter_mut() {
                for (r, &ir) in rows.groups[gi].iter().enumerate() {
                    b.row_mut(r).mapv_inplace(|x| x * f[ir]);
                }
            }
        }
    }

    pub fn scale(&mut self, f: f64) {
        for row in &mut self.blk {
            for (_, b) in row.iter_mut() {
                b.mapv_inplace(|x| x * f);
            }
        }
    }

    /// The columns `idx` (global, in this order) as a matrix over the column
    /// partition `new_cols` (`new_cols.n() == idx.len()`).
    pub fn select_cols(&self, idx: &[usize], new_cols: Arc<Partition>) -> Self {
        assert_eq!(idx.len(), new_cols.n());
        // per new group: (old group, old position) of each of its columns
        let src: Vec<Vec<(usize, usize)>> = new_cols
            .groups
            .iter()
            .map(|g| g.iter().map(|&k| self.cols.place[idx[k]]).collect())
            .collect();
        let mut out = Self::zeros(self.rows.clone(), new_cols.clone());
        for (gi, row) in self.blk.iter().enumerate() {
            let nr = self.rows.groups[gi].len();
            for (h, s) in src.iter().enumerate() {
                if s.is_empty() {
                    continue;
                }
                let mut b: Option<Array2<f64>> = None;
                for (c, &(og, op)) in s.iter().enumerate() {
                    if let Ok(p) = row.binary_search_by_key(&og, |x| x.0) {
                        let bb = b.get_or_insert_with(|| Array2::zeros((nr, s.len())));
                        bb.column_mut(c).assign(&row[p].1.column(op));
                    }
                }
                if let Some(b) = b {
                    out.blk[gi].push((h, b));
                }
            }
        }
        out
    }

    /// The sub-matrix on rows `ridx` (new partition `new_rows`) and columns
    /// `cidx` (new partition `new_cols`).
    pub fn select(
        &self,
        ridx: &[usize],
        new_rows: Arc<Partition>,
        cidx: &[usize],
        new_cols: Arc<Partition>,
    ) -> Self {
        self.transpose()
            .select_cols(ridx, new_rows)
            .transpose()
            .select_cols(cidx, new_cols)
    }

    /// Row `i` as (column, value) pairs, ascending column-group order.
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let (gi, pi) = self.rows.place[i];
        self.blk[gi].iter().flat_map(move |(gj, b)| {
            self.cols.groups[*gj]
                .iter()
                .enumerate()
                .map(move |(c, &ic)| (ic, b[(pi, c)]))
        })
    }

    /// The diagonal of a square matrix (rows and columns share a partition).
    pub fn diag(&self) -> Vec<f64> {
        assert!(same(&self.rows, &self.cols));
        let mut d = vec![0.0; self.rows.n()];
        for (gi, row) in self.blk.iter().enumerate() {
            if let Ok(p) = row.binary_search_by_key(&gi, |x| x.0) {
                for (r, &ir) in self.rows.groups[gi].iter().enumerate() {
                    d[ir] = row[p].1[(r, r)];
                }
            }
        }
        d
    }

    pub fn trace(&self) -> f64 {
        self.diag().iter().sum()
    }

    /// Upper bound on the spectrum of a symmetric matrix (Gershgorin).
    pub fn gershgorin_upper(&self) -> f64 {
        let mut rs = vec![0.0f64; self.rows.n()];
        let mut dg = vec![0.0f64; self.rows.n()];
        for (gi, row) in self.blk.iter().enumerate() {
            let ri = &self.rows.groups[gi];
            for (gj, b) in row {
                let cj = &self.cols.groups[*gj];
                for (r, &ir) in ri.iter().enumerate() {
                    for (c, &ic) in cj.iter().enumerate() {
                        if ir == ic {
                            dg[ir] += b[(r, c)];
                        } else {
                            rs[ir] += b[(r, c)].abs();
                        }
                    }
                }
            }
        }
        dg.iter()
            .zip(&rs)
            .map(|(d, r)| d + r)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// `max |self − I|` over every element (a missing diagonal block counts
    /// its full deviation).
    pub fn max_dev_from_identity(&self) -> f64 {
        assert!(same(&self.rows, &self.cols));
        let mut dev: f64 = 0.0;
        for (gi, row) in self.blk.iter().enumerate() {
            let mut saw_diag = false;
            for (gj, b) in row {
                let is_diag = *gj == gi;
                saw_diag |= is_diag;
                for ((r, c), x) in b.indexed_iter() {
                    let t = if is_diag && r == c { 1.0 } else { 0.0 };
                    dev = dev.max((x - t).abs());
                }
            }
            if !saw_diag && !self.rows.groups[gi].is_empty() {
                dev = dev.max(1.0);
            }
        }
        dev
    }
}

/// Residual at which [`newton_schulz_inv_sqrt`] counts as converged; one
/// more (polishing) iteration follows, which the quadratic convergence takes
/// to the rounding level.
pub(crate) const NS_SWITCH_TOL: f64 = 1e-8;

/// Coupled Newton-Schulz inverse square root of a symmetric positive
/// definite block-sparse `o`: `Y₀ = o/c`, `Z₀ = I` with `c` the Gershgorin
/// bound (so the spectrum of `Y₀` lies in (0, 1]); `T = 3I − ZY`,
/// `Y ← ½YT`, `Z ← ½TZ`. At the first iterate with `max|ZY − I| <
/// NS_SWITCH_TOL` one more iteration is taken and its `Z/√c` returned. Each
/// product is truncated at `tau`; with `tau > 0` a residual that stops
/// falling below 1e-4 (the truncation floor) also ends the iteration, after
/// the same polishing step.
///
/// `mutate_stop_early` (mutation tests only) returns instead the LAST
/// UNCONVERGED iterate (residual >= `NS_SWITCH_TOL`), i.e. the iteration
/// stopped one step before convergence.
///
/// Errors when the iteration has not converged in 100 steps (a near-singular
/// set; [`crate::lmp2_amplitude::lowdin`] rejects a minimum eigenvalue below
/// 1e-10, which needs ~40 iterations here). Returns `o^{-1/2}` and the
/// iteration count (products of the polishing step included).
pub(crate) fn newton_schulz_inv_sqrt(
    o: &Bsr,
    tau: f64,
    mutate_stop_early: bool,
    cnt: &mut BlockCount,
    who: &str,
) -> Result<(Bsr, usize), FerricError> {
    const MAX_ITER: usize = 100;
    let p = o.rows.clone();
    let c = o.gershgorin_upper();
    if !(c.is_finite() && c > 0.0) {
        return Err(FerricError::General(format!(
            "local VV-HV {who}: Newton-Schulz on a matrix with Gershgorin bound {c:.3e}"
        )));
    }
    let eye = Bsr::identity(p.clone());
    let mut y = o.clone();
    y.scale(1.0 / c);
    let mut z = eye.clone();
    let mut prev_z: Option<Bsr> = None;
    let mut best = f64::INFINITY;
    let mut polish = false;
    for it in 0..MAX_ITER {
        let zy = z.mul(&y, tau, cnt);
        if polish {
            let mut out = z;
            out.scale(1.0 / c.sqrt());
            return Ok((out, it));
        }
        let r = zy.max_dev_from_identity();
        let stagnated = tau > 0.0 && r < 1e-4 && r >= 0.5 * best;
        if r < NS_SWITCH_TOL || stagnated {
            if mutate_stop_early {
                let mut out = prev_z.unwrap_or(z);
                out.scale(1.0 / c.sqrt());
                return Ok((out, it));
            }
            polish = true;
        }
        best = best.min(r);
        let t = eye.axpby(3.0, &zy, -1.0, 0.0, cnt);
        let mut y2 = y.mul(&t, tau, cnt);
        y2.scale(0.5);
        let mut z2 = t.mul(&z, tau, cnt);
        z2.scale(0.5);
        prev_z = Some(std::mem::replace(&mut z, z2));
        y = y2;
    }
    Err(FerricError::General(format!(
        "local VV-HV {who}: Newton-Schulz inverse square root did not converge in {MAX_ITER} \
         iterations (near-singular set)"
    )))
}

/// Tie rule of [`heap_pivoted_cholesky`]: the production rule is the dense
/// build's ([`crate::lmp2_amplitude::PIVOT_TIE_RTOL`], lowest index); the
/// reversed rule exists for the mutation test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TieRule {
    Lowest,
    #[cfg(test)]
    Highest,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapItem(f64, usize);

impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0).then_with(|| o.1.cmp(&self.1)) // lower index first on equal value
    }
}

/// Greedy pivoted Cholesky of a PSD block-sparse `m` to `rank` pivots, with a
/// max-heap (lazy deletion) for the argmax and sparse rows of L (entries
/// below `tau_l` in magnitude dropped). Reproduces
/// [`crate::lmp2_amplitude::pivoted_cholesky_order`]'s pivot sequence at
/// `tau_l = 0`, including its tie rule (every residual diagonal within
/// `PIVOT_TIE_RTOL` relative of the maximum is a tie, and the lowest index
/// wins). Returns the pivots and the heap pops.
pub(crate) fn heap_pivoted_cholesky(
    m: &Bsr,
    rank: usize,
    tau_l: f64,
    tie: TieRule,
) -> Result<(Vec<usize>, u64), FerricError> {
    use std::collections::BinaryHeap;
    let rtol = crate::lmp2_amplitude::PIVOT_TIE_RTOL;
    let n = m.rows.n();
    let mut d = m.diag();
    let mut elim = vec![false; n];
    let mut heap: BinaryHeap<HeapItem> =
        d.iter().enumerate().map(|(i, &x)| HeapItem(x, i)).collect();
    let valid = |it: &HeapItem, d: &[f64], elim: &[bool]| {
        !elim[it.1] && it.0.to_bits() == d[it.1].to_bits()
    };
    let mut lrow: Vec<Vec<(usize, f64)>> = Vec::with_capacity(rank);
    let mut lcol: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut scratch = vec![0.0f64; n];
    let mut touched_flag = vec![false; n];
    let mut touched: Vec<usize> = Vec::new();
    let mut piv = Vec::with_capacity(rank);
    let mut pops = 0u64;
    for k in 0..rank {
        // the maximum, then every valid entry within the tie band
        let top = loop {
            let it = heap.pop().ok_or_else(|| {
                FerricError::General(format!(
                    "local VV-HV: pivoted Cholesky ran out of candidates at k={k}"
                ))
            })?;
            pops += 1;
            if it.0.is_nan() {
                return Err(FerricError::General(
                    "local VV-HV: NaN in pivoted Cholesky diagonal".to_string(),
                ));
            }
            if valid(&it, &d, &elim) {
                break it;
            }
        };
        let floor = top.0 - rtol * top.0.abs();
        let mut band = vec![top];
        while let Some(it) = heap.peek().copied() {
            if !valid(&it, &d, &elim) {
                heap.pop();
                pops += 1;
                continue;
            }
            if it.0 >= floor {
                heap.pop();
                pops += 1;
                band.push(it);
            } else {
                break;
            }
        }
        let chosen = match tie {
            TieRule::Lowest => band.iter().map(|x| x.1).min(),
            #[cfg(test)]
            TieRule::Highest => band.iter().map(|x| x.1).max(),
        }
        .expect("non-empty band");
        for it in band {
            if it.1 != chosen {
                heap.push(it);
            }
        }
        let (j, dj) = (chosen, d[chosen]);
        if dj <= 0.0 {
            return Err(FerricError::General(format!(
                "local VV-HV: pivoted Cholesky broke down at k={k}"
            )));
        }
        piv.push(j);
        // row k of L: (M[j,:] − Σ_{k'<k} L[k',j] L[k',:]) / √d_j
        for (c, v) in m.row(j) {
            if !touched_flag[c] {
                touched_flag[c] = true;
                touched.push(c);
            }
            scratch[c] += v;
        }
        for &(kp, ljk) in &lcol[j] {
            for &(c, v) in &lrow[kp] {
                if !touched_flag[c] {
                    touched_flag[c] = true;
                    touched.push(c);
                }
                scratch[c] -= ljk * v;
            }
        }
        let scale = 1.0 / dj.sqrt();
        touched.sort_unstable();
        let mut row = Vec::with_capacity(touched.len());
        for &c in &touched {
            let v = scratch[c] * scale;
            scratch[c] = 0.0;
            touched_flag[c] = false;
            if tau_l > 0.0 && v.abs() < tau_l {
                continue;
            }
            row.push((c, v));
        }
        touched.clear();
        for &(c, v) in &row {
            d[c] -= v * v;
            lcol[c].push((k, v));
            if !elim[c] && c != j {
                heap.push(HeapItem(d[c], c));
            }
        }
        lrow.push(row);
        elim[j] = true;
    }
    Ok((piv, pops))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(n: usize, gsize: usize) -> Arc<Partition> {
        let labels: Vec<usize> = (0..n).map(|i| i / gsize).collect();
        Partition::from_labels(&labels, n.div_ceil(gsize))
    }

    fn rand_mat(r: usize, c: usize, seed: u64) -> Array2<f64> {
        let mut s = seed;
        Array2::from_shape_simple_fn((r, c), || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
    }

    fn max_abs(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
    }

    /// At tau = 0 every operation equals its dense counterpart to rounding,
    /// and a nonzero tau is SEEN (a dropped block changes the product).
    #[test]
    fn products_match_dense_at_tau_zero_and_drops_are_seen() {
        let (pa, pb, pc) = (part(13, 4), part(9, 2), part(7, 3));
        let a = rand_mat(13, 9, 1);
        let b = rand_mat(9, 7, 2);
        let mut cnt = BlockCount::default();
        let ab = Bsr::from_dense(&a, pa.clone(), pb.clone(), 0.0, &mut cnt);
        let bb = Bsr::from_dense(&b, pb.clone(), pc.clone(), 0.0, &mut cnt);
        assert_eq!(cnt.dropped, 0);
        let c = ab.mul(&bb, 0.0, &mut cnt).to_dense();
        let dense = a.dot(&b);
        assert!(max_abs(&(&c - &dense)) <= 1e-14 * max_abs(&dense));
        assert!(max_abs(&(ab.transpose().to_dense() - a.t())) == 0.0);
        let s = ab.axpby(2.0, &ab, -0.5, 0.0, &mut cnt).to_dense();
        assert!(max_abs(&(&s - &(&a * 1.5))) <= 1e-15);
        let cd = ab.col_dots(&ab);
        for (k, v) in cd.iter().enumerate() {
            assert!((v - a.column(k).dot(&a.column(k))).abs() <= 1e-14);
        }
        let idx = [8usize, 0, 3, 3, 5];
        let np = part(5, 2);
        let sel = ab.select_cols(&idx, np).to_dense();
        for (k, &i) in idx.iter().enumerate() {
            assert_eq!(sel.column(k), a.column(i));
        }
        // a drop is seen
        let mut cnt2 = BlockCount::default();
        let c2 = ab.mul(&bb, 0.9, &mut cnt2).to_dense();
        assert!(cnt2.dropped > 0 && max_abs(&(&c2 - &dense)) > 1e-3);
    }

    fn spd(n: usize, seed: u64) -> Array2<f64> {
        let a = rand_mat(n, n, seed);
        a.t().dot(&a) + Array2::<f64>::eye(n) * 0.3
    }

    /// Newton-Schulz equals the eigh Löwdin factor; one iteration short
    /// does not.
    #[test]
    fn newton_schulz_matches_eigh_and_one_short_fails() {
        use ndarray_linalg::Eigh;
        let n = 11;
        let o = spd(n, 7);
        let (w, v) = o.eigh(ndarray_linalg::UPLO::Lower).unwrap();
        let mut vs = v.clone();
        for (k, wk) in w.iter().enumerate() {
            vs.column_mut(k).mapv_inplace(|x| x / wk.sqrt());
        }
        let exact = vs.dot(&v.t());
        let p = part(n, 3);
        let mut cnt = BlockCount::default();
        let ob = Bsr::from_dense(&o, p.clone(), p, 0.0, &mut cnt);
        let (z, it) = newton_schulz_inv_sqrt(&ob, 0.0, false, &mut cnt, "t").unwrap();
        assert!(it > 3);
        let dev = max_abs(&(z.to_dense() - &exact));
        assert!(dev <= 1e-12, "NS vs eigh {dev:.2e}");
        let (z1, _) = newton_schulz_inv_sqrt(&ob, 0.0, true, &mut cnt, "t").unwrap();
        let dev1 = max_abs(&(z1.to_dense() - &exact));
        assert!(dev1 > 1e-9, "stopped early must be seen ({dev1:.2e})");
    }

    /// The heap pivoted Cholesky takes the dense routine's pivots, ties
    /// included; the reversed tie rule does not.
    #[test]
    fn heap_pivots_match_dense_including_ties() {
        let n = 12;
        let mut o = spd(n, 3);
        // exact ties: duplicate the structure of indices 2 and 7
        for k in 0..n {
            let v = o[(2, k)];
            o[(7, k)] = if k == 7 {
                o[(2, 2)]
            } else if k == 2 {
                o[(2, 7)]
            } else {
                v
            };
        }
        for k in 0..n {
            o[(k, 7)] = o[(7, k)];
        }
        o[(7, 7)] = o[(2, 2)];
        let o = (&o + &o.t()) * 0.5 + Array2::<f64>::eye(n) * 5.0;
        let dense = crate::lmp2_amplitude::pivoted_cholesky_order(&o, 8).unwrap();
        let p = part(n, 5);
        let mut cnt = BlockCount::default();
        let ob = Bsr::from_dense(&o, p.clone(), p, 0.0, &mut cnt);
        let (heap, pops) = heap_pivoted_cholesky(&ob, 8, 0.0, TieRule::Lowest).unwrap();
        assert_eq!(heap, dense);
        assert!(pops >= 8);
        // the tie is real (2 and 7 tie at the first pivot) and the rule is seen
        let d0: Vec<f64> = (0..n).map(|i| o[(i, i)]).collect();
        assert_eq!(d0[2], d0[7]);
        let (rev, _) = heap_pivoted_cholesky(&ob, 8, 0.0, TieRule::Highest).unwrap();
        assert_ne!(rev, dense);
    }
}
