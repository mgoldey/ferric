//! Maximum-Overlap Method (MOM) for ROHF/ROKS orbital ordering.
//!
//! After diagonalizing the Fock matrix to get a new C, MOM picks the
//! `n_closed` and `n_open` new MOs that have maximum *AO-overlap* with the
//! previous-iteration occupied set, rather than the n_occ MOs with lowest
//! orbital energy. This pins the SOMO identity through SCF iterations and
//! breaks the DIIS oscillation we see on doublet OH/LDA at 0.97 Å.
//!
//! Reference: Gilbert, Besley, Gill, JPC A 112, 13164, 2008.

use ndarray::Array2;

/// Reorder columns of `c_new` so the leading `n_closed` columns have
/// maximum overlap with the previous closed-MO set, and the next
/// `n_open` columns have maximum overlap with the previous open-MO set.
///
/// `c_new`: shape (n, n), columns are the new MO coefficients in AO basis.
/// `s`:     overlap matrix in AO basis.
/// `c_prev_closed`: shape (n, n_closed) — previous accepted closed-MO block.
/// `c_prev_open`:   shape (n, n_open)   — previous accepted open-MO block.
///
/// Returns a new (n, n) matrix with the same columns as `c_new` but
/// reordered. Virtuals at the end keep their original (ε-sorted) order.
pub fn mom_reorder(
    c_new: &Array2<f64>,
    s: &Array2<f64>,
    c_prev_closed: &Array2<f64>,
    c_prev_open: &Array2<f64>,
    n_closed: usize,
    n_open: usize,
) -> Array2<f64> {
    let n = c_new.nrows();
    debug_assert_eq!(c_new.ncols(), n);
    debug_assert_eq!(s.dim(), (n, n));
    debug_assert_eq!(c_prev_closed.nrows(), n);
    debug_assert_eq!(c_prev_closed.ncols(), n_closed);
    debug_assert_eq!(c_prev_open.nrows(), n);
    debug_assert_eq!(c_prev_open.ncols(), n_open);

    // 1. Score every new MO against the previous closed set.
    //    score_closed[p] = sum_i (C_prev_closed[:,i]^T S C_new[:,p])^2
    let scored_closed = if n_closed > 0 {
        let o_closed = c_prev_closed.t().dot(s).dot(c_new); // (n_closed, n)
        (0..n)
            .map(|p| {
                let col = o_closed.column(p);
                col.iter().map(|x| x * x).sum::<f64>()
            })
            .collect::<Vec<f64>>()
    } else {
        vec![0.0; n]
    };

    // Index list 0..n, paired with score. Sort by score descending; ties
    // broken by original index ascending (stable, so virtuals retain ε-order).
    let mut idx_by_closed: Vec<usize> = (0..n).collect();
    idx_by_closed.sort_by(|&a, &b| {
        scored_closed[b]
            .partial_cmp(&scored_closed[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let closed_chosen: Vec<usize> = idx_by_closed[..n_closed].to_vec();
    let remaining_after_closed: Vec<usize> = idx_by_closed[n_closed..].to_vec();

    // 2. From `remaining_after_closed`, score against previous open set.
    let scored_open = if n_open > 0 {
        let mut c_remaining = Array2::<f64>::zeros((n, remaining_after_closed.len()));
        for (k, &p) in remaining_after_closed.iter().enumerate() {
            c_remaining.column_mut(k).assign(&c_new.column(p));
        }
        let o_open = c_prev_open.t().dot(s).dot(&c_remaining); // (n_open, remaining)
        (0..remaining_after_closed.len())
            .map(|k| {
                let col = o_open.column(k);
                col.iter().map(|x| x * x).sum::<f64>()
            })
            .collect::<Vec<f64>>()
    } else {
        vec![0.0; remaining_after_closed.len()]
    };

    let mut idx_by_open: Vec<usize> = (0..remaining_after_closed.len()).collect();
    idx_by_open.sort_by(|&a, &b| {
        scored_open[b]
            .partial_cmp(&scored_open[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let open_chosen_local: Vec<usize> = idx_by_open[..n_open].to_vec();
    let virt_chosen_local: Vec<usize> = idx_by_open[n_open..].to_vec();
    // Map local indices back to indices in c_new.
    let open_chosen: Vec<usize> = open_chosen_local
        .iter()
        .map(|&k| remaining_after_closed[k])
        .collect();
    let mut virt_chosen: Vec<usize> = virt_chosen_local
        .iter()
        .map(|&k| remaining_after_closed[k])
        .collect();
    // Sort virtuals by original index (preserves ε-order for diagnostic clarity).
    virt_chosen.sort();

    // 3. Assemble output.
    let mut out = Array2::<f64>::zeros((n, n));
    for (target, &source) in closed_chosen.iter().enumerate() {
        out.column_mut(target).assign(&c_new.column(source));
    }
    for (target, &source) in open_chosen.iter().enumerate() {
        out.column_mut(n_closed + target).assign(&c_new.column(source));
    }
    for (target, &source) in virt_chosen.iter().enumerate() {
        out.column_mut(n_closed + n_open + target)
            .assign(&c_new.column(source));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// MOM is the identity when `c_new == c_prev` (block-diagonal overlap).
    #[test]
    fn mom_reorder_identity_when_c_new_equals_c_prev() {
        let s = Array2::<f64>::eye(3);
        let c_new = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],];
        let c_prev_closed = c_new.slice(ndarray::s![.., 0..1]).to_owned();
        let c_prev_open = c_new.slice(ndarray::s![.., 1..2]).to_owned();
        let out = mom_reorder(&c_new, &s, &c_prev_closed, &c_prev_open, 1, 1);
        assert_eq!(out, c_new);
    }

    /// If c_new's columns are permuted vs c_prev, MOM should put them back
    /// in the previous order.
    #[test]
    fn mom_reorder_undoes_a_permutation() {
        let s = Array2::<f64>::eye(3);
        // c_prev: identity. c_new: same columns, but swapped 0 <-> 1.
        let c_new = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],];
        let c_prev_closed = array![[1.0], [0.0], [0.0]];
        let c_prev_open = array![[0.0], [1.0], [0.0]];
        let out = mom_reorder(&c_new, &s, &c_prev_closed, &c_prev_open, 1, 1);
        let expected = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],];
        assert_eq!(out, expected);
    }
}
