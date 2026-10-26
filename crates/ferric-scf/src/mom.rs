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
/// `eps_new`: the eigenvalues of `c_new`, in the same (diagonalisation) order.
///
/// Returns `(C, ε)`: the columns of `c_new` reordered, and `eps_new` permuted
/// the SAME way, so `ε[p]` stays the energy of column `p`. Virtuals at the end
/// keep their original (ε-sorted) order. Every consumer that reads energies by
/// column index (UHF `density_fractional`, RHF Fermi smearing) needs the
/// permuted ε: before 2026-10-24 only C was returned, so after a MOM swap
/// those paths grouped/occupied the columns by the energies of other orbitals.
pub fn mom_reorder(
    c_new: &Array2<f64>,
    eps_new: &[f64],
    s: &Array2<f64>,
    c_prev_closed: &Array2<f64>,
    c_prev_open: &Array2<f64>,
    n_closed: usize,
    n_open: usize,
) -> (Array2<f64>, Vec<f64>) {
    debug_assert_eq!(eps_new.len(), c_new.ncols());
    let order = mom_order(c_new, s, c_prev_closed, c_prev_open, n_closed, n_open);
    let mut out = Array2::<f64>::zeros(c_new.raw_dim());
    for (target, &source) in order.iter().enumerate() {
        out.column_mut(target).assign(&c_new.column(source));
    }
    let eps = order.iter().map(|&p| eps_new[p]).collect();
    (out, eps)
}

/// The MOM column order: `order[target] = source` column of `c_new`
/// (closed block, then open block, then the virtuals in ascending index).
fn mom_order(
    c_new: &Array2<f64>,
    s: &Array2<f64>,
    c_prev_closed: &Array2<f64>,
    c_prev_open: &Array2<f64>,
    n_closed: usize,
    n_open: usize,
) -> Vec<usize> {
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

    // 3. Assemble the order: closed, open, virtual.
    closed_chosen
        .into_iter()
        .chain(open_chosen)
        .chain(virt_chosen)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Energies of the three test columns, in `c_new` order.
    const EPS: [f64; 3] = [-1.0, -0.5, 0.2];

    /// One closed + one open MOM reorder of a 3-orbital `c_new` against the
    /// identity (S = I) references `e0` (closed) and `e1` (open).
    fn reorder_against_identity(c_new: &Array2<f64>) -> (Array2<f64>, Vec<f64>) {
        let s = Array2::<f64>::eye(3);
        let c_prev_closed = array![[1.0], [0.0], [0.0]];
        let c_prev_open = array![[0.0], [1.0], [0.0]];
        mom_reorder(c_new, &EPS, &s, &c_prev_closed, &c_prev_open, 1, 1)
    }

    /// MOM is the identity when `c_new == c_prev` (block-diagonal overlap).
    #[test]
    fn mom_reorder_identity_when_c_new_equals_c_prev() {
        let c_new = Array2::<f64>::eye(3);
        let (out, eps_out) = reorder_against_identity(&c_new);
        assert_eq!(out, c_new);
        assert_eq!(eps_out, EPS);
    }

    /// If c_new's columns are permuted vs c_prev, MOM should put them back
    /// in the previous order.
    #[test]
    fn mom_reorder_undoes_a_permutation() {
        // c_prev: identity. c_new: same columns, but swapped 0 <-> 1.
        let c_new = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],];
        let (out, _) = reorder_against_identity(&c_new);
        assert_eq!(out, Array2::<f64>::eye(3));
    }

    /// The eigenvalues travel with their columns: after the 0 <-> 1 swap,
    /// output column 0 is column 1 of c_new, so its energy is EPS[1].
    #[test]
    fn mom_reorder_permutes_eigenvalues_with_columns() {
        let c_new = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],];
        let (_, eps_out) = reorder_against_identity(&c_new);
        assert_eq!(eps_out, vec![EPS[1], EPS[0], EPS[2]]);
    }
}
