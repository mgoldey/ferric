//! CAS-CI Hamiltonian: Slater–Condon matrix elements and the sigma-vector
//! (`sigma = H c`) build over the α/β-string determinant list.
//!
//! ## Scaling note (read before extending to Phase B/C)
//!
//! This spike computes `sigma = H c` via a **naive O(N_det^2)** double loop over
//! determinant pairs, applying the Slater–Condon rules to each pair. That is
//! correct and simple but does **not** scale: the number of determinants grows
//! combinatorially with the active space, and a real FCI code uses the
//! string-based *direct* sigma algorithm (Knowles–Handy / Olsen), which
//! organizes the work as `sigma1 (β-β)`, `sigma2 (α-α)`, and `sigma3 (α-β)`
//! contributions and never touches the full pair list. Phase B/C should replace
//! this `dense_hamiltonian` / `sigma` pair-loop with that string-based build.
//!
//! For the STO-3G spike systems (H2: 4 determinants; H2O full FCI:
//! C(7,5)^2 = 441 determinants; H2O CAS(6,8): C(6,4)^2 = 225 determinants) the
//! O(N_det^2) pair loop is trivially fast and unambiguously correct, which is
//! exactly what Phase A needs to prove.

use crate::integrals::ActiveSpaceIntegrals;
use crate::strings::phase_below;

/// A determinant = (α-string index, β-string index) into the string lists.
///
/// The determinant list is the outer product `alpha_strings x beta_strings`,
/// laid out row-major: determinant `d = ia * n_beta + ib`, where `ia` indexes
/// the α-string and `ib` the β-string.
pub struct DeterminantSpace {
    pub alpha_strings: Vec<u64>,
    pub beta_strings: Vec<u64>,
}

impl DeterminantSpace {
    pub fn n_det(&self) -> usize {
        self.alpha_strings.len() * self.beta_strings.len()
    }
}

/// One-electron MO integral `h_pq` from the active-space integrals.
#[inline]
fn h1(ints: &ActiveSpaceIntegrals, p: usize, q: usize) -> f64 {
    ints.h[(p, q)]
}

/// Chemist-notation two-electron integral `(pq|rs)`.
#[inline]
fn h2(ints: &ActiveSpaceIntegrals, p: usize, q: usize, r: usize, s: usize) -> f64 {
    ints.g(p, q, r, s)
}

/// Diagonal Slater–Condon element `<D|H|D>` (electronic part; excludes the
/// additive constant `e_core`).
///
/// For a determinant with α-occupied set A and β-occupied set B:
///   <D|H|D> = sum_{p in A∪B} h_pp
///           + sum_{p<q in A} [(pp|qq) - (pq|qp)]     (α-α)
///           + sum_{p<q in B} [(pp|qq) - (pq|qp)]     (β-β)
///           + sum_{p in A, q in B} (pp|qq)           (α-β, no exchange)
pub fn diagonal_element(ints: &ActiveSpaceIntegrals, a_occ: &[usize], b_occ: &[usize]) -> f64 {
    let mut e = 0.0;
    for &p in a_occ {
        e += h1(ints, p, p);
    }
    for &p in b_occ {
        e += h1(ints, p, p);
    }
    // same-spin Coulomb - exchange (α-α)
    for i in 0..a_occ.len() {
        for j in (i + 1)..a_occ.len() {
            let p = a_occ[i];
            let q = a_occ[j];
            e += h2(ints, p, p, q, q) - h2(ints, p, q, q, p);
        }
    }
    // same-spin (β-β)
    for i in 0..b_occ.len() {
        for j in (i + 1)..b_occ.len() {
            let p = b_occ[i];
            let q = b_occ[j];
            e += h2(ints, p, p, q, q) - h2(ints, p, q, q, p);
        }
    }
    // opposite-spin Coulomb only (α-β)
    for &p in a_occ {
        for &q in b_occ {
            e += h2(ints, p, p, q, q);
        }
    }
    e
}

/// The full diagonal of the CI Hamiltonian *including* `e_core`, one entry per
/// determinant. Used by the Davidson preconditioner.
pub fn hamiltonian_diagonal(ints: &ActiveSpaceIntegrals, space: &DeterminantSpace) -> Vec<f64> {
    let na = space.alpha_strings.len();
    let nb = space.beta_strings.len();
    let a_occ: Vec<Vec<usize>> = space
        .alpha_strings
        .iter()
        .map(|&s| crate::strings::occupied_orbitals(s))
        .collect();
    let b_occ: Vec<Vec<usize>> = space
        .beta_strings
        .iter()
        .map(|&s| crate::strings::occupied_orbitals(s))
        .collect();
    let mut diag = vec![0.0f64; na * nb];
    for ia in 0..na {
        for ib in 0..nb {
            diag[ia * nb + ib] =
                ints.e_core + diagonal_element(ints, &a_occ[ia], &b_occ[ib]);
        }
    }
    diag
}

/// Off-diagonal / general Slater–Condon element `<D'|H|D>` between two
/// determinants given as α/β bitmasks. Returns the *electronic* element (no
/// `e_core`; `e_core` only lands on the diagonal, added separately).
///
/// Uses the standard bit-difference count to classify D vs D' as identical,
/// single, or double excitation, then applies the corresponding Slater–Condon
/// formula. Phase factors come from the Jordan–Wigner ordering
/// ([`phase_below`]).
pub fn general_element(
    ints: &ActiveSpaceIntegrals,
    a_bra: u64,
    b_bra: u64,
    a_ket: u64,
    b_ket: u64,
) -> f64 {
    let da = (a_bra ^ a_ket).count_ones();
    let db = (b_bra ^ b_ket).count_ones();

    // Total spin-orbital excitation degree = (α diffs + β diffs)/2 each side;
    // number of differing spin-orbitals is da + db (each excitation flips 2
    // bits: one removed, one added, per spin channel involved).
    let ndiff = da + db;

    if ndiff == 0 {
        // Diagonal: handled by diagonal_element elsewhere; return it here too
        // so general_element is self-contained/testable.
        let a_occ = crate::strings::occupied_orbitals(a_ket);
        let b_occ = crate::strings::occupied_orbitals(b_ket);
        return diagonal_element(ints, &a_occ, &b_occ);
    }

    // Single excitation: differ by exactly one spatial orbital in exactly one
    // spin channel (2 bits total, both in α or both in β).
    if ndiff == 2 {
        if da == 2 && db == 0 {
            return single_excitation_same_spin(ints, a_bra, a_ket, b_ket);
        } else if da == 0 && db == 2 {
            return single_excitation_same_spin(ints, b_bra, b_ket, a_ket);
        } else {
            return 0.0;
        }
    }

    // Double excitation: 4 differing spin-orbitals.
    if ndiff == 4 {
        if da == 4 && db == 0 {
            return double_same_spin(ints, a_bra, a_ket);
        } else if da == 0 && db == 4 {
            return double_same_spin(ints, b_bra, b_ket);
        } else if da == 2 && db == 2 {
            return double_opposite_spin(ints, a_bra, a_ket, b_bra, b_ket);
        } else {
            return 0.0;
        }
    }

    // ndiff > 4: Hamiltonian has at most two-body terms; element is zero.
    0.0
}

/// Single excitation within one spin channel (call with the channel that
/// differs). `excited`/`ref_str` are the bra/ket strings *in the differing
/// channel*; `other_occ_str` is the (unchanged) string in the *other* channel.
///
/// The excitation moves an electron from orbital `i` (occupied in ket, empty in
/// bra) to orbital `a` (empty in ket, occupied in bra).
///
///   <D'|H|D> = phase * [ h_ai
///              + sum_{p in same-spin occ (both)} ( (ai|pp) - (ap|pi) )
///              + sum_{p in other-spin occ}       (ai|pp) ]
fn single_excitation_same_spin(
    ints: &ActiveSpaceIntegrals,
    bra: u64,
    ket: u64,
    other_occ_str: u64,
) -> f64 {
    let diff = bra ^ ket;
    // orbital removed from ket (occupied in ket, not in bra)
    let i_bit = diff & ket;
    // orbital added (occupied in bra, not in ket)
    let a_bit = diff & bra;
    let i = i_bit.trailing_zeros() as usize;
    let a = a_bit.trailing_zeros() as usize;

    // Phase: annihilate i in ket, then create a. Sign is the product of the
    // JW parities in the *ket* (for i) and in the ket-with-i-removed (for a).
    let phase_i = phase_below(ket, i);
    let ket_minus = ket & !(1u64 << i);
    let phase_a = phase_below(ket_minus, a);
    let phase = phase_i * phase_a;

    let mut val = h1(ints, a, i);

    // same-spin occupied orbitals present in BOTH bra and ket (the spectators).
    let same_spectators = ket & !(1u64 << i); // ket minus i == bra minus a; spectators are these minus a
    let same_spectators = same_spectators & !(1u64 << a);
    for p in crate::strings::occupied_orbitals(same_spectators) {
        val += h2(ints, a, i, p, p) - h2(ints, a, p, p, i);
    }
    // opposite-spin occupied: Coulomb only.
    for p in crate::strings::occupied_orbitals(other_occ_str) {
        val += h2(ints, a, i, p, p);
    }

    phase * val
}

/// Same-spin double excitation (both electrons in one channel). Two orbitals
/// `i<j` removed, two orbitals `a<b` added (within that channel).
///
///   <D'|H|D> = phase * [ (ai|bj) - (aj|bi) ]
fn double_same_spin(ints: &ActiveSpaceIntegrals, bra: u64, ket: u64) -> f64 {
    let diff = bra ^ ket;
    let removed = diff & ket; // i, j (occupied in ket)
    let added = diff & bra; // a, b (occupied in bra)
    let rem: Vec<usize> = crate::strings::occupied_orbitals(removed);
    let add: Vec<usize> = crate::strings::occupied_orbitals(added);
    // rem = [i, j] ascending, add = [a, b] ascending.
    let (i, j) = (rem[0], rem[1]);
    let (a, b) = (add[0], add[1]);

    // Phase: annihilate i then j from ket, create b then a. Use sequential JW
    // parities. Standard result: sign = (-1)^(...) computed by successive
    // removal/insertion. We compute it operationally.
    let phase = double_phase_same_spin(ket, i, j, a, b);

    phase * (h2(ints, a, i, b, j) - h2(ints, a, j, b, i))
}

/// Opposite-spin double excitation: one α orbital i→a and one β orbital j→b.
///
///   <D'|H|D> = phase_a * phase_b * (ai|bj)
fn double_opposite_spin(
    ints: &ActiveSpaceIntegrals,
    a_bra: u64,
    a_ket: u64,
    b_bra: u64,
    b_ket: u64,
) -> f64 {
    let da = a_bra ^ a_ket;
    let db = b_bra ^ b_ket;
    let i = (da & a_ket).trailing_zeros() as usize; // α removed
    let a = (da & a_bra).trailing_zeros() as usize; // α added
    let j = (db & b_ket).trailing_zeros() as usize; // β removed
    let b = (db & b_bra).trailing_zeros() as usize; // β added

    // α phase
    let pa_i = phase_below(a_ket, i);
    let pa_a = phase_below(a_ket & !(1u64 << i), a);
    // β phase
    let pb_j = phase_below(b_ket, j);
    let pb_b = phase_below(b_ket & !(1u64 << j), b);

    pa_i * pa_a * pb_j * pb_b * h2(ints, a, i, b, j)
}

/// Phase for a same-spin double excitation i,j -> a,b.
///
/// Operationally: annihilate i, then j (using JW parities in the successively
/// depleted string), then create b, then a (in the successively filled string),
/// so the net excitation is D -> D'. This yields the canonical Slater–Condon
/// sign for the ordered pairs (i<j), (a<b).
fn double_phase_same_spin(ket: u64, i: usize, j: usize, a: usize, b: usize) -> f64 {
    // Annihilate i, then j.
    let mut s = ket;
    let mut phase = phase_below(s, i);
    s &= !(1u64 << i);
    phase *= phase_below(s, j);
    s &= !(1u64 << j);
    // Now create b, then a (order chosen to match the (a<b) convention with
    // the (ai|bj)-(aj|bi) integral ordering above).
    phase *= phase_below(s, b);
    s |= 1u64 << b;
    phase *= phase_below(s, a);
    phase
}

/// Bytes the dense Hamiltonian would occupy: `N_det^2` f64.
///
/// Exposed so callers can size the request before making it — `N_det` itself
/// grows combinatorially with the active space, so this matrix grows as the
/// FOURTH power of the active-space binomial and the cliff is exactly one
/// increment wide: CAS(8,10) is 15.6 GB, CAS(10,12) is 3,147 GB.
pub fn dense_hamiltonian_bytes(n_det: usize) -> usize {
    n_det.saturating_mul(n_det).saturating_mul(8)
}

/// Build the dense CI Hamiltonian matrix `H` (n_det x n_det), including
/// `e_core` on the diagonal. O(N_det^2) — spike-scale only (see module doc).
///
/// # Memory
///
/// Fallible, and deliberately so. The production driver uses matrix-free
/// Davidson and never calls this; it is reached today only from this crate's
/// own tests. That makes it a footgun rather than a live exposure — but a
/// `pub fn` with no bound is exactly how the next caller inherits a 3 TB
/// allocation. Errors instead of allocating when the matrix exceeds the
/// resolved budget; pass `Some(b)` to pin the ceiling, `None` to resolve from
/// `FERRIC_MEM_BUDGET_GB` / detected RAM. See
/// `tests/mwe_casci_has_no_guard.rs`.
pub fn dense_hamiltonian(
    ints: &ActiveSpaceIntegrals,
    space: &DeterminantSpace,
    memory_budget_bytes: Option<usize>,
) -> Result<ndarray::Array2<f64>, ferric_core::FerricError> {
    let n_det = space.n_det();
    ferric_core::memory::check_alloc(
        &format!("CAS-CI dense Hamiltonian (N_det={n_det}, N_det^2 matrix)"),
        dense_hamiltonian_bytes(n_det),
        ferric_core::memory::resolve_budget_bytes(memory_budget_bytes),
    )?;
    Ok(dense_hamiltonian_impl(ints, space))
}

fn dense_hamiltonian_impl(
    ints: &ActiveSpaceIntegrals,
    space: &DeterminantSpace,
) -> ndarray::Array2<f64> {
    let na = space.alpha_strings.len();
    let nb = space.beta_strings.len();
    let ndet = na * nb;
    let mut hmat = ndarray::Array2::<f64>::zeros((ndet, ndet));
    for iad in 0..na {
        for ibd in 0..nb {
            let d = iad * nb + ibd;
            for jad in 0..na {
                for jbd in 0..nb {
                    let e = jad * nb + jbd;
                    if e < d {
                        continue; // symmetric; fill lower from upper
                    }
                    let mut val = general_element(
                        ints,
                        space.alpha_strings[iad],
                        space.beta_strings[ibd],
                        space.alpha_strings[jad],
                        space.beta_strings[jbd],
                    );
                    if d == e {
                        val += ints.e_core;
                    }
                    hmat[(d, e)] = val;
                    hmat[(e, d)] = val;
                }
            }
        }
    }
    hmat
}

/// Sigma-vector build `sigma = H c` over the determinant list, without forming
/// H densely. Currently the naive O(N_det^2) pair loop (see module doc). `c`
/// and the returned `sigma` are indexed `d = ia * n_beta + ib`.
pub fn sigma(
    ints: &ActiveSpaceIntegrals,
    space: &DeterminantSpace,
    c: &[f64],
) -> Vec<f64> {
    let na = space.alpha_strings.len();
    let nb = space.beta_strings.len();
    let ndet = na * nb;
    debug_assert_eq!(c.len(), ndet);
    let mut sig = vec![0.0f64; ndet];
    for iad in 0..na {
        for ibd in 0..nb {
            let d = iad * nb + ibd;
            for jad in 0..na {
                for jbd in 0..nb {
                    let e = jad * nb + jbd;
                    if e < d {
                        continue;
                    }
                    let mut val = general_element(
                        ints,
                        space.alpha_strings[iad],
                        space.beta_strings[ibd],
                        space.alpha_strings[jad],
                        space.beta_strings[jbd],
                    );
                    if d == e {
                        val += ints.e_core;
                        sig[d] += val * c[d];
                    } else {
                        sig[d] += val * c[e];
                        sig[e] += val * c[d];
                    }
                }
            }
        }
    }
    sig
}
