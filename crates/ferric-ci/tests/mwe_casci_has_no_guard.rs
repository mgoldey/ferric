//! MWE: is CAS-CI bounded by anything?
//!
//! `ferric-ci` has **no `check_alloc` call sites and no `memory_budget_bytes`
//! field on `CasCiConfig`** — it is the only method crate entirely outside the
//! budget system. Two distinct exposures, at very different scales:
//!
//! ```text
//!   CAS(n_el,n_orb)   N_det      Davidson (production)   dense H (tests only)
//!     CAS( 6, 8)        3,136              0.00 GB                  0.1 GB
//!     CAS( 8,10)       44,100              0.04 GB                 15.6 GB
//!     CAS(10,12)      627,264              0.50 GB              3,147.7 GB
//!     CAS(12,14)    9,018,009              7.21 GB            650,595.9 GB
//!     CAS(14,16)  130,873,600            104.70 GB        137,023,193.4 GB
//! ```
//!
//! **Production** goes through matrix-free Davidson, which holds TWO bases
//! (`v_basis` + `hv_basis`) of up to `max_subspace` vectors of `N_det` doubles:
//! `2 · max_subspace · N_det · 8`, with `max_subspace` defaulting to 50. That
//! is linear in `N_det` but carries a 100x multiplier — CAS(12,14) needs 7.2 GB
//! and CAS(14,16) needs 105 GB, both unguarded.
//!
//! **`dense_hamiltonian`** is `N_det²` and is currently reached only from this
//! crate's own tests. It is a footgun for any future caller rather than a live
//! production exposure — but `N_det` itself grows combinatorially, so the
//! matrix grows as the FOURTH power of the active-space binomial: one increment
//! from a fine CAS(8,10) sits a 3 TB CAS(10,12).
//!
//! Both deserve a gate, sized to their own formula. These contracts pin the
//! arithmetic before either exists. Pure combinatorics — no SCF, no
//! allocation, nothing large ever touched.

/// `N_det = C(n_orb, n_alpha) · C(n_orb, n_beta)`.
fn n_det(n_orb: usize, n_alpha: usize, n_beta: usize) -> usize {
    binom(n_orb, n_alpha).saturating_mul(binom(n_orb, n_beta))
}

fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc = 1usize;
    for i in 0..k {
        acc = acc.saturating_mul(n - i) / (i + 1);
    }
    acc
}

/// Production peak: two Davidson bases of `max_subspace` vectors each.
fn davidson_bytes(ndet: usize, max_subspace: usize) -> usize {
    ndet.saturating_mul(max_subspace).saturating_mul(2).saturating_mul(8)
}

/// The dense Hamiltonian, if a caller ever reaches for it.
fn dense_h_bytes(ndet: usize) -> usize {
    ndet.saturating_mul(ndet).saturating_mul(8)
}

/// CONTRACT 1: the binomial helper matches known determinant counts.
///
/// Everything below rests on this arithmetic, so pin it against values that can
/// be checked by hand before trusting any GB figure derived from it.
#[test]
fn determinant_counts_are_correct() {
    // CAS(6,8): C(8,3)^2 = 56^2 = 3136 — the in-tree H2O CASCI(6,8) test.
    assert_eq!(n_det(8, 3, 3), 3136);
    // CAS(8,10): C(10,4)^2 = 210^2 = 44100.
    assert_eq!(n_det(10, 4, 4), 44100);
    // Full FCI on H2O/STO-3G: 7 orbitals, 5 alpha / 5 beta = C(7,5)^2 = 441.
    assert_eq!(n_det(7, 5, 5), 441);
}

/// CONTRACT 2: an ordinary active space already exceeds this box.
///
/// The severity claim. CAS(14,16) is a request a chemist might reasonably
/// type, and it needs ~105 GB of Davidson vectors alone — on a 23 GB machine.
/// Nothing currently stops it.
#[test]
fn an_ordinary_active_space_exceeds_the_box() {
    const BOX_BYTES: usize = 23 * 1000 * 1000 * 1000;
    let ndet = n_det(16, 7, 7);
    let need = davidson_bytes(ndet, 50);

    assert!(
        need > BOX_BYTES,
        "CAS(14,16) needs {:.1} GB of Davidson vectors, which must exceed this \
         23 GB box for the exposure to be real; got {:.1} GB",
        need as f64 / 1e9,
        need as f64 / 1e9,
    );
}

/// CONTRACT 3: the Davidson peak scales with `max_subspace`.
///
/// `max_subspace` defaults to 50 and is user-settable, so it is a memory knob
/// whether or not it was designed as one. Any guard must read it rather than
/// assume the default — the same class of mistake as counting one copy of a
/// per-worker buffer.
#[test]
fn davidson_peak_scales_with_max_subspace() {
    let ndet = n_det(12, 5, 5);
    let small = davidson_bytes(ndet, 10);
    let default = davidson_bytes(ndet, 50);

    assert_eq!(
        default,
        small * 5,
        "the peak must scale linearly with max_subspace"
    );
}

/// CONTRACT 4: the dense Hamiltonian grows as the FOURTH power of the binomial.
///
/// Why a warning would be useless here and only a hard gate will do: between
/// "fine" and "impossible" there is a single increment of the active space.
#[test]
fn the_dense_hamiltonian_cliff_is_one_increment_wide() {
    let ok = dense_h_bytes(n_det(10, 4, 4)); // CAS(8,10)
    let cliff = dense_h_bytes(n_det(12, 5, 5)); // CAS(10,12)

    assert!(
        cliff / ok > 100,
        "one active-space increment must blow the dense H up by >100x \
         (got {}x): {:.1} GB -> {:.1} GB",
        cliff / ok,
        ok as f64 / 1e9,
        cliff as f64 / 1e9,
    );
}

/// CONTRACT 5: a guard must distinguish the two paths.
///
/// Charging the dense-`N_det²` figure on the production Davidson path would
/// refuse every non-trivial CAS-CI — an over-estimating gate, as broken as
/// none at all. The two formulas differ by orders of magnitude and must stay
/// separate.
#[test]
fn the_two_paths_need_separate_formulas() {
    let ndet = n_det(12, 5, 5); // CAS(10,12)
    let dav = davidson_bytes(ndet, 50);
    let dense = dense_h_bytes(ndet);

    assert!(
        dense > dav * 1000,
        "the dense H ({:.1} GB) must dwarf the Davidson peak ({:.2} GB), so a \
         single shared estimate would be wrong for one path or the other",
        dense as f64 / 1e9,
        dav as f64 / 1e9,
    );
}
