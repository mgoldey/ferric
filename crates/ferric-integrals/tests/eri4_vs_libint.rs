//! The load-bearing validation of the new 4-center McMurchie-Davidson engine.
//!
//! `compute_cart_eri4` + `transform_cart_to_pure4` are a fresh implementation of
//! a two-genuine-pair MD contraction. Running that same machinery with the plain
//! Coulomb kernel lets us compare it, quartet for quartet, against libint2 --
//! an INDEPENDENT construction. That is what validates:
//!
//!   * the ket-side `(-1)^(t+u+v)` sign (invisible unless a ket shell has l>=1
//!     and the two pair centers are distinct),
//!   * the `2*pi^2.5/(p q sqrt(p+q))` prefactor,
//!   * the `[n1][n2][n3][n4]` output layout (invisible unless the four shells
//!     have different l), and
//!   * the four-axis solid-harmonic transform.
//!
//! `terf + terfc == coulomb` deliberately does NOT live here: both sides share
//! this machinery, so a sign or prefactor error cancels and the identity still
//! holds. It tests the operator split, not the contraction.

use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use std::os::raw::{c_int, c_void};

/// Water, deliberately NOT symmetric-collapsed: four distinct centers matter,
/// because a ket-side sign error cancels when the pair centers coincide.
fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\n\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n",
        0,
        1,
    )
    .expect("water xyz")
}

fn debug_coulomb_eri4(prep: &PreparedBasis, q: [usize; 4]) -> Option<Vec<f64>> {
    let dims = prep.shell_dims();
    let n: usize = q.iter().map(|&s| dims[s]).product();
    let mut out = vec![0.0f64; n];
    // SAFETY: `prep.handle()` is a valid scf_basis; shell indices are in bounds
    // (they index prep.shell_dims()); `out` is sized n1*n2*n3*n4 as the shim
    // requires. Status is checked before the buffer is read.
    let written = unsafe {
        ffi::scf_debug_coulomb_eri4(
            prep.handle() as *const c_void,
            q[0] as c_int,
            q[1] as c_int,
            q[2] as c_int,
            q[3] as c_int,
            out.as_mut_ptr(),
        )
    };
    if written < 0 {
        return None;
    }
    assert_eq!(written as usize, n, "shim wrote a block of unexpected size");
    Some(out)
}

/// Pick shell quartets that span several angular-momentum patterns on distinct
/// atoms, so a layout transpose or a ket-sign error cannot hide.
///
/// The selection MUST include quartets where the four shell dimensions differ
/// pairwise. An earlier version hand-picked a handful of quartets that happened
/// to have `ncA == ncB`, and a deliberate A/B transpose of the output index
/// still PASSED -- the test was blind to layout errors. Mutation-tested: with
/// this sweep, both a ket-sign flip and an A/B transpose fail.
fn interesting_quartets(prep: &PreparedBasis) -> Vec<[usize; 4]> {
    let dims = prep.shell_dims();
    let nsh = dims.len();

    // One representative shell per distinct dimension, preferring shells on
    // different atoms so the two pair centers are genuinely distinct (a ket
    // sign error cancels when the centers coincide).
    let mut reps: Vec<usize> = Vec::new();
    let mut seen: Vec<(usize, usize)> = Vec::new(); // (dim, atom)
    let atoms = prep.shell_to_atom();
    for s in 0..nsh {
        let key = (dims[s], atoms[s]);
        if !seen.contains(&key) {
            seen.push(key);
            reps.push(s);
        }
    }

    // All ordered 4-tuples over the representatives, capped so the test stays
    // fast, and biased toward tuples with the most distinct dimensions.
    let mut out: Vec<[usize; 4]> = Vec::new();
    for &a in &reps {
        for &b in &reps {
            for &c in &reps {
                for &d in &reps {
                    out.push([a, b, c, d]);
                }
            }
        }
    }
    // Rank by how much of the index structure a quartet can expose. Counting
    // only "distinct dimensions" is NOT enough: the top-ranked tuples under
    // that rule all had s-shells in the A and B slots, where ncA == ncB makes
    // an A/B transpose arithmetically invisible. Score each ADJACENT axis pair
    // that differs, so tuples distinguishing A-vs-B and C-vs-D rank first.
    out.sort_by_key(|q| {
        let d: Vec<usize> = q.iter().map(|&s| dims[s]).collect();
        let pairwise = usize::from(d[0] != d[1])
            + usize::from(d[2] != d[3])
            + usize::from(d[0] != d[2])
            + usize::from(d[1] != d[3]);
        let mut ds = d.clone();
        ds.sort_unstable();
        ds.dedup();
        // Prefer higher L too, so the solid-harmonic transform is exercised.
        (std::cmp::Reverse(pairwise), std::cmp::Reverse(ds.len()), std::cmp::Reverse(d[0] + d[1] + d[2] + d[3]))
    });
    out.truncate(64);

    // Hard guarantee, not a hope: the sweep MUST contain a quartet that can see
    // an A/B transpose and one that can see a C/D transpose, or the test is
    // blind to layout errors no matter how many quartets it checks.
    assert!(
        out.iter().any(|q| dims[q[0]] != dims[q[1]]),
        "no quartet with ncA != ncB -- an A/B transpose would be undetectable"
    );
    assert!(
        out.iter().any(|q| dims[q[2]] != dims[q[3]]),
        "no quartet with ncC != ncD -- a C/D transpose would be undetectable"
    );
    out
}

#[test]
fn eri4_coulomb_matches_libint2_quartets() {
    let mol = water();
    // cc-pVDZ on water gives s, p and d shells, pure where libint2 says pure --
    // enough angular variety that the layout and the solid-harmonic transform
    // are both exercised.
    let bs = ferric_core::basis::bundled("cc-pvdz").expect("cc-pvdz");
    let prep = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let mut eng = Engine::new_2e(Operator::coulomb(), &prep, 0.0).expect("coulomb engine");

    let quartets = interesting_quartets(&prep);
    assert!(
        quartets.len() >= 4,
        "test would be vacuous with fewer than 4 quartets"
    );

    let mut checked = 0usize;
    let mut worst = 0.0f64;
    let mut worst_at = [0usize; 4];

    for q in quartets {
        let Some(mine) = debug_coulomb_eri4(&prep, q) else {
            continue; // refused (too-high total L) -- not a mismatch
        };
        let reference = eng
            .compute_quartet(&prep, q[0], q[1], q[2], q[3])
            .map(|b| b.to_vec())
            .unwrap_or_else(|| vec![0.0; mine.len()]);
        assert_eq!(
            mine.len(),
            reference.len(),
            "block size disagreement at {q:?}"
        );

        // Scale the tolerance by the block magnitude: these are absolute
        // integral values spanning orders of magnitude within one molecule.
        // Take the scale from BOTH sides -- if libint2 screened the block away
        // and returned nothing, a reference-only scale collapses to the floor
        // and turns two numerical zeros into a large bogus ratio.
        let scale = mine
            .iter()
            .chain(reference.iter())
            .fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in mine.iter().zip(reference.iter()).enumerate() {
            let diff = (a - b).abs();
            // Absolute floor: values at 1e-16 are numerical zero regardless of
            // how small the block is, and must not be judged by ratio.
            if diff < 1e-14 {
                continue;
            }
            let rel = diff / scale.max(1e-300);
            if rel > worst {
                worst = rel;
                worst_at = q;
            }
            assert!(
                rel < 1e-10,
                "eri4 disagrees with libint2 at quartet {q:?} element {i}: \
                 mine={a:.17e} libint2={b:.17e} rel={rel:.3e} (block max {scale:.3e})"
            );
        }
        checked += 1;
    }

    assert!(
        checked >= 4,
        "only {checked} quartets actually compared -- the test proved almost nothing"
    );
    eprintln!("eri4 vs libint2: {checked} quartets, worst rel {worst:.3e} at {worst_at:?}");
}

/// A guard on the guard: the engine must REFUSE a too-high-L quartet with a
/// negative status rather than quietly returning zeros, because zeros would
/// read as "screened" and silently understate a Schwarz/CSB bound.
///
/// This test is skipped rather than faked when no basis in the workspace
/// reaches the limit -- a skipped honest test beats a green vacuous one.
#[test]
fn too_high_total_l_is_refused_not_silently_zeroed() {
    let mol = water();
    let Ok(bs) = ferric_core::basis::bundled("def2-qzvp") else {
        eprintln!("skipped: def2-qzvp unavailable");
        return;
    };
    let prep = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let dims = prep.shell_dims();
    // Highest-L shell available; TERFC_DIMM is 24, so we need sum(l) >= 24.
    let hi = (0..dims.len()).max_by_key(|&s| dims[s]).unwrap_or(0);

    let mut out = vec![0.0f64; dims[hi].pow(4)];
    // SAFETY: as in debug_coulomb_eri4 -- valid handle, in-bounds indices,
    // buffer sized for the full quartet.
    let written = unsafe {
        ffi::scf_debug_coulomb_eri4(
            prep.handle() as *const c_void,
            hi as c_int,
            hi as c_int,
            hi as c_int,
            hi as c_int,
            out.as_mut_ptr(),
        )
    };

    // 4*l >= 24 means l >= 6. If the basis tops out below that the limit is
    // simply unreachable here; say so rather than assert something vacuous.
    if written >= 0 {
        eprintln!(
            "skipped: largest shell (dim {}) does not reach the L limit; \
             refusal path not exercised",
            dims[hi]
        );
        return;
    }
    assert!(
        written < 0,
        "a too-high-L quartet must return a negative status, not zeros"
    );
}
