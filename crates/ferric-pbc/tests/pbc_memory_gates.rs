//! Stage-1 PBC step 10: memory gates (reference/pbc/stage1-design.md §5).
//!
//! Every large ferric-pbc buffer is reserved against the budget BEFORE it is
//! allocated (`ferric_pbc::budget`), and the `pair_ft` output is G-chunked.
//!
//! * Refusal tests assert the quantity's NAME and its exact BYTE COUNT
//!   (computed here from the shapes), not just a word: a message that lost
//!   the number would still contain "budget".
//! * Chunk-invariance tests force MANY chunks with a budget just above the
//!   resident set and compare against the one-chunk (ample-budget) run:
//!   `pair_ft` and `periodic_hcore` must agree BITWISE (the G window uses the
//!   global max |G|; V_LR accumulates in G order); `DenseAftEri` changes the
//!   GEMM summation order, so it is held to a DERIVED roundoff bound
//!   (`dense_aft_chunking_changes_only_roundoff`).
//!
//! Mutations that should turn these red:
//! * `pair_ft_chunked` passing the chunk's own max |G| to the kernel instead
//!   of the global one → `pair_ft_chunked_is_bitwise_the_unchunked_transform`
//!   (only if a primitive pair's window edge falls between the two — the test
//!   prints how many G columns differ; 0 means the mutation was not reached).
//! * Dropping `ledger.reserve` for the matrices → the Some(1) test errors on
//!   a DIFFERENT quantity and its name assertion fails.
//! * `chunk = chunk_budget / per_g` → `/ 1` → the forced-chunk tests see 1
//!   chunk and fail their chunk-count assertion.

mod common;

use common::*;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::pair_ft::{
    pair_ft_bytes_per_g, pair_ft_chunked, pair_ft_with_thresh, DEFAULT_PAIR_FT_THRESH,
};
use ndarray::{Array2, Array3};
use num_complex::Complex64;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 30;

fn bits_equal(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn pair_ft_chunked_is_bitwise_the_unchunked_transform() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let nao = prep.nbasis();
    let gv = cell.gvectors(4.0).unwrap();
    let ng = gv.len();
    assert!(ng > 20, "need many G to force several chunks, got {ng}");
    let full = pair_ft_with_thresh(&cell, &prep, &gv, DEFAULT_PAIR_FT_THRESH).unwrap();

    let per_g = pair_ft_bytes_per_g(nao, 1);
    for (per_chunk, expect_chunks) in [(5usize, ng.div_ceil(5)), (ng, 1)] {
        let mut got = Array3::<Complex64>::zeros((nao, nao, ng));
        let mut seen = 0usize;
        let n = pair_ft_chunked(
            &cell,
            &prep,
            &gv,
            DEFAULT_PAIR_FT_THRESH,
            per_chunk * per_g,
            0,
            |g0, gs, p| {
                assert_eq!(g0, seen, "chunks must arrive in order");
                for j in 0..gs.len() {
                    for m in 0..nao {
                        for k in 0..nao {
                            got[[m, k, g0 + j]] = p[[m, k, j]];
                        }
                    }
                }
                seen += gs.len();
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(seen, ng);
        assert_eq!(n, expect_chunks, "{per_chunk} G per chunk");
        let ndiff = full
            .iter()
            .zip(got.iter())
            .filter(|(a, b)| a.re.to_bits() != b.re.to_bits() || a.im.to_bits() != b.im.to_bits())
            .count();
        eprintln!("{per_chunk} G/chunk: {n} chunks, {ndiff} differing entries");
        assert_eq!(
            ndiff, 0,
            "chunked pair_ft must be bitwise the unchunked one"
        );
    }
}

#[test]
fn pair_ft_chunked_refuses_a_chunk_that_cannot_hold_one_g() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let nao = prep.nbasis();
    let gv = cell.gvectors(2.0).unwrap();
    let extra = 1000;
    let per_g = pair_ft_bytes_per_g(nao, 0) + extra;
    let err = pair_ft_chunked(&cell, &prep, &gv, 1e-14, per_g - 1, extra, |_, _, _| Ok(()))
        .expect_err("one G does not fit");
    let msg = err.to_string();
    assert!(
        msg.contains("pair_ft chunk, one G vector")
            && msg.contains(&format!("({per_g} bytes; {} of the", per_g - 1)),
        "{msg}"
    );
    // Exactly one G fits: one chunk per G.
    let n = pair_ft_chunked(&cell, &prep, &gv, 1e-14, per_g, extra, |_, gs, _| {
        assert_eq!(gs.len(), 1);
        Ok(())
    })
    .unwrap();
    assert_eq!(n, gv.len());
}

#[test]
fn periodic_hcore_tiny_budget_names_the_matrices() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let n = prep.nbasis();
    let cfg = PeriodicHcoreConfig {
        budget_bytes: Some(1),
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    };
    let msg = periodic_hcore(&cell, &prep, &cfg)
        .expect_err("1-byte budget")
        .to_string();
    let need = 8 * 10 * n * n;
    assert!(
        msg.contains("periodic_hcore n×n matrices")
            && msg.contains(&format!("({need} bytes; 1 of the 1 byte budget left)")),
        "{msg}"
    );
}

#[test]
fn periodic_hcore_lr_chunk_gate_names_one_g() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let base = PeriodicHcoreConfig {
        budget_bytes: Some(AMPLE),
        ..PeriodicHcoreConfig::with_omega(1.3)
    };
    let hc = periodic_hcore(&cell, &prep, &base).unwrap();
    let per_g = hc.lr_bytes_per_g;
    let cfg = PeriodicHcoreConfig {
        budget_bytes: Some(hc.lr_resident_bytes + per_g - 1),
        ..base
    };
    let msg = periodic_hcore(&cell, &prep, &cfg)
        .expect_err("no room for one G")
        .to_string();
    assert!(
        msg.contains("pair_ft chunk, one G vector")
            && msg.contains(&format!("({per_g} bytes; {} of the", per_g - 1)),
        "{msg}"
    );
}

#[test]
fn periodic_hcore_chunking_is_bitwise_invariant() {
    let systems = [
        ("H2/STO-3G a=4", h2_cell(4.0), pyscf_sto3g_h(), OMEGA),
        ("triclinic 4H s+p", triclinic_cell(), sp_basis_h(), 1.3),
    ];
    for (name, cell, bs, omega) in systems {
        let prep = prep_for(&cell, &bs);
        let ample = PeriodicHcoreConfig {
            budget_bytes: Some(AMPLE),
            ..PeriodicHcoreConfig::with_omega(omega)
        };
        let one = periodic_hcore(&cell, &prep, &ample).unwrap();
        assert_eq!(
            one.n_lr_chunks, 1,
            "{name}: ample budget should be one chunk"
        );
        let per_chunk = 7;
        let forced = PeriodicHcoreConfig {
            budget_bytes: Some(one.lr_resident_bytes + per_chunk * one.lr_bytes_per_g),
            ..ample
        };
        let many = periodic_hcore(&cell, &prep, &forced).unwrap();
        eprintln!(
            "{name}: {} half-G, {} chunks forced (resident {} B, {} B/G)",
            many.n_g_half, many.n_lr_chunks, one.lr_resident_bytes, one.lr_bytes_per_g
        );
        assert_eq!(many.n_lr_chunks, one.n_g_half.div_ceil(per_chunk), "{name}");
        assert!(many.n_lr_chunks > 1, "{name}: chunking not exercised");
        assert!(bits_equal(&one.v_lr, &many.v_lr), "{name}: V_LR");
        assert!(bits_equal(&one.h, &many.h), "{name}: h");
        // The default (unified) budget takes the same one-chunk path.
        let dflt = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(omega)).unwrap();
        assert!(bits_equal(&one.h, &dflt.h), "{name}: default budget");
    }
}

#[test]
fn dense_aft_tiny_budget_names_the_tensor() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let nao = prep.nbasis();
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let need = 16 * nao.pow(4);
    let msg = DenseAftEri::build_budgeted(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        1e-10,
        DEFAULT_DENSE_AFT_MAX_BYTES,
        Some(need - 1),
    )
    .expect_err("tensor does not fit")
    .to_string();
    assert!(
        msg.contains("DenseAftEri nao^4 tensor")
            && msg.contains(&format!("({need} bytes; {} of the", need - 1)),
        "{msg}"
    );
}

/// Chunking reorders the GEMM sums, so the result is NOT bitwise stable.
/// Derived bound: each element `I_ab = Σ_g w_g (re_a re_b + im_a im_b)` with
/// `w_g > 0`, so `Σ_g |term| ≤ Σ_g w_g |P_a||P_b| ≤ √(I_aa I_bb)`
/// (Cauchy–Schwarz; the diagonal is a sum of non-negative terms). Any
/// summation order is within `γ_k √(I_aa I_bb)` of the exact sum (Higham,
/// `γ_k = kε/(1−kε)`, `k` = number of terms), so two orders differ by at most
/// `2γ_k √(I_aa I_bb)`; the final `½(I + Iᵀ)` adds ≤ `ε|I_ab|` to each.
/// `k = 2 n_G + 2` (re and im GEMMs, plus the chunk-sum additions).
#[test]
fn dense_aft_chunking_changes_only_roundoff() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let build = |budget| {
        DenseAftEri::build_budgeted(
            &cell,
            &prep,
            &hc.s,
            ExxDiv::Ewald,
            1e-10,
            DEFAULT_DENSE_AFT_MAX_BYTES,
            budget,
        )
        .unwrap()
    };
    let one = build(Some(AMPLE));
    assert_eq!(one.n_g_chunks(), 1);
    // Default budget: same single chunk, same bits.
    let dflt = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        1e-10,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    assert_eq!(dflt.n_g_chunks(), 1);
    assert!(bits_equal(one.eri(), dflt.eri()), "default vs ample budget");

    let per_chunk = 5;
    let many = build(Some(one.resident_bytes() + per_chunk * one.bytes_per_g()));
    assert_eq!(many.n_g_chunks(), one.n_g_half().div_ceil(per_chunk));
    assert!(many.n_g_chunks() > 1, "chunking not exercised");

    let (a, b) = (one.eri(), many.eri());
    let k = (2 * one.n_g_half() + 2) as f64;
    let eps = f64::EPSILON;
    let gamma = k * eps / (1.0 - k * eps);
    let n2 = a.nrows();
    let (mut worst_ratio, mut max_diff) = (0.0_f64, 0.0_f64);
    for i in 0..n2 {
        for j in 0..n2 {
            let d = (a[(i, j)] - b[(i, j)]).abs();
            let bound = 2.0 * gamma * (a[(i, i)] * a[(j, j)]).sqrt() + 2.0 * eps * a[(i, j)].abs();
            max_diff = max_diff.max(d);
            if bound > 0.0 {
                worst_ratio = worst_ratio.max(d / bound);
            } else {
                assert_eq!(d, 0.0, "({i},{j}): nonzero diff with zero bound");
            }
        }
    }
    eprintln!(
        "dense AFT: {} chunks vs 1: max|dI| = {max_diff:.3e}, worst |dI|/bound = {worst_ratio:.3e} (k = {k})",
        many.n_g_chunks()
    );
    assert!(
        worst_ratio <= 1.0,
        "chunking moved I beyond the roundoff bound"
    );

    // End to end: the SCF energy moves by roundoff only.
    let e1 = gamma_rhf(&cell, &prep, &hc, &one).energy;
    let e2 = gamma_rhf(&cell, &prep, &hc, &many).energy;
    eprintln!(
        "dense AFT: E(1 chunk) - E({} chunks) = {:.3e}",
        many.n_g_chunks(),
        e1 - e2
    );
    assert!((e1 - e2).abs() < 1e-12, "{e1} vs {e2}");
}
