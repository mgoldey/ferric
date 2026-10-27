//! cDFT-ET coupling: kernel unit tests on synthetic matrices, then He₂⁺
//! end-to-end identities. Kernel tests use S = I so det(Mσ) = det(C_aᵀ C_b).

use ferric_scf::cdft_coupling::biorth_pairing;
use ndarray::{array, Array2};

/// Identical occupied sets with S = I → det(M) = 1 (singular values all 1).
#[test]
fn pairing_identical_sets_det_one() {
    let s = Array2::<f64>::eye(3);
    // Two occupied orbitals = first two columns of I_3.
    let c = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
    let p = biorth_pairing(&c, &c, &s);
    assert!((p.det_m - 1.0).abs() < 1e-12, "det_m {}", p.det_m);
    assert_eq!(p.s_vals.len(), 2);
    for sv in p.s_vals.iter() {
        assert!((sv - 1.0).abs() < 1e-12);
    }
}

/// A column swap between the two sets flips the determinant sign (|det| = 1).
#[test]
fn pairing_swapped_columns_det_minus_one() {
    let s = Array2::<f64>::eye(3);
    let c_a = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
    let c_b = array![[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]; // columns swapped
    let p = biorth_pairing(&c_a, &c_b, &s);
    // SVD singular values are non-negative, so |det_m| = product = 1.
    assert!(
        (p.det_m.abs() - 1.0).abs() < 1e-12,
        "|det_m| {}",
        p.det_m.abs()
    );
}

/// For identical α and β sets (S = I, C_a = C_b), the one-body element equals
/// the ordinary expectation Σ_σ Σ_i ⟨i|Ô|i⟩ (since S_ab = 1 and all s_i = 1).
#[test]
fn cross_one_body_identical_is_expectation() {
    use ferric_scf::cdft_coupling::{biorth_pairing, cross_one_body};
    let s = Array2::<f64>::eye(3);
    let c = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]; // 2 occ
                                                        // Operator: diagonal AO operator diag(2,3,5).
    let mut op = Array2::<f64>::zeros((3, 3));
    op[(0, 0)] = 2.0;
    op[(1, 1)] = 3.0;
    op[(2, 2)] = 5.0;
    let pa = biorth_pairing(&c, &c, &s);
    let pb = biorth_pairing(&c, &c, &s);
    let s_ab = pa.det_m * pb.det_m; // = 1
    let val = cross_one_body(&op, &pa, &pb, s_ab);
    // Two spins, each occupying AO0 and AO1: ⟨0|op|0⟩+⟨1|op|1⟩ = 2+3 = 5 per
    // spin, ×2 spins = 10.
    assert!((val - 10.0).abs() < 1e-10, "got {val}");
}

/// Build α sets that share one orbital but whose second orbitals are mutually
/// orthogonal (one zero singular value). S_ab = 0, but the one-body element is
/// finite and equals (Π nonzero) · ⟨ã_k|Ô|b̃_k⟩ for the paired zero orbital.
/// β sets identical (det_β = 1) so the α-zero is the only zero.
#[test]
fn cross_one_body_single_zero_overlap_is_finite() {
    use ferric_scf::cdft_coupling::{biorth_pairing, cross_one_body};
    let s = Array2::<f64>::eye(4);
    // α_a occupies AO0, AO1 ; α_b occupies AO0, AO2 → second orbitals orthogonal.
    let c_a = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
    let c_b = array![[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
    let pa = biorth_pairing(&c_a, &c_b, &s);
    // One singular value should be ~1 (shared AO0) and one ~0 (orthogonal pair).
    let n_zero = pa.s_vals.iter().filter(|&&s| s < 1e-8).count();
    assert_eq!(
        n_zero, 1,
        "expected exactly one zero overlap, s={:?}",
        pa.s_vals
    );

    // β identical (occupies AO0, AO1).
    let cb = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
    let pb = biorth_pairing(&cb, &cb, &s);

    let s_ab = pa.det_m * pb.det_m;
    assert!(s_ab.abs() < 1e-10, "S_ab should be 0, got {s_ab}");

    // Operator that connects AO1 and AO2 (the orthogonal pair): off-diagonal.
    let mut op = Array2::<f64>::zeros((4, 4));
    op[(1, 2)] = 1.0;
    op[(2, 1)] = 1.0;
    let val = cross_one_body(&op, &pa, &pb, s_ab);
    // Finite and nonzero: the single zero-overlap pair carries the element.
    assert!(val.is_finite(), "element not finite: {val}");
    assert!(val.abs() > 1e-6, "expected nonzero element, got {val}");
}

/// COFACTOR BRANCH: the OTHER spin's determinant is a real factor.
///
/// `cross_one_body_single_zero_overlap_is_finite` above exercises the same
/// `nz == 1` branch but makes the β sets IDENTICAL, so det_β = 1 exactly and
/// the `* det_other` factor is invisible. MUTATION-PROVEN: deleting `det_other`
/// from that branch leaves the rest of this file GREEN. Here β is rotated so
/// det_β = cos(0.5) ≈ 0.8776 ≠ 1, and the element must scale with it.
///
/// The reference is INDEPENDENT of the branch under test: the element is
/// computed twice with two different β rotations and the RATIO is checked
/// against the ratio of the β determinants, so no absolute value from the
/// kernel is used to validate the kernel.
#[test]
fn cross_one_body_cofactor_carries_the_other_spin_determinant() {
    use ferric_scf::cdft_coupling::{biorth_pairing, cross_one_body};
    let s = Array2::<f64>::eye(4);
    // α: one shared orbital (AO0) and one orthogonal pair (AO1 vs AO2) → one zero.
    let c_a = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
    let c_b = array![[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
    let pa = biorth_pairing(&c_a, &c_b, &s);
    assert_eq!(
        pa.s_vals.iter().filter(|&&v| v < 1e-8).count(),
        1,
        "expected exactly one α zero, s = {:?}",
        pa.s_vals
    );

    // Operator connecting the orthogonal α pair.
    let mut op = Array2::<f64>::zeros((4, 4));
    op[(1, 2)] = 1.0;
    op[(2, 1)] = 1.0;

    // β: one occupied orbital, rotated by angle t in the (0,3) plane so
    // det_β = cos(t), tunable and ≠ 1.
    let beta_case = |t: f64| -> (f64, f64) {
        let cb_a = array![[1.0], [0.0], [0.0], [0.0]];
        let cb_b = array![[t.cos()], [0.0], [0.0], [t.sin()]];
        let pb = biorth_pairing(&cb_a, &cb_b, &s);
        let s_ab = pa.det_m * pb.det_m;
        (cross_one_body(&op, &pa, &pb, s_ab), pb.det_m)
    };
    let (v1, d1) = beta_case(0.5);
    let (v2, d2) = beta_case(1.0);

    // Reachability: the two β determinants must actually differ, else the ratio
    // check is vacuous.
    assert!(
        (d1 - d2).abs() > 0.1,
        "β determinants coincide ({d1:.6} vs {d2:.6}); cofactor factor untestable"
    );
    assert!(v1.abs() > 1e-6, "α cofactor element vanished: {v1}");

    // The element is (Π nonzero α s) · d_k^α · det_β, so v1/v2 == d1/d2 exactly.
    assert!(
        (v1 / v2 - d1 / d2).abs() < 1e-10,
        "cofactor does not scale with the other spin's determinant: \
         v1/v2 = {:.12} but det_β1/det_β2 = {:.12}",
        v1 / v2,
        d1 / d2
    );
}

/// ≥2 near-zero singular values ⇒ the one-body element is exactly zero.
///
/// A one-body operator connects determinants differing by at most ONE spin
/// orbital, so two or more broken pairings kill it. That branch had NO test:
/// MUTATION-PROVEN, replacing its `0.0` with `1.0` left the whole file green.
/// Both sub-cases are covered — two zeros in one spin, and one zero in each.
#[test]
fn cross_one_body_two_zero_overlaps_is_exactly_zero() {
    use ferric_scf::cdft_coupling::{biorth_pairing, cross_one_body};
    let s = Array2::<f64>::eye(6);
    let e = |rows: [usize; 2]| -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((6, 2));
        m[(rows[0], 0)] = 1.0;
        m[(rows[1], 1)] = 1.0;
        m
    };
    let e1 = |row: usize| -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((6, 1));
        m[(row, 0)] = 1.0;
        m
    };
    // An operator with support everywhere, so a nonzero answer is not excluded
    // for a trivial reason.
    let op = Array2::<f64>::from_elem((6, 6), 1.0);

    // Case 1: BOTH α pairs orthogonal (two zeros in α); β identical.
    let pa = biorth_pairing(&e([0, 1]), &e([2, 3]), &s);
    let pb = biorth_pairing(&e1(4), &e1(4), &s);
    assert_eq!(
        pa.s_vals.iter().filter(|&&v| v < 1e-8).count(),
        2,
        "case 1 needs two α zeros, s = {:?}",
        pa.s_vals
    );
    let v1 = cross_one_body(&op, &pa, &pb, pa.det_m * pb.det_m);
    assert!(v1.abs() < 1e-14, "two α zeros must give 0, got {v1}");

    // Case 2: one zero in α AND one in β.
    let pa2 = biorth_pairing(&e([0, 1]), &e([0, 2]), &s);
    let pb2 = biorth_pairing(&e1(3), &e1(4), &s);
    assert_eq!(
        pa2.s_vals.iter().filter(|&&v| v < 1e-8).count()
            + pb2.s_vals.iter().filter(|&&v| v < 1e-8).count(),
        2,
        "case 2 needs one zero in each spin"
    );
    let v2 = cross_one_body(&op, &pa2, &pb2, pa2.det_m * pb2.det_m);
    assert!(
        v2.abs() < 1e-14,
        "one zero in each spin must give 0, got {v2}"
    );

    // Reachability: the SAME operator gives a NONZERO element on the nz == 1
    // fixture, so these zeros come from the branch logic and not from an
    // operator that can never connect anything.
    let pb3 = biorth_pairing(&e1(4), &e1(4), &s);
    let pa3 = biorth_pairing(&e([0, 1]), &e([0, 2]), &s);
    let v3 = cross_one_body(&op, &pa3, &pb3, pa3.det_m * pb3.det_m);
    assert!(
        v3.abs() > 1e-6,
        "reachability failed: the nz == 1 element is also ~0 ({v3}), so the \
         zeros above prove nothing about the ≥2 branch"
    );
}

/// Identical states: S_ab = 1, and the degenerate-denominator guard returns
/// cleanly (no NaN/Inf). Uses a tiny synthetic state.
#[test]
fn coupling_identical_state_is_clean() {
    use ferric_scf::cdft_coupling::{coupling_hab, DiabaticState};
    let s = Array2::<f64>::eye(3);
    let c = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let w = Array2::<f64>::eye(3);
    let st = DiabaticState {
        c_a: &c,
        c_b: &c,
        nocc_a: 2,
        nocc_b: 1,
        energy: -3.0,
        lambda: 0.5,
        w: &w,
    };
    let r = coupling_hab(&st, &st, &s);
    assert!((r.s_ab - 1.0).abs() < 1e-10, "S_ab {}", r.s_ab);
    assert!(r.h_ab.is_finite(), "H_ab not finite");
}

/// Shared fixture for the a↔b symmetry / weight-pairing tests below.
///
/// EVERY DEGENERACY IS BROKEN ON PURPOSE. The previous version of this fixture
/// gave BOTH states the SAME weight operator (`Array2::eye(3)`), which makes the
/// (λ, W) pairing structure of the Wu–VV raw element VACUOUS: with W_a ≡ W_b the
/// two matrix elements ⟨a|W_a|b⟩ and ⟨a|W_b|b⟩ are numerically identical, so
/// mispairing λ_a with W_b (or dropping one W entirely) changes nothing. That
/// class of error was measured to be undetectable on the shared-`w` fixture and
/// is detectable here — see `coupling_pairs_each_lambda_with_its_own_weight`.
///
/// The four discriminating properties, all required:
///   * `W_A != W_B`, both symmetric, and NON-COMMUTING (W_A W_B != W_B W_A), so
///     no accidental simultaneous-eigenbasis collapse.
///   * BOTH ⟨a|W_A|b⟩ and ⟨a|W_B|b⟩ are O(1) and nonzero. An earlier attempt
///     used block-disjoint weights and got ⟨a|W_B|b⟩ = 0 EXACTLY, which silently
///     re-degenerates the λ_b term to nothing.
///   * λ_A != λ_B and E_A != E_B.
///   * S_ab = 0.848, i.e. O(1). On the real HeNe⁺ diabats S_ab ≈ 1e-13 at three
///     of four separations, which collapses every term of `h_raw` to ~0 and makes
///     any assertion there vacuous regardless of the rest of the construction.
///
/// Returns `(S, C_1, C_2, W_A, W_B)`.
#[allow(clippy::type_complexity)]
fn swap_fixture() -> (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let s = Array2::<f64>::eye(3);
    let c1 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // c2: a rotation in the (0,2) plane so S_ab ≠ 1 but is comfortably O(1).
    let t = 0.4_f64;
    let (ct, st_) = (t.cos(), t.sin());
    let c2 = array![[ct, 0.0, -st_], [0.0, 1.0, 0.0], [st_, 0.0, ct]];
    // Two DIFFERENT symmetric weight operators, shaped like fragment-localized
    // Becke weights: W_A weights mostly AO0, W_B mostly AO2, with off-diagonal
    // bleed so neither is diagonal and they do not commute.
    let w_a = array![[0.9, 0.1, 0.0], [0.1, 0.3, 0.2], [0.0, 0.2, 0.1]];
    let w_b = array![[0.1, 0.0, 0.2], [0.0, 0.7, 0.1], [0.2, 0.1, 0.9]];
    (s, c1, c2, w_a, w_b)
}

/// REACHABILITY CHECK for the fixture above, run before anything asserts on it.
///
/// A symmetry test can only detect an asymmetric bug if the fixture can actually
/// PRODUCE an asymmetry. This pins that the discriminating quantities are not
/// accidentally degenerate: the two weight elements differ, the weights do not
/// commute, and S_ab is O(1) rather than the ~1e-13 seen on real diabats.
///
/// Without this, a future edit that quietly re-degenerates the fixture (equal
/// weights, near-zero overlap) would turn the two tests below INERT while they
/// continued to read as coverage — exactly the failure this file is closing.
#[test]
fn swap_fixture_is_not_degenerate() {
    use ferric_scf::cdft_coupling::{biorth_pairing, cross_one_body};
    let (s, c1, c2, w_a, w_b) = swap_fixture();

    // Non-commuting, distinct, symmetric.
    let comm = w_a.dot(&w_b) - w_b.dot(&w_a);
    let comm_norm = comm.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        comm_norm > 1e-3,
        "W_A and W_B commute (‖[W_A,W_B]‖ = {comm_norm:.3e}); a shared \
         eigenbasis re-degenerates the λ-pairing"
    );
    for (wx, name) in [(&w_a, "W_A"), (&w_b, "W_B")] {
        let asym = (wx.to_owned() - wx.t())
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(asym < 1e-14, "{name} is not symmetric (asym {asym:.3e})");
    }

    // Both weight elements must be O(1) and DIFFERENT, on the same pairing the
    // kernel builds internally.
    let pa = biorth_pairing(
        &c1.slice(ndarray::s![.., ..2]).to_owned(),
        &c2.slice(ndarray::s![.., ..2]).to_owned(),
        &s,
    );
    let pb = biorth_pairing(
        &c1.slice(ndarray::s![.., ..1]).to_owned(),
        &c2.slice(ndarray::s![.., ..1]).to_owned(),
        &s,
    );
    let s_ab = pa.det_m * pb.det_m;
    assert!(
        s_ab.abs() > 0.1 && s_ab.abs() < 0.99,
        "S_ab = {s_ab:.6e} is not O(1); every h_raw term collapses"
    );
    let e_wa = cross_one_body(&w_a, &pa, &pb, s_ab);
    let e_wb = cross_one_body(&w_b, &pa, &pb, s_ab);
    assert!(
        e_wa.abs() > 0.1 && e_wb.abs() > 0.1,
        "a weight element vanished: ⟨a|W_A|b⟩ = {e_wa:.6e}, ⟨a|W_B|b⟩ = {e_wb:.6e}"
    );
    assert!(
        (e_wa - e_wb).abs() > 0.1,
        "⟨a|W_A|b⟩ == ⟨a|W_B|b⟩ ({e_wa:.6e} vs {e_wb:.6e}); the λ-pairing is vacuous"
    );

    // Finally, the pass condition itself is reachable: with these numbers the
    // ONE-SIDED raw element (keeping only the b-term of the Wu–VV average)
    // differs from the averaged one, so an averaging bug has room to show.
    // F_X = E_X + λ_X N_X with N_X each state's own population (the kernel's
    // constraint-offset-invariant raw element; N_A = 2.1, N_B = 1.429577).
    let (e_a, l_a) = (-3.0_f64, 0.5_f64);
    let (e_b, l_b) = (-2.9_f64, 0.4_f64);
    let own_pop = |c: &Array2<f64>, w: &Array2<f64>| -> f64 {
        let ca = c.slice(ndarray::s![.., ..2]).to_owned();
        let cb = c.slice(ndarray::s![.., ..1]).to_owned();
        ca.t().dot(w).dot(&ca).diag().sum() + cb.t().dot(w).dot(&cb).diag().sum()
    };
    let f_a = e_a + l_a * own_pop(&c1, &w_a);
    let f_b = e_b + l_b * own_pop(&c2, &w_b);
    let h_raw_avg = 0.5 * ((f_b * s_ab - l_b * e_wb) + (f_a * s_ab - l_a * e_wa));
    let h_raw_one_sided = f_b * s_ab - l_b * e_wb;
    // Measured gap 0.1036 (numpy rebuild of this fixture); bar 0.05.
    assert!(
        (h_raw_avg - h_raw_one_sided).abs() > 0.05,
        "averaged and one-sided h_raw coincide ({h_raw_avg:.9} vs \
         {h_raw_one_sided:.9}); the fixture cannot detect a missing average"
    );
}

/// a↔b symmetry: swapping the two states gives the same S_ab and H_ab.
///
/// MUTATION-PROVEN. Replacing the symmetric Wu–VV average in
/// `cdft_coupling.rs` with either half alone —
/// `let h_raw = f_b * s_ab - state_b.lambda * w_b_elem;` — must make this fail.
/// Predicted by an independent numpy rebuild of the mutant under the
/// F = E + λN raw element (2026-10-01; the earlier measured values belonged to
/// the superseded E-only element): H_ab = +0.587722 vs −0.151331 (swap
/// difference 0.739). Using state A's weight for both λ terms: difference
/// 0.237. Re-confirm by mutation after any change here. The non-degeneracy of
/// the inputs that makes those reachable is pinned by
/// `swap_fixture_is_not_degenerate`.
#[test]
fn coupling_symmetric_under_swap() {
    use ferric_scf::cdft_coupling::{coupling_hab, DiabaticState};
    let (s, c1, c2, w_a, w_b) = swap_fixture();
    let a = DiabaticState {
        c_a: &c1,
        c_b: &c1,
        nocc_a: 2,
        nocc_b: 1,
        energy: -3.0,
        lambda: 0.5,
        w: &w_a,
    };
    let b = DiabaticState {
        c_a: &c2,
        c_b: &c2,
        nocc_a: 2,
        nocc_b: 1,
        energy: -2.9,
        lambda: 0.4,
        w: &w_b,
    };
    let ab = coupling_hab(&a, &b, &s);
    let ba = coupling_hab(&b, &a, &s);
    assert!(
        (ab.s_ab - ba.s_ab).abs() < 1e-10,
        "S_ab asym {} {}",
        ab.s_ab,
        ba.s_ab
    );
    assert!(
        (ab.h_ab - ba.h_ab).abs() < 1e-10,
        "H_ab asym {} {}",
        ab.h_ab,
        ba.h_ab
    );
}

/// Each λ must multiply ITS OWN state's weight operator.
///
/// WHY THIS EXISTS SEPARATELY: swapping λ_a↔λ_b onto the wrong W is a
/// SWAP-SYMMETRIC error. `h_raw = ½[(F_b S − λ_b⟨a|W_a|b⟩) + (F_a S − λ_a⟨a|W_b|b⟩)]`
/// is invariant under a↔b just as the correct expression is, so
/// `coupling_symmetric_under_swap` above CANNOT detect it, no matter how
/// non-degenerate its fixture. Predicted (numpy rebuild, 2026-10-01): that
/// mis-pairing (each λ with the other state's W, in its transition element
/// AND its population N) gives H_ab = −0.274837 for BOTH orderings, vs the
/// correct +0.218195; mis-pairing the transition elements only gives
/// +0.374200. (Under the superseded E-only raw element the correct value was
/// −2.236144 — the kernel and this reference both moved to the F form.) A symmetry
/// test alone is therefore not sufficient coverage for this expression; a value
/// check is required, and this is it.
///
/// The reference is built by an INDEPENDENT construction, not by re-running the
/// kernel: the generalized Slater–Condon transition-density form
/// `⟨a|Ô|b⟩ = S_ab · tr(Ô · C_b (C_aᵀ S C_b)⁻¹ C_aᵀ)` summed over spins, which
/// uses an explicit matrix inverse and never touches the SVD/reduced-overlap
/// path the kernel takes. Agreement to 1e-12 is then evidence about the
/// CONSTRUCTION, not merely reproducibility of one code path.
#[test]
fn coupling_pairs_each_lambda_with_its_own_weight() {
    use ferric_scf::cdft_coupling::{coupling_hab, DiabaticState};
    use ndarray_linalg::Inverse;
    let (s, c1, c2, w_a, w_b) = swap_fixture();
    let (e_a, l_a) = (-3.0_f64, 0.5_f64);
    let (e_b, l_b) = (-2.9_f64, 0.4_f64);

    // --- independent reference ---------------------------------------------
    // Per spin: occupied blocks A (from state a) and B (from state b).
    let blocks: [(Array2<f64>, Array2<f64>); 2] = [
        (
            c1.slice(ndarray::s![.., ..2]).to_owned(),
            c2.slice(ndarray::s![.., ..2]).to_owned(),
        ),
        (
            c1.slice(ndarray::s![.., ..1]).to_owned(),
            c2.slice(ndarray::s![.., ..1]).to_owned(),
        ),
    ];
    let det2 = |m: &Array2<f64>| -> f64 {
        match m.nrows() {
            1 => m[(0, 0)],
            2 => m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)],
            n => panic!("reference det not implemented for n = {n}"),
        }
    };
    let mut s_ab_ref = 1.0;
    for (a_blk, b_blk) in &blocks {
        s_ab_ref *= det2(&a_blk.t().dot(&s).dot(b_blk));
    }
    let elem = |op: &Array2<f64>| -> f64 {
        let mut tr = 0.0;
        for (a_blk, b_blk) in &blocks {
            let m = a_blk.t().dot(&s).dot(b_blk);
            let m_inv = m.inv().expect("reference: MO-overlap block is singular");
            tr += op.dot(b_blk).dot(&m_inv).dot(&a_blk.t()).diag().sum();
        }
        s_ab_ref * tr
    };
    let w_a_elem = elem(&w_a);
    let w_b_elem = elem(&w_b);
    // Each state's own population N_X = Σ_σ tr(C_Xσᵀ W_X C_Xσ), from the
    // occupied blocks directly (independent of the kernel's helper).
    let pop = |side: usize, w: &Array2<f64>| -> f64 {
        blocks
            .iter()
            .map(|blk| {
                let c = if side == 0 { &blk.0 } else { &blk.1 };
                c.t().dot(w).dot(c).diag().sum()
            })
            .sum()
    };
    let f_a = e_a + l_a * pop(0, &w_a);
    let f_b = e_b + l_b * pop(1, &w_b);
    let h_raw_ref = 0.5 * ((f_b * s_ab_ref - l_b * w_b_elem) + (f_a * s_ab_ref - l_a * w_a_elem));
    let h_ab_ref = (h_raw_ref - 0.5 * (e_a + e_b) * s_ab_ref) / (1.0 - s_ab_ref * s_ab_ref);

    // --- kernel -------------------------------------------------------------
    let a = DiabaticState {
        c_a: &c1,
        c_b: &c1,
        nocc_a: 2,
        nocc_b: 1,
        energy: e_a,
        lambda: l_a,
        w: &w_a,
    };
    let b = DiabaticState {
        c_a: &c2,
        c_b: &c2,
        nocc_a: 2,
        nocc_b: 1,
        energy: e_b,
        lambda: l_b,
        w: &w_b,
    };
    let got = coupling_hab(&a, &b, &s);

    assert!(
        (got.s_ab.abs() - s_ab_ref.abs()).abs() < 1e-12,
        "S_ab: kernel {} vs independent reference {}",
        got.s_ab,
        s_ab_ref
    );
    assert!(
        (got.h_ab - h_ab_ref).abs() < 1e-12,
        "H_ab: kernel {} vs independent reference {} (difference {:.3e}). \
         A mis-paired λ/W gives -0.274837 here.",
        got.h_ab,
        h_ab_ref,
        (got.h_ab - h_ab_ref).abs()
    );

    // Guard the reference itself: if the two weight elements were equal, this
    // test would pass under a λ/W mis-pairing and be inert.
    assert!(
        (l_a * w_a_elem - l_b * w_b_elem).abs() > 0.1,
        "λ_a⟨a|W_A|b⟩ == λ_b⟨a|W_B|b⟩ ({:.6} vs {:.6}); mis-pairing undetectable",
        l_a * w_a_elem,
        l_b * w_b_elem
    );
}

/// CONSTRAINT-OFFSET INVARIANCE. The constrained problem (W, N) and
/// (W − c·S, N − c·N_e) — S the AO overlap, i.e. the number operator — have the
/// same states, the same λ and the same E, so H_ab must not move. The F = E + λN
/// raw element is invariant exactly; the superseded E-only element
/// (`h_raw = ½[(E_b S − λ_b⟨a|W_b|b⟩) + …]`) moves from −2.236143 to −0.724343
/// at c = 0.37 on this fixture (numpy rebuild, 2026-10-01), so this test is red
/// against it. Also pins the corrected value +0.218195 (same rebuild).
#[test]
fn coupling_is_invariant_under_constraint_offset() {
    use ferric_scf::cdft_coupling::{coupling_hab, DiabaticState};
    let (s, c1, c2, w_a, w_b) = swap_fixture();
    let run = |wa: &Array2<f64>, wb: &Array2<f64>| -> f64 {
        let a = DiabaticState {
            c_a: &c1,
            c_b: &c1,
            nocc_a: 2,
            nocc_b: 1,
            energy: -3.0,
            lambda: 0.5,
            w: wa,
        };
        let b = DiabaticState {
            c_a: &c2,
            c_b: &c2,
            nocc_a: 2,
            nocc_b: 1,
            energy: -2.9,
            lambda: 0.4,
            w: wb,
        };
        coupling_hab(&a, &b, &s).h_ab
    };
    let h0 = run(&w_a, &w_b);
    let c = 0.37;
    let h1 = run(&(&w_a - &(&s * c)), &(&w_b - &(&s * c)));
    assert!(
        (h0 - 0.218_195_497_525_9).abs() < 1e-9,
        "H_ab = {h0:.12} vs the independent numpy value 0.2181954975 (the E-only \
         element gives -2.2361434)"
    );
    assert!(
        (h1 - h0).abs() < 1e-12,
        "H_ab moved under the empty offset W -> W - {c} S: {h0:.12} -> {h1:.12}"
    );
}

// End-to-end: two charge-constrained He₂⁺ states → coupling. He₂⁺ is 3
// electrons (doublet, charge +1); constrain the +1 hole onto atom 0 vs atom 1
// (Total population target = 1.0 e on the He bearing the hole, i.e. its 2
// electrons minus 1). The two states are symmetric, so |H_ab| = ½ the 2×2
// adiabatic gap, and |H_ab| decays with separation.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::cdft::{build_weight_matrix, Constraint, SpinChannel};
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::overlap;
use ferric_integrals::operator::Operator;
use ferric_scf::cdft_coupling::{coupling_hab, DiabaticState};
use ferric_scf::cdft_driver::solve_cdft_uhf;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;

/// Run both diabatic states at separation `r_ang` (Å) and return |H_ab|.
fn he2_plus_hab(r_ang: f64) -> (f64, f64, f64, f64) {
    let xyz = format!("2\nHe2+\nHe 0 0 0\nHe 0 0 {r_ang}\n");
    // He₂⁺: charge +1, doublet (multiplicity 2).
    let mol = Molecule::parse_xyz(&xyz, 1, 2).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let s = overlap(&prep);

    // Weight matrices (rebuild on the driver's grid: 99×302).
    let gcfg = AtomicGridConfig {
        n_radial: 99,
        n_angular: 302,
        ..Default::default()
    };
    let grid = build_atomic_grid(&mol, &gcfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&mol, &bs, &pts).unwrap();
    let w0 = build_weight_matrix(&mol, &grid, &chi, &[0]);
    let w1 = build_weight_matrix(&mol, &grid, &chi, &[1]);

    // Hole on atom 0: atom 0 has 1 electron (2 − 1). target N(atom0) = 1.0.
    // level_shift damps the open-shell SOMO oscillation seen on the symmetric
    // He₂⁺ doublet (unconstrained UHF needs it too); not a tolerance change.
    // The He₂⁺ charge response N(λ) is a near-step: N=1.5 at λ=0, then a flat
    // localized plateau (N≈1.01) for λ∈[0.1,2], then a cliff to an unphysical
    // over-localized state (E≈-1.9) past λ≈4. Target N=1.0 sits just below the
    // plateau, so the achievable localized hole is N≈1.01; tol=1e-2 matches the
    // physically flat response (∂N/∂λ≈1e-3) and lands the driver on the
    // localized plateau rather than walking λ over the cliff.
    let cfg0 = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            spin: SpinChannel::Total,
            target: 1.0,
        }],
        cdft_lambda_tol: 1e-2,
        // Pinned so "this path converges" is not also an assertion about
        // how many outer iterations a particular CPU needs -- CI and this
        // box differ by more than the old hardcoded 30. See `cdft_max_outer`.
        //
        // 40, not higher: every path in this repo that converges at all does
        // so in <= 23 outer iters locally, and the cap is also the price paid
        // by paths that NEVER converge (each wasted outer iteration runs a
        // full inner SCF). At 64 this file took 462 s; the non-convergent
        // R = 2.5/3.0 sigma points burn the whole cap before failing.
        cdft_max_outer: 40,
        fractional_occ: false,
        dft_grid: Some(gcfg.clone()),
        level_shift: 0.2,
        ..Default::default()
    };
    let cfg1 = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![1],
            spin: SpinChannel::Total,
            target: 1.0,
        }],
        cdft_lambda_tol: 1e-2,
        // Pinned so "this path converges" is not also an assertion about
        // how many outer iterations a particular CPU needs -- CI and this
        // box differ by more than the old hardcoded 30. See `cdft_max_outer`.
        //
        // 40, not higher: every path in this repo that converges at all does
        // so in <= 23 outer iters locally, and the cap is also the price paid
        // by paths that NEVER converge (each wasted outer iteration runs a
        // full inner SCF). At 64 this file took 462 s; the non-convergent
        // R = 2.5/3.0 sigma points burn the whole cap before failing.
        cdft_max_outer: 40,
        fractional_occ: false,
        dft_grid: Some(gcfg.clone()),
        level_shift: 0.2,
        ..Default::default()
    };
    let ra = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg0).unwrap();
    let rb = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg1).unwrap();

    // nocc from charge+mult: nelec = 3, 2S=1 → nocc_a=2, nocc_b=1.
    let (nocc_a, nocc_b) = (2usize, 1usize);
    let ca_b = ra.scf.mos_beta.as_ref().unwrap();
    let cb_b = rb.scf.mos_beta.as_ref().unwrap();
    let state_a = DiabaticState {
        c_a: &ra.scf.mos_alpha,
        c_b: ca_b,
        nocc_a,
        nocc_b,
        energy: ra.scf.energy,
        lambda: ra.lambdas[0],
        w: &w0,
    };
    let state_b = DiabaticState {
        c_a: &rb.scf.mos_alpha,
        c_b: cb_b,
        nocc_a,
        nocc_b,
        energy: rb.scf.energy,
        lambda: rb.lambdas[0],
        w: &w1,
    };
    let res = coupling_hab(&state_a, &state_b, &s);
    (res.h_ab.abs(), res.s_ab, res.e_a, res.e_b)
}

#[test]
fn he2_plus_coupling_is_finite_and_symmetric() {
    let (hab, s_ab, e_a, e_b) = he2_plus_hab(2.5);
    eprintln!(
        "He2+ @2.5Å: |H_ab|={hab:.6} Ha ({:.4} eV), S_ab={s_ab:.6}, E_a={e_a:.6}, E_b={e_b:.6}",
        hab * 27.211386
    );
    // Symmetric system: the two diabatic energies must match.
    assert!(
        (e_a - e_b).abs() < 1e-4,
        "diabatic energies differ: {e_a} vs {e_b}"
    );
    assert!(hab.is_finite() && hab > 0.0, "|H_ab| = {hab}");
    // Physical coupling for He2+ at 2.5 Å is on the order of 0.01–0.2 Ha.
    assert!(hab < 1.0, "|H_ab| implausibly large: {hab}");
}

#[test]
fn he2_plus_coupling_decays_with_distance() {
    // Distance set shifted to 2.5/3.0/3.5 Å (from the plan's 2.0/2.5/3.0): at
    // R=2.0 the localized-hole plateau is N≈1.034, farther from target 1.0 than
    // the flat-response tol (1e-2), so the cDFT outer loop walks λ over the
    // over-localization cliff (E→-1.9) and the inner SCF fails. For R≥2.5 the
    // plateau is within tol and the driver converges on the physical state.
    // This is still a valid exponential-decay test (plan Task 5 Step 3).
    let (h25, s25, _, _) = he2_plus_hab(2.5);
    let (h30, s30, _, _) = he2_plus_hab(3.0);
    let (h35, s35, _, _) = he2_plus_hab(3.5);
    eprintln!("|H_ab|: R=2.5 → {h25:.6} (S={s25:.4}), R=3.0 → {h30:.6} (S={s30:.4}), R=3.5 → {h35:.6} (S={s35:.4})");
    assert!(
        h25 > h30 && h30 > h35,
        "|H_ab| not strictly decreasing: {h25}, {h30}, {h35}"
    );
}

// ---------------------------------------------------------------------------
// Second ET system: HeNe⁺, an ASYMMETRIC (donor ≠ acceptor) hole transfer.
//
// He₂⁺ above is a SYMMETRIC self-exchange, so E_a = E_b and λ_a = λ_b, and a
// family of checks passes on that degenerate special case. HeNe⁺ (charge +1,
// doublet, 11 e, def2-SVP, nbf = 19) breaks those degeneracies at the same
// cost: E_a ≠ E_b, λ_a ≠ ±λ_b, and the two Becke cells differ in size.
//
// Ported from the stranded `test/cdft-et-second-system` (04a7ea88) and
// re-measured on main after #569 (an unapplied constraint now takes a Newton
// step) and the default-on `cdft_stability_descent`. Only what main did not
// already cover is kept here. Covered elsewhere and NOT repeated:
//   * the vacuous-constraint anchor:
//     `cdft_uhf::hene_vacuous_constraint_is_still_accepted`;
//   * the hole-on-Ne state reached through N(He) = 2 and through N(Ne) = 9:
//     `cdft_uhf::cliff_side_targets_with_tight_tolerance_are_accepted`;
//   * a loose tolerance returning the unconstrained state:
//     `cdft_uhf::loose_tolerance_still_applies_the_constraint`;
//   * state B's saddle and the descent that leaves it:
//     `cdft_constrained_stability.rs` and `cdft_state_selection.rs`;
//   * why |H_ab| is not assertable on HeNe⁺ (S_ab is one tiny β singular value):
//     `cdft_coupling_hene.rs`.
//
// No external H_ab exists for this system: the literature databases
// (HAB11/HAB7/HAB18/HAB79, Wu–Van Voorhis Zn₂⁺) are all symmetric and larger.

/// The one separation at which both HeNe⁺ diabats are constructible.
///
/// RE-MEASURED on main (2026-10-27), hcore start, `cdft_lambda_tol` 1e-5,
/// default 30-iteration outer cap, for the four constraints N(He) = 1, N(Ne) = 10,
/// N(He) = 2, N(Ne) = 9:
///
/// ```text
///   R = 2.0  all four converge (A: 4 outer iterations, B: 7)
///   R = 2.5  all four fail (outer loop does not converge), descent on or off
///   R = 3.5  N(He)=1 / N(Ne)=10 return the unconstrained state, which already
///            has the hole on He (λ ~ 1e-4); both hole-on-Ne targets fail
///   R = 4.0  all four fail
/// ```
///
/// With the MINAO start R = 2.5 does converge, but the two hole-on-Ne routes
/// disagree by 6.1e-4 Ha, so that point is not used either.
const HENE_R_ANG: f64 = 2.0;

/// HeNe⁺ at [`HENE_R_ANG`], with the driver's (99, 302) grid.
struct Hene {
    mol: Molecule,
    bs: basis::BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn hene() -> Hene {
    let xyz = format!("2\nHeNe+\nHe 0 0 0\nNe 0 0 {HENE_R_ANG}\n");
    let mol = Molecule::parse_xyz(&xyz, 1, 2).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Hene {
        mol,
        bs,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

/// Base config. `use_sad_guess: false` (hcore) because whether the MINAO start
/// converges the integer target is machine-dependent (see
/// `cdft_state_selection::the_default_minao_path_converges_to_the_same_state_as_hcore`);
/// MEASURED (2026-10-27, this box, not asserted): both starts give the same A
/// and B to 1e-8 Ha. `level_shift` 0.5
/// is the value every other HeNe⁺ test uses for its near-degenerate SOMO.
fn hene_base_cfg() -> RhfConfig {
    RhfConfig {
        level_shift: 0.5,
        max_iter: 400,
        use_sad_guess: false,
        fractional_occ: false,
        ..Default::default()
    }
}

/// Solve one HeNe⁺ diabat with the default driver (stability descent ON).
fn hene_diabat(sys: &Hene, fragment: usize, target: f64) -> ferric_scf::cdft_driver::CdftResult {
    hene_diabat_with(sys, fragment, target, true)
}

/// State A (N(He) = 1) and state B (N(He) = 2) from the default driver, solved
/// once and shared by the tests below, which only read them.
fn hene_default_pair() -> &'static (
    ferric_scf::cdft_driver::CdftResult,
    ferric_scf::cdft_driver::CdftResult,
) {
    static PAIR: std::sync::OnceLock<(
        ferric_scf::cdft_driver::CdftResult,
        ferric_scf::cdft_driver::CdftResult,
    )> = std::sync::OnceLock::new();
    PAIR.get_or_init(|| {
        let sys = hene();
        (hene_diabat(&sys, 0, 1.0), hene_diabat(&sys, 0, 2.0))
    })
}

/// [`hene_diabat`] with the stability descent chosen explicitly.
fn hene_diabat_with(
    sys: &Hene,
    fragment: usize,
    target: f64,
    descent: bool,
) -> ferric_scf::cdft_driver::CdftResult {
    let cfg = RhfConfig {
        cdft_stability_descent: descent,
        constraints: vec![Constraint {
            fragment: vec![fragment],
            spin: SpinChannel::Total,
            target,
        }],
        cdft_lambda_tol: 1e-5,
        dft_grid: Some(AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
            ..Default::default()
        }),
        ..hene_base_cfg()
    };
    let r = solve_cdft_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bs, &sys.bounds, &cfg)
        .unwrap_or_else(|e| panic!("HeNe+ N(atom {fragment}) = {target}: {e}"));
    assert!(r.scf.converged, "inner SCF did not converge");
    r
}

/// Both unconstrained HeNe⁺ UHF states: the ²Π saddle the hcore start reaches
/// and the σ minimum the MINAO start reaches (−130.50035 / −130.50534 Ha). A
/// diabat equal to either is the unconstrained solution, not a diabat.
fn hene_unconstrained_energies(sys: &Hene) -> [f64; 2] {
    [false, true].map(|sad| {
        let cfg = RhfConfig {
            use_sad_guess: sad,
            ..hene_base_cfg()
        };
        let scf =
            ferric_scf::uhf::solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
        assert!(
            scf.converged,
            "unconstrained UHF (sad = {sad}) did not converge"
        );
        scf.energy
    })
}

/// A constrained diabat must be a DIFFERENT state from the unconstrained
/// molecule, on three independent witnesses: λ ≠ 0 (the constraint potential
/// does work), more than one outer iteration (λ moved off its zero start), and
/// an energy different from every unconstrained state. The branch this was
/// ported from shipped the failure these catch: a tolerance looser than the
/// λ = 0 residual returned the unconstrained state labelled as a diabat.
fn assert_genuinely_constrained(
    name: &str,
    r: &ferric_scf::cdft_driver::CdftResult,
    e_free: &[f64],
) {
    assert!(
        r.lambdas[0].abs() > 1e-3,
        "{name}: λ = {} — the constraint did no work",
        r.lambdas[0]
    );
    assert!(
        r.outer_iters > 1,
        "{name}: the λ loop never moved off λ = 0"
    );
    for &e in e_free {
        assert!(
            (r.scf.energy - e).abs() > 1e-3,
            "{name}: E = {} is the unconstrained state {e}",
            r.scf.energy
        );
    }
}

/// The constraint machinery on a HETERONUCLEAR fragment pair.
///
/// He₂⁺ cannot test this: its two Becke cells are identical by symmetry, so a
/// weight operator mishandling differing atomic sizes would still look right.
///
/// MEASURED (main, 2026-10-27): A (hole on He) N(He) = 0.99999321,
/// λ = +1.631409, E = −130.36185950; B (hole on Ne) N(He) = 1.99999993,
/// λ = −2.439010, E = −130.42670717. Unconstrained: −130.50034664 (²Π, hcore)
/// and −130.50534054 (σ, MINAO).
#[test]
fn hene_plus_constrained_states_carry_the_requested_charge() {
    let sys = hene();
    let e_free = hene_unconstrained_energies(&sys);
    let (a, b) = hene_default_pair();
    eprintln!(
        "HeNe+ @{HENE_R_ANG} Å: E_free = {e_free:?} | A λ = {:+.6} E = {:.8} N(He) = {:.8} \
         | B λ = {:+.6} E = {:.8} N(He) = {:.8}",
        a.lambdas[0], a.scf.energy, a.populations[0], b.lambdas[0], b.scf.energy, b.populations[0]
    );
    assert_genuinely_constrained("A (hole on He)", a, &e_free);
    assert_genuinely_constrained("B (hole on Ne)", b, &e_free);
    assert!(
        (a.populations[0] - 1.0).abs() < 1e-4,
        "A: N(He) = {}",
        a.populations[0]
    );
    assert!(
        (b.populations[0] - 2.0).abs() < 1e-4,
        "B: N(He) = {}",
        b.populations[0]
    );
}

/// INDEPENDENT CONSTRUCTION for the hole-on-He state: N(He) = 1 and
/// N(Ne) = 10 are the same state of the 11-electron cation reached through
/// different weight operators, residuals and λ. Agreement separates signal
/// from a systematic error in one weight operator; consistency across R or
/// basis would not.
///
/// MEASURED (main, 2026-10-27): λ = +1.631409 via He and −1.631220 via Ne,
/// energies 3.861e-6 Ha apart (both residuals are inside the 1e-5 population
/// tolerance, so the routes agree to ~λ·1e-5, not to machine precision). The
/// 1e-4 bar is 650× tighter than the 0.0648 Ha gap to the other diabat.
#[test]
fn hene_plus_hole_on_he_is_reachable_through_either_fragment() {
    let sys = hene();
    let via_he = &hene_default_pair().0;
    let via_ne = hene_diabat(&sys, 1, 10.0);
    let d = (via_he.scf.energy - via_ne.scf.energy).abs();
    eprintln!(
        "hole on He: via N(He)=1 λ = {:+.6} E = {:.10}; via N(Ne)=10 λ = {:+.6} E = {:.10}; \
         ΔE = {d:.3e}",
        via_he.lambdas[0], via_he.scf.energy, via_ne.lambdas[0], via_ne.scf.energy
    );
    assert!(
        d < 1e-4,
        "the two routes to one state disagree by {d:.3e} Ha"
    );
    assert!(
        via_he.lambdas[0] * via_ne.lambdas[0] < 0.0,
        "complementary fragments must carry opposite-signed λ: {} and {}",
        via_he.lambdas[0],
        via_ne.lambdas[0]
    );
    assert!(
        (via_he.lambdas[0].abs() - via_ne.lambdas[0].abs()).abs() < 1e-2,
        "|λ| should agree between the routes: {} and {}",
        via_he.lambdas[0],
        via_ne.lambdas[0]
    );
}

/// THE ASYMMETRY TEST: the hole is more stable on the lower-IP atom (He
/// 24.587 eV, Ne 21.565 eV), so E(hole on Ne) < E(hole on He).
///
/// CORRECTED VALUE. The source branch measured this gap as 1.0975 eV and
/// asserted only its sign. That number is REFUTED: its hole-on-Ne state
/// (−130.40219057, λ −2.753705) is a saddle of the λ-augmented manifold
/// (`cdft_constrained_stability.rs`), and the default driver now descends from
/// it. RE-MEASURED on main with the default driver:
///
/// ```text
///   E(hole on He) = -130.36185950   λ = +1.631409
///   E(hole on Ne) = -130.42670717   λ = -2.439010
///   gap           =    0.06484768 Ha = 1.7646 eV
/// ```
///
/// The hole-on-Ne state is the one NWChem finds when its Becke partition is
/// given ferric's He/Ne radius ratio (0.667; NWChem's own is 0.700):
/// −130.426701846, λ −2.439043 (`validation_cdft_state.rs`). The earlier
/// "0.563 eV above NWChem's LOW member" was that partition difference, not a
/// missing state. MEASURED by mutation (2026-10-27): patching NWChem's radii
/// (He 0.35, Ne 0.50 Å) into `becke.rs` moves the hole-on-Ne state to
/// −130.447403725, NWChem's NATIVE lower state to 2.3e-6 Ha, and fails the pin
/// below. The NWChem values are the checked-in
/// `testdata/reference/validation/cdft_state/hene_def2-svp.json`
/// (`states/low/ferric_partition/matched/energy_extrapolated` −130.426701846,
/// `states/low/native/matched/energy_extrapolated` −130.447401411). The
/// hole-on-He state has no external reference.
///
/// The NWChem pin below is a STATE-IDENTITY check (which of the two states at
/// N_He = 2 the driver returns; they are 0.0245 Ha apart), not the
/// quadrature-corrected energy comparison, which lives in
/// `validation_cdft_state.rs`. Because dE/dN = −λ, a population residual δ moves
/// E by |λ|·δ, so the pin first requires |N − 2| < 1e-6 (≤ 2.4e-6 Ha); a
/// residual merely inside `cdft_lambda_tol` (1e-5, up to 2.4e-5 Ha) could break
/// the 2e-5 bar on its own.
///
/// The gap is ferric-partition specific: a different radius ratio constrains a
/// different population. Its asymptote at this level is the UHF ΔIP 3.6305 eV
/// (`hene_atomic_ip_anchor.rs`), not the experimental 3.022 eV.
#[test]
fn hene_plus_hole_prefers_the_lower_ip_atom() {
    /// NWChem 7.2.2, Becke partition relabelled to ferric's radius ratio.
    const NWCHEM_MATCHED_HOLE_ON_NE: f64 = -130.426_701_846;
    let (hole_he, hole_ne) = hene_default_pair();
    let gap = hole_he.scf.energy - hole_ne.scf.energy;
    let ev = 27.211_386_245_988;
    eprintln!(
        "HeNe+ @{HENE_R_ANG} Å: E(hole on He) = {:.8} E(hole on Ne) = {:.8} gap = {gap:.8} Ha \
         = {:.4} eV",
        hole_he.scf.energy,
        hole_ne.scf.energy,
        gap * ev
    );
    assert!(
        gap > 0.0,
        "the hole on Ne (lower IP) must be the more stable diabat: gap = {gap}"
    );
    // The partition-matched NWChem state. The refuted saddle is 0.0245 Ha away.
    // State identity only; see the doc comment for why the population is
    // checked first.
    assert!(
        (hole_ne.populations[0] - 2.0).abs() < 1e-6,
        "hole-on-Ne population residual {:.3e} is too large for the 2e-5 Ha pin \
         (dE/dN = −λ = {:+.6})",
        hole_ne.populations[0] - 2.0,
        -hole_ne.lambdas[0]
    );
    assert!(
        (hole_ne.scf.energy - NWCHEM_MATCHED_HOLE_ON_NE).abs() < 2e-5,
        "hole-on-Ne E = {:.9} vs NWChem (matched partition) {NWCHEM_MATCHED_HOLE_ON_NE}",
        hole_ne.scf.energy
    );
    // Pinned so a drift is visible; 1e-4 Ha is 245× smaller than the step to
    // the refuted 1.0975 eV value.
    assert!(
        (gap - 0.064_847_68).abs() < 1e-4,
        "gap {gap:.8} Ha ({:.4} eV); measured 0.06484768 Ha = 1.7646 eV",
        gap * ev
    );
}

/// The Wu–Van Voorhis coupling is invariant under W → W − c·S on a REAL
/// asymmetric pair of converged constrained determinants.
///
/// Shifting each weight operator by a multiple of the overlap leaves the
/// constrained problem unchanged (N and the target move together), so H_ab
/// must not move. The old kernel used E·S where F = E + λN belongs (#247) and
/// was not invariant; on He₂⁺ that bug was exactly 3×, and the synthetic
/// `coupling_is_invariant_under_constraint_offset` covers arbitrary numbers.
/// Here E_a ≠ E_b and λ_a ≠ −λ_b. The second condition matters: the E·S bug
/// shifts H_raw by ½·c·N_e·S·(λ_a + λ_b), which vanishes when λ_a = −λ_b.
///
/// WHICH PAIR, and why not the default one. The default driver's pair (state A
/// and the descended hole-on-Ne state) has S_ab ≈ 7e-8 on this box but 8.7e-9
/// under `OPENBLAS_CORETYPE=Prescott`, which is below the kernel's near-zero
/// singular-value threshold `S_TOL` = 1e-8. In that branch the kernel drops
/// O(σ) terms, and MEASURED under Prescott the offset then moved H_ab by
/// 1.3e-8 against an H_ab of 1.2e-8. So the identity only holds above S_TOL,
/// and that pair straddles it depending on the BLAS kernel. Below S_TOL the
/// known non-invariance is ½·c·(N_e − 1)·S_ab·(λ_a + λ_b), here
/// ½·0.37·10·8.66e-9·0.808 = 1.3e-8, as measured; a separate change is to
/// replace S_TOL with an exact cofactor sum. This test uses the
/// descent-OFF hole-on-Ne state (λ −2.753705, the saddle pinned in
/// `cdft_constrained_stability.rs`) instead: it is a converged determinant of
/// the same constraint, S_ab ≈ 2.9e-2, and the identity is algebraic, so the
/// state's stability does not matter. No |H_ab| magnitude is asserted.
///
/// MEASURED (main, 2026-10-27): H_ab = −3.036e-2 Ha, S_ab = 2.946e-2.
#[test]
fn hene_plus_coupling_is_invariant_under_constraint_offset() {
    let sys = hene();
    let a = &hene_default_pair().0;
    let b = hene_diabat_with(&sys, 0, 2.0, false);
    let s = overlap(&sys.prep);
    let hab = |wa: &Array2<f64>, wb: &Array2<f64>| {
        let sa = DiabaticState {
            c_a: &a.scf.mos_alpha,
            c_b: a.scf.mos_beta.as_ref().unwrap(),
            nocc_a: 6,
            nocc_b: 5,
            energy: a.scf.energy,
            lambda: a.lambdas[0],
            w: wa,
        };
        let sb = DiabaticState {
            c_a: &b.scf.mos_alpha,
            c_b: b.scf.mos_beta.as_ref().unwrap(),
            nocc_a: 6,
            nocc_b: 5,
            energy: b.scf.energy,
            lambda: b.lambdas[0],
            w: wb,
        };
        coupling_hab(&sa, &sb, &s)
    };
    let (wa, wb) = (&a.weight_matrices[0], &b.weight_matrices[0]);
    let r0 = hab(wa, wb);
    let c = 0.37;
    let r1 = hab(&(wa - &(&s * c)), &(wb - &(&s * c)));
    // What the E·S bug would move H_ab by, for scale: ½·c·N_e·S·(λ_a + λ_b).
    let bug_shift = 0.5 * c * 11.0 * r0.s_ab * (a.lambdas[0] + b.lambdas[0]);
    eprintln!(
        "HeNe+ H_ab = {:.6e} S_ab = {:.6e} (λ_a {:+.6}, λ_b {:+.6}); offset moves it by \
         {:.3e}; the E·S bug would move it by ~{bug_shift:.3e}",
        r0.h_ab,
        r0.s_ab,
        a.lambdas[0],
        b.lambdas[0],
        (r1.h_ab - r0.h_ab).abs()
    );
    // The pair must be well above S_TOL and genuinely asymmetric in λ, or the
    // identity cannot tell the bug from the fix.
    assert!(r0.s_ab.abs() > 1e-4, "S_ab = {:.3e}", r0.s_ab);
    assert!(
        bug_shift.abs() > 1e-3,
        "the E·S bug would only move H_ab by {bug_shift:.3e}; the check is blind"
    );
    assert!(
        (r1.h_ab - r0.h_ab).abs() < 1e-10,
        "H_ab moved under W -> W - {c} S: {:.10e} -> {:.10e} (bug scale {bug_shift:.3e})",
        r0.h_ab,
        r1.h_ab
    );
}
