//! Exactness anchors for the AURORA-SCF linear algebra.
//!
//! Every approximation in AURORA has a trivial limit where it does nothing, and
//! per this repo's experimental protocol those limits are pinned BEFORE any
//! measurement is taken. This file holds the limits that need no integrals:
//!
//! * the thin-SVD geodesic must equal the dense matrix exponential (not merely
//!   be orthogonal — orthogonality alone would pass for the wrong rotation);
//! * vector transport must be the adjoint PROJECTION it claims to be (it is not
//!   an isometry — see the test, which pins the norm that is discarded);
//! * PCG must solve an SPD system it is allowed enough iterations for;
//! * **the auxiliary response must reproduce the finite-difference derivative of
//!   the gradient of the Fock operator built from the SAME three-index tensor.**
//!
//! That last one is the load-bearing anchor. It is an INDEPENDENT-CONSTRUCTION
//! check in the sense this repo requires: the analytic operator (an explicit
//! contraction, Eqs. S7-S8) and the finite difference (a numerical derivative of
//! a separately written Fock build) share no code, so an error in either factor,
//! sign, or index pairing separates them. Agreement across systems would not do
//! this — a construction bug is deterministic and reproduces perfectly.

use ndarray::{s, Array1, Array2, Array3};

use ferric_scf::aurora::{geodesic_rotation, transport_tangent};

/// Deterministic xorshift64 — the crate convention, avoiding a `rand` dep.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in (-1, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Dense matrix exponential by scaling-and-squaring with a Taylor series.
///
/// Deliberately written as a plain, obviously-correct reference: the point of
/// the anchor is that it shares NO code with the thin-SVD closed form.
fn dense_expm(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let norm = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    // Scale so the series converges fast, then square back up.
    let mut squarings = 0u32;
    let mut scale = 1.0f64;
    while norm * scale > 0.05 {
        scale *= 0.5;
        squarings += 1;
    }
    let as_ = a * scale;
    let mut term = Array2::<f64>::eye(n);
    let mut acc = Array2::<f64>::eye(n);
    for k in 1..40 {
        term = term.dot(&as_) / (k as f64);
        acc += &term;
        if term.iter().fold(0.0f64, |m, v| m.max(v.abs())) < 1e-18 {
            break;
        }
    }
    for _ in 0..squarings {
        acc = acc.dot(&acc);
    }
    acc
}

fn kappa_of(x: &Array2<f64>) -> Array2<f64> {
    let nvir = x.nrows();
    let nocc = x.ncols();
    let nmo = nocc + nvir;
    let mut k = Array2::<f64>::zeros((nmo, nmo));
    for a in 0..nvir {
        for i in 0..nocc {
            k[(nocc + a, i)] = x[(a, i)];
            k[(i, nocc + a)] = -x[(a, i)];
        }
    }
    k
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// ANCHOR: the thin-SVD closed form IS the matrix exponential.
///
/// Checked against an independently written dense `expm`, at several
/// (nvir, nocc) shapes including the non-square ones where a transposed thin
/// SVD would silently give an orthogonal-but-wrong answer.
#[test]
fn geodesic_thin_svd_equals_dense_matrix_exponential() {
    let mut rng = Xorshift64::new(0x5eed_1234);
    for &(nvir, nocc) in &[(5usize, 3usize), (2, 5), (4, 4), (7, 1), (1, 6)] {
        let mut x = Array2::<f64>::zeros((nvir, nocc));
        for v in x.iter_mut() {
            *v = 0.2 * rng.next_f64();
        }
        let u = geodesic_rotation(x.view()).unwrap();
        let reference = dense_expm(&kappa_of(&x));
        let d = max_abs_diff(&u, &reference);
        eprintln!("geodesic vs dense expm  nvir={nvir} nocc={nocc}: max|Δ| = {d:.3e}");
        assert!(
            d < 1e-12,
            "thin-SVD geodesic must equal exp[κ(X)] for ({nvir},{nocc}): max|Δ| = {d:.3e}"
        );

        // Orthonormality is necessary but NOT sufficient — asserted separately
        // so a failure distinguishes "not a rotation" from "the wrong rotation".
        let nmo = nocc + nvir;
        let ortho = max_abs_diff(&u.t().dot(&u), &Array2::eye(nmo));
        assert!(ortho < 1e-13, "rotation must be orthogonal: {ortho:.3e}");
    }
}

/// ANCHOR (trivial limit): a zero step is the identity rotation, exactly.
#[test]
fn geodesic_of_zero_step_is_exactly_the_identity() {
    let x = Array2::<f64>::zeros((4, 3));
    let u = geodesic_rotation(x.view()).unwrap();
    let eye = Array2::<f64>::eye(7);
    // Bit-identical, not merely close: the zero-norm early return must not go
    // anywhere near LAPACK.
    for (a, b) in u.iter().zip(eye.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "zero step must give exactly I");
    }
}

/// ANCHOR (trivial limit): transporting through the identity rotation is the
/// identity map on the tangent vector.
#[test]
fn transport_through_identity_is_the_identity_map() {
    let mut rng = Xorshift64::new(0xabc_def);
    let (nvir, nocc) = (5, 3);
    let mut v = Array2::<f64>::zeros((nvir, nocc));
    for e in v.iter_mut() {
        *e = rng.next_f64();
    }
    let u = Array2::<f64>::eye(nvir + nocc);
    let out = transport_tangent(v.view(), u.view(), nocc);
    let d = max_abs_diff(&out, &v);
    assert!(d < 1e-15, "identity transport must be exact: {d:.3e}");
}

/// Transport is an ADJOINT PROJECTION, not an isometry — and this test pins
/// exactly that, because getting it wrong in the other direction is the easy
/// mistake.
///
/// The map is `v ↦ [Uᵀ κ(v) U]_ov`. Conjugation by the orthogonal `U` preserves
/// the Frobenius norm of the FULL `κ` exactly, but the conjugated matrix
/// generally has nonzero occupied-occupied and virtual-virtual blocks. Keeping
/// only the ov block therefore DISCARDS norm: the map is a projection back onto
/// the tangent space, which is what both the paper (`Πₖ`, `Πₖ*`) and the
/// reference implementation ("adjoint/project") describe.
///
/// This was checked against the reference implementation after two initially
/// written isometry/round-trip assertions failed; the assertions were wrong, not
/// the code. Recorded here so the next reader does not "fix" the transport.
#[test]
fn transport_is_an_adjoint_projection_that_conserves_the_full_kappa_norm() {
    let mut rng = Xorshift64::new(0x1111_2222);
    let (nvir, nocc) = (6usize, 4usize);
    let nmo = nvir + nocc;
    let mut step = Array2::<f64>::zeros((nvir, nocc));
    for e in step.iter_mut() {
        *e = 0.15 * rng.next_f64();
    }
    let u = geodesic_rotation(step.view()).unwrap();

    let mut v = Array2::<f64>::zeros((nvir, nocc));
    for e in v.iter_mut() {
        *e = rng.next_f64();
    }

    // The FULL conjugation is an isometry, exactly.
    let kappa = kappa_of(&v);
    let full = u.t().dot(&kappa).dot(&u);
    let n_before: f64 = kappa.iter().map(|x| x * x).sum();
    let n_after: f64 = full.iter().map(|x| x * x).sum();
    eprintln!("full κ Frobenius²: before = {n_before:.12}, after = {n_after:.12}");
    assert!(
        (n_before - n_after).abs() < 1e-11 * n_before,
        "conjugation by an orthogonal U must preserve ‖κ‖: {n_before:.12} vs {n_after:.12}"
    );

    // The ov restriction is what `transport_tangent` returns, and it must agree
    // with reading that block off the full conjugation.
    let projected = transport_tangent(v.view(), u.view(), nocc);
    let block = full.slice(s![nocc.., ..nocc]).to_owned();
    let d = max_abs_diff(&projected, &block);
    assert!(
        d < 1e-14,
        "transport must equal the ov block of Uᵀ κ U: max|Δ| = {d:.3e}"
    );

    // And it genuinely LOSES norm, because the oo/vv blocks are populated. If a
    // future change made transport norm-preserving, that would mean the off-
    // diagonal blocks vanished — a different (wrong) map — so assert the loss.
    let oo: f64 = full.slice(s![..nocc, ..nocc]).iter().map(|x| x * x).sum();
    let vv: f64 = full.slice(s![nocc.., nocc..]).iter().map(|x| x * x).sum();
    eprintln!("discarded blocks: ‖oo‖² = {oo:.6}, ‖vv‖² = {vv:.6}");
    assert!(
        oo + vv > 1e-6,
        "for a nontrivial rotation the discarded blocks must be nonzero \
         (oo {oo:.3e}, vv {vv:.3e}) — transport is a projection, not an isometry"
    );
    let _ = nmo;
}

/// Transport through the identity rotation round-trips exactly — the trivial
/// limit of the projection, where nothing is discarded.
#[test]
fn transport_round_trips_exactly_in_the_no_rotation_limit() {
    let mut rng = Xorshift64::new(0x9999_7777);
    let (nvir, nocc) = (5usize, 4usize);
    let u = Array2::<f64>::eye(nvir + nocc);

    let mut v = Array2::<f64>::zeros((nvir, nocc));
    for e in v.iter_mut() {
        *e = rng.next_f64();
    }
    let there = transport_tangent(v.view(), u.view(), nocc);
    let back = transport_tangent(there.view(), u.t(), nocc);
    let d = max_abs_diff(&back, &v);
    eprintln!("identity-rotation round trip: max|Δ| = {d:.3e}");
    assert!(
        d < 1e-14,
        "identity transport must round trip exactly: {d:.3e}"
    );
}

/// As the rotation shrinks, transport approaches the identity map at first
/// order — the continuity property the L-BFGS secants actually rely on.
#[test]
fn transport_approaches_the_identity_as_the_rotation_shrinks() {
    let mut rng = Xorshift64::new(0x2468_1357);
    let (nvir, nocc) = (5usize, 3usize);
    let mut dir = Array2::<f64>::zeros((nvir, nocc));
    for e in dir.iter_mut() {
        *e = rng.next_f64();
    }
    let mut v = Array2::<f64>::zeros((nvir, nocc));
    for e in v.iter_mut() {
        *e = rng.next_f64();
    }

    let mut last = f64::INFINITY;
    for &scale in &[1e-1, 1e-2, 1e-3] {
        let u = geodesic_rotation((scale * &dir).view()).unwrap();
        let t = transport_tangent(v.view(), u.view(), nocc);
        let d = max_abs_diff(&t, &v);
        eprintln!("rotation scale {scale:.0e}: ‖T(v) - v‖∞ = {d:.3e}");
        assert!(
            d < last,
            "transport must approach the identity as the step shrinks"
        );
        last = d;
    }
    assert!(
        last < 1e-5,
        "at a 1e-3 rotation transport should be within 1e-5 of the identity: {last:.3e}"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// The load-bearing anchor: the auxiliary response IS the derivative of the
// gradient, for a three-index tensor we control exactly.
// ───────────────────────────────────────────────────────────────────────────

/// Build a synthetic symmetric three-index tensor `B[P,p,q] = B[P,q,p]`.
fn synthetic_b(naux: usize, nmo: usize, rng: &mut Xorshift64) -> Array3<f64> {
    let mut b = Array3::<f64>::zeros((naux, nmo, nmo));
    for p in 0..naux {
        for i in 0..nmo {
            for j in 0..=i {
                let v = 0.08 * rng.next_f64();
                b[(p, i, j)] = v;
                b[(p, j, i)] = v;
            }
        }
    }
    b
}

/// The closed-shell Fock matrix and scaled gradient for a model whose two-electron
/// integrals are exactly `(pq|rs) = Σ_P B[P,p,q] B[P,r,s]`.
///
/// This is an INDEPENDENT construction from `AuxCurvature::response`: it forms
/// J and K explicitly from the density and differentiates numerically.
fn model_fock_and_gradient(
    b: &Array3<f64>,
    hcore: &Array2<f64>,
    c: &Array2<f64>,
    nocc: usize,
) -> (Array2<f64>, Array2<f64>) {
    let naux = b.shape()[0];
    let nmo = hcore.nrows();
    let co = c.slice(s![.., ..nocc]);
    let density = 2.0 * co.dot(&co.t());

    let mut coulomb = Array2::<f64>::zeros((nmo, nmo));
    let mut exchange = Array2::<f64>::zeros((nmo, nmo));
    for p in 0..naux {
        let bp = b.slice(s![p, .., ..]);
        // J_pq = Σ_rs B_pq B_rs D_rs
        let tr: f64 = bp.iter().zip(density.iter()).map(|(x, y)| x * y).sum();
        coulomb.scaled_add(tr, &bp);
        // K_pq = Σ_rs B_pr D_rs B_sq
        exchange += &bp.dot(&density).dot(&bp);
    }
    let fock = hcore + &coulomb - &(0.5 * &exchange);
    let fock_mo = c.t().dot(&fock).dot(c);
    // Scaled gradient g̃ = 2 F_vo (the convention AURORA uses).
    let grad = 2.0 * fock_mo.slice(s![nocc.., ..nocc]).to_owned();
    (fock_mo, grad)
}

/// ANCHOR: the analytic auxiliary curvature equals the numerical derivative of
/// the gradient — every factor, sign and index pairing of Eqs. (S7)-(S9).
///
/// Artifact hypothesis, stated before the measurement: if the response were
/// implemented with a wrong prefactor (say 4 instead of 8), a dropped exchange
/// term, or a transposed contraction, the relative error would be O(1), NOT
/// O(h). If it is right, the error tracks the central-difference truncation
/// error and shrinks as `h` shrinks until roundoff takes over. Those two
/// outcomes are distinguishable, so the experiment is admissible.
#[test]
fn auxiliary_response_matches_finite_difference_of_the_gradient() {
    let mut rng = Xorshift64::new(0x31_31_31);
    let (nocc, nvir, naux) = (3usize, 4usize, 8usize);
    let nmo = nocc + nvir;

    let b = synthetic_b(naux, nmo, &mut rng);
    let mut hcore = Array2::<f64>::zeros((nmo, nmo));
    for k in 0..nmo {
        hcore[(k, k)] = -1.5 + 3.0 * (k as f64) / ((nmo - 1) as f64);
    }
    for i in 0..nmo {
        for j in 0..=i {
            let v = 0.02 * rng.next_f64();
            hcore[(i, j)] += v;
            hcore[(j, i)] += v;
        }
    }

    let c = Array2::<f64>::eye(nmo);
    let (fock_mo, _grad0) = model_fock_and_gradient(&b, &hcore, &c, nocc);

    let mut step = Array2::<f64>::zeros((nvir, nocc));
    for e in step.iter_mut() {
        *e = 0.1 * rng.next_f64();
    }

    // Analytic: ℬ[X] = 2(F_vv X - X F_oo) + R_aux(X), Eq. (S9).
    let f_vv = fock_mo.slice(s![nocc.., nocc..]).to_owned();
    let f_oo = fock_mo.slice(s![..nocc, ..nocc]).to_owned();
    let aux = ferric_scf::aurora::AuxCurvature::from_mo_tensor_for_test(
        b.view(),
        nocc,
        nvir,
        1.0, // full exact exchange
        0.05,
    );
    let analytic = 2.0 * (f_vv.dot(&step) - step.dot(&f_oo)) + &aux.response(step.view());

    // Numerical: central difference of the gradient along the geodesic.
    let mut prev_rel = f64::INFINITY;
    let mut seen_small = false;
    for &h in &[1e-4, 1e-5, 1e-6] {
        let up = geodesic_rotation((h * &step).view()).unwrap();
        let dn = geodesic_rotation((-h * &step).view()).unwrap();
        let (_f1, g_up) = model_fock_and_gradient(&b, &hcore, &c.dot(&up), nocc);
        let (_f2, g_dn) = model_fock_and_gradient(&b, &hcore, &c.dot(&dn), nocc);
        // The gradients live in rotated frames; transport back before differencing.
        let g_up_t = transport_tangent(g_up.view(), up.t(), nocc);
        let g_dn_t = transport_tangent(g_dn.view(), dn.t(), nocc);
        let fd = (&g_up_t - &g_dn_t) / (2.0 * h);

        let num = (&fd - &analytic).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let den = analytic.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let rel = num / den;
        eprintln!("aux response vs central difference  h={h:.0e}: rel err = {rel:.3e}");
        if rel < 1e-6 {
            seen_small = true;
        }
        prev_rel = prev_rel.min(rel);
    }
    assert!(
        seen_small,
        "auxiliary response must reproduce the gradient derivative: best rel err = {prev_rel:.3e}"
    );
}

/// ANCHOR (trivial limit): with the exchange fraction set to zero, the response
/// must be EXACTLY the Coulomb term and nothing else.
#[test]
fn zero_exchange_fraction_leaves_only_the_coulomb_response() {
    let mut rng = Xorshift64::new(0x4242);
    let (nocc, nvir, naux) = (3usize, 4usize, 6usize);
    let nmo = nocc + nvir;
    let b = synthetic_b(naux, nmo, &mut rng);

    let aux0 =
        ferric_scf::aurora::AuxCurvature::from_mo_tensor_for_test(b.view(), nocc, nvir, 0.0, 0.05);
    let mut x = Array2::<f64>::zeros((nvir, nocc));
    for e in x.iter_mut() {
        *e = rng.next_f64();
    }
    let got = aux0.response(x.view());

    // Hand-rolled Coulomb-only reference, Eq. (S7) first line.
    let mut want = Array2::<f64>::zeros((nvir, nocc));
    for p in 0..naux {
        let mut scalar = 0.0;
        for a in 0..nvir {
            for i in 0..nocc {
                scalar += b[(p, nocc + a, i)] * x[(a, i)];
            }
        }
        for a in 0..nvir {
            for i in 0..nocc {
                want[(a, i)] += 8.0 * b[(p, nocc + a, i)] * scalar;
            }
        }
    }
    let d = max_abs_diff(&got, &want);
    eprintln!("a_x=0 Coulomb-only response: max|Δ| = {d:.3e}");
    assert!(
        d < 1e-12,
        "a_x=0 must leave exactly the Coulomb term: {d:.3e}"
    );
}

/// ANCHOR: the assembled base operator `ℬₖ` (Eq. S9) — the exact expression the
/// inner PCG solve applies — matches the finite difference.
///
/// The earlier response test covers only `R_aux`; this one covers the full
/// operator INCLUDING the `2(F_vv X - X F_oo)` Fock-block term. That distinction
/// is not cosmetic: a mutation ledger run found that flipping the sign of that
/// term was caught by NO test, because the assembled operator had two
/// independent copies and only the uncovered one was mutated. The copies are now
/// unified behind `base_apply_ref`, and this test pins it.
#[test]
fn assembled_base_operator_matches_finite_difference_of_the_gradient() {
    let mut rng = Xorshift64::new(0x0ba5_e017);
    let (nocc, nvir, naux) = (3usize, 4usize, 8usize);
    let nmo = nocc + nvir;

    let b = synthetic_b(naux, nmo, &mut rng);
    let mut hcore = Array2::<f64>::zeros((nmo, nmo));
    for k in 0..nmo {
        hcore[(k, k)] = -1.5 + 3.0 * (k as f64) / ((nmo - 1) as f64);
    }
    for i in 0..nmo {
        for j in 0..=i {
            let v = 0.02 * rng.next_f64();
            hcore[(i, j)] += v;
            hcore[(j, i)] += v;
        }
    }

    let c = Array2::<f64>::eye(nmo);
    let (fock_mo, _g0) = model_fock_and_gradient(&b, &hcore, &c, nocc);
    let f_vv = fock_mo.slice(s![nocc.., nocc..]).to_owned();
    let f_oo = fock_mo.slice(s![..nocc, ..nocc]).to_owned();

    let mut step = Array2::<f64>::zeros((nvir, nocc));
    for e in step.iter_mut() {
        *e = 0.1 * rng.next_f64();
    }

    let aux =
        ferric_scf::aurora::AuxCurvature::from_mo_tensor_for_test(b.view(), nocc, nvir, 1.0, 0.05);
    // Zero shift: the unshifted base operator is what must equal the Hessian.
    let analytic = aux.base_apply_ref(step.view(), &f_vv, &f_oo, 0.0);

    let mut best = f64::INFINITY;
    for &h in &[1e-4, 1e-5, 1e-6] {
        let up = geodesic_rotation((h * &step).view()).unwrap();
        let dn = geodesic_rotation((-h * &step).view()).unwrap();
        let (_f1, g_up) = model_fock_and_gradient(&b, &hcore, &c.dot(&up), nocc);
        let (_f2, g_dn) = model_fock_and_gradient(&b, &hcore, &c.dot(&dn), nocc);
        let g_up_t = transport_tangent(g_up.view(), up.t(), nocc);
        let g_dn_t = transport_tangent(g_dn.view(), dn.t(), nocc);
        let fd = (&g_up_t - &g_dn_t) / (2.0 * h);
        let num = (&fd - &analytic).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let den = analytic.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let rel = num / den;
        eprintln!("assembled ℬ vs central difference  h={h:.0e}: rel err = {rel:.3e}");
        best = best.min(rel);
    }
    assert!(
        best < 1e-6,
        "the assembled base operator must equal the Hessian action: rel err = {best:.3e}"
    );
}

/// The shifted operator must actually differ from the unshifted one, and the
/// shift must enter as `+ λ·D·x`.
///
/// Without this, a `shift` argument that was silently ignored would be invisible
/// — the regularization ladder would appear to work while doing nothing.
#[test]
fn the_adaptive_shift_actually_shifts_the_operator() {
    let mut rng = Xorshift64::new(0xdd_11_ee);
    let (nocc, nvir, naux) = (3usize, 4usize, 6usize);
    let nmo = nocc + nvir;
    let b = synthetic_b(naux, nmo, &mut rng);
    let aux =
        ferric_scf::aurora::AuxCurvature::from_mo_tensor_for_test(b.view(), nocc, nvir, 1.0, 0.05);

    let mut f_vv = Array2::<f64>::zeros((nvir, nvir));
    let mut f_oo = Array2::<f64>::zeros((nocc, nocc));
    for a in 0..nvir {
        f_vv[(a, a)] = 0.6 + 0.2 * (a as f64);
    }
    for i in 0..nocc {
        f_oo[(i, i)] = -0.9 - 0.15 * (i as f64);
    }
    let mut x = Array2::<f64>::zeros((nvir, nocc));
    for e in x.iter_mut() {
        *e = rng.next_f64();
    }

    let y0 = aux.base_apply_ref(x.view(), &f_vv, &f_oo, 0.0);
    let lambda = 0.4;
    let y1 = aux.base_apply_ref(x.view(), &f_vv, &f_oo, lambda);
    let diag = aux.diagonal(&f_vv, &f_oo);

    let mut want = y0.clone();
    for a in 0..nvir {
        for i in 0..nocc {
            want[(a, i)] += lambda * diag[a * nocc + i] * x[(a, i)];
        }
    }
    let d = max_abs_diff(&y1, &want);
    eprintln!("shift application: max|Δ| = {d:.3e}");
    assert!(d < 1e-13, "the shift must enter as +λ·D·x: {d:.3e}");
    // And it must be a real change, not a no-op.
    let change = max_abs_diff(&y1, &y0);
    assert!(
        change > 1e-3,
        "a shift of {lambda} must visibly change the operator (got {change:.3e})"
    );
}

/// The base operator is symmetric, as any Hessian must be.
///
/// A transposed exchange contraction is the classic way to get a plausible but
/// non-symmetric operator, and PCG would then silently converge to nonsense.
#[test]
fn base_curvature_operator_is_symmetric() {
    let mut rng = Xorshift64::new(0x7f7f_7f7f);
    let (nocc, nvir, naux) = (3usize, 5usize, 7usize);
    let nmo = nocc + nvir;
    let b = synthetic_b(naux, nmo, &mut rng);
    let aux =
        ferric_scf::aurora::AuxCurvature::from_mo_tensor_for_test(b.view(), nocc, nvir, 1.0, 0.05);

    let mut f_vv = Array2::<f64>::zeros((nvir, nvir));
    let mut f_oo = Array2::<f64>::zeros((nocc, nocc));
    for a in 0..nvir {
        f_vv[(a, a)] = 0.5 + 0.3 * (a as f64);
    }
    for i in 0..nocc {
        f_oo[(i, i)] = -1.0 - 0.2 * (i as f64);
    }

    // Build the dense matrix by applying to unit vectors.
    let dim = nvir * nocc;
    let mut m = Array2::<f64>::zeros((dim, dim));
    for col in 0..dim {
        let mut e = Array2::<f64>::zeros((nvir, nocc));
        e[(col / nocc, col % nocc)] = 1.0;
        let y = 2.0 * (f_vv.dot(&e) - e.dot(&f_oo)) + &aux.response(e.view());
        for (row, v) in y.iter().enumerate() {
            m[(row, col)] = *v;
        }
    }
    let asym = m
        .indexed_iter()
        .fold(0.0f64, |acc, ((i, j), v)| acc.max((v - m[(j, i)]).abs()));
    let scale = m.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    eprintln!("base operator asymmetry: {asym:.3e} (scale {scale:.3e})");
    assert!(
        asym < 1e-12 * scale.max(1.0),
        "the curvature operator must be symmetric: asym = {asym:.3e}"
    );
}

/// PCG must actually solve an SPD system when given enough iterations.
///
/// Guards against a preconditioner or recursion bug that the loose production
/// forcing term (0.08, 3 iterations) would hide.
#[test]
fn pcg_solves_an_spd_system_to_tight_tolerance() {
    let mut rng = Xorshift64::new(0x00c0_ffee);
    let n = 14usize;
    let mut a = Array2::<f64>::zeros((n, n));
    for e in a.iter_mut() {
        *e = rng.next_f64();
    }
    let spd = a.t().dot(&a) + Array2::<f64>::eye(n) * 2.0;
    let mut rhs = Array1::<f64>::zeros(n);
    for e in rhs.iter_mut() {
        *e = rng.next_f64();
    }
    let diag: Array1<f64> = (0..n).map(|k| spd[(k, k)]).collect();

    let (x, info) =
        ferric_scf::aurora::pcg_solve_for_test(|v| spd.dot(v), &rhs, &diag, 200, 1, 1e-13, 1e-8);
    let resid = &spd.dot(&x) - &rhs;
    let rn = resid.dot(&resid).sqrt();
    eprintln!(
        "PCG on SPD({n}): iters = {}, residual = {rn:.3e}, negcurv = {}",
        info.iterations, info.negative_curvature
    );
    assert!(
        !info.negative_curvature,
        "SPD system must not report negative curvature"
    );
    assert!(
        rn < 1e-9,
        "PCG must solve an SPD system: residual = {rn:.3e}"
    );
}
