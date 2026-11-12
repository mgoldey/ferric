//! Canonical CASSCF orbitals (`ferric_ci::casscf::canonicalize`).
//!
//! At ferric's converged CASSCF orbitals `C₀` (point `p₀`), `canonicalize`
//! returns `C₁ = C₀ U` (`U` block diagonal over core, active, virtual) and the
//! point `p₁` with the CI solved again in `C₁`. Systems: H2O CAS(4,4)/6-31G,
//! N2 CAS(6,6)/6-31G at r = 1.6 Å, and O2 CAS(8,6) with `(n_alpha, n_beta) =
//! (5, 3)`; ferric exact-J/K RHF start.
//!
//! # Anchors and bars
//!
//! Every rotation in `U` is redundant, so:
//!
//! 1. Energy. Both Ritz values are of unitarily equivalent Hamiltonians:
//!    `|E₁ − E₀| ≤ ‖r₀‖ + ‖r₁‖ + 8 ε |E| √N_det`.
//! 2. RDM-rebuilt energy. `E_core + Σ h γ + ½ Σ (pq|rs) Γ` over the integrals
//!    of `p₁` and the returned RDMs equals `E₁` to
//!    `4 ε A √(N_det + n⁴) + ‖r₁‖` (`A` the sum of absolute terms; the bar of
//!    `tests/casci_rdms.rs`).
//! 3. The returned CI vector is an eigenvector of the Hamiltonian in `C₁`:
//!    `‖H₁c − E₁c‖` recomputed by the direct sigma is below
//!    `10 × ci_conv_thresh` (Davidson's acceptance), and `c·H₁c = E₁` to
//!    the rounding bar of 2.
//! 4. Natural orbitals. With `U_a = C₀,aᵀ S C₁,a`, `δγ = γ₁ − U_aᵀ γ₀ U_a`.
//!    `U_aᵀ γ₀ U_a` is diagonal to rounding, so `max|offdiag γ₁| ≤ max|δγ| +
//!    4 ε n N`. `δγ` comes from the second CI solve only:
//!    `max|δγ| ≤ 2 N (‖r₀‖ + ‖r₁‖) / Δ + 16 ε N`, `Δ` the gap to the next
//!    root of the sector (dense Hamiltonian in `C₁`), the first-order bound
//!    `‖c − c*‖ ≤ ‖r‖/Δ` with `‖E_pq‖ ≤ N`.
//! 5. Canonical blocks. `F^I` in `C₁` is `UᵀF^I U` (the core density is
//!    unchanged) and `F^A[C₁, γ₁] = UᵀF^A U + F^A[C₁, δγ]` (linear in γ,
//!    `C₁,a U_aᵀ γ₀ U_a C₁,aᵀ = D^a`). The first term has diagonal core and
//!    virtual blocks to `4 ε n_mo max|F|`; the second is bounded by
//!    `L Σ_{t≤u} |δγ_tu|`, `L = max_{t≤u} max|F^A[C₁, B_tu]|` over the
//!    symmetric unit matrices `B_tu`. The core and virtual diagonals are
//!    ascending and the occupations descending.
//! 6. Idempotence. Canonicalizing `C₁` again gives `C₂` with
//!    `O = C₁ᵀ S C₂` the identity. For `p ≠ q` in one block,
//!    `|O_pq| ≤ bar_pq + δ_O`, `bar_pq = 2 (e_qp/|λ_p − λ_q| + Σ_r e_qr e_rp
//!    / (|λ_p − λ_q| |λ_p − λ_r|))`, the first- and second-order eigenvector
//!    perturbation of the matrix `M` diagonalized (`F^T` block or `γ₁`), with
//!    `e = |M_offdiag| + 4 ε n max|M|` (eigensolver backward error), and
//!    `|O_pp − 1| ≤ Σ_q bar_pq² + δ_O`, asserted where every `bar_pq` of the
//!    row is below 0.1 (the expansion holds). `δ_O = n_mo (max|C₁ᵀ S C₁ − I|
//!    + 4 ε)`, since `O = (C₁ᵀ S C₁) U₂`; it also bounds the cross-block
//!    elements. Columns are signed by `sign_reference`, so the identity holds
//!    with sign `+1`. Degenerate pairs (the stretched N2 π pairs) have
//!    `bar_pq ≥ 0.1`: any rotation inside them is allowed, and their number
//!    is printed. In O2 from the singlet RHF start every pair has
//!    `bar_pq < 0.1`.
//!
//! Physics vs artifact: canonicalization changes the basis only. If the CI
//! vector were not carried into the new orbitals, 2 and 3 would fail at the
//! size of the active rotation and 4 at the size of `offdiag γ₀`; if `F^A` were
//! missing from the canonical Fock, 5 would fail at `offdiag UᵀF^A U`.
//!
//! # Measured (2026-11-11) and mutations
//!
//! Tables in `crates/ferric-ci/theory.md` ("Two-step CASSCF").

mod casscf_common;

use casscf_common::{maxabs, molecule_text, pyscf_reference, setup_text, EPS};
use ferric_ci::casscf::{canonicalize, CasScfConfig, CasScfPoint, CasScfProblem};
use ferric_ci::hamiltonian::{dense_hamiltonian, DeterminantSpace};
use ferric_ci::rdm::rdm_energy;
use ferric_ci::run_casscf;
use ferric_ci::sigma::DirectSigma;
use ferric_ci::strings::enumerate_strings;
use ferric_integrals::oneelectron::overlap;
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, UPLO};

fn blocks(nc: usize, na: usize, nmo: usize) -> [(usize, usize); 3] {
    [(0, nc), (nc, nc + na), (nc + na, nmo)]
}

/// The matrix whose blocks `canonicalize` diagonalizes at a point: `F^T` for
/// core and virtual, `γ` in the active block (zero elsewhere).
fn canonical_target(pt: &CasScfPoint, nc: usize, na: usize) -> Array2<f64> {
    let mut m = &pt.f_inactive + &pt.f_active;
    let nmo = m.nrows();
    for p in 0..nmo {
        for q in 0..nmo {
            let same = blocks(nc, na, nmo)
                .iter()
                .any(|&(lo, hi)| (lo..hi).contains(&p) && (lo..hi).contains(&q));
            if !same {
                m[(p, q)] = 0.0;
            }
        }
    }
    m.slice_mut(s![nc..nc + na, nc..nc + na])
        .assign(&pt.rdms().dm1);
    m
}

fn case(label: &str, xyz: &str, n_core: usize, n_active: usize, nel: (usize, usize)) {
    let (mol, prep, rhf) = setup_text(xyz, "6-31g");
    let cfg = CasScfConfig {
        n_core,
        n_active,
        n_elec_active: nel,
        ..Default::default()
    };
    let res = run_casscf(&mol, &prep, &rhf, &cfg).unwrap();
    assert!(res.converged, "{label}: not converged");
    let nmo = res.mo_coeff.ncols();
    let problem = CasScfProblem::new(&mol, &prep, nmo, &cfg).unwrap();
    let c0 = res.mo_coeff.clone();
    let p0 = problem.evaluate(&c0).unwrap();
    let (c1, p1) = canonicalize(&problem, &c0, &p0).unwrap();
    let (nc, na) = (n_core, n_active);
    let nn = (nel.0 + nel.1) as f64;
    let ndet = p1.ci.n_determinants as f64;
    let (r0, r1) = (p0.ci.residual_norm, p1.ci.residual_norm);
    let e1 = p1.e_total;

    // 1. energy
    let de = e1 - p0.e_total;
    let bar1 = r0 + r1 + 8.0 * EPS * e1.abs() * ndet.sqrt();

    // 2. RDM-rebuilt energy, 3. eigenvector in the new orbitals
    let ints = &p1.ci.integrals;
    let er = rdm_energy(ints, p1.rdms());
    let bar2 = 4.0 * EPS * er.abs_terms * (ndet + (na as f64).powi(4)).sqrt();
    let d2 = (er.e_total - e1).abs();
    let space = DeterminantSpace {
        alpha_strings: enumerate_strings(na, nel.0),
        beta_strings: enumerate_strings(na, nel.1),
    };
    let c = &p1.ci.ci_vector;
    let hc = DirectSigma::new(ints, &space).apply(c);
    let rq: f64 = c.iter().zip(&hc).map(|(a, b)| a * b).sum();
    let resid = hc
        .iter()
        .zip(c)
        .map(|(h, x)| (h - e1 * x).powi(2))
        .sum::<f64>()
        .sqrt();

    // 4. natural orbitals
    let s_ao = overlap(&prep);
    let ua = c0
        .slice(s![.., nc..nc + na])
        .t()
        .dot(&s_ao)
        .dot(&c1.slice(s![.., nc..nc + na]));
    let g0 = &p0.rdms().dm1;
    let g1 = &p1.rdms().dm1;
    let dgam = g1 - &ua.t().dot(g0).dot(&ua);
    let hdense = dense_hamiltonian(ints, &space, None).unwrap();
    let (ev, _) = hdense.eigh(UPLO::Upper).unwrap();
    let gap = ev[1] - ev[0];
    let bar_dg = 2.0 * nn * (r0 + r1) / gap + 16.0 * EPS * nn;
    let mut off_g = 0.0f64;
    for t in 0..na {
        for u in 0..na {
            if t != u {
                off_g = off_g.max(g1[(t, u)].abs());
            }
        }
    }
    let bar_offg = maxabs(&dgam) + 4.0 * EPS * na as f64 * nn;

    // 5. canonical core and virtual blocks
    let ft = &p1.f_inactive + &p1.f_active;
    let mut lip = 0.0f64;
    let mut sum_dg = 0.0;
    for t in 0..na {
        for u in t..na {
            let mut b = Array2::<f64>::zeros((na, na));
            b[(t, u)] = 1.0;
            b[(u, t)] = 1.0;
            lip = lip.max(maxabs(&problem.active_fock(&c1, &b).unwrap()));
            sum_dg += dgam[(t, u)].abs();
        }
    }
    let bar_f = 4.0 * EPS * nmo as f64 * maxabs(&ft) + lip * sum_dg;
    let mut off_f = 0.0f64;
    let mut ordered = true;
    for &(lo, hi) in [(0, nc), (nc + na, nmo)].iter() {
        for p in lo..hi {
            for q in lo..hi {
                if p != q {
                    off_f = off_f.max(ft[(p, q)].abs());
                }
            }
            if p + 1 < hi {
                ordered &= ft[(p, p)] <= ft[(p + 1, p + 1)] + bar_f;
            }
        }
    }
    for t in 0..na - 1 {
        ordered &= g1[(t, t)] + bar_offg >= g1[(t + 1, t + 1)];
    }

    // 6. idempotence
    let (c2, _p2) = canonicalize(&problem, &c1, &p1).unwrap();
    let o = c1.t().dot(&s_ao).dot(&c2);
    let m = canonical_target(&p1, nc, na);
    // O = (C₁ᵀ S C₁) U₂: the orthonormality error of C₁ enters every element.
    let orth = maxabs(&(c1.t().dot(&s_ao).dot(&c1) - Array2::<f64>::eye(nmo)));
    let o_round = nmo as f64 * (orth + 4.0 * EPS);
    let mut worst = 0.0f64; // largest |O_pq| / bar_pq
    let mut worst_at = (0usize, 0usize, 0.0f64, 0.0f64);
    let mut free_pairs = 0usize;
    for &(lo, hi) in blocks(nc, na, nmo).iter() {
        let mblk = m.slice(s![lo..hi, lo..hi]).to_owned();
        let mmax = maxabs(&mblk);
        let n = (hi - lo) as f64;
        // Backward error of the eigensolver `canonicalize` uses (eigh, lower
        // triangle, of ½(M + Mᵀ)), measured on this block: the residual
        // |MV − VΛ| plus the orthogonality loss of V times |M|. LAPACK bounds
        // it by c·n·ε·|M| with an unstated c, so it is measured, not assumed;
        // 4nε|M| stays as a floor.
        let delta = {
            let sym = 0.5 * (&mblk + &mblk.t());
            let (w, v) = sym.eigh(UPLO::Lower).unwrap();
            let resid = maxabs(&(sym.dot(&v) - &v * &w));
            let orth_v = maxabs(&(v.t().dot(&v) - Array2::<f64>::eye(hi - lo)));
            (resid + orth_v * mmax).max(4.0 * EPS * n * mmax)
        };
        let gap = |p: usize, q: usize| (m[(p, p)] - m[(q, q)]).abs();
        for p in lo..hi {
            let mut sum_sq = 0.0;
            for q in lo..hi {
                if p == q {
                    continue;
                }
                // First order E_qp/(λp−λq) plus the second-order term
                // Σ_r E_qr E_rp/((λp−λq)(λp−λr)), with E the off-diagonal
                // part of M plus the eigensolver backward error, times 2.
                let e = |a: usize, b: usize| m[(a, b)].abs() + delta;
                let first = e(q, p) / gap(p, q);
                let second: f64 = (lo..hi)
                    .filter(|&r| r != p && r != q)
                    .map(|r| e(q, r) * e(r, p) / (gap(p, q) * gap(p, r)))
                    .sum();
                let bar = 2.0 * (first + second);
                // The expansion holds for bar ≪ 1; above 0.1 (degenerate
                // or nearly degenerate pairs) the rotation is not
                // constrained.
                if bar >= 0.1 {
                    free_pairs += 1;
                    sum_sq = f64::INFINITY;
                    continue;
                }
                sum_sq += bar * bar;
                let ratio = o[(p, q)].abs() / (bar + o_round);
                if ratio > worst {
                    worst = ratio;
                    worst_at = (p, q, o[(p, q)], bar);
                }
            }
            let bar_d = sum_sq + o_round;
            if bar_d.is_finite() {
                let ratio = (o[(p, p)] - 1.0).abs() / bar_d;
                if ratio > worst {
                    worst = ratio;
                    worst_at = (p, p, o[(p, p)] - 1.0, bar_d);
                }
            }
        }
    }
    let cross = {
        let mut x = 0.0f64;
        for (bi, &(lo, hi)) in blocks(nc, na, nmo).iter().enumerate() {
            for (bj, &(lo2, hi2)) in blocks(nc, na, nmo).iter().enumerate() {
                if bi != bj {
                    x = x.max(maxabs(&o.slice(s![lo..hi, lo2..hi2]).to_owned()));
                }
            }
        }
        x
    };

    eprintln!(
        "{label}: E0 {:.13} E1 {e1:.13} dE {de:.1e} (bar {bar1:.1e}); |E_rdm - E1| {d2:.1e} \
         (bar {:.1e}); |c.Hc - E1| {:.1e}; |Hc - Ec| {resid:.1e}\n  \
         max|dgamma| {:.1e} (bar {bar_dg:.1e}, CI gap {gap:.3}); max|offdiag gamma1| {off_g:.1e} \
         (bar {bar_offg:.1e}); max|offdiag F^T core/virt| {off_f:.1e} (bar {bar_f:.1e}, L {lip:.2})\n  \
         idempotence: worst |O|/bar {worst:.2} at {worst_at:?}, cross-block {cross:.1e}, {free_pairs} degenerate \
         pairs unconstrained; |g0| {:.1e} |g1| {:.1e}",
        p0.e_total,
        bar2 + r1,
        (rq - e1).abs(),
        maxabs(&dgam),
        norm(&p0.gradient),
        norm(&p1.gradient),
    );
    assert!(de.abs() <= bar1, "{label}: energy moved by {de:e}");
    assert!(d2 <= bar2 + r1, "{label}: RDM energy off by {d2:e}");
    assert!((rq - e1).abs() <= bar2 + r1, "{label}: c.Hc off");
    assert!(
        resid <= 10.0 * cfg.ci_conv_thresh,
        "{label}: residual {resid:e}"
    );
    assert!(
        maxabs(&dgam) <= bar_dg,
        "{label}: dgamma {:e}",
        maxabs(&dgam)
    );
    assert!(off_g <= bar_offg, "{label}: gamma1 off-diagonal {off_g:e}");
    assert!(
        off_f <= bar_f,
        "{label}: F^T off-diagonal {off_f:e} > {bar_f:e}"
    );
    assert!(ordered, "{label}: eigenvalue order");
    assert!(
        worst <= 1.0,
        "{label}: second canonicalization moved orbitals"
    );
    // U₂ is block diagonal, so the cross-block elements of O come from the
    // orthonormality error of C₁ alone.
    let bar_cross = o_round;
    assert!(
        cross <= bar_cross,
        "{label}: blocks mixed ({cross:e} > {bar_cross:e})"
    );
    // The test must see a nontrivial rotation, or 2-5 cannot fail.
    let rot = maxabs(&(&ua.mapv(f64::abs) - &Array2::<f64>::eye(na)));
    assert!(
        rot > 1e-3,
        "{label}: active rotation {rot:e} too small to test"
    );
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn canonical_h2o_cas44_631g() {
    case("H2O CAS(4,4)", &molecule_text("water.xyz"), 3, 4, (2, 2));
}

#[test]
fn canonical_n2_stretched_cas66_631g() {
    let r = pyscf_reference("n2_r1.6_cas66_631g");
    case(
        "N2 r=1.6 CAS(6,6)",
        r["xyz"].as_str().unwrap(),
        4,
        6,
        (3, 3),
    );
}

#[test]
fn canonical_o2_triplet_cas86_631g() {
    case("O2 (5,3) CAS(8,6)", &molecule_text("o2.xyz"), 4, 6, (5, 3));
}
