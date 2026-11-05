//! Slater–Condon signs: `general_element` against an independent
//! second-quantization construction.
//!
//! The reference builds `H` in the determinant basis by applying
//! `a†_p a_q` and `a†_p a†_r a_s a_q` to one `u64` over SPIN-ORBITALS
//! (α-before-β: bit `p` = (p, α), bit `n + p` = (p, β)), with its own
//! sign routine (a bit-by-bit loop, not `strings::phase_below`). The
//! integrals are random with no spatial symmetry (random symmetric `h`,
//! random 8-fold-symmetric `(pq|rs)`), so no matrix element vanishes by
//! symmetry and every sign is visible. The comparison is element-wise, so a
//! sign error is caught even when it would leave the spectrum unchanged.
//!
//! Sectors include odd `N_α`: that is where a β-channel operator picks up
//! `(−1)^{N_α}` in the α-before-β order, and where `general_element`'s
//! omission of it (it cancels pairwise) is actually tested.
//!
//! Exact derivation of the same convention: `scripts/casci_slater_condon_phases_sympy.py`;
//! equations: `crates/ferric-ci/theory.md`.

use ferric_ci::hamiltonian::{general_element, hamiltonian_diagonal, sigma, DeterminantSpace};
use ferric_ci::integrals::ActiveSpaceIntegrals;
use ferric_ci::strings::enumerate_strings;
use ndarray::Array2;
use std::collections::HashMap;

/// Deterministic xorshift64* in [-1, 1).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

fn random_integrals(n: usize, seed: u64) -> ActiveSpaceIntegrals {
    let mut rng = Rng(seed);
    let mut h = Array2::<f64>::zeros((n, n));
    for p in 0..n {
        for q in 0..=p {
            let v = rng.next();
            h[(p, q)] = v;
            h[(q, p)] = v;
        }
    }
    let idx = |p: usize, q: usize, r: usize, s: usize| ((p * n + q) * n + r) * n + s;
    let mut eri = vec![f64::NAN; n.pow(4)];
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    if !eri[idx(p, q, r, s)].is_nan() {
                        continue;
                    }
                    let v = rng.next();
                    for (a, b, c, d) in [
                        (p, q, r, s),
                        (q, p, r, s),
                        (p, q, s, r),
                        (q, p, s, r),
                        (r, s, p, q),
                        (s, r, p, q),
                        (r, s, q, p),
                        (s, r, q, p),
                    ] {
                        eri[idx(a, b, c, d)] = v;
                    }
                }
            }
        }
    }
    ActiveSpaceIntegrals {
        n_active: n,
        h,
        eri,
        e_core: rng.next(),
    }
}

/// Annihilate spin-orbital `m` in `state`: `None` if empty, else the new
/// state and `(−1)^{#occupied spin-orbitals below m}`, counted bit by bit.
fn annihilate(state: u64, m: usize) -> Option<(u64, f64)> {
    if state >> m & 1 == 0 {
        return None;
    }
    let mut sign = 1.0;
    for k in 0..m {
        if state >> k & 1 == 1 {
            sign = -sign;
        }
    }
    Some((state & !(1u64 << m), sign))
}

fn create(state: u64, m: usize) -> Option<(u64, f64)> {
    if state >> m & 1 == 1 {
        return None;
    }
    let mut sign = 1.0;
    for k in 0..m {
        if state >> k & 1 == 1 {
            sign = -sign;
        }
    }
    Some((state | (1u64 << m), sign))
}

/// `H |ket>` as a map spin-orbital state -> amplitude (electronic part),
/// `H = Σ h_pq a†_{pσ} a_{qσ} + ½ Σ (pq|rs) a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}`.
fn h_column(ints: &ActiveSpaceIntegrals, ket: u64) -> HashMap<u64, f64> {
    let n = ints.n_active;
    let so = |p: usize, spin: usize| p + n * spin;
    let mut col: HashMap<u64, f64> = HashMap::new();
    for sg in 0..2 {
        for p in 0..n {
            for q in 0..n {
                let Some((s1, f1)) = annihilate(ket, so(q, sg)) else {
                    continue;
                };
                let Some((s2, f2)) = create(s1, so(p, sg)) else {
                    continue;
                };
                *col.entry(s2).or_default() += f1 * f2 * ints.h[(p, q)];
            }
        }
    }
    for sg in 0..2 {
        for tau in 0..2 {
            for p in 0..n {
                for q in 0..n {
                    for r in 0..n {
                        for s in 0..n {
                            let Some((s1, f1)) = annihilate(ket, so(q, sg)) else {
                                continue;
                            };
                            let Some((s2, f2)) = annihilate(s1, so(s, tau)) else {
                                continue;
                            };
                            let Some((s3, f3)) = create(s2, so(r, tau)) else {
                                continue;
                            };
                            let Some((s4, f4)) = create(s3, so(p, sg)) else {
                                continue;
                            };
                            *col.entry(s4).or_default() +=
                                0.5 * f1 * f2 * f3 * f4 * ints.g(p, q, r, s);
                        }
                    }
                }
            }
        }
    }
    col
}

/// Max |general_element - brute force| over every (bra, ket) pair, plus the
/// number of pairs of each excitation class actually exercised with a
/// nonzero reference (so a vacuous sector cannot pass silently).
fn compare(n: usize, na: usize, nb: usize, seed: u64) -> (f64, [usize; 5]) {
    let ints = random_integrals(n, seed);
    let alpha = enumerate_strings(n, na);
    let beta = enumerate_strings(n, nb);
    let enc = |a: u64, b: u64| a | (b << n);
    let mut worst = 0.0f64;
    // [diag, single α, single β, double same-spin, double α-β]
    let mut seen = [0usize; 5];
    for &ak in &alpha {
        for &bk in &beta {
            let col = h_column(&ints, enc(ak, bk));
            for &abr in &alpha {
                for &bbr in &beta {
                    let reference = col.get(&enc(abr, bbr)).copied().unwrap_or(0.0);
                    let got = general_element(&ints, abr, bbr, ak, bk);
                    worst = worst.max((got - reference).abs());
                    let da = (abr ^ ak).count_ones();
                    let db = (bbr ^ bk).count_ones();
                    let class = match (da, db) {
                        (0, 0) => Some(0),
                        (2, 0) => Some(1),
                        (0, 2) => Some(2),
                        (4, 0) | (0, 4) => Some(3),
                        (2, 2) => Some(4),
                        _ => None,
                    };
                    if let Some(c) = class {
                        if reference.abs() > 1e-6 {
                            seen[c] += 1;
                        }
                    }
                }
            }
        }
    }
    (worst, seen)
}

#[test]
fn general_element_matches_second_quantization_on_random_integrals() {
    // (n_orb, N_α, N_β): odd and even N_α / N_β, same-spin doubles with
    // spectators (needs >= 5 orbitals in that channel with 3 electrons).
    for (n, na, nb, seed) in [
        (4, 2, 2, 0x9E37_79B9_7F4A_7C15u64),
        (5, 3, 2, 0xD1B5_4A32_D192_ED03),
        (5, 2, 3, 0x8CB9_2BA7_2F3D_8DD7),
        (6, 3, 3, 0x2545_F491_4F6C_DD1D),
        (6, 1, 4, 0x6A09_E667_F3BC_C909),
    ] {
        let (worst, seen) = compare(n, na, nb, seed);
        eprintln!(
            "n={n} (Na,Nb)=({na},{nb}): max |diff| = {worst:.2e}, \
             nonzero [diag, sα, sβ, dσσ, dαβ] = {seen:?}"
        );
        assert!(
            worst < 1e-12,
            "n={n} ({na},{nb}): general_element differs from the brute-force \
             second-quantized H by {worst:.3e}"
        );
        // Every excitation class reachable in the sector must be exercised.
        assert!(seen[0] > 0 && seen[1] > 0 && seen[2] > 0, "{seen:?}");
        if na >= 2 && n - na >= 2 || nb >= 2 && n - nb >= 2 {
            assert!(seen[3] > 0, "no same-spin doubles exercised: {seen:?}");
        }
        assert!(seen[4] > 0, "no α-β doubles exercised: {seen:?}");
    }
}

/// `sigma` and `hamiltonian_diagonal` (which add `e_core`) against the same
/// reference: `sigma(c) == (H_ref + e_core) c` for a random `c`.
#[test]
fn sigma_and_diagonal_match_second_quantization() {
    let (n, na, nb) = (5, 3, 2);
    let ints = random_integrals(n, 0xA076_1D64_78BD_642F);
    let space = DeterminantSpace {
        alpha_strings: enumerate_strings(n, na),
        beta_strings: enumerate_strings(n, nb),
    };
    let nbeta = space.beta_strings.len();
    let enc = |a: u64, b: u64| a | (b << n);
    let mut rng = Rng(0xE703_7ED1_A0B4_28DB);
    let c: Vec<f64> = (0..space.n_det()).map(|_| rng.next()).collect();
    let mut want = vec![0.0f64; space.n_det()];
    let mut diag_want = vec![0.0f64; space.n_det()];
    for (ia, &ak) in space.alpha_strings.iter().enumerate() {
        for (ib, &bk) in space.beta_strings.iter().enumerate() {
            let k = ia * nbeta + ib;
            let col = h_column(&ints, enc(ak, bk));
            for (ja, &abr) in space.alpha_strings.iter().enumerate() {
                for (jb, &bbr) in space.beta_strings.iter().enumerate() {
                    let d = ja * nbeta + jb;
                    let mut v = col.get(&enc(abr, bbr)).copied().unwrap_or(0.0);
                    if d == k {
                        v += ints.e_core;
                        diag_want[d] = v;
                    }
                    want[d] += v * c[k];
                }
            }
        }
    }
    let got = sigma(&ints, &space, &c);
    let diag = hamiltonian_diagonal(&ints, &space);
    let ds = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let dd = diag
        .iter()
        .zip(&diag_want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("sigma max diff {ds:.2e}, diagonal max diff {dd:.2e}");
    assert!(ds < 1e-11, "sigma differs from H_ref c by {ds:.3e}");
    assert!(dd < 1e-12, "diagonal differs by {dd:.3e}");
}

/// Anchor for the reference itself: the brute-force H is Hermitian and
/// conserves N_α, N_β (no amplitude leaves the sector).
#[test]
fn brute_force_reference_is_hermitian_and_number_conserving() {
    let (n, na, nb) = (5, 3, 2);
    let ints = random_integrals(n, 0x1234_5678_9ABC_DEF1);
    let enc = |a: u64, b: u64| a | (b << n);
    let mask_a = (1u64 << n) - 1;
    let mut m: HashMap<(u64, u64), f64> = HashMap::new();
    for &ak in &enumerate_strings(n, na) {
        for &bk in &enumerate_strings(n, nb) {
            let ket = enc(ak, bk);
            for (bra, v) in h_column(&ints, ket) {
                assert_eq!((bra & mask_a).count_ones() as usize, na);
                assert_eq!((bra >> n).count_ones() as usize, nb);
                m.insert((bra, ket), v);
            }
        }
    }
    for (&(bra, ket), &v) in &m {
        let t = m.get(&(ket, bra)).copied().unwrap_or(0.0);
        assert!(
            (v - t).abs() < 1e-13,
            "H_ref not symmetric at {bra:b},{ket:b}"
        );
    }
}
