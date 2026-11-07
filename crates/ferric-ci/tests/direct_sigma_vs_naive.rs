//! Exactness anchor for the string-driven direct sigma
//! (`ferric_ci::sigma::DirectSigma`, reached through `hamiltonian::sigma`).
//!
//! Oracle: `hamiltonian::sigma_naive`, the `O(N_det²)` loop applying the
//! Slater–Condon `general_element` to every determinant pair. The two share
//! only `strings::phase_below`; the direct build never classifies pairs, so an
//! error in its excitation lists, phases, the `g_pq` one-body correction, the
//! ½ on same-spin products, or a missing σ1/σ2/σ3 term shows up element-wise.
//! (`tests/slater_condon_brute_force.rs` separately compares the direct sigma
//! with an independent bit-loop second-quantized H.)
//!
//! Artifact hypothesis: if the direct build were broken, a random `c` would
//! give a difference of order the integrals (≥ 1e-3); agreement at round-off
//! (≤ 1e-12) is only reachable when every term is right. Random `c` has no
//! symmetry, so cancellations that a ground-state vector could hide do not
//! occur. Sectors include `N_α ≠ N_β`, empty and full channels.

use ferric_ci::hamiltonian::{
    dense_hamiltonian, hamiltonian_diagonal, sigma, sigma_naive, DeterminantSpace,
};
use ferric_ci::integrals::{build_active_space_integrals, ActiveSpaceIntegrals};
use ferric_ci::sigma::DirectSigma;
use ferric_ci::strings::enumerate_strings;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

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

/// Random symmetric `h`, random 8-fold-symmetric `(pq|rs)`: no element
/// vanishes by spatial symmetry.
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

fn space(n: usize, na: usize, nb: usize) -> DeterminantSpace {
    DeterminantSpace {
        alpha_strings: enumerate_strings(n, na),
        beta_strings: enumerate_strings(n, nb),
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Direct vs naive sigma on `nvec` random vectors; returns the worst
/// difference and the largest |σ| (so the bar is visibly relative to scale).
fn compare(
    ints: &ActiveSpaceIntegrals,
    sp: &DeterminantSpace,
    seed: u64,
    nvec: usize,
) -> (f64, f64) {
    let direct = DirectSigma::new(ints, sp);
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for _ in 0..nvec {
        let c: Vec<f64> = (0..sp.n_det()).map(|_| rng.next()).collect();
        let want = sigma_naive(ints, sp, &c);
        let got = direct.apply(&c);
        worst = worst.max(max_abs_diff(&got, &want));
        scale = want.iter().fold(scale, |m, x| m.max(x.abs()));
        // The convenience entry point is the same operator.
        let got2 = sigma(ints, sp, &c);
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            got2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "hamiltonian::sigma must be DirectSigma::apply"
        );
    }
    (worst, scale)
}

#[test]
fn direct_sigma_matches_naive_on_random_integrals() {
    // (n, N_α, N_β): equal and unequal, odd/even, empty and full channels.
    for (n, na, nb, seed) in [
        (4, 2, 2, 0x9E37_79B9_7F4A_7C15u64),
        (5, 3, 2, 0xD1B5_4A32_D192_ED03),
        (5, 2, 3, 0x8CB9_2BA7_2F3D_8DD7),
        (6, 3, 3, 0x2545_F491_4F6C_DD1D),
        (6, 1, 4, 0x6A09_E667_F3BC_C909),
        (6, 4, 2, 0xBB67_AE85_84CA_A73B),
        (5, 0, 3, 0x3C6E_F372_FE94_F82B),
        (4, 4, 1, 0xA54F_F53A_5F1D_36F1),
        (7, 4, 3, 0x510E_527F_ADE6_82D1),
        (1, 1, 0, 0x9B05_688C_2B3E_6C1F),
    ] {
        let ints = random_integrals(n, seed);
        let sp = space(n, na, nb);
        let (worst, scale) = compare(&ints, &sp, seed ^ 0x5555, 3);
        eprintln!(
            "random n={n} ({na},{nb}) N_det={}: max |direct - naive| = {worst:.2e} (max |σ| {scale:.2e})",
            sp.n_det()
        );
        assert!(
            worst <= 1e-12,
            "n={n} ({na},{nb}): direct sigma differs from naive by {worst:.3e}"
        );
    }
}

fn h2o_sto3g() -> (Molecule, PreparedBasis, ferric_scf::ScfResult) {
    let xyz = "3\nH2O\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            ..Default::default()
        },
    )
    .unwrap();
    (mol, prep, rhf)
}

/// Real H2O/STO-3G integrals: full CI (441 det), frozen-core CAS(6,8)
/// (225 det), and unequal-spin sectors over the same orbitals. Also the
/// diagonal against the dense matrix's diagonal.
#[test]
fn direct_sigma_matches_naive_on_h2o_sto3g() {
    let (mol, prep, rhf) = h2o_sto3g();
    // (active_start, n_active, N_α, N_β, expected N_det)
    for (start, n, na, nb, ndet) in [
        (0usize, 7usize, 5usize, 5usize, 441usize),
        (1, 6, 4, 4, 225),
        (0, 7, 6, 4, 245),
        (0, 7, 4, 6, 245),
        (1, 6, 5, 3, 120),
        (2, 5, 3, 1, 50),
    ] {
        let ints = build_active_space_integrals(&mol, &prep, &rhf, start, n, None).unwrap();
        let sp = space(n, na, nb);
        assert_eq!(sp.n_det(), ndet);
        let (worst, scale) = compare(&ints, &sp, 0xC0FF_EE00 ^ (n as u64) ^ ((na as u64) << 8), 3);
        let dense = dense_hamiltonian(&ints, &sp, None).unwrap();
        let diag = hamiltonian_diagonal(&ints, &sp);
        let dd = (0..ndet)
            .map(|i| (diag[i] - dense[(i, i)]).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "H2O/STO-3G start={start} n={n} ({na},{nb}) N_det={ndet}: max |direct - naive| = \
             {worst:.2e} (max |σ| {scale:.2e}), max |diag - dense diag| = {dd:.2e}"
        );
        // |σ| ~ |E_core| ~ 1e2 Ha here, so 1e-12 absolute is ~1e-14 relative.
        assert!(worst <= 1e-12, "direct vs naive {worst:.3e}");
        assert!(dd <= 1e-12, "diagonal vs dense diagonal {dd:.3e}");
    }
}

/// The dense matrix's columns are `H e_j`: the direct sigma on unit vectors
/// reproduces them, including the exact zeros of non-connected pairs.
#[test]
fn direct_sigma_on_unit_vectors_reproduces_dense_columns() {
    let ints = random_integrals(5, 0x1F83_D9AB_FB41_BD6B);
    let sp = space(5, 3, 2);
    let dense = dense_hamiltonian(&ints, &sp, None).unwrap();
    let direct = DirectSigma::new(&ints, &sp);
    let mut worst = 0.0f64;
    for j in 0..sp.n_det() {
        let mut e = vec![0.0; sp.n_det()];
        e[j] = 1.0;
        let col = direct.apply(&e);
        for i in 0..sp.n_det() {
            worst = worst.max((col[i] - dense[(i, j)]).abs());
        }
    }
    eprintln!("unit-vector columns: max |direct - dense| = {worst:.2e}");
    assert!(worst <= 1e-12, "{worst:.3e}");
}
