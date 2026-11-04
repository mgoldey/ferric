//! Split-RI-J (`ferric_scf::split_rij::SplitRiJ`) against libint2.
//!
//! The reference is an INDEPENDENT construction: `(P|μν)` from libint2's
//! 3-centre engine (Obara-Saika/HGP, its own Boys function and cart→sph) with
//! primitive screening OFF (precision 0), contracted here in plain loops. The
//! split kernel is ferric's own McMurchie-Davidson code, so agreement is
//! corroboration, not self-consistency. The stored RI-J (`DfJ`, in core) is
//! compared as well, through the triangle inequality.
//!
//! # Tolerances (derived, not chosen)
//!
//! `u = 2^-53`, `γ_k = k u / (1 - k u)` (Higham). For a computed sum the
//! floating-point error is at most `γ_k · Σ|terms|`, with `k` the depth of the
//! longest rounding chain.
//!
//! * The split side: `SplitRiJ::fit_rhs_abs` / `coulomb_abs` run the SAME
//!   passes with every factor replaced by its absolute value (|D| or |c|, |E|,
//!   |R| from the absolute recursions, |C|). That is exactly `Σ|terms|` of the
//!   computed sums, `S`. `SplitRiJ::rounding_depth()` is `k`.
//! * The libint side: its integrals carry their own rounding, and the test's
//!   contraction adds `γ_{nao²}` (d) or `γ_{naux}` (J). `S ≥ Σ|W||B|`
//!   because `S` expands every element's terms before taking absolute values,
//!   so the libint side is bounded by `γ_{nao² + 64} S` (resp.
//!   `γ_{naux + 64} S`). The `+64` is an ASSUMED depth for libint2's VRR/HRR
//!   chain (≤ l_tot + contraction length, far below 64 for these bases).
//! * `FLOOR = 1e-16` absolute covers the primitive pairs the split kernel drops
//!   because `|c_a c_b| e^{-μR²} < 1e-30` (`PRIM_SCREEN`). Each contributes
//!   ≤ 1e-30 · `2π^{5/2}/(pq√(p+q))` (≤ ~2e3 for p ≥ 0.6 at R ≤ 10 bohr,
//!   q ≥ 0.05) · |E| growth (≤ R^l ~ 1e4) · |D| (≤ 10): ≤ ~1e-22 each, and
//!   far fewer than 1e6 of them per element.
//! * End to end, `c = V⁻¹ d` couples the two passes. With `M = V⁻¹ B`
//!   (computed here) and Cholesky backward error `|ΔV| ≤ γ_{3 naux}|L||Lᵀ|`,
//!   `|J − J_ref| ≤ tol_J(c) + |M|ᵀ (tol_d + γ_{3naux} |L||Lᵀ| (|c| + |c_ref|))
//!   + γ_{naux} |B|ᵀ|c_ref|`. The `|M|` factor carries the metric's
//!   conditioning (cond(V) ~ 1e8 for jkfit), so this bound is honest but loose.
//!
//! What a defect looks like: a wrong E coefficient, a sign in G, a missing
//! primitive or a cart→sph slip moves elements by O(1e-3..1) relative,
//! 8+ orders above every bar here.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_scf::df_j::DfJ;
use ferric_scf::fock::JBuilder;
use ferric_scf::split_rij::{SplitRiJ, FAR_EPS};
use ndarray::{Array1, Array2, Array3};
use ndarray_linalg::{Cholesky, Diag, SolveTriangular, UPLO};

const U: f64 = f64::EPSILON / 2.0;
const FLOOR: f64 = 1e-16;
const AUX: &str = "def2-universal-jkfit";

fn gamma(k: usize) -> f64 {
    let ku = k as f64 * U;
    ku / (1.0 - ku)
}

struct Sys {
    obs: PreparedBasis,
    aux: PreparedBasis,
    nao: usize,
    naux: usize,
}

fn system(xyz: &str, orb: &str) -> Sys {
    let path = format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    );
    let mol = Molecule::load_xyz(&path).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(orb).unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let (nao, naux) = (obs.nbasis(), aux.nbasis());
    Sys {
        obs,
        aux,
        nao,
        naux,
    }
}

/// `(P|μν)` from libint2, primitive screening off, `(naux, nao, nao)`.
fn reference_tensor(s: &Sys) -> Array3<f64> {
    let op = Operator::coulomb();
    let mut eng = Engine::new_3center(op, &s.obs, &s.aux, 0.0).unwrap();
    let (dob, oob) = (s.obs.shell_dims().to_vec(), s.obs.shell_offsets().to_vec());
    let (dax, oax) = (s.aux.shell_dims().to_vec(), s.aux.shell_offsets().to_vec());
    let mut b = Array3::<f64>::zeros((s.naux, s.nao, s.nao));
    for sp in 0..s.aux.nshells() {
        for s1 in 0..s.obs.nshells() {
            for s2 in 0..=s1 {
                let Some(blk) = eng.compute_eri3(&s.obs, &s.aux, sp, s1, s2) else {
                    continue;
                };
                let (n1, n2) = (dob[s1], dob[s2]);
                for p in 0..dax[sp] {
                    for i in 0..n1 {
                        for j in 0..n2 {
                            let v = blk[(p * n1 + i) * n2 + j];
                            b[[oax[sp] + p, oob[s1] + i, oob[s2] + j]] = v;
                            b[[oax[sp] + p, oob[s2] + j, oob[s1] + i]] = v;
                        }
                    }
                }
            }
        }
    }
    b
}

/// Deterministic, NON-symmetric (exercises the `D_μν + D_νμ` folding),
/// mixed-sign density with a dominant diagonal.
fn test_density(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        let x = ((i * 37 + j * 11 + 5) % 23) as f64 / 23.0 - 0.4;
        if i == j {
            1.0 + 0.5 * x
        } else {
            0.3 * x * (-0.05 * (i as f64 - j as f64).abs()).exp()
        }
    })
}

fn flat(b: &Array3<f64>) -> Array2<f64> {
    let (naux, n, _) = b.dim();
    b.view()
        .into_shape_with_order((naux, n * n))
        .unwrap()
        .to_owned()
}

/// Assert `|got - want| <= tol` elementwise; returns the worst `|diff| / tol`.
fn check<'a>(
    what: &str,
    got: impl IntoIterator<Item = &'a f64>,
    want: impl IntoIterator<Item = &'a f64>,
    tol: impl IntoIterator<Item = &'a f64>,
) -> f64 {
    let mut worst = 0.0_f64;
    for (i, ((g, w), t)) in got.into_iter().zip(want).zip(tol).enumerate() {
        let d = (g - w).abs();
        assert!(
            d <= *t,
            "{what}[{i}]: split {g:.17e} vs reference {w:.17e}, |diff| {d:.3e} > tol {t:.3e}"
        );
        worst = worst.max(d / t);
    }
    worst
}

fn check_system(xyz: &str, orb: &str) {
    let s = system(xyz, orb);
    let op = Operator::coulomb();
    let v = coulomb_metric_2c(op, &s.aux).unwrap();
    // thresh 0: the split kernel keeps every block, as the unscreened reference.
    let mut sr = SplitRiJ::with_metric(op, &s.obs, &s.aux, v.clone(), 0.0).unwrap();
    let k = sr.rounding_depth();
    let b = reference_tensor(&s);
    let b2 = flat(&b);
    let babs = b2.mapv(f64::abs);
    let d = test_density(s.nao);
    let dflat = Array1::from_iter(d.iter().copied());

    // --- d pass ----------------------------------------------------------
    let d_split = sr.fit_rhs(&d).unwrap();
    let d_ref = b2.dot(&dflat);
    let s_d = sr.fit_rhs_abs(&d).unwrap();
    // FAR: the far-field terms' relative error (`FAR_EPS` per term, bounded by
    // the same |term| sums), with a factor 2 margin.
    let far = 2.0 * FAR_EPS;
    let tol_d = s_d.mapv(|x| (gamma(k) + gamma(s.nao * s.nao + 64) + far) * x + FLOOR);
    let wd = check("d_P", d_split.iter(), d_ref.iter(), tol_d.iter());

    // --- J pass at fixed coefficients -------------------------------------
    let c_ref = sr.solve_metric(&d_ref).unwrap();
    let mut j_c = Array2::<f64>::zeros((s.nao, s.nao));
    sr.coulomb_from_coefficients(&c_ref, &mut j_c).unwrap();
    let j_c_ref = b2.t().dot(&c_ref);
    let mut s_j = Array2::<f64>::zeros((s.nao, s.nao));
    sr.coulomb_abs(&c_ref, &mut s_j).unwrap();
    let tol_jc = s_j.mapv(|x| (gamma(k) + gamma(s.naux + 64) + far) * x + FLOOR);
    let wjc = check("J(c)", j_c.iter(), j_c_ref.iter(), tol_jc.iter());

    // --- end to end --------------------------------------------------------
    let mut j = Array2::<f64>::zeros((s.nao, s.nao));
    sr.build(&d, &mut j).unwrap();
    let c_split = sr.solve_metric(&d_split).unwrap();
    let mut s_j2 = Array2::<f64>::zeros((s.nao, s.nao));
    sr.coulomb_abs(&c_split, &mut s_j2).unwrap();
    let l = v.cholesky(UPLO::Lower).unwrap();
    let y = l.solve_triangular(UPLO::Lower, Diag::NonUnit, &b2).unwrap();
    let m = l
        .t()
        .to_owned()
        .solve_triangular(UPLO::Upper, Diag::NonUnit, &y)
        .unwrap();
    let labs = l.mapv(f64::abs);
    let cc = c_split.mapv(f64::abs) + c_ref.mapv(f64::abs);
    let solve_err = labs.dot(&labs.t().dot(&cc)) * gamma(3 * s.naux);
    let prop = m.mapv(f64::abs).t().dot(&(&tol_d + &solve_err));
    let ref_dot = babs.t().dot(&c_ref.mapv(f64::abs)) * gamma(s.naux);
    let tol_j: Vec<f64> = s_j2
        .iter()
        .zip(prop.iter().zip(ref_dot.iter()))
        .map(|(&sj, (&pr, &rd))| (gamma(k) + gamma(s.naux + 64) + far) * sj + pr + rd + FLOOR)
        .collect();
    let wj = check("J", j.iter(), j_c_ref.iter(), tol_j.iter());

    // --- against the stored RI-J ---------------------------------------------
    // |J_split - J_DfJ| <= |J_split - J_ref| + |J_ref - J_DfJ|; the second term
    // (libint precision 1e-14 vs 0, packed summation order) is measured.
    let mut df_j = DfJ::new(op, &s.obs, &s.aux, usize::MAX).unwrap();
    let mut j_dfj = Array2::<f64>::zeros((s.nao, s.nao));
    df_j.build(&d, &mut j_dfj).unwrap();
    let tol_dfj: Vec<f64> = tol_j
        .iter()
        .zip(j_dfj.iter().zip(j_c_ref.iter()))
        .map(|(&t, (&a, &r))| t + (a - r).abs())
        .collect();
    check("J vs DfJ", j.iter(), j_dfj.iter(), tol_dfj.iter());
    let max_dfj = j
        .iter()
        .zip(j_dfj.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let e = |jm: &Array2<f64>| 0.5 * (&d * jm).sum();
    eprintln!(
        "{xyz}/{orb}: nao {} naux {} k {k}; worst |diff|/tol d {wd:.2e} J(c) {wjc:.2e} J {wj:.2e}; \
         max|J - J_DfJ| {max_dfj:.3e}; E_J split {:.15} DfJ {:.15}",
        s.nao,
        s.naux,
        e(&j),
        e(&j_dfj)
    );
    for i in 0..s.nao {
        for jj in 0..i {
            assert_eq!(
                j[(i, jj)].to_bits(),
                j[(jj, i)].to_bits(),
                "J not exactly symmetric"
            );
        }
    }
}

/// d functions in the orbital basis, g in the aux.
#[test]
fn split_rij_matches_libint_water_ccpvdz() {
    check_system("water.xyz", "cc-pvdz");
}

#[test]
fn split_rij_matches_libint_benzene_def2svp() {
    check_system("benzene.xyz", "def2-svp");
}

/// f functions on sulfur, g in the aux: l_tot up to 10.
#[test]
fn split_rij_matches_libint_h2s_def2tzvp() {
    check_system("h2s.xyz", "def2-tzvp");
}

/// g functions on oxygen: l_tot up to 12.
#[test]
fn split_rij_matches_libint_water_def2qzvp() {
    check_system("water.xyz", "def2-qzvp");
}

/// The lane kernel (4 bra primitive pairs per R evaluation) reproduces the
/// one-primitive-at-a-time kernel bit for bit: every lane runs the scalar
/// operations, and every accumulator still receives its terms in
/// primitive-pair order. Benzene/def2-SVP has contracted s and p shells
/// (full and partial lane groups); water/def2-QZVP reaches l_tot = 12.
#[test]
fn split_rij_lane_kernel_is_bit_identical_to_the_scalar_kernel() {
    let bits = |a: &[f64]| a.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
    for (xyz, orb) in [("benzene.xyz", "def2-svp"), ("water.xyz", "def2-qzvp")] {
        let s = system(xyz, orb);
        let mut sr = SplitRiJ::new(Operator::coulomb(), &s.obs, &s.aux, 1e-12).unwrap();
        sr.set_far_field(false); // both sides on the exact kernel
        let d = test_density(s.nao);
        let c = Array1::from_shape_fn(s.naux, |p| ((p * 13 % 17) as f64 - 8.0) * 0.01);
        let (lane, scalar) = (sr.fit_rhs(&d).unwrap(), sr.fit_rhs_scalar(&d).unwrap());
        assert!(lane.iter().any(|&x| x != 0.0), "{xyz}/{orb}: vacuous d_P");
        assert_eq!(
            bits(lane.as_slice().unwrap()),
            bits(scalar.as_slice().unwrap()),
            "{xyz}/{orb}: d_P"
        );
        let mut jl = Array2::<f64>::zeros((s.nao, s.nao));
        let mut js = Array2::<f64>::zeros((s.nao, s.nao));
        sr.coulomb_from_coefficients(&c, &mut jl).unwrap();
        sr.coulomb_scalar(&c, &mut js).unwrap();
        assert!(jl.iter().any(|&x| x != 0.0), "{xyz}/{orb}: vacuous J");
        assert_eq!(
            bits(jl.as_slice().unwrap()),
            bits(js.as_slice().unwrap()),
            "{xyz}/{orb}: J(c)"
        );
    }
}

/// The far-field terms (large-T Boys form, one derivative tensor per
/// primitive pair and aux atom) change d_P and J(c), so the test is not
/// vacuous, and stay within their bound: `FAR_EPS` relative per term, summed
/// as `|terms|` (the `_abs` passes), plus both runs' rounding.
#[test]
fn split_rij_far_field_is_within_its_bound_of_the_exact_kernel() {
    let s = system("alkane_10.xyz", "def2-svp");
    let mut sr = SplitRiJ::new(Operator::coulomb(), &s.obs, &s.aux, 1e-12).unwrap();
    let k = sr.rounding_depth();
    let d = test_density(s.nao);
    let c = Array1::from_shape_fn(s.naux, |p| ((p * 13 % 17) as f64 - 8.0) * 0.01);
    let d_far = sr.fit_rhs(&d).unwrap();
    let mut j_far = Array2::<f64>::zeros((s.nao, s.nao));
    sr.coulomb_from_coefficients(&c, &mut j_far).unwrap();
    sr.set_far_field(false);
    let d_exact = sr.fit_rhs(&d).unwrap();
    let mut j_exact = Array2::<f64>::zeros((s.nao, s.nao));
    sr.coulomb_from_coefficients(&c, &mut j_exact).unwrap();
    let s_d = sr.fit_rhs_abs(&d).unwrap();
    let mut s_j = Array2::<f64>::zeros((s.nao, s.nao));
    sr.coulomb_abs(&c, &mut s_j).unwrap();
    let scale = 2.0 * FAR_EPS + 2.0 * gamma(k);
    let changed_d = d_far.iter().zip(&d_exact).filter(|(a, b)| a != b).count();
    let changed_j = j_far.iter().zip(&j_exact).filter(|(a, b)| a != b).count();
    assert!(changed_d > 0 && changed_j > 0, "far field never engaged");
    let tol_d = s_d.mapv(|x| scale * x + FLOOR);
    let wd = check("d_P far", d_far.iter(), d_exact.iter(), tol_d.iter());
    let tol_j = s_j.mapv(|x| scale * x + FLOOR);
    let wj = check("J(c) far", j_far.iter(), j_exact.iter(), tol_j.iter());
    eprintln!(
        "alkane_10/def2-svp: far field changed {changed_d} of {} d_P and {changed_j} of {} J; \
         worst |diff|/tol d {wd:.2e} J {wj:.2e}",
        d_far.len(),
        j_far.len()
    );
}

/// Fixed work partition, fixed summation order: the bits of d_P, J(c) and the
/// full build do not depend on the rayon pool size.
#[test]
fn split_rij_is_bit_identical_across_thread_counts() {
    let s = system("benzene.xyz", "def2-svp");
    let mut sr = SplitRiJ::new(Operator::coulomb(), &s.obs, &s.aux, 1e-12).unwrap();
    let d = test_density(s.nao);
    let c = Array1::from_shape_fn(s.naux, |p| ((p * 13 % 17) as f64 - 8.0) * 0.01);
    let run = |sr: &mut SplitRiJ, n: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap();
        pool.install(|| {
            let d_p = sr.fit_rhs(&d).unwrap();
            let mut jc = Array2::<f64>::zeros((s.nao, s.nao));
            sr.coulomb_from_coefficients(&c, &mut jc).unwrap();
            let mut j = Array2::<f64>::zeros((s.nao, s.nao));
            sr.build(&d, &mut j).unwrap();
            (d_p, jc, j)
        })
    };
    let bits = |a: &[f64]| a.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
    let (d1, jc1, j1) = run(&mut sr, 1);
    for n in [2, 3, 6] {
        let (dn, jcn, jn) = run(&mut sr, n);
        assert_eq!(
            bits(d1.as_slice().unwrap()),
            bits(dn.as_slice().unwrap()),
            "d_P, {n} threads"
        );
        assert_eq!(
            bits(jc1.as_slice().unwrap()),
            bits(jcn.as_slice().unwrap()),
            "J(c), {n} threads"
        );
        assert_eq!(
            bits(j1.as_slice().unwrap()),
            bits(jn.as_slice().unwrap()),
            "J, {n} threads"
        );
    }
}

/// The (shell pair, aux shell) screen drops only elements with
/// `|(P|μν)| <= Q_μν Q_P <= Q_AB Q_K <= thresh` (Schwarz), so
/// `|d_P(thresh) - d_P(0)| <= thresh · Σ_μν |D_μν|` plus both runs' rounding.
#[test]
fn split_rij_screen_error_obeys_the_schwarz_bound() {
    let s = system("benzene.xyz", "def2-svp");
    let op = Operator::coulomb();
    let thresh = 1e-8;
    let full = SplitRiJ::new(op, &s.obs, &s.aux, 0.0).unwrap();
    let scr = SplitRiJ::new(op, &s.obs, &s.aux, thresh).unwrap();
    let (kept, total) = scr.block_counts();
    assert!(
        kept < total,
        "nothing screened at {thresh}: the test would be vacuous"
    );
    let d = test_density(s.nao);
    let d0 = full.fit_rhs(&d).unwrap();
    let d1 = scr.fit_rhs(&d).unwrap();
    let a0 = full.fit_rhs_abs(&d).unwrap();
    let a1 = scr.fit_rhs_abs(&d).unwrap();
    let sum_d: f64 = d.iter().map(|x| x.abs()).sum();
    let tol: Vec<f64> = a0
        .iter()
        .zip(a1.iter())
        .map(|(x, y)| {
            thresh * sum_d + gamma(full.rounding_depth()) * x + gamma(scr.rounding_depth()) * y
        })
        .collect();
    check("screened d_P", d1.iter(), d0.iter(), tol.iter());
    eprintln!("benzene/def2-svp screen {thresh}: kept {kept} of {total} blocks");
}

/// A zero density gives an exactly zero J (poisoned output fully overwritten).
#[test]
fn split_rij_zero_density_gives_exactly_zero() {
    let s = system("water.xyz", "cc-pvdz");
    let mut sr = SplitRiJ::new(Operator::coulomb(), &s.obs, &s.aux, 1e-12).unwrap();
    let mut j = Array2::<f64>::from_elem((s.nao, s.nao), 1.0);
    sr.build(&Array2::zeros((s.nao, s.nao)), &mut j).unwrap();
    assert!(j.iter().all(|&x| x == 0.0));
}
