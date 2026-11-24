//! Exactness anchors for the in-core four-centre J/K store
//! (`ferric_scf::incore_jk`).
//!
//! # The bar
//!
//! The store holds the same libint2 integrals the direct builder computes
//! (same engine, same precision, every quartet at threshold 0), so in-core and
//! direct J/K differ only by summation order. For one element, a sum of `L`
//! products has a floating-point error at most `γ_L · Σ|terms|`,
//! `γ_L = L u / (1 − L u)`, `u = 2^-53`, and `γ_a + γ_b ≤ γ_{a+b}`. The
//! direct builder accumulates at most `n²` products into a group partial and
//! then sums `G ≤ nsh²` group partials onto the output: error
//! `≤ γ_{n² + nsh²} · A`, with `A = Σ |(μν|λσ)| |D_λσ|` (J) or
//! `Σ |(μλ|νσ)| |D_λσ|` (K).
//!
//! * In-core J sums `npair + 1 ≤ n²` products (plus the pair-weight
//!   addition), each term once: `|J_incore − J_direct| ≤ 2 γ_{n² + nsh² + 1} · A`.
//! * In-core K is a row part and a column part, each a sum of at most `n²`
//!   products, added, less at most `n + 1` correction products. The terms
//!   with `pair(a, b) = pair(m, l)` enter three times (row part, column part,
//!   correction), so the summed magnitudes are at most `3A`, and the in-core
//!   error is at most `γ_{2n² + n + 2} · 3A ≤ γ_{6n² + 3n + 6} · A`. Adding
//!   the direct error, `|K_incore − K_direct| ≤ 2 γ_{L_K} · A` with
//!   `L_K = 7n² + nsh² + 3n + 6` (a factor 2 to spare).
//!
//! `A` is computed element by element from the store's absolute-value copy
//! and `|D|` (on that copy the correction removes exactly the doubled terms,
//! so it returns the true `A`). Each
//! test asserts the bar per element and reports the largest `|diff| / bar`
//! (rounding errors add like a random walk, so the ratio sits far below 1).
//!
//! # Artifact hypothesis
//!
//! A wrong index pattern (K contracted as J), a dropped pair weight or a
//! missed symmetry image changes elements by O(|X|), many orders above the
//! bar; a correct store differs from direct by O(√L u A). The two cannot be
//! confused, and each mutation is checked to fail here.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::incore_jk::{EriMode, IncoreEri};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const U: f64 = f64::EPSILON / 2.0;

fn testdata(rel: &str) -> String {
    format!("{}/../../testdata/{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn setup(xyz: &str, basis_name: &str) -> (Molecule, PreparedBasis) {
    let mol = Molecule::load_xyz(&testdata(&format!("molecules/{xyz}"))).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    (mol, prep)
}

fn store(prep: &PreparedBasis) -> IncoreEri {
    let ctx = ParallelContext::default();
    let plan = IncoreEri::plan(prep, usize::MAX / 4).unwrap();
    let charge = plan.commit().unwrap();
    IncoreEri::build(&ctx, prep, Operator::coulomb(), &plan, charge).unwrap()
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn random_density(n: usize, seed: u64, symmetric: bool) -> Array2<f64> {
    let mut s = seed;
    let mut d = Array2::from_shape_fn((n, n), |_| lcg(&mut s));
    if symmetric {
        d = 0.5 * (&d + &d.t());
    }
    d
}

/// Direct `(J, K)` of `d` at threshold 0 on fresh bounds (no store).
fn direct_jk(prep: &PreparedBasis, d: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), prep).unwrap();
    assert!(bounds.incore.get().is_none());
    // Fresh bounds: no SCF recorded a contraction rotation on them, so the
    // walk is the unrotated one the rounding bar is about.
    let n = d.nrows();
    let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    ferric_scf::rhf::build_jk(&ctx, prep, &bounds, 0.0, d, &mut j, &mut k).unwrap();
    (j, k)
}

/// Largest `|a − b| / bar` over elements, panicking on any element above its
/// bar. `bar = 2 γ_L · A` elementwise.
fn check_bar(what: &str, a: &Array2<f64>, b: &Array2<f64>, abs: &Array2<f64>, l: usize) -> f64 {
    let lu = l as f64 * U;
    let gamma = lu / (1.0 - lu);
    let mut worst = 0.0f64;
    for ((idx, &x), &y) in a.indexed_iter().zip(b.iter()) {
        let bar = 2.0 * gamma * abs[idx];
        let diff = (x - y).abs();
        assert!(
            diff <= bar,
            "{what}{idx:?}: |incore - direct| = {diff:e} above the rounding bar {bar:e} \
             (incore {x:e}, direct {y:e})"
        );
        if bar > 0.0 {
            worst = worst.max(diff / bar);
        }
    }
    worst
}

/// Compare the store's J and K of each density with the direct builder's.
fn anchor(label: &str, prep: &PreparedBasis, eri: &IncoreEri, ds: &[Array2<f64>]) {
    let n = prep.nbasis();
    let nsh2 = prep.nshells() * prep.nshells();
    let l = n * n + nsh2 + 1;
    let l_k = 7 * n * n + nsh2 + 3 * n + 6;
    let refs: Vec<&Array2<f64>> = ds.iter().collect();
    let (js, ks) = eri.jk_multi(&refs, &refs).unwrap();
    let abs = eri.abs_copy().unwrap();
    let dabs: Vec<Array2<f64>> = ds.iter().map(|d| d.mapv(f64::abs)).collect();
    let dabs_refs: Vec<&Array2<f64>> = dabs.iter().collect();
    let (ja, ka) = abs.jk_multi(&dabs_refs, &dabs_refs).unwrap();
    let (mut wj, mut wk) = (0.0f64, 0.0f64);
    for (i, d) in ds.iter().enumerate() {
        let (jd, kd) = direct_jk(prep, d);
        wj = wj.max(check_bar(
            &format!("{label} J[{i}]"),
            &js[i],
            &jd,
            &ja[i],
            l,
        ));
        wk = wk.max(check_bar(
            &format!("{label} K[{i}]"),
            &ks[i],
            &kd,
            &ka[i],
            l_k,
        ));
        // The comparison must not be vacuous: the matrices are O(1).
        let scale = kd.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            scale > 1e-3,
            "{label}: K[{i}] is {scale:e}, a vacuous comparison"
        );
    }
    eprintln!(
        "{label}: n = {n}, {} densities, worst |diff|/bar J {wj:.3e}, K {wk:.3e}",
        ds.len()
    );
}

#[test]
fn incore_matches_direct_on_water_symmetric_nonsymmetric_and_scf_densities() {
    let (mol, prep) = setup("water.xyz", "cc-pvdz");
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let mut cfg = RhfConfig::default();
    cfg.eri = EriMode::Direct;
    let scf = solve_rhf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg).unwrap();
    let eri = store(&prep);
    anchor(
        "water/cc-pVDZ",
        &prep,
        &eri,
        &[
            scf.density_total.clone(),
            random_density(n, 1, true),
            random_density(n, 2, false),
        ],
    );
}

#[test]
fn incore_matches_direct_on_a_40_density_block() {
    let (_, prep) = setup("water.xyz", "cc-pvdz");
    let n = prep.nbasis();
    let eri = store(&prep);
    let ds: Vec<Array2<f64>> = (0..40)
        .map(|i| random_density(n, 100 + i, i % 2 == 0))
        .collect();
    anchor("water/cc-pVDZ block", &prep, &eri, &ds);
    // A block call and forty single calls agree: J bit for bit (the same
    // row-block dot products), K within the same rounding bar (a one-column
    // contraction takes OpenBLAS's GEMV path, whose summation order differs
    // from the many-column GEMM's in the last bits).
    let refs: Vec<&Array2<f64>> = ds.iter().collect();
    let (jb, kb) = eri.jk_multi(&refs, &refs).unwrap();
    let abs = eri.abs_copy().unwrap();
    let l_k = 7 * n * n + prep.nshells() * prep.nshells() + 3 * n + 6;
    for (i, d) in ds.iter().enumerate() {
        let (j1, k1) = eri.jk_multi(&[d], &[d]).unwrap();
        assert_eq!(bits(&j1[0]), bits(&jb[i]), "J[{i}] block vs single");
        let da = d.mapv(f64::abs);
        let ka = abs.exchange(&[&da]).unwrap();
        check_bar(
            &format!("K[{i}] block vs single"),
            &k1[0],
            &kb[i],
            &ka[0],
            l_k,
        );
    }
}

#[test]
fn incore_matches_direct_on_benzene() {
    let (_, prep) = setup("benzene.xyz", "cc-pvdz");
    let n = prep.nbasis();
    let eri = store(&prep);
    anchor(
        "benzene/cc-pVDZ",
        &prep,
        &eri,
        &[random_density(n, 7, true), random_density(n, 8, false)],
    );
}

fn bits(a: &Array2<f64>) -> Vec<u64> {
    a.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn incore_bits_do_not_depend_on_the_thread_count() {
    let (_, prep) = setup("water.xyz", "cc-pvdz");
    let n = prep.nbasis();
    let ds = [random_density(n, 3, true), random_density(n, 4, false)];
    let refs: Vec<&Array2<f64>> = ds.iter().collect();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let eri = store(&prep);
                let (j, k) = eri.jk_multi(&refs, &refs).unwrap();
                let mut out: Vec<u64> = j.iter().flat_map(bits).collect();
                out.extend(k.iter().flat_map(bits));
                out
            })
    };
    let one = run(1);
    for t in [2, 6] {
        assert!(
            run(t) == one,
            "in-core J/K bits differ between 1 and {t} threads"
        );
    }
}

/// `auto` on water attaches the store, and the SCF on it reaches the direct
/// route's energy within the SCF's own convergence: both converge the same
/// functional, so the energies differ by the integral-screening floor and the
/// density convergence, far below 1e-9 Ha.
#[test]
fn auto_attaches_the_store_and_reproduces_the_direct_energy() {
    let (mol, prep) = setup("water.xyz", "cc-pvdz");
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let solve = |eri: EriMode| {
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let mut cfg = RhfConfig::default();
        cfg.eri = eri;
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        // The certificate names the route, and not as an approximation.
        let c = ferric_scf::certificate::scf_certificate(&mol, &r, &cfg);
        assert!(c.approximations.is_empty(), "{:?}", c.approximations);
        let named: Vec<&str> = c.routes.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(
            named,
            if eri == EriMode::Direct {
                vec![]
            } else {
                vec!["eri"]
            }
        );
        (r, bounds.incore.get().is_some(), cfg.eri_route.get())
    };
    let (direct, d_store, d_route) = solve(EriMode::Direct);
    let (auto, a_store, a_route) = solve(EriMode::Auto);
    let (incore, i_store, _) = solve(EriMode::Incore);
    assert!(!d_store && d_route.is_none());
    assert!(
        a_store && i_store,
        "water/cc-pVDZ fits a quarter of any budget"
    );
    assert_eq!(a_route.as_ref().map(|r| r.nbasis), Some(prep.nbasis()));
    assert_eq!(auto.energy.to_bits(), incore.energy.to_bits());
    let de = (auto.energy - direct.energy).abs();
    eprintln!(
        "water RHF: direct {:.12} auto {:.12} |dE| {de:e}",
        direct.energy, auto.energy
    );
    assert!(de < 1e-9, "|E_incore - E_direct| = {de:e}");
}

/// The store follows the geometry: a two-step optimize on `incore` gives the
/// direct route's energies at every evaluated point (a stale store would
/// give the first geometry's J/K at the second, an error of order the
/// first step's energy change, ~1e-3 Ha here).
#[test]
fn a_two_step_optimize_rebuilds_the_store_per_geometry() {
    use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};
    let mol = Molecule::load_xyz(&testdata("molecules/water.xyz")).unwrap();
    let ctx = ParallelContext::default();
    let opt = OptimizeConfig {
        max_steps: 2,
        ..Default::default()
    };
    let run = |eri: EriMode| {
        let mut cfg = RhfConfig::default();
        cfg.eri = eri;
        optimize_geometry(&ctx, &mol, "6-31g", Operator::coulomb(), &cfg, &opt).unwrap()
    };
    let direct = run(EriMode::Direct);
    let incore = run(EriMode::Incore);
    assert_eq!(direct.energy_trace.len(), incore.energy_trace.len());
    assert!(direct.energy_trace.len() >= 2);
    let first_step = (direct.energy_trace[1] - direct.energy_trace[0]).abs();
    assert!(
        first_step > 1e-5,
        "the step must move the energy ({first_step:e})"
    );
    for (i, (a, b)) in direct
        .energy_trace
        .iter()
        .zip(&incore.energy_trace)
        .enumerate()
    {
        let de = (a - b).abs();
        eprintln!("optimize point {i}: direct {a:.12} incore {b:.12} |dE| {de:e}");
        assert!(de < 1e-8, "point {i}: |E_incore - E_direct| = {de:e}");
    }
}

/// Finite-difference frequencies rebuild the SCF at every displaced geometry;
/// the in-core route must give the direct route's frequencies.
#[test]
fn fd_frequencies_rebuild_the_store_per_displacement() {
    use ferric_scf::frequencies::{harmonic_frequencies, FrequencyConfig, HessianMethod};
    let mol = Molecule::load_xyz(&testdata("molecules/water.xyz")).unwrap();
    let ctx = ParallelContext::default();
    let fc = FrequencyConfig {
        hessian: HessianMethod::FiniteDifference,
        ..Default::default()
    };
    // Core guess: the SAD guess runs free-atom SCFs of its own, and a
    // store-reuse defect that those SCFs happened to mask would go unseen.
    let run = |eri: EriMode| {
        let mut cfg = RhfConfig::default().with_guess(ferric_scf::options::Guess::Core);
        cfg.eri = eri;
        harmonic_frequencies(&ctx, &mol, "sto-3g", Operator::coulomb(), &cfg, &fc).unwrap()
    };
    let direct = run(EriMode::Direct);
    let incore = run(EriMode::Incore);
    for (a, b) in direct.frequencies.iter().zip(&incore.frequencies) {
        eprintln!("FD frequency: direct {a:.6} incore {b:.6} cm-1");
        assert!((a - b).abs() < 1e-2, "{a} vs {b} cm-1");
    }
}

#[test]
fn incore_on_a_fitted_route_is_an_error_and_auto_clears_the_slot() {
    let (mol, prep) = setup("water.xyz", "cc-pvdz");
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut cfg = RhfConfig::default();
    cfg.eri = EriMode::Incore;
    assert!(
        solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg)
            .unwrap()
            .converged
    );
    assert!(bounds.incore.get().is_some());
    cfg.df_j_aux = Some("def2-universal-jkfit".into());
    cfg.df_k_aux = Some("def2-universal-jkfit".into());
    let e = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap_err();
    assert!(e.to_string().contains("eri = \"incore\" needs"), "{e}");
    cfg.eri = EriMode::Auto;
    assert!(
        solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg)
            .unwrap()
            .converged
    );
    assert!(
        bounds.incore.get().is_none(),
        "a fitted SCF clears the slot"
    );
    assert!(cfg.eri_route.get().is_none());
}

#[test]
fn eri_mode_parse_is_strict() {
    for (s, m) in [
        ("auto", EriMode::Auto),
        ("direct", EriMode::Direct),
        ("incore", EriMode::Incore),
    ] {
        assert_eq!(EriMode::parse_config_str(s), Ok(m));
        assert_eq!(m.as_str(), s);
    }
    for bad in ["", "Auto", "in-core", "dense", "exact"] {
        assert!(EriMode::parse_config_str(bad).is_err(), "{bad}");
    }
}

/// `eri = "incore"` names the exact stage: the fitted DF-guess pre-stage in
/// front of it (the CLI's default for correlated kinds) runs on its own
/// route and the exact stage still takes the store.
#[test]
fn incore_with_a_df_guess_pre_stage_applies_to_the_exact_stage() {
    let (mol, prep) = setup("water.xyz", "cc-pvdz");
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut cfg = RhfConfig::default();
    cfg.eri = EriMode::Incore;
    let r = ferric_scf::ladder::solve_rhf_with_df_guess(&ctx, &mol, &prep, op, &bounds, &cfg, None);
    assert!(
        r.is_ok(),
        "the DF-guess SCF must run on eri = incore: {:?}",
        r.err()
    );
    let r = r.unwrap();
    assert!(r.result.converged);
    assert!(bounds.incore.get().is_some());
    assert_eq!(cfg.eri_route.get().map(|r| r.nbasis), Some(prep.nbasis()));
}
