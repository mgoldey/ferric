//! Exactness anchors for the direct J/K builder's blocked digestion and
//! streamed in-order fold (`DirectJK::build`, `DirectJK::build_uhf`).
//!
//! # What changed, and what can move
//!
//! The builder computes the same libint2 quartets under the same screen as
//! before. Two things differ:
//!
//! * **Digestion.** Each computed block `(s1 s2|s3 s4)` is contracted into
//!   small shell-block tiles (one dot product or axpy per integral row) that
//!   are then added once into the group partial, instead of adding every
//!   integral's eight J and eight K images one at a time. The Coulomb density
//!   enters as `D[λσ] + D[σλ]` (one rounding) where it entered as two
//!   products. The set of products in each element is unchanged; only the
//!   bracketing of their sum moves. Blocks under 16 integrals keep the
//!   per-element scatter, so a cc-pVDZ system exercises both paths.
//! * **Fold.** Group partials are added in the same order `0, 1, …` as the
//!   banded fold, so the fold changes no bit (tested in `reduce.rs`).
//!
//! # The bar
//!
//! For a sum of products evaluated in any bracketing, the error is at most
//! `γ_h · Σ|terms|`, `γ_h = h u / (1 − h u)`, `u = 2^-53`, with `h` the
//! largest number of roundings on any one term's path to the result. A J or K
//! element of the direct builder gathers at most `n²` products (`n` basis
//! functions) into its group partial and is then summed over `G ≤ nsh²`
//! groups; each product carries one rounding for the multiplication and, for
//! the symmetrized Coulomb density, one for `D[λσ] + D[σλ]`. So
//! `h ≤ n² + nsh² + 2` and `|X_direct − X| ≤ γ_{n² + nsh² + 2} · A`, with
//! `A = Σ |(μν|λσ)| |D_λσ|` (J) or `Σ |(μλ|νσ)| |D_λσ|` (K) built from the
//! absolute-value integrals.
//!
//! The reference is the in-core store (`incore_jk`), an independent
//! construction from the same libint2 integrals at threshold 0. Its errors
//! are bounded in `tests/incore_jk.rs`: `γ_{n² + 2} · A` for J and
//! `γ_{6n² + 3n + 6} · A` for K. Adding the two (`γ_a + γ_b ≤ γ_{a+b}`):
//!
//! * `|J_direct − J_incore| ≤ γ_{L_J} · A`, `L_J = 2n² + nsh² + 4`;
//! * `|K_direct − K_incore| ≤ γ_{L_K} · A`, `L_K = 7n² + nsh² + 3n + 8`.
//!
//! Each element is checked against `2 γ_L A` (a factor 2 to spare), and the
//! largest `|diff| / bar` is printed: rounding errors add like a random walk,
//! so it sits orders of magnitude below 1.
//!
//! # Artifact hypothesis
//!
//! A dropped symmetry image, a transposed image added from the wrong tile, a
//! symmetrized density missing its transpose, or a skipped quartet changes
//! elements by O(|X|) — many orders above the bar — while a correct kernel
//! differs from the reference by O(√L u A). The two cannot be confused. The
//! non-symmetric densities reach the branch that contracts a transposed image
//! from its own density tile; symmetric ones reach the branch that adds the
//! direct tile transposed.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_jk::{DirectJK, GcRotate};
use ferric_scf::incore_jk::IncoreEri;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const U: f64 = f64::EPSILON / 2.0;

fn testdata(rel: &str) -> String {
    format!("{}/../../testdata/{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn setup(xyz: &str, basis_name: &str, mult: usize) -> (Molecule, PreparedBasis) {
    let text = std::fs::read_to_string(testdata(&format!("molecules/{xyz}"))).unwrap();
    let mol = Molecule::parse_xyz(&text, 0, mult).unwrap();
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

/// A symmetric density except on the rows and columns of one shell (`i ∈
/// [lo, hi)`), where `D[i, j] ≠ D[j, i]`: some blocks are their mirror's
/// transpose and some are not, so the digestion's per-block choice between
/// the two transposed-image branches matters block by block.
fn partly_symmetric_density(n: usize, seed: u64, lo: usize, hi: usize) -> Array2<f64> {
    let mut d = random_density(n, seed, true);
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    for i in lo..hi {
        for j in 0..n {
            d[(i, j)] += 0.5 * lcg(&mut s);
        }
    }
    d
}

fn bits(a: &Array2<f64>) -> Vec<u64> {
    a.iter().map(|v| v.to_bits()).collect()
}

/// The two bar lengths of the module doc for this basis.
fn bar_lengths(prep: &PreparedBasis) -> (usize, usize) {
    let n = prep.nbasis();
    let nsh2 = prep.nshells() * prep.nshells();
    (2 * n * n + nsh2 + 4, 7 * n * n + nsh2 + 3 * n + 8)
}

/// Largest `|a − b| / bar` over elements, panicking on any element above its
/// bar `2 γ_L · abs`; also refuses a vacuous comparison (an `O(1)` matrix is
/// expected).
fn check_bar(what: &str, a: &Array2<f64>, b: &Array2<f64>, abs: &Array2<f64>, l: usize) -> f64 {
    let lu = l as f64 * U;
    let gamma = lu / (1.0 - lu);
    let mut worst = 0.0f64;
    for ((idx, &x), &y) in a.indexed_iter().zip(b.iter()) {
        let bar = 2.0 * gamma * abs[idx];
        let diff = (x - y).abs();
        assert!(
            diff <= bar,
            "{what}{idx:?}: |direct - incore| = {diff:e} above the rounding bar {bar:e} \
             (direct {x:e}, incore {y:e})"
        );
        if bar > 0.0 {
            worst = worst.max(diff / bar);
        }
    }
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        scale > 1e-3,
        "{what}: reference is {scale:e}, a vacuous comparison"
    );
    worst
}

/// Closed-shell `DirectJK::build` at threshold 0 against the in-core store.
fn anchor_closed(label: &str, prep: &PreparedBasis, ds: &[Array2<f64>]) {
    let n = prep.nbasis();
    let (l_j, l_k) = bar_lengths(prep);
    let eri = store(prep);
    let abs = eri.abs_copy();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), prep).unwrap();
    // The bar is the rounding analysis of the walk on the basis the store
    // holds, so the walk runs on it: no internal contraction rotation (that
    // route is checked against this one in `tests/gc_rotation.rs`).
    let mut djk =
        DirectJK::new(&ctx, prep, &bounds, 0.0, usize::MAX / 4).with_gc_rotation(GcRotate::Off);
    assert!(
        djk.incremental_pays(),
        "the builder must be on the direct route"
    );
    let (mut wj, mut wk) = (0.0f64, 0.0f64);
    for (i, d) in ds.iter().enumerate() {
        let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        djk.build(d, &mut j, &mut k).unwrap();
        let (jr, kr) = eri.jk_multi(&[d], &[d]);
        let da = d.mapv(f64::abs);
        let (ja, ka) = abs.jk_multi(&[&da], &[&da]);
        wj = wj.max(check_bar(
            &format!("{label} J[{i}]"),
            &j,
            &jr[0],
            &ja[0],
            l_j,
        ));
        wk = wk.max(check_bar(
            &format!("{label} K[{i}]"),
            &k,
            &kr[0],
            &ka[0],
            l_k,
        ));
    }
    eprintln!(
        "{label}: n = {n}, {} densities, worst |diff|/bar J {wj:.3e}, K {wk:.3e}",
        ds.len()
    );
}

#[test]
fn blocked_digest_matches_incore_on_water_scf_symmetric_and_nonsymmetric_densities() {
    let (mol, prep) = setup("water.xyz", "cc-pvdz", 1);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let scf = ferric_scf::rhf::solve_rhf(
        &ctx,
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    assert!(scf.converged);
    // The first p shell: its blocks with p and d shells take the blocked path.
    let sh = prep.shell_dims().iter().position(|&m| m == 3).unwrap();
    let (p_lo, p_hi) = (prep.shell_offsets()[sh], prep.shell_offsets()[sh] + 3);
    anchor_closed(
        "water/cc-pVDZ",
        &prep,
        &[
            scf.density_total.clone(),
            random_density(n, 1, true),
            random_density(n, 2, false),
            partly_symmetric_density(n, 3, p_lo, p_hi),
        ],
    );
}

#[test]
fn blocked_digest_matches_incore_on_benzene() {
    let (_, prep) = setup("benzene.xyz", "cc-pvdz", 1);
    let n = prep.nbasis();
    anchor_closed(
        "benzene/cc-pVDZ",
        &prep,
        &[random_density(n, 7, true), random_density(n, 8, false)],
    );
}

/// Open shell: `build_uhf` on the converged OH doublet UHF densities and on
/// a random unequal pair, against the in-core `spin_jk` (J of `D_α + D_β`,
/// `K` of each spin).
#[test]
fn blocked_digest_matches_incore_on_open_shell_oh() {
    let (mol, prep) = setup("oh.xyz", "cc-pvdz", 2);
    let n = prep.nbasis();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let scf =
        ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &RhfConfig::default()).unwrap();
    assert!(scf.converged);
    let pairs = [
        (
            scf.density_alpha.clone(),
            scf.density_beta.clone().expect("UHF beta density"),
        ),
        (random_density(n, 11, true), random_density(n, 12, false)),
    ];
    let (l_j, l_k) = bar_lengths(&prep);
    let eri = store(&prep);
    let abs = eri.abs_copy();
    // Fresh bounds: the SCF's bounds carry its in-core store, which
    // `build_uhf` would answer from instead of computing integrals.
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    // Unrotated walk, as in `anchor_closed`.
    let mut djk =
        DirectJK::new(&ctx, &prep, &bounds, 0.0, usize::MAX / 4).with_gc_rotation(GcRotate::Off);
    assert!(
        djk.incremental_pays(),
        "the builder must be on the direct route"
    );
    let (mut wj, mut wk) = (0.0f64, 0.0f64);
    for (i, (da, db)) in pairs.iter().enumerate() {
        assert!(bits(da) != bits(db), "pair {i}: the spins must differ");
        let dt = da + db;
        let (mut j, mut ka, mut kb) = (
            Array2::zeros((n, n)),
            Array2::zeros((n, n)),
            Array2::zeros((n, n)),
        );
        djk.build_uhf(&dt, da, db, &mut j, &mut ka, &mut kb)
            .unwrap();
        let (jr, kr) = eri.spin_jk([da, db], true);
        let kr = kr.expect("exchange requested");
        let (aa, ab) = (da.mapv(f64::abs), db.mapv(f64::abs));
        let at = &aa + &ab;
        let (ja, _) = abs.jk_multi(&[&at], &[]);
        let (_, kabs) = abs.jk_multi(&[], &[&aa, &ab]);
        wj = wj.max(check_bar(&format!("OH J[{i}]"), &j, &jr, &ja[0], l_j));
        wk = wk.max(check_bar(
            &format!("OH Ka[{i}]"),
            &ka,
            &kr[0],
            &kabs[0],
            l_k,
        ));
        wk = wk.max(check_bar(
            &format!("OH Kb[{i}]"),
            &kb,
            &kr[1],
            &kabs[1],
            l_k,
        ));
    }
    eprintln!("OH/cc-pVDZ UHF: n = {n}, worst |diff|/bar J {wj:.3e}, K {wk:.3e}");
}

/// The builders' bits do not depend on the worker count (1, 2, 6 threads) at
/// the production threshold, closed and open shell.
#[test]
fn direct_jk_bits_do_not_depend_on_the_thread_count() {
    let (_, prep) = setup("benzene.xyz", "cc-pvdz", 1);
    let n = prep.nbasis();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let d = random_density(n, 21, true);
    let (da, db) = (random_density(n, 22, true), random_density(n, 23, true));
    let dt = &da + &db;
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let ctx = ParallelContext::default();
                // A small band budget forces the fold window below the group
                // count, so workers wait on it and fold out of claim order.
                let mut djk = DirectJK::new(&ctx, &prep, &bounds, 1e-12, 40 * n * n * 8);
                let z = || Array2::<f64>::zeros((n, n));
                let (mut j, mut k) = (z(), z());
                djk.build(&d, &mut j, &mut k).unwrap();
                let (mut ju, mut ka, mut kb) = (z(), z(), z());
                djk.build_uhf(&dt, &da, &db, &mut ju, &mut ka, &mut kb)
                    .unwrap();
                [j, k, ju, ka, kb]
                    .iter()
                    .flat_map(bits)
                    .collect::<Vec<u64>>()
            })
    };
    let one = run(1);
    for t in [2, 6] {
        assert!(
            run(t) == one,
            "direct J/K bits differ between 1 and {t} threads"
        );
    }
}
