//! RESEARCH SPIKE tests for the pivoted-Cholesky sparse-PDEP seed
//! (`--features spike-cholesky-seed`; the file does not build without it).
//!
//! Ported from the 2026-07-16 spike (`tests/cholesky_screened_rpa.rs`, source
//! commit 14386e506). Record and verdict:
//! `reference/spikes/cholesky-vs-boys-pdep-seed.md`.
//!
//! Mutation ledger (2026-10-23, each run against this whole file, reverted):
//!   M-A `CholeskyScreened` dispatch arm returns Err -> the 3 run_pdep_rpa
//!       tests fail (proves the config reaches the Cholesky code).
//!   M-B no semicanonicalization in build_screened_bov -> the coincidence
//!       test and the production full-retention test fail (Cholesky 6.7 µHa
//!       vs Boys 0.29 µHa off Dense under the mutant).
//!   M-C pivoted Cholesky without the Schur-complement update -> all 7 fail.
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --features spike-cholesky-seed \
//!     --test spike_cholesky_seed -- --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::overlap;
use ferric_integrals::operator::Operator;
use ferric_rpa::cholesky_localize::{
    build_screened_bov_cholesky, cholesky_localize_occupied, pivoted_cholesky,
};
use ferric_rpa::config::{Chi0Sparsity, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::{s, Array2};

fn setup(xyz: &str) -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(xyz).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, op, rhf)
}

fn water() -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
    setup("../../testdata/molecules/water.xyz")
}

/// Tight solver settings of the spike's correctness tests: full rank, 1e-10.
fn tight_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 40,
            u0: 0.5,
        },
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver_conv_thresh: 1e-10,
        ..Default::default()
    }
}

/// The spike's production solver settings (`sparse_scaling.rs`): PDEP
/// truncation 1e-4, eigensolver convergence 1e-6. This is the regime where the
/// spike measured its 24-63 µHa Cholesky error.
fn production_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        trunc_thresh: 1e-4,
        eigensolver_conv_thresh: 1e-6,
        ..tight_cfg()
    }
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

fn e_rpa(
    sys: &(Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult),
    cfg: &PdepRpaConfig,
    sparsity: Chi0Sparsity,
) -> f64 {
    let (mol, obs, dfbs, op, rhf) = sys;
    let mut cfg = cfg.clone();
    cfg.chi0_sparsity = sparsity;
    run_pdep_rpa(mol, obs, dfbs, *op, rhf, &cfg).unwrap().e_rpa
}

const DENSE: Chi0Sparsity = Chi0Sparsity::Dense;
const fn boys(thresh: f64) -> Chi0Sparsity {
    Chi0Sparsity::BoysScreened {
        thresh,
        dist_cutoff: f64::INFINITY,
    }
}
const fn chol(thresh: f64) -> Chi0Sparsity {
    Chi0Sparsity::CholeskyScreened {
        thresh,
        dist_cutoff: f64::INFINITY,
    }
}

/// The factorization alone: a random rank-3 PSD 8x8 matrix is reconstructed
/// and its rank detected.
#[test]
fn pivoted_cholesky_detects_rank_and_reconstructs() {
    // Deterministic, well-conditioned rank-3 factor.
    let g = Array2::from_shape_fn((8, 3), |(i, j)| {
        ((i * 7 + j * 3) as f64 * 0.37).sin() + if i == j { 2.0 } else { 0.0 }
    });
    let a = g.dot(&g.t());
    let (l, piv) = pivoted_cholesky(&a, 8.0 * f64::EPSILON * 10.0);
    let err = max_abs_diff(&l.dot(&l.t()), &a);
    println!(
        "rank {} pivots {:?} max|LL^T - A| = {err:.2e}",
        l.ncols(),
        piv
    );
    assert_eq!(l.ncols(), 3);
    assert!(err < 1e-12, "reconstruction error {err:.2e}");
    // Diagonal pivoting: the first pivot is the largest diagonal entry.
    let argmax = (0..8).max_by(|&i, &j| a[(i, i)].total_cmp(&a[(j, j)]));
    assert_eq!(
        Some(piv[0]),
        argmax,
        "first pivot is not the largest diagonal"
    );
}

/// Water/cc-pVDZ: rank == nocc, `c_loc c_locᵀ = D`, and the Cholesky orbitals
/// are ORTHONORMAL in the AO overlap metric (they are `C_occ U`, U orthogonal)
/// while not orthonormal in the Euclidean coefficient sense.
#[test]
fn h2o_cholesky_orbitals_reconstruct_density_and_are_s_orthonormal() {
    let (_mol, obs, _dfbs, _op, rhf) = water();
    let nocc = 5;
    let chol = cholesky_localize_occupied(&rhf, &obs, 0, nocc).unwrap();
    assert_eq!(chol.rank, nocc);

    let c_occ = rhf.mos_r().slice(s![.., 0..nocc]).to_owned();
    let d_err = max_abs_diff(&chol.c_loc.dot(&chol.c_loc.t()), &c_occ.dot(&c_occ.t()));

    let s_ao = overlap(&obs);
    let eye = Array2::<f64>::eye(nocc);
    let s_err = max_abs_diff(&chol.c_loc.t().dot(&s_ao.dot(&chol.c_loc)), &eye);
    let euclid = max_abs_diff(&chol.c_loc.t().dot(&chol.c_loc), &eye);
    println!(
        "H2O cholesky: pivots {:?}  max|dD| = {d_err:.2e}  max|C^T S C - I| = {s_err:.2e}  \
         max|C^T C - I| = {euclid:.2e}",
        chol.pivots
    );
    assert!(d_err < 1e-10, "density reconstruction {d_err:.2e}");
    assert!(s_err < 1e-10, "S-orthonormality {s_err:.2e}");
    assert!(euclid > 1e-2, "Euclidean overlap unexpectedly identity");
}

/// Spike test 2 (ported): full-rank tight solve, thresh = 0 — Cholesky-seeded
/// screening reproduces Dense.
#[test]
fn h2o_cholesky_screened_matches_dense_tight_thresh_zero() {
    let sys = water();
    let e_dense = e_rpa(&sys, &tight_cfg(), DENSE);
    let e_chol = e_rpa(&sys, &tight_cfg(), chol(0.0));
    let e_boys = e_rpa(&sys, &tight_cfg(), boys(0.0));
    println!(
        "H2O tight thresh=0: dense {e_dense:.12}  |chol-dense| {:.2e}  |boys-dense| {:.2e}",
        (e_chol - e_dense).abs(),
        (e_boys - e_dense).abs()
    );
    assert!((e_chol - e_dense).abs() < 3e-6);
}

/// Spike test 3 (ported): thresh 1e-6 with the tight solver.
#[test]
fn h2o_cholesky_screened_matches_dense_tight_thresh_1e6() {
    let sys = water();
    let (mol, obs, dfbs, op, rhf) = &sys;
    let e_dense = e_rpa(&sys, &tight_cfg(), DENSE);
    let e_chol = e_rpa(&sys, &tight_cfg(), chol(1e-6));
    let (sb, _) =
        build_screened_bov_cholesky(mol, obs, dfbs, *op, rhf, 0, 1e-6, f64::INFINITY).unwrap();
    println!(
        "H2O tight thresh=1e-6: retained {}/{}  |chol-dense| {:.2e}",
        sb.total_retained,
        sb.n_occ_loc * sb.naux,
        (e_chol - e_dense).abs()
    );
    assert!((e_chol - e_dense).abs() < 3e-6);
}

/// The spike's finding-2 regime: PRODUCTION solver settings (PDEP truncation
/// on) at thresh = 0 (100% retention, so no screening error is possible).
/// The spike measured Cholesky 24-63 µHa off Dense here on n-hexane and
/// naphthalene while Boys held < 3 µHa. Boys is measured alongside so the
/// two seeds are compared on the same current code.
#[test]
fn h2o_cholesky_vs_boys_production_solver_full_retention() {
    let sys = water();
    let e_dense = e_rpa(&sys, &production_cfg(), DENSE);
    let e_chol = e_rpa(&sys, &production_cfg(), chol(0.0));
    let e_boys = e_rpa(&sys, &production_cfg(), boys(0.0));
    println!(
        "H2O production thresh=0: dense {e_dense:.12}  |chol-dense| {:.2e}  |boys-dense| {:.2e}",
        (e_chol - e_dense).abs(),
        (e_boys - e_dense).abs()
    );
    assert!((e_chol - e_dense).abs() < 3e-6);
}

/// On current main the comparison the spike ran is VACUOUS, and this test pins
/// why. Since a953d4ff5 (2026-07-28) `build_screened_bov` semicanonicalizes
/// the localized block it is given: it diagonalizes `C_locᵀ F C_loc` over the
/// WHOLE active occupied block and screens the resulting orbitals. Any
/// orthonormal basis of the occupied space (Boys, Cholesky, canonical) is
/// rotated to the same Fock eigenvectors, i.e. back to the canonical
/// occupieds (up to sign, for a non-degenerate spectrum). So the Boys- and
/// Cholesky-seeded screens must keep the SAME aux rows per orbital and the
/// same orbital energies — the localization method is no longer a variable.
#[test]
fn h2o_boys_and_cholesky_screens_coincide_after_semicanonicalization() {
    let (mol, obs, dfbs, op, rhf) = water();
    for thresh in [1e-3, 1e-4, 1e-6] {
        let (sb_c, _) =
            build_screened_bov_cholesky(&mol, &obs, &dfbs, op, &rhf, 0, thresh, f64::INFINITY)
                .unwrap();
        let (sb_b, _) = ferric_rpa::screen::build_screened_bov_boys(
            &mol,
            &obs,
            &dfbs,
            op,
            &rhf,
            0,
            thresh,
            f64::INFINITY,
        )
        .unwrap();
        let eps_canon = &rhf.eps_r()[0..sb_c.n_occ_loc];
        let d_eps = sb_c
            .eps_loc
            .iter()
            .zip(&sb_b.eps_loc)
            .zip(eps_canon)
            .fold(0.0_f64, |m, ((c, b), e)| {
                m.max((c - b).abs()).max((c - e).abs())
            });
        println!(
            "thresh {thresh:.0e}: retained chol {} boys {} of {}  max|eps_loc diff| {d_eps:.2e}",
            sb_c.total_retained,
            sb_b.total_retained,
            sb_c.n_occ_loc * sb_c.naux
        );
        assert_eq!(
            sb_c.p_lists, sb_b.p_lists,
            "retained sets differ at {thresh:e}"
        );
        assert!(d_eps < 1e-10, "eps_loc differ by {d_eps:.2e}");
    }
}

/// Order-sensitive fingerprint of a coefficient matrix (for the log only).
fn fingerprint(c: &Array2<f64>) -> f64 {
    c.indexed_iter()
        .map(|((m, k), v)| ((m + 1) * (k + 2)) as f64 * v)
        .sum()
}

/// Distance of an orthogonal matrix from a signed permutation:
/// max over rows of (1 − max_j |M_ij|). Zero iff the two orbital sets are the
/// same up to ordering and sign.
fn distance_from_signed_permutation(m: &Array2<f64>) -> f64 {
    m.rows()
        .into_iter()
        .map(|r| 1.0 - r.iter().fold(0.0_f64, |a, v| a.max(v.abs())))
        .fold(0.0_f64, f64::max)
}

/// Rules out the "dead knob" explanation of the identical Boys/Cholesky
/// screens. (1) The two localizers produce genuinely DIFFERENT orbitals
/// (Boys starts from the canonical occupieds, not from Cholesky, so this is
/// not "Cholesky only seeds Boys"). (2) After the semicanonicalization that
/// `build_screened_bov` applies, both become the canonical occupied
/// orbitals up to sign — which is why everything downstream coincides.
#[test]
fn h2o_localizers_differ_but_semicanonicalize_to_the_same_orbitals() {
    use ferric_core::linalg::{eigh_dc, Uplo};
    let (_mol, obs, _dfbs, _op, rhf) = water();
    let nocc = 5;
    let chol = cholesky_localize_occupied(&rhf, &obs, 0, nocc).unwrap();
    let boys = ferric_rpa::boys_localize::boys_localize_occupied(&rhf, &obs, 0, nocc).unwrap();
    let s_ao = overlap(&obs);

    let m_in = chol.c_loc.t().dot(&s_ao.dot(&boys.c_loc));
    let d_in = distance_from_signed_permutation(&m_in);

    let semi = |c: &Array2<f64>| -> Array2<f64> {
        let f_loc = c.t().dot(&rhf.fock_r().dot(c));
        let (_e, u) = eigh_dc(&f_loc, Uplo::Upper).unwrap();
        c.dot(&u)
    };
    let (sc, sb) = (semi(&chol.c_loc), semi(&boys.c_loc));
    let c_can = rhf.mos_r().slice(s![.., 0..nocc]).to_owned();
    // Column-wise sign-aligned difference after semicanonicalization.
    let aligned = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
        (0..nocc)
            .map(|k| {
                let (x, y) = (a.column(k), b.column(k));
                let sgn = x.dot(&y).signum();
                x.iter()
                    .zip(y.iter())
                    .fold(0.0_f64, |m, (p, q)| m.max((p - sgn * q).abs()))
            })
            .fold(0.0_f64, f64::max)
    };
    let d_cb = aligned(&sc, &sb);
    let d_cc = aligned(&sc, &c_can);
    println!(
        "fingerprints: chol {:.6}  boys {:.6} (boys converged {} in {} sweeps)\n\
         INPUT orbitals: distance of C_chol^T S C_boys from a signed permutation = {d_in:.3}\n\
         after semicanonicalization: max|C_chol - C_boys| = {d_cb:.2e}, max|C_chol - C_canonical| = {d_cc:.2e}",
        fingerprint(&chol.c_loc),
        fingerprint(&boys.c_loc),
        boys.converged,
        boys.iterations
    );
    assert!(
        d_in > 0.05,
        "localizers produced the same orbitals ({d_in:.2e})"
    );
    assert!(d_cb < 1e-8 && d_cc < 1e-8, "semicanonical sets differ");
}
