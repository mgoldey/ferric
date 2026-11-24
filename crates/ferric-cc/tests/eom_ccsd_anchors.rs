//! Exactness anchors of EE-EOM-CCSD singlets (`ferric_cc::eom_ccsd`).
//!
//! # What is checked
//!
//! 1. `t = 0`: the singles-singles block of the operator is the CIS matrix
//!    `δ_ij δ_ab (ε_a − ε_i) + 2(ia|jb) − (ij|ab)` on the same RI integrals,
//!    so it is symmetric and its eigenvalues equal
//!    `ferric_gw::bse::run_cis_tda`, a separate assembly of the same matrix
//!    (H2O / 6-31G, cc-pVDZ-RI).
//! 2. H2 / 6-31G: the dense packed `H̄ − E_CC` (every column one product)
//!    diagonalized by LAPACK equals the Davidson roots. The packed space has
//!    9 dimensions, which the solver's guesses span at once, so the same
//!    check is repeated on H2O / 6-31G (860 dimensions) with a 24-column
//!    subspace that forces iterations and collapses.
//!
//! # Bars
//!
//! * CIS: both sides diagonalize a symmetric matrix built from the same
//!   `B^P_pq`; the difference is the summation order of `Σ_P`, ~1e-15
//!   relative on elements of size ~1. Bar 1e-10 Ha.
//! * Dense vs Davidson: the Davidson stops at residual `conv_tol = 1e-10`;
//!   the eigenvalue error of this non-symmetric problem is first order in the
//!   residual times the eigenvector condition (≲ 10 here), so the bar is
//!   1e-8 Ha.
//!
//! # Physics vs artifact
//!
//! * If the product is right: CIS agrees to round-off and the dense and
//!   Davidson roots agree to the solver tolerance.
//! * A wrong singles-singles term (a missing exchange, a Fock sign) moves the
//!   CIS eigenvalues by the size of that term (≥ 1e-2 Ha). A wrong
//!   doubles-singles or singles-doubles term is invisible to (1) at `t = 0`
//!   only through `t`, and to (2), which checks the solver against the same
//!   product; those terms are tested against PySCF in
//!   `validation_eom_ccsd.rs` and against the CC Jacobian in
//!   `scripts/prototypes/eom_ccsd_proto.py`.
//! * A Davidson that skipped a root of an irrep its guesses never reach would
//!   return a higher root than the dense list; (2) on H2O catches that.
//!
//! # Mutations (run by hand)
//!
//! * The sign of the `2 W_maei r1_me` singles-singles term in `sigma`: the
//!   CIS check fails on its eigenvalue assertion.
//! * The sign of the doubles-singles `(ovvv·r1)_af t2` term in `sigma`: all
//!   three checks here still pass (the term vanishes at `t = 0`, and (2)
//!   compares the solver with the same product); `validation_eom_ccsd.rs`
//!   fails by 1.3e-3 Ha.

use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::eom_ccsd::{eom_ccsd_singlets, EomCcsdConfig, EomCcsdOperator};
use ferric_cc::eom_davidson::dense_eigenvalues;
use ferric_cc::{CcConfig, CcResult};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::{Array2, Array4};

const WATER: &str = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
const H2: &str = "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7414\n";

struct Case {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn case(xyz: &str, basis_name: &str, aux: &str) -> Case {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig::default()
        .with_max_iter(200)
        .with_energy_conv(1e-11)
        .with_density_conv(1e-9);
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged);
    Case {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

fn ccsd(c: &Case) -> CcResult {
    let cfg = CcConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        ..Default::default()
    };
    ccsd_closed_shell(&c.mol, &c.obs, &c.dfbs, Operator::coulomb(), &c.rhf, &cfg).unwrap()
}

/// Symmetric eigenvalues, ascending.
fn sym_eigenvalues(m: &Array2<f64>) -> Vec<f64> {
    let mut v: Vec<f64> = dense_eigenvalues(m).unwrap().iter().map(|p| p.0).collect();
    v.sort_by(f64::total_cmp);
    v
}

#[test]
fn t_zero_singles_block_is_cis() {
    let c = case(WATER, "6-31g", "cc-pvdz-ri");
    let nocc = (c.mol.nelec() / 2) as usize;
    let nv = c.obs.nbasis() - nocc;
    let t1 = Array2::<f64>::zeros((nocc, nv));
    let t2 = Array4::<f64>::zeros((nocc, nocc, nv, nv));
    let opr = EomCcsdOperator::new(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        &c.rhf,
        &t1,
        &t2,
        0,
        5,
        None,
        None,
    )
    .unwrap();
    let s = opr.singles_block().unwrap();
    let asym = (&s - &s.t()).iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(asym < 1e-12, "t=0 singles block is not symmetric: {asym:e}");
    let ours = sym_eigenvalues(&s);
    let cis = ferric_gw::bse::run_cis_tda(&c.mol, &c.obs, &c.dfbs, Operator::coulomb(), &c.rhf, 0)
        .unwrap();
    assert_eq!(ours.len(), cis.omega.len());
    let worst = ours
        .iter()
        .zip(&cis.omega)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!(
        "t=0 singles block vs run_cis_tda: max |dω| = {worst:.3e} over {} roots",
        ours.len()
    );
    assert!(worst < 1e-10, "t=0 singles block vs CIS: {worst:e}");
}

fn dense_vs_davidson(c: &Case, n_roots: usize, max_space: Option<usize>) -> usize {
    let cc = ccsd(c);
    let cfg = EomCcsdConfig {
        n_roots,
        conv_tol: 1e-10,
        max_iter: 300,
        max_space,
        memory_budget_bytes: None,
    };
    let res = eom_ccsd_singlets(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        &c.rhf,
        &cc,
        0,
        &cfg,
    )
    .unwrap();
    let opr = EomCcsdOperator::new(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        &c.rhf,
        cc.t1.as_ref().unwrap(),
        &cc.t2,
        0,
        n_roots,
        max_space,
        None,
    )
    .unwrap();
    let dense = dense_eigenvalues(&opr.dense().unwrap()).unwrap();
    for (k, &w) in res.excitation_energies.iter().enumerate() {
        let (re, im) = dense[k];
        println!(
            "root {k}: Davidson {w:.12} dense {re:.12} (im {im:.1e}) |d| = {:.2e}, res {:.1e}",
            (w - re).abs(),
            res.residuals[k]
        );
        assert!(
            im.abs() < 1e-10,
            "root {k}: dense eigenvalue is complex ({im:e})"
        );
        assert!(
            (w - re).abs() < 1e-8,
            "root {k}: Davidson {w} vs dense {re}"
        );
    }
    res.iterations
}

#[test]
fn h2_631g_dense_equals_davidson() {
    let c = case(H2, "6-31g", "cc-pvdz-ri");
    dense_vs_davidson(&c, 3, None);
}

#[test]
fn h2o_631g_dense_equals_davidson_through_collapses() {
    let c = case(WATER, "6-31g", "cc-pvdz-ri");
    let iters = dense_vs_davidson(&c, 4, Some(24));
    assert!(iters > 3, "the subspace did not have to iterate ({iters})");
}
