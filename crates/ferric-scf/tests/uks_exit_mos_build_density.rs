//! A converged UKS result's MOs must build its density: max |C_occ C_occᵀ − D|.
//!
//! SH and HCl+ doublets, UKS-LDA/STO-3G, density_conv 1e-6, no MOM, no
//! fractional occupation. Under LDA the β π hole has no integer state whose
//! occupied orbitals are also the lowest of their own F (SH's occupied β π sits
//! ~11 mHa above the empty one), so DIIS stagnates on a ΔP-settled state
//! (orbital gradient ~8e-4, under the 1e-3 veto floor) and the plain ΔP/ΔE gate
//! accepts it. D is then the integer density of the previous iteration's MOs.
//! MEASURED before the fix: the exit returned the aufbau eigenvectors of F[D],
//! which occupy the OTHER π: |C_occ C_occᵀ − D|_β = 1.05 (SH) / 0.92 (HCl+).
//! The energy is D's and was already right; only the MOs were wrong.
//!
//! The fix returns D's semicanonical orbitals: F[D] diagonalised separately in
//! D's occupied space and in its complement. This file pins:
//! - those MOs rebuild D to rounding, and diagonalise F within each block
//!   (but not across it — the state is not stationary);
//! - the Hessian still refuses the state (an occupied eigenvalue above a
//!   virtual one);
//! - an ordinary aufbau UKS run keeps the canonical eigenvectors of F (no
//!   occupied-virtual coupling left in Cᵀ F C), i.e. the replacement is gated;
//! - so does a LOOSELY converged aufbau UHF run (density_conv 1e-4/1e-3),
//!   whose MOs miss D above the 1e-3 prefilter but whose D is still the aufbau
//!   state of F[D]; the Hessian refuses it as before (MOs miss D);
//! - the Hessian refuses MOs that build D without being F's eigenvectors
//!   even when their eigenvalues are aufbau-ordered (its stationarity check).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::uhf_hessian;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, UPLO};

struct Sys {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn sys(xyz: &str, charge: i32) -> Sys {
    let mol = Molecule::parse_xyz(xyz, charge, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Sys { mol, prep, bounds }
}

fn ks(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        density_conv: 1e-6,
        max_iter: 200,
        ..Default::default()
    }
}

fn solve(sys: &Sys, cfg: &RhfConfig) -> ScfResult {
    solve_uhf(
        &ParallelContext::default(),
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        cfg,
    )
    .expect("UKS must solve")
}

fn max_abs<'a>(it: impl IntoIterator<Item = &'a f64>) -> f64 {
    it.into_iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// max |C_occ C_occᵀ − D|.
fn dev(c: &Array2<f64>, d: &Array2<f64>, n: usize) -> f64 {
    let occ = c.slice(s![.., ..n]);
    max_abs(&(occ.dot(&occ.t()) - d))
}

/// Cᵀ F C split into its blocks: (largest off-diagonal element inside the
/// occupied block and inside the virtual block, largest occupied-virtual
/// element).
fn fock_blocks(c: &Array2<f64>, f: &Array2<f64>, n: usize) -> (f64, f64) {
    let g = c.t().dot(f).dot(c);
    let mut within = 0.0f64;
    let mut across = 0.0f64;
    for ((p, q), v) in g.indexed_iter() {
        if p == q {
            continue;
        }
        if (p < n) == (q < n) {
            within = within.max(v.abs());
        } else {
            across = across.max(v.abs());
        }
    }
    (within, across)
}

fn ascending(e: &[f64]) -> bool {
    e.windows(2).all(|w| w[0] <= w[1])
}

/// The analytic UHF Hessian's refusal of this result's STATE, handed a plain
/// HF config so the KS config check does not fire first (as in
/// `uhf_degenerate_swap.rs`).
fn hessian_state_refusal(sys: &Sys, r: &ScfResult, cfg: &RhfConfig) -> String {
    let hf = RhfConfig {
        xc: None,
        ..cfg.clone()
    };
    uhf_hessian(
        &ParallelContext::default(),
        &sys.mol,
        &sys.prep,
        Operator::coulomb(),
        &sys.bounds,
        r,
        &hf,
    )
    .expect_err("a non-aufbau UHF state must be refused")
    .to_string()
}

fn check_stagnated(xyz: &str, charge: i32, nocc: (usize, usize)) {
    let sys = sys(xyz, charge);
    let cfg = ks("LDA");
    let r = solve(&sys, &cfg);
    assert!(r.converged);
    let (cb, db, fb) = (
        r.mos_beta.as_ref().unwrap(),
        r.density_beta.as_ref().unwrap(),
        r.fock_beta.as_ref().unwrap(),
    );
    let da = dev(&r.mos_alpha, &r.density_alpha, nocc.0);
    let dbv = dev(cb, db, nocc.1);
    let (within, across) = fock_blocks(cb, fb, nocc.1);
    let eps_b = r.eps_beta.as_ref().unwrap();
    println!(
        "charge {charge}: E {:.10} in {} iters; |D(C_occ)-D| alpha {da:.2e} beta {dbv:.2e}; \
         beta C^T F C within-block {within:.2e} across {across:.2e}",
        r.energy, r.iterations
    );
    assert!(dbv < 1e-12, "returned beta MOs do not build D: {dbv:.2e}");
    // α is an ordinary aufbau spin: it misses D only at the convergence scale
    // (MEASURED 6e-6 / 9e-6), below the replacement gate, so it keeps F's
    // canonical eigenvectors. The replacement is per spin.
    let (within_a, across_a) = fock_blocks(&r.mos_alpha, &r.fock_alpha, nocc.0);
    assert!(da < 1e-4, "alpha |D(C_occ)-D| {da:.2e}");
    assert!(
        within_a.max(across_a) < 1e-11,
        "alpha exit MOs are not F's eigenvectors (within {within_a:.2e}, across {across_a:.2e})"
    );
    // Semicanonical: F is diagonal inside each block ...
    assert!(
        within < 1e-10,
        "C^T F C not diagonal within blocks: {within:.2e}"
    );
    // The blocks are separate: the full C stays S-orthonormal, so no virtual
    // has leaked into the occupied space or vice versa.
    let s = oneelectron::overlap(&sys.prep);
    let orth = max_abs(&(cb.t().dot(&s).dot(cb) - Array2::<f64>::eye(cb.ncols())));
    assert!(orth < 1e-12, "beta C^T S C - I = {orth:.2e}");
    // ... and the eigenvalues are those diagonals, each block ascending.
    let g = cb.t().dot(fb).dot(cb);
    let diag_err = max_abs(
        &(0..eps_b.len())
            .map(|p| g[(p, p)] - eps_b[p])
            .collect::<Vec<_>>(),
    );
    assert!(
        diag_err < 1e-10,
        "eps_beta are not diag(C^T F C): {diag_err:.2e}"
    );
    assert!(
        ascending(&eps_b[..nocc.1]) && ascending(&eps_b[nocc.1..]),
        "occupied and virtual blocks must each be ascending: {eps_b:?}"
    );
    // The state is non-stationary-at-aufbau: the Hessian must still refuse it,
    // now through the eigenvalue check (C builds D, so the MOM check passes).
    let why = hessian_state_refusal(&sys, &r, &cfg);
    assert!(
        why.contains("occupied orbital above a virtual"),
        "refused for the wrong reason: {why}"
    );
}

#[test]
fn sh_uks_lda_exit_mos_build_the_density() {
    check_stagnated("2\nSH\nS 0 0 0\nH 0 0 1.3409\n", 0, (9, 8));
}

#[test]
fn hcl_cation_uks_lda_exit_mos_build_the_density() {
    check_stagnated("2\nHCl+\nCl 0 0 0\nH 0 0 1.275\n", 1, (9, 8));
}

/// An ordinary aufbau UKS exit keeps the canonical eigenvectors of F[D]:
/// Cᵀ F C is diagonal to rounding, occupied-virtual block included. The
/// semicanonical replacement would leave that block at the orbital gradient
/// (~err_max), so this fails if the replacement is applied ungated.
#[test]
fn ordinary_uks_exit_keeps_canonical_eigenvectors() {
    let sys = sys("3\nNH2\nN 0 0 0\nH 0 0.8 0.6\nH 0 -0.8 0.6\n", 0);
    let r = solve(&sys, &ks("PBE"));
    assert!(r.converged);
    let err_max = r.convergence.as_ref().map(|c| c.err_max).unwrap();
    let nocc = (5, 4);
    for (c, f, d, n, spin) in [
        (
            &r.mos_alpha,
            &r.fock_alpha,
            &r.density_alpha,
            nocc.0,
            "alpha",
        ),
        (
            r.mos_beta.as_ref().unwrap(),
            r.fock_beta.as_ref().unwrap(),
            r.density_beta.as_ref().unwrap(),
            nocc.1,
            "beta",
        ),
    ] {
        let (within, across) = fock_blocks(c, f, n);
        let dv = dev(c, d, n);
        println!(
            "NH2 PBE {spin}: err_max {err_max:.2e} |D(C_occ)-D| {dv:.2e} \
             C^T F C within {within:.2e} across {across:.2e}"
        );
        assert!(
            within.max(across) < 1e-11,
            "{spin}: exit MOs are not F's eigenvectors (within {within:.2e}, across {across:.2e})"
        );
        // An aufbau exit's MOs miss D only at the convergence scale, far
        // below the replacement gate.
        assert!(dv < 1e-4, "{spin}: |D(C_occ)-D| {dv:.2e}");
    }
    // Precondition that makes the eigenvector check discriminating: a
    // semicanonical replacement would leave the ov block near err_max.
    assert!(
        err_max > 1e-9,
        "err_max {err_max:.2e}: run too tight to tell"
    );
}

fn hf(density_conv: f64) -> RhfConfig {
    RhfConfig {
        density_conv,
        max_iter: 200,
        ..Default::default()
    }
}

const SH: &str = "2\nSH\nS 0 0 0\nH 0 0 1.3409\n";
const CH3: &str = "4\nCH3\nC 0 0 0\nH 0 1.079 0\nH 0.9344 -0.5395 0\nH -0.9344 -0.5395 0\n";

/// A loosely converged AUFBAU state: `scf_converged` accepts
/// `dp_max < 10·density_conv`, so at density_conv 1e-4/1e-3 the returned MOs
/// miss D above the 1e-3 prefilter. D is still the aufbau state of F[D], so
/// the exit must keep F's eigenpairs on both spins (Cᵀ F C diagonal), and the
/// Hessian must refuse the state (its MOs miss D). Review of b81e0a6e9: the
/// prefilter alone rewrote both spins of SH at 1e-4 and the Hessian then
/// RAN CPHF on the non-stationary state.
#[test]
fn loosely_converged_aufbau_uhf_keeps_eigenpairs_and_is_refused() {
    for (name, xyz, nocc, dconv) in [
        ("SH 1e-4", SH, (9, 8), 1e-4),
        ("SH 1e-3", SH, (9, 8), 1e-3),
        ("CH3 1e-3", CH3, (5, 4), 1e-3),
    ] {
        let sys = sys(xyz, 0);
        let cfg = hf(dconv);
        let r = solve(&sys, &cfg);
        assert!(r.converged, "{name}");
        let mut miss = 0.0f64;
        for (c, f, d, n) in [
            (&r.mos_alpha, &r.fock_alpha, &r.density_alpha, nocc.0),
            (
                r.mos_beta.as_ref().unwrap(),
                r.fock_beta.as_ref().unwrap(),
                r.density_beta.as_ref().unwrap(),
                nocc.1,
            ),
        ] {
            let (within, across) = fock_blocks(c, f, n);
            miss = miss.max(dev(c, d, n));
            assert!(
                within.max(across) < 1e-11,
                "{name}: exit MOs are not F's eigenvectors (within {within:.2e}, across {across:.2e})"
            );
        }
        println!(
            "{name}: E {:.10} in {} iters, max miss {miss:.2e}",
            r.energy, r.iterations
        );
        // Precondition: the miss clears the 1e-3 prefilter (MEASURED 2.3e-3
        // and 2.5e-3), so the aufbau decision is what keeps the eigenpairs.
        assert!(miss > 1e-3, "{name}: miss {miss:.2e} under the prefilter");
        let why = hessian_state_refusal(&sys, &r, &cfg);
        println!("{name}: refused: {why}");
    }
}

/// D's semicanonical orbitals (per spin), built independently of the solver
/// in the Löwdin basis: `S^{1/2} D S^{1/2}` is a projector, its eigenvalue-1
/// and eigenvalue-0 eigenvectors span D's occupied space and its complement,
/// and F is diagonalised in each.
fn semicanonical(
    s: &Array2<f64>,
    f: &Array2<f64>,
    d: &Array2<f64>,
    nocc: usize,
) -> (Vec<f64>, Array2<f64>) {
    let (sv, su) = s.eigh(UPLO::Upper).unwrap();
    let half = su.dot(&Array2::from_diag(&sv.mapv(f64::sqrt))).dot(&su.t());
    let inv_half = su
        .dot(&Array2::from_diag(&sv.mapv(|v| 1.0 / v.sqrt())))
        .dot(&su.t());
    let (pv, pu) = half.dot(d).dot(&half).eigh(UPLO::Upper).unwrap();
    let n = s.nrows();
    assert!(pv.iter().take(n - nocc).all(|v| v.abs() < 1e-6));
    assert!(pv.iter().skip(n - nocc).all(|v| (v - 1.0).abs() < 1e-6));
    let mut eps = Vec::with_capacity(n);
    let mut c = Array2::<f64>::zeros((n, n));
    let mut col = 0;
    for range in [n - nocc..n, 0..n - nocc] {
        let block = inv_half.dot(&pu.slice(s![.., range]));
        let (e, u) = block.t().dot(f).dot(&block).eigh(UPLO::Upper).unwrap();
        let k = e.len();
        c.slice_mut(s![.., col..col + k]).assign(&block.dot(&u));
        eps.extend(e.iter().copied());
        col += k;
    }
    (eps, c)
}

/// The Hessian's stationarity backstop: MOs that build D exactly and whose
/// eigenvalues are aufbau-ordered, but which are NOT F's eigenvectors (D's
/// semicanonical orbitals of a loosely converged SH UHF), must be refused as
/// non-stationary. This is the state b81e0a6e9 let through to CPHF.
#[test]
fn hessian_refuses_mos_that_build_d_but_are_not_stationary() {
    let sys = sys(SH, 0);
    let cfg = hf(1e-4);
    let mut r = solve(&sys, &cfg);
    let s = oneelectron::overlap(&sys.prep);
    let nocc = (9, 8);
    let (ea, ca) = semicanonical(&s, &r.fock_alpha, &r.density_alpha, nocc.0);
    let (eb, cbm) = semicanonical(
        &s,
        r.fock_beta.as_ref().unwrap(),
        r.density_beta.as_ref().unwrap(),
        nocc.1,
    );
    // Preconditions: the forged MOs build D, are aufbau-ordered, and carry a
    // real occupied-virtual block — so only the stationarity check can fire.
    let (_, across_a) = fock_blocks(&ca, &r.fock_alpha, nocc.0);
    let (_, across_b) = fock_blocks(&cbm, r.fock_beta.as_ref().unwrap(), nocc.1);
    println!("forged SH semicanonical: ov block alpha {across_a:.2e} beta {across_b:.2e}");
    assert!(dev(&ca, &r.density_alpha, nocc.0) < 1e-10);
    assert!(dev(&cbm, r.density_beta.as_ref().unwrap(), nocc.1) < 1e-10);
    for (e, n) in [(&ea, nocc.0), (&eb, nocc.1)] {
        let homo = e[..n].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lumo = e[n..].iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            homo < lumo,
            "forged eigenvalues cross: not the case under test"
        );
    }
    assert!(across_a.max(across_b) > 1e-4, "not a non-stationary state");
    r.mos_alpha = ca;
    r.eps_alpha = ea;
    r.mos_beta = Some(cbm);
    r.eps_beta = Some(eb);
    let why = hessian_state_refusal(&sys, &r, &cfg);
    assert!(
        why.contains("non-stationary"),
        "refused for the wrong reason: {why}"
    );
}

/// A symmetry-blocked UHF exit keeps the blocked diagonaliser's eigenpairs
/// (`uhf::exit_eigenpairs` skips the replacement: the blocked diagonaliser
/// records irrep labels column by column, so new columns would carry stale
/// labels). Excited 2A1 H2O+ (β hole forced into A1 by `irrep_nelec`) at
/// density_conv 1e-2, where the MOs miss D above the 1e-3 prefilter
/// (MEASURED 3.6e-3 α / 4.7e-3 β).
///
/// A GUARD, NOT A MUTATION KILLER: with the skip removed this still passes,
/// because this state's semicanonical eigenvalues do not cross, so the
/// aufbau decision keeps the eigenpairs anyway. Blocked runs are HF-only
/// (KS + symmetry is refused) and fix occupations per irrep, and no blocked
/// state that reaches the replacement has been found. The skip is defended by
/// construction, not by this test.
#[test]
fn symmetry_blocked_exit_keeps_blocked_eigenpairs() {
    let xyz = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    let mol = Molecule::parse_xyz(&xyz, 1, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let excited: Vec<(String, (usize, usize))> = vec![
        ("A1".into(), (3, 2)),
        ("B1".into(), (1, 1)),
        ("B2".into(), (1, 1)),
    ];
    for dconv in [1e-2, 1e-4] {
        let cfg = RhfConfig {
            symmetry: true,
            ..hf(dconv)
        };
        let u = ferric_scf::symmetric_open::solve_uhf_symmetric(
            &ParallelContext::default(),
            &mol,
            &prep,
            &bounds,
            &cfg,
            Some(&excited),
        )
        .expect("blocked H2O+ 2A1 must solve");
        let r = u.scf_in_frame();
        let nocc = (5, 4);
        for (c, f, d, n, spin) in [
            (
                &r.mos_alpha,
                &r.fock_alpha,
                &r.density_alpha,
                nocc.0,
                "alpha",
            ),
            (
                r.mos_beta.as_ref().unwrap(),
                r.fock_beta.as_ref().unwrap(),
                r.density_beta.as_ref().unwrap(),
                nocc.1,
                "beta",
            ),
        ] {
            let (within, across) = fock_blocks(c, f, n);
            println!(
                "blocked H2O+ 2A1 dconv {dconv:.0e} {spin}: E {:.10} iters {} miss {:.2e} \
                 within {within:.2e} across {across:.2e}",
                r.energy,
                r.iterations,
                dev(c, d, n)
            );
            assert!(
                across < 1e-11,
                "{spin}: blocked exit MOs replaced (occupied-virtual block {across:.2e})"
            );
        }
    }
}
