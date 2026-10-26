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
//!   occupied-virtual coupling left in Cᵀ F C), i.e. the replacement is gated.

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
