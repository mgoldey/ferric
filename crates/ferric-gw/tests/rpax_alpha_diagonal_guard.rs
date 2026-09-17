//! End-to-end evidence that `run_rpax_static_polarizability`'s alpha-diagonal
//! positivity guard both FIRES and stays QUIET on REAL SCF-derived tensors --
//! not just on hand-written arrays (those unit cases live in `bse.rs`'s
//! `mod tests`).
//!
//! The hazard being guarded: at the CLI/Python default `scissor = 0.0` the
//! RPAx@KS kernel pairs a static-screened exchange kernel with the same narrow,
//! GW-uncorrected KS gap on the response diagonal, which drives `(A-B)`
//! non-positive-definite and can yield a NEGATIVE diagonal element of alpha --
//! physically impossible (the molecule would polarize against the field). It is
//! a genuine excitonic instability, not an assembly bug; full root cause in
//! `docs/rpax-negative-diagonal-investigation.md`.
//!
//! System: water/STO-3G/PBE. Chosen because the investigation sweep
//! (`rpax_negative_diagonal_sweep.rs::sweep_negative_diagonal`) already
//! measured it as negative-diagonal at scissor=0.0 -- diag
//! (-0.4727, +10.2999, +5.5296) -- and STO-3G is the cheapest such case, so
//! this runs as a normal (non-`#[ignore]`d) regression gate.
//!
//! Run: OPENBLAS_NUM_THREADS=1 cargo test -p ferric-gw --release \
//!        --test rpax_alpha_diagonal_guard -- --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::bse::run_rpax_static_polarizability;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const WATER_XYZ: &str =
    "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: true,
        verbose: false,
    }
}

/// Run the CLI/Python-wired RPAx static-alpha path on water/STO-3G/PBE at the
/// given scissor, returning whatever the library returned.
fn run_water_sto3g(
    scissor: f64,
) -> Result<ferric_gw::bse::RpaxStaticPolarizabilityResult, ferric_core::error::FerricError> {
    let mol = Molecule::parse_xyz(WATER_XYZ, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let ks = solve_rhf(
        &ctx,
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            xc: Some("PBE".to_string()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        ks.converged,
        "KS reference must converge for this to mean anything"
    );
    run_rpax_static_polarizability(&mol, &obs, &dfbs, op, &ks, &pdep_cfg(), 0, scissor)
}

/// GUARD FIRES: the default `scissor = 0.0` is exactly the setting that
/// produces the unphysical tensor, and the library must now refuse it instead
/// of returning it.
#[test]
fn rpax_static_alpha_refuses_negative_diagonal_at_default_scissor() {
    let res = run_water_sto3g(0.0);
    let err = match res {
        Ok(r) => panic!(
            "GUARD DID NOT FIRE: water/STO-3G/PBE at scissor=0.0 returned Ok with diag \
             ({:+.4}, {:+.4}, {:+.4}), iso {:+.4}. Either the guard regressed or this system \
             stopped reproducing the instability -- if the latter, this test needs a new \
             reproducer, NOT deletion.",
            r.tensor[0][0], r.tensor[1][1], r.tensor[2][2], r.iso
        ),
        Err(e) => e.to_string(),
    };
    eprintln!("[guard fired, scissor=0.0] {err}");
    // It must fail for THE RIGHT REASON, not incidentally (e.g. a memory or
    // LAPACK error would also be an Err and would make this test pass
    // vacuously).
    assert!(
        err.contains("unphysical static polarizability"),
        "errored, but not via the alpha-diagonal guard: {err}"
    );
    assert!(
        err.contains("alpha_xx"),
        "guard fired but did not name the measured offending axis (xx): {err}"
    );
    assert!(
        err.contains("scissor"),
        "guard fired but gave no actionable remedy: {err}"
    );
}

/// GUARD STAYS QUIET: the documented remedy (a nonzero scissor in the
/// 0.3-0.4 Ha band) must produce a tensor the guard accepts. A guard that
/// always fires is as useless as one that never does.
///
/// This also checks the pass is REACHABLE FOR THE RIGHT REASON: it asserts the
/// returned diagonal is genuinely all-positive and physically sized, so the
/// test cannot pass merely because the guard was disabled.
#[test]
fn rpax_static_alpha_accepts_scissor_corrected_diagonal() {
    let r = run_water_sto3g(0.36).unwrap_or_else(|e| {
        panic!("scissor=0.36 should be a HEALTHY case but the run failed: {e}")
    });
    eprintln!(
        "[guard quiet, scissor=0.36] diag = ({:+.6}, {:+.6}, {:+.6})  iso = {:+.6}",
        r.tensor[0][0], r.tensor[1][1], r.tensor[2][2], r.iso
    );
    for (d, axis) in ["x", "y", "z"].iter().enumerate() {
        assert!(
            r.tensor[d][d] > 0.0,
            "alpha_{axis}{axis} = {:+.6e} is not positive -- the guard let an unphysical \
             tensor through",
            r.tensor[d][d]
        );
    }
    // Physically-sized, not a numerically-tiny positive that would pass the
    // sign test for the wrong reason. Water's static alpha is O(1-10) a.u.
    assert!(
        r.iso > 0.5 && r.iso < 50.0,
        "iso alpha {:+.6} a.u. is outside any plausible band for water -- the healthy case \
         may be passing for an unrelated reason",
        r.iso
    );
}
