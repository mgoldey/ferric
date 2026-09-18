//! The target J/K-build counter must actually count target J/K builds.
//!
//! The paper's headline claim is a reduction in target builds, so the counter is
//! load-bearing evidence: if it silently stayed at zero, or drifted from the
//! iteration count, the benchmark would report a meaningless ratio. A counter
//! nobody has watched fail is an assumption, so this file pins its contract.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::aurora::{target_jk_builds, AuroraConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn run(aurora: bool) -> (usize, usize, f64) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        aurora: AuroraConfig {
            enabled: aurora,
            ..Default::default()
        },
        ..Default::default()
    };
    let before = target_jk_builds();
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let delta = target_jk_builds() - before;
    assert!(r.converged);
    (delta, r.iterations, r.energy)
}

/// The counter must advance by exactly one per SCF iteration.
///
/// That equality is the whole contract: one target Fock build per pass of the
/// loop, auxiliary curvature work excluded. If the counter ever drifted from the
/// iteration count, every ratio the benchmark prints would be wrong.
///
/// Serialized onto one thread within this binary (`--test-threads=1` is not
/// assumed) by running the two cases sequentially in a single test, since the
/// counter is process-global.
#[test]
fn target_jk_counter_tracks_iterations_exactly() {
    let (jk_d, it_d, e_d) = run(false);
    eprintln!("DIIS:   {jk_d} J/K builds, {it_d} iterations, E = {e_d:.10}");
    assert!(jk_d > 0, "the counter must actually tick (got {jk_d})");
    assert_eq!(
        jk_d, it_d,
        "exactly one target J/K build per SCF iteration (DIIS)"
    );

    let (jk_a, it_a, e_a) = run(true);
    eprintln!("AURORA: {jk_a} J/K builds, {it_a} iterations, E = {e_a:.10}");
    assert!(jk_a > 0, "the counter must tick on the AURORA path too");
    assert_eq!(
        jk_a, it_a,
        "exactly one target J/K build per SCF iteration (AURORA)"
    );

    // Same answer, and on this system AURORA takes strictly fewer builds. The
    // inequality is asserted because it is the paper's central claim and it was
    // measured to hold here; if it ever stops holding, that is a finding.
    assert!(
        (e_a - e_d).abs() < 1e-9,
        "the fixed point must not move: ΔE = {:.3e}",
        (e_a - e_d).abs()
    );
    assert!(
        jk_a < jk_d,
        "on water/cc-pVDZ AURORA took {jk_a} target builds vs DIIS {jk_d}; \
         a regression here is a real finding, not a flaky test"
    );
}

/// The counter must NOT charge auxiliary-curvature work to the target count.
///
/// This is what makes the J/K ratio an honest metric rather than a tautology:
/// the auxiliary applications are real cost and are reported separately, but
/// they are not target Hamiltonian builds. An AURORA run does far more auxiliary
/// operator applications than it does target builds, so if auxiliary work were
/// being counted the totals would diverge wildly from the iteration count — which
/// the equality asserted above already rules out. Asserted here explicitly so the
/// intent survives a refactor.
#[test]
fn auxiliary_work_is_not_charged_to_the_target_build_count() {
    let (jk, iters, _e) = run(true);
    assert_eq!(
        jk, iters,
        "auxiliary curvature applications must not inflate the target J/K count"
    );
}
