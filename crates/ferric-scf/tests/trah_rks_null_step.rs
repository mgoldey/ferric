//! RKS and RHF: once the orbital gradient has converged, TRAH must decline the
//! null step instead of rejecting it over and over.
//!
//! This is the closed-shell twin of `trah_converges.rs`'s
//! `uks_trah_defers_to_diis_instead_of_cycling_null_steps`, and it lives in its
//! own test binary, as ONE test, on purpose: the TRAH counters are
//! process-wide atomics, so only here, with nothing else running in the
//! process, are their deltas this run's alone. That is what lets the test
//! assert that the guard's branch was TAKEN (`TRAH_NULL_STEPS_DECLINED`
//! advanced) and that nothing was rejected, not just that the run was short. A
//! short run alone cannot tell "the guard fired" from "the guard was never
//! reached".
//!
//! # Measured (water/cc-pVDZ, the two configurations below, 2026-10-04)
//!
//! | case | `rhf.rs` guard | iterations | TRAH steps | rejected | declined | wall  |
//! |------|----------------|-----------:|-----------:|---------:|---------:|------:|
//! | RKS/PBE | present     |          8 |          1 |        0 |        1 |  8 s  |
//! | RKS/PBE | removed     |         57 |         25 |       24 |        0 | 99 s  |
//! | RHF     | present     |          8 |          1 |        0 |        1 | 0.7 s |
//! | RHF     | removed     |         57 |         25 |       24 |        0 | 8.6 s |
//!
//! Without the guard the model's predicted change falls to |5.4e-15| Ha (RKS)
//! or |4.6e-15| Ha (RHF), the energy moves by noise, the step is rejected, and
//! the identical step is recomputed 24 times until the radius collapses and
//! DIIS takes over. DIIS alone takes 10 (RKS) and 12 (RHF) iterations.
//!
//! The configuration matters, and not monotonically. Over 48 closed-shell
//! water/cc-pVDZ configurations (RKS/PBE, RKS/B3LYP, RHF; `energy_conv`
//! 1e-9..1e-12 with `density_conv` 100x looser; level shift 0 and 0.2;
//! `trah_trigger` 1e-2 and 1e-3) the guard fired in all 48. Without it, none
//! of the 12 at `density_conv` 1e-7 cycled (the applied null step already
//! meets the density test; this includes the setting of
//! `rks_pbe_trah_matches_diis_and_engages_the_gga_kernel`), and 18 of the 36
//! tighter ones did, in no monotone pattern: RKS/PBE cycles at 1e-10/1e-8
//! with no level shift and trigger 1e-3 but not at 1e-12/1e-10 with a 0.2
//! shift and the same trigger. The likely reason (inferred, not measured
//! directly) is that whether the noise the null step produces scores as a
//! rejection depends on its sign, which is arithmetic, so the
//! iteration bound may not separate the two sides on every machine. The
//! decline counter does not depend on that sign, which is why it is asserted
//! too: a guard that is removed, or that can never fire, fails it on any
//! machine.
//!
//! Both cases carry the same bound; the RKS case is the one the guard was
//! written for (its GGA f_xc kernel enters the Hessian), and the RHF case
//! reaches the same branch in a tenth of the time.
//!
//! # Mutation ledger
//!
//! - `rhf.rs` guard removed: KILLED (both cases cycle; declined = 0).
//! - `TrahConfig::predicted_min` default set to 0: KILLED (`|predicted| < 0`
//!   is never true, so it is the guard removed).
//! - `TrahConfig::predicted_min` default set to 1e-6: KILLED, on the
//!   engagement assertion. The single step TRAH takes in each case predicts
//!   |3.0e-7| Ha (RKS) and |4.4e-7| Ha (RHF), so a 1e-6 bound declines it too
//!   and TRAH never steps: the bound would switch TRAH off for the whole tail.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::trah::{TRAH_NULL_STEPS_DECLINED, TRAH_STEPS_REJECTED, TRAH_STEPS_TAKEN};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The iteration bar. Measured: 8 with the guard and 57 without it, in both
/// cases (table above). 20 leaves 12 iterations of headroom over the guarded
/// runs for machine-dependent arithmetic and sits 37 below the unguarded ones.
const MAX_ITERATIONS: usize = 20;

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
        0,
        1,
    )
    .unwrap()
}

fn load(c: &AtomicUsize) -> usize {
    c.load(Ordering::Relaxed)
}

fn assert_declines_null_steps(what: &str, cfg_diis: RhfConfig) {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_trah = RhfConfig {
        trah_trigger: Some(1e-3),
        ..cfg_diis.clone()
    };
    let r_diis = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();

    let (s0, r0, d0) = (
        load(&TRAH_STEPS_TAKEN),
        load(&TRAH_STEPS_REJECTED),
        load(&TRAH_NULL_STEPS_DECLINED),
    );
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_trah).unwrap();
    let (steps, rejected, declined) = (
        load(&TRAH_STEPS_TAKEN) - s0,
        load(&TRAH_STEPS_REJECTED) - r0,
        load(&TRAH_NULL_STEPS_DECLINED) - d0,
    );
    eprintln!(
        "{what}  DIIS: E={:.12} iters={}   TRAH: E={:.12} iters={} \
         steps={steps} rejected={rejected} declined={declined}",
        r_diis.energy, r_diis.iterations, r.energy, r.iterations
    );

    assert!(r.converged, "{what}: TRAH must converge");
    assert!(
        steps >= 1,
        "{what}: TRAH must have engaged (took {steps} steps)"
    );
    assert!(
        declined >= 1,
        "{what}: the null-step guard's branch was never taken, so this run \
         does not exercise it"
    );
    assert!(
        r.iterations <= MAX_ITERATIONS,
        "{what}: {} iterations, {rejected} rejections: the null-step cycle is \
         back (guarded: 8; unguarded: 57)",
        r.iterations
    );
    assert_eq!(
        rejected, 0,
        "{what}: a rejection here is the null step being scored as noise / ~0"
    );
    assert!(
        (r.energy - r_diis.energy).abs() < 1e-8,
        "{what}: TRAH must reach the DIIS energy: ΔE = {:.3e}",
        (r.energy - r_diis.energy).abs()
    );
}

/// ONE test, two sequential cases, so the counter deltas belong to each case
/// alone (see the module doc).
#[test]
fn closed_shell_trah_declines_null_steps_instead_of_cycling() {
    assert_declines_null_steps(
        "RKS/PBE/H2O/cc-pVDZ",
        RhfConfig {
            xc: Some("PBE".into()),
            energy_conv: 1e-10,
            density_conv: 1e-8,
            max_iter: 200,
            level_shift: 0.0,
            ..Default::default()
        },
    );
    assert_declines_null_steps(
        "RHF/H2O/cc-pVDZ",
        RhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-8,
            max_iter: 200,
            level_shift: 0.2,
            ..Default::default()
        },
    );
}
