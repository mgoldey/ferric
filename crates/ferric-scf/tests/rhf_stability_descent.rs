//! Closed-shell RHF state selection: `RhfConfig::scf_stability_descent` on
//! `solve_rhf` (`rhf_stability_descent` in src/rhf.rs).
//!
//! # System
//!
//! N2 at r = 1.60 Å, def2-SVP (28 AOs, exact J/K). MINAO + plain DIIS
//! converges to an internal (singlet) SADDLE at −108.4825600107 Ha, singlet
//! λ_min −0.0654 (doubly degenerate, π-type). PySCF `newton()` + `stability()`
//! reaches the stable RHF minimum −108.5016165203 Ha, 19.06 mHa lower, whose
//! singlet spectrum has one exact zero mode then +0.1109
//! (testdata/reference/validation/scf_ladder/n2_r1.60_def2-svp.json). A PySCF
//! prototype of exactly this descent (Cayley rotation of the saddle MOs by
//! 0.4 / 0.8 / 1.2 rad along either degenerate eigenvector, density
//! 2 C_occ C_occᵀ, plain DIIS) reached −108.5016165203 from all six starts.
//!
//! # Hypotheses
//!
//! * Descent works: knob on ⇒ the stable minimum, verdict not UNSTABLE.
//! * Descent absent / never reached (knob ignored, verdict not consulted,
//!   rotation a no-op): knob on ⇒ still the saddle, so
//!   `descent_leaves_the_n2_saddle` fails by 19 mHa. MUTATION to run: in
//!   `solve_rhf`, change `if !config.scf_stability_descent {` to `if true {` —
//!   this test must then FAIL (the negative control alone cannot catch it).
//! * Knob leaks into the default path: `descent_off_stays_on_the_saddle` and
//!   the bit-identity tests fail.
//!
//! The energies are compared at 1e-6 Ha: this file is about WHICH state, the
//! like-for-like 1e-7 comparison lives in validation_scf_ladder.rs.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::ScfResult;

const N2_XYZ: &str = "../../testdata/molecules/validation/n2_r1.60.xyz";
/// PySCF stable RHF minimum (newton + stability, 4 guesses agree).
const E_STABLE: f64 = -108.5016165203;
/// PySCF plain-DIIS saddle.
const E_SADDLE: f64 = -108.4825600107;
/// PySCF singlet λ_min at the saddle, ferric convention.
const LAMBDA_SADDLE: f64 = -0.0654084877;
/// Which-state bar (the states are 1.9e-2 Ha apart).
const TOL_STATE: f64 = 1e-6;

fn n2() -> (Molecule, PreparedBasis, SchwarzBounds) {
    let mol = Molecule::load_xyz(N2_XYZ).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}

fn config(check: bool, descent: bool) -> RhfConfig {
    RhfConfig {
        density_conv: 1e-9,
        check_stability: check,
        scf_stability_descent: descent,
        ..Default::default()
    }
}

fn run(mol: &Molecule, prep: &PreparedBasis, bounds: &SchwarzBounds, cfg: &RhfConfig) -> ScfResult {
    let r = solve_rhf(
        &ParallelContext::default(),
        mol,
        prep,
        Operator::coulomb(),
        bounds,
        cfg,
    )
    .unwrap();
    assert!(r.converged, "SCF did not converge (E = {:.10})", r.energy);
    r
}

#[test]
fn descent_leaves_the_n2_saddle() {
    let (mol, prep, bounds) = n2();
    let r = run(&mol, &prep, &bounds, &config(true, true));
    eprintln!("descent on: E = {:.10}", r.energy);
    assert!(
        (r.energy - E_STABLE).abs() < TOL_STATE,
        "descent on: E = {:.10}, expected the stable minimum {E_STABLE:.10} \
         (saddle is {E_SADDLE:.10})",
        r.energy
    );
    let st = r.stability.as_ref().expect("check_stability was on");
    eprintln!("descent on: {}", st.summary());
    // The minimum has an exact zero mode, so MARGINAL is the honest verdict;
    // what must NOT come back is UNSTABLE.
    assert_ne!(st.verdict(), StabilityVerdict::Unstable, "{}", st.summary());
}

/// NEGATIVE CONTROL: knob off ⇒ the saddle, flagged UNSTABLE.
#[test]
fn descent_off_stays_on_the_saddle() {
    let (mol, prep, bounds) = n2();
    let r = run(&mol, &prep, &bounds, &config(true, false));
    assert!(
        (r.energy - E_SADDLE).abs() < TOL_STATE,
        "descent off: E = {:.10}, expected the saddle {E_SADDLE:.10}",
        r.energy
    );
    let st = r.stability.as_ref().expect("check_stability was on");
    assert_eq!(st.verdict(), StabilityVerdict::Unstable, "{}", st.summary());
    assert!(
        (st.lowest_eigenvalue - LAMBDA_SADDLE).abs() < 1e-5,
        "saddle lambda_min {:.10} vs PySCF {LAMBDA_SADDLE:.10}",
        st.lowest_eigenvalue
    );
}

/// The knob without `check_stability` has no verdict to act on: it must skip
/// and return exactly the knob-off result.
#[test]
fn descent_without_check_stability_is_bit_identical_to_off() {
    let (mol, prep, bounds) = n2();
    let off = run(&mol, &prep, &bounds, &config(false, false));
    let on = run(&mol, &prep, &bounds, &config(false, true));
    assert_eq!(off.energy.to_bits(), on.energy.to_bits());
    assert_eq!(off.iterations, on.iterations);
    assert!(off.density_total == on.density_total);
}

/// On a STABLE solution the descent must not touch the answer.
#[test]
fn descent_on_a_stable_solution_is_bit_identical() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let off = run(&mol, &prep, &bounds, &config(true, false));
    let on = run(&mol, &prep, &bounds, &config(true, true));
    let st = off.stability.as_ref().expect("check_stability was on");
    assert_eq!(st.verdict(), StabilityVerdict::Stable, "{}", st.summary());
    assert_eq!(off.energy.to_bits(), on.energy.to_bits());
    assert_eq!(off.iterations, on.iterations);
    assert!(off.density_total == on.density_total);
}
