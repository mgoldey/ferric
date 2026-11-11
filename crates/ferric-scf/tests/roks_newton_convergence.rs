//! ROKS second-order SCF (Newton / soscf / AH) must converge where DIIS does,
//! to the same state, in no more iterations.
//!
//! Before this test's fix, `solve_rohf`'s Newton and AH branches replaced C by
//! `eig(F_eff)` (built from the OLD density) and evaluated the gradient and
//! Hessian from the OLD-density spin Focks in that basis. There the gradient
//! is ≈ 0 by construction, so every "Newton" step was an undamped Roothaan
//! step plus a negligible correction, and undamped Roothaan limit-cycles on
//! these radicals; and the AH step's Davidson returned κ = 0 (its residual was
//! formed inside the subspace). MEASURED on this harness, 2026-11-11,
//! iterations (DIIS / Newton 1e-2 / soscf 1e-3 / AH 1e-2; NC = not converged
//! in 200):
//!   before: NH2/6-31G PBE 8 / 136 / 21 / 140; NH2/cc-pVDZ B3LYP 9 / NC / 136
//!           / 169; OH/cc-pVDZ B3LYP 18 / NC / 167 / NC.
//!   after:  8 / 6 / 6 / 6; 9 / 7 / 7 / 7; 18 / 8 / 9 / 8.
//! Mutations (applied, run, reverted): (M1) the Newton branch stepping from
//! `eig(F_eff)` again fails all three (136 its; NC; NC), even with the exact
//! Hessian; (M2) the diagonal-gap Fock part restored fails OH only (Newton NC,
//! soscf 105, AH NC); (M3) the old projected-residual AH Davidson fails all
//! three, NH2/6-31G PBE as a FALSE convergence 1.13e-6 Ha above DIIS (the
//! κ = 0 step leaves ΔP = ΔE = 0), caught by the energy check.
//!
//! Hypotheses, written before measuring the fix.
//!   Physics (fix works): every second-order mode converges to the DIIS
//!   energy (|ΔE| ≤ 1e-7) in ≤ DIIS iterations.
//!   Artifact (vacuous pass): the second-order branch never fires (the gate
//!   is off or the trigger is never met), so "≤ DIIS iterations" is DIIS
//!   against itself. Guarded: every second-order run must make ≥ 1 ROHF HVP
//!   on this thread (`hvp_calls_on_this_thread`).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::rohf_newton::hvp_calls_on_this_thread;
use ferric_scf::screening::SchwarzBounds;

const NH2: &str = "3\nNH2\nN 0 0 0\nH 0 0.80 -0.62\nH 0 -0.80 -0.62\n";
const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

/// Second-order mode under test.
#[derive(Clone, Copy, Debug)]
enum Mode {
    Diis,
    /// `newton_trigger = 1e-2`.
    Newton,
    /// The CLI's `[scf] soscf = true`: `newton_trigger = 1e-3`.
    Soscf,
    /// `ah_trigger = 1e-2`.
    Ah,
}

struct Run {
    energy: Option<f64>,
    iterations: usize,
    hvps: usize,
}

fn run(xyz: &str, basis_name: &str, xc: &str, fitted: bool, mode: Mode) -> Run {
    run_tol(xyz, basis_name, xc, fitted, mode, (1e-9, 1e-7))
}

/// `run` at `(energy_conv, density_conv)`.
fn run_tol(
    xyz: &str,
    basis_name: &str,
    xc: &str,
    fitted: bool,
    mode: Mode,
    (energy_conv, density_conv): (f64, f64),
) -> Run {
    let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let aux = fitted.then(|| "def2-universal-jkfit".to_string());
    let mut cfg = RohfConfig {
        xc: Some(xc.into()),
        energy_conv,
        density_conv,
        max_iter: 200,
        df_j_aux: aux.clone(),
        df_k_aux: aux,
        ..Default::default()
    };
    match mode {
        Mode::Diis => {}
        Mode::Newton => cfg.newton_trigger = 1e-2,
        Mode::Soscf => cfg.newton_trigger = 1e-3,
        Mode::Ah => cfg.ah_trigger = 1e-2,
    }
    let h0 = hvp_calls_on_this_thread();
    let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg);
    let hvps = hvp_calls_on_this_thread() - h0;
    match res {
        Ok(r) if r.converged => Run {
            energy: Some(r.energy),
            iterations: r.iterations,
            hvps,
        },
        Ok(r) => Run {
            energy: None,
            iterations: r.iterations,
            hvps,
        },
        Err(_) => Run {
            energy: None,
            iterations: cfg.max_iter,
            hvps,
        },
    }
}

/// Run DIIS and every second-order mode; assert each converges to the DIIS
/// energy within 1e-7 Ha in ≤ DIIS iterations, and that it took HVPs.
fn check(label: &str, xyz: &str, basis_name: &str, xc: &str, fitted: bool) {
    let diis = run(xyz, basis_name, xc, fitted, Mode::Diis);
    let e_ref = diis
        .energy
        .unwrap_or_else(|| panic!("{label}: DIIS reference did not converge"));
    eprintln!("{label} DIIS: E = {e_ref:.10}, iters = {}", diis.iterations);
    let mut failures = Vec::new();
    for mode in [Mode::Newton, Mode::Soscf, Mode::Ah] {
        let r = run(xyz, basis_name, xc, fitted, mode);
        eprintln!(
            "{label} {mode:?}: E = {:?}, iters = {}, HVPs = {}",
            r.energy, r.iterations, r.hvps
        );
        let Some(e) = r.energy else {
            failures.push(format!(
                "{label} {mode:?}: not converged in {} iterations",
                r.iterations
            ));
            continue;
        };
        if r.hvps == 0 {
            failures.push(format!("{label} {mode:?}: no ROHF HVP was made (vacuous)"));
        }
        if (e - e_ref).abs() > 1e-7 {
            failures.push(format!(
                "{label} {mode:?}: |E - E_DIIS| = {:.3e} > 1e-7",
                (e - e_ref).abs()
            ));
        }
        if r.iterations > diis.iterations {
            failures.push(format!(
                "{label} {mode:?}: {} iterations > DIIS {}",
                r.iterations, diis.iterations
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn nh2_631g_pbe_second_order_matches_diis() {
    check("NH2/6-31G PBE", NH2, "6-31g", "PBE", false);
}

#[test]
fn nh2_ccpvdz_b3lyp_fitted_second_order_matches_diis() {
    check("NH2/cc-pVDZ B3LYP", NH2, "cc-pvdz", "B3LYP", true);
}

#[test]
fn oh_ccpvdz_b3lyp_fitted_second_order_matches_diis() {
    check("OH/cc-pVDZ B3LYP", OH, "cc-pvdz", "B3LYP", true);
}

/// The #172 degenerate open shells (²Π CH and NO, PBE/6-31G, the F6 test
/// settings energy_conv 1e-10 / density_conv 1e-8), 1e-5 Å off-axis so the
/// π degeneracy is broken only by the geometry and the grid. Newton and soscf
/// must reach the DIIS energy (1e-7 Ha) in ≤ DIIS iterations.
///
/// MEASURED 2026-11-11 (iterations, DIIS / Newton / soscf): CH 22 / 8 / 10,
/// NO 26 / 12 / 11. Before the Newton fix both were 174+ or not converged.
/// Mutation (applied, run, reverted): the sub-tolerance step returning the
/// diagonal guess κ⁰ = −g/Δ again instead of 0 leaves CH soscf not converged
/// in 200 iterations (E settled to 1e-13 by iteration 9; |g| ~ 2e-8 / the
/// 1e-6 floored π gap rotates the hole ~0.02 per step, so ΔP never reaches
/// 1e-8). AH is not asserted here: on these two off-axis cases it did not
/// converge before or after the fix (see the PR).
#[test]
fn degenerate_pi_open_shells_newton_and_soscf_match_diis() {
    let mut failures = Vec::new();
    for (label, xyz) in [
        ("CH/6-31G PBE", "2\nCH\nC 0 0 0\nH 1e-5 0 1.1199\n"),
        ("NO/6-31G PBE", "2\nNO\nN 0 0 0\nO 1e-5 0 1.1508\n"),
    ] {
        let tol = (1e-10, 1e-8);
        let diis = run_tol(xyz, "6-31g", "PBE", false, Mode::Diis, tol);
        let e_ref = diis
            .energy
            .unwrap_or_else(|| panic!("{label}: DIIS reference did not converge"));
        for mode in [Mode::Newton, Mode::Soscf] {
            let r = run_tol(xyz, "6-31g", "PBE", false, mode, tol);
            eprintln!(
                "{label} {mode:?}: E = {:?}, iters = {} (DIIS {e_ref:.10}, {}), HVPs = {}",
                r.energy, r.iterations, diis.iterations, r.hvps
            );
            match r.energy {
                None => failures.push(format!("{label} {mode:?}: not converged")),
                Some(e) if (e - e_ref).abs() > 1e-7 => failures.push(format!(
                    "{label} {mode:?}: |E - E_DIIS| = {:.3e}",
                    (e - e_ref).abs()
                )),
                Some(_) if r.iterations > diis.iterations || r.hvps == 0 => failures.push(format!(
                    "{label} {mode:?}: {} iterations (DIIS {}), {} HVPs",
                    r.iterations, diis.iterations, r.hvps
                )),
                Some(_) => {}
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
