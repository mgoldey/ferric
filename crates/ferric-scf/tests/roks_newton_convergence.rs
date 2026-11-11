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
//! iterations (HVPs; one HVP ≈ one Fock build), DIIS / Newton 1e-2 / soscf
//! 1e-3 / AH 1e-2; NC = not converged in 200:
//!   before: NH2/6-31G PBE 8 / 136 (254) / 21 (31) / 140 (93);
//!           NH2/cc-pVDZ B3LYP 9 / NC (544) / 136 (309) / 169 (150);
//!           OH/cc-pVDZ B3LYP 18 / NC (1270) / 167 (695) / NC (193).
//!   after:  8 / 6 (6) / 6 (6) / 6 (25); 9 / 7 (8) / 7 (4) / 7 (30);
//!           18 / 8 (18) / 9 (18) / 8 (35-38).
//! AH HVPs before tuning its Davidson for one root (n_seed 8, n_block 6,
//! fixed tolerance 1e-7): 66 / 81 / 117; after (n_seed 2, n_block 2,
//! tolerance max(1e-7, 1e-2·min(1,‖g‖)·‖g‖)): 25 / 30 / 35-38 (the last
//! varies between runs with last-bit drift; asserted below the
//! many-root counts, `check`). AH remains the
//! costliest mode (~4x Newton's Fock-equivalents here).
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
use ferric_scf::rohf_ah::ah_fallbacks_on_this_thread;
use ferric_scf::rohf_newton::{hvp_calls_on_this_thread, steihaug_truncations_on_this_thread};
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
    /// AH steps whose Davidson did not converge (PCG Newton taken instead).
    ah_fallbacks: usize,
    /// Newton PCG solves stopped on negative curvature.
    truncations: usize,
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
    let (h0, f0, t0) = (
        hvp_calls_on_this_thread(),
        ah_fallbacks_on_this_thread(),
        steihaug_truncations_on_this_thread(),
    );
    let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg);
    let hvps = hvp_calls_on_this_thread() - h0;
    let ah_fallbacks = ah_fallbacks_on_this_thread() - f0;
    let truncations = steihaug_truncations_on_this_thread() - t0;
    let (energy, iterations) = match res {
        Ok(r) if r.converged => (Some(r.energy), r.iterations),
        Ok(r) => (None, r.iterations),
        Err(_) => (None, cfg.max_iter),
    };
    Run {
        energy,
        iterations,
        hvps,
        ah_fallbacks,
        truncations,
    }
}

/// Run DIIS and every second-order mode; assert each converges to the DIIS
/// energy within 1e-7 Ha in ≤ DIIS iterations, and that it took HVPs.
///
/// `ah_hvps_before_tuning` is the AH HVP count this system measured with the
/// Davidson set up for many roots (n_seed 8, n_block 6; module doc). AH must
/// now take FEWER, which guards the one-root tuning against regression. The
/// Newton/soscf HVP counts are guarded only through the iteration bound.
fn check(
    label: &str,
    xyz: &str,
    basis_name: &str,
    xc: &str,
    fitted: bool,
    ah_hvps_before_tuning: usize,
) {
    let diis = run(xyz, basis_name, xc, fitted, Mode::Diis);
    let e_ref = diis
        .energy
        .unwrap_or_else(|| panic!("{label}: DIIS reference did not converge"));
    eprintln!("{label} DIIS: E = {e_ref:.10}, iters = {}", diis.iterations);
    let mut failures = Vec::new();
    for mode in [Mode::Newton, Mode::Soscf, Mode::Ah] {
        let r = run(xyz, basis_name, xc, fitted, mode);
        eprintln!(
            "{label} {mode:?}: E = {:?}, iters = {}, HVPs = {}, AH fallbacks = {}, truncations = {}",
            r.energy, r.iterations, r.hvps, r.ah_fallbacks, r.truncations
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
        if matches!(mode, Mode::Ah) && r.hvps >= ah_hvps_before_tuning {
            failures.push(format!(
                "{label} AH: {} HVPs, not below the {ah_hvps_before_tuning} of the many-root Davidson",
                r.hvps
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn nh2_631g_pbe_second_order_matches_diis() {
    check("NH2/6-31G PBE", NH2, "6-31g", "PBE", false, 66);
}

#[test]
fn nh2_ccpvdz_b3lyp_fitted_second_order_matches_diis() {
    check("NH2/cc-pVDZ B3LYP", NH2, "cc-pvdz", "B3LYP", true, 81);
}

#[test]
fn oh_ccpvdz_b3lyp_fitted_second_order_matches_diis() {
    check("OH/cc-pVDZ B3LYP", OH, "cc-pvdz", "B3LYP", true, 117);
}

/// How a second-order mode is held against DIIS on a degenerate shell.
#[derive(Clone, Copy)]
enum Expect {
    /// |E − E_DIIS| ≤ 1e-7 Ha.
    SameEnergy,
    /// A hole orientation of the degenerate π shell. The orientations are
    /// degenerate up to the XC grid's rotational variance, which #172 measured
    /// at up to 3.4e-6 Ha on OH/6-31G (`rohf_degenerate_open_shell.rs`); the
    /// off-axis displacement (1e-5 Å) alone would move E by O(1e-10). So the
    /// energy must lie within 1e-5 Ha of DIIS (3x that variance), and nothing
    /// is claimed about which orientation either method picks.
    ///
    /// No iteration bound against DIIS: the two converge to DIFFERENT
    /// stationary points (orientations), so their iteration counts measure
    /// different paths, and no margin between them can be derived (NO AH
    /// sits exactly at DIIS's 26). Cost is held instead by `ah_hvps_ceiling`,
    /// the HVP count of the many-root AH Davidson (n_seed 8, n_block 6) on
    /// the same case, as in `check`.
    GridDegenerateOrientation { ah_hvps_ceiling: usize },
}

/// The #172 degenerate open shells (²Π OH, CH and NO, PBE/6-31G, the F6 test
/// settings energy_conv 1e-10 / density_conv 1e-8). Every listed mode must
/// converge with ≥ 1 ROHF HVP, either to the DIIS energy in ≤ DIIS
/// iterations or, for AH where noted, to a grid-degenerate hole orientation
/// in fewer HVPs than the many-root AH Davidson ([`Expect`]).
///
/// MEASURED 2026-11-11, iterations (HVPs), DIIS / Newton / soscf / AH:
///   OH on-axis   20 / 9 (21) / 11 (20) / –   (the relabel guard's case;
///                Newton and soscf did not converge before the fix)
///   OH off-axis  18 / 10 (24) / 10 (14) / 10 (46)
///   CH off-axis  22 / 8 (25) / 10 (22) / not converged (also before the
///                fix; not asserted)
///   NO off-axis  26 / 12 (20) / 13 (19) / 26 (251)
/// AH's Davidson tolerance now follows density_conv (1e-8 here, was a fixed
/// 1e-7): that costs NO AH one iteration and 25 HVPs (25 (226) before) and
/// OH one HVP. Many-root AH Davidson (the HVP ceilings): OH 116, NO 662;
/// that mutation (applied, run, reverted) fails both.
/// Off-axis AH converges on OH and NO, to −75.620835678 and −129.704259565
/// against DIIS's −75.620834299 and −129.704259009. Those equal the ON-axis
/// DIIS energies to 1e-9: AH lands on the other π-hole orientation, which is
/// degenerate with DIIS's up to the grid's rotational variance. Nothing is
/// claimed about either being the better state.
/// The Steihaug threshold is pinned here too: with every pᵀAp ≤ 0 truncated
/// (threshold 0, applied and reverted), NO off-axis Newton took 92 iterations
/// (460 HVPs) and soscf 60 (335), crawling between those orientations
/// (Rayleigh quotients −2e-7..−1e-4; `rohf_newton::NEG_CURVATURE`).
/// Mutation (applied, run, reverted): the sub-tolerance step returning the
/// diagonal guess κ⁰ = −g/Δ again instead of 0 leaves CH soscf not converged
/// in 200 iterations (E settled to 1e-13 by iteration 9; |g| ~ 2e-8 / the
/// 1e-6 floored π gap rotates the hole ~0.02 per step, so ΔP never reaches
/// 1e-8).
#[test]
fn degenerate_pi_open_shells_second_order_match_diis() {
    use Expect::SameEnergy as Same;
    let orient = |ah_hvps_ceiling| Expect::GridDegenerateOrientation { ah_hvps_ceiling };
    let mut failures = Vec::new();
    let cases: [(&str, &str, &[(Mode, Expect)]); 4] = [
        (
            "OH/6-31G PBE on-axis",
            "2\nOH\nO 0 0 0\nH 0 0 0.97\n",
            &[(Mode::Newton, Same), (Mode::Soscf, Same)],
        ),
        (
            "OH/6-31G PBE off-axis",
            "2\nOH\nO 0 0 0\nH 1e-5 0 0.97\n",
            &[
                (Mode::Newton, Same),
                (Mode::Soscf, Same),
                (Mode::Ah, orient(116)),
            ],
        ),
        (
            "CH/6-31G PBE off-axis",
            "2\nCH\nC 0 0 0\nH 1e-5 0 1.1199\n",
            &[(Mode::Newton, Same), (Mode::Soscf, Same)],
        ),
        (
            "NO/6-31G PBE off-axis",
            "2\nNO\nN 0 0 0\nO 1e-5 0 1.1508\n",
            &[
                (Mode::Newton, Same),
                (Mode::Soscf, Same),
                (Mode::Ah, orient(662)),
            ],
        ),
    ];
    for (label, xyz, modes) in cases {
        let tol = (1e-10, 1e-8);
        let diis = run_tol(xyz, "6-31g", "PBE", false, Mode::Diis, tol);
        let e_ref = diis
            .energy
            .unwrap_or_else(|| panic!("{label}: DIIS reference did not converge"));
        for &(mode, expect) in modes {
            let r = run_tol(xyz, "6-31g", "PBE", false, mode, tol);
            eprintln!(
                "{label} {mode:?}: E = {:?}, iters = {} (DIIS {e_ref:.10}, {}), HVPs = {}, AH fallbacks = {}, truncations = {}",
                r.energy, r.iterations, diis.iterations, r.hvps, r.ah_fallbacks, r.truncations
            );
            let Some(e) = r.energy else {
                failures.push(format!("{label} {mode:?}: not converged"));
                continue;
            };
            let bar = match expect {
                Expect::SameEnergy => 1e-7,
                Expect::GridDegenerateOrientation { .. } => 1e-5,
            };
            if (e - e_ref).abs() > bar {
                failures.push(format!(
                    "{label} {mode:?}: |E - E_DIIS| = {:.3e} > {bar:.0e}",
                    (e - e_ref).abs()
                ));
            }
            let over_budget = match expect {
                Expect::SameEnergy => r.iterations > diis.iterations,
                Expect::GridDegenerateOrientation { ah_hvps_ceiling } => r.hvps >= ah_hvps_ceiling,
            };
            if over_budget || r.hvps == 0 {
                failures.push(format!(
                    "{label} {mode:?}: {} iterations (DIIS {}), {} HVPs",
                    r.iterations, diis.iterations, r.hvps
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
