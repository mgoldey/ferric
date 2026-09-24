//! **F6: ROHF/ROKS on a degenerate open shell.** Convergence must not depend on
//! 1e-7 Å of geometry noise, and a converged EXCITED state must never be
//! returned as the answer.
//!
//! # What was wrong (MEASURED 2026-09-24, before `crate::rohf_occupation`)
//!
//! OH ²Π ROKS/PBE, O(0,0,0) H(0,0,0.97) Å, density_conv 1e-8, energy_conv
//! 1e-10, max_iter 200, exact J (this crate's default), via the prebuilt CLI:
//!
//! ```text
//!   H displaced by (Å)     STO-3G                        6-31G
//!   0                      converged, 61 it              NOT converged (200)
//!   (0, 0, 1e-7)           "converged" at −73.9911207    NOT converged
//!                          — 0.58 Ha HIGH, err_max 0.12
//!   (1e-6, 2e-6, 0)        NOT converged                 NOT converged
//!   (1e-5 | 1e-4 | 1e-3,0,0) NOT converged               NOT converged
//! ```
//!
//! NO and CH (²Π, 6-31G, ROKS/PBE) did not converge in 200 iterations at any
//! geometry tried. Plain ROHF on OH converged in 7 (STO-3G) / 10 (6-31G)
//! iterations at every perturbation — the defect is the ROKS occupation
//! dynamics, not the geometry.
//!
//! The mechanism (reproduced in a numpy replica of `solve_rohf` on PySCF
//! integrals, see `crate::rohf_occupation`'s module doc): the Guest-Saunders
//! effective Fock orders the OPEN π below the CLOSED π in ROKS/PBE, so index
//! aufbau moves the hole from π_x to π_y and back every iteration (dp_rms
//! pinned at 0.236 while E and the gradient are converged); in NO/CH it is the
//! single electron that hops between the two π*. DIIS fed that history
//! extrapolates wildly and can stagnate on a σ*-occupied configuration, which
//! the ΔP-only gate then accepts.
//!
//! # What these tests assert, and why they fail today
//!
//! * `oh_roks_pbe_perturbation_sweep_sto3g` / `..._631g`: every
//!   perturbation converges within an iteration budget, to ONE state (energy
//!   spread bounded), and that state is the ground state. FAILS TODAY: 5 of 6
//!   STO-3G points and 6 of 6 6-31G points either exhaust 200 iterations
//!   (`solve_rohf` returns `Err`) or converge 0.58 Ha high; the sixth STO-3G
//!   point converges, but in 61 iterations, over the 60-iteration budget.
//! * `no_and_ch_roks_pbe_converge`: FAILS TODAY (200 iterations, no
//!   convergence, both molecules, both geometries).
//! * `oh_roks_lda_098_sweep_loose_and_tight`: SVWN-LDA at 0.98 Å, with the
//!   settings of `open_shell_ks_opt_freq.rs` (loose) and of F6 (tight) — the
//!   config that caught a first, spin-Fock version of the lock. FAILS TODAY
//!   (pre-F6 CLI, exact J): loose, the unperturbed and 1e-7 Å points
//!   "converge" in 10 iterations to −73.4949868, the 0.58 Ha state; tight,
//!   the 1e-7 Å point exhausts 200 iterations.
//! * `oh_rohf_perturbation_sweep_is_unchanged`: the plain-ROHF CONTROL. It
//!   passes today and must keep passing — the guard must not perturb a system
//!   that never needed it.
//! * `a_converged_excited_state_is_not_returned`: a deliberately excited
//!   ROHF state handed to the solver as its starting density. FAILS TODAY: the
//!   solver holds it and reports it converged, 0.158 Ha (4.3 eV) high.
//! * `hcore_guess_excited_states_are_repaired`: three systems whose hcore
//!   guess converged to an excited state (4.3, 3.0, 1.8 eV high). FAILS TODAY.
//!
//! # Why the ROKS energy tolerance is 1e-5 Ha and not 1e-8
//!
//! A ²Π state's energy is invariant to rotating the hole about the axis only
//! if the integration grid is. The Becke-Lebedev grid is not, so the SCF can
//! settle at different hole orientations, which differ in energy by grid
//! error. MEASURED in the replica (same 75×110 unpruned Becke grid): spread
//! across the sweep 9.5e-7 Ha (STO-3G) and 1.5e-6 Ha (6-31G), 1.3e-6 Ha for CH
//! — with no change of STATE (the nearest other state is ≥ 0.1 Ha away). A
//! 1e-8 bound would be testing the grid, not the SCF. Plain ROHF has no grid,
//! and there the bound is 1e-7 — set by the geometry change itself: moving H
//! 1e-3 Å off-axis changes the ROHF energy by 5.3e-8 Ha (measured).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;

/// H displacements (Å) — the sweep from the F6 report.
const PERTURBATIONS_ANG: [(f64, f64, f64); 6] = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 1e-7),
    (1e-6, 2e-6, 0.0),
    (1e-5, 0.0, 0.0),
    (1e-4, 0.0, 0.0),
    (1e-3, 0.0, 0.0),
];

/// The spurious σ*-occupied OH ROKS/PBE/STO-3G state (exact J, measured), for
/// the failure message only.
const OH_PBE_STO3G_SPURIOUS: f64 = -73.991_120_7;

/// A diatomic A(0,0,0)–B(0,0,r) with B displaced by `d` Å.
fn diatomic(
    a: &str,
    b: &str,
    r: f64,
    d: (f64, f64, f64),
    mult: usize,
    basis_name: &str,
) -> (Molecule, PreparedBasis, SchwarzBounds) {
    let xyz = format!("2\n\n{a} 0.0 0.0 0.0\n{b} {} {} {}\n", d.0, d.1, r + d.2);
    let mol = Molecule::parse_xyz(&xyz, 0, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}

/// The F6 report's settings.
fn cfg(xc: Option<&str>) -> RhfConfig {
    RhfConfig {
        xc: xc.map(str::to_string),
        density_conv: 1e-8,
        energy_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    }
}

/// Run the sweep and assert: all converge within `max_iters`, the energies
/// agree to `spread_tol`, and every one is within 1e-3 Ha of `e_ground`
/// (i.e. it is the ground STATE; the gaps to every other state seen are
/// ≥ 0.1 Ha, so 1e-3 cannot confuse two states).
fn assert_sweep(
    label: &str,
    cfg: &RhfConfig,
    basis_name: &str,
    r_oh: f64,
    e_ground: f64,
    max_iters: usize,
    spread_tol: f64,
) {
    let ctx = ParallelContext::default();
    let mut rows: Vec<(String, f64, usize)> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for d in PERTURBATIONS_ANG {
        let (mol, prep, bounds) = diatomic("O", "H", r_oh, d, 2, basis_name);
        let tag = format!("{label} d=({:e},{:e},{:e})", d.0, d.1, d.2);
        match solve_rohf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, cfg) {
            Ok(r) => {
                println!("{tag}: E = {:.10}  iters = {}", r.energy, r.iterations);
                if r.iterations > max_iters {
                    failures.push(format!(
                        "{tag}: {} iterations > budget {max_iters}",
                        r.iterations
                    ));
                }
                if (r.energy - e_ground).abs() > 1e-3 {
                    failures.push(format!(
                        "{tag}: E = {:.10} is not the ground state {e_ground:.10} \
                         (spurious σ*-occupied state for reference: {OH_PBE_STO3G_SPURIOUS})",
                        r.energy
                    ));
                }
                rows.push((tag, r.energy, r.iterations));
            }
            Err(e) => failures.push(format!("{tag}: SCF FAILED: {e:?}")),
        }
    }
    if rows.len() == PERTURBATIONS_ANG.len() {
        let e_min = rows.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        let e_max = rows.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
        println!("{label}: energy spread {:.3e} Ha", e_max - e_min);
        if e_max - e_min > spread_tol {
            failures.push(format!(
                "{label}: energies spread {:.3e} Ha > {spread_tol:.0e} — the sweep \
                 did not land on one state",
                e_max - e_min
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// OH ROKS/PBE/STO-3G. Ground state from the replica on the same grid spec
/// (−74.5719105) agrees with ferric's own exact-J unperturbed run (−74.5719108)
/// to 3e-7. Budget 60: the replica needed ≤ 14 iterations at every point.
#[test]
fn oh_roks_pbe_perturbation_sweep_sto3g() {
    assert_sweep(
        "OH ROKS/PBE/STO-3G",
        &cfg(Some("PBE")),
        "sto-3g",
        0.97,
        -74.571_910_5,
        60,
        1e-5,
    );
}

/// OH ROKS/PBE/6-31G — failed at EVERY perturbation, including none. Ground
/// state from the replica (−75.6208441 .. −75.6208457 across the sweep).
/// Budget 60: the replica needed ≤ 23 iterations.
#[test]
fn oh_roks_pbe_perturbation_sweep_631g() {
    assert_sweep(
        "OH ROKS/PBE/6-31G",
        &cfg(Some("PBE")),
        "6-31g",
        0.97,
        -75.620_845_0,
        60,
        1e-5,
    );
}

/// The control: plain ROHF never swapped, converged in 7 / 10 iterations at
/// every perturbation before the fix, and must still. Energies from ferric's
/// CLI (STO-3G) and the PySCF reference in `rohf_state_selection.rs` (6-31G).
#[test]
fn oh_rohf_perturbation_sweep_is_unchanged() {
    let hf = cfg(None);
    assert_sweep(
        "OH ROHF/STO-3G",
        &hf,
        "sto-3g",
        0.97,
        -74.361_562_0,
        15,
        1e-7,
    );
    assert_sweep(
        "OH ROHF/6-31G",
        &hf,
        "6-31g",
        0.97,
        -75.361_846_292_5,
        15,
        1e-7,
    );
}

/// **The config that caught the first version of this fix.** OH at 0.98 Å,
/// STO-3G, SVWN-LDA, with `open_shell_ks_opt_freq.rs`'s `ks_config("LDA")`
/// settings (density_conv 1e-4, energy_conv 1e-7, max_iter 400), and the same
/// sweep at the tight F6 settings.
///
/// The first lock chose the labels by spin-Fock aufbau. Under LDA the β-Fock
/// order of the two π inverts with the occupation, so the LOCKED loop kept
/// swapping the hole every iteration and never converged (ferric:
/// `ScfConvergence { iterations: 400, last_energy: -74.0752222 }` — the right
/// energy, a density that would not settle). MEASURED in the replica: the
/// spin-Fock lock ran 400 iterations at dp_rms 0.236 with the gradient at
/// 5e-11; the continuity lock converges in 6 (loose) / 15-17 (tight)
/// iterations.
///
/// Pre-F6 ferric (CLI, exact J, measured): LOOSE — the unperturbed and 1e-7 Å
/// points report converged in 10 iterations at −73.4949868, the 0.58 Ha
/// σ*-occupied state (the other four reach −74.0752223 in 8-22); TIGHT — the
/// 1e-7 Å point exhausts 200 iterations. So `open_shell_ks_opt_freq.rs`'s
/// zero-step baseline (recorded there as −73.4949867620) was the excited
/// state all along; its assertions never looked at the energy.
///
/// −74.0752151 (replica) / −74.0752223 (ferric) is the SVWN ground state; the
/// 7e-6 difference is the two codes' grids, not a state.
#[test]
fn oh_roks_lda_098_sweep_loose_and_tight() {
    let loose = RhfConfig {
        xc: Some("LDA".into()),
        energy_conv: 1e-7,
        density_conv: 1e-4,
        max_iter: 400,
        ..Default::default()
    };
    assert_sweep(
        "OH ROKS/LDA/STO-3G 0.98 loose",
        &loose,
        "sto-3g",
        0.98,
        -74.075_215_1,
        40,
        1e-5,
    );
    assert_sweep(
        "OH ROKS/LDA/STO-3G 0.98 tight",
        &cfg(Some("LDA")),
        "sto-3g",
        0.98,
        -74.075_215_1,
        60,
        1e-5,
    );
}

/// NO and CH (²Π): the single-electron form of the flip. Unperturbed and with
/// H/O moved 1e-5 Å off-axis; both must converge, to one state.
#[test]
fn no_and_ch_roks_pbe_converge() {
    let ctx = ParallelContext::default();
    // (A, B, R/Å, replica ground state, iteration budget). Budgets are ~3x
    // the replica's worst over a six-point sweep: CH 26, NO 44 iterations.
    for (a, b, r, e_ground, budget) in [
        ("C", "H", 1.1199, -38.402_545_5, 80),
        ("N", "O", 1.1508, NO_PBE_631G, 120),
    ] {
        let mut energies = Vec::new();
        for d in [(0.0, 0.0, 0.0), (1e-5, 0.0, 0.0)] {
            let (mol, prep, bounds) = diatomic(a, b, r, d, 2, "6-31g");
            let res = solve_rohf(
                &ctx,
                &mol,
                &prep,
                Operator::coulomb(),
                &bounds,
                &cfg(Some("PBE")),
            )
            .unwrap_or_else(|e| panic!("{a}{b} ROKS/PBE/6-31G d={d:?}: {e:?}"));
            println!(
                "{a}{b} d={d:?}: E = {:.10}  iters = {}",
                res.energy, res.iterations
            );
            assert!(
                res.iterations <= budget,
                "{a}{b} d={d:?}: {} iterations > budget {budget}",
                res.iterations
            );
            assert!(
                (res.energy - e_ground).abs() < 1e-3,
                "{a}{b} d={d:?}: E = {:.10} is not the ground state {e_ground:.10}",
                res.energy
            );
            energies.push(res.energy);
        }
        assert!(
            (energies[0] - energies[1]).abs() < 1e-5,
            "{a}{b}: 1e-5 Å moved the ROKS energy by {:.3e} Ha — two states",
            energies[0] - energies[1]
        );
    }
}

/// NO ROKS/PBE/6-31G ground state (replica, same grid spec).
const NO_PBE_631G: f64 = -129.704_266_3;

/// OH/6-31G ROHF, the PySCF reference (`rohf_state_selection.rs::OH_631G`).
const OH_631G: f64 = -75.361_846_292_5;
/// The σ-hole (²Σ⁺-like) OH/6-31G ROHF stationary point that the hcore guess
/// used to converge to, 4.3 eV above `OH_631G` (pinned bit-identically in
/// `rohf_state_selection.rs::hcore_config_reproduces_the_pre_fix_answer_bit_identically`).
const OH_631G_EXCITED: f64 = -75.203_724_952_9;

fn tight(use_sad_guess: bool, guard: bool) -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        density_conv: 1e-10,
        energy_conv: 1e-11,
        use_sad_guess,
        rohf_occupation_guard: guard,
        ..Default::default()
    }
}

/// **A deliberately excited state is not reported as converged.**
///
/// The excited state's own converged density is handed to the solver as its
/// starting point, so the first thing the SCF does is converge — onto the
/// excited state. Before the fix that was the answer. Now the swap witness
/// evaluates the best one-electron move (here: the hole from 3σ into a π,
/// Koopmans estimate +0.46 Ha, "stable", but 0.154 Ha LOWER once evaluated)
/// and continues from it.
///
/// The negative control is part of the test: with the guard OFF the same
/// input must stay on the excited state, or the input was not what this test
/// claims it is and the positive half would prove nothing.
#[test]
fn a_converged_excited_state_is_not_returned() {
    let ctx = ParallelContext::default();
    let (mol, prep, bounds) = diatomic("O", "H", 0.97, (0.0, 0.0, 0.0), 2, "6-31g");
    let run = |cfg: &RhfConfig| {
        solve_rohf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, cfg)
            .expect("OH/6-31G ROHF must converge")
    };

    // 1. Produce the excited state: hcore guess, guard OFF (the pre-F6 path).
    let excited = run(&tight(false, false));
    assert!(
        (excited.energy - OH_631G_EXCITED).abs() < 1e-8,
        "premise: guard-off hcore OH/6-31G should land on the excited state \
         {OH_631G_EXCITED:.10}; got {:.10}",
        excited.energy
    );
    let seeded = |guard: bool| RhfConfig {
        init_guess_density: Some(excited.density_total.clone()),
        ..tight(false, guard)
    };

    // 2. Negative control: guard OFF holds the excited state it is given.
    let held = run(&seeded(false));
    println!(
        "guard OFF from the excited density: E = {:.10} ({} iters)",
        held.energy, held.iterations
    );
    assert!(
        (held.energy - OH_631G_EXCITED).abs() < 1e-6,
        "negative control: without the guard the excited density should be \
         HELD (it is a converged ROHF stationary point); got {:.10}",
        held.energy
    );

    // 3. The fix: guard ON must not return it.
    let fixed = run(&seeded(true));
    println!(
        "guard ON  from the excited density: E = {:.10} ({} iters)",
        fixed.energy, fixed.iterations
    );
    assert!(
        (fixed.energy - OH_631G).abs() < 1e-6,
        "the excited state ({OH_631G_EXCITED:.10}) was returned as converged, or \
         the solve went elsewhere: got {:.10}, ground state {OH_631G:.10}",
        fixed.energy
    );
}

/// **Three hcore-guess excited states are repaired** — the guess the default
/// no longer uses, but a caller can still ask for (`use_sad_guess = false`).
/// Pre-fix energies: `rohf_state_selection.rs` (OH, NH₂) and PySCF's own hcore
/// run (F₂⁺, which PySCF also converges to). Targets: the PySCF references.
#[test]
fn hcore_guess_excited_states_are_repaired() {
    let ctx = ParallelContext::default();
    let nh2_xyz = "3\nNH2\nN 0.0 0.0 0.0\nH 0.0 0.8020 0.5942\nH 0.0 -0.8020 0.5942\n";
    let nh2 = {
        let mol = Molecule::parse_xyz(nh2_xyz, 0, 2).unwrap();
        let prep = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        (mol, prep, bounds)
    };
    let f2p = {
        let mol = Molecule::parse_xyz("2\n\nF 0.0 0.0 0.0\nF 0.0 0.0 1.3220\n", 1, 2).unwrap();
        let prep = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        (mol, prep, bounds)
    };
    let oh = diatomic("O", "H", 0.97, (0.0, 0.0, 0.0), 2, "6-31g");
    for (name, (mol, prep, bounds), e_excited, e_ref) in [
        ("OH/6-31G", oh, OH_631G_EXCITED, OH_631G),
        ("F2+/6-31G", f2p, -197.925_208_475_7, -198.036_196_546_6),
        ("NH2/6-31G", nh2, -55.464_951_617_7, -55.530_432_218_7),
    ] {
        let e = solve_rohf(
            &ctx,
            &mol,
            &prep,
            Operator::coulomb(),
            &bounds,
            &tight(false, true),
        )
        .unwrap_or_else(|err| panic!("{name} hcore: {err:?}"))
        .energy;
        println!("{name}: hcore + guard = {e:.10}  (pre-fix {e_excited:.10}, ref {e_ref:.10})");
        assert!(
            (e - e_ref).abs() < 1e-6,
            "{name}: hcore guess with the guard should reach {e_ref:.10}; got {e:.10} \
             (the pre-fix excited state was {e_excited:.10})"
        );
    }
}
