//! MEASUREMENT: repairing a converged SCF saddle -- stability descent (#244)
//! vs TRAH (#249), on stretched N2 RHF.
//!
//! N2 at R = 1.6 Angstrom: plain DIIS converges to an internally UNSTABLE RHF
//! saddle (the instability breaks cylindrical symmetry; the stable state is
//! 41 mHa lower at STO-3G, 18 mHa at cc-pVDZ, 19.06 mHa at def2-SVP per #244).
//!
//! Arms (complete solves from the same guess):
//!   control  DIIS + check_stability                 (expected: the saddle)
//!   descent  DIIS + check_stability + scf_stability_descent (#244: rotate
//!            0.4/0.8/1.2 rad along the downhill eigenvector, re-converge with
//!            DIIS, keep the lowest; up to 3 rounds)
//!   trah-tail  TRAH armed once err_max < 1e-2, + check_stability
//!   trah-full  TRAH armed from the first eligible iteration, + check_stability
//!
//! Reported per run: energy, verdict, lambda_min, main-loop SCF iterations
//! (`aurora::target_jk_builds` -- counts every solve_rhf iteration including
//! descent's re-converges, NOT Davidson or TRAH Hessian matvecs), TRAH steps,
//! and wall time. Wall time is the honest total cost; run the arms
//! sequentially with a fixed thread count on a quiet box.

use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// The TRAH counters are process-wide; both tests hold this so a counter delta
/// taken around one solve sees only that solve.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ignore = "measurement: run serially on a quiet box with fixed OPENBLAS/RAYON threads (wall time is reported)"]
fn n2_saddle_descent_vs_trah() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mol = Molecule::parse_xyz("2\nn2\nN 0 0 0\nN 0 0 1.6\n", 0, 1).unwrap();
    let bases: Vec<String> = std::env::var("SADDLE_BASES")
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_else(|_| {
            vec![
                "sto-3g".into(),
                "6-31g".into(),
                "def2-svp".into(),
                "cc-pvdz".into(),
            ]
        });
    for bas in &bases {
        let bs = basis::bundled(bas).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        for (gname, sad) in [("minao", true), ("hcore", false)] {
            for arm in ["control", "descent", "trah-tail", "trah-full"] {
                let cfg = RhfConfig {
                    energy_conv: 1e-10,
                    density_conv: 1e-8,
                    max_iter: 300,
                    use_sad_guess: sad,
                    check_stability: true,
                    scf_stability_descent: arm == "descent",
                    trah_trigger: match arm {
                        "trah-tail" => Some(1e-2),
                        "trah-full" => Some(1e6),
                        _ => None,
                    },
                    ..Default::default()
                };
                let jk0 = ferric_scf::aurora::target_jk_builds();
                let ts0 = ferric_scf::trah::TRAH_STEPS_TAKEN.load(Relaxed);
                let tr0 = ferric_scf::trah::TRAH_STEPS_REJECTED.load(Relaxed);
                let t0 = Instant::now();
                let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg);
                let secs = t0.elapsed().as_secs_f64();
                let jk = ferric_scf::aurora::target_jk_builds() - jk0;
                let ts = ferric_scf::trah::TRAH_STEPS_TAKEN.load(Relaxed) - ts0;
                let tr = ferric_scf::trah::TRAH_STEPS_REJECTED.load(Relaxed) - tr0;
                match r {
                    Ok(r) => {
                        let (lam, v) = r
                            .stability
                            .as_ref()
                            .map(|s| (s.lowest_eigenvalue, s.verdict().label().to_string()))
                            .unwrap_or((f64::NAN, "none".into()));
                        println!(
                            "SADDLE {bas:8} {gname:5} {arm:9} conv={:<5} E={:.10} lam={lam:+.2e} {v:<8} \
                             scf_iters={jk:<4} trah_steps={ts:<3} trah_rej={tr:<2} wall={secs:.2}s",
                            r.converged, r.energy
                        );
                    }
                    Err(e) => {
                        println!("SADDLE {bas:8} {gname:5} {arm:9} ERROR {e:?} wall={secs:.2}s")
                    }
                }
            }
        }
    }
}

/// CoH triplet, def2-SVP, UHF. In the Python spike, DIIS-based re-convergence
/// fell back into saddles on transition-metal radicals; ferric's UHF descent
/// re-converges with DIIS too, and in an earlier ferric run it landed 56 mHa
/// above the lowest stable state found (-1381.73656575 Ha, PySCF integrals).
/// Needs SPIKE_DEF2SVP_JSON (ferric's bundled def2-svp has no Co): a BSE-JSON
/// written from PySCF's def2-svp by
/// scripts/scf_optimizer_spike/pyscf_basis_to_bse_json.py.
#[test]
#[ignore = "measurement: needs SPIKE_DEF2SVP_JSON (PySCF def2-svp with Co) and a quiet box with fixed threads"]
fn coh_uhf_descent_vs_trah() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mol = Molecule::parse_xyz("2\ncoh\nCo 0 0 0\nH 0 0 1.52\n", 0, 3).unwrap();
    let path = std::env::var("SPIKE_DEF2SVP_JSON").expect("SPIKE_DEF2SVP_JSON");
    let bs = basis::load_bse_json(&path).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    println!("COH nbf={}", prep.nbasis());
    let guesses: Vec<(&str, bool)> = match std::env::var("SADDLE_GUESS").as_deref() {
        Ok("minao") => vec![("minao", true)],
        Ok("hcore") => vec![("hcore", false)],
        _ => vec![("minao", true), ("hcore", false)],
    };
    for (gname, sad) in guesses {
        let only = std::env::var("SADDLE_ARMS").ok();
        for arm in ["control", "descent", "trah-tail", "trah-full"] {
            if only
                .as_ref()
                .is_some_and(|o| !o.split(',').any(|a| a == arm))
            {
                continue;
            }
            let cfg = RhfConfig {
                energy_conv: 1e-10,
                density_conv: 1e-8,
                max_iter: 300,
                use_sad_guess: sad,
                check_stability: true,
                scf_stability_descent: arm == "descent",
                trah_trigger: match arm {
                    "trah-tail" => Some(1e-2),
                    "trah-full" => Some(1e6),
                    _ => None,
                },
                ..Default::default()
            };
            let ts0 = ferric_scf::trah::TRAH_STEPS_TAKEN.load(Relaxed);
            let tr0 = ferric_scf::trah::TRAH_STEPS_REJECTED.load(Relaxed);
            let t0 = Instant::now();
            let r = ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg);
            let secs = t0.elapsed().as_secs_f64();
            let ts = ferric_scf::trah::TRAH_STEPS_TAKEN.load(Relaxed) - ts0;
            let tr = ferric_scf::trah::TRAH_STEPS_REJECTED.load(Relaxed) - tr0;
            match r {
                Ok(r) => {
                    let (lam, v) = r
                        .stability
                        .as_ref()
                        .map(|s| (s.lowest_eigenvalue, s.verdict().label().to_string()))
                        .unwrap_or((f64::NAN, "none".into()));
                    // NOT a work total for the descent arm: `ScfResult::iterations`
                    // is the RETURNED solve's own count (the initial solve, or the
                    // re-converge descent kept), omitting the initial solve and
                    // every discarded attempt. Neither `ScfResult` nor
                    // `uhf::stability_descent` exposes a cumulative count, and the
                    // `aurora::target_jk_builds` counter the N2 test uses is
                    // RHF-only. Compare work across arms by `wall`.
                    println!(
                        "COH {gname:5} {arm:9} conv={:<5} E={:.10} dE_ref={:+8.3} mHa lam={lam:+.2e} {v:<8} \
                         final_solve_iters={} trah_steps={ts:<3} trah_rej={tr:<2} wall={secs:.2}s",
                        r.converged,
                        r.energy,
                        1e3 * (r.energy - (-1_381.736_565_75)),
                        r.iterations
                    );
                }
                Err(e) => println!("COH {gname:5} {arm:9} ERROR {e:?} wall={secs:.2}s"),
            }
        }
    }
}
