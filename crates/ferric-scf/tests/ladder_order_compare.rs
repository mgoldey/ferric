//! MEASUREMENT: SCF ladder orderings, including AURORA -> DIIS -> TRAH.
//!
//! Not a CI gate (`#[ignore]`d) and not a change to the shipped ladder:
//! every ladder below is built locally from `ksdft_ladder` / `default_ladder_from`.
//!
//! Ladders:
//!   A  current           plain DIIS, ADIIS, LS 0.5, LS 1.0 + SOSCF, smearing
//!   B  current + TRAH    A with a TRAH rung appended after smearing
//!   C  proposed          [AURORA if applicable], plain DIIS, ADIIS, LS 0.5,
//!                        TRAH (replaces the LS 1.0 + SOSCF rung), smearing
//!
//! Why C puts TRAH BEFORE smearing: smearing returns a fractional-occupation
//! answer and the ladder accepts it as converged, so a rung after it (B) only
//! runs if smearing also fails. Why the AURORA rung is conditional: AURORA is
//! gated off below 10% exact exchange (`MIN_VALIDATED_EXCHANGE_FRACTION`), and
//! a gated-off AURORA rung silently runs plain DIIS -- a duplicate of the next
//! rung.
//!
//! Each run reports per-rung iterations, total iterations, wall time, the
//! endpoint energy and the endpoint's stability verdict. Wall time is only
//! meaningful on a quiet box; run single-threaded.
//!
//! The TRAH rungs (B, C) run TRAH with the trust-region hard-case fix that
//! landed on main as #249 (`723cab536`); numbers taken before that commit
//! measured a different TRAH.
//!
//! ```text
//! RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-scf \
//!     --test ladder_order_compare -- --ignored --nocapture --test-threads 1 [name]
//! ```

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ladder::{default_ladder_from, ksdft_ladder, solve_rhf_ladder, Rung};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;

const TERPINYL_XYZ: &str = include_str!("../../../testdata/molecules/terpinyl_relaxed.xyz");

fn base_ladder(base: &RhfConfig) -> Vec<Rung> {
    if base.xc.is_some() {
        ksdft_ladder(base)
    } else {
        default_ladder_from(base)
    }
}

/// B: the current ladder with TRAH appended (inherits the last rung, minus
/// smearing and SOSCF), as in `trah_hard_systems.rs`.
fn ladder_b(base: &RhfConfig) -> Vec<Rung> {
    let mut l = base_ladder(base);
    let mut c = l.last().unwrap().config.clone();
    c.smearing_sigma = None;
    c.newton_trigger = 0.0;
    c.trah_trigger = Some(1e-2);
    c.max_iter = 100;
    l.push(Rung {
        config: c,
        restart: false,
    });
    l
}

/// C: [AURORA] -> DIIS -> ADIIS -> LS 0.5 -> TRAH -> smearing.
/// `aurora_applies` is decided by the caller from the functional; the rung is
/// omitted rather than left to silently degrade to plain DIIS.
fn ladder_c(base: &RhfConfig, aurora_applies: bool) -> Vec<Rung> {
    let a = base_ladder(base);
    assert_eq!(a.len(), 5, "expected the 5-rung ladder shape");
    let mut l = Vec::new();
    if aurora_applies {
        let mut c = a[0].config.clone();
        c.aurora.enabled = true;
        l.push(Rung {
            config: c,
            restart: false,
        });
    }
    l.push(a[0].clone()); // plain DIIS
    l.push(a[1].clone()); // ADIIS
    l.push(a[2].clone()); // LS 0.5
                          // TRAH replaces the LS 1.0 + SOSCF rung: same iteration budget, no
                          // SOSCF, no extra level shift (TRAH's own trust region does the damping).
    let mut t = a[3].config.clone();
    t.newton_trigger = 0.0;
    t.level_shift = 0.0;
    t.trah_trigger = Some(1e-2);
    l.push(Rung {
        config: t,
        restart: false,
    });
    l.push(a[4].clone()); // smearing, still the last resort
    l
}

fn run(label: &str, mol: &Molecule, bas: &str, base: &RhfConfig, aurora_applies: bool) {
    let bs = basis::bundled(bas).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    eprintln!(
        "\n=== {label}: {} atoms, {} bf ===",
        mol.atoms.len(),
        prep.nbasis()
    );
    for (name, ladder) in [
        ("A current", base_ladder(base)),
        ("B current+TRAH", ladder_b(base)),
        ("C aurora>diis>trah", ladder_c(base, aurora_applies)),
    ] {
        let t0 = Instant::now();
        let res = solve_rhf_ladder(&ctx, mol, &prep, op, &bounds, &ladder);
        let secs = t0.elapsed().as_secs_f64();
        match res {
            Ok(r) => {
                let iters: Vec<usize> = r.rung_outcomes.iter().map(|o| o.iters).collect();
                let total: usize = iters.iter().sum();
                let (lam, verdict) = r
                    .result
                    .stability
                    .as_ref()
                    .map(|s| (s.lowest_eigenvalue, s.verdict().label().to_string()))
                    .unwrap_or((f64::NAN, "none".into()));
                eprintln!(
                    "LADDER {label:<22} {name:<20} conv={:<5} rung={}/{} iters={iters:?} total={total:<4} \
                     wall={secs:7.1}s E={:.10} lam={lam:+.2e} {verdict}",
                    r.converged,
                    r.rung_reached,
                    ladder.len() - 1,
                    r.result.energy
                );
            }
            Err(e) => eprintln!("LADDER {label:<22} {name:<20} ERROR {e:?} wall={secs:.1}s"),
        }
    }
}

fn hf(stab: bool) -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-8,
        density_conv: 1e-7,
        check_stability: stab,
        ..Default::default()
    }
}

fn ks(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        energy_conv: 1e-8,
        density_conv: 1e-7,
        check_stability: true,
        ..Default::default()
    }
}

/// The case the current ladder is documented to FAIL (`trah_hard_systems.rs`,
/// `terpinyl_cation_pbe_ladder_vs_trah`). PBE: AURORA gated off, so ladder C
/// has no AURORA rung here.
#[test]
#[ignore = "measurement: release build, quiet box, RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; 27-atom KS ladders take minutes"]
fn terpinyl_pbe() {
    let mol = Molecule::parse_xyz(TERPINYL_XYZ, 1, 1).unwrap();
    run("terpinyl+ PBE/6-31G", &mol, "6-31g", &ks("PBE"), false);
}

/// Same molecule with a hybrid, where the AURORA rung applies (20% HF).
#[test]
#[ignore = "measurement: release build, quiet box, RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; 27-atom KS ladders take minutes"]
fn terpinyl_b3lyp() {
    let mol = Molecule::parse_xyz(TERPINYL_XYZ, 1, 1).unwrap();
    run("terpinyl+ B3LYP/6-31G", &mol, "6-31g", &ks("B3LYP"), true);
}

/// Easy cost checks: every ladder should stop at its first rung.
#[test]
#[ignore = "measurement: release build, quiet box, RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; prints timings, asserts nothing"]
fn easy_systems() {
    let water = Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
        0,
        1,
    )
    .unwrap();
    run("water RHF/cc-pVDZ", &water, "cc-pvdz", &hf(true), true);
    let benz = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/benzene.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    run("benzene B3LYP/6-31G", &benz, "6-31g", &ks("B3LYP"), true);
}

/// A converged SADDLE: the ladder escalates only on non-convergence, so every
/// ordering is expected to accept it at its first rung (UNSTABLE verdict).
#[test]
#[ignore = "measurement: release build, quiet box, RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; prints timings, asserts nothing"]
fn stretched_n2_saddle() {
    let mol = Molecule::parse_xyz("2\nn2\nN 0 0 0\nN 0 0 1.6\n", 0, 1).unwrap();
    run("N2 1.6A RHF/cc-pVDZ", &mol, "cc-pvdz", &hf(true), true);
}

/// Diagnose the documented failure: CURRENT ladder only, terpinyl+ PBE/6-31G.
/// Run with FERRIC_SCF_TRACE=1 to see every iteration.
#[test]
#[ignore = "measurement: release build, quiet box, RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; 27-atom KS ladders take minutes"]
fn terpinyl_pbe_current_only() {
    let mol = Molecule::parse_xyz(TERPINYL_XYZ, 1, 1).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let base = ks("PBE");
    let ladder = base_ladder(&base);
    for (i, r) in ladder.iter().enumerate() {
        let c = &r.config;
        eprintln!(
            "RUNGCFG {i}: max_iter={} ls={} diis={:?} newton={} smear={:?} \
             stall={:?} div={:?} df_j={:?}",
            c.max_iter,
            c.level_shift,
            c.diis_flavor,
            c.newton_trigger,
            c.smearing_sigma,
            c.stall_window,
            c.divergence_tol,
            c.df_j_aux
        );
    }
    let t0 = Instant::now();
    let r = solve_rhf_ladder(
        &ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &ladder,
    )
    .unwrap();
    for (i, o) in r.rung_outcomes.iter().enumerate() {
        eprintln!(
            "RUNG {i}: iters={} exit={:?} E={:.10}",
            o.iters, o.exit, o.final_energy
        );
    }
    eprintln!(
        "RESULT conv={} rung={} E={:.10} wall={:.1}s",
        r.converged,
        r.rung_reached,
        r.result.energy,
        t0.elapsed().as_secs_f64()
    );
}
