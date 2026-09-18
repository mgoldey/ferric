//! Measured AURORA-vs-DIIS cost on named systems.
//!
//! Ignored by default (`#[ignore]`): this is a measurement harness, not a gate.
//! Run it deliberately, on a quiet box, with
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --test aurora_benchmark \
//!     --release -- --ignored --nocapture --test-threads=1
//! ```
//!
//! # What is measured, and why it is measured this way
//!
//! The paper's two headline numbers are wall-time ratio and *target* J/K-build
//! ratio. Both are recorded here. The J/K count comes from
//! [`ferric_scf::aurora::target_jk_builds`], which ticks once per target Fock
//! build and deliberately does NOT count auxiliary-curvature applications — the
//! auxiliary work is the method's cost, and hiding it in the headline metric
//! would beg the question. Auxiliary applications are reported separately so the
//! overhead is visible.
//!
//! Wall AND cpu time are both reported: on a contended box wall time inflates
//! while cpu time does not, and the divergence is the tell. This repo has a
//! documented history of benchmark traps (warm caches, contended boxes), so:
//!
//! * every pair runs DIIS first and AURORA second within the same process, so
//!   both see the same warmed integral/basis state;
//! * each configuration is run [`REPEATS`] times and the MINIMUM is reported —
//!   the minimum is the statistic least contaminated by scheduler noise;
//! * the spread across repeats is printed, so a reader can see whether the box
//!   was quiet enough for the number to mean anything.
//!
//! A single run of this harness is NOT evidence of a speedup. Read the spread.

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::aurora::{target_jk_builds, AuroraConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Repeats per configuration; the minimum wall time is reported.
const REPEATS: usize = 3;

struct Measurement {
    wall_s: f64,
    cpu_s: f64,
    iterations: usize,
    jk_builds: usize,
    energy: f64,
    converged: bool,
    spread: f64,
}

/// Coarse process CPU time, for the wall-vs-cpu divergence check.
fn cpu_seconds() -> f64 {
    // /proc/self/stat fields 14,15 are utime,stime in clock ticks.
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return f64::NAN;
    };
    // The comm field may contain spaces inside parentheses; split after ')'.
    let Some(rest) = stat.rsplit_once(')') else {
        return f64::NAN;
    };
    let fields: Vec<&str> = rest.1.split_whitespace().collect();
    // After ") ", field indices shift: state is [0], so utime is [11], stime [12].
    if fields.len() < 13 {
        return f64::NAN;
    }
    let ticks = 100.0; // _SC_CLK_TCK is 100 on every Linux this runs on.
    let utime: f64 = fields[11].parse().unwrap_or(f64::NAN);
    let stime: f64 = fields[12].parse().unwrap_or(f64::NAN);
    (utime + stime) / ticks
}

thread_local! {
    /// Lets [`measure_once`] request a single repeat without duplicating the
    /// body of [`measure`].
    static REPEATS_OVERRIDE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// One timed solve with no repeats — for cases too expensive to repeat.
fn measure_once(xyz: &str, basis_name: &str, aurora: bool) -> Measurement {
    let saved = REPEATS_OVERRIDE.with(|c| c.replace(Some(1)));
    let m = measure("", xyz, basis_name, aurora);
    REPEATS_OVERRIDE.with(|c| c.set(saved));
    m
}

fn measure(label: &str, xyz: &str, basis_name: &str, aurora: bool) -> Measurement {
    let mol = Molecule::load_xyz(xyz).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
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

    let mut best = f64::INFINITY;
    let mut worst: f64 = 0.0;
    let mut out = None;
    let repeats = REPEATS_OVERRIDE.with(|c| c.get()).unwrap_or(REPEATS);
    for _ in 0..repeats {
        let jk0 = target_jk_builds();
        let c0 = cpu_seconds();
        let t0 = Instant::now();
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        let wall = t0.elapsed().as_secs_f64();
        let cpu = cpu_seconds() - c0;
        let jk = target_jk_builds() - jk0;
        if wall < best {
            best = wall;
            out = Some((cpu, r.iterations, jk, r.energy, r.converged));
        }
        worst = worst.max(wall);
    }
    let (cpu_s, iterations, jk_builds, energy, converged) = out.unwrap();
    let _ = label;
    Measurement {
        wall_s: best,
        cpu_s,
        iterations,
        jk_builds,
        energy,
        converged,
        spread: (worst - best) / best.max(1e-12),
    }
}

#[test]
#[ignore = "measurement harness: run deliberately on a quiet box"]
fn aurora_vs_diis_cost_on_named_systems() {
    // Named systems of genuinely different size, closed-shell RHF throughout —
    // the regime the paper's Table S1 measures and the regime this port
    // validates. (nbf noted for the reader; not asserted.)
    let systems: &[(&str, &str, &str)] = &[
        (
            "water/cc-pVDZ     (  25 bf)",
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
        ),
        (
            "methane/cc-pVDZ   (  35 bf)",
            "../../testdata/molecules/methane.xyz",
            "cc-pvdz",
        ),
        (
            "benzene/STO-3G    (  36 bf)",
            "../../testdata/molecules/benzene.xyz",
            "sto-3g",
        ),
        (
            "alkane_8/cc-pVDZ  ( 210 bf)",
            "../../testdata/molecules/alkane_8.xyz",
            "cc-pvdz",
        ),
        (
            "benzene/cc-pVDZ   ( 114 bf)",
            "../../testdata/molecules/benzene.xyz",
            "cc-pvdz",
        ),
    ];
    // alkane_16/cc-pVDZ (410 bf) is deliberately NOT in this list: one DIIS arm
    // there ran over 17 minutes, and at REPEATS = 3 for both arms it dominates
    // the harness. It lives in `aurora_vs_diis_large_case` below, to be run on
    // its own when a quiet box is available.

    println!();
    println!(
        "{:<30} {:>9} {:>9} {:>7} {:>7} {:>6} {:>6} {:>8} {:>8}",
        "system", "t_D (s)", "t_A (s)", "it_D", "it_A", "JK_D", "JK_A", "t_A/t_D", "JK_A/JK_D"
    );
    println!("{}", "-".repeat(110));

    let mut wall_ratios = Vec::new();
    let mut jk_ratios = Vec::new();

    for (label, xyz, basis_name) in systems {
        // DIIS first, AURORA second: both see the same warmed process state.
        let d = measure(label, xyz, basis_name, false);
        let a = measure(label, xyz, basis_name, true);

        let wr = a.wall_s / d.wall_s;
        let jr = a.jk_builds as f64 / d.jk_builds as f64;
        wall_ratios.push(wr);
        jk_ratios.push(jr);

        println!(
            "{:<30} {:>9.3} {:>9.3} {:>7} {:>7} {:>6} {:>6} {:>8.3} {:>8.3}",
            label, d.wall_s, a.wall_s, d.iterations, a.iterations, d.jk_builds, a.jk_builds, wr, jr
        );
        println!(
            "{:<30}   cpu {:.3}/{:.3} s   spread {:.1}%/{:.1}%   ΔE = {:.2e}   conv {}/{}",
            "",
            d.cpu_s,
            a.cpu_s,
            100.0 * d.spread,
            100.0 * a.spread,
            (a.energy - d.energy).abs(),
            d.converged,
            a.converged
        );

        assert!(
            d.converged && a.converged,
            "{label}: both runs must converge"
        );
        assert!(
            (a.energy - d.energy).abs() < 1e-8,
            "{label}: the fixed point must not move"
        );
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mw = mean(&wall_ratios);
    let mj = mean(&jk_ratios);
    println!("{}", "-".repeat(110));
    println!(
        "mean wall ratio t_A/t_D = {mw:.3}  ({:+.1}%)      mean J/K ratio = {mj:.3}  ({:+.1}%)",
        100.0 * (mw - 1.0),
        100.0 * (mj - 1.0)
    );
    println!("paper (16 direct CPU RHF pairs): 0.738 wall (-26.2%), 0.667 J/K (-33.3%)");
    println!();
    println!(
        "NOTE: a spread above a few percent means the box was not quiet enough \
         for the wall-time column to carry weight; the J/K column is immune to \
         load and is the more robust comparison."
    );

    // Deliberately NOT asserted: whether the ratio beats any particular number.
    // This harness reports; the report interprets.
}

/// The single large case, separated so it can be run alone on a quiet box.
///
/// alkane_16/cc-pVDZ is ~410 basis functions — inside the 137-1484 AO band that
/// the paper's Table S1 covers, and the only system in this file that is. Direct
/// SCF at that size takes many minutes per arm, so this runs ONE repeat of each
/// rather than [`REPEATS`], and its wall time must be read together with its cpu
/// time (a divergence between the two means the box was busy).
#[test]
#[ignore = "large measurement: run alone on a quiet box"]
fn aurora_vs_diis_large_case() {
    let label = "alkane_16/cc-pVDZ ( 410 bf)";
    let xyz = "../../testdata/molecules/alkane_16.xyz";

    let d = measure_once(xyz, "cc-pvdz", false);
    let a = measure_once(xyz, "cc-pvdz", true);
    println!();
    println!("{label}");
    println!(
        "  DIIS  : wall {:8.1} s  cpu {:8.1} s  {:3} it  {:3} J/K  E = {:.10}",
        d.wall_s, d.cpu_s, d.iterations, d.jk_builds, d.energy
    );
    println!(
        "  AURORA: wall {:8.1} s  cpu {:8.1} s  {:3} it  {:3} J/K  E = {:.10}",
        a.wall_s, a.cpu_s, a.iterations, a.jk_builds, a.energy
    );
    println!(
        "  ratios: wall {:.3}   cpu {:.3}   J/K {:.3}   ΔE = {:.2e}",
        a.wall_s / d.wall_s,
        a.cpu_s / d.cpu_s,
        a.jk_builds as f64 / d.jk_builds as f64,
        (a.energy - d.energy).abs()
    );
    assert!(d.converged && a.converged, "both runs must converge");
    assert!(
        (a.energy - d.energy).abs() < 1e-8,
        "the fixed point must not move"
    );
}
