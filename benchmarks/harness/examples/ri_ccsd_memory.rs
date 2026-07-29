//! Closed-shell CCSD ladder routes: peak RSS and per-iteration time.
//!
//! ```text
//! ri_ccsd_memory <xyz> <basis> <aux> <auto|dense|ri> mem
//! ri_ccsd_memory <xyz> <basis> <aux> <auto|dense|ri> time <iters>
//! ```
//!
//! `mem` runs CCSD to convergence and reports `VmHWM` (peak RSS). Run ONE
//! route per process: `VmHWM` is a high-water mark and never resets, so two
//! routes in one process would report the larger of the two.
//!
//! `time` isolates the per-iteration cost. It times a `max_iter = 0` call
//! (setup only: integrals, chemist blocks, and for dense the `vvvv` block)
//! and then a `max_iter = iters` call with an unreachable energy threshold,
//! so both stop on the iteration cap, and reports
//! `(t(iters) - t(0)) / iters`. Run under `OPENBLAS_NUM_THREADS=1` with a
//! fixed `RAYON_NUM_THREADS`, interleaving routes across processes.
//!
//! Library CCSD prints its own iteration chatter; grep for `RESULT`.

use ferric_cc::ccsd_closed_shell::{ccsd_closed_shell, closed_shell_ladder_route};
use ferric_cc::{CcConfig, CcsdLadder};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn vm_hwm_bytes() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let rest = s.lines().find_map(|l| l.strip_prefix("VmHWM:"))?;
    let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
    Some(kb * 1024)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: ri_ccsd_memory <xyz> <basis> <aux> <auto|dense|ri> <mem|time> [iters]");
        std::process::exit(2);
    }
    let (xyz, obs_name, aux_name, mode) = (&a[1], &a[2], &a[3], &a[5]);
    let ladder = CcsdLadder::parse_config_str(&a[4])?;

    let mol = Molecule::load_xyz(xyz)?;
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name)?)?;
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name)?)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs)?;
    let rhf_cfg = RhfConfig {
        energy_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &rhf_cfg,
    )?;
    let nbas = obs.nbasis();
    let nocc = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
    let (nv, naux) = (nbas - nocc, dfbs.nbasis());
    let cfg = CcConfig {
        energy_conv: 1e-8,
        ladder,
        ..Default::default()
    };
    let route = closed_shell_ladder_route(&obs, &dfbs, &rhf, &cfg)?;
    println!("system={xyz} nbas={nbas} naux={naux} nocc={nocc} nv={nv} route={route:?}");

    match mode.as_str() {
        "mem" => {
            let t0 = Instant::now();
            let r = ccsd_closed_shell(&mol, &obs, &dfbs, op, &rhf, &cfg)?;
            println!(
                "RESULT mem route={route:?} nv={nv} E_corr={:.10} wall={:.2}s peak_rss_gb={:.3}",
                r.correlation_energy,
                t0.elapsed().as_secs_f64(),
                vm_hwm_bytes().unwrap_or(0) as f64 / 1e9
            );
        }
        "time" => {
            let iters: usize = a.get(6).map(|s| s.parse()).transpose()?.unwrap_or(4);
            let capped = |max_iter: usize| -> f64 {
                let c = CcConfig {
                    max_iter,
                    energy_conv: 0.0,
                    ..cfg.clone()
                };
                let t0 = Instant::now();
                let r = ccsd_closed_shell(&mol, &obs, &dfbs, op, &rhf, &c);
                assert!(r.is_err(), "an unreachable threshold must stop on the cap");
                t0.elapsed().as_secs_f64()
            };
            let setup = capped(0);
            let total = capped(iters);
            println!(
                "RESULT time route={route:?} nv={nv} setup={setup:.3}s iters={iters} \
                 total={total:.3}s per_iter={:.4}s",
                (total - setup) / iters as f64
            );
        }
        other => return Err(format!("mode must be 'mem' or 'time', got {other}").into()),
    }
    Ok(())
}
