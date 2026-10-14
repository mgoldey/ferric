//! Incremental (ΔD) vs full-rebuild DirectJK Fock on the direct (non-DF) RHF
//! path: one SCF per process, so wall time, quartet count and iteration count
//! of the two modes are measured in identical, fresh processes (#415).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-benchmarks \
//!     --example incr_fock_bench -- <xyz> <basis> <on|off> \
//!     [integral_thresh] [density_conv] [max_iter]
//! ```
//!
//! Prints one `RESULT ...` line. The mode is selected with
//! `FERRIC_SCF_INCREMENTAL` (`1` = incremental, `0` = full rebuild), set before
//! the solve. To measure the energy-change noise, run with a `density_conv`
//! the SCF cannot reach (e.g. 1e-14) and a fixed `max_iter`, and set
//! `FERRIC_SCF_TRACE=1`: the tail of the per-iteration `dE` column is the
//! noise. `scripts/incr_fock_ab.py` drives both measurements.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: incr_fock_bench <xyz> <basis> <on|off> \
             [integral_thresh] [density_conv] [max_iter]"
        );
        std::process::exit(2);
    }
    let (path, bs_name, mode) = (&args[1], &args[2], args[3].as_str());
    let incremental = match mode {
        "on" => "1",
        "off" => "0",
        _ => panic!("mode must be on or off, got {mode}"),
    };
    std::env::set_var("FERRIC_SCF_INCREMENTAL", incremental);

    let mut cfg = RhfConfig::default();
    if let Some(t) = args.get(4) {
        cfg.integral_thresh = t.parse().expect("integral_thresh");
    }
    if let Some(d) = args.get(5) {
        cfg.density_conv = d.parse().expect("density_conv");
        // Run to the density gate alone; dE is then recorded, not judged.
        cfg.energy_conv = 0.0;
    }
    if let Some(m) = args.get(6) {
        cfg.max_iter = m.parse().expect("max_iter");
    }

    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bs = basis::bundled(bs_name).unwrap();
    let mol = Molecule::load_xyz(path).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();

    let t = Instant::now();
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let wall = t.elapsed().as_secs_f64();

    println!(
        "RESULT mol={path} basis={bs_name} mode={mode} nbf={} thresh={:e} \
         converged={} iters={} quartets={} wall={wall:.3} E={:.12}",
        prep.nbasis(),
        cfg.integral_thresh,
        res.converged,
        res.iterations,
        res.computed_quartets,
        res.energy,
    );
}
