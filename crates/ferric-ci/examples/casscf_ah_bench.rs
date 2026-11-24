//! Timing harness for the CASSCF orbital optimizers.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 FERRIC_TIMING=1 cargo run --release -p ferric-ci \
//!     --example casscf_ah_bench -- <xyz> <basis> <n_core> <n_active> <n_alpha> <n_beta> \
//!     <bfgs|ah> [quadratic|<eta>]
//! ```
//!
//! `<xyz>` is a file under `testdata/molecules`. The start is the exact-J/K
//! RHF of [`ferric_ci::driver::rhf_start`] with its defaults (what the CLI
//! runs); the CASSCF uses [`CasScfConfig`] defaults except the active space,
//! the optimizer and, for `ah`, the Davidson forcing
//! ([`CasScfConfig::ah_forcing`]). One line on stdout: the energy to 17
//! significant digits, `‖g‖`, macro iterations, convergence, and the RHF,
//! CASSCF and total wall times. With `FERRIC_TIMING=1` the CASSCF phase table
//! on stderr counts the Hessian-vector products.

use ferric_ci::casscf::{run_casscf, AhForcing, CasScfConfig, CasScfOptimizer};
use ferric_ci::driver::{rhf_start, ScfStart};
use ferric_core::mol::Molecule;
use ferric_scf::PreparedSystem;
use std::path::PathBuf;
use std::time::Instant;

fn forcing(arg: Option<&String>) -> AhForcing {
    match arg.map(String::as_str) {
        None => CasScfConfig::default().ah_forcing,
        Some("quadratic") => AhForcing::Quadratic,
        Some(s) => AhForcing::Constant(s.parse().expect("eta must be a number")),
    }
}

fn main() {
    ferric_core::blas_threads::init_threading();
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 8 {
        eprintln!(
            "usage: casscf_ah_bench <xyz> <basis> <n_core> <n_active> <n_alpha> <n_beta> <bfgs|ah> [quadratic|<eta>]"
        );
        std::process::exit(2);
    }
    let num = |i: usize| -> usize { a[i].parse().expect("integer argument") };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/molecules")
        .join(&a[1]);
    let t0 = Instant::now();
    let mol = Molecule::load_xyz(path.to_str().expect("utf-8 path")).expect("molecule");
    let bs = ferric_core::basis::bundled(&a[2]).expect("basis");
    let sys = PreparedSystem::new(mol, &bs).expect("basis on molecule");
    let (mol, prep) = (sys.mol(), sys.basis());
    let rhf = rhf_start(&sys, ScfStart::default(), None).expect("RHF");
    let t_rhf = t0.elapsed().as_secs_f64();
    let cfg = CasScfConfig {
        n_core: num(3),
        n_active: num(4),
        n_elec_active: (num(5), num(6)),
        optimizer: CasScfOptimizer::parse_config_str(&a[7]).expect("optimizer"),
        ah_forcing: forcing(a.get(8)),
        ..Default::default()
    };
    let t1 = Instant::now();
    let r = run_casscf(mol, prep, &rhf, &cfg).expect("CASSCF");
    let t_cas = t1.elapsed().as_secs_f64();
    println!(
        "E {:.16e} gnorm {:.3e} iterations {} converged {} rhf_s {:.3} casscf_s {:.3} total_s {:.3}",
        r.e_total,
        r.grad_norm,
        r.iterations,
        r.converged,
        t_rhf,
        t_cas,
        t0.elapsed().as_secs_f64()
    );
}
