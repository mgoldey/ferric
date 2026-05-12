mod config;

use config::load_config;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: ferric <input.toml>");
        std::process::exit(2);
    }
    let cfg = match load_config(&args[1]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    if cfg.method.kind != "rhf" {
        eprintln!("error: only method.kind = \"rhf\" supported");
        std::process::exit(1);
    }
    let mol = Molecule::load_xyz(&cfg.molecule.xyz).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let bs = if let Some(name) = &cfg.basis.name {
        basis::bundled(name)
    } else if let Some(path) = &cfg.basis.path {
        basis::load_g94(path)
    } else {
        Err(ferric_core::FerricError::Basis("no basis specified".into()))
    }
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    let prep = PreparedBasis::new(&mol, &bs).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let rhf_config = RhfConfig {
        max_iter: cfg.scf.max_iter,
        energy_conv: cfg.scf.energy_conv,
        density_conv: cfg.scf.density_conv,
        diis_size: cfg.scf.diis_size,
        integral_thresh: cfg.scf.integral_thresh,
    };
    let result = solve_rhf(&mol, &prep, op, &bounds, &rhf_config).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    println!("RHF/{} on {}", bs.name, cfg.molecule.xyz);
    println!("  nbasis     = {}", prep.nbasis());
    println!("  iterations = {}", result.iterations);
    println!("  converged  = {}", result.converged);
    println!("  energy     = {:.10} Hartree", result.energy);
}
