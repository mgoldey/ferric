mod config;

use config::load_config;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
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
    let method = cfg.method.kind.as_str();
    if !matches!(method, "rhf" | "rimp2" | "oo-rimp2") {
        eprintln!("error: unsupported method.kind = \"{method}\"; expected rhf, rimp2, or oo-rimp2");
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

    match method {
        "rhf" => {
            println!("RHF/{} on {}", bs.name, cfg.molecule.xyz);
            println!("  nbasis     = {}", prep.nbasis());
            println!("  iterations = {}", result.iterations);
            println!("  converged  = {}", result.converged);
            println!("  energy     = {:.10} Hartree", result.energy);
        }
        "rimp2" => {
            let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let mp2_result = ri_mp2(
                &mol,
                &prep,
                &dfbs,
                op,
                &result,
                &RiMp2Config {
                    frozen_core: cfg.mp2.frozen_core,
                },
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            println!(
                "RI-MP2/{} (aux: {}) on {}",
                bs.name, aux_name, cfg.molecule.xyz
            );
            println!("  nbasis     = {}", prep.nbasis());
            println!("  RHF energy = {:.10} Hartree", result.energy);
            println!("  MP2 corr   = {:.10} Hartree", mp2_result.mp2_corr);
            println!("  Total      = {:.10} Hartree", mp2_result.total_energy);
        }
        "oo-rimp2" => {
            let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let oo_config = OoRiMp2Config {
                frozen_core: cfg.mp2.frozen_core,
                ..Default::default()
            };
            let oo_result = oo_ri_mp2(&mol, &prep, &dfbs, op, &bounds, &result, &oo_config)
                .unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                });
            println!(
                "OO-RI-MP2/{} (aux: {}) on {}",
                bs.name, aux_name, cfg.molecule.xyz
            );
            println!("  nbasis     = {}", prep.nbasis());
            println!("  converged  = {}", oo_result.converged);
            println!("  iterations = {}", oo_result.iterations);
            println!("  grad_norm  = {:.2e}", oo_result.grad_norm);
            println!("  HF energy  = {:.10} Hartree", oo_result.hf_energy);
            println!("  MP2 corr   = {:.10} Hartree", oo_result.mp2_corr);
            println!("  Total      = {:.10} Hartree", oo_result.total_energy);
        }
        _ => unreachable!(),
    }
}
