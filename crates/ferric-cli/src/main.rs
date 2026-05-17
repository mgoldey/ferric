mod config;

use config::load_config;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config};
use ferric_mp2::laplace::laplace_ri_mp2;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::scs::{scs_mp2, scs_mp2_2terfc, ScsMp2Config, ScsMp2TerfcConfig};
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_core::parallel::ParallelContext;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::uhf::solve_uhf;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::gradient::{rohf_gradient, uhf_gradient};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};

fn main() {
    let ctx = ParallelContext::new();
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
    let task = cfg.method.task.as_str();
    if !matches!(method, "rhf" | "uhf" | "rohf" | "rimp2" | "oo-rimp2" | "att-rimp2" | "scs-mp2" | "scs-mp2-2terfc" | "laplace-mp2" | "pdep-rpa") {
        eprintln!("error: unsupported method.kind = \"{method}\"; expected rhf, uhf, rohf, rimp2, oo-rimp2, att-rimp2, scs-mp2, scs-mp2-2terfc, laplace-mp2, or pdep-rpa");
        std::process::exit(1);
    }
    if !matches!(task, "energy" | "optimize") {
        eprintln!("error: unsupported method.task = \"{task}\"; expected energy or optimize");
        std::process::exit(1);
    }
    let mol = Molecule::load_xyz_with_charge(&cfg.molecule.xyz, cfg.molecule.charge, cfg.molecule.multiplicity).unwrap_or_else(|e| {
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
        k_builder: cfg.scf.k_builder.clone(),
        df_j_aux: cfg.scf.df_j_aux.clone(),
        df_k_aux: cfg.scf.df_k_aux.clone(),
    };

    if task == "optimize" {
        let opt_config = OptimizeConfig {
            max_steps: cfg.optimize.max_steps.unwrap_or(100),
            g_max_thresh: cfg.optimize.g_max_thresh.unwrap_or(4.5e-4),
            g_rms_thresh: cfg.optimize.g_rms_thresh.unwrap_or(3.0e-4),
            e_conv: cfg.optimize.e_conv.unwrap_or(1e-6),
            trust_radius: cfg.optimize.trust_radius.unwrap_or(0.1),
        };
        match method {
            "rhf" => {
                let opt_result = optimize_geometry(&ctx, &mol, &bs.name, op, &rhf_config, &opt_config)
                    .unwrap_or_else(|e| {
                        eprintln!("error during optimization: {e}");
                        std::process::exit(1);
                    });
                println!("\nFinal Optimized Geometry (Bohr):");
                for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                    println!("  {:2} {:2} {:12.8} {:12.8} {:12.8}", i, atom.symbol, atom.x, atom.y, atom.zpos);
                }
                println!("\nOptimization Result:");
                println!("  converged  = {}", opt_result.converged);
                println!("  steps      = {}", opt_result.steps);
                println!("  final E    = {:.10} Hartree", opt_result.energy);
            }
            "pdep-rpa" => {
                let aux_name = cfg.rpa.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
                let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                });
                let scheme = match cfg.rpa.quadrature.as_deref().unwrap_or("gauss-legendre") {
                    "minimax" | "mm" => QuadratureScheme::MiniMax,
                    _ => QuadratureScheme::GaussLegendre,
                };
                let rpa_cfg = PdepRpaConfig {
                    frozen_core: cfg.rpa.frozen_core,
                    trunc_thresh: cfg.rpa.trunc_thresh.unwrap_or(1e-4),
                    davidson_max_vecs: 0,
                    davidson_conv_thresh: cfg.rpa.davidson_conv_thresh.unwrap_or(1e-8),
                    quadrature: QuadratureConfig {
                        scheme,
                        n_points: cfg.rpa.n_quad.unwrap_or(16),
                        u0: cfg.rpa.u0.unwrap_or(0.5),
                    },
                    sternheimer: SternheimerConfig::default(),
                    run_diagnostics: false,
                    eigensolver: ferric_rpa::Eigensolver::default(),
                    chi0_backend: ferric_rpa::config::Chi0Backend::default(),
                    chi0_sparsity: ferric_rpa::config::Chi0Sparsity::default(),
                };
                let h_fd = 5e-4;
                let opt_result =
                    ferric_rpa::optimize::optimize_geometry_rpa(&mol, &bs, &aux_bs, op, &rpa_cfg, &opt_config, h_fd)
                        .unwrap_or_else(|e| {
                            eprintln!("error during RPA optimization: {e}");
                            std::process::exit(1);
                        });
                println!("\nFinal Optimized Geometry (Bohr):");
                for (i, atom) in opt_result.mol.atoms.iter().enumerate() {
                    println!("  {:2} {:2} {:12.8} {:12.8} {:12.8}", i, atom.symbol, atom.x, atom.y, atom.zpos);
                }
                println!("\nRPA Optimization Result:");
                println!("  converged  = {}", opt_result.converged);
                println!("  steps      = {}", opt_result.steps);
                println!("  final E    = {:.10} Hartree (RHF + RPA)", opt_result.energy);
            }
            _ => {
                eprintln!("error: geometry optimization is currently only supported for method.kind = \"rhf\" or \"pdep-rpa\"");
                std::process::exit(1);
            }
        }
        return;
    }

    if method == "uhf" {
        let result = solve_uhf(&ctx, &mol, &prep, op, &bounds, &rhf_config).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
        let s_ov = ferric_integrals::oneelectron::overlap(&prep);
        let nelec = mol.nelec() as i64;
        let two_s = mol.multiplicity as i64 - 1;
        let nocc_a = ((nelec + two_s) / 2) as usize;
        let nocc_b = ((nelec - two_s) / 2) as usize;
        let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
        let s_ideal = s_true * (s_true + 1.0);
        let c_a = result.mos_a();
        let c_b = result.mos_b();
        let overlap_ab = c_a
            .slice(ndarray::s![.., ..nocc_a])
            .t()
            .dot(&s_ov)
            .dot(&c_b.slice(ndarray::s![.., ..nocc_b]));
        let sum_sq: f64 = overlap_ab.iter().map(|v| v * v).sum();
        let s2 = s_ideal + (nocc_b as f64) - sum_sq;
        println!("UHF/{} on {}", bs.name, cfg.molecule.xyz);
        println!("  nbasis     = {}", prep.nbasis());
        println!("  mult       = {} (nocc_a={}, nocc_b={})", mol.multiplicity, nocc_a, nocc_b);
        println!("  iterations = {}", result.iterations);
        println!("  converged  = {}", result.converged);
        println!("  energy     = {:.10} Hartree", result.energy);
        println!("  <S^2>      = {:.6} (ideal {:.6})", s2, s_ideal);
        if task == "optimize" {
            // TODO: UHF geometry optimization not yet wired; print the gradient.
            match uhf_gradient(&mol, &prep, op, &bounds, &result) {
                Ok(g) => {
                    println!("UHF gradient (Hartree/Bohr):");
                    for (i, atom) in mol.atoms.iter().enumerate() {
                        println!("  {:2} {:2} {:14.8} {:14.8} {:14.8}",
                                 i, atom.symbol, g[(i,0)], g[(i,1)], g[(i,2)]);
                    }
                }
                Err(e) => eprintln!("UHF gradient error: {e}"),
            }
        }
        return;
    }

    if method == "rohf" {
        let result = solve_rohf(&ctx, &mol, &prep, op, &bounds, &rhf_config).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
        let nelec = mol.nelec() as i64;
        let two_s = mol.multiplicity as i64 - 1;
        let nocc_open = two_s as usize;
        let nocc_double = ((nelec - two_s) / 2) as usize;
        let s_true = 0.5 * two_s as f64;
        let s_ideal = s_true * (s_true + 1.0);
        println!("ROHF/{} on {}", bs.name, cfg.molecule.xyz);
        println!("  nbasis     = {}", prep.nbasis());
        println!(
            "  mult       = {} (nocc_double={}, nocc_open={})",
            mol.multiplicity, nocc_double, nocc_open
        );
        println!("  iterations = {}", result.iterations);
        println!("  converged  = {}", result.converged);
        println!("  energy     = {:.10} Hartree", result.energy);
        println!("  <S^2>      = {:.6} (exact by construction)", s_ideal);
        if task == "optimize" {
            // TODO: ROHF geometry optimization not yet wired; print the gradient.
            match rohf_gradient(&mol, &prep, op, &bounds, &result) {
                Ok(g) => {
                    println!("ROHF gradient (Hartree/Bohr):");
                    for (i, atom) in mol.atoms.iter().enumerate() {
                        println!("  {:2} {:2} {:14.8} {:14.8} {:14.8}",
                                 i, atom.symbol, g[(i,0)], g[(i,1)], g[(i,2)]);
                    }
                }
                Err(e) => eprintln!("ROHF gradient error: {e}"),
            }
        }
        return;
    }

    let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &rhf_config).unwrap_or_else(|e| {
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
        "att-rimp2" => {
            let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let angstrom_to_bohr = 1.8897259886;
            let att_config = AttenuatedMp2Config {
                r0: cfg.mp2.r0.unwrap_or(1.05) * angstrom_to_bohr,
                scaling: 1.0,
                frozen_core: cfg.mp2.frozen_core,
            };
            let att_result = attenuated_ri_mp2(&mol, &prep, &dfbs, &result, &att_config)
                .unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                });
            let r0_ang = cfg.mp2.r0.unwrap_or(1.05);
            println!(
                "Attenuated RI-MP2/{} (aux: {}, r0={:.2} A) on {}",
                bs.name, aux_name, r0_ang, cfg.molecule.xyz
            );
            println!("  nbasis     = {}", prep.nbasis());
            println!("  RHF energy = {:.10} Hartree", result.energy);
            println!("  MP2 corr   = {:.10} Hartree", att_result.mp2_corr);
            println!("  E_OS       = {:.10} Hartree", att_result.spin_components.e_os);
            println!("  E_SS       = {:.10} Hartree", att_result.spin_components.e_ss);
            println!("  Total      = {:.10} Hartree", att_result.total_energy);
        }
        "scs-mp2" => {
            let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let scs_config = ScsMp2Config {
                c_os: cfg.mp2.c_os.unwrap_or(6.0 / 5.0),
                c_ss: cfg.mp2.c_ss.unwrap_or(1.0 / 3.0),
                frozen_core: cfg.mp2.frozen_core,
            };
            let scs_result = scs_mp2(&mol, &prep, &dfbs, &result, &scs_config)
                .unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                });
            println!(
                "SCS-MP2/{} (aux: {}, c_OS={:.3}, c_SS={:.3}) on {}",
                bs.name, aux_name, scs_config.c_os, scs_config.c_ss, cfg.molecule.xyz
            );
            println!("  nbasis     = {}", prep.nbasis());
            println!("  RHF energy = {:.10} Hartree", result.energy);
            println!("  SCS corr   = {:.10} Hartree", scs_result.scs_corr);
            println!("  E_OS       = {:.10} Hartree", scs_result.e_os);
            println!("  E_SS       = {:.10} Hartree", scs_result.e_ss);
            println!("  Total      = {:.10} Hartree", scs_result.total_energy);
        }
        "scs-mp2-2terfc" => {
            let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let angstrom_to_bohr = 1.8897259886;
            let terfc_config = ScsMp2TerfcConfig {
                r0_bonded: cfg.mp2.r0_bonded.unwrap_or(0.75) * angstrom_to_bohr,
                r0_nonbonded: cfg.mp2.r0_nonbonded.unwrap_or(1.05) * angstrom_to_bohr,
                c_os: cfg.mp2.c_os.unwrap_or(1.27),
                c_ss: cfg.mp2.c_ss.unwrap_or(4.05),
                frozen_core: cfg.mp2.frozen_core,
            };
            let terfc_result = scs_mp2_2terfc(&mol, &prep, &dfbs, &result, &terfc_config)
                .unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                });
            let r0b = cfg.mp2.r0_bonded.unwrap_or(0.75);
            let r0n = cfg.mp2.r0_nonbonded.unwrap_or(1.05);
            println!(
                "SCS-MP2(2terfc)/{} (aux: {}, r0_1={:.2} A, r0_2={:.2} A) on {}",
                bs.name, aux_name, r0b, r0n, cfg.molecule.xyz
            );
            println!("  nbasis     = {}", prep.nbasis());
            println!("  RHF energy = {:.10} Hartree", result.energy);
            println!("  SCS corr   = {:.10} Hartree", terfc_result.scs_corr);
            println!("  E_OS       = {:.10} Hartree", terfc_result.e_os);
            println!("  E_SS       = {:.10} Hartree", terfc_result.e_ss);
            println!("  Total      = {:.10} Hartree", terfc_result.total_energy);
        }
        "laplace-mp2" => {
            let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let n_quad = cfg.mp2.n_quad.unwrap_or(7);
            let lap_result = laplace_ri_mp2(
                &mol,
                &prep,
                &dfbs,
                op,
                &result,
                n_quad,
                cfg.mp2.frozen_core,
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            println!(
                "Laplace RI-MP2/{} (aux: {}, n_quad={}) on {}",
                bs.name, aux_name, n_quad, cfg.molecule.xyz
            );
            println!("  nbasis     = {}", prep.nbasis());
            println!("  RHF energy = {:.10} Hartree", result.energy);
            println!("  MP2 corr   = {:.10} Hartree", lap_result.mp2_corr);
            println!("  E_OS       = {:.10} Hartree", lap_result.e_os);
            println!("  E_SS       = {:.10} Hartree", lap_result.e_ss);
            println!("  Total      = {:.10} Hartree", lap_result.total_energy);
        }
        "pdep-rpa" => {
            let aux_name = cfg.rpa.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let scheme = match cfg.rpa.quadrature.as_deref().unwrap_or("gauss-legendre") {
                "minimax" | "mm" => QuadratureScheme::MiniMax,
                _ => QuadratureScheme::GaussLegendre,
            };
            let rpa_cfg = PdepRpaConfig {
                frozen_core: cfg.rpa.frozen_core,
                trunc_thresh: cfg.rpa.trunc_thresh.unwrap_or(1e-4),
                davidson_max_vecs: 0,
                davidson_conv_thresh: cfg.rpa.davidson_conv_thresh.unwrap_or(1e-6),
                quadrature: QuadratureConfig {
                    scheme,
                    n_points: cfg.rpa.n_quad.unwrap_or(20),
                    u0: cfg.rpa.u0.unwrap_or(0.5),
                },
                sternheimer: SternheimerConfig::default(),
                run_diagnostics: cfg.rpa.run_diagnostics,
                eigensolver: ferric_rpa::Eigensolver::default(),
                chi0_backend: ferric_rpa::config::Chi0Backend::default(),
                chi0_sparsity: ferric_rpa::config::Chi0Sparsity::default(),
            };
            let rpa_result = run_pdep_rpa(&mol, &prep, &dfbs, op, &result, &rpa_cfg)
                .unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                });
            println!(
                "PDEP-RPA/{} (aux: {}) on {}",
                bs.name, aux_name, cfg.molecule.xyz
            );
            println!("  nbasis     = {}", prep.nbasis());
            println!("RHF energy:            {:>20.10} Hartree", result.energy);
            println!("RPA correlation:       {:>20.10} Hartree", rpa_result.e_rpa);
            println!("Total (RHF+RPA):       {:>20.10} Hartree", result.energy + rpa_result.e_rpa);
            println!("Eigenpotentials kept:  {} / {}", rpa_result.n_eigenpotentials, rpa_result.eigenvalues_static.len());
            if let Some(e_diag) = rpa_result.e_rpa_dft_diag {
                println!("RI-dRPA check:         {:>20.10} Hartree", e_diag);
            }
            if let Some(prefix) = cfg.rpa.export_eigpot_prefix.as_deref() {
                use ferric_export::cube::GridSpec;
                use ferric_export::export_basis_function_cube;
                let spacing = cfg.rpa.cube_spacing.unwrap_or(0.2);
                let margin = cfg.rpa.cube_margin.unwrap_or(4.0);
                let n_export = cfg.rpa.export_eigpot_count
                    .unwrap_or(10)
                    .min(rpa_result.n_eigenpotentials);
                let grid = GridSpec::bounding_box(&mol, margin, spacing);
                println!(
                    "Exporting {} eigenpotential cubes (grid {}×{}×{}, spacing {} Bohr)…",
                    n_export, grid.n_x, grid.n_y, grid.n_z, spacing
                );
                for alpha in 0..n_export {
                    let coeffs: Vec<f64> = rpa_result.eigenpotentials
                        .column(alpha).iter().copied().collect();
                    let lam = rpa_result.eigenvalues_static[alpha];
                    let path = format!("{prefix}_eigpot_{:03}.cube", alpha);
                    let comment = format!(
                        "PDEP eigenpotential α={alpha} λ(0)={lam:.6} (basis {aux_name})"
                    );
                    if let Err(e) = export_basis_function_cube(&path, &mol, &aux_bs, &grid, &coeffs, &comment) {
                        eprintln!("  warning: failed to write {}: {}", path, e);
                    } else {
                        println!("  wrote {} (λ(0)={:.6})", path, lam);
                    }
                }
            }
            // NPZ feature bundle for diffusion-model export.
            if let Some(npz_path) = cfg.rpa.export_npz.as_deref() {
                use ferric_export::export_npz;
                use ferric_rpa::properties::{
                    electric_field_at_atoms, esp_at_atoms, hirshfeld_charges, lowdin_charges,
                    pdep_polarizability_hirshfeld,
                    pdep_polarizability_static,
                };
                use ndarray::Array2;

                let compute_esp = cfg.rpa.compute_esp.unwrap_or(true);
                let compute_pol = cfg.rpa.compute_polarizability.unwrap_or(true);
                let compute_ef = cfg.rpa.compute_electric_field.unwrap_or(true);
                let compute_alpha_atomic = cfg.rpa.compute_alpha_atomic.unwrap_or(true);

                let coords_arr = {
                    let mut a = Array2::<f64>::zeros((mol.atoms.len(), 3));
                    for (i, atom) in mol.atoms.iter().enumerate() {
                        a[(i, 0)] = atom.x;
                        a[(i, 1)] = atom.y;
                        a[(i, 2)] = atom.zpos;
                    }
                    a
                };
                let znums: Vec<usize> =
                    mol.atoms.iter().map(|a| a.z as usize).collect();

                let esp_vec = if compute_esp {
                    match esp_at_atoms(&mol, &prep, result.density_total()) {
                        Ok(v) => Some(v),
                        Err(e) => {
                            eprintln!("warning: esp_at_atoms failed: {e}");
                            None
                        }
                    }
                } else {
                    None
                };

                let ef_vec = if compute_ef {
                    match electric_field_at_atoms(&mol, &prep, result.density_total()) {
                        Ok(v) => Some(v),
                        Err(e) => {
                            eprintln!("warning: electric_field_at_atoms failed: {e}");
                            None
                        }
                    }
                } else {
                    None
                };

                let alpha_arr = if compute_pol {
                    match pdep_polarizability_static(
                        &mol, &prep, &dfbs, &result, op, &rpa_cfg,
                    ) {
                        Ok(p) => {
                            println!(
                                "Polarizability α (a.u.):  iso={:.4}, principal=[{:.4}, {:.4}, {:.4}]",
                                p.iso, p.principal[0], p.principal[1], p.principal[2]
                            );
                            Some(p.tensor)
                        }
                        Err(e) => {
                            eprintln!("warning: polarizability failed: {e}");
                            None
                        }
                    }
                } else {
                    None
                };

                let alpha_atomic_vec = if compute_alpha_atomic {
                    match pdep_polarizability_hirshfeld(
                        &mol, &prep, &bs, &dfbs, &result, op, &rpa_cfg,
                    ) {
                        Ok(v) => {
                            println!(
                                "Per-atom Hirshfeld α (iso, a.u.): {:?}",
                                v.iter()
                                    .map(|t| (t[0][0] + t[1][1] + t[2][2]) / 3.0)
                                    .collect::<Vec<_>>()
                            );
                            Some(v)
                        }
                        Err(e) => {
                            eprintln!("warning: per-atom α (Hirshfeld) failed: {e}");
                            None
                        }
                    }
                } else {
                    None
                };

                let compute_dm = cfg.rpa.compute_density_matrix.unwrap_or(true);
                let dm_ref = if compute_dm { Some(result.density_total()) } else { None };

                let compute_lq = cfg.rpa.compute_lowdin_charges.unwrap_or(true);
                let lq_vec = if compute_lq {
                    match lowdin_charges(&mol, &prep, result.density_total()) {
                        Ok(q) => {
                            println!(
                                "Löwdin charges (e): {:?}",
                                q.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>()
                            );
                            Some(q)
                        }
                        Err(e) => {
                            eprintln!("warning: Löwdin charges failed: {e}");
                            None
                        }
                    }
                } else {
                    None
                };

                let compute_hq = cfg.rpa.compute_hirshfeld_charges.unwrap_or(true);
                let hq_vec = if compute_hq {
                    match hirshfeld_charges(&mol, &bs, result.density_total()) {
                        Ok(q) => {
                            println!(
                                "Hirshfeld charges (e): {:?}",
                                q.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>()
                            );
                            Some(q)
                        }
                        Err(e) => {
                            eprintln!("warning: Hirshfeld charges failed: {e}");
                            None
                        }
                    }
                } else {
                    None
                };

                if let Err(e) = export_npz(
                    npz_path,
                    None,
                    Some(result.eps_r()),
                    Some(&rpa_result.eigenpotentials),
                    None,
                    Some(&coords_arr),
                    Some(&znums),
                    esp_vec.as_deref(),
                    alpha_arr.as_ref(),
                    ef_vec.as_deref(),
                    dm_ref,
                    alpha_atomic_vec.as_deref(),
                    hq_vec.as_deref(),
                    lq_vec.as_deref(),
                ) {
                    eprintln!("warning: failed to write {}: {}", npz_path, e);
                } else {
                    println!("Wrote NPZ feature bundle: {}", npz_path);
                }
            }
        }
        _ => unreachable!(),
    }
}
