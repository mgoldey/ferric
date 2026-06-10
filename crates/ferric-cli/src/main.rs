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
use ferric_mp2::scs::{scs_mp2, ScsMp2Config};
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_core::parallel::ParallelContext;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::uhf::solve_uhf;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::gradient::{rohf_gradient, uhf_gradient};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};

/// Run `f` on a private single-thread rayon pool.
///
/// Free-atom / proatom SCFs are tiny (one atom, ~10-30 basis functions). On the
/// global multi-thread pool, rayon's per-task coordination overhead dwarfs the
/// actual Fock-build work — a single S atom at aug-cc-pVDZ took 179 s with
/// RAYON_NUM_THREADS=8 vs 9.6 s with 1 (18× slower). Since every TS volume and
/// Hirshfeld proatom triggers such a solve, the penalty made 2nd-row molecules
/// (h2s, hcl) take 40-60 min. Confining these inner solves to one thread keeps
/// the big molecular SCF/RPA fully parallel while making the atoms fast.
fn run_serial<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    match rayon::ThreadPoolBuilder::new().num_threads(1).build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(), // if pool creation fails, just run inline
    }
}

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
    if !matches!(method, "rhf" | "uhf" | "rohf" | "ksdft" | "rimp2" | "oo-rimp2" | "att-rimp2" | "scs-mp2" | "laplace-mp2" | "pdep-rpa" | "rs-mp2-rpa") {
        eprintln!("error: unsupported method.kind = \"{method}\"; expected rhf, uhf, rohf, ksdft, rimp2, oo-rimp2, att-rimp2, scs-mp2, laplace-mp2, pdep-rpa, or rs-mp2-rpa");
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
    // For ksdft, default RI-J/RI-K to def2-universal-jkfit (required for hybrids
    // and RSH; harmless for pure DFT). User can still override via [scf].
    let (xc, df_j_default, df_k_default) = if method == "ksdft" {
        let functional = cfg.dft.functional.clone().unwrap_or_else(|| "LDA".into());
        (
            Some(functional),
            Some("def2-universal-jkfit".to_string()),
            Some("def2-universal-jkfit".to_string()),
        )
    } else if matches!(method, "pdep-rpa" | "rpa") && cfg.rpa.xc.is_some() {
        // RPA on a KS-DFT reference (RPA@PBE0 etc.): run the closed-shell KS
        // solver for the reference orbitals. Hybrids need RI-J/RI-K.
        (
            cfg.rpa.xc.clone(),
            Some("def2-universal-jkfit".to_string()),
            Some("def2-universal-jkfit".to_string()),
        )
    } else if matches!(method, "pdep-rpa" | "rpa" | "rs-mp2-rpa") {
        // RPA@HF (no xc): the HF reference SCF defaults to RI-J/RI-K with
        // def2-universal-jkfit too. Exact 4-index J/K per iteration makes the
        // HF reference 10-20× slower than the RI-JK PBE reference (hcl/aug-cc-
        // pVTZ: 505 s vs 25 s) for no benefit — the RI-JK fitting error (~µHa)
        // is far below the C6 differences we study. Keep SCF aux separate from
        // the RPA correlation aux / SR-MP2+LR-RPA correlation aux
        // (see ferric-jk-aux-convention).
        (
            None,
            Some("def2-universal-jkfit".to_string()),
            Some("def2-universal-jkfit".to_string()),
        )
    } else {
        (None, None, None)
    };
    let rhf_config = RhfConfig {
        max_iter: cfg.scf.max_iter,
        energy_conv: cfg.scf.energy_conv,
        density_conv: cfg.scf.density_conv,
        diis_size: cfg.scf.diis_size,
        integral_thresh: cfg.scf.integral_thresh,
        k_builder: cfg.scf.k_builder.clone(),
        df_j_aux: cfg.scf.df_j_aux.clone().or(df_j_default),
        df_k_aux: cfg.scf.df_k_aux.clone().or(df_k_default),
        xc,
        dft_grid: None,
        nlc_grid: None,
        level_shift: cfg.scf.level_shift.unwrap_or(0.0),
        newton_trigger: 0.0,
        ah_trigger: 0.0,
        mom_after_iter: 0,
        constraints: Vec::new(),
        cdft_lambda_tol: 1e-5,
        fractional_occ: false,
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
            "rhf" | "ksdft" => {
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
        let result = solve_uhf(&ctx, &mol, &prep, &bounds, &rhf_config).unwrap_or_else(|e| {
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
        // For pdep-rpa with open-shell molecules the UHF dispatch inside the arm
        // handles convergence; the global RHF result is not used.
        if method == "pdep-rpa" && mol.multiplicity > 1 {
            // Return a dummy result — it will be shadowed immediately in the arm.
            // The SCF failure is expected here; suppress the exit.
            let _ = e;
            // We cannot construct a valid ScfResult without running SCF.
            // Fall back: run UHF here so `result` is valid even if the arm
            // never uses it (e.g. if the match falls through to _ => unreachable!).
            solve_uhf(&ctx, &mol, &prep, &bounds, &{
                let mut c = rhf_config.clone(); c.mom_after_iter = 5; c
            }).unwrap_or_else(|e2| {
                eprintln!("error (pre-UHF): {e2}");
                std::process::exit(1);
            })
        } else {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    });

    // Ad-hoc same-basis Hirshfeld proatom: neutral free-atom densities computed
    // in the molecule's OWN basis (basis-consistent partition; fixes the legacy
    // single-Slater H-starvation). Built lazily via atomic SCF; shared by all
    // Hirshfeld consumers (charges, effective volumes, per-atom polarizability).
    let proatom_radii: Vec<f64> = (1..=600).map(|k| k as f64 * 0.05).collect(); // 0.05..30 Bohr
    let proatom_gs_mult = |z: i32| -> usize {
        match z {
            // Doublets: H, Li, B, F, Na, Al, Cl, Ga, Br (one unpaired p/s e⁻)
            1 | 3 | 5 | 9 | 11 | 13 | 17 | 31 | 35 => 2,
            // Triplets (³P): C, O, Si, S, Ge, Se
            6 | 8 | 14 | 16 | 32 | 34 => 3,
            // Quartets (⁴S): N, P, As
            7 | 15 | 33 => 4,
            _ => 1,
        }
    };
    let proatom = |z: i32, qi: i32| -> Option<ferric_rpa::properties::RadialProatom> {
        if qi != 0 || z - qi <= 0 {
            return None; // neutral only; ions via fallback
        }
        let sym = ferric_core::elements::z_to_symbol(z).unwrap_or("X");
        let axyz = format!("1\n{sym}\n{sym} 0 0 0\n");
        let amol = Molecule::parse_xyz(&axyz, 0, proatom_gs_mult(z)).ok()?;
        let aobs = PreparedBasis::new(&amol, &bs).ok()?;
        let abounds = SchwarzBounds::compute(op, &aobs).ok()?;
        let mut acfg = rhf_config.clone();
        // Run the single-atom SCF on a 1-thread pool — see run_serial.
        let adens = run_serial(|| {
            if proatom_gs_mult(z) == 1 {
                solve_rhf(&ctx, &amol, &aobs, op, &abounds, &acfg)
                    .ok()
                    .map(|r| r.density_r().to_owned())
            } else {
                acfg.mom_after_iter = 5;
                // KS-DFT free-atom solve: fractional/ensemble occupation spreads
                // the open-shell electrons equally over degenerate frontier
                // orbitals (e.g. Br 4p⁵ ²P, O/S 2p³ ³P), restoring spherical
                // symmetry so the GGA XC potential doesn't oscillate. Pure HF
                // free-atom solves don't suffer this (K is orbital-invariant in
                // the degenerate subspace), so only enable when xc is set.
                if acfg.xc.is_some() {
                    acfg.fractional_occ = true;
                }
                solve_uhf(&ctx, &amol, &aobs, &abounds, &acfg)
                    .ok()
                    .map(|r| r.density_total().to_owned())
            }
        })?;
        ferric_rpa::properties::spherically_averaged_proatom(z, &bs, &adens, &proatom_radii).ok()
    };

    match method {
        "rhf" => {
            println!("RHF/{} on {}", bs.name, cfg.molecule.xyz);
            println!("  nbasis     = {}", prep.nbasis());
            println!("  iterations = {}", result.iterations);
            println!("  converged  = {}", result.converged);
            println!("  energy     = {:.10} Hartree", result.energy);
        }
        "ksdft" => {
            let functional = cfg.dft.functional.as_deref().unwrap_or("LDA");
            println!("KS-DFT[{functional}]/{} on {}", bs.name, cfg.molecule.xyz);
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
            let omega_ang_inv = cfg.mp2.omega.unwrap_or(0.420);
            let att_config = AttenuatedMp2Config {
                omega: omega_ang_inv * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
                scaling: 1.0,
                frozen_core: cfg.mp2.frozen_core,
                screen_thresh: None,
            };
            let att_result = attenuated_ri_mp2(&mol, &prep, &dfbs, &result, &att_config)
                .unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                });
            println!(
                "Attenuated RI-MP2/{} (aux: {}, ω={:.3} Å⁻¹) on {}",
                bs.name, aux_name, omega_ang_inv, cfg.molecule.xyz
            );
            println!("  nbasis     = {}", prep.nbasis());
            println!("  RHF energy = {:.10} Hartree", result.energy);
            println!("  MP2 corr   = {:.10} Hartree", att_result.mp2_corr);
            println!("  E_OS       = {:.10} Hartree", att_result.spin_components.e_os);
            println!("  E_SS       = {:.10} Hartree", att_result.spin_components.e_ss);
            println!("  Total      = {:.10} Hartree", att_result.total_energy);
        }
        "rs-mp2-rpa" => {
            let aux_name = cfg.mp2.auxbasis.as_deref().unwrap_or("cc-pvdz-ri");
            let aux_bs = basis::bundled(aux_name).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            let omega_ang_inv = cfg.mp2.omega.unwrap_or(0.420);
            let rs_cfg = ferric_rpa::rs_mp2_rpa::RsMp2RpaConfig {
                omega: omega_ang_inv * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
                frozen_core: cfg.mp2.frozen_core,
                ..Default::default()
            };
            let r = ferric_rpa::rs_mp2_rpa::rs_mp2_lr_rpa(&mol, &prep, &dfbs, &result, &rs_cfg)
                .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1); });
            println!(
                "RS-MP2-RPA/{} (aux: {}, ω={:.3} Å⁻¹) on {}",
                bs.name, aux_name, omega_ang_inv, cfg.molecule.xyz
            );
            println!("  nbasis     = {}", prep.nbasis());
            println!("RS-MP2-RPA (ω = {omega_ang_inv:.3} Å⁻¹ = {:.4} Bohr⁻¹)", rs_cfg.omega);
            println!("  E(MP2, Coulomb)      = {:>16.10} Hartree", r.e_mp2_full);
            println!("  E(SR-MP2, erfc)      = {:>16.10} Hartree", r.e_sr_mp2);
            println!("  E(LR-MP2, erf)       = {:>16.10} Hartree", r.e_lr_mp2);
            println!("  E(dMP2, erf)         = {:>16.10} Hartree", r.e_dmp2_lr);
            println!("  E(dRPA, erf)         = {:>16.10} Hartree", r.e_drpa_lr);
            println!("  E_corr naive (A)     = {:>16.10} Hartree   [diagnostic: misses SR×LR cross terms]", r.e_corr_naive);
            println!("  E_corr Δ-form (B)    = {:>16.10} Hartree", r.e_corr);
            println!("  Total energy         = {:>16.10} Hartree", r.total_energy);
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
            // For open-shell molecules (multiplicity > 1) re-run with UHF + MOM so
            // the reference is converged, then dispatch to the unrestricted RPA.
            // Shadow `result` so the rest of the arm (NPZ export, properties) uses
            // the correct SCF density.
            let (rpa_result, ref_label, result) = if mol.multiplicity > 1 {
                let mut uhf_cfg = rhf_config.clone();
                // MOM after 5 DIIS iters prevents orbital reordering on open-shell atoms.
                uhf_cfg.mom_after_iter = 5;
                let uhf_result = solve_uhf(&ctx, &mol, &prep, &bounds, &uhf_cfg)
                    .unwrap_or_else(|e| {
                        eprintln!("error (UHF): {e}");
                        std::process::exit(1);
                    });
                let rr = ferric_rpa::run_u_pdep_rpa(&mol, &prep, &dfbs, op, &uhf_result, &rpa_cfg)
                    .unwrap_or_else(|e| {
                        eprintln!("error (U-PDEP-RPA): {e}");
                        std::process::exit(1);
                    });
                (rr, "UHF", uhf_result)
            } else {
                let rr = run_pdep_rpa(&mol, &prep, &dfbs, op, &result, &rpa_cfg)
                    .unwrap_or_else(|e| {
                        eprintln!("error: {e}");
                        std::process::exit(1);
                    });
                (rr, "RHF", result)
            };
            println!(
                "PDEP-RPA/{} (aux: {}) on {}",
                bs.name, aux_name, cfg.molecule.xyz
            );
            println!("  nbasis     = {}", prep.nbasis());
            println!("{ref_label} energy:            {:>20.10} Hartree", result.energy);
            println!("RPA correlation:       {:>20.10} Hartree", rpa_result.e_rpa);
            println!("Total ({ref_label}+RPA):       {:>20.10} Hartree", result.energy + rpa_result.e_rpa);
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
                    pdep_polarizability_becke,
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
                    match pdep_polarizability_becke(
                        &mol, &prep, &bs, &dfbs, &result, op, &rpa_cfg,
                    ) {
                        Ok(v) => {
                            println!(
                                "Per-atom Becke α (iso, a.u.): {:?}",
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
                    match hirshfeld_charges(&mol, &bs, result.density_total(), Some(&proatom)) {
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

                // --- C6 dispersion (Phase 1: Tkatchenko-Scheffler model) ---
                let compute_c6 = cfg.rpa.compute_c6.unwrap_or(true);
                let mut c6_freqs_v: Vec<f64> = Vec::new();
                let mut c6_weights_v: Vec<f64> = Vec::new();
                let mut alpha_dyn_v: Vec<Vec<[[f64; 3]; 3]>> = Vec::new();
                let mut c6_iso_opt: Option<ndarray::Array2<f64>> = None;
                let mut c6_aniso_v: Vec<Vec<[[f64; 3]; 3]>> = Vec::new();
                if compute_c6 {
                    use ferric_rpa::dispersion::free_atom_ref::ts_free_atom;
                    use ferric_rpa::dispersion::{
                        casimir_polder_c6, pdep_dynamic_polarizability,
                        ts_dynamic_polarizability, DispersionPartition,
                    };
                    use ferric_rpa::properties::{
                        atomic_effective_volumes_hirshfeld,
                        pdep_polarizability_hirshfeld,
                    };
                    use ferric_rpa::quadrature::build_quadrature;

                    let use_pdep = cfg.rpa.c6_source.as_deref() == Some("pdep");
                    // Default partition: Hirshfeld for PDEP (correct anisotropy via
                    // proatom sum rule), Becke for TS (only affects alpha_static shape;
                    // volume ratio always uses Hirshfeld regardless).
                    let partition = match cfg.rpa.c6_partition.as_deref() {
                        Some("becke") => DispersionPartition::Becke,
                        Some("hirshfeld") => DispersionPartition::Hirshfeld,
                        _ => if use_pdep { DispersionPartition::Hirshfeld } else { DispersionPartition::Becke },
                    };

                    let res_opt = if use_pdep {
                        // Phase 2: PDEP-RPA dynamic α(iω). Origin-independent for
                        // the molecular total AND the per-atom intrinsic α^A
                        // (atom-centred (r−R_A); bond-axis anisotropy is a
                        // coupled/molecular property, not per-atom). Uses the
                        // shared ad-hoc same-basis Hirshfeld proatom (built once
                        // above) so the per-atom partition is basis-consistent.
                        match pdep_dynamic_polarizability(
                            &mol, &prep, &bs, &dfbs, &result, op, &rpa_cfg, partition,
                            Some(&proatom),
                        ) {
                            Ok(dp) => {
                                let res = casimir_polder_c6(&dp);
                                println!(
                                    "Computed PDEP-RPA C6: {} atoms, {} freqs; molecular C6 = {:.3} a.u.",
                                    mol.atoms.len(), dp.freqs.len(), res.c6_molecular_iso
                                );
                                Some(res)
                            }
                            Err(e) => {
                                eprintln!("warning: PDEP-RPA C6 failed: {e}");
                                None
                            }
                        }
                    } else {
                        // Phase 1: Tkatchenko-Scheffler single-pole model.
                        let alpha_static: Vec<[[f64; 3]; 3]> =
                            if partition == DispersionPartition::Hirshfeld {
                                pdep_polarizability_hirshfeld(
                                    &mol, &prep, &bs, &dfbs, &result, op, &rpa_cfg, Some(&proatom),
                                )
                                .unwrap_or_else(|_| vec![[[0.0; 3]; 3]; mol.atoms.len()])
                            } else {
                                match alpha_atomic_vec.as_ref() {
                                    Some(v) => v.clone(),
                                    None => pdep_polarizability_becke(
                                        &mol, &prep, &bs, &dfbs, &result, op, &rpa_cfg,
                                    )
                                    .unwrap_or_else(|_| vec![[[0.0; 3]; 3]; mol.atoms.len()]),
                                }
                            };
                        // TS volumes must always use Hirshfeld partition — TS was
                        // parameterized with Hirshfeld volumes (TS PRL 2009). Becke
                        // volumes blow up for π-system H atoms (vol_ratio >> 1)
                        // because Becke is atom-size-blind; Hirshfeld proatom weights
                        // correctly compress H relative to C. The c6_partition setting
                        // only governs the alpha_static shape tensor, not these volumes.
                        let vols = atomic_effective_volumes_hirshfeld(
                            &mol, &bs, result.density_total(), Some(&proatom),
                        )
                        .unwrap_or_else(|_| vec![1.0; mol.atoms.len()]);
                        let z: Vec<usize> = mol.atoms.iter().map(|a| a.z as usize).collect();

                        // Compute free-atom vol_free using Hirshfeld on isolated atoms.
                        // For a single atom Hirshfeld weight = 1 everywhere (only one
                        // proatom), so this gives ∫ ρ_free(r) |r|³ dr — same physics
                        // as the molecular Hirshfeld integral, consistent denominator.
                        let mut vol_free_computed: std::collections::HashMap<usize, f64> =
                            std::collections::HashMap::new();
                        for &zi in z.iter().collect::<std::collections::HashSet<_>>() {
                            let sym = ferric_core::elements::z_to_symbol(zi as i32)
                                .unwrap_or("X");
                            let free_xyz = format!("1\n{sym}\n{sym} 0 0 0\n");
                            // Correct atomic ground-state multiplicities (3P for
                            // C/O/Si/S, etc.). Reuse the proatom map — the prior
                            // ad-hoc match here gave C/O/S a singlet, which is
                            // wrong physics and HANGS the restricted SCF for S.
                            let mult = proatom_gs_mult(zi as i32);
                            if let Ok(free_mol) = Molecule::parse_xyz(&free_xyz, 0, mult) {
                                if let Ok(free_obs) = PreparedBasis::new(&free_mol, &bs) {
                                    let free_bounds = SchwarzBounds::compute(op, &free_obs)
                                        .unwrap_or_else(|_| SchwarzBounds::compute(op, &prep).unwrap());
                                    let mut free_cfg = rhf_config.clone();
                                    free_cfg.mom_after_iter = if mult > 1 { 5 } else { 0 };
                                    // 1-thread pool for the tiny atom solve — see run_serial.
                                    //
                                    // The free-atom volume must be on the SAME scale (same xc) as
                                    // the molecular volume (vols[i]) or the ratio is meaningless.
                                    // Open-shell xc atoms (³P: O/S/Si) used to NOT converge — their
                                    // degenerate p-shell makes the GGA potential orientation-
                                    // dependent and the SCF oscillates forever. Fractional/ensemble
                                    // occupation (fractional_occ) spreads the open-shell electrons
                                    // equally over the degenerate p orbitals, restoring spherical
                                    // symmetry and converging the UKS-PBE atom on the *consistent*
                                    // scale. The HF fallback below remains as a last-resort safety
                                    // net (a scale-mismatched table value is the worst case and
                                    // should now never be hit for O/S/Si).
                                    if mult > 1 {
                                        free_cfg.fractional_occ = true;
                                    }
                                    let solve_free = |cfg: &RhfConfig| -> Option<ndarray::Array2<f64>> {
                                        if mult > 1 {
                                            solve_uhf(&ctx, &free_mol, &free_obs, &free_bounds, cfg)
                                                .ok().map(|r| r.density_total().to_owned())
                                        } else {
                                            solve_rhf(&ctx, &free_mol, &free_obs, op, &free_bounds, cfg)
                                                .ok().map(|r| r.density_r().to_owned())
                                        }
                                    };
                                    let free_density = run_serial(|| {
                                        solve_free(&free_cfg).or_else(|| {
                                            // xc solve failed — retry pure HF/UHF for a converged,
                                            // scale-consistent density.
                                            let mut hf_cfg = free_cfg.clone();
                                            hf_cfg.xc = None;
                                            solve_free(&hf_cfg)
                                        })
                                    });
                                    if let Some(d) = free_density {
                                        // Single free atom: Hirshfeld weight = 1
                                        // everywhere (one proatom), so the
                                        // reference volume is partition-independent
                                        // — None (legacy path) is exact here.
                                        if let Ok(fv) = atomic_effective_volumes_hirshfeld(
                                            &free_mol, &bs, &d, None,
                                        ) {
                                            vol_free_computed.insert(zi, fv[0]);
                                        }
                                    }
                                }
                            }
                        }

                        let ratio: Vec<f64> = z
                            .iter()
                            .enumerate()
                            .map(|(i, &zi)| {
                                // Use the ferric-computed free-atom volume (consistent scale
                                // with the molecular vols[i]). The TS-PRL table v_free is a
                                // LAST resort only — it is on a different integration scale,
                                // so a ratio built from it is unreliable (see the free-atom
                                // solve above, which now retries pure HF to avoid this path).
                                let vf = vol_free_computed.get(&zi).copied()
                                    .or_else(|| ts_free_atom(zi).map(|(_, _, v)| v))
                                    .unwrap_or(1.0);
                                if vf > 1e-10 { vols[i] / vf } else { 1.0 }
                            })
                            .collect();
                        let (freqs, weights) = build_quadrature(&rpa_cfg.quadrature);
                        let is_mbd = cfg.rpa.c6_source.as_deref() == Some("mbd");
                        let dp = if is_mbd {
                            let positions: Vec<[f64; 3]> =
                                mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
                            ferric_rpa::dispersion::mbd_dynamic_polarizability(
                                &positions, &z, &ratio, &alpha_static, &freqs, &weights,
                            )
                        } else {
                            ts_dynamic_polarizability(&z, &ratio, &alpha_static, &freqs, &weights)
                        };
                        let ts_res = casimir_polder_c6(&dp);
                        println!(
                            "Computed {} C6: {} atoms; molecular C6 = {:.3} a.u.",
                            if is_mbd { "MBD" } else { "TS" },
                            z.len(),
                            ts_res.c6_molecular_iso
                        );
                        Some(ts_res)
                    };

                    if let Some(res) = res_opt {
                        c6_freqs_v = res.per_atom_dynamic.freqs.clone();
                        c6_weights_v = res.per_atom_dynamic.weights.clone();
                        alpha_dyn_v = res.per_atom_dynamic.per_atom.clone();
                        c6_iso_opt = Some(res.c6_iso_pair.clone());
                        c6_aniso_v = res.c6_aniso_pair.clone();
                    }
                }

                if let Err(e) = export_npz(
                    npz_path,
                    None,
                    if result.spin == ferric_scf::result::Spin::Restricted { Some(result.eps_r()) } else { None },
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
                    if c6_freqs_v.is_empty() { None } else { Some(c6_freqs_v.as_slice()) },
                    if c6_weights_v.is_empty() { None } else { Some(c6_weights_v.as_slice()) },
                    if alpha_dyn_v.is_empty() { None } else { Some(alpha_dyn_v.as_slice()) },
                    c6_iso_opt.as_ref(),
                    if c6_aniso_v.is_empty() { None } else { Some(c6_aniso_v.as_slice()) },
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
