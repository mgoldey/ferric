//! Sparse PDEP-RPA scaling experiment.
//!
//! Sweeps a set of test systems (linear alkanes, compact 3D, drug-scale)
//! across `Chi0Sparsity` configurations (Dense, BoysScreened{1e-3, 1e-4, 1e-5}).
//! Emits CSV-formatted rows to stdout with per-stage wall-clock timings,
//! retained pair counts, and RPA energies.
//!
//! Usage:
//!   cargo run --release -p ferric-rpa --example sparse_scaling
//!
//! Tip: set OPENBLAS_NUM_THREADS=1 to avoid the rayon × BLAS oversubscription
//! discussed in `ferric-rpa::lib`'s threading note.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Sparsity, PdepRpaConfig};
use ferric_rpa::{run_pdep_rpa, screen};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

#[derive(Clone, Copy)]
struct SparsityCfg {
    label: &'static str,
    sparsity: Chi0Sparsity,
}

fn cfgs() -> Vec<SparsityCfg> {
    vec![
        SparsityCfg { label: "Dense", sparsity: Chi0Sparsity::Dense },
        SparsityCfg { label: "Boys-1e-3", sparsity: Chi0Sparsity::BoysScreened { thresh: 1e-3 } },
        SparsityCfg { label: "Boys-1e-4", sparsity: Chi0Sparsity::BoysScreened { thresh: 1e-4 } },
        SparsityCfg { label: "Boys-1e-5", sparsity: Chi0Sparsity::BoysScreened { thresh: 1e-5 } },
    ]
}

struct System {
    name: &'static str,
    path: &'static str,
    /// Frozen core (1 per heavy atom is the conventional FC for C/N/O).
    frozen_core: usize,
}

fn systems() -> Vec<System> {
    vec![
        System { name: "n-hexane",     path: "testdata/molecules/scaling/n-hexane.xyz",     frozen_core: 6  },
        System { name: "naphthalene",  path: "testdata/molecules/scaling/naphthalene.xyz",  frozen_core: 10 },
        System { name: "n-decane",     path: "testdata/molecules/scaling/n-decane.xyz",     frozen_core: 10 },
        System { name: "adamantane",   path: "testdata/molecules/scaling/adamantane.xyz",   frozen_core: 10 },
        System { name: "caffeine",     path: "testdata/molecules/scaling/caffeine.xyz",     frozen_core: 14 },
        System { name: "n-hexadecane", path: "testdata/molecules/scaling/n-hexadecane.xyz", frozen_core: 16 },
        System { name: "n-icosane",    path: "testdata/molecules/scaling/n-icosane.xyz",    frozen_core: 20 },
    ]
}

fn main() {
    let ctx = ParallelContext::default();

    // Optional CSV output to file via FERRIC_SCALING_CSV env var; always
    // duplicated to stdout so the suite stays greppable from logs.
    let csv_path = std::env::var("FERRIC_SCALING_CSV").ok();
    let append = std::env::var("FERRIC_SCALING_APPEND").ok().as_deref() == Some("1");
    let mut csv_file: Option<std::fs::File> = csv_path.as_ref().map(|p| {
        std::fs::OpenOptions::new()
            .create(true).write(true).append(append).truncate(!append)
            .open(p)
            .expect("open CSV file")
    });
    let mut emit_csv = |line: String| {
        use std::io::Write;
        println!("{line}");
        if let Some(f) = csv_file.as_mut() {
            writeln!(f, "{line}").expect("write csv");
            let _ = f.flush();
        }
        let _ = std::io::stdout().flush();
    };

    // Optional system filter: comma-separated list of names in FERRIC_SCALING_ONLY.
    let only: Option<std::collections::HashSet<String>> = std::env::var("FERRIC_SCALING_ONLY")
        .ok()
        .map(|s| s.split(',').map(|t| t.trim().to_string()).collect());
    // Skip header emission if user is appending across runs (FERRIC_SCALING_NOHEADER=1).
    if std::env::var("FERRIC_SCALING_NOHEADER").ok().as_deref() != Some("1") {
        emit_csv(
            "system,n_atoms,n_ao,n_aux,config,thresh,retained_pairs,total_pairs,retained_frac,e_rpa,t_rhf_s,t_intermediates_s,t_eigensolver_s,t_pdep_total_s,t_total_s,status".to_string()
        );
    }

    // Optional sparsity filter (csv of labels), e.g. "Dense,Boys-1e-4".
    let cfg_filter: Option<std::collections::HashSet<String>> = std::env::var("FERRIC_SCALING_CFGS")
        .ok()
        .map(|s| s.split(',').map(|t| t.trim().to_string()).collect());

    for sys in systems() {
        if let Some(set) = only.as_ref() {
            if !set.contains(sys.name) { continue; }
        }
        eprintln!("\n=== {} ===", sys.name);
        let mol = match Molecule::load_xyz(sys.path) {
            Ok(m) => m,
            Err(e) => { eprintln!("  load failed: {e}"); continue; }
        };
        let n_atoms = mol.atoms.len();

        let obs_set = match basis::bundled("cc-pvdz") {
            Ok(b) => b, Err(e) => { eprintln!("  basis: {e}"); continue; }
        };
        let dfbs_set = match basis::bundled("cc-pvdz-ri") {
            Ok(b) => b, Err(e) => { eprintln!("  aux basis: {e}"); continue; }
        };
        let obs = match PreparedBasis::new(&mol, &obs_set) {
            Ok(b) => b, Err(e) => { eprintln!("  obs prep: {e}"); continue; }
        };
        let dfbs = match PreparedBasis::new(&mol, &dfbs_set) {
            Ok(b) => b, Err(e) => { eprintln!("  dfbs prep: {e}"); continue; }
        };
        let n_ao = obs.nbasis();
        let n_aux = dfbs.nbasis();
        eprintln!("  n_atoms={n_atoms}  n_AO={n_ao}  n_aux={n_aux}");

        // -------- RHF (one solve reused across configs) --------
        let op = Operator::coulomb();
        let bounds = match SchwarzBounds::compute(op, &obs) {
            Ok(b) => b, Err(e) => { eprintln!("  Schwarz: {e}"); continue; }
        };
        // Use DF-J for Coulomb (RPA's cc-pvdz-ri aux is reused) and LinkK for
        // exchange — direct K with linear-scaling Schwarz screening. DF-K is
        // skipped because cc-pvdz-ri is an MP2-fit basis, not a JK-fit basis
        // (would introduce mHa-scale error in K). With DF-J + LinkK, RHF cost
        // on drug-sized systems drops from hours to minutes.
        let rhf_cfg = RhfConfig {
            max_iter: 200,
            energy_conv: 1e-7,
            density_conv: 1e-6,
            df_j_aux: Some("cc-pvdz-ri".to_string()),
            k_builder: Some("link".to_string()),
            ..Default::default()
        };

        let t0 = Instant::now();
        let rhf = match solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_cfg) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  RHF did not converge: {e} — skipping system");
                emit_csv(format!(
                    "{name},{n_atoms},{n_ao},{n_aux},RHF,,,,,,,,,,FAIL",
                    name = sys.name
                ));
                continue;
            }
        };
        let t_rhf = t0.elapsed().as_secs_f64();
        eprintln!("  RHF E={:.8}  t={:.2}s", rhf.energy, t_rhf);

        // -------- Sparsity sweep --------
        for cfg in cfgs() {
            if let Some(set) = cfg_filter.as_ref() {
                if !set.contains(cfg.label) { continue; }
            }
            let pdep_cfg = PdepRpaConfig {
                frozen_core: sys.frozen_core,
                trunc_thresh: 1e-4,
                eigensolver_conv_thresh: 1e-6,
                chi0_sparsity: cfg.sparsity,
                ..Default::default()
            };

            // Quickly probe retained pairs (cheap relative to RPA itself).
            let (retained, total, thresh_val) = match cfg.sparsity {
                // Auto picks Dense/Boys by atom count at runtime; for this diagnostic
                // we report it like Dense (no static screening stats to show).
                Chi0Sparsity::Dense | Chi0Sparsity::Auto { .. } => (0usize, 0usize, 0.0_f64),
                Chi0Sparsity::BoysScreened { thresh } => {
                    match screen::build_screened_bov_boys(&mol, &obs, &dfbs, op, &rhf, sys.frozen_core, thresh) {
                        Ok((sb, _)) => (sb.total_retained, sb.n_occ_loc * sb.naux, thresh),
                        Err(e) => { eprintln!("  screen failed ({}): {e}", cfg.label); (0, 0, thresh) }
                    }
                }
            };

            eprintln!("  -> {} ...", cfg.label);
            let t1 = Instant::now();
            let r = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg);
            let t_pdep = t1.elapsed().as_secs_f64();
            let t_total = t_rhf + t_pdep;

            match r {
                Ok(res) => {
                    let frac = if total > 0 { retained as f64 / total as f64 } else { 1.0 };
                    eprintln!("     E_RPA={:.8}  retained={}/{} ({:.3})  t_pdep={:.2}s",
                        res.e_rpa, retained, total, frac, t_pdep);
                    // Per-stage breakdown is not exposed by run_pdep_rpa, so we
                    // report the whole t_pdep under t_intermediates and leave
                    // t_eigensolver=0 as a placeholder. The total is the
                    // wall-clock signal that matters for the scaling fit.
                    emit_csv(format!(
                        "{name},{n_atoms},{n_ao},{n_aux},{label},{thr:.0e},{ret},{tot},{frac:.5},{e:.10},{trhf:.3},{tpdep:.3},0.000,{tpdep:.3},{ttot:.3},OK",
                        name = sys.name,
                        label = cfg.label,
                        thr = thresh_val,
                        ret = retained,
                        tot = total,
                        e = res.e_rpa,
                        trhf = t_rhf,
                        tpdep = t_pdep,
                        ttot = t_total,
                    ));
                }
                Err(e) => {
                    eprintln!("     FAILED: {e}");
                    emit_csv(format!(
                        "{name},{n_atoms},{n_ao},{n_aux},{label},{thr:.0e},{ret},{tot},,,{trhf:.3},,,,{ttot:.3},FAIL",
                        name = sys.name,
                        label = cfg.label,
                        thr = thresh_val,
                        ret = retained,
                        tot = total,
                        trhf = t_rhf,
                        ttot = t_rhf + t_pdep,
                    ));
                }
            }
        }
    }
}
