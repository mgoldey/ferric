//! Riccati-direct / PDEP shared-zero anchors + PDEP log-det end-to-end timing.
//!
//! One RHF per molecule, then every method runs on the SAME reference so the
//! comparisons are free of SCF noise. Two modes:
//!
//!   anchor  — PDEP(τ=0, GL n_quad 32 and 64, eigen path) and PDEP(τ=0, 32,
//!             log-det path); Riccati-direct with TRIVIAL maps at ε=0
//!             (compute_reference=true → canonical plasmon zero as well);
//!             Riccati global-B at ε=0. Prints every energy and every pairwise
//!             difference. Sizes ≤C8 ONLY: the ε=0 Riccati anchor is a
//!             MEMORY ceiling, not a convenience — C12 thrashed for 3 h and
//!             was OOM-killed under a 10G cgroup cap (C8 ran in 43 s). See
//!             wiki/amplitude-threshold-drpa.md, "2026-09-06 — shared-zero
//!             anchors on the DIRECT path", §4.
//!             WARNING: do not run anchor mode above C8.
//!   pdep    — PDEP only: τ ∈ {0, 1e-2} × n_quad ∈ {32, 64} × {eigen, log-det}
//!             with wall per call and FERRIC_TIMING stage lines on stderr.
//!             Sizes where no plasmon reference can be built (C20–C48).
//!             C48 may be refused outright: as of 2026-09-06 the PDEP memory
//!             preflight charges full-width per-worker scratch (~31 GB at 12
//!             rayon workers) whatever the budget (same note, Phase 2 §5).
//!
//! Frozen core = number of carbon atoms (Z == 6), so it is correct only
//! for hydrocarbons; other molecules get a wrong frozen-core count.
//!
//! Run (from the repo root, single BLAS thread, under scripts/ferric-limited):
//!   OPENBLAS_NUM_THREADS=1 FERRIC_TIMING=1 cargo run --release \
//!     -p ferric-benchmarks --example drpa_pdep_anchor -- \
//!     testdata/molecules/alkane_8.xyz anchor
//!
//! Env knobs: FERRIC_ANCHOR_BUDGET_GB (PDEP/eri3 budget, default 6),
//! FERRIC_ANCHOR_NQUAD (comma list, default "32,64"),
//! FERRIC_ANCHOR_TAUS (comma list, default "0,1e-2" in pdep mode; anchor mode
//! always uses τ=0), FERRIC_ANCHOR_SKIP_GLOBAL=1 skips the global-B Riccati.

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::drpa_amplitude::{amplitude_drpa, amplitude_drpa_direct, AmplitudeDrpaConfig};
use ferric_mp2::lmp2_direct::DirectConfig;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_rpa::run_pdep_rpa;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_list(name: &str, default: &str) -> Vec<f64> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("bad float in env list"))
        .collect()
}

fn pdep_cfg(fc: usize, tau: f64, n_quad: usize, need_eig: bool, budget: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: fc,
        trunc_thresh: tau,
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: 1e-9,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: n_quad,
            u0: 0.5,
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: false,
        eigensolver: Eigensolver::default(),
        chi0_backend: Chi0Backend::default(),
        chi0_sparsity: Chi0Sparsity::default(),
        memory_budget_bytes: Some(budget),
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: need_eig,
        verbose: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_pdep(
    label: &str,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &PdepRpaConfig,
) -> f64 {
    let t = Instant::now();
    let r = run_pdep_rpa(mol, obs, dfbs, op, rhf, cfg).expect("pdep");
    let w = t.elapsed().as_secs_f64();
    println!(
        "PDEP {label:<28} e_rpa={:.12}  modes={}  conv={}  wall={w:.3}s",
        r.e_rpa, r.n_eigenpotentials, r.eigensolver_converged
    );
    r.e_rpa
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: drpa_pdep_anchor <xyz> <anchor|pdep>");
        std::process::exit(2);
    }
    let xyz = &args[1];
    let mode = args[2].as_str();
    let budget = (env_f64("FERRIC_ANCHOR_BUDGET_GB", 6.0) * (1u64 << 30) as f64) as usize;
    let nquads: Vec<usize> = env_list("FERRIC_ANCHOR_NQUAD", "32,64")
        .iter()
        .map(|x| *x as usize)
        .collect();

    let mol = Molecule::load_xyz(xyz).expect("xyz");
    let fc = mol.atoms.iter().filter(|a| a.z == 6).count();
    let obs_bs = basis::bundled("6-31g").expect("6-31g");
    let dfbs_bs = basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("prep");
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).expect("dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();

    let t = Instant::now();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("rhf");
    assert!(rhf.converged, "RHF did not converge");
    let nocc = (mol.nelec() as usize) / 2;
    println!(
        "SYSTEM {xyz} natoms={} nbf={} naux={} nocc={} fc={} e_rhf={:.10} rhf_wall={:.2}s",
        mol.atoms.len(),
        obs.nbasis(),
        dfbs.nbasis(),
        nocc,
        fc,
        rhf.energy,
        t.elapsed().as_secs_f64()
    );

    match mode {
        "anchor" => {
            let mut e_pdep = Vec::new();
            for &nq in &nquads {
                let e = run_pdep(
                    &format!("tau=0 nq={nq} eigen"),
                    &mol,
                    &obs,
                    &dfbs,
                    op,
                    &rhf,
                    &pdep_cfg(fc, 0.0, nq, true, budget),
                );
                e_pdep.push((nq, e));
            }
            let nq0 = nquads[0];
            let e_logdet = run_pdep(
                &format!("tau=0 nq={nq0} logdet"),
                &mol,
                &obs,
                &dfbs,
                op,
                &rhf,
                &pdep_cfg(fc, 0.0, nq0, false, budget),
            );
            println!(
                "DIFF pdep eigen-vs-logdet (nq={nq0}) = {:.3e}",
                e_pdep[0].1 - e_logdet
            );
            if e_pdep.len() > 1 {
                println!(
                    "DIFF pdep nq={} - nq={} = {:.3e}",
                    e_pdep[0].0,
                    e_pdep[1].0,
                    e_pdep[0].1 - e_pdep[1].1
                );
            }

            // Riccati-direct, TRIVIAL maps, eps = 0, with the canonical plasmon zero.
            let cfg = AmplitudeDrpaConfig {
                eps: 0.0,
                frozen_core: fc,
                eri3_budget_bytes: Some(budget),
                compute_reference: true,
                diis: Some(6),
                ..Default::default()
            };
            let dcfg = DirectConfig {
                aux_radius_bohr: 1e6,
                virt_radius_bohr: None,
                ao_tail: 0.0,
                schwarz_skip: 0.0,
                batch_merge: 1,
                ..Default::default()
            };
            let t = Instant::now();
            let (rd, st) = amplitude_drpa_direct(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, &dcfg)
                .expect("drpa direct");
            println!(
                "RICCATI direct(trivial,eps=0)   e_corr={:.12}  plasmon={:.12}  iters={}  \
                 relres={:.2e}  keep={:.4}  strips_max={}x{}  wall={:.3}s \
                 (maps {:.2} eri3 {:.2} metric {:.2} pairs {:.2})",
                rd.e_corr,
                rd.e_corr_plasmon_canonical,
                rd.iterations,
                rd.relres,
                rd.keep_fraction,
                st.strip_rows_max,
                st.strip_cols_max,
                t.elapsed().as_secs_f64(),
                st.t_maps_s,
                st.t_eri3_s,
                st.t_metric_s,
                st.t_pairs_s
            );
            println!(
                "DIFF riccati_direct - plasmon = {:.3e}",
                rd.e_corr - rd.e_corr_plasmon_canonical
            );
            for (nq, e) in &e_pdep {
                println!(
                    "DIFF pdep(nq={nq}) - plasmon = {:.3e}    \
                     pdep(nq={nq}) - riccati_direct = {:.3e}",
                    e - rd.e_corr_plasmon_canonical,
                    e - rd.e_corr
                );
            }

            if std::env::var("FERRIC_ANCHOR_SKIP_GLOBAL").is_err() {
                let cfg_g = AmplitudeDrpaConfig {
                    compute_reference: false,
                    ..cfg.clone()
                };
                let t = Instant::now();
                let rg = amplitude_drpa(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg_g)
                    .expect("drpa global");
                println!(
                    "RICCATI global(eps=0)           e_corr={:.12}  iters={}  relres={:.2e}  wall={:.3}s",
                    rg.e_corr, rg.iterations, rg.relres, t.elapsed().as_secs_f64()
                );
                println!(
                    "DIFF riccati_direct - riccati_global = {:.3e}",
                    rd.e_corr - rg.e_corr
                );
                println!(
                    "DIFF riccati_global - plasmon = {:.3e}",
                    rg.e_corr - rd.e_corr_plasmon_canonical
                );
            }
        }
        "pdep" => {
            let taus = env_list("FERRIC_ANCHOR_TAUS", "0,1e-2");
            let mut rows: Vec<(f64, usize, bool, f64)> = Vec::new();
            for &tau in &taus {
                for &nq in &nquads {
                    for need_eig in [true, false] {
                        let lab = format!(
                            "tau={tau:e} nq={nq} {}",
                            if need_eig { "eigen" } else { "logdet" }
                        );
                        let e = run_pdep(
                            &lab,
                            &mol,
                            &obs,
                            &dfbs,
                            op,
                            &rhf,
                            &pdep_cfg(fc, tau, nq, need_eig, budget),
                        );
                        rows.push((tau, nq, need_eig, e));
                    }
                }
            }
            for &tau in &taus {
                let get = |nq: usize, eig: bool| {
                    rows.iter()
                        .find(|r| r.0 == tau && r.1 == nq && r.2 == eig)
                        .map(|r| r.3)
                };
                for &nq in &nquads {
                    if let (Some(a), Some(b)) = (get(nq, true), get(nq, false)) {
                        println!("DIFF tau={tau:e} nq={nq} eigen-logdet = {:.3e}", a - b);
                    }
                }
                if nquads.len() > 1 {
                    if let (Some(a), Some(b)) = (get(nquads[0], true), get(nquads[1], true)) {
                        println!(
                            "DIFF tau={tau:e} nq={} - nq={} = {:.3e}",
                            nquads[0],
                            nquads[1],
                            a - b
                        );
                    }
                }
            }
            if taus.len() > 1 {
                for &nq in &nquads {
                    let a = rows
                        .iter()
                        .find(|r| r.0 == taus[0] && r.1 == nq && r.2)
                        .map(|r| r.3);
                    let b = rows
                        .iter()
                        .find(|r| r.0 == taus[1] && r.1 == nq && r.2)
                        .map(|r| r.3);
                    if let (Some(a), Some(b)) = (a, b) {
                        println!(
                            "DIFF nq={nq} tau={:e} - tau={:e} (truncation error) = {:.3e}",
                            taus[1],
                            taus[0],
                            b - a
                        );
                    }
                }
            }
        }
        _ => {
            eprintln!("mode must be anchor|pdep");
            std::process::exit(2);
        }
    }
}
