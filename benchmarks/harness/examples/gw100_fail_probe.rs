//! Diagnose why specific GW100 molecules return None (FAILED) from gw100_full.
//!
//! Runs the G0W0-only path stage by stage, printing the Err at each `.ok()?`
//! site so we can see WHICH stage fails (parse / basis / prepared / schwarz /
//! neutral RHF / GW), instead of the opaque "FAILED" summary.
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=12 cargo run --release \
//!     --example gw100_fail_probe -p ferric-benchmarks aug-cc-pvdz

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::io::Write;

const HA_TO_EV: f64 = 27.211386245988_f64;

/// say! + immediate flush so a file-redirected run shows live progress.
macro_rules! say {
    ($($a:tt)*) => {{ println!($($a)*); let _ = std::io::stdout().flush(); }};
}

struct Case {
    name: &'static str,
    xyz: &'static str,
    ip_ref: f64,
}

fn cases() -> Vec<Case> {
    vec![
Case { name: "CCuN", xyz: "3
mol
C 0.0000 0.0000 0.0000
N 0.0000 0.0000 1.158
Cu 0.0000 0.0000 -1.832
", ip_ref: f64::NAN },
Case { name: "Cu2", xyz: "2
mol
Cu 0.0 0.0 0.0
Cu 0.0 0.0 2.2197
", ip_ref: 7.46 },
    ]
}

fn probe(case: &Case, obs_name: &str, dfbs_name: &str) {
    say!("\n========== {} (ref IP {:.2} eV) ==========", case.name, case.ip_ref);
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();

    let neutral = match Molecule::parse_xyz(case.xyz, 0, 1) {
        Ok(m) => m,
        Err(e) => { say!("  parse(neutral) ERR: {e:?}"); return; }
    };
    say!("  parsed: {} atoms, nelec={}, NRE={:.6}",
        neutral.atoms.len(), neutral.nelec(), neutral.nuclear_repulsion());
    for a in &neutral.atoms {
        say!("    Z={:<3} {:>10.4} {:>10.4} {:>10.4}", a.z, a.x, a.y, a.zpos);
    }

    let obs_bs = match basis::bundled(obs_name) {
        Ok(b) => b, Err(e) => { say!("  basis({obs_name}) ERR: {e:?}"); return; }
    };
    let dfbs_bs = match basis::bundled(dfbs_name) {
        Ok(b) => b, Err(e) => { say!("  basis({dfbs_name}) ERR: {e:?}"); return; }
    };
    let obs_n = match PreparedBasis::new(&neutral, &obs_bs) {
        Ok(p) => p, Err(e) => { say!("  PreparedBasis(obs) ERR: {e:?}"); return; }
    };
    let dfbs_n = match PreparedBasis::new(&neutral, &dfbs_bs) {
        Ok(p) => p, Err(e) => { say!("  PreparedBasis(dfbs) ERR: {e:?}"); return; }
    };
    let bounds_n = match SchwarzBounds::compute(op, &obs_n) {
        Ok(b) => b, Err(e) => { say!("  SchwarzBounds ERR: {e:?}"); return; }
    };

    // Neutral RHF — the prime suspect. Solve once; report; then escalate iters
    // / level-shift if it failed, to learn whether it's oscillation vs. a wall.
    let cfg = RhfConfig::default();
    say!("  RHF(neutral, default {} iter)...", cfg.max_iter);
    let rhf_n = match solve_rhf(&ctx, &neutral, &obs_n, op, &bounds_n, &cfg) {
        Ok(r) => {
            say!("  RHF OK conv={} iters={} E={:.8} HOMO_eps={:.6}",
                r.converged, r.iterations, r.energy,
                r.eps_r()[(neutral.nelec() as usize)/2 - 1]);
            r
        }
        Err(e) => {
            say!("  RHF ERR: {e:?}");
            // Driver uses level_shift=0.5; replicate it, then escalate to 1.0.
            let cfg2 = RhfConfig { max_iter: 500, level_shift: 0.5, ..Default::default() };
            say!("  retry RHF(500 iter, level_shift=0.5)...");
            match solve_rhf(&ctx, &neutral, &obs_n, op, &bounds_n, &cfg2) {
                Ok(r) if r.converged => { say!("  retry OK conv={} iters={} E={:.8}", r.converged, r.iterations, r.energy); r }
                _ => {
                    let cfg3 = RhfConfig { max_iter: 500, level_shift: 1.0, ..Default::default() };
                    say!("  retry2 RHF(500 iter, level_shift=1.0)...");
                    match solve_rhf(&ctx, &neutral, &obs_n, op, &bounds_n, &cfg3) {
                        Ok(r) => { say!("  retry2 conv={} iters={} E={:.8}", r.converged, r.iterations, r.energy); r }
                        Err(e) => { say!("  retry2 STILL ERR: {e:?} -> RHF is the failure point"); return; }
                    }
                }
            }
        }
    };
    if !rhf_n.converged {
        say!("  -> RHF returned non-converged; skipping GW");
        return;
    }
    let homo_abs = (neutral.nelec() as usize) / 2 - 1;
    let pdep_cfg = PdepRpaConfig {
        quadrature: QuadratureConfig { scheme: QuadratureScheme::GaussLegendre, n_points: 16, u0: 0.5 },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 1e-4,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false,
    };
    let gcfg = GwConfig { method: GwMethod::G0W0, max_ev_iter: 8, ev_conv_thresh: 1e-4, ..Default::default() };
    say!("  G0W0@HF...");
    match run_gw(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &pdep_cfg, &gcfg, None) {
        Ok(res) => match res.mo_indices.iter().position(|&i| i == homo_abs) {
            Some(local) => say!("OK IP={:.3} eV", -res.eps_qp[local] * HA_TO_EV),
            None => say!("OK but HOMO not in mo_indices {:?}", res.mo_indices),
        },
        Err(e) => say!("ERR: {e:?}"),
    }
}

fn main() {
    let obs_name = std::env::args().nth(1).unwrap_or_else(|| "aug-cc-pvdz".to_string());
    let dfbs_name = format!("{obs_name}-rifit");
    for c in cases() {
        probe(&c, &obs_name, &dfbs_name);
    }
}
