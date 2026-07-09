//! Stage-resolved timing for the GW100 per-molecule pipeline — to MEASURE
//! (not estimate) where wall-time goes and confirm whether the diagnosed
//! 4×-GW-column setup redundancy is actually the dominant cost.
//!
//! Mirrors gw100_full.rs's neutral-side GW pipeline for one molecule and prints
//! per-stage milliseconds, so we can answer: is it ERI3/metric (shareable setup)
//! or the Davidson eigensolve / quadrature (NOT shareable)?
//!
//! Run:
//!   cargo run --release --example gw_profile -p ferric-gw -- <file.xyz> <obs> <ri-aux>
//! e.g.
//!   ... -- scripts/gw100/geom/CO.xyz def2-tzvp def2-tzvp-rifit

use std::time::Instant;

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
use ferric_rpa::run_pdep_rpa;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

macro_rules! timed {
    ($label:expr, $body:expr) => {{
        let t = Instant::now();
        let r = $body;
        println!("{:<28} {:>9.1} ms", $label, t.elapsed().as_secs_f64() * 1e3);
        r
    }};
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        davidson_conv_thresh: 1e-7,
        davidson_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: gw_profile <file.xyz> <obs> <ri-aux>");
    let obs_name = args.next().unwrap_or_else(|| "def2-tzvp".to_string());
    let aux_name = args.next().unwrap_or_else(|| "def2-tzvp-rifit".to_string());

    let xyz = std::fs::read_to_string(&path).expect("read xyz");
    let mol = Molecule::parse_xyz(&xyz, 0, 1).expect("parse xyz");
    let obs_bs = basis::bundled(&obs_name).expect("obs");
    let aux_bs = basis::bundled(&aux_name).expect("aux");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();

    println!("# gw_profile: {} / {}  (nbf will print below)", path, obs_name);
    let obs = timed!("PreparedBasis(obs)", PreparedBasis::new(&mol, &obs_bs).expect("obs"));
    let dfbs = timed!("PreparedBasis(aux)", PreparedBasis::new(&mol, &aux_bs).expect("aux"));
    let bounds = timed!("SchwarzBounds", SchwarzBounds::compute(op, &obs).expect("schwarz"));
    let rhf = timed!("solve_rhf", solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("rhf"));

    // One standalone PDEP-RPA: this is what each of the 4 run_gw calls rebuilds.
    let _pdep = timed!("run_pdep_rpa (1x)", run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg()).expect("pdep"));

    // Each method's FULL run_gw (includes its own run_pdep_rpa + build_full_b).
    // Sum of G0W0+COHSEX+evGW0 ≈ the redundant-setup cost we'd save by sharing.
    let cfg = pdep_cfg();
    for method in [GwMethod::G0W0, GwMethod::Cohsex, GwMethod::EvGw0, GwMethod::EvGw] {
        let tag = match method {
            GwMethod::G0W0 => "run_gw[G0W0]",
            GwMethod::Cohsex => "run_gw[COHSEX]",
            GwMethod::EvGw0 => "run_gw[evGW0]",
            GwMethod::EvGw => "run_gw[evGW]",
            GwMethod::ScCohsex => "run_gw[scCOHSEX]",
        };
        let gcfg = GwConfig { method, max_ev_iter: 8, ev_conv_thresh: 1e-4, ..Default::default() };
        let _ = timed!(tag, run_gw(&mol, &obs, &dfbs, op, &rhf, &cfg, &gcfg, None).expect("gw"));
    }
    println!("# Reading: if run_gw[G0W0]≈run_gw[COHSEX]≈run_gw[evGW0] and each is");
    println!("# dominated by 'run_pdep_rpa (1x)'-scale setup, sharing W0 across the");
    println!("# 3 W0-methods saves ~2x that setup. If the Davidson/quadrature inside");
    println!("# dominates instead, the hoist saves little — measure, don't assume.");
}
