//! RI-MP2 stage-split profiler: times each phase of `ri_mp2_spin_components`
//! separately (2c metric, V^{-1/2}, raw 3-index source build, streamed dressed
//! MO transform, energy accumulation) so the serial/parallel split is visible.
//!
//! Run at RAYON_NUM_THREADS=1 and =12 and compare per-stage: anything that does
//! not shrink is serial work, and serial work is what sets the Amdahl ceiling.
//!
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=12 \
//!     cargo run --release -p ferric-benchmarks --example rimp2_stage_split
use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex;
use ferric_mp2::rimp2::{
    active_occ, eri3_budget_bytes, metric_inverse_sqrt, spin_components_from_b_ov,
    stream_dressed_mo_band,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let mol_name = std::env::var("FERRIC_MOL").unwrap_or_else(|_| "benzene".into());
    let obs_name = std::env::var("FERRIC_OBS").unwrap_or_else(|_| "aug-cc-pvtz".into());
    let aux_name = std::env::var("FERRIC_AUX").unwrap_or_else(|_| "aug-cc-pvtz-rifit".into());
    let frozen: usize = std::env::var("FERRIC_FROZEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);

    let path = format!("testdata/molecules/{mol_name}.xyz");
    let mol = Molecule::load_xyz(&path).unwrap();
    let obs_set = basis::bundled(&obs_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    let rhf_cfg = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        max_iter: 100,
        ..Default::default()
    };

    eprintln!("[setup] {mol_name}/{obs_name}, aux={aux_name}, nbasis={}", obs.nbasis());
    let t = Instant::now();
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &rhf_cfg).unwrap();
    eprintln!(
        "[setup] RHF converged={} iters={} E={:.10} ({:.2}s, not counted below)",
        rhf.converged,
        rhf.iterations,
        rhf.energy,
        t.elapsed().as_secs_f64()
    );

    let dfbs_set = basis::bundled(&aux_name).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();

    // --- mirror ri_mp2_spin_components, stage by stage ---
    let nbas = obs.nbasis();
    let nocc_total = mol.nelec() as usize / 2;
    let nocc = active_occ(nocc_total, frozen).unwrap();
    let first_occ = frozen;
    let nvir = nbas - nocc_total;
    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();
    eprintln!(
        "[dims] nbasis={nbas} naux={} nocc={nocc} nvir={nvir} (frozen={frozen})",
        dfbs.nbasis()
    );

    let t_all = Instant::now();

    let t0 = Instant::now();
    let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
    let t_metric = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let v2c_inv_sqrt = metric_inverse_sqrt(&v2c, op).unwrap();
    let t_invsqrt = t0.elapsed().as_secs_f64();

    let budget = eri3_budget_bytes(None);
    let t0 = Instant::now();
    let mut src = ThreeIndexSource::build(op, &obs, &dfbs, budget).unwrap();
    let t_src = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let b_flat = stream_dressed_mo_band(&mut src, &v2c_inv_sqrt, &c_occ, &c_vir, None).unwrap();
    let t_stream = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let sc = spin_components_from_b_ov(&b_flat, eps, nocc, nvir, first_occ, nocc_total);
    let t_energy = t0.elapsed().as_secs_f64();

    let total = t_all.elapsed().as_secs_f64();

    println!("\nMP2 corr = {:.10} Ha  (OS {:.10}, SS {:.10})", sc.e_total, sc.e_os, sc.e_ss);
    println!("Total    = {:.10} Ha", rhf.energy + sc.e_total);
    println!("\n{:<34} {:>9} {:>7}", "stage", "sec", "%");
    let row = |n: &str, v: f64| println!("{:<34} {:>9.3} {:>6.1}%", n, v, 100.0 * v / total);
    row("2c metric (P|Q)", t_metric);
    row("metric V^{-1/2}", t_invsqrt);
    row("raw 3-index source build", t_src);
    row("streamed dressed MO transform", t_stream);
    row("energy accumulation", t_energy);
    let acct = t_metric + t_invsqrt + t_src + t_stream + t_energy;
    row("unattributed", total - acct);
    println!("{:<34} {:>9.3}", "TOTAL", total);
}
