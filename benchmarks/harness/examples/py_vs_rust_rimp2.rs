//! Rust-side timing partner for scripts/py_rimp2_bench.py: solves RHF once
//! (DF-JK, not timed), then times ONLY the ri_mp2 call, repeated 3x, so the
//! Python-vs-Rust comparison never relies on subtracting a noisy ~100 s SCF.
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-benchmarks \
//!     --example py_vs_rust_rimp2 -- <xyz> <basis> <auxbasis>
use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (xyz, obs_name, aux_name) = (&args[1], &args[2], &args[3]);

    let mol = Molecule::load_xyz(xyz).unwrap();
    let obs_set = basis::bundled(obs_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    // DF-JK just to make the (untimed) reference SCF fast; the ~1e-5 DF orbital
    // noise is irrelevant to the MP2-stage wall-clock being measured.
    let rhf_cfg = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let t_rhf = Instant::now();
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &rhf_cfg).unwrap();
    assert!(rhf.converged);
    eprintln!("RHF {:.10} Ha ({:.2}s, untimed reference)", rhf.energy, t_rhf.elapsed().as_secs_f64());

    let dfbs_set = basis::bundled(aux_name).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
    let mp2_cfg = RiMp2Config::default();

    for rep in 0..3 {
        let t = Instant::now();
        let result = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg).unwrap();
        println!(
            "rep {} ri_mp2 wall {:.3}s  corr {:.10} Ha",
            rep,
            t.elapsed().as_secs_f64(),
            result.mp2_corr
        );
    }
}
