//! Head-to-head wall-time: Lanczos vs Davidson eigensolver in run_pdep_rpa.
//!
//! Measures both trunc_thresh = 0 (both solvers use the eye seed) and
//! trunc_thresh > 0 (Davidson uses the atom seed; Lanczos uses eye after the
//! FD-stability fix). Confirms energies agree, reports the speed ratio.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo test -p ferric-rpa --release --test lanczos_vs_davidson_bench -- --ignored --nocapture

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::Eigensolver;
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn time_solver(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ferric_scf::ScfResult,
    es: Eigensolver,
    trunc: f64,
    reps: usize,
) -> (f64, f64, usize) {
    let cfg = PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: trunc,
        eigensolver: es,
        ..Default::default()
    };
    let r0 = run_pdep_rpa(mol, obs, dfbs, op, rhf, &cfg).unwrap();
    let t = Instant::now();
    for _ in 0..reps {
        let _ = run_pdep_rpa(mol, obs, dfbs, op, rhf, &cfg).unwrap();
    }
    let dt = t.elapsed().as_secs_f64() / reps as f64;
    (dt, r0.e_rpa, r0.n_eigenpotentials)
}

fn run_case(name: &str, xyz: &str, obs_name: &str, dfbs_name: &str, reps: usize) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(dfbs_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let naux = dfbs.nbasis();

    println!("\n=== {name}  (obs={obs_name}, aux={dfbs_name}, naux={naux}) ===");
    for &trunc in &[0.0_f64, 1e-4] {
        let (td, ed, md) = time_solver(&mol, &obs, &dfbs, op, &rhf, Eigensolver::Davidson, trunc, reps);
        let (tl, el, ml) = time_solver(&mol, &obs, &dfbs, op, &rhf, Eigensolver::Lanczos, trunc, reps);
        let de = (ed - el).abs();
        println!(
            "  trunc={trunc:.0e}: Davidson {:7.1}ms (M={md}, E={ed:.8})  |  Lanczos {:7.1}ms (M={ml}, E={el:.8})  |  Lanczos/Davidson={:.2}x  ΔE={de:.1e}",
            td * 1e3,
            tl * 1e3,
            tl / td,
        );
    }
}

#[test]
#[ignore = "benchmark: run with --release --ignored --nocapture"]
fn lanczos_vs_davidson() {
    println!("\n=== Lanczos vs Davidson wall-time in run_pdep_rpa ===");
    run_case(
        "water",
        "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        "cc-pvdz",
        "cc-pvdz-ri",
        3,
    );
    run_case(
        "methane / aug-cc-pVTZ",
        "5\n\nC 0.0 0.0 0.0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\n",
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
        2,
    );
}
