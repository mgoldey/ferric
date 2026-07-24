//! TEMPORARY benchmark: incremental vs full-rebuild DirectJK Fock on the direct
//! (non-DF) RHF path. Not committed. Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-harness --example incr_fock_bench
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn run(label: &str, path: &str, bs_name: &str) {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bs = basis::bundled(bs_name).unwrap();
    let mol = Molecule::load_xyz(path).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig::default();

    // Full-rebuild-every-iteration.
    std::env::set_var("FERRIC_SCF_INCREMENTAL", "0");
    let t = Instant::now();
    let full = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let t_full = t.elapsed().as_secs_f64();
    std::env::remove_var("FERRIC_SCF_INCREMENTAL");

    // Incremental.
    let t = Instant::now();
    let incr = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let t_incr = t.elapsed().as_secs_f64();

    println!(
        "{label:14} nbf={:3}  FULL: {:.3}s ({} it, {} quartets, E={:.9})  \
         INCR: {:.3}s ({} it, {} quartets, E={:.9})  \
         speedup={:.2}x  quartet-ratio={:.2}  dE={:.2e}",
        prep.nbasis(),
        t_full, full.iterations, full.computed_quartets, full.energy,
        t_incr, incr.iterations, incr.computed_quartets, incr.energy,
        t_full / t_incr,
        incr.computed_quartets as f64 / full.computed_quartets.max(1) as f64,
        (incr.energy - full.energy).abs(),
    );
}

fn main() {
    // Extended linear chains: the quartet-ratio (incr/full) is the key,
    // basis-robust metric for whether ΔD-weighted screening prunes at scale.
    run("C8/631g", "testdata/molecules/alkane_8.xyz", "6-31g");
    run("C12/sto3g", "testdata/molecules/alkane_12.xyz", "sto-3g");
    run("C20/sto3g", "testdata/molecules/alkane_20.xyz", "sto-3g");
}
