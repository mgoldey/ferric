//! Does raising BLAS threads for the CCSD einsum GEMMs change the answer?
//!
//! ferric-cc contains NO rayon (only `ccsd_t.rs` does), so its `einsum!` GEMMs
//! run outside any rayon region and the `blas_threads` call-path proof for a
//! raise holds. The open question is not safety but REPRODUCIBILITY: threaded
//! OpenBLAS splits a GEMM's k-axis across threads, which reorders the
//! floating-point accumulation.
//!
//! The CLI only prints 10 digits, which is not enough to answer that. This
//! prints the full f64 bit pattern of the correlation energy so a last-bit
//! difference is visible.
//!
//!   for bt in 1 2 4 8 12; do FERRIC_BLAS_THREADS=$bt OPENBLAS_NUM_THREADS=1 \
//!     cargo run --release -p ferric-benchmarks --example ccsd_blas_threads; done
use std::time::Instant;

use ferric_cc::ccsd::ccsd;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let mol_name = std::env::var("FERRIC_MOL").unwrap_or_else(|_| "water".into());
    let obs_name = std::env::var("FERRIC_OBS").unwrap_or_else(|_| "aug-cc-pvdz".into());
    let aux_name = std::env::var("FERRIC_AUX").unwrap_or_else(|_| "aug-cc-pvdz-rifit".into());

    let mol = Molecule::load_xyz(&format!("testdata/molecules/{mol_name}.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(&obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(&aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { max_iter: 100, ..Default::default() },
    )
    .unwrap();
    assert!(rhf.converged, "RHF must converge");

    let t = Instant::now();
    let cfg = ferric_cc::CcConfig::default();
    let res = ccsd(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let wall = t.elapsed().as_secs_f64();

    let e = res.correlation_energy;
    println!(
        "FERRIC_BLAS_THREADS={:<5} wall={:7.2}s  E_corr={:.17e}  bits=0x{:016x}",
        std::env::var("FERRIC_BLAS_THREADS").unwrap_or_else(|_| "unset".into()),
        wall,
        e,
        e.to_bits()
    );
}
