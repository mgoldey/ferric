//! Regression: raising `FERRIC_LANCZOS_BLAS_THREADS` must not abort.
//!
//! `78bc70b` (2026-07-07) defaulted the Lanczos BLAS raise to 1 because
//! `available_parallelism()` as the DEFAULT made the aug-cc-pV{D,T}Z PDEP-RPA
//! tests abort with `stack overflow` — the crash mode documented in
//! `lanczos.rs::lanczos_blas_threads`.
//!
//! That crash no longer reproduces (verified 2026-07-26: full ferric-rpa suite,
//! 8 concurrent CLI solves at 12 BLAS threads each, and a direct probe of
//! `eigh` at n=1200 on a 2 MB stack and inside a rayon worker — all clean).
//! The most likely reason is that the eigensolve moved onto
//! `ferric_core::linalg::eigh_dc`/`eigvalsh_dc` (`c284e48`), which use the
//! standard LAPACK workspace-query convention and HEAP-allocate the workspace,
//! whereas the previous path put it on the caller's stack.
//!
//! This test pins that: it runs the exact configuration `78bc70b` reported as
//! aborting. If a future change reintroduces a stack-allocated workspace, this
//! fails loudly instead of the knob silently becoming dangerous again.
//!
//! It does NOT change the default, which stays at 1 for reproducibility (a
//! threaded GEMM/eigh reorders reductions). It only asserts the knob is safe
//! to use, and that using it does not move the energy.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn h2o_atz_rpa(blas_threads: &str) -> f64 {
    std::env::set_var("FERRIC_LANCZOS_BLAS_THREADS", blas_threads);
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvtz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvtz-rifit").unwrap()).unwrap();
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
    assert!(rhf.converged, "H2O/aug-cc-pVTZ RHF must converge");
    let e = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &PdepRpaConfig::default()).unwrap().e_rpa;
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");
    e
}

/// aug-cc-pVTZ is the basis `78bc70b` names as aborting. Reaching the end of
/// this test at all is the assertion — a stack overflow aborts the process, it
/// does not return an `Err`.
#[test]
fn lanczos_blas_raise_does_not_abort_on_aug_cc_pvtz() {
    let e_serial = h2o_atz_rpa("1");
    let e_raised = h2o_atz_rpa("12");

    assert!(e_serial.is_finite() && e_serial < 0.0, "serial RPA energy is not sane: {e_serial}");
    assert!(e_raised.is_finite() && e_raised < 0.0, "raised RPA energy is not sane: {e_raised}");

    // Raising BLAS threads reorders reductions, so this is not expected to be
    // bit-identical — but it must agree far below anything physically
    // meaningful. Measured on benzene/aug-cc-pVTZ: identical to all 10 printed
    // digits at 1/4/6/12 threads.
    let diff = (e_serial - e_raised).abs();
    assert!(
        diff < 1e-9,
        "RPA correlation energy moved by {diff:.3e} Ha when BLAS threads were \
         raised (serial {e_serial:.12}, raised {e_raised:.12}) — a threaded \
         reduction-order change should be orders smaller than this",
    );
}
