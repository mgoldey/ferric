//! Raising `FERRIC_LANCZOS_BLAS_THREADS` must not change the RPA energy.
//!
//! # Why this test exists
//!
//! The eigensolve's BLAS thread count defaults to 1, and the *only* remaining
//! reason is bit-reproducibility — the stack-overflow hazard that originally
//! motivated it was re-verified as stale on 2026-07-26 and is pinned by
//! `lanczos_blas_raise_no_crash.rs`.
//!
//! "Bit-reproducibility only" is a much weaker justification than "correctness",
//! and it is worth holding the line on it explicitly: a threaded reduction
//! reorders sums, which costs bit-identity but must NOT cost accuracy. This test
//! pins the accuracy half of that claim, so that anyone weighing the ~1.9×
//! eigensolve speedup against losing bit-identity can see exactly what is and is
//! not at risk.
//!
//! # Deliberately not a benchmark
//!
//! Asserts only on ENERGIES. The machine this was written on was heavily
//! contested, and wall clocks from a loaded box are untrustworthy in both
//! directions — see `boys_screening_crossover.rs` for what that cost. The
//! speedup figures live in the perf memory, measured on a quiet box.

use ferric_core::{basis, mol::Molecule, parallel::ParallelContext};
use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
use ferric_rpa::config::PdepRpaConfig;
use ferric_rpa::run_pdep_rpa;
use ferric_scf::{
    rhf::{solve_rhf, RhfConfig},
    screening::SchwarzBounds,
};
use std::sync::Mutex;

/// Serializes the runs that mutate the BLAS-thread env var, so a parallel test
/// never observes another's setting.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn rpa_energy_at(threads: Option<&str>) -> f64 {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    match threads {
        Some(t) => std::env::set_var("FERRIC_LANCZOS_BLAS_THREADS", t),
        None => std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS"),
    }

    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = RhfConfig { density_conv: 1e-9, max_iter: 200, ..Default::default() };
    let rhf = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &scf).unwrap();
    assert!(rhf.converged, "SCF must converge");

    let cfg = PdepRpaConfig::default();
    let e = run_pdep_rpa(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &cfg).unwrap().e_rpa;

    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");
    e
}

/// Raising the eigensolve's BLAS threads must leave the correlation energy
/// unchanged to well below chemical relevance.
///
/// The tolerance is 1e-9 Ha rather than bit-equality precisely BECAUSE the
/// documented cost of raising this knob is a reordered reduction. Demanding
/// bit-equality here would assert the very thing the knob gives up; demanding
/// 1e-9 asserts the thing it must never give up.
#[test]
fn raising_eigensolve_blas_threads_preserves_the_energy() {
    let e_default = rpa_energy_at(None);
    eprintln!("FERRIC_LANCZOS_BLAS_THREADS unset : E_rpa = {e_default:.12}");

    for t in ["1", "2", "4"] {
        let e = rpa_energy_at(Some(t));
        let d = (e - e_default).abs();
        eprintln!("                            = {t:>2} : E_rpa = {e:.12}  |dE| = {d:.3e}");
        assert!(
            d < 1e-9,
            "raising the eigensolve BLAS threads to {t} changed E_rpa by {d:.3e} Ha \
             ({e:.12} vs {e_default:.12}); a reordered reduction may cost bit-identity \
             but must never cost accuracy"
        );
    }
}

/// `= 1` must be exactly the default, since the default IS 1.
///
/// Guards against the knob silently taking a different code path at its own
/// default value — which would mean the reproducibility argument for keeping the
/// default at 1 does not actually hold.
#[test]
fn explicit_one_matches_the_default_bit_for_bit() {
    let e_default = rpa_energy_at(None);
    let e_one = rpa_energy_at(Some("1"));
    eprintln!("default = {e_default:.12}   explicit 1 = {e_one:.12}");
    assert_eq!(
        e_default.to_bits(),
        e_one.to_bits(),
        "explicit FERRIC_LANCZOS_BLAS_THREADS=1 must be bit-identical to the default"
    );
}
