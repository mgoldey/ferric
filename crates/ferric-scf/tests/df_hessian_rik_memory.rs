//! Memory pre-flight of the RI-K Hessian: the tensors it keeps resident
//! (`L^x/Z^x`, `∂_xΓ`, `V^x`, `Ĉ` copies, `K^x`, see `ferric_scf::hessian_df_k`)
//! are charged to the global pool before the first allocation, the same
//! estimate is the pre-SCF probe `Auto` consults, and the estimate grows like
//! N^4.
//!
//! Own test binary: the pool is process-global, so installing a small one must
//! not share a process with the numerical tests; one test function, so that
//! nothing runs concurrently with an installed pool.
//!
//! Hypotheses. Right: a pool one byte below the documented peak
//! ([`rik_peak_bytes`]) is refused with a typed error naming the RI-K Hessian
//! reservation, a pool at the peak plus the raw three-index tensor (the only
//! other reservation of the call) runs and returns the skeleton it returns with
//! no pool, the pre-SCF probe refuses under the small pool and accepts under
//! the larger one, `Auto` then falls back to finite differences while
//! `Analytic` stays an error. Artifact: the charge is too small (a pool at the
//! peak still fails) or too large (the pool just below it succeeds), or the
//! estimate stops scaling like N^4.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, FrequencyConfig, HessianMethod, HessianSource,
};
use ferric_scf::hessian::analytic_hessian_available;
use ferric_scf::hessian::rij_support::{rik_terms_with_aux, RikMutants};
use ferric_scf::hessian_df_k::rik_peak_bytes;
use ferric_scf::rhf::RhfConfig;
use ndarray::Array2;

const WATER: &str = "3\nH2O\nO 0.01 -0.02 0.12\nH 0.03 0.77 -0.46\nH -0.02 -0.74 -0.49\n";

/// Documented peaks (module doc of `hessian_df_k`), within 1 %.
fn assert_documented(label: &str, got: usize, documented: f64) {
    let rel = (got as f64 - documented).abs() / documented;
    eprintln!("{label}: {got} bytes, documented {documented:.4e}, relative difference {rel:.1e}");
    assert!(rel < 0.01, "{label}: {got} vs documented {documented:.4e}");
}

#[test]
fn the_charge_matches_the_documentation_scales_as_n4_and_gates_the_run_and_the_probe() {
    // --- the documented numbers and the scaling (no pool involved) ----------
    assert_documented(
        "water cc-pVDZ/jkfit",
        rik_peak_bytes(3, 24, 5, 113),
        2.7246e6,
    );
    assert_documented(
        "30 atoms, 300 bf, m = 60, A = 1500",
        rik_peak_bytes(30, 300, 60, 1500),
        9.8244e9,
    );
    // Per-atom shape of the 30-atom example: A = 50 N, n = 10 N, m = 2 N.
    let at = |n: usize| rik_peak_bytes(n, 10 * n, 2 * n, 50 * n) as f64;
    let slope = (at(60) / at(30)).log2();
    let slope_71 = (at(71) / at(30)).ln() / (71.0f64 / 30.0).ln();
    eprintln!("scaling exponent 30 -> 60 atoms {slope:.2}, 30 -> 71 atoms {slope_71:.2}");
    assert!((3.6..4.1).contains(&slope), "exponent {slope}");
    assert_documented("71 atoms, same per-atom shape", at(71) as usize, 2.7097e11);

    if ferric_integrals::engine::libint_max_deriv_order() < 1 {
        eprintln!("SKIP: libint2 without first derivatives");
        return;
    }
    // --- the reservation inside the call -------------------------------------
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let aux = || PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let n = prep.nbasis();
    let naux = aux().nbasis();
    let x = Array2::from_shape_fn((n, 2), |(i, c)| 0.3 + 0.1 * (i + c) as f64);
    let run = || {
        rik_terms_with_aux(
            3,
            &prep,
            aux(),
            Operator::coulomb(),
            usize::MAX,
            &x,
            &RikMutants::default(),
        )
    };
    clear_global();
    let reference = run().expect("no pool: inert");
    let need = rik_peak_bytes(3, n, 2, naux);
    let raw = 8 * naux * n * n;

    install_global(MemoryPool::with_capacity_bytes(need - 1));
    let err = run()
        .expect_err("one byte below the peak must refuse")
        .to_string();
    eprintln!("pool at peak - 1: {err}");
    assert!(err.contains("RI-K Hessian"), "{err}");

    install_global(MemoryPool::with_capacity_bytes(need + 2 * raw));
    let got = run().expect("the peak plus the raw tensor must fit");
    let d = got
        .skeleton
        .iter()
        .zip(reference.skeleton.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("peak {need} bytes (+ raw {raw}); skeleton difference with/without pool {d:.1e}");
    assert_eq!(d, 0.0);

    // --- the pre-SCF probe and the dispatch ----------------------------------
    // RI-K for the 6-31G water of this file: probe sizes are those of the
    // molecule and the named fitting basis.
    let prep_p = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let cfg = RhfConfig {
        df_j_aux: Some(String::new()),
        df_k_aux: Some("cc-pvdz-ri".into()),
        ..Default::default()
    };
    let op = Operator::coulomb();
    let probe_need = rik_peak_bytes(3, prep_p.nbasis(), 5, naux);
    install_global(MemoryPool::with_capacity_bytes(probe_need - 1));
    let e = analytic_hessian_available(&mol, &prep_p, op, &cfg)
        .expect_err("probe must refuse below the estimate")
        .to_string();
    eprintln!("probe below: {e}");
    assert!(e.contains("memory pool"), "{e}");
    install_global(MemoryPool::with_capacity_bytes(probe_need + 2 * raw));
    if ferric_integrals::engine::libint_max_deriv_order() >= 2 {
        analytic_hessian_available(&mol, &prep_p, op, &cfg).expect("probe accepts above");
    }

    // Auto falls back to finite differences, Analytic stays an error: a pool
    // between the FD run's needs and the Hessian estimate.
    install_global(MemoryPool::with_capacity_bytes(probe_need - 1));
    let freq = |method| {
        harmonic_frequencies(
            &ParallelContext::default(),
            &mol,
            "6-31g",
            op,
            &cfg,
            &FrequencyConfig {
                hessian: method,
                ..Default::default()
            },
        )
    };
    let auto = freq(HessianMethod::Auto);
    let strict = freq(HessianMethod::Analytic);
    clear_global();
    let auto = auto.expect("Auto must fall back to finite differences");
    assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
    assert!(strict.is_err(), "Analytic must stay an error");
}
