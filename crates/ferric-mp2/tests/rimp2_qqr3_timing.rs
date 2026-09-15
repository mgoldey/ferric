//! Does QQR-3 screening make the RI-MP2 3-index build FASTER, not just smaller?
//!
//! A triple-count reduction is not a speedup. The screened loop still walks
//! every `(P, s1, s2)` and evaluates a bound, and the skipped work is integral
//! evaluation only -- the MO transform and the energy contraction are unchanged
//! because the tensor keeps its dense shape (skipped blocks are zeros).
//!
//! So the expected ceiling is: (fraction of integral time) x (fraction of
//! triples dropped). If the 3-index build is not dominant, ~21% fewer triples
//! buys a lot less than 21% overall. That is what this measures.
//!
//! `#[ignore]`d: it is a timing harness, not a correctness gate, and timings on
//! a shared box are noisy. Run with `--ignored --nocapture`.
//!
//! # MEASURED 2026-09-14, decane / cc-pVDZ + cc-pVDZ-RI (nbf=250, naux=868)
//!
//! ```text
//!   operator      dense    screened(1e-10)   speedup   dE
//!   coulomb       1.29 s      1.59 s          0.81x    1.8e-13 Ha
//!   terfc(r0=2)  38.50 s     36.53 s          1.05x    6.8e-14 Ha
//! ```
//!
//! Reproduced twice, second time on a quieter box (load 1.7 vs 7.5).
//!
//! **Coulomb is a NET LOSS.** Isolating the pieces explains why: building the
//! QQR-3 bound costs 0.405 s against a 0.467 s dense integral build, and the
//! screened build then takes exactly as long as the dense one (ratio 1.000).
//! The bound is pure overhead at this size -- it costs nearly as much as the
//! work it is trying to avoid.
//!
//! **terfc gains only 1.05x end-to-end** (1.11x on the isolated 3-index build)
//! despite dropping ~20% of the integral cost. Checked, not assumed: the
//! dropped triples are not disproportionately cheap -- cost-weighted, screening
//! keeps 79.8% against 85.3% by triple count, so it drops slightly MORE
//! expensive work than average. The 3-index build is ~97% of terfc's runtime,
//! so a 20% cost cut predicts ~1.25x; the shortfall to 1.11x is bound-evaluation
//! overhead plus the fact that `compute_eri3` already screens internally at
//! 1e-14, making some QQR-3-dropped triples near-free no-ops already.
//!
//! CONCLUSION: the screen is OFF by default (`eri3_screen_thresh: None`). It is
//! wired, exact at thresh=0, and available for the terfc/large-system case where
//! it is mildly positive -- but it is not a default-worthy win, and enabling it
//! for Coulomb would make RI-MP2 slower.

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn tables_available() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

#[test]
#[ignore = "timing harness; run with --ignored --nocapture"]
fn qqr3_screening_wall_time() {
    let name = std::env::var("FERRIC_BENCH_MOL").unwrap_or_else(|_| "alkane_10".into());
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{name}.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let ctx = ferric_core::parallel::ParallelContext::default();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &RhfConfig::default())
        .unwrap();

    eprintln!(
        "\n=== {name} / cc-pVDZ + cc-pVDZ-RI: nbf={} naux={} ===",
        obs.nbasis(),
        dfbs.nbasis()
    );

    let mut ops: Vec<(&str, Operator)> = vec![("coulomb", Operator::coulomb())];
    if tables_available() {
        ops.push(("terfc(r0=2)", Operator::terfc(2.0)));
    }

    for (label, op) in ops {
        let run = |thresh: Option<f64>| -> (f64, f64) {
            let cfg = RiMp2Config { eri3_screen_thresh: thresh, ..Default::default() };
            // One warm pass, then time the second: the first touches allocator
            // and page-cache state the second does not pay for.
            let _ = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
            let t = Instant::now();
            let e = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
            (t.elapsed().as_secs_f64(), e.mp2_corr)
        };
        let (t_dense, e_dense) = run(None);
        let (t_scr, e_scr) = run(Some(1e-10));
        eprintln!(
            "{label:12}  dense {t_dense:7.3}s   screened(1e-10) {t_scr:7.3}s   \
             speedup {:.3}x   dE {:.2e} Ha",
            t_dense / t_scr.max(1e-9),
            (e_scr - e_dense).abs()
        );
    }
}
