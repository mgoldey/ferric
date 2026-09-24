//! Stage-1 PBC step 8: SR nucleus-screening study (reference/pbc/stage1-design.md
//! §4 "Screening", §6 row 8). A MEASUREMENT commit, not a production path.
//!
//! `periodic_hcore`'s short-range erfc lattice sum (Gaussian nucleus, 3c)
//! skips a (pair image, nucleus image) triplet when its per-triplet bound
//! `B(d) = q Z_max (1 + 2√(p_max/π)) e^{−ω_p² (d − 2)²}` is below the
//! threshold. `hcore::sr_screening_study` fixes the candidate/pair image sets
//! (built at `cand_thresh`), sweeps the per-triplet screen threshold, and
//! reports for each: triplets computed, max|ΔV| vs the unscreened sum over the
//! same sets, and the bound's own predicted error (Σ of B over the skipped
//! triplets, per element).
//!
//! Hypotheses, stated before measuring (Experimental Protocol):
//! * PHYSICS/BOUND CORRECT: `Derived` has zero violations (`|ΔV_ij| ≤
//!   predicted_ij` everywhere, above the roundoff floor) at every threshold,
//!   and max|ΔV| falls with the threshold.
//! * ARTIFACT: if the screen and the prediction were wired to different
//!   quantities (e.g. predicting from a bound the loop does not use), the
//!   prediction could exceed the error trivially. The NEGATIVE CONTROLS guard
//!   against a table that cannot fail: `SrBound::NoGaussianExtent` (ω_p = ω,
//!   drops the product-Gaussian convolution) and `SrBound::NoMargin` (drops
//!   the 2-Bohr allowance for l > 0 prefactors) are anti-conservative by
//!   construction and should show violations somewhere in the table. If NO
//!   mutant ever violates, the table cannot distinguish a conservative bound
//!   from an untested one — redesign (smaller ω_p·margin products, more
//!   diffuse bases) before concluding the bound is safe.
//!
//! Fast anchors (run by default):
//! * `sr_screen_threshold_to_zero_is_the_unscreened_sum` — screen at 1e-300
//!   ≡ unscreened (threshold 0) to ≤ 1e-15.
//! * `sr_attraction_matrix_is_what_periodic_hcore_uses` — the harness'
//!   matrix symmetrised is BITWISE `periodic_hcore(..).v_sr` at the same
//!   precision (the study measures the production loop, not a copy).
//!
//! The measurement (ignored; release, serial, quiet box):
//! ```text
//! OPENBLAS_NUM_THREADS=1 scripts/ferric-limited -- cargo test --release -p ferric-pbc \
//!     --test pbc_sr_screening -- --ignored --nocapture sr_screening_table
//! ```
//! `FERRIC_PBC_SR_STUDY=h2,tri` restricts the systems (default: h2,tri,h2o;
//! the H2O/cc-pVDZ unscreened reference is the expensive one).

mod common;

use common::*;
use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::ewald::default_ewald_omega;
use ferric_pbc::hcore::{
    periodic_hcore, sr_attraction_matrix, sr_screening_study, PeriodicHcoreConfig, SrBound,
};
use ferric_pbc::lattice::Cell;

#[test]
fn sr_screen_threshold_to_zero_is_the_unscreened_sum() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let (omega, cand) = (0.8, 1e-10);
    let (v0, n0) = sr_attraction_matrix(&cell, &prep, omega, cand, 0.0, SrBound::Derived).unwrap();
    let (vt, nt) =
        sr_attraction_matrix(&cell, &prep, omega, cand, 1e-300, SrBound::Derived).unwrap();
    let d = max_abs_diff(&v0, &vt);
    eprintln!("unscreened {n0} triplets, screen 1e-300 {nt} triplets, max|dV| {d:.3e}");
    assert!(
        d <= 1e-15,
        "threshold -> 0 must reproduce the unscreened sum: {d:.3e}"
    );
    // Non-vacuity: inside this candidate set the screen does skip work at a
    // loose threshold, and that visibly changes V (else the anchor is empty).
    let (vl, nl) = sr_attraction_matrix(&cell, &prep, omega, cand, 1e-4, SrBound::Derived).unwrap();
    eprintln!(
        "screen 1e-4: {nl} triplets, max|dV| {:.3e}",
        max_abs_diff(&v0, &vl)
    );
    assert!(nl < n0, "the screen skipped nothing at 1e-4 ({nl} of {n0})");
    assert!(max_abs_diff(&v0, &vl) > 0.0);
}

#[test]
fn sr_attraction_matrix_is_what_periodic_hcore_uses() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let (omega, prec) = (1.3, 1e-10);
    let cfg = PeriodicHcoreConfig {
        precision: prec,
        ..PeriodicHcoreConfig::with_omega(omega)
    };
    let hc = periodic_hcore(&cell, &prep, &cfg).unwrap();
    let (v, n) = sr_attraction_matrix(&cell, &prep, omega, prec, prec, SrBound::Derived).unwrap();
    let sym = 0.5 * (&v + &v.t());
    assert_eq!(n, hc.n_sr_triplets);
    assert!(
        sym.iter()
            .zip(hc.v_sr.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "harness V_SR differs from periodic_hcore's: max {:.3e}",
        max_abs_diff(&sym, &hc.v_sr)
    );
}

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

fn study_systems() -> Vec<(&'static str, Cell, PreparedBasis)> {
    let want = std::env::var("FERRIC_PBC_SR_STUDY").unwrap_or_else(|_| "h2,tri,h2o".into());
    let want: Vec<&str> = want.split(',').map(str::trim).collect();
    let mut out = Vec::new();
    if want.contains(&"h2") {
        let cell = h2_cell(4.0);
        let prep = prep_for(&cell, &pyscf_sto3g_h());
        out.push(("H2/STO-3G a=4", cell, prep));
    }
    if want.contains(&"tri") {
        let cell = triclinic_cell();
        let prep = prep_for(&cell, &sp_basis_h());
        out.push(("triclinic 4H s+p", cell, prep));
    }
    if want.contains(&"h2o") {
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let cell = Cell::new(mol, cubic(8.0)).unwrap();
        let prep = PreparedBasis::new(cell.mol(), &bundled("cc-pvdz").unwrap()).unwrap();
        out.push(("H2O/cc-pVDZ a=8", cell, prep));
    }
    out
}

#[test]
#[ignore = "measurement: run manually, prints a table"]
fn sr_screening_table() {
    let thresholds = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 0.0];
    let bounds = [
        SrBound::Derived,
        SrBound::NoGaussianExtent,
        SrBound::NoMargin,
    ];
    // Candidate/pair image sets: 100x below the tightest swept threshold.
    let cand = 1e-18;
    for (name, cell, prep) in study_systems() {
        for omega in [default_ewald_omega(&cell), 1.3] {
            let t0 = std::time::Instant::now();
            let st =
                sr_screening_study(&cell, &prep, omega, cand, &thresholds, &bounds, None).unwrap();
            println!(
                "\n== {name}  omega = {omega:.4}  cand_thresh = {cand:.0e}  images {}  \
                 nucleus candidates {}  unscreened triplets {}  roundoff floor {:.2e}  ({:.1} s)",
                st.n_images,
                st.n_candidates,
                st.n_triplets_unscreened,
                st.roundoff_floor,
                t0.elapsed().as_secs_f64()
            );
            println!(
                "{:<17} {:>8} {:>10} {:>11} {:>11} {:>6} {:>10}",
                "bound", "thresh", "triplets", "max|dV|", "max pred", "viol", "worst"
            );
            let mut viol = std::collections::HashMap::new();
            for r in &st.rows {
                println!(
                    "{:<17} {:>8.0e} {:>10} {:>11.3e} {:>11.3e} {:>6} {:>10.3e}",
                    format!("{:?}", r.bound),
                    r.thresh,
                    r.n_triplets,
                    r.max_abs_dv,
                    r.max_predicted,
                    r.n_violations,
                    r.worst_ratio
                );
                *viol.entry(format!("{:?}", r.bound)).or_insert(0usize) += r.n_violations;
            }
            let derived = viol.get("Derived").copied().unwrap_or(0);
            let mutants: usize = viol
                .iter()
                .filter(|(k, _)| k.as_str() != "Derived")
                .map(|(_, v)| v)
                .sum();
            println!(
                "verdict: Derived violations {derived} ({}); mutant violations {mutants} \
                 (negative control {})",
                if derived == 0 {
                    "conservative here"
                } else {
                    "NOT CONSERVATIVE"
                },
                if mutants > 0 {
                    "fires"
                } else {
                    "SILENT - table cannot certify the bound"
                }
            );
        }
    }
}
