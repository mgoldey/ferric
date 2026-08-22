//! SCRATCH diagnostic probe for the terfc far-field overshoot. Not a gate.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;

const A2B: f64 = 1.889_725_988_6;

fn have_tables() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

/// Compare terfc vs Coulomb elementwise for the REAL alkane_4 basis at several
/// r0, in the 2-center metric and the 3-index tensor. If terfc -> Coulomb from
/// below pointwise, every element deviation should shrink with r0 and the
/// terfc value should never exceed Coulomb where Coulomb > 0.
#[test]
#[ignore = "benchmark: terfc table probe; --release --ignored --nocapture"]
fn probe_terfc_vs_coulomb_elementwise() {
    if !have_tables() {
        eprintln!("skip: no tables");
        return;
    }
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/alkane_4.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();

    let v_c = threeindex::coulomb_metric_2c(Operator::coulomb(), &dfbs).unwrap();
    let b_c = threeindex::eri3_tensor(Operator::coulomb(), &obs, &dfbs).unwrap();

    for &r0a in &[1.0_f64, 1.5, 2.0, 3.0, 6.0, 12.0] {
        let op = Operator::terfc(r0a * A2B);
        let v_t = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let b_t = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();

        // 2-center
        let mut max_abs = 0.0f64;
        let mut max_over = 0.0f64; // largest positive (terfc - coulomb) where coulomb>0
        let mut sum_abs = 0.0f64;
        for (a, b) in v_t.iter().zip(v_c.iter()) {
            let d = a - b;
            max_abs = max_abs.max(d.abs());
            sum_abs += d.abs();
            // terfc <= coulomb pointwise for the kernel; sign per element is
            // basis-dependent, so record the signed extreme relative to |coul|.
            if *b > 0.0 && d > max_over {
                max_over = d;
            }
        }
        // 3-index
        let mut max_abs3 = 0.0f64;
        let mut sum_abs3 = 0.0f64;
        for (a, b) in b_t.iter().zip(b_c.iter()) {
            let d = a - b;
            max_abs3 = max_abs3.max(d.abs());
            sum_abs3 += d.abs();
        }
        eprintln!(
            "r0={r0a:5.1}A  2c: maxabs={max_abs:.3e} sumabs={sum_abs:.3e} maxpos={max_over:.3e} \
             | 3c: maxabs={max_abs3:.3e} sumabs={sum_abs3:.3e}"
        );
    }
}

/// DECISIVE probe. By shim construction terfc + terf == MD-Coulomb exactly
/// (same code path, only the final combine differs). So
///   D = (terfc + terf) - libint_coulomb
/// is the PURE MD-vs-libint Coulomb discrepancy with NO cancellation involved.
/// A residue here at the size of the energy discrepancy implicates the MD
/// Coulomb pass, not the terf tables/series and not catastrophic cancellation.
#[test]
#[ignore = "benchmark: terfc table probe; --release --ignored --nocapture"]
fn probe_terfc_plus_terf_equals_coulomb() {
    if !have_tables() {
        eprintln!("skip: no tables");
        return;
    }
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/alkane_4.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();

    let b_c = threeindex::eri3_tensor(Operator::coulomb(), &obs, &dfbs).unwrap();
    let v_c = threeindex::coulomb_metric_2c(Operator::coulomb(), &dfbs).unwrap();
    let cmax3 = b_c.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let cmax2 = v_c.iter().fold(0.0f64, |m, v| m.max(v.abs()));

    for &r0a in &[1.0_f64, 2.0, 3.0] {
        let b_t = threeindex::eri3_tensor(Operator::terfc(r0a * A2B), &obs, &dfbs).unwrap();
        let b_l = threeindex::eri3_tensor(Operator::terf(r0a * A2B), &obs, &dfbs).unwrap();
        let v_t = threeindex::coulomb_metric_2c(Operator::terfc(r0a * A2B), &dfbs).unwrap();
        let v_l = threeindex::coulomb_metric_2c(Operator::terf(r0a * A2B), &dfbs).unwrap();

        let mut d3 = 0.0f64;
        for ((a, b), c) in b_t.iter().zip(b_l.iter()).zip(b_c.iter()) {
            d3 = d3.max((a + b - c).abs());
        }
        let mut d2 = 0.0f64;
        for ((a, b), c) in v_t.iter().zip(v_l.iter()).zip(v_c.iter()) {
            d2 = d2.max((a + b - c).abs());
        }
        eprintln!(
            "r0={r0a:5.1}A  |terfc+terf-coul| 3c max={d3:.3e} (rel {:.2e})  2c max={d2:.3e} (rel {:.2e})",
            d3 / cmax3,
            d2 / cmax2
        );
    }
}
