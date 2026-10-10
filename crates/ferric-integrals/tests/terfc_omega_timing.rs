//! Cost of the 3-index build vs operator and omega (alkane_4, 6-31G /
//! cc-pVDZ-RI; min of 3 reps). Same shell-triple count for every operator
//! (no Schwarz), so time ratios ARE per-quartet cost ratios. Ignored: run on
//! a quiet box for absolute numbers; the ratios are what this reports.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::eri3_tensor;

#[test]
#[ignore]
fn time_eri3_vs_omega() {
    let xyz = format!(
        "{}/../../testdata/molecules/alkane_4.xyz",
        env!("CARGO_MANIFEST_DIR")
    );
    let mol = Molecule::load_xyz(&xyz).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let r0 = ferric_core::units::ANGSTROM_TO_BOHR;
    let mut ops: Vec<(String, Operator)> = vec![("coulomb".into(), Operator::coulomb())];
    for r0w in [0.5f64.sqrt(), 4.0, 8.0, 16.0] {
        ops.push((format!("erfc r0w={r0w:.3}"), Operator::erfc(r0w / r0)));
        ops.push((
            format!("terfc r0w={r0w:.3}"),
            Operator::terfc_with_omega(r0, r0w / r0),
        ));
    }
    let mut base = 0.0;
    for (name, op) in ops {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let e3 = eri3_tensor(op, &obs, &dfbs).unwrap();
            best = best.min(t0.elapsed().as_secs_f64());
            std::hint::black_box(e3);
        }
        if base == 0.0 {
            base = best;
        }
        eprintln!(
            "TIME {name:<22} eri3 {best:8.3} s  ({:.2}x coulomb)",
            best / base
        );
    }
}
