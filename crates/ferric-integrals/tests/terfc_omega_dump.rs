//! Dumps the AO 3-index (P|terfc|mu nu) and 2-index (P|terfc|Q) tensors of
//! water/cc-pVDZ + cc-pVDZ-RI at several (r0, r0*omega) as raw little-endian
//! f64 files, for the independent Fourier-space comparison in
//! `scripts/validation/terfc_omega_fourier_check.py` (ordering-invariant
//! singular-value comparison). Ignored: needs FERRIC_TERF_TABLE_DIR and
//! FERRIC_DUMP_DIR.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::{coulomb_metric_2c, eri3_tensor};
use std::io::Write;

fn dump(path: &str, it: impl Iterator<Item = f64>) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for x in it {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}

#[test]
#[ignore]
fn dump_terfc_omega_tensors() {
    let dir = std::env::var("FERRIC_DUMP_DIR").expect("FERRIC_DUMP_DIR");
    let xyz = format!(
        "{}/../../testdata/molecules/water.xyz",
        env!("CARGO_MANIFEST_DIR")
    );
    let mol = Molecule::load_xyz(&xyz).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    // Default: r0 = 1 A x r0*omega {0.7071, 4, 8, 16}. Override with
    // FERRIC_DUMP_CASES="tag:r0_ang:r0w,..."; cases.txt is written for the
    // python check, which then uses each case's own r0.
    let cases: Vec<(String, f64, f64)> = match std::env::var("FERRIC_DUMP_CASES") {
        Ok(v) => v
            .split(',')
            .map(|c| {
                let p: Vec<&str> = c.split(':').collect();
                (
                    p[0].to_string(),
                    p[1].parse().unwrap(),
                    p[2].parse().unwrap(),
                )
            })
            .collect(),
        Err(_) => vec![
            ("0.7071".into(), 1.0, 0.5f64.sqrt()),
            ("4".into(), 1.0, 4.0),
            ("8".into(), 1.0, 8.0),
            ("16".into(), 1.0, 16.0),
        ],
    };
    let mut ct = String::new();
    for (tag, r0a, r0w) in &cases {
        ct += &format!("{tag} {r0a} {r0w}\n");
    }
    std::fs::write(format!("{dir}/cases.txt"), ct).unwrap();
    for (tag, r0a, r0w) in cases.iter().map(|(t, a, w)| (t.as_str(), *a, *w)) {
        let r0 = r0a * ferric_core::units::ANGSTROM_TO_BOHR;
        let op = Operator::terfc_with_omega(r0, r0w / r0);
        let t0 = std::time::Instant::now();
        let e3 = eri3_tensor(op, &obs, &dfbs).unwrap();
        let t3 = t0.elapsed().as_secs_f64();
        let e2 = coulomb_metric_2c(op, &dfbs).unwrap();
        eprintln!(
            "DUMP r0w={tag} 3c shape {:?} t3={t3:.3}s nao={} naux={}",
            e3.shape(),
            obs.nbasis(),
            dfbs.nbasis()
        );
        dump(&format!("{dir}/e3_{tag}.bin"), e3.iter().copied());
        dump(&format!("{dir}/e2_{tag}.bin"), e2.iter().copied());
    }
}
