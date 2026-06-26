//! Diagnostic: how much does QQR-3 erfc screening perturb the RAW (P|μν)
//! tensor, as a function of threshold and system size? The SR-MP2 b_ov test on
//! decane showed a 3e-5 element-level diff at thresh=1e-10 — is that the raw
//! integral error (loose bound) or a V^{-1/2} amplification? This isolates the
//! raw-tensor part (no SCF, no metric) so it runs fast.
//!
//! Run: cargo run --release -p ferric-integrals --example sr_screen_accuracy

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::qqr3::QqrBounds3;
use ferric_integrals::threeindex::{eri3_tensor, eri3_tensor_screened_qqr};

fn main() {
    let bs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::erfc(0.222);

    println!("# RAW (P|μν) erfc screening accuracy — cc-pVDZ/cc-pVDZ-RI, ω=0.222");
    println!("# maxdiff = max |dense_erfc − qqr_screened_erfc| over the raw AO tensor");
    println!("{:>4} {:>8} | {:>8} {:>10} {:>12}", "C", "thresh", "kept%", "maxdiff", "sum|drop|");

    for n in [4usize, 6, 8, 10] {
        let path = format!("testdata/molecules/alkane_{n}.xyz");
        let mol = match Molecule::load_xyz(&path) { Ok(m) => m, Err(_) => continue };
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux).unwrap();
        let dense = eri3_tensor(op, &obs, &dfbs).unwrap();
        // thresh=0 is bit-identical (no drop); 1e-12/1e-10 expose the bound bug:
        // they drop triples of true magnitude ~3e-5 (error does NOT shrink as
        // thresh tightens ⇒ the exp(-ω²R²) envelope underestimates the integral).
        for thresh in [0.0, 1e-12, 1e-10] {
            let bounds = QqrBounds3::new(op, &mol, &obs, &dfbs).unwrap();
            let (scr, nk, nt) = eri3_tensor_screened_qqr(op, &obs, &dfbs, &bounds, thresh).unwrap();
            let mut maxdiff = 0.0f64;
            let mut sumdrop = 0.0f64;
            for (a, b) in dense.iter().zip(scr.iter()) {
                let d = (a - b).abs();
                if d > maxdiff { maxdiff = d; }
                sumdrop += d;
            }
            println!(
                "{:>4} {:>8.0e} | {:>7.1}% {:>10.3e} {:>12.3e}",
                n, thresh, 100.0 * nk as f64 / nt as f64, maxdiff, sumdrop,
            );
        }
    }
}
