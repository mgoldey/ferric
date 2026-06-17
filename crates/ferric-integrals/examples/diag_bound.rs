//! Diagnostic / validity falsifier for the QQR-3 distance-screening bound.
//!
//! Walks EVERY shell triple (P, s1, s2) of water and alkane_6 for both Coulomb
//! and erfc(0.222) and reports the worst `|true integral| / estimate3` ratio.
//! A ratio > 1 means the shipped `QqrBounds3::estimate3` UNDER-estimates a real
//! 3-index integral and would wrongly drop it — i.e. the bound is INVALID. A
//! valid bound has worst ratio ≤ 1 everywhere.
//!
//! This is the falsifier whose absence let two earlier invalid bounds ship: the
//! permanent unit tests in `qqr3.rs` assert the same property, but this example
//! prints the actual numbers for quick inspection.
//!
//! Run: cargo run --release -p ferric-integrals --example diag_bound
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::qqr3::QqrBounds3;
use ferric_integrals::threeindex::eri3_tensor;

fn worst_ratio(path: &str, op: Operator) -> f64 {
    let mol = Molecule::load_xyz(path).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let qqr = QqrBounds3::new(op, &mol, &obs, &aux).unwrap();
    let dense = eri3_tensor(op, &obs, &aux).unwrap();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_aux = aux.shell_dims();
    let offs_aux = aux.shell_offsets();
    let mut worst = 0.0f64;
    for p in 0..qqr.nsh_aux() {
        for s1 in 0..qqr.nsh_obs() {
            for s2 in 0..=s1 {
                let bound = qqr.estimate3(p, s1, s2);
                let mut t = 0.0f64;
                for pp in 0..dims_aux[p] {
                    for ii in 0..dims_obs[s1] {
                        for jj in 0..dims_obs[s2] {
                            let v = dense[(offs_aux[p] + pp, offs_obs[s1] + ii, offs_obs[s2] + jj)].abs();
                            if v > t { t = v; }
                        }
                    }
                }
                if t > 1e-14 {
                    let r = t / bound.max(1e-300);
                    if r > worst { worst = r; }
                }
            }
        }
    }
    worst
}

fn main() {
    for path in ["testdata/molecules/water.xyz", "testdata/molecules/alkane_6.xyz"] {
        let c = worst_ratio(path, Operator::coulomb());
        let e = worst_ratio(path, Operator::erfc(0.222));
        println!("{path}");
        println!("  Coulomb worst |true|/bound = {c:.6}  ({})", if c <= 1.0 + 1e-9 { "VALID" } else { "INVALID" });
        println!("  erfc    worst |true|/bound = {e:.6}  ({})", if e <= 1.0 + 1e-9 { "VALID" } else { "INVALID" });
    }
}
