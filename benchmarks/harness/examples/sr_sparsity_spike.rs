//! SR-MP2 sparsity spike — the falsifier for "linear-scaling SR-MP theory".
//!
//! Claim under test: the erfc(ωr)/r kernel localizes the significant shell-pair
//! list EXPONENTIALLY, so the pair count grows O(N) (fixed neighbors per shell),
//! whereas the bare Coulomb kernel gives the usual O(N²) growth (every shell pair
//! survives the Schwarz bound on a compact molecule).
//!
//! If erfc pairs/shell PLATEAUS to a constant while Coulomb pairs/shell keeps
//! climbing, the locality premise holds and a linear-scaling SR pathway is worth
//! building. If erfc tracks Coulomb, the premise is wrong — stop.
//!
//! Geometry series: linear alkanes C1..C20 (testdata/molecules/alkane_N.xyz) — a
//! 1D chain, so the asymptote is the strongest possible: a truly local kernel
//! must reach a HARD constant neighbor count, not a slowly-growing one.
//!
//! Run:  cargo run --release -p ferric-rpa --example sr_sparsity_spike
//! Pure geometry/screening — no SCF, no integrals beyond the Schwarz diagonal.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::pairs::SignificantPairs;
use ferric_scf::screening::{Bound, SchwarzBounds};

/// ω in Bohr⁻¹. 0.222 Bohr⁻¹ = 0.420 Å⁻¹ — the production erfc-optimal value
/// (Goldey & Head-Gordon 2012) the whole SR-MP2+LR-RPA lane is tuned to.
const OMEGA_BOHR: f64 = 0.222;

/// Integral-bound threshold for "significant" — matches the LinK production
/// screening default. Pairs with diagonal Schwarz bound ≤ this are dropped.
const THRESH: f64 = 1e-10;

fn count_pairs(prep: &PreparedBasis, op: Operator) -> (usize, usize) {
    let bounds = SchwarzBounds::compute(op, prep).unwrap();
    let nsh = prep.nshells();
    // SignificantPairs uses the SAME diagonal bound estimate(i,j,i,j) the real
    // SR pair list would, so this is the production locality estimator verbatim.
    let sig = SignificantPairs::build(&bounds as &dyn Bound, nsh, THRESH);
    (nsh, sig.total_pairs())
}

fn main() {
    let bs = basis::bundled("cc-pvdz").unwrap();
    let erfc = Operator::erfc(OMEGA_BOHR);
    let coul = Operator::coulomb();

    println!(
        "# SR sparsity spike — cc-pVDZ, ω={OMEGA_BOHR} Bohr⁻¹, thresh={THRESH:.0e}\n\
         # erfc pairs/shell should PLATEAU (O(N)); Coulomb should CLIMB (O(N²))\n"
    );
    println!(
        "{:>4} {:>6} {:>6} | {:>10} {:>9} | {:>10} {:>9} | {:>8}",
        "C", "atoms", "nsh", "coul_prs", "coul/sh", "erfc_prs", "erfc/sh", "ratio"
    );
    println!("{}", "-".repeat(78));

    for n in 1..=20usize {
        let path = format!("testdata/molecules/alkane_{n}.xyz");
        let mol = match Molecule::load_xyz(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let prep = PreparedBasis::new(&mol, &bs).unwrap();

        let (nsh, coul_prs) = count_pairs(&prep, coul);
        let (_, erfc_prs) = count_pairs(&prep, erfc);

        let coul_per = coul_prs as f64 / nsh as f64;
        let erfc_per = erfc_prs as f64 / nsh as f64;
        let ratio = coul_prs as f64 / erfc_prs.max(1) as f64;

        println!(
            "{:>4} {:>6} {:>6} | {:>10} {:>9.2} | {:>10} {:>9.2} | {:>8.2}",
            n, prep.natoms(), nsh, coul_prs, coul_per, erfc_prs, erfc_per, ratio
        );
    }

    println!(
        "\n# Verdict heuristic: if erfc/sh flattens to a constant past ~C8 while\n\
         # coul/sh keeps rising linearly in nsh, the O(N) SR premise HOLDS."
    );
}
