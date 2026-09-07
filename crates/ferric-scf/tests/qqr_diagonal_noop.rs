//! QQR is IDENTICALLY Schwarz on diagonal quartets `(ij|ij)` — which is the
//! only place either LinK pair-list builder consults the bound.
//!
//! This is the structural fact that decides whether wiring `QqrBounds` into
//! LinK can pay off, so it is pinned as a test rather than left as a code
//! reading.
//!
//! Both `SignificantPairs::build` and `DensityPairs::build` (pairs.rs) screen
//! with `bound.estimate(i, j, i, j)` — bra pair == ket pair. QQR's distance
//! envelope is a function of the separation between the bra and ket pair
//! CENTERS, which for `(ij|ij)` is exactly zero, so `QqrBounds::estimate`
//! takes its `r < 1e-14` early return and hands back the raw Schwarz product.
//!
//! Consequence: swapping Schwarz for QQR cannot shrink either pair list. It can
//! only prune inside LinK's innermost quartet test, on quartets that the pair
//! lists have ALREADY admitted — a far smaller population than the full
//! `nsh^4` space over which QQR's 2-5% benefit was measured.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::qqr::QqrBounds;
use ferric_scf::screening::{Bound, SchwarzBounds};

#[test]
fn qqr_equals_schwarz_on_every_diagonal_quartet() {
    let mol = Molecule::load_xyz("../../testdata/molecules/alkane_8.xyz").expect("alkane_8");
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let op = Operator::coulomb();

    let sb = SchwarzBounds::compute(op, &prep).expect("Schwarz bounds");
    let qb = QqrBounds::new(sb.clone(), &mol, &bs, &prep, op);

    let nsh = prep.nshells();
    let mut checked = 0usize;
    for i in 0..nsh {
        for j in 0..nsh {
            let s = Bound::estimate(&sb, i, j, i, j);
            let q = Bound::estimate(&qb, i, j, i, j);
            assert_eq!(
                s, q,
                "QQR must equal Schwarz on the diagonal quartet ({i} {j}|{i} {j}) — \
                 the pair-list builders screen on exactly this quantity"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, nsh * nsh, "expected a full nsh^2 diagonal sweep");

    // Reachability: QQR must NOT be a global no-op, or the assertion above
    // would be vacuous. Off-diagonal quartets between well-separated pairs
    // must be strictly smaller under QQR.
    let mut strictly_smaller = 0usize;
    for i in 0..nsh {
        for k in 0..nsh {
            if Bound::estimate(&qb, i, i, k, k) < Bound::estimate(&sb, i, i, k, k) {
                strictly_smaller += 1;
            }
        }
    }
    assert!(
        strictly_smaller > nsh,
        "QQR tightened only {strictly_smaller} off-diagonal quartets — the diagonal \
         equality above would be measuring a no-op bound, not a real property"
    );
}
