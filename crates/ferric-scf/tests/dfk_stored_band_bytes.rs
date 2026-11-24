//! The DF-K band an f32 RI-J conversion counts beside it is sized in the
//! layout the built DF-K stores (`DfK::stored_layout`): packed f32 when f32
//! was kept, the square f64 tensor when the square layout is in force (the
//! device K mode, or the test hook here) and, without either, when the packed
//! dressing was not taken (the band does not fit), which keeps the square f64
//! tensor too. Its own process: the square hook is process-wide.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::{DressedLayout, ThreeIndexSource};
use ferric_scf::df_k::{set_dfk_square_for_test, DfK, RiKPrecision};

/// The square hook is process-wide: the tests here take turns with it.
static HOOK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn stored_band_bytes_follow_the_built_layout() {
    let _hook = HOOK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mol = Molecule::parse_xyz(
        "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let op = Operator::coulomb();
    let (naux, nao) = (dfbs.nbasis(), obs.nbasis());
    let square = naux * nao * nao * 8;
    let packed32 = ThreeIndexSource::dressed_band_bytes(naux, nao, DressedLayout::PackedF32);
    let ample = 1usize << 40;
    let build = |budget: usize, p: RiKPrecision| {
        DfK::new_with_precision(op, &obs, &dfbs, budget, p).unwrap()
    };

    set_dfk_square_for_test(false);
    let kept = build(ample, RiKPrecision::F32);
    assert_eq!(kept.stored_layout(), DressedLayout::PackedF32);
    assert_eq!(kept.stored_band_bytes(), packed32);

    // Regression: an f32 request whose packed dressing is not taken (the band
    // does not fit this budget) keeps the square f64 tensor, with no device
    // mode and no hook; it is counted at the square size, not a packed one.
    let declined = build(packed32 / 2, RiKPrecision::F32);
    assert_eq!(declined.precision(), RiKPrecision::F64);
    assert_eq!(declined.stored_layout(), DressedLayout::Square);
    assert_eq!(declined.stored_band_bytes(), square);

    // The square hook: the square size, where the packed-room figure of the
    // raw placement is 0.
    set_dfk_square_for_test(true);
    let forced = build(ample, RiKPrecision::F64);
    set_dfk_square_for_test(false);
    assert_eq!(forced.stored_layout(), DressedLayout::Square);
    assert_eq!(forced.stored_band_bytes(), square);
    assert_eq!(
        DfK::packed_band_bytes(naux, nao, RiKPrecision::F64),
        packed32 * 2
    );
}

/// The only DF-K input to the f32 RI-J decision, `stored_band_bytes`, is the
/// same for one layout whether the band stayed in core or spilled (which also
/// reads the pool): a resident and a spilled square band give one figure.
#[test]
fn stored_band_bytes_do_not_depend_on_residency() {
    let _hook = HOOK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mol = Molecule::parse_xyz(
        "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let op = Operator::coulomb();
    let square = dfbs.nbasis() * obs.nbasis() * obs.nbasis() * 8;
    set_dfk_square_for_test(true);
    let resident = DfK::new_with_precision(op, &obs, &dfbs, 1 << 40, RiKPrecision::F64).unwrap();
    let spilled = DfK::new_with_precision(op, &obs, &dfbs, square / 4, RiKPrecision::F64).unwrap();
    set_dfk_square_for_test(false);
    assert_ne!(
        resident.resident_tensor_bytes(),
        spilled.resident_tensor_bytes()
    );
    assert_eq!(resident.stored_layout(), spilled.stored_layout());
    assert_eq!(resident.stored_band_bytes(), square);
    assert_eq!(spilled.stored_band_bytes(), square);
}
