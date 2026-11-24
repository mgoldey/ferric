//! The DF-K band's layout and sizes. `stored_layout` / `stored_band_bytes`
//! report what was built: packed f32 when f32 was kept, the square f64 tensor
//! when the square layout is in force (the device K mode, or the test hook
//! here) and, without either, when the packed dressing was not taken (the
//! band does not fit the budget or the pool). `nominal_layout` /
//! `nominal_band_bytes`, what the f32 RI-J decision counts, depend on the
//! request, the inputs and the budget only: square only when forced, so a
//! pool-forced square band gives the same figure as a packed one. Its own
//! process: the square hook and the global pool are process-wide.

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

    assert_eq!(kept.nominal_layout(), DressedLayout::PackedF32);
    assert_eq!(kept.nominal_band_bytes(), packed32);

    // An f32 request whose packed dressing is not taken (the band does not
    // fit this budget) keeps the square f64 tensor, with no device mode and no
    // hook. `stored_layout` reports it; the deterministic figure
    // (`nominal_band_bytes`, what the f32 RI-J decision counts) stays packed
    // f32.
    let declined = build(packed32 / 2, RiKPrecision::F32);
    assert_eq!(declined.precision(), RiKPrecision::F64);
    assert_eq!(declined.stored_layout(), DressedLayout::Square);
    assert_eq!(declined.stored_band_bytes(), square);
    assert_eq!(declined.nominal_layout(), DressedLayout::PackedF32);
    assert_eq!(declined.nominal_band_bytes(), packed32);

    // The same at an ample budget when the POOL cannot hold the packed band:
    // a pool-forced square band gives the decision the same figure as the
    // packed band built without the pool (same shapes, same budget).
    let pool = ferric_core::memory::pool::MemoryPool::with_capacity_bytes(square + packed32 / 2);
    ferric_core::memory::pool::install_global(pool);
    let pooled = build(ample, RiKPrecision::F32);
    ferric_core::memory::pool::clear_global();
    assert_eq!(
        pooled.stored_layout(),
        DressedLayout::Square,
        "the pool must force square"
    );
    assert_eq!(pooled.nominal_band_bytes(), kept.nominal_band_bytes());

    // The square hook: the square size, where the packed-room figure of the
    // raw placement is 0.
    set_dfk_square_for_test(true);
    let forced = build(ample, RiKPrecision::F64);
    set_dfk_square_for_test(false);
    assert_eq!(forced.stored_layout(), DressedLayout::Square);
    assert_eq!(forced.stored_band_bytes(), square);
    // A forced square layout is the budget-determined one: counted as square.
    assert_eq!(forced.nominal_layout(), DressedLayout::Square);
    assert_eq!(forced.nominal_band_bytes(), square);
    assert_eq!(
        DfK::packed_band_bytes(naux, nao, RiKPrecision::F64),
        packed32 * 2
    );
}

/// For one layout, the band's size is the same whether it stayed in core or
/// spilled (which also reads the pool): a resident and a spilled square band
/// give one figure, stored and nominal.
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
    assert_eq!(resident.nominal_band_bytes(), spilled.nominal_band_bytes());
}
