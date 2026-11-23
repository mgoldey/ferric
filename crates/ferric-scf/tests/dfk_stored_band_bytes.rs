//! The DF-K band an f32 RI-J conversion counts beside it is sized in the
//! layout DF-K stores: the square f64 tensor when the square layout is in
//! force (the device K mode, or the test hook here), the packed f64 triangle
//! after an f32 request fell back to f64, and the packed f32 triangle when f32
//! was kept. Its own process: the square hook is process-wide.

use ferric_integrals::three_index_source::{DressedLayout, ThreeIndexSource};
use ferric_scf::df_k::{set_dfk_square_for_test, DfK, RiKPrecision};

#[test]
fn stored_band_bytes_follow_the_stored_layout() {
    let (band, nao) = (120, 24);
    let square = band * nao * nao * 8;
    let packed64 = ThreeIndexSource::dressed_band_bytes(band, nao, DressedLayout::PackedF64);
    let packed32 = ThreeIndexSource::dressed_band_bytes(band, nao, DressedLayout::PackedF32);
    assert!(packed32 < packed64 && packed64 < square);

    set_dfk_square_for_test(false);
    assert_eq!(
        DfK::stored_band_bytes(RiKPrecision::F32, band, nao),
        packed32
    );
    // An f32 request whose band fell back to f64 is charged at the f64 size.
    assert_eq!(
        DfK::stored_band_bytes(RiKPrecision::F64, band, nao),
        packed64
    );

    // The square f64 layout: the band's full size, where the packed-room
    // figure of the raw placement is 0.
    set_dfk_square_for_test(true);
    assert_eq!(DfK::stored_band_bytes(RiKPrecision::F64, band, nao), square);
    assert_eq!(DfK::packed_band_bytes(band, nao, RiKPrecision::F64), 0);
    set_dfk_square_for_test(false);
}
