//! Dress-first RI-JK (`ferric_integrals::three_index_source::ThreeIndexSource::
//! dress_first_wanted`): when the shared raw `(P|μν)` tensor and DF-K's packed
//! dressed band do not fit the budget together, raw is recomputed in one
//! streamed pass to dress K and is never held.
//!
//! WHAT IS PINNED (one BLAS thread, as in production's parallel regions):
//!  * The packed f64 band dressed from a STREAMED raw (recomputed or spilled)
//!    is bitwise the band dressed from a resident raw (unpacked or packed),
//!    for the full band and for a sub-band: the streamed route issues the same
//!    per-tile GEMMs on the same Q grid.
//!  * End to end: an RHF RI-JK SCF whose budget takes the route converges in
//!    the same iterations to the same energy bits as the ample-budget run
//!    (K bitwise; J is RI-J's f32 tensor in both, the rounding of the same
//!    integrals).
//!
//! These live in `ferric-scf` because its test binary links OpenBLAS; the
//! `cblas_dgemm` reference makes this file fail to LINK without it.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::{
    set_jk_storage, DressedLayout, DressedRows, F32Rounding, JkStorage, PackedDressOutcome,
    ThreeIndexSource,
};
use ferric_scf::df_j::RiJPrecision;
use ferric_scf::df_k::{DfK, RiKPrecision};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::sync::{Mutex, MutexGuard};

extern "C" {
    fn cblas_dgemm();
}

static SLOT: Mutex<()> = Mutex::new(());
const AUX: &str = "def2-universal-jkfit";

fn slot() -> MutexGuard<'static, ()> {
    let g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    assert_ne!(cblas_dgemm as *const () as usize, 0);
    assert_eq!(
        std::env::var("OPENBLAS_NUM_THREADS").as_deref(),
        Ok("1"),
        "bit-identity holds at one BLAS thread; run with OPENBLAS_NUM_THREADS=1"
    );
    g
}

fn mol(name: &str) -> Molecule {
    Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// A dense, well-conditioned stand-in for V^{-1/2} (the dressing is linear in
/// it; a full matrix makes every Q contribute to every P).
fn metric(naux: usize) -> Array2<f64> {
    Array2::from_shape_fn((naux, naux), |(i, j)| {
        if i == j {
            1.5
        } else {
            0.01 / ((i as f64 - j as f64).abs() + 1.0)
        }
    })
}

fn packed_bits(raw: &mut ThreeIndexSource, m: &Array2<f64>, band: (usize, usize)) -> Vec<u64> {
    let out = ThreeIndexSource::build_dressed_band_packed(
        raw,
        m,
        usize::MAX / 4,
        band,
        DressedLayout::PackedF64,
        F32Rounding::Nearest,
    )
    .unwrap();
    let PackedDressOutcome::Built(src) = out else {
        panic!(
            "the packed f64 band was not built (raw spilled: {})",
            raw.is_spilled_for_test()
        );
    };
    match src.dressed_packed() {
        Some(DressedRows::F64(rows)) => rows.iter().map(|x| x.to_bits()).collect(),
        _ => panic!("not a packed f64 band"),
    }
}

/// Odd nao (the triangle does not split evenly), naux past two Q blocks.
#[test]
fn streamed_raw_dresses_the_packed_band_bitwise() {
    let _s = slot();
    let mol = mol("alkane_3.xyz");
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let (naux, nao) = (aux.nbasis(), obs.nbasis());
    assert_eq!(nao % 2, 1, "fixture must have an odd nao");
    assert!(
        naux > 2 * 128,
        "fixture must cross the Q grid twice: naux {naux}"
    );
    let op = Operator::coulomb();
    let m = metric(naux);
    let ample = usize::MAX / 4;
    let packed_budget = naux * nao * (nao + 1) / 2 * 8 + 64 * nao * nao * 8;
    for band in [(0, naux), (37, naux - 11)] {
        let mut dense = ThreeIndexSource::build(op, &obs, &aux, ample).unwrap();
        assert!(dense.is_incore());
        let reference = packed_bits(&mut dense, &m, band);
        let mut packed = ThreeIndexSource::build(op, &obs, &aux, packed_budget).unwrap();
        assert!(packed.is_packed_incore_for_test());
        assert!(
            reference == packed_bits(&mut packed, &m, band),
            "{band:?}: packed raw"
        );
        let mut stream = ThreeIndexSource::build_dress_stream(op, &mol, &obs, &aux, ample).unwrap();
        assert!(!stream.is_resident());
        assert!(
            reference == packed_bits(&mut stream, &m, band),
            "{band:?}: recomputed raw"
        );
        set_jk_storage(Some(JkStorage::Disk));
        let spilled = ThreeIndexSource::build_for_jk(op, &mol, &obs, &aux, ample, (0, naux), true);
        set_jk_storage(None);
        let mut spilled = spilled.unwrap();
        assert!(spilled.is_spilled_for_test());
        assert!(
            reference == packed_bits(&mut spilled, &m, band),
            "{band:?}: spilled raw"
        );
    }
}

/// End to end in the route's budget band, against the ample-budget run.
#[test]
fn dress_first_scf_is_bitwise_the_resident_run() {
    let _s = slot();
    let mol = mol("alkane_6.xyz");
    let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let (naux, nao) = (aux.nbasis(), obs.nbasis());
    let dressed = DfK::packed_band_bytes(naux, nao, RiKPrecision::F64);
    let stream = ThreeIndexSource::dress_stream_bytes_for(nao, naux);
    assert!(stream < dressed, "no budget band at this size");
    let budget = dressed + stream + (dressed - stream) / 2;
    assert!(ThreeIndexSource::dress_first_wanted(
        nao, naux, budget, dressed
    ));
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let run = |budget: usize| {
        let cfg = RhfConfig {
            energy_conv: 1e-3,
            density_conv: 1e-8,
            df_j_aux: Some(AUX.into()),
            df_k_aux: Some(AUX.into()),
            three_index_budget_bytes: budget,
            ri_j_precision: Some(RiJPrecision::F32),
            ri_k_precision: Some(RiKPrecision::F64),
            ..Default::default()
        };
        solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap()
    };
    let before = ferric_scf::dress_first_count_for_test();
    let resident = run(usize::MAX / 4);
    assert_eq!(ferric_scf::dress_first_count_for_test(), before);
    let streamed = run(budget);
    assert_eq!(
        ferric_scf::dress_first_count_for_test(),
        before + 1,
        "the budget did not take the route"
    );
    assert!(resident.converged && streamed.converged);
    assert_eq!(resident.iterations, streamed.iterations);
    assert_eq!(
        resident.energy.to_bits(),
        streamed.energy.to_bits(),
        "{} vs {}",
        resident.energy,
        streamed.energy
    );
    let route = streamed.df_jk.as_ref().unwrap();
    assert_eq!(route.ri_j_precision, RiJPrecision::F32);
    assert_eq!(route.ri_k_precision, RiKPrecision::F64);
}
