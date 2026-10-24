//! A refusal from the libint2-native terf engine's hook must come back as
//! `Err(status)` from the checked compute paths, not hit an
//! `assert!(written >= 0)` panic. Its own test binary: the fault injection it
//! uses is process-global and would corrupt concurrent tests.
#![cfg(feature = "libint2_terf")]

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::{Engine, TerfLibint2Braket};
use ferric_integrals::ffi;

#[test]
fn hook_refusal_is_err_not_panic() {
    let dir = std::env::var("FERRIC_TERF_TABLE_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../terf-tables").to_string());
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let r0 = 2.0_f64;
    let omega = 1.0 / (r0 * 2.0_f64.sqrt());
    let np = obs.max_nprim() as usize;
    let mk = |bk| Engine::new_terf_libint2(r0, omega, bk, np, 2, 0.0, Some(&dir)).unwrap();
    let (mut e4, mut e3, mut e2) = (
        mk(TerfLibint2Braket::XxXx),
        mk(TerfLibint2Braket::XsXx),
        mk(TerfLibint2Braket::XsXs),
    );

    let run = |e4: &mut Engine, e3: &mut Engine, e2: &mut Engine| {
        (
            e4.compute_quartet_checked(&obs, 0, 0, 0, 0)
                .map(|b| b.map(<[f64]>::to_vec)),
            e3.compute_eri3_checked(&obs, &obs, 0, 0, 0)
                .map(|b| b.map(<[f64]>::to_vec)),
            e2.compute_eri2_checked(&obs, 0, 0).map(<[f64]>::to_vec),
        )
    };
    let before = run(&mut e4, &mut e3, &mut e2);
    assert!(
        matches!(&before.0, Ok(Some(_))) && matches!(&before.1, Ok(Some(_))) && before.2.is_ok()
    );

    // SAFETY: plain flag store; this test binary runs nothing else.
    unsafe { ffi::scf_terf_libint2_inject_hook_fault(1) };
    let faulted = run(&mut e4, &mut e3, &mut e2);
    let folded = e4.compute_quartet(&obs, 0, 0, 0, 0).is_none()
        && e3.compute_eri3(&obs, &obs, 0, 0, 0).is_none();
    unsafe { ffi::scf_terf_libint2_inject_hook_fault(0) };
    assert_eq!(faulted.0, Err(ffi::SCF_EINTERNAL), "quartet");
    assert_eq!(faulted.1, Err(ffi::SCF_EINTERNAL), "eri3");
    assert_eq!(faulted.2, Err(ffi::SCF_EINTERNAL), "eri2");
    assert!(folded, "unchecked quartet/eri3 fold a refusal into None");

    // The engines stay usable and give the same bits once the fault clears.
    let after = run(&mut e4, &mut e3, &mut e2);
    assert_eq!(before, after);
}
