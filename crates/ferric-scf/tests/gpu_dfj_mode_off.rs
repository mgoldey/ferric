#![cfg(feature = "gpu")]
//! With the feature built and `[gpu] mode` at its default (off), the device RI-J
//! must be invisible: every counter unchanged, J bit-identical to the forced-host
//! build (the code path of a build without the feature).
use ferric_core::basis;
use ferric_core::gpu::{settings, stats, GpuMode};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::DfJ;
use ferric_scf::fock::JBuilder;
use ndarray::Array2;
use std::sync::atomic::Ordering;

#[test]
fn mode_off_leaves_every_counter_alone_and_j_is_the_host_j() {
    if settings().mode != GpuMode::Off {
        eprintln!("skipping: FERRIC_GPU is set in this environment");
        return;
    }
    let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let n = obs.nbasis();
    let d = Array2::from_shape_fn((n, n), |(i, j)| 0.01 * ((i * 7 + j * 3) % 11) as f64);
    let mut dfj = DfJ::new(Operator::coulomb(), &obs, &aux, usize::MAX).unwrap();
    let before = stats();
    let (mut j1, mut j2) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    dfj.build(&d, &mut j1).unwrap();
    dfj.build(&d, &mut j2).unwrap();
    assert_eq!(stats(), before, "mode off must not move any GPU counter");
    assert!(!dfj.device_resident_for_test());
    let bits = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&j1), bits(&j2));
    ferric_scf::df_j_gpu::FORCE_HOST.store(true, Ordering::SeqCst);
    let mut j3 = Array2::zeros((n, n));
    dfj.build(&d, &mut j3).unwrap();
    ferric_scf::df_j_gpu::FORCE_HOST.store(false, Ordering::SeqCst);
    assert_eq!(bits(&j1), bits(&j3));
}
