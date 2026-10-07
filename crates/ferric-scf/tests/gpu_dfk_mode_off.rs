#![cfg(feature = "gpu")]
//! With the feature built and `[gpu] mode` at its default (off), the device path
//! must be invisible: every counter unchanged, K bit-identical across builds.
//! (The CPU path's own bit-identity across thread counts is pinned by the
//! `df_k_*bit_identical*` unit tests of `df_k.rs`, which also run under this
//! feature.)
use ferric_core::basis;
use ferric_core::gpu::{settings, stats, GpuMode};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_k::DfK;
use ferric_scf::fock::KBuilder;
use ndarray::Array2;

#[test]
fn mode_off_leaves_every_counter_alone_and_k_is_reproducible() {
    if settings().mode != GpuMode::Off {
        eprintln!("skipping: FERRIC_GPU is set in this environment");
        return;
    }
    let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let n = obs.nbasis();
    let c = Array2::from_shape_fn((n, 5), |(mu, i)| {
        0.05 * (((mu * 5 + i * 3) % 17) as f64 - 8.0)
    });
    let mut dfk = DfK::new(Operator::coulomb(), &obs, &aux, usize::MAX).unwrap();
    let before = stats();
    let (mut k1, mut k2) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    dfk.build_from_occ(&c, &mut k1).unwrap();
    dfk.build_from_occ(&c, &mut k2).unwrap();
    assert_eq!(stats(), before, "mode off must not move any GPU counter");
    assert_eq!(
        k1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        k2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
