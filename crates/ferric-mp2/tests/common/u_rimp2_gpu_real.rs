//! A real UHF (RI-JK) reference and its U-RI-MP2 intermediates.
use crate::fixture::Chan;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{RiMp2Config, RpaIntermediates};
use ferric_mp2::u_rimp2::{compute_u_mp2_amplitudes, UMp2Amplitudes};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};

/// The full inputs of one open-shell case, so `u_ri_mp2` can be re-run on them.
pub struct Case {
    pub mol: Molecule,
    pub obs: PreparedBasis,
    pub dfbs: PreparedBasis,
    pub scf: ferric_scf::ScfResult,
    pub amps: UMp2Amplitudes,
}

/// UHF with RI-JK (`def2-universal-jkfit`, loose thresholds: every arm compares
/// on the SAME orbitals) and the cc-pVDZ / cc-pvdz-ri RI-MP2 intermediates.
pub fn real_case(xyz: &str, charge: i32, mult: usize) -> Case {
    let root = env!("CARGO_MANIFEST_DIR");
    let mol = Molecule::load_xyz_with_charge(
        &format!("{root}/../../testdata/molecules/{xyz}"),
        charge,
        mult,
    )
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = UhfConfig {
        max_iter: 200,
        energy_conv: 1e-8,
        density_conv: 1e-6,
        integral_thresh: 1e-9,
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let scf = solve_uhf(&ParallelContext::default(), &mol, &obs, &bounds, &cfg).unwrap();
    assert!(scf.converged, "{xyz}: UHF did not converge");
    let amps =
        compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &scf, &RiMp2Config::default()).unwrap();
    Case {
        mol,
        obs,
        dfbs,
        scf,
        amps,
    }
}

/// The two spin channels of a case (owned copies of the intermediates).
pub fn chans(a: &UMp2Amplitudes) -> (Chan, Chan) {
    let mk = |i: &RpaIntermediates, eps: &[f64]| Chan {
        b: i.b_ov.clone(),
        eps: eps.to_vec(),
        nocc: i.nocc,
        nvir: i.nvir,
        first_occ: i.first_occ,
        nocc_total: i.nocc_total,
    };
    (mk(&a.inter_a, &a.eps_a), mk(&a.inter_b, &a.eps_b))
}

/// The two spin channels of `case` with the RI-MP2 intermediates (and metric)
/// under `op` instead of Coulomb.
#[allow(dead_code)] // not every binary that includes this module uses it
pub fn chans_for_op(case: &Case, op: Operator) -> (Chan, Chan) {
    let amps = compute_u_mp2_amplitudes(
        &case.mol,
        &case.obs,
        &case.dfbs,
        op,
        &case.scf,
        &RiMp2Config::default(),
    )
    .unwrap();
    chans(&amps)
}
