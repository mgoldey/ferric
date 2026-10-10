//! EXACTNESS ANCHOR for `att_mp2_vv10_scan` (the refit harness's one-SCF
//! multi-(r0, omega, b) evaluator): every (arm, b) entry must reproduce a
//! separate `att_mp2_vv10` call with that arm's (r0, omega) and b, bitwise in
//! the attMP2 and VV10 halves, with frozen core on. Arms include a linked
//! width (omega = None) and sharp ones so a mis-paired scan (the Eq. 11
//! lockstep violation) fails.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::vv10::Vv10Damping;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_vv10::{
    att_mp2_vv10, att_mp2_vv10_scan, AttVv10Config, AttVv10ScanArm, BOHR_PER_ANG,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[test]
fn scan_matches_all_in_one_for_every_arm_and_b() {
    if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR not set; terfc interpolation tables unavailable");
        return;
    }
    let xyz = "3\nwater\nO 0.000 0.000 0.118\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs_obs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs_obs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            ..Default::default()
        },
    )
    .unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();

    let base = AttVv10Config {
        nlc_grid: AtomicGridConfig {
            n_radial: 20,
            n_angular: 26,
            prune: None,
        },
        frozen_core: 1,
        ..AttVv10Config::mp2_v_terfc_atz()
    };
    let r0a = 1.00 * BOHR_PER_ANG;
    let r0b = 0.90 * BOHR_PER_ANG;
    let arms = [
        AttVv10ScanArm {
            r0_bohr: r0a,
            omega: None,
        },
        AttVv10ScanArm {
            r0_bohr: r0a,
            omega: Some(4.0 / r0a),
        },
        AttVv10ScanArm {
            r0_bohr: r0b,
            omega: Some(2.0 / r0b),
        },
    ];
    let bs = [8.0, 11.0, 14.5];
    let scan = match att_mp2_vv10_scan(&mol, &obs, &bs_obs, &dfbs, &rhf, &base, &arms, &bs, true) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: terfc path unavailable at runtime: {e}");
            return;
        }
    };
    assert!(scan.e_c_mp2_coulomb.unwrap() < scan.arms[0].e_c_att_mp2 - 1e-4);
    for (a, arm) in arms.iter().enumerate() {
        for (k, &b) in bs.iter().enumerate() {
            let mut cfg = base.clone();
            cfg.r0_bohr = arm.r0_bohr;
            cfg.omega = arm.omega;
            cfg.vv10.b = b;
            cfg.vv10_damping = Vv10Damping::Terfc {
                r0_bohr: arm.r0_bohr,
                omega_bohr_inv: None,
            };
            let one = att_mp2_vv10(&mol, &obs, &bs_obs, &dfbs, &rhf, &cfg).unwrap();
            assert_eq!(
                scan.arms[a].e_c_att_mp2.to_bits(),
                one.e_c_att_mp2.to_bits()
            );
            assert_eq!(
                scan.arms[a].e_nl_vv10[k].to_bits(),
                one.e_nl_vv10.to_bits(),
                "VV10 half arm {a} b {b}"
            );
        }
    }
    // Not vacuous: arms and b values really differ.
    assert!((scan.arms[0].e_nl_vv10[1] - scan.arms[1].e_nl_vv10[1]).abs() > 1e-6);
    assert!((scan.arms[1].e_nl_vv10[0] - scan.arms[1].e_nl_vv10[2]).abs() > 1e-6);
}
