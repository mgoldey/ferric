//! EXACTNESS ANCHOR + guards for the DECOUPLED MP2-V evaluation used by the
//! full refit (`att_mp2_vv10_decoupled_scan`, `AttVv10Config::vv10_decoupling`).
//!
//! * every (MP2 arm, VV10 tuple) of the scan reproduces a separate
//!   `att_mp2_vv10` call bit for bit (attMP2 half and VV10 half);
//! * the anchor is not vacuous: swapping the two seams (MP2 seam <-> VV10
//!   seam) changes BOTH halves, so a scan that mixed them up fails;
//! * decoupling to the SAME seam reproduces the lockstep (published) path
//!   bit for bit, i.e. lockstep is a special case and the default is untouched;
//! * the lockstep guard still fires, and decoupled-only misuse is a hard error.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::vv10::{Vv10Damping, Vv10Form};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_vv10::{
    att_mp2_vv10, att_mp2_vv10_decoupled_scan, AttVv10Config, AttVv10ScanArm, Vv10Decoupling,
    Vv10Tuple, BOHR_PER_ANG,
};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

struct Rig {
    mol: Molecule,
    bs_obs: basis::BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
    base: AttVv10Config,
}

fn rig() -> Option<Rig> {
    if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR not set; terfc interpolation tables unavailable");
        return None;
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
    Some(Rig {
        mol,
        bs_obs,
        obs,
        dfbs,
        rhf,
        base,
    })
}

/// A single full call with the arm's MP2 seam and the tuple's VV10 parameters.
fn full_call(r: &Rig, arm: &AttVv10ScanArm, t: &Vv10Tuple) -> ferric_mp2::att_vv10::AttVv10Result {
    let mut cfg = r.base.clone();
    cfg.r0_bohr = arm.r0_bohr;
    cfg.omega = arm.omega;
    cfg.vv10.b = t.b;
    cfg.vv10.c = t.c;
    cfg.vv10_decoupling = Some(t.damping);
    att_mp2_vv10(&r.mol, &r.obs, &r.bs_obs, &r.dfbs, &r.rhf, &cfg).unwrap()
}

fn tuples(r0: f64) -> Vec<Vv10Tuple> {
    let dec = |form, r0_bohr, omega| Vv10Decoupling {
        form,
        r0_bohr,
        omega,
    };
    vec![
        // published-shaped: Eq. 11, linked seam
        Vv10Tuple {
            b: 11.0,
            c: 0.0089,
            damping: dec(Vv10Form::Eq11, r0, None),
        },
        // different form, own sharp seam, own b and C
        Vv10Tuple {
            b: 8.0,
            c: 0.02,
            damping: dec(Vv10Form::Terf, 0.9 * r0, Some(4.0 / (0.9 * r0))),
        },
        // bare VV10 (no seam)
        Vv10Tuple {
            b: 11.0,
            c: 0.0089,
            damping: dec(Vv10Form::Bare, 0.0, None),
        },
        // Eq. 11 at a different r0_vv and sharpness
        Vv10Tuple {
            b: 14.5,
            c: 0.0089,
            damping: dec(Vv10Form::Eq11, 1.2 * r0, Some(2.0 / (1.2 * r0))),
        },
        // exact duplicate of tuple 0 (exercises kernel/weight dedupe)
        Vv10Tuple {
            b: 11.0,
            c: 0.0089,
            damping: dec(Vv10Form::Eq11, r0, None),
        },
    ]
}

#[test]
fn decoupled_scan_matches_all_in_one_for_every_arm_and_tuple() {
    let Some(r) = rig() else { return };
    let r0 = BOHR_PER_ANG;
    let arms = [
        AttVv10ScanArm {
            r0_bohr: r0,
            omega: None,
        },
        AttVv10ScanArm {
            r0_bohr: 1.1 * r0,
            omega: Some(4.0 / (1.1 * r0)),
        },
    ];
    let ts = tuples(r0);
    let scan = match att_mp2_vv10_decoupled_scan(
        &r.mol, &r.obs, &r.bs_obs, &r.dfbs, &r.rhf, &r.base, &arms, &ts, true,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: terfc path unavailable at runtime: {e}");
            return;
        }
    };
    assert_eq!(scan.e_nl_vv10.len(), ts.len());
    // Every tuple against arm 0; tuple 1 (own seam, own C) also against arm 1.
    for (a, t) in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 1)] {
        let one = full_call(&r, &arms[a], &ts[t]);
        assert_eq!(
            scan.arms[a].e_c_att_mp2.to_bits(),
            one.e_c_att_mp2.to_bits(),
            "attMP2 half arm {a}"
        );
        assert_eq!(
            scan.e_nl_vv10[t].to_bits(),
            one.e_nl_vv10.to_bits(),
            "VV10 half tuple {t} (arm {a})"
        );
    }
    // E_nl does not depend on the MP2 arm: the duplicate tuple is identical.
    assert_eq!(scan.e_nl_vv10[0].to_bits(), scan.e_nl_vv10[4].to_bits());
    // Not vacuous: the tuples really differ, and so do the arms.
    for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2)] {
        assert!(
            (scan.e_nl_vv10[i] - scan.e_nl_vv10[j]).abs() > 1e-6,
            "tuples {i} and {j} too close"
        );
    }
    assert!((scan.arms[0].e_c_att_mp2 - scan.arms[1].e_c_att_mp2).abs() > 1e-6);
}

/// MUTATION GUARD (artifact hypothesis: "the scan pairs the seams wrongly").
/// Swapping the MP2 seam and the VV10 seam must change both halves; the
/// anchor above compares bitwise, so a scan that fed the arm's seam to the
/// VV10 damping (or the tuple's to the MP2 operator) cannot pass it.
#[test]
fn swapping_the_two_seams_changes_both_halves() {
    let Some(r) = rig() else { return };
    let r0 = BOHR_PER_ANG;
    let arm = AttVv10ScanArm {
        r0_bohr: r0,
        omega: None,
    };
    let t = Vv10Tuple {
        b: 8.0,
        c: 0.02,
        damping: Vv10Decoupling {
            form: Vv10Form::Eq11,
            r0_bohr: 0.9 * r0,
            omega: Some(4.0 / (0.9 * r0)),
        },
    };
    let scan = match att_mp2_vv10_decoupled_scan(
        &r.mol,
        &r.obs,
        &r.bs_obs,
        &r.dfbs,
        &r.rhf,
        &r.base,
        &[arm],
        &[t],
        false,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: terfc path unavailable at runtime: {e}");
            return;
        }
    };
    // The swapped evaluation: MP2 takes the tuple's seam, VV10 the arm's.
    let swapped_arm = AttVv10ScanArm {
        r0_bohr: t.damping.r0_bohr,
        omega: t.damping.omega,
    };
    let swapped_t = Vv10Tuple {
        damping: Vv10Decoupling {
            form: Vv10Form::Eq11,
            r0_bohr: arm.r0_bohr,
            omega: arm.omega,
        },
        ..t
    };
    let sw = full_call(&r, &swapped_arm, &swapped_t);
    assert_ne!(scan.arms[0].e_c_att_mp2.to_bits(), sw.e_c_att_mp2.to_bits());
    assert_ne!(scan.e_nl_vv10[0].to_bits(), sw.e_nl_vv10.to_bits());
    assert!((scan.arms[0].e_c_att_mp2 - sw.e_c_att_mp2).abs() > 1e-6);
    assert!((scan.e_nl_vv10[0] - sw.e_nl_vv10).abs() > 1e-6);
}

/// Lockstep is a special case: decoupling to the SAME (r0, omega) as the MP2
/// operator is bit-identical to the published guarded path (and that path is
/// itself unchanged by the new `vv10_decoupling: None` default).
#[test]
fn decoupling_to_the_same_seam_reproduces_the_lockstep_path() {
    let Some(r) = rig() else { return };
    let r0 = BOHR_PER_ANG;
    for omega in [None, Some(4.0 / r0)] {
        let mut lock = r.base.clone();
        lock.r0_bohr = r0;
        lock.omega = omega;
        lock.vv10_damping = Vv10Damping::Terfc {
            r0_bohr: r0,
            omega_bohr_inv: None,
        };
        let mut dec = lock.clone();
        dec.vv10_decoupling = Some(Vv10Decoupling {
            form: Vv10Form::Eq11,
            r0_bohr: r0,
            omega,
        });
        let a = match att_mp2_vv10(&r.mol, &r.obs, &r.bs_obs, &r.dfbs, &r.rhf, &lock) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("SKIP: terfc path unavailable at runtime: {e}");
                return;
            }
        };
        let b = att_mp2_vv10(&r.mol, &r.obs, &r.bs_obs, &r.dfbs, &r.rhf, &dec).unwrap();
        assert_eq!(
            a.e_nl_vv10.to_bits(),
            b.e_nl_vv10.to_bits(),
            "omega {omega:?}"
        );
        assert_eq!(a.total.to_bits(), b.total.to_bits());
    }
}

#[test]
fn guards_lockstep_still_errors_and_decoupled_misuse_is_rejected() {
    let Some(r) = rig() else { return };
    let r0 = BOHR_PER_ANG;
    // Lockstep default: a damping omega that contradicts config.omega is a hard error.
    let mut lock = r.base.clone();
    lock.omega = None;
    lock.vv10_damping = Vv10Damping::Terfc {
        r0_bohr: r0,
        omega_bohr_inv: Some(3.0),
    };
    assert!(lock.effective_vv10_damping().is_err());
    assert!(lock.effective_vv10_weight().is_err());
    // Decoupled mode: the superseded damping may not carry its own omega.
    let mut both = lock.clone();
    both.vv10_decoupling = Some(Vv10Decoupling {
        form: Vv10Form::Eq11,
        r0_bohr: r0,
        omega: None,
    });
    assert!(both.effective_vv10_weight().is_err());
    // Bare with an omega, non-positive seam values.
    for bad in [
        Vv10Decoupling {
            form: Vv10Form::Bare,
            r0_bohr: 0.0,
            omega: Some(1.0),
        },
        Vv10Decoupling {
            form: Vv10Form::Terf,
            r0_bohr: -1.0,
            omega: None,
        },
        Vv10Decoupling {
            form: Vv10Form::Eq11,
            r0_bohr: r0,
            omega: Some(0.0),
        },
    ] {
        let mut c = r.base.clone();
        c.vv10_decoupling = Some(bad);
        assert!(c.effective_vv10_weight().is_err(), "{bad:?}");
    }
    // Scan-level validation of b and C.
    let bad_t = Vv10Tuple {
        b: -1.0,
        c: 0.0089,
        damping: Vv10Decoupling {
            form: Vv10Form::Bare,
            r0_bohr: 0.0,
            omega: None,
        },
    };
    let e = att_mp2_vv10_decoupled_scan(
        &r.mol,
        &r.obs,
        &r.bs_obs,
        &r.dfbs,
        &r.rhf,
        &r.base,
        &[],
        &[bad_t],
        false,
    );
    assert!(e.is_err());
}

/// EXPERIMENTAL `Mp2vSession`: every call must equal the one-shot
/// `att_mp2_vv10` bit for bit (lockstep totals at several seams/b/C, and the
/// decoupled tuples), and repeated calls on the held state must be repeatable.
#[test]
fn session_matches_one_shot_bitwise_and_is_repeatable() {
    let Some(r) = rig() else { return };
    let Some(r2) = rig() else { return };
    let sess = ferric_mp2::att_vv10::Mp2vSession::new(
        r2.mol,
        r2.bs_obs,
        r2.obs,
        r2.dfbs,
        r2.rhf,
        r2.base.clone(),
    )
    .unwrap();
    assert!(sess.memory_bytes_estimate() > 0);
    let r0 = BOHR_PER_ANG;
    let arms = [
        AttVv10ScanArm {
            r0_bohr: r0,
            omega: None,
        },
        AttVv10ScanArm {
            r0_bohr: 0.9 * r0,
            omega: Some(4.0 / (0.9 * r0)),
        },
    ];
    for (arm, b, c) in [
        (&arms[0], 11.0, 0.0089),
        (&arms[1], 8.0, 0.02),
        (&arms[1], 14.5, 0.0089),
    ] {
        let (hf, ec, en) = match sess.total(arm, b, c) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("SKIP: terfc path unavailable at runtime: {e}");
                return;
            }
        };
        // One-shot LOCKSTEP path (no decoupling field): the published guarded code.
        let mut cfg = r.base.clone();
        cfg.r0_bohr = arm.r0_bohr;
        cfg.omega = arm.omega;
        cfg.vv10.b = b;
        cfg.vv10.c = c;
        cfg.vv10_damping = Vv10Damping::Terfc {
            r0_bohr: arm.r0_bohr,
            omega_bohr_inv: None,
        };
        let one = att_mp2_vv10(&r.mol, &r.obs, &r.bs_obs, &r.dfbs, &r.rhf, &cfg).unwrap();
        assert_eq!(hf.to_bits(), one.e_hf.to_bits());
        assert_eq!(ec.to_bits(), one.e_c_att_mp2.to_bits(), "attMP2 b={b}");
        assert_eq!(en.to_bits(), one.e_nl_vv10.to_bits(), "VV10 b={b} c={c}");
        assert_eq!((hf + ec + en).to_bits(), one.total.to_bits());
    }
    // Decoupled tuples through the session == the decoupled one-shot.
    let ts = tuples(r0);
    let e = sess.vv10(&ts).unwrap();
    for (t, tup) in ts.iter().enumerate() {
        let one = full_call(&r, &arms[0], tup);
        assert_eq!(e[t].to_bits(), one.e_nl_vv10.to_bits(), "tuple {t}");
    }
    // Repeatability on the held state (no hidden mutation).
    let e2 = sess.vv10(&ts).unwrap();
    assert_eq!(
        e.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        e2.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    let m1 = sess.att_mp2(&arms[1]).unwrap().e_c_att_mp2;
    let m2 = sess.att_mp2(&arms[1]).unwrap().e_c_att_mp2;
    assert_eq!(m1.to_bits(), m2.to_bits());
    // Not vacuous: the two arms differ.
    assert!((sess.att_mp2(&arms[0]).unwrap().e_c_att_mp2 - m1).abs() > 1e-6);
    // Invalid input is a hard error.
    assert!(sess
        .vv10(&[Vv10Tuple {
            b: 0.0,
            c: 0.0089,
            damping: ts[0].damping
        }])
        .is_err());
}
