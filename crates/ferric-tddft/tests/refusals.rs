//! Fast (non-ignored) refusal tests for the user-facing `run_tddft`.
//!
//! The defect these pin (validation campaign F1): a KS reference used to run
//! WITHOUT the `(ia|f_xc|jb)` kernel term, print a warning, and return a
//! number. Every case below must now come back as an `Err` — never a number.
//!
//! Each refusal is asserted under a STARVED memory budget as well: if the
//! error were coming from the memory gate (or anything else after the
//! functional/spin gates), the message check would fail. That pins the order —
//! a refused functional must cost the user no integral work.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_tddft::{run_tddft, TddftConfig, TddftMethod};

/// 1 kB: any run that reached the memory gate would fail THERE, with a
/// "memory plan" message instead of the refusal under test.
const STARVED: usize = 1_000;

fn water_hf() -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    ferric_scf::ScfResult,
) {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    assert!(rhf.converged);
    (mol, obs, dfbs, rhf)
}

#[test]
fn meta_gga_reference_is_refused_not_approximated() {
    let (mol, obs, dfbs, rhf) = water_hf();
    for method in [TddftMethod::Tda, TddftMethod::Casida] {
        for xc in ["SCAN", "r2SCAN"] {
            let cfg = TddftConfig {
                n_roots: 3,
                method,
                memory_budget_bytes: Some(STARVED),
                xc: Some(xc.to_string()),
                ..Default::default()
            };
            let err = run_tddft(&mol, &obs, &dfbs, &rhf, &cfg)
                .expect_err("meta-GGA has no tau f_xc kernel; a number here is the F1 defect")
                .to_string();
            assert!(
                err.contains("meta-GGA"),
                "{method:?}/{xc}: refusal must name the reason (and must come from the \
                 functional gate, not the memory gate): {err}"
            );
        }
    }
}

#[test]
fn range_separated_and_vv10_references_are_refused() {
    let (mol, obs, dfbs, rhf) = water_hf();
    for xc in ["wB97X-V", "HYB_GGA_XC_CAM_B3LYP"] {
        let cfg = TddftConfig {
            xc: Some(xc.to_string()),
            memory_budget_bytes: Some(STARVED),
            ..Default::default()
        };
        let err = run_tddft(&mol, &obs, &dfbs, &rhf, &cfg)
            .expect_err("no complete kernel; must be refused")
            .to_string();
        assert!(
            !err.contains("memory plan"),
            "{xc}: refused by the memory gate, i.e. the functional gate never fired: {err}"
        );
    }
}

#[test]
fn c_hf_override_without_a_functional_is_refused() {
    let (mol, obs, dfbs, rhf) = water_hf();
    let cfg = TddftConfig {
        c_hf_override: Some(0.2),
        memory_budget_bytes: Some(STARVED),
        ..Default::default()
    };
    let err = run_tddft(&mol, &obs, &dfbs, &rhf, &cfg)
        .expect_err("c_hf = 0.2 on an HF reference is the old kernel-less hybrid number")
        .to_string();
    assert!(err.contains("c_hf"), "unhelpful refusal: {err}");
}

#[test]
fn open_shell_reference_is_refused_not_panicking() {
    // OH doublet. Before this fix `run_tddft` called `eps_r()`/`mos_r()`,
    // which ASSERT a restricted reference — an open-shell input panicked the
    // process instead of returning an error.
    let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let ctx = ParallelContext::default();
    let scf_cfg = RhfConfig {
        max_iter: 200,
        ..Default::default()
    };
    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &scf_cfg).unwrap();
    assert!(uhf.converged, "OH/STO-3G UHF did not converge");

    for xc in [None, Some("PBE".to_string())] {
        let cfg = TddftConfig {
            xc: xc.clone(),
            ..Default::default()
        };
        let err = run_tddft(&mol, &obs, &dfbs, &uhf, &cfg)
            .expect_err("open-shell reference must be refused")
            .to_string();
        assert!(
            err.contains("closed-shell"),
            "{xc:?}: refusal must say why: {err}"
        );
    }
}
