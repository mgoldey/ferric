//! EXACTNESS ANCHOR for `terfc_pair::mp2_v_and_scs_2terfc` (the S66 shared-SCF
//! entry point, mgoldey/ferric#278).
//!
//! The SEPARATE path mirrors what the per-method bindings do: a fresh
//! `solve_rhf` for each method, then that method's library call. The SHARED
//! path solves the SCF once and calls the pair function. Every reported energy
//! must agree BITWISE: the SCF is deterministic for a fixed config, and each
//! half is the same library call on an identical reference.
//!
//! Artifact hypothesis: if the shared function threaded a config wrongly
//! (frozen core, r0 order, Coulomb control's RiMp2Config, the VV10 grid) or if
//! one SCF did not reproduce another, the bitwise compare fails. Water is used
//! because nocc = 5 gives a real same-spin space (H2 makes E_SS ≡ 0 and hides
//! the SCS same-spin difference; see scs.rs).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_vv10::{att_mp2_vv10, AttVv10Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::scs::{scs_mp2_2terfc, ScsMp2TerfcConfig};
use ferric_mp2::terfc_pair::mp2_v_and_scs_2terfc;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

const WATER: &str = "3\nwater\nO 0.000 0.000 0.118\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n";

fn tables_present() -> bool {
    if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR not set; terfc interpolation tables unavailable");
        return false;
    }
    true
}

/// A fresh SCF, exactly as every `run_*` binding does it (library-default
/// `RhfConfig`, Coulomb operator, Schwarz bounds rebuilt).
fn fresh_rhf(mol: &Molecule, obs: &PreparedBasis) -> ScfResult {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        mol,
        obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    assert!(rhf.converged);
    rhf
}

fn configs() -> (AttVv10Config, ScsMp2TerfcConfig) {
    let mp2v = AttVv10Config {
        // Small grid: the anchor is about threading, not VV10 grid accuracy.
        nlc_grid: AtomicGridConfig {
            n_radial: 20,
            n_angular: 26,
            prune: None,
        },
        frozen_core: 1,
        ..AttVv10Config::mp2_v_terfc_atz()
    };
    let scs = ScsMp2TerfcConfig {
        frozen_core: 1,
        ..ScsMp2TerfcConfig::default()
    };
    (mp2v, scs)
}

#[test]
fn shared_scf_matches_separate_per_method_calls_bitwise() {
    if !tables_present() {
        return;
    }
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let (mp2v_cfg, scs_cfg) = configs();

    // SEPARATE: one SCF per method, as the per-method bindings do.
    let sep_mp2v = att_mp2_vv10(&mol, &obs, &bs, &dfbs, &fresh_rhf(&mol, &obs), &mp2v_cfg).unwrap();
    let sep_scs = scs_mp2_2terfc(&mol, &obs, &dfbs, &fresh_rhf(&mol, &obs), &scs_cfg).unwrap();
    let rimp2_cfg = RiMp2Config {
        frozen_core: 1,
        ..Default::default()
    };
    let sep_mp2 = ri_mp2(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &fresh_rhf(&mol, &obs),
        &rimp2_cfg,
    )
    .unwrap();

    // SHARED: one SCF.
    let rhf = fresh_rhf(&mol, &obs);
    let shared =
        mp2_v_and_scs_2terfc(&mol, &obs, &bs, &dfbs, &rhf, &mp2v_cfg, &scs_cfg, true).unwrap();
    let mp2 = shared.mp2_coulomb.as_ref().unwrap();

    let pairs = [
        ("e_hf", shared.e_hf, sep_mp2v.e_hf),
        ("mp2v total", shared.mp2_v.total, sep_mp2v.total),
        (
            "mp2v e_c_att",
            shared.mp2_v.e_c_att_mp2,
            sep_mp2v.e_c_att_mp2,
        ),
        ("mp2v e_nl", shared.mp2_v.e_nl_vv10, sep_mp2v.e_nl_vv10),
        (
            "scs total",
            shared.scs_2terfc.total_energy,
            sep_scs.total_energy,
        ),
        ("scs corr", shared.scs_2terfc.scs_corr, sep_scs.scs_corr),
        ("scs e_os", shared.scs_2terfc.e_os, sep_scs.e_os),
        ("scs e_ss", shared.scs_2terfc.e_ss, sep_scs.e_ss),
        ("mp2 total", mp2.total_energy, sep_mp2.total_energy),
        ("mp2 corr", mp2.mp2_corr, sep_mp2.mp2_corr),
    ];
    for (name, a, b) in pairs {
        eprintln!(
            "{name:12}: shared {a:.15} separate {b:.15} diff {:+.3e}",
            a - b
        );
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{name}: shared {a:.17e} != separate {b:.17e}"
        );
    }
    // Non-vacuity: the three methods are genuinely different numbers, and the
    // same-spin difference SCS relies on is non-zero.
    assert!((shared.mp2_v.total - shared.scs_2terfc.total_energy).abs() > 1e-4);
    assert!((mp2.total_energy - shared.scs_2terfc.total_energy).abs() > 1e-4);
    assert!(shared.scs_2terfc.e_ss.abs() > 1e-6);
    assert!(shared.timings.mp2_v_s > 0.0 && shared.timings.scs_2terfc_s > 0.0);
}

#[test]
fn mismatched_frozen_core_is_a_hard_error() {
    if !tables_present() {
        return;
    }
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let rhf = fresh_rhf(&mol, &obs);
    let (mp2v_cfg, mut scs_cfg) = configs();
    scs_cfg.frozen_core = 0;
    let err = mp2_v_and_scs_2terfc(&mol, &obs, &bs, &dfbs, &rhf, &mp2v_cfg, &scs_cfg, false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("frozen_core differs"), "{err}");
}
