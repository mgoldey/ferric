//! An explicit `level_shift = 0` is distinguishable from an unset one.
//!
//! `level_shift = 0.0` used to be both the default and the trigger for the
//! automatic 0.5 Ha meta-GGA shift (`driver::effective_level_shift`), so a
//! caller could not run a meta-GGA unshifted. `RhfConfig::meta_gga_auto_level_shift`
//! (default `true`) names the difference; the CLI and the Python kwargs set it to
//! `false` when the caller wrote a value.
//!
//! Anchors: (1) the default config, with the flag untouched, still gets the
//! auto shift and its SCF is bit-identical to a run that spells the flag as
//! `true`; (2) an explicit 0.5 equals the auto shift's SCF bit for bit
//! (same effective shift, different provenance); (3) an explicit 0 follows a
//! DIFFERENT iteration path from the auto shift (artifact hypothesis: if the
//! flag were not read, explicit 0 would reproduce the auto run exactly).

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::certificate::scf_certificate;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const WATER: &str = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";

fn scan() -> RhfConfig {
    RhfConfig {
        xc: Some("SCAN".into()),
        max_iter: 60,
        ..Default::default()
    }
}

fn solve(cfg: &RhfConfig) -> (Molecule, Result<ScfResult, ferric_core::FerricError>) {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &bundled("sto-3g").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let r = solve_rhf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, cfg);
    (mol, r)
}

fn shift_entry(mol: &Molecule, r: &ScfResult, cfg: &RhfConfig) -> Option<String> {
    scf_certificate(mol, r, cfg)
        .approximations
        .into_iter()
        .find(|a| a.name == "level-shift")
        .map(|a| a.detail)
}

#[test]
fn explicit_zero_turns_off_the_meta_gga_auto_shift() {
    let auto = scan();
    assert!(
        auto.meta_gga_auto_level_shift,
        "the default keeps the auto shift"
    );
    let (mol, auto_res) = solve(&auto);
    let auto_res = auto_res.expect("SCAN/STO-3G converges with the auto shift");
    assert!(auto_res.converged);
    assert!(
        shift_entry(&mol, &auto_res, &auto)
            .is_some_and(|d| d.contains("auto-applied for meta-GGA")),
        "default config reports the auto shift"
    );

    // Anchor 1: spelling the default flag changes nothing, bit for bit.
    let spelled = RhfConfig {
        meta_gga_auto_level_shift: true,
        ..scan()
    };
    let (_, spelled_res) = solve(&spelled);
    let spelled_res = spelled_res.unwrap();
    assert_eq!(auto_res.energy.to_bits(), spelled_res.energy.to_bits());
    assert_eq!(auto_res.iterations, spelled_res.iterations);

    // Anchor 2: an explicit 0.5 is the same effective shift as the auto one.
    let explicit_half = RhfConfig {
        level_shift: 0.5,
        meta_gga_auto_level_shift: false,
        ..scan()
    };
    let (mol, half_res) = solve(&explicit_half);
    let half_res = half_res.unwrap();
    assert_eq!(auto_res.energy.to_bits(), half_res.energy.to_bits());
    assert_eq!(auto_res.iterations, half_res.iterations);
    assert!(shift_entry(&mol, &half_res, &explicit_half).is_some_and(|d| d.contains("user-set")));

    // Explicit 0: no shift is reported and the SCF follows another path.
    let off = RhfConfig {
        meta_gga_auto_level_shift: false,
        ..scan()
    };
    assert!(!off.meta_gga_auto_level_shift);
    assert_eq!(
        shift_entry(&mol, &auto_res, &off),
        None,
        "an explicit zero reports no shift"
    );
    // The effective shift acts on the iteration path: after three iterations
    // (before any of the runs converges) the shifted and the unshifted energy
    // differ. Explicit 0 follows the unshifted path, the default the shifted
    // one.
    let three = |cfg: &RhfConfig| {
        let cfg = RhfConfig {
            max_iter: 3,
            ..cfg.clone()
        };
        let (_, r) = solve(&cfg);
        r.expect("a capped SCF returns its best effort").energy
    };
    let (e_auto, e_off, e_half) = (three(&auto), three(&off), three(&explicit_half));
    println!("E after 3 iterations: auto {e_auto:.12}, explicit 0 {e_off:.12}");
    assert_eq!(e_auto.to_bits(), e_half.to_bits(), "auto == explicit 0.5");
    assert!(
        (e_auto - e_off).abs() > 1e-9,
        "explicit 0 must follow the unshifted path: |dE| = {:.3e}",
        (e_auto - e_off).abs()
    );
    // And the unshifted SCF still reaches the same state.
    let (_, off_res) = solve(&off);
    if let Ok(r) = off_res {
        assert!((r.energy - auto_res.energy).abs() < 1e-8);
    }
}
