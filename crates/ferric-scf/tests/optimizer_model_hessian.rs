//! Lindh (1995) model Hessian and the ratio trust-radius rule.
//!
//! # Exactness anchor
//!
//! `initial_hessian = Some(Simple)` with `trust_update = Fixed` is the search
//! that existed before both options. [`RECORDED`] holds every evaluated energy
//! of six such optimizations, captured with this file's runner on commit
//! 842f9716d before `optimize.rs` was changed. The step count is asserted
//! exactly. The energies are asserted bitwise when `FERRIC_BITWISE_ANCHOR=1`
//! (the capture box) and to 1e-12 Ha otherwise: the same GEMM dispatch caveat
//! as `internal_coord_optimize.rs` (CI reproduced the H2 anchor to 5 ulp,
//! 1.1e-15 Ha).
//!
//! # Hypotheses
//!
//! Physics: a model Hessian that separates stiff stretches from soft torsions,
//! and a trust radius that grows on good steps, cut the step count on floppy
//! molecules. Artifact: a rejected step that is still counted as a move, or a
//! predicted change with the wrong sign, would show up as accepted uphill steps
//! or as a radius that never grows; [`ratio_rule_never_keeps_an_uphill_step`]
//! and the `ratio_trust_update` unit table target those.

use ferric_core::model_hessian::lindh_cartesian_hessian;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::optimize::{
    optimize_coordinates, optimize_geometry, CoordSystem, InitialHessian, OptimizeConfig,
    OptimizeResult, TrustUpdate, TRUST_ENERGY_NOISE,
};
use ferric_scf::rhf::RhfConfig;

fn rhf_cfg() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-10,
        ..Default::default()
    }
}

fn water_distorted() -> Molecule {
    Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n", 0, 1).unwrap()
}

fn h2o2_distorted() -> Molecule {
    Molecule::parse_xyz(
        "4\nH2O2\nO  0.000  0.734  -0.055\nO  0.000 -0.734  -0.055\n\
         H  0.839  0.885   0.435\nH -0.839 -0.885   0.435\n",
        0,
        1,
    )
    .unwrap()
}

fn ethane_distorted() -> Molecule {
    Molecule::parse_xyz(
        "8\nethane\n\
         C  0.000  0.000  0.800\nC  0.000  0.000 -0.800\n\
         H  1.050  0.000  1.180\nH -0.525  0.909  1.180\nH -0.525 -0.909  1.180\n\
         H  1.000  0.150 -1.180\nH -0.575  0.840 -1.180\nH -0.425 -0.990 -1.180\n",
        0,
        1,
    )
    .unwrap()
}

fn molecule(tag: &str) -> Molecule {
    match tag {
        "h2o" => water_distorted(),
        "h2o2" => h2o2_distorted(),
        "ethane" => ethane_distorted(),
        _ => unreachable!(),
    }
}

fn coords(tag: &str) -> CoordSystem {
    match tag {
        "cart" => CoordSystem::Cartesian,
        "int" => CoordSystem::RedundantInternal,
        _ => unreachable!(),
    }
}

fn run(mol: &Molecule, cfg: &OptimizeConfig) -> OptimizeResult {
    let ctx = ParallelContext::default();
    optimize_geometry(&ctx, mol, "sto-3g", Operator::coulomb(), &rhf_cfg(), cfg).unwrap()
}

fn cfg(cs: CoordSystem, ih: InitialHessian, tu: TrustUpdate) -> OptimizeConfig {
    OptimizeConfig {
        trust_radius: 0.1,
        coord_system: cs,
        initial_hessian: Some(ih),
        trust_update: tu,
        ..Default::default()
    }
}

/// `(system, coordinates, steps, energy bits of every evaluation)`, RHF/STO-3G,
/// trust radius 0.1, captured on 842f9716d (see the module doc).
const RECORDED: &[(&str, &str, usize, &[u64])] = &[
    (
        "h2o",
        "cart",
        7,
        &[
            0xc052bc6536494687,
            0xc052bd8658366454,
            0xc052bdced31d1bed,
            0xc052bdd0ea3972a5,
            0xc052bdd121d1cb79,
            0xc052bdd14d248f7c,
            0xc052bdd153184bc3,
            0xc052bdd153563e07,
        ],
    ),
    (
        "h2o",
        "int",
        5,
        &[
            0xc052bc6536494687,
            0xc052bd5d6fa230d5,
            0xc052bdb87c5c7ea2,
            0xc052bdd101fda83a,
            0xc052bdd152ae7f9d,
            0xc052bdd153559dcc,
        ],
    ),
    (
        "h2o2",
        "cart",
        6,
        &[
            0xc062984471a5f321,
            0xc06298793295dcc2,
            0xc062987a776663a8,
            0xc062987a975cac67,
            0xc062987ab5b3eb05,
            0xc062987abed17f8f,
            0xc062987abfc99dbd,
        ],
    ),
    (
        "h2o2",
        "int",
        9,
        &[
            0xc062984471a5f321,
            0xc0629872c5f85643,
            0xc0629878c8d8f902,
            0xc062987989bb211a,
            0xc062987a93c7835d,
            0xc062987aaed43a0b,
            0xc062987acadae2a0,
            0xc062987ad39220e5,
            0xc062987ad9dbf0df,
            0xc062987ada3fc2c8,
        ],
    ),
    (
        "ethane",
        "cart",
        44,
        &[
            0xc05392d200935c46,
            0xc0539319faa99b98,
            0xc0539342bda9e8bc,
            0xc053934ba86e85c2,
            0xc053934d615db84e,
            0xc053934eed317603,
            0xc053934f8c4254a0,
            0xc053934fba3575b1,
            0xc053934fcd188ea4,
            0xc053934fe60ba7a5,
            0xc053935002be9ae0,
            0xc05393501d05c3cf,
            0xc053935035b9d89a,
            0xc05393505d5ac84b,
            0xc0539350b9325d1a,
            0xc0539351b00f4786,
            0xc0539352c420dad2,
            0xc0539353ecddbefd,
            0xc05393551d020280,
            0xc05393565195057a,
            0xc05393579a704178,
            0xc053935925112db4,
            0xc053935b3cd9dcb6,
            0xc053935e39c20f1c,
            0xc0539362656fbe0c,
            0xc0539367cc5d0cee,
            0xc053936e18779b42,
            0xc0539374d81665d2,
            0xc053937ba34e121a,
            0xc05393821527bd36,
            0xc0539387e9a3f13a,
            0xc053938cc4b22600,
            0xc0539390f125d351,
            0xc0539392c043d2e2,
            0xc0539394b5414684,
            0xc0539396341d1360,
            0xc05393976fad818e,
            0xc0539397fd632110,
            0xc053939827b55ee0,
            0xc053939845edb0be,
            0xc05393985507cba6,
            0xc05393985c8d638c,
            0xc053939863492022,
            0xc0539398698cb5f0,
            0xc05393986d983dec,
        ],
    ),
    (
        "ethane",
        "int",
        19,
        &[
            0xc05392d200935c46,
            0xc05393263c812dca,
            0xc05393440e641707,
            0xc05393525bdce0dc,
            0xc0539349ba6c39b4,
            0xc053935b7ee99dda,
            0xc053936991af0345,
            0xc0539366573677dd,
            0xc053937bc8875686,
            0xc05393830e9d6226,
            0xc05393863d7f091a,
            0xc053939021e0db2c,
            0xc053939719fe8a6f,
            0xc0539395750d9b6e,
            0xc053939275e6717d,
            0xc05393920a156b0f,
            0xc0539397f8175a12,
            0xc0539398653d85f2,
            0xc0539398705e94c4,
            0xc053939872613af6,
        ],
    ),
];

/// Energies may differ from the capture box by GEMM rounding; 1e-12 Ha is six
/// orders below `e_conv` and about 1e3 times the measured 1.1e-15 Ha spread.
const TRACE_BAND: f64 = 1e-12;

#[test]
fn simple_hessian_and_fixed_trust_reproduce_the_recorded_trajectories() {
    let bitwise = std::env::var("FERRIC_BITWISE_ANCHOR").is_ok_and(|v| v == "1");
    for &(tag, ctag, steps, trace) in RECORDED {
        let r = run(
            &molecule(tag),
            &cfg(coords(ctag), InitialHessian::Simple, TrustUpdate::Fixed),
        );
        assert!(r.converged, "{tag} {ctag}");
        assert_eq!(r.steps, steps, "{tag} {ctag}: step count");
        assert_eq!(r.rejected_steps, 0, "{tag} {ctag}");
        assert_eq!(
            r.energy_trace.len(),
            trace.len(),
            "{tag} {ctag}: evaluations"
        );
        for (k, (&e, &bits)) in r.energy_trace.iter().zip(trace).enumerate() {
            let want = f64::from_bits(bits);
            if bitwise {
                assert_eq!(
                    e.to_bits(),
                    bits,
                    "{tag} {ctag} evaluation {k}: {e:.17} vs {want:.17}"
                );
            } else {
                assert!(
                    (e - want).abs() < TRACE_BAND,
                    "{tag} {ctag} evaluation {k}: {e:.17} vs {want:.17}"
                );
            }
        }
        assert_eq!(r.energy.to_bits(), r.energy_trace.last().unwrap().to_bits());
    }
}

/// The Rust Lindh Hessian equals the brute-force numpy sum of
/// `scripts/lindh_hessian_sympy.py` (no neighbour-list screening there), for
/// water and ethanol. 1e-12 Ha/Bohr²: the entries are O(1) sums of a few
/// hundred terms, so rounding is ~1e-15.
#[test]
fn lindh_hessian_matches_the_numpy_reference() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/lindh_model_hessian.json"
    );
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for name in ["water", "ethanol"] {
        let sys = &v["systems"][name];
        let atoms = sys["atoms"].as_array().unwrap();
        let mut xyz = format!("{}\n{name}\n", atoms.len());
        for a in atoms {
            let z = a[0].as_i64().unwrap() as i32;
            let p: Vec<f64> = a[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let sym = match z {
                1 => "H",
                6 => "C",
                8 => "O",
                _ => panic!("reference has Z = {z}"),
            };
            xyz.push_str(&format!("{sym} {:.17} {:.17} {:.17}\n", p[0], p[1], p[2]));
        }
        let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
        let h = lindh_cartesian_hessian(&mol).unwrap();
        let reference = sys["hessian"].as_array().unwrap();
        let mut worst = 0.0f64;
        for (i, row) in reference.iter().enumerate() {
            for (j, x) in row.as_array().unwrap().iter().enumerate() {
                worst = worst.max((h[(i, j)] - x.as_f64().unwrap()).abs());
            }
        }
        assert!(worst < 1e-12, "{name}: max |H_rust - H_numpy| = {worst:e}");
    }
}

/// Under the ratio rule no accepted point is higher than the one before it
/// (beyond the noise band), whatever the initial Hessian or coordinates. The
/// trace holds every evaluation; an evaluation above the last kept energy must
/// be one the optimizer counted as rejected, and the returned energy is the
/// last kept one. Ethane in internals does take uphill trial steps under this
/// rule (6 with Simple, 5 with Lindh when this test was written), so the count
/// is asserted non-zero there: a search that keeps every step fails.
#[test]
fn ratio_rule_never_keeps_an_uphill_step() {
    for tag in ["h2o", "h2o2", "ethane"] {
        for cs in [CoordSystem::Cartesian, CoordSystem::RedundantInternal] {
            for ih in [InitialHessian::Simple, InitialHessian::Lindh] {
                let r = run(&molecule(tag), &cfg(cs, ih, TrustUpdate::Ratio));
                assert!(r.converged, "{tag} {cs:?} {ih:?}");
                let mut kept = r.energy_trace[0];
                let mut uphill = 0;
                for &e in &r.energy_trace[1..] {
                    if e < kept + TRUST_ENERGY_NOISE {
                        kept = e;
                    } else {
                        uphill += 1;
                    }
                }
                assert_eq!(
                    uphill, r.rejected_steps,
                    "{tag} {cs:?} {ih:?}: {uphill} uphill evaluations, {} rejected",
                    r.rejected_steps
                );
                assert_eq!(
                    kept.to_bits(),
                    r.energy.to_bits(),
                    "{tag} {cs:?} {ih:?}: the returned energy is not the last kept one"
                );
                if tag == "ethane" && cs == CoordSystem::RedundantInternal {
                    assert!(uphill > 0, "{tag} {cs:?} {ih:?}: no uphill trial step");
                }
                eprintln!(
                    "[ratio] {tag:6} {cs:?} {ih:?}: steps {} rejected {} E {:.10}",
                    r.steps, r.rejected_steps, r.energy
                );
            }
        }
    }
}

/// Every option combination reaches the same minimum as a tight reference.
/// Upper bar: near the minimum the excess energy is about
/// ½ gᵀ H⁻¹ g ≤ ‖g‖² / (2 λ_min). The convergence test caps the RMS gradient
/// at 3e-4 Ha/Bohr, so ‖g‖² ≤ 3N (3e-4)² = 1.1e-6 for H2O2 (N = 4, the
/// largest soft case); with λ_min ≥ 0.01 Ha/Bohr² (the HOOH torsion) that is
/// 5.4e-5 Ha, and 1e-4 sits above it. The worst measured case is 1.3e-5
/// (H2O2, Cartesian, simple), and the bar is a hundred times below the 1e-2 Ha
/// between H2O2's cis and trans stationary points. Lower bar: the tight
/// reference (gradient 1e-6) is within ~1e-9 Ha of the minimum, so no run can
/// end more than SCF noise below it; -1e-8 allows for that.
#[test]
fn every_option_reaches_the_reference_minimum() {
    for tag in ["h2o", "h2o2", "ethane"] {
        let tight = OptimizeConfig {
            g_max_thresh: 1e-6,
            g_rms_thresh: 1e-6,
            e_conv: 1e-10,
            max_steps: 400,
            ..cfg(
                CoordSystem::RedundantInternal,
                InitialHessian::Lindh,
                TrustUpdate::Ratio,
            )
        };
        let reference = run(&molecule(tag), &tight);
        assert!(reference.converged, "{tag}: tight reference");
        for cs in [CoordSystem::Cartesian, CoordSystem::RedundantInternal] {
            for ih in [InitialHessian::Simple, InitialHessian::Lindh] {
                for tu in [TrustUpdate::Fixed, TrustUpdate::Ratio] {
                    let r = run(&molecule(tag), &cfg(cs, ih, tu));
                    let de = r.energy - reference.energy;
                    assert!(
                        r.converged && (-1e-8..1e-4).contains(&de),
                        "{tag} {cs:?} {ih:?} {tu:?}: E - E_ref = {de:e}, converged {}",
                        r.converged
                    );
                }
            }
        }
    }
}

/// `optimize_coordinates` has no atoms: an explicit Lindh request is refused,
/// never silently replaced by the identity.
#[test]
fn coordinate_only_optimizer_refuses_lindh() {
    let c = OptimizeConfig {
        initial_hessian: Some(InitialHessian::Lindh),
        ..Default::default()
    };
    let err = optimize_coordinates(&[0.1, 0.0, 0.0], &c, |x| {
        Ok((x[0] * x[0], vec![2.0 * x[0], 0.0, 0.0]))
    })
    .unwrap_err();
    assert!(err.to_string().contains("lindh"), "{err}");
}
