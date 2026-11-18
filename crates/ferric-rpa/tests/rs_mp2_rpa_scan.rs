//! `rs_mp2_lr_rpa_scan` anchors.
//!
//! # What is claimed
//!
//! Each (ω, formulation) entry of a scan equals a separate single-ω
//! `rs_mp2_lr_rpa` call BIT FOR BIT on every field of `RsMp2RpaResult`.
//! The scan changes no equation: the Coulomb spin components and the Coulomb
//! dRPA do not depend on ω, and each ω's erf/erfc intermediates are built
//! fresh (see
//! `book/src/26-rs-mp2-rpa/06-shared-intermediates.md#shared-intermediates-transform-per-operator-omega`).
//!
//! # Bar
//!
//! Exact (`to_bits` equality), not a tolerance: every entry is produced by the
//! same operations on the same inputs in the same order as a one-point call,
//! so any difference is contamination between ω points or formulations. The
//! single-ω entry point is itself the one-point scan, so the independent
//! witness for the pre-scan code is the CLI comparison against a base build
//! recorded in the commit report, not this file.
//!
//! # Mutation
//!
//! Reusing the previous ω's erf intermediates for the next ω (or dropping T's
//! Coulomb dRPA) must make these fail; see the commit report.
//!
//! Systems: water and the S22 water dimer at aug-cc-pVDZ (aug-cc-pvdz-rifit),
//! ω ∈ {0.2, 0.42, 0.673} Å⁻¹. The Terf case needs the interpolation tables
//! (`FERRIC_TERF_TABLE_DIR`) and skips with a note without them.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::rs_mp2_rpa::{
    rs_mp2_lr_rpa, rs_mp2_lr_rpa_scan, scan_peak_bytes, Attenuator, RsMp2RpaConfig,
    RsMp2RpaFormulation, RsMp2RpaResult,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

const ANG_INV: f64 = ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV;
const OMEGAS_ANG: [f64; 3] = [0.2, 0.42, 0.673];
const BOTH: [RsMp2RpaFormulation; 2] = [
    RsMp2RpaFormulation::DeltaLr,
    RsMp2RpaFormulation::CoupledRings,
];

const WATER: &str = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn setup(xyz: &str) -> Sys {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvdz-rifit").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig {
            max_iter: 400,
            energy_conv: 1e-11,
            density_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(rhf.converged);
    Sys {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

fn dimer_xyz() -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/molecules/s22/water_dimer.xyz");
    std::fs::read_to_string(p).unwrap()
}

fn bits(x: f64) -> u64 {
    x.to_bits()
}

fn bits_opt(x: Option<f64>) -> Option<u64> {
    x.map(f64::to_bits)
}

fn assert_bitwise(ctx: &str, a: &RsMp2RpaResult, b: &RsMp2RpaResult) {
    assert_eq!(bits(a.e_mp2_full), bits(b.e_mp2_full), "{ctx} e_mp2_full");
    assert_eq!(bits(a.e_sr_mp2), bits(b.e_sr_mp2), "{ctx} e_sr_mp2");
    assert_eq!(bits(a.e_lr_mp2), bits(b.e_lr_mp2), "{ctx} e_lr_mp2");
    assert_eq!(bits(a.e_dmp2_lr), bits(b.e_dmp2_lr), "{ctx} e_dmp2_lr");
    assert_eq!(
        bits_opt(a.e_drpa_lr),
        bits_opt(b.e_drpa_lr),
        "{ctx} e_drpa_lr"
    );
    assert_eq!(
        bits_opt(a.e_corr_naive),
        bits_opt(b.e_corr_naive),
        "{ctx} e_corr_naive"
    );
    assert_eq!(
        bits_opt(a.e_delta_drpa_full),
        bits_opt(b.e_delta_drpa_full),
        "{ctx} e_delta_drpa_full"
    );
    assert_eq!(
        bits_opt(a.e_delta_drpa_sr),
        bits_opt(b.e_delta_drpa_sr),
        "{ctx} e_delta_drpa_sr"
    );
    assert_eq!(bits(a.e_corr), bits(b.e_corr), "{ctx} e_corr");
    assert_eq!(bits(a.total_energy), bits(b.total_energy), "{ctx} total");
}

fn scan_matches_singles(sys: &Sys, base: &RsMp2RpaConfig, omegas_bohr: &[f64], label: &str) {
    let scan = rs_mp2_lr_rpa_scan(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        &sys.rhf,
        base,
        omegas_bohr,
        &BOTH,
    )
    .unwrap();
    assert_eq!(scan.len(), omegas_bohr.len() * BOTH.len());
    for (i, &w) in omegas_bohr.iter().enumerate() {
        for (j, &f) in BOTH.iter().enumerate() {
            let mut cfg = base.clone();
            cfg.formulation = f;
            match cfg.attenuator {
                Attenuator::Erf => cfg.omega = w,
                Attenuator::Terf => cfg.terf_omega = Some(w),
            }
            let single = rs_mp2_lr_rpa(&sys.mol, &sys.obs, &sys.dfbs, &sys.rhf, &cfg).unwrap();
            assert_bitwise(
                &format!("{label} w={w} {f:?}"),
                &scan[i * BOTH.len() + j],
                &single,
            );
        }
    }
    // Reachability: the points differ from each other (a scan that returned
    // one point three times would pass the equalities above only if the
    // singles were equal too).
    let e = |i: usize| scan[i * BOTH.len()].e_corr;
    assert!(
        e(0) != e(1) && e(1) != e(2),
        "{label}: omega points must differ"
    );
}

fn omegas_bohr() -> Vec<f64> {
    OMEGAS_ANG.iter().map(|w| w * ANG_INV).collect()
}

#[test]
fn scan_is_bitwise_the_single_omega_calls_water() {
    let sys = setup(WATER);
    scan_matches_singles(
        &sys,
        &RsMp2RpaConfig::default(),
        &omegas_bohr(),
        "water erf",
    );
}

#[test]
fn scan_is_bitwise_the_single_omega_calls_water_dimer() {
    let sys = setup(&dimer_xyz());
    scan_matches_singles(
        &sys,
        &RsMp2RpaConfig::default(),
        &omegas_bohr(),
        "water dimer erf",
    );
}

#[test]
fn scan_is_bitwise_the_single_omega_calls_terf() {
    if std::env::var("FERRIC_TERF_TABLE_DIR")
        .map(|d| std::path::Path::new(&d).join("16_4_2.bin").exists())
        != Ok(true)
    {
        eprintln!("SKIP scan terf: FERRIC_TERF_TABLE_DIR tables absent");
        return;
    }
    let sys = setup(WATER);
    let cfg = RsMp2RpaConfig {
        attenuator: Attenuator::Terf,
        r0: 3.18,
        ..Default::default()
    };
    scan_matches_singles(&sys, &cfg, &omegas_bohr(), "water terf r0=3.18");
}

/// A single formulation and a reordered list return the matching entries.
#[test]
fn single_formulation_and_order_are_respected() {
    let sys = setup(WATER);
    let ws = omegas_bohr();
    let cfg = RsMp2RpaConfig::default();
    let t_only = rs_mp2_lr_rpa_scan(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        &sys.rhf,
        &cfg,
        &ws[..2],
        &[RsMp2RpaFormulation::CoupledRings],
    )
    .unwrap();
    let rev = rs_mp2_lr_rpa_scan(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        &sys.rhf,
        &cfg,
        &ws[..2],
        &[
            RsMp2RpaFormulation::CoupledRings,
            RsMp2RpaFormulation::DeltaLr,
        ],
    )
    .unwrap();
    assert_eq!(t_only.len(), 2);
    assert!(t_only[0].e_drpa_lr.is_none() && t_only[0].e_delta_drpa_sr.is_some());
    assert_bitwise("T-only vs reversed", &t_only[1], &rev[2]);
    assert!(rev[3].e_drpa_lr.is_some());
}

/// Misuse is a hard error, never a silent default.
#[test]
fn scan_input_errors() {
    let sys = setup(WATER);
    let cfg = RsMp2RpaConfig::default();
    let run = |w: &[f64], f: &[RsMp2RpaFormulation]| {
        rs_mp2_lr_rpa_scan(&sys.mol, &sys.obs, &sys.dfbs, &sys.rhf, &cfg, w, f)
    };
    assert!(run(&[], &BOTH).is_err(), "empty omegas");
    assert!(run(&[0.2], &[]).is_err(), "empty formulations");
    assert!(run(&[0.0], &BOTH).is_err(), "zero omega");
    assert!(run(&[f64::NAN], &BOTH).is_err(), "NaN omega");
    assert!(
        run(&[0.2], &[BOTH[0], BOTH[0]]).is_err(),
        "repeated formulation"
    );
}

/// T's Coulomb ΔdRPA is the same number at every ω and equals the value from
/// the standalone entry points (`run_pdep_rpa`, `ri_mp2_spin_components`),
/// an independent route that shares no intermediates with the driver. A scan
/// that dropped or zeroed the Coulomb dRPA fails the comparison. Bar: 1e-10 Ha
/// (the fused and standalone routes build the same intermediates by the same
/// code, so they agree to rounding; the bar is the existing fusion test's).
#[test]
fn t_carries_the_independently_computed_coulomb_drpa() {
    let sys = setup(WATER);
    let cfg = RsMp2RpaConfig::default();
    let scan = rs_mp2_lr_rpa_scan(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        &sys.rhf,
        &cfg,
        &omegas_bohr(),
        &[RsMp2RpaFormulation::CoupledRings],
    )
    .unwrap();
    let (sc, _) = ferric_mp2::rimp2::ri_mp2_spin_components(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &sys.rhf,
        &ferric_mp2::rimp2::RiMp2Config::default(),
    )
    .unwrap();
    let drpa = ferric_rpa::run_pdep_rpa(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &sys.rhf,
        &cfg.drpa,
    )
    .unwrap();
    let want = drpa.e_rpa - 2.0 * sc.e_os;
    let first = scan[0].e_delta_drpa_full.unwrap();
    assert!((first - want).abs() < 1e-10, "{first} vs {want}");
    for r in &scan {
        assert_eq!(bits(r.e_delta_drpa_full.unwrap()), bits(first));
    }
}

/// The preflight estimate does not grow with the number of ω points, covers
/// each phase of one operator's set, and is tensor-sized.
#[test]
fn preflight_estimate_is_independent_of_the_number_of_points() {
    let shape = ferric_rpa::budget::PeakEstimateShape {
        naux: 1512,
        nocc: 21,
        nvir: 393,
        n_quad: 1,
        n_workers: 8,
        n_keep: 1512,
        grid: None,
        need_inv_dielectric: false,
        nao: 414,
    };
    let budget = 1usize << 40;
    let one = scan_peak_bytes(shape, 1, budget);
    assert_eq!(scan_peak_bytes(shape, 5, budget), one);
    assert_eq!(scan_peak_bytes(shape, 50, budget), one);
    // At an ample budget the AO tensor is in core, and the build phase holds
    // it beside the persistent planes.
    let ao = ferric_rpa::budget::ao_tensor_bytes(shape);
    let persistent = ferric_rpa::budget::scan_persistent_bytes(shape);
    assert!(one >= persistent + ao, "{one} < {persistent} + {ao}");
    // The dRPA phase holds the eigensolve planes and the full-width scratch.
    let drpa = ferric_rpa::budget::static_eigensolve_bytes(shape)
        + ferric_rpa::budget::quad_scratch_bytes(shape);
    assert!(one >= persistent + drpa);
    assert!(one > 100_000_000, "estimate is tensor-sized: {one}");
}
