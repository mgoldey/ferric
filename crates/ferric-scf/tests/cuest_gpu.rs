//! SCF on cuEST (GPU) vs ferric's own exact-J/K SCF on the CPU.
//!
//! Needs `--features cuest` and a device cuEST supports (tested on compute
//! capability 8.0 and newer). Without a usable device the
//! test prints SKIP and passes, unless FERRIC_CUEST_REQUIRE_DEVICE=1.
//!
//!     FERRIC_CUEST_REQUIRE_DEVICE=1 cargo test --release -p ferric-scf --features cuest \
//!         --test cuest_gpu -- --nocapture --test-threads=1
//!
//! Writes `$CARGO_TARGET_TMPDIR/cuest-scf-report.json` (energies, |ΔE|, AO
//! calibration, cuEST timings) for every case, pass or fail; when a required
//! device cannot be set up it holds one `setup` case with the error.
#![cfg(feature = "cuest")]

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_cuest::handle::CuestHandle;
use ferric_cuest::CuestError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cuest_route::{solve_rhf_cuest, solve_uhf_cuest, Calibration};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use serde::Serialize;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

/// cuEST and libint are both fp64; cuEST screens J/K at a density-weighted
/// Schwarz 1e-13 and ferric at its integral threshold. Expected agreement is
/// ~1e-10 Ha; any convention defect in J or K (order, factor, sign) is
/// >= 1e-3 Ha. 1e-8 Ha leaves two orders of margin over the first.
const BAR_E: f64 = 1e-8;

#[derive(Serialize)]
struct Case {
    name: String,
    e_cpu: Option<f64>,
    e_cuest: Option<f64>,
    abs_de: Option<f64>,
    iterations_cpu: Option<usize>,
    iterations_cuest: Option<usize>,
    calibration: Option<Calibration>,
    cuest_timings: Option<String>,
    error: Option<String>,
    pass: bool,
}

fn require_device() -> bool {
    std::env::var("FERRIC_CUEST_REQUIRE_DEVICE").is_ok_and(|v| v == "1")
}

fn run_case(name: &str, basis_name: &str, charge: i32, mult: usize, xc: Option<&str>) -> Case {
    let mut case = Case {
        name: name.into(),
        e_cpu: None,
        e_cuest: None,
        abs_de: None,
        iterations_cpu: None,
        iterations_cuest: None,
        calibration: None,
        cuest_timings: None,
        error: None,
        pass: false,
    };
    let mol = Molecule::parse_xyz(WATER, charge, mult).expect("water");
    let bs = basis::bundled(basis_name).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let mut cfg = RhfConfig::default()
        .with_density_conv(1e-9)
        .with_df_j_aux(String::new())
        .with_df_k_aux(String::new());
    if let Some(x) = xc {
        cfg = cfg.with_xc(x);
    }
    let open = mult > 1;
    let cpu = if open {
        solve_uhf(&ctx, &mol, &prep, &bounds, &cfg)
    } else {
        solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg)
    };
    match cpu {
        Ok(r) => {
            case.e_cpu = Some(r.energy);
            case.iterations_cpu = Some(r.iterations);
        }
        Err(e) => {
            case.error = Some(format!("CPU reference: {e}"));
            return case;
        }
    }
    let gpu = if open {
        solve_uhf_cuest(&ctx, &mol, &prep, &bounds, &cfg)
    } else {
        solve_rhf_cuest(&ctx, &mol, &prep, op, &bounds, &cfg)
    };
    match gpu {
        Ok(run) => {
            case.e_cuest = Some(run.result.energy);
            case.iterations_cuest = Some(run.result.iterations);
            case.calibration = Some(run.calibration);
            case.cuest_timings = Some(run.timings);
            let de = (run.result.energy - case.e_cpu.unwrap()).abs();
            case.abs_de = Some(de);
            case.pass = run.result.converged && de < BAR_E;
        }
        Err(e) => case.error = Some(format!("cuEST SCF: {e}")),
    }
    case
}

fn write_report(cases: &[Case]) -> std::path::PathBuf {
    let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("cuest-scf-report.json");
    std::fs::write(&out, serde_json::to_string_pretty(cases).unwrap()).unwrap();
    println!("REPORT: {}", out.display());
    out
}

#[test]
fn scf_on_cuest_matches_ferric() {
    let setup_error = match CuestHandle::new() {
        Ok(_) => None,
        Err(CuestError::DeviceUnavailable(why)) if !require_device() => {
            println!("SKIP scf_on_cuest_matches_ferric: {why} (set FERRIC_CUEST_REQUIRE_DEVICE=1 to fail instead)");
            return;
        }
        Err(e) => Some(format!("cuEST handle creation failed: {e}")),
    };
    if let Some(why) = setup_error {
        let out = write_report(&[Case {
            name: "setup".into(),
            e_cpu: None,
            e_cuest: None,
            abs_de: None,
            iterations_cpu: None,
            iterations_cuest: None,
            calibration: None,
            cuest_timings: None,
            error: Some(why.clone()),
            pass: false,
        }]);
        panic!("{why}; send {}", out.display());
    }
    let cases = vec![
        run_case("water/sto-3g RHF", "sto-3g", 0, 1, None),
        run_case("water/6-31g* RHF (Cartesian d)", "6-31g*", 0, 1, None),
        run_case("water/cc-pvdz RHF", "cc-pvdz", 0, 1, None),
        run_case("water/cc-pvdz PBE", "cc-pvdz", 0, 1, Some("PBE")),
        run_case("water/cc-pvdz PBE0", "cc-pvdz", 0, 1, Some("PBE0")),
        run_case("water+/cc-pvdz UHF", "cc-pvdz", 1, 2, None),
        run_case("water+/cc-pvdz UKS PBE0", "cc-pvdz", 1, 2, Some("PBE0")),
        run_case("water/cc-pvtz RHF (f shells)", "cc-pvtz", 0, 1, None),
    ];
    for c in &cases {
        println!(
            "{:<34} pass={:<5} E_cpu={:?} E_cuest={:?} |dE|={:?} iters={:?}/{:?}\n    {}\n    {}\n    error={:?}",
            c.name,
            c.pass,
            c.e_cpu,
            c.e_cuest,
            c.abs_de,
            c.iterations_cpu,
            c.iterations_cuest,
            c.calibration.as_ref().map(|x| x.summary()).unwrap_or_default(),
            c.cuest_timings.as_deref().unwrap_or(""),
            c.error
        );
    }
    let out = write_report(&cases);
    let failed: Vec<&str> = cases
        .iter()
        .filter(|c| !c.pass)
        .map(|c| c.name.as_str())
        .collect();
    assert!(
        failed.is_empty(),
        "cuEST SCF failed for {failed:?}; send {}",
        out.display()
    );
}
