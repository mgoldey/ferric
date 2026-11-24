//! `task = "frequencies"` prints and logs RRHO thermochemistry and IR
//! intensities, and the logged numbers are the library's.
//!
//! The run log's `frequencies` record carries `thermochemistry` and
//! `ir_intensities_km_mol`. The thermochemistry must equal
//! `rrho_thermochemistry` on the record's own wavenumbers and energy to
//! 1e-12 (the same arithmetic on the same inputs; only JSON round-tripping
//! separates them). The intensities must equal `ir_spectrum` run by the
//! library on an independently computed analytic Hessian with the same SCF
//! settings: both runs solve the same SCFs to density_conv 1e-10, so they
//! differ by SCF exits only, `K n dc / h` ~ 1e-6 e on the dipole derivative,
//! i.e. relative ~1e-5 on an intensity; asserted at 1e-4 relative.
//!
//! Negative controls: `symmetry_number` changes `S_rot` by exactly
//! `R ln 2` (a key the CLI ignored would leave it unchanged), and
//! `ir_intensities = false` drops the column. A correlated kind with
//! `ir_intensities = true` is refused before any SCF.

use std::path::PathBuf;
use std::process::{Command, Output};

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{harmonic_frequencies, FrequencyConfig, FrequencyReference};
use ferric_scf::infrared::{ir_spectrum, DEFAULT_IR_FIELD};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::thermo::{rrho_thermochemistry, ThermoConfig};

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

const WATER: &str = "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n";

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn scratch() -> PathBuf {
    let dir = workspace_root()
        .join("target")
        .join("frequencies_thermo_ir_cli_test");
    std::fs::create_dir_all(&dir).expect("create the temp dir");
    dir
}

fn run_cli(tag: &str, kind: &str, extra: &str) -> (Output, Option<serde_json::Value>) {
    let dir = scratch();
    let xyz = dir.join(format!("{tag}.xyz"));
    std::fs::write(&xyz, WATER).expect("write xyz");
    let toml_path = dir.join(format!("{tag}.toml"));
    let log = dir.join(format!("{tag}.jsonl"));
    let _ = std::fs::remove_file(&log);
    let mp2 = if kind == "rimp2" {
        "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n"
    } else {
        ""
    };
    std::fs::write(
        &toml_path,
        format!(
            "[molecule]\nxyz = \"{}\"\n[basis]\nname = \"sto-3g\"\n[method]\nkind = \"{kind}\"\n\
             task = \"frequencies\"\n{mp2}[scf]\nenergy_conv = 1e-10\ndensity_conv = 1e-10\n\
             [frequencies]\n{extra}",
            xyz.display()
        ),
    )
    .expect("write toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml_path)
        .arg("--json")
        .arg(&log)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("run ferric-cli");
    let rec = std::fs::read_to_string(&log)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|r| r["record"] == "frequencies");
    (out, rec)
}

fn f64s(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

#[test]
fn thermo_and_ir_are_printed_logged_and_match_the_library() {
    let (out, rec) = run_cli("water", "rhf", "symmetry_number = 2\ntemperature = 350.0\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("IR (km/mol)"), "{stdout}");
    assert!(
        stdout.contains("Thermochemistry (RRHO): T = 350.00 K"),
        "{stdout}"
    );
    let rec = rec.expect("frequencies record");
    let freqs = f64s(&rec["frequencies_cm1"]);
    let e0 = rec["energy"].as_f64().unwrap();
    let th = &rec["thermochemistry"];
    assert_eq!(th["symmetry_number"], 2);

    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let cfg = ThermoConfig {
        temperature: 350.0,
        symmetry_number: 2,
        ..ThermoConfig::default()
    };
    let lib = rrho_thermochemistry(&mol, &freqs, e0, &cfg).unwrap();
    for (part, t) in [
        ("translational", &lib.translational),
        ("rotational", &lib.rotational),
        ("vibrational", &lib.vibrational),
        ("total", &lib.total),
    ] {
        for (q, v) in [
            ("energy", t.energy),
            ("enthalpy", t.enthalpy),
            ("entropy", t.entropy),
            ("gibbs", t.gibbs),
        ] {
            let logged = th[part][q].as_f64().unwrap();
            assert!(
                (logged - v).abs() <= 1e-12 * v.abs().max(1.0),
                "{part}.{q}: {logged} vs {v}"
            );
        }
    }

    // Intensities: the library on the same SCF settings.
    let scf = RhfConfig::default()
        .with_energy_conv(1e-10)
        .with_density_conv(1e-10);
    let ctx = ParallelContext::default();
    let fc = FrequencyConfig::default();
    let fr = harmonic_frequencies(&ctx, &mol, "sto-3g", Operator::coulomb(), &scf, &fc).unwrap();
    let spec = ir_spectrum(
        &ctx,
        &mol,
        "sto-3g",
        Operator::coulomb(),
        &scf,
        FrequencyReference::Rhf,
        &fr,
        DEFAULT_IR_FIELD,
    )
    .unwrap();
    let logged = f64s(&rec["ir_intensities_km_mol"]);
    assert_eq!(logged.len(), 3);
    for (a, b) in logged.iter().zip(&spec.intensities_km_mol) {
        assert!((a - b).abs() <= 1e-4 * b.abs().max(1e-3), "{a} vs {b}");
    }
    assert!(logged.iter().all(|a| *a > 0.0), "{logged:?}");
}

#[test]
fn symmetry_number_moves_s_rot_and_ir_can_be_switched_off() {
    let (o1, r1) = run_cli("water_s1", "rhf", "ir_intensities = false\n");
    let (o2, r2) = run_cli(
        "water_s2",
        "rhf",
        "ir_intensities = false\nsymmetry_number = 2\n",
    );
    assert!(o1.status.success() && o2.status.success());
    let (r1, r2) = (r1.unwrap(), r2.unwrap());
    let s = |r: &serde_json::Value| {
        r["thermochemistry"]["rotational"]["entropy"]
            .as_f64()
            .unwrap()
    };
    let r_eh = 1.380_649e-23 / 4.359_744_722_207_1e-18;
    assert!((s(&r1) - s(&r2) - r_eh * 2f64.ln()).abs() < 1e-15);
    assert!(r1["ir_intensities_km_mol"].is_null());
    let stdout = String::from_utf8_lossy(&o1.stdout);
    assert!(!stdout.contains("IR (km/mol)"));
    assert!(stdout.contains("ir_intensities = false"), "{stdout}");
    assert!(stdout.contains("symmetry number 1 (default"), "{stdout}");
}

#[test]
fn ir_on_a_correlated_kind_is_refused() {
    let (out, rec) = run_cli("water_mp2", "rimp2", "ir_intensities = true\n");
    assert!(!out.status.success());
    assert!(rec.is_none());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("ir_intensities = true") && err.contains("rimp2"),
        "{err}"
    );
}

#[test]
fn a_bad_thermo_key_is_refused_before_the_scf() {
    let (out, rec) = run_cli("water_t0", "rhf", "temperature = -5.0\n");
    assert!(!out.status.success());
    assert!(rec.is_none());
    assert!(String::from_utf8_lossy(&out.stderr).contains("temperature"));
}
