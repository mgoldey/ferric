//! VALIDATION tier: "Correlated harmonic frequencies" (`task = "frequencies"`
//! for `rimp2` closed and open shell and `pdep-rpa`; `scs-mp2` and `oo-rimp2`
//! are not covered).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --release \
//!   --test validation_correlated_frequencies -- --ignored --nocapture
//! ```
//!
//! # What is compared
//!
//! The CLI's `frequencies` record (Cartesian Hessian and wavenumbers) against
//! `testdata/reference/validation/correlated_frequencies/<case>.json`, written
//! by `scripts/validation/gen_correlated_frequencies.py`. The reference
//! Hessian is the central difference, at the CLI's own step (5e-3 Bohr), of a
//! 5-point finite-difference gradient of PySCF's total energy (DF-MP2,
//! DF-UMP2, DF-SCS-MP2, RPA at 40 Gauss-Legendre points, exact-J/K SCF) on
//! ferric's bases, so the two Hessians carry the same O(delta^2) truncation
//! and differ only by the gradient disagreement. The script's header derives
//! the Hessian-element bar `b = (eps_g + eps_ref) / delta` and the per-mode
//! wavenumber bar `nu / (2 lambda) * b * sum_c 1/m_c`; both are stored in the
//! JSON and used here unchanged.
//!
//! Systems: strained, asymmetric, rotated H2O (`rimp2`, `pdep-rpa`) and OH
//! (`rimp2`, UHF), 6-31G / cc-pvdz-ri, so no Cartesian component of the gradient is
//! zero by symmetry or orientation.
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! * Right: the CLI differentiates the kind's own analytic gradient, with the
//!   frozen core, auxiliary basis, scaling and quadrature of the input, and
//!   matches the reference inside the derived bar.
//! * Wrong, still runs: any one of those settings not reaching the displaced
//!   gradients (frozen core dropped, the SCF Hessian returned instead of the
//!   correlated one). That moves the wavenumbers by tens of cm^-1 or more, far
//!   above the bars (0.1 to 1 cm^-1); the stored SCF-level wavenumbers are the
//!   negative control and the test requires them to miss by 10 bars.
//!
//! # Exactness anchor
//!
//! Before any wavenumber is compared, the energy of the run at the reference
//! geometry must equal PySCF's (`E_TOL`), which shows basis, auxiliary basis,
//! frozen core, scaling and quadrature are like-for-like.
use std::path::PathBuf;
use std::process::Command;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

const BOHR_IN_ANGSTROM: f64 = 0.529_177_210_92;
/// Energy anchor, Ha. PySCF and ferric agree to <= 1e-9 on RI-MP2 and 2e-11 on
/// RPA at converged SCF (`validation_rimp2_gradient`, `validation_rpa_gradient`).
const E_TOL: f64 = 1e-8;
/// The SCF-level wavenumbers must miss the reference by this many bars.
const MUST_MISS_BARS: f64 = 10.0;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn floats(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("float"))
        .collect()
}

fn run_case(case: &str) {
    let root = workspace_root();
    let ref_path = root
        .join("testdata/reference/validation/correlated_frequencies")
        .join(format!("{case}.json"));
    let r: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&ref_path).unwrap_or_else(|e| panic!("{ref_path:?}: {e}")),
    )
    .expect("reference JSON");
    let kind = r["kind"].as_str().unwrap();
    let basis = r["basis"].as_str().unwrap();
    let aux = r["auxbasis"].as_str().unwrap();
    let fc = r["frozen_core"].as_u64().unwrap();
    let mult = r["multiplicity"].as_u64().unwrap();
    let symbols: Vec<&str> = r["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    let coords: Vec<Vec<f64>> = r["geometry_bohr"]
        .as_array()
        .unwrap()
        .iter()
        .map(floats)
        .collect();

    let dir = root
        .join("target")
        .join("validation_correlated_frequencies");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let xyz = dir.join(format!("{case}.xyz"));
    let mut text = format!("{}\n{case}\n", symbols.len());
    for (s, c) in symbols.iter().zip(&coords) {
        text += &format!(
            "{s} {:.15} {:.15} {:.15}\n",
            c[0] * BOHR_IN_ANGSTROM,
            c[1] * BOHR_IN_ANGSTROM,
            c[2] * BOHR_IN_ANGSTROM
        );
    }
    std::fs::write(&xyz, text).expect("write xyz");

    let method = match kind {
        "pdep-rpa" => format!(
            "[method]\nkind = \"pdep-rpa\"\ntask = \"frequencies\"\n[rpa]\nauxbasis = \"{aux}\"\n\
             frozen_core = {fc}\nn_quad = 40\nquadrature = \"gauss-legendre\"\nu0 = 0.5\n"
        ),
        k => format!(
            "[method]\nkind = \"{k}\"\ntask = \"frequencies\"\n[mp2]\nauxbasis = \"{aux}\"\n\
             frozen_core = {fc}\n"
        ),
    };
    // `[scf] jk` is read by the MP2 kinds only; pdep-rpa is always four-centre.
    let jk = if kind == "pdep-rpa" {
        ""
    } else {
        "jk = \"exact\"\n"
    };
    let toml = format!(
        "[molecule]\nxyz = \"{}\"\nmultiplicity = {mult}\n[basis]\nname = \"{basis}\"\n{method}\
         [scf]\nenergy_conv = 1e-12\ndensity_conv = 1e-10\nri_j_precision = \"f64\"\n{jk}",
        xyz.display()
    );
    let toml_path = dir.join(format!("{case}.toml"));
    let log = dir.join(format!("{case}.jsonl"));
    let _ = std::fs::remove_file(&log);
    std::fs::write(&toml_path, toml).expect("write toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml_path)
        .arg("--json")
        .arg(&log)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("run ferric-cli");
    assert!(
        out.status.success(),
        "{case}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let records: Vec<serde_json::Value> = std::fs::read_to_string(&log)
        .expect("run log")
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let rec = records
        .iter()
        .find(|x| x["record"] == "frequencies")
        .unwrap_or_else(|| panic!("{case}: no frequencies record"));

    // Exactness anchor: the same energy surface at the same geometry.
    let e = rec["energy"].as_f64().unwrap();
    let e_ref = r["e_total"].as_f64().unwrap();
    assert!(
        (e - e_ref).abs() < E_TOL,
        "{case}: anchor |E - E_PySCF| = {:.3e} > {E_TOL:.0e}",
        (e - e_ref).abs()
    );

    // Cartesian Hessian, element by element.
    let b = r["bar"]["hessian_element"].as_f64().unwrap();
    let h = floats(&rec["cartesian_hessian"]);
    let h_ref = floats(&r["cartesian_hessian"]);
    assert_eq!(h.len(), h_ref.len(), "{case}: Hessian size");
    let dh = h
        .iter()
        .zip(&h_ref)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0_f64, f64::max);
    println!("{case}: max |dH| = {dh:.3e} Ha/Bohr^2 (bar {b:.3e})");
    assert!(dh < b, "{case}: max |dH| {dh:.3e} >= bar {b:.3e}");

    // Wavenumbers, mode by mode.
    let nu = floats(&rec["frequencies_cm1"]);
    let nu_ref = floats(&r["frequencies_cm1"]);
    let bar = floats(&r["bar"]["cm1_per_mode"]);
    assert_eq!(nu.len(), nu_ref.len(), "{case}: mode count");
    let mut worst = 0.0_f64;
    for (i, ((a, c), bi)) in nu.iter().zip(&nu_ref).zip(&bar).enumerate() {
        println!(
            "{case}: mode {i}: {a:.4} vs {c:.4} cm^-1, |d| {:.4}, bar {bi:.4}",
            (a - c).abs()
        );
        assert!(
            (a - c).abs() < *bi,
            "{case}: mode {i}: |{a:.5} - {c:.5}| = {:.5} >= bar {bi:.5} cm^-1",
            (a - c).abs()
        );
        worst = worst.max((a - c).abs() / bi);
    }
    println!("{case}: worst |d nu| / bar = {worst:.3}");

    // Negative control: the SCF-level wavenumbers miss by many bars.
    let scf = floats(&r["scf_frequencies_cm1"]);
    let max_bar = bar.iter().cloned().fold(0.0_f64, f64::max);
    let miss = scf
        .iter()
        .zip(&nu_ref)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        miss > MUST_MISS_BARS * max_bar,
        "{case}: the SCF frequencies miss by {miss:.3} cm^-1, under {MUST_MISS_BARS} bars \
         ({max_bar:.3}): the reference does not discriminate the correlated Hessian"
    );
}

macro_rules! case {
    ($name:ident, $case:literal) => {
        #[test]
        #[ignore = "validation: Correlated harmonic frequencies"]
        fn $name() {
            run_case($case);
        }
    };
}

case!(rimp2_h2o, "rimp2_h2o");
case!(rimp2_oh_uhf, "rimp2_oh_uhf");
case!(pdep_rpa_h2o, "pdep-rpa_h2o");
