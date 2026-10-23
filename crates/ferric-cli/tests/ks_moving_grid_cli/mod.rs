//! Shared harness of `ks_moving_grid_frequencies.rs` (per-commit) and
//! `validation_ks_moving_grid_frequencies.rs` (validation tier): run the real
//! `ferric-cli` binary on a frequency TOML and read the full-precision
//! `frequencies` record of its JSON run log.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use ferric_scf::rhf::RhfConfig;

/// Near-equilibrium water (HF/cc-pVDZ geometry), relative to the workspace root.
pub const WATER: &str = "testdata/molecules/water.xyz";
/// H2, relative to the workspace root.
pub const H2: &str = "testdata/molecules/h2.xyz";
/// HI, written under `target/` on first use (iodine carries a def2 ECP).
pub const HI: &str = "target/ks_moving_grid_cli_hi.xyz";

/// Rayon width of the CLI runs AND of the in-process library side, so the two
/// run the same reduction order.
pub const CLI_THREADS: usize = 2;

/// Tight SCF: the finite-difference Hessian differentiates SCF gradients, so
/// its noise is the SCF's. `df_guess = false` keeps the CLI on the one-stage
/// SCF the library runs.
pub const SCF: &str =
    "[scf]\nmax_iter = 200\nenergy_conv = 1e-13\ndensity_conv = 1e-11\ndf_guess = false\n";

/// See `ccsd_dispatch.rs`: resolved at RUN time for `cargo nextest archive`.
pub fn ferric_cli_bin() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(profile_dir) = exe.parent().and_then(Path::parent) {
            let p = profile_dir.join("ferric-cli");
            if p.is_file() {
                return p;
            }
        }
    }
    PathBuf::from(env!("CARGO_BIN_EXE_ferric-cli"))
}

/// See `ccsd_dispatch.rs`: resolved at RUN time for `cargo nextest archive`.
pub fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("examples").is_dir() && p.join("testdata").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

pub fn has_deriv2() -> bool {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
    }
    order >= 2
}

/// A full TOML: `mol` (path relative to the root), `basis`, `method.kind`,
/// `task = "frequencies"`, the tight [`SCF`], then `dft` and `rest` verbatim.
pub fn toml_body(mol: &str, basis: &str, kind: &str, dft: &str, rest: &str) -> String {
    format!(
        "[molecule]\nxyz = \"{mol}\"\n\n[basis]\nname = \"{basis}\"\n\n\
         [method]\nkind = \"{kind}\"\ntask = \"frequencies\"\n\n{SCF}\n{dft}\n{rest}\n"
    )
}

/// Run the binary on `toml`; `(success, stdout, stderr)`. The JSON run log
/// goes to `target/ks_mg_<tag>.jsonl`.
pub fn run_cli_raw(tag: &str, toml: &str) -> (bool, String, String) {
    let root = workspace_root();
    let hi = root.join(HI);
    if !hi.is_file() {
        std::fs::write(&hi, "2\nHI\nH 0 0 0\nI 0 0 1.75\n").expect("write HI xyz");
    }
    let toml_path = root.join("target").join(format!("ks_mg_{tag}.toml"));
    let log_path = root.join("target").join(format!("ks_mg_{tag}.jsonl"));
    let _ = std::fs::remove_file(&log_path);
    std::fs::write(&toml_path, toml).expect("write temp toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml_path)
        .arg("--json")
        .arg(&log_path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", CLI_THREADS.to_string())
        .output()
        .expect("failed to run ferric-cli binary");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Number of SCF iteration records (`scf_iter`, `guess_scf_iter`) in the run
/// log of `tag`.
pub fn scf_iterations_logged(tag: &str) -> usize {
    let log_path = workspace_root()
        .join("target")
        .join(format!("ks_mg_{tag}.jsonl"));
    let log = std::fs::read_to_string(&log_path).unwrap_or_default();
    log.lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["record"] == "scf_iter" || v["record"] == "guess_scf_iter")
        .count()
}

/// The `frequencies` record of a run log.
pub struct Freq {
    pub source: String,
    pub energy: f64,
    pub frequencies: Vec<f64>,
    pub hessian: Vec<f64>,
    pub n_grad: u64,
    pub asymmetry: f64,
}

fn floats(v: &serde_json::Value, what: &str) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("{what} is not an array: {v}"))
        .iter()
        .map(|x| {
            x.as_f64()
                .unwrap_or_else(|| panic!("{what}: {x} is not a number"))
        })
        .collect()
}

/// A successful KS frequency run on `mol`/STO-3G with functional `xc`;
/// `freq` is the `[frequencies]` section (or empty for the defaults) and
/// `dft_extra` extra `[dft]` lines.
pub fn run_cli_ok(tag: &str, mol: &str, xc: &str, freq: &str, dft_extra: &str) -> Freq {
    let dft = format!("[dft]\nfunctional = \"{xc}\"\n{dft_extra}");
    let toml = toml_body(mol, "sto-3g", "ksdft", &dft, freq);
    let (ok, stdout, stderr) = run_cli_raw(tag, &toml);
    assert!(ok, "{tag} failed.\nstdout:\n{stdout}\nstderr:\n{stderr}");
    let log_path = workspace_root()
        .join("target")
        .join(format!("ks_mg_{tag}.jsonl"));
    let log = std::fs::read_to_string(&log_path).expect("read JSON run log");
    let recs: Vec<serde_json::Value> = log
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["record"] == "frequencies")
        .collect();
    assert_eq!(
        recs.len(),
        1,
        "{tag}: expected one frequencies record:\n{log}"
    );
    let r = &recs[0];
    Freq {
        source: r["hessian_source"].as_str().unwrap().to_string(),
        energy: r["energy"].as_f64().unwrap(),
        frequencies: floats(&r["frequencies_cm1"], "frequencies_cm1"),
        hessian: floats(&r["cartesian_hessian"], "cartesian_hessian"),
        n_grad: r["n_gradient_evaluations"].as_u64().unwrap(),
        asymmetry: r["asymmetry"].as_f64().unwrap(),
    }
}

/// The `RhfConfig` the CLI builds from [`run_cli_ok`]'s TOML: a Kohn-Sham
/// SCF fits J and K over the default basis, the rest are the CLI defaults.
pub fn library_config(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.to_string()),
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        df_k_aux: Some("def2-universal-jkfit".to_string()),
        max_iter: 200,
        energy_conv: 1e-13,
        density_conv: 1e-11,
        ..Default::default()
    }
}

/// Run `f` on a rayon pool of [`CLI_THREADS`] threads.
pub fn with_cli_threads<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(CLI_THREADS)
        .build()
        .expect("rayon pool")
        .install(f)
}

pub fn assert_bits_eq(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths differ");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}]: {x:e} vs {y:e}");
    }
}

/// `max_i |a_i - b_i|`.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}
