//! Failure 1 through the real binary: a 3-index tensor that fits
//! `[memory] budget_gb` once but not twice used to end the run with
//! `error: DF-guess SCF failed: ... memory pool exhausted`.
//!
//! The budget is derived from the tensor's bytes, so the test stays in the band
//! however the basis data changes. Butane, because on anything much smaller the
//! fast path's per-worker plan refuses the band before the pool is ever asked. HOW THIS FAILS IF BROKEN: the budget-only
//! in-core decision exits 1 at the dressed tensor's charge.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use std::path::PathBuf;
use std::process::Command;

/// The compile-time `CARGO_BIN_EXE_` path does not exist where a nextest archive
/// is unpacked, so prefer the sibling of the running test binary; see
/// `ccsd_dispatch.rs`.
fn ferric_cli_bin() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(profile_dir) = exe.parent().and_then(std::path::Path::parent) {
            let p = profile_dir.join("ferric-cli");
            if p.is_file() {
                return p;
            }
        }
    }
    PathBuf::from(env!("CARGO_BIN_EXE_ferric-cli"))
}

/// `CARGO_MANIFEST_DIR` is stale in an unpacked nextest archive as well.
fn root() -> PathBuf {
    let looks_like_root = |p: &std::path::Path| {
        p.join("Cargo.toml").is_file() && p.join("examples").is_dir() && p.join("testdata").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&std::path::Path> = Some(cwd.as_path());
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

fn run(tag: &str, budget_gb: f64) -> (std::process::Output, f64) {
    let dir = std::env::temp_dir().join(format!("ferric_df_guess_mem_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let (toml, log) = (
        dir.join(format!("df_guess_mem_{tag}.toml")),
        dir.join(format!("df_guess_mem_{tag}.jsonl")),
    );
    let _ = std::fs::remove_file(&log);
    std::fs::write(
        &toml,
        format!(
            "[molecule]\nxyz = \"testdata/molecules/alkane_4.xyz\"\n\
             [basis]\nname = \"6-31g\"\n\
             [method]\nkind = \"rimp2\"\n\
             [mp2]\nauxbasis = \"cc-pvdz-ri\"\n\
             [memory]\nbudget_gb = {budget_gb:.9}\n"
        ),
    )
    .expect("write toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml)
        .arg("--json")
        .arg(&log)
        .current_dir(root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("run ferric-cli");
    // The last SCF iteration, since a later correlated stage may refuse a budget
    // this small, and that is not what this test is about.
    let energy = std::fs::read_to_string(&log)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|r| r["record"] == "scf_iter")
        .filter_map(|r| r["energy"].as_f64())
        .next_back()
        .unwrap_or(f64::NAN);
    (out, energy)
}

#[test]
fn a_tensor_that_fits_the_budget_once_but_not_twice_still_runs_the_guess() {
    let mol = Molecule::load_xyz(
        root()
            .join("testdata/molecules/alkane_4.xyz")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let aux = basis::bundled(ferric_scf::ladder::DF_GUESS_DEFAULT_AUX).unwrap();
    let aux = PreparedBasis::new(&mol, &aux).unwrap();
    let needed = aux.nbasis() * obs.nbasis() * obs.nbasis() * 8;
    let gib = (1u64 << 30) as f64;

    let (ample, e_ample) = run("ample", 1.0);
    assert!(
        ample.status.success(),
        "{}",
        String::from_utf8_lossy(&ample.stderr)
    );

    let (band, e_band) = run("band", needed as f64 * 1.5 / gib);
    let stderr = String::from_utf8_lossy(&band.stderr);
    assert!(!stderr.contains("SCF failed"), "{stderr}");
    assert!(
        stderr.contains("spilling it to disk"),
        "must be in the band:\n{stderr}"
    );
    assert!(
        stderr.contains("SCF: DF-guess pre-stage via"),
        "guess must run:\n{stderr}"
    );
    assert!(
        (e_ample - e_band).abs() < 1e-7,
        "{e_ample:.12} vs {e_band:.12}"
    );
}
