//! Through the real binary: a budget that holds the raw 3-index tensor but not
//! also the XC working set must run, and say that XC is not covered. Before XC
//! was committed last, this budget exited 1 at the raw tensor's charge. Derived
//! from the tensor's bytes so it stays in that band if basis data changes.
//!
//! Water on the default grid at one thread has `L` = 11.05 `T`, so 11.5 `T` is
//! inside the old `[L, L + T)` band, and XC cannot fit beside raw's `T`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use std::path::{Path, PathBuf};
use std::process::Command;

/// See `ccsd_dispatch.rs`: paths baked in at compile time do not survive a nextest
/// archive.
fn ferric_cli_bin() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        // .../target/<profile>/deps/<test-binary>  ->  .../target/<profile>/
        if let Some(profile_dir) = exe.parent().and_then(Path::parent) {
            let p = profile_dir.join("ferric-cli");
            if p.is_file() {
                return p;
            }
        }
    }
    PathBuf::from(env!("CARGO_BIN_EXE_ferric-cli"))
}

/// Workspace root, resolved at RUN TIME.
///
/// `env!("CARGO_MANIFEST_DIR")` is baked in when the test binary is COMPILED.
/// Under `cargo nextest archive` the binary is built in one job and run in
/// another, whose checkout lives at a different path -- so the compile-time
/// directory does not exist and `Command::current_dir` fails with a bare
/// `NotFound` that reads as "the ferric-cli binary is missing" (MEASURED: 36
/// of 37 shard failures, all of them this).
///
/// So: walk up from the CURRENT directory to the nearest ancestor holding a
/// workspace `Cargo.toml` alongside `examples/` and `testdata/`, and fall back
/// to the compile-time path when that fails (the ordinary `cargo test` case,
/// where it is correct and the cwd may be anywhere).
fn workspace_root() -> PathBuf {
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

#[test]
fn a_budget_that_holds_raw_but_not_xc_runs_and_says_so() {
    let root = workspace_root();
    let mol = Molecule::load_xyz(root.join("testdata/molecules/water.xyz").to_str().unwrap());
    let mol = mol.unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = basis::bundled("def2-universal-jkfit").unwrap();
    let aux = PreparedBasis::new(&mol, &aux).unwrap();
    let tensor = aux.nbasis() * obs.nbasis() * obs.nbasis() * 8;
    let budget_gb = (23 * tensor / 2) as f64 / (1u64 << 30) as f64;

    let dir = root.join("target");
    std::fs::create_dir_all(&dir).unwrap();
    let toml = dir.join("ks_xc_uncharged.toml");
    let log = dir.join("ks_xc_uncharged.jsonl");
    let _ = std::fs::remove_file(&log);
    std::fs::write(
        &toml,
        format!(
            "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\
             [basis]\nname = \"cc-pvdz\"\n\
             [method]\nkind = \"ksdft\"\n\
             [dft]\nfunctional = \"PBE\"\n\
             [memory]\nbudget_gb = {budget_gb:.9}\n"
        ),
    )
    .unwrap();
    let out = Command::new(ferric_cli_bin())
        .arg(&toml)
        .arg("--json")
        .arg(&log)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(
        stderr.contains("KS-DFT grid AO blocks recomputed"),
        "{stderr}"
    );
    assert!(stderr.contains("runs uncharged"), "{stderr}");
    assert!(stderr.contains("can exceed its budget"), "{stderr}");
    assert_eq!(
        stderr.matches("KS-DFT grid AO blocks recomputed").count(),
        1,
        "{stderr}"
    );
    // The run log carries the amount, not just stderr.
    let records = std::fs::read_to_string(&log).unwrap();
    let noted = records
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|r| r.to_string().contains("xc_storage"))
        .unwrap_or_else(|| panic!("no xc_storage record:\n{records}"));
    assert!(noted.to_string().contains("uncharged_bytes"), "{noted}");
    assert!(
        !noted.to_string().contains("\"uncharged_bytes\":0"),
        "{noted}"
    );
}
