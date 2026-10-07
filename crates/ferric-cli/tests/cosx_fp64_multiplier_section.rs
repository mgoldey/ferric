//! `[scf] cosx_fp64_multiplier` is a router knob: it is read only with
//! `k_builder = "cosx"`, `[gpu] precision = "mixed"` and `cosx-kern` in
//! `mixed_kernels`; anywhere else it is a dead knob and a hard error naming
//! the keys (Review Focus 5 of the GPU roadmap). `cosx-kern` is named but not
//! shipped, so a build with the default allowlist can never enable it.
use std::path::{Path, PathBuf};
use std::process::Command;

fn ferric_cli_bin() -> PathBuf {
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

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

/// Run the CLI on `body` with the FERRIC_GPU* env cleared (TOML is the only source).
fn run_toml(tag: &str, body: &str) -> std::process::Output {
    let root = workspace_root();
    let dir = root.join("target");
    std::fs::create_dir_all(&dir).expect("create target/ for the temp toml");
    let path = dir.join(format!("cosx_fp64_multiplier_section_{tag}.toml"));
    std::fs::write(&path, body).expect("write temp toml");
    Command::new(ferric_cli_bin())
        .arg(&path)
        .arg("--no-json")
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .env_remove("FERRIC_GPU")
        .env_remove("FERRIC_GPU_DEVICE")
        .env_remove("FERRIC_GPU_MEM_GB")
        .env_remove("FERRIC_GPU_MIN_FLOPS")
        .env_remove("FERRIC_GPU_PRECISION")
        .env_remove("FERRIC_GPU_MIXED_KERNELS")
        .output()
        .expect("failed to run ferric-cli binary")
}

const WATER: &str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
"#;

fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

const DEAD_KNOB: &str = "[scf] cosx_fp64_multiplier requires k_builder = \"cosx\" and [gpu] precision = \"mixed\" with cosx-kern in mixed_kernels";

#[test]
fn cosx_fp64_multiplier_without_cosx_or_without_mixed_is_refused() {
    // 1. k_builder = "direct": the generic cosx_* refusal, naming the key.
    let out = run_toml(
        "direct",
        &format!("{WATER}\n[scf]\nk_builder = \"direct\"\ncosx_fp64_multiplier = 1e5\n"),
    );
    assert!(!out.status.success(), "{}", stderr(&out));
    let e = stderr(&out);
    assert!(
        e.contains("cosx_fp64_multiplier") && e.contains("only read with k_builder = \"cosx\""),
        "{e}"
    );
    // 2. cosx with no [gpu] section (f64): the dead-knob message, verbatim.
    let out = run_toml(
        "f64",
        &format!("{WATER}\n[scf]\nk_builder = \"cosx\"\ncosx_fp64_multiplier = 1e5\n"),
    );
    assert!(!out.status.success(), "{}", stderr(&out));
    let e = stderr(&out);
    assert!(e.contains(DEAD_KNOB), "{e}");
    // 3. cosx + mixed with the SHIPPED allowlist (no cosx-kern): refused, same message
    //    (it names cosx-kern and mixed_kernels).
    let out = run_toml(
        "mixed_unshipped",
        &format!(
            "{WATER}\n[gpu]\nmode = \"auto\"\nprecision = \"mixed\"\n[scf]\nk_builder = \"cosx\"\ncosx_fp64_multiplier = 1e5\n"
        ),
    );
    assert!(!out.status.success(), "{}", stderr(&out));
    let e = stderr(&out);
    assert!(e.contains(DEAD_KNOB), "{e}");
    // 3b. naming cosx-kern in mixed_kernels is refused as not shipped, naming it.
    let out = run_toml(
        "mixed_cosx_kern",
        &format!(
            "{WATER}\n[gpu]\nmode = \"auto\"\nprecision = \"mixed\"\nmixed_kernels = [\"cosx-kern\"]\n[scf]\nk_builder = \"cosx\"\ncosx_fp64_multiplier = 1e5\n"
        ),
    );
    assert!(!out.status.success(), "{}", stderr(&out));
    let e = stderr(&out);
    assert!(
        e.contains("cosx-kern not shipped yet")
            && e.contains("shipped in this build: rimp2-energy"),
        "{e}"
    );
    // 4. The same cosx + mixed TOML WITHOUT the key is accepted and runs.
    let out = run_toml(
        "mixed_no_key",
        &format!(
            "{WATER}\n[gpu]\nmode = \"auto\"\nprecision = \"mixed\"\n[scf]\nk_builder = \"cosx\"\n"
        ),
    );
    assert!(out.status.success(), "{}", stderr(&out));
}
