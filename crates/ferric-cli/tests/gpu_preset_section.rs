//! `[gpu] preset` and the root shorthand `gpu = "<preset>"`: both spellings
//! resolve identically, a disagreeing key is a refusal naming both, and the
//! default config is unchanged. Device-free (the `off` and `auto` arms never
//! need a device).
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard};

/// Serializes every test that can touch a CUDA device (poison-recovering).
static DEVICE: Mutex<()> = Mutex::new(());

fn device_lock() -> MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(|p| p.into_inner())
}

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

/// Run the CLI on `body` with every FERRIC_GPU* env var cleared except those
/// in `env`, so the TOML and `env` are the only sources.
fn run_toml_with_env(tag: &str, body: &str, env: &[(&str, &str)]) -> std::process::Output {
    let root = workspace_root();
    let dir = root.join("target");
    std::fs::create_dir_all(&dir).expect("create target/ for the temp toml");
    let path = dir.join(format!("gpu_preset_section_{tag}.toml"));
    std::fs::write(&path, body).expect("write temp toml");
    Command::new(ferric_cli_bin())
        .arg(&path)
        .arg("--no-json")
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .env_remove("FERRIC_GPU")
        .env_remove("FERRIC_GPU_PRESET")
        .env_remove("FERRIC_GPU_PRECISION")
        .env_remove("FERRIC_GPU_MIXED_KERNELS")
        .env_remove("FERRIC_GPU_DEVICE")
        .env_remove("FERRIC_GPU_MEM_GB")
        .env_remove("FERRIC_GPU_MIN_FLOPS")
        .envs(env.iter().copied())
        .output()
        .expect("failed to run ferric-cli binary")
}

fn run_toml(tag: &str, body: &str) -> std::process::Output {
    run_toml_with_env(tag, body, &[])
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

fn gpu_lines(s: &str) -> Vec<String> {
    s.lines()
        .filter(|l| l.contains("FERRIC_GPU"))
        .map(str::to_string)
        .collect()
}

#[test]
fn root_shorthand_off_prints_the_preset_audit_line_and_runs_on_the_cpu() {
    let _g = device_lock();
    let out = run_toml("root_off", &format!("gpu = \"off\"\n{WATER}"));
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("FERRIC_GPU_PRESET: off  [source: explicit (config/TOML/kwarg)]"),
        "{err}"
    );
    assert!(err.contains("FERRIC_GPU: off  [source: preset]"), "{err}");
}

#[test]
fn root_shorthand_and_section_form_resolve_identically() {
    let _g = device_lock();
    let a = stderr(&run_toml("root_auto", &format!("gpu = \"auto\"\n{WATER}")));
    let b = stderr(&run_toml(
        "sec_auto",
        &format!("{WATER}\n[gpu]\npreset = \"auto\"\n"),
    ));
    assert_eq!(gpu_lines(&a), gpu_lines(&b));
    assert!(a.contains("FERRIC_GPU: auto  [source: preset]"), "{a}");
    assert!(
        a.contains("FERRIC_GPU_PRESET: auto  [source: explicit (config/TOML/kwarg)]"),
        "{a}"
    );
}

#[test]
fn an_env_preset_is_labelled_env_and_a_toml_preset_beats_it() {
    let _g = device_lock();
    let out = run_toml_with_env("env_preset", WATER, &[("FERRIC_GPU_PRESET", "off")]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("FERRIC_GPU_PRESET: off  [source: env]"),
        "{}",
        stderr(&out)
    );
    let out = run_toml_with_env(
        "toml_over_env_preset",
        &format!("gpu = \"off\"\n{WATER}"),
        &[("FERRIC_GPU_PRESET", "auto")],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("FERRIC_GPU_PRESET: off  [source: explicit (config/TOML/kwarg)]"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn root_shorthand_disagreeing_with_env_mode_is_refused_naming_both() {
    let _g = device_lock();
    let out = run_toml_with_env(
        "root_conflict",
        &format!("gpu = \"off\"\n{WATER}"),
        &[("FERRIC_GPU", "on")],
    );
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains(
            "error: [gpu] preset = off (explicit (config/TOML/kwarg)) implies mode = off but \
             FERRIC_GPU / [gpu] mode = on (env); set one of them"
        ),
        "{err}"
    );
}

#[test]
fn section_preset_with_a_disagreeing_precision_is_refused_naming_both() {
    let _g = device_lock();
    let out = run_toml(
        "sec_conflict",
        &format!("{WATER}\n[gpu]\npreset = \"mixed\"\nprecision = \"f64\"\n"),
    );
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains(
            "[gpu] preset = mixed (explicit (config/TOML/kwarg)) implies precision = mixed but \
             FERRIC_GPU_PRECISION / [gpu] precision = f64 (explicit (config/TOML/kwarg)); set one of them"
        ),
        "{err}"
    );
}

#[test]
fn section_preset_with_a_disagreeing_mode_is_refused_naming_both() {
    let _g = device_lock();
    let out = run_toml(
        "sec_mode_conflict",
        &format!("{WATER}\n[gpu]\npreset = \"auto\"\nmode = \"off\"\n"),
    );
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains(
            "[gpu] preset = auto (explicit (config/TOML/kwarg)) implies mode = auto but \
             FERRIC_GPU / [gpu] mode = off (explicit (config/TOML/kwarg)); set one of them"
        ),
        "{err}"
    );
}

#[test]
fn section_preset_composes_with_the_orthogonal_keys() {
    let _g = device_lock();
    let out = run_toml(
        "sec_orthogonal",
        &format!("{WATER}\n[gpu]\npreset = \"off\"\ndevice = 0\nmemory_gb = 1.5\n"),
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("FERRIC_GPU_MEM_GB: 1.5  [source: explicit"),
        "{err}"
    );
    assert!(err.contains("FERRIC_GPU: off  [source: preset]"), "{err}");
}

#[test]
fn root_shorthand_and_a_gpu_table_together_are_refused_by_the_parser() {
    let _g = device_lock();
    let out = run_toml(
        "both_forms",
        &format!("gpu = \"off\"\n{WATER}\n[gpu]\npreset = \"off\"\n"),
    );
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("duplicate key"), "{err}");
    assert!(err.contains("gpu"), "{err}");
}

#[test]
fn unknown_preset_and_unknown_section_key_keep_their_messages() {
    let _g = device_lock();
    let err = stderr(&run_toml("bad", &format!("gpu = \"fast\"\n{WATER}")));
    assert!(
        err.contains(
            "[gpu] preset: invalid GPU preset \"fast\" (expected off, auto, on, mixed or auto-mixed)"
        ),
        "{err}"
    );
    let err = stderr(&run_toml(
        "badkey",
        &format!("{WATER}\n[gpu]\npreset = \"off\"\nmodee = \"on\"\n"),
    ));
    assert!(err.contains("unknown field `modee`"), "{err}");
    let err = stderr(&run_toml("badtype", &format!("gpu = 3\n{WATER}")));
    assert!(err.contains("gpu: expected a preset string"), "{err}");
}

#[test]
fn the_default_config_prints_the_unchanged_audit_lines() {
    let _g = device_lock();
    let out = run_toml("default", WATER);
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("FERRIC_GPU: off  [source: default]"), "{err}");
    assert!(
        err.contains("FERRIC_GPU_PRECISION: f64  [source: default]"),
        "{err}"
    );
    assert!(
        err.contains("FERRIC_GPU_PRESET: off  [source: default]"),
        "{err}"
    );
}
