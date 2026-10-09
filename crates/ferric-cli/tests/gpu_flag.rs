//! `ferric --gpu <preset>`: the flag is the top source (flag > TOML preset >
//! env preset > default), a fine key that disagrees is refused naming both,
//! and a run without the flag is unchanged. Device-free: the `on` and `mixed`
//! arms are asserted only where no device can be needed (a refusal, or a build
//! without the gpu feature).
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard};

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

const WATER: &str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
"#;

fn base_command() -> Command {
    let mut c = Command::new(ferric_cli_bin());
    c.current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .env_remove("FERRIC_GPU")
        .env_remove("FERRIC_GPU_PRESET")
        .env_remove("FERRIC_GPU_PRECISION")
        .env_remove("FERRIC_GPU_MIXED_KERNELS")
        .env_remove("FERRIC_GPU_DEVICE")
        .env_remove("FERRIC_GPU_MEM_GB")
        .env_remove("FERRIC_GPU_MIN_FLOPS");
    c
}

fn write_toml(tag: &str, body: &str) -> PathBuf {
    let dir = workspace_root().join("target");
    std::fs::create_dir_all(&dir).expect("create target/ for the temp toml");
    let path = dir.join(format!("gpu_flag_{tag}.toml"));
    std::fs::write(&path, body).expect("write temp toml");
    path
}

/// Run the CLI on `body` with `flags` placed before the TOML path.
fn run(tag: &str, body: &str, flags: &[&str], env: &[(&str, &str)]) -> std::process::Output {
    let path = write_toml(tag, body);
    base_command()
        .args(flags)
        .arg(&path)
        .arg("--no-json")
        .envs(env.iter().copied())
        .output()
        .expect("failed to run ferric-cli binary")
}

fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn stdout(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn gpu_lines(s: &str) -> Vec<String> {
    s.lines()
        .filter(|l| l.contains("FERRIC_GPU"))
        .map(str::to_string)
        .collect()
}

const NO_FEATURE_NOTICE: &str = "gpu: built without the gpu feature; running on the CPU in f64";

#[test]
fn flag_off_labels_the_preset_command_line_and_runs() {
    let _g = device_lock();
    let out = run("off", WATER, &["--gpu", "off"], &[]);
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("FERRIC_GPU_PRESET: off  [source: command line]"),
        "{err}"
    );
    assert!(err.contains("FERRIC_GPU: off  [source: preset]"), "{err}");
}

#[test]
fn equals_form_and_flag_after_the_path_are_the_same() {
    let _g = device_lock();
    let a = stderr(&run("eq", WATER, &["--gpu=auto"], &[]));
    let path = write_toml("eq", WATER);
    let out = base_command()
        .arg(&path)
        .arg("--gpu")
        .arg("auto")
        .arg("--no-json")
        .output()
        .expect("run");
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(gpu_lines(&a), gpu_lines(&stderr(&out)));
    assert!(
        a.contains("FERRIC_GPU_PRESET: auto  [source: command line]"),
        "{a}"
    );
    assert!(a.contains("FERRIC_GPU: auto  [source: preset]"), "{a}");
}

#[test]
fn auto_presets_degrade_with_the_one_notice_when_the_build_has_no_gpu() {
    let _g = device_lock();
    for p in ["auto", "auto-mixed"] {
        let out = run(&format!("notice_{p}"), WATER, &["--gpu", p], &[]);
        assert!(out.status.success(), "--gpu {p}: {}", stderr(&out));
        let err = stderr(&out);
        assert!(
            err.contains(&format!("FERRIC_GPU_PRESET: {p}  [source: command line]")),
            "{err}"
        );
        if !cfg!(feature = "gpu") {
            assert_eq!(err.matches(NO_FEATURE_NOTICE).count(), 1, "{err}");
        }
    }
}

#[cfg(not(feature = "gpu"))]
#[test]
fn on_and_mixed_are_refused_on_a_build_without_the_gpu_feature() {
    let _g = device_lock();
    for p in ["on", "mixed"] {
        let out = run(&format!("nofeat_{p}"), WATER, &["--gpu", p], &[]);
        assert_eq!(out.status.code(), Some(1), "--gpu {p}");
        let err = stderr(&out);
        assert!(err.contains("error: "), "{err}");
        assert!(err.contains("feature"), "--gpu {p}: {err}");
    }
}

#[test]
fn flag_beats_a_toml_preset_root_and_section_forms() {
    let _g = device_lock();
    let out = run(
        "beats_root",
        &format!("gpu = \"on\"\n{WATER}"),
        &["--gpu", "off"],
        &[],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("FERRIC_GPU_PRESET: off  [source: command line]"),
        "{err}"
    );
    assert!(err.contains("FERRIC_GPU: off  [source: preset]"), "{err}");
    let out = run(
        "beats_section",
        &format!("{WATER}\n[gpu]\npreset = \"mixed\"\n"),
        &["--gpu=off"],
        &[],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("FERRIC_GPU_PRESET: off  [source: command line]"),
        "{err}"
    );
}

#[test]
fn flag_beats_an_env_preset() {
    let _g = device_lock();
    let out = run(
        "beats_env",
        WATER,
        &["--gpu", "off"],
        &[("FERRIC_GPU_PRESET", "on")],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("FERRIC_GPU_PRESET: off  [source: command line]"),
        "{err}"
    );
}

#[test]
fn flag_disagreeing_with_a_toml_mode_is_refused_naming_both() {
    let _g = device_lock();
    let out = run(
        "mode_conflict",
        &format!("{WATER}\n[gpu]\nmode = \"on\"\n"),
        &["--gpu", "off"],
        &[],
    );
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains(
            "error: [gpu] preset = off (command line) implies mode = off but \
             FERRIC_GPU / [gpu] mode = on (explicit (config/TOML/kwarg)); set one of them"
        ),
        "{err}"
    );
}

#[test]
fn an_overriding_flag_is_checked_against_the_fine_keys_of_the_overridden_preset() {
    let _g = device_lock();
    // TOML: preset auto + mode auto (consistent). The flag overrides the
    // preset to off, so the now-disagreeing mode key is refused naming the flag.
    let out = run(
        "override_then_conflict",
        &format!("{WATER}\n[gpu]\npreset = \"auto\"\nmode = \"auto\"\n"),
        &["--gpu", "off"],
        &[],
    );
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains(
            "error: [gpu] preset = off (command line) implies mode = off but \
             FERRIC_GPU / [gpu] mode = auto (explicit (config/TOML/kwarg)); set one of them"
        ),
        "{err}"
    );
}

#[test]
fn flag_disagreeing_with_env_mode_and_env_precision_is_refused_naming_both() {
    let _g = device_lock();
    let out = run(
        "env_mode",
        WATER,
        &["--gpu", "off"],
        &[("FERRIC_GPU", "auto")],
    );
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains(
            "error: [gpu] preset = off (command line) implies mode = off but \
             FERRIC_GPU / [gpu] mode = auto (env); set one of them"
        ),
        "{err}"
    );
    let out = run(
        "env_prec",
        WATER,
        &["--gpu", "auto-mixed"],
        &[("FERRIC_GPU_PRECISION", "f64")],
    );
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains(
            "error: [gpu] preset = auto-mixed (command line) implies precision = mixed but \
             FERRIC_GPU_PRECISION / [gpu] precision = f64 (env); set one of them"
        ),
        "{err}"
    );
}

#[test]
fn flag_disagreeing_with_a_toml_precision_is_refused_naming_both() {
    let _g = device_lock();
    let out = run(
        "prec_conflict",
        &format!("{WATER}\n[gpu]\nprecision = \"f64\"\n"),
        &["--gpu", "auto-mixed"],
        &[],
    );
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains(
            "error: [gpu] preset = auto-mixed (command line) implies precision = mixed but \
             FERRIC_GPU_PRECISION / [gpu] precision = f64 (explicit (config/TOML/kwarg)); set one of them"
        ),
        "{err}"
    );
}

#[test]
fn flag_agreeing_with_a_fine_key_is_kept() {
    let _g = device_lock();
    let out = run(
        "agree",
        &format!("{WATER}\n[gpu]\nmode = \"off\"\n"),
        &["--gpu", "off"],
        &[],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("FERRIC_GPU: off  [source: explicit (config/TOML/kwarg)]"),
        "{err}"
    );
    assert!(
        err.contains("FERRIC_GPU_PRESET: off  [source: command line]"),
        "{err}"
    );
}

#[test]
fn bad_flag_usage_is_refused_with_exit_code_2() {
    let _g = device_lock();
    let missing = "error: --gpu requires a preset (off, auto, on, mixed or auto-mixed)";
    let out = run("missing", WATER, &["--gpu", "--verbose"], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains(missing), "{}", stderr(&out));
    let out = base_command()
        .arg("x.toml")
        .arg("--gpu")
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains(missing), "{}", stderr(&out));
    let out = run("twice", WATER, &["--gpu", "off", "--gpu=auto"], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(
        stderr(&out).contains("error: --gpu given more than once; give one preset"),
        "{}",
        stderr(&out)
    );
    let out = run("unknown", WATER, &["--gpu", "fast"], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(
        stderr(&out).contains(
            "error: --gpu: invalid GPU preset \"fast\" (expected off, auto, on, mixed or auto-mixed)"
        ),
        "{}",
        stderr(&out)
    );
}

#[test]
fn help_lists_the_flag_and_version_is_untouched() {
    let out = base_command().arg("--help").output().expect("run");
    assert!(out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("--gpu <preset>  off, auto, on, mixed"),
        "{err}"
    );
    assert!(err.contains("[--gpu <preset>] <input.toml>"), "{err}");
    let out = base_command().arg("--version").output().expect("run");
    assert!(out.status.success());
    assert!(!stdout(&out).contains("gpu"), "{}", stdout(&out));
}

#[test]
fn without_the_flag_the_run_is_unchanged() {
    let _g = device_lock();
    let plain = run("plain", WATER, &[], &[]);
    let off = run("plain_off", WATER, &["--gpu", "off"], &[]);
    assert!(plain.status.success() && off.status.success());
    let err = stderr(&plain);
    assert!(!err.contains("command line"), "{err}");
    assert!(err.contains("FERRIC_GPU: off  [source: default]"), "{err}");
    assert!(
        err.contains("FERRIC_GPU_PRESET: off  [source: default]"),
        "{err}"
    );
    // --gpu off changes only the source labels, never the numbers.
    assert_eq!(stdout(&plain), stdout(&off));
}
