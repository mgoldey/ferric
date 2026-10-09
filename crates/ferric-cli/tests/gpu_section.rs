//! The `[gpu]` section: documented keys only, refusals are errors not silent
//! CPU runs, and the audit line states the resolved mode.
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

/// Run the CLI on `body` with the FERRIC_GPU* env cleared so the TOML is the
/// only source.
fn run_toml(tag: &str, body: &str) -> std::process::Output {
    let root = workspace_root();
    let dir = root.join("target");
    std::fs::create_dir_all(&dir).expect("create target/ for the temp toml");
    let path = dir.join(format!("gpu_section_{tag}.toml"));
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

#[test]
fn gpu_mode_off_prints_the_audit_line_and_runs() {
    let _g = device_lock();
    let out = run_toml("gpu_off", &format!("{WATER}\n[gpu]\nmode = \"off\"\n"));
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("FERRIC_GPU: off  [source: explicit"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn gpu_section_on_a_non_gpu_build_is_refused() {
    if ferric_core::gpu::gpu_compiled() {
        eprintln!("skipping: this binary has the gpu feature");
        return;
    }
    let out = run_toml(
        "gpu_on_nofeature",
        &format!("{WATER}\n[gpu]\nmode = \"on\"\n"),
    );
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("built without the gpu feature"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn gpu_section_auto_on_a_non_gpu_build_notes_it_and_runs() {
    if ferric_core::gpu::gpu_compiled() {
        eprintln!("skipping: this binary has the gpu feature");
        return;
    }
    let out = run_toml(
        "gpu_auto_nofeature",
        &format!("{WATER}\n[gpu]\nmode = \"auto\"\n"),
    );
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("built without the gpu feature; running on the CPU"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn mode_on_without_a_device_is_an_error() {
    if !ferric_core::gpu::gpu_compiled() {
        eprintln!("skipping: no gpu feature");
        return;
    }
    let _g = device_lock();
    // Ordinal 99 never exists: this exercises the `on` refusal on every machine.
    let out = run_toml(
        "gpu_on_dev99",
        &format!("{WATER}\n[gpu]\nmode = \"on\"\ndevice = 99\n"),
    );
    assert!(!out.status.success());
    let e = stderr(&out);
    assert!(e.contains("mode = on") && e.contains("99"), "{e}");
    assert!(!e.contains("panicked"), "{e}");
}

#[test]
fn mode_auto_without_a_device_runs_and_says_so() {
    if !ferric_core::gpu::gpu_compiled() {
        eprintln!("skipping: no gpu feature");
        return;
    }
    let _g = device_lock();
    let out = run_toml(
        "gpu_auto_dev99",
        &format!("{WATER}\n[gpu]\nmode = \"auto\"\ndevice = 99\n"),
    );
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("gpu: unavailable") && stderr(&out).contains("running on the CPU"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn mode_on_with_a_real_device_installs_it() {
    if !ferric_core::gpu::gpu_compiled() {
        eprintln!("skipping: no gpu feature");
        return;
    }
    let _g = device_lock();
    let out = run_toml("gpu_on_dev0", &format!("{WATER}\n[gpu]\nmode = \"on\"\n"));
    if !out.status.success() && stderr(&out).contains("no usable device") {
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "no device but FERRIC_GPU_TESTS_REQUIRED=1: {}",
            stderr(&out)
        );
        eprintln!("skipping: no usable CUDA device");
        return;
    }
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stderr(&out).contains("gpu: device 0 "), "{}", stderr(&out));
}

#[test]
fn unknown_gpu_key_is_rejected() {
    let out = run_toml("gpu_badkey", &format!("{WATER}\n[gpu]\nmoed = \"on\"\n"));
    assert!(!out.status.success());
    assert!(stderr(&out).contains("moed"), "{}", stderr(&out));
}

#[test]
fn bad_gpu_values_are_rejected_by_name() {
    let out = run_toml(
        "gpu_badmode",
        &format!("{WATER}\n[gpu]\nmode = \"maybe\"\n"),
    );
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("[gpu] mode") && stderr(&out).contains("maybe"),
        "{}",
        stderr(&out)
    );
    let out = run_toml("gpu_badmem", &format!("{WATER}\n[gpu]\nmemory_gb = -1.0\n"));
    assert!(!out.status.success());
    assert!(stderr(&out).contains("[gpu] memory_gb"), "{}", stderr(&out));
}
