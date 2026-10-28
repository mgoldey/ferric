//! The `[gpu]` precision keys: refusals are errors naming the keys, and the
//! audit lines state the resolved precision and allowlist.
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
    let path = dir.join(format!("gpu_precision_section_{tag}.toml"));
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

#[test]
fn precision_mixed_with_mode_off_is_an_error_naming_both_keys() {
    let _g = device_lock();
    let out = run_toml(
        "prec_mixed_off",
        &format!("{WATER}\n[gpu]\nmode = \"off\"\nprecision = \"mixed\"\n"),
    );
    assert!(!out.status.success(), "must refuse: {}", stderr(&out));
    let e = stderr(&out);
    assert!(e.contains("[gpu] precision = mixed requires mode"), "{e}");
}

#[test]
fn an_unknown_or_never_f32_kernel_name_is_refused_with_the_valid_list() {
    let _g = device_lock();
    let out = run_toml(
        "prec_bad_kernel",
        &format!(
            "{WATER}\n[gpu]\nmode = \"auto\"\nprecision = \"mixed\"\nmixed_kernels = [\"diis\"]\n"
        ),
    );
    assert!(!out.status.success(), "{}", stderr(&out));
    let e = stderr(&out);
    assert!(e.contains("diis") && e.contains("rimp2-energy"), "{e}");
}

#[test]
fn kernel_list_without_mixed_precision_is_refused() {
    let _g = device_lock();
    let out = run_toml(
        "prec_list_f64",
        &format!("{WATER}\n[gpu]\nmode = \"auto\"\nmixed_kernels = [\"rimp2-energy\"]\n"),
    );
    assert!(!out.status.success(), "{}", stderr(&out));
    assert!(stderr(&out).contains("mixed_kernels"), "{}", stderr(&out));
}

#[test]
fn precision_f64_prints_both_audit_lines_and_runs() {
    let _g = device_lock();
    let out = run_toml(
        "prec_f64",
        &format!("{WATER}\n[gpu]\nmode = \"off\"\nprecision = \"f64\"\n"),
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let e = stderr(&out);
    assert!(
        e.contains("FERRIC_GPU_PRECISION: f64  [source: explicit"),
        "{e}"
    );
    assert!(e.contains("FERRIC_GPU_MIXED_KERNELS: "), "{e}");
}

#[test]
fn cli_cfg_without_precision_resolves_to_precision_default() {
    use ferric_core::gpu::precision::PRECISION_DEFAULT;
    // Documented in the same file as the other CLI checks: an absent `precision`
    // key reaches the resolver as None, which resolves to PRECISION_DEFAULT.
    let _g = device_lock();
    let out = run_toml("prec_default", &format!("{WATER}\n[gpu]\nmode = \"off\"\n"));
    assert!(out.status.success(), "{}", stderr(&out));
    let e = stderr(&out);
    assert!(
        e.contains(&format!(
            "FERRIC_GPU_PRECISION: {PRECISION_DEFAULT}  [source: default]"
        )),
        "{e}"
    );
}

#[test]
fn shipping_rimp2_energy_leaves_the_default_f64_and_admits_mixed_with_a_device_mode() {
    let _g = device_lock();
    // default: f64, and the allowlist the build ships is printed
    let out = run_toml(
        "prec_shipped_default",
        &format!("{WATER}\n[gpu]\nmode = \"off\"\n"),
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let e = stderr(&out);
    assert!(
        e.contains("FERRIC_GPU_PRECISION: f64  [source: default]"),
        "{e}"
    );
    assert!(
        e.contains("FERRIC_GPU_MIXED_KERNELS: rimp2-energy  [source: default]"),
        "{e}"
    );
    // explicit mixed with mode off is still an error naming the keys
    let out = run_toml(
        "prec_shipped_off",
        &format!("{WATER}\n[gpu]\nmode = \"off\"\nprecision = \"mixed\"\n"),
    );
    assert!(!out.status.success(), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("[gpu] precision = mixed requires mode"),
        "{}",
        stderr(&out)
    );
    // mode auto + mixed + rimp2-energy: accepted; the run states `mixed` and the
    // kernel (device present), or degrades to the CPU in f64 with the notice
    // (no usable device / no gpu feature), never an error
    let out = run_toml(
        "prec_shipped_auto",
        &format!(
            "{WATER}\n[gpu]\nmode = \"auto\"\nprecision = \"mixed\"\nmixed_kernels = [\"rimp2-energy\"]\n"
        ),
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let e = stderr(&out);
    assert!(
        e.contains("FERRIC_GPU_PRECISION: mixed  [source: explicit"),
        "{e}"
    );
    assert!(
        e.contains("running on the CPU in f64")
            || e.contains("precision mixed, mixed kernels rimp2-energy"),
        "neither the degrade notice nor the device line states the precision: {e}"
    );
    // a kernel that is not shipped yet is refused, naming it and the shipped set
    let out = run_toml(
        "prec_shipped_ccsd",
        &format!(
            "{WATER}\n[gpu]\nmode = \"auto\"\nprecision = \"mixed\"\nmixed_kernels = [\"ccsd-amplitudes\"]\n"
        ),
    );
    assert!(!out.status.success(), "{}", stderr(&out));
    let e = stderr(&out);
    assert!(
        e.contains("ccsd-amplitudes not shipped yet")
            && e.contains("shipped in this build: rimp2-energy"),
        "{e}"
    );
}
