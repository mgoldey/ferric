//! Configs the CLI used to ACCEPT and then quietly answer a different
//! question for must now fail, with a message that says why.
//!
//! Each case was reproduced against the pre-fix release binary before it was
//! guarded (the measured output is quoted in each test). The pure config
//! logic is unit-tested in `config.rs` (`compat_guard_tests`); these tests pin
//! that `run()` actually calls it, and that the configurations the guards
//! must NOT touch still run.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Path to the `ferric-cli` binary, resolved at RUN TIME (a sibling of the
/// running test binary under nextest, the compile-time path under plain
/// `cargo test`). Same helper as `dispersion_guards.rs`; see there for why
/// a raw `env!` breaks under `cargo nextest archive`.
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

/// Workspace root at RUN time (same helper as `dispersion_guards.rs`).
fn workspace_root() -> PathBuf {
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

fn run_toml(tag: &str, body: &str) -> std::process::Output {
    let root = workspace_root();
    let path = root
        .join("target")
        .join(format!("silent_fallback_{tag}.toml"));
    std::fs::write(&path, body).expect("write temp toml");
    Command::new(ferric_cli_bin())
        .arg(&path)
        .arg("--no-json")
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("failed to run ferric-cli binary")
}

/// Molecule + STO-3G + `[method]`, followed by `extra` verbatim.
fn body(xyz: &str, multiplicity: usize, kind: &str, task: &str, extra: &str) -> String {
    format!(
        "[molecule]\nxyz = \"testdata/molecules/{xyz}\"\nmultiplicity = {multiplicity}\n\n\
         [basis]\nname = \"sto-3g\"\n\n\
         [method]\nkind = \"{kind}\"\ntask = \"{task}\"\n\n{extra}"
    )
}

fn assert_refused(out: &std::process::Output, needles: &[&str]) -> String {
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        !out.status.success(),
        "config must be REFUSED, but the run succeeded.\nstdout: {}\nstderr: {err}",
        String::from_utf8_lossy(&out.stdout)
    );
    for n in needles {
        assert!(err.contains(n), "error must mention {n:?}, got: {err}");
    }
    err
}

fn assert_runs(out: &std::process::Output, what: &str) {
    assert!(
        out.status.success(),
        "{what} must still RUN; the guard is too broad.\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ─── Open-shell molecule on a closed-shell-only kind (items 2 + 3) ─────────

/// Pre-fix: `error: SCF ladder failed: ScfConvergence { iterations: 0,
/// last_energy: 0.0 }` -- a convergence complaint for what is really
/// "kind = ksdft is closed-shell". Reverting the `validate_multiplicity` call
/// in `run()` brings that message back (the library backstop then reports
/// the multiplicity but not the Python/Rust UKS routes), failing the
/// "open-shell Kohn-Sham" check.
#[test]
fn ksdft_on_a_doublet_says_open_shell_ks_is_unavailable() {
    let out = run_toml(
        "ksdft_oh",
        &body(
            "oh.xyz",
            2,
            "ksdft",
            "energy",
            "[dft]\nfunctional = \"PBE\"\n",
        ),
    );
    let err = assert_refused(&out, &["multiplicity = 2", "open-shell Kohn-Sham"]);
    assert!(!err.contains("ScfConvergence"), "{err}");
}

/// Pre-fix: triplet water under `rimp2` printed `Total = -74.9987495795`,
/// the singlet's RI-MP2 energy, and exited 0. Reverting the CLI guard alone
/// leaves the library backstop's message ("multiplicity 3", no " = "), which
/// fails the first needle; reverting it AND `solve_rhf`'s
/// `require_closed_shell` makes the run succeed again.
#[test]
fn rimp2_on_triplet_water_is_refused_not_answered_as_the_singlet() {
    let out = run_toml(
        "rimp2_triplet",
        &body(
            "water.xyz",
            3,
            "rimp2",
            "energy",
            "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n",
        ),
    );
    assert_refused(&out, &["multiplicity = 3", "closed-shell"]);
}

/// Reachability anchor: the open-shell kinds and closed-shell singlets are
/// untouched.
#[test]
fn uhf_doublet_and_closed_shell_rimp2_still_run() {
    assert_runs(
        &run_toml("uhf_oh", &body("oh.xyz", 2, "uhf", "energy", "")),
        "uhf on the OH doublet",
    );
    assert_runs(
        &run_toml(
            "rimp2_singlet",
            &body(
                "water.xyz",
                1,
                "rimp2",
                "energy",
                "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n",
            ),
        ),
        "rimp2 on singlet water",
    );
}

// ─── [dft] keys the selected kind never reads (item 1) ──────────────────────

/// Pre-fix: `kind = "uhf"` + `[dft] functional = "PBE"` on OH printed
/// `energy = -74.3626375456`, the UHF energy (identical to the run without
/// the key), and exited 0. Removing the `validate_dft_section` call from
/// `run()` makes this run succeed again. The no-key uhf run in
/// `uhf_doublet_and_closed_shell_rimp2_still_run` is its anchor.
#[test]
fn uhf_with_a_dft_functional_is_refused_not_run_as_hf() {
    let out = run_toml(
        "uhf_pbe",
        &body(
            "oh.xyz",
            2,
            "uhf",
            "energy",
            "[dft]\nfunctional = \"PBE\"\n",
        ),
    );
    assert_refused(&out, &["[dft] functional", "\"uhf\"", "solve_uhf"]);
}
