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

// ─── Keys a task path never reads (items 4 + 5) ─────────────────────────────

/// Pre-fix: H2/STO-3G, pdep-rpa, task = "optimize", `[rpa] xc = "PBE"`
/// converged to `final E = -1.1375270338 Hartree (RHF + RPA)` -- RPA@HF, the
/// functional silently dropped. Removing the `validate_task_compat` call
/// from `run()` makes this run (and succeed) again.
#[test]
fn rpa_optimize_with_a_ks_reference_is_refused() {
    let out = run_toml(
        "rpa_opt_xc",
        &body(
            "h2.xyz",
            1,
            "pdep-rpa",
            "optimize",
            "[rpa]\nauxbasis = \"cc-pvdz-ri\"\nxc = \"PBE\"\n",
        ),
    );
    assert_refused(&out, &["[rpa] xc", "optimize"]);
}

/// Pre-fix: water/STO-3G, rhf, task = "optimize", `k_builder = "cosx"` ran
/// to completion with COSX energies and an exact-exchange gradient (FD
/// mismatch -8.9e-6 Ha/Bohr on one H z at STO-3G, -1.4e-5 at cc-pVDZ).
/// Removing the cosx branch of `validate_task_compat` lets it run again. The
/// anchor is the same molecule with COSX on task = "energy", which must run.
#[test]
fn cosx_with_optimize_is_refused_but_a_cosx_energy_runs() {
    let cosx = "[scf]\nk_builder = \"cosx\"\n";
    let out = run_toml("cosx_opt", &body("h2.xyz", 1, "rhf", "optimize", cosx));
    assert_refused(&out, &["k_builder = \"cosx\"", "optimize"]);
    assert_runs(
        &run_toml("cosx_energy", &body("h2.xyz", 1, "rhf", "energy", cosx)),
        "an rhf COSX energy",
    );
}

// ─── [scf] df_j_aux / df_k_aux spellings (item 6) ───────────────────────────

/// Pre-fix: `[scf] df_j_aux = "exact"` failed with "unknown bundled basis:
/// exact", although Python `run_dft` reads that spelling as conventional J.
/// If `ScfCfg::df_*_aux_resolved` stops going through
/// `ferric_scf::rhf::normalize_df_aux`, this run fails the same way again.
#[test]
fn scf_df_aux_accepts_the_shared_opt_out_spellings() {
    let out = run_toml(
        "dfaux_exact",
        &body(
            "water.xyz",
            1,
            "rhf",
            "energy",
            "[scf]\ndf_j_aux = \"exact\"\ndf_k_aux = \"none\"\n",
        ),
    );
    assert_runs(&out, "rhf with df_j_aux = \"exact\"");
}

// ─── The J/K log line (item 7) ──────────────────────────────────────────────

/// Pre-fix: ksdft/PBE with `df_j_aux = ""` logged `[ferric] SCF J/K: RI-JK
/// via ` -- an empty basis name, on the one run that had turned density
/// fitting off. Reverting to the old `Some(aux) => "RI-JK via {aux}"` match
/// brings that line back and fails both assertions.
#[test]
fn jk_log_line_names_exact_coulomb_when_df_is_off() {
    let out = run_toml(
        "jk_log",
        &body(
            "water.xyz",
            1,
            "ksdft",
            "energy",
            "[dft]\nfunctional = \"PBE\"\n\n[scf]\ndf_j_aux = \"\"\n",
        ),
    );
    assert_runs(&out, "ksdft/PBE with df_j_aux = \"\"");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("[ferric] SCF J/K: exact J (four-centre); no K (pure functional)"),
        "{err}"
    );
    assert!(!err.contains("RI-JK via \n"), "{err}");
}
