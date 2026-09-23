//! MWE: `[memory] budget_gb` must reach the DFT grid AO cache.
//!
//! # The defect: a fix that was written, documented, and never wired up
//!
//! `ferric_dft::ks::KsXc::new_with_omega_budgeted` exists specifically to
//! carry a caller's budget into the grid cache, and its own doc states the
//! problem verbatim:
//!
//! ```text
//! /// The grid AO cache is the largest single allocation in a DFT job, and
//! /// the constructors resolved it with `resolve_budget_bytes(None)` — the
//! /// env / cgroup / RAM-auto-detected ceiling — which silently DISCARDS a
//! /// user's `[memory] budget_gb`. Setting `budget_gb = 4` on a 64 GB box
//! /// still sized the grid cache against ~51 GB. The guard message even
//! /// tells the user to raise `FERRIC_MEM_BUDGET_GB`, which works, while the
//! /// documented primary knob did not.
//! ```
//!
//! It had **zero production callers**. `rg new_with_omega_budgeted` returned
//! four hits, all inside `ks.rs` itself: two definitions and two self-calls
//! that hardcode `None`. Meanwhile the three SCF entry points each called the
//! UNbudgeted constructor:
//!
//! ```text
//! rhf.rs:463   KsXc::new_with_omega(mol, prep.basis_set(), name, &main, &nlc, config.xc_omega)
//! uhf.rs:124   KsXcUks::new_with_omega(..)
//! rohf.rs:203  KsXcUks::new_with_omega(..)
//! ```
//!
//! — while `config.three_index_budget_bytes` was in scope at all three, and is
//! exactly what the CLI populates (`ferric-cli/src/lib.rs:405`).
//!
//! So the documented primary knob did nothing for the largest allocation in a
//! DFT job, and only the env var worked.
//!
//! # What this test asserts, and why it is shaped this way
//!
//! Directly: that `[memory] budget_gb` alone — with **no env var set** —
//! changes the grid cache's behaviour. The env var must stay untouched,
//! because it worked before and would mask the defect entirely.
//!
//! The observable is the guard's own refusal. A budget far too small for the
//! grid must make a KS-DFT run FAIL with a budget message; before the fix it
//! ran happily, because the config value never reached the sizing decision.
//!
//! Run with `OPENBLAS_NUM_THREADS=1` per the project's rayon/BLAS convention.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Path to the `ferric-cli` binary, resolved at RUN TIME.
///
/// `env!("CARGO_BIN_EXE_ferric-cli")` is baked in at COMPILE time. Under
/// `cargo nextest archive` the binary is extracted to a fresh temporary
/// directory in the RUNNING job, and that constant still points at the BUILD
/// job's `target/debug/` -- which does not exist there. MEASURED: the archive
/// does carry the executable ("419 binaries, including 3 non-test binaries"),
/// so the failure is the stale path, not a missing file.
///
/// Prefer a sibling of the currently-running test binary
/// (`<extract-dir>/target/debug/deps/<test>` -> `../ferric-cli`), which is
/// where nextest puts it, then fall back to the compile-time path for plain
/// `cargo test`.
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

/// Run a KS-DFT job at the given `[memory] budget_gb`, with the budget env
/// vars explicitly CLEARED.
///
/// Clearing them is the crux: `FERRIC_MEM_BUDGET_GB` already reached the grid
/// cache before this fix, so leaving it set (or inherited from the CI
/// environment, which sets it to 3) would let the test pass against the
/// unfixed tree.
fn run_dft(tag: &str, budget_gb: &str) -> (bool, String, String) {
    let root = workspace_root();
    let path = root
        .join("target")
        .join(format!("mwe_dft_grid_budget_{tag}.toml"));
    std::fs::write(
        &path,
        format!(
            "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
             [basis]\nname = \"cc-pvdz\"\n\n\
             [method]\nkind = \"ksdft\"\n\n\
             [dft]\nfunctional = \"PBE\"\n\n\
             [memory]\nbudget_gb = {budget_gb}\n"
        ),
    )
    .expect("write temp toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env_remove("FERRIC_MEM_BUDGET_GB")
        .env_remove("FERRIC_OOC_BUDGET_GB")
        .env_remove("FERRIC_ERI3_BUDGET_GB")
        .output()
        .expect("failed to run ferric-cli binary");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// CONTRACT 1: a starvation `budget_gb` must CHANGE the grid storage mode.
///
/// # Observable (rewritten 2026-09 for the screened-batch XC path)
///
/// `KsXc` now integrates the main grid in screened spatial batches
/// (`ferric_dft::xc_batch`). The budget chooses only whether the per-batch AO
/// blocks stay resident or are recomputed every Fock build, and the two modes
/// are BIT-IDENTICAL by construction (pinned in `ks.rs`'s `storage_tests`).
/// The energy is therefore the wrong observable: this contract used to assert
/// that the ample and starvation energies DIFFER (the retired dense
/// `GridCache::Full` vs batched fallback differed by ~2.5e-8 Ha); that
/// difference is now zero on purpose, and a test that still compared energies
/// would pass or fail for reasons unrelated to the budget.
///
/// The observable is the mode itself, which a subprocess can only see through
/// the warning `KsXc` prints when the BUDGET (not the caller) forces
/// recompute: `ferric_dft::ks::RECOMPUTE_WARNING`. Starvation must print it;
/// the ample run must not (asserting both directions, so a warning printed
/// unconditionally, or never, fails).
///
/// # HONEST SCOPE — still does NOT distinguish the config wiring from the pool
///
/// `ferric-cli` installs a memory pool sized from the same `budget_gb`, and
/// `KsXc` sizes against the pool ledger when one is installed. So reverting
/// `rhf.rs`'s `new_with_omega_budgeted` wiring to `None` would still leave the
/// starvation run in recompute mode via the pool — the same limitation the
/// original energy-based version of this test recorded after mutation-testing
/// it. What this contract does establish is that `[memory] budget_gb` reaches
/// the grid storage decision end to end through the CLI.
#[test]
fn a_starvation_config_budget_reaches_the_grid_sizing_decision() {
    let (ample_ok, _ample_out, ample_err) = run_dft("ample_ref", "8.0");
    assert!(ample_ok, "reference run must succeed.\n{ample_err}");
    let (starve_ok, _starve_out, starve_err) = run_dft("starve", "0.00001");
    assert!(
        starve_ok,
        "a tiny budget must DEGRADE (recomputed grid AO blocks), not fail — the driver has \
         a graceful fallback by design.\n--- stderr ---\n{starve_err}"
    );
    let marker = ferric_dft::ks::RECOMPUTE_WARNING;
    assert!(
        starve_err.contains(marker),
        "a 1e-5 GB `[memory] budget_gb` must force the grid AO blocks into recompute mode \
         (warning `{marker}`), i.e. the config budget must reach the grid storage \
         decision.\n--- stderr ---\n{starve_err}"
    );
    assert!(
        !ample_err.contains(marker),
        "an 8 GB budget must keep water/cc-pVDZ's grid AO blocks resident, yet the \
         recompute warning was printed.\n--- stderr ---\n{ample_err}"
    );
}

/// CONTRACT 2 (the over-rejection guard): an ample `budget_gb` still runs.
///
/// "An over-estimating guard is also a bug." A change that refused every
/// KS-DFT job, or that mis-read the 0-means-unset sentinel and clamped
/// everything to nothing, would be a regression dressed as a fix.
#[test]
fn an_ample_config_budget_still_runs_to_completion() {
    let (ok, stdout, stderr) = run_dft("ample", "8.0");
    assert!(
        ok,
        "an 8 GiB [memory] budget_gb must still run water/cc-pVDZ PBE.\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("energy") || stdout.contains("Total"),
        "the ample run must actually report an energy.\n--- stdout ---\n{stdout}"
    );
}

/// CONTRACT 3: omitting `[memory]` entirely still auto-detects.
///
/// The 0-means-unset sentinel (`config.three_index_budget_bytes == 0`) must
/// keep meaning "resolve from env/auto", exactly as
/// `rhf::resolve_three_index_budget` already defines it. A fix that treated 0
/// as a literal zero-byte ceiling would refuse every default-configured DFT
/// run in the repo.
#[test]
fn an_absent_budget_section_still_auto_detects() {
    let root = workspace_root();
    let path = root.join("target").join("mwe_dft_grid_budget_absent.toml");
    std::fs::write(
        &path,
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
         [basis]\nname = \"cc-pvdz\"\n\n\
         [method]\nkind = \"ksdft\"\n\n\
         [dft]\nfunctional = \"PBE\"\n",
    )
    .unwrap();
    let out = Command::new(ferric_cli_bin())
        .arg(&path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env_remove("FERRIC_MEM_BUDGET_GB")
        .env_remove("FERRIC_OOC_BUDGET_GB")
        .env_remove("FERRIC_ERI3_BUDGET_GB")
        .output()
        .expect("failed to run ferric-cli binary");
    assert!(
        out.status.success(),
        "a config with no [memory] section must still run (0 means unset, not zero bytes).\n\
         --- stderr ---\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
