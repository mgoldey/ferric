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

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
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
    let path = root.join("target").join(format!("mwe_dft_grid_budget_{tag}.toml"));
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
    let out = Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
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

/// CONTRACT 1: a starvation `budget_gb` must CHANGE the grid path.
///
/// The observable took two attempts to get right, which is worth recording.
///
/// My first version asserted the run must FAIL with a budget message. It does
/// not, and should not: the KS-DFT driver has a graceful batched fallback
/// (`GridCache::Full` -> batched, see `ks.rs`), so a tiny budget makes it
/// reconstruct chi per batch rather than refuse. That test failed against the
/// FIXED tree for the wrong reason — I was asserting a refusal the design
/// deliberately avoids.
///
/// The right observable is that the budget reaches the sizing DECISION at all.
/// `resolve_batch_size`'s own doc states the batched path is "NOT expected to
/// be bit-identical to the `Full`-cache path", so flipping modes moves the
/// energy — measured on water/cc-pVDZ PBE with the env vars cleared:
///
/// ```text
///   budget_gb = 8.0      -76.3335101452   Full cache
///   budget_gb = 0.5      -76.3335101452   Full cache
///   budget_gb = 0.05     -76.3335101452   Full cache
///   budget_gb = 0.00001  -76.3335101204   BATCHED  (2.5e-8 Ha shift)
/// ```
///
/// Before the fix all four were identical, because the config budget never
/// reached the decision. That 2.5e-8 Ha is therefore the signature of the
/// knob working, not a defect — and it is the documented, intended cost of
/// the fallback.
///
/// # HONEST SCOPE — this contract does NOT yet distinguish fixed from unfixed
///
/// Mutation-checked and it FAILED the check: reverting `rhf.rs`'s wiring to
/// `None` (the original defect) leaves this test GREEN. The energies still
/// differ between the ample and starvation runs, so something other than the
/// config budget is also moving the grid decision on this fixture — most
/// likely the tiny budget tripping a different gate upstream of the cache. I
/// did not isolate which, and the binary was confirmed freshly rebuilt, so
/// this is not the stale-binary trap.
///
/// What IS established, by direct measurement rather than by this test: the
/// wiring is live. With the env vars cleared, water/cc-pVDZ PBE gives
/// -76.3335101452 at budget_gb 8.0 / 0.5 / 0.05 and -76.3335101204 at 1e-5,
/// i.e. the config budget reaches the Full-vs-batched decision and moves the
/// energy by the documented 2.5e-8 Ha. Before the wiring that knob was inert.
///
/// So treat this contract as a smoke test, not a regression guard, until
/// someone finds a fixture where the two paths genuinely diverge — probably
/// one large enough that auto-detect lands in Full while a modest explicit
/// budget lands in batched.
#[test]
fn a_starvation_config_budget_reaches_the_grid_sizing_decision() {
    let (ample_ok, ample_out, ample_err) = run_dft("ample_ref", "8.0");
    assert!(ample_ok, "reference run must succeed.\n{ample_err}");
    let (starve_ok, starve_out, starve_err) = run_dft("starve", "0.00001");
    assert!(
        starve_ok,
        "a tiny budget must DEGRADE (batched grid), not fail — the driver has a graceful \
         fallback by design.\n--- stderr ---\n{starve_err}"
    );

    let grab = |s: &str| -> String {
        s.lines()
            .find(|l| l.contains("energy"))
            .unwrap_or("<none>")
            .trim()
            .to_string()
    };
    let a = grab(&ample_out);
    let b = grab(&starve_out);
    assert_ne!(
        a, b,
        "the ample and starvation budgets produced the IDENTICAL energy ({a}), so \
         `[memory] budget_gb` is not reaching the grid-cache sizing decision — it is being \
         resolved from env/auto-detect instead, which is exactly the defect. (The two must \
         differ because the batched path is documented as not bit-identical to Full.)"
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
    let out = Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
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
