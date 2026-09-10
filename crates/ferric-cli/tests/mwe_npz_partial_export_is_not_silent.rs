//! MWE: a requested NPZ property that FAILS must not exit 0 with a silent gap.
//!
//! # The incident this encodes
//!
//! Every property arm in the `export_npz` path is
//! `match ... { Ok(v) => Some(v), Err(e) => { eprintln!("warning: .."); None } }`,
//! and the bundle is then written regardless with the missing fields as `None`.
//! There is no `process::exit` anywhere in that block, so the run finishes 0.
//!
//! The result is a well-formed NPZ missing `alpha_tensor` / `alpha_atomic` /
//! `c6_iso` / `c6_aniso`, and a caller that has no way to tell it apart from a
//! complete one except by re-reading stderr. During a 500-molecule QM9 feature
//! regeneration, that lost `alpha_atomic` on 476 of 500 molecules without a
//! single failing job — the budget gate refused the polarizability step, each
//! run warned once, and the bulk driver saw 500 successes.
//!
//! The same arm swallows the WRITE failure: a file the caller asked for and did
//! not get was also only a warning.
//!
//! # What is and is not being changed
//!
//! The per-property warnings stay — a partial bundle is sometimes genuinely
//! what a user wants, and forcing a rerun of an expensive SCF to get the
//! properties that DID work would be worse. What changes is that the run no
//! longer *reports success*: it exits nonzero, so a bulk driver notices, and
//! `[rpa] allow_partial_npz = true` is the explicit opt-in for callers who
//! accept gaps.
//!
//! Numerically inert: only the exit code and one config key. No computed
//! quantity moves.
//!
//! # What this test can and cannot reach — read before trusting it
//!
//! The failure STATE is not reachable from the shipped fixtures by
//! configuration alone, and this file says so rather than pretending
//! otherwise. Measured while writing it:
//!
//! * Tuning `[memory] budget_gb` does not isolate a property failure. The RPA
//!   preflight and the DF-dressing gate read the SAME budget and fire FIRST, so
//!   every value either fails the whole run before the property block
//!   (water/cc-pVDZ at 0.001 GiB: "DF dressing (in-core) requires 0.00 GB";
//!   benzene/cc-pVDZ at 0.05: "PDEP-RPA preflight .. requires 0.12 GB") or
//!   succeeds completely (water at 0.01-0.3, benzene at 0.2 — all exit 0 with a
//!   complete bundle). There is no window in between.
//! * The Z>18 route (`ts_free_atom` returns `None`, which the TS C6 branch
//!   warns-and-skips) is not reachable either: `aug-cc-pvdz-pp` has no Br
//!   shells in this build ("basis error: no basis shells for Z=35"), and
//!   HBr/aug-cc-pVDZ completes with `alpha_tensor`, `alpha_atomic`, `c6_iso`,
//!   `c6_aniso` and `alpha_atomic_dynamic` all PRESENT and zero warnings.
//!
//! So CONTRACTS 1 and 2 below are NOT the regression guard for this fix — they
//! are aspirational, `#[ignore]`d with the reason recorded, and should be
//! un-ignored the moment a fixture reaches the state (a molecule whose property
//! gate refuses while the RPA preflight passes). What actually guards the
//! change today is CONTRACT 3 and CONTRACT 4: that the new check does NOT fire
//! on a healthy run or on a deliberately disabled property. That is a weaker
//! claim than "the bug is fixed", and stating it plainly is the point — the
//! gap-tracking logic is reviewed-by-reading, not proven-by-test.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

/// Run the real binary on a generated config. `extra` goes into `[rpa]`.
///
/// The budget is set deliberately tiny so the polarizability/per-atom-alpha
/// gates refuse — that refusal is the situation under test, and it is exactly
/// what a too-small `[memory] budget_gb` produced in the incident.
fn run(tag: &str, budget_gb: &str, extra: &str) -> (bool, String, String) {
    let root = workspace_root();
    let toml_path = root.join("target").join(format!("mwe_npz_partial_{tag}.toml"));
    let npz_path = root.join("target").join(format!("mwe_npz_partial_{tag}.npz"));
    let _ = std::fs::remove_file(&npz_path);
    std::fs::write(
        &toml_path,
        format!(
            "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
             [basis]\nname = \"cc-pvdz\"\n\n\
             [method]\nkind = \"pdep-rpa\"\n\n\
             [mp2]\nauxbasis = \"cc-pvdz-ri\"\n\n\
             [memory]\nbudget_gb = {budget_gb}\n\n\
             [rpa]\nexport_npz = \"{}\"\n\
             compute_polarizability = true\n\
             compute_alpha_atomic = true\n\
             compute_c6 = false\n{extra}\n",
            npz_path.display()
        ),
    )
    .expect("write temp toml");
    let out = Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
        .arg(&toml_path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli binary");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// CONTRACT 1: when a requested property fails, the run must NOT exit 0.
///
/// This is the whole point. Before the fix the process warned once and returned
/// success, which is what made the 476/500 loss invisible to the bulk driver.
///
/// Note the test asserts on the exit code, not on stderr text: the warnings
/// were always there and a human reading one log would have seen them. What
/// failed was the machine-readable signal.
#[test]
#[ignore = "the failure state is not reachable from shipped fixtures: the RPA preflight and DF-dressing gates read the same budget and fire before the property block, so every budget either fails the whole run or succeeds completely (measurements in the module doc). Un-ignore when a fixture exists whose property gate refuses while the preflight passes."]
fn a_failed_requested_property_makes_the_run_fail() {
    // NOTE: this budget does NOT reach the intended state — it fails the RPA
    // preflight first (see the module doc). Kept as written so that whoever
    // un-ignores this test starts from the intended shape.
    let (ok, stdout, stderr) = run("fail", "0.000001", "");
    assert!(
        !ok,
        "a run whose REQUESTED NPZ properties failed reported SUCCESS. That is the \
         mechanism that lost alpha_atomic on 476 of 500 molecules without a failing job.\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    // And it must say which, so the failure is actionable rather than mysterious.
    assert!(
        stderr.contains("alpha") || stderr.contains("polarizability") || stderr.contains("α"),
        "the failure must name the property that did not make it into the bundle.\n\
         --- stderr ---\n{stderr}"
    );
}

/// CONTRACT 2: `allow_partial_npz = true` restores the old behaviour.
///
/// The escape hatch, and the reason CONTRACT 1 is safe to land: a caller who
/// genuinely wants whatever properties succeeded says so explicitly and keeps
/// exit 0. Without this the change would just break a workflow rather than
/// making it honest.
#[test]
#[ignore = "same unreachable state as a_failed_requested_property_makes_the_run_fail — with no way to induce a gap, there is nothing for allow_partial_npz to forgive."]
fn allow_partial_npz_opts_back_in_to_a_partial_bundle() {
    let (ok, stdout, stderr) = run("allow", "0.000001", "allow_partial_npz = true\n");
    assert!(
        ok,
        "with allow_partial_npz = true an incomplete bundle must be accepted (exit 0).\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}

/// CONTRACT 3 (the over-rejection guard): a run whose properties all SUCCEED
/// must still exit 0. **This is one of the two contracts that actually guards
/// this change** — see the module doc on reachability.
///
/// "An over-estimating guard is also a bug" — a check that failed every
/// export-npz run, or that keyed off the warnings' mere presence rather than an
/// actual requested-and-failed property, would be a regression dressed as a
/// fix. An ample budget must be indistinguishable from before.
#[test]
fn an_ample_budget_still_exports_and_exits_zero() {
    let (ok, stdout, stderr) = run("ample", "8.0", "");
    assert!(
        ok,
        "a run at an ample budget must still succeed — the new check must fire only on a \
         REQUESTED property that actually failed.\n--- stdout ---\n{stdout}\n\
         --- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("Wrote NPZ feature bundle"),
        "the bundle must still be written at an ample budget.\n--- stdout ---\n{stdout}"
    );
}

/// CONTRACT 4: a property that was NOT requested must not trip the check.
///
/// **The second of the two contracts that actually guards this change.**
///
/// The reachability trap. If the gate counted "any None field in the bundle"
/// rather than "a requested property that failed", then every run that left
/// `compute_c6 = false` would fail — and CONTRACT 3 would catch that only if
/// its fixture happened to request everything. Assert the distinction directly:
/// `compute_c6 = false` above means C6 is absent from every bundle here, and
/// the ample run must still pass.
#[test]
fn an_unrequested_property_is_not_a_failure() {
    let (ok, _stdout, stderr) = run("unrequested", "8.0", "compute_electric_field = false\n");
    assert!(
        ok,
        "switching a property OFF must not be treated as that property failing.\n\
         --- stderr ---\n{stderr}"
    );
}
