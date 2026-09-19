//! `[dft] dispersion` must be REFUSED wherever it would be silently dropped.
//!
//! The correction is evaluated in exactly one place — `report_ksdft`, on the
//! `method.kind = "ksdft"`, `method.task = "energy"` path. Every other route
//! through the CLI reaches its result without ever reading the key. Accepting
//! the config there does not produce a wrong number so much as a MISLABELLED
//! one: the run prints an uncorrected energy from a file that asks for a
//! corrected one, and nothing in the output says the correction was skipped.
//!
//! Both guards are hard exits rather than warnings. A warning on stderr is not
//! a reliable channel — it is routinely redirected away in batch runs — and the
//! failure mode here is a number that looks entirely normal.
//!
//! There is no correct fallback to substitute either. D3(BJ)'s damping
//! parameters are fitted PER FUNCTIONAL, so "D3(BJ) for Hartree-Fock" is not a
//! defined quantity, and the D3(BJ) nuclear gradient is not implemented at all.
//!
//! ## What these tests would MISS
//!
//! They pin the refusal, not the correction. `d3bj_reference.json` and the
//! `ferric-d3` unit tests cover the values. A guard test passing tells you the
//! wrong configs are rejected — it says nothing about whether the accepted one
//! computes the right energy.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Path to the `ferric-cli` binary, resolved at RUN TIME.
///
/// Same trap as `workspace_root` below, different constant.
/// `env!("CARGO_BIN_EXE_ferric-cli")` is baked in at COMPILE time; under
/// `cargo nextest archive` the binary is EXTRACTED to a fresh temporary
/// directory in the running job while that constant still names the build
/// job's `target/debug/`. MEASURED: the archive DOES carry the executable
/// (`target/debug/ferric-cli` is in the tarball), so the failure is the stale
/// path, not a missing file -- which is why the error says NotFound and means
/// something else.
///
/// Prefer a sibling of the currently-running test binary, which is where
/// nextest puts it, then fall back to the compile-time path for plain
/// `cargo test`. Copied from `ccsd_dispatch.rs`, which already had both
/// helpers; writing this file with raw `env!` reintroduced a fixed bug.
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

/// Resolve the workspace root at RUN time, not compile time.
///
/// `env!("CARGO_MANIFEST_DIR")` is baked in when the test binary is COMPILED.
/// Under `cargo nextest archive` the binary is built in one job and run in
/// another whose checkout is elsewhere, so that path does not exist and
/// `Command::current_dir` fails with
///
///     failed to run ferric-cli binary: NotFound "No such file or directory"
///
/// which names the BINARY and means the CWD. This file hit exactly that on CI
/// after being written with the old helper; the repo already had the fix in
/// `ccsd_dispatch.rs` and this is the same shape. `--workspace-remap` does NOT
/// help -- it remaps cargo's view, not a string compiled into the test.
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
    // Correct under plain `cargo test`, where the compile and the run share a
    // checkout.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

fn run_toml(tag: &str, body: &str) -> std::process::Output {
    let root = workspace_root();
    let path = root.join("target").join(format!("disp_guard_{tag}.toml"));
    std::fs::write(&path, body).expect("write temp toml");
    Command::new(ferric_cli_bin())
        .arg(&path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli binary")
}

/// H2/STO-3G: two basis functions. These runs are expected to EXIT before any
/// SCF work, so the molecule only has to parse.
fn body(kind: &str, task: &str, dispersion: bool) -> String {
    let disp = if dispersion {
        "dispersion = \"d3bj(pbe)\"\n"
    } else {
        ""
    };
    format!(
        "[molecule]\nxyz = \"testdata/molecules/h2.xyz\"\n\n\
         [basis]\nname = \"sto-3g\"\n\n\
         [method]\nkind = \"{kind}\"\ntask = \"{task}\"\n\n\
         [dft]\nfunctional = \"PBE\"\n{disp}"
    )
}

#[test]
fn dispersion_is_refused_for_a_non_ksdft_method() {
    // An `rhf` ENERGY run: it clears the task guard, so this is reachable only
    // through the method guard. Before that guard existed this run succeeded
    // and printed a plain HF energy.
    let out = run_toml("rhf", &body("rhf", "energy", true));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "an rhf run with [dft] dispersion must FAIL, not silently drop the \
         correction.\nstdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        err.contains("method.kind") && err.contains("ksdft"),
        "the error must name the offending key and the supported value, got: {err}"
    );
}

#[test]
fn dispersion_works_for_optimize_and_is_refused_for_frequencies() {
    // `optimize` USED TO BE REFUSED, because the D3(BJ) nuclear gradient did
    // not exist. It does now, and it is threaded through
    // `optimize_geometry_with_correction`, so the energy and the gradient
    // describe the same surface and the run must SUCCEED.
    //
    // MEASURED end to end, Ar2/PBE/STO-3G from a short start: 6.7917 Bohr
    // uncorrected vs 6.6418 with d3bj -- the attraction shortens the bond,
    // which is the direction that says the correction reached the GRADIENT
    // and not only the energy.
    let out = run_toml("opt", &body("ksdft", "optimize", true));
    assert!(
        out.status.success(),
        "a task=optimize run with [dft] dispersion must now SUCCEED -- the \
         analytic D3(BJ) gradient is implemented.\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // `frequencies` is STILL refused, and the refusal is narrower than it was:
    // the gradient exists, but the finite-difference Hessian built from it has
    // never been validated against anything. Refusing an unvalidated number is
    // the point -- it would be 6N extra SCF+D3 evaluations producing a Hessian
    // nobody has checked.
    let freq = run_toml("freq", &body("ksdft", "frequencies", true));
    let err = String::from_utf8_lossy(&freq.stderr);
    assert!(
        !freq.status.success(),
        "task=frequencies with dispersion must still FAIL.\nstdout: {}",
        String::from_utf8_lossy(&freq.stdout)
    );
    assert!(
        err.contains("frequencies") && err.contains("validated"),
        "the error must name the task AND say why it is refused (the Hessian \
         is unvalidated, not that the gradient is missing), got: {err}"
    );
}

/// THE EXACTNESS ANCHOR for both guards above.
///
/// Without this, a guard that rejected EVERYTHING would pass both tests. The
/// pass condition has to be reachable: the one supported combination must still
/// run, and the two unsupported ones must still fail WITHOUT the key.
#[test]
fn the_supported_combination_still_runs_and_the_key_is_what_refuses() {
    // ksdft + energy + dispersion: the one accepted configuration.
    let ok = run_toml("ok", &body("ksdft", "energy", true));
    assert!(
        ok.status.success(),
        "ksdft/energy with dispersion must RUN; the guards rejected the only \
         supported combination.\nstderr: {}",
        String::from_utf8_lossy(&ok.stderr)
    );

    // ...and the same two rejected configs must SUCCEED once the key is gone,
    // which pins that `dispersion` is what refuses them and not the method or
    // task by itself.
    for (tag, kind, task) in [
        ("rhf_nodisp", "rhf", "energy"),
        ("opt_nodisp", "ksdft", "optimize"),
    ] {
        let out = run_toml(tag, &body(kind, task, false));
        assert!(
            out.status.success(),
            "{kind}/{task} must run fine WITHOUT the dispersion key; if it \
             fails here the guard tests above prove nothing.\nstderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
