//! `method.kind = "ccsd"` must route a closed-shell (restricted) reference to
//! the SPIN-ADAPTED solver, not the spin-orbital one.
//!
//! Both solvers compute the same CCSD energy, but the spin-adapted one works in
//! spatial orbitals (no/nv) instead of spin orbitals (2no/2nv), so its O(N^6)
//! VVVV block is 16x smaller. On water/aug-cc-pVDZ that is 24.6 s -> 1.10 s
//! (22x) and peak RSS 1.82 GB -> 0.48 GB. The CLI previously called the
//! spin-orbital `ccsd()` unconditionally, so every closed-shell job paid the
//! larger cost for an identical answer.
//!
//! This pins BOTH halves: that the dispatch picks the fast path, and that the
//! energy it returns is still right. A dispatch test alone would pass if the
//! fast path were silently wrong.

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

fn run_toml(body: &str) -> std::process::Output {
    let root = workspace_root();
    let path = root.join("target").join("ccsd_dispatch_test.toml");
    std::fs::write(&path, body).expect("write temp toml");
    Command::new(ferric_cli_bin())
        .arg(&path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli binary")
}

#[test]
fn closed_shell_ccsd_uses_the_spin_adapted_solver() {
    // H2O / STO-3G: small enough that the spin-orbital path would also finish
    // quickly, so this test is about WHICH solver runs, not about wall time.
    let output = run_toml(
        r#"
[molecule]
xyz = "testdata/molecules/water.xyz"

[basis]
name = "sto-3g"

[method]
kind = "ccsd"

[mp2]
auxbasis = "cc-pvdz-ri"
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "ferric-cli ccsd failed:\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The header names the solver actually used.
    assert!(
        stdout.contains("spin-adapted"),
        "closed-shell CCSD must dispatch to the spin-adapted solver; got:\n{stdout}"
    );
    assert!(
        !stdout.contains("spin-orbital"),
        "closed-shell CCSD must NOT use the spin-orbital solver; got:\n{stdout}"
    );
    // And the solver itself announces which one converged.
    assert!(
        stdout.contains("closed-shell CCSD converged"),
        "expected the closed-shell solver's convergence line; got:\n{stdout}"
    );

    // Correctness guard: dispatching fast is worthless if the answer moved.
    // The reference is PySCF's exact-integral `cc.CCSD` on the same system,
    // -0.0495134885 Ha (generated 2026-07-26); ferric's RI-CCSD reproduces it
    // to 4.2e-7 Ha, which is the RI fitting floor. The 1e-5 tolerance is
    // therefore ~24x the observed RI error — tight enough that a wrong solver
    // (the two formulations differ by ~5e-5 at cc-pVDZ scale, and a genuinely
    // broken one by orders more) cannot slip through.
    let corr: f64 = stdout
        .lines()
        .find_map(|l| l.trim().strip_prefix("CCSD corr  = "))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("could not parse 'CCSD corr' from:\n{stdout}"));
    let pyscf_ref: f64 = -0.049_513_488_5; // PySCF cc.CCSD, exact integrals
    assert!(
        (corr - pyscf_ref).abs() < 1e-5,
        "H2O/STO-3G CCSD correlation energy {corr:.10} differs from the PySCF \
         reference {pyscf_ref:.10} by {:.3e} Ha (tolerance 1e-5) — the \
         spin-adapted dispatch may be returning a wrong energy, which no \
         timing or dispatch check would catch",
        (corr - pyscf_ref).abs(),
    );
}
