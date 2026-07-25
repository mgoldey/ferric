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

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
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
    Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
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
