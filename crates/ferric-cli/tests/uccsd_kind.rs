//! `method.kind = "ccsd"` / `"ccsd(t)"` on an open-shell molecule through the
//! real `ferric-cli` binary.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --test uccsd_kind
//! ```
//!
//! The UCCSD and UCCSD(T) numbers are validated in `ferric-cc`
//! (`tests/uccsd_anchors.rs`, `tests/validation_uccsd.rs`). What the CLI can
//! get wrong is the wiring: which reference runs, which solver it reaches,
//! whether `[mp2] frozen_core` arrives, and which numbers are logged. These
//! tests pin it on the JSON run log of OH 2Pi / 6-31G:
//!
//! * `cli_uccsd_t_equals_the_library`: the UHF reference, the CCSD energy and
//!   its aa / ab / bb split, the (T) energy and its four classes, and the
//!   logged total equal `u_ccsd` + `u_ccsd_t` called in-process on a UHF of
//!   the same molecule (bar 1e-7 Ha: the two UHF solves converge separately).
//!   `ccsd` and `ccsd(t)` print the same CCSD energy, and frozen core 1 moves
//!   it (the key reaches the solver).
//! * `open_shell_refusals_are_named`: `ccd`, `eom-ccsd` and `ccsd` with
//!   `task = "optimize"` are refused on a doublet with a message naming what
//!   is available.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use ferric_cc::u_ccsd::u_ccsd;
use ferric_cc::u_ccsd_t::u_ccsd_t;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use serde_json::Value;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

/// See `ccsd_dispatch.rs`: resolved at RUN time for `cargo nextest archive`.
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

const OH_XYZ: &str = "testdata/molecules/validation/oh.xyz";

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
    log: Vec<Value>,
}

fn run_cli(toml: &str) -> Run {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let root = workspace_root();
    let n = CALLS.fetch_add(1, Ordering::Relaxed);
    let stem = format!("uccsd_kind_{}_{n}", std::process::id());
    let toml_path = root.join("target").join(format!("{stem}.toml"));
    let log_path = root.join("target").join(format!("{stem}.jsonl"));
    std::fs::write(&toml_path, toml).expect("write toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml_path)
        .arg("--json")
        .arg(&log_path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli");
    let log = std::fs::read_to_string(&log_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let _ = std::fs::remove_file(&toml_path);
    let _ = std::fs::remove_file(&log_path);
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        log,
    }
}

fn toml(kind: &str, task: &str, extra: &str) -> String {
    format!(
        "[molecule]\nxyz = \"{OH_XYZ}\"\nmultiplicity = 2\n\n[basis]\nname = \"6-31g\"\n\n\
         [method]\nkind = \"{kind}\"\ntask = \"{task}\"\n\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n{extra}"
    )
}

/// The `result` record's components.
fn components(run: &Run) -> (f64, Value) {
    assert!(
        run.ok,
        "ferric-cli failed:\nstdout:\n{}\nstderr:\n{}",
        run.stdout, run.stderr
    );
    let rec = run
        .log
        .iter()
        .find(|r| r["record"] == "result")
        .unwrap_or_else(|| panic!("no result record in the run log: {:?}", run.log));
    let total = rec["total"].as_f64().expect("total");
    (total, rec["components"].clone())
}

fn f(v: &Value, key: &str) -> f64 {
    v.pointer(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("component {key} missing in {v}"))
}

fn close(what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{what:<12} cli {got:+.12} library {want:+.12} |d| {d:.1e}");
    assert!(
        d < tol,
        "{what}: CLI {got:.12} vs library {want:.12} (|d| {d:.1e})"
    );
}

#[test]
fn cli_uccsd_t_equals_the_library() {
    let run = run_cli(&toml("ccsd(t)", "energy", ""));
    assert!(
        run.stdout.contains("UCCSD(T)/") && run.stdout.contains("spin-blocked"),
        "{}",
        run.stdout
    );
    assert!(
        run.stdout.contains("reference  = UHF, multiplicity 2"),
        "{}",
        run.stdout
    );
    let (total, c) = components(&run);

    let mol = Molecule::load_xyz_with_charge(workspace_root().join(OH_XYZ).to_str().unwrap(), 0, 2)
        .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let scf_cfg = RhfConfig::default()
        .with_density_conv(1e-10)
        .with_energy_conv(1e-11);
    let scf = solve_uhf(&ParallelContext::default(), &mol, &obs, &bounds, &scf_cfg).unwrap();
    let cfg = CcConfig::default();
    let cc = u_ccsd(&mol, &obs, &dfbs, op, &scf, &cfg).unwrap();
    let t = u_ccsd_t(&mol, &obs, &dfbs, op, &scf, &cc, &cfg).unwrap();

    let tol = 1e-7;
    close("E_UHF", f(&c, "/e_scf_reference"), scf.energy, tol);
    close("E_CCSD", f(&c, "/e_ccsd_corr"), cc.correlation_energy, tol);
    close("aa", f(&c, "/e_ccsd_aa"), cc.e_aa, tol);
    close("ab", f(&c, "/e_ccsd_ab"), cc.e_ab, tol);
    close("bb", f(&c, "/e_ccsd_bb"), cc.e_bb, tol);
    close("E_(T)", f(&c, "/e_t"), t.e_t, tol);
    close("aaa", f(&c, "/e_t_classes/aaa"), t.aaa, tol);
    close("aab", f(&c, "/e_t_classes/aab"), t.aab, tol);
    close("abb", f(&c, "/e_t_classes/abb"), t.abb, tol);
    close("bbb", f(&c, "/e_t_classes/bbb"), t.bbb, tol);
    close(
        "total",
        total,
        scf.energy + cc.correlation_energy + t.e_t,
        tol,
    );
    assert_eq!(c["reference"], "uhf");

    // `ccsd` prints the same CCSD energy and no (T).
    let run_cc = run_cli(&toml("ccsd", "energy", ""));
    assert!(run_cc.stdout.contains("UCCSD/"), "{}", run_cc.stdout);
    assert!(!run_cc.stdout.contains("(T) corr"), "{}", run_cc.stdout);
    let (_, c_cc) = components(&run_cc);
    close(
        "ccsd vs (t)",
        f(&c_cc, "/e_ccsd_corr"),
        f(&c, "/e_ccsd_corr"),
        1e-10,
    );

    // `[mp2] frozen_core` reaches the solver: the frozen-core run matches the
    // library at frozen_core = 1 and differs from the all-electron one.
    let run_fc = run_cli(&toml("ccsd", "energy", "frozen_core = 1\n"));
    assert!(
        run_fc.stdout.contains("frozen     = 1"),
        "{}",
        run_fc.stdout
    );
    let (_, c_fc) = components(&run_fc);
    let cc_fc = u_ccsd(
        &mol,
        &obs,
        &dfbs,
        op,
        &scf,
        &CcConfig {
            frozen_core: 1,
            ..Default::default()
        },
    )
    .unwrap();
    close(
        "E_CCSD fc1",
        f(&c_fc, "/e_ccsd_corr"),
        cc_fc.correlation_energy,
        tol,
    );
    assert!((cc_fc.correlation_energy - cc.correlation_energy).abs() > 1e-4);
}

#[test]
fn open_shell_refusals_are_named() {
    for (kind, task, needle) in [
        ("ccd", "energy", "kind = \"ccsd\" or \"ccsd(t)\""),
        ("eom-ccsd", "energy", "kind = \"ccsd\" or \"ccsd(t)\""),
        ("ccsd", "optimize", "task = \"energy\" only"),
        ("ccsd(t)", "frequencies", "task = \"energy\" only"),
    ] {
        let run = run_cli(&toml(kind, task, ""));
        assert!(
            !run.ok,
            "{kind}/{task} on a doublet must be refused:\n{}",
            run.stdout
        );
        assert!(
            run.stderr.contains(needle),
            "{kind}/{task}: refusal does not say `{needle}`:\n{}",
            run.stderr
        );
    }
}
