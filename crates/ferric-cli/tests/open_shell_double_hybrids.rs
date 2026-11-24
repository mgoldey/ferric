//! `b2plyp` and `dsd-pbep86` on an open-shell molecule (a UKS reference with
//! the unrestricted RI-MP2 spin components), and `[dft] dispersion = "d3bj"`
//! on both kinds.
//!
//! The numbers are held to PySCF in
//! `crates/ferric-mp2/tests/validation_udh_pyscf.rs` (library level, settings
//! matched). These tests pin what the CLI adds: the route is taken
//! (`reference = UKS`), the printed total is the printed pieces, the D3(BJ)
//! energy is ferric-d3's for the input geometry and is the only difference
//! between B2PLYP and B2PLYP-D3(BJ), DSD-PBEP86 with `d3bj` is the refit set
//! as a whole, and the combinations the guards must refuse are refused.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

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

const NH2: &str = "testdata/molecules/nh2.xyz";

fn body(kind: &str, task: &str, mult: usize, extra: &str) -> String {
    format!(
        "[molecule]\nxyz = \"{NH2}\"\nmultiplicity = {mult}\n\n[basis]\nname = \"sto-3g\"\n\n\
         [method]\nkind = \"{kind}\"\ntask = \"{task}\"\n\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n\n{extra}"
    )
}

fn run(tag: &str, text: &str) -> std::process::Output {
    let root = workspace_root();
    let path = root
        .join("target")
        .join(format!("udh_{tag}_{}.toml", std::process::id()));
    std::fs::write(&path, text).expect("write temp toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&path)
        .arg("--no-json")
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("failed to run ferric-cli");
    let _ = std::fs::remove_file(&path);
    out
}

fn run_ok(tag: &str, text: &str) -> std::process::Output {
    let out = run(tag, text);
    assert!(
        out.status.success(),
        "{tag} must run.\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn value(out: &std::process::Output, label: &str) -> f64 {
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with(label))
        .and_then(|l| l.split_once('=').map(|(_, rest)| rest))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no {label:?} line:\n{stdout}"))
}

fn stdout(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Printed to 10 decimals, so sums of printed pieces agree to a few 1e-10.
const PRINT: f64 = 5e-10;

#[test]
fn open_shell_double_hybrid_runs_uks_and_the_total_is_its_pieces() {
    for kind in ["b2plyp", "dsd-pbep86"] {
        let out = run_ok(&format!("uks_{kind}"), &body(kind, "energy", 2, ""));
        let text = stdout(&out);
        assert!(
            text.contains("reference    = UKS (mult = 2"),
            "{kind}:\n{text}"
        );
        let e_ks = value(&out, "E_KS");
        let corr = value(&out, "scaled corr");
        let total = value(&out, "Total");
        assert!(
            (e_ks + corr - total).abs() < PRINT,
            "{kind}: {e_ks} + {corr} != {total}"
        );
        let (e_os, e_ss) = (value(&out, "E_OS (raw)"), value(&out, "E_SS (raw)"));
        assert!(
            e_os < -1e-4 && e_ss < -1e-5,
            "{kind}: E_OS {e_os}, E_SS {e_ss}"
        );
        let (c_os, c_ss) = if kind == "b2plyp" {
            (0.27, 0.27)
        } else {
            (0.51, 0.36)
        };
        assert!(
            (c_os * e_os + c_ss * e_ss - corr).abs() < PRINT,
            "{kind}: the scaled correlation is not c_os E_OS + c_ss E_SS"
        );
    }
}

#[test]
fn b2plyp_d3bj_adds_ferric_d3s_energy_and_nothing_else() {
    let plain = run_ok("b2plyp_plain", &body("b2plyp", "energy", 2, ""));
    let d3 = run_ok(
        "b2plyp_d3bj",
        &body("b2plyp", "energy", 2, "[dft]\ndispersion = \"d3bj\"\n"),
    );
    let e_disp = value(&d3, "E_disp");
    assert!(e_disp < -1e-6, "dispersion energy {e_disp}");
    // The SCF and the PT2 term are the same run: only the geometry term adds.
    assert!((value(&d3, "E_KS") - value(&plain, "E_KS")).abs() < 1e-12);
    assert!((value(&d3, "E_OS (raw)") - value(&plain, "E_OS (raw)")).abs() < 1e-12);
    assert!(
        (value(&d3, "Total") - value(&plain, "Total") - e_disp).abs() < PRINT,
        "total difference is not E_disp"
    );
    // ... and E_disp is ferric-d3's energy of the input geometry with the
    // b2plyp row.
    let mol = ferric_core::mol::Molecule::load_xyz_with_charge(
        workspace_root().join(NH2).to_str().unwrap(),
        0,
        2,
    )
    .unwrap();
    let want = ferric_d3::d3bj_params_for_functional("b2plyp")
        .and_then(|p| ferric_d3::d3bj_energy_for_molecule(&mol, &p))
        .unwrap();
    assert!(
        (e_disp - want).abs() < PRINT,
        "E_disp {e_disp} vs ferric-d3 {want}"
    );
    assert!(stdout(&d3).contains("B2PLYP-D3(BJ)/"), "{}", stdout(&d3));
}

#[test]
fn dsd_pbep86_d3bj_is_the_refit_set_as_a_whole() {
    let plain = run_ok("dsd_plain", &body("dsd-pbep86", "energy", 2, ""));
    let d3 = run_ok(
        "dsd_d3bj",
        &body("dsd-pbep86", "energy", 2, "[dft]\ndispersion = \"d3bj\"\n"),
    );
    let text = stdout(&d3);
    assert!(text.contains("c_os=0.52, c_ss=0.22"), "{text}");
    assert!(stdout(&plain).contains("c_os=0.51, c_ss=0.36"));
    // The refit set has its own SCF (0.69 HF, not 0.72): E_KS moves by far
    // more than the D3 term could explain, which a "plain set plus the row"
    // implementation would leave at zero.
    assert!(
        (value(&d3, "E_KS") - value(&plain, "E_KS")).abs() > 1e-4,
        "E_KS did not change: the D3(BJ) set reused the no-dispersion functional"
    );
    let mol = ferric_core::mol::Molecule::load_xyz_with_charge(
        workspace_root().join(NH2).to_str().unwrap(),
        0,
        2,
    )
    .unwrap();
    let want = ferric_d3::d3bj_params_for_functional("dsdpbep86")
        .and_then(|p| ferric_d3::d3bj_energy_for_molecule(&mol, &p))
        .unwrap();
    assert!((value(&d3, "E_disp") - want).abs() < PRINT);
}

#[test]
fn double_hybrid_refusals() {
    // No nuclear gradient for the open-shell double hybrids.
    for task in ["optimize", "frequencies"] {
        let out = run(&format!("task_{task}"), &body("b2plyp", task, 2, ""));
        assert!(!out.status.success(), "{task} must be refused");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("multiplicity = 2"), "{task}: {err}");
    }
    // wB97X-L-V stays closed shell.
    let out = run("wb97_open", &body("wb97x-l-v", "energy", 2, ""));
    assert!(!out.status.success());
    // A named functional would substitute another fit; MBD has no DH beta.
    for spec in ["d3bj(pbe)", "mbd"] {
        let out = run(
            &format!("spec_{}", spec.replace(['(', ')'], "_")),
            &body(
                "b2plyp",
                "energy",
                2,
                &format!("[dft]\ndispersion = \"{spec}\"\n"),
            ),
        );
        assert!(!out.status.success(), "{spec} must be refused");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains("double hybrid") && err.contains("d3bj"),
            "{spec}: {err}"
        );
    }
}
