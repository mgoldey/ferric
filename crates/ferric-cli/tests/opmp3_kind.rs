//! `method.kind = "opmp3"` through the real `ferric-cli` binary.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --test opmp3_kind
//! ```
//!
//! The OP-MP3 numbers are validated in `ferric-mp2` (`tests/opmp3*.rs`).
//! What the CLI can get wrong is the wiring: which reference runs, which
//! `[opmp3]` / `[scf]` key reaches which config field, which number is logged
//! as the total. These tests pin it on the full-precision `result` record of
//! the JSON run log:
//!
//! * `cli_equals_the_driver_bit_for_bit`: CH2 ³B1 / cc-pVDZ (the Q-Chem
//!   geometry of `testdata/reference/opmp3/qchem_cc-pvdz.json`), ROHF and
//!   ROKS/PBE0. `e_hf`, `e_nbs`, every model correlation energy and the
//!   logged total (`E_HF + E_NBS + E_c(SSD-OPMP3)`) equal `run_opmp3`
//!   called in-process with the same settings, bit for bit (same
//!   deterministic code, same rayon width). `[opmp3] x_ss = 0.3` reaches the
//!   SSD-OPMP3 number (it differs from the paper-default run).
//! * `cli_refusals_are_named`: every refused key or section names itself.
//!
//! # MUTATIONS (run 2026-11-15; each failed on an assertion)
//!
//! * `opmp3::resolve_plan` dropping `x_ss` (`x_ss: None`): the x_ss case of
//!   `cli_equals_the_driver_bit_for_bit` fails.
//! * `run_opmp3` logging `r.total(m.op_mp3)` as the total: the total check
//!   fails.
//! * the `[scf]` key filter accepting every key: `cli_refusals_are_named`
//!   fails on `[scf] df_j_aux`.

use std::path::{Path, PathBuf};
use std::process::Command;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_mp2::opmp3_driver::{run_opmp3, Opmp3Config, Opmp3Integrals, Opmp3RefKind};
use serde_json::Value;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

const CLI_THREADS: usize = 2;

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

fn ch2_xyz() -> String {
    let p = workspace_root().join("testdata/reference/opmp3/qchem_cc-pvdz.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    v["molecules"]["ch2"]["xyz"].as_str().unwrap().to_string()
}

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
    log: Vec<Value>,
}

fn run_cli(tag: &str, toml: &str) -> Run {
    let root = workspace_root();
    let xyz_path = root.join("target").join(format!("opmp3_{tag}.xyz"));
    let toml_path = root.join("target").join(format!("opmp3_{tag}.toml"));
    let log_path = root.join("target").join(format!("opmp3_{tag}.jsonl"));
    let _ = std::fs::remove_file(&log_path);
    std::fs::write(&xyz_path, ch2_xyz()).expect("write xyz");
    std::fs::write(
        &toml_path,
        toml.replace("{xyz}", &xyz_path.display().to_string()),
    )
    .expect("write toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml_path)
        .arg("--json")
        .arg(&log_path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", CLI_THREADS.to_string())
        .output()
        .expect("failed to run ferric-cli");
    let log = std::fs::read_to_string(&log_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        log,
    }
}

fn toml_with(opmp3: &str, extra: &str) -> String {
    format!(
        "[molecule]\nxyz = \"{{xyz}}\"\nmultiplicity = 3\n\n[basis]\nname = \"cc-pvdz\"\n\n\
         [method]\nkind = \"opmp3\"\n\n[opmp3]\n{opmp3}\n{extra}"
    )
}

#[test]
fn cli_equals_the_driver_bit_for_bit() {
    let mol = Molecule::parse_xyz(&ch2_xyz(), 0, 3).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(CLI_THREADS)
        .build()
        .unwrap();
    let mut ssd = vec![];
    for (tag, section, kind, x_ss) in [
        ("hf", "reference = \"hf\"", Opmp3RefKind::Hf, None),
        (
            "pbe0",
            "reference = \"pbe0\"",
            Opmp3RefKind::Dft("pbe0".into()),
            None,
        ),
        (
            "xss",
            "reference = \"pbe0\"\nx_ss = 0.3",
            Opmp3RefKind::Dft("pbe0".into()),
            Some(0.3),
        ),
    ] {
        let run = run_cli(tag, &toml_with(section, "[scf]\ndensity_conv = 1e-9\n"));
        assert!(run.ok, "{tag}:\n{}\n{}", run.stdout, run.stderr);
        let hits: Vec<&Value> = run.log.iter().filter(|r| r["record"] == "result").collect();
        assert_eq!(hits.len(), 1, "{tag}: {}", run.stderr);
        let res = hits[0];
        let cfg = Opmp3Config {
            x_ss,
            scf_density_conv: Some(1e-9),
            ..Opmp3Config::new(kind)
        };
        let lib = pool
            .install(|| {
                run_opmp3(
                    &ParallelContext::default(),
                    &mol,
                    &prep,
                    Opmp3Integrals::Exact,
                    &cfg,
                )
            })
            .unwrap();
        let comps = &res["components"];
        let get = |k: &str| {
            comps[k]
                .as_f64()
                .unwrap_or_else(|| panic!("{tag}: no {k} in {res}"))
        };
        assert_eq!(get("e_hf"), lib.e_hf, "{tag} e_hf");
        assert_eq!(get("e_nbs"), lib.e_nbs, "{tag} e_nbs");
        let m = &lib.models;
        for (k, e) in [
            ("MP2", m.mp2),
            ("MP3", m.mp3),
            ("P-MP3", m.p_mp3),
            ("OP-MP3", m.op_mp3),
            ("SOP-MP3", m.sop_mp3),
            ("SSD-OPMP3", m.ssd_opmp3),
        ] {
            assert_eq!(comps["correlation"][k].as_f64(), Some(e), "{tag} {k}");
        }
        assert_eq!(
            res["total"].as_f64(),
            Some(lib.total(m.ssd_opmp3)),
            "{tag} total"
        );
        assert!(run.stderr.contains("Smoke-grade"), "{tag}: {}", run.stderr);
        ssd.push(m.ssd_opmp3);
    }
    assert!(
        (ssd[1] - ssd[2]).abs() > 1e-6,
        "x_ss must move SSD-OPMP3: {ssd:?}"
    );
}

#[test]
fn cli_refusals_are_named() {
    let cases: &[(&str, String, &str)] = &[
        (
            "on_rhf",
            "[molecule]\nxyz = \"{xyz}\"\n[basis]\nname = \"cc-pvdz\"\n[method]\nkind = \"rhf\"\n\
             [opmp3]\nreference = \"hf\"\n"
                .to_string(),
            "[opmp3] is read by method.kind = \"opmp3\" only",
        ),
        (
            "scf_key",
            toml_with("", "[scf]\ndf_j_aux = \"cc-pvdz-ri\"\n"),
            "[scf] df_j_aux is not read",
        ),
        (
            "section",
            toml_with("", "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n"),
            "[mp2] is not read",
        ),
        (
            "task",
            toml_with("", "").replace("kind = \"opmp3\"", "kind = \"opmp3\"\ntask = \"optimize\""),
            "task = \"energy\" only",
        ),
        (
            "no_x_ss",
            toml_with("reference = \"b3lyp\"", ""),
            "prints no SSD-OPMP3 x_ss",
        ),
        (
            "integrals",
            toml_with("integrals = \"df\"", ""),
            "expected \"exact\" or \"ri\"",
        ),
        (
            "ri_no_aux",
            toml_with("integrals = \"ri\"", ""),
            "needs auxbasis",
        ),
        (
            "aux_exact",
            toml_with("auxbasis = \"cc-pvdz-ri\"", ""),
            "auxbasis is read with integrals = \"ri\" only",
        ),
        ("unknown", toml_with("xss = 0.5", ""), "unknown field `xss`"),
    ];
    for (tag, toml, want) in cases {
        let r = run_cli(&format!("refuse_{tag}"), toml);
        assert!(!r.ok, "{tag} should fail:\n{}", r.stdout);
        assert!(
            r.stderr.contains(want),
            "{tag}: wanted {want:?} in:\n{}",
            r.stderr
        );
        assert!(
            !r.log.iter().any(|x| x["record"] == "scf_iter"),
            "{tag}: refused after an SCF iteration"
        );
    }
}
