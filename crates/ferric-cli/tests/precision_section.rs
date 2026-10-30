//! The `[precision]` section and the precision provenance record in the run
//! log, end to end through the CLI binary. No device: the FERRIC_GPU* env is
//! cleared and `[gpu] mode` stays off.
//!
//! * No `[precision]`, `[precision] mode = "f64"` and the deprecated alias
//!   `[gpu] precision = "f64"` give bit-identical energies (the JSON log's
//!   shortest round-trip decimal, equal strings ⇔ equal bits) and every SCF /
//!   RI-MP2 record reads `{"precision": "f64", "kernels": []}`.
//! * Parsing is strict (unknown key, bad mode, bad / empty / comma-joined
//!   kernel list).
//! * The alias against the section: equal accepted (with the deprecation
//!   note), different refused naming both.
//! * CPU-only mixed (`mode = "mixed"` with the GPU off) is expressible but
//!   refused for every kernel this build admits, since none has a CPU mixed
//!   path; the message says so. The accept side is covered with an injected
//!   shipped set in `ferric-core/tests/precision_section_config.rs`.
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Run the CLI on `body` (FERRIC_GPU* cleared). With `json`, the run log is
/// written and returned as its records.
fn run(tag: &str, body: &str, json: bool) -> (std::process::Output, Vec<serde_json::Value>) {
    let root = workspace_root();
    let dir = root.join("target").join("precision_section_test");
    std::fs::create_dir_all(&dir).expect("create the temp dir");
    let toml = dir.join(format!("{tag}.toml"));
    let log = dir.join(format!("{tag}.jsonl"));
    let _ = std::fs::remove_file(&log);
    std::fs::write(&toml, body).expect("write temp toml");
    let mut cmd = Command::new(ferric_cli_bin());
    cmd.arg(&toml);
    if json {
        cmd.arg("--json").arg(&log);
    } else {
        cmd.arg("--no-json");
    }
    let out = cmd
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .env_remove("FERRIC_GPU")
        .env_remove("FERRIC_GPU_PRESET")
        .env_remove("FERRIC_GPU_DEVICE")
        .env_remove("FERRIC_GPU_MEM_GB")
        .env_remove("FERRIC_GPU_MIN_FLOPS")
        .env_remove("FERRIC_GPU_PRECISION")
        .env_remove("FERRIC_GPU_MIXED_KERNELS")
        .output()
        .expect("run ferric-cli");
    let records = if json && out.status.success() {
        std::fs::read_to_string(&log)
            .expect("read the run log")
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    } else {
        Vec::new()
    };
    (out, records)
}

fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn record<'a>(records: &'a [serde_json::Value], kind: &str) -> &'a serde_json::Value {
    records
        .iter()
        .find(|r| r["record"] == kind)
        .unwrap_or_else(|| panic!("no {kind} record in {records:?}"))
}

fn water(kind: &str, extra: &str) -> String {
    format!(
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n[basis]\nname = \"sto-3g\"\n\
         [method]\nkind = \"{kind}\"\n{extra}"
    )
}

fn f64_record() -> serde_json::Value {
    serde_json::json!({"precision": "f64", "kernels": []})
}

/// The energy as written (a string compare of serde_json's shortest
/// round-trip decimal is a bit compare).
fn energy_text(v: &serde_json::Value) -> String {
    v.to_string()
}

#[test]
fn an_f64_section_and_its_alias_are_bit_identical_to_no_section() {
    for (kind, dft) in [
        ("rhf", ""),
        ("ksdft", "[dft]\nfunctional = \"LDA\"\n"),
        ("rimp2", "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n"),
    ] {
        let arms = [
            ("none", String::new()),
            ("section", "[precision]\nmode = \"f64\"\n".to_string()),
            ("alias", "[gpu]\nprecision = \"f64\"\n".to_string()),
        ];
        let mut energies = Vec::new();
        for (arm, extra) in &arms {
            let tag = format!("bitid_{}_{arm}", kind.replace('-', "_"));
            let (out, recs) = run(&tag, &water(kind, &format!("{dft}{extra}")), true);
            assert!(out.status.success(), "{kind}/{arm}: {}", stderr(&out));
            let end = record(&recs, "run_end");
            assert_eq!(end["extra"]["precision"], f64_record(), "{kind}/{arm}");
            let mut e = vec![energy_text(&end["energy"])];
            if kind == "rimp2" {
                let res = record(&recs, "result");
                assert_eq!(res["components"]["precision"], f64_record(), "{kind}/{arm}");
                e.push(energy_text(&res["total"]));
            }
            if *arm == "alias" {
                assert!(
                    stderr(&out).contains("note: [gpu] precision is deprecated"),
                    "{}",
                    stderr(&out)
                );
            } else {
                assert!(!stderr(&out).contains("deprecated"), "{}", stderr(&out));
            }
            energies.push(e);
        }
        assert_eq!(
            energies[0], energies[1],
            "{kind}: [precision] mode = f64 moved a bit"
        );
        assert_eq!(
            energies[0], energies[2],
            "{kind}: [gpu] precision = f64 moved a bit"
        );
    }
}

#[test]
fn the_precision_section_is_parsed_strictly() {
    let cases = [
        ("unknown_key", "[precision]\nmodes = \"f64\"\n", "unknown field `modes`"),
        (
            "bad_mode",
            "[precision]\nmode = \"half\"\n",
            "[precision] mode: invalid GPU precision \"half\" (expected f64 or mixed)",
        ),
        (
            "bad_kernel",
            "[precision]\nmode = \"mixed\"\nmixed_kernels = [\"diis\"]\n",
            "[precision] mixed_kernels: unknown mixed-precision kernel \"diis\" (valid: rimp2-energy, ccsd-amplitudes, dfk-occ, cosx-kern, dfj-pack, xc-vxc)",
        ),
        (
            "empty_list",
            "[precision]\nmode = \"mixed\"\nmixed_kernels = []\n",
            "[precision] mixed_kernels: unknown mixed-precision kernel \"\"",
        ),
        (
            "comma_item",
            "[precision]\nmode = \"mixed\"\nmixed_kernels = [\"rimp2-energy,xc-vxc\"]\n",
            "[precision] mixed_kernels: kernel names are separate list items, got \"rimp2-energy,xc-vxc\"",
        ),
        (
            "not_a_table",
            "precision = \"mixed\"\n",
            "precision",
        ),
    ];
    for (tag, extra, want) in cases {
        let (out, _) = run(&format!("strict_{tag}"), &water("rhf", extra), false);
        assert!(!out.status.success(), "{tag} must be refused");
        assert!(stderr(&out).contains(want), "{tag}: {}", stderr(&out));
    }
}

#[test]
fn the_alias_must_agree_with_the_section() {
    let (out, _) = run(
        "alias_conflict_mode",
        &water(
            "rhf",
            "[gpu]\nmode = \"auto\"\nprecision = \"f64\"\n[precision]\nmode = \"mixed\"\n",
        ),
        false,
    );
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains(
            "[precision] mode = mixed and [gpu] precision = f64 disagree ([gpu] precision is a \
             deprecated alias of [precision] mode); set only [precision] mode"
        ),
        "{}",
        stderr(&out)
    );
    let (out, _) = run(
        "alias_conflict_kernels",
        &water(
            "rhf",
            "[gpu]\nmode = \"auto\"\nmixed_kernels = [\"rimp2-energy\", \"xc-vxc\"]\n\
             [precision]\nmode = \"mixed\"\nmixed_kernels = [\"rimp2-energy\"]\n",
        ),
        false,
    );
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("[precision] mixed_kernels = rimp2-energy and [gpu] mixed_kernels = rimp2-energy,xc-vxc disagree"),
        "{}",
        stderr(&out)
    );
    let (out, _) = run(
        "alias_agrees",
        &water(
            "rhf",
            "[gpu]\nprecision = \"f64\"\n[precision]\nmode = \"f64\"\n",
        ),
        false,
    );
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("note: [gpu] precision is deprecated"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn cpu_only_mixed_is_refused_for_device_kernels_with_the_reason() {
    let (out, _) = run(
        "cpu_only_default",
        &water("rhf", "[precision]\nmode = \"mixed\"\n"),
        false,
    );
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains(
            "[precision] mode = mixed requires mode = auto or on (FERRIC_GPU / [gpu] mode is \
             off): rimp2-energy runs only on a device; CPU mixed-precision kernels in this \
             build: none"
        ),
        "{}",
        stderr(&out)
    );
    let (out, _) = run(
        "cpu_only_cosx",
        &water(
            "rhf",
            "[precision]\nmode = \"mixed\"\nmixed_kernels = [\"cosx-kern\"]\n",
        ),
        false,
    );
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("[precision] mixed_kernels: cosx-kern not shipped yet"),
        "{}",
        stderr(&out)
    );
}
