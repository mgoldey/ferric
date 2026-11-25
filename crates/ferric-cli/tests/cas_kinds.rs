//! `method.kind = "casscf"` and `"nevpt2"` through the real `ferric-cli`
//! binary, on the systems PySCF references exist for.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --test cas_kinds
//! ```
//!
//! The CASSCF and SC-NEVPT2 numbers are validated in `ferric-ci`
//! (`casscf_vs_pyscf.rs`, `nevpt2.rs`). What the CLI can get wrong is the
//! wiring: which RHF starts the run, which `[cas]` / `[cas.ci]` / `[scf]` key
//! reaches which config field, and which number is logged as the total.
//! These tests pin it by reading the full-precision `result` record of the
//! JSON run log:
//!
//! * `cli_casscf_equals_the_driver_bit_for_bit`: H2O CAS(4,4), N2 CAS(6,6)
//!   at 1.0977 and 1.6 Å and O2 CAS(8,6) in the `(5, 3)` sector, 6-31G
//!   (`testdata/reference/casscf_*_pyscf.json`). The logged total equals
//!   `ferric_ci::driver::rhf_start` + `casscf_on` called in-process with
//!   the same settings, bit for bit (both run the same deterministic code on
//!   the same rayon width), and it lies within 1e-9 Ha of PySCF (measured
//!   gaps are at most 1.5e-11 Ha; the derived bar is in `casscf_vs_pyscf.rs`).
//!   The Python `run_casscf` runs the same driver; its bit-for-bit check
//!   against this binary is `crates/ferric-python/tests/test_cas_cli.py`.
//! * `cli_nevpt2_equals_the_driver_bit_for_bit`: H2O CAS(4,4), N2 CAS(6,6)
//!   at 1.0977 and 2.2 Å (`testdata/reference/nevpt2_refs.json`), the CASSCF
//!   gradient tolerance of the reference file passed as `[cas] grad_tol`.
//!   Every class energy and the total equal `nevpt2_on` in-process bit
//!   for bit, and each class lies within the PySCF bar of `nevpt2.rs`.
//! * `cli_keys_reach_the_config`: settings that move the answer reach it
//!   (`[cas] max_iter = 2` leaves CASSCF unconverged, warned and not
//!   graded; `[cas] stability` is logged and `"off"` drops it; `[cas] reference = "casci"` gives the CAS-CI reference energy
//!   and a capped grade; `[cas] n_core` moves the window).
//! * `cli_refusals_are_named`: every refused key or section names itself
//!   and fails before any SCF iteration.
//!
//! # Mutation ledger (each measured to fail)
//!
//! * `resolve_plan` dropping `[cas.ci] conv_thresh` (default kept): the
//!   bit-for-bit tests pass (both sides default) but `cli_keys_reach_the_config`
//!   fails on its CI-threshold case.
//! * `run_cas` passing the CAS-CI reference energy as the NEVPT2 total
//!   (`r.e_reference` alone): `cli_nevpt2_equals_the_driver_bit_for_bit` fails.
//! * the `[scf]` key filter accepting every key: `cli_refusals_are_named`
//!   fails on `[scf] df_j_aux`.

use std::path::{Path, PathBuf};
use std::process::Command;

use ferric_ci::driver::{casscf_on, nevpt2_on, rhf_start, Nevpt2On, ScfStart};
use ferric_ci::{CasScfConfig, Nevpt2Classes};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_scf::PreparedSystem;
use serde_json::Value;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

/// Rayon width of the CLI runs AND of the in-process library side, so the two
/// run the same reduction order.
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

fn read_json(rel: &str) -> Value {
    let path = workspace_root().join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// One CLI run: `(success, stdout, stderr, run-log records)`.
struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
    log: Vec<Value>,
}

impl Run {
    fn record(&self, name: &str) -> &Value {
        let hits: Vec<&Value> = self.log.iter().filter(|r| r["record"] == name).collect();
        assert_eq!(hits.len(), 1, "{name} records: {hits:?}\n{}", self.stderr);
        hits[0]
    }
    fn result(&self) -> &Value {
        self.record("result")
    }
    fn scf_iterations(&self) -> usize {
        self.log
            .iter()
            .filter(|r| r["record"] == "scf_iter" || r["record"] == "guess_scf_iter")
            .count()
    }
}

/// Run the binary on `toml` (the molecule written to `target/cas_<tag>.xyz`
/// from `xyz` and substituted for `{xyz}`).
fn run_cli(tag: &str, xyz: &str, toml: &str) -> Run {
    run_cli_env(tag, xyz, toml, &[])
}

/// [`run_cli`] with `env` set on the binary (`None`: removed).
fn run_cli_env(tag: &str, xyz: &str, toml: &str, env: &[(&str, Option<&str>)]) -> Run {
    let root = workspace_root();
    let xyz_path = root.join("target").join(format!("cas_{tag}.xyz"));
    let toml_path = root.join("target").join(format!("cas_{tag}.toml"));
    let log_path = root.join("target").join(format!("cas_{tag}.jsonl"));
    let _ = std::fs::remove_file(&log_path);
    std::fs::write(&xyz_path, xyz).expect("write xyz");
    let body = toml.replace("{xyz}", &xyz_path.display().to_string());
    std::fs::write(&toml_path, body).expect("write toml");
    let mut cmd = Command::new(ferric_cli_bin());
    cmd.arg(&toml_path)
        .arg("--json")
        .arg(&log_path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", CLI_THREADS.to_string());
    for (k, v) in env {
        match v {
            Some(v) => cmd.env(k, v),
            None => cmd.env_remove(k),
        };
    }
    let out = cmd.output().expect("failed to run ferric-cli");
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

fn run_cli_ok(tag: &str, xyz: &str, toml: &str) -> Run {
    let r = run_cli(tag, xyz, toml);
    assert!(r.ok, "{tag} failed:\n{}\n{}", r.stdout, r.stderr);
    r
}

fn toml_for(kind: &str, basis: &str, cas: &str) -> String {
    format!(
        "[molecule]\nxyz = \"{{xyz}}\"\n\n[basis]\nname = \"{basis}\"\n\n\
         [method]\nkind = \"{kind}\"\n\n[cas]\n{cas}\n"
    )
}

fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(CLI_THREADS)
        .build()
        .unwrap()
}

/// The prepared system (singlet RHF start) and converged RHF exactly as
/// `ferric_cli::cas::run_cas` builds them.
fn library_start(xyz: &str, basis_name: &str) -> (PreparedSystem, ferric_scf::ScfResult) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let sys = PreparedSystem::new(mol, &bs).unwrap();
    let rhf = rhf_start(&sys, ScfStart::default(), None).unwrap();
    (sys, rhf)
}

fn usize_of(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

fn sector(v: &Value) -> (usize, usize) {
    let a = v.as_array().unwrap();
    (usize_of(&a[0]), usize_of(&a[1]))
}

const CASSCF_SYSTEMS: &[&str] = &[
    "h2o_cas44_631g",
    "n2_cas66_631g",
    "n2_r1.6_cas66_631g",
    "o2_triplet_cas86_631g",
];

#[test]
fn cli_casscf_equals_the_driver_bit_for_bit() {
    for name in CASSCF_SYSTEMS {
        let r = read_json(&format!("testdata/reference/casscf_{name}_pyscf.json"));
        let xyz = r["xyz"].as_str().unwrap();
        let basis_name = r["basis"].as_str().unwrap();
        let (na, nb) = sector(&r["n_elec_active"]);
        let n_active = usize_of(&r["n_active"]);
        let n_core = usize_of(&r["n_core"]);
        let cli = run_cli_ok(
            name,
            xyz,
            &toml_for(
                "casscf",
                basis_name,
                &format!("n_active = {n_active}\nn_elec_active = [{na}, {nb}]\n"),
            ),
        );
        let res = cli.result();
        assert_eq!(res["kind"], "casscf");
        let total = res["total"].as_f64().unwrap();

        let lib = pool().install(|| {
            let (sys, rhf) = library_start(xyz, basis_name);
            let cfg = CasScfConfig {
                n_active,
                n_elec_active: (na, nb),
                n_core,
                ..Default::default()
            };
            casscf_on(&sys, &rhf, &cfg).unwrap()
        });
        assert!(lib.converged, "{name}");
        assert_eq!(
            total.to_bits(),
            lib.e_total.to_bits(),
            "{name}: CLI {total:.15} vs driver {:.15}",
            lib.e_total
        );
        assert_eq!(
            res["components"]["iterations"].as_u64(),
            Some(lib.iterations as u64)
        );
        let cert = &res["components"]["certificate"];
        assert_eq!(cert["capability"], "casscf");
        assert_eq!(cert["grade"], "proven", "{cert}");
        let e_ref = r["e_total"].as_f64().unwrap();
        eprintln!(
            "{name}: CLI {total:.13} == driver; PySCF {e_ref:.13}, diff {:.1e}",
            total - e_ref
        );
        assert!(
            (total - e_ref).abs() < 1e-9,
            "{name}: {total} vs PySCF {e_ref}"
        );
    }
}

#[test]
fn cli_nevpt2_equals_the_driver_bit_for_bit() {
    let refs = read_json("testdata/reference/nevpt2_refs.json");
    for s in refs["systems"].as_array().unwrap() {
        let label = s["system"].as_str().unwrap();
        let xyz = s["xyz"].as_str().unwrap();
        let basis_name = s["basis"].as_str().unwrap();
        let (na, nb) = sector(&s["n_elec_active"]);
        let n_active = usize_of(&s["n_active"]);
        let n_core = usize_of(&s["n_core"]);
        let grad_tol = s["ferric_casscf"]["casscf_grad_tol"].as_f64().unwrap();
        let cli = run_cli_ok(
            &format!("nevpt2_{label}"),
            xyz,
            &toml_for(
                "nevpt2",
                basis_name,
                &format!(
                    "n_active = {n_active}\nn_elec_active = [{na}, {nb}]\ngrad_tol = {grad_tol:e}\n"
                ),
            ),
        );
        let res = cli.result();
        assert_eq!(res["kind"], "nevpt2");
        let comp = &res["components"];
        let lib = pool().install(|| {
            let (sys, rhf) = library_start(xyz, basis_name);
            let cfg = CasScfConfig {
                n_active,
                n_elec_active: (na, nb),
                n_core,
                grad_tol,
                ..Default::default()
            };
            nevpt2_on(&sys, &rhf, Nevpt2On::Casscf, &cfg).unwrap()
        });
        let r = &lib.result;
        let total = res["total"].as_f64().unwrap();
        assert_eq!(
            total.to_bits(),
            (r.e_reference + r.e_corr).to_bits(),
            "{label}: total"
        );
        assert_eq!(
            comp["e_corr"].as_f64().unwrap().to_bits(),
            r.e_corr.to_bits()
        );
        let p = &s["pyscf"];
        let dk = (lib.reference_grad_norm.unwrap() + p["grad_norm"].as_f64().unwrap())
            / p["lambda_min"].as_f64().unwrap();
        for (name, want) in Nevpt2Classes::NAMES.iter().zip(r.energies.to_array()) {
            let got = comp["classes"][*name].as_f64().unwrap();
            assert_eq!(got.to_bits(), want.to_bits(), "{label} {name}");
            let pyscf = p["classes"][*name].as_f64().unwrap();
            let bar = p["dE_dkappa"][*name].as_f64().unwrap() * dk
                + 2.0 * p["degenerate_spread"][*name].as_f64().unwrap()
                + 1e-10;
            eprintln!(
                "{label} {name:6} CLI {got:.12} PySCF {pyscf:.12} diff {:.1e} bar {bar:.1e}",
                got - pyscf
            );
            assert!((got - pyscf).abs() < bar, "{label} {name}");
        }
        assert_eq!(comp["certificate"]["grade"], "proven");
    }
}

const H2O_631G: &str = "h2o_cas44_631g";

fn h2o() -> (String, Value) {
    let r = read_json(&format!("testdata/reference/casscf_{H2O_631G}_pyscf.json"));
    (r["xyz"].as_str().unwrap().to_string(), r)
}

#[test]
fn cli_keys_reach_the_config() {
    let (xyz, _) = h2o();
    let base = "n_active = 4\nn_elec_active = [2, 2]\n";
    let conv = run_cli_ok("keys_base", &xyz, &toml_for("casscf", "6-31g", base));
    let e_conv = conv.result()["total"].as_f64().unwrap();

    // [cas] max_iter: unconverged, warned, capped.
    let short = run_cli_ok(
        "keys_max_iter",
        &xyz,
        &toml_for("casscf", "6-31g", &format!("{base}max_iter = 2\n")),
    );
    let c = &short.result()["components"];
    assert_eq!(c["converged"], false);
    assert_eq!(c["iterations"], 2);
    assert_eq!(c["certificate"]["grade"], "ungraded");
    assert!(
        short.stderr.contains("CASSCF did not converge"),
        "{}",
        short.stderr
    );
    assert_eq!(short.record("run_end")["converged"], false);

    // [cas.ci] conv_thresh: a loose CI threshold moves the converged energy
    // by more than rounding (the CASSCF stops on an inexact CI root).
    let loose = run_cli_ok(
        "keys_ci",
        &xyz,
        &toml_for(
            "casscf",
            "6-31g",
            &format!("{base}\n[cas.ci]\nconv_thresh = 1e-3\n"),
        ),
    );
    let e_loose = loose.result()["total"].as_f64().unwrap();
    assert_ne!(
        e_loose.to_bits(),
        e_conv.to_bits(),
        "[cas.ci] conv_thresh ignored"
    );

    // [cas] canonicalize: mo_energy appears, the energy stays.
    let canon = run_cli_ok(
        "keys_canon",
        &xyz,
        &toml_for("casscf", "6-31g", &format!("{base}canonicalize = true\n")),
    );
    let cc = &canon.result()["components"];
    assert!(cc["mo_energy"].is_array());
    assert!(conv.result()["components"]["mo_energy"].is_null());
    assert!((canon.result()["total"].as_f64().unwrap() - e_conv).abs() < 1e-11);

    // [cas] optimizer = "bfgs": the default (AH) minimum in more macro
    // iterations (tests/casscf_newton.rs bounds the energy difference by
    // 1e-12).
    let bfgs = run_cli_ok(
        "keys_optimizer",
        &xyz,
        &toml_for("casscf", "6-31g", &format!("{base}optimizer = \"bfgs\"\n")),
    );
    let it = |r: &Run| usize_of(&r.result()["components"]["iterations"]);
    assert!(
        it(&bfgs) > it(&conv),
        "[cas] optimizer ignored: {} vs {} iterations",
        it(&bfgs),
        it(&conv)
    );
    assert!((bfgs.result()["total"].as_f64().unwrap() - e_conv).abs() < 1e-10);

    // [cas] integrals = "ri" with auxbasis: the density-fitted Hamiltonian
    // (a different energy, RI-error sized), recorded on the certificate,
    // which carries no exact-integral anchor and is capped to Proven (narrow).
    let ri = run_cli_ok(
        "keys_ri",
        &xyz,
        &toml_for(
            "casscf",
            "6-31g",
            &format!("{base}integrals = \"ri\"\nauxbasis = \"def2-universal-jkfit\"\n"),
        ),
    );
    let e_ri = ri.result()["total"].as_f64().unwrap();
    let rc = &ri.result()["components"]["certificate"];
    assert!(
        (e_ri - e_conv).abs() > 1e-7 && (e_ri - e_conv).abs() < 1e-3,
        "RI energy {e_ri} vs exact {e_conv}"
    );
    assert_eq!(rc["grade"], "proven_narrow", "{rc}");
    assert!(rc["measured_error"].is_null(), "{rc}");
    assert!(
        rc["approximations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["name"] == "ri" && a["detail"] == "aux def2-universal-jkfit"),
        "{rc}"
    );

    // [cas] stability: the default ("off") logs no check; "check" is logged
    // and recorded on the certificate and leaves the energy unchanged.
    assert!(conv.result()["components"]["stability"].is_null());
    let chk = run_cli_ok(
        "keys_stability_check",
        &xyz,
        &toml_for("casscf", "6-31g", &format!("{base}stability = \"check\"\n")),
    );
    let st = &chk.result()["components"]["stability"];
    assert_eq!(st["verdict"], "STABLE", "{st}");
    assert!(st["lowest_eigenvalue"].as_f64().unwrap() > 0.0);
    let routes = chk.result()["components"]["certificate"]["routes"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        routes.iter().any(|r| r["name"] == "stability"),
        "{routes:?}"
    );
    assert_eq!(
        chk.result()["total"].as_f64().unwrap().to_bits(),
        e_conv.to_bits()
    );

    // [cas] n_core: one orbital lower, CAS(6,5) above 2 core, another energy.
    let window = run_cli_ok(
        "keys_core",
        &xyz,
        &toml_for(
            "casscf",
            "6-31g",
            "n_active = 5\nn_elec_active = [3, 3]\nn_core = 2\n",
        ),
    );
    assert_eq!(window.record("run_start")["config"]["n_core"], 2);
    assert!(window.result()["total"].as_f64().unwrap() < e_conv - 1e-4);

    // [cas] reference = "casci": the CAS-CI reference on RHF orbitals lies
    // above CASSCF, and the grade is capped.
    let casci = run_cli_ok(
        "keys_casci",
        &xyz,
        &toml_for("nevpt2", "6-31g", &format!("{base}reference = \"casci\"\n")),
    );
    let c = &casci.result()["components"];
    assert_eq!(c["reference"], "casci");
    assert!(c["e_reference"].as_f64().unwrap() > e_conv);
    assert_eq!(c["certificate"]["grade"], "ungraded");
    assert!(c["reference_grad_norm"].is_null());
}

#[test]
fn cli_refusals_are_named() {
    let (xyz, _) = h2o();
    let base = "n_active = 4\nn_elec_active = [2, 2]\n";
    let cases: &[(&str, String, &str)] = &[
        (
            "no_cas",
            "[molecule]\nxyz = \"{xyz}\"\n[basis]\nname = \"6-31g\"\n[method]\nkind = \"casscf\"\n"
                .to_string(),
            "needs a [cas] section",
        ),
        (
            "cas_on_rhf",
            format!(
                "[molecule]\nxyz = \"{{xyz}}\"\n[basis]\nname = \"6-31g\"\n[method]\nkind = \
                 \"rhf\"\n[cas]\n{base}"
            ),
            "[cas] is read by method.kind",
        ),
        (
            "scf_key",
            format!(
                "{}\n[scf]\ndf_j_aux = \"cc-pvdz-ri\"\n",
                toml_for("casscf", "6-31g", base)
            ),
            "[scf] df_j_aux is not read",
        ),
        (
            "section",
            format!(
                "{}\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n",
                toml_for("casscf", "6-31g", base)
            ),
            "[mp2] is not read",
        ),
        (
            "task",
            toml_for("casscf", "6-31g", base).replace(
                "kind = \"casscf\"",
                "kind = \"casscf\"\ntask = \"optimize\"",
            ),
            "task = \"energy\" only",
        ),
        (
            "multiplicity",
            toml_for("casscf", "6-31g", base)
                .replace("xyz = \"{xyz}\"", "xyz = \"{xyz}\"\nmultiplicity = 3"),
            "multiplicity must be 1",
        ),
        (
            "reference_on_casscf",
            toml_for("casscf", "6-31g", &format!("{base}reference = \"casci\"\n")),
            "[cas] reference is read by method.kind = \"nevpt2\" only",
        ),
        (
            "stability_value",
            toml_for("casscf", "6-31g", &format!("{base}stability = \"on\"\n")),
            "[cas] stability must be \"check\", \"follow\" or \"off\"",
        ),
        (
            "stability_on_casci",
            toml_for(
                "nevpt2",
                "6-31g",
                &format!("{base}reference = \"casci\"\nstability = \"check\"\n"),
            ),
            "[cas] stability sets the CASSCF orbital optimization",
        ),
        (
            "canonicalize_on_nevpt2",
            toml_for("nevpt2", "6-31g", &format!("{base}canonicalize = true\n")),
            "[cas] canonicalize is read by method.kind = \"casscf\" only",
        ),
        (
            "ri_without_aux",
            toml_for("casscf", "6-31g", &format!("{base}integrals = \"ri\"\n")),
            "integrals = \"ri\" needs auxbasis",
        ),
        (
            "aux_without_ri",
            toml_for(
                "casscf",
                "6-31g",
                &format!("{base}auxbasis = \"def2-universal-jkfit\"\n"),
            ),
            "auxbasis is read by integrals = \"ri\" only",
        ),
        (
            "casci_ri",
            toml_for(
                "nevpt2",
                "6-31g",
                &format!(
                    "{base}reference = \"casci\"\nintegrals = \"ri\"\nauxbasis = \"def2-universal-jkfit\"\n"
                ),
            ),
            "[cas] integrals sets the CASSCF orbital optimization",
        ),
        (
            "bad_reference",
            toml_for("nevpt2", "6-31g", &format!("{base}reference = \"rhf\"\n")),
            "reference must be \"casscf\" or \"casci\"",
        ),
        (
            "casci_orbital_key",
            toml_for(
                "nevpt2",
                "6-31g",
                &format!("{base}reference = \"casci\"\ngrad_tol = 1e-8\n"),
            ),
            "[cas] grad_tol sets the CASSCF orbital optimization",
        ),
        (
            "bad_optimizer",
            toml_for(
                "casscf",
                "6-31g",
                &format!("{base}optimizer = \"newton\"\n"),
            ),
            "[cas] optimizer must be \"ah\" or \"bfgs\"",
        ),
        (
            "casci_optimizer",
            toml_for(
                "nevpt2",
                "6-31g",
                &format!("{base}reference = \"casci\"\noptimizer = \"ah\"\n"),
            ),
            "[cas] optimizer sets the CASSCF orbital optimization",
        ),
        (
            "open_shell_nevpt2",
            toml_for("nevpt2", "6-31g", "n_active = 4\nn_elec_active = [3, 1]\n"),
            "closed-shell active spaces",
        ),
        (
            "bad_tol",
            toml_for("casscf", "6-31g", &format!("{base}grad_tol = -1.0\n")),
            "[cas] grad_tol must be a positive finite number",
        ),
        (
            "zero_ci_iter",
            toml_for(
                "casscf",
                "6-31g",
                &format!("{base}\n[cas.ci]\nmax_iter = 0\n"),
            ),
            "[cas] ci.max_iter must be at least 1",
        ),
        (
            "unknown_key",
            toml_for("casscf", "6-31g", &format!("{base}n_virtual = 3\n")),
            "unknown field `n_virtual`",
        ),
        (
            "odd_core",
            toml_for("casscf", "6-31g", "n_active = 4\nn_elec_active = [2, 1]\n"),
            "do not leave an even core count",
        ),
    ];
    for (tag, toml, want) in cases {
        let r = run_cli(&format!("refuse_{tag}"), &xyz, toml);
        assert!(!r.ok, "{tag}: accepted\n{}", r.stdout);
        assert!(
            r.stderr.contains(want),
            "{tag}: {:?} not in\n{}",
            want,
            r.stderr
        );
        assert_eq!(
            r.scf_iterations(),
            0,
            "{tag}: an SCF ran before the refusal"
        );
    }
}

/// `[scf] gc_rotation` is read by the CAS kinds (the RHF start and the
/// direct CASSCF J/K follow it), not refused by the `[scf]` key filter, and
/// the run log records it. On 6-31G (nothing to rotate) every value is the
/// default run bit for bit; a value of another case is refused at load.
#[test]
fn scf_gc_rotation_is_read() {
    let (xyz, _) = h2o();
    let cas = "n_active = 4\nn_elec_active = [2, 2]\nintegrals = \"direct\"\n";
    let base = run_cli_ok("gc_default", &xyz, &toml_for("casscf", "6-31g", cas));
    let e0 = base.result()["total"].as_f64().unwrap();
    assert!(base.record("run_start")["config"]["gc_rotation"].is_null());
    for v in ["off", "auto"] {
        let r = run_cli_ok(
            &format!("gc_{v}"),
            &xyz,
            &format!(
                "{}\n[scf]\ngc_rotation = \"{v}\"\n",
                toml_for("casscf", "6-31g", cas)
            ),
        );
        assert_eq!(r.record("run_start")["config"]["gc_rotation"], v);
        let e = r.result()["total"].as_f64().unwrap();
        assert_eq!(e.to_bits(), e0.to_bits(), "{v}");
    }
    let bad = run_cli(
        "gc_bad",
        &xyz,
        &format!(
            "{}\n[scf]\ngc_rotation = \"OFF\"\n",
            toml_for("nevpt2", "6-31g", cas)
        ),
    );
    assert!(!bad.ok);
    assert!(bad.stderr.contains("gc_rotation"), "{}", bad.stderr);
}

/// `[scf] jk_digest` is read by the CAS kinds like `gc_rotation` (the RHF
/// start and the direct CASSCF J/K follow it), recorded in the run log as
/// requested and as resolved (`FERRIC_JK_DIGEST` deciding an unset key),
/// and the two digests agree to rounding; a value of another case is
/// refused at load.
#[test]
fn scf_jk_digest_is_read() {
    let (xyz, _) = h2o();
    let cas = "n_active = 4\nn_elec_active = [2, 2]\nintegrals = \"direct\"\n";
    let unset = [("FERRIC_JK_DIGEST", None)];
    let base = run_cli_env(
        "jkd_default",
        &xyz,
        &toml_for("casscf", "6-31g", cas),
        &unset,
    );
    assert!(base.ok, "{}", base.stderr);
    let e0 = base.result()["total"].as_f64().unwrap();
    let start = &base.record("run_start")["config"];
    assert!(start["jk_digest"].is_null());
    assert_eq!(start["jk_digest_resolved"], "pair");
    let by_env = run_cli_env(
        "jkd_env",
        &xyz,
        &toml_for("casscf", "6-31g", cas),
        &[("FERRIC_JK_DIGEST", Some("legacy"))],
    );
    assert!(by_env.ok, "{}", by_env.stderr);
    let start = &by_env.record("run_start")["config"];
    assert!(start["jk_digest"].is_null());
    assert_eq!(start["jk_digest_resolved"], "legacy");
    for v in ["pair", "legacy"] {
        let r = run_cli_ok(
            &format!("jkd_{v}"),
            &xyz,
            &format!(
                "{}\n[scf]\njk_digest = \"{v}\"\n",
                toml_for("casscf", "6-31g", cas)
            ),
        );
        assert_eq!(r.record("run_start")["config"]["jk_digest"], v);
        assert_eq!(r.record("run_start")["config"]["jk_digest_resolved"], v);
        let e = r.result()["total"].as_f64().unwrap();
        assert!((e - e0).abs() < 1e-10, "{v}: {e} vs default {e0}");
    }
    let bad = run_cli(
        "jkd_bad",
        &xyz,
        &format!(
            "{}\n[scf]\njk_digest = \"LEGACY\"\n",
            toml_for("nevpt2", "6-31g", cas)
        ),
    );
    assert!(!bad.ok);
    assert!(bad.stderr.contains("jk_digest"), "{}", bad.stderr);
}

/// `[scf] gc_kernel` is read by the CAS kinds like `jk_digest`: it switches
/// the RHF start's rotated cc-pVDZ walk, and is passed to the direct CASSCF
/// backend, whose pooled J/K builds keep libint (so it changes nothing
/// there). Recorded in the run log as requested and as resolved
/// (`FERRIC_GC_KERNEL` deciding an unset key, default `off`); the two
/// settings agree to rounding; a value of another case is refused at load.
#[test]
fn scf_gc_kernel_is_read() {
    let (xyz, _) = h2o();
    let cas = "n_active = 4\nn_elec_active = [2, 2]\nintegrals = \"direct\"\n";
    let base = run_cli_env(
        "gck_default",
        &xyz,
        &toml_for("casscf", "cc-pvdz", cas),
        &[("FERRIC_GC_KERNEL", None)],
    );
    assert!(base.ok, "{}", base.stderr);
    let e0 = base.result()["total"].as_f64().unwrap();
    let start = &base.record("run_start")["config"];
    assert!(start["gc_kernel"].is_null());
    assert_eq!(start["gc_kernel_resolved"], "off");
    let by_env = run_cli_env(
        "gck_env",
        &xyz,
        &toml_for("casscf", "cc-pvdz", cas),
        &[("FERRIC_GC_KERNEL", Some("auto"))],
    );
    assert!(by_env.ok, "{}", by_env.stderr);
    let start = &by_env.record("run_start")["config"];
    assert!(start["gc_kernel"].is_null());
    assert_eq!(start["gc_kernel_resolved"], "auto");
    for v in ["auto", "off"] {
        let r = run_cli_ok(
            &format!("gck_{v}"),
            &xyz,
            &format!(
                "{}\n[scf]\ngc_kernel = \"{v}\"\n",
                toml_for("casscf", "cc-pvdz", cas)
            ),
        );
        assert_eq!(r.record("run_start")["config"]["gc_kernel"], v);
        assert_eq!(r.record("run_start")["config"]["gc_kernel_resolved"], v);
        let e = r.result()["total"].as_f64().unwrap();
        assert!((e - e0).abs() < 1e-10, "{v}: {e} vs default {e0}");
    }
    let bad = run_cli(
        "gck_bad",
        &xyz,
        &format!(
            "{}\n[scf]\ngc_kernel = \"AUTO\"\n",
            toml_for("nevpt2", "6-31g", cas)
        ),
    );
    assert!(!bad.ok);
    assert!(bad.stderr.contains("gc_kernel"), "{}", bad.stderr);
}
