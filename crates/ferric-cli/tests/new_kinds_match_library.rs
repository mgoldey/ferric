//! The CLI kinds `ccd`, `ccsd(t)`, `drpa` and `linlccd-amplitude` must report
//! the SAME energy as a direct call of the library function they wrap.
//!
//! These kinds are wiring only -- the library functions (and their Python
//! bindings) already exist and are validated on their own. What wiring can get
//! wrong is the part these tests pin: which function is called, which `[mp2]`
//! knob reaches which config field (aux, frozen core, eps, variant, the sweep
//! points), and which number is printed/logged as the total. Each test runs
//! the real `ferric-cli` binary, reads the `result` record from its JSON run
//! log (full f64 precision, unlike the 10-decimal printout), and compares it
//! with the library called in-process on an RHF solved with the same SCF
//! settings.
//!
//! Tolerance 1e-9 Ha. Both sides run the same exact-integral `solve_rhf`
//! (`[scf] df_guess = false` so the CLI does not take its two-stage DF-first
//! SCF) to `density_conv = 1e-10`, so they follow the same trajectory; what is
//! left is reduction-order round-off, expected orders below the bar. The
//! knobs a mis-wiring would drop must move the answer by far more than the
//! bar, and the tests assert that where it is not obvious: the two sweep
//! points differ by > 1e-7, hh vs drivers-only by > 1e-6, and the (T) term is
//! > 1e-6 (it is -6.74e-5 Ha on water/STO-3G per the PySCF cross-check in
//! the validation matrix).

use std::path::{Path, PathBuf};
use std::process::Command;

use ferric_cc::linlccd::LadderVariant;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const TOL: f64 = 1e-9;

/// See `ccsd_dispatch.rs`: resolved at RUN time for `cargo nextest archive`.
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

const SCF: &str = "[scf]\nmax_iter = 200\ndensity_conv = 1e-10\ndf_guess = false\n";

/// Run water/`basis` under `kind` with `[mp2]` = `mp2`, and return the stdout
/// and every `result` record of the JSON run log, in order.
fn run_cli(tag: &str, basis: &str, kind: &str, mp2: &str) -> (String, Vec<serde_json::Value>) {
    let root = workspace_root();
    let toml_path = root.join("target").join(format!("new_kinds_{tag}.toml"));
    let log_path = root.join("target").join(format!("new_kinds_{tag}.jsonl"));
    let _ = std::fs::remove_file(&log_path);
    let body = format!(
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
         [basis]\nname = \"{basis}\"\n\n\
         [method]\nkind = \"{kind}\"\n\n\
         {SCF}\n\
         [mp2]\nauxbasis = \"cc-pvdz-ri\"\n{mp2}\n"
    );
    std::fs::write(&toml_path, body).expect("write temp toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml_path)
        .arg("--json")
        .arg(&log_path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("failed to run ferric-cli binary");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{kind} run failed.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let log = std::fs::read_to_string(&log_path).expect("read JSON run log");
    let results: Vec<serde_json::Value> = log
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["record"] == "result")
        .collect();
    assert!(
        !results.is_empty(),
        "no result record in the run log:\n{log}"
    );
    for r in &results {
        assert_eq!(r["kind"], kind, "{r}");
    }
    (stdout, results)
}

fn num(v: &serde_json::Value, what: &str) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("{what} is not a number: {v}"))
}

/// The library side: water/`basis`, exact-integral RHF with the CLI's SCF
/// settings above.
struct Lib {
    mol: Molecule,
    obs_bs: basis::BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn library_setup(basis_name: &str) -> Lib {
    let root = workspace_root();
    let mol = Molecule::load_xyz(root.join("testdata/molecules/water.xyz").to_str().unwrap())
        .expect("load water");
    let obs_bs = basis::bundled(basis_name).expect("bundled obs");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("prep obs");
    let dfbs =
        PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").expect("aux")).expect("prep aux");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            max_iter: 200,
            density_conv: 1e-10,
            ..Default::default()
        },
    )
    .expect("library RHF");
    assert!(rhf.converged, "library RHF did not converge");
    Lib {
        mol,
        obs_bs,
        obs,
        dfbs,
        rhf,
    }
}

fn assert_close(cli: f64, lib: f64, what: &str) {
    assert!(
        (cli - lib).abs() < TOL,
        "{what}: CLI {cli:.12} vs library {lib:.12} (|diff| {:.3e} >= {TOL:.0e})",
        (cli - lib).abs()
    );
}

/// `kind = "ccd"` == `ferric_cc::ccd::ccd` (Python `run_ccd`).
#[test]
fn ccd_matches_the_library() {
    let (stdout, res) = run_cli("ccd", "sto-3g", "ccd", "");
    assert!(stdout.contains("CCD corr"), "{stdout}");
    let lib = library_setup("sto-3g");
    let r = ferric_cc::ccd::ccd(
        &lib.mol,
        &lib.obs,
        &lib.dfbs,
        Operator::coulomb(),
        &lib.rhf,
        &CcConfig::default(),
    )
    .expect("library ccd");
    assert_close(
        num(&res[0]["components"]["e_corr"], "e_corr"),
        r.correlation_energy,
        "CCD correlation",
    );
    assert_close(
        num(&res[0]["total"], "total"),
        lib.rhf.energy + r.correlation_energy,
        "CCD total",
    );
}

/// `kind = "ccsd(t)"` == `ccsd_closed_shell` + `ccsd_t_closed_shell` (Python
/// `run_ccsd_t`). The total must INCLUDE the (T) term: a total that dropped
/// it would equal the plain CCSD total and miss by |E_(T)| ~ 7e-5.
#[test]
fn ccsd_t_matches_the_library() {
    let (stdout, res) = run_cli("ccsd_t", "sto-3g", "ccsd(t)", "");
    assert!(stdout.contains("(T) corr"), "{stdout}");
    let lib = library_setup("sto-3g");
    let cfg = CcConfig::default();
    let cc = ferric_cc::ccsd_closed_shell::ccsd_closed_shell(
        &lib.mol,
        &lib.obs,
        &lib.dfbs,
        Operator::coulomb(),
        &lib.rhf,
        &cfg,
    )
    .expect("library ccsd_closed_shell");
    let e_t = ferric_cc::ccsd_t_closed_shell::ccsd_t_closed_shell(
        &lib.mol,
        &lib.obs,
        &lib.dfbs,
        Operator::coulomb(),
        &lib.rhf,
        &cc,
        &cfg,
    )
    .expect("library ccsd_t_closed_shell");
    assert!(e_t.abs() > 1e-6, "(T) must be non-trivial here: {e_t:e}");
    let c = &res[0]["components"];
    assert_close(
        num(&c["e_ccsd_corr"], "e_ccsd_corr"),
        cc.correlation_energy,
        "CCSD corr",
    );
    assert_close(num(&c["e_t"], "e_t"), e_t, "(T)");
    assert_close(
        num(&res[0]["total"], "total"),
        lib.rhf.energy + cc.correlation_energy + e_t,
        "CCSD(T) total",
    );
}

fn drpa_lib_cfg(
    eps: f64,
    compute_reference: bool,
) -> ferric_mp2::drpa_amplitude::AmplitudeDrpaConfig {
    // The Python binding's defaults, which the CLI documents that it matches.
    ferric_mp2::drpa_amplitude::AmplitudeDrpaConfig {
        eps,
        frozen_core: 1,
        compute_reference,
        diis: Some(8),
        eps_rtol_factor: Some(0.1),
        ..Default::default()
    }
}

/// `kind = "drpa"`, single eps with the opt-in reference == `amplitude_drpa`.
#[test]
fn drpa_matches_the_library() {
    let (stdout, res) = run_cli(
        "drpa",
        "6-31g",
        "drpa",
        "frozen_core = 1\ndrpa_eps = 1e-3\ndrpa_reference = true",
    );
    assert_eq!(res.len(), 1, "one eps => one result record");
    assert!(stdout.contains("threshold error"), "{stdout}");
    let lib = library_setup("6-31g");
    let r = ferric_mp2::drpa_amplitude::amplitude_drpa(
        &lib.mol,
        &lib.obs,
        &lib.obs_bs,
        &lib.dfbs,
        Operator::coulomb(),
        &lib.rhf,
        &drpa_lib_cfg(1e-3, true),
    )
    .expect("library amplitude_drpa");
    let c = &res[0]["components"];
    assert_close(num(&c["eps"], "eps"), 1e-3, "eps echoed");
    assert_close(num(&c["e_corr"], "e_corr"), r.e_corr, "dRPA corr");
    assert_close(
        num(&c["e_corr_plasmon_canonical"], "reference"),
        r.e_corr_plasmon_canonical,
        "dRPA canonical reference",
    );
    assert_close(num(&res[0]["total"], "total"), r.e_total, "dRPA total");
}

/// `[mp2] drpa_eps_sweep` gives one result per (sorted, de-duplicated) eps,
/// each equal to a SEPARATE library `amplitude_drpa` call at that eps. The
/// sweep is written unsorted with a duplicate on purpose. Without a
/// reference the log field is null, not the library's NaN sentinel.
#[test]
fn drpa_eps_sweep_matches_separate_library_calls() {
    let (stdout, res) = run_cli(
        "drpa_sweep",
        "6-31g",
        "drpa",
        "frozen_core = 1\ndrpa_eps_sweep = [1e-3, 0.0, 1e-3]",
    );
    assert_eq!(res.len(), 2, "sweep must yield one record per distinct eps");
    assert!(stdout.contains("not computed"), "{stdout}");
    let lib = library_setup("6-31g");
    let mut corrs = Vec::new();
    for (rec, eps) in res.iter().zip([0.0, 1e-3]) {
        let r = ferric_mp2::drpa_amplitude::amplitude_drpa(
            &lib.mol,
            &lib.obs,
            &lib.obs_bs,
            &lib.dfbs,
            Operator::coulomb(),
            &lib.rhf,
            &drpa_lib_cfg(eps, false),
        )
        .expect("library amplitude_drpa");
        let c = &rec["components"];
        assert_close(num(&c["eps"], "eps"), eps, "sweep order");
        assert!(c["e_corr_plasmon_canonical"].is_null(), "{c}");
        let e = num(&c["e_corr"], "e_corr");
        assert_close(e, r.e_corr, &format!("sweep dRPA corr at eps={eps:e}"));
        assert_close(num(&rec["total"], "total"), r.e_total, "sweep dRPA total");
        corrs.push(e);
    }
    // Reachability: the two points must actually differ, or a sweep that
    // ignored the per-point eps would pass.
    assert!(
        (corrs[0] - corrs[1]).abs() > 1e-7,
        "sweep points identical: {corrs:?}"
    );
}

fn linlccd_lib(
    lib: &Lib,
    variant: LadderVariant,
    eps: f64,
) -> ferric_cc::linlccd_amplitude::AmplitudeLinLccdResult {
    ferric_cc::linlccd_amplitude::amplitude_linlccd(
        &lib.mol,
        &lib.obs,
        &lib.obs_bs,
        &lib.dfbs,
        Operator::coulomb(),
        &lib.rhf,
        &ferric_cc::linlccd_amplitude::AmplitudeLinLccdConfig {
            eps,
            frozen_core: 1,
            ..Default::default()
        },
        variant,
    )
    .expect("library amplitude_linlccd")
}

/// `kind = "linlccd-amplitude"` == `amplitude_linlccd`, for the default
/// variant (hh) and for `drivers-only`. Two variants so a CLI that ignored
/// `linlccd_variant` (always hh) fails the second comparison.
#[test]
fn linlccd_amplitude_matches_the_library() {
    let lib = library_setup("6-31g");
    for (tag, knob, variant, name) in [
        ("linlccd_amp_hh", "", LadderVariant::Hh, "hh"),
        (
            "linlccd_amp_drivers",
            "linlccd_variant = \"drivers-only\"\n",
            LadderVariant::DriversOnly,
            "drivers-only",
        ),
    ] {
        let (_, res) = run_cli(
            tag,
            "6-31g",
            "linlccd-amplitude",
            &format!("frozen_core = 1\nlinlccd_eps = 1e-3\n{knob}"),
        );
        let r = linlccd_lib(&lib, variant, 1e-3);
        let c = &res[0]["components"];
        assert_eq!(c["variant"], name, "{c}");
        assert_close(
            num(&c["e_corr"], "e_corr"),
            r.e_corr,
            &format!("LinLCCD({name}) corr"),
        );
        assert_close(
            num(&res[0]["total"], "total"),
            r.e_total,
            &format!("LinLCCD({name}) total"),
        );
    }
    // The two variants must differ by far more than TOL, or the variant
    // comparison above could not tell them apart.
    let hh = linlccd_lib(&lib, LadderVariant::Hh, 1e-3).e_corr;
    let dr = linlccd_lib(&lib, LadderVariant::DriversOnly, 1e-3).e_corr;
    assert!((hh - dr).abs() > 1e-6, "hh {hh} vs drivers-only {dr}");
}
