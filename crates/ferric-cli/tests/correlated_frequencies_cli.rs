//! `task = "frequencies"` on the correlated kinds that have an analytic
//! nuclear gradient (RI-MP2, SCS-MP2, OO-RI-MP2, attenuated RI-MP2,
//! pdep-rpa), through the CLI
//! binary: the Hessian is the central difference of the same gradient
//! `task = "optimize"` follows.
//!
//! Independent route: on H2 (bond along z) the Cartesian Hessian element
//! `H[(0,z),(0,z)]` is the bond force constant `k = d2E/dr2`, which a
//! five-point central difference of the ENERGY of the same method (the CLI's own energy run, no gradient
//! involved) reproduces.
//!
//! Both runs converge the SCF to energy_conv 1e-12 / density_conv 1e-10: the
//! default 1e-6 energy criterion alone puts ~1e-2 of noise on the stencil, and
//! the energy task would otherwise take the f32 RI-J tensor (jitter ~1e-6 Ha,
//! ~1e-2 on a second difference at this step); geometry tasks use f64. The J/K
//! route is pinned to `exact` because the energy task defaults to RI-JK while
//! the analytic gradients here differentiate the four-centre reference.
//!
//! Bar, derived from the two stencils: the gradient stencil (`delta = 5e-3`
//! Bohr) has truncation `delta^2 g''' / 6` ~ 4e-6 Ha/Bohr^2 for `g''' ~ 1`
//! and gradient noise ~1e-9 / delta = 2e-7; the energy stencil (`h = 0.0378`
//! Bohr, fifth order) has truncation ~`h^4 E^(6)/90` < 1e-6 and noise
//! ~1e-10 / h^2 < 1e-7. Both sum under 1e-5; the assertion uses 1e-4, which
//! is 0.03% of `k` ~ 0.37 and ~3 orders below the RHF-vs-correlated gap
//! (the mutation in the commit message routes to RHF and fails it).
use std::path::PathBuf;
use std::process::{Command, Output};

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;
const R0_ANG: f64 = 0.7414;
const H_ANG: f64 = 0.02;
const BAR: f64 = 1e-4;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn scratch() -> PathBuf {
    let dir = workspace_root()
        .join("target")
        .join("correlated_frequencies_cli_test");
    std::fs::create_dir_all(&dir).expect("create the temp dir");
    dir
}

fn h2_xyz(tag: &str, r_ang: f64) -> PathBuf {
    let path = scratch().join(format!("{tag}.xyz"));
    std::fs::write(
        &path,
        format!("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 {r_ang:.12}\n"),
    )
    .expect("write xyz");
    path
}

fn run_cli(tag: &str, toml: &str) -> (Output, Vec<serde_json::Value>) {
    let dir = scratch();
    let toml_path = dir.join(format!("{tag}.toml"));
    let log = dir.join(format!("{tag}.jsonl"));
    let _ = std::fs::remove_file(&log);
    std::fs::write(&toml_path, toml).expect("write toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml_path)
        .arg("--json")
        .arg(&log)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("run ferric-cli");
    let records = std::fs::read_to_string(&log)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    (out, records)
}

/// The `[method]`/`[mp2]`/`[rpa]` text of one kind (no `task`).
fn method_block(kind: &str) -> String {
    match kind {
        "pdep-rpa" => "[method]\nkind = \"pdep-rpa\"\n[rpa]\nauxbasis = \"cc-pvdz-ri\"\n".into(),
        k => format!("[method]\nkind = \"{k}\"\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n"),
    }
}

/// A well-formed input: `task` inside `[method]`.
fn toml_for(xyz: &std::path::Path, kind: &str, task: &str, extra: &str) -> String {
    let m = method_block(kind).replacen("[method]\n", &format!("[method]\ntask = \"{task}\"\n"), 1);
    // `[scf] jk` is read by the MP2 kinds only; pdep-rpa is always four-centre.
    let jk = if kind == "pdep-rpa" {
        ""
    } else {
        "jk = \"exact\"\n"
    };
    format!(
        "[molecule]\nxyz = \"{}\"\n[basis]\nname = \"sto-3g\"\n{m}[scf]\nenergy_conv = 1e-12\ndensity_conv = 1e-10\nri_j_precision = \"f64\"\n{jk}{extra}",
        xyz.display()
    )
}

fn energy(kind: &str, r_ang: f64, tag: &str) -> f64 {
    let xyz = h2_xyz(tag, r_ang);
    let (out, records) = run_cli(tag, &toml_for(&xyz, kind, "energy", ""));
    assert!(
        out.status.success(),
        "{tag}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    records
        .iter()
        .rev()
        .find(|r| r["record"] == "result" && r["kind"] == kind)
        .unwrap_or_else(|| panic!("{tag}: no result record for {kind}"))["total"]
        .as_f64()
        .expect("total energy")
}

/// Five-point central difference of `E(r)`, in Hartree/Bohr^2.
fn force_constant_from_energies(kind: &str) -> f64 {
    let h = H_ANG / ANGSTROM_PER_BOHR;
    let e = |k: i32| {
        energy(
            kind,
            R0_ANG + f64::from(k) * H_ANG,
            &format!("e_{kind}_{k}"),
        )
    };
    let (m2, m1, p0, p1, p2) = (e(-2), e(-1), e(0), e(1), e(2));
    (-p2 + 16.0 * p1 - 30.0 * p0 + 16.0 * m1 - m2) / (12.0 * h * h)
}

/// `(k from the frequency run's Hessian, the run's `frequencies` record)`.
fn force_constant_from_frequencies(kind: &str, tag: &str) -> (f64, serde_json::Value) {
    let tag = format!("{tag}_{kind}");
    let xyz = h2_xyz(&tag, R0_ANG);
    let (out, records) = run_cli(&tag, &toml_for(&xyz, kind, "frequencies", ""));
    assert!(
        out.status.success(),
        "{tag}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rec = records
        .iter()
        .find(|r| r["record"] == "frequencies")
        .unwrap_or_else(|| panic!("{tag}: no frequencies record"))
        .clone();
    let h: Vec<f64> = rec["cartesian_hessian"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(h.len(), 36, "{tag}: 6x6 Hessian");
    // The bond is along z: H[(0,z),(0,z)] is the force constant (the x and y
    // diagonals are g/r, nonzero off the minimum).
    (h[14], rec)
}

fn check_kind(kind: &str) {
    let (k_freq, rec) = force_constant_from_frequencies(kind, "freq");
    let k_fd = force_constant_from_energies(kind);
    assert!(
        (k_freq - k_fd).abs() < BAR,
        "{kind}: k from the gradient Hessian {k_freq:.8} vs energy stencil {k_fd:.8}"
    );
    // The Hessian says the gradient was differentiated, and which one.
    let cert = &rec["certificate"];
    let approx = |name: &str| -> String {
        cert["approximations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["name"] == name)
            .unwrap_or_else(|| panic!("{kind}: no {name} approximation in {cert}"))["detail"]
            .as_str()
            .unwrap()
            .to_string()
    };
    assert!(
        approx("hessian").contains("central differences of the analytic gradient"),
        "{kind}: {cert}"
    );
    let gradient = approx("gradient");
    assert!(gradient.contains("analytic"), "{kind}: {gradient}");
    assert_eq!(rec["n_gradient_evaluations"], 12, "{kind}: 6N for N = 2");
}

#[test]
fn rimp2_frequencies_match_an_energy_stencil() {
    check_kind("rimp2");
}

#[test]
fn scs_mp2_frequencies_match_an_energy_stencil() {
    check_kind("scs-mp2");
}

#[test]
fn oo_rimp2_frequencies_match_an_energy_stencil() {
    check_kind("oo-rimp2");
}

#[test]
fn att_rimp2_frequencies_match_an_energy_stencil() {
    check_kind("att-rimp2");
}

#[test]
fn pdep_rpa_frequencies_match_an_energy_stencil() {
    check_kind("pdep-rpa");
}

/// The correlated frequency run is not the RHF one: its force constant sits
/// at least 50 bars from the SCF Hessian of the same molecule (measured gap 0.0099).
#[test]
fn rimp2_frequencies_differ_from_the_rhf_hessian() {
    let (k_mp2, _) = force_constant_from_frequencies("rimp2", "freq_vs_rhf");
    let xyz = h2_xyz("freq_rhf_ref", R0_ANG);
    let (out, records) = run_cli(
        "freq_rhf_ref",
        &format!(
            "[molecule]\nxyz = \"{}\"\n[basis]\nname = \"sto-3g\"\n[method]\nkind = \"rhf\"\ntask = \"frequencies\"\n",
            xyz.display()
        ),
    );
    assert!(out.status.success());
    let rec = records
        .iter()
        .find(|r| r["record"] == "frequencies")
        .unwrap();
    let h: Vec<f64> = rec["cartesian_hessian"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let k_rhf = h[14];
    assert!((k_mp2 - k_rhf).abs() > 50.0 * BAR, "{k_mp2} vs {k_rhf}");
}

/// A kind with no nuclear gradient is refused, and the message lists the
/// kinds that have one.
#[test]
fn a_kind_without_a_gradient_is_refused_with_the_supported_list() {
    let xyz = h2_xyz("freq_refused", R0_ANG);
    let (out, _) = run_cli(
        "freq_refused",
        &format!(
            "[molecule]\nxyz = \"{}\"\n[basis]\nname = \"sto-3g\"\n[method]\nkind = \"ccsd\"\ntask = \"frequencies\"\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n",
            xyz.display()
        ),
    );
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("needs a method.kind with a nuclear gradient"),
        "{err}"
    );
    for kind in [
        "rhf",
        "ksdft",
        "uhf",
        "rohf",
        "pdep-rpa",
        "rimp2",
        "scs-mp2",
        "oo-rimp2",
        "att-rimp2",
    ] {
        assert!(err.contains(kind), "{kind} missing from: {err}");
    }
}

/// An SCF Hessian request on a correlated kind is refused, not ignored.
#[test]
fn an_scf_hessian_key_on_a_correlated_kind_is_refused() {
    let xyz = h2_xyz("freq_hessian_key", R0_ANG);
    let (out, _) = run_cli(
        "freq_hessian_key",
        &toml_for(
            &xyz,
            "rimp2",
            "frequencies",
            "[frequencies]\nhessian = \"analytic\"\n",
        ),
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("SCF Hessian"));
}

/// Open-shell RI-MP2 and attenuated RI-MP2 (OH doublet, UHF reference,
/// unrestricted gradient): the frequency Hessian's zz element against a
/// five-point stencil of the same kind's energy, same bar as above (the bond
/// is stiffer, ~0.5 Ha/Bohr^2, and the stencil error scales with the same
/// derivatives).
fn check_open_shell_kind(kind: &str) {
    const R0: f64 = 0.97;
    let doublet = |toml: String| toml.replacen("[basis]", "multiplicity = 2\n[basis]", 1);
    let oh_xyz = |tag: &str, r: f64| {
        let path = scratch().join(format!("{tag}.xyz"));
        std::fs::write(&path, format!("2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 {r:.12}\n")).unwrap();
        path
    };
    let energy_at = |k: i32| {
        let tag = format!("oh_{kind}_e_{k}");
        let xyz = oh_xyz(&tag, R0 + f64::from(k) * H_ANG);
        let (out, records) = run_cli(&tag, &doublet(toml_for(&xyz, kind, "energy", "")));
        assert!(
            out.status.success(),
            "{tag}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        records
            .iter()
            .rev()
            .find(|r| r["record"] == "result" && r["kind"] == kind)
            .unwrap_or_else(|| panic!("{tag}: no result record"))["total"]
            .as_f64()
            .unwrap()
    };
    let h = H_ANG / ANGSTROM_PER_BOHR;
    let (m2, m1, p0, p1, p2) = (
        energy_at(-2),
        energy_at(-1),
        energy_at(0),
        energy_at(1),
        energy_at(2),
    );
    let k_fd = (-p2 + 16.0 * p1 - 30.0 * p0 + 16.0 * m1 - m2) / (12.0 * h * h);

    let tag = format!("oh_{kind}_freq");
    let xyz = oh_xyz(&tag, R0);
    let (out, records) = run_cli(&tag, &doublet(toml_for(&xyz, kind, "frequencies", "")));
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rec = records
        .iter()
        .find(|r| r["record"] == "frequencies")
        .unwrap();
    let k_freq = rec["cartesian_hessian"][14].as_f64().unwrap();
    eprintln!(
        "OH {kind}: k_freq {k_freq:.8} k_fd {k_fd:.8} diff {:.3e}",
        k_freq - k_fd
    );
    assert!(
        (k_freq - k_fd).abs() < BAR,
        "OH {kind}: k from the gradient Hessian {k_freq:.8} vs energy stencil {k_fd:.8}"
    );
}

#[test]
fn open_shell_rimp2_frequencies_match_an_energy_stencil() {
    check_open_shell_kind("rimp2");
}

#[test]
fn open_shell_att_rimp2_frequencies_match_an_energy_stencil() {
    check_open_shell_kind("att-rimp2");
}
