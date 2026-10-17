//! The `[cell] kmesh` + `ksdft` CLI path is the library `solve_krks`, end to end.
//!
//! Anchor: the CLI energy (read at full precision from the JSON run log's
//! `run_end` record, not the 10-decimal stdout line) equals the library
//! `ferric_pbc::solve_krks` energy bit for bit on the same input. The
//! unsupported combinations (open-shell meta-GGA, open-shell and force
//! range-separated hybrids) are refused by name before any integral is built.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::{
    solve_krks, Cell, KPointMesh, KRksConfig, PeriodicGridConfig, PeriodicHcoreConfig,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

const OMEGA: f64 = 0.8;
const A_BOHR: f64 = 4.0;

/// H2 / STO-3G in a 4 Bohr cube on a 1x1x2 Gamma-centred mesh.
fn input(tag: &str, kind: &str, multiplicity: usize, dft: &str, extra_cell: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferric-kdft-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let toml = dir.join("in.toml");
    let xyz = workspace_root().join("testdata/molecules/h2.xyz");
    std::fs::write(
        &toml,
        format!(
            r#"[molecule]
xyz = "{xyz}"
multiplicity = {multiplicity}
[basis]
name = "sto-3g"
[method]
kind = "{kind}"
{dft}
[output]
json = "{json}"
[cell]
unit = "bohr"
lattice = [[{a}, 0.0, 0.0], [0.0, {a}, 0.0], [0.0, 0.0, {a}]]
kmesh = [1, 1, 2]
exxdiv = "ewald"
omega = {OMEGA}
n_radial = 40
n_angular = 110
{extra_cell}
"#,
            xyz = xyz.display(),
            json = dir.join("run.jsonl").display(),
            a = A_BOHR,
        ),
    )
    .unwrap();
    toml
}

fn run(toml: &Path) -> Output {
    Command::new(ferric_cli_bin())
        .arg(toml)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("run ferric-cli")
}

/// `run_end.energy` from the JSON run log, as the exact f64.
fn logged_energy(toml: &Path) -> f64 {
    let log = std::fs::read_to_string(toml.with_file_name("run.jsonl")).expect("run log");
    for line in log.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if v.get("record").and_then(|r| r.as_str()) == Some("run_end") {
            return v["energy"]
                .as_f64()
                .unwrap_or_else(|| panic!("no energy in {line}"));
        }
    }
    panic!("no run_end record in:\n{log}");
}

fn library_energy(functional: &str) -> f64 {
    let xyz = workspace_root().join("testdata/molecules/h2.xyz");
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let a = A_BOHR;
    let cell = Cell::new(mol, [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]).unwrap();
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let mut c = KRksConfig::new(&cell, ExxDiv::Ewald, functional);
    c.krhf.scf.max_iter = 200;
    c.krhf.scf.energy_conv = 1e-12;
    c.krhf.scf.grad_conv = 1e-9;
    c.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.grid = PeriodicGridConfig::with_size(40, 110);
    solve_krks(&cell, &prep, None, &mesh, &c)
        .unwrap()
        .scf
        .energy
}

#[test]
fn cli_kpoint_ksdft_equals_library_solve_krks_bit_for_bit() {
    for (tag, kind, dft, functional) in [
        ("lda", "ksdft", "", "LDA"),
        ("pbe", "rhf", "[dft]\nfunctional = \"PBE\"", "PBE"),
        ("pbe0", "ksdft", "[dft]\nfunctional = \"PBE0\"", "PBE0"),
    ] {
        let toml = input(tag, kind, 1, dft, "");
        let out = run(&toml);
        assert!(
            out.status.success(),
            "{tag}:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let cli = logged_energy(&toml);
        let lib = library_energy(functional);
        assert_eq!(
            cli.to_bits(),
            lib.to_bits(),
            "{tag}: CLI {cli:.15} vs library {lib:.15}"
        );
    }
}

fn refusal(tag: &str, kind: &str, mult: usize, dft: &str, cell: &str) -> String {
    let out = run(&input(tag, kind, mult, dft, cell));
    assert!(!out.status.success(), "{tag} must be refused");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn unsupported_kpoint_ks_combinations_are_refused_by_name() {
    // Open-shell meta-GGA stays refused (UKS/ROKS meta-GGA has no tau term).
    let e = refusal("uks-meta", "ksdft", 3, "[dft]\nfunctional = \"SCAN\"", "");
    assert!(e.contains("meta-GGA"), "{e}");
    // Open-shell range-separated hybrids stay refused on the dense J/K too.
    // The early gate (config.rs), not a later library refusal, names it.
    for (tag, kind) in [("rsh-uks", "uhf"), ("rsh-roks", "rohf")] {
        let e = refusal(tag, kind, 3, "[dft]\nfunctional = \"HSE06\"", "");
        assert!(
            e.contains("[cell] kmesh with functional \"HSE06\"") && e.contains("range-separated"),
            "{tag}: {e}"
        );
    }
    // task = "optimize" keeps the range-separated refusal (closed-shell LDA /
    // GGA / global hybrid / SCAN forces are wired; periodic_kpoint_ks_optimize.rs).
    let toml = input("opt-hse", "ksdft", 1, "[dft]\nfunctional = \"HSE06\"", "");
    let src = std::fs::read_to_string(&toml)
        .unwrap()
        .replace("kind = \"ksdft\"", "task = \"optimize\"\nkind = \"ksdft\"");
    std::fs::write(&toml, src).unwrap();
    let out = run(&toml);
    assert!(!out.status.success(), "opt-hse");
    let e = String::from_utf8_lossy(&out.stderr);
    assert!(
        e.contains("method.task = \"optimize\"") && e.contains("range-separated"),
        "opt-hse: {e}"
    );
}

/// The CLI range-separated hybrid on the default dense-AFT J/K is the library
/// `solve_krks` on the same J/K, bit for bit, and is not the global hybrid.
#[test]
fn cli_kpoint_rsh_dense_equals_library_solve_krks_bit_for_bit() {
    for (tag, name) in [("hse", "HSE06"), ("cam", "HYB_GGA_XC_CAM_B3LYP")] {
        let dft = format!("[dft]\nfunctional = \"{name}\"");
        let toml = input(&format!("dense-{tag}"), "ksdft", 1, &dft, "");
        let out = run(&toml);
        assert!(
            out.status.success(),
            "{tag}:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let cli = logged_energy(&toml);
        let lib = library_energy(name);
        assert_eq!(
            cli.to_bits(),
            lib.to_bits(),
            "{tag}: CLI {cli:.15} vs library {lib:.15}"
        );
        assert!(
            (cli - library_energy("PBE0")).abs() > 1e-4,
            "{tag}: RSH equals PBE0?"
        );
    }
}

/// Library energy of `functional` on the RS-GDF J/K (cc-pVDZ-RI aux), the
/// same cell, mesh, grid and hcore as [`library_energy`].
fn library_energy_rsgdf(functional: &str) -> f64 {
    use ferric_pbc::KJkKind;
    let xyz = workspace_root().join("testdata/molecules/h2.xyz");
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let a = A_BOHR;
    let cell = Cell::new(mol, [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]).unwrap();
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let mut c = KRksConfig::new(&cell, ExxDiv::Ewald, functional);
    c.krhf.jk = KJkKind::RsGdf;
    c.krhf.scf.max_iter = 200;
    c.krhf.scf.energy_conv = 1e-12;
    c.krhf.scf.grad_conv = 1e-9;
    c.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.grid = PeriodicGridConfig::with_size(40, 110);
    solve_krks(&cell, &prep, Some(&aux), &mesh, &c)
        .unwrap()
        .scf
        .energy
}

const RSGDF_CELL: &str = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";

/// The CLI range-separated hybrid on `jk = "rsgdf"` is the library
/// `solve_krks` on the same J/K (the SR column rotation of the CLI's
/// hcore/fit matches the library's default to the screening precision, so the
/// bar is 1e-8, not bits); the mixed Madelung constant is what the log reports.
#[test]
fn cli_kpoint_rsh_rsgdf_equals_library_solve_krks() {
    for (tag, name) in [("hse", "HSE06"), ("cam", "HYB_GGA_XC_CAM_B3LYP")] {
        let dft = format!("[dft]\nfunctional = \"{name}\"");
        let toml = input(&format!("rsgdf-{tag}"), "ksdft", 1, &dft, RSGDF_CELL);
        let out = run(&toml);
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(
            out.status.success(),
            "{tag}:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let cli = logged_energy(&toml);
        let lib = library_energy_rsgdf(name);
        assert!(
            (cli - lib).abs() < 1e-8,
            "{tag}: CLI {cli:.12} vs library {lib:.12}"
        );
        // The Madelung line carries the MIXED constant, not the full-Coulomb one.
        let plain = library_energy_rsgdf("PBE0");
        assert!((cli - plain).abs() > 1e-4, "{tag}: RSH equals PBE0?");
        let line = stdout
            .lines()
            .find(|l| l.trim_start().starts_with("madelung"))
            .unwrap_or_else(|| panic!("{tag}: no madelung line:\n{stdout}"));
        let full = 0.451_460_452_613; // PySCF madelung(cell, 1x1x2 kpts), H2 a = 4 (pbc_krsh.rs)
        let printed: f64 = line
            .split('=')
            .nth(1)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            (printed - full).abs() > 1e-3,
            "{tag}: printed madelung {printed} is the full-Coulomb constant"
        );
    }
}

/// With `jk = "rsgdf"` the range-separated energy is accepted, but the rest of
/// the surface stays refused by name: open-shell routes and
/// `task = "optimize"` (forces/stress are not implemented).
#[test]
fn cli_kpoint_rsh_rsgdf_keeps_forces_and_open_shell_refused() {
    let hse = "[dft]\nfunctional = \"HSE06\"";
    let e = refusal("rsgdf-uks", "ksdft", 3, hse, RSGDF_CELL);
    assert!(
        e.contains("[cell] kmesh with functional \"HSE06\"") && e.contains("range-separated"),
        "{e}"
    );
    let toml = input("rsgdf-opt", "ksdft", 1, hse, RSGDF_CELL);
    let src = std::fs::read_to_string(&toml)
        .unwrap()
        .replace("kind = \"ksdft\"", "task = \"optimize\"\nkind = \"ksdft\"");
    std::fs::write(&toml, src).unwrap();
    let out = run(&toml);
    assert!(!out.status.success(), "rsgdf-opt must be refused");
    let e = String::from_utf8_lossy(&out.stderr);
    assert!(
        e.contains("method.task = \"optimize\"") && e.contains("range-separated"),
        "rsgdf-opt: {e}"
    );
}
