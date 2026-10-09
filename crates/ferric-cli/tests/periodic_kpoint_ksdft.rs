//! The `[cell] kmesh` + `ksdft` CLI path is the library `solve_krks`, end to end.
//!
//! Anchor: the CLI energy (read at full precision from the JSON run log's
//! `run_end` record, not the 10-decimal stdout line) equals the library
//! `ferric_pbc::solve_krks` energy bit for bit on the same input. The
//! unsupported combinations (open shell, meta-GGA, range-separated hybrids,
//! k-point forces) are refused by name before any integral is built.

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
    let e = refusal("uks", "ksdft", 3, "", "");
    assert!(e.contains("UKS") && e.contains("not implemented"), "{e}");
    let e = refusal("roks", "rohf", 3, "[dft]\nfunctional = \"PBE\"", "");
    assert!(e.contains("ROKS") && e.contains("not implemented"), "{e}");
    let e = refusal("meta", "ksdft", 1, "[dft]\nfunctional = \"SCAN\"", "");
    assert!(e.contains("meta-GGA"), "{e}");
    let e = refusal("rsh", "ksdft", 1, "[dft]\nfunctional = \"HSE06\"", "");
    assert!(e.contains("range-separated"), "{e}");
    let toml = input("force", "ksdft", 1, "", "");
    let src = std::fs::read_to_string(&toml)
        .unwrap()
        .replace("kind = \"ksdft\"", "task = \"optimize\"\nkind = \"ksdft\"");
    std::fs::write(&toml, src).unwrap();
    let out = run(&toml);
    assert!(!out.status.success());
    let e = String::from_utf8_lossy(&out.stderr);
    assert!(
        e.contains("optimize") && e.contains("k-point forces"),
        "{e}"
    );
}
