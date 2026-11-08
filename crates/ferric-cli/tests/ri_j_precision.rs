//! `[scf] ri_j_precision`, end to end through the CLI: the default stores the
//! RI-J tensor as f32, `"f64"` and `"f32"` are honoured as given, the run log
//! records the precision the tensor was stored at (`ri_j_tensor`) and the
//! certificate names it, the f64 run equals the library `solve_rhf` with
//! `ri_j_precision: Some(F64)` bit for bit, and an unknown value is refused
//! before any integral is built.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::RiJPrecision;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
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

const SCF: &str = "max_iter = 200\nenergy_conv = 1e-12\ndensity_conv = 1e-10\n";

fn input(tag: &str, scf_extra: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferric-rijp-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let toml = dir.join("in.toml");
    let xyz = workspace_root().join("testdata/molecules/water.xyz");
    std::fs::write(
        &toml,
        format!(
            "[molecule]\nxyz = \"{xyz}\"\n[basis]\nname = \"6-31g\"\n[method]\nkind = \"rhf\"\n\
             task = \"energy\"\n[output]\njson = \"{json}\"\n[scf]\n{SCF}\
             df_j_aux = \"def2-universal-jkfit\"\n{scf_extra}",
            xyz = xyz.display(),
            json = dir.join("run.jsonl").display(),
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

fn records(toml: &Path, kind: &str) -> Vec<serde_json::Value> {
    let log = std::fs::read_to_string(toml.with_file_name("run.jsonl")).expect("run log");
    log.lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v.get("record").and_then(|r| r.as_str()) == Some(kind))
        .collect()
}

fn text(o: &Output) -> String {
    format!(
        "status {:?}\nstdout:\n{}\nstderr:\n{}",
        o.status,
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

/// What one run stored the RI-J tensor at, its energy and the certificate's
/// `ri-j` detail.
fn outcome(toml: &Path) -> (String, f64, String) {
    let o = run(toml);
    assert!(o.status.success(), "{}", text(&o));
    // The MINAO free-atom SCFs log their own (always f64) RI-J tensors first;
    // the molecule's SCF is the last record.
    let tensor = records(toml, "ri_j_tensor");
    assert!(!tensor.is_empty(), "no ri_j_tensor record");
    for atom in &tensor[..tensor.len() - 1] {
        assert_eq!(atom["precision"], "f64", "free-atom guess SCF: {atom}");
    }
    let used = tensor.last().unwrap()["precision"]
        .as_str()
        .unwrap()
        .to_string();
    let end = records(toml, "run_end");
    let end = end.last().expect("run_end");
    let e = end["energy"].as_f64().unwrap();
    let detail = end["extra"]["certificate"]["approximations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["name"] == "ri-j")
        .map(|a| a["detail"].as_str().unwrap().to_string())
        .expect("ri-j approximation");
    (used, e, detail)
}

fn library_energy(p: RiJPrecision) -> f64 {
    let xyz = workspace_root().join("testdata/molecules/water.xyz");
    let mol = Molecule::load_xyz(xyz.to_str().unwrap()).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some("def2-universal-jkfit".into()),
        ri_j_precision: Some(p),
        ..Default::default()
    };
    solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg)
        .unwrap()
        .energy
}

#[test]
fn the_default_is_f32_and_each_explicit_value_is_honoured_and_recorded() {
    let (d_used, d_e, d_cert) = outcome(&input("default", ""));
    assert_eq!(d_used, "f32", "default");
    assert!(d_cert.ends_with("f32 tensor storage"), "{d_cert}");

    let (f32_used, f32_e, _) = outcome(&input("f32", "ri_j_precision = \"f32\"\n"));
    assert_eq!(f32_used, "f32");
    assert_eq!(f32_e.to_bits(), d_e.to_bits(), "explicit f32 != default");

    let (f64_used, f64_e, f64_cert) = outcome(&input("f64", "ri_j_precision = \"f64\"\n"));
    assert_eq!(f64_used, "f64");
    assert!(f64_cert.ends_with("f64 tensor storage"), "{f64_cert}");
    assert_ne!(
        f64_e.to_bits(),
        d_e.to_bits(),
        "f32 left the energy unchanged"
    );
    // The CLI's f64 run is the library's f64 run.
    assert_eq!(f64_e.to_bits(), library_energy(RiJPrecision::F64).to_bits());
    assert_eq!(d_e.to_bits(), library_energy(RiJPrecision::F32).to_bits());
}

#[test]
fn an_unknown_ri_j_precision_is_refused() {
    let toml = input("bad", "ri_j_precision = \"half\"\n");
    let o = run(&toml);
    assert!(!o.status.success(), "{}", text(&o));
    let t = text(&o);
    assert!(t.contains("ri_j_precision") && t.contains("\"f64\""), "{t}");
}
