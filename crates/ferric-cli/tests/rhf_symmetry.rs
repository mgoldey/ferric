//! `[scf] symmetry = true` is the library `solve_rhf_symmetric`, end to end (#294).
//!
//! Anchor: the CLI energy (full precision, from the JSON run log's `run_end`
//! record) equals the library energy bit for bit on the same RhfConfig. The
//! unsupported kinds and tasks are refused by name before any integral is
//! built, and `symmetry = false` (the default) runs the unchanged path.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::symmetric::solve_rhf_symmetric;
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

fn input(tag: &str, kind: &str, task: &str, extra: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferric-sym-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let toml = dir.join("in.toml");
    let xyz = workspace_root().join("testdata/molecules/water.xyz");
    std::fs::write(
        &toml,
        format!(
            "[molecule]\nxyz = \"{xyz}\"\n[basis]\nname = \"6-31g\"\n[method]\nkind = \"{kind}\"\n\
             task = \"{task}\"\n[output]\njson = \"{json}\"\n{extra}",
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

fn run_end(toml: &Path) -> serde_json::Value {
    let log = std::fs::read_to_string(toml.with_file_name("run.jsonl")).expect("run log");
    for line in log.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if v.get("record").and_then(|r| r.as_str()) == Some("run_end") {
            return v;
        }
    }
    panic!("no run_end record in:\n{log}");
}

fn text(o: &Output) -> String {
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn library(irrep_nelec: Option<Vec<(String, usize)>>) -> (f64, String) {
    let xyz = workspace_root().join("testdata/molecules/water.xyz");
    let mol = Molecule::load_xyz(xyz.to_str().unwrap()).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let config = RhfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        symmetry: true,
        irrep_nelec,
        ..Default::default()
    };
    let r =
        solve_rhf_symmetric(&ParallelContext::new(), &mol, &prep, op, &bounds, &config).unwrap();
    (r.energy(), r.group().to_string())
}

#[test]
fn cli_symmetry_equals_library_solve_rhf_symmetric_bit_for_bit() {
    for (tag, extra, req) in [
        ("ground", String::new(), None),
        (
            "excited",
            "irrep_nelec = { A1 = 8, B1 = 0, B2 = 2 }\n".to_string(),
            Some(vec![("A1".into(), 8), ("B1".into(), 0), ("B2".into(), 2)]),
        ),
    ] {
        let toml = input(
            tag,
            "rhf",
            "energy",
            &format!("[scf]\n{SCF}symmetry = true\n{extra}"),
        );
        let out = run(&toml);
        assert!(out.status.success(), "{tag}:\n{}", text(&out));
        let (lib, group) = library(req);
        let end = run_end(&toml);
        assert_eq!(
            end["energy"].as_f64().unwrap(),
            lib,
            "{tag}: CLI vs library"
        );
        let sym = &end["extra"]["symmetry"];
        assert_eq!(sym["point_group"], group);
        assert_eq!(sym["geometry"], "symmetrized");
        assert!(sym["max_symmetrization_shift_bohr"].as_f64().unwrap() < 1e-4);
        let n: u64 = sym["irrep_nelec"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_u64().unwrap())
            .sum();
        assert_eq!(n, 10, "{tag}: occupations must sum to the electron count");
        let stdout = text(&out);
        assert!(stdout.contains("point group = C2v"), "{stdout}");
        assert!(stdout.contains("irrep electrons = "), "{stdout}");
        if tag == "excited" {
            assert_eq!(sym["irrep_nelec"]["A1"], 8);
            assert_eq!(sym["irrep_nelec"]["B1"], 0);
        }
    }
}

#[test]
fn symmetry_off_is_the_unchanged_path() {
    let plain = input("off-default", "rhf", "energy", &format!("[scf]\n{SCF}"));
    let off = input(
        "off-false",
        "rhf",
        "energy",
        &format!("[scf]\n{SCF}symmetry = false\n"),
    );
    assert!(run(&plain).status.success() && run(&off).status.success());
    let (a, b) = (run_end(&plain), run_end(&off));
    assert_eq!(a["energy"], b["energy"]);
    assert!(a["extra"].get("symmetry").is_none() && b["extra"].get("symmetry").is_none());
}

#[test]
fn unsupported_combinations_are_refused_by_name() {
    for (tag, kind, task, extra, needle) in [
        (
            "optimize",
            "rhf",
            "optimize",
            "[scf]\nsymmetry = true\n",
            "task = \"energy\" only",
        ),
        (
            "freq",
            "rhf",
            "frequencies",
            "[scf]\nsymmetry = true\n",
            "task = \"energy\" only",
        ),
        (
            "uhf",
            "uhf",
            "energy",
            "[scf]\nsymmetry = true\n",
            "kind = \"rhf\"",
        ),
        (
            "rimp2",
            "rimp2",
            "energy",
            "[scf]\nsymmetry = true\n",
            "kind = \"rhf\"",
        ),
        (
            "ksdft",
            "ksdft",
            "energy",
            "[scf]\nsymmetry = true\n",
            "kind = \"rhf\"",
        ),
        (
            "dft",
            "rhf",
            "energy",
            "[scf]\nsymmetry = true\n[dft]\nfunctional = \"PBE\"\n",
            "[dft] functional",
        ),
        (
            "orphan",
            "rhf",
            "energy",
            "[scf]\nirrep_nelec = { A1 = 10 }\n",
            "requires [scf] symmetry",
        ),
        (
            "ladder",
            "rhf",
            "energy",
            "[scf]\nsymmetry = true\n[[scf.ladder]]\nmax_iter = 3\n",
            "[[scf.ladder]]",
        ),
        (
            "dfguess",
            "rhf",
            "energy",
            "[scf]\nsymmetry = true\ndf_guess = true\n",
            "df_guess",
        ),
        (
            "cosx",
            "rhf",
            "energy",
            "[scf]\nsymmetry = true\nk_builder = \"cosx\"\n",
            "cosx",
        ),
        (
            "solvent",
            "rhf",
            "energy",
            "[scf]\nsymmetry = true\n[cosmo]\nepsilon = 78.4\n",
            "symmetry",
        ),
    ] {
        let toml = input(tag, kind, task, extra);
        let out = run(&toml);
        assert!(
            !out.status.success(),
            "{tag} must be refused:\n{}",
            text(&out)
        );
        assert!(
            text(&out).contains(needle),
            "{tag}: wanted {needle:?} in\n{}",
            text(&out)
        );
    }
}
