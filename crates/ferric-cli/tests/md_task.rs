//! `method.task = "md"`: the shipped example runs end to end and writes its
//! trajectory and energy files, the CLI trajectory is the library's
//! `ferric_scf::run_aimd` trajectory, and every `[md]` validation rule
//! refuses its input before any SCF.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_md::{MdConfig, Thermostat};
use ferric_scf::rhf::RhfConfig;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

/// Workspace root, resolved at RUN time (see `epistemic_warning.rs`).
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

/// A fresh directory under `target/` for one test's input and outputs.
fn scratch(tag: &str) -> PathBuf {
    let dir = workspace_root()
        .join("target")
        .join(format!("md_task_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Run the CLI on `body`, written to `<scratch(tag)>/<tag>.toml`, from the
/// workspace root (so `testdata/...` paths resolve).
fn run_toml(tag: &str, body: &str) -> (std::process::Output, PathBuf) {
    let dir = scratch(tag);
    let path = dir.join(format!("{tag}.toml"));
    std::fs::write(&path, body).expect("write toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&path)
        .arg("--no-json")
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("failed to run ferric-cli binary");
    (out, dir)
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// The energy CSV as rows of numbers (header checked and dropped).
fn csv_rows(path: &Path) -> Vec<Vec<f64>> {
    let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = s.lines();
    assert_eq!(
        lines.next(),
        Some("step,time_fs,e_pot,e_kin,e_tot,conserved,temperature_k")
    );
    lines
        .map(|l| l.split(',').map(|t| t.parse().unwrap()).collect())
        .collect()
}

/// The step of each frame of a multi-frame XYZ file (each a count line, a
/// comment line `step=.. time_fs=..` and that many atom lines).
fn xyz_steps(path: &Path) -> Vec<usize> {
    let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = s.lines();
    let mut steps = Vec::new();
    while let Some(count) = lines.next() {
        let atoms: usize = count.trim().parse().expect("atom count line");
        let comment = lines.next().expect("comment line");
        let step = comment
            .strip_prefix("step=")
            .and_then(|r| r.split_whitespace().next())
            .and_then(|t| t.parse().ok())
            .unwrap_or_else(|| panic!("bad comment line {comment:?}"));
        steps.push(step);
        for _ in 0..atoms {
            let f: Vec<&str> = lines
                .next()
                .expect("atom line")
                .split_whitespace()
                .collect();
            assert_eq!(f.len(), 4, "atom line {f:?}");
        }
    }
    steps
}

/// The shipped example: 20 NVE steps of water/STO-3G write 21 frames to the
/// default `<stem>.md.xyz` / `.md.csv` beside the input, the conserved
/// energy stays flat, and the ungraded `md:rhf` certificate is printed.
#[test]
fn shipped_example_writes_trajectory_and_energies() {
    let root = workspace_root();
    let body = std::fs::read_to_string(root.join("examples/water-rhf-md.toml")).unwrap();
    let (out, dir) = run_toml("water-rhf-md", &body);
    let stdout = text(&out.stdout);
    assert!(
        out.status.success(),
        "stdout:\n{stdout}\nstderr:\n{}",
        text(&out.stderr)
    );
    assert_eq!(
        xyz_steps(&dir.join("water-rhf-md.md.xyz")),
        (0..=20).collect::<Vec<_>>()
    );
    let rows = csv_rows(&dir.join("water-rhf-md.md.csv"));
    assert_eq!(rows.len(), 21);
    assert_eq!(rows[0][0], 0.0);
    assert_eq!(rows[20][0], 20.0);
    assert!((rows[20][1] - 5.0).abs() < 1e-12, "time of step 20");
    // Initial temperature is rescaled to exactly 300 K over 3N - 3 = 6 dof.
    assert!((rows[0][6] - 300.0).abs() < 1e-6, "T0 = {}", rows[0][6]);
    let c0 = rows[0][5];
    let drift = rows.iter().map(|r| (r[5] - c0).abs()).fold(0.0, f64::max);
    // Measured 1.8e-5 Hartree (the velocity Verlet error at 0.25 fs) while
    // the potential energy swings by 3.8e-3.
    assert!(drift < 5e-5, "conserved drift {drift:.3e} Hartree");
    let swing = rows
        .iter()
        .map(|r| (r[2] - rows[0][2]).abs())
        .fold(0.0, f64::max);
    assert!(
        swing > 1e-3,
        "the potential energy barely moved: {swing:.3e}"
    );
    assert!(stdout.contains("md:rhf"), "no md certificate:\n{stdout}");
    assert!(stdout.contains("Molecular Dynamics Result:"), "{stdout}");
}

/// H2 / STO-3G, `kind = "rhf"`, with `extra` keys in `[md]`.
fn h2_body(extra_md: &str) -> String {
    format!(
        "[molecule]\nxyz = \"testdata/molecules/h2.xyz\"\n\n[basis]\nname = \"sto-3g\"\n\n\
         [method]\nkind = \"rhf\"\ntask = \"md\"\n\n\
         [scf]\ndensity_conv = 1e-10\nenergy_conv = 1e-10\n\n[md]\n{extra_md}"
    )
}

/// The CLI's Langevin trajectory (explicit output paths, stride 2) equals the
/// library's `run_aimd` with the same settings, frame for frame.
#[test]
fn cli_trajectory_is_the_library_trajectory() {
    let dir = scratch("h2_langevin_paths");
    let xyz = dir.join("traj.xyz");
    let csv = dir.join("e.csv");
    let body = h2_body(&format!(
        "n_steps = 6\ndt_fs = 0.25\nthermostat = \"langevin\"\nfriction_per_ps = 50.0\n\
         temperature_k = 500.0\nseed = 7\nstride = 2\ntrajectory = {:?}\nenergies = {:?}\n",
        xyz.display().to_string(),
        csv.display().to_string()
    ));
    let (out, _) = run_toml("h2_langevin", &body);
    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        text(&out.stdout),
        text(&out.stderr)
    );
    assert_eq!(xyz_steps(&xyz), [0, 2, 4, 6]);
    let rows = csv_rows(&csv);
    assert_eq!(
        rows.iter().map(|r| r[0]).collect::<Vec<_>>(),
        [0.0, 2.0, 4.0, 6.0]
    );

    let path = workspace_root().join("testdata/molecules/h2.xyz");
    let mol = Molecule::load_xyz_with_charge(path.to_str().unwrap(), 0, 1).unwrap();
    let mut scf = RhfConfig::default()
        .with_density_conv(1e-10)
        .with_energy_conv(1e-10);
    scf.ri_j_task = ferric_scf::df_j::RiJTask::Geometry;
    let config = ferric_scf::AimdConfig {
        md: MdConfig {
            n_steps: 6,
            dt_fs: 0.25,
            thermostat: Thermostat::Langevin {
                friction_per_ps: 50.0,
            },
            temperature_k: Some(500.0),
            seed: 7,
            stride: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let r = ferric_scf::run_aimd(
        &ParallelContext::default(),
        &mol,
        "sto-3g",
        Operator::coulomb(),
        &scf,
        ferric_scf::Spin::Restricted,
        &config,
        None,
        &mut [],
    )
    .unwrap();
    assert_eq!(r.md.frames.len(), rows.len());
    for (f, row) in r.md.frames.iter().zip(&rows) {
        // The CSV prints 12 decimals.
        assert!(
            (f.potential - row[2]).abs() < 1e-10,
            "{} vs {}",
            f.potential,
            row[2]
        );
        assert!(
            (f.temperature_k - row[6]).abs() < 1e-6,
            "{} vs {}",
            f.temperature_k,
            row[6]
        );
    }
}

/// Run `body` and require a refusal whose stderr contains every needle.
fn assert_refused(tag: &str, body: &str, needles: &[&str]) {
    let (out, _) = run_toml(tag, body);
    let err = text(&out.stderr);
    assert!(
        !out.status.success(),
        "{tag} must be refused; stdout:\n{}",
        text(&out.stdout)
    );
    for n in needles {
        assert!(err.contains(n), "{tag}: stderr lacks {n:?}:\n{err}");
    }
    assert!(
        !text(&out.stdout).contains("SCF converged"),
        "{tag}: refused only after an SCF"
    );
}

#[test]
fn md_section_without_md_task_is_refused() {
    let body = h2_body("n_steps = 2\n").replace("task = \"md\"", "task = \"energy\"");
    assert_refused(
        "md_wrong_task",
        &body,
        &["[md] is read by method.task = \"md\" only"],
    );
}

#[test]
fn md_task_needs_md_section_and_n_steps() {
    let no_section = h2_body("").replace("\n[md]\n", "\n");
    assert_refused("md_no_section", &no_section, &["needs an [md] section"]);
    assert_refused(
        "md_no_n_steps",
        &h2_body("dt_fs = 0.5\n"),
        &["[md] n_steps is required"],
    );
}

#[test]
fn md_task_refuses_kinds_without_an_scf_gradient() {
    let body = h2_body("n_steps = 2\n").replace("kind = \"rhf\"", "kind = \"rimp2\"");
    assert_refused(
        "md_rimp2",
        &body,
        &["\"rhf\", \"uhf\", \"rohf\", \"ksdft\"", "got \"rimp2\""],
    );
}

#[test]
fn md_task_refuses_unwired_features() {
    let base = h2_body("n_steps = 2\n");
    let cases = [
        ("md_cosmo", "[cosmo]\nepsilon = 78.4\n", "[cosmo]"),
        ("md_pcm", "[pcm]\nsolvent = \"water\"\n", "[pcm]"),
        (
            "md_cell",
            "[cell]\nlattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]\n",
            "[cell]",
        ),
        (
            "md_qmmm",
            "[qmmm]\npqr = \"testdata/molecules/water_na.pqr\"\nqm_indices = [0, 1, 2]\n",
            "[qmmm]",
        ),
    ];
    for (tag, extra, what) in cases {
        let body = format!("{base}\n{extra}");
        assert_refused(
            tag,
            &body,
            &[what, "not supported with method.task = \"md\""],
        );
    }
    let rel = base.replace("[scf]\n", "[scf]\nrelativistic = \"x2c1e\"\n");
    assert_refused("md_x2c", &rel, &["[scf] relativistic", "task = \"md\""]);
    let sym = base.replace("[scf]\n", "[scf]\nsymmetry = true\n");
    assert_refused("md_symmetry", &sym, &["[scf] symmetry", "task = \"md\""]);
}

#[test]
fn md_thermostat_parameters_are_strict() {
    let cases = [
        (
            "md_berendsen_no_tau",
            "n_steps = 2\nthermostat = \"berendsen\"\ntemperature_k = 300.0\n",
            "requires tau_fs",
        ),
        (
            "md_langevin_no_friction",
            "n_steps = 2\nthermostat = \"langevin\"\ntemperature_k = 300.0\n",
            "requires friction_per_ps",
        ),
        (
            "md_nve_tau",
            "n_steps = 2\ntau_fs = 100.0\n",
            "tau_fs applies only to berendsen",
        ),
        (
            "md_unknown_thermostat",
            "n_steps = 2\nthermostat = \"nose-hoover\"\n",
            "unknown thermostat",
        ),
        (
            "md_langevin_no_temperature",
            "n_steps = 2\nthermostat = \"langevin\"\nfriction_per_ps = 10.0\n",
            "requires temperature_k",
        ),
        (
            "md_bad_dt",
            "n_steps = 2\ndt_fs = 0.0\n",
            "dt_fs must be finite and positive",
        ),
        (
            "md_bad_stride",
            "n_steps = 2\nstride = 0\n",
            "stride must be at least 1",
        ),
        (
            "md_unknown_key",
            "n_steps = 2\nnhc_chain = 3\n",
            "nhc_chain",
        ),
    ];
    for (tag, md, needle) in cases {
        assert_refused(tag, &h2_body(md), &[needle]);
    }
}
