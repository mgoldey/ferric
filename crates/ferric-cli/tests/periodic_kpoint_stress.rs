//! `[cell] kmesh` + `stress = true`: the CLI prints the analytic k-point stress
//! and writes it to the run log's `run_end` record (issue #357).
//!
//! * EXACTNESS ANCHOR: the 3x3 tensor read from `run_end.extra` (full f64
//!   precision, not the printed digits) equals the library
//!   `ferric_pbc::kpoint_*_stress` of the SAME cell, mesh, grid and configs
//!   BIT FOR BIT, for RHF, UHF, ROHF, RKS (PBE, PBE0), UKS and ROKS on a
//!   Gamma-centred AND a shifted (Monkhorst-Pack, `centring = "mp"`) 1x1x2
//!   mesh. The library side is the same call sequence the Python
//!   `run_*_kpts(..., with_stress=True)` makes (checked by hand against the
//!   built extension; see the commit message).
//! * The GPa tensor is `sigma * ferric_core::units::HARTREE_PER_BOHR3_TO_GPA`
//!   bit for bit.
//! * Refusals (range-separated hybrid, open-shell meta-GGA, the Gamma point,
//!   `task = "optimize"`, a correlated route) are TOML-level errors naming
//!   `[cell] stress = true`.
//! * Mutants of this wiring (tensor rows reversed, GPa factor wrong, tensor
//!   not logged) are in the commit message; the in-test mutant checks assert
//!   the anchors can tell a reversed or rescaled tensor apart.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::units::{BOHR_RADIUS_ANGSTROM, HARTREE_PER_BOHR3_TO_GPA};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::kroks::{KRohfConfig, KRoksConfig};
use ferric_pbc::kstress::{kpoint_rhf_stress, kpoint_uhf_stress, KStressConfig};
use ferric_pbc::kxcgrad::KXcGrid;
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStressConfig};
use ferric_pbc::{
    kpoint_rohf_stress, kpoint_roks_stress, kpoint_uks_stress, solve_krhf_keep, solve_krks_keep,
    solve_krohf_keep, solve_kroks_keep, solve_kuhf_keep, solve_kuks_keep, Cell, ExxDiv, KPointMesh,
    KRhfConfig, KRksConfig, KScfConfig, KUhfConfig, KUksConfig, KUksStressConfig, MeshCentring,
    PeriodicGridConfig, PeriodicHcoreConfig,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const A_BOHR: f64 = 4.0;
const OMEGA: f64 = 0.8;
const N_RADIAL: usize = 20;
const N_ANGULAR: usize = 50;
const CUTOFF_BOHR: f64 = 14.0;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.6]];
const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Route {
    Rhf,
    Uhf,
    Rohf,
    Rks,
    Uks,
    Roks,
}

#[derive(Clone, Copy)]
struct Case {
    tag: &'static str,
    route: Route,
    functional: &'static str,
}

const CASES: [Case; 7] = [
    Case {
        tag: "rhf",
        route: Route::Rhf,
        functional: "",
    },
    Case {
        tag: "rks-pbe",
        route: Route::Rks,
        functional: "PBE",
    },
    Case {
        tag: "rks-pbe0",
        route: Route::Rks,
        functional: "PBE0",
    },
    Case {
        tag: "uhf",
        route: Route::Uhf,
        functional: "",
    },
    Case {
        tag: "uks-pbe",
        route: Route::Uks,
        functional: "PBE",
    },
    Case {
        tag: "roks-pbe",
        route: Route::Roks,
        functional: "PBE",
    },
    Case {
        tag: "rohf",
        route: Route::Rohf,
        functional: "",
    },
];

impl Case {
    fn open(self) -> bool {
        !matches!(self.route, Route::Rhf | Route::Rks)
    }
    fn atoms(self) -> &'static [[f64; 3]] {
        if self.open() {
            &H3
        } else {
            &H2
        }
    }
    fn multiplicity(self) -> usize {
        if self.open() {
            2
        } else {
            1
        }
    }
    fn ks(self) -> bool {
        !self.functional.is_empty()
    }
}

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

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferric-kstress-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn xyz(atoms_bohr: &[[f64; 3]]) -> String {
    let b = BOHR_RADIUS_ANGSTROM;
    let mut s = format!("{}\nk-point stress test (Angstrom)\n", atoms_bohr.len());
    for p in atoms_bohr {
        s += &format!("H {:.12} {:.12} {:.12}\n", p[0] * b, p[1] * b, p[2] * b);
    }
    s
}

/// Write the input; `cell_extra` / `method_extra` are appended verbatim.
fn write_input(c: Case, tag: &str, cell_extra: &str, method_extra: &str, kind: &str) -> PathBuf {
    let dir = scratch(tag);
    let xyz_path = dir.join("m.xyz");
    std::fs::write(&xyz_path, xyz(c.atoms())).unwrap();
    let dft = if c.ks() {
        format!("[dft]\nfunctional = \"{}\"", c.functional)
    } else {
        String::new()
    };
    let grid = if c.ks() {
        format!("n_radial = {N_RADIAL}\nn_angular = {N_ANGULAR}\nneighbour_cutoff = {CUTOFF_BOHR}")
    } else {
        String::new()
    };
    let toml = dir.join("in.toml");
    std::fs::write(
        &toml,
        format!(
            r#"[molecule]
xyz = "{xyz}"
multiplicity = {mult}
[basis]
name = "sto-3g"
[method]
kind = "{kind}"
{method_extra}
{dft}
[output]
json = "{json}"
[cell]
unit = "bohr"
lattice = [[{a}, 0.0, 0.0], [0.0, {a}, 0.0], [0.0, 0.0, {a}]]
exxdiv = "ewald"
omega = {OMEGA}
{grid}
{cell_extra}
"#,
            xyz = xyz_path.display(),
            json = dir.join("run.jsonl").display(),
            mult = c.multiplicity(),
            a = A_BOHR,
        ),
    )
    .unwrap();
    toml
}

fn kind_of(c: Case) -> &'static str {
    match c.route {
        Route::Rhf | Route::Rks => "rhf",
        Route::Uhf | Route::Uks => "uhf",
        Route::Rohf | Route::Roks => "rohf",
    }
}

fn run(toml: &Path) -> Output {
    Command::new(ferric_cli_bin())
        .arg(toml)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("run ferric-cli")
}

/// The `run_end` record's `extra` (strings and flags only: serde_json's
/// default f64 parse is not correctly rounded, so tensors go through
/// [`mat`], which reads the raw text with `str::parse::<f64>`).
fn run_end_line(toml: &Path) -> String {
    let log = std::fs::read_to_string(toml.with_file_name("run.jsonl")).expect("run log");
    for line in log.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if v.get("record").and_then(|r| r.as_str()) == Some("run_end") {
            return line.to_string();
        }
    }
    panic!("no run_end record in:\n{log}");
}

/// The 3x3 tensor under `"key":[[..],[..],[..]]` in a `run_end` line, parsed
/// from the raw text with the correctly rounded `str::parse::<f64>`.
fn mat(line: &str, key: &str) -> [[f64; 3]; 3] {
    let tag = format!("\"{key}\":[[");
    let start = line
        .find(&tag)
        .unwrap_or_else(|| panic!("no {key} in run_end: {line}"))
        + tag.len();
    let body = &line[start..];
    let body = &body[..body.find("]]").expect("closing ]]")];
    let rows: Vec<Vec<f64>> = body
        .split("],[")
        .map(|r| r.split(',').map(|x| x.parse().expect("f64")).collect())
        .collect();
    assert_eq!(rows.len(), 3, "{key}: {body}");
    let mut out = [[0.0; 3]; 3];
    for (o, r) in out.iter_mut().zip(&rows) {
        o.copy_from_slice(r);
    }
    out
}

fn scf() -> KScfConfig {
    KScfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        grad_conv: 1e-9,
        ..Default::default()
    }
}

fn grid() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(N_RADIAL, N_ANGULAR);
    g.neighbour_cutoff = Some(CUTOFF_BOHR);
    g
}

/// The stress settings of a run whose SCF retained `kept`: the retained hcore
/// and J/K are charged against the unified budget (the Python bindings' and the
/// CLI's choice; the budget sets the blocking, hence the last ulps).
fn sconf(kept: &ferric_pbc::KRetained) -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(kept.derivative_budget(None)),
        ..Default::default()
    }
}

fn xc_cfg(kept: &ferric_pbc::KRetained, ao: f64) -> KRksStressConfig {
    KRksStressConfig {
        stress: sconf(kept),
        grid: KXcGrid::Atom(grid()),
        ao_threshold: ao,
        exxdiv: Some(ExxDiv::Ewald),
        ..Default::default()
    }
}

/// Library stress of the case: the Python `with_stress` call sequence
/// (`solve_k*_keep`, then `kpoint_*_stress` through the retained hcore / J/K).
fn library(c: Case, centring: MeshCentring) -> [[f64; 3]; 3] {
    let dir = scratch("lib");
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(c.atoms())).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, c.multiplicity()).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).unwrap();
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let mesh = KPointMesh::new(&cell, [1, 1, 2], centring).unwrap();
    // As the CLI and the Python bindings: the dense oracle stays unrotated.
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA)
        .with_sr_column_rotation(ferric_pbc::SrColumnRotation::Off);
    let f = c.functional;
    let open = |kept: &ferric_pbc::KRetained, ao: f64| KUksStressConfig {
        base: xc_cfg(kept, ao),
        ..Default::default()
    };
    match c.route {
        Route::Rhf => {
            let mut k = KRhfConfig::for_cell(&cell, ExxDiv::Ewald);
            (k.scf, k.hcore) = (scf(), hcfg);
            let (r, kept) = solve_krhf_keep(&cell, &prep, None, &mesh, &k).unwrap();
            let jk = kept.stress_jk(None, &k.rsgdf).unwrap();
            kpoint_rhf_stress(
                &cell,
                &prep,
                &mesh,
                &k.hcore,
                &kept.hk,
                jk,
                &r,
                ExxDiv::Ewald,
                &sconf(&kept),
            )
            .unwrap()
            .sigma
        }
        Route::Uhf => {
            let mut k = KUhfConfig::for_cell(&cell, ExxDiv::Ewald);
            (k.scf, k.hcore) = (scf(), hcfg);
            let (r, kept) = solve_kuhf_keep(&cell, &prep, None, &mesh, &k).unwrap();
            let jk = kept.stress_jk(None, &k.rsgdf).unwrap();
            kpoint_uhf_stress(
                &cell,
                &prep,
                &mesh,
                &k.hcore,
                &kept.hk,
                jk,
                &r.scf,
                ExxDiv::Ewald,
                &sconf(&kept),
            )
            .unwrap()
            .sigma
        }
        Route::Rks => {
            let mut k = KRksConfig::new(&cell, ExxDiv::Ewald, f);
            (k.krhf.scf, k.krhf.hcore, k.grid) = (scf(), hcfg, grid());
            let (r, kept) = solve_krks_keep(&cell, &prep, None, &mesh, &k).unwrap();
            let jk = kept.stress_jk(None, &k.krhf.rsgdf).unwrap();
            let x = xc_cfg(&kept, k.xc.ao_threshold);
            kpoint_rks_stress(
                &cell,
                &prep,
                &mesh,
                &k.krhf.hcore,
                &kept.hk,
                jk,
                &r.scf,
                f,
                &x,
            )
            .unwrap()
            .sigma
        }
        Route::Uks => {
            let mut k = KUksConfig::new(&cell, ExxDiv::Ewald, f);
            (k.kuhf.scf, k.kuhf.hcore, k.grid) = (scf(), hcfg, grid());
            let (r, kept) = solve_kuks_keep(&cell, &prep, None, &mesh, &k).unwrap();
            let jk = kept.stress_jk(None, &k.kuhf.rsgdf).unwrap();
            let x = open(&kept, k.xc.ao_threshold);
            kpoint_uks_stress(
                &cell,
                &prep,
                &mesh,
                &k.kuhf.hcore,
                &kept.hk,
                jk,
                &r.scf,
                f,
                &x,
            )
            .unwrap()
            .sigma
        }
        Route::Roks => {
            let mut k = KRoksConfig::new(&cell, ExxDiv::Ewald, f);
            (k.rohf.scf, k.rohf.hcore, k.grid) = (scf(), hcfg, grid());
            let (r, kept) = solve_kroks_keep(&cell, &prep, None, &mesh, &k).unwrap();
            let jk = kept.stress_jk(None, &k.rohf.rsgdf).unwrap();
            let x = open(&kept, k.xc.ao_threshold);
            kpoint_roks_stress(
                &cell,
                &prep,
                &mesh,
                &k.rohf.hcore,
                &kept.hk,
                jk,
                &r.scf,
                f,
                &x,
            )
            .unwrap()
            .sigma
        }
        Route::Rohf => {
            let mut k = KRohfConfig::for_cell(&cell, ExxDiv::Ewald);
            (k.scf, k.hcore) = (scf(), hcfg);
            let (r, kept) = solve_krohf_keep(&cell, &prep, None, &mesh, &k).unwrap();
            let jk = kept.stress_jk(None, &k.rsgdf).unwrap();
            let x = KUksStressConfig {
                base: KRksStressConfig {
                    stress: sconf(&kept),
                    ..Default::default()
                },
                ..Default::default()
            };
            kpoint_rohf_stress(
                &cell,
                &prep,
                &mesh,
                &k.hcore,
                &kept.hk,
                jk,
                &r.scf,
                ExxDiv::Ewald,
                &x,
            )
            .unwrap()
            .sigma
        }
    }
}

fn bits(m: &[[f64; 3]; 3]) -> Vec<u64> {
    m.iter().flatten().map(|v| v.to_bits()).collect()
}

fn check_case(c: Case, centring_key: &str, centring: MeshCentring) {
    let tag = format!("{}-{centring_key}", c.tag);
    let toml = write_input(
        c,
        &tag,
        &format!("kmesh = [1, 1, 2]\ncentring = \"{centring_key}\"\nstress = true"),
        "",
        kind_of(c),
    );
    let out = run(&toml);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{tag}:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let line = run_end_line(&toml);
    let cli = mat(&line, "stress_hartree_per_bohr3");
    let lib = library(c, centring);
    assert_eq!(
        bits(&cli),
        bits(&lib),
        "{tag}: CLI stress {cli:?} != library {lib:?}"
    );
    // The GPa tensor is the same tensor times the units constant, bit for bit.
    let gpa = mat(&line, "stress_gpa");
    let want = lib.map(|r| r.map(|v| v * HARTREE_PER_BOHR3_TO_GPA));
    assert_eq!(bits(&gpa), bits(&want), "{tag}: GPa tensor");
    assert!(
        line.contains("\"stress_unit\":\"hartree_per_bohr3\""),
        "{line}"
    );
    // The printout carries both blocks; the stress is not the zero tensor and
    // not isotropic (a reversed row order or a transposed tensor would show).
    assert!(stdout.contains("stress (GPa):"), "{tag}:\n{stdout}");
    assert!(
        lib.iter().flatten().any(|v| v.abs() > 1e-6),
        "{tag}: stress is zero: {lib:?}"
    );
    assert!(
        (lib[0][0] - lib[2][2]).abs() > 1e-7,
        "{tag}: sigma_xx == sigma_zz, row order would be untestable: {lib:?}"
    );
    // Mutant checks: the anchor distinguishes a reversed tensor and a wrong
    // unit factor.
    let reversed = [lib[2], lib[1], lib[0]];
    assert_ne!(bits(&cli), bits(&reversed), "{tag}: reversed rows");
    assert_ne!(
        bits(&gpa),
        bits(&lib.map(|r| r.map(|v| v * 29421.0))),
        "{tag}: rounded GPa factor"
    );
}

/// One test per (route, centring): each is a CLI run plus a library run, so
/// every test stays far below the 120 s budget and they run in parallel.
macro_rules! anchor {
    ($name:ident, $case:expr, $key:literal, $centring:expr) => {
        #[test]
        fn $name() {
            check_case(CASES[$case], $key, $centring);
        }
    };
}

anchor!(rhf_gamma_centred, 0, "gamma", MeshCentring::Gamma);
anchor!(rhf_shifted, 0, "mp", MeshCentring::MonkhorstPack);
anchor!(rks_pbe_gamma_centred, 1, "gamma", MeshCentring::Gamma);
anchor!(rks_pbe_shifted, 1, "mp", MeshCentring::MonkhorstPack);
anchor!(rks_pbe0_gamma_centred, 2, "gamma", MeshCentring::Gamma);
anchor!(rks_pbe0_shifted, 2, "mp", MeshCentring::MonkhorstPack);
anchor!(uhf_gamma_centred, 3, "gamma", MeshCentring::Gamma);
anchor!(uhf_shifted, 3, "mp", MeshCentring::MonkhorstPack);
anchor!(uks_pbe_gamma_centred, 4, "gamma", MeshCentring::Gamma);
anchor!(uks_pbe_shifted, 4, "mp", MeshCentring::MonkhorstPack);
anchor!(roks_pbe_gamma_centred, 5, "gamma", MeshCentring::Gamma);
anchor!(roks_pbe_shifted, 5, "mp", MeshCentring::MonkhorstPack);
anchor!(rohf_gamma_centred, 6, "gamma", MeshCentring::Gamma);
anchor!(rohf_shifted, 6, "mp", MeshCentring::MonkhorstPack);

/// The shifted mesh is a different mesh: its stress is not the Gamma-centred
/// one (so the `mp` runs above are not silently Gamma-centred).
#[test]
fn shifted_mesh_stress_differs_from_gamma_centred() {
    let c = CASES[3];
    let g = library(c, MeshCentring::Gamma);
    let m = library(c, MeshCentring::MonkhorstPack);
    assert_ne!(bits(&g), bits(&m));
    assert!((g[2][2] - m[2][2]).abs() > 1e-8, "{g:?} vs {m:?}");
}

fn refused(c: Case, tag: &str, kind: &str, cell: &str, method: &str) -> String {
    let out = run(&write_input(c, tag, cell, method, kind));
    assert!(!out.status.success(), "{tag} must be refused");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn stress_refusals_name_the_toml_key() {
    let km = "kmesh = [1, 1, 2]\nstress = true";
    let hse = Case {
        tag: "h",
        route: Route::Rks,
        functional: "HSE06",
    };
    let e = refused(hse, "hse", "rhf", km, "");
    assert!(
        e.contains("[cell] stress = true") && e.contains("range-separated"),
        "{e}"
    );
    let scan_open = Case {
        tag: "s",
        route: Route::Uks,
        functional: "SCAN",
    };
    let e = refused(scan_open, "scan-uks", "uhf", km, "");
    assert!(e.contains("meta-GGA"), "{e}");
    let rhf = CASES[0];
    let e = refused(rhf, "gamma", "rhf", "stress = true", "");
    assert!(
        e.contains("[cell] stress = true") && e.contains("kmesh"),
        "{e}"
    );
    let e = refused(rhf, "opt", "rhf", km, "task = \"optimize\"");
    assert!(
        e.contains("[cell] stress = true") && e.contains("optimize"),
        "{e}"
    );
    let e = refused(
        rhf,
        "mp2",
        "rimp2",
        &format!("{km}\ndenominators = \"shifted\""),
        "",
    );
    assert!(
        e.contains("[cell] stress = true") && e.contains("no analytic stress"),
        "{e}"
    );
}
