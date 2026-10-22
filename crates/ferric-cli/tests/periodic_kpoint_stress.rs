//! `[cell] kmesh` + `stress = true`: the CLI prints the analytic k-point stress
//! and writes it to the run log's `run_end` record (issue #357).
//!
//! * EXACTNESS ANCHOR: the 3x3 tensor read from `run_end.extra` (full f64
//!   precision, not the printed digits) equals the library
//!   `ferric_pbc::kpoint_*_stress` of the SAME cell, mesh, grid and configs
//!   BIT FOR BIT, for RHF, UHF, ROHF, RKS (PBE, PBE0), UKS and ROKS on a
//!   Gamma-centred AND a shifted (Monkhorst-Pack, `centring = "mp"`) 1x1x2
//!   mesh (H2 / H3 doublet, STO-3G, dense J/K, small XC grid), the
//!   range-separated RKS-HSE06 and the open-shell UKS-SCAN / r2SCAN stress
//!   (dense J/K; issue #498), plus an RKS-PBE
//!   run with EVERYTHING at its default (omega, XC grid 75x302, cutoff) and a
//!   ROKS-PBE0 run on `jk = "rsgdf"` (cc-pVDZ-RI).
//! * Python cross-check (done by hand against the extension built from this
//!   branch, `run_{rhf,rks,uhf,uks,roks,rohf}_kpts(..., with_stress=True)`
//!   fed the same cell in Angstrom): the CLI tensor equals the Python one bit
//!   for bit in all 14 (route x centring) cases above. The coordinator's
//!   review repeated it for the RS-GDF and all-defaults runs.
//! * The GPa tensor is `sigma * ferric_core::units::HARTREE_PER_BOHR3_TO_GPA`
//!   bit for bit.
//! * Refusals are TOML-level errors: the stress gate names `[cell] stress =
//!   true` (closed-shell HSE06 on RS-GDF, open-shell HSE06, Gamma point,
//!   optimize, correlated route, split RS-GDF, open-shell SCAN on RS-GDF);
//!   hybrid meta-GGA (and ROKS meta-GGA) are refused earlier by the
//!   k-point ENERGY gate (`[cell] kmesh with functional ...`), which does not
//!   name the key.
//!   An ECP cell (HI / def2-SVP, RKS-PBE and ROHF) passes the gate and
//!   reaches the SCF (the ECP derivatives are validated in the library,
//!   `ferric-pbc/tests/validation_kks_ecp_derivs.rs`).
//! * Mutants run against `rhf_shifted` and the others (all KILLED): rows
//!   reversed / tensor transposed, GPa factor 29421.0, sign flip, default XC
//!   grid in place of the configured one, `exxdiv = none` in the stress call,
//!   stress dropped from the JSON. SURVIVED (equivalent for these fixtures): the stress
//!   `budget_bytes` wiring (it only sets blocking; no fixture is big enough
//!   to block differently).

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

/// What the run leaves at the CLI / library defaults.
#[derive(Clone, Copy)]
struct V {
    /// Small XC grid (20x50, 14 Bohr cutoff) instead of the defaults (75x302).
    small_grid: bool,
    /// Leave `omega` unset (the cell-derived hcore split).
    default_omega: bool,
    /// `jk = "rsgdf"` with cc-pVDZ-RI (default gdf_omega, rotation on).
    rsgdf: bool,
}

const SMALL: V = V {
    small_grid: true,
    default_omega: false,
    rsgdf: false,
};

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

const CASES: [Case; 10] = [
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
    // Issue #498: the closed-shell range-separated hybrid stress (dense-AFT
    // J/K) and the open-shell UKS meta-GGA stress (dense-AFT J/K).
    Case {
        tag: "rks-hse06",
        route: Route::Rks,
        functional: "HSE06",
    },
    Case {
        tag: "uks-scan",
        route: Route::Uks,
        functional: "SCAN",
    },
    Case {
        tag: "uks-r2scan",
        route: Route::Uks,
        functional: "r2SCAN",
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

fn scratch_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ferric-kstress-{}-{tag}", std::process::id()))
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferric-kstress-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
fn write_input(
    c: Case,
    v: V,
    tag: &str,
    cell_extra: &str,
    method_extra: &str,
    kind: &str,
) -> PathBuf {
    let dir = scratch(tag);
    let xyz_path = dir.join("m.xyz");
    std::fs::write(&xyz_path, xyz(c.atoms())).unwrap();
    let dft = if c.ks() {
        format!("[dft]\nfunctional = \"{}\"", c.functional)
    } else {
        String::new()
    };
    let grid = if c.ks() && v.small_grid {
        format!("n_radial = {N_RADIAL}\nn_angular = {N_ANGULAR}\nneighbour_cutoff = {CUTOFF_BOHR}")
    } else {
        String::new()
    };
    let omega = if v.default_omega {
        String::new()
    } else {
        format!("omega = {OMEGA}")
    };
    let jk = if v.rsgdf {
        "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\""
    } else {
        ""
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
{omega}
{jk}
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

fn grid(v: V) -> PeriodicGridConfig {
    if !v.small_grid {
        return PeriodicGridConfig::with_size(75, 302);
    }
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

fn xc_cfg(v: V, kept: &ferric_pbc::KRetained, ao: f64) -> KRksStressConfig {
    KRksStressConfig {
        stress: sconf(kept),
        grid: KXcGrid::Atom(grid(v)),
        ao_threshold: ao,
        exxdiv: Some(ExxDiv::Ewald),
        ..Default::default()
    }
}

/// Library stress of the case: the Python `with_stress` call sequence
/// (`solve_k*_keep`, then `kpoint_*_stress` through the retained hcore / J/K).
fn library(c: Case, v: V, centring: MeshCentring, tag: &str) -> [[f64; 3]; 3] {
    let dir = scratch(&format!("lib-{tag}"));
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(c.atoms())).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, c.multiplicity()).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).unwrap();
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let mesh = KPointMesh::new(&cell, [1, 1, 2], centring).unwrap();
    // As the CLI and the Python bindings: the dense oracle stays unrotated.
    let omega = if v.default_omega {
        PeriodicHcoreConfig::for_cell_and_gdf_omega(&cell, None).omega
    } else {
        OMEGA
    };
    let rot = if v.rsgdf {
        ferric_pbc::SrColumnRotation::on()
    } else {
        ferric_pbc::SrColumnRotation::Off
    };
    let hcfg = PeriodicHcoreConfig::with_omega(omega).with_sr_column_rotation(rot);
    let aux_prep = v
        .rsgdf
        .then(|| PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap());
    let aux = aux_prep.as_ref();
    let rs = ferric_pbc::KRsGdfConfig {
        gdf: ferric_pbc::RsGdfConfig {
            omega: ferric_pbc::rsgdf::DEFAULT_RSGDF_OMEGA,
            sr_column_rotation: rot,
            ..Default::default()
        },
        ..Default::default()
    };
    let f = c.functional;
    let open = |kept: &ferric_pbc::KRetained, ao: f64| KUksStressConfig {
        base: xc_cfg(v, kept, ao),
        ..Default::default()
    };
    match c.route {
        Route::Rhf => {
            let mut k = KRhfConfig::for_cell(&cell, ExxDiv::Ewald);
            assert!(!v.rsgdf);
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
            assert!(!v.rsgdf);
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
            assert!(!v.rsgdf, "RS-GDF is anchored on ROKS only");
            (k.krhf.scf, k.krhf.hcore, k.grid) = (scf(), hcfg, grid(v));
            let (r, kept) = solve_krks_keep(&cell, &prep, None, &mesh, &k).unwrap();
            let jk = kept.stress_jk(None, &k.krhf.rsgdf).unwrap();
            let x = xc_cfg(v, &kept, k.xc.ao_threshold);
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
            assert!(!v.rsgdf);
            let mut k = KUksConfig::new(&cell, ExxDiv::Ewald, f);
            (k.kuhf.scf, k.kuhf.hcore, k.grid) = (scf(), hcfg, grid(v));
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
            (k.rohf.scf, k.rohf.hcore, k.grid) = (scf(), hcfg, grid(v));
            if v.rsgdf {
                (k.rohf.jk, k.rohf.rsgdf) = (ferric_pbc::KJkKind::RsGdf, rs);
            }
            let (r, kept) = solve_kroks_keep(&cell, &prep, aux, &mesh, &k).unwrap();
            let jk = kept.stress_jk(aux, &k.rohf.rsgdf).unwrap();
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
            assert!(!v.rsgdf);
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

fn check_case(c: Case, v: V, centring_key: &str, centring: MeshCentring) {
    let tag = format!(
        "{}-{centring_key}-g{}o{}r{}",
        c.tag, v.small_grid as u8, v.default_omega as u8, v.rsgdf as u8
    );
    let toml = write_input(
        c,
        v,
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
    let lib = library(c, v, centring, &tag);
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
    let _ = std::fs::remove_dir_all(toml.parent().unwrap());
    let _ = std::fs::remove_dir_all(scratch_path(&format!("lib-{tag}")));
}

/// One test per (route, centring): each is a CLI run plus a library run, so
/// every test stays far below the 120 s budget and they run in parallel.
macro_rules! anchor {
    ($name:ident, $case:expr, $key:literal, $centring:expr) => {
        #[test]
        fn $name() {
            check_case(CASES[$case], SMALL, $key, $centring);
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
anchor!(rks_hse06_gamma_centred, 7, "gamma", MeshCentring::Gamma);
anchor!(rks_hse06_shifted, 7, "mp", MeshCentring::MonkhorstPack);
anchor!(uks_scan_gamma_centred, 8, "gamma", MeshCentring::Gamma);
anchor!(uks_r2scan_shifted, 9, "mp", MeshCentring::MonkhorstPack);

/// The issue #498 stress routes are not their global siblings: HSE06 vs PBE0
/// and SCAN vs PBE move the library tensor by far more than one ulp, so the
/// bit-equal anchors above cannot pass on a mis-routed run.
#[test]
fn rsh_and_open_shell_mgga_stress_differ_from_their_siblings() {
    let pairs = [(7, 2), (8, 4)];
    for (a, b) in pairs {
        let x = library(CASES[a], SMALL, MeshCentring::Gamma, &format!("sib-{a}"));
        let y = library(CASES[b], SMALL, MeshCentring::Gamma, &format!("sib-{b}"));
        let d = x
            .iter()
            .flatten()
            .zip(y.iter().flatten())
            .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
        eprintln!(
            "{} vs {}: max |d sigma| {d:.2e}",
            CASES[a].tag, CASES[b].tag
        );
        assert!(d > 1e-7, "{} vs {}: {d:e}", CASES[a].tag, CASES[b].tag);
    }
}

/// The shifted mesh is a different mesh: its stress is not the Gamma-centred
/// one (so the `mp` runs above are not silently Gamma-centred).
#[test]
fn shifted_mesh_stress_differs_from_gamma_centred() {
    let c = CASES[3];
    let g = library(c, SMALL, MeshCentring::Gamma, "diff-g");
    let m = library(c, SMALL, MeshCentring::MonkhorstPack, "diff-m");
    assert_ne!(bits(&g), bits(&m));
    assert!((g[2][2] - m[2][2]).abs() > 1e-8, "{g:?} vs {m:?}");
}

fn refused(c: Case, tag: &str, kind: &str, cell: &str, method: &str) -> String {
    let out = run(&write_input(c, SMALL, tag, cell, method, kind));
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
    // The dense-AFT HSE06 stress runs (anchored above); on RS-GDF and open
    // shell it is refused by the stress gate, which names the key.
    let rs = format!("{km}\njk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"");
    let e = refused(hse, "hse-rsgdf", "rhf", &rs, "");
    assert!(
        e.contains("[cell] stress = true") && e.contains("not wired to the bindings yet"),
        "{e}"
    );
    let hse_open = Case {
        tag: "ho",
        route: Route::Uks,
        functional: "HSE06",
    };
    let e = refused(hse_open, "hse-uks", "uhf", km, "");
    assert!(
        e.contains("[cell] stress = true")
            && e.contains("UKS range-separated hybrid force and stress are not wired"),
        "{e}"
    );
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

const ROKS_PBE0: Case = Case {
    tag: "roks-pbe0",
    route: Route::Roks,
    functional: "PBE0",
};

/// Everything at its default: omega, XC grid (75x302), neighbour cutoff.
#[test]
fn rks_pbe_all_defaults_equals_library() {
    let v = V {
        small_grid: false,
        default_omega: true,
        rsgdf: false,
    };
    check_case(CASES[1], v, "gamma", MeshCentring::Gamma);
}

/// Open-shell hybrid stress on the unsplit RS-GDF J/K with default omegas.
#[test]
fn roks_pbe0_rsgdf_equals_library() {
    let v = V {
        small_grid: true,
        default_omega: true,
        rsgdf: true,
    };
    check_case(ROKS_PBE0, v, "mp", MeshCentring::MonkhorstPack);
}

/// An ECP cell (HI / def2-SVP, the def2 ECP on I) passes the stress
/// gate for Kohn-Sham and for ROHF and reaches the SCF (validated in the
/// library: `crates/ferric-pbc/tests/validation_kks_ecp_derivs.rs`). The SCF is
/// capped at one iteration so the test stays cheap: the run must end
/// `not_converged` with the stress requested, not with a refusal. RS-GDF: the
/// dense AFT of iodine's tight primitives exceeds the G-vector cap.
#[test]
fn ecp_cells_pass_the_stress_gate_and_reach_the_scf() {
    for (tag, kind, dft, mult) in [
        ("ecp-rks", "rhf", "[dft]\nfunctional = \"PBE\"", 1),
        ("ecp-rohf", "rohf", "", 3),
    ] {
        let dir = scratch(tag);
        std::fs::write(dir.join("hi.xyz"), "2\nHI\nH 0 0 0\nI 0 0 1.61\n").unwrap();
        let grid = if dft.is_empty() {
            String::new()
        } else {
            format!("n_radial = {N_RADIAL}\nn_angular = {N_ANGULAR}\n")
        };
        let toml = dir.join("in.toml");
        std::fs::write(
            &toml,
            format!(
                "[molecule]\nxyz = \"{}\"\nmultiplicity = {mult}\n[basis]\nname = \"def2-svp\"\n\
                 [method]\nkind = \"{kind}\"\n{dft}\n[output]\njson = \"{}\"\n[cell]\n\
                 unit = \"bohr\"\nlattice = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]]\n\
                 kmesh = [1, 1, 2]\nstress = true\n{grid}\
                 jk = \"rsgdf\"\nauxbasis = \"def2-svp-rifit\"\n\
                 [scf]\nmax_iter = 1\n",
                dir.join("hi.xyz").display(),
                dir.join("run.jsonl").display(),
            ),
        )
        .unwrap();
        let out = run(&toml);
        let e = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{tag}: {e}");
        assert!(!e.contains("not validated"), "{tag}: {e}");
        let line = run_end_line(&toml);
        assert!(line.contains("\"exit\":\"not_converged\""), "{tag}: {line}");
        assert!(line.contains("\"stress_requested\":true"), "{tag}: {line}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A non-converged SCF (RHF reports it as a result, the Kohn-Sham and
/// open-shell retained solvers as an error) closes the run log with
/// `exit = "not_converged"` and carries no stress tensor; the process exits 0.
#[test]
fn unconverged_scf_logs_not_converged_without_a_tensor() {
    for c in [CASES[0], CASES[1], CASES[4], CASES[5]] {
        let tag = format!("unconv-{}", c.tag);
        let toml = write_input(
            c,
            SMALL,
            &tag,
            "kmesh = [1, 1, 2]\nstress = true\n[scf]\nmax_iter = 1",
            "",
            kind_of(c),
        );
        let out = run(&toml);
        let line = run_end_line(&toml);
        let msg = format!(
            "{}: {line}\n{}",
            c.tag,
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success(), "{msg}");
        assert!(line.contains("\"converged\":false"), "{msg}");
        assert!(line.contains("\"exit\":\"not_converged\""), "{msg}");
        assert!(line.contains("\"stress_requested\":true"), "{msg}");
        assert!(!line.contains("stress_hartree_per_bohr3"), "{msg}");
        if c.route == Route::Rhf {
            assert!(line.contains("\"scf_iterations\":1"), "{msg}");
        }
        let _ = std::fs::remove_dir_all(toml.parent().unwrap());
    }
}
