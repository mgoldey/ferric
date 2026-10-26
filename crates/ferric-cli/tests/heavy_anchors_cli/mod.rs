//! Shared harness of `periodic_heavy_anchors.rs` (per-commit) and
//! `validation_periodic_heavy_anchors.rs` (validation tier): the issue #498
//! CLI-vs-library anchors of `periodic_rsh_open_mgga_derivs.rs` /
//! `periodic_kpoint_stress.rs`, generalised from H/STO-3G to cells whose
//! basis carries p and d shells and to cells with a heavy (Li) atom.
//!
//! One anchor = one CLI run (`task = "optimize"` with `max_steps = 1`, so the
//! returned geometry is a MOVED one, or `task = "energy"` with `[cell] stress
//! = true`) whose `run_end` numbers must equal the library call at the same
//! cell, mesh, grid and configs BIT FOR BIT. The library side restates the
//! CLI's config builders field by field, exactly as the two H/STO-3G files do.
//!
//! Basis guard: every anchor also asserts that the CLI and the library both
//! saw the intended number of basis functions (`nbasis = ...` in the CLI
//! printout, `PreparedBasis::nbasis()` on the library side), so a run that
//! silently fell back to a bundled s-only basis cannot pass.
#![allow(dead_code)]

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::units::BOHR_RADIUS_ANGSTROM;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::DEFAULT_DENSE_AFT_PRECISION;
use ferric_pbc::gamma_rsh::{
    gamma_rks_any_keep, gamma_rks_rsh_gradient, gamma_uks_any_keep, gamma_uks_rsh_gradient,
    GammaRshSizing,
};
use ferric_pbc::kstress::KStressConfig;
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KXcGrid};
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStressConfig};
use ferric_pbc::rsgdf::DEFAULT_RSGDF_OMEGA;
use ferric_pbc::{
    kpoint_uks_gradient, kpoint_uks_stress, periodic_hcore, solve_krks_keep, solve_kuks_keep, Cell,
    DenseAftEri, ExxDiv, GammaRksConfig, GammaUhfIntegrals, GammaUksConfig, KDenseAftConfig,
    KGradConfig, KPointMesh, KRksConfig, KScfConfig, KUksConfig, KUksGradConfig, KUksStressConfig,
    MeshCentring, PeriodicGridConfig, PeriodicHcoreConfig, SrColumnRotation,
};
use ferric_scf::rhf::RhfConfig;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub const OMEGA: f64 = 0.8;
pub const CUTOFF_BOHR: f64 = 14.0;
pub const N_RADIAL: usize = 20;
pub const N_ANGULAR: usize = 50;
pub const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";
/// One BFGS step: two evaluations, the returned geometry is a moved one.
const OPT: &str = "max_steps = 1\ng_max_thresh = 1e-9\ng_rms_thresh = 1e-9\ne_conv = 1e-14";

/// H STO-3G (the bundled digits) plus one p shell (zeta 1.0): the cheapest
/// basis with l = 1 that keeps the dense-AFT G sphere of the H/STO-3G cells
/// (it is set by the tightest exponent, 3.425).
pub const H_SP: &str = "****
H     0
S   3   1.00
      0.3425250914E+01       0.1543289673E+00
      0.6239137298E+00       0.5353281423E+00
      0.1688554040E+00       0.4446345422E+00
P   1   1.00
      1.0000000000E+00       1.0000000000E+00
****
";
/// [`H_SP`] plus one pure d shell (zeta 1.0; G94 `D` is spherical).
pub const H_SPD: &str = "****
H     0
S   3   1.00
      0.3425250914E+01       0.1543289673E+00
      0.6239137298E+00       0.5353281423E+00
      0.1688554040E+00       0.4446345422E+00
P   1   1.00
      1.0000000000E+00       1.0000000000E+00
D   1   1.00
      1.0000000000E+00       1.0000000000E+00
****
";

#[derive(Clone, Copy)]
pub enum Basis {
    /// A G94 text, written next to the input and read by `[basis] path`.
    G94(&'static str),
    /// A bundled set (`[basis] name`).
    Bundled(&'static str),
}

/// One test cell: species and Bohr positions, multiplicity, cubic edge, basis
/// and the number of basis functions the cell must carry.
#[derive(Clone, Copy)]
pub struct System {
    pub symbols: &'static [&'static str],
    pub atoms: &'static [[f64; 3]],
    pub mult: usize,
    pub a: f64,
    pub basis: Basis,
    pub nbasis: usize,
}

/// H2 (the H/STO-3G anchors' geometry) with s + p on each H: 8 functions.
pub const H2_SP: System = System {
    symbols: &["H", "H"],
    atoms: &[[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]],
    mult: 1,
    a: 4.0,
    basis: Basis::G94(H_SP),
    nbasis: 8,
};
/// H3 doublet (the H/STO-3G anchors' geometry) with s + p: 12 functions.
pub const H3_SP: System = System {
    symbols: &["H", "H", "H"],
    atoms: &[[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]],
    mult: 2,
    a: 4.0,
    basis: Basis::G94(H_SP),
    nbasis: 12,
};
/// H2 with s + p + d (pure): 18 functions.
pub const H2_SPD: System = System {
    basis: Basis::G94(H_SPD),
    nbasis: 18,
    ..H2_SP
};
/// H3 doublet with s + p + d: 27 functions.
pub const H3_SPD: System = System {
    basis: Basis::G94(H_SPD),
    nbasis: 27,
    ..H3_SP
};
/// LiH, bundled STO-3G (Li 1s + 2sp: real p shells on a Z = 3 nucleus), bond
/// 3.05 Bohr off every axis.
pub const LIH: System = System {
    symbols: &["Li", "H"],
    atoms: &[[0.3, 0.2, 0.1], [1.9, 1.4, 2.4]],
    mult: 1,
    a: 4.0,
    basis: Basis::Bundled("sto-3g"),
    nbasis: 6,
};
/// LiH2 doublet, bundled STO-3G.
pub const LIH2: System = System {
    symbols: &["Li", "H", "H"],
    atoms: &[[0.3, 0.2, 0.1], [1.9, 1.4, 2.4], [0.5, 2.9, 1.2]],
    mult: 2,
    a: 4.0,
    basis: Basis::Bundled("sto-3g"),
    nbasis: 7,
};

pub fn ferric_cli_bin() -> PathBuf {
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
    let dir = std::env::temp_dir().join(format!("ferric-heavy498-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn xyz(s: &System) -> String {
    let b = BOHR_RADIUS_ANGSTROM;
    let mut out = format!("{}\nissue 498 heavy anchors (Angstrom)\n", s.atoms.len());
    for (sym, p) in s.symbols.iter().zip(s.atoms) {
        out += &format!("{sym} {:.12} {:.12} {:.12}\n", p[0] * b, p[1] * b, p[2] * b);
    }
    out
}

/// The `[basis]` lines of `s` (a G94 basis is written into `dir`).
fn basis_toml(s: &System, dir: &Path) -> String {
    match s.basis {
        Basis::Bundled(n) => format!("name = \"{n}\""),
        Basis::G94(text) => {
            let p = dir.join("basis.g94");
            std::fs::write(&p, text).unwrap();
            format!("path = \"{}\"", p.display())
        }
    }
}

fn load_basis(s: &System, dir: &Path) -> BasisSet {
    match s.basis {
        Basis::Bundled(n) => basis::bundled(n).unwrap(),
        Basis::G94(text) => {
            let p = dir.join("lib-basis.g94");
            std::fs::write(&p, text).unwrap();
            basis::load_g94(p.to_str().unwrap()).unwrap()
        }
    }
}

/// One CLI input. `kind`: "rhf" (closed shell) or "uhf" (UKS); `kmesh` empty
/// = Gamma point; `task` "optimize" or "energy" (then `stress = true`).
pub struct Input<'a> {
    pub tag: &'a str,
    pub sys: System,
    pub kind: &'a str,
    pub functional: &'a str,
    pub task: &'a str,
    pub kmesh: &'a str,
}

fn write(i: &Input) -> PathBuf {
    let dir = scratch(i.tag);
    let xyz_path = dir.join("m.xyz");
    std::fs::write(&xyz_path, xyz(&i.sys)).unwrap();
    let (optimize, stress) = if i.task == "optimize" {
        (format!("[optimize]\n{OPT}"), "")
    } else {
        (String::new(), "stress = true")
    };
    let kmesh = if i.kmesh.is_empty() {
        String::new()
    } else {
        format!("kmesh = {}", i.kmesh)
    };
    let toml = dir.join("in.toml");
    std::fs::write(
        &toml,
        format!(
            r#"[molecule]
xyz = "{xyz}"
multiplicity = {mult}
[basis]
{basis}
[method]
kind = "{kind}"
task = "{task}"
[dft]
functional = "{f}"
[output]
json = "{json}"
[cell]
unit = "bohr"
lattice = [[{a}, 0.0, 0.0], [0.0, {a}, 0.0], [0.0, 0.0, {a}]]
{kmesh}
exxdiv = "ewald"
omega = {OMEGA}
n_radial = {N_RADIAL}
n_angular = {N_ANGULAR}
neighbour_cutoff = {CUTOFF_BOHR}
{stress}
{optimize}
"#,
            xyz = xyz_path.display(),
            mult = i.sys.mult,
            basis = basis_toml(&i.sys, &dir),
            kind = i.kind,
            task = i.task,
            f = i.functional,
            json = dir.join("run.jsonl").display(),
            a = i.sys.a,
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

/// Run the input; return the `run_end` record. Also checks the basis guard
/// (the printed `nbasis`).
fn cli(i: &Input) -> serde_json::Value {
    let toml = write(i);
    let o = run(&toml);
    let stdout = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(
        o.status.success(),
        "{}: ferric-cli failed\n{stdout}\n{}",
        i.tag,
        String::from_utf8_lossy(&o.stderr)
    );
    let want = format!("nbasis     = {}\n", i.sys.nbasis);
    assert!(
        stdout.contains(&want),
        "{}: no `{want}` in\n{stdout}",
        i.tag
    );
    let log = std::fs::read_to_string(toml.with_file_name("run.jsonl")).expect("run log");
    let end = log
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .find(|v| v.get("record").and_then(|r| r.as_str()) == Some("run_end"))
        .unwrap_or_else(|| panic!("{}: no run_end record in:\n{log}", i.tag));
    let _ = std::fs::remove_dir_all(toml.parent().unwrap());
    end
}

fn floats(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| x.as_f64().expect("f64"))
        .collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The cell the CLI built: the same xyz round trip, then (for an optimize
/// run) the logged Bohr positions written into the parsed molecule exactly as
/// the CLI's optimizer does. Asserts the basis guard.
fn cell_at(s: &System, at: Option<&[f64]>) -> (Cell, PreparedBasis) {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = scratch(&format!("lib{n}"));
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(s)).unwrap();
    let mut mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, s.mult).expect("xyz");
    if let Some(x) = at {
        for (k, a) in mol.atoms.iter_mut().enumerate() {
            (a.x, a.y, a.zpos) = (x[3 * k], x[3 * k + 1], x[3 * k + 2]);
        }
    }
    let lat = [[s.a, 0.0, 0.0], [0.0, s.a, 0.0], [0.0, 0.0, s.a]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &load_basis(s, &dir)).unwrap();
    assert_eq!(prep.nbasis(), s.nbasis, "library basis guard");
    let _ = std::fs::remove_dir_all(&dir);
    (cell, prep)
}

fn grid() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(N_RADIAL, N_ANGULAR);
    g.neighbour_cutoff = Some(CUTOFF_BOHR);
    g
}

/// The CLI's dense-AFT cap (`[cell] max_eri_gb` default 0.5).
fn cli_max_eri_bytes() -> usize {
    ferric_core::memory::gib_to_bytes(0.5)
}

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(SrColumnRotation::Off)
}

/// `crate::periodic::kscf_config` with the periodic defaults.
fn kscf() -> KScfConfig {
    KScfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        grad_conv: 1e-9,
        ..Default::default()
    }
}

fn kdense() -> KDenseAftConfig {
    KDenseAftConfig {
        max_bytes: cli_max_eri_bytes(),
        ..Default::default()
    }
}

fn sizing() -> GammaRshSizing {
    GammaRshSizing {
        rsgdf_omega: DEFAULT_RSGDF_OMEGA,
        budget_bytes: None,
        max_eri_bytes: cli_max_eri_bytes(),
        range_split: false,
    }
}

fn mesh(cell: &Cell) -> KPointMesh {
    KPointMesh::new(cell, [1, 1, 2], MeshCentring::Gamma).unwrap()
}

/// The Gamma hcore and dense J/K of `crate::periodic::gamma_system`.
fn gamma_ints(cell: &Cell, prep: &PreparedBasis) -> (ferric_pbc::PeriodicHcore, DenseAftEri) {
    let hc = periodic_hcore(cell, prep, &hcore_cfg()).unwrap();
    let eri = DenseAftEri::build(
        cell,
        prep,
        &hc.s,
        ExxDiv::Ewald,
        DEFAULT_DENSE_AFT_PRECISION,
        cli_max_eri_bytes(),
    )
    .unwrap();
    (hc, eri)
}

fn krks(cell: &Cell, f: &str) -> KRksConfig {
    let mut c = KRksConfig::new(cell, ExxDiv::Ewald, f);
    (c.krhf.scf, c.krhf.hcore, c.krhf.dense, c.grid) = (kscf(), hcore_cfg(), kdense(), grid());
    c
}

fn kuks(cell: &Cell, f: &str) -> KUksConfig {
    let mut c = KUksConfig::new(cell, ExxDiv::Ewald, f);
    (c.kuhf.scf, c.kuhf.hcore, c.kuhf.dense, c.grid) = (kscf(), hcore_cfg(), kdense(), grid());
    c
}

/// Library k-point RKS `(E, dE/dR)` at `x` (`crate::periodic::krks_config`
/// and `KForceCtx::xc_cfg`) on the dense-AFT J/K.
pub fn lib_krks_force(s: &System, f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    let (cell, prep) = cell_at(s, Some(x));
    let mesh = mesh(&cell);
    let c = krks(&cell, f);
    let (r, kept) = solve_krks_keep(&cell, &prep, None, &mesh, &c).expect("scf");
    let gc = KRksGradConfig {
        grad: KGradConfig {
            budget_bytes: Some(kept.derivative_budget(None)),
            ..Default::default()
        },
        grid: KXcGrid::Atom(grid()),
        ao_threshold: c.xc.ao_threshold,
        exxdiv: Some(ExxDiv::Ewald),
        ..Default::default()
    };
    let jk = kept.grad_jk(None, &c.krhf.rsgdf).unwrap();
    let g = kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &c.krhf.hcore,
        &kept.hk,
        jk,
        &r.scf,
        f,
        &gc,
    )
    .expect("force");
    (r.scf.energy, g.grad.iter().copied().collect())
}

/// Library k-point UKS `(E, dE/dR)` at `x` (`crate::periodic::kuks_config`).
pub fn lib_kuks_force(s: &System, f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    let (cell, prep) = cell_at(s, Some(x));
    let mesh = mesh(&cell);
    let c = kuks(&cell, f);
    let (r, kept) = solve_kuks_keep(&cell, &prep, None, &mesh, &c).expect("scf");
    let gc = KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(kept.derivative_budget(None)),
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid()),
            ao_threshold: c.xc.ao_threshold,
            exxdiv: Some(ExxDiv::Ewald),
            ..Default::default()
        },
        ..Default::default()
    };
    let jk = kept.grad_jk(None, &c.kuhf.rsgdf).unwrap();
    let g = kpoint_uks_gradient(
        &cell,
        &prep,
        &mesh,
        &c.kuhf.hcore,
        &kept.hk,
        jk,
        &r.scf,
        f,
        &gc,
    )
    .expect("force");
    (r.scf.energy, g.grad.iter().copied().collect())
}

/// Library Gamma RKS range-separated `(E, dE/dR)` at `x`
/// (`crate::periodic::rks_config`, `gamma_rks_any_keep`,
/// `gamma_rks_rsh_gradient`).
pub fn lib_gamma_rks_rsh_force(s: &System, f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    let (cell, prep) = cell_at(s, Some(x));
    let (hc, eri) = gamma_ints(&cell, &prep);
    let mut c = GammaRksConfig::new(f);
    (c.grid, c.exxdiv) = (grid(), ExxDiv::Ewald);
    c.scf = RhfConfig {
        use_sad_guess: false,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    };
    let (r, ex) = gamma_rks_any_keep(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        None,
        sizing(),
        &c,
    )
    .expect("scf");
    let ex = ex.expect("a range-separated hybrid keeps its exchange");
    let g = gamma_rks_rsh_gradient(
        &cell,
        &prep,
        &hcore_cfg(),
        &r.scf,
        &c,
        &ex,
        &KGradConfig::default(),
    )
    .expect("force");
    (r.scf.energy, g.grad.iter().copied().collect())
}

/// Library Gamma UKS range-separated `(E, dE/dR)` at `x`
/// (`crate::periodic::uks_config`, `gamma_uks_any_keep`,
/// `gamma_uks_rsh_gradient`).
pub fn lib_gamma_uks_rsh_force(s: &System, f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    let (cell, prep) = cell_at(s, Some(x));
    let (hc, eri) = gamma_ints(&cell, &prep);
    let mut c = GammaUksConfig::new(f);
    (c.grid, c.exxdiv, c.ewald_start) = (grid(), ExxDiv::Ewald, ferric_pbc::EwaldStart::Staged);
    c.scf = RhfConfig {
        use_sad_guess: false,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    };
    let (r, ex) = gamma_uks_any_keep(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        None,
        sizing(),
        &c,
    )
    .expect("scf");
    let ex = ex.expect("a range-separated hybrid keeps its exchange");
    let g = gamma_uks_rsh_gradient(
        &cell,
        &prep,
        &hcore_cfg(),
        &r.scf,
        &c,
        &ex,
        &KGradConfig::default(),
    )
    .expect("force");
    (r.scf.energy, g.grad.iter().copied().collect())
}

fn xc_stress_cfg(kept: &ferric_pbc::KRetained, ao: f64) -> KRksStressConfig {
    KRksStressConfig {
        stress: KStressConfig {
            budget_bytes: Some(kept.derivative_budget(None)),
            ..Default::default()
        },
        grid: KXcGrid::Atom(grid()),
        ao_threshold: ao,
        exxdiv: Some(ExxDiv::Ewald),
        ..Default::default()
    }
}

/// Library k-point RKS `(E, sigma)` (the Python `with_stress` sequence:
/// `solve_krks_keep`, then `kpoint_rks_stress` on the retained hcore / J/K).
pub fn lib_krks_stress(s: &System, f: &str) -> (f64, Vec<f64>) {
    let (cell, prep) = cell_at(s, None);
    let mesh = mesh(&cell);
    let c = krks(&cell, f);
    let (r, kept) = solve_krks_keep(&cell, &prep, None, &mesh, &c).unwrap();
    let jk = kept.stress_jk(None, &c.krhf.rsgdf).unwrap();
    let x = xc_stress_cfg(&kept, c.xc.ao_threshold);
    let sig = kpoint_rks_stress(
        &cell,
        &prep,
        &mesh,
        &c.krhf.hcore,
        &kept.hk,
        jk,
        &r.scf,
        f,
        &x,
    )
    .unwrap()
    .sigma;
    (r.scf.energy, sig.iter().flatten().copied().collect())
}

/// Library k-point UKS `(E, sigma)` (`solve_kuks_keep`, `kpoint_uks_stress`).
pub fn lib_kuks_stress(s: &System, f: &str) -> (f64, Vec<f64>) {
    let (cell, prep) = cell_at(s, None);
    let mesh = mesh(&cell);
    let c = kuks(&cell, f);
    let (r, kept) = solve_kuks_keep(&cell, &prep, None, &mesh, &c).unwrap();
    let jk = kept.stress_jk(None, &c.kuhf.rsgdf).unwrap();
    let x = KUksStressConfig {
        base: xc_stress_cfg(&kept, c.xc.ao_threshold),
        ..Default::default()
    };
    let sig = kpoint_uks_stress(
        &cell,
        &prep,
        &mesh,
        &c.kuhf.hcore,
        &kept.hk,
        jk,
        &r.scf,
        f,
        &x,
    )
    .unwrap()
    .sigma;
    (r.scf.energy, sig.iter().flatten().copied().collect())
}

/// Library force routes, by (closed shell?, k-mesh?).
pub type ForceFn = fn(&System, &str, &[f64]) -> (f64, Vec<f64>);

/// CLI `task = "optimize"` (one BFGS step) equals the library at the
/// returned geometry, energy and force, bit for bit.
pub fn force_anchor(tag: &str, sys: System, kind: &str, f: &str, kmesh: &str, lib: ForceFn) {
    let v = cli(&Input {
        tag,
        sys,
        kind,
        functional: f,
        task: "optimize",
        kmesh,
    });
    let e = v["energy"].as_f64().expect("energy");
    let x = floats(&v["extra"]["final_geometry_bohr"]);
    let g = floats(&v["extra"]["final_gradient_hartree_per_bohr"]);
    assert_eq!(x.len(), 3 * sys.atoms.len());
    let x0: Vec<f64> = sys.atoms.iter().flatten().copied().collect();
    let moved = x
        .iter()
        .zip(&x0)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    // The xyz round trip alone moves a coordinate by ~1e-13 Bohr.
    assert!(
        moved > 1e-6,
        "{tag}: the optimizer did not move ({moved:e})"
    );
    let (el, gl) = lib(&sys, f, &x);
    let gmax = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    eprintln!("{tag}: E {e:.16} max|g| {gmax:.3e}; library E {el:.16}");
    assert!(gmax > 1e-3, "{tag}: trivial force");
    assert_eq!(e.to_bits(), el.to_bits(), "{tag}: energy {e} vs {el}");
    assert_eq!(bits(&g), bits(&gl), "{tag}: force {g:?} vs {gl:?}");
}

/// CLI `task = "energy"` with `[cell] stress = true` equals the library
/// energy and stress tensor bit for bit.
pub fn stress_anchor(
    tag: &str,
    sys: System,
    kind: &str,
    f: &str,
    lib: fn(&System, &str) -> (f64, Vec<f64>),
) {
    let v = cli(&Input {
        tag,
        sys,
        kind,
        functional: f,
        task: "energy",
        kmesh: "[1, 1, 2]",
    });
    let e = v["energy"].as_f64().expect("energy");
    let s: Vec<f64> = v["extra"]["stress_hartree_per_bohr3"]
        .as_array()
        .expect("stress rows")
        .iter()
        .flat_map(floats)
        .collect();
    let (el, sl) = lib(&sys, f);
    let smax = sl.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    eprintln!("{tag}: E {e:.16} max|sigma| {smax:.3e}; library E {el:.16}");
    assert!(smax > 1e-6, "{tag}: trivial stress");
    assert_eq!(e.to_bits(), el.to_bits(), "{tag}: energy {e} vs {el}");
    assert_eq!(bits(&s), bits(&sl), "{tag}: stress {s:?} vs {sl:?}");
}
