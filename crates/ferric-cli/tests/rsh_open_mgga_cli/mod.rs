//! Shared harness of `periodic_rsh_open_mgga_derivs.rs` (per commit) and
//! `validation_periodic_rsh_open_mgga_derivs.rs` (weekly): the issue #498
//! CLI-vs-library anchors for the ROKS range-separated energies and the
//! closed- / open-shell range-separated and UKS meta-GGA forces. Fixtures:
//! H2 (closed shell) and an H3 doublet in a 4 Bohr cubic cell, STO-3G, dense
//! AFT, exxdiv ewald, XC grid 20x50 with a 14 Bohr cutoff.
#![allow(dead_code)]

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::units::BOHR_RADIUS_ANGSTROM;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::DEFAULT_DENSE_AFT_PRECISION;
use ferric_pbc::gamma_rsh::{
    gamma_rks_any_keep, gamma_rks_rsh_gradient, gamma_roks_any, gamma_uks_any_keep,
    gamma_uks_rsh_gradient, GammaRshSizing,
};
use ferric_pbc::kroks::{KRohfConfig, KRoksConfig};
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KXcGrid};
use ferric_pbc::rohf::GammaRoksConfig;
use ferric_pbc::rsgdf::DEFAULT_RSGDF_OMEGA;
use ferric_pbc::{
    kpoint_uks_gradient, periodic_hcore, solve_krks_keep, solve_kroks, solve_kuks_keep, Cell,
    DenseAftEri, ExxDiv, GammaRksConfig, GammaUhfIntegrals, GammaUksConfig, KDenseAftConfig,
    KGradConfig, KJkKind, KPointMesh, KRksConfig, KRsGdfConfig, KScfConfig, KUksConfig,
    KUksGradConfig, MeshCentring, PeriodicGridConfig, PeriodicHcoreConfig, RsGdfConfig,
    SrColumnRotation,
};
use ferric_scf::rhf::RhfConfig;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub const A_BOHR: f64 = 4.0;
pub const OMEGA: f64 = 0.8;
pub const CUTOFF_BOHR: f64 = 14.0;
pub const N_RADIAL: usize = 20;
pub const N_ANGULAR: usize = 50;
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
pub const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
/// Optimizer steps of a force anchor: two, so the returned geometry is a
/// moved one (the run log must report the gradient AT that geometry).
pub const OPT_STEPS: usize = 2;
/// Convergence keys that never stop the optimizer early.
pub const OPT: &str = "g_max_thresh = 1e-9\ng_rms_thresh = 1e-9\ne_conv = 1e-14";
pub const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";

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

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

pub fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferric-b498-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn xyz(atoms_bohr: &[[f64; 3]]) -> String {
    let b = BOHR_RADIUS_ANGSTROM;
    let mut s = format!("{}\nissue 498 (Angstrom)\n", atoms_bohr.len());
    for p in atoms_bohr {
        s += &format!("H {:.12} {:.12} {:.12}\n", p[0] * b, p[1] * b, p[2] * b);
    }
    s
}

/// One CLI input. `kind`: "rhf" (closed shell), "uhf" (UKS) or "rohf" (ROKS);
/// `kmesh` empty = Gamma point; `cell_extra` appended to `[cell]`.
pub struct Input<'a> {
    pub tag: &'a str,
    pub kind: &'a str,
    pub functional: &'a str,
    pub task: &'a str,
    pub kmesh: &'a str,
    pub cell_extra: &'a str,
    /// `[optimize] max_steps` (read only when `task = "optimize"`).
    pub steps: usize,
}

impl Input<'_> {
    pub fn atoms(&self) -> &'static [[f64; 3]] {
        if self.kind == "rhf" {
            &H2
        } else {
            &H3
        }
    }
}

/// Write the input (and its xyz) into a fresh scratch directory.
pub fn write(i: &Input) -> PathBuf {
    let dir = scratch(i.tag);
    let xyz_path = dir.join("m.xyz");
    std::fs::write(&xyz_path, xyz(i.atoms())).unwrap();
    let optimize = if i.task == "optimize" {
        format!("[optimize]\nmax_steps = {}\n{OPT}", i.steps)
    } else {
        String::new()
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
name = "sto-3g"
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
{extra}
{optimize}
"#,
            xyz = xyz_path.display(),
            mult = if i.kind == "rhf" { 1 } else { 2 },
            kind = i.kind,
            task = i.task,
            f = i.functional,
            json = dir.join("run.jsonl").display(),
            a = A_BOHR,
            extra = i.cell_extra,
        ),
    )
    .unwrap();
    toml
}

pub fn run(toml: &Path) -> Output {
    Command::new(ferric_cli_bin())
        .arg(toml)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("run ferric-cli")
}

pub fn ok(toml: &Path, tag: &str) -> serde_json::Value {
    let o = run(toml);
    assert!(
        o.status.success(),
        "{tag}: ferric-cli failed\n{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    let log = std::fs::read_to_string(toml.with_file_name("run.jsonl")).expect("run log");
    for line in log.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if v.get("record").and_then(|r| r.as_str()) == Some("run_end") {
            return v;
        }
    }
    panic!("{tag}: no run_end record in:\n{log}");
}

pub fn refused(i: &Input) -> String {
    let o = run(&write(i));
    assert!(!o.status.success(), "{} must be refused", i.tag);
    String::from_utf8_lossy(&o.stderr).into_owned()
}

pub fn floats(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| x.as_f64().expect("f64"))
        .collect()
}

pub fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The cell the CLI built: the same xyz round trip, then (for an optimize
/// run) the logged Bohr positions written into the parsed molecule exactly
/// as the CLI's optimizer does.
pub fn cell_at(atoms: &[[f64; 3]], mult: usize, at: Option<&[f64]>) -> (Cell, PreparedBasis) {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let p = scratch(&format!("lib{n}")).join("l.xyz");
    std::fs::write(&p, xyz(atoms)).unwrap();
    let mut mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, mult).expect("xyz");
    if let Some(x) = at {
        for (k, a) in mol.atoms.iter_mut().enumerate() {
            (a.x, a.y, a.zpos) = (x[3 * k], x[3 * k + 1], x[3 * k + 2]);
        }
    }
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &basis::bundled("sto-3g").unwrap()).unwrap();
    (cell, prep)
}

pub fn grid() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(N_RADIAL, N_ANGULAR);
    g.neighbour_cutoff = Some(CUTOFF_BOHR);
    g
}

/// The CLI's dense-AFT cap (`[cell] max_eri_gb` default 0.5).
pub fn cli_max_eri_bytes() -> usize {
    ferric_core::memory::gib_to_bytes(0.5)
}

pub fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(SrColumnRotation::Off)
}

/// `crate::periodic::kscf_config` with the periodic defaults.
pub fn kscf() -> KScfConfig {
    KScfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        grad_conv: 1e-9,
        ..Default::default()
    }
}

pub fn kdense() -> KDenseAftConfig {
    KDenseAftConfig {
        max_bytes: cli_max_eri_bytes(),
        ..Default::default()
    }
}

/// `crate::periodic::krohf_config` (dense).
pub fn krohf(cell: &Cell) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, ExxDiv::Ewald);
    c.scf = kscf();
    c.hcore = hcore_cfg();
    c.dense = kdense();
    c
}

pub fn sizing() -> GammaRshSizing {
    GammaRshSizing {
        rsgdf_omega: DEFAULT_RSGDF_OMEGA,
        budget_bytes: None,
        max_eri_bytes: cli_max_eri_bytes(),
        range_split: false,
    }
}

/// The Gamma hcore and dense J/K of `crate::periodic::gamma_system`.
pub fn gamma_ints(cell: &Cell, prep: &PreparedBasis) -> (ferric_pbc::PeriodicHcore, DenseAftEri) {
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

// ───────────────────────────────────────────────────────── ROKS energies ──

pub fn lib_kroks_energy(f: &str) -> f64 {
    let (cell, prep) = cell_at(&H3, 2, None);
    let mesh = KPointMesh::new(&cell, [1, 1, 2], MeshCentring::Gamma).unwrap();
    let mut c = KRoksConfig::new(&cell, ExxDiv::Ewald, f);
    (c.rohf, c.grid) = (krohf(&cell), grid());
    solve_kroks(&cell, &prep, None, &mesh, &c)
        .unwrap_or_else(|e| panic!("solve_kroks {f}: {e}"))
        .scf
        .energy
}

/// `crate::periodic::roks_config` with `[scf]` unwritten: the
/// `GammaRoksConfig::new` level shift and iteration cap, density_conv 1e-10.
pub fn lib_gamma_roks_energy(f: &str) -> f64 {
    let (cell, prep) = cell_at(&H3, 2, None);
    let (hc, eri) = gamma_ints(&cell, &prep);
    let mut c = GammaRoksConfig::new(f);
    (c.grid, c.exxdiv, c.ewald_start) = (grid(), ExxDiv::Ewald, ferric_pbc::EwaldStart::Staged);
    c.scf = RhfConfig {
        level_shift: c.scf.level_shift,
        max_iter: c.scf.max_iter,
        use_sad_guess: false,
        density_conv: 1e-10,
        ..Default::default()
    };
    gamma_roks_any(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        None,
        sizing(),
        &c,
    )
    .unwrap_or_else(|e| panic!("gamma_roks_any {f}: {e}"))
    .scf
    .energy
}

pub fn roks_energy_anchor(tag: &str, f: &str, kmesh: &str, lib: f64, sibling: f64) {
    let i = Input {
        tag,
        kind: "rohf",
        functional: f,
        task: "energy",
        kmesh,
        cell_extra: "",
        steps: 0,
    };
    let v = ok(&write(&i), tag);
    let cli = v["energy"].as_f64().expect("energy");
    eprintln!("{tag}: CLI {cli:.16} library {lib:.16} sibling {sibling:.16}");
    assert_eq!(cli.to_bits(), lib.to_bits(), "{tag}: {cli} vs {lib}");
    assert!((lib - sibling).abs() > 1e-5, "{tag}: sibling too close");
}

// ──────────────────────────────────────────────────────────────── forces ──

/// The optimize run's `(final energy, geometry Bohr, gradient)` from `run_end`.
pub fn optimize(tag: &str, kind: &str, f: &str, kmesh: &str) -> (f64, Vec<f64>, Vec<f64>) {
    optimize_on(tag, kind, f, kmesh, "")
}

/// [`optimize`] with `cell_extra` appended to `[cell]` (e.g. the RS-GDF J/K).
pub fn optimize_on(
    tag: &str,
    kind: &str,
    f: &str,
    kmesh: &str,
    cell_extra: &str,
) -> (f64, Vec<f64>, Vec<f64>) {
    optimize_steps(tag, kind, f, kmesh, cell_extra, OPT_STEPS)
}

/// [`optimize_on`] with `steps` optimizer steps; `0` returns the INPUT
/// geometry and the gradient there (one CLI SCF + gradient, the cheapest
/// force anchor).
pub fn optimize_steps(
    tag: &str,
    kind: &str,
    f: &str,
    kmesh: &str,
    cell_extra: &str,
    steps: usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    let i = Input {
        tag,
        kind,
        functional: f,
        task: "optimize",
        kmesh,
        cell_extra,
        steps,
    };
    let v = ok(&write(&i), tag);
    let e = v["energy"].as_f64().expect("energy");
    let x = floats(&v["extra"]["final_geometry_bohr"]);
    let g = floats(&v["extra"]["final_gradient_hartree_per_bohr"]);
    assert_eq!(x.len(), g.len());
    (e, x, g)
}

/// Library k-point RKS `(E, dE/dR)` at `x` (`crate::periodic::krks_config`
/// and `KForceCtx::xc_cfg`) on the dense-AFT J/K.
pub fn lib_krks_force(f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    lib_krks_force_jk(f, x, false)
}

/// [`lib_krks_force`] on the RS-GDF J/K (`[cell] jk = "rsgdf"`, cc-pVDZ-RI,
/// default gdf_omega): the plan turns the SR column rotation on for the hcore
/// and the fit.
pub fn lib_krks_force_rsgdf(f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    lib_krks_force_jk(f, x, true)
}

pub fn lib_krks_force_jk(f: &str, x: &[f64], rsgdf: bool) -> (f64, Vec<f64>) {
    let (cell, prep) = cell_at(&H2, 1, Some(x));
    let mesh = KPointMesh::new(&cell, [1, 1, 2], MeshCentring::Gamma).unwrap();
    let mut c = KRksConfig::new(&cell, ExxDiv::Ewald, f);
    (c.krhf.scf, c.krhf.hcore, c.krhf.dense, c.grid) = (kscf(), hcore_cfg(), kdense(), grid());
    let aux_prep = rsgdf
        .then(|| PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap());
    if rsgdf {
        let rot = SrColumnRotation::on();
        c.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(rot);
        c.krhf.jk = KJkKind::RsGdf;
        c.krhf.rsgdf = KRsGdfConfig {
            gdf: RsGdfConfig {
                omega: DEFAULT_RSGDF_OMEGA,
                sr_column_rotation: rot,
                ..Default::default()
            },
            ..Default::default()
        };
    }
    let aux = aux_prep.as_ref();
    let (r, kept) = solve_krks_keep(&cell, &prep, aux, &mesh, &c).expect("scf");
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
    let jk = kept.grad_jk(aux, &c.krhf.rsgdf).unwrap();
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
pub fn lib_kuks_force(f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    let (cell, prep) = cell_at(&H3, 2, Some(x));
    let mesh = KPointMesh::new(&cell, [1, 1, 2], MeshCentring::Gamma).unwrap();
    let mut c = KUksConfig::new(&cell, ExxDiv::Ewald, f);
    (c.kuhf.scf, c.kuhf.hcore, c.kuhf.dense, c.grid) = (kscf(), hcore_cfg(), kdense(), grid());
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

/// Library Gamma RKS `(E, dE/dR)` at `x` (`crate::periodic::rks_config`,
/// `gamma_system` with `for_gradient`, `gamma_rks_any_keep`, and for a
/// range-separated hybrid `gamma_rks_rsh_gradient`).
pub fn lib_gamma_rks_rsh_force(f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    let (cell, prep) = cell_at(&H2, 1, Some(x));
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

/// Library Gamma UKS `(E, dE/dR)` of a range-separated hybrid at `x`
/// (`crate::periodic::uks_config` with `[scf]` unwritten, `gamma_system` with
/// `for_gradient`, `gamma_uks_any_keep`, `gamma_uks_rsh_gradient`).
pub fn lib_gamma_uks_rsh_force(f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    let (cell, prep) = cell_at(&H3, 2, Some(x));
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

/// CLI `(E, x, g)` equals the library at `x` bit for bit, and the force is
/// not trivially zero. Returns the CLI triple.
pub fn force_bits_anchor(
    tag: &str,
    cli: (f64, Vec<f64>, Vec<f64>),
    lib: &impl Fn(&str, &[f64]) -> (f64, Vec<f64>),
    f: &str,
) -> (f64, Vec<f64>, Vec<f64>) {
    let (e, x, g) = cli;
    let (el, gl) = lib(f, &x);
    let gmax = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    eprintln!("{tag}: E {e:.16} max|g| {gmax:.3e}; library E {el:.16}");
    assert!(gmax > 1e-3, "{tag}: trivial force");
    assert_eq!(e.to_bits(), el.to_bits(), "{tag}: energy {e} vs {el}");
    assert_eq!(bits(&g), bits(&gl), "{tag}: force {g:?} vs {gl:?}");
    (e, x, g)
}

/// CLI `(E, x, g)` equals the library at `x` bit for bit; the library force
/// of `sibling` at the same `x` differs (the anchor can see a mis-route).
pub fn force_anchor(
    tag: &str,
    cli: (f64, Vec<f64>, Vec<f64>),
    lib: impl Fn(&str, &[f64]) -> (f64, Vec<f64>),
    f: &str,
    sibling: &str,
) {
    let (_, x, g) = force_bits_anchor(tag, cli, &lib, f);
    let (_, gs) = lib(sibling, &x);
    let d = g
        .iter()
        .zip(&gs)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("{tag}: {sibling} force differs by {d:.2e}");
    assert!(d > 1e-6, "{tag}: sibling {sibling} force too close: {d:e}");
}

pub const RSGDF: &str = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";
