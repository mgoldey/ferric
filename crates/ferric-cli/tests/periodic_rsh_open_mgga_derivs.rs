//! Issue #498: the library routes wired to the CLI by it.
//!
//! * ROKS with range-separated hybrids (energy; library `solve_kroks`,
//!   `gamma_rsh::gamma_roks_any`) at a k-mesh and at the Gamma point.
//! * Closed-shell range-separated hybrid FORCES on the dense-AFT J/K
//!   (`task = "optimize"`; library `kpoint_rks_gradient`,
//!   `gamma_rsh::gamma_rks_rsh_gradient`) at a k-mesh and at the Gamma point.
//! * Open-shell UKS SCAN / r2SCAN forces on the dense-AFT J/K
//!   (`kpoint_uks_gradient`). The matching STRESS anchors (closed-shell
//!   HSE06, UKS SCAN, closed-shell CAM-B3LYP on RS-GDF) are cases of
//!   `periodic_kpoint_stress.rs`.
//! * Open-shell UKS range-separated hybrid FORCES on the dense-AFT J/K
//!   (`kpoint_uks_gradient` at a k-mesh, `gamma_rsh::gamma_uks_rsh_gradient`
//!   through `gamma_uks_any_keep` at the Gamma point).
//! * Closed-shell range-separated hybrid FORCES on the k-point RS-GDF J/K
//!   (`kpoint_rks_gradient` on the `KRsGdf::build_rsh` source the SCF kept).
//!
//! EXACTNESS ANCHOR (CLI vs library, BIT FOR BIT): the CLI energy, and for an
//! optimize run the returned geometry and the gradient evaluated there, are
//! read at full precision from the run log's `run_end` record (serde_json
//! parses floats correctly rounded since #489). The library is then called
//! on the SAME cell (the logged Bohr positions written into the parsed
//! molecule), mesh, grid and configs (the CLI's own config builders restated
//! field by field), and every number must be identical to the last bit.
//! Fixtures: H2 (closed shell) and an H3 doublet in a 4 Bohr cubic cell,
//! STO-3G, dense AFT, exxdiv ewald, XC grid 20x50 with a 14 Bohr cutoff.
//!
//! Each anchor also checks its library value against a sibling that a
//! mis-routed CLI would have produced (the global hybrid PBE0 for HSE06, PBE
//! for SCAN): those differ by far more than one ulp, so a bit-equal pass
//! cannot come from the wrong route.
//!
//! Measured (2026-10-20, the UKS RSH and RS-GDF rows 2026-10-23): every
//! anchor bit-equal. Siblings at the same geometry: k ROKS HSE06 vs PBE0
//! 2.5e-4 Ha, Gamma ROKS CAM-B3LYP vs B3LYP 2.5e-2 Ha; force k HSE06 vs PBE0
//! 5.1e-4, Gamma CAM-B3LYP vs HSE06 2.0e-3, UKS SCAN vs PBE 4.9e-3, r2SCAN vs
//! SCAN 1.4e-3, k UKS HSE06 vs PBE0 4.2e-4, Gamma UKS CAM-B3LYP vs HSE06
//! 3.6e-3, k RKS CAM-B3LYP vs B3LYP on RS-GDF 5.7e-4 Ha/Bohr.
//!
//! Mutation ledger (applied to the CLI source, rebuilt, this file run, then
//! reverted), all KILLED:
//! * Gamma ROKS with `exxdiv = none` forced: CLI -1.11639 vs library -1.57434.
//! * k-point ROKS HSE06 grid `n_radial + 1`: CLI -1.62024 vs -1.62172.
//! * k-point force XC config with `exxdiv = none`; the Gamma RSH force given a
//!   config with `n_radial + 1`; the UKS force on a 110-point angular grid:
//!   each makes the CLI run fail (the library's exxdiv / Fock guards refuse
//!   the mismatched derivative), so the anchor test fails.
//! * The run log reporting the FIRST evaluated gradient instead of the one at
//!   the returned geometry: all four force anchors fail on the bits (e.g. k
//!   HSE06 z-force -1.74e-2 vs +5.2e-3).
//! * (2026-10-23) the Gamma UKS range-separated force given a config with
//!   `n_radial + 1`: the CLI run fails (the routed k-point Fock guard refuses
//!   it, fock_alpha off by 1.17e-3); the k-point UKS force run with the
//!   K-swap spin defect (`KUSpinMutation::PulayKSwap`): the HSE06 anchor fails
//!   on the bits (z-force of atom 0 0.142 vs 0.107).
//! * (periodic_kpoint_stress.rs) the k-point stress XC config with `exxdiv =
//!   none`: the HSE06 stress anchors fail (the CLI run errors); the SCAN /
//!   r2SCAN ones pass (no exact exchange: an equivalent mutant there).
//!
//! Refusals with exact wording: the Gamma RKS range-separated derivatives on
//! RS-GDF, derivatives of an unvalidated range-separated hybrid (wB97X; RKS
//! and UKS, k and Gamma), the UKS range-separated stress, the UKS
//! range-separated force on RS-GDF (k and Gamma), every ROKS range-separated
//! derivative (k force and stress, Gamma force), ROKS meta-GGA, UKS meta-GGA
//! derivatives on RS-GDF.

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

const A_BOHR: f64 = 4.0;
const OMEGA: f64 = 0.8;
const CUTOFF_BOHR: f64 = 14.0;
const N_RADIAL: usize = 20;
const N_ANGULAR: usize = 50;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
/// Two optimizer steps: the returned geometry is a moved one.
const OPT: &str = "max_steps = 2\ng_max_thresh = 1e-9\ng_rms_thresh = 1e-9\ne_conv = 1e-14";
const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";

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
    let dir = std::env::temp_dir().join(format!("ferric-b498-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn xyz(atoms_bohr: &[[f64; 3]]) -> String {
    let b = BOHR_RADIUS_ANGSTROM;
    let mut s = format!("{}\nissue 498 (Angstrom)\n", atoms_bohr.len());
    for p in atoms_bohr {
        s += &format!("H {:.12} {:.12} {:.12}\n", p[0] * b, p[1] * b, p[2] * b);
    }
    s
}

/// One CLI input. `kind`: "rhf" (closed shell), "uhf" (UKS) or "rohf" (ROKS);
/// `kmesh` empty = Gamma point; `cell_extra` appended to `[cell]`.
struct Input<'a> {
    tag: &'a str,
    kind: &'a str,
    functional: &'a str,
    task: &'a str,
    kmesh: &'a str,
    cell_extra: &'a str,
}

impl Input<'_> {
    fn atoms(&self) -> &'static [[f64; 3]] {
        if self.kind == "rhf" {
            &H2
        } else {
            &H3
        }
    }
}

/// Write the input (and its xyz) into a fresh scratch directory.
fn write(i: &Input) -> PathBuf {
    let dir = scratch(i.tag);
    let xyz_path = dir.join("m.xyz");
    std::fs::write(&xyz_path, xyz(i.atoms())).unwrap();
    let optimize = if i.task == "optimize" {
        format!("[optimize]\n{OPT}")
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

fn run(toml: &Path) -> Output {
    Command::new(ferric_cli_bin())
        .arg(toml)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("run ferric-cli")
}

fn ok(toml: &Path, tag: &str) -> serde_json::Value {
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

fn refused(i: &Input) -> String {
    let o = run(&write(i));
    assert!(!o.status.success(), "{} must be refused", i.tag);
    String::from_utf8_lossy(&o.stderr).into_owned()
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
/// run) the logged Bohr positions written into the parsed molecule exactly
/// as the CLI's optimizer does.
fn cell_at(atoms: &[[f64; 3]], mult: usize, at: Option<&[f64]>) -> (Cell, PreparedBasis) {
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

/// `crate::periodic::krohf_config` (dense).
fn krohf(cell: &Cell) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, ExxDiv::Ewald);
    c.scf = kscf();
    c.hcore = hcore_cfg();
    c.dense = kdense();
    c
}

fn sizing() -> GammaRshSizing {
    GammaRshSizing {
        rsgdf_omega: DEFAULT_RSGDF_OMEGA,
        budget_bytes: None,
        max_eri_bytes: cli_max_eri_bytes(),
        range_split: false,
    }
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

// ───────────────────────────────────────────────────────── ROKS energies ──

fn lib_kroks_energy(f: &str) -> f64 {
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
fn lib_gamma_roks_energy(f: &str) -> f64 {
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

fn roks_energy_anchor(tag: &str, f: &str, kmesh: &str, lib: f64, sibling: f64) {
    let i = Input {
        tag,
        kind: "rohf",
        functional: f,
        task: "energy",
        kmesh,
        cell_extra: "",
    };
    let v = ok(&write(&i), tag);
    let cli = v["energy"].as_f64().expect("energy");
    eprintln!("{tag}: CLI {cli:.16} library {lib:.16} sibling {sibling:.16}");
    assert_eq!(cli.to_bits(), lib.to_bits(), "{tag}: {cli} vs {lib}");
    assert!((lib - sibling).abs() > 1e-5, "{tag}: sibling too close");
}

#[test]
fn kpoint_roks_hse06_energy_equals_the_library_bit_for_bit() {
    let lib = lib_kroks_energy("HSE06");
    roks_energy_anchor(
        "kroks-hse",
        "HSE06",
        "[1, 1, 2]",
        lib,
        lib_kroks_energy("PBE0"),
    );
}

#[test]
fn gamma_roks_cam_b3lyp_energy_equals_the_library_bit_for_bit() {
    let lib = lib_gamma_roks_energy(CAM);
    roks_energy_anchor("groks-cam", CAM, "", lib, lib_gamma_roks_energy("B3LYP"));
}

// ──────────────────────────────────────────────────────────────── forces ──

/// The optimize run's `(final energy, geometry Bohr, gradient)` from `run_end`.
fn optimize(tag: &str, kind: &str, f: &str, kmesh: &str) -> (f64, Vec<f64>, Vec<f64>) {
    optimize_on(tag, kind, f, kmesh, "")
}

/// [`optimize`] with `cell_extra` appended to `[cell]` (e.g. the RS-GDF J/K).
fn optimize_on(
    tag: &str,
    kind: &str,
    f: &str,
    kmesh: &str,
    cell_extra: &str,
) -> (f64, Vec<f64>, Vec<f64>) {
    let i = Input {
        tag,
        kind,
        functional: f,
        task: "optimize",
        kmesh,
        cell_extra,
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
fn lib_krks_force(f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    lib_krks_force_jk(f, x, false)
}

/// [`lib_krks_force`] on the RS-GDF J/K (`[cell] jk = "rsgdf"`, cc-pVDZ-RI,
/// default gdf_omega): the plan turns the SR column rotation on for the hcore
/// and the fit.
fn lib_krks_force_rsgdf(f: &str, x: &[f64]) -> (f64, Vec<f64>) {
    lib_krks_force_jk(f, x, true)
}

fn lib_krks_force_jk(f: &str, x: &[f64], rsgdf: bool) -> (f64, Vec<f64>) {
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
fn lib_kuks_force(f: &str, x: &[f64]) -> (f64, Vec<f64>) {
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
fn lib_gamma_rks_rsh_force(f: &str, x: &[f64]) -> (f64, Vec<f64>) {
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
fn lib_gamma_uks_rsh_force(f: &str, x: &[f64]) -> (f64, Vec<f64>) {
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

/// CLI `(E, x, g)` equals the library at `x` bit for bit; the library force
/// of `sibling` at the same `x` differs (the anchor can see a mis-route).
fn force_anchor(
    tag: &str,
    cli: (f64, Vec<f64>, Vec<f64>),
    lib: impl Fn(&str, &[f64]) -> (f64, Vec<f64>),
    f: &str,
    sibling: &str,
) {
    let (e, x, g) = cli;
    let (el, gl) = lib(f, &x);
    let gmax = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    eprintln!("{tag}: E {e:.16} max|g| {gmax:.3e}; library E {el:.16}");
    assert!(gmax > 1e-3, "{tag}: trivial force");
    assert_eq!(e.to_bits(), el.to_bits(), "{tag}: energy {e} vs {el}");
    assert_eq!(bits(&g), bits(&gl), "{tag}: force {g:?} vs {gl:?}");
    let (_, gs) = lib(sibling, &x);
    let d = g
        .iter()
        .zip(&gs)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("{tag}: {sibling} force differs by {d:.2e}");
    assert!(d > 1e-6, "{tag}: sibling {sibling} force too close: {d:e}");
}

#[test]
fn kpoint_rks_hse06_force_equals_the_library_bit_for_bit() {
    let cli = optimize("krks-hse-opt", "rhf", "HSE06", "[1, 1, 2]");
    force_anchor("krks-hse-opt", cli, lib_krks_force, "HSE06", "PBE0");
}

#[test]
fn gamma_rks_cam_b3lyp_force_equals_the_library_bit_for_bit() {
    let cli = optimize("grks-cam-opt", "rhf", CAM, "");
    force_anchor("grks-cam-opt", cli, lib_gamma_rks_rsh_force, CAM, "HSE06");
}

#[test]
fn kpoint_uks_scan_force_equals_the_library_bit_for_bit() {
    let cli = optimize("kuks-scan-opt", "uhf", "SCAN", "[1, 1, 2]");
    force_anchor("kuks-scan-opt", cli, lib_kuks_force, "SCAN", "PBE");
}

#[test]
fn kpoint_uks_r2scan_force_equals_the_library_bit_for_bit() {
    let cli = optimize("kuks-r2scan-opt", "uhf", "r2SCAN", "[1, 1, 2]");
    force_anchor("kuks-r2scan-opt", cli, lib_kuks_force, "r2SCAN", "SCAN");
}

#[test]
fn kpoint_uks_hse06_force_equals_the_library_bit_for_bit() {
    let cli = optimize("kuks-hse-opt", "uhf", "HSE06", "[1, 1, 2]");
    force_anchor("kuks-hse-opt", cli, lib_kuks_force, "HSE06", "PBE0");
}

#[test]
fn gamma_uks_cam_b3lyp_force_equals_the_library_bit_for_bit() {
    let cli = optimize("guks-cam-opt", "uhf", CAM, "");
    force_anchor("guks-cam-opt", cli, lib_gamma_uks_rsh_force, CAM, "HSE06");
}

#[test]
fn kpoint_rks_cam_b3lyp_rsgdf_force_equals_the_library_bit_for_bit() {
    let cli = optimize_on("krks-cam-gdf-opt", "rhf", CAM, "[1, 1, 2]", RSGDF);
    force_anchor("krks-cam-gdf-opt", cli, lib_krks_force_rsgdf, CAM, "B3LYP");
}

// ────────────────────────────────────────────────────────────── refusals ──

const RSGDF: &str = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";

#[test]
fn refusals_have_exact_wording() {
    // Closed-shell RSH derivatives: on RS-GDF the k-point force and stress
    // are wired (anchored above and in periodic_kpoint_stress.rs); the Gamma
    // RS-GDF optimize stays refused (the Gamma route has no RS-GDF exchange
    // derivative).
    let base = Input {
        tag: "",
        kind: "rhf",
        functional: "HSE06",
        task: "optimize",
        kmesh: "",
        cell_extra: RSGDF,
    };
    let e = refused(&Input {
        tag: "r-g-opt",
        ..base
    });
    assert!(
        e.contains("method.task = \"optimize\" with [cell]: ")
            && e.contains(
                "HSE06: the Gamma-point RKS range-separated hybrid force and stress run on \
                 the dense-AFT J/K only (the Gamma route has no RS-GDF exchange derivative"
            ),
        "{e}"
    );
    // Range-separated hybrids outside the force/stress validation (wB97X)
    // are refused by name for derivatives, at a k-mesh and at Gamma (dense).
    let wb = "HYB_GGA_XC_WB97X: the range-separated hybrid force and stress are validated \
              for HSE06 / CAM-B3LYP (HYB_GGA_XC_CAM_B3LYP) only";
    for (tag, kmesh, prefix) in [
        (
            "wb-k-opt",
            "[1, 1, 2]",
            "method.task = \"optimize\" with [cell] kmesh: ",
        ),
        ("wb-g-opt", "", "method.task = \"optimize\" with [cell]: "),
    ] {
        let e = refused(&Input {
            tag,
            functional: "HYB_GGA_XC_WB97X",
            kmesh,
            cell_extra: "",
            ..base
        });
        assert!(e.contains(prefix) && e.contains(wb), "{tag}: {e}");
    }
    let e = refused(&Input {
        tag: "wb-k-stress",
        functional: "HYB_GGA_XC_WB97X",
        task: "energy",
        kmesh: "[1, 1, 2]",
        cell_extra: "stress = true",
        ..base
    });
    assert!(
        e.contains("[cell] stress = true: ") && e.contains(wb),
        "{e}"
    );
    // ... and for the open-shell UKS force (k-point and Gamma).
    for (tag, kmesh) in [("wb-uks-k-opt", "[1, 1, 2]"), ("wb-uks-g-opt", "")] {
        let e = refused(&Input {
            tag,
            kind: "uhf",
            functional: "HYB_GGA_XC_WB97X",
            kmesh,
            cell_extra: "",
            ..base
        });
        assert!(e.contains("optimize") && e.contains(wb), "{tag}: {e}");
    }
    // Open-shell RSH derivatives that stay refused: the UKS stress (k-point;
    // the CLI has no Gamma stress), the UKS force on RS-GDF (k and Gamma) and
    // every ROKS derivative (k-point force and stress, Gamma force).
    let open = Input {
        tag: "",
        kind: "uhf",
        functional: "HSE06",
        task: "optimize",
        kmesh: "[1, 1, 2]",
        cell_extra: "",
    };
    let e = refused(&Input {
        tag: "o-k-stress-uks",
        task: "energy",
        cell_extra: "stress = true",
        ..open
    });
    assert!(
        e.contains(
            "[cell] stress = true: General error: HSE06: the k-point UKS range-separated hybrid \
             stress is not implemented (the UKS stress has no range-separated term); the UKS \
             force is available on the dense-AFT J/K"
        ),
        "{e}"
    );
    let e = refused(&Input {
        tag: "o-k-opt-uks-gdf",
        cell_extra: RSGDF,
        ..open
    });
    assert!(
        e.contains(
            "method.task = \"optimize\" with [cell] kmesh: General error: HSE06: the k-point \
             UKS range-separated hybrid force runs on the dense-AFT J/K only in the bindings (the \
             library RS-GDF force, kpoint_uks_gradient, is not wired yet); use jk = \"dense\""
        ),
        "{e}"
    );
    let e = refused(&Input {
        tag: "o-g-opt-uks-gdf",
        kmesh: "",
        cell_extra: RSGDF,
        ..open
    });
    assert!(
        e.contains("method.task = \"optimize\" with [cell]: ")
            && e.contains(
                "HSE06: the Gamma-point UKS range-separated hybrid force runs on the dense-AFT \
                 J/K only (the Gamma route has no RS-GDF exchange derivative"
            ),
        "{e}"
    );
    let roks_k = "General error: HSE06: the k-point ROKS range-separated hybrid force and \
                  stress are not implemented in the bindings (the library has the dense-AFT \
                  and RS-GDF ROKS force, kpoint_roks_gradient, unwired; the ROKS stress has no \
                  range-separated term); the UKS force is available on the dense-AFT J/K";
    let ro = Input {
        tag: "",
        kind: "rohf",
        ..open
    };
    let e = refused(&Input {
        tag: "o-k-opt-roks",
        ..ro
    });
    assert!(
        e.contains(&format!(
            "method.task = \"optimize\" with [cell] kmesh: {roks_k}"
        )),
        "{e}"
    );
    let e = refused(&Input {
        tag: "o-k-stress-roks",
        task: "energy",
        cell_extra: "stress = true",
        ..ro
    });
    assert!(
        e.contains(&format!("[cell] stress = true: {roks_k}")),
        "{e}"
    );
    let e = refused(&Input {
        tag: "o-g-opt-roks",
        kmesh: "",
        ..ro
    });
    assert!(
        e.contains(
            "HSE06: the Gamma-point ROKS range-separated hybrid force and stress are not \
             implemented in the bindings (energy only; the library has the dense-AFT ROKS \
             force, gamma_roks_rsh_gradient, unwired); the UKS force is available on the \
             dense-AFT J/K"
        ),
        "{e}"
    );
    // ROKS meta-GGA (energy gate) and UKS meta-GGA derivatives on RS-GDF (the
    // energy gate already names the J/K).
    let e = refused(&Input {
        tag: "roks-scan",
        kind: "rohf",
        functional: "SCAN",
        task: "energy",
        kmesh: "[1, 1, 2]",
        cell_extra: "",
    });
    assert!(
        e.contains(
            "k-point ROKS supports LDA, GGA, global-hybrid and range-separated hybrid \
                    functionals only (no meta-GGA for ROKS"
        ),
        "{e}"
    );
    let e = refused(&Input {
        tag: "uks-scan-rsgdf",
        kind: "uhf",
        functional: "SCAN",
        task: "optimize",
        kmesh: "[1, 1, 2]",
        cell_extra: RSGDF,
    });
    assert!(e.contains("dense-AFT J/K only"), "{e}");
}
