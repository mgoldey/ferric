//! `task = "optimize"` with `[cell] kmesh` (k-point RHF / UHF forces) is wired
//! to the library force.
//!
//! * EXACTNESS ANCHOR (CLI vs library): with loose optimizer thresholds the run
//!   stops after a few steps; the printed final gradient is the force at the
//!   printed final geometry, which must equal `kpoint_rhf_gradient` /
//!   `kpoint_uhf_gradient` called directly on the same cell, mesh and configs
//!   (evaluated at the printed 8-decimal geometry, bar 5e-7). A mesh, spin,
//!   unit or wrong-J/K slip in the wiring moves it by orders of magnitude
//!   (mesh mutant asserted).
//! * A real optimization (stretched H2, 1x1x3) converges to the optimizer's
//!   gradient thresholds, lowers the energy, and the printed final gradient
//!   equals the library force at the printed final geometry.
//! * Refusals: ROHF (k-point Kohn-Sham: periodic_kpoint_ks_optimize.rs).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::units::BOHR_RADIUS_ANGSTROM;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::{
    kpoint_rhf_gradient, kpoint_uhf_gradient, solve_krhf_keep, solve_kuhf_keep, Cell, ExxDiv,
    KGradConfig, KJkKind, KPointMesh, KRhfConfig, KUhfConfig, MeshCentring, PeriodicHcoreConfig,
    SrColumnRotation,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const A_BOHR: f64 = 5.0;
const OMEGA: f64 = 0.8;

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
    let dir = std::env::temp_dir().join(format!("ferric-kopt-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs an optimize input (STO-3G, dense J/K) and returns the process output.
fn run(tag: &str, xyz: &str, kind: &str, mult: usize, kmesh: &str, optimize: &str) -> Output {
    run_with(
        tag,
        xyz,
        kind,
        mult,
        kmesh,
        optimize,
        "sto-3g",
        "jk = \"dense\"",
    )
}

/// [`run`] with an explicit basis and J/K lines.
fn run_with(
    tag: &str,
    xyz: &str,
    kind: &str,
    mult: usize,
    kmesh: &str,
    optimize: &str,
    basis_name: &str,
    jk: &str,
) -> Output {
    let dir = scratch(tag);
    let xyz_path = dir.join("m.xyz");
    std::fs::write(&xyz_path, xyz).unwrap();
    let toml = dir.join("in.toml");
    std::fs::write(
        &toml,
        format!(
            r#"[molecule]
xyz = "{xyz}"
multiplicity = {mult}
[basis]
name = "{basis_name}"
[method]
kind = "{kind}"
task = "optimize"
[cell]
unit = "bohr"
lattice = [[{a}, 0.0, 0.0], [0.0, {a}, 0.0], [0.0, 0.0, {a}]]
kmesh = {kmesh}
exxdiv = "ewald"
omega = {omega}
{jk}
[optimize]
{optimize}
"#,
            xyz = xyz_path.display(),
            a = A_BOHR,
            omega = OMEGA,
            basis_name = basis_name,
            jk = jk,
        ),
    )
    .unwrap();
    Command::new(ferric_cli_bin())
        .arg(&toml)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("run ferric-cli")
}

fn ok(o: &Output, tag: &str) -> String {
    let out = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(
        o.status.success(),
        "{tag}: ferric-cli failed\n{out}\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    out
}

/// The `n` rows `SYM x y z` after the line starting with `header`.
fn rows(stdout: &str, header: &str, n: usize) -> Vec<[f64; 3]> {
    let mut it = stdout
        .lines()
        .skip_while(|l| !l.trim_start().starts_with(header));
    it.next()
        .unwrap_or_else(|| panic!("no {header:?} in:\n{stdout}"));
    it.take(n)
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .skip(1)
                .map(|t| t.parse().expect("number"))
                .collect();
            [v[0], v[1], v[2]]
        })
        .collect()
}

fn value(stdout: &str, key: &str) -> String {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with(key))
        .unwrap_or_else(|| panic!("no {key:?} in:\n{stdout}"))
        .split('=')
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

fn xyz(atoms_bohr: &[[f64; 3]]) -> String {
    let b = BOHR_RADIUS_ANGSTROM;
    let mut s = format!("{}\nk-point optimize test (Angstrom)\n", atoms_bohr.len());
    for p in atoms_bohr {
        s += &format!("H {:.12} {:.12} {:.12}\n", p[0] * b, p[1] * b, p[2] * b);
    }
    s
}

/// Library force at `atoms_bohr` with the configs the CLI builds for this input.
fn library_gradient(
    atoms_bohr: &[[f64; 3]],
    mult: usize,
    mesh_n: [usize; 3],
    uhf: bool,
    mesh_override: Option<[usize; 3]>,
) -> Vec<f64> {
    let dir = scratch("lib");
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(atoms_bohr)).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, mult).expect("xyz");
    let bs = basis::bundled("sto-3g").unwrap();
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let mesh =
        KPointMesh::new(&cell, mesh_override.unwrap_or(mesh_n), MeshCentring::Gamma).unwrap();
    let hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    let g = if uhf {
        let mut c = KUhfConfig::for_cell(&cell, ExxDiv::Ewald);
        c.scf.energy_conv = 1e-12;
        c.scf.grad_conv = 1e-9;
        c.hcore = hcore;
        let (r, kept) = solve_kuhf_keep(&cell, &prep, None, &mesh, &c).expect("scf");
        kpoint_uhf_gradient(
            &cell,
            &prep,
            &mesh,
            &c.hcore,
            &kept.hk,
            kept.grad_jk(None, &c.rsgdf).unwrap(),
            &r.scf,
            ExxDiv::Ewald,
            &KGradConfig::default(),
        )
    } else {
        let mut c = KRhfConfig::for_cell(&cell, ExxDiv::Ewald);
        c.scf.energy_conv = 1e-12;
        c.scf.grad_conv = 1e-9;
        c.hcore = hcore;
        let (r, kept) = solve_krhf_keep(&cell, &prep, None, &mesh, &c).expect("scf");
        kpoint_rhf_gradient(
            &cell,
            &prep,
            &mesh,
            &c.hcore,
            &kept.hk,
            kept.grad_jk(None, &c.rsgdf).unwrap(),
            &r,
            ExxDiv::Ewald,
            &KGradConfig::default(),
        )
    }
    .expect("gradient");
    g.grad.iter().copied().collect()
}

/// Printed geometry has 8 Angstrom decimals; the force moves by ~k*1e-8 over it.
const BAR: f64 = 5e-7;
const LOOSE: &str = "g_max_thresh = 1.0\ng_rms_thresh = 1.0\ne_conv = 1.0\nmax_steps = 3";
const H2_OFFAXIS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const H3_OFFAXIS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];

fn printed_gradient(stdout: &str, n: usize) -> Vec<f64> {
    rows(stdout, "final gradient", n)
        .into_iter()
        .flatten()
        .collect()
}

/// The printed final geometry (Angstrom, 8 decimals) in Bohr.
fn printed_atoms(stdout: &str, n: usize) -> Vec<[f64; 3]> {
    let b = BOHR_RADIUS_ANGSTROM;
    rows(stdout, "final geometry", n)
        .into_iter()
        .map(|r| [r[0] / b, r[1] / b, r[2] / b])
        .collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn printed_k_point_force_equals_the_library_force_rhf() {
    let o = run("rhf-grad", &xyz(&H2_OFFAXIS), "rhf", 1, "[1, 1, 3]", LOOSE);
    let out = ok(&o, "rhf");
    assert!(out.contains("analytic k-point RHF gradient"), "{out}");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let lib = library_gradient(&at, 1, [1, 1, 3], false, None);
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &lib);
    eprintln!("RHF CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}\n{g:?}\n{lib:?}");
    // MUTANT: a wrong mesh (1x1x1) moves the force by far more than the bar.
    let bad = library_gradient(&at, 1, [1, 1, 3], false, Some([1, 1, 1]));
    assert!(max_diff(&g, &bad) > 1e-4, "mesh mutant not detected");
}

#[test]
fn printed_k_point_force_equals_the_library_force_uhf() {
    let o = run("uhf-grad", &xyz(&H3_OFFAXIS), "uhf", 2, "[1, 1, 3]", LOOSE);
    let out = ok(&o, "uhf");
    assert!(out.contains("analytic k-point UHF gradient"), "{out}");
    let g = printed_gradient(&out, 3);
    let at = printed_atoms(&out, 3);
    let lib = library_gradient(&at, 2, [1, 1, 3], true, None);
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &lib);
    eprintln!("UHF CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}\n{g:?}\n{lib:?}");
    // MUTANT: a wrong mesh (1x1x1) moves the open-shell force beyond the bar.
    let bad = library_gradient(&at, 2, [1, 1, 3], true, Some([1, 1, 1]));
    assert!(max_diff(&g, &bad) > 1e-4, "mesh mutant not detected");
}

#[test]
fn k_point_optimize_converges_and_the_final_force_is_the_library_force() {
    let start = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 / BOHR_RADIUS_ANGSTROM]];
    let o = run("opt", &xyz(&start), "rhf", 1, "[1, 1, 3]", "max_steps = 30");
    let out = ok(&o, "opt");
    assert_eq!(value(&out, "converged"), "true", "{out}");
    let b = BOHR_RADIUS_ANGSTROM;
    let fin = rows(&out, "final geometry", 2);
    let atoms: Vec<[f64; 3]> = fin.iter().map(|r| [r[0] / b, r[1] / b, r[2] / b]).collect();
    let bond = ((atoms[1][2] - atoms[0][2]).powi(2)
        + (atoms[1][1] - atoms[0][1]).powi(2)
        + (atoms[1][0] - atoms[0][0]).powi(2))
    .sqrt();
    eprintln!("optimized bond {bond:.5} Bohr");
    assert!(
        bond < 1.0 / b - 0.05,
        "stretched H2 should contract: {bond}"
    );
    let g = printed_gradient(&out, 2);
    assert!(g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) < 4.5e-4);
    let lib = library_gradient(&atoms, 1, [1, 1, 3], false, None);
    // The printed geometry has 8 decimals (Angstrom): the force changes by
    // ~k * 1e-8 over the rounding, below the bar.
    assert!(max_diff(&g, &lib) < 5e-7, "{g:?} vs {lib:?}");
}

/// Library RS-GDF RHF force with the configs the CLI builds for
/// `jk = "rsgdf"`, `auxbasis = "cc-pvdz-ri"` (default column rotation ON and
/// default RS-GDF omega); also returns the rotated hcore column count.
fn library_gradient_rsgdf_ccpvdz(atoms_bohr: &[[f64; 3]]) -> (Vec<f64>, usize) {
    let dir = scratch("lib-gdf");
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(atoms_bohr)).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, 1).expect("xyz");
    let bs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let auxp = PreparedBasis::new(cell.mol(), &aux).unwrap();
    let mesh = KPointMesh::new(&cell, [1, 1, 2], MeshCentring::Gamma).unwrap();
    let mut c = KRhfConfig::for_cell(&cell, ExxDiv::Ewald);
    c.scf.energy_conv = 1e-12;
    c.scf.grad_conv = 1e-9;
    c.hcore =
        PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(SrColumnRotation::on());
    c.jk = KJkKind::RsGdf;
    c.rsgdf.gdf.sr_column_rotation = SrColumnRotation::on();
    let (r, kept) = solve_krhf_keep(&cell, &prep, Some(&auxp), &mesh, &c).expect("scf");
    let rotated = kept.hk.sr_rotated_columns;
    let g = kpoint_rhf_gradient(
        &cell,
        &prep,
        &mesh,
        &c.hcore,
        &kept.hk,
        kept.grad_jk(Some(&auxp), &c.rsgdf).unwrap(),
        &r,
        ExxDiv::Ewald,
        &KGradConfig::default(),
    )
    .expect("gradient");
    (g.grad.iter().copied().collect(), rotated)
}

#[test]
fn rotating_basis_rsgdf_k_point_force_equals_the_library_force() {
    let o = run_with(
        "rsgdf-rot",
        &xyz(&H2_OFFAXIS),
        "rhf",
        1,
        "[1, 1, 2]",
        LOOSE,
        "cc-pvdz",
        "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"",
    );
    let out = ok(&o, "rsgdf-rot");
    // The default column rotation is on (the optimizer prints no stage table,
    // so the rotated-column count is read from the library run below).
    assert!(out.contains("sr_column_rotation = on (default)"), "{out}");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let (lib, lib_rotated) = library_gradient_rsgdf_ccpvdz(&at);
    assert!(lib_rotated > 0);
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &lib);
    eprintln!("RS-GDF rotated CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}\n{g:?}\n{lib:?}");
}
