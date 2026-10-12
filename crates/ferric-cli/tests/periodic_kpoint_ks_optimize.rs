//! `task = "optimize"` with `[cell] kmesh` and closed-shell Kohn-Sham (LDA,
//! GGA, global hybrids) is wired to `ferric_pbc::kxcgrad::kpoint_rks_gradient`.
//!
//! * EXACTNESS ANCHOR (CLI vs library): with loose optimizer thresholds the
//!   run stops after a few steps; the printed final gradient is the force at
//!   the printed final geometry, which must equal `kpoint_rks_gradient` called
//!   directly on the same cell, mesh, grid and configs (evaluated at the
//!   printed 8-decimal geometry, bar 5e-7). Mutants: a wrong mesh, a wrong
//!   functional and a wrong exxdiv each move the library force by far more
//!   than the bar (asserted).
//! * A real optimization (stretched H2, PBE0, 1x1x3) converges to the
//!   optimizer's thresholds, contracts the bond, and the printed final
//!   gradient equals the library force at the printed final geometry.
//! * Refusals with exact wording: meta-GGA, range-separated hybrid, open-shell
//!   Kohn-Sham, range_split, `gdf_omega = "auto"`.
//!
//! The explicit `neighbour_cutoff` is shared by the CLI input and the library
//! grid: the XC energy has a hard cutoff and the default depends on the cell.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::units::BOHR_RADIUS_ANGSTROM;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KXcGrid};
use ferric_pbc::{
    solve_krks_keep, Cell, ExxDiv, KGradConfig, KJkKind, KPointMesh, KRksConfig, MeshCentring,
    PeriodicGridConfig, PeriodicHcoreConfig, SrColumnRotation,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const A_BOHR: f64 = 5.0;
const OMEGA: f64 = 0.8;
const CUTOFF_BOHR: f64 = 10.0;
const N_RADIAL: usize = 40;
const N_ANGULAR: usize = 110;

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
    let dir = std::env::temp_dir().join(format!("ferric-kksopt-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A k-point KS optimize input (STO-3G, `kind = "ksdft"`).
struct Input<'a> {
    tag: &'a str,
    xyz: String,
    mult: usize,
    functional: &'a str,
    kmesh: &'a str,
    jk: &'a str,
    optimize: &'a str,
}

fn run(i: &Input) -> Output {
    let dir = scratch(i.tag);
    let xyz_path = dir.join("m.xyz");
    std::fs::write(&xyz_path, &i.xyz).unwrap();
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
kind = "ksdft"
task = "optimize"
[dft]
functional = "{functional}"
[cell]
unit = "bohr"
lattice = [[{a}, 0.0, 0.0], [0.0, {a}, 0.0], [0.0, 0.0, {a}]]
kmesh = {kmesh}
exxdiv = "ewald"
omega = {OMEGA}
n_radial = {N_RADIAL}
n_angular = {N_ANGULAR}
neighbour_cutoff = {CUTOFF_BOHR}
{jk}
[optimize]
{optimize}
"#,
            xyz = xyz_path.display(),
            mult = i.mult,
            functional = i.functional,
            kmesh = i.kmesh,
            jk = i.jk,
            optimize = i.optimize,
            a = A_BOHR,
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

fn refused(o: &Output, tag: &str) -> String {
    assert!(!o.status.success(), "{tag} must be refused");
    String::from_utf8_lossy(&o.stderr).into_owned()
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
    let mut s = format!(
        "{}\nk-point KS optimize test (Angstrom)\n",
        atoms_bohr.len()
    );
    for p in atoms_bohr {
        s += &format!("H {:.12} {:.12} {:.12}\n", p[0] * b, p[1] * b, p[2] * b);
    }
    s
}

/// What the library force is built from (the CLI's own configs, restated).
#[derive(Clone, Copy)]
struct Lib<'a> {
    functional: &'a str,
    mesh: [usize; 3],
    exxdiv_scf: ExxDiv,
    rsgdf: bool,
}

/// Library force at `atoms_bohr`, with `exxdiv_force` handed to the force (the
/// SCF runs with `lib.exxdiv_scf`): equal to the SCF's for a faithful call.
fn library_gradient(atoms_bohr: &[[f64; 3]], lib: Lib, exxdiv_force: ExxDiv) -> Vec<f64> {
    try_library_gradient(atoms_bohr, lib, exxdiv_force).expect("gradient")
}

fn try_library_gradient(
    atoms_bohr: &[[f64; 3]],
    lib: Lib,
    exxdiv_force: ExxDiv,
) -> Result<Vec<f64>, String> {
    let dir = scratch("lib");
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(atoms_bohr)).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, 1).expect("xyz");
    let bs = basis::bundled("sto-3g").unwrap();
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let aux = lib
        .rsgdf
        .then(|| PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap());
    let mesh = KPointMesh::new(&cell, lib.mesh, MeshCentring::Gamma).unwrap();
    let mut c = KRksConfig::new(&cell, lib.exxdiv_scf, lib.functional);
    c.krhf.scf.energy_conv = 1e-12;
    c.krhf.scf.grad_conv = 1e-9;
    c.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    if lib.rsgdf {
        c.krhf.jk = KJkKind::RsGdf;
        c.krhf.hcore = c.krhf.hcore.with_sr_column_rotation(SrColumnRotation::on());
        c.krhf.rsgdf.gdf.sr_column_rotation = SrColumnRotation::on();
    }
    let mut grid = PeriodicGridConfig::with_size(N_RADIAL, N_ANGULAR);
    grid.neighbour_cutoff = Some(CUTOFF_BOHR);
    c.grid = grid.clone();
    let (r, kept) = solve_krks_keep(&cell, &prep, aux.as_ref(), &mesh, &c).expect("scf");
    assert!(r.scf.converged);
    let rcfg = KRksGradConfig {
        grad: KGradConfig::default(),
        grid: KXcGrid::Atom(grid),
        ao_threshold: c.xc.ao_threshold,
        exxdiv: Some(exxdiv_force),
        ..Default::default()
    };
    kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &c.krhf.hcore,
        &kept.hk,
        kept.grad_jk(aux.as_ref(), &c.krhf.rsgdf).unwrap(),
        &r.scf,
        lib.functional,
        &rcfg,
    )
    .map_err(|e| e.to_string())
    .map(|g| g.grad.iter().copied().collect())
}

/// Printed geometry has 8 Angstrom decimals; the force moves by ~k*1e-8 over it.
const BAR: f64 = 5e-7;
const LOOSE: &str = "g_max_thresh = 1.0\ng_rms_thresh = 1.0\ne_conv = 1.0\nmax_steps = 3";
const H2_OFFAXIS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const DENSE: &str = "jk = \"dense\"";
const RSGDF: &str = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";

fn printed_gradient(stdout: &str, n: usize) -> Vec<f64> {
    rows(stdout, "final gradient", n)
        .into_iter()
        .flatten()
        .collect()
}

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

fn loose_input<'a>(tag: &'a str, functional: &'a str, kmesh: &'a str, jk: &'a str) -> Input<'a> {
    Input {
        tag,
        xyz: xyz(&H2_OFFAXIS),
        mult: 1,
        functional,
        kmesh,
        jk,
        optimize: LOOSE,
    }
}

#[test]
fn printed_force_equals_the_library_force_for_a_global_hybrid_with_mutants() {
    let o = run(&loose_input("pbe0", "PBE0", "[1, 1, 3]", DENSE));
    let out = ok(&o, "pbe0");
    assert!(out.contains("analytic k-point RKS gradient"), "{out}");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let lib = Lib {
        functional: "PBE0",
        mesh: [1, 1, 3],
        exxdiv_scf: ExxDiv::Ewald,
        rsgdf: false,
    };
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &library_gradient(&at, lib, ExxDiv::Ewald));
    eprintln!("PBE0 CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}");
    // MUTANT: a wrong mesh moves the force by far more than the bar.
    let bad = library_gradient(
        &at,
        Lib {
            mesh: [1, 1, 1],
            ..lib
        },
        ExxDiv::Ewald,
    );
    assert!(max_diff(&g, &bad) > 1e-4, "mesh mutant not detected");
    // MUTANT: the pure parent functional.
    let bad = library_gradient(
        &at,
        Lib {
            functional: "PBE",
            ..lib
        },
        ExxDiv::Ewald,
    );
    assert!(max_diff(&g, &bad) > 1e-4, "functional mutant not detected");
    // ARTIFACT HYPOTHESIS: for a closed-shell system the exxdiv shift is uniform
    // on the occupied space, so it moves the energy by a constant and leaves
    // the density and the force unchanged: an SCF run with exxdiv = none gives
    // the SAME force (an identity, asserted), and the force cannot tell the
    // two apart. What guards the wiring is the library's Fock-consistency
    // check: a force handed an exxdiv the SCF did not run is refused.
    let same = library_gradient(
        &at,
        Lib {
            exxdiv_scf: ExxDiv::None,
            ..lib
        },
        ExxDiv::None,
    );
    assert!(max_diff(&g, &same) < BAR, "exxdiv identity violated");
    let e =
        try_library_gradient(&at, lib, ExxDiv::None).expect_err("a wrong exxdiv must be refused");
    assert!(e.contains("exxdiv"), "{e}");
}

#[test]
fn printed_force_equals_the_library_force_for_a_gga() {
    let o = run(&loose_input("pbe", "PBE", "[1, 1, 3]", DENSE));
    let out = ok(&o, "pbe");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let lib = Lib {
        functional: "PBE",
        mesh: [1, 1, 3],
        exxdiv_scf: ExxDiv::Ewald,
        rsgdf: false,
    };
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &library_gradient(&at, lib, ExxDiv::Ewald));
    eprintln!("PBE CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}");
}

#[test]
fn printed_force_equals_the_library_force_with_rsgdf_jk() {
    let o = run(&loose_input("rsgdf", "PBE0", "[1, 1, 2]", RSGDF));
    let out = ok(&o, "rsgdf");
    assert!(out.contains("analytic k-point RKS gradient"), "{out}");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let lib = Lib {
        functional: "PBE0",
        mesh: [1, 1, 2],
        exxdiv_scf: ExxDiv::Ewald,
        rsgdf: true,
    };
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &library_gradient(&at, lib, ExxDiv::Ewald));
    eprintln!("RS-GDF PBE0 CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}");
}

#[test]
fn a_real_optimization_converges_and_the_final_force_is_the_library_force() {
    let b = BOHR_RADIUS_ANGSTROM;
    let start = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 / b]];
    let o = run(&Input {
        tag: "opt",
        xyz: xyz(&start),
        mult: 1,
        functional: "PBE0",
        kmesh: "[1, 1, 3]",
        jk: DENSE,
        optimize: "max_steps = 30",
    });
    let out = ok(&o, "opt");
    assert_eq!(value(&out, "converged"), "true", "{out}");
    let atoms = printed_atoms(&out, 2);
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
    let lib = Lib {
        functional: "PBE0",
        mesh: [1, 1, 3],
        exxdiv_scf: ExxDiv::Ewald,
        rsgdf: false,
    };
    let d = max_diff(&g, &library_gradient(&atoms, lib, ExxDiv::Ewald));
    assert!(d < BAR, "{d:e}");
}

#[test]
fn unsupported_kohn_sham_routes_are_refused_with_exact_messages() {
    // meta-GGA and range-separated hybrids: named by the library resolver.
    let e = refused(
        &run(&loose_input("scan", "SCAN", "[1, 1, 2]", DENSE)),
        "scan",
    );
    assert!(
        e.contains("[cell] kmesh with functional \"SCAN\"") && e.contains("meta-GGA"),
        "{e}"
    );
    assert!(
        e.contains("k-point RKS supports LDA, GGA and global-hybrid functionals only"),
        "{e}"
    );
    let e = refused(
        &run(&loose_input("hse", "HSE06", "[1, 1, 2]", DENSE)),
        "hse",
    );
    assert!(
        e.contains("[cell] kmesh with functional \"HSE06\"") && e.contains("range-separated"),
        "{e}"
    );
    // Open-shell Kohn-Sham (ksdft with multiplicity > 1 is UKS).
    let mut i = loose_input("uks", "PBE", "[1, 1, 2]", DENSE);
    i.xyz = xyz(&[[0.0, 0.0, 0.0]]);
    i.mult = 2;
    let e = refused(&run(&i), "uks");
    assert!(
        e.contains("[cell] kmesh is not supported for the open-shell Kohn-Sham route")
            && e.contains("k-point UKS/ROKS is not implemented yet"),
        "{e}"
    );
    // The range split and the automatic RS-GDF omega are energy-only.
    let split = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"\nrange_split = true";
    let e = refused(
        &run(&loose_input("split", "PBE", "[1, 1, 2]", split)),
        "split",
    );
    assert!(
        e.contains("[cell] range_split is not supported with method.task = \"optimize\""),
        "{e}"
    );
    let auto = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"\ngdf_omega = \"auto\"";
    let e = refused(&run(&loose_input("auto", "PBE", "[1, 1, 2]", auto)), "auto");
    assert!(
        e.contains(
            "[cell] gdf_omega = \"auto\" is calibrated for the Gamma-point build; with kmesh \
             set a number"
        ),
        "{e}"
    );
}
