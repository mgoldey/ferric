//! Shared harness of `periodic_kpoint_ks_optimize.rs` (per commit) and
//! `validation_periodic_kpoint_ks_optimize.rs` (weekly): the k-point
//! closed-shell Kohn-Sham `task = "optimize"` CLI runs and the library force
//! (`kpoint_rks_gradient`) restated with the CLI's configs.
#![allow(dead_code, unused_imports)]

pub use ferric_core::basis;
pub use ferric_core::mol::Molecule;
pub use ferric_core::units::BOHR_RADIUS_ANGSTROM;
pub use ferric_integrals::basis_bridge::PreparedBasis;
pub use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KXcGrid};
pub use ferric_pbc::{
    solve_krks_keep, Cell, ExxDiv, KGradConfig, KJkKind, KPointMesh, KRksConfig, MeshCentring,
    PeriodicGridConfig, PeriodicHcoreConfig, SrColumnRotation,
};
pub use std::path::{Path, PathBuf};
pub use std::process::{Command, Output};

pub const A_BOHR: f64 = 5.0;
pub const OMEGA: f64 = 0.8;
/// Explicit, NOT the default (max(10, covering bound) Bohr): a mutant below drops it and
/// moves the force by ~3e-6 (>> BAR), and 10 Bohr would also cross a ~9e-11 Ha
/// one-step energy discontinuity under a finite displacement.
pub const CUTOFF_BOHR: f64 = 14.0;
pub const N_RADIAL: usize = 40;
pub const N_ANGULAR: usize = 110;

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
    let dir = std::env::temp_dir().join(format!("ferric-kksopt-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A k-point KS optimize input (STO-3G, `kind = "ksdft"`).
pub struct Input<'a> {
    pub tag: &'a str,
    pub xyz: String,
    pub mult: usize,
    pub basis: &'a str,
    pub functional: &'a str,
    pub kmesh: &'a str,
    pub jk: &'a str,
    pub optimize: &'a str,
}

pub fn run(i: &Input) -> Output {
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
name = "{basis}"
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
            basis = i.basis,
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

pub fn ok(o: &Output, tag: &str) -> String {
    let out = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(
        o.status.success(),
        "{tag}: ferric-cli failed\n{out}\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    out
}

pub fn refused(o: &Output, tag: &str) -> String {
    assert!(!o.status.success(), "{tag} must be refused");
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// The `n` rows `SYM x y z` after the line starting with `header`.
pub fn rows(stdout: &str, header: &str, n: usize) -> Vec<[f64; 3]> {
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

pub fn value(stdout: &str, key: &str) -> String {
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

pub fn xyz(atoms_bohr: &[[f64; 3]]) -> String {
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
pub struct Lib<'a> {
    pub functional: &'a str,
    pub mesh: [usize; 3],
    pub exxdiv_scf: ExxDiv,
    pub rsgdf: bool,
    /// Hand the explicit neighbour cutoff to the grid (false = the default).
    pub cutoff: bool,
}

/// Library force at `atoms_bohr`, with `exxdiv_force` handed to the force (the
/// SCF runs with `lib.exxdiv_scf`): equal to the SCF's for a faithful call.
pub fn library_gradient(atoms_bohr: &[[f64; 3]], lib: Lib, exxdiv_force: ExxDiv) -> Vec<f64> {
    try_library_gradient(atoms_bohr, lib, exxdiv_force).expect("gradient")
}

pub fn try_library_gradient(
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
    grid.neighbour_cutoff = lib.cutoff.then_some(CUTOFF_BOHR);
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
pub const BAR: f64 = 5e-8;
pub const LOOSE: &str = "g_max_thresh = 1.0\ng_rms_thresh = 1.0\ne_conv = 1.0\nmax_steps = 3";
pub const H2_OFFAXIS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
pub const DENSE: &str = "jk = \"dense\"";
pub const RSGDF: &str = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";
pub const ECP_RSGDF: &str = "jk = \"rsgdf\"\nauxbasis = \"def2-svp-rifit\"";

pub fn printed_gradient(stdout: &str, n: usize) -> Vec<f64> {
    rows(stdout, "final gradient", n)
        .into_iter()
        .flatten()
        .collect()
}

pub fn printed_atoms(stdout: &str, n: usize) -> Vec<[f64; 3]> {
    let b = BOHR_RADIUS_ANGSTROM;
    rows(stdout, "final geometry", n)
        .into_iter()
        .map(|r| [r[0] / b, r[1] / b, r[2] / b])
        .collect()
}

pub fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

pub fn loose_input<'a>(
    tag: &'a str,
    functional: &'a str,
    kmesh: &'a str,
    jk: &'a str,
) -> Input<'a> {
    Input {
        tag,
        xyz: xyz(&H2_OFFAXIS),
        mult: 1,
        basis: "sto-3g",
        functional,
        kmesh,
        jk,
        optimize: LOOSE,
    }
}
