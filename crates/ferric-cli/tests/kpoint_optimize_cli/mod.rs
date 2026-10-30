//! Shared harness of `periodic_kpoint_optimize.rs` (per commit) and
//! `validation_periodic_kpoint_optimize.rs` (weekly): the k-point HF
//! `task = "optimize"` CLI runs and the printed-output parsers.
#![allow(dead_code, unused_imports)]

pub use ferric_core::basis;
pub use ferric_core::mol::Molecule;
pub use ferric_core::units::BOHR_RADIUS_ANGSTROM;
pub use ferric_integrals::basis_bridge::PreparedBasis;
pub use ferric_pbc::kroks::KRohfConfig;
pub use ferric_pbc::kxcgrad::KRksGradConfig;
pub use ferric_pbc::{
    kpoint_rhf_gradient, kpoint_rohf_gradient, kpoint_uhf_gradient, solve_krhf_keep,
    solve_krohf_keep, solve_kuhf_keep, Cell, ExxDiv, KGradConfig, KJkKind, KPointMesh, KRhfConfig,
    KUhfConfig, KUksGradConfig, MeshCentring, PeriodicHcoreConfig, SrColumnRotation,
};
pub use std::path::{Path, PathBuf};
pub use std::process::{Command, Output};

pub const A_BOHR: f64 = 5.0;
pub const OMEGA: f64 = 0.8;

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
    let dir = std::env::temp_dir().join(format!("ferric-kopt-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs an optimize input (STO-3G, dense J/K) and returns the process output.
pub fn run(tag: &str, xyz: &str, kind: &str, mult: usize, kmesh: &str, optimize: &str) -> Output {
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
pub fn run_with(
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

pub fn ok(o: &Output, tag: &str) -> String {
    let out = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(
        o.status.success(),
        "{tag}: ferric-cli failed\n{out}\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    out
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
    let mut s = format!("{}\nk-point optimize test (Angstrom)\n", atoms_bohr.len());
    for p in atoms_bohr {
        s += &format!("H {:.12} {:.12} {:.12}\n", p[0] * b, p[1] * b, p[2] * b);
    }
    s
}

/// Library force at `atoms_bohr` with the configs the CLI builds for this input.
pub fn library_gradient(
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
pub const BAR: f64 = 5e-7;
pub const LOOSE: &str = "g_max_thresh = 1.0\ng_rms_thresh = 1.0\ne_conv = 1.0\nmax_steps = 3";
pub const H2_OFFAXIS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
pub const H3_OFFAXIS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];

pub fn printed_gradient(stdout: &str, n: usize) -> Vec<f64> {
    rows(stdout, "final gradient", n)
        .into_iter()
        .flatten()
        .collect()
}

/// The printed final geometry (Angstrom, 8 decimals) in Bohr.
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
