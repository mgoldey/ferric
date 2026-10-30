//! Shared harness of `periodic_kpoint_open_shell.rs` (per commit) and
//! `validation_periodic_kpoint_open_shell.rs` (weekly): the k-point
//! open-shell (UKS / ROKS / ROHF) CLI runs on the H3 doublet and the library
//! energies / forces restated with the CLI's configs.
#![allow(dead_code, unused_imports)]

pub use ferric_core::basis;
pub use ferric_core::mol::Molecule;
pub use ferric_core::units::BOHR_RADIUS_ANGSTROM;
pub use ferric_integrals::basis_bridge::PreparedBasis;
pub use ferric_pbc::dense_aft::DEFAULT_DENSE_AFT_PRECISION;
pub use ferric_pbc::gamma_rsh::{gamma_uks_any, GammaRshSizing};
pub use ferric_pbc::kroks::{KRohfConfig, KRoksConfig};
pub use ferric_pbc::kxcgrad::{KRksGradConfig, KXcGrid};
pub use ferric_pbc::rsgdf::DEFAULT_RSGDF_OMEGA;
pub use ferric_pbc::{
    kpoint_rohf_gradient, kpoint_roks_gradient, kpoint_uks_gradient, periodic_hcore,
    solve_krohf_keep, solve_kroks_keep, solve_kuks, solve_kuks_keep, Cell, DenseAftEri, ExxDiv,
    GammaUhfIntegrals, GammaUksConfig, KDenseAftConfig, KGradConfig, KJkKind, KPointMesh,
    KScfConfig, KUksConfig, KUksGradConfig, MeshCentring, PeriodicGridConfig, PeriodicHcoreConfig,
    SrColumnRotation,
};
pub use ferric_scf::rhf::RhfConfig;
pub use std::path::{Path, PathBuf};
pub use std::process::{Command, Output};

pub const A_BOHR: f64 = 4.0;
pub const OMEGA: f64 = 0.8;
pub const CUTOFF_BOHR: f64 = 14.0;
pub const N_RADIAL: usize = 20;
pub const N_ANGULAR: usize = 50;
/// Printed geometry has 8 Angstrom decimals; the force moves by ~k*1e-8 over it.
pub const BAR: f64 = 5e-8;
pub const E_BAR: f64 = 5e-10;
pub const LOOSE: &str = "g_max_thresh = 1.0\ng_rms_thresh = 1.0\ne_conv = 1.0\nmax_steps = 3";
pub const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Method {
    Uks,
    Roks,
    Rohf,
}

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
    let dir = std::env::temp_dir().join(format!("ferric-kopen-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn xyz(atoms_bohr: &[[f64; 3]]) -> String {
    let b = BOHR_RADIUS_ANGSTROM;
    let mut s = format!("{}\nk-point open-shell test (Angstrom)\n", atoms_bohr.len());
    for p in atoms_bohr {
        s += &format!("H {:.12} {:.12} {:.12}\n", p[0] * b, p[1] * b, p[2] * b);
    }
    s
}

pub struct Input<'a> {
    pub tag: &'a str,
    pub method: Method,
    pub functional: &'a str,
    pub task: &'a str,
    pub jk: &'a str,
    pub kmesh: &'a str,
    pub optimize: &'a str,
}

pub fn run(i: &Input) -> Output {
    let dir = scratch(i.tag);
    let xyz_path = dir.join("m.xyz");
    std::fs::write(&xyz_path, xyz(&H3)).unwrap();
    let (kind, dft) = match i.method {
        Method::Uks => ("uhf", format!("[dft]\nfunctional = \"{}\"", i.functional)),
        Method::Roks => ("rohf", format!("[dft]\nfunctional = \"{}\"", i.functional)),
        Method::Rohf => ("rohf", String::new()),
    };
    let toml = dir.join("in.toml");
    std::fs::write(
        &toml,
        format!(
            r#"[molecule]
xyz = "{xyz}"
multiplicity = 2
[basis]
name = "sto-3g"
[method]
kind = "{kind}"
task = "{task}"
{dft}
[cell]
unit = "bohr"
lattice = [[{a}, 0.0, 0.0], [0.0, {a}, 0.0], [0.0, 0.0, {a}]]
{kmesh}
exxdiv = "ewald"
omega = {OMEGA}
{grid}
{jk}
{optimize}
"#,
            xyz = xyz_path.display(),
            task = i.task,
            grid = if i.method == Method::Rohf {
                String::new()
            } else {
                format!(
                    "n_radial = {N_RADIAL}\nn_angular = {N_ANGULAR}\nneighbour_cutoff = {CUTOFF_BOHR}"
                )
            },
            // An empty `kmesh` is a Gamma-point run.
            kmesh = if i.kmesh.is_empty() {
                String::new()
            } else {
                format!("kmesh = {}", i.kmesh)
            },
            jk = i.jk,
            optimize = if i.task == "optimize" {
                format!("[optimize]\n{}", i.optimize)
            } else {
                format!("[output]\njson = \"{}\"", dir.join("run.jsonl").display())
            },
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

/// What the library force is built from (the CLI's own configs, restated).
#[derive(Clone, Copy)]
pub struct Lib<'a> {
    pub method: Method,
    pub functional: &'a str,
    pub mesh: [usize; 3],
    pub centring: MeshCentring,
}

pub fn grid() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(N_RADIAL, N_ANGULAR);
    g.neighbour_cutoff = Some(CUTOFF_BOHR);
    g
}

pub fn rohf_cfg(cell: &Cell) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, ExxDiv::Ewald);
    c.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.scf.max_iter = 200;
    c.scf.energy_conv = 1e-12;
    c.scf.grad_conv = 1e-9;
    c
}

/// Library `(energy, dE/dR)` at `atoms_bohr`; `exxdiv_force` goes to the force.
pub fn library(atoms_bohr: &[[f64; 3]], lib: Lib, exxdiv_force: ExxDiv) -> (f64, Vec<f64>) {
    let dir = scratch("lib");
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(atoms_bohr)).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, 2).expect("xyz");
    let bs = basis::bundled("sto-3g").unwrap();
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let mesh = KPointMesh::new(&cell, lib.mesh, lib.centring).unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let xc = |ao: f64| KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig::default(),
            grid: KXcGrid::Atom(grid()),
            ao_threshold: ao,
            exxdiv: Some(exxdiv_force),
            ..Default::default()
        },
        ..Default::default()
    };
    match lib.method {
        Method::Uks => {
            let mut c = KUksConfig::new(&cell, ExxDiv::Ewald, lib.functional);
            c.kuhf.hcore = hcfg;
            c.kuhf.scf.max_iter = 200;
            c.grid = grid();
            let (r, kept) = solve_kuks_keep(&cell, &prep, None, &mesh, &c).expect("scf");
            let g = kpoint_uks_gradient(
                &cell,
                &prep,
                &mesh,
                &c.kuhf.hcore,
                &kept.hk,
                kept.grad_jk(None, &c.kuhf.rsgdf).unwrap(),
                &r.scf,
                lib.functional,
                &xc(c.xc.ao_threshold),
            )
            .expect("force");
            (r.scf.energy, g.grad.iter().copied().collect())
        }
        Method::Roks => {
            let mut c = KRoksConfig::new(&cell, ExxDiv::Ewald, lib.functional);
            c.rohf = rohf_cfg(&cell);
            c.grid = grid();
            let (r, kept) = solve_kroks_keep(&cell, &prep, None, &mesh, &c).expect("scf");
            let g = kpoint_roks_gradient(
                &cell,
                &prep,
                &mesh,
                &c.rohf.hcore,
                &kept.hk,
                kept.grad_jk(None, &c.rohf.rsgdf).unwrap(),
                &r.scf,
                lib.functional,
                &xc(c.xc.ao_threshold),
            )
            .expect("force");
            (r.scf.energy, g.grad.iter().copied().collect())
        }
        Method::Rohf => {
            let c = rohf_cfg(&cell);
            let (r, kept) = solve_krohf_keep(&cell, &prep, None, &mesh, &c).expect("scf");
            let g = kpoint_rohf_gradient(
                &cell,
                &prep,
                &mesh,
                &c.hcore,
                &kept.hk,
                kept.grad_jk(None, &c.rsgdf).unwrap(),
                &r.scf,
                exxdiv_force,
                &xc(1e-12),
            )
            .expect("force");
            (r.scf.energy, g.grad.iter().copied().collect())
        }
    }
}

pub fn input<'a>(tag: &'a str, method: Method, functional: &'a str, task: &'a str) -> Input<'a> {
    Input {
        tag,
        method,
        functional,
        task,
        jk: "jk = \"dense\"",
        kmesh: "[1, 1, 2]",
        optimize: LOOSE,
    }
}

pub fn force_anchor(
    tag: &str,
    method: Method,
    functional: &str,
    label: &str,
) -> (Vec<f64>, Vec<[f64; 3]>) {
    let out = ok(&run(&input(tag, method, functional, "optimize")), tag);
    assert!(
        out.contains(&format!("analytic k-point {label} gradient")),
        "{out}"
    );
    let g = printed_gradient(&out, 3);
    let at = printed_atoms(&out, 3);
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial force"
    );
    let lib = Lib {
        method,
        functional,
        mesh: [1, 1, 2],
        centring: MeshCentring::Gamma,
    };
    let (_, gl) = library(&at, lib, ExxDiv::Ewald);
    let d = max_diff(&g, &gl);
    eprintln!("{tag}: CLI vs library force {d:.2e}");
    assert!(d < BAR, "{tag}: {d:e}");
    (g, at)
}

/// The CLI energy of the H3 doublet on the half-shifted Monkhorst-Pack 1x1x2
/// mesh (no Gamma point) equals the library energy of `method` on the same
/// mesh (bar [`E_BAR`]); MUTANT: the Gamma-centred mesh of the same size is a
/// different calculation, and the anchor must tell the two apart.
pub fn mp_mesh_energy_anchor(tag: &str, method: Method, f: &str) {
    let mut i = input(tag, method, f, "energy");
    i.kmesh = "[1, 1, 2]\ncentring = \"mp\"";
    let out = ok(&run(&i), tag);
    assert_eq!(value(&out, "converged"), "true", "{out}");
    let e: f64 = value(&out, "energy").parse().unwrap();
    let lib = |centring| Lib {
        method,
        functional: f,
        mesh: [1, 1, 2],
        centring,
    };
    let (el, _) = library(&H3, lib(MeshCentring::MonkhorstPack), ExxDiv::Ewald);
    assert!((e - el).abs() < E_BAR, "{tag}: {e} vs {el}");
    let (eg, _) = library(&H3, lib(MeshCentring::Gamma), ExxDiv::Ewald);
    assert!((e - eg).abs() > 1e-6, "{tag}: centring mutant not detected");
}
