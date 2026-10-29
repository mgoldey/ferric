//! Shared harness of `periodic_open_rsh_derivs.rs` (per-commit) and
//! `validation_periodic_open_rsh_derivs.rs` (validation tier), issue #498:
//! the open-shell range-separated hybrid (RSH) derivatives and the Gamma RSH
//! derivatives on RS-GDF, wired to the CLI.
//!
//! [`cli`] runs the real `ferric-cli` binary and returns its run log's
//! `run_end` record (serde_json parses floats correctly rounded, #489). The
//! `lib_*` functions restate the CLI's config builders field by field
//! (`crate::periodic::{kuks,kroks,uks,roks,rks}_config`, `gamma_system`,
//! `rsh_sizing`, `KForceCtx::xc_cfg`, `KStressCtx`) and call the library on
//! the SAME cell, so the anchors can compare bit for bit.
//!
//! Fixtures: H2 (closed shell) and an H3 doublet in a 4 Bohr cubic cell,
//! STO-3G, or STO-3G plus one p shell (zeta 1.0) on every H ([`H_SP`], the
//! p-cell of #561, inlined here since that PR is not on main yet), exxdiv
//! ewald, omega 0.8, XC grid 20x50 with a 14 Bohr cutoff; `jk = "rsgdf"`
//! takes cc-pVDZ-RI at the default gdf_omega with the SR column rotation on
//! (the plan's choice for an RS-GDF run). Every run asserts the number of
//! basis functions on both sides (CLI printout, `PreparedBasis::nbasis`), so
//! a silent fall back to the s-only basis cannot pass.
#![allow(dead_code)]

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::units::BOHR_RADIUS_ANGSTROM;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION};
use ferric_pbc::gamma_rsh::{
    gamma_rks_any_keep, gamma_rks_rsh_gradient, gamma_rks_rsh_stress, gamma_roks_any,
    gamma_roks_any_keep, gamma_roks_rsh_gradient, gamma_uks_any, gamma_uks_any_keep,
    gamma_uks_rsh_gradient, gamma_uks_rsh_stress, GammaRshExchange, GammaRshSizing,
};
use ferric_pbc::kroks::{KRoScfResult, KRohfConfig, KRoksConfig};
use ferric_pbc::kstress::KStressConfig;
use ferric_pbc::kxcgrad::{KRksGradConfig, KXcGrid};
use ferric_pbc::rohf::GammaRoksConfig;
use ferric_pbc::rsgdf::DEFAULT_RSGDF_OMEGA;
use ferric_pbc::{
    kpoint_roks_gradient, kpoint_roks_stress, kpoint_uks_gradient, kpoint_uks_stress,
    periodic_hcore, solve_kroks_keep, solve_kuks_keep, Cell, DenseAftEri, ExxDiv, GammaRksConfig,
    GammaUhfIntegrals, GammaUksConfig, KDenseAftConfig, KGradConfig, KJkKind, KPointMesh,
    KRetained, KRksStressConfig, KRsGdfConfig, KScfConfig, KUksConfig, KUksGradConfig,
    KUksStressConfig, MeshCentring, PeriodicGridConfig, PeriodicHcoreConfig, RsGdf, RsGdfConfig,
    SrColumnRotation,
};
use ferric_scf::result::ScfResult;
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
pub const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";
pub const KMESH: [usize; 3] = [1, 1, 2];

/// H STO-3G (the bundled digits) plus one p shell (zeta 1.0): the cheapest
/// basis with l = 1 that keeps the dense-AFT G sphere of the H/STO-3G cells
/// (set by the tightest exponent, 3.425). The same text as #561's `H_SP`.
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

/// The SCF family of a run (`[method] kind` "rhf" / "uhf" / "rohf" with a
/// functional = RKS / UKS / ROKS).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Spin {
    Rks,
    Uks,
    Roks,
}

impl Spin {
    fn kind(self) -> &'static str {
        match self {
            Spin::Rks => "rhf",
            Spin::Uks => "uhf",
            Spin::Roks => "rohf",
        }
    }
    fn mult(self) -> usize {
        if self == Spin::Rks {
            1
        } else {
            2
        }
    }
    pub fn atoms(self) -> &'static [[f64; 3]] {
        if self == Spin::Rks {
            &H2
        } else {
            &H3
        }
    }
}

/// One CLI run.
#[derive(Clone, Copy)]
pub struct Run<'a> {
    pub tag: &'a str,
    pub spin: Spin,
    pub functional: &'a str,
    /// `false` = Gamma point, `true` = the 1x1x2 Gamma-centred [`KMESH`].
    pub kpoints: bool,
    pub rsgdf: bool,
    /// `"energy"`, `"optimize"` (`steps` optimizer steps; 0 = the gradient of
    /// the input geometry) or `"stress"` (energy + `[cell] stress = true`).
    pub task: &'a str,
    /// Atom positions in Bohr (`None` = [`Spin::atoms`]).
    pub atoms: Option<&'a [[f64; 3]]>,
    /// Uniaxial strain `eps_zz`: lattice c and every z scaled by `1 + eps`.
    pub strain_zz: f64,
    /// STO-3G plus a p shell ([`H_SP`]) instead of the bundled STO-3G.
    pub sp: bool,
    /// BFGS steps of an optimize run.
    pub steps: usize,
}

impl<'a> Run<'a> {
    pub fn new(tag: &'a str, spin: Spin, functional: &'a str, task: &'a str) -> Self {
        Run {
            tag,
            spin,
            functional,
            kpoints: true,
            rsgdf: false,
            task,
            atoms: None,
            strain_zz: 0.0,
            sp: false,
            steps: 0,
        }
    }
    fn atoms(&self) -> Vec<[f64; 3]> {
        let s = 1.0 + self.strain_zz;
        self.atoms
            .unwrap_or_else(|| self.spin.atoms())
            .iter()
            .map(|p| [p[0], p[1], p[2] * s])
            .collect()
    }
    /// The basis functions the cell must carry (the basis guard).
    pub fn nbasis(&self) -> usize {
        self.spin.atoms().len() * if self.sp { 4 } else { 1 }
    }
    /// The same run with another functional / J/K (a sibling).
    pub fn with(&self, functional: &'a str, rsgdf: bool) -> Self {
        Run {
            functional,
            rsgdf,
            ..*self
        }
    }
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

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

pub fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferric-b498g-{}-{tag}", std::process::id()));
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

pub const RSGDF_KEYS: &str = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";

/// The `[basis]` lines of `r` (the p-cell G94 text is written into `dir`).
fn basis_toml(r: &Run, dir: &Path) -> String {
    if !r.sp {
        return "name = \"sto-3g\"".into();
    }
    let p = dir.join("basis.g94");
    std::fs::write(&p, H_SP).unwrap();
    format!("path = \"{}\"", p.display())
}

/// Write the input (and its xyz) into a fresh scratch directory.
pub fn write(r: &Run) -> PathBuf {
    let dir = scratch(r.tag);
    let xyz_path = dir.join("m.xyz");
    std::fs::write(&xyz_path, xyz(&r.atoms())).unwrap();
    let (task, tail) = match r.task {
        "optimize" => (
            "optimize",
            format!(
                "[optimize]\nmax_steps = {}\ng_max_thresh = 1e-9\ng_rms_thresh = 1e-9\n\
                 e_conv = 1e-14",
                r.steps
            ),
        ),
        _ => ("energy", String::new()),
    };
    let kmesh = if r.kpoints {
        format!("kmesh = {KMESH:?}")
    } else {
        String::new()
    };
    let stress = if r.task == "stress" {
        "stress = true"
    } else {
        ""
    };
    let jk = if r.rsgdf { RSGDF_KEYS } else { "" };
    let c = A_BOHR * (1.0 + r.strain_zz);
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
lattice = [[{a:?}, 0.0, 0.0], [0.0, {a:?}, 0.0], [0.0, 0.0, {c:?}]]
{kmesh}
exxdiv = "ewald"
omega = {OMEGA}
n_radial = {N_RADIAL}
n_angular = {N_ANGULAR}
neighbour_cutoff = {CUTOFF_BOHR}
{jk}
{stress}
{tail}
"#,
            xyz = xyz_path.display(),
            mult = r.spin.mult(),
            basis = basis_toml(r, &dir),
            kind = r.spin.kind(),
            f = r.functional,
            json = dir.join("run.jsonl").display(),
            a = A_BOHR,
        ),
    )
    .unwrap();
    toml
}

fn exec(toml: &Path) -> Output {
    Command::new(ferric_cli_bin())
        .arg(toml)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("run ferric-cli")
}

/// Run `r` through the CLI; it must succeed and print the expected basis
/// size. Returns the `run_end` record.
pub fn cli(r: &Run) -> serde_json::Value {
    let toml = write(r);
    let o = exec(&toml);
    let stdout = String::from_utf8_lossy(&o.stdout);
    assert!(
        o.status.success(),
        "{}: ferric-cli failed\n{stdout}\n{}",
        r.tag,
        String::from_utf8_lossy(&o.stderr)
    );
    let want = format!("nbasis     = {}\n", r.nbasis());
    assert!(
        stdout.contains(&want),
        "{}: CLI basis guard ({want:?})\n{stdout}",
        r.tag
    );
    let log = std::fs::read_to_string(toml.with_file_name("run.jsonl")).expect("run log");
    let _ = std::fs::remove_dir_all(toml.parent().unwrap());
    for line in log.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if v.get("record").and_then(|r| r.as_str()) == Some("run_end") {
            return v;
        }
    }
    panic!("{}: no run_end record in:\n{log}", r.tag);
}

/// Run `r` through the CLI; it must be refused. Returns stderr.
pub fn refused(r: &Run) -> String {
    let toml = write(r);
    let o = exec(&toml);
    assert!(!o.status.success(), "{} must be refused", r.tag);
    let _ = std::fs::remove_dir_all(toml.parent().unwrap());
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

pub fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()))
}

/// `(E, geometry Bohr, dE/dR)` of a CLI optimize run: the returned geometry
/// and the gradient evaluated there.
pub fn cli_force(r: &Run) -> (f64, Vec<f64>, Vec<f64>) {
    assert_eq!(r.task, "optimize");
    let v = cli(r);
    let e = v["energy"].as_f64().expect("energy");
    let x = floats(&v["extra"]["final_geometry_bohr"]);
    let g = floats(&v["extra"]["final_gradient_hartree_per_bohr"]);
    assert_eq!(x.len(), g.len());
    (e, x, g)
}

/// `(E, sigma row-major)` of a CLI `[cell] stress = true` run.
pub fn cli_stress(r: &Run) -> (f64, Vec<f64>) {
    assert_eq!(r.task, "stress");
    let v = cli(r);
    let e = v["energy"].as_f64().expect("energy");
    let rows = v["extra"]["stress_hartree_per_bohr3"]
        .as_array()
        .unwrap_or_else(|| panic!("{}: no stress in {v}", r.tag));
    let s: Vec<f64> = rows.iter().flat_map(floats).collect();
    assert_eq!(s.len(), 9);
    (e, s)
}

/// The energy of a CLI `task = "energy"` run.
pub fn cli_energy(r: &Run) -> f64 {
    assert_eq!(r.task, "energy");
    cli(r)["energy"].as_f64().expect("energy")
}

// ───────────────────────────────────────────────────────────── library ──

/// The cell the CLI built for `r`: the same xyz round trip (and G94 basis
/// text), then (for a force run) the logged Bohr positions written into the
/// parsed molecule exactly as the CLI's optimizer does. Asserts the basis
/// guard.
pub fn cell_at(r: &Run, at: Option<&[f64]>) -> (Cell, PreparedBasis) {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = scratch(&format!("lib{n}"));
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(r.spin.atoms())).unwrap();
    let mut mol =
        Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, r.spin.mult()).expect("xyz");
    let bs: BasisSet = if r.sp {
        let b = dir.join("l.g94");
        std::fs::write(&b, H_SP).unwrap();
        basis::load_g94(b.to_str().unwrap()).unwrap()
    } else {
        basis::bundled("sto-3g").unwrap()
    };
    let _ = std::fs::remove_dir_all(&dir);
    if let Some(x) = at {
        for (k, a) in mol.atoms.iter_mut().enumerate() {
            (a.x, a.y, a.zpos) = (x[3 * k], x[3 * k + 1], x[3 * k + 2]);
        }
    }
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    assert_eq!(prep.nbasis(), r.nbasis(), "{}: library basis guard", r.tag);
    (cell, prep)
}

fn aux_of(cell: &Cell, rsgdf: bool) -> Option<PreparedBasis> {
    rsgdf.then(|| PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap())
}

pub fn grid() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(N_RADIAL, N_ANGULAR);
    g.neighbour_cutoff = Some(CUTOFF_BOHR);
    g
}

/// The plan's SR column rotation: on only with `jk = "rsgdf"`.
fn rotation(rsgdf: bool) -> SrColumnRotation {
    if rsgdf {
        SrColumnRotation::on()
    } else {
        SrColumnRotation::Off
    }
}

/// `crate::periodic::khcore_config` (and the Gamma `gamma_system` hcore).
pub fn hcore_cfg(rsgdf: bool) -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(rotation(rsgdf))
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

/// `crate::periodic::kdense_config` (`[cell] max_eri_gb` default 0.5).
fn kdense() -> KDenseAftConfig {
    KDenseAftConfig {
        max_bytes: ferric_core::memory::gib_to_bytes(0.5),
        ..Default::default()
    }
}

/// `crate::periodic::rsgdf_config` (no budget, no range split).
fn rsgdf_cfg(rsgdf: bool) -> RsGdfConfig {
    RsGdfConfig {
        omega: DEFAULT_RSGDF_OMEGA,
        sr_column_rotation: rotation(rsgdf),
        ..Default::default()
    }
}

fn krsgdf(rsgdf: bool) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: rsgdf_cfg(rsgdf),
        ..Default::default()
    }
}

fn jk_kind(rsgdf: bool) -> KJkKind {
    if rsgdf {
        KJkKind::RsGdf
    } else {
        KJkKind::Dense
    }
}

enum KOpenScf<'a> {
    Uks(&'a ferric_pbc::KUScfResult),
    Roks(&'a KRoScfResult),
}

struct KOpen<'a> {
    cell: &'a Cell,
    prep: &'a PreparedBasis,
    aux: Option<&'a PreparedBasis>,
    mesh: &'a KPointMesh,
    kept: &'a KRetained,
    rcfg: &'a KRsGdfConfig,
    hcfg: &'a PeriodicHcoreConfig,
    ao: f64,
    scf: KOpenScf<'a>,
    func: &'a str,
}

impl KOpen<'_> {
    fn energy(&self) -> f64 {
        match self.scf {
            KOpenScf::Uks(r) => r.energy,
            KOpenScf::Roks(r) => r.energy,
        }
    }
}

/// An open-shell k-point SCF (UKS or ROKS) through `solve_k*_keep` with the
/// CLI's configs, handed to `f` with its retained J/K.
fn with_kscf<T>(r: &Run, x: Option<&[f64]>, f: impl FnOnce(&KOpen<'_>) -> T) -> T {
    let (cell, prep) = cell_at(r, x);
    let mesh = KPointMesh::new(&cell, KMESH, MeshCentring::Gamma).unwrap();
    let (rsgdf, func) = (r.rsgdf, r.functional);
    let aux = aux_of(&cell, rsgdf);
    let open = |kept, rcfg, hcfg, ao, scf| KOpen {
        cell: &cell,
        prep: &prep,
        aux: aux.as_ref(),
        mesh: &mesh,
        kept,
        rcfg,
        hcfg,
        ao,
        scf,
        func,
    };
    match r.spin {
        Spin::Uks => {
            let mut c = KUksConfig::new(&cell, ExxDiv::Ewald, func);
            (c.kuhf.scf, c.kuhf.hcore, c.kuhf.dense, c.grid) =
                (kscf(), hcore_cfg(rsgdf), kdense(), grid());
            (c.kuhf.jk, c.kuhf.rsgdf) = (jk_kind(rsgdf), krsgdf(rsgdf));
            let (s, kept) = solve_kuks_keep(&cell, &prep, aux.as_ref(), &mesh, &c).expect("scf");
            let (ao, scf) = (c.xc.ao_threshold, KOpenScf::Uks(&s.scf));
            f(&open(&kept, &c.kuhf.rsgdf, &c.kuhf.hcore, ao, scf))
        }
        Spin::Roks => {
            let mut c = KRoksConfig::new(&cell, ExxDiv::Ewald, func);
            let mut ro = KRohfConfig::for_cell(&cell, ExxDiv::Ewald);
            (ro.scf, ro.hcore, ro.dense) = (kscf(), hcore_cfg(rsgdf), kdense());
            (ro.jk, ro.rsgdf) = (jk_kind(rsgdf), krsgdf(rsgdf));
            (c.rohf, c.grid) = (ro, grid());
            let (s, kept) = solve_kroks_keep(&cell, &prep, aux.as_ref(), &mesh, &c).expect("scf");
            let (ao, scf) = (c.xc.ao_threshold, KOpenScf::Roks(&s.scf));
            f(&open(&kept, &c.rohf.rsgdf, &c.rohf.hcore, ao, scf))
        }
        Spin::Rks => panic!("open-shell helper"),
    }
}

/// Library k-point UKS / ROKS `(E, dE/dR)` at `x` (`KForceCtx::xc_cfg`).
fn lib_k_force(r: &Run, x: Option<&[f64]>) -> (f64, Vec<f64>) {
    with_kscf(r, x, |k| {
        let gc = KUksGradConfig {
            base: KRksGradConfig {
                grad: KGradConfig {
                    budget_bytes: Some(k.kept.derivative_budget(None)),
                    ..Default::default()
                },
                grid: KXcGrid::Atom(grid()),
                ao_threshold: k.ao,
                exxdiv: Some(ExxDiv::Ewald),
                ..Default::default()
            },
            ..Default::default()
        };
        let jk = k.kept.grad_jk(k.aux, k.rcfg).unwrap();
        let (c, p, m, h, hk) = (k.cell, k.prep, k.mesh, k.hcfg, &k.kept.hk);
        let g = match k.scf {
            KOpenScf::Uks(s) => kpoint_uks_gradient(c, p, m, h, hk, jk, s, k.func, &gc),
            KOpenScf::Roks(s) => kpoint_roks_gradient(c, p, m, h, hk, jk, s, k.func, &gc),
        }
        .expect("force");
        (k.energy(), g.grad.iter().copied().collect())
    })
}

/// Library k-point UKS / ROKS `(E, sigma row-major)` at the fixture geometry
/// (`KStressCtx`'s stress settings).
fn lib_k_stress(r: &Run) -> (f64, Vec<f64>) {
    with_kscf(r, None, |k| {
        let sc = KUksStressConfig {
            base: KRksStressConfig {
                stress: KStressConfig {
                    budget_bytes: Some(k.kept.derivative_budget(None)),
                    ..Default::default()
                },
                grid: KXcGrid::Atom(grid()),
                ao_threshold: k.ao,
                exxdiv: Some(ExxDiv::Ewald),
                ..Default::default()
            },
            ..Default::default()
        };
        let jk = k.kept.stress_jk(k.aux, k.rcfg).unwrap();
        let (c, p, m, h, hk) = (k.cell, k.prep, k.mesh, k.hcfg, &k.kept.hk);
        let s = match k.scf {
            KOpenScf::Uks(s) => kpoint_uks_stress(c, p, m, h, hk, jk, s, k.func, &sc),
            KOpenScf::Roks(s) => kpoint_roks_stress(c, p, m, h, hk, jk, s, k.func, &sc),
        }
        .expect("stress");
        (k.energy(), s.sigma.iter().flatten().copied().collect())
    })
}

/// `crate::periodic::rsh_sizing` of a run without `[memory]` or a range split.
fn sizing(rsgdf: bool) -> GammaRshSizing {
    GammaRshSizing {
        rsgdf_omega: DEFAULT_RSGDF_OMEGA,
        budget_bytes: None,
        max_eri_bytes: if rsgdf {
            DEFAULT_DENSE_AFT_MAX_BYTES
        } else {
            ferric_core::memory::gib_to_bytes(0.5)
        },
        range_split: false,
    }
}

/// `crate::periodic::gamma_scf_config` (density_conv 1e-10, 200 iterations).
fn gamma_scf() -> RhfConfig {
    RhfConfig {
        use_sad_guess: false,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    }
}

/// The Gamma SCF of a range-separated hybrid on the CLI's `gamma_system(..,
/// for_gradient = true)` (dense AFT or `RsGdf::build_for_gradient`) through
/// `gamma_*_any_keep` with `rsh_sizing`, handed to `f` with its kept
/// exchange: `f(cell, prep, hcore config, SCF, configs, exchange)`.
enum GammaCfg {
    Rks(GammaRksConfig),
    Uks(GammaUksConfig),
    Roks(GammaRoksConfig),
}

fn with_gamma<T>(
    r: &Run,
    x: Option<&[f64]>,
    f: impl FnOnce(
        &Cell,
        &PreparedBasis,
        &PeriodicHcoreConfig,
        &ScfResult,
        &GammaCfg,
        &GammaRshExchange,
    ) -> T,
) -> T {
    let (cell, prep) = cell_at(r, x);
    let hcfg = hcore_cfg(r.rsgdf);
    let hc = periodic_hcore(&cell, &prep, &hcfg).unwrap();
    let aux = aux_of(&cell, r.rsgdf);
    let (dense, fit);
    let ints = match &aux {
        None => {
            let max = ferric_core::memory::gib_to_bytes(0.5);
            let prec = DEFAULT_DENSE_AFT_PRECISION;
            dense = DenseAftEri::build(&cell, &prep, &hc.s, ExxDiv::Ewald, prec, max).unwrap();
            GammaUhfIntegrals::DenseAft(&dense)
        }
        Some(a) => {
            let cfg = RsGdfConfig {
                exxdiv: ExxDiv::Ewald,
                ..rsgdf_cfg(true)
            };
            fit = RsGdf::build_for_gradient(&cell, &prep, a, &hc.s, &cfg).unwrap();
            GammaUhfIntegrals::RsGdf(&fit)
        }
    };
    let (sz, func) = (sizing(r.rsgdf), r.functional);
    let (cl, pr, hcr, aux) = (&cell, &prep, &hc, aux.as_ref());
    let keep = "a range-separated hybrid keeps its exchange";
    match r.spin {
        Spin::Rks => {
            let mut c = GammaRksConfig::new(func);
            (c.grid, c.exxdiv, c.scf) = (grid(), ExxDiv::Ewald, gamma_scf());
            let (s, ex) = gamma_rks_any_keep(cl, pr, hcr, ints, aux, sz, &c).expect("scf");
            f(cl, pr, &hcfg, &s.scf, &GammaCfg::Rks(c), &ex.expect(keep))
        }
        Spin::Uks => {
            let mut c = GammaUksConfig::new(func);
            (c.grid, c.exxdiv) = (grid(), ExxDiv::Ewald);
            c.ewald_start = ferric_pbc::EwaldStart::Staged;
            c.scf = gamma_scf();
            let (s, ex) = gamma_uks_any_keep(cl, pr, hcr, ints, aux, sz, &c).expect("scf");
            f(cl, pr, &hcfg, &s.scf, &GammaCfg::Uks(c), &ex.expect(keep))
        }
        Spin::Roks => {
            // `crate::periodic::roks_config` with `[scf]` unwritten: the
            // `GammaRoksConfig::new` level shift and iteration cap.
            let mut c = GammaRoksConfig::new(func);
            (c.grid, c.exxdiv) = (grid(), ExxDiv::Ewald);
            c.ewald_start = ferric_pbc::EwaldStart::Staged;
            c.scf = RhfConfig {
                level_shift: c.scf.level_shift,
                max_iter: c.scf.max_iter,
                ..gamma_scf()
            };
            let (s, ex) = gamma_roks_any_keep(cl, pr, hcr, ints, aux, sz, &c).expect("scf");
            f(cl, pr, &hcfg, &s.scf, &GammaCfg::Roks(c), &ex.expect(keep))
        }
    }
}

/// Library Gamma RKS / UKS / ROKS `(E, dE/dR)` of a range-separated hybrid
/// at `x`: the routed `gamma_*_rsh_gradient` on the kept exchange.
fn lib_gamma_force(r: &Run, x: Option<&[f64]>) -> (f64, Vec<f64>) {
    with_gamma(r, x, |cell, prep, hcfg, scf, cfg, ex| {
        let kg = KGradConfig::default();
        let g = match cfg {
            GammaCfg::Rks(c) => {
                gamma_rks_rsh_gradient(cell, prep, hcfg, scf, c, ex, &kg).map(|g| g.grad)
            }
            GammaCfg::Uks(c) => {
                gamma_uks_rsh_gradient(cell, prep, hcfg, scf, c, ex, &kg).map(|g| g.grad)
            }
            GammaCfg::Roks(c) => {
                gamma_roks_rsh_gradient(cell, prep, hcfg, scf, c, ex, &kg).map(|g| g.grad)
            }
        };
        (scf.energy, g.expect("force").iter().copied().collect())
    })
}

/// Library Gamma RKS / UKS `(E, sigma row-major)` of a range-separated
/// hybrid at the fixture geometry (`gamma_*_rsh_stress` on the kept
/// exchange). The CLI has no Gamma stress: this is the library side of the
/// Python `run_rks_gamma` / `run_uks_gamma(with_stress=True)` pins (the
/// binding's Gamma settings are the CLI's ones restated here).
fn lib_gamma_stress(r: &Run) -> (f64, Vec<f64>) {
    with_gamma(r, None, |cell, prep, hcfg, scf, cfg, ex| {
        let sc = KStressConfig::default();
        let s = match cfg {
            GammaCfg::Rks(c) => {
                gamma_rks_rsh_stress(cell, prep, hcfg, scf, c, ex, &sc).map(|s| s.sigma)
            }
            GammaCfg::Uks(c) => {
                gamma_uks_rsh_stress(cell, prep, hcfg, scf, c, ex, &sc).map(|s| s.sigma)
            }
            GammaCfg::Roks(_) => panic!("no Gamma ROKS range-separated stress"),
        };
        (
            scf.energy,
            s.expect("stress").iter().flatten().copied().collect(),
        )
    })
}

/// Library force of `r` (k-point or Gamma) at `x` (`None` = the fixture
/// geometry as the CLI parses it).
pub fn lib_force(r: &Run, x: Option<&[f64]>) -> (f64, Vec<f64>) {
    if r.kpoints {
        lib_k_force(r, x)
    } else {
        lib_gamma_force(r, x)
    }
}

/// Library stress of `r` (k-point or Gamma) at the fixture geometry.
pub fn lib_stress(r: &Run) -> (f64, Vec<f64>) {
    if r.kpoints {
        lib_k_stress(r)
    } else {
        lib_gamma_stress(r)
    }
}

/// CLI force `(E, x, g)` equals the library at `x` bit for bit.
pub fn assert_force_bit_equal(tag: &str, cli: &(f64, Vec<f64>, Vec<f64>), lib: (f64, Vec<f64>)) {
    let (e, _, g) = cli;
    let (el, gl) = lib;
    let gmax = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    eprintln!("{tag}: E {e:?} max|g| {gmax:.3e}; library E {el:?}");
    eprintln!("{tag}: x {:?}", cli.1);
    eprintln!("{tag}: g {g:?}");
    assert!(gmax > 1e-3, "{tag}: trivial force");
    assert_eq!(e.to_bits(), el.to_bits(), "{tag}: energy {e} vs {el}");
    assert_eq!(bits(g), bits(&gl), "{tag}: force {g:?} vs {gl:?}");
}

/// CLI stress `(E, sigma)` equals the library bit for bit.
pub fn assert_stress_bit_equal(tag: &str, cli: &(f64, Vec<f64>), lib: (f64, Vec<f64>)) {
    let (e, s) = cli;
    let (el, sl) = lib;
    eprintln!("{tag}: E {e:?} library E {el:?}");
    eprintln!("{tag}: sigma {s:?}");
    assert!(s.iter().any(|v| v.abs() > 1e-6), "{tag}: zero stress");
    assert!((s[0] - s[8]).abs() > 1e-7, "{tag}: xx == zz, untestable");
    assert_eq!(e.to_bits(), el.to_bits(), "{tag}: energy {e} vs {el}");
    assert_eq!(bits(s), bits(&sl), "{tag}: stress {s:?} vs {sl:?}");
}

/// Run `r` (optimize) through the CLI and assert it equals the library bit
/// for bit; returns the CLI `(E, x, g)`.
pub fn force_anchor(r: &Run) -> (f64, Vec<f64>, Vec<f64>) {
    let c = cli_force(r);
    assert_force_bit_equal(r.tag, &c, lib_force(r, Some(&c.1)));
    c
}

/// Run `r` (stress) through the CLI and assert it equals the library bit
/// for bit; returns the CLI `(E, sigma)`.
pub fn stress_anchor(r: &Run) -> (f64, Vec<f64>) {
    let c = cli_stress(r);
    assert_stress_bit_equal(r.tag, &c, lib_stress(r));
    c
}

/// Library Gamma UKS / ROKS energy of an energy run on the dense-AFT J/K
/// (`crate::periodic::{uks,roks}_config` with `[scf]` unwritten, through the
/// shared `gamma_*_any` entries the CLI calls).
fn lib_gamma_open_energy(r: &Run) -> f64 {
    assert!(
        !r.rsgdf,
        "{}: the Gamma open-shell energy helper is dense only",
        r.tag
    );
    let (cell, prep) = cell_at(r, None);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg(false)).unwrap();
    let (max, prec) = (
        ferric_core::memory::gib_to_bytes(0.5),
        DEFAULT_DENSE_AFT_PRECISION,
    );
    let eri = DenseAftEri::build(&cell, &prep, &hc.s, ExxDiv::Ewald, prec, max).unwrap();
    let ints = GammaUhfIntegrals::DenseAft(&eri);
    let (sz, func) = (sizing(false), r.functional);
    match r.spin {
        Spin::Uks => {
            let mut c = GammaUksConfig::new(func);
            (c.grid, c.exxdiv) = (grid(), ExxDiv::Ewald);
            c.ewald_start = ferric_pbc::EwaldStart::Staged;
            c.scf = gamma_scf();
            gamma_uks_any(&cell, &prep, &hc, ints, None, sz, &c)
                .expect("scf")
                .scf
                .energy
        }
        Spin::Roks => {
            let mut c = GammaRoksConfig::new(func);
            (c.grid, c.exxdiv) = (grid(), ExxDiv::Ewald);
            c.ewald_start = ferric_pbc::EwaldStart::Staged;
            c.scf = RhfConfig {
                level_shift: c.scf.level_shift,
                max_iter: c.scf.max_iter,
                ..gamma_scf()
            };
            gamma_roks_any(&cell, &prep, &hc, ints, None, sz, &c)
                .expect("scf")
                .scf
                .energy
        }
        Spin::Rks => panic!("open-shell helper"),
    }
}

/// Library energy of an open-shell energy run `r`: the k-point
/// `solve_k*_keep` with the CLI's configs, or the Gamma `gamma_*_any`.
pub fn lib_energy(r: &Run) -> f64 {
    if r.kpoints {
        with_kscf(r, None, |k| k.energy())
    } else {
        lib_gamma_open_energy(r)
    }
}

/// Run `r` (energy) through the CLI and assert it equals the library bit for
/// bit; returns the energy.
pub fn energy_anchor(r: &Run) -> f64 {
    let e = cli_energy(r);
    let el = lib_energy(r);
    eprintln!("{}: CLI E {e:?} library E {el:?}", r.tag);
    assert_eq!(e.to_bits(), el.to_bits(), "{}: energy {e} vs {el}", r.tag);
    e
}
