//! k-point open-shell routes at `[cell] kmesh`: UKS (`uhf` / `ksdft` with a
//! functional), ROKS (`rohf` + `[dft] functional`) and ROHF, energies and
//! `task = "optimize"` forces, wired to `solve_kuks`/`solve_kroks`/
//! `solve_krohf` and `kpoint_{uks,roks,rohf}_gradient`.
//!
//! * EXACTNESS ANCHOR (CLI vs library): the printed total energy equals the
//!   library SCF energy of the same cell, mesh, grid and configs (bar 5e-10),
//!   and the printed final gradient of a loose optimize run (stopped after
//!   three steps) equals the library force at the printed 8-decimal final
//!   geometry (bar 5e-8). Both are anchored on an H3 doublet in a 4 Bohr cubic
//!   cell (1x1x2 Gamma-centred mesh, STO-3G, dense AFT J/K, exxdiv ewald).
//! * Mutants of the wiring itself are in the report of the commit that adds
//!   this file; the in-test mutants assert that the library force of a
//!   DIFFERENT method, mesh or functional misses the printed force by far more
//!   than the bar, i.e. the anchors could tell a mis-routed run apart.
//! * Open-shell SCAN and HSE06 ENERGIES (issue #357): the CLI energy, read
//!   at full precision from the JSON run log, equals the library `solve_kuks`
//!   (Gamma-centred and Monkhorst-Pack 1x1x2) and, with no kmesh, the
//!   library `gamma_uks_any` to 1e-12 (the run log holds exactly the library
//!   value; see [`SAME_ARITHMETIC`]); the Gamma HSE06 run also equals the CLI
//!   1x1x1 run to SCF precision.
//!   Mutation ledger (run 2026-10-18, then reverted): the CLI Gamma UKS with
//!   `exxdiv = none` forced into its config: KILLED (CLI -1.3319 vs library
//!   -1.5513); the CLI Gamma UKS with a direct Ewald start and density_conv
//!   1e-6: SURVIVED (moved E by 2.2e-16; this 3-orbital cell converges to the
//!   same state either way, an equivalent mutant here). In-test mutants:
//!   the sibling global functional misses by >= 2.3e-4 (HSE06 vs PBE0) /
//!   2.5e-2 (SCAN vs PBE), the other centring by >= 1.7e-2.
//! * Refusals with exact wording: ROKS meta-GGA, open-shell HSE06 forces, UKS
//!   meta-GGA on RS-GDF and at Gamma.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::units::BOHR_RADIUS_ANGSTROM;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::DEFAULT_DENSE_AFT_PRECISION;
use ferric_pbc::gamma_rsh::{gamma_uks_any, GammaRshSizing};
use ferric_pbc::kroks::{KRohfConfig, KRoksConfig};
use ferric_pbc::kxcgrad::{KRksGradConfig, KXcGrid};
use ferric_pbc::rsgdf::DEFAULT_RSGDF_OMEGA;
use ferric_pbc::{
    kpoint_rohf_gradient, kpoint_roks_gradient, kpoint_uks_gradient, periodic_hcore,
    solve_krohf_keep, solve_kroks_keep, solve_kuks, solve_kuks_keep, Cell, DenseAftEri, ExxDiv,
    GammaUhfIntegrals, GammaUksConfig, KDenseAftConfig, KGradConfig, KJkKind, KPointMesh,
    KScfConfig, KUksConfig, KUksGradConfig, MeshCentring, PeriodicGridConfig, PeriodicHcoreConfig,
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
/// Printed geometry has 8 Angstrom decimals; the force moves by ~k*1e-8 over it.
const BAR: f64 = 5e-8;
const E_BAR: f64 = 5e-10;
const LOOSE: &str = "g_max_thresh = 1.0\ng_rms_thresh = 1.0\ne_conv = 1.0\nmax_steps = 3";
const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Method {
    Uks,
    Roks,
    Rohf,
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

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferric-kopen-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn xyz(atoms_bohr: &[[f64; 3]]) -> String {
    let b = BOHR_RADIUS_ANGSTROM;
    let mut s = format!("{}\nk-point open-shell test (Angstrom)\n", atoms_bohr.len());
    for p in atoms_bohr {
        s += &format!("H {:.12} {:.12} {:.12}\n", p[0] * b, p[1] * b, p[2] * b);
    }
    s
}

struct Input<'a> {
    tag: &'a str,
    method: Method,
    functional: &'a str,
    task: &'a str,
    jk: &'a str,
    kmesh: &'a str,
    optimize: &'a str,
}

fn run(i: &Input) -> Output {
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

/// What the library force is built from (the CLI's own configs, restated).
#[derive(Clone, Copy)]
struct Lib<'a> {
    method: Method,
    functional: &'a str,
    mesh: [usize; 3],
    centring: MeshCentring,
}

fn grid() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(N_RADIAL, N_ANGULAR);
    g.neighbour_cutoff = Some(CUTOFF_BOHR);
    g
}

fn rohf_cfg(cell: &Cell) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, ExxDiv::Ewald);
    c.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.scf.max_iter = 200;
    c.scf.energy_conv = 1e-12;
    c.scf.grad_conv = 1e-9;
    c
}

/// Library `(energy, dE/dR)` at `atoms_bohr`; `exxdiv_force` goes to the force.
fn library(atoms_bohr: &[[f64; 3]], lib: Lib, exxdiv_force: ExxDiv) -> (f64, Vec<f64>) {
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

fn input<'a>(tag: &'a str, method: Method, functional: &'a str, task: &'a str) -> Input<'a> {
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

fn force_anchor(
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

#[test]
fn printed_force_equals_the_library_force_for_uks_with_mutants() {
    let (g, at) = force_anchor("uks", Method::Uks, "PBE", "UKS");
    let lib = Lib {
        method: Method::Uks,
        functional: "PBE",
        mesh: [1, 1, 2],
        centring: MeshCentring::Gamma,
    };
    // MUTANT: a different method (the ROKS force of the same functional) is
    // not the UKS force; the anchor must be able to tell them apart.
    let (_, roks) = library(
        &at,
        Lib {
            method: Method::Roks,
            ..lib
        },
        ExxDiv::Ewald,
    );
    assert!(max_diff(&g, &roks) > 1e-5, "method mutant not detected");
    // MUTANT: wrong mesh, wrong functional.
    let (_, bad) = library(
        &at,
        Lib {
            mesh: [1, 1, 1],
            ..lib
        },
        ExxDiv::Ewald,
    );
    assert!(max_diff(&g, &bad) > 1e-4, "mesh mutant not detected");
    let (_, bad) = library(
        &at,
        Lib {
            functional: "LDA",
            ..lib
        },
        ExxDiv::Ewald,
    );
    assert!(max_diff(&g, &bad) > 1e-4, "functional mutant not detected");
}

#[test]
fn printed_force_equals_the_library_force_for_roks() {
    force_anchor("roks", Method::Roks, "PBE", "ROKS");
}

#[test]
fn printed_force_equals_the_library_force_for_rohf() {
    force_anchor("rohf", Method::Rohf, "", "ROHF");
}

#[test]
fn printed_force_equals_the_library_force_for_a_uks_global_hybrid() {
    force_anchor("uks-pbe0", Method::Uks, "PBE0", "UKS");
}

#[test]
fn printed_energy_equals_the_library_energy_for_every_open_shell_route() {
    for (tag, method, f, key) in [
        ("e-uks", Method::Uks, "PBE", "E_xc"),
        ("e-roks", Method::Roks, "PBE", "E_xc"),
        ("e-rohf", Method::Rohf, "", "<S^2>"),
    ] {
        let out = ok(&run(&input(tag, method, f, "energy")), tag);
        assert_eq!(value(&out, "converged"), "true", "{out}");
        let e: f64 = value(&out, "energy").parse().unwrap();
        let (el, _) = library(
            &H3,
            Lib {
                method,
                functional: f,
                mesh: [1, 1, 2],
                centring: MeshCentring::Gamma,
            },
            ExxDiv::Ewald,
        );
        eprintln!(
            "{tag}: CLI energy {e:.10} vs library {el:.10}: {:.2e}",
            (e - el).abs()
        );
        assert!((e - el).abs() < E_BAR, "{tag}: {e} vs {el}");
        // The route's own printout (KS routes print E_xc, ROHF does not).
        assert!(out.contains(key), "{tag}: {out}");
        assert_eq!(out.contains("E_xc"), method != Method::Rohf, "{tag}: {out}");
        assert!(
            out.contains("<S^2>") && out.contains("gaps"),
            "{tag}: {out}"
        );
    }
}

/// A Monkhorst-Pack mesh (half-shifted at an even size, so no Gamma point)
/// reaches every open-shell driver: the CLI energy is the library energy on
/// the same mesh, and differs from the Gamma-centred mesh of the same size.
#[test]
fn printed_energy_equals_the_library_energy_on_a_monkhorst_pack_mesh() {
    for (tag, method, f) in [
        ("mp-uks", Method::Uks, "PBE"),
        ("mp-roks", Method::Roks, "PBE"),
        ("mp-rohf", Method::Rohf, ""),
    ] {
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
        // MUTANT: the Gamma-centred mesh of the same size is a different
        // calculation; the anchor must tell the two apart.
        let (eg, _) = library(&H3, lib(MeshCentring::Gamma), ExxDiv::Ewald);
        assert!((e - eg).abs() > 1e-6, "{tag}: centring mutant not detected");
    }
}

/// `run_end.energy` from the JSON run log of a `task = "energy"` input, as
/// the exact f64 (the stdout line has 10 decimals).
fn logged_energy(tag: &str) -> f64 {
    let log = std::fs::read_to_string(scratch(tag).join("run.jsonl")).expect("run log");
    for line in log.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if v.get("record").and_then(|r| r.as_str()) == Some("run_end") {
            return v["energy"]
                .as_f64()
                .unwrap_or_else(|| panic!("no energy in {line}"));
        }
    }
    panic!("no run_end record in:\n{log}");
}

/// The H3 doublet cell of [`library`] (same xyz round trip as the CLI).
fn h3_cell() -> (Cell, PreparedBasis) {
    // One file per call: the tests of this binary run in parallel.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let p = scratch("lib-e").join(format!("l{n}.xyz"));
    std::fs::write(&p, xyz(&H3)).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, 2).expect("xyz");
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &basis::bundled("sto-3g").unwrap()).unwrap();
    (cell, prep)
}

/// The CLI's dense-AFT cap (`[cell] max_eri_gb` default 0.5).
fn cli_max_eri_bytes() -> usize {
    ferric_core::memory::gib_to_bytes(0.5)
}

/// Library k-point UKS energy on the dense-AFT J/K with the CLI's configs
/// restated field by field (`crate::periodic::kuks_config`), so the two runs
/// are the same arithmetic.
fn library_kuks_energy(functional: &str, mesh: [usize; 3], centring: MeshCentring) -> f64 {
    let (cell, prep) = h3_cell();
    let mesh = KPointMesh::new(&cell, mesh, centring).unwrap();
    let mut c = KUksConfig::new(&cell, ExxDiv::Ewald, functional);
    c.kuhf.scf = KScfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        grad_conv: 1e-9,
        ..Default::default()
    };
    c.kuhf.hcore =
        PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(SrColumnRotation::Off);
    c.kuhf.dense = KDenseAftConfig {
        max_bytes: cli_max_eri_bytes(),
        ..Default::default()
    };
    c.grid = grid();
    solve_kuks(&cell, &prep, None, &mesh, &c)
        .unwrap_or_else(|e| panic!("solve_kuks {functional}: {e}"))
        .scf
        .energy
}

/// Library Gamma UKS energy (dense AFT) through the bindings' shared entry
/// `gamma_uks_any`, with the CLI's Gamma configs restated.
fn library_gamma_uks_energy(functional: &str) -> f64 {
    let (cell, prep) = h3_cell();
    let hcfg = PeriodicHcoreConfig {
        sr_column_rotation: SrColumnRotation::Off,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    };
    let hc = periodic_hcore(&cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        DEFAULT_DENSE_AFT_PRECISION,
        cli_max_eri_bytes(),
    )
    .unwrap();
    let mut c = GammaUksConfig::new(functional);
    (c.grid, c.exxdiv) = (grid(), ExxDiv::Ewald);
    c.scf = RhfConfig {
        use_sad_guess: false,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    };
    let sizing = GammaRshSizing {
        rsgdf_omega: DEFAULT_RSGDF_OMEGA,
        budget_bytes: None,
        max_eri_bytes: cli_max_eri_bytes(),
        range_split: false,
    };
    gamma_uks_any(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        None,
        sizing,
        &c,
    )
    .unwrap_or_else(|e| panic!("gamma_uks_any {functional}: {e}"))
    .scf
    .energy
}

fn energy_input<'a>(tag: &'a str, functional: &'a str, kmesh: &'a str) -> Input<'a> {
    Input {
        kmesh,
        ..input(tag, Method::Uks, functional, "energy")
    }
}

/// CLI vs library bar for the same arithmetic in two processes. The run log
/// holds exactly the library value (k-point HSE06: -1.6224626688847699 in the
/// file on every run); the test reads it back through serde_json, which is
/// built without `float_roundtrip` and parses that literal 1 ulp off
/// (-1.62246266888477, deterministically). `==` would fail on that parse
/// artifact, so the bar is 1e-12, far below every mutant (>= 2.3e-4).
const SAME_ARITHMETIC: f64 = 1e-12;

/// Open-shell SCAN and HSE06 (issue #357; library PRs #479 / #480): the CLI
/// UKS energy on a k-mesh (Gamma-centred 1x1x2 and Monkhorst-Pack 1x1x2) is
/// the library `solve_kuks` energy (read at full precision from the JSON run
/// log) to [`SAME_ARITHMETIC`]. MUTANTS the anchor must tell apart: the global
/// functional of the same family (PBE for SCAN, PBE0 for HSE06) and the other
/// mesh centring.
#[test]
fn cli_open_shell_scan_and_hse06_equal_library_solve_kuks() {
    for (f, sibling) in [("SCAN", "PBE"), ("HSE06", "PBE0")] {
        for (kmesh, centring, ctag) in [
            ("[1, 1, 2]", MeshCentring::Gamma, "g"),
            (
                "[1, 1, 2]\ncentring = \"mp\"",
                MeshCentring::MonkhorstPack,
                "mp",
            ),
        ] {
            let tag = format!("bits-{f}-{ctag}");
            let out = ok(&run(&energy_input(&tag, f, kmesh)), &tag);
            assert_eq!(value(&out, "converged"), "true", "{out}");
            let cli = logged_energy(&tag);
            let lib = library_kuks_energy(f, [1, 1, 2], centring);
            eprintln!("{tag}: CLI {cli:?} library {lib:?}");
            assert!(
                (cli - lib).abs() <= SAME_ARITHMETIC,
                "{tag}: {cli:?} vs {lib:?}"
            );
            let other = match centring {
                MeshCentring::Gamma => MeshCentring::MonkhorstPack,
                MeshCentring::MonkhorstPack => MeshCentring::Gamma,
            };
            let d_mesh = (cli - library_kuks_energy(f, [1, 1, 2], other)).abs();
            let d_f = (cli - library_kuks_energy(sibling, [1, 1, 2], centring)).abs();
            eprintln!("{tag}: centring mutant {d_mesh:.2e}, {sibling} mutant {d_f:.2e}");
            assert!(
                d_mesh > 1e-6 && d_f > 1e-4,
                "{tag}: a mutant was not detected"
            );
        }
    }
}

/// Gamma-point UKS HSE06 (no kmesh): the CLI energy is the library
/// `gamma_uks_any` (Gamma J/XC/SCF + the k = 0 attenuated K) to
/// [`SAME_ARITHMETIC`] (measured 0), and
/// agrees with the CLI's 1x1x1 k-point UKS (a different J, XC and SCF code
/// over the same cell) to SCF precision. MUTANT: PBE0 on the same cell.
#[test]
fn cli_gamma_uks_hse06_equals_library_and_the_one_point_mesh() {
    let out = ok(&run(&energy_input("gam-hse", "HSE06", "")), "gam-hse");
    assert_eq!(value(&out, "converged"), "true", "{out}");
    let cli = logged_energy("gam-hse");
    let lib = library_gamma_uks_energy("HSE06");
    eprintln!("gam-hse: CLI {cli:?} library {lib:?}");
    assert!((cli - lib).abs() <= SAME_ARITHMETIC, "{cli:?} vs {lib:?}");
    ok(
        &run(&energy_input("k111-hse", "HSE06", "[1, 1, 1]")),
        "k111-hse",
    );
    let k111 = logged_energy("k111-hse");
    eprintln!("Gamma - 1x1x1: {:.2e}", cli - k111);
    assert!((cli - k111).abs() < 1e-8, "{cli:.12} vs {k111:.12}");
    let pbe0 = library_gamma_uks_energy("PBE0");
    assert!((cli - pbe0).abs() > 1e-4, "PBE0 mutant not detected");
}

/// HSE06 on the RS-GDF J/K at a k-mesh: the CLI energy is the library
/// `solve_kuks` with the same aux, within the SR-column-rotation screening
/// difference of the CLI's fit (bar 1e-8, as the closed-shell RS-GDF anchor),
/// and differs from the dense-AFT energy by the fit error only.
#[test]
fn cli_open_shell_hse06_rsgdf_equals_library_solve_kuks() {
    let mut i = energy_input("rsgdf-hse", "HSE06", "[1, 1, 2]");
    i.jk = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";
    ok(&run(&i), "rsgdf-hse");
    let cli = logged_energy("rsgdf-hse");
    let (cell, prep) = h3_cell();
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mesh = KPointMesh::new(&cell, [1, 1, 2], MeshCentring::Gamma).unwrap();
    let mut c = KUksConfig::new(&cell, ExxDiv::Ewald, "HSE06");
    c.kuhf.jk = KJkKind::RsGdf;
    c.kuhf.scf.max_iter = 200;
    c.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.grid = grid();
    let lib = solve_kuks(&cell, &prep, Some(&aux), &mesh, &c)
        .unwrap()
        .scf
        .energy;
    let dense = library_kuks_energy("HSE06", [1, 1, 2], MeshCentring::Gamma);
    eprintln!(
        "rsgdf-hse: CLI - library {:.2e}, CLI - dense {:.2e}",
        cli - lib,
        cli - dense
    );
    assert!((cli - lib).abs() < 1e-8, "{cli:.12} vs {lib:.12}");
    assert!((cli - dense).abs() < 1e-3, "{cli:.12} vs dense {dense:.12}");
}

#[test]
fn open_shell_refusals_have_exact_messages() {
    // k-point ROKS has no meta-GGA (the UKS route has it; the ROKS
    // range-separated hybrid energy runs: periodic_rsh_open_mgga_derivs.rs).
    let tag = "roks-SCAN";
    let e = refused(&run(&input(tag, Method::Roks, "SCAN", "energy")), tag);
    assert!(
        e.contains("[cell] kmesh with functional \"SCAN\"")
            && e.contains("no meta-GGA for ROKS; the UKS route"),
        "{tag}: {e}"
    );
    // ROKS HSE06 has an energy here but no force: optimize is refused by the
    // derivative allowlist, before any integral (the UKS SCAN and HSE06
    // forces run: periodic_rsh_open_mgga_derivs.rs).
    let tag = "roks-opt-HSE06";
    let e = refused(&run(&input(tag, Method::Roks, "HSE06", "optimize")), tag);
    assert!(
        e.contains("method.task = \"optimize\" with [cell] kmesh")
            && e.contains("ROKS range-separated hybrid force and stress are not implemented"),
        "{tag}: {e}"
    );
    // Open-shell meta-GGA needs the dense-AFT J/K.
    let mut i = input("uks-scan-rsgdf", Method::Uks, "SCAN", "energy");
    i.jk = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";
    let e = refused(&run(&i), "uks-scan-rsgdf");
    assert!(e.contains("dense-AFT J/K only"), "{e}");
    // Gamma-point UKS meta-GGA: no tau term at Gamma; the 1x1x1 mesh has it.
    let e = refused(&run(&energy_input("gam-scan", "SCAN", "")), "gam-scan");
    assert!(
        e.contains("Gamma point") && e.contains("kmesh = [1, 1, 1]"),
        "{e}"
    );
}
