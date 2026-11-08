//! IEF-PCM analytic nuclear gradient threaded through the SCF gradient
//! dispatchers (`restricted_scf_gradient`, `unrestricted_scf_gradient`,
//! `restricted_open_scf_gradient`; `crates/ferric-pcm/theory.md` §8).
//!
//! * Central finite differences of ferric's own solvated SCF energy
//!   (RHF water, RHF C1 methanol, RKS-LDA water, UHF and ROHF NH2).
//! * ε → 1: the solvated gradient approaches the vacuum gradient.
//! * Translational invariance of the full solvated gradient, and the
//!   rotational torque against a finite-difference rigid rotation (the
//!   Lebedev frame is fixed in the lab, so the torque is not zero; it must
//!   be the derivative of the energy the code computes).
//! * PySCF `pcm.grad` on the same cavity (`rhf_pcm_gradient_matches_pyscf`).
//!
//! Physics vs artifact hypothesis: if the PCM term is right, analytic − FD
//! stays inside the step-derived bound; if the PCM term were missing (the
//! vacuum-form gradient of a solvated energy) every FD check misses by the
//! PCM gradient itself, 1e-3..1e-2 Ha/Bohr (asserted as a control in
//! `rhf_water_matches_fd`).
//!
//! FD tolerance: central differences at `h` and `2h` differ by three times
//! the `O(h²)` truncation error at `h`; the bar is that difference plus
//! `4·δE/h`, `δE` the SCF energy convergence (1e-11 Ha).
// RI-J pinned to the f64 tensor (`ri_j_precision: Some(F64)` in every config
// here): these tests validate f64 implementation code against finite
// differences, another tier or f64 reference values at bars below the f32
// default's effect (its energy shift is ~1e-6 Ha, and its energy surface jitters
// by ~1e-7..1e-6 Ha between nearby geometries; site/src/methods/scf.md, "RI-J
// tensor precision").

use std::path::PathBuf;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_pcm::{PcmConfig, PcmContext, ProbeKind};
use ferric_scf::gradient::{
    restricted_open_scf_gradient, restricted_scf_gradient, unrestricted_scf_gradient,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{solve_rohf, solve_uhf, ScfResult};
use ndarray::Array2;

const H: f64 = 1e-3;
const E_CONV: f64 = 1e-11;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Rhf,
    Uhf,
    Rohf,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn water() -> Molecule {
    Molecule::load_xyz(
        workspace_root()
            .join("testdata/molecules/water.xyz")
            .to_str()
            .unwrap(),
    )
    .unwrap()
}

fn methanol_c1() -> Molecule {
    let xyz = "6\nmethanol C1\n\
        C -0.046800  0.663000  0.000000\n\
        O -0.046800 -0.758000  0.000000\n\
        H -1.092600  0.969000  0.120000\n\
        H  0.438000  1.080000  0.890000\n\
        H  0.438000  1.080000 -0.890000\n\
        H  0.861000 -1.075000  0.050000\n";
    Molecule::parse_xyz(xyz, 0, 1).unwrap()
}

/// NH2 doublet off its symmetric geometry (a non-degenerate SOMO: OH's π
/// pair is split only by the Lebedev grid's anisotropy, so its ROHF SOMO
/// orientation flips between FD displacements).
fn nh2() -> Molecule {
    Molecule::parse_xyz(
        "3\nNH2\nN 0.0 0.0 0.0\nH 0.0 0.80 0.62\nH 0.05 -0.78 0.60\n",
        0,
        2,
    )
    .unwrap()
}

fn config(pcm: Option<PcmConfig>, xc: Option<&str>) -> RhfConfig {
    RhfConfig {
        ri_j_precision: Some(ferric_scf::df_j::RiJPrecision::F64),
        max_iter: 300,
        energy_conv: E_CONV,
        density_conv: 1e-9,
        pcm,
        xc: xc.map(str::to_string),
        ..Default::default()
    }
}

struct Run {
    energy: f64,
    grad: Array2<f64>,
}

fn solve(mol: &Molecule, cfg: &RhfConfig, kind: Kind) -> (ScfResult, PreparedBasis, SchwarzBounds) {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let r = match kind {
        Kind::Rhf => solve_rhf(&ctx, mol, &prep, op, &bounds, cfg),
        Kind::Uhf => solve_uhf(&ctx, mol, &prep, &bounds, cfg),
        Kind::Rohf => solve_rohf(&ctx, mol, &prep, op, &bounds, cfg),
    }
    .unwrap();
    assert!(r.converged, "SCF not converged");
    (r, prep, bounds)
}

fn energy(mol: &Molecule, cfg: &RhfConfig, kind: Kind) -> f64 {
    solve(mol, cfg, kind).0.energy
}

fn run(mol: &Molecule, cfg: &RhfConfig, kind: Kind) -> Run {
    let (r, prep, bounds) = solve(mol, cfg, kind);
    let bs = basis::bundled("sto-3g").unwrap();
    let op = Operator::coulomb();
    let grad = match kind {
        Kind::Rhf => restricted_scf_gradient(mol, &prep, &bs, op, &bounds, cfg, &r),
        Kind::Uhf => unrestricted_scf_gradient(mol, &prep, &bs, op, &bounds, cfg, &r),
        Kind::Rohf => restricted_open_scf_gradient(mol, &prep, &bs, op, &bounds, cfg, &r),
    }
    .unwrap();
    Run {
        energy: r.energy,
        grad,
    }
}

fn displaced(mol: &Molecule, a: usize, ax: usize, dx: f64) -> Molecule {
    let mut m = mol.clone();
    match ax {
        0 => m.atoms[a].x += dx,
        1 => m.atoms[a].y += dx,
        _ => m.atoms[a].zpos += dx,
    }
    m
}

fn central(mol: &Molecule, cfg: &RhfConfig, kind: Kind, a: usize, ax: usize, h: f64) -> f64 {
    (energy(&displaced(mol, a, ax, h), cfg, kind) - energy(&displaced(mol, a, ax, -h), cfg, kind))
        / (2.0 * h)
}

/// Analytic vs FD on every coordinate; returns (max |analytic − FD|, the
/// FD gradient).
fn check_fd(label: &str, mol: &Molecule, cfg: &RhfConfig, kind: Kind) -> (f64, Array2<f64>) {
    let g = run(mol, cfg, kind).grad;
    let mut fd = Array2::<f64>::zeros(g.dim());
    let mut worst = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for ax in 0..3 {
            let fd_h = central(mol, cfg, kind, a, ax, H);
            let fd_2h = central(mol, cfg, kind, a, ax, 2.0 * H);
            let tol = (fd_2h - fd_h).abs() + 4.0 * E_CONV / H;
            let err = (g[(a, ax)] - fd_h).abs();
            assert!(
                err < tol,
                "{label}: atom {a} axis {ax}: analytic {:.10e} FD {:.10e} |diff| {err:.3e} > tol {tol:.3e}",
                g[(a, ax)],
                fd_h
            );
            fd[(a, ax)] = fd_h;
            worst = worst.max(err);
        }
    }
    eprintln!("{label}: max |analytic - FD| = {worst:.3e} Ha/Bohr");
    (worst, fd)
}

fn max_abs(g: &Array2<f64>) -> f64 {
    g.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

#[test]
fn rhf_water_matches_fd() {
    let mol = water();
    let cfg = config(Some(PcmConfig::water()), None);
    let (_, fd) = check_fd("RHF water", &mol, &cfg, Kind::Rhf);
    // CONTROL: the vacuum-form gradient (PCM term dropped) evaluated on the
    // SAME solvated SCF misses FD by the size of the PCM term.
    let (r, prep, _) = solve(&mol, &cfg, Kind::Rhf);
    let g_pcm =
        ferric_pcm::pcm_gradient(&mol, &prep, r.density_total(), &PcmConfig::water()).unwrap();
    let g = run(&mol, &cfg, Kind::Rhf).grad;
    let miss = max_abs(&(&(&g - &g_pcm) - &fd));
    eprintln!(
        "RHF water: PCM term max {:.3e}, control miss {miss:.3e}",
        max_abs(&g_pcm)
    );
    assert!(
        miss > 1e-4,
        "control: dropping the PCM term must miss FD ({miss:e})"
    );
}

#[test]
fn rhf_methanol_c1_matches_fd() {
    let pcm = PcmConfig::water().with_lebedev_order(50).unwrap();
    check_fd(
        "RHF methanol C1",
        &methanol_c1(),
        &config(Some(pcm), None),
        Kind::Rhf,
    );
}

#[test]
fn rks_lda_water_gaussian_probe_matches_fd() {
    let pcm = PcmConfig {
        probe: ProbeKind::GaussianSmeared,
        ..PcmConfig::water().with_lebedev_order(50).unwrap()
    };
    check_fd(
        "RKS SVWN water",
        &water(),
        &config(Some(pcm), Some("SVWN")),
        Kind::Rhf,
    );
}

#[test]
fn uhf_and_rohf_nh2_match_fd() {
    let cfg = config(Some(PcmConfig::water()), None);
    check_fd("UHF NH2", &nh2(), &cfg, Kind::Uhf);
    check_fd("ROHF NH2", &nh2(), &cfg, Kind::Rohf);
}

/// ε → 1 (f = 1e-9): the solvated gradient is the vacuum gradient.
#[test]
fn eps_to_one_recovers_the_vacuum_gradient() {
    let mol = water();
    let vac = run(&mol, &config(None, None), Kind::Rhf).grad;
    let near = run(
        &mol,
        &config(Some(PcmConfig::with_epsilon(1.0 + 2e-9).unwrap()), None),
        Kind::Rhf,
    )
    .grad;
    let solv = run(&mol, &config(Some(PcmConfig::water()), None), Kind::Rhf).grad;
    let d_near = max_abs(&(&near - &vac));
    let d_solv = max_abs(&(&solv - &vac));
    eprintln!("eps->1: |g - g_vac| = {d_near:.3e}; eps=78.4: {d_solv:.3e}");
    assert!(d_solv > 1e-3);
    assert!(
        d_near < 1e-8,
        "ε→1 gradient differs from vacuum by {d_near:e}"
    );
}

/// Σ_A dE/dR_A = 0 with the cavity moving rigidly with the molecule.
#[test]
fn solvated_gradient_is_translationally_invariant() {
    let g = run(
        &methanol_c1(),
        &config(Some(PcmConfig::water()), None),
        Kind::Rhf,
    )
    .grad;
    for ax in 0..3 {
        let s: f64 = g.column(ax).sum();
        assert!(s.abs() < 1e-8, "axis {ax}: Σ_A g = {s:e}");
    }
}

/// The Lebedev directions are fixed in the lab frame, so a rigid rotation
/// changes the energy and the torque Σ_A (R_A × g_A) is not zero. It must
/// equal dE/dθ of a finite-difference rigid rotation.
#[test]
fn torque_matches_fd_rigid_rotation() {
    let mol = methanol_c1();
    let cfg = config(
        Some(PcmConfig::water().with_lebedev_order(50).unwrap()),
        None,
    );
    let g = run(&mol, &cfg, Kind::Rhf).grad;
    // Rotation about z through the origin: dE/dθ = Σ_A (x_A g_Ay − y_A g_Ax).
    let torque_z: f64 = mol
        .atoms
        .iter()
        .enumerate()
        .map(|(a, at)| at.x * g[(a, 1)] - at.y * g[(a, 0)])
        .sum();
    let rotated = |th: f64| {
        let mut m = mol.clone();
        for at in m.atoms.iter_mut() {
            let (x, y) = (at.x, at.y);
            at.x = x * th.cos() - y * th.sin();
            at.y = x * th.sin() + y * th.cos();
        }
        energy(&m, &cfg, Kind::Rhf)
    };
    let fd = |h: f64| (rotated(h) - rotated(-h)) / (2.0 * h);
    let (fd_h, fd_2h) = (fd(H), fd(2.0 * H));
    let tol = (fd_2h - fd_h).abs() + 4.0 * E_CONV / H;
    eprintln!("torque_z analytic {torque_z:.6e} FD {fd_h:.6e} tol {tol:.2e}");
    assert!(
        torque_z.abs() > 10.0 * tol,
        "torque too small to test ({torque_z:e})"
    );
    assert!((torque_z - fd_h).abs() < tol);
}

/// PySCF `scf.RHF(mol).PCM()` (IEF-PCM, 302 points/sphere, Gaussian
/// probes) `nuc_grad_method()`, from `scripts/validation/gen_pcm_grad.py`.
/// ferric builds the same cavity from the molecule with
/// `lebedev_order = 302`; the tessera count, total area and area centroid
/// are asserted first, then the energy, then the gradient.
#[test]
fn rhf_pcm_gradient_matches_pyscf() {
    for system in ["water", "ch3oh"] {
        let path = workspace_root().join(format!(
            "testdata/reference/validation/pcm/grad_{system}_sto-3g.json"
        ));
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let r: serde_json::Value = serde_json::from_str(&text).unwrap();
        let mol = Molecule::load_xyz(
            workspace_root()
                .join(format!("testdata/molecules/{system}.xyz"))
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let pcm = PcmConfig {
            probe: ProbeKind::GaussianSmeared,
            ..PcmConfig::water().with_lebedev_order(302).unwrap()
        };

        let ctx = PcmContext::new(&mol, &pcm).unwrap();
        let tess = ctx.tesserae();
        let cav = &r["cavity"];
        assert_eq!(
            tess.len() as u64,
            cav["n_tesserae"].as_u64().unwrap(),
            "{system}"
        );
        let area: f64 = tess.iter().map(|t| t.area).sum();
        let want_area = cav["total_area"].as_f64().unwrap();
        assert!(
            (area - want_area).abs() < 1e-10 * want_area,
            "{system}: area {area} vs {want_area}"
        );
        for ax in 0..3 {
            let c: f64 = tess.iter().map(|t| t.area * t.position[ax]).sum::<f64>() / area;
            let want = cav["area_centroid"][ax].as_f64().unwrap();
            assert!(
                (c - want).abs() < 1e-10,
                "{system}: centroid {ax}: {c} vs {want}"
            );
        }

        let cfg = config(Some(pcm), None);
        let got = run(&mol, &cfg, Kind::Rhf);
        let e_ref = r["energy"].as_f64().unwrap();
        assert!(
            (got.energy - e_ref).abs() < 1e-9,
            "{system}: E {} vs {e_ref}",
            got.energy
        );
        let mut worst = 0.0_f64;
        for (a, row) in r["gradient"].as_array().unwrap().iter().enumerate() {
            for ax in 0..3 {
                let want = row[ax].as_f64().unwrap();
                worst = worst.max((got.grad[(a, ax)] - want).abs());
            }
        }
        eprintln!("{system}: |ferric - PySCF| gradient max {worst:.3e} Ha/Bohr");
        assert!(
            worst < PYSCF_GRAD_TOL,
            "{system}: gradient misses PySCF by {worst:e}"
        );
    }
}

/// ferric vs PySCF PCM gradient bar (Ha/Bohr). Set from the measured
/// agreement (see the test's eprintln output and theory.md §8).
const PYSCF_GRAD_TOL: f64 = 1e-6;
