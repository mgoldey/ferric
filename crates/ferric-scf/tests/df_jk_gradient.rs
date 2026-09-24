//! Analytic gradients of density-fitted SCF energies (RI-J, RI-K, RSH).
//!
//! Before `ferric_scf::df_gradient` every gradient entry point differentiated
//! the EXACT four-centre J/K even when the SCF energy was density-fitted —
//! RI-J is the default for every KS-DFT run and RSH exchange is always fitted.
//! MEASURED pre-fix (ferric's own central FD, h = 1e-3 Bohr, a pre-fix
//! release build of the Python bindings, `ferric.run_qmmm` with an empty MM
//! region — which runs the same solve + gradient calls):
//!
//! | system (water)                    | max |analytic − FD| |
//! |-----------------------------------|---------------------|
//! | symmetric, cc-pVDZ, RHF (exact)   | 1.4e-7 (FD floor)   |
//! | symmetric, cc-pVDZ, PBE (RI-J)    | 9.3e-6              |
//! | symmetric, cc-pVDZ, B3LYP (RI-JK) | 3.6e-6              |
//! | symmetric, cc-pVDZ, wB97X-V       | 2.3e-5              |
//! | distorted, 6-31G,  PBE            | 7.9e-6              |
//! | distorted, 6-31G,  B3LYP          | 6.4e-6              |
//! | distorted, 6-31G,  wB97X          | 7.7e-6              |
//! | distorted, cc-pVDZ, wB97X         | 5.2e-6              |
//! | HO2 (below), 6-31G, UKS B3LYP     | 1.68e-5             |
//!
//! The PySCF-predicted size of the defect (DF gradient minus exact-J/K
//! gradient at the same density, `scripts/` prototype) at the geometries
//! below: RI-J 1.19e-5 / RI-JK 2.22e-5 (cc-pVDZ HF), PBE 7.2e-6 and B3LYP
//! 6.0e-6 (6-31G), OH UHF RI-JK 1.59e-5 (cc-pVDZ), HO2 UKS B3LYP 1.65e-5
//! (6-31G) — the last two agreeing with the measured pre-fix residual above.
//!
//! Thresholds (derived, not guessed):
//! * `TOL = 1e-6`: the FD floor at h = 1e-3 Bohr measured on an exact-integral
//!   RHF is 1.4e-7, so 1e-6 leaves ~7× headroom while sitting ≥ 6× below
//!   every predicted defect.
//! * `NEG_MIN`: the negative control — the OLD pairing (the same DF result
//!   with its recorded route removed, i.e. exact-J/K derivative of a fitted
//!   energy) — must miss FD by at least half the smallest predicted defect of
//!   its case, so each test is shown able to see the bug it guards.
//!
//! KS cases also carry a grid-integration residual (not a DF effect). They
//! are therefore checked DIFFERENTIALLY: the residual (analytic − FD) of the
//! fitted run must equal the residual of the same functional run with exact
//! four-centre J/K (`df_*_aux = Some("")`), in which only the grid residual
//! remains. The grid residual cancels; the DF defect does not.
//!
//! Cost: every FD case solves 2·3·natoms displaced SCFs (in parallel).

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{
    hf_gradient_with_density, rhf_gradient, rhf_gradient_exact_jk, rohf_gradient, uhf_gradient,
};
use ferric_scf::ks_gradient::{ks_gradient_closed, ks_gradient_uks};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use rayon::prelude::*;
use serde::Deserialize;
use std::path::PathBuf;

const JKFIT: &str = "def2-universal-jkfit";
/// Distorted water (Å): no symmetry-zero gradient components.
const WATER: &str = "3\nH2O\nO 0 0 0\nH 0.1 0.76 0.59\nH -0.05 -0.74 0.62\n";
/// OH radical (Å), off-axis so every component is nonzero.
const OH: &str = "2\nOH\nO 0 0 0\nH 0.05 0.1 0.97\n";
/// HO2 radical (Å, ²A''): a NON-degenerate, non-linear open shell for the
/// KS open-shell case. Not OH: its singly occupied π pair is degenerate, the
/// grid is not invariant under rotation about the bond, and each displaced
/// UKS SCF picks its own π orientation — measured pre-fix FD residuals of
/// 5.6e-4 (PBE) / 6.8e-4 (B3LYP) at OH/6-31G that have nothing to do with
/// density fitting. (Grid-free UHF/ROHF on OH is unaffected.) Not NH2 or CH3
/// either: their RI-J and RI-K defects nearly cancel at 6-31G (PySCF
/// prototype: 1.3e-6 / 1.8e-6), too close to `TOL` to discriminate; HO2 gives
/// 1.65e-5.
const HO2: &str = "3\nHO2\nO 0 0 0\nO 0 0 1.33\nH 0.05 0.92 1.6\n";
const FD_STEP: f64 = 1e-3; // Bohr
const TOL: f64 = 1e-6;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Rhf,
    Uhf,
    Rohf,
}

fn base_cfg() -> RhfConfig {
    RhfConfig {
        // ΔE is only a sanity bound under DF (see RhfConfig docs); ΔP drives
        // convergence. Energy error of a variational SCF is O(ΔP²).
        energy_conv: 1e-8,
        density_conv: 1e-9,
        max_iter: 500,
        ..Default::default()
    }
}

fn solve(kind: Kind, mol: &Molecule, bs: &BasisSet, cfg: &RhfConfig) -> ScfResult {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let r = match kind {
        Kind::Rhf => solve_rhf(&ctx, mol, &prep, op, &bounds, cfg),
        Kind::Uhf => solve_uhf(&ctx, mol, &prep, &bounds, cfg),
        Kind::Rohf => solve_rohf(&ctx, mol, &prep, op, &bounds, cfg),
    }
    .unwrap();
    assert!(r.converged, "{kind:?} SCF did not converge: {:?}", r.exit);
    r
}

fn analytic(
    kind: Kind,
    xc: Option<&str>,
    mol: &Molecule,
    bs: &BasisSet,
    res: &ScfResult,
) -> Array2<f64> {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    match (kind, xc) {
        (Kind::Rhf, None) => rhf_gradient(mol, &prep, op, &bounds, res, None),
        (Kind::Uhf, None) => uhf_gradient(mol, &prep, op, &bounds, res, None),
        (Kind::Rohf, None) => rohf_gradient(mol, &prep, op, &bounds, res, None),
        (Kind::Rhf, Some(x)) => ks_gradient_closed(mol, &prep, bs, op, &bounds, x, res, None),
        (Kind::Uhf, Some(x)) => ks_gradient_uks(mol, &prep, bs, op, &bounds, x, res, None),
        (Kind::Rohf, Some(x)) => {
            ferric_scf::ks_gradient::ks_gradient_roks(mol, &prep, bs, op, &bounds, x, res, None)
        }
    }
    .unwrap()
}

/// The OLD pairing: the same fitted SCF result, differentiated with exact
/// four-centre J/K (the route removed, which is exactly what every gradient
/// entry point did before the fix).
fn analytic_old_pairing(
    kind: Kind,
    xc: Option<&str>,
    mol: &Molecule,
    bs: &BasisSet,
    res: &ScfResult,
) -> Array2<f64> {
    assert!(res.df_jk.is_some(), "negative control needs a fitted SCF");
    let mut exact = res.clone();
    exact.df_jk = None;
    analytic(kind, xc, mol, bs, &exact)
}

/// Central finite differences of ferric's OWN SCF energy, every component,
/// each displaced SCF seeded from the reference density (keeps every
/// displacement in the same electronic basin).
fn fd(
    kind: Kind,
    mol: &Molecule,
    bs: &BasisSet,
    cfg: &RhfConfig,
    seed: &Array2<f64>,
) -> Array2<f64> {
    let natoms = mol.atoms.len();
    let cfg = RhfConfig {
        init_guess_density: Some(seed.clone()),
        use_sad_guess: false,
        ..cfg.clone()
    };
    let pairs: Vec<(usize, usize)> = (0..natoms)
        .flat_map(|a| (0..3).map(move |c| (a, c)))
        .collect();
    let vals: Vec<f64> = pairs
        .par_iter()
        .map(|&(a, c)| {
            let e = |s: f64| {
                let mut m = mol.clone();
                match c {
                    0 => m.atoms[a].x += s * FD_STEP,
                    1 => m.atoms[a].y += s * FD_STEP,
                    _ => m.atoms[a].zpos += s * FD_STEP,
                }
                solve(kind, &m, bs, &cfg).energy
            };
            (e(1.0) - e(-1.0)) / (2.0 * FD_STEP)
        })
        .collect();
    let mut g = Array2::zeros((natoms, 3));
    for (&(a, c), v) in pairs.iter().zip(vals) {
        g[(a, c)] = v;
    }
    g
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Absolute check (no grid): analytic of the fitted energy == FD of it, and
/// the old pairing is visibly off.
fn check_absolute(
    label: &str,
    kind: Kind,
    xyz: &str,
    mult: usize,
    basis: &str,
    cfg: RhfConfig,
    neg_min: f64,
) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = basis::bundled(basis).unwrap();
    let res = solve(kind, &mol, &bs, &cfg);
    let g = analytic(kind, None, &mol, &bs, &res);
    let g_old = analytic_old_pairing(kind, None, &mol, &bs, &res);
    let g_fd = fd(kind, &mol, &bs, &cfg, res.density_total());
    let err = max_abs_diff(&g, &g_fd);
    let err_old = max_abs_diff(&g_old, &g_fd);
    eprintln!("{label}: route={:?}", res.df_jk);
    eprintln!("{label}: max|analytic − FD| = {err:.3e} (tol {TOL:.0e}); OLD pairing {err_old:.3e} (must exceed {neg_min:.1e})");
    // Fails if the fix is reverted: the analytic gradient is then the
    // exact-J/K derivative and misses FD by `err_old` (~1e-5).
    assert!(
        err < TOL,
        "{label}: DF gradient misses FD of the DF energy by {err:.3e}"
    );
    // Negative control: the test can see the defect it guards.
    assert!(
        err_old > neg_min,
        "{label}: old pairing only {err_old:.3e} off — the test cannot see the defect"
    );
}

/// Differential check for KS: the fitted run's residual must equal the
/// residual of the same functional with exact J/K (grid residual cancels).
fn check_differential(
    label: &str,
    kind: Kind,
    xc: &str,
    xyz: &str,
    mult: usize,
    basis: &str,
    fitted: RhfConfig,
    neg_min: f64,
) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = basis::bundled(basis).unwrap();
    let exact_cfg = RhfConfig {
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..fitted.clone()
    };

    let res_f = solve(kind, &mol, &bs, &fitted);
    assert!(
        res_f.df_jk.is_some(),
        "{label}: fitted run recorded no DF route"
    );
    let fd_f = fd(kind, &mol, &bs, &fitted, res_f.density_total());
    let r_f = &analytic(kind, Some(xc), &mol, &bs, &res_f) - &fd_f;
    let r_old = &analytic_old_pairing(kind, Some(xc), &mol, &bs, &res_f) - &fd_f;

    let res_x = solve(kind, &mol, &bs, &exact_cfg);
    assert!(
        res_x.df_jk.is_none(),
        "{label}: exact run must record no DF route"
    );
    let r_x = &analytic(kind, Some(xc), &mol, &bs, &res_x)
        - &fd(kind, &mol, &bs, &exact_cfg, res_x.density_total());

    let grid = r_x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let err = max_abs_diff(&r_f, &r_x);
    let err_old = max_abs_diff(&r_old, &r_x);
    eprintln!("{label}: route={:?}", res_f.df_jk);
    eprintln!(
        "{label}: exact-J/K residual (grid) {grid:.3e}; fitted residual − grid = {err:.3e} (tol {TOL:.0e}); \
         OLD pairing {err_old:.3e} (must exceed {neg_min:.1e})"
    );
    // Fails if the fix is reverted: the fitted residual then carries the RI
    // defect (6-7e-6 here) on top of the grid residual.
    assert!(
        err < TOL,
        "{label}: DF gradient residual differs from the exact-J/K residual by {err:.3e}"
    );
    assert!(
        err_old > neg_min,
        "{label}: old pairing only {err_old:.3e} off — the test cannot see the defect"
    );
}

fn df(j: bool, k: bool) -> RhfConfig {
    RhfConfig {
        df_j_aux: j.then(|| JKFIT.to_string()),
        df_k_aux: k.then(|| JKFIT.to_string()),
        ..base_cfg()
    }
}

// ---------------------------------------------------------------- anchors --

#[test]
fn hf_ri_j_gradient_matches_fd_of_ri_energy() {
    // RI-J only; K from the exact four-centre builder. Predicted defect 1.19e-5.
    check_absolute(
        "HF RI-J water/cc-pVDZ",
        Kind::Rhf,
        WATER,
        1,
        "cc-pvdz",
        df(true, false),
        5e-6,
    );
}

#[test]
fn hf_ri_jk_gradient_matches_fd_of_ri_energy() {
    // Predicted defect 2.22e-5.
    check_absolute(
        "HF RI-JK water/cc-pVDZ",
        Kind::Rhf,
        WATER,
        1,
        "cc-pvdz",
        df(true, true),
        5e-6,
    );
}

#[test]
fn uhf_ri_jk_gradient_matches_fd_of_ri_energy() {
    // Open shell: per-spin exchange channels. Predicted defect 1.59e-5.
    check_absolute(
        "UHF RI-JK OH/cc-pVDZ",
        Kind::Uhf,
        OH,
        2,
        "cc-pvdz",
        df(true, true),
        5e-6,
    );
}

#[test]
fn rohf_ri_jk_gradient_matches_fd_of_ri_energy() {
    check_absolute(
        "ROHF RI-JK HO2/cc-pVDZ",
        Kind::Rohf,
        HO2,
        2,
        "cc-pvdz",
        df(true, true),
        5e-6,
    );
}

#[test]
fn pbe_default_ri_j_gradient_matches_fd() {
    // The default KS path: df_j_aux unset → RI-J auto-default. Predicted 7.2e-6.
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        ..base_cfg()
    };
    check_differential(
        "PBE (default RI-J) water/6-31G",
        Kind::Rhf,
        "PBE",
        WATER,
        1,
        "6-31g",
        cfg,
        3e-6,
    );
}

#[test]
fn b3lyp_default_ri_jk_gradient_matches_fd() {
    // Hybrid default: RI-J and RI-K both auto-default. Predicted 6.0e-6.
    let cfg = RhfConfig {
        xc: Some("B3LYP".into()),
        ..base_cfg()
    };
    check_differential(
        "B3LYP (default RI-JK) water/6-31G",
        Kind::Rhf,
        "B3LYP",
        WATER,
        1,
        "6-31g",
        cfg,
        3e-6,
    );
}

#[test]
fn uks_b3lyp_ri_jk_gradient_matches_fd() {
    // Open-shell KS: no auto-default on the UHF path, so name the aux basis.
    let cfg = RhfConfig {
        xc: Some("B3LYP".into()),
        ..df(true, true)
    };
    // Predicted defect 1.65e-5 (PySCF prototype, same geometry and basis).
    check_differential(
        "UKS B3LYP RI-JK HO2/6-31G",
        Kind::Uhf,
        "B3LYP",
        HO2,
        2,
        "6-31g",
        cfg,
        5e-6,
    );
}

/// Range-separated hybrid: SR (erfc) and LR (erf) exchange are ALWAYS fitted,
/// the LR metric has eigenvalues below DfK's 1e-10 cut (and a noise band up to
/// 1e-7 that the gradient treats as dropped — see `df_gradient` module doc).
/// No exact-K run exists to difference against, so this is an absolute check
/// with the grid residual inside the tolerance: 2e-6, i.e. twice `TOL` for
/// the (unmeasured here, ≤~1e-6 from the pre-fix PBE/B3LYP comparison
/// against PySCF's predicted defect) grid residual. Pre-fix residual on this
/// exact system: 7.7e-6.
#[test]
fn wb97x_rsh_gradient_matches_fd() {
    let xc = "HYB_GGA_XC_WB97X"; // wB97X without VV10: no NLC-grid residual
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let cfg = RhfConfig {
        xc: Some(xc.into()),
        ..base_cfg()
    };
    let res = solve(Kind::Rhf, &mol, &bs, &cfg);
    let route = res
        .df_jk
        .clone()
        .expect("RSH SCF must record its fitted exchange");
    assert!(route.rsh_k.is_some() && route.j_aux.is_some(), "{route:?}");
    let g = analytic(Kind::Rhf, Some(xc), &mol, &bs, &res);
    let g_old = analytic_old_pairing(Kind::Rhf, Some(xc), &mol, &bs, &res);
    let g_fd = fd(Kind::Rhf, &mol, &bs, &cfg, res.density_total());
    let err = max_abs_diff(&g, &g_fd);
    let err_old = max_abs_diff(&g_old, &g_fd);
    eprintln!("wB97X water/6-31G: max|analytic − FD| = {err:.3e}; OLD pairing {err_old:.3e}");
    // Fails if the fix is reverted (7.7e-6 measured).
    assert!(err < 2e-6, "wB97X: DF gradient misses FD by {err:.3e}");
    assert!(err_old > 4e-6, "wB97X: old pairing only {err_old:.3e} off");
}

// -------------------------------------------------------------- cross-code --

#[derive(Deserialize)]
struct DfGradRef {
    e_total: f64,
    grad: Vec<[f64; 3]>,
    converged: bool,
}

fn load_ref(name: &str) -> DfGradRef {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

/// ferric vs `pyscf.df.grad` (scripts/gen_pyscf_df_grad_refs.py), same aux
/// basis, auxbasis_response on. Tolerance 1e-6 Ha/Bohr: the pre-fix
/// difference is the defect itself (1.2e-5 – 2.2e-5 here), while both codes
/// differentiate identical math (the Coulomb metric of these systems has no
/// mode below DfK's cut), so agreement is limited only by SCF convergence
/// and integral screening. The DF energies are compared first (1e-7 Ha) so a
/// gradient failure cannot come from a different aux basis or fit.
fn check_vs_pyscf(label: &str, kind: Kind, xyz: &str, mult: usize, cfg: RhfConfig, refname: &str) {
    let r = load_ref(refname);
    assert!(r.converged, "{refname} not converged");
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let res = solve(kind, &mol, &bs, &cfg);
    let de = (res.energy - r.e_total).abs();
    let g = analytic(kind, None, &mol, &bs, &res);
    let g_old = analytic_old_pairing(kind, None, &mol, &bs, &res);
    let mut err: f64 = 0.0;
    let mut err_old: f64 = 0.0;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            err = err.max((g[(a, c)] - r.grad[a][c]).abs());
            err_old = err_old.max((g_old[(a, c)] - r.grad[a][c]).abs());
        }
    }
    eprintln!("{label}: |ΔE| vs PySCF {de:.2e}; max|Δg| {err:.3e}; OLD pairing {err_old:.3e}");
    assert!(
        de < 1e-7,
        "{label}: DF energy differs from PySCF by {de:.2e} (aux basis / fit mismatch)"
    );
    assert!(
        err < 1e-6,
        "{label}: gradient differs from PySCF DF gradient by {err:.3e}"
    );
    assert!(
        err_old > 5e-6,
        "{label}: old pairing only {err_old:.3e} from PySCF"
    );
}

#[test]
fn hf_ri_j_gradient_matches_pyscf_only_dfj() {
    check_vs_pyscf(
        "HF RI-J",
        Kind::Rhf,
        WATER,
        1,
        df(true, false),
        "h2o_cc-pvdz_hf_rij_dfgrad.json",
    );
}

#[test]
fn hf_ri_jk_gradient_matches_pyscf_df() {
    check_vs_pyscf(
        "HF RI-JK",
        Kind::Rhf,
        WATER,
        1,
        df(true, true),
        "h2o_cc-pvdz_hf_rijk_dfgrad.json",
    );
}

#[test]
fn uhf_ri_jk_gradient_matches_pyscf_df() {
    check_vs_pyscf(
        "UHF RI-JK",
        Kind::Uhf,
        OH,
        2,
        df(true, true),
        "oh_cc-pvdz_uhf_rijk_dfgrad.json",
    );
}

// ---------------------------------------------------- routing + regression --

/// The solver records EXACTLY the builders it used, across the resolution
/// rules (auto-defaults, the `Some("")` opt-out, the consumption gates, the
/// open-shell "no auto-default" rule, and RSH's always-fitted exchange).
/// Fails if any solver stops recording, or records a builder it skipped
/// (e.g. the DF-K a pure functional never builds).
#[test]
fn scf_records_the_df_route_it_used() {
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let oh = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let rhf = |cfg: RhfConfig| {
        solve(
            Kind::Rhf,
            &mol,
            &bs,
            &RhfConfig {
                density_conv: 1e-7,
                ..cfg
            },
        )
    };
    let uhf = |cfg: RhfConfig| {
        solve(
            Kind::Uhf,
            &oh,
            &bs,
            &RhfConfig {
                density_conv: 1e-7,
                ..cfg
            },
        )
    };
    let j = Some(JKFIT.to_string());

    // Plain HF: nothing fitted, no route (exact gradient path).
    assert!(rhf(RhfConfig::default()).df_jk.is_none());
    // PBE: RI-J auto-default; the requested-but-unconsumed DF-K is NOT recorded.
    let r = rhf(RhfConfig {
        xc: Some("PBE".into()),
        df_k_aux: j.clone(),
        ..Default::default()
    })
    .df_jk
    .unwrap();
    assert_eq!(
        (r.j_aux.clone(), r.k_aux.clone(), r.rsh_k.clone()),
        (j.clone(), None, None)
    );
    // B3LYP: both auto-default.
    let r = rhf(RhfConfig {
        xc: Some("B3LYP".into()),
        ..Default::default()
    })
    .df_jk
    .unwrap();
    assert_eq!((r.j_aux.clone(), r.k_aux.clone()), (j.clone(), j.clone()));
    // B3LYP with the explicit opt-out: exact J and K, no route at all.
    assert!(rhf(RhfConfig {
        xc: Some("B3LYP".into()),
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    })
    .df_jk
    .is_none());
    // RSH: RI-J + the SR/LR pair at the functional's ω; no ω = 0 DF-K.
    let r = rhf(RhfConfig {
        xc: Some("HYB_GGA_XC_WB97X".into()),
        ..Default::default()
    })
    .df_jk
    .unwrap();
    assert_eq!(r.k_aux, None);
    let (aux, omega) = r.rsh_k.clone().unwrap();
    assert_eq!(aux, JKFIT);
    assert!((omega - 0.3).abs() < 1e-12, "wB97X ω = {omega}");
    // Open shell: no auto-default, so PBE UKS without names is all exact...
    assert!(uhf(RhfConfig {
        xc: Some("PBE".into()),
        ..Default::default()
    })
    .df_jk
    .is_none());
    // ...and named aux bases are recorded as used.
    let r = uhf(RhfConfig {
        xc: Some("B3LYP".into()),
        df_j_aux: j.clone(),
        df_k_aux: j.clone(),
        ..Default::default()
    })
    .df_jk
    .unwrap();
    assert_eq!((r.j_aux.clone(), r.k_aux.clone()), (j.clone(), j.clone()));
    // Open-shell RSH: J exact (the UHF path fits no J when ω > 0), exchange fitted.
    let r = uhf(RhfConfig {
        xc: Some("HYB_GGA_XC_WB97X".into()),
        ..Default::default()
    })
    .df_jk
    .unwrap();
    assert_eq!(r.j_aux, None);
    assert!(r.rsh_k.is_some());
}

/// No regression: an exact-integral SCF records no route, and `rhf_gradient`
/// on it is BIT-IDENTICAL to the pre-fix composition
/// `hf_gradient_with_density(D, W)` (the exact four-centre path, untouched).
/// Fails if the all-exact case is ever sent through the routed function or
/// the route is recorded for an unfitted run.
#[test]
fn exact_jk_gradient_is_bit_identical_to_the_unrouted_path() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    for cfg in [
        base_cfg(),
        RhfConfig {
            df_j_aux: Some(String::new()),
            df_k_aux: Some(String::new()),
            ..base_cfg()
        },
    ] {
        let res = solve(Kind::Rhf, &mol, &bs, &cfg);
        assert!(
            res.df_jk.is_none(),
            "exact SCF recorded a DF route: {:?}",
            res.df_jk
        );
        let g = rhf_gradient(&mol, &prep, op, &bounds, &res, None).unwrap();
        let nocc = (mol.nelec() / 2) as usize;
        let w = ferric_scf::gradient::build_energy_weighted_density(&res, nocc);
        let mut g_ref =
            hf_gradient_with_density(&mol, &prep, op, &bounds, res.density_r(), &w, None).unwrap();
        // rhf_gradient's own tail (all-electron: a zero array, same += as in rhf_gradient).
        g_ref += &ferric_scf::gradient::ecp_gradient(&mol, &prep, res.density_r()).unwrap();
        assert!(
            g.iter()
                .zip(g_ref.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "exact-JK rhf_gradient is not bit-identical to the unrouted composition"
        );
        // And the explicit exact variant agrees bitwise too.
        let g_x = rhf_gradient_exact_jk(&mol, &prep, op, &bounds, &res, None).unwrap();
        assert!(g
            .iter()
            .zip(g_x.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

/// Exact-J/K ROHF on a radical whose open shell couples to the closed shells.
/// Guards the energy-weighted density (`rohf_energy_weighted_density`): the
/// former `C diag(2ε_c, ε_o) Cᵀ` from the Roothaan effective Fock missed FD by
/// 1.2e-2 (STO-3G) / 2.5e-2 (cc-pVDZ) Ha/Bohr here, equal and opposite on the
/// two oxygens along the O–O bond. OH cannot see it (its π SOMO is decoupled
/// from the σ closed shells by symmetry).
fn exact_open_shell_vs_fd(label: &str, kind: Kind, xc: Option<&str>, basis: &str, tol: f64) {
    let mol = Molecule::parse_xyz(HO2, 0, 2).unwrap();
    let bs = basis::bundled(basis).unwrap();
    let cfg = RhfConfig {
        xc: xc.map(Into::into),
        ..df(false, false)
    };
    let res = solve(kind, &mol, &bs, &cfg);
    assert!(res.df_jk.is_none(), "{label}: expected exact J/K");
    let g = analytic(kind, xc, &mol, &bs, &res);
    let g_fd = fd(kind, &mol, &bs, &cfg, res.density_total());
    let err = max_abs_diff(&g, &g_fd);
    eprintln!("{label}: max|analytic − FD| = {err:.3e} (tol {tol:.0e})");
    assert!(
        err < tol,
        "{label}: gradient misses FD of its own energy by {err:.3e}"
    );
}

#[test]
fn rohf_exact_gradient_matches_fd_on_a_coupled_radical() {
    exact_open_shell_vs_fd("ROHF exact HO2/STO-3G", Kind::Rohf, None, "sto-3g", TOL);
}

#[test]
fn roks_exact_gradient_matches_fd_on_a_coupled_radical() {
    exact_open_shell_vs_fd(
        "ROKS B3LYP exact HO2/STO-3G",
        Kind::Rohf,
        Some("B3LYP"),
        "sto-3g",
        2e-6,
    );
}
