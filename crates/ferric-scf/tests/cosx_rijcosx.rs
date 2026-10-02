//! RIJCOSX: density-fitted Coulomb (RI-J) with COSX exchange, the pruned
//! `sgx` COSX grid, and the final-grid pass.
//!
//! RIJCOSX is what `k_builder = "cosx"` does whenever an RI-J aux basis is
//! active (named, or auto-defaulted for a functional): J from `DfJ`, K from
//! COSX (Neese et al. 2009; ORCA RIJCOSX; Psi4 DFDIRJ+COSX). Before it, any
//! active DF switched COSX off.
//!
//! PRE-REGISTERED (before any of these ran):
//!
//! * ENERGY SEPARATION. RI-J and COSX perturb different terms of a
//!   variational energy, so to first order their errors add:
//!   `E(RI-J, COSX) - E(RI-J, K) == E(J, COSX) - E(J, K)` up to a second-order
//!   cross term (density response to one error times the other error),
//!   expected <= 1e-9 Ha on water/6-31G (both errors ~1e-5..1e-4), and
//!   scaling like `e_cosx · e_rij` (checked on two grids). If the
//!   implementation silently dropped either half the identity misses by the
//!   size of the other error (1e-5..1e-4): RI-J ignored -> `cross = -(RI-J
//!   error)`, COSX ignored -> `cross = -(COSX error)`.
//! * GRADIENT. Analytic vs central FD (h = 1e-4) of the RIJCOSX energy, same
//!   bar as `cosx_gradient.rs` (`TOL_FD = 1e-7`, fit off; `TOL_FIT = 3e-8`
//!   fitted). Negative controls: (i) exact-J + COSX gradient (the RI-J route
//!   dropped) and (ii) RI-J + EXACT-K gradient (the routed J+K path, i.e. COSX
//!   dropped) must each miss FD by >= 10 x the bar.
//! * PRUNED GRID. The Becke weight response is per point, so pruning needs no
//!   new term; the fixed-density anchor and the SCF FD test on the `sgx` grid
//!   must pass at the flat-grid bars. ARTIFACT HYPOTHESIS: if the response
//!   grid and the energy grid disagreed (e.g. the response builder ignored
//!   pruning) the gradient would miss by the flat-vs-pruned energy difference
//!   per Bohr, O(1e-3..1e-2) — the bitwise grid-identity anchors in
//!   `ferric-dft` exclude that first.
//! * FINAL PASS. With `final_grid == grid` the pass is exactly zero.
//!
//! Commands:
//!   OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-scf --test cosx_rijcosx
//!   OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-scf --test cosx_rijcosx -- --include-ignored --nocapture

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::prune::PruneScheme;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_gradient::{cosx_exchange_gradient, scf_exchange_is_cosx};
use ferric_scf::cosx_k::{CosxConfig, CosxHalfTransform, CosxK};
use ferric_scf::fock::KBuilder;
use ferric_scf::gradient::{
    build_energy_weighted_density, gradient_task_config, oneelectron_gradient,
    restricted_scf_gradient, rhf_gradient, twoelectron_j_gradient, uhf_gradient,
    unrestricted_scf_gradient,
};
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use rayon::prelude::*;

const TOL_FD: f64 = 1e-7;
const TOL_FIT: f64 = 3e-8;
const FD_H: f64 = 1e-4;
const JK: &str = "def2-universal-jkfit";

const WATER: &str = "3
water, distorted
O   0.00  0.02  0.11
H   0.03  0.76 -0.47
H  -0.02 -0.75 -0.45
";

const HO2: &str = "3\nHO2\nO 0 0 0\nO 0 0 1.33\nH 0.05 0.92 1.6\n";

fn mol_of(xyz: &str, mult: usize) -> Molecule {
    Molecule::parse_xyz(xyz, 0, mult).expect("xyz")
}

fn displaced(mol: &Molecule, atom: usize, c: usize, d: f64) -> Molecule {
    let mut m = mol.clone();
    match c {
        0 => m.atoms[atom].x += d,
        1 => m.atoms[atom].y += d,
        _ => m.atoms[atom].zpos += d,
    }
    m
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// COSX with the screens and the fit OFF: the energy the gradient
/// differentiates exactly.
fn cosx_exact() -> CosxConfig {
    CosxConfig {
        overlap_fit: false,
        screen_thresh: None,
        half_transform: CosxHalfTransform::Dense,
        ..CosxConfig::default()
    }
}

/// The fitted, unscreened (30,110) grid of `cosx_gradient.rs`'s fitted tests.
fn cosx_fitted() -> CosxConfig {
    let mut c = CosxConfig {
        overlap_fit: true,
        ..cosx_exact()
    };
    c.grid.n_radial = 30;
    c.grid.n_angular = 110;
    c
}

/// The pruned `sgx` grid at peak 194 (ORCA GridX3 / PySCF SGX level 2 regions).
fn sgx(cfg: CosxConfig, n_radial: usize) -> CosxConfig {
    let mut c = cfg;
    c.grid.n_radial = n_radial;
    c.grid.n_angular = 194;
    c.grid.prune = Some(PruneScheme::Sgx);
    c
}

/// `xc = None`: HF. `j_fit`: RI-J on (named), else exact J (`""`).
/// `cosx = None`: exact K (direct, or `""` DF-K for a functional).
fn cfg(xc: Option<&str>, j_fit: bool, cosx: Option<CosxConfig>) -> RhfConfig {
    RhfConfig {
        k_builder: cosx.as_ref().map(|_| "cosx".to_string()),
        cosx: cosx.unwrap_or_default(),
        xc: xc.map(str::to_string),
        df_j_aux: Some(if j_fit { JK.to_string() } else { String::new() }),
        df_k_aux: Some(String::new()),
        energy_conv: 1e-8,
        density_conv: 1e-10,
        max_iter: 300,
        ..Default::default()
    }
}

fn prep_of(
    mol: &Molecule,
    basis: &str,
) -> (ferric_core::basis::BasisSet, PreparedBasis, SchwarzBounds) {
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(mol, &bs).expect("prep");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
    (bs, prep, bounds)
}

fn solve_r(mol: &Molecule, basis: &str, c: &RhfConfig) -> ScfResult {
    let (_, prep, bounds) = prep_of(mol, basis);
    let r = solve_rhf(
        &ParallelContext::default(),
        mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        c,
    )
    .expect("rhf");
    assert!(r.converged, "SCF did not converge: {:?}", r.exit);
    r
}

fn solve_u(mol: &Molecule, basis: &str, c: &RhfConfig) -> ScfResult {
    let (_, prep, bounds) = prep_of(mol, basis);
    let r = solve_uhf(&ParallelContext::default(), mol, &prep, &bounds, c).expect("uhf");
    assert!(r.converged, "UHF did not converge: {:?}", r.exit);
    r
}

/// Central-FD gradient; every displaced SCF is seeded from `seed`.
fn fd(mol: &Molecule, solve: &(dyn Fn(&Molecule) -> f64 + Sync)) -> Array2<f64> {
    let n = mol.atoms.len();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|a| (0..3).map(move |c| (a, c))).collect();
    let vals: Vec<f64> = pairs
        .par_iter()
        .map(|&(a, c)| {
            (solve(&displaced(mol, a, c, FD_H)) - solve(&displaced(mol, a, c, -FD_H)))
                / (2.0 * FD_H)
        })
        .collect();
    let mut g = Array2::zeros((n, 3));
    for (&(a, c), v) in pairs.iter().zip(vals) {
        g[(a, c)] = v;
    }
    g
}

fn seeded(c: &RhfConfig, seed: &Array2<f64>) -> RhfConfig {
    RhfConfig {
        init_guess_density: Some(seed.clone()),
        use_sad_guess: false,
        ..c.clone()
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// `k_builder = "cosx"` stays COSX under RI-J (named or auto-defaulted) and is
/// refused next to a NAMED df_k_aux. Fails if the old rule (any DF switches
/// COSX off) is restored, or if the conflict is resolved silently.
#[test]
fn rijcosx_dispatch_rules() {
    let named_j = cfg(None, true, Some(cosx_exact()));
    assert!(scf_exchange_is_cosx(&named_j, false).unwrap());
    assert!(scf_exchange_is_cosx(&named_j, true).unwrap());
    // B3LYP with the RHF auto-defaults (df_j_aux / df_k_aux unset): RIJCOSX.
    let auto = RhfConfig {
        k_builder: Some("cosx".into()),
        xc: Some("B3LYP".into()),
        ..Default::default()
    };
    assert!(scf_exchange_is_cosx(&auto, false).unwrap());
    // A named DF-K next to COSX is a conflict, in the predicate AND the SCF.
    let conflict = RhfConfig {
        df_k_aux: Some(JK.into()),
        ..named_j.clone()
    };
    let err = scf_exchange_is_cosx(&conflict, false).unwrap_err();
    assert!(format!("{err}").contains("df_k_aux"), "{err}");
    let mol = mol_of(WATER, 1);
    let (_, prep, bounds) = prep_of(&mol, "sto-3g");
    let err = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &conflict,
    )
    .unwrap_err();
    assert!(
        format!("{err}").contains("conflicts with df_k_aux"),
        "{err}"
    );
    // A pure functional or an RSH functional still ignores COSX.
    let pbe = RhfConfig {
        xc: Some("PBE".into()),
        ..auto.clone()
    };
    assert!(!scf_exchange_is_cosx(&pbe, false).unwrap());
    let rsh = RhfConfig {
        xc: Some("wB97X-V".into()),
        ..auto
    };
    assert!(!scf_exchange_is_cosx(&rsh, false).unwrap());
}

// ---------------------------------------------------------------------------
// Energy separation
// ---------------------------------------------------------------------------

/// The four corners of (J: exact | RI) x (K: exact | COSX). Returns
/// `(E_rc, E_re, E_xc, E_xe)`.
fn four_corners(
    mol: &Molecule,
    basis: &str,
    xc: Option<&str>,
    cosx: CosxConfig,
    open_shell: bool,
) -> (f64, f64, f64, f64) {
    let e = |j_fit: bool, c: Option<CosxConfig>| -> f64 {
        let c = cfg(xc, j_fit, c);
        if open_shell {
            solve_u(mol, basis, &c).energy
        } else {
            solve_r(mol, basis, &c).energy
        }
    };
    (
        e(true, Some(cosx.clone())),
        e(true, None),
        e(false, Some(cosx)),
        e(false, None),
    )
}

fn separation_case(label: &str, corners: (f64, f64, f64, f64)) {
    let (rc, re, xc, xe) = corners;
    let cosx_err_rij = rc - re;
    let cosx_err_exact = xc - xe;
    let rij_err = re - xe;
    let cross = cosx_err_rij - cosx_err_exact;
    let ratio = cross.abs() / (cosx_err_exact * rij_err).abs();
    println!(
        "{label}: COSX error under RI-J {cosx_err_rij:+.6e}, under exact J {cosx_err_exact:+.6e}, \
         RI-J error {rij_err:+.6e}, cross term {cross:+.3e} (|cross| / |e_cosx e_rij| = {ratio:.2} /Ha)"
    );
    // The harness can see both failure modes: dropping either half makes
    // |cross| equal to the other error, i.e. a ratio of 1/|e_other| >= 1e3.
    assert!(
        cosx_err_exact.abs() > 1e-8,
        "{label}: COSX error too small to test"
    );
    assert!(
        rij_err.abs() > 1e-8,
        "{label}: RI-J error too small to test"
    );
    assert!(
        ratio < CROSS_RATIO_BAR || cross.abs() < CROSS_ABS_FLOOR,
        "{label}: RIJCOSX != RI-J + COSX error beyond second order (cross {cross:e}, ratio {ratio})"
    );
}

/// The cross term is SECOND order, `cross ~ c · e_cosx · e_rij`. MEASURED `c`
/// (/Ha): water/6-31G HF 2.81 (50,110), 5.76 (30,50); B3LYP 3.01; HO2 UHF
/// 3.04. Bar 30: 5-10x the measurements, while dropping either half gives
/// `c >= 1/e_other ~ 2.6e4`.
const CROSS_RATIO_BAR: f64 = 30.0;

/// Absolute floor of the cross-term check, for grids so fine that the cross
/// term reaches the RI-J SCF's own reproducibility: MEASURED 2.3e-11 at
/// (75,302) (ratio 27 there, i.e. the second-order law no longer resolves it).
const CROSS_ABS_FLOOR: f64 = 1e-10;

/// HF water/6-31G, RIJCOSX with the fit off, at the (50,110) and a coarse
/// (30,50) COSX grid: the cross term must scale with the COSX error.
#[test]
fn rijcosx_energy_is_rij_plus_the_cosx_error_hf() {
    let mol = mol_of(WATER, 1);
    let corners = four_corners(&mol, "6-31g", None, cosx_exact(), false);
    separation_case("water/6-31G HF (50,110)", corners);
    let mut coarse = cosx_exact();
    coarse.grid.n_radial = 30;
    coarse.grid.n_angular = 50;
    let corners = four_corners(&mol, "6-31g", None, coarse, false);
    separation_case("water/6-31G HF (30,50)", corners);
}

/// On a fine COSX grid the cross term falls with the COSX error (to ~1e-11).
#[test]
#[ignore = "slow: four water/6-31G SCFs on a (75,302) COSX grid"]
fn rijcosx_energy_separation_on_a_fine_grid() {
    let mol = mol_of(WATER, 1);
    let mut fine = cosx_exact();
    fine.grid.n_radial = 75;
    fine.grid.n_angular = 302;
    separation_case(
        "water/6-31G HF (75,302)",
        four_corners(&mol, "6-31g", None, fine, false),
    );
}

/// Same identity for B3LYP (the KS path, where RI-J is the run_dft default).
#[test]
#[ignore = "slow: four B3LYP/6-31G SCFs"]
fn rijcosx_energy_is_rij_plus_the_cosx_error_b3lyp() {
    let mol = mol_of(WATER, 1);
    let corners = four_corners(&mol, "6-31g", Some("B3LYP"), cosx_exact(), false);
    separation_case("water/6-31G B3LYP", corners);
}

/// Same identity for UHF (HO2), the open-shell routing.
#[test]
#[ignore = "slow: four HO2/6-31G UHF SCFs"]
fn rijcosx_energy_is_rij_plus_the_cosx_error_uhf() {
    let mol = mol_of(HO2, 2);
    let corners = four_corners(&mol, "6-31g", None, cosx_exact(), true);
    separation_case("HO2/6-31G UHF", corners);
}

/// The B3LYP auto-default: `k_builder = "cosx"` with df_j_aux/df_k_aux UNSET
/// must equal the explicitly-named RIJCOSX run bit for bit (the auto-default
/// RI-J is on, the auto-default RI-K is replaced by COSX).
#[test]
fn b3lyp_auto_default_is_rijcosx() {
    let mol = mol_of(WATER, 1);
    let auto = RhfConfig {
        k_builder: Some("cosx".into()),
        cosx: cosx_exact(),
        xc: Some("B3LYP".into()),
        energy_conv: 1e-8,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    };
    let named = RhfConfig {
        df_j_aux: Some(JK.into()),
        df_k_aux: Some(String::new()),
        ..auto.clone()
    };
    let a = solve_r(&mol, "sto-3g", &auto);
    let b = solve_r(&mol, "sto-3g", &named);
    println!("B3LYP auto {:.12} named {:.12}", a.energy, b.energy);
    assert_eq!(a.energy.to_bits(), b.energy.to_bits());
    let route = a.df_jk.as_ref().expect("RI-J route recorded");
    assert_eq!(route.j_aux.as_deref(), Some(JK));
    assert!(route.k_aux.is_none(), "COSX replaced the RI-K auto-default");
}

// ---------------------------------------------------------------------------
// Gradients
// ---------------------------------------------------------------------------

/// Explicit construction of the two wrong pairings for a fit-off HF RIJCOSX
/// run: (exact-J + COSX K) and (RI-J + exact K).
fn wrong_pairings(
    mol: &Molecule,
    basis: &str,
    c: &RhfConfig,
    r: &ScfResult,
) -> (Array2<f64>, Array2<f64>) {
    let (_, prep, bounds) = prep_of(mol, basis);
    let op = Operator::coulomb();
    let d = r.density_r();
    let w = build_energy_weighted_density(r, (mol.nelec() / 2) as usize);
    let mut g_exact_j = oneelectron_gradient(mol, &prep, d, &w, None).expect("1e");
    g_exact_j += &twoelectron_j_gradient(&prep, op, &bounds, d).expect("J");
    g_exact_j += &cosx_exchange_gradient(mol, &prep, &c.cosx, &[(d, -0.25)]).expect("K");
    // rhf_gradient reads the recorded RI-J route and differentiates EXACT K.
    let g_exact_k = rhf_gradient(mol, &prep, op, &bounds, r, None).expect("rhf");
    (g_exact_j, g_exact_k)
}

fn rhf_rijcosx_fd_case(basis: &str, cosx: CosxConfig, tol: f64) {
    let mol = mol_of(WATER, 1);
    let c = cfg(None, true, Some(cosx));
    let r = solve_r(&mol, basis, &c);
    assert!(r
        .df_jk
        .as_ref()
        .is_some_and(|x| x.j_aux.is_some() && x.k_aux.is_none()));
    let (bs, prep, bounds) = prep_of(&mol, basis);
    let g = restricted_scf_gradient(&mol, &prep, &bs, Operator::coulomb(), &bounds, &c, &r)
        .expect("rijcosx gradient");
    let cs = seeded(&c, r.density_r());
    let g_fd = fd(&mol, &|m: &Molecule| solve_r(m, basis, &cs).energy);
    let e = max_abs(&(&g - &g_fd));
    let (g_ej, g_ek) = wrong_pairings(&mol, basis, &c, &r);
    let e_ej = max_abs(&(&g_ej - &g_fd));
    let e_ek = max_abs(&(&g_ek - &g_fd));
    println!(
        "RIJCOSX RHF water/{basis} fit={} prune={:?}: max|g - FD| = {e:.3e}; exact-J pairing \
         {e_ej:.3e}; exact-K pairing {e_ek:.3e}",
        c.cosx.overlap_fit, c.cosx.grid.prune
    );
    assert!(e < tol, "RIJCOSX gradient off FD by {e:e}");
    assert!(
        e_ej > 10.0 * tol,
        "NEGATIVE CONTROL blind (exact J): {e_ej:e}"
    );
    assert!(
        e_ek > 10.0 * tol,
        "NEGATIVE CONTROL blind (exact K): {e_ek:e}"
    );
}

#[test]
fn rhf_rijcosx_gradient_matches_fd_sto3g() {
    rhf_rijcosx_fd_case("sto-3g", cosx_exact(), TOL_FD);
}

/// Fitted COSX: the Z-vector's Coulomb response must be the RI-J one.
#[test]
#[ignore = "slow: 18 fitted RIJCOSX SCFs"]
fn fitted_rhf_rijcosx_gradient_matches_fd_sto3g() {
    rhf_rijcosx_fd_case("sto-3g", cosx_fitted(), TOL_FIT);
}

/// Pruned `sgx` grid (35 radial, peak 194): fit off, then fitted.
#[test]
#[ignore = "slow: 18 RIJCOSX SCFs on the pruned grid"]
fn rhf_rijcosx_gradient_on_the_pruned_grid_matches_fd() {
    rhf_rijcosx_fd_case("sto-3g", sgx(cosx_exact(), 35), TOL_FD);
}

#[test]
#[ignore = "slow: 18 fitted RIJCOSX SCFs on the pruned grid"]
fn fitted_rhf_rijcosx_gradient_on_the_pruned_grid_matches_fd() {
    rhf_rijcosx_fd_case("sto-3g", sgx(cosx_fitted(), 35), TOL_FIT);
}

/// B3LYP RIJCOSX (fit off): the exchange-isolated residual, relative to the
/// same residual of an RI-J + exact-K B3LYP run (cancels the XC gradient's own
/// FD floor), as in `cosx_gradient.rs::restricted_case`.
#[test]
#[ignore = "slow: 36 B3LYP SCFs for two central-FD gradients"]
fn b3lyp_rijcosx_gradient_matches_fd_sto3g() {
    let basis = "sto-3g";
    let mol = mol_of(WATER, 1);
    let (bs, prep, bounds) = prep_of(&mol, basis);
    let op = Operator::coulomb();
    let run = |c: &RhfConfig| -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let r = solve_r(&mol, basis, c);
        let g = restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, c, &r).expect("grad");
        // ks_gradient_closed = the routed RI-J + EXACT-K derivative.
        let g_k = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, "B3LYP", &r, None).expect("ks");
        let cs = seeded(c, r.density_r());
        let g_fd = fd(&mol, &|m: &Molecule| solve_r(m, basis, &cs).energy);
        (g, g_k, g_fd)
    };
    let c_x = cfg(Some("B3LYP"), true, Some(cosx_exact()));
    let c_k = cfg(Some("B3LYP"), true, None);
    let (g_x, g_xk, fd_x) = run(&c_x);
    let (g_k, _, fd_k) = run(&c_k);
    let res_ref = &g_k - &fd_k;
    let rel = max_abs(&(&(&g_x - &fd_x) - &res_ref));
    let rel_old = max_abs(&(&(&g_xk - &fd_x) - &res_ref));
    println!(
        "B3LYP RIJCOSX water/STO-3G: exchange-isolated residual {rel:.3e} (exact-K pairing \
         {rel_old:.3e}); RI-J+K run's own residual {:.3e}",
        max_abs(&res_ref)
    );
    assert!(rel < TOL_FD, "B3LYP RIJCOSX gradient off FD by {rel:e}");
    assert!(
        rel_old > 10.0 * TOL_FD,
        "NEGATIVE CONTROL blind: {rel_old:e}"
    );
}

/// UHF HO2 RIJCOSX (fit off).
#[test]
#[ignore = "slow: 18 HO2 UHF SCFs"]
fn uhf_rijcosx_gradient_matches_fd_ho2_sto3g() {
    let basis = "sto-3g";
    let mol = mol_of(HO2, 2);
    let c = cfg(None, true, Some(cosx_exact()));
    let r = solve_u(&mol, basis, &c);
    let (bs, prep, bounds) = prep_of(&mol, basis);
    let op = Operator::coulomb();
    let g = unrestricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &c, &r).expect("grad");
    let g_k = uhf_gradient(&mol, &prep, op, &bounds, &r, None).expect("uhf exact-K");
    let g_fd = fd(&mol, &|m: &Molecule| solve_u(m, basis, &c).energy);
    let e = max_abs(&(&g - &g_fd));
    let e_k = max_abs(&(&g_k - &g_fd));
    println!("RIJCOSX UHF HO2/STO-3G: max|g - FD| = {e:.3e}; exact-K pairing {e_k:.3e}");
    assert!(e < TOL_FD, "UHF RIJCOSX gradient off FD by {e:e}");
    assert!(e_k > 10.0 * TOL_FD, "NEGATIVE CONTROL blind: {e_k:e}");
}

/// SCF-free anchor on the PRUNED grid: at a fixed density, the COSX exchange
/// gradient is the derivative of `tr[D K(D; R)]` (4-point stencil). Same bar as
/// the flat-grid anchor in `cosx_gradient.rs` (measured 7e-11 there).
#[test]
fn pruned_grid_exchange_gradient_is_the_derivative_at_fixed_density() {
    let basis = "6-31g";
    let mol = mol_of(WATER, 1);
    let r = solve_r(&mol, basis, &cfg(None, false, None));
    let d = r.density_r().clone();
    let c = sgx(cosx_exact(), 35);
    let t_of = |m: &Molecule| -> f64 {
        let (_, prep, _) = prep_of(m, basis);
        let ctx = ParallelContext::default();
        let mut kb = CosxK::new(&ctx, m, &prep, c.clone(), 0).expect("cosx");
        let mut k = Array2::zeros(d.dim());
        kb.build(&d, &mut k).expect("build");
        (&d * &k).sum()
    };
    let (_, prep, _) = prep_of(&mol, basis);
    let g = cosx_exchange_gradient(&mol, &prep, &c, &[(&d, 1.0)]).expect("grad");
    // Negative control: the FLAT (35,194) derivative at the same density.
    let mut flat = c.clone();
    flat.grid.prune = None;
    let g_flat = cosx_exchange_gradient(&mol, &prep, &flat, &[(&d, 1.0)]).expect("flat grad");
    let h = 2e-4;
    let mut worst = 0.0_f64;
    let mut worst_flat = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for comp in 0..3 {
            let f = |s: f64| t_of(&displaced(&mol, a, comp, s * h));
            let fdv = (8.0 * (f(1.0) - f(-1.0)) - (f(2.0) - f(-2.0))) / (12.0 * h);
            worst = worst.max((g[(a, comp)] - fdv).abs());
            worst_flat = worst_flat.max((g_flat[(a, comp)] - fdv).abs());
        }
    }
    println!(
        "pruned sgx (35,194): max|dT/dR - FD| = {worst:.3e}; flat-grid derivative against the \
         pruned FD {worst_flat:.3e}"
    );
    assert!(worst < TOL_FD, "pruned-grid dT/dR off FD by {worst:e}");
    assert!(
        worst_flat > 10.0 * TOL_FD,
        "NEGATIVE CONTROL blind: {worst_flat:e}"
    );
}

// ---------------------------------------------------------------------------
// Final pass
// ---------------------------------------------------------------------------

/// EXACTNESS ANCHOR: a final grid identical to the SCF grid reproduces the
/// SCF-grid energy BIT FOR BIT (RHF and UHF), and the record says so.
/// Fails if the pass reads a stale K (K of the previous density) or builds
/// with different knobs.
#[test]
fn final_pass_on_the_scf_grid_is_bit_identical() {
    let mol = mol_of(WATER, 1);
    let base = cfg(None, true, Some(CosxConfig::default()));
    let r0 = solve_r(&mol, "sto-3g", &base);
    assert!(r0.cosx_final.is_none());
    let mut same = base.clone();
    same.cosx.final_grid = Some(same.cosx.grid.clone());
    let r1 = solve_r(&mol, "sto-3g", &same);
    let rec = r1.cosx_final.expect("final pass ran");
    println!(
        "final==scf grid: E0 {:.15} E1 {:.15} rec {rec:?}",
        r0.energy, r1.energy
    );
    assert_eq!(r1.energy.to_bits(), r0.energy.to_bits());
    assert_eq!(rec.e_final.to_bits(), rec.e_scf_grid.to_bits());
    assert_eq!(rec.npts_final, rec.npts_scf);

    let ho2 = mol_of(HO2, 2);
    let base_u = cfg(None, false, Some(CosxConfig::default()));
    let u0 = solve_u(&ho2, "sto-3g", &base_u);
    let mut same_u = base_u.clone();
    same_u.cosx.final_grid = Some(same_u.cosx.grid.clone());
    let u1 = solve_u(&ho2, "sto-3g", &same_u);
    assert!(u1.cosx_final.is_some());
    assert_eq!(u1.energy.to_bits(), u0.energy.to_bits());
}

/// A LARGER final grid moves the energy towards exact exchange (water/6-31G
/// HF, coarse (30,50) SCF grid, (75,302) final grid), and the record carries
/// both energies. Bars from the measurement printed here.
#[test]
fn final_pass_on_a_larger_grid_moves_towards_exact_exchange() {
    let mol = mol_of(WATER, 1);
    let basis = "6-31g";
    let exact = solve_r(&mol, basis, &cfg(None, false, None)).energy;
    let mut coarse = cosx_exact();
    coarse.grid.n_radial = 30;
    coarse.grid.n_angular = 50;
    let mut fine = coarse.grid.clone();
    fine.n_radial = 75;
    fine.n_angular = 302;
    let mut c = cfg(None, false, Some(coarse));
    c.cosx.final_grid = Some(fine);
    let r = solve_r(&mol, basis, &c);
    let rec = r.cosx_final.expect("final pass");
    let e_scf = rec.e_scf_grid - exact;
    let e_fin = rec.e_final - exact;
    println!(
        "water/6-31G: E_scf_grid - E_exact = {e_scf:+.3e} ({} pts), E_final - E_exact = {e_fin:+.3e} ({} pts)",
        rec.npts_scf, rec.npts_final
    );
    assert_eq!(r.energy, rec.e_final);
    assert!(
        e_fin.abs() < 0.1 * e_scf.abs(),
        "final pass did not improve: {e_scf:e} -> {e_fin:e}"
    );
}

/// The geometry drivers run without the final pass (it is energy-only).
#[test]
fn geometry_drivers_drop_the_final_pass() {
    let mut c = cfg(None, true, Some(CosxConfig::default()));
    c.cosx.final_grid = Some(c.cosx.grid.clone());
    let g = gradient_task_config(&c);
    assert!(g.cosx.final_grid.is_none());
    let plain = cfg(None, true, Some(CosxConfig::default()));
    assert!(matches!(
        gradient_task_config(&plain),
        std::borrow::Cow::Borrowed(_)
    ));
}

/// MEASUREMENT behind what the gradient differentiates when the final pass
/// is on (`CosxFinalPass` doc). Water/6-31G HF, SCF grid sgx(35,194), final
/// grid sgx(50,302), FD of E_final (h = 1e-4). Two candidates:
///
/// (a) ferric: the exact gradient of the SCF-grid energy (what ships);
/// (b) ORCA-style: the same gradient formula evaluated on the FINAL grid at
///     the SCF-grid density (no orbital response for the grid change).
///
/// Prints `max|g - FD(E_final)|` for both, and `max|FD(E_final) -
/// FD(E_scf)|`, the size of the inconsistency (a) accepts by not pairing its
/// gradient with E_final. Asserts only that (a) is the derivative of E_scf.
#[test]
#[ignore = "measurement: 36 SCFs with a final pass"]
fn final_pass_gradient_options_measured() {
    let basis = "6-31g";
    let mol = mol_of(WATER, 1);
    for fit in [false, true] {
        let mut scf = sgx(cosx_exact(), 35);
        scf.overlap_fit = fit;
        let mut c = cfg(None, false, Some(scf.clone()));
        let mut fg = scf.grid.clone();
        fg.n_radial = 50;
        fg.n_angular = 302;
        c.cosx.final_grid = Some(fg.clone());
        let r = solve_r(&mol, basis, &c);
        let rec = r.cosx_final.expect("final pass");
        let (bs, prep, bounds) = prep_of(&mol, basis);
        let op = Operator::coulomb();
        let g_a = restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &c, &r).expect("a");
        let mut c_b = c.clone();
        c_b.cosx.grid = fg;
        c_b.cosx.final_grid = None;
        let g_b = restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &c_b, &r).expect("b");
        let cs = seeded(&c, r.density_r());
        let fd_final = fd(&mol, &|m: &Molecule| solve_r(m, basis, &cs).energy);
        let fd_scf = fd(&mol, &|m: &Molecule| {
            solve_r(m, basis, &cs).cosx_final.expect("pass").e_scf_grid
        });
        let ea = max_abs(&(&g_a - &fd_final));
        let eb = max_abs(&(&g_b - &fd_final));
        let ea_scf = max_abs(&(&g_a - &fd_scf));
        let gap = max_abs(&(&fd_final - &fd_scf));
        println!(
            "fit={fit}: E_final - E_scf = {:+.3e}; (a) |g_scf - FD(E_final)| = {ea:.3e}, \
             |g_scf - FD(E_scf)| = {ea_scf:.3e}; (b) |g_final_grid - FD(E_final)| = {eb:.3e}; \
             |FD(E_final) - FD(E_scf)| = {gap:.3e}",
            rec.e_final - rec.e_scf_grid
        );
        let tol = if fit { TOL_FIT } else { TOL_FD };
        assert!(
            ea_scf < tol,
            "(a) is not the derivative of E_scf: {ea_scf:e}"
        );
    }
}
