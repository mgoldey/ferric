//! The COSX exchange gradient differentiates the COSX energy.
//!
//! Before `cosx_gradient.rs`, every gradient path differentiated EXACT
//! exchange, so `k_builder = "cosx"` + optimize paired COSX energies with an
//! exact-K gradient (measured by FD on water: -8.9e-6 Ha/Bohr at STO-3G,
//! -1.4e-5 at cc-pVDZ; the Python prototype `scripts/cosx_gradient_proto.py`
//! saw 2e-5..3e-4 at a (30,110) grid).
//!
//! PRE-REGISTERED (from the prototype, before any of these ran in Rust):
//!
//! * `TOL_FD = 1e-7` Ha/Bohr. The prototype's analytic-vs-FD residual on the
//!   same molecule was <= 2.1e-9 with every one of the three gradient terms
//!   O(0.1) on its own, so 1e-7 is reachable by 50x and no missing term can
//!   hide under it. Central FD at h = 1e-4 Bohr: truncation O(h^2 E''') ~
//!   1e-9, SCF noise ~ E-noise / 2h with density_conv 1e-10.
//! * NEGATIVE CONTROL: the old pairing (COSX SCF + exact-K gradient) must miss
//!   FD by >= 10 x TOL_FD (prototype: 2e-5..3e-4, i.e. 200x..3000x), or the
//!   harness could not see the defect it exists to catch.
//! * ARTIFACT HYPOTHESIS: a sign/layout error in the libint2 derivative blocks,
//!   a missing home-atom (grid-riding) term or a missing weight response each
//!   give an O(1e-2..1) mismatch (prototype ablation: 7e-2..9e-1), grid- and
//!   basis-independent in character; a correct gradient leaves ~1e-9.
//!
//! OVERLAP FIT (the energy default; `fitted_*` tests below). The fitted energy
//! is not variational, so the gradient adds a Z-vector response
//! (`cosx_gradient::fitted_exchange_response`). PRE-REGISTERED from
//! `scripts/cosx_fit_gradient_proto.py` ((30,110) grid, water, FD h = 1e-4):
//!
//! * `TOL_FIT = 3e-8` Ha/Bohr: the prototype's residual WITH the response was
//!   1.7e-9 / 1.9e-9 (RHF water STO-3G / 6-31G) and 4.8e-9 (UHF HO2/STO-3G),
//!   and the Rust fit-off FD floor on the same systems 1.6e-9..3.7e-9 (3.3e-9
//!   HO2), so 3e-8 sits >= 6x above every measured floor.
//! * NEGATIVE CONTROL: WITHOUT the response (explicit derivative only, all of
//!   dQ included) the prototype missed FD by 1.06e-6 / 1.37e-6 (RHF) and
//!   7.2e-6 (UHF HO2) — >= 3.5x above 10 x TOL_FIT = 3e-7, so the assertion
//!   `e_noresp > 10 x TOL_FIT` is reachable and a dropped response cannot pass.
//! * The fitted tests use a (30,110) COSX grid on purpose: the fit's
//!   non-variational residual is a grid-error effect and shrinks on finer
//!   grids, which would erode the negative control's margin.
//!
//! Commands (tests are `#[ignore]`d where they run tens of SCFs):
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --test cosx_gradient
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --test cosx_gradient -- --include-ignored --nocapture

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_gradient::{
    check_fitted_ks_supported, check_gradient_supported, cosx_exchange_gradient,
    cosx_exchange_gradient_bilinear, cosx_exchange_gradient_bilinear_with_q,
    fitted_exchange_response_with_q, scf_exchange_is_cosx, FitQ, SpinOrbitals,
};
use ferric_scf::cosx_k::{CosxConfig, CosxHalfTransform, CosxK};
use ferric_scf::fock::KBuilder;
use ferric_scf::gradient::{
    build_energy_weighted_density, build_energy_weighted_density_uhf, oneelectron_gradient,
    preflight_cosx_restricted, restricted_scf_gradient, rhf_gradient, rhf_gradient_cosx,
    rhf_gradient_cosx_with_q, twoelectron_j_gradient, uhf_gradient, unrestricted_scf_gradient,
};
use ferric_scf::ks_gradient::{ks_gradient_closed, ks_gradient_closed_with_exchange};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use rayon::prelude::*;

const TOL_FD: f64 = 1e-7;
/// Fitted-gradient tolerance (see the module doc for its derivation).
const TOL_FIT: f64 = 3e-8;
const FD_H: f64 = 1e-4;

/// Distorted water (Angstrom): no symmetry, so every Cartesian component is
/// live and a component-swapping layout error cannot cancel.
const WATER: &str = "3
water, distorted
O   0.00  0.02  0.11
H   0.03  0.76 -0.47
H  -0.02 -0.75 -0.45
";

const OH: &str = "2
OH radical
O   0.00  0.00  0.00
H   0.05  0.02  0.97
";

/// HO2 radical (Å, ²A''): a non-degenerate open shell. OH's doubly
/// degenerate π SOMO lets each displaced UHF pick its own orientation against
/// the COSX grid, so FD on OH measures that choice (3.4e-4 here, the same for
/// the new and old gradients), not the gradient.
const HO2: &str = "3\nHO2\nO 0 0 0\nO 0 0 1.33\nH 0.05 0.92 1.6\n";

/// COSX with the screens OFF (dense half transforms, no pair screen) and the
/// overlap fit OFF: the energy the gradient differentiates EXACTLY.
fn cosx_exact() -> CosxConfig {
    CosxConfig {
        overlap_fit: false,
        screen_thresh: None,
        half_transform: CosxHalfTransform::Dense,
        ..CosxConfig::default()
    }
}

/// Production knobs (default screens, sparse half transforms, (50,110)) with
/// the fit OFF — the configuration a gradient user actually runs.
fn cosx_production_nofit() -> CosxConfig {
    CosxConfig {
        overlap_fit: false,
        ..CosxConfig::default()
    }
}

/// Tight SCF. `density_conv` is ferric's primary criterion; `energy_conv` is a
/// sanity bound (see `cosx_scf.rs::tight`), reachable here because nothing is
/// density-fitted.
fn scf_cfg(xc: Option<&str>, cosx: Option<CosxConfig>) -> RhfConfig {
    RhfConfig {
        k_builder: cosx.as_ref().map(|_| "cosx".to_string()),
        cosx: cosx.unwrap_or_default(),
        xc: xc.map(str::to_string),
        // Conventional four-centre J (and K when not COSX): the ONLY exchange
        // approximation under test is COSX.
        df_j_aux: xc.map(|_| String::new()),
        df_k_aux: xc.map(|_| String::new()),
        energy_conv: 1e-8,
        density_conv: 1e-10,
        max_iter: 300,
        ..Default::default()
    }
}

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

fn solve_restricted(mol: &Molecule, basis: &str, cfg: &RhfConfig) -> ScfResult {
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let r = solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, cfg).expect("rhf");
    assert!(r.converged, "SCF did not converge: {:?}", r.exit);
    r
}

/// (dispatched gradient, exact-K gradient) for a converged restricted SCF.
fn restricted_grads(
    mol: &Molecule,
    basis: &str,
    cfg: &RhfConfig,
    r: &ScfResult,
) -> (Array2<f64>, Array2<f64>) {
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let g_new = restricted_scf_gradient(mol, &prep, &bs, op, &bounds, cfg, r).expect("dispatch");
    let g_old = match cfg.xc.as_deref() {
        Some(xc) => ks_gradient_closed(mol, &prep, &bs, op, &bounds, xc, r, None).expect("ks"),
        None => rhf_gradient(mol, &prep, op, &bounds, r, None).expect("rhf"),
    };
    (g_new, g_old)
}

/// Central-FD gradient of the restricted SCF energy, every displaced SCF seeded
/// from the reference density (keeps all 2*3N solves in one basin).
fn fd_restricted(mol: &Molecule, basis: &str, cfg: &RhfConfig, seed: &Array2<f64>) -> Array2<f64> {
    let n = mol.atoms.len();
    let cfg = RhfConfig {
        init_guess_density: Some(seed.clone()),
        use_sad_guess: false,
        ..cfg.clone()
    };
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|a| (0..3).map(move |c| (a, c))).collect();
    let vals: Vec<f64> = pairs
        .par_iter()
        .map(|&(a, c)| {
            let ep = solve_restricted(&displaced(mol, a, c, FD_H), basis, &cfg).energy;
            let em = solve_restricted(&displaced(mol, a, c, -FD_H), basis, &cfg).energy;
            (ep - em) / (2.0 * FD_H)
        })
        .collect();
    let mut g = Array2::zeros((n, 3));
    for (&(a, c), v) in pairs.iter().zip(vals) {
        g[(a, c)] = v;
    }
    g
}

/// ANCHOR (SCF-free, the tightest and fastest check of `cosx_exchange_gradient`
/// alone): at a FIXED AO density `D`, `T(R) = tr[D K_COSX(D; R)]` is a smooth
/// function of the nuclear positions (grid, weights, AOs and 3c1e integrals all
/// move), and `cosx_exchange_gradient(.., [(D, 1)])` must be its derivative.
/// 4-point central stencil (h = 2e-4 Bohr): truncation O(h^4) ~ 1e-15 x T^(5),
/// round-off ~ 1e-13 / (12 h) x 18 ~ 1e-9.
///
/// Fails if reverted to nothing to test (the function would not exist); fails
/// with O(0.1) if any of the three terms (AO, ESP-integral, weight response) is
/// dropped or sign-flipped (prototype ablation 7e-2..9e-1), and with O(1) if the
/// libint2 derivative blocks are mis-assigned (bra/ket/charge).
#[test]
fn cosx_exchange_gradient_is_the_derivative_of_tr_dk_at_fixed_density() {
    for basis in ["sto-3g", "6-31g"] {
        let mol = mol_of(WATER, 1);
        let r = solve_restricted(&mol, basis, &scf_cfg(None, None));
        let d = r.density_r().clone();
        let cfg = cosx_exact();
        let t_of = |m: &Molecule| -> f64 {
            let bs = bundled(basis).expect("basis");
            let prep = PreparedBasis::new(m, &bs).expect("prep");
            let ctx = ParallelContext::default();
            let mut kb = CosxK::new(&ctx, m, &prep, cfg.clone(), 0).expect("cosx");
            let mut k = Array2::zeros(d.dim());
            kb.build(&d, &mut k).expect("build");
            (&d * &k).sum()
        };
        let bs = bundled(basis).expect("basis");
        let prep = PreparedBasis::new(&mol, &bs).expect("prep");
        let g = cosx_exchange_gradient(&mol, &prep, &cfg, &[(&d, 1.0)]).expect("grad");
        let h = 2e-4;
        let mut worst = 0.0_f64;
        for a in 0..mol.atoms.len() {
            for c in 0..3 {
                let f = |s: f64| t_of(&displaced(&mol, a, c, s * h));
                let fd = (8.0 * (f(1.0) - f(-1.0)) - (f(2.0) - f(-2.0))) / (12.0 * h);
                let e = (g[(a, c)] - fd).abs();
                println!(
                    "{basis} atom {a} comp {c}: analytic {:+.10e} fd {fd:+.10e} |diff| {e:.2e}",
                    g[(a, c)]
                );
                worst = worst.max(e);
            }
        }
        println!(
            "{basis}: max|dT/dR analytic - FD| = {worst:.3e} (T = {:.6})",
            t_of(&mol)
        );
        assert!(worst < TOL_FD, "{basis}: COSX dT/dR off FD by {worst:e}");
    }
}

/// SAME anchor with the PRODUCTION screens on (density-driven pair screen 1e-7,
/// sparse half transforms 1e-10): the gradient differentiates the UNSCREENED
/// energy, so this MEASURES the screen's footprint on the derivative. Bound
/// (derived, not fitted): the screen perturbs each G element by < 1e-7 x |X|
/// and T by ~1e-9 on water (K error 1.6e-10 at t = 1e-7, `cosx_screen_sweep`),
/// and a screen decision flipping between stencil points turns that into an FD
/// error <= 1.5 x dT / h ~ 1e-5. Printed so the actual number is on record.
#[test]
fn production_screens_move_the_fixed_density_derivative_by_less_than_their_bound() {
    let basis = "6-31g";
    let mol = mol_of(WATER, 1);
    let r = solve_restricted(&mol, basis, &scf_cfg(None, None));
    let d = r.density_r().clone();
    let cfg = cosx_production_nofit();
    let t_of = |m: &Molecule| -> f64 {
        let bs = bundled(basis).expect("basis");
        let prep = PreparedBasis::new(m, &bs).expect("prep");
        let ctx = ParallelContext::default();
        let mut kb = CosxK::new(&ctx, m, &prep, cfg.clone(), 0).expect("cosx");
        let mut k = Array2::zeros(d.dim());
        kb.build(&d, &mut k).expect("build");
        (&d * &k).sum()
    };
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let g = cosx_exchange_gradient(&mol, &prep, &cfg, &[(&d, 1.0)]).expect("grad");
    let h = 2e-4;
    let mut worst = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let f = |s: f64| t_of(&displaced(&mol, a, c, s * h));
            let fd = (8.0 * (f(1.0) - f(-1.0)) - (f(2.0) - f(-2.0))) / (12.0 * h);
            worst = worst.max((g[(a, c)] - fd).abs());
        }
    }
    println!("production screens: max|dT/dR (unscreened analytic) - FD(screened T)| = {worst:.3e}");
    assert!(
        worst < 1e-5,
        "screened-energy derivative off the unscreened gradient by {worst:e}"
    );
}

/// One restricted case: the new gradient vs FD of the COSX SCF energy, with the
/// old pairing as the negative control. For a functional, the residual is taken
/// RELATIVE to the same residual of a direct-K run (`r_cosx - r_direct`), which
/// cancels whatever the XC gradient itself leaves against FD and isolates the
/// exchange term; for HF the absolute residual is asserted too.
fn restricted_case(basis: &str, xc: Option<&str>, cosx: CosxConfig) {
    let mol = mol_of(WATER, 1);
    let label = format!("{basis}/{}", xc.unwrap_or("HF"));

    let cfg_c = scf_cfg(xc, Some(cosx));
    assert!(
        scf_exchange_is_cosx(&cfg_c, false).unwrap(),
        "{label}: COSX must be in effect"
    );
    let r_c = solve_restricted(&mol, basis, &cfg_c);
    let (g_new, g_old) = restricted_grads(&mol, basis, &cfg_c, &r_c);
    let fd_c = fd_restricted(&mol, basis, &cfg_c, r_c.density_r());
    let res_new = &g_new - &fd_c;
    let res_old = &g_old - &fd_c;

    let cfg_d = scf_cfg(xc, None);
    let r_d = solve_restricted(&mol, basis, &cfg_d);
    let (g_d, _) = restricted_grads(&mol, basis, &cfg_d, &r_d);
    let fd_d = fd_restricted(&mol, basis, &cfg_d, r_d.density_r());
    let res_d = &g_d - &fd_d;

    let rel_new = max_abs(&(&res_new - &res_d));
    let rel_old = max_abs(&(&res_old - &res_d));
    println!(
        "{label}: |E_cosx - E_direct| = {:.3e}; max|g_new - FD| = {:.3e}; max|g_exactK - FD| (old) = {:.3e}; \
         direct-run max|g - FD| = {:.3e}; exchange-isolated: new {rel_new:.3e}, old {rel_old:.3e}",
        (r_c.energy - r_d.energy).abs(),
        max_abs(&res_new),
        max_abs(&res_old),
        max_abs(&res_d),
    );
    // Fix reverted (exact-K gradient after a COSX SCF): rel_new == rel_old,
    // which the negative control below shows is >= 10 x TOL_FD.
    assert!(
        rel_new < TOL_FD,
        "{label}: COSX gradient off FD(COSX energy) by {rel_new:e}"
    );
    if xc.is_none() {
        assert!(
            max_abs(&res_new) < TOL_FD,
            "{label}: absolute residual {:e}",
            max_abs(&res_new)
        );
    }
    assert!(
        rel_old > 10.0 * TOL_FD,
        "{label}: NEGATIVE CONTROL blind — the old pairing is only {rel_old:e} off FD, the \
         harness cannot see the defect"
    );
}

#[test]
fn rhf_cosx_gradient_matches_fd_sto3g() {
    restricted_case("sto-3g", None, cosx_exact());
}

#[test]
#[ignore = "slow: 36 water/6-31G SCFs for two central-FD gradients"]
fn rhf_cosx_gradient_matches_fd_631g() {
    restricted_case("6-31g", None, cosx_exact());
}

#[test]
#[ignore = "slow: 36 B3LYP SCFs for two central-FD gradients"]
fn b3lyp_cosx_gradient_matches_fd_sto3g() {
    restricted_case("sto-3g", Some("B3LYP"), cosx_exact());
}

#[test]
#[ignore = "slow: 36 B3LYP/6-31G SCFs for two central-FD gradients"]
fn b3lyp_cosx_gradient_matches_fd_631g() {
    restricted_case("6-31g", Some("B3LYP"), cosx_exact());
}

/// The production knobs (default screens + sparse half transforms, fit off)
/// through the full SCF: the bar is the screen bound of the fixed-density test
/// (1e-5), not TOL_FD, because the SCF energy is then the SCREENED one.
#[test]
#[ignore = "slow: 36 SCFs; measures the production screens' footprint end to end"]
fn rhf_cosx_gradient_with_production_screens_sto3g() {
    let basis = "sto-3g";
    let mol = mol_of(WATER, 1);
    let cfg = scf_cfg(None, Some(cosx_production_nofit()));
    let r = solve_restricted(&mol, basis, &cfg);
    let (g_new, g_old) = restricted_grads(&mol, basis, &cfg, &r);
    let fd = fd_restricted(&mol, basis, &cfg, r.density_r());
    let e_new = max_abs(&(&g_new - &fd));
    let e_old = max_abs(&(&g_old - &fd));
    println!("production screens, water/STO-3G RHF: new {e_new:.3e}, old pairing {e_old:.3e}");
    assert!(
        e_new < 1e-5,
        "production-screen COSX gradient off FD by {e_new:e}"
    );
    assert!(
        e_old > 10.0 * e_new,
        "old pairing {e_old:e} not separated from new {e_new:e}"
    );
}

/// UHF: exchange `-1/2 sum_s tr[D_s K(D_s)]`, same term set with two densities.
#[test]
#[ignore = "slow: 18 UHF SCFs for a central-FD gradient"]
fn uhf_cosx_gradient_matches_fd_ho2_sto3g() {
    let basis = "sto-3g";
    let mol = mol_of(HO2, 2);
    let cfg = scf_cfg(None, Some(cosx_exact()));
    assert!(scf_exchange_is_cosx(&cfg, true).unwrap());
    let solve = |m: &Molecule| -> (ScfResult, PreparedBasis, SchwarzBounds) {
        let bs = bundled(basis).expect("basis");
        let prep = PreparedBasis::new(m, &bs).expect("prep");
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
        let r = solve_uhf(&ParallelContext::default(), m, &prep, &bounds, &cfg).expect("uhf");
        assert!(r.converged);
        (r, prep, bounds)
    };
    let (r, prep, bounds) = solve(&mol);
    let bs = bundled(basis).expect("basis");
    let op = Operator::coulomb();
    let g_new =
        unrestricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &cfg, &r).expect("uhf cosx grad");
    let g_old = uhf_gradient(&mol, &prep, op, &bounds, &r, None).expect("uhf grad");
    let n = mol.atoms.len();
    let mut fd = Array2::<f64>::zeros((n, 3));
    for a in 0..n {
        for c in 0..3 {
            let ep = solve(&displaced(&mol, a, c, FD_H)).0.energy;
            let em = solve(&displaced(&mol, a, c, -FD_H)).0.energy;
            fd[(a, c)] = (ep - em) / (2.0 * FD_H);
        }
    }
    let e_new = max_abs(&(&g_new - &fd));
    let e_old = max_abs(&(&g_old - &fd));
    println!("HO2/STO-3G UHF COSX: max|g_new - FD| = {e_new:.3e}; old pairing {e_old:.3e}");
    assert!(e_new < TOL_FD, "UHF COSX gradient off FD by {e_new:e}");
    assert!(
        e_old > 10.0 * TOL_FD,
        "NEGATIVE CONTROL blind: old pairing only {e_old:e} off"
    );
}

/// TRIVIAL-LIMIT ANCHOR: as the COSX grid is refined, the COSX gradient (each
/// at its own converged COSX SCF) must approach the exact-exchange gradient of
/// the direct SCF. An INDEPENDENT construction check — four-centre derivative
/// integrals vs grid + 3c1e derivatives — that a self-consistent FD test cannot
/// provide. Prototype (PySCF grid, water/6-31G): (20,50) 1.7e-3, (50,110)
/// 1.2e-4, (75,302) 5.3e-7 — monotone over these angular orders (NOT over every
/// grid: (30,110) -> (50,110) went 1.03e-4 -> 1.15e-4). Pre-registered: strictly
/// falling across (25,50) -> (50,110) -> (75,302) and >= 100x overall. A
/// gradient with a wrong term plateaus at O(0.1) instead.
#[test]
#[ignore = "slow: (75,302) COSX grid SCF + gradient"]
fn cosx_gradient_approaches_exact_exchange_gradient_as_the_grid_is_refined() {
    let basis = "6-31g";
    let mol = mol_of(WATER, 1);
    let cfg_d = scf_cfg(None, None);
    let r_d = solve_restricted(&mol, basis, &cfg_d);
    let (g_exact, _) = restricted_grads(&mol, basis, &cfg_d, &r_d);
    let mut errs = Vec::new();
    for (nr, na) in [(25usize, 50usize), (50, 110), (75, 302)] {
        let mut cosx = cosx_exact();
        cosx.grid.n_radial = nr;
        cosx.grid.n_angular = na;
        let cfg = scf_cfg(None, Some(cosx));
        let r = solve_restricted(&mol, basis, &cfg);
        let (g, _) = restricted_grads(&mol, basis, &cfg, &r);
        let e = max_abs(&(&g - &g_exact));
        println!(
            "({nr},{na}): max|g_cosx - g_exact| = {e:.3e}; |E_cosx - E_exact| = {:.3e}",
            (r.energy - r_d.energy).abs()
        );
        errs.push(e);
    }
    assert!(
        errs[1] < errs[0] && errs[2] < errs[1],
        "not monotone: {errs:?}"
    );
    assert!(
        errs[2] < errs[0] / 100.0,
        "fell only {:.1}x: {errs:?}",
        errs[0] / errs[2]
    );
}

/// EXACT-K PATHS UNCHANGED. The dispatchers and the `_with_exchange` entry point
/// must reproduce the historical calls BIT FOR BIT whenever COSX is not the
/// SCF's exchange — including `k_builder = "cosx"` under the KS RI-J
/// auto-default, where the SCF ignores COSX: there the SCF energy must also be
/// bit-identical to the no-cosx run, which is what ties `scf_exchange_is_cosx`
/// to the solver's own rule end to end.
#[test]
fn exact_k_gradients_are_bit_identical_through_the_dispatch() {
    let basis = "sto-3g";
    let mol = mol_of(WATER, 1);
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");

    // RHF, no k_builder.
    let cfg = scf_cfg(None, None);
    let r = solve_restricted(&mol, basis, &cfg);
    let a = restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &cfg, &r).unwrap();
    let b = rhf_gradient(&mol, &prep, op, &bounds, &r, None).unwrap();
    assert!(a == b, "RHF dispatch differs from rhf_gradient");

    // B3LYP with the default DF auto-default AND k_builder = "cosx": the SCF
    // ignores COSX (RI-J/RI-K active), so must the gradient.
    // Loose-ish convergence on purpose: this compares two runs bit for bit, and
    // under DF the energy random-walks at ~1e-8 (cosx_scf.rs::tight), so a
    // tighter energy_conv would make convergence a coin flip.
    let plain = RhfConfig {
        xc: Some("B3LYP".into()),
        energy_conv: 1e-6,
        density_conv: 1e-7,
        max_iter: 300,
        ..Default::default()
    };
    let with_cosx = RhfConfig {
        k_builder: Some("cosx".into()),
        cosx: cosx_production_nofit(),
        ..plain.clone()
    };
    assert!(!scf_exchange_is_cosx(&with_cosx, false).unwrap());
    let r_plain = solve_restricted(&mol, basis, &plain);
    let r_cosx = solve_restricted(&mol, basis, &with_cosx);
    assert!(
        r_plain.energy.to_bits() == r_cosx.energy.to_bits(),
        "SCF did not ignore COSX under DF: {} vs {} — scf_exchange_is_cosx disagrees with solve_rhf",
        r_plain.energy,
        r_cosx.energy
    );
    let a = restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &with_cosx, &r_cosx).unwrap();
    let b = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, "B3LYP", &r_cosx, None).unwrap();
    let c = ks_gradient_closed_with_exchange(
        &mol, &prep, &bs, op, &bounds, "B3LYP", &r_cosx, None, None,
    )
    .unwrap();
    assert!(
        a == b && b == c,
        "KS dispatch / _with_exchange(None) differ from ks_gradient_closed"
    );
}

/// REFUSALS, never a silent approximate gradient. The overlap fit itself is
/// now SUPPORTED for Hartree-Fock (RHF/UHF, Z-vector); what stays refused:
/// fitted COSX with a KS functional (its Z-vector needs the XC Fock-matrix
/// nuclear derivative, which ferric lacks), pruned COSX grids, UKS, ROHF/ROKS.
///
/// Fails if the fitted-HF refusal is restored (the first two asserts), or if
/// the KS/pruned/UKS/ROHF refusals are dropped.
#[test]
fn unsupported_cosx_gradients_are_refused() {
    // The default (fitted) config is accepted by the HF-level check ...
    check_gradient_supported(&CosxConfig::default()).expect("fitted COSX is supported");
    assert!(check_fitted_ks_supported(&CosxConfig::default(), None).is_ok());
    // ... but refused for a functional, and fit-off KS stays accepted.
    let err = check_fitted_ks_supported(&CosxConfig::default(), Some("B3LYP")).unwrap_err();
    assert!(format!("{err}").contains("overlap_fit"), "{err}");
    assert!(check_fitted_ks_supported(&cosx_exact(), Some("B3LYP")).is_ok());

    // A pruned COSX grid has no weight response.
    let mut pruned = cosx_exact();
    pruned.grid.prune = Some(ferric_dft::prune::PruneScheme::NwchemLike);
    let err = check_gradient_supported(&pruned).unwrap_err();
    assert!(format!("{err}").contains("pruned"), "{err}");

    let basis = "sto-3g";
    let mol = mol_of(WATER, 1);
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let r = solve_restricted(&mol, basis, &scf_cfg(Some("B3LYP"), None));

    // Fitted COSX + B3LYP: the KS gradient entry point and the pre-SCF
    // preflight both refuse, naming the fit.
    let err = ks_gradient_closed_with_exchange(
        &mol,
        &prep,
        &bs,
        op,
        &bounds,
        "B3LYP",
        &r,
        None,
        Some(&CosxConfig::default()),
    )
    .unwrap_err();
    assert!(format!("{err}").contains("overlap_fit"), "{err}");
    let fitted_ks = scf_cfg(Some("B3LYP"), Some(CosxConfig::default()));
    assert!(scf_exchange_is_cosx(&fitted_ks, false).unwrap());
    let err = preflight_cosx_restricted(&fitted_ks).unwrap_err();
    assert!(format!("{err}").contains("overlap_fit"), "{err}");
    // The fitted HF preflight passes.
    preflight_cosx_restricted(&scf_cfg(None, Some(CosxConfig::default())))
        .expect("fitted HF passes the preflight");

    // UKS with COSX in effect is refused before any gradient work.
    let uks = RhfConfig {
        k_builder: Some("cosx".into()),
        cosx: cosx_exact(),
        xc: Some("B3LYP".into()),
        ..Default::default()
    };
    assert!(scf_exchange_is_cosx(&uks, true).unwrap());
    let oh = mol_of(OH, 2);
    let prep_oh = PreparedBasis::new(&oh, &bs).expect("prep");
    let bounds_oh = SchwarzBounds::compute(op, &prep_oh).expect("schwarz");
    let r_oh = solve_uhf(
        &ParallelContext::default(),
        &oh,
        &prep_oh,
        &bounds_oh,
        &scf_cfg(None, None),
    )
    .expect("uhf");
    let err =
        unrestricted_scf_gradient(&oh, &prep_oh, &bs, op, &bounds_oh, &uks, &r_oh).unwrap_err();
    assert!(format!("{err}").contains("UKS"), "{err}");
    assert!(
        ferric_scf::gradient::refuse_cosx_restricted_open(&RhfConfig {
            k_builder: Some("cosx".into()),
            ..Default::default()
        })
        .is_err()
    );
}

/// The open-shell solvers treat `df_j_aux = Some("")` / `df_k_aux = Some("")`
/// as the exact-J/K SENTINEL (`fock_assembly::build_df_jk` filters empty names),
/// so COSX stays the exchange builder and the gradient must be the COSX one.
///
/// Mutation: restoring the old `df_j_aux.is_some() || df_k_aux.is_some()`
/// open-shell predicate makes the first two asserts fail (it reported "not
/// COSX", and `unrestricted_scf_gradient` then paired an exact-K gradient with
/// a COSX energy). The last two asserts keep a real DF name switching COSX off.
#[test]
fn open_shell_empty_df_name_keeps_cosx_active() {
    let base = RhfConfig {
        k_builder: Some("cosx".into()),
        cosx: cosx_exact(),
        ..Default::default()
    };
    let empty_j = RhfConfig {
        df_j_aux: Some(String::new()),
        ..base.clone()
    };
    let empty_jk = RhfConfig {
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..base.clone()
    };
    assert!(scf_exchange_is_cosx(&empty_j, true).unwrap());
    assert!(scf_exchange_is_cosx(&empty_jk, true).unwrap());
    let named_j = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        ..base.clone()
    };
    let named_k = RhfConfig {
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..base
    };
    assert!(!scf_exchange_is_cosx(&named_j, true).unwrap());
    assert!(!scf_exchange_is_cosx(&named_k, true).unwrap());
}

// ---------------------------------------------------------------------------
// Overlap-fitted COSX (the energy default): Z-vector gradient.
// ---------------------------------------------------------------------------

/// Fitted COSX, screens off, on the (30,110) grid the prototype measured (see
/// the module doc for why not the (50,110) default).
fn cosx_fitted() -> CosxConfig {
    let mut c = CosxConfig {
        overlap_fit: true,
        screen_thresh: None,
        half_transform: CosxHalfTransform::Dense,
        ..CosxConfig::default()
    };
    c.grid.n_radial = 30;
    c.grid.n_angular = 110;
    c
}

/// ANCHOR (SCF-free) for the fitted explicit derivative and the general
/// bilinear kernel: at FIXED AO matrices `Y != B`, `T(R) = tr[Y K_f(B; R)]`
/// (`K_f` from the production fitted builder, dense, unscreened) must have
/// `cosx_exchange_gradient_bilinear(.., [(Y, B, 1)])` as its derivative. This
/// is the only check of the `dQ` terms (`dS` and `dS_num`) independent of the
/// Z-vector, and of the `H != F` kernel. Same stencil and floor as the fit-off
/// fixed-density anchor (measured 7e-11 there).
///
/// Fails at O(1e-3..1e-1) if the `dS` or `dS_num` term is dropped or has its
/// sign flipped, or if `M = Y Q` is replaced by `Q Y` (Q is not symmetric).
#[test]
fn fitted_explicit_derivative_is_the_derivative_of_tr_y_kf_b_at_fixed_matrices() {
    let basis = "sto-3g";
    let mol = mol_of(WATER, 1);
    let r = solve_restricted(&mol, basis, &scf_cfg(None, None));
    let b = r.density_r().clone();
    // A symmetric Y that is NOT B (so Y Q != B Q and H != F).
    let y = {
        let m = b.dot(&b) * 0.3 + Array2::<f64>::eye(b.nrows()) * 0.2;
        (&m + &m.t()) * 0.5
    };
    let cfg = cosx_fitted();
    let t_of = |m: &Molecule| -> f64 {
        let bs = bundled(basis).expect("basis");
        let prep = PreparedBasis::new(m, &bs).expect("prep");
        let ctx = ParallelContext::default();
        let mut kb = CosxK::new(&ctx, m, &prep, cfg.clone(), 0).expect("cosx");
        let mut k = Array2::zeros(b.dim());
        kb.build(&b, &mut k).expect("build");
        (&y * &k).sum()
    };
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let g = cosx_exchange_gradient_bilinear(&mol, &prep, &cfg, &[(&y, &b, 1.0)]).expect("grad");
    let h = 2e-4;
    let mut worst = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let f = |s: f64| t_of(&displaced(&mol, a, c, s * h));
            let fd = (8.0 * (f(1.0) - f(-1.0)) - (f(2.0) - f(-2.0))) / (12.0 * h);
            worst = worst.max((g[(a, c)] - fd).abs());
        }
    }
    println!("fitted: max|d tr[Y K_f(B)]/dR analytic - FD| = {worst:.3e}");
    assert!(
        worst < TOL_FD,
        "fitted explicit derivative off FD by {worst:e}"
    );
}

/// EXACTNESS ANCHOR of the fitted path: its trivial limit `Q = I`
/// (`FitQ::Identity`) makes `K_f = L`, so `Delta`, the right-hand side and the
/// Z-vector are EXACTLY zero, and the whole Lagrangian assembly (response `W`,
/// `D + Zs` one-electron density, response Coulomb, general bilinear exchange
/// kernel) must reproduce the fit-off gradient. Prototype: 2e-16.
///
/// Blind spot (stated, per the anchor-blind-spots rule): with `z = 0` this
/// cannot see an error in a term PROPORTIONAL to `z` (e.g. a wrong factor on
/// `dT(Zs, D)` or on `sym(C_v z e_o C_o^T)`); those are covered by the FD tests
/// and their no-response negative controls. It does catch a wrong occupied
/// weight or a missing `e_o` in `W`, and a general-kernel error.
#[test]
fn fitted_q_identity_anchor_recovers_the_plain_gradient() {
    let basis = "sto-3g";
    let mol = mol_of(WATER, 1);
    let cfg = cosx_exact(); // fit OFF: the SCF energy whose gradient must come back
    let r = solve_restricted(&mol, basis, &scf_cfg(None, Some(cfg.clone())));
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");

    let resp = fitted_exchange_response_with_q(
        &mol,
        &prep,
        &bounds,
        &cfg,
        &[SpinOrbitals {
            c: r.mos_r(),
            eps: r.eps_r(),
            nocc: (mol.nelec() / 2) as usize,
            d: r.density_r(),
        }],
        1.0,
        FitQ::Identity,
    )
    .expect("identity response");
    assert!(
        resp.zs[0].iter().all(|v| *v == 0.0),
        "Q = I must give z = 0 exactly"
    );
    assert_eq!(resp.iterations, 0);
    let w_ref = build_energy_weighted_density(&r, (mol.nelec() / 2) as usize);
    let dw = max_abs(&(&resp.w - &w_ref));
    assert!(
        dw < 1e-12,
        "Q = I Lagrangian W differs from the plain W by {dw:e}"
    );

    let d = r.density_r();
    let fast = cosx_exchange_gradient(&mol, &prep, &cfg, &[(d, 1.0)]).expect("fast");
    let general =
        cosx_exchange_gradient_bilinear_with_q(&mol, &prep, &cfg, &[(d, d, 1.0)], FitQ::Identity)
            .expect("general");
    let dk = max_abs(&(&fast - &general));
    assert!(
        dk < 1e-11,
        "general (H != F) kernel differs from the H = F path by {dk:e}"
    );

    let g_plain = rhf_gradient_cosx(&mol, &prep, op, &bounds, &r, None, &cfg).expect("plain");
    let g_id = rhf_gradient_cosx_with_q(&mol, &prep, op, &bounds, &r, None, &cfg, FitQ::Identity)
        .expect("identity");
    let dg = max_abs(&(&g_plain - &g_id));
    println!("Q = I anchor: |W - W_plain| {dw:.2e}, kernel {dk:.2e}, gradient {dg:.2e}");
    assert!(
        dg < 1e-10,
        "Q = I fitted gradient differs from the plain one by {dg:e}"
    );
}

/// Explicit-only fitted RHF gradient (the pre-Z-vector construction): the
/// NEGATIVE CONTROL of the fitted FD tests.
fn fitted_rhf_explicit_only(
    mol: &Molecule,
    basis: &str,
    cfg: &CosxConfig,
    r: &ScfResult,
) -> Array2<f64> {
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let d = r.density_r();
    let w = build_energy_weighted_density(r, (mol.nelec() / 2) as usize);
    let mut g = oneelectron_gradient(mol, &prep, d, &w, None).expect("1e");
    g += &twoelectron_j_gradient(&prep, op, &bounds, d).expect("J");
    g += &cosx_exchange_gradient(mol, &prep, cfg, &[(d, -0.25)]).expect("K");
    g
}

/// Fitted RHF: the Z-vector gradient vs central FD of the FITTED SCF energy,
/// and the explicit-only gradient as the negative control. Fails if the fix is
/// reverted (the fitted gradient is refused, `expect` panics) or if the
/// response is dropped (`e_new` rises to the negative control's ~1e-6).
fn fitted_restricted_case(basis: &str) {
    let mol = mol_of(WATER, 1);
    let cfg = scf_cfg(None, Some(cosx_fitted()));
    assert!(scf_exchange_is_cosx(&cfg, false).unwrap());
    let r = solve_restricted(&mol, basis, &cfg);
    let (g_new, g_exact_k) = restricted_grads(&mol, basis, &cfg, &r);
    let g_nr = fitted_rhf_explicit_only(&mol, basis, &cfg.cosx, &r);
    let fd = fd_restricted(&mol, basis, &cfg, r.density_r());
    let e_new = max_abs(&(&g_new - &fd));
    let e_nr = max_abs(&(&g_nr - &fd));
    let e_k = max_abs(&(&g_exact_k - &fd));
    println!(
        "fitted RHF water/{basis}: max|g - FD| = {e_new:.3e}; without response {e_nr:.3e} \
         ({:.0}x); exact-K pairing {e_k:.3e}",
        e_nr / e_new
    );
    assert!(
        e_new < TOL_FIT,
        "fitted RHF/{basis}: gradient off FD by {e_new:e}"
    );
    assert!(
        e_nr > 10.0 * TOL_FIT,
        "fitted RHF/{basis}: NEGATIVE CONTROL blind — the explicit-only gradient is only \
         {e_nr:e} off FD"
    );
}

#[test]
fn fitted_rhf_cosx_gradient_matches_fd_sto3g() {
    fitted_restricted_case("sto-3g");
}

#[test]
#[ignore = "slow: 18 water/6-31G fitted-COSX SCFs for a central-FD gradient"]
fn fitted_rhf_cosx_gradient_matches_fd_631g() {
    fitted_restricted_case("6-31g");
}

/// Fitted UHF (HO2, non-degenerate — see `HO2`): per-spin Z-vector gradient vs
/// FD of the fitted UHF energy, explicit-only as the negative control.
#[test]
#[ignore = "slow: 18 HO2 fitted-COSX UHF SCFs for a central-FD gradient"]
fn fitted_uhf_cosx_gradient_matches_fd_ho2_sto3g() {
    let basis = "sto-3g";
    let mol = mol_of(HO2, 2);
    let cfg = scf_cfg(None, Some(cosx_fitted()));
    assert!(scf_exchange_is_cosx(&cfg, true).unwrap());
    let solve = |m: &Molecule| -> (ScfResult, PreparedBasis, SchwarzBounds) {
        let bs = bundled(basis).expect("basis");
        let prep = PreparedBasis::new(m, &bs).expect("prep");
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
        let r = solve_uhf(&ParallelContext::default(), m, &prep, &bounds, &cfg).expect("uhf");
        assert!(r.converged);
        (r, prep, bounds)
    };
    let (r, prep, bounds) = solve(&mol);
    let bs = bundled(basis).expect("basis");
    let op = Operator::coulomb();
    let g_new =
        unrestricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &cfg, &r).expect("fitted uhf");
    // Negative control: explicit-only (no Z-vector).
    let nelec = mol.nelec() as usize;
    let (na, nb) = (nelec.div_ceil(2), (nelec - 1) / 2);
    let d_a = &r.density_alpha;
    let d_b = r.density_beta.as_ref().expect("beta");
    let d_t = d_a + d_b;
    let w = build_energy_weighted_density_uhf(&r, na, nb);
    let mut g_nr = oneelectron_gradient(&mol, &prep, &d_t, &w, None).expect("1e");
    g_nr += &twoelectron_j_gradient(&prep, op, &bounds, &d_t).expect("J");
    g_nr +=
        &cosx_exchange_gradient(&mol, &prep, &cfg.cosx, &[(d_a, -0.5), (d_b, -0.5)]).expect("K");
    let n = mol.atoms.len();
    let mut fd = Array2::<f64>::zeros((n, 3));
    for a in 0..n {
        for c in 0..3 {
            let ep = solve(&displaced(&mol, a, c, FD_H)).0.energy;
            let em = solve(&displaced(&mol, a, c, -FD_H)).0.energy;
            fd[(a, c)] = (ep - em) / (2.0 * FD_H);
        }
    }
    let e_new = max_abs(&(&g_new - &fd));
    let e_nr = max_abs(&(&g_nr - &fd));
    println!("fitted UHF HO2/STO-3G: max|g - FD| = {e_new:.3e}; without response {e_nr:.3e}");
    assert!(e_new < TOL_FIT, "fitted UHF gradient off FD by {e_new:e}");
    assert!(
        e_nr > 10.0 * TOL_FIT,
        "NEGATIVE CONTROL blind: explicit-only fitted UHF only {e_nr:e} off FD"
    );
}
