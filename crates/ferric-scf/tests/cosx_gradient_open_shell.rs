//! COSX exchange gradient for UKS and ROHF/ROKS references (fit off).
//!
//! The open-shell COSX exchange energy is `-1/2 c_x sum_s tr[D_s K_COSX(D_s)]`;
//! fit off it is a quadratic form whose `D_s`-derivative is the `K_COSX(D_s)`
//! the SCF put in its spin Focks, so the UKS / ROKS energy stays variational
//! and the gradient is the exact-K one with the four-centre exchange
//! derivative replaced by `cosx_exchange_gradient([(D_a, -c_x/2), (D_b,
//! -c_x/2)])` — the per-spin form `uhf_gradient_cosx` already uses.
//!
//! PRE-REGISTERED (before any of these ran):
//!
//! * PHYSICS: the dispatched gradient matches central FD of the COSX SCF
//!   energy on the SAME grid. The exchange-isolated residual (the COSX run's
//!   residual minus the same functional's direct-K run's residual, which
//!   cancels what the XC gradient itself leaves against FD) is at the level
//!   the closed-shell B3LYP and UHF HO2 cases reached (<= 1e-8), far under
//!   `TOL_FD = 1e-7`.
//! * ARTIFACT: a missing per-spin factor (`-c_x/4` instead of `-c_x/2`), a
//!   dropped spin or the exact-K exchange derivative (the pre-fix pairing)
//!   leaves the COSX-vs-exact exchange gradient difference, O(1e-5..1e-4) on
//!   these grids (closed-shell water measured 2e-5..3e-4) — the NEGATIVE
//!   CONTROL asserts the exact-K pairing misses by >= 10 x TOL_FD.
//! * ANCHOR (no FD): a closed-shell RKS COSX result relabelled as UKS (α = β)
//!   or ROKS (no open shell) must give the RKS COSX gradient to rounding.
//!
//! Commands:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test cosx_gradient_open_shell
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test cosx_gradient_open_shell \
//!     -- --include-ignored --nocapture

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_gradient::scf_exchange_is_cosx;
use ferric_scf::cosx_k::{CosxConfig, CosxHalfTransform};
use ferric_scf::gradient::{
    restricted_open_scf_gradient, restricted_scf_gradient, rohf_gradient, unrestricted_scf_gradient,
};
use ferric_scf::ks_gradient::{
    ks_gradient_closed_with_exchange, ks_gradient_roks, ks_gradient_roks_with_exchange,
    ks_gradient_uks, ks_gradient_uks_with_exchange,
};
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf_with_guess;
use ndarray::Array2;
use rayon::prelude::*;

const TOL_FD: f64 = 1e-7;
const FD_H: f64 = 1e-4;
/// Bar of the relabelled closed-shell anchors (absolute, Ha/Bohr).
const ANCHOR_TOL: f64 = 1e-11;

/// HO2 radical (Å, ²A''): non-degenerate open shell, no symmetry.
const HO2: &str = "3\nHO2\nO 0 0 0\nO 0 0 1.33\nH 0.05 0.92 1.6\n";
/// NH2 radical (Å, ²B1), slightly distorted so every component is live.
const NH2: &str = "3\nNH2\nN 0.00 0.01 0.142\nH 0.02 0.802 -0.497\nH -0.01 -0.79 -0.48\n";
/// Distorted water (Å) for the closed-shell anchors.
const WATER: &str = "3\nwater\nO 0.00 0.02 0.11\nH 0.03 0.76 -0.47\nH -0.02 -0.75 -0.45\n";

/// Fit-off, unscreened COSX on a flat (50,110) grid: the energy the gradient
/// differentiates exactly.
fn cosx_flat() -> CosxConfig {
    CosxConfig {
        overlap_fit: false,
        screen_thresh: None,
        half_transform: CosxHalfTransform::Dense,
        grid: ferric_dft::grid::AtomicGridConfig {
            n_radial: 50,
            n_angular: 110,
            prune: None,
        },
        final_grid: None,
        ..CosxConfig::flat_reference()
    }
}

/// The same on the pruned `sgx` grid (peak 194, ORCA/PySCF regions).
fn cosx_pruned() -> CosxConfig {
    let mut c = cosx_flat();
    c.grid.n_radial = 35;
    c.grid.n_angular = 194;
    c.grid.prune = Some(ferric_dft::prune::PruneScheme::Sgx);
    c
}

/// Tight SCF, four-centre J (and K when not COSX): COSX is the only exchange
/// approximation under test.
fn scf_cfg(xc: Option<&str>, cosx: Option<CosxConfig>) -> RhfConfig {
    RhfConfig {
        k_builder: cosx.as_ref().map(|_| "cosx".to_string()),
        cosx: cosx.unwrap_or_default(),
        xc: xc.map(str::to_string),
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        energy_conv: 1e-9,
        density_conv: 1e-10,
        max_iter: 300,
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug)]
enum Reference {
    Uks,
    Roks,
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
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

fn setup(
    mol: &Molecule,
    basis: &str,
) -> (ferric_core::basis::BasisSet, PreparedBasis, SchwarzBounds) {
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(mol, &bs).expect("prep");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
    (bs, prep, bounds)
}

/// Converged open-shell SCF, optionally seeded from `guess` (keeps all FD
/// solves on one state).
fn solve(
    reference: Reference,
    mol: &Molecule,
    basis: &str,
    cfg: &RhfConfig,
    guess: Option<&ScfResult>,
) -> ScfResult {
    let (_, prep, bounds) = setup(mol, basis);
    let ctx = ParallelContext::default();
    let r = match reference {
        Reference::Uks => solve_uhf_with_guess(
            &ctx,
            mol,
            &prep,
            &bounds,
            cfg,
            guess.map(|g| (&g.mos_alpha, g.mos_beta.as_ref().expect("beta MOs"))),
        ),
        Reference::Roks => {
            let mut c = cfg.clone();
            if let Some(g) = guess {
                c.init_guess_density = Some(g.density_total.clone());
                c.use_sad_guess = false;
            }
            solve_rohf(&ctx, mol, &prep, Operator::coulomb(), &bounds, &c)
        }
    }
    .expect("scf");
    assert!(
        r.converged,
        "{reference:?} SCF did not converge: {:?}",
        r.exit
    );
    r
}

/// (dispatched gradient, exact-K gradient) for a converged open-shell result.
fn grads(
    reference: Reference,
    mol: &Molecule,
    basis: &str,
    cfg: &RhfConfig,
    r: &ScfResult,
) -> (Array2<f64>, Array2<f64>) {
    let (bs, prep, bounds) = setup(mol, basis);
    let op = Operator::coulomb();
    match reference {
        Reference::Uks => {
            let g = unrestricted_scf_gradient(mol, &prep, &bs, op, &bounds, cfg, r).expect("grad");
            let g_k = match cfg.xc.as_deref() {
                Some(xc) => ks_gradient_uks(mol, &prep, &bs, op, &bounds, xc, r, None),
                None => ferric_scf::gradient::uhf_gradient(mol, &prep, op, &bounds, r, None),
            }
            .expect("exact-K grad");
            (g, g_k)
        }
        Reference::Roks => {
            let g =
                restricted_open_scf_gradient(mol, &prep, &bs, op, &bounds, cfg, r).expect("grad");
            let g_k = match cfg.xc.as_deref() {
                Some(xc) => ks_gradient_roks(mol, &prep, &bs, op, &bounds, xc, r, None),
                None => rohf_gradient(mol, &prep, op, &bounds, r, None),
            }
            .expect("exact-K grad");
            (g, g_k)
        }
    }
}

/// Central FD of the open-shell SCF energy, every solve seeded from `r`.
fn fd(
    reference: Reference,
    mol: &Molecule,
    basis: &str,
    cfg: &RhfConfig,
    r: &ScfResult,
) -> Array2<f64> {
    let n = mol.atoms.len();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|a| (0..3).map(move |c| (a, c))).collect();
    let vals: Vec<f64> = pairs
        .par_iter()
        .map(|&(a, c)| {
            let e = |s: f64| {
                solve(
                    reference,
                    &displaced(mol, a, c, s * FD_H),
                    basis,
                    cfg,
                    Some(r),
                )
                .energy
            };
            (e(1.0) - e(-1.0)) / (2.0 * FD_H)
        })
        .collect();
    let mut g = Array2::zeros((n, 3));
    for (&(a, c), v) in pairs.iter().zip(vals) {
        g[(a, c)] = v;
    }
    g
}

/// One open-shell case: the COSX gradient vs FD of the COSX energy (same
/// grid), exchange-isolated against the direct-K run of the same functional,
/// with the exact-K pairing as the negative control. Returns
/// (exchange-isolated residual, absolute residual, exact-K pairing residual).
fn open_case(
    label: &str,
    reference: Reference,
    xyz: &str,
    basis: &str,
    xc: Option<&str>,
    cosx: CosxConfig,
) -> (f64, f64, f64) {
    let mol = Molecule::parse_xyz(xyz, 0, 2).expect("xyz");
    let cfg_c = scf_cfg(xc, Some(cosx));
    assert!(
        scf_exchange_is_cosx(&cfg_c, true).unwrap(),
        "{label}: COSX must be in effect"
    );
    let r_c = solve(reference, &mol, basis, &cfg_c, None);
    let (g_new, g_old) = grads(reference, &mol, basis, &cfg_c, &r_c);
    let fd_c = fd(reference, &mol, basis, &cfg_c, &r_c);

    let cfg_d = scf_cfg(xc, None);
    let r_d = solve(reference, &mol, basis, &cfg_d, Some(&r_c));
    let (g_d, _) = grads(reference, &mol, basis, &cfg_d, &r_d);
    let fd_d = fd(reference, &mol, basis, &cfg_d, &r_d);
    let res_d = &g_d - &fd_d;

    let res_new = &g_new - &fd_c;
    let rel_new = max_abs(&(&res_new - &res_d));
    let rel_old = max_abs(&(&(&g_old - &fd_c) - &res_d));
    let abs_new = max_abs(&res_new);
    println!(
        "{label}: |E_cosx - E_direct| = {:.3e}; max|g| = {:.3e}; max|g_cosx - FD| = {abs_new:.3e}; \
         direct-run max|g - FD| = {:.3e}; exchange-isolated: new {rel_new:.3e}, exact-K pairing \
         {rel_old:.3e}; translation max|sum_A g| = {:.1e}",
        (r_c.energy - r_d.energy).abs(),
        max_abs(&g_new),
        max_abs(&res_d),
        (0..3).map(|k| g_new.column(k).sum().abs()).fold(0.0_f64, f64::max),
    );
    assert!(
        rel_new < TOL_FD,
        "{label}: COSX gradient off FD(COSX energy) by {rel_new:e}"
    );
    assert!(
        rel_old > 10.0 * TOL_FD,
        "{label}: NEGATIVE CONTROL blind — the exact-K pairing is only {rel_old:e} off FD"
    );
    (rel_new, abs_new, rel_old)
}

#[test]
fn uks_b3lyp_cosx_gradient_matches_fd_nh2_flat() {
    open_case(
        "NH2/STO-3G UKS-B3LYP flat(50,110)",
        Reference::Uks,
        NH2,
        "sto-3g",
        Some("B3LYP"),
        cosx_flat(),
    );
}

#[test]
#[ignore = "slow: 36 HO2 UKS-B3LYP SCFs for two central-FD gradients"]
fn uks_b3lyp_cosx_gradient_matches_fd_ho2_flat() {
    open_case(
        "HO2/STO-3G UKS-B3LYP flat(50,110)",
        Reference::Uks,
        HO2,
        "sto-3g",
        Some("B3LYP"),
        cosx_flat(),
    );
}

#[test]
#[ignore = "slow: 36 NH2 UKS-B3LYP SCFs on the pruned grid"]
fn uks_b3lyp_cosx_gradient_matches_fd_nh2_pruned() {
    open_case(
        "NH2/STO-3G UKS-B3LYP sgx(35,194)",
        Reference::Uks,
        NH2,
        "sto-3g",
        Some("B3LYP"),
        cosx_pruned(),
    );
}

#[test]
#[ignore = "slow: 36 HO2 UKS-B3LYP SCFs on the pruned grid"]
fn uks_b3lyp_cosx_gradient_matches_fd_ho2_pruned() {
    open_case(
        "HO2/STO-3G UKS-B3LYP sgx(35,194)",
        Reference::Uks,
        HO2,
        "sto-3g",
        Some("B3LYP"),
        cosx_pruned(),
    );
}

#[test]
#[ignore = "slow: 36 HO2 ROKS-B3LYP SCFs for two central-FD gradients"]
fn roks_b3lyp_cosx_gradient_matches_fd_ho2_flat() {
    open_case(
        "HO2/STO-3G ROKS-B3LYP flat(50,110)",
        Reference::Roks,
        HO2,
        "sto-3g",
        Some("B3LYP"),
        cosx_flat(),
    );
}

#[test]
#[ignore = "slow: 36 HO2 ROKS-B3LYP SCFs on the pruned grid"]
fn roks_b3lyp_cosx_gradient_matches_fd_ho2_pruned() {
    open_case(
        "HO2/STO-3G ROKS-B3LYP sgx(35,194)",
        Reference::Roks,
        HO2,
        "sto-3g",
        Some("B3LYP"),
        cosx_pruned(),
    );
}

#[test]
fn rohf_cosx_gradient_matches_fd_nh2_flat() {
    open_case(
        "NH2/STO-3G ROHF flat(50,110)",
        Reference::Roks,
        NH2,
        "sto-3g",
        None,
        cosx_flat(),
    );
}

/// The converged RKS result re-expressed as UKS (α = β): same MOs, orbital
/// energies and Fock for both spins, D_s = D/2.
fn as_unrestricted(r: &ScfResult) -> ScfResult {
    let mut u = r.clone();
    u.spin = Spin::Unrestricted;
    u.density_alpha = 0.5 * r.density_r();
    u.density_beta = Some(0.5 * r.density_r());
    u.mos_beta = Some(r.mos_alpha.clone());
    u.eps_beta = Some(r.eps_alpha.clone());
    u.fock_beta = Some(r.fock_alpha.clone());
    u
}

/// The converged RKS result re-expressed as ROKS with no open shell: both
/// spin densities D/2 and both spin Focks F, so W = sym(D_a F D_a + D_a F D_b)
/// is the closed-shell D F D / 2.
fn as_restricted_open(r: &ScfResult) -> ScfResult {
    let mut u = as_unrestricted(r);
    u.spin = Spin::RestrictedOpen;
    u.mos_beta = None;
    u.eps_beta = None;
    u.fock_beta = None;
    u.rohf_spin_focks = Some((r.fock_alpha.clone(), r.fock_alpha.clone()));
    u
}

/// EXACTNESS ANCHOR (no FD, no new SCF): an RKS-B3LYP COSX result run through
/// the UKS and the ROKS COSX gradients (α = β) must change the gradient, relative
/// to the same reference's exact-K gradient expression, by exactly what the RKS
/// COSX gradient (`ks_gradient_closed_with_exchange`, FD-validated) changes
/// relative to the RKS exact-K one. The comparison is of these COSX INCREMENTS,
/// because the polarized and closed-shell XC gradient drivers themselves
/// differ by ~5e-10 at α = β (printed; it is the same with and without COSX).
/// Catches a per-spin exchange coefficient of `-c_x/4` (the closed-shell one)
/// instead of `-c_x/2`, a dropped spin, or J from the wrong density — each moves
/// the increment by the size of a COSX exchange term, O(0.1); the
/// reachability check asserts that term is that large here.
#[test]
fn open_shell_cosx_gradients_reduce_to_the_closed_shell_one() {
    let basis = "sto-3g";
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    let (bs, prep, bounds) = setup(&mol, basis);
    let op = Operator::coulomb();
    let cosx = cosx_flat();
    let cfg = scf_cfg(Some("B3LYP"), Some(cosx.clone()));
    let r = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).expect("rks");
    assert!(r.converged);
    let g_c = restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &cfg, &r).expect("rks grad");
    let g_c2 = ks_gradient_closed_with_exchange(
        &mol,
        &prep,
        &bs,
        op,
        &bounds,
        "B3LYP",
        &r,
        None,
        Some(&cosx),
    )
    .expect("rks grad");
    assert!(
        g_c == g_c2,
        "dispatch differs from ks_gradient_closed_with_exchange"
    );
    let g_ck =
        ks_gradient_closed_with_exchange(&mol, &prep, &bs, op, &bounds, "B3LYP", &r, None, None)
            .expect("rks exact-K");
    let inc_c = &g_c - &g_ck;
    let u = as_unrestricted(&r);
    let g_u = ks_gradient_uks_with_exchange(
        &mol,
        &prep,
        &bs,
        op,
        &bounds,
        "B3LYP",
        &u,
        None,
        Some(&cosx),
    )
    .expect("uks grad");
    let g_ud =
        unrestricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &cfg, &u).expect("uks dispatch");
    let g_uk =
        ks_gradient_uks(&mol, &prep, &bs, op, &bounds, "B3LYP", &u, None).expect("uks exact-K");
    let o = as_restricted_open(&r);
    let g_o = ks_gradient_roks_with_exchange(
        &mol,
        &prep,
        &bs,
        op,
        &bounds,
        "B3LYP",
        &o,
        None,
        Some(&cosx),
    )
    .expect("roks grad");
    let g_od = restricted_open_scf_gradient(&mol, &prep, &bs, op, &bounds, &cfg, &o)
        .expect("roks dispatch");
    let g_ok =
        ks_gradient_roks(&mol, &prep, &bs, op, &bounds, "B3LYP", &o, None).expect("roks exact-K");
    assert!(
        g_u == g_ud,
        "UKS dispatch differs from ks_gradient_uks_with_exchange"
    );
    assert!(
        g_o == g_od,
        "ROKS dispatch differs from ks_gradient_roks_with_exchange"
    );
    let du = max_abs(&(&(&g_u - &g_uk) - &inc_c));
    let d_o = max_abs(&(&(&g_o - &g_ok) - &inc_c));
    let k_term = ferric_scf::cosx_gradient::cosx_exchange_gradient(
        &mol,
        &prep,
        &cosx,
        &[(r.density_r(), -0.25 * 0.2)],
    )
    .expect("k term");
    println!(
        "anchor: COSX increment UKS {du:.3e}, ROKS {d_o:.3e}; polarized-vs-closed driver \
         offset (exact K) UKS {:.3e}, ROKS {:.3e}; max|COSX increment| = {:.3e}; \
         max|B3LYP COSX exchange term| = {:.3e}",
        max_abs(&(&g_uk - &g_ck)),
        max_abs(&(&g_ok - &g_ck)),
        max_abs(&inc_c),
        max_abs(&k_term)
    );
    for (what, d) in [("UKS", du), ("ROKS", d_o)] {
        assert!(
            d < ANCHOR_TOL,
            "{what}: COSX increment differs from the RKS one by {d:.3e}"
        );
    }
    assert!(
        max_abs(&k_term) > 1e6 * ANCHOR_TOL,
        "exchange term too small to resolve"
    );
}
