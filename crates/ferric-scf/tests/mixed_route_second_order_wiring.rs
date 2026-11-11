//! The SCF solvers lend their OWN J/K builders to every second-order product:
//! Newton on the mixed routes with a direct half (DF-J + direct K, DF-K +
//! conventional J), and TRAH and stability analysis on fitted routes. The
//! matvecs themselves are FD-anchored in `mixed_route_response_fd.rs`; these
//! tests pin the call sites: no direct four-centre response build runs, the
//! route's unfitted builder answers its half, and the converged energies match
//! DIIS.
//!
//! RIJCOSX is the exception, pinned here as a decision: its COSX K is NOT lent
//! (one COSX K costs more than the direct J+K it would replace; measurements
//! in `rhf::lend_scf_jk`), so its K half stays the direct response and only
//! the RI-J half is the SCF's own.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rhf_newton::unfitted_response_builds_on_this_thread;
use ferric_scf::screening::SchwarzBounds;

const AUX: &str = "def2-universal-jkfit";
const WATER: &str = "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n";
const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";
const NH2: &str = "3\nNH2\nN 0 0 0\nH 0 0.80 -0.62\nH 0 -0.80 -0.62\n";

#[derive(Clone, Copy, Debug)]
enum Route {
    RiJCosx,
    DfJDirectK,
    DfKDirectJ,
    /// RI-J + DF-K (both halves fitted).
    Fitted,
}

/// HF on `route`, tight convergence.
fn config(route: Route, level_shift: f64) -> RhfConfig {
    let (j, k, kb): (&str, &str, Option<&str>) = match route {
        Route::RiJCosx => (AUX, "", Some("cosx")),
        Route::DfJDirectK => (AUX, "", None),
        Route::DfKDirectJ => ("", AUX, None),
        Route::Fitted => (AUX, AUX, None),
    };
    RhfConfig {
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        level_shift,
        df_j_aux: Some(j.into()),
        df_k_aux: Some(k.into()),
        k_builder: kb.map(Into::into),
        ..Default::default()
    }
}

struct Sys {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn sys(xyz: &str, mult: usize) -> Sys {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Sys {
        mol,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Rhf,
    Uhf,
    Rohf,
}

fn solve(s: &Sys, kind: Kind, cfg: &RhfConfig) -> ferric_scf::result::ScfResult {
    let op = Operator::coulomb();
    match kind {
        Kind::Rhf => ferric_scf::rhf::solve_rhf(&s.ctx, &s.mol, &s.prep, op, &s.bounds, cfg),
        Kind::Uhf => ferric_scf::uhf::solve_uhf(&s.ctx, &s.mol, &s.prep, &s.bounds, cfg),
        Kind::Rohf => ferric_scf::rohf::solve_rohf(&s.ctx, &s.mol, &s.prep, op, &s.bounds, cfg),
    }
    .unwrap()
}

/// `(HVPs, direct response builds)` on this thread for `kind`'s matvec.
fn counters(kind: Kind) -> (usize, usize) {
    match kind {
        Kind::Rhf => (
            ferric_scf::rhf_newton::hvp_calls_on_this_thread(),
            ferric_scf::rhf_newton::direct_response_builds_on_this_thread(),
        ),
        Kind::Uhf => (
            ferric_scf::uhf_newton::hvp_calls_on_this_thread(),
            ferric_scf::uhf_newton::direct_response_builds_on_this_thread(),
        ),
        Kind::Rohf => (
            ferric_scf::rohf_newton::hvp_calls_on_this_thread(),
            ferric_scf::uhf_newton::direct_response_builds_on_this_thread(),
        ),
    }
}

/// What the SCF lends on `route`: on DF-J + direct K and DF-K + conventional J
/// every product takes its unfitted half from the SCF's builder and no direct
/// build runs; on RIJCOSX the COSX K is not lent, so no unfitted build runs and
/// the K half is the direct response (at least one build per product).
fn assert_lending(route: Route, hvps: usize, direct: usize, unfitted: usize, what: &str) {
    match route {
        Route::RiJCosx => {
            assert_eq!(
                unfitted, 0,
                "RIJCOSX: the SCF must not lend its COSX K to {what} (see rhf::lend_scf_jk)"
            );
            assert!(
                direct >= hvps,
                "RIJCOSX: {what}'s K half must be the direct response ({direct} builds for \
                 {hvps} products)"
            );
        }
        Route::Fitted => {
            assert_eq!(
                direct, 0,
                "{route:?}: {what}'s products must build no direct J/K"
            );
            assert_eq!(unfitted, 0, "{route:?}: nothing unfitted to lend");
        }
        _ => {
            assert_eq!(
                direct, 0,
                "{route:?}: {what}'s products must build no direct J/K"
            );
            assert!(
                unfitted >= hvps,
                "{route:?}: every {what} product must take its unfitted half from the SCF's \
                 builder ({unfitted} unfitted builds for {hvps} products)"
            );
        }
    }
}

/// Newton on a mixed route converges to the DIIS energy of the SAME route,
/// with the lending `assert_lending` pins for that route.
///
/// Bar: 1e-7 Ha, as in `open_shell_newton_fitted_response.rs`: both runs
/// minimise the same functional and stop at energy_conv 1e-9 / density_conv
/// 1e-7 on their own paths; 100·energy_conv covers two independently
/// terminated iterations.
fn newton_matches_diis_on_route(xyz: &str, mult: usize, kind: Kind, route: Route) {
    let s = sys(xyz, mult);
    // As in the fitted-response tests: UHF shifts, ROHF does not.
    let cfg_diis = config(route, if kind == Kind::Rohf { 0.0 } else { 0.2 });
    let cfg_newton = RhfConfig {
        newton_trigger: 1e-2,
        ..cfg_diis.clone()
    };
    let r_diis = solve(&s, kind, &cfg_diis);
    let (h0, d0) = counters(kind);
    let u0 = unfitted_response_builds_on_this_thread();
    let r_newton = solve(&s, kind, &cfg_newton);
    let (h1, d1) = counters(kind);
    let unfitted = unfitted_response_builds_on_this_thread() - u0;
    let (hvps, direct) = (h1 - h0, d1 - d0);
    let de = (r_diis.energy - r_newton.energy).abs();
    eprintln!(
        "{route:?} {}: E_diis={:.10} ({} it) E_newton={:.10} ({} it) dE={de:.2e} hvps={hvps} \
         direct={direct} unfitted={unfitted}",
        match kind {
            Kind::Rhf => "RHF",
            Kind::Uhf => "UHF",
            Kind::Rohf => "ROHF",
        },
        r_diis.energy,
        r_diis.iterations,
        r_newton.energy,
        r_newton.iterations
    );
    assert!(r_diis.converged && r_newton.converged);
    assert!(hvps >= 1, "the Newton branch must engage (0 HVPs)");
    assert_lending(route, hvps, direct, unfitted, "Newton");
    assert!(
        de < 1e-7,
        "{route:?}: Newton vs DIIS energy differ by {de:.3e}"
    );
}

#[test]
fn rhf_newton_rijcosx() {
    newton_matches_diis_on_route(WATER, 1, Kind::Rhf, Route::RiJCosx);
}

#[test]
fn rhf_newton_dfj_directk() {
    newton_matches_diis_on_route(WATER, 1, Kind::Rhf, Route::DfJDirectK);
}

#[test]
fn rhf_newton_dfk_directj() {
    newton_matches_diis_on_route(WATER, 1, Kind::Rhf, Route::DfKDirectJ);
}

#[test]
fn uhf_newton_rijcosx() {
    newton_matches_diis_on_route(OH, 2, Kind::Uhf, Route::RiJCosx);
}

#[test]
fn uhf_newton_dfj_directk() {
    newton_matches_diis_on_route(OH, 2, Kind::Uhf, Route::DfJDirectK);
}

#[test]
fn uhf_newton_dfk_directj() {
    newton_matches_diis_on_route(OH, 2, Kind::Uhf, Route::DfKDirectJ);
}

#[test]
fn rohf_newton_rijcosx() {
    newton_matches_diis_on_route(OH, 2, Kind::Rohf, Route::RiJCosx);
}

#[test]
fn rohf_newton_dfj_directk() {
    newton_matches_diis_on_route(OH, 2, Kind::Rohf, Route::DfJDirectK);
}

#[test]
fn rohf_newton_dfk_directj() {
    newton_matches_diis_on_route(OH, 2, Kind::Rohf, Route::DfKDirectJ);
}

/// TRAH on a fitted (RI-J + DF-K) SCF takes its Hessian products from the
/// SCF's own builders (no direct build); on RIJCOSX it lends the RI-J only
/// (`assert_lending`). Both reach the DIIS energy of the same route (bar as
/// for Newton).
fn trah_matches_diis_on_route(xyz: &str, mult: usize, kind: Kind, route: Route) {
    let s = sys(xyz, mult);
    let cfg_diis = config(route, 0.0);
    let cfg_trah = RhfConfig {
        trah_trigger: Some(1e-2),
        ..cfg_diis.clone()
    };
    let r_diis = solve(&s, kind, &cfg_diis);
    let (h0, d0) = counters(kind);
    let u0 = unfitted_response_builds_on_this_thread();
    let r_trah = solve(&s, kind, &cfg_trah);
    let (h1, d1) = counters(kind);
    let unfitted = unfitted_response_builds_on_this_thread() - u0;
    let (hvps, direct) = (h1 - h0, d1 - d0);
    let de = (r_diis.energy - r_trah.energy).abs();
    eprintln!(
        "TRAH {route:?}: E_diis={:.10} ({} it) E_trah={:.10} ({} it) dE={de:.2e} hvps={hvps} \
         direct={direct} unfitted={unfitted}",
        r_diis.energy, r_diis.iterations, r_trah.energy, r_trah.iterations
    );
    assert!(r_diis.converged && r_trah.converged);
    assert!(hvps >= 1, "TRAH must engage (0 HVPs)");
    assert_lending(route, hvps, direct, unfitted, "TRAH");
    assert!(
        de < 1e-7,
        "{route:?}: TRAH vs DIIS energy differ by {de:.3e}"
    );
}

#[test]
fn rhf_trah_fitted() {
    trah_matches_diis_on_route(WATER, 1, Kind::Rhf, Route::Fitted);
}

#[test]
fn rhf_trah_rijcosx() {
    trah_matches_diis_on_route(WATER, 1, Kind::Rhf, Route::RiJCosx);
}

#[test]
fn uhf_trah_rijcosx() {
    trah_matches_diis_on_route(OH, 2, Kind::Uhf, Route::RiJCosx);
}

/// `check_stability` on a fitted SCF analyses the Hessian of the FITTED
/// energy: no direct response build runs in either RHF channel (internal
/// singlet, external triplet) or in the UHF analysis, and λ_min differs from
/// the exact-response λ_min at the converged point by the fitting error only.
///
/// The ceiling (1e-3 relative) is a sanity bar, not a precision claim: the
/// matvecs differ by the JK-fit error, measured at ~1e-4 relative in the
/// fitted-response FD anchors.
#[test]
fn stability_on_a_fitted_scf_builds_no_direct_response() {
    use ferric_scf::stability::StabilityConfig;
    for kind in [Kind::Rhf, Kind::Uhf] {
        // NH2 ²B₁, not OH ²Π: OH's λ_min is the exact zero mode of the
        // degenerate π hole's orientation, where a relative comparison is
        // meaningless (measured 2026-11-11: fitted-response λ_min 8.8e-10,
        // exact-response 5.6e-6 at the same fitted SCF point -- the exact
        // response puts spurious curvature on that mode, which is the
        // mismatch `UhfNewtonInputs::j_response` documents for TRAH).
        let (xyz, mult) = if kind == Kind::Rhf {
            (WATER, 1)
        } else {
            (NH2, 2)
        };
        let s = sys(xyz, mult);
        let cfg = RhfConfig {
            check_stability: true,
            ..config(Route::Fitted, 0.0)
        };
        let (h0, d0) = counters(kind);
        let res = solve(&s, kind, &cfg);
        let (h1, d1) = counters(kind);
        assert!(res.converged);
        let st = res.stability.as_ref().expect("stability was requested");
        assert!(h1 - h0 >= 1, "stability must run products");
        assert_eq!(
            d1 - d0,
            0,
            "a fitted SCF's stability products must build no direct J/K"
        );

        // The exact-response λ_min at the same converged point.
        let budget = ferric_core::memory::resolve_budget_bytes(None);
        let c_a = &res.mos_alpha;
        let f_a = c_a.t().dot(&res.fock_alpha).dot(c_a);
        let (lam_exact, ext) = match kind {
            Kind::Rhf => {
                let inp = ferric_scf::rhf_newton::RhfNewtonInputs {
                    prep: &s.prep,
                    bounds: &s.bounds,
                    c: c_a,
                    f_mo: &f_a,
                    nocc: s.mol.nelec() as usize / 2,
                    k_mix_sr: 1.0,
                    rsh: None,
                    fxc: None,
                    thresh: cfg.integral_thresh,
                    ooc_budget: budget,
                };
                let int = ferric_scf::stability::rhf_internal_stability(
                    &s.ctx,
                    &inp,
                    &StabilityConfig::default(),
                )
                .unwrap();
                let ext = ferric_scf::stability::rhf_external_stability(
                    &s.ctx,
                    &inp,
                    &StabilityConfig::default(),
                )
                .unwrap();
                (int.lowest_eigenvalue, Some(ext.lowest_eigenvalue))
            }
            _ => {
                let c_b = res.mos_beta.as_ref().unwrap();
                let f_b = c_b.t().dot(res.fock_beta.as_ref().unwrap()).dot(c_b);
                let inp = ferric_scf::uhf_newton::UhfNewtonInputs {
                    prep: &s.prep,
                    bounds: &s.bounds,
                    c_a,
                    c_b,
                    f_a_mo: &f_a,
                    f_b_mo: &f_b,
                    nocc_a: (s.mol.nelec() as usize).div_ceil(2),
                    nocc_b: s.mol.nelec() as usize / 2,
                    k_mix_sr: 1.0,
                    rsh: None,
                    fxc: None,
                    thresh: cfg.integral_thresh,
                    ooc_budget: budget,
                    j_response: None,
                    k_response: None,
                };
                let int = ferric_scf::stability::uhf_internal_stability(
                    &s.ctx,
                    &inp,
                    &StabilityConfig::default(),
                )
                .unwrap();
                (int.lowest_eigenvalue, None)
            }
        };
        let rel = (st.lowest_eigenvalue - lam_exact).abs() / lam_exact.abs();
        eprintln!(
            "stability {}: lambda_fitted={:.8e} lambda_exact={lam_exact:.8e} rel={rel:.2e} \
             (Davidson iters {}); external fitted={:?} exact={ext:?}",
            if kind == Kind::Rhf { "RHF" } else { "UHF" },
            st.lowest_eigenvalue,
            st.iterations,
            res.stability_external.as_ref().map(|e| e.lowest_eigenvalue),
        );
        assert!(
            rel < 1e-3,
            "fitted vs exact lambda_min rel {rel:.2e} >= 1e-3"
        );
        if let (Some(e_fit), Some(e_ex)) = (
            res.stability_external.as_ref().map(|e| e.lowest_eigenvalue),
            ext,
        ) {
            let rel = (e_fit - e_ex).abs() / e_ex.abs();
            assert!(
                rel < 1e-3,
                "external: fitted vs exact rel {rel:.2e} >= 1e-3"
            );
        }
    }
}
