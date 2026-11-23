//! The J/K route decision (`ferric_scf::resolve_jk_route`) that RHF, UHF and
//! ROHF build from, and its read-back by the certificate.
//!
//! # What is tested
//!
//! 1. `route_table_*`: every (functional class, spin case, key) row of the
//!    table in `fock_assembly`'s module doc, on the pure decision. No SCF.
//! 2. `solvers_record_the_route_they_resolve`: for a set of routes on each
//!    solver, the SCF's recorded `df_jk` names the builders the route names,
//!    and the certificate's `ri-j` / `ri-k` / `link` / `cosx` entries are the
//!    route's (the record is what ran; the route is what the configuration
//!    says should run; they must not differ).
//! 3. The fitted-J fix for open-shell range-separated hybrids, measured:
//!    * `open_shell_rsh_fitted_j_moves_by_the_ri_j_error`: UKS and ROKS with
//!      CAM-B3LYP, fitted J minus four-centre J, against the same difference
//!      for RKS on the closed-shell parent at the same geometry.
//!    * `closed_shell_uks_cam_b3lyp_equals_rks`: water through UKS and RKS on
//!      the same fitted route.
//!
//! # Hypotheses (stated before measuring)
//!
//! * Physics: with J fitted for ω > 0, the open-shell energy moves by an RI-J
//!   fitting error of the same size and sign as the closed-shell parent's
//!   (the fit error is a property of the aux basis on these atoms: ~1e-5 Ha
//!   for water/6-31G on def2-universal-jkfit).
//! * Artifact, J still four-centre: the open-shell difference is exactly 0.
//! * Artifact, J fitted from the wrong density (one spin, or 2x): the
//!   difference is not a fit error at all but a Coulomb energy error, mHa to
//!   Ha. Both artifacts fall outside the band asserted below.
//!
//! # Bars
//!
//! * RI-J band: `|ΔE_open| ∈ [0.25, 4] · |ΔE_closed|`, same sign. The removed
//!   or added electron changes the fitted density by one orbital's worth, so
//!   the fit error stays of the parent's order; the band is wide on purpose
//!   (measured ratios are printed and in the commit message), its job is to
//!   exclude the two artifacts (0 and ≥ 1e-3 Ha), which differ from the
//!   parent's ~1e-5 Ha by orders of magnitude.
//! * UKS singlet vs RKS: 1e-8 Ha. Both converge to `energy_conv = 1e-10`,
//!   `density_conv = 1e-8`; the energy is variational, so its error is second
//!   order in the density error (< 1e-12), and the f64 RI-J tensors are the
//!   same fit. 1e-8 leaves room for DIIS end-point differences between the
//!   two drivers and still sits three orders below the RI-J error the test
//!   would see if UKS dropped the fit (~1e-5 Ha).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::xc_trait::KMix;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::certificate::{scf_certificate, scf_jk_route};
use ferric_scf::df_j::RiJPrecision;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{resolve_jk_route, JkExchange, JkRoute, ScfResult, Spin};

const AUX: &str = "def2-universal-jkfit";
const OTHER: &str = "cc-pvtz-jkfit";

const WATER_XYZ: &str = "3\nwater\nO 0.0000 0.0000 0.1173\n\
                         H 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n";

// ── 1. the table ─────────────────────────────────────────────────────────

fn hf() -> JkExchange {
    JkExchange {
        functional: false,
        k_mix: KMix::default(),
    }
}
fn pure() -> JkExchange {
    JkExchange {
        functional: true,
        k_mix: KMix {
            sr: 0.0,
            lr: 0.0,
            omega: 0.0,
        },
    }
}
fn hybrid() -> JkExchange {
    JkExchange {
        functional: true,
        k_mix: KMix {
            sr: 0.25,
            lr: 0.25,
            omega: 0.0,
        },
    }
}
fn rsh() -> JkExchange {
    JkExchange {
        functional: true,
        k_mix: KMix {
            sr: 0.19,
            lr: 0.65,
            omega: 0.33,
        },
    }
}

fn cfg(j: Option<&str>, k: Option<&str>, kb: Option<&str>) -> RhfConfig {
    let mut c = RhfConfig::default();
    c.df_j_aux = j.map(str::to_string);
    c.df_k_aux = k.map(str::to_string);
    c.k_builder = ferric_scf::options::KBuilderKind::parse_opt(kb).expect("spelling");
    c
}

fn route(x: JkExchange, spin: Spin, j: Option<&str>, k: Option<&str>) -> JkRoute {
    resolve_jk_route(&cfg(j, k, None), x, spin).unwrap()
}

const OPEN: [Spin; 2] = [Spin::Unrestricted, Spin::RestrictedOpen];
const ALL: [Spin; 3] = [Spin::Restricted, Spin::Unrestricted, Spin::RestrictedOpen];

fn s(x: &str) -> Option<String> {
    Some(x.to_string())
}

/// The J column of the table. MUTATION: give the open-shell arm `None` for
/// J under ω > 0 (the pre-fix `j_aux_eff`) and the RSH-named rows fail.
#[test]
fn route_table_j() {
    for spin in ALL {
        for x in [hf(), pure(), hybrid(), rsh()] {
            assert_eq!(route(x, spin, Some(""), None).j_aux, None, "{spin:?} {x:?}");
            assert_eq!(
                route(x, spin, Some(OTHER), None).j_aux,
                s(OTHER),
                "{spin:?} {x:?}"
            );
        }
        assert_eq!(route(hf(), spin, None, None).j_aux, None, "HF {spin:?}");
    }
    for x in [pure(), hybrid(), rsh()] {
        assert_eq!(
            route(x, Spin::Restricted, None, None).j_aux,
            s(AUX),
            "{x:?}"
        );
        for spin in OPEN {
            assert_eq!(route(x, spin, None, None).j_aux, None, "{spin:?} {x:?}");
        }
    }
}

/// The K column, the SR/LR pair and `k_df_routed`.
#[test]
fn route_table_k() {
    for spin in ALL {
        // HF: exact unless named.
        let r = route(hf(), spin, None, None);
        assert_eq!((r.k_aux, r.rsh_pair, r.k_df_routed), (None, None, false));
        assert_eq!(route(hf(), spin, None, Some("")).k_aux, None);
        let r = route(hf(), spin, None, Some(OTHER));
        assert_eq!((r.k_aux, r.k_df_routed), (s(OTHER), true));
        // Pure: no K, a named fit is dropped but still routes K as fitted.
        let r = route(pure(), spin, None, Some(OTHER));
        assert_eq!(
            (r.k_aux, r.rsh_pair, r.k_df_routed, r.k_consumed),
            (None, None, true, false)
        );
        let r = route(pure(), spin, None, None);
        assert_eq!((r.k_aux, r.k_df_routed), (None, false));
        // Global hybrid named / opted out.
        assert_eq!(route(hybrid(), spin, None, Some(OTHER)).k_aux, s(OTHER));
        assert_eq!(route(hybrid(), spin, None, Some("")).k_aux, None);
        // RSH: no ω = 0 K, the pair on the named or default aux.
        let r = route(rsh(), spin, None, None);
        assert_eq!(
            (r.k_aux, r.rsh_pair, r.k_df_routed),
            (None, Some((AUX.into(), 0.33)), true)
        );
        let r = route(rsh(), spin, None, Some(OTHER));
        assert_eq!(r.rsh_pair, Some((OTHER.into(), 0.33)));
    }
    // The restricted auto-default for a hybrid; none for the open shells.
    assert_eq!(route(hybrid(), Spin::Restricted, None, None).k_aux, s(AUX));
    for spin in OPEN {
        let r = route(hybrid(), spin, None, None);
        assert_eq!((r.k_aux, r.k_df_routed), (None, false), "{spin:?}");
    }
}

/// `k_builder`: COSX replaces the ω = 0 DF-K (and is refused next to a named
/// one); LinK needs exact J and K; neither runs for pure or RSH exchange.
/// MUTATION: `pluggable_k` ignoring RI-J (LinK certified under fitted J, the
/// pre-route certificate rule) fails the LinK-under-RI-J row.
#[test]
fn route_table_k_builder() {
    use ferric_scf::options::KBuilderKind as K;
    for spin in ALL {
        let c = resolve_jk_route(&cfg(Some(AUX), None, Some("cosx")), hybrid(), spin).unwrap();
        assert!(c.cosx_k && c.k_aux.is_none() && c.j_aux.as_deref() == Some(AUX));
        assert_eq!(
            c.pluggable_k(Some(K::Cosx)),
            Some(K::Cosx),
            "RIJCOSX {spin:?}"
        );
        assert!(resolve_jk_route(&cfg(None, Some(OTHER), Some("cosx")), hf(), spin).is_err());
        let c = resolve_jk_route(&cfg(None, Some(""), Some("cosx")), hf(), spin).unwrap();
        assert!(c.cosx_k);
        for x in [pure(), rsh()] {
            let c = resolve_jk_route(&cfg(None, None, Some("cosx")), x, spin).unwrap();
            assert!(!c.cosx_k);
            assert_eq!(c.pluggable_k(Some(K::Cosx)), None, "{x:?}");
            assert_eq!(c.pluggable_k(Some(K::Link)), None, "{x:?}");
        }
        let exact = route(hf(), spin, Some(""), Some(""));
        assert_eq!(exact.pluggable_k(Some(K::Link)), Some(K::Link));
        assert_eq!(exact.pluggable_k(Some(K::Direct)), None);
        let rij = route(hf(), spin, Some(AUX), Some(""));
        assert_eq!(
            rij.pluggable_k(Some(K::Link)),
            None,
            "LinK under RI-J {spin:?}"
        );
        let rik = route(hf(), spin, Some(""), Some(AUX));
        assert_eq!(
            rik.pluggable_k(Some(K::Link)),
            None,
            "LinK under RI-K {spin:?}"
        );
    }
}

// ── 2. solvers record the route; the certificate reads it ────────────────

struct Sys {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn sys(xyz: &str, charge: i32, mult: usize, basis_name: &str) -> Sys {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Sys { mol, prep, bounds }
}

fn solve(sy: &Sys, spin: Spin, config: &RhfConfig) -> ScfResult {
    let ctx = ParallelContext::default();
    let r = match spin {
        Spin::Restricted => solve_rhf(
            &ctx,
            &sy.mol,
            &sy.prep,
            Operator::coulomb(),
            &sy.bounds,
            config,
        ),
        Spin::Unrestricted => solve_uhf(&ctx, &sy.mol, &sy.prep, &sy.bounds, config),
        Spin::RestrictedOpen => solve_rohf(
            &ctx,
            &sy.mol,
            &sy.prep,
            Operator::coulomb(),
            &sy.bounds,
            config,
        ),
    }
    .unwrap();
    assert!(r.converged, "{spin:?} {:?} did not converge", config.xc);
    r
}

fn tight(xc: Option<&str>, j: Option<&str>, k: Option<&str>, kb: Option<&str>) -> RhfConfig {
    let mut c = cfg(j, k, kb);
    c.xc = xc.map(str::to_string);
    c.energy_conv = 1e-10;
    c.density_conv = 1e-8;
    c.ri_j_precision = Some(RiJPrecision::F64);
    c
}

/// The certificate's J/K entries a route predicts: `ri-j` for RI-J, one
/// `ri-k` for RI-K or the SR/LR pair, and the pluggable builder that runs.
fn predicted(route: &JkRoute, k_builder: Option<&str>) -> Vec<&'static str> {
    let mut v = Vec::new();
    if route.j_aux.is_some() {
        v.push("ri-j");
    }
    if route.k_aux.is_some() || route.rsh_pair.is_some() {
        v.push("ri-k");
    }
    if let Some(p) = route
        .pluggable_k(ferric_scf::options::KBuilderKind::parse_opt(k_builder).expect("spelling"))
    {
        v.push(p.as_str());
    }
    v
}

fn certified(sy: &Sys, r: &ScfResult, config: &RhfConfig) -> Vec<String> {
    scf_certificate(&sy.mol, r, config)
        .approximations
        .iter()
        .map(|a| a.name.clone())
        .filter(|n| matches!(n.as_str(), "ri-j" | "ri-k" | "link" | "cosx"))
        .collect()
}

/// For each route, on each solver: the recorded `df_jk` names the route's
/// builders, and the certificate lists exactly the route's J/K entries.
///
/// MUTATIONS: UHF recording `None` for J under ω > 0 while it still fits J
/// fails the record assertion (UKS CAM-B3LYP row); the certificate dropping
/// its `ri-j` entry when an SR/LR pair is recorded fails the certificate
/// assertion (RKS CAM-B3LYP row).
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn solvers_record_the_route_they_resolve() {
    let water = sys(WATER_XYZ, 0, 1, "sto-3g");
    let cation = sys(WATER_XYZ, 1, 2, "sto-3g");
    let cam = Some("HYB_GGA_XC_CAM_B3LYP");
    type Case = (
        Option<&'static str>,
        Option<&'static str>,
        Option<&'static str>,
        Option<&'static str>,
    );
    let cases: [Case; 8] = [
        (None, None, None, None),
        (None, Some(AUX), Some(AUX), None),
        (None, Some(AUX), Some(""), Some("link")),
        (Some("PBE"), Some(AUX), Some(AUX), None),
        (Some("B3LYP"), Some(AUX), Some(AUX), None),
        (cam, Some(AUX), Some(AUX), None),
        (cam, Some(""), None, None),
        (cam, None, None, None),
    ];
    for (xc, j, k, kb) in cases {
        for spin in ALL {
            let sy = if spin == Spin::Restricted {
                &water
            } else {
                &cation
            };
            let config = tight(xc, j, k, kb);
            let r = solve(sy, spin, &config);
            let route = scf_jk_route(&r, &config).expect("route resolves");
            assert!(
                route.matches_record(r.df_jk.as_ref()),
                "{spin:?} {xc:?} j={j:?} k={k:?}: route {route:?} but the SCF recorded {:?}",
                r.df_jk
            );
            assert_eq!(
                certified(sy, &r, &config),
                predicted(&route, kb),
                "{spin:?} {xc:?} j={j:?} k={k:?} kb={kb:?}: certificate vs route"
            );
        }
    }
}

// ── 3. the fitted-J anchors ──────────────────────────────────────────────

/// UKS and ROKS with CAM-B3LYP: fitted J minus four-centre J sits in the
/// RI-J band of the closed-shell parent (RKS water, same geometry, basis
/// and aux). The four-centre arm is the base build's open-shell route.
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn open_shell_rsh_fitted_j_moves_by_the_ri_j_error() {
    let cam = Some("HYB_GGA_XC_CAM_B3LYP");
    let water = sys(WATER_XYZ, 0, 1, "6-31g");
    let cation = sys(WATER_XYZ, 1, 2, "6-31g");
    let d_closed = solve(
        &water,
        Spin::Restricted,
        &tight(cam, Some(AUX), Some(AUX), None),
    )
    .energy
        - solve(
            &water,
            Spin::Restricted,
            &tight(cam, Some(""), Some(AUX), None),
        )
        .energy;
    eprintln!("RKS water: E(RI-J) - E(4c J) = {d_closed:.3e}");
    assert!(
        d_closed.abs() > 1e-7,
        "closed-shell RI-J error {d_closed:e} is no fit at all"
    );
    for spin in OPEN {
        let fitted = solve(&cation, spin, &tight(cam, Some(AUX), Some(AUX), None));
        let exact = solve(&cation, spin, &tight(cam, Some(""), Some(AUX), None));
        assert_eq!(
            fitted.df_jk.as_ref().and_then(|r| r.j_aux.as_deref()),
            Some(AUX),
            "{spin:?}: J was not fitted"
        );
        let d = fitted.energy - exact.energy;
        let ratio = d / d_closed;
        eprintln!("{spin:?} cation: E(RI-J) - E(4c J) = {d:.3e}, ratio to RKS {ratio:.3}");
        assert!(
            (0.25..=4.0).contains(&ratio),
            "{spin:?}: fitted-J shift {d:e} is outside the RI-J band of the parent {d_closed:e}"
        );
    }
}

/// Closed-shell water through UKS with CAM-B3LYP equals RKS on the same
/// fitted route (RI-J and the SR/LR pair on def2-universal-jkfit).
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn closed_shell_uks_cam_b3lyp_equals_rks() {
    let cam = Some("HYB_GGA_XC_CAM_B3LYP");
    let water = sys(WATER_XYZ, 0, 1, "6-31g");
    let config = tight(cam, Some(AUX), Some(AUX), None);
    let rks = solve(&water, Spin::Restricted, &config);
    let uks = solve(&water, Spin::Unrestricted, &config);
    assert_eq!(
        rks.df_jk.as_ref().and_then(|r| r.j_aux.as_deref()),
        Some(AUX)
    );
    assert_eq!(
        uks.df_jk.as_ref().and_then(|r| r.j_aux.as_deref()),
        Some(AUX)
    );
    let d = (uks.energy - rks.energy).abs();
    eprintln!("UKS - RKS (water, CAM-B3LYP, RI-J) = {d:.3e}");
    assert!(
        d < 1e-8,
        "UKS {} vs RKS {} differ by {d:e}",
        uks.energy,
        rks.energy
    );
}
