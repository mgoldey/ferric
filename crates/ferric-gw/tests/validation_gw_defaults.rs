//! VALIDATION tier — the `gw` row's error AT THE DEFAULTS.
//!
//! `validation_gw.rs` proves ferric's GW against PySCF `gw_ac`/`ugw_ac` at
//! matched settings (`n_quad = 100`, `trunc_thresh = 0`, evGW to 1e-7 Ha).
//! The CLI and Python defaults are coarser: Gauss–Legendre
//! `n_quad = ferric_gw::DEFAULT_N_QUAD` (40), `trunc_thresh = 1e-4`,
//! `ev_conv_thresh = 1e-4`, `max_ev_iter = 20`. This file pins the difference
//! between the two, ferric against ferric on the same SCF:
//!
//! |ε_QP(defaults) − ε_QP(validated)| ≤ `BAR_MEV` on HOMO and LUMO.
//!
//! Runs in the weekly `validation` job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-gw --release \
//!     --test validation_gw_defaults --run-ignored only
//! ```
//!
//! # The calibration this pins (2026-11-06, plain Σc quadrature)
//!
//! One knob at a time from the validated settings, H2O, NH3, CH4, CO, N2, HF,
//! C2H4, CH2O at cc-pVDZ (H2O, NH3, N2 also at aug-cc-pVDZ), OH and NH2
//! U-G0W0@UHF at cc-pVDZ. Worst |ΔE_QP| over HOMO and LUMO:
//!
//! | knob | G0W0@PBE | @HF starts (G0W0, evGW₀, evGW, U-G0W0@UHF) |
//! |---|---:|---:|
//! | n_quad 20 (old default) | 21.5 meV (NH3/cc-pVDZ HOMO) | 0.05 meV |
//! | n_quad 30 | 9.3 meV (CH2O LUMO) | 0.06 meV |
//! | n_quad 40 (default) | 5.7 meV (NH3/cc-pVDZ HOMO) | 0.06 meV |
//! | trunc_thresh 1e-4 (default) | 0.26 meV | 0.04 meV |
//! | ev_conv_thresh 1e-4 (default) | — | 0.11 meV |
//!
//! `BAR_MEV` = 10 meV. Why that bar: it is an order of magnitude below the
//! ~0.1 eV scatter between GW codes on GW100 and below the basis-set error of
//! cc-pVDZ, so the defaults do not change any conclusion drawn at the
//! validated settings.
//!
//! Since 2026-11-09 the closed-shell drivers integrate the Σc kernel exactly
//! too (`sigma::sigma_c_at_z_product`; wiki/notes/gw-deep-states-2026-11-09).
//! Most of the n_quad error above was the plain sum's: the pinned @PBE cases
//! are now ≤ 0.27 meV at the defaults (CH2O HOMO), and the WHOLE default
//! window HOMO−2 … LUMO+2 of G0W0@PBE (CH2O, NH3) moves ≤ 0.06 meV between
//! n_quad 40 and 100 (`CLOSED_NQUAD_BAR_MEV`); before, CH2O HOMO−1 moved
//! 282 meV. The old n_quad = 20 control no longer misses 10 meV (NH3 HOMO
//! 0.5 meV), so the frequency-grid control is the window test's 16 points
//! (30–35 meV). LiH was dropped from the set in 2026-11-06 (its Li 1s at @HF
//! and its HOMO at @PBE never settled in n_quad; not re-measured).
//! U-G0W0@UKS/PBE (OH, NH2) has its own n_quad bar
//! (`UKS_NQUAD_BAR_MEV`, below): before 2026-11-08 its frontier states moved
//! by up to 66 meV between n_quad = 80 and 100, because the plain
//! Gauss–Legendre sum cannot resolve the Σc kernel's ridge when a state
//! sits 0.015 Ha from the per-spin ε_F; the U drivers now integrate the
//! kernel exactly (`sigma::sigma_c_at_z_product`).
//!
//! # Physics vs artifact hypothesis
//!
//! * If the defaults are good enough, the default run and the validated run
//!   differ by the measured few meV.
//! * If the test cannot see the knob (the default is not what the run used,
//!   or `n_points` does not reach W), the two runs agree to µeV, and so does
//!   the n_quad = 16 control of the window tests, which must MISS by more than
//!   their 1 meV bar (measured 30–35 meV), so this file fails if it is blind.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::vxc_mo::vxc_diagonal_mo_on_grids;
use ferric_gw::{run_gw, run_u_gw, GwConfig, GwMethod, DEFAULT_N_QUAD};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::ks_gradient::KsGradGrids;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

const HA_TO_MEV: f64 = 27_211.386_245_988;
/// Worst |ΔE_QP| on HOMO and LUMO the defaults may show (see the module doc).
const BAR_MEV: f64 = 10.0;

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-gw manifest dir should be <root>/crates/ferric-gw")
        .to_path_buf()
}

/// The validated settings of `validation_gw.rs`.
fn validated_pdep() -> PdepRpaConfig {
    PdepRpaConfig {
        trunc_thresh: 0.0,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 100,
            u0: 0.5,
        },
        need_inv_dielectric_freq: true,
        ..Default::default()
    }
}

fn validated_gw(method: GwMethod) -> GwConfig {
    GwConfig {
        method,
        max_ev_iter: 60,
        ev_conv_thresh: 1e-7,
        ..Default::default()
    }
}

/// The CLI / Python defaults (`ferric-cli` `gw` arm, `run_gw`, `run_u_gw`).
fn default_pdep(n_quad: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            n_points: n_quad,
            ..Default::default()
        },
        need_inv_dielectric_freq: true,
        ..Default::default()
    }
}

struct Sys {
    label: String,
    mol: Molecule,
    obs_bs: basis::BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn load(system: &str, mult: usize) -> Sys {
    let xyz = workspace_root()
        .join("testdata/molecules/validation")
        .join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    Sys {
        label: format!("{system}/cc-pvdz"),
        mol,
        obs_bs,
        obs,
        dfbs,
        bounds,
        ctx: ParallelContext::default(),
    }
}

fn scf_config(xc: Option<&str>) -> RhfConfig {
    let mut c = RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    };
    if let Some(x) = xc {
        // As `run_gw(xc=...)` and the CLI KS reference: RI-J/RI-K.
        c.xc = Some(x.into());
        c.df_j_aux = Some("def2-universal-jkfit".into());
        c.df_k_aux = Some("def2-universal-jkfit".into());
    }
    c
}

/// (HOMO, LUMO) QP energies of one spin channel (Ha).
fn frontier(mo: &[usize], eps: &[f64], nocc: usize) -> [f64; 2] {
    let at = |p: usize| {
        eps[mo
            .iter()
            .position(|&m| m == p)
            .expect("frontier MO in window")]
    };
    [at(nocc - 1), at(nocc)]
}

/// max over HOMO/LUMO of |a − b| in meV, printing each.
fn worst_mev(ctx: &str, what: &str, a: [f64; 2], b: [f64; 2]) -> f64 {
    let mut w = 0.0_f64;
    for (lab, (x, y)) in ["HOMO", "LUMO"].iter().zip(a.iter().zip(b.iter())) {
        let d = (x - y).abs() * HA_TO_MEV;
        eprintln!("{ctx}: {what} {lab}: {x:+.8} vs validated {y:+.8} Ha, |d| {d:.4} meV");
        w = w.max(d);
    }
    w
}

/// Closed shell: (validated, defaults) frontier QP energies.
fn closed(system: &str, xc: Option<&str>, method: GwMethod) -> [[f64; 2]; 2] {
    let sys = load(system, 1);
    let cfg = scf_config(xc);
    let scf = solve_rhf(
        &sys.ctx,
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{}: SCF failed: {e:?}", sys.label));
    assert!(scf.converged, "{}: SCF did not converge", sys.label);
    let vxc = xc.map(|x| {
        let g = KsGradGrids::from_config(&cfg);
        vxc_diagonal_mo_on_grids(&sys.mol, &sys.obs_bs, x, &scf, &g.main, &g.nlc)
            .unwrap_or_else(|e| panic!("{}: v_xc: {e:?}", sys.label))
            .0
    });
    let nocc = usize::try_from(sys.mol.nelec()).unwrap() / 2;
    let go = |pdep: &PdepRpaConfig, gw: &GwConfig| {
        let r = run_gw(
            &sys.mol,
            &sys.obs,
            &sys.dfbs,
            Operator::coulomb(),
            &scf,
            pdep,
            gw,
            vxc.as_ref(),
        )
        .unwrap_or_else(|e| panic!("{}: run_gw failed: {e:?}", sys.label));
        assert!(
            r.qp_converged.iter().all(|&c| c),
            "{}: QP Newton unconverged",
            sys.label
        );
        assert!(r.outer_converged, "{}: outer loop unconverged", sys.label);
        frontier(&r.mo_indices, r.eps_qp.as_slice().unwrap(), nocc)
    };
    let gw_default = GwConfig {
        method,
        ..Default::default()
    };
    [
        go(&validated_pdep(), &validated_gw(method)),
        go(&default_pdep(DEFAULT_N_QUAD), &gw_default),
    ]
}

fn assert_defaults_within_bar(ctx: &str, e: [[f64; 2]; 2]) -> f64 {
    let w = worst_mev(ctx, "defaults", e[1], e[0]);
    assert!(
        w <= BAR_MEV,
        "{ctx}: defaults miss the validated settings by {w:.3} meV > {BAR_MEV} meV"
    );
    w
}

// The default values themselves are pinned in `gw_defaults_pinned.rs`, which
// runs in every shard; every test here is validation tier.

/// NH3: the worst HOMO of the 2026-11-06 calibration (plain quadrature).
#[test]
#[ignore = "validation: GW defaults vs validated settings"]
fn g0w0_pbe_nh3_defaults_within_bar() {
    assert_defaults_within_bar("nh3 G0W0@PBE", closed("nh3", Some("pbe"), GwMethod::G0W0));
}

/// CH2O: the worst LUMO of the calibration (9.3 meV at 30 points, 3.6 at 40).
#[test]
#[ignore = "validation: GW defaults vs validated settings"]
fn g0w0_pbe_ch2o_defaults_within_bar() {
    assert_defaults_within_bar("ch2o G0W0@PBE", closed("ch2o", Some("pbe"), GwMethod::G0W0));
}

#[test]
#[ignore = "validation: GW defaults vs validated settings"]
fn g0w0_pbe_h2o_defaults_within_bar() {
    assert_defaults_within_bar("h2o G0W0@PBE", closed("h2o", Some("pbe"), GwMethod::G0W0));
}

/// HF starts sit far inside the bar; evGW also exercises ev_conv_thresh 1e-4.
#[test]
#[ignore = "validation: GW defaults vs validated settings"]
fn hf_starts_h2o_defaults_within_bar() {
    for m in [GwMethod::G0W0, GwMethod::EvGw0, GwMethod::EvGw] {
        assert_defaults_within_bar(&format!("h2o {m:?}@HF"), closed("h2o", None, m));
    }
}

/// U-G0W0@UHF on OH (the marginal-UHF radical of validation_gw.rs).
#[test]
#[ignore = "validation: GW defaults vs validated settings"]
fn u_g0w0_uhf_oh_defaults_within_bar() {
    let sys = load("oh", 2);
    let cfg = RhfConfig {
        density_conv: 1e-9,
        check_stability: true,
        scf_stability_descent: true,
        ..scf_config(None)
    };
    let scf = solve_uhf(&sys.ctx, &sys.mol, &sys.obs, &sys.bounds, &cfg)
        .unwrap_or_else(|e| panic!("oh: UHF failed: {e:?}"));
    assert!(scf.converged, "oh: UHF did not converge");
    let ne = usize::try_from(sys.mol.nelec()).unwrap();
    let (na, nb) = (ne.div_ceil(2), ne / 2);
    let go = |pdep: &PdepRpaConfig, gw: &GwConfig| {
        let r = run_u_gw(
            &sys.mol,
            &sys.obs,
            &sys.dfbs,
            Operator::coulomb(),
            &scf,
            pdep,
            gw,
            None,
        )
        .unwrap_or_else(|e| panic!("oh: run_u_gw failed: {e:?}"));
        assert!(r.qp_converged_a.iter().chain(&r.qp_converged_b).all(|&c| c));
        (
            frontier(&r.mo_indices, r.eps_qp_a.as_slice().unwrap(), na),
            frontier(&r.mo_indices, r.eps_qp_b.as_slice().unwrap(), nb),
        )
    };
    let v = go(&validated_pdep(), &validated_gw(GwMethod::G0W0));
    let d = go(&default_pdep(DEFAULT_N_QUAD), &GwConfig::default());
    assert_defaults_within_bar("oh U-G0W0@UHF alpha", [v.0, d.0]);
    assert_defaults_within_bar("oh U-G0W0@UHF beta", [v.1, d.1]);
}

/// Closed-shell G0W0@PBE over the WHOLE default window (HOMO−2 … LUMO+2):
/// the run_gw drivers integrate the Σc kernel exactly
/// (`sigma::sigma_c_at_z_product`), so every window state converges in
/// n_quad. MEASURED 2026-11-09 (wiki/notes/gw-deep-states-2026-11-09), the
/// plain Gauss–Legendre sum vs the exact dRPA pole sum: CH2O/cc-pVDZ HOMO−1
/// −281 / +1.1 / −1.6 meV at 40 / 100 / 200 points, NH3 HOMO−1 +126 / +111 /
/// −48 meV; the product rule ≤ 0.06 meV from 40 points on.
const CLOSED_NQUAD_BAR_MEV: f64 = 1.0;
/// The negative control's grid.
const CLOSED_COARSE_N_QUAD: usize = 16;

/// run_gw on one closed-shell KS SCF (as `run_gw(xc=...)` builds it), as a
/// closure of n_quad returning every window QP energy (Ha) and asserting
/// every QP solve returned a root.
fn closed_window(system: &str, xc: &str) -> impl Fn(usize) -> Vec<f64> {
    let sys = load(system, 1);
    let cfg = scf_config(Some(xc));
    let scf = solve_rhf(
        &sys.ctx,
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{}: SCF failed: {e:?}", sys.label));
    assert!(scf.converged, "{}: SCF did not converge", sys.label);
    let g = KsGradGrids::from_config(&cfg);
    let vxc = vxc_diagonal_mo_on_grids(&sys.mol, &sys.obs_bs, xc, &scf, &g.main, &g.nlc)
        .unwrap_or_else(|e| panic!("{}: v_xc: {e:?}", sys.label))
        .0;
    move |n_quad: usize| {
        let r = run_gw(
            &sys.mol,
            &sys.obs,
            &sys.dfbs,
            Operator::coulomb(),
            &scf,
            &pdep_at(n_quad, 0.0),
            &GwConfig::default(),
            Some(&vxc),
        )
        .unwrap_or_else(|e| panic!("{}: run_gw failed: {e:?}", sys.label));
        assert!(
            r.qp_converged.iter().all(|&c| c),
            "{}: a QP solve returned no root",
            sys.label
        );
        r.eps_qp.to_vec()
    }
}

/// max over the window of |a − b| in meV, printing each state.
fn window_worst_mev(ctx: &str, what: &str, a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "{ctx}: window length");
    let mut w = 0.0_f64;
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs() * HA_TO_MEV;
        eprintln!(
            "{ctx}: {what} window state {k}: {x:+.8} vs n_quad 100 {y:+.8} Ha, |d| {d:.4} meV"
        );
        w = w.max(d);
    }
    w
}

/// n_quad 40 (the default) vs 100 within `CLOSED_NQUAD_BAR_MEV` on every
/// window state; the coarse control MISSES that bar, so the comparison sees
/// the frequency grid.
fn closed_nquad_case(system: &str) {
    let go = closed_window(system, "pbe");
    let ctx = format!("{system} G0W0@PBE");
    let v = go(100);
    let w40 = window_worst_mev(&ctx, "n_quad = 40", &go(DEFAULT_N_QUAD), &v);
    assert!(
        w40 <= CLOSED_NQUAD_BAR_MEV,
        "{ctx}: n_quad 40 vs 100 differ by {w40:.4} meV > {CLOSED_NQUAD_BAR_MEV} meV on the window"
    );
    let wc = window_worst_mev(&ctx, "n_quad = 16", &go(CLOSED_COARSE_N_QUAD), &v);
    assert!(
        wc > CLOSED_NQUAD_BAR_MEV,
        "{ctx}: n_quad = {CLOSED_COARSE_N_QUAD} misses n_quad = 100 by only {wc:.4} meV; the \
         comparison does not see the frequency grid"
    );
}

#[test]
#[ignore = "validation: GW defaults vs validated settings"]
fn g0w0_pbe_ch2o_window_converges_in_n_quad() {
    closed_nquad_case("ch2o");
}

#[test]
#[ignore = "validation: GW defaults vs validated settings"]
fn g0w0_pbe_nh3_window_converges_in_n_quad() {
    closed_nquad_case("nh3");
}

/// U-G0W0@UKS/PBE (OH, NH2): the frequency integral of Σc is done by product
/// integration (`sigma::sigma_c_at_z_product`), so the frontier QP energies
/// converge in n_quad. MEASURED 2026-11-08 (wiki/notes/ugw-uks-2026-11-08):
/// OH and NH2 frontier states agree between n_quad = 40 and 100 to
/// ≤ 0.005 meV; with the old plain quadrature the OH β HOMO moved 66 meV
/// between 80 and 100 and was off by up to 1.7 eV at 40.
const UKS_NQUAD_BAR_MEV: f64 = 1.0;
/// The negative control's grid: product integration at 16 points still
/// misses by 10–23 meV on these frontier states (prototype, notes).
const UKS_COARSE_N_QUAD: usize = 16;

/// UKS/PBE as `run_u_gw(xc = "pbe")` builds it (RI-J/K, MOM after 5),
/// returning a closure that runs U-G0W0 at given settings and yields the
/// frontier QP energies ([α HOMO, α LUMO], [β HOMO, β LUMO]), asserting
/// every QP solve returned a root.
fn uks_frontier(system: &str) -> impl Fn(&PdepRpaConfig, &GwConfig) -> [[f64; 2]; 2] {
    let sys = load(system, 2);
    let cfg = RhfConfig {
        mom_after_iter: 5,
        ..scf_config(Some("pbe"))
    };
    let scf = solve_uhf(&sys.ctx, &sys.mol, &sys.obs, &sys.bounds, &cfg)
        .unwrap_or_else(|e| panic!("{system}: UKS failed: {e:?}"));
    assert!(scf.converged, "{system}: UKS did not converge");
    let g = KsGradGrids::from_config(&cfg);
    let (va, vb) = vxc_diagonal_mo_on_grids(&sys.mol, &sys.obs_bs, "pbe", &scf, &g.main, &g.nlc)
        .unwrap_or_else(|e| panic!("{system}: v_xc: {e:?}"));
    let ne = usize::try_from(sys.mol.nelec()).unwrap();
    let (na, nb) = (ne.div_ceil(2), ne / 2);
    let label = sys.label.clone();
    move |pdep: &PdepRpaConfig, gw: &GwConfig| {
        let r = run_u_gw(
            &sys.mol,
            &sys.obs,
            &sys.dfbs,
            Operator::coulomb(),
            &scf,
            pdep,
            gw,
            Some((&va, &vb)),
        )
        .unwrap_or_else(|e| panic!("{label}: run_u_gw failed: {e:?}"));
        assert!(
            r.qp_converged_a.iter().chain(&r.qp_converged_b).all(|&c| c),
            "{label}: a U-G0W0@UKS QP solve returned no root"
        );
        [
            frontier(&r.mo_indices, r.eps_qp_a.as_slice().unwrap(), na),
            frontier(&r.mo_indices, r.eps_qp_b.as_slice().unwrap(), nb),
        ]
    }
}

fn pdep_at(n_quad: usize, trunc: f64) -> PdepRpaConfig {
    PdepRpaConfig {
        trunc_thresh: trunc,
        ..default_pdep(n_quad)
    }
}

/// U-G0W0@UKS converges in n_quad: the default grid (40) vs 100 at full
/// rank within `UKS_NQUAD_BAR_MEV`; the full defaults within the row's
/// `BAR_MEV`; and the coarse control (16 points) MISSES the 1 meV bar, so
/// the comparison sees the frequency grid.
fn uks_case(system: &str) {
    let go = uks_frontier(system);
    let gw0 = GwConfig::default();
    let v = go(&pdep_at(100, 0.0), &gw0);
    let d40 = go(&pdep_at(DEFAULT_N_QUAD, 0.0), &gw0);
    let coarse = go(&pdep_at(UKS_COARSE_N_QUAD, 0.0), &gw0);
    let defaults = go(&default_pdep(DEFAULT_N_QUAD), &gw0);
    let mut worst_coarse = 0.0_f64;
    for (s, spin) in ["alpha", "beta"].iter().enumerate() {
        let ctx = format!("{system} U-G0W0@UKS {spin}");
        let w = worst_mev(&ctx, "n_quad = 40", d40[s], v[s]);
        assert!(
            w <= UKS_NQUAD_BAR_MEV,
            "{ctx}: n_quad 40 vs 100 differ by {w:.4} meV > {UKS_NQUAD_BAR_MEV} meV"
        );
        assert_defaults_within_bar(&ctx, [v[s], defaults[s]]);
        worst_coarse = worst_coarse.max(worst_mev(&ctx, "n_quad = 16", coarse[s], v[s]));
    }
    assert!(
        worst_coarse > UKS_NQUAD_BAR_MEV,
        "{system} U-G0W0@UKS: n_quad = {UKS_COARSE_N_QUAD} misses n_quad = 100 by only \
         {worst_coarse:.4} meV; the comparison does not see the frequency grid"
    );
}

#[test]
#[ignore = "validation: GW defaults vs validated settings"]
fn u_g0w0_uks_oh_converges_in_n_quad() {
    uks_case("oh");
}

#[test]
#[ignore = "validation: GW defaults vs validated settings"]
fn u_g0w0_uks_nh2_converges_in_n_quad() {
    uks_case("nh2");
}
