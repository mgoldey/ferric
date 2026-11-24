//! The aux-basis log-det route (`ferric_rpa::energy_route`) against the
//! full-rank static-eigenbasis route it replaces for energy-only runs.
//!
//! # Exactness anchor
//!
//! With every mode kept the static eigenvectors `U` are a full orthogonal
//! rotation of the dressed aux basis, and `ln det ε̃ + tr(I − ε̃)` is invariant
//! under it. So per frequency the aux-basis summand `S_aux(k)` must equal, up
//! to rounding only,
//!
//! * `S_eig(k)`: the same log-det in the eigenbasis
//!   (`eval_trace_log_summands_budgeted` with `U`), and
//! * `S_lam(k) = Σ_α f(λ_α(iω_k))`: the eigenvalue route, which shares no
//!   log-det code with the other two.
//!
//! # The rounding bar (derived, then measured against)
//!
//! Each summand is `ln det A + (M − tr A)` for an `M x M` matrix `A` with
//! eigenvalues in `[1, λ_max]`, `λ_max = λ_max(iω_k)` (`Π ⪰ 0`). Its two
//! terms are computed separately, so rounding does not cancel between them:
//!
//! * `tr A`: `M` additions of entries `≤ λ_max`, error `≲ M·u·λ_max`;
//! * `ln det A` by LU: a backward error `E` with entries `≲ √M·u·λ_max` (the
//!   probabilistic `√M` of a length-`M` inner product) shifts it by
//!   `tr(A⁻¹E)`, a sum of `M` terms each `≲ √M·u·λ_max` (`‖A⁻¹‖₂ ≤ 1`) with
//!   uncorrelated signs: `≲ M·u·λ_max`;
//! * the eigenvalue route instead carries `M` eigenvalue errors
//!   `≲ u·λ_max` each, damped by `|f'(λ)| = |1/λ − 1| < 1`: `≲ M·u·λ_max`.
//!
//! * the assembly `A = I + Σ_ia g_ia y_ia y_iaᵀ` itself: each entry is a
//!   length-`N = nocc·nvir` sum with error `≲ √N·u·λ_max` (Cauchy-Schwarz
//!   bounds every partial sum by the diagonal), and an error `E` in `A` moves
//!   the summand by `tr((A⁻¹ − I)E)`; `‖A⁻¹ − I‖_F ≤ √M`, so with
//!   uncorrelated signs `≲ √M·√N·u·λ_max` per route. This term dominates when
//!   `N ≫ M` (every alkane here);
//! * the eigenbasis and eigenvalue routes first project `y = Uᵀ b̃`: entry
//!   errors `≲ √M·u·|y|` reach the summand the same way, `≲ M·u·λ_max`.
//!
//! The largest pair (aux route against the eigenbasis log-det) sums to
//! `σ_k = (5·M + 2·√(M·N))·u·λ_max(iω_k)`, used for both pairs.
//!
//! Every `√` above is the random-sign model: `σ_k` is a standard deviation,
//! not a bound, and one frequency may sit several `σ` out. Each summand is
//! held to `TAIL·σ_k` with `TAIL = 6` (a two-sided Gaussian tail of 2e-9 per
//! comparison). The energy is held to `(1/2π) Σ_k w_k σ_k`, which already
//! exceeds the standard deviation of the weighted sum by about `√n_k`. The
//! weights grow with `ω` while `λ_max → 1` there, which is why the residual
//! grows with the number of points and with `M` (the 1e-11 to 1e-9 of memory
//! `pdep-logdet-and-eps0-anchor-ceiling`).
//!
//! MEASURED (2026-11-14, cc-pVDZ / cc-pVDZ-RI, RI-JK reference, 20 GL
//! points; serial BLAS sections threaded, so the last bits move between runs):
//! largest `|ΔS_k| / σ_k` 0.48-0.56 (water, M = 84, N = 76), 2.2 (n-octane,
//! M = 700, N = 4225) and 1.0-2.1 (n-hexadecane, M = 1372, N = 16121) over two
//! runs; energies |ΔE| 1.5e-14 (bar 4.3e-12), 1.9e-11 (5.2e-11) and 1.8e-11
//! (1.2e-10) Eh.
//!
//! # Artifact hypothesis
//!
//! If the aux route dropped `tr Π`, or assembled `ε̃` from the wrong tensor,
//! `S_aux` would differ from `S_lam` by `O(tr Π)` (≥ 1e-1 here), orders above
//! a bar of ~1e-12. If the eigenbasis were truncated, `S_eig` would differ
//! from both at the size of the dropped modes' summands.
//!
//! # Mutations (applied to `src/`, run, restored; `git diff` clean)
//!
//! * `energy.rs::dielectric_trace_log_summand` returns `log_det` only (the
//!   `tr Π` term dropped) → `water_ccpvdz_aux_route_matches_the_eigensolve_route`
//!   FAILS on the `S_lam` comparison (both log-det routes drop the same
//!   invariant, so only the eigenvalue route can see it).
//! * `energy_route.rs::static_route_required` without its `trunc_thresh > 0`
//!   branch → `a_truncating_threshold_keeps_the_static_route` FAILS (the run
//!   reports the aux route).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config, RpaIntermediates};
use ferric_rpa::energy::{
    eval_trace_log_summands_aux_budgeted, eval_trace_log_summands_budgeted,
    rpa_correlation_energy_from_summands, trace_log_sum,
};
use ferric_rpa::{
    run_pdep_rpa_energy, run_pdep_rpa_energy_from_intermediates, run_pdep_rpa_from_intermediates,
    PdepRpaConfig, RpaEnergyRoute,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

const JK_AUX: &str = "def2-universal-jkfit";
const RI_AUX: &str = "cc-pvdz-ri";

struct System {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
    inter: RpaIntermediates,
    frozen_core: usize,
}

fn molecule(file: &str) -> Molecule {
    let path = format!(
        "{}/../../testdata/molecules/{file}",
        env!("CARGO_MANIFEST_DIR")
    );
    Molecule::load_xyz(&path).unwrap_or_else(|e| panic!("{path}: {e:?}"))
}

/// RI-JK RHF / cc-pVDZ and the RPA intermediates on cc-pVDZ-RI.
fn system(file: &str, frozen_core: usize) -> System {
    let mol = molecule(file);
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(RI_AUX).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig::default()
        .with_df_j_aux(JK_AUX)
        .with_df_k_aux(JK_AUX)
        .with_density_conv(1e-9);
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "{file}: RHF not converged");
    let mp2_cfg = RiMp2Config {
        frozen_core,
        ..Default::default()
    };
    let inter = compute_rpa_intermediates(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg).unwrap();
    System {
        mol,
        obs,
        dfbs,
        rhf,
        inter,
        frozen_core,
    }
}

fn energy_cfg(frozen_core: usize, trunc_thresh: f64) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core,
        trunc_thresh,
        need_eigenvalues_freq: false,
        ..Default::default()
    }
}

fn eps_slices(s: &System) -> (Vec<f64>, Vec<f64>) {
    let i = &s.inter;
    let e = s.rhf.eps_r();
    (
        e[i.first_occ..i.first_occ + i.nocc].to_vec(),
        e[i.nocc_total..i.nocc_total + i.nvir].to_vec(),
    )
}

/// The anchor: per-frequency summands of the three routes within the derived
/// bar, and the energy entry on the aux route returning the aux summands'
/// energy. Returns the largest |ΔS_k| / σ_k seen.
fn check_routes_agree(tag: &str, s: &System) -> f64 {
    const U: f64 = f64::EPSILON / 2.0;
    const TAIL: f64 = 6.0;
    let op = Operator::coulomb();
    let (eps_occ, eps_vir) = eps_slices(s);

    // Full-rank static route with the per-frequency eigenvalues kept.
    let eig_cfg = PdepRpaConfig {
        frozen_core: s.frozen_core,
        trunc_thresh: 0.0,
        need_eigenvalues_freq: true,
        ..Default::default()
    };
    let eig =
        run_pdep_rpa_from_intermediates(&s.inter, &s.mol, &s.obs, &s.dfbs, op, &s.rhf, &eig_cfg)
            .unwrap();
    let m = s.inter.naux;
    let n_ov = s.inter.nocc * s.inter.nvir;
    let width = 5.0 * m as f64 + 2.0 * ((m * n_ov) as f64).sqrt();
    assert_eq!(eig.n_eigenpotentials, m, "{tag}: full rank expected");
    println!(
        "{tag}: lambda_min(0) - 1 = {:.3e}",
        smallest_static_mode(&eig.eigenvalues_static) - 1.0
    );

    let s_eig = eval_trace_log_summands_budgeted(
        &eig.dressed_eigenvectors,
        &s.inter.b_ov,
        &eps_occ,
        &eps_vir,
        &eig.quad_freqs,
        None,
    )
    .unwrap();
    let s_aux = eval_trace_log_summands_aux_budgeted(
        &s.inter.b_ov,
        &eps_occ,
        &eps_vir,
        &eig.quad_freqs,
        None,
    )
    .unwrap();

    let mut worst: f64 = 0.0;
    let mut e_bar = 0.0;
    for (k, row) in eig.eigenvalues_freq.outer_iter().enumerate() {
        let lam_max = row.iter().copied().fold(1.0, f64::max);
        let sigma = width * U * lam_max;
        e_bar += eig.quad_weights[k] * sigma / (2.0 * std::f64::consts::PI);
        let s_lam = trace_log_sum(row.iter());
        for (what, other) in [("eigenbasis log-det", s_eig[k]), ("eigenvalue", s_lam)] {
            let d = (s_aux[k] - other).abs();
            assert!(
                d <= TAIL * sigma,
                "{tag}: k={k} (omega {:.3e}): aux {:.17e} vs {what} {other:.17e}: \
                 |d| {d:.2e} > {TAIL} sigma = {:.2e}",
                eig.quad_freqs[k],
                s_aux[k],
                TAIL * sigma
            );
            worst = worst.max(d / sigma);
        }
    }

    let e_aux = rpa_correlation_energy_from_summands(&eig.quad_weights, &s_aux);
    let d_e = (e_aux - eig.e_rpa).abs();
    println!(
        "{tag}: M = {m}, N = {n_ov}, E_aux {e_aux:.13} vs E_eigenvalue {:.13}: |dE| {d_e:.2e} \
         (bar {e_bar:.2e}); worst |dS_k|/sigma_k {worst:.3}",
        eig.e_rpa
    );
    assert!(
        d_e <= e_bar,
        "{tag}: energy |dE| {d_e:.2e} > bar {e_bar:.2e}"
    );

    // The energy-only entry takes the aux route and returns that energy.
    let routed = run_pdep_rpa_energy_from_intermediates(
        &s.inter,
        &s.mol,
        &s.obs,
        &s.dfbs,
        op,
        &s.rhf,
        &energy_cfg(s.frozen_core, 0.0),
    )
    .unwrap();
    assert_eq!(routed.route, RpaEnergyRoute::AuxBasisLogDet, "{tag}");
    assert_eq!(routed.e_rpa.to_bits(), e_aux.to_bits(), "{tag}");
    worst
}

#[test]
fn water_ccpvdz_aux_route_matches_the_eigensolve_route() {
    check_routes_agree("water/cc-pVDZ", &system("water.xyz", 1));
}

#[test]
#[ignore = "tier slow (>=60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
fn octane_ccpvdz_aux_route_matches_the_eigensolve_route() {
    check_routes_agree("C8/cc-pVDZ", &system("alkane_8.xyz", 8));
}

#[test]
#[ignore = "tier slow (>=60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
fn hexadecane_ccpvdz_aux_route_matches_the_eigensolve_route() {
    check_routes_agree("C16/cc-pVDZ", &system("alkane_16.xyz", 16));
}

/// A positive `trunc_thresh` keeps the static route, and there it truncates:
/// `Π(0)` of water/cc-pVDZ is singular (`λ_min(0) − 1` at rounding, see
/// [`check_routes_agree`]'s print), so the default `1e-4` drops modes (28 of
/// 84 measured) and moves the energy.
#[test]
fn a_truncating_threshold_keeps_the_static_route() {
    let s = system("water.xyz", 1);
    let run = |thresh: f64| {
        run_pdep_rpa_energy_from_intermediates(
            &s.inter,
            &s.mol,
            &s.obs,
            &s.dfbs,
            Operator::coulomb(),
            &s.rhf,
            &energy_cfg(1, thresh),
        )
        .unwrap()
    };
    let full = run(0.0);
    assert_eq!(full.route, RpaEnergyRoute::AuxBasisLogDet);
    let truncated = run(PdepRpaConfig::default().trunc_thresh);
    assert_eq!(truncated.route, RpaEnergyRoute::StaticEigenbasis);
    println!(
        "water/cc-pVDZ fc=1: {} of {} modes kept at 1e-4",
        truncated.n_eigenpotentials, s.inter.naux
    );
    assert!(truncated.n_eigenpotentials < s.inter.naux);
    assert!((truncated.e_rpa - full.e_rpa).abs() > 0.0);
}

fn smallest_static_mode(eigenvalues_static: &[f64]) -> f64 {
    eigenvalues_static
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

/// A configuration that asks for more than the energy keeps the static route
/// without building anything twice.
#[test]
fn an_eigenvalue_request_keeps_the_static_route() {
    let s = system("water.xyz", 1);
    let cfg = PdepRpaConfig {
        need_eigenvalues_freq: true,
        ..energy_cfg(1, 0.0)
    };
    let r =
        run_pdep_rpa_energy(&s.mol, &s.obs, &s.dfbs, Operator::coulomb(), &s.rhf, &cfg).unwrap();
    assert_eq!(r.route, RpaEnergyRoute::StaticEigenbasis);
    assert_eq!(r.eigenvalues_freq.ncols(), s.inter.naux);
}
