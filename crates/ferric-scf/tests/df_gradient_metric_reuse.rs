//! The DF-K gradient reuses the `eigh(V)` of the auxiliary metric that the
//! SCF's DF-K computed for `V^{-1/2}` (`ferric_scf::raw_handoff`, the
//! `KeptMetric` half of the hand-off) instead of diagonalising `V` again in
//! `dfg:metric-eigh`, and gets the gradient it would have recomputed.
//!
//! Exactness: DF-K (`v_inv_sqrt_lindep`) and the gradient both build `V` with
//! `coulomb_metric_2c(op, dfbs)` and decompose it with `eigh_dc(Upper)` under
//! the same BLAS-thread wrapper; the gradient takes the kept decomposition only
//! when its own `V` hashes to the same bits. Given the same matrix, the same
//! routine and the same thread count, the recomputed `eigh` is the kept one, so
//! the expected difference is exactly 0 and the bar is 1e-12 Ha/Bohr. What
//! would move it: a decomposition of another geometry (O(step)), of another
//! operator (erfc vs erf, O(1)), or a mis-ordered eigenpair.
//!
//! The raw-tensor half of the hand-off is emptied before each reuse gradient,
//! so these tests compare the metric reuse ALONE.
//!
//! Reachability: the slot holds the expected number of metrics after the SCF
//! (else "reuse" would silently be a recompute equal to itself), and the
//! `dfg:metric-eigh` counter tells reuse (0 eighs) from recompute.
//!
//! Mutations seen to fail (2026-11-11):
//! - `KeptMetric::check` without the bit-hash comparison:
//!   `mismatched_geometry_metric_is_not_used` fails ("a metric eigh from
//!   another geometry was used").
//! - Without the bit-hash AND the operator comparison: also the wB97X case of
//!   `reused_metric_eigh_gradient_equals_recomputed` (the erf pass gets the
//!   erfc decomposition; needs the reversed slot order below).
//! - Without the name, shape and bit-hash comparisons:
//!   `mismatched_aux_basis_metric_is_not_used` fails.
//! - `with_metric_capture` installing `None`: every reuse test fails on the
//!   reachability count.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_gradient::dfg_metric_eighs_computed_for_test;
use ferric_scf::gradient::{restricted_scf_gradient, unrestricted_scf_gradient};
use ferric_scf::raw_handoff::{with_fresh_handoff, RawHandoff};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const AUX: &str = "def2-universal-jkfit";
const BASIS: &str = "def2-svp";
/// The expected difference is exactly 0 (module docs).
const BAR: f64 = 1e-12;

fn geometry_config(xc: &str, aux: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some(aux.into()),
        df_k_aux: Some(aux.into()),
        ..RhfConfig::default()
    }
    .for_geometry_task()
}

fn handoff(cfg: &RhfConfig) -> &RawHandoff {
    cfg.raw_handoff.as_ref().expect("hand-off installed")
}

struct Setup {
    mol: Molecule,
    bs: ferric_core::basis::BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn setup(mol: Molecule) -> Setup {
    let bs = ferric_core::basis::bundled(BASIS).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
    Setup {
        mol,
        bs,
        prep,
        bounds,
    }
}

fn scf(s: &Setup, open: bool, cfg: &RhfConfig) -> ScfResult {
    let ctx = ParallelContext::default();
    let res = if open {
        solve_uhf(&ctx, &s.mol, &s.prep, &s.bounds, cfg)
    } else {
        solve_rhf(&ctx, &s.mol, &s.prep, Operator::coulomb(), &s.bounds, cfg)
    }
    .expect("SCF");
    assert!(res.converged, "SCF did not converge");
    res
}

/// The gradient and how many metric `eigh`s its DF-K passes computed.
fn gradient(s: &Setup, open: bool, cfg: &RhfConfig, res: &ScfResult) -> (Array2<f64>, usize) {
    let op = Operator::coulomb();
    let before = dfg_metric_eighs_computed_for_test();
    let g = if open {
        unrestricted_scf_gradient(&s.mol, &s.prep, &s.bs, op, &s.bounds, cfg, res)
    } else {
        restricted_scf_gradient(&s.mol, &s.prep, &s.bs, op, &s.bounds, cfg, res)
    }
    .expect("gradient");
    (g, dfg_metric_eighs_computed_for_test() - before)
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// SCF with a hand-off (raw tensor discarded), gradient reusing the kept
/// metric `eigh`s, gradient recomputing them; returns max |Δg| after asserting
/// the reuse really happened for all `n_metrics` metrics.
fn reuse_vs_recompute(s: &Setup, open: bool, cfg: &RhfConfig, n_metrics: usize) -> f64 {
    let base = cfg.clone();
    let cfg = with_fresh_handoff(cfg);
    let res = scf(s, open, &cfg);
    let slot = handoff(&cfg);
    drop(slot.take()); // metric reuse alone
    // Reverse the kept order (DF-K offers erfc then erf, the order the
    // gradient asks), so a lookup that ignored which metric it matched
    // would hand the erf pass the erfc decomposition and fail.
    let mut kept = slot.take_metrics();
    kept.reverse();
    for k in kept {
        slot.put_metric_for_test(k);
    }
    assert_eq!(
        slot.metric_count(),
        n_metrics,
        "the SCF kept its metric eigh(s) (reachability)"
    );

    let (g_reuse, eighs_reuse) = gradient(s, open, &cfg, &res);
    assert_eq!(slot.metric_count(), 0, "the gradient took the metrics");
    assert_eq!(eighs_reuse, 0, "reuse computed a metric eigh");
    let (g_rec, eighs_rec) = gradient(s, open, &base, &res);
    assert_eq!(eighs_rec, n_metrics, "recompute computed every metric eigh");
    assert!(
        g_rec.iter().any(|v| v.abs() > 1e-3),
        "vacuous: the gradient is ~zero"
    );
    max_abs_diff(&g_reuse, &g_rec)
}

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz")
}

#[test]
fn reused_metric_eigh_gradient_equals_recomputed() {
    let s = setup(water());
    // ω = 0 hybrid: one Coulomb metric, shared by RI-J and RI-K.
    let d = reuse_vs_recompute(&s, false, &geometry_config("B3LYP", AUX), 1);
    eprintln!("water/def2-SVP B3LYP: max|g_reuse - g_recompute| = {d:.3e}");
    assert!(d <= BAR, "B3LYP: {d:e}");

    // Range-separated hybrid: the erfc and erf metrics (two decompositions,
    // the erf one with dropped modes), each matched to its own operator.
    let d = reuse_vs_recompute(&s, false, &geometry_config("HYB_GGA_XC_WB97X", AUX), 2);
    eprintln!("water/def2-SVP wB97X: max|g_reuse - g_recompute| = {d:.3e}");
    assert!(d <= BAR, "wB97X: {d:e}");
}

#[test]
fn reused_metric_eigh_open_shell_gradient_equals_recomputed() {
    let oh = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
    let s = setup(Molecule::parse_xyz(oh, 0, 2).expect("OH"));
    let d = reuse_vs_recompute(&s, true, &geometry_config("B3LYP", AUX), 1);
    assert!(d <= BAR, "OH UKS: {d:e}");
}

/// Geometry B's gradient with geometry A's kept metric in the slot: it must
/// recompute (the metric bits differ) and equal the plain gradient.
#[test]
fn mismatched_geometry_metric_is_not_used() {
    let a = setup(water());
    let mut moved = water();
    moved.atoms[1].y += 0.01;
    let b = setup(moved);
    let base = geometry_config("B3LYP", AUX);

    let cfg_a = with_fresh_handoff(&base);
    let _ = scf(&a, false, &cfg_a);
    let mut kept_a = handoff(&cfg_a).take_metrics();
    assert_eq!(kept_a.len(), 1, "A kept its metric");

    let cfg_b = with_fresh_handoff(&base);
    let res_b = scf(&b, false, &cfg_b);
    drop(handoff(&cfg_b).take());
    assert_eq!(handoff(&cfg_b).metric_count(), 1, "B kept its metric");
    // Same operator and aux name: replaces B's own.
    handoff(&cfg_b).put_metric_for_test(kept_a.pop().unwrap());
    assert_eq!(handoff(&cfg_b).metric_count(), 1);
    let (g_stale, eighs) = gradient(&b, false, &cfg_b, &res_b);
    assert_eq!(eighs, 1, "a metric eigh from another geometry was used");

    let (g_plain, _) = gradient(&b, false, &base, &res_b);
    let d = max_abs_diff(&g_stale, &g_plain);
    assert!(d <= BAR, "mismatched hand-off moved the gradient by {d:e}");
}

/// A gradient fitted in another aux basis than the kept metric's must not
/// use it (and must equal its plain gradient).
#[test]
fn mismatched_aux_basis_metric_is_not_used() {
    let s = setup(water());
    let other = "def2-svp-rifit";
    let cfg_a = with_fresh_handoff(&geometry_config("B3LYP", other));
    let _ = scf(&s, false, &cfg_a);
    let mut kept_a = handoff(&cfg_a).take_metrics();
    assert_eq!(kept_a.len(), 1, "the {other} SCF kept its metric");

    let base = geometry_config("B3LYP", AUX);
    let cfg_b = with_fresh_handoff(&base);
    let res_b = scf(&s, false, &cfg_b);
    drop(handoff(&cfg_b).take());
    drop(handoff(&cfg_b).take_metrics());
    handoff(&cfg_b).put_metric_for_test(kept_a.pop().unwrap());
    let (g_stale, eighs) = gradient(&s, false, &cfg_b, &res_b);
    assert_eq!(eighs, 1, "a metric eigh in another aux basis was used");
    let (g_plain, _) = gradient(&s, false, &base, &res_b);
    let d = max_abs_diff(&g_stale, &g_plain);
    assert!(d <= BAR, "mismatched aux moved the gradient by {d:e}");
}

/// A solve without a hand-off (an energy-only run) keeps nothing; the
/// gradient after it recomputes.
#[test]
fn no_handoff_keeps_no_metric() {
    let s = setup(water());
    let base = geometry_config("B3LYP", AUX);
    let res = scf(&s, false, &base);
    let (_, eighs) = gradient(&s, false, &base, &res);
    assert_eq!(eighs, 1);
}
