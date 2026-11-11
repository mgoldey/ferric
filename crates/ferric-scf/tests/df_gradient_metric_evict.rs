//! The kept metric `eigh` (`ferric_scf::raw_handoff::KeptMetric`) is an
//! evictable pool reservation: debited while kept, counted as available, and
//! freed when a reservation needs its bytes. After an eviction the gradient
//! diagonalises `V` itself and equals the plain gradient.
//!
//! Own test binary: it installs the process-global memory pool, which every
//! other test in a shared binary would see.
//!
//! Mutations seen to fail (2026-11-11):
//! - `MemoryPool::reserve` without the evict-and-retry: the forcing
//!   reservation errors ("fits once the kept metric is evicted").
//! - The evict closure not freeing the data: `is_evicted` stays false.
//! - `with_metric_capture` installing `None`: the reachability count fails.

use ferric_core::memory::pool::{self, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_gradient::dfg_metric_eighs_computed_for_test;
use ferric_scf::gradient::restricted_scf_gradient;
use ferric_scf::raw_handoff::with_fresh_handoff;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const AUX: &str = "def2-universal-jkfit";

#[test]
fn an_evicted_metric_is_recomputed_and_the_gradient_is_unchanged() {
    let p = MemoryPool::with_capacity_bytes(8_000_000_000);
    pool::install_global(p.clone());

    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let bs = ferric_core::basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let base = RhfConfig {
        xc: Some("B3LYP".into()),
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        ..RhfConfig::default()
    }
    .for_geometry_task();
    let cfg = with_fresh_handoff(&base);
    let ctx = ParallelContext::default();
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).expect("SCF");
    assert!(res.converged);
    let slot = cfg.raw_handoff.as_ref().unwrap();
    drop(slot.take());

    let kept = slot.take_metrics();
    assert_eq!(kept.len(), 1, "the SCF kept its metric (reachability)");
    let naux = PreparedBasis::new(&mol, &ferric_core::basis::bundled(AUX).unwrap())
        .unwrap()
        .nbasis();
    assert_eq!(kept[0].reserved_bytes(), (naux * naux + naux) * 8);
    assert_eq!(p.evictable_bytes(), kept[0].reserved_bytes());
    assert!(!kept[0].is_evicted());

    // A reservation that fits only without the kept metric evicts it.
    let free_firm = p.capacity_bytes() - p.outstanding_bytes();
    let forcing = p
        .reserve("forcing", free_firm + 1)
        .expect("fits once the kept metric is evicted");
    assert!(kept[0].is_evicted(), "the kept metric was not freed");
    assert_eq!(p.evictable_bytes(), 0);
    drop(forcing);
    for k in kept {
        slot.put_metric_for_test(k);
    }

    let before = dfg_metric_eighs_computed_for_test();
    let g_evicted =
        restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &cfg, &res).expect("gradient");
    assert_eq!(
        dfg_metric_eighs_computed_for_test() - before,
        1,
        "the gradient did not recompute the evicted metric"
    );
    let g_plain =
        restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &base, &res).expect("gradient");
    let d = g_evicted
        .iter()
        .zip(g_plain.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert_eq!(d, 0.0, "evicted-then-recomputed gradient differs by {d:e}");
    pool::clear_global();
}
