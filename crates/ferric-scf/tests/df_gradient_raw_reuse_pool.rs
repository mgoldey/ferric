//! A kept PACKED raw tensor under a memory pool with no room for the
//! gradient's unpack block beside it: the gradient must drop the tensor and
//! recompute (the old behaviour), never return a hard pool error.
//!
//! Window: the pool has room for the intermediates `I` plus one unpacked
//! block `B` only after the kept tensor `R` is released
//! (`R + I <= C < R + I + B`, made by a ballast reservation leaving
//! `I + B - 1` free beside `R`). The reuse path then must not take the
//! tensor; the recompute path reserves `I` and then `B`, which fit once `R`
//! is gone (`R >= B`). The second half leaves exactly `I + B` free: reuse.
//!
//! Its own test binary: it installs the process-global pool.
//!
//! Mutation seen to fail (2026-11-11): `charge_intermediates` ignoring the
//! unpack block (try-reserving `I` only, the pass then hard-reserving `B`):
//! the window case returns "memory pool exhausted: \"DF-JK gradient 3-index
//! block (computed once, not stored)\" needs 0.113 GB but only 0.113 GB ... is
//! free" (n-octane/def2-SVP).

use ferric_core::memory::pool;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_gradient::{
    df_jk_gradient_at_precision, df_jk_gradient_with_raw, dfg_3c_blocks_computed_for_test,
    pass_block_bytes, DfKChannel,
};
use ferric_scf::df_j::RiJPrecision;
use ferric_scf::raw_handoff::{gradient_intermediate_bytes, with_fresh_handoff};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const AUX: &str = "def2-universal-jkfit";

#[test]
fn packed_reuse_under_a_tight_pool_recomputes_instead_of_failing() {
    pool::install_global(pool::MemoryPool::with_capacity_bytes(64 << 30));
    let mol = Molecule::load_xyz("../../testdata/molecules/alkane_8.xyz").expect("octane");
    let bs = ferric_core::basis::bundled("def2-svp").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let auxbs = ferric_core::basis::bundled(AUX).unwrap();
    let dfbs = PreparedBasis::new(&mol, &auxbs).unwrap();
    let (nao, naux) = (obs.nbasis(), dfbs.nbasis());
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    // The budget that holds the raw tensor packed (see df_gradient_raw_reuse.rs).
    let full = naux * nao * nao * 8;
    let dressed = ferric_scf::df_k::DfK::packed_band_bytes(
        naux,
        nao,
        ferric_scf::df_k::RiKPrecision::resolve(None, nao),
    );
    let budget = full + dressed - 1;
    let cfg = with_fresh_handoff(
        &RhfConfig {
            xc: Some("B3LYP".into()),
            df_j_aux: Some(AUX.into()),
            df_k_aux: Some(AUX.into()),
            three_index_budget_bytes: budget,
            ..RhfConfig::default()
        }
        .for_geometry_task(),
    );
    let res = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(res.converged);
    let slot = cfg.raw_handoff.as_ref().unwrap();
    let d = res.density_total();
    let nocc = (mol.nelec() / 2) as usize;
    let ch = [DfKChannel {
        density: d,
        alpha: -0.05,
    }];
    let i_bytes = gradient_intermediate_bytes(naux, &[nocc]);
    let block = pass_block_bytes(&obs, &dfbs, budget);
    let plain = |budget| {
        df_jk_gradient_at_precision(
            mol.atoms.len(),
            &obs,
            &dfbs,
            op,
            Some(d),
            &ch,
            budget,
            RiJPrecision::F64,
        )
        .unwrap()
    };

    for (free, expect_reuse) in [(i_bytes + block - 1, false), (i_bytes + block, true)] {
        let kept = slot.take().unwrap_or_else(|| {
            // Second round: rebuild the slot by re-solving.
            let r = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
            assert!(r.converged);
            slot.take().expect("re-solve kept its tensor")
        });
        assert_eq!(
            kept.bytes(),
            naux * nao * (nao + 1) / 2 * 8,
            "not the packed tier"
        );
        // The kept tensor is charged to the pool; leave exactly `free` beside it.
        let avail = pool::global_available_bytes().unwrap();
        let ballast = pool::reserve_global("test ballast", avail - free).unwrap();
        let before = dfg_3c_blocks_computed_for_test();
        let g = df_jk_gradient_with_raw(
            &mol,
            (&obs, &dfbs, AUX),
            op,
            Some(d),
            &ch,
            budget,
            RiJPrecision::F64,
            Some(kept),
        )
        .unwrap_or_else(|e| panic!("free = {free}: the gradient failed under the pool: {e}"));
        let computed = dfg_3c_blocks_computed_for_test() - before;
        drop(ballast);
        assert_eq!(
            computed == 0,
            expect_reuse,
            "free = {free}: computed {computed} blocks"
        );
        let reference = plain(budget);
        for (a, b) in g.iter().zip(reference.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "free = {free}: {a:e} vs {b:e}");
        }
    }
    pool::clear_global();
}
