//! The packed in-core tier is hard-charged against the shared memory pool, so
//! it must be admitted against what the pool has FREE, not against
//! `budget_bytes` alone.
//!
//! An RSH-style J/K pair (or `df_j_aux != df_k_aux`, or the multi-rank RI-JK
//! path) builds two raw tensors under ONE budget while the first is still
//! resident. Both fit the budget on their own; only one fits the pool. Before
//! the pool check the second was admitted as packed in-core and refused by the
//! pool ("memory pool exhausted") where it used to spill and run.
//!
//! Own test binary: it installs the PROCESS-GLOBAL pool (see
//! `mwe_three_index_pool_charges.rs` for why that must not share a process
//! with unit tests).

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;

#[test]
fn a_second_packed_eligible_source_falls_to_spill_instead_of_being_refused() {
    let mol = Molecule::load_xyz("../../testdata/molecules/alkane_3.xyz").expect("alkane_3.xyz");
    let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let op = Operator::coulomb();
    let (nao, naux) = (obs.nbasis(), dfbs.nbasis());
    let unpacked = naux * nao * nao * 8;
    let packed = naux * (nao * (nao + 1) / 2) * 8;
    let headroom = 64 * nao * nao * 8;
    // One packed tensor + scratch fits; two do not. The budget is below the
    // unpacked tensor, so neither source is unpacked in-core.
    let budget = packed + headroom + packed / 2;
    assert!(budget < unpacked, "fixture must force the packed tier");

    clear_global();
    install_global(MemoryPool::with_capacity_bytes(budget));
    let first = ThreeIndexSource::build(op, &obs, &dfbs, budget).expect("first source builds");
    let second = ThreeIndexSource::build(op, &obs, &dfbs, budget)
        .expect("the second source must fall to spill, not be refused by the pool");
    let first_packed = first.is_packed_incore_for_test();
    let second_spilled = second.is_spilled_for_test();
    drop((first, second));
    clear_global();
    assert!(first_packed, "the first source takes the packed tier");
    assert!(
        second_spilled,
        "the pool has no room for a second packed tensor: it must spill"
    );

    // Negative control: with no pool installed both are admitted as packed.
    let a = ThreeIndexSource::build(op, &obs, &dfbs, budget).unwrap();
    let b = ThreeIndexSource::build(op, &obs, &dfbs, budget).unwrap();
    assert!(a.is_packed_incore_for_test() && b.is_packed_incore_for_test());
}
