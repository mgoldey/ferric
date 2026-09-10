//! MWE: an over-budget 3-index tensor should RECOMPUTE, not spill to disk.
//!
//! # Why a third backend
//!
//! `ThreeIndexSource` had exactly two modes: the whole tensor resident, or the
//! whole tensor written to disk. There was no in-memory middle, so a budget
//! either admitted an allocation up to 100% of itself or dropped off the cliff
//! the code's own warning describes:
//!
//! ```text
//! [ferric] warning: 3-index tensor (N GB) exceeds the memory budget and is
//! being spilled to disk ... DfK::build re-reads the ENTIRE spilled tensor on
//! every SCF iteration, so each iteration pays roughly
//! (tensor size / disk bandwidth) in pure IO on top of its compute
//! ```
//!
//! `Backend::Recompute` sits between them: hold ONE block, and rebuild each
//! block from `eri3_block` as `for_each_block` walks the tensor. Same bounded
//! footprint as the spill path, but it trades integral recompute for disk IO
//! and writes nothing.
//!
//! ```text
//!   backend      resident            disk      per-pass cost
//!   InCore       naux·nao²·8         none      none
//!   Recompute    block·nao²·8        NONE      recompute integrals
//!   DiskSpill    block·nao²·8        full      re-read whole file
//! ```
//!
//! # Why this needs no `Molecule`
//!
//! `eri3_block(op, &obs, &dfbs, p0, p1)` takes only the two prepared bases —
//! no molecule, no basis-set clone, no FFI rebuild. So the backend borrows
//! what the caller already handed `build`, which is why every existing call
//! site is unchanged and only the TYPE gains a lifetime.
//!
//! # The invariant
//!
//! Recompute must be BIT-IDENTICAL to in-core. `eri3_block` is a pure function
//! of `(op, obs, dfbs, p0, p1)` and is write-once per element, so rebuilding a
//! block reproduces exactly the bytes the in-core tensor held. Anything less
//! means the block boundaries are perturbing the integrals, which would be a
//! real defect rather than a storage choice.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::{AuxBlock, ThreeIndexSource};
use ndarray::Array3;
use std::sync::Arc;

fn fixture() -> (Molecule, Arc<PreparedBasis>, Arc<PreparedBasis>) {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n",
        0,
        1,
    )
    .unwrap();
    let obs = Arc::new(PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap());
    let dfbs = Arc::new(PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap());
    (mol, obs, dfbs)
}

fn materialize(src: &mut ThreeIndexSource, naux: usize, nao: usize) -> Array3<f64> {
    let mut out = Array3::<f64>::zeros((naux, nao, nao));
    src.for_each_block(&mut |blk: AuxBlock| {
        let n = blk.data.shape()[0];
        out.slice_mut(ndarray::s![blk.p0..blk.p0 + n, .., ..]).assign(&blk.data);
        Ok(())
    })
    .unwrap();
    out
}

/// CONTRACT 1: the recomputed tensor is BIT-IDENTICAL to the in-core one.
///
/// The load-bearing contract. `eri3_block` is pure and write-once per element,
/// so rebuilding per block must reproduce the exact bytes — no reassociation
/// is involved because nothing is summed across blocks here.
#[test]
fn the_recomputed_tensor_is_bit_identical_to_in_core() {
    let (_mol, obs, dfbs) = fixture();
    let op = Operator::coulomb();
    let naux = dfbs.nbasis();
    let nao = obs.nbasis();

    let mut in_core = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX / 4).unwrap();
    let reference = materialize(&mut in_core, naux, nao);

    let full = naux * nao * nao * 8;
    let mut recomp =
        ThreeIndexSource::build_recomputing(op, obs.clone(), dfbs.clone(), full / 8).unwrap();
    let got = materialize(&mut recomp, naux, nao);

    let differing =
        reference.iter().zip(got.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(
        differing, 0,
        "the recomputed tensor differs from in-core at {differing} of {} elements. \
         eri3_block is a pure function of (op, obs, dfbs, p0, p1), so rebuilding a block \
         must reproduce it exactly.",
        reference.len()
    );
}

/// CONTRACT 2: an over-budget source picks Recompute, NOT DiskSpill.
///
/// The behavioural claim. Without this, CONTRACT 1 would pass just as well if
/// the tight budget silently kept using the disk path — the whole point is
/// that no file is written.
#[test]
fn an_over_budget_source_recomputes_instead_of_spilling() {
    let (_mol, obs, dfbs) = fixture();
    let op = Operator::coulomb();
    let full = dfbs.nbasis() * obs.nbasis() * obs.nbasis() * 8;
    let src =
        ThreeIndexSource::build_recomputing(op, obs.clone(), dfbs.clone(), full / 8).unwrap();
    assert!(
        src.is_recompute_for_test(),
        "a budget below the full tensor must select the Recompute backend"
    );
    assert!(
        !src.is_spilled_for_test(),
        "and it must NOT spill: writing the tensor to disk is the performance cliff this \
         backend exists to avoid"
    );
}

/// CONTRACT 3: an ample budget still goes in-core.
///
/// The over-rejection guard. A change that pushed every source onto Recompute
/// would trade a free in-memory read for repeated integral evaluation on jobs
/// that comfortably fit — a large slowdown dressed as a memory fix.
#[test]
fn an_ample_budget_still_uses_the_in_core_backend() {
    let (_mol, obs, dfbs) = fixture();
    let op = Operator::coulomb();
    let src =
        ThreeIndexSource::build_recomputing(op, obs.clone(), dfbs.clone(), usize::MAX / 4).unwrap();
    assert!(
        !src.is_recompute_for_test() && !src.is_spilled_for_test(),
        "an ample budget must keep the in-core fast path"
    );
}

/// CONTRACT 4: streaming twice yields the same bytes both times.
///
/// Recompute rebuilds on every pass, unlike in-core which returns views of one
/// stored tensor. A source consumed once per SCF iteration must therefore give
/// identical data on pass 2 as on pass 1 — if it did not, the SCF would see a
/// silently drifting integral set.
#[test]
fn a_second_streaming_pass_reproduces_the_first() {
    let (_mol, obs, dfbs) = fixture();
    let op = Operator::coulomb();
    let naux = dfbs.nbasis();
    let nao = obs.nbasis();
    let full = naux * nao * nao * 8;
    let mut src =
        ThreeIndexSource::build_recomputing(op, obs.clone(), dfbs.clone(), full / 8).unwrap();
    let first = materialize(&mut src, naux, nao);
    let second = materialize(&mut src, naux, nao);
    let differing =
        first.iter().zip(second.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(
        differing, 0,
        "a second pass over the same Recompute source differed at {differing} elements — \
         the SCF streams this once per iteration and must see stable integrals"
    );
}
