//! MWE: the disk spill must store the packed μν triangle, not the full square.
//!
//! # Why
//!
//! `(P|μν)` is symmetric in μν — `eri3_block` computes only `s2 in 0..=s1` and
//! then writes BOTH `(μν)` and `(νμ)`:
//!
//! ```text
//! *base.add(pl * stride0 + mm * stride1 + nn) = val;
//! *base.add(pl * stride0 + nn * stride1 + mm) = val;
//! ```
//!
//! So the spill file held every off-diagonal element twice. Storing the lower
//! triangle (`nao*(nao+1)/2` per aux row, PySCF's `nao_pair`) halves it:
//!
//! ```text
//!   system                disk full   disk packed
//!   benzene/aug-cc-pVTZ     2.07 GB       1.04 GB
//!   danuglipron/def2-SVP   10.98 GB       5.50 GB
//!   danuglipron/def2-TZVP 102.40 GB      51.23 GB
//! ```
//!
//! That is 2x less written AND 2x less re-read on every SCF iteration — which
//! is the whole cost of the spill path, per its own warning: "DfK::build
//! re-reads the ENTIRE spilled tensor on every SCF iteration ... at {x} GB and
//! ~150 MB/s that is ~{y} min per iteration."
//!
//! # Deliberately scoped to the SPILL backend only
//!
//! The in-core backend keeps the full `(band, nao, nao)` layout. Packing it too
//! would be ~2x resident RAM as well, but it would change the DF-J contraction
//! from `Σ_μν` over `nao²` terms to `Σ_{μ≥ν}` over `nao(nao+1)/2` terms with
//! off-diagonals doubled — mathematically exact, numerically NOT bit-identical,
//! and on the default SCF path where this repo pins reference energies. That is
//! a separate, deliberate numerics change; this is a pure storage change.
//!
//! # The invariant that makes this safe
//!
//! `for_each_block` already reads into a `scratch` buffer and yields a VIEW of
//! it. Unpacking on read therefore keeps the `AuxBlock` shape at
//! `(b, nao, nao)` exactly as before, so all 15 `for_each_block` consumers
//! across df_j/df_k/rimp2/oo_rimp2 are untouched and see bit-identical values.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::{AuxBlock, ThreeIndexSource};
use ndarray::Array3;

/// Water/cc-pVDZ with an RI aux basis: naux=84, nao=24 — small, but large
/// enough that the triangle is a real saving (24*25/2 = 300 vs 576).
fn fixture() -> (Molecule, PreparedBasis, PreparedBasis) {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n",
        0,
        1,
    )
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    (mol, obs, dfbs)
}

/// Materialise the whole tensor through the public streaming API.
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

/// CONTRACT 1: the spilled tensor is BIT-IDENTICAL to the in-core one.
///
/// The load-bearing contract. Packing on disk and unpacking on read is a pure
/// storage change: every element the consumer sees must be the same bits it
/// saw before. Anything less means the packing lost or reordered data.
///
/// Note this is a stronger bar than the sibling
/// `spilled_dressed_tensor_stays_within_a_few_ulp_of_in_core`, and correctly
/// so — that one covers the DRESSED tensor, where k-blocking legitimately
/// regroups a sum. Here nothing is summed; bytes are stored and returned.
#[test]
fn the_spilled_tensor_is_bit_identical_to_the_in_core_one() {
    let (_mol, obs, dfbs) = fixture();
    let op = Operator::coulomb();
    let naux = dfbs.nbasis();
    let nao = obs.nbasis();

    let mut in_core = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX / 4).unwrap();
    let reference = materialize(&mut in_core, naux, nao);

    // A budget far below the full tensor forces the spill backend.
    let full_bytes = naux * nao * nao * 8;
    let mut spilled = ThreeIndexSource::build(op, &obs, &dfbs, full_bytes / 8).unwrap();
    let got = materialize(&mut spilled, naux, nao);

    let differing =
        reference.iter().zip(got.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(
        differing, 0,
        "the spilled tensor differs from in-core at {differing} of {} elements. Packing the \
         μν triangle on disk must be a pure STORAGE change — unpacking on read has to \
         reproduce the exact same (b, nao, nao) block the consumer saw before.",
        reference.len()
    );
}

/// CONTRACT 2: μν symmetry actually holds, so packing is valid at all.
///
/// The premise the whole change rests on. If `(P|μν) != (P|νμ)` for any
/// element, storing one triangle silently loses information — and CONTRACT 1
/// would not catch it, because both paths would be equally wrong.
#[test]
fn the_raw_tensor_is_symmetric_in_mu_nu() {
    let (_mol, obs, dfbs) = fixture();
    let op = Operator::coulomb();
    let naux = dfbs.nbasis();
    let nao = obs.nbasis();
    let mut src = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX / 4).unwrap();
    let t = materialize(&mut src, naux, nao);

    for p in 0..naux {
        for mu in 0..nao {
            for nu in 0..mu {
                assert_eq!(
                    t[(p, mu, nu)].to_bits(),
                    t[(p, nu, mu)].to_bits(),
                    "(P|μν) is not symmetric at P={p}, μ={mu}, ν={nu}: {} vs {}. Packing the \
                     lower triangle would discard the difference.",
                    t[(p, mu, nu)],
                    t[(p, nu, mu)]
                );
            }
        }
    }
}

/// CONTRACT 3 (reachability): the tight budget really did spill.
///
/// Without this, CONTRACTS 1-2 could both pass while the "spilled" source
/// silently stayed in-core, testing nothing — the inertness trap this repo has
/// hit repeatedly.
#[test]
fn the_tight_budget_actually_spills() {
    let (_mol, obs, dfbs) = fixture();
    let op = Operator::coulomb();
    let naux = dfbs.nbasis();
    let nao = obs.nbasis();
    let full_bytes = naux * nao * nao * 8;
    let src = ThreeIndexSource::build(op, &obs, &dfbs, full_bytes / 8).unwrap();
    assert!(
        src.is_spilled_for_test(),
        "the tight budget did not force a spill, so the packing path was never exercised \
         and CONTRACTS 1-2 asserted nothing"
    );
    let in_core = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX / 4).unwrap();
    assert!(
        !in_core.is_spilled_for_test(),
        "the reference must be the in-core backend"
    );
}
