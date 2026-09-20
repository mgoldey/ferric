//! 1-2/1-3 exclusion and 1-4 pair derivation on RINGS.
//!
//! The two OpenMM cross-validation cases (`vs_openmm.rs`) are ethane and
//! ethanol -- both acyclic. In an acyclic molecule every atom pair has exactly
//! one path, so the BFS in `MmTopology::new` is never asked the question that
//! `topology.rs`'s module docs answer in prose:
//!
//! > a pair reachable by both a length-<=2 path and a length-3 path is
//! > excluded, not scaled -- exclusion always wins.
//!
//! Rings are where that arises, and rings are most drug-like ligands. Until
//! this file the rule was documented and untested.
//!
//! The expected sets below are not this author's reading of the AMBER
//! convention. They were taken from OpenMM's own
//! `NonbondedForce::createExceptionsFromBonds`, queried directly on the same
//! bond graphs (see the benzene case), so this is a cross-check against an
//! independent implementation rather than a restatement of the same belief.

use ferric_mm::{Bond, LjParams, MmTopology};
use std::collections::HashSet;

/// A bare carbon ring: `n` atoms bonded in a cycle, no hydrogens, uniform
/// parameters. Only the bond GRAPH matters for exclusion derivation.
fn ring(n: usize) -> MmTopology {
    let charges = vec![0.0; n];
    let lj = vec![
        LjParams {
            sigma: 3.4,
            epsilon: 0.1
        };
        n
    ];
    let bonds: Vec<Bond> = (0..n)
        .map(|i| Bond {
            i,
            j: (i + 1) % n,
            k: 300.0,
            r0: 1.5,
        })
        .collect();
    MmTopology::new(charges, lj, bonds, vec![], vec![]).expect("ring topology")
}

fn pairs(list: &[(usize, usize)]) -> HashSet<(usize, usize)> {
    list.iter().copied().collect()
}

#[test]
fn a_three_ring_excludes_every_pair_and_has_no_one_four() {
    // In cyclopropane every pair of carbons is simultaneously 1-2 (direct
    // bond) and 1-3 (around the ring). There is nowhere for a 1-4 to live, so
    // an implementation that let the long path win would produce a scaled
    // pair here and a different nonbonded energy.
    let top = ring(3);
    assert_eq!(*top.exclusions(), pairs(&[(0, 1), (0, 2), (1, 2)]));
    assert!(
        top.pairs14().is_empty(),
        "a 3-ring has no 1-4 pairs, got {:?}",
        top.pairs14()
    );
}

#[test]
fn four_and_five_rings_are_fully_excluded_with_no_one_four() {
    // Every pair is within three bonds the short way round, so the whole ring
    // is exclusions and `pairs14` stays empty. This is the assertion that
    // fails loudest if the depth bookkeeping is off by one.
    for n in [4usize, 5] {
        let top = ring(n);
        let expected: HashSet<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();
        assert_eq!(*top.exclusions(), expected, "{n}-ring exclusions");
        assert!(
            top.pairs14().is_empty(),
            "{n}-ring should have no 1-4 pairs, got {:?}",
            top.pairs14()
        );
    }
}

#[test]
fn a_six_ring_scales_only_the_para_pairs() {
    // Benzene is the discriminating case: the para pairs are genuine 1-4s,
    // while the meta pairs (0,2)-style reach depth 3 the long way round and
    // depth 2 the short way. Exclusion must win for those.
    //
    // Cross-checked against OpenMM, which returns exactly these 15 exceptions
    // for the same bond graph -- 12 excluded, and (0,3)/(1,4)/(2,5) scaled:
    //
    //   nb.createExceptionsFromBonds([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)], ..)
    let top = ring(6);
    assert_eq!(
        *top.pairs14(),
        pairs(&[(0, 3), (1, 4), (2, 5)]),
        "only para pairs are 1-4"
    );
    let expected_exclusions = pairs(&[
        (0, 1),
        (0, 2),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
    ]);
    assert_eq!(*top.exclusions(), expected_exclusions);
    assert_eq!(top.exclusions().len() + top.pairs14().len(), 15);
}

#[test]
fn exclusions_and_one_four_pairs_never_overlap() {
    // The invariant the module docs promise, asserted on every ring size and
    // on a fused bicyclic where two rings share an edge -- the case where a
    // pair genuinely has several short paths of different lengths.
    for n in 3..=8 {
        let top = ring(n);
        let overlap: Vec<_> = top.exclusions().intersection(top.pairs14()).collect();
        assert!(overlap.is_empty(), "{n}-ring overlap: {overlap:?}");
    }

    // Bicyclo: two fused 5-rings sharing the 0-1 bond.
    let charges = vec![0.0; 8];
    let lj = vec![
        LjParams {
            sigma: 3.4,
            epsilon: 0.1
        };
        8
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (1, 5),
        (5, 6),
        (6, 7),
        (7, 0),
    ];
    let bonds: Vec<Bond> = edges
        .iter()
        .map(|&(i, j)| Bond {
            i,
            j,
            k: 300.0,
            r0: 1.5,
        })
        .collect();
    let top = MmTopology::new(charges, lj, bonds, vec![], vec![]).expect("bicyclic");
    let overlap: Vec<_> = top.exclusions().intersection(top.pairs14()).collect();
    assert!(overlap.is_empty(), "bicyclic overlap: {overlap:?}");
}

#[test]
fn a_chain_still_behaves_so_the_ring_rule_costs_nothing() {
    // Negative control: butane's C1-C4 IS a 1-4 pair. If the ring handling
    // were implemented by suppressing long paths outright, this would come
    // back empty and every acyclic 1-4 interaction would silently vanish.
    let charges = vec![0.0; 4];
    let lj = vec![
        LjParams {
            sigma: 3.4,
            epsilon: 0.1
        };
        4
    ];
    let bonds: Vec<Bond> = [(0, 1), (1, 2), (2, 3)]
        .iter()
        .map(|&(i, j)| Bond {
            i,
            j,
            k: 300.0,
            r0: 1.5,
        })
        .collect();
    let top = MmTopology::new(charges, lj, bonds, vec![], vec![]).expect("chain");
    assert_eq!(*top.pairs14(), pairs(&[(0, 3)]), "C1-C4 must be a 1-4 pair");
    assert_eq!(
        *top.exclusions(),
        pairs(&[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    );
}
