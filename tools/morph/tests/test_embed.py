"""Tests for analogue 3D embedding.

The failure mode worth guarding: an analogue that CANNOT be embedded (strained
ring, bad valence, untypable atom for MMFF) must report `usable == False` and
carry an explanatory `error`. If it instead came back with zero conformers and
no error, a downstream ranking would treat it as evaluated-and-unremarkable
rather than not-evaluated -- the same fabrication class as `tools/tox`'s 0.0.
"""
from __future__ import annotations

import math

import pytest

from tools.morph.design import Analogue, PharmacophoreSpec
from tools.morph.embed import embed_analogue

# Small and fast: embedding the 71-atom parent many times over would dominate
# the suite's runtime for no extra coverage of the code paths here.
# A minimal spec: the library must not depend on any campaign's pharmacophore.
CARBOXYLIC_ACID = PharmacophoreSpec(
    features=(("acid", "[CX3](=O)[OX2H1]", 1),)
)

SMALL = Analogue(
    label="ibuprofen-like",
    smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    hypothesis="test-fixture",
    rationale="small, well-behaved, MMFF-typable molecule for embedding tests",
    pharmacophore=CARBOXYLIC_ACID,
)


def test_embeds_a_well_behaved_molecule():
    e = embed_analogue(SMALL, n_conformers=3)
    assert e.usable, e.error
    assert e.error is None
    assert 1 <= e.n_conformers <= 3
    assert len(e.symbols) == len(e.conformers[0])
    assert len(e.mmff_energies) == e.n_conformers


def test_conformers_are_three_dimensional():
    """A degenerate all-planar or all-zero embedding would still have the right
    shape, so check the geometry is actually spatial."""
    e = embed_analogue(SMALL, n_conformers=2)
    coords = e.conformers[0]
    for axis in range(3):
        spread = max(c[axis] for c in coords) - min(c[axis] for c in coords)
        assert spread > 0.5, f"axis {axis} has no extent -- degenerate embedding"


def test_bond_lengths_are_physical():
    """Guards against a unit error or a collapsed geometry: the closest pair of
    atoms must be a plausible bond (>0.85 A), and no two atoms coincident."""
    e = embed_analogue(SMALL, n_conformers=1)
    coords = e.conformers[0]
    dmin = min(
        math.dist(coords[i], coords[j])
        for i in range(len(coords))
        for j in range(i + 1, len(coords))
    )
    assert 0.85 < dmin < 2.0, f"closest atom pair {dmin:.3f} A is not a bond length"


def test_best_index_is_the_lowest_energy_conformer():
    e = embed_analogue(SMALL, n_conformers=4)
    assert e.best_index is not None
    assert e.mmff_energies[e.best_index] == pytest.approx(min(e.mmff_energies))


def test_embedding_is_reproducible_with_a_fixed_seed():
    """ETKDG is stochastic. Without a fixed seed, an analogue's ranking would
    change between runs for reasons unrelated to chemistry."""
    a = embed_analogue(SMALL, n_conformers=3, random_seed=1234)
    b = embed_analogue(SMALL, n_conformers=3, random_seed=1234)
    assert a.n_conformers == b.n_conformers
    for ca, cb in zip(a.conformers, b.conformers):
        for pa, pb in zip(ca, cb):
            for x, y in zip(pa, pb):
                assert x == pytest.approx(y, abs=1e-6)


def test_different_seeds_can_give_different_geometries():
    """Reachability check for the test above: confirm the seed is actually
    doing something, so the reproducibility assert isn't passing trivially
    because the embedder is deterministic regardless."""
    a = embed_analogue(SMALL, n_conformers=6, random_seed=1)
    b = embed_analogue(SMALL, n_conformers=6, random_seed=999_983)
    same = a.n_conformers == b.n_conformers and all(
        all(
            all(abs(x - y) < 1e-6 for x, y in zip(pa, pb))
            for pa, pb in zip(ca, cb)
        )
        for ca, cb in zip(a.conformers, b.conformers)
    )
    assert not same, "seed has no effect -- the reproducibility test is vacuous"


def test_unparseable_smiles_is_reported_not_raised():
    bad = Analogue(label="bad", smiles="C1CC(((", hypothesis="h", rationale="r",
                   pharmacophore=CARBOXYLIC_ACID)
    e = embed_analogue(bad)
    assert not e.usable
    assert e.error is not None and "unparseable" in e.error
    assert e.n_conformers == 0
    assert e.best_index is None, (
        "an unevaluated analogue must not expose a best conformer"
    )


def test_impossible_geometry_is_reported_as_unevaluated():
    """A molecule ETKDG cannot embed must come back UNUSABLE with a reason --
    never as zero conformers and no error, which a ranking would read as
    'evaluated, unremarkable'.
    """
    # A [2.2.2]propellane-like over-constrained cage; if RDKit ever manages to
    # embed it, the assertion below still holds (usable => no error).
    hard = Analogue(
        label="overconstrained",
        smiles="C12C3C1C1C2C31",
        hypothesis="h",
        rationale="deliberately over-constrained cage to exercise the failure path",
        pharmacophore=CARBOXYLIC_ACID,
    )
    e = embed_analogue(hard, n_conformers=2)
    assert e.usable == (e.error is None), (
        "usable and error must never disagree: an analogue is either evaluated "
        "with geometry, or unevaluated WITH a stated reason"
    )
    if not e.usable:
        assert e.error and len(e.error) > 20


def test_write_xyz_round_trips_through_the_active_site_reader():
    """The embedder's output must be consumable by the existing pipeline
    unchanged -- otherwise the two halves of the campaign don't connect."""
    import tempfile

    from tools.active_site.ligand_embedding import _read_xyz_atoms

    e = embed_analogue(SMALL, n_conformers=2)
    with tempfile.TemporaryDirectory() as d:
        paths = e.write_xyz(d)
        assert len(paths) == e.n_conformers
        atoms = _read_xyz_atoms(paths[0])
        assert [a.symbol for a in atoms] == e.symbols
        for got, want in zip(atoms, e.conformers[0]):
            assert got.x == pytest.approx(want[0], abs=1e-7)
            assert got.y == pytest.approx(want[1], abs=1e-7)
            assert got.z == pytest.approx(want[2], abs=1e-7)
