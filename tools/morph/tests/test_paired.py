"""Tests for the paired-ddE construction.

THE EXACTNESS ANCHOR COMES FIRST, per the repo's experimental protocol: pairing
a molecule with ITSELF must give ddE == 0 exactly, with zero spread. It is
written before any variance-reduction measurement, and no sweep runs until it
passes.
"""

from __future__ import annotations

import math

import pytest

from tools.morph.paired import PairedPose, paired_ddE

rdkit = pytest.importorskip("rdkit", reason="paired construction needs RDKit")


def _self_pairs(n=6):
    """n poses of a molecule paired with itself -- the trivial limit."""
    import random

    rng = random.Random(7)
    syms = ["C", "O", "H", "H"]
    out = []
    for i in range(n):
        c = [(rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in syms]
        out.append(
            PairedPose(i, syms, c, syms, list(c), [(j, j) for j in range(len(syms))])
        )
    return out


def _spread_energy():
    """An energy with LARGE pose-to-pose scatter -- the thing pairing must cancel."""

    def e(symbols, coords):
        return 40.0 * coords[0][0] + 3.0 * len(symbols)

    return e


def test_self_pairing_gives_exactly_zero_the_exactness_anchor():
    """THE ANCHOR. A molecule against itself: every difference is exactly 0.

    This is the trivial limit in which the pairing does nothing, and it is the
    one assertion a broken pairing cannot satisfy -- if pose k of "A" and pose k
    of "B" are not the same coordinates, the differences are nonzero.
    """
    res = paired_ddE(_self_pairs(), _spread_energy())
    assert res.n_pairs == 6
    assert res.differences == [0.0] * 6, "self-pairing produced a nonzero difference"
    assert res.ddE_paired == 0.0
    assert res.sd_paired == 0.0
    assert res.sem_paired == 0.0
    # And the UNPAIRED estimate over the same data is also zero in the MEAN --
    # the means are identical -- but carries the full pose scatter in its sd.
    assert res.ddE_unpaired == pytest.approx(0.0, abs=1e-9)
    assert res.sd_unpaired > 10.0, (
        "the unpaired sd should still see the pose scatter; if it does not, "
        "the test's energy function is not exercising the variance at all"
    )


def test_anchor_fails_when_the_pairing_is_broken():
    """MUTATION: shuffle B's poses so pose k of A no longer matches pose k of B.

    A test never seen to fail is an assumption. This is the deliberately broken
    version: the same POSES, the same ENERGIES, only the correspondence
    destroyed. The anchor must reject it.
    """
    pairs = _self_pairs()
    rolled = [p.coords_b for p in pairs]
    rolled = rolled[1:] + rolled[:1]
    for p, c in zip(pairs, rolled):
        p.coords_b = c
    res = paired_ddE(pairs, _spread_energy())
    assert res.sd_paired > 1.0, (
        "a broken correspondence still gave a tight paired sd -- the anchor "
        "cannot detect the very error it exists to catch"
    )


def test_pairing_reduces_variance_when_the_noise_is_common():
    """The mechanism: a shared pose term cancels, a substituent term does not."""
    import random

    rng = random.Random(11)
    pairs = []
    for i in range(40):
        shared = rng.gauss(0, 30.0)  # pose-conformational, COMMON
        pairs.append(
            PairedPose(
                i, ["C"], [(shared, 0.0, 0.0)], ["N"], [(shared, 0.0, 0.0)], [(0, 0)]
            )
        )

    def e(symbols, coords):
        # The shared coordinate dominates; the element shifts by exactly 2.
        return coords[0][0] + (2.0 if symbols[0] == "N" else 0.0)

    res = paired_ddE(pairs, e)
    assert res.ddE_paired == pytest.approx(2.0, abs=1e-9)
    assert res.sd_paired == pytest.approx(0.0, abs=1e-9)
    assert res.sd_unpaired > 20.0
    assert res.rho > 0.99, "perfectly common noise must show as rho ~ 1"
    assert res.variance_reduction > 100.0


def test_pairing_buys_nothing_when_the_noise_is_independent():
    """The NEGATIVE control, and it needs the same bar as the positive one.

    If the substitution scrambles the pose, the noise is not common and pairing
    cannot help. A module that reported a win here would be fabricating one.
    """
    import random

    rng = random.Random(13)
    pairs = [
        PairedPose(
            i,
            ["C"],
            [(rng.gauss(0, 30.0), 0.0, 0.0)],
            ["N"],
            [(rng.gauss(0, 30.0), 0.0, 0.0)],
            [(0, 0)],
        )
        for i in range(60)
    ]

    def e(symbols, coords):
        return coords[0][0]

    res = paired_ddE(pairs, e)
    assert abs(res.rho) < 0.4, "independent draws should show no correlation"
    # sd_paired ~ sd*sqrt(2) ~ sd_unpaired: the pairing is a relabeling.
    assert res.variance_reduction == pytest.approx(1.0, rel=0.35)


def test_paired_and_unpaired_estimates_are_ALGEBRAICALLY_EQUAL():
    """Records why an earlier guard was INERT, so it is not reintroduced.

    `ddE_paired = mean(E_B - E_A)` and `ddE_unpaired = mean(E_B) - mean(E_A)`
    are the same number -- the mean of a difference IS the difference of means.
    A guard comparing them can never fire. Pairing changes the estimator's
    VARIANCE, never its value, which is the whole point of the technique.
    """
    import random

    rng = random.Random(3)
    pairs = [
        PairedPose(
            i,
            ["C"],
            [(rng.gauss(0, 20), 0, 0)],
            ["N"],
            [(rng.gauss(0, 20), 0, 0)],
            [(0, 0)],
        )
        for i in range(30)
    ]
    res = paired_ddE(pairs, lambda s, c: c[0][0] + (2.0 if s[0] == "N" else 0.0))
    assert res.ddE_paired == pytest.approx(res.ddE_unpaired, abs=1e-9)
    assert res.sd_paired != pytest.approx(res.sd_unpaired, rel=1e-6), (
        "the SDs must differ even though the means cannot -- otherwise this "
        "test proves nothing about pairing"
    )


def test_self_anchor_bias_is_flagged():
    """THE REAL GUARD: pairing the parent with itself must give ddE == 0.

    MEASURED on paracetamol-like/MMFF: it gives +13.8 kcal/mol, because the
    construction re-embeds the B side under a hard scaffold constraint and a
    constrained re-embedding sits above the relaxed geometry it came from.
    """
    pairs = _self_pairs(8)
    res = paired_ddE(pairs, _spread_energy(), self_anchor_ddE=13.841)
    assert res.mean_shift_is_suspicious
    assert res.reembedding_bias == pytest.approx(0.0 - 13.841)
    assert any("SELF-ANCHOR IS NONZERO" in n for n in res.notes)


def test_a_clean_self_anchor_does_not_trip_the_guard():
    """MUTATION of the above: a zero anchor must NOT be flagged."""
    res = paired_ddE(_self_pairs(8), _spread_energy(), self_anchor_ddE=0.0)
    assert not res.mean_shift_is_suspicious
    assert res.reembedding_bias == pytest.approx(0.0)


def test_omitting_the_self_anchor_says_so_rather_than_passing_silently():
    """An unmeasured bias must read as UNMEASURED, never as absent."""
    res = paired_ddE(_self_pairs(8), _spread_energy())
    assert res.reembedding_bias is None
    assert not res.mean_shift_is_suspicious
    assert any("UNMEASURED" in n for n in res.notes), (
        "a missing anchor produced no warning, so a caller who never measured "
        "the bias would read the result as unbiased"
    )


def test_fewer_than_two_pairs_reports_nothing_rather_than_zero_width():
    res = paired_ddE(
        [PairedPose(0, ["C"], [(0.0, 0, 0)], ["N"], [(0.0, 0, 0)], [(0, 0)])],
        lambda s, c: 1.0,
    )
    assert math.isnan(res.sd_paired)
    assert any("at least 2" in n for n in res.notes)


def test_unscorable_pairs_are_dropped_symmetrically_and_reported():
    """Dropping ONE side would unbalance the means silently."""
    pairs = _self_pairs(5)

    calls = {"n": 0}

    def e(symbols, coords):
        calls["n"] += 1
        return None if calls["n"] == 3 else coords[0][0]

    res = paired_ddE(pairs, e)
    assert res.n_pairs == 4, "a half-scored pair must be dropped whole"
    assert any("dropped" in n for n in res.notes)


def test_failed_poses_are_reported_not_silently_shortened():
    pairs = _self_pairs(4)
    pairs[1].error = "constrained embedding failed"
    res = paired_ddE(pairs, lambda s, c: c[0][0])
    assert res.n_pairs == 3
    assert any("could not be paired" in n for n in res.notes)


def test_relax_substituent_leaves_unusable_poses_alone():
    """A pose that cannot be typed must survive, not vanish from the ensemble."""
    from tools.morph.paired import relax_substituent

    pairs = _self_pairs(3)
    pairs[1].error = "constrained embedding failed"
    out = relax_substituent(pairs)
    assert len(out) == 3, "relaxation changed the ensemble SIZE, biasing the mean"
    assert out[1].error == "constrained embedding failed"


def test_pairing_defaults_to_relaxed_because_hard_fails_the_anchor():
    """The default must be the arm that PASSES the self-anchor.

    Hard-pinning charges +13.841 kcal/mol for pairing the parent with itself
    (wiki/paired-ddE-substitution-2026-09-19.md). Defaulting to it would ship
    a construction known to be biased.
    """
    import inspect

    from tools.morph.paired import pair_poses_by_scaffold

    sig = inspect.signature(pair_poses_by_scaffold)
    assert sig.parameters["relax"].default is True
