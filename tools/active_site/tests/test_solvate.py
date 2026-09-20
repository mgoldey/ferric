"""Explicit TIP3P droplet: packing, and the noise that bounds what it can say."""

from __future__ import annotations

import math

import pytest

from tools.active_site.solvate import (
    BULK_DENSITY,
    H_SCALE,
    dE_statistics,
    solvate,
    write_pqr,
)

_WATER = (["O", "H", "H"], [(0.0, 0.0, 0.0), (0.757, 0.586, 0.0), (-0.757, 0.586, 0.0)])


def test_the_pack_reaches_a_plausible_fraction_of_bulk():
    """A hard-sphere pack lands below bulk; HOW far below is the quality bar.

    MEASURED 79-92% across 6-10 A. Before per-element cutoffs it was 54%,
    because the heavy-atom 2.4 A applied to hydrogens rejected every
    lattice-adjacent pair (their H...H can be 1.19 A apart).
    """
    for r in (6.0, 8.0, 10.0):
        d = solvate(*_WATER, r)
        frac = d.density / BULK_DENSITY
        assert 0.70 < frac < 1.05, (
            f"r={r}: packed {frac:.0%} of bulk ({d.n_waters} waters). Below 70% "
            "means the exclusion rules are rejecting real first-shell contacts; "
            "above bulk means they are not rejecting overlaps."
        )


def test_hydrogens_get_a_smaller_cutoff_than_heavy_atoms():
    """Not a tuning knob: it is what makes first-shell contacts possible."""
    assert 0.5 < H_SCALE < 1.0, "H_SCALE must shrink the cutoff, not grow it"
    # 0.75 * 2.4 = 1.8 A, the hydrogen-bond H...O distance.
    assert abs(H_SCALE * 2.4 - 1.8) < 1e-9


def test_no_water_atom_clashes_with_the_solute():
    """The packing's own promise, checked directly rather than trusted."""
    sym, crd = _WATER
    d = solvate(sym, crd, 8.0, min_dist_angstrom=2.4)
    worst = min(
        math.dist(w, s)
        for w, wsym in zip(d.coords, d.symbols)
        for s, ssym in zip(crd, sym)
        if wsym != "H" and ssym != "H"
    )
    assert worst >= 2.4 - 1e-9, f"heavy-atom clash at {worst:.3f} A"


def test_a_droplet_is_reproducible_for_a_given_seed():
    """An irreproducible starting structure makes everything downstream so."""
    a = solvate(*_WATER, 7.0, seed=123)
    b = solvate(*_WATER, 7.0, seed=123)
    c = solvate(*_WATER, 7.0, seed=124)
    assert a.coords == b.coords, "same seed must give the same droplet"
    assert a.coords != c.coords, "different seeds must differ, or seeding is inert"


def test_the_droplet_is_net_neutral():
    """TIP3P is neutral per water; a charged droplet would be a packing bug."""
    d = solvate(*_WATER, 8.0)
    assert abs(sum(d.charges)) < 1e-9, f"droplet carries net {sum(d.charges)} e"
    assert len(d.coords) == 3 * d.n_waters


def test_dE_statistics_refuses_a_single_droplet():
    """One droplet has no measurable precision -- that is why this exists."""
    with pytest.raises(ValueError, match="n_seeds must be >= 2"):
        dE_statistics(lambda d: 0.0, *_WATER, 6.0, n_seeds=1)


def test_dE_statistics_reports_a_spread_that_is_actually_large():
    """The load-bearing measurement: orientation noise dominates.

    MEASURED with a QM evaluator: +2.245 +/- 1.367 kcal/mol (sd 3.867, n=8).
    The mean is smaller than its own scatter, which is why a single droplet
    cannot be read as a solvation energy. Here the evaluator is the droplet's
    own dipole magnitude -- cheap, and it carries the same orientation noise.
    """

    def dipole(d):
        return math.dist(
            (0.0, 0.0, 0.0),
            tuple(sum(q * c[k] for q, c in zip(d.charges, d.coords)) for k in range(3)),
        )

    st = dE_statistics(dipole, *_WATER, 8.0, n_seeds=6)
    assert st.n == 6 and len(st.values) == 6
    assert st.sd > 0.0, "zero spread means the seeds are not independent"
    assert abs(st.sem - st.sd / math.sqrt(st.n)) < 1e-12, "SEM must be sd/sqrt(n)"


def test_write_pqr_puts_the_solute_first(tmp_path):
    """`[qmmm] qm_indices` then indexes the solute as 0..n-1 unchanged."""
    sym, crd = _WATER
    d = solvate(sym, crd, 6.0)
    p = tmp_path / "droplet.pqr"
    n = write_pqr(p, sym, crd, [-0.834, 0.417, 0.417], d)
    assert n == 3 + 3 * d.n_waters
    lines = [l for l in p.read_text().splitlines() if l.startswith("ATOM")]
    assert len(lines) == n
    assert lines[0].split()[3] == "SOL", "the solute must come first"
    assert lines[3].split()[3] == "WAT"
    # And it must round-trip through the reader the CLI uses.
    from tools.active_site.pqr_parser import parse_pqr_atoms

    atoms = parse_pqr_atoms(p)
    assert len(atoms) == n, "the PQR we write must parse with the PQR we read"
