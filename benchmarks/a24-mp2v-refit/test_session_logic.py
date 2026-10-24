"""Optimiser-logic tests for session.py with a FAKE pool (no ferric needed).

Exactness anchor: with an error model whose pooled RMSD has a known minimum in
b (and known seam dependence), `optimal_b` must return that minimum, report an
edge when the minimum lies outside the allowed range, and `adaptive_search`
must find the planted (r0, s) node. A mutated `optimal_b` (argmax instead of
argmin) must fail the same checks -- see `test_the_test_can_fail`.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session as S  # noqa: E402


class FakePool:
    """errors = a_sys * (ln b - ln b*(r0, s)) + c_sys * (u - u0)^2-ish, noiseless."""

    def __init__(
        self, r0_star=1.2, s_star=3.0, b_of=lambda r0: 11.0 * (r0 / 1.0) ** 2.31
    ):
        self.systems = {("fake", k): 0.0 for k in range(6)}
        self.r0_star, self.s_star, self.b_of = r0_star, s_star, b_of
        self.cache = type("C", (), {"save": lambda self: None})()
        self.calls = 0

    def errors_b(self, ids, r0, s, bs, cp, c=0.0089, vv=None):
        self.calls += 1
        sv = S._sval(s)
        off = 0.08 * (
            (math.log(r0 / self.r0_star)) ** 2 + 0.5 * math.log(sv / self.s_star) ** 2
        )
        bstar = self.b_of(r0)
        rows = []
        for k in range(len(ids)):
            slope = 0.3 + 0.1 * k
            rows.append(
                slope * (np.log(np.asarray(bs)) - math.log(bstar)) + off * (1 + 0.1 * k)
            )
        return np.array(rows)


def test_optimal_b_finds_the_known_minimum():
    p = FakePool()
    ids = sorted(p.systems)
    d = S.optimal_b(p, ids, 1.2, 3.0, True)  # at the planted seam the offset is 0
    assert d["edge"] is None
    assert abs(d["b"] - p.b_of(1.2)) / p.b_of(1.2) < 0.01, d


def test_optimal_b_reports_an_edge():
    p = FakePool(b_of=lambda r0: 5000.0)
    d = S.optimal_b(p, sorted(p.systems), 1.0, 2.0, True)
    assert d["edge"] == "high", d


def test_adaptive_search_finds_the_planted_node():
    p = FakePool(r0_star=1.2, s_star=3.0)
    rules = dict(S.RULES)
    rules["max_nodes"] = 80
    out = S.adaptive_search(
        p, sorted(p.systems), True, rules=rules, log=lambda *_: None
    )
    best = out["best"]
    assert abs(math.log(best["r0"] / 1.2)) < 0.08, best
    assert abs(math.log(S._sval(best["s"]) / 3.0)) < 0.15, best
    assert out["n_nodes"] <= rules["max_nodes"]


def test_hessian_is_positive_definite_at_a_minimum_and_flags_the_valley():
    p = FakePool(r0_star=1.2, s_star=3.0)
    ids = sorted(p.systems)
    best = S.node(p, ids, 1.2, 3.0, True)
    h = S.hessian(p, ids, best, True)
    ev = np.array(h["eigenvalues"])
    assert ev[0] > 0
    # the (ln r0, ln b) valley (b ~ r0^2.31) must show up as the softest eigenvector
    v = np.array(h["eigenvectors"][0])
    assert abs(v[2]) > 0.3 and abs(v[0]) > 0.2, v


def test_the_test_can_fail(monkeypatch):
    p = FakePool()
    orig = np.argmin
    monkeypatch.setattr(np, "argmin", np.argmax)
    try:
        d = S.optimal_b(p, sorted(p.systems), 1.2, 3.0, True)
    finally:
        monkeypatch.setattr(np, "argmin", orig)
    assert not (d["edge"] is None and abs(d["b"] - p.b_of(1.2)) / p.b_of(1.2) < 0.01)
