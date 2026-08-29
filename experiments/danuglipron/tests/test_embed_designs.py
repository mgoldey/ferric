"""Does every DESIGNED danuglipron analogue actually produce 3D geometry?

Campaign-specific counterpart to `tools/morph/tests/test_embed.py`, which tests
the generic embedder. This one asserts that this particular hypothesis set is
embeddable -- if an arm fails to embed, that arm silently has no data.

NOTE (RESULTS.md M7): embedding SUCCEEDING is necessary but far from sufficient.
Every conformer generated here misses the experimentally determined bound pose by
2.2-4.1 A, against a 2.0 A docking-success bar, so these geometries are usable
for strain but NOT for pocket fit. See `tools.campaign.align.pose_quality_gate`.
"""
from __future__ import annotations

from experiments.danuglipron.design import danuglipron_analogues
from tools.morph.embed import embed_analogue


def test_all_designed_analogues_embed():
    """The real check on the design set: every proposed structure must actually
    produce 3D geometry, or that arm of the campaign silently has no data.
    One conformer each keeps this affordable.
    """
    failures = []
    for a in danuglipron_analogues():
        e = embed_analogue(a, n_conformers=1)
        if not e.usable:
            failures.append((a.label, e.error))
    assert not failures, f"analogues that failed to embed: {failures}"
