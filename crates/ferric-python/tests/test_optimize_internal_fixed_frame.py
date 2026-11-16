"""`coordinates="internal"` is refused when the energy is not rotation/translation invariant.

Fixed charges and a uniform field exert a net force and torque that a step in
bonds, angles and dihedrals cannot follow, so the search would not converge
(water in a 0.02 a.u. field: no convergence in 100 steps, 14 mHa high; Cartesian
converges in 22). The call raises ValueError before any SCF (the guard follows the basis lookup,
so the basis here is a real one). Gas phase and PCM
keep the invariance and are not refused. The library rule is
`crates/ferric-scf/tests/internal_coords_fixed_frame.rs`.
"""

from __future__ import annotations

import ferric
import pytest

WATER = ferric.Molecule.from_xyz_string(
    "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n", 0, 1
)
BASIS = "sto-3g"


@pytest.mark.parametrize(
    ("kwargs", "cause"),
    [
        ({"point_charges": [(0.4, 4.0, 0.0, 0.0)]}, "charges"),
        ({"external_field": (0.02, 0.0, 0.0)}, "field"),
    ],
)
@pytest.mark.parametrize("entry", ["run_optimize", "run_optimize_uhf"])
def test_internal_with_fixed_frame_is_refused(entry, kwargs, cause):
    with pytest.raises(ValueError, match="cannot optimize with") as info:
        getattr(ferric, entry)(WATER, BASIS, coordinates="internal", **kwargs)
    msg = str(info.value)
    assert cause in msg
    assert "cartesian" in msg


def test_internal_with_fixed_frame_is_refused_for_every_method():
    for method, extra in [
        ("rimp2", {"auxbasis": "def2-svp-rifit"}),
        ("oo-rimp2", {"auxbasis": "def2-svp-rifit"}),
    ]:
        with pytest.raises(ValueError, match="cannot optimize with"):
            ferric.run_optimize(
                WATER,
                BASIS,
                method=method,
                coordinates="internal",
                external_field=(0.02, 0.0, 0.0),
                **extra,
            )


def test_pcm_with_internal_is_not_refused_and_converges():
    r = ferric.run_optimize(
        WATER, BASIS, coordinates="internal", solvent="water", max_steps=60
    )
    assert r.converged


def test_cartesian_with_a_field_is_not_refused():
    r = ferric.run_optimize(
        WATER,
        "sto-3g",
        coordinates="cartesian",
        external_field=(0.02, 0.0, 0.0),
        max_steps=60,
    )
    assert r.converged
