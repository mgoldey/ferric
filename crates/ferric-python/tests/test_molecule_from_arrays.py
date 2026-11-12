"""`Molecule.from_arrays` equals the XYZ parser's molecule field for field.

The anchor is the independent construction `from_xyz_string` of the same
atoms written with `repr()` coordinates (shortest round-trip decimal), which
parses back to the same doubles and multiplies by the same Angstrom->Bohr
constant, so every field must agree exactly (float `==`), not approximately.
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

SYMBOLS = ["O", "H", "H", "Cl"]
COORDS = [
    (0.0123456789012, -0.1, 0.117),
    (0.0, 0.7571234567, -0.467),
    (1e-7, -0.757, -0.4671),
    (2.5, 1.25, -3.0000000001),
]
BOHR_PER_ANGSTROM = 1.0 / 0.52917721092


def _xyz(symbols, coords, ghost=None):
    lines = [str(len(symbols)), "from_arrays anchor"]
    for k, (s, (x, y, z)) in enumerate(zip(symbols, coords)):
        mark = "@" if ghost and ghost[k] else ""
        lines.append(f"{mark}{s} {float(x)!r} {float(y)!r} {float(z)!r}")
    return "\n".join(lines) + "\n"


def _fields(m):
    return (
        m.symbols(),
        m.atomic_numbers(),
        m.coords_bohr(),
        m.is_ghost(),
        m.nelec(),
        m.nuclear_repulsion(),
    )


@pytest.mark.parametrize("charge, mult", [(0, 2), (1, 1), (-1, 1)])
def test_from_arrays_equals_from_xyz_string_exactly(charge, mult):
    ref = ferric.Molecule.from_xyz_string(
        _xyz(SYMBOLS, COORDS), charge=charge, multiplicity=mult
    )
    for elements in (SYMBOLS, [8, 1, 1, 17], np.array([8, 1, 1, 17])):
        for coords in (COORDS, np.asarray(COORDS)):
            got = ferric.Molecule.from_arrays(
                elements, coords, charge=charge, multiplicity=mult
            )
            assert _fields(got) == _fields(ref)
            assert got.to_xyz_string() == ref.to_xyz_string()


def test_bohr_units_store_the_coordinates_unchanged():
    bohr = [(x * BOHR_PER_ANGSTROM, y, z * 2.0) for x, y, z in COORDS]
    m = ferric.Molecule.from_arrays(SYMBOLS, bohr, units="bohr", multiplicity=2)
    assert m.coords_bohr() == [tuple(r) for r in bohr]
    upper = ferric.Molecule.from_arrays(SYMBOLS, bohr, units="Bohr", multiplicity=2)
    assert upper.coords_bohr() == m.coords_bohr()


def test_ghost_flags_match_the_xyz_marker():
    ghost = [False, False, False, True]
    ref = ferric.Molecule.from_xyz_string(_xyz(SYMBOLS, COORDS, ghost))
    got = ferric.Molecule.from_arrays(SYMBOLS, COORDS, ghost=ghost)
    assert _fields(got) == _fields(ref)
    assert got.is_ghost() == ghost


def test_an_energy_through_from_arrays_is_the_xyz_energy():
    bs = ferric.BasisSet.bundled("sto-3g")
    xyz = ferric.Molecule.from_xyz_string(_xyz(SYMBOLS[:3], COORDS[:3]))
    arr = ferric.Molecule.from_arrays(SYMBOLS[:3], COORDS[:3])
    assert ferric.run_rhf(arr, bs).energy == ferric.run_rhf(xyz, bs).energy


@pytest.mark.parametrize(
    "args, kw, match",
    [
        ((SYMBOLS, COORDS), {"units": "nm"}, "units"),
        ((SYMBOLS[:3], COORDS), {}, "coordinate rows"),
        ((SYMBOLS, [c[:2] for c in COORDS]), {}, "expected 3"),
        ((SYMBOLS, COORDS), {"ghost": [True]}, "ghost flags"),
        ((["O", "H", "H", "Xx"], COORDS), {}, "unknown element"),
        (([8, 1, 1, 999], COORDS), {}, "unknown atomic number"),
        ((["O", "H", "H", "@Cl"], COORDS), {}, "ghost="),
        ((SYMBOLS, [*COORDS[:3], (np.nan, 0.0, 0.0)]), {}, "not finite"),
        ((SYMBOLS, COORDS), {"multiplicity": 1}, "multiplicity"),
        (([None, 1, 1, 17], COORDS), {}, "neither a symbol"),
    ],
    ids=[
        "unit",
        "length",
        "row",
        "ghost-len",
        "symbol",
        "z",
        "at-prefix",
        "nan",
        "parity",
        "type",
    ],
)
def test_from_arrays_refusals(args, kw, match):
    with pytest.raises(ValueError, match=match):
        ferric.Molecule.from_arrays(*args, **kw)
