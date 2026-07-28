"""Shared fixtures and import guard for the ferric Python binding tests.

IMPORT MECHANISM. `ferric` is a compiled pyo3 extension. There is no pure-Python
fallback, so if the extension is not importable there is nothing to test. Rather
than letting every test in the suite blow up with an opaque ImportError, this
module turns a failed import into a single collection-time skip carrying the
exact command needed to fix it.

The extension reaches site-packages as a *symlink* to `target/release/libferric.so`
(see CLAUDE.md, "Python .so symlink -> target/release"), NOT as a maturin-copied
artifact. The practical consequence is that `cargo build --release -p ferric-python`
is sufficient to pick up binding changes -- no reinstall step -- but equally that
a stale or absent `target/release/libferric.so` silently serves old bindings.
"""

import math

import pytest

_IMPORT_HINT = """\
ferric (the compiled pyo3 extension) is not importable.

Build it, then re-run:

    cargo build --release -p ferric-python

The extension is exposed to Python as a symlink from site-packages to
target/release/libferric.so, so a successful release build is normally all
that is required. If the symlink itself is missing, reinstall with:

    uv run maturin develop --release

Original error: {err}"""

try:
    import ferric
except Exception as _e:  # pragma: no cover - exercised only on a broken build
    ferric = None
    _IMPORT_ERROR = _IMPORT_HINT.format(err=_e)
else:
    _IMPORT_ERROR = None


def pytest_ignore_collect(collection_path, config):
    """Skip the whole suite with one informative message if the build is missing.

    This must be `pytest_ignore_collect` rather than the more obvious
    `pytest_collection_modifyitems`: the test modules do `import ferric` at
    module scope, so a missing extension raises during COLLECTION, before any
    item exists to attach a skip marker to. Handled at the wrong hook, the
    suite reports a collection ERROR (exit code 2, no message about how to fix
    it) instead of a clean skip.
    """
    if _IMPORT_ERROR is None:
        return None
    if collection_path.suffix == ".py" and collection_path.name.startswith("test_"):
        print(f"\nSKIPPING {collection_path.name}:\n{_IMPORT_ERROR}\n")
        return True
    return None


# ── Physical constants, duplicated deliberately ──
#
# These mirror ferric's own values. They are written out as literals rather
# than read back from the module under test on purpose: a test that sources its
# expected value from the code it is testing cannot detect a change to that
# value. Test 'constants_match_module' asserts the two agree, which is the
# check that actually has teeth.

# ferric_core::mol's Angstrom->Bohr factor is 1/0.529_177_210_92.
BOHR_PER_ANGSTROM = 1.0 / 0.529_177_210_92

# ferric_core::conformers::BOLTZMANN_HARTREE_PER_K
KB_HARTREE_PER_K = 3.166_811_563_455_608e-6

# ferric_core::conformers::DEFAULT_TEMPERATURE_K (thermochemical standard)
DEFAULT_T = 298.15

KT_AT_DEFAULT_T = KB_HARTREE_PER_K * DEFAULT_T


# Water, in Angstrom -- the geometry used throughout. Small enough that every
# SCF in this suite is sub-second at STO-3G.
WATER_ANGSTROM = [
    [0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.9578],
    [0.9266, 0.0000, -0.2400],
]
WATER_SYMBOLS = ["O", "H", "H"]


def water_xyz_string():
    """The same water geometry as an XYZ string, for parse_xyz cross-checks.

    Written with the identical decimal literals as WATER_ANGSTROM so that any
    difference between the two paths is attributable to the code, not to the
    input.
    """
    rows = "\n".join(
        f"{s} {r[0]!r} {r[1]!r} {r[2]!r}"
        for s, r in zip(WATER_SYMBOLS, WATER_ANGSTROM)
    )
    return f"{len(WATER_SYMBOLS)}\n\n{rows}\n"


@pytest.fixture
def water_ensemble():
    """A one-conformer water ensemble."""
    return ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM], WATER_SYMBOLS
    )


@pytest.fixture
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


def nuclear_repulsion_from_bohr(coords_bohr, charges):
    """Independent NRE from explicit Bohr coordinates.

    Deliberately reimplemented in Python: comparing ferric's NRE against
    ferric's own coordinates via ferric's own formula would be circular. This
    is the outside check that the stored numbers really are Bohr.
    """
    total = 0.0
    n = len(charges)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.dist(coords_bohr[i], coords_bohr[j])
            total += charges[i] * charges[j] / d
    return total
