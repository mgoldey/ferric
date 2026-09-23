"""`guess=` and `diis=` are strict: an unknown value raises, it never defaults.

`guess=` used to accept any string -- everything except "hcore" silently ran
the MINAO guess -- and `diis=` had its own copy of the CLI's parser. Both now go
through the shared `ferric_scf` parsers the CLI's `[scf] guess` / `[scf] diis`
use, so the two surfaces accept exactly the same spellings.

The bad-value calls raise while resolving the config, before any SCF runs.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")


def _h2():
    return ferric.Molecule.from_xyz_string("2\nh2\nH 0 0 0\nH 0 0 0.74\n", 0, 1)


@pytest.mark.parametrize("bad", ["hcroe", "core", "sad-smallbasis"])
def test_unknown_guess_raises_and_lists_the_valid_values(bad):
    with pytest.raises(ValueError) as exc:
        ferric.run_rhf(_h2(), ferric.BasisSet.bundled("sto-3g"), guess=bad)
    msg = str(exc.value)
    for valid in ("'minao'", "'sad'", "'hcore'"):
        assert valid in msg, msg


@pytest.mark.parametrize("bad", ["cdiis", "diis"])
def test_unknown_diis_raises_and_lists_the_valid_values(bad):
    with pytest.raises(ValueError) as exc:
        ferric.run_rhf(_h2(), ferric.BasisSet.bundled("sto-3g"), diis=bad)
    msg = str(exc.value)
    for valid in ("'pulay'", "'adiis'", "'ediis'"):
        assert valid in msg, msg


@pytest.mark.parametrize("guess", ["minao", "sad", "hcore"])
def test_valid_guess_values_run(guess):
    """Positive control: the rejection above is about the value, not the kwarg."""
    res = ferric.run_rhf(_h2(), ferric.BasisSet.bundled("sto-3g"), guess=guess)
    assert res.converged
