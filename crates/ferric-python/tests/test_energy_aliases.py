"""`energy` / `total_energy` are both readable on every SCF result class.

Callers (e.g. the smeltery app) read ``raw.energy if hasattr(raw, "energy")
else raw.total_energy``; each SCF result carries both spellings, with the
same value as the original getter.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

XYZ = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692"
OH = "2\n\nO 0 0 0\nH 0 0 0.97"


def _water():
    return ferric.Molecule.from_xyz_string(XYZ, 0, 1), ferric.BasisSet.bundled("sto-3g")


_OH = lambda: ferric.Molecule.from_xyz_string(OH, 0, 2)  # noqa: E731

# Solvers are built inside the test, not at collection time, so collecting
# (or deselecting) these cases does no numerical work.
_SOLVERS = {
    "RhfResult": lambda mol, bs: ferric.run_rhf(mol, bs),
    "SymmetricRhfResult": lambda mol, bs: ferric.run_rhf(mol, bs, symmetry=True),
    "UhfResult": lambda mol, bs: ferric.run_uhf(_OH(), bs),
    "UhfResult(rohf)": lambda mol, bs: ferric.run_rohf(_OH(), bs),
    "DftResult": lambda mol, bs: ferric.run_dft(mol, bs, "LDA"),
    "UhfResult(uks)": lambda mol, bs: ferric.run_uks(_OH(), bs, "LDA"),
    "UhfResult(roks)": lambda mol, bs: ferric.run_roks(_OH(), bs, "LDA"),
}


@pytest.mark.parametrize("name", sorted(_SOLVERS))
def test_energy_and_total_energy_agree(name):
    result = _SOLVERS[name](*_water())
    assert result.energy == result.total_energy, name
    assert isinstance(result.energy, float)


def test_periodic_scf_results_have_both_spellings(tmp_path):
    import json

    from test_pbc_bindings import (
        H2_ATOMS,
        H_ATOM,
        OMEGA,
        PYSCF_STO3G_H,
        _cubic,
        _xyz,
    )

    p = tmp_path / "h.json"
    p.write_text(json.dumps(PYSCF_STO3G_H))
    basis = ferric.BasisSet.from_bse_json(str(p))
    h = ferric.Molecule.from_xyz_string(_xyz(H_ATOM), multiplicity=2)
    h2 = ferric.Molecule.from_xyz_string(_xyz(H2_ATOMS))
    lat = _cubic()
    kw = {"omega": OMEGA}
    results = {
        "GammaRhfResult": ferric.run_rhf_gamma(h2, lat, basis, **kw),
        "GammaRksResult": ferric.run_rks_gamma(h2, lat, basis, "LDA", **kw),
        "GammaOpenShellResult": ferric.run_uhf_gamma(h, lat, basis, **kw),
        "KpointRksResult": ferric.run_rks_kpts(h2, lat, basis, "LDA", (1, 1, 2), **kw),
        "KpointScfResult": ferric.run_rhf_kpts(h2, lat, basis, (1, 1, 2), **kw),
    }
    for name, r in results.items():
        assert type(r).__name__ == name
        assert r.energy == r.total_energy, name


def test_cdft_result_has_both_spellings():
    mol = ferric.Molecule.from_xyz_string("2\nLiH\nLi 0 0 0\nH 0 0 1.60\n", 0, 1)
    r = ferric.run_cdft(
        mol,
        ferric.BasisSet.bundled("sto-3g"),
        [ferric.CdftConstraint([0], 2.2, kind="charge")],
        lambda_tol=1e-5,
        max_outer=40,
    )
    assert type(r).__name__ == "CdftResult"
    assert r.energy == r.total_energy
    assert isinstance(r.energy, float)
