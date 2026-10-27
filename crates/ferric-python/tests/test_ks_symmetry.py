"""run_dft / run_ksdft(symmetry=True): point-group-blocked RKS through Python (#294).

Reference: PySCF dft.RKS PBE with mol.symmetry = True on ferric's KS recipe
(75x110 grid, Becke partition, RI-J with def2-universal-jkfit), the `rks_pbe`
blocks of testdata/reference/validation/scf_symmetry/*.json
(scripts/validation/gen_scf_symmetry.py).
"""

import json
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "testdata" / "reference" / "validation" / "scf_symmetry"
TIGHT = dict(density_conv=1e-10, energy_conv=1e-8, max_iter=200)


def _water():
    return ferric.Molecule.from_xyz(str(ROOT / "testdata" / "molecules" / "water.xyz"))


def _h2(tmp_path, r):
    path = tmp_path / "h2.xyz"
    path.write_text(f"2\nh2\nH 0 0 {r / 2}\nH 0 0 {-r / 2}\n")
    return ferric.Molecule.from_xyz(str(path))


def _ref(name):
    return json.loads((REF / name).read_text())["rks_pbe"]


@pytest.mark.parametrize("entry", ["run_dft", "run_ksdft"])
def test_blocked_rks_matches_pyscf(entry):
    ref = _ref("h2o_6-31g.json")["ground"]
    run = getattr(ferric, entry)
    r = run(_water(), ferric.BasisSet.bundled("6-31g"), "PBE", symmetry=True, **TIGHT)
    assert isinstance(r, ferric.SymmetricRhfResult)
    assert r.converged
    assert r.point_group == ref["group"] == "C2v"
    assert abs(r.energy - ref["energy"]) < 1e-8
    mine = {k: v // 2 for k, v in r.irrep_occupations().items() if v}
    want = {k: v for k, v in ref["docc_per_irrep"].items() if v}
    assert mine == want


def test_symmetry_false_is_bit_identical_to_the_default():
    mol, bs = _water(), ferric.BasisSet.bundled("sto-3g")
    a = ferric.run_dft(mol, bs, "PBE")
    b = ferric.run_dft(mol, bs, "PBE", symmetry=False)
    assert type(a) is type(b) is ferric.DftResult
    assert a.total_energy == b.total_energy


def test_stretched_h2_reaches_sigma_g_squared_only_with_irrep_nelec(tmp_path):
    mol, bs = _h2(tmp_path, 10000.0), ferric.BasisSet.bundled("cc-pvdz")
    ref = _ref("h2_r10000_cc-pvdz.json")
    r = ferric.run_dft(mol, bs, "PBE", symmetry=True, irrep_nelec={"Ag": 2}, **TIGHT)
    assert abs(r.energy - ref["sigma_g2"]["energy"]) < 1e-9
    assert r.orbital_irreps[0] == "Ag"
    # Without irrep_nelec the sigma_g / sigma_u shell is partially filled and
    # degenerate: refused (the grid could pick the occupation), not guessed.
    with pytest.raises(ValueError, match=r"symmetry=True.*degenerate shell"):
        ferric.run_dft(mol, bs, "PBE", symmetry=True)


def test_refusals_are_valueerrors():
    mol, bs = _water(), ferric.BasisSet.bundled("sto-3g")
    with pytest.raises(ValueError, match="requires symmetry=True"):
        ferric.run_dft(mol, bs, "PBE", irrep_nelec={"A1": 6})
    with pytest.raises(ValueError, match=r"symmetry=True.*with_gradient"):
        ferric.run_dft(mol, bs, "PBE", symmetry=True, with_gradient=True)
    with pytest.raises(ValueError, match=r"symmetry=True.*dispersion"):
        ferric.run_dft(mol, bs, "PBE", symmetry=True, dispersion="d3bj")
    for kwargs in (
        {"irrep_nelec": {"Z9": 2}},
        {"k_builder": "cosx"},
        {"point_charges": [(0.1, 5.0, 0.0, 0.0)]},
    ):
        with pytest.raises(ValueError, match=r"run_dft\(symmetry=True\)"):
            ferric.run_dft(mol, bs, "PBE", symmetry=True, **kwargs)
