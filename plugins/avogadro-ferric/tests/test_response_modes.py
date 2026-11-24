"""Response Modes: what Avogadro would draw equals ferric's mode densities.

``orbitals.avogadro_mo_values`` ports Avogadro 2.1.0's evaluation of a cjson
basisSet + MO coefficients (primitive normalisation applied by Avogadro). The
exported auxiliary basis + mode columns, evaluated that way, must equal
ferric's own evaluation sum_P c_P chi_P(r) / ||rho||_2 at points the per-shell
fit never saw, and the values in ferric's cube file for the same mode.

Bars: the per-shell map is accepted at relative residual 1e-9
(``orbitals.FIT_TOLERANCE``) and is exact for these shells (signed
permutations and diagonal scalings), so the evaluation agrees to round-off;
1e-10 relative to the largest value, as for MOs in test_orbitals.py. The cube
prints 5 significant digits: 1e-5 relative there.

MUTATION (run by hand): dropping the 1/l2_norms scaling in
``response_modes.modes_cjson`` fails the first two tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from avoferric_testlib import WATER, make_cjson, menu_input, run_plugin

import ferric

pytestmark = pytest.mark.skipif(
    not hasattr(ferric, "response_modes") or not hasattr(ferric.BasisSet, "shells"),
    reason="needs ferric.response_modes and BasisSet.shells",
)

XYZ = "3\n\n" + "".join(f"{s} {a[1]} {a[2]} {a[3]}\n" for s, a in zip("OHH", WATER))


@pytest.fixture(scope="module")
def water_modes():
    mol = ferric.Molecule.from_xyz_string(XYZ)
    bs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    res = ferric.run_rhf(mol, bs)
    modes = ferric.response_modes(mol, bs, aux, res, n_modes=6)
    return mol, aux, modes


def test_avogadro_evaluation_of_exported_modes_matches_ferric(water_modes):
    from avogadro_ferric import orbitals, response_modes

    mol, aux, modes = water_modes
    data = response_modes.modes_cjson(ferric, mol, aux, modes, [8, 1, 1])
    cj = make_cjson(WATER)
    cj.update(data)
    rng = np.random.default_rng(11)
    pts = rng.normal(size=(400, 3)) * 1.6  # Bohr
    ours = orbitals.avogadro_mo_values(cj, pts)
    c = np.asarray(modes.coefficients) / np.asarray(modes.l2_norms)
    ref = np.asarray(ferric.eval_ao_on_points(mol, aux, pts)) @ c
    assert ours.shape == (400, 6)
    assert np.abs(ours - ref).max() < 1e-10 * np.abs(ref).max()
    orb = data["orbitals"]
    assert orb["energies"] == pytest.approx(list(modes.eigenvalues), rel=1e-15)
    assert orb["occupations"] == [0] * 6 and orb["electronCount"] == 0
    assert len(orb["moCoefficients"]) == 6 * modes.naux


def test_cube_and_cjson_hold_the_same_mode(water_modes, tmp_path):
    from avogadro_ferric import orbitals, response_modes

    mol, aux, modes = water_modes
    cj = make_cjson(WATER)
    cj.update(response_modes.modes_cjson(ferric, mol, aux, modes, [8, 1, 1]))
    path = tmp_path / "mode1.cube"
    ferric.write_response_mode_cube(
        str(path), mol, aux, modes, 1, spacing=0.5, padding=3.0
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = abs(int(lines[2].split()[0]))
    origin = np.array(lines[2].split()[1:4], dtype=float)
    axes = [lines[3 + i].split() for i in range(3)]
    shape = [int(a[0]) for a in axes]
    steps = np.array([[float(x) for x in a[1:]] for a in axes])
    vals = np.array(" ".join(lines[6 + natoms :]).split(), dtype=float)
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    pts = origin + np.stack([ii, jj, kk], axis=-1).reshape(-1, 3) @ steps
    avo = orbitals.avogadro_mo_values(cj, pts)[:, 1]
    assert np.abs(vals - avo).max() < 1e-5 * np.abs(avo).max()


def test_response_modes_menu_command(plugin_env):
    env = plugin_env(method="HF", basis="sto-3g")
    payload = menu_input(WATER, options={"n_modes": 4, "auxbasis": 0, "omega": 0.0})
    out, _ = run_plugin(env, ["response_modes"], payload)
    assert "error" not in out, out
    orb = out["cjson"]["orbitals"]
    nao = sum(
        {0: 1, 1: 3, 2: 6, -2: 5, 3: 10, -3: 7}[st]
        for st in out["cjson"]["basisSet"]["shellTypes"]
    )
    assert len(orb["energies"]) == 4
    assert len(orb["moCoefficients"]) == 4 * nao
    assert all(e >= 1.0 for e in orb["energies"])
    assert "dielectric eigenvalue" in out["message"]


def test_response_modes_is_a_static_options_menu_command():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    cmd = next(
        c
        for c in meta["tool"]["avogadro"]["menu-commands"]
        if c["identifier"] == "response_modes"
    )
    opts = tomllib.loads((root / cmd["user-options"]).read_text(encoding="utf-8"))
    assert set(opts) == {"n_modes", "auxbasis", "omega"}
    assert all("type" in v for v in opts.values())
