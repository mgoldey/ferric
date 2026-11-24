"""`ferric.viz`: py3Dmol helpers and the cube reader behind them.

The drawing functions need py3Dmol and skip without it; everything else
(`molecule_to_xyz`, `read_cube`, `cube_surfaces`, `diverging_colors`) is
tested with numpy alone.

Anchors:
  * a molecule written by `molecule_to_xyz` and parsed by ferric is the same
    molecule (12 decimals in Angstrom is below 1e-11 Bohr);
  * `read_cube` places value [i, j, k] at origin + i a + j b + k c, checked
    against `eval_density_on_points` at those points. The grid is
    asymmetric (distinct n_x, n_y, n_z and an asymmetric molecule), so a
    transposed reshape cannot pass. A cube value has 6 significant digits
    (E13.5), a relative error of at most 5e-6, so the bar is rtol 1e-5;
  * a density is drawn as one surface and an orbital (values below
    -isovalue) as two.
"""

import sys

import numpy as np
import pytest

import ferric

try:
    from ferric import viz
except ImportError as err:  # the extension without the python package layer
    pytest.skip(f"ferric.viz is not importable: {err}", allow_module_level=True)

WATER_XYZ = (
    "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.100 -0.757 -0.469\n"
)


@pytest.fixture(scope="module")
def water_scf():
    mol = ferric.Molecule.from_xyz_string(WATER_XYZ, 0, 1)
    bs = ferric.BasisSet.bundled("sto-3g")
    return mol, bs, ferric.run_rhf(mol, bs)


@pytest.fixture(scope="module")
def cubes(water_scf, tmp_path_factory):
    mol, bs, rhf = water_scf
    d = tmp_path_factory.mktemp("cubes")
    grid = {}
    for name, kw in (("density", {}), ("homo", {"what": "mo", "mo_index": 4})):
        origin, shape = ferric.write_cube(
            str(d / f"{name}.cube"), mol, bs, rhf, spacing=0.5, **kw
        )
        grid[name] = (np.array(origin), tuple(shape))
    return d, grid


def test_molecule_to_xyz_round_trip(water_scf):
    mol = water_scf[0]
    text = viz.molecule_to_xyz(mol)
    back = ferric.Molecule.from_xyz_string(text, 0, 1)
    assert back.symbols() == mol.symbols()
    assert (
        np.abs(np.array(back.coords_bohr()) - np.array(mol.coords_bohr())).max() < 1e-11
    )


def test_read_cube_places_values_on_the_grid(water_scf, cubes):
    mol, bs, rhf = water_scf
    d, grid = cubes
    cube = viz.read_cube(d / "density.cube")
    origin, shape = grid["density"]
    assert cube["values"].shape == shape
    assert len(set(shape)) == 3  # distinct n_x, n_y, n_z
    np.testing.assert_allclose(cube["origin"], origin)
    np.testing.assert_allclose(cube["axes"], np.diag([0.5, 0.5, 0.5]))
    assert list(cube["numbers"]) == [8, 1, 1]
    np.testing.assert_allclose(cube["coords"], mol.coords_bohr(), atol=1e-5)

    rng = np.random.default_rng(7)
    idx = np.column_stack([rng.integers(0, n, 60) for n in shape])
    # Add the grid point nearest the oxygen, where the density is large.
    idx = np.vstack(
        [idx, np.rint((np.array(mol.coords_bohr()[0]) - origin) / 0.5).astype(int)]
    )
    points = origin + idx @ cube["axes"]
    expected = ferric.eval_density_on_points(mol, bs, rhf, points)
    got = cube["values"][idx[:, 0], idx[:, 1], idx[:, 2]]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-12)


def test_read_cube_rejects_a_truncated_grid(cubes, tmp_path):
    d, _ = cubes
    lines = (d / "density.cube").read_text().splitlines()
    (tmp_path / "short.cube").write_text("\n".join(lines[:-3]) + "\n")
    with pytest.raises(ValueError, match="values for a"):
        viz.read_cube(tmp_path / "short.cube")


def test_cube_surfaces_density_is_one_lobe_and_an_orbital_two(cubes):
    d, _ = cubes
    density = viz.read_cube(d / "density.cube")["values"]
    homo = viz.read_cube(d / "homo.cube")["values"]
    assert density.min() >= 0
    assert homo.min() < -0.05 < 0.05 < homo.max()
    assert viz.cube_surfaces(density, 0.05) == [0.05]
    assert viz.cube_surfaces(homo, 0.05) == [0.05, -0.05]
    # An isovalue above every negative value leaves one lobe; `signed` overrides the detection.
    assert viz.cube_surfaces(homo, 10.0 * abs(homo.min())) == [10.0 * abs(homo.min())]
    assert viz.cube_surfaces(density, 0.05, signed=True) == [0.05, -0.05]
    assert viz.cube_surfaces(homo, 0.05, signed=False) == [0.05]
    with pytest.raises(ValueError, match="positive"):
        viz.cube_surfaces(homo, 0.0)


def test_diverging_colors_endpoints_and_order():
    assert viz.diverging_colors([0.0, -1.0, 1.0, -5.0, 5.0], vlim=1.0) == [
        "#ffffff",
        "#ff0000",
        "#0000ff",
        "#ff0000",
        "#0000ff",
    ]
    # Default scale is max |value|: the extremes saturate, half scale is half faded.
    cols = viz.diverging_colors([-2.0, 1.0, 2.0])
    assert cols[0] == "#ff0000" and cols[2] == "#0000ff"
    assert cols[1] == "#8080ff"
    with pytest.raises(ValueError, match="vlim"):
        viz.diverging_colors([1.0], vlim=0.0)


def test_views_need_py3dmol(water_scf, monkeypatch):
    monkeypatch.setitem(sys.modules, "py3Dmol", None)
    with pytest.raises(ImportError, match="pip install py3Dmol"):
        viz.show_molecule(water_scf[0])


def _html(view):
    return view._make_html()


def test_show_molecule_draws_every_atom(water_scf):
    pytest.importorskip("py3Dmol")
    html = _html(viz.show_molecule(water_scf[0]))
    assert html.count("addModel") == 1
    for sym in ("O ", "H "):
        assert sym in html
    with pytest.raises(ValueError, match="unknown style"):
        viz.show_molecule(water_scf[0], style="ribbon")


def test_show_cube_isosurface_count(water_scf, cubes):
    pytest.importorskip("py3Dmol")
    mol = water_scf[0]
    d, _ = cubes
    homo = _html(viz.show_cube(d / "homo.cube", mol=mol, isovalue=0.05))
    density = _html(viz.show_cube(d / "density.cube", mol=mol, isovalue=0.05))
    assert homo.count("addVolumetricData") == 2
    assert '"isoval": -0.05' in homo and '"isoval": 0.05' in homo
    assert density.count("addVolumetricData") == 1
    assert '"isoval": -0.05' not in density


def test_show_surface_values_one_sphere_per_point(water_scf):
    pytest.importorskip("py3Dmol")
    mol = water_scf[0]
    pts = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, -3.0], [3.0, 0.0, 0.0]])
    html = _html(viz.show_surface_values(mol, pts, [-1.0, 0.0, 1.0]))
    assert html.count("addSphere") == 3
    assert "#ff0000" in html and "#ffffff" in html and "#0000ff" in html
    with pytest.raises(ValueError, match="points must be"):
        viz.show_surface_values(mol, pts, [1.0, 2.0])
