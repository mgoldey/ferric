"""`ferric.eval_{ao,density,mo}_on_points` and `ferric.write_cube`.

All lengths are Bohr. Anchors that do not share code with the binding:
  * the density integrates to N on a uniform grid (trapezoid on Gaussians is
    spectrally accurate), read back from the written cube file;
  * rho(r) == 2 * sum_occ psi_i(r)^2 (MO path vs density-matrix path);
  * rho(r) and the s-AO values match PySCF on the same points when PySCF is
    importable;
  * the cube header carries the atom positions in Bohr, so an Angstrom/Bohr
    mix-up in origin, spacing or atom lines fails the header round trip.
"""

import numpy as np
import pytest

import ferric

H2_XYZ = "2\nh2\nH 0 0 0\nH 0 0 0.74\n"
WATER_XYZ = (
    "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n"
)
SPACING = 0.25
PADDING = 6.0


def read_cube(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    natoms, *org = lines[2].split()
    natoms = int(natoms)
    grid = [lines[3 + k].split() for k in range(3)]
    n = [int(g[0]) for g in grid]
    steps = [[float(x) for x in g[1:4]] for g in grid]
    atoms = [lines[6 + k].split() for k in range(natoms)]
    vals = np.array(" ".join(lines[6 + natoms :]).split(), dtype=float)
    return {
        "natoms": natoms,
        "origin": np.array(org, dtype=float),
        "n": n,
        "steps": np.array(steps),
        "atoms": atoms,
        "data": vals.reshape(n),
    }


@pytest.fixture(scope="module")
def h2():
    mol = ferric.Molecule.from_xyz_string(H2_XYZ)
    bs = ferric.BasisSet.bundled("sto-3g")
    return mol, bs, ferric.run_rhf(mol, bs)


@pytest.fixture(scope="module")
def water():
    mol = ferric.Molecule.from_xyz_string(WATER_XYZ)
    bs = ferric.BasisSet.bundled("sto-3g")
    return mol, bs, ferric.run_rhf(mol, bs)


def test_density_cube_header_and_integral(h2, tmp_path):
    mol, bs, rhf = h2
    path = str(tmp_path / "rho.cube")
    origin, shape = ferric.write_cube(
        path, mol, bs, rhf, spacing=SPACING, padding=PADDING
    )
    cube = read_cube(path)
    xyz = np.array(mol.coords_bohr())
    assert cube["natoms"] == 2
    assert tuple(cube["n"]) == tuple(shape)
    np.testing.assert_allclose(cube["origin"], origin, atol=1e-5)
    np.testing.assert_allclose(cube["origin"], xyz.min(axis=0) - PADDING, atol=1e-5)
    np.testing.assert_allclose(cube["steps"], np.eye(3) * SPACING, atol=1e-6)
    got = np.array([[float(x) for x in a[2:5]] for a in cube["atoms"]])
    np.testing.assert_allclose(got, xyz, atol=1e-5)  # Bohr, not Angstrom
    assert [int(a[0]) for a in cube["atoms"]] == [1, 1]
    # Box edge is >= padding past the last atom.
    far = cube["origin"] + (np.array(cube["n"]) - 1) * SPACING
    assert np.all(far >= xyz.max(axis=0) + PADDING - 1e-9)
    # integral of rho = N electrons
    assert cube["data"].sum() * SPACING**3 == pytest.approx(mol.nelec(), abs=1e-4)


def test_cube_values_match_points_evaluator(h2, tmp_path):
    mol, bs, rhf = h2
    path = str(tmp_path / "rho.cube")
    ferric.write_cube(
        path, mol, bs, rhf, origin=(-1.0, -1.0, -1.0), shape=(4, 5, 6), spacing=0.7
    )
    cube = read_cube(path)
    ix, iy, iz = np.meshgrid(range(4), range(5), range(6), indexing="ij")
    pts = np.stack([ix, iy, iz], axis=-1).reshape(-1, 3) * 0.7 - 1.0
    rho = np.array(ferric.eval_density_on_points(mol, bs, rhf, pts))
    np.testing.assert_allclose(cube["data"].reshape(-1), rho, rtol=1e-4, atol=1e-8)


def test_mo_cube_normalised_and_matches_points(water, tmp_path):
    mol, bs, rhf = water
    path = str(tmp_path / "mo.cube")
    origin, shape = ferric.write_cube(
        path, mol, bs, rhf, what="mo", mo_index=4, spacing=SPACING, padding=PADDING
    )
    cube = read_cube(path)
    assert (cube["data"] ** 2).sum() * SPACING**3 == pytest.approx(1.0, abs=1e-4)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, np.array(shape), size=(20, 3))
    pts = np.array(origin) + idx * SPACING
    psi = ferric.eval_mo_on_points(mol, bs, rhf, [4], pts)[:, 0]
    ref = cube["data"][idx[:, 0], idx[:, 1], idx[:, 2]]
    np.testing.assert_allclose(psi, ref, rtol=1e-4, atol=1e-7)


def test_density_equals_twice_sum_of_occupied_mo_squares(water):
    mol, bs, rhf = water
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=1.5, size=(40, 3))
    nocc = mol.nelec() // 2
    psi = ferric.eval_mo_on_points(mol, bs, rhf, list(range(nocc)), pts)
    rho = np.array(ferric.eval_density_on_points(mol, bs, rhf, pts))
    np.testing.assert_allclose(rho, 2.0 * (psi**2).sum(axis=1), rtol=1e-7, atol=1e-10)
    assert psi.shape == (40, nocc)


def test_ao_shape_and_psi_from_coefficients(water):
    mol, bs, rhf = water
    pts = np.random.default_rng(2).normal(size=(7, 3))
    chi = ferric.eval_ao_on_points(mol, bs, pts)
    assert chi.shape == (7, 7)  # water/STO-3G: 7 AOs
    C = np.asarray(rhf.mo_coefficients())
    psi = ferric.eval_mo_on_points(mol, bs, rhf, [0, 3], pts)
    np.testing.assert_allclose(psi, chi @ C[:, [0, 3]], rtol=1e-12, atol=1e-13)


def test_against_pyscf(h2):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto, scf

    mol, bs, rhf = h2
    pm = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom", verbose=0)
    pts = np.random.default_rng(3).normal(scale=1.2, size=(30, 3))  # Bohr
    ao = pm.eval_gto("GTOval_sph", pts)
    np.testing.assert_allclose(
        ferric.eval_ao_on_points(mol, bs, pts), ao, rtol=1e-6, atol=1e-9
    )
    mf = scf.RHF(pm).run()
    rho_ref = dft.numint.eval_rho(pm, ao, mf.make_rdm1())
    rho = np.array(ferric.eval_density_on_points(mol, bs, rhf, pts))
    np.testing.assert_allclose(rho, rho_ref, rtol=1e-5, atol=1e-8)


def test_errors(h2, tmp_path):
    mol, bs, rhf = h2
    pts = np.zeros((3, 2))
    with pytest.raises(ValueError, match="shape"):
        ferric.eval_ao_on_points(mol, bs, pts)
    with pytest.raises(IndexError):
        ferric.eval_mo_on_points(mol, bs, rhf, [5], np.zeros((1, 3)))
    p = str(tmp_path / "x.cube")
    with pytest.raises(ValueError, match="mo_index"):
        ferric.write_cube(p, mol, bs, rhf, what="mo")
    with pytest.raises(ValueError, match="unknown what"):
        ferric.write_cube(p, mol, bs, rhf, what="spin")
