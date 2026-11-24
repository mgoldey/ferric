"""`ferric.molden`: Molden files from SCF results, and UhfResult's orbitals.

Anchors:

* AO convention (runs without PySCF): the values at fixed points of the AOs
  a Molden file describes, as PySCF's Molden reader defines them, recorded by
  ``scripts/gen_molden_refs.py`` for an O-H pair in cc-pVQZ (spherical d, f,
  g) and in the same shells made Cartesian (6D 10F 15G). ferric's AO values
  must equal them after ``ao_map``: ``chi_ferric[:, perm] == chi_molden *
  scale``. Both sides evaluate the same Gaussians in double precision with
  contractions normalised the same way, so they agree to ~1e-14 of the
  largest value; the bar is 1e-10 of it. A swapped pair of d, f or g
  components, or a dropped sqrt(S_ii) factor, moves an AO by O(1) of its
  size, and the guard tests below check each of those fails the bar.
* Round trip (needs PySCF): ``pyscf.tools.molden.load`` reads ferric's file
  for RHF, KS, UHF and ROHF results over s-g shells, spherical and
  Cartesian, with an ECP and with a ghost atom. Orbital energies and
  occupations come back bit-identical (17 significant digits round-trip a
  double exactly). MO values and the density on points, evaluated by PySCF
  from the loaded file and by ferric from the result, agree to 1e-10 of
  their largest value: the coefficients are exact, so the difference is the
  two codes' evaluation of the same functions (~1e-14). The loaded MOs are
  orthonormal in PySCF's own overlap to 1e-10 (ferric's are to ~1e-13), and
  tr(D S) is the electron count.
* Cross-check (needs PySCF): PySCF's own SCF on the same numbers, written by
  ``pyscf.tools.molden.from_scf`` and loaded back, has the density of
  ferric's loaded file. Both SCFs converge the density to ~1e-9 (ferric
  density_conv 1e-10 RMS, PySCF conv_tol 1e-12, i.e. gradient ~1e-6), so the
  densities agree to ~1e-7 of their peak; the bar is 1e-5 of it.
* Independent construction: the Avogadro plugin measures its AO map by a
  least-squares fit of ferric's AO values against a port of Avogadro's
  evaluator, whose d and f conventions (order and per-component
  normalisation) are Molden's. ``ao_map`` builds the map from tables; the
  two must agree shell by shell (s-f) to 1e-8.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import ferric

try:
    from ferric import molden
except ImportError as err:  # the extension without the python package layer
    pytest.skip(f"ferric.molden is not importable: {err}", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "testdata" / "reference"
AO_REF = REF / "molden_ao_values_pyscf.json"
CART_BASIS = REF / "molden_cart_spdfg_basis.json"
PLUGIN_SRC = ROOT / "plugins" / "avogadro-ferric" / "src"

AO_BAR = 1e-10
ROUND_TRIP_BAR = 1e-10
CROSS_BAR = 1e-5

WATER = [
    ("O", (0.0, 0.0, 0.2214)),
    ("H", (0.0, 1.4305, -0.8857)),
    ("H", (0.12, -1.4305, -0.8857)),
]


# ── AO convention, no PySCF ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def ao_ref():
    ref = json.loads(AO_REF.read_text())
    syms = [a[0] for a in ref["atoms"]]
    coords = [a[2] for a in ref["atoms"]]
    mol = ferric.Molecule.from_arrays(syms, coords, units="bohr", multiplicity=2)
    bases = {
        "cc-pvqz": ferric.BasisSet.bundled("cc-pvqz"),
        "cart-spdfg": ferric.BasisSet.from_bse_json(str(CART_BASIS)),
    }
    return ref, mol, bases, np.asarray(ref["points"])


def _ao_error(ao_ref, case, perm=None, scale=None):
    ref, mol, bases, points = ao_ref
    bs = bases[case]
    chi_f = np.asarray(ferric.eval_ao_on_points(mol, bs, points))
    p0, s0 = molden.ao_map(mol, bs)
    perm = p0 if perm is None else perm
    scale = s0 if scale is None else scale
    chi_m = np.asarray(ref["cases"][case]["ao_values"]) * scale
    return np.max(np.abs(chi_f[:, perm] - chi_m)) / np.max(np.abs(chi_m))


@pytest.mark.parametrize("case", ["cc-pvqz", "cart-spdfg"])
def test_ao_map_reproduces_pyscf_molden_aos(ao_ref, case):
    ref, mol, bases, _ = ao_ref
    assert bases[case].n_functions(mol) == ref["cases"][case]["nao"]
    err = _ao_error(ao_ref, case)
    assert err < AO_BAR, f"{case}: max relative AO error {err:.2e}"


def _first_shell_offset(mol, bs, ang):
    off = 0
    for _, l_, pure, _, _ in molden._shell_list(mol, bs):
        n = 2 * l_ + 1 if pure else (l_ + 1) * (l_ + 2) // 2
        if l_ == ang:
            return off
        off += n
    raise AssertionError(f"no l={ang} shell")


@pytest.mark.parametrize("case", ["cc-pvqz", "cart-spdfg"])
@pytest.mark.parametrize("ang", [2, 3, 4])
def test_swapped_components_fail_the_ao_bar(ao_ref, case, ang):
    """Guard: the AO anchor sees two components of one shell exchanged."""
    _, mol, bases, _ = ao_ref
    perm, _ = molden.ao_map(mol, bases[case])
    off = _first_shell_offset(mol, bases[case], ang)
    perm = perm.copy()
    perm[[off, off + 1]] = perm[[off + 1, off]]
    assert _ao_error(ao_ref, case, perm=perm) > 1e-2


def test_dropped_normalisation_fails_the_ao_bar(ao_ref):
    """Guard: Cartesian components need ferric's AO norms, sqrt(S_ii)."""
    _, mol, bases, _ = ao_ref
    _, scale = molden.ao_map(mol, bases["cart-spdfg"])
    assert _ao_error(ao_ref, "cart-spdfg", scale=np.ones_like(scale)) > 1e-2


def test_flags_follow_shell_types():
    sph = [(0, 2, True, [], []), (0, 3, True, [], []), (0, 4, True, [], [])]
    assert molden._flags(sph) == ["[5D7F]", "[9G]"]
    assert molden._flags([(0, 2, True, [], []), (0, 3, False, [], [])]) == ["[5D10F]"]
    assert molden._flags([(0, 2, False, [], []), (0, 3, True, [], [])]) == ["[7F]"]
    assert molden._flags([(0, 2, False, [], []), (0, 4, False, [], [])]) == []
    with pytest.raises(ValueError, match="mixes spherical and Cartesian d"):
        molden._flags([(0, 2, True, [], []), (1, 2, False, [], [])])


def test_writes_open_shell_file_layout(tmp_path):
    mol = ferric.Molecule.from_arrays(
        ["O", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.83)], units="bohr", multiplicity=2
    )
    bs = ferric.BasisSet.bundled("sto-3g")
    res = ferric.run_uhf(mol, bs)
    path = tmp_path / "oh.molden"
    molden.write(path, mol, bs, res, title="OH")
    text = path.read_text()
    nao = bs.n_functions(mol)
    assert text.startswith("[Molden Format]\n[Title]\n OH\n[Atoms] AU\n")
    assert text.count("Spin= Alpha") == nao and text.count("Spin= Beta") == nao
    assert "[5D" not in text and "[core]" not in text
    occ = [float(line.split("=")[1]) for line in text.splitlines() if "Occup=" in line]
    assert sum(occ) == pytest.approx(9.0, abs=1e-10)


def test_uhf_result_orbital_layout():
    """density_s == C_s diag(occ_s) C_s^T, columns in energy order (G3 layout).

    UHF returns the last iteration's density next to the orbitals of its Fock
    matrix, so the two agree to the density convergence: with density_conv
    1e-10 (dp_max gated at 1e-9) the elements differ by ~1e-9; the bar is
    1e-7. ROHF's density is built from its returned orbitals (exact)."""
    mol = ferric.Molecule.from_arrays(
        ["O", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.83)], units="bohr", multiplicity=2
    )
    bs = ferric.BasisSet.bundled("6-31g*")
    s = ferric.AoIntegrals(mol, bs).overlap()
    for run, nmo_sets in ((ferric.run_uhf, 2), (ferric.run_rohf, 1)):
        res = run(mol, bs, energy_conv=1e-12, density_conv=1e-10)
        ca, cb = res.mo_coefficients_alpha(), res.mo_coefficients_beta()
        oa, ob = res.occupations_alpha(), res.occupations_beta()
        ea, eb = res.orbital_energies_alpha(), res.orbital_energies_beta()
        assert ca.shape == cb.shape and ca.shape[1] == len(ea) == len(eb)
        assert np.all(np.diff(ea) >= 0) and np.all(np.diff(eb) >= 0)
        for c in (ca, cb):
            np.testing.assert_allclose(c.T @ s @ c, np.eye(c.shape[1]), atol=1e-10)
        for c, o, d in ((ca, oa, res.density_alpha()), (cb, ob, res.density_beta())):
            np.testing.assert_allclose((c * o) @ c.T, d, atol=1e-7)
        assert oa.sum() == pytest.approx(5.0, abs=1e-10)
        assert ob.sum() == pytest.approx(4.0, abs=1e-10)
        if nmo_sets == 1:  # ROHF: one MO set
            np.testing.assert_array_equal(ca, cb)
            np.testing.assert_array_equal(ea, eb)


def test_refuses_results_without_orbitals():
    mol = ferric.Molecule.from_arrays(
        ["H", "H"], [(0, 0, 0), (0, 0, 1.4)], units="bohr"
    )
    bs = ferric.BasisSet.bundled("sto-3g")
    with pytest.raises(TypeError, match="carries no molecular orbitals"):
        molden.to_string(mol, bs, object())


# ── Independent construction: the Avogadro plugin's measured map ─────────


@pytest.mark.parametrize("basis", ["cc-pvtz", "6-31g*"])
def test_ao_map_matches_the_avogadro_plugins_fitted_map(basis):
    if not PLUGIN_SRC.is_dir():
        pytest.skip("plugins/avogadro-ferric is not in this checkout")
    sys.path.insert(0, str(PLUGIN_SRC))
    try:
        from avogadro_ferric import orbitals as avo
    finally:
        sys.path.remove(str(PLUGIN_SRC))
    mol = ferric.Molecule.from_arrays(
        [s for s, _ in WATER], [xyz for _, xyz in WATER], units="bohr"
    )
    bs = ferric.BasisSet.bundled(basis)
    _, maps = avo.basis_and_maps(ferric, mol, bs, mol.atomic_numbers())
    perm, scale = molden.ao_map(mol, bs)
    nao = len(perm)
    p_molden = np.zeros((nao, nao))
    p_molden[np.arange(nao), perm] = scale
    p_avo = np.zeros((nao, nao))
    for cols, m in maps:
        p_avo[cols, cols] = m
    np.testing.assert_allclose(p_molden, p_avo, atol=1e-8)


# ── Round trip and cross-check through PySCF ─────────────────────────────


def _points(coords_bohr, n=40, seed=7):
    rng = np.random.default_rng(seed)
    c = np.asarray(coords_bohr)
    return c[np.arange(n) % len(c)] + rng.normal(scale=0.8, size=(n, 3))


def _channels(mo_coeff, mo_occ):
    if isinstance(mo_coeff, tuple):
        return list(zip(mo_coeff, mo_occ, strict=True))
    return [(mo_coeff, mo_occ)]


def _pyscf_ao(pmol, points):
    return pmol.eval_gto("GTOval_cart" if pmol.cart else "GTOval_sph", points)


def _rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def _round_trip(tmp_path, mol, bs, res, n_channels):
    from pyscf.tools import molden as pmolden

    path = tmp_path / "out.molden"
    molden.write(path, mol, bs, res)
    pmol, energies, coeffs, occs, _, _ = pmolden.load(str(path))
    chans = molden.orbitals(mol, bs, res)
    loaded = _channels(coeffs, occs)
    energies = list(energies) if isinstance(energies, tuple) else [energies]
    assert len(loaded) == len(chans) == n_channels

    points = _points(mol.coords_bohr())
    chi_p = _pyscf_ao(pmol, points)
    chi_f = np.asarray(ferric.eval_ao_on_points(mol, bs, points))
    s_p = pmol.intor("int1e_ovlp")
    rho_f = np.zeros(len(points))
    rho_p = np.zeros(len(points))
    for ch, (c_p, o_p), e_p in zip(chans, loaded, energies, strict=True):
        np.testing.assert_array_equal(e_p, ch["energies"])
        np.testing.assert_array_equal(o_p, ch["occupations"])
        mo_f = chi_f @ ch["coefficients"]
        mo_p = chi_p @ c_p
        assert _rel(mo_p, mo_f) < ROUND_TRIP_BAR
        ortho = c_p.T @ s_p @ c_p
        assert np.max(np.abs(ortho - np.eye(len(ortho)))) < ROUND_TRIP_BAR
        assert np.trace((c_p * o_p) @ c_p.T @ s_p) == pytest.approx(
            o_p.sum(), abs=1e-10
        )
        rho_f += np.einsum("pk,k,pk->p", mo_f, ch["occupations"], mo_f)
        rho_p += np.einsum("pk,k,pk->p", mo_p, o_p, mo_p)
    assert _rel(rho_p, rho_f) < ROUND_TRIP_BAR
    return path, pmol


def _water(ghost_he=False):
    syms = [s for s, _ in WATER]
    coords = [xyz for _, xyz in WATER]
    ghost = None
    if ghost_he:
        syms.append("He")
        coords.append((0.0, 0.0, 3.5))
        ghost = [False, False, False, True]
    return ferric.Molecule.from_arrays(syms, coords, units="bohr", ghost=ghost)


@pytest.fixture(scope="module")
def pyscf_molden():
    pytest.importorskip("pyscf")


@pytest.mark.parametrize(
    "basis",
    ["cc-pvtz", "cc-pvqz", "6-31g*", "cart-spdfg"],
)
def test_rhf_round_trip_through_pyscf(pyscf_molden, tmp_path, basis):
    mol = _water()
    bs = (
        ferric.BasisSet.from_bse_json(str(CART_BASIS))
        if basis == "cart-spdfg"
        else ferric.BasisSet.bundled(basis)
    )
    _, pmol = _round_trip(tmp_path, mol, bs, ferric.run_rhf(mol, bs), 1)
    assert pmol.cart == (basis in ("6-31g*", "cart-spdfg"))


def test_dft_round_trip_through_pyscf(pyscf_molden, tmp_path):
    mol = _water()
    bs = ferric.BasisSet.bundled("def2-svp")
    _round_trip(tmp_path, mol, bs, ferric.run_dft(mol, bs, functional="PBE"), 1)


@pytest.mark.parametrize("method", ["uhf", "rohf"])
def test_open_shell_round_trip_through_pyscf(pyscf_molden, tmp_path, method):
    mol = ferric.Molecule.from_arrays(
        ["O", "H"], [(0.0, 0.0, 0.0), (0.3, 0.0, 1.83)], units="bohr", multiplicity=2
    )
    bs = ferric.BasisSet.bundled("cc-pvtz")
    run = ferric.run_uhf if method == "uhf" else ferric.run_rohf
    _, pmol = _round_trip(tmp_path, mol, bs, run(mol, bs), 2)
    assert pmol.spin == 1


def test_ecp_round_trip_writes_valence_charge_and_core(pyscf_molden, tmp_path):
    mol = ferric.Molecule.from_arrays(
        ["H", "I"], [(0, 0, 0), (0, 0, 3.04)], units="bohr"
    )
    bs = ferric.BasisSet.bundled("def2-svp")
    assert bs.ecp_core_electrons(53) == 28 and bs.ecp_core_electrons(1) == 0
    path, _ = _round_trip(tmp_path, mol, bs, ferric.run_rhf(mol, bs), 1)
    lines = path.read_text().splitlines()
    atoms = lines[lines.index("[Atoms] AU") + 1 : lines.index("[Atoms] AU") + 3]
    assert [int(a.split()[2]) for a in atoms] == [1, 25]
    assert lines[lines.index("[core]") + 1] == "2 : 28"


def test_ghost_atom_round_trip(pyscf_molden, tmp_path):
    mol = _water(ghost_he=True)
    bs = ferric.BasisSet.bundled("cc-pvdz")
    path, _ = _round_trip(tmp_path, mol, bs, ferric.run_rhf(mol, bs), 1)
    lines = path.read_text().splitlines()
    he = lines[lines.index("[Atoms] AU") + 4].split()
    assert he[:3] == ["He", "4", "0"]


def _pyscf_mol_from_ferric(mol, bs, spin):
    from pyscf import gto

    syms = mol.symbols()
    basis = {
        s: [[ang, *zip(e, c, strict=True)] for ang, _, e, c in bs.shells(z)]
        for s, z in zip(syms, mol.atomic_numbers(), strict=True)
    }
    pure = {
        p
        for z in set(mol.atomic_numbers())
        for ang, p, _, _ in bs.shells(z)
        if ang >= 2
    }
    return gto.M(
        atom=[(s, xyz) for s, xyz in zip(syms, mol.coords_bohr(), strict=True)],
        unit="Bohr",
        basis=basis,
        cart=pure == {False},
        spin=spin,
        verbose=0,
    )


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_density_matches_a_pyscf_written_molden(pyscf_molden, tmp_path, method):
    from pyscf import scf
    from pyscf.tools import molden as pmolden

    if method == "rhf":
        mol, spin = _water(), 0
        bs = ferric.BasisSet.bundled("cc-pvtz")
        res = ferric.run_rhf(mol, bs, energy_conv=1e-12, density_conv=1e-10)
    else:
        # NH2 (2B1): a non-degenerate open shell. OH would not do: its two pi
        # states are degenerate, and the codes may land on either.
        mol = ferric.Molecule.from_arrays(
            ["N", "H", "H"],
            [(0.0, 0.0, 0.0), (0.0, 1.55, 1.10), (0.0, -1.52, 1.12)],
            units="bohr",
            multiplicity=2,
        )
        spin = 1
        bs = ferric.BasisSet.bundled("cc-pvdz")
        res = ferric.run_uhf(mol, bs, energy_conv=1e-12, density_conv=1e-10)
    ours = tmp_path / "ferric.molden"
    molden.write(ours, mol, bs, res)

    pmol = _pyscf_mol_from_ferric(mol, bs, spin)
    mf = (scf.RHF if spin == 0 else scf.UHF)(pmol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    theirs = tmp_path / "pyscf.molden"
    pmolden.from_scf(mf, str(theirs))

    points = _points(mol.coords_bohr())
    rho = []
    eps = []
    for path in (ours, theirs):
        lmol, e, c, o, _, _ = pmolden.load(str(path))
        chi = _pyscf_ao(lmol, points)
        rho.append(
            sum(
                np.einsum("pk,k,pk->p", chi @ ck, ok, chi @ ck)
                for ck, ok in _channels(c, o)
            )
        )
        eps.append(np.concatenate(e) if isinstance(e, tuple) else e)
    assert _rel(rho[0], rho[1]) < CROSS_BAR
    np.testing.assert_allclose(eps[0], eps[1], atol=1e-6)
