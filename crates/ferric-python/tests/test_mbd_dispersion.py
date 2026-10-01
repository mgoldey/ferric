"""MBD@rsSCS dispersion through `run_dft(dispersion="mbd")`.

`run_dft` takes Hirshfeld volume ratios of the converged density, evaluates
MBD@rsSCS with the functional's published beta, adds it to the energy, and --
with `with_gradient=True` -- adds its analytic gradient (explicit term plus the
Hirshfeld-volume term with the density matrix held fixed) to the KS gradient.

What these pin, and the defect each would catch:
  * `total_energy == e_scf + e_dispersion` exactly, and `e_scf` equals the
    plain run's: the correction is post-SCF and additive (catches a correction
    folded into the SCF, or added twice / not at all).
  * `e_dispersion` equals the standalone `mbd_rsscs_energy` on the reported
    `volume_ratios`: the wiring feeds MBD exactly the ratios it reports
    (catches wrong beta, wrong atom order, ratios from a different density).
  * `dispersion=None` is byte-identical and reports None, never 0.0.
  * Unknown spellings and functionals without a published beta raise.
  * The gradient gains a finite, nonzero (natoms, 3) contribution whose size
    is consistent with the energy (catches the MBD gradient being dropped).

Kept fast: water at STO-3G / PBE.
"""

import math

import pytest

ferric = pytest.importorskip(
    "ferric",
    reason="the compiled ferric extension is not importable; build it with "
    "`cargo build --release -p ferric-python` and make sure the .so is on the "
    "site-packages path (see CLAUDE.md).",
)


WATER_XYZ = """3
water
O  0.0000000  0.0000000  0.1177900
H  0.0000000  0.7554530 -0.4711610
H  0.0000000 -0.7554530 -0.4711610
"""


@pytest.fixture(scope="module")
def water():
    return ferric.Molecule.from_xyz_string(WATER_XYZ, 0, 1)


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


@pytest.fixture(scope="module")
def plain(water, sto3g):
    return ferric.run_dft(water, sto3g, functional="PBE")


@pytest.fixture(scope="module")
def mbd(water, sto3g):
    return ferric.run_dft(water, sto3g, functional="PBE", dispersion="mbd")


def test_mbd_is_added_exactly_and_leaves_the_scf_alone(plain, mbd):
    assert mbd.converged
    assert mbd.dispersion_model == "MBD@rsSCS"
    assert mbd.e_scf == plain.e_scf, (
        "MBD@rsSCS is a post-SCF additive term; the SCF energy must not move"
    )
    assert mbd.total_energy == mbd.e_scf + mbd.e_dispersion


def test_mbd_energy_is_negative_and_small(mbd):
    e = mbd.e_dispersion
    assert e is not None
    assert -1e-2 < e < 0.0, f"water MBD@rsSCS should be a small attraction, got {e}"


def test_volume_ratios_are_reported_and_physical(water, mbd):
    r = mbd.volume_ratios
    assert r is not None and len(r) == len(water.symbols())
    # Atoms in a molecule are compressed relative to the free atom, but not
    # by an order of magnitude.
    for v in r:
        assert math.isfinite(v) and 0.2 < v < 1.5, r


def test_energy_matches_standalone_mbd_on_the_reported_ratios(water, mbd):
    ref = ferric.mbd_rsscs_energy(water, mbd.volume_ratios, functional="PBE")
    assert ref.beta == 0.83
    assert abs(mbd.e_dispersion - ref.energy) <= 1e-14, (
        f"run_dft {mbd.e_dispersion!r} vs standalone {ref.energy!r}"
    )


def test_explicit_functional_and_case_insensitive_spelling(water, sto3g, mbd):
    res = ferric.run_dft(water, sto3g, functional="PBE", dispersion="MBD(pbe)")
    assert res.e_dispersion == mbd.e_dispersion
    # A different published beta must give a different energy on the same
    # ratios, or the functional argument is being ignored.
    pbe0 = ferric.run_dft(water, sto3g, functional="PBE", dispersion="mbd(pbe0)")
    ref = ferric.mbd_rsscs_energy(water, pbe0.volume_ratios, functional="PBE0")
    assert abs(pbe0.e_dispersion - ref.energy) <= 1e-14
    assert pbe0.e_dispersion != mbd.e_dispersion


def test_no_dispersion_is_byte_identical(plain):
    assert plain.e_dispersion is None
    assert plain.dispersion_model is None
    assert plain.volume_ratios is None
    assert plain.total_energy == plain.e_scf


def test_d3_reports_its_model_and_no_ratios(water, sto3g):
    res = ferric.run_dft(water, sto3g, functional="PBE", dispersion="d3bj")
    assert res.dispersion_model == "D3(BJ)"
    assert res.volume_ratios is None


@pytest.mark.parametrize(
    "bad", ["mbd()", "mbd@rsscs", "mbd-nl", "ts", "mbd(b3lyp)", "mbd(not-a-functional)"]
)
def test_unknown_mbd_spec_raises(water, sto3g, bad):
    with pytest.raises(ValueError):
        ferric.run_dft(water, sto3g, functional="PBE", dispersion=bad)


def test_mbd_without_a_published_beta_for_the_running_functional_raises(water, sto3g):
    with pytest.raises(ValueError) as exc:
        ferric.run_dft(water, sto3g, functional="B3LYP", dispersion="mbd")
    assert "B3LYP" in str(exc.value)


def test_unknown_spec_error_lists_every_accepted_spelling(water, sto3g):
    with pytest.raises(ValueError) as exc:
        ferric.run_dft(water, sto3g, functional="PBE", dispersion="d4")
    msg = str(exc.value)
    for spelling in ['"d3bj"', '"d3(bj)"', "d3bj(<functional>)", '"mbd"', "mbd(<functional>)"]:
        assert spelling in msg, f"{spelling} missing from: {msg}"


def test_gradient_includes_the_mbd_gradient(water, sto3g, mbd):
    np = pytest.importorskip("numpy")
    g_plain = ferric.run_dft(water, sto3g, functional="PBE", with_gradient=True)
    g_mbd = ferric.run_dft(
        water, sto3g, functional="PBE", dispersion="mbd", with_gradient=True
    )
    a = np.asarray(g_plain.gradient())
    b = np.asarray(g_mbd.gradient())
    natoms = len(water.symbols())
    assert a.shape == (natoms, 3) and b.shape == (natoms, 3)
    # The gradient request must not change the energy it belongs to.
    assert g_mbd.e_dispersion == mbd.e_dispersion
    assert g_mbd.e_scf == g_plain.e_scf
    diff = b - a
    assert np.all(np.isfinite(diff))
    assert np.max(np.abs(diff)) > 0.0, "the MBD@rsSCS gradient was not added"
    # A dispersion gradient is bounded by the dispersion energy over a bond
    # length, not by the SCF gradient; an order-of-magnitude bound catches a
    # unit (Angstrom vs Bohr) or factor error that a nonzero check cannot.
    assert np.max(np.abs(diff)) < 10.0 * abs(mbd.e_dispersion), (
        f"MBD gradient {np.max(np.abs(diff))} is implausibly large for "
        f"E_MBD = {mbd.e_dispersion}"
    )
