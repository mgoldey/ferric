"""D3(BJ) dispersion through the Python bindings.

These cover the surface `tools/pipeline` tier 4 actually calls (`run_dft`) plus
the standalone `d3bj_energy`, and they pin the two properties that matter most
for not silently changing existing results:

  1. `dispersion=None` leaves the energy EXACTLY as it was, and reports
     `e_dispersion is None` (UNEVALUATED) rather than 0.0.
  2. Anything that cannot be computed RAISES, rather than contributing zero.
"""

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


@pytest.fixture
def water():
    return ferric.Molecule.from_xyz_string(WATER_XYZ, 0, 1)


@pytest.fixture
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


def test_standalone_dispersion_is_attractive_and_functional_dependent(water):
    """The correction must be negative, and must actually depend on the
    functional -- a lookup that ignored its argument would return one number."""
    e_pbe = ferric.d3bj_energy(water, "PBE")
    e_b3lyp = ferric.d3bj_energy(water, "B3LYP")
    assert e_pbe < 0.0, f"dispersion must be attractive, got {e_pbe}"
    assert e_b3lyp < 0.0
    assert e_pbe != e_b3lyp, (
        "PBE and B3LYP have different published D3(BJ) parameters, so they must "
        "give different corrections; identical values mean the functional "
        "argument is being ignored"
    )


def test_single_atom_has_exactly_zero_dispersion():
    """The exactness anchor, through the Python surface: one atom has no pairs."""
    atom = ferric.Molecule.from_xyz_string("1\nne\nNe 0.0 0.0 0.0\n", 0, 1)
    assert ferric.d3bj_energy(atom, "PBE") == 0.0


def test_unknown_functional_raises_rather_than_defaulting(water):
    """No silent fallback to some other functional's parameters."""
    with pytest.raises(Exception) as exc:
        ferric.d3bj_energy(water, "definitely-not-a-functional")
    assert "definitely-not-a-functional" in str(exc.value), (
        "the error must name the functional it rejected so the caller can act"
    )


def test_run_dft_without_dispersion_is_unchanged(water, sto3g):
    """`dispersion=None` must not perturb the energy at all, and must report
    UNEVALUATED rather than a zero correction."""
    res = ferric.run_dft(water, sto3g, functional="PBE")
    assert res.converged
    assert res.e_dispersion is None, (
        "not asking for dispersion must give None (UNEVALUATED), never 0.0 -- a "
        "reported zero is a physics claim that no dispersion was found"
    )
    assert res.total_energy == res.e_scf


def test_run_dft_with_dispersion_adds_exactly_the_standalone_value(water, sto3g):
    """The SCF must be untouched and the correction must be added exactly."""
    plain = ferric.run_dft(water, sto3g, functional="PBE")
    disp = ferric.run_dft(water, sto3g, functional="PBE", dispersion="d3bj")

    assert disp.e_scf == plain.e_scf, (
        "adding a dispersion correction must not change the SCF itself -- D3 is "
        "a post-SCF additive term, not a modification of the Fock operator"
    )
    assert disp.e_dispersion == ferric.d3bj_energy(water, "PBE")
    assert disp.total_energy == disp.e_scf + disp.e_dispersion
    assert disp.total_energy < plain.total_energy, "dispersion must lower the energy"


def test_dispersion_parameter_override(water, sto3g):
    """`d3bj(<name>)` uses the named functional's parameters, not the running
    functional's -- needed when ferric's XC name differs from the D3 fit's."""
    res = ferric.run_dft(water, sto3g, functional="PBE", dispersion="d3bj(b3lyp)")
    assert res.e_dispersion == ferric.d3bj_energy(water, "B3LYP")
    assert res.e_dispersion != ferric.d3bj_energy(water, "PBE")


@pytest.mark.parametrize("bad", ["d4", "xdm", "vv10", "none", "off", "true", "d3bj()"])
def test_unknown_dispersion_scheme_raises(water, sto3g, bad):
    """There is deliberately no spelling that means "compute a zero
    correction" -- including the ones a user might expect to mean "off"."""
    with pytest.raises(Exception):
        ferric.run_dft(water, sto3g, functional="PBE", dispersion=bad)


def test_matches_simple_dftd3_when_available(water):
    """Cross-check against the independent reference implementation, if it is
    installed. Skipped rather than failed when it is not, because this is a
    validation convenience and not a runtime dependency of ferric."""
    dftd3 = pytest.importorskip(
        "dftd3.interface", reason="simple-dftd3 not installed; cross-check skipped"
    )
    import numpy as np

    sym2z = {"H": 1, "C": 6, "N": 7, "O": 8}
    nums = np.array([sym2z[s] for s in water.symbols()], dtype=np.int32)
    coords = np.array(water.coords_bohr())

    model = dftd3.DispersionModel(nums, coords)
    for name, p in [
        ("PBE", dict(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407)),
        ("B3LYP", dict(s6=1.0, s8=1.9889, a1=0.3981, a2=4.4211)),
    ]:
        pairwise = model.get_pairwise_dispersion(dftd3.RationalDampingParam(**p))
        # Compare against the reference's TWO-BODY term: ferric does not
        # implement the ATM three-body term, so comparing against the total
        # would hide a real two-body disagreement inside the tolerance.
        ref_two_body = float(pairwise["additive pairwise energy"].sum())
        got = ferric.d3bj_energy(water, name)
        assert abs(got - ref_two_body) < 1e-12, (
            f"{name}: ferric {got:.14e} vs simple-dftd3 two-body {ref_two_body:.14e}"
        )
