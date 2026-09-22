"""PDEP must see the environment its reference SCF sits in.

`run_pdep_rpa` computes a dielectric response from an RHF reference. That
reference was always built by `rhf_config_budgeted(...)`, which sets neither
`external_potential` nor `pcm` -- so the response was always that of an
ISOLATED molecule, with no way to pass a field in.

Screening a ligand-pocket interaction needs the response of the ligand as it
sits in the pocket, so the point charges have to reach that SCF.
"""

import pytest

ferric = pytest.importorskip("ferric")

WATER = "/home/matt/qc/ferric/testdata/molecules/water.xyz"


@pytest.fixture(scope="module")
def setup():
    mol = ferric.Molecule.from_xyz(WATER)
    return mol, ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled(
        "cc-pvdz-ri"
    )


def test_no_environment_is_BIT_IDENTICAL_to_omitting_the_kwargs(setup):
    """The trivial-limit anchor: an absent field must change nothing."""
    mol, bs, aux = setup
    plain = ferric.run_pdep_rpa(mol, bs, aux)
    explicit = ferric.run_pdep_rpa(
        mol, bs, aux, point_charges=None, external_field=None, solvent=None
    )
    assert plain.total_energy == explicit.total_energy
    assert plain.e_rpa == explicit.e_rpa


def test_a_point_charge_REACHES_the_reference_scf(setup):
    """Without this the kwarg would be accepted and silently ignored.

    That is the failure mode being prevented: a caller passes a pocket, gets
    an isolated-molecule response back, and nothing says so.
    """
    mol, bs, aux = setup
    vac = ferric.run_pdep_rpa(mol, bs, aux)
    charged = ferric.run_pdep_rpa(
        mol, bs, aux, point_charges=[(0.5, 0.0, 0.0, 6.0)]
    )
    assert abs(charged.total_energy - vac.total_energy) > 1e-9, (
        "a 0.5 e charge 6 Bohr away changed nothing -- it never reached the SCF"
    )
    # The SCF energy itself must move, not just a downstream aggregate.
    assert abs(charged.rhf_energy - vac.rhf_energy) > 1e-9


def test_a_bigger_charge_perturbs_more(setup):
    """Monotonicity: the response scales with the field, not a constant."""
    mol, bs, aux = setup
    vac = ferric.run_pdep_rpa(mol, bs, aux).rhf_energy
    small = ferric.run_pdep_rpa(
        mol, bs, aux, point_charges=[(0.25, 0.0, 0.0, 6.0)]
    ).rhf_energy
    big = ferric.run_pdep_rpa(
        mol, bs, aux, point_charges=[(1.0, 0.0, 0.0, 6.0)]
    ).rhf_energy
    assert abs(small - vac) < abs(big - vac)


def test_solvent_reaches_the_reference_scf(setup):
    mol, bs, aux = setup
    vac = ferric.run_pdep_rpa(mol, bs, aux)
    aq = ferric.run_pdep_rpa(mol, bs, aux, solvent="water")
    assert aq.rhf_energy < vac.rhf_energy, "solvation must stabilise water"


def test_an_unknown_solvent_still_ERRORS_here(setup):
    """The validation must not be bypassed on this path."""
    mol, bs, aux = setup
    with pytest.raises(ValueError, match="not recognised"):
        ferric.run_pdep_rpa(mol, bs, aux, solvent="watr")
