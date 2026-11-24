"""run_rs_mp2_rpa with list-valued omega / formulation: one SCF, a list of
results, each bit-identical to the matching scalar call."""

import ferric
import pytest

_WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
_FORMS = ["delta-lr", "coupled-rings"]
_OMEGAS = [0.2, 0.42]
_FIELDS = (
    "total_energy",
    "e_corr",
    "e_mp2_full",
    "e_sr_mp2",
    "e_lr_mp2",
    "e_dmp2_lr",
    "e_drpa_lr",
    "e_corr_naive",
    "e_delta_drpa_full",
    "e_delta_drpa_sr",
)


def _args():
    mol = ferric.Molecule.from_xyz_string(_WATER)
    return (
        mol,
        ferric.BasisSet.bundled("cc-pvdz"),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
    )


def test_scan_matches_scalar_calls_bitwise():
    mol, bs, aux = _args()
    scan = ferric.run_rs_mp2_rpa(mol, bs, aux, omega=_OMEGAS, formulation=_FORMS)
    assert isinstance(scan, list) and len(scan) == 4
    k = 0
    for w in _OMEGAS:
        for f in _FORMS:
            one = ferric.run_rs_mp2_rpa(mol, bs, aux, omega=w, formulation=f)
            assert not isinstance(one, list)
            for name in _FIELDS:
                assert getattr(scan[k], name) == getattr(one, name), (w, f, name)
            k += 1


def test_scalar_calls_return_one_result():
    mol, bs, aux = _args()
    assert not isinstance(ferric.run_rs_mp2_rpa(mol, bs, aux), list)
    # A one-element list is still a scan.
    r = ferric.run_rs_mp2_rpa(mol, bs, aux, omega=[0.42])
    assert isinstance(r, list) and len(r) == 1


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"omega": []}, "empty"),
        ({"formulation": []}, "empty"),
        ({"formulation": ["delta-lr", "nope"]}, "unknown formulation"),
        ({"omega": [0.2], "attenuator": "terf"}, "terf_omega"),
        ({"terf_omega": [0.2, 0.4]}, "attenuator"),
        ({"omega": [0.2, -1.0]}, "omega"),
        ({"formulation": ["delta-lr", "delta-lr"]}, "twice"),
    ],
)
def test_scan_misuse_is_a_value_error(kwargs, needle):
    mol, bs, aux = _args()
    with pytest.raises(ValueError, match=needle):
        ferric.run_rs_mp2_rpa(mol, bs, aux, **kwargs)
