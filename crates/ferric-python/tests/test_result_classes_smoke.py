"""Smoke coverage for result classes that previously had no test."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if (REPO / "terf-tables").is_dir():
    os.environ.setdefault("FERRIC_TERF_TABLE_DIR", str(REPO / "terf-tables"))

ferric = pytest.importorskip("ferric")


def _water():
    return ferric.Molecule.from_xyz_string(
        "3\n\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"
    )


def _oh():
    return ferric.Molecule.from_xyz_string("2\n\nO 0 0 0\nH 0 0 0.97\n", charge=0, multiplicity=2)


@pytest.fixture(scope="module")
def water():
    return _water()


@pytest.fixture(scope="module")
def oh():
    return _oh()


@pytest.fixture(scope="module")
def bs():
    return ferric.BasisSet.bundled("sto-3g")


@pytest.fixture(scope="module")
def aux():
    return ferric.BasisSet.bundled("cc-pvdz-ri")


def test_double_hybrid_result(water, bs, aux):
    r = ferric.run_double_hybrid(water, bs, aux, kind="b2plyp")
    assert type(r).__name__ == "DoubleHybridResult"
    assert isinstance(r.total_energy, float)

def test_wb97x_l_v_result(water, bs, aux):
    r = ferric.run_double_hybrid(water, bs, aux, kind="wb97x-l-v", lambda_=0.6, omega=0.1)
    assert type(r).__name__ == "DoubleHybridResult"
    assert r.c_os == 0.6


def test_mp3_result(water, bs, aux):
    r = ferric.run_mp3(water, bs, aux)
    assert type(r).__name__ == "Mp3Result"


def test_scs_mp2_result(water, bs, aux):
    r = ferric.run_scs_mp2(water, bs, aux)
    assert type(r).__name__ == "ScsMp2Result"


def test_attenuated_mp2_result(water, bs, aux):
    r = ferric.run_attenuated_rimp2(water, bs, aux)
    assert type(r).__name__ == "AttenuatedMp2Result"


def test_omega_ri_result(water, bs, aux):
    r = ferric.run_omega_ri(water, bs, aux, 0.0, method="rimp2")
    assert type(r).__name__ == "OmegaRiResult"


def test_rs_mp2_rpa_result(water, bs, aux):
    r = ferric.run_rs_mp2_rpa(water, bs, aux, omega=0.5)
    assert type(r).__name__ == "RsMp2RpaResult"


def test_tdhf_result(water, bs, aux):
    r = ferric.run_tdhf_static_polarizability(water, bs, aux, xc="lda", scissor=0.3)
    assert type(r).__name__ == "TdhfStaticPolarizabilityResult"


def test_oo_rimp2_result(water, bs, aux):
    r = ferric.run_oo_rimp2(water, bs, aux)
    assert type(r).__name__ == "OoRiMp2Result"


def test_casscf_result(water, bs):
    r = ferric.run_casscf(water, bs, 2, (1, 1))
    assert type(r).__name__ == "CasScfResult"


def test_nevpt2_result(water, bs):
    r = ferric.run_nevpt2(water, bs, 2, (1, 1), reference="casci")
    assert type(r).__name__ == "Nevpt2Result"


def test_eom_ccsd_result(water, bs, aux):
    r = ferric.run_eom_ccsd(water, bs, aux, n_roots=1)
    assert type(r).__name__ == "EomCcsdResult"


def test_mp2_v_result(water, bs, aux):
    r = ferric.run_mp2_v(water, bs, aux, omega=0.5)
    assert type(r).__name__ == "Mp2VResult"


def test_opmp3_result():
    water = _water()
    bs_cc = ferric.BasisSet.bundled("cc-pvdz")
    aux_cc = ferric.BasisSet.bundled("cc-pvdz-ri")
    r = ferric.run_opmp3(water, bs_cc, auxbasis=aux_cc)
    assert type(r).__name__ == "Opmp3Result"


def test_ugw_result(oh, bs, aux):
    r = ferric.run_u_gw(oh, bs, aux, xc=None)
    assert type(r).__name__ == "UGwResult"


def test_bse_result(water, bs):
    # bse needs gw? run_bse_tda requires auxbasis? check signature
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    r = ferric.run_bse_tda(water, bs, aux)
    assert type(r).__name__ == "BseResult"


def test_boltzmann_weights():
    r = ferric.boltzmann_weights([0.0, 1.0], temperature_k=300.0)
    assert type(r).__name__ == "BoltzmannWeights"


def test_response_modes(water, bs, aux):
    r = ferric.response_modes(water, bs, aux, ferric.run_rhf(water, bs))
    assert type(r).__name__ == "ResponseModes"
