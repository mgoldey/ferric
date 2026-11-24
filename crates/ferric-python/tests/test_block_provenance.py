"""Provenance of the building blocks: ``AoIntegrals.provenance()`` and
``certificate_from_blocks`` (ferric_core::certificate::blocks).

Anchors:

* a record round-trips through JSON, in Python and through the Rust parser
  (``certificate_from_blocks`` returns the records it was given);
* a record built with RI-J (``df_jk(aux)``, ``veff(aux=)``, ``df_mo(aux)``)
  names the auxiliary basis, and so does the composed certificate;
* recording does not change an array: the blocks return the same bits on a
  fresh object and on one that has logged many calls;
* a block composition is never graded, even on a Proven reference.

HOW THESE FAIL IF BROKEN: dropping the aux from the ``df_jk`` use
(``op_use(.., None)`` in blocks.rs) fails
``test_ri_j_record_names_the_aux_basis`` (seen to fail).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
OBS, RI, JK = "cc-pvdz", "cc-pvdz-ri", "def2-universal-jkfit"


def _setup():
    mol = ferric.Molecule.from_xyz_string(WATER)
    bs = ferric.BasisSet.bundled(OBS)
    return mol, bs, ferric.AoIntegrals(mol, bs)


def _density(n: int) -> np.ndarray:
    a = np.random.default_rng(7).standard_normal((n, n)) * 1e-2
    return a + a.T


def _uses(rec: dict, block: str) -> list[dict]:
    return [u for u in rec["uses"] if u["block"] == block]


def test_a_fresh_record_names_build_basis_and_element_sources():
    _, _, ints = _setup()
    rec = ints.provenance()
    assert rec["object"] == "AoIntegrals"
    assert rec["uses"] == []
    assert rec["approximations"] == []
    b = ferric.build_info()
    assert rec["build"]["commit"] == b["commit"]
    assert rec["basis"]["name"] == OBS
    assert [e["z"] for e in rec["basis"]["elements"]] == [1, 8]
    for e in rec["basis"]["elements"]:
        assert e["source"] == "BSE cc-pVDZ v1, BSE package 0.12", e


def test_record_round_trips_through_json():
    _, _, ints = _setup()
    n = ints.nao
    ints.overlap()
    ints.jk(_density(n))
    ints.df_mo(ferric.BasisSet.bundled(RI), np.eye(n)[:, :5], np.eye(n)[:, 5:])
    ints.xc_functional("pbe")
    ints.eri(operator="erf", omega=0.4)
    rec = ints.provenance()
    assert json.loads(json.dumps(rec)) == rec
    cert = ferric.certificate_from_blocks("roundtrip", [rec])
    assert cert["blocks"] == [rec]
    assert json.loads(json.dumps(cert)) == cert


def test_ri_j_record_names_the_aux_basis():
    _, _, ints = _setup()
    d = _density(ints.nao)
    jk_aux = ferric.BasisSet.bundled(JK)
    ints.df_jk(jk_aux).j(d)
    ints.veff(d, aux=jk_aux)
    rec = ints.provenance()
    (dfjk,) = _uses(rec, "df_jk")
    assert dfjk["aux"] is not None, "the RI-J record dropped its aux basis"
    assert dfjk["aux"]["name"] == JK
    assert dfjk["operator"] == "coulomb"
    (veff,) = _uses(rec, "veff")
    assert veff["aux"]["name"] == JK
    assert veff["xc"] is None
    for e in dfjk["aux"]["elements"]:
        assert e["source"].startswith("BSE def2-universal-JKFIT v"), e
    cert = ferric.certificate_from_blocks("ri-j", [rec])
    ri = [a["detail"] for a in cert["approximations"] if a["name"] == "ri"]
    assert ri == [f"aux {JK} (df_jk)", f"aux {JK} (veff)"]


def test_uses_collapse_and_count_and_carry_operator_xc_and_screening():
    _, _, ints = _setup()
    d = _density(ints.nao)
    ints.overlap()
    ints.overlap()
    ints.jk(d, thresh=1e-10)
    ints.jk(d, operator="erfc", omega=0.3)
    xc = ints.xc_functional("pbe")
    ints.veff(d, xc=xc)
    rec = ints.provenance()
    (ov,) = _uses(rec, "overlap")
    assert ov["calls"] == 2
    assert ov["operator"] is None and ov["aux"] is None
    jk = _uses(rec, "jk")
    assert [u["operator"] for u in jk] == ["coulomb", "erfc omega=0.3"]
    assert jk[0]["approximations"] == [
        {"name": "screening", "detail": "Schwarz threshold 1e-10 (jk)"}
    ]
    (veff,) = _uses(rec, "veff")
    assert veff["xc"].startswith("pbe on ("), veff
    assert _uses(rec, "xc_functional")[0]["xc"] == veff["xc"]
    names = [
        a["name"] for a in ferric.certificate_from_blocks("x", [rec])["approximations"]
    ]
    assert names == ["screening", "operator", "screening", "xc-grid"]


def test_recording_does_not_change_an_array():
    mol, bs, fresh = _setup()
    _, _, used = _setup()
    n = fresh.nao
    d = _density(n)
    ri = ferric.BasisSet.bundled(RI)
    c_o, c_v = np.eye(n)[:, :5], np.eye(n)[:, 5:]
    for _ in range(3):
        used.overlap()
        used.hcore()
        used.df_mo(ri, c_o, c_v)
        used.eri3(ri)
        used.jk(d)
    assert len(used.provenance()["uses"]) == 5
    pairs = [
        (fresh.overlap(), used.overlap()),
        (fresh.hcore(), used.hcore()),
        (fresh.eri3(ri), used.eri3(ri)),
        (fresh.df_mo(ri, c_o, c_v), used.df_mo(ri, c_o, c_v)),
        (fresh.eri2(ri), used.eri2(ri)),
    ]
    for a, b in pairs:
        assert type(a) is np.ndarray and type(b) is np.ndarray
        assert a.dtype == b.dtype == np.float64
        assert np.array_equal(a, b), "recording changed an array"


def test_a_block_composition_is_never_graded():
    mol, bs, ints = _setup()
    ints.hcore()
    ref = ferric.run_rhf(mol, bs).certificate
    assert ref["grade"] == "proven"
    cert = ferric.certificate_from_blocks(
        "my-method",
        [ints.provenance()],
        approximations=[("frozen-core", "1 core orbital frozen")],
        warnings=["user note"],
        reference=ref,
    )
    assert cert["capability"] == "blocks:my-method"
    assert cert["grade"] == "ungraded"
    assert cert["grade_source"] is None
    assert cert["build"]["commit"] == ref["build"]["commit"]
    assert cert["approximations"][-1] == {
        "name": "frozen-core",
        "detail": "1 core orbital frozen",
    }
    assert cert["warnings"] == ["user note"]


def test_records_from_another_build_warn():
    _, _, ints = _setup()
    a = ints.provenance()
    b = json.loads(json.dumps(a))
    b["build"]["commit"] = "0" * 40
    cert = ferric.certificate_from_blocks("x", [a, b])
    assert len(cert["warnings"]) == 1
    assert "different builds" in cert["warnings"][0]


def test_malformed_records_are_refused():
    _, _, ints = _setup()
    rec = ints.provenance()
    with pytest.raises(ValueError, match="at least one"):
        ferric.certificate_from_blocks("x", [])
    bad = dict(rec, objekt="AoIntegrals")
    with pytest.raises(ValueError, match=r"records\[0\]"):
        ferric.certificate_from_blocks("x", [bad])
    with pytest.raises(ValueError, match="reference"):
        ferric.certificate_from_blocks("x", [rec], reference={"grade": "proven"})
