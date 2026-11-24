"""The Single Point cache that keeps Avogadro's 10 s electrostatic calls cheap.

Each test drives the plugin as Avogadro does (one subprocess per call) with a
private cache directory (avoferric_testlib.plugin_env). The stderr lines
``ferric cache: ...`` are the observable that a call was served from, or
started from, the cache; the values are compared with a cold run in a
separate, empty cache directory.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from avoferric_testlib import WATER, menu_input, run_plugin


def _charges_input(atoms, charge=0):
    return {"cjson": menu_input(atoms, charge)["cjson"]}


def _cold_env(plugin_env, tmp_path_factory, **cfg):
    """An environment whose cache directory is new and empty."""
    env = plugin_env(**cfg)
    env["AVOGADRO_FERRIC_CACHE"] = str(tmp_path_factory.mktemp("cold-cache"))
    return env


@pytest.mark.parametrize(
    "ident", ["ferric_lowdin", "ferric_hirshfeld", "ferric_chelpg"]
)
def test_single_point_serves_every_menu_charge_model(
    plugin_env, tmp_path_factory, ident
):
    """Single Point stores all three models Avogadro lists, whatever the configured
    one; each --charges call is then a cache read with the cold values."""
    env = plugin_env(method="HF", basis="sto-3g", charge_model="lowdin")
    run_plugin(env, ["sp"], menu_input(WATER))
    out, err = run_plugin(env, [ident, "--charges"], _charges_input(WATER))
    assert "charges from the cache" in err
    cold, cold_err = run_plugin(
        _cold_env(plugin_env, tmp_path_factory, method="HF", basis="sto-3g"),
        [ident, "--charges"],
        _charges_input(WATER),
    )
    assert "ferric cache" not in cold_err
    # Both are cold SCFs of the same input: identical up to run-to-run round-off.
    assert out["charges"] == pytest.approx(cold["charges"], abs=1e-10)


def test_potentials_start_from_the_stored_density(plugin_env, tmp_path_factory):
    """--potentials cannot be precomputed (Avogadro chooses the points), so its SCF
    starts from the stored converged density.

    A converged start must finish in at most 2 iterations (ferric's dm0 contract;
    a cold water/STO-3G RHF takes ~10). The ESP agrees with a cold run to the SCF
    convergence bar: density_conv 1e-6 RMS moves the potential by O(1e-6)
    Hartree/e, x 1.89 for e/Angstrom; 1e-5 leaves a factor ~5.
    """
    env = plugin_env(method="HF", basis="sto-3g")
    run_plugin(env, ["sp"], menu_input(WATER))
    payload = _charges_input(WATER)
    payload["points"] = [[3.0, 0.0, 0.0], [0.0, 2.5, 1.0], [0.0, 0.0, -3.0]]
    out, err = run_plugin(env, ["ferric_hirshfeld", "--potentials"], payload)
    if "cached density not used" in err and "dm0" in err:
        pytest.skip("this ferric build has no dm0 (needs ferric after #657)")
    m = re.search(r"ferric cache: SCF took (\d+) iterations", err)
    assert m, err
    assert int(m.group(1)) <= 2
    cold, _ = run_plugin(
        _cold_env(plugin_env, tmp_path_factory, method="HF", basis="sto-3g"),
        ["ferric_hirshfeld", "--potentials"],
        payload,
    )
    assert np.asarray(out["potentials"]) == pytest.approx(cold["potentials"], abs=1e-5)


def test_a_moved_atom_is_a_miss(plugin_env):
    env = plugin_env(method="HF", basis="sto-3g")
    run_plugin(env, ["sp"], menu_input(WATER))
    moved = [WATER[0], (1, 0.0, 0.758, -0.469), WATER[2]]
    _, err = run_plugin(env, ["ferric_lowdin", "--charges"], _charges_input(moved))
    assert "ferric cache" not in err


def test_other_settings_are_a_miss(plugin_env):
    """The electrostatic models run with the configured method/basis; after the
    configuration changes, the Single Point entry no longer applies."""
    run_plugin(plugin_env(method="HF", basis="sto-3g"), ["sp"], menu_input(WATER))
    env = plugin_env(method="PBE", basis="sto-3g")
    _, err = run_plugin(env, ["ferric_lowdin", "--charges"], _charges_input(WATER))
    assert "ferric cache" not in err


def test_an_unreadable_entry_is_recomputed(plugin_env, tmp_path):
    env = plugin_env(method="HF", basis="sto-3g")
    run_plugin(env, ["sp"], menu_input(WATER))
    entries = list((tmp_path / "cache").glob("*.npz"))
    assert len(entries) == 1
    entries[0].write_bytes(b"not an npz file")
    out, err = run_plugin(env, ["ferric_lowdin", "--charges"], _charges_input(WATER))
    assert "ignoring unreadable" in err
    assert "charges from the cache" not in err
    assert sum(out["charges"]) == pytest.approx(0.0, abs=1e-6)


def test_key_separates_every_input(monkeypatch, tmp_path):
    """In-process: the key is stable for one job and changes with each input."""
    monkeypatch.setenv("AVOGADRO_FERRIC_CACHE", str(tmp_path))
    from avogadro_ferric import cache
    from avogadro_ferric.engine import Job

    base = dict(
        numbers=[8, 1, 1],
        coords_angstrom=np.array([a[1:] for a in WATER], dtype=float),
        charge=0,
        multiplicity=1,
        method="HF",
        basis="sto-3g",
    )
    k0 = cache.key(Job(**base))
    assert cache.key(Job(**base)) == k0
    variants = [
        {"numbers": [8, 1, 9]},
        {"coords_angstrom": base["coords_angstrom"] + 1e-12},
        {"charge": 2, "multiplicity": 1},
        {"multiplicity": 3},
        {"method": "PBE"},
        {"basis": "6-31g"},
    ]
    keys = {cache.key(Job(**{**base, **v})) for v in variants}
    assert k0 not in keys and len(keys) == len(variants)
    # Spelling of method/basis is not a different calculation.
    assert cache.key(Job(**{**base, "method": "hf", "basis": "STO-3G"})) == k0
