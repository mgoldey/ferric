"""The energy model through Avogadro's text protocol (scriptenergy.cpp, 2.1.0).

The finite-difference test is the unit-conversion check: the plugin reports the
energy in kJ/mol (Hartree x 2625.4996) and the gradient in kJ/mol/Angstrom
(Hartree/Bohr x 2625.4996 / 0.52917721092). A central difference of the
REPORTED energies over a step in Angstrom must reproduce the REPORTED gradient,
so a wrong length factor, a missing factor or a sign flip fails it. The
absolute-energy test pins the energy factor itself against an independent
run_rhf.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

import numpy as np
import pytest
from avoferric_testlib import OH, WATER, make_cjson

import ferric

KJ_PER_HARTREE = 4.35974434e-18 * 6.02214129e23 / 1000.0  # CODATA 2010, ~2625.4996


class EnergyClient:
    """Plays Avogadro's side of the text protocol."""

    def __init__(self, env, atoms, charge=0, mult=1):
        self.natoms = len(atoms)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "avogadro_ferric", "ferric", "--lang", "en_US"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        bootstrap = {
            "charge": charge,
            "spin": mult,
            "selectedAtoms": [],
            "cjson": make_cjson(atoms, charge, mult),
            "options": {},
        }
        self.proc.stdin.write(json.dumps(bootstrap) + "\n")
        self.proc.stdin.flush()

    def evaluate(self, coords):
        for x, y, z in np.asarray(coords, dtype=float).reshape(-1, 3):
            # Avogadro writes QByteArray::number(v, 'g', 17) (scriptenergy.cpp)
            self.proc.stdin.write(
                " ".join(format(float(v), ".17g") for v in (x, y, z)) + "\n"
            )
        self.proc.stdin.flush()
        first = self.proc.stdout.readline()
        if not first:
            raise RuntimeError(self.proc.stderr.read())
        assert first.startswith("AvogadroEnergy:"), first
        energy = float(first.split(":", 1)[1])
        assert self.proc.stdout.readline().strip() == "AvogadroGradient:"
        grad = [
            [float(v) for v in self.proc.stdout.readline().split()]
            for _ in range(self.natoms)
        ]
        return energy, np.asarray(grad)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=60)
        return self.proc.returncode


def coords_of(atoms):
    return np.array([a[1:] for a in atoms], dtype=float)


def fd_check(client, x0, components, h=1e-3):
    e0, g0 = client.evaluate(x0)
    for atom, axis in components:
        xp = x0.copy()
        xp[atom, axis] += h
        xm = x0.copy()
        xm[atom, axis] -= h
        ep, _ = client.evaluate(xp)
        em, _ = client.evaluate(xm)
        fd = (ep - em) / (2 * h)
        an = g0[atom, axis]
        # Analytic gradient vs central difference: the h^2 term is ~1e-6 of g
        # at h = 1e-3 Angstrom; SCF noise at density_conv 1e-6 is far below.
        assert abs(fd - an) < 2e-3 * max(1.0, abs(an)), (atom, axis, fd, an)
    return e0, g0


@pytest.mark.parametrize(
    "method,basis,atoms,mult",
    [
        ("HF", "sto-3g", WATER, 1),  # RHF
        ("PBE", "sto-3g", WATER, 1),  # RKS
        ("HF", "sto-3g", OH, 2),  # UHF
    ],
)
def test_reported_gradient_is_the_derivative_of_the_reported_energy(
    plugin_env, method, basis, atoms, mult
):
    env = plugin_env(method=method, basis=basis)
    client = EnergyClient(env, atoms, mult=mult)
    try:
        x0 = coords_of(atoms)
        # Distort so the gradient is far from zero in every checked component.
        x0[0] += [0.02, -0.03, 0.05]
        components = [(0, 0), (0, 2), (1, 1), (len(atoms) - 1, 2)]
        _, g0 = fd_check(client, x0, components)
        assert np.abs(g0).max() > 1.0, "gradient too small to test the conversion"
        # Translational invariance of an exact gradient.
        assert np.abs(g0.sum(axis=0)).max() < 1e-4 * np.abs(g0).max()
    finally:
        assert client.close() == 0


def test_energy_is_hartree_times_kj_per_mol(plugin_env):
    env = plugin_env(method="HF", basis="sto-3g")
    client = EnergyClient(env, WATER)
    try:
        e_kj, g_kj = client.evaluate(coords_of(WATER))
    finally:
        client.close()
    xyz = "3\n\n" + "".join(
        f"{s} {x} {y} {z}\n" for s, (_, x, y, z) in zip(["O", "H", "H"], WATER)
    )
    ref = ferric.run_rhf(
        ferric.Molecule.from_xyz_string(xyz), ferric.BasisSet.bundled("sto-3g")
    )
    assert e_kj == pytest.approx(ref.energy * KJ_PER_HARTREE, rel=1e-10)
    assert abs(KJ_PER_HARTREE - 2625.4996) < 1e-3


def test_repeated_geometry_is_answered_from_the_cache(plugin_env):
    """Avogadro's evaluate() sends the same geometry twice (gradient, then value)."""
    env = plugin_env(method="HF", basis="sto-3g")
    client = EnergyClient(env, WATER)
    try:
        x = coords_of(WATER)
        a = client.evaluate(x)
        b = client.evaluate(x)
    finally:
        client.close()
    assert a[0] == b[0]
    assert np.array_equal(a[1], b[1])


def test_a_failed_evaluation_reports_nan_and_keeps_serving(plugin_env):
    env = plugin_env(method="HF", basis="sto-3g")
    client = EnergyClient(env, WATER)
    try:
        bad = coords_of(WATER)
        bad[1] = bad[0]  # two nuclei on top of each other
        e_bad, _ = client.evaluate(bad)
        e_ok, _ = client.evaluate(coords_of(WATER))
    finally:
        client.close()
    # The bad geometry got an answer (NaN when ferric raised), and the server
    # is still in step: the next reply belongs to the next geometry.
    assert not np.isfinite(e_bad) or abs(e_bad) > 1e6
    assert np.isfinite(e_ok)


def test_reply_format_parses_like_avogadro():
    """format_reply's text matches what scriptenergy.cpp's parser expects."""
    from avogadro_ferric import energy_model

    # numpy scalars as well as Python floats: under numpy 2, repr(np.float64)
    # is "np.float64(...)", which the C++ parser reads as no number at all
    # (seen in the Flatpak smoke run, whose pixi env has numpy 2.5).
    energy = np.float64(-1.5e5) + np.float64(0.1)
    grad = np.array([[1.0, -2.0, 3.5], [0.25, 0.0, -1e-9]])
    text = energy_model.format_reply(energy, grad)
    lines = text.splitlines()
    assert lines[0].startswith("AvogadroEnergy: ")
    number = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
    token = lines[0].split(":", 1)[1].strip()
    assert number.match(token), token
    assert float(token) == float(energy)  # round-trips exactly
    assert lines[1] == "AvogadroGradient:"
    for line, row in zip(lines[2:], grad):
        toks = line.split()
        assert all(number.match(t) for t in toks), line
        assert [float(t) for t in toks] == list(row)
    assert text.endswith("\n")


def test_unit_factors():
    from avogadro_ferric import energy_model, units

    e, g = energy_model.to_avogadro_units(1.0, np.array([[1.0, 0.0, 0.0]]))
    assert e == pytest.approx(2625.4996, abs=1e-3)
    # 1 Hartree/Bohr = 2625.4996 / 0.52917721092 kJ/mol/Angstrom = 4961.475...
    assert g[0, 0] == pytest.approx(2625.4996 / 0.52917721092, rel=1e-7)
    assert units.ANGSTROM_TO_BOHR * units.BOHR_TO_ANGSTROM == pytest.approx(1.0)
