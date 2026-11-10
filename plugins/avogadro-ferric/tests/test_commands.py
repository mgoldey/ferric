"""Menu commands, electrostatic models and Configure..., driven as Avogadro does."""

from __future__ import annotations

import json

import numpy as np
import pytest
from avoferric_testlib import H3O, OH, WATER, menu_input, run_plugin

import ferric


def _xyz(atoms):
    sym = {1: "H", 6: "C", 7: "N", 8: "O"}
    return f"{len(atoms)}\n\n" + "".join(
        f"{sym[a[0]]} {a[1]} {a[2]} {a[3]}\n" for a in atoms
    )


def _cj_input(atoms, charge=0, mult=1):
    payload = menu_input(atoms, charge, mult)
    # electrostatic models get {"cjson": <object>} (input-cjson = true), no charge/spin keys
    return {"cjson": payload["cjson"]}


# ---------------------------------------------------------------- charges / ESP


@pytest.mark.parametrize(
    "ident", ["ferric_lowdin", "ferric_hirshfeld", "ferric_chelpg"]
)
@pytest.mark.parametrize("atoms,charge", [(WATER, 0), (H3O, 1)])
def test_charges_sum_to_the_total_charge(plugin_env, ident, atoms, charge):
    env = plugin_env(method="HF", basis="sto-3g")
    out, _ = run_plugin(env, [ident, "--charges"], _cj_input(atoms, charge))
    q = out["charges"]
    assert len(q) == len(atoms)
    assert sum(q) == pytest.approx(charge, abs=1e-4)
    # O is the negative end in both.
    assert q[0] < min(q[1:])


def test_lowdin_charges_match_ferric(plugin_env):
    env = plugin_env(method="PBE", basis="sto-3g")
    out, _ = run_plugin(env, ["ferric_lowdin", "--charges"], _cj_input(WATER))
    mol = ferric.Molecule.from_xyz_string(_xyz(WATER))
    bs = ferric.BasisSet.bundled("sto-3g")
    from avogadro_ferric import engine

    # The plugin runs Kohn-Sham with f64 RI-J storage (engine._with_f64_rij).
    res = engine._with_f64_rij(ferric.run_dft, mol, bs, functional="PBE")
    ref = ferric.lowdin_charges(mol, bs, res)
    assert out["charges"] == pytest.approx(ref, abs=1e-8)
    # ... which is not ferric's default single-point surface (f32 RI-J), but
    # the charges barely notice: the f32/f64 choice matters for gradients.
    plain = ferric.lowdin_charges(mol, bs, ferric.run_dft(mol, bs, functional="PBE"))
    assert out["charges"] == pytest.approx(plain, abs=1e-5)


def test_potential_units_are_e_per_angstrom(plugin_env):
    """Far from H3O+, the ESP is the monopole term q/r with r in Angstrom.

    Avogadro's own fallback potential is sum_i q_i / r_i (Angstrom), so the
    plugin must convert ferric's Hartree/e (= e/Bohr) by x 1.8897 (Bohr per
    Angstrom). The 1/r^2 dipole term is ~1e-4 here; a missing or inverted
    conversion is off by 1.89x or 3.57x.
    """
    env = plugin_env(method="HF", basis="sto-3g")
    payload = _cj_input(H3O, 1)
    r = 30.0
    payload["points"] = [[r, 0.0, 0.0], [0.0, -r, 0.0], [0.0, 0.0, r]]
    out, _ = run_plugin(env, ["ferric_hirshfeld", "--potentials"], payload)
    phi = np.asarray(out["potentials"])
    assert phi == pytest.approx(1.0 / r, rel=2e-2)


def test_open_shell_charges_are_refused_with_a_message(plugin_env):
    env = plugin_env(method="HF", basis="sto-3g")
    payload = _cj_input(OH, 0, 2)
    out, _ = run_plugin(env, ["ferric_lowdin", "--charges"], payload)
    assert "charges" not in out and "closed-shell" in out["error"]


# ---------------------------------------------------------------- single point


def test_single_point_sets_energy_and_charges(plugin_env):
    env = plugin_env(method="HF", basis="sto-3g", charge_model="lowdin")
    out, _ = run_plugin(env, ["sp"], menu_input(WATER))
    cj = out["cjson"]
    ref = ferric.run_rhf(
        ferric.Molecule.from_xyz_string(_xyz(WATER)), ferric.BasisSet.bundled("sto-3g")
    )
    assert cj["properties"]["totalEnergy"] == pytest.approx(ref.energy, abs=1e-9)
    q = cj["atoms"]["partialCharges"]["ferric-lowdin"]
    assert sum(q) == pytest.approx(0.0, abs=1e-6)
    assert "Hartree" in out["message"]


def test_single_point_open_shell_reports_energy(plugin_env):
    env = plugin_env(method="HF", basis="sto-3g")
    out, _ = run_plugin(env, ["sp"], menu_input(OH, 0, 2))
    ref = ferric.run_uhf(
        ferric.Molecule.from_xyz_string(_xyz(OH), 0, 2),
        ferric.BasisSet.bundled("sto-3g"),
    )
    assert out["cjson"]["properties"]["totalEnergy"] == pytest.approx(
        ref.energy, abs=1e-7
    )


def test_inconsistent_charge_and_multiplicity_is_an_error(plugin_env):
    env = plugin_env(method="HF", basis="sto-3g")
    out, _ = run_plugin(env, ["sp"], menu_input(WATER, 0, 2))
    assert "multiplicity" in out["error"]


# ---------------------------------------------------------------- optimize


def test_optimize_returns_a_relaxed_cjson(plugin_env):
    env = plugin_env(method="HF", basis="sto-3g")
    start = [(8, 0.0, 0.0, 0.1), (1, 0.0, 0.85, -0.5), (1, 0.0, -0.70, -0.45)]
    payload = menu_input(start, bonds=[(0, 1), (0, 2)])
    payload["cjson"]["vibrations"] = {"frequencies": [1.0, 2.0, 3.0]}  # stale data
    out, _ = run_plugin(env, ["opt"], payload)
    cj = out["cjson"]
    assert out["moleculeFormat"] == "cjson"
    assert cj["atoms"]["elements"]["number"] == [8, 1, 1]
    assert cj["bonds"] == payload["cjson"]["bonds"]
    assert "vibrations" not in cj
    xyz = np.asarray(cj["atoms"]["coords"]["3d"]).reshape(-1, 3)
    assert xyz.shape == (3, 3)
    # The returned geometry (Angstrom) is a stationary point of the same surface.
    from avogadro_ferric import engine

    job = engine.Job([8, 1, 1], xyz, method="HF", basis="sto-3g")
    e, g = engine.energy_gradient(job)
    assert np.abs(g).max() < 1e-3  # Hartree/Bohr
    e0, _ = engine.energy_gradient(
        engine.Job([8, 1, 1], np.array([a[1:] for a in start]))
    )
    assert e < e0
    assert cj["properties"]["totalEnergy"] == pytest.approx(e, abs=1e-7)
    # Water/STO-3G RHF: r(OH) ~ 0.989 A.
    r = np.linalg.norm(xyz[1] - xyz[0])
    assert r == pytest.approx(0.989, abs=0.01)


# ---------------------------------------------------------------- frequencies

MASS = {1: 1.00794, 8: 15.9994}  # standard atomic weights


def test_frequencies_and_cartesian_modes(plugin_env):
    """Modes are ferric's frequencies, CARTESIAN (mass-unweighted) and unit-normalised.

    Mass-unweighted Cartesian modes keep the centre of mass fixed
    (sum_a m_a d_a = 0) and are orthogonal in the MASS metric; mass-weighted
    eigenvectors q would instead satisfy sum_a sqrt(m_a) q_a = 0 and plain
    orthogonality. Both distinctions are asserted.
    """
    env = plugin_env(method="HF", basis="sto-3g")
    from avogadro_ferric import engine

    job = engine.Job([8, 1, 1], np.array([a[1:] for a in WATER]))
    xyz, *_ = engine.optimize(job)
    atoms = [(z, *p) for z, p in zip([8, 1, 1], xyz)]
    out, _ = run_plugin(env, ["freq"], menu_input(atoms))
    vib = out["cjson"]["vibrations"]
    freqs = vib["frequencies"]
    ref = ferric.run_frequencies(ferric.Molecule.from_xyz_string(_xyz(atoms)), "sto-3g")
    assert freqs == pytest.approx(list(ref.frequencies), abs=1e-6)
    assert len(vib["eigenVectors"]) == 3
    m = np.repeat([MASS[z] for z in [8, 1, 1]], 3)
    d = np.asarray(vib["eigenVectors"])
    assert np.linalg.norm(d, axis=1) == pytest.approx(1.0, abs=1e-12)
    com = (d * m).reshape(3, 3, 3).sum(axis=1)  # mode, atom, xyz -> sum over atoms
    assert np.abs(com).max() < 1e-3
    gram_mass = (d * m) @ d.T
    off = gram_mass - np.diag(np.diag(gram_mass))
    assert np.abs(off).max() < 1e-3 * np.abs(np.diag(gram_mass)).max()
    # ... and NOT orthogonal in the plain metric (so the check above can fail).
    plain = d @ d.T
    assert np.abs(plain - np.eye(3)).max() > 1e-2
    # Independent check: each vector is an eigenvector of the mass-weighted
    # Hessian once mass-weighted (H_mw (sqrt(m) d) = lambda (sqrt(m) d)).
    h = np.asarray(ref.cartesian_hessian)
    hmw = h / np.sqrt(np.outer(m, m))
    for v in d:
        q = np.sqrt(m) * v
        q /= np.linalg.norm(q)
        lam = q @ hmw @ q
        assert np.linalg.norm(hmw @ q - lam * q) < 1e-4 * abs(lam)


def test_imaginary_frequency_is_reported(plugin_env):
    env = plugin_env(method="HF", basis="sto-3g")
    linear = [(8, 0.0, 0.0, 0.0), (1, 0.0, 0.95, 0.0), (1, 0.0, -0.95, 0.0)]
    out, _ = run_plugin(env, ["freq"], menu_input(linear))
    assert min(out["cjson"]["vibrations"]["frequencies"]) < 0
    assert "imaginary" in out["message"]


# ---------------------------------------------------------------- configure


def test_configure_dialog_round_trip(plugin_env, tmp_path):
    env = plugin_env(method="HF", basis="sto-3g")
    out, _ = run_plugin(env, ["config", "--user-options"], {})
    opts = out["userOptions"]
    assert all("type" in v for v in opts.values())  # Avogadro needs a type per option
    assert opts["method"]["values"][opts["method"]["default"]] == "HF"
    # Avogadro sends the chosen stringList entry as the string.
    choice = {
        "method": "B3LYP",
        "basis": "def2-svp",
        "charge_model": "chelpg",
        "threads": 2,
    }
    choice["memory_gb"] = 1.5
    out, _ = run_plugin(
        env, ["config"], {"options": choice, "cjson": menu_input(WATER)["cjson"]}
    )
    assert "cjson" not in out  # leave the molecule alone
    stored = json.loads(open(env["AVOGADRO_FERRIC_CONFIG"], encoding="utf-8").read())
    assert stored["method"] == "B3LYP" and stored["basis"] == "def2-svp"
    assert stored["threads"] == 2 and stored["memory_gb"] == 1.5
    out, _ = run_plugin(env, ["config", "--user-options"], {})
    assert out["userOptions"]["charge_model"]["default"] == 2


def test_unknown_method_is_refused(plugin_env):
    env = plugin_env()
    out, _ = run_plugin(env, ["config"], {"options": {"method": "MP2"}})
    assert "unknown method" in out["error"]


# ---------------------------------------------------------------- per-run dialog


def test_job_commands_ask_for_method_and_basis(plugin_env):
    """Single Point / Optimize / Frequencies / Orbitals open a method + basis
    dialog (user-options = "dynamic"), pre-filled from the Configure defaults."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")

    from avoferric_testlib import PLUGIN_DIR

    meta = tomllib.loads((PLUGIN_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    cmds = {c["identifier"]: c for c in meta["tool"]["avogadro"]["menu-commands"]}
    for ident in ("sp", "opt", "freq", "orbitals"):
        assert cmds[ident].get("user-options") == "dynamic", ident
    env = plugin_env(method="B3LYP", basis="cc-pvdz")
    for ident in ("sp", "opt", "freq", "orbitals"):
        out, _ = run_plugin(env, [ident, "--user-options"], {})
        opts = out["userOptions"]
        assert set(opts) == {"method", "basis"}, ident
        assert all("type" in v for v in opts.values())
        assert opts["method"]["values"][opts["method"]["default"]] == "B3LYP"
        assert opts["basis"]["values"][opts["basis"]["default"]] == "cc-pvdz"


def test_dialog_choice_overrides_the_default_for_one_run(plugin_env):
    env = plugin_env(method="PBE", basis="def2-svp")
    payload = menu_input(WATER, options={"method": "HF", "basis": "sto-3g"})
    out, _ = run_plugin(env, ["sp"], payload)
    assert "RHF/sto-3g" in out["message"], out["message"]
    # The stored default is untouched.
    stored = json.loads(open(env["AVOGADRO_FERRIC_CONFIG"], encoding="utf-8").read())
    assert stored["method"] == "PBE" and stored["basis"] == "def2-svp"
