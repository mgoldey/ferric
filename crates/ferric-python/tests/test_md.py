"""`run_aimd` / `run_ffmd`: the Python molecular dynamics entry points.

The integrators themselves are tested in `crates/ferric-md/tests` and the
force providers in `crates/ferric-scf/tests/md_aimd.rs` and
`crates/ferric-mm/tests/md_ffmd.rs`; these tests pin what the binding adds:
the kwargs reach the driver, the arrays have the documented shapes and units,
bad input raises `ValueError` before any SCF, and a seed reproduces a run.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

H2 = "2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7414\n"
TIGHT = {"scf_density_conv": 1e-10, "scf_energy_conv": 1e-10}

#: Angstrom per Bohr and fs per atomic time unit (CODATA 2018), for the unit
#: checks below.
BOHR_ANGSTROM = 0.529177210903
AU_TIME_FS = 2.4188843265857e-2


def h2() -> ferric.Molecule:
    return ferric.Molecule.from_xyz_string(H2)


def test_aimd_h2_nve_arrays_and_conservation():
    r = ferric.run_aimd(
        h2(), "sto-3g", 20, dt_fs=0.2, init_temperature_k=300.0, **TIGHT
    )
    n = 21
    for a in (
        r.times_fs,
        r.potential_energies,
        r.kinetic_energies,
        r.total_energies,
        r.conserved,
        r.temperatures,
    ):
        assert a.shape == (n,)
    assert r.steps == list(range(n))
    np.testing.assert_allclose(r.times_fs, 0.2 * np.arange(n), atol=1e-12)
    assert r.positions_angstrom.shape == (n, 2, 3)
    assert r.velocities.shape == (n, 2, 3)
    assert r.n_dof == 3  # 3N - 3: COM motion removed
    assert r.n_force_evaluations == n
    # The draw is rescaled to the requested temperature exactly.
    assert r.temperatures[0] == pytest.approx(300.0, abs=1e-9)
    # NVE: nothing is injected, so conserved == total == pot + kin.
    np.testing.assert_allclose(
        r.total_energies, r.potential_energies + r.kinetic_energies, atol=1e-12
    )
    np.testing.assert_array_equal(r.conserved, r.total_energies)
    # Measured 8.4e-6 Hartree (the velocity Verlet error at 0.2 fs; the
    # potential energy swings by about 1e-3).
    assert r.max_conserved_drift < 2e-5
    assert np.ptp(r.potential_energies) > 10 * r.max_conserved_drift
    assert r.max_conserved_drift == pytest.approx(
        np.abs(r.conserved - r.conserved[0]).max()
    )
    # Step 0 is the input geometry, in Angstrom.
    np.testing.assert_allclose(r.positions_angstrom[0, :, 2], [0.0, 0.7414], atol=1e-10)
    # Kinetic energy from the reported velocities (Angstrom/fs) and the
    # hydrogen mass reproduces the reported kinetic energy (Hartree).
    m_h = 1.00794 / 5.48579909065e-4  # amu -> electron masses
    v_au = r.velocities[0] / (BOHR_ANGSTROM / AU_TIME_FS)
    assert 0.5 * m_h * (v_au**2).sum() == pytest.approx(r.kinetic_energies[0], rel=1e-4)
    # Final frame and returned molecule agree.
    assert r.mol() is not None
    assert len(r.scf_iterations) == n
    assert r.label == "rhf/sto-3g"


def test_aimd_certificate_is_ungraded_and_records_the_settings():
    r = ferric.run_aimd(
        h2(),
        "sto-3g",
        2,
        thermostat="langevin",
        friction_per_ps=20.0,
        temperature_k=300.0,
        seed=3,
        **TIGHT,
    )
    c = r.certificate
    assert c["capability"] == "md:rhf/sto-3g"
    assert c["grade"] == "ungraded"
    approx = {a["name"]: a["detail"] for a in c["approximations"]}
    assert approx["integrator"] == "Langevin BAOAB"
    assert "friction = 20 ps^-1" in approx["thermostat"]
    assert approx["seed"] == "3"
    assert "dt = 0.5 fs" in approx["time-step"]
    # Langevin counts 3N degrees of freedom: its noise does not conserve momentum.
    assert r.n_dof == 6


def test_aimd_seed_reproduces_the_trajectory():
    kw = dict(thermostat="langevin", friction_per_ps=50.0, temperature_k=500.0, **TIGHT)
    a = ferric.run_aimd(h2(), "sto-3g", 5, seed=11, **kw)
    b = ferric.run_aimd(h2(), "sto-3g", 5, seed=11, **kw)
    c = ferric.run_aimd(h2(), "sto-3g", 5, seed=12, **kw)
    np.testing.assert_array_equal(a.positions_angstrom, b.positions_angstrom)
    np.testing.assert_array_equal(a.conserved, b.conserved)
    assert not np.array_equal(a.positions_angstrom, c.positions_angstrom)


def test_aimd_without_a_temperature_starts_at_rest():
    r = ferric.run_aimd(h2(), "sto-3g", 1, **TIGHT)
    assert r.kinetic_energies[0] == 0.0
    assert r.temperatures[0] == 0.0


def test_aimd_writes_trajectory_and_energies(tmp_path):
    xyz = tmp_path / "t.xyz"
    csv = tmp_path / "e.csv"
    r = ferric.run_aimd(
        h2(),
        "sto-3g",
        4,
        stride=2,
        init_temperature_k=300.0,
        trajectory_path=str(xyz),
        energies_path=str(csv),
        **TIGHT,
    )
    assert r.steps == [0, 2, 4]
    lines = xyz.read_text().splitlines()
    assert len(lines) == 3 * 4
    assert lines[1].startswith("step=0 ") and lines[5].startswith("step=2 ")
    rows = csv.read_text().splitlines()
    assert rows[0] == "step,time_fs,e_pot,e_kin,e_tot,conserved,temperature_k"
    assert [int(row.split(",")[0]) for row in rows[1:]] == [0, 2, 4]
    e_pot = [float(row.split(",")[2]) for row in rows[1:]]
    np.testing.assert_allclose(e_pot, r.potential_energies, atol=1e-11)


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"thermostat": "berendsen", "temperature_k": 300.0}, "requires tau_fs"),
        (
            {"thermostat": "langevin", "temperature_k": 300.0},
            "requires friction_per_ps",
        ),
        ({"tau_fs": 100.0}, "tau_fs applies only to berendsen"),
        (
            {
                "thermostat": "berendsen",
                "tau_fs": 100.0,
                "friction_per_ps": 1.0,
                "temperature_k": 300.0,
            },
            "friction_per_ps applies only to langevin",
        ),
        ({"thermostat": "langevin", "friction_per_ps": 10.0}, "requires temperature_k"),
        ({"thermostat": "nose-hoover"}, "unknown thermostat"),
        ({"dt_fs": -0.5}, "dt_fs must be finite and positive"),
        ({"stride": 0}, "stride must be at least 1"),
        ({"reference": "ghf"}, "unknown reference"),
        ({"dispersion": "d3bj"}, "dispersion= requires xc="),
        ({"masses_amu": [1.0]}, "masses"),
    ],
)
def test_aimd_bad_kwargs_raise_value_error(kwargs, needle):
    with pytest.raises(ValueError, match=needle):
        ferric.run_aimd(h2(), "sto-3g", 2, **kwargs)


def test_aimd_ghost_atoms_are_refused():
    mol = ferric.Molecule.from_arrays(
        ["H", "H", "He"],
        [[0, 0, 0], [0, 0, 0.74], [0, 0, 3.0]],
        ghost=[False, False, True],
    )
    with pytest.raises(ValueError, match="ghost"):
        ferric.run_aimd(mol, "sto-3g", 1)


# ── run_ffmd ──

#: A flexible TIP3P-like water dimer: charges and LJ of TIP3P, bonds and angle
#: of a flexible SPC-style model (AMBER units: kcal/mol, Angstrom, degrees).
DIMER_SYMBOLS = ["O", "H", "H", "O", "H", "H"]
DIMER_XYZ = [
    [0.0, 0.0, 0.0],
    [0.9572, 0.0, 0.0],
    [-0.2400, 0.9266, 0.0],
    [2.90, 0.0, 0.0],
    [3.20, 0.90, 0.0],
    [3.20, -0.45, 0.78],
]


def dimer_topology() -> ferric.MmTopology:
    bonds, angles = [], []
    for o in (0, 3):
        bonds += [(o, o + 1, 450.0, 0.9572), (o, o + 2, 450.0, 0.9572)]
        angles.append((o + 1, o, o + 2, 55.0, 104.52))
    return ferric.MmTopology.from_amber_units(
        charges=[-0.834, 0.417, 0.417] * 2,
        sigmas_angstrom=[3.15061, 1.0, 1.0] * 2,
        epsilons_kcal=[0.1521, 0.0, 0.0] * 2,
        bonds=bonds,
        angles=angles,
        torsions=[],
    )


def test_ffmd_water_dimer_nve():
    r = ferric.run_ffmd(
        dimer_topology(),
        DIMER_SYMBOLS,
        np.array(DIMER_XYZ),
        200,
        dt_fs=0.25,
        init_temperature_k=300.0,
    )
    assert r.potential_energies.shape == (201,)
    assert r.positions_angstrom.shape == (201, 6, 3)
    assert r.n_dof == 15
    assert r.temperatures[0] == pytest.approx(300.0, abs=1e-9)
    np.testing.assert_allclose(r.positions_angstrom[0], DIMER_XYZ, atol=1e-12)
    assert r.max_conserved_drift < 1e-2 * np.ptp(r.potential_energies)
    assert r.mol() is None
    assert r.scf_iterations is None
    assert r.label == "mm"
    assert r.certificate["capability"] == "md:mm"


def test_ffmd_cutoff_changes_the_label_and_the_surface():
    top = dimer_topology()
    plain = ferric.run_ffmd(top, DIMER_SYMBOLS, DIMER_XYZ, 1)
    cut = ferric.run_ffmd(top, DIMER_SYMBOLS, DIMER_XYZ, 1, cutoff_angstrom=9.0)
    assert cut.label == "mm:cutoff"
    assert cut.potential_energies[0] != plain.potential_energies[0]


def test_ffmd_seed_reproduces_the_trajectory():
    kw = dict(thermostat="langevin", friction_per_ps=5.0, temperature_k=300.0)
    top = dimer_topology()
    a = ferric.run_ffmd(top, DIMER_SYMBOLS, DIMER_XYZ, 20, seed=5, **kw)
    b = ferric.run_ffmd(top, DIMER_SYMBOLS, DIMER_XYZ, 20, seed=5, **kw)
    np.testing.assert_array_equal(a.positions_angstrom, b.positions_angstrom)


@pytest.mark.parametrize(
    "args, kwargs, needle",
    [
        ((DIMER_SYMBOLS[:5], DIMER_XYZ), {}, "the topology has 6 atoms"),
        ((["O", "H", "H", "O", "H", "Xx"], DIMER_XYZ), {}, "no standard atomic mass"),
        ((DIMER_SYMBOLS, DIMER_XYZ), {"cutoff_angstrom": -1.0}, "cutoff_angstrom"),
        (
            (DIMER_SYMBOLS, DIMER_XYZ),
            {"thermostat": "berendsen", "temperature_k": 300.0},
            "requires tau_fs",
        ),
        ((DIMER_SYMBOLS, [row[:2] for row in DIMER_XYZ]), {}, "expected 3"),
    ],
)
def test_ffmd_bad_input_raises_value_error(args, kwargs, needle):
    with pytest.raises(ValueError, match=needle):
        ferric.run_ffmd(dimer_topology(), *args, 2, **kwargs)
