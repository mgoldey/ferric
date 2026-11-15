"""`run_optimize` beyond RHF: the routes, and the kwargs each route refuses.

Every route calls the library driver the CLI's `task = "optimize"` calls for
the same `method.kind` (`optimize_geometry{,_uhf,_rohf}_with_scf_correction`,
`ferric_mp2::optimize::optimize_geometry_rimp2_on`,
`ferric_mp2::optimize::optimize_geometry_scs_mp2_on`,
`ferric_mp2::optimize::optimize_geometry_oo_rimp2`), so on the same molecule
and settings the two surfaces run the same arithmetic. The CLI prints the
final energy to 10 decimals and the geometry to 8 (Bohr); the bar here is
that the Python result, formatted the same way, prints identically, and that
the step count and convergence flag agree. A route on a different driver or a
different SCF config (grid, J/K route, convergence) moves a multi-step BFGS
trajectory far more than the last printed digit.

The CLI comparisons need the `ferric-cli` binary of the same tree:
`FERRIC_CLI`, else `target/release/ferric-cli` (skipped if absent).

Mutations measured to fail (each on an assertion, then restored):
- `run_optimize_scf` routing `Spin::Unrestricted` to the ROHF driver fails
  `test_matches_the_cli[uks-pbe-nh2]` (final E -55.05795 vs -55.05862);
- `run_optimize_rimp2` dropping the RI-JK route (always `exact_scf_jk()`)
  fails `test_matches_the_cli[rimp2-rijk-water]` (4th decimal);
- `run_optimize_oo_rimp2` ignoring `oo_grad_conv` fails
  `test_matches_the_cli[oo-rimp2-water]`, but only in the 10th printed
  decimal (-75.0062761163 vs ...154): a thin margin;
- `run_optimize_oo_rimp2` dropping the RI-JK route (never applying
  `scf_jk`) fails `test_matches_the_cli[oo-rimp2-rijk-water]` (final E
  -75.0062761154 vs -75.0063668362);
- `oo_ri_mp2_gradient` building its J/K four-centre on an RI-JK reference
  fails `test_oo_rimp2_rijk_endpoint_is_stationary_on_the_fitted_surface`
  (analytic vs fitted FD 1.09e-5, bar 1e-6);
- `run_optimize_scs_mp2` ignoring `c_os` / `c_ss` (always 6/5, 1/3) fails
  `test_matches_the_cli[scs-mp2-sos-water]` (final E -75.01214 vs -75.00345);
- `parse_optimize_reference` mapping unset to ROHF on an open shell fails
  `test_open_shell_defaults_to_uhf`;
- `optimize_config` ignoring `coordinates` (always Cartesian) fails all three
  `test_internal_coordinates_match_the_cli` cases (steps 7 vs 5, 8 vs 5,
  7 vs 6) and `test_certificate_names_the_coordinates`;
  `test_run_optimize_uhf_passes_coordinates` cannot see it (both sides
  drop the kwarg alike) and guards only the wrapper's pass-through;
- `refuse_unread_optimize_kwargs` returning early fails the nine cases it
  owns (two of them, "needs auxbasis" and "closed-shell only", then fail on
  the downstream RuntimeError rather than the ValueError match); the other
  cases are refused by the method/reference parsers or the J/K resolver.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

ROOT = pathlib.Path(__file__).resolve().parents[3]

WATER = "3\ndistorted water\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n"
NH2 = "3\nNH2 radical\nN 0 0 0\nH 0 0.95 0.2\nH 0 -0.6 0.85\n"
SCF_CONV = {"scf_energy_conv": 1e-8, "scf_density_conv": 1e-8}
SCF_TOML = "[scf]\nenergy_conv = 1e-8\ndensity_conv = 1e-8\n"
JKFIT = "def2-universal-jkfit"


def _cli_binary():
    env = os.environ.get("FERRIC_CLI")
    if env:
        assert pathlib.Path(env).is_file(), f"FERRIC_CLI={env} does not exist"
        return pathlib.Path(env)
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def _run_cli(tmp_path, xyz, mult, body):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    geom = tmp_path / "mol.xyz"
    geom.write_text(xyz)
    toml = tmp_path / "opt.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{geom}"\nmultiplicity = {mult}\n\n'
        f'[basis]\nname = "sto-3g"\n\n{body}'
    )
    env = dict(os.environ)
    # Same rayon width as this process, so both run the same reduction order.
    env.setdefault("RAYON_NUM_THREADS", str(len(os.sched_getaffinity(0))))
    out = subprocess.run(
        [str(cli), str(toml), "--json", str(tmp_path / "run.jsonl")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def _cli_result(stdout):
    """(final-energy text, geometry lines, steps, converged) from the CLI."""
    energy = re.search(r"final E    = (\S+) Hartree", stdout).group(1)
    block = stdout.split("Final Optimized Geometry (Bohr):\n")[1]
    geom = []
    for line in block.splitlines():
        if not line.strip():
            break
        geom.append(line)
    steps = int(re.search(r"steps      = (\d+)", stdout).group(1))
    converged = re.search(r"converged  = (\w+)", stdout).group(1) == "true"
    return energy, geom, steps, converged


def _py_result(r):
    mol = r.mol()
    geom = [
        f"  {i:2} {s:2} {x:12.8f} {y:12.8f} {z:12.8f}"
        for i, (s, (x, y, z)) in enumerate(zip(mol.symbols(), mol.coords_bohr()))
    ]
    return f"{r.energy:.10f}", geom, r.steps, r.converged


# (id, xyz, multiplicity, run_optimize kwargs, CLI TOML after [basis]).
ROUTES = [
    pytest.param(
        WATER,
        1,
        dict(xc="B3LYP", df_j_aux=JKFIT, df_k_aux=JKFIT, **SCF_CONV),
        f'[method]\nkind = "ksdft"\ntask = "optimize"\n\n'
        f'[dft]\nfunctional = "B3LYP"\n\n{SCF_TOML}',
        id="rks-b3lyp-water",
    ),
    pytest.param(
        NH2,
        2,
        dict(xc="PBE", df_j_aux=JKFIT, df_k_aux=JKFIT, **SCF_CONV),
        f'[method]\nkind = "uhf"\ntask = "optimize"\n\n'
        f'[dft]\nfunctional = "PBE"\n\n{SCF_TOML}',
        id="uks-pbe-nh2",
    ),
    pytest.param(
        NH2,
        2,
        dict(reference="rohf", xc="PBE", df_j_aux=JKFIT, df_k_aux=JKFIT, **SCF_CONV),
        f'[method]\nkind = "rohf"\ntask = "optimize"\n\n'
        f'[dft]\nfunctional = "PBE"\n\n{SCF_TOML}',
        id="roks-pbe-nh2",
    ),
    pytest.param(
        WATER,
        1,
        dict(method="rimp2", auxbasis="cc-pvdz-ri"),
        '[method]\nkind = "rimp2"\ntask = "optimize"\n\n'
        '[mp2]\nauxbasis = "cc-pvdz-ri"\n',
        id="rimp2-water",
    ),
    pytest.param(
        WATER,
        1,
        dict(method="rimp2", auxbasis="cc-pvdz-ri", scf_jk="rijk"),
        '[method]\nkind = "rimp2"\ntask = "optimize"\n\n'
        '[mp2]\nauxbasis = "cc-pvdz-ri"\n\n[scf]\njk = "rijk"\n',
        id="rimp2-rijk-water",
    ),
    pytest.param(
        WATER,
        1,
        dict(method="oo-rimp2", auxbasis="cc-pvdz-ri", oo_grad_conv=1e-7, **SCF_CONV),
        '[method]\nkind = "oo-rimp2"\ntask = "optimize"\n\n'
        f'[mp2]\nauxbasis = "cc-pvdz-ri"\noo_grad_conv = 1e-7\n\n{SCF_TOML}',
        id="oo-rimp2-water",
    ),
    pytest.param(
        WATER,
        1,
        dict(
            method="oo-rimp2",
            auxbasis="cc-pvdz-ri",
            scf_jk="rijk",
            oo_grad_conv=1e-7,
            **SCF_CONV,
        ),
        '[method]\nkind = "oo-rimp2"\ntask = "optimize"\n\n'
        f'[mp2]\nauxbasis = "cc-pvdz-ri"\noo_grad_conv = 1e-7\n\n{SCF_TOML}jk = "rijk"\n',
        id="oo-rimp2-rijk-water",
    ),
    pytest.param(
        WATER,
        1,
        dict(method="scs-mp2", auxbasis="cc-pvdz-ri", c_os=1.0, c_ss=0.0),
        '[method]\nkind = "scs-mp2"\ntask = "optimize"\n\n'
        '[mp2]\nauxbasis = "cc-pvdz-ri"\nc_os = 1.0\nc_ss = 0.0\n',
        id="scs-mp2-sos-water",
    ),
    pytest.param(
        WATER,
        1,
        dict(method="scs-mp2", auxbasis="cc-pvdz-ri", scf_jk="rijk"),
        '[method]\nkind = "scs-mp2"\ntask = "optimize"\n\n'
        '[mp2]\nauxbasis = "cc-pvdz-ri"\n\n[scf]\njk = "rijk"\n',
        id="scs-mp2-rijk-water",
    ),
]


@pytest.mark.parametrize("xyz,mult,kwargs,toml", ROUTES)
def test_matches_the_cli(tmp_path, xyz, mult, kwargs, toml):
    cli = _cli_result(_run_cli(tmp_path, xyz, mult, toml))
    mol = ferric.Molecule.from_xyz_string(xyz, 0, mult)
    py = ferric.run_optimize(mol, "sto-3g", **kwargs)
    assert py.converged
    assert _py_result(py) == cli
    assert len(py.energy_trace) >= py.steps + 1
    assert py.energy_trace[-1] == py.energy


def test_certificates_name_the_route():
    water = ferric.Molecule.from_xyz_string(WATER)
    nh2 = ferric.Molecule.from_xyz_string(NH2, 0, 2)
    cases = [
        (water, dict(), "optimize:rhf"),
        (water, dict(xc="PBE"), "optimize:ksdft"),
        (nh2, dict(reference="rohf"), "optimize:rohf"),
        (water, dict(method="oo-rimp2", auxbasis="cc-pvdz-ri"), "optimize:oo-rimp2"),
        (water, dict(method="scs-mp2", auxbasis="cc-pvdz-ri"), "optimize:scs-mp2"),
    ]
    for mol, kwargs, capability in cases:
        r = ferric.run_optimize(mol, "sto-3g", **kwargs)
        assert r.certificate["capability"] == capability, kwargs


def test_open_shell_defaults_to_uhf():
    """Unset `reference` on a doublet is UHF: the same run as `run_optimize_uhf`
    and as `reference="uhf"`, bit for bit (one driver)."""
    nh2 = ferric.Molecule.from_xyz_string(NH2, 0, 2)
    auto = ferric.run_optimize(nh2, "sto-3g")
    explicit = ferric.run_optimize(nh2, "sto-3g", reference="uhf")
    alias = ferric.run_optimize_uhf(nh2, "sto-3g")
    assert auto.energy_trace == explicit.energy_trace == alias.energy_trace
    assert auto.certificate["capability"] == "optimize:uhf"


WATER_MOL = ferric.Molecule.from_xyz_string(WATER)
NH2_MOL = ferric.Molecule.from_xyz_string(NH2, 0, 2)
AUX = {"auxbasis": "cc-pvdz-ri"}


@pytest.mark.parametrize(
    "mol,kwargs,needle",
    [
        (WATER_MOL, dict(method="rimp2", c_os=1.0, **AUX), "c_os is not read"),
        (WATER_MOL, dict(c_ss=0.0), "c_ss is not read"),
        (NH2_MOL, dict(method="scs-mp2", **AUX), "closed-shell only"),
        (WATER_MOL, dict(method="scs-mp2", scf_jk="rijcosx", **AUX), "rijcosx"),
        (
            WATER_MOL,
            dict(method="scs-mp2", scf_density_conv=1e-8, **AUX),
            "scf_density_conv",
        ),
        (WATER_MOL, dict(method="mp3"), "unknown method"),
        (WATER_MOL, dict(reference="rks"), "unknown reference"),
        (WATER_MOL, dict(auxbasis="cc-pvdz-ri"), "auxbasis is not read"),
        (WATER_MOL, dict(oo_max_iter=5), "oo_max_iter is not read"),
        (WATER_MOL, dict(method="rimp2"), "needs auxbasis"),
        (WATER_MOL, dict(method="rimp2", xc="PBE", **AUX), "xc is not read"),
        (WATER_MOL, dict(method="rimp2", solvent="water", **AUX), "solvent"),
        (
            WATER_MOL,
            dict(method="rimp2", scf_density_conv=1e-8, **AUX),
            "scf_density_conv",
        ),
        (WATER_MOL, dict(method="rimp2", scf_jk="rijcosx", **AUX), "rijcosx"),
        (WATER_MOL, dict(method="rimp2", oo_grad_conv=1e-6, **AUX), "oo_grad_conv"),
        (NH2_MOL, dict(method="rimp2", **AUX), "closed-shell only"),
        (WATER_MOL, dict(method="oo-rimp2", frozen_core=1, **AUX), "frozen_core=0"),
        (
            WATER_MOL,
            dict(method="oo-rimp2", scf_jk="rijcosx", **AUX),
            "COSX exchange has no OO route",
        ),
        (
            WATER_MOL,
            dict(method="oo-rimp2", scf_ri_j_precision="f32", **AUX),
            "scf_ri_j_precision",
        ),
        (WATER_MOL, dict(dispersion="d3bj"), "requires xc"),
    ],
)
def test_unread_kwarg_is_refused(mol, kwargs, needle):
    with pytest.raises(ValueError, match=re.escape(needle)):
        ferric.run_optimize(mol, "sto-3g", **kwargs)


def _oo_rijk_endpoint():
    """The RI-JK OO-RI-MP2 minimum of distorted water (the CLI-matched route)."""
    mol = ferric.Molecule.from_xyz_string(WATER)
    r = ferric.run_optimize(
        mol,
        "sto-3g",
        method="oo-rimp2",
        auxbasis="cc-pvdz-ri",
        scf_jk="rijk",
        oo_grad_conv=1e-7,
        **SCF_CONV,
    )
    assert r.converged
    return r.mol()


def _oo_energy(symbols, coords_bohr, scf_jk):
    """OO-RI-MP2 total energy at `coords_bohr` on the `scf_jk` surface. The
    RI-J tensor is f64, as on the optimizer's surface."""
    mol = ferric.Molecule.from_arrays(symbols, coords_bohr, units="bohr")
    kw = {"scf_ri_j_precision": "f64"} if scf_jk == "rijk" else {}
    r = ferric.run_oo_rimp2(
        mol,
        ferric.BasisSet.bundled("sto-3g"),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
        scf_jk=scf_jk,
        grad_conv=1e-9,
        scf_energy_conv=1e-11,
        scf_density_conv=1e-10,
        **kw,
    )
    assert r.converged
    return r.total_energy


def _fd_gradient(symbols, coords, scf_jk, h=1e-3):
    import numpy as np

    x0 = np.asarray(coords, dtype=float)
    g = np.zeros_like(x0)
    for a in range(x0.shape[0]):
        for c in range(3):
            xp, xm = x0.copy(), x0.copy()
            xp[a, c] += h
            xm[a, c] -= h
            g[a, c] = (
                _oo_energy(symbols, xp, scf_jk) - _oo_energy(symbols, xm, scf_jk)
            ) / (2 * h)
    return g


def test_oo_rimp2_rijk_endpoint_is_stationary_on_the_fitted_surface():
    """The RI-JK optimizer stops at a stationary point of the RI-JK OO-RI-MP2
    energy, and the analytic gradient there is the derivative of that energy,
    not of the four-centre one.

    Physics vs artifact: on the fitted surface, a central finite difference of
    the FITTED energy at the endpoint is below the optimizer's max-gradient
    threshold (4.5e-4 Ha/Bohr) and equals the analytic gradient to the FD
    error; if any piece (reference, orbital steps or gradient) silently used
    four-centre J/K, the analytic gradient would instead track the FD of the
    EXACT energy, which differs from the fitted one by the fit's gradient
    error. The last assert shows that difference is resolvable at this bar.

    Measured (water / STO-3G, cc-pvdz-ri, def2-universal-jkfit): max |FD|
    1.40e-4; analytic vs FD 3.1e-7 at h = 1e-3 and 7.7e-8 at h = 5e-4 (the
    h^2 truncation, ratio 4.0); analytic vs the exact-surface FD 2.0e-5.
    FD_BAR = 1e-6 Ha/Bohr, 3x the h = 1e-3 truncation; the exact-surface
    control must miss by more than 10 bars.
    """
    import numpy as np

    end = _oo_rijk_endpoint()
    symbols, coords = end.symbols(), end.coords_bohr()
    g_fd = _fd_gradient(symbols, coords, "rijk")
    analytic = np.asarray(
        ferric.run_oo_rimp2(
            ferric.Molecule.from_arrays(symbols, coords, units="bohr"),
            ferric.BasisSet.bundled("sto-3g"),
            ferric.BasisSet.bundled("cc-pvdz-ri"),
            scf_jk="rijk",
            grad_conv=1e-9,
            scf_energy_conv=1e-11,
            scf_density_conv=1e-10,
            with_gradient=True,
        ).gradient()
    )
    fd_bar = 1e-6
    assert np.abs(g_fd).max() < 4.5e-4, g_fd
    assert np.abs(analytic - g_fd).max() < fd_bar
    g_exact = _fd_gradient(symbols, coords, "exact")
    miss = np.abs(analytic - g_exact).max()
    assert miss > 10 * fd_bar, miss


# `coordinates=` is the CLI's `[optimize] coordinates`. Each case optimizes the
# same molecule through both surfaces in redundant internal coordinates; a
# binding that dropped the kwarg would run the Cartesian BFGS, whose step
# count and final digits differ from the internal-coordinate path.
INTERNAL_ROUTES = [
    pytest.param(
        WATER,
        1,
        dict(**SCF_CONV),
        f'[method]\nkind = "rhf"\ntask = "optimize"\n\n{SCF_TOML}',
        id="rhf-water",
    ),
    pytest.param(
        NH2,
        2,
        dict(xc="PBE", df_j_aux=JKFIT, df_k_aux=JKFIT, **SCF_CONV),
        f'[method]\nkind = "uhf"\ntask = "optimize"\n\n'
        f'[dft]\nfunctional = "PBE"\n\n{SCF_TOML}',
        id="uks-pbe-nh2",
    ),
    pytest.param(
        WATER,
        1,
        dict(method="rimp2", auxbasis="cc-pvdz-ri"),
        '[method]\nkind = "rimp2"\ntask = "optimize"\n\n'
        '[mp2]\nauxbasis = "cc-pvdz-ri"\n',
        id="rimp2-water",
    ),
]


@pytest.mark.parametrize("xyz,mult,kwargs,toml", INTERNAL_ROUTES)
def test_internal_coordinates_match_the_cli(tmp_path, xyz, mult, kwargs, toml):
    toml = toml + '\n[optimize]\ncoordinates = "internal"\n'
    cli = _cli_result(_run_cli(tmp_path, xyz, mult, toml))
    mol = ferric.Molecule.from_xyz_string(xyz, 0, mult)
    py = ferric.run_optimize(mol, "sto-3g", coordinates="internal", **kwargs)
    assert py.converged
    assert _py_result(py) == cli
    cart = ferric.run_optimize(mol, "sto-3g", **kwargs)
    # The kwarg must change the path, or this test cannot see it dropped.
    assert (cart.steps, cart.energy_trace) != (py.steps, py.energy_trace)


def test_certificate_names_the_coordinates():
    water = ferric.Molecule.from_xyz_string(WATER)
    for coordinates, name in [
        (None, "Cartesian"),
        ("cartesian", "Cartesian"),
        ("internal", "RedundantInternal"),
    ]:
        kwargs = {} if coordinates is None else {"coordinates": coordinates}
        r = ferric.run_optimize(water, "sto-3g", max_steps=2, **kwargs)
        details = {a["name"]: a["detail"] for a in r.certificate["approximations"]}
        assert f"BFGS, {name} coordinates" in details["optimizer"], details


def test_run_optimize_uhf_passes_coordinates():
    nh2 = ferric.Molecule.from_xyz_string(NH2, 0, 2)
    a = ferric.run_optimize_uhf(nh2, "sto-3g", coordinates="internal", **SCF_CONV)
    b = ferric.run_optimize(
        nh2, "sto-3g", reference="uhf", coordinates="internal", **SCF_CONV
    )
    assert (a.steps, a.energy_trace) == (b.steps, b.energy_trace)


@pytest.mark.parametrize("bad", ["", "internl", "z-matrix", "delocalized"])
def test_unknown_coordinates_raise(bad):
    water = ferric.Molecule.from_xyz_string(WATER)
    with pytest.raises(ValueError, match='expected "cartesian" or "internal"'):
        ferric.run_optimize(water, "sto-3g", coordinates=bad)
