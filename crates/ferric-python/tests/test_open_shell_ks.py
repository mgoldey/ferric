"""Open-shell Kohn-Sham from Python: `run_uks` and `run_roks`.

`run_dft` refuses `multiplicity > 1` (see test_closed_shell_refuses_open_shell),
and the periodic `run_uks_gamma` family is periodic only, so molecular UKS and
ROKS were reachable from the CLI (`kind = "ksdft"` on an open-shell molecule,
`kind = "rohf"` with `[dft] functional`) and from `run_qmmm(method="uks")` but
not as a plain call. `run_uks` / `run_roks` run `solve_uhf` / `solve_rohf`
with `RhfConfig.xc` set, the library route all three share.

Anchors (independent constructions):

* BITWISE vs the CLI: the converged energy (the last `scf_iter` energy of the
  JSON run log, 17 digits) and the whole SCF certificate (every field but the
  record envelope and the build's dirty flag) for OH (doublet) and O2
  (triplet) at PBE and B3LYP / 6-31G, UKS and ROKS. The CLI builds its own
  `RhfConfig` from TOML, so equality says the Python defaults ARE the CLI
  defaults (RI-J/RI-K via def2-universal-jkfit, production grid, MINAO guess).
* BITWISE vs `run_qmmm(method="uks")` with no MM charges: a second binding
  that solves the same SCF and calls `ks_gradient_uks_with_exchange_on_grids`
  directly; energy and gradient.
* The gradient against the CLI's first optimizer step: the CLI prints the
  geometry after one BFGS step to 8 decimals, and the first step is
  `x1 = x0 - g` (initial Hessian = 1), so `x0 - run_uks(...).gradient()`
  must reproduce it to the print precision (5e-9 Bohr) plus the SCF error
  of the CLI's own gradient (its density_conv 1e-9).
* Finite differences of the binding's own energy (bar derived below).
* UKS and ROKS of a singlet equal RKS.

FD bar. Central difference at step h = 1e-3 Bohr: error = h^2 |E'''| / 6 +
delta_E / h. With |E'''| <= 1 Ha/Bohr^3 the truncation term is <= 1.7e-7;
delta_E (density_conv 1e-9, energy_conv 1e-10, RI-J f64) is <= 1e-11 so
delta_E / h <= 1e-8. A moving-grid GGA adds the quadrature's own non-smoothness
in h; FD_BAR = 2e-6 is ~10x the sum.

Mutation ledger: see the final report of the commit that added this file.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

ROOT = pathlib.Path(__file__).resolve().parents[3]
MOLECULES = ROOT / "testdata" / "molecules"
TIGHT = {"energy_conv": 1e-10, "density_conv": 1e-9}
FD_H = 1e-3
FD_BAR = 2e-6
ANG = 1.0 / 0.52917721092  # ferric's ANGSTROM_TO_BOHR

SYSTEMS = {"oh": ("oh.xyz", 2), "o2": ("o2.xyz", 3)}
NH2 = (["N", "H", "H"], [(0.0, 0.01, 0.0), (0.02, 0.80, 0.60), (-0.03, -0.81, 0.58)])


def _molecule(name):
    xyz, mult = SYSTEMS[name]
    return ferric.Molecule.from_xyz(str(MOLECULES / xyz), 0, mult)


@pytest.fixture(scope="module")
def b631g():
    return ferric.BasisSet.bundled("6-31g")


# ── the CLI ──


def _cli_binary():
    env = os.environ.get("FERRIC_CLI")
    if env:
        assert pathlib.Path(env).is_file(), f"FERRIC_CLI={env} does not exist"
        return pathlib.Path(env)
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def _run_cli(tmp_path, name, functional, ref):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    xyz, mult = SYSTEMS[name]
    kind = "ksdft" if ref == "uks" else "rohf"
    log = tmp_path / "run.jsonl"
    toml = tmp_path / "in.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{MOLECULES / xyz}"\ncharge = 0\nmultiplicity = {mult}\n\n'
        f'[basis]\nname = "6-31g"\n\n[method]\nkind = "{kind}"\ntask = "energy"\n\n'
        f'[dft]\nfunctional = "{functional}"\n\n'
        "[scf]\nenergy_conv = 1e-10\ndensity_conv = 1e-9\nmax_iter = 200\n"
        f'\n[output]\njson = "{log}"\n'
    )
    p = subprocess.run([str(cli), str(toml)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    events = [json.loads(line) for line in log.read_text().splitlines()]
    return p.stdout, events


def _py(run, name, functional, bs, **kw):
    return getattr(ferric, run)(_molecule(name), bs, functional, **TIGHT, **kw)


CASES = [
    pytest.param(name, xc, ref, id=f"{name}-{xc.lower()}-{ref}")
    for name in ("oh", "o2")
    for xc in ("PBE", "B3LYP")
    for ref in ("uks", "roks")
]


@pytest.mark.parametrize("name, functional, ref", CASES)
def test_energy_and_certificate_equal_the_cli_bitwise(
    tmp_path, b631g, name, functional, ref
):
    _, events = _run_cli(tmp_path, name, functional, ref)
    cli_energy = [e["energy"] for e in events if "energy" in e][-1]
    cli_cert = next(e for e in events if e["record"] == "certificate")
    r = _py(f"run_{ref}", name, functional, b631g)
    assert r.converged
    assert r.energy == cli_energy
    cert = r.certificate
    assert cert["capability"] == {"uks": "uhf", "roks": "rohf"}[ref]
    skip = {"seq", "t", "record", "build"}
    assert {k: v for k, v in cert.items() if k not in skip} == {
        k: v for k, v in cli_cert.items() if k not in skip
    }
    # RI-J, the production grid, and RI-K for a hybrid.
    expect = ["ri-j", "xc-grid"] if functional == "PBE" else ["ri-j", "ri-k", "xc-grid"]
    assert sorted(a["name"] for a in cert["approximations"]) == sorted(expect)


# ── the gradient ──


@pytest.mark.parametrize(
    "name, functional",
    [("oh", "PBE"), ("oh", "B3LYP"), ("o2", "PBE"), ("o2", "B3LYP")],
)
def test_uks_gradient_equals_run_qmmm_bitwise(b631g, name, functional):
    mol = _molecule(name)
    r = ferric.run_uks(mol, b631g, functional, with_gradient=True, **TIGHT)
    symbols = mol.symbols()
    system = ferric.QmmmSystem(
        symbols,
        [list(c) for c in mol.coords()],
        [0.0] * len(symbols),
        qm_indices=list(range(len(symbols))),
        multiplicity=SYSTEMS[name][1],
    )
    q = ferric.run_qmmm(
        system, "6-31g", method="uks", xc=functional, ri_j_precision="f64", **TIGHT
    )
    assert r.energy == q.energy
    assert np.array_equal(r.gradient(), q.qm_gradient())


@pytest.mark.parametrize("ref", ["uks", "roks"])
def test_gradient_is_the_cli_first_optimizer_step(tmp_path, b631g, ref):
    """x1 = x0 - g: the CLI's geometry after one BFGS step, printed to 8 decimals."""
    out = _run_cli_optimize(tmp_path, ref)
    block = out.split("Final Optimized Geometry (Bohr):")[1].splitlines()[1:3]
    x1 = np.array([[float(v) for v in line.split()[2:5]] for line in block])
    mol = _molecule("oh")
    r = getattr(ferric, f"run_{ref}")(mol, b631g, "PBE", with_gradient=True, **TIGHT)
    x0 = np.asarray(mol.coords_bohr())
    assert np.abs(r.gradient()).max() > 1e-2, "trivial gradient, the anchor is vacuous"
    miss = np.abs(x0 - r.gradient() - x1).max()
    print(f"{ref}: max |x0 - g - x1_cli| = {miss:.2e}")
    assert miss < 2e-8, miss


def _run_cli_optimize(tmp_path, ref):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    kind = "ksdft" if ref == "uks" else "rohf"
    toml = tmp_path / "opt.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{MOLECULES / "oh.xyz"}"\ncharge = 0\nmultiplicity = 2\n\n'
        f'[basis]\nname = "6-31g"\n\n[method]\nkind = "{kind}"\ntask = "optimize"\n\n'
        '[dft]\nfunctional = "PBE"\n\n'
        "[scf]\nenergy_conv = 1e-10\ndensity_conv = 1e-9\n\n"
        "[optimize]\nmax_steps = 1\n\n[output]\njson = false\n"
    )
    p = subprocess.run([str(cli), str(toml)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout


def _nh2(bohr=None):
    symbols, coords = NH2
    if bohr is None:
        return ferric.Molecule.from_arrays(symbols, coords, multiplicity=2)
    return ferric.Molecule.from_arrays(symbols, bohr, units="bohr", multiplicity=2)


def _fd_gradient(energy):
    base = np.asarray(NH2[1], dtype=float) * ANG
    g = np.zeros_like(base)
    for i in range(base.shape[0]):
        for x in range(3):
            plus, minus = base.copy(), base.copy()
            plus[i, x] += FD_H
            minus[i, x] -= FD_H
            g[i, x] = (energy(_nh2(plus)) - energy(_nh2(minus))) / (2.0 * FD_H)
    return g


@pytest.mark.parametrize(
    "ref, functional",
    [
        ("uks", "PBE"),
        ("uks", "B3LYP"),
        ("roks", "PBE"),
        ("roks", "B3LYP"),
        # Range-separated hybrid and meta-GGA: the open-shell gradient paths
        # that reach `ks_gradient_uks` / `ks_gradient_roks` beyond LDA/GGA.
        ("uks", "HYB_GGA_XC_WB97X"),
        ("roks", "HYB_GGA_XC_WB97X"),
        ("uks", "SCAN"),
        ("roks", "SCAN"),
    ],
)
def test_gradient_matches_fd_of_its_energy(b631g, ref, functional):
    run = getattr(ferric, f"run_{ref}")
    kw = {**TIGHT, "ri_j_precision": "f64"}
    analytic = run(_nh2(), b631g, functional, with_gradient=True, **kw).gradient()
    fd = _fd_gradient(lambda m: run(m, b631g, functional, **kw).energy)
    assert np.abs(analytic).max() > 1e-3, "trivial gradient, the FD check is vacuous"
    miss = np.abs(analytic - fd).max()
    print(f"{ref}/{functional}: max |analytic - FD| = {miss:.2e}")
    assert miss < FD_BAR


def test_the_gradient_is_none_without_the_flag_and_the_energy_is_unchanged(b631g):
    plain = ferric.run_uks(_molecule("oh"), b631g, "PBE", **TIGHT)
    assert plain.gradient() is None
    with_g = ferric.run_uks(_molecule("oh"), b631g, "PBE", with_gradient=True, **TIGHT)
    # A gradient run resolves an unset RI-J precision to f64 (geometry task), so
    # the energy moves by the f32 RI-J fitting-storage error only.
    assert abs(plain.energy - with_g.energy) < 1e-5
    assert with_g.gradient().shape == (2, 3)


# ── closed-shell limit and the result ──


@pytest.mark.parametrize("run", ["run_uks", "run_roks"])
def test_a_singlet_equals_rks(b631g, run):
    """Trivial limit: with no unpaired electron both reduce to the RKS solution."""
    water = ferric.Molecule.from_xyz(str(MOLECULES / "water.xyz"))
    kw = {**TIGHT, "ri_j_precision": "f64"}
    e_rks = ferric.run_dft(water, b631g, "PBE", **kw).total_energy
    e = getattr(ferric, run)(water, b631g, "PBE", **kw).energy
    assert abs(e - e_rks) < 1e-9, (e, e_rks)


def test_the_result_carries_both_spin_channels(b631g):
    mol = _molecule("oh")
    r = ferric.run_uks(mol, b631g, "PBE", **TIGHT)
    n = len(r.orbital_energies_alpha())
    assert r.mo_coefficients_alpha().shape == r.mo_coefficients_beta().shape == (n, n)
    # OH: 9 electrons, multiplicity 2 -> 5 alpha, 4 beta.
    assert abs(r.occupations_alpha().sum() - 5.0) < 1e-8
    assert abs(r.occupations_beta().sum() - 4.0) < 1e-8
    assert not np.allclose(r.orbital_energies_alpha(), r.orbital_energies_beta())
    # The functional is in the energy: UKS is not UHF.
    assert abs(r.energy - ferric.run_uhf(mol, b631g, **TIGHT).energy) > 0.1
    # ROKS has one set of MOs.
    q = ferric.run_roks(mol, b631g, "PBE", **TIGHT)
    assert np.array_equal(q.mo_coefficients_alpha(), q.mo_coefficients_beta())
    assert abs(q.occupations_alpha().sum() - 5.0) < 1e-8
    assert abs(q.occupations_beta().sum() - 4.0) < 1e-8


@pytest.mark.parametrize("run", ["run_uks", "run_roks"])
def test_dm0_restart_reproduces_the_energy_and_a_swapped_pair_is_refused(b631g, run):
    mol = _molecule("oh")
    r = getattr(ferric, run)(mol, b631g, "PBE", **TIGHT)
    pair = (r.density_alpha(), r.density_beta())
    again = getattr(ferric, run)(mol, b631g, "PBE", dm0=pair, **TIGHT)
    assert abs(again.energy - r.energy) < 1e-9
    if run == "run_uks":
        assert again.iterations <= 2, again.iterations
    # 5 alpha / 4 beta electrons: the swapped pair holds the wrong counts.
    with pytest.raises(ValueError, match="dm0"):
        getattr(ferric, run)(mol, b631g, "PBE", dm0=pair[::-1], **TIGHT)


# ── refusals ──


def test_refusals(b631g):
    oh = _molecule("oh")
    # The closed-shell entry point still refuses an open-shell molecule.
    with pytest.raises(RuntimeError, match="multiplicity 2"):
        ferric.run_dft(oh, b631g, "PBE")
    with pytest.raises(RuntimeError, match="nonsense"):
        ferric.run_uks(oh, b631g, "nonsense")
    with pytest.raises(RuntimeError, match="nonsense"):
        ferric.run_roks(oh, b631g, "nonsense")
    with pytest.raises(ValueError, match="guess"):
        ferric.run_uks(oh, b631g, "PBE", guess="hcoer")
    with pytest.raises(ValueError, match="grid_prune"):
        ferric.run_uks(oh, b631g, "PBE", grid_prune="nwchemm")
    # A functional is required: there is no silent LDA default.
    with pytest.raises(TypeError):
        ferric.run_uks(oh, b631g)
    # dm0 next to guess names two starting points.
    pair = (np.eye(11), np.eye(11))
    with pytest.raises(ValueError, match="dm0"):
        ferric.run_uks(oh, b631g, "PBE", dm0=pair, guess="hcore")
    # An odd electron count cannot be a singlet: the molecule is refused up front.
    with pytest.raises(Exception, match="multiplicity"):
        ferric.Molecule.from_xyz(str(MOLECULES / "oh.xyz"), 0, 1)
