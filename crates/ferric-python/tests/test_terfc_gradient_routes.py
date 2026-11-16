"""`run_optimize` / `run_frequencies(method="att-rimp2", att_operator="terfc")`
and `method="scs-mp2-2terfc"` equal the CLI's `optimize` / `frequencies` for
those kinds.

Both sides call the same library drivers:
`ferric_mp2::att_rimp2_gradient::total_terfc_rimp2_gradient_on` /
`total_u_terfc_rimp2_gradient_on` and
`ferric_mp2::scs_2terfc_gradient::total_scs_mp2_2terfc_gradient_on` /
`total_u_scs_mp2_2terfc_gradient_on` (FD-validated in
`crates/ferric-mp2/tests/terfc_rimp2_gradient_fd.rs`,
`scs_2terfc_gradient_fd.rs` and `u_terfc_gradient_fd.rs`). These tests pin
the binding: `att_operator`, `r0`, `r0_bonded` / `r0_nonbonded` and `c_os` /
`c_ss` reach the same surface the TOML keys build, and an open shell runs the
unrestricted gradient.

The CLI comparisons need the `ferric-cli` binary of the same tree:
`FERRIC_CLI` (the test FAILS if that path is missing), else
`target/release/ferric-cli` (skips if absent). The terfc table engine reads
`FERRIC_TERF_TABLE_DIR`; unset, it is pointed at the repository's
`terf-tables/`.

Mutations measured to fail (each on an assertion, then restored):
- `att_terfc_operator` returning `Ok(None)` for "terfc" (the erfc route
  runs) fails `test_optimize_equals_the_cli` and
  `test_frequencies_equal_the_cli_bit_for_bit`;
- `att_terfc_operator` ignoring `r0` (the basis default in its place) fails
  `test_r0_reaches_the_gradient`;
- `run_optimize` passing 1.05 Å in place of `r0_nonbonded` fails
  `test_scs_2terfc_optimize_equals_the_cli`;
- the CLI taking `c_os` for `c_ss` fails both scs-mp2-2terfc CLI tests;
- the open-shell terfc optimize calling the closed-shell driver fails
  `test_open_shell_terfc_optimize_equals_the_cli`, on an error rather than
  an assertion (the closed driver raises on a doublet).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

ROOT = pathlib.Path(__file__).resolve().parents[3]
WATER = "3\ndistorted water\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n"
NH2 = "3\nNH2 radical\nN 0 0 0\nH 0 0.95 0.2\nH 0 -0.6 0.85\n"
H2 = "2\nh2\nH 0 0 0\nH 0 0 0.78\n"
AUX = "cc-pvdz-ri"
R0 = 1.2


@pytest.fixture(autouse=True)
def _tables(monkeypatch):
    if not os.environ.get("FERRIC_TERF_TABLE_DIR"):
        tables = ROOT / "terf-tables"
        assert (tables / "16_4_2.bin").is_file(), f"no terfc tables at {tables}"
        monkeypatch.setenv("FERRIC_TERF_TABLE_DIR", str(tables))


def _cli_binary():
    env = os.environ.get("FERRIC_CLI")
    if env:
        assert pathlib.Path(env).is_file(), f"FERRIC_CLI={env} does not exist"
        return pathlib.Path(env)
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def _run_cli(tmp_path, xyz, basis, body, mult=1):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    geom = tmp_path / "mol.xyz"
    geom.write_text(xyz, encoding="utf-8")
    toml = tmp_path / "run.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{geom}"\nmultiplicity = {mult}\n\n'
        f'[basis]\nname = "{basis}"\n\n{body}',
        encoding="utf-8",
    )
    log = tmp_path / "run.jsonl"
    env = dict(os.environ)
    env.setdefault("RAYON_NUM_THREADS", str(len(os.sched_getaffinity(0))))
    out = subprocess.run(
        [str(cli), str(toml), "--json", str(log)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout, log


def _terfc_mp2(extra=""):
    return f'[mp2]\nauxbasis = "{AUX}"\natt_operator = "terfc"\natt_r0 = {R0}\n{extra}'


def test_optimize_equals_the_cli(tmp_path):
    """Same final energy (10 decimals), geometry (8), steps and flag."""
    stdout, _ = _run_cli(
        tmp_path,
        WATER,
        "sto-3g",
        '[method]\nkind = "att-rimp2"\ntask = "optimize"\n\n' + _terfc_mp2(),
    )
    assert "Attenuated RI-MP2 (terfc, r0=1.200 Å" in stdout
    energy = re.search(r"final E    = (\S+) Hartree", stdout).group(1)
    block = stdout.split("Final Optimized Geometry (Bohr):\n")[1]
    geom = []
    for line in block.splitlines():
        if not line.strip():
            break
        geom.append(line)
    steps = int(re.search(r"steps      = (\d+)", stdout).group(1))
    mol = ferric.Molecule.from_xyz_string(WATER)
    py = ferric.run_optimize(
        mol, "sto-3g", method="att-rimp2", auxbasis=AUX, att_operator="terfc", r0=R0
    )
    assert py.converged
    assert f"{py.energy:.10f}" == energy
    m = py.mol()
    py_geom = [
        f"  {i:2} {s:2} {x:12.8f} {y:12.8f} {z:12.8f}"
        for i, (s, (x, y, z)) in enumerate(
            zip(m.symbols(), m.coords_bohr(), strict=True)
        )
    ]
    assert py_geom == geom
    assert py.steps == steps
    assert py.certificate["capability"] == "optimize:att-rimp2"


def test_frequencies_equal_the_cli_bit_for_bit(tmp_path):
    _, log = _run_cli(
        tmp_path,
        H2,
        "cc-pvdz",
        '[method]\nkind = "att-rimp2"\ntask = "frequencies"\n\n'
        + _terfc_mp2()
        + '\n[frequencies]\ndelta = 5e-3\n\n[scf]\njk = "exact"\n',
    )
    recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    (rec,) = [r for r in recs if r.get("record") == "frequencies"]
    h2 = ferric.Molecule.from_xyz_string(H2)
    py = ferric.run_frequencies(
        h2,
        "cc-pvdz",
        method="att-rimp2",
        auxbasis=AUX,
        scf_jk="exact",
        delta=5e-3,
        att_operator="terfc",
        r0=R0,
    )
    assert rec["energy"] == py.energy
    assert rec["frequencies_cm1"] == py.frequencies
    assert rec["cartesian_hessian"] == [x for row in py.cartesian_hessian for x in row]
    assert py.certificate["capability"] == "frequencies:att-rimp2"


def test_r0_reaches_the_gradient():
    """Two r0 values give two surfaces, and terfc is not the erfc surface."""
    h2 = ferric.Molecule.from_xyz_string(H2)
    kw = dict(method="att-rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3)
    a = ferric.run_frequencies(h2, "cc-pvdz", att_operator="terfc", r0=R0, **kw)
    b = ferric.run_frequencies(h2, "cc-pvdz", att_operator="terfc", r0=0.8, **kw)
    e = ferric.run_frequencies(h2, "cc-pvdz", **kw)
    assert abs(a.energy - b.energy) > 1e-5
    assert abs(a.energy - e.energy) > 1e-5
    assert (
        max(abs(x - y) for x, y in zip(a.frequencies, b.frequencies, strict=True)) > 1.0
    )


def _final(stdout):
    energy = re.search(r"final E    = (\S+) Hartree", stdout).group(1)
    block = stdout.split("Final Optimized Geometry (Bohr):\n")[1]
    geom = []
    for line in block.splitlines():
        if not line.strip():
            break
        geom.append(line)
    steps = int(re.search(r"steps      = (\d+)", stdout).group(1))
    return energy, geom, steps


def _py_geom(py):
    m = py.mol()
    return [
        f"  {i:2} {s:2} {x:12.8f} {y:12.8f} {z:12.8f}"
        for i, (s, (x, y, z)) in enumerate(
            zip(m.symbols(), m.coords_bohr(), strict=True)
        )
    ]


# Off the published 0.75 / 1.05 / 1.27 / 4.05, so a key that falls back to
# its default changes the surface.
SCS2 = dict(r0_bonded=0.8, r0_nonbonded=1.2, c_os=1.2, c_ss=3.0)
SCS2_TOML = (
    f'[mp2]\nauxbasis = "{AUX}"\nr0_bonded = 0.8\nr0_nonbonded = 1.2\n'
    "c_os = 1.2\nc_ss = 3.0\n"
)


def test_open_shell_terfc_optimize_equals_the_cli(tmp_path):
    """NH2 doublet on a UHF: same final energy, geometry, steps."""
    stdout, _ = _run_cli(
        tmp_path,
        NH2,
        "sto-3g",
        '[method]\nkind = "att-rimp2"\ntask = "optimize"\n\n' + _terfc_mp2(),
        mult=2,
    )
    assert "UHF + att-UMP2" in stdout
    energy, geom, steps = _final(stdout)
    mol = ferric.Molecule.from_xyz_string(NH2, 0, 2)
    py = ferric.run_optimize(
        mol, "sto-3g", method="att-rimp2", auxbasis=AUX, att_operator="terfc", r0=R0
    )
    assert py.converged
    assert f"{py.energy:.10f}" == energy
    assert _py_geom(py) == geom
    assert py.steps == steps


def test_scs_2terfc_optimize_equals_the_cli(tmp_path):
    stdout, _ = _run_cli(
        tmp_path,
        WATER,
        "sto-3g",
        '[method]\nkind = "scs-mp2-2terfc"\ntask = "optimize"\n\n' + SCS2_TOML,
    )
    assert "RHF + SCS-MP2(2terfc)" in stdout
    energy, geom, steps = _final(stdout)
    mol = ferric.Molecule.from_xyz_string(WATER)
    py = ferric.run_optimize(
        mol, "sto-3g", method="scs-mp2-2terfc", auxbasis=AUX, **SCS2
    )
    assert py.converged
    assert f"{py.energy:.10f}" == energy
    assert _py_geom(py) == geom
    assert py.steps == steps
    assert py.certificate["capability"] == "optimize:scs-mp2-2terfc"


def test_scs_2terfc_frequencies_equal_the_cli_bit_for_bit(tmp_path):
    """OH doublet (UHF) / 6-31G, one frozen core, four-centre reference.

    6-31G, not STO-3G: OH / STO-3G has E_SS = 0, so it cannot see c_ss.
    """
    oh = "2\nOH\nO 0 0 0\nH 0 0 0.97\n"
    _, log = _run_cli(
        tmp_path,
        oh,
        "6-31g",
        '[method]\nkind = "scs-mp2-2terfc"\ntask = "frequencies"\n\n'
        + SCS2_TOML
        + 'frozen_core = 1\n\n[frequencies]\ndelta = 5e-3\n\n[scf]\njk = "exact"\n',
        mult=2,
    )
    recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    (rec,) = [r for r in recs if r.get("record") == "frequencies"]
    py = ferric.run_frequencies(
        ferric.Molecule.from_xyz_string(oh, 0, 2),
        "6-31g",
        method="scs-mp2-2terfc",
        auxbasis=AUX,
        scf_jk="exact",
        delta=5e-3,
        frozen_core=1,
        **SCS2,
    )
    assert rec["energy"] == py.energy
    assert rec["frequencies_cm1"] == py.frequencies
    assert py.certificate["capability"] == "frequencies:scs-mp2-2terfc"


WATER_MOL = ferric.Molecule.from_xyz_string(WATER)
NH2_MOL = ferric.Molecule.from_xyz_string(NH2, 0, 2)


@pytest.mark.parametrize("task", ["optimize", "frequencies"])
@pytest.mark.parametrize(
    "kwargs,needle",
    [
        (dict(r0_bonded=1.05, r0_nonbonded=0.75), "r0"),
        (dict(r0_bonded=0.75, r0_nonbonded=1.05, att_operator="terfc"), "not read by"),
        (dict(r0_bonded=0.75, r0_nonbonded=1.05, omega=0.4), "not read by"),
    ],
)
def test_scs_2terfc_refusals(task, kwargs, needle):
    run = ferric.run_optimize if task == "optimize" else ferric.run_frequencies
    with pytest.raises(ValueError, match=needle):
        run(WATER_MOL, "sto-3g", method="scs-mp2-2terfc", auxbasis=AUX, **kwargs)


@pytest.mark.parametrize("task", ["optimize", "frequencies"])
@pytest.mark.parametrize("method", ["rimp2", "scs-mp2", "att-rimp2"])
def test_other_methods_refuse_r0_bonded(task, method):
    run = ferric.run_optimize if task == "optimize" else ferric.run_frequencies
    with pytest.raises(ValueError, match="r0_bonded is not read by"):
        run(WATER_MOL, "sto-3g", method=method, auxbasis=AUX, r0_bonded=0.75)


def test_scs_2terfc_uhf_on_a_singlet_raises():
    with pytest.raises(ValueError, match="reference follows the molecule"):
        ferric.run_optimize(
            WATER_MOL,
            "sto-3g",
            method="scs-mp2-2terfc",
            auxbasis=AUX,
            reference="uhf",
            **SCS2,
        )


@pytest.mark.parametrize(
    "mol,kwargs,needle",
    [
        (WATER_MOL, dict(att_operator="terfc", omega=0.4), "omega is not read"),
        (WATER_MOL, dict(r0=1.0), "r0 is read with att_operator"),
        (WATER_MOL, dict(att_operator="terf"), "unknown att_operator"),
        (WATER_MOL, dict(att_operator="terfc", r0=-1.0), "r0 must be"),
    ],
)
@pytest.mark.parametrize("task", ["optimize", "frequencies"])
def test_terfc_refusals(mol, kwargs, needle, task):
    run = ferric.run_optimize if task == "optimize" else ferric.run_frequencies
    with pytest.raises(ValueError, match=needle):
        run(mol, "sto-3g", method="att-rimp2", auxbasis=AUX, **kwargs)


@pytest.mark.parametrize("task", ["optimize", "frequencies"])
@pytest.mark.parametrize("kwargs", [dict(att_operator="terfc"), dict(r0=1.0)])
def test_other_methods_refuse_the_terfc_kwargs(task, kwargs):
    run = ferric.run_optimize if task == "optimize" else ferric.run_frequencies
    with pytest.raises(ValueError, match="not read by"):
        run(WATER_MOL, "sto-3g", method="rimp2", auxbasis=AUX, **kwargs)
