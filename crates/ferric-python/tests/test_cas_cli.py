"""`run_casscf` / `run_nevpt2` equal the CLI kinds `casscf` / `nevpt2` bit
for bit.

Both surfaces call the same drivers (`ferric_ci::driver`); the CLI wiring is
pinned against those drivers in `crates/ferric-cli/tests/cas_kinds.rs`. These
tests pin the binding's side on the PySCF-referenced systems: the CASSCF
energies of `testdata/reference/casscf_*_pyscf.json` and the SC-NEVPT2 class
energies of `testdata/reference/nevpt2_refs.json`.

They need the `ferric-cli` binary of the same tree: `FERRIC_CLI` (set by the
weekly validation job, which builds it; the test FAILS if that path is
missing), else `target/release/ferric-cli` (skips if absent).

Mutation ledger (measured to fail): `run_nevpt2` ignoring
`casscf_grad_tol` (the library default 1e-7 in its place) fails the three
NEVPT2 cases, whose reference files set a tighter CASSCF gradient tolerance.
The CLI-side mutants are in `cas_kinds.rs`.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

ROOT = pathlib.Path(__file__).resolve().parents[3]
CASSCF_SYSTEMS = [
    "h2o_cas44_631g",
    "n2_cas66_631g",
    "n2_r1.6_cas66_631g",
    "o2_triplet_cas86_631g",
]
NAMES = ["Sr", "Si", "Sijrs", "Sijr", "Srsi", "Srs", "Sij", "Sir"]


def _cli_binary():
    """The CLI of this tree. `FERRIC_CLI` set but missing is a hard error (CI
    builds it and must not silently skip); unset, the local release build or
    `None`."""
    env = os.environ.get("FERRIC_CLI")
    if env:
        assert pathlib.Path(env).is_file(), f"FERRIC_CLI={env} does not exist"
        return pathlib.Path(env)
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def _run_cli(tmp_path, xyz, kind, basis, cas_lines):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    geom = tmp_path / "mol.xyz"
    geom.write_text(xyz)
    toml = tmp_path / "cas.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{geom}"\n\n[basis]\nname = "{basis}"\n\n'
        f'[method]\nkind = "{kind}"\n\n[cas]\n' + "".join(f"{x}\n" for x in cas_lines)
    )
    log = tmp_path / "run.jsonl"
    env = dict(os.environ)
    # Same rayon width as this process (rayon's default is the CPUs this
    # process may run on), so both run the same reduction order.
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
    recs = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    rec = [r for r in recs if r.get("record") == "result"]
    assert len(rec) == 1
    return rec[0]


@pytest.mark.validation
@pytest.mark.parametrize("name", CASSCF_SYSTEMS)
def test_run_casscf_equals_the_cli_bit_for_bit(tmp_path, name):
    ref = json.loads(
        (ROOT / "testdata" / "reference" / f"casscf_{name}_pyscf.json").read_text()
    )
    na, nb = ref["n_elec_active"]
    rec = _run_cli(
        tmp_path,
        ref["xyz"],
        "casscf",
        ref["basis"],
        [f"n_active = {ref['n_active']}", f"n_elec_active = [{na}, {nb}]"],
    )
    mol = ferric.Molecule.from_xyz_string(ref["xyz"])
    py = ferric.run_casscf(
        mol, ferric.BasisSet.bundled(ref["basis"]), ref["n_active"], (na, nb)
    )
    assert py.converged
    assert rec["total"] == py.e_total
    assert rec["components"]["iterations"] == py.iterations
    assert rec["components"]["certificate"]["grade"] == py.certificate["grade"]
    assert py.certificate["grade"] == "proven"
    assert abs(py.e_total - ref["e_total"]) < 1e-9


def _nevpt2_systems():
    refs = json.loads(
        (ROOT / "testdata" / "reference" / "nevpt2_refs.json").read_text()
    )
    return {s["system"]: s for s in refs["systems"]}


@pytest.mark.validation
@pytest.mark.parametrize("label", ["h2o", "n2", "n2-stretched"])
def test_run_nevpt2_equals_the_cli_bit_for_bit(tmp_path, label):
    s = _nevpt2_systems()[label]
    na, nb = s["n_elec_active"]
    grad_tol = s["ferric_casscf"]["casscf_grad_tol"]
    rec = _run_cli(
        tmp_path,
        s["xyz"],
        "nevpt2",
        s["basis"],
        [
            f"n_active = {s['n_active']}",
            f"n_elec_active = [{na}, {nb}]",
            f"grad_tol = {grad_tol!r}",
        ],
    )
    mol = ferric.Molecule.from_xyz_string(s["xyz"])
    py = ferric.run_nevpt2(
        mol,
        ferric.BasisSet.bundled(s["basis"]),
        s["n_active"],
        (na, nb),
        casscf_grad_tol=grad_tol,
    )
    comp = rec["components"]
    assert rec["total"] == py.e_total
    assert comp["e_corr"] == py.e_corr
    assert comp["e_reference"] == py.e_reference
    assert comp["classes"] == {n: py.energies[n] for n in NAMES}
    assert comp["certificate"]["grade"] == py.certificate["grade"] == "proven"
