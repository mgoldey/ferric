"""The CLI `[cdft]` section and `[qmmm]` polarizable embedding equal the Python
drivers bit for bit.

`[cdft]` runs `ferric_scf::cdft_driver::solve_cdft_uhf` on the CLI's reference
config and `run_cdft` runs it on its own; the polarizable `[qmmm]` runs
`solve_uhf` / `solve_rhf` with `RhfConfig.polarizable` and `run_qmmm` builds the
same config. Both pairs must land on the same bits, so a CLI default that
differs from the binding's (guess, screening, J/K route, grid, convergence
gates), a constraint sign, or a dropped polarizability shows up as a bit
difference rather than hiding inside a tolerance.

Cases are the existing cDFT and Thole test cases: LiH charge 2.2 and LiH+ spin
0.45 (`test_cdft.py`, `cdft_uhf.rs`), the pinned HeNe+ state
(`cdft_outer_loop.rs`) and the three-site water Thole embedding of
`test_qmmm.py` (`water_sto-3g_pe_three_sites*.json`, a PySCF-based prototype).

They need the `ferric-cli` binary of the same tree: `FERRIC_CLI` (set by the
weekly validation job, which builds it; the test FAILS if that path is
missing), else `target/release/ferric-cli` (skips if absent).

Mutation ledger (measured to fail, see the PR): the sign of a constraint
target in `cdft::constraints_for`, the polarizable sites dropped in
`QmmmCfg::polarizable_embedding`.
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
PQR = ROOT / "testdata" / "molecules" / "water_thole_sites.pqr"
ANG2BOHR_INV = 0.52917721092


def _cli_binary():
    env = os.environ.get("FERRIC_CLI")
    if env:
        assert pathlib.Path(env).is_file(), f"FERRIC_CLI={env} does not exist"
        return pathlib.Path(env)
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def _run_cli(tmp_path, toml_text):
    """Run the CLI on `toml_text`; return its run-log records and stdout."""
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    toml = tmp_path / "input.toml"
    toml.write_text(toml_text)
    log = tmp_path / "run.jsonl"
    env = dict(os.environ)
    # Same rayon width as this process, so both run the same reduction order.
    env.setdefault("RAYON_NUM_THREADS", str(len(os.sched_getaffinity(0))))
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
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
    return recs, out.stdout


def _one(recs, record, **match):
    hits = [
        r
        for r in recs
        if r.get("record") == record and all(r.get(k) == v for k, v in match.items())
    ]
    assert len(hits) == 1, (record, match, [r.get("record") for r in recs])
    return hits[0]


# ---------------------------------------------------------------------------
# [cdft]
# ---------------------------------------------------------------------------

LIH = "testdata/molecules/lih.xyz"


def _cdft_toml(*, charge, mult, constraints, scf="", dft="", cdft=""):
    body = "".join(
        f'[[cdft.constraint]]\natoms = {atoms}\ntarget = {target}\nkind = "{kind}"\n'
        for atoms, target, kind in constraints
    )
    return (
        f'[molecule]\nxyz = "{LIH}"\ncharge = {charge}\nmultiplicity = {mult}\n'
        '[basis]\nname = "def2-svp"\n[method]\nkind = "uhf"\ntask = "energy"\n'
        f"{scf}{dft}[cdft]\n{cdft}{body}"
    )


def _lih(charge=0, mult=1):
    return ferric.Molecule.from_xyz_string(
        "2\nLiH\nLi 0 0 0\nH 0 0 1.60\n", charge, mult
    )


def _same_certificate(cli_cert, py_cert):
    assert cli_cert["capability"] == py_cert["capability"]
    assert cli_cert["grade"] == py_cert["grade"] == "ungraded"
    cli_ap = {a["name"]: a["detail"] for a in cli_cert["approximations"]}
    py_ap = {a["name"]: a["detail"] for a in py_cert["approximations"]}
    assert cli_ap == py_ap
    assert cli_cert["warnings"] == py_cert["warnings"]


def _assert_cdft_equal(rec, py):
    comp = rec["components"]
    assert rec["total"] == py.energy
    assert comp["lambdas"] == py.lambdas
    assert comp["populations"] == py.populations
    assert comp["residuals"] == py.residuals
    assert comp["outer_iterations"] == py.outer_iterations
    assert comp["iterations"] == py.iterations
    assert comp["converged"] is True and py.converged
    assert comp["kinds"] == py.kinds
    _same_certificate(comp["certificate"], py.certificate)


@pytest.mark.validation
def test_lih_charge_constraint_equals_run_cdft_bit_for_bit(tmp_path):
    """LiH/def2-SVP, N(Li) = 2.2 (`cdft_uhf.rs::charge_constraint_is_satisfied`)."""
    recs, _ = _run_cli(
        tmp_path,
        _cdft_toml(
            charge=0,
            mult=1,
            constraints=[([0], 2.2, "charge")],
            cdft="lambda_tol = 1e-5\nmax_outer = 40\n",
        ),
    )
    py = ferric.run_cdft(
        _lih(),
        ferric.BasisSet.bundled("def2-svp"),
        [ferric.CdftConstraint([0], 2.2, kind="charge")],
        lambda_tol=1e-5,
        max_outer=40,
    )
    rec = _one(recs, "result", kind="cdft")
    _assert_cdft_equal(rec, py)
    assert abs(py.populations[0] - 2.2) < 1e-5
    # The constraint is applied: the energy is above the unconstrained UHF.
    uhf = ferric.run_uhf(_lih(), ferric.BasisSet.bundled("def2-svp"))
    assert py.energy > uhf.energy + 1e-6


@pytest.mark.validation
def test_lih_plus_spin_constraint_equals_run_cdft_bit_for_bit(tmp_path):
    """LiH+, N_a - N_b on Li = 0.45 (`cdft_uhf.rs::spin_constraint_is_satisfied_lih_plus`)."""
    recs, _ = _run_cli(
        tmp_path,
        _cdft_toml(
            charge=1,
            mult=2,
            constraints=[([0], 0.45, "spin")],
            cdft="lambda_tol = 1e-5\nmax_outer = 40\n",
        ),
    )
    py = ferric.run_cdft(
        _lih(1, 2),
        ferric.BasisSet.bundled("def2-svp"),
        [ferric.CdftConstraint([0], 0.45, kind="spin")],
        lambda_tol=1e-5,
        max_outer=40,
    )
    _assert_cdft_equal(_one(recs, "result", kind="cdft"), py)
    assert py.kinds == ["spin"]


@pytest.mark.validation
def test_uks_constraint_equals_run_cdft_bit_for_bit(tmp_path):
    """LiH/PBE, N(Li) = 2.2, exact J/K on both sides (the CLI's KS default is RI-JK)."""
    recs, _ = _run_cli(
        tmp_path,
        _cdft_toml(
            charge=0,
            mult=1,
            constraints=[([0], 2.2, "charge")],
            scf='[scf]\ndf_j_aux = "exact"\ndf_k_aux = "exact"\n',
            dft='[dft]\nfunctional = "PBE"\n',
            cdft="lambda_tol = 1e-5\nmax_outer = 40\n",
        ),
    )
    py = ferric.run_cdft(
        _lih(),
        ferric.BasisSet.bundled("def2-svp"),
        [ferric.CdftConstraint([0], 2.2)],
        functional="PBE",
        df_j_aux="exact",
        df_k_aux="exact",
        lambda_tol=1e-5,
        max_outer=40,
    )
    rec = _one(recs, "result", kind="cdft")
    _assert_cdft_equal(rec, py)
    assert rec["components"]["certificate"]["capability"] == "cdft:uks"


def _rust_const(name):
    src = (ROOT / "crates" / "ferric-scf" / "tests" / "cdft_outer_loop.rs").read_text()
    m = re.search(rf"const {name}: f64 = ([^;]+);", src)
    assert m, name
    return float(m.group(1).replace("_", ""))


@pytest.mark.validation
def test_henep_example_equals_run_cdft_and_the_pinned_state(tmp_path):
    """`examples/henep-cdft.toml` is `test_cdft.py`'s pinned HeNe+ call."""
    recs, _ = _run_cli(tmp_path, (ROOT / "examples" / "henep-cdft.toml").read_text())
    mol = ferric.Molecule.from_xyz_string(
        "2\nHeNe+\nHe 0.0 0.0 0.0\nNe 0.0 0.0 2.0\n", 1, 2
    )
    py = ferric.run_cdft(
        mol,
        ferric.BasisSet.bundled("def2-svp"),
        [ferric.CdftConstraint([0], 2.0, kind="charge")],
        guess="hcore",
        level_shift=0.5,
        scf_max_iter=400,
        lambda_tol=1e-5,
        max_outer=40,
        stability_descent=False,
        grid_radial=99,
        grid_angular=302,
    )
    rec = _one(recs, "result", kind="cdft")
    _assert_cdft_equal(rec, py)
    # The pinned Rust anchor, at that test's own bar.
    assert abs(rec["total"] - _rust_const("ANCHOR_E")) < 1e-6
    assert abs(rec["components"]["lambdas"][0] - _rust_const("ANCHOR_LAMBDA")) < 1e-3


# ---------------------------------------------------------------------------
# [qmmm] polarizable embedding
# ---------------------------------------------------------------------------

ALPHAS = [0.0, 0.0, 0.0, 0.0740923557432, 0.0740923557432, 0.163003182635]


def _pqr_atoms():
    atoms = []
    for line in PQR.read_text().splitlines():
        if line.startswith("ATOM"):
            f = line.split()
            atoms.append(
                (f[2][0], [float(f[5]), float(f[6]), float(f[7])], float(f[8]))
            )
    return atoms


def _py_system(alphas):
    atoms = _pqr_atoms()
    kw = {} if alphas is None else {"polarizabilities_angstrom3": alphas}
    return ferric.QmmmSystem(
        [a[0] for a in atoms],
        [a[1] for a in atoms],
        [a[2] for a in atoms],
        qm_indices=[0, 1, 2],
        **kw,
    )


def _qmmm_toml(kind, qmmm_extra):
    return (
        '[molecule]\nxyz = "unused.xyz"\n[basis]\nname = "sto-3g"\n'
        f'[method]\nkind = "{kind}"\ntask = "energy"\n'
        f'[qmmm]\npqr = "{PQR}"\nqm_indices = [0, 1, 2]\n{qmmm_extra}'
    )


def _alpha_line(alphas):
    return f"polarizabilities_angstrom3 = {alphas}\n"


@pytest.mark.validation
@pytest.mark.parametrize("thole", [None, 0.0])
def test_thole_embedding_equals_run_qmmm_bit_for_bit(tmp_path, thole):
    """UHF on the three-site water (`water_sto-3g_pe_three_sites[_nodamp]`).

    `thole = 0.0` is the undamped model, as in `run_qmmm(thole_a=0.0)`. The
    CLI's UHF and `run_qmmm(method="uhf")` are the same `solve_uhf` call on the
    same config, so energy, E_pol and every induced-dipole component agree to
    the bit.
    """
    extra = _alpha_line(ALPHAS) + ("" if thole is None else f"thole_a = {thole}\n")
    recs, stdout = _run_cli(tmp_path, _qmmm_toml("uhf", extra))
    py = ferric.run_qmmm(_py_system(ALPHAS), "sto-3g", method="uhf", thole_a=thole)
    assert py.converged
    note = _one(recs, "polarization")
    assert note["scf_energy"] == py.energy
    assert note["e_pol"] == py.e_pol
    assert note["e_pol"] != 0.0
    assert note["n_sites"] == 3
    assert note["induced_dipoles"] == [
        list(map(float, d)) for d in py.induced_dipoles()
    ]
    assert "E_pol" in stdout


@pytest.mark.validation
@pytest.mark.parametrize(
    "tag,thole",
    [
        ("water_sto-3g_pe_three_sites", None),
        ("water_sto-3g_pe_three_sites_nodamp", 0.0),
    ],
)
def test_thole_embedding_matches_the_pyscf_prototype(tmp_path, tag, thole):
    """The CLI against the PySCF-based reference, at `test_qmmm.py`'s bars
    (1e-7 Ha energy and E_pol, 1e-6 dipoles). The reference's sites are the
    PQR's, so this is an independent check of the CLI path, not of Python's."""
    ref = json.loads((ROOT / "testdata" / "reference" / f"{tag}.json").read_text())
    extra = _alpha_line(ALPHAS) + ("" if thole is None else f"thole_a = {thole}\n")
    recs, _ = _run_cli(tmp_path, _qmmm_toml("uhf", extra))
    note = _one(recs, "polarization")
    assert abs(note["scf_energy"] - ref["energy"]) < 1e-7
    assert abs(note["e_pol"] - ref["e_pol"]) < 1e-7
    got = note["induced_dipoles"]
    for row, want in zip(got, ref["induced_dipoles"]):
        assert max(abs(a - b) for a, b in zip(row, want)) < 1e-6


@pytest.mark.validation
def test_thole_example_matches_run_qmmm_rhf(tmp_path):
    """`examples/water-qmmm-thole.toml` (RHF, through the CLI's SCF ladder)
    agrees with `run_qmmm(method="rhf")` to the SCF convergence, and with the
    PySCF prototype to 1e-7."""
    recs, stdout = _run_cli(
        tmp_path, (ROOT / "examples" / "water-qmmm-thole.toml").read_text()
    )
    py = ferric.run_qmmm(_py_system(ALPHAS), "sto-3g", method="rhf")
    note = _one(recs, "polarization")
    assert abs(note["scf_energy"] - py.energy) < 1e-8
    assert abs(note["e_pol"] - py.e_pol) < 1e-8
    ref = json.loads(
        (
            ROOT / "testdata" / "reference" / "water_sto-3g_pe_three_sites.json"
        ).read_text()
    )
    assert abs(note["scf_energy"] - ref["energy"]) < 1e-7
    assert "E_pol" in stdout


@pytest.mark.validation
def test_embedding_without_polarizabilities_is_the_plain_path(tmp_path):
    """No `polarizabilities_angstrom3`: no polarization record, and the SCF is
    `run_qmmm`'s non-polarizable embedding, bit for bit (the CLI analogue of
    `polarizable_none_is_bit_identical_to_plain_scf`). The energy is read from
    the last SCF iteration record, which is the converged energy."""
    recs, stdout = _run_cli(tmp_path, _qmmm_toml("uhf", ""))
    assert not [r for r in recs if r.get("record") == "polarization"]
    assert "E_pol" not in stdout
    py = ferric.run_qmmm(_py_system(None), "sto-3g", method="uhf")
    assert py.e_pol == 0.0
    iters = [r for r in recs if r.get("record") == "scf_iter"]
    assert iters[-1]["energy"] == py.energy
