"""`run_frequencies(method="att-rimp2" | "pdep-rpa")` equals the CLI's
`task = "frequencies"` for the same kinds, bit for bit.

Both call the same library gradient (`total_att_rimp2_gradient_on`,
`total_rpa_gradient_by`) through `harmonic_frequencies_from_gradient`; the
CLI side is pinned against an energy stencil in
`crates/ferric-cli/tests/correlated_frequencies_cli.rs`. These tests pin the
binding: its kwargs reach the same configuration the TOML keys do.

They need the `ferric-cli` binary of the same tree: `FERRIC_CLI` (the test
FAILS if that path is missing), else `target/release/ferric-cli` (skips if
absent).

Mutation ledger (measured to fail, each on an assertion, then restored):
- `run_frequencies_att_rimp2` ignoring `omega` (0.420 in its place) fails
  `test_att_rimp2_omega_reaches_the_gradient`;
- `run_frequencies_att_rimp2` passing the `pdep-rpa` label / the wrong kind
  to `frequencies_certificate` fails the certificate capability assertion in
  `test_att_rimp2_equals_the_cli_bit_for_bit`;
- `run_frequencies_att_rimp2` ignoring `scf_jk` (always four-centre) fails
  `test_att_rimp2_rijk_reference_equals_the_cli`.

RI-MP2 family and open-shell att-rimp2 (five mutations applied in ONE build;
their failing sets are disjoint, and every other test passed):
- `run_frequencies_att_rimp2` certifying on a restricted reference fails
  `test_att_rimp2_open_shell_equals_the_cli_bit_for_bit` (a "final-point SCF" warning);
- `run_frequencies_mp2_family` ignoring `c_os` fails
  `test_scs_mp2_c_os_and_c_ss_reach_the_gradient` (energy);
- `oo_rimp2_setup` ignoring `oo_grad_conv` fails
  `test_oo_rimp2_loop_kwargs_reach_the_orbital_optimization` (energy);
- `run_frequencies_mp2_family` ignoring `frozen_core` fails
  `test_rimp2_frozen_core_reaches_the_gradient` (energy);
- `run_frequencies_mp2_family` passing the "pdep-rpa" kind to
  `frequencies_certificate` fails the capability assertion of the rimp2
  (closed, open shell), scs-mp2 and oo-rimp2 bit-for-bit tests.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

ROOT = pathlib.Path(__file__).resolve().parents[3]
H2 = "2\nh2\nH 0 0 0\nH 0 0 0.78\n"
AUX = "cc-pvdz-ri"


def _cli_binary():
    env = os.environ.get("FERRIC_CLI")
    if env:
        assert pathlib.Path(env).is_file(), f"FERRIC_CLI={env} does not exist"
        return pathlib.Path(env)
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def _cli_frequencies(
    tmp_path, kind, section, scf_extra, xyz=H2, molecule_extra="", basis="cc-pvdz"
):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    geom = tmp_path / "mol.xyz"
    geom.write_text(xyz)
    toml = tmp_path / "freq.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{geom}"\n{molecule_extra}\n[basis]\nname = "{basis}"\n\n'
        f'[method]\nkind = "{kind}"\ntask = "frequencies"\n\n{section}\n'
        f"[frequencies]\ndelta = 5e-3\n\n[scf]\n{scf_extra}"
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
    recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    rec = [r for r in recs if r.get("record") == "frequencies"]
    assert len(rec) == 1
    return rec[0]


@pytest.fixture(scope="module")
def h2():
    return ferric.Molecule.from_xyz_string(H2)


def _same(rec, py):
    assert rec["energy"] == py.energy
    assert rec["frequencies_cm1"] == py.frequencies
    assert rec["cartesian_hessian"] == [x for row in py.cartesian_hessian for x in row]
    assert rec["n_gradient_evaluations"] == py.n_gradient_evaluations


def test_att_rimp2_equals_the_cli_bit_for_bit(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path, "att-rimp2", f'[mp2]\nauxbasis = "{AUX}"\n', 'jk = "exact"\n'
    )
    py = ferric.run_frequencies(
        h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3
    )
    _same(rec, py)
    assert py.certificate["capability"] == "frequencies:att-rimp2"
    assert rec["certificate"]["capability"] == py.certificate["capability"]
    assert len(py.frequencies) == 1


def test_att_rimp2_omega_reaches_the_gradient(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path,
        "att-rimp2",
        f'[mp2]\nauxbasis = "{AUX}"\nomega = 0.8\n',
        'jk = "exact"\n',
    )
    py = ferric.run_frequencies(
        h2,
        "cc-pvdz",
        method="att-rimp2",
        auxbasis=AUX,
        scf_jk="exact",
        omega=0.8,
        delta=5e-3,
    )
    _same(rec, py)
    default = ferric.run_frequencies(
        h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3
    )
    assert abs(default.frequencies[0] - py.frequencies[0]) > 1.0  # cm^-1


def test_att_rimp2_rijk_reference_equals_the_cli(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path, "att-rimp2", f'[mp2]\nauxbasis = "{AUX}"\n', 'jk = "rijk"\n'
    )
    py = ferric.run_frequencies(
        h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, scf_jk="rijk", delta=5e-3
    )
    _same(rec, py)
    exact = ferric.run_frequencies(
        h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3
    )
    assert exact.frequencies[0] != py.frequencies[0]


def test_pdep_rpa_equals_the_cli_bit_for_bit(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path, "pdep-rpa", f'[rpa]\nauxbasis = "{AUX}"\nn_quad = 8\n', ""
    )
    py = ferric.run_frequencies(
        h2, "cc-pvdz", method="pdep-rpa", auxbasis=AUX, n_quad=8, delta=5e-3
    )
    _same(rec, py)
    assert py.certificate["capability"] == "frequencies:pdep-rpa"


def test_att_rimp2_refusals(h2):
    with pytest.raises(ValueError, match="auxbasis"):
        ferric.run_frequencies(h2, "cc-pvdz", method="att-rimp2")
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, xc="pbe"
        )
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, rpa_gradient="fd"
        )
    with pytest.raises(ValueError, match="omega"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, omega=-1.0
        )
    # att-rimp2-only kwargs are refused by the other methods.
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_frequencies(h2, "cc-pvdz", omega=0.4)
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="pdep-rpa", auxbasis=AUX, scf_jk="exact"
        )


# ---- method = "rimp2" | "scs-mp2" | "oo-rimp2" -------------------------------

LIH = "2\nlih\nLi 0 0 0\nH 0 0 1.6\n"
OH = "2\noh\nO 0 0 0\nH 0 0 0.97\n"


def test_rimp2_closed_shell_equals_the_cli_bit_for_bit(tmp_path, h2):
    # [scf] jk omitted: the CLI's RI-JK on the matched JK-fit set, which is
    # also run_frequencies' default.
    rec = _cli_frequencies(tmp_path, "rimp2", f'[mp2]\nauxbasis = "{AUX}"\n', "")
    py = ferric.run_frequencies(h2, "cc-pvdz", method="rimp2", auxbasis=AUX, delta=5e-3)
    _same(rec, py)
    assert py.certificate["capability"] == "frequencies:rimp2"
    assert rec["certificate"]["capability"] == py.certificate["capability"]
    exact = ferric.run_frequencies(
        h2, "cc-pvdz", method="rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3
    )
    assert exact.frequencies != py.frequencies  # scf_jk reaches the surface
    rec = _cli_frequencies(
        tmp_path, "rimp2", f'[mp2]\nauxbasis = "{AUX}"\n', 'jk = "exact"\n'
    )
    _same(rec, exact)


def test_rimp2_frozen_core_reaches_the_gradient(tmp_path):
    lih = ferric.Molecule.from_xyz_string(LIH)
    rec = _cli_frequencies(
        tmp_path,
        "rimp2",
        f'[mp2]\nauxbasis = "{AUX}"\nfrozen_core = 1\n',
        'jk = "exact"\n',
        xyz=LIH,
        basis="6-31g",
    )
    kw = dict(method="rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3)
    py = ferric.run_frequencies(lih, "6-31g", frozen_core=1, **kw)
    _same(rec, py)
    all_electron = ferric.run_frequencies(lih, "6-31g", **kw)
    assert all_electron.frequencies != py.frequencies


def test_rimp2_open_shell_equals_the_cli_bit_for_bit(tmp_path):
    oh = ferric.Molecule.from_xyz_string(OH, 0, 2)
    rec = _cli_frequencies(
        tmp_path,
        "rimp2",
        f'[mp2]\nauxbasis = "{AUX}"\n',
        'jk = "exact"\n',
        xyz=OH,
        molecule_extra="multiplicity = 2\n",
        basis="6-31g",
    )
    py = ferric.run_frequencies(
        oh, "6-31g", method="rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3
    )
    _same(rec, py)
    assert py.certificate["capability"] == "frequencies:rimp2"
    assert rec["certificate"]["capability"] == py.certificate["capability"]
    # A restricted reference on a doublet leaves the certificate ungraded.
    assert py.certificate["grade"] != "ungraded"
    assert py.certificate["grade"] == rec["certificate"]["grade"]
    assert py.certificate["warnings"] == rec["certificate"]["warnings"]


def test_scs_mp2_equals_the_cli_bit_for_bit(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path, "scs-mp2", f'[mp2]\nauxbasis = "{AUX}"\n', 'jk = "exact"\n'
    )
    kw = dict(method="scs-mp2", auxbasis=AUX, scf_jk="exact", delta=5e-3)
    py = ferric.run_frequencies(h2, "cc-pvdz", **kw)
    _same(rec, py)
    assert py.certificate["capability"] == "frequencies:scs-mp2"
    assert rec["certificate"]["capability"] == py.certificate["capability"]


def test_scs_mp2_c_os_and_c_ss_reach_the_gradient(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path,
        "scs-mp2",
        f'[mp2]\nauxbasis = "{AUX}"\nc_os = 1.0\nc_ss = 0.5\n',
        'jk = "exact"\n',
    )
    kw = dict(method="scs-mp2", auxbasis=AUX, scf_jk="exact", delta=5e-3)
    py = ferric.run_frequencies(h2, "cc-pvdz", c_os=1.0, c_ss=0.5, **kw)
    _same(rec, py)
    default = ferric.run_frequencies(h2, "cc-pvdz", **kw)
    assert abs(default.frequencies[0] - py.frequencies[0]) > 1.0  # cm^-1


def test_oo_rimp2_equals_the_cli_bit_for_bit(tmp_path, h2):
    rec = _cli_frequencies(tmp_path, "oo-rimp2", f'[mp2]\nauxbasis = "{AUX}"\n', "")
    py = ferric.run_frequencies(
        h2, "cc-pvdz", method="oo-rimp2", auxbasis=AUX, delta=5e-3
    )
    _same(rec, py)
    assert py.certificate["capability"] == "frequencies:oo-rimp2"
    assert rec["certificate"]["capability"] == py.certificate["capability"]


def test_oo_rimp2_loop_kwargs_reach_the_orbital_optimization(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path,
        "oo-rimp2",
        f'[mp2]\nauxbasis = "{AUX}"\noo_grad_conv = 1e-3\n',
        "",
    )
    kw = dict(method="oo-rimp2", auxbasis=AUX, delta=5e-3)
    py = ferric.run_frequencies(h2, "cc-pvdz", oo_grad_conv=1e-3, **kw)
    _same(rec, py)
    default = ferric.run_frequencies(h2, "cc-pvdz", **kw)
    assert default.cartesian_hessian != py.cartesian_hessian
    with pytest.raises(Exception, match="did not converge"):
        ferric.run_frequencies(h2, "cc-pvdz", oo_max_iter=1, **kw)


def test_mp2_family_frequency_refusals(h2):
    oh = ferric.Molecule.from_xyz_string(OH, 0, 2)
    for method in ("rimp2", "scs-mp2", "oo-rimp2"):
        with pytest.raises(ValueError, match="auxbasis"):
            ferric.run_frequencies(h2, "cc-pvdz", method=method)
        for kw in ({"xc": "pbe"}, {"hessian": "fd"}, {"scf_eri": "direct"}):
            with pytest.raises(ValueError, match="not read by"):
                ferric.run_frequencies(h2, "cc-pvdz", method=method, auxbasis=AUX, **kw)
        with pytest.raises(ValueError, match="omega"):
            ferric.run_frequencies(
                h2, "cc-pvdz", method=method, auxbasis=AUX, omega=0.4
            )
    # scs-mp2 and oo-rimp2 read their own kwargs only.
    with pytest.raises(ValueError, match="c_os"):
        ferric.run_frequencies(h2, "cc-pvdz", method="rimp2", auxbasis=AUX, c_os=1.0)
    with pytest.raises(ValueError, match="oo_grad_conv"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="scs-mp2", auxbasis=AUX, oo_grad_conv=1e-3
        )
    with pytest.raises(ValueError, match="c_ss"):
        ferric.run_frequencies(h2, "cc-pvdz", method="oo-rimp2", auxbasis=AUX, c_ss=0.5)
    # The other methods and "scf" refuse the new kwargs.
    for kw in ({"c_os": 1.0}, {"oo_max_iter": 5}, {"oo_grad_conv": 1e-4}):
        with pytest.raises(ValueError, match="not read by"):
            ferric.run_frequencies(h2, "cc-pvdz", **kw)
        with pytest.raises(ValueError, match="not read by"):
            ferric.run_frequencies(
                h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, **kw
            )
    # Frozen core, reference and spin scope as the CLI's.
    with pytest.raises(ValueError, match="frozen_core=0"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="oo-rimp2", auxbasis=AUX, frozen_core=1
        )
    for method in ("scs-mp2", "oo-rimp2"):
        with pytest.raises(ValueError, match="closed-shell"):
            ferric.run_frequencies(oh, "6-31g", method=method, auxbasis=AUX)
    with pytest.raises(ValueError, match="UHF"):
        ferric.run_frequencies(
            oh, "6-31g", method="rimp2", auxbasis=AUX, reference="rohf"
        )
    with pytest.raises(ValueError, match="does not match"):
        ferric.run_frequencies(
            oh, "6-31g", method="rimp2", auxbasis=AUX, reference="rhf"
        )
    with pytest.raises(ValueError, match="does not match"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="rimp2", auxbasis=AUX, reference="uhf"
        )
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="rimp2", auxbasis=AUX, point_charges=[(1, 0, 0, 5)]
        )


def test_att_rimp2_open_shell_equals_the_cli_bit_for_bit(tmp_path):
    oh = ferric.Molecule.from_xyz_string(OH, 0, 2)
    rec = _cli_frequencies(
        tmp_path,
        "att-rimp2",
        f'[mp2]\nauxbasis = "{AUX}"\n',
        'jk = "exact"\n',
        xyz=OH,
        molecule_extra="multiplicity = 2\n",
        basis="6-31g",
    )
    py = ferric.run_frequencies(
        oh, "6-31g", method="att-rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3
    )
    _same(rec, py)
    assert py.certificate["capability"] == "frequencies:att-rimp2"
    assert rec["certificate"]["capability"] == py.certificate["capability"]
    # att-rimp2 frequencies are ungraded by design, so the reference is read
    # from the warnings: a restricted reference on a doublet fails its final-point
    # SCF, and that failure is a warning the CLI's certificate does not have.
    assert not any("final-point SCF" in w for w in py.certificate["warnings"])
    assert py.certificate["warnings"] == rec["certificate"]["warnings"]
    with pytest.raises(ValueError, match="UHF"):
        ferric.run_frequencies(
            oh, "6-31g", method="att-rimp2", auxbasis=AUX, reference="rohf"
        )
    with pytest.raises(ValueError, match="does not match"):
        ferric.run_frequencies(
            oh, "6-31g", method="att-rimp2", auxbasis=AUX, reference="rhf"
        )
