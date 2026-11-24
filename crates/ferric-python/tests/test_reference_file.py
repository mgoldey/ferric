"""`save_reference=` / `read_reference=` on `run_rimp2`, `run_ccsd`, `run_ccsd_t`.

One call solves the reference SCF and saves it; the next reads it instead of
solving. Pinned here: the correlated energy on the read reference is bitwise
the fresh one's, for another entry point as well as the same one, closed shell
(RHF) and open shell (UHF); the read call runs no SCF iteration and its
certificate names the file; a file for another calculation, a missing one and
the keyword combinations that cannot work raise ``InputError`` (a
``ValueError``). The reads pass ``read_reference_other_build=True`` because a
test build is often a dirty tree, which the checkpoint never counts as the
same build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

MOLECULES = Path(__file__).resolve().parents[3] / "testdata" / "molecules"


def _system(name, basis="6-31g", multiplicity=1):
    mol = ferric.Molecule.from_xyz(str(MOLECULES / name), multiplicity=multiplicity)
    return mol, ferric.BasisSet.bundled(basis), ferric.BasisSet.bundled("cc-pvdz-ri")


def _read(path):
    return {"read_reference": str(path), "read_reference_other_build": True}


def _routes(result):
    return [r["detail"] for r in result.certificate.get("routes", [])]


def _check_read(capfd, path, fresh, read_call, energy):
    capfd.readouterr()
    read = read_call(**_read(path))
    err = capfd.readouterr().err
    assert "no SCF iterations run" in err, err
    assert energy(read).hex() == energy(fresh).hex()
    assert any(d.startswith(f"read from {path}") for d in _routes(read)), _routes(read)
    return read


@pytest.mark.parametrize(
    ("name", "multiplicity"), [("water.xyz", 1), ("oh.xyz", 2)], ids=["rhf", "uhf"]
)
def test_a_reference_saved_by_rimp2_gives_rimp2_ccsd_and_ccsd_t_their_fresh_energies(
    tmp_path, capfd, name, multiplicity
):
    mol, obs, aux = _system(name, multiplicity=multiplicity)
    path = tmp_path / "ref.npz"
    saved = ferric.run_rimp2(mol, obs, aux, save_reference=str(path))
    assert path.is_file()
    assert "SCF reference saved to" in capfd.readouterr().err
    fresh = ferric.run_rimp2(mol, obs, aux)
    assert saved.total_energy.hex() == fresh.total_energy.hex()
    _check_read(
        capfd,
        path,
        fresh,
        lambda **kw: ferric.run_rimp2(mol, obs, aux, **kw),
        lambda r: r.total_energy,
    )
    for run in (ferric.run_ccsd, ferric.run_ccsd_t):
        fresh = run(mol, obs, aux)
        read = _check_read(
            capfd,
            path,
            fresh,
            lambda **kw: run(mol, obs, aux, **kw),
            lambda r: r.correlation_energy,
        )
        if run is ferric.run_ccsd_t:
            assert read.t_correction.hex() == fresh.t_correction.hex()


def test_a_reference_for_another_calculation_or_a_missing_file_is_refused(tmp_path):
    mol, obs, aux = _system("water.xyz", basis="sto-3g")
    path = tmp_path / "ref.npz"
    ferric.run_ccsd(mol, obs, aux, save_reference=str(path))
    with pytest.raises(ferric.InputError, match="different calculation"):
        ferric.run_ccsd(mol, ferric.BasisSet.bundled("6-31g"), aux, **_read(path))
    with pytest.raises(ValueError, match="scf_density_conv|density_conv"):
        ferric.run_ccsd(mol, obs, aux, scf_density_conv=1e-10, **_read(path))
    with pytest.raises(ValueError, match="does not exist"):
        ferric.run_ccsd(mol, obs, aux, **_read(tmp_path / "missing.npz"))


def test_keyword_combinations_that_cannot_work_raise_before_the_scf(tmp_path):
    mol, obs, aux = _system("water.xyz", basis="sto-3g")
    p = str(tmp_path / "ref.npz")
    cases = [
        (ferric.run_ccsd, {"save_reference": p, "read_reference": p}, "cannot both"),
        (ferric.run_ccsd_t, {"read_reference_other_build": True}, "read only with"),
        (ferric.run_rimp2, {"save_reference": " "}, "empty string"),
        (
            ferric.run_rimp2,
            {"save_reference": p, "local": "amplitude-threshold", "eps": 0.0},
            "exact energy only",
        ),
        (
            ferric.run_rimp2,
            {"save_reference": p, "with_gradient": True},
            "exact energy",
        ),
        (ferric.run_ccsd, {"save_reference": str(tmp_path)}, "is a directory"),
    ]
    for run, kw, needle in cases:
        with pytest.raises(ValueError, match=needle):
            run(mol, obs, aux, **kw)
    assert not Path(p).exists()
