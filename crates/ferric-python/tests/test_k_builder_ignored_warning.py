"""`k_builder` superseded by density-fitted exchange must say so, `direct` included.

User report (0.1.0rc10): `run_dft(..., functional="B3LYP", k_builder="direct")`
was inert under the auto-defaulted RI-K (bitwise the default energy) and
printed nothing, while `k_builder="link"` in the same state printed the
IGNORED warning. A memory benchmark of "direct" was then read as a measurement
of the direct builder. The warning is printed by the SCF layer on stderr, so
these tests read it with `capfd` (file-descriptor capture sees Rust's
`eprintln!`).

Also pins `BasisSet.n_functions(mol)`, the documented `nbf` / `naux` count used
to size the density-fitting tensors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

WATER = str(
    Path(__file__).resolve().parents[3] / "testdata" / "molecules" / "water.xyz"
)
DIRECT_IGNORED = 'k_builder = "direct" is IGNORED because density-fitted J/K is active'


@pytest.fixture(scope="module")
def water():
    return ferric.Molecule.from_xyz(WATER), ferric.BasisSet.bundled("sto-3g")


def test_run_dft_hybrid_direct_is_inert_and_warns(water, capfd):
    """The reported case: B3LYP's default RI-K supersedes `direct`."""
    mol, bs = water
    default = ferric.run_dft(mol, bs, "B3LYP").total_energy
    capfd.readouterr()
    direct = ferric.run_dft(mol, bs, "B3LYP", k_builder="direct").total_energy
    err = capfd.readouterr().err
    assert direct == default, "k_builder='direct' under RI-K must be inert"
    assert DIRECT_IGNORED in err, err
    assert "exchange comes from DF-K" in err, err


def test_run_dft_hybrid_direct_with_exact_k_is_honoured_and_silent(water, capfd):
    """`df_k_aux=""` turns RI-K off: K then comes from the direct builder, so
    `direct` is honoured and no IGNORED line may appear."""
    mol, bs = water
    capfd.readouterr()
    r = ferric.run_dft(mol, bs, "B3LYP", k_builder="direct", df_k_aux="")
    err = capfd.readouterr().err
    assert r.converged
    assert "IGNORED" not in err, err


def test_run_rhf_direct_without_density_fitting_is_silent(water, capfd):
    """`run_rhf` is exact by default: `direct` is the builder used."""
    mol, bs = water
    capfd.readouterr()
    r = ferric.run_rhf(mol, bs, k_builder="direct")
    err = capfd.readouterr().err
    assert r.converged
    assert "IGNORED" not in err, err


def test_run_rhf_direct_with_named_df_k_warns(water, capfd):
    """HF with an explicitly named RI-K: `direct` is superseded, and says so."""
    mol, bs = water
    capfd.readouterr()
    ferric.run_rhf(
        mol,
        bs,
        k_builder="direct",
        df_j_aux="def2-universal-jkfit",
        df_k_aux="def2-universal-jkfit",
    )
    err = capfd.readouterr().err
    assert DIRECT_IGNORED in err, err


def test_n_functions_counts_orbital_and_auxiliary_functions(water):
    mol, sto3g = water
    # Water: O 1s2s2p (5) + 2 x H 1s = 7 at STO-3G; cc-pVDZ is 14 + 2 x 5 = 24.
    assert sto3g.n_functions(mol) == 7
    assert ferric.BasisSet.bundled("cc-pvdz").n_functions(mol) == 24
    # The auxiliary count, against an independent count from the shell list
    # (2l+1 per spherical shell, (l+1)(l+2)/2 per Cartesian one): O + 2 H.
    aux = ferric.BasisSet.bundled("def2-universal-jkfit")

    def per_atom(z):
        return sum(
            (2 * am + 1) if pure else (am + 1) * (am + 2) // 2
            for am, pure, _, _ in aux.shells(z)
        )

    assert aux.n_functions(mol) == per_atom(8) + 2 * per_atom(1)


def test_n_functions_refuses_an_uncovered_element():
    pople = ferric.BasisSet.bundled("6-31g")
    if 54 in pople.elements():
        pytest.skip("6-31g covers Xe in this build")
    xenon = ferric.Molecule.from_xyz_string("1\nXe\nXe 0 0 0\n")
    with pytest.raises(ValueError, match="no shells for Xe"):
        pople.n_functions(xenon)
