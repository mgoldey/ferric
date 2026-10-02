"""COSX from Python: RIJCOSX routing, the gradient, and the strict cosx_* kwargs.

`run_dft(k_builder="cosx")` is RIJCOSX (RI-J on by default, COSX replaces
RI-K). With `with_gradient=True` the gradient is routed through
`restricted_scf_gradient`, which differentiates the RI-J + COSX energy; the
overlap-fitted COSX + functional gradient (no XC Fock derivative in its
Z-vector) is refused BEFORE the SCF.

HOW THESE FAIL IF REVERTED: the FD test misses by the COSX error (~1e-5
Ha/Bohr, the exact-K pairing) if the KS gradient is used again; the kwarg
tests stop raising if a knob is silently ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")

WATER = str(
    Path(__file__).resolve().parents[3] / "testdata" / "molecules" / "water.xyz"
)
BOHR = 0.529177210903


@pytest.fixture(scope="module")
def setup():
    return ferric.Molecule.from_xyz(WATER), ferric.BasisSet.bundled("sto-3g")


def test_fitted_cosx_with_a_functional_gradient_is_refused(setup):
    mol, bs = setup
    with pytest.raises(ValueError, match="overlap_fit"):
        ferric.run_dft(mol, bs, "B3LYP", k_builder="cosx", with_gradient=True)


def _water_xyz(dz_bohr: float) -> str:
    lines = Path(WATER).read_text().splitlines()
    rows = [line.split() for line in lines[2:] if line.strip()]
    rows[1][3] = f"{float(rows[1][3]) + dz_bohr * BOHR:.12f}"
    return "\n".join([str(len(rows)), "water"] + [" ".join(r) for r in rows]) + "\n"


def test_rijcosx_b3lyp_gradient_matches_fd(setup):
    """Fit-off RIJCOSX B3LYP: analytic dE/dz(H1) vs central FD of the energy.

    MEASURED 1.6e-8 Ha/Bohr (2026-10-01); bar 2e-7. The exact-K pairing
    misses by ~1e-5 (module doc), so the bar separates the two by 50x.
    """
    _, bs = setup
    kw = dict(
        k_builder="cosx", cosx_overlap_fit=False, energy_conv=1e-8, density_conv=1e-9
    )
    mol0 = ferric.Molecule.from_xyz_string(_water_xyz(0.0))
    g = np.asarray(
        ferric.run_dft(mol0, bs, "B3LYP", with_gradient=True, **kw).gradient()
    )
    h = 1e-3
    ep = ferric.run_dft(
        ferric.Molecule.from_xyz_string(_water_xyz(h)), bs, "B3LYP", **kw
    )
    em = ferric.run_dft(
        ferric.Molecule.from_xyz_string(_water_xyz(-h)), bs, "B3LYP", **kw
    )
    fd = (ep.total_energy - em.total_energy) / (2 * h)
    print(f"analytic {g[1, 2]:+.10f} FD {fd:+.10f} diff {abs(g[1, 2] - fd):.2e}")
    assert abs(g[1, 2] - fd) < 2e-7


def test_cosx_kwargs_are_strict(setup):
    mol, bs = setup
    with pytest.raises(ValueError, match="k_builder"):
        ferric.run_rhf(mol, bs, cosx_grid=(35, 194, "sgx"))
    with pytest.raises(ValueError, match="prune"):
        ferric.run_rhf(mol, bs, k_builder="cosx", cosx_grid=(35, 194, "orca"))
    with pytest.raises(ValueError, match="peak"):
        ferric.run_rhf(mol, bs, k_builder="cosx", cosx_grid=(35, 26, "sgx"))
    with pytest.raises(ValueError, match="contradicts"):
        ferric.run_rhf(
            mol, bs, k_builder="cosx", cosx_final_pass=False, cosx_final_grid=(50, 302)
        )
    with pytest.raises(ValueError):
        ferric.run_rhf(mol, bs, k_builder="cosx", cosx_grid=(35,))


def test_final_pass_record(setup):
    mol, bs = setup
    r = ferric.run_rhf(
        mol,
        bs,
        k_builder="cosx",
        cosx_grid=(35, 194, "sgx"),
        cosx_final_grid=(50, 302, "sgx"),
    )
    fp = r.cosx_final_pass
    assert fp is not None
    assert r.energy == fp["e_final"]
    assert fp["npts_final_grid"] > fp["npts_scf_grid"]
    assert fp["npts_scf_grid"] == ferric.cosx_grid_point_count(mol, (35, 194, "sgx"))
    assert fp["gradient_differentiates"] == "e_scf_grid"
    plain = ferric.run_rhf(mol, bs, k_builder="cosx", cosx_grid=(35, 194, "sgx"))
    assert plain.cosx_final_pass is None
    assert plain.energy == fp["e_scf_grid"]


def test_run_dft_cosx_is_rijcosx(setup):
    mol, bs = setup
    rij = ferric.run_dft(mol, bs, "B3LYP", k_builder="cosx").total_energy
    exact_j = ferric.run_dft(
        mol, bs, "B3LYP", k_builder="cosx", df_j_aux=""
    ).total_energy
    assert abs(rij - exact_j) > 1e-6, "RI-J is not active under k_builder='cosx'"
    with pytest.raises(RuntimeError, match="df_k_aux"):
        ferric.run_dft(
            mol, bs, "B3LYP", k_builder="cosx", df_k_aux="def2-universal-jkfit"
        )
