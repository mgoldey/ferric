"""k-point closed-shell KS-DFT binding (`run_rks_kpts`, src/pbc.rs).

Physics validation (PySCF 2.13 KRKS pins, k-mesh == supercell/N) lives in
crates/ferric-pbc/tests/pbc_krks.rs; here the binding must reach that driver
with the same inputs and refuse the unsupported combinations by name.
Reuses the H2 / cubic a = 4 Bohr fixtures of test_pbc_bindings.py.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

from test_pbc_bindings import (  # noqa: E402,F401  (fixtures are re-exported)
    OMEGA,
    _cubic,
    basis,
    h2,
    h_atom,
)

# Small grid keeps these dense-AFT runs fast; both sides use the same grid.
GRID = {"n_radial": 40, "n_angular": 110}


@pytest.mark.parametrize("functional", ["LDA", "PBE", "PBE0"])
def test_gamma_mesh_equals_the_gamma_binding(h2, basis, functional):
    # k-mesh 1x1x1 (Gamma-centred) is the Gamma point: pbc_krks.rs pins the
    # library at 1e-9 Ha.
    g = ferric.run_rks_gamma(h2, _cubic(), basis, functional, omega=OMEGA, **GRID)
    k = ferric.run_rks_kpts(
        h2, _cubic(), basis, functional, (1, 1, 1), omega=OMEGA, **GRID
    )
    assert k.converged and k.nk == 1 and k.mesh == (1, 1, 1)
    assert k.functional == functional and k.n_grid_points == g.n_grid_points
    assert abs(k.energy - g.energy) < 1e-9, f"{k.energy!r} vs {g.energy!r}"
    assert abs(k.e_xc - g.e_xc) < 1e-9
    assert k.exact_exchange_fraction == g.exact_exchange_fraction


def test_two_k_mesh_runs_and_reports_the_mesh(h2, basis):
    r = ferric.run_rks_kpts(h2, _cubic(), basis, "PBE", (1, 1, 2), omega=OMEGA, **GRID)
    assert r.converged and r.nk == 2 and len(r.mo_energy) == 2 and len(r.kpts) == 2
    assert r.homo < r.lumo


def test_unsupported_combinations_are_refused_by_name(h2, h_atom, basis):
    lat = _cubic()
    with pytest.raises(ValueError, match="multiplicity 2"):
        ferric.run_rks_kpts(h_atom, lat, basis, "LDA", (1, 1, 2))
    with pytest.raises(ValueError, match="meta-GGA"):
        ferric.run_rks_kpts(h2, lat, basis, "SCAN", (1, 1, 2))
    with pytest.raises(ValueError, match="range-separated"):
        ferric.run_rks_kpts(h2, lat, basis, "HSE06", (1, 1, 2))
    with pytest.raises(ValueError, match="forces and stress"):
        ferric.run_rks_kpts(h2, lat, basis, "LDA", (1, 1, 2), with_gradient=True)
    with pytest.raises(ValueError, match="forces and stress"):
        ferric.run_rks_kpts(h2, lat, basis, "LDA", (1, 1, 2), with_stress=True)
