"""`range_split`: the opt-in RS-GDF range split through the Python surface.

The Rust suite (crates/ferric-pbc/tests/pbc_rsgdf_split.rs) owns the physics:
the moves-nothing anchor (bitwise), the exact-span anchor with everything
moved, the G = 0 mutants and the split-vs-unsplit agreement. This file checks
that the bindings THREAD the kwarg into the build and refuse it where the
split is not implemented:

* range_split=None and a lambda below every exponent give bit-identical
  energies (the split machinery runs but moves nothing; the counters prove
  it ran, so the agreement is not "the kwarg was dropped");
* lambda = 1 on H2/STO-3G (PySCF digits), a = 4 Bohr, cc-pvdz-ri moves
  primitives (counters > 0; otherwise the agreement below proves nothing)
  and agrees with the unsplit energy to 1e-9 (Rust: max|dI| < 1e-10 on the
  same system);
* True == lambda 1, False == None;
* the k-point bindings (run_rhf_kpts / run_uhf_kpts, 1x1x2) thread the
  kwarg the same way: a lambda below every exponent is bitwise the unsplit
  k energy (the split counters prove the split ran), lambda = 1 moves
  primitives and agrees with the unsplit k energy to 1e-9 (Rust,
  crates/ferric-pbc/tests/pbc_krsgdf_split.rs, owns the physics);
* refusals (jk="dense", with_gradient / with_stress, lambda <= 0) raise
  ValueError, a non-number raises TypeError.
"""

from __future__ import annotations

import json

import pytest

import ferric

BOHR_IN_ANGSTROM = 0.52917721092  # 1 / ferric's ANGSTROM_TO_BOHR
H2_ATOMS_BOHR = [(0.3, 0.2, 0.1), (0.3, 0.2, 1.5)]
A_BOHR = 4.0
AUX = "cc-pvdz-ri"

PYSCF_STO3G_H = {
    "name": "pyscf-sto-3g-H",
    "elements": {
        "1": {
            "electron_shells": [
                {
                    "function_type": "gto",
                    "angular_momentum": [0],
                    "exponents": ["3.42525091", "0.62391373", "0.1688554"],
                    "coefficients": [["0.15432897", "0.53532814", "0.44463454"]],
                }
            ]
        }
    },
}

SPLIT_COUNTERS = (
    "rsgdf split orbital prims",
    "rsgdf split orbital prims smooth",
    "rsgdf split aux prims",
    "rsgdf split aux prims smooth",
)


def _lattice():
    a = A_BOHR * BOHR_IN_ANGSTROM
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    p = tmp_path_factory.mktemp("basis") / "pyscf-sto-3g-h.json"
    p.write_text(json.dumps(PYSCF_STO3G_H))
    return ferric.BasisSet.from_bse_json(str(p))


@pytest.fixture(scope="module")
def h2():
    lines = [str(len(H2_ATOMS_BOHR)), "range split test (Angstrom)"]
    for x, y, z in H2_ATOMS_BOHR:
        lines.append(
            f"H {x * BOHR_IN_ANGSTROM!r} {y * BOHR_IN_ANGSTROM!r} "
            f"{z * BOHR_IN_ANGSTROM!r}"
        )
    return ferric.Molecule.from_xyz_string("\n".join(lines) + "\n")


def _rhf(h2, basis, **kw):
    return ferric.run_rhf_gamma(h2, _lattice(), basis, jk="rsgdf", auxbasis=AUX, **kw)


@pytest.fixture(scope="module")
def unsplit(h2, basis):
    return _rhf(h2, basis)


def _counters(r):
    return r.timings["counters"]


def test_unsplit_build_runs_no_split_path(unsplit):
    assert unsplit.converged
    for name in SPLIT_COUNTERS:
        assert name not in _counters(unsplit), name


def test_a_split_that_moves_nothing_is_bitwise_the_unsplit_energy(h2, basis, unsplit):
    # Orbital threshold lambda*omega^2/2 = 5e-13, aux 1e-12 (omega_gdf = 1):
    # below every exponent, so nothing moves.
    r = _rhf(h2, basis, range_split=1e-12)
    c = _counters(r)
    assert c["rsgdf split orbital prims"] > 0 and c["rsgdf split aux prims"] > 0, c
    assert c["rsgdf split orbital prims smooth"] == 0, c
    assert c["rsgdf split aux prims smooth"] == 0, c
    assert r.energy == unsplit.energy


def test_lambda_one_moves_blocks_and_matches_the_unsplit_energy(h2, basis, unsplit):
    r = _rhf(h2, basis, range_split=1.0)
    c = _counters(r)
    assert r.converged
    assert c["rsgdf split orbital prims smooth"] > 0, c
    assert c["rsgdf split aux prims smooth"] > 0, c
    assert abs(r.energy - unsplit.energy) <= 1e-9, r.energy - unsplit.energy


def test_true_is_lambda_one_and_false_is_off(h2, basis, unsplit):
    one = _rhf(h2, basis, range_split=1.0)
    assert _rhf(h2, basis, range_split=True).energy == one.energy
    off = _rhf(h2, basis, range_split=False)
    assert off.energy == unsplit.energy
    assert "rsgdf split orbital prims" not in _counters(off)


def test_gamma_uhf_threads_the_split(h2, basis):
    kw = dict(jk="rsgdf", auxbasis=AUX)
    e0 = ferric.run_uhf_gamma(h2, _lattice(), basis, **kw)
    e1 = ferric.run_uhf_gamma(h2, _lattice(), basis, range_split=1e-12, **kw)
    assert "rsgdf split orbital prims" in e1.timings["counters"]
    assert e1.energy == e0.energy


def test_dense_jk_refuses_a_split(h2, basis):
    with pytest.raises(ValueError, match="range_split"):
        ferric.run_rhf_gamma(h2, _lattice(), basis, range_split=True)
    with pytest.raises(ValueError, match="range_split"):
        ferric.run_uhf_gamma(h2, _lattice(), basis, range_split=1.0)
    with pytest.raises(ValueError, match="range_split"):
        ferric.run_mp2_gamma(
            h2, _lattice(), basis, "ewald", "shifted", range_split=True
        )


@pytest.mark.parametrize("kw", [{"with_gradient": True}, {"with_stress": True}])
def test_derivatives_refuse_a_split(h2, basis, kw):
    with pytest.raises(ValueError, match="range_split"):
        _rhf(h2, basis, range_split=True, **kw)
    with pytest.raises(ValueError, match="range_split"):
        ferric.run_uhf_gamma(
            h2, _lattice(), basis, jk="rsgdf", auxbasis=AUX, range_split=1.0, **kw
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_lambda_is_refused(h2, basis, bad):
    with pytest.raises(ValueError, match="range_split"):
        _rhf(h2, basis, range_split=bad)


def test_non_number_is_a_type_error(h2, basis):
    with pytest.raises(TypeError, match="range_split"):
        _rhf(h2, basis, range_split="yes")


KMESH = (1, 1, 2)


def _krhf(h2, basis, **kw):
    return ferric.run_rhf_kpts(
        h2, _lattice(), basis, KMESH, jk="rsgdf", auxbasis=AUX, **kw
    )


@pytest.fixture(scope="module")
def k_unsplit(h2, basis):
    return _krhf(h2, basis)


def test_kpoint_split_that_moves_nothing_is_bitwise_the_unsplit_energy(
    h2, basis, k_unsplit
):
    assert k_unsplit.converged
    for name in SPLIT_COUNTERS:
        assert name not in _counters(k_unsplit), name
    r = _krhf(h2, basis, range_split=1e-12)
    c = _counters(r)
    assert c["rsgdf split orbital prims"] > 0 and c["rsgdf split aux prims"] > 0, c
    assert c["rsgdf split orbital prims smooth"] == 0, c
    assert c["rsgdf split aux prims smooth"] == 0, c
    assert r.energy == k_unsplit.energy


def test_kpoint_lambda_one_moves_blocks_and_matches_the_unsplit_energy(
    h2, basis, k_unsplit
):
    r = _krhf(h2, basis, range_split=1.0)
    c = _counters(r)
    assert r.converged
    assert c["rsgdf split orbital prims smooth"] > 0, c
    assert c["rsgdf split aux prims smooth"] > 0, c
    assert abs(r.energy - k_unsplit.energy) <= 1e-9, r.energy - k_unsplit.energy


def test_kpoint_uhf_threads_the_split(h2, basis):
    kw = dict(jk="rsgdf", auxbasis=AUX)
    e0 = ferric.run_uhf_kpts(h2, _lattice(), basis, KMESH, **kw)
    e1 = ferric.run_uhf_kpts(h2, _lattice(), basis, KMESH, range_split=1e-12, **kw)
    assert "rsgdf split orbital prims" in e1.timings["counters"]
    assert e1.energy == e0.energy


def test_kpoint_dense_jk_refuses_a_split(h2, basis):
    with pytest.raises(ValueError, match="range_split"):
        ferric.run_rhf_kpts(h2, _lattice(), basis, KMESH, range_split=True)
    with pytest.raises(ValueError, match="range_split"):
        ferric.run_uhf_kpts(h2, _lattice(), basis, KMESH, range_split=1.0)
