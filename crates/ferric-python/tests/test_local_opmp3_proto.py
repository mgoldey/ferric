"""Local OP-MP3 prototype (scripts/prototypes/local_opmp3_proto.py): anchors.

Propane/STO-3G only, to keep the suite fast; the prototype's own sweeps
cover the rest (scripts/prototypes/local_opmp3_derivation.md).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyscf")

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "local_opmp3_proto", ROOT / "scripts/prototypes/local_opmp3_proto.py"
)
lop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lop)


@pytest.fixture(scope="module")
def propane():
    sys_ = lop.System(str(ROOT / "testdata/molecules/alkane_3.xyz"), "sto-3g")
    w_can = lop.canonical(sys_)
    ints = lop.local_ints(sys_)
    return sys_, w_can, ints


def test_local_opmp3_matches_exact_in_the_trivial_limit(propane):
    sys_, w_can, ints = propane
    ok, worst = lop.anchor_eps0(sys_, ints, w_can)
    assert ok, worst


def test_anchor_fails_without_the_fock_off_diagonals(propane):
    sys_, w_can, ints = propane
    ok, worst = lop.anchor_eps0(sys_, ints, w_can, mutate="semicanonical")
    assert not ok and worst > 1e-6


def test_trace_pade_is_invariant_under_occupied_rotation(propane):
    _, w_can, _ = propane
    assert lop.anchor_invariance(w_can) <= 1e-13


def test_e3_identity_holds_on_one_mask_and_fails_on_two(propane):
    sys_, _, ints = propane
    _, info = lop.local_run(sys_, ints, 1e-4)["mask"]
    assert abs(info["e3_identity_gap"]) <= info["e3_identity_bound"]
    _, bad = lop.local_run(sys_, ints, 1e-4, mutate="c2_mask_shift")["mask"]
    assert abs(bad["e3_identity_gap"]) > bad["e3_identity_bound"]


def test_finite_eps_e2_matches_the_lmp2_prototype(propane):
    sys_, _, ints = propane
    _, info = lop.local_run(sys_, ints, 1e-4, check_lmp2=True)["mask"]
    assert info["keep"] < 0.9  # the mask is active
    assert abs(info["e2"] - info["e2_lmp2_proto"]) <= lop.LMP2_BAR


def test_canonical_formulas_match_ferric_run_opmp3(propane):
    ferric = pytest.importorskip("ferric")
    if not hasattr(ferric, "run_opmp3"):
        pytest.skip("loaded ferric build has no run_opmp3")
    sys_, _, _ = propane
    dev_e, dev_w, _, _, _ = lop.ferric_cross_check(sys_)
    assert max(abs(v) for v in dev_e.values()) <= 1e-12, dev_e
    assert max(dev_w.values()) <= 1e-12, dev_w
    bad_e, _, _, _, _ = lop.ferric_cross_check(sys_, drop_term=7)
    assert abs(bad_e["mp3"]) > 1e-6
    assert np.isfinite(bad_e["op-mp3"])


@pytest.fixture(scope="module")
def butane_df():
    sys_ = lop.System(
        str(ROOT / "testdata/molecules/alkane_4.xyz"), "6-31g", auxbasis="cc-pvdz-ri"
    )
    return sys_, lop.local_ints(sys_)


def test_df_local_opmp3_matches_exact_in_the_trivial_limit(butane_df):
    sys_, ints = butane_df
    w_can = lop.canonical(sys_)
    assert sys_.canonical_ladder == "dense B.B"  # independent of df_ladder
    ok, worst = lop.anchor_eps0(sys_, ints, w_can)
    assert ok, worst


def test_df_domain_ladder_matches_dense_and_catches_a_domain_mixup(butane_df):
    sys_, ints = butane_df
    dr, rmax, keep = lop.df_domain_check(sys_, ints, 1e-3)
    assert keep < 0.2  # domains are truncated, so a mix-up can show
    assert dr <= lop.DF_DOMAIN_BAR * rmax
    bad, _, _ = lop.df_domain_check(sys_, ints, 1e-3, mutate="pair_domain_ii")
    assert bad > 1e-6 * rmax


@pytest.fixture(scope="module")
def butane_new_maps():
    """n-butane/6-31G with ferric's current LMP2 maps: converged Boys +
    lambda-Boys (lambda = 10) occupieds and the Fock-propagated screen."""
    lop.set_maps("new")
    try:
        sys_ = lop.System(str(ROOT / "testdata/molecules/alkane_4.xyz"), "6-31g")
    finally:
        lop.set_maps("old")
    return sys_, lop.canonical(sys_), lop.local_ints(sys_)


def test_lambda_boys_is_converged_and_raises_the_objective(butane_new_maps):
    sys_, _, _ = butane_new_maps
    lb = sys_.lambda_boys
    assert lb["max_grad"] < lop.LAMBDA_BOYS_GRAD_TOL
    assert lb["G_end"] >= lb["G_start"]
    assert lb["phi_lambda"] < lb["phi_boys"]  # F_oo is more diagonal


def test_lambda_boys_mutation_is_caught(butane_new_maps):
    sys_, _, _ = butane_new_maps
    r1 = sys_.mol.intor("int1e_r")
    c = sys_.co_boys  # the converged Boys start
    x = [c.T @ r1[k] @ c for k in range(3)]
    f = sys_.u_occ.T @ np.diag(sys_.eo) @ sys_.u_occ  # F_oo in the lambda orbitals
    lam_to_boys = c.T @ sys_.mol.intor("int1e_ovlp") @ sys_.co_loc
    f = lam_to_boys @ f @ lam_to_boys.T  # ... rotated back to the Boys start
    _, _, mg_ok, g0, g1_ok = lop.lambda_boys_rotation(x, f, 10.0)
    assert mg_ok < lop.LAMBDA_BOYS_GRAD_TOL and g1_ok > g0
    _, _, mg, _, g1 = lop.lambda_boys_rotation(x, f, 10.0, mutate="sign")
    assert mg >= lop.LAMBDA_BOYS_GRAD_TOL and g1 < g0


def test_new_maps_local_opmp3_matches_exact_in_the_trivial_limit(butane_new_maps):
    sys_, w_can, ints = butane_new_maps
    lop.set_maps("new")
    try:
        ok, worst = lop.anchor_eps0(sys_, ints, w_can)
        _, info = lop.local_run(sys_, ints, 1e-4)["mask"]
    finally:
        lop.set_maps("old")
    assert ok, worst
    assert abs(info["e3_identity_gap"]) <= info["e3_identity_bound"]


def test_second_order_screen_keeps_the_anchor_and_one_mask(butane_new_maps):
    """The 'new-r2' arm (screen on R(c1) too): inert at eps = 0, adds
    elements at finite eps, and keeps c1 and c2 on one mask (E3 identity)."""
    sys_, w_can, ints = butane_new_maps
    lop.set_maps("new-r2")
    try:
        ok, worst = lop.anchor_eps0(sys_, ints, w_can)
        _, info = lop.local_run(sys_, ints, 1e-4)["mask"]
    finally:
        lop.set_maps("old")
    assert ok, worst
    assert info["r2_added_frac"] > 0.0
    assert abs(info["e3_identity_gap"]) <= info["e3_identity_bound"]


def test_pbe0_reference_new_maps_match_exact_in_the_trivial_limit():
    """KS reference (HF Fock of the PBE0 density, semicanonical, NBS singles
    outside the mask): eps = 0 with the new maps equals canonical, and the E3
    identity holds at finite eps."""
    lop.set_maps("new")
    try:
        sys_ = lop.System(
            str(ROOT / "testdata/molecules/alkane_3.xyz"), "6-31g", reference="pbe0"
        )
        w_can = lop.canonical(sys_)
        ints = lop.local_ints(sys_)
        ok, worst = lop.anchor_eps0(sys_, ints, w_can)
        _, info = lop.local_run(sys_, ints, 1e-4)["mask"]
        bad, _ = lop.anchor_eps0(sys_, ints, w_can, mutate="semicanonical")
    finally:
        lop.set_maps("old")
    assert ok, worst
    assert not bad  # the anchor can fail on this reference
    assert abs(info["e3_identity_gap"]) <= info["e3_identity_bound"]
    assert sys_.e_nbs < 0.0  # a KS reference carries (stabilizing) non-Brillouin singles
