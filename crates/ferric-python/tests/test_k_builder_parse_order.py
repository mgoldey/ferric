"""Where an unknown `k_builder=` spelling is refused, relative to other errors.

Roadmap 15 parses `k_builder` at the binding (`KBuilderKind::parse`) instead
of in the SCF. The class and text are unchanged (`RuntimeError`, "General
error: unknown k_builder ..."), and each entry point parses it last of its
keywords, so a bad keyword parsed before it (guess, dm0 shape, a
fock_hook_grad without a hook, the cosx_* knobs, a CC ladder, run_dft's
symmetry scoping) and the basis preparation still win. What ran between the
keywords and the SCF before (the with_gradient / hook refusals, the libxc
functional lookup, and in the MP2 family the method keywords) now comes
after it: with two bad inputs the class can change from ValueError to
RuntimeError.
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")

XYZ = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
UNKNOWN = (
    "General error: unknown k_builder 'x': valid options are 'direct', 'link', 'cosx'"
)
GUESS = (
    "General error: unknown guess 'nope': valid options are 'minao' (default), 'sad' "
    "(alias of 'minao': the free-atom-SCF SAD guess is not selectable from config) and "
    "'hcore'"
)


def grad_only(who: str) -> str:
    return f"{who}: fock_hook_grad needs a fock_hook (it is the explicit derivative of the hook's energy)"


@pytest.fixture(scope="module")
def systems():
    w = ferric.Molecule.from_xyz_string(XYZ, 0, 1)
    wc = ferric.Molecule.from_xyz_string(XYZ, 1, 2)
    hi = ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nI 0 0 1.61\n", 0, 1)
    b = ferric.BasisSet.bundled
    return w, wc, hi, b("sto-3g"), b("cc-pvdz"), b("cc-pvdz-ri")


def hook_r(d):
    return 0.0, np.zeros_like(d)


def hook_u(a, b):
    return 0.0, np.zeros_like(a), np.zeros_like(b)


def grad(*_):
    return np.zeros((3, 3))


def raises(call, cls, msg):
    with pytest.raises(cls) as e:
        call()
    assert type(e.value) is cls
    assert str(e.value) == msg


def test_alone_is_the_old_runtime_error(systems):
    w, wc, _, s, _, ri = systems
    for call in (
        lambda: ferric.run_rhf(w, s, k_builder="x"),
        lambda: ferric.run_dft(w, s, "pbe", k_builder="x"),
        lambda: ferric.run_uhf(wc, s, k_builder="x"),
        lambda: ferric.run_uks(wc, s, "pbe", k_builder="x"),
        lambda: ferric.run_ccsd(w, s, ri, k_builder="x"),
        lambda: ferric.run_rimp2(w, s, ri, k_builder="x"),
    ):
        raises(call, RuntimeError, UNKNOWN)


def test_keywords_parsed_before_it_still_win(systems):
    w, wc, hi, s, dz, ri = systems
    cases = [
        (lambda: ferric.run_rhf(w, s, k_builder="x", guess="nope"), ValueError, GUESS),
        (
            lambda: ferric.run_rhf(w, s, k_builder="x", dm0=np.zeros((2, 2))),
            ValueError,
            "run_rhf: dm0 has shape (2, 2), expected (7, 7)",
        ),
        (
            lambda: ferric.run_rhf(w, s, k_builder="x", fock_hook_grad=grad),
            ValueError,
            grad_only("run_rhf"),
        ),
        (
            lambda: ferric.run_rhf(hi, dz, k_builder="x"),
            RuntimeError,
            'basis error: basis "cc-pvdz" has no shells for I (Z=53); it covers H-Ar, Ca-Kr',
        ),
        (
            lambda: ferric.run_dft(w, s, "pbe", k_builder="x", guess="nope"),
            ValueError,
            GUESS,
        ),
        (
            lambda: ferric.run_dft(
                w, s, "pbe", k_builder="x", symmetry=True, with_gradient=True
            ),
            ValueError,
            "run_dft(symmetry=True): with_gradient is not available; the blocked SCF runs in "
            "the reoriented, symmetrized frame. Use symmetry=False for gradients.",
        ),
        (
            lambda: ferric.run_dft(w, s, "pbe", k_builder="x", irrep_nelec={"A1": 10}),
            ValueError,
            "run_dft: irrep_nelec requires symmetry=True",
        ),
        (lambda: ferric.run_uhf(wc, s, k_builder="x", guess="nope"), ValueError, GUESS),
        (
            lambda: ferric.run_uhf(wc, s, k_builder="x", fock_hook_grad=grad),
            ValueError,
            grad_only("run_uhf"),
        ),
        (
            lambda: ferric.run_uks(wc, s, "pbe", k_builder="x", fock_hook_grad=grad),
            ValueError,
            grad_only("run_uks"),
        ),
        (
            lambda: ferric.run_uks(wc, s, "pbe", k_builder="x", cosx_overlap_fit=False),
            ValueError,
            "cosx_grid / cosx_final_pass / cosx_final_grid / cosx_overlap_fit are only read with "
            'k_builder="cosx" (got k_builder=Some("x"))',
        ),
        (
            lambda: ferric.run_ccsd(w, s, ri, k_builder="x", ladder="zz"),
            ValueError,
            'General error: unknown CCSD ladder "zz": expected "auto", "dense" or "ri"',
        ),
        (
            lambda: ferric.run_ccsd(
                hi, ferric.BasisSet.bundled("def2-svp"), ri, k_builder="x"
            ),
            RuntimeError,
            'basis error: basis "cc-pvdz-ri" has no shells for I (Z=53); it covers H-Ar, Ga-Kr',
        ),
    ]
    for call, cls, msg in cases:
        raises(call, cls, msg)


def test_it_beats_what_ran_between_the_keywords_and_the_scf(systems):
    w, wc, _, s, _, ri = systems
    for call in (
        lambda: ferric.run_rhf(w, s, k_builder="x", symmetry=True),
        lambda: ferric.run_rhf(
            w, s, k_builder="x", irrep_nelec={"A1": 10}, with_gradient=True
        ),
        lambda: ferric.run_rhf(
            w, s, k_builder="x", smearing_sigma=0.01, with_gradient=True
        ),
        lambda: ferric.run_dft(w, s, "nope", k_builder="x"),
        lambda: ferric.run_dft(
            w, s, "pbe", k_builder="x", fock_hook=hook_r, with_gradient=True
        ),
        lambda: ferric.run_uhf(
            wc, s, k_builder="x", fock_hook=hook_u, with_gradient=True
        ),
        lambda: ferric.run_uks(wc, s, "nope", k_builder="x"),
        lambda: ferric.run_rimp2(w, s, ri, k_builder="x", local="nope"),
    ):
        raises(call, RuntimeError, UNKNOWN)
