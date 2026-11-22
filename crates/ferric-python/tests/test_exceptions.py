"""The `ferric` exception hierarchy (ARCHITECTURE.md, Errors; roadmap item 10).

Every class derives from `ferric.FerricError` AND from the builtin class the
same failure raised before the hierarchy existed, with the message unchanged,
so existing `except ValueError` / `except RuntimeError` /
`except NotImplementedError` handlers keep catching what they caught. The
mapping from `FerricError` variants to classes is one function
(`errors::from_core`); `ferric.ferric._raise_core_error` runs each variant through it,
because some variants (`Unsupported`, `Lapack`, `Io`) are reached from Python
only through refusals no public keyword triggers (for example the local OP-MP3
assembly options). The rest are checked with real calls.
"""

import pickle
import re
import warnings

import pytest

import ferric

WATER = (
    "3\n\nO 0.000000 0.000000 0.117790\n"
    "H 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n"
)

ALL = [
    ferric.ConvergenceError,
    ferric.BasisError,
    ferric.NumericalError,
    ferric.MemoryBudgetError,
    ferric.UnsupportedError,
    ferric.ComputationError,
    ferric.InputError,
]


@pytest.fixture(scope="module")
def water(tmp_path_factory):
    path = tmp_path_factory.mktemp("exc") / "w.xyz"
    path.write_text(WATER)
    return ferric.Molecule.from_xyz(str(path), 0, 1)


def test_hierarchy_shape():
    assert issubclass(ferric.FerricError, Exception)
    assert not issubclass(ferric.FerricError, (RuntimeError, ValueError))
    for cls in ALL:
        assert issubclass(cls, ferric.FerricError), cls
        assert cls.__module__ == "ferric", cls
    for cls in (
        ferric.ConvergenceError,
        ferric.NumericalError,
        ferric.MemoryBudgetError,
        ferric.ComputationError,
    ):
        assert issubclass(cls, RuntimeError) and not issubclass(cls, ValueError)
    assert issubclass(ferric.UnsupportedError, NotImplementedError)
    assert issubclass(ferric.InputError, ValueError)
    assert not issubclass(ferric.InputError, RuntimeError)
    # The one dual class: a bad basis was a ValueError from `BasisSet.bundled`
    # and a RuntimeError from the drivers.
    assert issubclass(ferric.BasisError, ferric.InputError)
    assert issubclass(ferric.BasisError, ValueError)
    assert issubclass(ferric.BasisError, RuntimeError)


# variant -> (class, builtin it was raised as before, exact message)
CORE_TABLE = {
    "io": (ferric.ComputationError, RuntimeError, "IO error: boom"),
    "xyz_parse": (ferric.ComputationError, RuntimeError, "XYZ parse error: boom"),
    "basis": (ferric.BasisError, RuntimeError, "basis error: boom"),
    "libint": (ferric.ComputationError, RuntimeError, "libint error: boom"),
    "scf_convergence": (
        ferric.ConvergenceError,
        RuntimeError,
        "SCF did not converge after 7 iterations (last energy: -1.5000000000)",
    ),
    "lapack": (ferric.NumericalError, RuntimeError, "LAPACK error: boom"),
    "convergence": (ferric.ConvergenceError, RuntimeError, "Convergence error: boom"),
    "general": (ferric.ComputationError, RuntimeError, "General error: boom"),
    "memory_exhausted": (ferric.MemoryBudgetError, RuntimeError, "boom"),
    "unsupported": (
        ferric.UnsupportedError,
        NotImplementedError,
        "a feature is not supported: boom",
    ),
}


@pytest.mark.parametrize("variant", sorted(CORE_TABLE))
def test_every_core_variant_maps_to_its_class_and_keeps_its_builtin(variant):
    cls, builtin, message = CORE_TABLE[variant]
    with pytest.raises(builtin) as exc:
        ferric.ferric._raise_core_error(variant, "boom")
    assert type(exc.value) is cls
    assert isinstance(exc.value, ferric.FerricError)
    assert str(exc.value) == message
    assert exc.value.args == (message,)


def test_core_table_names_every_variant():
    # The names come from `FerricError::samples` (one per variant, in
    # ferric-core), so a new variant must be added to CORE_TABLE here too.
    assert sorted(ferric.ferric._core_error_variants()) == sorted(CORE_TABLE)
    with pytest.raises(ValueError, match="unknown core error variant"):
        ferric.ferric._raise_core_error("nope", "x")
    # The class a variant lands in is the only thing that may differ.
    assert {c for c, _, _ in CORE_TABLE.values()} == set(ALL) - {ferric.InputError}


@pytest.mark.parametrize("cls", ALL + [ferric.FerricError])
def test_pickle_round_trip(cls):
    exc = cls("message")
    back = pickle.loads(pickle.dumps(exc))
    assert type(back) is cls
    assert back.args == ("message",)
    assert type(pickle.loads(pickle.dumps(cls))) is type(cls)


def test_pickle_of_a_raised_exception():
    with pytest.raises(ferric.FerricError) as exc:
        ferric.ferric._raise_core_error("scf_convergence", "")
    back = pickle.loads(pickle.dumps(exc.value))
    assert type(back) is ferric.ConvergenceError
    assert str(back) == str(exc.value)


def test_scf_non_convergence_is_a_convergence_error(water):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError) as exc:
            ferric.run_frequencies(water, "sto-3g", scf_max_iter=1)
    assert type(exc.value) is ferric.ConvergenceError
    assert re.fullmatch(
        r"SCF did not converge after 1 iterations \(last energy: -74\.\d{10}\)",
        str(exc.value),
    )


def test_bad_basis_name_is_a_basis_error_both_builtins_catch():
    with pytest.raises(ValueError) as exc:
        ferric.BasisSet.bundled("no-such-basis")
    assert type(exc.value) is ferric.BasisError
    assert isinstance(exc.value, RuntimeError)
    assert str(exc.value).startswith(
        "basis error: unknown bundled basis: no-such-basis"
    )


def test_bad_aux_basis_name_from_a_driver_is_a_basis_error(water):
    bs = ferric.BasisSet.bundled("sto-3g")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError) as exc:
            ferric.run_rhf(water, bs, df_j_aux="no-such-aux", df_k_aux="no-such-aux")
    assert type(exc.value) is ferric.BasisError
    assert isinstance(exc.value, ValueError)
    assert str(exc.value).startswith("basis error: unknown bundled basis: no-such-aux")


def test_unreadable_basis_file_is_an_input_error():
    with pytest.raises(ValueError) as exc:
        ferric.BasisSet.from_bse_json("/nonexistent/basis.json")
    assert type(exc.value) is ferric.InputError
    assert not isinstance(exc.value, RuntimeError)
    assert str(exc.value).startswith("General error: cannot read basis file")


def test_invalid_option_values_are_input_errors(water):
    bs = ferric.BasisSet.bundled("sto-3g")
    with pytest.raises(ValueError) as exc:
        ferric.run_rhf(water, bs, guess="no-such-guess")
    assert type(exc.value) is ferric.InputError
    assert not isinstance(exc.value, RuntimeError)
    assert "unknown guess 'no-such-guess'" in str(exc.value)
    with pytest.raises(ValueError, match='unknown reference "roks"') as exc:
        ferric.run_frequencies(water, "sto-3g", reference="roks")
    assert type(exc.value) is ferric.InputError


def test_memory_budget_refusal_is_a_memory_budget_error(water):
    bs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="budget is") as exc:
            ferric.run_ccsd(water, bs, aux, memory_budget_gb=3.8e-3, ladder="dense")
    assert type(exc.value) is ferric.MemoryBudgetError


def test_other_failures_stay_runtime_errors_in_the_hierarchy(water):
    """`FerricError::General` carries untyped text (here an unknown
    functional), so it is a `ComputationError`: still a RuntimeError, still a
    FerricError, never a ValueError."""
    bs = ferric.BasisSet.bundled("sto-3g")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="NOPE") as exc:
            ferric.run_dft(water, bs, "NOPE")
    assert type(exc.value) is ferric.ComputationError
    assert not isinstance(exc.value, ValueError)


def test_argument_type_errors_stay_plain_type_errors(water):
    bs = ferric.BasisSet.bundled("sto-3g")
    with pytest.raises(TypeError) as exc:
        ferric.run_rhf(water, bs, diis=-1)
    assert not isinstance(exc.value, ferric.FerricError)


def test_prefixed_basis_failures_are_still_basis_errors():
    """`BasisSet.shells` and the auxbasis lookup keep the typed variant through
    a prefixed message."""
    bs = ferric.BasisSet.bundled("sto-3g")
    with pytest.raises(ValueError) as exc:
        bs.shells(92)
    assert type(exc.value) is ferric.BasisError


def test_helper_names_stay_out_of_the_public_surface():
    for name in ("_raise_core_error", "_core_error_variants"):
        assert name not in ferric.__all__
        assert not hasattr(ferric, name)
        assert hasattr(ferric.ferric, name)
