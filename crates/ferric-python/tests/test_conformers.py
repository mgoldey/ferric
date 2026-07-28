"""Regression suite for ferric's Python bindings.

Scope: the conformer-ensemble entry points (the RDKit-facing API), the unit
convention that separates them from the rest of ferric, the Boltzmann /
weighted-statistics numerics, and the error paths -- plus a smoke check on the
already-exposed Molecule / BasisSet / run_rhf surface.

WHY THIS EXISTS. `cargo test -p ferric-python` runs zero tests: the crate is
`crate-type = ["cdylib"]`, so there is no linkable Rust harness. Everything the
bindings do is therefore only reachable from Python, and until this file existed
nothing guarded it.

WHAT IS WORTH TESTING HERE. Two failure classes dominate, and both are silent:

1.  A unit error. RDKit hands out Angstrom, ferric stores Bohr. A dropped or
    doubled conversion produces a geometry that is wrong by 1.889726x, which
    still runs, still converges, and still returns a plausible-looking energy.
    Nothing crashes. `test_units_*` pin this from several independent angles.

2.  A panic crossing the pyo3 boundary. A Rust `panic!` reaching Python is not
    an exception the caller can catch -- in the general case it aborts the host
    interpreter. So every invariant violation must surface as a real Python
    exception. `test_error_*` walks the input-validation surface and asserts a
    catchable exception of the right class, which also proves the process
    survived to run the next assertion.
"""

import json
import math
import os

import numpy as np
import pytest

import ferric

from conftest import (
    BOHR_PER_ANGSTROM,
    DEFAULT_T,
    KB_HARTREE_PER_K,
    KT_AT_DEFAULT_T,
    WATER_ANGSTROM,
    WATER_SYMBOLS,
    nuclear_repulsion_from_bohr,
    water_xyz_string,
)

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testdata")


# ─────────────────────────────────────────────────────────────────────────────
# 1. from_coordinates -- construction and coordinate round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_from_coordinates_basic_shape(water_ensemble):
    ens = water_ensemble
    assert len(ens) == 1
    assert ens.n_conformers() == 1
    assert ens.n_atoms() == 3
    assert ens.elements() == ["O", "H", "H"]
    assert ens.atomic_numbers() == [8, 1, 1]
    assert ens.is_ghost() == [False, False, False]


def test_from_coordinates_multiple_conformers():
    shifted = [[x, y, z + 0.05] for x, y, z in WATER_ANGSTROM]
    ens = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM, shifted], WATER_SYMBOLS
    )
    assert len(ens) == 2
    assert len(ens.molecules()) == 2
    assert ens.n_atoms() == 3


def test_coordinates_roundtrip_angstrom(water_ensemble):
    """Angstrom in, Angstrom out, to ~1 ulp.

    NOT asserted bit-exact, and that is deliberate rather than a slack
    tolerance: the inbound Angstrom->Bohr step is a floating-point multiply,
    which has no exact inverse, so a round-trip is accurate to about 1 ulp but
    not bit-identical. The binding's own docstring says so. 1e-14 A is far
    below any chemically meaningful displacement while still being ~100x
    tighter than the error a genuinely broken conversion would produce.
    """
    back = water_ensemble.coordinates()
    assert len(back) == 1
    got = np.asarray(back[0])
    assert got.shape == (3, 3)
    assert np.abs(got - np.asarray(WATER_ANGSTROM)).max() < 1e-14


def test_coordinates_roundtrip_is_tight_not_merely_close(water_ensemble):
    """Pin the round-trip error at the ulp scale, not just 'small'.

    A 1e-14 tolerance would also pass if someone introduced an error at 1e-15.
    This asserts the error is at most a few ulp of the coordinate magnitude,
    which is the actual claim being made.
    """
    got = np.asarray(water_ensemble.coordinates()[0])
    want = np.asarray(WATER_ANGSTROM)
    err = np.abs(got - want)
    # A few ulp of ~1 A. np.spacing(1.0) is 2.2e-16.
    assert err.max() <= 4 * np.spacing(1.0)


def test_coordinates_bohr_matches_angstrom_times_factor(water_ensemble):
    """coordinates_bohr() is the stored value, returned with no arithmetic.

    Asserted BIT-EXACT against `angstrom * BOHR_PER_ANGSTROM` computed in
    Python. This is the strongest available statement that the stored numbers
    are the direct product of the input and the conversion factor -- no
    intermediate rounding, no second conversion, no compensation.
    """
    got = np.asarray(water_ensemble.coordinates_bohr()[0])
    want = np.asarray(WATER_ANGSTROM) * BOHR_PER_ANGSTROM
    assert np.array_equal(got, want), f"got {got!r}\nwant {want!r}"


def test_elements_accept_atomic_numbers():
    """Symbols and atomic numbers must produce an identical ensemble."""
    by_symbol = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM], ["O", "H", "H"]
    )
    by_number = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM], [8, 1, 1]
    )
    assert by_symbol.elements() == by_number.elements()
    assert by_symbol.atomic_numbers() == by_number.atomic_numbers()
    assert np.array_equal(
        np.asarray(by_symbol.coordinates_bohr()[0]),
        np.asarray(by_number.coordinates_bohr()[0]),
    )
    assert (
        by_symbol.molecule(0).nuclear_repulsion()
        == by_number.molecule(0).nuclear_repulsion()
    )


def test_ghost_atoms_carry_no_electrons():
    """A '@'-prefixed center is basis-only: it keeps its symbol but no charge."""
    ens = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM], ["@O", "H", "H"]
    )
    assert ens.elements() == ["O", "H", "H"]
    assert ens.is_ghost() == [True, False, False]
    # Only the two real hydrogens contribute electrons.
    assert ens.molecule(0).nelec() == 2


def test_from_multi_xyz_reads_every_frame(tmp_path):
    """from_multi_xyz reads all frames; Molecule.from_xyz reads only the first.

    This asymmetry is the reason the method exists, so it is worth pinning:
    a silent regression to first-frame-only would turn a 3-conformer ensemble
    into a 1-conformer one and quietly discard the ensemble average.
    """
    frame = water_xyz_string()
    path = tmp_path / "confs.xyz"
    path.write_text(frame * 3)

    ens = ferric.ConformerEnsemble.from_multi_xyz(str(path))
    assert len(ens) == 3
    assert ens.n_atoms() == 3

    # Molecule.from_xyz on the same file sees only frame 1 -- contrast, not a
    # defect, but it is what makes from_multi_xyz necessary.
    single = ferric.Molecule.from_xyz(str(path))
    assert single.natoms() == 3


def test_from_multi_xyz_matches_from_coordinates(tmp_path):
    """Both geometry entry points must land on identical stored coordinates."""
    path = tmp_path / "one.xyz"
    path.write_text(water_xyz_string())

    from_file = ferric.ConformerEnsemble.from_multi_xyz(str(path))
    from_arrays = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM], WATER_SYMBOLS
    )
    assert np.array_equal(
        np.asarray(from_file.coordinates_bohr()[0]),
        np.asarray(from_arrays.coordinates_bohr()[0]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. UNIT CONVENTION -- the highest-value tests in the file
# ─────────────────────────────────────────────────────────────────────────────
#
# RDKit gives Angstrom. ferric stores Bohr. A missing conversion is a factor of
# 1.889726 in every interatomic distance: the calculation still runs and still
# returns a number. These tests attack the convention from four independent
# directions so that no single-line change can satisfy all of them by accident.


def test_units_stored_values_are_bohr_not_angstrom(water_ensemble):
    """The stored numbers are numerically larger than the Angstrom input.

    The crudest possible check, and the one that would catch a wholesale
    missing conversion. Bohr are smaller units, so the same physical distance
    is a LARGER number in Bohr.
    """
    ang = np.asarray(water_ensemble.coordinates()[0])
    bohr = np.asarray(water_ensemble.coordinates_bohr()[0])
    nonzero = np.abs(ang) > 1e-12
    ratio = bohr[nonzero] / ang[nonzero]
    assert np.allclose(ratio, BOHR_PER_ANGSTROM, rtol=1e-15)
    # And the factor really is the one everyone recognises. ferric derives it
    # as 1/0.529_177_210_92 (CODATA-2010 bohr radius), which differs from the
    # CODATA-2018 value 1.889_726_125_4 in the 9th decimal -- far below any
    # chemical significance, but worth pinning so a change of constant is a
    # deliberate act rather than a silent drift.
    assert abs(BOHR_PER_ANGSTROM - 1.8897261245650618) < 1e-15


def test_units_from_coordinates_agrees_with_parse_xyz(water_ensemble):
    """from_coordinates and Molecule.parse_xyz must store the SAME Bohr values.

    The two entry points are independent code paths that both promise
    "Angstrom in". Nuclear repulsion is the observable used to compare them
    because it is a pure function of the stored coordinates and Molecule
    exposes no coordinate accessor of its own.

    Asserted BIT-IDENTICAL. That is achievable (and was verified) because
    from_coordinates is implemented by formatting an XYZ frame and handing it
    to the very same parse_xyz -- so any divergence at all means that shared
    path was broken.
    """
    from_arrays = water_ensemble.molecule(0)
    from_string = ferric.Molecule.from_xyz_string(water_xyz_string())
    assert (
        from_arrays.nuclear_repulsion() == from_string.nuclear_repulsion()
    ), (
        f"from_coordinates NRE {from_arrays.nuclear_repulsion()!r} != "
        f"parse_xyz NRE {from_string.nuclear_repulsion()!r}"
    )


def test_units_nuclear_repulsion_from_independent_bohr_calculation(
    water_ensemble,
):
    """Recompute NRE in Python from the returned Bohr coords and compare.

    This is the check that closes the loop without trusting any ferric
    internal: if coordinates_bohr() really returns Bohr, then the textbook
    sum Z_i Z_j / r_ij over those numbers must reproduce ferric's own NRE.
    """
    bohr = np.asarray(water_ensemble.coordinates_bohr()[0])
    charges = water_ensemble.atomic_numbers()
    ours = nuclear_repulsion_from_bohr(bohr.tolist(), charges)
    theirs = water_ensemble.molecule(0).nuclear_repulsion()
    assert abs(ours - theirs) < 1e-12, f"{ours!r} vs {theirs!r}"


def test_units_unconverted_input_would_be_detectably_wrong(water_ensemble):
    """The counterfactual: prove the trap is real and this suite would catch it.

    Compute NRE from the raw Angstrom numbers as though they had been stored
    without conversion. The result is a perfectly finite, plausible-looking
    energy -- wrong by exactly the Angstrom->Bohr factor. That is precisely why
    this failure is silent, and why the tests above are the ones worth having.
    """
    charges = water_ensemble.atomic_numbers()
    correct = water_ensemble.molecule(0).nuclear_repulsion()
    if_unconverted = nuclear_repulsion_from_bohr(WATER_ANGSTROM, charges)

    # NRE ~ 1/r, so skipping the conversion scales it by exactly the factor.
    assert abs(if_unconverted / correct - BOHR_PER_ANGSTROM) < 1e-12
    # Plausible-looking: same order of magnitude, finite, positive. Nothing
    # about the value itself announces the error.
    assert math.isfinite(if_unconverted) and if_unconverted > 0
    assert abs(if_unconverted - correct) > 1.0


def test_units_rdkit_positions_pass_straight_through():
    """The documented RDKit contract: GetPositions() needs no pre-scaling.

    Skipped rather than failed when RDKit is absent -- RDKit is not a ferric
    dependency, and its absence says nothing about the bindings.
    """
    rdkit_chem = pytest.importorskip("rdkit.Chem")
    allchem = pytest.importorskip("rdkit.Chem.AllChem")

    mol = rdkit_chem.AddHs(rdkit_chem.MolFromSmiles("CCO"))
    assert allchem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=0xF00D)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = [c.GetPositions() for c in mol.GetConformers()]

    ens = ferric.ConformerEnsemble.from_coordinates(coords, symbols)
    assert len(ens) == len(coords)
    assert ens.n_atoms() == len(symbols)

    # Bohr storage is the exact product of RDKit's Angstrom and the factor.
    for stored, given in zip(ens.coordinates_bohr(), coords):
        assert np.array_equal(np.asarray(stored), given * BOHR_PER_ANGSTROM)

    # And the Angstrom round-trip returns RDKit's own numbers to ~1 ulp.
    for back, given in zip(ens.coordinates(), coords):
        assert np.abs(np.asarray(back) - given).max() < 1e-14


def test_constants_match_module():
    """The literals this suite asserts against are ferric's own.

    conftest hardcodes kB and the default temperature deliberately (a test
    sourcing its expectation from the code under test proves nothing). This is
    the one place the two are reconciled, so a deliberate change to either
    constant fails loudly here rather than silently shifting every weight
    assertion in the file.
    """
    assert ferric.BOLTZMANN_HARTREE_PER_K == KB_HARTREE_PER_K
    assert ferric.DEFAULT_TEMPERATURE_K == DEFAULT_T


# ─────────────────────────────────────────────────────────────────────────────
# 3. Boltzmann weights vs hand-computed values
# ─────────────────────────────────────────────────────────────────────────────


def test_boltzmann_single_conformer_is_exactly_one():
    """One conformer carries the entire population. Exact, not approximate:
    the E_min shift makes its exponent exactly exp(0) = 1 and Z exactly 1."""
    w = ferric.boltzmann_weights([-76.02])
    assert w.weights == [1.0]
    assert w.partition_function == 1.0
    assert w.relative_energies == [0.0]
    assert w.min_index == 0
    assert len(w) == 1


def test_boltzmann_two_degenerate_are_exactly_half():
    """Two equal energies split exactly 0.5/0.5.

    Exact equality is correct here and is not a fragile assertion: both
    exponents are exp(0) = 1, Z is exactly 2.0, and 1.0/2.0 is exact in binary
    floating point.
    """
    w = ferric.boltzmann_weights([-76.02, -76.02])
    assert w.weights == [0.5, 0.5]
    assert w.partition_function == 2.0
    assert sum(w.weights) == 1.0


def test_boltzmann_ten_kt_matches_closed_form():
    """A conformer 10 kT above the minimum gets exp(-10)/(1+exp(-10)).

    = 4.5397868702434395e-05.

    Compared with a relative tolerance, not exact equality. The input energy is
    reconstructed as `E_min + 10*kB*T`, so the value fed to exp() is itself the
    result of two roundings; the agreement observed is ~2.6e-12 relative, which
    is float noise on that reconstruction rather than any error in the weights.
    Demanding bit-equality here would be asserting on rounding, not physics.
    """
    expected_hi = math.exp(-10.0) / (1.0 + math.exp(-10.0))
    expected_lo = 1.0 / (1.0 + math.exp(-10.0))
    assert expected_hi == pytest.approx(4.5397868702434395e-05, rel=1e-15)

    w = ferric.boltzmann_weights([-76.0, -76.0 + 10.0 * KT_AT_DEFAULT_T])
    assert w.weights[1] == pytest.approx(expected_hi, rel=1e-9)
    assert w.weights[0] == pytest.approx(expected_lo, rel=1e-15)
    assert w.min_index == 0
    assert w.relative_energies[0] == 0.0
    assert w.relative_energies[1] == pytest.approx(10.0 * KT_AT_DEFAULT_T, rel=1e-15)


def test_boltzmann_weights_sum_to_one():
    """Normalisation across a spread of separations, including underflow."""
    for spread in ([0.0], [0.0, 1.0], [0.0, 0.5, 1.0, 2.0, 5.0], [0.0, 1000.0]):
        energies = [-76.0 + s * KT_AT_DEFAULT_T for s in spread]
        w = ferric.boltzmann_weights(energies)
        assert sum(w.weights) == pytest.approx(1.0, abs=1e-15)
        assert all(x >= 0.0 for x in w.weights)


def test_boltzmann_deep_conformer_underflows_to_zero_not_error():
    """1000 kT up is exp(-1000) -> 0.0. That is the right answer, not a fault."""
    w = ferric.boltzmann_weights([0.0, 1000.0 * KT_AT_DEFAULT_T])
    assert w.weights == [1.0, 0.0]
    assert w.partition_function == 1.0


def test_boltzmann_kt_hartree_is_kb_times_t():
    w = ferric.boltzmann_weights([0.0], temperature_k=500.0)
    assert w.kt_hartree == pytest.approx(KB_HARTREE_PER_K * 500.0, rel=1e-15)
    assert w.temperature_k == 500.0


def test_boltzmann_temperature_scales_the_exponent():
    """Doubling T halves the exponent: a 10 kT(298) gap becomes 5 kT(596)."""
    gap = 10.0 * KT_AT_DEFAULT_T
    w = ferric.boltzmann_weights([0.0, gap], temperature_k=2.0 * DEFAULT_T)
    expected = math.exp(-5.0) / (1.0 + math.exp(-5.0))
    assert w.weights[1] == pytest.approx(expected, rel=1e-9)


def test_boltzmann_min_index_finds_the_lowest_not_the_first():
    """min_index is the argmin, not index 0."""
    w = ferric.boltzmann_weights([-76.0, -76.5, -76.2])
    assert w.min_index == 1
    assert w.relative_energies[1] == 0.0
    assert w.weights[1] == max(w.weights)


def test_boltzmann_default_temperature_is_298_15():
    explicit = ferric.boltzmann_weights([0.0, KT_AT_DEFAULT_T], temperature_k=298.15)
    default = ferric.boltzmann_weights([0.0, KT_AT_DEFAULT_T])
    assert default.temperature_k == 298.15
    assert default.weights == explicit.weights


def test_ensemble_boltzmann_matches_free_function(water_ensemble):
    """The method and the free function are the same computation."""
    shifted = [[x, y, z + 0.05] for x, y, z in WATER_ANGSTROM]
    ens = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM, shifted], WATER_SYMBOLS
    )
    energies = [-76.0, -76.0 + 2.0 * KT_AT_DEFAULT_T]
    assert ens.boltzmann_weights(energies).weights == (
        ferric.boltzmann_weights(energies).weights
    )


def test_ensemble_uses_attached_energies_when_none_passed():
    ens = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM, WATER_ANGSTROM], WATER_SYMBOLS, energies=[-76.0, -75.9]
    )
    assert ens.energies() == [-76.0, -75.9]
    assert ens.boltzmann_weights().weights == (
        ferric.boltzmann_weights([-76.0, -75.9]).weights
    )


def test_boltzmann_weights_passed_in_do_not_mutate_the_ensemble():
    """Documented contract: supplying energies to boltzmann_weights is read-only."""
    ens = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM, WATER_ANGSTROM], WATER_SYMBOLS, energies=[-76.0, -75.9]
    )
    ens.boltzmann_weights([-10.0, -20.0])
    assert ens.energies() == [-76.0, -75.9]


def test_set_energy_updates_in_place():
    ens = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM, WATER_ANGSTROM], WATER_SYMBOLS
    )
    ens.set_energy(0, -76.0)
    ens.set_energy(1, -75.5)
    assert ens.energies() == [-76.0, -75.5]


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def test_diagnostics_single_conformer_dominated():
    w = ferric.boltzmann_weights([0.0, 20.0 * KT_AT_DEFAULT_T])
    d = w.diagnostics()
    assert d.n_conformers == 2
    assert d.max_weight_index == 0
    assert d.max_weight == pytest.approx(1.0, abs=1e-8)
    assert d.effective_n_conformers == pytest.approx(1.0, abs=1e-7)
    assert d.is_single_conformer_dominated(0.95) is True
    assert d.temperature_k == DEFAULT_T
    assert isinstance(d.verdict, str) and d.verdict


def test_diagnostics_degenerate_ensemble_is_not_dominated():
    """N degenerate conformers give effective_n exactly N (IPR = 1/sum(w^2))."""
    w = ferric.boltzmann_weights([0.0, 0.0, 0.0, 0.0])
    d = w.diagnostics()
    assert d.max_weight == 0.25
    assert d.effective_n_conformers == pytest.approx(4.0, rel=1e-15)
    assert d.is_single_conformer_dominated(0.95) is False
    assert d.n_within_kt == 4
    assert d.n_within_2kt == 4
    assert d.n_within_5kt == 4


def test_diagnostics_energy_window_counts():
    """n_within_{1,2,5}kT count conformers inside each window of the minimum."""
    energies = [0.0, 0.5, 1.5, 3.0, 10.0]
    d = ferric.boltzmann_weights(
        [e * KT_AT_DEFAULT_T for e in energies]
    ).diagnostics()
    assert d.n_within_kt == 2  # 0.0, 0.5
    assert d.n_within_2kt == 3  # + 1.5
    assert d.n_within_5kt == 4  # + 3.0
    assert d.n_conformers == 5


def test_diagnostics_str_is_multiline_summary():
    d = ferric.boltzmann_weights([0.0, 0.0]).diagnostics()
    assert str(d)
    assert "EnsembleDiagnostics" in repr(d)


def test_ensemble_diagnostics_shortcut_matches(water_ensemble):
    energies = [-76.0]
    assert (
        water_ensemble.diagnostics(energies).max_weight
        == water_ensemble.boltzmann_weights(energies).diagnostics().max_weight
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Weighted statistics -- mean, std_dev, and catastrophic cancellation
# ─────────────────────────────────────────────────────────────────────────────


def test_weighted_stats_single_value_has_zero_spread():
    s = ferric.weighted_stats([-76.0267196], [1.0])
    assert s.mean == -76.0267196
    assert s.std_dev == 0.0
    assert s.min == s.max == -76.0267196


def test_weighted_stats_hand_computed():
    """Mean and population std against values worked out by hand.

    values 1,3 with weights 0.5,0.5: mean 2, variance 0.5*1 + 0.5*1 = 1,
    std 1.
    """
    s = ferric.weighted_stats([1.0, 3.0], [0.5, 0.5])
    assert s.mean == 2.0
    assert s.std_dev == 1.0
    assert s.min == 1.0
    assert s.max == 3.0


def test_weighted_stats_unequal_weights_hand_computed():
    """values 0,10 weights 0.9,0.1 -> mean 1.0, var 0.9*1 + 0.1*81 = 9, std 3."""
    s = ferric.weighted_stats([0.0, 10.0], [0.9, 0.1])
    assert s.mean == pytest.approx(1.0, rel=1e-15)
    assert s.std_dev == pytest.approx(3.0, rel=1e-14)


def test_weighted_stats_min_max_are_unweighted_range():
    """min/max ignore the weights -- they describe the sampled range."""
    s = ferric.weighted_stats([1.0, 3.0, -5.0], [0.98, 0.01, 0.01])
    assert s.min == -5.0
    assert s.max == 3.0


def test_weighted_stats_catastrophic_cancellation():
    """THE test for the shifted-form variance.

    Two absolute electronic energies near -76 Ha differing by 2e-7 Ha. The
    naive E[x^2] - E[x]^2 form evaluates to a NEGATIVE variance here (the
    squares are ~5780 and the difference being sought is ~1e-14, well below
    their representable resolution), so its square root is NaN. The shifted
    form sum w (x - mean)^2 recovers the exact answer.

    This is not a contrived regime: it is exactly the regime of conformer
    energies, which is why the binding must never be "simplified" to the naive
    form.
    """
    lo, hi = -76.0267196, -76.0267194
    weights = [0.5, 0.5]
    s = ferric.weighted_stats([lo, hi], weights)

    # Exact answer: two points at +/- half the gap from their midpoint.
    expected_std = abs(hi - lo) / 2.0
    assert s.std_dev == pytest.approx(expected_std, rel=1e-9)
    assert s.std_dev > 0.0
    assert math.isfinite(s.std_dev)

    # Demonstrate the naive form really does fail on this input, so the test
    # above is known to have teeth rather than merely passing.
    mean = sum(w * v for w, v in zip(weights, [lo, hi]))
    naive_var = sum(w * v * v for w, v in zip(weights, [lo, hi])) - mean * mean
    assert naive_var < 0.0, (
        f"naive variance {naive_var!r} was not negative; this test no longer "
        "demonstrates the cancellation it was written to guard"
    )


def test_weighted_stats_cancellation_extreme():
    """Same failure at 1e8 magnitude with a 1e-3 spread: naive var is ~2.0,
    i.e. wrong by four orders of magnitude, while ferric returns 5e-4."""
    lo, hi = 1e8, 1e8 + 1e-3
    s = ferric.weighted_stats([lo, hi], [0.5, 0.5])
    assert s.std_dev == pytest.approx(5e-4, rel=1e-5)

    mean = 0.5 * lo + 0.5 * hi
    naive_var = 0.5 * lo * lo + 0.5 * hi * hi - mean * mean
    assert abs(math.sqrt(max(naive_var, 0.0)) - 5e-4) > 1.0


def test_weighted_stats_degenerate_values_give_exactly_zero_std():
    """Identical values must give std exactly 0.0, never a NaN from a small
    negative variance (core clamps the variance at 0 for this reason)."""
    s = ferric.weighted_stats([-76.02671960] * 5, [0.2] * 5)
    assert s.std_dev == 0.0


def test_weighted_stats_vector_componentwise():
    """Vector stats are component-wise, and the spread is what exposes
    cancellation between conformers."""
    stats = ferric.weighted_stats_vector([[1.0, -2.0], [3.0, 2.0]], [0.5, 0.5])
    assert len(stats) == 2
    assert stats[0].mean == 2.0
    assert stats[0].std_dev == 1.0
    # x-component averages to 2.0; y-component cancels to 0.0 despite both
    # conformers being strongly polar -- the std_dev is the only signal.
    assert stats[1].mean == 0.0
    assert stats[1].std_dev == 2.0


def test_weighted_stats_tensor_elementwise():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[3.0, 2.0], [3.0, 8.0]]
    stats = ferric.weighted_stats_tensor([a, b], [0.5, 0.5])
    assert len(stats) == 2 and len(stats[0]) == 2
    assert [[c.mean for c in row] for row in stats] == [[2.0, 2.0], [3.0, 6.0]]
    assert [[c.std_dev for c in row] for row in stats] == [[1.0, 0.0], [0.0, 2.0]]


def test_weighted_stats_with_boltzmann_weights_end_to_end():
    """The documented workflow: energies -> weights -> weighted property."""
    energies = [-76.0, -76.0]  # degenerate
    w = ferric.boltzmann_weights(energies)
    s = ferric.weighted_stats([1.0, 3.0], w.weights)
    assert s.mean == 2.0
    assert s.std_dev == 1.0


def test_weighted_stats_repr_is_high_precision():
    """repr must not round away the spread it exists to report."""
    text = repr(ferric.weighted_stats([1.0, 3.0], [0.5, 0.5]))
    assert "WeightedStats" in text
    assert "mean" in text and "std_dev" in text


# ─────────────────────────────────────────────────────────────────────────────
# 5. ERROR PATHS -- every one of these must be a catchable Python exception
# ─────────────────────────────────────────────────────────────────────────────
#
# A Rust panic crossing the pyo3 boundary is not something the caller can
# except:. Each test below asserts a specific exception CLASS, which
# simultaneously proves (a) the invariant is enforced and (b) the process
# survived -- a panic would have taken the interpreter down with it.


def test_error_empty_conformer_list():
    with pytest.raises(ValueError, match="empty"):
        ferric.ConformerEnsemble.from_coordinates([], WATER_SYMBOLS)


def test_error_atom_count_mismatch():
    """Coordinates and the shared element list must agree per conformer."""
    with pytest.raises(ValueError) as exc:
        ferric.ConformerEnsemble.from_coordinates([WATER_ANGSTROM], ["O", "H"])
    assert "conformer 0" in str(exc.value)


def test_error_atom_count_mismatch_names_the_bad_conformer():
    """The message must identify WHICH conformer, not just that one is wrong."""
    two_atoms = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
    with pytest.raises(ValueError) as exc:
        ferric.ConformerEnsemble.from_coordinates(
            [WATER_ANGSTROM, two_atoms], WATER_SYMBOLS
        )
    assert "conformer 1" in str(exc.value)


def test_error_row_is_not_three_cartesians():
    bad = [[0.0, 0.0], [0.0, 0.0, 0.9], [0.9, 0.0, -0.2]]
    with pytest.raises(ValueError) as exc:
        ferric.ConformerEnsemble.from_coordinates([bad], WATER_SYMBOLS)
    assert "atom 0" in str(exc.value)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_error_non_finite_coordinate(bad_value):
    """NaN/inf geometries yield plausible-looking meaningless energies, so they
    are rejected at the boundary rather than propagated into an SCF."""
    bad = [[bad_value, 0.0, 0.0], [0.0, 0.0, 0.9], [0.9, 0.0, -0.2]]
    with pytest.raises(ValueError) as exc:
        ferric.ConformerEnsemble.from_coordinates([bad], WATER_SYMBOLS)
    assert "non-finite" in str(exc.value)


def test_error_bool_is_not_an_atomic_number():
    """THE bool-is-an-int-subclass trap.

    `isinstance(True, int)` is True in Python, and `int(True) == 1`, so a naive
    extraction turns [True, 1, 1] into three hydrogens -- a different molecule,
    silently. TypeError, not ValueError: passing a bool is a type confusion,
    not an out-of-range value.
    """
    with pytest.raises(TypeError, match="bool"):
        ferric.ConformerEnsemble.from_coordinates(
            [WATER_ANGSTROM], [True, 1, 1]
        )


def test_error_bool_rejected_in_any_position():
    """Not just position 0 -- the guard must apply element-wise."""
    with pytest.raises(TypeError, match="bool"):
        ferric.ConformerEnsemble.from_coordinates(
            [WATER_ANGSTROM], [8, 1, False]
        )


def test_error_unknown_atomic_number():
    with pytest.raises(ValueError, match="999"):
        ferric.ConformerEnsemble.from_coordinates([WATER_ANGSTROM], [999, 1, 1])


def test_error_unknown_element_symbol():
    with pytest.raises(ValueError):
        ferric.ConformerEnsemble.from_coordinates(
            [WATER_ANGSTROM], ["Xx", "H", "H"]
        )


def test_error_empty_element_list():
    with pytest.raises(ValueError, match="empty"):
        ferric.ConformerEnsemble.from_coordinates([WATER_ANGSTROM], [])


def test_error_elements_wrong_type():
    with pytest.raises(TypeError):
        ferric.ConformerEnsemble.from_coordinates(
            [WATER_ANGSTROM], [None, "H", "H"]
        )


def test_error_energy_count_mismatch_at_construction():
    with pytest.raises(ValueError):
        ferric.ConformerEnsemble.from_coordinates(
            [WATER_ANGSTROM], WATER_SYMBOLS, energies=[-76.0, -75.0]
        )


def test_error_molecule_index_out_of_range(water_ensemble):
    """molecule() uses IndexError -- the correct class for a bad index."""
    with pytest.raises(IndexError, match="out of range"):
        water_ensemble.molecule(5)


def test_error_energies_unset(water_ensemble):
    """Reading energies that were never set must raise, not return garbage:
    an unconverged SCF must never be silently averaged in."""
    with pytest.raises(ValueError):
        water_ensemble.energies()


@pytest.mark.parametrize(
    "bad_t", [0.0, -1.0, -298.15, float("nan"), float("inf")]
)
def test_error_bad_temperature(bad_t):
    """T <= 0 divides by zero / flips the exponent sign. Must raise."""
    with pytest.raises(ValueError, match="temperature"):
        ferric.boltzmann_weights([-76.0], temperature_k=bad_t)


def test_error_bad_temperature_on_ensemble_method(water_ensemble):
    with pytest.raises(ValueError, match="temperature"):
        water_ensemble.boltzmann_weights([-76.0], temperature_k=0.0)


def test_error_energy_count_mismatch_in_weights(water_ensemble):
    with pytest.raises(ValueError):
        water_ensemble.boltzmann_weights([-76.0, -75.0])


@pytest.mark.parametrize("bad_e", [float("nan"), float("inf")])
def test_error_non_finite_energy(bad_e):
    with pytest.raises(ValueError):
        ferric.boltzmann_weights([-76.0, bad_e])


def test_error_empty_energy_list():
    with pytest.raises(ValueError):
        ferric.boltzmann_weights([])


def test_error_set_energy_out_of_range(water_ensemble):
    """Out-of-range set_energy must raise.

    NOTE the class is ValueError, not the IndexError that molecule() raises for
    the same kind of mistake. Pinned as observed rather than as preferred -- see
    the suite's report; this is an API inconsistency, not a correctness bug.
    """
    with pytest.raises(ValueError):
        water_ensemble.set_energy(9, -76.0)


@pytest.mark.parametrize("bad_e", [float("nan"), float("inf")])
def test_error_set_energy_non_finite(water_ensemble, bad_e):
    with pytest.raises(ValueError):
        water_ensemble.set_energy(0, bad_e)


def test_error_weighted_stats_length_mismatch():
    with pytest.raises(ValueError):
        ferric.weighted_stats([1.0, 2.0], [1.0])


def test_error_weighted_stats_empty():
    with pytest.raises(ValueError):
        ferric.weighted_stats([], [])


@pytest.mark.parametrize("bad_v", [float("nan"), float("inf")])
def test_error_weighted_stats_non_finite_value(bad_v):
    with pytest.raises(ValueError):
        ferric.weighted_stats([1.0, bad_v], [0.5, 0.5])


def test_error_weighted_stats_vector_ragged():
    with pytest.raises(ValueError):
        ferric.weighted_stats_vector([[1.0, 2.0], [1.0]], [0.5, 0.5])


def test_error_weighted_stats_vector_length_mismatch():
    with pytest.raises(ValueError):
        ferric.weighted_stats_vector([[1.0], [2.0]], [1.0])


def test_error_weighted_stats_tensor_inconsistent_shape():
    with pytest.raises(ValueError):
        ferric.weighted_stats_tensor(
            [[[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]]], [0.5, 0.5]
        )


def test_error_from_multi_xyz_missing_file(tmp_path):
    with pytest.raises(ValueError):
        ferric.ConformerEnsemble.from_multi_xyz(str(tmp_path / "nope.xyz"))


def test_process_survives_the_whole_error_surface(water_ensemble):
    """Belt-and-braces: run a normal call AFTER every error test above.

    If any invariant violation had panicked across the FFI boundary instead of
    raising, the interpreter would already be gone and this would never run.
    """
    w = ferric.boltzmann_weights([-76.0, -76.0])
    assert w.weights == [0.5, 0.5]
    assert water_ensemble.n_atoms() == 3


# ─────────────────────────────────────────────────────────────────────────────
# 6. Smoke tests for the wider already-exposed surface
# ─────────────────────────────────────────────────────────────────────────────


def test_molecule_from_xyz_smoke():
    mol = ferric.Molecule.from_xyz(
        os.path.join(TESTDATA, "molecules", "water.xyz")
    )
    assert mol.natoms() == 3
    assert mol.nelec() == 10
    assert mol.nuclear_repulsion() == pytest.approx(9.189193229309746, abs=1e-6)


def test_molecule_from_xyz_string_smoke():
    mol = ferric.Molecule.from_xyz_string(water_xyz_string())
    assert mol.natoms() == 3
    assert mol.nelec() == 10


def test_molecule_charge_changes_electron_count():
    cation = ferric.Molecule.from_xyz_string(
        water_xyz_string(), charge=1, multiplicity=2
    )
    assert cation.nelec() == 9


def test_basis_set_bundled_smoke(sto3g):
    assert sto3g is not None
    assert ferric.BasisSet.bundled("cc-pvdz") is not None


def test_basis_set_unknown_name_raises():
    with pytest.raises(Exception):
        ferric.BasisSet.bundled("definitely-not-a-basis")


def test_run_rhf_water_sto3g(sto3g):
    """One cheap SCF with a real energy assertion.

    The expected value is read from the committed PySCF reference rather than
    hardcoded here, so there is exactly one copy of the number in the repo.
    """
    with open(
        os.path.join(TESTDATA, "reference", "h2o_sto-3g_rhf.json")
    ) as fh:
        ref = json.load(fh)

    mol = ferric.Molecule.from_xyz(
        os.path.join(TESTDATA, "molecules", "water.xyz")
    )
    result = ferric.run_rhf(mol, sto3g)
    assert result.converged
    assert result.energy == pytest.approx(ref["energy"], abs=5e-8)
    assert result.density().shape == (ref["nbasis"], ref["nbasis"])
    assert len(result.orbital_energies()) == ref["nbasis"]
    assert mol.nelec() == ref["nelec"]


def test_run_rhf_over_an_ensemble(sto3g):
    """The documented end-to-end workflow, at STO-3G water scale.

    Two identical conformers must give identical energies, hence exactly
    degenerate Boltzmann weights and zero ensemble spread. That makes this a
    real assertion about the pipeline rather than a "it ran" check.
    """
    ens = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM, WATER_ANGSTROM], WATER_SYMBOLS
    )
    energies = [ferric.run_rhf(m, sto3g).energy for m in ens.molecules()]
    assert energies[0] == energies[1]

    w = ens.boltzmann_weights(energies)
    assert w.weights == [0.5, 0.5]

    stats = ferric.weighted_stats(energies, w.weights)
    assert stats.mean == energies[0]
    assert stats.std_dev == 0.0
    assert w.diagnostics().effective_n_conformers == pytest.approx(2.0, rel=1e-15)


def test_ensemble_geometry_reaches_the_scf_unscaled(sto3g):
    """An ensemble-built Molecule and a parse_xyz Molecule give the SAME energy.

    This is the unit convention checked at the far end of the pipeline: a
    factor-1.889726 geometry error would shift the SCF energy by Hartrees while
    still converging. Bit-identical is the correct expectation because both
    paths reach parse_xyz with the same text.
    """
    ens = ferric.ConformerEnsemble.from_coordinates(
        [WATER_ANGSTROM], WATER_SYMBOLS
    )
    from_ens = ferric.run_rhf(ens.molecule(0), sto3g).energy
    from_xyz = ferric.run_rhf(
        ferric.Molecule.from_xyz_string(water_xyz_string()), sto3g
    ).energy
    assert from_ens == from_xyz

    # And it is a physically sensible water energy, not a scaled-geometry one.
    assert -76.0 < from_ens < -74.0
