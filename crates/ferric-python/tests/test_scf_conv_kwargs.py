"""SCF convergence keywords on the drivers that solve their own reference SCF.

`run_rimp2` and `run_tddft` converge a reference SCF internally. They accept
`max_iter`, `energy_conv` and `density_conv` with the names and defaults of
`run_rhf` (SCF default density_conv 1e-6, energy_conv 1e-3), so a caller can
tighten the orbitals the correlated / response step is built on.

Pinned here:
  * omitting the keywords is bit-identical to passing the `run_rhf` defaults
    explicitly (the keyword layer adds nothing when unset);
  * a tighter `density_conv` moves the reference SCF energy toward a tightly
    converged `run_rhf` and changes the result by a small amount only;
  * the keywords actually reach the SCF (a tiny `max_iter` fails to converge).
"""

from __future__ import annotations

import ferric
import pytest

MOL = ferric.Molecule.from_xyz("testdata/molecules/water.xyz", 0, 1)
BS = ferric.BasisSet.bundled("6-31g")
AUX = ferric.BasisSet.bundled("cc-pvdz-ri")

# RhfConfig defaults, as run_rhf documents them.
DEFAULTS = dict(energy_conv=1e-3, density_conv=1e-6)
TIGHT = dict(density_conv=1e-10)
ULTRA = dict(energy_conv=1e-13, density_conv=1e-12)


def _ref_energy(**kw) -> float:
    return ferric.run_rhf(MOL, BS, **kw).energy


def test_rimp2_omitted_is_bit_identical_to_explicit_defaults():
    a = ferric.run_rimp2(MOL, BS, AUX)
    b = ferric.run_rimp2(MOL, BS, AUX, **DEFAULTS)
    assert a.rhf_energy == b.rhf_energy
    assert a.total_energy == b.total_energy
    assert a.mp2_corr == b.mp2_corr


def test_rimp2_tighter_density_conv_moves_scf_toward_tight_run_rhf():
    e_ref = _ref_energy(**ULTRA)
    loose = ferric.run_rimp2(MOL, BS, AUX)
    tight = ferric.run_rimp2(MOL, BS, AUX, **TIGHT)
    err_loose = abs(loose.rhf_energy - e_ref)
    err_tight = abs(tight.rhf_energy - e_ref)
    assert err_tight < err_loose
    assert err_tight < 1e-9
    # The default SCF is already good to well inside the old tolerance.
    assert abs(tight.total_energy - loose.total_energy) < 1e-5


def test_rimp2_max_iter_reaches_the_scf():
    with pytest.raises(Exception, match="(?i)converge"):
        ferric.run_rimp2(MOL, BS, AUX, max_iter=1)


def test_tddft_omitted_is_bit_identical_to_explicit_defaults():
    a = ferric.run_tddft(MOL, BS, AUX, n_roots=3)
    b = ferric.run_tddft(MOL, BS, AUX, n_roots=3, **DEFAULTS)
    assert list(a.excitation_energies) == list(b.excitation_energies)
    assert list(a.oscillator_strengths) == list(b.oscillator_strengths)


def test_tddft_tighter_density_conv_moves_roots_toward_converged_orbitals():
    ultra = ferric.run_tddft(MOL, BS, AUX, n_roots=3, **ULTRA).excitation_energies
    loose = ferric.run_tddft(MOL, BS, AUX, n_roots=3).excitation_energies
    tight = ferric.run_tddft(MOL, BS, AUX, n_roots=3, **TIGHT).excitation_energies
    err_loose = abs(loose - ultra).max()
    err_tight = abs(tight - ultra).max()
    assert err_tight < err_loose
    # Under the old (default-SCF) tolerance of the CIS comparison.
    assert abs(tight - loose).max() < 1e-5


def test_tddft_max_iter_reaches_the_scf():
    with pytest.raises(Exception, match="(?i)converge"):
        ferric.run_tddft(MOL, BS, AUX, n_roots=3, max_iter=1)
