"""SCF convergence keywords on the drivers that solve their own reference SCF.

Naming rule: a bare `max_iter` / `energy_conv` / `density_conv` belongs to a
driver whose method IS the SCF (`run_rhf`, `run_uhf`, ...). Every other driver
(post-SCF methods, geometry and path drivers) takes `scf_max_iter`,
`scf_energy_conv`, `scf_density_conv` with the meaning and defaults of
`run_rhf`'s (density_conv 1e-6, energy_conv 1e-3), so a caller can tighten the
orbitals the correlated / response / geometry step is built on.

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
DEFAULTS = dict(scf_energy_conv=1e-3, scf_density_conv=1e-6)
TIGHT = dict(scf_density_conv=1e-10)
ULTRA = dict(scf_energy_conv=1e-13, scf_density_conv=1e-12)
RHF_ULTRA = dict(energy_conv=1e-13, density_conv=1e-12)


def _ref_energy(**kw) -> float:
    return ferric.run_rhf(MOL, BS, **kw).energy


def test_rimp2_omitted_is_bit_identical_to_explicit_defaults():
    a = ferric.run_rimp2(MOL, BS, AUX)
    b = ferric.run_rimp2(MOL, BS, AUX, **DEFAULTS)
    assert a.rhf_energy == b.rhf_energy
    assert a.total_energy == b.total_energy
    assert a.mp2_corr == b.mp2_corr


def test_rimp2_tighter_density_conv_moves_scf_toward_tight_run_rhf():
    e_ref = _ref_energy(**RHF_ULTRA)
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
        ferric.run_rimp2(MOL, BS, AUX, scf_max_iter=1)


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
        ferric.run_tddft(MOL, BS, AUX, n_roots=3, scf_max_iter=1)


# ── bare spellings are gone from post-SCF drivers (the rename is the contract) ──


@pytest.mark.parametrize("kw", ["max_iter", "energy_conv", "density_conv"])
def test_post_scf_driver_rejects_the_bare_spelling(kw):
    with pytest.raises(TypeError, match=kw):
        ferric.run_rimp2(MOL, BS, AUX, **{kw: 1})


# ── run_oo_rimp2: oo_max_iter is the orbital loop, scf_* the reference SCF ──


def test_oo_rimp2_scf_max_iter_reaches_the_reference_scf():
    with pytest.raises(Exception, match="(?i)converge"):
        ferric.run_oo_rimp2(MOL, BS, AUX, scf_max_iter=1)


def test_oo_rimp2_omitted_is_bit_identical_to_explicit_defaults():
    a = ferric.run_oo_rimp2(MOL, BS, AUX)
    b = ferric.run_oo_rimp2(MOL, BS, AUX, **DEFAULTS)
    assert a.total_energy == b.total_energy


def test_oo_rimp2_bare_max_iter_is_a_deprecated_alias_of_oo_max_iter():
    with pytest.warns(DeprecationWarning, match="oo_max_iter"):
        old = ferric.run_oo_rimp2(MOL, BS, AUX, max_iter=2)
    new = ferric.run_oo_rimp2(MOL, BS, AUX, oo_max_iter=2)
    assert old.total_energy == new.total_energy
    assert old.converged == new.converged
    with pytest.raises(ValueError, match="deprecated alias"):
        ferric.run_oo_rimp2(MOL, BS, AUX, max_iter=2, oo_max_iter=2)


# ── run_cdft / run_frequencies: bare spellings are deprecated aliases ──

H2 = ferric.Molecule.from_xyz("testdata/molecules/h2.xyz", 0, 1)
HE2P = ferric.Molecule.from_xyz_string("2\nHe2+\nHe 0 0 0\nHe 0 0 1.5\n", 1, 2)
SVP = ferric.BasisSet.bundled("def2-svp")


def _cdft(**kw):
    return ferric.run_cdft(
        HE2P,
        SVP,
        [ferric.CdftConstraint([0], 1.0)],
        lambda_tol=1e-2,
        max_outer=40,
        level_shift=0.2,
        grid_radial=99,
        grid_angular=302,
        **kw,
    )


def test_cdft_scf_max_iter_reaches_the_scf_and_bare_alias_warns():
    # cDFT reports an unconverged inner SCF through `scf_converged` (or fails
    # the outer loop) rather than raising; either way one iteration is not enough.
    try:
        r = _cdft(scf_max_iter=1)
    except RuntimeError as e:
        assert "converge" in str(e).lower()
    else:
        assert not (r.converged and r.scf_converged)
    assert _cdft().scf_converged
    with pytest.warns(DeprecationWarning, match="scf_max_iter"):
        try:
            old = _cdft(max_iter=1)
        except RuntimeError:
            old = None
    new = None
    try:
        new = _cdft(scf_max_iter=1)
    except RuntimeError:
        pass
    assert (old is None) == (new is None)
    with pytest.raises(ValueError, match="deprecated alias"):
        _cdft(max_iter=5, scf_max_iter=5)


def test_cdft_omitted_is_bit_identical_to_explicit_defaults():
    a, b = _cdft(), _cdft(**DEFAULTS)
    assert a.energy == b.energy


def test_frequencies_scf_kwargs_and_deprecated_aliases():
    a = ferric.run_frequencies(H2, "sto-3g")
    b = ferric.run_frequencies(H2, "sto-3g", **DEFAULTS)
    assert list(a.frequencies) == list(b.frequencies)
    with pytest.raises(Exception, match="(?i)converge"):
        ferric.run_frequencies(H2, "sto-3g", scf_max_iter=1)
    with pytest.warns(DeprecationWarning, match="scf_density_conv"):
        c = ferric.run_frequencies(H2, "sto-3g", density_conv=1e-6)
    assert list(c.frequencies) == list(a.frequencies)
    with pytest.raises(ValueError, match="deprecated alias"):
        ferric.run_frequencies(H2, "sto-3g", density_conv=1e-6, scf_density_conv=1e-6)


# ── geometry and path drivers: the knob reaches the SCF at every geometry ──

# Water, not H2: H2/STO-3G is symmetry-fixed and converges in one SCF iteration,
# which would let a dropped `scf_max_iter` pass unnoticed.
STRETCHED = ferric.Molecule.from_xyz_string(
    "3\nwater\nO 0 0 0\nH 0 0.80 0.60\nH 0 -0.80 0.60\n", 0, 1
)
H2_UHF_TRIPLET = ferric.Molecule.from_xyz_string("2\nH2\nH 0 0 0\nH 0 0 1.2\n", 0, 3)


def test_optimize_omitted_is_bit_identical_to_explicit_defaults():
    a = ferric.run_optimize(STRETCHED, "sto-3g", max_steps=2)
    b = ferric.run_optimize(STRETCHED, "sto-3g", max_steps=2, **DEFAULTS)
    assert a.energy == b.energy
    assert a.energy_trace == b.energy_trace


def test_optimize_scf_max_iter_reaches_the_per_step_scf():
    with pytest.raises(Exception, match="(?i)converge"):
        ferric.run_optimize(STRETCHED, "sto-3g", max_steps=2, scf_max_iter=1)


def test_optimize_tighter_scf_density_conv_differs_by_less_than_the_default_tolerance():
    loose = ferric.run_optimize(STRETCHED, "sto-3g", max_steps=2)
    tight = ferric.run_optimize(STRETCHED, "sto-3g", max_steps=2, **TIGHT)
    # The SCF energy is variational: tightening the density criterion moves it
    # only by the old tolerance's residual, never more than the default
    # density_conv. It must move at all, or the knob never reached the SCF.
    d = abs(loose.energy - tight.energy)
    assert 0.0 < d < 1e-6


def test_optimize_uhf_scf_max_iter_reaches_the_per_step_scf():
    a = ferric.run_optimize_uhf(H2_UHF_TRIPLET, "sto-3g", max_steps=1)
    b = ferric.run_optimize_uhf(H2_UHF_TRIPLET, "sto-3g", max_steps=1, **DEFAULTS)
    assert a.energy == b.energy
    with pytest.raises(Exception, match="(?i)converge"):
        ferric.run_optimize_uhf(H2_UHF_TRIPLET, "sto-3g", max_steps=1, scf_max_iter=1)


def test_optimize_qmmm_scf_max_iter_reaches_the_per_step_scf():
    sys = ferric.QmmmSystem(
        ["O", "H", "H"],
        [(0.0, 0.0, 0.0), (0.0, 0.80, 0.60), (0.0, -0.80, 0.60)],
        [0.0, 0.0, 0.0],
        qm_indices=[0, 1, 2],
    )
    a = ferric.run_optimize_qmmm(sys, "sto-3g", max_steps=2)
    b = ferric.run_optimize_qmmm(sys, "sto-3g", max_steps=2, **DEFAULTS)
    assert a.energy == b.energy
    with pytest.raises(Exception, match="(?i)converge"):
        ferric.run_optimize_qmmm(sys, "sto-3g", max_steps=2, scf_max_iter=1)


NH3 = ferric.Molecule.from_xyz_string(
    "4\nnh3 near planar\n"
    "N  0.0000  0.0000  0.0000\n"
    "H  0.0000  1.0100  0.1000\n"
    "H  0.8747 -0.5050  0.1000\n"
    "H -0.8747 -0.5050  0.1000\n"
)


def test_saddle_and_irc_scf_max_iter_reach_the_scf():
    with pytest.raises(Exception, match="(?i)converge"):
        ferric.run_saddle(NH3, "sto-3g", max_steps=2, scf_max_iter=1)
    mode = [0.0] * 12
    mode[2] = 1.0
    with pytest.raises(Exception, match="(?i)converge"):
        ferric.run_irc(NH3, "sto-3g", mode, max_steps=2, scf_max_iter=1)
