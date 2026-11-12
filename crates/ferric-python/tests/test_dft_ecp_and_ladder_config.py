"""`run_dft` on an ECP basis with `dm0=`, and the ladder rung's config.

* HI / def2-SVP (28-electron ECP on I): `run_dft` applies the ECP to the
  molecule, as `run_rhf` does, so the valence density of `run_rhf` is a valid
  `dm0`, and the run from it reaches the plain run's energy within the sum of
  both runs' `energy_conv`. The certificate's `ecp` approximation and the
  density's electron count (26 valence electrons) pin that the plain run
  itself is the valence calculation.
* A hooked `run_dft` whose first two ladder rungs cannot converge (the hook
  adds a random symmetric potential for its first calls) converges on a
  level-shifted rung, and the certificate records that rung's level shift.
"""

from __future__ import annotations

import numpy as np

import ferric

HI = "2\n\nH 0 0 0\nI 0 0 1.61\n"
WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
EC = 1e-9


def test_run_dft_applies_the_ecp_and_accepts_an_rhf_valence_dm0():
    mol = ferric.Molecule.from_xyz_string(HI)
    bs = ferric.BasisSet.bundled("def2-svp")
    rhf = ferric.run_rhf(mol, bs, energy_conv=EC, density_conv=1e-7)
    s = ferric.AoIntegrals(mol, bs).overlap()
    assert abs(np.sum(rhf.density() * s) - 26.0) < 1e-8
    plain = ferric.run_dft(mol, bs, "PBE", energy_conv=EC)
    assert abs(np.sum(plain.density() * s) - 26.0) < 1e-8
    names = [a["name"] for a in plain.certificate["approximations"]]
    assert "ecp" in names, names
    started = ferric.run_dft(mol, bs, "PBE", energy_conv=EC, dm0=rhf.density())
    assert started.converged
    assert abs(started.e_scf - plain.e_scf) < 2 * EC


def _noise_hook(n_noisy: int, n: int, seed: int = 7):
    """`(E, V)` hook: a random symmetric V of norm ~1 Ha for the first
    `n_noisy` calls, zero after."""
    rng = np.random.default_rng(seed)
    calls = [0]

    def hook(d):
        calls[0] += 1
        if calls[0] > n_noisy:
            return 0.0, np.zeros_like(d)
        a = rng.standard_normal((n, n))
        v = 0.5 * (a + a.T)
        return float(np.sum(d * v)), v

    return hook


def test_ladder_certificate_records_the_converged_rungs_level_shift():
    # Rung 0 (max_iter=1) and rung 1 (ADIIS, at most 60 iterations, no
    # shift) run inside the noise: 64 noisy calls outlast both even at one
    # hook call per iteration. Every later rung carries a level shift of at
    # least 0.5 Ha (ferric_scf::ladder::ksdft_ladder).
    mol = ferric.Molecule.from_xyz_string(WATER)
    bs = ferric.BasisSet.bundled("sto-3g")
    n = bs.n_functions(mol)
    r = ferric.run_dft(
        mol, bs, "PBE", max_iter=1, fock_hook=_noise_hook(64, n), energy_conv=1e-8
    )
    assert r.converged
    shifts = [
        a["detail"]
        for a in r.certificate["approximations"]
        if a["name"] == "level-shift"
    ]
    assert len(shifts) == 1, r.certificate["approximations"]
    assert float(shifts[0].split()[0]) >= 0.5, shifts
