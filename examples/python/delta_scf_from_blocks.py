"""Core-hole Delta-SCF written on ferric's blocks: a Fock-hook level shift.

Delta-SCF (Bagus, Phys. Rev. 139, A619 (1965)) takes an excitation or
ionization energy as the difference of two SCF energies, the second one
converged on a non-aufbau occupation. Here the second state is the O 1s
core-ionized cation, for the O 1s binding energy.

The non-aufbau state is held with a level shift on each spin's virtual space,
as STEP does (Carter-Fenk, Herbert, JCTC 16, 5067 (2020)):

    V_s = eta (S - S D_s S),

the projector onto the orbitals D_s leaves empty, scaled by eta. With eta
above the gap between the emptied core orbital and the highest occupied one,
aufbau on F_s + V_s keeps the occupation the start density picked. V_s
commutes with D_s (S D_s S D_s = S D_s for an idempotent D_s), so at
convergence F_s itself commutes with D_s: the state is a stationary point of
the plain UHF energy, and the hook returns E = 0 so the driver's energy is the
plain UHF energy.

eta is the gap plus ``margin`` (1 Ha), measured on the start density. The
cation is a saddle point: a level-shifted Roothaan step multiplies the
gradient along a hole-occupied rotation by eta / (eta - gap) > 1, so DIIS has
to extrapolate past a mode a plain step does not converge. MEASURED
2026-11-13 (water and NH3, cc-pVDZ, gaps 19.0 and 14.3 Ha): at the default
diis_size 8, 1 of 17 runs with eta from gap + 1 to 100 Ha converged in 200
iterations, the others stalling at an RMS density change of 1e-8 to 5e-7
with the energy within 1e-11 Ha of its limit; with diis_size 16 and
eta = gap + 1 they converge in 116 and 126 iterations.

The start is ``dm0``: the neutral RHF orbitals with one beta 1s electron
removed. ``run_uhf(fock_hook=)`` runs the SCF. ``uhf_check`` rebuilds the UHF
energy and the plain orbital gradient from ``AoIntegrals.veff_polarized``.
``mom_uhf`` is the independent check: the maximum overlap method (Gilbert,
Besley, Gill, JPCA 112, 13164 (2008)) as a numpy UHF on veff_polarized, from
the same start. J and K are exact. Bohr and Hartree.

Run as a script: the O 1s binding energy of water/cc-pVDZ, asserting it equals
``mom_uhf`` to 3e-11 Ha and the PySCF reference (``scf.addons.mom_occ``) to
1e-9 Ha.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ferric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhf_from_blocks import ROOT, WATER  # noqa: E402

PYSCF_REF = ROOT / "testdata" / "reference" / "block_ports_set2_pyscf.json"

EXACT_JK = {"df_j_aux": "exact", "df_k_aux": "exact"}
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}


def virtual_shift_hook(s: np.ndarray, eta: float):
    """The run_uhf hook (D_a, D_b) -> (0, V_a, V_b), V_s = eta (S - S D_s S)."""

    def shift(d):
        v = eta * (s - s @ d @ s)
        return 0.5 * (v + v.T)

    def hook(d_a, d_b):
        return 0.0, shift(d_a), shift(d_b)

    return hook


def shift_for(
    ints: ferric.AoIntegrals, c: np.ndarray, occ, margin: float = 1.0
) -> float:
    """eta: the largest occupied-minus-empty orbital energy gap at the start, + margin.

    ``occ`` holds the occupied column indices of ``c`` per spin (alpha, beta).
    """
    d = [c[:, o] @ c[:, o].T for o in occ]
    _, v_a, v_b = ints.veff_polarized(*d)
    gap = 0.0
    for o, v in zip(occ, (v_a, v_b)):
        e = np.diag(c.T @ (ints.hcore() + v) @ c)
        empty = np.setdiff1d(np.arange(c.shape[1]), o)
        gap = max(gap, e[o].max() - e[empty].min())
    return gap + margin


def uhf_check(ints: ferric.AoIntegrals, d_a: np.ndarray, d_b: np.ndarray):
    """(plain UHF energy, max |F_s D_s S - S D_s F_s|) of a density pair."""
    h, s = ints.hcore(), ints.overlap()
    e2, v_a, v_b = ints.veff_polarized(d_a, d_b)
    energy = float(np.sum((d_a + d_b) * h)) + e2 + ints.nuclear_repulsion()
    grad = 0.0
    for d, v in ((d_a, v_a), (d_b, v_b)):
        f = h + v
        grad = max(grad, float(np.abs(f @ d @ s - s @ d @ f).max()))
    return energy, grad


def mom_uhf(ints, c0, occ, e_conv=1e-11, d_conv=1e-9, max_iter=200, diis_size=8):
    """UHF energy by MOM + DIIS in numpy from the occupied columns ``occ`` of ``c0``."""
    h, s = ints.hcore(), ints.overlap()
    w, u = np.linalg.eigh(s)
    x = u / np.sqrt(w)
    c_occ = [c0[:, o] for o in occ]
    d = [co @ co.T for co in c_occ]
    focks, errors, e_old = [], [], 0.0
    for _ in range(max_iter):
        e2, *v = ints.veff_polarized(*d)
        f = [h + vs for vs in v]
        energy = float(np.sum((d[0] + d[1]) * h)) + e2 + ints.nuclear_repulsion()
        err = np.concatenate(
            [(x.T @ (fs @ ds @ s - s @ ds @ fs) @ x).ravel() for fs, ds in zip(f, d)]
        )
        focks, errors = (focks + [f])[-diis_size:], (errors + [err])[-diis_size:]
        n = len(focks)
        b = -np.ones((n + 1, n + 1))
        b[n, n] = 0.0
        b[:n, :n] = [[ei @ ej for ej in errors] for ei in errors]
        coef = np.linalg.solve(b, np.r_[np.zeros(n), -1.0])[:n]
        d_new = []
        for spin in range(2):
            fs = sum(ci * fk[spin] for ci, fk in zip(coef, focks))
            _, cs = np.linalg.eigh(x.T @ fs @ x)
            cs = x @ cs
            # MOM: keep the orbitals that overlap most with the last occupied set.
            p = np.sum(np.abs(c_occ[spin].T @ s @ cs), axis=0)
            c_occ[spin] = cs[:, np.sort(np.argsort(-p)[: c_occ[spin].shape[1]])]
            d_new.append(c_occ[spin] @ c_occ[spin].T)
        change = max(np.abs(dn - ds).max() for dn, ds in zip(d_new, d))
        d = d_new
        if abs(energy - e_old) < e_conv and change < d_conv:
            return uhf_check(ints, *d)[0]
        e_old = energy
    raise RuntimeError(f"MOM-UHF did not converge in {max_iter} iterations")


def core_ionization(xyz: Path, basis: str, hole: int = 0, mom: bool = False) -> dict:
    """Delta-SCF binding energy of the closed-shell orbital ``hole`` (Hartree).

    mom=False runs the level-shift hook on run_uhf; mom=True runs ``mom_uhf``
    from the same start instead (the independent check). Returns the neutral
    and cation energies, the binding energy ``ip``, the shift ``eta`` and, for
    the hook, the cation's rebuilt UHF energy ``e_check`` and orbital
    gradient ``grad``.
    """
    bs = ferric.BasisSet.bundled(basis)
    neutral = ferric.run_rhf(
        ferric.Molecule.from_xyz(str(xyz)), bs, **EXACT_JK, **TIGHT
    )
    cation = ferric.Molecule.from_xyz(str(xyz), charge=1, multiplicity=2)
    ints = ferric.AoIntegrals(cation, bs)
    nocc = (ints.nelec + 1) // 2
    c = neutral.mo_coefficients()
    occ = (list(range(nocc)), [i for i in range(nocc) if i != hole])
    eta = shift_for(ints, c, occ)
    out = {"e_neutral": neutral.energy, "eta": eta}
    if mom:
        out["e_cation"] = mom_uhf(ints, c, occ)
    else:
        dm0 = tuple(c[:, o] @ c[:, o].T for o in occ)
        hook = virtual_shift_hook(ints.overlap(), eta)
        ion = ferric.run_uhf(
            cation, bs, dm0=dm0, fock_hook=hook, diis_size=16, **EXACT_JK, **TIGHT
        )
        assert ion.converged
        out["e_cation"] = ion.energy
        out["e_check"], out["grad"] = uhf_check(
            ints, ion.density_alpha(), ion.density_beta()
        )
    out["ip"] = out["e_cation"] - neutral.energy
    return out


def main() -> None:
    hook = core_ionization(WATER, "cc-pvdz")
    mom = core_ionization(WATER, "cc-pvdz", mom=True)
    print(f"neutral RHF       {hook['e_neutral']:.10f}")
    print(f"level shift eta   {hook['eta']:.3f}")
    print(f"O 1s cation, hook {hook['e_cation']:.10f}   rebuilt {hook['e_check']:.10f}")
    print(f"orbital gradient  {hook['grad']:.1e}")
    print(f"O 1s IP, hook     {hook['ip']:.10f} Ha = {hook['ip'] * 27.211386:.4f} eV")
    print(f"O 1s IP, MOM      {mom['ip']:.10f}   diff {hook['ip'] - mom['ip']:.1e}")
    assert abs(hook["ip"] - mom["ip"]) < 3e-11
    if PYSCF_REF.exists():
        ref = json.loads(PYSCF_REF.read_text())["molecules"]["water"]["delta_scf"]
        diff = hook["ip"] - ref["ip"]
        print(f"O 1s IP, PySCF    {ref['ip']:.10f}   diff {diff:.1e}")
        assert abs(diff) < 1e-9


if __name__ == "__main__":
    main()
