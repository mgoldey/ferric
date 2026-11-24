"""Projection-based WF-in-DFT embedding written in numpy on ferric's blocks.

Manby, Stella, Goodpaster, Miller, JCTC 8, 2564 (2012). A low-level KS
calculation on the whole molecule gives the occupied orbitals; they are split
into subsystem A and B, with densities gamma_A and gamma_B (total, closed
shell). Subsystem A is then solved at the high level (HF, or a functional) in
the embedding potential

    v_emb = g_low[gamma_A + gamma_B] - g_low[gamma_A] + mu S C_B C_B^T S,

where g_low is the low-level two-electron + XC potential (``AoIntegrals.veff``)
and the level shift mu removes B's orbitals from A's variational space. The
total energy is

    E = E_high[gamma~_A] + E_low[gamma] - E_low[gamma_A]
        + tr[(gamma~_A - gamma_A) v_emb] + E_nn.

The subsystem SCF is ``run_rhf`` or ``run_dft`` on the same nuclei with
charge +N_B and ``fock_hook`` returning (tr[D v_emb], v_emb): the driver's
energy is then E_high[gamma~_A] + E_nn + tr[gamma~_A v_emb].

The A/B split is SPADE (Claudino, Mayhall, JCTC 15, 1053 (2019)): an SVD of
the occupied orbitals' Lowdin-orthogonalized rows on A's atoms, cut at the
largest gap in the singular values. It is unique, so the PySCF reference
reproduces it without matching a localization optimum.

With the high level equal to the low level and mu -> inf, the embedded
density equals gamma_A and E equals the full KS energy: the exactness anchor.
At finite mu the A density leaks into B's space and E is low by c / mu
(c = 6.96 Ha on water/cc-pVDZ PBE-in-PBE, MEASURED 2026-11-11), so the anchor
uses the two-point extrapolation E_inf = (10 E(1e6) - E(1e5)) / 9
(``mu_extrapolated``). J and K are exact (no RI) in every piece. Bohr and
Hartree.

Run as a script: HF-in-PBE and PBE-in-PBE on water/cc-pVDZ with the O atom
as subsystem A, asserting the extrapolated anchor to 1e-8 Ha and HF-in-PBE on
a flat 99 x 590 grid against the PySCF reference to 1e-6 Ha (the grid error
budget of the pytest is 2.9e-7 Ha there).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ferric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhf_from_blocks import ROOT, WATER  # noqa: E402

PYSCF_REF = ROOT / "testdata" / "reference" / "block_ports_pyscf.json"

EXACT_JK = {"df_j_aux": "exact", "df_k_aux": "exact"}
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}


def spade_split(ints: ferric.AoIntegrals, c_occ: np.ndarray, atoms_a: list[int]):
    """(C_A, C_B, singular values): occupied orbitals rotated and split by SPADE."""
    w, u = np.linalg.eigh(ints.overlap())
    s_half = (u * np.sqrt(w)) @ u.T
    rows = [mu for mu, atom in enumerate(ints.ao_atoms()) if atom in atoms_a]
    _, sv, vt = np.linalg.svd((s_half @ c_occ)[rows], full_matrices=True)
    if len(rows) == len(ints.ao_atoms()):
        n_a = len(sv)  # A holds every atom: B is empty, no gap to cut
    else:
        n_a = int(np.argmax(-np.diff(sv))) + 1 if len(sv) > 1 else len(sv)
    c = c_occ @ vt.T
    return c[:, :n_a], c[:, n_a:], sv


def embed(
    xyz: Path,
    basis: str,
    atoms_a: list[int],
    low: str = "PBE",
    high: str | None = None,
    mu: float = 1e6,
    grid: dict | None = None,
) -> dict:
    """Projection embedding of ``high`` (None = HF) in ``low`` on ``atoms_a``.

    Returns the embedded total energy ``e_emb``, the full low-level KS energy
    ``e_low_full``, the number of A orbitals ``n_a`` and SPADE's singular
    values ``sv``. ``grid`` holds run_dft's grid keywords for both levels.
    """
    grid = grid or {}
    mol = ferric.Molecule.from_xyz(str(xyz))
    bs = ferric.BasisSet.bundled(basis)
    ints = ferric.AoIntegrals(mol, bs)
    xc = ints.xc_functional(low, **grid)
    full = ferric.run_dft(mol, bs, functional=low, **EXACT_JK, **TIGHT, **grid)
    nocc = ints.nelec // 2
    c_a, c_b, sv = spade_split(ints, full.mo_coefficients()[:, :nocc], atoms_a)
    d_a, d = (
        2.0 * c_a @ c_a.T,
        2.0 * full.mo_coefficients()[:, :nocc] @ full.mo_coefficients()[:, :nocc].T,
    )
    h, s = ints.hcore(), ints.overlap()

    def e_low(dm):
        e2, v = ints.veff(dm, xc=xc)
        return float(np.sum(dm * h)) + e2, v

    e_full, v_full = e_low(d)
    e_sub, v_sub = e_low(d_a)
    p_b = s @ c_b @ c_b.T @ s
    # The hook contract needs a symmetric V. v_full - v_sub is a difference of
    # two separately built potentials, round-off asymmetric; with B empty it
    # is all round-off and the driver's relative symmetry check rejects it.
    v_emb = v_full - v_sub + mu * p_b
    v_emb = 0.5 * (v_emb + v_emb.T)
    mol_a = ferric.Molecule.from_xyz(str(xyz), charge=2 * c_b.shape[1])

    def hook(dm):
        return float(np.sum(dm * v_emb)), v_emb

    if high is None:
        r = ferric.run_rhf(mol_a, bs, fock_hook=hook, **EXACT_JK, **TIGHT)
        e_drv = r.energy
    else:
        r = ferric.run_dft(
            mol_a, bs, functional=high, fock_hook=hook, **EXACT_JK, **TIGHT, **grid
        )
        e_drv = r.total_energy
    assert r.converged
    e_emb = e_drv - float(np.sum(d_a * v_emb)) + e_full - e_sub
    return {
        "e_emb": e_emb,
        "e_low_full": full.total_energy,
        "e_low_check": e_full + ints.nuclear_repulsion(),
        "n_a": c_a.shape[1],
        "sv": sv,
    }


def mu_extrapolated(e_1e5: float, e_1e6: float) -> float:
    """The mu -> inf limit of E = E_inf + c / mu from mu = 1e5 and 1e6."""
    return (10.0 * e_1e6 - e_1e5) / 9.0


def main() -> None:
    runs = {mu: embed(WATER, "cc-pvdz", [0], high="PBE", mu=mu) for mu in (1e5, 1e6)}
    hf_in_dft = embed(WATER, "cc-pvdz", [0], high=None)
    full = runs[1e6]["e_low_full"]
    limit = mu_extrapolated(runs[1e5]["e_emb"], runs[1e6]["e_emb"])
    print(f"SPADE singular values {np.array2string(runs[1e6]['sv'], precision=4)}")
    print(f"A orbitals            {runs[1e6]['n_a']}")
    print(f"full PBE              {full:.10f}")
    for mu, r in runs.items():
        print(
            f"PBE-in-PBE, mu={mu:.0e} {r['e_emb']:.10f}   diff {r['e_emb'] - full:.2e}"
        )
    print(f"PBE-in-PBE, mu->inf   {limit:.10f}   diff {limit - full:.2e}")
    print(f"HF-in-PBE, mu=1e6     {hf_in_dft['e_emb']:.10f}")
    assert abs(limit - full) < 1e-8
    ref = json.loads(PYSCF_REF.read_text())["molecules"]["water"]["embedding"][0]
    grid = {"grid_radial": 99, "grid_angular": 590, "grid_prune": "none"}
    ours = embed(WATER, "cc-pvdz", [0], high=None, grid=grid)
    diff = ours["e_emb"] - ref["hf_in_low"]
    print(f"HF-in-PBE, 99x590     {ours['e_emb']:.10f}   PySCF diff {diff:.1e}")
    assert abs(diff) < 1e-6


if __name__ == "__main__":
    main()
