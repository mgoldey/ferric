"""Singlet ADC(2) excitation energies written in numpy on ferric's MO integrals.

The PySCF ``adc.RADC(method="adc(2)", method_type="ee")`` method: strict
second-order algebraic diagrammatic construction for the polarization
propagator (Schirmer, PRA 26, 2395 (1982); Trofimov, Schirmer, J. Phys. B 28,
2299 (1995)). In spin orbitals with <pq||rs> antisymmetrized and the
positive-denominator amplitudes tau_ijab = <ij||ab> / (e_a + e_b - e_i - e_j),
the singles block is

    M_ia,jb = (e_a - e_i) d_ij d_ab - <aj||bi>
              + 1/4 d_ij sum_ckl [tau_klac <kl||bc> + tau_klbc <kl||ac>]
              + 1/4 d_ab sum_cdk [tau_ikcd <jk||cd> + tau_jkcd <ik||cd>]
              - 1/2 sum_ck [tau_ikac <jk||bc> + tau_jkbc <ik||ac>],

the singles-doubles coupling is the CI one (the T1 terms of the CCSD doubles
equation), u_ijab = P(ij) sum_c <ab||cj> b_ic - P(ab) sum_k <kb||ij> b_ka, and
the doubles block is diagonal, e_a + e_b - e_i - e_j. Each root solves
omega = eig[M + 1/4 U^T (omega - D)^-1 U], projected onto singlet
spin-adapted singles (|i_alpha a_alpha> + |i_beta a_beta>)/sqrt 2.

Exact integrals from ``AoIntegrals.ao2mo``, reference orbitals from
``rhf_from_blocks.rhf``, all electrons correlated. ``order=1`` keeps only the
first-order singles block (ADC(1), which is CIS): the exactness anchor
against ``cis_from_blocks.cis_singlets``. The spin-orbital tensors scale as
(2 nmo)^4, so this port is for small bases. Hartree.

Run as a script: water/6-31G, asserting ADC(1) equals the CIS port to 1e-10
Ha and ADC(2) equals PySCF ``RADC`` to 1e-8 Ha (reference numbers from
scripts/gen_pyscf_block_ports_refs.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ferric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cis_from_blocks import cis_singlets  # noqa: E402
from rhf_from_blocks import ROOT, WATER, rhf  # noqa: E402

PYSCF_REF = ROOT / "testdata" / "reference" / "block_ports_pyscf.json"


def spin_orbital_integrals(ints: ferric.AoIntegrals, mf: dict):
    """(<pq||rs>, orbital energies, n_occ) in spin orbitals ordered
    occ-alpha, occ-beta, vir-alpha, vir-beta."""
    c, e = mf["mo_coeff"], mf["mo_energy"]
    nocc, nmo = int(np.count_nonzero(mf["mo_occ"])), len(e)
    eri = ints.ao2mo(c, c, c, c)
    occ, vir = np.arange(nocc), np.arange(nocc, nmo)
    sp = np.concatenate([occ, occ, vir, vir])
    spin = np.repeat([0, 1, 0, 1], [nocc, nocc, nmo - nocc, nmo - nocc])
    same = spin[:, None] == spin[None, :]
    g = eri[np.ix_(sp, sp, sp, sp)] * same[:, :, None, None] * same[None, None]
    phys = g.transpose(0, 2, 1, 3)
    return phys - phys.transpose(0, 1, 3, 2), e[sp], 2 * nocc


def adc2_singlets(
    ints: ferric.AoIntegrals,
    mf: dict,
    nroots: int = 3,
    order: int = 2,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    """Lowest ``nroots`` singlet ADC(``order``) excitation energies (Hartree).

    ``order`` is 1 (the first-order singles block alone) or 2. Each ADC(2)
    root is the fixed point of its own eigenvalue of the omega-dependent
    singles matrix, started from the static (second-order singles) root."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    g, eps, no = spin_orbital_integrals(ints, mf)
    o, v = slice(0, no), slice(no, None)
    eo, ev = eps[o], eps[v]
    nv = len(ev)
    io, iv = np.eye(no), np.eye(nv)
    m = -np.einsum("ajbi->iajb", g[v, o, v, o])
    m += np.einsum("ij,ab->iajb", io, np.diag(ev)) - np.einsum(
        "ij,ab->iajb", np.diag(eo), iv
    )
    dbl = (
        ev[None, None, :, None]
        + ev[None, None, None, :]
        - eo[:, None, None, None]
        - eo[None, :, None, None]
    )
    if order >= 2:
        tau = g[o, o, v, v] / dbl
        xv = np.einsum("klac,klbc->ab", tau, g[o, o, v, v])
        xo = np.einsum("ikcd,jkcd->ij", tau, g[o, o, v, v])
        y = np.einsum("ikac,jkbc->iajb", tau, g[o, o, v, v])
        m += 0.25 * np.einsum("ij,ab->iajb", io, xv + xv.T)
        m += 0.25 * np.einsum("ab,ij->iajb", iv, xo + xo.T)
        m -= 0.5 * (y + y.transpose(2, 3, 0, 1))
        gvvvo, govoo = g[v, v, v, o], g[o, v, o, o]
    m = m.reshape(no * nv, no * nv)
    nocc, nvir = no // 2, nv // 2
    proj = np.zeros((no, nv, nocc, nvir))
    for s in (0, 1):
        proj[s * nocc : (s + 1) * nocc, s * nvir : (s + 1) * nvir] = np.einsum(
            "ij,ab->iajb", np.eye(nocc), np.eye(nvir)
        ) / np.sqrt(2.0)
    proj = proj.reshape(no * nv, nocc * nvir)

    def coupling_slab(i, j):
        """U[(ij), (ab), (kc)] of one occupied pair as an (nv*nv, no*nv)
        matrix: d_ik <ab||cj> - d_jk <ab||ci> - d_ac <kb||ij> + d_bc <ka||ij>.
        The d terms are written as slices, so no identity tensor is formed."""
        s = np.zeros((nv, nv, no, nv))
        s[:, :, i, :] += gvvvo[:, :, :, j]
        s[:, :, j, :] -= gvvvo[:, :, :, i]
        s[iv_idx, :, :, iv_idx] -= govoo[:, :, i, j].T
        s[:, iv_idx, :, iv_idx] += govoo[:, :, i, j].T
        return s.reshape(nv * nv, no * nv) @ proj

    iv_idx = np.arange(nv)

    def singlet_eigs(omega):
        mat = proj.T @ m @ proj
        if order >= 2:
            for i in range(no):
                for j in range(no):
                    s = coupling_slab(i, j)
                    mat = mat + 0.25 * (s.T / (omega - dbl[i, j].ravel())) @ s
        return np.linalg.eigvalsh(mat)

    if order < 2:
        return singlet_eigs(0.0)[:nroots]
    roots = np.linalg.eigvalsh(proj.T @ m @ proj)[:nroots]
    for k in range(len(roots)):
        for _ in range(max_iter):
            new = singlet_eigs(roots[k])[k]
            converged = abs(new - roots[k]) < tol
            roots[k] = new
            if converged:
                break
        else:
            raise RuntimeError(f"ADC(2) root {k} did not converge")
    return roots


def main() -> None:
    mol = ferric.Molecule.from_xyz(str(WATER))
    ints = ferric.AoIntegrals(mol, ferric.BasisSet.bundled("6-31g"))
    mf = rhf(ints)
    adc1 = adc2_singlets(ints, mf, order=1)
    cis = cis_singlets(ints, mf, 3)
    adc2 = adc2_singlets(ints, mf)
    print("ADC(1)      ", np.array2string(adc1, precision=10))
    print("CIS port    ", np.array2string(cis, precision=10))
    print("ADC(2)      ", np.array2string(adc2, precision=10))
    assert np.abs(adc1 - cis).max() < 1e-10
    ref = json.loads(PYSCF_REF.read_text())["molecules"]["water"]["adc2"]
    want = np.asarray(ref["excitation_energies"])
    print("PySCF RADC  ", np.array2string(want, precision=10))
    assert np.abs(adc2 - want).max() < 1e-8


if __name__ == "__main__":
    main()
