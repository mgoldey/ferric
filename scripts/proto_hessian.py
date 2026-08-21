#!/usr/bin/env python3
"""Analytic RHF Hessian prototype for ferric.

Builds the full analytic RHF Hessian from scratch using PySCF's integral
derivative API, then validates against PySCF's production hessian.rhf.

Components:
  1. Nuclear repulsion Hessian (d²V_nn/dR_A dR_B)
  2. Skeleton 1-electron Hessian (kinetic + nuclear attraction 2nd derivatives)
  3. Overlap second derivatives × energy-weighted density
  4. Skeleton 2-electron Hessian (J and K second derivatives)
  5. CPKS orbital response (via pyscf.scf.cphf.solve)

Each component is validated individually and then the total is compared
to PySCF's analytic Hessian.

Usage:
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/proto_hessian.py
"""
import os
import sys
import numpy as np
from functools import reduce

sys.path.insert(0, os.environ.get("PYSCF_PATH", os.path.expanduser("~/qc/pyscf")))
from pyscf import gto, scf, lib
from pyscf.scf import cphf
from pyscf.hessian import rhf as pyscf_hess_rhf


# ---------------------------------------------------------------------------
# 1. Nuclear repulsion Hessian
# ---------------------------------------------------------------------------
def hess_nuc(mol):
    """d²V_nn / dR_A dR_B, shape (natm, natm, 3, 3)."""
    natm = mol.natm
    h = np.zeros((natm, natm, 3, 3))
    charges = np.array([mol.atom_charge(i) for i in range(natm)])
    coords = np.array([mol.atom_coord(i) for i in range(natm)])

    for i in range(natm):
        for j in range(natm):
            if i == j:
                continue
            rij = coords[i] - coords[j]
            r = np.linalg.norm(rij)
            qi, qj = charges[i], charges[j]
            # Off-diagonal: d²(qi*qj/|r_i-r_j|)/dR_i_x dR_j_y
            # = qi*qj * (-3 r_x r_y / r^5 + delta_xy / r^3)
            h[i, j] = qi * qj * (-3.0 * np.outer(rij, rij) / r**5
                                 + np.eye(3) / r**3)
            h[i, i] -= h[i, j]
    return h


# ---------------------------------------------------------------------------
# 2. Skeleton 1-electron Hessian (hcore = kinetic + nuclear attraction)
# ---------------------------------------------------------------------------
def skeleton_hess_1e(mol, dm, hessobj):
    """Skeleton one-electron Hessian: Tr[D · h^{(2)}_{AB}], shape (natm, natm, 3, 3).

    Uses PySCF's hcore_generator which correctly decomposes nuclear attraction
    derivatives by atom pair (each nuclear center contributes only to the
    pairs involving that center, via int1e_ipiprinv with rinv_at_nucleus).
    """
    natm = mol.natm
    hcore_deriv = hessobj.hcore_generator(mol)

    hess = np.zeros((natm, natm, 3, 3))
    for ia in range(natm):
        for ja in range(ia + 1):
            h = hcore_deriv(ia, ja)
            hess[ia, ja] += np.einsum('xypq,pq->xy', h, dm)
        for ja in range(ia):
            hess[ja, ia] = hess[ia, ja].T
    return hess


# ---------------------------------------------------------------------------
# 3. Overlap second derivative × energy-weighted density
# ---------------------------------------------------------------------------
def skeleton_hess_ovlp(mol, dme):
    """Overlap Hessian contribution: -Tr[W · S^{(2)}], shape (natm, natm, 3, 3).

    dme = energy-weighted density matrix W_μν = Σ_i ε_i C_μi C_νi (× 2 for RHF).
    """
    natm = mol.natm
    nao = mol.nao
    aoslices = mol.aoslice_by_atom()

    s1aa = mol.intor('int1e_ipipovlp', comp=9).reshape(3, 3, nao, nao)
    s1ab = mol.intor('int1e_ipovlpip', comp=9).reshape(3, 3, nao, nao)

    hess = np.zeros((natm, natm, 3, 3))
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        hess[ia, ia] -= np.einsum('xypq,pq->xy', s1aa[:, :, p0:p1], dme[p0:p1]) * 2
        for ja in range(ia + 1):
            q0, q1 = aoslices[ja][2:]
            hess[ia, ja] -= np.einsum('xypq,pq->xy', s1ab[:, :, p0:p1, q0:q1],
                                      dme[p0:p1, q0:q1]) * 2
    for ia in range(natm):
        for ja in range(ia):
            hess[ja, ia] = hess[ia, ja].T
    return hess


# ---------------------------------------------------------------------------
# 4. Skeleton 2-electron Hessian (Coulomb J and exchange K)
# ---------------------------------------------------------------------------
def skeleton_hess_2e(mol, dm):
    """Skeleton two-electron Hessian: J and K second derivative contributions.

    Uses PySCF's _vhf.direct_bindm for the integral contractions (these
    are C-level, not something we'd reimplement in a Python prototype).
    The logic mirrors PySCF's _partial_hess_ejk exactly.
    """
    from pyscf.scf import _vhf

    natm = mol.natm
    nao = mol.nao
    aoslices = mol.aoslice_by_atom()

    # Same-atom second derivative integrals (∂²/∂A_x ∂A_y on first index pair)
    vj1_diag, vk1_diag = _get_jk(
        mol, 'int2e_ipip1', 9, 's2kl',
        ['lk->s1ij', dm,    # J
         'jk->s1il', dm],   # K
        vhfopt=pyscf_hess_rhf._make_vhfopt(mol, dm, 'ipip1', 'int2e_ipip1ipip2')
    )
    vj1_diag = vj1_diag.reshape(3, 3, nao, nao)
    vk1_diag = vk1_diag.reshape(3, 3, nao, nao)

    ip1ip2_opt = pyscf_hess_rhf._make_vhfopt(mol, dm, 'ip1ip2', 'int2e_ip1ip2')
    ipvip1_opt = pyscf_hess_rhf._make_vhfopt(mol, dm, 'ipvip1', 'int2e_ipvip1ipvip2')

    ej = np.zeros((natm, natm, 3, 3))
    ek = np.zeros((natm, natm, 3, 3))

    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        shl0, shl1 = aoslices[ia][:2]
        shls_slice = (shl0, shl1) + (0, mol.nbas) * 3

        # Cross-atom: ∂/∂A on first pair, ∂/∂B on second pair
        vj1, vk1, vk2 = _get_jk(
            mol, 'int2e_ip1ip2', 9, 's1',
            ['ji->s1kl', dm[:, p0:p1],
             'li->s1kj', dm[:, p0:p1],
             'lj->s1ki', dm],
            shls_slice=shls_slice, vhfopt=ip1ip2_opt
        )
        vk1[:, :, p0:p1] += vk2

        vj2, vk2 = _get_jk(
            mol, 'int2e_ipvip1', 9, 's2kl',
            ['lk->s1ij', dm,
             'li->s1kj', dm[:, p0:p1]],
            shls_slice=shls_slice, vhfopt=ipvip1_opt
        )
        vj1[:, :, p0:p1] += vj2.transpose(0, 2, 1) * 0.5
        vk1 += vk2.transpose(0, 2, 1)
        vj1 = vj1.reshape(3, 3, nao, nao)
        vk1 = vk1.reshape(3, 3, nao, nao)

        # Same-atom diagonal
        ej[ia, ia] += np.einsum('xypq,pq->xy', vj1_diag[:, :, p0:p1], dm[p0:p1]) * 2
        ek[ia, ia] += np.einsum('xypq,pq->xy', vk1_diag[:, :, p0:p1], dm[p0:p1])

        for ja in range(ia + 1):
            q0, q1 = aoslices[ja][2:]
            ej[ia, ja] += np.einsum('xypq,pq->xy', vj1[:, :, q0:q1], dm[q0:q1]) * 4
            ek[ia, ja] += np.einsum('xypq,pq->xy', vk1[:, :, q0:q1], dm[q0:q1])

        for ja in range(ia):
            ej[ja, ia] = ej[ia, ja].T
            ek[ja, ia] = ek[ia, ja].T

    return ej, ek


def _get_jk(mol, intor, comp, aosym, script_dms,
            shls_slice=None, cintopt=None, vhfopt=None):
    """Thin wrapper around PySCF's _vhf.direct_bindm."""
    from pyscf.scf import _vhf
    intor = mol._add_suffix(intor)
    scripts = script_dms[::2]
    dms = script_dms[1::2]
    vs = _vhf.direct_bindm(intor, aosym, scripts, dms, comp,
                           mol._atm, mol._bas, mol._env, vhfopt=vhfopt,
                           cintopt=cintopt, shls_slice=shls_slice)
    for k, script in enumerate(scripts):
        if 's2' in script:
            hermi = 1
        elif 'a2' in script:
            hermi = 2
        else:
            continue
        shape = vs[k].shape
        if shape[-2] == shape[-1]:
            if comp > 1:
                for i in range(comp):
                    lib.hermi_triu(vs[k][i], hermi=hermi, inplace=True)
            else:
                lib.hermi_triu(vs[k], hermi=hermi, inplace=True)
    return vs


# ---------------------------------------------------------------------------
# 5. CPKS response contribution
# ---------------------------------------------------------------------------
def cpks_response(mol, mf, hess_partial):
    """CPKS orbital response contribution to the Hessian.

    Returns the full electronic Hessian including the response terms.
    """
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = int(mo_occ.sum()) // 2
    mocc = mo_coeff[:, :nocc]

    aoslices = mol.aoslice_by_atom()

    # Build h1ao: first-derivative Fock matrix in AO basis for each atom
    dm0 = np.dot(mocc, mocc.T) * 2
    hcore_grad = mf.nuc_grad_method().hcore_generator(mol)

    h1ao_list = [None] * mol.natm
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        shl0, shl1 = aoslices[ia][:2]
        shls_slice = (shl0, shl1) + (0, mol.nbas) * 3
        vj1, vj2, vk1, vk2 = _get_jk(
            mol, 'int2e_ip1', 3, 's2kl',
            ['ji->s2kl', -dm0[:, p0:p1],
             'lk->s1ij', -dm0,
             'li->s1kj', -dm0[:, p0:p1],
             'jk->s1il', -dm0],
            shls_slice=shls_slice
        )
        vhf = vj1 - vk1 * 0.5
        vhf[:, p0:p1] += vj2 - vk2 * 0.5
        h1 = vhf + vhf.transpose(0, 2, 1)
        h1 += hcore_grad(ia)
        h1ao_list[ia] = h1

    # Generate CPKS response function
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)

    def fx(mo1):
        mo1 = mo1.reshape(-1, nmo, nocc)
        nset = len(mo1)
        dm1 = np.empty((nset, nao, nao))
        for i, x in enumerate(mo1):
            d = reduce(np.dot, (mo_coeff, x * 2, mocc.T))
            dm1[i] = d + d.T
        v1 = vresp(dm1)
        v1vo = np.empty_like(mo1)
        for i, x in enumerate(v1):
            v1vo[i] = reduce(np.dot, (mo_coeff.T, x, mocc))
        return v1vo

    # Build overlap gradient
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    def _ao2mo(mat):
        return np.array([reduce(np.dot, (mo_coeff.T, x, mocc)) for x in mat])

    # Solve CPKS for each atom
    s1vo_all = []
    h1vo_all = []
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        s1ao = np.zeros((3, nao, nao))
        s1ao[:, p0:p1] += s1a[:, p0:p1]
        s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
        s1vo_all.append(_ao2mo(s1ao))
        h1vo_all.append(_ao2mo(h1ao_list[ia]))

    h1vo = np.vstack(h1vo_all)
    s1vo = np.vstack(s1vo_all)

    mo1, e1 = cphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo)
    mo1 = np.einsum('pq,xqi->xpi', mo_coeff, mo1).reshape(-1, 3, nao, nocc)
    e1 = e1.reshape(-1, 3, nocc, nocc)

    mo1s = {}
    e1s = {}
    for ia in range(mol.natm):
        mo1s[ia] = mo1[ia]
        e1s[ia] = e1[ia]

    # Assemble CPKS contribution
    de2 = hess_partial.copy()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        s1ao = np.zeros((3, nao, nao))
        s1ao[:, p0:p1] += s1a[:, p0:p1]
        s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
        s1oo = np.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)

        for ja in range(ia + 1):
            # *2 for double occupancy, *2 for +c.c.
            dm1 = np.einsum('ypi,qi->ypq', mo1s[ja], mocc)
            de2[ia, ja] += np.einsum('xpq,ypq->xy', h1ao_list[ia], dm1) * 4
            dm1 = np.einsum('ypi,qi,i->ypq', mo1s[ja], mocc,
                            mo_energy[:nocc])
            de2[ia, ja] -= np.einsum('xpq,ypq->xy', s1ao, dm1) * 4
            de2[ia, ja] -= np.einsum('xpq,ypq->xy', s1oo, e1s[ja]) * 2

        for ja in range(ia):
            de2[ja, ia] = de2[ia, ja].T

    return de2


# ---------------------------------------------------------------------------
# Main: validate on H2/STO-3G and H2O/STO-3G
# ---------------------------------------------------------------------------
def run_test(atom_str, basis, label):
    print(f"\n{'=' * 70}")
    print(f"  {label}: {basis}")
    print(f"{'=' * 70}")

    mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    print(f"  RHF energy: {mf.e_tot:.10f} Ha")

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nocc = int(mo_occ.sum()) // 2
    mocc = mo_coeff[:, :nocc]

    # Density matrix and energy-weighted density
    dm = np.dot(mocc, mocc.T) * 2
    dme = np.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[:nocc]) * 2

    # --- Component 1: Nuclear repulsion Hessian ---
    h_nuc = hess_nuc(mol)
    h_nuc_ref = pyscf_hess_rhf.hess_nuc(mol)
    err_nuc = np.max(np.abs(h_nuc - h_nuc_ref))
    print(f"\n  Nuclear Hessian max error vs PySCF: {err_nuc:.2e}")
    print(f"  Nuclear Hessian max element:        {np.max(np.abs(h_nuc)):.4f}")

    # --- Component 2: Skeleton 1-electron Hessian ---
    hessobj = mf.Hessian()
    hessobj.verbose = 0
    h_1e = skeleton_hess_1e(mol, dm, hessobj)

    # --- Component 3: Overlap second derivative contribution ---
    h_ovlp = skeleton_hess_ovlp(mol, dme)

    # --- Component 4: Skeleton 2-electron Hessian ---
    ej, ek = skeleton_hess_2e(mol, dm)

    # --- Combine skeleton (partial) electronic Hessian ---
    h_elec_partial = h_1e + h_ovlp + ej - ek
    print(f"\n  Component magnitudes (Frobenius norm):")
    print(f"    Nuclear repulsion:   {np.linalg.norm(h_nuc):12.6f}")
    print(f"    1e skeleton:         {np.linalg.norm(h_1e):12.6f}")
    print(f"    Overlap × W:         {np.linalg.norm(h_ovlp):12.6f}")
    print(f"    2e Coulomb (J):      {np.linalg.norm(ej):12.6f}")
    print(f"    2e Exchange (K):     {np.linalg.norm(ek):12.6f}")

    # Validate partial electronic Hessian against PySCF
    h_partial_ref = pyscf_hess_rhf.partial_hess_elec(
        hessobj, mo_energy, mo_coeff, mo_occ)
    err_partial = np.max(np.abs(h_elec_partial - h_partial_ref))
    print(f"\n  Partial electronic Hessian max error: {err_partial:.2e}")

    # --- Component 5: CPKS response ---
    h_elec_full = cpks_response(mol, mf, h_elec_partial)

    # --- Total Hessian = electronic + nuclear ---
    h_total = h_elec_full + h_nuc

    # Reference from PySCF
    h_ref = mf.Hessian()
    h_ref.verbose = 0
    h_pyscf = h_ref.kernel()  # shape (natm, natm, 3, 3)

    err_total = np.max(np.abs(h_total - h_pyscf))
    print(f"\n  ═══ TOTAL HESSIAN ═══")
    print(f"  Max error vs PySCF:       {err_total:.2e}")
    print(f"  Max Hessian element:      {np.max(np.abs(h_total)):.6f}")
    print(f"  Relative max error:       {err_total / np.max(np.abs(h_total)):.2e}")

    # Detailed comparison
    natm = mol.natm
    print(f"\n  Per-block max errors:")
    for i in range(natm):
        for j in range(natm):
            blk_err = np.max(np.abs(h_total[i, j] - h_pyscf[i, j]))
            blk_mag = np.max(np.abs(h_pyscf[i, j]))
            if blk_mag > 1e-10:
                print(f"    [{i},{j}]: err={blk_err:.2e}  mag={blk_mag:.6f}  "
                      f"rel={blk_err / blk_mag:.2e}")

    # Mass-weighted Hessian → frequencies (cm⁻¹)
    from pyscf.hessian.thermo import harmonic_analysis
    freq_info = harmonic_analysis(mol, h_pyscf)
    freqs = freq_info['freq_wavenumber']
    print(f"\n  Vibrational frequencies (cm⁻¹, PySCF reference):")
    for i, f in enumerate(freqs):
        if abs(f) > 10:  # skip trans/rot
            print(f"    ν_{i+1} = {f:.1f}")

    freq_info_ours = harmonic_analysis(mol, h_total)
    freqs_ours = freq_info_ours['freq_wavenumber']
    print(f"\n  Vibrational frequencies (cm⁻¹, our Hessian):")
    for i, f in enumerate(freqs_ours):
        if abs(f) > 10:
            print(f"    ν_{i+1} = {f:.1f}")

    # Frequency errors for real modes
    real_mask = np.abs(freqs) > 10
    if np.any(real_mask):
        freq_err = np.max(np.abs(freqs[real_mask] - freqs_ours[real_mask]))
        print(f"\n  Max frequency error: {freq_err:.4f} cm⁻¹")

    ok = err_total < 1e-6
    status = "PASS" if ok else "FAIL"
    print(f"\n  *** {status} *** (threshold 1e-6)")
    return ok


if __name__ == "__main__":
    np.set_printoptions(precision=8, linewidth=120)

    results = []
    results.append(run_test('H 0 0 0; H 0 0 0.74', 'sto-3g', 'H2'))
    results.append(run_test(
        'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
        'sto-3g', 'H2O'
    ))
    results.append(run_test(
        'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
        'cc-pvdz', 'H2O (cc-pVDZ)'
    ))

    print(f"\n{'=' * 70}")
    print(f"  Summary: {sum(results)}/{len(results)} passed")
    print(f"{'=' * 70}")
    if not all(results):
        sys.exit(1)
