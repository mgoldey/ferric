"""Periodic effective core potentials (semi-local ECP lattice sums) for the Gamma and k-point prototypes.

Revised-staging item "periodic ECP" (FINDINGS.md adversarial review).  The ECP is a sum of atom-centred
semi-local operators U_A = U_L(r_A) + sum_l sum_m |lm> [U_l(r_A) - U_L(r_A)] <lm|, each radial term a
Gaussian r^n e^{-zeta r^2}: every matrix element is SHORT-RANGE (no Coulomb tail, no G=0 subtlety).  So

    V_ECP_mn(k) = sum_L e^{ik.L} sum_{A, M} < phi_m(r) | U_A(r - R_A - M) | phi_n(r - L) >

is an ordinary real-space lattice sum over ECP-centre images M and orbital images L, with the molecular
ECP integral as the only kernel (PySCF `ECPscalar` over image supermolecules stands in for ferric's
libecpint shim).  Everything else is the all-electron machinery with the nuclear charge replaced by
Z_eff = Z - n_core everywhere a nucleus is a Coulomb source: Ewald E_nn, V_ne (SR and LR), the pair-FT
structure factor.  The Madelung constant v_M is a property of the lattice (a unit probe charge), not of
Z, so it is unchanged.

PySCF's Mole.atom_charges() already returns Z_eff when an ECP is assigned, and pbc_gamma.Cell reads its
Z from there, so an EcpCell only has to (1) build its Mole WITH the ECP and (2) add V_ECP to h.

Per-image resolution: V_img[M][m, L, n] is computed one ECP image M at a time (filtering Mole._ecpbas to
that image's atoms), so every truncation (ECP-image radius, orbital-image radius) is a partial sum of the
same numbers: the convergence study costs one full evaluation.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto

import pbc_kpts as PK
from pbc_gamma import Cell, _ewald, _parity

# Test-only mutation switch (monkeypatched by test_prototype.py).  None in production.
#   'z_full_vne' : V_ne (and the pair-FT structure factor) with the bare Z, E_nn with Z_eff
#   'z_full_enn' : Ewald E_nn with the bare Z, V_ne with Z_eff
#   'ecp_molecular' : the molecular ECP matrix of the home cell (M = 0 and L = 0 only): "called the molecular
#                     ECP routine on the cell basis" -- Hermitian and k-independent, so the SCF still runs
#   (M = 0 with all L is NOT Hermitian -- the (m0, nL) and (n0, m -L) elements see different centre sets --
#    and makes the k-SCF fail to converge; measured, see FINDINGS Iteration 14)
_MUTANT = None


class EcpCell(Cell):
    """pbc_gamma.Cell whose Mole carries an ECP.  Z = Z_eff (Mole.atom_charges), Zfull = bare Z."""

    def __init__(self, a, atoms, basis, ecp):
        self.ecp = ecp
        Cell.__init__(self, a, atoms, basis)
        self.mol = gto.M(atom=atoms, basis=basis, ecp=ecp, unit="B", cart=True, verbose=0, spin=_parity(atoms))
        self.Zfull = np.array([gto.charge(s) for s, _ in atoms], float)
        self.Z = self.mol.atom_charges().astype(float)
        self.ncore = np.array([self.mol.atom_nelec_core(i) for i in range(self.mol.natm)])
        assert np.allclose(self.Zfull - self.Z, self.ncore)

    def ecp_supermol(self, Ls):
        atoms = [(s, np.asarray(r) + L) for L in Ls for (s, r) in self.atoms]
        return gto.M(atom=atoms, basis=self.basis, ecp=self.ecp, unit="B", cart=True, verbose=0,
                     spin=_parity(atoms))


def ecp_ranges(cell, prec=1e-14):
    """Conservative image radii from the Gaussian product bound.  A term needs phi_m (at a home atom),
    the ECP radial Gaussian (at M) and phi_n (at L) to overlap: for the most diffuse basis exponent a and
    the most diffuse ECP exponent z, the (phi, U) product decays as exp(-a z/(a+z) d^2), so
    r_ecp = sqrt(ln(1/prec)(a+z)/(a z)) and the orbital images need r_ecp + the (phi, phi) overlap range."""
    mol = cell.mol
    a = min(mol.bas_exp(i).min() for i in range(mol.nbas))
    z = min(mol._env[mol._ecpbas[:, gto.PTR_EXP]].min(), 1e30)
    r_ecp = np.sqrt(np.log(1 / prec) * (a + z) / (a * z))
    r_orb = r_ecp + np.sqrt(2 * np.log(1 / prec) / a)
    return r_ecp, r_orb, a, z


def ecp_images(cell, rcut_ecp=None, rcut_orb=None, prec=1e-14):
    """Per-ECP-image blocks.  Returns dict(Lorb (nL,3), Lecp (nM,3), V (nM, nao, nL, nao)) with
    V[M, m, L, n] = sum_{A in image M} <phi_m 0 | U_A(. - M) | phi_n L>  (Cartesian, PySCF normalisation)."""
    r_e, r_o, _, _ = ecp_ranges(cell, prec)
    rcut_ecp = rcut_ecp or r_e
    rcut_orb = rcut_orb or r_o
    Lorb = cell.translations(rcut_orb)
    Lecp = cell.translations(rcut_ecp)
    # union, orbital images first (translations() sorts by |L|, so Lecp images already in Lorb keep their slot)
    key = lambda v: tuple(np.round(v, 8))  # noqa: E731
    pos = {key(L): i for i, L in enumerate(Lorb)}
    extra = [L for L in Lecp if key(L) not in pos]
    Lall = np.vstack([Lorb] + ([np.array(extra)] if extra else []))
    pos = {key(L): i for i, L in enumerate(Lall)}
    sm = cell.ecp_supermol(Lall)
    nat, nb0, nao = len(cell.atoms), cell.mol.nbas, cell.mol.nao
    nb_orb = len(Lorb) * nb0
    ecpbas_all = sm._ecpbas.copy()
    img_of = ecpbas_all[:, gto.ATOM_OF] // nat
    V = np.zeros((len(Lecp), nao, len(Lorb), nao))
    try:
        for iM, M in enumerate(Lecp):
            rows = ecpbas_all[img_of == pos[key(M)]]
            if len(rows) == 0:
                continue
            sm._ecpbas = rows
            blk = sm.intor("ECPscalar_cart", shls_slice=(0, nb0, 0, nb_orb))
            V[iM] = blk.reshape(nao, len(Lorb), nao)
    finally:
        sm._ecpbas = ecpbas_all
    return dict(Lorb=Lorb, Lecp=Lecp, V=V, rcut_ecp=rcut_ecp, rcut_orb=rcut_orb)


def ecp_k(img, kpts, rcut_ecp=None, rcut_orb=None):
    """V_ECP(k) = sum_{|L| <= rcut_orb} e^{ik.L} sum_{|M| <= rcut_ecp} V[M, :, L, :]  (nk, nao, nao).
    With _MUTANT == 'ecp_molecular' only M = 0, L = 0 is kept."""
    Lorb, Lecp, V = img["Lorb"], img["Lecp"], img["V"]
    mM = np.ones(len(Lecp), bool) if rcut_ecp is None else np.linalg.norm(Lecp, axis=1) <= rcut_ecp + 1e-9
    mL = np.ones(len(Lorb), bool) if rcut_orb is None else np.linalg.norm(Lorb, axis=1) <= rcut_orb + 1e-9
    if _MUTANT == "ecp_molecular":
        mM = np.linalg.norm(Lecp, axis=1) < 1e-9
        mL = np.linalg.norm(Lorb, axis=1) < 1e-9
    VL = V[mM].sum(0)[:, mL, :]  # (nao, nL, nao)
    ph = np.exp(1j * np.asarray(kpts).reshape(-1, 3) @ Lorb[mL].T)
    return np.einsum("kL,mLn->kmn", ph, VL)


def _apply_z_mutant(cell):
    """For the 'z_full_vne' mutant, switch cell.Z (read by the V_ne / structure-factor build) to the bare Z.
    The caller restores Z_eff."""
    if _MUTANT == "z_full_vne":
        cell.Z = cell.Zfull.copy()


def _enn(cell, Zeff):
    Z = cell.Zfull if _MUTANT == "z_full_enn" else Zeff
    return _ewald(cell, Z, cell.R, 1.0)


def rcut_overlap(cell, prec=1e-14):
    """S/T lattice-sum radius from the most diffuse exponent (pbc_kpts' fixed 22 Bohr is too short for
    a_min ~ 0.1: measured 3e-10 k-vs-supercell residual on HI/LANL2DZ at 22 Bohr)."""
    amin = min(cell.mol.bas_exp(i).min() for i in range(cell.mol.nbas))
    return max(22.0, np.sqrt(2 * np.log(1 / prec) / amin) + 2.0)


def gamma_ecp(cell, exxdiv="ewald", gcut=None, thresh=1e-14, img=None, rcut_1e=None):
    """Pure-AFT Gamma integrals (pbc_kpts.gamma_aft: J/K/V_ne with cell.Z = Z_eff) + Gamma V_ECP.
    Returns the gamma_aft dict with h including V_ECP, enn with Z_eff, 'Vecp', 'madelung'."""
    Zeff = cell.Z.copy()
    try:
        _apply_z_mutant(cell)
        g = PK.gamma_aft(cell, gcut=gcut, thresh=thresh, rcut_1e=rcut_1e or rcut_overlap(cell))
    finally:
        cell.Z = Zeff
    img = img or ecp_images(cell)
    Vecp = ecp_k(img, np.zeros(3))[0].real
    g["Vecp"] = Vecp
    g["h"] = g["h"] + Vecp
    g["enn"] = _enn(cell, Zeff)
    g["madelung"] = PK.kmesh_madelung(cell, (1, 1, 1)) if str(exxdiv).lower() == "ewald" else 0.0
    return g


def build_k_ecp(cell, n, exxdiv="ewald", gcut=None, thresh=1e-14, img=None, verbose=False, rcut_1e=None):
    """pbc_kpts.build_k (Z_eff via cell.Z) + V_ECP(k) on the same mesh."""
    Zeff = cell.Z.copy()
    try:
        _apply_z_mutant(cell)
        kb = PK.build_k(cell, n, exxdiv=exxdiv, gcut=gcut, thresh=thresh, verbose=verbose,
                        rcut_1e=rcut_1e or rcut_overlap(cell))
    finally:
        cell.Z = Zeff
    img = img or ecp_images(cell)
    Vk = ecp_k(img, kb["kpts"])
    kb["Vecp"] = Vk
    kb["h"] = kb["h"] + Vk
    kb["enn"] = _enn(cell, Zeff)
    return kb


def supercell_ecp(cell, n):
    t = np.array(list(np.ndindex(*n))) @ cell.a
    atoms = [(s, np.asarray(r, float) + tc) for tc in t for (s, r) in cell.atoms]
    return EcpCell(np.diag(np.asarray(n, float)) @ cell.a, atoms, cell.basis, cell.ecp)


# ------------------------------------------------------------------ molecular reference quantities
def molecular_reference(atoms, basis, ecp, conv=1e-12):
    """Molecular RHF (PySCF, cart, same basis + ECP) and the Makov-Payne a^-3 prediction for a cubic box
    under exxdiv='ewald' (G=0 dropped in J/V/E_nn, Madelung probe charge in K):
        c3 = -(4 pi / 3) Omega_I  -  (2 pi / 3) |p|^2
    Omega_I = sum_i <i|r^2|i> - sum_ij |<i|r|j>|^2 (occupied, gauge-invariant spread: the q^2 term of the
    exchange head), p = sum_A Z_eff,A R_A - 2 sum_i <i|r|i> (the total dipole: the Hartree G=0 cell of a
    neutral cell, tin-foil).  Iteration 1 checked the first term alone for H2 (p = 0)."""
    from pyscf import scf

    mol = gto.M(atom=atoms, basis=basis, ecp=ecp, unit="B", cart=True, verbose=0, spin=_parity(atoms))
    mf = scf.RHF(mol)
    mf.conv_tol = conv
    mf.kernel()
    C = mf.mo_coeff[:, mf.mo_occ > 0]
    r = np.einsum("xmn,mi,nj->xij", mol.intor("int1e_r_cart"), C, C)
    r2 = C.T @ mol.intor("int1e_r2_cart") @ C
    omega_I = np.trace(r2) - np.sum(r**2)
    p = mol.atom_charges() @ mol.atom_coords() - 2 * np.einsum("xii->x", r)
    pfull = np.array([gto.charge(s) for s, _ in atoms]) @ mol.atom_coords() - 2 * np.einsum("xii->x", r)
    c3 = -(4 * np.pi / 3) * omega_I - (2 * np.pi / 3) * p @ p
    return dict(e=mf.e_tot, omega_I=omega_I, p=p, p_full=pfull, c3=c3, nocc=C.shape[1], mf=mf)
