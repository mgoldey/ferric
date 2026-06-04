#!/usr/bin/env python3
"""Clean-room analytic relaxed-MP2 polarizability on REAL PySCF integrals.

Replaces the earlier random-integral clean-room (which used a non-physical
I=BᵀB and silently broke the CPHF/response identities). Here everything is a
real RHF/MP2 on water/STO-3G, so standard response relations hold and the
oracle is trustworthy.

Validation ladder (each rung must pass before the next):
  R0  RHF energy sane.
  R1  HF analytic α (CPHF) == HF finite-field α.
  R2  MP2 relaxed dipole == PySCF mp.MP2().make_rdm1(relaxed) dipole.
  R3  MP2 relaxed α (analytic response) == FF of the relaxed dipole.
Then decompose ∂z term-by-term to find/fix the Rust bug.

Run: PYTHONPATH=/home/matt/qc/pyscf python3 scripts/cpks/mp2_alpha_pyscf.py
"""
import numpy as np
from pyscf import gto, scf, mp, ao2mo

np.set_printoptions(precision=6, suppress=True, linewidth=140)

# ---- real small system ----
mol = gto.M(
    atom="O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161",
    basis="sto-3g", unit="Angstrom", verbose=0,
)
mf = scf.RHF(mol).run(conv_tol=1e-12)
print("R0: RHF E =", mf.e_tot)

C = mf.mo_coeff
e = mf.mo_energy
nmo = C.shape[1]
nocc = mol.nelectron // 2
nvir = nmo - nocc
O = slice(0, nocc); Vv = slice(nocc, nmo)
print("    nmo=%d nocc=%d nvir=%d" % (nmo, nocc, nvir))

# MO 2e integrals (chemist (pq|rs)), full nmo^4 (small).
eri_ao = mol.intor("int2e")
Imo = ao2mo.incore.full(eri_ao, C, compact=False).reshape(nmo, nmo, nmo, nmo)

# AO dipole -> MO (length gauge, nuclear-independent for polarizability).
dip_ao = mol.intor("int1e_r").reshape(3, nmo, nmo)  # x,y,z
r_mo = np.einsum("xpq,pi,qj->xij", dip_ao, C, C)     # MO dipole per axis

# core hamiltonian in MO (for field-SCF in MO basis)
hcore_ao = mf.get_hcore()
hcore_mo = C.T @ hcore_ao @ C
S_ao = mf.get_ovlp()


def t2_amp(Imatrix, evec):
    o = range(nocc); v = range(nocc, nmo)
    t = np.zeros((nocc, nvir, nocc, nvir))
    for i in o:
        for a in v:
            for j in o:
                for b in v:
                    d = evec[i] + evec[j] - evec[a] - evec[b]
                    t[i, a - nocc, j, b - nocc] = Imatrix[i, a, j, b] / d
    return t


def e_mp2(Imatrix, evec):
    t = t2_amp(Imatrix, evec)
    s = 0.0
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    K = 2 * Imatrix[i, nocc + a, j, nocc + b] - Imatrix[i, nocc + b, j, nocc + a]
                    s += t[i, a, j, b] * K
    return s


# sanity vs PySCF MP2
pt = mp.MP2(mf).run()
print("R0: E_MP2 mine =", e_mp2(Imo, e), " pyscf =", pt.e_corr)

# ---- PySCF relaxed MP2 dipole (the gold oracle for R2) ----
# relaxed 1-PDM in MO, then dipole = -Tr[D_ao r_ao] + nuclear.
dm1_mo = pt.make_rdm1()        # relaxed by default in recent pyscf? check both
dm1_ao = C @ dm1_mo @ C.T
nucl = np.einsum("g,gx->x", mol.atom_charges(), mol.atom_coords())  # Z*R
def dipole_from_dm_ao(dm_ao):
    el = -np.einsum("xpq,pq->x", dip_ao, dm_ao)
    return el + nucl
print("R2-ref: PySCF relaxed MP2 dipole =", dipole_from_dm_ao(dm1_ao))

# ===========================================================================
# R1: HF analytic α (CPHF) vs HF finite-field α.  Pin the operator + contraction.
# ===========================================================================
def hess_M():
    # static CPHF orbital Hessian (A+B): (ea-ei)δ + 4(ai|bj)-(ab|ij)-(aj|bi)
    M = np.zeros((nvir, nocc, nvir, nocc))
    for a in range(nvir):
        for i in range(nocc):
            for b in range(nvir):
                for j in range(nocc):
                    M[a,i,b,j] = (4*Imo[nocc+a,i,nocc+b,j]
                                  - Imo[nocc+a,nocc+b,i,j]
                                  - Imo[nocc+a,j,nocc+b,i])
            M[a,i,a,i] += e[nocc+a]-e[i]
    return M.reshape(nvir*nocc, nvir*nocc)

M = hess_M()

def hf_ff_alpha(axis_p, axis_q, h=1e-4):
    # FF: re-solve RHF with hcore += -F dip[axis_q]; read dipole[axis_p].
    def dip_p(F):
        mf2 = scf.RHF(mol)
        mf2.get_hcore = lambda *a: hcore_ao - F*dip_ao[axis_q]
        mf2.conv_tol = 1e-12
        mf2.kernel()
        dm = mf2.make_rdm1()
        return -np.einsum("pq,pq->", dip_ao[axis_p], dm) + nucl[axis_p]
    return -(dip_p(h)-dip_p(-h))/(2*h)

# analytic HF α: M U^q = -r^q_vo ; α_pq = -? Σ U^q r^p . Pin factor vs FF.
def cphf_U(axis):
    rvo = np.array([[r_mo[axis,nocc+a,i] for i in range(nocc)] for a in range(nvir)])
    return np.linalg.solve(M, (-rvo).reshape(-1)).reshape(nvir,nocc)

print("\n=== R1: HF α analytic vs FF ===")
for axis in range(3):
    U = cphf_U(axis)
    rvo = np.array([[r_mo[axis,nocc+a,i] for i in range(nocc)] for a in range(nvir)])
    ff = hf_ff_alpha(axis, axis)
    print(f"  axis {axis}: FF={ff:+.5f}  -4ΣUr={-4*np.sum(U*rvo):+.5f}  -2ΣUr={-2*np.sum(U*rvo):+.5f}")

# ===========================================================================
# R2/R3: MP2 relaxed density + analytic relaxed α, validated vs PySCF relaxed α.
# ===========================================================================
def lagrangian(Imatrix, t):
    # 4-term integral Lagrangian L_ck (matches ferric build_lagrangian integral part).
    L = np.zeros((nvir, nocc))
    for c in range(nvir):
        for k in range(nocc):
            g = 0.0
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        g += t[k,a,j,b]*(2*Imatrix[nocc+c,nocc+a,j,nocc+b]-Imatrix[nocc+c,nocc+b,j,nocc+a])
            for i in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        g += t[i,a,k,b]*(2*Imatrix[i,nocc+a,nocc+c,nocc+b]-Imatrix[i,nocc+b,nocc+c,nocc+a])
            for i in range(nocc):
                for j in range(nocc):
                    for b in range(nvir):
                        g -= t[i,c,j,b]*(2*Imatrix[i,k,j,nocc+b]-Imatrix[i,nocc+b,j,k])
            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvir):
                        g -= t[i,a,j,c]*(2*Imatrix[i,nocc+a,j,k]-Imatrix[i,k,j,nocc+a])
            L[c,k] = g
    return L

def relaxed_dm_mo(Imatrix, evec):
    t = t2_amp(Imatrix, evec)
    Poo = -np.einsum('iakb,jakb->ij', t, 2*t - t.transpose(0,3,2,1))
    Pvv =  np.einsum('iajc,ibjc->ab', t, 2*t - t.transpose(2,1,0,3))
    L = lagrangian(Imatrix, t)
    z = np.linalg.solve(M, L.reshape(-1)).reshape(nvir,nocc)   # NB: M built from unpert. Imo
    D = np.zeros((nmo,nmo))
    for i in range(nocc): D[i,i]+=2.0
    D[O,O]+=Poo; D[Vv,Vv]+=Pvv
    for a in range(nvir):
        for i in range(nocc):
            D[nocc+a,i]+=z[a,i]; D[i,nocc+a]+=z[a,i]
    return D, z, Poo, Pvv, t

# R2: does my relaxed dm (in MO) reproduce PySCF's relaxed dipole?
D0,_,_,_,_ = relaxed_dm_mo(Imo, e)
D0_ao = C @ D0 @ C.T
print("\n=== R2: relaxed MP2 dipole ===")
print("  mine  =", dipole_from_dm_ao(D0_ao))
print("  pyscf =", dipole_from_dm_ao(dm1_ao))

# Decompose PySCF relaxed dm1_mo vs mine, block by block.
Dmine, zmine, Poo_m, Pvv_m, t = relaxed_dm_mo(Imo, e)
print("\n=== R2 decomp: PySCF dm1_mo vs mine (MO basis) ===")
print("oo block PySCF diag:", np.round(np.diag(dm1_mo)[O],5), " mine:", np.round(np.diag(Dmine)[O],5))
print("vv block PySCF diag:", np.round(np.diag(dm1_mo)[Vv],5), " mine:", np.round(np.diag(Dmine)[Vv],5))
ov_ps = dm1_mo[Vv,O]; ov_me = Dmine[Vv,O]
print("ov(vir,occ) PySCF:\n", np.round(ov_ps,5))
print("ov(vir,occ) mine (z):\n", np.round(ov_me,5))
print("ratio ov mine/pyscf:\n", np.round(ov_me/np.where(np.abs(ov_ps)>1e-9,ov_ps,np.nan),4))
# Is mine's oo/vv (the Poo/Pvv) matching? (these are unrelaxed-density, should match)
print("‖oo mine-pyscf‖=", np.max(np.abs((Dmine-dm1_mo)[O,O])),
      " ‖vv‖=", np.max(np.abs((Dmine-dm1_mo)[Vv,Vv])),
      " ‖ov‖=", np.max(np.abs((Dmine-dm1_mo)[Vv,O])))

# ===========================================================================
# CORRECT ORACLE: relaxed dipole = dE_MP2_total/dh = FF of PySCF total MP2 energy
# w.r.t. the dipole field. (make_rdm1 above was UNRELAXED, ov=0.)
# ===========================================================================
def pyscf_mp2_etot(Ffield, axis):
    m = scf.RHF(mol)
    m.get_hcore = lambda *a: hcore_ao - Ffield*dip_ao[axis]
    m.conv_tol = 1e-12
    m.kernel()
    p = mp.MP2(m).run()
    return m.e_tot + p.e_corr

def pyscf_relaxed_dipole(axis, h=1e-4):
    # μ_axis = -dE/dF  (field couples h += -F r, so dE/dF = -<r>... sign: E(F)=E0 -F<r>+...
    return -(pyscf_mp2_etot(h,axis)-pyscf_mp2_etot(-h,axis))/(2*h)

print("\n=== CORRECT R2 oracle: relaxed dipole via FF of PySCF E_MP2 ===")
mu_relaxed = np.array([pyscf_relaxed_dipole(ax) for ax in range(3)])
print("  pyscf relaxed μ =", mu_relaxed)
print("  (unrelaxed make_rdm1 gave", dipole_from_dm_ao(dm1_ao),")")

# Now test mine WITH the ×2 fix on Poo/Pvv (PySCF uses doo+doo.T = 2×sym).
def relaxed_dm_mo_v2(Imatrix, evec):
    t = t2_amp(Imatrix, evec)
    Poo = -np.einsum('iakb,jakb->ij', t, 2*t - t.transpose(0,3,2,1))
    Pvv =  np.einsum('iajc,ibjc->ab', t, 2*t - t.transpose(2,1,0,3))
    L = lagrangian(Imatrix, t)
    z = np.linalg.solve(M, L.reshape(-1)).reshape(nvir,nocc)
    D = np.zeros((nmo,nmo))
    for i in range(nocc): D[i,i]+=2.0
    D[O,O]  += Poo + Poo.T          # ×2 via +.T  (PySCF doo+doo.T)
    D[Vv,Vv]+= Pvv + Pvv.T
    for a in range(nvir):
        for i in range(nocc):
            D[nocc+a,i]+=z[a,i]; D[i,nocc+a]+=z[a,i]
    return D
Dv2 = relaxed_dm_mo_v2(Imo,e); Dv2_ao = C@Dv2@C.T
print("  mine v2 (Poo+Poo.T) relaxed μ =", dipole_from_dm_ao(Dv2_ao))

# ===========================================================================
# EINSUM rewrite mirroring PySCF _gamma1_intermediates EXACTLY (your suggestion).
# PySCF t2 indexing: t2[i,j,a,b]. Ours: t[i,a,j,b]. Convert once.
#   P_vv(dm1vir)_ba = 2 einsum('jca,jcb',l2,t2) - einsum('jca,jbc',l2,t2)  [t2[i]=t2i: j,a,b? ]
# In PySCF t2i = t2[i] has shape (nocc_j, nvir_a, nvir_b) → indices (j,a,b).
# So full t2 is t2[i,j,a,b]. Build that from our t[i,a,j,b].
t_ijab = t.transpose(0,2,1,3)   # (i,a,j,b) -> (i,j,a,b)
# PySCF: doo_ij = 2 'iab,jab' - 'iab,jba'  (sum over the OTHER occ + both vir)
#        but note their loop is over i (the 4th hidden index). Full contraction:
doo = 2*np.einsum('ikab,jkab->ij', t_ijab, t_ijab) - np.einsum('ikab,jkba->ij', t_ijab, t_ijab)
dvv = 2*np.einsum('ijca,ijcb->ab', t_ijab, t_ijab) - np.einsum('ijca,ijbc->ab', t_ijab, t_ijab)
Poo_es = -doo
Pvv_es = dvv
print("\n=== EINSUM P-blocks vs my loop version ===")
print("‖Poo_einsum - Poo_loop‖ =", np.max(np.abs(Poo_es - Poo_m)))
print("‖Pvv_einsum - Pvv_loop‖ =", np.max(np.abs(Pvv_es - Pvv_m)))
print("Pvv_einsum diag:", np.round(np.diag(Pvv_es),5), " (PySCF dm1 vv diag was", np.round(np.diag(dm1_mo)[Vv],5),")")
print("  → einsum Pvv vs pyscf-stored (doo+doo.T means stored = Pvv+Pvv.T):",
      np.round(np.diag(Pvv_es+Pvv_es.T),5))
