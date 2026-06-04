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

# ===========================================================================
# EINSUM Lagrangian + relaxed density. Validate un-perturbed relaxed dipole
# against the FF-E_MP2 oracle (tests P×2 + z-vector together).
# ===========================================================================
# Slices of the MO ERI for readable einsum (chemist (pq|rs)=Imo[p,q,r,s]).
oo,ov,vo,vv = slice(0,nocc), slice(nocc,nmo), slice(nocc,nmo), slice(nocc,nmo)
def blk(p,q,rr,s): return Imo[p,q,rr,s]
ovov = Imo[oo,ov,oo,ov]   # (ia|jb)
oovv = Imo[oo,oo,ov,ov]   # (ij|ab)
ovvv = Imo[oo,ov,ov,ov]   # (ia|bc)
ooov = Imo[oo,oo,oo,ov]   # (ij|ka)

def lagrangian_es(t_iajb):
    # ferric build_lagrangian, in einsum. t indexed [i,a,j,b]; L is (vir c, occ k).
    # Term1 (i=k): t_kjab (2(ca|jb)-(cb|ja))  → sum over j,a,b
    #   (ca|jb)=Imo[v,v,o,v]=vvov ; (cb|ja) same tensor diff index
    vvov = Imo[vv,vv,oo,ov]   # (ca|jb): [c,a,j,b]
    L1 = np.einsum('kajb,cajb->ck', t_iajb, 2*vvov) - np.einsum('kajb,cbja->ck', t_iajb, vvov)
    # Term2 (j=k): t_ikab? our t[i,a,k,b]; (ia|cb)-(ib|ca) → ovvv blocks [i,a,c,b]
    ovvv2 = Imo[oo,ov,vv,vv]  # (ia|cb): [i,a,c,b]
    L2 = np.einsum('iakb,iacb->ck', t_iajb, 2*ovvv2) - np.einsum('iakb,ibca->ck', t_iajb, ovvv2)
    # Term3 (a=c): -t_icjb (2(ik|jb)-(ib|jk)) → oo ov blocks
    ooov3 = Imo[oo,oo,oo,ov]  # (ik|jb): [i,k,j,b]
    oovo3 = Imo[oo,ov,oo,oo]  # (ib|jk): [i,b,j,k]
    L3 = -(np.einsum('icjb,ikjb->ck', t_iajb, 2*ooov3) - np.einsum('icjb,ibjk->ck', t_iajb, oovo3))
    # Term4 (b=c): -t_iajc (2(ia|jk)-(ik|ja))
    ooov4 = Imo[oo,ov,oo,oo]  # (ia|jk): [i,a,j,k]
    ooov4b= Imo[oo,oo,oo,ov]  # (ik|ja): [i,k,j,a]
    L4 = -(np.einsum('iajc,iajk->ck', t_iajb, 2*ooov4) - np.einsum('iajc,ikja->ck', t_iajb, ooov4b))
    return L1+L2+L3+L4

# Validate einsum Lagrangian vs the loop version.
L_loop = lagrangian(Imo, t)
L_es = lagrangian_es(t)
print("\n=== einsum Lagrangian vs loop ===")
print("‖L_es - L_loop‖ =", np.max(np.abs(L_es - L_loop)))

# ===========================================================================
# Relaxed dipole: 2δ core + (Poo+Poo.T) + (Pvv+Pvv.T) + z (vo+ov). Validate the
# CORRELATION part of the dipole (relaxed - HF) against PySCF (relaxed - HF),
# removing the HF reference + sign-convention ambiguity.
# ===========================================================================
# HF dipole (from HF dm in MO = 2δ on occ).
Dhf = np.zeros((nmo,nmo))
for i in range(nocc): Dhf[i,i]=2.0
mu_hf = dipole_from_dm_ao(C@Dhf@C.T)

# PySCF: correlation dipole = relaxed_total - HF.  relaxed via FF of E_MP2corr only.
def pyscf_corr_dipole(axis, h=1e-4):
    def ecorr(F):
        m=scf.RHF(mol); m.get_hcore=lambda *a: hcore_ao - F*dip_ao[axis]; m.conv_tol=1e-12; m.kernel()
        return mp.MP2(m).run().e_corr
    return -(ecorr(h)-ecorr(-h))/(2*h)
mu_corr_ps = np.array([pyscf_corr_dipole(ax) for ax in range(3)])

# Mine: relaxed dm with ×2 P-blocks + z.  z solves M z = L (full-A operator).
t_iajb = t
Poo = -(2*np.einsum('ikab,jkab->ij', t_ijab, t_ijab) - np.einsum('ikab,jkba->ij', t_ijab, t_ijab))
Pvv =  (2*np.einsum('ijca,ijcb->ab', t_ijab, t_ijab) - np.einsum('ijca,ijbc->ab', t_ijab, t_ijab))
L = lagrangian_es(t_iajb)
z = np.linalg.solve(M, L.reshape(-1)).reshape(nvir,nocc)
Dcorr = np.zeros((nmo,nmo))
Dcorr[O,O]  += Poo + Poo.T
Dcorr[Vv,Vv]+= Pvv + Pvv.T
for a in range(nvir):
    for i in range(nocc):
        Dcorr[nocc+a,i]+=z[a,i]; Dcorr[i,nocc+a]+=z[a,i]
mu_corr_mine = dipole_from_dm_ao(C@Dcorr@C.T) - nucl   # pure electronic correlation part
print("\n=== correlation dipole (relaxed-HF) mine vs pyscf ===")
print("  mine  =", mu_corr_mine)
print("  pyscf =", mu_corr_ps)
print("  ratio z (mine/pyscf) on z-axis:", mu_corr_mine[2]/mu_corr_ps[2] if abs(mu_corr_ps[2])>1e-9 else 'na')

# Decompose: P-blocks-only dipole vs +z. And build PySCF's actual relaxed z via
# its grad _response_dm1 path to compare z directly.
Dp = np.zeros((nmo,nmo)); Dp[O,O]+=Poo+Poo.T; Dp[Vv,Vv]+=Pvv+Pvv.T
mu_P = dipole_from_dm_ao(C@Dp@C.T) - nucl
print("\n=== z-vector decomposition ===")
print("  P-blocks only dipole_z =", mu_P[2])
print("  +z (mine) dipole_z     =", mu_corr_mine[2], " (z adds", mu_corr_mine[2]-mu_P[2],")")
print("  pyscf total corr       =", mu_corr_ps[2])

# Build PySCF's relaxed dm1 via the grad machinery to get its z (ov block).
from pyscf.grad import mp2 as mp2grad
# replicate pyscf Xvo + _response_dm1 with OUR intermediates to compare z.
# PySCF Imat (Lagrangian-like) then Xvo then cphf. Easiest: call its _response_dm1.
# But that needs Xvo. Instead: get pyscf relaxed dm1 numerically = unrelaxed + FF-derived ov?
# Simplest valid check: pyscf relaxed dm1 via grad.make_rdm1 if available.
try:
    g = pt.Gradients() if hasattr(pt,'Gradients') else mp2grad.Gradients(pt)
    # trigger the rdm build
    d1 = mp2grad._response_dm1
    print("  (pyscf grad _response_dm1 available)")
except Exception as ex:
    print("  grad path:", ex)
# Direct: the z that reproduces pyscf corr dipole. PySCF corr = P-part + z-part.
# If P-part matches, the gap is z. Check P-part vs a pyscf P-only (unrelaxed corr dipole).
def pyscf_unrelaxed_corr_dipole():
    # unrelaxed = make_rdm1 (ov=0) correlation part
    return dipole_from_dm_ao(dm1_ao) - mu_hf
print("  pyscf UNRELAXED corr dipole_z =", (dipole_from_dm_ao(dm1_ao)-mu_hf)[2],
      " (= P-blocks only; mine P-only=", mu_P[2],")")

# ===========================================================================
# z RHS is missing PySCF's Xvo veff term: Xvo = (Imat.T - Imat)_vo + 2 Cv^T G[dm1_P] Co.
# My L ≈ the Imat part. Add the Fock-response of the MP2 P-density (doo+dvv).
# ===========================================================================
def G_mo(dm_mo):
    # G[D] = 2J - K in MO basis using Imo.
    J = np.einsum('pqrs,rs->pq', Imo, dm_mo)
    K = np.einsum('prqs,rs->pq', Imo, dm_mo)
    return 2*J - K

# MP2 P-density in MO (the doo+dvv part, the "separable" 1-PDM that sources veff).
dmP = np.zeros((nmo,nmo))
dmP[O,O]  += Poo + Poo.T
dmP[Vv,Vv]+= Pvv + Pvv.T
Gvo = G_mo(dmP)[Vv,O]   # (vir,occ) block of the Fock response
# RHS' = L + 2*Gvo  (PySCF Xvo adds 2 Cv^T vhf Co; here MO basis so 2*Gvo).
for scale in [1.0, 2.0]:
    rhs = L + scale*Gvo
    z2 = np.linalg.solve(M, rhs.reshape(-1)).reshape(nvir,nocc)
    Dc = np.zeros((nmo,nmo)); Dc[O,O]+=Poo+Poo.T; Dc[Vv,Vv]+=Pvv+Pvv.T
    for a in range(nvir):
        for i in range(nocc):
            Dc[nocc+a,i]+=z2[a,i]; Dc[i,nocc+a]+=z2[a,i]
    mz = (dipole_from_dm_ao(C@Dc@C.T)-nucl)[2]
    print(f"  RHS=L+{scale}·Gvo: corr dipole_z = {mz:.6f}  (target {mu_corr_ps[2]:.6f})")

# ===========================================================================
# Get PySCF's EXACT relaxed dm1 (with z) via its gradient rdm1 builder, compare
# z directly — stop guessing the RHS.
# ===========================================================================
from pyscf.grad import mp2 as mp2grad
import pyscf.lib as lib
# pyscf grad builds dm1mo internally; replicate just the response part using its
# own _response_dm1 with the Xvo we reconstruct from its Imat. Easier: monkey-call.
# Cleanest: use mp2grad.Gradients(pt).kernel() side-effect? Instead, call the
# documented make_rdm1 with relaxation if present:
relaxed_dm1_ao = None
try:
    gobj = mp2grad.Gradients(pt)
    # PySCF stores relaxed dm in grad via _grad_elec; reconstruct dm1mo:
    # Use the internal: build d1 intermediates then the response.
    d1 = mp.mp2._gamma1_intermediates(pt, pt.t2)
    doo_ps, dvv_ps = d1
    print("\n=== PySCF doo/dvv vs mine ===")
    print("  pyscf doo diag:", np.round(np.diag(doo_ps),6), " mine -Poo? ", np.round(np.diag(-Poo),6))
    print("  pyscf dvv diag:", np.round(np.diag(dvv_ps),6), " mine Pvv:", np.round(np.diag(Pvv),6))
except Exception as ex:
    print("grad introspection failed:", ex)

# ===========================================================================
# z is 0.57x. Test: is the z-vector OPERATOR wrong? The relaxed-density z-vector
# (Lagrangian/orbital response) may need a DIFFERENT Hessian than the dipole-CPHF.
# Solve M_s z = L for M with scaled coupling, find which reproduces the oracle.
# Target z-part of corr dipole = oracle - P-part = -0.026907 - 0.008079 = -0.034986
# ===========================================================================
target_z_dip = mu_corr_ps[2] - mu_P[2]
print("\n=== find correct z operator (target z-dipole_z = %.6f) ===" % target_z_dip)
Mdiag = np.zeros_like(M); Mcoup = M.copy()
for a in range(nvir):
    for i in range(nocc):
        idx=a*nocc+i
        Mdiag[idx,idx]=e[nocc+a]-e[i]
Mcoup = M - Mdiag
for s in [0.0, 0.5, 1.0, 2.0]:
    Ms = Mdiag + s*Mcoup
    zs = np.linalg.solve(Ms, L.reshape(-1)).reshape(nvir,nocc)
    Dc=np.zeros((nmo,nmo)); Dc[O,O]+=Poo+Poo.T; Dc[Vv,Vv]+=Pvv+Pvv.T
    for a in range(nvir):
        for i in range(nocc):
            Dc[nocc+a,i]+=zs[a,i]; Dc[i,nocc+a]+=zs[a,i]
    zdip=(dipole_from_dm_ao(C@Dc@C.T)-nucl)[2]-mu_P[2]
    print(f"  s={s}: z-dipole_z = {zdip:.6f}")
# Also: maybe L needs a factor. With full M (s=1), what L-scale hits target?
z1 = np.linalg.solve(M, L.reshape(-1)).reshape(nvir,nocc)
Dc=np.zeros((nmo,nmo))
for a in range(nvir):
    for i in range(nocc):
        Dc[nocc+a,i]+=z1[a,i]; Dc[i,nocc+a]+=z1[a,i]
zonly=(dipole_from_dm_ao(C@Dc@C.T)-nucl)[2]
print(f"  pure-z(s=1) dipole_z = {zonly:.6f}; L-scale to hit target = {target_z_dip/zonly:.4f}")

# ===========================================================================
# Get PySCF's TRUE relaxed dm1 (ov block = z) by running its gradient, which
# builds dm1mo internally. Patch _response_dm1 to capture it.
# ===========================================================================
import pyscf.grad.mp2 as gmod
captured = {}
_orig = gmod._response_dm1
def _cap(mp_, Xvo, *a, **k):
    out = _orig(mp_, Xvo, *a, **k)
    captured['Xvo'] = Xvo.copy()
    captured['resp'] = out.copy()
    return out
gmod._response_dm1 = _cap
try:
    gobj = gmod.Gradients(pt); gobj.kernel()
except Exception as ex:
    print("grad kernel:", type(ex).__name__, str(ex)[:80])
gmod._response_dm1 = _orig
if 'Xvo' in captured:
    Xvo_ps = captured['Xvo']                # (nvir,nocc)
    z_ps = captured['resp'][nocc:,:nocc]    # the response dvo
    print("\n=== PySCF TRUE z (response dvo) vs mine ===")
    print("  pyscf z:\n", np.round(z_ps,6))
    print("  mine  z (M\\L):\n", np.round(z1,6))
    print("  ratio mine/pyscf:\n", np.round(z1/np.where(np.abs(z_ps)>1e-9,z_ps,np.nan),4))
    print("  pyscf Xvo:\n", np.round(Xvo_ps,6))
    print("  mine L:\n", np.round(L,6))
    print("  ratio L/Xvo:\n", np.round(L/np.where(np.abs(Xvo_ps)>1e-9,Xvo_ps,np.nan),4))
else:
    print("did not capture Xvo")

# ===========================================================================
# Build Xvo PySCF-style and verify z matches PySCF's z.
# PySCF: Imat_full (nmo,nmo), then Xvo = Imat[o,v].T - Imat[v,o] + 2 Cv^T vhf(dm1P) Co.
# My L is the (vir,occ) Lagrangian ~ Imat[v,o]-ish. I need the FULL Imat (all blocks).
# Build full-MO Imat = orbital gradient: Imat_pq = sum contractions of t2 with (pq|..).
# Use the general MP2 Lagrangian L_pq (p any, q occ) — but simplest: reconstruct
# Imat from the relation PySCF uses, then form Xvo, solve, compare to z_ps.
# ===========================================================================
# General Imat (the energy-weighted/orbital-gradient intermediate). For canonical
# MP2 the vo-block orbital gradient is exactly my L (Lagrangian). PySCF's Imat is
# the FULL matrix incl ov, oo, vv. The Xvo antisymmetrization needs Imat[o,v] too.
# Imat[i,a] (occ,vir) = the "other" orbital gradient = -L-like with swapped roles.
# Build it via the same 4-term but for (i,a) instead of (c,k): by symmetry of the
# energy, Imat is NOT symmetric; the ov block comes from differentiating w.r.t.
# the occ-vir rotation in the other sense. Construct both from t2 + full ERIs.

# Simplest correct route: Xvo = -(L_vo) + 2*G[dmP]_vo  with the RIGHT sign, OR use
# pyscf's z_ps directly as the validated z and move on to ∂z. We HAVE z_ps now.
# Verify: does P-blocks + z_ps reproduce the oracle dipole?
Dc=np.zeros((nmo,nmo)); Dc[O,O]+=Poo+Poo.T; Dc[Vv,Vv]+=Pvv+Pvv.T
Dc[Vv,O]+=z_ps; Dc[O,Vv]+=z_ps.T
mu_with_psz = (dipole_from_dm_ao(C@Dc@C.T)-nucl)[2]
print("\n=== validate assembly with PySCF's true z ===")
print("  P+z_ps corr dipole_z =", mu_with_psz, " oracle =", mu_corr_ps[2],
      " match:", abs(mu_with_psz-mu_corr_ps[2])<1e-5)
