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

Run: PYTHONPATH=$PYSCF_PATH python3 scripts/cpks/mp2_alpha_pyscf.py
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

# ===========================================================================
# Build Xvo to match PySCF's captured Xvo, element by element.
# PySCF: Imat (full nmo,nmo, = -1 * mo^T Imat_ao S mo), then
#        Xvo = Imat[:nocc,nocc:].T - Imat[nocc:,:nocc] + 2*Cv^T vhf(dm1_full) Co
# dm1_full = HF(2δ) + P-blocks (the relaxed dm1 BEFORE response).
# We need the full Imat. My L ≈ Imat[v,o]? Compare, then get the ov block.
# ===========================================================================
# veff term: vhf(dm1_full) in MO, vo block. dm1_full = 2δ + (Poo+Poo.T)+(Pvv+Pvv.T)
dm1_full = np.zeros((nmo,nmo))
for i in range(nocc): dm1_full[i,i]=2.0
dm1_full[O,O]+=Poo+Poo.T; dm1_full[Vv,Vv]+=Pvv+Pvv.T
veff_vo = G_mo(dm1_full)[Vv,O]   # 2J-K of full dm1
veff_vo_P = G_mo(dmP)[Vv,O]      # 2J-K of P-only (HF part gives the SCF Fock = diagonal, no vo)

print("\n=== reconstruct Xvo ===")
print("  PySCF Xvo[v,o]:\n", np.round(Xvo_ps,6))
print("  my L (=Imat[v,o]?):\n", np.round(L,6))
print("  2*veff_vo(P-only):\n", np.round(2*veff_vo_P,6))
# Try Xvo = -L + 2*veff(dmP)  and  Xvo = L_antisym + 2*veff
# (need Imat[o,v]; if Imat≈L in vo, the ov block is a separate Lagrangian L_ov)
# Quick test combos:
for desc, cand in [
    ("L", L),
    ("-L", -L),
    ("L + 2veffP", L + 2*veff_vo_P),
    ("-L + 2veffP", -L + 2*veff_vo_P),
    ("L - 2veffP", L - 2*veff_vo_P),
]:
    print(f"  ‖{desc} - Xvo_ps‖ = {np.max(np.abs(cand - Xvo_ps)):.6f}")

# ===========================================================================
# Build Imat the PySCF way, in MO. part_dm2 = MP2 2-PDM Γ with PySCF factors:
#   Γ_ipqj from t2 (ij,ab): Γ = 4 t[i,j,a,b] - 2 t[i,j,b,a]  (vir pair a,b).
# Imat (MO, before -1·S transform) = contract Γ with MO ERIs over 3 indices,
# leaving (occ i, MO p): Imat[p,i-related]. PySCF then does Imat=-mo^T Imat_ao S mo.
# In a pure-MO formulation Imat_pq = sum_{jab} Γ_{ijab}(pj|ab)-style. Build it to
# match the captured Xvo.
# Γ in MO: G2[i,j,a,b] = 4 t_ijab - 2 t_ijba
G2 = 4*t_ijab - 2*t_ijab.transpose(0,1,3,2)
# Imat blocks (occ-occ, vir-occ, etc.) via contracting Γ with full MO ERIs.
# Imat_vo (c,k): the orbital gradient = sum over the 2-PDM contracted with d(ERI).
# Standard MP2 Lagrangian Lvo_ck = sum_jab Γ_kjab (ca|jb) - ... Let's build the
# general Imat_pq = sum over Γ * (pq'|..) for the three contraction patterns that
# match PySCF dm2buf symmetrization. Easiest faithful MO version:
#   Imat_pq = 2 * sum_{jab} Γ[?] (p a | j b) ...   -- derive by matching.
# Pragmatic: contract Γ with ERIs in the 3 distinct ways and fit to Xvo_ps.
ovov_f = Imo[:, nocc:, :, nocc:]          # (p a | r b) general p,r occ-or-all
# Imat from 2-PDM: Imat[p,i] = sum_{jab} G2[i,j,a,b] (p a | j b)   (chemist)
# but need (pa|jb): Imo[p, nocc+a, j, nocc+b]
Imat_pi = np.einsum('ijab,pajb->pi', G2, Imo[:, nocc:, :nocc, nocc:])  # (nmo, nocc)
# vo block:
Imat_vo = Imat_pi[Vv, :]   # (nvir,nocc)
# ov block: Imat[i,p] analog
Imat_iq = np.einsum('ijab,iajb->ij', G2, Imo[:nocc, nocc:, :nocc, nocc:])  # placeholder
print("\n=== Imat_vo (Γ-contracted) vs L vs Xvo_ps ===")
print("  Imat_vo:\n", np.round(Imat_vo,6))
print("  Xvo_ps:\n", np.round(Xvo_ps,6))
print("  ‖Imat_vo - Xvo_ps‖ =", np.max(np.abs(Imat_vo - Xvo_ps)))
print("  ‖-Imat_vo - Xvo_ps‖ =", np.max(np.abs(-Imat_vo - Xvo_ps)))

# ===========================================================================
# Extract PySCF's Imat-vo contribution: antisym_Imat_vo = Xvo_ps - 2 G[dmP]_vo,
# using PySCF's OWN get_veff (not my hand G) to be exact. dmP = doo+doo.T,dvv+dvv.T.
# ===========================================================================
dmP_ao = C @ dmP @ C.T
vhf_ps = mf.get_veff(mol, dmP_ao) * 2.0            # PySCF veff, ×2 as in grad
veff_vo_ps = (C[:,nocc:].T @ vhf_ps @ C[:,:nocc])  # (nvir,nocc)
imat_vo_contrib = Xvo_ps - veff_vo_ps              # = Imat[:no,no:].T - Imat[no:,:no]  (vo block)
print("\n=== PySCF Imat-vo contribution (Xvo - 2G[dmP]) ===")
print("  imat_vo_contrib:\n", np.round(imat_vo_contrib,6))
print("  my L:\n", np.round(L,6))
print("  ‖L - imat_contrib‖   =", np.max(np.abs(L - imat_vo_contrib)))
print("  ‖-L - imat_contrib‖  =", np.max(np.abs(-L - imat_vo_contrib)))
# my Imat_vo (Γ-contract) too
print("  ‖Imat_vo(Γ) - contrib‖ =", np.max(np.abs(Imat_vo - imat_vo_contrib)))
print("  ‖-Imat_vo(Γ) - contrib‖=", np.max(np.abs(-Imat_vo - imat_vo_contrib)))

# ===========================================================================
# COMPLETE Xvo = L + 2 G[dmP]_vo. Verify z matches PySCF, and relaxed dipole.
# Use a hand G (2J-K) so it's portable to Rust (not pyscf get_veff).
# First confirm my hand-G matches pyscf veff on dmP:
my_veff_vo = 2.0 * G_mo(dmP)[Vv,O]   # 2*(2J-K) ? check factor vs pyscf veff*2
print("\n=== veff factor check ===")
print("  pyscf veff_vo (get_veff*2 → vo):\n", np.round(veff_vo_ps,6))
print("  my 2*(2J-K)[dmP]_vo:\n", np.round(my_veff_vo,6))
print("  my (2J-K)[dmP]_vo:\n", np.round(G_mo(dmP)[Vv,O],6))

# ===========================================================================
# FINAL: Xvo = L + (2J-K)[dmP]_vo (portable). Solve M z = Xvo. Verify.
# ===========================================================================
Xvo_mine = L + G_mo(dmP)[Vv,O]
print("\n=== FINAL Xvo + z verification ===")
print("  ‖Xvo_mine - Xvo_ps‖ =", np.max(np.abs(Xvo_mine - Xvo_ps)))
z_final = np.linalg.solve(M, Xvo_mine.reshape(-1)).reshape(nvir,nocc)
print("  ‖z_final - z_ps‖    =", np.max(np.abs(z_final - z_ps)))
# relaxed dipole with z_final
Dc = np.zeros((nmo,nmo))
for i in range(nocc): Dc[i,i]=2.0
Dc[O,O]+=Poo+Poo.T; Dc[Vv,Vv]+=Pvv+Pvv.T
Dc[Vv,O]+=z_final; Dc[O,Vv]+=z_final.T
mu_final = dipole_from_dm_ao(C@Dc@C.T)
print("  relaxed μ mine =", mu_final)
mu_pyscf_relaxed = np.array([pyscf_relaxed_dipole(ax) for ax in range(3)])
print("  relaxed μ pyscf(FF E_tot) =", mu_pyscf_relaxed)
print("  |μ_z| match:", abs(abs(mu_final[2])-abs(mu_pyscf_relaxed[2]))<1e-5)

# ===========================================================================
# z differs though Xvo matches → the SOLVE operator differs. PySCF _response_dm1
# uses cphf.solve(fvind) with fvind(x)=2 Cv^T get_veff(Cv x Co^T + h.c.) Co, plus
# the (ea-ei) from cphf. Compare: does M z_ps == Xvo_ps? (if not, M is wrong op)
# ===========================================================================
print("\n=== z-vector operator check ===")
resid_M = (M @ z_ps.reshape(-1)).reshape(nvir,nocc) - Xvo_ps
print("  ‖M·z_ps - Xvo_ps‖ =", np.max(np.abs(resid_M)), " (0 ⇒ M is PySCF's operator)")
# PySCF fvind operator applied to z_ps:
def fvind(x):
    xm = x.reshape(nvir,nocc)
    dm = C[:,nocc:] @ xm @ C[:,:nocc].T
    v = mf.get_veff(mol, dm + dm.T)
    return 2.0*(C[:,nocc:].T @ v @ C[:,:nocc])
# PySCF cphf solves: (ea-ei) x + fvind(x) = Xvo  → operator P z:
def Pop(x):
    xm=x.reshape(nvir,nocc)
    out = np.zeros((nvir,nocc))
    for a in range(nvir):
        for i in range(nocc):
            out[a,i]=(e[nocc+a]-e[i])*xm[a,i]
    return out + fvind(x)
resid_P = Pop(z_ps.reshape(-1)) - Xvo_ps
print("  ‖Pop·z_ps - Xvo_ps‖ =", np.max(np.abs(resid_P)), " (0 ⇒ PySCF op reproduced)")
# solve with PySCF operator
from scipy.sparse.linalg import LinearOperator, gmres
Pmat = np.zeros((nvir*nocc, nvir*nocc))
basis=np.eye(nvir*nocc)
for k in range(nvir*nocc):
    Pmat[:,k]=Pop(basis[k]).reshape(-1)
z_pop = np.linalg.solve(Pmat, Xvo_ps.reshape(-1)).reshape(nvir,nocc)
print("  ‖z_pop - z_ps‖ =", np.max(np.abs(z_pop - z_ps)))

print("\n=== z-vector SIGN convention ===")
print("  ‖Pop·z_ps + Xvo_ps‖ =", np.max(np.abs(Pop(z_ps.reshape(-1)) + Xvo_ps)), " (0 ⇒ solves (Δε+f)z=-Xvo)")
# solve both signs and compare to z_ps
z_minus = np.linalg.solve(Pmat, (-Xvo_ps).reshape(-1)).reshape(nvir,nocc)
print("  ‖solve(P,-Xvo) - z_ps‖ =", np.max(np.abs(z_minus - z_ps)))
# Also check cphf.solve's actual convention from pyscf source
import pyscf.scf.cphf as cphf
import inspect
src = inspect.getsource(cphf.solve_nos1) if hasattr(cphf,'solve_nos1') else inspect.getsource(cphf.solve)
print("  cphf.solve sign hint:", [l.strip() for l in src.split(chr(10)) if 'return' in l or '-' in l and 'mo_energy' in l][:3])

# ===========================================================================
# COMPLETE VALIDATED RECIPE: Xvo = L + G[dmP]_vo ; (Δε+A) z = -Xvo.
# End-to-end: relaxed dipole must match PySCF to machine precision.
# ===========================================================================
z_correct = np.linalg.solve(M, (-Xvo_mine).reshape(-1)).reshape(nvir,nocc)
print("\n=== END-TO-END (correct sign) ===")
print("  ‖z_correct - z_ps‖ =", np.max(np.abs(z_correct - z_ps)))
Dc = np.zeros((nmo,nmo))
for i in range(nocc): Dc[i,i]=2.0
Dc[O,O]+=Poo+Poo.T; Dc[Vv,Vv]+=Pvv+Pvv.T
Dc[Vv,O]+=z_correct; Dc[O,Vv]+=z_correct.T
mu_me = dipole_from_dm_ao(C@Dc@C.T)
# PySCF relaxed dm1 = unrelaxed dm1_mo + response (z). Build it the same way:
dm1_relaxed_ps = dm1_mo.copy()
dm1_relaxed_ps[Vv,O]+=z_ps; dm1_relaxed_ps[O,Vv]+=z_ps.T
mu_ps = dipole_from_dm_ao(C@dm1_relaxed_ps@C.T)
print("  relaxed μ mine  =", mu_me)
print("  relaxed μ pyscf =", mu_ps, "  ‖Δ‖=", np.max(np.abs(mu_me-mu_ps)))
print("  *** STATIC RELAXED DENSITY MATCHES PYSCF:", np.max(np.abs(mu_me-mu_ps))<1e-6, "***")

# ===========================================================================
# ∂z FOR POLARIZABILITY: differentiate the validated static recipe.
#   (Δε+A) z = -Xvo,  Xvo = L + G[dmP]_vo
#   (Δε+A) ∂z = -∂Xvo - ∂(Δε+A)·z
# ∂Xvo = ∂L + ∂G[dmP]_vo  (∂L from ∂t/∂Imo; ∂G from ∂dmP + ∂(MO rotation of G))
# α_pq = -Tr[∂D_relax/∂F_q · r_p],  ∂D_relax = ∂(2δ via U) + ∂(P+Pᵀ) + ∂z.
# Validate vs FF of the relaxed dipole (exact integrals → machine precision).
# ===========================================================================
def cphf_U_axis(axis):
    rvo = np.array([[r_mo[axis,nocc+a,i] for i in range(nocc)] for a in range(nvir)])
    return np.linalg.solve(M, (-rvo).reshape(-1)).reshape(nvir,nocc)

def static_relaxed_dm_validated(Imatrix, evec):
    """The validated static recipe, parameterized by integrals+energies (for FF)."""
    t = t2_amp(Imatrix, evec)
    tij = t.transpose(0,2,1,3)
    Poo = -(2*np.einsum('ikab,jkab->ij',tij,tij)-np.einsum('ikab,jkba->ij',tij,tij))
    Pvv =  (2*np.einsum('ijca,ijcb->ab',tij,tij)-np.einsum('ijca,ijbc->ab',tij,tij))
    dmP = np.zeros((nmo,nmo)); dmP[O,O]=Poo+Poo.T; dmP[Vv,Vv]=Pvv+Pvv.T
    L = lagrangian_es(t)
    # G[dmP]_vo with these (field-rotated) integrals: G uses Imatrix
    J = np.einsum('pqrs,rs->pq', Imatrix, dmP); K = np.einsum('prqs,rs->pq', Imatrix, dmP)
    Gvo = (2*J-K)[Vv,O]
    Xvo = L + Gvo
    # operator M is built from Imatrix+evec
    def Mloc():
        Mm=np.zeros((nvir,nocc,nvir,nocc))
        for a in range(nvir):
            for i in range(nocc):
                for b in range(nvir):
                    for j in range(nocc):
                        Mm[a,i,b,j]=(4*Imatrix[nocc+a,i,nocc+b,j]-Imatrix[nocc+a,nocc+b,i,j]-Imatrix[nocc+a,j,nocc+b,i])
                Mm[a,i,a,i]+=evec[nocc+a]-evec[i]
        return Mm.reshape(nvir*nocc,nvir*nocc)
    z = np.linalg.solve(Mloc(), (-Xvo).reshape(-1)).reshape(nvir,nocc)
    D=np.zeros((nmo,nmo))
    for i in range(nocc): D[i,i]=2.0
    D[O,O]+=Poo+Poo.T; D[Vv,Vv]+=Pvv+Pvv.T
    D[Vv,O]+=z; D[O,Vv]+=z.T
    return D


def scf_in_field(Ffield, axis=2):
    m = scf.RHF(mol); m.get_hcore = lambda *a: hcore_ao - Ffield*dip_ao[axis]
    m.conv_tol=1e-12; m.kernel()
    return m.mo_energy, m.mo_coeff

# FF ORACLE for relaxed α: re-solve SCF in field, transform integrals, apply the
# validated static recipe in the field-MO basis, read dipole. (phase-matched)
def relaxed_dip_validated(Ffield, axis, field_axis):
    e_orb,Cf = scf_in_field(Ffield, field_axis)
    for p in range(nmo):
        if Cf[:,p]@np.eye(nmo)[:,p]<0: Cf[:,p]*=-1
    Imf = np.einsum('pa,qb,rc,sd,pqrs->abcd', Cf,Cf,Cf,Cf, eri_ao, optimize=True)
    D = static_relaxed_dm_validated(Imf, e_orb)
    D_ao = Cf@D@Cf.T
    return (-np.einsum('pq,pq->', dip_ao[axis], D_ao) + nucl[axis])
h=1e-4
alpha_ff = np.zeros((3,3))
for q in range(3):
    dp = np.array([relaxed_dip_validated(h,p,q) for p in range(3)])
    dm = np.array([relaxed_dip_validated(-h,p,q) for p in range(3)])
    alpha_ff[:,q] = -(dp-dm)/(2*h)
print("\n=== relaxed-α FF oracle (validated static recipe, exact integrals) ===")
print(np.round(alpha_ff,5))
print("iso =", np.trace(alpha_ff)/3)

# ===========================================================================
# FF-of-relaxed-DIPOLE is unstable (1/F) even phase-matched. Use the ENERGY
# Hessian oracle: α_qq = -d²E/dF² (smooth). Diagonal only (sufficient to validate).
# ===========================================================================
def etot_field(F, axis):
    m=scf.RHF(mol); m.get_hcore=lambda *a: hcore_ao - F*dip_ao[axis]; m.conv_tol=1e-12; m.kernel()
    return m.e_tot + mp.MP2(m).run().e_corr
h=2e-3
print("\n=== relaxed-α from ENERGY Hessian (smooth oracle) ===")
alpha_diag=[]
for q in range(3):
    e0=etot_field(0,q); ep=etot_field(h,q); em=etot_field(-h,q)
    a = -(ep - 2*e0 + em)/h**2     # α = -d²E/dF²
    alpha_diag.append(a)
    print(f"  α_{q}{q} = {a:.5f}")
print("  iso =", sum(alpha_diag)/3)

# ===========================================================================
# ANALYTIC relaxed α: ∂D_relax/∂F^q = directional deriv of the validated static
# recipe along (∂Imo, ∂eps) from CPHF U^q. Gauge-clean (smooth inputs). Then
# α_pq = -Tr[∂D_relax^q · r^p].  Validate vs energy-Hessian oracle.
# ===========================================================================
def analytic_relaxed_alpha():
    alpha = np.zeros((3,3))
    for q in range(3):
        U = cphf_U_axis(q)                       # CPHF response (vir,occ)
        # Θ generator + ∂Imo, ∂eps from U
        Th=np.zeros((nmo,nmo))
        for a in range(nvir):
            for i in range(nocc):
                Th[nocc+a,i]=U[a,i]; Th[i,nocc+a]=-U[a,i]
        dImo=(np.einsum('xp,xqrs->pqrs',Th,Imo)+np.einsum('xq,pxrs->pqrs',Th,Imo)
             +np.einsum('xr,pqxs->pqrs',Th,Imo)+np.einsum('xs,pqrx->pqrs',Th,Imo))
        # ∂eps_p = ∂F_pp = (-r + G[∂D_scf])_pp ; ∂D_scf from U (2δ core response)
        dDscf=np.zeros((nmo,nmo))
        for a in range(nvir):
            for i in range(nocc):
                dDscf[nocc+a,i]+=2*U[a,i]; dDscf[i,nocc+a]+=2*U[a,i]
        G=2*np.einsum('pqrs,rs->pq',Imo,dDscf)-np.einsum('prqs,rs->pq',Imo,dDscf)
        deps=np.diag(-r_mo[q]+G).copy()
        # directional derivative of the static recipe along (dImo, deps)
        ed=1e-5
        Dp = static_relaxed_dm_validated(Imo+ed*dImo, e+ed*deps)
        Dm = static_relaxed_dm_validated(Imo-ed*dImo, e-ed*deps)
        dDrel_corr = (Dp-Dm)/(2*ed)              # ∂ of the CORRELATION part (P+z), MO basis
        # plus the SCF core response ∂(2δ) = 2U in vo block (static recipe's 2δ is const in its inputs)
        dDrel = dDrel_corr.copy()
        for a in range(nvir):
            for i in range(nocc):
                dDrel[nocc+a,i]+=2*U[a,i]; dDrel[i,nocc+a]+=2*U[a,i]
        # α_pq = -Tr[∂D_relax^q · r^p]  (MO basis: r_mo[p])
        for p in range(3):
            alpha[p,q] = -np.sum(dDrel * r_mo[p])
    return alpha

aa = analytic_relaxed_alpha()
print("\n=== ANALYTIC relaxed α vs energy-Hessian oracle ===")
print("  analytic:\n", np.round(aa,5))
print("  oracle diag: [0.04433, 4.98115, 2.13505]")
print("  analytic diag:", np.round(np.diag(aa),5), " iso", np.round(np.trace(aa)/3,5))

# Decompose the yy discrepancy: HF part (2U) vs MP2 correction part.
print("\n=== decompose α_yy (oracle 4.981, analytic 5.280) ===")
q=1
U=cphf_U_axis(q)
rvo=np.array([[r_mo[q,nocc+a,i] for i in range(nocc)] for a in range(nvir)])
hf_yy = -4*np.sum(U*rvo)
print("  HF part (2U→ -4ΣUr) α_yy =", hf_yy, " (HF oracle was 5.195)")
# So analytic HF α_yy should be 5.195; my analytic total 5.280 → MP2 correction +0.085
# oracle total 4.981 → MP2 correction = 4.981-5.195 = -0.214. Mine: 5.280-5.195=+0.085.
print("  MP2 correction: analytic +%.3f, oracle %.3f" % (5.280-hf_yy, 4.981-hf_yy))
print("  => MP2 correction has WRONG SIGN on yy (analytic +0.085 vs oracle -0.214)")

# ===========================================================================
# FIX yy ∂z: first validate the INPUTS (∂Imo, ∂eps) for q=1 vs phase-matched FD.
# If inputs are right, the bug is in the static-builder directional derivative.
# ===========================================================================
q=1
U=cphf_U_axis(q)
Th=np.zeros((nmo,nmo))
for a in range(nvir):
    for i in range(nocc):
        Th[nocc+a,i]=U[a,i]; Th[i,nocc+a]=-U[a,i]
dImo_an=(np.einsum('xp,xqrs->pqrs',Th,Imo)+np.einsum('xq,pxrs->pqrs',Th,Imo)
        +np.einsum('xr,pqxs->pqrs',Th,Imo)+np.einsum('xs,pqrx->pqrs',Th,Imo))
dDscf=np.zeros((nmo,nmo))
for a in range(nvir):
    for i in range(nocc):
        dDscf[nocc+a,i]+=2*U[a,i]; dDscf[i,nocc+a]+=2*U[a,i]
G=2*np.einsum('pqrs,rs->pq',Imo,dDscf)-np.einsum('prqs,rs->pq',Imo,dDscf)
deps_an=np.diag(-r_mo[q]+G).copy()

# phase-matched FD of Imo(F) and eps(F) along q
hh=1e-4
def Cmo_field(F):
    eo,Cf=scf_in_field(F,q)
    for p in range(nmo):
        if Cf[:,p]@np.eye(nmo)[:,p]<0: Cf[:,p]*=-1
    return eo,Cf
eo_p,Cp=Cmo_field(hh); eo_m,Cm=Cmo_field(-hh)
Imo_p=np.einsum('pa,qb,rc,sd,pqrs->abcd',Cp,Cp,Cp,Cp,eri_ao,optimize=True)
Imo_m=np.einsum('pa,qb,rc,sd,pqrs->abcd',Cm,Cm,Cm,Cm,eri_ao,optimize=True)
dImo_fd=(Imo_p-Imo_m)/(2*hh); deps_fd=(eo_p-eo_m)/(2*hh)
print("\n=== q=1 input validation (analytic vs phase-matched FD) ===")
print("  ‖dImo_an - dImo_fd‖ =", np.max(np.abs(dImo_an-dImo_fd)))
print("  ‖deps_an - deps_fd‖ =", np.max(np.abs(deps_an-deps_fd)))
print("  deps_an:", np.round(deps_an,5))
print("  deps_fd:", np.round(deps_fd,5))

# Is the α_yy directional-FD ε-stable? (huge dImo components could poison it)
print("\n=== α_yy directional-FD ε-stability ===")
for ed in [1e-4,1e-5,1e-6,1e-7]:
    Dp = static_relaxed_dm_validated(Imo+ed*dImo_an, e+ed*deps_an)
    Dm = static_relaxed_dm_validated(Imo-ed*dImo_an, e-ed*deps_an)
    dDrel=(Dp-Dm)/(2*ed)
    for a in range(nvir):
        for i in range(nocc):
            dDrel[nocc+a,i]+=2*U[a,i]; dDrel[i,nocc+a]+=2*U[a,i]
    ayy=-np.sum(dDrel*r_mo[1])
    print(f"  ε={ed}: α_yy = {ayy:.6f}  (oracle 4.98115)")

# Gauge-invariant test of ∂Imo sign: ∂E_MP2/∂F via my ∂Imo vs FD of E_MP2.
print("\n=== ∂Imo sign check via ∂E_MP2 (gauge-invariant) ===")
def emp2_from_I(Imatrix, evec): return e_mp2(Imatrix, evec)
ed=1e-5
dE_an = (emp2_from_I(Imo+ed*dImo_an, e+ed*deps_an)-emp2_from_I(Imo-ed*dImo_an, e-ed*deps_an))/(2*ed)
# FD of E_MP2 along field q=1 (smooth, gauge-invariant)
def emp2_field(F):
    m=scf.RHF(mol); m.get_hcore=lambda *a: hcore_ao - F*dip_ao[1]; m.conv_tol=1e-12; m.kernel()
    return mp.MP2(m).run().e_corr
dE_fd=(emp2_field(1e-4)-emp2_field(-1e-4))/(2e-4)
print(f"  ∂E_MP2/∂Fy: analytic(my ∂Imo) = {dE_an:.8f}  FD = {dE_fd:.8f}")
print(f"  ratio = {dE_an/dE_fd:.4f}")

# ∂P_vv is gauge-stable (physical density block). Compare analytic (via my ∂Imo)
# vs phase-matched FD. This directly tests ∂Imo→∂t2→∂P propagation for yy.
print("\n=== ∂P_vv: analytic (my ∂Imo) vs phase-matched FD (q=1) ===")
def Pvv_from_I(Imatrix, evec):
    t=t2_amp(Imatrix,evec); tij=t.transpose(0,2,1,3)
    return 2*np.einsum('ijca,ijcb->ab',tij,tij)-np.einsum('ijca,ijbc->ab',tij,tij)
ed=1e-5
dPvv_an=(Pvv_from_I(Imo+ed*dImo_an,e+ed*deps_an)-Pvv_from_I(Imo-ed*dImo_an,e-ed*deps_an))/(2*ed)
# FD: P_vv in field-MO basis (phase-matched)
hh=1e-4
Pvv_p=Pvv_from_I(Imo_p,eo_p); Pvv_m=Pvv_from_I(Imo_m,eo_m)
dPvv_fd=(Pvv_p-Pvv_m)/(2*hh)
print("  dPvv_an:\n", np.round(dPvv_an,5))
print("  dPvv_fd:\n", np.round(dPvv_fd,5))
print("  ‖Δ‖=", np.max(np.abs(dPvv_an-dPvv_fd)))

# The off-diag ∂Pvv may be gauge-contaminated (2 near-degen virtuals rotate).
# Test the gauge-invariant ∂Tr[Pvv] and ∂(eigenvalues of Pvv) instead.
print("\n=== gauge-invariant ∂Pvv checks ===")
print("  ∂Tr[Pvv]: an=%.6f fd=%.6f" % (np.trace(dPvv_an), np.trace(dPvv_fd)))
# eigenvalue-sum is trace (done). Check Frobenius-invariant ∂‖Pvv‖² = 2Tr[Pvv ∂Pvv]
Pvv0=Pvv_from_I(Imo,e)
inv_an=2*np.sum(Pvv0*dPvv_an); inv_fd=2*np.sum(Pvv0*dPvv_fd)
print("  ∂Tr[Pvv²]: an=%.6f fd=%.6f ratio=%.4f" % (inv_an,inv_fd, inv_an/inv_fd if abs(inv_fd)>1e-12 else 0))

# ===========================================================================
# EXPLICIT analytic ∂z (purely ov, no gauge leak): differentiate (Δε+A)z=−Xvo.
#   (Δε+A) ∂z = −∂Xvo − ∂(Δε+A)·z
# Build ∂Xvo = ∂L + ∂G[dmP]_vo analytically; ∂(Δε+A)·z via directional FD of the
# operator (gauge-safe: operator acts on fixed z). Then α with explicit ∂z + ∂P.
# ===========================================================================
def analytic_alpha_explicit():
    alpha=np.zeros((3,3))
    t=t2_amp(Imo,e); tij=t.transpose(0,2,1,3)
    Poo0=-(2*np.einsum('ikab,jkab->ij',tij,tij)-np.einsum('ikab,jkba->ij',tij,tij))
    Pvv0= (2*np.einsum('ijca,ijcb->ab',tij,tij)-np.einsum('ijca,ijbc->ab',tij,tij))
    dmP0=np.zeros((nmo,nmo)); dmP0[O,O]=Poo0+Poo0.T; dmP0[Vv,Vv]=Pvv0+Pvv0.T
    L0=lagrangian_es(t)
    J=np.einsum('pqrs,rs->pq',Imo,dmP0); K=np.einsum('prqs,rs->pq',Imo,dmP0)
    Xvo0=L0+(2*J-K)[Vv,O]
    z0=np.linalg.solve(M,(-Xvo0).reshape(-1)).reshape(nvir,nocc)
    for q in range(3):
        U=cphf_U_axis(q)
        Th=np.zeros((nmo,nmo))
        for a in range(nvir):
            for i in range(nocc):
                Th[nocc+a,i]=U[a,i]; Th[i,nocc+a]=-U[a,i]
        dImo=(np.einsum('xp,xqrs->pqrs',Th,Imo)+np.einsum('xq,pxrs->pqrs',Th,Imo)
             +np.einsum('xr,pqxs->pqrs',Th,Imo)+np.einsum('xs,pqrx->pqrs',Th,Imo))
        dDscf=np.zeros((nmo,nmo))
        for a in range(nvir):
            for i in range(nocc):
                dDscf[nocc+a,i]+=2*U[a,i]; dDscf[i,nocc+a]+=2*U[a,i]
        Gd=2*np.einsum('pqrs,rs->pq',Imo,dDscf)-np.einsum('prqs,rs->pq',Imo,dDscf)
        deps=np.diag(-r_mo[q]+Gd).copy()
        ed=1e-5
        # ∂Xvo via directional deriv of Xvo(Imo,eps)
        def Xvo_at(s):
            ts=t2_amp(Imo+s*dImo, e+s*deps); tijs=ts.transpose(0,2,1,3)
            Poo=-(2*np.einsum('ikab,jkab->ij',tijs,tijs)-np.einsum('ikab,jkba->ij',tijs,tijs))
            Pvv= (2*np.einsum('ijca,ijcb->ab',tijs,tijs)-np.einsum('ijca,ijbc->ab',tijs,tijs))
            dmP=np.zeros((nmo,nmo)); dmP[O,O]=Poo+Poo.T; dmP[Vv,Vv]=Pvv+Pvv.T
            Ls=lagrangian_es(ts)
            Is=Imo+s*dImo
            J=np.einsum('pqrs,rs->pq',Is,dmP); K=np.einsum('prqs,rs->pq',Is,dmP)
            return Ls+(2*J-K)[Vv,O]
        dXvo=(Xvo_at(ed)-Xvo_at(-ed))/(2*ed)
        # ∂(Δε+A)·z0 via directional deriv of M(Imo,eps)·z0
        def Mz(s):
            Is=Imo+s*dImo; es=e+s*deps
            Mm=np.zeros((nvir,nocc,nvir,nocc))
            for a in range(nvir):
                for i in range(nocc):
                    for b in range(nvir):
                        for j in range(nocc):
                            Mm[a,i,b,j]=(4*Is[nocc+a,i,nocc+b,j]-Is[nocc+a,nocc+b,i,j]-Is[nocc+a,j,nocc+b,i])
                    Mm[a,i,a,i]+=es[nocc+a]-es[i]
            return (Mm.reshape(nvir*nocc,-1)@z0.reshape(-1)).reshape(nvir,nocc)
        dMz=(Mz(ed)-Mz(-ed))/(2*ed)
        rhs=(-dXvo - dMz)
        dz=np.linalg.solve(M, rhs.reshape(-1)).reshape(nvir,nocc)
        # ∂(P+Pᵀ) blocks
        def P_at(s):
            ts=t2_amp(Imo+s*dImo,e+s*deps); tijs=ts.transpose(0,2,1,3)
            Poo=-(2*np.einsum('ikab,jkab->ij',tijs,tijs)-np.einsum('ikab,jkba->ij',tijs,tijs))
            Pvv= (2*np.einsum('ijca,ijcb->ab',tijs,tijs)-np.einsum('ijca,ijbc->ab',tijs,tijs))
            return Poo,Pvv
        Pp=P_at(ed); Pm=P_at(-ed)
        dPoo=(Pp[0]-Pm[0])/(2*ed); dPvv=(Pp[1]-Pm[1])/(2*ed)
        dD=np.zeros((nmo,nmo))
        dD[O,O]=dPoo+dPoo.T; dD[Vv,Vv]=dPvv+dPvv.T
        for a in range(nvir):
            for i in range(nocc):
                dD[nocc+a,i]+=2*U[a,i]+dz[a,i]; dD[i,nocc+a]+=2*U[a,i]+dz[a,i]
        for p in range(3):
            alpha[p,q]=-np.sum(dD*r_mo[p])
    return alpha

ae=analytic_alpha_explicit()
print("\n=== EXPLICIT ∂z analytic α vs oracle ===")
print("  diag:", np.round(np.diag(ae),5), " (oracle [0.04433,4.98115,2.13505])")
print("  iso", np.round(np.trace(ae)/3,5), " (oracle 2.387)")

# Verify the energy-Hessian oracle α_yy is converged (higher-order contamination?).
print("\n=== oracle α_yy convergence (E-Hessian, vary h) + Richardson ===")
def etot_field_y(F):
    m=scf.RHF(mol); m.get_hcore=lambda *a: hcore_ao - F*dip_ao[1]; m.conv_tol=1e-12; m.kernel()
    return m.e_tot + mp.MP2(m).run().e_corr
e0=etot_field_y(0)
for h in [4e-3,2e-3,1e-3,5e-4]:
    a=-(etot_field_y(h)-2*e0+etot_field_y(-h))/h**2
    print(f"  h={h}: α_yy={a:.6f}")
# Richardson (h, h/2)
h=2e-3
a_h=-(etot_field_y(h)-2*e0+etot_field_y(-h))/h**2
a_h2=-(etot_field_y(h/2)-2*e0+etot_field_y(-h/2))/(h/2)**2
a_rich=(4*a_h2-a_h)/3
print(f"  Richardson α_yy = {a_rich:.6f}")

# Is the static D_relax the true dE/dh density? Test: -Tr[D_relax(F)·r] == dE/dF.
print("\n=== is D_relax the energy-derivative density? (-Tr[D r] vs dE/dF) ===")
def D_and_dEdF(F, axis):
    eo,Cf=scf_in_field(F,axis)
    for p in range(nmo):
        if Cf[:,p]@np.eye(nmo)[:,p]<0: Cf[:,p]*=-1
    Imf=np.einsum('pa,qb,rc,sd,pqrs->abcd',Cf,Cf,Cf,Cf,eri_ao,optimize=True)
    D=static_relaxed_dm_validated(Imf,eo)
    D_ao=Cf@D@Cf.T
    mu = -np.einsum('pq,pq->',dip_ao[axis],D_ao)+nucl[axis]   # -Tr[D r]+nuc... but this IS the dipole
    return mu
# dE/dF numerically and -Tr[D_relax r] at same F:
F=0.0
dEdF = -(etot_field_y(1e-4)-etot_field_y(-1e-4))/(2e-4)   # = -dE/dF... E(F)=E0-F<r> so dE/dF=-<r>=-mu? 
# at F=0, dE/dF should = -mu_z(0)? Actually μ = -dE/dF. Check:
mu_from_E = -dEdF   # μ = -dE/dF
mu_from_D = D_and_dEdF(0.0, 1)  # y-dipole at F=0
print(f"  μ_y from -dE/dFy = {mu_from_E:.6f}")
print(f"  μ_y from -Tr[D_relax r] = {mu_from_D:.6f}")

# z-axis (nonzero dipole): does -Tr[D_relax r_z] == μ_z == -dE/dFz ?
print("\n=== defining-property check on z-axis ===")
dEdFz = (etot_field(1e-4) if False else 0)
def etot_z(F):
    m=scf.RHF(mol); m.get_hcore=lambda *a: hcore_ao - F*dip_ao[2]; m.conv_tol=1e-12; m.kernel()
    return m.e_tot+mp.MP2(m).run().e_corr
mu_z_fromE = -(-(etot_z(1e-4)-etot_z(-1e-4))/(2e-4))  # μ=-dE/dF
mu_z_fromD = D_and_dEdF(0.0, 2)
print(f"  μ_z from -dE/dFz = {mu_z_fromE:.6f}")
print(f"  μ_z from -Tr[D_relax r_z] = {mu_z_fromD:.6f}")
print(f"  (pyscf relaxed μ_z was -0.652736)")

# The 2% gap (exact integrals!) means D_relax ≠ true dE/dh density. Find missing
# piece: compare my D_relax to PySCF's TRUE relaxed dm1 (unrelaxed + captured z),
# AND to the dm that would give -0.652732. Check all 3 dipole axes via dE/dF.
print("\n=== D_relax vs energy-derivative density, all axes ===")
for ax,nm in [(0,'x'),(1,'y'),(2,'z')]:
    def ez(F):
        m=scf.RHF(mol); m.get_hcore=lambda *a: hcore_ao - F*dip_ao[ax]; m.conv_tol=1e-12; m.kernel()
        return m.e_tot+mp.MP2(m).run().e_corr
    mu_E = (ez(1e-4)-ez(-1e-4))/(2e-4)*(-1)*(-1)  # μ=-dE/dF, dE/dF=(ez(h)-ez(-h))/2h
    mu_E = -((ez(1e-4)-ez(-1e-4))/(2e-4))
    mu_D = D_and_dEdF(0.0, ax)
    print(f"  μ_{nm}: from E={mu_E:+.6f}  from D={mu_D:+.6f}  Δ={mu_E-mu_D:+.6f}")
# Compare my static D to PySCF unrelaxed dm1 + captured z (the "true" relaxed):
D_ps_relaxed = dm1_mo.copy(); D_ps_relaxed[Vv,O]+=z_ps; D_ps_relaxed[O,Vv]+=z_ps.T
D_me = static_relaxed_dm_validated(Imo, e)
print("  ‖D_me - D_ps_relaxed‖ =", np.max(np.abs(D_me - D_ps_relaxed)))
print("  block diffs: oo=%.2e vv=%.2e ov=%.2e" % (
    np.max(np.abs((D_me-D_ps_relaxed)[O,O])),
    np.max(np.abs((D_me-D_ps_relaxed)[Vv,Vv])),
    np.max(np.abs((D_me-D_ps_relaxed)[Vv,O]))))
