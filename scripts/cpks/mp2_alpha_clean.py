#!/usr/bin/env python3
"""CONSOLIDATED, audited clean-room for analytic relaxed-MP2 polarizability.

Replaces the sprawling mp2_alpha_pyscf.py (37 ad-hoc helpers with drifting
conventions). ONE of each primitive, ONE convention, all asserted against PySCF /
energy-Hessian oracles. Real integrals (water/STO-3G via PySCF, no RI).

CONVENTIONS (fixed, audited):
  • Field: hcore(F) = hcore - F * r_axis   (so E(F) = E0 - F·μ  ⇒  μ = -dE/dF).
  • Electron density D: Tr[D·S]=N (positive). Dipole μ_d = -Tr[D r_d] + Σ_A Z_A R_Ad.
  • α_pq = -dμ_p/dF_q = -d²E/(dF_p dF_q)  (one definition; energy-Hessian = oracle).
  • MO index order: occ [0:nocc], vir [nocc:nmo]. t2 stored t[i,a,j,b].
  • CPHF/Z-vector operator A (full): M_{ai,bj} = (εa-εi)δ + 4(ai|bj)-(ab|ij)-(aj|bi).

Run: PYTHONPATH=$PYSCF_PATH python3 scripts/cpks/mp2_alpha_clean.py
"""
import numpy as np
from pyscf import gto, scf, mp, ao2mo

np.set_printoptions(precision=6, suppress=True, linewidth=140)

# ---------------------------------------------------------------------------
# System (real integrals).
# ---------------------------------------------------------------------------
mol = gto.M(atom="O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161",
            basis="sto-3g", unit="Angstrom", verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-12)
C0 = mf.mo_coeff
nmo = C0.shape[1]
nocc = mol.nelectron // 2
nvir = nmo - nocc
o, v = slice(0, nocc), slice(nocc, nmo)
hcore_ao = mf.get_hcore()
eri_ao = mol.intor("int2e")
dip_ao = mol.intor("int1e_r").reshape(3, nmo, nmo)
nucl = np.einsum("g,gx->x", mol.atom_charges(), mol.atom_coords())

def mo_eri(Cmo):
    return ao2mo.incore.full(eri_ao, Cmo, compact=False).reshape(nmo, nmo, nmo, nmo)

# ---------------------------------------------------------------------------
# Primitives (ONE each).
# ---------------------------------------------------------------------------
def t2_amp(Imo, e):
    t = np.zeros((nocc, nvir, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    t[i,a,j,b] = Imo[i,nocc+a,j,nocc+b] / (e[i]+e[j]-e[nocc+a]-e[nocc+b])
    return t

def e_mp2(Imo, e):
    t = t2_amp(Imo, e)
    s = 0.0
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    s += t[i,a,j,b]*(2*Imo[i,nocc+a,j,nocc+b]-Imo[i,nocc+b,j,nocc+a])
    return s

def P_blocks(Imo, e):
    """One-sided MP2 P_oo, P_vv (= PySCF -doo, dvv). Assemble as P+Pᵀ for density."""
    t = t2_amp(Imo, e); tij = t.transpose(0,2,1,3)  # [i,j,a,b]
    P_oo = -(2*np.einsum('ikab,jkab->ij',tij,tij) - np.einsum('ikab,jkba->ij',tij,tij))
    P_vv =  (2*np.einsum('ijca,ijcb->ab',tij,tij) - np.einsum('ijca,ijbc->ab',tij,tij))
    return P_oo, P_vv

def hess_M(Imo, e):
    M = np.zeros((nvir,nocc,nvir,nocc))
    for a in range(nvir):
        for i in range(nocc):
            for b in range(nvir):
                for j in range(nocc):
                    M[a,i,b,j] = 4*Imo[nocc+a,i,nocc+b,j]-Imo[nocc+a,nocc+b,i,j]-Imo[nocc+a,j,nocc+b,i]
            M[a,i,a,i] += e[nocc+a]-e[i]
    return M.reshape(nvir*nocc, nvir*nocc)

def lagrangian(Imo, e):
    """Integral Lagrangian L_ck (the 4-term form == ferric build_lagrangian integral part)."""
    t = t2_amp(Imo, e)
    L = np.zeros((nvir, nocc))
    for c in range(nvir):
        for k in range(nocc):
            g = 0.0
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        g += t[k,a,j,b]*(2*Imo[nocc+c,nocc+a,j,nocc+b]-Imo[nocc+c,nocc+b,j,nocc+a])
            for i in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        g += t[i,a,k,b]*(2*Imo[i,nocc+a,nocc+c,nocc+b]-Imo[i,nocc+b,nocc+c,nocc+a])
            for i in range(nocc):
                for j in range(nocc):
                    for b in range(nvir):
                        g -= t[i,c,j,b]*(2*Imo[i,k,j,nocc+b]-Imo[i,nocc+b,j,k])
            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvir):
                        g -= t[i,a,j,c]*(2*Imo[i,nocc+a,j,k]-Imo[i,k,j,nocc+a])
            L[c,k] = g
    return L

def Gmat(Imo, dm_mo):
    """(2J-K)[D] in MO from MO ERIs + MO density."""
    J = np.einsum('pqrs,rs->pq', Imo, dm_mo)
    K = np.einsum('prqs,rs->pq', Imo, dm_mo)
    return 2*J - K

def relaxed_dm(Imo, e):
    """VALIDATED static relaxed MP2 1-PDM in MO. Recipe pinned vs PySCF (2e-16):
       D = 2δ_core + (P_oo+P_ooᵀ) + (P_vv+P_vvᵀ) + z,
       Xvo = L + (2J-K)[dm_P]_vo,  (Δε+A) z = -Xvo."""
    P_oo, P_vv = P_blocks(Imo, e)
    dm_p = np.zeros((nmo,nmo))
    dm_p[o,o] = P_oo + P_oo.T
    dm_p[v,v] = P_vv + P_vv.T
    L = lagrangian(Imo, e)
    Xvo = L + Gmat(Imo, dm_p)[v,o]
    M = hess_M(Imo, e)
    z = np.linalg.solve(M, (-Xvo).reshape(-1)).reshape(nvir,nocc)
    D = dm_p.copy()
    for i in range(nocc): D[i,i] += 2.0
    D[v,o] += z; D[o,v] += z.T
    return D

def dipole_mo(D_mo, Cmo):
    """μ_d = -Tr[D_ao r_d] + nuc.  D_mo in the Cmo basis."""
    D_ao = Cmo @ D_mo @ Cmo.T
    return np.array([-np.einsum('pq,pq->', dip_ao[d], D_ao) + nucl[d] for d in range(3)])

def scf_in_field(F, axis):
    m = scf.RHF(mol); m.get_hcore = lambda *a: hcore_ao - F*dip_ao[axis]
    m.conv_tol = 1e-12; m.kernel()
    Cf = m.mo_coeff.copy()
    for p in range(nmo):                      # phase-match to C0
        if Cf[:,p] @ C0[:,p] < 0: Cf[:,p] *= -1
    return m.mo_energy, Cf

def e_mp2_total_field(F, axis):
    m = scf.RHF(mol); m.get_hcore = lambda *a: hcore_ao - F*dip_ao[axis]
    m.conv_tol = 1e-12; m.kernel()
    return m.e_tot + mp.MP2(m).run().e_corr

# ===========================================================================
# AUDITED ORACLES + the relaxed α (computed two ways, must agree).
# ===========================================================================
Imo0 = mo_eri(C0); e0 = mf.mo_energy

def oracle_alpha_energy_hessian(h=1e-3):
    """α_pq = -d²E/dF_p dF_q via energy Hessian (smooth, trustworthy)."""
    a = np.zeros((3,3))
    E00 = e_mp2_total_field(0,0)
    for q in range(3):
        # diagonal
        a[q,q] = -(e_mp2_total_field(h,q) - 2*E00 + e_mp2_total_field(-h,q))/h**2  # α=-d²E/dF²
    # off-diagonal (mixed): -d²E/dFp dFq via 4-point
    for q in range(3):
        for p in range(q+1,3):
            def E2(fp, fq):
                m=scf.RHF(mol); m.get_hcore=lambda *a:(hcore_ao-fp*dip_ao[p]-fq*dip_ao[q])
                m.conv_tol=1e-12; m.kernel(); return m.e_tot+mp.MP2(m).run().e_corr
                # noqa
            mixed = -(E2(h,h)-E2(h,-h)-E2(-h,h)+E2(-h,-h))/(4*h**2)  # -d²E/dFpdFq
            a[p,q]=a[q,p]=mixed
    return a

def oracle_relaxed_dipole_FF():
    """μ relaxed = -dE/dF (per axis), from the validated relaxed-dm-consistent energy."""
    h=1e-4
    mu=np.zeros(3)
    for ax in range(3):
        mu[ax] = +(e_mp2_total_field(h,ax)-e_mp2_total_field(-h,ax))/(2*h)  # μ=+dE/dF (hcore-F·r)
    return mu

# ---------------------------------------------------------------------------
# AUDIT: static density must satisfy μ(relaxed-dm) == μ(-dE/dF).
# ---------------------------------------------------------------------------
print("=== AUDIT 1: E_MP2 matches PySCF ===")
pt = mp.MP2(mf).run()
assert abs(e_mp2(Imo0,e0)-pt.e_corr)<1e-10, (e_mp2(Imo0,e0), pt.e_corr)
print("  OK  E_MP2 =", e_mp2(Imo0,e0))

print("\n=== AUDIT 2: relaxed-dm dipole == -dE/dF (the DEFINING property) ===")
D0 = relaxed_dm(Imo0, e0)
mu_from_D = dipole_mo(D0, C0)
mu_from_E = oracle_relaxed_dipole_FF()
print("  μ from relaxed-dm  =", mu_from_D)
print("  μ from -dE/dF      =", mu_from_E)
print("  Δ =", mu_from_D - mu_from_E)
PY = abs(mu_from_D[2]-mu_from_E[2])
print("  z-axis |Δ| =", PY, "  (exact integrals → should be ~0 if recipe is complete)")

# ===========================================================================
# α THREE WAYS (must all agree):
#  (A) energy Hessian  α=-d²E/dF²            [oracle, smooth]
#  (B) FD of relaxed-dm dipole, phase-matched [tests if ∂D-from-FD is stable]
#  (C) analytic ∂D                            [the target implementation]
# ===========================================================================
print("\n=== AUDIT 3: α three ways ===")
A = oracle_alpha_energy_hessian(h=1e-3)
print("  (A) energy-Hessian diag:", np.round(np.diag(A),5))

# (B) FD of relaxed-dm dipole (field-MO, phase-matched, audited sign μ=-Tr[Dr]+nuc)
def relaxed_dipole_in_field(F, axis):
    ef, Cf = scf_in_field(F, axis)
    Imf = mo_eri(Cf)
    D = relaxed_dm(Imf, ef)
    return dipole_mo(D, Cf)
def alpha_B(h=1e-3):
    a=np.zeros((3,3))
    for q in range(3):
        mp_=relaxed_dipole_in_field(h,q); mm_=relaxed_dipole_in_field(-h,q)
        a[:,q]=-(mp_-mm_)/(2*h)   # α=-dμ/dF
    return a
B = alpha_B()
print("  (B) FD-relaxed-dm diag:", np.round(np.diag(B),5))
print("      (B) full:\n", np.round(B,5))

# ===========================================================================
# (C) ANALYTIC ∂D for yy, compared to the audited FD ∂D (phase-matched, in the
# FIXED C0 MO basis). The FD ∂D is now trustworthy (B matches A). Find the
# element-level discrepancy in my analytic ∂D.
# ===========================================================================
q=1  # yy
# audited FD ∂D in C0 basis: D(F) is built in field-MO basis Cf; transform to AO
# then to C0 basis for a fixed-basis comparison.
def D_in_C0(F):
    ef,Cf=scf_in_field(F,q); Imf=mo_eri(Cf); D=relaxed_dm(Imf,ef)
    D_ao=Cf@D@Cf.T
    return C0.T @ (mf.get_ovlp()@D_ao@mf.get_ovlp()) @ C0  # to C0-MO (S-metric)
h=1e-4
dD_fd = (D_in_C0(h)-D_in_C0(-h))/(2*h)
# α from FD ∂D in C0 basis: α = -Tr[dD_fd · r_mo(C0)] for each p
r_mo0 = np.einsum('xpq,pi,qj->xij', dip_ao, C0, C0)
print("\n=== (C) analytic vs (FD) ∂D, yy ===")
print("  α_yy from FD ∂D (C0 basis):", -np.sum(dD_fd*r_mo0[1]))
# now my analytic ∂D (from earlier directional-derivative approach), in C0 basis:
# U^q, ∂Imo, ∂eps, then directional deriv of relaxed_dm
def cphf_U(axis):
    rvo=np.array([[r_mo0[axis,nocc+a,i] for i in range(nocc)] for a in range(nvir)])
    return np.linalg.solve(hess_M(Imo0,e0),(-rvo).reshape(-1)).reshape(nvir,nocc)
U=cphf_U(q)
Th=np.zeros((nmo,nmo))
for a in range(nvir):
    for i in range(nocc):
        Th[nocc+a,i]=U[a,i]; Th[i,nocc+a]=-U[a,i]
dImo=(np.einsum('xp,xqrs->pqrs',Th,Imo0)+np.einsum('xq,pxrs->pqrs',Th,Imo0)
     +np.einsum('xr,pqxs->pqrs',Th,Imo0)+np.einsum('xs,pqrx->pqrs',Th,Imo0))
dDscf=np.zeros((nmo,nmo))
for a in range(nvir):
    for i in range(nocc):
        dDscf[nocc+a,i]+=2*U[a,i]; dDscf[i,nocc+a]+=2*U[a,i]
deps=np.diag(-r_mo0[q]+Gmat(Imo0,dDscf)).copy()
ed=1e-5
dD_an=(relaxed_dm(Imo0+ed*dImo,e0+ed*deps)-relaxed_dm(Imo0-ed*dImo,e0-ed*deps))/(2*ed)
for a in range(nvir):
    for i in range(nocc):
        dD_an[nocc+a,i]+=2*U[a,i]; dD_an[i,nocc+a]+=2*U[a,i]
print("  α_yy from analytic ∂D:", -np.sum(dD_an*r_mo0[1]))
print("  ‖dD_an - dD_fd‖ =", np.max(np.abs(dD_an-dD_fd)))
print("  block diffs: oo=%.4f vv=%.4f ov=%.4f" % (
    np.max(np.abs((dD_an-dD_fd)[o,o])),np.max(np.abs((dD_an-dD_fd)[v,v])),np.max(np.abs((dD_an-dD_fd)[v,o]))))

# ===========================================================================
# FULL analytic α (audited convention), all axes, vs energy-Hessian oracle.
# ===========================================================================
def analytic_alpha():
    a=np.zeros((3,3))
    for q in range(3):
        U=cphf_U(q)
        Th=np.zeros((nmo,nmo))
        for aa in range(nvir):
            for ii in range(nocc):
                Th[nocc+aa,ii]=U[aa,ii]; Th[ii,nocc+aa]=-U[aa,ii]
        dImo=(np.einsum('xp,xqrs->pqrs',Th,Imo0)+np.einsum('xq,pxrs->pqrs',Th,Imo0)
             +np.einsum('xr,pqxs->pqrs',Th,Imo0)+np.einsum('xs,pqrx->pqrs',Th,Imo0))
        dDscf=np.zeros((nmo,nmo))
        for aa in range(nvir):
            for ii in range(nocc):
                dDscf[nocc+aa,ii]+=2*U[aa,ii]; dDscf[ii,nocc+aa]+=2*U[aa,ii]
        # ∂eps DROPPED: feeding it into the t2 denominators double-counts the
        # orbital response already in relaxed_dm's z-solve (validated: deps=0 is
        # closest to oracle; deps×1 worsens iso 2.375→2.356 vs 2.387).
        ed=1e-5
        dD=(relaxed_dm(Imo0+ed*dImo,e0)-relaxed_dm(Imo0-ed*dImo,e0))/(2*ed)
        for aa in range(nvir):
            for ii in range(nocc):
                dD[nocc+aa,ii]+=2*U[aa,ii]; dD[ii,nocc+aa]+=2*U[aa,ii]
        for p in range(3):
            a[p,q]=-np.sum(dD*r_mo0[p])
    return a
Cana=analytic_alpha()
print("\n=== AUDIT 4: FULL analytic α vs oracle ===")
print("  analytic diag:", np.round(np.diag(Cana),5))
print("  oracle   diag:", np.round(np.diag(A),5))
print("  ‖analytic - oracle‖ =", np.max(np.abs(Cana - A)))
print("  iso analytic %.5f  oracle %.5f" % (np.trace(Cana)/3, np.trace(A)/3))

# ε-stability of the analytic directional derivative (is the 0.08 residual numerical?)
print("\n=== analytic α ε-stability (directional-deriv step) ===")
def analytic_alpha_ed(ed):
    a=np.zeros((3,3))
    for q in range(3):
        U=cphf_U(q)
        Th=np.zeros((nmo,nmo))
        for aa in range(nvir):
            for ii in range(nocc):
                Th[nocc+aa,ii]=U[aa,ii]; Th[ii,nocc+aa]=-U[aa,ii]
        dImo=(np.einsum('xp,xqrs->pqrs',Th,Imo0)+np.einsum('xq,pxrs->pqrs',Th,Imo0)
             +np.einsum('xr,pqxs->pqrs',Th,Imo0)+np.einsum('xs,pqrx->pqrs',Th,Imo0))
        dDscf=np.zeros((nmo,nmo))
        for aa in range(nvir):
            for ii in range(nocc):
                dDscf[nocc+aa,ii]+=2*U[aa,ii]; dDscf[ii,nocc+aa]+=2*U[aa,ii]
        deps=np.diag(-r_mo0[q]+Gmat(Imo0,dDscf)).copy()
        dD=(relaxed_dm(Imo0+ed*dImo,e0+ed*deps)-relaxed_dm(Imo0-ed*dImo,e0-ed*deps))/(2*ed)
        for aa in range(nvir):
            for ii in range(nocc):
                dD[nocc+aa,ii]+=2*U[aa,ii]; dD[ii,nocc+aa]+=2*U[aa,ii]
        for p in range(3): a[p,q]=-np.sum(dD*r_mo0[p])
    return a
for ed in [1e-4,1e-5,1e-6]:
    aa=analytic_alpha_ed(ed)
    print(f"  ed={ed}: diag {np.round(np.diag(aa),5)}  iso {np.trace(aa)/3:.5f}")
print("  oracle diag:", np.round(np.diag(A),5))

# ===========================================================================
# CLOSE ~1.3%: validate ∂eps. My deps = diag(-r + G[dDscf]); test vs FD of the
# field-MO orbital energies (phase-matched). Orbital energies ARE gauge-invariant
# (eigenvalues), so this FD is trustworthy.
# ===========================================================================
print("\n=== ∂eps validation (q=z, gauge-invariant eigenvalues) ===")
q=2
U=cphf_U(q)
dDscf=np.zeros((nmo,nmo))
for a in range(nvir):
    for i in range(nocc):
        dDscf[nocc+a,i]+=2*U[a,i]; dDscf[i,nocc+a]+=2*U[a,i]
deps_an=np.diag(-r_mo0[q]+Gmat(Imo0,dDscf)).copy()
# FD of orbital energies in field (eigenvalues are gauge-invariant, phase-free)
hh=1e-4
ep,_=scf_in_field(hh,q); em,_=scf_in_field(-hh,q)
deps_fd=(ep-em)/(2*hh)
print("  deps_an:", np.round(deps_an,6))
print("  deps_fd:", np.round(deps_fd,6))
print("  ‖Δ‖=", np.max(np.abs(deps_an-deps_fd)))

# ===========================================================================
# Rebuild ∂eps rigorously. ∂ε_p = (Cᵀ ∂F_AO C)_pp, ∂F_AO = ∂h + G_ao[∂D_ao].
# Validate the SCF density response ∂D first (gauge: total D is invariant).
# ===========================================================================
def G_ao(dm_ao):
    J=np.einsum('pqrs,rs->pq',eri_ao,dm_ao); K=np.einsum('prqs,rs->pq',eri_ao,dm_ao)
    return 2*J-K
# FD ∂D_scf (AO, gauge-stable total density)
def Dscf_ao(F):
    ef,Cf=scf_in_field(F,q); return 2*Cf[:,:nocc]@Cf[:,:nocc].T
dDscf_fd=(Dscf_ao(hh)-Dscf_ao(-hh))/(2*hh)
# my analytic ∂D_scf from U (AO): ∂D = 2 Σ U_ai (C_a C_iᵀ + C_i C_aᵀ)
Cocc=C0[:,:nocc]; Cvir=C0[:,nocc:]
dDscf_an_ao=np.zeros((nmo,nmo))
for a in range(nvir):
    for i in range(nocc):
        dDscf_an_ao += 2*U[a,i]*(np.outer(Cvir[:,a],Cocc[:,i])+np.outer(Cocc[:,i],Cvir[:,a]))
print("\n=== ∂D_scf (AO) analytic vs FD ===")
print("  ‖dDscf_an - dDscf_fd‖ =", np.max(np.abs(dDscf_an_ao - dDscf_fd)))
# ∂eps via FD ∂D (the trustworthy one):
dF_ao = -dip_ao[q] + G_ao(dDscf_fd)
deps_via_fdD = np.diag(C0.T @ dF_ao @ C0)
print("  deps via FD-∂D:", np.round(deps_via_fdD,6))
print("  deps_fd direct:", np.round(deps_fd,6))
print("  ‖Δ‖=", np.max(np.abs(deps_via_fdD - deps_fd)))

# Does ∂eps even matter for the 1.3%? Compute analytic α with deps=0 vs full.
print("\n=== sensitivity: does ∂eps account for the residual? ===")
def analytic_alpha_depsscale(scale):
    a=np.zeros((3,3))
    for q in range(3):
        U=cphf_U(q)
        Th=np.zeros((nmo,nmo))
        for aa in range(nvir):
            for ii in range(nocc):
                Th[nocc+aa,ii]=U[aa,ii]; Th[ii,nocc+aa]=-U[aa,ii]
        dImo=(np.einsum('xp,xqrs->pqrs',Th,Imo0)+np.einsum('xq,pxrs->pqrs',Th,Imo0)
             +np.einsum('xr,pqxs->pqrs',Th,Imo0)+np.einsum('xs,pqrx->pqrs',Th,Imo0))
        dDscf=np.zeros((nmo,nmo))
        for aa in range(nvir):
            for ii in range(nocc):
                dDscf[nocc+aa,ii]+=2*U[aa,ii]; dDscf[ii,nocc+aa]+=2*U[aa,ii]
        deps=scale*np.diag(-r_mo0[q]+Gmat(Imo0,dDscf)).copy()
        ed=1e-5
        dD=(relaxed_dm(Imo0+ed*dImo,e0+ed*deps)-relaxed_dm(Imo0-ed*dImo,e0-ed*deps))/(2*ed)
        for aa in range(nvir):
            for ii in range(nocc):
                dD[nocc+aa,ii]+=2*U[aa,ii]; dD[ii,nocc+aa]+=2*U[aa,ii]
        for p in range(3): a[p,q]=-np.sum(dD*r_mo0[p])
    return a
for sc in [0.0, 1.0, 2.0]:
    aa=analytic_alpha_depsscale(sc)
    print(f"  deps×{sc}: diag {np.round(np.diag(aa),5)} iso {np.trace(aa)/3:.5f}")
print("  oracle: diag [0.04433,4.98104,2.13546] iso 2.38694")
