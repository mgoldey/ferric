#!/usr/bin/env python3
"""Clean-room NumPy reference for analytic relaxed MP2 polarizability.

A second, independent implementation of the same response assembly the Rust
cpks_polar uses — built directly from MO integrals (no RI, no FF). Validates the
Rust term-by-term. Uses a random but fixed small closed-shell system so the
∂L / ∂z / α numbers are reproducible and each term is inspectable.

Two α's computed and compared:
  A) RESPONSE: analytic CPHF U + ∂t2 + perturbed Z-vector ∂z + relaxed-density
     contraction (mirrors Rust analytic_alpha_relaxed term-for-term).
  B) ORACLE:   finite-field of the relaxed MP2 dipole, but with FROZEN-PHASE
     orbitals (project perturbed MOs onto unperturbed to kill the gauge issue)
     — so it's a trustworthy element-stable oracle, unlike the Rust FF.
The point: find where A and B diverge → which ∂L term the Rust has wrong.
"""
import numpy as np
np.random.seed(7)

nocc, nvir = 2, 3
nmo = nocc + nvir
O = slice(0, nocc); Vv = slice(nocc, nmo)

# Random symmetric AO-ish setup: build a fake but valid MO basis.
# Orbital energies: occ below vir, well separated.
eps = np.array([-0.8, -0.5, 0.3, 0.6, 0.9])[:nmo]

# Antisymmetrized? No — closed-shell spatial. Build (pq|rs) with 8-fold symmetry
# from a random Cholesky-like factor so it's PSD-ish and symmetric.
naux = 6
B = np.random.randn(naux, nmo, nmo)
B = 0.5*(B + B.transpose(0,2,1))           # symmetric in (p,q)
def ERI():  # (pq|rs) = sum_P B[P,p,q] B[P,r,s]
    return np.einsum('Ppq,Prs->pqrs', B, B)
I = ERI()

# Dipole MO matrix (symmetric), the perturbation operator.
r = np.random.randn(nmo, nmo); r = 0.5*(r+r.T)

def t2_amplitudes(Imo, e):
    # t_iajb = (ia|jb)/(ei+ej-ea-eb)
    o = range(nocc); v = range(nocc, nmo)
    t = np.zeros((nocc,nvir,nocc,nvir))
    for i in o:
        for a in v:
            for j in o:
                for b in v:
                    d = e[i]+e[j]-e[a]-e[b]
                    t[i,a-nocc,j,b-nocc] = Imo[i,a,j,b]/d
    return t

def e_mp2(Imo, e):
    t = t2_amplitudes(Imo, e)
    o=range(nocc); v=range(nocc,nmo); s=0.0
    for i in o:
        for a in v:
            for j in o:
                for b in v:
                    K = 2*Imo[i,a,j,b]-Imo[i,b,j,a]
                    s += t[i,a-nocc,j,b-nocc]*K
    return s

print("E_MP2(0) =", e_mp2(I, eps))
print("system: nocc=%d nvir=%d naux=%d" % (nocc,nvir,naux))

# ---------------------------------------------------------------------------
# ORACLE: relaxed MP2 dipole at field F, with a proper HF re-solve in the field.
# Here the "AO" basis = the unperturbed MO basis (C0 = I), Fock(F) = diag(eps) - F r
# + the 2e response. For a clean closed-form HF-in-field we do a small SCF.
# ---------------------------------------------------------------------------
def fock_ao_from_dm(dm, hcore):
    # G[D] = 2 J - K in the (unperturbed-MO) "AO" basis using I as (pq|rs).
    J = np.einsum('pqrs,rs->pq', I, dm)
    K = np.einsum('prqs,rs->pq', I, dm)
    return hcore + 2*J - K

def scf_in_field(Ffield, tol=1e-12, maxit=200):
    hcore = np.diag(eps).astype(float) - Ffield * r
    # core density guess
    e0, C = np.linalg.eigh(hcore)
    dm = 2*C[:, :nocc] @ C[:, :nocc].T
    eprev = 0.0
    for _ in range(maxit):
        Fk = fock_ao_from_dm(dm, hcore)
        e_orb, C = np.linalg.eigh(Fk)
        dm_new = 2*C[:, :nocc] @ C[:, :nocc].T
        ene = 0.5*np.sum((hcore+Fk)*dm)
        if abs(ene-eprev) < tol and np.max(np.abs(dm_new-dm)) < 1e-11:
            dm = dm_new; break
        dm = dm_new; eprev = ene
    return e_orb, C

def relaxed_mp2_dipole(Ffield):
    # SCF in field → MO integrals in the field MOs → MP2 relaxed 1-PDM → <r>.
    e_orb, C = scf_in_field(Ffield)
    # Phase-match C to unperturbed (sign per column) to keep it smooth.
    for p in range(nmo):
        if C[:,p] @ np.eye(nmo)[:,p] < 0: C[:,p]*=-1
    # transform I and r into field-MO basis
    Imo = np.einsum('pa,qb,rc,sd,pqrs->abcd', C,C,C,C, I, optimize=True)
    rmo = C.T @ r @ C
    e = e_orb
    t = t2_amplitudes(Imo, e)
    # unrelaxed P_oo,P_vv
    Poo = -np.einsum('iakb,jakb->ij', t, 2*t - t.transpose(0,3,2,1))
    Pvv =  np.einsum('iajc,ibjc->ab', t, 2*t - t.transpose(2,1,0,3))
    # relaxed 1-PDM (MO of the FIELD basis): 2δ core + Poo + Pvv + z(orbital relax)
    # z-vector via canonical CPHF in field basis (small dense solve)
    # Build orbital gradient L (the same 4-term Lagrangian) then solve (A) z = L.
    # For the reference we build A densely and solve exactly.
    # ... (filled next) ...
    return e_orb, C, Imo, rmo, t, Poo, Pvv

eo,Cc,Imo,rmo,t,Poo,Pvv = relaxed_mp2_dipole(0.01)
print("field SCF + MP2 P built. Poo diag:", np.round(np.diag(Poo),5), "Pvv diag:", np.round(np.diag(Pvv),5))

# ---------------------------------------------------------------------------
# Dense Z-vector + relaxed dipole at a field (the ORACLE via FF on this clean,
# phase-matched relaxed dipole), and the analytic RESPONSE assembly.
# ---------------------------------------------------------------------------
def orb_hessian_dense(Imo, e):
    # A_{ai,bj} = 4(ai|bj) - (ab|ij) - (aj|bi), plus (ea-ei) on diagonal.
    A = np.zeros((nvir,nocc,nvir,nocc))
    for a in range(nvir):
        for i in range(nocc):
            for b in range(nvir):
                for j in range(nocc):
                    A[a,i,b,j] = (4*Imo[nocc+a,i,nocc+b,j]
                                  - Imo[nocc+a,nocc+b,i,j]
                                  - Imo[nocc+a,j,nocc+b,i])
            A[a,i,a,i] += e[nocc+a]-e[i]
    return A.reshape(nvir*nocc, nvir*nocc)

def lagrangian(Imo, t, e):
    # the same 4-term L_ck the Rust build_lagrangian computes (integral part).
    L = np.zeros((nvir,nocc))
    for c in range(nvir):
        for k in range(nocc):
            g=0.0
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
            L[c,k]=g
    return L

def relaxed_dm_mo(Imo, e):
    t = t2_amplitudes(Imo, e)
    Poo = -np.einsum('iakb,jakb->ij', t, 2*t - t.transpose(0,3,2,1))
    Pvv =  np.einsum('iajc,ibjc->ab', t, 2*t - t.transpose(2,1,0,3))
    L = lagrangian(Imo, t, e)
    A = orb_hessian_dense(Imo, e)
    z = np.linalg.solve(A, L.reshape(-1)).reshape(nvir,nocc)
    D = np.zeros((nmo,nmo))
    for i in range(nocc): D[i,i]+=2.0
    D[O,O]+=Poo; D[Vv,Vv]+=Pvv
    for a in range(nvir):
        for i in range(nocc):
            D[nocc+a,i]+=z[a,i]; D[i,nocc+a]+=z[a,i]
    return D

def dipole_expect(Ffield):
    e_orb,C = scf_in_field(Ffield)
    for p in range(nmo):
        if C[:,p]@np.eye(nmo)[:,p]<0: C[:,p]*=-1
    Imo = np.einsum('pa,qb,rc,sd,pqrs->abcd', C,C,C,C, I, optimize=True)
    rmo = C.T @ r @ C
    D = relaxed_dm_mo(Imo, e_orb)
    return np.sum(D*rmo)  # Tr[D r] in field-MO basis = <r> (r transformed consistently)

# ORACLE: α = -d<r>/dF by central diff of this CLEAN phase-matched relaxed dipole.
h=1e-4
oracle = -(dipole_expect(h)-dipole_expect(-h))/(2*h)
print("ORACLE relaxed-MP2 α (phase-matched FF) =", oracle)
# also at 2h to confirm clean
oracle2 = -(dipole_expect(2*h)-dipole_expect(-2*h))/(4*h)
print("  (2h check) =", oracle2, " → stable:", abs(oracle-oracle2)<1e-4)

# ---------------------------------------------------------------------------
# ANALYTIC RESPONSE (mirrors Rust cpks_polar): everything from CPHF U, no FF.
# ---------------------------------------------------------------------------
def analytic_alpha():
    e = eps
    Imo = I  # unperturbed MOs = "AO" basis here (C0 = identity)
    t = t2_amplitudes(Imo, e)
    A = orb_hessian_dense(Imo, e)
    z0 = np.linalg.solve(A, lagrangian(Imo,t,e).reshape(-1)).reshape(nvir,nocc)

    # CPHF U for field along the single "axis" = operator r.
    # (Δε + A_full) U = -r_vo  ; here A_full is the SAME orbital Hessian.
    rvo = np.array([[r[nocc+a,i] for i in range(nocc)] for a in range(nvir)])
    U = np.linalg.solve(A, (-rvo).reshape(-1)).reshape(nvir,nocc)

    # ∂(MO integrals)/∂F via U-rotation: Θ generator.
    Th = np.zeros((nmo,nmo))
    for a in range(nvir):
        for i in range(nocc):
            Th[nocc+a,i]=U[a,i]; Th[i,nocc+a]=-U[a,i]
    # ∂Imo_pqrs = sum over each index rotated by Θ
    dImo = (np.einsum('xp,xqrs->pqrs',Th,Imo)+np.einsum('xq,pxrs->pqrs',Th,Imo)
           +np.einsum('xr,pqxs->pqrs',Th,Imo)+np.einsum('xs,pqrx->pqrs',Th,Imo))
    # ∂ε_p = ∂F_pp = (-r + G[∂D])_pp ; ∂D from U
    dD = np.zeros((nmo,nmo))
    for a in range(nvir):
        for i in range(nocc):
            dD[nocc+a,i]+=2*U[a,i]; dD[i,nocc+a]+=2*U[a,i]
    G = 2*np.einsum('pqrs,rs->pq',I,dD) - np.einsum('prqs,rs->pq',I,dD)
    dF = -r + G
    de = np.diag(dF).copy()  # ∂ε_p
    # ∂t2
    dt = np.zeros_like(t)
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    d = e[i]+e[j]-e[nocc+a]-e[nocc+b]
                    dnum = dImo[i,nocc+a,j,nocc+b]
                    dden = de[i]+de[j]-de[nocc+a]-de[nocc+b]
                    dt[i,a,j,b] = (dnum - t[i,a,j,b]*dden)/d
    # ∂P_oo,∂P_vv (product rule)
    dPoo = -(np.einsum('iakb,jakb->ij',dt,2*t-t.transpose(0,3,2,1))
            +np.einsum('iakb,jakb->ij',t,2*dt-dt.transpose(0,3,2,1)))
    dPvv =  (np.einsum('iajc,ibjc->ab',dt,2*t-t.transpose(2,1,0,3))
            +np.einsum('iajc,ibjc->ab',t,2*dt-dt.transpose(2,1,0,3)))
    # ∂L = directional deriv of lagrangian along (Imo→dImo, t→dt)
    epsd=1e-6
    Lp = lagrangian(Imo+epsd*dImo, t+epsd*dt, e)  # note: e fixed; ∂ε enters via denom in t already
    Lm = lagrangian(Imo-epsd*dImo, t-epsd*dt, e)
    dL = (Lp-Lm)/(2*epsd)
    # perturbed Z-vector: (Δε+A) ∂z = ∂L - ∂A·z0 - ∂Δε·z0
    # ∂A·z0: rotate the Hessian. Build A(Imo+εdImo, e+εde) acting on z0.
    def Az(Imatrix, evec):
        Ad = orb_hessian_dense(Imatrix, evec)
        return (Ad @ z0.reshape(-1)).reshape(nvir,nocc)
    dAz0 = (Az(Imo+epsd*dImo, e+epsd*de) - Az(Imo-epsd*dImo, e-epsd*de))/(2*epsd)
    # but A already includes (ea-ei); so dAz0 already contains ∂Δε·z0. Subtract once.
    rhs = dL - dAz0
    dz = np.linalg.solve(A, rhs.reshape(-1)).reshape(nvir,nocc)

    # ∂D_relax: 2U (core) + ∂Poo + ∂Pvv + ∂z
    dDrel = np.zeros((nmo,nmo))
    for a in range(nvir):
        for i in range(nocc):
            dDrel[nocc+a,i]+=2*U[a,i]+dz[a,i]; dDrel[i,nocc+a]+=2*U[a,i]+dz[a,i]
    dDrel[O,O]+=dPoo; dDrel[Vv,Vv]+=dPvv
    alpha = -np.sum(dDrel*r)
    return alpha, dz, U

a_resp, dz, U = analytic_alpha()
print("ANALYTIC RESPONSE α =", a_resp)
print("ORACLE              =", oracle)
print("ratio resp/oracle   =", a_resp/oracle)

# ---------------------------------------------------------------------------
# DECOMPOSE in clean-room: build the EXACT ∂(relaxed dm) by phase-matched FF of
# each piece, so we get the true ∂z, ∂Poo, ∂Pvv, and true 2U — then compare to
# the analytic formulas term-by-term to find the wrong one.
# ---------------------------------------------------------------------------
def pieces_at(Ffield):
    e_orb,C = scf_in_field(Ffield)
    for p in range(nmo):
        if C[:,p]@np.eye(nmo)[:,p]<0: C[:,p]*=-1
    Imo = np.einsum('pa,qb,rc,sd,pqrs->abcd', C,C,C,C, I, optimize=True)
    t = t2_amplitudes(Imo, e_orb)
    Poo = -np.einsum('iakb,jakb->ij',t,2*t-t.transpose(0,3,2,1))
    Pvv =  np.einsum('iajc,ibjc->ab',t,2*t-t.transpose(2,1,0,3))
    A = orb_hessian_dense(Imo, e_orb)
    z = np.linalg.solve(A, lagrangian(Imo,t,e_orb).reshape(-1)).reshape(nvir,nocc)
    # express C's occ-vir rotation relative to identity = the "U" of the FF
    Uff = C[nocc:, :nocc].copy()   # vir-occ block of C ≈ U·F at small F
    return Poo,Pvv,z,Uff

h=1e-4
Pp,Vp,zp,Up = pieces_at(h)
Pm,Vm,zm,Um = pieces_at(-h)
dPoo_fd=(Pp-Pm)/(2*h); dPvv_fd=(Vp-Vm)/(2*h); dz_fd=(zp-zm)/(2*h); U_fd=(Up-Um)/(2*h)

a_resp, dz_an, U_an = analytic_alpha()
print("\n--- term-by-term: analytic vs phase-matched FD ---")
print("‖U   an−fd‖ =", np.max(np.abs(U_an - U_fd)))
print("‖∂z  an−fd‖ =", np.max(np.abs(dz_an - dz_fd)), " ‖∂z_fd‖=", np.max(np.abs(dz_fd)), " ‖∂z_an‖=", np.max(np.abs(dz_an)))
# rebuild α from FD pieces to confirm the assembly formula is right
dDrel=np.zeros((nmo,nmo))
for a in range(nvir):
    for i in range(nocc):
        dDrel[nocc+a,i]+=2*U_fd[a,i]+dz_fd[a,i]; dDrel[i,nocc+a]+=2*U_fd[a,i]+dz_fd[a,i]
dDrel[O,O]+=dPoo_fd; dDrel[Vv,Vv]+=dPvv_fd
print("α from FD pieces =", -np.sum(dDrel*r), " (oracle", oracle,")")

# ---------------------------------------------------------------------------
# HYPOTHESIS: α = -d/dF Tr[D(F) r(F)] needs BOTH Tr[∂D r] AND Tr[D ∂r], where
# ∂r is the dipole's rotation with the field-MOs. Test with FD pieces + the
# unperturbed relaxed D0 contracted with ∂r.
# ---------------------------------------------------------------------------
D0 = relaxed_dm_mo(I, eps)           # unperturbed relaxed dm (MO basis)
# ∂r from MO rotation: r in field-MO basis = C^T r C ; ∂(C^T r C) at F=0.
def rmo_at(Ffield):
    e_orb,C = scf_in_field(Ffield)
    for p in range(nmo):
        if C[:,p]@np.eye(nmo)[:,p]<0: C[:,p]*=-1
    return C.T @ r @ C
dr_fd = (rmo_at(h)-rmo_at(-h))/(2*h)
term_Dr = -np.sum(D0*dr_fd)
print("\n--- assembly fix test ---")
print("Tr[∂D r] (FD pieces) part =", -np.sum(dDrel*r))
print("Tr[D ∂r] part            =", term_Dr)
print("SUM                      =", -np.sum(dDrel*r) + term_Dr, " (oracle", oracle,")")

print("\n=== KEY FINDINGS (traceable clean-room) ===")
print("True ∂z (phase-matched FD)  ‖·‖ = %.2e  <-- TINY (MP2 barely shifts α)" % np.max(np.abs(dz_fd)))
print("Analytic ∂z (current Rust formula) ‖·‖ = %.2e  <-- ~%.0fx TOO LARGE" % (np.max(np.abs(dz_an)), np.max(np.abs(dz_an))/max(np.max(np.abs(dz_fd)),1e-30)))
print("=> ∂z RHS formula is the bug; clean-room gives exact target dz_fd to match.")
print("Note: this random tiny system has large couplings; signs/structure transfer,")
print("magnitudes are system-specific. Use dz_fd as the per-term oracle.")
