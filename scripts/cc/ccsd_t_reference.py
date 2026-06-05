import numpy as np
from pyscf import gto, scf, cc, ao2mo

# Fully self-consistent INTERLEAVED (2k=alpha,2k+1=beta) spin-orbital reference,
# mirroring ferric's asym_phys + eo/ev exactly. t1/t2 converted to interleaved.
m=gto.M(atom="O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",basis="cc-pvdz",verbose=0)
mf=scf.RHF(m).run(); mycc=cc.CCSD(mf).run(); et_ref=mycc.ccsd_t()
mo=mf.mo_coeff; eps=mf.mo_energy; occ=mf.mo_occ
nao=mo.shape[1]; nocc_s=int(occ.sum()//2); nvir_s=nao-nocc_s
eri=ao2mo.restore(1,ao2mo.full(mf.mol,mo),nao)  # chemist (pq|rs), all MO

no=2*nocc_s; nv=2*nvir_s
# interleaved spin-orbital energies, occ block then vir block, each interleaved
eo=np.array([eps[i//2] for i in range(no)])          # occ spatial 0..nocc_s
ev=np.array([eps[nocc_s + a//2] for a in range(nv)]) # vir spatial
spn=lambda p:p&1
def g_block(P,Q,R,S, poff,qoff,roff,soff):
    """antisym <pq||rs> interleaved, p in space of size 2*lenP etc. offsets index into spatial eri."""
    pass

# Build needed antisym blocks directly in interleaved convention via asym_phys rule:
# <pq||rs> = (pr|qs)[spin p=r,q=s] - (ps|qr)[spin p=s,q=r], spatial index = global MO index.
def asym(pr_is_occ):
    pass

# general builder: given 4 spaces each a list of spatial MO indices, build <pq||rs>
occ_mo=list(range(nocc_s)); vir_mo=list(range(nocc_s,nao))
def build(spP,spQ,spR,spS):
    nP,nQ,nR,nS=len(spP)*2,len(spQ)*2,len(spR)*2,len(spS)*2
    out=np.zeros((nP,nQ,nR,nS))
    for p in range(nP):
     for q in range(nQ):
      for r in range(nR):
       for s in range(nS):
        v=0.0
        if spn(p)==spn(r) and spn(q)==spn(s): v+=eri[spP[p//2],spR[r//2],spQ[q//2],spS[s//2]]
        if spn(p)==spn(s) and spn(q)==spn(r): v-=eri[spP[p//2],spS[s//2],spQ[q//2],spR[r//2]]
        out[p,q,r,s]=v
    return out
O,V=occ_mo,vir_mo
bcei=build(V,V,V,O)  # <bc||ei>
majk=build(O,V,O,O)  # <ma||jk>
bcjk=build(V,V,O,O)  # <bc||jk>
oovv=build(O,O,V,V)

# t1/t2 in interleaved convention from RHF spatial amps
t1s=mycc.t1; t2s=mycc.t2  # t1s[i,a], t2s[i,j,a,b] spatial (alpha=beta)
t1=np.zeros((no,nv))
for i in range(no):
 for a in range(nv):
  if spn(i)==spn(a): t1[i,a]=t1s[i//2,a//2]
t2=np.zeros((no,no,nv,nv))
for i in range(no):
 for j in range(no):
  for a in range(nv):
   for b in range(nv):
    # spin-orbital t2 = <ij||ab>-type antisymmetric combination of spatial
    val=0.0
    if spn(i)==spn(a) and spn(j)==spn(b): val+=t2s[i//2,j//2,a//2,b//2]
    if spn(i)==spn(b) and spn(j)==spn(a): val-=t2s[i//2,j//2,b//2,a//2]
    t2[i,j,a,b]=val
d3=(eo[:,None,None,None,None,None]+eo[None,:,None,None,None,None]+eo[None,None,:,None,None,None]
    -ev[None,None,None,:,None,None]-ev[None,None,None,None,:,None]-ev[None,None,None,None,None,:])
t3c=(np.einsum('jkae,bcei->ijkabc',t2,bcei)-np.einsum('imbc,majk->ijkabc',t2,majk))
t3c=t3c-t3c.transpose(0,1,2,4,3,5)-t3c.transpose(0,1,2,5,4,3)
t3c=t3c-t3c.transpose(1,0,2,3,4,5)-t3c.transpose(2,1,0,3,4,5)
t3c/=d3
t3d=np.einsum('ia,bcjk->ijkabc',t1,bcjk); t3d/=d3
# NOTE: pyscf permutes t3d too; replicate
t3d2=np.einsum('ia,bcjk->ijkabc',t1,bcjk)
t3d2=t3d2-t3d2.transpose(0,1,2,4,3,5)-t3d2.transpose(0,1,2,5,4,3)
t3d2=t3d2-t3d2.transpose(1,0,2,3,4,5)-t3d2.transpose(2,1,0,3,4,5)
t3d2/=d3
et=np.einsum('ijkabc,ijkabc,ijkabc',(t3c+t3d2),d3,t3c)/36.0
print(f"ref et={et_ref:.10f}  interleaved mine={et:.10f}  diff={et-et_ref:.2e}")
