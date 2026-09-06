import numpy, scipy.linalg, traceback
from functools import reduce
from pyscf import gto, scf, lib

ri_bas = gto.basis.parse(open('/tmp/cc_pvdz_f12_optri.nw').read())  # nosec B108 -- local scratch input for a one-off reference run, not a shared temp file

def find_cabs(mol, auxmol, lindep=1e-8):
    cabs_mol=gto.conc_mol(mol,auxmol); nao=mol.nao_nr()
    s=cabs_mol.intor_symmetric('int1e_ovlp')
    ls12=scipy.linalg.solve(s[:nao,:nao],s[:nao,nao:],assume_a='pos')
    s[nao:,nao:]-=s[nao:,:nao].dot(ls12)
    w,v=scipy.linalg.eigh(s[nao:,nao:])
    c2=v[:,w>lindep]/numpy.sqrt(w[w>lindep]); c1=ls12.dot(c2)
    return cabs_mol, numpy.vstack((-c1,c2))
def trans(eri,mos):
    s=[m.shape for m in mos]
    e=numpy.dot(mos[0].T,eri.reshape(s[0][0],-1)).reshape(s[0][1],s[1][0],s[2][0],s[3][0])
    e=numpy.dot(mos[1].T,e.transpose(1,0,2,3).reshape(s[1][0],-1)).reshape(s[1][1],s[0][1],s[2][0],s[3][0]).transpose(1,0,2,3)
    e=numpy.dot(e.transpose(0,1,3,2).reshape(-1,s[2][0]),mos[2]).reshape(s[0][1],s[1][1],s[3][0],s[2][1]).transpose(0,1,3,2)
    e=numpy.dot(e.reshape(-1,s[3][0]),mos[3]).reshape(s[0][1],s[1][1],s[2][1],s[3][1])
    return e

mol=gto.Mole(); mol.atom='Ne 0 0 0'; mol.basis='cc-pvdz'; mol.build()
mf=scf.RHF(mol); mf.kernel()
aux=mol.copy(); aux.basis=ri_bas; aux.build(False,False)
zeta=1.0
mo_coeff=mf.mo_coeff; mo_energy=mf.mo_energy; nocc=numpy.count_nonzero(mf.mo_occ==2)
try:
    cabs_mol,cabs_coeff=find_cabs(mol,aux)
    print('CABS ok, nca', cabs_coeff.shape[0])
    nao,nmo=mo_coeff.shape; nca=cabs_coeff.shape[0]; mo_o=mo_coeff[:,:nocc]
    Pcoeff=numpy.vstack((mo_coeff,numpy.zeros((nca-nao,nmo)))); Pcoeff=numpy.hstack((Pcoeff,cabs_coeff))
    obs=(0,mol.nbas); cbs=(0,cabs_mol.nbas)
    mol.set_f12_zeta(zeta); Y=trans(mol.intor('int2e_yp'),[mo_o]*4)
    cabs_mol.set_f12_zeta(zeta)
    R=cabs_mol.intor('int2e_stg',shls_slice=obs+cbs+obs+cbs)
    RmPnQ=trans(R,[mo_o,Pcoeff,mo_o,Pcoeff])
    Rmpnq=RmPnQ[:,:nmo,:,:nmo]; Rmlnc=RmPnQ[:nocc,:nocc,:nocc,nmo:]; Rmcnl=Rmlnc.transpose(2,3,0,1)
    Rpiqj=Rmpnq.transpose(1,0,3,2); Rlicj=Rmlnc.transpose(0,1,3,2); Rcilj=Rlicj.transpose(2,3,0,1)
    cabs_mol.set_f12_zeta(zeta*2)
    Rbar=cabs_mol.intor('int2e_stg',shls_slice=cbs+obs+obs+obs).reshape(nca,nao,nao,nao)
    Rbar_minj=trans(Rbar[:nao],[mo_o]*4)
    v=cabs_mol.intor('int2e',shls_slice=cbs+obs+obs+obs).reshape(nca,nao,nao,nao)
    vpiqj=trans(v[:nao],[mo_coeff,mo_o,mo_coeff,mo_o]); vlicj=trans(v,[cabs_coeff,mo_o,mo_o,mo_o]).transpose(2,3,0,1); vcilj=vlicj.transpose(2,3,0,1)
    tminj=numpy.zeros([nocc]*4)
    for i in range(nocc):
        for j in range(nocc):
            tminj[i,i,j,j]=-3./(8*zeta); tminj[i,j,j,i]=-1./(8*zeta)
        tminj[i,i,i,i]=-.5/zeta
    V=Y.copy(); V-=numpy.einsum('mpnq,piqj->minj',Rmpnq,vpiqj); V-=numpy.einsum('mlnc,licj->minj',Rmlnc,vlicj); V-=numpy.einsum('mcnl,cilj->minj',Rmcnl,vcilj)
    eV=numpy.einsum('minj,minj',V,tminj)*4 - numpy.einsum('minj,nimj',V,tminj)*2
    X=Rbar_minj.copy(); X-=numpy.einsum('mpnq,piqj->minj',Rmpnq,Rpiqj); X-=numpy.einsum('mlnc,licj->minj',Rmlnc,Rlicj); X-=numpy.einsum('mcnl,cilj->minj',Rmcnl,Rcilj)
    e_mn=lib.direct_sum('i+j->ij',mo_energy[:nocc],mo_energy[:nocc])
    tmp=numpy.einsum('mknl,kilj->minj',tminj,X)
    eX=-numpy.einsum('mn,minj,minj',e_mn,tmp,tminj)*2 + numpy.einsum('mn,minj,nimj',e_mn,tmp,tminj)
    print('NE_RHF', mf.e_tot); 
    print('NE_dims nocc',nocc,'nmo',nmo,'nca',nca)
    print('NE_F12_V', eV); print('NE_F12_X', eX); print('NE_F12_VX', eV+eX)
except Exception:
    traceback.print_exc()
