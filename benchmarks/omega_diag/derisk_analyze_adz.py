#!/usr/bin/env python3
"""Standalone Stage-4 analysis for the aDZ arm ONLY (the overnight driver died in
the aTZ stage before reaching analysis). aDZ is 100% complete: dimer + plain
monomers (_A/_B) + ghost monomers (_cpA/_cpB) for 5 anchors × 5 ω × {B,T}.

ADDITIVE: reads existing derisk/out/*.out, runs missing per-fragment RHF (scs-mp2)
only if absent, writes derisk_results_adz.json + DERISK_REPORT_adz.md. Deletes
nothing, never touches aTZ.
"""
import os, re, json, subprocess

ROOT="/home/matt/qc/ferric/.claude/worktrees/sr-mp2-lr-rpa"
os.chdir(ROOT)
OUT="benchmarks/omega_diag/derisk"; GEO="benchmarks/grid/geoms"; BIN="target/release/ferric-cli"
env=dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1")
KCAL=627.509474
ANCHORS=[("01","ammonia_HB"),("02","water_HB"),("08","methane_D"),("09","ethene_D"),("11","benzene_PD")]
OMEGAS=[0.30,0.42,0.55,0.673,0.80]
CCSDT={"01":-3.13,"02":-4.99,"08":-0.53,"09":-1.47,"11":-2.65}
ADZ=[("aug-cc-pvdz","aug-cc-pvdz-rifit","adz")]

def fc_count(xyz):
    n=0
    for ln in open(xyz).read().splitlines()[2:]:
        if not ln.strip(): continue
        s=ln.split()[0]
        if s.startswith('@') or s.upper().startswith('H'): continue
        n+=1
    return n

def split_monomers(sid):
    lines=open(f"{GEO}/s22-{sid}_dimer.xyz").read().splitlines()
    nat=int(lines[0]); atoms=lines[2:2+nat]
    mAcp=open(f"{GEO}/s22-{sid}_mA_cp.xyz").read().splitlines()[2:2+nat]
    A=[atoms[i] for i,l in enumerate(mAcp) if not l.split()[0].startswith('@')]
    B=[atoms[i] for i,l in enumerate(mAcp) if l.split()[0].startswith('@')]
    pa=f"{OUT}/s22-{sid}_A.xyz"; pb=f"{OUT}/s22-{sid}_B.xyz"
    if not os.path.exists(pa): open(pa,'w').write(f"{len(A)}\nA\n"+"\n".join(A)+"\n")
    if not os.path.exists(pb): open(pb,'w').write(f"{len(B)}\nB\n"+"\n".join(B)+"\n")
    return f"{GEO}/s22-{sid}_dimer.xyz", pa, pb

def grab(key,pat):
    p=f"{OUT}/out/{key}.out"
    if not os.path.exists(p): return None
    m=re.search(pat,open(p).read())
    return float(m.group(1)) if m else None

def rhf_energy(xyz, basis, aux, fc, key):
    op=f"{OUT}/out/{key}.out"
    if os.path.exists(op):
        m=re.search(r'RHF energy\s*=\s*(-?[0-9.]+)',open(op).read())
        if m: return float(m.group(1))
    xyzabs = xyz if xyz.startswith("/") else f"{ROOT}/{xyz}"
    toml=f"""[molecule]
xyz = "{xyzabs}"
[basis]
name="{basis}"
[scf]
df_j_aux="def2-universal-jkfit"
df_k_aux="def2-universal-jkfit"
max_iter=400
[method]
kind="scs-mp2"
[mp2]
auxbasis="{aux}"
frozen_core={fc}
"""
    open(f"{OUT}/toml/{key}.toml",'w').write(toml)
    with open(op,'w') as f, open(op+".err",'w') as e:
        subprocess.run([BIN,f"{OUT}/toml/{key}.toml"],stdout=f,stderr=e,env=env,timeout=10800)
    m=re.search(r'RHF energy\s*=\s*(-?[0-9.]+)',open(op).read())
    return float(m.group(1)) if m else None

pats={
  "MP2": r'E\(MP2, Coulomb\)\s*=\s*(-?[0-9.]+)',
  "SRMP2": r'E\(SR-MP2, erfc\)\s*=\s*(-?[0-9.]+)',
  "naiveA": r'E_corr naive \(A\)\s*=\s*(-?[0-9.]+)',
}
results={}
for basis,aux,btag in ADZ:
    for sid,label in ANCHORS:
        dim,pa,pb=split_monomers(sid)
        rhf_d =rhf_energy(dim,basis,aux,fc_count(dim),f"{label}_{sid}_{btag}_RHF_dimer")
        rhf_pa=rhf_energy(pa,basis,aux,fc_count(pa),f"{label}_{sid}_{btag}_RHF_A")
        rhf_pb=rhf_energy(pb,basis,aux,fc_count(pb),f"{label}_{sid}_{btag}_RHF_B")
        cpA=f"{GEO}/s22-{sid}_mA_cp.xyz"; cpB=f"{GEO}/s22-{sid}_mB_cp.xyz"
        rhf_ca=rhf_energy(cpA,basis,aux,fc_count(cpA),f"{label}_{sid}_{btag}_RHF_cpA")
        rhf_cb=rhf_energy(cpB,basis,aux,fc_count(cpB),f"{label}_{sid}_{btag}_RHF_cpB")
        for omega in OMEGAS:
            kB=f"{label}_{sid}_{btag}_w{omega}_B"; kT=f"{label}_{sid}_{btag}_w{omega}_T"
            def piece(fr,pat): return grab(f"{kB}_{fr}",pat)
            def bind(method, arm):
                if arm=="noncp": rd,ra,rb,fa,fb=rhf_d,rhf_pa,rhf_pb,"A","B"
                else:            rd,ra,rb,fa,fb=rhf_d,rhf_ca,rhf_cb,"cpA","cpB"
                if None in (rd,ra,rb): return None
                if method in pats:
                    cd=piece("dimer",pats[method]); ca=piece(fa,pats[method]); cb=piece(fb,pats[method])
                    if None in (cd,ca,cb): return None
                    return ((rd+cd)-(ra+ca)-(rb+cb))*KCAL
                if method=="B":
                    td=grab(f"{kB}_dimer",r'Total energy\s*=\s*(-?[0-9.]+)')
                    ta=grab(f"{kB}_{fa}",r'Total energy\s*=\s*(-?[0-9.]+)')
                    tb=grab(f"{kB}_{fb}",r'Total energy\s*=\s*(-?[0-9.]+)')
                    if None in (td,ta,tb): return None
                    return (td-ta-tb)*KCAL
                if method=="T":
                    td=grab(f"{kT}_dimer",r'Total energy\s*=\s*(-?[0-9.]+)')
                    ta=grab(f"{kT}_{fa}",r'Total energy\s*=\s*(-?[0-9.]+)')
                    tb=grab(f"{kT}_{fb}",r'Total energy\s*=\s*(-?[0-9.]+)')
                    if None in (td,ta,tb): return None
                    return (td-ta-tb)*KCAL
            for method in ["MP2","SRMP2","naiveA","B","T"]:
                for arm in ["noncp","cp"]:
                    results[f"{label}|{sid}|{btag}|{omega}|{method}|{arm}"]=bind(method,arm)

json.dump({"binds":results,"ccsdt":CCSDT},open(f"{OUT}/derisk_results_adz.json","w"),indent=1)

# Per-method MAE/MSE across the 5 anchors at each omega, both arms -> error-cancellation view.
def stats(method, arm, omega):
    errs=[]
    for sid,label in ANCHORS:
        v=results.get(f"{label}|{sid}|adz|{omega}|{method}|{arm}")
        if v is not None: errs.append(v-CCSDT[sid])
    if not errs: return None
    mae=sum(abs(e) for e in errs)/len(errs); mse=sum(errs)/len(errs)
    return mae,mse,len(errs)

L=["# De-risk: CP vs non-CP × method × interaction-type — aDZ\n",
   "Binding kcal/mol; err vs CCSD(T)/CBS (dissertation Table A.4). Negative err = overbinding.\n",
   "aTZ arm incomplete (driver died in stage 3) — aDZ analyzed standalone.\n"]
for sid,label in ANCHORS:
    ref=CCSDT[sid]
    L.append(f"\n### {label} (S22 #{sid}) ref={ref:+.2f}\n")
    L.append("| ω | method | non-CP | err | CP | err |"); L.append("|---|---|---|---|---|---|")
    for omega in OMEGAS:
        for method in ["MP2","SRMP2","naiveA","B","T"]:
            n=results.get(f"{label}|{sid}|adz|{omega}|{method}|noncp")
            c=results.get(f"{label}|{sid}|adz|{omega}|{method}|cp")
            ns=f"{n:+.3f}" if n is not None else "—"; cs=f"{c:+.3f}" if c is not None else "—"
            ne=f"{n-ref:+.3f}" if n is not None else "—"; ce=f"{c-ref:+.3f}" if c is not None else "—"
            L.append(f"| {omega} | {method} | {ns} | {ne} | {cs} | {ce} |")

L.append("\n## Summary: MAE / MSE over 5 anchors (kcal/mol), per method × arm × ω\n")
L.append("MSE<0 = net overbinding. Compare non-CP vs CP MAE to see if CP helps or if error cancels.\n")
L.append("| ω | method | MAE noncp | MSE noncp | MAE CP | MSE CP |")
L.append("|---|---|---|---|---|---|")
for omega in OMEGAS:
    for method in ["MP2","SRMP2","naiveA","B","T"]:
        sn=stats(method,"noncp",omega); sc=stats(method,"cp",omega)
        def f(s): return (f"{s[0]:.3f}",f"{s[1]:+.3f}") if s else ("—","—")
        an=f(sn); ac=f(sc)
        L.append(f"| {omega} | {method} | {an[0]} | {an[1]} | {ac[0]} | {ac[1]} |")
open(f"{OUT}/DERISK_REPORT_adz.md","w").write("\n".join(L))
nvals=sum(1 for v in results.values() if v is not None)
print(f"wrote DERISK_REPORT_adz.md + derisk_results_adz.json ({nvals}/{len(results)} binds)")
