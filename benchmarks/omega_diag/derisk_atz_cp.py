#!/usr/bin/env python3
"""aTZ CP ω-selection for B/T. Reuses existing aTZ dimers, runs ONLY the missing
aTZ CP ghost monomers + per-fragment RHF, then writes the CP-arm ω-curve.

ADDITIVE / idempotent: skips any output already carrying its completion marker,
never overwrites, never touches non-CP or aDZ. aTZ benzene dimers are expensive and
mostly absent — analysis uses whatever dimers ARE present (n reported per ω).
Single-thread (OPENBLAS=1, RAYON=1).
"""
from pathlib import Path
import os, re, json, subprocess

ROOT = str(Path(__file__).resolve().parents[2])
os.chdir(ROOT)
OUT="benchmarks/omega_diag/derisk"; GEO="benchmarks/grid/geoms"; BIN="target/release/ferric-cli"
env=dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1")
KCAL=627.509474
ANCHORS=[("01","ammonia_HB"),("02","water_HB"),("08","methane_D"),("09","ethene_D"),("11","benzene_PD")]
LAB={sid:lab for sid,lab in ANCHORS}
OMEGAS=[0.30,0.42,0.55,0.673,0.80]
FORMS=[("delta-lr","B"),("coupled-rings","T")]
CCSDT={"01":-3.13,"02":-4.99,"08":-0.53,"09":-1.47,"11":-2.65}
BASIS,AUX,BT=("aug-cc-pvtz","aug-cc-pvtz-rifit","atz")
TOT=r'Total energy\s*=\s*(-?[0-9.]+)'
RHFPAT=r'RHF energy\s*=\s*(-?[0-9.]+)'
PIECE={"MP2":r'E\(MP2, Coulomb\)\s*=\s*(-?[0-9.]+)',
       "SRMP2":r'E\(SR-MP2, erfc\)\s*=\s*(-?[0-9.]+)',
       "naiveA":r'E_corr naive \(A\)\s*=\s*(-?[0-9.]+)'}

def fc_count(xyz):
    n=0
    for ln in open(xyz).read().splitlines()[2:]:
        if not ln.strip(): continue
        s=ln.split()[0]
        if s.startswith('@') or s.upper().startswith('H'): continue
        n+=1
    return n

def out(key): return f"{OUT}/out/{key}.out"
def grab(p,pat):
    if not os.path.exists(p): return None
    m=re.search(pat,open(p).read()); return float(m.group(1)) if m else None

def write_run(toml, key, marker, timeout=14400):
    op=out(key)
    if os.path.exists(op) and marker in open(op).read(): return
    open(f"{OUT}/toml/{key}.toml",'w').write(toml)
    print(f"[run] {key}",flush=True)
    with open(op,'w') as f, open(op+".err",'w') as e:
        subprocess.run([BIN,f"{OUT}/toml/{key}.toml"],stdout=f,stderr=e,env=env,timeout=timeout)

def rsmp2_toml(xyz, omega, form, fc):
    return f"""[molecule]
xyz = "{xyz}"
[basis]
name = "{BASIS}"
[scf]
df_j_aux = "def2-universal-jkfit"
df_k_aux = "def2-universal-jkfit"
max_iter = 400
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "{AUX}"
omega = {omega}
formulation = "{form}"
frozen_core = {fc}
[rpa]
trunc_thresh = 0.0
[quadrature]
n_points = 12
"""
def scf_toml(xyz, fc):
    return f"""[molecule]
xyz = "{xyz}"
[basis]
name="{BASIS}"
[scf]
df_j_aux="def2-universal-jkfit"
df_k_aux="def2-universal-jkfit"
max_iter=400
[method]
kind="scs-mp2"
[mp2]
auxbasis="{AUX}"
frozen_core={fc}
"""

def absxyz(p): return p if p.startswith("/") else f"{ROOT}/{p}"

# ---- 1) run missing aTZ rs-mp2-rpa jobs: dimer + both CP ghost monomers ----
#        (small anchors' dimers already exist & skip; benzene dimers are generated here)
#        Order: dimers FIRST per anchor (so binding can land as monomers follow), small
#        anchors before benzene (cheap trend first, benzene as the slow tail).
for sid,label in [a for a in ANCHORS if a[0]!="11"]+[a for a in ANCHORS if a[0]=="11"]:
    frags={"dimer":f"{GEO}/s22-{sid}_dimer.xyz",
           "cpA":f"{GEO}/s22-{sid}_mA_cp.xyz",
           "cpB":f"{GEO}/s22-{sid}_mB_cp.xyz"}
    for fr,xyz in frags.items():
        fc=fc_count(xyz)
        for omega in OMEGAS:
            for form,ftag in FORMS:
                write_run(rsmp2_toml(absxyz(xyz),omega,form,fc),
                          f"{label}_{sid}_{BT}_w{omega}_{ftag}_{fr}", "Total energy")

# ---- 2) RHF per fragment (omega-independent): dimer + cpA + cpB ----
def rhf(xyz, fc, key):
    v=grab(out(key),RHFPAT)
    if v is not None: return v
    write_run(scf_toml(absxyz(xyz),fc), key, "RHF energy")
    return grab(out(key),RHFPAT)

# ---- 3) analysis ----
results={}
for sid,label in ANCHORS:
    dimx=f"{GEO}/s22-{sid}_dimer.xyz"
    cpAx=f"{GEO}/s22-{sid}_mA_cp.xyz"; cpBx=f"{GEO}/s22-{sid}_mB_cp.xyz"
    rd=rhf(dimx,fc_count(dimx),f"{label}_{sid}_{BT}_RHF_dimer")
    rA=rhf(cpAx,fc_count(cpAx),f"{label}_{sid}_{BT}_RHF_cpA")
    rB=rhf(cpBx,fc_count(cpBx),f"{label}_{sid}_{BT}_RHF_cpB")
    for omega in OMEGAS:
        kB=f"{label}_{sid}_{BT}_w{omega}_B"; kT=f"{label}_{sid}_{BT}_w{omega}_T"
        for method in ["MP2","SRMP2","naiveA","B","T"]:
            v=None
            if method in PIECE:
                cd=grab(out(f"{kB}_dimer"),PIECE[method])
                ca=grab(out(f"{kB}_cpA"),PIECE[method]); cb=grab(out(f"{kB}_cpB"),PIECE[method])
                if None not in (rd,rA,rB,cd,ca,cb):
                    v=((rd+cd)-(rA+ca)-(rB+cb))*KCAL
            else:
                k=kB if method=="B" else kT
                td=grab(out(f"{k}_dimer"),TOT); ta=grab(out(f"{k}_cpA"),TOT); tb=grab(out(f"{k}_cpB"),TOT)
                if None not in (td,ta,tb): v=(td-ta-tb)*KCAL
            results[f"{label}|{sid}|{BT}|{omega}|{method}|cp"]=v

json.dump({"binds":results,"ccsdt":CCSDT},open(f"{OUT}/derisk_atz_cp.json","w"),indent=1)

def curve(method):
    L=[f"\n### {method} — aTZ CP MAE/MSE over available anchors\n",
       "| ω | n | MAE | MSE | bz_err |","|---|---|---|---|---|"]
    for omega in OMEGAS:
        errs=[]; bz=None
        for sid,lab in ANCHORS:
            v=results.get(f"{lab}|{sid}|{BT}|{omega}|{method}|cp")
            if v is not None:
                e=v-CCSDT[sid]; errs.append(e)
                if sid=="11": bz=e
        if errs:
            mae=sum(abs(e) for e in errs)/len(errs); mse=sum(errs)/len(errs)
            L.append(f"| {omega} | {len(errs)} | {mae:.3f} | {mse:+.3f} | {bz:+.3f if bz is not None else 0} |".replace("+0.000 if bz is not None else 0","—"))
            # simpler bz formatting:
            L[-1]=f"| {omega} | {len(errs)} | {mae:.3f} | {mse:+.3f} | {(f'{bz:+.3f}' if bz is not None else '—')} |"
        else:
            L.append(f"| {omega} | 0 | — | — | — |")
    return L

R=["# aTZ CP ω-selection for SR-MP2+LR-RPA (B,T)\n",
   "Binding kcal/mol vs CCSD(T)/CBS. CP arm only (correct convention for B/T).\n",
   "aTZ benzene dimers may be incomplete; MAE 'n' = anchors with a dimer present.\n"]
for m in ["B","T","naiveA","SRMP2","MP2"]:
    R+=curve(m)
open(f"{OUT}/DERISK_ATZ_CP.md","w").write("\n".join(R))
nv=sum(1 for v in results.values() if v is not None)
print(f"wrote DERISK_ATZ_CP.md + derisk_atz_cp.json ({nv}/{len(results)} binds)")
