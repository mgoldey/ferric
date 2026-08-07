#!/usr/bin/env python3
"""Overnight autonomous de-risk driver. Runs all remaining stages SEQUENTIALLY
(never oversubscribing), then writes a CP-vs-non-CP analysis report.

Stages:
  1. Wait for the running non-CP aDZ sweep to finish (don't double-run it).
  2. Run the CP arm (aDZ): ghost monomers only (dimers shared with non-CP).
  3. Run aTZ for BOTH non-CP and CP (gated; smaller systems will finish first).
  4. Analyze: CP and non-CP binding for MP2 / SR-MP2 / naive-A / B / T vs CCSD(T),
     per anchor, per basis, per omega. Write markdown report + json.

DELETES NOTHING. All runs idempotent (skip on existing complete output).
Single-thread (OPENBLAS=1, RAYON=1) so it coexists with the production grid.
"""
import os, sys, time, subprocess, re, json, glob

ROOT="/home/matt/qc/ferric"
os.chdir(ROOT)
OUT="benchmarks/omega_diag/derisk"
GEO="benchmarks/grid/geoms"
BIN="target/release/ferric-cli"
env=dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1")
LOG=open(f"{OUT}/overnight.log","a")
def log(m):
    LOG.write(f"[{time.strftime('%H:%M:%S')}] {m}\n"); LOG.flush()

ANCHORS=[("01","ammonia_HB"),("02","water_HB"),("08","methane_D"),
         ("09","ethene_D"),("11","benzene_PD")]
OMEGAS=[0.30,0.42,0.55,0.673,0.80]
FORMS=[("delta-lr","B"),("coupled-rings","T")]
# CCSD(T)/CBS refs from dissertation Table A.4 (PDF p89), S22 system numbers.
CCSDT={"01":-3.13,"02":-4.99,"08":-0.53,"09":-1.47,"11":-2.65}

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

def run(xyz, basis, aux, omega, form, fc, key):
    op=f"{OUT}/out/{key}.out"
    if os.path.exists(op) and "Total energy" in open(op).read():
        return  # idempotent
    xyzabs = xyz if xyz.startswith("/") else f"{ROOT}/{xyz}"
    toml=f"""[molecule]
xyz = "{xyzabs}"
[basis]
name = "{basis}"
[scf]
df_j_aux = "def2-universal-jkfit"
df_k_aux = "def2-universal-jkfit"
max_iter = 400
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "{aux}"
omega = {omega}
formulation = "{form}"
frozen_core = {fc}
[rpa]
trunc_thresh = 0.0
[quadrature]
n_points = 12
"""
    tp=f"{OUT}/toml/{key}.toml"
    open(tp,'w').write(toml)
    log(f"run {key} (fc={fc})")
    with open(op,'w') as f, open(op+".err",'w') as e:
        try:
            subprocess.run([BIN,tp],stdout=f,stderr=e,env=env,timeout=10800)
        except subprocess.TimeoutExpired:
            log(f"TIMEOUT {key}")

def sweep(bases, frag_set):
    """frag_set: 'noncp' -> dimer+plain A/B ; 'cp' -> ghost cpA/cpB only."""
    for sid,label in ANCHORS:
        dim,pa,pb=split_monomers(sid)
        if frag_set=="noncp":
            frags={"dimer":dim,"A":pa,"B":pb}
        else:
            frags={"cpA":f"{GEO}/s22-{sid}_mA_cp.xyz","cpB":f"{GEO}/s22-{sid}_mB_cp.xyz"}
        fcs={k:fc_count(v) for k,v in frags.items()}
        for (basis,aux,btag) in bases:
            for omega in OMEGAS:
                for form,ftag in FORMS:
                    for fr,xyz in frags.items():
                        run(xyz,basis,aux,omega,form,fcs[fr],
                            f"{label}_{sid}_{btag}_w{omega}_{ftag}_{fr}")

# ---- Stage 1: wait for the existing non-CP aDZ sweep ----
log("=== overnight driver start ===")
waited=0
while True:
    r=subprocess.run(["pgrep","-f","derisk_sweep.py"],capture_output=True)
    if r.returncode!=0: break
    time.sleep(30); waited+=30
    if waited>3600: log("non-CP sweep >1h, proceeding anyway"); break
log("non-CP aDZ sweep clear")

# ---- Stage 2: CP arm, aDZ ----
ADZ=[("aug-cc-pvdz","aug-cc-pvdz-rifit","adz")]
log("stage 2: CP arm aDZ")
sweep(ADZ,"cp")
log("stage 2 done")

# ---- Stage 3: aTZ both arms (smaller systems finish first; benzene aTZ is slow) ----
ATZ=[("aug-cc-pvtz","aug-cc-pvtz-rifit","atz")]
log("stage 3: aTZ non-CP")
sweep(ATZ,"noncp")
log("stage 3a done; aTZ CP")
sweep(ATZ,"cp")
log("stage 3 done")

# ---- Stage 4: analyze ----
KCAL=627.509474
def grab(key,pat):
    p=f"{OUT}/out/{key}.out"
    if not os.path.exists(p): return None
    m=re.search(pat,open(p).read())
    return float(m.group(1)) if m else None

# Per-method total-energy extractor. For a given (sid,basis,omega) we need:
#  MP2(full): E(MP2,Coulomb)+RHF ; SR-MP2: E(SR-MP2,erfc)+RHF ; naive-A: E_corr naive(A)+RHF ;
#  B: Total energy (delta-lr) ; T: Total energy (coupled-rings).
# RHF isn't printed by rs-mp2-rpa; reconstruct RHF = Total(B) - E_corr(B). We have B's
# Total and B's corr = (we can get B e_corr as Total - RHF... circular). Instead: for the
# correlation-difference methods, bind via TOTAL energies directly where printed (B,T),
# and for MP2/SR-MP2/naive reconstruct total = RHF + corr where RHF = Total_B - corr_B.
# corr_B = E_corr Δ-form: not printed as a single line, but Total_B - RHF. We DO have
# E(MP2,Coulomb), E(SR-MP2,erfc), naive(A) as correlation lines, plus Total_B.
# => RHF = Total_B - corr_B, and corr_B is NOT directly available. Simplest robust path:
# get RHF from a one-shot scs run per fragment (omega-independent). Do that here.
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

log("stage 4: analysis")
results={}
for basis,aux,btag in ADZ+ATZ:
    for sid,label in ANCHORS:
        dim,pa,pb=split_monomers(sid)
        # RHF per fragment (omega-independent), both non-CP and CP monomers
        fc_d=fc_count(dim)
        rhf_d=rhf_energy(dim,basis,aux,fc_d,f"{label}_{sid}_{btag}_RHF_dimer")
        rhf_pa=rhf_energy(pa,basis,aux,fc_count(pa),f"{label}_{sid}_{btag}_RHF_A")
        rhf_pb=rhf_energy(pb,basis,aux,fc_count(pb),f"{label}_{sid}_{btag}_RHF_B")
        rhf_ca=rhf_energy(f"{GEO}/s22-{sid}_mA_cp.xyz",basis,aux,fc_count(f"{GEO}/s22-{sid}_mA_cp.xyz"),f"{label}_{sid}_{btag}_RHF_cpA")
        rhf_cb=rhf_energy(f"{GEO}/s22-{sid}_mB_cp.xyz",basis,aux,fc_count(f"{GEO}/s22-{sid}_mB_cp.xyz"),f"{label}_{sid}_{btag}_RHF_cpB")
        for omega in OMEGAS:
            kB=f"{label}_{sid}_{btag}_w{omega}_B"
            kT=f"{label}_{sid}_{btag}_w{omega}_T"
            def corr(frag_key, pat): return grab(frag_key,pat)
            # correlation pieces from the B (delta-lr) run of each fragment
            def piece(fr,pat): return grab(f"{kB}_{fr}",pat)
            pats={
              "MP2": r'E\(MP2, Coulomb\)\s*=\s*(-?[0-9.]+)',
              "SRMP2": r'E\(SR-MP2, erfc\)\s*=\s*(-?[0-9.]+)',
              "naiveA": r'E_corr naive \(A\)\s*=\s*(-?[0-9.]+)',
            }
            def bind(method, arm):
                if arm=="noncp": rd,ra,rb=rhf_d,rhf_pa,rhf_pb; fa,fb="A","B"
                else:           rd,ra,rb=rhf_d,rhf_ca,rhf_cb; fa,fb="cpA","cpB"
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
                    v=bind(method,arm)
                    results[f"{label}|{sid}|{btag}|{omega}|{method}|{arm}"]=v

json.dump({"binds":results,"ccsdt":CCSDT},open(f"{OUT}/derisk_results.json","w"),indent=1)
log(f"wrote {OUT}/derisk_results.json with {sum(1 for v in results.values() if v is not None)} values")

# Markdown report
lines=["# De-risk: CP vs non-CP × method × interaction-type (aDZ, aTZ)\n",
       "Convention check for SR-MP2+LR-RPA. Binding kcal/mol; err vs CCSD(T)/CBS (dissertation Table A.4).\n"]
for basis,aux,btag in ADZ+ATZ:
    lines.append(f"\n## {btag.upper()}\n")
    for sid,label in ANCHORS:
        ref=CCSDT[sid]
        lines.append(f"\n### {label} (S22 #{sid}) ref={ref:+.2f}\n")
        lines.append("| ω | method | non-CP | err | CP | err |")
        lines.append("|---|---|---|---|---|---|")
        for omega in OMEGAS:
            for method in ["MP2","SRMP2","naiveA","B","T"]:
                n=results.get(f"{label}|{sid}|{btag}|{omega}|{method}|noncp")
                c=results.get(f"{label}|{sid}|{btag}|{omega}|{method}|cp")
                ns=f"{n:+.3f}" if n is not None else "—"
                cs=f"{c:+.3f}" if c is not None else "—"
                ne=f"{n-ref:+.3f}" if n is not None else "—"
                ce=f"{c-ref:+.3f}" if c is not None else "—"
                lines.append(f"| {omega} | {method} | {ns} | {ne} | {cs} | {ce} |")
open(f"{OUT}/DERISK_REPORT.md","w").write("\n".join(lines))
log("wrote DERISK_REPORT.md")
log("=== overnight driver DONE ===")
print("OVERNIGHT DRIVER DONE")
