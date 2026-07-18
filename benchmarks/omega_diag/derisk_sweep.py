#!/usr/bin/env python3
"""De-risk sweep: ω × {MP2, SR-MP2(erfc), naive-A, B, T} on anchor dimers,
both aDZ and aTZ, in the CORRECT convention (NON-CP + frozen core).
Naive-A and SR-MP2 come free in every delta-lr (B) run's output; T is a 2nd run.
MP2(full) is the E(MP2,Coulomb) line, also free. So per (system,basis,ω) we run
2 ferric jobs per fragment (B=delta-lr, T=coupled-rings) × 3 fragments = 6 jobs."""
import os, subprocess, itertools, re, json
ROOT="/home/matt/qc/ferric/.claude/worktrees/sr-mp2-lr-rpa"
os.chdir(ROOT)
OUT="benchmarks/omega_diag/derisk"; os.makedirs(OUT+"/toml",exist_ok=True); os.makedirs(OUT+"/out",exist_ok=True)
GEO="benchmarks/grid/geoms"
BIN="target/release/ferric-cli"
env=dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1")

# anchors: (s22 id, label, heavy-atom split point in dimer xyz)
ANCHORS=[("01","ammonia_HB"),("02","water_HB"),("08","methane_D"),
         ("09","ethene_D"),("11","benzene_PD")]
BASES=[("aug-cc-pvdz","aug-cc-pvdz-rifit","adz"),  # aTZ gated below
       ] + ([("aug-cc-pvtz","aug-cc-pvtz-rifit","atz")] if os.environ.get("DERISK_ATZ") else [])
OMEGAS=[0.30,0.42,0.55,0.673,0.80]
FORMS=[("delta-lr","B"),("coupled-rings","T")]

def fc_count(xyz):  # frozen core = # non-H atoms (1s each), Li-Ne→1
    n=0
    for ln in open(xyz).read().splitlines()[2:]:
        if ln.strip() and not ln.split()[0].lstrip('@').upper().startswith('H'): n+=1
    return n

def split_monomers(sid):
    """Write NON-CP plain monomers from the dimer (heavy fragment A then B)."""
    lines=open(f"{GEO}/s22-{sid}_dimer.xyz").read().splitlines()
    nat=int(lines[0]); atoms=lines[2:2+nat]
    # use the CP files' real-atom partition to know the split (real vs @ghost)
    mAcp=open(f"{GEO}/s22-{sid}_mA_cp.xyz").read().splitlines()[2:2+nat]
    A=[atoms[i] for i,l in enumerate(mAcp) if not l.split()[0].startswith('@')]
    B=[atoms[i] for i,l in enumerate(mAcp) if l.split()[0].startswith('@')]
    pa=f"{OUT}/s22-{sid}_A.xyz"; pb=f"{OUT}/s22-{sid}_B.xyz"
    open(pa,'w').write(f"{len(A)}\nA\n"+"\n".join(A)+"\n")
    open(pb,'w').write(f"{len(B)}\nB\n"+"\n".join(B)+"\n")
    return f"{GEO}/s22-{sid}_dimer.xyz", pa, pb

def run(xyz, basis, aux, omega, form, fc, key):
    t=f"""[molecule]
xyz = "{ROOT}/{xyz}" if not "{xyz}".startswith("/") else "{xyz}"
"""
    # simpler: absolute path
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
    tp=f"{OUT}/toml/{key}.toml"; op=f"{OUT}/out/{key}.out"
    if os.path.exists(op) and "Total energy" in open(op).read(): return  # idempotent skip
    open(tp,'w').write(toml)
    with open(op,'w') as f, open(op+".err",'w') as e:
        subprocess.run([BIN,tp],stdout=f,stderr=e,env=env,timeout=7200)

for sid,label in ANCHORS:
    dim,pa,pb=split_monomers(sid)
    frags={"dimer":dim,"A":pa,"B":pb}
    fcs={k:fc_count(v) for k,v in frags.items()}
    for (basis,aux,btag) in BASES:
        # skip aTZ for the largest if you want; keep all for now
        for omega in OMEGAS:
            for form,ftag in FORMS:
                for fr,xyz in frags.items():
                    key=f"{label}_{sid}_{btag}_w{omega}_{ftag}_{fr}"
                    print(f"[run] {key} (fc={fcs[fr]})",flush=True)
                    run(xyz,basis,aux,omega,form,fcs[fr],key)
print("DERISK SWEEP DONE",flush=True)
