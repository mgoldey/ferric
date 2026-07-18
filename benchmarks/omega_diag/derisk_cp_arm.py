#!/usr/bin/env python3
"""CP arm of the de-risk sweep — ADDITIVE, writes only NEW ghost-monomer outputs.

The non-CP sweep (derisk_sweep.py) already computed the DIMER and the plain
(non-CP) monomers. CP binding shares the SAME dimer; only the monomers differ
(ghost-augmented `_cp` files). So this script runs ONLY the ghost monomers,
under distinct keys ..._cpA / ..._cpB, and DELETES NOTHING / OVERWRITES NOTHING
(idempotent skip on existing complete outputs).

CP binding  = E(dimer) - E(ghost-A) - E(ghost-B)   [BSSE-corrected]
non-CP bind = E(dimer) - E(plain-A) - E(plain-B)    [already in the other sweep]
Both reuse the identical dimer output key  {label}_{sid}_{btag}_w{omega}_{ftag}_dimer.
"""
import os, subprocess

ROOT="/home/matt/qc/ferric/.claude/worktrees/sr-mp2-lr-rpa"
os.chdir(ROOT)
OUT="benchmarks/omega_diag/derisk"; os.makedirs(OUT+"/toml",exist_ok=True); os.makedirs(OUT+"/out",exist_ok=True)
GEO="benchmarks/grid/geoms"
BIN="target/release/ferric-cli"
env=dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1")

ANCHORS=[("01","ammonia_HB"),("02","water_HB"),("08","methane_D"),
         ("09","ethene_D"),("11","benzene_PD")]
BASES=[("aug-cc-pvdz","aug-cc-pvdz-rifit","adz")]  # aTZ gated separately
if os.environ.get("DERISK_ATZ"):
    BASES.append(("aug-cc-pvtz","aug-cc-pvtz-rifit","atz"))
OMEGAS=[0.30,0.42,0.55,0.673,0.80]
FORMS=[("delta-lr","B"),("coupled-rings","T")]

def fc_count(xyz):
    """Frozen core = # of REAL (non-ghost, non-H) atoms. Ghosts (@) have zero
    electrons → contribute zero frozen core."""
    n=0
    for ln in open(xyz).read().splitlines()[2:]:
        if not ln.strip(): continue
        sym=ln.split()[0]
        if sym.startswith('@'): continue           # ghost: no electrons
        if sym.upper().startswith('H'): continue    # H: no core
        n+=1
    return n

def run(xyz, basis, aux, omega, form, fc, key):
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
    if os.path.exists(op) and "Total energy" in open(op).read():
        return  # idempotent: never clobber a completed run
    open(tp,'w').write(toml)
    with open(op,'w') as f, open(op+".err",'w') as e:
        subprocess.run([BIN,tp],stdout=f,stderr=e,env=env,timeout=7200)

for sid,label in ANCHORS:
    # CP ghost monomers (real A + ghost B  /  ghost A + real B)
    ghostA=f"{GEO}/s22-{sid}_mA_cp.xyz"
    ghostB=f"{GEO}/s22-{sid}_mB_cp.xyz"
    cp_frags={"cpA":ghostA,"cpB":ghostB}
    fcs={k:fc_count(v) for k,v in cp_frags.items()}
    for (basis,aux,btag) in BASES:
        for omega in OMEGAS:
            for form,ftag in FORMS:
                for fr,xyz in cp_frags.items():
                    key=f"{label}_{sid}_{btag}_w{omega}_{ftag}_{fr}"
                    print(f"[cp-run] {key} (fc={fcs[fr]})",flush=True)
                    run(xyz,basis,aux,omega,form,fcs[fr],key)
print("DERISK CP-ARM DONE",flush=True)
