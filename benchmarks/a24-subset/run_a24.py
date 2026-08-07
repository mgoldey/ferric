#!/usr/bin/env python3
"""A24-subset decisive experiment for rs-mp2-rpa (spec 2026-06-09).
Geometries + CCSD(T)/CBS refs: psi4 A24.py (Rezac & Hobza A24 set).
Systems: 2 water-water, 5 ammonia-ammonia, 14 ethene-ethene C2v, 19 methane-methane D3d.
For each system x omega: dimer, monoA, monoB, monoA+ghostB, monoB+ghostA via ferric CLI.
"""
import re, subprocess, os, sys, json

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "release", "ferric-cli")
SRC = open("/tmp/a24/A24.py").read() if os.path.exists("/tmp/a24/A24.py") else open("/tmp/wd/A24.py").read()
REFS = {2: -5.014, 5: -3.157, 14: -1.110, 19: -0.538}
NAMES = {2: "H2O-H2O", 5: "NH3-NH3", 14: "C2H4-C2H4", 19: "CH4-CH4"}
OMEGAS = ["0.1", "0.2", "0.3", "0.42", "0.6"]
K = 627.509474

def frags(idx):
    m = re.search(r"GEOS\['%s-%s-dimer' % \(dbse, '"+str(idx)+r"'\)\] = qcdb\.Molecule\(\"\"\"(.*?)\"\"\"\)", SRC, re.S)
    body = m.group(1)
    parts = [p.strip() for p in body.split("--")]
    out = []
    for p in parts:
        lines = [l for l in p.splitlines() if l.strip() and not l.startswith(("0 1","units"))]
        atoms = [l.split() for l in lines if re.match(r"\s*[A-Za-z]", l) and "units" not in l]
        atoms = [(a[0], float(a[1]), float(a[2]), float(a[3])) for a in atoms if len(a)==4]
        out.append(atoms)
    assert len(out)==2, f"sys {idx}: {len(out)} frags"
    return out

def wxyz(path, atoms, comment):
    with open(path,"w") as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for s,x,y,z in atoms: f.write(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")

def toml(path, xyz, omega):
    open(path,"w").write(f'[molecule]\nxyz = "{xyz}"\n\n[basis]\nname = "cc-pvdz"\n\n[method]\nkind = "rs-mp2-rpa"\n\n[mp2]\nauxbasis = "cc-pvdz-ri"\nomega = {omega}\n')

def run(tomlpath):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="4")
    r = subprocess.run([BIN, tomlpath], capture_output=True, text=True, env=env, timeout=1800)
    g = lambda pat: float(re.search(pat + r"\s*=\s*(-?\d+\.\d+)", r.stdout).group(1))
    mp2, A, B, tot = g(r"E\(MP2, Coulomb\)"), g(r"E_corr naive \(A\)"), g(r"E_corr Δ-form \(B\)"), g(r"Total energy")
    rhf = tot - B
    return dict(rhf=rhf, mp2=rhf+mp2, A=rhf+A, B=tot)

ghost = lambda atoms: [("@"+s,x,y,z) for s,x,y,z in atoms]
results = {}
for idx in REFS:
    fA, fB = frags(idx)
    sysd = f"/tmp/a24/{idx}"
    os.makedirs(sysd, exist_ok=True)
    geos = {"dimer": fA+fB, "mA": fA, "mB": fB, "mA_cp": fA+ghost(fB), "mB_cp": ghost(fA)+fB}
    for tag, atoms in geos.items(): wxyz(f"{sysd}/{tag}.xyz", atoms, f"A24-{idx} {NAMES[idx]} {tag}")
    for w in OMEGAS:
        for tag in geos:
            tp = f"{sysd}/{tag}_w{w}.toml"; toml(tp, f"{sysd}/{tag}.xyz", w)
            e = run(tp)
            results[(idx,w,tag)] = e
            print(f"A24-{idx} w={w} {tag}: B_total={e['B']:.8f}", flush=True)

# tables
print("\n=== Interaction energies (kcal/mol), CP-corrected | (non-CP) ===")
print(f"{'system':12s} {'omega':>5s} {'RHF':>8s} {'MP2':>8s} {'naiveA':>8s} {'DeltaB':>8s} {'ref':>7s}")
err = {m: {w: [] for w in OMEGAS} for m in ("mp2","A","B")}
for idx in REFS:
    for w in OMEGAS:
        row = []
        for m in ("rhf","mp2","A","B"):
            d, a, b = results[(idx,w,"dimer")][m], results[(idx,w,"mA_cp")][m], results[(idx,w,"mB_cp")][m]
            cp = (d-a-b)*K
            an, bn = results[(idx,w,"mA")][m], results[(idx,w,"mB")][m]
            ncp = (d-an-bn)*K
            row.append((cp,ncp))
            if m!="rhf": err[m][w].append(cp-REFS[idx])
        print(f"{NAMES[idx]:12s} {w:>5s} " + " ".join(f"{c:8.3f}" for c,_ in row) + f" {REFS[idx]:7.3f}   (non-CP: " + " ".join(f"{n:7.3f}" for _,n in row) + ")")
print("\n=== CP MAE vs CCSD(T)/CBS (kcal/mol), 4 dimers ===")
print(f"{'omega':>5s} {'MP2':>7s} {'naiveA':>7s} {'DeltaB':>7s}")
for w in OMEGAS:
    mae = lambda m: sum(abs(x) for x in err[m][w])/len(err[m][w])
    print(f"{w:>5s} {mae('mp2'):7.3f} {mae('A'):7.3f} {mae('B'):7.3f}")
json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k,v in results.items()}, open("/tmp/a24/results.json","w"), indent=1)
