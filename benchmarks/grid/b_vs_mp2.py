"""B vs plain MP2 on identical footing: same SCF, same RI aux, same CP triple.

Every B sweep output prints E(MP2, Coulomb) alongside the B correction, so the
MP2 baseline needs no separate calculation and carries no basis/reference
mismatch. MP2 total = Total energy - (E_dRPA[terf] - E_dMP2[terf]), verified
against the printed E_corr to 10 digits.
"""
import re, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/grid")
import mae_spline as M

OUT = Path("benchmarks/grid/out")
K = 627.5094740631
FRAGS = ("dimer", "mA_cp", "mB_cp")

def parse(path):
    """{r0: (E_total_B, E_total_MP2)}"""
    out, cur, cache = {}, None, {}
    for line in path.read_text(errors="ignore").splitlines():
        m = re.search(r"r0 = ([0-9.]+)\s*Å", line)
        if m:
            cur = round(float(m.group(1)), 4); cache = {}
        for key in ("E(dMP2, terf)", "E(dRPA, terf)"):
            mm = re.search(re.escape(key) + r"\s*=\s*(-?\d+\.\d+)", line)
            if mm: cache[key] = float(mm.group(1))
        mt = re.search(r"Total energy\s*=\s*(-?\d+\.\d+)", line)
        if mt and cur is not None:
            tot = float(mt.group(1))
            if "E(dRPA, terf)" in cache and "E(dMP2, terf)" in cache:
                corr = cache["E(dRPA, terf)"] - cache["E(dMP2, terf)"]
                out[cur] = (tot, tot - corr)
    return out

# Gather every B output, keyed by (system, fragment)
data = {}
for p in OUT.glob("a24-*_aqz_*_B.out"):
    m = re.match(r"a24-(\d+)_(dimer|mA_cp|mB_cp)_aqz_(\w+)_B\.out$", p.name)
    if not m: continue
    toml = Path("benchmarks/grid/toml")/(p.stem + ".toml")
    if toml.exists():
        fm = re.search(r'formulation\s*=\s*"([a-z-]+)"', toml.read_text())
        if fm and fm.group(1) != "delta-lr": continue
    s, frag = int(m.group(1)), m.group(2)
    for r0, pair in parse(p).items():
        data.setdefault(r0, {}).setdefault(s, {})[frag] = pair

bind = M.load_bind()
print(f"{'r0':>6} {'n':>3} {'MAE(B)':>9} {'MAE(MP2)':>9} {'improve':>9} {'%':>7}")
print("-"*48)
for r0 in sorted(data):
    eb, em = [], []
    for s, fr in data[r0].items():
        if len(fr) < 3 or s not in bind: continue
        b  = (fr["dimer"][0]-fr["mA_cp"][0]-fr["mB_cp"][0])*K
        mp = (fr["dimer"][1]-fr["mA_cp"][1]-fr["mB_cp"][1])*K
        eb.append(abs(b-bind[s])); em.append(abs(mp-bind[s]))
    if len(eb) < 24: continue
    mb, mm = sum(eb)/len(eb), sum(em)/len(em)
    print(f"{r0:6.2f} {len(eb):3} {mb:9.4f} {mm:9.4f} {mm-mb:+9.4f} {100*(mm-mb)/mm:6.1f}%")
