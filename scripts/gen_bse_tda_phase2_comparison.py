"""
Compare ferric BSE-TDA[G0W0@HF]/aug-cc-pVDZ output against the
testdata/reference/thiel_set_subset.json TBE reference set. See
docs/bse-tda-phase2-results.md for the full writeup, methodology, and the
state-matching caveat this script's naive nearest-energy heuristic carries
(NOT a validated assignment -- see that doc before quoting these numbers).

Regenerate the logs this reads via:
  OPENBLAS_NUM_THREADS=1 cargo build --release -p ferric-cli --bin ferric
  for m in formaldehyde ethylene acetaldehyde butadiene cyclopropene furan \
           glyoxal pyrazine pyridine pyrimidine; do
    OPENBLAS_NUM_THREADS=1 target/release/ferric \
        examples/${m}-bse-tda-augdz.toml > \
        benchmarks/bse-tda-pilot/phase2-logs/${m}_bse.log
  done
"""
import json, re

with open("testdata/reference/thiel_set_subset.json") as f:
    ref = json.load(f)

logs = {
    "formaldehyde": "formaldehyde_bse.log",
    "ethylene": "ethylene_bse.log",
    "acetaldehyde": "acetaldehyde_bse.log",
    "butadiene": "butadiene_bse.log",
    "cyclopropene": "cyclopropene_bse.log",
    "furan": "furan_bse.log",
    "glyoxal": "glyoxal_bse.log",
    "pyrazine": "pyrazine_bse.log",
    "pyridine": "pyridine_bse.log",
    "pyrimidine": "pyrimidine_bse.log",
}
logdir = "benchmarks/bse-tda-pilot/phase2-logs/"

def parse_states(path, n=15):
    states = []
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if m:
                states.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
                if len(states) >= n:
                    break
    return states

print(f"{'Molecule':14} {'RefLabel':10} {'RefE(eV)':>9} {'Reff':>7}  {'BestMatch':>10} {'MatchE':>8} {'Mattf':>7} {'dE':>7}")
for m in ref["molecules"]:
    name = m["name"]
    computed = parse_states(logdir + logs[name])
    for s in m["states"]:
        ref_e = s["excitation_energy_eV"]
        ref_f = s.get("oscillator_strength")
        ref_f_val = ref_f if isinstance(ref_f, (int,float)) else 0.0
        # nearest-neighbor match on energy alone (simple heuristic, documented limitation)
        best = min(computed, key=lambda c: abs(c[1]-ref_e))
        de = best[1]-ref_e
        print(f"{name:14} {s['label']:10} {ref_e:9.3f} {ref_f_val:7.3f}  n={best[0]:<8} {best[1]:8.3f} {best[2]:7.3f} {de:+7.3f}")

print()
print("=== Aggregate (naive nearest-energy match, N=17 states) ===")
diffs = []
for m in ref["molecules"]:
    name = m["name"]
    computed = parse_states(logdir + logs[name])
    for s in m["states"]:
        ref_e = s["excitation_energy_eV"]
        best = min(computed, key=lambda c: abs(c[1]-ref_e))
        diffs.append(best[1]-ref_e)
import statistics
mae = statistics.mean(abs(d) for d in diffs)
mse = statistics.mean(diffs)
print(f"MAE = {mae:.3f} eV, MSE (signed mean) = {mse:+.3f} eV, N={len(diffs)}")
print(f"positive (ferric too high): {sum(1 for d in diffs if d>0)}/{len(diffs)}")
