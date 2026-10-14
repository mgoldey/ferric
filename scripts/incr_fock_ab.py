#!/usr/bin/env python3
"""Incremental vs full-rebuild closed-shell Fock A/B (#415).

Drives `benchmarks/harness/examples/incr_fock_bench.rs`, one SCF per process:

* cost: wall time, quartets and iterations at the default convergence settings;
* noise: the energy-change noise floor, i.e. the median |dE| over the
  iterations whose dp_rms is already below 1e-9 (the SCF has nothing left to
  descend there, so dE is noise), from a run whose density_conv cannot be met.

    cargo build --release -p ferric-benchmarks --example incr_fock_bench
    python3 scripts/incr_fock_ab.py cost
    python3 scripts/incr_fock_ab.py noise
"""

import os
import re
import statistics
import subprocess
import sys

BIN = "target/release/examples/incr_fock_bench"
MOL = "testdata/molecules"

COST_CASES = [
    ("water.xyz", "cc-pvdz"),
    ("benzene.xyz", "cc-pvdz"),
    ("hbr.xyz", "def2-svp"),
    ("alkane_10.xyz", "sto-3g"),
    ("alkane_20.xyz", "sto-3g"),
    ("alkane_32.xyz", "sto-3g"),
    ("alkane_48.xyz", "sto-3g"),
    ("alkane_10.xyz", "6-31g"),
]

NOISE_CASES = [
    ("water.xyz", "cc-pvdz", "1e-12"),
    ("benzene.xyz", "cc-pvdz", "1e-12"),
    ("hbr.xyz", "def2-svp", "1e-12"),
    ("sih4.xyz", "cc-pvdz", "1e-12"),
    ("alkane_10.xyz", "sto-3g", "1e-8"),
    ("alkane_10.xyz", "sto-3g", "1e-10"),
    ("alkane_10.xyz", "sto-3g", "1e-12"),
    ("alkane_10.xyz", "sto-3g", "1e-14"),
]

RESULT = re.compile(r"(\w+)=(\S+)")
TRACE = re.compile(r"SCF iter=\s*(\d+)\s+E=(\S+)\s+dE=(\S+)\s+dp_rms=(\S+)")


def run(args, trace=False):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    if trace:
        env["FERRIC_SCF_TRACE"] = "1"
    p = subprocess.run([BIN, *args], env=env, capture_output=True, text=True, check=True)
    line = next(ln for ln in p.stdout.splitlines() if ln.startswith("RESULT"))
    return dict(RESULT.findall(line)), p.stderr


def cost():
    print(f"{'case':24} {'nbf':>4} {'it on/off':>9} {'quartets on/off':>15} "
          f"{'wall on (s)':>11} {'wall off (s)':>12} {'off/on wall':>11} {'|E_on-E_off|':>12}")
    for mol, basis in COST_CASES:
        on, _ = run([f"{MOL}/{mol}", basis, "on"])
        off, _ = run([f"{MOL}/{mol}", basis, "off"])
        q = int(on["quartets"]) / int(off["quartets"])
        w_on, w_off = float(on["wall"]), float(off["wall"])
        de = abs(float(on["E"]) - float(off["E"]))
        print(f"{mol[:-4] + '/' + basis:24} {on['nbf']:>4} {on['iters'] + '/' + off['iters']:>9} "
              f"{q:>15.4f} {w_on:>11.2f} {w_off:>12.2f} {w_off / w_on:>11.3f} {de:>12.1e}")
        sys.stdout.flush()


def noise_of(stderr):
    rows = [(int(i), float(de), float(dp)) for i, _, de, dp in TRACE.findall(stderr)]
    tail = [abs(de) for _, de, dp in rows if dp < 1e-9]
    return statistics.median(tail) if tail else float("nan"), len(tail)


def noise():
    print(f"{'case':24} {'thresh':>6} {'noise on':>9} {'noise off':>9} {'on/off':>8}")
    for mol, basis, thresh in NOISE_CASES:
        args = [f"{MOL}/{mol}", basis]
        _, err_on = run([*args, "on", thresh, "1e-14", "40"], trace=True)
        _, err_off = run([*args, "off", thresh, "1e-14", "40"], trace=True)
        (n_on, k_on), (n_off, k_off) = noise_of(err_on), noise_of(err_off)
        print(f"{mol[:-4] + '/' + basis:24} {thresh:>6} {n_on:>9.1e} {n_off:>9.1e} "
              f"{n_on / n_off:>8.0f}   (tail iters {k_on}/{k_off})")
        sys.stdout.flush()


if __name__ == "__main__":
    {"cost": cost, "noise": noise}[sys.argv[1]]()
