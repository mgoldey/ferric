#!/usr/bin/env python3
"""Locate the B-formulation MAE minimum in r0 at aug-cc-pVQZ, CP-corrected.

Reads the committed `out/*_aqz_terfr0_*.out` sweep (terf split, B / delta-lr),
forms counterpoise-corrected interaction energies, computes the MAE against the
A24 CCSD(T)/CBS reference, and fits a natural cubic spline in r0 to locate the
minimum.

CP convention matches `collect.py`: E_int = E(dimer) - E(mA in dimer basis)
                                          - E(mB in dimer basis).
Only systems with a COMPLETE set of all three fragments at a given r0 are used
at that r0 -- a partially-computed r0 point would otherwise silently shift the
MAE by dropping a system from the average.

Usage:  python3 benchmarks/grid/r0_spline.py
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
K = 627.509474  # Hartree -> kcal/mol
FRAGS = ("dimer", "mA_cp", "mB_cp")

def load_bind():
    """A24 CCSD(T)/CBS reference values (kcal/mol) from `A24.py`.

    Parsed textually rather than imported: `A24.py` does `import qcdb` at
    module scope, which is not installed here, but the BIND block is plain
    numeric literals (Rezac & Hobza, dx.doi.org/10.1021/ct400057w).
    """
    txt = (ROOT / "A24.py").read_text()
    out = {}
    for m in re.finditer(r"BIND\['%s-%s'\s*%\s*\(dbse,\s*(\d+)\s*\)\]\s*=\s*(-?\d+\.\d+)", txt):
        out[f"A24-{int(m.group(1))}"] = float(m.group(2))
    if not out:
        raise SystemExit("could not parse BIND from A24.py")
    return out


BIND = load_bind()


def grab(text, label):
    m = re.search(re.escape(label) + r"\s*=\s*(-?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def collect():
    """{r0: {system: {frag: E_total_B}}} from the aQZ terf r0 sweep."""
    data = defaultdict(lambda: defaultdict(dict))
    pat = re.compile(r"^(a24-\d+)_(dimer|mA_cp|mB_cp)_aqz_terfr0_(\d+p\d+)_B\.out$")
    for p in sorted(OUT.glob("a24-*_aqz_terfr0_*_B.out")):
        m = pat.match(p.name)
        if not m:
            continue
        sysname, frag, r0s = m.groups()
        r0 = float(r0s.replace("p", "."))
        t = p.read_text()
        tot = grab(t, "Total energy")
        b = grab(t, "E_corr Δ-form (B)")
        if tot is None or b is None:
            continue  # failed/incomplete run
        data[r0][sysname][frag] = tot
    return data


def mae_at(sysmap, systems):
    """CP interaction-energy MAE (kcal/mol) over `systems`, or None."""
    errs = []
    for s in systems:
        f = sysmap.get(s, {})
        if not all(k in f for k in FRAGS):
            return None
        e_int = (f["dimer"] - f["mA_cp"] - f["mB_cp"]) * K
        ref = BIND.get(f"A24-{int(s.split('-')[1])}")
        if ref is None:
            return None
        errs.append(abs(e_int - ref))
    return sum(errs) / len(errs)


def natural_cubic_spline(xs, ys):
    """Natural cubic spline coefficients; returns evaluator f(x)."""
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = 3 * (ys[i + 1] - ys[i]) / h[i] - 3 * (ys[i] - ys[i - 1]) / h[i - 1]
    l = [1.0] + [0.0] * (n - 1)
    mu = [0.0] * n
    z = [0.0] * n
    for i in range(1, n - 1):
        l[i] = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    b = [0.0] * n
    c = [0.0] * n
    d = [0.0] * n
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    def f(x):
        i = min(max(0, sum(1 for v in xs[1:-1] if v <= x)), n - 2)
        dx = x - xs[i]
        return ys[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx ** 3

    return f


def main():
    data = collect()
    if not data:
        print("no aQZ terf r0 sweep output found", file=sys.stderr)
        return 1

    # Pick the LARGEST system set that is complete at the most r0 points.
    # The MAE must be over a FIXED system set at every r0 -- averaging over a
    # varying set would make the curve's shape an artifact of which systems
    # happened to finish, not of r0.
    per_r0_complete = {
        r0: {s for s, f in sysmap.items() if all(k in f for k in FRAGS)}
        for r0, sysmap in data.items()
    }
    all_sys = sorted(set().union(*per_r0_complete.values()))
    # r0 points where EVERY system in `all_sys` is complete.
    full_r0 = sorted(r0 for r0, ss in per_r0_complete.items() if set(all_sys) <= ss)
    print(f"r0 points found : {len(data)}   systems seen : {all_sys}")
    print(f"r0 points with ALL {len(all_sys)} systems complete : "
          f"{len(full_r0)}  {[f'{r:.3f}' for r in full_r0]}")
    partial = sorted(set(data) - set(full_r0))
    if partial:
        detail = {f"{r:.3f}": sorted(per_r0_complete[r]) for r in partial}
        print(f"r0 points EXCLUDED (incomplete system set) : {detail}")
    if len(all_sys) < 2 or len(full_r0) < 4:
        print("\nnot enough complete r0 points to fit a cubic spline", file=sys.stderr)
        return 1
    common = all_sys

    pts = []
    for r0 in full_r0:
        m = mae_at(data[r0], common)
        if m is not None:
            pts.append((r0, m))
    print(f"\n{'r0 (A)':>8}  {'MAE (kcal/mol)':>15}")
    for r0, m in pts:
        print(f"{r0:8.4f}  {m:15.4f}")

    if len(pts) < 4:
        print("\nfewer than 4 usable r0 points -- a cubic spline is not warranted",
              file=sys.stderr)
        return 1

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    f = natural_cubic_spline(xs, ys)

    # Dense scan for the interior minimum.
    lo, hi = xs[0], xs[-1]
    N = 20001
    best_x, best_y = None, float("inf")
    for i in range(N):
        x = lo + (hi - lo) * i / (N - 1)
        y = f(x)
        if y < best_y:
            best_x, best_y = x, y
    print(f"\ncubic-spline minimum : r0 = {best_x:.4f} A   MAE = {best_y:.4f} kcal/mol")
    interior = xs[0] < best_x < xs[-1]
    print(f"interior minimum     : {interior}"
          + ("" if interior else "  (at a scan boundary -- NOT a resolved optimum)"))
    grid_x, grid_y = min(pts, key=lambda p: p[1])
    print(f"best sampled point   : r0 = {grid_x:.4f} A   MAE = {grid_y:.4f}")
    print(f"spline depth below best sample : {grid_y - best_y:.4f} kcal/mol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
