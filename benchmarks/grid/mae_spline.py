#!/usr/bin/env python3
"""Locate the MAE-vs-r0 minimum of a range-separated formulation by cubic spline.

Generalizes `r0_spline.py`: works for either formulation (B = delta-lr,
T = coupled-rings), any A24 system subset, and any basis tag.

    # B over the 4-system r0-scan subset at aQZ
    python3 benchmarks/grid/mae_spline.py --form B

    # T over the 7-system weakly-bound subset
    python3 benchmarks/grid/mae_spline.py --form T --systems 12,15,20,21,22,23,24

Reads `out/a24-NN_{frag}_{basis}_terfr0_{r0}_{FORM}.out`, forms
counterpoise-corrected interaction energies, and fits a natural cubic spline to
MAE(r0) against the A24 CCSD(T)/CBS reference.

Two things this is careful about, because both silently corrupt the curve:

  * **Fixed system set.** The MAE must average the SAME systems at every r0.
    If a system is missing at one r0, including it elsewhere makes the curve's
    shape an artifact of which jobs happened to finish. r0 points without the
    full requested set are excluded, and reported.
  * **Interior vs boundary minimum.** A spline always returns some minimum. If
    it sits at the edge of the sampled range, the true optimum is outside it
    and the value is NOT an answer — that is reported explicitly rather than
    quoted as an optimum.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
K = 627.509474  # Hartree -> kcal/mol
FRAGS = ("dimer", "mA_cp", "mB_cp")

# The energy label differs per formulation.
FORM_LABEL = {
    "B": "E_corr Δ-form (B)",
    "T": "E_corr coupled (T)",
}


def load_bind():
    """A24 CCSD(T)/CBS references (kcal/mol), parsed from A24.py's literals.

    Parsed textually rather than imported: A24.py does `import qcdb` at module
    scope and qcdb is not installed here.
    """
    txt = (ROOT / "A24.py").read_text()
    out = {}
    for m in re.finditer(
        r"BIND\['%s-%s'\s*%\s*\(dbse,\s*(\d+)\s*\)\]\s*=\s*(-?\d+\.\d+)", txt
    ):
        out[int(m.group(1))] = float(m.group(2))
    if not out:
        raise SystemExit("could not parse BIND from A24.py")
    return out


def grab(text, label):
    m = re.search(re.escape(label) + r"\s*=\s*(-?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def collect(basis, form):
    """{r0: {sys_idx: {frag: E_total}}} for one basis/formulation."""
    label = FORM_LABEL[form]
    data = defaultdict(lambda: defaultdict(dict))
    pat = re.compile(
        rf"^a24-(\d+)_(dimer|mA_cp|mB_cp)_{re.escape(basis)}_terfr0_(\d+p\d+)_{form}\.out$"
    )
    for p in sorted(OUT.glob(f"a24-*_{basis}_terfr0_*_{form}.out")):
        m = pat.match(p.name)
        if not m:
            continue
        idx, frag, r0s = int(m.group(1)), m.group(2), m.group(3)
        t = p.read_text()
        tot, corr = grab(t, "Total energy"), grab(t, label)
        if tot is None or corr is None:
            continue  # failed / killed / still running
        data[float(r0s.replace("p", "."))][idx][frag] = tot
    return data


def natural_cubic_spline(xs, ys):
    """Natural cubic spline through (xs, ys); returns an evaluator."""
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


def analyze(basis, form, want, verbose=True):
    """Returns (points, spline_min_r0, spline_min_mae, interior) or None."""
    bind = load_bind()
    data = collect(basis, form)
    if not data:
        if verbose:
            print(f"no {basis}/{form} terf r0 output found", file=sys.stderr)
        return None

    complete = {
        r0: {s for s, f in m.items() if all(k in f for k in FRAGS)}
        for r0, m in data.items()
    }
    want = set(want) if want else set().union(*complete.values())
    usable = sorted(r0 for r0, ss in complete.items() if want <= ss)

    if verbose:
        print(f"basis={basis}  formulation={form}  systems={sorted(want)}")
        print(f"r0 points with the full system set : {len(usable)} "
              f"{[f'{r:.3f}' for r in usable]}")
        missing = sorted(set(data) - set(usable))
        if missing:
            gaps = {f"{r:.3f}": sorted(want - complete[r]) for r in missing}
            print(f"r0 points EXCLUDED (missing systems) : {gaps}")

    pts = []
    for r0 in usable:
        errs = []
        for s in sorted(want):
            f = data[r0][s]
            e_int = (f["dimer"] - f["mA_cp"] - f["mB_cp"]) * K
            errs.append(abs(e_int - bind[s]))
        pts.append((r0, sum(errs) / len(errs)))

    if verbose and pts:
        print(f"\n{'r0 (A)':>8}  {'MAE (kcal/mol)':>15}")
        for r0, m in pts:
            print(f"{r0:8.4f}  {m:15.4f}")

    if len(pts) < 4:
        if verbose:
            print("\nfewer than 4 usable r0 points -- cannot fit a cubic spline",
                  file=sys.stderr)
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    f = natural_cubic_spline(xs, ys)
    lo, hi = xs[0], xs[-1]
    best_x, best_y = lo, f(lo)
    N = 20001
    for i in range(N):
        x = lo + (hi - lo) * i / (N - 1)
        y = f(x)
        if y < best_y:
            best_x, best_y = x, y
    interior = xs[0] < best_x < xs[-1]

    if verbose:
        print(f"\nspline minimum : r0 = {best_x:.4f} A   MAE = {best_y:.4f} kcal/mol")
        if interior:
            gx, gy = min(pts, key=lambda p: p[1])
            print(f"INTERIOR minimum -- resolved. best sampled r0={gx:.3f} "
                  f"(MAE {gy:.4f}); spline is {gy - best_y:.4f} lower.")
        else:
            print("BOUNDARY minimum -- the optimum lies OUTSIDE the sampled "
                  f"range [{lo:.2f}, {hi:.2f}]. Extend the scan; do NOT quote "
                  "this as the optimum.")
    return pts, best_x, best_y, interior


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--form", choices=("B", "T"), default="B")
    ap.add_argument("--basis", default="aqz")
    ap.add_argument("--systems", default="",
                    help="comma A24 indices (default: all complete at some r0)")
    a = ap.parse_args()
    want = [int(x) for x in a.systems.split(",") if x.strip()] if a.systems else None
    return 0 if analyze(a.basis, a.form, want) else 1


if __name__ == "__main__":
    raise SystemExit(main())
