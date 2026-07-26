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


def parse_multi_r0(path, label):
    """{r0: E_total} from a `rs-mp2-rpa-sweep` output.

    That method (`kind = "rs-mp2-rpa-sweep"`, `[mp2] r0_sweep = [...]`) solves
    the SCF ONCE and reports several r0 points in one file — which is what makes
    a sweep ~5x cheaper than the same points as separate jobs. A collector that
    assumes one r0 per file silently sees only a fraction of the data (and, if
    it takes the file's single `Total energy` match, the WRONG r0's energy).
    """
    out = {}
    cur = None
    for line in path.read_text().splitlines():
        m = re.search(r"r0 = ([0-9.]+)\s*Å", line)
        if m:
            cur = float(m.group(1))
        m2 = re.search(r"Total energy\s*=\s*(-?\d+\.\d+)", line)
        if m2 and cur is not None:
            out[cur] = float(m2.group(1))
    return out


def collect(basis, form):
    """{r0: {sys_idx: {frag: E_total}}} for one basis/formulation.

    Reads BOTH layouts: single-r0 files (`..._terfr0_0p7000_B.out`) and
    multi-r0 sweep files (`..._terfr0sweep_*.out`).
    """
    label = FORM_LABEL[form]
    data = defaultdict(lambda: defaultdict(dict))

    single = re.compile(
        rf"^a24-(\d+)_(dimer|mA_cp|mB_cp)_{re.escape(basis)}_terfr0_(\d+p\d+)_{form}\.out$"
    )
    for p in sorted(OUT.glob(f"a24-*_{basis}_terfr0_*_{form}.out")):
        m = single.match(p.name)
        if not m:
            continue
        idx, frag, r0s = int(m.group(1)), m.group(2), m.group(3)
        t = p.read_text()
        tot, corr = grab(t, "Total energy"), grab(t, label)
        if tot is None or corr is None:
            continue  # failed / killed / still running
        data[float(r0s.replace("p", "."))][idx][frag] = tot

    # Multi-r0 sweep files. Many tag conventions exist for these
    # (`terfr0sweep_`, `r0scan_`, `r0fine_`, `r0tscan_`, `r0coarse_`, `r0text_`, …).
    # Match ANY tag via a wildcard rather than an allowlist: an allowlist has
    # silently hidden a completed scan three times now (r0fine_, r0tscan_, and
    # r0text_ each had to be added after the fact), and the failure mode is
    # invisible — the curve just comes back short with no error. A new tag must
    # never require editing this regex.
    #
    # Safety of the wildcard: the tag group cannot swallow a formulation because
    # the formulation is read back from the sibling TOML below, not parsed from
    # the filename, and non-sweep files are rejected by parse_multi_r0 returning
    # nothing. Single-r0 files are handled by the separate `terfr0_` branch above.
    multi = re.compile(
        rf"^a24-(\d+)_(dimer|mA_cp|mB_cp)_{re.escape(basis)}"
        rf"_([A-Za-z0-9]+)_(.+)\.out$"
    )
    want_form = {"B": "delta-lr", "T": "coupled-rings"}[form]
    for p in sorted(OUT.glob(f"a24-*_{basis}_*_*.out")):
        m = multi.match(p.name)
        if not m:
            continue
        idx, frag = int(m.group(1)), m.group(2)
        toml = ROOT / "toml" / (p.stem + ".toml")
        if toml.exists():
            tt = toml.read_text()
            fm = re.search(r'formulation\s*=\s*"([a-z-]+)"', tt)
            if fm and fm.group(1) != want_form:
                continue
        elif form != "B":  # unlabelled legacy sweeps were the B production run
            continue
        for r0, tot in parse_multi_r0(p, label).items():
            data[r0][idx][frag] = tot
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


def suggest(pts, best_x, interior, n=5, halfwidth=0.10):
    """Propose the next r0 points to run, informed by the current spline.

    Two regimes, because they need opposite things:

    * **Interior minimum** — the bracket is known, so refine INSIDE it: `n`
      points spanning `best_x ± halfwidth`, clipped to stay strictly inside the
      sampled range (extrapolating a cubic past its data is not evidence).
      Points already within half a step of an existing sample are dropped —
      re-running them buys nothing.
    * **Boundary minimum** — the optimum is outside the sampled range, so
      EXTEND in that direction by one grid step at a time rather than refining
      a minimum we have not actually bracketed yet.
    """
    xs = sorted(p[0] for p in pts)
    step = min(b - a for a, b in zip(xs, xs[1:])) if len(xs) > 1 else halfwidth

    if not interior:
        # Walk outward from whichever edge holds the minimum.
        if best_x <= xs[0]:
            out = [round(xs[0] - step * k, 3) for k in range(1, n + 1)]
            return [x for x in out if x > 0]
        return [round(xs[-1] + step * k, 3) for k in range(1, n + 1)]

    lo, hi = max(xs[0], best_x - halfwidth), min(xs[-1], best_x + halfwidth)
    if hi <= lo:
        return []
    cand = [lo + (hi - lo) * i / (n - 1) for i in range(n)] if n > 1 else [best_x]
    # "Already sampled" must be judged against the REFINEMENT spacing, not the
    # existing coarse spacing: with a 0.25 coarse grid and a +/-0.10 window,
    # a step/2 = 0.125 exclusion radius rejects the entire window and returns
    # nothing. Use half the new spacing instead, floored so we never dedupe
    # away points that are genuinely new information.
    fine = (hi - lo) / max(n - 1, 1)
    tol = min(step, fine) / 2
    keep = []
    for c in cand:
        c = round(c, 3)
        if all(abs(c - x) > tol for x in xs) and all(abs(c - k) > 1e-9 for k in keep):
            keep.append(c)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--form", choices=("B", "T"), default="B")
    ap.add_argument("--basis", default="aqz")
    ap.add_argument("--systems", default="",
                    help="comma A24 indices (default: all complete at some r0)")
    ap.add_argument("--suggest", action="store_true",
                    help="propose the next r0 points from the current spline")
    ap.add_argument("--n-suggest", type=int, default=5)
    ap.add_argument("--halfwidth", type=float, default=0.10,
                    help="refinement window half-width in Angstrom (interior case)")
    ap.add_argument("--toml", action="store_true",
                    help="emit the suggestion as a [mp2] r0_sweep line")
    a = ap.parse_args()
    want = [int(x) for x in a.systems.split(",") if x.strip()] if a.systems else None
    res = analyze(a.basis, a.form, want, verbose=not a.toml)
    if not res:
        return 1
    pts, bx, _by, interior = res
    if a.suggest or a.toml:
        s = suggest(pts, bx, interior, a.n_suggest, a.halfwidth)
        if a.toml:
            print("r0_sweep = [" + ", ".join(f"{x:.4f}" for x in s) + "]")
        else:
            kind = "refine around" if interior else "EXTEND past"
            print(f"\nsuggested next r0 ({kind} {bx:.4f} A): {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
