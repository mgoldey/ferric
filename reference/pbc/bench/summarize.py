"""Summary tables from bench_results/timing_*.jsonl (valid runs only).

python summarize.py bench_results/timing_n1.jsonl [...] [--metric wall] [--boot 10000] [--seed 228]
                    [--quiet-only] [--ladder dryice [--ladder-precision 1e-11]]

Sections (issue #228; the statistics are pre-registered in README.md "Session 4 pre-registration"):
1. Per (cell, N, code, input, box_state): median [min-max], IQR, IQR/median, relative MAD
   (median |x - median| / median) per metric, labelled "n too small" below the pre-registered rep
   count (5 at N=6, 3 at N=1), else NOISY when IQR/median of the chosen metric exceeds 10%, else ok.
   Rows written before `box_state` / `bin_sha256` / `run_id` existed get "unlabelled" / no-sha / None.
2. ferric/PySCF ratio of medians over the PAIRED reps, same cell, N, method and binary sha256 (groups
   of different binaries are never compared), with a PAIRED bootstrap 95% percentile CI: pairs are
   (session, run_id, rep) keys that have a valid row for both codes (last valid attempt of each);
   resampling draws pairs with replacement.  The RNG is random.Random(seed): reproducible.  The
   bootstrap pools sessions; section 3's range is the drift guard.
3. Between sessions, per (cell, N, code, input, sha): each session's median, the CV of those medians
   (sample stdev / mean) and max/min; and per comparable pair each session's ratio of medians with
   max/min.  QUOTABLE iff >= 2 sessions, max/min - 1 < 5%, every contributing row quiet, and every
   per-session group of both codes labelled ok.
4. (--ladder CELL) the dryice PBE grid-ladder gate of the session-4 pre-registration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scatter_driver import is_scf_stage  # noqa: E402

KEYS = [
    ("wall", "wall"),
    ("t_setup_s", "setup"),
    ("t_scf_s", "scf"),
    ("s_per_iter", "s/iter"),
    ("iterations", "iters"),
    ("t_df_build_s", "df"),
    ("t_hcore_s", "hcore"),
    ("foreign_cpu_frac", "foreign"),
]
NOISY = 0.10
QUOTE_RANGE = 0.05
# Pre-registered valid reps per group (README "Session 4 pre-registration"); other N: 3.
MIN_REPS = {1: 3, 6: 5}


def min_reps(threads):
    return MIN_REPS.get(threads, 3)


def q(v):
    v = sorted(v)
    if len(v) < 2:
        return 0.0
    qs = statistics.quantiles(v, n=4, method="inclusive")
    return qs[2] - qs[0]


def spread_stats(v, min_n=1):
    """median, min, max, IQR, IQR/median, relative MAD and a label.

    label: "n too small" below min_n reps (noisy = None: undecided), else "NOISY" iff
    IQR/median > 10%, else "ok"."""
    med = statistics.median(v)
    iqr = q(v)
    mad = statistics.median([abs(x - med) for x in v])
    iqr_rel = iqr / med if med else float("nan")
    if len(v) < min_n:
        noisy, label = None, "n too small"
    else:
        noisy = bool(iqr_rel > NOISY)
        label = "NOISY" if noisy else "ok"
    return dict(
        n=len(v),
        median=med,
        min=min(v),
        max=max(v),
        iqr=iqr,
        iqr_rel=iqr_rel,
        rel_mad=mad / med if med else float("nan"),
        noisy=noisy,
        label=label,
    )


def input_label(r):
    if r["code"] == "ferric":
        return r.get("toml", "").split("/")[-1]
    rule = r.get("conv_rule", "pyscf")
    extra = f" [{r['pyscf_args']}]" if r.get("pyscf_args") else ""
    return f"{r.get('method', 'rhf')} p{r.get('precision')} g{r.get('grid')} conv={rule}{extra}"


def normalise(r):
    """Recompute ferric's stage split from the raw stage table (one definition everywhere)."""
    if r["code"] == "ferric" and (r.get("stages") or {}).get("total") is not None:
        st = r["stages"]
        r["t_scf_s"] = sum(v for n, v in st.items() if is_scf_stage(n))
        r["t_setup_s"] = st["total"] - r["t_scf_s"]
        if r.get("iterations"):
            r["s_per_iter"] = r["t_scf_s"] / r["iterations"]
    r.setdefault("box_state", "unlabelled")
    r.setdefault("bin_sha256", None)
    r.setdefault("run_id", None)
    return r


def sha_short(sha):
    return sha[:12] if sha else "no-sha"


def is_valid(r):
    return bool(r.get("valid")) and r.get("exit") == 0 and not r.get("smoke")


def read_jsonl(path):
    """Rows of a JSONL file.  A truncated LAST line (a run killed mid-write) is skipped with a
    warning; a bad line anywhere else is an error."""
    lines = [ln for ln in open(path) if ln.strip()]
    rows = []
    for i, ln in enumerate(lines):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print(f"warning: {path}: skipping a truncated last line", file=sys.stderr)
    return rows


def load_rows(paths):
    return [normalise(r) for p in paths for r in read_jsonl(p)]


def pair_by_rep(rows_a, rows_b, metric):
    """[(a, b)] metric values for (session, run_id, rep) keys present in both.

    The last valid attempt of each key is used.  run_id separates driver invocations writing to the
    same --out (a top-up run restarts rep at 0)."""

    def last(rows):
        d = {}
        for r in sorted(rows, key=lambda r: r.get("attempt", 0)):
            if r.get(metric) is not None:
                d[(r.get("session"), r.get("run_id"), r.get("rep"))] = r[metric]
        return d

    da, db = last(rows_a), last(rows_b)
    keys = sorted(set(da) & set(db), key=lambda k: (str(k[0]), str(k[1]), k[2]))
    return [(da[k], db[k]) for k in keys]


def median_ratio(pairs):
    return statistics.median([a for a, _ in pairs]) / statistics.median(
        [b for _, b in pairs]
    )


def paired_bootstrap_ci(pairs, n_boot=10000, seed=228, level=0.95):
    """(point, lo, hi): ratio of medians and its paired-bootstrap percentile CI."""
    if not pairs:
        raise ValueError("no pairs")
    rng = random.Random(seed)
    n = len(pairs)
    stats = []
    for _ in range(n_boot):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        stats.append(median_ratio(s))
    stats.sort()
    lo_i = int((1 - level) / 2 * (n_boot - 1) + 0.5)
    hi_i = int((1 + level) / 2 * (n_boot - 1) + 0.5)
    return median_ratio(pairs), stats[lo_i], stats[hi_i]


def between_sessions(values_by_session):
    """{session: [values]} -> per-session medians, CV of the medians, max/min."""
    meds = {s: statistics.median(v) for s, v in sorted(values_by_session.items()) if v}
    m = list(meds.values())
    cv = statistics.stdev(m) / statistics.mean(m) if len(m) >= 2 else float("nan")
    return dict(
        medians=meds, cv=cv, max_over_min=max(m) / min(m) if m else float("nan")
    )


def fmt(x, nd=3):
    return "nan" if x is None or x != x else f"{x:.{nd}f}"


def section_spread(g, bad, metric):
    print(
        "\n## 1. cell N code (input) box_state | n_valid n_discarded | metric: median [min-max] "
        "IQR IQR/med relMAD | energy spread | load range"
    )
    for k in sorted(set(g) | set(bad), key=str):
        rs = g.get(k, [])
        if not rs:
            print(k, f"n=0 bad={bad[k]} | no valid runs")
            continue
        out = []
        label = "n too small"
        for key, nm in KEYS:
            v = [r[key] for r in rs if r.get(key) is not None]
            if v:
                st = spread_stats(v, min_reps(k[1]))
                if key == metric:
                    label = st["label"]
                out.append(
                    f"{nm} {st['median']:.3f} [{st['min']:.3f}-{st['max']:.3f}] IQR {st['iqr']:.3f} "
                    f"({100 * st['iqr_rel']:.1f}%) relMAD {100 * st['rel_mad']:.1f}%"
                )
        e = [r["energy"] for r in rs if r.get("energy") is not None]
        e_txt = f"E {e[0]:.10f} spread {max(e) - min(e):.1e}" if e else "E missing"
        rss = [r["child_peak_rss_bytes"] for r in rs if r.get("child_peak_rss_bytes")]
        if rss:
            e_txt += f" | RSS {statistics.median(rss) / 2**30:.3f} GiB"
        conv = {r.get("converged") for r in rs}
        e_txt += f" | converged {sorted(conv, key=str)}"
        lo = min(min(r["load_start"], r["load_end"]) for r in rs)
        hi = max(max(r["load_start"], r["load_end"]) for r in rs)
        print(
            k,
            f"n={len(rs)} bad={bad.get(k, 0)} {label} |",
            " | ".join(out),
            f"| {e_txt} | load {lo:.2f}-{hi:.2f}",
        )


def section_ratios(by_input, metric, n_boot, seed):
    print(
        f"\n## 2. ferric/PySCF ratio of {metric} medians over the PAIRED reps, paired on "
        f"(session, run_id, rep) and pooled across sessions of one binary; bootstrap {n_boot}, "
        f"seed {seed}"
    )
    for f, p in comparable_pairs(by_input):
        c, n, sha = f[0], f[1], sha_short(f[4])
        pairs = pair_by_rep(by_input[f], by_input[p], metric)
        if not pairs:
            print(c, f"N={n}", f"bin {sha}", f[3], "vs", p[3], "| no paired reps")
            continue
        pt, lo, hi = paired_bootstrap_ci(pairs, n_boot, seed)
        states = sorted({r["box_state"] for r in by_input[f] + by_input[p]})
        print(
            c,
            f"N={n}",
            f"bin {sha}",
            f[3],
            "vs",
            p[3],
            f"| pairs {len(pairs)} | ratio {pt:.3f} 95% CI [{lo:.3f}, {hi:.3f}]",
            f"| box {','.join(states)}",
        )


def section_sessions(by_input, metric):
    print(f"\n## 3. between sessions ({metric}): per-session medians, CV, max/min")
    for k in sorted(by_input, key=str):
        vs = defaultdict(list)
        for r in by_input[k]:
            if r.get(metric) is not None:
                vs[r.get("session")].append(r[metric])
        bs = between_sessions(vs)
        meds = " ".join(f"{s}:{v:.3f}" for s, v in bs["medians"].items())
        lab = k[:4] + (f"bin {sha_short(k[4])}",)
        print(lab, f"| {meds} | CV {fmt(bs['cv'])} | max/min {fmt(bs['max_over_min'])}")
    print(
        "\n   per-session ferric/PySCF ratio of medians.  QUOTABLE iff >= 2 sessions, max/min - 1 < "
        f"{100 * QUOTE_RANGE:.0f}%, every contributing row quiet, and every per-session group "
        "(each code) labelled ok (not NOISY, not n too small)"
    )
    for f, p in comparable_pairs(by_input):
        ratios = session_ratios(by_input[f], by_input[p], metric)
        if not ratios:
            continue
        ok, why = quotable(by_input[f], by_input[p], metric, f[1])
        mm = max(ratios.values()) / min(ratios.values())
        txt = " ".join(f"{s}:{x:.3f}" for s, x in ratios.items())
        print(
            f[0],
            f"N={f[1]}",
            f"bin {sha_short(f[4])}",
            f[3],
            "vs",
            p[3],
            f"| {txt} | max/min {mm:.3f}",
            "QUOTABLE" if ok else "not quotable (" + "; ".join(why) + ")",
        )


def comparable_pairs(by_input):
    """(ferric key, PySCF key) with the same cell, N, method and binary sha256.

    Keys are (cell, N, code, input, bin_sha256); groups of different binaries are never pooled
    or compared."""
    out = []
    for f in sorted(by_input, key=str):
        if f[2] != "ferric":
            continue
        for p in sorted(by_input, key=str):
            if p[2] != "pyscf" or (p[0], p[1], p[4]) != (f[0], f[1], f[4]):
                continue
            if same_method(by_input[f], by_input[p]):
                out.append((f, p))
    return out


def quotable(rows_f, rows_p, metric, threads):
    """(ok, reasons) for the README quoting rule on one (ferric input, PySCF input) pair."""
    why = []
    ratios = session_ratios(rows_f, rows_p, metric)
    if len(ratios) < 2:
        why.append(f"{len(ratios)} session(s) < 2")
    elif max(ratios.values()) / min(ratios.values()) - 1 >= QUOTE_RANGE:
        why.append("between-session range >= 5%")
    used = [r for r in rows_f + rows_p if r.get("session") in ratios]
    states = {r["box_state"] for r in used}
    if states - {"quiet"}:
        why.append("non-quiet rows: " + ",".join(sorted(states - {"quiet"})))
    for code, rows in (("ferric", rows_f), ("pyscf", rows_p)):
        per = defaultdict(list)
        for r in rows:
            if r.get(metric) is not None and r.get("session") in ratios:
                per[r.get("session")].append(r[metric])
        for sess, v in sorted(per.items(), key=lambda kv: str(kv[0])):
            lab = spread_stats(v, min_reps(threads))["label"]
            if lab != "ok":
                why.append(f"{code} {sess} {lab}")
    return (not why), why


def same_method(rows_f, rows_p):
    return {r.get("method", "rhf") for r in rows_f} == {
        r.get("method", "rhf") for r in rows_p
    }


def session_ratios(rows_f, rows_p, metric):
    """{session: median_f / median_p} over sessions with valid rows for both codes."""
    vf, vp = defaultdict(list), defaultdict(list)
    for r in rows_f:
        if r.get(metric) is not None:
            vf[r.get("session")].append(r[metric])
    for r in rows_p:
        if r.get(metric) is not None:
            vp[r.get("session")].append(r[metric])
    return {
        s: statistics.median(vf[s]) / statistics.median(vp[s])
        for s in sorted(set(vf) & set(vp), key=str)
    }


LADDER_TIMED, LADDER_REF, LADDER_REF_CHECK = "75x302", "250x590", "200x590"


def ladder_gate(e_f, e_p, ref_tol=1e-6, slack=1e-8):
    """dryice PBE ladder gate (README "Session 4 pre-registration").

    e_f, e_p: {"RxA": energy} per code.  The ref (250x590) is valid for a code only if it moves
    < ref_tol from 200x590; then pass iff |E_f(75x302) - E_P(75x302)| <= |E_f(ref) - E_f(75x302)|
    + |E_P(ref) - E_P(75x302)| + slack.  Returns dict(status = pass | fail | undecidable, ...)."""
    need = (LADDER_TIMED, LADDER_REF, LADDER_REF_CHECK)
    for name, e in (("ferric", e_f), ("pyscf", e_p)):
        miss = [g for g in need if g not in e]
        if miss:
            return dict(status="undecidable", reason=f"{name} missing {miss}")
    moves = {
        "ferric": abs(e_f[LADDER_REF] - e_f[LADDER_REF_CHECK]),
        "pyscf": abs(e_p[LADDER_REF] - e_p[LADDER_REF_CHECK]),
    }
    bad = [c for c, m in moves.items() if not m < ref_tol]
    if bad:
        return dict(
            status="undecidable", reason=f"ref not converged for {bad}", ref_moves=moves
        )
    t, r = LADDER_TIMED, LADDER_REF
    lhs = abs(e_f[t] - e_p[t])
    rhs = abs(e_f[r] - e_f[t]) + abs(e_p[r] - e_p[t]) + slack
    return dict(
        status="pass" if lhs <= rhs else "fail",
        lhs=lhs,
        rhs=rhs,
        ref_moves=moves,
        cross_at_ref=abs(e_f[r] - e_p[r]),
    )


def grid_label(r, cell):
    """ "RxA" of a PBE row, or None (ferric: <cell>_pbe.toml = 75x302 or <cell>_pbe_RxA.toml)."""
    if r["code"] == "pyscf":
        g = r.get("grid")
        return f"{g[0]}x{g[1]}" if g else None
    name = (r.get("toml") or "").split("/")[-1]
    if name == f"{cell}_pbe.toml":
        return "75x302"
    m = re.fullmatch(re.escape(cell) + r"_pbe_(\d+)x(\d+)\.toml", name)
    return f"{m.group(1)}x{m.group(2)}" if m else None


def section_ladder(rows, cell, precision=None):
    print(f"\n## 4. PBE grid ladder gate, cell {cell} (valid rows, any box state)")
    e = {"ferric": defaultdict(list), "pyscf": defaultdict(list)}
    shas, precs = set(), set()
    for r in rows:
        if r["cell"] != cell or r.get("method") != "pbe" or not is_valid(r):
            continue
        if (
            r["code"] == "pyscf"
            and precision is not None
            and r.get("precision") != precision
        ):
            continue
        g = grid_label(r, cell)
        if g is None or r.get("energy") is None:
            continue
        e[r["code"]][g].append(r["energy"])
        if r["code"] == "ferric":
            shas.add(r["bin_sha256"])
        else:
            precs.add(r.get("precision"))
    if len(shas) > 1 or len(precs) > 1:
        print(
            f"refused: {len(shas)} ferric binaries / PySCF precisions {sorted(precs)}"
        )
        return None
    ef = {g: statistics.median(v) for g, v in e["ferric"].items()}
    ep = {g: statistics.median(v) for g, v in e["pyscf"].items()}
    for g in sorted(set(ef) | set(ep), key=lambda x: [int(t) for t in x.split("x")]):
        a, b = ef.get(g), ep.get(g)
        d = f"{a - b:+.3e}" if a is not None and b is not None else "-"
        print(g, f"ferric {a}", f"pyscf {b}", f"f-P {d}")
    res = ladder_gate(ef, ep)
    print("gate:", res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--metric", default="wall")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=228)
    ap.add_argument(
        "--quiet-only", action="store_true", help="drop rows with box_state != quiet"
    )
    ap.add_argument(
        "--ladder", default=None, help="also run the PBE ladder gate on CELL"
    )
    ap.add_argument(
        "--ladder-precision",
        type=float,
        default=1e-11,
        help="PySCF rows used by --ladder",
    )
    a = ap.parse_args(argv)
    rows = load_rows(a.paths)
    g = defaultdict(list)
    bad = defaultdict(int)
    # (cell, N, code, input, bin_sha256) -> valid rows (all box states; quotable() checks them)
    by_input = defaultdict(list)
    for r in rows:
        if a.quiet_only and r["box_state"] != "quiet":
            continue
        k = (r["cell"], r["threads"], r["code"], input_label(r), r["box_state"])
        if is_valid(r):
            g[k].append(r)
            by_input[k[:4] + (r["bin_sha256"],)].append(r)
        else:
            bad[k] += 1
    section_spread(g, bad, a.metric)
    section_ratios(by_input, a.metric, a.boot, a.seed)
    section_sessions(by_input, a.metric)
    if a.ladder:
        section_ladder(rows, a.ladder, a.ladder_precision)


if __name__ == "__main__":
    main()
