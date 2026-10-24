"""Provenance check for the stage-1 evaluation cache (Addendum I.0).

The stage-1 cache was filled by two different ferric builds (one built from a
pre-commit working tree before the 18:15 SIGBUS on 2026-10-10, one after).
Neither build's stamp was recorded. This script re-evaluates a sample of cached
entries on the CURRENT build (`ferric.__build__` is printed) and reports
|cached - fresh| per entry, against the 1e-10 Ha bar.

Sample (chosen so the producing build is known from the run log, not guessed):
  * build 1: non-CP fragments (dimer/mA/mB) at r0 = 0.6 A, linked and s = 2 --
    stage1 nodes 1-3, first evaluated before the 18:15 crash;
  * build 2: CP fragments (dimer/mA_cp/mB_cp) at r0 = 1.00 A linked -- only
    reached in the CP valley / baseline phases (2026-10-11, 00:00-02:57).
Only seams whose cache key is exact (r0 0.6/1.0, s linked/2) are used. Cache
keys print b to 10 significant digits, and re-evaluating at that rounded b moves
E_nl by up to ~1e-10 Ha at small b (MEASURED, first version of this script).
So b is mapped back to the EXACT float the search used: the Chebyshev nodes of
`session._cheb_bs` over the registered window and its x4 shifts, plus b = 11.
Only entries whose b is recovered exactly are compared.

Usage: python recheck_cache.py CACHE.json [systems=4,2] > out.json
"""

import json
import math
import sys

import session as S

CACHE = sys.argv[1]
IDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "4,2").split(",")]
SEAMS = {"ncp": [(0.6, None), (0.6, 2.0)], "cp": [(1.0, None)]}
TAGS = {"ncp": ("dimer", "mA", "mB"), "cp": ("dimer", "mA_cp", "mB_cp")}
MAX_NL_PER_FRAG = 3

import ferric  # noqa: E402

old = json.load(open(CACHE))["entries"]


def exact_bs():
    """{key-string: exact float} for every b the F.5 rule can issue as a node."""
    out = {f"{11.0:.10g}": 11.0}
    lo_lim, hi_lim = S.RULES["b_limits"]
    wins, todo = set(), [tuple(S.RULES["b_window"])]
    while todo:
        lo, hi = todo.pop()
        if (lo, hi) in wins or lo < lo_lim / 4 or hi > hi_lim * 4:
            continue
        wins.add((lo, hi))
        todo += [(lo / 4, lo * 2), (hi / 2, hi * 4)]
    for lo, hi in wins:
        for x in S._cheb_bs(lo, hi, S.RULES["n_cheb"]):
            b = math.exp(x)
            out[f"{b:.10g}"] = b
    return out


EXACT_B = exact_bs()
pool = S.Pool(basis="adz", nlc=(50, 50), cache=None, fast_damping=True)
pool.add("a24", IDS)
rows = []


def cmp(kind, key, cached, fresh, build):
    rows.append(
        {
            "key": key,
            "kind": kind,
            "build": build,
            "cached": cached,
            "fresh": fresh,
            "abs_diff": abs(cached - fresh),
        }
    )


for i in IDS:
    for conv, build in (("ncp", 1), ("cp", 2)):
        for tag in TAGS[conv]:
            f = pool.frags[("a24", i, tag)]
            k = f"{f.name}|hf"
            if k in old:
                cmp("hf", k, old[k], f.hf(), build)
            for r0, s in SEAMS[conv]:
                k = f"{f.name}|att|{S._seam_key(r0, s)}"
                if k in old:
                    cmp("att", k, old[k], f.att(r0, s), build)
                if s is not None:
                    continue
                pre = f"{f.name}|nl|"
                tail = f"|0.0089|eq11|{r0:.8g}|L|fast"
                keys = [
                    x
                    for x in old
                    if x.startswith(pre)
                    and x.endswith(tail)
                    and x[len(pre) :].split("|")[0] in EXACT_B
                ]
                keys = sorted(keys)[:MAX_NL_PER_FRAG]
                bs = [EXACT_B[x[len(pre) :].split("|")[0]] for x in keys]
                fresh = f.nl([(b, 0.0089, "eq11", r0, None) for b in bs])
                for x, v in zip(keys, fresh):
                    cmp("nl", x, old[x], float(v), build)

worst = max(r["abs_diff"] for r in rows)
json.dump(
    {
        # a bare 40-hex sha trips the detect-secrets hook; keep it in a labelled string
        "ferric_build": f"{ferric.__build__['git_sha']} dirty={ferric.__build__['dirty']}",
        "cache": CACHE,
        "n": len(rows),
        "max_abs_diff_ha": worst,
        "pass_1e-10": worst <= 1e-10,
        "rows": rows,
    },
    sys.stdout,
    indent=1,
)
