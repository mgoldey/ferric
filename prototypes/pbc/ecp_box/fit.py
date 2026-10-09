import json, sys, numpy as np

d = sys.argv[1]
m = json.load(open(f"{d}/mol.json"))
gm = np.array(m["g"])
rows = [json.loads(l) for l in open(f"{d}/box.jsonl")]
C3, C3P = -70.95560087, 9.700799387772
print(
    f"| a (Bohr) | nG half | SCF it | E_box - E_mol | a^3 dE | g_box H z | d = g_box - g_mol (H z) | a^3 d | d_H + d_I | max transverse | wall s |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    g = np.array(r["g"])
    dd = g - gm
    a = r["a"]
    print(
        f"| {a:.0f} | {r['nG']} | {r['iters']} | {r['e'] - m['e']:+.9e} | {(r['e'] - m['e']) * a**3:+.5f} | {g[0, 2]:+.10e} | {dd[0, 2]:+.6e} | {dd[0, 2] * a**3:+.6f} | {dd[0, 2] + dd[1, 2]:+.1e} | {abs(g[:, :2]).max():.1e} | {r['wall']:.0f} |"
    )


def fit(amin, cols, key="f"):
    sel = [r for r in rows if r["a"] >= amin]
    a = np.array([r["a"] for r in sel])
    if key == "f":
        y = np.array([np.array(r["g"])[0, 2] - gm[0, 2] for r in sel])
    else:
        y = np.array([r["e"] - m["e"] for r in sel])
    X = np.column_stack([a ** (-p) if p else np.ones_like(a) for p in cols])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ c
    return c, np.abs(res).max(), len(a)


for key in ("f", "e"):
    for amin in (16, 20, 24):
        for cols in ([3, 5], [0, 3, 5], [0, 3, 5, 7]):
            c, r, n = fit(amin, cols, key)
            if n <= len(cols):
                continue
            print(
                key,
                f"a>={amin} n={n} terms a^-{cols}: "
                + " ".join(f"c{p}={v:+.6e}" for p, v in zip(cols, c))
                + f"  max|res| {r:.1e}",
            )
print("pred c3", C3, "c3'", C3P)
