import json, sys, numpy as np

d = sys.argv[1]
m = json.load(open(f"{d}/mol.json"))
gm = np.array(m["g"])
rows = [json.loads(l) for l in open(f"{d}/box.jsonl")]
C3, C3P = -70.95560087, 9.700799387772
print("| a | r = d - c3'/a^3 (H z) | a^5 r | rE = dE - c3/a^3 | a^5 rE |")
print("|---|---|---|---|---|")
A = []
R = []
RE = []
for r in rows:
    a = r["a"]
    dd = np.array(r["g"])[0, 2] - gm[0, 2]
    res = dd - C3P / a**3
    de = r["e"] - m["e"]
    re = de - C3 / a**3
    A.append(a)
    R.append(res)
    RE.append(re)
    print(
        f"| {a:.0f} | {res:+.4e} | {res * a**5:+.2f} | {re:+.4e} | {re * a**5:+.2f} |"
    )
A = np.array(A)
R = np.array(R)
RE = np.array(RE)
for lab, Y in (("force", R), ("energy", RE)):
    for amin in (16, 20, 24):
        for cols in ([5, 7], [0, 5], [0, 5, 7], [0, 3, 5]):
            s = A >= amin
            if s.sum() <= len(cols):
                continue
            X = np.column_stack([A[s] ** (-p) if p else np.ones(s.sum()) for p in cols])
            c, *_ = np.linalg.lstsq(X, Y[s], rcond=None)
            res = Y[s] - X @ c
            print(
                lab,
                f"c3' fixed, a>={amin} n={s.sum()} a^-{cols}: "
                + " ".join(f"c{p}={v:+.4e}" for p, v in zip(cols, c))
                + f" max|res| {abs(res).max():.1e}",
            )
