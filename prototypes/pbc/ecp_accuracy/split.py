import numpy as np
import ecp_clean as E, kval as KV

full = [(l, ts) for l, ts in E.ECP_I]


def setecp(chs):
    KV.ams.clear()
    KV.ns.clear()
    KV.ez.clear()
    KV.dz.clear()
    for l, ts, scale in chs:
        for n, z, d in ts:
            KV.ams.append(l)
            KV.ns.append(n)
            KV.ez.append(z)
            KV.dz.append(d * scale)


sa = E.I_SH[3]
sb = E.I_SH[3]
pa = E.I0
pb = E.I0 + E.LV
pc = E.I0 + np.array([1.5, -1.2, 0.8])
e = np.eye(3)[0]
hs = np.arange(-20, 21) * 2.5e-5
cases = {
    "local only (type 1)": [(l, ts, 1.0 if l == 3 else 0.0) for l, ts in full],
    "projectors only (type 2)": [(l, ts, 0.0 if l == 3 else 1.0) for l, ts in full],
}
for lp in (0, 1, 2):
    cases[f"projector l={lp} only"] = [
        (l, ts, 1.0 if l == lp else 0.0) for l, ts in full
    ]
for name, chs in cases.items():
    setecp(chs)
    v = np.array([KV.val(sa, pa, sb, pb, pc + h * e, 1e-15, 256, 1024) for h in hs])
    worst = 0
    for i in range(v.shape[1]):
        for j in range(v.shape[2]):
            c = np.polyfit(hs, v[:, i, j], 3)
            worst = max(worst, abs(v[:, i, j] - np.polyval(c, hs)).max())
    print(f"{name}: max jitter over block {worst:.2e}, max |V| {abs(v).max():.2e}")
