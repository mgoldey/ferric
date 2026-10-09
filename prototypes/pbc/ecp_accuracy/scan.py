import numpy as np, sys

sys.argv = ["x"]
import ecp_clean as E

pa = E.I0
sb = E.I_SH[3]
sa = E.I_SH[3]
pb = E.I0 + E.LV
pc = E.I0 + np.array([1.5, -1.2, 0.8])
e = np.eye(3)[0]
hs = np.arange(-20, 21) * 2.5e-5
vl = np.array([E.lib_value(sa, pa, sb, pb, pc + h * e)[0, 2] for h in hs])
vp = np.array([E.pyscf_value(sa, pa, sb, pb, pc + h * e)[0, 2] for h in hs])
# remove quadratic fit, show residuals
for name, v in (("lib", vl), ("pyscf", vp)):
    c = np.polyfit(hs, v, 3)
    r = v - np.polyval(c, hs)
    print(name, "cubic-fit residual max %.2e" % abs(r).max(), " slope at 0 %.8e" % c[2])
    print("  resid:", " ".join("%+.1e" % x for x in r[::2]))
print("lib - pyscf values:", " ".join("%+.2e" % x for x in (vl - vp)[::4]))
