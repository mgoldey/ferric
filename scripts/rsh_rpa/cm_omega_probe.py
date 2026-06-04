#!/usr/bin/env python3
"""Parameter-free Clausius-Mossotti omega(alpha/V) probe for dielectric-RSH-RPA-C6.
RESULT: CM dielectric catastrophes (y=(4pi/3)(alpha/V) > 1 for half the set ->
negative eps) — CM is invalid at molecular polarizability density. The dielectric
RESPONSE alpha/V is the right variable; the CM CLOSURE is wrong physics. A
parameter-free omega needs a proper finite-system dielectric (Brawand SX /
compressibility sum rule), not the classical CM formula.
Data: ferric runs 2026-06-04 (LC-wPBE/aug-cc-pVDZ)."""
import numpy as np
data = [("h2o",5.901,27.05,0.1759),("ch4",10.443,34.40,0.1255),
        ("nh3",8.529,32.43,0.1035),("co",8.348,46.35,0.3651),
        ("n2",7.886,44.80,0.4640),("co2",11.315,59.20,0.3712),
        ("c2h4",16.899,61.08,0.2267)]
a=np.array([d[1] for d in data]); V=np.array([d[2] for d in data])
wopt=np.array([d[3] for d in data]); aoV=a/V
y=(4*np.pi/3)*aoV; eps=(1+2*y)/(1-y)
for i,d in enumerate(data):
    flag="  <-- CM CATASTROPHE (y>1, eps<0)" if y[i]>1 else ""
    print(f"{d[0]:5} a/V={aoV[i]:.4f} y={y[i]:.3f} eps={eps[i]:8.2f}{flag}")
print(f"\n{(y>1).sum()}/{len(y)} molecules past the CM stability limit -> parameter-free CM omega INVALID.")
