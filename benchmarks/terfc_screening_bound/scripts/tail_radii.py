"""Stage (a): exact tail radius R(tau) = K^{-1}(tau) of K(r)=att(r)/r, monotone decreasing.
Inputs r0 [Angstrom], omega [Angstrom^-1]; K in Hartree (1/Bohr); R reported in Bohr (Angstrom).
Also prints the estimate r0 + sqrt(ln(1/tau))/omega (Angstrom) for comparison."""

import numpy as np
from scipy.special import erfc
from scipy.optimize import brentq

ANG = 1.8897261246257702  # Bohr per Angstrom


def terfc(r, r0, w):  # r, r0 Bohr, w Bohr^-1;  == 1-[erf(w(r-r0))+erf(w(r+r0))]/2
    return 0.5 * (erfc(w * (r - r0)) + erfc(w * (r + r0)))


OPS = {
    "linked(1.0,s=.7071)": ("terfc", 1.0, 0.70710678),
    "NCP(0.991,s=1.354)": ("terfc", 0.991, 1.354 / 0.991),
    "CP(1.408,s=1.164)": ("terfc", 1.408, 1.164 / 1.408),
    "s=2,r0=1": ("terfc", 1.0, 2.0),
    "s=4,r0=1": ("terfc", 1.0, 4.0),
    "erfc0.42": ("erfc", None, 0.42),
    "coulomb": ("coul", None, None),
}
taus = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]


def K(r, spec):
    kind, r0, w = spec
    if kind == "terfc":
        return terfc(r, r0 * ANG, w / ANG) / r
    if kind == "erfc":
        return erfc(w / ANG * r) / r
    return 1.0 / r


if __name__ == "__main__":
    print(
        "K(R)=tau tail radius, Bohr (Angstrom); est = r0+sqrt(ln(1/tau))/omega in Angstrom"
    )
    print("%-22s" % "operator" + "".join("%20s" % ("tau=%g" % t) for t in taus))
    for name, spec in OPS.items():
        row = []
        for t in taus:
            R = (
                brentq(lambda r: K(r, spec) - t, 1e-3, 1e7)
                if spec[0] != "coul"
                else 1 / t
            )
            row.append("%.2f (%.2f)" % (R, R / ANG))
        print("%-22s" % name + "".join("%20s" % x for x in row))
        if spec[0] == "terfc":
            est = [spec[1] + np.sqrt(np.log(1 / t)) / spec[2] for t in taus]
            print("%-22s" % "  estimate (A)" + "".join("%20.2f" % x for x in est))
