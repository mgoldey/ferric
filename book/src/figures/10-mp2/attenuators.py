"""Short-range attenuators of chapter 10, as fractions of 1/r.

Writes attenuators.svg next to this file. Analytic only (no ferric run):

* erfc(omega r) at omega = 0.420 1/Angstrom, the att-rimp2 erfc default;
* terfc(r; r0) = 1 - (erf(w(r - r0)) + erf(w(r + r0)))/2 with the
  curvature link w = 1/(r0 sqrt 2) (ferric-integrals operator.rs
  Operator::terfc), at r0 = 1.05 Angstrom (the att-rimp2 terfc default and
  the SCS-MP2(2terfc) r0^(2)) and 0.75 Angstrom (its r0^(1));
* the MP2-V VV10 damping 1 - terfc(r; r0)^2 at r0 = 1.00 Angstrom
  (ferric-dft vv10.rs Vv10Damping::Terfc).

Run: python book/src/figures/10-mp2/attenuators.py
"""

from math import erf, erfc, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["svg.hashsalt"] = "ferric-book"


def terfc(r: float, r0: float) -> float:
    w = 1.0 / (r0 * sqrt(2.0))
    return 1.0 - 0.5 * (erf(w * (r - r0)) + erf(w * (r + r0)))


def main() -> None:
    rs = [6.0 * k / 600 for k in range(601)]  # Angstrom
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(
        rs,
        [erfc(0.420 * r) for r in rs],
        label=r"erfc($\omega r$), $\omega$ = 0.420 Å$^{-1}$",
    )
    ax.plot(rs, [terfc(r, 1.05) for r in rs], label=r"terfc($r; r_0$), $r_0$ = 1.05 Å")
    ax.plot(rs, [terfc(r, 0.75) for r in rs], label=r"terfc($r; r_0$), $r_0$ = 0.75 Å")
    ax.plot(
        rs,
        [1.0 - terfc(r, 1.00) ** 2 for r in rs],
        "--",
        color="0.35",
        label=r"MP2-V damping $1-\mathrm{terfc}(r; 1.00\,\mathrm{Å})^2$",
    )
    ax.set_xlabel("r / Å")
    ax.set_ylabel(
        "multiplier of 1/r (solid) or of the VV10 kernel (dashed)", fontsize=8
    )
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    out = Path(__file__).with_suffix(".svg")
    fig.savefig(out, metadata={"Date": None, "Title": "attenuators (analytic)"})


if __name__ == "__main__":
    main()
