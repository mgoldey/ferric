#!/usr/bin/env python3
"""Figure 4.1: the erf, erfc, terf and terfc kernels in r and in k.

Analytic expressions only (crates/ferric-integrals/theory.md, "The terf and
terfc kernels" and "Kernels in Fourier space"). Writes kernels_rk.svg next to
this script. Deterministic: fixed svg.hashsalt and no date in the metadata.

    python book/src/figures/04-operators/kernels_rk.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import erf  # noqa: E402

plt.rcParams["svg.hashsalt"] = "ferric-book-kernels"
plt.rcParams["svg.fonttype"] = "none"

R0 = 2.0  # Bohr
W = 1.0 / (R0 * np.sqrt(2.0))  # curvature constraint, r0 * omega = 1/sqrt(2)


def terf(r):
    return 0.5 * (erf(W * (r - R0)) + erf(W * (r + R0)))


r = np.linspace(1e-6, 10.0, 800)
k = np.linspace(1e-3, 4.0, 800)

fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(9.0, 3.4))

ax_r.plot(r, erf(W * r), label=r"erf$(\omega r)$", color="#1f77b4")
ax_r.plot(r, 1 - erf(W * r), label=r"erfc$(\omega r)$", color="#1f77b4", ls="--")
ax_r.plot(r, terf(r), label=r"terf$(r; r_0, \omega)$", color="#d62728")
ax_r.plot(r, 1 - terf(r), label=r"terfc$(r; r_0, \omega)$", color="#d62728", ls="--")
ax_r.axvline(R0, color="0.6", lw=0.8, ls=":")
ax_r.set_xlabel(r"$r$ (Bohr)")
ax_r.set_ylabel(r"$r\,K(r)$")
ax_r.set_title(r"Kernels times $r$", fontsize=10)
ax_r.legend(fontsize=8, frameon=False)

damp = np.exp(-(k**2) / (4 * W**2))
ax_k.plot(k, damp, label="erf", color="#1f77b4")
ax_k.plot(k, 1 - damp, label="erfc", color="#1f77b4", ls="--")
ax_k.plot(k, np.cos(k * R0) * damp, label="terf", color="#d62728")
ax_k.plot(k, 1 - np.cos(k * R0) * damp, label="terfc", color="#d62728", ls="--")
ax_k.axhline(0.0, color="0.3", lw=0.6)
ax_k.axhline(1.0, color="0.6", lw=0.6, ls=":")
ax_k.axvline(np.pi / (2 * R0), color="0.6", lw=0.8, ls=":")
ax_k.annotate(
    r"$k r_0 = \pi/2$",
    (np.pi / (2 * R0), -0.12),
    xytext=(1.25, -0.2),
    fontsize=8,
)
ax_k.set_xlabel(r"$k$ (Bohr$^{-1}$)")
ax_k.set_ylabel(r"$\hat K(k)\,/\,(4\pi/k^2)$")
ax_k.set_title("Fourier transforms over the Coulomb one", fontsize=10)
ax_k.legend(fontsize=8, frameon=False)

fig.suptitle(rf"$r_0 = {R0:g}$ Bohr, $\omega = 1/(r_0\sqrt{{2}})$", fontsize=10, y=1.0)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".svg"), metadata={"Date": None})
