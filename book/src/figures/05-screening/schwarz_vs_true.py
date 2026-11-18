#!/usr/bin/env python3
"""Figure 5.1: the Schwarz and QQR bounds against the true integral.

Two primitive s pair densities, unit charge, exponents P1 = P2 = 2 Bohr^-2,
at centre distance R. All quantities are closed forms
(section @sec:schwarz-bound-admitted-kernels and
"Distance-dependent bounds (QQR)"):

    (12|K|12)    = erf(phi R)/R                   (Coulomb: phi = theta)
    Q1 Q2        = (11|K|11)^1/2 (22|K|22)^1/2     (independent of R)
    QQR envelope = min(1, 1.10 (e1 + e2) / max(0, R - e1 - e2)),  e = P^-1/2

for the Coulomb kernel and erfc(omega r)/r at omega = 0.5 Bohr^-1. Writes
schwarz_vs_true.svg next to this script. Deterministic.

    python book/src/figures/05-screening/schwarz_vs_true.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import erf  # noqa: E402

plt.rcParams["svg.hashsalt"] = "ferric-book-schwarz"
plt.rcParams["svg.fonttype"] = "none"

P1 = P2 = 2.0
OMEGA = 0.5
QQR_SAFETY = 1.10


def coulomb_ss(p, q, R):
    """(rho_p|1/r|rho_q) for unit-charge s Gaussians."""
    th = np.sqrt(p * q / (p + q))
    return np.where(R > 0, erf(th * R) / np.maximum(R, 1e-300), 2 * th / np.sqrt(np.pi))


def erf_ss(p, q, R, w):
    ph = 1.0 / np.sqrt(1 / p + 1 / q + 1 / w**2)
    return np.where(R > 0, erf(ph * R) / np.maximum(R, 1e-300), 2 * ph / np.sqrt(np.pi))


def erfc_ss(p, q, R, w):
    return coulomb_ss(p, q, R) - erf_ss(p, q, R, w)


R = np.linspace(0.0, 12.0, 600)
e1, e2 = P1**-0.5, P2**-0.5
reff = np.maximum(0.0, R - e1 - e2)
env = np.minimum(1.0, QQR_SAFETY * (e1 + e2) / np.maximum(reff, 1e-300))
env[reff == 0] = 1.0

fig, ax = plt.subplots(figsize=(6.0, 3.6))
for name, true, q1q2, color in (
    (
        "Coulomb",
        coulomb_ss(P1, P2, R),
        np.sqrt(coulomb_ss(P1, P1, 0.0) * coulomb_ss(P2, P2, 0.0)),
        "#1f77b4",
    ),
    (
        rf"erfc, $\omega = {OMEGA:g}$",
        erfc_ss(P1, P2, R, OMEGA),
        np.sqrt(erfc_ss(P1, P1, 0.0, OMEGA) * erfc_ss(P2, P2, 0.0, OMEGA)),
        "#d62728",
    ),
):
    ax.semilogy(R, true, color=color, label=f"{name}: true")
    ax.semilogy(
        R, np.full_like(R, q1q2), color=color, ls="--", label=f"{name}: Schwarz"
    )
    ax.semilogy(R, q1q2 * env, color=color, ls=":", label=f"{name}: QQR")
ax.set_ylim(1e-12, 3.0)
ax.set_xlabel(r"$R$ (Bohr)")
ax.set_ylabel(r"$|(12|12)|$ (Hartree)")
ax.set_title(r"Two $s$ pair densities, $P_1 = P_2 = 2$", fontsize=10)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".svg"), metadata={"Date": None})
