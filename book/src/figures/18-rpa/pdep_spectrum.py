"""Figure 18.3: the static PDEP spectrum and what truncation costs.

Water / cc-pVDZ, aux cc-pvdz-ri, RHF reference, all electrons, default
quadrature. Left: |lambda_xi(0) - 1| of the static dielectric eigenvalues,
sorted, with the default trunc_thresh. Right: the error of the truncated PDEP
energy against trunc_thresh = 0 versus the number of modes kept.

Writes pdep_spectrum.svg next to this file and records ferric.__build__ in
its title. Run (about ten seconds):
  box-lease run --label book-fig -- uv run --no-sync python book/src/figures/18-rpa/pdep_spectrum.py
"""
# SPDX-License-Identifier: MIT OR Apache-2.0

from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

import ferric  # noqa: E402

matplotlib.rcParams["svg.hashsalt"] = "ferric-book"

ROOT = Path(__file__).resolve().parents[4]
WATER = ROOT / "testdata" / "molecules" / "water.xyz"
THRESHOLDS = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5]
DEFAULT_TRUNC = 1e-4


def run(trunc: float):
    return ferric.run_pdep_rpa(
        ferric.Molecule.from_xyz(str(WATER)),
        ferric.BasisSet.bundled("cc-pvdz"),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
        trunc_thresh=trunc,
    )


def main() -> None:
    full = run(0.0)
    lam = sorted((abs(x - 1.0) for x in full.eigenvalues_static), reverse=True)
    if len(lam) == 0:
        raise SystemExit("trunc_thresh = 0 returned no static eigenvalues")
    rows = []
    for t in THRESHOLDS:
        r = run(t)
        rows.append((t, r.n_eigenpotentials, r.e_rpa - full.e_rpa))
    for t, n, d in rows:
        print(f"trunc {t:8.1e}  kept {n:4d}  dE {d:+.3e} Ha")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.4))
    a1.semilogy(range(1, len(lam) + 1), lam, ".", ms=3)
    a1.axhline(DEFAULT_TRUNC, color="0.5", lw=0.8, ls="--")
    a1.text(len(lam) * 0.55, DEFAULT_TRUNC * 1.6, "default trunc_thresh", fontsize=8)
    a1.set_xlabel(r"mode $\xi$ (sorted)")
    a1.set_ylabel(r"$|\lambda_\xi(0) - 1|$")
    a1.set_title(f"static spectrum, $N_{{aux}}$ = {len(lam)}", fontsize=10)

    a2.semilogy(
        [n for _, n, _ in rows], [max(abs(d), 1e-16) for _, _, d in rows], "o-", ms=4
    )
    for t, n, d in rows:
        a2.annotate(
            f"{t:.0e}",
            (n, max(abs(d), 1e-16)),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    a2.set_xlabel("modes kept")
    a2.set_ylabel(r"$|E_c(\mathrm{trunc}) - E_c(0)|$ (Ha)")
    a2.set_title("truncation error (labels: trunc_thresh)", fontsize=10)
    fig.tight_layout()
    b = ferric.__build__
    title = (
        f"pdep_spectrum.py; ferric {b['git_sha'][:12]}{' dirty' if b['dirty'] else ''}"
    )
    fig.savefig(
        Path(__file__).with_suffix(".svg"), metadata={"Date": None, "Title": title}
    )


if __name__ == "__main__":
    main()
