"""Figure for section 27.2/27.3: water/cc-pVDZ atomic charges by scheme, and the
isotropic static polarizability split into atoms plus the charge-transfer remainder.

One closed-shell `run_pdep_rpa(properties=True)` call (exact J/K reference,
aux cc-pvdz-ri, C6 off). Writes charges_and_alpha.svg next to this file; the
SVG title records ferric.__build__. Run from the repository root:

    box-lease run --label book-fig -- uv run --no-sync python \
        book/src/figures/27-ml/charges_and_alpha.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import ferric  # noqa: E402

WATER = """3
water
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
"""

SCHEMES = [
    ("Mulliken", "mulliken_charges"),
    ("Löwdin", "lowdin_charges"),
    ("Hirshfeld", "hirshfeld_charges"),
    ("CHELPG", "chelpg_charges"),
    ("RESP", "resp_charges"),
]


def iso(t):
    return float(np.trace(np.asarray(t))) / 3.0


def main():
    mol = ferric.Molecule.from_xyz_string(WATER)
    res = ferric.run_pdep_rpa(
        mol,
        ferric.BasisSet.bundled("cc-pvdz"),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
        properties=True,
        compute_c6=False,
    )
    if res.property_gaps:
        raise SystemExit(f"property gaps: {res.property_gaps}")

    plt.rcParams.update({"svg.hashsalt": "ferric-book-27-ml", "font.size": 9})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.0), width_ratios=[1.6, 1.0])

    # Panel a: q_O and q_H per scheme (the two H are equivalent; atom 1 shown).
    x = np.arange(len(SCHEMES))
    q = np.array([np.asarray(getattr(res, attr)) for _, attr in SCHEMES])
    ax0.bar(x - 0.18, q[:, 0], 0.36, label="O", color="#c0392b")
    ax0.bar(x + 0.18, q[:, 1], 0.36, label="H", color="#2e86c1")
    ax0.axhline(0.0, color="black", lw=0.6)
    ax0.set_xticks(x, [name for name, _ in SCHEMES])
    ax0.set_ylabel("charge / e")
    ax0.set_title("a) water/cc-pVDZ atomic charges", loc="left")
    ax0.legend(frameon=False)

    # Panel b: alpha_iso = alpha_O + 2 alpha_H + alpha_CT (stacked).
    a_at = np.asarray(res.alpha_atomic)
    parts = [
        ("O (intrinsic)", iso(a_at[0]), "#c0392b"),
        ("H + H (intrinsic)", iso(a_at[1]) + iso(a_at[2]), "#2e86c1"),
        ("charge transfer", iso(res.alpha_ct), "#7f8c8d"),
    ]
    bottom = 0.0
    for label, val, color in parts:
        ax1.bar([0], [val], 0.6, bottom=bottom, label=label, color=color)
        bottom += val
    ax1.plot(
        [-0.3, 0.3],
        [iso(res.alpha_tensor)] * 2,
        color="black",
        lw=1.2,
        label="molecular α (alpha_tensor)",
    )
    ax1.set_xlim(-0.45, 1.9)
    ax1.set_xticks([])
    ax1.set_ylabel("isotropic static α / bohr³")
    ax1.set_title("b) α split into atoms + CT", loc="left")
    ax1.legend(frameon=False, fontsize=7, loc="center right")

    fig.tight_layout()
    b = ferric.__build__ or {}
    title = (
        "ferric book figure 27-ml/charges_and_alpha; ferric build "
        f"{b.get('git_sha', 'unknown')} dirty={b.get('dirty', 'unknown')}"
    )
    out = Path(__file__).with_suffix(".svg")
    fig.savefig(out, metadata={"Date": None, "Title": title})
    print(f"wrote {out} ({title})")


if __name__ == "__main__":
    main()
