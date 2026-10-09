#!/usr/bin/env python3
"""Issue #280 step 1: how far do the lowest 5 BSE-TDA Omega move when the
ill-conditioned core/high-virtual QP energies are re-drawn (Q1) or replaced by
a windowed-QP + scissor recipe (Q2)?

Pure numpy on the STORED, QP-independent kernel `bse_kernel` -- no Rust build.
"""

import base64
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT / "testdata/reference/validation/bse"
SYSTEMS = ["h2o_cc-pvdz", "nh3_cc-pvdz", "ch2o_cc-pvdz", "h2o_aug-cc-pvdz"]
HA_EV = 27.211386245988


def load(sys_name):
    return json.loads((REF / f"{sys_name}.json").read_text())


def decode_kernel(d):
    nov = int(d["bse_kernel"]["nov"])
    raw = np.frombuffer(
        base64.b64decode(d["bse_kernel"]["upper_f64le_b64"]), dtype="<f8"
    )
    k = np.zeros((nov, nov))
    iu = np.triu_indices(nov)
    assert raw.size == iu[0].size, (raw.size, iu[0].size)
    k[iu] = raw
    k = k + k.T - np.diag(np.diag(k))
    return k


def amat(kernel, eps_qp, nocc, nvir):
    """A = K + diag(eps_a - eps_i), flat ia = i*nvir + a."""
    a = kernel.copy()
    for i in range(nocc):
        for b in range(nvir):
            a[i * nvir + b, i * nvir + b] += eps_qp[nocc + b] - eps_qp[i]
    return a


def omegas(kernel, eps_qp, nocc, nvir, n=5):
    w = np.linalg.eigvalsh(amat(kernel, eps_qp, nocc, nvir))
    return np.sort(w)[:n]


def windowed_eps(eps_mf, eps_qp, nocc, nmo, k):
    """QP inside HOMO-k..LUMO+k, rigid scissor outside.

    Window occupied: nocc-1-k .. nocc-1 (clamped at 0)
    Window virtual : nocc .. nocc+k    (clamped at nmo-1)
    Below the window: occupied p < lo get eps_mf[p] + (window's LOWEST occupied
    QP correction).  Above: virtual p > hi get eps_mf[p] + (window's HIGHEST
    virtual QP correction).
    """
    lo = max(0, nocc - 1 - k)
    hi = min(nmo - 1, nocc + k)
    corr = np.asarray(eps_qp) - np.asarray(eps_mf)
    shift_occ = corr[lo]
    shift_vir = corr[hi]
    out = np.asarray(eps_mf, dtype=float).copy()
    out[lo : hi + 1] = np.asarray(eps_qp)[lo : hi + 1]
    out[:lo] += shift_occ
    out[hi + 1 :] += shift_vir
    return out, lo, hi


def main():
    rows = []
    print("=" * 100)
    print(
        "ANCHOR: stored kernel + stored eps_qp must reproduce stored bse_singlet.omega"
    )
    print("=" * 100)
    anchor_ok = True
    data = {}
    for s in SYSTEMS:
        d = load(s)
        k = decode_kernel(d)
        nocc, nvir = int(d["nocc"]), int(d["nvir"])
        eps_qp = np.array(d["qp"]["eps_qp"])
        eps_mf = np.array(d["qp"]["eps_mf"])
        nmo = eps_qp.size
        assert nocc + nvir == nmo, (nocc, nvir, nmo)
        n_ref = len(d["bse_singlet"]["omega"])
        got = omegas(k, eps_qp, nocc, nvir, n=n_ref)
        ref = np.array(d["bse_singlet"]["omega"])
        dev = np.abs(got - ref).max()
        ok = dev <= 1e-12
        anchor_ok &= ok
        print(
            f"  {s:20s} nov={k.shape[0]:4d} n_ref={n_ref}  max|d| = {dev:.3e}  {'OK' if ok else 'FAIL'}"
        )
        data[s] = {
            "d": d,
            "k": k,
            "nocc": nocc,
            "nvir": nvir,
            "nmo": nmo,
            "eps_qp": eps_qp,
            "eps_mf": eps_mf,
        }
    if not anchor_ok:
        print(
            "\nANCHOR FAILED -- refusing to run the sweep (artifact hypothesis H-A1)."
        )
        return 1
    print("\nAnchor passes on all four systems. Proceeding.\n")

    # ---- vacuous-limit anchor for the window itself (H-A2) ----
    print("=" * 100)
    print(
        "WINDOW VACUOUS LIMIT: k >= nmo must reproduce all-MO exactly (0.0), k=2 must NOT"
    )
    print("=" * 100)
    for s in SYSTEMS:
        z = data[s]
        a_om = omegas(z["k"], z["eps_qp"], z["nocc"], z["nvir"])
        big, lo, hi = windowed_eps(
            z["eps_mf"], z["eps_qp"], z["nocc"], z["nmo"], z["nmo"]
        )
        assert lo == 0 and hi == z["nmo"] - 1, (lo, hi)
        assert np.array_equal(big, z["eps_qp"]), (
            "vacuous window must be bit-identical eps_qp"
        )
        b_om = omegas(z["k"], big, z["nocc"], z["nvir"])
        d_big = np.abs(a_om - b_om).max()
        w2, _, _ = windowed_eps(z["eps_mf"], z["eps_qp"], z["nocc"], z["nmo"], 2)
        d_k2 = np.abs(a_om - omegas(z["k"], w2, z["nocc"], z["nvir"])).max()
        print(
            f"  {s:20s} k=nmo: max|dOmega| = {d_big:.3e} (want 0)   k=2: {d_k2:.3e} (want >0)"
        )
        assert d_big == 0.0, "window at k=nmo is not a no-op -> H-A2 bug"
        assert d_k2 > 0.0, "window at k=2 is a no-op -> H-A2 bug"
    print(
        "\nVacuous-limit anchor passes; the window is neither a no-op nor applied to all.\n"
    )

    # ---- Q2: scissor vs all-MO ----
    print("=" * 100)
    print("Q2 (method change): lowest 5 Omega, all-MO (a) vs windowed+scissor (b)")
    print("=" * 100)
    q2 = {}
    for s in SYSTEMS:
        z = data[s]
        a_om = omegas(z["k"], z["eps_qp"], z["nocc"], z["nvir"])
        print(f"\n  {s}  (nocc={z['nocc']} nvir={z['nvir']} nmo={z['nmo']})")
        print("    (a) all-MO QP   Omega(Ha): " + "  ".join(f"{x:.8f}" for x in a_om))
        print(
            "                       (eV)  : "
            + "  ".join(f"{x * HA_EV:8.4f}" for x in a_om)
        )
        q2[s] = {}
        for kk in (2, 4, 8):
            w, lo, hi = windowed_eps(z["eps_mf"], z["eps_qp"], z["nocc"], z["nmo"], kk)
            b_om = omegas(z["k"], w, z["nocc"], z["nvir"])
            dd = b_om - a_om
            q2[s][kk] = {
                "om": b_om,
                "d": dd,
                "lo": lo,
                "hi": hi,
                "nsolved": hi - lo + 1,
                "nshift": z["nmo"] - (hi - lo + 1),
            }
            print(
                f"    (b) k={kk:<2d} MOs {lo}..{hi} solved ({hi - lo + 1}/{z['nmo']}), "
                f"{z['nmo'] - (hi - lo + 1)} shifted"
            )
            print("           Omega(Ha): " + "  ".join(f"{x:.8f}" for x in b_om))
            print(
                "           dOmega(Ha): "
                + "  ".join(f"{x:+.2e}" for x in dd)
                + f"   max|d| = {np.abs(dd).max():.3e} Ha = {np.abs(dd).max() * HA_EV:.4f} eV"
            )

    # ---- Q1: conditioning Monte Carlo ----
    print("\n" + "=" * 100)
    print("Q1 (conditioning): re-draw each QP energy within its MEASURED sensitivity,")
    print("                   200 draws, report max |dOmega| over draws for lowest 5")
    print("=" * 100)
    rng = np.random.default_rng(20261003)
    NDRAW = 200
    q1 = {}
    for s in SYSTEMS:
        z = data[s]
        sens = np.array(z["d"]["qp"]["sensitivity"])
        a_om = omegas(z["k"], z["eps_qp"], z["nocc"], z["nvir"])
        worst = np.zeros(5)
        # also: Q1 restricted to perturbing ONLY the out-of-window MOs (k=2),
        # which is exactly the set the window would remove.
        lo2 = max(0, z["nocc"] - 1 - 2)
        hi2 = min(z["nmo"] - 1, z["nocc"] + 2)
        mask_out = np.ones(z["nmo"], dtype=bool)
        mask_out[lo2 : hi2 + 1] = False
        worst_out = np.zeros(5)
        for _ in range(NDRAW):
            # sensitivity is a max-|d| over draws, so use it as a uniform radius
            pert = sens * rng.uniform(-1.0, 1.0, size=z["nmo"])
            worst = np.maximum(
                worst,
                np.abs(omegas(z["k"], z["eps_qp"] + pert, z["nocc"], z["nvir"]) - a_om),
            )
            pert_out = np.where(mask_out, pert, 0.0)
            worst_out = np.maximum(
                worst_out,
                np.abs(
                    omegas(z["k"], z["eps_qp"] + pert_out, z["nocc"], z["nvir"]) - a_om
                ),
            )
        q1[s] = {"all": worst, "out": worst_out}
        print(f"\n  {s}")
        print(
            "    max|dOmega| over draws, ALL MOs perturbed  (Ha): "
            + "  ".join(f"{x:.2e}" for x in worst)
            + f"   -> {worst.max() * HA_EV:.2e} eV worst"
        )
        print(
            "    max|dOmega| over draws, only MOs OUTSIDE k=2 (Ha): "
            + "  ".join(f"{x:.2e}" for x in worst_out)
            + f"   -> {worst_out.max() * HA_EV:.2e} eV worst"
        )

    # ---- eigenvector amplitude on the ill-conditioned rows (explains Q1) ----
    print("\n" + "=" * 100)
    print(
        "WHY: summed |X_n(ia)|^2 of the lowest 5 states on rows whose MO sensitivity > 1e-3"
    )
    print("=" * 100)
    for s in SYSTEMS:
        z = data[s]
        sens = np.array(z["d"]["qp"]["sensitivity"])
        a = amat(z["k"], z["eps_qp"], z["nocc"], z["nvir"])
        w, v = np.linalg.eigh(a)
        order = np.argsort(w)[:5]
        nocc, nvir = z["nocc"], z["nvir"]
        bad = sens > 1e-3
        rows = np.array(
            [bad[i] or bad[nocc + b] for i in range(nocc) for b in range(nvir)]
        )
        amp = [float((v[:, j][rows] ** 2).sum()) for j in order]
        print(
            f"  {s:20s} nbad_MO={int(bad.sum()):2d}/{z['nmo']}  nbad_rows={int(rows.sum()):4d}/{rows.size}"
            f"   sum|X|^2 on bad rows: " + "  ".join(f"{x:.2e}" for x in amp)
        )

    # ---- summary table ----
    print("\n" + "=" * 100)
    print("SUMMARY (lowest-5 worst movement, Ha)")
    print("=" * 100)
    print(
        f"{'system':20s} {'Q1 all':>10s} {'Q1 out-k2':>10s} {'Q2 k=2':>10s} {'Q2 k=4':>10s} {'Q2 k=8':>10s}"
    )
    for s in SYSTEMS:
        print(
            f"{s:20s} {q1[s]['all'].max():10.2e} {q1[s]['out'].max():10.2e} "
            f"{np.abs(q2[s][2]['d']).max():10.2e} {np.abs(q2[s][4]['d']).max():10.2e} "
            f"{np.abs(q2[s][8]['d']).max():10.2e}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
