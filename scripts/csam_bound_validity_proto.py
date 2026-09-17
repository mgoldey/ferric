#!/usr/bin/env python3
"""Is CSAM screening a rigorous upper bound? (Answer: no.)

Standalone adjudication of the CSAM screening bound, written to settle whether
ferric's failing `csam_is_valid_upper_bound_*` test was a PORT BUG or an
inherent property of CSAM.

It is deliberately INDEPENDENT of ferric and of libint2: two-electron integrals
over s-type Gaussians are evaluated from the closed-form Boys-F0 expression
below, in plain Python. It implements *Psi4's own* formula
(`psi4/src/psi4/libmints/twobody.cc`, `shell_significant_csam` + the
`shell_pair_exchange_values_` build loop), not ferric's, so a violation here
cannot be blamed on ferric's port.

Controls (both essential — without them this script proves nothing):

  1. The SAME harness evaluates the plain Schwarz bound, which must come out at
     worst-ratio exactly 1.000000 (saturated by the diagonal quartet). If it
     does not, the harness is broken and no CSAM conclusion follows.
  2. ferric's X formula is compared entry-by-entry against Psi4's, to check the
     direction of the known discrepancy (ferric takes max(numerator) over
     min(denominator) rather than max of the per-function ratios).

Measured result (2026-09-14), worst |true| / estimate:

    geometry                     Schwarz      Psi4-CSAM
    concentric s-shells          1.000000     1.786582
    two centers                  1.000000     2.532762
    multi-function shells        1.000000     1.388891

  erfc(omega r)/r, Psi4-CSAM:  1.82 (w=0.11) .. 9.37 (w=1.0)

########################################################################
# READ THIS BEFORE QUOTING ANY erfc NUMBER FROM THIS SCRIPT (2026-09-14)
########################################################################

This script's erfc rows were used to justify refusing attenuated operators
outright, with a "growth with molecular extent" table running
5.3x -> 49x -> 1.9e3 -> 2.7e5 -> 1.0e8 across a 2-to-6-center chain. **Those
numbers do not support that conclusion**, for two reasons that are properties
of THIS HARNESS, not of CSAM:

1. A RATIO IS NOT AN ENERGY ERROR, and nothing here bounds the latter.
   `audit()` maximizes `true/estimate` over every quartet, guarded only by
   `csam > 1e-300`. A ratio grows without bound; an energy error does not.
   The source paper restricts its own Table I/II statistics to "shell-quartets
   with exact norms above 1e-12" for this reason, and reports the ENERGY
   consequence separately (Tables III and V). This script never computed an
   energy error for any kernel, so no claim about "accuracy loss" — 1e5x or
   otherwise — was ever supported by it.

   `--min-magnitude` (added 2026-09-14) imposes the paper's filter. MEASURED,
   so the correction is not overstated: at the paper's own 1e-12 cut the
   chain-study numbers below are UNCHANGED (5.316 / 49.389 / 1878.821 /
   272514.949 / 101696864.802), and only at 1e-8 does the 6-center row fall
   back to the 5-center value. So the missing filter is a real methodological
   gap, but it is NOT what generates these particular figures — point 2 is.

2. erfc IS FORMED BY SUBTRACTION IN FLOAT64. `eri_erfc` returns
   `Coulomb - erf`, which is precisely the catastrophic cancellation the
   paper warns about on the same page ("Such integrals are calculated as the
   small difference between large values"). Verified against 200-bit mpmath on
   this harness's own two-s-Gaussian geometry at omega=1.0:

       separation(bohr)    float64        mpmath(200b)    rel. error
              0.00       1.477463e+01     1.477463e+01     6.4e-16
              2.64       3.105075e+00     3.105075e+00     2.1e-15
              5.28       3.057790e-02     3.057790e-02     6.9e-14
              7.92       1.646151e-05     1.646151e-05     1.2e-10
             10.56       5.642242e-10     5.642224e-10     3.2e-06
             13.20       7.105427e-15     1.299934e-15     4.5e+00  <-- 447%

   The 6-center row above (13.2 bohr extent) sits exactly where this harness's
   own `true` value is wrong by 447%. The 5-center row is contaminated at the
   1e-6 level. Those two rows measure IEEE754, not CSAM.

3. `Operator::erf` HAS NO PATH HERE AT ALL. Only Coulomb and
   erfc-by-subtraction are implemented — yet erf is the one attenuated kernel
   Psi4 actually screens with CSAM in production (`Libint2ErfERI` builds both
   its compute and its sieve engine on `libint2::Operator::erf_coulomb`,
   `psi4/src/psi4/libmints/eri.cc:179,182`). Refusing erf on the strength of
   this script was refusing an untested case.

WHAT SURVIVES: the COULOMB rows (1.39-2.53x). Those involve no cancellation
and no filter sensitivity at these magnitudes, and they do establish the one
thing this script was written to settle — CSAM is non-rigorous, and it is not
ferric's port that makes it so. For the ENERGY consequence of that non-rigor,
which this script never measured, see the paper's Tables III and V
(microhartree at theta=1e-10, NANOhartree at theta=1e-12).

`csam_x_table` now ROUTES erfc to the rigorous CSB bound — on the paper's own
recommendation for short-range operators, not on the strength of the numbers
above — and ACCEPTS erf. See `crates/ferric-integrals/src/csam.rs`'s module
header.

Run: python3 scripts/csam_bound_validity_proto.py    (no dependencies beyond numpy)
     python3 scripts/csam_bound_validity_proto.py --min-magnitude 1e-12
"""

import argparse
import itertools
import math

import numpy as np

# Quartets whose TRUE value is below this are excluded from the worst-ratio
# statistics. 0.0 reproduces the pre-2026-09-14 behaviour (no filter), which is
# what produced the 1e8 erfc figures this script's docstring retracts. The
# source paper uses 1e-12; pass `--min-magnitude 1e-12` to match it.
#
# This is deliberately a module global set once from argv rather than a
# parameter threaded through every helper: the helpers are a measurement
# harness, and a filter that some call sites silently skip would reintroduce
# exactly the inconsistency being corrected.
MIN_MAGNITUDE = 0.0


def boys_f0(t):
    """F_0(T) = int_0^1 exp(-T u^2) du, the only Boys function s-integrals need."""
    if t < 1e-14:
        return 1.0
    return 0.5 * math.sqrt(math.pi / t) * math.erf(math.sqrt(t))


def eri_coulomb(a, A, b, B, c, C, d, D):
    """(ab|cd) over UNNORMALIZED s-Gaussians exp(-a|r-A|^2), Coulomb kernel."""
    A, B, C, D = map(np.asarray, (A, B, C, D))
    p, q = a + b, c + d
    P, Q = (a * A + b * B) / p, (c * C + d * D) / q
    ab = np.dot(A - B, A - B)
    cd = np.dot(C - D, C - D)
    pq = np.dot(P - Q, P - Q)
    pref = 2 * math.pi**2.5 / (p * q * math.sqrt(p + q))
    k = math.exp(-a * b / p * ab - c * d / q * cd)
    return pref * k * boys_f0(p * q / (p + q) * pq)


def eri_erfc(a, A, b, B, c, C, d, D, w):
    """Short-range erfc(w r12)/r12 = Coulomb - erf(w r12)/r12.

    The erf-attenuated (long-range) part has the same closed form with
    rho -> rho w^2/(rho + w^2) and an extra sqrt(w^2/(rho+w^2)) prefactor.
    """
    A, B, C, D = map(np.asarray, (A, B, C, D))
    p, q = a + b, c + d
    P, Q = (a * A + b * B) / p, (c * C + d * D) / q
    ab = np.dot(A - B, A - B)
    cd = np.dot(C - D, C - D)
    pq = np.dot(P - Q, P - Q)
    pref = 2 * math.pi**2.5 / (p * q * math.sqrt(p + q))
    k = math.exp(-a * b / p * ab - c * d / q * cd)
    rho = p * q / (p + q)
    full = pref * k * boys_f0(rho * pq)
    scale = math.sqrt(w * w / (rho + w * w))
    long_range = pref * k * scale * boys_f0(rho * w * w / (rho + w * w) * pq)
    return full - long_range


def s_norm(a):
    """Normalization constant of an s-Gaussian: (2a/pi)^{3/4}."""
    return (2 * a / math.pi) ** 0.75


def make_fn(a, center):
    return (a, np.array(center, dtype=float))


def make_eri(w=None):
    """Normalized ERI over basis functions; Coulomb if w is None, else erfc(w)."""

    def eri(f1, f2, f3, f4):
        (a, A), (b, B), (c, C), (d, D) = f1, f2, f3, f4
        raw = (
            eri_coulomb(a, A, b, B, c, C, d, D)
            if w is None
            else eri_erfc(a, A, b, B, c, C, d, D, w)
        )
        return raw * s_norm(a) * s_norm(b) * s_norm(c) * s_norm(d)

    return eri


def build_tables(shells, eri):
    """Psi4's screening tables: squared Schwarz Q^2 and the exchange table X."""
    ns = len(shells)
    q2 = np.zeros((ns, ns))
    for m in range(ns):
        for n in range(ns):
            q2[m, n] = max(abs(eri(u, v, u, v)) for u in shells[m] for v in shells[n])

    norms = {}
    for m in range(ns):
        for i, p in enumerate(shells[m]):
            norms[(m, i)] = math.sqrt(abs(eri(p, p, p, p)))

    # Psi4: X_PQ = max over (p,q) of the per-function RATIO.
    x_psi4 = np.zeros((ns, ns))
    # ferric: X_PQ = max(numerator) / min(denominator)  -- looser, see csam.rs.
    x_ferric = np.zeros((ns, ns))
    for pp in range(ns):
        for qq in range(ns):
            ratios, nums, dens = [], [], []
            for i, p in enumerate(shells[pp]):
                for j, q in enumerate(shells[qq]):
                    num = abs(eri(p, p, q, q))
                    den = norms[(pp, i)] * norms[(qq, j)]
                    ratios.append(num / den)
                    nums.append(num)
                    dens.append(den)
            x_psi4[pp, qq] = max(ratios)
            x_ferric[pp, qq] = max(nums) / min(dens)
    return q2, x_psi4, x_ferric


def audit(shells, label, eri):
    """Worst true/bound for Schwarz (control) and for Psi4's CSAM."""
    ns = len(shells)
    q2, x_psi4, _ = build_tables(shells, eri)

    worst_schwarz = 0.0
    worst_csam = 0.0
    worst_quartet = None
    for m, n, r, s in itertools.product(range(ns), repeat=4):
        true = max(
            abs(eri(u, v, y, z))
            for u in shells[m]
            for v in shells[n]
            for y in shells[r]
            for z in shells[s]
        )
        # Same magnitude filter as `worst_ratio`; see MIN_MAGNITUDE's comment.
        # Applied to the SCHWARZ control too, deliberately: the control's whole
        # job is to run on identical quartets, so filtering only CSAM would
        # break the comparison that makes this harness trustworthy.
        if true < MIN_MAGNITUDE:
            continue
        schwarz = math.sqrt(q2[m, n] * q2[r, s])
        csam_2 = max(x_psi4[m, r] * x_psi4[n, s], x_psi4[m, s] * x_psi4[n, r])
        csam = math.sqrt(q2[m, n] * q2[r, s] * csam_2)

        worst_schwarz = max(worst_schwarz, true / schwarz)
        if csam > 1e-300 and true / csam > worst_csam:
            worst_csam = true / csam
            worst_quartet = (m, n, r, s, true, csam, schwarz)

    ok = "OK (bound holds)" if worst_schwarz <= 1 + 1e-9 else "HARNESS BUG"
    print(f"{label}")
    print(f"   SCHWARZ control worst true/bound = {worst_schwarz:.6f}   {ok}")
    verdict = "VIOLATES" if worst_csam > 1 + 1e-9 else "ok"
    print(f"   Psi4 CSAM       worst true/est   = {worst_csam:.6f}   {verdict}")
    if worst_quartet and worst_csam > 1 + 1e-9:
        m, n, r, s, true, csam, schwarz = worst_quartet
        print(
            f"   worst quartet ({m},{n}|{r},{s}): true={true:.6e} "
            f"csam={csam:.6e} schwarz={schwarz:.6e}"
        )
    print()
    return worst_schwarz, worst_csam


def compare_x_tables(shells, eri, label):
    """Direction check: ferric's X must never be TIGHTER than Psi4's."""
    ns = len(shells)
    _, x_psi4, x_ferric = build_tables(shells, eri)
    tighter = 0
    print(f"X-table comparison ({label}):")
    for p in range(ns):
        for q in range(ns):
            xp, xf = x_psi4[p, q], x_ferric[p, q]
            if xf < xp - 1e-15:
                flag = "ferric TIGHTER <-- would make ferric the cause"
                tighter += 1
            elif xf > xp + 1e-15:
                flag = "ferric looser"
            else:
                flag = "equal"
            print(f"  X[{p},{q}]  psi4={xp:.6f}  ferric={xf:.6f}   {flag}")
    print(f"  entries where ferric is tighter than Psi4: {tighter} (must be 0)\n")
    return tighter


def main():
    origin = [0.0, 0.0, 0.0]
    off = [0.0, 0.0, 1.8]

    # Concentric: mimics the (1,0|0,0) all-oxygen quartet that first failed in
    # ferric -- zero bra-ket separation, where a distance-decay refinement has
    # no justification.
    concentric = [
        [make_fn(130.7, origin)],
        [make_fn(23.8, origin)],
        [make_fn(6.44, origin)],
        [make_fn(0.8, origin)],
        [make_fn(0.25, origin)],
    ]
    two_centers = [
        [make_fn(130.7, origin)],
        [make_fn(5.03, origin)],
        [make_fn(1.0, off)],
        [make_fn(0.3, off)],
    ]
    multi_fn = [
        [make_fn(130.7, origin), make_fn(23.8, origin)],
        [make_fn(6.44, origin), make_fn(0.8, origin)],
        [make_fn(1.0, off), make_fn(0.3, off)],
    ]

    coulomb = make_eri()
    print("=== Coulomb 1/r12 ===\n")
    audit(concentric, "concentric s-shells (O-like exponents)", coulomb)
    audit(two_centers, "two centers", coulomb)
    audit(multi_fn, "multi-function shells", coulomb)

    print("=== erfc(w r12)/r12 (paper claims RIGOR for exponential decay) ===\n")
    for w in (0.11, 1.0):
        audit(concentric, f"concentric, erfc w={w}", make_eri(w))
        audit(two_centers, f"two centers, erfc w={w}", make_eri(w))

    print("=== ferric-vs-Psi4 X formula direction check ===\n")
    compare_x_tables(multi_fn, coulomb, "multi-function shells, Coulomb")

    print("=== WHY erfc IS REFUSED: growth with molecular extent ===\n")
    scaling_study()


def chain(n_centers, spacing=2.64):
    """Linear chain of two-shell centers (tight core + diffuse valence)."""
    shells = []
    for k in range(n_centers):
        c = [0.0, 0.0, spacing * k]
        shells.append([make_fn(3047.5, c), make_fn(457.4, c)])
        shells.append([make_fn(0.35, c), make_fn(0.12, c)])
    return shells


def worst_ratio(shells, w):
    """Worst |true|/estimate under Psi4's CSAM formula."""
    ns = len(shells)
    eri = make_eri(w)
    q2, x_psi4, _ = build_tables(shells, eri)
    worst = 0.0
    for m, n, r, s in itertools.product(range(ns), repeat=4):
        true = max(
            abs(eri(u, v, y, z))
            for u in shells[m]
            for v in shells[n]
            for y in shells[r]
            for z in shells[s]
        )
        # MAGNITUDE FILTER (added 2026-09-14). Without it a quartet whose true
        # value is ~1e-30 can dominate `worst` with a ratio of 1e8 while being
        # irrelevant to any energy — and, for erfc formed by float64
        # subtraction, such a `true` is mostly roundoff anyway. The source
        # paper's Table I/II captions impose exactly this filter at 1e-12.
        if true < MIN_MAGNITUDE:
            continue
        csam_2 = max(x_psi4[m, r] * x_psi4[n, s], x_psi4[m, s] * x_psi4[n, r])
        est = math.sqrt(q2[m, n] * q2[r, s] * csam_2)
        if est > 1e-300:
            worst = max(worst, true / est)
    return worst


def scaling_study():
    """RETRACTED AS EVIDENCE for refusing attenuated operators — read on.

    This study once carried the "1e8x growth" claim. Its erfc column is not
    trustworthy past ~8 bohr of extent: `eri_erfc` forms the kernel as
    `Coulomb - erf` in float64, and at the 6-center geometry (13.2 bohr) the
    resulting `true` value is wrong by 447% against 200-bit mpmath (5-center:
    3.2e-6). The ratios there track the cancellation error, not CSAM.

    It is also unfiltered by default, so a ratio can be set by a quartet too
    small to matter. Run with `--min-magnitude 1e-12` (the paper's own cut) to
    see the difference; the Coulomb column, which involves no cancellation, is
    the part that survives either way.

    Kept rather than deleted because the Coulomb column IS a real measurement
    and because the retraction is more useful recorded than erased. `erfc` is
    now routed to the rigorous CSB bound on the paper's recommendation for
    short-range operators — not on the strength of this table.
    """
    print(" centers  extent(bohr)      Coulomb      erfc(omega=1.0)")
    for n in (2, 3, 4, 5, 6):
        sh = chain(n)
        print(
            f"   {n:2d}       {2.64 * (n - 1):6.2f}   {worst_ratio(sh, None):12.3f}"
            f"   {worst_ratio(sh, 1.0):18.3f}"
        )
    print()
    print("Monotone in attenuation strength (4 centers, fixed geometry):")
    sh = chain(4)
    print(f"   Coulomb   {worst_ratio(sh, None):12.3f}")
    for w in (0.11, 0.3, 0.5, 1.0, 2.0):
        print(f"   w={w:<5.2f}  {worst_ratio(sh, w):12.3f}")
    print()
    print(
        "CAUTION: the erfc column above is float64 Coulomb-minus-erf. Beyond ~8 bohr\n"
        "of extent it is dominated by cancellation error (447% wrong at 13.2 bohr vs\n"
        "200-bit mpmath), so its large entries are NOT bound violations. The Coulomb\n"
        "column involves no cancellation and is the trustworthy one. For the energy\n"
        "consequence of CSAM's non-rigor — which this script does not measure — see\n"
        "Thompson & Ochsenfeld Tables III (uH at 1e-10) and V (nH at 1e-12)."
    )


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ap.add_argument(
        "--min-magnitude",
        type=float,
        default=0.0,
        metavar="X",
        help="exclude quartets whose TRUE value is below X from the worst-ratio "
        "statistics (the source paper uses 1e-12). Default 0.0 = no filter, "
        "which is what produced the retracted 1e8 erfc figures.",
    )
    _args = _ap.parse_args()
    MIN_MAGNITUDE = _args.min_magnitude
    if MIN_MAGNITUDE <= 0.0:
        print(
            "WARNING: running with NO magnitude filter. Ratios reported below can be\n"
            "         dominated by numerically-irrelevant quartets (true value ~1e-30),\n"
            "         and for erfc — which this harness forms as Coulomb - erf in\n"
            "         float64 — by outright cancellation error. See the module\n"
            "         docstring. Pass --min-magnitude 1e-12 to match the paper.\n"
        )
    main()
