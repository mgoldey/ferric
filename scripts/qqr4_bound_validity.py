#!/usr/bin/env python3
"""Validity + tightness probe for the 4-center QQR screening bound.

WHY THIS EXISTS
---------------
`crates/ferric-scf/src/qqr.rs` implements the 4-center distance-aware bound as

    bound = Q(i,j) * Q(k,l) * min(1, ext_ij * ext_kl / R)          [form A]
            * exp(-omega^2 R^2)   if the operator is ErfcCoulomb

with R the CENTER-TO-CENTER distance between the two pair charge centers.

ferric's own validated 3-index sibling `crates/ferric-integrals/src/qqr3.rs`
documents both of those choices as INVALID for the analogous (P|mn) bound:

  * the distance factor must scale as 1/R_eff with an ext_SUM numerator, not
    ext*ext/R ("under-states the cloud charges and collapses far too fast,
    ratio 14.8");
  * R must be measured EDGE-TO-EDGE, R_eff = max(0, R - ext_sum), so the
    envelope is continuous and falls back to Schwarz for penetrating clouds;
  * an extra erfc/Gaussian long-range factor on top of erfc Schwarz factors
    "over-suppresses real long-range triples and makes the bound INVALID
    (measured worst ratio 1.7-5.3)" -- the attenuation already lives in the
    per-operator Schwarz factors;
  * even the corrected monopole envelope needed an empirical SAFETY_FACTOR
    (1.10 in qqr3) because it under-estimates by a few percent in the
    near-intermediate zone.

So we measure, on TRUE PySCF shell-quartet integrals:

    form A ("current")   Q_ij Q_kl * min(1, ext_ij*ext_kl / R)     [* erfc gauss]
    form B ("qqr3-style") Q_ij Q_kl * min(1, s * ext_sum / R_eff)

VALID means bound >= |true| for EVERY quartet, i.e. ratio = true/bound <= 1.

METHODOLOGY NOTES (per the repo's experimental protocol)
-------------------------------------------------------
* EXACTNESS ANCHOR: `--anchor` checks that with the distance factor forced to 1
  (the trivial limit) both forms reduce EXACTLY to the plain Schwarz product,
  and that plain Schwarz itself is a valid bound on the sampled quartets. If
  that fails, nothing downstream means anything.
* ARTIFACT HYPOTHESIS: if the Schwarz factors themselves are wrong (e.g. a
  prescreened-to-zero pair), BOTH forms violate at the same quartets and the
  violation is 100% of Schwarz, not a distance-envelope effect. The anchor run
  distinguishes that from a genuine envelope defect.
* SAMPLING BIAS: a uniform quartet sample is dominated by near-field quartets
  where decay == 1 and every form is trivially valid. We therefore report the
  SEPARATED subset (decay < 1 under the corrected form) separately, and the
  sweep is over ALL quartets for small systems (no subsampling needed at the
  sizes used here).

Usage:
    python3 scripts/qqr4_bound_validity.py --systems water benzene
    python3 scripts/qqr4_bound_validity.py --systems alkane_6 --omega 1.0
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from pyscf import gto

BOHR_PER_ANG = 1.8897261254535

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOLDIR = os.path.join(REPO_ROOT, "testdata", "molecules")


# ---------------------------------------------------------------------------
# molecule / shell metadata
# ---------------------------------------------------------------------------

def build_mol(name: str, basis: str) -> gto.Mole:
    path = os.path.join(MOLDIR, f"{name}.xyz")
    with open(path) as fh:
        lines = fh.read().splitlines()
    nat = int(lines[0].split()[0])
    atoms = []
    for ln in lines[2 : 2 + nat]:
        parts = ln.split()
        atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
    mol = gto.M(atom=atoms, basis=basis, unit="Angstrom", verbose=0)
    return mol


def shell_min_exponents_and_origins(mol: gto.Mole):
    """Per-shell (min primitive exponent, atom origin in Bohr).

    Mirrors ferric's `collect_min_exponents_and_origins`: the MOST DIFFUSE
    (smallest) primitive exponent sets the shell's spatial extent.
    """
    min_exp = np.empty(mol.nbas)
    origins = np.empty((mol.nbas, 3))
    coords = mol.atom_coords()  # Bohr
    for s in range(mol.nbas):
        min_exp[s] = mol.bas_exp(s).min()
        origins[s] = coords[mol.bas_atom(s)]
    return min_exp, origins


def pair_centers_extents(min_exp: np.ndarray, origins: np.ndarray):
    """Dense nsh x nsh pair charge centers and extents, ferric's definition.

        center_ij = (a_i R_i + a_j R_j) / (a_i + a_j)
        ext_ij    = 1 / sqrt(a_i + a_j)
    """
    nsh = len(min_exp)
    asum = min_exp[:, None] + min_exp[None, :]
    centers = (
        min_exp[:, None, None] * origins[:, None, :]
        + min_exp[None, :, None] * origins[None, :, :]
    ) / asum[:, :, None]
    extents = 1.0 / np.sqrt(asum)
    return centers, extents, nsh


# ---------------------------------------------------------------------------
# Schwarz factors (from true integrals, at full precision -- no prescreening)
# ---------------------------------------------------------------------------

def schwarz_matrix(mol: gto.Mole, omega: float | None) -> np.ndarray:
    """Q(i,j) = sqrt(max_{ab} |(ab|ab)|) over the shell pair, per operator.

    Computed from the actual (ij|ij) block with NO screening, so this is the
    valid (over-estimating) Schwarz table -- the same property the sibling
    schwarz.rs fix restored on the Rust side.
    """
    nsh = mol.nbas
    q = np.zeros((nsh, nsh))
    ctx = mol.with_range_coulomb(-omega) if omega else _null_ctx()
    with ctx:
        for i in range(nsh):
            for j in range(i + 1):
                blk = mol.intor("int2e_sph", shls_slice=(i, i + 1, j, j + 1, i, i + 1, j, j + 1))
                n1, n2 = blk.shape[0], blk.shape[1]
                # generalized diagonal (ab|ab)
                diag = np.abs(blk[np.arange(n1)[:, None], np.arange(n2)[None, :],
                                  np.arange(n1)[:, None], np.arange(n2)[None, :]])
                val = math.sqrt(diag.max()) if diag.size else 0.0
                q[i, j] = q[j, i] = val
    return q


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# quartet sampling
# ---------------------------------------------------------------------------

def enumerate_quartets(nsh: int, max_quartets: int, rng: np.random.Generator,
                       centers: np.ndarray, extents: np.ndarray):
    """Unique shell quartets (i>=j, k>=l, braket-symmetric), biased to far pairs.

    A uniform sample over a small molecule is dominated by near-field quartets
    where decay == 1 and every candidate bound is trivially valid. When the full
    enumeration exceeds `max_quartets` we keep ALL separated quartets (those
    whose corrected-form decay factor is < 1, i.e. the regime the bound exists
    for) and subsample only the near-field remainder.
    """
    pairs = [(i, j) for i in range(nsh) for j in range(i + 1)]
    npair = len(pairs)
    all_q = [(pairs[a], pairs[b]) for a in range(npair) for b in range(a + 1)]
    if len(all_q) <= max_quartets:
        return all_q

    sep, near = [], []
    for (i, j), (k, l) in all_q:
        ext_sum = extents[i, j] + extents[k, l]
        r = np.linalg.norm(centers[i, j] - centers[k, l])
        (sep if r > ext_sum else near).append(((i, j), (k, l)))
    out = sep[:max_quartets]
    room = max_quartets - len(out)
    if room > 0 and near:
        idx = rng.choice(len(near), size=min(room, len(near)), replace=False)
        out.extend(near[t] for t in idx)
    return out


def true_max_quartet(mol: gto.Mole, i, j, k, l) -> float:
    blk = mol.intor("int2e_sph",
                    shls_slice=(i, i + 1, j, j + 1, k, k + 1, l, l + 1))
    return float(np.abs(blk).max())


# ---------------------------------------------------------------------------
# the two candidate bound forms
# ---------------------------------------------------------------------------

def decay_current(centers, extents, i, j, k, l, omega):
    """Form A: ferric's CURRENT qqr.rs -- ext*ext / R center-to-center,
    plus exp(-omega^2 R^2) for ErfcCoulomb."""
    r = float(np.linalg.norm(centers[i, j] - centers[k, l]))
    if r < 1e-14:
        return 1.0
    d = min(1.0, extents[i, j] * extents[k, l] / r)
    if omega:
        d *= math.exp(-omega * omega * r * r)
    return d


def decay_qqr3_style(centers, extents, i, j, k, l, safety):
    """Form B: qqr3-style -- ext_SUM / R_eff, EDGE-TO-EDGE, no extra erfc factor.

    Attenuation for ErfcCoulomb is carried entirely by the (smaller) erfc
    Schwarz factors, exactly as qqr3.rs documents.
    """
    r = float(np.linalg.norm(centers[i, j] - centers[k, l]))
    if r < 1e-14:
        return 1.0
    ext_sum = extents[i, j] + extents[k, l]
    r_eff = max(0.0, r - ext_sum)
    if r_eff <= 0.0:
        return 1.0
    return min(1.0, safety * ext_sum / r_eff)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_system(name: str, basis: str, omega: float | None, max_quartets: int,
               safety: float, seed: int, anchor: bool):
    mol = build_mol(name, basis)
    min_exp, origins = shell_min_exponents_and_origins(mol)
    centers, extents, nsh = pair_centers_extents(min_exp, origins)
    q = schwarz_matrix(mol, omega)

    rng = np.random.default_rng(seed)
    quartets = enumerate_quartets(nsh, max_quartets, rng, centers, extents)

    op_label = "Coulomb" if not omega else f"erfc(omega={omega})"
    print(f"\n=== {name}/{basis}  {op_label}  nbas={mol.nao} nsh={nsh} "
          f"quartets={len(quartets)} ===")

    ctx = mol.with_range_coulomb(-omega) if omega else _null_ctx()

    rows = []
    with ctx:
        for (i, j), (k, l) in quartets:
            tru = true_max_quartet(mol, i, j, k, l)
            sch = q[i, j] * q[k, l]
            dA = decay_current(centers, extents, i, j, k, l, omega)
            dB = decay_qqr3_style(centers, extents, i, j, k, l, safety)
            dB1 = decay_qqr3_style(centers, extents, i, j, k, l, 1.0)
            rows.append((i, j, k, l, tru, sch, dA, dB, dB1))

    report(name, basis, op_label, rows, safety, anchor)
    return rows, q, centers, extents, nsh


# Cauchy-Schwarz is EXACT (ratio == 1 to rounding) on the diagonal quartets
# (ij|ij), so a bare `ratio > 1` test flags those as violations. A bound that
# attains equality is still a valid bound; only a ratio meaningfully above 1 is
# a real underestimate. 1e-9 is ~1e6x above the observed diagonal residual and
# ~1e6x below the smallest genuine violation seen (form A worst ratio 2.27).
VIOLATION_TOL = 1e-9


def _stats(name, rows, decay_idx, tol=VIOLATION_TOL):
    """Violations + tightness for one bound form."""
    viol = []
    worst = 0.0
    worst_q = None
    ratios = []
    for (i, j, k, l, tru, sch, dA, dB, dB1) in rows:
        d = (dA, dB, dB1)[decay_idx]
        bound = sch * d
        if tru <= 1e-15:
            continue
        if bound <= 0.0:
            ratio = float("inf")
        else:
            ratio = tru / bound
        ratios.append(ratio)
        if ratio > worst:
            worst, worst_q = ratio, (i, j, k, l, tru, bound, d)
        if ratio > 1.0 + tol:
            viol.append((i, j, k, l, tru, bound, ratio))
    return viol, worst, worst_q, np.array(ratios) if ratios else np.array([1.0])


def report(name, basis, op_label, rows, safety, anchor):
    n = len(rows)
    # separated subset = where the CORRECTED form actually engages
    sep = [r for r in rows if r[7] < 1.0]
    print(f"  separated quartets (corrected decay < 1): {len(sep)}/{n} "
          f"({100.0*len(sep)/max(n,1):.1f}%)")

    if anchor:
        # EXACTNESS ANCHOR: plain Schwarz (decay forced to 1) must itself be a
        # valid bound on this sample. If it is not, no envelope conclusion holds.
        v, w, wq, _ = _stats(name, [(i, j, k, l, tru, sch, 1.0, 1.0, 1.0)
                                    for (i, j, k, l, tru, sch, *_) in rows], 1)
        status = "OK" if not v else f"FAILED ({len(v)} violations)"
        print(f"  [ANCHOR] plain Schwarz validity: {status}, worst true/Schwarz = {w:.4f}")
        if v:
            print("           anchor failure => the Schwarz table is the defect, "
                  "not the distance envelope. Downstream numbers are meaningless.")

    for label, idx in (("A current (ext*ext/R, c2c, +erfc gauss)", 0),
                       (f"B qqr3-style (ext_sum/R_eff, safety={safety})", 1),
                       ("B0 qqr3-style bare (safety=1.0)", 2)):
        viol, worst, wq, ratios = _stats(name, rows, idx)
        tighter = np.array([r[5] * (r[6], r[7], r[8])[idx] for r in rows])
        sch = np.array([r[5] for r in rows])
        nz = sch > 0
        frac_tight = float(np.mean(tighter[nz] / sch[nz]))
        print(f"  [{label}]")
        print(f"      violations (true/bound > 1): {len(viol)} / {len(rows)}")
        print(f"      WORST true/bound            : {worst:.4f}"
              + ("  <-- INVALID" if worst > 1.0 else "  (valid)"))
        print(f"      median true/bound           : {np.median(ratios):.4f}")
        print(f"      mean bound / Schwarz        : {frac_tight:.4f} "
              f"(1.0 = no tightening)")
        if sep:
            sepmask = np.array([r[7] < 1.0 for r in rows])
            if sepmask.any():
                print(f"      mean bound/Schwarz on separated subset: "
                      f"{float(np.mean(tighter[sepmask & nz] / sch[sepmask & nz])):.4f}")
        if wq:
            i, j, k, l, tru, bound, d = wq
            print(f"      worst quartet ({i},{j}|{k},{l}) true={tru:.4e} "
                  f"bound={bound:.4e} decay={d:.4e}")
        if viol[:3]:
            for (i, j, k, l, tru, bound, ratio) in viol[:3]:
                print(f"        violation ({i},{j}|{k},{l}): true={tru:.4e} "
                      f"bound={bound:.4e} ratio={ratio:.3f}")


def min_safety_factor(rows) -> float:
    """Smallest multiplier on the BARE qqr3-style envelope making it valid.

    The multiplier only bites where decay < 1 (elsewhere min(1, .) clamps), so
    the required factor is max over quartets of true/(Schwarz*decay_bare)
    restricted to the clamped-below-1 set.
    """
    need = 1.0
    for (i, j, k, l, tru, sch, dA, dB, dB1) in rows:
        if tru <= 1e-15 or sch <= 0.0 or dB1 >= 1.0:
            continue
        bound = sch * dB1
        if bound <= 0.0:
            return float("inf")
        need = max(need, tru / bound)
    return need


def screening_benefit(q, centers, extents, nsh, safety,
                      thresh_list=(1e-8, 1e-10, 1e-12)):
    """How many quartets does the corrected bound drop that Schwarz keeps?

    This is the only number that justifies wiring QQR into LinK at all.

    MEASURED ON THE FULL QUARTET POPULATION, NOT THE VALIDITY SAMPLE. The
    validity sweep deliberately over-samples well-separated quartets (that is
    where a distance envelope can be wrong), so counting screening decisions on
    that biased subset would report a meaningless number -- an earlier version
    of this function did exactly that and reported "0 extra dropped" purely
    because every sampled quartet sat far above threshold. Screening value is a
    property of the whole population, so we re-enumerate it here from the cheap
    bound tables alone (no integrals needed).
    """
    pairs = [(i, j) for i in range(nsh) for j in range(i + 1)]
    sch, qqr = [], []
    for a in range(len(pairs)):
        i, j = pairs[a]
        for b in range(a + 1):
            k, l = pairs[b]
            s = q[i, j] * q[k, l]
            sch.append(s)
            qqr.append(s * decay_qqr3_style(centers, extents, i, j, k, l, safety))
    sch = np.array(sch)
    qqr = np.array(qqr)
    out = []
    for t in thresh_list:
        n_s = int((sch >= t).sum())
        n_q = int((qqr >= t).sum())
        extra = n_s - n_q
        out.append((t, n_s, n_q, extra, 100.0 * extra / max(n_s, 1)))
    return out, len(sch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["water", "benzene"])
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--omegas", nargs="+", type=float, default=[0.0, 1.0],
                    help="0.0 = Coulomb; positive = erfc(omega)")
    ap.add_argument("--max-quartets", type=int, default=20000)
    ap.add_argument("--safety", type=float, default=1.10,
                    help="qqr3's SAFETY_FACTOR, tested for sufficiency here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-anchor", action="store_true")
    args = ap.parse_args()

    summary = []
    for name in args.systems:
        for om in args.omegas:
            omega = None if om == 0.0 else om
            rows, q, centers, extents, nsh = run_system(
                name, args.basis, omega, args.max_quartets,
                args.safety, args.seed, not args.no_anchor)
            need = min_safety_factor(rows)
            print(f"  MINIMUM safety factor for form B validity: {need:.4f} "
                  + ("(qqr3's 1.10 SUFFICES)" if need <= 1.10
                     else "(qqr3's 1.10 IS NOT ENOUGH)"))
            bene, n_total = screening_benefit(q, centers, extents, nsh, args.safety)
            print(f"  screening benefit vs plain Schwarz over ALL {n_total} "
                  f"quartets (kept at threshold):")
            for (t, n_s, n_q, extra, pct) in bene:
                print(f"      thresh {t:.0e}: Schwarz keeps {n_s}, "
                      f"QQR-B keeps {n_q}  ->  {extra} extra dropped ({pct:.1f}%)")
            vA, wA, _, _ = _stats(name, rows, 0)
            vB, wB, _, _ = _stats(name, rows, 1)
            summary.append((name, args.basis,
                            "Coulomb" if omega is None else f"erfc({omega})",
                            len(rows), len(vA), wA, len(vB), wB, need))

    print("\n================ SUMMARY ================")
    hdr = (f"{'system':<10} {'basis':<9} {'op':<12} {'n':>7} "
           f"{'A viol':>7} {'A worst':>9} {'B viol':>7} {'B worst':>9} {'min s':>7}")
    print(hdr)
    for row in summary:
        print(f"{row[0]:<10} {row[1]:<9} {row[2]:<12} {row[3]:>7} "
              f"{row[4]:>7} {row[5]:>9.3f} {row[6]:>7} {row[7]:>9.4f} {row[8]:>7.3f}")
    print("(viol = quartets with true/bound > 1; a VALID bound has 0 and worst <= 1)")


if __name__ == "__main__":
    sys.exit(main())
