#!/usr/bin/env python3
"""Atom-granular follow-up for the eps-linked aux-domain prototype (C1).

Why: ferric's lmp2_direct distance rule decides aux-shell membership by the
shell's ATOM distance, so domains are atom-index sets — that quantization is
what makes 6-9x of surviving pairs share a pair domain (wiki Sec-31), and
the grouped stage-5 hoists one d^3 Cholesky per DISTINCT domain. A
per-function magnitude rule destroys that sharing, so its d must shrink by
more than sharing^(1/3) to win the Cholesky part. This pass therefore
measures BOTH rules at atom granularity with the same energy metric as the
main prototype, plus the stage-5 cost model:
    cost = sum_distinct d^3/3  (Cholesky, grouped)
         + sum_pairs 2*d^2*nv + sum_pairs 2*d*nv^2   (fit apply + block GEMM)

Anchors inherited from proto_eps_linked_maps (module import); the trivial
limit (huge r / tau=0) is re-checked here at atom granularity.
Artifact hypotheses: same as the main script's C1. The additional
measurable here: n_distinct pair domains per rule — if magnitude selection
at atom granularity still shares domains comparably to distance, the
grouping objection dissolves; if not, the cost model decides.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto_eps_linked_maps as P  # noqa: E402

log = P.log


def aux_atoms(ctx):
    """Map aux function -> atom index via coordinates (aux functions sit on
    atom centers)."""
    axyz = ctx["mol"].atom_coords()
    d = np.linalg.norm(ctx["aux_xyz"][:, None, :] - axyz[None, :, :], axis=2)
    a = d.argmin(axis=1)
    assert d[np.arange(len(a)), a].max() < 1e-8, "aux function off-center?"
    return a


def doms_from_atomsets(atomsets, f2a):
    naux = len(f2a)
    doms = []
    for ats in atomsets:
        m = np.isin(f2a, list(ats))
        doms.append(np.nonzero(m)[0])
    return doms


def atomsets_dist(ctx, r):
    axyz = ctx["mol"].atom_coords()
    d = np.linalg.norm(axyz[None, :, :] - ctx["occ_cen"][:, None, :], axis=2)
    return [frozenset(np.nonzero(d[i] <= r)[0].tolist())
            for i in range(ctx["no"])]


def atomsets_mag(ctx, tau, f2a, mode="thresh"):
    """Atom score s_i(A) = max_{P in A} max_a |(P|ia)|; thresh keeps
    s_i(A) >= tau, tail drops smallest-sum atoms while sum <= tau."""
    s = P.aux_score(ctx, metric_norm=False)  # (no, naux)
    natom = ctx["mol"].natm
    sa = np.zeros((ctx["no"], natom))
    for A in range(natom):
        cols = np.nonzero(f2a == A)[0]
        sa[:, A] = s[:, cols].max(axis=1)
    out = []
    for i in range(ctx["no"]):
        if mode == "thresh":
            keep = np.nonzero(sa[i] >= tau)[0]
        else:  # tail
            order = np.argsort(sa[i])
            cum = np.cumsum(sa[i][order])
            ndrop = int(np.searchsorted(cum, tau, side="right"))
            keep = order[ndrop:]
        out.append(frozenset(keep.tolist()))
    return out


def stage5_cost(atomsets, f2a, no, nv):
    """Grouped stage-5 flop model over unique pairs i<=j."""
    natom_funcs = np.bincount(f2a)
    union_sizes = {}
    pair_costs = 0.0
    npairs = 0
    for i in range(no):
        for j in range(i, no):
            u = atomsets[i] | atomsets[j]
            d = int(sum(natom_funcs[a] for a in u))
            union_sizes.setdefault(u, d)
            pair_costs += 2.0 * d * d * nv + 2.0 * d * nv * nv
            npairs += 1
    chol = sum(d ** 3 / 3.0 for d in union_sizes.values())
    sizes = np.array(sorted(union_sizes.values()))
    return dict(n_pairs=npairs, n_distinct=len(union_sizes),
                chol_gf=chol / 1e9, gemm_gf=pair_costs / 1e9,
                total_gf=(chol + pair_costs) / 1e9,
                dmean_distinct=float(sizes.mean()))


def c2_domain_conservativeness(ctx, tag, outfh, r_aux=10.0, qloc_radius=10.0):
    """Does the kappa=1 Schwarz screen stay conservative against the
    DOMAIN-fitted J (per-pair D_ij, the actual direct-path object)? The
    global-Gram Cauchy-Schwarz theorem does NOT transfer (wiki Sec-23), so
    this is an empirical bound-slack measurement: count Eq-8-kept elements
    of J_dom escaping the candidate set, and the worst escape/eps ratio."""
    doms = P.aux_doms_dist(ctx, r_aux)
    Jd, _, _, nf = P.fit_domains(ctx["A"], ctx["V"], doms)
    assert nf == 0
    ql = P.q_fit_local(ctx, qloc_radius)
    for eps in (1e-3, 1e-4):
        mask8 = (np.abs(Jd) > eps) | (np.abs(Jd.transpose(0, 3, 2, 1)) > eps)
        for kappa in (0.5, 1.0):
            allow = P.allow_from_cand(P.cand_schwarz(ql, kappa * eps))
            esc = mask8 & ~allow
            n_esc = int(esc.sum())
            worst = float(np.abs(Jd[esc]).max()) if n_esc else 0.0
            row = (f"{tag} c2dom eps={eps:g} kappa={kappa:g} r_aux={r_aux:g} "
                   f"escapes={n_esc} worst|Jdom|esc={worst:.2e} "
                   f"worst/eps={worst / eps:.2f}")
            log("    " + row)
            outfh.write(row + "\n")
    outfh.flush()


def run(ctx, tag, outfh):
    f2a = aux_atoms(ctx)
    e_ref = P.solve_full_mp2_closed(ctx["Jg"], ctx["Foo"], ctx["Fvv"])
    # trivial-limit re-check at atom granularity
    doms = doms_from_atomsets(atomsets_dist(ctx, 1e9), f2a)
    Jd, _, _, nf = P.fit_domains(ctx["A"], ctx["V"], doms)
    d0 = np.abs(Jd - ctx["Jg"]).max()
    log(f"    atom-granular trivial limit: max|dJ|={d0:.3e} solvefail={nf}")
    if d0 > 1e-10 or nf:
        raise SystemExit("atom-granular trivial-limit anchor FAILED")
    sweeps = ([("dist", r, lambda r=r: atomsets_dist(ctx, r))
               for r in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0)]
              + [("magt", t, lambda t=t: atomsets_mag(ctx, t, f2a, "thresh"))
                 for t in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)]
              + [("tail", b, lambda b=b: atomsets_mag(ctx, b, f2a, "tail"))
                 for b in (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3)])
    for rule, knob, mk in sweeps:
        t0 = time.time()
        atomsets = mk()
        if any(len(s) == 0 for s in atomsets):
            log(f"    {rule} {knob:g} EMPTY — skipped")
            continue
        doms = doms_from_atomsets(atomsets, f2a)
        Jd, dmean, dmax, nfail = P.fit_domains(ctx["A"], ctx["V"], doms)
        e = P.solve_full_mp2_closed(Jd, ctx["Foo"], ctx["Fvv"])
        cost = stage5_cost(atomsets, f2a, ctx["no"], ctx["nv"])
        row = (f"{tag} auxatom {rule:>4s} knob={knob:<7g} "
               f"pairdom={dmean:7.1f}/{dmax:4d} of {ctx['naux']} "
               f"dE={e - e_ref:+.3e} distinct={cost['n_distinct']}/"
               f"{cost['n_pairs']} chol={cost['chol_gf']:.2f}GF "
               f"gemm={cost['gemm_gf']:.2f}GF tot={cost['total_gf']:.2f}GF "
               f"solvefail={nfail} ({time.time()-t0:.1f}s)")
        log("    " + row)
        outfh.write(row + "\n")
        outfh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = (f"{os.path.basename(a.xyz).replace('.xyz','')} {a.basis} "
           f"{'coul' if a.omega is None else f'erfc{a.omega:g}'}")
    outpath = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "out", "eps_linked_maps_atom.txt")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ctx = P.setup(a.xyz, a.basis, a.omega)
    with open(outpath, "a") as outfh:
        run(ctx, tag, outfh)
    log("  done.")


if __name__ == "__main__":
    main()
