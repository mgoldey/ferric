#!/usr/bin/env python3
"""Design prototypes for the two lmp2_direct follow-ups, in pure set
arithmetic on GEOMETRIC map models (no integrals — the questions are about
counts and reuse factors, which are ratios and robust to modest model error).

Q1 (recompute amortization): how many (P|μν) shell-triple evaluations do
    per-atom batches cost vs merged k-atom super-batches vs the global
    dedupe lower bound, and what does the batch scratch slab cost?
Q2 (V_DD reuse): how many DISTINCT pair aux-domains are there vs surviving
    pairs — is a memoized Cholesky cache worth building?

Model (alkanes, CnH2n+2, 6-31G / cc-pvdz-ri):
- active occupieds = bond orbitals (n-1 C-C + 2n+2 C-H = 3n+1 = exact
  active count at frozen_core=n); centroids = bond midpoints, spread σ
  calibrated so the R⁻⁶ gate reproduces the MEASURED survivors/occ
  (C20: erfc 10.5, coul 18.8; C48: 10.8, 19.7).
- aux shells: C≈16 sh, H≈6 sh per atom (ratio-level proxy); obs: C 5, H 2.
- domains: aux within r_aux=10 Bohr of centroid; virts atom-hosted within
  r_virt=12 Bohr; occ/virt AO supports = shells of atoms within r_supp of
  the centroid/atom; Schwarz cut modeled as obs-pair atom distance
  ≤ r_schwarz (both calibrated against measured Mtriples/strip rows).

Calibration table prints model-vs-measured; conclusions are the RATIOS.
"""
import math
import sys
from itertools import combinations

BOHR = 1.8897259886

def load_xyz(path):
    lines = open(path).read().split("\n")
    n = int(lines[0])
    atoms = []
    for ln in lines[2 : 2 + n]:
        p = ln.split()
        atoms.append((p[0], float(p[1]) * BOHR, float(p[2]) * BOHR, float(p[3]) * BOHR))
    return atoms

def dist(a, b):
    return math.dist(a[1:], b[1:])

AUX_SH = {"C": 16, "H": 6}
OBS_SH = {"C": 5, "H": 2}

class Model:
    def __init__(self, path, sigma_cc, sigma_ch, r_aux, r_virt, r_supp, r_schwarz):
        self.atoms = load_xyz(path)
        self.r_aux, self.r_virt, self.r_supp, self.r_schwarz = r_aux, r_virt, r_supp, r_schwarz
        # bonds -> occupied centroids
        self.occ = []  # (xyz, sigma)
        for i, j in combinations(range(len(self.atoms)), 2):
            a, b = self.atoms[i], self.atoms[j]
            d = dist(a, b)
            if a[0] == "C" and b[0] == "C" and d < 1.8 * BOHR:
                self.occ.append((mid(a, b), sigma_cc))
            elif {a[0], b[0]} == {"C", "H"} and d < 1.25 * BOHR:
                self.occ.append((mid(a, b), sigma_ch))
        # shells with per-atom ids: (atom_index, kind) counted
        self.aux_shells = [(ai,) * 1 + (k,) for ai, a in enumerate(self.atoms) for k in range(AUX_SH[a[0]])]
        self.obs_shells = [(ai, k) for ai, a in enumerate(self.atoms) for k in range(OBS_SH[a[0]])]
        self.no = len(self.occ)

    def gate_survivors(self, cal, eps):
        theta = 1e-2 * eps
        keep = {}
        for i, (ci, si) in enumerate(self.occ):
            for j, (cj, sj) in enumerate(self.occ):
                if i == j:
                    keep.setdefault(i, set()).add(j)
                    continue
                r2 = sum((x - y) ** 2 for x, y in zip(ci, cj))
                est = cal * (si * sj) ** 3 / max(r2 ** 3, 1e-12)
                if est >= theta:
                    keep.setdefault(i, set()).add(j)
        return keep

    def aux_dom(self, c):
        return {s for s in self.aux_shells if adist(self.atoms[s[0]], c) <= self.r_aux}

    def supp_shells(self, c, r):
        return {s for s in self.obs_shells if adist(self.atoms[s[0]], c) <= r}

def mid(a, b):
    return tuple((x + y) / 2 for x, y in zip(a[1:], b[1:]))

def adist(atom, c):
    return math.dist(atom[1:], c)

def run(path, name, cal, eps, meas):
    # calibrated constants (see __main__ notes)
    m = Model(path, sigma_cc=1.75, sigma_ch=1.55, r_aux=10.0, r_virt=12.0,
              r_supp=7.0, r_schwarz=8.5)
    keep = m.gate_survivors(cal, eps)
    surv_per_occ = sum(len(v) for v in keep.values()) / m.no
    # per-occupied extended sets
    occ_dom = [m.aux_dom(c) for c, _ in m.occ]
    occ_supp = [m.supp_shells(c, m.r_supp) for c, _ in m.occ]
    # virt hosting: virts live on atoms; virt domain of occ i = atoms within r_virt
    virt_atoms = [ {ai for ai, a in enumerate(m.atoms) if adist(a, c) <= m.r_virt} for c, _ in m.occ ]
    aux_ext, virt_ext = [], []
    for i in range(m.no):
        ad, vd = set(), set()
        for j in keep.get(i, {i}):
            ad |= occ_dom[j]
            vd |= virt_atoms[j]
        aux_ext.append(ad)
        virt_ext.append(vd)
    # batches: occupieds grouped by nearest atom
    batches = {}
    for i, (c, _) in enumerate(m.occ):
        amin = min(range(len(m.atoms)), key=lambda ai: adist(m.atoms[ai], c))
        batches.setdefault(amin, []).append(i)

    def batch_triples(groups):
        """count evaluated triples for a grouping = list of lists of occ ids"""
        total = 0
        distinct = set()
        peak_slab = 0
        for g in groups:
            P = set().union(*(aux_ext[i] for i in g))
            S = set().union(*(occ_supp[i] for i in g))
            # virt supports: shells of atoms within r_supp of any hosted virt atom
            vat = set().union(*(virt_ext[i] for i in g))
            N = {s for s in m.obs_shells
                 if any(math.dist(m.atoms[s[0]][1:], m.atoms[va][1:]) <= m.r_supp for va in vat)}
            U = sorted(S | N)
            for x, ua in enumerate(U):
                for ub in U[: x + 1]:
                    need = (ua in S and ub in N) or (ub in S and ua in N)
                    if not need:
                        continue
                    # Schwarz model: obs-pair atom distance cut
                    if math.dist(m.atoms[ua[0]][1:], m.atoms[ub[0]][1:]) > m.r_schwarz:
                        continue
                    total += len(P)
                    for sp in P:
                        distinct.add((sp, ua, ub))
            peak_slab = max(peak_slab, len(S) * 3 * len(N) * 3 * 8 * min(len(P) * 3, 10**9))
        return total, len(distinct), peak_slab

    cur, distinct, _ = batch_triples(list(batches.values()))
    rows = [f"{name} eps={eps:g} cal={cal:g}: survivors/occ model {surv_per_occ:.1f} vs measured {meas['ppo']:.1f}"]
    rows.append(f"  triples: per-atom batches {cur/1e6:.1f}M-sh, distinct-needed {distinct/1e6:.1f}M-sh "
                f"(recompute x{cur/max(distinct,1):.2f}) [measured eval {meas['mtrip']:.0f}M fn-triples]")
    # merged super-batches along the chain (atoms sorted by x)
    order = sorted(batches.keys(), key=lambda ai: m.atoms[ai][1])
    for k in (2, 4, 8):
        groups = []
        for s in range(0, len(order), k):
            g = []
            for ai in order[s : s + k]:
                g += batches[ai]
            groups.append(g)
        tot, _, slab = batch_triples(groups)
        rows.append(f"  merge k={k}: {tot/1e6:.1f}M-sh (x{tot/max(distinct,1):.2f} of distinct, "
                    f"{tot/cur:.2f} of per-atom)")
    # Q2: V_DD reuse
    npairs = 0
    doms = set()
    for i in range(m.no):
        for j in keep.get(i, set()):
            if j < i:
                continue
            npairs += 1
            doms.add(frozenset(occ_dom[i] | occ_dom[j]))
    rows.append(f"  V_DD: {npairs} unique pairs, {len(doms)} distinct domains "
                f"(reuse x{npairs/max(len(doms),1):.2f})")
    print("\n".join(rows))

if __name__ == "__main__":
    # measured references from wiki §29 (skip=1e-5 rows)
    cases = [
        ("alkane_20.xyz", "C20 erfc", 0.02, 1e-3, dict(ppo=643/61, mtrip=47.0)),
        ("alkane_20.xyz", "C20 coul", 0.7, 1e-3, dict(ppo=1145/61, mtrip=64.0)),
        ("alkane_48.xyz", "C48 erfc", 0.02, 1e-3, dict(ppo=1567/145, mtrip=137.1)),
        ("alkane_48.xyz", "C48 coul", 0.7, 1e-3, dict(ppo=2853/145, mtrip=195.2)),
    ]
    base = sys.argv[1] if len(sys.argv) > 1 else "testdata/molecules"
    for fn, name, cal, eps, meas in cases:
        run(f"{base}/{fn}", name, cal, eps, meas)
