#!/usr/bin/env python3
"""QM/MM boundary: link atoms, Z1/RC/RCD charges and the full-structure gradient.

Derivation: `book/src/23-embedding/02-qmmm-boundaries.md#qmmm-boundaries-cutting-covalent-bond`.
Code: crates/ferric-scf/src/qmmm.rs (QmmmSystem::with_link_atoms,
QmmmSystem::with_boundary_charges, active_charges, mm_forces, full_gradient).

The partition below is a line-for-line transcription of the code; it works on
plain Python numbers or sympy symbols. The energy model standing in for the
embedded SCF is a smooth, translation-invariant function of the QM nuclei
(link hydrogens included) and the active embedding charges.

Checks (each label carries "(claim X)"):
  Q1  Link-atom chain rule: R_L = (1 - g) R_Q + g R_M gives dE/dR_Q += (1 - g)
      dE/dR_L and dE/dR_M += g dE/dR_L, and the two rows sum to dE/dR_L
      (symbols, a general function of R_L).
  Q2  For n = 1..5 M2 neighbours with symbolic positions: RCD conserves the
      total charge and the dipole of the {M1, M2} group about any origin; RC
      conserves the charge and moves the dipole by (q/2)(mean(R_M2) - R_M1),
      which is nonzero in general; Z1 changes the charge by -q.
  Q3  full_gradient, as transcribed, equals a central difference of the
      model energy over every coordinate of the full structure, with the
      partition (link atoms, RC/RCD midpoints) rebuilt at each displacement,
      for Keep, Z1, RC and RCD. Negative controls: the link row split as
      (g, 1 - g) instead of (1 - g, g), and a midpoint force put wholly on
      M1, both miss. Tolerance: both sides are exact for the model except
      the central differences (outer h = 1e-4, inner h = 1e-5), whose O(h^2)
      truncation on derivatives of order one is ~1e-8; the bar is 1e-7.
  Q4  Translational invariance: the full gradient rows sum to zero (the
      model energy depends on relative positions only), for every scheme.

  Q5  The additive MM terms (qmmm_mm_terms) count each interaction once. On
      ethanol cut at the C-C bond (CH3 MM, CH2OH QM), every bond, angle and
      torsion is counted once by the wavefunction (all-QM) or the force field
      (touches an MM atom), every QM-QM pair and every QM-MM Coulomb pair once
      by the wavefunction, and every pair the force field counts (MM-MM
      Coulomb and LJ, QM-MM LJ) carries the full force field's weight (0 for
      1-2 and 1-3, scale_14 for 1-4, 1 beyond), the classes taken from an
      independent Floyd-Warshall bond distance against the transcribed BFS.
      Negative control: exclusions built without the cut bond give QM-MM
      Lennard-Jones pairs near the cut the wrong weight.

Mutations (each turns the script red; exact one-occurrence substitutions):
  link row onto the frontier atom with g instead of (1 - g)  (Q3 fails)
  midpoint force split 1.0 onto M1 instead of 0.5            (Q3 fails)
  RCD midpoint charge q0 instead of 2 q0                     (Q2 fails)
  a QM-MM Coulomb term added to the force field              (Q5 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# --- transcription of the partition -------------------------------------------


def partition(pos, charge, qm, bonds, scheme, g):
    """with_link_atoms + with_boundary_charges + active_charges.

    Returns (links, active): links = [(qm_full, mm_full, g, position)],
    active = [(q, position, target)] with target ("atom", i) or
    ("mid", (m1, m2)), in active_charges() order.
    """
    n = len(pos)
    is_qm = [i in qm for i in range(n)]
    links = []
    for a, b in bonds:
        if is_qm[a] == is_qm[b]:
            continue
        qf, mf = (a, b) if is_qm[a] else (b, a)
        p, q = pos[qf], pos[mf]
        links.append((qf, mf, g, [p[k] + g * (q[k] - p[k]) for k in range(3)]))
    eff = list(charge)
    boundary = []
    hosts = []
    for a, b in bonds:
        if is_qm[a] == is_qm[b]:
            continue
        m1 = b if is_qm[a] else a
        if m1 not in hosts:
            hosts.append(m1)
    if scheme != "keep":
        for m1 in hosts:
            q_m1 = charge[m1]
            if scheme == "z1":
                eff[m1] = 0
                continue
            m2s = []
            for a, b in bonds:
                other = b if a == m1 else a if b == m1 else None
                if other is None or is_qm[other]:
                    continue
                if other not in m2s:
                    m2s.append(other)
            q0 = q_m1 / len(m2s)
            eff[m1] = 0
            for m2 in m2s:
                mid = [(pos[m1][k] + pos[m2][k]) / 2 for k in range(3)]
                if scheme == "rcd":
                    eff[m2] = eff[m2] - q0
                    qb = 2 * q0
                else:
                    qb = q0
                boundary.append((qb, mid, ("mid", (m1, m2))))
    active = [
        (eff[i], pos[i], ("atom", i)) for i in range(n) if not is_qm[i] and eff[i] != 0
    ]
    active += [b for b in boundary if b[0] != 0]
    return links, active


def full_gradient(natoms, qm, links, active, qm_grad, mm_forces):
    """qmmm::full_gradient: QM rows, link chain rule, charge rows."""
    full = np.zeros((natoms, 3))
    for row, i in enumerate(qm):
        full[i] += qm_grad[row]
    for lnk, (q_full, m_full, g, _) in enumerate(links):
        d = qm_grad[len(qm) + lnk]
        full[q_full] += (1.0 - g) * d
        full[m_full] += g * d
    for (_, _, target), f in zip(active, mm_forces):
        if target[0] == "atom":
            full[target[1]] -= f
        else:
            m1, m2 = target[1]
            full[m1] -= 0.5 * f
            full[m2] -= 0.5 * f
    return full


# --- Q1 ------------------------------------------------------------------------


def claim_q1():
    g = sp.Symbol("g", positive=True)
    rq = sp.symbols("q0:3", real=True)
    rm = sp.symbols("m0:3", real=True)
    func = sp.Function("E")
    rl = [(1 - g) * rq[k] + g * rm[k] for k in range(3)]
    e = func(*rl)
    ok = True
    for k in range(3):
        de_dl = sp.diff(func(*sp.symbols("l0:3")), sp.Symbol(f"l{k}")).subs(
            dict(zip(sp.symbols("l0:3"), rl))
        )
        ok = ok and sp.simplify(sp.diff(e, rq[k]) - (1 - g) * de_dl) == 0
        ok = ok and sp.simplify(sp.diff(e, rm[k]) - g * de_dl) == 0
        ok = ok and sp.simplify(sp.diff(e, rq[k]) + sp.diff(e, rm[k]) - de_dl) == 0
    check("Q1 link chain rule (1 - g, g) and translation (claim Q1)", ok)


# --- Q2 ------------------------------------------------------------------------


def group_moments(pos, charges, idx, boundary):
    qt = sum(charges[i] for i in idx) + sum(b[0] for b in boundary)
    dip = [
        sum(charges[i] * pos[i][k] for i in idx) + sum(b[0] * b[1][k] for b in boundary)
        for k in range(3)
    ]
    return qt, dip


def claim_q2():
    q = sp.Symbol("q", real=True)
    ok = True
    rc_moves = True
    for n_m2 in range(1, 6):
        # atom 0 QM frontier, atom 1 = M1, atoms 2.. = M2
        pos = [
            [sp.Symbol(f"x{i}_{k}", real=True) for k in range(3)]
            for i in range(n_m2 + 2)
        ]
        cq = [sp.Integer(0), q] + [sp.Symbol(f"c{i}", real=True) for i in range(n_m2)]
        bonds = [(0, 1)] + [(1, 2 + j) for j in range(n_m2)]
        grp = list(range(1, n_m2 + 2))
        q_ref, d_ref = group_moments(pos, cq, grp, [])
        for scheme in ("rcd", "rc", "z1"):
            _, active = partition(pos, cq, [0], bonds, scheme, sp.Rational(7, 10))
            eff = [sp.Integer(0)] * len(pos)
            bnd = []
            for qa, p, t in active:
                if t[0] == "atom":
                    eff[t[1]] = qa
                else:
                    bnd.append((qa, p))
            qt, dp = group_moments(pos, eff, grp, bnd)
            dq = sp.simplify(qt - q_ref)
            dd = [sp.simplify(dp[k] - d_ref[k]) for k in range(3)]
            if scheme == "rcd":
                ok = ok and dq == 0 and all(x == 0 for x in dd)
            elif scheme == "rc":
                mean = [
                    sum(pos[2 + j][k] for j in range(n_m2)) / n_m2 for k in range(3)
                ]
                want = [q / 2 * (mean[k] - pos[1][k]) for k in range(3)]
                ok = ok and dq == 0
                ok = ok and all(sp.simplify(dd[k] - want[k]) == 0 for k in range(3))
                rc_moves = rc_moves and any(x != 0 for x in dd)
            else:
                ok = ok and sp.simplify(dq + q) == 0
    check(
        "Q2 RCD conserves charge and dipole, RC only charge, Z1 neither (claim Q2)",
        ok and rc_moves,
        "n_M2 = 1..5",
    )


# --- Q3, Q4 --------------------------------------------------------------------

ZQM = {0: 6.0, 1: 1.0, 2: 1.0, 3: 1.0}  # QM atoms: C, H, H, H


def model_energy(qm_pos, qm_z, charges):
    """Embedded-energy stand-in: nucleus-charge Coulomb, a Gaussian electron
    cloud on each QM atom seen by the charges, and QM-QM springs."""
    from scipy.special import erf

    e = 0.0
    for ra, za in zip(qm_pos, qm_z):
        for qc, rc in charges:
            r = np.linalg.norm(ra - rc)
            e += za * qc / r - 0.8 * za * qc * erf(r / 0.9) / r
    for a in range(len(qm_pos)):
        for b in range(a):
            r = np.linalg.norm(qm_pos[a] - qm_pos[b])
            e += 0.3 * (r - 1.8) ** 2 + 0.1 * qm_z[a] * qm_z[b] / r
    return e


def system():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],  # 0 C (QM frontier)
            [-0.7, 1.7, 0.3],  # 1 H
            [-0.6, -0.9, 1.6],  # 2 H
            [-0.8, -0.8, -1.5],  # 3 H
            [2.9, 0.1, -0.1],  # 4 C (M1)
            [3.6, 1.9, 0.4],  # 5 H (M2)
            [3.5, -0.9, 1.6],  # 6 H (M2)
            [3.7, -1.0, -1.5],  # 7 C (M2)
            [5.8, -0.9, -1.7],  # 8 O
        ]
    )
    charge = [0.0, 0.0, 0.0, 0.0, -0.18, 0.06, 0.06, 0.31, -0.55]
    qm = [0, 1, 2, 3]
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7), (7, 8)]
    return pos, charge, qm, bonds


def total_energy(flat, charge, qm, bonds, scheme, g):
    pos = flat.reshape(-1, 3)
    links, active = partition(list(pos), charge, qm, bonds, scheme, g)
    qm_pos = [pos[i] for i in qm] + [np.array(lk[3]) for lk in links]
    qm_z = [ZQM[i] for i in qm] + [1.0] * len(links)
    return model_energy(qm_pos, qm_z, [(qa, np.array(p)) for qa, p, _ in active])


def pieces(pos, charge, qm, bonds, scheme, g, h=1e-5):
    """qm_gradient (QM + link rows) and mm_forces by central difference of
    the model energy in its own variables: the stand-in for the SCF gradient
    and electric_field_at_points, not the quantity under test."""
    links, active = partition(list(pos), charge, qm, bonds, scheme, g)
    qm_pos = [pos[i].copy() for i in qm] + [np.array(lk[3]) for lk in links]
    qm_z = [ZQM[i] for i in qm] + [1.0] * len(links)
    chg = [(qa, np.array(p, dtype=float)) for qa, p, _ in active]
    grad = np.zeros((len(qm_pos), 3))
    for a in range(len(qm_pos)):
        for k in range(3):
            pp = [x.copy() for x in qm_pos]
            pm = [x.copy() for x in qm_pos]
            pp[a][k] += h
            pm[a][k] -= h
            grad[a, k] = (model_energy(pp, qm_z, chg) - model_energy(pm, qm_z, chg)) / (
                2 * h
            )
    forces = []
    for c in range(len(chg)):
        f = np.zeros(3)
        for k in range(3):
            cp = [(qa, p.copy()) for qa, p in chg]
            cm = [(qa, p.copy()) for qa, p in chg]
            cp[c][1][k] += h
            cm[c][1][k] -= h
            f[k] = -(
                model_energy(qm_pos, qm_z, cp) - model_energy(qm_pos, qm_z, cm)
            ) / (2 * h)
        forces.append(f)
    return links, active, grad, forces


def claim_q3_q4():
    pos, charge, qm, bonds = system()
    g = 1.09 / 1.53
    worst, worst_sum = 0.0, 0.0
    misses = []
    for scheme in ("keep", "z1", "rc", "rcd"):
        links, active, qg, mf = pieces(pos, charge, qm, bonds, scheme, g)
        full = full_gradient(len(pos), qm, links, active, qg, mf)
        h = 1e-4
        fd = np.zeros_like(full)
        flat = pos.ravel()
        for i in range(flat.size):
            fp, fm = flat.copy(), flat.copy()
            fp[i] += h
            fm[i] -= h
            fd.flat[i] = (
                total_energy(fp, charge, qm, bonds, scheme, g)
                - total_energy(fm, charge, qm, bonds, scheme, g)
            ) / (2 * h)
        worst = max(worst, np.abs(full - fd).max())
        worst_sum = max(worst_sum, np.abs(full.sum(axis=0)).max())
        if scheme == "rcd":
            swapped = [(a, b, 1 - gg, p) for a, b, gg, p in links]
            bad = full_gradient(len(pos), qm, swapped, active, qg, mf)
            misses.append(np.abs(bad - fd).max())
            on_m1 = [
                (qa, p, ("atom", t[1][0]) if t[0] == "mid" else t)
                for qa, p, t in active
            ]
            bad = full_gradient(len(pos), qm, links, on_m1, qg, mf)
            misses.append(np.abs(bad - fd).max())
    check(
        "Q3 full_gradient = FD of the partitioned energy, Keep/Z1/RC/RCD (claim Q3)",
        worst < 1e-7 and min(misses) > 1e-4,
        f"max err {worst:.1e} controls {[f'{m:.1e}' for m in misses]}",
    )
    check(
        "Q4 full gradient rows sum to zero (claim Q4)",
        worst_sum < 1e-8,
        f"max |sum| {worst_sum:.1e}",
    )


# --- Q5 ------------------------------------------------------------------------


def bfs_classes(n, bonds):
    """MmTopology::new: BFS to depth 3; depth 1-2 excluded, depth 3 a 1-4 pair."""
    adj = {a: [] for a in range(n)}
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)
    excl, p14 = set(), set()
    for start in range(n):
        depth, frontier = {start: 0}, [start]
        for d in (1, 2, 3):
            nxt = []
            for node in frontier:
                for nb in adj[node]:
                    if nb not in depth:
                        depth[nb] = d
                        nxt.append(nb)
            frontier = nxt
        for other, d in depth.items():
            pair = (min(start, other), max(start, other))
            if d in (1, 2):
                excl.add(pair)
            elif d == 3:
                p14.add(pair)
    return excl, p14


def additive_mm(n, bonds, terms, qm, s14):
    """qmmm_mm_terms bookkeeping: the weight the force field gives each term."""
    excl, p14 = bfs_classes(n, bonds)  # the full bond list, cut bond included
    w = {}
    for t in terms:  # bonded: kept iff it touches an MM atom
        w[t] = 1.0 if any(a not in qm for a in t) else 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ff = 0.0 if (i, j) in excl else (s14 if (i, j) in p14 else 1.0)
            if i in qm and j in qm:
                w[(i, j, "coul")] = w[(i, j, "lj")] = 0.0
            elif i in qm or j in qm:
                qm_mm_coul = 0.0  # the embedding holds it
                w[(i, j, "coul")] = qm_mm_coul
                w[(i, j, "lj")] = ff
            else:
                w[(i, j, "coul")] = w[(i, j, "lj")] = ff
    return w


def wavefunction_counts(n, terms, qm):
    """What the embedded SCF already contains: every all-QM term, QM-QM
    Coulomb and exchange-repulsion/dispersion, and QM-MM Coulomb (the
    point-charge embedding)."""
    w = {t: 1.0 if all(a in qm for a in t) else 0.0 for t in terms}
    for i in range(n):
        for j in range(i + 1, n):
            both, one = i in qm and j in qm, (i in qm) != (j in qm)
            w[(i, j, "coul")] = 1.0 if (both or one) else 0.0
            w[(i, j, "lj")] = 1.0 if both else 0.0
    return w


def graph_distance(n, bonds):
    """Floyd-Warshall shortest bond path: an independent route to 1-2/1-3/1-4."""
    big = 10**6
    dist = np.full((n, n), big)
    np.fill_diagonal(dist, 0)
    for a, b in bonds:
        dist[a, b] = dist[b, a] = 1
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


def claim_q5():
    # Ethanol cut at C-C: CH3 (0-3) is MM, CH2OH (4-8) is QM.
    n = 9
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7), (7, 8)]
    qm = {4, 5, 6, 7, 8}
    s14 = 0.5
    adj = {a: set() for a in range(n)}
    for a, b in bonds:
        adj[a].add(b)
        adj[b].add(a)
    angles = {
        (a, b, c) if a < c else (c, b, a)
        for b in range(n)
        for a in adj[b]
        for c in adj[b]
        if a != c
    }
    torsions = {
        (a, b, c, d) if a < d else (d, c, b, a)
        for b, c in bonds + [(y, x) for x, y in bonds]
        for a in adj[b] - {c}
        for d in adj[c] - {b, a}
    }
    terms = [tuple(sorted(b)) for b in bonds] + sorted(angles) + sorted(torsions)
    dist = graph_distance(n, bonds)

    def full_ff(i, j):
        d = dist[i, j]
        return 0.0 if d <= 2 else (s14 if d == 3 else 1.0)

    ff = additive_mm(n, bonds, terms, qm, s14)
    wf = wavefunction_counts(n, terms, qm)
    bad = []
    for key in ff:
        if len(key) == 3 and isinstance(key[2], str):
            # A pair the wavefunction holds must get no force-field weight;
            # any other pair gets exactly the full force field's weight.
            ok = ff[key] == 0.0 if wf[key] else ff[key] == full_ff(key[0], key[1])
            if not ok:
                bad.append(key)
        elif wf[key] + ff[key] != 1.0:
            bad.append(key)
    cross = [k for k in ff if len(k) == 3 and isinstance(k[2], str)]
    straddle = [
        t for t in terms if any(a in qm for a in t) and any(a not in qm for a in t)
    ]
    # Negative control: exclusions from the bond list without the cut bond
    # give the QM-MM Lennard-Jones pairs near the cut the wrong weight.
    ff_cut = additive_mm(n, [b for b in bonds if b != (0, 4)], terms, qm, s14)
    wrong = [
        k
        for k in cross
        if k[2] == "lj" and ff_cut[k] != full_ff(k[0], k[1]) and not wf[k]
    ]
    check(
        "Q5 additive MM terms: each term and pair once, FF pairs at full-FF weight (claim Q5)",
        not bad and len(wrong) > 0,
        f"{len(terms)} bonded ({len(straddle)} straddle the cut), {len(cross)} pair "
        f"components, double/missed {bad[:4]}; control: {len(wrong)} LJ pairs wrong "
        "without the cut bond",
    )


def main():
    claim_q1()
    claim_q2()
    claim_q3_q4()
    claim_q5()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all QM/MM boundary identities verified")


if __name__ == "__main__":
    main()
