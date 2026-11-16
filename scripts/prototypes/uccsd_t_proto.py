#!/usr/bin/env python3
"""Prototype of ferric's spin-blocked UCCSD(T) triples kernel (numpy).

The production kernel (crates/ferric-cc/src/u_ccsd_t.rs) streams over
canonical occupied triples per spin class and evaluates the spin-orbital (T)
formula of ccsd_t.rs directly on the UHF spin blocks:

    X(p,q,r; s,t,u) = sum_e t2[q,r,s,e] <tu||ep> - sum_m t2[p,m,t,u] <ms||qr>
    Y(p,q,r; s,t,u) = t1[p,s] <tu||qr>
    W = P(i/jk) P(a/bc) X,   V = P(i/jk) P(a/bc) Y
    E = sum_class w_class sum_{canonical occ} sum_{abc} (W + V) W / D

with classes aaa (i<j<k, w = 1/6), aab (i<j alpha, k beta, w = 1/2),
abb (i alpha, j<k beta, w = 1/2), bbb (w = 1/6). Virtuals run over the full
spin-matched blocks (a,b alpha, c beta for aab, and so on).

This script checks that design against PySCF on DF integrals: DF-UCCSD
amplitudes of OH (doublet, cc-pVDZ / cc-pVDZ-RI) and PySCF's UCCSD(T) on the
same DF ERIs. Frozen core 0 and 1.

Run: /home/matt/qc/ferric/.venv/bin/python scripts/prototypes/uccsd_t_proto.py
"""

import itertools

import numpy as np
from pyscf import cc, gto, lib, scf

A, B = 0, 1


def chem(b1, b2):
    return np.einsum("Ppq,Prs->pqrs", b1, b2)


class Blocks:
    """Per-spin RI blocks and amplitudes, indexed by spin (0 = alpha)."""

    def __init__(self, bov, boo, bvv, eo, ev, t1, t2):
        self.bov, self.boo, self.bvv = bov, boo, bvv
        self.eo, self.ev = eo, ev
        self.t1 = t1  # (t1a, t1b)
        self.t2aa, self.t2ab, self.t2bb = t2
        self.no = [eo[0].size, eo[1].size]
        self.nv = [ev[0].size, ev[1].size]
        # chemist blocks keyed by the two pair spins
        self.vvvo = {}
        self.oovo = {}
        self.vovo = {}
        for s1 in (A, B):
            for s2 in (A, B):
                bvo2 = bov[s2].transpose(0, 2, 1)
                self.vvvo[s1, s2] = chem(bvv[s1], bvo2)  # (te|up)
                self.oovo[s1, s2] = chem(boo[s1], bvo2)  # (mq|sr)
                self.vovo[s1, s2] = chem(bov[s1].transpose(0, 2, 1), bvo2)

    def t2_pair(self, q, r, ss, se):
        """t2[q,r,s,e] over (s,e) for fixed spin orbitals q, r."""
        (sq, iq), (sr, ir) = q, r
        if (sq, sr, ss, se) == (A, A, A, A):
            return self.t2aa[iq, ir]
        if (sq, sr, ss, se) == (B, B, B, B):
            return self.t2bb[iq, ir]
        if (sq, sr, ss, se) == (A, B, A, B):
            return self.t2ab[iq, ir]
        if (sq, sr, ss, se) == (A, B, B, A):
            return -self.t2ab[iq, ir].T
        if (sq, sr, ss, se) == (B, A, A, B):
            return -self.t2ab[ir, iq]
        if (sq, sr, ss, se) == (B, A, B, A):
            return self.t2ab[ir, iq].T
        return None

    def t2_row(self, p, sm, st, su):
        """t2[p,m,t,u] over (m,t,u) for a fixed spin orbital p."""
        sp_, ip = p
        key = (sp_, sm, st, su)
        if key == (A, A, A, A):
            return self.t2aa[ip]
        if key == (B, B, B, B):
            return self.t2bb[ip]
        if key == (A, B, A, B):
            return self.t2ab[ip]
        if key == (A, B, B, A):
            return -self.t2ab[ip].transpose(0, 2, 1)
        if key == (B, A, A, B):
            return -self.t2ab[:, ip]
        if key == (B, A, B, A):
            return self.t2ab[:, ip].transpose(0, 2, 1)
        return None

    def x_block(self, p, q, r, ss, st, su):
        """X(p,q,r; s,t,u) over spin-matched virtual ranges."""
        out = np.zeros((self.nv[ss], self.nv[st], self.nv[su]))
        sp_, ip = p
        sq, iq = q
        sr, ir = r
        # term 1: sum_e t2[q,r,s,e] <tu||ep>
        for se in (A, B):
            t2qr = self.t2_pair(q, r, ss, se)
            if t2qr is None:
                continue
            if st == se and su == sp_:  # (te|up)
                g = self.vvvo[st, su][:, :, :, ip]  # [t,e,u]
                out += np.einsum("se,teu->stu", t2qr, g)
            if st == sp_ and su == se:  # -(tp|ue)
                g = self.vvvo[su, st][:, :, :, ip]  # (ue|tp) [u,e,t]
                out -= np.einsum("se,uet->stu", t2qr, g)
        # term 2: - sum_m t2[p,m,t,u] <ms||qr>
        for sm in (A, B):
            t2p = self.t2_row(p, sm, st, su)
            if t2p is None:
                continue
            if sm == sq and ss == sr:  # (mq|sr)
                g = self.oovo[sm, ss][:, iq, :, ir]  # [m,s]
                out -= np.einsum("ms,mtu->stu", g, t2p)
            if sm == sr and ss == sq:  # -(mr|sq)
                g = self.oovo[sm, ss][:, ir, :, iq]
                out += np.einsum("ms,mtu->stu", g, t2p)
        return out

    def y_block(self, p, q, r, ss, st, su):
        """Y(p,q,r; s,t,u) = t1[p,s] <tu||qr>."""
        out = np.zeros((self.nv[ss], self.nv[st], self.nv[su]))
        sp_, ip = p
        sq, iq = q
        sr, ir = r
        if sp_ != ss:
            return out
        g = np.zeros((self.nv[st], self.nv[su]))
        if st == sq and su == sr:
            g += self.vovo[st, su][:, iq, :, ir]
        if st == sr and su == sq:
            g -= self.vovo[st, su][:, ir, :, iq]
        return np.einsum("s,tu->stu", self.t1[ss][ip], g)

    def wv(self, i, j, k, spins_v, fn):
        """P(i/jk) P(a/bc) fn over the canonical virtual layout spins_v."""
        sa, sb, sc = spins_v
        out = np.zeros((self.nv[sa], self.nv[sb], self.nv[sc]))
        for occ, so in (((i, j, k), 1.0), ((j, i, k), -1.0), ((k, j, i), -1.0)):
            out += so * fn(*occ, sa, sb, sc)
            out -= so * fn(*occ, sb, sa, sc).transpose(1, 0, 2)
            out -= so * fn(*occ, sc, sb, sa).transpose(2, 1, 0)
        return out

    def triple(self, i, j, k, spins_v):
        w = self.wv(i, j, k, spins_v, self.x_block)
        v = self.wv(i, j, k, spins_v, self.y_block)
        sa, sb, sc = spins_v
        d = (
            self.eo[i[0]][i[1]]
            + self.eo[j[0]][j[1]]
            + self.eo[k[0]][k[1]]
            - self.ev[sa][:, None, None]
            - self.ev[sb][None, :, None]
            - self.ev[sc][None, None, :]
        )
        return np.sum((w + v) * w / d)

    def energy(self):
        na, nb = self.no
        e = {}
        e["aaa"] = (
            sum(
                self.triple((A, i), (A, j), (A, k), (A, A, A))
                for i, j, k in itertools.combinations(range(na), 3)
            )
            / 6.0
        )
        e["aab"] = (
            sum(
                self.triple((A, i), (A, j), (B, k), (A, A, B))
                for i, j in itertools.combinations(range(na), 2)
                for k in range(nb)
            )
            / 2.0
        )
        e["abb"] = (
            sum(
                self.triple((A, i), (B, j), (B, k), (A, B, B))
                for i in range(na)
                for j, k in itertools.combinations(range(nb), 2)
            )
            / 2.0
        )
        e["bbb"] = (
            sum(
                self.triple((B, i), (B, j), (B, k), (B, B, B))
                for i, j, k in itertools.combinations(range(nb), 3)
            )
            / 6.0
        )
        return e


def run(frozen):
    mol = gto.M(
        atom="O 0 0 0; H 0 0 0.9697",
        basis="cc-pvdz",
        spin=1,
        verbose=0,
    )
    mf = scf.UHF(mol).density_fit(auxbasis="cc-pvdz-ri")
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.UCCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-9
    mycc.kernel()
    et_ref = mycc.ccsd_t()

    with_df = mf.with_df
    cderi = lib.unpack_tril(with_df._cderi)
    mask_a, mask_b = mycc.get_frozen_mask()
    nocca, noccb = mycc.nocc
    bov, boo, bvv, eo, ev = [], [], [], [], []
    for s, mask in ((0, mask_a), (1, mask_b)):
        c = mf.mo_coeff[s][:, mask]
        e = mf.mo_energy[s][mask]
        no = (nocca, noccb)[s]
        L = np.einsum("Pmn,mp,nq->Ppq", cderi, c, c)
        bov.append(L[:, :no, no:])
        boo.append(L[:, :no, :no])
        bvv.append(L[:, no:, no:])
        eo.append(e[:no])
        ev.append(e[no:])
    blk = Blocks(bov, boo, bvv, eo, ev, mycc.t1, mycc.t2)
    e = blk.energy()
    et = sum(e.values())
    print(
        f"frozen={frozen}: E(T) streaming = {et:.12f}  PySCF = {et_ref:.12f}  "
        f"diff = {et - et_ref:.2e}  classes = "
        + ", ".join(f"{k} {v:.3e}" for k, v in e.items())
    )
    assert abs(et - et_ref) < 1e-10, "streaming spin-blocked (T) != PySCF UCCSD(T)"


if __name__ == "__main__":
    run(0)
    run(1)
