#!/usr/bin/env python
"""Orbital Pade MP3 (OP-MP3, SOP-MP3, SSD-OPMP3) prototype in numpy on ferric.

After K. Bystrom, T. C. Berkelbach, D. Hait, arXiv:2609.25330 (2026). Working
equations, conventions and parameters: opmp3_derivation.md (equation numbers
there are the paper's). Correctness reference for a Rust port; nothing about
speed may be quoted from it except the cost table it prints itself.

Pipeline:

1. Reference: ferric ``run_rhf`` / ``run_uhf`` orbitals (canonical, so the
   semicanonicalization of the paper is the identity and E_NBS = 0).
2. Spatial MO integrals from ``AoIntegrals``: exact (``ao2mo``) or density
   fitted (``df_mo``, only to compare with ferric's RI ``run_mp3``).
3. Spin-orbital first- and second-order doubles amplitudes, Eqs. 7 and 9.
   Every block is spin-orbital except the particle-particle ladder, which is
   contracted against the spatial (ac|bd) of each spin pair (the spin-orbital
   VVVV tensor is 16x the spatial one).
4. W^(n)_ij, Eq. 11a, split by the spin of the summed occupied k into the
   same-spin and opposite-spin parts of Eqs. 16-18.
5. The resummed models (Eqs. 13-15, 19-22) and the scalar Pade (Eq. 1).

``--anchor`` runs one block of checks; ``all`` runs every one in order:
``trivial`` (series limit == ferric MP2 / MP3), ``cross`` (exact-integral
traces vs PySCF MP2 and ADC MP3), ``eq18`` (the two OS constructions agree),
``size`` (OP models size-consistent, scalar Pade not), ``repro`` (the paper's
W matrices give the paper's energies), ``wrepro`` (ferric W == paper W for
HF/aug-cc-pCVTZ molecules), ``mut`` (each mutation must fail a check),
``cost`` (wall time vs run_mp3).

Usage (one heavy op at a time on the shared box):
  cd ~/qc/ferric && PYTHONPATH=<shim> box-lease exec -- env \\
    OPENBLAS_NUM_THREADS=1 .venv/bin/python \\
    ~/qc/ferric-opmp3/scripts/prototypes/opmp3_proto.py --anchor all
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import ferric

EPS = np.finfo(float).eps
HARTREE_TO_KCAL = 627.509474  # CODATA 2018, as in PySCF's nist module

# SSD-OPMP3 same-spin damping per reference (paper Fig. 3; x_os = 1).
X_SS = {"KOOMP2": 0.45, "PBE0": 0.50, "HF": 0.0}

# ---------------------------------------------------------------------------
# Pade algebra on the W matrices (paper Eqs. 1, 13-15, 19-22)
# ---------------------------------------------------------------------------


def pade(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """P(A, B) = A (A - B)^{-1} A, Eq. 13.

    A is W^(2) (negative semidefinite), B is x W^(3). A zero-sized or all-zero
    block (a one-electron same-spin channel, frozen-core rows) contributes
    zero: rows/columns where A vanishes are dropped first, as the paper's
    W matrices carry frozen orbitals as zero rows and columns.
    """
    if a.size == 0:
        return a.copy()
    keep = np.flatnonzero(np.abs(a).sum(axis=0) + np.abs(a).sum(axis=1) > 0)
    out = np.zeros_like(a)
    if keep.size == 0:
        return out
    ak = a[np.ix_(keep, keep)]
    bk = b[np.ix_(keep, keep)]
    out[np.ix_(keep, keep)] = ak @ np.linalg.solve(ak - bk, ak)
    return out


def pade_series(a: np.ndarray, b: np.ndarray, order: int = 1) -> np.ndarray:
    """Truncated Neumann series of P(A, B): A + B + B A^{-1} B + ...

    order=1 is A + B, i.e. the resummation switched off: its trace is
    E^(2) + E^(3) exactly (the trivial limit of the anchor).
    """
    out = a + b
    if order >= 2:
        out = out + b @ np.linalg.solve(a, b)
    return out


def tr(m: np.ndarray) -> float:
    return float(np.trace(m))


@dataclass
class WSet:
    """The eight W matrices: w[(n, spin, channel)], n in {2,3}, spin in
    {'a','b'}, channel in {'ss','os'}; plus the non-Brillouin singles energy."""

    w: dict
    e_nbs: float = 0.0
    e_ref: float = 0.0

    def get(self, n: int, s: str, ch: str) -> np.ndarray:
        return self.w[(n, s, ch)]

    def full(self, n: int, s: str) -> np.ndarray:
        return self.w[(n, s, "ss")] + self.w[(n, s, "os")]

    def e2(self) -> float:
        return sum(tr(self.full(2, s)) for s in "ab")

    def e3(self) -> float:
        return sum(tr(self.full(3, s)) for s in "ab")


def e_mp2x(ws: WSet, x: float) -> float:
    return ws.e2() + x * ws.e3()


def e_scalar_pade(ws: WSet) -> float:
    """Eq. 1: E2^2 / (E2 - E3); NOT size-consistent (Eq. 3)."""
    e2, e3 = ws.e2(), ws.e3()
    if e2 == 0.0:  # one-electron system: no correlation (the paper's convention)
        return 0.0
    return e2 * e2 / (e2 - e3)


def e_opmp3(ws: WSet, x: float = 1.0, pfun=pade) -> float:
    """Eq. 15 (Eq. 14 at x = 1): Tr P(W2, x W3) on the spin-blocked W."""
    return sum(tr(pfun(ws.full(2, s), x * ws.full(3, s))) for s in "ab")


def e_sopmp3(ws: WSet, x_ss: float = 1.0, x_os: float = 1.0, pfun=pade) -> float:
    """Eqs. 19-21: separate Pade of the SS and OS channels, per spin."""
    e = 0.0
    for s in "ab":
        e += tr(pfun(ws.get(2, s, "ss"), x_ss * ws.get(3, s, "ss")))
        e += tr(pfun(ws.get(2, s, "os"), x_os * ws.get(3, s, "os")))
    return e


def e_ssd_opmp3(ws: WSet, x_ss: float, pfun=pade) -> float:
    """Eq. 22: SOP-MP3 with x_os = 1."""
    return e_sopmp3(ws, x_ss, 1.0, pfun)


# ---------------------------------------------------------------------------
# Integrals and the reference
# ---------------------------------------------------------------------------


@dataclass
class SpinRef:
    """Per-spin occupied/virtual MO coefficients and orbital energies."""

    co: tuple  # (C_occ_alpha, C_occ_beta)
    cv: tuple
    eo: tuple
    ev: tuple
    e_ref: float
    restricted: bool


def scf_reference(mol, basis, kind: str, **scf_kw) -> SpinRef:
    """RHF (closed shell) or UHF reference from ferric, canonical orbitals."""
    ints = ferric.AoIntegrals(mol, basis)
    s = ints.overlap()
    if kind == "rhf":
        res = ferric.run_rhf(mol, basis, **scf_kw)
        c = res.mo_coefficients()
        e = res.orbital_energies()
        no = int(round(np.trace(res.density() @ s))) // 2
        co, cv, eo, ev = c[:, :no], c[:, no:], e[:no], e[no:]
        return SpinRef((co, co), (cv, cv), (eo, eo), (ev, ev), res.energy, True)
    res = ferric.run_uhf(mol, basis, **scf_kw)
    out = []
    for c, e, d in (
        (
            res.mo_coefficients_alpha(),
            res.orbital_energies_alpha(),
            res.density_alpha(),
        ),
        (res.mo_coefficients_beta(), res.orbital_energies_beta(), res.density_beta()),
    ):
        no = int(round(np.trace(d @ s)))
        out.append((c[:, :no], c[:, no:], e[:no], e[no:]))
    (coa, cva, eoa, eva), (cob, cvb, eob, evb) = out
    return SpinRef((coa, cob), (cva, cvb), (eoa, eob), (eva, evb), res.energy, False)


class Integrals:
    """Spatial chemist (pq|rs) over MO blocks, exact or density fitted."""

    def __init__(self, mol, basis, aux=None, max_gb: float = 8.0):
        self.ints = ferric.AoIntegrals(mol, basis)
        self.aux = aux
        self.max_gb = max_gb
        self._b = {}

    def _bt(self, c1, c2):
        key = (id(c1), id(c2))
        if key not in self._b:
            self._b[key] = (self.ints.df_mo(self.aux, c1, c2), c1, c2)
        return self._b[key][0]

    def g(self, c1, c2, c3, c4) -> np.ndarray:
        if any(c.shape[1] == 0 for c in (c1, c2, c3, c4)):
            return np.zeros((c1.shape[1], c2.shape[1], c3.shape[1], c4.shape[1]))
        if self.aux is None:
            return self.ints.ao2mo(c1, c2, c3, c4, max_gb=self.max_gb)
        b12 = self._bt(c1, c2)
        b34 = self._bt(c3, c4)
        n1, n2 = b12.shape[1:]
        n3, n4 = b34.shape[1:]
        return (b12.reshape(-1, n1 * n2).T @ b34.reshape(-1, n3 * n4)).reshape(
            n1, n2, n3, n4
        )


# ---------------------------------------------------------------------------
# Spin-orbital amplitudes and W (paper Eqs. 7, 9, 11a, 16-18)
# ---------------------------------------------------------------------------


class SpinOrbitalMP:
    """Spin-orbital c^(1), c^(2) and W^(2), W^(3) from a SpinRef.

    Spin orbitals are ordered [alpha..., beta...] within the occupied and
    within the virtual space. <pq|rs> = (pr|qs) delta(s_p s_r) delta(s_q s_s).
    """

    def __init__(self, ref: SpinRef, ints: Integrals):
        self.ref = ref
        self.ints = ints
        self.nocc = (ref.co[0].shape[1], ref.co[1].shape[1])
        self.nvir = (ref.cv[0].shape[1], ref.cv[1].shape[1])
        self.osl = (slice(0, self.nocc[0]), slice(self.nocc[0], sum(self.nocc)))
        self.vsl = (slice(0, self.nvir[0]), slice(self.nvir[0], sum(self.nvir)))
        self.ospin = np.array([0] * self.nocc[0] + [1] * self.nocc[1])
        self.eo = np.concatenate(ref.eo)
        self.ev = np.concatenate(ref.ev)
        self.timings = {}

    def _c(self, kind, s):
        return (self.ref.co if kind == "o" else self.ref.cv)[s]

    def _sl(self, kind, s):
        return (self.osl if kind == "o" else self.vsl)[s]

    def _n(self, kind):
        return sum(self.nocc) if kind == "o" else sum(self.nvir)

    def chem_so(self, k1, k2, k3, k4) -> np.ndarray:
        """Spin-orbital chemist (12|34): nonzero for s1 == s2 and s3 == s4."""
        out = np.zeros((self._n(k1), self._n(k2), self._n(k3), self._n(k4)))
        cache = {}
        for s in (0, 1):
            for t in (0, 1):
                cs = (self._c(k1, s), self._c(k2, s), self._c(k3, t), self._c(k4, t))
                key = tuple(id(c) for c in cs)
                if key not in cache:
                    cache[key] = self.ints.g(*cs)
                out[
                    self._sl(k1, s), self._sl(k2, s), self._sl(k3, t), self._sl(k4, t)
                ] = cache[key]
        return out

    def build(self):
        t0 = time.perf_counter()
        ovov = self.chem_so("o", "v", "o", "v")
        phys_oovv = ovov.transpose(0, 2, 1, 3)  # <ij|ab> = (ia|jb)
        self.oovv = phys_oovv - phys_oovv.transpose(0, 1, 3, 2)  # <ij||ab>
        oooo = self.chem_so("o", "o", "o", "o").transpose(
            0, 2, 1, 3
        )  # <kl|ij> = (ki|lj)
        self.oooo = oooo - oooo.transpose(0, 1, 3, 2)
        kcbj = self.chem_so("o", "v", "v", "o")  # (kc|bj) -> <kb|cj>
        kjbc = self.chem_so("o", "o", "v", "v")  # (kj|bc) -> <kb|jc>
        v_kbcj = kcbj.transpose(0, 2, 1, 3)
        v_kbjc = kjbc.transpose(0, 2, 1, 3)
        self.ovvo = v_kbcj - v_kbjc.transpose(0, 1, 3, 2)  # <kb||cj>
        self.timings["integrals_small"] = time.perf_counter() - t0

        d = (
            self.ev[None, None, :, None]
            + self.ev[None, None, None, :]
            - self.eo[:, None, None, None]
            - self.eo[None, :, None, None]
        )
        self.delta = d
        self.c1 = -self.oovv / d  # Eq. 7

        t0 = time.perf_counter()
        ladder = self.pp_ladder(self.c1)
        self.timings["vvvv"] = time.perf_counter() - t0
        hh = 0.5 * np.einsum("klij,klab->ijab", self.oooo, self.c1, optimize=True)
        r = np.einsum("kbcj,ikac->ijab", self.ovvo, self.c1, optimize=True)
        ring = (
            r
            - r.transpose(1, 0, 2, 3)
            - r.transpose(0, 1, 3, 2)
            + r.transpose(1, 0, 3, 2)
        )
        self.c2 = -(ladder + hh + ring) / d  # Eq. 9
        return self

    def pp_ladder(self, c: np.ndarray) -> np.ndarray:
        """1/2 sum_cd <ab||cd> c_ijcd = sum_cd (ac|bd) c_ijcd, per spin pair."""
        out = np.zeros_like(c)
        cache = {}
        for s in (0, 1):
            for t in (0, 1):
                if self.nvir[s] == 0 or self.nvir[t] == 0:
                    continue
                cs = (
                    self._c("v", s),
                    self._c("v", s),
                    self._c("v", t),
                    self._c("v", t),
                )
                key = tuple(id(x) for x in cs)
                if key not in cache:
                    cache[key] = self.ints.g(*cs)
                g = cache[key]  # (ac|bd), a,c spin s; b,d spin t
                vs, vt = self.vsl[s], self.vsl[t]
                out[:, :, vs, vt] += np.einsum(
                    "acbd,ijcd->ijab", g, c[:, :, vs, vt], optimize=True
                )
        return out

    def energies(self):
        e2 = 0.25 * float(np.sum(self.c1 * self.oovv))
        e3 = 0.25 * float(np.sum(self.c2 * self.oovv))  # Eq. 6
        return e2, e3

    def w_by_k_spin(self, c: np.ndarray):
        """Eq. 11a split by the spin of k: returns {(spin_ij, 'ss'|'os'): W}."""
        out = {}
        for ksp in (0, 1):
            ks = self.osl[ksp]
            y = np.einsum("ikab,jkab->ij", c[:, ks], self.oovv[:, ks], optimize=True)
            w = 0.125 * (y + y.T)
            for s, name in ((0, "a"), (1, "b")):
                blk = w[self.osl[s], self.osl[s]]
                out[(name, "ss" if ksp == s else "os")] = blk
        return out

    def w_os_eq18(self, c: np.ndarray, s: int) -> np.ndarray:
        """Eq. 18 literally: 1/4 sum_{k sbar, a s, b sbar} c_ikab <jk|ab> + (i<->j)."""
        sb = 1 - s
        os_, ok = self.osl[s], self.osl[sb]
        va, vb = self.vsl[s], self.vsl[sb]
        cc = c[os_, ok, va, vb]
        # <jk|ab> (non-antisymmetrized), j spin s, k spin sbar: (ja|kb)
        g = self.ints.g(
            self.ref.co[s], self.ref.cv[s], self.ref.co[sb], self.ref.cv[sb]
        )
        phys = g.transpose(0, 2, 1, 3)
        y = np.einsum("ikab,jkab->ij", cc, phys, optimize=True)
        return 0.25 * (y + y.T)

    def wset(self) -> WSet:
        w = {}
        for n, c in ((2, self.c1), (3, self.c2)):
            for (s, ch), m in self.w_by_k_spin(c).items():
                w[(n, s, ch)] = m
        return WSet(w, 0.0, self.ref.e_ref)


def compute_wset(mol, basis, kind="rhf", aux=None, **scf_kw):
    ref = scf_reference(mol, basis, kind, **scf_kw)
    mp = SpinOrbitalMP(ref, Integrals(mol, basis, aux)).build()
    return mp, mp.wset()


# ---------------------------------------------------------------------------
# Geometries (Angstrom)
# ---------------------------------------------------------------------------

H2O = "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692"
NH3 = "N 0.0 0.0 0.1162\nH 0.0 0.9397 -0.2711\nH 0.8138 -0.4699 -0.2711\nH -0.8138 -0.4699 -0.2711"
N2 = "N 0.0 0.0 0.0\nN 0.0 0.0 1.0977"
CH4 = (
    "C 0.0 0.0 0.0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\n"
    "H -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276"
)
C2H4 = (
    "C 0.0 0.0 0.6670\nC 0.0 0.0 -0.6670\nH 0.0 0.9228 1.2321\nH 0.0 -0.9228 1.2321\n"
    "H 0.0 0.9228 -1.2321\nH 0.0 -0.9228 -1.2321"
)
OH = "O 0.0 0.0 0.0\nH 0.0 0.0 0.9697"
CH2_TRIPLET = "C 0.0 0.0 0.1027\nH 0.0 0.9940 -0.3081\nH 0.0 -0.9940 -0.3081"


def mol_from(atoms: str, charge: int = 0, mult: int = 1, shift: float = 0.0):
    lines = [ln.split() for ln in atoms.strip().splitlines()]
    body = "\n".join(
        f"{s} {float(x)} {float(y)} {float(z) + shift}" for s, x, y, z in lines
    )
    return ferric.Molecule.from_xyz_string(f"{len(lines)}\n\n{body}\n", charge, mult)


def pyscf_mol(atoms: str, basis: str, charge: int = 0, spin: int = 0):
    from pyscf import gto

    return gto.M(
        atom=atoms.replace("\n", "; "), basis=basis, charge=charge, spin=spin, verbose=0
    )


SCF_TIGHT = {"energy_conv": 1e-12, "density_conv": 1e-10}


class Report:
    """Collects (check, measured, bar, pass) rows; a failed row fails the run."""

    def __init__(self):
        self.rows = []

    def check(
        self, name: str, measured: float, bar: float, expect_fail: bool = False
    ) -> bool:
        ok = abs(measured) <= bar
        passed = (not ok) if expect_fail else ok
        tag = "PASS" if passed else "FAIL"
        rel = ">" if expect_fail else "<="
        print(f"  [{tag}] {name}: |{measured:.3e}| {rel} {bar:.1e}", flush=True)
        self.rows.append(
            {
                "check": name,
                "measured": measured,
                "bar": bar,
                "expect_fail": expect_fail,
                "pass": passed,
            }
        )
        return passed


# ---------------------------------------------------------------------------
# Anchor: trivial limit (resummation off == ferric MP2/MP3)
# ---------------------------------------------------------------------------


def taylor_checks(rep: Report, ws: WSet, label: str, pfun=pade) -> None:
    """The Pade's own Taylor coefficients at x = 0 (the [1/1] identity, Eq. 2).

    f(x) = Tr P(W2, x W3) = sum_n x^n c_n with c_0 = E2, c_1 = E3,
    c_n = Tr (W3 W2^-1)^(n-1) W3. Central differences at h; bars are the
    truncation terms, 10 h^2 |c_3| for f'(0) and 10 * 2 h^2 |c_4| for
    f''(0) = 2 c_2, plus rounding 100 eps |E2| / h^k.
    """
    h = 1e-3
    f = {x: e_opmp3(ws, x, pfun) for x in (-h, 0.0, h)}
    c = np.zeros(5)
    for s in "ab":
        a, b = ws.full(2, s), ws.full(3, s)
        keep = np.flatnonzero(np.abs(a).sum(axis=0) > 0)
        if keep.size == 0:
            continue
        a, b = a[np.ix_(keep, keep)], b[np.ix_(keep, keep)]
        ainv_b = np.linalg.solve(a, b)
        m = b.copy()
        c[0] += tr(a)
        for n in range(1, 5):
            c[n] += tr(m)
            m = m @ ainv_b
    e2, e3 = ws.e2(), ws.e3()
    rnd = 100 * EPS * abs(e2)
    rep.check(f"{label} f(0) - E2", f[0.0] - e2, rnd)
    d1 = (f[h] - f[-h]) / (2 * h)
    rep.check(f"{label} f'(0) - E3", d1 - e3, 10 * h * h * abs(c[3]) + rnd / h)
    d2 = (f[h] + f[-h] - 2 * f[0.0]) / (h * h)
    rep.check(
        f"{label} f''(0)/2 - Tr W3 W2^-1 W3",
        0.5 * d2 - c[2],
        0.5 * (10 * 2 * h * h * abs(c[4]) + 4 * rnd / (h * h)),
    )


def anchor_trivial(rep: Report, pfun=pade) -> None:
    print("== trivial: resummation off reproduces ferric MP2 / MP3 (RI, cc-pVDZ)")
    bs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    for name, atoms in (("H2O", H2O), ("NH3", NH3)):
        mol = mol_from(atoms)
        r3 = ferric.run_mp3(mol, bs, aux, scf_energy_conv=1e-12, scf_density_conv=1e-10)
        r2 = ferric.run_rimp2(
            mol, bs, aux, scf_energy_conv=1e-12, scf_density_conv=1e-10
        )
        mp, ws = compute_wset(mol, bs, "rhf", aux=aux, **SCF_TIGHT)
        e2a, e3a = mp.energies()
        print(
            f"  {name}: Tr W2 {ws.e2():.12f}  run_mp3.e_mp2 {r3.e_mp2:.12f}  "
            f"run_rimp2 {r2.mp2_corr:.12f}  Tr W3 {ws.e3():.12f}  run_mp3.e_mp3 {r3.e_mp3:.12f}"
        )
        # Bar 1e-10 Ha: same RI integrals, orbitals from the same SCF code at
        # density_conv 1e-10 (first-order energy error <= ~|E_c| * 1e-10).
        rep.check(f"{name} Tr W2 - run_mp3.e_mp2", ws.e2() - r3.e_mp2, 1e-10)
        rep.check(f"{name} Tr W2 - run_rimp2.mp2_corr", ws.e2() - r2.mp2_corr, 1e-10)
        rep.check(f"{name} Tr W3 - run_mp3.e_mp3", ws.e3() - r3.e_mp3, 1e-10)
        rep.check(
            f"{name} Tr W2 - 1/4 sum c1 <||> (Eq 11b)",
            ws.e2() - e2a,
            100 * EPS * abs(e2a),
        )
        rep.check(
            f"{name} Tr W3 - 1/4 sum c2 <||> (Eq 11b)",
            ws.e3() - e3a,
            100 * EPS * abs(e2a),
        )
        # Series switch-off: every model collapses to E2 + E3 (x = 1).
        ser = lambda a, b: pade_series(a, b, 1)  # noqa: E731
        for mname, val in (
            ("OP-MP3 series", e_opmp3(ws, 1.0, ser)),
            ("SOP-MP3 series", e_sopmp3(ws, 1.0, 1.0, ser)),
        ):
            rep.check(f"{name} {mname} - (MP2+MP3)", val - (r3.e_mp2 + r3.e_mp3), 1e-10)
        taylor_checks(rep, ws, f"{name} Pade", pfun)


# ---------------------------------------------------------------------------
# Anchor: exact-integral traces vs PySCF (independent construction)
# ---------------------------------------------------------------------------


def pyscf_mp2_mp3(atoms: str, basis: str, charge: int = 0, spin: int = 0):
    """PySCF MP2 (mp.MP2/UMP2) and MP3 (ADC(3) ground state) correlation."""
    from pyscf import adc, mp, scf

    m = pyscf_mol(atoms, basis, charge, spin)
    mf = (scf.RHF(m) if spin == 0 else scf.UHF(m)).set(
        conv_tol=1e-12, conv_tol_grad=1e-9
    )
    mf.kernel()
    e_mp2 = (mp.MP2(mf) if spin == 0 else mp.UMP2(mf)).kernel()[0]
    a = adc.RADC(mf) if spin == 0 else adc.UADC(mf)
    a.method = "adc(3)"
    a.verbose = 0
    e_mp3_total_corr = a.kernel_gs()[0]
    return mf.e_tot, e_mp2, e_mp3_total_corr - e_mp2


def anchor_cross(rep: Report) -> None:
    print("== cross: exact-integral Tr W vs PySCF MP2 / ADC(3) MP3 (cc-pVDZ)")
    bs = ferric.BasisSet.bundled("cc-pvdz")
    for name, atoms, mult, kind in (
        ("H2O", H2O, 1, "rhf"),
        ("OH", OH, 2, "uhf"),
        ("CH2(3B1)", CH2_TRIPLET, 3, "uhf"),
    ):
        mol = mol_from(atoms, 0, mult)
        mp, ws = compute_wset(mol, bs, kind, **SCF_TIGHT)
        e_hf, p2, p3 = pyscf_mp2_mp3(atoms, "cc-pvdz", 0, mult - 1)
        print(
            f"  {name}: E_HF ferric {ws.e_ref:.10f} pyscf {e_hf:.10f}; "
            f"E2 {ws.e2():.10f} / {p2:.10f}; E3 {ws.e3():.10f} / {p3:.10f}"
        )
        # Bar 1e-8 Ha: two SCF codes, two basis-set parsers; the HF energies
        # themselves are compared at the same bar.
        rep.check(f"{name} E_HF ferric - pyscf", ws.e_ref - e_hf, 1e-8)
        rep.check(f"{name} Tr W2 - pyscf MP2", ws.e2() - p2, 1e-8)
        rep.check(f"{name} Tr W3 - pyscf ADC(3) MP3", ws.e3() - p3, 1e-8)


def anchor_eq18(rep: Report) -> None:
    print(
        "== eq18: OS W from the k-spin split of Eq. 11a == Eq. 18 literal (UHF OH, cc-pVDZ)"
    )
    bs = ferric.BasisSet.bundled("cc-pvdz")
    mol = mol_from(OH, 0, 2)
    mp, ws = compute_wset(mol, bs, "uhf", **SCF_TIGHT)
    for n, c in ((2, mp.c1), (3, mp.c2)):
        for s, name in ((0, "a"), (1, "b")):
            lit = mp.w_os_eq18(c, s)
            diff = np.max(np.abs(lit - ws.get(n, name, "os")))
            rep.check(
                f"W{n} OS {name}: max|Eq18 - split|",
                diff,
                1e3 * EPS * np.max(np.abs(lit)),
            )
        full = 0.125 * np.einsum("ikab,jkab->ij", c, mp.oovv)
        full = full + full.T
        rep.check(
            f"W{n}: alpha-beta block of Eq. 11a",
            np.max(np.abs(full[mp.osl[0], mp.osl[1]])),
            1e3 * EPS * np.max(np.abs(full)),
        )


# ---------------------------------------------------------------------------
# Anchor: size consistency
# ---------------------------------------------------------------------------


def all_models(
    ws: WSet, x_ss: float = 0.45, pfun=pade, mut_nodamp: bool = False
) -> dict:
    xs = 1.0 if mut_nodamp else x_ss
    return {
        "MP2": ws.e2(),
        "MP3": ws.e2() + ws.e3(),
        "MP2.X(0.8365)": e_mp2x(ws, 0.8365),
        "scalar Pade (Eq 1)": e_scalar_pade(ws),
        "OP-MP3 (Eq 14)": e_opmp3(ws, 1.0, pfun),
        "OP-MP3(0.789)": e_opmp3(ws, 0.7890, pfun),
        "SOP-MP3 (Eq 21)": e_sopmp3(ws, 1.0, 1.0, pfun),
        f"SSD-OPMP3({x_ss})": e_ssd_opmp3(ws, xs, pfun),
    }


def anchor_size(rep: Report, pfun=pade, scalar_must_fail: bool = True) -> None:
    print("== size: N2 ... CH4 at 100 Angstrom (cc-pVDZ, exact integrals, RHF)")
    bs = ferric.BasisSet.bundled("cc-pvdz")
    _, wa = compute_wset(mol_from(N2), bs, "rhf", **SCF_TIGHT)
    _, wb = compute_wset(mol_from(CH4), bs, "rhf", **SCF_TIGHT)
    dimer = (
        N2
        + "\n"
        + "\n".join(
            f"{s} {x} {y} {float(z) + 100.0}"
            for s, x, y, z in (ln.split() for ln in CH4.splitlines())
        )
    )
    _, wab = compute_wset(mol_from(dimer), bs, "rhf", **SCF_TIGHT)
    ma, mb, mab = (
        all_models(wa, pfun=pfun),
        all_models(wb, pfun=pfun),
        all_models(wab, pfun=pfun),
    )
    rep.check("E_HF(AB) - E_HF(A) - E_HF(B)", wab.e_ref - wa.e_ref - wb.e_ref, 1e-9)
    # Bar 1e-9 Ha: SCF density_conv 1e-10 per fragment (first-order in the
    # orbitals), N2 quadrupole x CH4 octupole at 189 Bohr ~ 1e-13 Ha.
    for k in ma:
        dev = mab[k] - ma[k] - mb[k]
        print(
            f"  {k:22s} A {ma[k]:.10f} B {mb[k]:.10f} AB {mab[k]:.10f} dev {dev:+.3e}"
        )
        if k.startswith("scalar"):
            if scalar_must_fail:
                rep.check(
                    f"size {k} (must FAIL size consistency)",
                    dev,
                    1e-6,
                    expect_fail=True,
                )
        else:
            rep.check(f"size {k}", dev, 1e-9)


# ---------------------------------------------------------------------------
# Mutations: each must fail at least one check
# ---------------------------------------------------------------------------


def pade_swapped(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MUTANT: numerator and denominator exchanged, (A - B) A^{-1} (A - B)."""
    keep = np.flatnonzero(np.abs(a).sum(axis=0) > 0)
    out = np.zeros_like(a)
    if keep.size:
        ak, bk = a[np.ix_(keep, keep)], b[np.ix_(keep, keep)]
        out[np.ix_(keep, keep)] = (ak - bk) @ np.linalg.solve(ak, ak - bk)
    return out


def pade_slots_swapped(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MUTANT: W2 and W3 in each other's slot, P(xW3, W2)."""
    return pade(b, a) if np.abs(b).sum() > 0 else np.zeros_like(a)


def pade_scalar_in_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MUTANT: energy-level Pade of each block's traces (Eq. 1 per spin block)."""
    ta, tb = tr(a), tr(b)
    if a.size == 0 or ta == 0.0:
        return np.zeros_like(a)
    return np.eye(a.shape[0]) * (ta * ta / (ta - tb)) / a.shape[0]


def wset_ss_os_swapped(ws: WSet) -> WSet:
    """MUTANT: the same-spin / opposite-spin labels exchanged."""
    w = {}
    for (n, s, ch), m in ws.w.items():
        w[(n, s, "os" if ch == "ss" else "ss")] = m
    return WSet(w, ws.e_nbs, ws.e_ref)


# ---------------------------------------------------------------------------
# Cost: wall time vs run_mp3 (RI, same aux), interleaved A B B A A B
# ---------------------------------------------------------------------------


def time_mp3(mol, bs, aux):
    t0 = time.perf_counter()
    r = ferric.run_mp3(mol, bs, aux, scf_energy_conv=1e-12, scf_density_conv=1e-10)
    return time.perf_counter() - t0, r.e_mp2 + r.e_mp3


def time_proto(mol, bs, aux, x_ss=0.45):
    t0 = time.perf_counter()
    ref = scf_reference(mol, bs, "rhf", **SCF_TIGHT)
    t1 = time.perf_counter()
    mp = SpinOrbitalMP(ref, Integrals(mol, bs, aux)).build()
    t2 = time.perf_counter()
    ws = mp.wset()
    t3 = time.perf_counter()
    e = e_ssd_opmp3(ws, x_ss)
    t4 = time.perf_counter()
    parts = {
        "scf": t1 - t0,
        "amplitudes": t2 - t1,
        "W": t3 - t2,
        "pade": t4 - t3,
        "vvvv": mp.timings["vvvv"],
    }
    return t4 - t0, e, parts, (sum(mp.nocc) // 2, sum(mp.nvir) // 2)


def anchor_cost(rep: Report, reps: int = 3) -> None:
    print(
        "== cost: wall s, run_mp3 (A) vs prototype SSD-OPMP3 (B), RI cc-pVDZ / cc-pVDZ-RI,"
        " order A B B A A B per rep"
    )
    bs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    rows = []
    for name, atoms in (("H2O", H2O), ("NH3", NH3), ("CH4", CH4), ("C2H4", C2H4)):
        mol = mol_from(atoms)
        ta, tb, parts = [], [], None
        for _ in range(reps):
            for who in "ABBAAB":
                if who == "A":
                    ta.append(time_mp3(mol, bs, aux)[0])
                else:
                    t, _, parts, dims = time_proto(mol, bs, aux)
                    tb.append(t)
        try:
            load = open("/proc/loadavg").read().split()[0]
        except OSError:
            load = "?"
        row = {
            "mol": name,
            "no_nv": dims,
            "run_mp3": ta,
            "proto": tb,
            "proto_parts_last": parts,
            "load1": load,
        }
        rows.append(row)
        print(
            f"  {name} (no, nv)={dims}: run_mp3 {np.mean(ta):.3f} +- {np.std(ta, ddof=1):.3f} s; "
            f"proto {np.mean(tb):.3f} +- {np.std(tb, ddof=1):.3f} s; load {load}"
        )
        print(f"    run_mp3 reps {[round(t, 3) for t in ta]}")
        print(f"    proto   reps {[round(t, 3) for t in tb]}")
        print(
            f"    proto parts (last rep) { ({k: round(v, 3) for k, v in parts.items()}) }"
        )
    rep.rows.append({"cost": rows})


# ---------------------------------------------------------------------------
# Paper data (Zenodo 23482348, SI_pade_paper/)
# ---------------------------------------------------------------------------

W_LABELS = [
    (n, s, sn, ch)
    for n in (2, 3)
    for s, sn in (("a", "alpha"), ("b", "beta"))
    for ch in ("ss", "os")
]


def final_out(path: Path) -> dict:
    out = {}
    for ln in path.read_text().splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def load_paper_wset(structure_dir: Path) -> WSet:
    stem = structure_dir.name.rsplit(".", 1)[0]
    w = {
        (n, s, ch): np.load(structure_dir / f"{stem}.W{n}{sn}_{ch}.npy")
        for n, s, sn, ch in W_LABELS
    }
    fo = final_out(structure_dir / f"{stem}.final.out")
    e_ref = next(v for k, v in fo.items() if k.startswith("Reference energy (HF on"))
    return WSet(w, fo["Non-Brillouin singles correction"], e_ref)


def paper_dir(args, ref: str, name: str, basis: str = "aug-cc-pCVTZ") -> Path:
    return Path(args.zenodo) / "FINAL_RESULTS" / basis / ref / f"{name}.xyz"


GEOM_DIR = None  # set from --geom (ACCDB W4-17 .xyz files, the paper's geometries)


def w4_mol(name: str):
    return ferric.Molecule.from_xyz(
        str(Path(GEOM_DIR) / f"{name}.xyz"),
        *_charge_mult(Path(GEOM_DIR) / f"{name}.xyz"),
    )


def _charge_mult(path: Path):
    q, m = path.read_text().splitlines()[1].split()[:2]
    return int(q), int(m)


KCAL_PER_HA = 627.5094740630558  # CODATA 2018 via scipy, as the paper's config.py

# Paper text (Sec. "W4-17 nonMR", Fig. 2/3): SSD-OPMP3, kOOMP2 x_ss = 0.45:
# RMSE 1.30, MSE -0.45, MAE 0.88 kcal/mol (two decimals -> bar 0.005).
PAPER_HEADLINE = {"KOOMP2": {"RMSE": 1.30, "MSE": -0.45, "MAE": 0.88}}

# results/kOOMP2/TAE_W4-17nonMR.report.txt, fitted rows: (x, x_ss, x_os) printed
# to 4 decimals and their RMSE. Evaluated here at the printed parameters; bar
# 2e-3 kcal/mol covers the 5e-5 parameter rounding (|dRMSE/dx| < 10 kcal/mol).
PAPER_FITTED_KOOMP2 = [
    ("OP-MP3(x=0.7890)", lambda w: e_opmp3(w, 0.7890), 1.8225),
    ("SOP-MP3(0.3809, 1.0253)", lambda w: e_sopmp3(w, 0.3809, 1.0253), 1.2550),
    ("SSD-OPMP3(x_ss=0.4279)", lambda w: e_ssd_opmp3(w, 0.4279), 1.2634),
    ("MP2.X(0.8365)", lambda w: e_mp2x(w, 0.8365), 2.4468),
]

# Paper analysis_code README: x_ss of SSD-OPMP3 per reference.
X_SS.update(
    {"PBE": 0.70, "SPW92": 0.75, "R2SCAN": 0.60, "wB97X-V": 0.50, "wB97M-V": 0.50}
)


def _read_rxn(path: Path, dataset: str) -> dict:
    """The paper's reaction table (a flat YAML of name -> Dataset, counts,
    energy, structs), parsed by hand: the venv has no PyYAML."""
    out, cur, key = {}, None, None
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        if not ln.startswith(" "):
            cur = out.setdefault(ln.rstrip(":").strip(), {"counts": [], "structs": []})
            continue
        body = ln.strip()
        if body.startswith("- "):
            val = body[2:].strip()
            cur[key].append(float(val) if key == "counts" else val)
        else:
            k, _, v = body.partition(":")
            key = k.strip()
            if v.strip():
                cur[key] = v.strip()
    return {k: v for k, v in out.items() if v.get("Dataset") == dataset}


def _read_csv(path: Path):
    import csv

    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return rows


def repro_model_fns(x_ss: float, pfun=pade, mut_nodamp: bool = False):
    """Column label in the paper's csv -> our correlation energy."""
    xs = 1.0 if mut_nodamp else x_ss
    xs_label = f"{x_ss:g}"
    return {
        "MP2": lambda w: w.e2(),
        "MP3": lambda w: w.e2() + w.e3(),
        "OP-MP3": lambda w: e_opmp3(w, 1.0, pfun),
        "SOP-MP3": lambda w: e_sopmp3(w, 1.0, 1.0, pfun),
        f"SSD-OPMP3(x_ss={xs_label})": lambda w: e_ssd_opmp3(w, xs, pfun),
        "P-MP3": e_scalar_pade,
    }


def anchor_repro(rep: Report, args, pfun=pade, mut_nodamp: bool = False) -> None:
    """The paper's W matrices, run through this prototype's resummation, give
    the paper's W4-17 nonMR TAE errors (per reaction) and its headline stats.

    The per-reaction csv is printed to 1e-6 kcal/mol, so the per-reaction bar
    is 1e-5 kcal/mol (csv rounding plus E_HF/E_NBS read from final.out at
    1e-15 Ha). Statistics: 1e-4 against report.txt (4 decimals) and 0.005
    against the paper's text (2 decimals).
    """
    print(
        "== repro: paper W matrices -> W4-17 nonMR TAE errors vs the paper's results/"
    )
    root = Path(args.zenodo)
    rxn = _read_rxn(root / "results" / "RXN_ALL_plus_RE.yaml", "TAE_W4-17nonMR")
    for xc in ("KOOMP2", "PBE0", "PBE"):
        csvp = (
            root
            / "results"
            / {"KOOMP2": "kOOMP2"}.get(xc, xc)
            / "TAE_W4-17nonMR.default.csv"
        )
        if not csvp.exists():
            print(f"  {xc}: no {csvp.name}, skipped")
            continue
        rows = {r["reaction"]: r for r in _read_csv(csvp)}
        fns = repro_model_fns(X_SS[xc], pfun, mut_nodamp)
        cache = {}

        def etot(struct, model):
            if struct not in cache:
                cache[struct] = load_paper_wset(paper_dir(args, xc, struct))
            w = cache[struct]
            return w.e_ref + w.e_nbs + fns[model](w)

        errs = {m: [] for m in fns}
        worst = {m: 0.0 for m in fns}
        named = {}
        for rname, r in rxn.items():
            ref = float(rows[rname]["reference (kcal/mol)"])
            for m in fns:
                pred = KCAL_PER_HA * sum(
                    c * etot(st, m) for st, c in zip(r["structs"], r["counts"])
                )
                err = pred - ref
                errs[m].append(err)
                theirs = float(rows[rname][f"{m} error (kcal/mol)"])
                worst[m] = max(worst[m], abs(err - theirs))
                if r["structs"][0] in (
                    "W4-17_h2o",
                    "W4-17_nh3",
                    "W4-17_ch4",
                    "W4-17_hf",
                    "W4-17_co",
                    "W4-17_benzene",
                ) and m.startswith("SSD"):
                    named[r["structs"][0]] = (ref, pred, err, theirs)
        print(
            f"  {xc} ({len(rxn)} TAEs, x_ss = {X_SS[xc]}{', MUTANT x_ss=1' if mut_nodamp else ''}):"
        )
        for mol, (ref, pred, err, theirs) in sorted(named.items()):
            print(
                f"    {mol:14s} TAE ref {ref:10.4f}  ours {pred:10.4f}  err {err:+.6f}  paper err {theirs:+.6f}"
            )
        for m in fns:
            e = np.array(errs[m])
            print(
                f"    {m:24s} MAE {np.mean(np.abs(e)):.4f} RMSE {np.sqrt(np.mean(e * e)):.4f} "
                f"MSE {np.mean(e):+.4f}  max|ours - paper| per TAE {worst[m]:.2e}"
            )
            rep.check(f"{xc} {m}: max per-TAE |ours - paper|", worst[m], 1e-5)
        report = (csvp.parent / "TAE_W4-17nonMR.report.txt").read_text().splitlines()
        for m in fns:
            row = next(
                (ln.split() for ln in report if ln.strip().startswith(m + " ")), None
            )
            if row is not None:
                e = np.array(errs[m])
                rep.check(
                    f"{xc} {m} RMSE - report.txt {row[2]}",
                    np.sqrt(np.mean(e * e)) - float(row[2]),
                    1e-4,
                )
        if xc == "KOOMP2" and pfun is pade and not mut_nodamp:
            for label, fn, rmse in PAPER_FITTED_KOOMP2:
                e = np.array(
                    [
                        KCAL_PER_HA
                        * sum(
                            c * (cache[st].e_ref + cache[st].e_nbs + fn(cache[st]))
                            for st, c in zip(r["structs"], r["counts"])
                        )
                        - float(rows[rn]["reference (kcal/mol)"])
                        for rn, r in rxn.items()
                    ]
                )
                val = float(np.sqrt(np.mean(e * e)))
                print(f"    fitted {label}: RMSE {val:.4f} (report {rmse})")
                rep.check(f"{xc} fitted {label} RMSE - report {rmse}", val - rmse, 2e-3)
        ssd = np.array(errs[[m for m in fns if m.startswith("SSD")][0]])
        for stat, val in (
            ("RMSE", np.sqrt(np.mean(ssd * ssd))),
            ("MSE", np.mean(ssd)),
            ("MAE", np.mean(np.abs(ssd))),
        ):
            if xc in PAPER_HEADLINE:
                rep.check(
                    f"{xc} SSD-OPMP3 {stat} - paper text {PAPER_HEADLINE[xc][stat]}",
                    val - PAPER_HEADLINE[xc][stat],
                    0.005,
                )


_WREPRO_CACHE: dict = {}


def anchor_wrepro(rep: Report, args, mut_ssos: bool = False) -> None:
    """ferric RHF/aug-cc-pCVTZ W == the paper's HF-reference W (invariants).

    Occupied rotations make elementwise comparison meaningless (degenerate
    orbitals, two SCF codes), so compare rotation invariants: the sorted
    eigenvalues of each of the eight W matrices and the model energies.
    Bar 1e-6 Ha: the paper's own W-vs-Q-Chem test bar is 1e-5 (Tr W2), this
    is 10x tighter; the measured E_HF difference is printed beside it.
    """
    print(
        "== wrepro: ferric HF W vs paper HF W, aug-cc-pCVTZ (H: aug-cc-pVTZ), all-electron, exact"
    )
    bs = ferric.BasisSet.from_bse_json(args.basis_json)
    for name in args.wrepro_mols.split(","):
        pw = load_paper_wset(paper_dir(args, "HF", name))
        mol = w4_mol(name)
        t0 = time.perf_counter()
        if name not in _WREPRO_CACHE:
            _WREPRO_CACHE[name] = compute_wset(mol, bs, "rhf", **SCF_TIGHT)[1]
        ws = _WREPRO_CACHE[name]
        if mut_ssos:
            ws = wset_ss_os_swapped(ws)
        dt = time.perf_counter() - t0
        print(
            f"  {name}: nbf {bs.n_functions(mol)}, {dt:.1f} s; E_HF ferric {ws.e_ref:.10f} "
            f"paper {pw.e_ref:.10f}; E_NBS paper {pw.e_nbs:.2e}"
        )
        rep.check(f"{name} E_HF ferric - paper", ws.e_ref - pw.e_ref, 1e-6)
        worst = 0.0
        for n, s, _, ch in W_LABELS:
            ev_f = np.sort(np.linalg.eigvalsh(ws.get(n, s, ch)))
            ev_p = np.sort(np.linalg.eigvalsh(pw.get(n, s, ch)))
            worst = max(worst, float(np.max(np.abs(ev_f - ev_p))) if ev_f.size else 0.0)
        rep.check(f"{name} max |eig W ferric - paper| over 8 W", worst, 1e-6)
        mf, mpp = all_models(ws, x_ss=0.0), all_models(pw, x_ss=0.0)
        for k in mf:
            rep.check(f"{name} {k} ferric - paper", mf[k] - mpp[k], 1e-6)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def anchor_mut(rep: Report, args) -> None:
    """Each mutant must make at least one check fail (on its value, not a crash)."""
    print("== mut: every mutation must fail a check")
    mutants = [
        (
            "Pade numerator/denominator swapped: trivial limit",
            lambda r: anchor_trivial(r, pade_swapped),
        ),
        (
            "Pade numerator/denominator swapped: reproduction",
            lambda r: anchor_repro(r, args, pfun=pade_swapped),
        ),
        (
            "W2/W3 slots swapped: trivial limit",
            lambda r: anchor_trivial(r, pade_slots_swapped),
        ),
        (
            "same-spin damping dropped (x_ss = 1): reproduction",
            lambda r: anchor_repro(r, args, mut_nodamp=True),
        ),
        (
            "SS/OS channels swapped: W reproduction",
            lambda r: anchor_wrepro(r, args, mut_ssos=True),
        ),
        (
            "scalar Pade (Eq 1) in place of the orbital Pade: size consistency",
            lambda r: anchor_size(r, pfun=pade_scalar_in_block),
        ),
    ]
    for label, run in mutants:
        sub = Report()
        print(f"-- mutant: {label}")
        try:
            run(sub)
        except Exception as exc:  # a crash is not a detection
            print(
                f"  mutant CRASHED ({type(exc).__name__}: {exc}); not counted as detected"
            )
            sub.rows = []
        nfail = sum(1 for x in sub.rows if "pass" in x and not x["pass"])
        rep.check(
            f"mutant detected ({label}): failing rows = {nfail}",
            0.0 if nfail else 1.0,
            0.5,
        )


ANCHORS = {
    "trivial": lambda r, a: anchor_trivial(r),
    "cross": lambda r, a: anchor_cross(r),
    "eq18": lambda r, a: anchor_eq18(r),
    "size": lambda r, a: anchor_size(r),
    "repro": lambda r, a: anchor_repro(r, a),
    "wrepro": lambda r, a: anchor_wrepro(r, a),
    "mut": anchor_mut,
    "cost": lambda r, a: anchor_cost(r),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--anchor", default="all", help="one of: all, " + ", ".join(ANCHORS))
    p.add_argument("--zenodo", default=None, help="extracted SI_pade_paper/ directory")
    p.add_argument("--json", default=None, help="write the check table here")
    p.add_argument("--geom", default=None, help="directory of W4-17_<name>.xyz (ACCDB)")
    p.add_argument(
        "--basis-json", default=None, help="aug-cc-pCVTZ BSE JSON (H, He: aug-cc-pVTZ)"
    )
    p.add_argument("--wrepro-mols", default="W4-17_h2,W4-17_hf,W4-17_h2o,W4-17_nh3")
    args = p.parse_args(argv)
    global GEOM_DIR
    GEOM_DIR = args.geom
    print(f"ferric {ferric.__version__} build {ferric.__build__}")
    rep = Report()
    names = list(ANCHORS) if args.anchor == "all" else [args.anchor]
    for n in names:
        if n == "cost" and args.anchor == "all":
            continue
        ANCHORS[n](rep, args)
    nfail = sum(1 for r in rep.rows if "pass" in r and not r["pass"])
    print(f"RESULT: {len(rep.rows)} rows, {nfail} failed")
    if args.json:
        Path(args.json).write_text(json.dumps(rep.rows, indent=1, default=float))
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
