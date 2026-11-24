#!/usr/bin/env python
"""Spin-free X2C-1e prototype in numpy on ferric's integral blocks.

Correctness reference for a Rust spin-free X2C-1e core Hamiltonian (T048).
S, T, V and the pVp block W come from ``ferric.AoIntegrals`` in the
uncontracted basis; the X2C core Hamiltonian is built there, contracted back
to the orbital basis, and handed to ``ferric.run_rhf`` as a constant
``fock_hook``. Equations: x2c_derivation.md. Nothing about speed may be
quoted from this prototype.

Anchors (``--only`` picks one):

* ``hook0``: a hook returning (0, 0) gives a run bitwise identical to the
  plain ``run_rhf`` (energy, iterations, density).
* ``contract``: the contraction matrix maps the uncontracted S, T, V onto the
  contracted ones to rounding (checks the AO order of the two bases).
* ``cscan``: ||h_X2C(c) - (T + V)|| falls as c^-2 over c x {1, 10, 100, 1000}.
* ``pvp``: eigenvalues of S^-1/2 W S^-1/2 vs PySCF ``int1e_pnucp`` in the
  uncontracted basis (ordering-free).
* ``hcore``: eigenvalues of the contracted X2C core Hamiltonian (with S) vs
  PySCF ``sfx2c1e`` ``get_hcore``.
* ``scf``: E_X2C - E_NR vs PySCF, both codes on the same basis data.

PySCF settings matched: ``scf.RHF(mol).sfx2c1e()`` with ``approx = '1e'``,
``xuncontract = True`` (every atom, unique exponents per angular momentum),
``basis = None``, c = ``lib.param.LIGHT_SPEED``; the basis is ferric's bundled
cc-pVDZ JSON converted to PySCF's format, spherical d (``cart = False``).

``--mutate flip-w`` builds the X2C Hamiltonian with -W.

Usage (one heavy op at a time on the shared box):
  box-lease exec -- env OPENBLAS_NUM_THREADS=1 PYTHONPATH=<shim> \\
    /home/matt/qc/ferric/.venv/bin/python scripts/prototypes/x2c_proto.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.linalg

import ferric

REPO = Path(__file__).resolve().parents[2]
BUNDLED = REPO / "crates/ferric-core/src/basis/bundled"
ANGSTROM = 0.529177210903  # Bohr in Angstrom (PySCF's BOHR)
SYMBOLS = {1: "H", 10: "Ne", 18: "Ar", 35: "Br", 36: "Kr"}
SYSTEMS = {
    "Ne": "Ne 0 0 0",
    "Ar": "Ar 0 0 0",
    "Kr": "Kr 0 0 0",
    "HBr": "H 0 0 0\nBr 0 0 1.414",
}
# libint2 engine precision per primitive integral, times 10 and the three
# terms of W: the per-element rounding bar of crates/ferric-integrals/tests/pvp.rs.
ROUND = 3e-13


def light_speed() -> float:
    from pyscf import lib

    return float(lib.param.LIGHT_SPEED)


# ---------------------------------------------------------------- basis data


def load_shells(name: str, zs: list[int]) -> dict[int, list[dict]]:
    """Shells of the bundled JSON as ferric parses them: one entry per
    coefficient column (parse_bse_json), in file order."""
    data = json.loads((BUNDLED / f"{name}.json").read_text())["elements"]
    out: dict[int, list[dict]] = {}
    for z in zs:
        shells = []
        for sh in data[str(z)]["electron_shells"]:
            assert len(sh["angular_momentum"]) == 1, "sp shells not handled"
            ang = sh["angular_momentum"][0]
            exps = [float(e) for e in sh["exponents"]]
            for col in sh["coefficients"]:
                shells.append(
                    {
                        "l": ang,
                        "type": sh["function_type"],
                        "exps": exps,
                        "coefs": [float(c) for c in col],
                    }
                )
        out[z] = shells
    return out


def uncontracted(shells: dict[int, list[dict]]) -> dict[int, list[dict]]:
    """One single-primitive shell per unique exponent and l, l in order of
    first appearance, exponents in file order."""
    out = {}
    for z, shs in shells.items():
        prim: dict[int, list[float]] = {}
        types: dict[int, str] = {}
        for sh in shs:
            lst = prim.setdefault(sh["l"], [])
            types[sh["l"]] = sh["type"]
            for e in sh["exps"]:
                if e not in lst:
                    lst.append(e)
        out[z] = [
            {"l": ang, "type": types[ang], "exps": [e], "coefs": [1.0]}
            for ang, es in prim.items()
            for e in es
        ]
    return out


def write_json(shells: dict[int, list[dict]], path: Path) -> None:
    elements = {
        str(z): {
            "electron_shells": [
                {
                    "function_type": sh["type"],
                    "region": "",
                    "angular_momentum": [sh["l"]],
                    "exponents": [repr(e) for e in sh["exps"]],
                    "coefficients": [[repr(c) for c in sh["coefs"]]],
                }
                for sh in shs
            ]
        }
        for z, shs in shells.items()
    }
    path.write_text(json.dumps({"name": path.stem, "elements": elements}))


def ncomp(sh: dict) -> int:
    ang = sh["l"]
    return (
        2 * ang + 1
        if (sh["type"] == "gto_spherical" and ang >= 2)
        else (ang + 1) * (ang + 2) // 2
    )


def contraction_matrix(atoms: list[int], con: dict, unc: dict) -> np.ndarray:
    """C (n_unc x n_con): contracted AO = sum_k C[k, mu] uncontracted AO_k.
    Coefficients multiply unit-normalized primitives and each contraction has
    unit self-overlap; same-centre primitives of one l overlap as
    (2 sqrt(a b) / (a + b))^(l + 3/2)."""
    cols = []
    n_unc = sum(ncomp(sh) for z in atoms for sh in unc[z])
    row0 = 0
    for z in atoms:
        index = {}
        r = row0
        for sh in unc[z]:
            index[(sh["l"], sh["exps"][0])] = r
            r += ncomp(sh)
        for sh in con[z]:
            ang, a, c = sh["l"], np.array(sh["exps"]), np.array(sh["coefs"])
            sab = (2 * np.sqrt(np.outer(a, a)) / np.add.outer(a, a)) ** (ang + 1.5)
            c = c / np.sqrt(c @ sab @ c)
            for m in range(ncomp(sh)):
                col = np.zeros(n_unc)
                for e, ck in zip(sh["exps"], c):
                    col[index[(ang, e)] + m] += ck
                cols.append(col)
        row0 = r
    return np.array(cols).T


# ---------------------------------------------------------------- X2C


def x2c_hcore(s, t, v, w, c):
    """Spin-free X2C-1e core Hamiltonian in the basis of s, t, v, w.

    Modified Dirac equation  [[V, T], [T, W/(4c^2) - T]] C = [[S, 0], [0, T/(2c^2)]] C E;
    the n electronic solutions give X = C_S C_L^-1; renormalization
    R = S^-1/2 (S^-1/2 S~ S^-1/2)^-1/2 S^1/2 with S~ = S + X^T T X/(2c^2);
    h = R^T (V + T X + X^T T - X^T T X + X^T W X/(4c^2)) R."""
    n = s.shape[0]
    h4 = np.block([[v, t], [t, w / (4 * c * c) - t]])
    m4 = np.block([[s, np.zeros_like(s)], [np.zeros_like(s), t / (2 * c * c)]])
    _, vec = scipy.linalg.eigh(h4, m4)
    cl, cs = vec[:n, n:], vec[n:, n:]
    x = np.linalg.solve(cl.T, cs.T).T
    st = s + x.T @ t @ x / (2 * c * c)
    lag = v + t @ x + x.T @ t - x.T @ t @ x + x.T @ w @ x / (4 * c * c)
    es, us = np.linalg.eigh(s)
    s_half = (us * np.sqrt(es)) @ us.T
    s_mhalf = (us / np.sqrt(es)) @ us.T
    em, um = np.linalg.eigh(s_mhalf @ st @ s_mhalf)
    r = s_mhalf @ ((um / np.sqrt(em)) @ um.T) @ s_half
    h = r.T @ lag @ r
    return 0.5 * (h + h.T)


class System:
    """One molecule: contracted and uncontracted ferric integrals."""

    def __init__(self, label: str, basis: str = "cc-pvdz", tmp: Path | None = None):
        self.label = label
        self.xyz = SYSTEMS[label]
        lines = self.xyz.splitlines()
        self.atoms = [
            next(z for z, s in SYMBOLS.items() if s == ln.split()[0]) for ln in lines
        ]
        zs = sorted(set(self.atoms))
        self.con_shells = load_shells(basis, zs)
        self.unc_shells = uncontracted(self.con_shells)
        tmp = tmp or Path(tempfile.mkdtemp())
        unc_path = tmp / f"{basis}-unc.json"
        write_json(self.unc_shells, unc_path)
        self.mol = ferric.Molecule.from_xyz_string(f"{len(lines)}\n\n{self.xyz}\n")
        self.basis = ferric.BasisSet.from_bse_json(str(BUNDLED / f"{basis}.json"))
        self.unc_basis = ferric.BasisSet.from_bse_json(str(unc_path))
        self.ints = ferric.AoIntegrals(self.mol, self.basis)
        self.unc = ferric.AoIntegrals(self.mol, self.unc_basis)
        self.cmat = contraction_matrix(self.atoms, self.con_shells, self.unc_shells)
        self.s_u = self.unc.overlap()
        self.t_u = self.unc.kinetic()
        self.v_u = self.unc.nuclear()
        self.w_u = self.unc.pvp()

    def delta_h(self, c: float, flip_w: bool = False) -> np.ndarray:
        """C^T (h_X2C - T - V) C in the contracted basis."""
        w = -self.w_u if flip_w else self.w_u
        h = x2c_hcore(self.s_u, self.t_u, self.v_u, w, c)
        d = self.cmat.T @ (h - self.t_u - self.v_u) @ self.cmat
        return 0.5 * (d + d.T)

    def pyscf_mol(self, uncontract: bool = False):
        from pyscf import gto

        shells = self.unc_shells if uncontract else self.con_shells
        basis = {}
        for z, shs in shells.items():
            entries = []
            for sh in shs:
                entries.append(
                    [sh["l"]] + [[e, c] for e, c in zip(sh["exps"], sh["coefs"])]
                )
            basis[SYMBOLS[z]] = entries
        return gto.M(
            atom=self.xyz.replace("\n", "; "),
            basis=basis,
            unit="Angstrom",
            cart=False,
            verbose=0,
        )


def gen_eig_bar(a, s, n):
    """First-order rounding bound on the generalized eigenvalues of (a, s):
    |d lambda| <= (||dA|| + |lambda| ||dS||) / lambda_min(S), with each
    element carrying ROUND relative rounding (norms bounded by n max|.|)."""
    lam_min = np.linalg.eigvalsh(s).min()
    lam = scipy.linalg.eigh(a, s, eigvals_only=True)
    return lam, n * ROUND * (np.abs(a).max() + np.abs(lam) * np.abs(s).max()) / lam_min


# ---------------------------------------------------------------- anchors


def anchor_hook0(sys_: System, **kw) -> bool:
    plain = ferric.run_rhf(sys_.mol, sys_.basis, **kw)
    n = sys_.ints.nao

    def hook(d):
        return 0.0, np.zeros((n, n))

    hooked = ferric.run_rhf(sys_.mol, sys_.basis, fock_hook=hook, **kw)
    same = (
        plain.energy == hooked.energy
        and plain.iterations == hooked.iterations
        and np.array_equal(plain.density(), hooked.density())
    )
    print(
        f"hook0 {sys_.label}: E {plain.energy!r} vs {hooked.energy!r}, iters {plain.iterations}/{hooked.iterations}, bitwise {same}"
    )
    return same


def anchor_contract(sys_: System) -> bool:
    ok = True
    for name, unc, con in [
        ("S", sys_.s_u, sys_.ints.overlap()),
        ("T", sys_.t_u, sys_.ints.kinetic()),
        ("V", sys_.v_u, sys_.ints.nuclear()),
    ]:
        dev = np.abs(sys_.cmat.T @ unc @ sys_.cmat - con).max()
        bar = unc.shape[0] * ROUND * np.abs(con).max()
        print(f"contract {sys_.label} {name}: max dev {dev:.2e} (bar {bar:.2e})")
        ok &= dev <= bar
    return ok


def anchor_cscan(sys_: System, flip_w: bool = False) -> bool:
    c0 = light_speed()
    norms = []
    for f in (1, 10, 100, 1000):
        norms.append(np.linalg.norm(sys_.delta_h(c0 * f, flip_w)))
    ratios = [norms[i] / norms[i + 1] for i in range(3)]
    print(
        f"cscan {sys_.label}{' (flip-w)' if flip_w else ''}: ||dH|| {['%.4e' % x for x in norms]}, ratios {['%.4f' % r for r in ratios]}"
    )
    # c^-2 with an O(c^-2) relative correction: each ratio is 100 (1 + O(c^-2)),
    # and |ratio/100 - 1| must shrink by about 100 per decade.
    dev = [abs(r / 100 - 1) for r in ratios]
    return dev[1] < 0.05 and dev[2] < dev[1] / 10 + 1e-6


def anchor_pvp(sys_: System) -> bool:
    pm = sys_.pyscf_mol(uncontract=True)
    s_p, w_p = pm.intor_symmetric("int1e_ovlp"), pm.intor_symmetric("int1e_pnucp")
    lam_f, bar_f = gen_eig_bar(sys_.w_u, sys_.s_u, sys_.s_u.shape[0])
    lam_p, bar_p = gen_eig_bar(w_p, s_p, s_p.shape[0])
    if lam_f.shape != lam_p.shape:
        print(f"pvp {sys_.label}: nao differs ({lam_f.size} vs {lam_p.size})")
        return False
    dev = np.abs(lam_f - lam_p)
    ratio = (dev / (bar_f + bar_p)).max()
    print(
        f"pvp {sys_.label}: nao {lam_f.size}, eig range [{lam_f.min():.6e}, {lam_f.max():.6e}], max |d eig| {dev.max():.2e}, worst dev/bar {ratio:.2e}"
    )
    return ratio <= 1.0


def pyscf_x2c(sys_: System, conv_tol: float):
    from pyscf import scf

    pm = sys_.pyscf_mol()
    nr = scf.RHF(pm)
    nr.conv_tol = conv_tol
    e_nr = nr.kernel()
    rel = scf.RHF(pm).sfx2c1e()
    rel.conv_tol = conv_tol
    assert (
        rel.with_x2c.approx.upper() == "1E"
        and rel.with_x2c.xuncontract
        and rel.with_x2c.basis is None
    )
    e_rel = rel.kernel()
    return pm, rel, e_nr, e_rel, nr.converged and rel.converged


def anchor_hcore(sys_: System, flip_w: bool = False) -> bool:
    pm = sys_.pyscf_mol()
    from pyscf import scf

    h_p = scf.RHF(pm).sfx2c1e().get_hcore()
    s_p = pm.intor_symmetric("int1e_ovlp")
    h_f = (
        sys_.ints.kinetic() + sys_.ints.nuclear() + sys_.delta_h(light_speed(), flip_w)
    )
    s_f = sys_.ints.overlap()
    lam_f, bar_f = gen_eig_bar(h_f, s_f, s_f.shape[0])
    lam_p, bar_p = gen_eig_bar(h_p, s_p, s_p.shape[0])
    dev = np.abs(lam_f - lam_p)
    ratio = (dev / (bar_f + bar_p)).max()
    print(
        f"hcore {sys_.label}{' (flip-w)' if flip_w else ''}: lowest eig {lam_f[0]:.10f} (PySCF {lam_p[0]:.10f}), max |d eig| {dev.max():.2e}, worst dev/bar {ratio:.2e}"
    )
    return ratio <= 1.0


def anchor_scf(sys_: System, conv: float, flip_w: bool = False) -> bool:
    dh = sys_.delta_h(light_speed(), flip_w)

    def hook(d):
        # Constant one-electron term: the driver adds V to F and E to
        # 1/2 Tr[D(h + F)] computed without V, so E_hook = Tr[D dH] makes the
        # total the energy of the core Hamiltonian h + dH.
        return float(np.sum(d * dh)), dh

    kw = dict(energy_conv=conv, density_conv=1e-8, max_iter=200)
    nr = ferric.run_rhf(sys_.mol, sys_.basis, **kw)
    rel = ferric.run_rhf(sys_.mol, sys_.basis, fock_hook=hook, **kw)
    _, _, e_nr_p, e_rel_p, conv_p = pyscf_x2c(sys_, conv)
    d_f = rel.energy - nr.energy
    d_p = e_rel_p - e_nr_p
    bar = 4 * conv
    print(
        f"scf {sys_.label}{' (flip-w)' if flip_w else ''}: ferric E_NR {nr.energy:.10f} E_X2C {rel.energy:.10f} dE {d_f:.10f}; "
        f"PySCF E_NR {e_nr_p:.10f} E_X2C {e_rel_p:.10f} dE {d_p:.10f}; |ddE| {abs(d_f - d_p):.2e} (bar {bar:.1e}); "
        f"converged {nr.converged and rel.converged and conv_p}"
    )
    return abs(d_f - d_p) <= bar and nr.converged and rel.converged and conv_p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", choices=["hook0", "contract", "cscan", "pvp", "hcore", "scf"]
    )
    ap.add_argument("--systems", default="Ne,Ar,Kr,HBr")
    ap.add_argument("--mutate", choices=["flip-w"])
    ap.add_argument("--conv", type=float, default=1e-10)
    args = ap.parse_args()
    print(f"ferric {ferric.build_info()}")
    flip = args.mutate == "flip-w"
    ok = True
    for label in args.systems.split(","):
        sys_ = System(label)
        steps = (
            [args.only]
            if args.only
            else ["hook0", "contract", "cscan", "pvp", "hcore", "scf"]
        )
        for step in steps:
            if step == "hook0":
                res = anchor_hook0(sys_, energy_conv=args.conv, density_conv=1e-8)
            elif step == "contract":
                res = anchor_contract(sys_)
            elif step == "cscan":
                res = anchor_cscan(sys_, flip)
            elif step == "pvp":
                res = anchor_pvp(sys_)
            elif step == "hcore":
                res = anchor_hcore(sys_, flip)
            else:
                res = anchor_scf(sys_, args.conv, flip)
            print(f"  -> {step} {label}: {'PASS' if res else 'FAIL'}")
            ok &= res
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
