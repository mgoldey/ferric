"""Generate PySCF periodic (Gamma-point) references for the ferric-pbc Stage-0 tests.

Writes testdata/reference/pbc_<system>.json, one file per test cell, with:
  * lattice (rows = lattice vectors, Bohr), atoms (symbol, Bohr coords), basis name
  * energy_nuc   -- pyscf.pbc Cell.energy_nuc() (Ewald nuclear repulsion,
                    neutralising-background convention for charged cells)
  * madelung     -- pyscf.pbc.tools.madelung(cell, zeros((1,3))) = -2 * Ewald
                    energy of ONE unit point charge per cell (Gamma, omega=0)
  * gvectors     -- a few NONZERO G vectors (reciprocal-lattice AND one
                    arbitrary off-lattice vector), Bohr^-1
  * pair_ft_re / pair_ft_im -- [g][m][n] of
        P_mn(G) = sum_T int exp(-i G.r) phi_m(r) phi_n(r - T) dr
    from pyscf.pbc.df.ft_ao.ft_aopair at kpti = kptj = 0 (the docstring of
    ft_aopair states exactly this sign convention; see ft_ao.py:52-53).
  * ovlp_latt    -- pyscf pbc_intor('int1e_ovlp') lattice-summed overlap, a
                    cross-check only (ferric's own anchor uses libint).

AO basis identity (load-bearing): the basis is read from ferric's OWN bundled
BSE JSON and handed to PySCF as ONE PySCF shell per ferric shell -- each
coefficient column of a general contraction becomes its own shell, and an SP
block pairs column k with angular_momentum[k], exactly as
ferric_core::basis::parse_bse_json does. PySCF is spherical (cart=False):
s/p shells are identical to ferric's Cartesian s/p (PySCF orders pure p as
px,py,pz, the unit-normalised Cartesian set), and pure d+ use libcint's real
solid harmonics in m=-l..l order, which is what ferric_cart2sph is built from
(md3c1e.rs module doc). Shell order is asserted against the ferric order
below, so any PySCF reordering fails loudly here rather than silently in Rust.

Usage: OPENBLAS_NUM_THREADS=1 python3 scripts/gen_pyscf_pbc_refs.py
"""

import json
import os

import numpy as np
from pyscf.data.elements import ELEMENTS
from pyscf.pbc import gto as pgto
from pyscf.pbc import tools as ptools
from pyscf.pbc.df import ft_ao

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SYSTEMS = {
    # H2 / STO-3G (s only) in a triclinic cell.
    "h2_sto3g_triclinic": dict(
        basis="sto-3g",
        lattice=[[4.0, 0.0, 0.0], [0.8, 4.2, 0.0], [0.5, 0.6, 4.5]],
        atoms=[("H", [0.1, 0.2, 0.3]), ("H", [0.4, 0.5, 1.6])],
    ),
    # H2O / cc-pVDZ (s, p, pure d) in a triclinic cell.
    "h2o_ccpvdz_triclinic": dict(
        basis="cc-pvdz",
        lattice=[[6.5, 0.0, 0.0], [1.0, 6.8, 0.0], [-0.7, 0.9, 7.2]],
        atoms=[
            ("O", [0.3, 0.4, 0.5]),
            ("H", [0.3, 1.8304, 1.6075]),
            ("H", [0.3, -1.0304, 1.6075]),
        ],
    ),
    # A charged-cell-free ionic-ish case for Ewald only (s-only basis): LiH in a
    # skewed cell exercises unequal Z in the structure factor.
    "lih_sto3g_monoclinic": dict(
        basis="sto-3g",
        lattice=[[5.0, 0.0, 0.0], [0.0, 5.5, 0.0], [1.2, 0.0, 6.0]],
        atoms=[("Li", [0.0, 0.0, 0.0]), ("H", [0.2, 0.1, 3.0])],
    ),
}


def ferric_shells(basis):
    """Per-element list of (l, exps, coefs) in ferric's shell order."""
    path = os.path.join(ROOT, "crates/ferric-core/src/basis/bundled", f"{basis}.json")
    with open(path) as fh:
        d = json.load(fh)
    out = {}
    for z, elem in d["elements"].items():
        shells = []
        for sh in elem["electron_shells"]:
            exps = [float(x) for x in sh["exponents"]]
            cols = [[float(x) for x in col] for col in sh["coefficients"]]
            if len(sh["angular_momentum"]) == 1:
                l = sh["angular_momentum"][0]
                for col in cols:
                    shells.append((l, exps, col))
            else:
                for k, l in enumerate(sh["angular_momentum"]):
                    shells.append((l, exps, cols[k]))
        out[ELEMENTS[int(z)]] = shells
    return out


def build_cell(spec, precision):
    fsh = ferric_shells(spec["basis"])
    syms = sorted({s for s, _ in spec["atoms"]})
    # Zero-coefficient primitives (general-contraction columns) are dropped:
    # numerically identical, and PySCF's cutoff estimators divide by them.
    basis = {
        s: [[l] + [[e, c] for e, c in zip(exps, col) if c != 0.0] for (l, exps, col) in fsh[s]]
        for s in syms
    }
    cell = pgto.Cell()
    cell.a = np.array(spec["lattice"])
    cell.atom = [(s, r) for s, r in spec["atoms"]]
    cell.unit = "B"
    cell.basis = basis
    cell.cart = False
    cell.precision = precision
    cell.verbose = 0
    cell.build()
    # Assert PySCF kept ferric's shell order (atom-major, basis order within atom).
    expect = [(l, exps) for s, _ in spec["atoms"] for (l, exps, _c) in fsh[s]]
    assert cell.nbas == len(expect), (cell.nbas, len(expect))
    for ib, (l, exps) in enumerate(expect):
        assert cell.bas_angular(ib) == l, (ib, cell.bas_angular(ib), l)
        assert cell.bas_nctr(ib) == 1
        # PySCF may drop zero-coefficient primitives; compare the kept exponents.
        kept = sorted(cell.bas_exp(ib))
        assert all(any(abs(k - e) < 1e-12 * max(1, e) for e in exps) for k in kept), ib
    return cell


def main():
    ref_dir = os.path.join(ROOT, "testdata/reference")
    for name, spec in SYSTEMS.items():
        cell = build_cell(spec, precision=1e-13)
        cell_lo = build_cell(spec, precision=1e-10)
        e_nuc = cell.energy_nuc()
        e_nuc_lo = cell_lo.energy_nuc()
        mad = float(ptools.madelung(cell, np.zeros((1, 3))))
        mad_lo = float(ptools.madelung(cell_lo, np.zeros((1, 3))))
        b = cell.reciprocal_vectors()
        G = np.array([b[0], b[1], b[2], b[0] - b[1] + b[2], 2 * b[1], [0.37, -0.52, 0.81]])
        P = ft_ao.ft_aopair(cell, G, kpti_kptj=np.zeros((2, 3)))  # (ng, nao, nao)
        P0 = ft_ao.ft_aopair(cell, np.zeros((1, 3)), kpti_kptj=np.zeros((2, 3)))[0]
        S = cell.pbc_intor("int1e_ovlp", hermi=1)
        print(
            f"{name}: nao={cell.nao} E_nn={e_nuc:.14f} (prec-1e-10 diff {e_nuc - e_nuc_lo:.1e}) "
            f"madelung={mad:.14f} (diff {mad - mad_lo:.1e}) "
            f"|P(0)-S|={abs(P0 - S).max():.1e} |Im P(0)|={abs(P0.imag).max():.1e}"
        )
        out = dict(
            system=name,
            basis=spec["basis"],
            lattice=spec["lattice"],
            atoms=[dict(symbol=s, xyz_bohr=r) for s, r in spec["atoms"]],
            nao=int(cell.nao),
            energy_nuc=float(e_nuc),
            madelung=mad,
            gvectors=G.tolist(),
            pair_ft_re=P.real.tolist(),
            pair_ft_im=P.imag.tolist(),
            ovlp_latt=S.tolist(),
            generator="scripts/gen_pyscf_pbc_refs.py (pyscf 2.13.0, cell.precision=1e-13, cart=False)",
            ft_convention="P[g][m][n] = sum_T int exp(-i G.r) phi_m(r) phi_n(r-T) dr (ft_aopair, Gamma)",
        )
        path = os.path.join(ref_dir, f"pbc_{name}.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
        print("  wrote", os.path.relpath(path, ROOT))

    # Independent sanity anchor for the Madelung convention: simple cubic, a=1.
    # With a neutralising background, E = -1.4186487397/a per unit charge, so
    # PySCF's madelung (= -2 E) should be 2.8372974794/a.
    sc = pgto.Cell(a=np.eye(3) * 3.0, atom="H 0 0 0", basis="sto-3g", unit="B", spin=1, verbose=0)
    sc.build()
    print("simple-cubic a=3 madelung*a =", 3.0 * ptools.madelung(sc, np.zeros((1, 3))), "(expect 2.8372974794)")


if __name__ == "__main__":
    main()
