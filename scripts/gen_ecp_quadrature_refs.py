#!/usr/bin/env python3
"""Pinned reference values for ferric's own ECP quadrature (crates/ferric-integrals/src/ecp_quad/,
tests crates/ferric-integrals/tests/ecp_quadrature.rs). FINDINGS "Iteration 25".

PySCF and mpmath are ORACLES only: nothing here is used by the Rust engine. The fixture data (basis/ECP numbers) and
the oracles come from the design prototype in reference/pbc/ecp_quadrature/ (fixtures.py, oracle.py); the prototype's
own values (ecpq.py) are pinned alongside, so a Rust-vs-prototype disagreement is distinguishable from a
Rust-vs-PySCF one.

Writes testdata/reference/ecp_quadrature_refs.json:
  hi  : HI/LANL2DZ, bra = H STO-3G + I LANL2DZ at (H0, I0), ket = the same shells at +LV, 4 ECP centres (on the bra I
        shells, near the ket, near H, far) -> 5 x 5 x 4 = 100 triples;
  auh : AuH/def2-SVP + def2-ECP (f shells), bra at (Au0, H), ket at +AU_L, 3 ECP centres -> 15 x 15 x 3 = 675 triples;
        per triple: spherical block (libcint order = ferric order), PySCF ECPscalar_sph with the 1e-3 zero-weight
        screen guard (FINDINGS Iteration 22 (2a)) and the prototype value;
  tiny: the AuH element where PySCF is off (Au s(tight) - Au d_L around the ECP at the H end, ~2e-12): the independent
        radial x Lebedev quadrature (1202 x 200 per interval) value;
  bessel: ktil_n(z) = exp(-z) i_n(z), n = 0..16, mpmath at 40 digits, z over {0} U [1e-300, 1e5] incl. regime edges.
Shell coefficients are BARE (PySCF contraction normalisation, gto_norm folded in) -- ferric's EcpGaussianShell
convention.

Run (the PySCF venv; ~2-4 min):
  OPENBLAS_NUM_THREADS=1 /home/matt/qc/ferric/.venv/bin/python scripts/gen_ecp_quadrature_refs.py
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "reference", "pbc", "ecp_quadrature"))

import ecpq as E  # noqa: E402  (prototype values, pinned for comparison only)
import fixtures as F  # noqa: E402
import oracle as O  # noqa: E402
from measure import AU_CS, AU_L, HI_CS  # noqa: E402

OUT = os.path.join(ROOT, "testdata", "reference", "ecp_quadrature_refs.json")
COMMAND = "OPENBLAS_NUM_THREADS=1 /home/matt/qc/ferric/.venv/bin/python scripts/gen_ecp_quadrature_refs.py"


def shell_json(s):
    return dict(l=int(s["l"]), center=[float(x) for x in s["center"]], exps=[float(x) for x in s["exponents"]],
                coefs=[float(x) for x in s["coefficients"]])


def ecp_json(e):
    return dict(center=[float(x) for x in e["center"]], ams=[int(x) for x in e["ams"]], ns=[int(x) for x in e["ns"]],
                exps=[float(x) for x in e["exponents"]], coefs=[float(x) for x in e["coefficients"]])


def proto_val(sa, sb, e):
    return E.to_sph(E.triple_cart(sa, sb, e), sa["l"], sb["l"])


def fixture(bra, ket, ecps, label):
    t0 = time.time()
    pys, pro = [], []
    for sa in bra:
        for sb in ket:
            for e in ecps:
                pys.append(O.pyscf_value(sa, sb, e).ravel().tolist())
                pro.append(proto_val(sa, sb, e).ravel().tolist())
    d = max(np.abs(np.array(a) - np.array(b)).max() for a, b in zip(pys, pro))
    print(f"{label}: {len(pys)} triples, max |prototype - PySCF| {d:.2e} ({time.time() - t0:.0f} s)", flush=True)
    return dict(bra=[shell_json(s) for s in bra], ket=[shell_json(s) for s in ket], ecps=[ecp_json(e) for e in ecps],
                order="triples in (bra a, ket b, ecp u) order, u fastest; each a row-major (2la+1) x (2lb+1) block",
                pyscf=pys, proto=pro)


def bessel_table():
    mp.mp.dps = 40
    zs = [0.0, 1e-300, 1e-30, 1e-9, 1e-8 * (1 - 1e-9), 1e-8, 1e-8 * (1 + 1e-9), 1e-6]
    zs += list(np.geomspace(1e-4, 1e5, 150))
    for n in range(4, 17):  # upward/downward switch max(16, 4 n)
        zs += [4.0 * n - 1e-3, 4.0 * n, 4.0 * n + 1e-3]
    zs = sorted(set(float(z) for z in zs))
    vals = []
    for z in zs:
        row = []
        for n in range(17):
            if z == 0.0:
                row.append(float(n == 0))
            else:
                zz = mp.mpf(z)
                row.append(float(mp.exp(-zz) * mp.sqrt(mp.pi / (2 * zz)) * mp.besseli(n + mp.mpf(1) / 2, zz)))
        vals.append(row)
    return dict(z=zs, nmax=16, values=vals)


def main():
    import pyscf
    out = dict(command=COMMAND, pyscf_version=pyscf.__version__,
               note="PySCF ECPscalar_sph with the 1e-3 zero-weight screen-guard primitive; prototype = "
                    "reference/pbc/ecp_quadrature/ecpq.py")
    out["hi"] = fixture(F.hi_shells(F.H0, F.I0), F.hi_shells(F.H0 + F.LV, F.I0 + F.LV),
                        [F.ecp_i(c) for c in HI_CS], "HI/LANL2DZ")
    aub = F.auh_shells(F.AU0, F.H_AU)
    auk = F.auh_shells(F.AU0 + AU_L, F.H_AU + AU_L)
    out["auh"] = fixture(aub, auk, [F.ecp_au(c) for c in AU_CS], "AuH/def2-SVP")
    t0 = time.time()
    e = F.ecp_au(AU_CS[2])
    leb = O.lebedev_value(aub[0], auk[9], e, 1202, 200)
    out["tiny"] = dict(bra=shell_json(aub[0]), ket=shell_json(auk[9]), ecp=ecp_json(e),
                       lebedev=leb.ravel().tolist(), pyscf=O.pyscf_value(aub[0], auk[9], e).ravel().tolist(),
                       proto=proto_val(aub[0], auk[9], e).ravel().tolist(), nang=1202, nper=200)
    print(f"tiny: max |proto - Lebedev| {np.abs(proto_val(aub[0], auk[9], e) - leb).max():.2e} "
          f"({time.time() - t0:.0f} s)", flush=True)
    out["bessel"] = bessel_table()
    with open(OUT, "w") as f:
        json.dump(out, f)
        f.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
