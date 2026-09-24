#!/usr/bin/env python3
"""Generate PySCF UHF reference data for the HeNe+ cDFT-ET lane anchors.

Feeds two test files:

  crates/ferric-scf/tests/hene_atomic_ip_anchor.rs
      - four atomic energies in def2-SVP AND def2-TZVP, plus each basis's dIP.
        The four energies and their dIP combination are the R -> infinity limit
        of the HeNe+ diabatic gap, and the only externally-referenced fact in
        that lane.
      - The SECOND basis is not redundant. It is what makes the anchor
        PERTURBABLE: an adversarial review stubbed the Rust solver to return the
        def2-SVP constants and every assertion still passed, because a pure
        reference comparison cannot distinguish "reproduced" from "echoed". A
        stub keyed on (symbol, charge, multiplicity) cannot see the basis
        argument, so E(TZVP) < E(SVP) and the TZVP references both break it.
      - dIP is printed as its own line so the Rust side can store it as an
        INDEPENDENT constant. Recomputing dIP from the four stored energies on
        both sides of the assertion would apply any formula error twice and
        cancel it.

  crates/ferric-scf/tests/hene_atom_polarizability.rs
      - finite-field static polarizabilities of He and Ne in def2-SVP. The
        z-field is folded into get_hcore as h = T + V + f*z, matching ferric's
        ExternalPotential.field sign convention.

Usage: OPENBLAS_NUM_THREADS=1 python scripts/gen_pyscf_atomic_ip_refs.py
"""

from pyscf import gto, scf

HA2EV = 27.211386245988
ATOMS = [
    ("He", "He", 0, 0),
    ("He+", "He", 1, 1),
    ("Ne", "Ne", 0, 0),
    ("Ne+", "Ne", 1, 1),
]


def run(sym, charge, spin, basis):
    m = gto.M(
        atom=f"{sym} 0 0 0",
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=0,
        unit="Angstrom",
    )
    mf = scf.UHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    e = mf.kernel()
    assert mf.converged, f"{sym} q={charge} s={spin}/{basis} NOT converged"
    return e, mf.spin_square(), m.nao


def atomic_block(basis):
    print(f"--- UHF/{basis} atomic energies ---")
    res = {}
    for label, sym, q, s in ATOMS:
        e, ss, nao = run(sym, q, s, basis)
        res[label] = e
        print(
            f"{label:4s} nao={nao:3d}  E = {e:.12f}  <S^2>={ss[0]:.6f} 2S+1={ss[1]:.4f}"
        )
    dip = (res["He+"] + res["Ne"]) - (res["He"] + res["Ne+"])
    print(
        f"IP(He) = {res['He+'] - res['He']:.12f} Ha = {(res['He+'] - res['He']) * HA2EV:.6f} eV"
    )
    print(
        f"IP(Ne) = {res['Ne+'] - res['Ne']:.12f} Ha = {(res['Ne+'] - res['Ne']) * HA2EV:.6f} eV"
    )
    # Store THIS line in the Rust test as DIP_PYSCF / DIP_PYSCF_TZ, not a
    # recomputation from the four constants above.
    print(f"dIP    = {dip:.12f} Ha = {dip * HA2EV:.6f} eV")
    print()
    return res


def energy_in_field(sym, f):
    """UHF/def2-SVP energy of a neutral atom in a uniform z-field.

    Sign convention: h = T + V + f*z, matching ferric's
    ExternalPotential{field: Some([0, 0, f])}.
    """
    m = gto.M(atom=f"{sym} 0 0 0", basis="def2-svp", charge=0, spin=0, verbose=0)
    mf = scf.UHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 300
    if f != 0.0:
        with m.with_common_orig((0, 0, 0)):
            dz = m.intor("int1e_r", comp=3)[2]
        h = m.intor("int1e_kin") + m.intor("int1e_nuc") + f * dz
        mf.get_hcore = lambda *a: h
    e = mf.kernel()
    assert mf.converged, f"{sym} F={f} NOT converged"
    return e


def polarizability_block():
    print("--- UHF/def2-SVP finite-field alpha_zz (a.u.) ---")
    print("alpha_zz = -[E(+h) - 2E(0) + E(-h)] / h^2, central 3-point.")
    print("The Rust test stores the h=0.01 row; the other steps show the")
    print("O(h^2) approach to the limit, alpha(h) = alpha + (gamma/12) h^2, so")
    print("successive changes fall by 4x per halving. The second difference")
    print("itself is alpha*h^2 (~1e-5 Ha at h=0.005 for He); what limits small h")
    print("is its f64 cancellation error, ~1e-13 Ha on Ne's ~-128 Ha energies,")
    print("which h^2 amplifies into alpha. The Rust test therefore uses steps")
    print("0.04..0.005, where that error is ~1e-9 in alpha against 5e-7 changes.")
    for sym in ("He", "Ne"):
        e0 = energy_in_field(sym, 0.0)
        prev = None
        print(f"  {sym}: E(F=0) = {e0:.12f} Ha")
        for h in (0.04, 0.02, 0.01, 0.005):
            a = -(energy_in_field(sym, h) - 2 * e0 + energy_in_field(sym, -h)) / (h * h)
            d = "" if prev is None else f"   (change {a - prev:+.2e})"
            print(f"    h={h:<8} alpha_zz = {a:.8f}{d}")
            prev = a
    print()
    print("UNIT WARNING: the often-quoted He value 0.205 is ANGSTROM^3, not a.u.")
    print("  a.u.: alpha_He = 1.3838, alpha_Ne = 2.6693 (experimental).")
    print("  def2-SVP has no diffuse functions and gives ~1/3 of those; that is")
    print("  expected, and the lane needs the METHOD-CONSISTENT differential.")


if __name__ == "__main__":
    for basis in ("def2-svp", "def2-tzvp"):
        atomic_block(basis)
    polarizability_block()
