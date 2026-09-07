#!/usr/bin/env python3
"""Generate PySCF direct-SCF J/K references for ferric's DirectJK / LinK builders.

WHY THIS FILE EXISTS
--------------------
Before this script, ferric's only cross-check on the direct (non-DF) J/K path
was `crates/ferric-scf/tests/mpi_direct_jk_correctness.rs`, whose RHF anchor
(-74.963227299664 Ha) was SELF-MEASURED from a ferric run — it pins ferric to
its own past behaviour, not to physics. And the LinK-vs-dense tests in
`crates/ferric-scf/src/link_k.rs` all run at `thresh = 1e-14` on 3-5 atom
molecules, where `SignificantPairs::build` retains essentially every pair, so
no quartet is ever screened and the assertions pass for reasons unrelated to
screening. This script supplies the INDEPENDENT reference the anchor needed.

WHAT IS COMPARED, AND HOW (read before changing anything)
---------------------------------------------------------
Two DIFFERENT comparisons, deliberately:

1. ENERGY: each code's OWN converged E_tot. Both codes solve the same
   variational problem, so their converged energies are comparable without
   sharing anything. Tolerance 1e-8 Ha.

2. J AND K MATRICES: contracted from ONE density supplied to BOTH codes.
   This script converges PySCF's RHF and dumps that converged AO density
   `dm`, and ferric's test READS `dm` from this JSON and builds J/K on it.
   Comparing each code's J/K at its OWN converged density would compare SCF
   SOLUTIONS (which differ at the 1e-9-ish convergence floor and by orbital
   phase/degeneracy choices), not the J/K BUILDERS — the thing under test.
   The J/K deviation would then be dominated by density differences and would
   tell you nothing about screening correctness.

DIRECT-SCF VERIFICATION (repo memory `pyscf-caches-eri-benchmark-trap`)
-----------------------------------------------------------------------
`scf.RHF` silently caches the full in-core AO ERI tensor in `mf._eri` whenever
it estimates the tensor fits in `mol.max_memory`. If that happens the run did
NOT exercise PySCF's direct path, and the reference is for a different code
path than the one ferric's DirectJK implements. We assert `mf._eri is None`
after the SCF and FAIL LOUDLY if it is populated, and we additionally assert
`mf.direct_scf` is still True. `mol.max_memory` is pinned low to make the
in-core route unattractive to PySCF's own heuristic as well.

THE BASIS MUST BE BUILT FROM FERRIC'S OWN JSON (measured, not assumed)
----------------------------------------------------------------------
`gto.M(basis="ccpvdz")` does NOT give AO-comparable matrices, even though it
is the same basis set mathematically. PySCF's internal cc-pVDZ stores carbon's
s block as an 8-primitive/2-column general contraction plus a separate
1-primitive shell; ferric's bundled `cc-pvdz.json` stores all 9 s primitives
in ONE block with 3 coefficient columns, and ferric's BSE parser splits each
column into its own segmented shell (renormalizing it to unit self-overlap).

Both layouts span the SAME space and give the same SCF energy — measured, not
assumed: PySCF run against ferric's layout gives -157.30705853138934 Ha on
alkane_4 vs -157.30705853139253 Ha against its own cc-pVDZ, a difference of
3.2e-12 Ha (round-off). But the individual AO FUNCTIONS differ, so J and K are
NOT elementwise comparable between the two layouts. Attempting it produced a
max overlap deviation of 2.9e-1 — not a subtle tolerance question, and NOT an
AO-ordering problem (the sparsity patterns matched exactly; the d shells, being
single-primitive, agreed to machine precision while every contracted s/p AO
disagreed).

So this script constructs the PySCF `Mole` from ferric's OWN bundled JSON,
one segmented shell per coefficient column, exactly mirroring ferric's parser.
With that, the AO orderings coincide (the permutation is the identity) and the
overlap matrices agree to 1.2e-15. `ao_permutation` is still emitted and still
verified on the Rust side against the overlap, so if either code's convention
ever shifts, the test fails loudly on the overlap BEFORE any J/K number is
trusted — which is exactly how this discrepancy was caught in the first place.

Usage:
    OPENBLAS_NUM_THREADS=1 python3 scripts/gen_pyscf_directjk_refs.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from pyscf import gto, scf

ROOT = Path(__file__).resolve().parents[1]
REFDIR = ROOT / "testdata" / "reference"
MOLDIR = ROOT / "testdata" / "molecules"

# (molecule tag, xyz file stem, store_matrices)
#
# `store_matrices` controls whether the density/J/K/overlap matrices are
# written. They are dense (no useful sparsity: 76-85% of elements exceed
# 1e-10), so for alkane_8 they run to ~20k floats each and the file lands at
# 1.4 MB — far outside this repo's convention, where the largest existing
# reference is 35 KB, and over the 500 KB pre-commit ceiling.
#
# alkane_4 (387 KB) carries the matrix-level cross-check for the J/K BUILDERS,
# which is where a contraction defect would show. alkane_8 stores energies
# only; its role is the SIZE-SCALING evidence (the screening defect grows from
# 5e-9 to 1.1e-4 Ha between the two), and that needs no stored matrices —
# `known_defect_screening_shifts_coulomb_energy_with_size` contracts ferric's
# own density at two thresholds and compares against itself.
CASES = [
    ("alkane_4", "alkane_4", True),
    ("alkane_8", "alkane_8", False),
]

BASIS_TAG = "ccpvdz"
# ferric's bundled basis JSON, parsed here into PySCF's basis format so that
# both codes see BIT-IDENTICAL AO functions (see the module docstring).
FERRIC_BASIS_JSON = ROOT / "crates" / "ferric-core" / "src" / "basis" / "bundled" / "cc-pvdz.json"

# Elements appearing in the alkane test set.
ELEMENTS = {"H": "1", "C": "6"}


def ferric_basis():
    """Build a PySCF basis dict from ferric's bundled cc-pVDZ JSON.

    Mirrors `parse_bse_json` in `crates/ferric-core/src/basis.rs`: every
    coefficient COLUMN of an `electron_shells` entry becomes its own segmented
    shell, in file order, so PySCF's AO layout matches ferric's exactly.

    ferric additionally renormalizes each contraction to unit self-overlap
    (`renormalize_contraction`). PySCF normalizes contractions on input too, so
    the two agree; the Rust-side overlap check is what actually proves it,
    and it does — to 1.2e-15.
    """
    data = json.loads(FERRIC_BASIS_JSON.read_text())
    basis = {}
    for sym, z in ELEMENTS.items():
        shells = []
        for sh in data["elements"][z]["electron_shells"]:
            ang = sh["angular_momentum"]
            exps = [float(x) for x in sh["exponents"]]
            cols = [[float(x) for x in c] for c in sh["coefficients"]]
            if len(ang) == 1:
                # Each coefficient column is a separate contraction.
                for col in cols:
                    shells.append([ang[0]] + [[e, c] for e, c in zip(exps, col)])
            else:
                # Multiple angular momenta (SP-style): column k belongs to ang[k].
                for k, l in enumerate(ang):
                    col = cols[k]
                    shells.append([l] + [[e, c] for e, c in zip(exps, col)])
        basis[sym] = shells
    return basis


def spherical_m_order_pyscf(l):
    """PySCF's within-shell m ordering for a spherical shell of momentum `l`.

    PySCF orders real solid harmonics as m = -l, -l+1, ..., 0, ..., l-1, l.
    """
    return list(range(-l, l + 1))


def spherical_m_order_libint(l):
    """libint2's within-shell m ordering for a spherical shell (SHELL_ORDER_STANDARD).

    Same as PySCF: m = -l .. +l. Kept as a separate named function so that if a
    future libint2 build flips to a different solid-harmonic ordering, exactly
    one place needs changing and the overlap cross-check in the Rust test will
    have already caught the mismatch.
    """
    return list(range(-l, l + 1))


def build_ao_permutation(mol):
    """perm[i] = PySCF AO index of what ferric calls AO i.

    Both codes emit AOs atom-major, then in basis-file shell order, then
    contraction-major within a shell, then by m. The one structural difference
    is that PySCF keeps GENERAL contractions (cc-pVDZ carbon's s block is a
    single `_bas` entry with `nctr = 2` over 8 primitives) while ferric's BSE
    parser splits every coefficient column into its own segmented shell. That
    changes the shell COUNT but not the AO ORDER, because PySCF lays a general
    contraction out as contraction-major — exactly the order ferric's
    consecutive segmented shells produce. So this walker expands each PySCF
    shell into `nctr` blocks of `2l+1` (or `(l+1)(l+2)/2` when Cartesian) and
    maps m-slot to m-slot inside each block.

    For cc-pVDZ the two m-orderings coincide, so this returns the identity;
    the function exists so a future basis or a libint2 ordering change is a
    one-line edit here rather than a silent wrong comparison. The Rust test
    verifies the result against the overlap matrix regardless, so a wrong
    permutation fails loudly on the overlap before any J/K number is trusted.
    """
    perm = []
    off = 0
    cart = mol.cart
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nfn = (l + 1) * (l + 2) // 2 if cart else 2 * l + 1
        for _ in range(mol.bas_nctr(ib)):
            if cart or l < 2:
                # l <= 1 is Cartesian in both codes with the same ordering
                # (s: 1 function; p: x, y, z).
                perm.extend(range(off, off + nfn))
            else:
                py = spherical_m_order_pyscf(l)
                fe = spherical_m_order_libint(l)
                for m in fe:
                    perm.append(off + py.index(m))
            off += nfn
    assert off == mol.nao_nr(), (off, mol.nao_nr())
    assert sorted(perm) == list(range(off)), "permutation is not a bijection"
    return perm


def pack_lower(m):
    """Symmetric matrix -> flat lower triangle (row-major, i >= j), 12 s.f.

    Halves the stored element count and drops the 17-digit repr noise; the
    Rust side rebuilds the full square. 12 significant digits is far tighter
    than the 1e-9 J/K bar these references are compared against.
    """
    a = np.asarray(m)
    n = a.shape[0]
    asym = np.abs(a - a.T).max()
    if asym > 1e-12:
        raise SystemExit(f"matrix is not symmetric (max|M - M^T| = {asym:.3e}); cannot pack")
    idx = np.tril_indices(n)
    return [float(f"{v:.12g}") for v in a[idx]]


def run_case(tag, stem, store_matrices):
    xyz = MOLDIR / f"{stem}.xyz"
    mol = gto.M(
        atom=str(xyz),
        basis=ferric_basis(),
        unit="Angstrom",
        cart=False,
        verbose=0,
        # Keep the in-core ERI route unattractive to PySCF's own heuristic.
        # The hard guarantee is the `mf._eri is None` assert below; this just
        # avoids burning memory before we get there.
        max_memory=200,
    )
    nbf = mol.nao_nr()
    print(f"[{tag}] natm={mol.natm} nbf={nbf} nelec={mol.nelectron}")

    mf = scf.RHF(mol)
    mf.direct_scf = True
    mf.conv_tol = 1e-12
    mf.init_guess = "hcore"
    mf.max_cycle = 200
    e_tot = mf.kernel()

    if not mf.converged:
        raise SystemExit(f"[{tag}] PySCF RHF did NOT converge — refusing to write a reference")

    # --- the direct-path guarantee (repo memory: pyscf-caches-eri-benchmark-trap) ---
    if mf._eri is not None:
        raise SystemExit(
            f"[{tag}] FATAL: mf._eri is populated (shape {np.asarray(mf._eri).shape}) after the "
            "SCF. PySCF built the in-core AO ERI tensor, so this run did NOT exercise the "
            "direct path and the resulting J/K are NOT a reference for ferric's DirectJK. "
            "Lower mol.max_memory or set mf.direct_scf explicitly and re-run."
        )
    if not mf.direct_scf:
        raise SystemExit(f"[{tag}] FATAL: mf.direct_scf is False after the SCF")
    print(f"[{tag}] direct-path verified: mf._eri is None, mf.direct_scf is True")

    dm = mf.make_rdm1()
    # PySCF's get_jk on the same density — the J/K under comparison.
    j, k = mf.get_jk(mol, dm, hermi=1)
    ovlp = mol.intor("int1e_ovlp")

    perm = build_ao_permutation(mol)

    rec = {
        "molecule": tag,
        "basis": "cc-pVDZ",
        # NOT PySCF's built-in "ccpvdz": the Mole was built from ferric's own
        # bundled cc-pvdz.json, one segmented shell per coefficient column, so
        # the AO functions are bit-identical across the two codes and J/K are
        # elementwise comparable. See this script's module docstring.
        "basis_source": "ferric bundled crates/ferric-core/src/basis/bundled/cc-pvdz.json",
        "method": "rhf-directjk",
        "source": "pyscf",
        "pyscf_version": __import__("pyscf").__version__,
        "natm": int(mol.natm),
        "nbf": int(nbf),
        "nelec": int(mol.nelectron),
        "conv_tol": 1e-12,
        "init_guess": "hcore",
        "direct_scf": True,
        "eri_cached": False,
        "converged": bool(mf.converged),
        "e_tot": float(e_tot),
        "e_nuc": float(mol.energy_nuc()),
        "ao_permutation": [int(p) for p in perm],
        # Matrices are in PYSCF AO order, stored as the LOWER TRIANGLE ONLY
        # (row-major, i >= j), because all four are symmetric — verified to
        # <= 4.5e-16 on both molecules. Full square storage pushed alkane_8
        # past the repo's 500 KB per-file limit for no information gain.
        # Values are rounded to 12 significant digits, ~3 orders tighter than
        # the tightest bar any consumer asserts (1e-9 on J/K).
        "storage": "lower_triangle_rowmajor" if store_matrices else "energy_only",
    }
    if store_matrices:
        rec["density"] = pack_lower(dm)
        rec["j"] = pack_lower(j)
        rec["k"] = pack_lower(k)
        rec["overlap"] = pack_lower(ovlp)

    out = REFDIR / f"{tag}_{BASIS_TAG}_directjk.json"
    out.write_text(json.dumps(rec))
    size_mb = out.stat().st_size / 1e6
    print(f"[{tag}] E_tot = {e_tot:.12f} Ha  ->  {out.name} ({size_mb:.1f} MB)")
    return rec


def main():
    REFDIR.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or None
    for tag, stem, store_matrices in CASES:
        if only and tag not in only:
            continue
        run_case(tag, stem, store_matrices)


if __name__ == "__main__":
    main()
