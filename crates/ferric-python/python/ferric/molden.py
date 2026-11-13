"""Molden files from ferric SCF results.

`write(path, mol, basis_set, result)` writes the geometry, the basis and the
molecular orbitals of a `run_rhf`, `run_dft`/`run_ksdft`, `run_uhf` or
`run_rohf` result in the Molden format, which Avogadro, Jmol, IBOView,
Multiwfn, JANPA and PySCF (`pyscf.tools.molden.load`) read.

Conventions of the file (Bohr, Hartree):

* ``[Atoms] AU``: one line per atom, ``symbol index charge x y z``. The charge
  column is the nuclear charge the SCF used: ``Z`` minus the ECP core
  electrons, and 0 for a ghost atom.
* ``[GTO]``: every atom's contracted shells in ferric's AO order, with
  coefficients of normalised primitives (`BasisSet.shells`).
* Shell flags: ``[5D7F]``, ``[5D10F]`` or ``[7F]`` when d and/or f shells
  are spherical, ``[9G]`` when g shells are; no flag means Cartesian.
  Molden has one setting per angular momentum, so a basis that mixes
  spherical and Cartesian shells of the same l raises ValueError, as does an
  h or higher shell (Molden stops at g).
* ``[MO]``: one block per orbital (``Sym= A``, ``Ene=``, ``Spin=``,
  ``Occup=``, then every AO coefficient). An open-shell result writes all
  alpha orbitals, then all beta orbitals. ROHF writes its single MO set
  twice, with the alpha and beta occupations.
* Numbers are written so that every double reads back exactly: 17
  significant digits, and Python's shortest round-trip form for
  occupations (``2.0``).

Molden orders and normalises the AOs of a shell its own way: spherical
components m = 0, +1, -1, +2, -2, ... and Cartesian components in a fixed
list (d: xx yy zz xy xz yz; f: xxx yyy zzz xyy xxy xxz xzz yzz yyz xyz;
g: xxxx yyyy zzzz xxxy xxxz yyyx yyyz zzzx zzzy xxyy xxzz yyzz xxyz yyxz
zzxy), each Cartesian component normalised on its own. ferric (libint2)
orders spherical components m = -l..l and Cartesian ones lexicographically
(xx xy xz yy yz zz), with all Cartesian components of a shell sharing the
normalisation of x^l. The writer permutes each shell's coefficients and
scales coefficient i by sqrt(S_ii), the norm of ferric's AO i, so that
``chi_ferric @ C_ferric == chi_molden @ C_molden`` at every point.

ECPs: Molden has no standard ECP block. The file carries the valence basis
and valence orbitals, the reduced charge column and a ``[core]`` section
(``atom index : core electrons``, the Molden2AIM extension that PySCF and
Multiwfn read); readers that do not know ``[core]`` skip it and see a
valence-only molecule.

Ghost atoms are written as atoms with charge 0: their basis functions carry
orbital amplitude, so they must be in ``[GTO]``. A viewer draws them as
atoms of their element, and a reader that ignores the charge column (PySCF)
treats them as real atoms; the AOs, orbitals and density are unaffected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["ao_map", "orbitals", "to_string", "write"]

# Molden's Cartesian component order per l, as (lx, ly, lz).
_MOLDEN_CART = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    3: [
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
        (1, 2, 0),
        (2, 1, 0),
        (2, 0, 1),
        (1, 0, 2),
        (0, 1, 2),
        (0, 2, 1),
        (1, 1, 1),
    ],
    4: [
        (4, 0, 0),
        (0, 4, 0),
        (0, 0, 4),
        (3, 1, 0),
        (3, 0, 1),
        (1, 3, 0),
        (0, 3, 1),
        (1, 0, 3),
        (0, 1, 3),
        (2, 2, 0),
        (2, 0, 2),
        (0, 2, 2),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
    ],
}
_LETTERS = "spdfg"
_MAX_L = 4


def _ferric_cart(ang: int) -> list[tuple[int, int, int]]:
    """libint2's Cartesian order: lx from l down, then ly from l - lx down."""
    return [
        (lx, ang - lx - lz, lz)
        for lx in range(ang, -1, -1)
        for lz in range(0, ang - lx + 1)
    ]


def _shell_permutation(ang: int, pure: bool) -> list[int]:
    """perm[j] = ferric component index of Molden component j."""
    if pure and ang >= 2:
        ms = [0] + [s * m for m in range(1, ang + 1) for s in (1, -1)]
        return [m + ang for m in ms]
    ferric = _ferric_cart(ang)
    return [ferric.index(c) for c in _MOLDEN_CART[ang]]


def _shell_list(mol, basis_set) -> list[tuple[int, int, bool, list, list]]:
    """(atom, l, pure, exponents, coefficients) per shell, in ferric's AO order."""
    shells = []
    cache: dict[int, list] = {}
    for atom, z in enumerate(mol.atomic_numbers()):
        if z not in cache:
            cache[z] = basis_set.shells(z)
        for ang, pure, exps, coefs in cache[z]:
            shells.append((atom, int(ang), bool(pure and ang >= 2), exps, coefs))
    return shells


def ao_map(mol, basis_set, overlap=None) -> tuple[np.ndarray, np.ndarray]:
    """(perm, scale) taking ferric MO coefficients to Molden's AOs.

    ``C_molden = scale[:, None] * C_ferric[perm]``: Molden AO j is ferric AO
    ``perm[j]`` and ``scale[j]`` is that AO's norm, sqrt(S_ii). ``overlap``
    is ferric's AO overlap matrix; it is computed when not given.
    """
    from . import AoIntegrals

    shells = _shell_list(mol, basis_set)
    for _, ang, _, _, _ in shells:
        if ang > _MAX_L:
            raise ValueError(
                f"Molden supports s, p, d, f and g shells; this basis has an l={ang} shell"
            )
    perm = []
    offset = 0
    for _, ang, pure, _, _ in shells:
        local = _shell_permutation(ang, pure)
        perm.extend(offset + i for i in local)
        offset += len(local)
    if overlap is None:
        overlap = AoIntegrals(mol, basis_set).overlap()
    s_diag = np.diagonal(np.asarray(overlap, dtype=float))
    if s_diag.shape[0] != offset:
        raise ValueError(
            f"basis places {s_diag.shape[0]} AOs on the molecule but its shells list {offset}"
        )
    perm = np.asarray(perm, dtype=np.int64)
    return perm, np.sqrt(s_diag[perm])


# An occupation this close to an integer is that integer (round-off of the
# projection diag(C^T S D S C) is ~1e-13 at convergence).
OCC_SNAP = 1e-8


def _snap(occ) -> np.ndarray:
    occ = np.asarray(occ, dtype=float)
    near = np.round(occ)
    return np.where(np.abs(occ - near) < OCC_SNAP, near, occ)


def _occupations(c: np.ndarray, d: np.ndarray, s: np.ndarray) -> np.ndarray:
    """diag(C^T S D S C): the occupation of each MO column in density D."""
    sc = s @ c
    return np.einsum("mk,mn,nk->k", sc, d, sc)


def orbitals(mol, basis_set, result, overlap=None) -> list[dict]:
    """The orbital sets of `result` in ferric's AO basis.

    One dict per spin channel (``spin`` "Alpha"/"Beta", ``coefficients``
    (nao, nmo), ``energies``, ``occupations``): one channel for a
    closed-shell result (occupations 0..2), two for an open-shell one.
    Closed-shell occupations are ``diag(C^T S D S C)`` of the result's
    density; open-shell ones are the result's ``occupations_alpha/beta``.
    An occupation within ``OCC_SNAP`` of an integer is written as that
    integer.
    """
    from . import AoIntegrals

    if hasattr(result, "mo_coefficients_alpha"):
        return [
            {
                "spin": "Alpha",
                "coefficients": np.asarray(result.mo_coefficients_alpha()),
                "energies": np.asarray(result.orbital_energies_alpha()),
                "occupations": _snap(result.occupations_alpha()),
            },
            {
                "spin": "Beta",
                "coefficients": np.asarray(result.mo_coefficients_beta()),
                "energies": np.asarray(result.orbital_energies_beta()),
                "occupations": _snap(result.occupations_beta()),
            },
        ]
    if not (hasattr(result, "mo_coefficients") and hasattr(result, "density")):
        raise TypeError(
            f"{type(result).__name__} carries no molecular orbitals; Molden export "
            "takes a run_rhf, run_dft/run_ksdft, run_uhf or run_rohf result"
        )
    c = np.asarray(result.mo_coefficients())
    if overlap is None:
        overlap = AoIntegrals(mol, basis_set).overlap()
    occ = _occupations(c, np.asarray(result.density()), np.asarray(overlap))
    return [
        {
            "spin": "Alpha",
            "coefficients": c,
            "energies": np.asarray(result.orbital_energies()),
            "occupations": _snap(occ),
        }
    ]


def _flags(shells) -> list[str]:
    """The Molden shell-type flags for this basis (empty: all Cartesian)."""
    purity: dict[int, set[bool]] = {}
    for _, ang, pure, _, _ in shells:
        if ang >= 2:
            purity.setdefault(ang, set()).add(pure)
    for ang, kinds in purity.items():
        if len(kinds) > 1:
            raise ValueError(
                f"the basis mixes spherical and Cartesian {_LETTERS[ang]} shells; "
                "Molden has one setting per angular momentum"
            )
    d_pure = purity.get(2) == {True}
    f_pure = purity.get(3) == {True}
    f_cart = purity.get(3) == {False}
    flags = []
    if d_pure:
        flags.append("[5D10F]" if f_cart else "[5D7F]")
    elif f_pure:
        flags.append("[7F]")
    if purity.get(4) == {True}:
        flags.append("[9G]")
    return flags


def _num(x: float) -> str:
    return f"{float(x): .16e}"


def to_string(mol, basis_set, result, title: str | None = None) -> str:
    """The Molden file text for `result` (see the module docstring)."""
    from . import AoIntegrals
    from . import ferric as _ext

    shells = _shell_list(mol, basis_set)
    overlap = np.asarray(AoIntegrals(mol, basis_set).overlap())
    perm, scale = ao_map(mol, basis_set, overlap)
    channels = orbitals(mol, basis_set, result, overlap)
    nao = len(perm)

    ghosts = mol.is_ghost()
    numbers = mol.atomic_numbers()
    cores = [
        0 if g else int(basis_set.ecp_core_electrons(z))
        for z, g in zip(numbers, ghosts, strict=True)
    ]
    out = ["[Molden Format]", "[Title]"]
    version = getattr(_ext, "__version__", "")
    out.append(f" {title}" if title else f" ferric {version}".rstrip())
    out.append("[Atoms] AU")
    for i, (sym, z, g, core, xyz) in enumerate(
        zip(mol.symbols(), numbers, ghosts, cores, mol.coords_bohr(), strict=True)
    ):
        charge = 0 if g else z - core
        coords = " ".join(_num(v) for v in xyz)
        out.append(f"{sym:<3s}{i + 1:6d}{charge:4d} {coords}")

    out.append("[GTO]")
    current = None
    for atom, ang, _, exps, coefs in shells:
        if atom != current:
            if current is not None:
                out.append("")
            out.append(f"{atom + 1:4d} 0")
            current = atom
        out.append(f" {_LETTERS[ang]} {len(exps):4d} 1.00")
        for e, c in zip(exps, coefs, strict=True):
            out.append(f"  {_num(e)} {_num(c)}")
    out.append("")
    out.extend(_flags(shells))

    if any(cores):
        out.append("[core]")
        out.extend(f"{i + 1} : {n}" for i, n in enumerate(cores) if n)

    out.append("[MO]")
    for ch in channels:
        c = np.asarray(ch["coefficients"], dtype=float)
        if c.shape[0] != nao:
            raise ValueError(
                f"result has {c.shape[0]} AO rows but the basis places {nao} AOs on mol"
            )
        cm = scale[:, None] * c[perm]
        for k in range(c.shape[1]):
            out.append(" Sym= A")
            out.append(f" Ene= {_num(ch['energies'][k])}")
            out.append(f" Spin= {ch['spin']}")
            out.append(f" Occup= {float(ch['occupations'][k])!r}")
            out.extend(f"{j + 1:5d} {_num(v)}" for j, v in enumerate(cm[:, k]))
    return "\n".join(out) + "\n"


def write(path, mol, basis_set, result, title: str | None = None) -> None:
    """Write `result`'s orbitals to a Molden file at `path`.

    `mol` and `basis_set` are the molecule and orbital basis the SCF ran on.
    `result` is a `RhfResult`, `DftResult` or `UhfResult` (UHF or ROHF);
    A `run_rhf(symmetry=True)` result carries no MO coefficients in the input
    frame and raises NotImplementedError; a result without orbitals raises
    TypeError. `title` defaults to "ferric <version>".
    """
    Path(path).write_text(to_string(mol, basis_set, result, title))
