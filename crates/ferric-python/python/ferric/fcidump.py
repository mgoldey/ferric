"""FCIDUMP files: the active-space Hamiltonian for DMRG and selected-CI codes.

`write(path, mol, basis_set, mo_coeff, n_core, n_active, nelec, ms2)` writes
the Hamiltonian of ``n_active`` orbitals above ``n_core`` doubly occupied ones
in the standard FCIDUMP format that block2, Dice, PySCF
(`pyscf.tools.fcidump.read`) and most CI codes read. `read(path)` returns it.
Both are built only from ferric's AO blocks (`AoIntegrals.hcore`, `veff`,
`ao2mo`).

Conventions (Hartree, spatial orbitals, real orbitals):

* Header ``&FCI NORB=, NELEC=, MS2=, ORBSYM=1,..., ISYM=1, &END``. ``NELEC``
  counts the ACTIVE electrons only; the ``n_core`` doubly occupied orbitals
  are folded into ``ECORE`` and the one-electron integrals.
* ``h_pq = C_p^T (h + J[D_c] - K[D_c]/2) C_q`` with ``D_c = 2 C_core C_core^T``
  the inactive density: the core-electron potential is in ``h``.
* ``ECORE = E_nn + Tr[D_c h] + 1/2 Tr[D_c (J[D_c] - K[D_c]/2)]``: nuclear
  repulsion plus the inactive electrons' energy (the same constant as
  ``CasCiResult.e_core`` and ``CasScfResult.e_core``; with ECP atoms the
  ECP is in ``h`` and the core charge in ``E_nn``).
* Two-electron integrals ``(pq|rs)`` in chemist notation, one line per
  8-fold-unique element ``p>=q, r>=s, pq>=rs`` (1-based, ``val p q r s``),
  one-electron lines ``val p q 0 0`` (``p>=q``), then ``ECORE 0 0 0 0``.
  Numbers are written in Python's shortest round-trip form, so every double
  reads back exactly.
* The energy of the file's Hamiltonian in the ``(n_alpha, n_beta)`` sector
  is ``E_total`` of CAS-CI / CASSCF for the same orbitals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["integrals", "read", "to_string", "write"]


def integrals(mol, basis_set, mo_coeff, n_core: int, n_active: int):
    """``(h, eri, ecore)`` of the active space of `mo_coeff` (nao, nmo).

    Orbitals ``n_core .. n_core + n_active`` are active, the first `n_core`
    are doubly occupied, the rest are discarded. `h` is (n, n), `eri` is
    ``(pq|rs)`` with shape (n,)*4, `ecore` is the constant of the module
    docstring.
    """
    from . import AoIntegrals

    c = np.asarray(mo_coeff, dtype=float)
    if n_core < 0 or n_active < 1 or n_core + n_active > c.shape[1]:
        raise ValueError(
            f"n_core={n_core} + n_active={n_active} does not fit the {c.shape[1]} orbitals"
        )
    ao = AoIntegrals(mol, basis_set)
    hcore = np.asarray(ao.hcore())
    ecore = float(ao.nuclear_repulsion())
    h_eff = hcore
    if n_core:
        cc = c[:, :n_core]
        d_core = 2.0 * cc @ cc.T
        e_veff, v = ao.veff(d_core)
        ecore += float(np.sum(d_core * hcore)) + float(e_veff)
        h_eff = hcore + np.asarray(v)
    ca = c[:, n_core : n_core + n_active]
    h = ca.T @ h_eff @ ca
    eri = np.asarray(ao.ao2mo(ca, ca, ca, ca))
    return h, eri, ecore


def _unique_pairs(n: int):
    return [(p, q) for p in range(n) for q in range(p + 1)]


def to_string(h, eri, ecore: float, nelec: int, ms2: int) -> str:
    """The FCIDUMP text of integrals ``h``, ``eri``, ``ecore`` (see `integrals`).

    `nelec` is the active electron count, `ms2` is ``n_alpha - n_beta``.
    Only the 8-fold-unique element of each integral class is written (the
    ``p>=q, r>=s, pq>=rs`` one); the rest of the tensor is not read.
    """
    h = np.asarray(h, dtype=float)
    eri = np.asarray(eri, dtype=float)
    n = h.shape[0]
    if h.shape != (n, n) or eri.shape != (n,) * 4:
        raise ValueError(
            f"h {h.shape} and eri {eri.shape} do not describe {n} orbitals"
        )
    if nelec < 0 or nelec > 2 * n or (nelec - ms2) % 2 or abs(ms2) > nelec:
        raise ValueError(
            f"nelec={nelec}, ms2={ms2} is not a spin sector of {n} orbitals"
        )
    out = [
        f" &FCI NORB={n:4d},NELEC={nelec:4d},MS2={ms2:4d},",
        "  ORBSYM=" + ",".join("1" for _ in range(n)) + ",",
        "  ISYM=1,",
        " &END",
    ]
    pairs = _unique_pairs(n)
    for ij, (i, j) in enumerate(pairs):
        for kl, (k, ll) in enumerate(pairs[: ij + 1]):
            out.append(f" {float(eri[i, j, k, ll])!r} {i + 1} {j + 1} {k + 1} {ll + 1}")
    for i, j in pairs:
        out.append(f" {float(h[i, j])!r} {i + 1} {j + 1} 0 0")
    out.append(f" {float(ecore)!r} 0 0 0 0")
    return "\n".join(out) + "\n"


def write(
    path, mol, basis_set, mo_coeff, n_core: int, n_active: int, nelec: int, ms2: int
) -> None:
    """Write the active-space FCIDUMP of `mo_coeff` to `path`.

    `nelec` is the number of ACTIVE electrons and `ms2` is
    ``n_alpha - n_beta`` of the target sector (0 for a singlet). Pass
    `CasScfResult.mo_coeff` and its `n_core` for the CASSCF orbitals, or the
    canonical RHF orbitals for the CAS-CI ones.
    """
    h, eri, ecore = integrals(mol, basis_set, mo_coeff, n_core, n_active)
    Path(path).write_text(to_string(h, eri, ecore, nelec, ms2))


def read(path) -> dict:
    """Read an FCIDUMP: ``{"norb", "nelec", "ms2", "h", "eri", "ecore"}``.

    `h` is (n, n) and `eri` is (n,)*4 chemist ``(pq|rs)``, both expanded
    from the unique elements by the 8-fold symmetry. Entries a file does not
    list are zero.
    """
    text = Path(path).read_text()
    lines = text.splitlines()
    end = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.strip().upper() in ("&END", "/")
            or ln.strip().upper().endswith("&END")
        ),
        None,
    )
    if end is None:
        raise ValueError("FCIDUMP header has no &END")
    header = " ".join(lines[: end + 1]).upper().replace("&FCI", " ")
    fields = {}
    for key in ("NORB", "NELEC", "MS2"):
        pos = header.find(key + "=")
        if pos < 0:
            if key == "MS2":
                fields[key] = 0
                continue
            raise ValueError(f"FCIDUMP header has no {key}")
        tail = header[pos + len(key) + 1 :].lstrip()
        digits = ""
        for ch in tail:
            if ch in "+-0123456789":
                digits += ch
            else:
                break
        fields[key] = int(digits)
    n = fields["NORB"]
    h = np.zeros((n, n))
    eri = np.zeros((n,) * 4)
    ecore = 0.0
    for ln in lines[end + 1 :]:
        parts = ln.split()
        if not parts:
            continue
        v = float(parts[0].replace("D", "E").replace("d", "e"))
        i, j, k, ll = (int(x) for x in parts[1:5])
        if i == 0:
            ecore = v
        elif k == 0:
            h[i - 1, j - 1] = h[j - 1, i - 1] = v
        else:
            i, j, k, ll = i - 1, j - 1, k - 1, ll - 1
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, ll), (ll, k)):
                    eri[a, b, c, d] = eri[c, d, a, b] = v
    return {
        "norb": n,
        "nelec": fields["NELEC"],
        "ms2": fields["MS2"],
        "h": h,
        "eri": eri,
        "ecore": ecore,
    }
