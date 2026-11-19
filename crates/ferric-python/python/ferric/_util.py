"""Private helpers shared by the pure-Python modules (not public API)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
    "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs "
    "Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl "
    "Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh "
    "Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_Z = {s.lower(): i for i, s in enumerate(_SYMBOLS, 1)}


def _plain(obj: Any) -> Any:
    """``obj`` with dict keys as ``str``, tuples as lists and numpy values as
    Python values (JSON-ready)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def default_multiplicity(
    symbols: Sequence[Any], charge: int, ghost: Sequence[bool] | None = None
) -> int:
    """1 for an even electron count, else 2. The count is the sum of the atomic
    numbers of the real (non-ghost) atoms minus ``charge``. ``ValueError`` for an
    unknown element symbol."""
    total = 0
    for i, sym in enumerate(symbols):
        if ghost is not None and ghost[i]:
            continue
        if isinstance(sym, (int, np.integer)):
            total += int(sym)
            continue
        z = _Z.get(str(sym).strip().lower())
        if z is None:
            raise ValueError(f"unknown element {sym!r} at atom {i}")
        total += z
    return 1 if (total - int(charge)) % 2 == 0 else 2
