"""Private helpers shared by the pure-Python modules (not public API)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


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
    numbers of the real (non-ghost) atoms minus ``charge``. Atomic numbers come
    from ferric's own element table (an all-ghost probe molecule has no
    electrons, so it always builds); an unknown symbol raises ``ValueError``."""
    from . import Molecule

    n = len(symbols)
    if n == 0:
        return 1
    if ghost is not None and len(ghost) != n:
        raise ValueError(f"ghost has {len(ghost)} flags for {n} atoms")
    probe = Molecule.from_arrays(list(symbols), np.zeros((n, 3)), ghost=[True] * n)
    zs = probe.atomic_numbers()
    total = sum(int(z) for i, z in enumerate(zs) if ghost is None or not ghost[i])
    return 1 if (total - int(charge)) % 2 == 0 else 2
