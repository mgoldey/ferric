"""On-disk cache of closed-shell SCF results, shared across plugin invocations.

Avogadro gives the electrostatic models a hard 10 s (``PythonScript::execute``
in avogadrolibs 2.1.0, used for ``--charges`` and ``--potentials``) and a plugin
cannot change it. Menu commands have no limit. So Single Point stores what it
computed, and the timed calls reuse it:

* charges: Single Point computes every charge model Avogadro lists for ferric
  (``ELECTROSTATIC_MODELS``) and stores them; a ``--charges`` call for the same
  molecule and settings returns the stored values without an SCF.
* potentials: the points are not known in advance, so the converged AO density
  is stored instead and the ``--potentials`` call starts its SCF from it
  (ferric's ``dm0``; a converged start reconverges in <= 2 iterations).

An entry is keyed by everything the result depends on: atomic numbers, the
exact coordinates (``repr`` of each float, so two geometries share an entry only
when Avogadro sent identical numbers), charge, multiplicity, method, basis and
the ferric build. A missing, unreadable or mismatched entry is a miss; the cache
never changes a result, only whether it is recomputed.

Location: ``$AVOGADRO_FERRIC_CACHE`` if set, else
``$XDG_CACHE_HOME/avogadro-ferric`` (``~/.cache/...`` when unset). Writes are
atomic (temp file + rename) and the directory keeps the ``MAX_ENTRIES`` most
recently written entries.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from .engine import Job

MAX_ENTRIES = 32
# The charge models Avogadro lists for ferric (pyproject.toml [tool.avogadro]
# electrostatic entries; __init__.CHARGE_MODELS maps identifiers to these).
ELECTROSTATIC_MODELS = ("lowdin", "hirshfeld", "chelpg")


def cache_dir() -> Path:
    env = os.environ.get("AVOGADRO_FERRIC_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "avogadro-ferric"


def _ferric_identity() -> dict:
    """What identifies the ferric build: version, baked-in commit/dirty flag and
    the extension module's size and mtime (a rebuilt dirty tree keeps its
    commit, so the file identity is the guard there)."""
    from . import engine

    f = engine.ferric()
    ident: dict = {"version": getattr(f, "__version__", None)}
    build = getattr(f, "__build__", None)
    if isinstance(build, dict):
        ident["build"] = {k: build[k] for k in sorted(build)}
    ext = getattr(getattr(f, "ferric", None), "__file__", None) or getattr(
        f, "__file__", None
    )
    if ext:
        try:
            st = os.stat(ext)
            ident["module"] = [st.st_size, st.st_mtime_ns]
        except OSError:
            pass
    return ident


def key(job: Job) -> str:
    payload = {
        "numbers": [int(z) for z in job.numbers],
        "coords": [repr(float(c)) for c in np.asarray(job.coords_angstrom).ravel()],
        "charge": int(job.charge),
        "multiplicity": int(job.multiplicity),
        "method": job.method.strip().upper(),
        "basis": job.basis.strip().lower(),
        "ferric": _ferric_identity(),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _path(k: str) -> Path:
    return cache_dir() / f"{k}.npz"


def load(job: Job) -> dict | None:
    """{"energy", "density", "charges": {model: [q]}} or None on any miss."""
    path = _path(key(job))
    try:
        with np.load(path, allow_pickle=False) as data:
            entry = {
                "energy": float(data["energy"]),
                "density": np.array(data["density"], dtype=float),
                "charges": json.loads(str(data["charges"])),
            }
    except FileNotFoundError:
        return None
    except Exception as exc:  # corrupt or foreign file: recompute
        print(f"ferric cache: ignoring unreadable {path.name}: {exc}", file=sys.stderr)
        return None
    n = len(job.numbers)
    if entry["density"].ndim != 2 or any(
        len(q) != n for q in entry["charges"].values()
    ):
        return None
    return entry


def store(job: Job, energy: float, density, charges: dict) -> None:
    """Merge into this job's entry (charges of other models are kept) and prune.

    A failure to write is reported on stderr and otherwise ignored: the cache
    is an optimisation, never a reason for a command to fail.
    """
    try:
        old = load(job)
        merged = dict(old["charges"]) if old else {}
        merged.update({m: [float(x) for x in q] for m, q in charges.items()})
        directory = cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(
                    fh,
                    energy=np.float64(energy),
                    density=np.asarray(density, dtype=float),
                    charges=np.array(json.dumps(merged)),
                )
            os.replace(tmp, _path(key(job)))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        _prune(directory)
    except Exception as exc:
        print(f"ferric cache: not stored: {exc}", file=sys.stderr)


def _prune(directory: Path) -> None:
    entries = sorted(
        directory.glob("*.npz"), key=lambda p: p.stat().st_mtime_ns, reverse=True
    )
    for stale in entries[MAX_ENTRIES:]:
        stale.unlink(missing_ok=True)
