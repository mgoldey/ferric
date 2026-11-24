"""The energy-model server Avogadro's force-field / AutoOptimization tools drive.

Protocol (avogadrolibs 2.1.0, qtplugins/forcefield/scriptenergy.cpp, text mode):

1. Avogadro starts ``avogadro-ferric ferric --lang <locale>`` and writes ONE line
   of JSON on stdin: ``{"charge", "spin", "selectedAtoms", "cjson", "options"}``.
   No reply is expected.
2. Each request is N lines ``x y z`` (Angstrom). The SAME request is sent for an
   energy and for a gradient (``evaluate()`` sends gradient then value), so every
   reply carries both::

       AvogadroEnergy: <E in kJ/mol>
       AvogadroGradient:
       <dE/dx> <dE/dy> <dE/dz>      (kJ/mol/Angstrom, one line per atom)

   The last geometry's result is cached, so the second request of a pair costs
   nothing.
3. Avogadro waits for a reply with ``waitForReadyRead()``, i.e. 30 s, and then
   reads only the complete lines available, so the reply goes out in one write.
   A reply that arrives after Avogadro gave up would be read as the answer to
   the NEXT request and desynchronise every step after it; a calculation that
   overruns ``REPLY_DEADLINE_S`` is therefore not answered at all (Avogadro
   sees no energy for that step) and a message goes to stderr.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from . import avo, engine, units

REPLY_DEADLINE_S = 29.0


def _num(x) -> str:
    """Round-trip float text. Not repr(): numpy 2 reprs np.float64 as
    "np.float64(...)", which Avogadro's parser cannot read."""
    return format(float(x), ".17g")


def format_reply(energy_kj: float, grad_kj_per_a: np.ndarray) -> str:
    lines = [f"AvogadroEnergy: {_num(energy_kj)}", "AvogadroGradient:"]
    for gx, gy, gz in np.asarray(grad_kj_per_a, dtype=float).reshape(-1, 3):
        lines.append(f"{_num(gx)} {_num(gy)} {_num(gz)}")
    return "\n".join(lines) + "\n"


def failure_reply(natoms: int) -> str:
    """NaN energy: Avogadro reads it as a failed evaluation (non-finite gradient
    components are zeroed by EnergyCalculator::cleanGradients)."""
    return "AvogadroEnergy: nan\nAvogadroGradient:\n" + "nan nan nan\n" * natoms


def to_avogadro_units(energy_ha: float, grad_ha_per_bohr: np.ndarray):
    """(kJ/mol, kJ/mol/Angstrom) from (Hartree, Hartree/Bohr)."""
    e = energy_ha * units.HARTREE_TO_KJ_PER_MOL
    g = (
        np.asarray(grad_ha_per_bohr, dtype=float)
        * units.HARTREE_PER_BOHR_TO_KJ_PER_MOL_ANGSTROM
    )
    return e, g


def read_coordinates(stream, natoms: int) -> np.ndarray | None:
    rows = []
    while len(rows) < natoms:
        line = stream.readline()
        if not line:
            return None  # Avogadro closed stdin / killed us
        line = line.strip()
        if not line:
            continue
        rows.append([float(v) for v in line.split()[:3]])
    return np.asarray(rows, dtype=float)


def serve(bootstrap: dict, stream=None, compute=None) -> None:
    stream = sys.stdin if stream is None else stream
    job = avo.job_from_input(bootstrap)
    compute = engine.energy_gradient if compute is None else compute
    engine.ferric()  # import now, not inside the first timed request
    natoms = len(job.numbers)
    print(
        f"[avogadro-ferric] energy model {job.label}, {natoms} atoms", file=sys.stderr
    )

    last_coords: np.ndarray | None = None
    last_reply = ""
    while True:
        coords = read_coordinates(stream, natoms)
        if coords is None:
            return
        if last_coords is not None and np.array_equal(coords, last_coords):
            avo.write_raw(last_reply)
            continue
        start = time.monotonic()
        try:
            e_ha, g_ha = compute(job, coords)
            e, g = to_avogadro_units(e_ha, g_ha)
            text = format_reply(e, g)
        except Exception as exc:  # report, keep serving
            print(f"[avogadro-ferric] evaluation failed: {exc}", file=sys.stderr)
            text = failure_reply(natoms)
        elapsed = time.monotonic() - start
        last_coords, last_reply = coords, text
        if elapsed > REPLY_DEADLINE_S:
            print(
                f"[avogadro-ferric] {job.label} took {elapsed:.1f} s, past Avogadro's "
                "30 s reply window; not replying (choose a smaller basis or method)",
                file=sys.stderr,
            )
            continue
        avo.write_raw(text)
