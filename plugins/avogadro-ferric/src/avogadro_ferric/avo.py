"""Avogadro-side plumbing: clean stdout, cjson in/out, progress messages."""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

from . import config
from .engine import Job

_PROTOCOL_FD: int | None = None


def claim_stdout() -> None:
    """Reserve the real stdout for Avogadro's protocol and send everything else to stderr.

    ferric's Rust code prints progress with println!, which writes to file
    descriptor 1 directly and would corrupt the JSON / energy replies Avogadro
    parses. So fd 1 is duplicated for our own replies and then pointed at
    stderr for the rest of the process (Python prints included).
    """
    global _PROTOCOL_FD
    if _PROTOCOL_FD is not None:
        return
    sys.stdout.flush()
    _PROTOCOL_FD = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr


def write_raw(text: str) -> None:
    """One os.write of the whole reply (Avogadro's text energy protocol reads
    only the complete lines available when its wait returns)."""
    data = text.encode("utf-8")
    fd = _PROTOCOL_FD if _PROTOCOL_FD is not None else 1
    while data:
        n = os.write(fd, data)
        data = data[n:]


def reply(obj) -> None:
    write_raw(json.dumps(obj) + "\n")


def progress(
    message: str, value: int | None = None, maximum: int | None = None
) -> None:
    """Progress line for menu commands (Avogadro >= 2.1 sets AVO_PROGRESS_PROTOCOL)."""
    if not os.environ.get("AVO_PROGRESS_PROTOCOL"):
        return
    payload: dict = {"message": message}
    if value is not None and maximum is not None:
        payload["value"] = value
        payload["maximum"] = maximum
    write_raw(json.dumps({"avogadro": payload}) + "\n")


def job_from_input(avo_input: dict, cfg: dict | None = None) -> Job:
    """Build a Job from Avogadro's input JSON (cjson + charge/spin) and the config."""
    cfg = config.load() if cfg is None else cfg
    cjson = avo_input.get("cjson")
    if isinstance(cjson, str):
        cjson = json.loads(cjson)
    if not isinstance(cjson, dict):
        raise ValueError("Avogadro did not send the molecule as cjson")
    numbers = [int(z) for z in cjson["atoms"]["elements"]["number"]]
    if not numbers:
        raise ValueError("the molecule has no atoms")
    coords = np.asarray(cjson["atoms"]["coords"]["3d"], dtype=float).reshape(-1, 3)
    props = cjson.get("properties", {}) or {}
    charge = avo_input.get("charge", props.get("totalCharge", 0))
    mult = avo_input.get("spin", props.get("totalSpinMultiplicity", 1))
    chosen = config.job_settings(cfg, avo_input.get("options") or {})
    job = Job(
        numbers=numbers,
        coords_angstrom=coords,
        charge=int(charge or 0),
        multiplicity=int(mult or 1),
        method=str(chosen["method"]),
        basis=str(chosen["basis"]),
    )
    job.check()
    return job


def cjson_of(avo_input: dict) -> dict:
    cjson = avo_input.get("cjson")
    if isinstance(cjson, str):
        cjson = json.loads(cjson)
    return copy.deepcopy(cjson)


def strip_stale(cjson: dict) -> dict:
    """Drop data that no longer describes the molecule after a geometry change."""
    for key in ("vibrations", "basisSet", "orbitals", "cube"):
        cjson.pop(key, None)
    atoms = cjson.get("atoms", {})
    atoms.pop("partialCharges", None)
    cjson.pop("partialCharges", None)
    return cjson
