"""Persistent plugin defaults (Extensions > ferric > Configure...).

Stored as JSON in ``$AVOGADRO_FERRIC_CONFIG`` if set, else
``$XDG_CONFIG_HOME/avogadro-ferric/config.json`` (``~/.config/...`` when
XDG_CONFIG_HOME is unset). Inside the Avogadro flatpak XDG_CONFIG_HOME is
``~/.var/app/org.openchemistry.Avogadro2/config``.

Thread and memory settings are applied as environment variables BEFORE ferric
is imported (``apply_runtime_env``): ferric reads RAYON_NUM_THREADS and
FERRIC_MEM_BUDGET_GB when its thread pool / budgets are first built.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

METHODS = ["HF", "LDA", "PBE", "BLYP", "BP86", "PBE0", "B3LYP", "r2SCAN"]

BASES = [
    "sto-3g",
    "6-31g",
    "6-31g*",
    "6-31g**",
    "def2-svp",
    "def2-tzvp",
    "cc-pvdz",
    "cc-pvtz",
    "aug-cc-pvdz",
]

CHARGE_MODELS = ["lowdin", "hirshfeld", "chelpg", "mulliken", "resp"]

DEFAULTS: dict = {
    "method": "PBE",
    "basis": "def2-svp",
    "threads": 0,  # 0 = let ferric/rayon decide (all cores)
    "memory_gb": 0.0,  # 0 = ferric's auto-detected budget
    "charge_model": "lowdin",
}


def config_path() -> Path:
    env = os.environ.get("AVOGADRO_FERRIC_CONFIG")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "avogadro-ferric" / "config.json"


def load() -> dict:
    cfg = dict(DEFAULTS)
    path = config_path()
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = {}
        for key in DEFAULTS:
            if key in stored:
                cfg[key] = stored[key]
    return cfg


def save(cfg: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: cfg[k] for k in DEFAULTS if k in cfg}
    path.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
    return path


def apply_runtime_env(cfg: dict) -> None:
    """Export thread/memory settings for ferric. Must run before ``import ferric``."""
    threads = int(cfg.get("threads") or 0)
    if threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    mem = float(cfg.get("memory_gb") or 0.0)
    if mem > 0:
        os.environ["FERRIC_MEM_BUDGET_GB"] = str(mem)


def user_options(cfg: dict) -> dict:
    """The Configure... dialog, pre-filled with the stored values."""

    def index_or_zero(values: list, value) -> int:
        lowered = [str(v).lower() for v in values]
        try:
            return lowered.index(str(value).lower())
        except ValueError:
            return 0

    return {
        "method": {
            "type": "stringList",
            "label": "Method (HF or functional)",
            "values": METHODS,
            "default": index_or_zero(METHODS, cfg["method"]),
            "order": 1.0,
        },
        "basis": {
            "type": "stringList",
            "label": "Basis set",
            "values": BASES,
            "default": index_or_zero(BASES, cfg["basis"]),
            "order": 2.0,
        },
        "charge_model": {
            "type": "stringList",
            "label": "Charges shown by Single Point",
            "values": CHARGE_MODELS,
            "default": index_or_zero(CHARGE_MODELS, cfg["charge_model"]),
            "order": 3.0,
        },
        "threads": {
            "type": "integer",
            "label": "Threads (0 = all cores)",
            "minimum": 0,
            "maximum": 1024,
            "default": int(cfg["threads"]),
            "order": 4.0,
        },
        "memory_gb": {
            "type": "float",
            "label": "Memory budget, GB (0 = auto)",
            "minimum": 0.0,
            "maximum": 4096.0,
            "default": float(cfg["memory_gb"]),
            "order": 5.0,
        },
    }


def job_options(cfg: dict) -> dict:
    """The dialog shown before Single Point / Optimize / Frequencies / Orbitals:
    method and basis, pre-filled from the Configure... defaults. A choice here
    applies to this run only. Charge and multiplicity are not asked: Avogadro
    sends the molecule's own with every command."""
    full = user_options(cfg)
    return {"method": full["method"], "basis": full["basis"]}


def _pick(values: list, value):
    """A stringList reply may be the index or the string; return the string."""
    if isinstance(value, int) and not isinstance(value, bool):
        return values[value]
    if str(value).lower() not in [v.lower() for v in values]:
        raise ValueError(f"unknown value {value!r}; expected one of {values}")
    return value


def job_settings(cfg: dict, options: dict) -> dict:
    """Method/basis for one run: the dialog's choice, else the stored default."""
    return {
        "method": _pick(METHODS, options["method"])
        if "method" in options
        else cfg["method"],
        "basis": _pick(BASES, options["basis"]) if "basis" in options else cfg["basis"],
    }


def update_from_options(options: dict) -> dict:
    """Merge the dialog's reply into the stored config and save it."""
    cfg = load()
    for key, values in (
        ("method", METHODS),
        ("basis", BASES),
        ("charge_model", CHARGE_MODELS),
    ):
        if key in options:
            value = options[key]
            if isinstance(value, int) and not isinstance(value, bool):
                value = values[value]
            if str(value).lower() not in [v.lower() for v in values]:
                raise ValueError(f"unknown {key} {value!r}; expected one of {values}")
            cfg[key] = value
    if "threads" in options:
        cfg["threads"] = max(0, int(options["threads"]))
    if "memory_gb" in options:
        cfg["memory_gb"] = max(0.0, float(options["memory_gb"]))
    save(cfg)
    return cfg
