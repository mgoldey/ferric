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

# First entry of the job dialogs' lists (data/job_options.toml): run with the
# method or basis saved by Configure... .
CONFIGURED_DEFAULT = "(Configured default)"
JOB_METHODS = [CONFIGURED_DEFAULT, *METHODS]
JOB_BASES = [CONFIGURED_DEFAULT, *BASES]

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


def _pick(values: list, value):
    """A stringList reply may be the index or the string; return the string."""
    if isinstance(value, int) and not isinstance(value, bool):
        return values[value]
    if str(value).lower() not in [v.lower() for v in values]:
        raise ValueError(f"unknown value {value!r}; expected one of {values}")
    return value


def job_settings(cfg: dict, options: dict) -> dict:
    """Method/basis for one run: the dialog's choice, or the stored default when
    the choice is CONFIGURED_DEFAULT or the option is absent."""

    def chosen(key: str, values: list) -> str:
        if key not in options:
            return cfg[key]
        value = _pick(values, options[key])
        return cfg[key] if value == CONFIGURED_DEFAULT else value

    return {
        "method": chosen("method", JOB_METHODS),
        "basis": chosen("basis", JOB_BASES),
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
