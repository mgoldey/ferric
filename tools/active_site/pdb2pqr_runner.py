"""Subprocess wrapper around the PDB2PQR CLI (pdb2pqr30)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

PDB2PQR_INSTALL_HINT = (
    "pdb2pqr30 not found on PATH. Install it with `pip install pdb2pqr` "
    "(tested against pdb2pqr 3.7.1)."
)


def find_pdb2pqr() -> str:
    """Locate the pdb2pqr30 executable, raising a clear error if absent."""
    exe = shutil.which("pdb2pqr30")
    if exe is None:
        raise RuntimeError(PDB2PQR_INSTALL_HINT)
    return exe


def run_pdb2pqr(pdb_path: str | Path, pqr_path: str | Path, ff: str = "AMBER") -> Path:
    """Run PDB2PQR on `pdb_path`, writing a PQR file to `pqr_path`.

    Returns the path to the written PQR file. Raises RuntimeError with the
    captured stderr on failure.
    """
    exe = find_pdb2pqr()
    pdb_path = Path(pdb_path)
    pqr_path = Path(pqr_path)
    result = subprocess.run(
        [exe, f"--ff={ff}", str(pdb_path), str(pqr_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not pqr_path.exists():
        raise RuntimeError(
            f"pdb2pqr30 failed (exit {result.returncode}) on {pdb_path}:\n"
            f"{result.stderr}"
        )
    return pqr_path
