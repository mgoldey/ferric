"""Arm A, first half: the FREE-solution conformer scan for danuglipron.

Relaxes all 20 committed conformers in vacuum with GFN2-xTB and identifies the
global free minimum, which is the reference state every strain number is
measured against. Writes JSON to experiments/danuglipron/out/.

Run:
    LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu:$HOME/.local/lib \
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python experiments/danuglipron/run_arm_a_free.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.campaign.strain import free_reference, load_xyz_ensemble  # noqa: E402
from tools.campaign.xtb_engine import HARTREE_TO_KCAL_MOL, verify_xtb_build  # noqa: E402

ENSEMBLE = "testdata/molecules/c9_systems/danuglipron"
OUT = Path("experiments/danuglipron/out/arm_a_free.json")


def main() -> int:
    ok, err = verify_xtb_build()
    if not ok:
        print(f"ABORT: xtb build check failed -- {err}", flush=True)
        return 1
    print("xtb build check: PASS (gradients verified against GFN2 water)", flush=True)

    ens = load_xyz_ensemble(ENSEMBLE)
    print(
        f"ensemble: {len(ens)} conformers, formula {ens.formula}, "
        f"shared atom order: {ens.shared_order}",
        flush=True,
    )

    t0 = time.time()
    ref = free_reference(
        ens.symbols_per_conformer, ens.conformers, labels=ens.labels
    )
    dt = time.time() - t0

    print(f"\nrelaxed {ref.n_converged}/{ref.n_considered} in {dt:.1f}s", flush=True)
    if ref.e_min is None:
        print("ABORT: no conformer converged; strain is undefined", flush=True)
        return 1

    print(f"free minimum: {ref.label}  E = {ref.e_min:.8f} Ha")
    spread = ref.spread_kcal
    print(f"ensemble spread: {spread:.2f} kcal/mol" if spread else "spread: n/a")

    print(f"\n{'conformer':22s} {'E_relaxed (Ha)':>16s} {'rel (kcal/mol)':>15s} {'conv':>5s}")
    rows = []
    for c in sorted(ref.per_conformer, key=lambda c: (c.e_relaxed is None, c.e_relaxed)):
        if c.ok:
            rel = (c.e_relaxed - ref.e_min) * HARTREE_TO_KCAL_MOL
            print(f"{c.label:22s} {c.e_relaxed:16.8f} {rel:15.2f} {str(c.converged):>5s}")
        else:
            rel = None
            print(f"{c.label:22s} {'FAILED':>16s} {'--':>15s} {'--':>5s}  {c.error}")
        rows.append({
            "label": c.label,
            "e_singlepoint_ha": c.e_singlepoint,
            "e_relaxed_ha": c.e_relaxed,
            "rel_kcal_mol": rel,
            "converged": c.converged,
            "error": c.error,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "method": ref.method,
        "ensemble_dir": ENSEMBLE,
        "formula": ens.formula,
        "shared_atom_order": ens.shared_order,
        "n_considered": ref.n_considered,
        "n_converged": ref.n_converged,
        "free_min_label": ref.label,
        "free_min_energy_ha": ref.e_min,
        "spread_kcal_mol": spread,
        "wall_seconds": dt,
        "conformers": rows,
        "relaxed_coords": {
            c.label: c.relaxed_coords for c in ref.per_conformer if c.ok
        },
        "symbols_per_conformer": {
            lbl: syms for lbl, syms in zip(ens.labels, ens.symbols_per_conformer)
        },
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
