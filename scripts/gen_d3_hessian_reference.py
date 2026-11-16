"""Generate crates/ferric-d3/testdata/d3bj_hessian_reference.json.

Reference: central differences of the simple-dftd3 (`dftd3` Python package)
analytic D3(BJ) gradient, an independent Fortran implementation, two-body
only (s9 = 0), with (s6, s8, a1, a2) passed explicitly. Each system stores
the FD Hessian at step h and at h/2 (Bohr), so the consumer can derive the
truncation bar from the data: `H_fd(h/2)` misses the exact Hessian by
`|H_fd(h) - H_fd(h/2)|/3` to leading order.

Consumed by crates/ferric-d3/tests/hessian_vs_dftd3.rs.

Run (needs the dftd3 package):
    uv run --no-sync python scripts/gen_d3_hessian_reference.py
"""

import json
import pathlib

import dftd3
import numpy as np
from dftd3.interface import DispersionModel, RationalDampingParam

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "crates/ferric-d3/testdata/d3bj_hessian_reference.json"
B3LYP = (1.0, 1.9889, 0.3981, 4.4211)
STEP = 2e-3

# Water near equilibrium, a C/H cluster with coordination numbers ~0.5 to ~2
# and a mixed C/F/O/H/N fragment (tests/hessian_vs_fd.rs), plus CHBr3 for a
# heavy element.
SYSTEMS = {
    "water": (
        [8, 1, 1],
        [[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]],
    ),
    "branched": (
        [6, 6, 6, 6, 1, 1, 1, 1],
        [
            [0.0, 0.0, 0.0],
            [2.9, 0.0, 0.0],
            [-1.45, 2.51, 0.0],
            [-1.45, -2.51, 0.0],
            [0.0, 0.0, 2.07],
            [3.9, 1.7, 0.0],
            [-2.45, 2.9, 1.7],
            [-2.45, -2.9, -1.7],
        ],
    ),
    "mixed_cf2onh": (
        [6, 9, 9, 8, 1, 7],
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.3, 0.1],
            [-1.2, 2.2, -0.3],
            [-1.0, -1.1, 2.3],
            [-0.5, -1.9, -1.6],
            [1.1, -2.6, 3.4],
        ],
    ),
    "chbr3": (
        [6, 1, 35, 35, 35],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.06],
            [3.43, 0.0, -1.21],
            [-1.715, 2.97, -1.21],
            [-1.715, -2.97, -1.21],
        ],
    ),
}


def gradient(numbers, coords):
    s6, s8, a1, a2 = B3LYP
    model = DispersionModel(np.array(numbers), np.array(coords, dtype=float))
    p = RationalDampingParam(s6=s6, s8=s8, s9=0.0, a1=a1, a2=a2, alp=14.0)
    return np.asarray(model.get_dispersion(p, grad=True)["gradient"]).ravel()


def fd_hessian(numbers, coords, h):
    x0 = np.array(coords, dtype=float)
    n3 = x0.size
    out = np.zeros((n3, n3))
    for b in range(n3):
        xp, xm = x0.copy(), x0.copy()
        xp[b // 3, b % 3] += h
        xm[b // 3, b % 3] -= h
        out[:, b] = (gradient(numbers, xp) - gradient(numbers, xm)) / (2.0 * h)
    return out


def main():
    out = {
        "source": f"simple-dftd3 (dftd3 python package) {dftd3.__version__}",
        "generator": "scripts/gen_d3_hessian_reference.py",
        "note": "Hartree/Bohr^2, row/col = 3*atom + xyz. Central differences of "
        "the dftd3 analytic gradient at step h and h/2 (Bohr). "
        "RationalDampingParam(s6,s8,a1,a2,s9=0,alp=14) passed explicitly.",
        "params": dict(zip(("s6", "s8", "a1", "a2"), B3LYP)),
        "step_bohr": STEP,
        "systems": {},
    }
    for name, (numbers, coords) in SYSTEMS.items():
        h1 = fd_hessian(numbers, coords, STEP)
        h2 = fd_hessian(numbers, coords, 0.5 * STEP)
        print(
            f"{name:14s} max|H| {np.abs(h2).max():.3e} |H(h)-H(h/2)| {np.abs(h1 - h2).max():.3e}"
        )
        out["systems"][name] = {
            "numbers": numbers,
            "coords_bohr": coords,
            "hessian_fd_h": h1.tolist(),
            "hessian_fd_h2": h2.tolist(),
        }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
