"""NWChem references for the VALIDATION.md "cDFT" row (validation tier W1).

Consumer: crates/ferric-scf/tests/validation_cdft.rs.
Output:   testdata/reference/validation/cdft/<system>_<basis>.json
Inputs:   scripts/validation/nwchem/cdft/<system>_<basis>_<run>_<grid>.nw
          (written by this script; the committed files are exactly what ran)

Systems (geometries in testdata/molecules/validation/):

    lih   LiH   neutral singlet   fragment = Li   charge targets 2.60 2.75 2.95
    hf    HF    neutral singlet   fragment = F    charge targets 8.80 8.90 9.10
    h2o   H2O+  cation doublet    fragment = O    charge targets 7.30 7.40 7.60
                                                  + spin target (N_a - N_b) 0.90

x bases 6-31G and def2-SVP from FERRIC's bundled JSON (common.ferric_shells;
def2-SVP d shells are spherical, 6-31G has none on these atoms). The natural
(unconstrained) Becke populations are Li 2.83, F 9.00-9.03, O 7.49-7.51 and
O spin 0.97-0.98, so every system has targets on both sides of its natural
value.

LIKE-FOR-LIKE (what was matched, from the NWChem 7.2.2 source):

* Population operator. `cdft ... pop becke` (NWChem's DEFAULT is `lowdin`,
  so `pop becke` is mandatory). It builds W_C = sum over grid points whose
  HOME atom is in C of (quadrature weight x Becke partition weight) chi chi
  (`frag_dens`/`acc_sa` in src/nwdft/scf_dft/cdft_util.F). ferric builds
  W_C = sum over ALL points of w_g w_C(r_g) chi chi. Both are quadratures of
  the SAME integral  int w_C(r) chi chi  -- they differ only in quadrature
  error, which is why the converged-grid reference exists (below).
* Becke partition. NWChem's partition function is whatever `grid` selects;
  `grid ... becke` gives Becke 1988 with the atomic-size adjustment
  a_AB = u/(u^2-1), u = (chi-1)/(chi+1), chi = R_A/R_B, clamped to +-1/2
  (grid_setspac_params.F, grid_beckew.F) -- the same formula as ferric's
  `becke_weights_all`, with the same Bragg-Slater radii for H/Li/O/F
  (0.35/1.45/0.60/0.50 A). NWChem additionally zeroes a cell function when
  |mu| > 0.95 (a ~1e-8-relative truncation; measured effect on a LiH
  population 1.6e-9).
* Constraint convention. NWChem converts `charge q` to an electron target
  N = sum(Z_frag) - q (cdft_init) and adds +lambda W to the Fock
  (dft_scf.F), residual Tr[W D] - N: the SAME sign convention as ferric's
  `solve_cdft_uhf` (F += lambda W, c = N_C - target), so lambda compares
  directly and dE/dN_target = -lambda in both. Spin constraints add +lambda W
  to alpha and -lambda W to beta in both codes.
* Pure Hartree-Fock cannot be used: NWChem builds the XC grid (and hence
  the Becke W) only when a DFT functional is present, so `xc hfexch` + cdft
  leaves W = 0 and aborts ("multipliers go over limit"). PBE is used:
  NWChem `xpbe96 cpbe96` == libxc GGA_X_PBE + GGA_C_PBE.
* Grid. `grid lebedev <nrad> <iang> becke treutler` with pruning OFF
  (`set dft:no_prune T`). NWChem's Treutler nodes are the Perez-Jorda
  Gauss-Chebyshev variant, not ferric's Chebyshev-2 nodes, so the point sets
  are NOT identical; at 99/302 the UNCONSTRAINED PBE energies still agree to
  ~1e-10 (both radially converged), but the fragment populations carry a
  ~1e-6 quadrature difference. Two NWChem grids are therefore recorded:
    matched   : 99 radial x 302 angular (ferric's cDFT default grid)
    converged : 300 radial x 974 angular (NWChem's grid limit; ferric's
                Lebedev table stops at 302)
* Integrals: `direct`, `tolerances tight`; no density fitting.
* SCF: odft (UKS) for every system; `convergence energy 1e-11 density 1e-9`;
  cdft multiplier convergence 1e-10.

Run (light step; ~1-2 min total):
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_cdft.py [system ...]
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "cdft"
ROW_NAME = "cDFT"
BASES = ("6-31g", "def2-svp")
NWCHEM = "/usr/bin/nwchem"
INPUT_DIR = common.ROOT / "scripts" / "validation" / "nwchem" / "cdft"
SCRATCH_ROOT = common.ROOT / "target" / "nwchem-cdft"
XC = "xpbe96 cpbe96"
FUNCTIONAL = "PBE"
# name -> (radial points, NWChem Lebedev index, angular points)
GRIDS = {
    "matched": (99, 11, 302),
    "converged": (300, 16, 974),
}
# system -> (charge, multiplicity, fragment atoms 0-based, [(kind, target)])
SYSTEMS = {
    "lih": (0, 1, [0], [("charge", 2.60), ("charge", 2.75), ("charge", 2.95)]),
    "hf": (0, 1, [0], [("charge", 8.80), ("charge", 8.90), ("charge", 9.10)]),
    "h2o": (
        1,
        2,
        [0],
        [("charge", 7.30), ("charge", 7.40), ("charge", 7.60), ("spin", 0.90)],
    ),
}
SCF_CONV = {"energy": 1e-11, "density": 1e-9}
CDFT_CONV = 1e-10
NWCHEM_SHELL = "SPDFGHIK"


def nwchem_basis_block(basis_name: str, symbols: list[str]) -> str:
    """ferric's basis as an NWChem `basis` block (renormalized coefficients;
    NWChem renormalizes contractions itself, same convention as ferric)."""
    lines, pure = [], set()
    for s in dict.fromkeys(symbols):
        for sh in common.ferric_shells(basis_name, common.z_of(s)):
            if sh["l"] >= 2:
                pure.add(sh["pure"])
            lines.append(f"{s} {NWCHEM_SHELL[sh['l']]}")
            for e, c in zip(sh["exps"], sh["coefs"]):
                lines.append(f"  {e!r} {c!r}")
    if len(pure) > 1:
        raise ValueError(f"{basis_name}: mixed spherical/Cartesian l>=2 shells")
    kind = "cartesian" if pure == {False} else "spherical"
    return f'basis "ao basis" {kind}\n' + "\n".join(lines) + "\nend\n"


def nwchem_input(
    symbols, coords, basis_name, charge, mult, grid, cdft_line: str | None
) -> str:
    nrad, iang, _ = GRIDS[grid]
    geom = "\n".join(
        f"  {s} {x!r} {y!r} {z!r}" for s, (x, y, z) in zip(symbols, coords)
    )
    out = [
        "start cdft",
        "title ferric-validation-cdft",
        # Bohr, no reorientation/centering/symmetry: ferric's frame exactly.
        "geometry units bohr noautoz noautosym nocenter",
        "  symmetry c1",
        geom,
        "end",
        nwchem_basis_block(basis_name, symbols).rstrip("\n"),
        f"charge {charge}",
        "dft",
        "  odft",
        f"  mult {mult}",
        f"  xc {XC}",
        f"  grid lebedev {nrad} {iang} becke treutler",
        f"  convergence energy {SCF_CONV['energy']:.0e} "
        f"density {SCF_CONV['density']:.0e} nolevelshifting",
        "  tolerances tight",
        "  iterations 300",
        "  direct",
    ]
    if cdft_line:
        out.append(f"  {cdft_line}")
    out += [
        "end",
        "set dft:no_prune T",
        "set dft:cdft_maxiter 200",
        "task dft energy",
        "",
    ]
    return "\n".join(out)


NWCHEM_TIMEOUT_S = 1800
NWCHEM_FAILURE_MARKERS = (
    "Calculation failed to converge",
    "CDFT failed to optimize multipliers",
)


def run_nwchem(inp_text: str, inp_path: Path) -> str:
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    inp_path.write_text(inp_text)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=SCRATCH_ROOT))
    try:
        shutil.copy(inp_path, work / "cdft.nw")
        try:
            proc = subprocess.run(
                [NWCHEM, "cdft.nw"],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=NWCHEM_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{inp_path.name}: NWChem timed out after {NWCHEM_TIMEOUT_S} s"
            ) from exc
        out = proc.stdout
        if proc.returncode != 0:
            err = "\n".join(proc.stderr.splitlines()[-20:])
            raise RuntimeError(
                f"{inp_path.name}: NWChem exited {proc.returncode}\n{err}"
            )
        # dft_scf.F: SCF non-convergence and the cDFT multiplier loop hitting
        # cdft_maxiter both print a message; neither may become a reference.
        for bad in NWCHEM_FAILURE_MARKERS:
            if bad in out:
                raise RuntimeError(f"{inp_path.name}: NWChem reported {bad!r}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def parse(out: str, label: str) -> dict:
    e = re.findall(r"Total DFT energy =\s+(-?\d+\.\d+)", out)
    if not e:
        tail = "\n".join(out.splitlines()[-25:])
        raise RuntimeError(f"{label}: no final energy\n{tail}")
    enuc = float(re.findall(r"Nuclear repulsion energy =\s+(-?\d+\.\d+)", out)[-1])
    lam = re.findall(r"CDFT multipliers:\s*\n\s*1\s+(-?\d+\.\d+)", out)
    ver = re.search(r"\(NWChem\)\s+(\S+)", out)
    rho = re.findall(r"Grid integrated density:\s+(-?\d+\.\d+)", out)
    if "Grid pruning is: off" not in out or "Spatial weights used: Becke" not in out:
        raise RuntimeError(f"{label}: grid is not the unpruned Becke grid requested")
    if "Treutler" not in out:
        raise RuntimeError(f"{label}: radial quadrature is not Treutler")
    return {
        "energy": float(e[-1]),
        "lambda": float(lam[-1]) if lam else None,
        "nuclear_repulsion": enuc,
        "grid_integrated_density": float(rho[-1]) if rho else None,
        "version": ver.group(1) if ver else "unknown",
    }


def main() -> int:
    only = set(sys.argv[1:])
    written = []
    version = None
    for system, (charge, mult, frag, targets) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        z_frag = sum(common.z_of(symbols[a]) for a in frag)
        # NWChem atom ranges are 1-based and contiguous.
        lo, hi = min(frag) + 1, max(frag) + 1
        assert list(range(lo - 1, hi)) == sorted(frag), "fragment must be contiguous"
        for basis_name in BASES:
            tag = f"{system}_{basis_name}"
            unc = {}
            for grid in GRIDS:
                label = f"{tag}_unconstrained_{grid}"
                txt = nwchem_input(
                    symbols, coords, basis_name, charge, mult, grid, None
                )
                r = parse(run_nwchem(txt, INPUT_DIR / f"{label}.nw"), label)
                version = r["version"]
                unc[grid] = {
                    "energy": r["energy"],
                    "grid_integrated_density": r["grid_integrated_density"],
                }
                enuc = r["nuclear_repulsion"]
            cons = []
            for kind, target in targets:
                entry = {"kind": kind, "target": target}
                if kind == "charge":
                    # NWChem's `charge q` is the fragment CHARGE; N = Z - q.
                    q = z_frag - target
                    line = f"cdft {lo} {hi} charge {q!r} pop becke convergence {CDFT_CONV:.0e}"
                    entry["nwchem_fragment_charge"] = q
                else:
                    line = f"cdft {lo} {hi} spin {target!r} pop becke convergence {CDFT_CONV:.0e}"
                entry["nwchem_cdft_line"] = line
                for grid in GRIDS:
                    label = f"{tag}_{kind}{target:.2f}_{grid}"
                    txt = nwchem_input(
                        symbols, coords, basis_name, charge, mult, grid, line
                    )
                    r = parse(run_nwchem(txt, INPUT_DIR / f"{label}.nw"), label)
                    if r["lambda"] is None:
                        raise RuntimeError(f"{label}: no CDFT multiplier printed")
                    entry[grid] = {
                        "energy": r["energy"],
                        "lambda": r["lambda"],
                        "e_minus_unconstrained": r["energy"] - unc[grid]["energy"],
                        "grid_integrated_density": r["grid_integrated_density"],
                    }
                cons.append(entry)
                print(
                    f"{tag:14s} {kind:6s} {target:5.2f} "
                    + " | ".join(
                        f"{g}: E={entry[g]['energy']:.10f} lam={entry[g]['lambda']:+.8f}"
                        for g in GRIDS
                    ),
                    flush=True,
                )
            payload = {
                "row": ROW_NAME,
                "system": system,
                "basis": basis_name,
                "charge": charge,
                "multiplicity": mult,
                "functional": FUNCTIONAL,
                "fragment": frag,
                "nuclear_repulsion": enuc,
                "nao": common.ferric_nao(basis_name, symbols),
                "grids": {
                    g: {"radial": n, "nwchem_lebedev_index": i, "angular": a}
                    for g, (n, i, a) in GRIDS.items()
                },
                "unconstrained": unc,
                "constrained": cons,
                "provenance": common.provenance(
                    code="NWChem",
                    version=version,
                    keywords={
                        "module": "dft (odft)",
                        "xc": XC,
                        "cdft": "pop becke, convergence 1e-10, cdft_maxiter 200",
                        "grid": "lebedev <nrad> <iang> becke treutler; set dft:no_prune T",
                        "integrals": "direct, tolerances tight, no density fitting",
                        "inputs": str(INPUT_DIR.relative_to(common.ROOT))
                        + f"/{tag}_*.nw",
                    },
                    basis_name=basis_name,
                    xyz_path=xyz,
                    coords_bohr=coords,
                    symbols=symbols,
                    grid={
                        g: f"{n}x{a} Treutler(Perez-Jorda Chebyshev) x Lebedev, Becke "
                        "size-adjusted, unpruned"
                        for g, (n, _, a) in GRIDS.items()
                    },
                    aux=None,
                    frozen_core=None,
                    scf_conv={**SCF_CONV, "cdft_multiplier": CDFT_CONV},
                    stability=None,
                    generator="scripts/validation/gen_cdft.py",
                ),
            }
            path = common.write_reference(ROW, system, basis_name, payload)
            written.append(path)
    print(f"GEN_CDFT_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
