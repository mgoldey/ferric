"""ORCA 6.1.1 cross-check (third code) for the "RHF+ECP" row.

Consumer: crates/ferric-scf/tests/validation_ecp.rs (`rhf_ecp_vs_orca`).
Inputs:   scripts/validation/orca/ecp/<system>_<basis>.inp   (committed)
Output:   testdata/reference/validation/ecp/orca_<system>_<basis>.json

Same systems/bases as gen_ecp.py, closed-shell RHF energies only (the I-atom
UHF and the gradients are PySCF-only; ORCA adds an independent ECP integral
code for the ENERGY, which is where a mis-mapped ECP shows up first).

LIKE-FOR-LIKE (default, `ecp_source = "ferric-json NewECP"`):
    ! RHF def2-SVP VeryTightSCF NoRI NoFrozenCore Bohrs
  plus a `%basis` block with ferric's basis as `NewGTO` AND ferric's ECP as
  `NewECP` (common.orca_basis_block), geometry in Bohr with ferric's constant.
  The def2 keyword is kept only so that nothing ORCA auto-assigns can differ
  from the explicit blocks; every element in these systems is overridden.
  `NoRI` switches off RI-J/RIJCOSX; `NoFrozenCore` is a no-op for SCF but is
  spelled out per design §2.3. VeryTightSCF (TolE 1e-9) instead of TightSCF
  (TolE 1e-8) keeps ORCA's convergence floor well under the 1e-6 bar.

FALLBACK (`--builtin-ecp`, `ecp_source = "orca-builtin def2-ECP"`): ferric's
  basis via NewGTO but ORCA's OWN def2-ECP (no NewECP). The parameters are the
  same published ones (Peterson 2003 for I, Leininger 1996 for Rb, Metz 2000
  for Sn), but the digit strings are ORCA's library copy, not ferric's, so the
  Rust test widens its ORCA bar to 1e-5 for such files and says so. Use only if
  the explicit NewECP input turns out not to be accepted by ORCA.

Per run the generator REFUSES to write unless ORCA terminated normally, the
SCF converged, ORCA's electron count equals sum(Z) - sum(N_core) - charge,
and ORCA's printed nuclear repulsion (computed with Z - N_core) matches
PySCF-with-ferric's-geometry's E_nuc to 1e-6 (ORCA prints 8 decimals) —
the checks that N_core and the geometry reached ORCA intact.

Run (light; each ORCA job is seconds to a couple of minutes on 1 core):
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_ecp_orca.py [--builtin-ecp] \
        [--write-only] [system ...]
`--write-only` regenerates the committed .inp files without running ORCA.
ORCA runs in a temporary directory (copies of the .inp), so no .gbw/.out
litter lands in the repo; the full input text is stored in the JSON.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from gen_ecp import ECP_MOL_DIR, ROW, SYSTEMS  # noqa: E402

ROW_NAME = "RHF+ECP (ORCA cross-check)"
INP_DIR = Path(__file__).resolve().parent / "orca" / "ecp"
ORCA_BASIS_KEYWORD = {"def2-svp": "def2-SVP", "def2-tzvp": "def2-TZVP"}
KEYWORDS = "RHF {basis} VeryTightSCF NoRI NoFrozenCore"
TOL_ENUC = 1e-6


def nuclear_repulsion_bohr(symbols, coords, ncore) -> float:
    """E_nn with ECP-reduced charges, computed here in pure Python so the
    ORCA path needs no PySCF (same formula as ferric's nuclear_repulsion)."""
    e = 0.0
    q = [common.z_of(s) - n for s, n in zip(symbols, ncore)]
    for i in range(len(symbols)):
        for j in range(i):
            r = sum((a - b) ** 2 for a, b in zip(coords[i], coords[j])) ** 0.5
            e += q[i] * q[j] / r
    return e


def main() -> int:
    args = sys.argv[1:]
    builtin = "--builtin-ecp" in args
    write_only = "--write-only" in args
    only = {a for a in args if not a.startswith("--")}
    ecp_source = "orca-builtin def2-ECP" if builtin else "ferric-json NewECP"
    written = []
    for system, (charge, mult, method, bases) in SYSTEMS.items():
        if method != "rhf" or (only and system not in only):
            continue
        xyz = ECP_MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        for basis_name in bases:
            ecp_json = common.basis_json_path(basis_name)
            ncore = common.ecp_core_electrons(ecp_json, symbols)
            keywords = KEYWORDS.format(basis=ORCA_BASIS_KEYWORD[basis_name])
            suffix = "_builtin" if builtin else ""
            inp = common.write_orca_input(
                INP_DIR / f"{system}_{basis_name}{suffix}.inp",
                keywords,
                xyz,
                basis_name,
                charge=charge,
                multiplicity=mult,
                ecp_json=None if builtin else ecp_json,
            )
            if write_only:
                print(f"wrote {inp.relative_to(common.ROOT)}")
                continue
            with tempfile.TemporaryDirectory(prefix="ferric-orca-ecp-") as tmp:
                run_inp = Path(tmp) / inp.name
                shutil.copy(inp, run_inp)
                parsed = common.run_orca(run_inp)
            nelec_want = sum(common.z_of(s) for s in symbols) - sum(ncore) - charge
            if int(parsed.get("n_electrons", -1)) != nelec_want:
                raise RuntimeError(
                    f"{inp.name}: ORCA electron count {parsed.get('n_electrons')} != "
                    f"{nelec_want} (N_core not applied as ferric applies it?)"
                )
            enuc_want = nuclear_repulsion_bohr(symbols, coords, ncore)
            if abs(parsed["nuclear_repulsion"] - enuc_want) > TOL_ENUC:
                raise RuntimeError(
                    f"{inp.name}: ORCA E_nuc {parsed['nuclear_repulsion']} != "
                    f"{enuc_want} (geometry or N_core mismatch)"
                )
            payload = {
                "row": ROW_NAME,
                "system": system,
                "basis": basis_name,
                "charge": charge,
                "multiplicity": mult,
                "method": "rhf",
                "ecp_source": ecp_source,
                "nelectron": nelec_want,
                "ecp_core_electrons": ncore,
                "nuclear_repulsion": parsed["nuclear_repulsion"],
                "energy": parsed["energy"],
                "scf_iterations": parsed.get("scf_iterations"),
                "provenance": common.provenance(
                    code="ORCA",
                    version=parsed.get("version", "unknown"),
                    keywords=inp.read_text(),
                    basis_name=basis_name,
                    xyz_path=xyz,
                    coords_bohr=coords,
                    symbols=symbols,
                    grid=None,
                    aux=None,
                    frozen_core="NoFrozenCore",
                    scf_conv="VeryTightSCF",
                    stability=None,
                    ecp_json=None if builtin else ecp_json,
                    generator="scripts/validation/gen_ecp_orca.py",
                    extra={
                        "orca_input": str(inp.relative_to(common.ROOT)),
                        "orca_binary": str(common.ORCA_BINARY),
                    },
                ),
            }
            path = common.write_reference(
                ROW, f"orca_{system}{suffix}", basis_name, payload
            )
            written.append(path)
            print(
                f"{system:7s} {basis_name:9s} [{ecp_source}] "
                f"E {parsed['energy']:.10f} E_nuc {parsed['nuclear_repulsion']:.8f}"
            )
    print(f"GEN_ECP_ORCA_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
