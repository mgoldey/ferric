"""Atomic properties on an ECP atom use the screened nuclear charge.

`ferric.esp_at_atoms`, `mulliken_charges` and `lowdin_charges` pair the
valence-only density of an ECP atom with `Z - n_core_ecp`; a bare `Molecule`
carries no ECP until a basis is attached, so the bindings apply the basis's ECP
before calling the Rust properties. Reference: PySCF RHF + ECP on the same
basis JSON and Bohr geometry (`scripts/validation/gen_ecp_properties.py`).

Artifact hypothesis: with the bare `Z` the iodine charge is 53 instead of 25, so
every ESP at the H nucleus is off by 28 / r (~9 a.u.) and the Mulliken charges
by 28 e; no bar below can pass by accident.
"""

import json
from pathlib import Path

import pytest

import ferric

ROOT = Path(__file__).resolve().parents[3]
REF_DIR = ROOT / "testdata" / "reference" / "validation" / "ecp_properties"
MOL_DIR = ROOT / "testdata" / "molecules" / "validation" / "ecp"

# The same chain as crates/ferric-scf/tests/properties_effective_z.rs
# (TOL_CHAIN = 2e-8, measured max 1.8e-9 there); the bar here is 5x looser to
# absorb the binding's own SCF defaults.
TOL = 1e-7


@pytest.mark.parametrize("system", ["hi", "ch3i"])
def test_ecp_atom_properties_match_pyscf(system):
    ref = json.loads((REF_DIR / f"{system}_def2-svp.json").read_text())["properties"]
    mol = ferric.Molecule.from_xyz(str(MOL_DIR / f"{system}.xyz"))
    bs = ferric.BasisSet.bundled("def2-svp")
    rhf = ferric.run_rhf(mol, bs, density_conv=1e-9, energy_conv=1e-11)
    got = {
        "esp_at_atoms": ferric.esp_at_atoms(mol, bs, rhf),
        "mulliken_charges": ferric.mulliken_charges(mol, bs, rhf),
        "lowdin_charges": ferric.lowdin_charges(mol, bs, rhf),
    }
    for key, values in got.items():
        want = ref[key]
        assert len(values) == len(want)
        for a, (g, w) in enumerate(zip(values, want)):
            assert abs(g - w) < TOL, f"{system} {key}[{a}]: {g} vs PySCF {w}"
