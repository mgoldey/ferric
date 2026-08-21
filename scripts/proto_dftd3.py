"""Prototype: DFT-D3(BJ) dispersion correction integrated with ferric.

Demonstrates how to add Grimme's D3(BJ) dispersion correction to a ferric
DFT energy using the `dftd3` Python package (grimme-lab/simple-dftd3).

The D3 correction depends only on nuclear geometry and the functional name —
it is independent of the basis set and electronic structure, so it can be
computed once and added to any DFT energy.

Validated against PySCF PBE/cc-pVDZ + standalone D3(BJ) on the same geometry.
"""

import sys
import numpy as np

BOHR_PER_ANGSTROM = 1.8897259886

# ── Parse water geometry for D3 ──

def parse_xyz(path):
    """Parse XYZ file, return (atomic_numbers, coords_bohr)."""
    symbol_to_z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6,
                   "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12,
                   "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18}
    with open(path) as f:
        natom = int(f.readline().strip())
        f.readline()  # comment
        numbers = []
        coords = []
        for _ in range(natom):
            parts = f.readline().split()
            numbers.append(symbol_to_z[parts[0]])
            coords.append([float(x) * BOHR_PER_ANGSTROM for x in parts[1:4]])
    return np.array(numbers, dtype=np.int32), np.array(coords)


def compute_d3bj(numbers, coords_bohr, method="pbe"):
    """Compute D3(BJ) dispersion energy and gradient.

    Args:
        numbers: atomic numbers, shape (natom,)
        coords_bohr: coordinates in Bohr, shape (natom, 3)
        method: functional name for BJ parameter lookup (e.g. "pbe", "b3lyp")

    Returns:
        (energy_hartree, gradient_hartree_per_bohr)
    """
    from dftd3.interface import DispersionModel, RationalDampingParam

    model = DispersionModel(numbers, coords_bohr)
    param = RationalDampingParam(method=method)
    res = model.get_dispersion(param, grad=True)
    return res["energy"], np.array(res["gradient"])


def main():
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import ferric

    xyz_path = "testdata/molecules/water.xyz"
    numbers, coords_bohr = parse_xyz(xyz_path)

    print("=" * 60)
    print("  DFT-D3(BJ) Prototype — water/cc-pVDZ/PBE")
    print("=" * 60)

    # ── ferric DFT ──
    mol = ferric.Molecule.from_xyz(xyz_path)
    bs = ferric.BasisSet.bundled("cc-pvdz")
    dft_result = ferric.run_dft(mol, bs, functional="PBE")
    e_dft = dft_result.total_energy
    print(f"\nferric PBE/cc-pVDZ energy:    {e_dft:.10f} Ha")

    # ── D3(BJ) correction ──
    e_d3, grad_d3 = compute_d3bj(numbers, coords_bohr, method="pbe")
    print(f"D3(BJ) correction (PBE):     {e_d3:.10f} Ha")

    e_total = e_dft + e_d3
    print(f"PBE-D3(BJ)/cc-pVDZ total:    {e_total:.10f} Ha")

    # ── PySCF reference ──
    sys.path.insert(0, "/home/matt/pyscf-local")
    from pyscf import gto, dft

    mol_pyscf = gto.M(
        atom="O 0.000000 0.000000 0.117790; "
             "H 0.000000 0.755453 -0.471161; "
             "H 0.000000 -0.755453 -0.471161",
        basis="cc-pvdz", unit="Angstrom", verbose=0,
    )
    mf = dft.RKS(mol_pyscf)
    mf.xc = "pbe"
    e_pyscf = mf.kernel()
    print(f"\nPySCF PBE/cc-pVDZ energy:    {e_pyscf:.10f} Ha")

    # D3 on the PySCF geometry (same molecule, different Bohr conversion path)
    pyscf_coords = mol_pyscf.atom_coords()
    pyscf_numbers = mol_pyscf.atom_charges()
    e_d3_pyscf, grad_d3_pyscf = compute_d3bj(pyscf_numbers, pyscf_coords, method="pbe")
    e_total_pyscf = e_pyscf + e_d3_pyscf
    print(f"D3(BJ) correction (PySCF):   {e_d3_pyscf:.10f} Ha")
    print(f"PySCF PBE-D3(BJ) total:      {e_total_pyscf:.10f} Ha")

    # ── Comparison ──
    print("\n" + "-" * 60)
    dft_diff = abs(e_dft - e_pyscf)
    d3_diff = abs(e_d3 - e_d3_pyscf)
    total_diff = abs(e_total - e_total_pyscf)
    print(f"DFT energy difference:       {dft_diff:.2e} Ha")
    print(f"D3 energy difference:        {d3_diff:.2e} Ha")
    print(f"Total energy difference:     {total_diff:.2e} Ha")

    # D3 should agree to ~1e-9 (same geometry up to Bohr conversion)
    assert d3_diff < 1e-6, f"D3 energies disagree by {d3_diff:.2e} Ha"
    # DFT should agree to ferric's usual PBE tolerance (~1e-4 to 1e-6)
    assert dft_diff < 1e-4, f"DFT energies disagree by {dft_diff:.2e} Ha"

    print("\n  D3(BJ) gradient (Hartree/Bohr):")
    print(f"  {'Atom':>4}  {'dE/dx':>12}  {'dE/dy':>12}  {'dE/dz':>12}")
    symbols = ["O", "H", "H"]
    for i, sym in enumerate(symbols):
        print(f"  {sym:>4}  {grad_d3[i,0]:12.6e}  {grad_d3[i,1]:12.6e}  {grad_d3[i,2]:12.6e}")

    # Gradient should agree between the two geometry representations
    grad_max_diff = np.max(np.abs(grad_d3 - grad_d3_pyscf))
    print(f"\n  Gradient max diff (ferric vs PySCF geom): {grad_max_diff:.2e}")
    assert grad_max_diff < 1e-6, f"Gradients disagree by {grad_max_diff:.2e}"

    # ── Show supported functionals ──
    print("\n" + "=" * 60)
    print("  Functionals with D3(BJ) parameters:")
    print("=" * 60)
    supported = []
    for name in ["pbe", "b3lyp", "pbe0", "blyp", "bp86", "scan", "r2scan",
                 "wb97x-v", "tpss", "revpbe", "pw91"]:
        try:
            from dftd3.interface import RationalDampingParam
            RationalDampingParam(method=name)
            supported.append(name)
        except Exception:
            pass
    print("  " + ", ".join(supported))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
