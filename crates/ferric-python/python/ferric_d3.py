"""DFT-D3(BJ) dispersion correction integration for ferric.

Wraps the `dftd3` Python package (grimme-lab/simple-dftd3) to add Grimme's
D3(BJ) dispersion correction to ferric DFT energies and gradients.

Usage::

    import ferric
    from ferric_d3 import d3bj_correction, dft_d3

    # Standalone correction
    e_d3, grad_d3 = d3bj_correction("testdata/molecules/water.xyz", method="pbe")

    # Combined DFT+D3
    result = dft_d3("testdata/molecules/water.xyz", "cc-pvdz", functional="PBE")
    print(result["total_energy"])  # E_DFT + E_D3
"""

import numpy as np

BOHR_PER_ANGSTROM = 1.8897259886


def _parse_xyz(path):
    symbol_to_z = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6,
        "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12,
        "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24,
        "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
        "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    }
    with open(path) as f:
        natom = int(f.readline().strip())
        f.readline()
        numbers = []
        coords = []
        for _ in range(natom):
            parts = f.readline().split()
            numbers.append(symbol_to_z[parts[0]])
            coords.append([float(x) * BOHR_PER_ANGSTROM for x in parts[1:4]])
    return np.array(numbers, dtype=np.int32), np.array(coords)


def d3bj_correction(xyz_path, method="pbe"):
    """Compute D3(BJ) dispersion energy and gradient for an XYZ file.

    Args:
        xyz_path: Path to XYZ file (coordinates in Angstrom).
        method: Functional name for BJ damping parameter lookup.

    Returns:
        (energy_hartree, gradient_array) where gradient is (natom, 3) in Ha/Bohr.

    Raises:
        ImportError: if the `dftd3` package is not installed.
    """
    try:
        from dftd3.interface import DispersionModel, RationalDampingParam
    except ImportError:
        raise ImportError(
            "D3(BJ) requires the 'dftd3' package. Install with: pip install dftd3"
        )

    numbers, coords_bohr = _parse_xyz(xyz_path)
    model = DispersionModel(numbers, coords_bohr)
    param = RationalDampingParam(method=method)
    res = model.get_dispersion(param, grad=True)
    return res["energy"], np.array(res["gradient"])


def dft_d3(xyz_path, basis, functional="PBE", d3_method=None, **dft_kwargs):
    """Run ferric KS-DFT and add D3(BJ) dispersion correction.

    Args:
        xyz_path: Path to XYZ file.
        basis: Basis set name (e.g. "cc-pvdz").
        functional: XC functional name for both DFT and D3 parameter lookup.
        d3_method: Override the D3 parameter lookup name (default: lowercase functional).
        **dft_kwargs: Passed to ferric.run_dft.

    Returns:
        dict with keys: total_energy, e_dft, e_d3, d3_gradient.
    """
    import ferric

    mol = ferric.Molecule.from_xyz(xyz_path)
    bs = ferric.BasisSet.bundled(basis)
    dft_result = ferric.run_dft(mol, bs, functional=functional, **dft_kwargs)

    d3_name = d3_method or functional.lower()
    e_d3, grad_d3 = d3bj_correction(xyz_path, method=d3_name)

    return {
        "total_energy": dft_result.total_energy + e_d3,
        "e_dft": dft_result.total_energy,
        "e_d3": e_d3,
        "d3_gradient": grad_d3,
        "dft_result": dft_result,
    }
