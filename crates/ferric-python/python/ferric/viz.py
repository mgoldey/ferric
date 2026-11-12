"""Small py3Dmol views for ferric results.

py3Dmol is imported only when a view is built, so importing this module needs
nothing beyond numpy. The functions that do not draw (`molecule_to_xyz`,
`read_cube`, `cube_surfaces`, `diverging_colors`) work without it.

Units follow the rest of ferric: cube files are in Bohr, `Molecule.coords()`
is in Angstrom, and the 3Dmol viewer works in Angstrom.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

__all__ = [
    "cube_surfaces",
    "diverging_colors",
    "molecule_to_xyz",
    "read_cube",
    "show_cube",
    "show_molecule",
    "show_surface_values",
]

_BOHR_TO_ANGSTROM = 0.529_177_210_92
_INSTALL_HINT = "py3Dmol is required for ferric.viz views: pip install py3Dmol"
# Default isosurface level for orbitals (Bohr^-3/2) and densities (e/Bohr^3).
DEFAULT_ISOVALUE = 0.05


def _py3dmol():
    try:
        import py3Dmol
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err
    return py3Dmol


def molecule_to_xyz(mol) -> str:
    """XYZ text (Angstrom) for a `ferric.Molecule`, to 12 decimals."""
    symbols = mol.symbols()
    lines = [str(len(symbols)), "ferric"]
    for sym, (x, y, z) in zip(symbols, mol.coords(), strict=True):
        lines.append(f"{sym} {x:.12f} {y:.12f} {z:.12f}")
    return "\n".join(lines) + "\n"


def read_cube(path) -> dict:
    """Parse a Gaussian cube file (positive voxel counts, Bohr axes).

    Returns `numbers` (natoms,), `coords` (natoms, 3) in Bohr, `origin` (3,),
    `axes` (3, 3) (one voxel step per row) and `values` (nx, ny, nz) with x
    the slowest-varying index, as the format stores it.
    """
    lines = Path(path).read_text().splitlines()
    natoms, *origin = lines[2].split()[:4]
    natoms = int(natoms)
    if natoms < 0:
        raise ValueError(
            "cube files with orbital records (negative atom count) are not supported"
        )
    shape, axes = [], []
    for row in lines[3:6]:
        n, *step = row.split()[:4]
        if int(n) <= 0:
            raise ValueError("cube files with Angstrom voxel counts are not supported")
        shape.append(int(n))
        axes.append([float(s) for s in step])
    atoms = np.array([line.split()[:5] for line in lines[6 : 6 + natoms]], dtype=float)
    values = np.array(" ".join(lines[6 + natoms :]).split(), dtype=float)
    if values.size != math.prod(shape):
        raise ValueError(f"cube has {values.size} values for a {shape} grid")
    return {
        "numbers": atoms[:, 0].astype(int),
        "coords": atoms[:, 2:5],
        "origin": np.array(origin, dtype=float),
        "axes": np.array(axes),
        "values": values.reshape(shape),
    }


def cube_surfaces(
    values, isovalue: float = DEFAULT_ISOVALUE, signed: bool | None = None
):
    """The isosurfaces `show_cube` draws, as a list of isovalues (positive lobe first).

    `signed=None` draws both lobes when the data has values below
    `-isovalue` (an orbital) and only the positive surface otherwise (a
    density).
    """
    if isovalue <= 0:
        raise ValueError("isovalue must be positive")
    values = np.asarray(values)
    if signed is None:
        signed = bool(values.min() < -isovalue)
    return [isovalue, -isovalue] if signed else [isovalue]


def diverging_colors(values, vlim: float | None = None) -> list[str]:
    """Blue-white-red hex colours for `values`, symmetric about zero.

    Negative values are red and positive values blue (the electrostatic
    potential convention: red marks electron-rich regions). `vlim` is the
    magnitude mapped to full colour; it defaults to `max(abs(values))`.
    """
    values = np.asarray(values, dtype=float)
    if vlim is None:
        vlim = float(np.max(np.abs(values))) if values.size else 1.0
    if vlim <= 0:
        raise ValueError("vlim must be positive")
    t = np.clip(values / vlim, -1.0, 1.0)
    fade = np.rint(255.0 * (1.0 - np.abs(t))).astype(int)
    colours = []
    for ti, f in zip(t, fade, strict=True):
        r, g, b = (255, f, f) if ti < 0 else (f, f, 255)
        colours.append(f"#{r:02x}{g:02x}{b:02x}")
    return colours


def _new_view(width, height, view):
    if view is not None:
        return view
    return _py3dmol().view(width=width, height=height)


def show_molecule(mol, *, style="stick", width=480, height=360, view=None):
    """Draw a `ferric.Molecule`. `style` is "stick", "sphere" or "line".

    Pass `view` to draw into an existing py3Dmol view. Returns the view; in a
    notebook, end a cell with it (or call `.show()`) to display it.
    """
    if style not in ("stick", "sphere", "line"):
        raise ValueError(f"unknown style {style!r}; use 'stick', 'sphere' or 'line'")
    v = _new_view(width, height, view)
    v.addModel(molecule_to_xyz(mol), "xyz")
    v.setStyle({style: {}})
    v.zoomTo()
    return v


def show_cube(
    path,
    *,
    mol=None,
    isovalue: float = DEFAULT_ISOVALUE,
    signed: bool | None = None,
    colors=("#3b6fd6", "#d64545"),
    opacity: float = 0.8,
    width=480,
    height=360,
    view=None,
):
    """Draw a cube file (`ferric.write_cube`) as isosurfaces over its atoms.

    See `cube_surfaces` for `isovalue` and `signed`. `colors` is the (positive,
    negative) pair. Pass the `ferric.Molecule` as `mol` to draw the atoms from
    it; without it the atoms come from the cube text, which then sits in the
    page twice. Returns the py3Dmol view.
    """
    cube = read_cube(path)
    surfaces = cube_surfaces(cube["values"], isovalue, signed)
    text = Path(path).read_text()
    v = _new_view(width, height, view)
    if mol is None:
        v.addModel(text, "cube")
    else:
        v.addModel(molecule_to_xyz(mol), "xyz")
    v.setStyle({"stick": {"radius": 0.12}})
    for level, color in zip(surfaces, colors[: len(surfaces)], strict=True):
        v.addVolumetricData(
            text, "cube", {"isoval": level, "color": color, "opacity": opacity}
        )
    v.zoomTo()
    return v


def show_surface_values(
    mol, points, values, *, vlim=None, radius=0.1, width=480, height=360, view=None
):
    """Draw the molecule and colour sample points by a scalar.

    `points` is (N, 3) in Bohr (for example the first return value of
    `ferric.esp_on_surface`) and `values` is N numbers coloured by
    `diverging_colors`. Returns the py3Dmol view.
    """
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(values) != len(points):
        raise ValueError("points must be (N, 3) with N values")
    v = show_molecule(mol, width=width, height=height, view=view)
    for (x, y, z), color in zip(
        points * _BOHR_TO_ANGSTROM, diverging_colors(values, vlim), strict=True
    ):
        v.addSphere(
            {"center": {"x": x, "y": y, "z": z}, "radius": radius, "color": color}
        )
    return v
