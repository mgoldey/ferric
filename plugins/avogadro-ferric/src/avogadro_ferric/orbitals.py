"""Export ferric's basis set and molecular orbitals in Avogadro's cjson conventions.

Avogadro (avogadrolibs 2.1.0, ``core/gaussianset.cpp`` + ``core/gaussiansettools.cpp``)
evaluates a cjson basis set its own way:

* ``basisSet.coefficients`` are contraction coefficients of NORMALISED primitives;
  Avogadro multiplies in each primitive's normalisation itself and never
  renormalises the contraction.
* every component is normalised on its own (cartesian ``xy`` is a unit function,
  not the ``x^2`` constant), and the component ORDER is fixed per shell type:
  P x,y,z; D xx,yy,zz,xy,xz,yz; D5 d0,d+1,d-1,d+2,d-2; F xxx,yyy,zzz,xyy,xxy,
  xxz,xzz,yzz,yyz,xyz; F7 0,+1,-1,+2,-2,+3,-3.
* ``orbitals.moCoefficients`` is flat with each MO's AO coefficients contiguous
  (``MOs[i + j*nAO]``); ``orbitals.energies`` are in eV.

ferric's AO basis (libint2 order and normalisation) differs in order and, for
cartesian shells, in per-component normalisation. Rather than hard-code a table
that could silently drift, the transformation is MEASURED per shell: each
ferric shell's functions are evaluated with ``ferric.eval_ao_on_points`` and the
Avogadro functions with a line-by-line port of Avogadro's evaluator
(``avogadro_shell_values`` below) at sample points around the shell centre, and
a least-squares fit gives the matrix ``M`` with ``chi_ferric = chi_avogadro @ M``.
A fit residual above ``FIT_TOLERANCE`` (relative) raises instead of exporting
orbitals Avogadro would draw wrong. MO coefficients then map as
``C_avogadro = M @ C_ferric`` shell by shell.
"""

from __future__ import annotations

import math

import numpy as np

from . import units

FIT_TOLERANCE = 1e-9

# cjson shellTypes code for (l, pure).
SHELL_TYPE = {
    (0, False): 0,
    (0, True): 0,
    (1, False): 1,
    (1, True): 1,  # cjson has no pure-p code; Avogadro's P is x, y, z
    (2, False): 2,
    (2, True): -2,
    (3, False): 3,
    (3, True): -3,
}

# Number of Avogadro components per cjson shellTypes code.
N_COMPONENTS = {0: 1, 1: 3, 2: 6, -2: 5, 3: 10, -3: 7}


def _norms(shell_type: int, alpha: float) -> list[float]:
    """Per-component primitive normalisation, as in GaussianSet::initCalculation()."""
    pi3 = math.pi**3
    if shell_type == 0:
        return [alpha**0.75 * 0.71270547]
    if shell_type == 1:
        return [alpha**1.25 * 1.425410941] * 3
    if shell_type == 2:
        a = alpha**1.75 * 1.645922781
        b = alpha**1.75 * 2.850821881
        return [a, a, a, b, b, b]
    if shell_type == -2:
        n0 = (2048 * alpha**7 / (9.0 * pi3)) ** 0.25
        n1 = (2048 * alpha**7 / pi3) ** 0.25
        n2p = (128 * alpha**7 / pi3) ** 0.25
        return [n0, n1, n1, n2p, n1]
    if shell_type == 3:
        norm1, norm2, norm3 = 1.4721580892990938, 3.291845561298979, 5.701643762839922
        s = alpha**2.25
        return [s * norm1] * 3 + [s * norm2] * 6 + [s * norm3]
    if shell_type == -3:
        return [alpha**2.25 * 1.4721580892990935] * 7
    raise ValueError(f"shell type {shell_type} is not exported")


def _angular(shell_type: int, d: np.ndarray) -> np.ndarray:
    """Angular factors per component, as in GaussianSetTools::point*()."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    dr2 = x * x + y * y + z * z
    if shell_type == 0:
        return np.ones((len(d), 1))
    if shell_type == 1:
        return np.stack([x, y, z], axis=1)
    if shell_type == 2:
        return np.stack([x * x, y * y, z * z, x * y, x * z, y * z], axis=1)
    if shell_type == -2:
        return np.stack(
            [(3.0 * z * z - dr2) / 2.0, x * z, y * z, x * x - y * y, x * y], axis=1
        )
    if shell_type == 3:
        return np.stack(
            [
                x * x * x,
                y * y * y,
                z * z * z,
                x * y * y,
                x * x * y,
                x * x * z,
                x * z * z,
                y * z * z,
                y * y * z,
                x * y * z,
            ],
            axis=1,
        )
    if shell_type == -3:
        root6, root60, root360 = (
            2.449489742783178,
            7.745966692414834,
            18.973665961010276,
        )
        xxx, xxy, xxz = x * x * x, x * x * y, x * x * z
        xyy, xyz, xzz = x * y * y, x * y * z, x * z * z
        yyy, yyz, yzz, zzz = y * y * y, y * y * z, y * z * z, z * z * z
        return np.stack(
            [
                zzz - 1.5 * (xxz + yyz),
                (6.0 * xzz - 1.5 * (xxx + xyy)) / root6,
                (6.0 * yzz - 1.5 * (xxy + yyy)) / root6,
                (15.0 * (xxz - yyz)) / root60,
                (30.0 * xyz) / root60,
                (15.0 * xxx - 45.0 * xyy) / root360,
                (45.0 * xxy - 15.0 * yyy) / root360,
            ],
            axis=1,
        )
    raise ValueError(f"shell type {shell_type} is not exported")


def avogadro_shell_values(
    shell_type: int, exponents, coefficients, center_bohr, points_bohr
) -> np.ndarray:
    """Values (npts, ncomp) of one cjson shell as Avogadro 2.1.0 evaluates it (Bohr)."""
    d = np.asarray(points_bohr, dtype=float) - np.asarray(center_bohr, dtype=float)
    dr2 = np.einsum("ij,ij->i", d, d)
    radial = np.zeros((len(d), N_COMPONENTS[shell_type]))
    for alpha, c in zip(exponents, coefficients):
        radial += np.exp(-alpha * dr2)[:, None] * (
            c * np.asarray(_norms(shell_type, alpha))
        )
    return radial * _angular(shell_type, d)


def avogadro_ao_values(cjson: dict, points_bohr: np.ndarray) -> np.ndarray:
    """All AO values (npts, nAO) of a cjson basisSet, Avogadro's evaluation."""
    basis = cjson["basisSet"]
    coords = np.asarray(cjson["atoms"]["coords"]["3d"], dtype=float).reshape(-1, 3)
    coords = coords * units.ANGSTROM_TO_BOHR
    cols = []
    p = 0
    for st, nprim, atom in zip(
        basis["shellTypes"], basis["primitivesPerShell"], basis["shellToAtomMap"]
    ):
        exps = basis["exponents"][p : p + nprim]
        coefs = basis["coefficients"][p : p + nprim]
        p += nprim
        cols.append(avogadro_shell_values(st, exps, coefs, coords[atom], points_bohr))
    return np.hstack(cols)


def avogadro_mo_values(cjson: dict, points_bohr: np.ndarray) -> np.ndarray:
    """MO values (npts, nMO) from a cjson's basisSet + orbitals.moCoefficients."""
    ao = avogadro_ao_values(cjson, points_bohr)
    nao = ao.shape[1]
    c = np.asarray(cjson["orbitals"]["moCoefficients"], dtype=float)
    cmat = c.reshape(-1, nao).T  # MOs[i + j*nAO] -> C[i, j]
    return ao @ cmat


def basis_and_maps(f, mol, basis_set, numbers: list[int], seed: int = 20261109):
    """(cjson basisSet dict, per-shell list of (ferric AO slice, M)) for this molecule."""
    centers, offsets, dims = f.shell_info(mol, basis_set)
    centers = np.asarray(centers, dtype=float)
    shells_per_z: dict[int, list] = {}
    shell_list = []  # (atom index, l, pure, exps, coefs)
    for ai, z in enumerate(numbers):
        if z not in shells_per_z:
            shells_per_z[z] = basis_set.shells(z)
        for ang, pure, exps, coefs in shells_per_z[z]:
            shell_list.append((ai, ang, pure, exps, coefs))
    if len(shell_list) != len(dims):
        raise RuntimeError(
            f"ferric reports {len(dims)} shells but BasisSet.shells() lists {len(shell_list)}"
        )

    rng = np.random.default_rng(seed)
    npts = 48
    point_blocks = []
    for (ai, ang, pure, exps, coefs), center in zip(shell_list, centers):
        if ang > 3:
            raise ValueError(
                "orbital export supports s, p, d and f shells; this basis has an "
                f"l={ang} shell (pick a smaller basis for orbitals)"
            )
        scale = 1.0 / math.sqrt(min(exps))
        dirs = rng.normal(size=(npts, 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        radii = scale * rng.uniform(0.2, 1.5, size=npts)
        point_blocks.append(center + dirs * radii[:, None])
    points = np.vstack(point_blocks)
    ferric_ao = np.asarray(
        f.eval_ao_on_points(mol, basis_set, np.ascontiguousarray(points))
    )

    shell_types, prims, atoms, all_exps, all_coefs = [], [], [], [], []
    maps = []
    for si, ((ai, ang, pure, exps, coefs), center) in enumerate(
        zip(shell_list, centers)
    ):
        st = SHELL_TYPE[(ang, bool(pure))]
        rows = slice(si * npts, (si + 1) * npts)
        cols = slice(int(offsets[si]), int(offsets[si]) + int(dims[si]))
        xf = ferric_ao[rows, cols]
        xa = avogadro_shell_values(st, exps, coefs, center, points[rows])
        m, *_ = np.linalg.lstsq(xa, xf, rcond=None)
        resid = np.linalg.norm(xa @ m - xf) / max(np.linalg.norm(xf), 1e-300)
        if not resid < FIT_TOLERANCE:
            raise RuntimeError(
                f"shell {si} (l={ang}, pure={pure}) does not map onto Avogadro's "
                f"type {st} functions (relative residual {resid:.2e}); refusing to "
                "export orbitals Avogadro would draw wrong"
            )
        shell_types.append(st)
        prims.append(len(exps))
        atoms.append(ai)
        all_exps.extend(float(e) for e in exps)
        all_coefs.extend(float(c) for c in coefs)
        maps.append((cols, m))

    basis = {
        "shellTypes": shell_types,
        "primitivesPerShell": prims,
        "shellToAtomMap": atoms,
        "exponents": all_exps,
        "coefficients": all_coefs,
    }
    return basis, maps


def to_avogadro_mos(maps, c_ferric: np.ndarray) -> np.ndarray:
    """Map an (nAO, nMO) ferric coefficient matrix onto Avogadro's AO functions."""
    blocks = [m @ c_ferric[cols, :] for cols, m in maps]
    return np.vstack(blocks)


def orbitals_cjson(f, mol, basis_set, result, numbers: list[int], nelec: int) -> dict:
    """{"basisSet": ..., "orbitals": ...} for a closed-shell RHF/RKS result."""
    basis, maps = basis_and_maps(f, mol, basis_set, numbers)
    c_ferric = np.asarray(result.mo_coefficients(), dtype=float)
    c_avo = to_avogadro_mos(maps, c_ferric)
    energies = np.asarray(result.orbital_energies(), dtype=float) * units.HARTREE_TO_EV
    nmo = c_avo.shape[1]
    nocc = nelec // 2
    return {
        "basisSet": basis,
        "orbitals": {
            "electronCount": int(nelec),
            "occupations": [2] * nocc + [0] * (nmo - nocc),
            "energies": [float(e) for e in energies],
            "moCoefficients": [float(v) for v in c_avo.T.reshape(-1)],
        },
    }
