"""Extensions > ferric > Response Modes: RPA dielectric eigenmodes as orbitals.

Each mode alpha of the closed-shell RPA dielectric is a charge density in the
RI auxiliary basis, rho_alpha(r) = sum_P c_P chi_P(r) (``ferric.response_modes``).
That is the same shape of data as a molecular orbital, so the auxiliary basis is
written as the cjson ``basisSet`` and each mode as one ``orbitals`` column, and
Avogadro's Orbitals panel draws the isosurfaces unchanged. The per-shell
ferric-to-Avogadro map is the measured one ``orbitals.basis_and_maps`` builds
for MOs (it refuses a shell above f, or one it cannot map to 1e-9).

What the panel shows:

* each "orbital" is one mode, normalised so that the integral of rho^2 is 1
  (isovalues comparable to an MO's); its sign is arbitrary;
* the "energy" column holds the dielectric eigenvalue lambda_alpha (a pure
  number, not eV), descending: an electron density rho_alpha added to the
  molecule induces -(lambda_alpha - 1) rho_alpha from independent electrons and
  is screened to rho_alpha / lambda_alpha by the full RPA response;
* every mode is unoccupied (electronCount 0), so Avogadro labels the first one
  "LUMO"; that label means nothing here.
"""

from __future__ import annotations

import numpy as np

from . import avo, engine, orbitals

# The order of data/response_modes_options.toml's auxbasis values. Their g
# shells, which the export refuses, start at Ga (cc-pVDZ-RI) and at Sc
# (aug-cc-pVDZ-RI, def2-SVP-RI).
AUX_BASES = ["cc-pvdz-ri", "aug-cc-pvdz-ri", "def2-svp-ri"]


def _option(options: dict, key: str, default):
    value = options.get(key, default)
    return default if value is None else value


def modes_cjson(f, mol, auxbasis, modes, numbers: list[int]) -> dict:
    """{"basisSet": ..., "orbitals": ...} with one L2-normalised mode per column."""
    basis, maps = orbitals.basis_and_maps(f, mol, auxbasis, numbers)
    c = np.asarray(modes.coefficients, dtype=float) / np.asarray(
        modes.l2_norms, dtype=float
    )
    c_avo = orbitals.to_avogadro_mos(maps, c)
    lam = [float(x) for x in modes.eigenvalues]
    return {
        "basisSet": basis,
        "orbitals": {
            "electronCount": 0,
            "occupations": [0] * len(lam),
            "energies": lam,
            "moCoefficients": [float(v) for v in c_avo.T.reshape(-1)],
        },
    }


def command(avo_input: dict) -> dict:
    """The menu command: SCF (closed shell), dielectric modes, cjson reply."""
    options = avo_input.get("options") or {}
    n_modes = int(_option(options, "n_modes", 10))
    if n_modes < 1:
        raise ValueError("Number of modes must be at least 1")
    aux_name = _option(options, "auxbasis", 0)
    if isinstance(aux_name, int) and not isinstance(aux_name, bool):
        aux_name = AUX_BASES[aux_name]
    omega = float(_option(options, "omega", 0.0))

    job = avo.job_from_input(avo_input)
    if job.open_shell:
        raise ValueError("response modes need a closed-shell molecule (multiplicity 1)")
    f = engine.ferric()
    if not hasattr(f, "response_modes"):
        raise RuntimeError("this ferric build has no response_modes()")
    avo.progress(f"ferric {job.label}: SCF")
    mol, bs, res = engine.closed_shell_scf(job)
    aux = f.BasisSet.bundled(str(aux_name))
    avo.progress(f"ferric {job.label}: dielectric modes in {aux_name}")
    modes = f.response_modes(mol, bs, aux, res, n_modes=n_modes, omega=omega)
    data = modes_cjson(f, mol, aux, modes, job.numbers)

    cjson = avo.cjson_of(avo_input)
    for key in ("basisSet", "orbitals"):
        cjson.pop(key, None)
    cjson.update(data)
    lam = data["orbitals"]["energies"]
    freq = "static" if omega == 0.0 else f"imaginary frequency {omega:g} Ha"
    msg = (
        f"ferric {job.label}: {len(lam)} RPA response modes ({freq}, "
        f"aux basis {aux_name}, {modes.naux} functions)\n"
        "Orbitals panel: each entry is one mode's density, integral rho^2 = 1; "
        "the 'energy' column is the dielectric eigenvalue lambda (not eV).\n"
        f"lambda: {', '.join(f'{x:.4f}' for x in lam[:5])}"
        + (" ..." if len(lam) > 5 else "")
    )
    return {"moleculeFormat": "cjson", "cjson": cjson, "message": msg}
