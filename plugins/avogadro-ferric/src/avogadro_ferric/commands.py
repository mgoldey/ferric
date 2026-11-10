"""Menu commands (Extensions > ferric) and the electrostatic model.

Menu-command replies (avogadrolibs 2.1.0, qtgui/interfacescript.cpp): ``cjson``
replaces the molecule (``readProperties`` merges properties only), ``message``
is shown in a dialog, ``error`` in an error dialog.
"""

from __future__ import annotations

import numpy as np

from . import avo, config, engine, orbitals, units


def _energy_message(label: str, energy_ha: float, extra: str = "") -> str:
    return (
        f"ferric {label}\n"
        f"E = {energy_ha:.10f} Hartree\n"
        f"  = {energy_ha * units.HARTREE_TO_EV:.6f} eV\n"
        f"  = {energy_ha * units.HARTREE_TO_KJ_PER_MOL:.4f} kJ/mol\n"
        f"  = {energy_ha * units.HARTREE_TO_KCAL_PER_MOL:.4f} kcal/mol" + extra
    )


def single_point(avo_input: dict) -> dict:
    cfg = config.load()
    job = avo.job_from_input(avo_input, cfg)
    cjson = avo.cjson_of(avo_input)
    avo.progress(f"ferric {job.label}: SCF")
    extra = ""
    if job.open_shell:
        energy, _ = engine.energy_gradient(job)
        extra = (
            "\n(open shell: ferric's charge evaluators take closed-shell results only)"
        )
    else:
        scf = engine.closed_shell_scf(job)
        res = scf[2]
        energy = float(res.energy if hasattr(res, "energy") else res.total_energy)
        model = cfg["charge_model"]
        avo.progress(f"ferric {job.label}: {model} charges")
        q = engine.charges(job, model, scf)
        cjson.setdefault("atoms", {}).setdefault("partialCharges", {})[
            f"ferric-{model}"
        ] = q
        extra = f"\n{model} charges stored as 'ferric-{model}' (sum {sum(q):+.6f})"
    cjson.setdefault("properties", {})["totalEnergy"] = energy
    return {
        "moleculeFormat": "cjson",
        "cjson": cjson,
        "message": _energy_message(job.label, energy, extra),
    }


def optimize(avo_input: dict) -> dict:
    job = avo.job_from_input(avo_input)
    avo.progress(f"ferric {job.label}: optimizing geometry")
    coords, energy, converged, steps = engine.optimize(job)
    cjson = avo.strip_stale(avo.cjson_of(avo_input))
    cjson["atoms"]["coords"]["3d"] = [float(v) for v in np.asarray(coords).reshape(-1)]
    cjson.setdefault("properties", {})["totalEnergy"] = energy
    out = {"moleculeFormat": "cjson", "cjson": cjson}
    status = "converged" if converged else "NOT converged"
    out["message"] = _energy_message(
        job.label, energy, f"\noptimization {status} in {steps} steps"
    )
    return out


def normalized_modes(modes) -> list[list[float]]:
    """Cartesian displacement vectors scaled to unit length (Avogadro animates
    eigenVectors as Angstrom displacements and does not normalise them)."""
    out = []
    for row in modes:
        v = np.asarray(row, dtype=float)
        n = np.linalg.norm(v)
        out.append([float(x) for x in (v / n if n > 0 else v)])
    return out


def frequencies(avo_input: dict) -> dict:
    job = avo.job_from_input(avo_input)
    avo.progress(f"ferric {job.label}: Hessian")
    fr = engine.frequencies(job)
    freqs = [float(f) for f in fr.frequencies]
    if not hasattr(fr, "normal_modes"):
        raise RuntimeError("this ferric build has no FrequencyResult.normal_modes")
    modes = normalized_modes(fr.normal_modes)
    cjson = avo.cjson_of(avo_input)
    cjson["vibrations"] = {
        "frequencies": freqs,
        "modes": list(range(1, len(freqs) + 1)),
        "eigenVectors": modes,
    }
    cjson.setdefault("properties", {})["totalEnergy"] = float(fr.energy)
    n_imag = sum(1 for f in freqs if f < 0)
    msg = f"ferric {job.label}: {len(freqs)} vibrational modes ({fr.hessian_source} Hessian)"
    if n_imag:
        msg += (
            f"\n{n_imag} imaginary frequenc{'y' if n_imag == 1 else 'ies'} "
            "(shown negative): not a minimum, optimize first."
        )
    msg += "\nNo IR intensities: ferric does not compute dipole derivatives here."
    return {"moleculeFormat": "cjson", "cjson": cjson, "message": msg}


def molecular_orbitals(avo_input: dict) -> dict:
    job = avo.job_from_input(avo_input)
    f = engine.ferric()
    if not hasattr(f.BasisSet, "shells") or (
        not job.is_hf and not hasattr(f.DftResult, "mo_coefficients")
    ):
        raise RuntimeError(
            "orbital export needs a ferric with BasisSet.shells() and "
            "DftResult.mo_coefficients() (ferric 0.1.0rc11 or newer)"
        )
    avo.progress(f"ferric {job.label}: SCF")
    mol, bs, res = engine.closed_shell_scf(job)
    nelec = sum(job.numbers) - job.charge
    data = orbitals.orbitals_cjson(f, mol, bs, res, job.numbers, nelec)
    cjson = avo.cjson_of(avo_input)
    for key in ("basisSet", "orbitals"):
        cjson.pop(key, None)
    cjson.update(data)
    energy = float(res.energy if hasattr(res, "energy") else res.total_energy)
    cjson.setdefault("properties", {})["totalEnergy"] = energy
    nocc = nelec // 2
    eps = data["orbitals"]["energies"]
    msg = _energy_message(job.label, energy)
    msg += f"\nHOMO {eps[nocc - 1]:.4f} eV"
    if nocc < len(eps):
        msg += f", LUMO {eps[nocc]:.4f} eV"
    return {"moleculeFormat": "cjson", "cjson": cjson, "message": msg}


def configure(avo_input: dict, want_options: bool) -> dict:
    if want_options:
        return {"userOptions": config.user_options(config.load())}
    cfg = config.update_from_options(avo_input.get("options", {}) or {})
    # No "cjson" in the reply: Avogadro then leaves the molecule untouched.
    return {"message": f"ferric defaults saved to {config.config_path()}:\n{cfg}"}


def electrostatics(avo_input: dict, model: str, want_potentials: bool) -> dict:
    job = avo.job_from_input(avo_input)
    scf = engine.closed_shell_scf(job)
    if want_potentials:
        pts = np.asarray(avo_input.get("points", []), dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            return {"potentials": []}
        phi = engine.esp_at_points(job, pts, scf)
        # Hartree/e (= e/Bohr) -> e/Angstrom, the unit of Avogadro's own
        # sum_i q_i / r_i fallback (calc/chargemodel.cpp).
        return {"potentials": [v * units.ANGSTROM_TO_BOHR for v in phi]}
    return {"charges": engine.charges(job, model, scf)}
