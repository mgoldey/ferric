"""Plain-data views of ferric objects: dicts, JSON, DataFrame-ready records.

    import ferric, ferric.serialize as fs

    res = ferric.run_rhf(mol, basis)
    fs.to_dict(res)                 # {"type": "RhfResult", "energy": ..., ...}
    fs.to_json(res, indent=2)       # non-finite floats become null
    pandas.DataFrame([fs.to_dict(r) for r in results])

A result becomes its public scalar attributes (``int``, ``float``, ``bool``,
``str``, ``None``; read-only getters, not methods; for a ferric class, the
getters its entry in ``RESULT_SCHEMA`` lists), ``"type"`` (the class
name), a normalised ``"energy"`` (the ``energy`` or ``total_energy`` attribute;
the original keys stay), the ``certificate`` when the result has one, and
``"ferric_build"`` (``ferric.__build__``). ``arrays=True`` adds the result's
array accessors (density, MO coefficients, orbital energies, gradient; for a
ferric class, the ones its schema entry lists) as nested lists. A
``ferric.Molecule`` becomes ``symbols``, ``coords_bohr`` and ``ghost``,
``charge`` and ``multiplicity`` (the Molecule's getters).
``molecule_from_dict`` reverses it.

``RESULT_SCHEMA`` maps each ferric class name to the getters (``fields``) and
array accessors (``arrays``) ``to_dict`` reads, in output order; a ferric class
with no entry raises ``TypeError``. It is not free-form: a drift test
(``tests/test_serialize_schema.py``) keeps each entry equal to its class's
public getters, minus that class's names in ``NOT_SERIALIZED``. So a getter
added to a class must be added to its entry (and is then serialized) or, to
keep it out on purpose, to ``NOT_SERIALIZED`` (empty today).

A field is written only when its value is a scalar (above). Fields listed in
the schema whose getters return a list or dict (``frequencies``,
``normal_modes``, ``weights``, ``dm1``, ``ci_vector``...) are therefore filtered
out and never emitted, and an optional dict (``cosx_final_pass``,
``guess_basis``...) appears only when it is ``None``; changing that is an
output change left for its own release. A getter that raises
``AttributeError`` or ``NotImplementedError`` is left out. Any other object
(one ferric does not define, or a ferric exception) is read by reflection: its
public non-callable attributes.

``read_run_log`` parses the CLI's JSON Lines run log (``--json`` /
``<input>.ferric.jsonl``): one object per line, each with ``record`` (kind),
``seq`` and ``t``; the first is ``run_start`` (``schema``, ``ferric_version``,
``git_sha``, ``timestamp``, ``config``, ``molecule``).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, NamedTuple

import numpy as np

import ferric

from ._util import _plain, default_multiplicity

__all__ = ["molecule_from_dict", "read_run_log", "to_dict", "to_json"]

_ARRAY_ACCESSORS = (
    "density",
    "density_alpha",
    "density_beta",
    "mo_coefficients",
    "mo_coefficients_alpha",
    "mo_coefficients_beta",
    "orbital_energies",
    "orbital_energies_alpha",
    "orbital_energies_beta",
    "gradient",
    "cartesian_hessian",
)
_SCALARS = (bool, int, float, str, type(None))


class ResultSchema(NamedTuple):
    """What ``to_dict`` reads from one ferric class: the getters, in output
    order (sorted, as the reflection it replaced listed them), and the
    array accessors ``arrays=True`` adds, in ``_ARRAY_ACCESSORS`` order."""

    fields: tuple[str, ...]
    arrays: tuple[str, ...] = ()


def _s(fields: str, arrays: str = "") -> ResultSchema:
    return ResultSchema(tuple(fields.split()), tuple(arrays.split()))


# Getters deliberately kept out of a class's schema: class name -> names. The
# drift test checks RESULT_SCHEMA[cls] == the class's public getters minus
# these, so this is the one place to exclude a getter. Empty today.
NOT_SERIALIZED: dict[str, frozenset[str]] = {}

# The result schema (roadmap item 13), generated from the reflected field set
# of each class so the output did not change. tests/test_serialize_schema.py
# keeps every entry sorted and equal to its class's public getters minus
# NOT_SERIALIZED, and requires an entry for every exported ferric class. List-
# and dict-valued fields in it are filtered out by to_dict (module docstring).
RESULT_SCHEMA: dict[str, ResultSchema] = {
    "AoIntegrals": _s("nao nelec"),
    "AttenuatedMp2Result": _s(
        (
            "certificate e_os e_ss jk_auxbasis jk_label mp2_corr reference rhf_energy "
            "scf_jk total_energy"
        ),
    ),
    "BasisSet": _s(""),
    "BoltzmannWeights": _s(
        (
            "kt_hartree min_index partition_function relative_energies "
            "temperature_k weights"
        ),
    ),
    "BoysResult": _s("converged iterations"),
    "BseResult": _s(
        "certificate eps_qp nocc nvir omega oscillator_strength qp_suspect_mos",
    ),
    "CasCiResult": _s(
        (
            "active_start certificate ci_vector converged dm1 dm2 e_active "
            "e_core e_total eri_active h_active iterations mo_coeff_active "
            "n_determinants rhf_energy"
        ),
    ),
    "CasScfResult": _s(
        (
            "auxbasis certificate ci_residual_norm ci_vector converged dm1 dm2 "
            "e_core e_total grad_norm integrals iterations mo_coeff mo_energy "
            "n_core n_determinants rhf_energy stability_follow_steps "
            "stability_lowest_eigenvalue stability_verdict"
        ),
    ),
    "CcResult": _s(
        (
            "certificate correlation_energy ladder_block ladder_route reference "
            "spin_components t_correction"
        ),
    ),
    "CdftConstraint": _s("atoms kind target"),
    "CdftCouplingResult": _s("certificate e_a e_b h_ab s_ab"),
    "CdftResult": _s(
        (
            "certificate converged energy hamiltonian_key iterations kinds lambda_tol "
            "lambdas max_constraint_error nocc outer_iterations populations residuals "
            "scf_converged targets total_energy"
        ),
        "density_alpha density_beta orbital_energies_alpha orbital_energies_beta",
    ),
    "ConformerEnsemble": _s(""),
    "Cphf": _s("", "density"),
    "DfJk": _s("naux"),
    "DftResult": _s(
        (
            "certificate converged cosx_final_pass dispersion_model "
            "e_dispersion e_scf e_xc energy exit_reason guess_basis iterations "
            "one_step_energy_drop precision ri_j_precision ri_k_precision "
            "stationarity_tol total_energy volume_ratios"
        ),
        "density mo_coefficients orbital_energies gradient",
    ),
    "DoubleHybridResult": _s(
        (
            "c_os c_ss certificate e_corr_scaled e_dispersion e_ks e_os e_ss "
            "reference total_energy"
        ),
    ),
    "EnsembleDiagnostics": _s(
        (
            "effective_n_conformers max_weight max_weight_index n_conformers "
            "n_within_2kt n_within_5kt n_within_kt temperature_k verdict"
        ),
    ),
    "EomCcsdResult": _s(
        (
            "ccsd_correlation_energy certificate excitation_energies iterations "
            "ladder_route products residuals rhf_energy singles_weight"
        ),
    ),
    "FrequencyResult": _s(
        (
            "asymmetry cartesian_hessian certificate e_dispersion energy "
            "frequencies hessian_source is_linear n_gradient_evaluations "
            "normal_modes trans_rot_frequencies"
        ),
        "cartesian_hessian",
    ),
    "GammaCorrelationResult": _s(
        (
            "auxbasis certificate converged correlation_energy denominators "
            "e_os e_scf e_ss energy exxdiv iterations jk madelung method naux "
            "nocc_active nvir occ_shift quad_points timings"
        ),
    ),
    "GammaOpenShellResult": _s(
        (
            "auxbasis certificate converged e_nuc e_xc electrons_on_grid energy "
            "ewald_start exact_exchange_fraction exxdiv functional gap_alpha "
            "gap_beta gaps_satisfied iterations jk madelung method "
            "n_grid_points nalpha nao nbeta none_stage_energy s2 timings "
            "total_energy"
        ),
        "density gradient",
    ),
    "GammaRhfResult": _s(
        (
            "auxbasis certificate converged e_nuc energy exxdiv iterations jk "
            "madelung n_dropped n_g_half nao naux naux_kept omega timings "
            "total_energy"
        ),
        "density gradient",
    ),
    "GammaRksResult": _s(
        (
            "auxbasis certificate converged e_nuc e_xc electrons_on_grid energy "
            "exact_exchange_fraction exxdiv functional iterations jk madelung "
            "n_grid_points nao neighbour_cutoff timings total_energy"
        ),
        "density gradient",
    ),
    "GwResult": _s(
        (
            "certificate eps_mf eps_qp mo_indices n_ev_iter outer_converged "
            "qp_converged ref_energy sigma_c sigma_x z_factor"
        ),
    ),
    "IrcBranch": _s("converged coords energy steps symbols"),
    "IrcResult": _s("forward reverse saddle_energy"),
    "KpointCorrelationResult": _s(
        (
            "auxbasis centring certificate converged correlation_energy "
            "denominators drpa_energy e_direct e_os e_scf e_ss energy exxdiv "
            "iterations jk lindep_min_kept lindep_near_noise_floor "
            "lindep_total_kept madelung mesh method naux nk nocc_active nvir "
            "occ_shift per_q quad_points timings"
        ),
    ),
    "KpointRksResult": _s(
        (
            "auxbasis centring certificate converged e_nuc e_xc "
            "electrons_on_grid energy exact_exchange_fraction exxdiv functional "
            "homo iterations jk kpts lumo madelung mesh mo_energy n_grid_points "
            "nao nk timings total_energy"
        ),
        "gradient",
    ),
    "KpointScfResult": _s(
        (
            "auxbasis centring certificate converged e_nuc e_xc "
            "electrons_on_grid energy ewald_start exact_exchange_fraction "
            "exxdiv functional gap_alpha gap_beta homo iterations jk kpts "
            "level_shift lindep_max_kept lindep_min_kept "
            "lindep_near_noise_floor lindep_threshold lindep_total_kept lumo "
            "madelung mesh method mo_energy mo_energy_beta n_grid_points nalpha "
            "nao nbeta nk none_stage_energy s2 timings total_energy"
        ),
        "gradient",
    ),
    "LaplaceMp2Result": _s(
        "certificate e_os e_ss jk_auxbasis jk_label mp2_corr scf_jk total_energy"
    ),
    "MbdRsscsResult": _s(
        (
            "alpha_0_rsscs alpha_0_ts beta c6_rsscs c6_ts energy omega_rsscs "
            "r_vdw_rsscs r_vdw_ts"
        ),
    ),
    "MdResult": _s(
        (
            "certificate conserved kinetic_energies label max_conserved_drift "
            "n_dof n_force_evaluations positions_angstrom potential_energies "
            "scf_iterations steps temperatures times_fs total_energies "
            "velocities"
        ),
    ),
    "MmTopology": _s(""),
    "Mp2VResult": _s(
        (
            "att_mp2_corr certificate e_os e_ss jk_auxbasis jk_label n_nlc_points "
            "rhf_energy scf_jk total_energy vv10_e_nl"
        ),
    ),
    "Mp3Result": _s("certificate e_corr e_hf e_mp2 e_mp3 e_total"),
    "Nevpt2Result": _s(
        (
            "certificate ci_residual_norm e_corr e_reference e_total energies "
            "mo_coeff mo_energy n_active n_core n_virtual norms reference "
            "reference_grad_norm reference_iterations "
            "reference_stability_lowest_eigenvalue rhf_energy"
        ),
    ),
    "OmegaRiResult": _s(
        (
            "e_corr e_os e_ss fit mean_rank_occ mean_rank_vir method "
            "pair_fraction rhf_energy total_energy virtual_fraction"
        ),
    ),
    "OoRiMp2Result": _s(
        (
            "certificate converged grad_norm hf_energy iterations jk_auxbasis jk_label "
            "mp2_corr scf_jk total_energy"
        ),
        "gradient",
    ),
    "Opmp3Result": _s("certificate correlation e_hf e_nbs e_scf local max_ov x_ss"),
    "OptimizeResult": _s("certificate converged energy energy_trace steps"),
    "PdepRpaResult": _s(
        (
            "alpha_atomic alpha_atomic_dynamic alpha_ct alpha_ct_dynamic "
            "alpha_tensor c6_aniso c6_freqs c6_iso c6_molecular_iso "
            "c6_partition c6_source c6_weights certificate chelpg_charges "
            "density_matrix density_second_moment dipole dressed_eigenvectors "
            "e_rpa e_rpa_dft_diag eigenpotentials eigensolver_converged "
            "eigenvalues_freq eigenvalues_static electric_field esp_atoms "
            "esp_points esp_surface hirshfeld_charges inv_dielectric_freq "
            "lowdin_charges mo_coeffs mulliken_charges n_eigenpotentials "
            "orbital_centers orbital_energies orbital_spreads property_gaps "
            "property_warnings quad_freqs quad_weights resp_charges rhf_energy "
            "route total_energy written_files"
        ),
        "orbital_energies gradient",
    ),
    "QmmmOptimizeResult": _s("certificate converged energy steps"),
    "QmmmResult": _s("certificate converged e_pol energy iterations mm_energy"),
    "QmmmSystem": _s(""),
    "ResponseModes": _s(
        (
            "alpha_bare coefficients eigenvalues field_couplings l2_norms "
            "n_modes naux omega"
        ),
    ),
    "RhfResult": _s(
        (
            "certificate computed_quartets conv_thresholds converged "
            "convergence_summary cosx_final_pass energy exit_reason final_de "
            "final_diis_err_max final_dp_max final_dp_rms guess_basis "
            "iterations one_step_energy_drop precision ri_j_precision "
            "ri_k_precision stationarity_tol total_energy unmet_criteria"
        ),
        "density mo_coefficients orbital_energies gradient",
    ),
    "RiMp2Result": _s(
        (
            "certificate jk_auxbasis jk_label local mp2_corr precision reference "
            "rhf_energy scf_jk scf_precision total_energy"
        ),
        "gradient",
    ),
    "RsMp2RpaResult": _s(
        (
            "certificate e_corr e_corr_naive e_delta_drpa_full e_delta_drpa_sr "
            "e_dmp2_lr e_drpa_lr e_lr_mp2 e_mp2_full e_sr_mp2 jk_auxbasis jk_label "
            "reference rhf_energy scf_jk total_energy"
        ),
    ),
    "SaddleResult": _s(
        (
            "certificate converged coords energy hessian_source imaginary_mode "
            "lowest_eigenvalue n_imaginary steps symbols"
        ),
    ),
    "ScsMp2Result": _s(
        (
            "certificate e_os e_ss jk_auxbasis jk_label reference rhf_energy scf_jk "
            "scs_corr total_energy"
        ),
    ),
    "SosMp2Result": _s(
        (
            "c_os certificate e_os formulation jk_auxbasis jk_label n_quad rhf_energy "
            "scf_jk sos_corr total_energy"
        ),
    ),
    "SymmetricRhfResult": _s(
        (
            "converged energy iterations max_symmetrization_shift "
            "orbital_irreps point_group total_energy"
        ),
        "density mo_coefficients orbital_energies gradient",
    ),
    "TddftResult": _s(
        (
            "c_hf c_lr certificate excitation_energies fxc_included grid "
            "iterations method n_roots omega oscillator_strengths reference "
            "residuals s2 s2_reference solver spin"
        ),
    ),
    "TdhfStaticPolarizabilityResult": _s("certificate iso nocc nvir tensor"),
    "UGwResult": _s(
        (
            "certificate eps_mf_a eps_mf_b eps_qp_a eps_qp_b mo_indices "
            "n_ev_iter outer_converged qp_converged_a qp_converged_b ref_energy "
            "sigma_c_a sigma_c_b sigma_x_a sigma_x_b z_factor_a z_factor_b"
        ),
    ),
    "Ucphf": _s("", "density"),
    "UhfResult": _s(
        (
            "certificate computed_quartets converged cosx_final_pass energy "
            "guess_basis iterations one_step_energy_drop precision "
            "stationarity_tol total_energy"
        ),
        (
            "density_alpha density_beta mo_coefficients_alpha "
            "mo_coefficients_beta orbital_energies_alpha orbital_energies_beta "
            "gradient"
        ),
    ),
    "WeightedStats": _s("max mean min std_dev"),
    "XcFunctional": _s("exchange_mix name"),
}


def _build() -> Any:
    return _plain(getattr(ferric, "__build__", None))


def _molecule_dict(mol: Any) -> dict[str, Any]:
    ghost = [bool(g) for g in mol.is_ghost()]
    return {
        "type": "Molecule",
        "symbols": list(mol.symbols()),
        "coords_bohr": [list(map(float, xyz)) for xyz in mol.coords_bohr()],
        "ghost": ghost,
        "charge": int(mol.charge),
        "multiplicity": int(mol.multiplicity),
        "ferric_build": _build(),
    }


def _schema(obj: Any) -> ResultSchema | None:
    """The schema entry of ``obj``'s class when ferric defines that class."""
    cls = type(obj)
    entry = RESULT_SCHEMA.get(cls.__name__)
    if entry is not None and getattr(ferric, cls.__name__, None) is cls:
        return entry
    return None


def _is_ferric_class(cls: type) -> bool:
    """A class ferric exports, other than an exception (read by reflection)."""
    return getattr(ferric, cls.__name__, None) is cls and not issubclass(
        cls, BaseException
    )


def _reflected_fields(obj: Any) -> list[str]:
    """Public attribute names of an object ferric does not define."""
    return [name for name in dir(obj) if not name.startswith("_")]


def to_dict(obj: Any, arrays: bool = False) -> dict[str, Any]:
    """A plain dict (JSON-able apart from non-finite floats) for a Molecule or a
    result object. See the module docstring."""
    if isinstance(obj, ferric.Molecule):
        return _molecule_dict(obj)
    schema = _schema(obj)
    if schema is None and _is_ferric_class(type(obj)):
        raise TypeError(
            f"ferric.serialize has no schema for ferric.{type(obj).__name__}; "
            "add it to RESULT_SCHEMA"
        )
    fields = schema.fields if schema is not None else _reflected_fields(obj)
    accessors = schema.arrays if schema is not None else _ARRAY_ACCESSORS
    out: dict[str, Any] = {"type": type(obj).__name__}
    for name in fields:
        try:
            value = getattr(obj, name)
        except (AttributeError, NotImplementedError):  # noqa: S112 - not a field
            continue
        if callable(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if name == "certificate":
            out[name] = _plain(value)
        elif isinstance(value, _SCALARS):
            out[name] = value
    certificate = getattr(obj, "certificate", None)
    if callable(certificate):
        out["certificate"] = _plain(certificate())
    if "energy" not in out:
        for key in ("total_energy",):
            if isinstance(out.get(key), float):
                out["energy"] = out[key]
                break
    if arrays:
        for name in accessors:
            accessor = getattr(obj, name, None)
            if accessor is None:
                continue
            try:
                value = accessor() if callable(accessor) else accessor
            except (RuntimeError, ValueError, TypeError):  # noqa: S112 - not for this result
                continue
            if isinstance(value, (np.ndarray, list, tuple)):
                out[name] = _plain(np.asarray(value, dtype=float))
    out["ferric_build"] = _build()
    return out


def _null_nonfinite(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _null_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_null_nonfinite(v) for v in obj]
    return obj


def to_json(obj: Any, arrays: bool = False, **json_kw: Any) -> str:
    """``json.dumps(to_dict(obj, arrays))`` with NaN and infinities as ``null``.
    Extra keyword arguments (``indent``...) go to ``json.dumps``."""
    return json.dumps(_null_nonfinite(to_dict(obj, arrays)), allow_nan=False, **json_kw)


def molecule_from_dict(
    d: dict[str, Any], charge: int | None = None, multiplicity: int | None = None
) -> ferric.Molecule:
    """A ``ferric.Molecule`` from ``to_dict(molecule)``. Coordinates are taken as
    Bohr, so the round trip is exact. ``charge`` defaults to ``d["charge"]``
    (else 0) and ``multiplicity`` to ``d["multiplicity"]``; a dict from before
    multiplicity was recorded falls back to 1 for an even electron count and 2
    otherwise."""
    if "symbols" not in d or "coords_bohr" not in d:
        raise ValueError("molecule_from_dict needs 'symbols' and 'coords_bohr'")
    if charge is None:
        charge = int(d.get("charge", 0))
    if multiplicity is None:
        multiplicity = d.get("multiplicity")
    kw = {"units": "bohr", "charge": charge, "ghost": d.get("ghost") or None}
    coords = np.asarray(d["coords_bohr"], dtype=float)
    if multiplicity is None:
        multiplicity = default_multiplicity(d["symbols"], charge, kw["ghost"])
    return ferric.Molecule.from_arrays(
        d["symbols"], coords, multiplicity=int(multiplicity), **kw
    )


def read_run_log(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Records of a ferric CLI JSON Lines run log, in file order.

    A malformed line raises ``ValueError`` naming it, except an unterminated
    LAST line (no trailing newline: a job killed mid-write), which is dropped.
    """
    with open(path, encoding="utf-8") as fh:
        lines = [(i, ln) for i, ln in enumerate(fh, 1) if ln.strip()]
    records = []
    for pos, (num, text) in enumerate(lines):
        try:
            rec = json.loads(text)
        except json.JSONDecodeError as err:
            if pos == len(lines) - 1 and not text.endswith("\n"):
                break
            raise ValueError(f"{path}: line {num} is not valid JSON: {err}") from err
        if not isinstance(rec, dict):
            raise ValueError(f"{path}: line {num} is not a JSON object")  # noqa: TRY004
        records.append(rec)
    return records
