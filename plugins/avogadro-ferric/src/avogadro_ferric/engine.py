"""Thin layer over the ferric Python API, in ferric's units (Hartree, Bohr).

Everything Avogadro-specific (cjson, kJ/mol, Angstrom) lives in the callers.

Which ferric calls are used, and why:

* energy + analytic gradient for all four references (RHF, UHF, RKS, UKS) go
  through ``run_qmmm`` on a QmmmSystem whose every atom is QM. With an empty
  MM region ferric passes no external potential, so this is the gas-phase SCF
  bit for bit (checked against ``run_rhf`` in the tests); it is the one
  Python entry point that returns an analytic gradient for open-shell and
  Kohn-Sham references alike.
* geometry optimisation uses ``run_optimize_qmmm`` (same partition), which
  warm-starts each step's SCF from the previous geometry's orbitals.
* charges, ESP and orbitals need a closed-shell result object
  (``run_rhf`` / ``run_dft``): ferric's charge/ESP/MO evaluators take
  ``RhfResult | DftResult`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import units

ELEMENT_SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
]  # fmt: skip


def ferric():
    """Import ferric lazily, so --print-options etc. never pay for it."""
    import ferric as _ferric

    return _ferric


@dataclass
class Job:
    """One calculation: geometry in Angstrom plus the electronic state."""

    numbers: list[int]
    coords_angstrom: np.ndarray  # (N, 3)
    charge: int = 0
    multiplicity: int = 1
    method: str = "HF"
    basis: str = "sto-3g"

    @property
    def symbols(self) -> list[str]:
        return [ELEMENT_SYMBOLS[z] for z in self.numbers]

    @property
    def is_hf(self) -> bool:
        return self.method.strip().upper() == "HF"

    @property
    def open_shell(self) -> bool:
        return self.multiplicity != 1

    @property
    def reference(self) -> str:
        """run_qmmm's method string."""
        if self.is_hf:
            return "uhf" if self.open_shell else "rhf"
        return "uks" if self.open_shell else "rks"

    @property
    def label(self) -> str:
        prefix = (
            ("U" if self.open_shell else "R")
            if self.is_hf
            else ("UKS " if self.open_shell else "")
        )
        return f"{prefix}{self.method}/{self.basis}"

    def check(self) -> None:
        nelec = sum(self.numbers) - self.charge
        if self.multiplicity < 1:
            raise ValueError(f"multiplicity must be >= 1, got {self.multiplicity}")
        if (nelec - (self.multiplicity - 1)) % 2 != 0 or nelec < self.multiplicity - 1:
            raise ValueError(
                f"{nelec} electrons (charge {self.charge}) cannot have multiplicity "
                f"{self.multiplicity}"
            )

    def xyz(self, coords_angstrom: np.ndarray | None = None) -> str:
        coords = self.coords_angstrom if coords_angstrom is None else coords_angstrom
        lines = [str(len(self.numbers)), "avogadro-ferric"]
        for sym, (x, y, z) in zip(self.symbols, coords):
            lines.append(f"{sym} {x:.12f} {y:.12f} {z:.12f}")
        return "\n".join(lines) + "\n"

    def molecule(self, coords_angstrom: np.ndarray | None = None):
        return ferric().Molecule.from_xyz_string(
            self.xyz(coords_angstrom), self.charge, self.multiplicity
        )

    def qmmm_system(self, coords_angstrom: np.ndarray | None = None):
        coords = self.coords_angstrom if coords_angstrom is None else coords_angstrom
        n = len(self.numbers)
        return ferric().QmmmSystem(
            self.symbols,
            [tuple(float(c) for c in row) for row in coords],
            [0.0] * n,
            qm_indices=list(range(n)),
            charge=self.charge,
            multiplicity=self.multiplicity,
        )


def _with_f64_rij(fn, *args, **kwargs):
    """Call a ferric SCF driver with f64 RI-J storage when it runs Kohn-Sham.

    Kohn-Sham runs RI-J, whose f32 storage default jitters the energy between
    nearby geometries by ~0.3 % of the gradient (the plugin's finite-difference
    test fails on it); ferric's own geometry drivers default to f64 for that
    reason. Single points use it too, so every energy the plugin reports lies
    on one surface. ferric releases before that default (<= 0.1.0rc10) store
    f64 already and have no such keyword.
    """
    if "xc" in kwargs or "functional" in kwargs:
        try:
            return fn(*args, ri_j_precision="f64", **kwargs)
        except TypeError as exc:
            if "ri_j_precision" not in str(exc):
                raise
    return fn(*args, **kwargs)


def energy_gradient(job: Job, coords_angstrom: np.ndarray | None = None):
    """(energy in Hartree, gradient dE/dR in Hartree/Bohr as (N, 3))."""
    f = ferric()
    kwargs = {"method": job.reference}
    if not job.is_hf:
        kwargs["xc"] = job.method
    res = _with_f64_rij(
        f.run_qmmm, job.qmmm_system(coords_angstrom), job.basis, **kwargs
    )
    if not res.converged:
        raise RuntimeError(f"ferric SCF did not converge ({job.label})")
    grad = np.asarray(res.full_gradient(), dtype=float).reshape(len(job.numbers), 3)
    return float(res.energy), grad


def optimize(job: Job, max_steps: int | None = None):
    """Relax the geometry. Returns (coords in Angstrom (N, 3), energy Ha, converged, steps)."""
    f = ferric()
    kwargs = {"method": job.reference, "max_steps": max_steps}
    if not job.is_hf:
        kwargs["xc"] = job.method
    res = f.run_optimize_qmmm(job.qmmm_system(), job.basis, **kwargs)
    coords = np.asarray(res.system().atom_coords_angstrom(), dtype=float)
    return coords, float(res.energy), bool(res.converged), int(res.steps)


def closed_shell_scf(job: Job):
    """(Molecule, BasisSet, RhfResult | DftResult) for the property evaluators."""
    if job.open_shell:
        raise ValueError(
            "ferric's charge, ESP and orbital evaluators take a closed-shell (RHF/RKS) "
            f"result; this molecule has multiplicity {job.multiplicity}"
        )
    f = ferric()
    mol = job.molecule()
    bs = f.BasisSet.bundled(job.basis)
    if job.is_hf:
        res = f.run_rhf(mol, bs)
    else:
        res = _with_f64_rij(f.run_dft, mol, bs, functional=job.method)
    if not res.converged:
        raise RuntimeError(f"ferric SCF did not converge ({job.label})")
    return mol, bs, res


def charges(job: Job, model: str, scf=None) -> list[float]:
    f = ferric()
    mol, bs, res = scf if scf is not None else closed_shell_scf(job)
    fn = {
        "lowdin": f.lowdin_charges,
        "hirshfeld": f.hirshfeld_charges,
        "mulliken": f.mulliken_charges,
        "chelpg": f.chelpg_charges,
        "resp": f.resp_charges,
    }[model]
    return [float(q) for q in fn(mol, bs, res)]


def esp_at_points(job: Job, points_angstrom: np.ndarray, scf=None) -> list[float]:
    """Total (nuclear + electronic) ESP in Hartree/e at points given in Angstrom."""
    f = ferric()
    mol, bs, res = scf if scf is not None else closed_shell_scf(job)
    pts = np.ascontiguousarray(np.asarray(points_angstrom, dtype=float).reshape(-1, 3))
    return [
        float(v) for v in f.esp_at_points(mol, bs, res, pts * units.ANGSTROM_TO_BOHR)
    ]


def frequencies(job: Job):
    """FrequencyResult at the job's geometry (same reference/functional as the energy model)."""
    f = ferric()
    kwargs = {
        "reference": "uhf" if job.open_shell else "rhf",
        "multiplicity": job.multiplicity,
    }
    if not job.is_hf:
        kwargs["xc"] = job.method
    return f.run_frequencies(job.molecule(), job.basis, **kwargs)
