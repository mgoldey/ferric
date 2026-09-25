"""Python bindings for ferric -- a Rust-native quantum chemistry engine."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

# ── Module-level constants ──

DEFAULT_TEMPERATURE_K: float
BOLTZMANN_HARTREE_PER_K: float

# ── Classes ──

class Molecule:
    """A molecular geometry (atoms, charge, multiplicity). Coordinates stored in Bohr; XYZ input in Angstrom."""

    @staticmethod
    def from_xyz(path: str, charge: int = 0, multiplicity: int = 1) -> Molecule:
        """Load a molecule from an XYZ file on disk."""
        ...

    @staticmethod
    def from_xyz_string(s: str, charge: int = 0, multiplicity: int = 1) -> Molecule:
        """Parse a molecule from an XYZ-format string."""
        ...

    def nuclear_repulsion(self) -> float:
        """Classical nuclear repulsion energy in Hartree."""
        ...

    def natoms(self) -> int:
        """Number of atoms."""
        ...

    def nelec(self) -> int:
        """Total electron count (accounts for charge and any ECP core electrons)."""
        ...

    def coords(self) -> list[tuple[float, float, float]]:
        """Cartesian coordinates in Angstrom, one (x, y, z) per atom in symbols() order."""
        ...

    def coords_bohr(self) -> list[tuple[float, float, float]]:
        """Cartesian coordinates in Bohr (the internal values)."""
        ...

    def symbols(self) -> list[str]:
        """Element symbols in atom order."""
        ...

class QmmmSystem:
    """A QM/MM partition: full structure split into a QM region and fixed MM point charges.

    Constructor takes Angstrom; point_charges() returns Bohr (the units run_rhf/run_optimize
    take for point_charges=). Select the QM region with qm_indices OR qm_seeds + qm_radius_angstrom
    (+ optional residue_ids to pull in whole residues instead of individual atoms).
    """

    def __init__(
        self,
        symbols: list[str],
        coords_angstrom: list[tuple[float, float, float]],
        charges: list[float],
        qm_indices: list[int] | None = None,
        qm_seeds: list[int] | None = None,
        qm_radius_angstrom: float | None = None,
        residue_ids: list[int] | None = None,
        charge: int = 0,
        multiplicity: int = 1,
        widths_angstrom: list[float] | None = None,
        polarizabilities_angstrom3: list[float] | None = None,
    ) -> None:
        """widths_angstrom: one Gaussian-smearing width per atom (0.0 = point charge).
        polarizabilities_angstrom3: one isotropic polarisability per atom (0.0 = not a
        polarizable site) for Thole-damped induced-dipole embedding -- see run_qmmm's
        thole_a. Both ignored for atoms in the QM region, like charges."""
        ...

    def with_link_atoms(
        self, bonds: list[tuple[int, int]], scale: float | None = None
    ) -> QmmmSystem:
        """Cap each cut bond with a scaled-position link H (default scale 1.09/1.53). Returns a new system."""
        ...

    def with_boundary_charges(
        self, bonds: list[tuple[int, int]], scheme: str
    ) -> QmmmSystem:
        """Boundary charge scheme for the MM host of each cut bond: "keep", "delete-host", "rc" or "rcd"."""
        ...

    def qm_molecule(self) -> Molecule:
        """The QM region (link hydrogens appended last) as a Molecule."""
        ...

    def point_charges(self) -> list[tuple[float, float, float, float]]:
        """Every POINT embedding charge as (q, x, y, z) in Bohr. Gaussian-smeared
        charges (width > 0) are excluded here -- see smeared_charges()."""
        ...

    def smeared_charges(self) -> list[tuple[float, float, float, float, float]]:
        """Every Gaussian-smeared embedding charge as (q, x, y, z, width_bohr)."""
        ...

    def mm_charge_positions(self) -> list[tuple[float, float, float]]:
        """Positions (Bohr) of every embedding charge, in the SAME canonical
        order as QmmmResult.mm_forces()'s rows: atom-centred charges (point
        and Gaussian-smeared interleaved, ascending full-structure atom
        index) first, then RC/RCD midpoints. NOT the concatenation of
        point_charges() then smeared_charges() unless every charge is one
        kind -- use this order, not those two lists', to interpret
        mm_forces()."""
        ...

    def qm_indices(self) -> list[int]: ...
    def mm_indices(self) -> list[int]: ...
    def qm_atom_count(self) -> int: ...
    def natoms(self) -> int: ...
    def atom_coords_angstrom(self) -> list[tuple[float, float, float]]:
        """Every atom's current position in Angstrom, in full-structure index order
        (the same ordering qm_indices()/mm_indices() index into), regardless of QM/MM role.
        """
        ...

    def link_atom_positions(self) -> list[tuple[float, float, float]]:
        """Link hydrogen positions in Angstrom."""
        ...

    def min_link_to_charge_distance(self) -> float | None:
        """Shortest link-H-to-MM-charge distance in Angstrom (diagnostic), or None."""
        ...

    def boundary_scheme(self) -> str: ...

class MmTopology:
    """Explicit-parameter AMBER-form MM force field topology (ferric-mm). Assigns no
    parameters of its own -- every number is caller-supplied data (see
    tools/active_site/mm_topology.py::topology_from_openmm for one source of that data).
    """

    @staticmethod
    def from_amber_units(
        charges: list[float],
        sigmas_angstrom: list[float],
        epsilons_kcal: list[float],
        bonds: list[tuple[int, int, float, float]],
        angles: list[tuple[int, int, int, float, float]],
        torsions: list[tuple[int, int, int, int, int, float, float]],
    ) -> MmTopology:
        """AMBER-convention units (kcal/mol, Angstrom, degrees), converted once to a.u.

        bonds: (i, j, k_kcal_per_mol_per_ang2, r0_angstrom).
        angles: (i, j, k, k_theta_kcal_per_mol_per_rad2, theta0_degrees).
        torsions: (i, j, k, l, periodicity, k_phi_kcal_per_mol, phase_degrees).
        """
        ...

    def n_atoms(self) -> int: ...

class QmmmResult:
    """Result of run_qmmm. Gradients are dE/dR (Hartree/Bohr); mm_forces() is the FORCE on each charge."""

    @property
    def energy(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def e_pol(self) -> float:
        """Thole-damped polarizable-embedding polarisation energy (Hartree).
        0.0 when no atom carries a nonzero polarisability."""
        ...
    @property
    def mm_energy(self) -> dict[str, float]:
        """MM force-field energy components (Hartree): bond/angle/torsion/lj/coulomb/total.
        All-zero when run_qmmm was not given mm_topology=.
        """
        ...

    def induced_dipoles(self) -> np.ndarray | None:
        """(n_sites, 3) converged Thole-damped induced dipoles (a.u.), in
        QmmmSystem.mm_indices() order filtered to alpha > 0. None when no
        atom was polarizable."""
        ...

    def qm_gradient(self) -> np.ndarray:
        """(n_qm + n_link, 3) dE/dR on the QM molecule as solved."""
        ...

    def mm_forces(self) -> np.ndarray:
        """(n_charges, 3) force on each embedding charge, in
        QmmmSystem.mm_charge_positions() order: atom-centred charges (point
        and Gaussian-smeared interleaved, ascending full-structure atom
        index), then RC/RCD midpoints. NOT QmmmSystem.point_charges() order
        once any charge is Gaussian-smeared -- point_charges() only lists the
        point subset, so zipping it with mm_forces() silently misaligns rows
        whenever a smeared charge is present. Zipping point_charges() with
        mm_forces() is only valid when smeared_charges() == [] (no smeared
        charges at all); otherwise read mm_charge_positions() alongside this
        array."""
        ...

    def full_gradient(self) -> np.ndarray:
        """(natoms_full, 3) dE/dR on every real atom: link rows projected onto hosts, MM forces
        mapped to atoms, PLUS the MM force-field gradient when mm_topology= was given.
        """
        ...

def run_qmmm(
    system: QmmmSystem,
    basis_name: str,
    method: str | None = None,
    xc: str | None = None,
    max_iter: int | None = None,
    energy_conv: float | None = None,
    density_conv: float | None = None,
    level_shift: float | None = None,
    mom_after_iter: int | None = None,
    guess: str | None = None,
    mm_topology: MmTopology | None = None,
    thole_a: float | None = None,
) -> QmmmResult:
    """Embedded SCF ("rhf" default, "uhf", "rks" or "uks") energy + QM gradient + MM forces + full gradient.

    xc is required for "rks"/"uks" and rejected for "rhf"/"uhf".

    mm_topology, if given, adds ferric-mm's AMBER-form force field under the additive QM/MM
    convention (MM-MM bonded/nonbonded + QM-MM Lennard-Jones; no QM-MM Coulomb, already inside
    the embedding). Omitting it is bit-identical to a topology with zero energy/gradient
    everywhere (QmmmResult.mm_energy reports all-zero, not absent).

    thole_a controls Thole damping for polarizable sites (system built with
    polarizabilities_angstrom3=): None/omitted = the standard default 2.1304;
    0.0 disables damping (bare point-dipole tensor); ignored when no atom is
    polarizable. QmmmResult.e_pol/.induced_dipoles() report the result. The
    QM gradient's polarizable Fock-term contribution is included for all
    four methods ("rhf"/"uhf"/"rks"/"uks"). full_gradient() DOES include
    every polarizable force term (a site's own dE_pol/dR_site plus the
    reaction force every embedding charge feels from the other sites'
    induced dipoles) for every method -- verified against finite differences
    on a colocated charge+alpha MM atom and a two-site mutual-induction case.
    Boundary charges (RC/RCD) combined with a polarizable site have no FD
    cross-check yet.
    """
    ...

class QmmmOptimizeResult:
    """Result of run_optimize_qmmm."""

    @property
    def energy(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def steps(self) -> int: ...
    def system(self) -> QmmmSystem:
        """The partition at the final (optimized) geometry."""
        ...

    def energies(self) -> list[float]:
        """Total energy (Hartree) at every step, in order (length steps + 1)."""
        ...

def run_optimize_qmmm(
    system: QmmmSystem,
    basis_name: str,
    method: str | None = None,
    xc: str | None = None,
    move_mm: str | tuple[str, float] | tuple[str, list[int]] | None = None,
    mm_topology: MmTopology | None = None,
    max_steps: int | None = None,
    e_conv: float | None = None,
) -> QmmmOptimizeResult:
    """Optimize a QmmmSystem's geometry. Real QM atoms always move; MM atoms move per move_mm.

    move_mm: "none" (default, only QM atoms move), "all" (every MM atom moves too),
    ("within", radius_angstrom) (MM atoms within that distance of any QM atom, measured once
    at the starting geometry), or ("residues", [residue_id, ...]) (requires the system to have
    been built with residue_ids=; ValueError otherwise). Any value other than "none"/None
    requires mm_topology (ValueError otherwise).

    method/xc follow run_qmmm's convention: "rhf" (default), "uhf", "rks", "uks"; xc is
    required for the KS variants and rejected otherwise.
    """
    ...

class BasisSet:
    """A Gaussian basis set (orbital or auxiliary/RI-fitting)."""

    @staticmethod
    def bundled(name: str) -> BasisSet:
        """Load a bundled basis set by name (e.g. 'sto-3g', 'cc-pvdz-ri')."""
        ...

    @staticmethod
    def from_bse_json(path: str) -> BasisSet:
        """Load a Basis Set Exchange JSON file (contractions renormalised)."""
        ...

class RhfResult:
    """Result of a closed-shell RHF (or run_ksdft KS-DFT) calculation."""

    @property
    def energy(self) -> float:
        """Total SCF energy in Hartree."""
        ...

    @property
    def converged(self) -> bool:
        """Whether the SCF met convergence thresholds."""
        ...

    @property
    def iterations(self) -> int:
        """Number of SCF iterations run."""
        ...

    @property
    def computed_quartets(self) -> int:
        """Number of unique two-electron integral quartets actually computed."""
        ...

    def density(self) -> NDArray[np.float64]:
        """AO-basis density matrix (n_bf x n_bf)."""
        ...

    def orbital_energies(self) -> NDArray[np.float64]:
        """Molecular orbital energies (Hartree), ascending order."""
        ...

    def mo_coefficients(self) -> NDArray[np.float64]:
        """MO coefficient matrix C (n_bf x n_mo), column k = MO k."""
        ...

class UhfResult:
    """Result of an open-shell UHF or ROHF calculation."""

    @property
    def energy(self) -> float:
        """Total SCF energy in Hartree."""
        ...

    @property
    def converged(self) -> bool:
        """Whether the SCF met convergence thresholds."""
        ...

    @property
    def iterations(self) -> int:
        """Number of SCF iterations run."""
        ...

    @property
    def computed_quartets(self) -> int:
        """Number of unique two-electron integral quartets actually computed."""
        ...

    def density_alpha(self) -> NDArray[np.float64]:
        """Alpha-spin AO-basis density matrix (n_bf x n_bf)."""
        ...

    def density_beta(self) -> NDArray[np.float64]:
        """Beta-spin AO-basis density matrix (n_bf x n_bf)."""
        ...

    def orbital_energies_alpha(self) -> NDArray[np.float64]:
        """Alpha-spin orbital energies (Hartree), ascending."""
        ...

    def orbital_energies_beta(self) -> NDArray[np.float64]:
        """Beta-spin orbital energies (Hartree), ascending."""
        ...

class CdftConstraint:
    """One constrained-DFT fragment constraint.

    ``target`` is a Becke fragment POPULATION in electrons, not a net charge:
    ``kind="charge"`` constrains N_alpha + N_beta on the fragment (2.0 on a He
    atom = neutral He, 1.0 = He+); ``kind="spin"`` constrains N_alpha - N_beta.
    ``atoms`` are 0-based, non-empty and unique. An unknown ``kind``, an empty or
    negative/duplicated atom list, or a non-finite target raise ``ValueError``.
    """

    def __init__(
        self, atoms: list[int], target: float, kind: str = "charge"
    ) -> None: ...
    @property
    def atoms(self) -> list[int]: ...
    @property
    def target(self) -> float: ...
    @property
    def kind(self) -> str:
        """``"charge"`` or ``"spin"``."""
        ...

class CdftResult:
    """Result of ``run_cdft`` (constrained UHF/UKS).

    A returned result always has a converged outer (lambda) loop -- an
    unconverged one raises ``RuntimeError`` -- but the inner SCF at the final
    lambda may not be; ``converged`` requires both.
    """

    @property
    def energy(self) -> float:
        """Energy (Ha) at the constrained density, without the constraint term."""
        ...

    @property
    def converged(self) -> bool:
        """``scf_converged`` and every |population - target| < ``lambda_tol``."""
        ...

    @property
    def scf_converged(self) -> bool:
        """Whether the inner SCF at the final lambda converged."""
        ...

    @property
    def iterations(self) -> int:
        """Inner SCF iterations of the final solve."""
        ...

    @property
    def outer_iterations(self) -> int:
        """Outer lambda-Newton iterations."""
        ...

    @property
    def lambdas(self) -> list[float]:
        """Lagrange multipliers (Ha per electron), one per constraint."""
        ...

    @property
    def populations(self) -> list[float]:
        """Achieved fragment populations (electrons), one per constraint."""
        ...

    @property
    def targets(self) -> list[float]:
        """Requested targets, one per constraint."""
        ...

    @property
    def kinds(self) -> list[str]:
        """Constraint kinds (``"charge"``/``"spin"``), one per constraint."""
        ...

    @property
    def max_constraint_error(self) -> float:
        """max |population - target| over constraints (electrons)."""
        ...

    @property
    def lambda_tol(self) -> float:
        """The outer-loop tolerance used."""
        ...

    def density_alpha(self) -> NDArray[np.float64]:
        """Alpha-spin AO-basis density matrix (n_bf x n_bf)."""
        ...

    def density_beta(self) -> NDArray[np.float64]:
        """Beta-spin AO-basis density matrix (n_bf x n_bf)."""
        ...

    def orbital_energies_alpha(self) -> NDArray[np.float64]:
        """Alpha-spin orbital energies (Hartree) of the lambda-augmented Fock."""
        ...

    def orbital_energies_beta(self) -> NDArray[np.float64]:
        """Beta-spin orbital energies (Hartree) of the lambda-augmented Fock."""
        ...

    @property
    def nocc(self) -> tuple[int, int]:
        """Occupied (alpha, beta) orbital counts."""
        ...

    def mo_coeff_alpha(self) -> NDArray[np.float64]:
        """Alpha MO coefficients (n_bf x n_mo); the first nocc[0] columns are occupied."""
        ...

    def mo_coeff_beta(self) -> NDArray[np.float64]:
        """Beta MO coefficients (n_bf x n_mo); the first nocc[1] columns are occupied."""
        ...

    def weight_matrix(self, index: int) -> NDArray[np.float64]:
        """AO-basis Becke weight operator W of constraint ``index``; the population of a
        density pair is trace(W @ (Da + Db)) (charge) or trace(W @ (Da - Db)) (spin)."""
        ...

class CdftCouplingResult:
    """Wu-Van Voorhis coupling between two cDFT diabats (``cdft_coupling``)."""

    @property
    def h_ab(self) -> float:
        """Orthogonalized coupling (Ha). The sign is a phase convention; compare |h_ab|."""
        ...

    @property
    def s_ab(self) -> float:
        """Determinant overlap <Psi_a|Psi_b>."""
        ...

    @property
    def e_a(self) -> float:
        """Energy of state A (Ha)."""
        ...

    @property
    def e_b(self) -> float:
        """Energy of state B (Ha)."""
        ...

class OptimizeResult:
    """Result of a geometry optimization."""

    @property
    def energy(self) -> float:
        """Final optimized energy in Hartree."""
        ...

    @property
    def converged(self) -> bool:
        """Whether the optimization converged."""
        ...

    @property
    def steps(self) -> int:
        """Number of optimization steps."""
        ...

    def mol(self) -> Molecule:
        """The optimized geometry as a new Molecule."""
        ...
    @property
    def energy_trace(self) -> list[float]:
        """Energy at every point the optimizer EVALUATED, in order.

        `energy`, `steps` and `converged` cannot distinguish "ran out of steps
        near a minimum" from "walked uphill and oscillated". MEASURED: a
        diverging embedded optimization shows +0.2045 Ha above its best here
        while reporting the same three scalars as a healthy run. Feed it to
        `tools.viz.energy_plots.optimization_trace`.
        """
        ...

class FrequencyResult:
    """Result of a harmonic vibrational frequency calculation."""

    @property
    def frequencies(self) -> list[float]:
        """Vibrational wavenumbers in cm^-1, ascending. Negative = imaginary."""
        ...

    @property
    def trans_rot_frequencies(self) -> list[float]:
        """Projected-out translation/rotation modes (cm^-1). Should be ~0."""
        ...

    @property
    def is_linear(self) -> bool:
        """Whether the molecule is linear."""
        ...

    @property
    def asymmetry(self) -> float:
        """Largest |H_ij - H_ji| in the raw Cartesian Hessian (Hartree/Bohr^2)."""
        ...

    @property
    def n_gradient_evaluations(self) -> int:
        """Number of gradient evaluations performed."""
        ...

    @property
    def energy(self) -> float:
        """Electronic energy at the undisplaced geometry."""
        ...

class WeightedStats:
    """A weighted mean with its spread."""

    @property
    def mean(self) -> float:
        """Weighted mean."""
        ...

    @property
    def std_dev(self) -> float:
        """Weighted population standard deviation."""
        ...

    @property
    def min(self) -> float:
        """Smallest value across conformers (unweighted)."""
        ...

    @property
    def max(self) -> float:
        """Largest value across conformers (unweighted)."""
        ...

class EnsembleDiagnostics:
    """Population-structure readout for a Boltzmann-weighted ensemble."""

    @property
    def n_conformers(self) -> int: ...
    @property
    def n_within_kt(self) -> int:
        """Conformers within kT of the minimum."""
        ...

    @property
    def n_within_2kt(self) -> int:
        """Conformers within 2kT of the minimum."""
        ...

    @property
    def n_within_5kt(self) -> int:
        """Conformers within 5kT of the minimum."""
        ...

    @property
    def max_weight(self) -> float:
        """Largest single Boltzmann population."""
        ...

    @property
    def max_weight_index(self) -> int:
        """Index of the conformer carrying max_weight."""
        ...

    @property
    def effective_n_conformers(self) -> float:
        """Inverse participation ratio 1 / sum(w_i^2)."""
        ...

    @property
    def temperature_k(self) -> float: ...
    @property
    def verdict(self) -> str:
        """Plain-language verdict on whether the ensemble was needed."""
        ...

    def is_single_conformer_dominated(self, threshold: float = 0.95) -> bool:
        """True when one conformer carries at least threshold of the population."""
        ...

class BoltzmannWeights:
    """Boltzmann populations of an ensemble at one temperature."""

    @property
    def weights(self) -> list[float]:
        """Normalized populations in ensemble order. Sum to 1."""
        ...

    @property
    def relative_energies(self) -> list[float]:
        """Energies relative to the ensemble minimum, in Hartree (all >= 0)."""
        ...

    @property
    def temperature_k(self) -> float: ...
    @property
    def kt_hartree(self) -> float:
        """kT at that temperature, in Hartree."""
        ...

    @property
    def min_index(self) -> int:
        """Index of the lowest-energy conformer."""
        ...

    @property
    def partition_function(self) -> float:
        """Z = sum(exp(-(E_i - E_min)/kT)), always >= 1."""
        ...

    def diagnostics(self) -> EnsembleDiagnostics:
        """Population-structure diagnostics for this weighting."""
        ...

    def __len__(self) -> int: ...

class ConformerEnsemble:
    """A set of conformers of one chemical species."""

    @staticmethod
    def from_coordinates(
        coordinates: list[list[list[float]]],
        elements: Sequence[str] | Sequence[int],
        charge: int = 0,
        multiplicity: int = 1,
        energies: list[float] | None = None,
    ) -> ConformerEnsemble:
        """Build from coordinate arrays (Angstrom) and shared element list."""
        ...

    @staticmethod
    def from_multi_xyz(
        path: str,
        charge: int = 0,
        multiplicity: int = 1,
        energies: list[float] | None = None,
    ) -> ConformerEnsemble:
        """Build from a multi-frame XYZ file."""
        ...

    def __len__(self) -> int: ...
    def n_conformers(self) -> int:
        """Number of conformers (always >= 1)."""
        ...

    def n_atoms(self) -> int:
        """Number of atoms per conformer (shared by construction)."""
        ...

    def molecules(self) -> list[Molecule]:
        """Conformer geometries as Molecule objects."""
        ...

    def molecule(self, index: int) -> Molecule:
        """Geometry of conformer index as a Molecule."""
        ...

    def elements(self) -> list[str]:
        """Element symbols, shared by all conformers."""
        ...

    def atomic_numbers(self) -> list[int]:
        """Atomic numbers, shared by all conformers."""
        ...

    def is_ghost(self) -> list[bool]:
        """Per-atom ghost flags, shared by all conformers."""
        ...

    def coordinates(self) -> list[NDArray[np.float64]]:
        """All conformer geometries as natoms x 3 arrays in Angstrom."""
        ...

    def coordinates_bohr(self) -> list[NDArray[np.float64]]:
        """All conformer geometries as natoms x 3 arrays in Bohr."""
        ...

    def energies(self) -> list[float]:
        """Conformer energies in Hartree, in ensemble order."""
        ...

    def set_energy(self, index: int, energy: float) -> None:
        """Set the energy (Hartree) of one conformer."""
        ...

    def boltzmann_weights(
        self,
        energies: list[float] | None = None,
        temperature_k: float = ...,
    ) -> BoltzmannWeights:
        """Boltzmann weights at temperature_k Kelvin (default 298.15 K)."""
        ...

    def diagnostics(
        self,
        energies: list[float] | None = None,
        temperature_k: float = ...,
    ) -> EnsembleDiagnostics:
        """Population-structure diagnostics at temperature_k."""
        ...

class RiMp2Result:
    """Result of a run_rimp2 calculation."""

    @property
    def total_energy(self) -> float:
        """Reference SCF + MP2 correlation energy, Hartree."""
        ...

    @property
    def rhf_energy(self) -> float:
        """Converged reference SCF energy, Hartree: RHF for a closed-shell
        molecule, UHF for an open-shell one (see `reference`)."""
        ...

    @property
    def mp2_corr(self) -> float:
        """MP2 correlation energy (always negative), Hartree."""
        ...

    @property
    def reference(self) -> str:
        """The SCF reference: "RHF" (closed-shell RI-MP2) or "UHF"
        (unrestricted RI-MP2, for multiplicity > 1)."""
        ...

class OoRiMp2Result:
    """Result of an orbital-optimized RI-MP2 calculation."""

    @property
    def total_energy(self) -> float: ...
    @property
    def hf_energy(self) -> float: ...
    @property
    def mp2_corr(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def grad_norm(self) -> float: ...

class Mp3Result:
    """Result of an MP3 calculation."""

    @property
    def e_hf(self) -> float: ...
    @property
    def e_mp2(self) -> float: ...
    @property
    def e_mp3(self) -> float: ...
    @property
    def e_corr(self) -> float: ...
    @property
    def e_total(self) -> float: ...

class LaplaceMp2Result:
    """Result of a Laplace RI-MP2 calculation."""

    @property
    def total_energy(self) -> float: ...
    @property
    def mp2_corr(self) -> float: ...
    @property
    def e_os(self) -> float: ...
    @property
    def e_ss(self) -> float: ...

class SosMp2Result:
    """Result of a Laplace SOS-MP2 calculation."""

    @property
    def total_energy(self) -> float: ...
    @property
    def rhf_energy(self) -> float: ...
    @property
    def sos_corr(self) -> float:
        """The SCALED correlation energy, c_os * e_os."""
        ...

    @property
    def e_os(self) -> float:
        """The UNSCALED opposite-spin correlation energy."""
        ...

    @property
    def c_os(self) -> float:
        """The c_os actually applied."""
        ...

    @property
    def n_quad(self) -> int:
        """Quadrature points actually used."""
        ...

    @property
    def formulation(self) -> str:
        """'mo' or 'ao', echoed for provenance."""
        ...

class AttenuatedMp2Result:
    """Result of an attenuated RI-MP2 calculation."""

    @property
    def total_energy(self) -> float: ...
    @property
    def rhf_energy(self) -> float: ...
    @property
    def mp2_corr(self) -> float: ...
    @property
    def e_os(self) -> float: ...
    @property
    def e_ss(self) -> float: ...

class ScsMp2Result:
    """Result of a spin-component-scaled MP2 calculation."""

    @property
    def total_energy(self) -> float: ...
    @property
    def rhf_energy(self) -> float: ...
    @property
    def scs_corr(self) -> float: ...
    @property
    def e_os(self) -> float: ...
    @property
    def e_ss(self) -> float: ...

class Mp2VResult:
    """Result of an MP2-V (attenuated MP2 + VV10) calculation."""

    @property
    def total_energy(self) -> float:
        """E_HF + E_c^attMP2 + E_nl^VV10."""
        ...

    @property
    def rhf_energy(self) -> float: ...
    @property
    def att_mp2_corr(self) -> float:
        """Attenuated MP2 correlation energy."""
        ...

    @property
    def vv10_e_nl(self) -> float:
        """VV10 nonlocal correlation energy."""
        ...

    @property
    def e_os(self) -> float: ...
    @property
    def e_ss(self) -> float: ...
    @property
    def n_nlc_points(self) -> int:
        """Grid points in the VV10 nonlocal integration."""
        ...

class RsMp2RpaResult:
    """Result of an RS-MP2-RPA (SR-MP2 + LR-dRPA) calculation."""

    @property
    def total_energy(self) -> float: ...
    @property
    def rhf_energy(self) -> float: ...
    @property
    def e_corr(self) -> float: ...
    @property
    def e_corr_naive(self) -> float | None:
        """Diagnostic naive sum (delta-lr only; None for coupled-rings)."""
        ...

    @property
    def e_mp2_full(self) -> float: ...
    @property
    def e_sr_mp2(self) -> float: ...
    @property
    def e_lr_mp2(self) -> float: ...
    @property
    def e_dmp2_lr(self) -> float: ...
    @property
    def e_drpa_lr(self) -> float | None:
        """E_dRPA[erf] (DeltaLr only; None for CoupledRings)."""
        ...

    @property
    def e_delta_drpa_full(self) -> float | None:
        """Delta-dRPA[Coulomb] (CoupledRings only)."""
        ...

    @property
    def e_delta_drpa_sr(self) -> float | None:
        """Delta-dRPA[erfc] (CoupledRings only)."""
        ...

class DftResult:
    """Result of a KS-DFT calculation."""

    @property
    def total_energy(self) -> float:
        """SCF energy plus the dispersion correction, if one was requested.

        Equals `e_scf` exactly when `dispersion=None`.
        """
        ...

    @property
    def e_scf(self) -> float:
        """The Kohn-Sham SCF energy alone, with no dispersion correction."""
        ...

    @property
    def e_dispersion(self) -> float | None:
        """D3(BJ) dispersion correction in Hartree, or None if not requested.

        `None` means UNEVALUATED, not zero: a DFT energy with no dispersion and
        one whose dispersion is small are different claims.
        """
        ...

    @property
    def converged(self) -> bool: ...
    def vxc(self) -> NDArray[np.float64]:
        """Exchange-correlation potential matrix (n_bf x n_bf)."""
        ...

    def density(self) -> NDArray[np.float64]:
        """AO-basis density matrix (n_bf x n_bf)."""
        ...

    def gradient(self) -> NDArray[np.float64] | None:
        """Analytic nuclear gradient (natoms x 3) if with_gradient=True, else None."""
        ...

class CcResult:
    """Result of a coupled-cluster calculation."""

    @property
    def correlation_energy(self) -> float: ...
    @property
    def t_correction(self) -> float | None: ...

class PdepRpaResult:
    """Result of a PDEP-RPA calculation."""

    @property
    def rhf_energy(self) -> float: ...
    @property
    def e_rpa(self) -> float: ...
    @property
    def total_energy(self) -> float: ...
    @property
    def n_eigenpotentials(self) -> int: ...
    @property
    def e_rpa_dft_diag(self) -> float | None: ...
    @property
    def eigensolver_converged(self) -> bool:
        """Whether the static-dielectric eigensolve met its convergence tolerance."""
        ...

    @property
    def eigenvalues_static(self) -> NDArray[np.float64]:
        """Static dielectric eigenvalues, sorted descending."""
        ...

    @property
    def eigenpotentials(self) -> NDArray[np.float64]:
        """PDEP eigenpotential coefficients (naux, M)."""
        ...

    @property
    def quad_freqs(self) -> NDArray[np.float64]:
        """Imaginary-frequency quadrature points."""
        ...

    @property
    def quad_weights(self) -> NDArray[np.float64]:
        """Imaginary-frequency quadrature weights."""
        ...

    @property
    def eigenvalues_freq(self) -> NDArray[np.float64]:
        """Eigenvalue tensor at imaginary frequencies (N_quad, M)."""
        ...

    def save_scree_plot(self, path: str, title: str | None = None) -> None:
        """Write a scree plot of static dielectric eigenvalues to path (PNG)."""
        ...

class GwResult:
    """Result of a closed-shell GW calculation."""

    @property
    def ref_energy(self) -> float: ...
    @property
    def mo_indices(self) -> list[int]:
        """MO indices for which QP energies were computed."""
        ...

    @property
    def eps_mf(self) -> NDArray[np.float64]:
        """Mean-field orbital energies (Ha)."""
        ...

    @property
    def eps_qp(self) -> NDArray[np.float64]:
        """QP energies (Ha)."""
        ...

    @property
    def sigma_x(self) -> NDArray[np.float64]:
        """Exchange self-energy (Ha)."""
        ...

    @property
    def sigma_c(self) -> NDArray[np.float64]:
        """Correlation self-energy (Ha)."""
        ...

    @property
    def z_factor(self) -> NDArray[np.float64]:
        """Z-factor (renormalization), dimensionless."""
        ...

    @property
    def qp_converged(self) -> list[bool]:
        """Per-state QP Newton-solve convergence flag."""
        ...

    @property
    def n_ev_iter(self) -> int:
        """evGW/evGW0 outer iteration count (0 for G0W0/COHSEX)."""
        ...

    @property
    def outer_converged(self) -> bool:
        """Whether the outer eigenvalue self-consistency loop converged."""
        ...

class UGwResult:
    """Result of an open-shell U-GW calculation."""

    @property
    def ref_energy(self) -> float: ...
    @property
    def mo_indices(self) -> list[int]: ...
    @property
    def eps_mf_a(self) -> NDArray[np.float64]: ...
    @property
    def eps_qp_a(self) -> NDArray[np.float64]: ...
    @property
    def sigma_x_a(self) -> NDArray[np.float64]: ...
    @property
    def sigma_c_a(self) -> NDArray[np.float64]: ...
    @property
    def z_factor_a(self) -> NDArray[np.float64]: ...
    @property
    def eps_mf_b(self) -> NDArray[np.float64]: ...
    @property
    def eps_qp_b(self) -> NDArray[np.float64]: ...
    @property
    def sigma_x_b(self) -> NDArray[np.float64]: ...
    @property
    def sigma_c_b(self) -> NDArray[np.float64]: ...
    @property
    def z_factor_b(self) -> NDArray[np.float64]: ...
    @property
    def qp_converged_a(self) -> list[bool]: ...
    @property
    def qp_converged_b(self) -> list[bool]: ...
    @property
    def n_ev_iter(self) -> int: ...
    @property
    def outer_converged(self) -> bool: ...

class BseResult:
    """Result of a BSE-TDA calculation."""

    @property
    def nocc(self) -> int: ...
    @property
    def nvir(self) -> int: ...
    @property
    def omega(self) -> NDArray[np.float64]:
        """Singlet excitation energies (Hartree), ascending."""
        ...

    @property
    def eps_qp(self) -> NDArray[np.float64]:
        """GW quasiparticle energies used for the diagonal (Ha)."""
        ...

    @property
    def oscillator_strength(self) -> NDArray[np.float64]:
        """Length-gauge oscillator strengths (dimensionless)."""
        ...

    def lowest_ev(self) -> float:
        """Lowest singlet excitation energy in eV."""
        ...

    def lowest_oscillator_strength(self) -> float:
        """Oscillator strength of the lowest singlet."""
        ...

class TdhfStaticPolarizabilityResult:
    """Result of an RPAx@KS static polarizability calculation."""

    @property
    def nocc(self) -> int: ...
    @property
    def nvir(self) -> int: ...
    @property
    def iso(self) -> float:
        """Isotropic average (1/3) Tr(alpha), a.u."""
        ...

    @property
    def tensor(self) -> NDArray[np.float64]:
        """Cartesian alpha_ij(0) tensor (3x3, a.u.)."""
        ...

class TddftResult:
    """Result of a closed-shell linear-response TDA/TDDFT (or CIS/TDHF) calculation."""

    @property
    def n_roots(self) -> int:
        """Number of excitation energies returned."""
        ...

    @property
    def method(self) -> str:
        """Which equations were solved: "Tda" or "Casida"."""
        ...

    @property
    def c_hf(self) -> float:
        """Exact-exchange fraction used on the -c_HF terms (1.0 for HF)."""
        ...

    @property
    def fxc_included(self) -> bool:
        """True when the (ia|f_xc|jb) kernel block was included -- always for a
        KS reference; False only for an HF reference (CIS/TDHF)."""
        ...

    @property
    def excitation_energies(self) -> NDArray[np.float64]:
        """Singlet excitation energies (Hartree), ascending."""
        ...

    @property
    def oscillator_strengths(self) -> NDArray[np.float64]:
        """Length-gauge oscillator strengths (dimensionless), one per root,
        PySCF convention (f = 2/3 * omega * |sqrt(2) sum_ia (X+Y)_ia <i|r|a>|^2)."""
        ...

    def lowest_ev(self) -> float:
        """Lowest excitation energy in eV (0.0 if no roots)."""
        ...

class DoubleHybridResult:
    """Result of an MP2-based double hybrid (B2PLYP / DSD-PBEP86)."""

    @property
    def total_energy(self) -> float:
        """E_KS + scaled MP2 correlation (Hartree)."""
        ...

    @property
    def e_ks(self) -> float:
        """Energy of the double hybrid's own KS reference (Hartree)."""
        ...

    @property
    def e_corr_scaled(self) -> float:
        """c_OS * E_OS + c_SS * E_SS (Hartree)."""
        ...

    @property
    def e_os(self) -> float: ...
    @property
    def e_ss(self) -> float: ...
    @property
    def c_os(self) -> float: ...
    @property
    def c_ss(self) -> float: ...

class BoysResult:
    """Result of Boys localization."""

    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    def c_loc(self) -> NDArray[np.float64]:
        """Localized MO coefficients (n_bf, n_orb)."""
        ...

    def centers(self) -> NDArray[np.float64]:
        """Boys centers <i|r|i> (n_orb, 3) in Bohr."""
        ...

# ── Functions ──

class GammaRhfResult:
    """Result of run_rhf_gamma (closed-shell Gamma-point periodic RHF)."""

    @property
    def timings(self) -> dict[str, Any]:
        """Stage timings and counters (observation only; energies are unaffected):
        {"wall_s": float, "cpu_s": float | None, "unattributed_wall_s": float,
        "stages": {name: {"wall_s": float, "cpu_s": float | None, "calls": int}},
        "counters": {name: int}}. Stages are disjoint, in run order (hcore, J/K
        build, SCF J/K/XC calls, correlation); counters include SR triplets,
        G vectors, chunks and RS-GDF aux dropped."""
        ...
    @property
    def energy(self) -> float:
        """Total energy per cell (Hartree), incl. e_nuc and any Madelung shift."""
        ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def e_nuc(self) -> float:
        """Ewald nuclear repulsion per cell (Hartree)."""
        ...
    @property
    def madelung(self) -> float:
        """Gamma-point Madelung constant (a.u.); applied only for exxdiv='ewald'."""
        ...
    @property
    def exxdiv(self) -> str: ...
    @property
    def omega(self) -> float:
        """Nuclear-attraction Ewald split used, in 1/Angstrom."""
        ...
    @property
    def nao(self) -> int: ...
    @property
    def n_g_half(self) -> int | None:
        """Dense-AFT half-sphere G count; None for jk='rsgdf'."""
        ...
    @property
    def jk(self) -> str:
        """J/K builder used: 'dense' or 'rsgdf'."""
        ...
    @property
    def auxbasis(self) -> str | None:
        """RS-GDF aux basis name; None for jk='dense'."""
        ...
    @property
    def naux(self) -> int | None: ...
    @property
    def naux_kept(self) -> int | None:
        """Aux-metric eigenvectors kept (naux - n_dropped); None for dense."""
        ...
    @property
    def n_dropped(self) -> int | None:
        """Aux-metric eigenvalues <= 1e-10 dropped (lindep); None for dense."""
        ...
    def mo_energy(self) -> NDArray[np.float64]: ...
    def mo_coeff(self) -> NDArray[np.float64]: ...
    def density(self) -> NDArray[np.float64]: ...
    def overlap(self) -> NDArray[np.float64]:
        """Lattice-summed Gamma-point AO overlap."""
        ...

def run_rhf_gamma(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    exxdiv: str = "ewald",
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    density_conv: float = 1e-10,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
) -> GammaRhfResult:
    """Closed-shell Gamma-point periodic RHF.

    jk="dense" (default): the toy-scale dense pure-AFT nao^4 ERI oracle,
    hard-capped by max_eri_gb (default 0.5 GiB). jk="rsgdf": range-separated
    Gaussian density fitting; REQUIRES auxbasis (BasisSet or bundled name,
    e.g. "cc-pvdz-ri"), bounded by memory_budget_gb (None = ferric's unified
    budget). auxbasis/memory_budget_gb with "dense", or max_eri_gb with
    "rsgdf", raise ValueError. lattice: 3x3 rows in Angstrom; omega in
    1/Angstrom; exxdiv "ewald" | "none" and jk (strict). Charged cells, open
    shells and odd (valence) electron counts raise ValueError. An ECP basis
    (e.g. def2-* for Z > 36, *-pp) is applied to the cell's molecule as in
    run_rhf, and the lattice-summed V_ECP enters the periodic hcore.
    """
    ...

# ── Periodic drivers beyond run_rhf_gamma (src/pbc.rs) ──
#
# Shared contract: Molecule coordinates and lattice rows in Angstrom, omega
# in 1/Angstrom, energies in Hartree per cell. jk="dense" (toy-scale AFT
# oracle, capped by max_eri_gb) | "rsgdf" (REQUIRES auxbasis, bounded by
# memory_budget_gb); a knob the chosen path ignores is a ValueError. Charged
# cells raise ValueError. ECP bases are supported by every driver (ECP applied
# to the cell's molecule as in run_rhf; lattice-summed V_ECP in the periodic
# hcore; electron counts and frozen_core are valence counts); Rust-side
# refusals raise ValueError with their message; numerical failures (incl. SCF
# non-convergence) raise RuntimeError.

class GammaOpenShellResult:
    """Result of run_uhf_gamma / run_rohf_gamma / run_uks_gamma / run_roks_gamma."""

    @property
    def timings(self) -> dict[str, Any]:
        """Stage timings and counters (observation only; energies are unaffected):
        {"wall_s": float, "cpu_s": float | None, "unattributed_wall_s": float,
        "stages": {name: {"wall_s": float, "cpu_s": float | None, "calls": int}},
        "counters": {name: int}}. Stages are disjoint, in run order (hcore, J/K
        build, SCF J/K/XC calls, correlation); counters include SR triplets,
        G vectors, chunks and RS-GDF aux dropped."""
        ...
    @property
    def method(self) -> str:
        """'uhf', 'rohf', 'uks' or 'roks'."""
        ...
    @property
    def functional(self) -> str | None: ...
    @property
    def energy(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def e_nuc(self) -> float: ...
    @property
    def madelung(self) -> float: ...
    @property
    def exxdiv(self) -> str: ...
    @property
    def ewald_start(self) -> str | None:
        """'staged' | 'direct' for exxdiv='ewald'; None for 'none'."""
        ...
    @property
    def none_stage_energy(self) -> float | None: ...
    @property
    def nalpha(self) -> int: ...
    @property
    def nbeta(self) -> int: ...
    @property
    def s2(self) -> float:
        """<S^2> with the lattice overlap."""
        ...
    @property
    def gap_alpha(self) -> float | None: ...
    @property
    def gap_beta(self) -> float | None: ...
    @property
    def gaps_satisfied(self) -> bool:
        """Every per-spin gap >= the applied Madelung shift (Ewald-trap check)."""
        ...
    @property
    def e_xc(self) -> float | None: ...
    @property
    def exact_exchange_fraction(self) -> float | None: ...
    @property
    def n_grid_points(self) -> int | None: ...
    @property
    def electrons_on_grid(self) -> float | None: ...
    @property
    def nao(self) -> int: ...
    @property
    def jk(self) -> str: ...
    @property
    def auxbasis(self) -> str | None: ...
    def mo_energy_alpha(self) -> NDArray[np.float64]: ...
    def mo_energy_beta(self) -> NDArray[np.float64] | None:
        """None for ROHF/ROKS (one MO set)."""
        ...
    def density(self) -> NDArray[np.float64]: ...

class GammaRksResult:
    """Result of run_rks_gamma (closed-shell Gamma-point periodic RKS)."""

    @property
    def timings(self) -> dict[str, Any]:
        """Stage timings and counters (observation only; energies are unaffected):
        {"wall_s": float, "cpu_s": float | None, "unattributed_wall_s": float,
        "stages": {name: {"wall_s": float, "cpu_s": float | None, "calls": int}},
        "counters": {name: int}}. Stages are disjoint, in run order (hcore, J/K
        build, SCF J/K/XC calls, correlation); counters include SR triplets,
        G vectors, chunks and RS-GDF aux dropped."""
        ...
    @property
    def functional(self) -> str: ...
    @property
    def energy(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def e_nuc(self) -> float: ...
    @property
    def madelung(self) -> float: ...
    @property
    def exxdiv(self) -> str: ...
    @property
    def e_xc(self) -> float: ...
    @property
    def exact_exchange_fraction(self) -> float: ...
    @property
    def n_grid_points(self) -> int: ...
    @property
    def neighbour_cutoff(self) -> float:
        """Neighbour cutoff used, Angstrom."""
        ...
    @property
    def electrons_on_grid(self) -> float: ...
    @property
    def nao(self) -> int: ...
    @property
    def jk(self) -> str: ...
    @property
    def auxbasis(self) -> str | None: ...
    def mo_energy(self) -> NDArray[np.float64]: ...
    def density(self) -> NDArray[np.float64]: ...

class GammaCorrelationResult:
    """Result of run_mp2_gamma / run_drpa_gamma (Gamma RHF + correlation)."""

    @property
    def timings(self) -> dict[str, Any]:
        """Stage timings and counters (observation only; energies are unaffected):
        {"wall_s": float, "cpu_s": float | None, "unattributed_wall_s": float,
        "stages": {name: {"wall_s": float, "cpu_s": float | None, "calls": int}},
        "counters": {name: int}}. Stages are disjoint, in run order (hcore, J/K
        build, SCF J/K/XC calls, correlation); counters include SR triplets,
        G vectors, chunks and RS-GDF aux dropped."""
        ...
    @property
    def method(self) -> str:
        """'mp2' or 'drpa'."""
        ...
    @property
    def energy(self) -> float:
        """e_scf + correlation_energy."""
        ...
    @property
    def e_scf(self) -> float: ...
    @property
    def correlation_energy(self) -> float: ...
    @property
    def e_os(self) -> float | None: ...
    @property
    def e_ss(self) -> float | None: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def madelung(self) -> float: ...
    @property
    def occ_shift(self) -> float: ...
    @property
    def nocc_active(self) -> int: ...
    @property
    def nvir(self) -> int: ...
    @property
    def naux(self) -> int | None: ...
    @property
    def quad_points(self) -> int | None: ...
    @property
    def exxdiv(self) -> str: ...
    @property
    def denominators(self) -> str: ...
    @property
    def jk(self) -> str: ...
    @property
    def auxbasis(self) -> str | None: ...

class KpointScfResult:
    """Result of run_rhf_kpts / run_uhf_kpts (energies per cell)."""

    @property
    def timings(self) -> dict[str, Any]:
        """Stage timings and counters (observation only; energies are unaffected):
        {"wall_s": float, "cpu_s": float | None, "unattributed_wall_s": float,
        "stages": {name: {"wall_s": float, "cpu_s": float | None, "calls": int}},
        "counters": {name: int}}. Stages are disjoint, in run order (hcore, J/K
        build, SCF J/K/XC calls, correlation); counters include SR triplets,
        G vectors, chunks and RS-GDF aux dropped."""
        ...
    @property
    def method(self) -> str: ...
    @property
    def energy(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def e_nuc(self) -> float: ...
    @property
    def madelung(self) -> float:
        """Mesh (supercell) Madelung constant."""
        ...
    @property
    def exxdiv(self) -> str: ...
    @property
    def ewald_start(self) -> str | None: ...
    @property
    def none_stage_energy(self) -> float | None: ...
    @property
    def mesh(self) -> tuple[int, int, int]: ...
    @property
    def centring(self) -> str: ...
    @property
    def nk(self) -> int: ...
    @property
    def kpts(self) -> list[list[float]]:
        """Cartesian k-points, 1/Angstrom."""
        ...
    @property
    def mo_energy(self) -> list[list[float]]: ...
    @property
    def mo_energy_beta(self) -> list[list[float]] | None: ...
    @property
    def homo(self) -> float | None: ...
    @property
    def lumo(self) -> float | None: ...
    @property
    def nalpha(self) -> int | None: ...
    @property
    def nbeta(self) -> int | None: ...
    @property
    def s2(self) -> float | None:
        """UHF: <S^2> of the giant (supercell) determinant, not per cell."""
        ...
    @property
    def gap_alpha(self) -> float | None: ...
    @property
    def gap_beta(self) -> float | None: ...
    @property
    def lindep_threshold(self) -> float: ...
    @property
    def lindep_min_kept(self) -> int: ...
    @property
    def lindep_max_kept(self) -> int: ...
    @property
    def lindep_total_kept(self) -> int: ...
    @property
    def lindep_near_noise_floor(self) -> bool: ...
    @property
    def nao(self) -> int: ...
    @property
    def jk(self) -> str: ...
    @property
    def auxbasis(self) -> str | None: ...

class KpointCorrelationResult:
    """Result of run_mp2_kpts / run_drpa_kpts (k-point RHF + correlation)."""

    @property
    def timings(self) -> dict[str, Any]:
        """Stage timings and counters (observation only; energies are unaffected):
        {"wall_s": float, "cpu_s": float | None, "unattributed_wall_s": float,
        "stages": {name: {"wall_s": float, "cpu_s": float | None, "calls": int}},
        "counters": {name: int}}. Stages are disjoint, in run order (hcore, J/K
        build, SCF J/K/XC calls, correlation); counters include SR triplets,
        G vectors, chunks and RS-GDF aux dropped."""
        ...
    @property
    def method(self) -> str: ...
    @property
    def energy(self) -> float: ...
    @property
    def e_scf(self) -> float: ...
    @property
    def correlation_energy(self) -> float: ...
    @property
    def e_os(self) -> float | None: ...
    @property
    def e_ss(self) -> float | None: ...
    @property
    def e_direct(self) -> float | None: ...
    @property
    def per_q(self) -> list[float] | None: ...
    @property
    def drpa_energy(self) -> str | None: ...
    @property
    def quad_points(self) -> int | None: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def madelung(self) -> float: ...
    @property
    def occ_shift(self) -> float: ...
    @property
    def nocc_active(self) -> int: ...
    @property
    def nvir(self) -> list[int]: ...
    @property
    def naux(self) -> list[int] | None: ...
    @property
    def mesh(self) -> tuple[int, int, int]: ...
    @property
    def centring(self) -> str: ...
    @property
    def nk(self) -> int: ...
    @property
    def exxdiv(self) -> str: ...
    @property
    def denominators(self) -> str: ...
    @property
    def lindep_total_kept(self) -> int: ...
    @property
    def lindep_min_kept(self) -> int: ...
    @property
    def lindep_near_noise_floor(self) -> bool: ...
    @property
    def jk(self) -> str: ...
    @property
    def auxbasis(self) -> str | None: ...

def run_uhf_gamma(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    exxdiv: str = "ewald",
    ewald_start: str | None = None,
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    density_conv: float = 1e-10,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
) -> GammaOpenShellResult:
    """Gamma-point periodic UHF. ewald_start "staged" (default) | "direct",
    exxdiv="ewald" only (ValueError with "none")."""
    ...

def run_rohf_gamma(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    exxdiv: str = "ewald",
    ewald_start: str | None = None,
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    density_conv: float = 1e-10,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
) -> GammaOpenShellResult:
    """Gamma-point periodic ROHF. KNOWN LIMITATION: does not converge on the
    triclinic 4H s+p triplet (non-convergence is an error, not a number)."""
    ...

def run_uks_gamma(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    functional: str,
    exxdiv: str = "ewald",
    ewald_start: str | None = None,
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    density_conv: float = 1e-10,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
    n_radial: int = 75,
    n_angular: int = 302,
    neighbour_cutoff: float | None = None,
) -> GammaOpenShellResult:
    """Gamma-point periodic UKS (LDA/GGA/global hybrids; RSH/meta-GGA/VV10
    refused). neighbour_cutoff in Angstrom (None = max(10 Bohr, covering
    bound))."""
    ...

def run_roks_gamma(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    functional: str,
    exxdiv: str = "ewald",
    ewald_start: str | None = None,
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int | None = None,
    density_conv: float = 1e-10,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
    n_radial: int = 75,
    n_angular: int = 302,
    neighbour_cutoff: float | None = None,
) -> GammaOpenShellResult:
    """Gamma-point periodic ROKS (functional/grid contract of run_uks_gamma).
    max_iter=None: 600 with a 0.05 Ha ramped level shift for a hybrid (a > 0),
    else 200 and no shift."""
    ...

def run_rks_gamma(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    functional: str,
    exxdiv: str = "ewald",
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    density_conv: float = 1e-10,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
    n_radial: int = 75,
    n_angular: int = 302,
    neighbour_cutoff: float | None = None,
) -> GammaRksResult:
    """Closed-shell Gamma-point periodic RKS; open shells raise ValueError."""
    ...

def run_mp2_gamma(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    exxdiv: str,
    denominators: str,
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    density_conv: float = 1e-10,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
    frozen_core: int = 0,
) -> GammaCorrelationResult:
    """Gamma RHF + MP2 in one call. exxdiv (the reference's) and denominators
    ("shifted" | "unshifted") are required."""
    ...

def run_drpa_gamma(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    exxdiv: str,
    denominators: str,
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    density_conv: float = 1e-10,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
    frozen_core: int = 0,
    quad_points: int | None = None,
) -> GammaCorrelationResult:
    """Gamma RHF + dRPA in one call. jk="dense" is the exact plasmon formula
    (quad_points there is a ValueError); jk="rsgdf" uses frequency quadrature
    (quad_points, None = 40)."""
    ...

def run_rhf_kpts(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    mesh: tuple[int, int, int],
    exxdiv: str = "ewald",
    centring: str = "gamma",
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    energy_conv: float = 1e-12,
    grad_conv: float = 1e-9,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
) -> KpointScfResult:
    """Closed-shell k-point RHF. centring "gamma" | "mp" (strict)."""
    ...

def run_uhf_kpts(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    mesh: tuple[int, int, int],
    exxdiv: str = "ewald",
    centring: str = "gamma",
    ewald_start: str | None = None,
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    energy_conv: float = 1e-12,
    grad_conv: float = 1e-9,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
) -> KpointScfResult:
    """k-point UHF; s2 is the giant (supercell) determinant's <S^2>."""
    ...

def run_mp2_kpts(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    mesh: tuple[int, int, int],
    exxdiv: str,
    denominators: str,
    centring: str = "gamma",
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    energy_conv: float = 1e-12,
    grad_conv: float = 1e-9,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
    frozen_core: int = 0,
) -> KpointCorrelationResult:
    """k-point RHF + KMP2 in one call (exxdiv and denominators required)."""
    ...

def run_drpa_kpts(
    mol: Molecule,
    lattice: Sequence[Sequence[float]],
    basis_set: BasisSet,
    mesh: tuple[int, int, int],
    exxdiv: str,
    denominators: str,
    centring: str = "gamma",
    omega: float | None = None,
    max_eri_gb: float | None = None,
    max_iter: int = 200,
    energy_conv: float = 1e-12,
    grad_conv: float = 1e-9,
    jk: str = "dense",
    auxbasis: BasisSet | str | None = None,
    memory_budget_gb: float | None = None,
    frozen_core: int = 0,
    energy: str = "quadrature",
    quad_points: int | None = None,
) -> KpointCorrelationResult:
    """k-point RHF + k-dRPA. energy "quadrature" (default) | "plasmon" |
    "second-order"; quad_points only with "quadrature"."""
    ...

def run_rhf(
    mol: Molecule,
    basis_set: BasisSet,
    max_iter: int | None = None,
    energy_conv: float | None = None,
    density_conv: float | None = None,
    diis_size: int | None = None,
    integral_thresh: float | None = None,
    k_builder: str | None = None,
    df_j_aux: str | None = None,
    df_k_aux: str | None = None,
    level_shift: float | None = None,
    mom_after_iter: int | None = None,
    guess: str | None = None,
    diis: str | None = None,
    smearing_sigma: float | None = None,
    soscf: bool | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
    smeared_charges: list[tuple[float, float, float, float, float]] | None = None,
    memory_budget_gb: float | None = None,
    solvent: float | str | None = None,
    pcm_lebedev_order: int | None = None,
) -> RhfResult:
    """Closed-shell Restricted Hartree-Fock.

    df_j_aux / df_k_aux: an aux basis name selects RI-J / RI-K; "" / "exact" /
    "none" / "off" / "conventional" select conventional four-centre integrals
    (the same spellings run_dft accepts, also for run_uhf / run_rohf).

    solvent: IEF-PCM implicit solvation, as a dielectric constant (> 1.0) or a
    solvent name (water, dmso, methanol, ethanol, acetone, dichloromethane,
    thf, chloroform, toluene, hexane). None = vacuum.
    pcm_lebedev_order: tesserae per atomic sphere (6/14/26/50/110/302).

    smeared_charges: list of (q, x, y, z, width) Gaussian-smeared classical
    charges (Bohr / Hartree atomic units).
    """
    ...

def run_uhf(
    mol: Molecule,
    basis_set: BasisSet,
    max_iter: int | None = None,
    energy_conv: float | None = None,
    density_conv: float | None = None,
    diis_size: int | None = None,
    integral_thresh: float | None = None,
    k_builder: str | None = None,
    df_j_aux: str | None = None,
    df_k_aux: str | None = None,
    level_shift: float | None = None,
    mom_after_iter: int | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
    memory_budget_gb: float | None = None,
    guess: str | None = None,
    stability_descent: bool | None = None,
) -> UhfResult:
    """Unrestricted Hartree-Fock (open-shell).

    ``guess`` is ``"minao"`` (default; ``"sad"`` is an alias) or ``"hcore"``. ``stability_descent=True``
    checks internal stability and follows a downhill orbital-Hessian mode off a
    saddle (e.g. O2 triplet/STO-3G, whose default-guess solution is a saddle
    1.33 mHa above the UHF minimum).
    """
    ...

def run_rohf(
    mol: Molecule,
    basis_set: BasisSet,
    max_iter: int | None = None,
    energy_conv: float | None = None,
    density_conv: float | None = None,
    diis_size: int | None = None,
    integral_thresh: float | None = None,
    k_builder: str | None = None,
    df_j_aux: str | None = None,
    df_k_aux: str | None = None,
    level_shift: float | None = None,
    mom_after_iter: int | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
    memory_budget_gb: float | None = None,
) -> UhfResult:
    """Restricted Open-Shell Hartree-Fock."""
    ...

def run_cdft(
    mol: Molecule,
    basis_set: BasisSet,
    constraints: list[CdftConstraint],
    functional: str | None = None,
    lambda_tol: float | None = None,
    max_outer: int | None = None,
    stability_descent: bool | None = None,
    max_iter: int | None = None,
    energy_conv: float | None = None,
    density_conv: float | None = None,
    diis_size: int | None = None,
    integral_thresh: float | None = None,
    k_builder: str | None = None,
    df_j_aux: str | None = None,
    df_k_aux: str | None = None,
    level_shift: float | None = None,
    mom_after_iter: int | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
    memory_budget_gb: float | None = None,
    guess: str | None = None,
    grid_radial: int | None = None,
    grid_angular: int | None = None,
) -> CdftResult:
    """Constrained UHF/UKS (Wu-Van Voorhis cDFT) with Becke fragment populations.

    ``functional`` None or ``"HF"`` = UHF (the validated path); any other name = UKS.
    ``lambda_tol`` (default 1e-5 electrons) and ``max_outer`` (default 30) control the
    outer lambda-Newton loop; exceeding ``max_outer`` raises ``RuntimeError``.
    ``stability_descent`` defaults to True (unlike ``run_uhf``): a constrained saddle is
    descended from and the lower constraint-satisfying state kept (skipped for UKS).
    The weight grid defaults to 99 x 302. Unset SCF knobs take the Rust ``RhfConfig``
    defaults (``max_iter`` 200); ``df_*_aux`` unset = exact J/K.
    """
    ...

def cdft_coupling(state_a: CdftResult, state_b: CdftResult) -> CdftCouplingResult:
    """Wu-Van Voorhis H_ab between two ``run_cdft`` diabats.

    Each state must carry exactly one ``kind="charge"`` constraint, be ``converged``,
    and come from the same molecule/geometry/basis/charge/multiplicity; otherwise
    ``ValueError``. Identical states (|S_ab| -> 1) also raise.
    """
    ...

def run_optimize(
    mol: Molecule,
    basis_name: str,
    max_steps: int | None = None,
    e_conv: float | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
) -> OptimizeResult:
    """Geometry optimization (RHF)."""
    ...

def run_frequencies(
    mol: Molecule,
    basis_name: str,
    reference: str | None = None,
    xc: str | None = None,
    delta: float | None = None,
    multiplicity: int | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
) -> FrequencyResult:
    """Harmonic vibrational frequencies via finite-difference of analytic gradients.

    `point_charges` ((q, x, y, z) in Bohr) and `external_field` embed the QM
    region in an MM field, the same way `run_optimize` does.
    """
    ...

class SaddleResult:
    """A first-order saddle point found by P-RFO."""

    @property
    def energy(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def steps(self) -> int: ...
    @property
    def n_imaginary(self) -> int: ...
    @property
    def lowest_eigenvalue(self) -> float: ...
    @property
    def symbols(self) -> list[str]: ...
    @property
    def coords(self) -> list[tuple[float, float, float]]: ...
    @property
    def imaginary_mode(self) -> list[float] | None:
        """The followed mode as a flat 3N Cartesian displacement vector.

        **`None` unless `n_imaginary == 1`.** A gradient-converged point with
        zero or two imaginary modes is not a transition state and has no single
        mode to follow, so there is nothing to return. Check
        `is_transition_state()` before passing this to `run_irc`, which needs a
        real vector -- the typed `| None` is what stops a checker accepting
        `run_irc(mode=result.imaginary_mode)` unguarded.
        """
        ...

    def is_transition_state(self) -> bool:
        """Gradient convergence AND exactly one imaginary mode.

        A method rather than a property on purpose: convergence alone is
        satisfied by every stationary point, so reading `converged` as "this is
        a transition state" is the mistake this guards.
        """
        ...

class IrcBranch:
    """One direction of an intrinsic reaction coordinate walk."""

    @property
    def energy(self) -> float: ...
    @property
    def converged(self) -> bool:
        """True when the walk reached a flat region, NOT that the endpoint is a
        minimum -- confirming that needs a Hessian there."""
        ...
    @property
    def steps(self) -> int: ...
    @property
    def symbols(self) -> list[str]: ...
    @property
    def coords(self) -> list[tuple[float, float, float]]: ...

class IrcResult:
    """Both branches of an IRC, and the barriers they imply."""

    @property
    def saddle_energy(self) -> float: ...
    @property
    def forward(self) -> IrcBranch: ...
    @property
    def reverse(self) -> IrcBranch: ...
    def forward_barrier(self) -> float: ...
    def reverse_barrier(self) -> float: ...
    def both_converged(self) -> bool: ...

def run_saddle(
    mol: Molecule,
    basis_name: str,
    xc: str | None = None,
    multiplicity: int | None = None,
    max_steps: int | None = None,
    trust_radius: float | None = None,
    follow_mode: int | None = None,
    delta: float | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
) -> SaddleResult:
    """Partitioned rational function optimization to a first-order saddle.

    Closed-shell references only. Starting with no negative projected Hessian
    eigenvalue is a hard error naming that eigenvalue, rather than a converged
    minimum labelled as a transition state.
    """
    ...

def run_irc(
    mol: Molecule,
    basis_name: str,
    mode: list[float],  # NOT `| None`: see SaddleResult.imaginary_mode
    xc: str | None = None,
    multiplicity: int | None = None,
    step: float | None = None,
    max_steps: int | None = None,
    g_max_thresh: float | None = None,
    initial_displacement: float | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
) -> IrcResult:
    """Follow the intrinsic reaction coordinate both ways from a saddle.

    `mode` is the imaginary-mode VECTOR (3N, e.g. `SaddleResult.imaginary_mode`),
    not an index. Walk it in the SAME field the saddle was found in: a
    gas-phase walk from an embedded saddle descends a different surface.
    """
    ...

def run_rimp2(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
    kappa: float | None = None,
) -> RiMp2Result:
    """Resolution-of-identity (density-fitted) MP2.

    RHF reference for a singlet; UHF reference + unrestricted RI-MP2 (UMP2)
    for multiplicity > 1 (`result.reference` says which). `kappa` is
    closed-shell only and raises ValueError on an open-shell molecule.
    """
    ...

def run_oo_rimp2(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    max_iter: int | None = None,
    grad_conv: float | None = None,
    level_shift: float | None = None,
    diis_size: int | None = None,
    memory_budget_gb: float | None = None,
) -> OoRiMp2Result:
    """Orbital-optimized RI-MP2."""
    ...

def run_mp3(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> Mp3Result:
    """Spin-orbital MP3 via einsum."""
    ...

def run_laplace_mp2(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    n_quad: int | None = None,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> LaplaceMp2Result:
    """Laplace-transform RI-MP2."""
    ...

def run_laplace_sos_mp2(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    c_os: float | None = None,
    n_quad: int | None = None,
    frozen_core: int | None = None,
    formulation: str | None = None,
    domain_cutoff_bohr: float | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> SosMp2Result:
    """Laplace-transform SOS-MP2: E = c_os * E_OS."""
    ...

def run_attenuated_rimp2(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    omega: float | None = None,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> AttenuatedMp2Result:
    """Attenuated RI-MP2 (erfc(omega*r)/r). omega in Angstrom^-1."""
    ...

def run_terfc_rimp2(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    r0: float | None = None,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> RiMp2Result:
    """MP2(terfc): RI-MP2 with exact tempered-erfc operator. r0 in Angstrom."""
    ...

def run_scs_mp2(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    c_os: float | None = None,
    c_ss: float | None = None,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> ScsMp2Result:
    """Spin-component-scaled MP2."""
    ...

def run_scs_mp2_2terfc(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    r0_bonded: float | None = None,
    r0_nonbonded: float | None = None,
    c_os: float | None = None,
    c_ss: float | None = None,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> ScsMp2Result:
    """SCS-MP2(2terfc): dual-attenuated SCS-MP2 with exact terfc. r0 in Angstrom."""
    ...

def run_mp2_v(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    r0: float | None = None,
    b: float | None = None,
    c: float | None = None,
    omega: float | None = None,
    attenuator: str | None = None,
    vv10_damping: str | None = None,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> Mp2VResult:
    """MP2-V: attenuated MP2 + Eq-11-damped VV10. r0 in Angstrom, omega in Angstrom^-1."""
    ...

def run_rs_mp2_rpa(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    omega: float | None = None,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    formulation: str | None = None,
    attenuator: str | None = None,
    r0: float | None = None,
    terf_omega: float | None = None,
    memory_budget_gb: float | None = None,
) -> RsMp2RpaResult:
    """SR-MP2 + LR-dRPA. omega/terf_omega in Angstrom^-1, r0 in Angstrom."""
    ...

def run_dft(
    mol: Molecule,
    basis_set: BasisSet,
    functional: str | None = None,
    k_builder: str | None = None,
    with_gradient: bool = False,
    max_iter: int | None = None,
    energy_conv: float | None = None,
    density_conv: float | None = None,
    level_shift: float | None = None,
    mom_after_iter: int | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
    memory_budget_gb: float | None = None,
    dispersion: str | None = None,
    df_j_aux: str | None = None,
    df_k_aux: str | None = None,
    grid_radial: int | None = None,
    grid_angular: int | None = None,
    grid_prune: str | None = None,
) -> DftResult:
    """Kohn-Sham DFT (closed-shell).

    `dispersion` adds an empirical dispersion correction to the SCF energy:
    `"d3bj"` uses the damping parameters published for `functional`, and
    `"d3bj(<name>)"` uses `<name>`'s instead. `None` (the default) applies no
    correction and leaves the energy exactly as it was. Any other value raises
    -- there is no spelling that means "compute a zero correction".

    `grid_radial` / `grid_angular` / `grid_prune` set the main XC grid. All
    `None` (the default) is the 75x110 unpruned grid, unchanged. `grid_angular`
    must be a supported Lebedev order (6, 14, 26, 50, 110, 302).
    `grid_prune="nwchem"` applies NWChem-style radial-region angular pruning
    (~23% fewer points at 75x110, ~1e-10 Ha); `"none"` is the flat grid, and
    any other value raises ValueError. Pruning has no table at
    `grid_angular=50`. Any grid kwarg with `with_gradient=True` raises
    ValueError: the analytic gradient is built on the default grid.
    `k_builder="cosx"` with `with_gradient=True` also raises ValueError: COSX
    has no analytic gradient, and the exact-exchange gradient is not the
    derivative of a COSX energy.
    """
    ...

def dft_grid_point_count(
    mol: Molecule,
    grid_radial: int | None = None,
    grid_angular: int | None = None,
    grid_prune: str | None = None,
) -> int:
    """Number of points in the main XC grid `run_dft` builds with these kwargs."""
    ...

def run_ksdft(
    mol: Molecule,
    basis_set: BasisSet,
    functional: str | None = None,
    k_builder: str | None = None,
    with_gradient: bool = False,
    max_iter: int | None = None,
    energy_conv: float | None = None,
    density_conv: float | None = None,
    level_shift: float | None = None,
    mom_after_iter: int | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
    memory_budget_gb: float | None = None,
    dispersion: str | None = None,
    df_j_aux: str | None = None,
    df_k_aux: str | None = None,
    grid_radial: int | None = None,
    grid_angular: int | None = None,
    grid_prune: str | None = None,
) -> DftResult:
    """Kohn-Sham DFT (closed-shell). Alias of run_dft (same grid_* kwargs).

    `dispersion` adds an empirical dispersion correction to the SCF energy:
    `"d3bj"` uses the damping parameters published for `functional`, and
    `"d3bj(<name>)"` uses `<name>`'s instead. `None` (the default) applies no
    correction and leaves the energy exactly as it was. Any other value raises
    -- there is no spelling that means "compute a zero correction".
    """
    ...

def d3bj_energy(mol: Molecule, functional: str) -> float:
    """Grimme D3(BJ) dispersion energy in Hartree (two-body term).

    `functional` names the XC functional whose published D3(BJ) damping
    parameters to use; the correction is fitted per functional, so this is
    required. Raises for an unknown functional or for an element outside the
    D3 parameterisation, rather than silently returning a smaller number.
    """
    ...

def run_ccd(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> CcResult:
    """Coupled-cluster doubles (CCD)."""
    ...

def run_ccsd(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> CcResult:
    """Coupled-cluster singles and doubles (CCSD), spin-adapted."""
    ...

def run_ccsd_t(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> CcResult:
    """CCSD(T): CCSD plus perturbative triples correction, spin-adapted."""
    ...

def run_pdep_rpa(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    frozen_core: int | None = None,
    n_quad: int | None = None,
    quadrature: str | None = None,
    u0: float | None = None,
    trunc_thresh: float | None = None,
    eigensolver_conv_thresh: float | None = None,
    run_diagnostics: bool = False,
    k_builder: str | None = None,
    chi0_sparsity: str | None = None,
    memory_budget_gb: float | None = None,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    external_field: tuple[float, float, float] | None = None,
    solvent: float | str | None = None,
    pcm_lebedev_order: int | None = None,
) -> PdepRpaResult:
    """PDEP-RPA (dielectric eigendecomposition RPA).

    point_charges / external_field / solvent embed the REFERENCE RHF, so the
    dielectric response is that of the molecule in its environment. Units match
    run_rhf: point_charges are (q, x, y, z) with q in e and positions in BOHR
    (Molecule.coords() is Angstrom -- convert first); external_field is
    (Ex, Ey, Ez) in atomic units (Hartree / (e * Bohr)). solvent and
    pcm_lebedev_order are as in run_rhf.
    """
    ...

def run_gw(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    method: str | None = None,
    xc: str | None = None,
    qp_mos: tuple[int, int] | None = None,
    max_ev_iter: int | None = None,
    ev_conv_thresh: float | None = None,
    pade_npts: int | None = None,
    qp_newton_damp: float | None = None,
    frozen_core: int | None = None,
    n_quad: int | None = None,
    quadrature: str | None = None,
    u0: float | None = None,
    trunc_thresh: float | None = None,
    eigensolver_conv_thresh: float | None = None,
    k_builder: str | None = None,
    chi0_sparsity: str | None = None,
    memory_budget_gb: float | None = None,
) -> GwResult:
    """Closed-shell G0W0/COHSEX/evGW0/evGW on an RHF or RKS reference."""
    ...

def run_u_gw(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    reference: str | None = None,
    method: str | None = None,
    xc: str | None = None,
    qp_mos: tuple[int, int] | None = None,
    max_ev_iter: int | None = None,
    ev_conv_thresh: float | None = None,
    pade_npts: int | None = None,
    qp_newton_damp: float | None = None,
    frozen_core: int | None = None,
    n_quad: int | None = None,
    quadrature: str | None = None,
    u0: float | None = None,
    trunc_thresh: float | None = None,
    eigensolver_conv_thresh: float | None = None,
    k_builder: str | None = None,
    chi0_sparsity: str | None = None,
    memory_budget_gb: float | None = None,
) -> UGwResult:
    """Open-shell U-G0W0/U-COHSEX/U-evGW0/U-evGW on a UHF/ROHF reference."""
    ...

def run_bse_tda(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    frozen_core: int | None = None,
    n_quad: int | None = None,
    quadrature: str | None = None,
    u0: float | None = None,
    trunc_thresh: float | None = None,
    eigensolver_conv_thresh: float | None = None,
    k_builder: str | None = None,
    chi0_sparsity: str | None = None,
    memory_budget_gb: float | None = None,
) -> BseResult:
    """BSE-TDA singlet excitation energies on a closed-shell RHF reference."""
    ...

def run_tdhf_static_polarizability(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    xc: str,
    scissor: float | None = None,
    frozen_core: int | None = None,
    n_quad: int | None = None,
    quadrature: str | None = None,
    u0: float | None = None,
    trunc_thresh: float | None = None,
    eigensolver_conv_thresh: float | None = None,
    k_builder: str | None = None,
    chi0_sparsity: str | None = None,
    memory_budget_gb: float | None = None,
) -> TdhfStaticPolarizabilityResult:
    """RPAx@KS static polarizability (omega=0). xc is REQUIRED.

    Not validated: water/cc-pVDZ/PBE at scissor=0.36 gives 5.20 a.u. vs DOSD
    9.64 (46% low). At the default scissor=0.0 the kernel is often unstable
    and the call raises on a negative alpha diagonal; use scissor ~0.3-0.4 Ha.
    """
    ...

def run_tddft(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    functional: str | None = None,
    n_roots: int = 3,
    method: str = "tda",
) -> TddftResult:
    """Closed-shell singlet linear response. functional=None runs on an RHF
    reference (CIS for method="tda"/"cis", TDHF for "casida"/"rpa"/"tddft"/
    "tdhf"); a functional converges that KS reference (RI-JK,
    def2-universal-jkfit, default 75x110 grid) and adds the (ia|f_xc|jb)
    kernel block to A (and B for Casida) on the same grid. Meta-GGA, VV10 and
    range-separated functionals raise ValueError (no complete kernel); there
    is no kernel-less DFT result. Coulomb/exchange response integrals are RI
    over `auxbasis`."""
    ...

def run_double_hybrid(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    kind: str = "b2plyp",
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> DoubleHybridResult:
    """MP2-based double hybrid: kind="b2plyp" or "dsd-pbep86"."""
    ...

def run_lmp2(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    eps: float | None = None,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
    compute_reference: bool | None = None,
) -> dict[str, object]:
    """Amplitude-threshold local MP2 (closed-shell). Returns a dict."""
    ...

def run_lmp2_direct(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    eps: float | None = None,
    frozen_core: int | None = None,
    aux_radius_bohr: float | None = None,
    virt_radius_bohr: float | None = None,
    ao_tail: float | None = None,
    schwarz_skip: float | None = None,
    batch_merge: int | None = None,
    pair_gate_cal: float | None = None,
    virt_schwarz_kappa: float | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
    compute_reference: bool | None = None,
) -> dict[str, object]:
    """Integral-direct amplitude-threshold local MP2 (closed-shell). Returns
    the run_lmp2 dict plus strip/eri3 counters and stage timings."""
    ...

def run_drpa(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    eps: float | None = None,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
    compute_reference: bool | None = None,
    diis: int | None = None,
    eps_rtol_factor: float | None = None,
) -> dict[str, object]:
    """Amplitude-threshold direct RPA (closed-shell). Returns a dict."""
    ...

def run_drpa_scan(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    eps_list: list[float],
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
    compute_reference: bool | None = None,
    diis: int | None = None,
    eps_rtol_factor: float | None = None,
) -> list[dict[str, object]]:
    """Amplitude-threshold dRPA over a list of eps values. Returns list of dicts."""
    ...

def run_linlccd_amplitude(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    variant: str | None = None,
    eps: float | None = None,
    frozen_core: int | None = None,
    k_builder: str | None = None,
    memory_budget_gb: float | None = None,
) -> dict[str, object]:
    """Amplitude-threshold LinLCCD (closed-shell). Returns a dict."""
    ...

def tune_omega(
    mol: Molecule,
    basis_set: BasisSet,
    functional: str,
    omega_lo: float | None = None,
    omega_hi: float | None = None,
    omega_tol: float | None = None,
    max_evals: int | None = None,
) -> dict[str, object]:
    """Optimal tuning of range-separation omega for an RSH functional."""
    ...

def esp_at_atoms(
    mol: Molecule,
    basis_set: BasisSet,
    result: RhfResult | DftResult,
) -> list[float]:
    """Electrostatic potential at each nucleus (Hartree atomic units)."""
    ...

def esp_at_points(
    mol: Molecule,
    basis_set: BasisSet,
    result: RhfResult | DftResult,
    points: NDArray[np.float64],
) -> list[float]:
    """Electrostatic potential at arbitrary points (N,3) in Bohr."""
    ...

def hirshfeld_charges(
    mol: Molecule,
    basis_set: BasisSet,
    result: RhfResult | DftResult,
) -> list[float]:
    """Hirshfeld partial charges (units of e)."""
    ...

def lowdin_charges(
    mol: Molecule,
    basis_set: BasisSet,
    result: RhfResult | DftResult,
) -> list[float]:
    """Lowdin (symmetric-orthogonalization) partial charges (units of e)."""
    ...

def mulliken_charges(
    mol: Molecule,
    basis_set: BasisSet,
    result: RhfResult | DftResult,
) -> list[float]:
    """Mulliken partial charges (units of e)."""
    ...

def chelpg_charges(
    mol: Molecule,
    basis_set: BasisSet,
    result: RhfResult | DftResult,
) -> list[float]:
    """CHELPG (ESP-fitted) partial charges (units of e)."""
    ...

def resp_charges(
    mol: Molecule,
    basis_set: BasisSet,
    result: RhfResult | DftResult,
) -> list[float]:
    """RESP (restrained ESP-fitted) partial charges (units of e)."""
    ...

def hirshfeld_polarizability(
    mol: Molecule,
    basis_set: BasisSet,
    auxbasis: BasisSet,
    result: RhfResult | DftResult,
    memory_budget_gb: float | None = None,
) -> NDArray[np.float64]:
    """Per-atom Hirshfeld-partitioned static dipole polarizability (natoms, 3, 3) in Bohr^3."""
    ...

def orbital_moments(
    mol: Molecule,
    basis_set: BasisSet,
    result: RhfResult | DftResult,
) -> tuple[list[list[float]], list[float]]:
    """Per-orbital centroids and spatial spreads. Restricted only."""
    ...

def density_second_moment(
    mol: Molecule,
    basis_set: BasisSet,
    result: RhfResult | DftResult,
) -> list[list[float]]:
    """Density second-moment tensor (3x3, Bohr^2) about the origin."""
    ...

def boltzmann_weights(
    energies: list[float],
    temperature_k: float = ...,
) -> BoltzmannWeights:
    """Boltzmann weights from energies (Hartree) at temperature_k (default 298.15 K)."""
    ...

def weighted_stats(
    values: list[float],
    weights: list[float],
) -> WeightedStats:
    """Weighted mean and standard deviation of a scalar property."""
    ...

def weighted_stats_vector(
    values: list[list[float]],
    weights: list[float],
) -> list[WeightedStats]:
    """Weighted mean and standard deviation of a vector-valued property, per component."""
    ...

def weighted_stats_tensor(
    values: list[list[list[float]]],
    weights: list[float],
) -> list[list[WeightedStats]]:
    """Weighted mean and standard deviation of a rank-2 tensor property, per element."""
    ...

def compute_eri3(
    mol: Molecule,
    basis_set: BasisSet,
    aux_basis_set: BasisSet,
) -> NDArray[np.float64]:
    """Raw 3-center Coulomb integrals (P|mu nu), shape (naux, n_bf, n_bf)."""
    ...

def compute_eri3_mo(
    mol: Molecule,
    basis_set: BasisSet,
    aux_basis_set: BasisSet,
    c_left: NDArray[np.float64],
    c_right: NDArray[np.float64],
    memory_budget_gb: float = 2.0,
    operator: str = "coulomb",
    omega: float | None = None,
    r0: float | None = None,
) -> NDArray[np.float64]:
    """Blocked MO-basis 3-center integrals (P|pq), shape (naux, n_left, n_right)."""
    ...

def compute_metric_2c(
    mol: Molecule,
    basis_set: BasisSet,
    aux_basis_set: BasisSet,
    operator: str = "coulomb",
    omega: float | None = None,
    r0: float | None = None,
) -> NDArray[np.float64]:
    """2-center metric (P|w|Q) over the auxiliary basis (naux, naux)."""
    ...

def boys_localize(
    mol: Molecule,
    basis_set: BasisSet,
    c_orbs: NDArray[np.float64],
    max_iter: int = 200,
) -> BoysResult:
    """Boys (Foster-Boys) localization of given orbitals."""
    ...

def shell_info(
    mol: Molecule,
    basis_set: BasisSet,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Shell geometry: (centers (n_shells,3), offsets (n_shells,), dims (n_shells,))."""
    ...
