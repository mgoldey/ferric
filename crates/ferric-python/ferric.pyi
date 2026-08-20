"""Python bindings for ferric -- a Rust-native quantum chemistry engine."""

from __future__ import annotations

from typing import Sequence

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


class BasisSet:
    """A Gaussian basis set (orbital or auxiliary/RI-fitting)."""

    @staticmethod
    def bundled(name: str) -> BasisSet:
        """Load a bundled basis set by name (e.g. 'sto-3g', 'cc-pvdz-ri')."""
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
        """RHF + MP2 correlation energy, Hartree."""
        ...

    @property
    def rhf_energy(self) -> float:
        """Converged reference RHF energy, Hartree."""
        ...

    @property
    def mp2_corr(self) -> float:
        """MP2 correlation energy (always negative), Hartree."""
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
    def total_energy(self) -> float: ...

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
) -> RhfResult:
    """Closed-shell Restricted Hartree-Fock."""
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
) -> UhfResult:
    """Unrestricted Hartree-Fock (open-shell)."""
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
) -> UhfResult:
    """Restricted Open-Shell Hartree-Fock."""
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
) -> FrequencyResult:
    """Harmonic vibrational frequencies via finite-difference of analytic gradients."""
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
    """Resolution-of-identity (density-fitted) MP2 on a closed-shell RHF reference."""
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
) -> DftResult:
    """Kohn-Sham DFT (closed-shell)."""
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
) -> DftResult:
    """Kohn-Sham DFT (closed-shell). Alias of run_dft."""
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
) -> PdepRpaResult:
    """PDEP-RPA (dielectric eigendecomposition RPA)."""
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
    """RPAx@KS static polarizability (omega=0). xc is REQUIRED."""
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
