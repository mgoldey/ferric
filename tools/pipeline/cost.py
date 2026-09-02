"""Cost model for the DFT tier — what actually drives its size.

This module exists because of a REASONING error, not a coding one. The
campaign record twice claimed tier 4's blowup "is not a size effect" on the
grounds that a 234-basis-function job outran a 330-function one. That
comparison is void: in ferric, KS-DFT's grid is a flat 75x110 Becke-Lebedev
grid **per atom** (`crates/ferric-dft/src/grid.rs`), so the exchange-correlation
cost scales with ATOM COUNT and is independent of the basis set.

The practical consequence is counter-intuitive enough to be worth encoding:
**shrinking the basis can INCREASE resident memory**, because the AO cache is
`4 * nbf * npts * 8` bytes and `npts` is fixed by the geometry. Going from a
32-atom alkane at def2-SVP to a 70-atom drug at STO-3G cuts nbf by 0.71x but
raises npts by 2.19x, for a net 1.55x -- a bigger cache from a smaller basis.

Numbers here are structural (grid dimensions read from the Rust source), not
timings, so they carry no machine dependence.
"""
from __future__ import annotations

from dataclasses import dataclass

# ferric-dft/src/grid.rs AtomicGridConfig default: the "fine" production grid.
N_RADIAL = 75
N_ANGULAR = 110
GRID_POINTS_PER_ATOM = N_RADIAL * N_ANGULAR

# ks.rs: chi + dchi = AoGridKind::ValueAndGrad.planes() == 4 planes of f64.
AO_CACHE_PLANES = 4
BYTES_PER_F64 = 8


@dataclass(frozen=True)
class DftSize:
    """The two axes that set a KS-DFT job's cost, kept separate on purpose."""

    n_atoms: int
    n_basis_functions: int

    @property
    def grid_points(self) -> int:
        """Total XC grid points. Depends ONLY on atom count."""
        return self.n_atoms * GRID_POINTS_PER_ATOM

    @property
    def ao_cache_bytes(self) -> int:
        """Resident chi + grad-chi cache: 4 * nbf * npts * 8."""
        return (AO_CACHE_PLANES * self.n_basis_functions
                * self.grid_points * BYTES_PER_F64)

    @property
    def ao_cache_gb(self) -> float:
        return self.ao_cache_bytes / 1e9

    @property
    def xc_work(self) -> int:
        """nbf x npts -- the size of one pass over the grid."""
        return self.n_basis_functions * self.grid_points

    @property
    def xc_fock_work(self) -> int:
        """nbf^2 x npts -- the XC Fock assembly, done EVERY SCF iteration.

        This is the term that actually sets KS-DFT wall time at drug scale.
        `vxc.rs` assembles V_xc with `buf.dot(&chi.t())`, an (nbf, npts) x
        (npts, nbf) GEMM. Since npts is proportional to atom count, this is
        cubic in molecular size -- and it is paid per iteration, not once.

        Calibrated against measured alkane runs (STO-3G/PBE, 10 iterations
        each), predicting from alkane_5 alone:

            atoms   predicted   actual
               32       17.8 s   19.6 s
               62      134.3 s  130.2 s

        Within 10% across a 54x span of cost.

        **This term is PER ITERATION; the iteration count is a separate
        factor and it is NOT constant across chemistries.** Every alkane
        converged in exactly 10 iterations, but danuglipron took **18**
        (measured 2026-09-02, 612.4 s, converged, on a verified-quiet box).
        So predicting a wall time needs BOTH factors:

            wall ~ xc_fock_work x (seconds per iteration) x (iterations)

        Scaling alkane_20's 130.2 s by work alone predicts 6.8 min and the
        actual is 10.2 min (1.50x). Multiply through by the real iteration
        ratio (18/10) instead and the per-iteration model lands within 20%,
        slightly conservative.

        `predicted_seconds` assumes a COMPARABLE iteration count and is
        therefore a LOWER BOUND for a molecule that converges more slowly than
        the reference. Do not quote it as a promised wall time; use
        `PyDftResult.iterations` from a real run to calibrate a new chemistry.
        """
        return (self.n_basis_functions ** 2) * self.grid_points

    def predicted_seconds(self, reference: "DftSize", reference_seconds: float) -> float:
        """Scale a measured runtime from `reference` to this system.

        Uses `xc_fock_work`, NOT atom count and NOT nbf alone -- both of those
        mispredict badly (see the module docstring). Assumes a comparable
        iteration count, which held at exactly 10 across 17-62 atoms.
        """
        if reference.xc_fock_work == 0:
            raise ValueError("reference system has zero XC work")
        return reference_seconds * self.xc_fock_work / reference.xc_fock_work


# STO-3G contracted functions per atom, by row. Enough to compare two
# molecules' XC work without building a basis; NOT a general basis-set model.
_STO3G_NBF = {"H": 1, "He": 1,
              "C": 5, "N": 5, "O": 5, "F": 5, "B": 5, "Be": 5, "Li": 5, "Ne": 5}


def sto3g_basis_functions(symbols: list[str]) -> int:
    """Rough nbf for a symbol list at STO-3G. Unknown elements count as 5.

    Exists so two candidates can be compared on XC work (nbf x npts) rather
    than on atom count, which hides composition. An alkane is hydrogen-padded
    (1 function per H) while a drug is heavy-atom rich (5 per C/N/O/F), so
    two molecules with the SAME atom count can differ ~2x in XC work.
    """
    return sum(_STO3G_NBF.get(s, 5) for s in symbols)


def ri_tensor_gb(n_basis_functions: int, n_aux_functions: int) -> float:
    """Resident RI three-index `(P|mn)` tensor: naux x nbf^2 x 8 bytes.

    `run_dft` always enables RI-J with `def2-universal-jkfit`, which is sized
    for large orbital bases. At STO-3G that aux basis is grossly oversized:
    measured on danuglipron, **3,635 aux functions against 235 orbital
    functions (15x)**, for a 1.61 GB tensor -- a sixth of the run's ~9.75 GB
    peak, spent on fitting accuracy the orbital basis cannot use.

    Counted from `crates/ferric-core/src/basis/bundled/def2-universal-jkfit.json`,
    not estimated: a first guess of "naux ~700-950" was low by ~4x and left
    1.2 GB wrongly filed as unexplained.
    """
    return n_aux_functions * n_basis_functions * n_basis_functions * 8 / 1e9


def fits_in_budget(size: DftSize, budget_gb: float) -> bool:
    """Whether the full AO cache fits, i.e. whether ferric avoids batching.

    ks.rs falls back to walking the grid in point-batches (recomputing AO
    values per batch) when the cache exceeds the resolved budget. That is a
    performance cliff, so a caller that cares about wall time should check.
    """
    return size.ao_cache_gb <= budget_gb
