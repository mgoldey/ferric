#!/usr/bin/env python3
"""protokit: shared helpers for Python prototypes of ferric algorithms.

A prototype answers "is the algorithm right, and does the mechanism pay?" before
any Rust is written. The slow parts of that loop are always the same: build a
molecule, converge a reference SCF, generate 3-index integrals, time a kernel
with a pinned thread count, compare against a reference. This module does each
once, and caches the expensive inputs on disk so a re-run of an edited
prototype starts at the kernel under study.

    from protokit import alkane, df_inputs, rhf_reference, bench, threads

    mol = alkane(12, "cc-pvdz")
    ref = rhf_reference(mol, "def2-universal-jkfit")     # cached
    ints = df_inputs(mol, "def2-universal-jkfit")        # cached
    with threads(4):
        t = bench(lambda: my_k(ints, ref.c_occ), reps=5)
    print(t.median, jk_error(t.result, ref.k))

Needs numpy, scipy, PySCF and threadpoolctl (`pip install threadpoolctl`; it
is not a ferric dependency), and, for `ferric_inputs`, an importable ferric.

Cache: `scripts/queue/out/protokit-cache/` (gitignored), or $PROTOKIT_CACHE.
Entries are keyed on the molecule (atoms, coordinates, basis), the auxiliary
basis and a per-builder version tag; delete the directory to start clean.

A numpy timing is evidence about FLOP counts and BLAS shapes. It is not a
prediction of the Rust wall time: report it as a ratio between two numpy
kernels run under the same `threads(n)`, never as an absolute.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle  # nosec B403 - reads only this module's own cache files, see `cached`
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MOLDIR = REPO / "testdata" / "molecules"
CACHE = Path(
    os.environ.get(
        "PROTOKIT_CACHE", REPO / "scripts" / "queue" / "out" / "protokit-cache"
    )
)


# --------------------------------------------------------------------------- threads / timing


@contextlib.contextmanager
def threads(n: int):
    """Pin numpy's BLAS and PySCF's OpenMP pool to `n` threads inside the block."""
    from pyscf import lib
    from threadpoolctl import threadpool_limits

    before = lib.num_threads()
    lib.num_threads(n)
    try:
        with threadpool_limits(limits=n):
            yield
    finally:
        lib.num_threads(before)


@dataclass
class Bench:
    """Wall times of `reps` calls; `result` is the last call's return value."""

    times: list[float]
    result: Any = None

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def min(self) -> float:
        return min(self.times)

    def __str__(self) -> str:
        return f"{self.median:.4f} / {self.min:.4f} s [n={len(self.times)}]"


def bench(fn: Callable[[], Any], reps: int = 3, warmup: int = 1) -> Bench:
    """Median/min wall time of `fn()` after `warmup` discarded calls."""
    for _ in range(warmup):
        fn()
    times, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return Bench(times, out)


@dataclass
class Stages:
    """Accumulating stage timer: `with st("half-transform"): ...`, then `st.report()`."""

    secs: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    @contextlib.contextmanager
    def __call__(self, label: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.secs[label] = self.secs.get(label, 0.0) + time.perf_counter() - t0
            self.calls[label] = self.calls.get(label, 0) + 1

    def report(self, title: str = "stages") -> str:
        total = sum(self.secs.values()) or 1.0
        lines = [f"{title}: {total:.3f} s"]
        for k, v in sorted(self.secs.items(), key=lambda kv: -kv[1]):
            lines.append(
                f"  {k:<28s} {v:8.3f} s  {100 * v / total:5.1f}%  x{self.calls[k]}"
            )
        out = "\n".join(lines)
        print(out)
        return out


# --------------------------------------------------------------------------- molecules


def mol_from_xyz(name_or_path: str, basis: str, **kw):
    """PySCF Mole from an .xyz path or a `testdata/molecules` stem (e.g. "alkane_12")."""
    from pyscf import gto

    p = Path(name_or_path)
    if not p.exists():
        p = MOLDIR / f"{name_or_path}.xyz"
    return gto.M(atom=str(p), basis=basis, verbose=0, **kw)


def alkane(n_carbon: int, basis: str, **kw):
    """All-anti n-alkane from `testdata/molecules/alkane_<n>.xyz`."""
    return mol_from_xyz(f"alkane_{n_carbon}", basis, **kw)


def mol_key(mol) -> str:
    """Stable digest of atoms, coordinates, basis, ECPs, charge and spin."""
    h = hashlib.sha256()
    h.update(
        repr(
            [
                (mol.atom_symbol(i), mol.atom_coord(i).round(10).tolist())
                for i in range(mol.natm)
            ]
        ).encode()
    )
    h.update(
        repr((sorted(mol._basis.items()), mol.charge, mol.spin, mol.cart)).encode()
    )
    # ECPs live outside `_basis` and change the effective nuclear charges.
    if mol._ecp:
        h.update(repr(sorted(mol._ecp.items())).encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- cache


def cached(key: str, build: Callable[[], Any], refresh: bool = False) -> Any:
    """`build()` once per `key`; later calls unpickle it from the cache directory.

    The cache holds only what this process family wrote there itself (a
    gitignored scratch directory). Do not point `PROTOKIT_CACHE` at files from
    anywhere else: unpickling runs code.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)  # nosec B301 - self-written local cache only
    out = build()
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return out


# --------------------------------------------------------------------------- DF inputs / references


@dataclass
class DfInputs:
    """Raw density-fitting integrals.

    `eri3[P, mu, nu] = (P|mu nu)` (square, C order) and `metric[P, Q] = (P|Q)`.
    """

    eri3: np.ndarray
    metric: np.ndarray

    @property
    def naux(self) -> int:
        return self.eri3.shape[0]

    @property
    def nao(self) -> int:
        return self.eri3.shape[1]

    def dressed(self) -> np.ndarray:
        """`B = L^-1 (P|mu nu)` with `metric = L L^T`, so `(mu nu|la si) ~ sum_P B_P B_P`."""
        import scipy.linalg as sla

        low = np.linalg.cholesky(self.metric)
        n = self.nao
        return sla.solve_triangular(
            low, self.eri3.reshape(self.naux, n * n), lower=True
        ).reshape(self.naux, n, n)


def df_inputs(mol, auxbasis: str, refresh: bool = False) -> DfInputs:
    """Cached `(P|mu nu)` and `(P|Q)` for `mol` over `auxbasis`."""
    from pyscf import df

    def build():
        auxmol = df.addons.make_auxmol(mol, auxbasis)
        eri3 = np.ascontiguousarray(
            df.incore.aux_e2(mol, auxmol, intor="int3c2e").transpose(2, 0, 1)
        )
        return DfInputs(eri3, auxmol.intor("int2c2e"))

    return cached(f"df-v1-{mol_key(mol)}-{auxbasis}", build, refresh)


@dataclass
class RhfReference:
    """Converged closed-shell RHF: `dm` is the TOTAL density, `j`/`k` are built on it."""

    energy: float
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    dm: np.ndarray
    j: np.ndarray
    k: np.ndarray
    iterations: int

    @property
    def nocc(self) -> int:
        return int((self.mo_occ > 0).sum())

    @property
    def c_occ(self) -> np.ndarray:
        """Occupied MO coefficients, so `dm = 2 c_occ c_occ^T`."""
        return np.ascontiguousarray(self.mo_coeff[:, self.mo_occ > 0])


def rhf_reference(
    mol, auxbasis: str | None = None, conv_tol: float = 1e-11, refresh: bool = False
) -> RhfReference:
    """Cached converged RHF: RI-JK over `auxbasis`, or exact four-centre J/K when it is None."""
    from pyscf import scf

    def build():
        mf = scf.RHF(mol)
        if auxbasis is not None:
            mf = mf.density_fit(auxbasis=auxbasis)
        mf.conv_tol = conv_tol
        iters = [0]
        mf.callback = lambda _env: iters.__setitem__(0, iters[0] + 1)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("reference RHF did not converge")
        dm = mf.make_rdm1()
        j, k = mf.get_jk(dm=dm)
        return RhfReference(
            mf.e_tot, mf.mo_coeff, mf.mo_occ, np.asarray(dm), j, k, iters[0]
        )

    return cached(f"rhf-v1-{mol_key(mol)}-{auxbasis}-{conv_tol:g}", build, refresh)


# --------------------------------------------------------------------------- comparisons


def jk_error(
    got: np.ndarray, ref: np.ndarray, dm: np.ndarray | None = None
) -> dict[str, float]:
    """Max element error, relative Frobenius error and, with `dm`, the energy-like `|tr(dm * diff)|`."""
    diff = got - ref
    out = {
        "max_abs": float(np.abs(diff).max()),
        "rel_fro": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-300)),
        "asym": float(np.abs(got - got.T).max()),
    }
    if dm is not None:
        out["tr_dm"] = float(abs(np.einsum("ij,ij->", dm, diff)))
    return out


def gflops(flops: float, secs: float) -> float:
    """GFLOP/s of a kernel whose FLOP count is known."""
    return flops / secs / 1e9


# --------------------------------------------------------------------------- ferric bridge


def ferric_build() -> dict | None:
    """`ferric.__build__` of the importable extension (which commit a comparison ran), or None."""
    try:
        import ferric
    except ImportError:
        return None
    return {"file": ferric.__file__, **(getattr(ferric, "__build__", None) or {})}


def ferric_inputs(xyz_name_or_path: str, basis: str):
    """`(ferric.Molecule, ferric.BasisSet)` for the same file `mol_from_xyz` reads."""
    import ferric

    p = Path(xyz_name_or_path)
    if not p.exists():
        p = MOLDIR / f"{xyz_name_or_path}.xyz"
    return ferric.Molecule.from_xyz(str(p)), ferric.BasisSet.bundled(basis)
