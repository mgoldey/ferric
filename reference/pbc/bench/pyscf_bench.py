"""PySCF side of the issue-#228 ferric-vs-PySCF timing (one cell, one fresh process per call).

Usage (from anywhere):
  OPENBLAS_NUM_THREADS=1 python pyscf_bench.py CELL --threads 1 --precision 1e-10 [--method rhf|pbe]
      [--aux cc-pvdz-ri] [--kmesh 2 2 2] [--df gdf|rsdf] [--basis-repr general|segmented]
      [--lindep-s 1e-6] [--init-guess hcore] [--verbose 0] [--tag T] [--out file.jsonl]
      [--match-ferric-conv]   (stop on ferric's periodic SCF rule; see the comment block below)

CELL = basename of <CELL>.xyz / <CELL>.lattice next to this file (Angstrom xyz, Bohr lattice rows).
Matched to the ferric CLI inputs under toml/ (see README.md): spherical cc-pVDZ and the aux basis
read from ferric's own bundled BSE JSON, GDF (= _RSGDFBuilder in PySCF 2.13), exxdiv=ewald, hcore
guess, eigendecomposed aux metric with absolute cut 1e-10, orbital lindep cut 1e-6.

  --df gdf (default) is `density_fit` (in 2.13 the range-separated _RSGDFBuilder); --df rsdf is
  `rs_density_fit` (the older RSGDF class, which also builds through _RSGDFBuilder, so the metric
  and builder instrumentation below see it). RSJK is REFUSED: PySCF 2.13 `SCF.energy_nuc`
  (pbc/scf/hf.py:760) has an operator-precedence bug that gives any 3-D RSJK run the molecular
  E_nn (prototypes/pbc/FINDINGS.md "Iteration 1").
  --basis-repr general (default) hands ferric's columns to PySCF as one general contraction per
  shell (PySCF's native, fast form; same function space); segmented passes ferric's exact
  per-column shells including the zero coefficients (what ferric itself computes with).

One record = one process = one DF build (the cderi file is written under a private temp dir and
deleted before exit, so nothing is ever reused). Stages are timed separately:
  df_build (GDF.build, includes the cderi write), hcore (get_hcore), scf (kernel minus hcore),
  iterations and seconds/iteration (median gap between callbacks).
The aux-metric drop count is captured by wrapping eigenvalue_decomposed_metric.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
# Angstrom per Bohr: ferric_core::units::BOHR_RADIUS_ANGSTROM == pyscf.data.nist.BOHR (CODATA 2010).
# Until 2026-10-24 this file had the CODATA 2014 value 0.52917721067, so the PySCF atoms sat 4.7e-10
# (relative) off ferric's while the Bohr lattice rows matched.
BOHR = 0.52917721092


def read_cell(name):
    base = name if os.path.isabs(name) else os.path.join(HERE, name)
    lines = open(base + ".xyz").read().split("\n")
    n = int(lines[0])
    atoms = []
    for ln in lines[2 : 2 + n]:
        t = ln.split()
        atoms.append((t[0], tuple(float(v) / BOHR for v in t[1:4])))
    lat = [
        [float(v) for v in r.split()]
        for r in open(base + ".lattice").read().split("\n")
        if r.strip()
    ]
    return atoms, lat


def ferric_basis_pyscf(name, symbols, repr_="general"):
    """ferric's bundled BSE JSON -> PySCF basis dict.

    repr_ "general": one block per BSE shell with all its columns (a general contraction).
    repr_ "segmented": one shell per column, zero coefficients kept, i.e. ferric's own shells.
    """
    if repr_ not in ("general", "segmented"):
        raise ValueError(f"repr_ must be 'general' or 'segmented', got {repr_!r}")
    from pyscf.data.elements import _symbol

    d = json.load(
        open(os.path.join(ROOT, "crates/ferric-core/src/basis/bundled", name + ".json"))
    )
    out = {}
    for sym in sorted(set(symbols)):
        z = None
        for k in d["elements"]:
            if _symbol(int(k)) == sym:
                z = k
        blocks = []
        for sh in d["elements"][z]["electron_shells"]:
            ex = [float(x) for x in sh["exponents"]]
            cols = [[float(x) for x in c] for c in sh["coefficients"]]
            ls = sh["angular_momentum"]
            for k, ang in enumerate(ls):
                # fused SP-type shells: one coefficient row per angular momentum
                cc = [cols[k]] if len(ls) > 1 else cols
                groups = [[c] for c in cc] if repr_ == "segmented" else [cc]
                for g in groups:
                    blocks.append(
                        [ang] + [[ex[i]] + [c[i] for c in g] for i in range(len(ex))]
                    )
        out[sym] = blocks
    return out


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _num(x):
    return None if x is None else float(x)


def _ints(v):
    return None if v is None else [int(x) for x in v]


def load1():
    return float(open("/proc/loadavg").read().split()[0])


# ---------------------------------------------------------------------------------------------
# --match-ferric-conv: ferric's periodic SCF stopping rule as a PySCF `mf.check_convergence` hook.
# Read from origin/main 5a536617a (line numbers are at that commit):
#
# Gamma point (RHF and RKS; both go through ferric_scf::rhf::solve_rhf_injected ->
# solve_rhf_impl, crates/ferric-scf/src/rhf.rs:1411 / :1716, loop at :2349):
#   density_conv = 1e-10   crates/ferric-cli/src/config.rs:5347-5351 (periodic_scf_knobs default when
#                          [scf] density_conv is not written; every toml_ri input leaves it unwritten)
#   energy_conv  = 1e-3    RhfConfig::default, crates/ferric-scf/src/rhf.rs:463, reached through
#                          gamma_scf_config's `..Default::default()` (crates/ferric-cli/src/periodic.rs:629-636)
#   gate         crates/ferric-scf/src/rhf.rs:853-866 `scf_converged`:
#                dp_rms < density_conv AND dp_max < 10*density_conv AND |dE| < energy_conv  (all strict)
#   dp_rms/max   crates/ferric-scf/src/driver.rs:207-219 `density_change`: sqrt(sum(dP^2) / len) over ALL
#                n*n elements of dP = D_new - D_old (both triangles, len = nao^2), and max |dP_ij|.
#                D = 2 C_occ C_occ^T (rhf.rs:3289, :3360), the same normalisation as PySCF make_rdm1.
#   pairing      ScfMonitor (driver.rs:445-459): the check at iteration n+1 uses dE = E(D_{n+1}) - E(D_n)
#                and dP = D_{n+1} - D_n, where D_{n+1} came from diagonalising the (DIIS) Fock of D_n.
#                PySCF's kernel (scf/hf.py:172-198 in 2.13.1) hands check_convergence exactly that
#                pair: e_tot/last_hf_e and dm/dm_last.  ferric counts the guess-energy pass as
#                iteration 1, so ferric `iterations` = PySCF `cycles` + 1 under the same rule.
#   veto         an accepted exit is refused if max|X^T (F D S - S D F) X| > max(sqrt(density_conv), 1e-3)
#                (rhf.rs:902-927, applied at :2708-2718 after the gate at :2694-2698; 1e-3 = rohf_occupation.rs:107), F = the un-extrapolated
#                Fock of the current D (PySCF's `fock` in envs, hf.py:187), X = canonical orthogonaliser
#                U_kept diag(1/sqrt(lambda)) with an ABSOLUTE cut lambda >= 1e-6 on the raw eigenvalues
#                of S (rhf.rs:3596, :3652-3676).
#   Not reproduced (inactive in the bench inputs): the ADIIS/EDIIS veto (Pulay is the default),
#   the COSX schedule gate, divergence/stall exits.
#
# k-mesh (crates/ferric-pbc/src/kscf.rs:842, KRHF):
#   it > 0 AND |dE| < energy_conv (1e-12 per cell) AND max_k max|X_k^H (F D S - (F D S)^H) X_k| < grad_conv
#   (1e-9), elementwise complex modulus; defaults kscf.rs:105-106 and config.rs:5352-5356 / :5816.
#   X_k: same absolute 1e-6 cut (kscf.rs:407-432).  D(k) = 2 C_occ C_occ^H (kscf.rs:657-661).
#   dm starts at 0 (it = 0 is the D = 0 pass, it = 1 the hcore-guess energy), so ferric's 0-based
#   `it` at exit (KScfResult.iterations, kscf.rs:842-858) is PySCF `cycles` + 1 under the same rule.
#
# PySCF's post-convergence `conv_check` extra cycle (hf.py:212-238) has no ferric counterpart and would
# re-run this hook; --match-ferric-conv therefore sets mf.conv_check = False.
# Basis-dependence caveat: max|X^T G X| depends on the eigenvector basis inside a DEGENERATE eigenspace
# of S, in ferric as well; it only enters through the 1e-3 veto (Gamma) or the 1e-9 bound (k).
FERRIC_GAMMA_DENSITY_CONV = 1e-10
FERRIC_GAMMA_ENERGY_CONV = 1e-3
FERRIC_VETO_MIN_GRAD = 1e-3
FERRIC_LINDEP = 1e-6
FERRIC_K_ENERGY_CONV = 1e-12
FERRIC_K_GRAD_CONV = 1e-9


def ferric_canonical_orthogonalizer(s, lindep=FERRIC_LINDEP):
    import numpy as np

    w, u = np.linalg.eigh(s)
    keep = w >= lindep
    return u[:, keep] / np.sqrt(w[keep])


def ferric_density_change(d_new, d_old):
    """(rms, max) of dP over all n*n elements (driver.rs:207-219)."""
    import numpy as np

    diff = np.asarray(d_new) - np.asarray(d_old)
    a = np.abs(diff)
    return float(np.sqrt(np.sum(a * a) / max(diff.size, 1))), float(a.max())


def ferric_orthogonal_gradient_max(fock, dm, s, lindep=FERRIC_LINDEP, x=None):
    """max |X^H (F D S - S D F) X| (rhf.rs:921-927; kscf.rs:831-841 per k).

    x: a precomputed orthogonaliser of s (S is fixed during one SCF); None = build it."""
    import numpy as np

    if x is None:
        x = ferric_canonical_orthogonalizer(s, lindep)
    fds = np.asarray(fock) @ np.asarray(dm) @ np.asarray(s)
    g = fds - fds.conj().T
    return float(np.abs(x.conj().T @ g @ x).max())


def ferric_gamma_signals(e, e_last, dm, dm_last, fock, s, x=None):
    dp_rms, dp_max = ferric_density_change(dm, dm_last)
    sig = dict(de=abs(e - e_last), dp_rms=dp_rms, dp_max=dp_max, grad_max=None)
    gate = (
        dp_rms < FERRIC_GAMMA_DENSITY_CONV
        and dp_max < 10.0 * FERRIC_GAMMA_DENSITY_CONV
        and sig["de"] < FERRIC_GAMMA_ENERGY_CONV
    )
    if gate:  # ferric evaluates the veto gradient only on an accepted exit
        sig["grad_max"] = ferric_orthogonal_gradient_max(fock, dm, s, x=x)
        bound = max(FERRIC_GAMMA_DENSITY_CONV**0.5, FERRIC_VETO_MIN_GRAD)
        gate = sig["grad_max"] <= bound
    sig["converged"] = bool(gate)
    return sig


def ferric_k_signals(e, e_last, dm_k, fock_k, s_k, x_k=None):
    if x_k is None:
        x_k = [None] * len(s_k)
    grad = max(
        ferric_orthogonal_gradient_max(f, d, s, x=x)
        for f, d, s, x in zip(fock_k, dm_k, s_k, x_k)
    )
    de = abs(e - e_last)
    return dict(
        de=de,
        grad_max=grad,
        converged=bool(de < FERRIC_K_ENERGY_CONV and grad < FERRIC_K_GRAD_CONV),
    )


def make_ferric_conv_check(kpt_run, trace):
    """mf.check_convergence(envs) applying ferric's rule; appends each cycle's signals to trace."""

    cache = {}  # id(s1e) -> orthogonaliser(s); S is built once per kernel call

    def orth(s1e):
        key = id(s1e)
        if key not in cache:
            cache.clear()
            cache[key] = (
                [ferric_canonical_orthogonalizer(s) for s in s1e]
                if kpt_run
                else ferric_canonical_orthogonalizer(s1e)
            )
        return cache[key]

    def check(envs):
        x = orth(envs["s1e"])
        if kpt_run:
            sig = ferric_k_signals(
                envs["e_tot"],
                envs["last_hf_e"],
                envs["dm"],
                envs["fock"],
                envs["s1e"],
                x,
            )
        else:
            sig = ferric_gamma_signals(
                envs["e_tot"],
                envs["last_hf_e"],
                envs["dm"],
                envs["dm_last"],
                envs["fock"],
                envs["s1e"],
                x,
            )
        sig["cycle"] = int(envs["cycle"]) + 1
        trace.append(sig)
        return sig["converged"]

    return check


FERRIC_PERIODIC_MAX_ITER = (
    200  # config.rs:5341-5346 (periodic_scf_knobs, max_iter not written)
)


def resolve_max_cycle(max_cycle, match_ferric_conv):
    if max_cycle is not None:
        return max_cycle
    return FERRIC_PERIODIC_MAX_ITER if match_ferric_conv else 100


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--method", choices=["rhf", "pbe"], default="rhf")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--aux", default="cc-pvdz-ri")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--precision", type=float, default=1e-12)
    ap.add_argument("--lindep-aux", type=float, default=1e-10)
    ap.add_argument(
        "--kmesh",
        type=int,
        nargs=3,
        default=None,
        help="Gamma-centred k-mesh (KRHF); omit = Gamma point",
    )
    # Defaults: Gamma 1e-10 / 1e-6; k-mesh 1e-12 / 1e-9 (ferric's k-point stopping rule).
    ap.add_argument("--conv-tol", type=float, default=None)
    ap.add_argument("--conv-tol-grad", type=float, default=None)
    ap.add_argument(
        "--max-cycle",
        type=int,
        default=None,
        help="default 100; 200 (ferric's periodic max_iter) with --match-ferric-conv",
    )
    ap.add_argument("--max-memory", type=int, default=8000)
    ap.add_argument("--grid", type=int, nargs=2, default=[75, 302])
    ap.add_argument("--df", choices=["gdf", "rsdf", "rsjk"], default="gdf")
    ap.add_argument("--basis-repr", choices=["general", "segmented"], default="general")
    ap.add_argument(
        "--lindep-s", type=float, default=1e-6, help="orbital S eigenvalue cut"
    )
    ap.add_argument("--init-guess", default="hcore")
    ap.add_argument("--verbose", type=int, default=0, help="PySCF cell/mf verbosity")
    ap.add_argument(
        "--match-ferric-conv",
        action="store_true",
        help="stop on ferric's periodic SCF rule (Gamma: dP rms/max + dE + gradient veto; "
        "k: dE + max orbital-gradient element), via mf.check_convergence; conv_check off",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    a.max_cycle = resolve_max_cycle(a.max_cycle, a.match_ferric_conv)
    if a.match_ferric_conv and "FERRIC_LINDEP_THRESH" in os.environ:
        sys.exit(
            "refused: FERRIC_LINDEP_THRESH is set; --match-ferric-conv hard-codes ferric's "
            "default orthogonaliser cut 1e-6"
        )
    if a.df == "rsjk":
        sys.exit(
            "refused: PySCF 2.13 RSJK gets E_nn wrong for every 3-D cell "
            "(pbc/scf/hf.py:760, FINDINGS Iteration 1)"
        )
    kpt_run = a.kmesh is not None
    if a.conv_tol is None:
        a.conv_tol = 1e-12 if kpt_run else 1e-10
    if a.conv_tol_grad is None:
        a.conv_tol_grad = 1e-9 if kpt_run else 1e-6
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(a.threads)

    import numpy as np
    import pyscf
    from pyscf import lib
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from pyscf.pbc import dft as pdft
    from pyscf.pbc.df import rsdf_builder
    import pyscf.scf.hf as hf_mod

    lib.num_threads(a.threads)
    hf_mod.remove_overlap_zero_eigenvalue = True  # PySCF's default; stated, not assumed
    hf_mod.overlap_zero_eigenvalue_threshold = a.lindep_s
    atoms, lat = read_cell(a.cell)
    syms = [s for s, _ in atoms]
    load0 = load1()
    t_all = time.perf_counter()
    cell = pgto.Cell()
    cell.a = lat
    cell.unit = "B"
    cell.atom = atoms
    cell.basis = ferric_basis_pyscf(a.basis, syms, a.basis_repr)
    cell.cart = False
    cell.precision = a.precision
    cell.max_memory = a.max_memory
    cell.verbose = a.verbose
    w0 = time.perf_counter()
    cell.build()
    t_cell = time.perf_counter() - w0
    aux = ferric_basis_pyscf(a.aux, syms)

    Builder = rsdf_builder._RSGDFBuilder
    Builder.j2c_eig_always = True
    drops = []
    orig = Builder.eigenvalue_decomposed_metric

    def wrapped(self, j2c):
        r = orig(self, j2c)  # returns (v1 = kept/sqrt(w), negative, tag); no extra eigh
        n = int(np.asarray(j2c).shape[0])
        drops.append(dict(n=n, kept=int(r[0].shape[0]), dropped=n - int(r[0].shape[0])))
        return r

    Builder.eigenvalue_decomposed_metric = wrapped
    builders = []
    orig_build = Builder.build

    def wb(self, *args, **kw):
        builders.append(self)
        return orig_build(self, *args, **kw)

    Builder.build = wb

    kpts = cell.make_kpts(a.kmesh) if kpt_run else None
    if a.method == "rhf":
        if kpt_run:
            mf = pscf.KRHF(cell, kpts, exxdiv="ewald")
        else:
            mf = pscf.RHF(cell, exxdiv="ewald")
    else:
        if kpt_run:
            sys.exit("k-point PBE is not part of this benchmark")
        mf = pdft.RKS(cell, exxdiv="ewald")
        mf.xc = "PBE"
        # pbc BeckeGrids (atom grids on image atoms, cut to the cell, original Becke
        # partition), unpruned. The pbc RKS default is UniformGrids, on which atom_grid
        # has NO effect (issue #228 addendum).
        mf.grids = pdft.gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = tuple(a.grid)
        mf.grids.prune = None
    mf = (
        mf.density_fit(auxbasis=aux)
        if a.df == "gdf"
        else mf.rs_density_fit(auxbasis=aux)
    )
    if a.method == "pbe":
        # density_fit() rebuilds the grids object with DEFAULT settings (atom_grid {},
        # pruning back on): measured in the second #228 session, where the PySCF PBE
        # energy did not move between 75x302 and 150x974.  Set them again here.
        mf.grids.atom_grid = tuple(a.grid)
        mf.grids.prune = None
        assert isinstance(mf.grids, pdft.gen_grid.BeckeGrids), type(mf.grids)
    df = mf.with_df
    df.linear_dep_threshold = a.lindep_aux
    if a.df == "rsdf":
        # RSDF copies its own __dict__ into the builder, so its instance attribute (default
        # False) would override the class-level True set above.
        df.j2c_eig_always = True
    tmpdir = tempfile.mkdtemp(prefix="pyscf228_", dir=os.environ.get("TMPDIR"))
    df._cderi_to_save = os.path.join(tmpdir, "cderi.h5")
    mf.init_guess = a.init_guess
    mf.conv_tol = a.conv_tol
    mf.conv_tol_grad = a.conv_tol_grad
    mf.max_cycle = a.max_cycle
    mf.verbose = a.verbose
    conv_trace = []
    if a.match_ferric_conv:
        mf.check_convergence = make_ferric_conv_check(kpt_run, conv_trace)
        mf.conv_check = False

    w0, c0 = time.perf_counter(), time.process_time()
    df.build()
    t_df = time.perf_counter() - w0
    c_df = time.process_time() - c0
    cderi_bytes = (
        os.path.getsize(df._cderi_to_save)
        if os.path.exists(df._cderi_to_save)
        else None
    )
    rss_after_df = peak_rss_bytes()

    tt = {}
    orig_hcore = mf.get_hcore

    def gh(*args, **kw):
        w = time.perf_counter()
        r = orig_hcore(*args, **kw)
        tt["hcore"] = tt.get("hcore", 0.0) + time.perf_counter() - w
        tt["hcore_calls"] = tt.get("hcore_calls", 0) + 1
        return r

    mf.get_hcore = gh
    stamps = []
    mf.callback = lambda envs: stamps.append(time.perf_counter())
    w0, c0 = time.perf_counter(), time.process_time()
    e = mf.kernel()
    t_kernel = time.perf_counter() - w0
    c_kernel = time.process_time() - c0
    gaps = [b - x for x, b in zip(stamps[:-1], stamps[1:])]
    gaps.sort()
    med_iter = gaps[len(gaps) // 2] if gaps else None
    rec = dict(
        code="pyscf",
        pyscf_version=pyscf.__version__,
        cell=a.cell,
        method=a.method,
        tag=a.tag,
        threads=a.threads,
        precision=a.precision,
        aux=a.aux,
        df=a.df,
        basis_repr=a.basis_repr,
        lindep_s=a.lindep_s,
        init_guess=a.init_guess,
        energy=float(e),
        converged=bool(mf.converged),
        cycles=int(mf.cycles),
        nao=int(cell.nao_nr()),
        # --df rsdf: its builder subclass overrides build(), so `builders` stays empty and
        # these come from the DF object itself.
        naux=int((builders[0] if builders else df).auxcell.nao_nr()),
        aux_metric=drops,
        omega=_num((builders[0] if builders else df).omega),
        mesh=_ints((builders[0] if builders else df).mesh),
        t_cell_s=t_cell,
        t_df_build_s=t_df,
        cpu_df_build_s=c_df,
        t_hcore_s=tt.get("hcore"),
        hcore_calls=tt.get("hcore_calls", 0),
        kmesh=a.kmesh,
        conv_rule="ferric" if a.match_ferric_conv else "pyscf",
        conv_tol=None if a.match_ferric_conv else a.conv_tol,
        conv_tol_grad=None if a.match_ferric_conv else a.conv_tol_grad,
        conv_check=bool(mf.conv_check),
        max_cycle=int(mf.max_cycle),
        ferric_conv_final=conv_trace[-1] if conv_trace else None,
        ferric_conv_trace=conv_trace or None,
        grid=a.grid if a.method == "pbe" else None,
        n_grid_points=int(mf.grids.weights.size) if a.method == "pbe" else None,
        t_kernel_s=t_kernel,
        cpu_kernel_s=c_kernel,
        t_scf_excl_hcore_s=t_kernel - tt.get("hcore", 0.0),
        s_per_iter_median=med_iter,
        t_total_s=time.perf_counter() - t_all,
        cderi_file_bytes=cderi_bytes,
        peak_rss_bytes=peak_rss_bytes(),
        rss_after_df_bytes=rss_after_df,
        load_start=load0,
        load_end=load1(),
    )
    shutil.rmtree(tmpdir, ignore_errors=True)
    line = json.dumps(rec)
    print(line)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
