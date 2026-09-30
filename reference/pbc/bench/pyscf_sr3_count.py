"""PySCF 2.13 GDF build (_RSGDFBuilder): per-stage timings + the SR 3-centre triplet count.

Companion of pyscf_bench.py (same cell files, same ferric-bundled basis/aux numbers, same
spherical basis, same eigendecomposed metric). One run = one GDF build of the Gamma-point RHF
density fit (`pscf.RHF(cell).density_fit(auxbasis)`, i.e. `pyscf.pbc.df.df.GDF` ->
`rsdf_builder._RSGDFBuilder`), then a count of the short-range 3-centre shell triplets that
libcint evaluates in that build's `outcore_auxe2` pass.

(a) TIMINGS: the builder's methods are wrapped (monkeypatched, no numerical change):
    setup            _RSGDFBuilder.build: omega/mesh guess, rs_cell split, supmol (images)
    SR j3c total     outcore_auxe2 (pass 1), which contains
      SR int3c init    gen_int3c_kernel: q_cond (sindex), cintopt
      SR int3c libcint every call of the int3c kernel = PBCfill_nr3c_drv (THE SR integrals)
      SR dd block      _outcore_dd_block: smooth x smooth AO pairs, FFT on the dd mesh
      (rest = HDF5 swap writes + merge_dd)
    j2c              get_2c2e (SR compact-aux metric + G-space metric)
    j2c decompose    decompose_j2c (eigh; j2c_eig_always as in pyscf_bench)
    LR pair FT       the supmol_ft ft kernel (AO-pair FT on the GDF mesh)
    LR aux FT        weighted_ft_ao (aux FT x coulG)
    LR GEMM          add_ft_j3c
    solve_cderi      j2c^-1/2 x j3c
    make_j3c total / df_build total ("rest" = total - the listed top-level parts)
  Wall = perf_counter, CPU = process_time (all threads of the process).

(b) COUNT: the SR 3-centre integrals are evaluated in C (lib/pbc/cint3c2e.c PBCint3c2e_loop,
  driven by fill_ints.c PBCfill_nr3c_drv). For every primitive-cell shell pair (ish >= jsh
  under aosym s2) with cell0_ovlp_mask set, every compact aux piece k (aux sits in cell 0 only),
  every rs-cell segment (steep/local piece of the shell; smooth x smooth pairs are removed by
  q_cond = INDEX_MIN) and every supmol IMAGE i of ish and j of jsh, libcint is called iff
      (theta r2 + log(r2 + 1e-30)) * 32 + cutoff + (log(w^2)/4 - l_k log(8 theta_k)) * 32 < sindex[i, j]
  with theta_k = w^2 a_k / (w^2 + a_k), theta = theta_k a_ij / (theta_k + a_ij) from the SMALLEST
  exponents of the three pieces, r2 = |R_k - (a_i R_i + a_j R_j)/a_ij|^2, cutoff =
  int(log(direct_scf_tol) * 32) and sindex = get_q_cond(supmol) (float32 arithmetic, as in C).
  Two counters, both over exactly those (i image, j image, k) libcint calls:
    --count replica   numpy re-implementation of that test on the builder's own arrays (fast);
    --count callback  PySCF's OWN C screening: PBCfill_nr3c_drv is re-invoked with
                      is_pbcintor = 0, which routes through fill_ints.c _assemble3c (the
                      line-for-line twin of PBCint3c2e_loop's screen, see its source) and calls
                      a ctypes callback in place of libcint; the callback counts and returns 0.
                      EXACT but slow (a Python call per triplet, serialised on the GIL).
    --count both      run both and assert equality (the toy-cell validation).
  Reported: libcint calls under aosym s2 (what the build does) and s1 (all ordered primitive-cell
  pairs: what ferric's counter walks), primitive triplets (sum nprim_i nprim_j nprim_k; an upper
  bound on libcint's primitive work, which screens primitives again with PTR_EXPCUTOFF), the
  contracted-column triplets (sum nctr_i nctr_j nctr_k), and the distinct surviving
  (i image, j image) pairs and pair translations dL = L_j - L_i.

COMPARISON WITH ferric's "rsgdf SR3 triplets" (crates/ferric-pbc/src/rsgdf.rs sr3_pair_image,
split.rs sr3_call): ferric counts (mu shell in cell 0, nu shell at image L, aux shell at image T)
triples passing its erfc bound, over ALL ORDERED shell pairs (nsh^2), one per libint call; with
range_split on, each pair is two calls ((chi_i, chi_j^c | X^c) + (chi_i^c, chi_j^s | X^c)) and only
compact aux pieces. PySCF images mu and nu and keeps aux in cell 0: (i_L1 j_L2 | k_0) ==
(i_0 j_{L2-L1} | k_{-L1}), a bijection of the triple sets, so the s1 count is the like-for-like
quantity. Remaining mismatches: (1) basis unit: PySCF shells are GENERAL contractions (one call
covers nctr_i nctr_j nctr_k of ferric's per-column segmented shells) split into steep/local/smooth
exponent pieces; ferric shells are per-column and (with range_split) split into compact/smooth
pieces by lambda omega^2 -- the "ctr" count is the closer match to ferric's shell unit; (2) the
screens are different bounds (PySCF: Schwarz-like log index + distance to the pair centre; ferric:
erfc charge bound + distance to the pair segment), different omega and different precision
derivations (PySCF cutoff = precision / lattice_sum_factor * 0.1); (3) PySCF moves smooth x smooth
AO pairs (all aux) and smooth aux to the FFT/AFT parts with its own ke-cutoff-based split, not
ferric's lambda omega^2 rule.

Threads: --threads N sets OMP/OPENBLAS/MKL threads BEFORE numpy is imported and lib.num_threads(N).

Usage (from the worktree root; PySCF lives in the main checkout's venv):
  PY=/home/matt/qc/ferric/.venv/bin/python
  OMP_NUM_THREADS=1 $PY reference/pbc/bench/pyscf_sr3_count.py dryice --threads 6 --precision 1e-12
  $PY reference/pbc/bench/pyscf_sr3_count.py dryice --no-build        # setup + count only
  $PY reference/pbc/bench/pyscf_sr3_count.py /abs/path/toy --count both  # <toy>.xyz/.lattice
  $PY reference/pbc/bench/pyscf_sr3_count.py --selftest                 # toy-cell validation
Output: one JSON line on stdout (prefixed "JSON "), a human summary, and
reference/pbc/bench/out/pyscf_sr3_<cell>_<aux>_<basis>_t<N>_p<prec>.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import tempfile
import time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "cell",
        nargs="?",
        default="dryice",
        help="basename of <cell>.xyz/.lattice here, or an absolute path prefix",
    )
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument(
        "--aux",
        default="cc-pvdz-ri",
        help="recorded dry-ice PySCF timings used cc-pvdz-ri",
    )
    ap.add_argument("--basis-repr", choices=["general", "segmented"], default="general")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument(
        "--precision",
        type=float,
        default=1e-12,
        help="cell.precision (recorded PySCF timings: 1e-12)",
    )
    ap.add_argument("--lindep-aux", type=float, default=1e-10)
    ap.add_argument("--max-memory", type=int, default=8000, help="PySCF max_memory, MB")
    ap.add_argument(
        "--count", choices=["replica", "callback", "both", "none"], default="replica"
    )
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="skip the j3c/j2c build: builder setup + count only",
    )
    ap.add_argument(
        "--selftest", action="store_true", help="run the toy-cell validation and exit"
    )
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = str(ARGS.threads)

import numpy as np  # noqa: E402
import pyscf  # noqa: E402
from pyscf import gto, lib  # noqa: E402
from pyscf.pbc import gto as pgto  # noqa: E402
from pyscf.pbc import scf as pscf  # noqa: E402
from pyscf.pbc.df import ft_ao, incore, rsdf_builder  # noqa: E402
from pyscf.scf import _vhf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pyscf_bench import BOHR, ferric_basis_pyscf, peak_rss_bytes, read_cell  # noqa: E402

LOG_ADJUST = incore.LOG_ADJUST  # 32, == lib/pbc/pbc.h
Builder = rsdf_builder._RSGDFBuilder

# ---------------------------------------------------------------------------------------------
# (a) timings
# ---------------------------------------------------------------------------------------------
TIMES: dict[str, list[float]] = {}
CAPTURED: dict[str, object] = {}


def _acc(name, w, c):
    row = TIMES.setdefault(name, [0.0, 0.0, 0])
    row[0] += w
    row[1] += c
    row[2] += 1


def _timed_fn(fn, name):
    def wrapper(*a, **k):
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            return fn(*a, **k)
        finally:
            _acc(name, time.perf_counter() - w0, time.process_time() - c0)

    return wrapper


def install_timers():
    """Wrap the builder stages. Idempotent per process (call once)."""
    for attr, name in [
        ("build", "setup (omega, rs_cell, supmol)"),
        ("outcore_auxe2", "SR j3c total (pass 1)"),
        ("_outcore_dd_block", "SR dd block (FFT)"),
        ("get_2c2e", "j2c (get_2c2e)"),
        ("decompose_j2c", "j2c decompose"),
        ("weighted_ft_ao", "LR aux FT (weighted_ft_ao)"),
        ("add_ft_j3c", "LR GEMM (add_ft_j3c)"),
        ("solve_cderi", "solve_cderi"),
    ]:
        setattr(Builder, attr, _timed_fn(getattr(Builder, attr), name))

    make_j3c = Builder.make_j3c

    def make_j3c_capture(self, *a, **k):
        CAPTURED["builder"] = self
        return _timed_fn(make_j3c, "make_j3c total")(self, *a, **k)

    Builder.make_j3c = make_j3c_capture

    gen3 = incore.Int3cBuilder.gen_int3c_kernel

    def gen3_timed(self, *a, **k):
        w0, c0 = time.perf_counter(), time.process_time()
        kern = gen3(self, *a, **k)
        _acc(
            "SR int3c init (q_cond, cintopt)",
            time.perf_counter() - w0,
            time.process_time() - c0,
        )
        return _timed_fn(kern, "SR int3c libcint (PBCfill_nr3c)")

    incore.Int3cBuilder.gen_int3c_kernel = gen3_timed

    genft = ft_ao.ExtendedMole.gen_ft_kernel

    def genft_timed(self, *a, **k):
        return _timed_fn(genft(self, *a, **k), "LR pair FT (ft kernel)")

    ft_ao.ExtendedMole.gen_ft_kernel = genft_timed


# ---------------------------------------------------------------------------------------------
# (b) the SR triplet count
# ---------------------------------------------------------------------------------------------
class Sr3Setup:
    """Exactly the arrays gen_int3c_kernel hands to PBCfill_nr3c_drv (Gamma point)."""

    def __init__(self, b, intor="int3c2e"):
        cell = b.cell
        assert b.supmol.bas_mask.shape[0] == 1, "Gamma point only (bvk_ncells == 1)"
        self.b = b
        self.cell = cell
        self.supmol = supmol = b.supmol
        if b.exclude_d_aux and cell.dimension > 0:
            self.rs_aux = b.rs_auxcell.compact_basis_cell()
        else:
            self.rs_aux = b.rs_auxcell
        self.intor = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), None)[0]
        self.cutoff = b.direct_scf_tol
        self.log_cutoff = int(np.log(self.cutoff) * LOG_ADJUST)
        self.sindex = b.get_q_cond(supmol)
        self.nbasp = cell.nbas
        ovlp_mask = self.sindex > self.log_cutoff
        bvk = lib.condense("np.any", ovlp_mask, supmol.sh_loc)
        self.cell0_mask = (
            bvk.reshape(1, self.nbasp, 1, self.nbasp)
            .any(axis=2)
            .any(axis=0)
            .astype(np.int8)
        )
        self.atm, self.bas, self.env = gto.conc_env(
            supmol._atm,
            supmol._bas,
            supmol._env,
            self.rs_aux._atm,
            self.rs_aux._bas,
            self.rs_aux._env,
        )
        self.cell0_ao_loc = incore._conc_locs(cell.ao_loc, b.auxcell.ao_loc)
        self.seg_loc = incore._conc_locs(supmol.seg_loc, self.rs_aux.sh_loc)
        self.seg2sh = incore._conc_locs(supmol.seg2sh, np.arange(self.rs_aux.nbas + 1))
        self.omega = float(supmol._env[gto.PTR_RANGE_OMEGA])
        assert self.omega < 0, "expected the SR (erfc) supmol"
        # per concatenated shell: smallest exponent (C: env[PTR_EXP + nprim - 1]), l, nprim, nctr, coords
        bas = self.bas
        self.nprim = bas[:, gto.NPRIM_OF].astype(np.int64)
        self.nctr = bas[:, gto.NCTR_OF].astype(np.int64)
        self.l = bas[:, gto.ANG_OF].astype(np.int64)
        self.amin = self.env[bas[:, gto.PTR_EXP] + bas[:, gto.NPRIM_OF] - 1]
        ptr = self.atm[bas[:, gto.ATOM_OF], gto.PTR_COORD]
        self.xyz = self.env[ptr[:, None] + np.arange(3)]
        # image index of every supmol shell (bas_mask order: [bvk, rs_bas, img]) -> translation
        nimgs = supmol.bas_mask.shape[2]
        self.img = np.where(supmol.bas_mask.ravel())[0] % nimgs
        self.Ls = supmol.Ls

    def aux_segments(self):
        """(concatenated shell ids of the compact aux pieces, in C's ksh, kseg order)."""
        out = []
        nbas_bvk = self.nbasp  # bvk_ncells == 1
        for ksh in range(self.b.auxcell.nbas):
            kb = ksh + nbas_bvk
            for kseg in range(self.seg_loc[kb], self.seg_loc[kb + 1]):
                out.append(self.seg2sh[kseg])
        return np.asarray(out, dtype=np.int64)


def _new_tally():
    return dict(calls=0, prim=0, ctr=0, pairs=set(), dls=set(), pair_dl=set())


def _tally_finish(t):
    return dict(
        libcint_calls=int(t["calls"]),
        prim_triplets=int(t["prim"]),
        ctr_triplets=int(t["ctr"]),
        image_pairs=len(t["pairs"]),
        pair_translations=len(t["dls"]),
        shellpair_translations=len(t["pair_dl"]),
    )


def count_replica(s: Sr3Setup, aosym="s2", keep_triples=False):
    """numpy twin of PBCint3c2e_loop's SR screen (float32, as in C). Returns the tallies."""
    f32 = np.float32
    omega = f32(s.omega)
    omega2 = f32(omega * omega)
    ks = s.aux_segments()
    ak = s.amin[ks].astype(f32)
    lk = s.l[ks].astype(f32)
    theta_k = f32(omega2) * ak / (omega2 + ak)
    fac = np.log(omega2) / f32(4) - lk * np.log(theta_k * f32(8))
    ij_cut = (f32(s.log_cutoff) + fac * f32(LOG_ADJUST)).astype(f32)
    rk = s.xyz[ks].astype(f32)
    kprim, kctr = s.nprim[ks], s.nctr[ks]
    sindex = s.sindex
    tally = _new_tally()
    triples = [] if keep_triples else None
    nbasp = s.nbasp
    loose = f32(ij_cut.min()) + f32(
        np.log(f32(1e-30)) * LOG_ADJUST
    )  # smallest possible LHS
    for ish in range(nbasp):
        for jsh in range(nbasp if aosym == "s1" else ish + 1):
            if not s.cell0_mask[ish, jsh]:
                continue
            for iseg in range(s.seg_loc[ish], s.seg_loc[ish + 1]):
                sh_i = np.arange(s.seg2sh[iseg], s.seg2sh[iseg + 1])
                if sh_i.size == 0:
                    continue
                ai = f32(s.amin[sh_i[0]])
                for jseg in range(s.seg_loc[jsh], s.seg_loc[jsh + 1]):
                    sh_j = np.arange(s.seg2sh[jseg], s.seg2sh[jseg + 1])
                    if sh_j.size == 0:
                        continue
                    aj = f32(s.amin[sh_j[0]])
                    aij = f32(ai + aj)
                    ci, cj = f32(ai / aij), f32(aj / aij)
                    S = sindex[np.ix_(sh_i, sh_j)].astype(f32)
                    ii, jj = np.nonzero(
                        S > loose
                    )  # can pass for SOME k (exact pre-filter)
                    if ii.size == 0:
                        continue
                    Sp = S[ii, jj]
                    # C: xcond = (float)(ci * x_i) + cj * x_j
                    ctr = (ci * s.xyz[sh_i[ii]]).astype(f32) + cj * s.xyz[
                        sh_j[jj]
                    ].astype(f32)
                    theta = theta_k * aij / (theta_k + aij)  # (nk,)
                    ok = np.empty((ks.size, ii.size), dtype=bool)
                    step = max(
                        1, (1 << 22) // ks.size
                    )  # (nk, step) float32 temporaries
                    for p0 in range(0, ii.size, step):
                        c = ctr[p0 : p0 + step]
                        d = rk[:, None, :] - c[None, :, :]
                        r2 = (
                            d[..., 0] * d[..., 0]
                            + d[..., 1] * d[..., 1]
                            + d[..., 2] * d[..., 2]
                        )
                        tr2 = theta[:, None] * r2 + np.log(r2 + f32(1e-30))
                        ok[:, p0 : p0 + step] = (
                            tr2 * f32(LOG_ADJUST) + ij_cut[:, None]
                            < Sp[None, p0 : p0 + step]
                        )
                    nk_per_pair = ok.sum(axis=0)
                    kk, pp = np.nonzero(ok)
                    if kk.size == 0:
                        continue
                    si, sj = sh_i[ii[pp]], sh_j[jj[pp]]
                    tally["calls"] += kk.size
                    tally["prim"] += int((s.nprim[si] * s.nprim[sj] * kprim[kk]).sum())
                    tally["ctr"] += int((s.nctr[si] * s.nctr[sj] * kctr[kk]).sum())
                    live = nk_per_pair > 0
                    li, lj = sh_i[ii[live]], sh_j[jj[live]]
                    tally["pairs"].update(zip(li.tolist(), lj.tolist()))
                    dl = np.round(s.Ls[s.img[lj]] - s.Ls[s.img[li]], 6)
                    for v in map(tuple, dl.tolist()):
                        tally["dls"].add(v)
                        tally["pair_dl"].add((ish, jsh) + v)
                    if triples is not None:
                        triples.extend(zip(si.tolist(), sj.tolist(), ks[kk].tolist()))
    return _tally_finish(tally), triples


_CB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
)


def count_callback(s: Sr3Setup, aosym="s2", keep_triples=False):
    """PySCF's own C screen (fill_ints.c _assemble3c via PBCfill_nr3c_drv, is_pbcintor = 0) with a
    ctypes callback in place of libcint. Exact; one Python call per surviving triplet."""
    nprim, nctr = s.nprim, s.nctr
    tally = _new_tally()
    triples = [] if keep_triples else None
    img, Ls = s.img, s.Ls
    # supmol shell -> primitive-cell shell (for the pair_dl tally)
    sh2cell = np.repeat(np.arange(s.nbasp), np.diff(s.supmol.sh_loc))

    def cb(buf, dims, shls, atm, natm, bas, nbas, env, opt, cache):
        i, j, k = shls[0], shls[1], shls[2]
        tally["calls"] += 1
        tally["prim"] += int(nprim[i] * nprim[j] * nprim[k])
        tally["ctr"] += int(nctr[i] * nctr[j] * nctr[k])
        if (i, j) not in tally["pairs"]:
            tally["pairs"].add((i, j))
            v = tuple(np.round(Ls[img[j]] - Ls[img[i]], 6).tolist())
            tally["dls"].add(v)
            tally["pair_dl"].add((int(sh2cell[i]), int(sh2cell[j])) + v)
        if triples is not None:
            triples.append((i, j, k))
        return 0  # no value: _assemble3c adds nothing and the sort is skipped

    cfun = _CB(cb)
    cell = s.cell
    naux = s.b.auxcell.nao
    nao = cell.nao
    nrow = nao * nao if aosym == "s1" else nao * (nao + 1) // 2
    out = np.zeros((1, nrow, naux))
    cintopt = _vhf.make_cintopt(s.atm, s.bas, s.env, s.intor)
    dims = s.cell0_ao_loc[1:] - s.cell0_ao_loc[:-1]
    dijk = int(dims[: s.nbasp].max()) ** 2 * int(dims[s.nbasp :].max())
    cache_size = (
        max(
            incore._get_cache_size(cell, s.intor),
            incore._get_cache_size(s.rs_aux, s.intor),
        )
        + 3 * dijk
    )
    expLk = np.ones((s.supmol.bvkmesh_Ls.shape[0], 1))
    reindex_k = np.zeros(1, dtype=np.int32)
    shls_slice = (ctypes.c_int * 6)(
        0, s.nbasp, 0, s.nbasp, s.nbasp, s.nbasp + s.b.auxcell.nbas
    )
    nimgs = s.supmol.bas_mask.shape[2]
    libpbc = incore.libpbc
    libpbc.PBCfill_nr3c_drv(
        cfun,
        getattr(libpbc, f"PBCfill_nr3c_g{aosym}"),
        ctypes.c_int(0),  # is_pbcintor = 0 -> _assemble3c + our callback
        out.ctypes.data_as(ctypes.c_void_p),
        np.zeros(0).ctypes.data_as(ctypes.c_void_p),
        expLk.ctypes.data_as(ctypes.c_void_p),
        np.zeros_like(expLk).ctypes.data_as(ctypes.c_void_p),
        reindex_k.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(1),
        ctypes.c_int(1),
        ctypes.c_int(nimgs),
        ctypes.c_int(1),
        ctypes.c_int(s.nbasp),
        ctypes.c_int(1),
        s.seg_loc.ctypes.data_as(ctypes.c_void_p),
        s.seg2sh.ctypes.data_as(ctypes.c_void_p),
        s.cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
        shls_slice,
        s.cell0_mask.ctypes.data_as(ctypes.c_void_p),
        s.sindex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(s.log_cutoff),
        cintopt,
        ctypes.c_int(cache_size),
        s.atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(s.supmol.natm),
        s.bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(s.supmol.nbas),
        s.env.ctypes.data_as(ctypes.c_void_p),
    )
    assert not out.any(), "callback returned 0: nothing may be written"
    return _tally_finish(tally), triples


def brute_force_size(s: Sr3Setup, aosym="s2"):
    """Every (i image, j image, k piece) the C loop visits before the per-triplet test (cell0 mask
    applied, no sindex test): the unscreened upper bound."""
    nk = s.aux_segments().size
    tot = 0
    for ish in range(s.nbasp):
        for jsh in range(s.nbasp if aosym == "s1" else ish + 1):
            if not s.cell0_mask[ish, jsh]:
                continue
            ni = sum(
                s.seg2sh[g + 1] - s.seg2sh[g]
                for g in range(s.seg_loc[ish], s.seg_loc[ish + 1])
            )
            nj = sum(
                s.seg2sh[g + 1] - s.seg2sh[g]
                for g in range(s.seg_loc[jsh], s.seg_loc[jsh + 1])
            )
            tot += int(ni) * int(nj) * nk
    return tot


def run_counts(b, how):
    s = Sr3Setup(b)
    res = dict(
        log_cutoff=s.log_cutoff,
        direct_scf_tol=s.cutoff,
        n_aux_pieces=int(s.aux_segments().size),
        supmol_nbas=int(s.supmol.nbas),
        supmol_nimgs=int(s.supmol.bas_mask.shape[2]),
        rs_cell_nbas=int(b.rs_cell.nbas),
        cell_nbas=int(s.nbasp),
        aux_nbas=int(b.auxcell.nbas),
        cell0_pairs_s2=int(np.tril(s.cell0_mask).sum()),
        cell0_pairs_s1=int(s.cell0_mask.sum()),
    )
    res["unscreened_s2"] = brute_force_size(s, "s2")
    res["unscreened_s1"] = brute_force_size(s, "s1")
    for aosym in ("s2", "s1"):
        if how in ("replica", "both"):
            w0 = time.perf_counter()
            res[f"replica_{aosym}"], _ = count_replica(s, aosym)
            res[f"replica_{aosym}"]["count_wall_s"] = time.perf_counter() - w0
        if how in ("callback", "both"):
            w0 = time.perf_counter()
            res[f"callback_{aosym}"], _ = count_callback(s, aosym)
            res[f"callback_{aosym}"]["count_wall_s"] = time.perf_counter() - w0
        if how == "both":
            a, c = dict(res[f"replica_{aosym}"]), dict(res[f"callback_{aosym}"])
            a.pop("count_wall_s"), c.pop("count_wall_s")
            res[f"replica_equals_callback_{aosym}"] = a == c
    return res


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------
def build_cell(atoms, lat, basis, precision, max_memory, verbose):
    cell = pgto.Cell()
    cell.a = lat
    cell.unit = "B"
    cell.atom = atoms
    cell.basis = basis
    cell.cart = False
    cell.precision = precision
    cell.max_memory = max_memory
    cell.verbose = verbose
    cell.build()
    return cell


def make_df(cell, auxbasis, lindep_aux):
    Builder.j2c_eig_always = True  # as pyscf_bench (GDF ignores the instance attribute)
    mf = pscf.RHF(cell, exxdiv="ewald").density_fit(auxbasis=auxbasis)
    mf.with_df.linear_dep_threshold = lindep_aux
    return mf.with_df


def run(a, cell_atoms=None):
    lib.num_threads(a.threads)
    atoms, lat = cell_atoms if cell_atoms is not None else read_cell(a.cell)
    symbols = [sym for sym, _ in atoms]
    TIMES.clear()
    CAPTURED.clear()
    load0 = os.getloadavg()
    cell = build_cell(
        atoms,
        lat,
        ferric_basis_pyscf(a.basis, symbols, a.basis_repr),
        a.precision,
        a.max_memory,
        a.verbose,
    )
    auxbasis = ferric_basis_pyscf(a.aux, symbols, "general")
    df = make_df(cell, auxbasis, a.lindep_aux)
    w0, c0 = time.perf_counter(), time.process_time()
    if a.no_build:
        from pyscf.pbc.df.df import make_modrho_basis

        auxcell = make_modrho_basis(cell, auxbasis, df.exp_to_discard)
        b = Builder(cell, auxcell, df.kpts)
        b.linear_dep_threshold = a.lindep_aux
        b.build()
    else:
        df.build()
        b = CAPTURED["builder"]
    _acc(
        "df_build total" if not a.no_build else "setup only (no build)",
        time.perf_counter() - w0,
        time.process_time() - c0,
    )
    top = [
        "setup (omega, rs_cell, supmol)",
        "SR j3c total (pass 1)",
        "j2c (get_2c2e)",
        "j2c decompose",
        "LR pair FT (ft kernel)",
        "LR aux FT (weighted_ft_ao)",
        "LR GEMM (add_ft_j3c)",
        "solve_cderi",
    ]
    if "df_build total" in TIMES:
        TIMES["rest (df_build - listed top-level)"] = [
            TIMES["df_build total"][0] - sum(TIMES[k][0] for k in top if k in TIMES),
            TIMES["df_build total"][1] - sum(TIMES[k][1] for k in top if k in TIMES),
            1,
        ]
    sr = [
        k
        for k in (
            "SR int3c init (q_cond, cintopt)",
            "SR int3c libcint (PBCfill_nr3c)",
            "SR dd block (FFT)",
        )
        if k in TIMES
    ]
    if "SR j3c total (pass 1)" in TIMES:
        TIMES["SR rest (h5 swap, merge_dd)"] = [
            TIMES["SR j3c total (pass 1)"][0] - sum(TIMES[k][0] for k in sr),
            TIMES["SR j3c total (pass 1)"][1] - sum(TIMES[k][1] for k in sr),
            1,
        ]
    stages = {k: dict(wall_s=v[0], cpu_s=v[1], calls=v[2]) for k, v in TIMES.items()}
    counts = run_counts(b, a.count) if a.count != "none" else None
    rs_cell = b.rs_cell
    rec = dict(
        code="pyscf",
        pyscf_version=pyscf.__version__,
        cell=a.cell,
        args={k: v for k, v in vars(a).items()},
        threads=a.threads,
        precision=cell.precision,
        omega=float(b.omega),
        mesh=[int(v) for v in b.mesh],
        ke_cutoff=float(b.ke_cutoff),
        nao=int(cell.nao_nr()),
        naux=int(b.auxcell.nao_nr()),
        cell_rcut=float(cell.rcut),
        rs_cell_bas_types=dict(
            zip(
                ("steep", "local", "smooth"),
                (
                    int((rs_cell.bas_type == t).sum())
                    for t in (ft_ao.STEEP_BASIS, ft_ao.LOCAL_BASIS, ft_ao.SMOOTH_BASIS)
                ),
            )
        ),
        exclude_dd_block=bool(b.exclude_dd_block),
        exclude_d_aux=bool(b.exclude_d_aux),
        stages=stages,
        sr3=counts,
        loadavg_start=load0,
        loadavg_end=os.getloadavg(),
        peak_rss_bytes=peak_rss_bytes(),
    )
    return rec


def summary(rec):
    print(
        f"[sr3] {rec['cell']}: PySCF {rec['pyscf_version']}  threads {rec['threads']}  precision {rec['precision']:.0e}"
    )
    print(
        f"[sr3] omega {rec['omega']:.6f} Bohr^-1  LR mesh {rec['mesh']}  ke_cutoff {rec['ke_cutoff']:.3f}  "
        f"nao {rec['nao']}  naux {rec['naux']}  rs_cell {rec['rs_cell_bas_types']}"
    )
    for k, v in rec["stages"].items():
        print(
            f"[sr3]   {k:40s} wall {v['wall_s']:9.3f} s  cpu {v['cpu_s']:9.3f} s  calls {v['calls']}"
        )
    c = rec["sr3"]
    if c:
        print(
            f"[sr3] SR screen: direct_scf_tol {c['direct_scf_tol']:.3e} (log_cutoff {c['log_cutoff']})  "
            f"supmol nbas {c['supmol_nbas']} ({c['supmol_nimgs']} image slots)  compact aux pieces {c['n_aux_pieces']}"
        )
        for key in ("replica_s2", "callback_s2", "replica_s1", "callback_s1"):
            if key in c:
                t = c[key]
                print(
                    f"[sr3]   {key:12s} libcint calls {t['libcint_calls']:>12d}  prim {t['prim_triplets']:>14d}  "
                    f"ctr {t['ctr_triplets']:>12d}  image pairs {t['image_pairs']:>9d}  dL {t['pair_translations']:>5d}  "
                    f"(count {t['count_wall_s']:.1f} s)"
                )
        print(
            f"[sr3]   unscreened (cell0-mask only): s2 {c['unscreened_s2']}  s1 {c['unscreened_s1']}"
        )
        for aosym in ("s2", "s1"):
            if f"replica_equals_callback_{aosym}" in c:
                print(
                    f"[sr3]   replica == C callback ({aosym}): {c[f'replica_equals_callback_{aosym}']}"
                )
        print(
            "[sr3] compare the s1 'libcint calls' (or 'ctr') with ferric's 'rsgdf SR3 triplets'"
        )


# ---------------------------------------------------------------------------------------------
# self-test (toy cell, <~ a few s of one core)
# ---------------------------------------------------------------------------------------------
TOYS = {
    # name: (xyz body, cubic box in Angstrom, basis, aux)
    "h2": ("H 0.0 0.0 0.0\nH 0.74 0.0 0.0\n", 3.0, "sto-3g", "cc-pvdz-ri"),
    "lih": ("Li 0.0 0.0 0.0\nH 1.6 0.0 0.0\n", 4.0, "cc-pvdz", "cc-pvdz-ri"),
}


def _toy(tmp, name):
    body, box, basis, aux = TOYS[name]
    pre = os.path.join(tmp, "toy_" + name)
    with open(pre + ".xyz", "w") as f:
        f.write(f"{body.count(chr(10))}\ntoy {name}\n{body}")
    np.savetxt(pre + ".lattice", np.eye(3) * box / BOHR)
    return pre, basis, aux


def _toy_builder(pre, basis, aux, prec):
    from pyscf.pbc.df.df import make_modrho_basis

    atoms, lat = read_cell(pre)
    symbols = sorted({sym for sym, _ in atoms})
    cell = build_cell(atoms, lat, ferric_basis_pyscf(basis, symbols), prec, 2000, 0)
    auxcell = make_modrho_basis(cell, ferric_basis_pyscf(aux, symbols), None)
    b = Builder(cell, auxcell, np.zeros((1, 3)))
    b.build()
    return b


def _sr_tensor_from_triples(s: Sr3Setup, triples):
    """Sum libcint (i j | k)_erfc over the given supmol triples into the cell-0 s1 layout."""
    nao, naux = s.cell.nao, s.b.auxcell.nao
    out = np.zeros((nao, nao, naux))
    cell_of = np.repeat(np.arange(s.nbasp), np.diff(s.supmol.sh_loc))
    aux_of = np.repeat(np.arange(s.b.auxcell.nbas), np.diff(s.rs_aux.sh_loc))
    ao, xo = s.cell.ao_loc, s.b.auxcell.ao_loc
    nsup = s.supmol.nbas
    for i, j, k in triples:
        v = gto.moleintor.getints3c(
            s.intor,
            s.atm,
            s.bas,
            s.env,
            shls_slice=(i, i + 1, j, j + 1, k, k + 1),
            aosym="s1",
        )
        ci, cj, ck = cell_of[i], cell_of[j], aux_of[k - nsup]
        out[ao[ci] : ao[ci + 1], ao[cj] : ao[cj + 1], xo[ck] : xo[ck + 1]] += v.reshape(
            ao[ci + 1] - ao[ci], ao[cj + 1] - ao[cj], xo[ck + 1] - xo[ck]
        )
    return out.reshape(nao * nao, naux)


def selftest():
    """Toy-cell validation, ~10 s of one core."""
    lib.num_threads(1)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        # 1. replica == PySCF's own C screen, s2 and s1, all tallies; monotone in precision.
        for toy, precs in (("h2", (1e-6, 1e-8, 1e-10, 1e-12)), ("lih", (1e-6,))):
            pre, basis, aux = _toy(tmp, toy)
            calls = []
            for prec in precs:
                s = Sr3Setup(_toy_builder(pre, basis, aux, prec))
                eq = True
                for aosym in ("s2", "s1"):
                    r, c = count_replica(s, aosym)[0], count_callback(s, aosym)[0]
                    eq &= r == c
                    if aosym == "s1":
                        calls.append(c["libcint_calls"])
                        print(
                            f"[selftest] {toy} prec {prec:.0e}: omega {s.b.omega:.4f} calls s1 {c['libcint_calls']} "
                            f"prim {c['prim_triplets']} unscreened s1 {brute_force_size(s, 's1')}  replica==C {eq}"
                        )
                ok &= eq
            mono = all(x <= y for x, y in zip(calls, calls[1:]))
            print(
                f"[selftest] {toy}: s1 count non-decreasing as precision tightens: {mono} {calls}"
            )
            ok &= mono

        # 2. unscreened anchor: every visited triplet passes -> both counters == the brute-force product.
        pre, basis, aux = _toy(tmp, "h2")
        b = _toy_builder(pre, basis, aux, 1e-6)
        s = Sr3Setup(b)
        s_open = Sr3Setup(b)
        s_open.log_cutoff = -(2**30)
        s_open.sindex = np.full_like(
            s.sindex, 32000
        )  # every pair passes, even the dd block
        s_open.cell0_mask[:] = 1
        n_open = count_callback(s_open, "s1")[0]["libcint_calls"]
        n_open_r = count_replica(s_open, "s1")[0]["libcint_calls"]
        bf = brute_force_size(s_open, "s1")
        print(
            f"[selftest] unscreened anchor: C callback {n_open}  replica {n_open_r}  brute force {bf}"
        )
        ok &= n_open == n_open_r == bf

        # 3. the counted triples ARE the SR integrals PySCF sums: libcint over them == PySCF's kernel.
        t_rep = count_replica(s, "s1", keep_triples=True)[1]
        t_cb = count_callback(s, "s1", keep_triples=True)[1]
        same = sorted(t_rep) == sorted(t_cb)
        kern = b.gen_int3c_kernel("int3c2e", "s1", j_only=True, rs_auxcell=s.rs_aux)
        ref = kern()[0][0]
        mine = _sr_tensor_from_triples(s, t_cb)
        kept = set(t_cb)
        # screened-out triples, excluding the smooth x smooth pairs PySCF does by FFT (q_cond INDEX_MIN)
        cut = [
            t
            for t in count_callback(s_open, "s1", keep_triples=True)[1]
            if t not in kept and s.sindex[t[0], t[1]] > rsdf_builder.INDEX_MIN
        ]
        rng = np.random.default_rng(0)
        drop = [
            cut[n]
            for n in rng.choice(len(cut), size=min(len(cut), 20000), replace=False)
        ]
        dropped = _sr_tensor_from_triples(s, drop)
        err = float(abs(ref - mine).max())
        print(
            f"[selftest] h2 1e-6: triple SETS identical (replica vs C) {same}; "
            f"max|PySCF SR j3c - libcint summed over the counted triples| = {err:.2e} "
            f"(max|j3c| {abs(ref).max():.2e}; {len(drop)} of {len(cut)} screened-out non-dd triples sum to "
            f"max {abs(dropped).max():.2e})"
        )
        ok &= same and err < 1e-12
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main(a=None):
    a = a if a is not None else ARGS
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    install_timers()
    rec = run(a)
    print("JSON " + json.dumps(rec))
    summary(rec)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    name = os.path.basename(a.cell)
    out = a.out or os.path.join(
        HERE,
        "out",
        f"pyscf_sr3_{name}_{a.aux}_{a.basis}_t{a.threads}_p{a.precision:.0e}.json",
    )
    json.dump(rec, open(out, "w"), indent=1)
    print(f"[sr3] -> {out}")


if __name__ == "__main__":
    main()
