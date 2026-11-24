"""PySCF references for the NEVPT RDM tests (crates/ferric-ci/tests/casci_rdm3.rs).

Input: a dump of one ferric CAS-CI run, written by

    OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-ci --example casci_rdm_dump -- \\
        <system> <dump.json>

for system in h2o, n2_r1.10, n2_r2.20 (the windows of
crates/ferric-ci/tests/casci_rdm_common). The dump holds ferric's active-space
integrals, the alpha/beta strings and ferric's CI vector.

What PySCF does, on ferric's CI vector and ferric's (aa|aa) only (no PySCF
orbitals, no PySCF eigensolve):

1. The CI vector is mapped to PySCF's string order with `cistring.str2addr`
   (both codes use ascending creation order within a string, so no per-string
   sign; gen_casci_rdm_refs.py checks the same mapping through make_rdm12).
2. dm1..dm4 = `fci.rdm.make_dm1234("FCI4pdm_kern_sf", ...)` followed by
   `fci.rdm.reorder_dm1234` (normal-ordered spin-free RDMs,
   dm_k[p1,q1,..,pk,qk] = sum_spins <a+p1 .. a+pk a_qk .. a_q1>). dm4 is used
   for checks only and is not written.
3. f3ca / f3ac = `mrpt.nevpt2._contract4pdm` with kernels NEVPTkern_cedf_aedf /
   NEVPTkern_aedf_ecdf on the chemist (aa|aa) block, as PySCF's NEVPT2 driver
   calls it. The generator asserts their definition against the E-product
   4-RDM R4[a,b,c,d,e,f,g,h] = <E_ab E_cd E_ef E_gh> (make_dm1234 without
   reordering):
       f3ca[a,b,c,d,e,f] = sum_ijk (ki|fj) R4[a,b,c,d,e,j,k,i]
       f3ac[a,b,c,d,e,f] = sum_ijk (ik|je) R4[a,b,c,d,j,f,i,k]
   (the einsum comments in make_a16/make_a22 that use the second argument
   order; the comment `kbij,rpqjkiac` above the first f3ca use in make_a16 does
   not match the kernel).
4. hdm1..hdm3 = `mrpt.nevpt2.make_hdm1/2/3` on the E-product RDMs of
   `fci.rdm.make_dm123` (no reordering), as PySCF's NEVPT2 driver passes them.
5. Checks (refused above the bar): partial traces of dm3 and dm2, and
   reorder consistency between make_dm123 and make_dm1234.

Storage. Values are rounded to 15 decimal places (absolute error <= 5e-16,
four orders below the test bar 100 * N_det * eps * max|dm_k| for every tensor
here). dm3 and hdm3 are stored as their symmetry-unique elements
(`dm3_unique`, `hdm3_unique`): the 12 index images under a permutation of the
three index pairs and the simultaneous swap within every pair (hermiticity of
a real state); pairs are (0,1),(2,3),(4,5) for dm3 and (0,3),(1,4),(2,5) for
hdm3. Kept are the row-major index tuples equal to the smallest of their
images, in row-major order. The generator refuses a PySCF tensor that breaks
the symmetry beyond the bar. f3ca/f3ac have no such symmetry; they are stored
in full for h2o only (`f3ca`, `f3ac`, 4096 elements), and for every system as
`N_PROJ` projections <P_k, f3> onto +-1 tensors P_k from the splitmix64
finalizer of `k * 0x9E3779B97F4A7C15 + i * 0xBF58476D1CE4E5B9 + 1` (i the
row-major element index, sign = top bit), so that the 400-determinant N2
files stay small.

Output: testdata/reference/casci_<system>_rdm3.json.

Run (light; seconds):
    uv run --no-sync python scripts/gen_casci_rdm3_refs.py <dump.json> [<dump.json> ...]
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "testdata/reference"
EPS = 2.220446049250313e-16
DECIMALS = 15
N_PROJ = 32
FULL_F3 = {"h2o"}
DM3_PAIRS = ((0, 1), (2, 3), (4, 5))
HDM3_PAIRS = ((0, 3), (1, 4), (2, 5))


def q(a):
    import numpy as np

    return [float(x) for x in np.round(np.asarray(a).ravel(), DECIMALS)]


def images(idx, pairs):
    from itertools import permutations

    out = []
    for perm in permutations(range(3)):
        for swap in (False, True):
            new = [0] * 6
            for dst, src in zip(pairs, (pairs[k] for k in perm)):
                a, b = idx[src[0]], idx[src[1]]
                if swap:
                    a, b = b, a
                new[dst[0]], new[dst[1]] = a, b
            out.append(tuple(new))
    return out


def unique(t, pairs, bar):
    """Symmetry-unique elements of a rank-6 tensor; refuses a broken symmetry."""
    import numpy as np

    n = t.shape[0]
    vals, worst = [], 0.0
    for idx in np.ndindex(*t.shape):
        ims = images(idx, pairs)
        worst = max(worst, max(abs(t[im] - t[idx]) for im in ims))
        if min(ims) == idx:
            vals.append(float(t[idx]))
    assert worst < bar, f"symmetry broken by {worst:.2e} >= {bar:.2e} (n={n})"
    return [float(x) for x in np.round(vals, DECIMALS)], worst


def signs(k, size):
    import numpy as np

    m = np.uint64(0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        i = np.arange(size, dtype=np.uint64)
        z = (
            np.uint64(k) * np.uint64(0x9E3779B97F4A7C15)
            + i * np.uint64(0xBF58476D1CE4E5B9)
            + np.uint64(1)
        ) & m
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return np.where((z >> np.uint64(63)) == 0, 1.0, -1.0)


def projections(t):
    flat = t.ravel()
    return [float(signs(k, flat.size) @ flat) for k in range(N_PROJ)]


def gen(dump_path: Path) -> Path:
    import numpy as np
    import pyscf
    from pyscf.fci import cistring, rdm
    from pyscf.mrpt import nevpt2

    d = json.loads(dump_path.read_text())
    n = int(d["n_active"])
    na, nb = (int(x) for x in d["n_elec_active"])
    nelec = (na, nb)
    ne = na + nb
    eri = np.array(d["eri"]).reshape(n, n, n, n)
    na_str, nb_str = comb(n, na), comb(n, nb)
    ndet = na_str * nb_str

    a_addr = [int(cistring.str2addr(n, na, int(s))) for s in d["alpha_strings"]]
    b_addr = [int(cistring.str2addr(n, nb, int(s))) for s in d["beta_strings"]]
    cf = np.array(d["ci_vector"]).reshape(len(a_addr), len(b_addr))
    ci = np.zeros((na_str, nb_str))
    ci[np.ix_(a_addr, b_addr)] = cf
    norm = float(np.linalg.norm(ci))
    assert abs(norm - 1.0) < 1e-12, f"CI vector norm {norm}"

    raw = rdm.make_dm1234("FCI4pdm_kern_sf", ci, ci, n, nelec)
    r4 = raw[3].copy()
    dm1, dm2, dm3, dm4 = rdm.reorder_dm1234(*[x.copy() for x in raw])
    e1, e2, e3 = rdm.make_dm123("FCI3pdm_kern_sf", ci, ci, n, nelec)
    _, n2c, n3c = rdm.reorder_dm123(e1.copy(), e2.copy(), e3.copy())

    checks = {}
    scale3 = float(np.abs(dm3).max())
    checks["dm3_partial_trace"] = float(
        np.abs(np.einsum("pqrsxx->pqrs", dm3) - (ne - 2) * dm2).max()
    )
    checks["dm2_partial_trace"] = float(
        np.abs(np.einsum("pqxx->pq", dm2) - (ne - 1) * dm1).max()
    )
    checks["dm123_vs_dm1234_reorder"] = float(
        max(np.abs(n2c - dm2).max(), np.abs(n3c - dm3).max())
    )
    bar = 100 * ndet * EPS * max(scale3, 1.0) * n
    for k, v in checks.items():
        assert v < bar, f"{k} = {v:.2e} >= {bar:.2e}"

    li = (
        cistring.gen_linkstr_index(range(n), na),
        cistring.gen_linkstr_index(range(n), nb),
    )
    f3ca = nevpt2._contract4pdm("NEVPTkern_cedf_aedf", eri, ci, n, nelec, li)
    f3ac = nevpt2._contract4pdm("NEVPTkern_aedf_ecdf", eri, ci, n, nelec, li)
    ca = np.einsum("kifj,abcdejki->abcdef", eri, r4)
    ac = np.einsum("ikje,abcdjfik->abcdef", eri, r4)
    fscale = float(max(np.abs(f3ca).max(), np.abs(f3ac).max()))
    checks["f3ca_vs_r4_einsum"] = float(np.abs(f3ca - ca).max())
    checks["f3ac_vs_r4_einsum"] = float(np.abs(f3ac - ac).max())
    fbar = 100 * ndet * EPS * fscale * n**3
    for k in ("f3ca_vs_r4_einsum", "f3ac_vs_r4_einsum"):
        assert checks[k] < fbar, f"{k} = {checks[k]:.2e} >= {fbar:.2e}"

    hdm1 = nevpt2.make_hdm1(e1)
    hdm2 = nevpt2.make_hdm2(e1, e2)
    hdm3 = nevpt2.make_hdm3(e1, e2, e3, hdm1, hdm2)
    dm3_u, checks["dm3_symmetry"] = unique(dm3, DM3_PAIRS, 100 * ndet * EPS * scale3)
    hscale = float(np.abs(hdm3).max())
    hdm3_u, checks["hdm3_symmetry"] = unique(
        hdm3, HDM3_PAIRS, 100 * ndet * EPS * hscale
    )

    payload = {
        "system": d["system"],
        "basis": d["basis"],
        "active_start": d["active_start"],
        "n_active": n,
        "n_elec_active": [na, nb],
        "n_determinants": ndet,
        "alpha_strings": d["alpha_strings"],
        "beta_strings": d["beta_strings"],
        "ci_vector": d["ci_vector"],
        "h": d["h"],
        "eri": d["eri"],
        "e_core": d["e_core"],
        "generator_checks": checks,
        "dm1": q(dm1),
        "dm2": q(dm2),
        "dm3_unique": dm3_u,
        "f3ca_projections": projections(f3ca),
        "f3ac_projections": projections(f3ac),
        "f3ca_max_abs": float(np.abs(f3ca).max()),
        "f3ac_max_abs": float(np.abs(f3ac).max()),
        "hdm1": q(hdm1),
        "hdm2": q(hdm2),
        "hdm3_unique": hdm3_u,
        "provenance": {
            "generator": "scripts/gen_casci_rdm3_refs.py",
            "input": "crates/ferric-ci/examples/casci_rdm_dump.rs",
            "pyscf": pyscf.__version__,
            "numpy": np.__version__,
            "rdms": "fci.rdm.make_dm1234 + reorder_dm1234 on ferric's CI vector",
            "f3ca_f3ac": "mrpt.nevpt2._contract4pdm (NEVPTkern_cedf_aedf, NEVPTkern_aedf_ecdf)",
            "hdm": "mrpt.nevpt2.make_hdm1/2/3 on fci.rdm.make_dm123 (E-product)",
            "rounding": f"values rounded to {DECIMALS} decimal places",
            "storage": "dm3/hdm3 symmetry-unique; f3ca/f3ac full for h2o, projections for all",
        },
    }
    if d["system"] in FULL_F3:
        payload["f3ca"] = q(f3ca)
        payload["f3ac"] = q(f3ac)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"casci_{d['system']}_rdm3.json"
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(
        f"{d['system']}: N_det {ndet} max|dm3| {scale3:.3f} max|f3| {fscale:.3f} "
        + " ".join(f"{k} {v:.1e}" for k, v in checks.items())
        + f" -> {out} ({out.stat().st_size / 1e3:.0f} kB)"
    )
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for a in argv:
        gen(Path(a))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
