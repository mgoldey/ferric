"""PySCF references for the CAS-CI RDM tests (crates/ferric-ci/tests/casci_rdms.rs).

Input: a dump of one ferric CAS-CI run with RDMs, written by

    OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-ci --example casci_rdm_dump -- \\
        <system> <dump.json>

for system in n2_r1.10, h2o. The dump holds ferric's active orbitals
(`c_act`), ferric's active-space integrals (`h` with the inactive-core
potential, `eri`, `e_core`), the alpha/beta strings, the CI vector and ferric's
`dm1`/`dm2`.

What PySCF does (on ferric's integrals only; no PySCF orbitals are made):

1. Reference state: the dense active-space Hamiltonian over every determinant
   (`fci.direct_spin1.pspace(h, eri, n, nelec, np=N_det)`), numpy `eigh`, the
   lowest root. `gap` = second root minus the first (same Ms).
2. `dm1, dm2 = fci.direct_spin1.make_rdm12(ci, n, nelec)` on that root
   (`dm1[p,q] = <E_pq>`, `dm2[p,q,r,s] = sum_st <p_s^+ r_t^+ s_t q_s>`).
3. Anchor: `e_core + sum h dm1 + 1/2 sum eri dm2` equals the dense eigenvalue
   (`pyscf_rdm_energy_anchor_diff`, refused above 1e-10).
4. Construction check: ferric's CI vector, mapped to PySCF's string order with
   `cistring.str2addr`, through `make_rdm12`, against ferric's RDMs from the
   dump (`construction_max_diff_dm1/dm2`). Both codes use ascending creation
   order within a string, so no per-string sign is applied; the overlap of the
   two ground states (`ci_overlap_abs`) confirms it.

Output: crates/ferric-ci/tests/data/casci_rdm/<system>.json with `c_act`
copied from the dump (the test rotates the reference into its own orbitals
with C_ref^T S C_now).

Run (light; seconds):
    uv run --no-sync python scripts/gen_casci_rdm_refs.py <dump.json> [<dump.json> ...]
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

OUT_DIR = (
    Path(__file__).resolve().parent.parent / "crates/ferric-ci/tests/data/casci_rdm"
)
TOL_ANCHOR = 1e-10
MIN_GAP = 1e-4


def gen(dump_path: Path) -> Path:
    import numpy as np
    import pyscf
    from pyscf.fci import cistring, direct_spin1

    d = json.loads(dump_path.read_text())
    n = int(d["n_active"])
    na, nb = (int(x) for x in d["n_elec_active"])
    nelec = (na, nb)
    h = np.array(d["h"]).reshape(n, n)
    eri = np.array(d["eri"]).reshape(n, n, n, n)
    e_core = float(d["e_core"])
    ndet = comb(n, na) * comb(n, nb)

    addr, hdense = direct_spin1.pspace(h, eri, n, nelec, np=ndet)
    assert hdense.shape == (ndet, ndet), hdense.shape
    w, v = np.linalg.eigh(hdense)
    gap = float(w[1] - w[0])
    assert gap > MIN_GAP, f"ground state (near-)degenerate: gap {gap:.2e}"
    na_str = comb(n, na)
    nb_str = comb(n, nb)
    ci = np.zeros(na_str * nb_str)
    ci[np.asarray(addr)] = v[:, 0]
    ci = ci.reshape(na_str, nb_str)
    dm1, dm2 = direct_spin1.make_rdm12(ci, n, nelec)
    e_rdm = e_core + np.einsum("pq,pq", h, dm1) + 0.5 * np.einsum("pqrs,pqrs", eri, dm2)
    e_dense = float(w[0]) + e_core
    anchor = float(e_rdm - e_dense)
    assert abs(anchor) < TOL_ANCHOR, f"PySCF RDM energy anchor {anchor:.2e}"

    # ferric's CI vector in PySCF's string order.
    a_addr = [int(cistring.str2addr(n, na, int(s))) for s in d["alpha_strings"]]
    b_addr = [int(cistring.str2addr(n, nb, int(s))) for s in d["beta_strings"]]
    cf = np.array(d["ci_vector"]).reshape(len(a_addr), len(b_addr))
    ci_f = np.zeros((na_str, nb_str))
    ci_f[np.ix_(a_addr, b_addr)] = cf
    overlap = float(abs(np.sum(ci_f * ci)))
    f1, f2 = direct_spin1.make_rdm12(ci_f, n, nelec)
    c1 = float(np.max(np.abs(f1 - np.array(d["dm1"]).reshape(n, n))))
    c2 = float(np.max(np.abs(f2 - np.array(d["dm2"]).reshape(n, n, n, n))))

    payload = {
        "system": d["system"],
        "basis": d["basis"],
        "active_start": d["active_start"],
        "n_active": n,
        "n_elec_active": [na, nb],
        "n_determinants": ndet,
        "nbas": d["nbas"],
        "c_act": d["c_act"],
        "e_core": e_core,
        "e_total": e_dense,
        "ferric_e_total": d["e_total"],
        "gap": gap,
        "pyscf_rdm_energy_anchor_diff": anchor,
        "ci_overlap_abs": overlap,
        "construction_max_diff_dm1": c1,
        "construction_max_diff_dm2": c2,
        "dm1": dm1.ravel().tolist(),
        "dm2": dm2.ravel().tolist(),
        "provenance": {
            "generator": "scripts/gen_casci_rdm_refs.py",
            "input": "crates/ferric-ci/examples/casci_rdm_dump.rs",
            "pyscf": pyscf.__version__,
            "numpy": np.__version__,
            "reference_state": "dense fci.direct_spin1.pspace over all determinants, numpy eigh",
            "rdms": "fci.direct_spin1.make_rdm12 (reorder=True)",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{d['system']}.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(
        f"{d['system']:9s} N_det {ndet:4d} E_pyscf {e_dense:.12f} E_ferric {d['e_total']:.12f} "
        f"gap {gap:.4f} anchor {anchor:.1e} |<f|p>| {overlap:.12f} "
        f"construction dm1 {c1:.1e} dm2 {c2:.1e}",
        flush=True,
    )
    return out


def main(argv) -> int:
    if not argv:
        raise SystemExit(__doc__)
    for p in argv:
        gen(Path(p))
    print(f"GEN_CASCI_RDM_DONE written={len(argv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
