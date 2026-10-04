#!/usr/bin/env python3
"""Issue #280 step 1, part (c): a genuinely INDEPENDENT re-run of the
generator's QP stage, compared against the stored eps_qp, and the lowest-5
Omega from the stored kernel with each of the two QP vectors.
"""

import json
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts/validation"))
import common
import gen_bse
import gen_gw

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from measure_qp_window import decode_kernel, omegas

HA_EV = 27.211386245988
WANT = [
    ("h2o", "cc-pvdz"),
    ("nh3", "cc-pvdz"),
    ("ch2o", "cc-pvdz"),
    ("h2o", "aug-cc-pvdz"),
]


def qp_only(system, basis_name):
    xyz_rel, _ = gen_bse.SYSTEMS[system]
    xyz = common.MOL_DIR / xyz_rel
    aux_name = gen_gw.AUX_FOR[basis_name]
    mol, symbols, _c, _ll = gen_gw.mol_and_prov_base(xyz, basis_name)
    aux = gen_gw.aux_dict(aux_name, symbols)
    mf, _stable = gen_gw.rhf_exact(mol)
    nmo = mf.mo_energy.size
    e_hf = np.asarray(mf.mo_energy, float)
    gw = gen_bse.run_gw_all(mf, aux, nmo)
    zn, coef, _fn, _ni = gen_bse.sigma_nodes(gw, mf)
    return np.array(
        [gen_bse.ferric_qp(zn, coef[:, p], float(e_hf[p]))[0] for p in range(nmo)]
    ), e_hf


for system, basis in WANT:
    name = f"{system}_{basis}"
    d = json.loads(
        (_ROOT / "testdata/reference/validation/bse" / f"{name}.json").read_text()
    )
    stored = np.array(d["qp"]["eps_qp"])
    sens = np.array(d["qp"]["sensitivity"])
    fresh, e_hf = qp_only(system, basis)
    dq = np.abs(fresh - stored)
    print("=" * 92)
    print(f"(c) INDEPENDENT RE-RUN  {name}   nocc={d['nocc']} nvir={d['nvir']}")
    print("=" * 92)
    print("  MO  eps_qp(stored)   eps_qp(rerun)    |d| (Ha)    sensitivity")
    for p in range(stored.size):
        flag = "  <-- in HOMO-2..LUMO+2" if p in d["qp"]["window"] else ""
        print(
            f"  {p:2d}  {stored[p]:+14.9f} {fresh[p]:+14.9f}   {dq[p]:9.2e}   {sens[p]:9.1e}{flag}"
        )
    print(
        f"  max |d eps_qp| over all MOs    = {dq.max():.3e} Ha  (MO {int(dq.argmax())})"
    )
    win = d["qp"]["window"]
    print(f"  max |d eps_qp| in HOMO-2..LUMO+2 = {dq[win].max():.3e} Ha")

    k = decode_kernel(d)
    nocc, nvir = int(d["nocc"]), int(d["nvir"])
    om_s = omegas(k, stored, nocc, nvir)
    om_f = omegas(k, fresh, nocc, nvir)
    print("\n  lowest 5 Omega, stored QP (Ha): " + "  ".join(f"{x:.9f}" for x in om_s))
    print("  lowest 5 Omega, rerun  QP (Ha): " + "  ".join(f"{x:.9f}" for x in om_f))
    dd = om_f - om_s
    print("  dOmega (Ha)                   : " + "  ".join(f"{x:+.2e}" for x in dd))
    print(
        f"  max |dOmega| = {np.abs(dd).max():.3e} Ha = {np.abs(dd).max() * HA_EV:.2e} eV\n"
    )
