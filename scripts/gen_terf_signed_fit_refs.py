#!/usr/bin/env python3
"""Reference numbers for the terf long-range fit (Ne2, counterpoise).

The terf kernel is not positive definite, so its auxiliary metric (P|terf|Q)
is indefinite. This script evaluates the long-range interaction
K_lr[ia,jb] = (ia|terf|jb) on Ne2 by three independent fitted constructions
and by the positive-subspace construction that drops the negative modes:

  signed_pinv   J_t^T V_t^+ J_t, pseudo-inverse over |lambda| > 1e-10 (both signs)
  sandwich      (V_C^-1 J_C)^T V_t (V_C^-1 J_C): Coulomb-fitted densities, terf between
  c_minus_sr    J_C^T V_C^-1 J_C - J_s^T V_s^-1 J_s: Coulomb RI minus terfc RI
  drop          J_t^T V_t,+^+ J_t, eigenvalues below 1e-10 dropped (negative ones too)

with J_x = (P|x|ia) and V_x = (P|x|Q). For each it records dMP2[lr] =
-2 sum K^2 / (D_ia + D_jb) (= 2 E_OS[lr]) and the closed-shell dRPA energy
by the plasmon formula 1/2 (sum Omega - sum D - 2 tr K), Omega^2 the
eigenvalues of D^1/2 (D + 4K) D^1/2 (no frequency quadrature), for the dimer
and the ghost monomer, so the counterpoise interaction is dimer - 2 monomer.
c_minus_sr is what ferric's rs-mp2-rpa computes for the terf attenuator;
crates/ferric-rpa/tests/terf_signed_fit.rs rebuilds all four in Rust on its
own SCF and asserts against them there. This file is the record quoted in
the report and the book: Ne2, R = 3.6 A, aug-cc-pVDZ / aug-cc-pVDZ-RIFIT,
r0 = 1.35 A, r0*omega in {1/sqrt2, 2, 4}; c_minus_sr is repeated with
aug-cc-pVTZ-RIFIT at r0*omega = 4 to show the fit is converged in the aux
basis.

Run (single-threaded, about a minute; needs FERRIC_TERF_TABLE_DIR):
  OPENBLAS_NUM_THREADS=1 python scripts/gen_terf_signed_fit_refs.py \
      testdata/reference/ne2_terf_signed_fit.json
"""

import json
import os
import sys
import tempfile

import numpy as np

import ferric

A2B = 1.8897261246257702
R_ANG = 3.6
R0_ANG = 1.35
CS = [1.0 / np.sqrt(2.0), 2.0, 4.0]
THR = 1e-10


def molecule(ghost):
    second = "@Ne" if ghost else "Ne"
    with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as f:
        f.write(f"2\nNe2\nNe 0 0 0\n{second} 0 0 {R_ANG}\n")
        path = f.name
    try:
        return ferric.Molecule.from_xyz(path)
    finally:
        os.unlink(path)


def pinv(v, signed):
    w, u = np.linalg.eigh(v)
    keep = (np.abs(w) > THR) if signed else (w > THR)
    return (u[:, keep] / w[keep]) @ u[:, keep].T, int((w < -THR).sum())


def dmp2(k, d):
    return float(-2.0 * np.sum(k * k / (d[:, None] + d[None, :])))


def plasmon(k, d):
    sd = np.sqrt(d)
    w = np.linalg.eigvalsh(sd[:, None] * (np.diag(d) + 4.0 * k) * sd[None, :])
    if w.min() <= 0.0:
        raise RuntimeError(f"unstable dRPA: min eigenvalue {w.min()}")
    return float(0.5 * (np.sqrt(w).sum() - d.sum() - 2.0 * np.trace(k)))


def system(ghost, aux_names):
    mol = molecule(ghost)
    obs = ferric.BasisSet.bundled("aug-cc-pvdz")
    hf = ferric.run_rhf(mol, obs)
    nocc = 5 if ghost else 10
    c = np.asarray(hf.mo_coefficients())
    e = np.asarray(hf.orbital_energies())
    co = np.ascontiguousarray(c[:, :nocc])
    cv = np.ascontiguousarray(c[:, nocc:])
    d = (e[nocc:][None, :] - e[:nocc][:, None]).ravel()
    out = {"hf": float(hf.energy)}
    r0 = R0_ANG * A2B
    for aux_name in aux_names:
        aux = ferric.BasisSet.bundled(aux_name)

        def j3(**kw):
            m = ferric.compute_eri3_mo(mol, obs, aux, co, cv, **kw)
            return np.asarray(m).reshape(-1, d.size)

        def v2(**kw):
            return np.asarray(ferric.compute_metric_2c(mol, obs, aux, **kw))

        jc, vc = j3(), v2()
        vci_jc = np.linalg.solve(vc, jc)
        kc = jc.T @ vci_jc
        block = {"coulomb": {"dmp2": dmp2(kc, d), "drpa": plasmon(kc, d)}}
        for cval in CS:
            lr = dict(omega=cval / r0, r0=r0)
            jt, vt = j3(operator="terf", **lr), v2(operator="terf", **lr)
            js, vs = j3(operator="terfc", **lr), v2(operator="terfc", **lr)
            p_signed, n_neg = pinv(vt, True)
            p_drop, _ = pinv(vt, False)
            ks = {
                "signed_pinv": jt.T @ p_signed @ jt,
                "sandwich": vci_jc.T @ vt @ vci_jc,
                "c_minus_sr": kc - js.T @ np.linalg.solve(vs, js),
                "drop": jt.T @ p_drop @ jt,
            }
            entry = {"n_neg_metric": n_neg, "naux": int(vt.shape[0])}
            for name, k in ks.items():
                entry[name] = {"dmp2": dmp2(k, d), "drpa": plasmon(k, d)}
            block[f"{cval:.8f}"] = entry
        out[aux_name] = block
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "ne2_terf_signed_fit.json"
    auxs = ["aug-cc-pvdz-rifit", "aug-cc-pvtz-rifit"]
    dim = system(False, auxs)
    mon = system(True, auxs)
    res = {
        "_doc": __doc__.strip().splitlines()[0],
        "generator": "scripts/gen_terf_signed_fit_refs.py",
        "ferric_build": ferric.__build__,
        "system": {"R_angstrom": R_ANG, "r0_angstrom": R0_ANG, "basis": "aug-cc-pvdz"},
        "dimer": dim,
        "monomer_ghost": mon,
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    # Summary: CP interaction of dRPA - dMP2 (uHa) per construction.
    for aux in auxs:
        for cval in CS:
            key = f"{cval:.8f}"
            row = []
            for name in ("signed_pinv", "sandwich", "c_minus_sr", "drop"):
                dd = dim[aux][key][name]
                mm = mon[aux][key][name]
                e_int = (dd["drpa"] - dd["dmp2"]) - 2.0 * (mm["drpa"] - mm["dmp2"])
                row.append(f"{name} {e_int * 1e6:9.3f}")
            dm = dim[aux][key]["c_minus_sr"]["dmp2"] * 1e6
            print(
                f"{aux} r0*w={cval:.4f}: E_int[dRPA-dMP2](lr) uHa: "
                + "  ".join(row)
                + f" | dimer dMP2[lr] c_minus_sr {dm:.2f}"
            )


if __name__ == "__main__":
    main()
