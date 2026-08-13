#!/usr/bin/env python3
"""Ne2 seam-artifact test: does range-separated correlation relax Dutoi's
Coulomb-curvature constraint?

Background (Dutoi & Head-Gordon, JPCA 112, 2110 (2008)): with the complement
DISCARDED (attenuated MP2 alone), a curvature-violating short-range operator
produces an unphysical second minimum in Ne2 near the truncation radius, and
the safe family is r0*omega <= 2.07 with optimum 1/sqrt(2). The erf/erfc
splitter violates Coulomb curvature at EVERY finite omega (their p. 2111) —
which makes it the right probe here, since ferric's terf tables hardcode the
curvature-optimal link and cannot violate it by construction.

Hypotheses (pre-registered):
  - relaxation REAL: B-formulation (delta-lr) SR-MP2+LR-RPA curves with the
    erf splitter remain artifact-free (no second minimum, monotone tail, CP
    interaction energy tracking RI-MP2) even at sharp omega where the
    SR-alone control is grossly curvature-broken.
  - constraint SURVIVES (weakened): B curves develop a seam artifact
    (non-monotone tail / spurious feature near the crossover ~1-2/omega)
    that onsets at some omega — its onset measures the NEW bound.
  - positive control: attenuated RI-MP2 (erfc, complement discarded) at the
    same omegas must visibly distort/lose the dispersion well — if it does
    not, the test has no teeth at this basis and must be redesigned.
Anchors: omega -> 0.42 (production) B curve should be smooth; terf-linked B
(r0 = 3.18 A) is the curvature-preserving baseline; all interaction energies
counterpoise-corrected (ghost '@Ne'), per the de-risk rule.

SERIAL by design: one calculation at a time, OPENBLAS_NUM_THREADS=1 assumed.

Usage: python scripts/ne2_seam_test.py [basis] [auxbasis]
       (defaults aug-cc-pvdz / aug-cc-pvdz-rifit)
"""
import os
import sys
import tempfile

import ferric

R_LIST = [2.6, 2.8, 3.0, 3.1, 3.3, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]  # Angstrom
ERF_OMEGAS = [0.42, 1.0, 2.0, 4.0]  # Angstrom^-1, production -> sharp
TERF_R0 = 3.18  # Angstrom (production aTZ point; omega derived = linked)
ATT_OMEGAS = [0.42, 1.0]  # erfc attenuated-MP2 controls (complement discarded)

HA_TO_UHA = 1e6


def make_mol(r_ang, ghost_second):
    sym2 = "@Ne" if ghost_second else "Ne"
    xyz = f"2\nNe2 R={r_ang}\nNe 0.0 0.0 0.0\n{sym2} 0.0 0.0 {r_ang}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False,
                                     dir=os.environ.get("TMPDIR", "/tmp")) as f:
        f.write(xyz)
        path = f.name
    mol = ferric.Molecule.from_xyz(path)
    os.unlink(path)
    return mol


def energies(mol, obs, aux):
    """Total energies for every method variant on one geometry."""
    out = {}
    out["rimp2"] = ferric.run_rimp2(mol, basis_set=obs, auxbasis=aux).total_energy
    for w in ERF_OMEGAS:
        r = ferric.run_rs_mp2_rpa(mol, basis_set=obs, auxbasis=aux,
                                  omega=w, formulation="delta-lr", attenuator="erf")
        out[f"B_erf_w{w}"] = r.total_energy
    # terf-linked baseline is optional: the standalone table engine needs
    # regenerated .bin tables (terf-tables/ holds only generators right now).
    try:
        r = ferric.run_rs_mp2_rpa(mol, basis_set=obs, auxbasis=aux,
                                  formulation="delta-lr", attenuator="terf", r0=TERF_R0)
        out[f"B_terf_r{TERF_R0}"] = r.total_energy
    except RuntimeError as e:
        if "terf" not in str(e):
            raise
        # skip silently after first warning
        if not getattr(energies, "_terf_warned", False):
            print(f"# NOTE: terf baseline skipped ({e})", flush=True)
            energies._terf_warned = True
    for w in ATT_OMEGAS:
        r = ferric.run_attenuated_rimp2(mol, basis_set=obs, auxbasis=aux, omega=w)
        out[f"attMP2_erfc_w{w}"] = r.total_energy
    return out


def main():
    obs_name = sys.argv[1] if len(sys.argv) > 1 else "aug-cc-pvdz"
    aux_name = sys.argv[2] if len(sys.argv) > 2 else "aug-cc-pvdz-rifit"
    obs = ferric.BasisSet.bundled(obs_name)
    aux = ferric.BasisSet.bundled(aux_name)
    print(f"# Ne2 seam test  {obs_name}/{aux_name}  CP-corrected (ghost @Ne)")
    print(f"# R grid (A): {R_LIST}")

    methods = None
    rows = {}
    for r_ang in R_LIST:
        dim = energies(make_mol(r_ang, ghost_second=False), obs, aux)
        mono = energies(make_mol(r_ang, ghost_second=True), obs, aux)
        if methods is None:
            methods = list(dim.keys())
            print("# E_int(R) = E_dimer - 2*E_monomer(ghost partner), microHartree\n")
            print(f"{'R(A)':>6} " + " ".join(f"{m:>16}" for m in methods))
        eint = {m: (dim[m] - 2.0 * mono[m]) * HA_TO_UHA for m in methods}
        rows[r_ang] = eint
        print(f"{r_ang:>6.2f} " + " ".join(f"{eint[m]:>16.2f}" for m in methods), flush=True)

    # Artifact detector: after the global minimum, E_int should rise
    # monotonically to ~0. Report any method with a second descent.
    print("\n# Artifact scan (post-minimum second descent = seam artifact):")
    for m in methods:
        vals = [rows[r][m] for r in R_LIST]
        i_min = vals.index(min(vals))
        descents = [
            (R_LIST[k], vals[k + 1] - vals[k])
            for k in range(i_min, len(vals) - 1)
            if vals[k + 1] < vals[k] - 0.5  # >0.5 uHa second descent
        ]
        tail = vals[-1]
        flag = f"SECOND DESCENT at R>{descents[0][0]}" if descents else "clean"
        print(f"  {m:>16}: min {min(vals):8.2f} uHa at R={R_LIST[i_min]:.2f}; tail(8A) {tail:7.2f}; {flag}")


if __name__ == "__main__":
    main()
