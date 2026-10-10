#!/usr/bin/env python3
"""Harness validation (cheap: water monomer/dimer, aug-cc-pVDZ, frozen core).

Checks (each prints PASS/FAIL with the actual number):
  V1  split evaluation (one scan) == all-in-one run_mp2_v total, <= 1e-9 Ha,
      for linked and decoupled-sharp arms and several b.
  V2  CP bookkeeping: harness fragments + analyze.interaction == a hand-built
      ghost-atom calculation (independent xyz strings, independent run_mp2_v
      calls), <= 1e-9 Ha.
  V3  omega=None is byte-identical (bitwise float equality) to the published
      linked path (run_mp2_v with omega not passed).
  V4  MUTATION: pairing the attMP2 half of r0=1.00 with the VV10 half damped
      at a different r0 (the Eq.-11 lockstep violation) must make V1 FAIL.
  V5  analyze.py self-test: synthetic data with a known b* is recovered; an
      edge minimum is flagged.
  V6  (informational) NLC grid sensitivity of the VV10 half on the water dimer.

Run:  PYTHONPATH=<shim containing ferric built from this tree> \
      OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2 FERRIC_TERF_TABLE_DIR=<terf-tables> \
      python3 validate.py [--json out.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import harness  # noqa: E402

TOL = 1e-9
RESULTS = {}


def check(name, ok, detail):
    RESULTS[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def main():
    import ferric

    sets = harness.load_sets()["a24"]["systems"]["2"]  # water dimer (A24 #2)
    frs = harness.fragments(sets)
    basis = "adz"
    obs_name, aux_name = harness.BASES[basis]
    obs = ferric.BasisSet.bundled(obs_name)
    aux = ferric.BasisSet.bundled(aux_name)

    def mol_of(atoms):
        return ferric.Molecule.from_xyz_string(harness.xyz_text(atoms), 0, 1)

    arms = [
        {"r0": 1.00, "r0omega": None},
        {"r0": 1.00, "r0omega": 4.0},
        {"r0": 0.90, "r0omega": None},
    ]
    bs = [8.0, 11.0, 14.5]
    nlc = (50, 50)

    # ---- V1: split vs all-in-one on the dimer --------------------------------
    scan = harness.run_fragment(
        ferric, frs["dimer"], basis, arms, bs, nlc, True, df_exact=True
    )
    fc = scan["frozen_core"]
    mol = mol_of(frs["dimer"])
    worst = 0.0
    n_v1 = 0
    for a, arm in enumerate(arms):
        for k, b in enumerate(bs):
            if b == 11.0 and a != 1:
                continue  # keep cost down: b=11 only for the sharp arm
            n_v1 += 1
            w = harness.omega_from(arm["r0"], arm["r0omega"])
            ref = ferric.run_mp2_v(
                mol, obs, aux, r0=arm["r0"], b=b, c=0.0089, omega=w,
                frozen_core=fc,
            )  # fmt: skip
            tot = (
                scan["rhf_energy"]
                + scan["arms"][a]["att_mp2_corr"]
                + scan["arms"][a]["vv10_e_nl"][k]
            )
            worst = max(worst, abs(tot - ref.total_energy))
    check(
        "V1 split==all-in-one",
        worst <= TOL,
        f"max |dE| = {worst:.3e} Ha over {n_v1} combos (tol {TOL:.0e})",
    )

    # ---- V3: omega=None bitwise equal to the published linked path -----------
    ref_pub = ferric.run_mp2_v(
        mol, obs, aux, frozen_core=fc
    )  # r0=1.00, b=11.0, omega unset
    k11 = bs.index(11.0)
    tot_none = (
        scan["rhf_energy"]
        + scan["arms"][0]["att_mp2_corr"]
        + scan["arms"][0]["vv10_e_nl"][k11]
    )
    d_bits = abs(tot_none - ref_pub.total_energy)
    check("V3 omega=None == published (bitwise parts)",
          scan["arms"][0]["att_mp2_corr"] == ref_pub.att_mp2_corr
          and scan["arms"][0]["vv10_e_nl"][k11] == ref_pub.vv10_e_nl
          and scan["rhf_energy"] == ref_pub.rhf_energy,
          f"att {scan['arms'][0]['att_mp2_corr']!r} vs {ref_pub.att_mp2_corr!r}; "
          f"nl {scan['arms'][0]['vv10_e_nl'][k11]!r} vs {ref_pub.vv10_e_nl!r}; |dtotal|={d_bits:.1e}")  # fmt: skip

    # ---- V4: mutation (lockstep violation) must be caught by V1 --------------
    # attMP2 from arm0 (r0=1.00) + VV10 damped with arm2's r0=0.90: the Eq.-11
    # "same r0 in both halves" error. Compare against run_mp2_v(r0=1.00).
    mut_worst = 0.0
    for k, b in enumerate(bs):
        if b != 11.0:
            continue
        ref = ferric.run_mp2_v(mol, obs, aux, r0=1.00, b=b, c=0.0089, frozen_core=fc)
        tot_mut = (
            scan["rhf_energy"]
            + scan["arms"][0]["att_mp2_corr"]
            + scan["arms"][2]["vv10_e_nl"][k]
        )
        mut_worst = max(mut_worst, abs(tot_mut - ref.total_energy))
    check("V4 mutation caught", mut_worst > 1e-6,
          f"mutated (r0_damp=0.90 vs r0_MP2=1.00) differs from run_mp2_v by {mut_worst:.3e} Ha "
          f"(must exceed 1e-6; the V1 tolerance is {TOL:.0e})")  # fmt: skip

    # ---- V2: CP bookkeeping vs hand-built ghost calculation ------------------
    # Harness side: run all five fragments through run_fragment, then
    # analyze.interaction on a db assembled exactly like harness.main does.
    db = {"meta": {"set": "a24", "basis": basis, "arms": arms, "bs": bs, "nlc": list(nlc),
                   "coulomb_mp2": True}, "fragments": {}}  # fmt: skip
    for tag in ("dimer", "mA_cp", "mB_cp", "mA", "mB"):
        db["fragments"][f"a24-02|{tag}"] = harness.run_fragment(
            ferric, frs[tag], basis, arms, bs, nlc, True, df_exact=True)  # fmt: skip
    inter_cp = analyze.interaction(db, "cp")[2]
    inter_ncp = analyze.interaction(db, "ncp")[2]
    # Hand-built side: literal '@' xyz strings typed from the A24 #2 water dimer
    # (A = first 3 atoms, B = last 3), no use of harness.ghost/fragments.
    A = sets["frag_a"]
    B = sets["frag_b"]

    def lines(atoms, ghost):
        return [
            f"{'@' if ghost else ''}{s} {x:.8f} {y:.8f} {z:.8f}" for s, x, y, z in atoms
        ]

    def mk(ls):
        return ferric.Molecule.from_xyz_string(
            f"{len(ls)}\nhand\n" + "\n".join(ls) + "\n", 0, 1
        )

    def e(m, fcnt, r0, b, w=None):
        return ferric.run_mp2_v(
            m, obs, aux, r0=r0, b=b, c=0.0089, omega=w, frozen_core=fcnt
        ).total_energy

    worst_cp = 0.0
    for a, arm in enumerate(arms[:2]):
        for k, b in enumerate(bs):
            if b != 11.0:
                continue
            w = harness.omega_from(arm["r0"], arm["r0omega"])
            ed = e(mk(lines(A, False) + lines(B, False)), 2, arm["r0"], b, w)
            ea = e(mk(lines(A, False) + lines(B, True)), 1, arm["r0"], b, w)
            eb = e(mk(lines(A, True) + lines(B, False)), 1, arm["r0"], b, w)
            hand = (ed - ea - eb) * analyze.K
            worst_cp = max(worst_cp, abs(hand - inter_cp["mp2v"][a][k]))
    if True:
        check("V2 CP bookkeeping == hand-built ghosts", worst_cp <= TOL * analyze.K,
              f"max |dE_int| = {worst_cp:.3e} kcal/mol (tol {TOL*analyze.K:.1e}); "
              f"water dimer MP2-V(published) CP={inter_cp['mp2v'][0][k11]:+.4f}, "
              f"nonCP={inter_ncp['mp2v'][0][k11]:+.4f} kcal/mol, ref {sets['ref']:+.3f}")  # fmt: skip

    # ---- V5: analyze.py self-test -------------------------------------------
    rng = np.random.default_rng(7)
    bgrid = np.arange(6.0, 16.01, 0.5)
    arms_s = [{"r0": 1.0, "r0omega": None}]
    ids = list(range(1, 9))
    refs_s = analyze.refs("a24")
    b_true = 10.3
    frs_s = {}
    for i in ids:
        slope = rng.uniform(0.2, 0.8)
        sign = rng.choice([-1, 1])
        # error(b) = slope*(b-b_true) + small per-system offset  -> b* near b_true
        off = rng.normal(0, 0.02)
        e_int = refs_s[i] + sign * slope * (bgrid - b_true) + off
        # distribute onto fragments: dimer carries it all, monomers zero.
        for tag, val in (
            ("dimer", e_int / analyze.K),
            ("mA_cp", 0 * e_int),
            ("mB_cp", 0 * e_int),
        ):
            frs_s[f"a24-{i:02d}|{tag}"] = {
                "rhf_energy": 0.0, "mp2_coulomb_corr": None,
                "arms": [{"att_mp2_corr": 0.0, "vv10_e_nl": (val).tolist(), "e_os": 0, "e_ss": 0}],
            }  # fmt: skip
    syn = {"meta": {"set": "a24", "basis": "x", "arms": arms_s, "bs": bgrid.tolist(),
                    "nlc": [1, 1], "coulomb_mp2": False}, "fragments": frs_s}  # fmt: skip
    res = analyze.analyze(syn, [], "cp", analyze.A24_CLASS)
    bfit = res["arms"][0]["fit"]["b"]
    check(
        "V5a fit recovers known b*",
        abs(bfit - b_true) < 0.6,
        f"b*={bfit:.2f} (true {b_true}, grid 0.5)",
    )
    # edge: true minimum outside the grid
    syn2 = json.loads(json.dumps(syn))
    syn2["meta"]["bs"] = np.arange(12.0, 16.01, 0.5).tolist()
    for k, f in syn2["fragments"].items():
        f["arms"][0]["vv10_e_nl"] = f["arms"][0]["vv10_e_nl"][
            -len(syn2["meta"]["bs"]) :
        ]
    res2 = analyze.analyze(syn2, [], "cp", analyze.A24_CLASS)
    check("V5b edge minimum flagged", res2["arms"][0]["fit"]["edge"], f"edge={res2['arms'][0]['fit']['edge']} b*={res2['arms'][0]['fit']['b']}")  # fmt: skip

    # ---- V6: NLC grid sensitivity (informational) ---------------------------
    out6 = {}
    for g in ((30, 50), (50, 50), (75, 110)):
        r = harness.run_fragment(
            ferric, frs["dimer"], basis, arms[:2], [11.0], g, False, df_exact=True
        )
        out6[str(g)] = [
            r["arms"][0]["vv10_e_nl"][0],
            r["arms"][1]["vv10_e_nl"][0],
            r["n_nlc_points"],
        ]
    print(
        "V6 NLC grid sensitivity (water dimer aDZ, b=11): grid -> [E_nl linked, E_nl sharp(4), npts]"
    )
    for g, v in out6.items():
        print(f"    {g}: {v[0]:.8f} {v[1]:.8f} {int(v[2])}")
    RESULTS["V6"] = out6

    ok = all(v["pass"] for k, v in RESULTS.items() if k != "V6")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(
            json.dumps(RESULTS, indent=1)
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
