"""PySCF references for the open-shell (UKS) double hybrids.

Consumer: crates/ferric-mp2/tests/validation_udh_pyscf.rs.
Output:   testdata/reference/validation/udh_pyscf/<system>_<basis>.json

THE DEFINITIONS BUILT HERE (nothing is copied from ferric)

    E = E_UKS[a_x HF + b_x X, a_c C] + c_os E_OS + c_ss E_SS   (+ E_D3(BJ))

  b2plyp           0.53 HF + 0.47 B88, 0.73 LYP        c_os 0.27, c_ss 0.27
                   (Grimme, J. Chem. Phys. 124, 034108 (2006))
  dsd-pbep86       0.72 HF + 0.28 PBE, 0.44 P86        c_os 0.51, c_ss 0.36
                   (Kozuch and Martin, J. Comput. Chem. 34, 2327 (2013), no
                   dispersion; Psi4 "DSD-PBEP86")
  dsd-pbep86-d3bj  0.69 HF + 0.31 PBE, 0.44 P86        c_os 0.52, c_ss 0.22,
                   D3(BJ) s6 0.48, a1 0, s8 0, a2 5.6  (same paper; Psi4
                   "DSD-PBEP86-D3BJ", ORCA "DSD-PBEP86/2013 D3BJ")
  b2plyp-d3bj      the b2plyp SCF and scalings plus the D3(BJ) row of
                   Goerigk and Grimme, PCCP 13, 6670 (2011)

E_OS and E_SS are the opposite-spin (alpha-beta) and same-spin (alpha-alpha +
beta-beta) pair energies of `mp.dfump2.DFUMP2` on the UKS orbitals and orbital
energies. PySCF has no per-spin-pair split for the unrestricted DF-MP2 other
than its e_corr_os / e_corr_ss tags, which are used, and the scaling is
applied here. D3(BJ) comes from the `dftd3` package (simple-dftd3) with the
method name, with the explicit parameters as a second evaluation.

LIKE-FOR-LIKE SETTINGS (what ferric's run_mp2_double_hybrid_arm and
u_mp2_double_hybrid do)

  * basis, geometry: ferric's JSON and Bohr coordinates through common.py;
  * KS grid: (75, 110) unpruned, Becke partition, Becke radii;
  * KS J/K: density_fit over def2-universal-jkfit (J and the HF K);
  * PT2: df.DF over the aux basis named per case, frozen core 0 and 1;
  * state: common.run_open_shell converges from minao, atom and huckel guesses
    to an internally stable UKS state and keeps the lowest; every stable
    guess's energy is stored under `guess_energies`, and a case whose stable
    states differ by more than 1e-8 Ha is refused (a state-selection
    difference would read as an integral error);
  * convergence: conv_tol 1e-12, conv_tol_grad 1e-9.

Run:
    box-lease exec -- env OPENBLAS_NUM_THREADS=1 \
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_udh_pyscf.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_urpa  # noqa: E402

ROW = "udh_pyscf"
JKFIT = "def2-universal-jkfit"
MAIN_GRID = (75, 110)
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
FROZEN = (0, 1)
STATE_TOL = 1e-8

# set -> (libxc string, c_os, c_ss, D3(BJ) method name or None)
SETS = {
    "b2plyp": ("0.53*HF + 0.47*B88, 0.73*LYP", 0.27, 0.27, None),
    "dsd-pbep86": ("0.72*HF + 0.28*PBE, 0.44*P86", 0.51, 0.36, None),
    "dsd-pbep86-d3bj": ("0.69*HF + 0.31*PBE, 0.44*P86", 0.52, 0.22, "dsdpbep86"),
}
# The SCF of b2plyp-d3bj is b2plyp's; only the D3 row is added.
D3_ONLY = {"b2plyp-d3bj": ("b2plyp", "b2plyp")}
D3_EXPLICIT = {
    "dsdpbep86": dict(s6=0.48, s8=0.0, a1=0.0, a2=5.6),
    "b2plyp": dict(s6=0.64, s8=0.9147, a1=0.3065, a2=5.0570),
}

# (system, multiplicity, basis, PT2 aux basis)
CASES = {
    "nh2_cc-pvdz": ("nh2", 2, "cc-pvdz", "cc-pvdz-ri"),
    "ch3_cc-pvdz": ("ch3", 2, "cc-pvdz", "cc-pvdz-ri"),
    "nh2_def2-svp": ("nh2", 2, "def2-svp", "def2-svp-rifit"),
    "ch3_def2-svp": ("ch3", 2, "def2-svp", "def2-svp-rifit"),
}


def check_functional(xc, a_x, w_x, w_c):
    """The libxc string parses to a_x HF, w_x of its X and w_c of its C."""
    from pyscf import dft

    (hyb, _alpha, _omega), funcs = dft.libxc.parse_xc(xc)
    assert abs(hyb - a_x) < 1e-15, (xc, hyb)
    weights = sorted(float(w) for _i, w in funcs)
    assert weights == sorted([w_x, w_c]), (xc, weights)


def make_uks_factory(jkfit):
    def factory(mol):
        from pyscf import dft

        mf = dft.UKS(mol, xc=factory.xc).density_fit(auxbasis=jkfit)
        mf.grids.atom_grid = MAIN_GRID
        mf.grids.prune = None
        mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
        return mf

    factory.xc = None
    return factory


def pt2(mf, mol, aux, frozen):
    from pyscf import df
    from pyscf.mp import dfump2

    pt = dfump2.DFUMP2(mf)
    pt.with_df = df.DF(mol, auxbasis=aux)
    pt.frozen = frozen
    pt.verbose = 0
    pt.kernel()
    return {
        "e_os": float(pt.e_corr_os),
        "e_ss": float(pt.e_corr_ss),
        "e_corr": float(pt.e_corr),
    }


def d3bj(symbols, coords_bohr, method):
    import numpy as np
    from dftd3.interface import DispersionModel, RationalDampingParam

    z = np.array([common.z_of(s) for s in symbols])
    model = DispersionModel(z, np.array(coords_bohr))
    by_name = model.get_dispersion(
        RationalDampingParam(method=method, atm=False), grad=False
    )["energy"]
    p = D3_EXPLICIT[method]
    explicit = model.get_dispersion(
        RationalDampingParam(s9=0.0, **p), grad=False
    )["energy"]
    assert abs(by_name - explicit) < 1e-14, (method, by_name, explicit)
    return float(by_name), p


def gen_case(key):
    import numpy
    import pyscf

    system, mult, basis_name, aux_name = CASES[key]
    t0 = time.perf_counter()
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name, multiplicity=mult)
    common.check_basis_like_for_like(mol, basis_name, symbols)
    jk, jk_cart = common.pyscf_basis(JKFIT, symbols)
    assert not jk_cart, "Cartesian jkfit shells cannot be matched"
    aux = gen_urpa.aux_dict(aux_name, symbols)
    out = {
        "row": "UKS double hybrids",
        "system": system,
        "basis": basis_name,
        "pt2_aux": aux_name,
        "charge": 0,
        "multiplicity": mult,
        "nao": int(mol.nao_nr()),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "sets": {},
    }
    for name, (xc, c_os, c_ss, d3name) in SETS.items():
        a_x = float(xc.split("*HF")[0])
        w_x = float(xc.split("+")[1].split("*")[0])
        w_c = float(xc.split(",")[1].split("*")[0])
        check_functional(xc, a_x, w_x, w_c)
        factory = make_uks_factory(jk)
        factory.xc = xc
        res, mf = common.run_open_shell(
            mol,
            "uks",
            conv_tol=CONV_TOL,
            conv_tol_grad=CONV_TOL_GRAD,
            mf_factory=factory,
            return_mf=True,
            distinct_tol=STATE_TOL,
        )
        if res["multiple_stable_minima"]:
            raise RuntimeError(f"{key}/{name}: several stable UKS states: {res['guess_scan']}")
        entry = {
            "xc": xc,
            "c_os": c_os,
            "c_ss": c_ss,
            "ks": {
                "energy": res["energy"],
                "s_squared": res["s_squared"],
                "homo_alpha": res["homo_alpha"],
                "lumo_alpha": res["lumo_alpha"],
                "homo_beta": res["homo_beta"],
                "lumo_beta": res["lumo_beta"],
            },
            "stability": res["stability"],
            "guess_energies": [
                g["energy"] for g in res["guess_scan"] if g["converged"] and g["internal_stable"]
            ],
        }
        # Reproduction: the same case from the atom guess alone, to the same
        # stable state; the spread of E_KS, E_OS, E_SS bounds PySCF's own
        # run-to-run term (the consumer's bar is built from it).
        res2, mf2 = common.run_open_shell(
            mol,
            "uks",
            conv_tol=CONV_TOL,
            conv_tol_grad=CONV_TOL_GRAD,
            mf_factory=factory,
            return_mf=True,
            guesses=("atom",),
            distinct_tol=STATE_TOL,
        )
        repro = {"e_ks": abs(res["energy"] - res2["energy"])}
        for fc in FROZEN:
            p = pt2(mf, mol, aux, fc)
            p2 = pt2(mf2, mol, aux, fc)
            p["total"] = res["energy"] + c_os * p["e_os"] + c_ss * p["e_ss"]
            entry[f"fc{fc}"] = p
            repro[f"fc{fc}_e_os"] = abs(p["e_os"] - p2["e_os"])
            repro[f"fc{fc}_e_ss"] = abs(p["e_ss"] - p2["e_ss"])
        entry["reproduction"] = {
            "note": "|d| between the best-of-three-guesses and the atom-guess PySCF runs of this case and set",
            **repro,
        }
        if d3name is not None:
            e_d, params = d3bj(symbols, coords, d3name)
            entry["d3bj"] = {"method": d3name, "energy": e_d, "params": params}
            for fc in FROZEN:
                entry[f"fc{fc}"]["total_with_d3bj"] = entry[f"fc{fc}"]["total"] + e_d
        out["sets"][name] = entry
    # B2PLYP-D3(BJ): the b2plyp SCF and PT2 plus its D3(BJ) row.
    for name, (base, d3name) in D3_ONLY.items():
        b = out["sets"][base]
        e_d, params = d3bj(symbols, coords, d3name)
        entry = {
            "xc": b["xc"],
            "c_os": b["c_os"],
            "c_ss": b["c_ss"],
            "same_scf_as": base,
            "d3bj": {"method": d3name, "energy": e_d, "params": params},
        }
        for fc in FROZEN:
            entry[f"fc{fc}"] = {
                "total_with_d3bj": b[f"fc{fc}"]["total"] + e_d,
            }
        out["sets"][name] = entry
    out["provenance"] = common.provenance(
        code="PySCF (dft.UKS density_fit + mp.dfump2.DFUMP2) + dftd3",
        version=pyscf.__version__,
        keywords={
            "ks_jk": f"density_fit(auxbasis={JKFIT}) for J and K",
            "pt2": "mp.dfump2.DFUMP2 on the UKS orbitals, with_df = df.DF(pt2 aux), frozen 0 and 1; e_corr_os / e_corr_ss tags scaled here",
            "state": "common.run_open_shell: minao/atom/huckel guesses, internal stability loop, lowest stable state",
            "numpy": numpy.__version__,
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid={
            "main": {
                "atom_grid": list(MAIN_GRID),
                "prune": None,
                "partition": "Becke (original_becke)",
                "radii_adjust": "becke_atomic_radii_adjust",
            }
        },
        aux={
            "jk": JKFIT,
            "jk_json": str(common.basis_json_path(JKFIT).relative_to(common.ROOT)),
            "jk_sha256": common.sha256_file(common.basis_json_path(JKFIT)),
            "pt2": aux_name,
            "pt2_json": str(common.basis_json_path(aux_name).relative_to(common.ROOT)),
            "pt2_sha256": common.sha256_file(common.basis_json_path(aux_name)),
        },
        frozen_core=list(FROZEN),
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability={"internal_stable": True, "kind": "internal, real (common.run_open_shell)"},
        generator="scripts/validation/gen_udh_pyscf.py",
    )
    path = common.write_reference(ROW, system, basis_name, out)
    print(
        f"{key}: "
        + ", ".join(
            f"{n} {e['fc0']['total']:.10f}" for n, e in out["sets"].items() if "ks" in e
        )
        + f" ({time.perf_counter() - t0:.1f} s) -> {path.relative_to(common.ROOT)}"
    )


def main(argv):
    want = list(argv) or list(CASES)
    unknown = [k for k in want if k not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown} (known: {sorted(CASES)})")
    for k in want:
        gen_case(k)
    print(f"GEN_UDH_PYSCF_DONE written={len(want)}")


if __name__ == "__main__":
    main(sys.argv[1:])
