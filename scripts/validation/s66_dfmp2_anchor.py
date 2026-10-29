"""Exactness anchor for the S66 benchmark (mgoldey/ferric#278, criterion 1).

ferric `run_rimp2` vs PySCF `mp.dfmp2.DFMP2` on the same S66 dimers, the same
basis, the same RI basis, the same SCF J/K treatment and the same frozen core.
The default `--scf rijk` is the configuration the benchmark runs (DF-HF:
RI-J + RI-K on aug-cc-pvtz-jkfit, Matt 2026-10-29); `--scf exact` is the
former four-centre anchor. Per dimer, both codes run on
five systems: AB, A, B, A-in-AB (ghosted B) and B-in-AB (ghosted A). The script
reports E_HF, E_corr and E_total per system, and the interaction energy both
counterpoise-corrected (CP: AB - A_in_AB - B_in_AB) and uncorrected (noCP:
AB - A - B) for each code, in Ha and kcal/mol.

Gates (exit code 1 if any fails; the JSON is written either way):
  |ddE_int| <= --bar-int-kcal (default 1e-6 kcal/mol), CP AND noCP
  |dE_corr| <= --bar-corr-ha  (default 1e-8 Ha) on every system
  smoke only: also |dE_HF| and |dE_total| <= 1e-8 Ha on every system
The ddE_int bar (1.6e-9 Ha) is TIGHTER than the per-system corr bar on purpose:
it relies on error cancellation between the five systems. If a per-system bar
passes and ddE_int fails, that is a finding about the numerical floor of the
comparison, not a reason to loosen the bar.

Settings matched to `ferric.run_rimp2` (exact path, crates/ferric-python/src/lib.rs):
  orbital basis  ferric's bundled BSE JSON, converted shell-by-shell to PySCF
                 format (general contractions kept; PySCF renormalizes each
                 contraction to unit self-overlap exactly as ferric's
                 parse_bse_json does). Spherical d/f/g (BSE "gto_spherical" for
                 l>=2, Cartesian label on s/p is the same function count), so
                 PySCF cart=False.
  RI basis       the same bundled JSON as `--aux` (aug-cc-pvtz-rifit, the aux
                 the S66 driver passes to ferric); placed on ghost atoms too
                 (PySCF falls back from GHOST-X to the X key; nao is asserted).
  SCF  --scf rijk (default; what s66_terfc_benchmark.py runs): DF-HF in
                 both codes. ferric run_rimp2(scf_jk="rijk",
                 jk_auxbasis=--jk-aux) = RI-J + RI-K on that one aux (the
                 shared ferric_scf::scf_jk rule, #595); the result's echoed
                 scf_jk / jk_auxbasis are asserted. PySCF
                 scf.RHF(mol).density_fit(auxbasis=<the SAME bundled JSON,
                 converted like the orbital/RI bases>), on ghost atoms too;
                 its naux is asserted. DFMP2 is then given its OWN df.DF on
                 the RI aux: DFRMP2.__init__ would otherwise reuse the SCF's
                 JK-fit object (mf.with_df) for the MP2 integrals.
       --scf exact: four-centre J/K in both codes. run_rimp2 with no scf_jk
                 (RhfConfig::default(): df_j_aux/df_k_aux None); PySCF plain
                 scf.RHF, no .density_fit().
  SCF convergence ferric density_conv (rms dP) --density-conv, default 1e-9
                 (the Rust validation tests' value). PySCF has no rms-dP gate;
                 conv_tol 1e-12 / conv_tol_grad 1e-9 is the equivalent
                 tightness. NOT the identical criterion -- both are tight
                 enough that the residual is far below the bars.
  lindep         ferric drops S eigenvalues < 1e-6 (rhf::LINDEP_THRESH,
                 overridable by FERRIC_LINDEP_THRESH). PySCF gets the same
                 canonical orthogonalization (scf.addons) when min eig(S) is
                 below it; the number dropped is recorded.
  frozen core    1 per real first-row heavy atom (ghosts freeze nothing), the
                 same count via the same function as s66_terfc_benchmark.py;
                 both codes freeze the lowest-energy orbitals.
  geometry       s66_terfc_benchmark.py's reader, fragment split (S66_FRAGA)
                 and calc builder. PySCF gets Bohr coordinates converted with
                 ferric's factor; NRE and electron count are asserted equal.
NOT matched (cannot be): ferric's Schwarz integral_thresh 1e-12 vs PySCF
direct_scf_tol 1e-13; ferric's Cholesky (P|Q)^-1/2 vs PySCF's Cholesky
j2c with eigh fallback. Both are below the bars on the smoke.

Usage (from the repo root):
  python scripts/validation/s66_dfmp2_anchor.py --dry-run
  OPENBLAS_NUM_THREADS=1 python scripts/validation/s66_dfmp2_anchor.py --smoke
  OPENBLAS_NUM_THREADS=1 python scripts/validation/s66_dfmp2_anchor.py --smoke --scf exact
  scripts/validation/run_slot.sh -- python scripts/validation/s66_dfmp2_anchor.py \
      --so-dir ~/s66-ferric-so --dimers 1,2
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s66_terfc_benchmark as s66  # noqa: E402  (geometry, split, CP calcs)

ROOT = s66.ROOT
HARTREE_TO_KCAL = s66.HARTREE_TO_KCAL
OUT_DIR = ROOT / "scripts/queue/out"

DEFAULT_DIMERS = "1,2"
DENSITY_CONV = 1e-9
PYSCF_CONV_TOL = 1e-12
PYSCF_CONV_TOL_GRAD = 1e-9
BAR_INT_KCAL = 1e-6
BAR_CORR_HA = 1e-8
SMOKE_BAR_HA = 1e-8  # the exactness anchor: H2 / STO-3G
SCF_MODES = ("rijk", "exact")
FERRIC_LINDEP_DEFAULT = 1e-6  # ferric_scf::rhf::LINDEP_THRESH

# Smoke: (H2)2 / STO-3G with cc-pvdz-ri. Fragment A is the first H2 (that
# monomer IS the H2/STO-3G anchor); the dimer exercises ghosts and CP.
SMOKE_BASIS, SMOKE_AUX = "sto-3g", "cc-pvdz-ri"
SMOKE_ATOMS = [
    ("H", 0.0, 0.0, 0.0),
    ("H", 0.0, 0.0, 0.74),
    ("H", 3.0, 0.0, 0.0),
    ("H", 3.0, 0.0, 0.74),
]
SMOKE_A_SIZE = 2

# ferric pilot wall times (s), RI-MP2 only, aTZ, from #474 (s66_atz.jsonl).
PILOT = s66.PILOT_JSONL


# ── systems ────────────────────────────────────────────────────────────────


def systems(a) -> list[dict]:
    if a.smoke:
        return [{"label": "smoke_(H2)2", "atoms": SMOKE_ATOMS, "a_size": SMOKE_A_SIZE}]
    files, fraga = s66.s66_files(), s66.s66_fraga()
    return [
        {
            "label": files[k].name,
            "dimer": k,
            "atoms": s66.read_atoms(files[k]),
            "a_size": fraga[k],
        }
        for k in s66.parse_dimers(a.dimers)
    ]


def calc_plan(sysd: dict, basis: str, aux: str) -> list[dict]:
    out = []
    for calc in s66.CALCS:
        sub, real = s66.calc_atoms(sysd["atoms"], sysd["a_size"], calc)
        _, nfc = s66.build_calc(sysd["atoms"], sysd["a_size"], calc)
        out.append(
            {
                "calc": calc,
                "natoms_real": sum(real),
                "natoms_ghost": len(real) - sum(real),
                "nbf": sum(s66.nbf_per_element(basis)[s] for s, *_ in sub),
                "naux": sum(s66.nbf_per_element(aux)[s] for s, *_ in sub),
                "frozen_core": nfc,
            }
        )
    return out


# ── PySCF side ─────────────────────────────────────────────────────────────


def pyscf_basis(name: str, sym: str) -> list:
    """ferric's bundled BSE JSON for one element, in PySCF's internal format."""
    z = s66._SYM[sym]
    data = json.loads((s66.BUNDLED / f"{name}.json").read_text())["elements"][z]
    out = []
    for sh in data["electron_shells"]:
        exps = [float(e) for e in sh["exponents"]]
        cols = [[float(c) for c in col] for col in sh["coefficients"]]
        if len(sh["angular_momentum"]) == 1:
            ll = sh["angular_momentum"][0]
            out.append([ll, *[[e, *(c[i] for c in cols)] for i, e in enumerate(exps)]])
        else:  # SP-type: column k belongs to angular_momentum[k]
            for ll, col in zip(sh["angular_momentum"], cols):
                out.append([ll, *[[e, col[i]] for i, e in enumerate(exps)]])
    return out


def run_pyscf(sub, real, nfc: int, a) -> dict:
    from pyscf import df, gto, scf
    from pyscf.mp import dfmp2
    from pyscf.scf import addons

    b = s66.ANGSTROM_TO_BOHR
    atom = [
        (s if r else f"ghost-{s}", (x * b, y * b, z * b))
        for (s, x, y, z), r in zip(sub, real)
    ]
    elems = sorted({s for s, *_ in sub})
    mol = gto.M(
        atom=atom,
        unit="Bohr",
        basis={s: pyscf_basis(a.basis, s) for s in elems},
        cart=False,
        verbose=0,
        max_memory=a.pyscf_max_memory,
    )
    want_nbf = sum(s66.nbf_per_element(a.basis)[s] for s, *_ in sub)
    assert mol.nao_nr() == want_nbf, (mol.nao_nr(), want_nbf)
    t0 = time.perf_counter()
    mf = scf.RHF(mol)
    if a.scf == "rijk":
        mf = mf.density_fit(auxbasis={s: pyscf_basis(a.jk_aux, s) for s in elems})
        mf.with_df.max_memory = a.pyscf_max_memory
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = a.conv_tol, a.conv_tol_grad, 200
    import numpy as np

    s_eig = np.linalg.eigvalsh(mol.intor("int1e_ovlp"))
    lindep = float(os.environ.get("FERRIC_LINDEP_THRESH", FERRIC_LINDEP_DEFAULT))
    n_drop = int((s_eig < lindep).sum())
    if n_drop:
        mf._eigh = addons._eigh_with_canonical_orth(lindep)
    e_hf = mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge")
    t_scf = time.perf_counter() - t0
    naux_jk = None
    if a.scf == "rijk":
        naux_jk = mf.with_df.auxmol.nao_nr()
        want = sum(s66.nbf_per_element(a.jk_aux)[s] for s, *_ in sub)
        assert naux_jk == want, (naux_jk, want)
    ri_df = df.DF(mol, auxbasis={s: pyscf_basis(a.aux, s) for s in elems})
    ri_df.max_memory = a.pyscf_max_memory
    if a.scf == "exact":
        # DFRMP2.__init__ takes mf.with_df when present; otherwise it builds a
        # default even-tempered aux, which fails on ghost atoms (KeyError
        # GHOSTH). Attached AFTER the SCF converged, so the RHF stays exact-JK.
        mf.with_df = ri_df
    pt = dfmp2.DFMP2(mf, frozen=nfc)
    # Under rijk, __init__ just took the SCF's JK-fit object: the MP2 must use
    # the RI aux instead, exactly as ferric's (separate) `auxbasis`.
    pt.with_df = ri_df
    want_naux = sum(s66.nbf_per_element(a.aux)[s] for s, *_ in sub)
    e_corr = pt.kernel()[0]
    naux = pt.with_df.auxmol.nao_nr()
    assert naux == want_naux, (naux, want_naux)
    return {
        "e_hf": float(e_hf),
        "e_corr": float(e_corr),
        "e_total": float(e_hf + e_corr),
        "nre": float(mol.energy_nuc()),
        "nelec": int(sum(mol.nelec)),
        "nbf": mol.nao_nr(),
        "naux": naux,
        "naux_jk": naux_jk,
        "min_s_eig": float(s_eig[0]),
        "lindep_dropped": n_drop,
        "scf_cycles": int(getattr(mf, "cycles", -1)),
        "wall_scf_s": t_scf,
        "wall_s": time.perf_counter() - t0,
    }


# ── ferric side ────────────────────────────────────────────────────────────


def run_ferric(ferric, xyz: str, nfc: int, a) -> dict:
    mol = ferric.Molecule.from_xyz_string(xyz)
    kw = {"frozen_core": nfc}
    if a.mem_gb is not None:
        kw["memory_budget_gb"] = a.mem_gb
    if a.ferric_density_conv is not None:
        kw["density_conv"] = a.ferric_density_conv
    if a.scf == "rijk":
        kw["scf_jk"], kw["jk_auxbasis"] = "rijk", a.jk_aux
    t0 = time.perf_counter()
    r = ferric.run_rimp2(
        mol, ferric.BasisSet.bundled(a.basis), ferric.BasisSet.bundled(a.aux), **kw
    )
    echo = (getattr(r, "scf_jk", "exact"), getattr(r, "jk_auxbasis", None))
    want = (a.scf, a.jk_aux if a.scf == "rijk" else None)
    if echo != want:
        sys.exit(f"ferric ran SCF J/K {echo}, asked for {want}: refusing the row")
    return {
        "scf_jk": echo[0],
        "jk_auxbasis": echo[1],
        "e_hf": r.rhf_energy,
        "e_corr": r.mp2_corr,
        "e_total": r.total_energy,
        "nre": mol.nuclear_repulsion(),
        "nelec": mol.nelec(),
        "reference": r.reference,
        "wall_s": time.perf_counter() - t0,
    }


def ferric_density_conv(ferric, a) -> float | None:
    """The density_conv to pass, or None for the library default (1e-6).
    Mainline builds without the kwarg are allowed for the smoke only."""
    sig = getattr(ferric.run_rimp2, "__text_signature__", "") or ""
    if "density_conv" in sig:
        return a.density_conv
    if a.smoke or a.allow_default_scf:
        print(
            "NOTE: this ferric build's run_rimp2 has no density_conv; using the "
            "library default rms dP 1e-6",
            flush=True,
        )
        return None
    sys.exit(
        "this ferric build's run_rimp2 has no density_conv (needs a45d5d555 / "
        "val/s66-all66-278); point --so-dir at that build or pass --allow-default-scf"
    )


def require_scf_jk(ferric, a) -> None:
    """`--scf rijk` needs run_rimp2(scf_jk=, jk_auxbasis=) (#595). A build
    without it is refused, never silently run as exact."""
    sig = getattr(ferric.run_rimp2, "__text_signature__", "") or ""
    if a.scf == "rijk" and not ("scf_jk" in sig and "jk_auxbasis" in sig):
        sys.exit(
            f"this ferric build ({ferric.__file__}) has no run_rimp2 scf_jk/jk_auxbasis "
            "(needs feat/mp2-scf-rijk-rijcosx, #595); point --so-dir at such a build, "
            "or pass --scf exact for the four-centre anchor"
        )


# ── report ─────────────────────────────────────────────────────────────────


def interaction(rows: dict, code: str, key: str) -> dict:
    e = {c: rows[c][code][key] for c in s66.CALCS}
    return {
        "cp": e["AB"] - e["A_in_AB"] - e["B_in_AB"],
        "nocp": e["AB"] - e["A"] - e["B"],
    }


def check_system(sysd: dict, rows: dict, a) -> tuple[dict, list[str]]:
    fails = []
    bar_c = SMOKE_BAR_HA if a.smoke else a.bar_corr_ha
    for c in s66.CALCS:
        f, p = rows[c]["ferric"], rows[c]["pyscf"]
        d = {k: f[k] - p[k] for k in ("e_hf", "e_corr", "e_total")}
        rows[c]["diff"] = d
        print(
            f"  {c:8s} nfc={rows[c]['frozen_core']}  "
            f"E_HF {f['e_hf']:.12f} / {p['e_hf']:.12f} (d {d['e_hf']:+.2e})  "
            f"E_corr {f['e_corr']:.12f} / {p['e_corr']:.12f} (d {d['e_corr']:+.2e})  "
            f"E_tot d {d['e_total']:+.2e}  [ferric / pyscf, Ha]"
        )
        if abs(f["nre"] - p["nre"]) > 1e-10 * max(1.0, abs(p["nre"])):
            fails.append(f"{c}: NRE ferric {f['nre']} != pyscf {p['nre']}")
        if f["nelec"] != p["nelec"]:
            fails.append(f"{c}: nelec ferric {f['nelec']} != pyscf {p['nelec']}")
        if abs(d["e_corr"]) > bar_c:
            fails.append(f"{c}: |dE_corr| {abs(d['e_corr']):.2e} > {bar_c:.0e} Ha")
        if a.smoke:
            for k in ("e_hf", "e_total"):
                if abs(d[k]) > SMOKE_BAR_HA:
                    fails.append(
                        f"{c}: |d{k}| {abs(d[k]):.2e} > {SMOKE_BAR_HA:.0e} Ha (smoke)"
                    )
    out = {}
    for key in ("e_hf", "e_corr", "e_total"):
        fe, pe = interaction(rows, "ferric", key), interaction(rows, "pyscf", key)
        out[key] = {
            cp: {
                "ferric_ha": fe[cp],
                "pyscf_ha": pe[cp],
                "diff_ha": fe[cp] - pe[cp],
                "ferric_kcal": fe[cp] * HARTREE_TO_KCAL,
                "pyscf_kcal": pe[cp] * HARTREE_TO_KCAL,
                "diff_kcal": (fe[cp] - pe[cp]) * HARTREE_TO_KCAL,
            }
            for cp in ("cp", "nocp")
        }
    for cp in ("cp", "nocp"):
        t = out["e_total"][cp]
        print(
            f"  dE_int[{cp:4s}] ferric {t['ferric_kcal']:+.9f}  pyscf {t['pyscf_kcal']:+.9f} "
            f"kcal/mol  ddE {t['diff_kcal']:+.2e} kcal/mol ({t['diff_ha']:+.2e} Ha)"
        )
        if abs(t["diff_kcal"]) > a.bar_int_kcal:
            fails.append(
                f"dE_int[{cp}]: |ddE| {abs(t['diff_kcal']):.2e} > {a.bar_int_kcal:.0e} kcal/mol"
            )
    return out, fails


def write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)


# ── dry run ────────────────────────────────────────────────────────────────


def pilot_walls() -> dict:
    """{(dimer, calc): wall_s} for ferric RI-MP2/aTZ from the #474 pilot."""
    out = {}
    if PILOT.exists():
        for ln in PILOT.read_text().splitlines():
            r = json.loads(ln)
            if r.get("method") == "mp2" and r.get("basis") == s66.BASIS:
                out[(r["dimer"], r["calc"])] = r["wall_s"]
    return out


def cmd_dry_run(a, settings: dict) -> None:
    print("PLAN (dry run: no ferric, no PySCF, no SCF)")
    for k, v in settings.items():
        print(f"  {k:22s} {v}")
    pilot = pilot_walls()
    ref_nbf = None
    if (1, "AB") in pilot and not a.smoke:
        ref_nbf = calc_plan(
            systems(argparse.Namespace(smoke=False, dimers="1"))[0], a.basis, a.aux
        )
        ref_nbf = {r["calc"]: r["nbf"] for r in ref_nbf}
    total = 0.0
    for sysd in systems(a):
        print(f"\n  {sysd['label']}  a_size={sysd['a_size']}")
        for r in calc_plan(sysd, a.basis, a.aux):
            est = ""
            if ref_nbf and a.basis == s66.BASIS:
                key = (sysd.get("dimer"), r["calc"])
                if key in pilot:
                    w, how = pilot[key], "pilot"
                else:  # N^3.5 from dimer 1's same calc (ESTIMATE)
                    w = pilot[(1, r["calc"])] * (r["nbf"] / ref_nbf[r["calc"]]) ** 3.5
                    how = "N^3.5 extrap."
                total += w
                est = f"  ferric ~{w:6.0f} s ({how})"
            print(
                f"    {r['calc']:8s} real={r['natoms_real']:2d} ghost={r['natoms_ghost']:2d} "
                f"nbf={r['nbf']:4d} naux={r['naux']:4d} frozen_core={r['frozen_core']}{est}"
            )
    if total:
        print(
            f"\n  ESTIMATE (not measured): ferric ~{total / 60:.0f} min from the #474 pilot "
            f"(load unknown, exact-JK SCF); PySCF {a.scf} RHF + DF-MP2 unmeasured here, assume 1-2x "
            f"ferric -> total ~{2 * total / 60:.0f}-{3 * total / 60:.0f} min on a quiet box"
        )


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dimers", default=DEFAULT_DIMERS, help="S66 indices, e.g. 1,2")
    p.add_argument("--basis", default=s66.BASIS)
    p.add_argument("--aux", default=s66.AUX)
    p.add_argument(
        "--scf",
        choices=SCF_MODES,
        default="rijk",
        help="reference SCF J/K in BOTH codes: rijk = DF-HF on --jk-aux (the "
        "benchmark's setting), exact = four-centre",
    )
    p.add_argument(
        "--jk-aux",
        help=f"JK-fitting basis for --scf rijk (default {s66.JK_AUX}); refused with exact",
    )
    p.add_argument("--density-conv", type=float, default=DENSITY_CONV)
    p.add_argument("--conv-tol", type=float, default=PYSCF_CONV_TOL)
    p.add_argument("--conv-tol-grad", type=float, default=PYSCF_CONV_TOL_GRAD)
    p.add_argument("--bar-int-kcal", type=float, default=BAR_INT_KCAL)
    p.add_argument("--bar-corr-ha", type=float, default=BAR_CORR_HA)
    p.add_argument("--mem-gb", type=float, help="ferric memory_budget_gb")
    p.add_argument("--pyscf-max-memory", type=int, default=8000, help="MB")
    p.add_argument("--so-dir", help="frozen ferric package dir (s66 freeze-so)")
    p.add_argument("--out", help="JSON path (default scripts/queue/out/...)")
    p.add_argument("--allow-default-scf", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help=f"(H2)2 {SMOKE_BASIS}/{SMOKE_AUX}: plumbing + the <=1e-8 Ha anchor",
    )
    a = p.parse_args()
    if a.smoke:
        a.basis, a.aux = SMOKE_BASIS, SMOKE_AUX
    if a.scf == "exact" and a.jk_aux is not None:
        p.error("--jk-aux is only read with --scf rijk")
    if a.scf == "rijk" and a.jk_aux is None:
        a.jk_aux = s66.JK_AUX
    stem = "s66_dfmp2_anchor" + ("_smoke" if a.smoke else "")
    out = Path(a.out or OUT_DIR / f"{stem}{'_exact' if a.scf == 'exact' else ''}.json")
    settings = {
        "mode": "smoke" if a.smoke else "anchor",
        "dimers": "smoke (H2)2" if a.smoke else a.dimers,
        "basis": a.basis,
        "aux": a.aux,
        "scf_jk": a.scf,
        "jk_aux": a.jk_aux,
        "scf": (
            f"DF-HF in both: ferric run_rimp2(scf_jk='rijk', jk_auxbasis='{a.jk_aux}'); "
            f"PySCF RHF.density_fit(auxbasis=bundled {a.jk_aux}); DFMP2 on its own {a.aux} DF"
            if a.scf == "rijk"
            else "exact 4-centre J/K in both (ferric run_rimp2 scf_jk default; PySCF RHF, no DF)"
        ),
        "ferric_density_conv": a.density_conv,
        "pyscf_conv_tol": a.conv_tol,
        "pyscf_conv_tol_grad": a.conv_tol_grad,
        "frozen_core": "1 per real first-row heavy atom (ghosts 0)",
        "counterpoise": "CP (ghost basis + ghost aux) AND noCP, both gated",
        "lindep_thresh": float(
            os.environ.get("FERRIC_LINDEP_THRESH", FERRIC_LINDEP_DEFAULT)
        ),
        "bar_dE_int_kcal": a.bar_int_kcal,
        "bar_dE_corr_ha": SMOKE_BAR_HA if a.smoke else a.bar_corr_ha,
        "out": str(out),
    }
    if a.dry_run:
        cmd_dry_run(a, settings)
        return

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    ferric = s66.import_frozen_ferric(a.so_dir)
    import pyscf
    from pyscf import lib

    require_scf_jk(ferric, a)
    a.ferric_density_conv = ferric_density_conv(ferric, a)
    settings["ferric_density_conv"] = a.ferric_density_conv or "library default 1e-6"
    doc = {
        "schema": "s66_dfmp2_anchor/1",
        "settings": settings,
        "versions": {
            "ferric_build": getattr(ferric, "__build__", None),
            "ferric_file": ferric.__file__,
            "pyscf": pyscf.__version__,
            "pyscf_threads": lib.num_threads(),
            "python": platform.python_version(),
        },
        "git": s66.git_info(),
        "host": platform.node(),
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "systems": [],
    }
    print(f"ferric {doc['versions']['ferric_build']}  pyscf {pyscf.__version__}")
    all_fails = []
    for sysd in systems(a):
        print(f"\n{sysd['label']}  ({a.basis} / {a.aux})")
        rows = {}
        for plan in calc_plan(sysd, a.basis, a.aux):
            c = plan["calc"]
            sub, real = s66.calc_atoms(sysd["atoms"], sysd["a_size"], c)
            xyz, nfc = s66.build_calc(sysd["atoms"], sysd["a_size"], c)
            rows[c] = {
                **plan,
                "ferric": run_ferric(ferric, xyz, nfc, a),
                "pyscf": run_pyscf(sub, real, nfc, a),
            }
        inter, fails = check_system(sysd, rows, a)
        all_fails += [f"{sysd['label']} {f}" for f in fails]
        doc["systems"].append(
            {
                "label": sysd["label"],
                "dimer": sysd.get("dimer"),
                "a_size": sysd["a_size"],
                "calcs": rows,
                "interaction": inter,
                "failures": fails,
            }
        )
        write(out, doc)
    doc["passed"] = not all_fails
    doc["failures"] = all_fails
    write(out, doc)
    print(f"\nwrote {out}")
    if all_fails:
        print("FAIL:\n  " + "\n  ".join(all_fails))
        sys.exit(1)
    print("PASS: all gates")


if __name__ == "__main__":
    main()
