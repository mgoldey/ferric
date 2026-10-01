"""NWChem references for the VALIDATION.md "cDFT-ET coupling (Wu-Van Voorhis)" row.

Consumer: crates/ferric-scf/tests/validation_cdft_et.rs.
Output:   testdata/reference/validation/cdft_et/he2p_r<R>_<basis>.json
Inputs:   scripts/validation/nwchem/cdft_et/he2p_r<R>_<basis>_<grid>.nw
          (written by this script; the committed files are exactly what ran)

System: He2+ (charge +1, doublet, 3 electrons) at R = 2.50 / 3.00 / 3.50 A
(testdata/molecules/validation/he2p_r<R>.xyz), x def2-SVP and aug-cc-pVDZ from
FERRIC's bundled JSON (s and p shells only, so there is no spherical/Cartesian
or d-ordering question; p is x,y,z in NWChem, PySCF and ferric). AO ORDER TRAP:
He aug-cc-pVDZ is s,s,p,s,p in the BSE file; ferric and the NWChem input keep
that order, PySCF regroups by l. The stored MOs are in ferric order; the PySCF
recompute permutes rows (ferric_to_pyscf_rows) behind a hard orthonormality
gate. Without the permutation H1(RP) was off by 9e-3 Ha (caught by the gate).

Two charge-localized diabats per geometry, each a constrained UKS/PBE state:
    A ("reactants"): Becke population of He1 = 1.0 e  (hole on atom 0)
    B ("products"):  Becke population of He2 = 1.0 e  (hole on atom 1)
NWChem's `cdft <a> <a> charge 1.0 pop becke` (charge q -> N = Z - q = 1). Same
Fock sign convention as ferric (+lambda W, residual Tr[W D] - N); see
gen_cdft.py for the source-level like-for-like notes (grid, Becke partition,
pure HF impossible in NWChem cdft), which apply unchanged here. He2 is
homonuclear, so the Becke size adjustment is exactly zero and the He
Bragg-Slater radius (which differs between codes) cannot matter.

WHAT NWChem's `et` MODULE COMPUTES (read from src/etrans/et_calc.F, et_fock.F,
et_movecs_read.F of NWChem 7.2.2; reproduced to <=5e-10 Ha by the independent
PySCF recomputation below). It is NOT the Wu-Van Voorhis coupling:

    S_RP  = det(M_alpha) det(M_beta),  M_s = C_B,occ^T S C_A,occ  (signed,
            via SVD: prod(singular values) * det(U) det(V))
    H(RP) = <A|H|B> with the FULL electronic Hamiltonian (T + V_ne + 1/r12),
            generalized Slater-Condon via cofactor-weighted transition
            densities; 2e part = HF-form J - K of the (non-symmetric)
            transition density. No XC functional, no constraint potential.
    H(RR), H(PP) = the SCF energies stored in the movecs files MINUS E_nuc
            (for a DFT run these are the PBE energies, NOT <A|H|A>).
    V(RP) = | [H(RP) - S_RP (H(RR)+H(PP))/2] / (1 - S_RP^2) |

The final symmetric-orthogonalization step is IDENTICAL to ferric's
`coupling_hab`. The raw element differs by construction: ferric uses the
Wu-VV approximation <A|H|B> ~ 1/2[(E_B S - lam_B <A|w_B|B>) + (A<->B)], with no
two-electron transition integrals. With PBE determinants NWChem's V also mixes
a HF-form off-diagonal with PBE diagonals (an inconsistency of the ET module
when fed DFT vectors), so it is a well-defined function of the determinants but
not a "more correct" coupling than Wu-VV. Hence what the test can compare
like-for-like is (a) the state ingredients (E, lambda, |S_AB|), (b) NWChem's
V(RP) recomputed by ferric's own integrals on NWChem's determinants, and (c)
the log-slope d ln|H| / dR, which is insensitive to a constant prefactor.

Grids, as in gen_cdft.py: "matched" 99 x 302 (ferric's cDFT default) and
"converged" 300 x 974 (NWChem's limit). Integrals `direct`, `tolerances tight`;
`et` 2e screening tol2e = 1e-14 (NWChem's default is min(1e-7, |S| 1e-7),
which costs ~4e-10 Ha in H2(RP) here).

Every run is cross-checked: the movecs files are parsed (Fortran unformatted,
layout from et_movecs_read.F) and S_RP, H1(RP), H2(RP), V(RP) are recomputed in
PySCF from ferric's basis; disagreement with NWChem's printout beyond 1e-8 Ha
aborts. The full-precision S_RP (NWChem prints 3 significant figures) and the
occupied MO coefficients are written to the reference, so the ferric test can
evaluate its own kernels on NWChem's determinants.

Run (light step, ~2-4 min total):
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_cdft_et.py [R ...]
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from gen_cdft import nwchem_basis_block  # noqa: E402

ROW = "cdft_et"
ROW_NAME = "cDFT-ET coupling (Wu-Van Voorhis)"
SEPARATIONS = ("2.50", "3.00", "3.50")
BASES = ("def2-svp", "aug-cc-pvdz")
NWCHEM = "/usr/bin/nwchem"
INPUT_DIR = common.ROOT / "scripts" / "validation" / "nwchem" / "cdft_et"
SCRATCH_ROOT = common.ROOT / "target" / "nwchem-cdft-et"
XC = "xpbe96 cpbe96"
FUNCTIONAL = "PBE"
CHARGE, MULT = 1, 2
TARGET_N = 1.0  # electrons on the hole-bearing He
GRIDS = {
    "matched": (99, 11, 302),
    "converged": (300, 16, 974),
}
SCF_CONV = {"energy": 1e-11, "density": 1e-9}
CDFT_CONV = 1e-10
ET_TOL2E = 1e-14
# NWChem-printout vs PySCF-recompute agreement required before writing.
RECOMPUTE_TOL = 1e-8
NWCHEM_TIMEOUT_S = 1800
NWCHEM_FAILURE_MARKERS = (
    "Calculation failed to converge",
    "CDFT failed to optimize multipliers",
    "not converged",
)


def dft_block(grid: str, cdft_atom: int | None, movecs_out: str) -> str:
    nrad, iang, _ = GRIDS[grid]
    lines = [
        "dft",
        "  odft",
        f"  mult {MULT}",
        f"  xc {XC}",
        f"  grid lebedev {nrad} {iang} becke treutler",
        f"  convergence energy {SCF_CONV['energy']:.0e} "
        f"density {SCF_CONV['density']:.0e} nolevelshifting",
        "  tolerances tight",
        "  iterations 300",
        "  direct",
    ]
    if cdft_atom is not None:
        # NWChem `charge q` is the fragment CHARGE: N = Z - q = 2 - 1 = 1.
        q = 2.0 - TARGET_N
        lines.append(
            f"  cdft {cdft_atom} {cdft_atom} charge {q!r} pop becke "
            f"convergence {CDFT_CONV:.0e}"
        )
    lines += [f"  vectors input atomic output {movecs_out}", "end", "task dft energy"]
    return "\n".join(lines)


def nwchem_input(symbols, coords, basis_name: str, grid: str) -> str:
    geom = "\n".join(
        f"  {s} {x!r} {y!r} {z!r}" for s, (x, y, z) in zip(symbols, coords)
    )
    # ORDER MATTERS: NWChem dft directives persist in the rtdb, so the
    # unconstrained task runs FIRST (before any cdft line has been set); the
    # B block then replaces A's cdft line.
    return "\n".join(
        [
            "start he2p",
            "title ferric-validation-cdft-et",
            "permanent_dir .",
            "scratch_dir .",
            "geometry units bohr noautoz noautosym nocenter",
            "  symmetry c1",
            geom,
            "end",
            nwchem_basis_block(basis_name, symbols).rstrip("\n"),
            f"charge {CHARGE}",
            "set dft:no_prune T",
            "set dft:cdft_maxiter 200",
            dft_block(grid, None, "unc.movecs"),
            dft_block(grid, 1, "a.movecs"),
            dft_block(grid, 2, "b.movecs"),
            "et",
            "  vectors reactants a.movecs",
            "  vectors products b.movecs",
            f"  tol2e {ET_TOL2E:.0e}",
            "end",
            "task dft et",
            "",
        ]
    )


def run_nwchem(inp_text: str, inp_path: Path) -> tuple[str, dict]:
    """Run one input; return (stdout, {name: movecs bytes-parsed})."""
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    inp_path.write_text(inp_text)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=SCRATCH_ROOT))
    try:
        shutil.copy(inp_path, work / "he2p.nw")
        try:
            proc = subprocess.run(
                [NWCHEM, "he2p.nw"],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=NWCHEM_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{inp_path.name}: NWChem timed out") from exc
        out = proc.stdout
        if proc.returncode != 0:
            err = "\n".join(proc.stderr.splitlines()[-20:])
            raise RuntimeError(
                f"{inp_path.name}: NWChem exited {proc.returncode}\n{err}"
            )
        for bad in NWCHEM_FAILURE_MARKERS:
            if bad in out:
                raise RuntimeError(f"{inp_path.name}: NWChem reported {bad!r}")
        movecs = {n: read_movecs(work / f"{n}.movecs") for n in ("unc", "a", "b")}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out, movecs


def read_movecs(path: Path) -> dict:
    """NWChem movecs (Fortran unformatted sequential), layout per
    src/etrans/et_movecs_read.F: 6 header records, nsets, nbf, nmo(nsets),
    then per set occ(nbf), evals(nbf), nmo vector records, then (energy, enrep).
    """
    from scipy.io import FortranFile

    f = FortranFile(str(path), "r")
    recs = []
    while True:
        try:
            recs.append(f.read_record(np.uint8).tobytes())
        except Exception:  # EOF
            break
    f.close()
    it = np.int64 if len(recs[6]) == 8 else np.int32
    nsets = int(np.frombuffer(recs[6], it)[0])
    nbf = int(np.frombuffer(recs[7], it)[0])
    nmo = [int(v) for v in np.frombuffer(recs[8], it)]
    if nsets != 2:
        raise RuntimeError(f"{path}: nsets={nsets}, expected 2 (UKS)")
    k, sets = 9, []
    for s in range(nsets):
        occ = np.frombuffer(recs[k], np.float64).copy()
        k += 1
        k += 1  # evals
        c = np.array([np.frombuffer(recs[k + i], np.float64) for i in range(nmo[s])]).T
        k += nmo[s]
        if c.shape[0] != nbf:
            raise RuntimeError(f"{path}: MO record length {c.shape[0]} != nbf {nbf}")
        sets.append((occ, c))
    energy, enrep = np.frombuffer(recs[k], np.float64)[:2]
    return {"sets": sets, "energy": float(energy), "enrep": float(enrep)}


def occupied(mv: dict) -> list[np.ndarray]:
    out = []
    for occ, c in mv["sets"]:
        n = int(round(occ.sum()))
        if not np.allclose(occ[:n], 1.0) or not np.allclose(occ[n:], 0.0):
            raise RuntimeError("non-aufbau or fractional occupation in movecs")
        out.append(c[:, :n].copy())
    return out


def ferric_to_pyscf_rows(mol, basis_name: str, symbols: list[str]) -> np.ndarray:
    """Row permutation p with C_pyscf = C_ferric[p].

    ferric (and the NWChem input this script writes) keeps each atom's shells
    in BSE-file order; for He aug-cc-pVDZ that is s,s,p,s,p. PySCF groups an
    atom's shells by l. Derived from mol._bas (atom, l) with a STABLE match
    inside each (atom, l) group; validated by the hard orthonormality gate in
    recompute_et (a wrong permutation breaks C^T S C = I by O(1)).
    """
    ferric_ao = []  # (atom, l, k-th shell of that (atom,l), component) -> ferric AO index
    idx = 0
    for ia, sym in enumerate(symbols):
        seen = {}
        for sh in common.ferric_shells(basis_name, common.z_of(sym)):
            ell = sh["l"]
            if sh["pure"]:
                raise ValueError("l>=2 pure shells not expected for He")
            k = seen.get(ell, 0)
            seen[ell] = k + 1
            for comp in range((ell + 1) * (ell + 2) // 2):
                ferric_ao.append(((ia, ell, k, comp), idx))
                idx += 1
    lookup = dict(ferric_ao)
    perm, seen = [], {}
    for ib in range(mol.nbas):
        ia, ell = mol.bas_atom(ib), mol.bas_angular(ib)
        if mol.bas_nctr(ib) != 1:
            raise ValueError("expected segmented shells in the PySCF basis")
        k = seen.get((ia, ell), 0)
        seen[(ia, ell)] = k + 1
        for comp in range((ell + 1) * (ell + 2) // 2):
            perm.append(lookup[(ia, ell, k, comp)])
    if sorted(perm) != list(range(len(perm))):
        raise RuntimeError("ferric->PySCF AO permutation is not a bijection")
    return np.array(perm)


ORTHO_GATE = 1e-10


def recompute_et(mol, occ_a, occ_b, e_a_elec, e_b_elec) -> dict:
    """Independent re-implementation of NWChem's et_calc in PySCF integrals:
    generalized Slater-Condon with the explicit-inverse transition density
    P_s = B (A^T S B)^-1 A^T (no SVD/cofactor path), exact 4-index ERIs."""
    s_ao = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    s_ab, ps, svals, ortho = 1.0, [], [], 0.0
    for a, b in zip(occ_a, occ_b):
        ortho = max(
            ortho,
            float(abs(a.T @ s_ao @ a - np.eye(a.shape[1])).max()),
            float(abs(b.T @ s_ao @ b - np.eye(b.shape[1])).max()),
        )
        m = a.T @ s_ao @ b
        s_ab *= float(np.linalg.det(m))
        svals.append(np.linalg.svd(m, compute_uv=False).tolist())
        ps.append(b @ np.linalg.inv(m) @ a.T)
    one = sum(float(np.einsum("mn,nm", h, p)) for p in ps)
    pt = ps[0] + ps[1]
    j = float(np.einsum("mnlk,nm,kl", eri, pt, pt))
    k = sum(float(np.einsum("mnlk,km,nl", eri, p, p)) for p in ps)
    h1, h2 = s_ab * one, s_ab * 0.5 * (j - k)
    h_rp = h1 + h2
    v = (h_rp - s_ab * 0.5 * (e_a_elec + e_b_elec)) / (1.0 - s_ab * s_ab)
    if ortho > ORTHO_GATE:
        raise RuntimeError(
            f"NWChem MOs not orthonormal under PySCF overlap ({ortho:.2e}): AO order mismatch"
        )
    return {
        "s_rp": s_ab,
        "h1_rp": h1,
        "h2_rp": h2,
        "h_rp": h_rp,
        "v_rp_signed": v,
        "v_rp_abs": abs(v),
        "singular_values": svals,
        "max_mo_orthonormality_error_pyscf_overlap": ortho,
    }


def parse(out: str, label: str) -> dict:
    energies = [float(x) for x in re.findall(r"Total DFT energy =\s+(-?\d+\.\d+)", out)]
    # NWChem 7.2.2 quirk (seen on he2p_r3.00_aug-cc-pvdz_converged, state B):
    # the cdft loop internally zeroes the multiplier for one inner step (state A
    # of the same run shows a transient `CDFT multipliers: 0.0` that kicks the
    # DIIS error to 1.9e-2 before lambda is restored), and when that lands on
    # the last step `CDFT final multipliers` prints 0.0 although the converged
    # state and energy are the constrained ones (E_B == E_A to 7e-11). So:
    # take the printed final value, and if it is EXACTLY 0.0 fall back to the
    # last nonzero `CDFT multipliers:` value of that task, and flag it.
    lams, lam_fallback = [], []
    for sect in out.split("CDFT final multipliers")[1:]:
        m = re.match(r"\s*\n\s*1\s+(-?\d+\.\d+)", sect)
        if m is None:
            raise RuntimeError(f"{label}: unparsable CDFT final multipliers block")
        lams.append(float(m.group(1)))
    sections = out.split("CDFT final multipliers")[:-1]
    for i, val in enumerate(lams):
        if val == 0.0:
            # The task's own iterations: text between the previous task's
            # final print (or start) and this one.
            seq = [
                float(x)
                for x in re.findall(
                    r"CDFT multipliers:\s*\n\s*1\s+(-?\d+\.\d+)", sections[i]
                )
            ]
            nonzero = [x for x in seq if x != 0.0]
            if not nonzero:
                raise RuntimeError(
                    f"{label}: cdft task {i} never had a nonzero multiplier"
                )
            lams[i] = nonzero[-1]
            lam_fallback.append(i)
    s2 = [
        float(x)
        for x in re.findall(
            r"Expectation value of S2:\s*\n[-\s]*\n?\s*<S2> =\s+(\d+\.\d+)", out
        )
    ]
    if len(energies) != 3 or len(lams) != 2:
        tail = "\n".join(out.splitlines()[-30:])
        raise RuntimeError(
            f"{label}: expected 3 energies / 2 multipliers, got "
            f"{len(energies)} / {len(lams)}\n{tail}"
        )
    enuc = float(re.findall(r"Nuclear repulsion energy =\s+(-?\d+\.\d+)", out)[-1])
    ver = re.search(r"\(NWChem\)\s+(\S+)", out)
    if (
        out.count("Grid pruning is: off") < 3
        or out.count("Spatial weights used: Becke") < 3
    ):
        raise RuntimeError(f"{label}: grid is not the unpruned Becke grid requested")

    def et(tag):
        m = re.search(rf"{re.escape(tag)}\s+(-?\d+\.\d+)", out)
        if not m:
            raise RuntimeError(f"{label}: no {tag} in et output")
        return float(m.group(1))

    s_rp = re.search(r"S\(RP\)\s*:\s*(-?\d+\.\d+D[-+]\d+)", out)
    return {
        "energies": energies,
        "lambdas": lams,
        "lambda_final_printed_zero": lam_fallback,
        "s2": s2,
        "nuclear_repulsion": enuc,
        "version": ver.group(1) if ver else "unknown",
        "et": {
            "h_rr": et("H(RR)"),
            "h_pp": et("H(PP)"),
            "s_rp_printed_3sf": float(s_rp.group(1).replace("D", "E"))
            if s_rp
            else None,
            "h1_rp": et("H1(RP)"),
            "h2_rp": et("H2(RP)"),
            "h_rp": et("H(RP)"),
            "v_rp_abs": et("|V(RP)|"),
        },
    }


def main() -> int:
    from pyscf import gto

    only = set(sys.argv[1:])
    written, version = [], None
    for r_tag in SEPARATIONS:
        if only and r_tag not in only:
            continue
        system = f"he2p_r{r_tag}"
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        n_elec = sum(common.z_of(s) for s in symbols) - CHARGE
        for basis_name in BASES:
            bas, cart = common.pyscf_basis(basis_name, symbols)
            mol = gto.M(
                atom=common.pyscf_atom_bohr(symbols, coords),
                unit="Bohr",
                basis=bas,
                cart=bool(cart),
                charge=CHARGE,
                spin=MULT - 1,
            )
            per_grid = {}
            for grid in GRIDS:
                label = f"{system}_{basis_name}_{grid}"
                txt = nwchem_input(symbols, coords, basis_name, grid)
                out, mv = run_nwchem(txt, INPUT_DIR / f"{label}.nw")
                p = parse(out, label)
                version = p["version"]
                enuc = p["nuclear_repulsion"]
                e_unc, e_a, e_b = p["energies"]
                lam_a, lam_b = p["lambdas"]
                # Mirror symmetry guards the fallback above: the two diabats
                # are reflections, so their multipliers must agree.
                if abs(lam_a - lam_b) > 1e-6 or abs(e_a - e_b) > 1e-8:
                    raise RuntimeError(
                        f"{label}: mirror diabats disagree: lam {lam_a} vs {lam_b}, "
                        f"E {e_a} vs {e_b}"
                    )
                occ_a, occ_b = occupied(mv["a"]), occupied(mv["b"])
                perm = ferric_to_pyscf_rows(mol, basis_name, symbols)
                rc = recompute_et(
                    mol,
                    [c[perm] for c in occ_a],
                    [c[perm] for c in occ_b],
                    e_a - enuc,
                    e_b - enuc,
                )
                # The movecs energy record must be the printed DFT energy.
                for nm, e in (("a", e_a), ("b", e_b)):
                    if abs(mv[nm]["energy"] - e) > 1e-9:
                        raise RuntimeError(
                            f"{label}: movecs {nm} energy {mv[nm]['energy']} != {e}"
                        )
                for key in ("h1_rp", "h2_rp", "h_rp", "v_rp_abs"):
                    d = abs(rc[key] - p["et"][key])
                    if d > RECOMPUTE_TOL:
                        raise RuntimeError(
                            f"{label}: PySCF recompute of {key} {rc[key]:.12f} vs NWChem "
                            f"{p['et'][key]:.12f} differs by {d:.2e}"
                        )
                if (
                    p["et"]["s_rp_printed_3sf"] is not None
                    and abs(rc["s_rp"] - p["et"]["s_rp_printed_3sf"])
                    > 0.006 * abs(rc["s_rp"]) + 1e-12
                ):
                    raise RuntimeError(
                        f"{label}: S_RP recompute {rc['s_rp']} vs printed"
                    )
                # Wu-VV coupling from NWChem's ingredients, SYMMETRIC CASE ONLY.
                # With lam_a = lam_b = lam and the Becke partition of unity
                # (w_He1 + w_He2 = 1 pointwise), <A|w_1|B> + <A|w_2|B> =
                # N_e S_AB to grid quadrature, so no W matrix is needed:
                #   textbook (F_X = E_X + lam N_X):  lam S (N - N_e/2)/(1-S^2)
                #   E-only form (ferric's current): -lam S (N_e/2)/(1-S^2)
                lam_bar = 0.5 * (lam_a + lam_b)
                s = rc["s_rp"]
                wu_vv = {
                    "note": "derived, not computed by NWChem; symmetric-dimer identity "
                    "(lam_a = lam_b, partition of unity); see module docstring",
                    "lambda_mean": lam_bar,
                    "lambda_asymmetry": lam_a - lam_b,
                    "h_ab_textbook_F_eq_E_plus_lamN": lam_bar
                    * s
                    * (TARGET_N - 0.5 * n_elec)
                    / (1.0 - s * s),
                    "h_ab_E_only_form": -lam_bar * s * (0.5 * n_elec) / (1.0 - s * s),
                }
                per_grid[grid] = {
                    "unconstrained": {"energy": e_unc},
                    "state_a": {
                        "constrained_atom": 0,
                        "target": TARGET_N,
                        "energy": e_a,
                        "lambda": lam_a,
                        "mo_occ_alpha": occ_a[0].tolist(),
                        "mo_occ_beta": occ_a[1].tolist(),
                    },
                    "state_b": {
                        "constrained_atom": 1,
                        "target": TARGET_N,
                        "energy": e_b,
                        "lambda": lam_b,
                        "mo_occ_alpha": occ_b[0].tolist(),
                        "mo_occ_beta": occ_b[1].tolist(),
                    },
                    "s2_printed": p["s2"],
                    "lambda_final_printed_zero_states": [
                        ["state_a", "state_b"][i]
                        for i in p["lambda_final_printed_zero"]
                    ],
                    "et_nwchem": p["et"],
                    "et_pyscf_recompute": rc,
                    "wu_vv_from_nwchem_ingredients": wu_vv,
                }
                print(
                    f"{label:32s} E_a={e_a:.10f} E_b={e_b:.10f} lam={lam_a:+.8f}/{lam_b:+.8f} "
                    f"S={s:+.6e} |V|={rc['v_rp_abs']:.8e} WuVV(F)={abs(wu_vv['h_ab_textbook_F_eq_E_plus_lamN']):.8e} "
                    f"WuVV(E)={abs(wu_vv['h_ab_E_only_form']):.8e}",
                    flush=True,
                )
            payload = {
                "row": ROW_NAME,
                "system": system,
                "r_angstrom": float(r_tag),
                "basis": basis_name,
                "charge": CHARGE,
                "multiplicity": MULT,
                "n_electrons": n_elec,
                "functional": FUNCTIONAL,
                "nuclear_repulsion": enuc,
                "nao": common.ferric_nao(basis_name, symbols),
                "mo_layout": "AO rows in FERRIC/NWChem order (each atom's shells in BSE-file "
                "order, general contractions split, p as x,y,z; NOT PySCF's l-grouped order), "
                "occupied columns only, unit-normalized contracted AOs",
                "grids": {
                    g: {"radial": n, "nwchem_lebedev_index": i, "angular": a}
                    for g, (n, i, a) in GRIDS.items()
                },
                "results": per_grid,
                "provenance": common.provenance(
                    code="NWChem",
                    version=version,
                    keywords={
                        "module": "dft (odft) x3 then et",
                        "xc": XC,
                        "cdft": f"cdft <atom> <atom> charge {2.0 - TARGET_N} pop becke, "
                        f"convergence {CDFT_CONV:.0e}, cdft_maxiter 200",
                        "et": f"vectors reactants a.movecs products b.movecs; tol2e {ET_TOL2E:.0e}; "
                        "fock (default 2e method)",
                        "grid": "lebedev <nrad> <iang> becke treutler; set dft:no_prune T",
                        "integrals": "direct, tolerances tight, no density fitting",
                        "inputs": str(INPUT_DIR.relative_to(common.ROOT))
                        + f"/{system}_{basis_name}_*.nw",
                    },
                    basis_name=basis_name,
                    xyz_path=xyz,
                    coords_bohr=coords,
                    symbols=symbols,
                    grid={
                        g: f"{n}x{a} Treutler(Perez-Jorda Chebyshev) x Lebedev, Becke, unpruned"
                        for g, (n, _, a) in GRIDS.items()
                    },
                    aux=None,
                    frozen_core=None,
                    scf_conv={**SCF_CONV, "cdft_multiplier": CDFT_CONV},
                    stability=None,
                    generator="scripts/validation/gen_cdft_et.py",
                    extra={
                        "stability_note": "not run: NWChem has no stability analysis for "
                        "cdft-constrained states; the ferric test checks mirror symmetry "
                        "and <S^2> instead",
                        "et_recompute_check": f"PySCF explicit-inverse Slater-Condon, "
                        f"agreement with NWChem printout required < {RECOMPUTE_TOL:.0e} Ha",
                    },
                ),
            }
            written.append(common.write_reference(ROW, system, basis_name, payload))
    print(f"GEN_CDFT_ET_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
