"""NWChem references for the VALIDATION.md row "cDFT constrained state selection".

Consumer: crates/ferric-scf/tests/validation_cdft_state.rs.
Output:   testdata/reference/validation/cdft_state/<system>_def2-svp.json
Inputs:   scripts/validation/nwchem/cdft_state/<system>_<variant>_<state>_<grid>_<eps>.nw
          (written by this script; the committed files are exactly what ran)

WHAT THE ROW TESTS
------------------
At an over-constrained Becke target the constrained UHF problem has several
solutions, and which one a solver returns depends on where it starts. This
script exports NWChem's converged alpha/beta orbitals for every state, in
FERRIC's AO order, so the Rust test can seed ferric's constrained solve with
them and check that it STAYS on that state.

Systems (geometries in testdata/molecules/validation/):

    hene    HeNe+  doublet  R = 2.0 A  fragment He  target N = 2.000 (integer;
            natural 1.954)          states: low = atomic guess, high = hcore
    lih_r3  LiH+   doublet  R = 3.0 A  fragment Li  target N = 2.800
            (natural ~2.1-2.3)      states: low = atomic guess,
                                            high = atomic guess + swap alpha 2 3

LiH+ is the second system because a guess sweep (R = 1.6/3.0 A, targets
2.5/2.8/2.95/3.0, guesses atomic/hcore/swap 2-3/swap 2-4) found it multi-valued
ONLY at R = 3.0, N = 2.8: atomic and hcore agree (E = -7.53128, lambda = -0.432)
while swap 2-3 converges to a different sigma state 2.3 eV higher (E = -7.44598,
lambda = -0.254). Both alpha HOMOs are Li 2s + H 1s mixtures (no pi
occupation). At R = 1.6 every guess that converged agreed; at N = 3.0 NWChem's
multipliers diverge; swap 2-4 never converged.

THE BECKE PARTITION IS NOT THE SAME IN THE TWO CODES FOR He AND Ne
------------------------------------------------------------------
Both codes use Becke 1988 cells with the Bragg-Slater size adjustment, which
depends on the radius RATIO chi = R_A/R_B only. The radii tables agree for H,
Li, O, F (the gen_cdft.py systems, and LiH+ here) but NOT for the noble gases:

    ferric (crates/ferric-dft/src/becke.rs BRAGG_ANGSTROM)   He 0.30  Ne 0.45
    NWChem (src/nwdft/grid/grid_atom_type_info.F BSrad)      He 0.35  Ne 0.50

so for HeNe+ chi = 0.6667 in ferric and 0.7000 in NWChem: the two codes impose
DIFFERENT constraints at "N_He = 2.000". Measured consequence: NWChem's native
LOW density has N_He = 2 - 8.75e-3 under ferric's W (Rust test log,
2026-09-30), which at lambda ~ -2.3 is the whole historical "0.563 eV above
NWChem LOW". The independent PySCF-grid population check below
(`becke_check`) reproduces that offset from NWChem's density with both radius
pairs, so it is a property of the partition, not of either code's quadrature.

NWChem has no input to set per-element Becke radii. The ferric-partition
variant therefore RELABELS the atoms: each center gets a different element tag
whose NWChem radius gives ferric's ratio exactly (He -> C 0.70, Ne -> Be 1.05;
0.70/1.05 = 0.30/0.45), with the true nuclear charge restored by the
geometry's per-atom `charge` keyword and the basis keyed by the new tag.
What else the element tag controls in NWChem, and why it does not matter:
  * the atomic guess -- not used: these runs read NWChem's own native-partition
    orbitals for the same state (`vectors input guess.movecs`);
  * the radial grid's per-element scale -- quadrature only, hence the two grids;
  * nothing in the HF energy: the relabeled UNCONSTRAINED energy must equal the
    native one (asserted, 1e-9 Ha).
For LiH+ the radii already agree, so the native runs ARE like-for-like and no
relabeled variant is run (`partition.substitute_tags` is null).

WHY "slater eps" AND THE EXTRAPOLATION
--------------------------------------
NWChem builds the Becke weight operator only on its XC grid, and builds that
grid only if some non-HF functional has |weight| > 1e-8 (`xc_gotxc`,
src/nwdft/xc/xc_util.F). With `xc hfexch` alone the cdft solve aborts; this
build has neither nwxc nor libxc. Each constrained state is therefore run as
`xc hfexch 1.0 slater eps` at eps = 1e-7, 2e-7, 3e-7: E0 = 2 E(1e-7) - E(2e-7)
(the HF energy at a constrained stationary point is stationary on the
constraint surface, so the eps response enters at second order), the second
difference E(1e-7) - 2 E(2e-7) + E(3e-7) is recorded and must be at noise (a
state change between eps values shows here first), lambda likewise. The
exported orbitals are the eps = 1e-7 ones; PySCF evaluates their HF energy as
an independent check of E0. The unconstrained anchor is pure `xc hfexch`.

AO ORDER NWChem -> ferric
-------------------------
NWChem is fed ferric's basis shell-by-shell in ferric's order
(gen_cdft.nwchem_basis_block), spherical. NWChem keeps input shell order per
atom, p as (x, y, z) and spherical d as m = -2..2, the same order and
normalization as ferric (libint2 standard) -- but its d(m=+1) (xz) function has
the OPPOSITE SIGN. The map is therefore a permutation-free SIGN FLIP of every
d(m=+1) row (`ao_signs`). Measured, not read off the source: NWChem's own AO
overlap, recovered as S = (C C^T)^-1 from a full-rank orbital set, was compared
with PySCF's in ferric order on HeNe+ at a generic orientation (every Ne d
component overlapping He s/p): m = -2, -1, 0, +2 agree to 1e-15, m = +1 agrees
to 1e-15 only after negation, every non-d block to 9e-15. Without the flip the
HeNe+ orbitals miss orthonormality by 5.1e-4 and their PySCF energy misses
NWChem's by 6.0e-7 Ha -- the check below caught exactly that. f and higher
shells are REFUSED (their NWChem sign convention was not measured). Tested on
every exported set: C^T S C = I with PySCF's overlap in ferric order, and
PySCF's UHF energy of the orbitals vs NWChem's (1e-8 unconstrained, 1e-7 vs
the extrapolated constrained E0). The Rust test repeats both inside ferric.

Columns are stored OCCUPIED FIRST (stable within occupied/virtual), the
`solve_uhf_fockmod` guess contract.

LIKE-FOR-LIKE otherwise as gen_cdft.py: `cdft ... pop becke`, `grid lebedev
<nrad> <iang> becke treutler`, `set dft:no_prune T`, direct, tolerances tight,
same constraint sign convention (lambda compares directly). Two grids:
matched (99 x 302, ferric's cDFT default) and converged (300 x 974).

Run (~1-2 min per system; one system per light step):
    cd ~/qc/ferric && <worktree>/scripts/validation/run_slot.sh --light -- \\
        .venv/bin/python <worktree>/scripts/validation/gen_cdft_state.py hene
(and again with lih_r3). Run from the main checkout so its .venv (PySCF) is
used; common.ROOT resolves to the worktree that holds this script.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_cdft  # noqa: E402

ROW = "cdft_state"
ROW_NAME = "cDFT constrained state selection"
BASIS = "def2-svp"
NWCHEM = "/usr/bin/nwchem"
INPUT_DIR = common.ROOT / "scripts" / "validation" / "nwchem" / "cdft_state"
SCRATCH_ROOT = common.ROOT / "target" / "nwchem-cdft-state"
GRIDS = gen_cdft.GRIDS  # matched (99, 302) and converged (300, 974)
EPS = (1e-7, 2e-7, 3e-7)
SECOND_DIFF_MAX = 1e-9  # Ha; a state change between eps values is >> this
ORTHO_MAX = 1e-8
E_ANCHOR_MAX = 1e-8  # PySCF E of NWChem's unconstrained orbitals vs NWChem
E_STATE_MAX = 1e-7  # PySCF E_HF of eps=1e-7 orbitals vs extrapolated E0
E_RELABEL_MAX = 1e-9  # relabeled vs native UNCONSTRAINED energy
FERRIC_BECKE_RS = common.ROOT / "crates" / "ferric-dft" / "src" / "becke.rs"
# NWChem 7.2.2 BSrad (Angstrom), src/nwdft/grid/grid_atom_type_info.F, Z = 1..10.
NWCHEM_BSRAD = [0.35, 0.35, 1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.50]

# system -> geometry file, charge, mult, fragment, target, {state: vectors}
SYSTEMS = {
    "hene": {
        "xyz": "hene.xyz",
        "charge": 1,
        "mult": 2,
        "fragment": [0],
        "target": 2.0,
        "states": {"low": None, "high": "input hcore"},
    },
    "lih_r3": {
        "xyz": "lih_r3.xyz",
        "charge": 1,
        "mult": 2,
        "fragment": [0],
        "target": 2.8,
        "states": {"low": None, "high": "input atomic swap alpha 2 3"},
    },
}


# ---------------------------------------------------------------------------
# Becke radii: ferric's table, NWChem's table, and the relabeling
# ---------------------------------------------------------------------------


def ferric_bragg_angstrom() -> list[float]:
    """ferric's BRAGG_ANGSTROM, parsed from the source it is compiled from."""
    src = FERRIC_BECKE_RS.read_text()
    m = re.search(r"const BRAGG_ANGSTROM: \[f64; (\d+)\] = \[(.*?)\];", src, re.S)
    if not m:
        raise RuntimeError(f"{FERRIC_BECKE_RS}: BRAGG_ANGSTROM not found")
    body = re.sub(r"//[^\n]*", "", m.group(2))
    vals = [float(x) for x in body.replace("\n", " ").split(",") if x.strip()]
    if len(vals) != int(m.group(1)):
        raise RuntimeError("BRAGG_ANGSTROM length mismatch")
    return vals  # index = Z


def substitute_tags(symbols: list[str]) -> dict | None:
    """None if NWChem's radius ratio already equals ferric's; else the element
    tags whose NWChem radii reproduce ferric's ratio exactly."""
    if len(symbols) != 2:
        raise ValueError("partition matching is implemented for diatomics only")
    fb = ferric_bragg_angstrom()
    za, zb = (common.z_of(s) for s in symbols)
    r_f = fb[za] / fb[zb]
    r_n = NWCHEM_BSRAD[za - 1] / NWCHEM_BSRAD[zb - 1]
    if abs(r_f - r_n) < 1e-12:
        return None
    for ta in range(1, 11):
        for tb in range(1, 11):
            if ta == tb:
                continue
            if abs(NWCHEM_BSRAD[ta - 1] / NWCHEM_BSRAD[tb - 1] - r_f) < 1e-12:
                return {
                    symbols[0]: common.ELEMENTS[ta],
                    symbols[1]: common.ELEMENTS[tb],
                }
    raise RuntimeError(f"no NWChem element pair reproduces ferric's ratio {r_f}")


def relabel(txt: str, tags: dict, vectors_file: str) -> str:
    """Retag atoms (geometry + basis), restore the true nuclear charge with the
    per-atom `charge` keyword, and read the guess from `vectors_file`."""
    for old, new in tags.items():
        z = common.z_of(old)
        txt, n = re.subn(
            rf"^  {old} (\S+ \S+ \S+)$", rf"  {new} \1 charge {z}.0", txt, flags=re.M
        )
        if n != 1:
            raise RuntimeError(f"relabel: {n} geometry lines for {old}")
        txt = re.sub(rf"^{old} ([SPDF])$", rf"{new} \1", txt, flags=re.M)
    txt = re.sub(r"^  vectors .*\n", "", txt, flags=re.M)
    # `output` is explicit: with a `vectors input <file>` NWChem otherwise
    # writes the converged orbitals back over <file>.
    return txt.replace(
        "  direct\n",
        f"  direct\n  vectors input {vectors_file} output cdftstate.movecs\n",
        1,
    )


# ---------------------------------------------------------------------------
# NWChem
# ---------------------------------------------------------------------------


def nw_input(symbols, coords, sysdef, grid, xc, cdft_line, vectors) -> str:
    txt = gen_cdft.nwchem_input(
        symbols, coords, BASIS, sysdef["charge"], sysdef["mult"], grid, cdft_line
    )
    txt = txt.replace("start cdft", "start cdftstate", 1)
    txt = txt.replace(
        "title ferric-validation-cdft", "title ferric-validation-cdft-state"
    )
    old = f"  xc {gen_cdft.XC}"
    assert old in txt
    txt = txt.replace(old, f"  xc {xc}")
    if vectors:
        txt = txt.replace("  direct\n", f"  direct\n  vectors {vectors}\n", 1)
    return txt


def run_nwchem(
    inp_text: str, inp_path: Path, guess: bytes | None = None
) -> tuple[str, bytes]:
    """Run NWChem; return (stdout, movecs bytes). `guess` is written to
    guess.movecs in the work directory first."""
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    inp_path.write_text(inp_text)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=SCRATCH_ROOT))
    try:
        shutil.copy(inp_path, work / "run.nw")
        if guess is not None:
            (work / "guess.movecs").write_bytes(guess)
        proc = subprocess.run(
            [NWCHEM, "run.nw"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=gen_cdft.NWCHEM_TIMEOUT_S,
        )
        out = proc.stdout
        if proc.returncode != 0:
            tail = "\n".join(out.splitlines()[-15:])
            raise RuntimeError(
                f"{inp_path.name}: NWChem exited {proc.returncode}\n{tail}"
            )
        for bad in gen_cdft.NWCHEM_FAILURE_MARKERS:
            if bad in out:
                raise RuntimeError(f"{inp_path.name}: NWChem reported {bad!r}")
        movecs = (work / "cdftstate.movecs").read_bytes()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out, movecs


def parse_energy(out: str, label: str) -> dict:
    e = re.findall(r"Total DFT energy =\s+(-?\d+\.\d+)", out)
    if not e:
        raise RuntimeError(f"{label}: no final energy")
    enuc = float(re.findall(r"Nuclear repulsion energy =\s+(-?\d+\.\d+)", out)[-1])
    ver = re.search(r"\(NWChem\)\s+(\S+)", out)
    return {
        "energy": float(e[-1]),
        "nuclear_repulsion": enuc,
        "version": ver.group(1) if ver else "unknown",
    }


def _records(blob: bytes) -> list[bytes]:
    """Fortran sequential unformatted records (4-byte little-endian markers)."""
    out, i = [], 0
    while i < len(blob):
        (n,) = struct.unpack_from("<i", blob, i)
        rec = blob[i + 4 : i + 4 + n]
        (n2,) = struct.unpack_from("<i", blob, i + 4 + n)
        if n != n2:
            raise ValueError("movecs: record markers disagree")
        out.append(rec)
        i += 8 + n
    return out


def read_movecs(blob: bytes) -> dict:
    """NWChem .movecs (layout from contrib/mov2asc/mov2asc.F): header records,
    nsets, nbf, nmo(nsets), then per set occupations, eigenvalues and nmo
    vectors of length nbf. Integers are 8-byte in this build (asserted by the
    record length)."""
    import numpy as np

    rec = _records(blob)

    def one_int(r: bytes) -> int:
        if len(r) != 8:
            raise ValueError(f"movecs: expected an 8-byte integer record, got {len(r)}")
        return struct.unpack("<q", r)[0]

    nsets, nbf = one_int(rec[6]), one_int(rec[7])
    nmo = list(struct.unpack(f"<{nsets}q", rec[8]))
    k, sets = 9, []
    for s in range(nsets):
        occ = np.frombuffer(rec[k], "<f8").copy()
        eig = np.frombuffer(rec[k + 1], "<f8").copy()
        k += 2
        c = np.array([np.frombuffer(rec[k + i], "<f8") for i in range(nmo[s])]).T
        k += nmo[s]
        if c.shape != (nbf, nmo[s]):
            raise ValueError("movecs: vector shape")
        sets.append({"occ": occ, "eig": eig, "C": c})
    return {"nbf": nbf, "nmo": nmo, "sets": sets}


def ao_signs(symbols: list[str]) -> list[float]:
    """Per-ferric-AO factor taking an NWChem coefficient row to ferric's
    convention: -1 on spherical d(m=+1), +1 elsewhere (see module doc)."""
    signs: list[float] = []
    for s in symbols:
        for sh in common.ferric_shells(BASIS, common.z_of(s)):
            ell = sh["l"]
            if ell >= 3:
                raise ValueError("NWChem sign convention for l>=3 was not measured")
            if ell == 2 and not sh["pure"]:
                raise ValueError("Cartesian d shells: NWChem convention not measured")
            n = 2 * ell + 1
            signs += [-1.0 if (ell == 2 and k == 3) else 1.0 for k in range(n)]
    return signs


def occupied_first(st: dict, nocc: int, signs: list[float]) -> dict:
    """Map rows to ferric's AO convention and reorder columns occupied-first;
    check the occupation count."""
    import numpy as np

    occ = st["occ"]
    if not np.all((np.abs(occ) < 1e-12) | (np.abs(occ - 1.0) < 1e-12)):
        raise ValueError(f"non-integer occupations {occ}")
    idx_o = [i for i in range(len(occ)) if occ[i] > 0.5]
    idx_v = [i for i in range(len(occ)) if occ[i] <= 0.5]
    if len(idx_o) != nocc:
        raise ValueError(f"{len(idx_o)} occupied orbitals, expected {nocc}")
    order = idx_o + idx_v
    return {
        "C": np.asarray(signs)[:, None] * st["C"][:, order],
        "occ": occ[order],
        "eig": st["eig"][order],
        "nwchem_column_order": order,
    }


# ---------------------------------------------------------------------------
# PySCF checks (independent of ferric)
# ---------------------------------------------------------------------------


def pyscf_checks(mol, perm, orbsets: dict, nocc: tuple[int, int]):
    """For each orbital set (ferric order): max|C^T S C - I| and PySCF's UHF
    energy of the occupied orbitals."""
    import numpy as np
    from pyscf import scf

    p = np.asarray(perm)
    s_f = mol.intor("int1e_ovlp")[np.ix_(p, p)]
    mf = scf.UHF(mol)
    out = {}
    for name, (ca, cb) in orbsets.items():
        ortho = max(
            float(np.abs(c.T @ s_f @ c - np.eye(c.shape[1])).max()) for c in (ca, cb)
        )
        e = float(mf.energy_tot(np.array(dm_pair(perm, ca, cb, nocc))))
        out[name] = {"max_abs_CtSC_minus_I": ortho, "pyscf_uhf_energy": e}
    return out, s_f


def dm_pair(perm, ca, cb, nocc):
    """alpha/beta densities in PySCF AO order from ferric-order orbitals."""
    import numpy as np

    p = np.asarray(perm)
    dms = []
    for c, n in zip((ca, cb), nocc):
        cp = np.empty_like(c)
        cp[p, :] = c  # ferric AO i is PySCF AO perm[i]
        dms.append(cp[:, :n] @ cp[:, :n].T)
    return dms


def becke_population(mol, dm_total, frag: int, radii_angstrom: tuple[float, float]):
    """Fragment Becke population of a diatomic with an EXPLICIT radius pair:
    Becke 1988 cells, size adjustment a = u/(u^2-1) clamped to +-1/2 (the
    formula both codes use), home-atom partition on PySCF Treutler-Ahlrichs
    (99, 302) atomic grids. An independent quadrature of the same integral
    either code computes, so it agrees with each to ~1e-6."""
    import numpy as np
    from pyscf.dft import gen_grid, numint, radi

    g = gen_grid.Grids(mol)
    atomic = g.gen_atomic_grids(
        mol, atom_grid=(99, 302), radi_method=radi.treutler_ahlrichs, prune=None
    )
    pts, wts, home = [], [], []
    for ia in range(mol.natm):
        c, w = atomic[mol.atom_symbol(ia)]
        pts.append(c + mol.atom_coord(ia))
        wts.append(w)
        home.append(np.full(len(w), ia))
    pts, wts, home = np.vstack(pts), np.concatenate(wts), np.concatenate(home)
    xyz = mol.atom_coords()
    dist = np.linalg.norm(pts[:, None, :] - xyz[None, :, :], axis=2)
    rab = float(np.linalg.norm(xyz[0] - xyz[1]))
    chi = radii_angstrom[0] / radii_angstrom[1]
    u = (chi - 1) / (chi + 1)
    a = min(max(u / (u * u - 1), -0.5), 0.5)

    def cell(i, j, aij):
        mu = (dist[:, i] - dist[:, j]) / rab
        nu = mu + aij * (1 - mu * mu)
        for _ in range(3):
            nu = 1.5 * nu - 0.5 * nu**3
        return 0.5 * (1 - nu)

    p0, p1 = cell(0, 1, a), cell(1, 0, -a)
    w0 = p0 / (p0 + p1)
    w_frag = w0 if frag == 0 else 1 - w0
    w_home = np.where(home == 0, w0, 1 - w0)
    rho = numint.eval_rho(mol, numint.eval_ao(mol, pts), dm_total)
    return float((wts * w_home * w_frag * rho).sum()), float((wts * w_home * rho).sum())


# ---------------------------------------------------------------------------


def mat(a) -> list:
    return [[float(x) for x in row] for row in a]


def orb_json(o) -> dict:
    return {
        "coefficients": mat(o["C"]),
        "occupations": [float(x) for x in o["occ"]],
        "eigenvalues": [float(x) for x in o["eig"]],
        "nwchem_column_order": o["nwchem_column_order"],
    }


def run_state(system, variant, state, mk_input, nocc, signs, guess=None) -> dict:
    """One state on both grids x three eps. Returns the extrapolated block
    plus the matched-grid eps=1e-7 orbitals and raw movecs."""
    block = {}
    for grid in GRIDS:
        runs = []
        for eps in EPS:
            label = f"{system}_{variant}_{state}_{grid}_eps{eps:.0e}"
            out, mv = run_nwchem(mk_input(grid, eps), INPUT_DIR / f"{label}.nw", guess)
            r = gen_cdft.parse(out, label)
            if r["lambda"] is None:
                raise RuntimeError(f"{label}: no CDFT multiplier printed")
            runs.append(
                {
                    "eps": eps,
                    "energy": r["energy"],
                    "lambda": r["lambda"],
                    "grid_integrated_density": r["grid_integrated_density"],
                }
            )
            if grid == "matched" and eps == EPS[0]:
                m = read_movecs(mv)
                block["_orbs"] = [
                    occupied_first(st, n, signs) for st, n in zip(m["sets"], nocc)
                ]
                block["_movecs"] = mv
            print(
                f"{label:44s} E = {r['energy']:.12f} lambda = {r['lambda']:+.10f}",
                flush=True,
            )
        e1, e2, e3 = (x["energy"] for x in runs)
        l1, l2, l3 = (x["lambda"] for x in runs)
        d2 = e1 - 2 * e2 + e3
        if abs(d2) > SECOND_DIFF_MAX:
            raise RuntimeError(
                f"{system}/{variant}/{state}/{grid}: E(eps) not linear (2nd diff {d2:.2e}); "
                "a state change between eps values?"
            )
        block[grid] = {
            "energy_extrapolated": 2 * e1 - e2,
            "lambda_extrapolated": 2 * l1 - l2,
            "energy_second_difference": d2,
            "lambda_second_difference": l1 - 2 * l2 + l3,
            "runs": runs,
        }
    return block


def main() -> int:
    import gen_properties as gp  # PySCF-dependent; imported here on purpose

    only = set(sys.argv[1:])
    written = []
    fb = ferric_bragg_angstrom()
    for system, sd in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / sd["xyz"]
        symbols, coords = common.read_xyz(xyz)
        nelec = sum(common.z_of(s) for s in symbols) - sd["charge"]
        nocc = ((nelec + sd["mult"] - 1) // 2, (nelec - sd["mult"] + 1) // 2)
        z_frag = sum(common.z_of(symbols[a]) for a in sd["fragment"])
        lo, hi = min(sd["fragment"]) + 1, max(sd["fragment"]) + 1
        q = z_frag - sd["target"]
        cdft_line = f"cdft {lo} {hi} charge {q!r} pop becke convergence {gen_cdft.CDFT_CONV:.0e}"
        nao = common.ferric_nao(BASIS, symbols)
        signs = ao_signs(symbols)
        assert len(signs) == nao
        tags = substitute_tags(symbols)
        variants = ["native"] + (["ferric_partition"] if tags else [])

        # -- unconstrained pure UHF anchor (no grid is built for hfexch) -----
        def unc_input(grid="matched"):
            return nw_input(symbols, coords, sd, grid, "hfexch", None, None)

        label = f"{system}_native_unconstrained_hf"
        out, unc_mv = run_nwchem(unc_input(), INPUT_DIR / f"{label}.nw")
        unc = parse_energy(out, label)
        version, enuc = unc["version"], unc["nuclear_repulsion"]
        m = read_movecs(unc_mv)
        if m["nbf"] != nao or m["nmo"] != [nao, nao]:
            raise RuntimeError(
                f"{label}: movecs nbf/nmo {m['nbf']}/{m['nmo']} vs nao {nao}"
            )
        unc_orbs = [occupied_first(st, n, signs) for st, n in zip(m["sets"], nocc)]
        print(f"{label:44s} E = {unc['energy']:.12f}", flush=True)
        unc_relabeled = None
        if tags:
            label = f"{system}_ferric_partition_unconstrained_hf"
            out, _ = run_nwchem(
                relabel(unc_input(), tags, "guess.movecs"),
                INPUT_DIR / f"{label}.nw",
                unc_mv,
            )
            unc_relabeled = parse_energy(out, label)["energy"]
            print(f"{label:44s} E = {unc_relabeled:.12f}", flush=True)
            if abs(unc_relabeled - unc["energy"]) > E_RELABEL_MAX:
                raise RuntimeError(
                    f"{system}: relabeled unconstrained E {unc_relabeled} != native "
                    f"{unc['energy']} -- the element tag changes more than the partition"
                )

        # -- constrained states ---------------------------------------------
        states: dict = {}
        for state, vectors in sd["states"].items():

            def native_input(grid, eps, vectors=vectors):
                return nw_input(
                    symbols,
                    coords,
                    sd,
                    grid,
                    f"hfexch 1.0 slater {eps:.1e}",
                    cdft_line,
                    vectors,
                )

            entry = {
                "nwchem_vectors": vectors or "default (atomic)",
                "like_for_like": variants[-1],
                "native": run_state(system, "native", state, native_input, nocc, signs),
            }
            if tags:

                def rel_input(grid, eps, f=native_input):
                    return relabel(f(grid, eps), tags, "guess.movecs")

                entry["ferric_partition"] = run_state(
                    system,
                    "ferric_partition",
                    state,
                    rel_input,
                    nocc,
                    signs,
                    guess=entry["native"]["_movecs"],
                )
                entry["ferric_partition"]["nwchem_guess"] = (
                    "this state's native-partition matched-grid eps=1e-7 orbitals"
                )
            states[state] = entry

        # -- PySCF checks: AO map + Becke populations under both radius pairs -
        mol = common.build_pyscf_mol(xyz, BASIS, sd["charge"], sd["mult"])
        perm = gp.ferric_ao_permutation(mol, BASIS, symbols)
        orbsets = {"unconstrained": (unc_orbs[0]["C"], unc_orbs[1]["C"])}
        want = {"unconstrained": unc["energy"]}
        for st, e in states.items():
            for v in variants:
                o = e[v]["_orbs"]
                orbsets[f"{st}/{v}"] = (o[0]["C"], o[1]["C"])
                want[f"{st}/{v}"] = e[v]["matched"]["energy_extrapolated"]
        checks, s_f = pyscf_checks(mol, perm, orbsets, nocc)
        radii = {
            "ferric": tuple(fb[common.z_of(s)] for s in symbols),
            "nwchem": tuple(NWCHEM_BSRAD[common.z_of(s) - 1] for s in symbols),
        }
        for name, c in checks.items():
            c["reference_energy"] = want[name]
            c["pyscf_minus_reference"] = c["pyscf_uhf_energy"] - want[name]
            bar = E_ANCHOR_MAX if name == "unconstrained" else E_STATE_MAX
            ca, cb = orbsets[name]
            dt = sum(dm_pair(perm, ca, cb, nocc))
            c["becke_population"] = {}
            for rname, rr in radii.items():
                n_frag, n_tot = becke_population(mol, dt, sd["fragment"][0], rr)
                c["becke_population"][f"{rname}_radii"] = n_frag
                c["becke_population"]["total_electrons_quadrature"] = n_tot
            bp = c["becke_population"]
            print(
                f"{system:8s} {name:24s} max|CtSC-I| {c['max_abs_CtSC_minus_I']:.2e}  "
                f"PySCF-NWChem {c['pyscf_minus_reference']:+.2e} (bar {bar:.0e})  "
                f"N_frag ferric-radii {bp['ferric_radii']:.7f} nwchem-radii "
                f"{bp['nwchem_radii']:.7f}",
                flush=True,
            )
            if c["max_abs_CtSC_minus_I"] > ORTHO_MAX:
                raise RuntimeError(
                    f"{system}/{name}: orbitals not orthonormal in ferric order"
                )
            if abs(c["pyscf_minus_reference"]) > bar:
                raise RuntimeError(
                    f"{system}/{name}: PySCF energy of NWChem orbitals misses"
                )

        def clean(block):
            out = {k: v for k, v in block.items() if not k.startswith("_")}
            out["alpha"] = orb_json(block["_orbs"][0])
            out["beta"] = orb_json(block["_orbs"][1])
            return out

        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": BASIS,
            "charge": sd["charge"],
            "multiplicity": sd["mult"],
            "method": "UHF (NWChem: xc hfexch 1.0 slater eps, eps -> 0 extrapolated)",
            "fragment": sd["fragment"],
            "target": sd["target"],
            "nwchem_cdft_line": cdft_line,
            "nuclear_repulsion": enuc,
            "nao": nao,
            "nocc": list(nocc),
            "ao_order": "ferric (NWChem rows multiplied by nwchem_to_ferric_ao_sign; "
            "verified, see generator doc)",
            "nwchem_to_ferric_ao_sign": signs,
            "partition": {
                "ferric_bragg_angstrom": dict(zip(symbols, radii["ferric"])),
                "nwchem_bsrad_angstrom": dict(zip(symbols, radii["nwchem"])),
                "ratio_ferric": radii["ferric"][0] / radii["ferric"][1],
                "ratio_nwchem": radii["nwchem"][0] / radii["nwchem"][1],
                "substitute_tags": tags,
                "unconstrained_energy_relabeled": unc_relabeled,
                "like_for_like_variant": variants[-1],
            },
            "grids": {
                g: {"radial": n, "nwchem_lebedev_index": i, "angular": a}
                for g, (n, i, a) in GRIDS.items()
            },
            "overlap_pyscf_ferric_order": mat(s_f),
            "unconstrained": {
                "energy": unc["energy"],
                "alpha": orb_json(unc_orbs[0]),
                "beta": orb_json(unc_orbs[1]),
            },
            "states": {
                st: {
                    k: (clean(v) if isinstance(v, dict) and "_orbs" in v else v)
                    for k, v in e.items()
                }
                for st, e in states.items()
            },
            "pyscf_checks": checks,
            "provenance": common.provenance(
                code="NWChem",
                version=version,
                keywords={
                    "module": "dft (odft)",
                    "xc": "hfexch (unconstrained); hfexch 1.0 slater eps (constrained), "
                    f"eps in {list(EPS)}",
                    "cdft": cdft_line + "; cdft_maxiter 200",
                    "grid": "lebedev <nrad> <iang> becke treutler; set dft:no_prune T",
                    "integrals": "direct, tolerances tight, no density fitting",
                    "partition": "ferric_partition variant: atoms retagged "
                    f"{tags} with per-atom nuclear charge restored",
                    "inputs": str(INPUT_DIR.relative_to(common.ROOT))
                    + f"/{system}_*.nw",
                    "pyscf_check": "PySCF UHF energy_tot + Becke populations of the "
                    "exported orbitals",
                },
                basis_name=BASIS,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid={
                    g: f"{n}x{a} Treutler(Perez-Jorda Chebyshev) x Lebedev, Becke "
                    "size-adjusted, unpruned"
                    for g, (n, _, a) in GRIDS.items()
                },
                aux=None,
                frozen_core=None,
                scf_conv={**gen_cdft.SCF_CONV, "cdft_multiplier": gen_cdft.CDFT_CONV},
                stability={
                    "note": "NWChem has no cDFT stability analysis; the Rust test "
                    "measures ferric's lambda-augmented internal stability at each "
                    "seeded state"
                },
                generator="scripts/validation/gen_cdft_state.py",
            ),
        }
        path = common.write_reference(ROW, system, BASIS, payload)
        size = path.stat().st_size
        if size > 500_000:
            raise RuntimeError(f"{path}: {size} bytes > 500 KB")
        written.append(path)
        print(f"wrote {path} ({size} bytes)", flush=True)
    print(f"GEN_CDFT_STATE_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
