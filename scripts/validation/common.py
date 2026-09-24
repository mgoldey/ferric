"""Like-for-like harness for ferric's validation tier (design §2.3).

Every external reference that a `crates/*/tests/validation_*.rs` test reads
is produced through this module, so the traps below are encoded ONCE:

* BASIS. The reference code is fed ferric's OWN bundled BSE JSON
  (`crates/ferric-core/src/basis/bundled/<name>.json`), never its built-in copy
  of the "same" basis. PySCF's built-in cc-pVDZ is segmented differently from
  ferric's general-contraction file, and def2 files ship Turbomole-raw
  coefficients (see `renormalize_contraction` below). `basis_sha256` goes into
  every provenance block so a reference is tied to the exact file.
* GEOMETRY. Coordinates are converted Angstrom -> Bohr with FERRIC's constant
  (`ANGSTROM_TO_BOHR` in `crates/ferric-core/src/mol.rs`, CODATA 2010
  0.52917721092) and handed to the reference code IN BOHR. Handing it Angstrom
  would let the reference code apply its own (possibly CODATA 2018) constant,
  a ~1e-9 relative geometry shift that is invisible at 1e-6 and visible at the
  1e-8 bars this tier uses. The nuclear repulsion energy is recorded so the
  Rust side can assert it first — a geometry/constant mismatch then fails as a
  geometry mismatch, not as a mystery 1e-7 energy error.
* STATE. Open-shell references are run from several guesses, each followed
  to a stable state with a `stability()` loop, and a reference whose final
  state is not internally stable is REFUSED (`RuntimeError`), never written.
* FITTING / FROZEN CORE / GRID. Recorded explicitly in provenance even when
  "none" — an absent key and "exact" must not be confusable.

Usage (the build agent runs these through `scripts/validation/run_slot.sh`):

    uv run --no-sync python scripts/validation/common.py --self-check
        Basis like-for-like self-check: builds each validation molecule with
        ferric's bundled 6-31G / def2-SVP in PySCF and asserts the contracted
        AO self-overlaps are 1 and the AO count equals ferric's.

This module imports PySCF lazily so that the ORCA helpers and the JSON/basis
readers work in an environment without PySCF.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLED_DIR = ROOT / "crates" / "ferric-core" / "src" / "basis" / "bundled"
REF_DIR = ROOT / "testdata" / "reference" / "validation"
MOL_DIR = ROOT / "testdata" / "molecules" / "validation"

# ferric's constant, copied from crates/ferric-core/src/mol.rs:
#     const ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_210_92;
BOHR_IN_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1.0 / BOHR_IN_ANGSTROM

# ORCA 6.1.1 on this box. NOT /usr/bin/orca, which is the GNOME screen reader.
ORCA_BINARY = (
    Path.home()
    / "Downloads"
    / "orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg"
    / "orca"
)

ELEMENTS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe",
]  # fmt: skip
SYMBOL_TO_Z = {s.upper(): z for z, s in enumerate(ELEMENTS) if z > 0}
ORCA_SHELL_LETTER = "SPDFGHIK"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def basis_json_path(name: str) -> Path:
    """Path of ferric's bundled BSE JSON for `name` (lower-case file stem)."""
    p = BUNDLED_DIR / f"{name.lower()}.json"
    if not p.is_file():
        raise FileNotFoundError(f"no bundled basis JSON {p}")
    return p


def z_of(symbol: str) -> int:
    try:
        return SYMBOL_TO_Z[symbol.upper()]
    except KeyError:
        raise ValueError(f"unknown element symbol {symbol!r}") from None


def git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------
# Basis: a faithful replica of ferric's `parse_bse_json`
# ---------------------------------------------------------------------------


def renormalize_contraction(
    exps: list[float], coefs: list[float], ell: int
) -> list[float]:
    """Replica of `renormalize_contraction` in crates/ferric-core/src/basis.rs.

    BSE coefficients multiply NORMALIZED primitives. The self-overlap of the
    contraction is then

        S = sum_pq c_p c_q (2 sqrt(a_p a_q) / (a_p + a_q))^(l + 3/2)

    and ferric divides every c_p by sqrt(S) so the contracted AO has unit norm.

    WHY PYSCF AGREES WITHOUT BEING TOLD. PySCF's `gto.M` also reads basis
    coefficients as normalized-primitive coefficients (it multiplies in
    `gto_norm(l, a)` itself) and then normalizes each contracted function
    (`_nomalize_contracted_ao`). Both codes therefore end at the same function
    up to a POSITIVE overall scale, and a normalized function has no remaining
    scale freedom — so feeding PySCF either the raw or the renormalized
    coefficients yields the identical AO. We feed the renormalized ones, so the
    numbers in PySCF's `mol._basis` are literally ferric's, and
    `check_basis_like_for_like` verifies the AO diagonal is 1.

    Scope: this argument is exact for s/p and for SPHERICAL l >= 2. For a
    CARTESIAN l >= 2 shell the two codes use different per-component
    conventions (libint2 normalizes the x^l component; PySCF/libcint applies a
    single radial factor), which rescales individual AOs — harmless for
    energies (the SCF energy is invariant to any per-AO scaling) but NOT for
    AO-basis matrices. `build_pyscf_mol` therefore refuses Cartesian l >= 2
    unless `allow_cartesian=True`, and the self-check skips those diagonals.
    """
    lf = float(ell)
    s = 0.0
    for a, ca in zip(exps, coefs):
        for b, cb in zip(exps, coefs):
            s += ca * cb * (2.0 * math.sqrt(a * b) / (a + b)) ** (lf + 1.5)
    if s > 0.0:
        scale = 1.0 / math.sqrt(s)
        return [c * scale for c in coefs]
    return list(coefs)


def ferric_shells(basis_name: str, z: int) -> list[dict]:
    """The shells ferric builds for element `z`, in ferric's order.

    Mirrors `parse_bse_json`: every coefficient COLUMN of an `electron_shells`
    entry becomes its own segmented shell (general contractions are split),
    multi-l (SP) entries give one shell per l, and `pure` is true only for
    `function_type == "gto_spherical"` AND l >= 2 ("gto" or absent means
    Cartesian). Returns dicts {l, pure, exps, coefs} with renormalized coefs.
    """
    data = json.loads(basis_json_path(basis_name).read_text())
    elem = data["elements"].get(str(z))
    if elem is None or "electron_shells" not in elem:
        raise KeyError(f"basis {basis_name} has no electron shells for Z={z}")
    out = []
    for sh in elem["electron_shells"]:
        exps = [float(x) for x in sh["exponents"]]
        pure_flag = sh.get("function_type") == "gto_spherical"
        ang = sh["angular_momentum"]
        cols = [[float(x) for x in c] for c in sh["coefficients"]]
        if len(ang) == 1:
            pairs = [(ang[0], col) for col in cols]
        else:
            if len(cols) < len(ang):
                raise ValueError(
                    f"{basis_name} Z={z}: shell missing a coefficient column"
                )
            pairs = list(zip(ang, cols))
        for ell, col in pairs:
            if len(col) != len(exps):
                raise ValueError(
                    f"{basis_name} Z={z}: {len(col)} coeffs vs {len(exps)} exps"
                )
            out.append(
                {
                    "l": ell,
                    "pure": pure_flag and ell >= 2,
                    "exps": exps,
                    "coefs": renormalize_contraction(exps, col, ell),
                }
            )
    return out


def ferric_nao(basis_name: str, symbols: list[str]) -> int:
    n = 0
    for s in symbols:
        for sh in ferric_shells(basis_name, z_of(s)):
            ell = sh["l"]
            n += (2 * ell + 1) if sh["pure"] else (ell + 1) * (ell + 2) // 2
    return n


def pyscf_basis(basis_name: str, symbols: list[str]) -> tuple[dict, bool | None]:
    """PySCF `mol.basis` dict built from ferric's JSON, plus the `cart` flag.

    Returns (basis, cart) where `cart` is True/False if the used elements carry
    any l >= 2 shell (all of which must agree — PySCF's `cart` is global), or
    None when there is no l >= 2 shell and the choice is immaterial.
    """
    basis = {}
    pure_flags = set()
    for s in dict.fromkeys(symbols):
        shells = []
        for sh in ferric_shells(basis_name, z_of(s)):
            if sh["l"] >= 2:
                pure_flags.add(sh["pure"])
            shells.append([sh["l"]] + [[e, c] for e, c in zip(sh["exps"], sh["coefs"])])
        basis[s] = shells
    if len(pure_flags) > 1:
        raise ValueError(
            f"{basis_name}: mixed spherical/Cartesian l>=2 shells across {sorted(set(symbols))}; "
            "PySCF's `cart` is global, so this cannot be matched like-for-like"
        )
    cart = None if not pure_flags else (not pure_flags.pop())
    return basis, cart


def pyscf_ecp(ecp_json: Path, symbols: list[str]) -> dict:
    """PySCF `mol.ecp` dict from a BSE-JSON ECP block (ferric's `def2-ecp.json`
    or an `-pp` basis file that carries `ecp_potentials`).

    Conventions (checked against ferric's `ecp.rs` and PySCF's NWChem parser):
      * ferric/BSE: the channel with the LARGEST angular momentum is the local
        U_L term; lower channels are the semilocal (U_l - U_L) projectors, the
        -U_L part already folded into their term lists. PySCF uses the same
        (U_l - U_L) form and tags the local channel l = -1.
      * `r_exponents[k]` is the NWChem/BSE `n` of r^(n-2); PySCF indexes the
        term lists by that same n (its NWChem parser appends to slot `n`).
    Returns {} for elements with no ECP.
    """
    data = json.loads(Path(ecp_json).read_text())
    out = {}
    for s in dict.fromkeys(symbols):
        elem = data["elements"].get(str(z_of(s)), {})
        pots = elem.get("ecp_potentials")
        ncore = elem.get("ecp_electrons")
        if not pots or ncore is None:
            continue
        lmax = max(p["angular_momentum"][0] for p in pots)
        channels = []
        for p in pots:
            (ell,) = p["angular_momentum"]
            (coefs,) = p["coefficients"]
            by_n: dict[int, list] = {}
            for n, g, c in zip(p["r_exponents"], p["gaussian_exponents"], coefs):
                by_n.setdefault(int(n), []).append([float(g), float(c)])
            nmax = max(by_n)
            terms = [by_n.get(n, []) for n in range(nmax + 1)]
            channels.append([-1 if ell == lmax else ell, terms])
        out[s] = [int(ncore), channels]
    return out


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def read_xyz(path: Path) -> tuple[list[str], list[list[float]]]:
    """Read an XYZ file (Angstrom) the way ferric's `Molecule::parse_xyz` does:
    count line, comment line, then `symbol x y z` rows. Returns
    (symbols, coords_in_BOHR) using ferric's conversion constant."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    symbols, coords = [], []
    for row in lines[2 : 2 + n]:
        tok = row.split()
        symbols.append(tok[0].capitalize())
        coords.append([float(v) * ANGSTROM_TO_BOHR for v in tok[1:4]])
    if len(symbols) != n:
        raise ValueError(f"{path}: header says {n} atoms, found {len(symbols)}")
    return symbols, coords


def pyscf_atom_bohr(symbols: list[str], coords_bohr: list[list[float]]) -> list:
    """PySCF `atom=` list; build the Mole with `unit="Bohr"`."""
    return [(s, tuple(c)) for s, c in zip(symbols, coords_bohr)]


def build_pyscf_mol(
    xyz_path: Path,
    basis_name: str,
    charge: int = 0,
    multiplicity: int = 1,
    ecp_json: Path | None = None,
    allow_cartesian: bool = False,
    verbose: int = 0,
):
    """A PySCF Mole with ferric's geometry (Bohr) and ferric's basis."""
    from pyscf import gto

    symbols, coords = read_xyz(xyz_path)
    basis, cart = pyscf_basis(basis_name, symbols)
    if cart and not allow_cartesian:
        raise ValueError(
            f"{basis_name} has Cartesian l>=2 shells; AO normalization differs between "
            "libint2 and libcint for those (energies still agree). Pass allow_cartesian=True "
            "only for energy-level comparisons."
        )
    mol = gto.M(
        atom=pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=basis,
        ecp=pyscf_ecp(ecp_json, symbols) if ecp_json else {},
        cart=bool(cart),
        charge=charge,
        spin=multiplicity - 1,
        verbose=verbose,
    )
    return mol


def check_basis_like_for_like(
    mol, basis_name: str, symbols: list[str], tol: float = 1e-12
) -> dict:
    """Self-check that `mol` carries ferric's basis. Raises AssertionError.

    1. AO count equals what ferric's parser builds (catches a dropped
       coefficient column / an SP entry split wrongly / cart-vs-pure).
    2. The Python replica of `renormalize_contraction` really gives unit
       self-overlap (catches a wrong exponent in the (l+3/2) power).
    3. PySCF's AO overlap diagonal is 1 for every s/p and spherical AO
       (catches a normalization convention mismatch between the two codes).
    Cartesian l>=2 diagonals are reported but not asserted (see
    `renormalize_contraction`'s docstring).
    """
    import numpy as np

    expected = ferric_nao(basis_name, symbols)
    assert mol.nao_nr() == expected, (
        f"{basis_name}: PySCF nao {mol.nao_nr()} != ferric {expected}"
    )

    worst_replica = 0.0
    for s in dict.fromkeys(symbols):
        for sh in ferric_shells(basis_name, z_of(s)):
            e, c, ell = sh["exps"], sh["coefs"], sh["l"]
            self_ov = sum(
                ca * cb * (2.0 * math.sqrt(a * b) / (a + b)) ** (ell + 1.5)
                for a, ca in zip(e, c)
                for b, cb in zip(e, c)
            )
            worst_replica = max(worst_replica, abs(self_ov - 1.0))
    assert worst_replica < tol, (
        f"{basis_name}: renormalized self-overlap off by {worst_replica:.2e}"
    )

    diag = np.diag(mol.intor("int1e_ovlp"))
    checked, worst_diag, skipped = 0, 0.0, 0
    off = 0
    for ib in range(mol.nbas):
        ell = mol.bas_angular(ib)
        nfn = (ell + 1) * (ell + 2) // 2 if mol.cart else 2 * ell + 1
        for _ in range(mol.bas_nctr(ib)):
            if mol.cart and ell >= 2:
                skipped += nfn
            else:
                worst_diag = max(
                    worst_diag, float(np.max(np.abs(diag[off : off + nfn] - 1.0)))
                )
                checked += nfn
            off += nfn
    assert off == mol.nao_nr()
    assert worst_diag < 1e-10, (
        f"{basis_name}: PySCF AO self-overlap off by {worst_diag:.2e}"
    )
    return {
        "nao": expected,
        "replica_self_overlap_max_dev": worst_replica,
        "pyscf_diag_max_dev": worst_diag,
        "diag_checked": checked,
        "diag_skipped_cartesian": skipped,
    }


# ---------------------------------------------------------------------------
# Provenance + JSON output
# ---------------------------------------------------------------------------


def provenance(
    *,
    code: str,
    version: str,
    keywords: dict | str,
    basis_name: str,
    xyz_path: Path,
    coords_bohr: list[list[float]],
    symbols: list[str],
    grid=None,
    aux=None,
    frozen_core=None,
    scf_conv=None,
    stability=None,
    ecp_json: Path | None = None,
    generator: str | None = None,
    extra: dict | None = None,
) -> dict:
    """The provenance block required by design §2.2.

    `None` for grid/aux/frozen_core is written as the explicit string "none"
    so that "not applicable / exact" cannot be confused with "forgot to say".
    """

    def _explicit(v):
        return "none" if v is None else v

    prov = {
        "code": code,
        "version": version,
        "keywords": keywords,
        "grid": _explicit(grid),
        "aux_basis": _explicit(aux),
        "frozen_core": _explicit(frozen_core),
        "scf_convergence": scf_conv,
        "stability": stability,
        "basis": basis_name,
        "basis_json": str(basis_json_path(basis_name).relative_to(ROOT)),
        "basis_sha256": sha256_file(basis_json_path(basis_name)),
        "ecp_json": str(Path(ecp_json).relative_to(ROOT)) if ecp_json else "none",
        "ecp_sha256": sha256_file(ecp_json) if ecp_json else "none",
        "geometry_xyz": str(Path(xyz_path).resolve().relative_to(ROOT)),
        "geometry_xyz_sha256": sha256_file(xyz_path),
        "geometry_bohr": [[s, *c] for s, c in zip(symbols, coords_bohr)],
        "bohr_in_angstrom": BOHR_IN_ANGSTROM,
        "generator": generator,
        "git_head": git_head(),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    if extra:
        prov.update(extra)
    return prov


def reference_path(row: str, system: str, basis_name: str) -> Path:
    return REF_DIR / row / f"{system}_{basis_name.lower()}.json"


def write_reference(row: str, system: str, basis_name: str, payload: dict) -> Path:
    """Write `testdata/reference/validation/<row>/<system>_<basis>.json`.

    Refuses a payload without a provenance block, and refuses any open-shell
    block whose recorded stability verdict is not stable.
    """
    if "provenance" not in payload:
        raise ValueError("refusing to write a reference with no provenance block")
    for key, block in payload.items():
        if isinstance(block, dict) and "stability" in block:
            st = block["stability"]
            if st is not None and st.get("internal_stable") is False:
                raise RuntimeError(
                    f"refusing to write {system}/{basis_name}: {key} state is UNSTABLE"
                )
    path = reference_path(row, system, basis_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Open-shell SCF with a stability loop (PySCF)
# ---------------------------------------------------------------------------


def _stability_status(mf):
    """(mo_internal, internal_stable) across PySCF's UHF/ROHF return shapes."""
    out = mf.stability(internal=True, external=False, return_status=True)
    # PySCF >= 2.1: (mo_i, mo_e, stable_i, stable_e)
    if isinstance(out, tuple) and len(out) == 4:
        mo_i, _mo_e, stable_i, _stable_e = out
        return mo_i, bool(stable_i)
    raise RuntimeError(
        f"unexpected PySCF stability() return shape: {type(out)} len={len(out)}"
    )


def uhf_lambda_min(mf) -> float:
    """Lowest eigenvalue of PySCF's own UHF orbital Hessian, by DENSE build.

    Same construction as scripts/gen_pyscf_hene_stability.py, whose number
    ferric's Davidson matched to 3.8e-10 on HeNe+ — i.e. the SAME Hessian
    convention as `ferric_scf::stability::StabilityResult::lowest_eigenvalue`.
    Dense on purpose: a Davidson can converge inside one symmetry block and
    miss the true minimum (see `StabilityConfig::n_block`). Dimension is
    nocc*nvir per spin, ~1300 for allyl/def2-SVP, i.e. light.
    """
    import numpy as np
    from pyscf.soscf import newton_ah

    out = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ)
    h_op, h_diag = out[-2], out[-1]
    n = len(h_diag)
    hess = np.empty((n, n))
    e = np.zeros(n)
    for i in range(n):
        e[:] = 0.0
        e[i] = 1.0
        hess[:, i] = h_op(e)
    hess = 0.5 * (hess + hess.T)
    return float(np.linalg.eigvalsh(hess)[0])


def run_open_shell(
    mol,
    method: str,
    conv_tol: float = 1e-11,
    conv_tol_grad: float = 1e-8,
    max_stab_rounds: int = 10,
    guesses: tuple[str, ...] = ("minao", "atom", "huckel"),
    distinct_tol: float = 1e-6,
) -> dict:
    """Converge `method` ("uhf" | "rohf") to an internally STABLE state.

    For each initial guess: converge, then loop { stability(); if unstable,
    restart from the unstable direction } until stable or `max_stab_rounds`.
    The lowest-energy stable solution over all guesses is returned. Raises
    RuntimeError if no guess reaches a stable, converged state — a reference
    on an unstable state is exactly the defect class (2026-09-17 open-shell
    guess fix) this tier exists to catch, so it is never written.

    Each guess's outcome is recorded in `guess_scan`, and
    `multiple_stable_minima` flags when two stable solutions differ by more
    than `distinct_tol` — a reviewer then knows the state is multi-valued and
    ferric landing on the higher one would be a state-selection difference,
    not an integral one.
    """
    from pyscf import scf

    cls = {"uhf": scf.UHF, "rohf": scf.ROHF}[method]
    scan = []
    best = None
    for guess in guesses:
        mf = cls(mol)
        mf.conv_tol = conv_tol
        mf.conv_tol_grad = conv_tol_grad
        mf.max_cycle = 500
        mf.init_guess = guess
        mf.verbose = 0
        mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            mf.kernel(mf.mo_coeff, mf.mo_occ)
        rounds, stable, trajectory = 0, False, [mf.e_tot]
        while True:
            if not mf.converged:
                break
            mo_i, stable = _stability_status(mf)
            if stable or rounds >= max_stab_rounds:
                break
            dm = mf.make_rdm1(mo_i, mf.mo_occ)
            mf.kernel(dm0=dm)
            rounds += 1
            trajectory.append(mf.e_tot)
        entry = {
            "guess": guess,
            "energy": float(mf.e_tot),
            "converged": bool(mf.converged),
            "internal_stable": bool(stable),
            "stability_rounds": rounds,
            "energy_trajectory": [float(x) for x in trajectory],
        }
        scan.append(entry)
        if mf.converged and stable and (best is None or mf.e_tot < best[0].e_tot):
            best = (mf, entry)
    if best is None:
        raise RuntimeError(
            f"{method}: no guess reached a converged, internally stable state: {scan}"
        )
    mf, entry = best
    stable_es = [g["energy"] for g in scan if g["converged"] and g["internal_stable"]]
    s2, mult = mf.spin_square()
    na, nb = mol.nelec
    result = {
        "energy": float(mf.e_tot),
        "converged": True,
        "s_squared": float(s2),
        "multiplicity_from_s2": float(mult),
        "nelec_alpha": int(na),
        "nelec_beta": int(nb),
        "stability": {
            "internal_stable": True,
            "kind": f"PySCF {method.upper()} internal (real)",
            "rounds": entry["stability_rounds"],
            "selected_guess": entry["guess"],
        },
        "guess_scan": scan,
        "multiple_stable_minima": (max(stable_es) - min(stable_es)) > distinct_tol,
    }
    if method == "uhf":
        ea, eb = mf.mo_energy
        result.update(
            {
                "homo_alpha": float(ea[na - 1]),
                "lumo_alpha": float(ea[na]),
                "homo_beta": float(eb[nb - 1]) if nb > 0 else None,
                "lumo_beta": float(eb[nb]),
                "somo_alpha": [float(x) for x in ea[nb:na]],
            }
        )
        result["stability"]["lambda_min"] = uhf_lambda_min(mf)
    else:
        e = mf.mo_energy
        result.update(
            {
                "orbital_energy_convention": (
                    "PySCF ROHF Roothaan effective-Fock eigenvalues; convention-dependent, "
                    "recorded for inspection, NOT comparable across codes"
                ),
                "somo_roothaan": [float(x) for x in e[nb:na]],
                "homo_docc_roothaan": float(e[nb - 1]) if nb > 0 else None,
                "lumo_roothaan": float(e[na]),
            }
        )
    return result


# ---------------------------------------------------------------------------
# ORCA: input writer + output parser
# ---------------------------------------------------------------------------
#
# The UHF/ROHF row (W0) does NOT use ORCA: PySCF with ferric's own basis is a
# complete like-for-like reference there. What follows is what W1 needs
# (RHF+ECP, RI-MP2 gradient, OO-RI-MP2): ferric's basis as `NewGTO` blocks,
# ferric's ECP as `NewECP` blocks (RHF+ECP row, gen_ecp_orca.py), a Bohr
# geometry, and a parser for the numbers those rows compare.


def orca_basis_block(
    basis_name: str, symbols: list[str], ecp_json: Path | None = None
) -> str:
    """`%basis NewGTO ... end end` carrying ferric's basis for every element.

    ORCA reads NewGTO coefficients as normalized-primitive coefficients and
    renormalizes the contraction — the same convention as ferric and PySCF,
    so the renormalized coefficients are emitted as-is. General contractions
    are emitted as ferric splits them (one segmented shell per column; same
    span). ORCA is spherical-only for l >= 2, so a Cartesian-d basis is
    refused rather than silently changed.

    With `ecp_json`, ferric's ECP for every element that has one is appended
    as `NewECP` entries inside the same `%basis` block (see `orca_ecp_block`).
    Without it ORCA assigns whatever ECP its keyword line implies (for a def2
    keyword: its BUILT-IN def2-ECP for Z >= 37) — a looser, not like-for-like,
    check that the caller must label as such.
    """
    lines = ["%basis"]
    for s in dict.fromkeys(symbols):
        lines.append(f"  NewGTO {s}")
        for sh in ferric_shells(basis_name, z_of(s)):
            if sh["l"] >= 2 and not sh["pure"]:
                raise ValueError(
                    f"{basis_name}: Cartesian l={sh['l']} shell; ORCA is spherical-only"
                )
            lines.append(f"    {ORCA_SHELL_LETTER[sh['l']]} {len(sh['exps'])}")
            for i, (e, c) in enumerate(zip(sh["exps"], sh["coefs"]), start=1):
                lines.append(f"      {i:3d} {e:.10E} {c:.10E}")
        lines.append("  end")
    if ecp_json is not None:
        ecp = orca_ecp_block(ecp_json, symbols)
        if ecp:
            lines.append(ecp)
    lines.append("end")
    return "\n".join(lines)


def ecp_core_electrons(ecp_json: Path, symbols: list[str]) -> list[int]:
    """Per-ATOM core-electron count ferric's `Molecule::apply_ecp` would set
    from `ecp_json` (0 for an element with no `ecp_potentials`)."""
    data = json.loads(Path(ecp_json).read_text())
    out = []
    for s in symbols:
        elem = data["elements"].get(str(z_of(s)), {})
        has = elem.get("ecp_potentials") and elem.get("ecp_electrons") is not None
        out.append(int(elem["ecp_electrons"]) if has else 0)
    return out


def orca_ecp_block(ecp_json: Path, symbols: list[str]) -> str:
    """`NewECP <El> ... end` entries carrying ferric's ECP, for inside `%basis`.

    Returns "" when no element in `symbols` has an ECP in `ecp_json`.

    Format (ORCA manual, "Effective Core Potentials"; identical to what the
    Basis Set Exchange's ORCA writer emits from the same BSE JSON):

        NewECP I
          N_core 28
          lmax f
          s 7
            1  <gaussian exponent>  <coefficient>  <n>
            ...
          f 4
            ...
        end

    Conventions, and why no number is transformed:
      * `<n>` is the BSE `r_exponents` value verbatim. ORCA, NWChem, PySCF and
        libecpint all read it as the power in r^(n-2) (the def2 ECPs are
        Gaussian-only, n = 2 throughout, i.e. r^0).
      * `lmax` is the LOCAL channel U_L (the largest angular momentum present,
        the same rule as ferric's `EcpDef::max_angular_momentum` and PySCF's
        l = -1 tag); the lower channels are the semilocal (U_l - U_L) terms, in
        the same form the BSE JSON already stores them.
      * Exponents and coefficients are written with `repr(float(...))`, which
        round-trips the IEEE double ferric parses from the same string.
    A like-for-like mismatch here (a channel mis-tagged, n shifted by 2) moves
    the energy by tenths of a Hartree or more, and ORCA's printed nuclear
    repulsion (which uses Z - N_core) is checked separately by the generator,
    so a wrong N_core cannot hide either.
    """
    data = json.loads(Path(ecp_json).read_text())
    blocks = []
    for s in dict.fromkeys(symbols):
        elem = data["elements"].get(str(z_of(s)), {})
        pots = elem.get("ecp_potentials")
        ncore = elem.get("ecp_electrons")
        if not pots or ncore is None:
            continue
        lmax = max(p["angular_momentum"][0] for p in pots)
        lines = [
            f"  NewECP {s}",
            f"    N_core {int(ncore)}",
            f"    lmax {ORCA_SHELL_LETTER[lmax].lower()}",
        ]
        # ORCA wants channels in increasing l with the local (lmax) one last;
        # ordering is cosmetic but makes the input diff-stable.
        for p in sorted(pots, key=lambda q: q["angular_momentum"][0]):
            (ell,) = p["angular_momentum"]
            (coefs,) = p["coefficients"]
            rexp, gexp = p["r_exponents"], p["gaussian_exponents"]
            if not (len(rexp) == len(gexp) == len(coefs)):
                raise ValueError(f"{ecp_json} {s} l={ell}: ragged ECP term lists")
            lines.append(f"    {ORCA_SHELL_LETTER[ell].lower()} {len(rexp)}")
            for i, (n, g, c) in enumerate(zip(rexp, gexp, coefs), start=1):
                lines.append(f"      {i:3d} {float(g)!r} {float(c)!r} {int(n)}")
        lines.append("  end")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def check_ecp_like_for_like(mol, ecp_json: Path, symbols: list[str]) -> dict:
    """Self-check that PySCF `mol` carries ferric's ECP. Raises AssertionError.

    1. Per-atom core-electron count: PySCF's `atom_nelec_core(i)` equals what
       ferric's `apply_ecp` sets from the same JSON (0 for all-electron atoms).
    2. Electron count: `mol.nelectron` = sum(Z) - sum(N_core) - charge.
    3. Term count: every ECP term in the JSON reached `mol._ecp` (catches a
       channel dropped by the l = -1 local-channel mapping, or a term list
       indexed by the wrong power of r and silently discarded).
    """
    want = ecp_core_electrons(ecp_json, symbols)
    got = [int(mol.atom_nelec_core(i)) for i in range(mol.natm)]
    assert got == want, f"ECP core electrons: PySCF {got} != ferric {want}"
    n_all = sum(z_of(s) for s in symbols)
    assert mol.nelectron == n_all - sum(want) - mol.charge, (
        f"nelectron {mol.nelectron} != {n_all} - {sum(want)} - {mol.charge}"
    )
    data = json.loads(Path(ecp_json).read_text())
    for s in dict.fromkeys(symbols):
        elem = data["elements"].get(str(z_of(s)), {})
        pots = elem.get("ecp_potentials") or []
        n_json = sum(len(p["r_exponents"]) for p in pots)
        n_pyscf = 0
        if s in mol._ecp:
            for _l, by_r in mol._ecp[s][1]:
                n_pyscf += sum(len(t) for t in by_r)
        assert n_json == n_pyscf, f"{s}: {n_json} ECP terms in JSON, {n_pyscf} in PySCF"
    return {"ecp_core_electrons": got, "nelectron": int(mol.nelectron)}


def write_orca_input(
    path: Path,
    keywords: str,
    xyz_path: Path,
    basis_name: str,
    charge: int = 0,
    multiplicity: int = 1,
    extra_blocks: str = "",
    nprocs: int = 1,
    ecp_json: Path | None = None,
) -> Path:
    """Write an ORCA input with ferric's geometry (in Bohr, `! Bohrs`) and
    ferric's basis (`NewGTO`). `keywords` must spell out the like-for-like
    choices (e.g. "UHF NoRI NoFrozenCore VeryTightSCF"); no default is
    supplied here on purpose — ORCA's defaults (RIJCOSX for hybrids, frozen
    core for MP2) are exactly the silent mismatches §2.3 lists. `ecp_json`
    adds ferric's ECP as `NewECP` entries (see `orca_basis_block`)."""
    symbols, coords = read_xyz(xyz_path)
    body = [
        f"! {keywords} Bohrs",
        f"%pal nprocs {nprocs} end",
        orca_basis_block(basis_name, symbols, ecp_json=ecp_json),
    ]
    if extra_blocks:
        body.append(extra_blocks)
    body.append(f"* xyz {charge} {multiplicity}")
    body += [
        f"  {s:<2} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(symbols, coords)
    ]
    body.append("*")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n")
    return path


_ORCA_PATTERNS = {
    "energy": re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)"),
    "version": re.compile(r"Program Version\s+(\S+)"),
    "s_squared": re.compile(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)"),
    "scf_iterations": re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES"),
    # ORCA prints V_nn with the ECP-reduced charges Z - N_core, so this is
    # the check that N_core reached ORCA (8 decimals printed).
    "nuclear_repulsion": re.compile(r"Nuclear Repulsion\s*:\s*(-?\d+\.\d+)\s*Eh"),
    "n_electrons": re.compile(r"Number of Electrons\s+NEL\s+\.+\s+(\d+)"),
}


def parse_orca_output(text: str) -> dict:
    """Numbers W1 needs from an ORCA .out. The LAST match wins (an optimization
    or a stability-restarted SCF prints several). A normal-termination check is
    mandatory: an aborted run can still print an energy."""
    out = {}
    for key, pat in _ORCA_PATTERNS.items():
        hits = pat.findall(text)
        if hits:
            out[key] = hits[-1] if key == "version" else float(hits[-1])
    out["normal_termination"] = "ORCA TERMINATED NORMALLY" in text
    out["scf_not_converged"] = "SCF NOT CONVERGED" in text
    return out


def run_orca(inp: Path, timeout: int | None = None) -> dict:
    """Run ORCA on `inp` (must be invoked by absolute path for its MPI
    launcher) and parse the output. Raises if ORCA did not terminate normally."""
    inp = Path(inp).resolve()
    if not ORCA_BINARY.is_file():
        raise FileNotFoundError(f"ORCA not found at {ORCA_BINARY}")
    res = subprocess.run(
        [str(ORCA_BINARY), inp.name],
        cwd=inp.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    inp.with_suffix(".out").write_text(res.stdout)
    parsed = parse_orca_output(res.stdout)
    if not parsed["normal_termination"] or parsed["scf_not_converged"]:
        raise RuntimeError(
            f"ORCA run {inp} failed (rc={res.returncode}); see {inp.with_suffix('.out')}"
        )
    return parsed


# ---------------------------------------------------------------------------
# Self-check entry point
# ---------------------------------------------------------------------------


def _self_check() -> int:
    xyzs = sorted(MOL_DIR.glob("*.xyz"))
    if not xyzs:
        print(f"no validation geometries under {MOL_DIR}", file=sys.stderr)
        return 1
    failures = 0
    for basis_name in ("6-31g", "def2-svp"):
        for xyz in xyzs:
            symbols, _ = read_xyz(xyz)
            # multiplicity only affects the electron count, not the basis check
            nelec = sum(z_of(s) for s in symbols)
            mol = build_pyscf_mol(xyz, basis_name, multiplicity=1 + nelec % 2)
            try:
                rep = check_basis_like_for_like(mol, basis_name, symbols)
                print(f"OK   {basis_name:9s} {xyz.stem:12s} {rep}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {basis_name:9s} {xyz.stem:12s} {e}")
    print(f"SELF_CHECK_DONE failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--self-check" in sys.argv[1:]:
        sys.exit(_self_check())
    print(__doc__)
