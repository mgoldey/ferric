"""References for the "OO-RI-MP2 energy" and "OO-MP2 orbital and nuclear
gradients" validation rows (VALIDATION.md 115 and 249-386).

Consumer: crates/ferric-mp2/tests/validation_oo_rimp2.rs
Output:   testdata/reference/validation/oo_rimp2/<system>_<basis>.json
ORCA inputs (committed): scripts/validation/orca/oo_rimp2/<system>_<basis>_{oo,rimp2}.inp

WHAT FERRIC COMPUTES (read from crates/ferric-mp2/src/oo_rimp2.rs,
u_oo_rimp2.rs and oo_rimp2_gradient.rs, not assumed):
`E(C) = E_HF(C) + E_MP2^RI(C)` minimised over occ-vir rotations, with the MP2
part evaluated in the SEMICANONICAL frame of C (active occ-occ and vir-vir
blocks of CᵀF(C)C diagonalised). E_HF(C) and F(C) use EXACT four-centre J/K
(`build_jk_with_pool`, no fitting); only the MP2 amplitudes/energy are
density-fitted, with the Coulomb metric. All electrons are correlated
(`frozen_core = 0`). That is the textbook full-Fock OMP2 functional with RI
in the correlation part only. The nuclear gradient (closed shell only;
`oo_ri_mp2_gradient`) is the z = 0 envelope gradient of that functional.

THE TIGHT ENERGY REFERENCE is an independent numpy OO-RI-MP2 (`numpy_oo`
block; see "Independent numpy OO-RI-MP2" below): the same functional written
directly on PySCF integrals (exact J/K, Coulomb-metric RI with ferric's aux),
minimised with scipy BFGS plus a gradient-only BFGS on a finite-difference
orbital gradient to max |dE/dkappa| <= 1e-10 (measured 5.7e-11 to 8.8e-11;
refused above 1e-8), anchored at kappa = 0 to PySCF
SCF + DF-MP2 (RHF 1e-12, UHF 1e-10). Its stored `max_orbital_gradient` is the
achieved convergence; the energy error is second order in it.

ORCA is a CROSS-CHECK. It stops ABOVE the minimum of this functional: on all
three systems ORCA's total is 3.7e-8 to 7.0e-8 Ha higher than the numpy
minimum (h2o 6.6e-8, nh3 7.0e-8, ch3 3.7e-8), with its reported ||g|| at
~3e-10, and its reference/doubles split differs by ~1.5e-5 (first order in
the orbital difference). `cross_check.orca_minus_numpy_{total,reference}`
record the measured offsets.

The ORCA cross-check is ORCA 6.1.1 `! OO-RI-MP2 NoRI NoFrozenCore
ExtremeSCF [UHF] EnGrad`, ferric's orbital basis as `NewGTO` and ferric's aux
basis as `NewAuxCGTO` (the /C slot), geometry in Bohr. `NoRI` makes ORCA's OO
Fock builds exact four-centre ("CLOSED-SHELL FOCK OPERATOR ... SHARK Fock
matrix driver" each orbital iteration); the MP2 part uses the AuxC basis
(printed "Dimension of the AuxC basis", checked against PySCF's count). ORCA
also forms semicanonical orbitals each iteration, i.e. the same functional.
ORCA's `FINAL SINGLE POINT ENERGY` is `Total Energy(D)` (reference + doubles);
the perturbative-singles `Total Energy(SD)` it also prints is NOT ferric's
functional and is stored only for the record.

ORCA's own convergence (the SCF settings feed the OO loop): ExtremeSCF with
`%scf TolE 1e-12 end` (see OO_SCF_BLOCK), orbital gradient 1e-9; the generator
refuses a run whose final ||g|| exceeds 1e-8.

Also generated, all as independent anchors/controls:
  * ORCA plain `RI-MP2 NoRI NoFrozenCore ExtremeSCF [UHF]` at the SAME
    geometry/bases: the SCF and non-OO RI-MP2 energies. These anchor ferric's
    RHF/UHF and plain RI-MP2 (basis + aux like-for-like, independent of the
    OO machinery) and are the NEGATIVE CONTROL: plain RI-MP2 must miss the
    OO total by >> the bar.
  * PySCF 2.13.1 exact-J/K SCF (RHF; UHF through `common.run_open_shell`,
    stability-checked) with ferric's basis: must agree with ORCA's SCF to
    1e-8, and (closed shell) PySCF DF-MP2 E_corr with ferric's aux must agree
    with ORCA's RI-MP2 E_corr to 1e-8 (the check that ORCA used ferric's aux).
  * Closed shell only: a 5-point central finite difference (h = H_FD Bohr) of
    ORCA's OO-RI-MP2 TOTAL energy, every Cartesian coordinate, each point a
    fully re-converged OO run. This FD rests only on ORCA's OO energy, which
    is itself not at the minimum (above), so it is a LOOSE gradient check;
    the tight nuclear-gradient anchor is ferric's analytic gradient against a
    finite difference of ferric's own OO energy (the `*_self_fd` tests). ORCA's
    ANALYTIC OO-RI-MP2 gradient is stored too, but it is NOT exact for ORCA's
    own energy: on distorted H2O/cc-pVDZ it misses the FD by up to 8.0e-6
    Ha/Bohr (H1 y), while the FD itself is step-converged (h = 1e-3, 2e-3,
    4e-3 give 0.0143345329, ...5329, ...5322: spread 7.5e-10) and ORCA's
    orbital gradient at convergence is ~1e-12. Both gradients are translation
    invariant (columns sum to zero), so the analytic miss is a real
    approximation/defect in ORCA's OO gradient, not an FD artifact. The
    generator therefore gates the analytic gradient only against a gross
    ORCA_ANALYTIC_GROSS_TOL and records the measured miss in `cross_check`.
    There is no PySCF OMP2 (pyscf.mp has no orbital-optimised MP2 in 2.13.1).

Run (light: each ORCA job is ~5 s on 1 core; the FD is 36/48 jobs):
    scripts/validation/run_slot.sh --light -- \\
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_oo_rimp2.py [system ...]
`--write-only` regenerates the committed .inp files without running anything;
`--no-fd` skips the ORCA finite difference (for a quick look; refuses to write).
`--numpy-only` runs no ORCA: it loads each existing JSON, recomputes the
`numpy_oo` block and `cross_check.orca_minus_numpy_*`, and rewrites the file
(measured: H2O 80 s alone, NH3 ~3.5 min and CH3 ~6.5 min run two at a time).
ORCA runs in temporary directories (copies of the .inp).
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from gen_rimp2_gradient import orca_aux_c_lines, parse_engrad, pyscf_dfmp2  # noqa: E402

ROW = "oo_rimp2"
ROW_NAME = "OO-RI-MP2 energy; OO-MP2 orbital and nuclear gradients"
INP_DIR = Path(__file__).resolve().parent / "orca" / ROW

# system -> (orbital basis, aux basis, charge, multiplicity, nuclear gradient?)
# CH3 has no nuclear gradient: ferric has no U-OO-RI-MP2 nuclear gradient.
SYSTEMS = {
    "h2o_distorted": ("cc-pvdz", "cc-pvdz-ri", 0, 1, True),
    "nh3_distorted": ("cc-pvdz", "cc-pvdz-ri", 0, 1, True),
    "ch3": ("cc-pvdz", "cc-pvdz-ri", 0, 2, False),
}

OO_KEYWORDS = "OO-RI-MP2 NoRI NoFrozenCore ExtremeSCF{uhf}{engrad}"
MP2_KEYWORDS = "RI-MP2 NoRI NoFrozenCore ExtremeSCF{uhf}"
# ExtremeSCF's TolE 1e-14 is passed to ORCA's OO loop, where Delta-E then
# bounces at machine precision for a ~-56 Ha energy (a displaced NH3 point
# failed to report convergence). TolE 1e-12 keeps TolG 1e-9 (the orbital
# gradient bar; energy error ~||g||^2) and the ExtremeSCF integral thresholds.
OO_SCF_BLOCK = "%scf TolE 1e-12 end"
H_FD = 2e-3  # Bohr, 5-point stencil
STENCIL = [(-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12)]
TOL_SCF = 1e-8
TOL_ECORR = 1e-8
MAX_OO_GNORM = 1e-8
ORCA_ANALYTIC_GROSS_TOL = 1e-4


# ---------------------------------------------------------------------------
# ORCA
# ---------------------------------------------------------------------------


def input_text(
    keywords: str, symbols, coords, basis: str, aux: str, charge: int, mult: int
) -> str:
    basis_block = common.orca_basis_block(basis, symbols).splitlines()
    assert basis_block[-1] == "end"
    basis_block = basis_block[:-1] + orca_aux_c_lines(aux, symbols) + ["end"]
    body = [f"! {keywords} Bohrs", "%pal nprocs 1 end"]
    if keywords.startswith("OO-"):
        body.append(OO_SCF_BLOCK)
    body += basis_block
    body.append(f"* xyz {charge} {mult}")
    body += [
        f"  {s:<2} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(symbols, coords)
    ]
    body.append("*")
    return "\n".join(body) + "\n"


def _last(pat: str, text: str, cast=float):
    m = re.findall(pat, text)
    return cast(m[-1]) if m else None


def parse_oo(text: str) -> dict:
    """Numbers from ORCA's OO-RI-MP2 block (last occurrence = final)."""
    out = {
        "e_reference": _last(r"Reference Energy\s*=\s*(-?\d+\.\d+)\s*Eh", text),
        "e_doubles": _last(r"Doubles Energy\s*=\s*(-?\d+\.\d+)\s*Eh", text),
        "e_singles_perturbative": _last(
            r"Singles Energy\s*=\s*(-?\d+\.\d+)\s*Eh", text
        ),
        "e_total_d_printed": _last(r"Total Energy\(D\)\s*=\s*(-?\d+\.\d+)\s*Eh", text),
        "e_total_sd_printed": _last(
            r"Total Energy\(SD\)\s*=\s*(-?\d+\.\d+)\s*Eh", text
        ),
        "final_orbital_gnorm": _last(r"\|\|g\|\|\s*=\s*(\S+)", text),
        "orbital_iterations": _last(r"ORBITAL ITERATION\s+(\d+)", text, int),
        "oo_converged": "CONVERGENCE REACHED" in text,
        "nao": _last(r"Dimension of the orbital basis\s*\.+\s*(\d+)", text, int),
        "naux": _last(r"Dimension of the AuxC basis\s*\.+\s*(\d+)", text, int),
        "exact_fock": "SHARK Fock matrix driver" in text,
    }
    return out


def parse_scf_mp2(text: str) -> dict:
    out = {}
    m = re.findall(r"Total Energy\s*:\s*(-?\d+\.\d+)\s*Eh", text)
    if m:
        out["e_scf"] = float(m[0])  # first "Total Energy :" = SCF block
    out["e_corr_printed"] = _last(r"RI-MP2 CORRELATION ENERGY:\s*(-?\d+\.\d+)", text)
    out["nao"] = _last(r"Dimension of the orbital basis\s*\.+\s*(\d+)", text, int)
    out["naux"] = _last(r"Dimension of the AuxC basis\s*\.+\s*(\d+)", text, int)
    return out


def run_orca_text(text: str, name: str, natm: int | None = None) -> dict:
    """Run ORCA on an input text in a temporary directory; parse the lot."""
    with tempfile.TemporaryDirectory(prefix="ferric-orca-oo-") as tmp:
        inp = Path(tmp) / f"{name}.inp"
        inp.write_text(text)
        parsed = common.run_orca(inp)
        out_text = inp.with_suffix(".out").read_text()
        parsed["_out"] = out_text
        if natm is not None:
            parsed["gradient"] = parse_engrad(
                inp.with_suffix(".engrad").read_text(), natm
            )
    return parsed


def run_oo(text: str, name: str, natm: int | None = None) -> dict:
    r = run_orca_text(text, name, natm)
    oo = parse_oo(r["_out"])
    if not oo["oo_converged"] or oo["final_orbital_gnorm"] is None:
        keep = Path(tempfile.gettempdir()) / f"{name}_not_converged.out"
        keep.write_text(r["_out"])
        print(f"{name}: kept {keep}")
        raise RuntimeError(f"{name}: ORCA OO-RI-MP2 did not report convergence")
    if oo["final_orbital_gnorm"] > MAX_OO_GNORM:
        raise RuntimeError(f"{name}: ORCA OO ||g|| {oo['final_orbital_gnorm']:.1e}")
    if not oo["exact_fock"]:
        raise RuntimeError(
            f"{name}: ORCA OO Fock builds are not the exact SHARK driver"
        )
    # FINAL SINGLE POINT ENERGY (12 decimals) must be Total Energy(D).
    if abs(r["energy"] - oo["e_total_d_printed"]) > 2e-9:
        raise RuntimeError(f"{name}: FINAL SINGLE POINT ENERGY != Total Energy(D)")
    oo.update({k: v for k, v in r.items() if k != "_out"})
    return oo


# ---------------------------------------------------------------------------
# Independent numpy OO-RI-MP2 (the tight energy reference)
# ---------------------------------------------------------------------------
#
# The functional of the module docstring, written directly in numpy on PySCF
# integrals: exact four-centre (mn|ls) for J/K, and the Coulomb-metric fitted
# B^P_mn = sum_Q (L^-1)_PQ (Q|mn), L L^T = (P|Q), with ferric's aux, for the
# doubles. The orbitals are C(kappa) = C_SCF expm(K), K antisymmetric with
# only the vir-occ block free (per spin for UHF). E(kappa) = E_ref(C) +
# E_doubles(C), E_doubles in the semicanonical frame of C. The gradient is a
# 5-point central difference in kappa (no analytic OO derivative anywhere, so
# nothing is shared with ferric's or ORCA's orbital-gradient code); scipy BFGS
# minimises E. Anchor: at kappa = 0 the energy equals PySCF SCF + DF-MP2
# (DFMP2 / DFUMP2 with the same auxmol), asserted below.

NUMPY_FD_STEP = 1e-3  # kappa step; truncation ~h^4, round-off ~1e-11
NUMPY_GTOL = 1e-10  # BFGS target on max |dE/dkappa|
NUMPY_MAX_GRAD = 1e-8  # refuse a result whose final max |dE/dkappa| exceeds this
TOL_ANCHOR_RHF = 1e-12
TOL_ANCHOR_UHF = 1e-10


def _fitted_integrals(mol, aux: str, symbols):
    """(eri_J, eri_K, B): (mn|ls) as n^2 x n^2 matrices for J and K, and the
    Coulomb-metric fitted 3-index B[P, m, n]; also the auxmol."""
    import scipy.linalg as sla
    from pyscf import df

    n = mol.nao_nr()
    eri = mol.intor("int2e", aosym="s1").reshape(n, n, n, n)
    eri_j = eri.reshape(n * n, n * n)
    # K_mn = sum_ls (ml|ns) D_ls: reorder to [m, n, l, s].
    eri_k = np.ascontiguousarray(eri.transpose(0, 2, 1, 3)).reshape(n * n, n * n)
    aux_bas, aux_cart = common.pyscf_basis(aux, symbols)
    assert not aux_cart
    auxmol = df.addons.make_auxmol(mol, aux_bas)
    auxmol.cart = False
    auxmol.build()
    int3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")  # (n, n, naux)
    naux = int3.shape[2]
    low = np.linalg.cholesky(auxmol.intor("int2c2e"))
    b = sla.solve_triangular(low, int3.reshape(n * n, naux).T, lower=True)
    return eri_j, eri_k, b.reshape(naux, n, n), auxmol


def _rotate(c, kappa, nocc):
    import scipy.linalg as sla

    nmo = c.shape[1]
    k = np.zeros((nmo, nmo))
    k[nocc:, :nocc] = kappa.reshape(nmo - nocc, nocc)
    k -= k.T
    return c @ sla.expm(k)


def _semicanonical_b(b, f, co, cv):
    """Occ/vir orbital energies of the semicanonical frame and B[P, i, a]."""
    eo, uo = np.linalg.eigh(co.T @ f @ co)
    ev, uv = np.linalg.eigh(cv.T @ f @ cv)
    co, cv = co @ uo, cv @ uv
    return eo, ev, np.matmul(np.matmul(co.T, b), cv)


def _pair(ba, bb):
    """(ia|jb) as [i, a, j, b] from B[P, i, a] and B[P, j, b]."""
    naux, no1, nv1 = ba.shape
    _, no2, nv2 = bb.shape
    g = ba.reshape(naux, -1).T @ bb.reshape(naux, -1)
    return g.reshape(no1, nv1, no2, nv2)


def _denom(eo1, ev1, eo2, ev2):
    return (
        eo1[:, None, None, None]
        - ev1[None, :, None, None]
        + eo2[None, None, :, None]
        - ev2[None, None, None, :]
    )


def _fd_gradient(energy, x, h=NUMPY_FD_STEP):
    g = np.zeros_like(x)
    for k in range(len(x)):
        e = np.zeros_like(x)
        e[k] = h
        g[k] = (
            -energy(x + 2 * e)
            + 8 * energy(x + e)
            - 8 * energy(x - e)
            + energy(x - 2 * e)
        ) / (12 * h)
    return g


def _minimise(energy, n):
    import scipy.optimize as so

    x0 = np.zeros(n)
    res = so.minimize(
        energy,
        x0,
        jac=lambda x: _fd_gradient(energy, x),
        method="BFGS",
        options={"gtol": NUMPY_GTOL, "norm": np.inf, "maxiter": 2000},
    )
    # scipy's BFGS stops ("precision loss") once the energy decrease a line
    # search must see, ~g^2/H, falls under the energy round-off (~3e-14 Ha);
    # the gradient is then still ~1e-7. Continue with a GRADIENT-ONLY BFGS:
    # the line search is a secant on the directional derivative g(x + t d).d,
    # which the FD gradient resolves to ~1e-10 where the energy cannot.
    x = res.x.copy()
    hinv = np.array(res.hess_inv)
    g = _fd_gradient(energy, x)
    best = (float(np.abs(g).max()), x, g)
    polish = stall = 0
    eye = np.eye(n)
    while best[0] > NUMPY_GTOL and polish < 200 and stall < 8:
        d = -hinv @ g
        dphi0 = float(g @ d)
        if dphi0 >= 0:  # not a descent direction: restart from steepest descent
            hinv = eye * (np.abs(d).max() / max(np.abs(g).max(), 1e-300))
            d = -hinv @ g
            dphi0 = float(g @ d)
        t = 1.0
        g_t = _fd_gradient(energy, x + d)
        dphi1 = float(g_t @ d)
        if abs(dphi1) > 0.5 * abs(dphi0) and dphi0 != dphi1:
            t = dphi0 / (dphi0 - dphi1)  # secant zero of the directional derivative
            t = min(max(t, 0.05), 5.0)
            g_t = _fd_gradient(energy, x + t * d)
        s_vec = t * d
        y = g_t - g
        sy = float(s_vec @ y)
        x, g = x + s_vec, g_t
        if sy > 0:
            rho = 1.0 / sy
            hinv = (eye - rho * np.outer(s_vec, y)) @ hinv @ (
                eye - rho * np.outer(y, s_vec)
            ) + rho * np.outer(s_vec, s_vec)
        polish += 1
        gmax_now = float(np.abs(g).max())
        if gmax_now < best[0]:
            best, stall = (gmax_now, x.copy(), g.copy()), 0
        else:
            stall += 1
    gmax, x, g = best
    if gmax > NUMPY_MAX_GRAD:
        raise RuntimeError(
            f"numpy OO-RI-MP2 not converged: max |dE/dkappa| {gmax:.2e} "
            f"({res.message}, {res.nit} BFGS it + {polish} gradient-only it)"
        )
    return (
        x,
        gmax,
        int(res.nit) + polish,
        (f"scipy BFGS {res.nit} it ({res.message}) + {polish} gradient-only BFGS it"),
    )


def numpy_oo_rhf(symbols, coords, basis: str, aux: str) -> dict:
    """Closed-shell numpy OO-RI-MP2 from a PySCF exact-J/K RHF."""
    e_rhf, e_corr_df, _dm, mf, _naux = pyscf_dfmp2(symbols, coords, basis, aux)
    mol = mf.mol
    eri_j, eri_k, b, _auxmol = _fitted_integrals(mol, aux, symbols)
    h = mf.get_hcore()
    enuc = mol.energy_nuc()
    c0 = mf.mo_coeff
    nocc = mol.nelectron // 2
    nvir = c0.shape[1] - nocc
    n = c0.shape[0]

    def parts(kappa):
        c = _rotate(c0, kappa, nocc)
        co, cv = c[:, :nocc], c[:, nocc:]
        d = 2.0 * co @ co.T
        j = (eri_j @ d.ravel()).reshape(n, n)
        k = (eri_k @ d.ravel()).reshape(n, n)
        f = h + j - 0.5 * k
        e_ref = 0.5 * np.sum(d * (h + f)) + enuc
        eo, ev, bia = _semicanonical_b(b, f, co, cv)
        g = _pair(bia, bia)
        e_d = np.sum(g * (2.0 * g - g.transpose(0, 3, 2, 1)) / _denom(eo, ev, eo, ev))
        return e_ref, e_d

    def energy(kappa):
        e_ref, e_d = parts(kappa)
        return e_ref + e_d

    anchor = float(energy(np.zeros(nvir * nocc)) - (e_rhf + e_corr_df))
    if abs(anchor) > TOL_ANCHOR_RHF:
        raise RuntimeError(f"numpy OO kappa=0 vs PySCF RHF+DFMP2: {anchor:.2e}")
    x, gmax, nit, msg = _minimise(energy, nvir * nocc)
    e_ref, e_d = parts(x)
    return {
        "e_total": float(e_ref + e_d),
        "e_reference": float(e_ref),
        "e_doubles": float(e_d),
        "max_orbital_gradient": gmax,
        "iterations": nit,
        "optimizer_message": msg,
        "anchor_kappa0_diff": anchor,
        "e_scf_start": float(e_rhf),
        "method": (
            "numpy OO-RI-MP2 (RHF): E(kappa) = E_RHF(C) + E_MP2^RI(C), "
            "C = C_RHF expm(K), K antisymmetric vir-occ; exact four-centre J/K from "
            "PySCF int2e; doubles in the semicanonical frame with Coulomb-metric "
            "fitted B from PySCF int3c2e/int2c2e and ferric's aux; scipy BFGS on a "
            f"5-point central-difference gradient (h = {NUMPY_FD_STEP}); anchor at "
            "kappa = 0: PySCF scf.RHF + mp.dfmp2.DFMP2 with the same auxmol"
        ),
    }


def numpy_oo_uhf(symbols, coords, basis: str, aux: str, charge: int, mult: int) -> dict:
    """Spin-unrestricted numpy OO-RI-MP2 from a stability-checked PySCF UHF."""
    from pyscf import df, gto
    from pyscf.mp import dfump2

    bas, cart = common.pyscf_basis(basis, symbols)
    assert not cart
    mol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=bas,
        cart=False,
        charge=charge,
        spin=mult - 1,
        verbose=0,
    )
    res, mf = common.run_open_shell(
        mol, "uhf", conv_tol=1e-12, conv_tol_grad=1e-9, return_mf=True
    )
    eri_j, eri_k, b, auxmol = _fitted_integrals(mol, aux, symbols)
    mp = dfump2.DFUMP2(mf)
    mp.with_df = df.DF(mol)
    mp.with_df.auxmol = auxmol
    mp.kernel()
    e_corr_df = mp.e_corr

    h = mf.get_hcore()
    enuc = mol.energy_nuc()
    ca0, cb0 = mf.mo_coeff
    noa, nob = mol.nelec
    n = ca0.shape[0]
    nva, nvb = ca0.shape[1] - noa, cb0.shape[1] - nob
    na = nva * noa

    def parts(kappa):
        ca = _rotate(ca0, kappa[:na], noa)
        cb = _rotate(cb0, kappa[na:], nob)
        cao, cav, cbo, cbv = ca[:, :noa], ca[:, noa:], cb[:, :nob], cb[:, nob:]
        da, db = cao @ cao.T, cbo @ cbo.T
        j = (eri_j @ (da + db).ravel()).reshape(n, n)
        fa = h + j - (eri_k @ da.ravel()).reshape(n, n)
        fb = h + j - (eri_k @ db.ravel()).reshape(n, n)
        e_ref = 0.5 * (np.sum(da * (h + fa)) + np.sum(db * (h + fb))) + enuc
        eoa, eva, ba = _semicanonical_b(b, fa, cao, cav)
        eob, evb, bb = _semicanonical_b(b, fb, cbo, cbv)
        e_d = 0.0
        for bx, eo, ev in ((ba, eoa, eva), (bb, eob, evb)):
            g = _pair(bx, bx)  # same spin: 1/2 sum (ia|jb)[(ia|jb) - (ib|ja)] / D
            e_d += 0.5 * np.sum(
                g * (g - g.transpose(0, 3, 2, 1)) / _denom(eo, ev, eo, ev)
            )
        g = _pair(ba, bb)  # opposite spin
        e_d += np.sum(g * g / _denom(eoa, eva, eob, evb))
        return e_ref, e_d

    def energy(kappa):
        e_ref, e_d = parts(kappa)
        return e_ref + e_d

    anchor = float(energy(np.zeros(na + nvb * nob)) - (mf.e_tot + e_corr_df))
    if abs(anchor) > TOL_ANCHOR_UHF:
        raise RuntimeError(f"numpy U-OO kappa=0 vs PySCF UHF+DFUMP2: {anchor:.2e}")
    x, gmax, nit, msg = _minimise(energy, na + nvb * nob)
    e_ref, e_d = parts(x)
    return {
        "e_total": float(e_ref + e_d),
        "e_reference": float(e_ref),
        "e_doubles": float(e_d),
        "max_orbital_gradient": gmax,
        "iterations": nit,
        "optimizer_message": msg,
        "anchor_kappa0_diff": anchor,
        "e_scf_start": float(mf.e_tot),
        "stability": res["stability"],
        "method": (
            "numpy OO-RI-MP2 (UHF): E(kappa_a, kappa_b) = E_UHF(C_a, C_b) + "
            "E_UMP2^RI(C_a, C_b), C_s = C_UHF,s expm(K_s), independent antisymmetric "
            "vir-occ K_a, K_b; exact four-centre J/K from PySCF int2e; doubles = "
            "same-spin aa + bb (1/2 sum (ia|jb)[(ia|jb)-(ib|ja)]/D) + opposite-spin ab "
            "(sum (ia|jb)^2/D) in each spin's semicanonical frame, Coulomb-metric "
            "fitted B from PySCF int3c2e/int2c2e and ferric's aux; scipy BFGS on a "
            f"5-point central-difference gradient (h = {NUMPY_FD_STEP}); start: "
            "common.run_open_shell('uhf') stability-checked UHF; anchor at kappa = 0: "
            "PySCF UHF + mp.dfump2.DFUMP2 with the same auxmol"
        ),
    }


def numpy_oo_block(symbols, coords, basis, aux, charge, mult) -> dict:
    if mult == 1:
        if charge != 0:
            raise ValueError("numpy_oo_rhf builds a neutral PySCF mol (pyscf_dfmp2)")
        return numpy_oo_rhf(symbols, coords, basis, aux)
    return numpy_oo_uhf(symbols, coords, basis, aux, charge, mult)


def attach_numpy(payload: dict, numpy_oo: dict) -> None:
    """Add the numpy block and the ORCA-minus-numpy offsets to a payload."""
    payload["numpy_oo"] = numpy_oo
    cross = payload.setdefault("cross_check", {})
    cross["orca_minus_numpy_total"] = (
        payload["orca_oo"]["e_total"] - numpy_oo["e_total"]
    )
    cross["orca_minus_numpy_reference"] = (
        payload["orca_oo"]["e_reference"] - numpy_oo["e_reference"]
    )
    payload["provenance"]["keywords"]["numpy_oo"] = numpy_oo["method"]


def refresh_numpy(system, basis, aux, charge, mult, symbols, coords) -> Path:
    """`--numpy-only`: add/refresh `numpy_oo` in the existing JSON (no ORCA)."""
    import json

    path = common.reference_path(ROW, system, basis)
    payload = json.loads(path.read_text())
    stored = [row[1:] for row in payload["provenance"]["geometry_bohr"]]
    if np.abs(np.array(stored) - np.array(coords)).max() > 1e-12:
        raise RuntimeError(f"{system}: xyz geometry differs from the stored reference")
    block = numpy_oo_block(symbols, coords, basis, aux, charge, mult)
    d_scf = abs(block["e_scf_start"] - payload["pyscf"]["e_scf"])
    if d_scf > 1e-9:
        raise RuntimeError(f"{system}: numpy start SCF vs stored PySCF SCF {d_scf:.2e}")
    attach_numpy(payload, block)
    common.write_reference(ROW, system, basis, payload)
    c = payload["cross_check"]
    print(
        f"{system:14s} numpy E_OO {block['e_total']:.12f} (ref {block['e_reference']:.12f} "
        f"dbl {block['e_doubles']:.12f}, max|g| {block['max_orbital_gradient']:.1e}, "
        f"{block['iterations']} it, anchor {block['anchor_kappa0_diff']:.1e}) "
        f"ORCA-numpy total {c['orca_minus_numpy_total']:+.3e} "
        f"ref {c['orca_minus_numpy_reference']:+.3e}"
    )
    return path


# ---------------------------------------------------------------------------


def main() -> int:
    args = sys.argv[1:]
    write_only = "--write-only" in args
    no_fd = "--no-fd" in args
    numpy_only = "--numpy-only" in args
    only = {a for a in args if not a.startswith("--")}
    written = []
    for system, (basis, aux, charge, mult, want_grad) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        natm = len(symbols)
        if numpy_only:
            written.append(
                refresh_numpy(system, basis, aux, charge, mult, symbols, coords)
            )
            continue
        uhf = " UHF" if mult > 1 else ""
        oo_kw = OO_KEYWORDS.format(uhf=uhf, engrad=" EnGrad" if want_grad else "")
        mp2_kw = MP2_KEYWORDS.format(uhf=uhf)
        oo_text = input_text(oo_kw, symbols, coords, basis, aux, charge, mult)
        mp2_text = input_text(mp2_kw, symbols, coords, basis, aux, charge, mult)
        INP_DIR.mkdir(parents=True, exist_ok=True)
        inp_oo = INP_DIR / f"{system}_{basis}_oo.inp"
        inp_mp2 = INP_DIR / f"{system}_{basis}_rimp2.inp"
        inp_oo.write_text(oo_text)
        inp_mp2.write_text(mp2_text)
        if write_only:
            print(f"wrote {inp_oo.relative_to(common.ROOT)} (+ _rimp2)")
            continue

        import pyscf
        from pyscf import gto

        oo = run_oo(oo_text, f"{system}_oo", natm if want_grad else None)
        mp2 = run_orca_text(mp2_text, f"{system}_rimp2")
        mp2.update(parse_scf_mp2(mp2.pop("_out")))
        # Precise E_corr = FINAL - E_scf (both 12+ decimals); check the print.
        mp2["e_corr"] = mp2["energy"] - mp2["e_scf"]
        if abs(mp2["e_corr"] - mp2["e_corr_printed"]) > 2e-9:
            raise RuntimeError(f"{system}: RI-MP2 E_total - E_scf != printed E_corr")

        # ---- PySCF SCF anchor (exact J/K, ferric's basis) ----
        bas, cart = common.pyscf_basis(basis, symbols)
        assert not cart
        mol = gto.M(
            atom=common.pyscf_atom_bohr(symbols, coords),
            unit="Bohr",
            basis=bas,
            cart=False,
            charge=charge,
            spin=mult - 1,
            verbose=0,
        )
        stability = None
        pyscf_block = {}
        if mult == 1:
            e_rhf, e_corr_df, _dm, _mf, naux = pyscf_dfmp2(symbols, coords, basis, aux)
            pyscf_block = {"e_scf": float(e_rhf), "e_corr_dfmp2": float(e_corr_df)}
            d_corr = abs(e_corr_df - mp2["e_corr"])
            if d_corr > TOL_ECORR:
                raise RuntimeError(
                    f"{system}: ORCA vs PySCF RI-MP2 E_corr {d_corr:.2e}"
                )
            pyscf_block["orca_vs_pyscf_e_corr"] = d_corr
        else:
            res = common.run_open_shell(mol, "uhf", conv_tol=1e-12, conv_tol_grad=1e-9)
            pyscf_block = {"e_scf": res["energy"], "uhf": res}
            stability = res["stability"]
            naux = None
        d_scf = abs(pyscf_block["e_scf"] - mp2["e_scf"])
        if d_scf > TOL_SCF:
            raise RuntimeError(f"{system}: ORCA vs PySCF SCF {d_scf:.2e}")
        pyscf_block["orca_vs_pyscf_e_scf"] = d_scf
        if oo["nao"] != mol.nao_nr() or mp2["nao"] != mol.nao_nr():
            raise RuntimeError(
                f"{system}: ORCA nao {oo['nao']}/{mp2['nao']} != {mol.nao_nr()}"
            )
        if oo["naux"] != mp2["naux"] or (naux is not None and oo["naux"] != naux):
            raise RuntimeError(
                f"{system}: ORCA naux {oo['naux']}/{mp2['naux']} != {naux}"
            )

        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": basis,
            "aux_basis": aux,
            "charge": charge,
            "multiplicity": mult,
            "nao": int(mol.nao_nr()),
            "naux": int(oo["naux"]),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "orca_oo": {
                "e_total": oo["energy"],
                "e_reference": oo["e_reference"],
                "e_doubles": oo["e_doubles"],
                "e_total_sd_not_ferric_functional": oo["e_total_sd_printed"],
                "e_singles_perturbative": oo["e_singles_perturbative"],
                # ORCA prints <S^2> only for the starting SCF (see orca_rimp2).
                "orbital_iterations": oo["orbital_iterations"],
                "final_orbital_gnorm": oo["final_orbital_gnorm"],
                "input": str(inp_oo.relative_to(common.ROOT)),
            },
            "orca_rimp2": {
                "e_scf": mp2["e_scf"],
                "e_corr": mp2["e_corr"],
                "e_total": mp2["energy"],
                "s_squared_scf": mp2.get("s_squared"),
                "input": str(inp_mp2.relative_to(common.ROOT)),
            },
            "pyscf": pyscf_block,
        }
        cross = {
            "orca_oo_minus_orca_rimp2_total": oo["energy"] - mp2["energy"],
        }
        if want_grad:
            g_an = oo["gradient"]
            payload["orca_oo"]["gradient"] = g_an.tolist()
            if no_fd:
                print(f"{system}: --no-fd, not writing")
                print(f"  E_OO {oo['energy']:.12f} E_RIMP2 {mp2['energy']:.12f}")
                continue
            c0 = np.array(coords)
            g_fd = np.zeros_like(c0)
            for a in range(natm):
                for k in range(3):
                    acc = 0.0
                    for step, w in STENCIL:
                        c = c0.copy()
                        c[a, k] += step * H_FD
                        t = input_text(
                            OO_KEYWORDS.format(uhf=uhf, engrad=""),
                            symbols,
                            c.tolist(),
                            basis,
                            aux,
                            charge,
                            mult,
                        )
                        acc += w * run_oo(t, f"{system}_fd")["energy"]
                    g_fd[a, k] = acc / H_FD
            d_g = float(np.abs(g_an - g_fd).max())
            if d_g > ORCA_ANALYTIC_GROSS_TOL:
                print(
                    f"{system}: ORCA analytic\n{g_an}\nORCA FD\n{g_fd}\ndiff\n{g_an - g_fd}"
                )
                raise RuntimeError(
                    f"{system}: ORCA OO analytic vs FD {d_g:.2e} (gross)"
                )
            payload["orca_oo"]["gradient_fd"] = g_fd.tolist()
            payload["orca_oo"]["fd"] = {"stencil": "5-point central", "step_bohr": H_FD}
            cross["orca_oo_analytic_vs_fd_gradient_max"] = d_g
        payload["cross_check"] = cross
        payload["provenance"] = common.provenance(
            code=f"ORCA {oo.get('version', 'unknown')} + PySCF {pyscf.__version__}",
            version=f"ORCA {oo.get('version', 'unknown')}; PySCF {pyscf.__version__}",
            keywords={
                "orca_oo": oo_text,
                "orca_rimp2": mp2_text,
                "orca_fd": (
                    f"orca_oo input without EnGrad at displaced geometries, 5-point "
                    f"central, h = {H_FD} Bohr"
                    if want_grad
                    else "none"
                ),
                "pyscf": (
                    "scf.RHF exact J/K conv_tol 1e-12 + mp.dfmp2.DFMP2 with auxmol from "
                    "ferric's aux JSON"
                    if mult == 1
                    else "common.run_open_shell('uhf', conv_tol 1e-12, conv_tol_grad 1e-9), "
                    "exact J/K, stability loop"
                ),
            },
            basis_name=basis,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid=None,
            aux={
                "correlation": aux,
                "correlation_json": str(
                    common.basis_json_path(aux).relative_to(common.ROOT)
                ),
                "correlation_sha256": common.sha256_file(common.basis_json_path(aux)),
                "scf_and_oo_fock": "none (exact four-centre J/K: ORCA NoRI, PySCF plain SCF)",
            },
            frozen_core="none (ORCA NoFrozenCore; PySCF frozen=None)",
            scf_conv=(
                "ORCA ExtremeSCF (TolG 1e-9) with OO inputs at %scf TolE 1e-12 (OO loop "
                "energy 1e-12, orbital gradient 1e-9); PySCF conv_tol 1e-12"
            ),
            stability=stability,
            generator="scripts/validation/gen_oo_rimp2.py",
            extra={
                "orca_binary": str(common.ORCA_BINARY),
                "orca_note": "ORCA switches SHARK to segmented contraction for MP2 "
                "(same contracted functions, printed nao unchanged)",
            },
        )
        attach_numpy(payload, numpy_oo_block(symbols, coords, basis, aux, charge, mult))
        path = common.write_reference(ROW, system, basis, payload)
        written.append(path)
        print(
            f"{system:14s} E_OO {oo['energy']:.12f} (ref {oo['e_reference']:.9f} "
            f"dbl {oo['e_doubles']:.9f}, {oo['orbital_iterations']} it, "
            f"|g| {oo['final_orbital_gnorm']:.1e}) E_RIMP2 {mp2['energy']:.12f} "
            f"OO-RIMP2 {cross['orca_oo_minus_orca_rimp2_total']:+.3e} "
            f"|dE_scf| {d_scf:.1e}"
            + (
                f" |g_an-g_fd| {cross['orca_oo_analytic_vs_fd_gradient_max']:.1e}"
                if want_grad
                else ""
            )
        )
    print(f"GEN_OO_RIMP2_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
