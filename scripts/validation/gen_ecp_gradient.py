"""PySCF + ORCA references for the VALIDATION.md row "ECP gradients".

Consumer: crates/ferric-scf/tests/validation_ecp_gradient.rs.
Output:   testdata/reference/validation/ecp_gradient/<system>_<basis>.json
          testdata/reference/validation/ecp_gradient/orca_<system>_<basis>.json
ORCA inputs (committed): scripts/validation/orca/ecp_gradient/<system>_<basis>.inp

Same like-for-like machinery as gen_ecp.py (the "RHF+ECP" energy row): ferric's
bundled def2 JSON is the basis AND the ECP (its inline `ecp_potentials`), handed
to PySCF through `common.pyscf_ecp` and to ORCA as NewGTO/NewECP blocks, with
ferric's geometry in Bohr. `common.check_ecp_like_for_like` asserts the per-atom
core-electron counts, electron count and ECP term count before any number is
written.

Systems (geometries in testdata/molecules/validation/ecp/, all OFF equilibrium):

    hi     HI        RHF   def2-SVP, def2-TZVP   one ECP centre, 1-D gradient
    ch3i   CH3I      RHF   def2-SVP, def2-TZVP   distorted: x, y, z all move
    snh4   SnH4      RHF   def2-TZVP only        (ferric's def2-SVP has no Sn)
    ch2i   CH2I*     UHF   def2-SVP, def2-TZVP   doublet radical, open-shell ECP
                                                 gradient; non-degenerate SOMO
                                                 (HI+ was rejected: its 2Pi hole
                                                 is spatially degenerate)
    hbr    HBr       RHF   def2-SVP, def2-TZVP   CONTROL: Br is all-electron

Correlated / KS cases (file `<case>_<basis>.json`; see the section comments at
`KS_CASES` and `MP2_CASES` for the recipes):

    ch3i_rks_pbe   CH3I   RKS/PBE   def2-SVP   PySCF analytic, grid_response
    ch2i_uks_pbe   CH2I*  UKS/PBE   def2-SVP   PySCF analytic, grid_response
    ch2i_roks_pbe  CH2I*  ROKS/PBE  def2-SVP   5-point FD of PySCF ROKS energy
    ch3i_rimp2     CH3I   RI-MP2    def2-SVP / def2-svp-rifit
                                    5-point FD of PySCF DF-MP2 energy; ECP term
                                    from d/dlambda of E(hcore + lambda dV_ECP/dR);
                                    ORCA RI-MP2 EnGrad as a third code

Per system x basis the PySCF JSON carries:
  * the SCF energy (RHF conv_tol 1e-12, stability-checked; UHF via
    `common.run_open_shell`, several guesses, stability-followed, refused if
    unstable) and, for UHF, <S^2>;
  * the ANALYTIC gradient `mf.nuc_grad_method().kernel()`;
  * `gradient_ecp_term`: PySCF's own ECP piece of that gradient,
    sum_uv D_uv dV_ECP_uv/dR_A, built from the SAME integrals PySCF's
    `grad.rhf.hcore_generator` uses (ECPscalar_ipnuc on the bra-atom rows +
    ECPscalar_iprinv at the ECP nucleus, symmetrised) at PySCF's converged
    total density. SELF-CHECKED before writing against a central finite
    difference of tr[D V_ECP(R)] at FIXED D (the AO basis moves with the
    atoms), h = 1e-4 Bohr, every Cartesian coordinate; a miss > 1e-7 Ha/Bohr
    refuses the file. This makes the term an independent quantity the Rust
    test can compare ferric's `ecp_gradient` against in isolation;
  * `gradient_without_ecp` = gradient - gradient_ecp_term (what a gradient
    that DROPS the ECP derivative would have to match).

A reference whose max |g| < 1e-3 Ha/Bohr is refused (vacuous comparison).

ORCA 6.1.1 (`! RHF|UHF <def2> VeryTightSCF NoRI NoFrozenCore EnGrad`, NewGTO +
NewECP from ferric's JSON) is a third code for the gradient. The ORCA file is
written only if ORCA's energy matches PySCF's to 1e-6 Ha (same state), its
electron count is sum(Z) - sum(N_core) - charge, and its V_nn matches.
`--no-orca` skips it.

Usage:
    scripts/validation/run_slot.sh --light -- \\
        python scripts/validation/gen_ecp_gradient.py [--no-orca] [system ...]
Unknown system names are rejected (exit 2).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "ecp_gradient"
ROW_NAME = "ECP gradients"
ECP_MOL_DIR = common.MOL_DIR / "ecp"
ORCA_INP_DIR = Path(__file__).resolve().parent / "orca" / "ecp_gradient"
BASES = ("def2-svp", "def2-tzvp")
ORCA_BASIS_KEYWORD = {"def2-svp": "def2-SVP", "def2-tzvp": "def2-TZVP"}
# system -> (charge, multiplicity, method, bases)
SYSTEMS = {
    "hi": (0, 1, "rhf", BASES),
    "ch3i": (0, 1, "rhf", BASES),
    "snh4": (0, 1, "rhf", ("def2-tzvp",)),
    "ch2i": (0, 2, "uhf", BASES),
    "hbr": (0, 1, "rhf", BASES),
}
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-8
MIN_GRAD = 1e-3
ECP_FD_STEP = 1e-4
ECP_FD_TOL = 1e-7
ORCA_STATE_TOL = 1e-6
ORCA_ENUC_TOL = 1e-6
GUESSES = ("minao", "atom", "huckel")


def _write(row, name, basis_name, payload):
    _assert_finite_payload(payload)
    return common.write_reference(row, name, basis_name, payload)


def _assert_finite_payload(obj, path="payload"):
    """Refuse to write NaN/Infinity: json.dumps would emit them as non-standard
    JSON that serde_json rejects, and they pass every tolerance comparison."""
    import math

    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite_payload(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_finite_payload(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"non-finite value at {path}: {obj}")


def pyscf_ecp_gradient_term(mol, dm_total):
    """sum_uv D_uv dV_ECP_uv / dR_A, (natm, 3), from the integrals PySCF's
    grad.rhf.hcore_generator uses for its ECP piece (`pyscf_ecp_deriv_mats`).
    Zero without an ECP."""
    import numpy as np

    out = np.zeros((mol.natm, 3))
    if not mol.has_ecp():
        return out
    for ia, dv in enumerate(pyscf_ecp_deriv_mats(mol)):
        out[ia] = np.einsum("xij,ij->x", dv, dm_total)
    return out


def ecp_term_fd(mol, dm_total, h=ECP_FD_STEP):
    """Central FD of tr[D V_ECP(R)] at fixed D; the AO basis follows the atoms."""
    import numpy as np

    coords0 = mol.atom_coords(unit="Bohr").copy()
    out = np.zeros((mol.natm, 3))
    m = mol.copy()
    for ia in range(mol.natm):
        for c in range(3):
            e = []
            for s in (1.0, -1.0):
                xyz = coords0.copy()
                xyz[ia, c] += s * h
                m.set_geom_(xyz, unit="Bohr", symmetry=None)
                e.append(float(np.einsum("ij,ij->", m.intor("ECPscalar"), dm_total)))
            out[ia, c] = (e[0] - e[1]) / (2.0 * h)
    return out


def run_rhf(mol):
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    _mo, stable = common._stability_status(mf)
    if not stable:
        raise RuntimeError("RHF internal instability; refusing to write")
    return mf, {
        "energy": float(mf.e_tot),
        "converged": True,
        "stability": {"internal_stable": True, "kind": "PySCF RHF internal (real)"},
    }


def gradient_block(mol, mf, grad=None, grid_response=False) -> dict:
    """Gradient + separated ECP term. `grad` overrides the analytic gradient
    (the ROKS case passes its finite-difference reference); `grid_response`
    is set on PySCF's KS gradient object (ferric's KS gradients include it)."""
    import numpy as np

    if grad is None:
        g = mf.nuc_grad_method()
        if grid_response:
            g.grid_response = True
        g.verbose = 0
        grad = g.kernel()
    grad = np.asarray(grad)
    gmax = float(np.max(np.abs(grad)))
    if gmax < MIN_GRAD:
        raise RuntimeError(f"max |gradient| {gmax:.2e} < {MIN_GRAD:.0e}: vacuous")
    dm = mf.make_rdm1()
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    g_ecp = pyscf_ecp_gradient_term(mol, dm)
    fd = ecp_term_fd(mol, dm)
    fd_err = float(np.max(np.abs(g_ecp - fd)))
    if fd_err > ECP_FD_TOL:
        raise RuntimeError(
            f"PySCF ECP gradient term misses its own fixed-D FD by {fd_err:.2e} "
            f"(> {ECP_FD_TOL:.0e}); the term decomposition is wrong, refusing"
        )
    rows = lambda a: [[float(v) for v in r] for r in a]  # noqa: E731
    return {
        "gradient": rows(grad),
        "gradient_units": "Hartree/Bohr, atoms in xyz order",
        "gradient_max_abs": gmax,
        "gradient_sum": [float(v) for v in grad.sum(axis=0)],
        "gradient_ecp_term": rows(g_ecp),
        "gradient_ecp_term_max_abs": float(np.max(np.abs(g_ecp))),
        "gradient_ecp_term_fd_fixed_density_max_err": fd_err,
        "gradient_without_ecp": rows(grad - g_ecp),
    }


def parse_engrad(path: Path, natm: int) -> tuple[float, list[list[float]]]:
    """ORCA .engrad: comment-delimited blocks (natoms, energy, 3N gradient)."""
    blocks, cur = [], []
    for line in path.read_text().splitlines():
        if line.strip().startswith("#"):
            if cur:
                blocks.append(cur)
                cur = []
            continue
        if line.strip():
            cur.append(line.strip())
    if cur:
        blocks.append(cur)
    if int(blocks[0][0]) != natm:
        raise RuntimeError(f"{path}: engrad natoms {blocks[0][0]} != {natm}")
    energy = float(blocks[1][0])
    flat = [float(x) for x in blocks[2]]
    if len(flat) != 3 * natm:
        raise RuntimeError(f"{path}: {len(flat)} gradient entries, expected {3 * natm}")
    return energy, [flat[3 * i : 3 * i + 3] for i in range(natm)]


def run_orca_gradient(system, basis_name, charge, mult, method, xyz, pyscf_block):
    import numpy as np

    symbols, coords = common.read_xyz(xyz)
    ecp_json = common.basis_json_path(basis_name)
    ncore = common.ecp_core_electrons(ecp_json, symbols)
    extra = "%scf\n  STABPerform true\nend" if method == "uhf" else ""
    keywords = (
        f"{method.upper()} {ORCA_BASIS_KEYWORD[basis_name]} VeryTightSCF NoRI "
        "NoFrozenCore EnGrad"
    )
    inp = common.write_orca_input(
        ORCA_INP_DIR / f"{system}_{basis_name}.inp",
        keywords,
        xyz,
        basis_name,
        charge=charge,
        multiplicity=mult,
        extra_blocks=extra,
        ecp_json=ecp_json,
    )
    with tempfile.TemporaryDirectory(prefix="ferric-orca-ecpgrad-") as tmp:
        run_inp = Path(tmp) / inp.name
        shutil.copy(inp, run_inp)
        parsed = common.run_orca(run_inp)
        e_grad, grad = parse_engrad(run_inp.with_suffix(".engrad"), len(symbols))
    nelec_want = sum(common.z_of(s) for s in symbols) - sum(ncore) - charge
    if int(parsed.get("n_electrons", -1)) != nelec_want:
        raise RuntimeError(
            f"{inp.name}: ORCA NEL {parsed.get('n_electrons')} != {nelec_want}"
        )
    q = [common.z_of(s) - n for s, n in zip(symbols, ncore)]
    enuc = sum(
        q[i] * q[j] / float(np.linalg.norm(np.subtract(coords[i], coords[j])))
        for i in range(len(q))
        for j in range(i)
    )
    if abs(parsed["nuclear_repulsion"] - enuc) > ORCA_ENUC_TOL:
        raise RuntimeError(
            f"{inp.name}: ORCA E_nuc {parsed['nuclear_repulsion']} != {enuc}"
        )
    e_state = abs(e_grad - pyscf_block["energy"])
    if e_state > ORCA_STATE_TOL:
        raise RuntimeError(
            f"{inp.name}: ORCA E {e_grad:.10f} vs PySCF {pyscf_block['energy']:.10f} "
            f"(|d| {e_state:.2e}): not the same state, refusing"
        )
    d_pyscf = float(np.max(np.abs(np.subtract(grad, pyscf_block["gradient"]))))
    payload = {
        "row": ROW_NAME + " (ORCA cross-check)",
        "system": system,
        "basis": basis_name,
        "charge": charge,
        "multiplicity": mult,
        "method": method,
        "ecp_source": "ferric-json NewECP",
        "nelectron": nelec_want,
        "ecp_core_electrons": ncore,
        "nuclear_repulsion": parsed["nuclear_repulsion"],
        "energy": e_grad,
        "s_squared": parsed.get("s_squared"),
        "gradient": grad,
        "gradient_units": "Hartree/Bohr, atoms in xyz order (ORCA .engrad)",
        "max_abs_diff_vs_pyscf_gradient": d_pyscf,
        "provenance": common.provenance(
            code="ORCA",
            version=parsed.get("version", "unknown"),
            keywords=inp.read_text(),
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid=None,
            aux=None,
            frozen_core="NoFrozenCore",
            scf_conv="VeryTightSCF",
            stability={"kind": "ORCA %scf STABPerform"} if method == "uhf" else None,
            ecp_json=ecp_json,
            generator="scripts/validation/gen_ecp_gradient.py",
            extra={
                "orca_input": str(inp.relative_to(common.ROOT)),
                "orca_binary": str(common.ORCA_BINARY),
            },
        ),
    }
    path = _write(ROW, f"orca_{system}", basis_name, payload)
    print(
        f"  ORCA {system:5s} {basis_name:9s} E {e_grad:.10f} "
        f"max|g_ORCA - g_PySCF| {d_pyscf:.2e}"
    )
    return path


# ---------------------------------------------------------------------------
# KS cases: the RKS / UKS / ROKS ECP gradient paths (ks_gradient.rs)
# ---------------------------------------------------------------------------
#
# Grid recipe = the KS-energy and meta-GGA gradient rows (gen_ks_energies.py,
# gen_mgga_gradients.py): (75,110) unpruned, Becke partition, Becke-1988 radii
# adjustment, PySCF's default Treutler-Ahlrichs radial grid — what ferric's
# AtomicGridConfig::default() builds for the SCF and ks_gradient_*. Both grids
# use the BARE nuclear charge of an ECP atom (ferric grid.rs/becke.rs read
# atom.z; PySCF gen_atomic_grids reads the element), so the ECP does not move
# the grid. Coulomb is EXACT four-centre on both sides (plain PySCF KS; ferric
# df_j_aux = Some("") closed shell / None open shell), so no aux basis enters.
# PySCF's analytic gradients are run with grid_response = True, which ferric's
# KS gradients include.

KS_GRID = (75, 110)
KS_XC = {"pbe": "PBE"}
# case -> (system, charge, multiplicity, kind, xc, basis)
KS_CASES = {
    "ch3i_rks_pbe": ("ch3i", 0, 1, "rks", "pbe", "def2-svp"),
    "ch2i_uks_pbe": ("ch2i", 0, 2, "uks", "pbe", "def2-svp"),
    "ch2i_roks_pbe": ("ch2i", 0, 2, "roks", "pbe", "def2-svp"),
}
KS_CONV_TOL = 1e-11
FD_CONV_TOL = 1e-12
FD_STEP = 2e-3  # Bohr, 5-point stencil (ROKS and MP2 references)
FD_BASIN = 1e-2  # a displaced energy further than this from E0 changed state
STENCIL5 = ((-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12))


def _ks(kind, mol, xc, conv_tol=KS_CONV_TOL):
    from pyscf import dft

    cls = {"rks": dft.RKS, "uks": dft.UKS, "roks": dft.ROKS}[kind]
    mf = cls(mol, xc=KS_XC[xc])
    mf.grids.atom_grid = KS_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    return mf


def fd5_gradient(energy_at, coords0, h=FD_STEP):
    """5-point central FD of `energy_at(coords_bohr)` over every coordinate."""
    import numpy as np

    coords0 = np.asarray(coords0, dtype=float)
    g = np.zeros_like(coords0)
    for a in range(coords0.shape[0]):
        for c in range(3):
            acc = 0.0
            for k, w in STENCIL5:
                x = coords0.copy()
                x[a, c] += k * h
                acc += w * energy_at(x)
            g[a, c] = acc / h
    return g


def _header(case, system, basis_name, charge, mult, method, mol, ecp_check):
    return {
        "row": ROW_NAME,
        "case": case,
        "system": system,
        "basis": basis_name,
        "charge": charge,
        "multiplicity": mult,
        "method": method,
        "nao": mol.nao_nr(),
        "nelectron": int(mol.nelectron),
        "ecp_core_electrons": ecp_check["ecp_core_electrons"],
        "has_ecp": bool(mol.has_ecp()),
        "nuclear_repulsion": float(mol.energy_nuc()),
    }


def gen_ks_case(case: str) -> Path:
    import numpy as np
    import pyscf

    system, charge, mult, kind, xc, basis_name = KS_CASES[case]
    xyz = ECP_MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    ecp_json = common.basis_json_path(basis_name)
    mol = common.build_pyscf_mol(
        xyz, basis_name, charge=charge, multiplicity=mult, ecp_json=ecp_json
    )
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    ecp_check = common.check_ecp_like_for_like(mol, ecp_json, symbols)
    payload = _header(case, system, basis_name, charge, mult, kind, mol, ecp_check)
    payload["xc"] = KS_XC[xc]

    if kind == "rks":
        mf = _ks("rks", mol, xc)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"{case}: RKS did not converge")
        _mo, stable = common._stability_status(mf)
        if not stable:
            raise RuntimeError(f"{case}: RKS internal instability; refusing")
        block = {
            "energy": float(mf.e_tot),
            "converged": True,
            "stability": {"internal_stable": True, "kind": "PySCF RKS internal (real)"},
        }
        block.update(gradient_block(mol, mf, grid_response=True))
        block["gradient_source"] = "PySCF analytic, grid_response=True"
    else:
        block, mf = common.run_open_shell(
            mol,
            "uks" if kind == "uks" else "rohf",
            conv_tol=KS_CONV_TOL,
            conv_tol_grad=CONV_TOL_GRAD,
            guesses=GUESSES,
            mf_factory=lambda m: _ks(kind, m, xc),
            return_mf=True,
        )
        if kind == "uks":
            block.update(gradient_block(mol, mf, grid_response=True))
            block["gradient_source"] = "PySCF analytic, grid_response=True"
        else:
            # ROKS: the reference is a 5-point FD of PySCF's ROKS energy, each
            # displaced SCF seeded from the reference density. PySCF's analytic
            # ROKS gradient is recorded next to it as a second number.
            dm0, e0 = mf.make_rdm1(), mf.e_tot

            def energy_at(x):
                m = mol.set_geom_(x, unit="Bohr", inplace=False)
                f = _ks("roks", m, xc, conv_tol=FD_CONV_TOL)
                e = f.kernel(dm0=dm0)
                if not f.converged:
                    raise RuntimeError(f"{case}: displaced ROKS did not converge")
                if abs(e - e0) > FD_BASIN:
                    raise RuntimeError(
                        f"{case}: displaced ROKS changed state ({e - e0:+.3e})"
                    )
                return e

            g_fd = fd5_gradient(energy_at, mol.atom_coords(unit="Bohr"))
            ga = mf.nuc_grad_method()
            ga.grid_response = True
            ga.verbose = 0
            g_an = np.asarray(ga.kernel())
            block.update(gradient_block(mol, mf, grad=g_fd))
            block["gradient_source"] = (
                f"5-point central FD of PySCF ROKS energy, h = {FD_STEP} Bohr, "
                f"conv_tol {FD_CONV_TOL}, seeded from the reference density"
            )
            block["pyscf_analytic_gradient"] = [[float(v) for v in r] for r in g_an]
            block["pyscf_analytic_vs_fd"] = float(np.max(np.abs(g_an - g_fd)))
    payload[kind] = block
    payload["provenance"] = common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords={
            "method": f"dft.{kind.upper()} xc={KS_XC[xc]}",
            "gradient": block["gradient_source"],
            "ecp_term": "ECPscalar_ipnuc + ECPscalar_iprinv, FD-checked at fixed D",
            "eri": "exact 4-index J (no density fitting)",
            "numpy": np.__version__,
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid={
            "atom_grid": list(KS_GRID),
            "prune": None,
            "partition": "Becke (original_becke)",
            "radii_adjust": "becke_atomic_radii_adjust",
            "radial": "PySCF default (treutler_ahlrichs)",
            "grid_response": True,
        },
        aux=None,
        frozen_core=None,
        scf_conv={"conv_tol": KS_CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability=block["stability"],
        ecp_json=ecp_json,
        generator="scripts/validation/gen_ecp_gradient.py",
        extra={"basis_self_check": basis_check, "ecp_self_check": ecp_check},
    )
    path = _write(ROW, case, basis_name, payload)
    extra = (
        f" analytic-vs-FD {block['pyscf_analytic_vs_fd']:.1e}" if kind == "roks" else ""
    )
    print(
        f"{case:14s} {basis_name:9s} E {block['energy']:.10f} "
        f"|g|max {block['gradient_max_abs']:.3e} "
        f"|g_ecp|max {block['gradient_ecp_term_max_abs']:.3e} "
        f"ecp-term FD err {block['gradient_ecp_term_fd_fixed_density_max_err']:.1e} "
        f"sum(g) {max(abs(v) for v in block['gradient_sum']):.1e}"
        f"{' <S2> %.6f' % block['s_squared'] if 's_squared' in block else ''}{extra}"
    )
    return path


# ---------------------------------------------------------------------------
# RI-MP2 case (crates/ferric-mp2 mp2_relaxed_lagrangian_gradient)
# ---------------------------------------------------------------------------
#
# Same recipe as the RI-MP2 gradient row (gen_rimp2_gradient.py): exact-J/K RHF
# + pyscf.mp.dfmp2.DFMP2 (all electrons correlated) with the auxmol built from
# ferric's aux JSON. PySCF 2.13 has NO analytic DF-MP2 gradient, so the
# reference is a 5-point FD of the DF-MP2 total energy. The ECP piece of the
# MP2 gradient, sum D_relaxed dV_ECP/dR, is obtained WITHOUT a relaxed-density
# code: for each (atom, xyz), perturb hcore by lambda * X, X = dV_ECP/dR_{A,c}
# (the matrices of `pyscf_ecp_deriv_mats`), and take the 5-point derivative in
# lambda of the relaxed (SCF re-solved) DF-MP2 energy — by definition
# tr[D_relaxed X]. The same lambda construction on the RHF energy alone must
# reproduce the analytic RHF ECP term (a self-check, refused if > 1e-7).
# ORCA 6.1.1 `RI-MP2 NoRI NoFrozenCore ExtremeSCF EnGrad` with NewGTO + NewECP +
# NewAuxCGTO from ferric's JSONs is a third code.

# case -> (system, charge, multiplicity, basis, aux)
MP2_CASES = {"ch3i_rimp2": ("ch3i", 0, 1, "def2-svp", "def2-svp-rifit")}
LAMBDA_STEP = 1e-3
LAMBDA_TOL = 1e-7
MP2_ORCA_E_TOL = 1e-7


def pyscf_ecp_deriv_mats(mol):
    """Per atom, the (3, n, n) dV_ECP/dR_A matrices of grad.rhf.hcore_generator."""
    import numpy as np

    ecp_atoms = set(int(a) for a in mol._ecpbas[:, 0])
    ipnuc = -mol.intor("ECPscalar_ipnuc", comp=3)
    aoslices = mol.aoslice_by_atom()
    out = []
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        a = np.zeros_like(ipnuc)
        if ia in ecp_atoms:
            with mol.with_rinv_at_nucleus(ia):
                a += mol.intor("ECPscalar_iprinv", comp=3)
        a[:, p0:p1] += ipnuc[:, p0:p1]
        out.append(a + a.transpose(0, 2, 1))
    return out


def _dfmp2(mol, aux_bas, dm0=None, h_extra=None, mp2=True):
    """(E_RHF, E_corr, mf, naux): exact-J/K RHF (+ DF-MP2, all electrons)."""
    from pyscf import df, scf
    from pyscf.mp import dfmp2

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.verbose = 0
    if h_extra is not None:
        h0 = mf.get_hcore()  # includes V_ECP
        mf.get_hcore = lambda *a, **k: h0 + h_extra
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge")
    if not mp2:
        return mf.e_tot, 0.0, mf, 0
    pt = dfmp2.DFMP2(mf, frozen=None)
    pt.with_df = df.DF(mol)
    pt.with_df.auxmol = df.addons.make_auxmol(mol, aux_bas)
    pt.with_df.auxmol.cart = False
    pt.with_df.auxmol.build()
    pt.with_df.auxbasis = aux_bas
    pt.verbose = 0
    pt.kernel()
    return mf.e_tot, pt.e_corr, mf, pt.with_df.auxmol.nao_nr()


def _lambda_derivative(mol, aux_bas, dm0, mats, mp2):
    """d E / d lambda of E(hcore + lambda X_{A,c}) for every (A, c)."""
    import numpy as np

    out = np.zeros((mol.natm, 3))
    for a in range(mol.natm):
        for c in range(3):
            acc = 0.0
            for k, w in STENCIL5:
                e_rhf, e_corr, *_ = _dfmp2(
                    mol, aux_bas, dm0=dm0, h_extra=k * LAMBDA_STEP * mats[a][c], mp2=mp2
                )
                acc += w * (e_rhf + e_corr)
            out[a, c] = acc / LAMBDA_STEP
    return out


def _orca_mp2(case, xyz, basis_name, aux, charge, mult, ecp_json):
    import gen_rimp2_gradient as grg

    symbols, coords = common.read_xyz(xyz)
    block = common.orca_basis_block(basis_name, symbols, ecp_json=ecp_json).splitlines()
    assert block[-1] == "end"
    block = block[:-1] + grg.orca_aux_c_lines(aux, symbols) + ["end"]
    body = [
        "! RI-MP2 NoRI NoFrozenCore ExtremeSCF EnGrad Bohrs",
        "%pal nprocs 1 end",
        *block,
        f"* xyz {charge} {mult}",
    ]
    body += [
        f"  {s:<2} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(symbols, coords)
    ]
    body.append("*")
    inp = ORCA_INP_DIR / f"{case}_{basis_name}.inp"
    inp.parent.mkdir(parents=True, exist_ok=True)
    inp.write_text("\n".join(body) + "\n")
    return inp, grg.run_orca(inp, len(symbols))


def gen_mp2_case(case: str, do_orca: bool) -> list[Path]:
    import numpy as np
    import pyscf

    system, charge, mult, basis_name, aux = MP2_CASES[case]
    xyz = ECP_MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    ecp_json = common.basis_json_path(basis_name)
    mol = common.build_pyscf_mol(
        xyz, basis_name, charge=charge, multiplicity=mult, ecp_json=ecp_json
    )
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    ecp_check = common.check_ecp_like_for_like(mol, ecp_json, symbols)
    aux_bas, aux_cart = common.pyscf_basis(aux, symbols)
    assert not aux_cart
    rows = lambda a: [[float(v) for v in r] for r in np.asarray(a)]  # noqa: E731

    e_rhf, e_corr, mf, naux = _dfmp2(mol, aux_bas)
    if naux != common.ferric_nao(aux, symbols):
        raise RuntimeError(f"{case}: aux nao {naux} != ferric's")
    _mo, stable = common._stability_status(mf)
    if not stable:
        raise RuntimeError(f"{case}: RHF internal instability; refusing")
    dm0 = mf.make_rdm1()
    g_rhf = np.asarray(mf.nuc_grad_method().kernel())
    mats = pyscf_ecp_deriv_mats(mol)

    # Self-check of the lambda construction on the RHF energy alone.
    g_ecp_rhf_analytic = pyscf_ecp_gradient_term(mol, dm0)
    g_ecp_rhf_lambda = _lambda_derivative(mol, aux_bas, dm0, mats, mp2=False)
    lam_err = float(np.max(np.abs(g_ecp_rhf_lambda - g_ecp_rhf_analytic)))
    if lam_err > LAMBDA_TOL:
        raise RuntimeError(
            f"{case}: lambda-derivative RHF ECP term misses the analytic one by "
            f"{lam_err:.2e}; refusing"
        )
    g_ecp = _lambda_derivative(mol, aux_bas, dm0, mats, mp2=True)

    e_tot0 = e_rhf + e_corr

    def energy_at(x):
        m = mol.set_geom_(x, unit="Bohr", inplace=False)
        er, ec, *_ = _dfmp2(m, aux_bas, dm0=dm0)
        if abs(er + ec - e_tot0) > FD_BASIN:
            raise RuntimeError(f"{case}: displaced RHF changed state")
        return er + ec

    g_fd = fd5_gradient(energy_at, mol.atom_coords(unit="Bohr"))
    block = {
        "e_rhf": float(e_rhf),
        "e_corr": float(e_corr),
        "e_total": float(e_tot0),
        "gradient": rows(g_fd),
        "gradient_source": f"5-point central FD of PySCF DF-MP2 total energy, h = {FD_STEP} Bohr",
        "gradient_max_abs": float(np.max(np.abs(g_fd))),
        "gradient_sum": [float(v) for v in g_fd.sum(axis=0)],
        "gradient_ecp_term": rows(g_ecp),
        "gradient_ecp_term_source": (
            "5-point d/dlambda of the relaxed DF-MP2 energy with hcore + lambda dV_ECP/dR, "
            f"lambda step {LAMBDA_STEP}"
        ),
        "gradient_ecp_term_max_abs": float(np.max(np.abs(g_ecp))),
        "gradient_without_ecp": rows(g_fd - g_ecp),
        "rhf_gradient": rows(g_rhf),
        "rhf_gradient_ecp_term": rows(g_ecp_rhf_analytic),
        "lambda_self_check_rhf_max_err": lam_err,
    }
    payload = _header(case, system, basis_name, charge, mult, "rimp2", mol, ecp_check)
    payload["aux_basis"] = aux
    payload["naux"] = int(naux)
    payload["rimp2"] = block
    written = []
    orca_note = "not run"
    if do_orca:
        inp, orca = _orca_mp2(case, xyz, basis_name, aux, charge, mult, ecp_json)
        d_rhf = abs(orca["e_rhf"] - e_rhf)
        d_corr = abs(orca["e_corr"] - e_corr)
        if orca.get("nao") != mol.nao_nr() or orca.get("naux") != naux:
            raise RuntimeError(
                f"{case}: ORCA nao/naux {orca.get('nao')}/{orca.get('naux')}"
            )
        if d_rhf > MP2_ORCA_E_TOL or d_corr > MP2_ORCA_E_TOL:
            raise RuntimeError(
                f"{case}: ORCA vs PySCF |dE_rhf| {d_rhf:.2e} |dE_corr| {d_corr:.2e}; refusing"
            )
        d_g = float(np.max(np.abs(orca["gradient"] - g_fd)))
        block["orca"] = {
            "e_rhf": orca["e_rhf"],
            "e_corr": orca["e_corr"],
            "e_total": orca["energy"],
            "gradient": rows(orca["gradient"]),
            "input": str(inp.relative_to(common.ROOT)),
            "vs_pyscf": {"e_rhf": d_rhf, "e_corr": d_corr, "gradient_max": d_g},
        }
        orca_note = f"ORCA |dE_rhf| {d_rhf:.1e} |dE_corr| {d_corr:.1e} |dg| {d_g:.1e}"
    payload["provenance"] = common.provenance(
        code="PySCF" + (" + ORCA" if do_orca else ""),
        version=pyscf.__version__,
        keywords={
            "pyscf": "scf.RHF (exact J/K, conv_tol 1e-12, conv_tol_grad 1e-10) + "
            "mp.dfmp2.DFMP2(frozen=None), auxmol from ferric's aux JSON",
            "gradient": block["gradient_source"],
            "ecp_term": block["gradient_ecp_term_source"],
            "orca": "RI-MP2 NoRI NoFrozenCore ExtremeSCF EnGrad, NewGTO+NewECP+NewAuxCGTO",
            "numpy": np.__version__,
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux={
            "correlation": aux,
            "correlation_sha256": common.sha256_file(common.basis_json_path(aux)),
            "scf": "none (exact four-centre J/K)",
        },
        frozen_core="none (all electrons outside the ECP core correlated)",
        scf_conv={"conv_tol": 1e-12, "conv_tol_grad": 1e-10},
        stability={"internal_stable": True, "kind": "PySCF RHF internal (real)"},
        ecp_json=ecp_json,
        generator="scripts/validation/gen_ecp_gradient.py",
        extra={"basis_self_check": basis_check, "ecp_self_check": ecp_check},
    )
    written.append(_write(ROW, case, basis_name, payload))
    print(
        f"{case:14s} {basis_name:9s} E_tot {e_tot0:.10f} E_corr {e_corr:.10f} "
        f"|g|max {block['gradient_max_abs']:.3e} |g_ecp|max "
        f"{block['gradient_ecp_term_max_abs']:.3e} "
        f"|g_ecp(MP2)-g_ecp(RHF)| {float(np.max(np.abs(g_ecp - g_ecp_rhf_analytic))):.2e} "
        f"lambda self-check {lam_err:.1e} sum(g) {max(abs(v) for v in block['gradient_sum']):.1e} "
        f"{orca_note}"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    import numpy
    import pyscf

    args = sys.argv[1:] if argv is None else argv
    flags = {a for a in args if a.startswith("--")}
    only = [a for a in args if not a.startswith("--")]
    bad_flags = flags - {"--no-orca"}
    known = [*SYSTEMS, *KS_CASES, *MP2_CASES]
    unknown = [s for s in only if s not in known]
    if bad_flags or unknown:
        print(
            f"unknown flag(s) {sorted(bad_flags)} / case(s) {unknown}; "
            f"cases are {known}",
            file=sys.stderr,
        )
        return 2
    do_orca = "--no-orca" not in flags
    written = []
    for system, (charge, mult, method, bases) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = ECP_MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        for basis_name in bases:
            ecp_json = common.basis_json_path(basis_name)
            mol = common.build_pyscf_mol(
                xyz, basis_name, charge=charge, multiplicity=mult, ecp_json=ecp_json
            )
            basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
            ecp_check = common.check_ecp_like_for_like(mol, ecp_json, symbols)
            payload = {
                "row": ROW_NAME,
                "system": system,
                "basis": basis_name,
                "charge": charge,
                "multiplicity": mult,
                "method": method,
                "nao": mol.nao_nr(),
                "nelectron": int(mol.nelectron),
                "ecp_core_electrons": ecp_check["ecp_core_electrons"],
                "has_ecp": bool(mol.has_ecp()),
                "nuclear_repulsion": float(mol.energy_nuc()),
            }
            if method == "rhf":
                mf, block = run_rhf(mol)
            else:
                block, mf = common.run_open_shell(
                    mol,
                    "uhf",
                    conv_tol=CONV_TOL,
                    conv_tol_grad=CONV_TOL_GRAD,
                    guesses=GUESSES,
                    return_mf=True,
                )
            block.update(gradient_block(mol, mf))
            payload[method] = block
            payload["provenance"] = common.provenance(
                code="PySCF",
                version=pyscf.__version__,
                keywords={
                    "methods": [
                        f"scf.{method.upper()}",
                        "nuc_grad_method() (analytic)",
                    ],
                    "ecp_term": "ECPscalar_ipnuc + ECPscalar_iprinv (grad.rhf."
                    "hcore_generator's ECP piece), FD-checked at fixed D",
                    "conv_tol": CONV_TOL,
                    "conv_tol_grad": CONV_TOL_GRAD,
                    "ecp": "mol.ecp built by common.pyscf_ecp from the basis JSON's "
                    "inline ecp_potentials",
                    "eri": "exact 4-index (no density fitting)",
                    "numpy": numpy.__version__,
                },
                basis_name=basis_name,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid=None,
                aux=None,
                frozen_core=None,
                scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                stability=block["stability"],
                ecp_json=ecp_json,
                generator="scripts/validation/gen_ecp_gradient.py",
                extra={"basis_self_check": basis_check, "ecp_self_check": ecp_check},
            )
            written.append(_write(ROW, system, basis_name, payload))
            print(
                f"{system:5s} {basis_name:9s} {method.upper()} E {block['energy']:.10f} "
                f"|g|max {block['gradient_max_abs']:.3e} "
                f"|g_ecp|max {block['gradient_ecp_term_max_abs']:.3e} "
                f"ecp-term FD err {block['gradient_ecp_term_fd_fixed_density_max_err']:.1e} "
                f"sum(g) {max(abs(v) for v in block['gradient_sum']):.1e}"
            )
            if do_orca:
                written.append(
                    run_orca_gradient(
                        system, basis_name, charge, mult, method, xyz, block
                    )
                )
    for case in KS_CASES:
        if not only or case in only:
            written.append(gen_ks_case(case))
    for case in MP2_CASES:
        if not only or case in only:
            written += gen_mp2_case(case, do_orca)
    print(f"GEN_ECP_GRADIENT_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
