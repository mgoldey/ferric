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


def pyscf_ecp_gradient_term(mol, dm_total):
    """sum_uv D_uv dV_ECP_uv / dR_A, (natm, 3), from the integrals PySCF's
    grad.rhf.hcore_generator uses for its ECP piece. Zero without an ECP."""
    import numpy as np

    natm = mol.natm
    out = np.zeros((natm, 3))
    if not mol.has_ecp():
        return out
    ecp_atoms = set(int(a) for a in mol._ecpbas[:, 0])  # gto.ATOM_OF == 0
    ipnuc = -mol.intor("ECPscalar_ipnuc", comp=3)  # get_hcore's sign
    aoslices = mol.aoslice_by_atom()
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        a = np.zeros_like(ipnuc)
        if ia in ecp_atoms:
            with mol.with_rinv_at_nucleus(ia):
                a += mol.intor("ECPscalar_iprinv", comp=3)
        a[:, p0:p1] += ipnuc[:, p0:p1]
        dv = a + a.transpose(0, 2, 1)
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


def gradient_block(mol, mf) -> dict:
    import numpy as np

    g = mf.nuc_grad_method()
    g.verbose = 0
    grad = np.asarray(g.kernel())
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
    path = common.write_reference(ROW, f"orca_{system}", basis_name, payload)
    print(
        f"  ORCA {system:5s} {basis_name:9s} E {e_grad:.10f} "
        f"max|g_ORCA - g_PySCF| {d_pyscf:.2e}"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    import numpy
    import pyscf

    args = sys.argv[1:] if argv is None else argv
    flags = {a for a in args if a.startswith("--")}
    only = [a for a in args if not a.startswith("--")]
    bad_flags = flags - {"--no-orca"}
    unknown = [s for s in only if s not in SYSTEMS]
    if bad_flags or unknown:
        print(
            f"unknown flag(s) {sorted(bad_flags)} / system(s) {unknown}; "
            f"systems are {list(SYSTEMS)}",
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
            written.append(common.write_reference(ROW, system, basis_name, payload))
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
    print(f"GEN_ECP_GRADIENT_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
