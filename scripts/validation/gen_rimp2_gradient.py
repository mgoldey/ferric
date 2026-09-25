"""References for the "RI-MP2 gradient" validation row (VALIDATION.md 112).

Consumer: crates/ferric-mp2/tests/validation_rimp2_gradient.rs
Output:   testdata/reference/validation/rimp2_gradient/<system>_<basis>.json
ORCA inputs (committed): scripts/validation/orca/rimp2_gradient/<system>_<basis>{,_fc}.inp

WHAT FERRIC COMPUTES (read from crates/ferric-mp2/src/gradient.rs and
crates/ferric-scf/src/rhf.rs, not assumed): `rimp2_gradient_analytical` on an
RHF from `solve_rhf` with `RhfConfig::default()` fitting fields
(`df_j_aux = df_k_aux = None`, `k_builder = None`), i.e. EXACT four-centre
J and K in the SCF; the z-vector (`zvector.rs`) and the 2e-derivative block
(`twoelectron_gradient_bilinear`) are exact four-centre too. Only the MP2
correlation part is density-fitted, with the Coulomb metric. So the matched
references are:

  * ORCA 6.1.1: `! RI-MP2 NoRI NoFrozenCore ExtremeSCF EnGrad`, ferric's
    orbital basis as `NewGTO` and ferric's aux basis as `NewAuxCGTO` (the /C
    slot), geometry in Bohr. `NoRI` = exact J/K in the SCF and the Z-vector.
    The gradient is read from ORCA's `.engrad` file (full precision).
  * PySCF 2.13.1: exact-integral RHF + `pyscf.mp.dfmp2.DFMP2` with
    `with_df.auxmol` built from ferric's aux JSON. PySCF has NO analytic
    DF-MP2 gradient (`DFMP2.nuc_grad_method` and `dfmp2_native.DFRMP2
    .nuc_grad_method` both raise NotImplementedError in 2.13.1; `pyscf.grad`
    has only conventional mp2/ump2), so the PySCF gradient is a 5-point
    central finite difference of the DF-MP2 TOTAL energy, step H_FD Bohr.
    Truncation O(h^4 E^(5)) ~ 1e-11; SCF noise with conv_tol 1e-12 /
    conv_tol_grad 1e-10 is ~1e-12 Ha in E, /h -> ~1e-9 in g.

Two independent codes and two independent methods (analytic Z-vector vs FD).
The generator REFUSES to write unless:
  * ORCA and PySCF RHF energies agree to 1e-8 (basis + geometry like-for-like);
  * ORCA and PySCF RI-MP2 correlation energies agree to 1e-8 — the check that
    ORCA really used ferric's aux basis (a different /C set moves E_corr by
    >1e-5 at these sizes);
  * ORCA analytic and PySCF FD gradients agree to CROSS_TOL (1e-6).
Also stored, as controls for the Rust test:
  * the PySCF analytic RHF gradient (exact J/K) — the exactness anchor for
    everything but the correlation part, and a negative control (the MP2
    gradient must MISS it by >> the bar);
  * ORCA's FROZEN-CORE RI-MP2 gradient — a negative control that the
    reference discriminates the one setting most likely to be silently wrong.

Run (light: every ORCA job is seconds on 1 core, the PySCF FD is ~50 energies):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_rimp2_gradient.py [system ...]
`--write-only` regenerates the committed .inp files without running anything.
ORCA runs in a temporary directory (copies of the .inp).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "rimp2_gradient"
ROW_NAME = "RI-MP2 gradient"
INP_DIR = Path(__file__).resolve().parent / "orca" / ROW

# system -> (orbital basis, aux basis, number of core orbitals for the FC control)
SYSTEMS = {
    "h2o_distorted": ("cc-pvdz", "cc-pvdz-ri", 1),
    "hcn_bent": ("cc-pvdz", "cc-pvdz-ri", 2),
    "nh3_distorted": ("def2-svp", "def2-svp-rifit", 1),
}

KEYWORDS = "RI-MP2 NoRI {fc} ExtremeSCF EnGrad"
H_FD = 2e-3  # Bohr, 5-point stencil
TOL_RHF = 1e-8
TOL_ECORR = 1e-8
CROSS_TOL = 1e-6


# ---------------------------------------------------------------------------
# ORCA
# ---------------------------------------------------------------------------


def orca_aux_c_lines(aux_name: str, symbols: list[str]) -> list[str]:
    """`NewAuxCGTO <El> ... end` entries (inside `%basis`) carrying ferric's
    aux basis, emitted exactly as `common.orca_basis_block` emits NewGTO
    (renormalized contraction coefficients, one segmented shell per column)."""
    lines = []
    for s in dict.fromkeys(symbols):
        lines.append(f"  NewAuxCGTO {s}")
        for sh in common.ferric_shells(aux_name, common.z_of(s)):
            if sh["l"] >= 2 and not sh["pure"]:
                raise ValueError(f"{aux_name}: Cartesian l={sh['l']} aux shell")
            lines.append(f"    {common.ORCA_SHELL_LETTER[sh['l']]} {len(sh['exps'])}")
            for i, (e, c) in enumerate(zip(sh["exps"], sh["coefs"]), start=1):
                lines.append(f"      {i:3d} {e:.10E} {c:.10E}")
        lines.append("  end")
    return lines


def write_input(path: Path, xyz: Path, basis: str, aux: str, frozen: bool) -> Path:
    symbols, coords = common.read_xyz(xyz)
    basis_block = common.orca_basis_block(basis, symbols).splitlines()
    assert basis_block[-1] == "end"
    basis_block = basis_block[:-1] + orca_aux_c_lines(aux, symbols) + ["end"]
    kw = KEYWORDS.format(fc="FrozenCore" if frozen else "NoFrozenCore")
    body = [f"! {kw} Bohrs", "%pal nprocs 1 end", *basis_block, "* xyz 0 1"]
    body += [
        f"  {s:<2} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(symbols, coords)
    ]
    body.append("*")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n")
    return path


def parse_engrad(text: str, natm: int) -> np.ndarray:
    """ORCA .engrad: comment-delimited sections; the gradient section is 3N
    numbers in Eh/bohr, atom-major (x1 y1 z1 x2 ...)."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "The current gradient in Eh/bohr" in ln:
            vals = []
            j = i + 1
            while len(vals) < 3 * natm:
                t = lines[j].strip()
                j += 1
                if t.startswith("#") or not t:
                    continue
                vals.append(float(t))
            return np.array(vals).reshape(natm, 3)
    raise RuntimeError("no gradient section in .engrad")


def parse_orca_mp2(text: str) -> dict:
    import re

    out = {}
    m = re.findall(r"Total Energy\s*:\s*(-?\d+\.\d+)\s*Eh", text)
    if m:
        out["e_rhf"] = float(m[0])  # first "Total Energy" = SCF block
    m = re.findall(r"RI-MP2 CORRELATION ENERGY:\s*(-?\d+\.\d+)", text)
    if m:
        out["e_corr"] = float(m[-1])
    m = re.findall(r"Dimension of the orbital basis\s*\.+\s*(\d+)", text)
    if m:
        out["nao"] = int(m[0])
    m = re.findall(r"Dimension of the AuxC basis\s*\.+\s*(\d+)", text)
    if m:
        out["naux"] = int(m[0])
    return out


def run_orca(inp: Path, natm: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="ferric-orca-rimp2grad-") as tmp:
        run_inp = Path(tmp) / inp.name
        shutil.copy(inp, run_inp)
        parsed = common.run_orca(run_inp)
        out_text = run_inp.with_suffix(".out").read_text()
        parsed.update(parse_orca_mp2(out_text))
        # The printed RI-MP2 correlation energy has 9 decimals; the difference
        # of the two 12-14-decimal totals is the precise one. Both are stored.
        parsed["e_corr_printed"] = parsed["e_corr"]
        parsed["e_corr"] = parsed["energy"] - parsed["e_rhf"]
        if abs(parsed["e_corr"] - parsed["e_corr_printed"]) > 2e-9:
            raise RuntimeError(f"{inp.name}: E_total - E_rhf != printed E_corr")
        parsed["gradient"] = parse_engrad(
            run_inp.with_suffix(".engrad").read_text(), natm
        )
    return parsed


# ---------------------------------------------------------------------------
# PySCF
# ---------------------------------------------------------------------------


def pyscf_mol(symbols, coords_bohr, basis):
    from pyscf import gto

    bas, cart = common.pyscf_basis(basis, symbols)
    assert not cart
    return gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords_bohr),
        unit="Bohr",
        basis=bas,
        cart=False,
        verbose=0,
    )


def pyscf_dfmp2(symbols, coords_bohr, basis, aux, dm0=None, frozen=None):
    """(e_rhf, e_corr, dm, mf) for exact-J/K RHF + DF-MP2 with ferric's aux."""
    from pyscf import df, scf
    from pyscf.mp import dfmp2

    mol = pyscf_mol(symbols, coords_bohr, basis)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge")
    aux_bas, aux_cart = common.pyscf_basis(aux, symbols)
    assert not aux_cart
    mp = dfmp2.DFMP2(mf, frozen=frozen)
    mp.with_df = df.DF(mol)
    mp.with_df.auxmol = df.addons.make_auxmol(mol, aux_bas)
    mp.with_df.auxmol.cart = False
    mp.with_df.auxmol.build()
    mp.with_df.auxbasis = aux_bas
    mp.kernel()
    return mf.e_tot, mp.e_corr, mf.make_rdm1(), mf, mp.with_df.auxmol.nao_nr()


def pyscf_fd_gradient(symbols, coords, basis, aux, dm0) -> np.ndarray:
    c0 = np.array(coords)
    g = np.zeros_like(c0)
    stencil = [(-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12)]
    for a in range(len(symbols)):
        for k in range(3):
            acc = 0.0
            for step, w in stencil:
                c = c0.copy()
                c[a, k] += step * H_FD
                e_rhf, e_corr, *_ = pyscf_dfmp2(symbols, c.tolist(), basis, aux, dm0)
                acc += w * (e_rhf + e_corr)
            g[a, k] = acc / H_FD
    return g


# ---------------------------------------------------------------------------


def main() -> int:
    args = sys.argv[1:]
    write_only = "--write-only" in args
    only = {a for a in args if not a.startswith("--")}
    written = []
    for system, (basis, aux, ncore) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        natm = len(symbols)
        inp = write_input(INP_DIR / f"{system}_{basis}.inp", xyz, basis, aux, False)
        inp_fc = write_input(
            INP_DIR / f"{system}_{basis}_fc.inp", xyz, basis, aux, True
        )
        if write_only:
            print(f"wrote {inp.relative_to(common.ROOT)} (+ _fc)")
            continue

        import pyscf

        orca = run_orca(inp, natm)
        orca_fc = run_orca(inp_fc, natm)

        e_rhf, e_corr, dm, mf, naux = pyscf_dfmp2(symbols, coords, basis, aux)
        g_rhf = mf.nuc_grad_method().kernel()
        mol = mf.mol
        e_nuc = mol.energy_nuc()

        # like-for-like gates
        d_rhf = abs(orca["e_rhf"] - e_rhf)
        d_corr = abs(orca["e_corr"] - e_corr)
        if orca.get("nao") != mol.nao_nr() or orca.get("naux") != naux:
            raise RuntimeError(
                f"{system}: ORCA nao/naux {orca.get('nao')}/{orca.get('naux')} "
                f"!= {mol.nao_nr()}/{naux}"
            )
        if d_rhf > TOL_RHF or d_corr > TOL_ECORR:
            raise RuntimeError(
                f"{system}: ORCA vs PySCF |dE_rhf| {d_rhf:.2e} |dE_corr| {d_corr:.2e}"
            )

        g_fd = pyscf_fd_gradient(symbols, coords, basis, aux, dm)
        d_cross = float(np.abs(orca["gradient"] - g_fd).max())
        if d_cross > CROSS_TOL:
            raise RuntimeError(f"{system}: ORCA vs PySCF-FD gradient {d_cross:.2e}")
        d_fc = float(np.abs(orca["gradient"] - orca_fc["gradient"]).max())
        d_hf = float(np.abs(orca["gradient"] - g_rhf).max())

        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": basis,
            "aux_basis": aux,
            "charge": 0,
            "multiplicity": 1,
            "nao": int(mol.nao_nr()),
            "naux": int(naux),
            "nuclear_repulsion": float(e_nuc),
            "orca": {
                "e_rhf": orca["e_rhf"],
                "e_corr": orca["e_corr"],
                "e_total": orca["energy"],
                "gradient": orca["gradient"].tolist(),
                "input": str(inp.relative_to(common.ROOT)),
            },
            "orca_frozen_core": {
                "n_frozen": ncore,
                "e_corr": orca_fc["e_corr"],
                "gradient": orca_fc["gradient"].tolist(),
                "input": str(inp_fc.relative_to(common.ROOT)),
            },
            "pyscf": {
                "e_rhf": float(e_rhf),
                "e_corr": float(e_corr),
                "e_total": float(e_rhf + e_corr),
                "gradient_fd": g_fd.tolist(),
                "fd": {"stencil": "5-point central", "step_bohr": H_FD},
                "rhf_gradient": np.asarray(g_rhf).tolist(),
            },
            "cross_check": {
                "orca_vs_pyscf_e_rhf": d_rhf,
                "orca_vs_pyscf_e_corr": d_corr,
                "orca_vs_pyscf_fd_gradient_max": d_cross,
                "orca_mp2_vs_orca_fc_gradient_max": d_fc,
                "orca_mp2_vs_rhf_gradient_max": d_hf,
            },
            "provenance": common.provenance(
                code=f"ORCA {orca.get('version', 'unknown')} + PySCF {pyscf.__version__}",
                version=f"ORCA {orca.get('version', 'unknown')}; PySCF {pyscf.__version__}",
                keywords={
                    "orca": inp.read_text(),
                    "orca_frozen_core": inp_fc.read_text(),
                    "pyscf": "scf.RHF (exact J/K, conv_tol 1e-12, conv_tol_grad 1e-10) "
                    "+ mp.dfmp2.DFMP2(frozen=None) with auxmol from ferric's aux JSON; "
                    f"gradient = 5-point central FD of E_total, h = {H_FD} Bohr",
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
                    "correlation_sha256": common.sha256_file(
                        common.basis_json_path(aux)
                    ),
                    "scf": "none (exact four-centre J/K: ORCA NoRI, PySCF RHF)",
                },
                frozen_core="none (ORCA NoFrozenCore; PySCF frozen=None)",
                scf_conv="ORCA ExtremeSCF (TolE 1e-14, TolG 1e-9, Thresh 3e-16); PySCF conv_tol 1e-12 conv_tol_grad 1e-10",
                stability=None,
                generator="scripts/validation/gen_rimp2_gradient.py",
                extra={"orca_binary": str(common.ORCA_BINARY)},
            ),
        }
        path = common.write_reference(ROW, system, basis, payload)
        written.append(path)
        print(
            f"{system:14s} {basis:8s} E_tot {orca['energy']:.10f} "
            f"|dE_rhf| {d_rhf:.1e} |dE_corr| {d_corr:.1e} "
            f"|g_orca-g_fd| {d_cross:.1e} |g-g_fc| {d_fc:.1e} |g-g_hf| {d_hf:.1e}"
        )
    print(f"GEN_RIMP2_GRADIENT_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
