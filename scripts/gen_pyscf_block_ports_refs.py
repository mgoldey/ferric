"""PySCF references for the three building-block ports in examples/python/.

The ports are ferric's Phase 1 exit test (wiki/moonshot-reference-code.md,
Pillar 2): published methods written in numpy on ferric's blocks only.

* ``drpa_from_blocks.py``: PySCF ``gw.rpa.RPA`` on an RHF reference
  (dRPA@HF, all electrons, cc-pVDZ / cc-pVDZ-RI, nw = 40, x0 = 0.5, the same
  Legendre nodes as the port).
* ``adc2_from_blocks.py``: PySCF ``adc.RADC`` with method "adc(2)" and
  method_type "ee" (singlet excitation energies, 6-31G, all electrons).
* ``embedding_from_blocks.py``: projection-based embedding (Manby et al.,
  JCTC 8, 2564 (2012)) written here with PySCF's own pieces: its RKS, its
  ``get_veff`` and an RHF/RKS on the charged subsystem whose core Hamiltonian
  carries v_emb. The A/B split is the same SPADE rule as the port, applied to
  PySCF's orbitals. PBE, cc-pVDZ, the heavy atom as subsystem A: mu = 1e6 on
  two flat grids (so a test can bound the grid error) and mu = 1e5 on the
  first (so the O(1/mu) leakage term can be extrapolated away).

Every orbital and auxiliary basis is ferric's bundled BSE JSON handed to
PySCF, so the comparison measures the methods, not two basis libraries.
PySCF orbitals are never passed to ferric: each code converges its own SCF.

Usage: python scripts/gen_pyscf_block_ports_refs.py
Writes testdata/reference/block_ports_pyscf.json
"""

import json
import os

import numpy as np
from pyscf import adc, df, dft, gto, scf
from pyscf.gw import rpa

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUNDLED = os.path.join(ROOT, "crates", "ferric-core", "src", "basis", "bundled")
OUT = os.path.join(ROOT, "testdata", "reference", "block_ports_pyscf.json")
SYMBOLS = {1: "H", 7: "N", 8: "O"}
MOLECULES = ("water", "nh3")
# (grid, mu) pairs: a grid change at mu = 1e6, and a mu change on the first grid.
EMBEDDING_RUNS = (((99, 590), 1e6), ((150, 974), 1e6), ((99, 590), 1e5))


def ferric_basis(name: str) -> dict:
    """Ferric's bundled BSE JSON as a PySCF per-element basis dict."""
    with open(os.path.join(BUNDLED, f"{name}.json")) as f:
        data = json.load(f)
    out = {}
    for z, sym in SYMBOLS.items():
        shells = []
        for sh in data["elements"][str(z)]["electron_shells"]:
            ams = sh["angular_momentum"]
            exps = [float(x) for x in sh["exponents"]]
            if len(ams) == 1:
                ams = ams * len(sh["coefficients"])
            for ang, coefs in zip(ams, sh["coefficients"]):
                c = [float(x) for x in coefs]
                prims = [(e, ci) for e, ci in zip(exps, c) if ci != 0.0]
                shells.append([ang, *prims])
        out[sym] = shells
    return out


def molecule(name: str, basis: str, charge: int = 0) -> gto.Mole:
    with open(os.path.join(ROOT, "testdata", "molecules", f"{name}.xyz")) as f:
        lines = f.read().splitlines()[2:]
    atom = "\n".join(line for line in lines if line.strip())
    return gto.M(
        atom=atom,
        basis=ferric_basis(basis),
        unit="Angstrom",
        charge=charge,
        verbose=0,
    )


def converged(mf, dm0=None, conv_tol_grad=1e-9):
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = 300
    mf.kernel(dm0=dm0)
    assert mf.converged, f"{type(mf).__name__} did not converge"
    return mf


def drpa(name: str) -> dict:
    mf = converged(scf.RHF(molecule(name, "cc-pvdz")))
    r = rpa.RPA(mf)
    r.with_df = df.DF(mf.mol, auxbasis=ferric_basis("cc-pvdz-ri"))
    e_corr = r.kernel(nw=40, x0=0.5)
    return {
        "basis": "cc-pvdz",
        "auxbasis": "cc-pvdz-ri",
        "nw": 40,
        "x0": 0.5,
        "rhf_energy": mf.e_tot,
        "e_corr": e_corr,
    }


# Residual bar for the ADC Davidson roots. For the symmetric ADC matrix the
# eigenvalue error is at most r^2 / gap; with the smallest root spacing here
# (6.4e-3 Ha) a 1e-6 residual leaves under 2e-10 Ha, far inside the 1e-6 Ha
# the port tests compare at. PySCF's own tol_residual (1e-8) is not reached by
# this solve (measured 3e-8 to 5.5e-8 at the returned Ritz pairs on water).
ADC_RESIDUAL_BAR = 1e-6


def davidson_residuals(run):
    """Call `run()` and return (its result, the residual norm of each Ritz pair
    of the ADC Davidson solve). PySCF's ADC `kernel()` warns and returns the
    current Ritz values when the solve hits its iteration limit, and the
    solver's own converged flags are not a usable check (all False on the
    shipped water/6-31G solve although the roots are stable to 1e-11), so the
    residual |A x - e x| is evaluated here from the returned Ritz pairs."""
    from pyscf.lib import linalg_helper

    residuals = []
    original = linalg_helper.davidson_nosym1

    def recording(aop, *args, **kwargs):
        _, e, xs = out = original(aop, *args, **kwargs)
        ax = aop([np.asarray(x) for x in xs])
        residuals.append(
            [float(np.linalg.norm(a - ek * x)) for a, ek, x in zip(ax, e, xs)]
        )
        return out

    linalg_helper.davidson_nosym1 = recording
    try:
        result = run()
    finally:
        linalg_helper.davidson_nosym1 = original
    return result, residuals


def adc2(name: str, nroots: int = 3) -> dict:
    mf = converged(scf.RHF(molecule(name, "6-31g")))
    a = adc.RADC(mf)
    a.method = "adc(2)"
    a.method_type = "ee"
    a.conv_tol = 1e-12
    a.tol_residual = 1e-8
    a.compute_properties = False
    out, residuals = davidson_residuals(lambda: a.kernel(nroots=nroots))
    assert len(residuals) == 1 and max(residuals[0]) < ADC_RESIDUAL_BAR, (
        f"ADC(2) Davidson residuals {residuals} exceed {ADC_RESIDUAL_BAR}"
    )
    e = out[0]
    return {
        "basis": "6-31g",
        "rhf_energy": mf.e_tot,
        "mp2_corr": a.e_corr,
        "excitation_energies": sorted(float(x) for x in e),
    }


def spade_split(mol, s, c_occ, atoms_a):
    """The port's SPADE rule on PySCF's orbitals and AO order."""
    w, u = np.linalg.eigh(s)
    s_half = (u * np.sqrt(w)) @ u.T
    rows = [mu for mu, lab in enumerate(mol.ao_labels(fmt=False)) if lab[0] in atoms_a]
    _, sv, vt = np.linalg.svd((s_half @ c_occ)[rows], full_matrices=True)
    n_a = int(np.argmax(-np.diff(sv))) + 1
    c = c_occ @ vt.T
    return c[:, :n_a], c[:, n_a:], sv


def rks(mol, xc, grid):
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.atom_grid = grid
    mf.grids.prune = None
    return mf


def embedding(name: str, grid, mu: float, atoms_a=(0,), low: str = "PBE") -> dict:
    mol = molecule(name, "cc-pvdz")
    full = converged(rks(mol, low, grid))
    nocc = mol.nelectron // 2
    s, h = full.get_ovlp(), full.get_hcore()
    c_a, c_b, sv = spade_split(mol, s, full.mo_coeff[:, :nocc], atoms_a)
    d_a = 2.0 * c_a @ c_a.T
    d = 2.0 * full.mo_coeff[:, :nocc] @ full.mo_coeff[:, :nocc].T

    def e_low(dm):
        v = full.get_veff(mol, dm)
        return float(np.sum(dm * h)) + v.ecoul + v.exc, np.asarray(v)

    e_full, v_full = e_low(d)
    e_sub, v_sub = e_low(d_a)
    p_b = s @ c_b @ c_b.T @ s
    v_emb = v_full - v_sub + 0.5 * mu * (p_b + p_b.T)
    mol_a = molecule(name, "cc-pvdz", charge=2 * c_b.shape[1])
    out = {
        "grid": list(grid),
        "low": low,
        "atoms_a": list(atoms_a),
        "mu": mu,
        "n_a": c_a.shape[1],
        "spade_sv": sv.tolist(),
        "e_low_full": full.e_tot,
    }
    for label, sub in (
        ("hf_in_low", scf.RHF(mol_a)),
        ("low_in_low", rks(mol_a, low, grid)),
    ):
        sub.get_hcore = lambda *args, **kwargs: h + v_emb
        # Start from gamma_A. The mu-shifted projector puts round-off of order
        # mu * 1e-16 into the orbital gradient, so the gradient threshold is
        # 1e-8; the energy error is quadratic in it.
        e_drv = converged(sub, dm0=d_a, conv_tol_grad=1e-8).e_tot
        out[label] = e_drv - float(np.sum(d_a * v_emb)) + e_full - e_sub
    return out


def main() -> None:
    import pyscf

    ref = {
        "generator": "scripts/gen_pyscf_block_ports_refs.py",
        "pyscf_version": pyscf.__version__,
        "molecules": {},
    }
    for name in MOLECULES:
        ref["molecules"][name] = {
            "drpa": drpa(name),
            "adc2": adc2(name),
            "embedding": [embedding(name, g, mu) for g, mu in EMBEDDING_RUNS],
        }
    with open(OUT, "w") as f:
        json.dump(ref, f, indent=2)
        f.write("\n")
    print(json.dumps(ref, indent=2))


if __name__ == "__main__":
    main()
