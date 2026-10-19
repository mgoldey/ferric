"""PySCF k-point HF FORCE pins for ferric's k-point forces (issue #421).

PySCF 2.13.1 has k-point RHF/UHF nuclear gradients (`pbc.grad.krhf`,
`pbc.grad.kuhf`) but no k-point HF stress. Two limits of those gradients:

* the J/K derivative (`with_df.get_jk_e1`) exists only for FFTDF;
* `pbc.grad.krhf.get_hcore` raises NotImplementedError unless the cell has a
  GTH pseudopotential (`cell._pseudo`). ferric has no GTH pseudopotentials
  (its ECPs are semilocal Gaussian ECPs), so no cell exists on which both codes
  run the same Hamiltonian unmodified.

This script therefore lifts that one gate: `get_hcore_all_electron` is
PySCF's own pseudo branch of `get_hcore` (int1e_ipkin + the local potential
on the FFT grid, bra derivative) run for an ALL-ELECTRON cell, for which
PySCF's `get_vlocG` is the bare `Z * coulG` (`pp_int.get_gth_vlocG_part1`
with no pseudo entry) and `get_alphas` is 0, i.e. exactly the potential
FFTDF's `get_nuc` puts into the SCF Hamiltonian (`rhoG * coulG`, G = 0
dropped). Nothing else is changed: the operator derivative
(`hcore_generator`), the J/K derivative (`get_jk_e1`), the Pulay term and
the Ewald `grad_nuc` are PySCF's. The patch is validated here against a
central finite difference of PySCF's OWN FFTDF energy (no ferric code
involved), printed per system.

Hydrogen cells only (all-electron FFTDF converges to machine precision at
mesh 41^3 in a 4 Bohr box for STO-3G H: the energy is unchanged at 61^3 and
equals the AFTDF energy to 5e-15).

Usage: OMP_NUM_THREADS=3 uv run --no-sync python scripts/gen_pyscf_kforce_pins.py
Prints the pins pasted into crates/ferric-pbc/tests/validation_kstress_coverage.rs
(section H).
"""

import sys

import numpy as np
import pyscf
from pyscf.pbc import df, gto, scf, tools
from pyscf.pbc.dft.numint import eval_ao_kpts
from pyscf.pbc.grad import krhf as gkrhf
from pyscf.pbc.gto.pseudo.pp import get_alphas, get_vlocG


def get_hcore_all_electron(cell, kpts):
    """PySCF's pseudo branch of pbc.grad.krhf.get_hcore, for any cell."""
    h1 = np.asarray(cell.pbc_intor("int1e_ipkin", kpts=kpts)).astype(complex)
    si = cell.get_SI()
    coords = cell.get_uniform_grids()
    vloc_g = -np.einsum("ij,ij->j", si, get_vlocG(cell))
    vloc_g[0] = np.sum(get_alphas(cell))
    vloc_r = tools.ifft(vloc_g, cell.mesh).real
    for kn, kpt in enumerate(kpts):
        aos = eval_ao_kpts(cell, coords, kpt, deriv=1)[0]
        h1[kn] += np.einsum("agi,g,gj->aij", aos[1:].conj(), vloc_r, aos[0])
    return h1


# `get_hcore_all_electron` copies PySCF internals (the pseudo branch of
# `pbc.grad.krhf.get_hcore`); re-read them before trusting another version.
assert pyscf.__version__ == "2.13.1", (
    f"pins were generated with PySCF 2.13.1, not {pyscf.__version__}"
)
gkrhf.get_hcore = get_hcore_all_electron

# PySCF's sto-3g for H (crates/ferric-pbc/tests/common: pyscf_sto3g_h) and the
# prototype's s+p basis (sp_basis_h: STO-3G s plus one Cartesian p(0.8)).
STO3G = [
    [0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]
]
SP = STO3G + [[1, [0.8, 1.0]]]
# pbc_grad.rs GRAD_H2_ATOMS (Bohr), cubic edge 4.
H2_ATOMS = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]]
MESH = 41
H = 1e-4


def make_cell(atoms, basis, spin, mesh=MESH):
    c = gto.Cell()
    c.atom = [["H", p] for p in atoms]
    c.unit = "B"
    c.a = np.eye(3) * 4.0
    c.basis = {"H": basis}
    c.cart = True
    c.precision = 1e-12
    c.mesh = [mesh] * 3
    c.spin = spin
    c.verbose = 0
    c.build()
    return c


def kpts_of(cell, nk, shifted):
    if shifted:
        return cell.make_kpts(nk, with_gamma_point=False)
    return cell.make_kpts(nk)


def solve(cell, nk, shifted, uhf, exxdiv, aft=False):
    kpts = kpts_of(cell, nk, shifted)
    mf = (scf.KUHF if uhf else scf.KRHF)(cell, kpts, exxdiv=exxdiv)
    if aft:
        mf.with_df = df.AFTDF(cell, kpts)
    assert aft or isinstance(mf.with_df, df.FFTDF), type(mf.with_df)
    if uhf:
        # ferric's k-UHF occupies the lowest n_alpha / n_beta bands at EVERY
        # k; PySCF's KUHF aufbau is global over k. Pin the per-k occupation
        # so both codes solve the same state.
        na, nb = cell.nelec

        def per_k_occ(mo_energy_kpts=None, mo_coeff_kpts=None):
            if mo_energy_kpts is None:
                mo_energy_kpts = mf.mo_energy
            occ = np.zeros_like(np.asarray(mo_energy_kpts))
            for s, n in enumerate((na, nb)):
                for k, eps in enumerate(mo_energy_kpts[s]):
                    occ[s][k][np.argsort(eps)[:n]] = 1.0
            return occ

        mf.get_occ = per_k_occ
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 300
    e = mf.kernel()
    if not mf.converged:
        # The s+p triplet k-UHF limit-cycles under plain DIIS; second-order
        # SCF from the DIIS end point converges it.
        mf = mf.newton()
        e = mf.kernel()
    assert mf.converged
    return e, mf


SYSTEMS = [
    # (name, basis, spin, nk, shifted, uhf, exxdiv)
    ("H2_RHF_113_NONE", STO3G, 0, [1, 1, 3], False, False, None),
    ("H2_RHF_113_EWALD", STO3G, 0, [1, 1, 3], False, False, "ewald"),
    ("H2_RHF_MP112_NONE", STO3G, 0, [1, 1, 2], True, False, None),
    ("H2_TRIPLET_SP_UHF_113_NONE", SP, 2, [1, 1, 3], False, True, None),
]


def heavy(s, p):
    """The heavy-atom s/p set of validation_kstress_coverage.rs `rot_basis`."""
    return [
        [0, [s[0], 0.3], [s[1], 0.5], [s[2], 0.4]],
        [0, [s[2], 1.0]],
        [1, [p[0], 0.4], [p[1], 0.8]],
        [1, [p[1], 1.0]],
    ]


ECP_BASIS = {
    "H": "cc-pvdz",
    "I": heavy([3.0, 1.0, 0.35], [2.0, 0.7]),
    "As": heavy([3.4, 1.2, 0.42], [2.2, 0.75]),
    "Sb": heavy([2.6, 0.9, 0.32], [1.6, 0.55]),
}
ECP_SYSTEMS = [
    # (name, atoms (Bohr), lattice diagonal, 2S)
    ("HI", [["H", [0.3, 0.2, 0.4]], ["I", [0.9, -0.4, 3.3]]], [6.0, 6.0, 7.0], 0),
    (
        "ASSB_SEPTET",
        [["As", [0.9, -0.4, 3.3]], ["Sb", [0.4, 0.3, 7.9]]],
        [6.0, 6.0, 10.0],
        6,
    ),
]


def ecp_energy_pins():
    """Gamma RHF/UHF AFTDF energies with LANL2DZ ECPs (exxdiv None):
    the external ENERGY pin of the two-ECP-element cell (no force: PySCF's
    k-point HF gradient has no semilocal-ECP derivative)."""
    for name, atoms, diag, spin in ECP_SYSTEMS:
        c = gto.Cell()
        c.atom = atoms
        c.unit = "B"
        c.a = np.diag(diag)
        c.basis = ECP_BASIS
        c.ecp = {"I": "lanl2dz", "As": "lanl2dz", "Sb": "lanl2dz"}
        c.spin = spin
        c.cart = True
        c.precision = 1e-12
        c.verbose = 0
        c.build()
        mf = (scf.UHF if spin else scf.RHF)(c, exxdiv=None)
        mf.with_df = df.AFTDF(c)
        mf.conv_tol = 1e-11
        mf.max_cycle = 300
        e = mf.kernel()
        assert mf.converged
        print(f"const PY_{name}_GAMMA_E: f64 = {e!r}; // AFTDF mesh {list(c.mesh)}")


def main(only=None):
    for name, basis, spin, nk, shifted, uhf, exxdiv in SYSTEMS:
        if only and name not in only:
            continue
        cell = make_cell(H2_ATOMS, basis, spin)
        e, mf = solve(cell, nk, shifted, uhf, exxdiv)
        g = mf.nuc_grad_method().kernel()
        e_aft, _ = solve(cell, nk, shifted, uhf, exxdiv, aft=True)
        # Self-check of the patched gradient against PySCF's own energy.
        fd = []
        for a, x in [(0, 0), (1, 2)]:
            es = []
            for s in (1, -1):
                p = [list(r) for r in H2_ATOMS]
                p[a][x] += s * H
                es.append(solve(make_cell(p, basis, spin), nk, shifted, uhf, exxdiv)[0])
            fd.append((a, x, (es[0] - es[1]) / (2 * H), g[a, x]))
        print(f"// {name}: E_fftdf {e:.12f}, E_aftdf - E_fftdf {e_aft - e:.2e}")
        for a, x, f, an in fd:
            print(
                f"//   PySCF FD vs patched analytic ({a},{x}): {f:.10f} vs {an:.10f} ({f - an:.2e})"
            )
        print(f"const {name}_E: f64 = {e!r};")
        rows = ", ".join("[" + ", ".join(repr(float(v)) for v in r) + "]" for r in g)
        print(f"const {name}_F: [[f64; 3]; 2] = [{rows}];")


if __name__ == "__main__":
    if sys.argv[1:] == ["ecp"]:
        ecp_energy_pins()
    else:
        main(sys.argv[1:])
