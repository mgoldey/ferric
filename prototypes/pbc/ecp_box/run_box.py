"""ECP periodic force big-box limit, ferric on both sides (see FINDINGS 2026-09-27).
Usage: python run_box.py mol | box a1 a2 ...

System: HI, H STO-3G + I LANL2DZ + LANL2DZ ECP (Iteration 22 MOL, on z, Bohr), cubic box a, exxdiv ewald,
jk dense (pure AFT, default precision 1e-14). Molecular side: Richardson central FD (h 2e-3 / 1e-3) of ferric's
molecular RHF energy (exact J/K), same BSE-JSON basis + ECP. Both sides are ferric.

PREDICTIONS (written 2026-09-27 BEFORE the sweep; c3, c3' from the PySCF molecular reference via
pbc_ecp.molecular_reference / pbc_grad_ecp.c3_prime: c3 = -70.9556, c3'(H z) = +9.70080, c3'(I z) = -9.70080):
  energy  a^3 (E_box - E_mol) -> c3 = -70.956 as a -> inf.
  force   a^3 (g_box - g_mol)_Hz -> c3' = +9.7008; g_I = -g_H (|sum F| ~1e-13); transverse = 0 by the axis
          reflection symmetry of the cubic lattice (numerical floor ~1e-8).
  fit     d(a) = c0 + c3'/a^3 + c5'/a^5 over a >= 20 (image-exchange overlap exp(-0.1053 a^2/2) is 5e-4 at a = 12,
          1.4e-6 at 16, 7e-10 at 20: a <= 16 is excluded from the fit):
          physics  -> fitted c3' within ~1% of 9.7008 and c0 at the combined floor (periodic force vs FD 1.3e-8
                      LANL2DZ, molecular FD ~1e-9): |c0| <~ 2e-8.
          artifact -> an ECP-term error is short range, hence a-INDEPENDENT: |c0| >> 2e-8 while a^3*d drifts like
                      c0 a^3. Positive control: the libecpint backend (derivative vs value inconsistency ~1e-7, FINDINGS
                      "ECP quadrature is the default") should show |c0| ~ 1e-7; the quadrature backend should not.
"""

import sys, time, json, os
import numpy as np
import ferric as F
from basis import write

B2A = 0.52917721092
MOL = [("H", (0.0, 0.0, 0.0)), ("I", (0.0, 0.0, 3.04))]  # Bohr (Iteration 22 MOL)
HERE = os.path.dirname(os.path.abspath(__file__))
BS = F.BasisSet.from_bse_json(write(os.path.join(HERE, "hi-lanl2dz.json")))


def xyz(atoms):
    return (
        "\n".join(
            [str(len(atoms)), "HI"]
            + [f"{s} {x * B2A!r} {y * B2A!r} {z * B2A!r}" for s, (x, y, z) in atoms]
        )
        + "\n"
    )


def emol(atoms):
    r = F.run_rhf(
        F.Molecule.from_xyz_string(xyz(atoms)),
        BS,
        df_j_aux="",
        df_k_aux="",
        energy_conv=1e-12,
        density_conv=1e-10,
        integral_thresh=1e-14,
    )
    assert r.converged if hasattr(r, "converged") else True
    return r.energy if hasattr(r, "energy") else r.total_energy


def disp(A, k, h):
    return [
        (s, tuple(np.array(p) + (h * np.eye(3)[k] if i == A else 0)))
        for i, (s, p) in enumerate(MOL)
    ]


def mol():
    t = time.time()
    e0 = emol(MOL)
    out = {"e": e0}
    g = np.zeros((2, 3))
    for A in range(2):
        for k in range(3):
            fd = {}
            for h in (2e-3, 1e-3):
                fd[h] = (emol(disp(A, k, h)) - emol(disp(A, k, -h))) / (2 * h)
            rich = (4 * fd[1e-3] - fd[2e-3]) / 3
            g[A, k] = rich
            print(
                f"mol FD A{A} k{k}: h2e-3 {fd[2e-3]:+.12f} h1e-3 {fd[1e-3]:+.12f} Rich {rich:+.12f} "
                f"(h1e-3 - Rich {fd[1e-3] - rich:+.1e})",
                flush=True,
            )
    out["g"] = g.tolist()
    print(
        f"E_mol {e0:.12f}; g_mol (Richardson FD)\n{g}\n sum {g.sum(0)} ({time.time() - t:.0f}s)",
        flush=True,
    )
    json.dump(out, open(os.path.join(HERE, "mol.json"), "w"))


def box(alist):
    ref = json.load(open(os.path.join(HERE, "mol.json")))
    gm = np.array(ref["g"])
    for a in alist:
        t = time.time()
        shift = np.array([0.37, 0.21, 0.5 * a - 1.52])
        atoms = [(s, tuple(np.array(p) + shift)) for s, p in MOL]
        lat = (np.eye(3) * a * B2A).tolist()
        r = F.run_rhf_gamma(
            F.Molecule.from_xyz_string(xyz(atoms)),
            lat,
            BS,
            exxdiv="ewald",
            jk="dense",
            with_gradient=True,
            density_conv=1e-10,
            max_eri_gb=1.0,
        )
        g = np.array(r.gradient())
        d = g - gm
        rec = dict(
            a=a,
            e=r.energy,
            conv=r.converged,
            iters=r.iterations,
            nG=r.n_g_half,
            g=g.tolist(),
            madelung=r.madelung,
            wall=time.time() - t,
        )
        print(
            f"a {a:5.1f}: conv {r.converged} it {r.iterations} nG {r.n_g_half} E-E_mol {r.energy - ref['e']:+.9e} "
            f"a^3(E-E_mol) {(r.energy - ref['e']) * a**3:+.6f}\n  g_box\n{g}\n  g_box-g_mol H z {d[0, 2]:+.6e} I z {d[1, 2]:+.6e} "
            f"a^3*dHz {d[0, 2] * a**3:+.6f}; transverse max|g| {abs(g[:, :2]).max():.1e}; |sum F| {abs(g.sum(0)).max():.1e} "
            f"({rec['wall']:.0f}s)",
            flush=True,
        )
        with open(os.path.join(HERE, "box.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    {"mol": lambda: mol(), "box": lambda: box([float(x) for x in sys.argv[2:]])}[
        sys.argv[1]
    ]()
