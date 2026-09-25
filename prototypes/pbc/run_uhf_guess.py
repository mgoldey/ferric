"""Stage-4 SCF-convergence / guess study at Gamma (open shell).

Claim under test (derived, then measured): at Gamma, v_M S D_s S = v_M x (occupied projector of spin s),
so exxdiv='ewald' and exxdiv=None have IDENTICAL stationary densities and energies differing by the
constant -v_M (N_a+N_b)/2.  But ewald lowers every occupied level by v_M, so a state that is
NON-aufbau under None (a hole below the Fermi level) can be aufbau-self-consistent under ewald and trap
an ewald SCF.  Necessary condition for the HF minimum (single-swap second variation, positive kernel):
per-spin ewald gap eps_LUMO - eps_HOMO >= v_M.
Strategy tested: converge with exxdiv=None, then switch Madelung on (D unchanged, converges in 1-2 it).
"""

import sys
import time

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals  # noqa: E402
from pbc_uhf import dense_jk, uhf  # noqa: E402
from test_prototype import SP_BASIS, TRI_A, TRI_ATOMS  # noqa: E402


def gaps(u, na, nb):
    return u["eps_a"][na] - u["eps_a"][na - 1], (
        u["eps_b"][nb] - u["eps_b"][nb - 1]
    ) if nb else np.inf


def show(tag, u, na, nb, vm):
    ga, gb = gaps(u, na, nb)
    print(
        f"   {tag:38s} E {u['e']:.12f}  <S2> {u['s2']:.8f}  gap_a {ga:.4f} gap_b {gb:.4f}"
        f"  {'min gap < v_M: TRAPPED' if min(ga, gb) < vm else ''}  it {u['it']}",
        flush=True,
    )


t0 = time.time()
print("1. tri 4H s+p triplet (na 3, nb 1)")
I = build_integrals(
    Cell(TRI_A, TRI_ATOMS, SP_BASIS), None, exxdiv="ewald", verbose=False
)
vm = I["madelung"]
args = (I["S"], I["h"], I["I"], I["enn"])
un = uhf(*args, 3, 1, conv=1e-12)
show("none, core guess", un, 3, 1, 0.0)
ue_core = uhf(*args, 3, 1, conv=1e-12, kshift=vm)
show("ewald, core guess", ue_core, 3, 1, vm)
ue_n = uhf(*args, 3, 1, conv=1e-12, kshift=vm, guess=(un["Da"], un["Db"]))
show("ewald, from converged none", ue_n, 3, 1, vm)
print(
    f"   (ewald-from-none) - none = {ue_n['e'] - un['e']:+.12f}  vs -v_M N/2 {-vm * 2:+.12f}; "
    f"its D vs none D {max(abs(ue_n['Da'] - un['Da']).max(), abs(ue_n['Db'] - un['Db']).max()):.1e}"
)
# the trapped ewald state, evaluated under None: a stationary point with a hole below the Fermi level
Da, Db = ue_core["Da"], ue_core["Db"]
jk = dense_jk(I["I"])
Jt = jk(Da + Db)[0]
Fa, Fb = I["h"] + Jt - jk(Da)[1], I["h"] + Jt - jk(Db)[1]
e_t = 0.5 * (np.sum(Da * (I["h"] + Fa)) + np.sum(Db * (I["h"] + Fb))) + I["enn"]
comm = max(abs(F @ D @ I["S"] - I["S"] @ D @ F).max() for F, D in ((Fa, Da), (Fb, Db)))
print(
    f"   trapped state under None: E {e_t:.12f} (ewald E + v_M N/2 - this: "
    f"{ue_core['e'] + 2 * vm - e_t:+.1e}; |[F,D]| {comm:.1e} -> also stationary under None); eps_a none {np.round(ue_core['eps_a'][:4] + np.r_[vm, vm, vm, 0], 5)}"
    f"  (occ, occ, occ | vir) -> hole below Fermi level"
)
for lab, g in [("none, beta HOMO/LUMO mix 0.3", 0.3), ("ewald, beta mix 0.3", 0.3)]:
    u = uhf(*args, 3, 1, conv=1e-12, mix=g, kshift=vm if "ewald" in lab else 0.0)
    show(lab, u, 3, 1, vm if "ewald" in lab else 0.0)

print("2. symmetry breaking of closed shells (singlet, na = nb)")
for lab, (cell_args, nel) in {
    "tri 4H s+p": ((TRI_A, TRI_ATOMS, SP_BASIS), 4),
    "stretched H2 R=4 in a=10": (
        (np.eye(3) * 10.0, [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 4.1))], "sto-3g"),
        2,
    ),
}.items():
    J = (
        I
        if lab == "tri 4H s+p"
        else build_integrals(
            Cell(*cell_args),
            1.0 * 0.8,
            rcut_bra=18.0,
            rcut_2e=18.0 + 6.0 / 0.8,
            exxdiv="ewald",
            verbose=False,
        )
    )
    a2 = (J["S"], J["h"], J["I"], J["enn"])
    n = nel // 2
    for ex in ("none", "ewald"):
        k = J["madelung"] if ex == "ewald" else 0.0
        u0 = uhf(*a2, n, n, conv=1e-12, kshift=k)
        show(f"{lab} {ex}, no mix (stays RHF)", u0, n, n, k)
        for mix in (0.3, 0.7):
            u1 = uhf(*a2, n, n, conv=1e-12, kshift=k, mix=mix, maxiter=500)
            show(f"{lab} {ex}, beta mix {mix}", u1, n, n, k)

print("3. O2/STO-3G triplet in a=12 box: core guess vs molecular-UHF density")
atoms = [("O", (0.3, 0.2, 0.1)), ("O", (0.3, 0.2, 2.382))]
mol = gto.M(atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0, spin=2)
mf = scf.UHF(mol)
mf.conv_tol = 1e-13
mf.kernel()
J = build_integrals(
    Cell(np.eye(3) * 12.0, atoms, "sto-3g"),
    8.0 / 12,
    rcut_bra=18.0,
    rcut_2e=18.0 + 6.0 / (8.0 / 12),
    exxdiv="ewald",
    verbose=False,
)
a2 = (J["S"], J["h"], J["I"], J["enn"])
k = J["madelung"]
for ex in ("none", "ewald"):
    kk = k if ex == "ewald" else 0.0
    for lab, kw in [
        ("core", {}),
        ("molecular UHF D", {"guess": mf.make_rdm1()}),
        ("core, level shift 0.5", {"level_shift": 0.5}),
    ]:
        try:
            u = uhf(*a2, 9, 7, conv=1e-12, kshift=kk, maxiter=300, **kw)
            show(f"O2 {ex}, {lab}", u, 9, 7, kk)
        except RuntimeError as exc:
            print(f"   O2 {ex}, {lab}: {exc}")
print(f"done [{time.time() - t0:.0f}s]")
