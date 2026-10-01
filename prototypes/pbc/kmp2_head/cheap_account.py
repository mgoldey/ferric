"""k-MP2 q = 0 head: the ~3-5% residual after E3 (lattice-weighted head + first-order Fock head).
Cheap pass: ONLY the saved head_anomaly_data/*.npz (no integral builds, no SCF).

Fixture: H2/STO-3G, a = 6, bond along z (the E3 residual -7.0e-5 x n^-3 of FINDINGS "k-point MP2 q = 0 head anomaly").

What this script measures
  R0  re-derive E3 - E1 = (lattice quad - cubic quad) + F from the rows (anchor: FINDINGS -7.29e-5 x n^-3 at n = 3).
      NOTE E3 - E1 involves no E_inf: the residual is "theory minus the flat E1 plateau", so it stands or falls
      with E1 being flat (d).
  (b) OS / SS (direct / exchange) split of every head piece: L, quad (from the exact lambda-quadratic), F.
      Artifact hypothesis: the head is added only at q = ka - ki = 0 (V[x, :, x]); the exchange integral
      X_{ki,kj,ka} = V_{ki,kj,kb} carries it at kb = ki (q' = 0).  If the SS channel's head were mis-weighted the
      SS share of the residual would be O(1) (e.g. SS quad != 0).  Physical expectation: the quad (|h|^2) term is
      pure OS/direct at O(1/Nk) (V_head * X_head needs ki = kj = ka, weight 1/Nk^2); the SS channel only carries
      a linear-in-h term, whose cubic average is exact (l <= 2).  So SS cannot host an l = 4 mis-weighting.
  (c/e) off-diagonal (ov) element of the fixed-k Fock head dK in the MO basis per k.  If it is ZERO at every k,
      first-order orbital relaxation (e) is identically absent and the only O(1/Nk) orbital effect is the
      eigenvalue shift already in F.  Artifact hypothesis: a symmetry (mirror z, TRIM) forcing dK_ia = 0 would give
      exactly 0 at kz = 0 and at TRIM points but NOT at general kz; a bug (wrong C) would give O(1) everywhere.
"""

import sys

import numpy as np

sys.path.insert(0, "/home/matt/qc/ferric-pbc/reference/pbc")
from pbc_kpts import mesh_index, mp_mesh  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from run_kpts_anchor import H2_ATOMS  # noqa: E402

DATA = "/home/matt/qc/ferric-pbc/reference/pbc/head_anomaly_data"
PHI_Z = np.array([1.0, 1 / 3, -0.17159902, -0.34844273, -0.54823523])


def load(a, m, suf=""):
    z = np.load(f"{DATA}/head_anomaly_a{a:g}_n{m}{suf}.npz")
    return z, {k: v for k, v in zip(z["keys"], z["rows"])}


def st_from(z, cell, m):
    n = (m, m, m)
    ints, kpts = mp_mesh(cell, n)
    C = z["C"]
    return dict(
        V=z["V"],
        Kq=z["Kq"],
        Co=C[:, :, :1],
        Cv=C[:, :, 1:],
        n=n,
        ints=ints,
        Nk=len(kpts),
    )


def with_head(st, P, lam=1.0):
    """AO-level q = 0 head (same algebra as pbc_kcorr._accumulate at iq = 0, kof = identity)."""
    new = dict(st)
    V = st["V"].copy()
    Co, Cv = st["Co"], st["Cv"]
    P = np.sqrt(lam) * P
    Bov = np.einsum("jmi,jmng,jna->jiag", Co.conj(), P, Cv, optimize=True)
    Bvo = np.einsum("jma,jmng,jni->jaig", Cv.conj(), P, Co, optimize=True).conj()
    T = np.einsum("xiag,ybjg->xyiajb", Bov, Bvo, optimize=True)
    for x in range(st["Nk"]):
        V[x, :, x] += T[x]
    new["V"] = V
    return new


def kmp2_split(st, eo, ev):
    """(OS, SS) per cell; OS + SS = the closed-shell KMP2 (pbc_kcorr.kmp2 convention)."""
    V, n, ints, Nk = st["V"], st["n"], st["ints"], st["Nk"]
    ii = np.arange(Nk)
    KB = np.array(
        [
            [[mesh_index(n, ints[a] + ints[b] - ints[c]) for c in ii] for b in ii]
            for a in ii
        ]
    )
    eo, ev = np.array(eo), np.array(ev)
    X = V[ii[:, None, None], ii[None, :, None], KB].transpose(0, 1, 2, 3, 6, 5, 4)
    d = (
        eo[:, None, None, :, None, None, None]
        - ev[None, None, :, None, :, None, None]
        + eo[None, :, None, None, None, :, None]
        - ev[KB][:, :, :, None, None, None, :]
    )
    t = V.conj() / d
    return np.sum(t * V).real / Nk**3, np.sum(t * (V - X)).real / Nk**3


def fock_dK(cell, z, Nk):
    """Fixed-k dK (AO) per k, as run_kcorr_head_anomaly.fock_head_shifts, and its MO matrix."""
    A, B, P0, C = z["A"], z["B"], z["P0"], z["C"]
    pref = 4 * np.pi / (cell.vol * Nk)
    out = []
    for k in range(Nk):
        dm = 2 * C[k][:, :1] @ C[k][:, :1].conj().T
        S = P0[k]
        M = np.zeros((2,) * 4, complex)
        for ax in range(3):
            M += np.einsum("ml,ns->mlns", A[k, ..., ax], A[k, ..., ax].conj())
            M += 0.5 * (
                np.einsum("ml,ns->mlns", B[k, ..., ax], S.conj())
                + np.einsum("ml,ns->mlns", S, B[k, ..., ax].conj())
            )
        M /= 3
        dK = pref * np.einsum("mlns,ls->mn", M, dm)
        out.append(C[k].conj().T @ dK @ C[k])
    return np.array(out)


def main():
    a = 6.0
    cell = Cell(np.eye(3) * a, H2_ATOMS, "sto-3g")
    for m in (3, 4):
        z, row = load(a, m)
        st = st_from(z, cell, m)
        Nk, n3 = st["Nk"], m**3
        eps, vm = z["eps"], float(z["vm"])
        eo = [e[:1] - vm for e in eps]
        ev = [e[1:] for e in eps]
        eoF = [x - y for x, y in zip(eo, z["deo"])]
        evF = [x - y for x, y in zip(ev, z["dev"])]
        cub = z["A"] * np.sqrt(4 * np.pi / (3 * cell.vol))
        S = np.array(kmp2_split(st, eo, ev))
        assert abs(S.sum() - row["sh"][0]) < 1e-14, (S.sum(), row["sh"][0])
        # exact quadratic in lambda: E(l) = S + l L3 + l^2 Q3 (cubic head at l = 1)
        e1 = np.array(kmp2_split(with_head(st, cub, 1.0), eo, ev)) - S
        e2 = np.array(kmp2_split(with_head(st, cub, 2.0), eo, ev)) - S
        Qc = (e2 - 2 * e1) / 2  # cubic-head quadratic part (t = 1/3)
        Lc = e1 - Qc
        # lattice quad: quadratic coefficient in t is Qc * 9; lattice weight Phi(u_z^4)
        Qlat = Qc * 9 * PHI_Z[2]
        F = np.array(kmp2_split(st, eoF, evF)) - S
        assert abs(e1.sum() - (row["cub1"][0] - row["sh"][0])) < 1e-14
        assert abs(F.sum() - (row["fock"][0] - row["sh"][0])) < 1e-14
        res = (Qlat - Qc) + F
        print(f"a=6 z n={m}  (x n^3)       {'OS':>12s} {'SS':>12s} {'total':>12s}")
        for name, v in (
            ("L (cubic, linear)", Lc),
            ("quad cubic", Qc),
            ("quad lattice", Qlat),
            ("F (Fock head)", F),
            ("E3-E1 = dquad+F", res),
        ):
            print(f"  {name:22s} {v[0] * n3:+.4e} {v[1] * n3:+.4e} {v.sum() * n3:+.4e}")
        print(f"  S itself (not x n^3): OS {S[0]:.10e} SS {S[1]:.10e}")
        # (c/e) off-diagonal Fock head in MO basis
        dKmo = fock_dK(cell, z, Nk)
        ints, kpts = mp_mesh(cell, (m, m, m))
        off = abs(dKmo[:, 0, 1])
        gap = eps[:, 1] - eps[:, 0]
        print(
            f"  dK MO diag occ k-avg {dKmo[:, 0, 0].real.mean() * n3:+.4e} vir {dKmo[:, 1, 1].real.mean() * n3:+.4e} (x n^3)"
        )
        print(
            f"  |dK_ov| max {off.max() * n3:.3e} k-avg {off.mean() * n3:.3e} (x n^3); first-order |kappa| max "
            f"{(0.5 * off / gap).max() * n3:.3e} (x n^3)"
        )
        for k in np.argsort(-off)[:4]:
            print(
                f"     k-int {ints[k]} |dK_ov| x n^3 {off[k] * n3:.3e}  gap {gap[k]:.4f}"
            )
        kz0 = [k for k in range(Nk) if ints[k][2] == 0]
        print(f"  max |dK_ov| on kz = 0 plane {off[kz0].max():.1e}")


if __name__ == "__main__":
    main()
