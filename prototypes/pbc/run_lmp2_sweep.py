"""Stage 8b truncation sweep: Gamma LMP2 on 1 x 1 x N supercells ("needles") of an H2 molecular
crystal (primitive cubic a0 = 7 Bohr, one tilted H2 per cell, 6-31G, cc-pvdz-ri spherical aux,
RS-GDF w = 0.5, shifted denominators).  Written, including the predictions below, BEFORE the sweep
was run (2026-09-24).

HYPOTHESES (stated before measuring)
Physics P1 (the claim under test): past an onset size, kept occupied pairs grow ~linearly in N
  (partners per molecule saturate) and the error at fixed eps is size-intensive per molecule.
Physics P2 (derived here, specific to Gamma): the periodic kernel gives EVERY pair a
  distance-independent coupling.  For transition densities rho_ia (neutral, dipole mu_ia) the
  G_par = 0 Fourier components of the image lattice of j form periodic dipole sheets whose field is
  uniform: (ia|jb) -> -(4 pi / Omega_sc) mu_ia,z mu_jb,z  for pairs separated by d >> a_lat/2pi
  (G_par != 0 terms decay as exp(-2 pi d / a_lat), a_lat = 7: e-fold length 1.1 Bohr).  This is
  the same 1/Omega tinfoil term that gave the a^-3 box law in Iterations 1/3/4.  Consequences:
  (a) the Eq-8 integral mask keeps ALL pairs while (4 pi/Omega_sc) mu*^2 > eps, i.e. for
      N < N*(eps) = 4 pi mu*^2 / (Omega_prim eps), mu* = max_{i,a} |mu_ia,z|; partners/molecule = N
      there (quadratic kept-pair growth), then drop to a near-field constant k_near(eps);
  (b) far-pair energies e_ij ~ -2 sum_ab (4 pi mu_ia mu_jb / Omega_sc)^2 / Delta ~ N^-2, and there
      are ~N^2 of them: their total is O(1), so dropping them costs O(1) in total, O(1/N) per
      molecule -> per-molecule error = e_near(eps) + b/N past the onset (fit the tail with 1/N);
  (c) with a minimum-image distance cutoff R_c the onset is geometric instead: partners =
      2 floor(R_c/a0) + 1 once N a0 > 2 R_c, per-molecule error = e_near(R_c) + b'/N.
  The molecular-limit picture (dipole-dipole R^-3 integrals) would instead predict an eps onset
  at N a0 ~ 2 (mu*^2/eps)^(1/3) -- N ~ 2 for eps=1e-3 vs N* ~ 10 from P2: distinguishable.
Artifact hypotheses:
  A1 non-periodic localisation / non-minimum-image distances -> equivalent molecules get different
     partner counts (partners min != max, kept pairs not a multiple of N) and unequal pair energies:
     measured every row (partners min/max) + run_lmp2_translation.py.
  A2 translation symmetry makes E_err exactly N x (per-molecule error) at EVERY N, so
     "extensive" is automatic and is NOT evidence.  Evidence = saturation of partners(N) and of
     E_err/N with N, at the predicted onset.
  A3 below the onset every pair is kept (error ~0, fraction ~1): no locality statement is possible
     there, positive or negative.
Usage: python3 run_lmp2_sweep.py N1 N2 ...   (prints one block per N)
"""

import sys
import time

import numpy as np

import pbc_lmp2 as L
from pbc_supercell import Supercell, build_supercell
from run_lmp2_anchor import A0, AUX, H2_ATOMS

EPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
RCUT = (3.5, 10.5, 17.5)


def far_pair_check(p, d, r0, sc):
    """Measured (ia|jb) and e_ij for the farthest pairs vs the P2 uniform-field prediction."""
    mu = L.transition_dipoles_z(p, d, axis=2)  # (nocc, nvir)
    vol = sc.sc.vol
    dist = L.pair_distances(p)
    J = p["J"]
    no = J.shape[0]
    far = [(i, j) for i in range(no) for j in range(no) if i != j and dist[i, j] > 13.9]
    if not far:
        return mu, None
    fo = np.diag(p["Foo"])
    lv, Vv = np.linalg.eigh(p["Fvv"])
    rel, emeas, epred = [], [], []
    for i, j in far:
        pred = -(4 * np.pi / vol) * np.outer(mu[i], mu[j])
        rel.append(abs(J[i, :, j, :] - pred).max() / abs(J[i, :, j, :]).max())
        # direct-term pair energy of the uniform block with the FULL non-canonical Fvv (Sylvester via
        # the eigenbasis of Fvv; Foo off-diagonal and the exchange-type (ib|ja) ~ 0 neglected)
        Pp = Vv.T @ pred @ Vv
        epred.append(-2 * np.sum(Pp**2 / (lv[:, None] + lv[None, :] - fo[i] - fo[j])))
        emeas.append(r0["epair"][i, j])
    dmax = dist.max()
    fj = [(i, j) for i, j in far if abs(dist[i, j] - dmax) < 1e-6]
    i, j = fj[0]
    return mu, dict(
        nfar=len(far),
        relJ_max=max(rel),
        relJ_farthest=rel[far.index((i, j))],
        Jmax_farthest=abs(J[i, :, j, :]).max(),
        sum_e_far=float(np.sum(emeas)),
        sum_e_far_pred=float(np.sum(epred)),
    )


def run(N, ragged=False, eps_list=EPS):
    """ragged=True (large N): skip the dense eps=0 anchor/far-pair energies, solve eps>0 and R_c rows with
    the ragged per-pair solver (exact same masked problem; cross-checked vs dense by --xcheck)."""
    t0 = time.time()
    sc = Supercell(A0, H2_ATOMS, "6-31g", (1, 1, N))
    d = build_supercell(sc, AUX, w=0.5)
    t1 = time.time()
    scf = L.gamma_scf(d, 2 * N)
    ec = L.canonical_mp2(scf, d["B"])
    p = L.prepare(sc, d, scf)
    r0 = L.solve(p, 0.0) if not ragged else dict(e=ec, epair=np.full((N, N), np.nan))
    mu, fc = far_pair_check(p, d, r0, sc)
    solver = "ragged" if ragged else "dense"
    mustar = abs(mu).max()
    print(
        f"== N={N}: nao {d['S'].shape[0]} naux {d['info']['naux']} (kept {d['info']['naux_kept']}) nG {d['info']['nG']}  "
        f"build {t1 - t0:.0f}s  E_HF/N {scf['e'] / N:.10f}  E_can {ec:.10e}  E_can/N {ec / N:.10e}  "
        f"anchor dE {'skipped' if ragged else format(r0['e'] - ec, '+.1e')}  Berghold |grad| {p['loc']['grad']:.1e}",
        flush=True,
    )
    print(
        f"   mu*_z {mustar:.5f}  N*(eps) = 4 pi mu*^2/(Omega_p eps): "
        + " ".join(f"{e:g}:{4 * np.pi * mustar**2 / (343.0 * e):.1f}" for e in EPS)
    )
    if fc:
        print(
            f"   far pairs (d>13.9): n {fc['nfar']}  max rel |J - J_uniform| {fc['relJ_max']:.2e} (farthest pair "
            f"{fc['relJ_farthest']:.2e}, max|J| {fc['Jmax_farthest']:.3e})  sum e_ij(far) {fc['sum_e_far']:.4e} "
            f"pred {fc['sum_e_far_pred']:.4e}"
        )
    for eps in eps_list:
        r = L.solve(p, eps, solver=solver)
        print(
            f"   eps {eps:<7g} dE {r['e'] - ec:+.4e}  dE/N {(r['e'] - ec) / N:+.4e}  keep {r['keep']:.4f}  pairs "
            f"{r['pairs_kept']:5d}/{N * N}  partners {r['partners'].min()}/{r['partners'].max()}  cg {r['niter']}",
            flush=True,
        )
    for rc in RCUT:
        r = L.solve(p, 0.0, pair_cut=rc, solver=solver)
        print(
            f"   Rc  {rc:<7g} dE {r['e'] - ec:+.4e}  dE/N {(r['e'] - ec) / N:+.4e}  keep {r['keep']:.4f}  pairs "
            f"{r['pairs_kept']:5d}/{N * N}  partners {r['partners'].min()}/{r['partners'].max()}  cg {r['niter']}",
            flush=True,
        )
    print(f"   [{time.time() - t0:.0f} s]", flush=True)


def xcheck(N, settings=((1e-3, None), (0.0, 10.5))):
    """Ragged vs dense solver on identical masks (must agree to CG tolerance)."""
    sc = Supercell(A0, H2_ATOMS, "6-31g", (1, 1, N))
    d = build_supercell(sc, AUX, w=0.5)
    p = L.prepare(sc, d, L.gamma_scf(d, 2 * N))
    for eps, rc in settings:
        a, b = (
            L.solve(p, eps, pair_cut=rc),
            L.solve(p, eps, pair_cut=rc, solver="ragged"),
        )
        print(
            f"   xcheck N={N} eps={eps:g} Rc={rc}: |E_dense - E_ragged| {abs(a['e'] - b['e']):.1e}",
            flush=True,
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--xcheck":
        xcheck(int(args[1]))
    elif args and args[0] == "--ragged":
        for N in [int(x) for x in args[1:]]:
            run(N, ragged=True, eps_list=(1e-2, 3e-3, 1e-3))
    else:
        for N in [int(x) for x in args] or [2, 4, 6, 8, 12, 16]:
            run(N)
