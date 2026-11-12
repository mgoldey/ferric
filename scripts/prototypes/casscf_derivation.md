# Two-step CASSCF: equations for the Rust port

This file lists the equations, intermediates and conventions that
`casscf_proto.py` implements and the Rust port needs, and records how each
was checked. Real orbitals, closed-shell-type (spin-summed) RDMs, Hartree,
chemist notation (pq|rs).

The CAS-CI step in the prototype is `pyscf.fci.direct_spin1` behind the
`PyscfFci.solve(h1, eri, ncas, nelecas, ci0) -> CiResult` interface. A
ferric-ci solver with RDMs (WP4) replaces it by implementing the same call
with the same RDM convention. Integrals and starting orbitals come from
ferric only (`AoIntegrals.hcore/jk/ao2mo`, `run_rhf`); PySCF orbitals are
never used by the prototype.

## Partition and parameters

MOs C (nao x nmo) split into core i, j (n_c), active t, u, v, w (n_a) and
virtual a (n_v). Indices p, q, r, s run over all MOs.

| Input | Meaning |
|---|---|
| `ncore` | doubly occupied inactive orbitals |
| `ncas`, `nelecas = (n_alpha, n_beta)` | active orbitals and electrons |
| starting C | ferric RHF canonical orbitals, active = the `ncas` orbitals above the core by energy |

## Energy

Spin-summed RDMs in PySCF's convention:

- gamma_tu = <E_tu>
- Gamma_tuvw = <a+_t a+_v a_w a_u> summed over spins, so that
  E_act = sum_tu h_tu gamma_tu + 1/2 sum_tuvw (tu|vw) Gamma_tuvw.

Symmetries for real wavefunctions: Gamma_tuvw = Gamma_vwtu = Gamma_utwv.

AO intermediates (one `jk` call each, `jk` takes the total density and
F = h + J - K/2):

| Symbol | Definition | Build |
|---|---|---|
| D^c | 2 C_c C_c^T | |
| F^I | h + J[D^c] - K[D^c]/2 | `jk(D^c)` |
| E_core | E_nuc + 1/2 Tr[D^c (h + F^I)] | |
| D^a | C_a gamma C_a^T | |
| F^A | J[D^a] - K[D^a]/2 | `jk(D^a)` |
| (pu\|vw) | p over all MOs, u, v, w active | `ao2mo(C, C_a, C_a, C_a)`, nmo n_a^3 |

CAS-CI Hamiltonian: h_eff = C_a^T F^I C_a, eri = (tu|vw) (the active rows of
(pu|vw)), constant E_core. E_CASCI = E_core + E_act.

## Generalized Fock and orbital gradient

Rotation C' = C exp(K), K antisymmetric, K_pq = kappa_pq for p > q. In the MO
basis h' = U^T h U with U = 1 + K to first order. With the generalized Fock
F_pq = sum_r h_pr D_rq + sum_rst (pr|st) Gamma_qrst (nonzero only for q
occupied):

    dE/dkappa_pq = 2 (F_pq - F_qp),   p > q.

Columns of F in terms of the intermediates (MO basis, F^I and F^A
transformed with C):

| Column | Value |
|---|---|
| core i | F_pi = 2 (F^I_pi + F^A_pi) |
| active t | F_pt = sum_u F^I_pu gamma_ut + sum_uvw (pu\|vw) Gamma_tuvw |
| virtual a | 0 |

Nonredundant pairs (p > q) and the gradient element of each class:

| Class | Pair | g |
|---|---|---|
| core-active | (t, i) | 2 (F_ti - F_it) |
| core-virtual | (a, i) | 4 (F^I_ai + F^A_ai) |
| active-virtual | (a, t) | 2 F_at |

Active-active and core-core rotations are redundant for CASSCF and are
excluded. At a converged CI the active-active block of F - F^T is zero, which
the full-space anchor checks.

## Approximate diagonal Hessian (preconditioner)

With F^T = F^I + F^A in the MO basis and F the generalized Fock above:

| Class | H_diag |
|---|---|
| (a, i) | 4 (F^T_aa - F^T_ii) |
| (a, t) | 2 gamma_tt F^T_aa - 2 F_tt |
| (t, i) | 4 (F^T_tt - F^T_ii) + 2 gamma_tt F^T_ii - 2 F_tt |

This form follows Chaban, Schmidt and Gordon, Theor. Chem. Acc. 97, 88
(1997). The prototype uses it only as the initial inverse Hessian of BFGS,
with |H_diag| floored at 0.05. Unverified as a Hessian: its agreement with
the finite-difference diagonal is reported in the table below and is not
an anchor.

## Optimizer (prototype choice)

Two-step: each macro iteration converges the CI, builds g, takes
s = -B^-1 g with an inverse-BFGS update of B^-1 in the current orbital frame
(no parallel transport of earlier steps), caps max |s_k| at 0.2, and rotates
C <- C expm(K(s)). B^-1 resets to 1/H_diag when the energy rises.
Convergence: |dE| < 1e-12 Ha and ||g||_2 < 1e-7. The CI step uses a dense
eigensolver up to 2500 determinants and Davidson (warm-started) above.

The optimizer is a prototype choice. A Rust port should use an augmented
Hessian or Newton step with exact orbital Hessian-vector products; those
products are unverified here.

## Anchors and measurements

MEASURED 2026-11-11, ferric build 9e466807a, PySCF 2.13.1, 1 BLAS thread.
Run: `scripts/prototypes/casscf_proto.py` (all anchors, exit 0 = pass).

### Anchor 1: active space = full space

With n_c = n_v = 0 every rotation is redundant, so the check is the whole
matrix 2(F - F^T) and the energy against PySCF FCI built from PySCF's own RHF
integrals. The total energies differ by the RHF-level integral offset between
the two codes (dE(RHF)), so the correlation energy E - E_RHF is the compared
quantity, bar 1e-8 Ha. 2(F - F^T) is first order in the CI error, so its bar is
10 x the CI residual ||Hc - Ec|| (floor 1e-12).

| System | ndet | CI | dE (Ha) | dE(RHF) (Ha) | dE_corr (Ha) | max \|2(F-F^T)\| | CI residual |
|---|---|---|---|---|---|---|---|
| H2O/STO-3G | 441 | dense | -2.5e-8 | -2.4e-8 | -9.1e-10 | 3.1e-15 | 2.9e-14 |
| N2/STO-3G | 14400 | Davidson | -5.5e-8 | -5.1e-8 | -4.1e-9 | 8.4e-9 | 4.7e-8 |

### Anchor 2: analytic gradient vs finite difference, per class

Point: ferric RHF orbitals rotated by a random kappa (normal, scale 0.05, seed
7) over all nonredundant pairs, so no gradient class vanishes by symmetry or
Brillouin. Each element: 5-point central difference of the CI-relaxed CAS-CI
energy at h = 1e-3 and h/2, Richardson-extrapolated, (16 FD(h/2) - FD(h))/15.

Bar derivation, 1e-8 in g:

- truncation: the 5-point stencil error is h^4 f^(5)/30. Richardson removes
  it, leaving O(h^6). The size of the removed term is FD(h) - FD(h/2)
  (table row), at most 1.5e-8 (N2 core-active), so the residual O(h^6) term
  is below 1e-10;
- noise: the energy carries an error dE of about 3e-14 Ha (eps |E| for
  |E| = 109 Ha plus the 1e-14 `jk` screening threshold; dense CI). The
  extrapolated stencil amplifies it by about 3.3/h, giving 1e-10.

The bar sits two orders above both terms and five or more below every
mutation effect.

| System | Row | core-active | core-virtual | active-virtual |
|---|---|---|---|---|
| H2O CAS(4,4)/6-31G | max \|g\| | 2.22 | 7.96 | 0.513 |
| | FD(h) - FD(h/2) | 2.1e-9 | 6.4e-11 | 2.1e-11 |
| | analytic - FD | 6.8e-11 | 5.4e-11 | 1.6e-11 |
| N2 CAS(6,6)/6-31G | max \|g\| | 5.42 | 6.48 | 0.602 |
| | FD(h) - FD(h/2) | 1.5e-8 | 1.1e-10 | 2.5e-11 |
| | analytic - FD | 6.9e-11 | 8.6e-11 | 2.9e-11 |

Comparing against FD(h) without extrapolation fails the core-active class on
N2 (1.6e-8), with an error equal to FD(h) - FD(h/2), the stencil truncation.
The anchor therefore uses the extrapolated value.

Mutations (max |analytic - FD| per class; a class fails above 1e-8):

| Mutation | What changes | H2O ca / cv / av | N2 ca / cv / av | Caught by |
|---|---|---|---|---|
| `drop_FA_core` | F_pi = 2 F^I_pi (no F^A) | 1.76 / 2.00 / 1.6e-11 | 1.53 / 1.34 / 2.9e-11 | ca, cv |
| `drop_2rdm` | no Gamma term in F_pt | 0.493 / 5.4e-11 / 2.13 | 0.849 / 8.6e-11 / 1.70 | ca, av |
| `flip_core_virt` | g(cv) sign flipped | 6.8e-11 / 15.9 / 1.6e-11 | 6.9e-11 / 13.0 / 2.9e-11 | cv |
| `zero_act_virt` | g(av) = 0 (class dropped) | 6.8e-11 / 5.4e-11 / 0.513 | 6.9e-11 / 8.6e-11 / 0.602 | av |

Each mutation fails exactly the classes whose formula it touches and leaves
the others at the unmutated error. The script exits nonzero if any mutation
passes.

### Anchor 3: converged energies vs `pyscf.mcscf.CASSCF`

Two PySCF references, both converged from PySCF's own RHF orbitals:

- same integrals: PySCF RHF and CASSCF with hcore, overlap, ERIs and E_nuc
  taken from `AoIntegrals` (`init_guess="1e"`). Only convergence differs; the
  gate uses this reference;
- native: PySCF end to end with its own 6-31G integrals. This difference
  includes the integral offset and is reported, not gated.

Bar: near a minimum E - E_min <= ||g||^2 / (2 lambda_min) for each code, with
lambda_min the lowest eigenvalue of the CI-relaxed orbital Hessian
(central FD of the analytic gradient, h = 1e-4, at the converged
orbitals), plus PySCF's energy tolerance `conv_tol` = 1e-12.
Thresholds: prototype ||g||_2 < 1e-7 and |dE| < 1e-12; PySCF
`conv_tol_grad` = 1e-6 (1e-7 did not converge for H2O in 200 macro
iterations), `fcisolver.conv_tol` = 1e-14.

| Quantity | H2O CAS(4,4)/6-31G | N2 CAS(6,6)/6-31G |
|---|---|---|
| E prototype (Ha) | -76.0369690965018 | -109.0155467897007 |
| E PySCF, same integrals (Ha), run 1 / run 2 | -76.0369690965004 / -76.0369690964873 | -109.0155467897005 (both) |
| difference (Ha), run 1 / run 2 | -1.4e-12 / -1.4e-11 | -1.8e-13 / -1.7e-13 |
| bar (Ha) | 2.6e-11 | 1.1e-12 |
| \|\|g\|\| prototype / PySCF (run 1, run 2) | 9.5e-8 / 9.8e-7, 9.9e-7 | 3.4e-8 / 7.4e-8 |
| lambda_min (Ha) | 0.0195 | 0.0402 |
| RHF difference, same integrals (Ha) | 2.8e-14 | -2.8e-14 |
| E PySCF native (Ha) | -76.0369691047364 | -109.0155468530299 |
| difference, native (Ha) | 8.2e-9 | 6.3e-8 |
| RHF difference, native (Ha) | 7.0e-9 | 8.1e-8 |
| prototype macro iterations | 62 | 21 |
| <S^2> | 4e-31 | 7e-30 |
| approx H_diag vs FD diagonal, median / max rel. error | 0.28 / 0.79 | 0.24 / 0.69 |

The two runs are identical for the prototype. PySCF's H2O energy moves by
1.3e-11 Ha between runs at ||g|| = 1e-6, inside the bar of 2.6e-11 Ha, which
that gradient norm dominates.

The native difference follows the RHF-level offset between ferric's and
PySCF's 6-31G integrals (same sign and order), so it does not test the
CASSCF construction.

## Unverified items

- The approximate diagonal Hessian. It is a preconditioner only; its
  median relative error against the FD diagonal is 24-28 % (table above).
- Exact orbital Hessian-vector products and orbital-CI coupling, needed
  for an augmented-Hessian or Newton optimizer. The prototype forms the
  Hessian only by FD for lambda_min.
- Convergence behaviour: BFGS without frame transport, step cap 0.2 and
  the reset rule are tested on two systems from RHF starts. Root flipping,
  state averaging, symmetry-broken starts and near-degenerate active/inactive
  boundaries are untested.
- Open-shell (n_alpha != n_beta) active spaces. The code path accepts them,
  but no anchor covers them.
- The ferric-ci RDM convention. Replacing `PyscfFci` assumes ferric-ci returns
  PySCF-convention spin-summed RDMs. Anchor 1 and anchor 2 rerun with the new
  solver check that.

## Port checklist

1. `jk(D^c)` -> F^I, E_core; the active Hamiltonian from F^I and (tu|vw).
2. CAS-CI with spin-summed gamma, Gamma (ferric-ci, WP4).
3. `jk(D^a)` -> F^A; (pu|vw) for all p (nmo n_a^3 doubles).
4. F columns (table above) and g over the three classes.
5. Step and C <- C expm(K); K antisymmetric with K_pq = kappa_pq, p > q.
6. Tests: anchors 1-3 as above, with the four mutations.
