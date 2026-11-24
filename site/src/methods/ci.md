# CAS-CI and CASSCF

ferric runs complete-active-space configuration interaction (CAS-CI) and
two-step CASSCF from a closed-shell RHF reference, from Rust
(`ferric_ci::run_cas_ci`, `ferric_ci::run_casscf`) or Python
(`ferric.run_casci`, `ferric.run_casscf`), and SC-NEVPT2 on top of either
(`ferric_ci::sc_nevpt2`, `ferric.run_nevpt2`). CASSCF and SC-NEVPT2 are also
the CLI kinds `casscf` and `nevpt2` (see [From the CLI](#from-the-cli));
CAS-CI has no CLI kind.

The Hamiltonian, the determinant phase convention and the Slater-Condon rules
are derived in `book/src/17-multireference/02-slater-condon.md#slater-condon-rules`. The phase conventions are checked
by `scripts/casci_slater_condon_phases_sympy.py`.

## Scope

| | Implemented |
|---|---|
| Space | full CI inside `n_active` spatial orbitals; no RAS1/RAS3 restrictions |
| Reference | restricted (closed-shell) RHF start; the active-space sector `(n_alpha, n_beta)` may be unequal (see [CASSCF](#casscf)) |
| Roots | the single lowest root |
| Solver | CI-native Davidson, diagonal preconditioner `1/(H_ii - λ)` |
| Entry points | Rust `ferric_ci::run_cas_ci` (re-exported as `ferric::ci`); Python `ferric.run_casci` |
| Density matrices | spin-summed 1- and 2-RDM of the root, opt-in (`compute_rdms`) |
| Orbital optimization | two-step CASSCF, single root of any $(n_\alpha, n_\beta)$ sector, optional canonical orbitals (see [CASSCF](#casscf)) |
| Perturbation theory | SC-NEVPT2 on the single root, $n_\alpha = n_\beta$ (see [SC-NEVPT2](#sc-nevpt2)) |
| Not implemented | open-shell or non-singlet references, S² control, multiple roots, state averaging, a CLI kind for plain CAS-CI, spin-resolved RDMs |

With no inactive orbitals the method is FCI in the basis. A call that fails
the bookkeeping check, `2 * active_start + n_alpha + n_beta == nelec`, returns
an error. So does a non-restricted reference, `n_active == 0`, or
`n_active > 63` (the `u64` string limit).

## Energy

$$
E_\text{CASCI} = E_\text{core} + \lambda_\text{min}(H_\text{act}),
$$

where $E_\text{core}$ is the nuclear repulsion plus the energy of the
`active_start` doubly occupied inactive orbitals, and $H_\text{act}$ carries
the inactive-core Fock potential in its one-electron part. `e_active` is
`e_total - e_core`, the quantity PySCF reports as `CASCI.e_cas`.

## Reduced density matrices

With `compute_rdms = true` the result carries the spin-summed RDMs of the
converged CI vector $C$ over the active orbitals, in the convention of PySCF
`fci.direct_spin1.make_rdm12`:

$$
\gamma_{pq} = \langle C|E_{pq}|C\rangle ,\qquad
\Gamma_{pqrs} = \sum_{\sigma\tau}\langle C|a^\dagger_{p\sigma} a^\dagger_{r\tau}
a_{s\tau} a_{q\sigma}|C\rangle ,
$$

`dm1[p,q]` $=\gamma_{pq}$, `dm2[p,q,r,s]` $=\Gamma_{pqrs}$, with
$E_{pq} = \sum_\sigma a^\dagger_{p\sigma}a_{q\sigma}$ and index $p$ = active
orbital `active_start + p`. Then

$$
E_\text{CASCI} = E_\text{core} + \sum_{pq} h_{pq}\gamma_{pq}
 + \tfrac12 \sum_{pqrs} (pq|rs)\,\Gamma_{pqrs},
$$

with $h$ (inactive-core potential included) and $(pq|rs)$ the active-space
integrals returned in the result (`integrals` in Rust, `h_active` and
`eri_active` in Python). $\operatorname{tr}\gamma = N_\text{act}$ and
$\sum_{pr}\Gamma_{pprr} = N_\text{act}(N_\text{act}-1)$.

The RDMs are built from the same single-excitation string lists as the sigma
vector, as vectors $E_{rs}C$ and one $n^2 \times n^2$ matrix product per block
of determinants; no determinant pair is visited. Cost is
$O(N_\text{det}\, n^4)$; the scratch is charged to the CAS-CI memory budget.
Derivation: `book/src/17-multireference/03-direct-sigma-davidson.md#direct-sigma-davidson-split`.

## Library use

```rust,ignore
use ferric_ci::{run_cas_ci, CasCiConfig};

// `mol`, `prep` and `rhf` come from a converged restricted solve_rhf
// with exact J/K (no density fitting), as in the validation tests.
let cfg = CasCiConfig {
    n_active: 6,
    n_elec_active: (3, 3),
    active_start: 4,          // MOs 0..4 are doubly occupied and inactive
    ..Default::default()
};
let res = run_cas_ci(&mol, &prep, &rhf, cfg)?;
println!("{res}");            // energy, determinant count, iterations
```

`CasCiResult` holds `e_total`, `e_active`, `e_core`, `n_determinants`,
`ci_vector` (index `ia * n_beta_strings + ib`, where `n_beta_strings = C(n_active, n_beta)` is the number of beta strings), `converged`, `iterations`,
`integrals` and, with `compute_rdms`, `rdms` (`dm1`, `dm2`).

From Python:

```python
import numpy as np
import ferric

mol = ferric.Molecule.from_xyz("h2o.xyz")
r = ferric.run_casci(mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 2))
e = r.e_core + np.einsum("pq,pq", r.h_active, r.dm1) \
    + 0.5 * np.einsum("pqrs,pqrs", r.eri_active, r.dm2)   # == r.e_total
print(np.linalg.eigvalsh(r.dm1), r.certificate["grade"])   # natural occupations
```

`run_casci` runs its own exact-J/K RHF; `active_start` defaults to
$(N - N_\alpha - N_\beta)/2$. The certificate's capability is `casci`; it is
not graded.
Defaults: `conv_thresh = 1e-9`, `max_subspace = 50`, `max_iter = 200`.

## CASSCF

`run_casscf` optimizes the orbitals for the lowest CAS-CI root. Each macro
iteration solves the CAS-CI with RDMs in the current orbitals, builds the
inactive and active Fock matrices and the generalized Fock matrix $F$, and
takes an orbital step $C \leftarrow C\exp(K)$ over the core-active,
core-virtual and active-virtual rotations, with gradient
$\partial E/\partial\kappa_{pq} = 2(F_{pq} - F_{qp})$. Equations:
`book/src/17-multireference/05-casscf.md#casscf-energy-gradient-step`.

| Setting | Default | Meaning |
|---|---|---|
| `grad_tol` (Python `casscf_grad_tol`) | 1e-7 | converged when $\lVert g\rVert_2$ is below this |
| `energy_tol` (`casscf_energy_tol`) | 1e-12 Ha | and the energy change is below this at the same iteration |
| `max_iter` (`casscf_max_iter`) | 300 | macro iterations |
| `max_step` (`casscf_max_step`) | 0.2 | cap on the largest step element (rad) |
| `optimizer` (`casscf_optimizer`) | `"ah"` | `"ah"`: augmented-Hessian step on the exact orbital Hessian at fixed RDMs, with a trust radius; `"bfgs"`: inverse BFGS from an approximate diagonal Hessian |
| `integrals` (`casscf_integrals`) | `"auto"` | `"auto"`: dense when the block fits in half of the memory budget, else direct; `"direct"`: J, K and $(pu\|vw)$ from shell quartets every iteration, screening threshold `integral_thresh` (`casscf_integral_thresh`, 1e-12); `"dense"`: the $n_\text{bas}^4$ AO block; `"ri"`: density-fitted integrals in `auxbasis` (`casscf_auxbasis`, required, a bundled set such as `def2-universal-jkfit`), $n_\text{aux}n_\text{bas}^2$ held, a different Hamiltonian; never chosen by `"auto"` |
| `ci_conv_thresh` | 1e-10 | Davidson residual $\lVert Hc - Ec\rVert$ of every CI solve; a solve that misses it returns an error |

The default step (`optimizer = "ah"`) is the lowest eigenvector of the
augmented Hessian built from Hessian-vector products of the exact orbital
Hessian at fixed RDMs, under a trust radius. The products come from a
kernel over occupied MO integrals, built once per macro iteration, so each
costs two small matrix-vector products rather than a J/K pass.
`optimizer = "bfgs"` is inverse BFGS, started from an approximate diagonal
orbital Hessian and reset to it when the energy rises; it needs 1.4 to 5
times the macro iterations (CAS(6,6)/cc-pVDZ benzene: 89 instead of 19).
On the five tested systems AH takes less wall time: benzene CAS(6,6)
27 s against 42 s, N2 CAS(10,10)/cc-pVDZ 19 s against 27 s (timings on a
loaded machine, in `book/src/17-multireference/05-casscf.md#casscf-proven-timings`). Both reach the same stationary point on the
tested systems; neither moves benzene to the lower minimum PySCF finds. A run that reaches the
iteration cap returns `converged = false`; Python adds a `RuntimeWarning` and
a certificate warning; the certificate (capability `casscf`, graded Proven)
is then capped to not graded.

Convergence means a small gradient, so the converged point can be a saddle.
After convergence `[cas] stability` (Python `casscf_stability`) checks the
second-order condition. It finds the lowest eigenvalue of the CI-relaxed
orbital Hessian $H_{oo} - H_{oc}H_{cc}^{-1}H_{co}$ by Davidson. Each product
solves one CI response equation, so the eigenvalue includes the CI response
and not only the fixed-RDM orbital block:

| `stability` | Effect |
|---|---|
| `"off"` (default) | no check |
| `"check"` | compute $\lambda_\text{min}$ and record it on the result and in the certificate |
| `"follow"` | as `"check"`; on a negative $\lambda_\text{min}$, step along its eigenvector (the lower of $\pm 0.3v$) and optimize again, up to 5 times |

The check is off by default because it costs about as much as the CASSCF or
more. Measured from the CLI phase timings (cc-pVDZ, 3 runs each, shared
machine): benzene CAS(6,6), 400 determinants, 8.6 s for the check against
19.0 s for the CASSCF (0.45 times); N2 CAS(10,10), 63,504 determinants,
35.2 s against 11.1 s (3.2 times). Each of the 100 to 180 Hessian products
solves one CI linear system, so the cost grows with the determinant count.

The verdict is `STABLE`, `UNSTABLE` ($\lambda_\text{min} < -10^{-6}$),
`MARGINAL` ($|\lambda_\text{min}| \le 10^{-6}$) or `INDETERMINATE` (Davidson
or a CI response did not converge). An `UNSTABLE` point is a saddle: its
certificate is capped to not graded and leaves the PySCF anchor scope.
Python exposes `stability_lowest_eigenvalue`, `stability_verdict` and
`stability_follow_steps` (`None` with `"off"`); the CLI prints the verdict
and writes it to the run log.

Measured on N2 CAS(10,10)/cc-pVDZ started from the RHF orbitals: BFGS, AH and
PySCF stop at the same second-order saddle, $E = -109.12947$ Ha, whose
active space holds one component of a degenerate $\pi_u$ pair. The check
reports it as `UNSTABLE`; `"follow"` reaches the local minimum 5.29 mHa
lower, which has a symmetric $\sigma_g$ and $\sigma_u$ active space. The
benzene CAS(6,6)/cc-pVDZ dense endpoint, 5.9 mHa above PySCF's, is a local
minimum (positive $\lambda_\text{min}$), so `"follow"` leaves it in place.
Numbers, cost and the comparison with finite differences are in
`book/src/17-multireference/05-casscf.md#casscf-second-order-check`; the tests are
`crates/ferric-ci/tests/casscf_stability.rs` and
`casscf_stability_saddle.rs`.

```python
import ferric

mol = ferric.Molecule.from_xyz("water.xyz")
r = ferric.run_casscf(mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 2))
print(r.e_total, r.iterations, r.grad_norm, r.converged)

# Canonical orbitals: diagonal core and virtual blocks of F^I + F^A,
# natural active orbitals; mo_energy is the Fock diagonal.
rc = ferric.run_casscf(mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 2),
                       canonicalize=True)
print(rc.mo_energy, rc.dm1.diagonal())        # occupations, descending

# Lowest M_s = 1 state (a triplet) in the same active space.
rt = ferric.run_casscf(mol, ferric.BasisSet.bundled("6-31g"), 4, (3, 1))
```

`canonicalize = true` (Python `canonicalize=True`) diagonalizes the core and
virtual blocks of $F^I + F^A$ (ascending) and turns the active orbitals into
natural orbitals (occupations descending), then solves the CAS-CI again in
those orbitals, so `ci_vector`, `dm1` and `dm2` are in the returned
`mo_coeff`. The energy is unchanged. `mo_energy` is the diagonal of
$F^I + F^A$; it is `None` without canonicalization.

`n_elec_active = (n_alpha, n_beta)` with `n_alpha != n_beta` gives the lowest
state of that $M_s$ sector, with one restricted orbital set and a doubly
occupied core. The molecule is given with an even electron count and
multiplicity 1, because the start is RHF; the sector sets the state.

Limits:

- single root, no state averaging;
- the start is the canonical RHF orbitals with the active window directly
  above the core; there is no orbital selection or reordering; odd-electron
  molecules have no RHF start;
- both optimizers are two-step: no orbital-CI coupling in the step (the
  `"ah"` Hessian is the orbital block at fixed RDMs; the CI response enters
  only the stability check);
  convergence is tested on five systems from RHF starts;
- the dense `nbasis⁴` AO block is held when it fits in half of the memory
  budget; otherwise (or with `[cas] integrals = "direct"`) the integrals are
  recomputed from shell quartets every macro iteration, with no `nbasis⁴`
  array. On benzene CAS(6,6)/cc-pVDZ an iteration costs 4.1 s on the direct
  route and 0.29 s on the dense route.
- `[cas] integrals = "ri"` with `auxbasis` runs CASSCF (and SC-NEVPT2 on
  that reference) on the density-fitted Hamiltonian, holding the
  $n_\text{aux}n_\text{bas}^2$ tensor only. Its energy differs from the exact
  CASSCF by the density-fitting error; the certificate records the `ri`
  approximation and caps the grade at Proven (narrow), outside the
  exact-integral anchors. `reference = "casci"` refuses it. `ferric.fcidump`
  always writes the exact integrals, so an FCIDUMP of RI orbitals holds the
  exact Hamiltonian at those orbitals.
  On benzene CAS(6,6)/cc-pVDZ it uses 203 CPU s and 0.75 GB against the
  dense route's 302 CPU s and 2.35 GB; at cc-pVTZ (264 basis functions,
  where the dense block needs 50 GB) it converges in 2.8 GB.

Validation (`crates/ferric-ci/tests/casscf.rs`, `casscf_vs_pyscf.rs`,
`casscf_canonical.rs`, measured 2026-11-11; equations and bars in
`book/src/17-multireference/05-casscf.md#casscf-proven`). PySCF `mcscf.CASSCF` runs on ferric's AO
integrals from PySCF's own RHF orbitals (`scripts/gen_casscf_refs.py`);
ferric starts from its own RHF:

| System, 6-31G | $E$ (Ha) | $E$ − PySCF (Ha) | bar (Ha) | analytic − FD gradient, worst class |
|---|---:|---:|---:|---:|
| H2O CAS(4,4) | -76.0369690965018 | -1.5e-11 | 1.0e-10 | 2.4e-10 |
| N2 CAS(6,6), 1.0977 Å | -109.0155467897006 | -9.9e-14 | 5.2e-12 | 2.0e-10 |
| N2 CAS(6,6), 1.6 Å | -108.8482292478625 | -5.8e-13 | 5.1e-12 | not run |
| O2 CAS(8,6), $(n_\alpha, n_\beta) = (5, 3)$ | -149.6367298537082 | -2.6e-13 | 4.0e-12 | 4.6e-10 |

The bar is $(\lVert g\rVert^2 + \lVert g_\text{PySCF}\rVert^2)/(2\lambda_\text{min})$
plus PySCF's energy tolerance and rounding; for H2O it is set by PySCF's
gradient, 1e-6. With the active space equal to the full space (STO-3G: H2O
in the $M_s = 0$ and $M_s = 1$ sectors, N2, O2 at $M_s = 1$) the energy equals
the full-window `run_cas_ci` of the same sector to 1e-13 Ha and
$2(F - F^\mathsf{T})$ is below 8e-11. Canonicalization moves the energy by at
most 5.7e-14 Ha; the RDMs of the returned CI vector rebuild it to 2.8e-14 Ha.

## SC-NEVPT2

`run_nevpt2` (Rust `ferric_ci::sc_nevpt2`) adds strongly contracted NEVPT2
(Angeli et al., JCP 117, 9138 (2002)) to a CASSCF (`reference="casscf"`,
the default) or CAS-CI (`reference="casci"`, on the RHF orbitals) root. The
zeroth-order Hamiltonian is Dyall's; the core and virtual orbitals are the
eigenvectors of the core and virtual blocks of $F^I + F^A$, as in PySCF
`mrpt.NEVPT`. The result carries the eight class energies and norms
(`Sr`, `Si`, `Sijrs`, `Sijr`, `Srsi`, `Srs`, `Sij`, `Sir`), their sum
`e_corr`, and `e_total = e_reference + e_corr`. Equations:
`book/src/17-multireference/06-nevpt2.md#nevpt2-method`.

```python
import ferric

mol = ferric.Molecule.from_xyz("water.xyz")
r = ferric.run_nevpt2(mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 2))
print(r.e_total, r.e_corr, r.energies["Sir"])
```

It refuses an unconverged CASSCF or CAS-CI reference, $n_\alpha \ne n_\beta$,
a CI vector that is not an eigenvector of the active Hamiltonian to 1e-8,
and a memory budget that cannot hold the dense AO integrals, the MO blocks
and the $n_\text{act}^6$ intermediates (checked before any integral is
built). Single root. The certificate (capability `nevpt2`) is graded Proven
on a CASSCF reference and capped to not graded on a CAS-CI reference.

Validation (`crates/ferric-ci/tests/nevpt2.rs`, 6-31G, measured 2026-11-12):

| Check | Largest difference | Bar |
|---|---:|---:|
| Classes with an empty external space (H2, LiH) | exactly 0 | 0 |
| One closed-shell active orbital vs MP2 pair sums (H2O) | 8.1e-15 Ha | 1e-10 Ha |
| numpy prototype on the same orbitals and CI vector (H2O CAS(4,4), N2 CAS(6,6) at 1.0977 and 2.2 Å) | 6.2e-14 Ha | 2.3e-10 to 4.6e-9 Ha |
| PySCF `mrpt.NEVPT` on its own CASSCF, per class: H2O / N2 / stretched N2 | 1.3e-8 / 3.2e-8 / 2.1e-8 Ha | per class, from both CASSCF gradients |

The PySCF bar per class is $|dE_\text{class}/d\kappa|$ times the orbital
distance between the two converged references,
$(\lVert g\rVert + \lVert g_\text{PySCF}\rVert)/\lambda_\text{min}$, plus
twice the class spread under rotations inside degenerate core or virtual
orbitals, plus 1e-10 Ha. PySCF's gradients (2e-7 to 1e-6) set it.

## Limits

- The active-space integrals come from a dense AO ERI tensor, `O(nbasis⁴)`
  memory. Use it on minimal and double-zeta basis sets.
- The sigma build $\sigma = Hc$ is the string-driven direct algorithm
  (Knowles-Handy/Olsen): β-β, α-α and α-β contractions over per-string
  single-excitation lists, `O(N_det · n_exc_α · n_exc_β)` per call with
  `n_exc = k(n-k)+k`, parallel over α strings. It never visits determinant
  pairs. The `O(N_det²)` pair loop (`hamiltonian::sigma_naive`) and the
  dense Hamiltonian remain as test oracles.
- Measured on N2/cc-pVDZ valence active spaces (release, one process per
  run, `OPENBLAS_NUM_THREADS=1`, median of 3):

  | Active space | N_det | sigma, pair loop | sigma, direct (1 / 6 threads) | CAS-CI wall, direct (1 / 6 threads) | CAS-CI wall, pair loop |
  |---|---:|---:|---:|---:|---:|
  | 6e in 8 orbitals | 3,136 | 0.054 s | 0.0020 / 0.0026 s | 0.10 / 0.10 s | 0.77 s |
  | 8e in 8 | 4,900 | 0.127 s | 0.0025 / 0.0028 s | 0.11 / 0.11 s | 1.93 s |
  | 10e in 10 | 63,504 | 17.2 s | 0.035 / 0.015 s | 0.71 / 0.37 s | 274 s |
  | 12e in 12 | 853,776 | not run | 0.83 / 0.18 s | 15.4 / 5.0 s | not run |

  The CAS-CI wall includes the integral transform; peak RSS at 12e in 12
  is 0.31 GB. Raw logs: `wiki/notes/t020-sigma-2026-11-06/`.
- The Davidson subspace needs `2 * max_subspace * N_det * 8` bytes.
  `memory_budget_bytes` bounds the integral transform and the subspace;
  `None` resolves through `FERRIC_MEM_BUDGET_GB`, then 0.8 of available RAM,
  then a 2 GiB floor.

## Validation

Compared with PySCF `mcscf.CASCI` on an exact-integral RHF with the same
orbital window, and with a dense diagonalization of the full active-space
Hamiltonian:

| System | Active space | Basis | Largest difference |
|---|---|---|---|
| N2, r = 1.10 and 2.20 Å | CAS(6,6) | cc-pVDZ | E_CASCI 1.4e-11 Ha |
| H2O | CAS(4,4) | 6-31G | e_core and e_active 2.6e-10 Ha |

Each system is also run with the window moved down one orbital. The test
tolerance is 3e-9 Ha. The stretched N2 reference is the symmetric RHF, which
is internally unstable toward a symmetry-broken RHF 0.19 Ha lower; both codes
use the symmetric state.

The sign convention is checked three ways, because energy comparisons cannot
detect a diagonal $\pm1$ similarity of $H$:

1. `tests/slater_condon_brute_force.rs` compares `general_element`, the
   direct `sigma` and the diagonal with an independent bit-loop second-quantized $H$ on
   random asymmetric integrals, CAS(4,4) to CAS(6,6), odd and even $N_\alpha$.
   The largest difference is 8.9e-15.
2. `scripts/casci_slater_condon_phases_sympy.py` builds the
   Jordan-Wigner operators symbolically, verifies the anticommutation
   relations, and shows that a line-for-line port of `general_element` equals
   the explicit $H$ as an exact linear form in $h_{pq}$ and $(pq|rs)$ for four
   $(n, N_\alpha, N_\beta)$ sectors. Four sign mutations and the interleaved
   orbital order each fail.
3. `tests/direct_sigma_vs_naive.rs` compares the direct sigma with the
   pair-loop `sigma_naive` on random vectors: ten random-integral sectors
   (unequal $N_\alpha, N_\beta$, empty and full channels) and real
   H2O/STO-3G integrals (full CI, 441 determinants; 8 electrons in 6
   orbitals, 225; four unequal-spin sectors). The largest difference is
   9.9e-14 Ha on vectors with $|\sigma| \approx 74$ Ha. Dropping one phase
   factor, the α-β term, the β-β term, the ½ on same-spin products, the
   $-\tfrac12\sum_r (pr|rq)$ one-body correction, or the β sign in the
   α-β gather each fail it.
4. The PySCF energies above fix the spectrum.

The RDMs are checked in `crates/ferric-ci/tests/casci_rdms.rs` on the N2
CAS(6,6) and H2O CAS(4,4) systems above:

| Check | N2 CAS(6,6) | H2O CAS(4,4) |
|---|---:|---:|
| RDM energy vs Davidson eigenvalue | 1.4e-14 Ha | 2.8e-14 Ha |
| `dm1` vs PySCF `make_rdm12` | 4.6e-13 | 4.9e-15 |
| `dm2` vs PySCF `make_rdm12` | 9.3e-12 | 1.0e-14 |

PySCF diagonalizes the dense Hamiltonian built from ferric's active-space
integrals and evaluates `make_rdm12` on that eigenvector
(`scripts/gen_casci_rdm_refs.py`); the reference is rotated into each run's
orbitals through the AO overlap. The test also checks the traces
$N$ and $N(N-1)$, the partial trace, the pair and transpose symmetries, and
natural occupations in $[0, 2]$.

## From the CLI

`method.kind = "casscf"` and `"nevpt2"` run the same Rust drivers
(`ferric_ci::driver`) as `run_casscf` and `run_nevpt2`, so one input gives
the same energy bit for bit on both surfaces. The active space is the
`[cas]` section; `[cas.ci]` holds the CI solver settings and `[scf]` the
convergence of the RHF start:

```toml
[molecule]
xyz = "testdata/molecules/water.xyz"

[basis]
name = "6-31g"

[method]
kind = "nevpt2"

[cas]
n_active = 4
n_elec_active = [2, 2]
```

The molecule must be a singlet (`multiplicity = 1`); an open-shell CASSCF
takes its sector from `n_elec_active`, as `examples/o2-casscf-triplet.toml`
does. Keys and refusals are in the [input reference](../reference/input.md#cas).
`task = "energy"` only.

The grade is on [Capabilities and validation](../reference/validation.md).
Sources: `crates/ferric-ci/tests/validation_casci.rs` and
`scripts/validation/gen_casci.py`.
