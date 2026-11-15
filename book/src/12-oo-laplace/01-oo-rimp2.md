# 12.1 Closed-shell orbital-optimized RI-MP2

<!-- section: 12.1; status: ready; registry: oo-rimp2;
     theory: crates/ferric-mp2/theory.md#"7. Orbital-optimized RI-MP2" -->

*Notation follows `book/notation.md`. Code references are to the
`docs/book-p5` tree. Line numbers are left out because they drift; each
reference names a function.*

Orbital-optimized MP2 treats the orbitals as variational parameters of the
Hartree–Fock plus second-order energy. The amplitudes are those of
section 10.1, taken in a frame that is no longer canonical. The orbitals move
until the total energy is stationary under occupied–virtual rotations. At
that point the energy needs no orbital response: its nuclear gradient takes
the same form as a variational SCF gradient (the gradients chapter).

| Index | Range | Code name |
|---|---|---|
| $i, j, k$ | active occupied orbitals | `0..nocc`, offset `first_occ` |
| $a, b, c$ | virtual orbitals | `0..nvir`, offset `nocc_total` |
| $p, q$ | all orbitals | `0..nmo` |
| $\mu, \nu$ | AO functions | `0..nbas` |
| $P, Q$ | auxiliary functions | `0..naux` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/oo_rimp2.rs::oo_ri_mp2`; configuration `OoRiMp2Config` |
| CLI | `method.kind = "oo-rimp2"`, keys `oo_max_iter`, `oo_grad_conv`, `oo_level_shift`, `oo_diis_size` in `[mp2]` (`examples/water-oo-rimp2.toml`); `task = "optimize"` runs `optimize_geometry_oo_rimp2` |
| Python | `run_oo_rimp2` (`crates/ferric-python/ferric.pyi`); `run_optimize(method="oo-rimp2")` |

The CLI hands an open-shell reference under the same kind to the unrestricted
driver (section 12.2). The Python entry point runs closed shell only.

## 12.1.1 The functional

For a coefficient matrix $C$ with $n_{\mathrm{occ}}$ doubly occupied columns,
let $D = 2C_oC_o^{\mathsf T}$ over all occupied orbitals, including frozen
core, and $F = h + J[D] - \tfrac12 K[D]$. The functional is

$$ E(C) = \tfrac12\operatorname{Tr}\big[D\,(h + F)\big] + V_{nn} + E^{(2)}_{\mathrm{RI}}(\tilde C), \qquad \tilde C = C\begin{pmatrix} U_o & \\ & U_v\end{pmatrix}. \tag{12.1} $$

$U_o$ diagonalizes the active occupied block of $C^{\mathsf T}FC$, and $U_v$
the virtual block. The frozen-core block is not touched
(`ferric-mp2/src/oo_rimp2.rs::semicanonical_rotation`). $E^{(2)}_{\mathrm{RI}}$
is equation (10.3) with the diagonal elements of $\tilde C^{\mathsf T}F\tilde C$
in the denominators. The code evaluates (12.1) in
`ferric-mp2/src/oo_rimp2.rs::evaluate_point`: first `compute_hf_energy`, then
the rotation, then `compute_rimp2_with_orbitals`.

Taking the MP2 part in the semicanonical frame makes $E$ invariant to
occupied–occupied and virtual–virtual rotations of $C$. This is the
full-Fock Hylleraas functional of Bozkaya and Sherrill (OMP2). The
off-diagonal occupied–virtual Fock block, which is non-zero away from the HF
solution, enters only through the orbital gradient (12.3); the amplitudes
ignore it. The invariance is tested directly
(`test_oo_energy_is_invariant_to_occ_occ_and_vir_vir_rotations` in
`crates/ferric-mp2/src/oo_rimp2.rs`). Only active-occupied/virtual rotations
are optimized. Frozen-core orbitals are not rotated into the virtuals, so
`frozen_core > 0` gives a constrained minimum.

## 12.1.2 The rotation

The rotation follows the convention in notation section 6. $\kappa$ is
antisymmetric, with $\kappa_{ai}$ stored and $\kappa_{ia} = -\kappa_{ai}$.
The update is $C \leftarrow C\,U(\kappa)$ with the Cayley form

$$ U(\kappa) = \big(1 - \tfrac12\kappa\big)^{-1}\big(1 + \tfrac12\kappa\big). \tag{12.2} $$

> **Verified equation 12.2** · Cayley rotation
>
> For antisymmetric $\kappa$, $U(\kappa)$ is orthogonal and
> $U(\kappa) - e^{\kappa} = O(\kappa^3)$.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 6 |
> | Negative control | the reversed form $(1 + \tfrac12\kappa)^{-1}(1 - \tfrac12\kappa)$ must track $e^{-\kappa}$, not $e^{\kappa}$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/orbital_rotation.rs::cayley_rotation` |
> | Code test | `test_cayley_is_unitary` in `crates/ferric-mp2/src/oo_rimp2.rs` |
>
> <!-- verified: id=oo.cayley; script=scripts/mp2_variant_identities.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/orbital_rotation.rs::cayley_rotation; rust_test=ferric-mp2/src/oo_rimp2.rs::test_cayley_is_unitary -->

`cayley_rotation` factors $1 - \tfrac12\kappa$ once (LU) and solves for every
column against that factor.

## 12.1.3 The orbital gradient

The gradient is $g_{ai} = +\partial E/\partial\kappa_{ai}$. It is assembled in
the semicanonical frame and rotated back to the frame of $C$,
$g = U_v\,g_{\mathrm{sc}}\,U_o^{\mathsf T}$. In the semicanonical frame:

$$ g_{ck} = 4F_{ck} + 2\big(G_V + G_O\big)_{ck} + 4\big[C^{\mathsf T}\,\mathcal G[CWC^{\mathsf T}]\,C\big]_{ck} + 2\big(FW - WF\big)_{ck}, \qquad \mathcal G[X] = J[X] - \tfrac12K[X]. \tag{12.3} $$

- $4F_{ck}$ is the Hartree–Fock (Brillouin) term.
- $G_V + G_O$ is the response of the RI integrals $(ia|jb)$ to the rotation.
  It is a contraction of the amplitudes with $B^P_{ca}$ and with the
  $(ik|jb)$ block
  (`ferric-mp2/src/oo_rimp2.rs::hf_and_integral_response_gradient_timed`,
  GEMM kernels in `ferric-mp2/src/orbital_gradient_gemm.rs`).
- The last two terms are the response of the denominators. They come through
  the Fock matrix contracted with the unrelaxed MP2 density $W = P + P^{\mathsf T}$
  (`ferric-mp2/src/oo_rimp2.rs::fock_response_gradient`).

The two blocks of $P$ are

$$ P_{ij} = -\sum_{kab} t_{ik}^{ab}\big(2t_{jk}^{ab} - t_{jk}^{ba}\big), \qquad P_{ab} = \sum_{ijc} t_{ij}^{ac}\big(2t_{ij}^{bc} - t_{ij}^{cb}\big), \qquad t_{ij}^{ab} = -\frac{(ia|jb)}{\Delta_{ij}^{ab}} . \tag{12.4} $$

The code stores $t$ dense as an $(n_on_v)\times(n_on_v)$ matrix and builds
$P$ in `ferric-mp2/src/oo_rimp2.rs::build_mp2_density`.

> **Verified equation 12.3** · closed-shell OO-MP2 orbital gradient
>
> On a random closed-shell model in the semicanonical frame with
> $F_{ov} \ne 0$, the gradient (12.3), with $W$ from (12.4), equals
> $\partial E/\partial\kappa_{ck}$ of the spin-orbital energy, with the spatial
> $\kappa$ applied to both spins. The script transcribes the code's scalar
> reference loop (`orbital_gradient_scalar_reference`) and
> `fock_response_gradient` (part B3). It also checks the generalized-Fock
> gradient against three independent finite-difference routes (part B2).
>
> | | |
> |---|---|
> | Proof | `scripts/oomp2_mp3_identities_sympy.py`, the check labelled "claim B3" (the code's loop against $\partial E/\partial\kappa$); the check labelled "claim B2" is the finite-difference anchor for the generalized Fock gradient |
> | Negative control | mutations `kappa_sign` (generator $C(1-\kappa)$) and `drop_two_particle` (no correlation two-particle density) turn B2 and B3 red (`OOMP2_MP3_MUTATION`) |
> | Runner | `crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py` |
> | Code | `ferric-mp2/src/oo_rimp2.rs::fock_response_gradient`, `ferric-mp2/src/oo_rimp2.rs::hf_and_integral_response_gradient_timed` |
> | Code test | `test_oo_rimp2_gradient_fd_off_reference_h2o_sto3g` and `test_oo_rimp2_gradient_fd_off_reference_nh3_631g` in `crates/ferric-mp2/src/oo_rimp2.rs`: the production gradient against a finite difference of (12.1), away from the HF point |
>
> <!-- verified: id=oo.cs.gradient; script=scripts/oomp2_mp3_identities_sympy.py; claim=B3; runner=crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py; code=ferric-mp2/src/oo_rimp2.rs::fock_response_gradient; rust_test=ferric-mp2/src/oo_rimp2.rs::test_oo_rimp2_gradient_fd_off_reference_h2o_sto3g -->

**The sign.** `crates/ferric-mp2/src/orbital_rotation.rs` records that the
closed-shell driver once used the reversed Cayley form, with a gradient of
the opposite sign matched to it. Both drivers now use (12.2) with
$g = +\partial E/\partial\kappa$. That module's documentation still describes
the migration in terms of a "$-4F_{ai}$" term; the code today adds $+4F_{ai}$.

## 12.1.4 The algorithm

One iteration of `ferric-mp2/src/oo_rimp2.rs::oo_ri_mp2`:

| Step | Operation | Function |
|---|---|---|
| 1 | Evaluate (12.1) at $C$: $J$, $K$, $F$, $E_{\mathrm{HF}}$, semicanonical rotation, $B$, $t$, $E^{(2)}$ | `evaluate_point` |
| 2 | Gradient (12.3), back-rotated; stop if $\lVert g\rVert_2 <$ `grad_conv` | `compute_orbital_gradient` |
| 3 | Step $\kappa_{ai} = -g_{ai}/(\varepsilon_a - \varepsilon_i + \mu)$, $\varepsilon = \operatorname{diag}(C^{\mathsf T}FC)$ in the frame of $C$ | main loop |
| 4 | If $\max\lvert\kappa_{ai}\rvert > 0.3$, scale $\kappa$ uniformly to 0.3 rad | main loop |
| 5 | $C_{\mathrm{new}} = C\,U(\kappa)$ | `cayley_rotation` |
| 6 | DIIS on $C_{\mathrm{new}}$ with error $C_{\mathrm{new}}\,G\,C_{\mathrm{new}}^{\mathsf T}$, $G$ the antisymmetric embedding of $g$; then $C \leftarrow C(C^{\mathsf T}SC)^{-1/2}$ | `lowdin_orthonormalize` |
| 7 | Evaluate the trial point | `evaluate_point` |
| 8 | If $E_{\mathrm{trial}} > E + 10^{-4}$: halve $\kappa$ up to ten times without DIIS, stopping at the first trial with $E \le E + 10^{-12}$; reset DIIS | main loop |
| 9 | If $\lvert E_{\mathrm{trial}} - E\rvert <$ `energy_conv` and the iteration is past the first: recompute the gradient at the accepted point and stop if $\lVert g\rVert_2 < 10\,$`grad_conv` | main loop |

Notes on the table:

- **Step 3.** The diagonal of $C^{\mathsf T}FC$ serves only as a
  preconditioner. The gradient and the energy use the semicanonical frame.
- **Step 8.** If none of the ten halvings is accepted, the closed-shell
  driver takes the last halved step anyway. The unrestricted driver
  (section 12.2) keeps the old point instead.
- **Step 9.** The energy change is measured on the trial of step 7, before
  any backtracking.
- **The norm.** The stopping test uses the Euclidean norm over the active
  occupied–virtual block, not the largest element.

**J and K.** Each evaluation rebuilds $J$ and $K$ on the reference SCF's
route (`OoJk`).

- When the reference ran RI-J plus RI-K with one auxiliary basis and no COSX
  or range-separated exchange, the OO loop uses that fitted route. The HF
  part of (12.1) is then the fitted HF energy. The test
  `oo_hf_energy_on_an_rijk_reference_is_the_scf_energy` checks that it equals
  the SCF energy at the reference orbitals.
- Otherwise the loop builds exact four-centre $J$ and $K$ at a $10^{-12}$
  threshold. `difference_jk` optionally builds them from the density change
  since the previous point; `oo_rimp2_difference_jk_matches_plain_builds`
  checks the result against plain builds.
- COSX has no OO route. The Python entry point raises `ValueError` for
  `scf_jk="rijcosx"`.

## 12.1.5 The kernel

The denominator response, from
`ferric-mp2/src/oo_rimp2.rs::fock_response_gradient`, after
$W_{\mathrm{AO}} = CWC^{\mathsf T}$ has been passed through the J/K route:

Abridged from `ferric-mp2/src/oo_rimp2.rs::fock_response_gradient` (not a verbatim excerpt):
```rust
let g_ao = &w_jk.j - &(0.5 * &w_jk.k);
let g_mo = c.t().dot(&g_ao).dot(c);
let fw = f_mo.dot(&w);
let wf = w.dot(f_mo);
// ...
g[(a, i)] = 4.0 * g_mo[(a_mo, i_mo)] + 2.0 * (fw[(a_mo, i_mo)] - wf[(a_mo, i_mo)]);
```

| Code | Equation |
|---|---|
| `w` | $W = P + P^{\mathsf T}$ of (12.4), embedded in the $n_{\mathrm{mo}}\times n_{\mathrm{mo}}$ matrix |
| `w_jk.j`, `w_jk.k` | $J[CWC^{\mathsf T}]$, $K[CWC^{\mathsf T}]$ |
| `g_mo` | $C^{\mathsf T}\mathcal G[CWC^{\mathsf T}]C$ |
| `f_mo` | $F$ in the semicanonical frame |
| `g[(a, i)]` | the third and fourth terms of (12.3) |

The first two terms of (12.3) are added in
`hf_and_integral_response_gradient_timed`.

## 12.1.6 Reproducibility and memory

- The gradient GEMMs run in fixed blocks with one-thread BLAS per block, so
  the result does not depend on the thread count. The test
  `test_oo_gradient_bit_identical_across_thread_counts` checks this.
- Every gradient evaluation builds the full MO three-index tensor
  $B^P_{pq}$, of $N_{\mathrm{aux}}n_{\mathrm{mo}}^2$ doubles, and the dense
  amplitudes, of $n_o^2n_v^2$ doubles.
  `check_b_full_and_t2_alloc` refuses a run in which
  $N_{\mathrm{aux}}n_{\mathrm{mo}}^2 + 2(n_on_v)^2$ doubles exceed the
  budget. It runs before both allocations, and again before the
  convergence recheck of step 9.
- The MO-side transform is charged against the memory pool
  (`charge_mo_side`) as a hard limit.
- The spilled and in-core paths agree; `test_spill_budget_paths_match_incore`
  checks this.

## 12.1.7 Options

| Option | Default | Effect |
|---|---|---|
| `grad_conv` (`oo_grad_conv`) | $10^{-4}$ | bound on $\lVert g\rVert_2$ |
| `energy_conv` | $10^{-8}$ Ha | energy-change test of step 9 |
| `max_iter` (`oo_max_iter`) | 100 | orbital iterations |
| `level_shift` (`oo_level_shift`) | 0.1 Ha | $\mu$ in step 3 |
| `diis_size` (`oo_diis_size`) | 6 | DIIS subspace; `use_diis` turns DIIS off |
| `frozen_core` | 0 | removes core orbitals from $i, j$ and from the rotations |
| `difference_jk` | off | incremental exact J/K |
| external potential, ECP | none | folded into $h$ by `hcore_ecp_with_external` |

- A relativistic reference is refused, because the loop rebuilds the
  nonrelativistic core Hamiltonian.
- The analytic nuclear gradient (Python `with_gradient=True`,
  `task = "optimize"`) requires `frozen_core = 0`. It is valid only at a
  converged point; an unconverged loop raises an error.

## 12.1.8 Anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| OO-RI-MP2 total energy, H2O and NH3 / cc-pVDZ | independent numpy OO-RI-MP2 | $7.5\times10^{-13}$ Ha measured, bar $10^{-11}$ | `oo_rimp2_h2o_ccpvdz_energy_and_gradient_vs_orca`, `oo_rimp2_nh3_ccpvdz_energy_and_gradient_vs_orca` in `crates/ferric-mp2/tests/validation_oo_rimp2.rs` |
| Same minimum | ORCA 6.1.1 | ORCA stops $3.7\times10^{-8}$ to $7.0\times10^{-8}$ Ha above it | same tests |
| Analytic nuclear gradient | finite difference of ferric's own energy | $8.0\times10^{-9}$ Ha/Bohr measured, bar $10^{-7}$ | `oo_rimp2_h2o_ccpvdz_gradient_self_fd`, `oo_rimp2_nh3_ccpvdz_gradient_self_fd` |
| Converged point is stationary | $g = 0$ | asserted | `test_oo_rimp2_converged_gradient_vanishes_h2o_ccpvdz` and its siblings in `crates/ferric-mp2/src/oo_rimp2.rs` |

<!-- figures: site/src/reference/validation.md row "oo-rimp2"; crates/ferric-mp2/tests/validation_oo_rimp2.rs module doc table -->

The validation test also runs negative controls. Plain RI-MP2 must miss the
OO energy by more than $10^{-6}$ Ha (the measured gaps are
$9.2\times10^{-4}$ Ha for H2O and $8.7\times10^{-4}$ Ha for NH3). It must
also miss the gradient by more than $10^{-5}$ Ha/Bohr.

The validation page grades the method Proven (narrow). The analytic gradient
is also Proven (narrow), for a closed shell with frozen core 0 on the exact
and RI-JK J/K routes (section 15.3).

**Known limits (2026-11-15).**

- The step uses a diagonal Hessian. There is no second-order or
  trust-region solver.
- Frozen-core rotations are not optimized.
- COSX and range-separated exchange references have no OO route.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, section 7 | equations 12.1 to 12.4, the step |
| `crates/ferric-mp2/src/oo_rimp2.rs`, `orbital_rotation.rs` | every rule in the algorithm table, read from the loop |
| `scripts/oomp2_mp3_identities_sympy.py`, part B; `scripts/mp2_variant_identities.py`, claim 6 | verified equations 12.2 and 12.3 |
| U. Bozkaya and C. D. Sherrill, J. Chem. Phys. 135, 104103 (2011) | the functional and the level-shifted step, as cited in the code |
| `site/src/reference/validation.md`; `crates/ferric-mp2/tests/validation_oo_rimp2.rs` | the anchor figures |

Changes from the sources:

- The amplitudes are written $t = -(ia|jb)/\Delta$, with a positive $\Delta$,
  as in `theory.md`.
- The DIIS error uses the rotated coefficients $C_{\mathrm{new}}$, and the
  convergence recheck and the "last halved step" rule are stated from the
  loop; `theory.md` section 7 states them the same way.
