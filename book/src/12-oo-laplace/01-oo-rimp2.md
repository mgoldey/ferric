# Closed-shell orbital-optimized RI-MP2 {#oo-rimp2}

<!-- status: ready; registry: oo-rimp2 -->

*Notation follows `book/notation.md`. Line numbers are left out because they
drift; each code reference names a function.*

Hartree–Fock orbitals minimize the energy of one determinant, in which each
electron feels the average field of the others. Correlated electrons avoid
each other, which changes that field, so the best orbitals for the
correlated energy differ slightly from the Hartree–Fock ones.
Orbital-optimized MP2 treats the orbitals as variational parameters of the
Hartree–Fock plus second-order energy and rotates them until the total is
stationary. At a stationary point the nuclear gradient needs no orbital
response (the Z-vector solve of section @sec:lagrangian-zvector), so it takes the form of a
variational SCF gradient. The relaxation also lowers the energy: for H2O and
NH3 in cc-pVDZ, plain RI-MP2 lies $9.2\times10^{-4}$ Ha and
$8.7\times10^{-4}$ Ha above the OO-RI-MP2 minimum (section @sec:oo-rimp2-anchors).

The plan:

1. Write the functional, and show why it is invariant to rotations within
   the occupied and within the virtual space (@sec:oo-rimp2-functional).
2. Parametrize the rotation (@sec:oo-rimp2-rotation) and differentiate the functional
   (@sec:oo-rimp2-orbital-gradient).
3. Read the optimization loop and its kernel (@sec:oo-rimp2-algorithm to @sec:oo-rimp2-kernel), then the
   memory rules, options and anchors.

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
driver (section @sec:unrestricted-oo-rimp2). The Python entry point runs closed shell only.

## The functional {#oo-rimp2-functional}

Take a coefficient matrix $C$ with $n_{\mathrm{occ}}$ doubly occupied
columns. Build the density over all occupied orbitals, frozen core
included, $D = 2C_oC_o^{\mathsf T}$, and its Fock matrix
$F = h + J[D] - \tfrac12 K[D]$. The functional is

$$ E(C) = \tfrac12\operatorname{Tr}\big[D\,(h + F)\big] + V_{nn} + E^{(2)}_{\mathrm{RI}}(\tilde C), \qquad \tilde C = C\begin{pmatrix} U_o & \\ & U_v\end{pmatrix}. \label{eq:oo-rimp2-1} $$

The first two terms are the Hartree–Fock energy of the determinant built
from $C$. The last term is RI-MP2, equation (@eq:closed-shell-ri-mp2-3), evaluated in rotated
orbitals $\tilde C$:

- $U_o$ diagonalizes the active occupied block of $C^{\mathsf T}FC$, and
  $U_v$ the virtual block. The frozen-core block is not touched
  (`ferric-mp2/src/oo_rimp2.rs::semicanonical_rotation`).
- The denominators of (@eq:closed-shell-ri-mp2-3) use the diagonal elements of
  $\tilde C^{\mathsf T}F\tilde C$.

$\tilde C$ is the **semicanonical** frame. Away from the Hartree–Fock
solution, $C^{\mathsf T}FC$ has a non-zero occupied–virtual block, so no
rotation of $C$ makes $F$ diagonal. The semicanonical frame makes the two
diagonal blocks diagonal and leaves the off-diagonal block alone. That is
enough for the MP2 formula of chapter @sec:mp2, which only reads the
occupied–occupied and virtual–virtual blocks through the orbital energies.

The code evaluates (@eq:oo-rimp2-1) in `ferric-mp2/src/oo_rimp2.rs::evaluate_point`:
first `compute_hf_energy`, then the rotation, then
`compute_rimp2_with_orbitals`.

**Why only occupied–virtual rotations matter.** Rotate $C$ among its active
occupied columns. $D$ does not change, so neither do $F$ and the
Hartree–Fock energy. The semicanonical rotation then absorbs the change:
$\tilde C$ is the same set of orbitals, up to sign and order, and
$E^{(2)}$ is a sum over all pairs, so it does not depend on either. The same
holds for the virtual columns. So $E$ is invariant to occupied–occupied and
virtual–virtual rotations, and making it stationary under occupied–virtual
rotations makes it stationary under all of them.
`test_oo_energy_is_invariant_to_occ_occ_and_vir_vir_rotations` in
`crates/ferric-mp2/src/oo_rimp2.rs` tests the invariance directly.

(@eq:oo-rimp2-1) is the full-Fock Hylleraas functional of Bozkaya and Sherrill (OMP2).
The off-diagonal occupied–virtual Fock block enters only through the orbital
gradient (@eq:oo-cs-gradient); the amplitudes ignore it.

**Frozen core.** Only active-occupied/virtual rotations are optimized.
Frozen-core orbitals are not rotated into the virtuals, so
`frozen_core > 0` gives a constrained minimum: stationary in the
active–virtual rotations, not in the core–active or core–virtual ones.

## The rotation {#oo-rimp2-rotation}

The rotation follows notation section @sec:notation-orbital-rotations. $\kappa$ is antisymmetric, with
$\kappa_{ai}$ stored and $\kappa_{ia} = -\kappa_{ai}$. The update is
$C \leftarrow C\,U(\kappa)$ with the Cayley form

$$ U(\kappa) = \big(1 - \tfrac12\kappa\big)^{-1}\big(1 + \tfrac12\kappa\big). \label{eq:oo-cayley} $$

Section @sec:second-order-orbital-gradient-hessian derives the two properties that make (@eq:oo-cayley) a valid
parametrization: $U$ is exactly orthogonal for antisymmetric $\kappa$, and
its expansion agrees with $e^{\kappa}$ through $\kappa^2$. The first and
second derivatives of $E(CU(t\kappa))$ at $t = 0$ are therefore those of the
exponential map. Section @sec:second-order-rotation shows why one inverse is enough to form it.

To first order $U \approx 1 + \kappa$, so $C \leftarrow C(1 + \kappa)$ and
the generator is $+\kappa$. With the gradient $g = +\partial E/\partial\kappa$
the descent step is $\kappa = -g/(\text{gap} + \mu)$ (section @sec:oo-rimp2-algorithm).
`crates/ferric-mp2/src/orbital_rotation.rs` fixes this convention for both
OO drivers.

> **Verified equation @eq:oo-cayley** · Cayley rotation
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

## The orbital gradient {#oo-rimp2-orbital-gradient}

The gradient is $g_{ai} = +\partial E/\partial\kappa_{ai}$. Start with what a
single rotation $\kappa_{ck}$ does to the orbitals. With $U \approx 1 + \kappa$
and $\kappa_{ck} = -\kappa_{kc}$, the first-order change of $C$ is

$$ \delta C_k = C_c, \qquad \delta C_c = -C_k , $$

per unit $\kappa_{ck}$: the occupied orbital $k$ picks up some of $c$, and
$c$ gives up the same amount of $k$.

$E$ depends on $C$ in three ways, and the chain rule gives one term for
each.

**1. The Hartree–Fock energy.** $D = 2\sum_k C_kC_k^{\mathsf T}$ changes by
$\delta D = 2(C_cC_k^{\mathsf T} + C_kC_c^{\mathsf T})$. The derivative of the
Hartree–Fock energy is $\operatorname{Tr}[F\,\delta D]$. Substitute
$\delta D$: the two outer products each give $F_{ck}$, so the term is
$4F_{ck}$, the Brillouin condition: it vanishes at the
Hartree–Fock solution.

**2. The integrals, at fixed amplitudes.** The MP2 part is the Hylleraas
functional, which is stationary in the amplitudes. So the amplitudes'
response to the rotation drops out, and we differentiate at fixed $t$
(`scripts/oomp2_mp3_identities_sympy.py` checks this: the gradient at fixed
$t$ equals a finite difference of the fully relaxed energy). Substitute
$\delta C$ into each index of $(ia|jb)$. An occupied index equal to $k$
becomes $c$; a virtual index equal to $c$ becomes $-k$:

$$ \frac{\partial (ia|jb)}{\partial\kappa_{ck}} = \delta_{ik}(ca|jb) + \delta_{jk}(ia|cb) - \delta_{ac}(ik|jb) - \delta_{bc}(ia|jk) . $$

Contracted with the amplitude combination $\tilde T = 2t - t^{\mathrm{swap}}$
of (@eq:closed-shell-ri-mp2-3), these four terms give the integral response $G_V + G_O$: the
two occupied terms contract the amplitudes with $B^P_{ca}$, and the two
virtual terms with the $(ik|jb)$ block.

**3. The denominators.** The orbital energies are diagonal elements of $F$,
and $F$ changes with $C$. The derivative of $E^{(2)}$ with respect to the
Fock matrix, at fixed integrals, is the unrelaxed MP2 density $W$. The
change of $F$ in the MO basis has two parts: the rotation of the basis,
$FX - XF$ with $X$ the unit generator, and the change of the density,
$C^{\mathsf T}\mathcal G[\delta D]C$. Contract each with $W$. For the second
part use the symmetry $\operatorname{Tr}(X\,\mathcal G[Y]) = \operatorname{Tr}(Y\,\mathcal G[X])$
to move $\mathcal G$ off $\delta D$ and onto $CWC^{\mathsf T}$; the same
two-outer-product count as in term 1 gives the factor 4.

Assembled in the semicanonical frame:

$$ g_{ck} = 4F_{ck} + 2\big(G_V + G_O\big)_{ck} + 4\big[C^{\mathsf T}\,\mathcal G[CWC^{\mathsf T}]\,C\big]_{ck} + 2\big(FW - WF\big)_{ck}, \qquad \mathcal G[X] = J[X] - \tfrac12K[X]. \label{eq:oo-cs-gradient} $$

The code computes this in the semicanonical frame and rotates it back to
the frame of $C$, $g = U_v\,g_{\mathrm{sc}}\,U_o^{\mathsf T}$. The invariance
of @sec:oo-rimp2-functional is what makes the back-rotation valid:
$E(CU(\kappa)) = E(\tilde CU(U^{\mathsf T}\kappa U))$. The solver keeps its own
frame because the semicanonical frame jumps between iterations (eigenvector
signs and order are arbitrary), and DIIS needs one continuous frame.

The pieces in the code:

- $4F_{ck}$ and $G_V + G_O$ are added in
  `ferric-mp2/src/oo_rimp2.rs::hf_and_integral_response_gradient_timed`,
  with the GEMM kernels in `ferric-mp2/src/orbital_gradient_gemm.rs`.
- The last two terms are the denominator response, built from
  $W = P + P^{\mathsf T}$ in `ferric-mp2/src/oo_rimp2.rs::fock_response_gradient`.

The two blocks of $P$ are

$$ P_{ij} = -\sum_{kab} t_{ik}^{ab}\big(2t_{jk}^{ab} - t_{jk}^{ba}\big), \qquad P_{ab} = \sum_{ijc} t_{ij}^{ac}\big(2t_{ij}^{bc} - t_{ij}^{cb}\big), \qquad t_{ij}^{ab} = -\frac{(ia|jb)}{\Delta_{ij}^{ab}} . \label{eq:oo-rimp2-4} $$

Read the signs physically. $P_{ab}$ is positive on the diagonal: correlation
puts electron density into the virtuals. $P_{ij}$ is negative: it takes the
same amount out of the occupied orbitals. The code stores $t$ dense as an
$(n_on_v)\times(n_on_v)$ matrix and builds $P$ in
`ferric-mp2/src/oo_rimp2.rs::build_mp2_density`.

Check the Hartree–Fock limit. Without correlation, $t = 0$, so $G_V$, $G_O$,
$P$ and $W$ vanish, and $g_{ck} = 4F_{ck}$: the orbital gradient of
Hartree–Fock, zero at the SCF solution.

> **Verified equation @eq:oo-cs-gradient** · closed-shell OO-MP2 orbital gradient
>
> On a random closed-shell model in the semicanonical frame with
> $F_{ov} \ne 0$, the gradient (@eq:oo-cs-gradient), with $W$ from (@eq:oo-rimp2-4), equals
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
> | Code test | `test_oo_rimp2_gradient_fd_off_reference_h2o_sto3g` and `test_oo_rimp2_gradient_fd_off_reference_nh3_631g` in `crates/ferric-mp2/src/oo_rimp2.rs`: the production gradient against a finite difference of (@eq:oo-rimp2-1), away from the HF point |
>
> <!-- verified: id=oo.cs.gradient; script=scripts/oomp2_mp3_identities_sympy.py; claim=B3; runner=crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py; code=ferric-mp2/src/oo_rimp2.rs::fock_response_gradient; rust_test=ferric-mp2/src/oo_rimp2.rs::test_oo_rimp2_gradient_fd_off_reference_h2o_sto3g -->

## The algorithm {#oo-rimp2-algorithm}

The solver takes a level-shifted diagonal step, as in Bozkaya and Sherrill:
divide each gradient element by its orbital-energy gap plus a shift $\mu$.
The shift keeps the step finite when a gap is small. One iteration of
`ferric-mp2/src/oo_rimp2.rs::oo_ri_mp2`:

| Step | Operation | Function |
|---|---|---|
| 1 | Evaluate (@eq:oo-rimp2-1) at $C$: $J$, $K$, $F$, $E_{\mathrm{HF}}$, semicanonical rotation, $B$, $t$, $E^{(2)}$ | `evaluate_point` |
| 2 | Gradient (@eq:oo-cs-gradient), back-rotated; stop if $\lVert g\rVert_2 <$ `grad_conv` | `compute_orbital_gradient` |
| 3 | Step $\kappa_{ai} = -g_{ai}/(\varepsilon_a - \varepsilon_i + \mu)$, $\varepsilon = \operatorname{diag}(C^{\mathsf T}FC)$ in the frame of $C$ | main loop |
| 4 | If $\max\lvert\kappa_{ai}\rvert > 0.3$, scale $\kappa$ uniformly to 0.3 rad | main loop |
| 5 | $C_{\mathrm{new}} = C\,U(\kappa)$ | `cayley_rotation` |
| 6 | DIIS on $C_{\mathrm{new}}$ with error $C_{\mathrm{new}}\,G\,C_{\mathrm{new}}^{\mathsf T}$, $G$ the antisymmetric embedding of $g$; then $C \leftarrow C(C^{\mathsf T}SC)^{-1/2}$ | `lowdin_orthonormalize` |
| 7 | Evaluate the trial point | `evaluate_point` |
| 8 | If $E_{\mathrm{trial}} > E + 10^{-4}$: halve $\kappa$ up to ten times without DIIS, stopping at the first trial with $E \le E + 10^{-12}$; reset DIIS | main loop |
| 9 | If $\lvert E_{\mathrm{trial}} - E\rvert <$ `energy_conv` and the iteration is past the first: recompute the gradient at the accepted point and stop if $\lVert g\rVert_2 < 10\,$`grad_conv`; otherwise that gradient is reused at the next iteration. The energy change is the one measured before any halving | main loop |

Notes on the table:

- **Step 3.** The diagonal of $C^{\mathsf T}FC$ serves only as a
  preconditioner. The gradient and the energy use the semicanonical frame.
- **The step length.** Compare the denominator with the exact Hessian. At
  the Hartree–Fock point the diagonal of the Hartree–Fock part of
  $\partial^2E/\partial\kappa_{ai}^2$ is
  $4\big[(\varepsilon_a - \varepsilon_i) + 3(ai|ai) - (aa|ii)\big]$ (section
  @sec:second-order-orbital-gradient-hessian, which stores a quarter of it). The step's denominator has no
  factor 4 and no two-electron terms, so it is not the diagonal Newton step;
  it is longer than that step by roughly a factor 4. The 0.3 rad cap of step
  4 and the backtracking of step 8 bound it.
- **Step 6.** DIIS returns a linear combination of coefficient matrices,
  which is not orthonormal. The energy of a non-orthonormal $C$ is not a
  determinant energy, and (@eq:oo-cs-gradient) assumes orthonormality, so Löwdin
  orthonormalization follows every extrapolation. It raises an error if
  $C^{\mathsf T}SC$ is not positive definite.
- **Step 8.** If none of the ten halvings is accepted, the closed-shell
  driver takes the last halved step anyway. The unrestricted driver
  (section @sec:unrestricted-oo-rimp2) keeps the old point instead.
- **Step 9.** The energy change is measured on the trial of step 7, before
  any backtracking.
- **The norm.** The stopping test uses the Euclidean norm over the active
  occupied–virtual block, not the largest element.

**J and K.** Each evaluation rebuilds $J$ and $K$ on the reference SCF's
route (`OoJk`).

- When the reference ran RI-J plus RI-K with one auxiliary basis and no COSX
  or range-separated exchange, the OO loop uses that fitted route. The HF
  part of (@eq:oo-rimp2-1) is then the fitted HF energy. The test
  `oo_hf_energy_on_an_rijk_reference_is_the_scf_energy` checks that it equals
  the SCF energy at the reference orbitals. The derivation of (@eq:oo-cs-gradient) used
  only that $\mathcal G$ is linear and symmetric, which the Coulomb-metric
  fit satisfies, so (@eq:oo-cs-gradient) holds on this route unchanged. The
  nuclear gradient follows the same fitted route, through
  `ferric-scf/src/response_jk.rs::ResponseJk` (the fitted bilinear
  two-electron derivative of section @sec:rimp2-gradient).
- Otherwise the loop builds exact four-centre $J$ and $K$ at a $10^{-12}$
  threshold. `difference_jk` optionally builds them from the density change
  since the previous point; `oo_rimp2_difference_jk_matches_plain_builds`
  checks the result against plain builds.
- COSX has no OO route. The Python entry point raises `ValueError` for
  `scf_jk="rijcosx"`.

## The kernel {#oo-rimp2-kernel}

The denominator response is the last two terms of (@eq:oo-cs-gradient). Below is the end of
`ferric-mp2/src/oo_rimp2.rs::fock_response_gradient`, after $W$ has been
assembled from (@eq:oo-rimp2-4) and $W_{\mathrm{AO}} = CWC^{\mathsf T}$ passed through
the J/K route; the timing calls and the `OrbitalSpace` destructuring are
removed, and the comments are added:

```rust
let g_ao = &w_jk.j - &(0.5 * &w_jk.k);       // G[C W C^T] = J - K/2
let g_mo = c.t().dot(&g_ao).dot(c);          // C^T G[C W C^T] C
let fw = f_mo.dot(&w);                       // F W  (semicanonical F)
let wf = w.dot(f_mo);                        // W F
let mut g = Array2::<f64>::zeros((nvir, nocc));
for a in 0..nvir {
    let a_mo = nocc_total + a;
    for i in 0..nocc {
        let i_mo = first_occ + i;            // active occupied only
        g[(a, i)] = 4.0 * g_mo[(a_mo, i_mo)] + 2.0 * (fw[(a_mo, i_mo)] - wf[(a_mo, i_mo)]);
    }
}
```

Three points about it:

- **One J/K build.** The density response costs one $J$ and one $K$ build
  on $CWC^{\mathsf T}$, the same cost as one SCF Fock build.
- **The indices.** `i` runs over the active occupied orbitals only, so the
  frozen-core columns of $g$ never appear and the core is never rotated.
- **The rest of (@eq:oo-cs-gradient).** The first two terms are added in
  `hf_and_integral_response_gradient_timed`.

## Reproducibility and memory {#oo-rimp2-reproducibility-memory}

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

## Options {#oo-rimp2-options}

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
  converged point; an unconverged loop raises an error. Both rules follow
  from @sec:oo-rimp2-functional: the gradient assumes zero orbital response, which holds only
  where every rotation the energy depends on is stationary.

## Anchors {#oo-rimp2-anchors}

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
and RI-JK J/K routes (section @sec:oo-rimp2-gradient).

**Known limits (2026-11-15).**

- The step uses a diagonal Hessian. There is no second-order or
  trust-region solver.
- Frozen-core rotations are not optimized.
- COSX and range-separated exchange references have no OO route.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/oo_rimp2.rs`, `orbital_rotation.rs` | every rule in the algorithm table, read from the loop |
| `scripts/oomp2_mp3_identities_sympy.py`, part B; `scripts/mp2_variant_identities.py`, claim 6 | verified equations @eq:oo-cayley and @eq:oo-cs-gradient; the fixed-amplitude derivative |
| U. Bozkaya and C. D. Sherrill, J. Chem. Phys. 135, 104103 (2011) | the functional and the level-shifted step, as cited in the code |
| `site/src/reference/validation.md`; `crates/ferric-mp2/tests/validation_oo_rimp2.rs` | the anchor figures |

Changes from the sources:

- The amplitudes are written $t = -(ia|jb)/\Delta$, with a positive $\Delta$.
- The DIIS error uses the rotated coefficients $C_{\mathrm{new}}$, and the
  convergence recheck and the "last halved step" rule are stated from the
  loop.
- The Hessian diagonal quoted under "The step length" comes from section
  @sec:second-order-orbital-gradient-hessian and was checked by a finite difference on a model RHF energy.
