# CCD and CCSD in spin orbitals {#ccd-ccsd-spin-orbital}

<!-- status: ready; registry: ccd, ccsd;
     theory: crates/ferric-cc/theory.md#"CCD", "CCSD in spin orbitals" -->

MP2 lets each electron pair correlate once, in the field of a frozen
Hartree–Fock background. In a real molecule the pairs also push on each
other, so the amplitudes should be solved together, and a method truncated at
double excitations must still describe two pairs excited at the same time in
distant parts of a molecule. Coupled cluster does both: it writes the
wavefunction as $e^{\hat T}|\Phi_0\rangle$, whose powers of $\hat T$ supply
the simultaneous excitations, and fixes the amplitudes by projection. The
spin-orbital solvers of this section transcribe the textbook equations with
antisymmetrized integrals; `ccd` is the production CCD, and the spin-orbital
`ccsd` is the cross-check of the closed-shell CCSD of section @sec:closed-shell-ccsd.

The plan: @sec:ccd-ccsd-spin-orbital-equations motivates the exponential, derives the projected
equations, reads the CCD residual term by term and checks three limits (MP2,
CCD inside CCSD, two-electron full CI); @sec:ccd-ccsd-spin-orbital-algorithm shows the loop that solves
them.

| Index | Range | Code name |
|---|---|---|
| $i, j, k, l, m, n$ | active occupied spin orbitals | `0..2*no` |
| $a, b, c, d, e, f$ | virtual spin orbitals | `0..2*nv` |
| $P$ | auxiliary | `0..naux` |

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/ccd.rs::{ccd,ccd_spinorbital}`, `ferric-cc/src/ccsd.rs::ccsd` |
| CLI | `method.kind = "ccd"` (`examples/water-ccd.toml`); the spin-orbital CCSD has no CLI kind |
| Python | `run_ccd` |

## The equations {#ccd-ccsd-spin-orbital-equations}

**Why an exponential.** Take two molecules A and B far apart. Each has its
own doubles operator, $\hat T_2^A$ and $\hat T_2^B$. The excitation
operators commute, so

$$ e^{\hat T_2^A + \hat T_2^B}|\Phi_0\rangle = e^{\hat T_2^A}e^{\hat T_2^B}|\Phi_0\rangle , $$

a product of the two monomer wavefunctions, and the energy is the sum of the
two monomer energies. Expand one exponential:
$e^{\hat T_2} = 1 + \hat T_2 + \tfrac12\hat T_2^2 + \dots$ The term
$\tfrac12\hat T_2^2$ holds the quadruple excitations "pair in A excited and
pair in B excited", with coefficients that are products of doubles
amplitudes. A linear doubles expansion (CID) has no such term, so its energy
for A + B is not the sum of the two. Energies that add for non-interacting
fragments are called size-consistent. This is why coupled cluster
parametrizes the wavefunction through $\hat T$ (`theory.md`, "Exponential ansatz"):

$$|\Psi_{CC}\rangle = e^{\hat{T}} |\Phi_0\rangle,\qquad \hat{T} = \hat{T}_1 + \hat{T}_2 + \dots$$

**The projected equations.** Insert the ansatz into
$\hat H|\Psi\rangle = E|\Psi\rangle$ and multiply on the left by
$e^{-\hat T}$:

$$ e^{-\hat T}\hat He^{\hat T}|\Phi_0\rangle = E\,|\Phi_0\rangle . $$

Project onto $\langle\Phi_0|$ for the energy and onto each excited
determinant $\langle\Phi_\mu|$ in the space of $\hat T$ for the amplitude
equations. The right-hand side has no component on $\Phi_\mu$, so

$$ R_\mu = \langle\Phi_\mu|e^{-\hat T}\hat He^{\hat T}|\Phi_0\rangle = 0,\qquad
E_c = \langle\Phi_0|e^{-\hat T}\hat He^{\hat T}|\Phi_0\rangle - E_{\rm HF} . \label{eq:cc-ccd-residual} $$

**Why the transform terminates.** Expand by the Baker–Campbell–Hausdorff
series:

$$ e^{-\hat T}\hat He^{\hat T} = \hat H + [\hat H,\hat T] + \tfrac12[[\hat H,\hat T],\hat T]
 + \tfrac16[[[\hat H,\hat T],\hat T],\hat T] + \tfrac1{24}[[[[\hat H,\hat T],\hat T],\hat T],\hat T] . $$

The series stops after the fourfold commutator. The excitation operators in
$\hat T$ commute with each other, so each commutator must contract a
$\hat T$ with an index of $\hat H$, and a two-electron $\hat H$ has four.
On the ten-spin-orbital model of the proof script below, with random
$t_1$ and $t_2$, the largest matrix element of the fourfold commutator is
0.10 and that of the fivefold one is below $4\times10^{-18}$. Projected onto doubles, the CCD residual is
therefore at most quadratic in $t_2$, and the CCSD residual at most quartic
in $t_1$.

With canonical orbitals the Fock operator contributes only
$\Delta^{ab}_{ij}t^{ab}_{ij}$ to the doubles (and $\Delta^a_i t^a_i$ to the
singles). The code writes $R = \Delta t + r$ and iterates
$t \leftarrow r/(-\Delta)$.

**CCD.** Keep only $\hat T_2 = \tfrac14\sum t_{ij}^{ab}\, a^\dagger_a a^\dagger_b a_j a_i$,
with $t$ antisymmetric in $ij$ and in $ab$. The doubles projection
$\langle\Phi_{ij}^{ab}| e^{-\hat T_2}\hat H e^{\hat T_2} |\Phi_0\rangle = 0$ is
$R_{ij}^{ab} = \Delta_{ij}^{ab} t_{ij}^{ab} + r_{ij}^{ab}$
(`ferric-cc/src/ccd.rs::ccd_spinorbital`; `ccd` delegates to it), with

$$
\begin{aligned}
r_{ij}^{ab} ={}& \langle ij\|ab\rangle
 + \tfrac12\sum_{cd}\langle ab\|cd\rangle t_{ij}^{cd}
 + \tfrac12\sum_{kl}\langle kl\|ij\rangle t_{kl}^{ab}
 + P(ij)P(ab)\sum_{kc}\langle kb\|cj\rangle t_{ik}^{ac} \\
 &+ \tfrac14\sum_{klcd}\langle kl\|cd\rangle t_{ij}^{cd} t_{kl}^{ab}
 + \tfrac12 P(ij)P(ab)\sum_{klcd}\langle kl\|cd\rangle t_{ik}^{ac} t_{jl}^{bd} \\
 &- \tfrac12 P(ab)\sum_{klcd}\langle kl\|cd\rangle t_{ij}^{ac} t_{kl}^{bd}
 - \tfrac12 P(ij)\sum_{klcd}\langle kl\|cd\rangle t_{ik}^{ab} t_{jl}^{cd},
\end{aligned}
$$

and $P(ij)x = x - x_{i\leftrightarrow j}$
(`ferric-cc/src/helpers.rs::{p_ij,p_ab,p_ij_ab}`). Read the terms in order:

- $\langle ij\|ab\rangle$ is the driver, the same integral that drives MP2.
- The two ladders let the excited pair scatter: the particles $a,b$ off each
  other into $c,d$ (particle–particle), or the holes $i,j$ into $k,l$
  (hole–hole).
- The ring term moves one particle–hole pair through $\langle kb\|cj\rangle$
  into another; $P(ij)P(ab)$ makes the result antisymmetric.
- The four quadratic terms come from $\tfrac12\hat T_2^2$: two pairs
  excited at once, projected back onto the doubles by one integral.

The energy needs only the linear term of the series, since $\hat H$ can
de-excite at most a double:

$$E_c = \tfrac14 \sum_{ijab} \langle ij\|ab\rangle\, t_{ij}^{ab}.$$

Its closed-shell spatial form is
$E_c = \sum_{ijab} (ia|jb)\,(2 t_{ij}^{ab} - t_{ij}^{ba})$, equation (@eq:so-cc-amplitude-expansion).

**The MP2 limit.** Start from $t = 0$. Every term of $r$ except the driver
vanishes, so one Jacobi step gives
$t_{ij}^{ab} = \langle ij\|ab\rangle/(-\Delta_{ij}^{ab})$, the first-order
amplitude. Put it in the energy:
$\tfrac14\sum\langle ij\|ab\rangle^2/(-\Delta) = -\tfrac14\sum|\langle ij\|ab\rangle|^2/\Delta$,
which is the MP2 energy (@eq:closed-shell-ri-mp2-1). The code starts from this guess, so the
energy it computes at iteration 0 is MP2. It then stops when the energy
changes by less than $10^{-10}$, with DIIS on the increment.

> **Verified equation @eq:cc-ccd-residual** · the CCD residual and energy
>
> On a ten-spin-orbital model with a diagonal Fock matrix by construction,
> the eight-term $r + \Delta t$ of `ccd_spinorbital` at random antisymmetric
> $t_2$ equals $\langle\Phi_{ij}^{ab}|e^{-\hat T_2}\hat He^{\hat T_2}|\Phi_0\rangle$
> built from Jordan–Wigner matrices, and $\tfrac14\langle ij\|ab\rangle t$
> equals the projected energy. The brute-force route has no diagrams, no
> intermediates and no Wick contractions.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_ccsd_eom_brute_force_sympy.py`, check `ccd-residual` |
> | Negative control | the runner's mutation: the $\tfrac12$ of the second quadratic term set to 1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/ccd.rs::ccd_spinorbital`, `ferric-cc/src/helpers.rs::{p_ij,p_ab,p_ij_ab}` |
> | Code test | `ccd_matches_df_reference` in `crates/ferric-cc/tests/validation_cc.rs` (against PySCF `CCD` on the same DF integrals) |
>
> <!-- verified: id=cc.ccd.residual; script=scripts/cc_ccsd_eom_brute_force_sympy.py; claim=ccd-residual; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/ccd.rs::ccd_spinorbital; rust_test=ferric-cc/tests/validation_cc.rs::ccd_matches_df_reference; registry=ccd -->

**CCSD.** Add the singles, $\hat T_1 = \sum t_i^a a^\dagger_a a_i$, so
$\hat T = \hat T_1 + \hat T_2$. There are now two projections,
$R_i^a = \langle\Phi_i^a| e^{-\hat T}\hat H e^{\hat T} |\Phi_0\rangle = 0$ and
$R_{ij}^{ab} = \langle\Phi_{ij}^{ab}| e^{-\hat T}\hat H e^{\hat T} |\Phi_0\rangle = 0$.
Physically, $\hat T_1$ relaxes the orbitals in the presence of correlation:
$e^{\hat T_1}$ applied to a determinant gives another determinant with
rotated orbitals. $\hat T_2$ carries the pair correlation.

`ferric-cc/src/ccsd.rs::ccsd` evaluates the residuals in the factorization of
Stanton, Gauss, Watts and Bartlett, JCP 94, 4334 (1991), with $f_{ia} = 0$
and the diagonal Fock moved into $\Delta$. The singles enter the doubles
mostly through two dressed amplitudes,

$$\tilde\tau_{ij}^{ab} = t_{ij}^{ab} + \tfrac12(t_i^at_j^b - t_j^at_i^b),\qquad
\tau_{ij}^{ab} = t_{ij}^{ab} + t_i^at_j^b - t_j^at_i^b ,$$

where $\tau$ is the amplitude of $\hat T_2 + \tfrac12\hat T_1^2$ on a doubly
excited determinant. The one-body intermediates are

$$
\begin{aligned}
F_{ae} &= \sum_{mf} t_m^f\langle ma\|fe\rangle - \tfrac12\sum_{mnf}\tilde\tau_{mn}^{af}\langle mn\|ef\rangle, &
F_{mi} &= \sum_{ne} t_n^e\langle mn\|ie\rangle + \tfrac12\sum_{nef}\tilde\tau_{in}^{ef}\langle mn\|ef\rangle, \\
F_{me} &= \sum_{nf} t_n^f\langle mn\|ef\rangle, &
W_{mnij} &= \langle mn\|ij\rangle + P(ij)\sum_e t_j^e\langle mn\|ie\rangle + \tfrac14\sum_{ef}\tau_{ij}^{ef}\langle mn\|ef\rangle,
\end{aligned}
$$

and the two two-body intermediates of the ladder and the ring are

$$
W_{abef} = \langle ab\|ef\rangle - P(ab)\sum_m t_m^b\langle am\|ef\rangle + \tfrac14\sum_{mn}\tau_{mn}^{ab}\langle mn\|ef\rangle,
$$

$$
W_{mbej} = \langle mb\|ej\rangle + \sum_f t_j^f\langle mb\|ef\rangle - \sum_n t_n^b\langle mn\|ej\rangle
 - \sum_{nf}\big(\tfrac12 t_{jn}^{fb} + t_j^ft_n^b\big)\langle mn\|ef\rangle .
$$

$F_{ae}$ and $F_{mi}$ are the virtual and occupied Fock blocks dressed by
the amplitudes; $W_{mnij}$, $W_{abef}$ and $W_{mbej}$ are the hole–hole
ladder, particle–particle ladder and ring vertices of CCD, dressed the same
way. With them the residuals are $R = \Delta t + r$ and

$$
\begin{aligned}
r_i^a ={}& \sum_e t_i^eF_{ae} - \sum_m t_m^aF_{mi} + \sum_{me} t_{im}^{ae}F_{me}
 - \sum_{nf} t_n^f\langle na\|if\rangle
 - \tfrac12\sum_{mef} t_{im}^{ef}\langle ma\|ef\rangle
 - \tfrac12\sum_{men} t_{mn}^{ae}\langle nm\|ei\rangle , \\
r_{ij}^{ab} ={}& \langle ij\|ab\rangle
 + P(ab)\sum_e t_{ij}^{ae}\big(F_{be} - \tfrac12\sum_m t_m^bF_{me}\big)
 - P(ij)\sum_m t_{im}^{ab}\big(F_{mj} + \tfrac12\sum_e t_j^eF_{me}\big) \\
 &+ \tfrac12\sum_{mn}\tau_{mn}^{ab}W_{mnij} + \tfrac12\sum_{ef}\tau_{ij}^{ef}W_{abef}
 + P(ij)P(ab)\sum_{me}\big(t_{im}^{ae}W_{mbej} - t_i^et_m^a\langle mb\|ej\rangle\big) \\
 &+ P(ij)\sum_e t_i^e\langle ab\|ej\rangle - P(ab)\sum_m t_m^a\langle mb\|ij\rangle .
\end{aligned}
$$

Both ranks update as $t \leftarrow r/(-\Delta)$; DIIS acts on $t_2$ only.
The energy is

$$E_c = \tfrac14\sum_{ijab}\langle ij\|ab\rangle t_{ij}^{ab} + \tfrac12\sum_{ijab}\langle ij\|ab\rangle t_i^at_j^b .$$

These two terms are the only connected closed contributions: a third,
$\sum f_{ia}t_i^a$, vanishes because $f_{ia} = 0$. The energy is
$\tfrac14\sum\langle ij\|ab\rangle\tau_{ij}^{ab}$, since antisymmetry of the
integral turns the two $t_1t_1$ terms of $\tau$ into twice one of them.

**Two limits.**

- **CCD inside CCSD.** Set $t_1 = 0$. Every intermediate loses its singles
  terms, $\tau = \tilde\tau = t_2$, and the doubles residual is the CCD
  residual above, term for term.
- **Two electrons.** With $2n_o = 2$ spin orbitals occupied, $\hat T$ can
  excite at most both electrons, so
  $e^{\hat T}|\Phi_0\rangle = (1 + \hat T_1 + \hat T_2 + \tfrac12\hat T_1^2)|\Phi_0\rangle$
  and every higher power vanishes. This spans every determinant there is,
  and the projections onto all singles and doubles are the full CI
  eigenvalue equations. CCSD for a two-electron system is therefore full CI.
  CCD lacks $\hat T_1$ and is not exact here.

> **Verified equation {#eq:cc-ccsd-spin-orbital}** · the CCSD residuals and energy
>
> At random $t_1$ and $t_2$ on the same model, the Stanton–Gauss singles and
> doubles residuals of `ccsd` (intermediates $F_{ae}$, $F_{mi}$, $F_{me}$,
> $W_{mnij}$, $W_{abef}$, $W_{mbej}$, $\tau$, $\tilde\tau$) plus $\Delta t$
> equal the brute-force projections; the energy
> $\tfrac14\langle ij\|ab\rangle t_{ij}^{ab} + \tfrac12\langle ij\|ab\rangle t_i^at_j^b$
> equals the projected energy; and the doubles residual at $t_1 = 0$ is the
> CCD residual of (@eq:cc-ccd-residual).
>
> | | |
> |---|---|
> | Proof | `scripts/cc_ccsd_eom_brute_force_sympy.py`, checks `ccsd-residual`, `ccsd-energy`, `ccsd-ccd-limit` |
> | Negative control | the runner's mutations: $t_2$ instead of $\tfrac12t_2$ in $W_{mbej}$; the $t_1t_1$ energy weight $\tfrac14$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/ccsd.rs::ccsd` |
> | Code test | `ccsd_spin_orbital_matches_df_reference` in `crates/ferric-cc/tests/validation_cc.rs` |
>
> <!-- verified: id=cc.ccsd.spin-orbital; script=scripts/cc_ccsd_eom_brute_force_sympy.py; claim=ccsd-residual; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/ccsd.rs::ccsd; rust_test=ferric-cc/tests/validation_cc.rs::ccsd_spin_orbital_matches_df_reference; registry=ccsd -->

The proof builds $\hat H$, $\hat T$ and $e^{-\hat T}\hat He^{\hat T}$ as
Jordan–Wigner matrices on the model and compares $r + \Delta t$ and $E_c$
with the projections. It checks the equations the code evaluates; it does
not check that a converged solution exists. The termination of the BCH
series and the two-electron limit are not claims of the script.

## The algorithm {#ccd-ccsd-spin-orbital-algorithm}

Both solvers share one setup and one loop shape.

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF; $B^P_{ov}, B^P_{oo}, B^P_{vv}$ (section @sec:factorization-and-metric) | $N_{\rm aux}N^2n$ | `transform_3center_*`, `build_b` |
| 2 | spatial chemist blocks, then antisymmetrized spin-orbital blocks $\langle pq\|rs\rangle$ | $N_{\rm aux}n_v^4$ for VVVV | `asym_oovv`, `asym_same`, `asym_ovvo` (CCD); `asym_phys`, 11 blocks (CCSD) |
| 3 | MP2 guess $t_2 = \langle ij\|ab\rangle/(-\Delta)$, $t_1 = 0$ | | |
| 4 | residual $r$ by `einsum!` GEMMs; the pp ladder dominates | $(2n_o)^2(2n_v)^4$ per iteration | |
| 5 | $t \leftarrow r/(-\Delta)$; DIIS on $t_2$, error = increment | | `ferric_scf::diis::Diis` |
| 6 | stop when the energy changes by less than $10^{-10}$ (CCD) or $\min(\texttt{energy\_conv}, 10^{-9})$ (CCSD) | | |

Each residual term is one or two `einsum!` contractions, and the
antisymmetrizers act after the contraction. The ring term of CCD, with its
permutation pair applied once to the contracted block:

<!-- excerpt: ferric-cc/src/ccd.rs::ccd_spinorbital -->
```rust
let bjia: ArrayD<f64> = einsum!("kbcj,ikac->bjia", &ovvo_t, &t_t);
let x = permute_to_owned(bjia.view().permuted_axes(IxDyn(&[2, 1, 3, 0])));
r = r + p_ij_ab(&x);
```

| Symbol | Code |
|---|---|
| $\langle kb\|cj\rangle$ | `ovvo_t` |
| $t_{ik}^{ac}$ | `t_t` |
| $P(ij)P(ab)$ | `p_ij_ab` |
| $r_{ij}^{ab}$ | `r` |

- The contraction runs over $k, c$ and leaves the free indices in the order
  `bjia`, which is what the GEMM produces without a copy. One permutation
  then puts them in `ijab`.
- `p_ij_ab` applies all four signed swaps at once, so the ring term is
  contracted once, not four times.
- The quadratic terms are contracted in two steps through a small
  intermediate: for example $\sum_{kl}\langle kl\|cd\rangle t_{kl}^{bd}$ is
  a $(2n_v)\times(2n_v)$ matrix built first, then multiplied into
  $t_{ij}^{ac}$. With $n_v \ge n_o$, no term costs more than the pp ladder.

**Memory.** Every spin-orbital tensor is $2^4 = 16$ times its spatial
counterpart, and the $(2n_v)^4$ VVVV block is held in core (CCSD holds it
twice: the block and the per-iteration $W_{abef}$). Both drivers commit a
`MemoryPlan` before the first allocation and refuse a job that cannot fit,
naming the dominant term. This factor of 16 is why the production CCSD is
the spin-adapted one of section @sec:closed-shell-ccsd.

## Options {#ccd-ccsd-spin-orbital-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | frozen occupied spatial orbitals |
| `CcConfig::max_iter` | integer | 50 | iteration cap; non-convergence is an error (library field; the CLI uses the default) |
| `CcConfig::diis_subspace` | integer | 6 | DIIS history length (library field) |
| `[memory] budget_gb` | GB | auto | preflight of the plan |

## Numerical anchors {#ccd-ccsd-spin-orbital-numerical-anchors}

From `site/src/reference/validation.md` and the tests it names.

| Quantity | Systems | Error | Test |
|---|---|---|---|
| RI-CCD vs PySCF `CCD` on the same DF integrals | H2O, NH3 / cc-pVDZ, def2-SVP | $\le 1.9\times10^{-11}$ Ha | `crates/ferric-cc/tests/validation_cc.rs` |
| RI-CCSD, spin-orbital and spin-adapted, vs PySCF `CCSD` | H2O, NH3, CH4, HF / cc-pVDZ, def2-SVP and others | $\le 4.7\times10^{-11}$ Ha | `crates/ferric-cc/tests/validation_cc.rs` |

## Sources {#ccd-ccsd-spin-orbital-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "CCD", "CCSD in spin orbitals" | the working equations above |
| `scripts/cc_ccsd_eom_brute_force_sympy.py` | the brute-force check of both residuals |
| `site/src/methods/cc.md`, "CCSD and CCSD(T)" | the user guide |

**Known limits (2026-11-15).** The solvers of this section read restricted
orbitals only (`mos_r` and `eps_r` assert a restricted result); an
unrestricted reference goes to the spin-blocked solver of section @sec:uccsd or its
cross-check `ccsd_uhf_spin_orbital`. Canonical HF orbitals are assumed
($f_{ia} = 0$, diagonal $f_{oo}$ and $f_{vv}$). The spin-orbital CCSD is a
library cross-check with no CLI kind.
