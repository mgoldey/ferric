# RHF, UHF and ROHF energy functionals {#energy-functionals}

<!-- status: ready; registry: rhf, uhf, rohf;
     theory: crates/ferric-scf/theory.md#"Hartree-Fock energy functionals (RHF, UHF, ROHF)" -->

Hartree–Fock replaces the many-electron problem by one electron at a time in
the averaged field of the rest. Each electron is repelled by the total charge
cloud (the Coulomb term), and the exchange term removes the part of that
repulsion an electron would feel from itself and lowers the repulsion
between electrons of equal spin. The energy is then a function of the
density alone, and the best orbitals are the ones that make it stationary.
RHF puts two electrons of opposite spin in each orbital, UHF gives the two
spins their own orbitals, and ROHF keeps one set of orbitals but lets the top
few hold a single $\alpha$ electron. Hartree–Fock is the reference for every
correlated method in this book, and its Fock matrix is the object every
builder in this chapter computes.

This section writes the three energy functionals, derives the Fock matrices
as their derivatives, and states the stationarity conditions the convergence
tests of chapter @sec:convergence check.

| Index | Range | Code name |
|---|---|---|
| $\mu, \nu, \lambda, \sigma$ | AO functions | `0..nbas` |
| $i, j$ | occupied orbitals (closed shell) | `0..nocc` |
| $a, b$ | virtual orbitals | `nocc..nmo` |
| $c$, $o$, $v$ | ROHF closed, open and virtual blocks | `0..n_beta`, `n_beta..n_alpha`, `n_alpha..nmo` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/rhf.rs::solve_rhf`, `ferric-scf/src/uhf.rs::solve_uhf`, `ferric-scf/src/rohf.rs::solve_rohf` |
| CLI | `method.kind = "rhf"`, `"uhf"`, `"rohf"` (`examples/water-rhf.toml`, `examples/h_uhf.toml`) |
| Python | `run_rhf`, `run_uhf`, `run_rohf` (`crates/ferric-python/ferric.pyi`) |

Spin and charge belong to the molecule, not to the solver call. A
multiplicity above one under `kind = "rhf"` is refused.

## The functionals {#energy-functionals-functionals}

### Conventions

The AO basis $\{\chi_\mu\}$ is real, with overlap $S$ and core Hamiltonian
$h = T + V_{ne}$ (plus $V_{\rm ECP}$ and external terms when present). The
two-electron integrals $(\mu\nu|\lambda\sigma)$ are real and have 8-fold
symmetry. The orbital coefficients $C$ are $S$-orthonormal,
$C^{\mathsf T}SC = 1$. Indices $i, j$ run over occupied orbitals, $a, b$ over
virtual ones, $p, q$ over all.

The Coulomb and exchange builders (`JBuilder` and `KBuilder` in
`crates/ferric-scf/src/fock.rs`) compute two linear maps of a density:

$$ J(D)_{\mu\nu} = \sum_{\lambda\sigma}(\mu\nu|\lambda\sigma)D_{\lambda\sigma}, \qquad K(D)_{\mu\nu} = \sum_{\lambda\sigma}(\mu\lambda|\nu\sigma)D_{\lambda\sigma} . $$

$J(D)_{\mu\nu}$ is the electrostatic potential of the charge cloud $D$,
taken between $\chi_\mu$ and $\chi_\nu$. $K$ swaps one index of the pair
across the two electrons; it has no classical analogue. Both maps are linear.
By the 8-fold symmetry each is also symmetric under the trace pairing:

$$ \operatorname{tr}[A\,J(B)] = \operatorname{tr}[B\,J(A)], \qquad \operatorname{tr}[A\,K(B)] = \operatorname{tr}[B\,K(A)] . $$

That identity is what lets us differentiate the quadratic terms below.

### RHF

The code's closed-shell density carries the occupation 2,
$D = 2C_oC_o^{\mathsf T}$. The energy is

$$ E[D] = \operatorname{tr}(Dh) + \tfrac12\operatorname{tr}[D\,J(D)] - \tfrac14\operatorname{tr}[D\,K(D)] + E_{nn} . $$

The $\tfrac12$ on the Coulomb term stops each pair of electrons from being
counted twice. The exchange weight is $\tfrac14$ because exchange acts only
between electrons of equal spin, and each spin carries half of $D$.

Differentiate. Vary $D \to D + \delta$ and keep first order. The one-electron
term gives $\operatorname{tr}(\delta h)$. Expand the Coulomb term by
linearity:

$$ \tfrac12\operatorname{tr}[(D+\delta)J(D+\delta)] - \tfrac12\operatorname{tr}[D\,J(D)] = \tfrac12\operatorname{tr}[\delta\,J(D)] + \tfrac12\operatorname{tr}[D\,J(\delta)] + O(\delta^2) . $$

Use the trace symmetry on the second term, $\operatorname{tr}[D\,J(\delta)] = \operatorname{tr}[\delta\,J(D)]$,
so the change is $\operatorname{tr}[\delta\,J(D)]$. The exchange term goes
the same way and gives $-\tfrac12\operatorname{tr}[\delta\,K(D)]$. Collect:
$\delta E = \operatorname{tr}[\delta\,(h + J(D) - \tfrac12K(D))]$, and the
Fock matrix is the bracket:

$$ F = \frac{\partial E}{\partial D} = h + J(D) - \tfrac12K(D) . $$

Substitute $F$ back into $E$. Half of $\operatorname{tr}(DF)$ is
$\tfrac12\operatorname{tr}(Dh) + \tfrac12\operatorname{tr}[DJ] - \tfrac14\operatorname{tr}[DK]$,
which has the two-electron terms of $E$ and half its one-electron term. Add
the other half, $\tfrac12\operatorname{tr}(Dh)$:

$$ E = \tfrac12\operatorname{tr}[D(h + F)] + E_{nn} . $$

The loop evaluates this form (`e_elec_no_xc` in `solve_rhf`). It
needs the Fock matrix, which the loop has already built, and no separate
contraction of $J$ or $K$ with $D$.

Check one doubly occupied orbital $\phi$. Then $\operatorname{tr}[DJ(D)] = 4(\phi\phi|\phi\phi)$
and $\operatorname{tr}[DK(D)] = 4(\phi\phi|\phi\phi)$, so
$E - E_{nn} = 2h_{\phi\phi} + 2(\phi\phi|\phi\phi) - (\phi\phi|\phi\phi) = 2h_{\phi\phi} + (\phi\phi|\phi\phi)$:
two electrons, each with its one-electron energy, and one repulsion between
them. The exchange term has removed the half of the Coulomb term that counted
each electron against itself.

Now make $E$ stationary under a change of the orbitals, $C \to C + \delta C$,
with the constraint $C^{\mathsf T}SC = 1$. With $D = 2C_oC_o^{\mathsf T}$,
$\delta D = 2(\delta C_o C_o^{\mathsf T} + C_o\delta C_o^{\mathsf T})$ and
$\delta E = \operatorname{tr}(F\,\delta D) = 4\operatorname{tr}(\delta C_o^{\mathsf T}FC_o)$,
since $F$ is symmetric. The constraint enters with a symmetric matrix of
multipliers $\varepsilon$, whose variation gives $4\operatorname{tr}(\delta C_o^{\mathsf T}SC_o\varepsilon)$.
Setting the sum to zero for every $\delta C_o$ gives $FC_o = SC_o\varepsilon$.
A rotation among the occupied orbitals does not change $D$, so it can be
chosen to make $\varepsilon$ diagonal. Those are the Roothaan equations,

$$ FC = SC\varepsilon . $$

An equivalent form uses only $D$. Multiply $FC_o = SC_o\varepsilon$ by
$2C_o^{\mathsf T}S$ on the right: $FDS = 2SC_o\varepsilon C_o^{\mathsf T}S$.
Transpose the Roothaan equation to $C_o^{\mathsf T}F = \varepsilon C_o^{\mathsf T}S$
and multiply by $2SC_o$ on the left: $SDF = 2SC_o\varepsilon C_o^{\mathsf T}S$.
The right-hand sides are equal, so at convergence

$$ FDS - SDF = 0 , $$

which is the DIIS error vector of section @sec:diis. In the MO basis the energy is
$2\sum_i h_{ii} + \sum_{ij}[2(ii|jj) - (ij|ij)] + E_{nn}$, the closed-shell
Slater–Condon expression.

### UHF

UHF gives each spin its own orbitals. The per-spin densities are
$D_\sigma = C_{\sigma o}C_{\sigma o}^{\mathsf T}$ (occupation 1) and the total
is $D = D_\alpha + D_\beta$. Coulomb repulsion sees the total charge;
exchange acts within one spin:

$$ E = \operatorname{tr}(Dh) + \tfrac12\operatorname{tr}[D\,J(D)] - \tfrac12\sum_\sigma\operatorname{tr}[D_\sigma K(D_\sigma)] + E_{nn} . $$

Differentiate with respect to $D_\alpha$ as above. $D$ depends on $D_\alpha$
with unit weight, so the Coulomb term gives $J(D)$; the exchange term has
only its $\alpha$ part, which gives $K(D_\alpha)$. The same holds for
$\beta$:

$$ F_\sigma = \frac{\partial E}{\partial D_\sigma} = h + J(D) - K(D_\sigma), \qquad E = \tfrac12\sum_\sigma\operatorname{tr}[D_\sigma(h + F_\sigma)] + E_{nn} . $$

Stationarity per spin gives the Pople–Nesbet equations
$F_\sigma C_\sigma = SC_\sigma\varepsilon_\sigma$.

Two limits check the expression. At $D_\alpha = D_\beta = D/2$ the exchange
sum is $2\cdot\tfrac12\operatorname{tr}[\tfrac{D}{2}K(\tfrac{D}{2})] = \tfrac14\operatorname{tr}[DK(D)]$,
so the UHF energy is the RHF energy of $D$, and $F_\alpha = F_\beta = h + J(D) - \tfrac12K(D) = F$.
And for a single electron in orbital $\phi$, $D_\alpha = \phi\phi^{\mathsf T}$
and $D_\beta = 0$, both $\operatorname{tr}[D_\alpha J(D_\alpha)]$ and
$\operatorname{tr}[D_\alpha K(D_\alpha)]$ equal $(\phi\phi|\phi\phi)$, and
$E - E_{nn} = h_{\phi\phi}$. A hydrogen atom has no self-repulsion in
Hartree–Fock.

> **Verified equation {#eq:scf-fock-is-de-dd}** · the Fock matrix is the density derivative of the energy
>
> $F = \partial E/\partial D$ for the RHF functional and
> $F^\sigma = \partial E/\partial D^\sigma$ for both UHF spins, symbolically on
> a two-function model and by central differences on a random seven-function
> model.
>
> | | |
> |---|---|
> | Proof | `scripts/hf_energy_sympy.py`, claims S1, S2 and N1 |
> | Negative control | the RHF exchange weight $\tfrac12 \to 1$ must fail N2 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/rhf.rs::solve_rhf`, `ferric-scf/src/uhf.rs::solve_uhf` |
>
> <!-- verified: id=scf.fock-is-de-dd; script=scripts/hf_energy_sympy.py; claim=S1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/rhf.rs::solve_rhf, ferric-scf/src/uhf.rs::solve_uhf; registry=rhf -->

### The RHF Fock matrix in the code

The closed-shell loop keeps $J$ and $K$ in buffers and assembles $F$ in
place. Here is the Hartree–Fock arm of the assembly in `solve_rhf`, with the
range-separated branch and the timing calls removed:

```rust
let f = &mut st.f;
f.assign(&inp.env.h);                          // F = h
*f += &self.j_buf;                             //   + J(D)
// ... range-separated branch removed ...
f.scaled_add(-0.5 * k_mix.sr, &self.k_buf);    //   - 1/2 c_x K(D), c_x = 1 for HF
```

and the energy, a few lines later:

```rust
let e_elec_no_xc: f64 = 0.5 * (d * &(&env.h + &*f)).sum();   // 1/2 tr[D(h + F)]
```

- `(d * &(h + f)).sum()` is the elementwise product summed, which is
  $\sum_{\mu\nu}D_{\mu\nu}(h+F)_{\mu\nu} = \operatorname{tr}[D(h+F)]$ for
  symmetric matrices.
- `k_mix.sr` is the exact-exchange fraction. It is 1 for Hartree–Fock and
  $c_x$ for a hybrid functional (chapter @sec:kohn-sham); a pure functional skips the
  line.
- The UHF and ROHF loops evaluate the same trace per spin,
  $\tfrac12\sum_\sigma\operatorname{tr}[D_\sigma(h + F_\sigma)]$:
  `0.5 * ((&(h + &*f_a) * d_a).sum() + (&(h + &*f_b) * d_b).sum())`.

### ROHF

ROHF uses one orbital set $C$ whose columns split into closed ($c$), open
($o$, $\alpha$ only) and virtual ($v$). The occupation vectors are
$n_\alpha = (1_c, 1_o, 0_v)$ and $n_\beta = (1_c, 0_o, 0_v)$, and the
densities are

$$ D_\beta = C_cC_c^{\mathsf T}, \qquad D_\alpha = D_\beta + C_oC_o^{\mathsf T}, \qquad D_\sigma = C\operatorname{diag}(n_\sigma)C^{\mathsf T} . $$

The energy is the UHF functional at these densities, with the same
$F_\sigma = h + J(D_\alpha + D_\beta) - K(D_\sigma)$. `solve_rohf` evaluates
exactly the UHF trace formula. In MO integrals it is the high-spin energy

$$ E = 2\sum_c h_{cc} + \sum_t h_{tt} + \sum_{cc'}[2J_{cc'} - K_{cc'}] + \sum_{ct}[2J_{ct} - K_{ct}] + \tfrac12\sum_{tu}[J_{tu} - K_{tu}] + E_{nn}, $$

with $t, u$ open, $J_{pq} = (pp|qq)$ and $K_{pq} = (pq|pq)$.

What is new in ROHF is the stationarity condition. UHF has two Fock matrices
and two orbital sets, one equation each. ROHF has two Fock matrices and one
orbital set, so neither $F_\alpha$ nor $F_\beta$ alone can be diagonalized.
The condition comes from the orbital gradient.

**Orbital gradient.** Rotate the orbitals, $C \to Ce^{\kappa}$ with $\kappa$
real and antisymmetric, and write $f_\sigma = C^{\mathsf T}F_\sigma C$ for
the Fock matrices in the MO basis. In the MO basis
$D_\sigma = \operatorname{diag}(n_\sigma)$, and to first order
$\delta D_\sigma = \kappa\,n_\sigma - n_\sigma\kappa$, that is,
$(\delta D_\sigma)_{pq} = (n_{\sigma q} - n_{\sigma p})\kappa_{pq}$. The energy
changes by $\delta E = \sum_\sigma\operatorname{tr}(f_\sigma\delta D_\sigma)$.

A rotation inside $c$, $o$ or $v$ mixes orbitals of equal occupation, so
$n_{\sigma p} = n_{\sigma q}$ and neither density changes. Take a rotation
between $v$ and $c$, with $\kappa_{vc} = -\kappa_{cv}$. For $\alpha$,
$(\delta D_\alpha)_{vc} = (1 - 0)\kappa_{vc}$ and
$(\delta D_\alpha)_{cv} = (0 - 1)\kappa_{cv} = \kappa_{vc}$, so the trace
picks up $2(f_\alpha)_{vc}\kappa_{vc}$. $\beta$ gives the same with $f_\beta$.
For a rotation between $v$ and $o$, only $\alpha$ has different occupations,
and for $o$ and $c$ only $\beta$ does. The three blocks (row index the
orbital rotated into) are

$$ \frac{\partial E}{\partial\kappa_{pq}} = 2g_{pq}, \qquad g_{vc} = (f_\alpha + f_\beta)_{vc}, \quad g_{vo} = (f_\alpha)_{vo}, \quad g_{oc} = (f_\beta)_{oc} . $$

ROHF convergence means $g = 0$. Read the blocks physically. A closed-shell
electron moving to a virtual orbital takes one electron of each spin with
it, so both Fock matrices count. An open-shell electron is $\alpha$, so
promoting it to a virtual involves $F_\alpha$ only. Filling the open orbital
from a closed one moves a $\beta$ electron, so $F_\beta$ only.
`ferric-scf/src/rohf.rs::rohf_gradient_mo` builds these blocks (its three
loops are the three equations), and the DIIS error is
$SC(g - g^{\mathsf T})C^{\mathsf T}S$.

**Effective Fock matrix.** To use an ordinary diagonalization, ROHF builds
one matrix $R$ whose eigenvectors satisfy $g = 0$.
`ferric-scf/src/rohf.rs::roothaan_fock_inner` is PySCF's `get_roothaan_fock`
with the Guest–Saunders choice. With $F_c = (F_\alpha + F_\beta)/2$ and the
projectors $P_c = D_\beta S$, $P_o = (D_\alpha - D_\beta)S$,
$P_v = 1 - D_\alpha S$,

$$ R = M + M^{\mathsf T}, \qquad M = \tfrac12P_c^{\mathsf T}F_cP_c + \tfrac12P_o^{\mathsf T}F_cP_o + \tfrac12P_v^{\mathsf T}F_cP_v + P_o^{\mathsf T}F_\beta P_c + P_o^{\mathsf T}F_\alpha P_v + P_v^{\mathsf T}F_cP_c . $$

Why projectors? $P_cC = C_cC_c^{\mathsf T}SC$ keeps the closed columns of $C$
and zeroes the others, because $C^{\mathsf T}SC = 1$. So
$C^{\mathsf T}P_c^{\mathsf T}XP_cC$ is the $cc$ block of $C^{\mathsf T}XC$
with zeros elsewhere, and likewise for $o$ and $v$. Each term of $M$ places
one Fock matrix in one block, and $C^{\mathsf T}RC$ has the block table

| | $c$ | $o$ | $v$ |
|---|---|---|---|
| $c$ | $F_c$ | $F_\beta$ | $F_c$ |
| $o$ | $F_\beta$ | $F_c$ | $F_\alpha$ |
| $v$ | $F_c$ | $F_\alpha$ | $F_c$ |

Compare the off-diagonal blocks with the gradient: they are $g/2$ ($vc$) and
$g$ ($vo$, $oc$). So $RC = SC\varepsilon$ at self-consistency is the same
condition as $g = 0$. The diagonal blocks are a canonicalization choice: they
fix the orbital energies (eigenvalues of $R$) and the orbitals inside each
block, but not the energy or the densities. There is no user knob for the
coupling.

<!-- excerpt: ferric-scf/src/rohf.rs::roothaan_fock_inner -->
```rust
let mut f = 0.5 * p_c_t.dot(&f_c).dot(&p_c);
    f = &f + &(0.5 * p_o_t.dot(&f_c).dot(&p_o));
    f = &f + &(0.5 * p_v_t.dot(&f_c).dot(&p_v));
    f = &f + &p_o_t.dot(f_b).dot(&p_c);
    f = &f + &p_o_t.dot(f_a).dot(&p_v);
    f = &f + &p_v_t.dot(&f_c).dot(&p_c);
    let f_sym = &f + &f.t();
```

| Code | Equation |
|---|---|
| `p_c`, `p_o`, `p_v` | $P_c = D_\beta S$, $P_o = (D_\alpha - D_\beta)S$, $P_v = 1 - D_\alpha S$ |
| `f_c` | $F_c = (F_\alpha + F_\beta)/2$ |
| the six lines | the six terms of $M$, in order |
| `f_sym` | $R = M + M^{\mathsf T}$ |

The $\tfrac12$ on the three diagonal-block terms compensates for the
symmetrization: a diagonal block appears in both $M$ and $M^{\mathsf T}$,
while an off-diagonal block of $M$ lands in the opposite triangle of
$M^{\mathsf T}$.

> **Verified equation {#eq:scf-rohf-gradient}** · the ROHF orbital gradient and the effective Fock matrix
>
> $\partial E/\partial\kappa_{pq} = 2g_{pq}$ with the three blocks
> $g_{vc} = (f^\alpha + f^\beta)_{vc}$, $g_{vo} = f^\alpha_{vo}$,
> $g_{oc} = f^\beta_{oc}$, and the effective Fock matrix $R$ has the block table
> of the included derivation, so $RC = SC\varepsilon$ holds exactly when $g = 0$.
>
> | | |
> |---|---|
> | Proof | `scripts/hf_energy_sympy.py`, claims N3, N4 and N5 |
> | Negative control | the open–closed block $F^\beta \to F^\alpha$ must fail N3 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/rohf.rs::{rohf_gradient_mo,roothaan_fock_inner}` |
>
> <!-- verified: id=scf.rohf-gradient; script=scripts/hf_energy_sympy.py; claim=N4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/rohf.rs::{rohf_gradient_mo,roothaan_fock_inner}; registry=rohf -->

`scripts/hf_energy_sympy.py` proves $F = \partial E/\partial D$ (RHF, both UHF
spins) symbolically and by central differences, the closed-shell limit, the
$\Gamma$ forms used by the gradient, the block table, $\partial E/\partial\kappa = 2g$
by differences of $E(Ce^\kappa)$ at a random orbital set, the stationarity of
the converged ROHF orbitals under $c$-$o$, $c$-$v$ and $o$-$v$ rotations, the
high-spin MO energy, and $E_{\rm UHF} \le E_{\rm ROHF}$. The last inequality
holds because ROHF is UHF with a constraint on the orbitals.

| Equation | Code |
|---|---|
| $J$, $K$ conventions | `crates/ferric-scf/src/fock.rs` (`JBuilder`, `KBuilder`) |
| RHF $F$, $E$ | `solve_rhf` (`f.assign(&h)`, `e_elec_no_xc`) |
| UHF $F_\sigma$, $E$ | `solve_uhf` |
| ROHF $E$, $R$, $g$ | `solve_rohf`, `roothaan_fock_inner`, `rohf_gradient_mo` |

## One iteration {#energy-functionals-iteration}

Each solver repeats the same five steps. The rest of this chapter is about
step 2, and chapter @sec:convergence about steps 3 and 4.

| Step | Quantity | Section |
|---|---|---|
| 1 | density from the current orbitals: $D = 2C_oC_o^{\mathsf T}$, or $D^\alpha, D^\beta$ | 7.1 |
| 2 | $J[D]$ and $K[D]$ (or $K[D^\sigma]$), then $F = h + J - \tfrac12 K$, or $F^\sigma = h + J[D^\alpha + D^\beta] - K[D^\sigma]$ | 7.3 to 7.5 |
| 3 | the energy $E = \tfrac12\operatorname{tr}[D(h+F)] + E_{nn}$, and the error vector | 8.1 |
| 4 | an extrapolated or shifted Fock matrix | 8.1, 8.2 |
| 5 | $\tilde F = X^{\mathsf T} F X$, its eigenvectors, $C = X\tilde C$ | 7.2 |

ROHF diagonalizes the effective Fock matrix $R$ in step 5 and uses the UHF
energy of its two densities in step 3. The open-shell solvers share the
J/K route decision with the closed-shell solver
(`ferric-scf/src/fock_assembly.rs::resolve_jk_route`).

Step 2 dominates the cost. $J$ and $K$ each contract $N^4$ integrals with
the density, against $N^3$ for the diagonalization of step 5. Sections @sec:direct-and-incremental to
7.5 are different ways of doing that contraction.

## Options that change the equation {#energy-functionals-options-change-equation}

| Option | Effect | Where |
|---|---|---|
| `[dft] functional` | replaces $-\tfrac12 K$ by $-\tfrac12 c_x K + V_{xc}$ (RKS, UKS, ROKS) | chapter @sec:kohn-sham |
| `[external_potential]`, `[qmmm]` | adds the point-charge and field terms to $h$ once, before the loop | chapter @sec:implicit-solvation |
| `[cosmo]`, `[pcm]` | adds a reaction-field term to $F$ every iteration | chapter @sec:embedding |
| `fock_hook` (Python) | adds $V = \partial E_h/\partial D$ to $F$ every iteration | section @sec:fock-hooks |
| `relativistic = "x2c1e"` | replaces $T + V$ in $h$ by the spin-free X2C-1e Hamiltonian | the user guide |

## Numerical anchors {#energy-functionals-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| RHF energy, 45 orbital-set cases on small molecules | PySCF 2.13.1, each basis taken from the Basis Set Exchange by name | 1.8e-12 Ha | `crates/ferric-mp2/tests/validation_basis_coverage.rs` |
| UHF and ROHF energies, HO2, NO2, CH2 (triplet), allyl / 6-31G, def2-SVP | PySCF `UHF`/`ROHF` with `stability()` | 4.0e-12 Ha | `crates/ferric-scf/tests/validation_open_shell_scf.rs` |

<!-- figures: site/src/reference/validation.md "Bundled basis sets: every registered orbital, RI and JK set loads to the published contractions", "UHF and ROHF energies, stability-checked"; crates/ferric-core/src/certificate/anchors.rs methods "RHF", "UHF", "ROHF" -->

These are the registered energy anchors of the three methods
(`crates/ferric-core/src/certificate/anchors.rs`). A run's certificate
carries them as its measured error.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Hartree-Fock energy functionals (RHF, UHF, ROHF)" | the functionals, Fock matrices, ROHF gradient and effective Fock |
| `scripts/hf_energy_sympy.py` | verified equations @eq:scf-fock-is-de-dd and @eq:scf-rohf-gradient |
| `site/src/reference/validation.md` | the anchors |

The intermediate steps of @sec:energy-functionals-functionals that are not in the theory section (the
expansion of the Coulomb term, the one-orbital and one-electron limits, the
$FDS = SDF$ step, the $\delta D_\sigma$ element bookkeeping and the
projector argument) were checked numerically on a random seven-function
model for this section.
