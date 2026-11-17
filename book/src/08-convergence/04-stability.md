# Stability analysis {#stability}

<!-- status: ready; registry: rhf, uhf, rohf -->

A converged SCF is a stationary point of the energy, not necessarily a
minimum. Stretch the O–H bonds of water to 2.0 Å and the RHF solution still
converges, but a spin-broken UHF solution lies lower; on N2 at 1.60 Å plain DIIS converges to an RHF saddle
1.906e-2 Ha above the RHF minimum (section @sec:stability-numerical-anchors). Stability analysis
computes the lowest eigenvalues of the orbital Hessian at the converged
orbitals. A negative one means a lower solution exists along its
eigenvector: of the same kind (internal) or with broken spin symmetry
(external, RHF → UHF). The descent then follows it.

The plan: show why the lowest eigenvalue decides, derive the singlet and
triplet operators from the UHF Hessian, read the verdict the code returns,
and describe the descent.

| Symbol | Meaning | Code name |
|---|---|---|
| $\lambda_{\min}$ | lowest eigenvalue of the stability operator | `lowest_eigenvalue` |
| noise floor | $\max(10^{-6}, \text{residual})$ | `noise_floor` |
| $\lambda_T$ | lowest eigenvalue of the triplet (RHF → UHF) operator | the external result |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/stability.rs::{rhf_internal_stability,rhf_external_stability,uhf_internal_stability}`, `ferric-scf/src/uhf.rs::stability_descent`, `ferric-scf/src/rhf.rs::rhf_stability_descent` |
| CLI | `[scf] check_stability`, `stability_descent` |
| Python | `stability_descent=` on the SCF entry points |

## The stability operators {#stability-stability-operators}

### Why the lowest eigenvalue decides

At a converged solution the gradient is zero, so the quadratic model of
section @sec:second-order-orbital-gradient-hessian has no linear term. For RHF, rotate along a unit vector $v$:

$$ E(tv) = E(0) + \frac{t^2}{2}\cdot 4\,v\cdot(\mathbf Hv) + O(t^3) . $$

Take $v$ to be the eigenvector of the lowest eigenvalue $\lambda_{\min}$ of the
code's operator $\mathbf H$. Then $v\cdot(\mathbf Hv) = \lambda_{\min}$, and the
energy along $v$ is $E(0) + 2\lambda_{\min}t^2$. If $\lambda_{\min} > 0$, every
direction curves upward and the solution is a local minimum in the space
analysed. If $\lambda_{\min} < 0$, the energy falls along $v$ and the solution
is a saddle point. The eigenvalues carry the code's scale: the true RHF
curvature along a unit vector is $4\lambda$, the UHF one $2\lambda$.

The stability operators are the Newton products of section @sec:second-order at the
converged orbitals (`crates/ferric-scf/src/stability.rs`):

| Analysis | Operator | Function |
|---|---|---|
| RHF internal (real, singlet) | $\mathbf H$ of RHF: $(\varepsilon_a-\varepsilon_i)\delta + 4(ai\vert bj) - c_x[(ab\vert ij)+(aj\vert bi)]$ | `rhf_internal_stability` |
| RHF external (RHF → UHF, triplet) | $(\varepsilon_a-\varepsilon_i)\delta - c_x[(ab\vert ij)+(aj\vert bi)]$ | `rhf_external_stability` |
| UHF internal | $\mathbf H$ of UHF | `uhf_internal_stability` |

At converged canonical orbitals the diagonal-gap form of the one-electron
term is exact, so the operators are the full Hessians in their spaces.

### The singlet and triplet channels

The external operator comes from the UHF Hessian evaluated at the RHF
solution. There both spins have the same orbitals, so
$f^\alpha = f^\beta = f$ and the two-electron integrals are the same for both
spins. Take the UHF product of section @sec:second-order-orbital-gradient-hessian and split the rotation into two
directions.

In the direction $\kappa^\alpha = \kappa^\beta = \kappa$ the Coulomb sum over
$\tau$ gives $2(ai|bj)\kappa_{bj} + 2(ai|bj)\kappa_{bj} = 4(ai|bj)\kappa_{bj}$, and
the exchange term is unchanged. The result is the RHF singlet operator, the
same in both spin components.

In the direction $\kappa^\alpha = -\kappa^\beta = \kappa$ the two Coulomb terms
cancel, $2(a_\alpha i_\alpha|b_\alpha j_\alpha)\kappa_{bj} - 2(a_\alpha i_\alpha|b_\beta j_\beta)\kappa_{bj} = 0$:
the density changes of the two spins are equal and opposite, so the total
density does not move and $J$ does not respond. The exchange term stays,
because $K$ acts within one spin. What is left is the triplet operator of
the table.

A vector $(w, w)$ is mapped to the form $(u, u)$ and a vector $(w, -w)$ to the
form $(u, -u)$, so the two channels do not couple. Now take a unit
eigenvector $v$ of the triplet operator with eigenvalue $\lambda_T$ and rotate
the UHF orbitals along $(v, -v)$. The UHF second derivative is
$2\sum_\sigma\kappa^\sigma\cdot(\mathbf H\kappa)^\sigma$:

$$ \frac{d^2E}{dt^2} = 2\big(v\cdot\lambda_Tv + (-v)\cdot(-\lambda_Tv)\big) = 4\lambda_T . $$

So $\lambda_T < 0$ means a UHF solution lower than the RHF one.

> **Verified equation {#eq:scf-stability-channels}** · the singlet and triplet channels
>
> At an RHF point the UHF Hessian product in the direction
> $\kappa^\alpha = \kappa^\beta$ is the RHF singlet operator and in the
> direction $\kappa^\alpha = -\kappa^\beta$ the triplet operator; the UHF
> energy along $(v, -v)$ has curvature $4\lambda_T$.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_orbital_hessian_sympy.py`, claims S1 and S2 |
> | Negative control | the triplet exchange term set to zero must fail S1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/stability.rs::{rhf_external_stability,uhf_internal_stability}` |
>
> <!-- verified: id=scf.stability-channels; script=scripts/scf_orbital_hessian_sympy.py; claim=S1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/stability.rs::{rhf_external_stability,uhf_internal_stability}; registry=uhf -->

### The triplet product in code

`ferric-scf/src/rhf_newton.rs::triplet_hessian_matvec_jk`, with the comments
replaced by the terms:

```rust
let dd_ao = 2.0 * &ao_from_ov(inp.c, k, no, n);        // same delta D as the singlet
let df: Array2<f64> = match inp.rsh {
    Some(rsh) => -0.5 * &rsh.exchange_response(&dd_ao)?, // range-separated -K/2
    None => -0.5 * inp.k_mix_sr * &triplet_exchange(ctx, inp, k, pool, jk, &dd_ao)?, // -c_x K/2
};
let df_mo = inp.c.t().dot(&df).dot(inp.c);
let mut h = occ_virt_block(&df_mo, no, n);               // -c_x [(ab|ij) + (aj|bi)] k
let f_diag: Vec<f64> = (0..n).map(|i| inp.f_mo[(i, i)]).collect();
for (ir, a) in (no..n).enumerate() {
    for (ic, i) in (0..no).enumerate() {
        h[(ir, ic)] += (f_diag[a] - f_diag[i]) * k[(ir, ic)]; // (e_a - e_i) k_ai
    }
}
```

Two points about it:

- **The triplet character is in the response, not in $\delta D$.** The
  density perturbation is the singlet one; what changes is that $\delta J$
  is not built. Building a different $\delta D$ here would give an operator
  that looks plausible and is wrong.
- **No $f_{xc}$ term.** The triplet XC kernel is $f_{\alpha\alpha} - f_{\alpha\beta}$,
  which the closed-shell response closure does not provide (it returns the
  singlet $f_{\alpha\alpha} + f_{\alpha\beta}$). `rhf_external_stability`
  therefore refuses a Kohn–Sham reference before any product runs. The
  hybrid exact-exchange fraction `k_mix_sr` is honoured.

### Finding the lowest eigenvalues

The lowest eigenvalues come from a block Davidson (`davidson_lowest`) with
the `StabilityConfig` defaults: 8 seed vectors, blocks of 6, a subspace of
at most 40, residual $10^{-6}$, and the diagonal preconditioner
$\varepsilon_a - \varepsilon_i$. Why a block? The orbital Hessian is block
diagonal in the molecule's symmetry. A single-root Davidson seeded inside
one symmetry block can converge, with a small residual, to the lowest root
of that block and miss a lower root elsewhere. Tracking several roots and
requiring the whole block to converge leaves at least one residual large
when a root lies below the block.

## The verdict {#stability-verdict}

The verdict is four-way and is what callers branch on.
`ferric-scf/src/stability.rs::verdict`, shown whole:

```rust
if !self.converged {
    StabilityVerdict::Indeterminate          // Davidson did not reach its residual
} else if self.is_marginal() {
    StabilityVerdict::Marginal               // |lambda_min| <= noise floor
} else if self.lowest_eigenvalue > 0.0 {
    StabilityVerdict::Stable                 // a local minimum
} else {
    StabilityVerdict::Unstable               // lambda_min < -noise floor: a saddle
}
```

| Code | Meaning |
|---|---|
| `converged` | the Davidson solve reached its residual |
| `is_marginal()` | $\lvert\lambda_{\min}\rvert \le$ noise floor: neither a proven minimum nor a proven saddle |
| `lowest_eigenvalue > 0.0` | a local minimum in the directions analysed |

The noise floor is $\max(10^{-6}, \text{residual})$: a value smaller in
magnitude than the residual of the solve that produced it is not trusted
for its sign. The order of the tests matters. An unconverged solve proves nothing,
so it is checked first; a marginal value is reported as such before its
sign is read.

The eigenvalues carry the code's Hessian scale (a quarter of the RHF energy
curvature, half of the UHF one), so a threshold on $\lambda$ is not a
threshold in Hartree per radian squared.

## The descent {#stability-descent}

| Step | What |
|---|---|
| 1 | the lowest eigenvector $v$ |
| 2 | orbitals rotated by $\epsilon v$, $\epsilon \in \{0.4, 0.8, 1.2\}$ |
| 3 | each start reconverged; the lowest energy below the incumbent kept |
| 4 | repeated up to three rounds, with a warning if still descending |

The quadratic model says which way is downhill. It does not say how far the
next minimum is, so the descent tries three step lengths
(`DESCENT_STEPS`) and lets a full SCF from each find its own basin. The
rotation is the Cayley map of section @sec:second-order-rotation. A candidate is kept only if its
SCF converged and its energy is below both the incumbent and every earlier
candidate of the round (`accepts_candidate`). If no step goes lower, the
saddle is returned with a message that it is not established as the lowest
state. Each round costs one Davidson solve and up to three SCFs, and there
are at most three rounds (`MAX_DESCENT_ROUNDS`).

The RHF descent is skipped, with a printed reason, when the SCF did not
converge (the Hessian would be evaluated at a non-stationary point), when
Fermi smearing is on (the integer-occupation Hessian does not describe
fractional occupations), and when no stability verdict was computed.

Real-to-complex, UHF-to-GHF and ROHF stability are not implemented.

## Numerical anchors {#stability-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| lowest Hessian eigenvalues, UHF/UKS internal, RHF internal and external: N2⁺, OH, water at 0.9572 and 2.0 Å, NH2 / 6-31G | PySCF dense Hessians and `stability()` | ≤ 4.5e-10 Ha | `crates/ferric-scf/tests/validation_scf_stability.rs` |
| triplet operator against the UHF Hessian at the RHF point | an independent construction | ≤ 2.3e-13 Ha | `crates/ferric-scf/tests/scf_stability_external.rs` |
| water at 2.0 Å: internal and external $\lambda_{\min}$ | PySCF | internal STABLE (+1.97e-2), external UNSTABLE (−3.07e-1), verdicts match | `crates/ferric-scf/tests/validation_scf_stability.rs` |
| convergence ladder with descent: N2 at 1.60 Å, CuCN, Cr(CO)6 / def2-SVP | PySCF `.newton()` from four guesses | energy ≤ 8.8e-10 Ha; plain DIIS lands on the N2 saddle 1.906e-2 Ha higher | `crates/ferric-scf/tests/validation_scf_ladder.rs` |

<!-- figures: site/src/reference/validation.md "SCF stability: lowest orbital-Hessian eigenvalues and verdict", "SCF convergence ladder with RHF internal stability descent" -->

Water at 2.0 Å shows the two channels apart: the RHF solution is a minimum
among RHF solutions ($\lambda_{\min} = +1.97\times10^{-2}$) and a saddle once
the spins may differ ($\lambda_T = -3.07\times10^{-1}$). OH is an exact zero
mode; ferric reports it MARGINAL where PySCF reports stable, which is the
intended reading of the noise floor. Kohn–Sham references are refused for
the external channel, which would need the triplet XC kernel.

## Doc fixes needed {#stability-doc-fixes-needed}

| Text | Code |
|---|---|
| eigenvalues described as Hartree per radian squared | they are the quarter-scale (RHF) or half-scale (UHF) Newton operator's |
| the `check_stability` doc | name both the internal and the external result it fills |

## Sources

| Source | Used for |
|---|---|
| `scripts/scf_orbital_hessian_sympy.py` | verified equation @eq:scf-stability-channels |
| `site/src/reference/validation.md` | the anchors |
