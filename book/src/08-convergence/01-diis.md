# DIIS and its variants {#diis}

<!-- status: ready; registry: rhf, uhf, rohf -->

Plain Roothaan iteration builds $F$ from the current density, diagonalizes
it, and occupies the lowest orbitals. The new orbitals are the best ones for
the old field, but their density makes a different field, and on many
molecules the next step overshoots: the density swings back and forth
between two shapes or runs away (section @sec:level-shift-mom-smearing-level-shift-damps-step works this out for one
rotation). DIIS keeps the last few Fock matrices together with a measure of
how far each is from self-consistency, and takes the combination whose
error is smallest. Every SCF solver in ferric extrapolates this way.

The plan: show that the error ferric stores is the orbital gradient, pose
the extrapolation as a least-squares problem with one constraint, read the
code that solves it, and then state when the loop stops.

| Symbol | Meaning | Code name |
|---|---|---|
| $e = FDS - SDF$ | the AO error (commutator) of one iteration | the stored error vector |
| $B_{ij}$ | $\langle e_i, e_j\rangle$, the Frobenius inner product of AO matrices | `gram(si, sj)` |
| $c_i$, $\lambda$ | extrapolation coefficients and the Lagrange multiplier | the solution vector |
| $m$ | number of stored pairs, at most `diis_size` (8) | `m` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/diis.rs::{Diis,DiisFlavor,EnergyDiis}` |
| CLI | `[scf] diis_size`, `diis` (`"pulay"`, `"adiis"`, `"ediis"`), `diis_switch_thresh`, `density_conv`, `energy_conv` |
| Python | the same names as keyword arguments of the SCF entry points |

## The error vector and the extrapolation {#diis-error-vector-extrapolation}

### The commutator is the orbital gradient

Work in the orthonormal MO basis of the current orbitals $C$. Write
$f = C^{\mathsf T}FC$ for the Fock matrix there and
$n = \mathrm{diag}(2_o, 0_v)$ for the closed-shell occupations, so that
$D = CnC^{\mathsf T}$. Transform the error $e = FDS - SDF$ with $C$ and use
$C^{\mathsf T}SC = 1$ twice:

$$ C^{\mathsf T}FDSC = C^{\mathsf T}FC\,n\,C^{\mathsf T}SC = fn, \qquad C^{\mathsf T}SDFC = C^{\mathsf T}SC\,n\,C^{\mathsf T}FC = nf . $$

So $C^{\mathsf T}eC = fn - nf$. The occupation $n$ is constant on each block,
so the occupied–occupied and virtual–virtual blocks of $fn - nf$ vanish.
Read off the two off-diagonal blocks, with $n_i = 2$ and $n_a = 0$:

$$ (fn - nf)_{ai} = f_{ai}n_i - n_af_{ai} = 2f_{ai}, \qquad (fn - nf)_{ia} = f_{ia}n_a - n_if_{ia} = -2f_{ia} . $$

Since $f$ is symmetric, $e$ is antisymmetric.

Now compare with the energy. Rotate the orbitals, $C \to Ce^{\kappa}$, with
$\kappa$ antisymmetric and only its virtual–occupied elements $\kappa_{ai}$
free ($\kappa_{ia} = -\kappa_{ai}$). Write $P = \mathrm{diag}(1_o, 0_v)$, so
$D = 2CPC^{\mathsf T}$. To first order in $\kappa$ the density moves by

$$ \delta D = 2C(\kappa P - P\kappa)C^{\mathsf T} . $$

The matrix $\kappa P - P\kappa$ is symmetric with $\kappa_{ai}$ in both the
$(a,i)$ and the $(i,a)$ corner: $\kappa P$ keeps the occupied columns of
$\kappa$, and $-P\kappa$ keeps its occupied rows with the sign flipped. The
closed-shell energy is $E = \mathrm{tr}(hD) + \tfrac12\mathrm{tr}(D\,G[D])$
with $G = J - \tfrac12K$ linear in $D$, so $dE = \mathrm{tr}(F\,\delta D)$ with
$F = h + G[D]$. Substitute $\delta D$ and sum the two corners:

$$ dE = 2\,\mathrm{tr}\big(f(\kappa P - P\kappa)\big) = 2\sum_{ai}(f_{ia} + f_{ai})\kappa_{ai} = 4\sum_{ai}f_{ai}\kappa_{ai}, \qquad \frac{\partial E}{\partial\kappa_{ai}} = 4f_{ai} . $$

The virtual–occupied block of $C^{\mathsf T}eC$ is $2f_{ai}$, half of this
gradient. Check the limit: at a stationary point every $f_{ai}$ is zero, so
$e = 0$; and $e = 0$ forces every $f_{ai}$ to zero. The condition $e = 0$
is therefore the Roothaan condition $FC = SC\varepsilon$ up to rotations inside the occupied and inside
the virtual space, which leave the energy unchanged and which $e$ cannot see.

For UHF the same algebra holds per spin, with
$e^\sigma = F^\sigma D^\sigma S - SD^\sigma F^\sigma$ and
$\partial E/\partial\kappa^\sigma_{ai} = 2f^\sigma_{ai}$ (each spin orbital is
singly occupied). ROHF uses the gradient of its energy functional mapped to
the AO basis, $e = SC(g - g^{\mathsf T})C^{\mathsf T}S$, and extrapolates the
Roothaan effective Fock matrix.

> **Verified equation {#eq:scf-diis-error}** · the error vector is the orbital gradient
>
> $e = FDS - SDF$ is antisymmetric; in the MO basis its occupied–virtual block
> is $2f_{ia}$, half of $\partial E/\partial\kappa_{ai} = 4f_{ai}$, so $e = 0$
> exactly at a stationary point. The UHF error is the same per spin.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_diis_sympy.py`, claims S1, S2, N1, N2 and N5 |
> | Negative control | $FDS + SDF$ must fail S1; the gradient factor 4 → 2 must fail N1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/diis.rs::Diis` |
>
> <!-- verified: id=scf.diis-error; script=scripts/scf_diis_sympy.py; claim=N1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/diis.rs::Diis; registry=rhf -->

### DIIS as a constrained least-squares problem

Model the error near the solution as a linear function of the Fock
matrix, $e(F) = L(F - F^*)$, with $L$ linear and $F^*$ the self-consistent
Fock matrix, where the error is zero. Combine stored Fock matrices with
coefficients $c_i$, apply $L$, and add and subtract $\sum_ic_iF^*$:

$$ e\Big(\sum_ic_iF_i\Big) = L\Big(\sum_ic_iF_i - F^*\Big) = \sum_ic_iL(F_i - F^*) + \Big(\sum_ic_i - 1\Big)LF^* = \sum_ic_ie_i + \Big(\sum_ic_i - 1\Big)LF^* . $$

The last term vanishes for every $F^*$ only if $\sum_ic_i = 1$, and then the
error of the combination is the same combination of the errors. That fixes
the constraint. The best combination minimizes

$$ \Big\lvert\sum_ic_ie_i\Big\rvert^2 = \sum_{ij}c_ic_jB_{ij} = c^{\mathsf T}Bc, \qquad B_{ij} = \sum_{\mu\nu}(e_i)_{\mu\nu}(e_j)_{\mu\nu}, $$

subject to $\mathbf 1^{\mathsf T}c = 1$. $B$ is the Gram matrix of the AO error
matrices as stored, not orthogonalized. It is positive semidefinite, so the
objective is convex and a stationary point on the constraint plane is the
minimum.

Add the constraint with a Lagrange multiplier. Divide $B$ by its largest
element first, $\tilde B = B/\max_{ij}\lvert B_{ij}\rvert$, as the code does:

$$ \mathcal L(c, \lambda) = c^{\mathsf T}\tilde Bc + 2\lambda\,(\mathbf 1^{\mathsf T}c - 1) . $$

Set the derivatives to zero. $\partial\mathcal L/\partial c = 2\tilde Bc + 2\lambda\mathbf 1 = 0$
and $\partial\mathcal L/\partial\lambda = 0$ is the constraint. Together they are
the bordered linear system the code solves:

$$\begin{pmatrix}\tilde B & \mathbf 1\\ \mathbf 1^{\mathsf T} & 0\end{pmatrix}\begin{pmatrix}c\\ \lambda\end{pmatrix} = \begin{pmatrix}0\\ 1\end{pmatrix} .$$

Solve it by hand. The first row gives $c = -\lambda\tilde B^{-1}\mathbf 1$.
Substitute into the constraint: $-\lambda\,\mathbf 1^{\mathsf T}\tilde B^{-1}\mathbf 1 = 1$.
Eliminate $\lambda$:

$$ c = \frac{\tilde B^{-1}\mathbf 1}{\mathbf 1^{\mathsf T}\tilde B^{-1}\mathbf 1} . $$

Multiply the first row $\tilde Bc = -\lambda\mathbf 1$ on the left by
$c^{\mathsf T}$ and use $c^{\mathsf T}\mathbf 1 = 1$: $\lambda = -c^{\mathsf T}\tilde Bc$.
So $-\lambda$ is the squared norm of the extrapolated error, in units of the
largest Gram element. Replacing $B$ by $sB$ for any $s > 0$ scales both the
numerator and the denominator of $c$ by $1/s$, so the scaling changes
$\lambda$ and leaves $c$ unchanged. The verified box below writes the
multiplier term as $-2\lambda(\cdot)$; that flips the sign of $\lambda$ and
gives the same $c$.

Check two limits. With one stored pair the constraint alone gives $c = 1$,
the plain Fock matrix; the code returns it without a solve. With two pairs
whose errors lie along one rotation, $e_2 = re_1$, the conditions
$c_1 + c_2r = 0$ and $c_1 + c_2 = 1$ give

$$ c_1 = -\frac{r}{1 - r}, \qquad c_2 = \frac{1}{1 - r}, $$

and the extrapolated error is exactly zero. The bordered matrix of this case
(Gram block before scaling, $e_1$ of unit norm) has determinant $-(1 - r)^2$, so it is
singular only when the two errors are equal, $r = 1$. Then the history holds
no new direction, and the code drops the older vector (section @sec:diis-algorithm).

What does this mean for the SCF? DIIS uses the stored iterations as a basis
and picks, inside their span, the Fock matrix whose predicted error, and so
whose orbital gradient, is smallest. It needs no Hessian: the history of
errors supplies the information about how the gradient responds to a change
in $F$.

> **Verified equation {#eq:scf-diis-system}** · the bordered DIIS system
>
> The stationarity conditions of $|\sum_ic_ie_i|^2 - 2\lambda(\sum_ic_i - 1)$
> are the bordered system with the $+1$ border; its solution is
> $c = \tilde B^{-1}\mathbf 1/(\mathbf 1^{\mathsf T}\tilde B^{-1}\mathbf 1)$,
> unchanged by scaling $B$, and it equals the minimizer over rotations of the
> stored errors.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_diis_sympy.py`, claims S3, N3 and N4 |
> | Negative control | the border $+1 \to -1$ in the code's system must fail N3 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/diis.rs::solve_diis_coeffs` |
>
> <!-- verified: id=scf.diis-system; script=scripts/scf_diis_sympy.py; claim=N3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/diis.rs::solve_diis_coeffs; registry=rhf -->

## The algorithm {#diis-algorithm}

| Step | Quantity | Where |
|---|---|---|
| 1 | $F$ from the current $D$ (chapter @sec:hartree-fock) | the SCF loop |
| 2 | $e = FDS - SDF$ (RHF, per spin for UHF); for ROHF $SC(g - g^{\mathsf T})C^{\mathsf T}S$ | the SCF loop |
| 3 | store $(F, e)$, dropping the oldest beyond `diis_size` | `ferric-scf/src/diis.rs::Diis` |
| 4 | solve the bordered system, shrinking from the oldest end on a singular pivot | `ferric-scf/src/diis.rs::solve_diis_coeffs` |
| 5 | $\bar F = \sum_ic_iF_i$, then an optional level shift (section @sec:level-shift-mom-smearing) | the SCF loop |

The solve is `ferric-scf/src/diis.rs::solve_diis_coeffs`. Here it is with its
comments replaced by the equation terms and the scaling loops collapsed:

```rust
for m in (2..=live.len()).rev() {                  // m = len, len-1, ..., 2
    let kept = &live[live.len() - m..];            // the newest m pairs
    let dim = m + 1;
    let mut a = vec![0.0f64; dim * dim];
    let mut rhs = vec![0.0f64; dim];
    for (i, &si) in kept.iter().enumerate() {
        for (j, &sj) in kept.iter().enumerate() {
            a[i * dim + j] = gram(si, sj);         // B_ij
        }
        a[i * dim + m] = 1.0;                      // border column 1
        a[m * dim + i] = 1.0;                      // border row 1^T
    }
    rhs[m] = 1.0;                                  // sum_i c_i = 1
    // ... g_scale = max |B_ij| over the Gram block only ...
    // ... B~ = B / g_scale, the border is left at 1 ...
    if let Some(c) = solve_linear(a, rhs, dim) {   // (c, lambda)
        return Some((c, kept.to_vec()));
    }
}
None
```

Four points about it:

- **The loop is over subspace sizes.** As the SCF converges, successive
  errors become nearly parallel and $B$ goes singular, as the two-vector
  example above shows at $r = 1$. The code then drops the oldest vector and
  retries, down to $m = 2$, for this step only. The newest vectors carry
  the current search direction, so they are the ones kept. If even the last
  two are dependent, `Diis::step` returns the current $F$ unextrapolated.
- **Only the Gram block is scaled.** $B_{ij}$ shrinks as $\lVert e\rVert^2$.
  Scaling the whole bordered matrix would let the border of ones set the
  scale, and the Gram block would become relative zero. Dividing the block
  alone keeps the two parts commensurate and leaves $c$ unchanged, by the
  scaling argument above.
- **The pivot test is relative.** `solve_linear` eliminates with partial
  pivoting on the matrix divided by its largest entry and rejects a pivot
  below $10^{-13}$ (`DIIS_PIVOT_RTOL`). A converging SCF does not trip it;
  linear dependence among the errors does.
- **One Gram row per iteration.** The Gram cache keeps $B_{ij}$ for the live
  history and recomputes only the row and column of the new error: $m$
  Frobenius products of $n_{\rm bas}\times n_{\rm bas}$ matrices per step.

## Variants {#diis-variants}

| Flavor | What it minimizes | Who reads it |
|---|---|---|
| Pulay (`diis = "pulay"`, the default) | $\lvert\sum_ic_ie_i\rvert^2$, $\sum c_i = 1$ | RHF, UHF, ROHF |
| `"ediis"`: EDIIS, then Pulay | an energy model on the simplex while $\max\lvert e\rvert \ge$ `diis_switch_thresh` (0.1), Pulay below | RHF only |
| `"adiis"`: ADIIS, then Pulay | the ADIIS model on the simplex, same switch | RHF only |

Far from convergence the linear error model of section @sec:diis-error-vector-extrapolation does not hold.
EDIIS and ADIIS
(`ferric-scf/src/diis.rs::EnergyDiis`) instead minimize a quadratic model of the
energy, $g^{\mathsf T}c + \tfrac12c^{\mathsf T}Hc$, over the simplex
$c_i \ge 0$, $\sum_ic_i = 1$:

| Flavor | $g_i$ | $H_{ij}$ |
|---|---|---|
| EDIIS | $E_i$ | $-\langle D_i - D_j, F_i - F_j\rangle$ |
| ADIIS | $2\langle D_i - D_n, F_n\rangle$ | $2\langle D_i - D_n, F_j - F_n\rangle$ |

Here $n$ is the newest entry. The code removes the inequality constraints by
the substitution $c_i = t_i^2/\sum_kt_k^2$, which satisfies both for any real
$t$, and runs backtracking gradient descent in $t$ from the uniform point
$c_i = 1/m$ (`minimize_on_simplex`). The driver uses the energy model while
$\max\lvert e\rvert \ge$ `diis_switch_thresh` (default 0.1) and Pulay below
it. Pulay receives every iteration's error throughout, so its history is
current at the switch.

The energy model is meaningful only for idempotent densities, whose energy
is bounded below by the ground state. A guess density (MINAO, SAD or a
user-supplied one) is not idempotent, so the RHF loop checks the first
density and keeps a non-idempotent one out of the energy history
(`DiisDriver::step_checked`).

UHF solves one system on the summed Gram matrix of both spins,
$B_{ij} = \langle e^\alpha_i, e^\alpha_j\rangle + \langle e^\beta_i, e^\beta_j\rangle$,
and applies the same coefficients to $F^\alpha$ and $F^\beta$, so the two
spins stay synchronized (`Diis::step_pair`). `diis_size = 1` leaves fewer
than two vectors, so the step is plain Roothaan iteration.

## When the SCF stops {#diis-scf-stops}

The exit test is `ferric-scf/src/rhf.rs::scf_converged`, shown whole:

```rust
let dp_rms_ok = sig.dp_rms < density_conv;          // Delta D_rms < density_conv
let dp_max_ok = sig.dp_max < 10.0 * density_conv;   // Delta D_max < 10 density_conv
let energy_not_descending = sig.de < energy_conv;   // |Delta E| < energy_conv
if dp_rms_ok && dp_max_ok && energy_not_descending {
    Some(crate::result::ScfExit::Converged)
} else {
    None
}
```

| Test | Default |
|---|---|
| $\Delta D_{\rm rms}$ between successive densities | `density_conv` $= 10^{-6}$ |
| $\Delta D_{\max}$ | $10 \times$ `density_conv` |
| $\lvert\Delta E\rvert$ | `energy_conv` $= 10^{-3}$, a sanity bound, not the target |
| exit veto | an exit with $\max\lvert X^{\mathsf T}eX\rvert > \max(\sqrt{\texttt{density\_conv}}, 10^{-3})$ is refused and DIIS restarts (`ferric-scf/src/rhf.rs::veto_nonstationary_rhf_exit`) |

The test runs from iteration 2 on, and the exit diagonalizes the
unextrapolated, unshifted $F$.

Why is the density the gate and not the commutator? With density fitting
the commutator stops falling at a floor set by the fit, while the density
change keeps draining to zero at the fitted fixed point. The $\Delta D_{\max}$
companion catches a single density element that is still moving while the
root-mean-square change is small.

The density test alone can accept a point that is not stationary. A stagnated extrapolation can return
the same density on every step without that density being stationary: the
doc comment of `GradientVeto` records OH UKS-LDA/STO-3G frozen at a
non-aufbau state 0.586 Ha above the stationary ground state. The veto
catches this by evaluating the gradient once the density test has passed.
It uses $X^{\mathsf T}eX$, the error in the orthonormal basis $X$
($X^{\mathsf T}SX = 1$), which has the same Frobenius norm as
$C^{\mathsf T}eC$ and the same $2f_{ai}$ block (claim N6 of
`scripts/scf_diis_sympy.py`). At the default `density_conv` the bound is
$\max(10^{-3}, 10^{-3}) = 10^{-3}$. A refused exit clears the DIIS history.

## Numerical anchors {#diis-numerical-anchors}

The convergence machinery is tested by the energies it converges to (the
anchors of section @sec:energy-functionals) and by the convergence-ladder row of
`site/src/reference/validation.md`: on N2 at 1.60 Å / def2-SVP the plain-DIIS
rung of `solve_rhf_ladder` lands on a saddle 1.906e-2 Ha above the minimum
that the stability descent of section @sec:stability then reaches.

<!-- figures: site/src/reference/validation.md "SCF convergence ladder with RHF internal stability descent" -->

## Doc fixes needed {#diis-doc-fixes-needed}

| Text | Code |
|---|---|
| `[scf] diis` reads as an option of every SCF | only the RHF solver reads `RhfConfig::diis_flavor`; UHF and ROHF always run Pulay |
| `crates/ferric-scf/theory.md` writes $(fn - nf)_{ai} = -2f_{ai}$, $(fn - nf)_{ia} = 2f_{ia}$ | the signs are the other way round: $(fn - nf)_{ai} = 2f_{ai}$, which claims S2 and N1 of `scripts/scf_diis_sympy.py` check |

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "DIIS: the error vector and the extrapolation" | the equations |
| `scripts/scf_diis_sympy.py` | verified equations @eq:scf-diis-error and @eq:scf-diis-system |
| `crates/ferric-scf/src/diis.rs` | the system and its shrink rule |
| `site/src/reference/validation.md` | the ladder anchor |
