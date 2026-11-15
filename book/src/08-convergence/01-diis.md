# 8.1 DIIS and its variants

<!-- section: 8.1; status: ready; registry: rhf, uhf, rohf;
     theory: crates/ferric-scf/theory.md#"DIIS: the error vector and the extrapolation" -->

Plain Roothaan iteration (diagonalize $F[D]$, rebuild $D$, repeat) oscillates
or diverges on many molecules. Every SCF solver in ferric extrapolates the
Fock matrix from its recent history instead. This section states the error
vector, why it vanishes exactly at a stationary point, the linear system the
code solves for the extrapolation, and the convergence test.

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

## 8.1.1 The equations

<!-- include: crates/ferric-scf/theory.md#"DIIS: the error vector and the extrapolation" -->

> **Verified equation 8.1** · the error vector is the orbital gradient
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

> **Verified equation 8.2** · the bordered DIIS system
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

## 8.1.2 The algorithm

| Step | Quantity | Where |
|---|---|---|
| 1 | $F$ from the current $D$ (chapter 7) | the SCF loop |
| 2 | $e = FDS - SDF$ (RHF, per spin for UHF); for ROHF $SC(g - g^{\mathsf T})C^{\mathsf T}S$ | the SCF loop |
| 3 | store $(F, e)$, dropping the oldest beyond `diis_size` | `ferric-scf/src/diis.rs::Diis` |
| 4 | solve the bordered system, shrinking from the oldest end on a singular pivot | `ferric-scf/src/diis.rs::solve_diis_coeffs` |
| 5 | $\bar F = \sum_ic_iF_i$, then an optional level shift (section 8.2) | the SCF loop |

The border of the system is filled in the same loop as the Gram block:

<!-- excerpt: ferric-scf/src/diis.rs::solve_diis_coeffs -->
```rust
for (i, &si) in kept.iter().enumerate() {
            for (j, &sj) in kept.iter().enumerate() {
                a[i * dim + j] = gram(si, sj);
            }
            a[i * dim + m] = 1.0;
            a[m * dim + i] = 1.0;
        }
        // Constraint row: Σ c_i = 1.
        rhs[m] = 1.0;
```

| Code | Equation |
|---|---|
| `kept` | the newest $m$ stored iterations |
| `gram(si, sj)` | $B_{ij}$ |
| `a[i * dim + m]`, `a[m * dim + i]` | the border $\mathbf 1$ |
| `rhs[m]` | the constraint $\sum_ic_i = 1$ |
| `dim` | $m + 1$ |

The Gram block alone is then divided by its largest element, which leaves
$c$ unchanged and keeps it commensurate with the border as $\lVert e\rVert$
shrinks.

## 8.1.3 Variants

| Flavor | What it minimizes | Who reads it |
|---|---|---|
| Pulay (`diis = "pulay"`, the default) | $\lvert\sum_ic_ie_i\rvert^2$, $\sum c_i = 1$ | RHF, UHF, ROHF |
| `"ediis"`: EDIIS, then Pulay | an energy model on the simplex while $\max\lvert e\rvert \ge$ `diis_switch_thresh` (0.1), Pulay below | RHF only |
| `"adiis"`: ADIIS, then Pulay | the ADIIS model on the simplex, same switch | RHF only |

UHF solves one system on the summed Gram matrix of both spins and applies the
same coefficients to $F^\alpha$ and $F^\beta$. `diis_size = 1` leaves fewer
than two vectors, so the step is plain Roothaan iteration.

## 8.1.4 When the SCF stops

| Test | Default |
|---|---|
| $\Delta D_{\rm rms}$ between successive densities | `density_conv` $= 10^{-6}$ |
| $\Delta D_{\max}$ | $10 \times$ `density_conv` |
| $\lvert\Delta E\rvert$ | `energy_conv` $= 10^{-3}$, a sanity bound, not the target |
| exit veto | an exit with $\max\lvert X^{\mathsf T}eX\rvert > \max(\sqrt{\texttt{density\_conv}}, 10^{-3})$ is refused and DIIS restarts (`ferric-scf/src/rhf.rs::veto_nonstationary_rhf_exit`) |

The density test is the gate; the commutator is not. A converged density with
a large commutator is the signature of a run that stalled with a small step,
which the veto catches.

## 8.1.5 Numerical anchors

The convergence machinery is tested by the energies it converges to (the
anchors of section 7.1) and by the convergence-ladder row of
`site/src/reference/validation.md`: on N2 at 1.60 Å / def2-SVP the plain-DIIS
rung of `solve_rhf_ladder` lands on a saddle 1.906e-2 Ha above the minimum
that the stability descent of section 8.4 then reaches.

<!-- figures: site/src/reference/validation.md "SCF convergence ladder with RHF internal stability descent" -->

## 8.1.6 Doc fixes needed

| Text | Code |
|---|---|
| `[scf] diis` reads as an option of every SCF | only the RHF solver reads `RhfConfig::diis_flavor`; UHF and ROHF always run Pulay |

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "DIIS: the error vector and the extrapolation" | the equations |
| `scripts/scf_diis_sympy.py` | verified equations 8.1 and 8.2 |
| `crates/ferric-scf/src/diis.rs` | the system and its shrink rule |
| `site/src/reference/validation.md` | the ladder anchor |
