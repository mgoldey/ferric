# 7.1 RHF, UHF and ROHF energy functionals

<!-- section: 7.1; status: ready; registry: rhf, uhf, rohf;
     theory: crates/ferric-scf/theory.md#"Hartree-Fock energy functionals (RHF, UHF, ROHF)" -->

Hartree–Fock is the reference for every correlated method in this book, and
its Fock matrix is the object every builder in this chapter computes. This
section states the three energy functionals the solvers minimize, the Fock
matrices that are their derivatives, and the stationarity conditions the
convergence tests of chapter 8 check.

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

## 7.1.1 The functionals

The derivation of record is in `crates/ferric-scf/theory.md`, included here.
Its notation is the book's: $D = 2C_oC_o^{\mathsf T}$ for the closed shell,
$D^\sigma = C^\sigma_o C^{\sigma\,\mathsf T}_o$ per spin, and
$F = h + J[D] - \tfrac12 K[D]$.

<!-- include: crates/ferric-scf/theory.md#"Hartree-Fock energy functionals (RHF, UHF, ROHF)" -->

> **Verified equation 7.1** · the Fock matrix is the density derivative of the energy
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
> <!-- verified: id=scf.fock-is-de-dd; script=scripts/hf_energy_sympy.py; claim=S1 dE_RHF/dD = F_RHF (symbolic, n=2); runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/rhf.rs::solve_rhf, ferric-scf/src/uhf.rs::solve_uhf; registry=rhf -->

> **Verified equation 7.2** · the ROHF orbital gradient and the effective Fock matrix
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
> <!-- verified: id=scf.rohf-gradient; script=scripts/hf_energy_sympy.py; claim=N4 dE/dkappa = 2 g (rohf_gradient_mo) off convergence; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/rohf.rs::{rohf_gradient_mo,roothaan_fock_inner}; registry=rohf -->

## 7.1.2 One iteration

Each solver repeats the same five steps. The rest of this chapter is about
step 2, and chapter 8 about steps 3 and 4.

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

## 7.1.3 Options that change the equation

| Option | Effect | Where |
|---|---|---|
| `[dft] functional` | replaces $-\tfrac12 K$ by $-\tfrac12 c_x K + V_{xc}$ (RKS, UKS, ROKS) | chapter 9 |
| `[external_potential]`, `[qmmm]` | adds the point-charge and field terms to $h$ once, before the loop | chapter 22 |
| `[cosmo]`, `[pcm]` | adds a reaction-field term to $F$ every iteration | chapter 23 |
| `fock_hook` (Python) | adds $V = \partial E_h/\partial D$ to $F$ every iteration | section 7.6 |
| `relativistic = "x2c1e"` | replaces $T + V$ in $h$ by the spin-free X2C-1e Hamiltonian | the user guide |

## 7.1.4 Numerical anchors

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
| `scripts/hf_energy_sympy.py` | verified equations 7.1 and 7.2 |
| `site/src/reference/validation.md` | the anchors |
