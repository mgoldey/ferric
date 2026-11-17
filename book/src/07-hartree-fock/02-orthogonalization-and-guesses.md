# Orthogonalization and starting guesses {#orthogonalization-and-guesses}

<!-- status: edit; registry: ;
     theory: crates/ferric-scf/theory.md#"Orthogonalization and starting guesses" -->

Atomic orbitals on neighbouring atoms overlap, so the Roothaan equations
$FC = SC\varepsilon$ carry the overlap $S$, and a standard eigensolver wants
an orthonormal basis instead. With large or diffuse basis sets, two diffuse
functions on nearby atoms can describe almost the same function, so some
combination of basis functions has almost zero length, and normalizing it
multiplies it, and any rounding error along it, by a large factor. The code
changes once per geometry to an orthonormal basis that leaves such
combinations out. Every SCF also starts from a density it did not compute,
because the first Fock matrix needs one; a better start means fewer
iterations and less risk of converging to the wrong state.

This section derives the orthogonalizer (@sec:orthogonalization-and-guesses-equations), states what it drops
(@sec:orthogonalization-and-guesses-dropped), and describes the default guess and the options that replace it
(@sec:orthogonalization-and-guesses-default-guess, @sec:orthogonalization-and-guesses-options-change-start).

| Symbol | Meaning | Code name |
|---|---|---|
| $S = U\lambda U^{\mathsf T}$ | overlap and its eigendecomposition | `s_evals`, `s_evecs` |
| $\tau$ | linear-dependence threshold, absolute | `lindep_thresh()` (`LINDEP_THRESH`, $10^{-6}$) |
| $X$ | $N \times m$ orthogonalizer, $X^{\mathsf T}SX = 1_m$ | `x` |
| $m$ | number of retained functions | `x.ncols()` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/rhf.rs::canonical_orthogonalizer`, `ferric-scf/src/guess.rs::{hcore_guess,minao_projection_guess}` |
| CLI | `[scf] guess = "minao"` (default; alias `"sad"`) or `"hcore"`; `[scf] guess_basis` |
| Python | `run_rhf(guess=..., guess_basis=..., dm0=...)` and the open-shell analogues (`crates/ferric-python/ferric.pyi`) |

## The equations {#orthogonalization-and-guesses-equations}

### Canonical orthogonalization

Diagonalize the overlap once, $S = U\lambda U^{\mathsf T}$, with the
eigenvalues $\lambda_i$ in ascending order. Each eigenvector $u_i$ is a
combination of basis functions, and $\lambda_i = u_i^{\mathsf T}Su_i$ is its
squared length. Keep the columns with $\lambda_i \ge \tau$ and scale each to
unit length:

$$ X_{\mu c} = U_{\mu i_c}\,\lambda_{i_c}^{-1/2}, \qquad X^{\mathsf T}SX = 1_m, \quad m \le N . $$

Check the identity. Write $U_k$ for the kept columns and $\lambda_k$ for
their eigenvalues, so $X = U_k\lambda_k^{-1/2}$. Then
$X^{\mathsf T}SX = \lambda_k^{-1/2}U_k^{\mathsf T}U\lambda U^{\mathsf T}U_k\lambda_k^{-1/2}$.
The eigenvectors are orthonormal, so $U^{\mathsf T}U_k$ picks out the kept
rows of the identity and the middle product is $\lambda_k$. What remains is
$\lambda_k^{-1/2}\lambda_k\lambda_k^{-1/2} = 1_m$.

Now move the Roothaan equations into this basis. Write $C = X\tilde C$,
substitute into $FC = SC\varepsilon$, and multiply on the left by
$X^{\mathsf T}$:
$X^{\mathsf T}FX\tilde C = X^{\mathsf T}SX\tilde C\varepsilon = \tilde C\varepsilon$.
So each iteration diagonalizes

$$ \tilde F = X^{\mathsf T}FX = \tilde C\varepsilon\tilde C^{\mathsf T}, \qquad C = X\tilde C, $$

and the orbitals satisfy $C^{\mathsf T}SC = \tilde C^{\mathsf T}X^{\mathsf T}SX\tilde C = 1_m$.
They span the retained space only.

Why drop small eigenvalues? Column $c$ of $X$ is $u_c/\sqrt{\lambda_c}$, so
the diagonal element of $\tilde F$ along it is
$\tilde F_{cc} = u_c^{\mathsf T}Fu_c/\lambda_c$. An error $\delta F$ in the
Fock matrix moves that element by $u_c^{\mathsf T}\delta Fu_c/\lambda_c$,
which is at most $\lVert\delta F\rVert_2/\lambda_c$ in magnitude. At
$\lambda_c = 10^{-6}$ the factor is $10^6$. The function $u_c$ is almost the
zero function, so dropping it loses almost nothing from the variational
space.

Check the limits. If nothing is dropped ($m = N$), $X$ is square and
invertible, and $C = X\tilde C$ spans the full space: the result is the
Roothaan solution in the full basis. If one column is dropped, the SCF runs
in $N - 1$ dimensions and returns one fewer orbital with physical meaning.

$\tau$ is absolute, $10^{-6}$ (`LINDEP_THRESH`), overridden by the
environment variable `FERRIC_LINDEP_THRESH` (`lindep_thresh`). Each
iteration diagonalizes $\tilde F$ in `ferric-scf/src/driver.rs::diagonalize_rect`,
and the $N - m$ dropped columns are padded with zeros and energy $10^6$. A
guess that needs more occupied orbitals than $m$ is an error.

ROHF uses the symmetric $X = U\lambda^{-1/2}U^{\mathsf T}$ when nothing is
dropped (`rohf_orthogonalizer`). It also satisfies $X^{\mathsf T}SX = 1$, and
when $m = N$ both orthogonalizers solve the same full-space eigenproblem.

### The core guess

`hcore_guess` drops the electron repulsion altogether: diagonalize
$X^{\mathsf T}hX$ and occupy the lowest $n_o$ orbitals,
$D = 2C_oC_o^{\mathsf T}$. It ignores electron repulsion. The documentation of
`RhfConfig::use_sad_guess` records that the default guess below cures a class
of heavy-atom RHF divergences (COSe, C2H3Br) that the core guess triggers.

### The default guess

The default (`[scf] guess = "minao"`, alias `"sad"`;
`minao_projection_guess`) is a superposition of atomic densities. $D$ is
block diagonal over atoms, with zero inter-atomic blocks. Each atomic block is
built in the target basis; there is no projection from a minimal basis,
despite the name. Two kinds of block are used:

- $Z \le 20$ with no $g$ or higher shell: the density of a free-atom SCF in
  that atom's basis. The atom is solved with RHF for a singlet ground state,
  otherwise UHF with MOM from iteration 5. The density is then spherically
  averaged: each diagonal shell block is replaced by its mean diagonal times
  the unit matrix, and the cross-$l$ blocks are zeroed.
- Otherwise: a generalized Wolfsberg–Helmholz (GWH) block,

  $$ F_{\mu\mu} = h_{\mu\mu}, \qquad F_{\mu\nu} = \tfrac12 KS_{\mu\nu}(h_{\mu\mu} + h_{\nu\nu}), \quad K = 1.75, $$

  diagonalized in the atom's own orthogonalizer, with occupation 2 on the
  lowest doubly occupied orbitals and the $2S$ unpaired electrons spread
  equally over the orbitals within $10^{-4}$ Ha of the first partly filled
  one (`atomic_hcore_density`).

The GWH matrix is a Hückel-type model. The diagonal $h_{\mu\mu}$ stands in
for the energy of an electron in $\chi_\mu$, and the coupling of two
functions is proportional to their overlap and to their mean energy. It needs
only one-electron integrals, so it works for any element and any shell.
Spreading the unpaired electrons over the degenerate frontier orbitals keeps
the atomic density spherical; filling two of three $p$ orbitals would give
the atom a shape.

The blocks are those of neutral atoms, so $\operatorname{tr}(DS) = N_{\mathrm{el}}$
holds for a neutral molecule only; the first Fock build and the aufbau step
restore the electron count.

### Open-shell guesses

Open-shell guesses start from the same closed-shell $D$. UHF sets
$D^\alpha = D^\beta = D/2$, builds $F^\alpha = F^\beta = h + J[D] - K[D/2]$
without exchange–correlation, diagonalizes per spin, and for
$n_\alpha = n_\beta$ mixes the $\beta$ HOMO and LUMO by 0.1 rad
(`uhf_guess_mos`, `mix_beta_homo_lumo`). Without the mixing the two spins
would start equal and stay equal, and a broken-symmetry solution could not
be reached. ROHF scales $D^\sigma = D\,n_\sigma/N_{\mathrm{el}}$ and
diagonalizes the Roothaan effective Fock matrix of section @sec:energy-functionals
(`rohf_guess_mos`).

### The secondary basis

With `[scf] guess_basis` (`crates/ferric-scf/src/basis2.rs`) the same SCF is
first converged in a smaller basis, and its result is moved into the target
basis. Write $P = S_{bb}^{+}S_{bs}$, where $S_{bb}^{+} = XX^{\mathsf T}$ is the
lindep-filtered inverse of the target overlap and $S_{bs}$ is the overlap
between the target and the small basis. The Fock projection carries the
converged small-basis densities, $D^\sigma = PD^\sigma_{\mathrm{small}}P^{\mathsf T}$.
The overlap projection carries the occupied orbitals,
$C = PC_{o,\mathrm{small}}$, re-orthonormalized as
$C(C^{\mathsf T}S_{bb}C)^{-1/2}$. It is refused when the smallest eigenvalue
of $C^{\mathsf T}S_{bb}C$ is below $10^{-8}$, because the projected orbitals
are then nearly dependent.

## What is dropped {#orthogonalization-and-guesses-dropped}

The threshold is absolute, not relative to the largest eigenvalue. A column
with $\lambda_i < 10^{-6}$ is removed, so a basis with near-dependent diffuse
functions solves in $m < N$ dimensions and the result's MO coefficient matrix
carries $N - m$ zero columns at energy $10^6$
(`ferric-scf/src/driver.rs::diagonalize_rect`). The environment variable
`FERRIC_LINDEP_THRESH` changes $\tau$ for one process. ROHF uses the
symmetric $S^{-1/2}$ when no column is dropped
(`ferric-scf/src/rohf.rs::rohf_orthogonalizer`) and the canonical form
otherwise.

The selection is three lines of
`ferric-scf/src/rhf.rs::canonical_orthogonalizer`, shown here with the BLAS
thread wrapper and the error mapping around the eigensolver removed and the
`UPLO` path shortened:

```rust
let (s_evals, s_evecs) = s.eigh(UPLO::Upper)?;           // S = U λ U^T, λ ascending
let thresh = lindep_thresh();                            // τ, 1e-6 unless overridden
let kept: Vec<usize> = (0..n).filter(|&i| s_evals[i] >= thresh).collect();
let m = kept.len();
let mut x = Array2::<f64>::zeros((n, m));
for (col, &i) in kept.iter().enumerate() {
    let scale = 1.0 / s_evals[i].sqrt();                 // λ_i^{-1/2}
    for mu in 0..n {
        x[(mu, col)] = s_evecs[(mu, i)] * scale;         // X_{μc} = U_{μi} λ_i^{-1/2}
    }
}
```

- `X` is rectangular, and not symmetric even when $m = N$. Callers must use
  the transform $X^{\mathsf T}FX$ and $C = X\tilde C$, not treat $X$ as
  $S^{-1/2}$.
- The comparison is `>=`, so an eigenvalue equal to $\tau$ is kept.
- The doc comment of `lindep_thresh` records why the default is not raised
  globally: some diffuse metal clusters converge only with a larger $\tau$,
  while a well-conditioned aromatic system at the same basis has modes in
  $[10^{-6}, 10^{-4})$ that its SCF handles and a larger $\tau$ would drop.
  The override is per process for that reason.

## The default guess {#orthogonalization-and-guesses-default-guess}

The default is a block-diagonal atomic density, built atom by atom in the
target basis. The GWH branch, used for $Z \ge 21$ or any atom with a $g$ or
higher shell, is a single loop:

<!-- excerpt: ferric-scf/src/guess.rs::atomic_hcore_density -->
```rust
for i in 0..nao_h {
        f_eff[(i, i)] = h[(i, i)];
        for j in (i + 1)..nao_h {
            let fij = 0.5 * GWH_K * s[(i, j)] * (h[(i, i)] + h[(j, j)]);
            f_eff[(i, j)] = fij;
            f_eff[(j, i)] = fij;
        }
    }
```

| Code | Equation |
|---|---|
| `h` | the atom's core Hamiltonian block $h$ |
| `s` | the atom's overlap block $S$ |
| `GWH_K` | $K = 1.75$ |
| `f_eff` | the GWH matrix $F$ diagonalized for the atomic density |

The diagonal is set separately because the off-diagonal formula at
$\mu = \nu$, for a normalized function ($S_{\mu\mu} = 1$), would give $Kh_{\mu\mu}$ instead of
$h_{\mu\mu}$.

Three facts about the default that its name and docs do not convey:

| Fact | Consequence |
|---|---|
| No minimal-basis projection takes place: each atomic block is a free-atom SCF or a GWH density in the target basis | the name "MINAO projection" describes the history, not the algorithm |
| The atoms are neutral | $\operatorname{tr}(DS) = N_{\mathrm{el}}$ only for a neutral molecule; for an ion the first aufbau step fixes the count |
| Inter-atomic blocks are zero | the first Fock matrix has no bonding density; DIIS starts from it at iteration 1 |

## Options that change the start {#orthogonalization-and-guesses-options-change-start}

| Option | Effect |
|---|---|
| `guess = "hcore"` | diagonalize $X^{\mathsf T}hX$ and occupy the lowest orbitals |
| `guess_basis = "<name>"` | converge in a smaller basis, then project (`ferric-scf/src/basis2.rs`) |
| `dm0=` (Python) | start from a supplied density, checked for finiteness and electron count |
| `mom_after_iter` | applies from the guess onward; see section @sec:level-shift-mom-smearing |

## Status {#orthogonalization-and-guesses-status}

No proof script covers this section. The canonical orthogonalizer identity
$X^{\mathsf T}SX = 1_m$ is elementary (@sec:orthogonalization-and-guesses-equations writes it out). The guesses are
heuristics, not derivations, so their correctness is a matter of the SCF
converging to the same state from them, which the anchors of section @sec:energy-functionals
test. The section stays `edit` until the doc fixes below land.

| Doc fix needed | Where |
|---|---|
| the `use_sad_guess` documentation says it runs `sad_guess`; the code runs `minao_projection_guess` | `crates/ferric-scf/src/rhf.rs` (`RhfConfig`) |
| "MINAO projection" projects nothing | `crates/ferric-scf/src/guess.rs` module doc |
| the electron count is exact for neutral molecules only | `crates/ferric-scf/src/guess.rs` |
| the `[scf] guess` field doc calls `"sad"` the legacy free-atom-SCF superposition; the parser maps it to `minao` | `crates/ferric-cli/src/config.rs` |
| orphaned doc comments detached from their functions | `crates/ferric-scf/src/guess.rs`, `crates/ferric-scf/src/rohf.rs` |

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Orthogonalization and starting guesses" | the equations |
| `crates/ferric-scf/src/guess.rs` | the atomic blocks and the GWH constant |
| `crates/ferric-scf/tests/basis2_guess.rs` | the secondary-basis projection |
| `crates/ferric-scf/src/rhf.rs` (`canonical_orthogonalizer`, `lindep_thresh`) | the selection loop and the reason the threshold is per process |
