# 7.2 Orthogonalization and starting guesses

<!-- section: 7.2; status: edit; registry: ;
     theory: crates/ferric-scf/theory.md#"Orthogonalization and starting guesses" -->

Every SCF iteration diagonalizes the Fock matrix in an orthonormal basis, and
every SCF starts from a density it did not compute. This section states how
the code builds that basis, which functions it drops, and what the default
starting density is.

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

## 7.2.1 The equations

<!-- include: crates/ferric-scf/theory.md#"Orthogonalization and starting guesses" -->

## 7.2.2 What is dropped

The threshold is absolute, not relative to the largest eigenvalue. A column
with $\lambda_i < 10^{-6}$ is removed, so a basis with near-dependent diffuse
functions solves in $m < N$ dimensions and the result's MO coefficient matrix
carries $N - m$ zero columns at energy $10^6$
(`ferric-scf/src/driver.rs::diagonalize_rect`). The environment variable
`FERRIC_LINDEP_THRESH` changes $\tau$ for one process. ROHF uses the
symmetric $S^{-1/2}$ when no column is dropped
(`ferric-scf/src/rohf.rs::rohf_orthogonalizer`) and the canonical form
otherwise.

## 7.2.3 The default guess

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

Three facts about the default that its name and docs do not convey:

| Fact | Consequence |
|---|---|
| No minimal-basis projection takes place: each atomic block is a free-atom SCF or a GWH density in the target basis | the name "MINAO projection" describes the history, not the algorithm |
| The atoms are neutral | $\operatorname{tr}(DS) = N_{\mathrm{el}}$ only for a neutral molecule; for an ion the first aufbau step fixes the count |
| Inter-atomic blocks are zero | the first Fock matrix has no bonding density; DIIS starts from it at iteration 1 |

## 7.2.4 Options that change the start

| Option | Effect |
|---|---|
| `guess = "hcore"` | diagonalize $X^{\mathsf T}hX$ and occupy the lowest orbitals |
| `guess_basis = "<name>"` | converge in a smaller basis, then project (`ferric-scf/src/basis2.rs`) |
| `dm0=` (Python) | start from a supplied density, checked for finiteness and electron count |
| `mom_after_iter` | applies from the guess onward; see section 8.2 |

## 7.2.5 Status

No proof script covers this section. The canonical orthogonalizer identity
$X^{\mathsf T}SX = 1_m$ is elementary. The guesses are heuristics, not
derivations, so their correctness is a matter of the SCF converging to the
same state from them, which the anchors of section 7.1 test. The section stays
`edit` until the doc fixes below land.

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
