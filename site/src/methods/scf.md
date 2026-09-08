# SCF and DFT

Ground-state self-consistent field methods, plus analytical nuclear gradients.

## Hartree–Fock

- **RHF** (closed-shell) with DIIS, Schwarz screening, and a choice of
  direct, LinK, density-fitted (RI-J / RI-K) or seminumerical (COSX) Fock
  builds — see *Choosing how exchange is built* below
- **UHF / ROHF** (open-shell) with per-spin DIIS, virtual-space level shifting,
  augmented-Hessian Newton, and **Maximum-Overlap-Method (MOM)** orbital
  tracking for near-degenerate cases

The convergence machinery is not decoration. Heavy atoms and near-degenerate
frontier orbitals genuinely break plain DIIS; the virtual-block level shift and
MOM exist because specific systems failed without them.

## Kohn–Sham DFT

Closed- and open-shell (RKS / UKS / ROKS) via **libxc**:

- LDA, GGA, hybrid, and range-separated-hybrid functionals — LDA, PBE, B3LYP,
  ωB97X-V
- **Becke–Lebedev** grids, with pruning
- **VV10** nonlocal correlation

## Gradients

Analytical nuclear gradients for RHF, UHF, ROHF and KS-DFT — **including grid
response** — validated against finite differences.

Hessians are **not** implemented. The mpqc4 libint2 export does not carry
second-derivative integrals, so the CPKS machinery is stubbed with explicit
TODOs rather than silently absent. See [Installation](../using/installation.md).

## Convergence

A few things worth knowing before debugging a stubborn SCF:

**Check `converged`.** These routines return a result whether or not they
converged; a non-converged SCF is a result with `converged = false`, not an
error. Downstream code that ignores the flag will happily consume a
half-converged density.

**Multiple solutions are real.** For systems like alkane chains, different
initial guesses converge to genuinely different SCF solutions — not a
convergence failure but a different basin. The guess picks the basin.

**Density-fitting has a noise floor.** DF-JK introduces an error floor that
makes energy-based convergence criteria below roughly 1e-9 meaningless; SCF
gates on the density RMS change instead.

**Near-linear-dependence.** Diffuse (aug-) basis sets on close-packed systems
can drive the overlap matrix near-singular; the canonical-orthogonalization
threshold is tunable via `FERRIC_LINDEP_THRESH`.

## Screening

- **Schwarz** bounds on every 4-centre path, built so they can never
  underestimate (a zero-valued table entry once cost 1.5e-4 Ha; it is now
  floored)
- **LinK** (Ochsenfeld, White & Head-Gordon 1998) — exchange via
  significant-pair lists; measured ~N^1.4 on alkane chains at def2-SVP
- **QQR** (Maurer, Lambrecht & Ochsenfeld 2012) is implemented and
  validated as a bound but is *not* used in production: on LinK it screens
  only 0.009% more quartets than Schwarz at alkane_16, because LinK's
  pair-list intersection already removes the long-range population QQR
  targets
- **COSX** shell-pair screening uses a primitive-level Hölder bound that
  provably never underestimates; an earlier overlap-based bound did, and
  silently corrupted K

## Choosing how exchange is built

Four ways to build K, and the choice is a real one — measured on this
code, not a rule of thumb. Butane, one thread; the QZ column is def2-QZVP
(528 functions), the TZ column def2-TZVP (184).

| `[scf]` setting | what it is | exact? | K at TZ | K at QZ | scope |
|---|---|---|---|---|---|
| *(default)* | Schwarz-screened direct 4-centre J+K | yes | — | 400 s (J+K) | all SCF types |
| `k_builder = "link"` | LinK — pair-list-screened direct K | yes | 6.5 s | 256 s | RHF only |
| `df_j_aux` / `df_k_aux` | density-fitted J and K (RI-JK) | ~1e-5 Ha | 0.05 s | 0.43 s | all SCF types |
| `k_builder = "cosx"` | seminumerical (COSX) K on a grid | grid error, see below | 358 s* | 137 s | RHF, Coulomb only, no gradients |

\* full SCF at TZ was 358 s for COSX against 98 s direct — COSX is *slower*
at TZ.

**Start with density fitting.** RI-JK is two to three orders of magnitude
faster than anything else here whenever its three-index tensor fits in memory
(`n_aux × n_bf² × 8` bytes — 1 GB for butane/QZVP, 7 GB for octane/QZVP), and
it spills to disk when it does not. Its error with a JK-fitting auxiliary basis
is a few µHa. If your system fits, this is the answer and the rest of this
section is about when it does not.

**Need exact exchange?** LinK for anything beyond a few heavy atoms; its cost
grows as roughly N^1.5 on alkane chains where plain direct grows faster. Both
are exact to the screening threshold as *K builders*. LinK is RHF-only — UHF
and ROHF silently fall back to direct. Known defect at the time of writing:
the LinK *SCF* path was non-variational beyond water (its density-dependent
pair lists were not refreshed after the first iteration — butane/def2-SVP
converged to −162.76 Ha against −157.19); check the validation page for
whether the fix has landed before using `k_builder = "link"` for an SCF.

**COSX is for large basis sets on systems too big for RI-JK.** Its cost per
grid point barely moves with angular momentum while analytic exchange grows
roughly tenfold from SVP to QZVP, so it crosses analytic K only at
quadruple-zeta: 34× slower than LinK at def2-SVP, 12× at TZVP, **0.54×** at
QZVP. Below QZ it is the wrong tool: on alkanes at def2-SVP it is 5.6–7.9×
slower than LinK at every size from C4 to C20.

Its integral work scales with system size the way LinK's does — a density-
driven pair screen (on the product of the integral bound and the local
half-transformed density) gives an A-build tail exponent of N^1.5 on C12–C20,
the same as LinK on the same series, with a K error below 2e-6 Ha at the
default threshold. The half-transforms `D·X` and `X·Gᵀ` are still dense
GEMMs, which grow faster and are a third of the build by C20; until they are
made sparse (the standard next step), expect the full build to scale roughly
N^2 past a dozen heavy atoms even though the integrals do not.

COSX's error is a grid error, and it is not µHa-small: 5e-6 Ha on water/cc-pVDZ
and 1.2e-4 Ha on butane/def2-TZVP at the default grid. Reaction energies
cancel most of it (0.02 kcal/mol on an isodesmic alkane reaction at the same
grid); absolute energies do not. Three knobs, all optional:

- `cosx_grid = { radial = 50, angular = 110 }` is the default and the coarsest
  grid that meets a 0.1 kcal/mol reaction-energy bar. Coarser grids fail it.
  Finer grids reduce the error roughly tenfold per step and cost proportionally.
- `cosx_overlap_fit = true` (default) applies the Izsák–Neese overlap
  correction. At the default grid it helps; on coarser grids it makes things
  *worse*, and its benefit is strongly molecule-dependent — large on water,
  nil to negative on ethane — so do not expect the factor quoted in the
  literature.
- `cosx_backend = "md3c1e"` (default) is the batched McMurchie–Davidson
  integral kernel. `"cosx-a"` is the per-point libint2 path, about three times
  slower and kept only as the cross-check the kernel is anchored against.
- `cosx_screen_thresh = 1e-7` (default) is the density-driven pair-screening
  threshold. `0.0` disables screening bit-identically; `1e-6` already fails a
  1e-6 Ha K-error bar on butane. Not available with the `"cosx-a"` backend.

Setting any `cosx_*` key without `k_builder = "cosx"`, or `k_builder` together
with `df_k_aux`, is refused or warned about rather than silently ignored.

## Determinism

The Fock build's reduction folds partial matrices in a **strict ascending group
order**, independent of thread count and of the memory band width. Results are
bit-identical across `RAYON_NUM_THREADS` — a property pinned by tests, not just
intended.

This matters more than it might seem: a tree-fold reduction would be equally
*deterministic* but would produce *different* bits, since floating-point
addition is not associative. The ascending order is load-bearing.
