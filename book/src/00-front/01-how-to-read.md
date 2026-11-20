# How to read this book {#how-to-read}

<!-- status: edit; registry: none -->

This book explains how each equation in ferric becomes code. Pick any
equation and the book takes you to the function that contracts it, then to
the script that proves it and the test that runs the script.

You should know quantum chemistry at the level of a first graduate course:
Hartree–Fock, the idea of a basis set, perturbation theory, a determinant.
You should also read Rust and Python, because every derivation ends at a
real function. The book does not teach the methods from scratch. It is
written for someone who implements or audits an electronic-structure
method, and it shows each step so that you can follow a derivation with a
pencil and match each line of the code with the source open.

The book describes the code as it is. Where a limit is a current fact, a
section ends with a short "Known limits" list that carries the date it was
checked. The book keeps no history of what was tried and withdrawn.

## What a section contains

A section has two halves. The first explains the method; the second is the
reference you return to when you change the code. A section omits an item
only when it has nothing to say there, and the order does not change.

The explanation comes first:

1. **The problem, physically.** What goes wrong without the method, or what
   it computes, in a chemist's terms: electrons avoiding each other, an
   orbital sitting on a bond, an integral that is nearly zero.
2. **The plan.** In a section with three or more subsections, one sentence
   or a short list.
3. **Notation and entry points.** A table of the index ranges the section
   uses, with their names in the code, and the entry points in Rust, the CLI
   and Python.
4. **The derivation.** Every step written out, with the operation named
   (expand, substitute, sum over spin). After each key result the text checks
   a limit you can do in your head: one pair, one atom, $\omega \to 0$, a
   threshold of zero.
5. **What it means.** One paragraph of interpretation after the important
   equation, with a number from a runnable example where there is one.
6. **The code.** An excerpt of the real function, with each line commented
   by the term of the equation it computes, followed by a few points on sign
   conventions, data layout and how the loop differs from the textbook form.
7. **Design choices.** Why the code does X instead of the obvious Y, and the
   cost or accuracy reason.
8. **What is proven.** The verified-equation boxes, and one sentence on any
   step the proof does not cover.

The reference part follows:

| Item | What it gives you |
|---|---|
| The algorithm | A table of steps with cost, memory and the function for each step |
| A kernel excerpt | The innermost loop, with a table from symbols to variable names |
| Reproducibility and memory | Rules that affect a result, such as summation order and the memory budget |
| Options | The settings that change the equation |
| Numerical anchors | Agreement figures with an independent reference, each traced to a test or to the validation page |
| Running it | A TOML input or a Python call you can paste |
| Sources | Where each equation came from, and what the section changed |
| Known limits | The current limits, dated |

Section @sec:closed-shell-ri-mp2 (closed-shell RI-MP2) and section @sec:amplitude-lmp2 (amplitude-threshold
local MP2) show the whole shape. Read one of them first if you want to see
how the parts fit.

## Verified and unverified equations

A working equation sits in a box. The box says what checks it. A verified
box looks like this:

> **Verified equation {#eq:how-to-read-box}** · what a verified box looks like
>
> The statement of the identity, in the notation of the Notation chapter.
>
> | | |
> |---|---|
> | Proof | the script and the claim in it |
> | Negative control | a plausible wrong variant that must fail |
> | Runner | the test that executes the script |
> | Code | the function that implements the equation |

Read the rows from top to bottom. The **Proof** row names a script under
`scripts/` and a claim inside it. The **Negative control** is a wrong
variant of the equation, such as the same-spin MP2 energy without its
exchange term, that the script must reject. A proof that also accepted the
wrong variant would prove nothing. The **Runner** is the test that executes
the script, and the **Code** row names the function that computes the
equation.

The build refuses a box whose script, claim or runner does not exist. The
claim checks the identity the box states. It does not check that the Rust
function computes that identity. That second check belongs to a Rust test
(named in the box when there is one) or to a numerical anchor. A box with
neither is an identity-only box, and the build says so.

An **unverified** box has the same shape and the heading "Unverified
equation". Its Proof row states what proof is missing and where to add it,
and a Status row says what does exist, for example "derived in the text; no
script, no test". Treat such an equation as derived by hand and read against
the code, not as machine checked. Appendix A lists the unverified boxes and
their count. The count may only fall between releases: a release build
compares it with the ceiling in `book/unverified-ceiling.txt`.

To run one proof yourself:

```
python3 scripts/mp2_variant_identities.py
```

To run every proof, each also under its recorded mutations (a mutation is a
one-line change to the script that must turn it red):

```
OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest crates/ferric-python/tests/test_derivation_scripts.py -q
```

The runner is `crates/ferric-python/tests/test_derivation_scripts.py`. It
needs `sympy` and `numpy`, and no compiled extension.

## Code references

The book cites code as a path under `crates/`, then `::`, then the item
name, for example `ferric-mp2/src/rimp2.rs::pair_energy`. The item is
declared in that file, and the book checker fails if it is not. A braced
list after the `::` cites several items in one file.

The book never cites a line number of repository code, because line numbers
move with every edit; the checker rejects a `path:line` reference.
Local documents outside the repository, such as the dissertation text, are
the exception. The HTML build resolves each name to a line at the build commit and
links to it. A bare function name in the text, without the path, is not
checked, so the first mention of an item in a section uses the full form.

A fenced Rust or Python block marked as an excerpt occurs in the cited item,
with whitespace and comments ignored, so the comments that map code to
symbols are the book's own. A change to that kernel fails the build until
the excerpt and the text around it are reread. An excerpt may be trimmed;
the text says what was removed, and a `// ...` (or `# ...`) line marks each
gap. The pieces between gaps must occur in the item in the order shown.

## Status of a section

Each section file starts with a header comment that records its status.
Some chapter overviews repeat it in their section tables.

| Status | Meaning |
|---|---|
| `ready` | A committed derivation with function names, and a proof script with a runner. |
| `edit` | A committed derivation or dissertation text exists, but it is thin, has no runner-backed proof for some equations, or needs restructuring. |
| `gap` | No committed derivation. |
| `verify` | The only source is a working note or an unpublished draft, or the registry records the committed text as stale or unproven. |

Reviewers set the status. A `gap` or `verify` section can still be worth
reading, but check each equation in it against the code before you rely on
it.

## Numbers and grades

A figure in an anchors table comes from the validation page
(`site/src/reference/validation.md`) or from a committed test, and the
section names which. The book contains no number from an unrecorded run.

The validation page grades every CLI `method.kind`. These are the grades
ferric prints at run time.

| Grade | Meaning |
|---|---|
| Proven | Agreement at a stated tolerance with an independent reference, pinned by tests. An independent reference is another code, a numpy reference, a published value, or an exact limit the method must reduce to. |
| Proven (narrow) | As Proven, but only on the stated class of systems, or against the stated kind of reference. |
| Smoke | Runs end to end and some pieces are checked, but there is no independent reference for the headline number, or only a loose one. Do not quote such a result as a reference value. |
| Spike | Built on new infrastructure and not compared with any reference code. For exploration only. |
| not graded | Dispatched with no recorded grade. Treat it as unproven. |

A method can be implemented without being validated. A chapter that
describes a method states its grade and refers to the validation page for
the evidence. The grades are the `METHOD_KIND_GRADES` table in
`crates/ferric-core/src/certificate.rs`, and a test keeps that table equal to
the validation page.

A status and a grade answer different questions. The status says how well
the book's text is supported. The grade says how well ferric's numbers are
supported.

## How the chapters depend on each other

Chapters @sec:second-quantization to @sec:density-fitting are the foundations: second quantization and spin
adaptation (@sec:second-quantization), molecules, basis sets and the numerical rules (@sec:molecules-and-numerics), integrals
(@sec:integrals), attenuated operators (@sec:operators), screening (@sec:screening) and density fitting (@sec:density-fitting). Every
later chapter uses them. Read chapter @sec:second-quantization and section @sec:factorization-and-metric before anything that
writes a working equation; the rest you can look up when a section points
to it.

Most of the book is reached through the self-consistent field, chapters @sec:hartree-fock
and @sec:convergence. Three paths through it:

- **Correlation and local correlation.** @sec:hartree-fock → @sec:mp2 → @sec:rimp2-algorithms → @sec:oo-laplace → @sec:local-mp2. Chapter @sec:mp2
  derives MP2 and its variants on top of Hartree–Fock, chapter @sec:rimp2-algorithms builds the
  RI-MP2 algorithm and its memory model, chapter @sec:oo-laplace adds orbital
  optimization and the Laplace forms, and chapter @sec:local-mp2 truncates the
  canonical calculation of chapter @sec:mp2 to local pairs.
- **Density functional theory and excited states.** @sec:hartree-fock → @sec:kohn-sham → @sec:tddft. Chapter @sec:kohn-sham
  adds the exchange-correlation functional to the SCF of chapter @sec:hartree-fock, and
  chapter @sec:tddft builds linear response on the converged reference, using the
  $f_{xc}$ kernel of section @sec:fxc.
- **Forces.** @sec:hartree-fock → @sec:derivatives → @sec:correlated-gradients. Chapter @sec:derivatives differentiates the SCF energy;
  chapter @sec:correlated-gradients adds the Z-vector that correlated gradients need.

Beyond these: coupled cluster (@sec:coupled-cluster) builds on chapter @sec:mp2, the multireference
chapter (@sec:multireference) on chapters @sec:second-quantization and @sec:convergence, and RPA (@sec:rpa) on MP2, whose ring
diagrams it sums to infinite order. GW and the
Bethe–Salpeter equation (@sec:gw-bse) and the polarizabilities (@sec:polarizabilities) build on
chapter @sec:rpa. Chapters @sec:implicit-solvation to @sec:dispersion cover solvation, embedding, periodic systems
and dispersion. Chapter @sec:rs-mp2-rpa combines short-range MP2 (chapter @sec:mp2 and
the range-split cross term of section @sec:range-split-cross-term) with long-range dRPA
(chapter @sec:rpa). Chapter @sec:ml lists where the computed
quantities feed machine-learning work.

## Running the examples

Python snippets use bundled molecules and basis sets, with paths relative
to the repository root (`testdata/molecules/water.xyz`, `cc-pvdz`), so run
them from there. Install the `ferric` package as the user guide describes
(`site/src/using/installation.md`). A CLI example is a TOML file run as

```
cargo run --release -- examples/water-rimp2.toml
```

A snippet that prints numbers names the build that produced them, for
example "from the `a410529d0` build". Check `ferric.__build__` or
`ferric.build_info()` to see which commit your installed package was built
from. The printed values were run with `OPENBLAS_NUM_THREADS=1`; other
thread counts can change the last bits of a result, as section @sec:determinism-and-memory
explains.

## Where the equations live

The derivations live in this book. Each crate's `theory.md` is a short
generated index that lists the book sections citing that crate's code, with
their proof scripts and verified boxes. The verified boxes tie each
derivation to its proof script and to the code.

The Notation chapter fixes one notation for the whole book. Where it differs
from the literature, the Notation chapter says so.

## Scope

| In the book | In the user guide (`site/`) |
|---|---|
| The equations, the algorithms and the proofs behind the methods | How to run a method, its keywords, measured timings, the validation matrix |
| Memory and determinism rules that change a result | Installation, the Python and Avogadro surfaces, ASE and QCSchema |

The semiempirical interface is a one-page appendix (Appendix D). The user
guide documents every CLI key and Python keyword.

## Authorship and licence

Matthew Goldey, with ferric contributors. The book source is licensed as
ferric is, under MIT OR Apache-2.0. Method definitions taken from the
author's dissertation are cited to it. Its benchmark tables and figures are
not reproduced; any figure in the book comes from a ferric run.
