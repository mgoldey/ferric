# How to read this book

<!-- section: 0.1; status: edit; registry: ;
     theory: -->

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

Section 10.1 (closed-shell RI-MP2) and section 13.1 (amplitude-threshold
local MP2) show the whole shape. Read one of them first if you want to see
how the parts fit.

## Verified and unverified equations

A working equation sits in a box. The box says what checks it. A verified
box looks like this:

> **Verified equation 0.1** · what a verified box looks like
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

The book never cites a line number, because line numbers move with every
edit. The HTML build resolves each name to a line at the build commit and
links to it. A bare function name in the text, without the path, is not
checked, so the first mention of an item in a section uses the full form.

A fenced Rust block marked as an excerpt occurs in the cited item, with
whitespace ignored. A change to that kernel fails the build until the
excerpt and the text around it are reread. An excerpt may be trimmed; the
text says what was removed, and `// ...` marks each gap.

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

Chapters 1 to 6 are the foundations: second quantization and spin
adaptation (1), molecules, basis sets and the numerical rules (2), integrals
(3), attenuated operators (4), screening (5) and density fitting (6). Every
later chapter uses them. Read chapter 1 and section 6.1 before anything that
writes a working equation; the rest you can look up when a section points
to it.

Most of the book is reached through the self-consistent field, chapters 7
and 8. Three paths through it:

- **Correlation and local correlation.** 7 → 10 → 11 → 12 → 13. Chapter 10
  derives MP2 and its variants on top of Hartree–Fock, chapter 11 builds the
  RI-MP2 algorithm and its memory model, chapter 12 adds orbital
  optimization and the Laplace forms, and chapter 13 truncates the
  canonical calculation of chapter 10 to local pairs.
- **Density functional theory and excited states.** 7 → 9 → 20. Chapter 9
  adds the exchange-correlation functional to the SCF of chapter 7, and
  chapter 20 builds linear response on the converged reference, using the
  $f_{xc}$ kernel of section 9.6.
- **Forces.** 7 → 14 → 15. Chapter 14 differentiates the SCF energy;
  chapter 15 adds the Z-vector that correlated gradients need.

Beyond these: coupled cluster (16) builds on chapter 10, the multireference
chapter (17) on chapters 1 and 8, and RPA (18) on MP2, whose ring
diagrams it sums to infinite order. GW and the
Bethe–Salpeter equation (19) and the polarizabilities (21) build on
chapter 18. Chapters 22 to 25 cover solvation, embedding, periodic systems
and dispersion. Chapter 26 combines short-range MP2 (chapter 10 and
the range-split cross term of section 10.7) with long-range dRPA
(chapter 18). Chapter 27 lists where the computed
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
thread counts can change the last bits of a result, as section 2.3
explains.

## Where the equations live

Each crate keeps its derivations in `theory.md`. The header comment of a
book section names the `theory.md` heading it covers. Chapters 1 to 21
write the derivation in their own prose and keep every fact of that
heading. Chapters 22 to 24 still take the derivation from `theory.md` by
heading, so a change to `theory.md` changes those sections directly.
`theory.md` stays the terse reference, and the verified boxes tie both to
the proof scripts and the code.

The Notation chapter fixes one notation for the whole book. Where it differs
from a `theory.md` file or from the literature, the Notation chapter says so.

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
