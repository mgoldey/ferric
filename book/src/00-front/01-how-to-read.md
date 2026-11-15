# 0.1 How to read this book

<!-- section: 0.1; status: edit; registry: ;
     theory: -->

This book explains how each equation in ferric becomes code. Start from any
equation, find the function that contracts it, and find the script and the
test that check it. The audience is someone who implements or audits an
electronic-structure method. It does not teach the methods from scratch.

The book describes the code as it is now. Where a limit is a current fact, a
section ends with a short, dated "Known limits" list. It does not keep a
history of what was tried and withdrawn.

## What a section contains

Every section follows one template, in this order. A section omits an item
only when it has nothing to say there.

| # | Item | What it gives you |
|---|---|---|
| 1 | Header tables | The index ranges the section uses, and the entry points in Rust, the CLI and Python |
| 2 | Equations | Each working equation, in a verified or an unverified box |
| 3 | The algorithm | A table of steps with cost, memory and the function for each step |
| 4 | A kernel excerpt | The innermost loop, with a table from symbols to variable names |
| 5 | Reproducibility and memory | Rules that affect a result, such as summation order and the memory budget |
| 6 | Options | The settings that change the equation |
| 7 | Numerical anchors | Agreement figures with an independent reference, each traced to a test or to the validation page |
| 8 | Sources | Where each equation came from, and what the section changed |

## Verified and unverified equations

A working equation sits in a box. The box says what checks it.

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

A **verified** box names a proof script under `scripts/`, a claim inside it,
and the test that runs the script. The build refuses a box whose script,
claim or runner does not exist. The claim checks the identity the box
states. It does not check that the Rust function computes that identity. The
second half belongs to a Rust test (named in the box when there is one) or to
a numerical anchor. A box with neither is an identity-only box, and the build
says so.

An **unverified** box has the same shape and the heading "Unverified
equation". It states what proof is missing and where to add it. Treat such an
equation as derived by hand and read against the code, not as machine
checked. The count of unverified boxes appears in Appendix A and may only
fall between releases.

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

Code is cited as a path under `crates/`, then `::`, then the item name, for
example `ferric-mp2/src/rimp2.rs::pair_energy`. A cited item is declared in that file.
The book never cites a line number, because line numbers drift with every
edit. The HTML build resolves each name to a line at the build commit and
links to it. A braced list after the `::` cites several items in one file.

A fenced Rust block marked as an excerpt occurs in the cited item, with
whitespace ignored. A change to that kernel fails the build until the
excerpt and the text around it are reread.

## Numbers and grades

A figure in an anchors table comes from the validation page
(`site/src/reference/validation.md`) or from a committed test, and the
section names which. The book contains no number from an unrecorded run.

The validation page grades every CLI `method.kind`. The grades are the ones
ferric prints at run time.

| Grade | Meaning |
|---|---|
| Proven | Agreement at a stated tolerance with an independent reference, pinned by tests. An independent reference is another code, a numpy reference, a published value, or an exact limit the method must reduce to. |
| Proven (narrow) | As Proven, but only on the stated class of systems, or against the stated kind of reference. |
| Smoke | Runs end to end and some pieces are checked, but there is no independent reference for the headline number, or only a loose one. Do not quote such a result as a reference value. |
| Spike | Built on new infrastructure and not compared with any reference code. For exploration only. |
| not graded | Dispatched with no recorded grade. Treat it as unproven. |

Implemented is not validated. A chapter that describes a method states its
grade and refers to the validation page for the evidence. The grades are the
`METHOD_KIND_GRADES` table in `crates/ferric-core/src/certificate.rs`, kept
equal to the validation page by a test.

## Where the equations live

Each crate keeps its derivations in `theory.md`. A section of this book
supplies the reading order, the shared notation, the algorithm and the
connecting text, and takes the derivation from `theory.md` by heading. A
change to a method's working equations therefore changes the book when it
changes `theory.md`.

The Notation chapter fixes one notation for the whole book. Where it differs
from a `theory.md` file or from the literature, the Notation chapter says so.

## Scope

| In the book | In the user guide (`site/`) |
|---|---|
| The equations, the algorithms and the proofs behind the methods | How to run a method, its keywords, measured timings, the validation matrix |
| Memory and determinism rules that change a result | Installation, the Python and Avogadro surfaces, ASE and QCSchema |

The semiempirical interface is a one-page appendix. The user guide documents
every CLI key and Python keyword.

## Authorship and licence

Matthew Goldey, with ferric contributors. The book source is licensed as
ferric is, under MIT OR Apache-2.0. Method definitions taken from the
author's dissertation are cited to it. Its benchmark tables and figures are
not reproduced; any figure in the book comes from a ferric run.
