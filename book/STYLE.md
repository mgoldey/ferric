# Book style guide

This guide covers the prose of `book/src`. The model sections are
closed-shell RI-MP2 (`10-mp2/01-closed-shell-ri-mp2.md`) and the first two
subsections of amplitude-threshold local MP2
(`13-local-mp2/01-amplitude-lmp2.md`).
Read one of them before you edit a chapter.

The target reader knows quantum chemistry at the level of a first graduate
course and can read Rust and Python. They want to know three things: why a
method works, how the equations follow from each other, and what the code
does line by line. Write so that a reader with a pencil can follow every
step, and a reader with the source open can match every line.

## Voice

Write conversational technical prose, the way a good lecturer explains a
derivation at the board.

- Short declarative sentences. One idea per sentence.
- Address the reader as "you" when they run, port or change the code ("if
  you port this loop"). Use "we" sparingly, inside a derivation ("we sum
  over spin").
- A question may open a paragraph if the next sentence answers it ("Why is
  $E_{\mathrm{SS}}$ smaller? The minus sign in ... is the Pauli principle").
- Name the operation at each step: expand, substitute, use $\mathcal A t^* =
  -J$, move $\mathcal A$ to the left. The reader should never wonder how one
  line became the next.
- State facts plainly. Replace a figure of speech with the fact it points
  at.

### Banned

These read as filler. Do not use them or close variants.

| Pattern | Examples | Write instead |
|---|---|---|
| Idioms and aphorisms | "the exception proves the rule", "X or not at all", "pays off many times", "the whole reason", "keep that in mind" | the concrete fact: "diverges as $1/\Delta$, see (@eq:closed-shell-ri-mp2-3)" |
| Setup and punchline | "There's a bonus.", "Here's the trick.", "Be honest about what this buys." | state the point: "Cholesky also checks the input" |
| Reassurance and teasers | "That isn't a workaround.", "it comes back later", "the cheapest honest account" | cut, or give the cross-reference |
| Rhetorical colour | "the one thing computers are reliably fast at", "to the last digit you'd trust" | "near peak FLOP rate", "up to rounding" |
| Announcing attention | "worth noticing", "note that", "importantly", "it turns out" | start with the point |
| Intensifiers and house jargon | genuinely, truly, really, simply, just, crucially; load-bearing, lane, footgun, the trap | delete, or name the mechanism |
| Contrast framing | "not X but Y" as a device | say what it is. "X, not Y" is allowed once when Y is the likely wrong guess |
| Em-dash chains, ALL CAPS, scattered bold | | colons and periods; bold only for run-in labels |
| Revision history | "originally", "we used to", "was fixed" | current facts only |

The CONTRIBUTING.md "Writing documentation" rules apply except where this
guide differs. The book allows "you", "we", questions answered at once, and
a one-line plan at the top of a section. The site docs do not.

## Section shape

Not every section needs every part. The order is fixed.

1. **The problem, physically.** One or two paragraphs. What goes wrong
   without this method, or what it computes, in terms a chemist would use
   (electrons avoiding each other, an orbital sitting on a bond, an integral
   that is nearly zero). No equations yet.
2. **The plan.** Only for a section with three or more subsections: one
   sentence or a short numbered list. Shorter sections go straight on.
   Keep the motivation to one paragraph of at most five sentences.
3. **Notation and entry points.** The existing tables stay; trim the prose
   around them.
4. **Derivation.** Every step written out. After each key result, check a
   limit you can do in your head: the canonical limit, $\varepsilon = 0$,
   one atom, one pair, $\omega \to 0$. Intermediate lines are untagged
   display math.
5. **What it means.** One paragraph of interpretation after the important
   equation. Give a number from a runnable example where one exists.
6. **The code.** An excerpt of the real function (see "Code excerpts"),
   then a short list of points about it: sign conventions, data layout,
   what each loop is the unit of, and why the code differs from the
   textbook form.
7. **Design choices.** Why the code does X instead of the obvious Y, with
   the cost or accuracy reason.
8. **What is proven and what is not.** The verified-equation boxes, and one
   plain sentence on any step the proof does not cover.

## Content invariants

A rewrite changes the explanation, never the facts.

- **Keep every number, unit, limit, error condition, threshold and
  constant name.** You may move them; you may not drop or round them.
- **Verified-equation boxes stay byte-identical**, including the
  `<!-- verified: ... -->` comment. Unverified-equation boxes stay as they
  are. You may move a box to sit after the derivation it certifies.
- **Label equations, reference by label; never write a number.** A display
  equation that anything refers to carries `\label{eq:name}`; a reference is
  `(@eq:name)`, `section @sec:name` or `chapter @sec:name`. The build writes
  the numbers, so moving or adding an equation or a section is safe. Keep
  existing labels; do not label new algebra that nothing refers to. See
  `book/BUILD.md`, "Numbers: label, reference by label, never write a number".
- **Headings keep their labels** (`## Title {#name}`). You may reword the
  title text; the label stays, because references and `book/PLAN.md` use it.
  A heading without a label is not numbered.
- **Code references stay in the checked form** `crate/src/file.rs::item`.
  A bare function name is not checked by `book_build.py`. The first mention
  of an item in a section uses the checked form.
- **The book owns its text.** The derivations live here and nowhere else.
  Each `crates/<crate>/theory.md` is a generated index of the book pages
  that cite the crate's code (`scripts/book_build.py --write-theory-index`);
  never write content into it. The verified boxes tie the book to the sympy
  proofs and the code.
- **The section header comment stays**, including `status:`. Do not change
  the status. Reviewers set it.
- **Every new claim is either in the cited sympy script or checked by
  you.** If you add a derivation step that is not in those
  sources, verify it (a few lines of sympy or numpy in your scratch
  directory) before you write it. If you cannot verify it, leave it out.

## Code excerpts

- Copy from the current source. Do not paraphrase code.
- Trim freely, and say what you trimmed: "with the κ-regularization branch
  removed", "only the flop counter has been removed". Mark elisions with
  `// ...`.
- Comment each line with the term of the equation it computes
  (`// F t_ij (particle a)`). Comments explain the mathematics. Do not
  restate the code.
- Keep excerpts under about 20 lines. Show the core loop, not the setup.

## Runnable examples

- Python or TOML that a reader can paste. Use bundled molecules and basis
  sets (`testdata/molecules/water.xyz`, `cc-pvdz`).
- Every printed number comes from running the snippet on the current build.
  Check `ferric.__build__` first, and name the commit in the text ("from the
  `a410529d0` build"). If you did not run it, do not show output.
- Run snippets under `box-lease exec --` with `OPENBLAS_NUM_THREADS=1`.
  Keep them small: water or methane, a double-zeta basis, seconds of run
  time.

## Checks before you commit

```bash
python3 scripts/book_build.py --check                  # must pass
python3 scripts/check_doc_style.py --verbose <files>   # reread every hit
grep -nE "exception proves|bonus|honest|pays off|worth noticing|whole reason|keep that in mind|or not at all|it turns out|note that|genuinely|simply|just " <files>
```

`book_build.py --check` can fail only because `book/PLAN.md` or a generated
appendix is stale. In that case, regenerate them so the check passes
(`--write-plan`, `--write-appendices`), but do not commit them. Integration
regenerates them once for the batch.
