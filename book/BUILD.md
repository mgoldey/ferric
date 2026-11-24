# Build proposal

This is a proposal. It adds no build configuration and no CI file. It
describes the output formats, the verified-equation box, the CI job, and the
checks that keep the book from drifting away from the code.

## 1. Recommendation

Write the book once, in Markdown with `$...$` and `$$...$$` math, under
`book/src/`. Build two outputs from that source:

- **HTML with mdBook**, on every pull request that touches the book, a
  generated `theory.md` index, a proof script or the registry. The pull-request gate is this build.
- **PDF with pandoc and LaTeX**, on release tags only. The PDF carries a
  version number and can be cited.

One small preprocessor, `scripts/book_build.py` (proposed), sits in front of
both. It is also the checker. It is plain Python with no dependency outside
the standard library, so the check runs without building any Rust.

## 2. Formats compared

| | Second mdBook | pandoc to LaTeX and PDF | Both from one source |
|---|---|---|---|
| Toolchain already in CI | Yes. `docs-check.yml` and `docs.yml` install mdBook 0.4.40. | No. Needs a LaTeX distribution, about 1 GB, or `tectonic`. | mdBook on pull requests, LaTeX on tags only. |
| Build time | Seconds. | Minutes. | Seconds on pull requests. |
| Search, deep links, links to source | Built in. | Weak. | From the HTML output. |
| Equation numbers and cross-references | Weak. MathJax `\tag` and `\eqref` work inside one page only. | Strong. | The preprocessor numbers equations and rewrites cross-references for both outputs. |
| Wide tables with code names | Scroll horizontally. | Overflow the page. The section tables in the plan need `longtable` with a narrow code column, or the code names moved to notes. | The PDF layout needs a table style of its own. |
| Citable, versioned artifact | No. | Yes. | Yes. |
| Unicode in prose and code (Å, κ, superscripts) | Works. | Needs a font with coverage and `xelatex` or `tectonic`. | One font decision. |
| Risk | Low. | Medium: LaTeX errors surface late. | Low on pull requests, medium on tags. |

A PDF alone would make every pull request pay for LaTeX and would lose the
links from a function name to its source. HTML alone gives no artifact to
cite. Building both costs one preprocessor that is needed anyway for the
boxes and the code references.

**Math delimiters.** The existing site uses mdBook's built-in MathJax, which
needs `\\(` and `\\[` in the source (`site/book.toml:22-25`,
`site/src/methods/dft-derivations.md:13`). The book source uses `$` and
`$$`, which GitHub renders and pandoc reads. For the mdBook output the preprocessor rewrites them to the escaped
MathJax delimiters and protects underscores and backslashes inside math from
the Markdown parser. This keeps the site's reason for avoiding the KaTeX
preprocessor: no extra `cargo install` in CI.

**Directory names.** `site/book/` is the build output of the user guide
(`.gitignore:107`). The top-level `book/` directory holds this book's source.
The book's build output goes to `book/out/`, ignored by git, and is published
beside the user guide under a `book/` path on the documentation site.

## 3. The verified-equation box

A box is a Markdown blockquote followed by one HTML comment. The blockquote
is what a reader sees, on GitHub and in both outputs. The comment is what the
checker reads.

```markdown
> **Verified equation @eq:mp2-cs-spin-adaptation** · spin adaptation of the MP2 energy
>
> The spatial sums ... equal the opposite-spin and same-spin parts ...
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 1 |
> | Negative control | E_SS without the exchange term must differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::pair_energy` |
>
> <!-- verified: id=mp2.cs.spin-adaptation; script=scripts/mp2_variant_identities.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::pair_energy -->
```

Fields of the comment:

| Field | Required | Checked how |
|---|---|---|
| `id` | yes | unique in the book |
| `script` | yes | the file exists |
| `claim` | yes | the script defines that claim (a `claimN` function, a lettered part, or a named check string; each script's header lists them) |
| `runner` | yes | the test file exists and names the script |
| `code` | yes | every `file.rs::item` resolves (section 5) |
| `rust_test` | no | the named test function exists in the named file |
| `registry` | no | the method id is under `methods` in `testdata/derivations/registry.json` |
| `runner_state=pending-merge` | no | allowed only while the runner lives on an unmerged branch; the checker lists these and a release build refuses them |

An equation with no machine check uses the same shape with the heading
**Unverified equation** and the comment `<!-- unverified: id=...; todo=... -->`.
The sample section has one. The checker counts unverified boxes and prints
them in Appendix A. A release build compares the count with a ceiling stored
in the repository, and the ceiling may only go down.

**Rendering.** In HTML the box gets a border and three links: the script at
the build commit, the runner, and each code reference at its resolved line.
In the PDF the box is a framed environment and the links are footnotes with
the path and commit.

**What a box promises.** The script checks the identity the box states. It
does not check that the Rust function computes that identity. The second half
is the job of the Rust test named in `rust_test`, or of the numerical anchor
in the section's anchors table. A box with neither is flagged by the checker
as "identity only".

## 4. The CI job

One job, `book`, proposed for `docs-check.yml` or a workflow of its own.

**Triggers.** Pull requests that touch `book/**`, `crates/*/theory.md` (the generated indexes),
`scripts/*_sympy.py`, `scripts/mp2_variant_identities.py`,
`testdata/derivations/registry.json`, or `scripts/book_build.py`. The
reference check of section 5 also runs in the code CI, because a rename in
`crates/**/src` can break a book reference without touching the book.

**Steps.**

1. Check out. Install Python with `sympy`, `numpy`, `scipy`, `mpmath`.
2. **Run the proofs.** Run the derivation runner and the registry test:
   `test_derivation_scripts.py` and `test_derivation_registry.py` under
   `crates/ferric-python/tests/` (both on `docs/derivations-backfill`).
   Two properties of the current test layout would make this step pass
   without running anything, and the job must close both:
   - `crates/ferric-python/tests/conftest.py:44-59` ignores every test file
     when the `ferric` extension cannot be imported. The book job builds no
     Rust, so it must run these two files with `--noconftest`. They import
     only `pytest` and the standard library.
   - The runner starts with `pytest.importorskip("sympy")`. The job must
     fail when any test is skipped, so that a missing dependency is red.
3. **Check the book.** `python3 scripts/book_build.py --check` (section 5).
4. **Build the HTML.** `mdbook build` on the preprocessed tree.
5. **Check links.** `scripts/check_doc_links.py` on the output, as the
   user-guide job does.

A red proof stops the job at step 2, before the book is built. A box whose
script is not in the runner's case list fails at step 3.

**Slow proofs.** The runner marks scripts with a runtime budget at or
above its slow threshold (600 s on `docs/derivations-backfill`) as
`validation`, the weekly tier. Pull requests run the fast tier. The job that
publishes the book runs every proof, fast and slow, in the same job before it
deploys. Publishing therefore never rests on a weekly result from another
commit.

**PDF.** A separate job on release tags runs steps 1 to 3, then pandoc. It
refuses boxes marked `runner_state=pending-merge` and enforces the ceiling on
unverified boxes.

## 5. Keeping the book in step with the code

The checker enforces the following. Each rule exists because of a drift that
is already visible in the repository.

1. **Code references are names, not line numbers.** The book writes
   `` `ferric-mp2/src/rimp2.rs::pair_energy` `` or a braced list. The checker
   requires the file to exist under `crates/` and the item to be declared in
   it (`fn`, `struct`, `enum`, `trait`, `const`, `static` or `type`), or in a
   submodule directory of the same name. The build resolves the line and
   links to it at the build commit. Reason: a line reference drifts with every
   edit to the file it points into. A `file.rs:123` pin into the repository
   (any `crates/`, `scripts/` or `site/` file) is an error; a line inside a
   function that has to be shown is quoted in a marked excerpt instead.
   Documents outside the repository that do not change, such as the
   dissertation text, may still be cited by line.
2. **Code excerpts must match the source.** A fenced block preceded by
   `<!-- excerpt: file.rs::item -->` (Rust) or
   `<!-- excerpt: scripts/x.py::function -->` (Python) must occur inside that
   item. Comments are removed on both sides before the comparison and
   whitespace is normalized, so an excerpt carries the per-line comments that
   STYLE.md asks for. A `// ...` line (`# ...` in Python) marks a gap: the
   pieces between gaps must each occur, in order. Any other omission has to be
   marked with a gap line, or the check fails. A change to a kernel then fails
   the book until the excerpt and the text around it are reread.
3. **Sections name their registry ids.** A section header comment lists the
   registry method ids it covers and its status. The checker fails when a
   `ready` section covers an id that the registry lists under `gaps`, when a
   registry id is covered by no section, and when the status table in
   `book/PLAN.md` differs from the count of section statuses.
4. **The book owns the derivations.** Each `crates/<crate>/theory.md` is
   generated by `--write-theory-index`: a pointer to the book and a table of
   the pages that cite the crate's code, with their proof scripts and verified
   boxes. `--check` fails when an index differs from what the generator
   writes, when a crate's index lists no page, and when a section header
   carries a `theory:` field. An include, if one is ever used, names its file
   and heading text, never a number.
5. **Quoted numbers come from one place.** A figure in an anchors table
   carries a comment naming its source row in
   `site/src/reference/validation.md` or its entry in
   `crates/ferric-core/src/certificate/anchors.rs`. The checker requires the
   quoted digits to occur in that source.
6. **Paths resolve.** Every repository path in backticks must exist. This
   file is exempt, because it names files that are only proposed.
7. **Local-only sources stay local.** The dissertation and the wiki are not
   in the repository. The checker rejects any link to them. When
   `dissertation.txt` is present on the machine, it also reports any book
   paragraph that reproduces a long run of it, so that quotation stays
   short and attributed.
8. **The build is stamped.** Every HTML page footer and the PDF title page
   carry the commit the references were resolved against.
9. **Style report.** `scripts/check_doc_style.py book/src` runs as a report,
   as it does for the user guide.

Rules 1, 3 and 6 were prototyped while writing the plan. The prototype
resolved 362 distinct code references in 185 source files across the four
files in `book/`, with none unresolved, and confirmed that all 64 registry
method ids appear in the plan. It was checked against a broken
reference, a missing file, a missing script and a removed registry id, and
reported each. It is not committed.

## 6. Source layout (proposed)

```
book/
  PLAN.md            this plan; its status table is recomputed by the checker
  BUILD.md           this file
  src/
    SUMMARY.md       generated from the chapter order
    00-notation.md   from book/notation.md
    10-mp2/
      01-closed-shell-ri-mp2.md   from book/sample-ri-mp2.md
      ...
  out/               build output, ignored by git
scripts/
  book_build.py      preprocessor and checker
```

Each section file starts with a header comment:

```markdown
<!-- status: ready; registry: ri-mp2 -->
```

## 7. What this proposal does not settle

- The PDF page design: fonts, table style, box style.
- Where the published HTML lives relative to the user guide, and whether the
  book is public before a first complete release.

## 8. Building locally

Needs Python 3 and, for HTML, `mdbook` 0.4.40 on `PATH`. No Rust build.

| Goal | Command |
|---|---|
| Check the book against the code | `python3 scripts/book_build.py --check` |
| The strict release check (ceiling, registry coverage, no `pending-merge`) | `python3 scripts/book_build.py --check --release` |
| Regenerate `book/src/SUMMARY.md` after adding or moving a section | `python3 scripts/book_build.py --write-summary` |
| Regenerate appendices A, B and C (committed; `--check` fails when stale) | `python3 scripts/book_build.py --write-appendices` |
| Sync the section statuses and counts in `book/PLAN.md` with the section headers | `python3 scripts/book_build.py --write-plan` |
| Regenerate the per-crate indexes `crates/*/theory.md` (committed; `--check` fails when stale) | `python3 scripts/book_build.py --write-theory-index` |
| Build the HTML to `book/out/html` | `python3 scripts/book_build.py --build` |
| One Markdown file for pandoc | `python3 scripts/book_build.py --concat /tmp/book.md` |
| Run the proofs and the checker's own tests | `python3 -m pytest --noconftest crates/ferric-python/tests/test_derivation_scripts.py crates/ferric-python/tests/test_book_build.py` |

Conventions the checker enforces:

- Sections are discovered by the glob `book/src/NN-slug/MM-name.md`; nothing lists them.
  The header comment may follow the `#` title line, as in the sample.
- `book/src/NN-slug/index.md` (optional) supplies the chapter title.
- `<!-- include: file.md#"Heading text" -->` pulls in the body under that heading of another
  Markdown file; the book uses none today, and a derivation is never included from `theory.md`.
- `book/unverified-ceiling.txt` holds the largest allowed number of unverified boxes. Exceeding
  it is a warning in `--check` and an error in `--release`; CI also fails a pull request that raises
  the value above its value on the base branch.
- `book/book.toml` reads the preprocessed tree `book/out/src`, so run `--build`, not bare `mdbook build`.

CI is `.github/workflows/book.yml`. Not yet implemented: the box links to source lines in the HTML,
the quoted-number check (rule 5) and the `check_doc_style.py` report.

On a pull request the job runs only the proofs the change can affect. `scripts/select_proofs.py`
reads the changed files and prints `all`, `none` or a list of proof scripts, and the runner
reads that list from `FERRIC_PROOF_SCRIPTS`:

- `all` when a runner (`test_derivation_*.py`, `test_book_build.py`), `book_build.py`, the
  selector, `book.yml`, the registry's shape, or any other file under `scripts/` or
  `testdata/derivations/` changed, or when the selector cannot tell.
- The changed proof scripts, plus the scripts of each registry entry that changed, plus each
  script that names a changed file it reads (a Rust source or a reference JSON).
- `none` for book prose and `crates/*/theory.md`. The step then logs "no proof inputs changed".

The registry test, the checker's tests and the selector's tests run every time; together they take
about 15 s. The nightly schedule (07:23 UTC), tag pushes and `workflow_dispatch` run every proof.
"Every proof" means the runner's fast tier. The `validation` tier is deselected by `addopts` in
`pyproject.toml`, as it was before. To run one script's proofs locally:
`FERRIC_PROOF_SCRIPTS=omega_ri_sympy.py python3 -m pytest --noconftest crates/ferric-python/tests/test_derivation_scripts.py`.

Appendices live beside `SUMMARY.md` as `book/src/appendix-X-name.md`. A (verified equations and
proof scripts), B (functions cited) and C (validation grades) are generated and committed; D is
written by hand. Each carries a `<!-- status: ... -->` header and a labelled title
(`# Appendix D: ... {#appendix-d}`), checked like a section's. The status of every page in
`book/PLAN.md` section 3, whose rows are keyed by the page's title label (`` `closed-shell-ri-mp2` ``),
must equal its header's status, and the table in section 4 is the recount between its
`status-table` markers.

### Numbers: label, reference by label, never write a number

No chapter, section or equation number is written in `book/src`. The preprocessor derives each
one from position, so inserting, moving or deleting a section or an equation renumbers everything
after it, and every reference follows.

| Thing | Source | Rendered |
|---|---|---|
| Chapter, section, subsection | `## The energy, and why it splits in two {#closed-shell-ri-mp2-energy-splits}` | `## 10.1.1 The energy, ...` |
| An unnumbered heading | `## Sources` (no label) | as written; it does not advance the count |
| Display equation | `$$ ... \label{eq:closed-shell-ri-mp2-3} $$` | `\tag{10.3}`, with an anchor in front |
| A box that states its own equation | `> **Verified equation {#eq:scf-rij-packed}** · ...` | `Verified equation 7.4` |
| A box that certifies a labelled equation | `> **Verified equation @eq:mp2-cs-spin-adaptation** · ...` | that equation's number |
| Reference | `(@eq:closed-shell-ri-mp2-3)`, `section @sec:fxc`, `chapter @sec:rpa` | the bare number, linked (plain in `--concat`) |
| Reference before a letter or digit | `{@eq:mp2-u-grad-assembly}b` | `15.21b` |

Chapters are numbered by the position of their directory among `book/src/NN-*` (the `NN` prefix
orders them; its value is not the number), sections by their place in the chapter, and equations
and numbered boxes share one series per chapter in page order. A label is lowercase kebab-case,
unique across the book; a reference names its kind (`@sec:` for any heading, `@eq:` for an
equation or a box). `--check` fails on a duplicate label, an undefined reference or one of the
wrong kind, a `\tag{`, a heading that writes its own number, a `section:` header field, and a
hard-coded section, chapter or equation number in prose. A number that belongs to another work is
not a reference: a clause (or table cell and its column header) that names the dissertation, a
paper, or a table or figure of one is exempt, and `book/number-allowlist.txt`
lists the few remaining exceptions by page. `scripts/book_autonumber_migrate.py` is the one-shot
migration that derived the labels (2026-11-17); it is not part of the build.
