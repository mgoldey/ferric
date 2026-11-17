"""scripts/book_build.py: the book's preprocessor and checker (book/BUILD.md 3-5).

Builds a tiny valid book tree in a temp directory, checks it passes, then seeds
one error of each kind and requires the checker to fail on it (mutation tests).
Standard library and pytest only; no ferric extension is needed.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "book_build.py"
spec = importlib.util.spec_from_file_location("book_build", SCRIPT)
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)

SECTION = """# 10.1 Toy section

<!-- section: 10.1; status: ready; registry: toy;
     theory: crates/ferric-toy/theory.md#"1. Toy theory" -->

Entry point `ferric-toy/src/lib.rs::pair_energy`, also
`ferric-toy/src/lib.rs::{pair_energy, other_fn}` and `scripts/toy_sympy.py`.

<!-- include: crates/ferric-toy/theory.md#"1. Toy theory" -->

> **Verified equation 10.1** . toy identity
>
> <!-- verified: id=toy.a; script=scripts/toy_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-toy/src/lib.rs::pair_energy; rust_test=ferric-toy/src/lib.rs::toy_test -->

> **Unverified equation 10.2** . toy
>
> <!-- unverified: id=toy.b; todo=add a claim -->

<!-- excerpt: ferric-toy/src/lib.rs::pair_energy -->
```rust
let e = a * b;
```

Inline math $E_{ij} = |a|^2$ and a link [theory](../../crates/ferric-toy/theory.md).
"""

VALIDATION = """# Capabilities and validation

## Grades

| Grade | Meaning |
|---|---|
| **Proven** | Agrees with an independent reference. |
| Smoke | Runs end to end. |

## CLI `method.kind` matrix

| `method.kind` | Family | Reference (CLI) | Python | Grade | Caveat |
|---|---|---|---|---|---|
| `toyscf` | [SCF](../methods/scf.md) | RHF | `run_toy` | [Proven](#anchors) | - |
| `toymp2` | MP2 | RHF | `run_toy2` | Smoke | a \\| b |

## Anchors
"""

FILES = {
    "scripts/book_build.py": "# the generator\n",
    "site/src/reference/validation.md": VALIDATION,
    "testdata/derivations/registry.json": json.dumps(
        {"methods": {"toy": {}, "toy2": {}}, "gaps": {"holey": {"reason": "no proof"}}}
    ),
    "scripts/toy_sympy.py": '"""Claims:\n  1  toy identity\n"""\n',
    "crates/ferric-python/tests/test_derivation_scripts.py": 'CASES = ["toy_sympy.py"]\n',
    "crates/ferric-toy/theory.md": "# Toy\n\n## 1. Toy theory\n\nBody $x_1$.\n\n## 2. Next\n\nOther.\n",
    "crates/ferric-toy/src/lib.rs": (
        "pub fn pair_energy(a: f64, b: f64) -> f64 {\n    let e = a * b;\n    e\n}\n"
        "fn other_fn() {}\n#[test]\nfn toy_test() {}\n"
    ),
    "crates/ferric-toy/src/sub/deep.rs": "pub fn in_submodule() {}\n",
    "book/src/00-notation.md": "# Notation\n\nText.\n",
    "book/src/10-mp2/index.md": "# 10. Toy chapter\n\nText.\n",
    "book/src/10-mp2/01-toy.md": SECTION,
    "book/unverified-ceiling.txt": "1\n",
}


def write(root: Path, files: dict) -> None:
    for name, text in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


@pytest.fixture
def tree(tmp_path):
    bb._src_cache.clear()
    write(tmp_path, FILES)
    (tmp_path / "crates/ferric-toy/src/sub.rs").write_text("")
    bb.write_appendices(tmp_path)
    (tmp_path / "book/src/SUMMARY.md").write_text(bb.build_summary(tmp_path))
    return tmp_path


def check(root, regen=True, **kw):
    """Run the checker. The generated appendices are rewritten first so a mutation
    is judged by the check it targets and not by the appendices going stale; the
    appendix tests pass regen=False."""
    bb._src_cache.clear()
    if regen:
        bb.write_appendices(root)
    return bb.run_check(root, **kw)[0]


def resummarize(root):
    (root / "book/src/SUMMARY.md").write_text(bb.build_summary(root))


def sub(root, name, old, new):
    p = root / name
    text = p.read_text()
    assert old in text, f"mutation target {old!r} not in {name}"
    p.write_text(text.replace(old, new))


SEC = "book/src/10-mp2/01-toy.md"


def test_valid_tree_passes(tree):
    rep = check(tree)
    assert rep.errors == []


def test_unseen_chapter_is_accepted_and_listed(tree):
    write(
        tree,
        {
            "book/src/42-new/index.md": "# 42. New\n",
            "book/src/42-new/03-thing.md": "# 42.3 Thing\n\n<!-- section: 42.3; status: edit; registry: toy2 -->\n",
        },
    )
    (tree / "book/src/SUMMARY.md").write_text(bb.build_summary(tree))
    assert check(tree).errors == []
    assert "42-new/03-thing.md" in (tree / "book/src/SUMMARY.md").read_text()


def test_chapter_without_index_is_an_error(tree):
    (tree / "book/src/10-mp2/index.md").unlink()
    resummarize(tree)
    assert any("no index.md" in e for e in check(tree).errors)


def test_submodule_directory_resolves(tree):
    sub(
        tree,
        SEC,
        "`scripts/toy_sympy.py`.",
        "`scripts/toy_sympy.py` and `ferric-toy/src/sub.rs::in_submodule`.",
    )
    (tree / "crates/ferric-toy/src/sub.rs").write_text("")
    assert check(tree).errors == []


MUTATIONS = {
    "script missing": (
        SEC,
        "script=scripts/toy_sympy.py",
        "script=scripts/nope_sympy.py",
    ),
    "claim undefined": (SEC, "claim=1;", "claim=9;"),
    "runner missing": (
        SEC,
        "runner=crates/ferric-python/tests/test_derivation_scripts.py",
        "runner=crates/ferric-python/tests/test_nope.py",
    ),
    "runner does not name script": (
        "crates/ferric-python/tests/test_derivation_scripts.py",
        "toy_sympy.py",
        "other.py",
    ),
    "code item undeclared": (
        SEC,
        "code=ferric-toy/src/lib.rs::pair_energy",
        "code=ferric-toy/src/lib.rs::missing_fn",
    ),
    "code file missing": (
        SEC,
        "code=ferric-toy/src/lib.rs::pair_energy",
        "code=ferric-toy/src/gone.rs::pair_energy",
    ),
    "braced list item undeclared": (
        SEC,
        "{pair_energy, other_fn}",
        "{pair_energy, nope_fn}",
    ),
    "prose code ref undeclared": (
        SEC,
        "Entry point `ferric-toy/src/lib.rs::pair_energy`",
        "Entry point `ferric-toy/src/lib.rs::renamed`",
    ),
    "rust_test missing": (
        SEC,
        "rust_test=ferric-toy/src/lib.rs::toy_test",
        "rust_test=ferric-toy/src/lib.rs::no_test",
    ),
    "box registry unknown": (
        SEC,
        "code=ferric-toy/src/lib.rs::pair_energy;",
        "code=ferric-toy/src/lib.rs::pair_energy; registry=nonesuch;",
    ),
    "duplicate box id": (SEC, "id=toy.b", "id=toy.a"),
    "unverified without todo": (SEC, "; todo=add a claim", ""),
    "header registry unknown": (SEC, "registry: toy;", "registry: nonesuch;"),
    "ready covers gap": (SEC, "registry: toy;", "registry: holey;"),
    "bad status": (SEC, "status: ready", "status: done"),
    "section number mismatch": (SEC, "section: 10.1", "section: 10.2"),
    "missing header": (
        SEC,
        '<!-- section: 10.1; status: ready; registry: toy;\n     theory: crates/ferric-toy/theory.md#"1. Toy theory" -->',
        "",
    ),
    "theory heading missing": (
        SEC,
        'theory.md#"1. Toy theory" -->\n\nEntry',
        'theory.md#"9. Nonexistent" -->\n\nEntry',
    ),
    "include heading missing": (
        SEC,
        '<!-- include: crates/ferric-toy/theory.md#"1. Toy theory" -->',
        '<!-- include: crates/ferric-toy/theory.md#"No such heading" -->',
    ),
    "theory file missing": (
        SEC,
        'crates/ferric-toy/theory.md#"1. Toy theory" -->\n\nEntry',
        'crates/ferric-gone/theory.md#"1. Toy theory" -->\n\nEntry',
    ),
    "excerpt differs": (SEC, "let e = a * b;", "let e = a + b;"),
    "path does not exist": (SEC, "`scripts/toy_sympy.py`", "`scripts/nope_sympy.py`"),
    "link to local-only source": (
        SEC,
        "../../crates/ferric-toy/theory.md",
        "../../wiki/notes.md",
    ),
    "section title carries another number": (
        SEC,
        "# 10.1 Toy section",
        "# 11.1 Toy section",
    ),
    "section title has no number": (SEC, "# 10.1 Toy section", "# Toy section"),
    "chapter title has no number": (
        "book/src/10-mp2/index.md",
        "# 10. Toy chapter",
        "# Toy chapter",
    ),
    "stale summary": ("book/src/SUMMARY.md", "Toy section", "Renamed section"),
    "control character from a lost backslash": (
        SEC,
        "Inline math",
        "Inline\x08eta math",
    ),
    "lone carriage return from a lost \\rm": (
        SEC,
        "Inline math",
        "Inline $E_{\rm OS}$ math",
    ),
    "code span inside inline math": (
        SEC,
        "$E_{ij} = |a|^2$",
        "$E_{ij} = x \\quad (`build_s_d`)$",
    ),
    "code span inside display math": (
        SEC,
        "Inline math",
        "$$ f = \\frac{a}{b} \\quad (`build_k_r`) $$\n\nInline math",
    ),
    "code span inside math from an included theory section": (
        "crates/ferric-toy/theory.md",
        "Body $x_1$.",
        "Body $x_1 (`a_b`)$.",
    ),
    "unknown registry file": (
        "testdata/derivations/registry.json",
        '"toy": {}',
        '"toyx": {}',
    ),
}


def test_crlf_line_endings_are_not_control_characters(tree):
    # A Windows checkout, or a page written in text mode there, ends lines
    # with CRLF; only a lone carriage return is a lost "\r".
    p = tree / SEC
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert b"\r\n" in p.read_bytes()
    assert check(tree).errors == []


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mutation_fails(tree, name):
    assert check(tree).errors == []
    sub(tree, *MUTATIONS[name])
    assert check(tree).errors, f"checker accepted: {name}"


def test_mutation_stale_summary_new_file(tree):
    write(
        tree,
        {
            "book/src/11-x/01-y.md": "# 11.1 Y\n\n<!-- section: 11.1; status: edit; registry: - -->\n"
        },
    )
    assert any("SUMMARY" in e for e in check(tree).errors)


def test_release_ceiling_and_coverage(tree):
    rep = check(tree, release=True)
    assert any("covered by no section" in e for e in rep.errors)  # toy2 uncovered
    assert not any("covered by no section" in w for w in check(tree).errors)
    sub(tree, SEC, "registry: toy;", "registry: toy, toy2;")
    assert check(tree, release=True).errors == []
    (tree / "book/unverified-ceiling.txt").write_text("0\n")
    assert any("ceiling" in e for e in check(tree, release=True).errors)
    assert check(tree).errors == []  # a warning outside release builds


def test_release_refuses_pending_merge_runner(tree):
    sub(tree, SEC, "claim=1;", "claim=1; runner_state=pending-merge;")
    sub(
        tree,
        SEC,
        "runner=crates/ferric-python/tests/test_derivation_scripts.py",
        "runner=crates/ferric-python/tests/test_unmerged.py",
    )
    sub(tree, SEC, "registry: toy;", "registry: toy, toy2;")
    assert check(tree).errors == []
    assert any("pending-merge" in e for e in check(tree, release=True).errors)


def test_ceiling_may_only_go_down(tree):
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": __import__("os").environ["PATH"],
    }
    for cmd in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "commit.gpgsign=false", "commit", "-qm", "base"],
        ["tag", "base"],
    ):
        subprocess.run(["git", "-C", str(tree), *cmd], check=True, env=env)
    assert check(tree, ceiling_base="base").errors == []
    (tree / "book/unverified-ceiling.txt").write_text("5\n")
    assert any("only go down" in e for e in check(tree, ceiling_base="base").errors)
    (tree / "book/unverified-ceiling.txt").write_text("0\n")
    assert check(tree, ceiling_base="base").errors == []


def test_math_rewrite_protects_markdown_metacharacters():
    out = bb.rewrite_math(
        r"a $x_i^*\{y\}$ b $$\frac{a}{b}$$ and `$code$` and $5 and $6"
    )
    assert r"\\(x\_i^\*\\\\{y\\\\}\\)" in out or r"\\(x\_i^\*" in out
    assert r"\\[\\frac{a}{b}\\]" in out
    assert "`$code$`" in out
    assert "$5 and $6" in out


def test_include_expands_section_body(tree):
    text = bb.expand_includes(tree, (tree / SEC).read_text())
    assert "Body $x_1$." in text and "Other." not in text


def test_notes_flag_identity_only_boxes(tree):
    sub(tree, SEC, "; rust_test=ferric-toy/src/lib.rs::toy_test", "")
    assert any("identity-only" in n for n in check(tree).notes)


LABELLED = (
    '"""Claims:\n  1  toy identity\n"""\n'
    'check("B0 model: a long printed name (claim B0)", ok)\n'
    'check("B01 other check (claim B01)", ok)\n'
)


def test_claim_label_is_matched_exactly(tree):
    """claim=B0 resolves to a check labelled "(claim B0)", never to B01 or to prose."""
    write(tree, {"scripts/toy_sympy.py": LABELLED})
    sub(tree, SEC, "claim=1;", "claim=B0;")
    assert check(tree).errors == []
    sub(tree, SEC, "claim=B0;", "claim=B02;")
    assert any("defines no claim B02" in e for e in check(tree).errors)


def test_claim_label_needs_the_label_not_the_printed_name(tree):
    """The label form does not loosen matching: a bare prefix of the printed
    name, or the name without its label, is not a claim."""
    write(
        tree,
        {"scripts/toy_sympy.py": '"""x"""\ncheck("B0 model: spin-orbital Fock", ok)\n'},
    )
    sub(tree, SEC, "claim=1;", "claim=B0;")
    assert any("defines no claim B0" in e for e in check(tree).errors)
    sub(tree, SEC, "claim=B0;", "claim=B0 model: spin-orbital Fock;")
    assert check(tree).errors == []  # the full quoted string still resolves, as before


def test_every_cited_claim_in_the_real_book_resolves_by_label():
    """The book's own oomp2 and hf_energy boxes cite labels, not quoted strings."""
    root = Path(__file__).resolve().parents[3]
    import re

    cited = set()
    for page in (root / "book/src").rglob("*.md"):
        for m in re.finditer(
            r"script=(scripts/(?:oomp2_mp3_identities|hf_energy)_sympy\.py);\s*claim=([^;]+);",
            page.read_text(),
        ):
            cited.add(m.groups())
    assert cited, "no boxes cite the oomp2 or hf_energy scripts"
    for script, claim in cited:
        assert len(claim) <= 3, (
            f"{script}: claim {claim!r} is a quoted string, not a label"
        )
        assert bb.claim_defined((root / script).read_text(), claim)


FIG = "book/src/figures/10-mp2/toy.svg"


def test_build_tree_copies_figures_and_keeps_them_out_of_the_chapter_glob(tree):
    write(
        tree,
        {
            FIG: "<svg xmlns='http://www.w3.org/2000/svg'/>\n",
            "book/src/figures/10-mp2/toy.py": "# makes toy.svg\n",
        },
    )
    sub(tree, SEC, "Inline math", "![toy](../figures/10-mp2/toy.svg)\n\nInline math")
    bb.write_appendices(tree)
    rep, ctx = bb.run_check(tree)
    assert rep.errors == []
    assert all(
        not bb.rel(tree, p).startswith("book/src/figures")
        for p in bb.section_files(tree)
    )
    assert "figures" not in bb.build_summary(tree)
    out = tree / "book" / "out"
    bb.build_tree(tree, ctx, out)
    assert (out / "src" / "figures/10-mp2/toy.svg").read_text().startswith("<svg")
    assert (out / "src" / "figures/10-mp2/toy.py").is_file()
    # The page's relative link resolves inside the preprocessed tree too.
    page = out / "src" / "10-mp2" / "01-toy.md"
    assert (page.parent / "../figures/10-mp2/toy.svg").resolve().is_file()


def test_missing_figure_is_an_error(tree):
    sub(tree, SEC, "Inline math", "![toy](../figures/10-mp2/gone.svg)\n\nInline math")
    assert any("image not found" in e for e in check(tree).errors)


def test_item_body_skips_semicolons_inside_brackets():
    """A `;` in an array type in the signature must not end the item early."""
    src = "pub fn f(a: [[f64; 3]; 3]) -> Result<([[f64; 3]; 3], u8)> {\n    let x = 1;\n    x\n}\nfn g();\n"
    body = bb.item_body(src, 0)
    assert body.rstrip().endswith("}") and "let x = 1;" in body
    assert bb.item_body("fn g();\nfn h() {}", 0) == "fn g();"


# ---------------------------------------------------------------- appendices
APP = {n: f"book/src/{f}" for n, f in bb.GENERATED.items()}

APPENDIX_D = """# Appendix D: Toy interface

<!-- section: D; status: gap; registry: none; theory: none -->

Entry `ferric-toy/src/lib.rs::other_fn`.
"""


def test_appendices_current_in_a_valid_tree(tree):
    assert check(tree, regen=False).errors == []


def test_generated_output_is_deterministic(tree):
    first = {k: (tree / v).read_text() for k, v in APP.items()}
    bb.write_appendices(tree)
    assert {k: (tree / v).read_text() for k, v in APP.items()} == first


@pytest.mark.parametrize("letter", sorted(bb.GENERATED))
def test_stale_appendix_is_an_error(tree, letter):
    p = tree / APP[letter]
    p.write_text(p.read_text() + "\nhand edit\n")
    errs = check(tree, regen=False).errors
    assert any(APP[letter] in e and "stale" in e for e in errs)


@pytest.mark.parametrize("letter", sorted(bb.GENERATED))
def test_missing_appendix_is_an_error(tree, letter):
    (tree / APP[letter]).unlink()
    errs = check(tree, regen=False).errors
    assert any(APP[letter] in e and "missing" in e for e in errs)


def test_book_change_makes_the_appendices_stale(tree):
    """Renaming a box id, or a cited function, must show up in A and B."""
    sub(tree, SEC, "id=toy.a;", "id=toy.renamed;")
    assert any("appendix-a" in e for e in check(tree, regen=False).errors)
    sub(tree, SEC, "id=toy.renamed;", "id=toy.a;")
    assert check(tree, regen=False).errors == []
    sub(
        tree,
        SEC,
        "also\n`ferric-toy/src/lib.rs::{pair_energy, other_fn}`",
        "also\n`ferric-toy/src/lib.rs::{pair_energy}`",
    )
    assert any("appendix-b" in e for e in check(tree, regen=False).errors)


def test_appendix_a_lists_boxes_scripts_registry_and_todos(tree):
    text = (tree / APP["A"]).read_text()
    assert "| `toy.a` | 10.1 | `scripts/toy_sympy.py` | 1 | - |" in text
    assert "| `scripts/toy_sympy.py` | 1 | 10.1 |" in text
    assert "| toy | 10.1 |" in text and "| toy2 | - |" in text
    assert "| holey | no proof |" in text
    a4 = text.split("## A.4")[1]
    assert "| `toy.b` | 10.1 | add a claim |" in a4 and "toy.a" not in a4
    assert "1 verified equations and 1 unverified" in text


def test_appendix_b_lists_each_function_once_with_its_sections(tree):
    write(
        tree,
        {
            "book/src/11-x/index.md": "# 11. X\n",
            "book/src/11-x/01-y.md": (
                "# 11.1 Y\n\n<!-- section: 11.1; status: edit; registry: toy2 -->\n\n"
                "Uses `ferric-toy/src/lib.rs::pair_energy` and "
                "`ferric-toy/src/sub/deep.rs::in_submodule`.\n"
            ),
        },
    )
    resummarize(tree)
    assert check(tree).errors == []
    text = (tree / APP["B"]).read_text()
    assert "## B.ferric-toy" in text
    assert "| `ferric-toy/src/lib.rs::pair_energy` | 10.1, 11.1 |" in text
    assert text.count("`ferric-toy/src/lib.rs::pair_energy`") == 1
    assert "| `ferric-toy/src/lib.rs::other_fn` | 10.1 |" in text
    assert "ferric-toy/src/lib.rs::toy_test` | 10.1 |" in text  # a rust_test field
    # the generated appendices do not index themselves
    assert "App. " not in text


def test_appendix_b_sorts_sections_numerically(tree):
    assert bb.label_key("9.2") < bb.label_key("10.1") < bb.label_key("App. D")
    assert bb.label_key("Notation") < bb.label_key("1.1")
    assert bb.label_key("ch. 10") < bb.label_key("10.1")


def test_appendix_c_is_generated_from_the_validation_page(tree):
    assert check(tree).errors == []
    text = (tree / APP["C"]).read_text()
    assert "| Proven | Agrees with an independent reference. |" in text
    assert "| `toyscf` | SCF | `run_toy` | Proven |" in text
    assert "| `toymp2` | MP2 | `run_toy2` | Smoke |" in text
    assert "2 kinds: 1 Proven, 1 Smoke." in text
    sub(tree, "site/src/reference/validation.md", "| Smoke |", "| Spike |")
    assert check(tree).errors == []
    assert "2 kinds: 1 Proven, 1 Spike." in (tree / APP["C"]).read_text()


def test_appendix_c_goes_stale_when_a_grade_changes(tree):
    assert check(tree, regen=False).errors == []
    sub(
        tree,
        "site/src/reference/validation.md",
        "`run_toy` | [Proven]",
        "`run_toy` | Smoke",
    )
    assert any("appendix-c" in e for e in check(tree, regen=False).errors)


def test_handwritten_appendix_is_checked_indexed_and_listed_last(tree):
    write(tree, {"book/src/appendix-d-toy.md": APPENDIX_D})
    resummarize(tree)
    assert check(tree).errors == []
    assert (
        "| `ferric-toy/src/lib.rs::other_fn` | 10.1, App. D |"
        in (tree / APP["B"]).read_text()
    )
    lines = (tree / "book/src/SUMMARY.md").read_text().splitlines()
    assert lines[-1].endswith("(appendix-d-toy.md)")
    assert (
        lines.index("[Notation](00-notation.md)")
        < lines.index(
            next(ln for ln in lines if ln.startswith("- [") and "MP2" in ln.upper())
        )
        < lines.index(
            "[Appendix A: Verified equations and proof scripts](appendix-a-equations.md)"
        )
    )
    sub(tree, "book/src/appendix-d-toy.md", "section: D", "section: E")
    assert any("does not match" in e for e in check(tree).errors)
    sub(tree, "book/src/appendix-d-toy.md", "section: E", "section: D")
    sub(tree, "book/src/appendix-d-toy.md", "other_fn", "no_such_fn")
    assert any("not declared" in e for e in check(tree).errors)


# ---------------------------------------------------------------- PLAN.md
PLAN = """# Plan

## 3. Table of contents

### Part X. Toy part

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| 10.1 | Toy section | none | none | ready |
| 10.2 | Not written yet | none | none | gap |

### Appendices

| § | Section | Sources | Code | Status |
|---|---|---|---|---|
| A | Equations | none | none | ready |
| B | Functions | none | none | gap |
| C | Grades | none | none | ready |

## 4. Coverage

<!-- status-table:begin -->
<!-- status-table:end -->
"""


def test_plan_in_step_with_headers_passes(tree):
    write(tree, {"book/PLAN.md": PLAN})
    (tree / "book/PLAN.md").write_text(bb.plan_updated(tree))
    assert check(tree, regen=False).errors == []


def test_plan_status_drift_is_reported_with_the_section(tree):
    write(tree, {"book/PLAN.md": PLAN})
    (tree / "book/PLAN.md").write_text(bb.plan_updated(tree))
    sub(
        tree,
        "book/PLAN.md",
        "| Toy section | none | none | ready |",
        "| Toy section | none | none | edit |",
    )
    errs = check(tree, regen=False).errors
    assert any(
        "10.1 (PLAN edit, section ready)" in e and "--write-plan" in e for e in errs
    )


def test_plan_update_fixes_statuses_and_recounts(tree):
    write(tree, {"book/PLAN.md": PLAN.replace("| ready |\n| 10.2", "| edit |\n| 10.2")})
    new = bb.plan_updated(tree)
    assert "| 10.1 | Toy section | none | none | ready |" in new
    assert "| 10.2 | Not written yet | none | none | gap |" in new  # unwritten: kept
    assert "| X. Toy part | 1 | 0 | 1 | 0 | 2 |" in new
    assert "| B | Functions | none | none | ready |" in new  # follows the header
    assert "| Appendices | 3 | 0 | 0 | 0 | 3 |" in new
    assert "| **Total** | **4** | **0** | **1** | **0** | **5** |" in new
    assert "5 sections: 80% `ready`" in new
    (tree / "book/PLAN.md").write_text(new)
    assert bb.plan_updated(tree) == new  # idempotent
    assert check(tree, regen=False).errors == []


def test_section_without_a_plan_row_is_an_error(tree):
    write(tree, {"book/PLAN.md": PLAN})
    (tree / "book/PLAN.md").write_text(bb.plan_updated(tree))
    write(
        tree,
        {
            "book/src/12-z/01-w.md": "# 12.1 W\n\n<!-- section: 12.1; status: edit; registry: - -->\n"
        },
    )
    errs = check(tree).errors
    assert any("section 12.1 has a file but no row" in e for e in errs)


def test_no_plan_file_means_no_plan_check(tree):
    assert not (tree / "book/PLAN.md").exists()
    assert check(tree).errors == []


def test_the_real_books_appendices_and_plan_are_current():
    root = Path(__file__).resolve().parents[3]
    bb._src_cache.clear()
    rep = bb.Report()
    bb.check_appendices(root, rep)
    bb.check_plan(root, rep)
    assert rep.errors == []


def test_code_span_with_a_dollar_is_not_math(tree):
    sub(tree, SEC, "Inline math", "Set `$HOME` and `$PATH`, then $x_1$. Inline math")
    assert check(tree).errors == []


def test_code_span_after_math_is_accepted(tree):
    sub(
        tree, SEC, "Inline math", "$$ f = \\frac{a}{b} $$ (`build_k_r`).\n\nInline math"
    )
    assert check(tree).errors == []


def test_concat_names_figures_from_book_src(tree):
    # Pandoc resolves an image against its resource path, not the page directory.
    sub(tree, SEC, "Inline math", "![plot](../figures/10-mp2/a.svg)\n\nInline math")
    text = bb.concat(tree, {})
    assert "](figures/10-mp2/a.svg)" in text
    assert "](../figures" not in text


def test_pdf_job_has_what_the_figures_and_a_dry_run_need():
    wf = (SCRIPT.parents[1] / ".github/workflows/book.yml").read_text(encoding="utf-8")
    assert "--resource-path=.:book/src" in wf
    assert "librsvg2-bin" in wf, "pandoc converts SVG for xelatex with rsvg-convert"
    assert "github.event_name == 'workflow_dispatch'" in wf


def test_pdf_job_does_not_number_sections_twice():
    # Titles carry their own numbers ("3.2 Two-, three- and four-center
    # integrals"); pandoc's --number-sections printed "11 3.2 ..." in rc12's PDF.
    wf = (SCRIPT.parents[1] / ".github/workflows/book.yml").read_text(encoding="utf-8")
    pandoc = wf[wf.index('pandoc "$RUNNER_TEMP/ferric-book.md"') :]
    pandoc = pandoc[: pandoc.index('-o "ferric-book-')]
    assert "--number-sections" not in pandoc
