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

FILES = {
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
    (tmp_path / "book/src/SUMMARY.md").write_text(bb.build_summary(tmp_path))
    return tmp_path


def check(root, **kw):
    bb._src_cache.clear()
    return bb.run_check(root, **kw)[0]


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
            "book/src/42-new/03-thing.md": "# 42.3 Thing\n\n<!-- section: 42.3; status: edit; registry: toy2 -->\n"
        },
    )
    (tree / "book/src/SUMMARY.md").write_text(bb.build_summary(tree))
    assert check(tree).errors == []
    assert "42-new/03-thing.md" in (tree / "book/src/SUMMARY.md").read_text()


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
    "stale summary": ("book/src/SUMMARY.md", "Toy section", "Renamed section"),
    "unknown registry file": (
        "testdata/derivations/registry.json",
        '"toy": {}',
        '"toyx": {}',
    ),
}


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
