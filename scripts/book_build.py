#!/usr/bin/env python3
"""Preprocessor and checker for the ferric book (book/BUILD.md sections 3-5).

Standard library only, so it runs without building any Rust.

    book_build.py --check [--release] [--ceiling-base REF]
    book_build.py --write-summary
    book_build.py --write-appendices # regenerate appendices A, B and C (committed)
    book_build.py --write-plan       # sync book/PLAN.md section statuses and counts
    book_build.py --build            # check, preprocess to book/out/src, mdbook build
    book_build.py --concat FILE.md   # one Markdown file for pandoc (PDF)

Chapters are discovered by glob (book/src/NN-slug/MM-section.md); nothing is
listed here. Checks:

  * every ``verified`` box names an existing script, a claim the script defines,
    a runner that names the script, and code references that resolve;
  * ``unverified`` boxes are counted against book/unverified-ceiling.txt (an error
    under --release, and against --ceiling-base REF the ceiling may only go down);
  * ``file.rs::item`` code references resolve to a declaration under crates/;
  * a section header lists registry ids that exist; a ``ready`` section may not
    cover a registry gap (under --release every registry id must be covered);
  * ``theory:`` headers and ``<!-- include: file.md#"Heading" -->`` resolve by
    heading text, never by number;
  * ``<!-- excerpt: file.rs::item -->`` fenced blocks occur in the item;
  * backticked repository paths exist; no link points at the local-only
    dissertation or wiki; SUMMARY.md matches the chapter glob;
  * appendices A (verified equations), B (functions cited) and C (validation grades)
    are generated, committed, and must equal what ``--write-appendices`` writes;
  * the status of each section in book/PLAN.md section 3 equals its header's status,
    and the counts in section 4 are the recount (``--write-plan`` rewrites both).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

STATUSES = {"ready", "edit", "gap", "verify"}
ROOTS = ("crates/", "scripts/", "site/", "testdata/", "examples/", "book/", ".github/")
LOCAL_ONLY = re.compile(r"(wiki/|papers/|dissertation)", re.I)
BOX_RE = re.compile(r"<!--\s*(verified|unverified):(.*?)-->", re.S)
HEADER_RE = re.compile(r"\A\s*(?:#[^\n]*\n\s*)?<!--\s*(section:.*?)-->", re.S)
INCLUDE_RE = re.compile(r"<!--\s*include:\s*(.*?)\s*-->")
EXCERPT_RE = re.compile(
    r"<!--\s*excerpt:\s*(\S+?)\s*-->\s*\n```[^\n]*\n(.*?)\n```", re.S
)
CODE_REF_RE = re.compile(r"([\w\-./]+\.rs)::(\{[^}]*\}|\w+(?:::\w+)*)")
FENCE_RE = re.compile(r"^(```|~~~)")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def err(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"{where}: {msg}")


# ----------------------------------------------------------------- discovery
def book_src(root: Path) -> Path:
    return root / "book" / "src"


def section_files(root: Path) -> list[Path]:
    """book/src/NN-slug/MM-name.md, in chapter then section order."""
    return sorted(book_src(root).glob("[0-9][0-9]-*/[0-9][0-9]-*.md"))


def appendix_files(root: Path) -> list[Path]:
    """book/src/appendix-X-name.md, in letter order. A, B and C are generated."""
    return sorted(book_src(root).glob("appendix-[a-z]-*.md"))


GENERATED = {
    "A": "appendix-a-equations.md",
    "B": "appendix-b-functions.md",
    "C": "appendix-c-grades.md",
}


def all_pages(root: Path) -> list[Path]:
    src = book_src(root)
    return sorted(p for p in src.rglob("*.md") if p.name != "SUMMARY.md")


def rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


# ------------------------------------------------------------------ markdown
def strip_fences(text: str) -> str:
    """Blank out fenced code blocks (keeps line numbers)."""
    out, fenced = [], False
    for line in text.split("\n"):
        if FENCE_RE.match(line.lstrip()):
            fenced = not fenced
            out.append("")
        else:
            out.append("" if fenced else line)
    return "\n".join(out)


def headings(text: str) -> list[tuple[int, str]]:
    res = []
    for line in strip_fences(text).split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            res.append((len(m.group(1)), m.group(2).strip()))
    return res


def slug(h: str) -> str:
    s = re.sub(r"[^\w\s-]", "", h.lower())
    return re.sub(r"\s", "-", s.strip())


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def find_heading(text: str, want: str) -> bool:
    return any(norm(h) == norm(want) or slug(h) == want for _, h in headings(text))


def section_body(text: str, want: str) -> str | None:
    """Lines under heading ``want`` up to the next heading of the same or higher level."""
    lines = text.split("\n")
    fenced, start, level = False, None, 0
    for i, line in enumerate(lines):
        if FENCE_RE.match(line.lstrip()):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        if start is None:
            if norm(m.group(2)) == norm(want) or slug(m.group(2)) == want:
                start, level = i + 1, len(m.group(1))
        elif len(m.group(1)) <= level:
            return "\n".join(lines[start:i]).strip("\n")
    return "\n".join(lines[start:]).strip("\n") if start is not None else None


def parse_fields(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in body.split(";"):
        part = part.strip()
        if not part:
            continue
        for sep in ("=", ":"):
            if sep in part:
                k, v = part.split(sep, 1)
                if re.fullmatch(r"[\w-]+", k.strip()):
                    out[k.strip()] = v.strip()
                    break
    return out


# ------------------------------------------------------------- code resolving
_src_cache: dict[Path, str] = {}


def read(p: Path) -> str:
    if p not in _src_cache:
        _src_cache[p] = p.read_text(errors="replace")
    return _src_cache[p]


def find_rs(root: Path, path: str) -> Path | None:
    for cand in (root / path, root / "crates" / path):
        if cand.is_file():
            return cand
    return None


def decl_re(name: str) -> re.Pattern:
    return re.compile(
        rf"\b(?:fn|struct|enum|trait|const|static|type|mod|union)\s+{re.escape(name)}\b"
    )


def resolve_item(root: Path, path: str, item: str) -> tuple[Path, int] | None:
    """File and offset of ``item`` declared in ``path`` or in its submodule directory."""
    f = find_rs(root, path)
    if f is None:
        return None
    name = item.split("::")[-1]
    pat = decl_re(name)
    cands = [f]
    sub = f.with_suffix("")
    if sub.is_dir():
        cands += sorted(sub.rglob("*.rs"))
    for c in cands:
        m = pat.search(read(c))
        if m:
            return c, m.start()
    return None


def item_body(text: str, start: int) -> str:
    """Text from ``start`` to the end of the braced item (comments and strings skipped)."""
    i, n, depth, opened, nest = start, len(text), 0, False, 0
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            i = text.find("\n", i)
            if i < 0:
                break
        elif text.startswith("/*", i):
            i = text.find("*/", i)
            if i < 0:
                break
            i += 1
        elif c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c == "{":
            depth, opened = depth + 1, True
        elif c == "}":
            depth -= 1
            if opened and depth == 0:
                return text[start : i + 1]
        elif c in "([":
            nest += 1
        elif c in ")]":
            nest -= 1
        elif c == ";" and not opened and nest == 0:
            # A `;` inside brackets, e.g. `[[f64; 3]; 3]` in a signature, does not end the item.
            return text[start : i + 1]
        i += 1
    return text[start:]


def expand_refs(spec: str) -> list[tuple[str, str]]:
    """'a.rs::{x, y}' -> [(a.rs, x), (a.rs, y)]."""
    out = []
    for path, items in CODE_REF_RE.findall(spec):
        if items.startswith("{"):
            out += [(path, s.strip()) for s in items[1:-1].split(",") if s.strip()]
        else:
            out.append((path, items))
    return out


def check_code_refs(root: Path, where: str, spec: str, rep: Report) -> None:
    for path, item in expand_refs(spec):
        if find_rs(root, path) is None:
            rep.err(where, f"code reference file not found: {path}")
        elif resolve_item(root, path, item) is None:
            rep.err(where, f"code reference not declared: {path}::{item}")


# ----------------------------------------------------------------- the checks
def claim_defined(script: str, claim: str) -> bool:
    c = re.escape(claim)
    pats = [
        rf"def\s+claim_?{c}\b",
        rf"(?i)\bclaim\s+{c}\b",
        rf"\[{c}\]",
        rf"""["']{c}["']""",
    ]
    if claim.isdigit():
        pats.append(rf"(?m)^\s{{0,4}}{c}\s{{2,}}\S")
    return any(re.search(p, script) for p in pats)


def check_box(
    root: Path,
    where: str,
    kind: str,
    body: str,
    rep: Report,
    ids: set[str],
    release: bool,
    registry: dict,
) -> bool:
    f = parse_fields(body)
    bid = f.get("id", "")
    if not bid:
        rep.err(where, f"{kind} box has no id")
        return False
    if bid in ids:
        rep.err(where, f"duplicate box id {bid}")
    ids.add(bid)
    if kind == "unverified":
        if not f.get("todo"):
            rep.err(where, f"unverified box {bid} has no todo")
        return True
    for k in ("script", "claim", "runner", "code"):
        if not f.get(k):
            rep.err(where, f"box {bid}: missing field {k}")
    script = f.get("script")
    if script:
        sp = root / script
        if not sp.is_file():
            rep.err(where, f"box {bid}: script not found: {script}")
        elif f.get("claim") and not claim_defined(read(sp), f["claim"]):
            rep.err(where, f"box {bid}: script {script} defines no claim {f['claim']}")
    runner = f.get("runner")
    pending = f.get("runner_state") == "pending-merge"
    if runner:
        rp = root / runner
        if not rp.is_file():
            if not pending:
                rep.err(where, f"box {bid}: runner not found: {runner}")
            elif release:
                rep.err(
                    where, f"box {bid}: runner_state=pending-merge in a release build"
                )
        else:
            if script and Path(script).name not in read(rp):
                rep.err(
                    where,
                    f"box {bid}: runner {runner} does not name {Path(script).name}",
                )
            if pending:
                rep.warn(
                    where, f"box {bid}: runner exists, drop runner_state=pending-merge"
                )
    if f.get("code"):
        check_code_refs(root, f"{where} box {bid}", f["code"], rep)
    if f.get("rust_test"):
        check_code_refs(root, f"{where} box {bid} rust_test", f["rust_test"], rep)
    else:
        rep.notes.append(f"{where}: box {bid} is identity-only (no rust_test)")
    reg = f.get("registry")
    if reg and reg not in registry.get("methods", {}):
        rep.err(where, f"box {bid}: registry id {reg} not under methods")
    return False


def split_ids(v: str) -> list[str]:
    return [
        s.strip()
        for s in v.split(",")
        if s.strip() and s.strip().lower() not in {"-", "none"}
    ]


def check_theory(root: Path, where: str, spec: str, rep: Report) -> None:
    pat = re.compile(
        r'([\w./\-]+\.md)(?:#((?:"[^"]*"|[^\s,;"]+)(?:\s*,\s*(?:"[^"]*"|[^\s,;"]+))*))?'
    )
    for m in pat.finditer(spec):
        path, anchors = m.group(1), m.group(2)
        fp = root / path
        if not fp.is_file():
            rep.err(where, f"theory file not found: {path}")
            continue
        # The include is expanded from text read in text mode, which turns a lone
        # carriage return into a newline: only the byte check sees it.
        check_control_chars(fp, f"{where} {path}", rep)
        for a in re.findall(r'"([^"]*)"|([^\s,"]+)', anchors or ""):
            want = a[0] or a[1]
            if not find_heading(read(fp), want):
                rep.err(where, f"heading not found in {path}: {want!r}")


# Any C0 control byte but newline, and DEL. A LaTeX command written through a
# non-raw Python string loses its backslash this way: "\beta" becomes a
# backspace and "eta", "\frac" a form feed and "rac", "\rm" a carriage return.
# HTML renders the damage silently; xelatex rejects the file ("Text line
# contains an invalid character"), which first showed on the v0.1.0rc12 PDF.
# A carriage return before a newline is a CRLF line ending (a Windows checkout
# or a file written in text mode there), not damage; a lone one is.
CONTROL_RE = re.compile(rb"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]|\r(?!\n)")
CONTROL_NAMES = {7: r"\a", 8: r"\b", 9: r"\t", 11: r"\v", 12: r"\f", 13: r"\r"}


def check_control_chars(p: Path, where: str, rep: Report) -> None:
    # Read bytes: text mode turns a lone carriage return into a newline.
    data = p.read_bytes()
    for m in CONTROL_RE.finditer(data):
        line = data.count(b"\n", 0, m.start()) + 1
        byte = m.group()[0]
        hint = CONTROL_NAMES.get(byte)
        cause = f" (a lost backslash? {hint} was read as an escape)" if hint else ""
        rep.err(where, f"line {line}: control character 0x{byte:02x}{cause}")


# A code span inside math: pandoc hands the backticks and underscores to LaTeX
# ("! Double subscript" on `build_s_d_kind`), and MathJax shows the backticks.
# A formula does not span a blank line or a table cell. Put the span after it: $$ ... $$ (`name`).
# An inline formula stays on one line; it may hold a vertical bar (|a|^2).
MATH_RE = re.compile(
    r"\$\$(?P<disp>(?:(?!\n[ \t]*\n).)+?)\$\$"
    r"|(?<![\\$])\$(?=\S)(?P<inl>[^\n]+?)(?<=\S)\$(?![\d$])",
    re.S,
)
# A table row, possibly quoted. Its unescaped bars separate cells.
TABLE_ROW_RE = re.compile(r"^(?:[ \t]*>)*[ \t]*\|.*$", re.M)


def check_math_code_spans(text: str, where: str, rep: Report) -> None:
    # A code span that holds a dollar sign (`$HOME`) is code, not math: blank it
    # first so its dollar does not pair with one further down the page.
    # Match whole spans left to right, so the closing backtick of one span never
    # pairs with the opening backtick of the next. A span opens and closes with
    # backtick runs of the same length (``a`b``), as in CommonMark.
    text = re.sub(
        r"(?<!`)(`+)(?!`)[^\n]*?(?<!`)\1(?!`)",
        lambda m: " " * len(m.group()) if "$" in m.group() else m.group(),
        text,
    )
    # A table cell ends a formula: turn each row's cell separators into blank lines.
    text = TABLE_ROW_RE.sub(lambda m: re.sub(r"(?<!\\)\|", "\n\n", m.group()), text)
    for m in MATH_RE.finditer(text):
        body = m.group("disp") or m.group("inl")
        if "`" in body:
            snippet = " ".join(body[body.index("`") :].split())[:40]
            rep.err(where, f"code span inside math, move it outside the $: {snippet}")


def check_page(root: Path, p: Path, rep: Report, ctx: dict, release: bool) -> None:
    where = rel(root, p)
    check_control_chars(p, where, rep)
    text = p.read_text()
    prose = strip_fences(text)
    # Included theory sections are part of the page; line numbers are of the
    # expanded text.
    check_math_code_spans(strip_fences(expand_includes(root, text)), where, rep)

    for kind, body in BOX_RE.findall(prose):
        if check_box(
            root, where, kind, body, rep, ctx["ids"], release, ctx["registry"]
        ):
            ctx["unverified"].append((where, parse_fields(body)))
        ctx["boxes"] += 1

    for spec in INCLUDE_RE.findall(prose):
        check_theory(root, f"{where} include", spec, rep)

    for ref, block in EXCERPT_RE.findall(text):
        for path, item in expand_refs(ref):
            hit = resolve_item(root, path, item)
            if hit is None:
                rep.err(where, f"excerpt target not found: {ref}")
            elif norm(block) not in norm(item_body(read(hit[0]), hit[1])):
                rep.err(where, f"excerpt does not occur in {path}::{item}")

    # Code references in prose, outside boxes' own comments (those were checked above).
    for path, item in expand_refs(BOX_RE.sub("", INCLUDE_RE.sub("", prose))):
        if find_rs(root, path) is None:
            rep.err(where, f"code reference file not found: {path}")
        elif resolve_item(root, path, item) is None:
            rep.err(where, f"code reference not declared: {path}::{item}")

    # Local-only sources must not be linked; relative image targets must exist.
    for target in re.findall(r"\]\(([^)\s]+)\)", prose):
        if LOCAL_ONLY.search(target) and not target.startswith("http"):
            rep.err(where, f"link to a local-only source: {target}")
    for target in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", prose):
        if not re.match(r"[a-z]+:", target) and not (p.parent / target).is_file():
            rep.err(where, f"image not found: {target}")

    # Backticked repository paths exist.
    scrubbed = BOX_RE.sub("", CODE_REF_RE.sub("CODEREF", prose))
    for tok in re.findall(r"`([^`\n]+)`", scrubbed):
        tok = tok.strip().split("#")[0].split(" ")[0].rstrip(".,:;")
        tok = re.sub(
            r":\d+(?:-\d+)?$", "", tok
        )  # file:line pins are checked for the file only
        if not tok.startswith(ROOTS) or re.search(r"[*<>{}$]|\.\.\.|…", tok):
            continue
        if not (root / tok).exists():
            rep.err(where, f"path does not exist: {tok}")


def section_label(p: Path) -> str | None:
    """'10.1' for book/src/10-mp2/01-x.md, 'D' for appendix-d-x.md."""
    a = re.match(r"appendix-([a-z])-", p.name)
    if a:
        return a.group(1).upper()
    num = re.match(r"(\d\d)-.*/(\d\d)-", f"{p.parent.name}/{p.name}")
    return f"{int(num.group(1))}.{int(num.group(2))}" if num else None


def check_section_header(root: Path, p: Path, rep: Report, ctx: dict) -> None:
    where = rel(root, p)
    m = HEADER_RE.match(p.read_text())
    if not m:
        rep.err(
            where,
            "no section header comment (<!-- section: N.M; status: ...; registry: ...; theory: ... -->)",
        )
        return
    f = parse_fields(m.group(1))
    want = section_label(p)
    ctx["headers"][want] = f
    if f.get("section") != want:
        rep.err(
            where, f"header section {f.get('section')!r} does not match path ({want})"
        )
    status = f.get("status", "")
    if status not in STATUSES:
        rep.err(where, f"status {status!r} not one of {sorted(STATUSES)}")
    ctx["status"][status] = ctx["status"].get(status, 0) + 1
    reg = ctx["registry"]
    for rid in split_ids(f.get("registry", "")):
        ctx["covered"].add(rid)
        if rid in reg.get("gaps", {}):
            if status == "ready":
                rep.err(
                    where,
                    f"ready section covers registry gap {rid} ({reg['gaps'][rid].get('reason', '')})",
                )
        elif rid not in reg.get("methods", {}):
            rep.err(where, f"registry id {rid} is in neither methods nor gaps")
    if f.get("theory"):
        check_theory(root, where + " theory", f["theory"], rep)


def chapter_dirs(root: Path) -> list[Path]:
    return sorted(d for d in book_src(root).glob("[0-9][0-9]-*") if d.is_dir())


def chapter_intro(chapter: Path) -> Path | None:
    """The chapter's own page: index.md, or 00-chapter.md when it carries a header."""
    for name in ("index.md", "00-chapter.md"):
        if (chapter / name).is_file():
            return chapter / name
    return None


def chapter_sections(chapter: Path) -> list[Path]:
    return sorted(
        p for p in chapter.glob("[0-9][0-9]-*.md") if p.name != "00-chapter.md"
    )


def build_summary(root: Path) -> str:
    """Front matter and notation are unnumbered prefix chapters; every other chapter
    is a numbered part whose page and section titles carry their own number in the
    H1 (book.toml turns mdBook's automatic section labels off)."""
    src = book_src(root)

    def h1(p: Path, default: str) -> str:
        for lvl, h in headings(p.read_text(encoding="utf-8")):
            if lvl == 1:
                return h
        return default

    out = ["# Summary", ""]
    for chapter in chapter_dirs(root):
        if chapter.name.startswith("00-"):
            for s in chapter_sections(chapter):
                out.append(f"[{h1(s, s.stem)}]({chapter.name}/{s.name})")
    for p in sorted(src.glob("*.md")):
        if p.name != "SUMMARY.md" and not p.name.startswith("appendix-"):
            out.append(f"[{h1(p, p.stem)}]({p.name})")
    out.append("")
    for chapter in chapter_dirs(root):
        if chapter.name.startswith("00-"):
            continue
        secs = chapter_sections(chapter)
        intro = chapter_intro(chapter)
        if intro is None:
            # check_chapter_titles reports the missing page as an error.
            out.append(f"- [{chapter.name}]()")
        else:
            out.append(f"- [{h1(intro, chapter.name)}]({chapter.name}/{intro.name})")
        for s in secs:
            out.append(f"  - [{h1(s, s.stem)}]({chapter.name}/{s.name})")
    apps = appendix_files(root)
    if apps:
        out.append("")
        out += [f"[{h1(a, a.stem)}]({a.name})" for a in apps]
    return "\n".join(out) + "\n"


def first_h1(text: str) -> str | None:
    for lvl, h in headings(text):
        if lvl == 1:
            return h
    return None


def check_chapter_titles(root: Path, rep: Report) -> None:
    """One number per title, and it is the right one.

    A chapter page is titled 'N. Title'; a section page 'N.M Title' with N.M equal
    to its `section:` marker and N equal to its chapter's number. Front matter and
    the notation page are unnumbered. Appendices are titled 'Appendix X'."""
    for chapter in chapter_dirs(root):
        n = int(chapter.name[:2])
        where = rel(root, chapter)
        intro = chapter_intro(chapter)
        if n > 0 and intro is None:
            rep.err(where, "chapter has no index.md (a chapter intro page is required)")
        for p in ([intro] if intro else []) + chapter_sections(chapter):
            w = rel(root, p)
            h = first_h1(p.read_text(encoding="utf-8"))
            if h is None:
                rep.err(w, "no H1 title")
                continue
            if n == 0:
                if re.match(r"\d", h):
                    rep.err(w, f"front matter is unnumbered, title is {h!r}")
                continue
            if p == intro:
                if not re.match(rf"{n}\. \S", h):
                    rep.err(w, f"chapter title must start '{n}. ', got {h!r}")
                continue
            # The path fixes the number; check_section_header ties it to the marker.
            want = section_label(p)
            if not re.match(rf"{re.escape(str(want))} \S", h):
                rep.err(w, f"section title must start '{want} ', got {h!r}")
    for p in appendix_files(root):
        h = first_h1(p.read_text(encoding="utf-8")) or ""
        if re.match(r"\d", h):
            rep.err(rel(root, p), f"appendix title must not start with a number: {h!r}")
    n0 = book_src(root) / "00-notation.md"
    if n0.is_file() and re.match(r"\d", first_h1(n0.read_text(encoding="utf-8")) or ""):
        rep.err(rel(root, n0), "notation title must be unnumbered")


def git_show(root: Path, ref: str, path: str) -> str | None:
    r = subprocess.run(
        ["git", "-C", str(root), "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
    )
    return r.stdout if r.returncode == 0 else None


# ------------------------------------------------------------------ appendices
GEN_NOTE = (
    "Generated by `scripts/book_build.py --write-appendices`. Do not edit by hand: "
    "the checker fails when this page differs from what the generator writes."
)


def page_label(root: Path, p: Path) -> str:
    """'10.1' for a section, 'ch. 10' for a chapter index, 'Notation', 'App. D'."""
    lab = section_label(p)
    if lab and p.name.startswith("appendix-"):
        return f"App. {lab}"
    if lab:
        return lab
    m = re.match(r"(\d\d)-", p.parent.name)
    if p.name == "index.md" and m:
        return f"ch. {int(m.group(1))}"
    if p.name.startswith("00-"):
        return "Notation"
    return p.stem


def label_key(lab: str) -> tuple:
    """Sort key: Notation, chapters and sections numerically, then appendices."""
    m = re.match(r"(\d+)\.(\d+)$", lab)
    if m:
        return (1, int(m.group(1)), int(m.group(2)), "")
    m = re.match(r"ch\. (\d+)$", lab)
    if m:
        return (1, int(m.group(1)), -1, "")
    if lab.startswith("App. "):
        return (2, 0, 0, lab)
    return (0, 0, 0, lab)


def cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def source_pages(root: Path) -> list[Path]:
    """Pages the indexes are built from: everything except the generated appendices."""
    gen = {GENERATED[k] for k in GENERATED}
    return [p for p in all_pages(root) if p.name not in gen]


def scan_boxes(root: Path) -> list[tuple[str, str, dict]]:
    """(kind, page label, fields) for every box, in book order."""
    out = []
    for p in source_pages(root):
        lab = page_label(root, p)
        for kind, body in BOX_RE.findall(strip_fences(p.read_text())):
            out.append((kind, lab, parse_fields(body)))
    out.sort(key=lambda b: (label_key(b[1]), b[2].get("id", "")))
    return out


def scan_code_refs(root: Path) -> dict[tuple[str, str], set[str]]:
    """{(path under crates/, item): page labels} over every ``file.rs::item`` reference."""
    refs: dict[tuple[str, str], set[str]] = {}
    for p in source_pages(root):
        lab = page_label(root, p)
        text = re.sub(
            r"<!--\s*unverified:.*?-->", "", strip_fences(p.read_text()), flags=re.S
        )
        for path, item in expand_refs(text):
            path = path.removeprefix("crates/")
            refs.setdefault((path, item), set()).add(lab)
    return refs


def load_registry(root: Path) -> dict:
    rp = root / "testdata/derivations/registry.json"
    return json.loads(rp.read_text()) if rp.is_file() else {}


def appendix_a(root: Path) -> str:
    boxes = scan_boxes(root)
    verified = [b for b in boxes if b[0] == "verified"]
    unverified = [b for b in boxes if b[0] == "unverified"]
    reg = load_registry(root)
    out = [
        "# Appendix A: Verified equations and proof scripts",
        "",
        "<!-- section: A; status: ready; registry: none; theory: none -->",
        "",
        GEN_NOTE,
        "",
        f"{len(verified)} verified equations and {len(unverified)} unverified equations. "
        "A verified equation names a proof script, the claim in it, the test runner "
        "that runs the script, and the code it describes; the checker resolves every "
        "one of them.",
        "",
        "## A.1 Verified equations",
        "",
        "| Box | Section | Proof script | Claim | Registry |",
        "|---|---|---|---|---|",
    ]
    for _, lab, f in verified:
        out.append(
            f"| `{cell(f.get('id', ''))}` | {lab} | `{cell(f.get('script', ''))}` | "
            f"{cell(f.get('claim', ''))} | {cell(f.get('registry', '')) or '-'} |"
        )
    by_script: dict[str, list[tuple[str, str]]] = {}
    for _, lab, f in verified:
        by_script.setdefault(f.get("script", ""), []).append((lab, f.get("id", "")))
    out += [
        "",
        "## A.2 Proof scripts",
        "",
        "| Script | Boxes | Sections |",
        "|---|---|---|",
    ]
    for script in sorted(by_script):
        labs = sorted({lab for lab, _ in by_script[script]}, key=label_key)
        out.append(
            f"| `{cell(script)}` | {len(by_script[script])} | {', '.join(labs)} |"
        )
    covered: dict[str, set[str]] = {}
    for p in section_files(root):
        m = HEADER_RE.match(p.read_text())
        if m:
            for rid in split_ids(parse_fields(m.group(1)).get("registry", "")):
                covered.setdefault(rid, set()).add(page_label(root, p))
    out += [
        "",
        "## A.3 Registry methods",
        "",
        "`testdata/derivations/registry.json` is the coverage ledger: every method "
        "with a derivation and a proof, and every recorded gap.",
        "",
        "| Method | Sections | Registry scripts |",
        "|---|---|---|",
    ]
    for mid in sorted(reg.get("methods", {})):
        scripts = reg["methods"][mid].get("scripts", [])
        labs = sorted(covered.get(mid, set()), key=label_key)
        out.append(
            f"| {mid} | {', '.join(labs) or '-'} | "
            f"{', '.join(f'`{cell(s)}`' for s in scripts) or '-'} |"
        )
    out += ["", "Recorded gaps (no proof yet):", "", "| Gap | Reason |", "|---|---|"]
    for gid in sorted(reg.get("gaps", {})):
        out.append(f"| {gid} | {cell(reg['gaps'][gid].get('reason', ''))} |")
    out += [
        "",
        "## A.4 Unverified equations",
        "",
        "Equations stated with no machine check yet.",
        "",
        "| Box | Section | To do |",
        "|---|---|---|",
    ]
    for _, lab, f in unverified:
        out.append(f"| `{cell(f.get('id', ''))}` | {lab} | {cell(f.get('todo', ''))} |")
    return "\n".join(out) + "\n"


def appendix_b(root: Path) -> str:
    refs = scan_code_refs(root)
    crates: dict[str, list[tuple[str, str]]] = {}
    for path, item in refs:
        crates.setdefault(path.split("/")[0], []).append((path, item))
    out = [
        "# Appendix B: Functions cited",
        "",
        "<!-- section: B; status: ready; registry: none; theory: none -->",
        "",
        GEN_NOTE,
        "",
        f"{len(refs)} items in {len(crates)} crates are named in the book as "
        "a Rust file path and an item name joined by two colons, in the text, in a verified equation's `code` or "
        "`rust_test` field, or in an excerpt header. The checker resolves each to "
        "a declaration. Each row lists the sections that name the item.",
    ]
    for crate in sorted(crates):
        out += ["", f"## B.{crate}", "", "| Item | Sections |", "|---|---|"]
        for path, item in sorted(crates[crate]):
            labs = sorted(refs[(path, item)], key=label_key)
            out.append(f"| `{path}::{item}` | {', '.join(labs)} |")
    return "\n".join(out) + "\n"


def md_table(text: str, header_first: str) -> list[list[str]]:
    """Rows (cells) of the first Markdown table whose header starts with ``header_first``."""
    rows: list[list[str]] = []
    inside = False
    for line in text.split("\n"):
        if not inside:
            if line.startswith("|") and line.strip("| ").startswith(header_first):
                inside = True
                rows.append(split_row(line))
            continue
        if not line.startswith("|"):
            break
        if re.fullmatch(r"\|[\s:|-]+\|?", line):
            continue
        rows.append(split_row(line))
    return rows


def split_row(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip())[1:-1]
    return [c.strip() for c in cells]


def plain(s: str) -> str:
    """Markdown link text only; markup that renders badly in a table cell dropped."""
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    return s.replace("**", "")


def appendix_c(root: Path) -> str:
    vp = root / "site/src/reference/validation.md"
    text = vp.read_text() if vp.is_file() else ""
    grades = md_table(text, "Grade")
    matrix = md_table(text, "`method.kind`")
    out = [
        "# Appendix C: Validation grades",
        "",
        "<!-- section: C; status: ready; registry: none; theory: none -->",
        "",
        GEN_NOTE,
        "",
        "The grades are those of `site/src/reference/validation.md`, which is the "
        "authority; the certificate every graded run prints carries the same word. "
        "A grade is for the exact method named; the Caveat column of the source "
        "page says what each grade does and does not cover, and the Anchors table "
        "there holds the numbers.",
        "",
        "## C.1 The grades",
        "",
        "| Grade | Meaning |",
        "|---|---|",
    ]
    for r in grades[1:]:
        if len(r) >= 2:
            out.append(f"| {cell(plain(r[0]))} | {cell(plain(r[1]))} |")
    out += ["", "## C.2 Grade of each CLI method kind", ""]
    if matrix:
        head = matrix[0]
        want = {
            "kind": "`method.kind`",
            "family": "Family",
            "py": "Python",
            "grade": "Grade",
        }
        ix = {k: head.index(v) for k, v in want.items() if v in head}
        tally: dict[str, int] = {}
        rows = []
        for r in matrix[1:]:
            if len(r) != len(head) or len(ix) != len(want):
                continue
            g = cell(plain(r[ix["grade"]]))
            tally[g] = tally.get(g, 0) + 1
            rows.append(
                f"| {cell(r[ix['kind']])} | {cell(plain(r[ix['family']]))} | "
                f"{cell(r[ix['py']])} | {g} |"
            )
        out.append(
            f"{len(rows)} kinds: "
            + ", ".join(f"{n} {g}" for g, n in sorted(tally.items()))
            + "."
        )
        out += ["", "| `method.kind` | Family | Python | Grade |", "|---|---|---|---|"]
        out += rows
    else:
        out.append("The validation page has no method matrix.")
    return "\n".join(out) + "\n"


APPENDIX_BUILDERS = {"A": appendix_a, "B": appendix_b, "C": appendix_c}


def write_appendices(root: Path) -> list[str]:
    """Write appendices A-C; returns the files written."""
    written = []
    for letter, name in GENERATED.items():
        dest = book_src(root) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(APPENDIX_BUILDERS[letter](root))
        written.append(rel(root, dest))
    return written


def check_appendices(root: Path, rep: Report) -> None:
    for letter, name in GENERATED.items():
        dest = book_src(root) / name
        where = rel(root, dest)
        if not dest.is_file():
            rep.err(where, "missing; run scripts/book_build.py --write-appendices")
        elif dest.read_text() != APPENDIX_BUILDERS[letter](root):
            rep.err(where, "stale; run scripts/book_build.py --write-appendices")


# ------------------------------------------------------------------ PLAN.md
PLAN_ROW_RE = re.compile(r"^\|\s*(\d+\.\d+|[A-Z])\s*\|")
PLAN_BEGIN = "<!-- status-table:begin -->"
PLAN_END = "<!-- status-table:end -->"


def plan_rows(text: str) -> list[tuple[int, str, str, str]]:
    """(line index, part, section label, status) for each section row of PLAN section 3."""
    rows, part, in3 = [], "", False
    for i, line in enumerate(text.split("\n")):
        if line.startswith("## "):
            in3 = line.startswith("## 3.")
        if not in3:
            continue
        if line.startswith("### "):
            part = line[4:].replace("Part ", "", 1).strip()
            continue
        r = PLAN_ROW_RE.match(line)
        if r:
            status = line.rstrip().rstrip("|").rsplit("|", 1)[-1].strip()
            rows.append((i, part, r.group(1), status))
    return rows


def section_status(root: Path) -> dict[str, str]:
    """Header status of every written section and appendix, by label."""
    out = {}
    for p in section_files(root) + appendix_files(root):
        m = HEADER_RE.match(p.read_text())
        if m:
            out[section_label(p)] = parse_fields(m.group(1)).get("status", "")
    return out


def plan_counts(rows: list[tuple[int, str, str, str]], actual: dict[str, str]) -> str:
    parts: dict[str, dict[str, int]] = {}
    order: list[str] = []
    for _, part, lab, status in rows:
        st = actual.get(lab, status)
        if part not in parts:
            parts[part] = {s: 0 for s in ("ready", "edit", "gap", "verify")}
            order.append(part)
        parts[part][st] = parts[part].get(st, 0) + 1
    tot = {
        s: sum(parts[p].get(s, 0) for p in order)
        for s in ("ready", "edit", "gap", "verify")
    }
    n = sum(tot.values())
    lines = [
        "| Part | ready | edit | gap | verify | Sections |",
        "|---|---|---|---|---|---|",
    ]
    for part in order:
        c = parts[part]
        lines.append(
            f"| {part} | {c['ready']} | {c['edit']} | {c['gap']} | {c['verify']} | {sum(c.values())} |"
        )
    lines.append(
        f"| **Total** | **{tot['ready']}** | **{tot['edit']}** | **{tot['gap']}** | "
        f"**{tot['verify']}** | **{n}** |"
    )
    pct = lambda k: round(100 * tot[k] / n) if n else 0  # noqa: E731
    lines += [
        "",
        f"{n} sections: {pct('ready')}% `ready`, {pct('edit')}% `edit`, "
        f"{pct('gap')}% `gap` and {pct('verify')}% `verify`.",
    ]
    return "\n".join(lines)


def plan_updated(root: Path) -> str | None:
    """PLAN.md with the section 3 statuses and the section 4 table brought up to date."""
    pp = root / "book" / "PLAN.md"
    if not pp.is_file():
        return None
    text = pp.read_text()
    rows = plan_rows(text)
    actual = section_status(root)
    lines = text.split("\n")
    for i, _, lab, status in rows:
        want = actual.get(lab)
        if want and want != status:
            head, _, _ = lines[i].rstrip().rstrip("|").rpartition("|")
            lines[i] = f"{head}| {want} |"
    text = "\n".join(lines)
    if PLAN_BEGIN in text and PLAN_END in text:
        a = text.index(PLAN_BEGIN) + len(PLAN_BEGIN)
        b = text.index(PLAN_END)
        text = text[:a] + "\n" + plan_counts(rows, actual) + "\n" + text[b:]
    return text


def check_plan(root: Path, rep: Report) -> None:
    pp = root / "book" / "PLAN.md"
    if not pp.is_file():
        return
    planned = {lab for _, _, lab, _ in plan_rows(pp.read_text())}
    for lab in sorted(set(section_status(root)) - planned, key=label_key):
        rep.err("book/PLAN.md", f"section {lab} has a file but no row in section 3")
    want = plan_updated(root)
    if want is not None and want != pp.read_text():
        drift = []
        actual = section_status(root)
        for _, _, lab, status in plan_rows(pp.read_text()):
            if lab in actual and actual[lab] != status:
                drift.append(f"{lab} (PLAN {status}, section {actual[lab]})")
        rep.err(
            "book/PLAN.md",
            "section statuses or counts differ from the section headers"
            + (": " + ", ".join(drift) if drift else "")
            + "; run scripts/book_build.py --write-plan",
        )


def run_check(
    root: Path, release: bool = False, ceiling_base: str | None = None
) -> tuple[Report, dict]:
    rep = Report()
    reg_path = root / "testdata/derivations/registry.json"
    registry = json.loads(reg_path.read_text()) if reg_path.is_file() else {}
    if not registry:
        rep.err("registry", f"{reg_path} not found")
    ctx = {
        "ids": set(),
        "unverified": [],
        "boxes": 0,
        "covered": set(),
        "status": {},
        "registry": registry,
        "headers": {},
    }

    sections = section_files(root)
    if not sections:
        rep.err("book/src", "no section files found")
    for p in sections + appendix_files(root):
        check_section_header(root, p, rep, ctx)
    for p in all_pages(root):
        check_page(root, p, rep, ctx, release)

    summary = book_src(root) / "SUMMARY.md"
    if not summary.is_file() or summary.read_text() != build_summary(root):
        rep.err(
            "book/src/SUMMARY.md",
            "missing or stale; run scripts/book_build.py --write-summary",
        )

    check_chapter_titles(root, rep)
    check_appendices(root, rep)
    check_plan(root, rep)

    uncovered = sorted(set(registry.get("methods", {})) - ctx["covered"])
    if uncovered:
        msg = f"{len(uncovered)} registry methods covered by no section: {', '.join(uncovered)}"
        (rep.err if release else rep.warn)("registry", msg)

    count = len(ctx["unverified"])
    cfile = root / "book/unverified-ceiling.txt"
    ceiling = (
        int(cfile.read_text().strip())
        if cfile.is_file() and cfile.read_text().strip().isdigit()
        else None
    )
    if ceiling is None:
        (rep.err if release else rep.warn)(
            "book/unverified-ceiling.txt", "missing or not an integer"
        )
    elif count > ceiling:
        (rep.err if release else rep.warn)(
            "unverified", f"{count} unverified boxes exceed the ceiling {ceiling}"
        )
    if ceiling_base and ceiling is not None:
        base = git_show(root, ceiling_base, "book/unverified-ceiling.txt")
        if base and base.strip().isdigit() and ceiling > int(base.strip()):
            rep.err(
                "book/unverified-ceiling.txt",
                f"ceiling raised from {base.strip()} to {ceiling}; it may only go down",
            )

    rep.notes.insert(
        0,
        f"{len(sections)} sections, {ctx['boxes']} boxes ({count} unverified), statuses {ctx['status']}",
    )
    return rep, ctx


# ----------------------------------------------------------------- preprocess
def rewrite_math(text: str) -> str:
    """$..$ and $$..$$ -> MathJax \\( \\) / \\[ \\] safe for the Markdown parser."""

    def protect(m: str, inline: bool) -> str:
        m = m.replace("\\|", "\\Vert ").replace("\\\\", "\x00")
        if inline:
            m = m.replace("|", "\\vert ")
        m = m.replace("\\", "\\\\").replace("\x00", "\\\\\\\\")
        return (
            m.replace("_", "\\_")
            .replace("*", "\\*")
            .replace("<", "\\lt ")
            .replace(">", "\\gt ")
        )

    def do_line_chunk(chunk: str) -> str:
        chunk = re.sub(
            r"\$\$(.+?)\$\$",
            lambda m: "\\\\[" + protect(m.group(1), False) + "\\\\]",
            chunk,
            flags=re.S,
        )
        return re.sub(
            r"(?<![\\$])\$(?=\S)(.+?)(?<=\S)\$(?![\d$])",
            lambda m: "\\\\(" + protect(m.group(1), True) + "\\\\)",
            chunk,
        )

    # Split into code (kept) and prose chunks: fenced blocks and inline code spans.
    parts = re.split(
        r"(^(?:```|~~~)[^\n]*\n.*?^(?:```|~~~)[^\n]*$|`[^`\n]+`)",
        text,
        flags=re.S | re.M,
    )
    for i in range(0, len(parts), 2):
        parts[i] = do_line_chunk(parts[i])
    return "".join(parts)


def expand_includes(root: Path, text: str) -> str:
    def sub(m: re.Match) -> str:
        parts = []
        for path, anchors in re.findall(
            r'([\w./\-]+\.md)#((?:"[^"]*"|[^\s,"]+)(?:\s*,\s*(?:"[^"]*"|[^\s,"]+))*)',
            m.group(1),
        ):
            for a in re.findall(r'"([^"]*)"|([^\s,"]+)', anchors):
                body = section_body((root / path).read_text(), a[0] or a[1])
                parts.append(body or "")
        return "\n\n".join(parts)

    return INCLUDE_RE.sub(sub, text)


def commit_of(root: Path) -> str:
    r = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() or "unknown"


def build_tree(root: Path, ctx: dict, out: Path) -> None:
    src = out / "src"
    if src.exists():
        shutil.rmtree(src)
    stamp = f"\n\n---\n*Built from commit {commit_of(root)}.*\n"
    for p in all_pages(root):
        dest = src / p.relative_to(book_src(root))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rewrite_math(expand_includes(root, p.read_text())) + stamp)
    (src / "SUMMARY.md").write_text(build_summary(root))
    # Figures (book/src/figures/<NN-slug>/: the generating script and its SVG) are
    # copied verbatim; mdBook serves them next to the preprocessed pages.
    figures = book_src(root) / "figures"
    if figures.is_dir():
        shutil.copytree(figures, src / "figures", dirs_exist_ok=True)


def concat(root: Path, ctx: dict) -> str:
    parts = []
    for p in sorted(book_src(root).glob("*.md")):
        if p.name != "SUMMARY.md" and not p.name.startswith("appendix-"):
            parts.append(p.read_text())
    for p in section_files(root):
        parts.append(p.read_text())
    for p in appendix_files(root):
        parts.append(p.read_text())
    text = expand_includes(root, "\n\n".join(parts))
    # Pages link figures as ../figures/<chapter>/<name>.svg, relative to the page.
    # The concatenation has no page directory: pandoc resolves paths against its
    # resource path, so name them from book/src (pass --resource-path=.:book/src).
    return text.replace("](../figures/", "](figures/")


# ------------------------------------------------------------------------ cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--release",
        action="store_true",
        help="strict: ceiling, pending-merge and registry coverage are errors",
    )
    ap.add_argument(
        "--ceiling-base", help="git ref; the ceiling may not exceed its value there"
    )
    ap.add_argument("--write-summary", action="store_true")
    ap.add_argument(
        "--write-appendices",
        action="store_true",
        help="regenerate appendices A, B and C",
    )
    ap.add_argument(
        "--write-plan",
        action="store_true",
        help="sync the section statuses and counts of book/PLAN.md with the section headers",
    )
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--concat", type=Path)
    a = ap.parse_args(argv)
    root = a.root.resolve()

    if a.write_summary:
        (book_src(root) / "SUMMARY.md").write_text(build_summary(root))
        print("wrote book/src/SUMMARY.md")
        if not (a.check or a.build or a.concat or a.write_appendices or a.write_plan):
            return 0

    if a.write_appendices:
        for name in write_appendices(root):
            print("wrote", name)
    if a.write_plan:
        new = plan_updated(root)
        if new is not None:
            (root / "book" / "PLAN.md").write_text(new)
            print("wrote book/PLAN.md")
    if (a.write_appendices or a.write_plan) and not (
        a.check or a.build or a.concat or a.write_summary
    ):
        return 0

    rep, ctx = run_check(root, a.release, a.ceiling_base)
    for n in rep.notes:
        print("note:", n)
    for w in rep.warnings:
        print("warning:", w)
    for e in rep.errors:
        print("error:", e)
    if rep.errors:
        print(f"RESULT: FAIL ({len(rep.errors)} errors)")
        return 1
    print("RESULT: ok")

    if a.concat:
        a.concat.write_text(concat(root, ctx))
        print(f"wrote {a.concat}")
    if a.build:
        out = root / "book" / "out"
        build_tree(root, ctx, out)
        if shutil.which("mdbook") is None:
            print("error: mdbook not on PATH")
            return 1
        return subprocess.call(["mdbook", "build", str(root / "book")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
