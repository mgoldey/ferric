#!/usr/bin/env python3
"""Preprocessor and checker for the ferric book (book/BUILD.md sections 3-5).

Standard library only, so it runs without building any Rust.

    book_build.py --check [--release] [--ceiling-base REF]
    book_build.py --write-summary
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
    dissertation or wiki; SUMMARY.md matches the chapter glob.
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
        for a in re.findall(r'"([^"]*)"|([^\s,"]+)', anchors or ""):
            want = a[0] or a[1]
            if not find_heading(read(fp), want):
                rep.err(where, f"heading not found in {path}: {want!r}")


def check_page(root: Path, p: Path, rep: Report, ctx: dict, release: bool) -> None:
    where = rel(root, p)
    text = p.read_text()
    prose = strip_fences(text)

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
    num = re.match(r"(\d\d)-.*/(\d\d)-", f"{p.parent.name}/{p.name}")
    want = f"{int(num.group(1))}.{int(num.group(2))}" if num else None
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


def build_summary(root: Path, appendix: bool = False) -> str:
    src = book_src(root)

    def h1(p: Path, default: str) -> str:
        for lvl, h in headings(p.read_text()):
            if lvl == 1:
                return h
        return default

    out = ["# Summary", ""]
    for p in sorted(src.glob("*.md")):
        if p.name not in {"SUMMARY.md"}:
            out.append(f"[{h1(p, p.stem)}]({p.name})")
    out.append("")
    for chapter in sorted(d for d in src.glob("[0-9][0-9]-*") if d.is_dir()):
        secs = sorted(chapter.glob("[0-9][0-9]-*.md"))
        if not secs:
            continue
        idx = chapter / "index.md"
        title = h1(idx, "") if idx.exists() else ""
        title = title or chapter.name.split("-", 1)[1].replace("-", " ").upper()
        target = f"{chapter.name}/index.md" if idx.exists() else ""
        out.append(f"- [{title}]({target})")
        for s in secs:
            out.append(f"  - [{h1(s, s.stem)}]({chapter.name}/{s.name})")
    if appendix:
        out += ["", "[Appendix A: unverified equations](appendix-a-unverified.md)"]
    return "\n".join(out) + "\n"


def git_show(root: Path, ref: str, path: str) -> str | None:
    r = subprocess.run(
        ["git", "-C", str(root), "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
    )
    return r.stdout if r.returncode == 0 else None


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
    }

    sections = section_files(root)
    if not sections:
        rep.err("book/src", "no section files found")
    for p in sections:
        check_section_header(root, p, rep, ctx)
    for p in all_pages(root):
        check_page(root, p, rep, ctx, release)

    summary = book_src(root) / "SUMMARY.md"
    if not summary.is_file() or summary.read_text() != build_summary(root):
        rep.err(
            "book/src/SUMMARY.md",
            "missing or stale; run scripts/book_build.py --write-summary",
        )

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


def appendix(ctx: dict) -> str:
    rows = [
        "# Appendix A: unverified equations",
        "",
        "Equations stated in the book with no machine check yet.",
        "",
        "| Box | Page | To do |",
        "|---|---|---|",
    ]
    for where, f in ctx["unverified"]:
        rows.append(f"| {f.get('id', '')} | `{where}` | {f.get('todo', '')} |")
    return "\n".join(rows) + "\n"


def build_tree(root: Path, ctx: dict, out: Path) -> None:
    src = out / "src"
    if src.exists():
        shutil.rmtree(src)
    stamp = f"\n\n---\n*Built from commit {commit_of(root)}.*\n"
    for p in all_pages(root):
        dest = src / p.relative_to(book_src(root))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rewrite_math(expand_includes(root, p.read_text())) + stamp)
    (src / "appendix-a-unverified.md").write_text(appendix(ctx) + stamp)
    (src / "SUMMARY.md").write_text(build_summary(root, appendix=True))
    # Figures (book/src/figures/<NN-slug>/: the generating script and its SVG) are
    # copied verbatim; mdBook serves them next to the preprocessed pages.
    figures = book_src(root) / "figures"
    if figures.is_dir():
        shutil.copytree(figures, src / "figures", dirs_exist_ok=True)


def concat(root: Path, ctx: dict) -> str:
    parts = []
    for p in sorted(book_src(root).glob("*.md")):
        if p.name != "SUMMARY.md":
            parts.append(p.read_text())
    for p in section_files(root):
        parts.append(p.read_text())
    parts.append(appendix(ctx))
    return expand_includes(root, "\n\n".join(parts))


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
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--concat", type=Path)
    a = ap.parse_args(argv)
    root = a.root.resolve()

    if a.write_summary:
        (book_src(root) / "SUMMARY.md").write_text(build_summary(root))
        print("wrote book/src/SUMMARY.md")
        if not (a.check or a.build or a.concat):
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
