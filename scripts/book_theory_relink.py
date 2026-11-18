#!/usr/bin/env python3
"""One-shot relink of references into crates/*/theory.md to the book (2026-11-17).

Kept in the repository as evidence of how each reference was resolved; it is not
part of the build and is not meant to run again.

The 17 ``crates/<crate>/theory.md`` files held the derivations until the book
took them over; each is now a generated index (``book_build.py
--write-theory-index``). References that pointed at a heading, section number,
anchor or line of one of those files would dangle. This script rewrites each to
the book section that carries that heading's content.

Resolution. A reference names a crate and one of: a GitHub anchor
(``theory.md#94-attenuated-erfc-ri-mp2-gradient``), quoted heading text
(``theory.md#"Heading"``, ```theory.md`, "Heading"``, ```theory.md` ("Heading")``),
a section number (``§9.4``, ``section 7``) or a line (``theory.md:87``). It is
resolved against the theory.md of BASE (the last commit with the full files):
anchor, text and number to a heading, a line to the deepest heading containing
it. The heading maps to a book label through the coverage map of COVERAGE
(``book/theory-coverage.md``, first book section listed for the heading), or
through OVERRIDES. A label that no longer exists falls back to its page label.

Rendering. Inside ``book/src`` a reference becomes ``section @sec:label`` (the
build numbers it); a Sources-table row whose targets all lie on the page itself
is dropped. Elsewhere it becomes ```book/src/<page>#<label>```; in
``testdata/**/*.json`` the bare string. A bare ``crates/<crate>/theory.md``
with no locator already points at the crate's index and is left as written.

    python3 scripts/book_theory_relink.py [--write] [--only PREFIX ...] [--skip PREFIX ...]

Without --write it prints every rewrite and every reference it could not resolve.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import book_build as bb  # noqa: E402

BASE = "22770952b"  # full theory.md files
COVERAGE = "cf74ebe09"  # book/theory-coverage.md, the heading -> book map

# (crate, heading text) -> book label, where the coverage map's first entry is
# not the best home (absent headings, content moved by the gap-filling commit).
OVERRIDES = {
    (
        "ferric-mp2",
        "9.5 terfc-attenuated RI-MP2 gradient",
    ): "rimp2-gradient-attenuated-erfc-ri-mp2",
    (
        "ferric-scf",
        "Density-fitted gradient (RI-J) on a pair-screened tensor",
    ): "df-jk-gradient-pair-screened-ri-j",
    (
        "ferric-scf",
        "Peak memory of the fitted MP2 gradient",
    ): "rimp2-gradient-fitted-peak-memory",
    ("ferric-scf", "Factored RI-K in the open-shell CPHF"): "hf-hessian-factored-ri-k",
    ("ferric-scf", "Geometry searches"): "geometry-search",
    ("ferric-scf", "Self-Consistent Field (SCF) Theory"): "hartree-fock",
    ("ferric-mp2", "MP2 Family Theory"): "mp2",
    (
        "ferric-mp2",
        "6.1 Spin-blocked unrestricted MP3",
    ): "mp3-spin-blocked-unrestricted-mp3",
    (
        "ferric-mp2",
        "14.1 The integral-response loop",
    ): "unrestricted-oo-rimp2-integral-response-loop",
    (
        "ferric-integrals",
        "Derivative integrals of terf and terfc",
    ): "terf-table-engine-derivative-integrals",
    ("ferric-rpa", "The energy as a function of B and F"): "rpa-gradient-energy-b-f",
    ("ferric-rpa", "Frozen core"): "rpa-gradient-frozen-core",
    ("ferric-rpa", "What is reused from ferric-mp2"): "rpa-gradient-reuse",
    ("ferric-rpa", "Gradient"): "rpa-gradient-assembly",
    ("ferric-rpa", "Lagrangian and Z-vector"): "rpa-gradient-orbital-response",
}

# (crate, quoted text) -> the heading the quote abbreviates, where the quote is
# ambiguous or names no heading.
TEXT_ALIASES = {
    (
        "ferric-integrals",
        "Derivative integrals",
    ): "Derivative integrals and buffer sizing",
    ("ferric-scf", "Open shell"): "Kohn-Sham XC Hessian terms (fixed and moving grid)",
    (
        "ferric-scf",
        "RI-J gradient on a pair-screened tensor",
    ): "Density-fitted gradient (RI-J) on a pair-screened tensor",
    (
        "ferric-mp2",
        "Attenuated RI-MP2 gradient",
    ): "9.4 Attenuated (erfc) RI-MP2 gradient",
}

# The crate prefix is optional: inside crates/<crate>/ a bare ``theory.md``
# means that crate's file.
CRATE = r"(?:(?:crates/)?(?P<crate>ferric-[\w-]+)/)?theory\.md"
QUOTED = r'"[^"]+"(?:\s*(?:,\s*and|,|and)\s*(?:#\s*|//[/!]?\s*)?"[^"]+")*'
NUM = r"\d+(?:\.\d+)*[a-z]?"
REF_RE = re.compile(
    r"(?P<open>`?)"
    + CRATE
    + r"(?:#(?P<anchor>"
    + QUOTED
    + r"|[\w.-]+)|:(?P<line>\d+)(?:-(?P<line2>\d+))?)?"
    + r"(?P<close>`?)"
    + r"(?:(?P<hsep>,?[ \t]*(?:sections?|headings?)?[ \t]*\n?[ \t]*(?:#|//[/!]?)?[ \t]*|[ \t]*\([ \t]*(?:sections?[ \t]+)?)(?P<heads>"
    + QUOTED
    + r")(?P<hclose>\))?"
    + r"|(?P<nsep>,?[ \t]*\(?)(?:§|sections?|Section|(?=\d))[ \t]*(?P<num>"
    + NUM
    + r")"
    + r"(?:[ \t]*(?:–|-|to|and|,)[ \t]*§?(?P<num2>"
    + NUM
    + r"))?(?P<nclose>\))?)?"
)


def git_show(ref: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip("`")


def old_headings(crate: str) -> list[tuple[int, int, str]]:
    """(1-based line, level, text) of the BASE theory.md, outside code fences."""
    out, fenced = [], False
    for i, line in enumerate(
        git_show(BASE, f"crates/{crate}/theory.md").split("\n"), 1
    ):
        if bb.FENCE_RE.match(line.lstrip()):
            fenced = not fenced
            continue
        m = None if fenced else re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            out.append((i, len(m.group(1)), m.group(2).strip()))
    return out


def coverage_map() -> dict[tuple[str, int], str]:
    """(crate, heading line) -> first book target 'path#label' or 'path'."""
    out, crate = {}, None
    for line in git_show(COVERAGE, "book/theory-coverage.md").split("\n"):
        m = re.match(r"^## (ferric-[\w-]+) \(", line)
        if m:
            crate = m.group(1)
            continue
        m = re.match(r"^\| L(\d+) \|[^|]*\|(?P<cell>[^|]*)\|", line)
        if crate and m:
            tok = re.search(r"`([\w./-]+\.md)(#[\w:-]+)?`", m.group("cell"))
            if tok:
                out[(crate, int(m.group(1)))] = tok.group(1) + (tok.group(2) or "")
    return out


class Resolver:
    def __init__(self) -> None:
        self.nb = bb.numbering(ROOT)
        self.cov = coverage_map()
        self.heads: dict[str, list[tuple[int, int, str]]] = {}

    def headings(self, crate: str):
        if crate not in self.heads:
            try:
                self.heads[crate] = old_headings(crate)
            except subprocess.CalledProcessError:
                self.heads[crate] = []
        return self.heads[crate]

    def find(self, crate: str, *, anchor=None, text=None, num=None, line=None):
        hs = self.headings(crate)
        if line is not None:
            best = None
            for ln, lvl, h in hs:
                if ln <= line:
                    best = (ln, lvl, h)
            return best
        if text is not None:
            # A quoted heading may wrap across comment lines ("of the\n//! fitted").
            text = re.sub(r"\n\s*(?://[/!]?|#|\*|>)?\s*", " ", text)
            text = TEXT_ALIASES.get((crate, norm(text)), text)
            want = norm(text).lower()
            # Exact text, then without a leading number ("3. "), then a unique
            # heading that starts with the quoted words ("Two-step CASSCF
            # (`src/casscf.rs`)").
            bare = [
                (ln, lvl, h, re.sub(rf"^{NUM}\.?\s+", "", norm(h)).lower())
                for ln, lvl, h in hs
            ]
            for ln, lvl, h, b in bare:
                if norm(h).lower() == want or b == want:
                    return ln, lvl, h
            pre = [
                (ln, lvl, h)
                for ln, lvl, h, b in bare
                if b.startswith(want) or norm(h).lower().startswith(want)
            ]
            return pre[0] if len(pre) == 1 else None
        for ln, lvl, h in hs:
            if anchor is not None and (
                bb.slug(h) == anchor or bb.slug(norm(h)) == anchor
            ):
                return ln, lvl, h
            if num is not None and re.match(rf"{re.escape(num)}\.?\s", h):
                return ln, lvl, h
        return None

    def label(self, crate: str, head: tuple[int, int, str]) -> tuple[str, str] | None:
        """(label, page path under book/src) for an old heading."""
        ln, _, h = head
        lab = OVERRIDES.get((crate, h))
        if lab is None:
            tgt = self.cov.get((crate, ln))
            if tgt is None:
                return None
            path, _, lab = tgt.partition("#")
            if not lab or ("sec", lab) not in self.nb.targets:
                page = Path(path)
                lab = self.nb.page_key.get(page)
                if lab is None:
                    return None
        t = self.nb.targets.get(("sec", lab))
        if t is None:
            return None
        return lab, t.page.as_posix()


def targets_of(res: Resolver, m: re.Match, crate: str) -> list[tuple[str, str]] | None:
    heads = []
    if m.group("anchor"):
        a = m.group("anchor")
        if a.startswith('"'):
            heads = [res.find(crate, text=t) for t in re.findall(r'"([^"]+)"', a)]
        else:
            heads = [res.find(crate, anchor=a)]
    elif m.group("line"):
        heads = [res.find(crate, line=int(m.group("line2") or m.group("line")))]
    elif m.group("heads"):
        heads = [
            res.find(crate, text=t) for t in re.findall(r'"([^"]+)"', m.group("heads"))
        ]
    elif m.group("num"):
        heads = [res.find(crate, num=m.group("num"))]
        if m.group("num2"):
            heads.append(res.find(crate, num=m.group("num2")))
    else:
        return []  # bare file: the index
    if any(h is None for h in heads):
        return None
    out = []
    for h in heads:
        lp = res.label(crate, h)
        if lp is None:
            return None
        if lp not in out:
            out.append(lp)
    return out


def render(file: str, tgts: list[tuple[str, str]]) -> str:
    if file.startswith("book/src/"):
        refs = [f"@sec:{lab}" for lab, _ in tgts]
        word = "section" if len(refs) == 1 else "sections"
        return f"{word} " + (
            ", ".join(refs[:-1]) + " and " + refs[-1] if len(refs) > 1 else refs[0]
        )
    if file.endswith(".json"):
        return ", ".join(f"book/src/{p}#{lab}" for lab, p in tgts)
    return ", ".join(f"`book/src/{p}#{lab}`" for lab, p in tgts)


def rewrite(res: Resolver, file: str, text: str, log: list[str]) -> str:
    page = file.removeprefix("book/src/")
    out, pos = [], 0
    drop_rows: set[int] = set()
    home = re.match(r"crates/(ferric-[\w-]+)/", file)
    for m in REF_RE.finditer(text):
        crate = m.group("crate") or (home.group(1) if home else None)
        if crate is None:
            continue
        tg = targets_of(res, m, crate)
        lineno = text.count("\n", 0, m.start()) + 1
        if tg == []:
            continue
        if tg is None:
            log.append(f"UNRESOLVED {file}:{lineno}: {norm(m.group())}")
            continue
        # A Sources-table row citing only its own page goes.
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        row = text[line_start : line_end if line_end >= 0 else len(text)]
        if (
            file.startswith("book/src/")
            and row.lstrip().startswith("|")
            and all(p == page for _, p in tg)
        ):
            drop_rows.add(line_start)
            log.append(f"DROP-ROW {file}:{lineno}: {norm(row)[:120]}")
            continue
        if file.startswith("book/src/") and all(p == page for _, p in tg):
            log.append(f"SELF {file}:{lineno}: {norm(row)[:120]}")
        start, end = m.start(), m.end()
        new = render(file, tg)
        # keep a parenthesis the match opened but did not close, and vice versa
        grabbed = m.group()
        opened = grabbed.count("(") - grabbed.count(")")
        if opened > 0:
            new = "(" * opened + new
        elif opened < 0:
            new = new + ")" * (-opened)
        if file.endswith(".json") and grabbed.startswith("`"):
            new = "`" + new
        out.append(text[pos:start] + new)
        pos = end
        log.append(f"REWRITE {file}:{lineno}: {norm(grabbed)}  ->  {new}")
    out.append(text[pos:])
    new_text = "".join(out)
    if drop_rows:
        lines, keep, off = new_text.split("\n"), [], 0
        # recompute row starts on the rewritten text: drop rows by content
        dropped = {
            text[s : (text.find("\n", s) if text.find("\n", s) >= 0 else len(text))]
            for s in drop_rows
        }
        keep = [ln for ln in lines if ln not in dropped]
        new_text = "\n".join(keep)
    return new_text


def candidate_files(only: list[str], skip: list[str]) -> list[str]:
    files = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-l", "theory\\.md"],
        capture_output=True,
        text=True,
    ).stdout.split()
    out = []
    for f in files:
        if re.fullmatch(r"crates/[^/]+/theory\.md", f) or f in {
            "book/theory-coverage.md",
            "scripts/book_theory_relink.py",
            "scripts/book_build.py",
            "scripts/book_autonumber_migrate.py",
            # Its fixtures build toy theory.md files on purpose.
            "crates/ferric-python/tests/test_book_build.py",
            # Its mutation test names an old theory.md anchor on purpose.
            "crates/ferric-python/tests/test_derivation_registry.py",
            # Notation pages discuss theory.md's own notation line by line;
            # they are edited by hand.
            "book/notation.md",
            "book/src/00-notation.md",
        }:
            continue
        if only and not any(f.startswith(p) for p in only):
            continue
        if any(f.startswith(p) for p in skip):
            continue
        out.append(f)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    res = Resolver()
    log: list[str] = []
    for f in candidate_files(a.only, a.skip):
        p = ROOT / f
        text = p.read_text()
        new = rewrite(res, f, text, log)
        if a.write and new != text:
            p.write_text(new)
    for line in log:
        print(line)
    kinds = {}
    for line in log:
        k = line.split()[0]
        kinds[k] = kinds.get(k, 0) + 1
    print(kinds, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
