#!/usr/bin/env python3
"""One-shot migration of book/src from hand-written numbers to labels (2026-11-17).

Kept in the repository as evidence of how the labels were derived; it is not part
of the build and is not meant to run again (it refuses a tree that has no
numbered headings left).

Before this migration every number in the book was typed by hand: headings
(``## 10.1.2 Title``), ``\\tag{10.2}`` in display math, verified-box titles
(``**Verified equation 10.2**``), prose references (``(10.2)``, ``equation
10.2``, ``section 10.3``, ``§13.1.1``, ``chapter 13``), index-page link texts
(``[10.1 Title](01-x.md)``), the ``section:`` field of each header comment and
the rows of ``book/PLAN.md``. ``scripts/book_build.py`` now numbers from
position, so the source carries labels instead:

* a numbered heading becomes ``## Title {#label}``; the label is the page label
  (chapter directory or section file stem, without the number prefix) followed
  by up to four words of the title;
* ``\\tag{N.M}`` becomes ``\\label{eq:name}``; a verified box whose own number
  names no tagged equation on its page defines its number with ``{#eq:name}``;
  an equation label comes from the id of the box that certifies it, otherwise
  from the page label and the equation's ordinal on the page;
* every reference that matches a KNOWN number becomes ``@sec:name`` or
  ``@eq:name`` (``{@eq:name}`` when a letter follows, as in "15.8b").

Duplicate numbers: each duplicate definition gets its own label. A reference to a
duplicated number resolves to the nearest definition on the same page, else to
the nearest page; "Verified equation N" and "box N" prefer a box, everything else
a tagged equation. Every such case is printed for review. References that name
no known number, and references whose clause (or table cell and its column
header) cites the dissertation, a paper or a theory.md file, are left as written
and listed. A parenthesized two-level number is an equation when one has that
number, except on a chapter's intro page, where it names a section; "(0.x)" is a
value. The front matter's reading guide names chapters by bare number ("7 → 10")
and is converted as such.

FORCE_SKIP, FORCE_SECTION and FORCE_EQUATION record the cases a reviewer of the
dry-run report decided by hand, keyed by (page, line, number).

    python3 scripts/book_autonumber_migrate.py           # dry run: report only
    python3 scripts/book_autonumber_migrate.py --write   # rewrite book/src and book/PLAN.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import book_build as bb  # noqa: E402

STOP = set(
    "a an and are as at by for from how in is it its of on or the this that to "
    "what when where which why with without into one two not".split()
)
NUM = r"\d{1,2}(?:\.\d{1,2}){0,2}(?:[a-z](?![A-Za-z]))?(?![\d]|\.\d)"
NUMD = r"\d{1,2}(?:\.\d{1,2}){1,2}(?![\d]|\.\d)"
SEP = r"(?:[ \t]*,[ \t]*(?:and[ \t]+|or[ \t]+)?|[ \t]+(?:and|or|to|through)[ \t]+|[ \t]*[–—-][ \t]*)"
PAT = re.compile(
    rf"(?P<box>\*\*(?:Verified|Unverified) equations? )(?P<boxnums>[^*\n]+)\*\*"
    rf"|(?P<kw>\b(?:[Ss]ections?|[Cc]hapters?|[Ee]quations?|[Ee]qs?\.|[Bb]ox(?:es)?)|§)"
    rf"(?P<ws>[ \t]*\n?[ \t>]*)(?P<list>\(?{NUM}\)?(?:{SEP}\(?{NUM}\)?)*)"
    rf"|\((?P<plist>{NUMD}(?:{SEP}{NUMD})*)\)"
    rf"|(?<=\[)(?P<lnum>{NUMD})(?=[ \t][^\]\n]*\]\()"
    rf"|(?m:^(?P<bpre>[ \t]*(?:>[ \t]*)*[-*][ \t]+))(?P<bnum>{NUMD})(?=[ \t,])"
    rf"|(?<![\w.])(?P<b3>\d{{1,2}}\.\d{{1,2}}\.\d{{1,2}})(?![\d]|\.\d|\w)"
)
# Reviewed by hand (2026-11-17). A clause that names the literature is skipped,
# but these lines cite the book next to a source (FORCE_CONVERT), or cite a source
# without naming it in the clause (FORCE_SKIP). Keys: (page, line, number).
FORCE_CONVERT = {
    ("10-mp2/05-scs-mp2-2terfc.md", 8, "10.4"),  # "Attenuated MP2 (section 10.4)"
    (
        "10-mp2/06-attenuation-basis-limit.md",
        20,
        "10.12",
    ),  # "terfc (10.12)" under a dissertation column
    (
        "10-mp2/07-range-split-cross-term.md",
        156,
        "10.18",
    ),  # "theory.md and (10.18) both use"
    ("11-rimp2-algorithms/01-four-step-algorithm.md", 105, "11.5.1"),
    ("12-oo-laplace/03-laplace-mp2.md", 418, "12.3.4"),  # "operation counts of 12.3.4"
}
FORCE_SKIP = {
    (
        "04-operators/01-erf-terf.md",
        86,
        "4",
    ),  # the dissertation's "chapter 4, Eqs. 4.1–4.2"
    ("04-operators/01-erf-terf.md", 86, "4.1"),
    ("04-operators/01-erf-terf.md", 86, "4.2"),
    (
        "11-rimp2-algorithms/01-four-step-algorithm.md",
        185,
        "10.1.3",
    ),  # sections of book/sample-ri-mp2.md
    ("11-rimp2-algorithms/03-pair-kernel.md", 190, "10.1.3"),
    ("11-rimp2-algorithms/03-pair-kernel.md", 190, "10.1.4"),
}
# A parenthesized two-level number is an equation when one has the number; these
# name sections.
FORCE_SECTION = {
    ("11-rimp2-algorithms/05-parallel-performance.md", 61, "11.2"),  # "(11.2, 11.3)"
    ("11-rimp2-algorithms/05-parallel-performance.md", 61, "11.3"),
}
# On a chapter's intro page a parenthesized number names a section; these two
# name equations.
FORCE_EQUATION = {
    ("06-density-fitting/index.md", 8, "6.1"),  # "the factorization (6.1)"
    (
        "12-oo-laplace/00-chapter.md",
        45,
        "12.1",
    ),  # "Equations are numbered (12.1) and up"
}
VERSION_CUE = re.compile(
    r"(PySCF|ORCA|NWChem|Q-Chem|Psi4|simple-dftd3|pymbd|mdBook|libint2?|libxc|Python|version|v)\s*$",
    re.I,
)


def kebab(s: str) -> str:
    s = re.sub(r"`[^`]*`|\$[^$]*\$", " ", s)
    s = s.lower().replace("κ", "kappa").replace("ω", "omega")
    return "-".join(w for w in re.split(r"[^a-z0-9]+", s) if w)


def short(title: str, n: int = 4) -> str:
    words = [w for w in kebab(title).split("-") if w and w not in STOP]
    return "-".join(words[:n]) or "part"


class Labels:
    def __init__(self) -> None:
        self.used: set[str] = set()

    def take(self, want: str, alt: str | None = None) -> str:
        want = re.sub(r"-+", "-", want).strip("-")
        for cand in (want, alt):
            if cand and cand not in self.used:
                self.used.add(cand)
                return cand
        k = 2
        while f"{want}-{k}" in self.used:
            k += 1
        self.used.add(f"{want}-{k}")
        return f"{want}-{k}"


def box_label(box_id: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", box_id.lower())).strip("-")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a = ap.parse_args()
    root = a.root.resolve()
    src = bb.book_src(root)
    gen = set(bb.GENERATED.values())
    pages = [x for x in bb.book_pages(root) if x[0].name not in gen]
    texts = {p: p.read_text(encoding="utf-8") for p, *_ in pages}
    if not any(
        bb.NUMBERED_HEADING_RE.match(l)
        for t in texts.values()
        for l in bb.blank_spans(t, False).split("\n")
    ):
        print("no numbered headings left; the migration has already run")
        return 1
    labels = Labels()
    report: dict[str, list[str]] = {
        k: []
        for k in (
            "heading-mismatch",
            "ambiguous",
            "unresolved",
            "literature-skip",
            "version-skip",
            "box",
            "inline-def",
            "review",
            "paren-both",
        )
    }
    counts: dict[str, int] = {}

    def count(k: str, n: int = 1) -> None:
        counts[k] = counts.get(k, 0) + n

    # ---------------------------------------------------------- 1. headings
    sec: dict[str, tuple[Path, str]] = {}  # old number -> (page, label); notation 'n:k'
    page_label: dict[Path, str] = {}
    page_idx = {p: i for i, (p, *_) in enumerate(pages)}
    # A chapter is labelled by its directory slug; it takes the label before any
    # section file of the same stem (03-integrals/03-derivatives.md is then
    # integrals-derivatives, and chapter 14 is derivatives).
    chapter_label = {
        p: labels.take(re.sub(r"^\d\d-", "", p.parent.name))
        for p, kind, *_ in pages
        if kind == "chapter"
    }
    for p, kind, pnum, ci in pages:
        text = texts[p]
        lines = text.split("\n")
        stem = re.sub(r"^\d\d-", "", p.stem)
        chap = re.sub(r"^\d\d-", "", p.parent.name)
        counters = [0] * 7
        for i, lvl, title, label in bb.label_headings(text):
            if label:
                continue
            if lvl == 1:
                m = re.match(r"(\d+(?:\.\d+)?)\.?\s+(.*)$", title)
                if kind in ("chapter", "section"):
                    if not m or m.group(1) != pnum:
                        report["heading-mismatch"].append(
                            f"{p}: H1 {title!r} vs position {pnum}"
                        )
                    title = m.group(2) if m else title
                base = {
                    "chapter": chap,
                    "section": stem,
                    "front": stem,
                    "root": stem,
                    "notation": stem,
                    "appendix": f"appendix-{(pnum or '').lower()}",
                }[kind]
                lab = (
                    chapter_label[p]
                    if kind == "chapter"
                    else labels.take(base, f"{chap}-{base}")
                )
                page_label[p] = lab
                if pnum:
                    sec[pnum] = (p, lab)
                if kind == "notation":
                    sec["notation"] = (p, lab)
                if p.name == "00-chapter.md":
                    sec[f"{pnum}.0"] = (p, lab)  # PLAN.md keyed these intros N.0
                lines[i] = f"# {title} {{#{lab}}}"
                count("headings")
                continue
            m = re.match(r"((?:\d+|[A-Z])(?:\.\d+)*)\.?\s+(.*)$", title)
            if not m or not (re.match(r"\d", m.group(1)) or "." in m.group(1)):
                continue
            old, rest = m.group(1), m.group(2)
            counters[lvl] += 1
            for d in range(lvl + 1, 7):
                counters[d] = 0
            base = [pnum] if pnum and kind != "notation" else []
            want = ".".join(base + [str(c) for c in counters[2 : lvl + 1]])
            if kind == "notation":
                key = f"n:{old}"
            else:
                key = old
            if old != want:
                report["heading-mismatch"].append(
                    f"{p.relative_to(src)}: {title!r} is number {want} by position"
                )
            lab = labels.take(f"{page_label[p]}-{short(rest)}")
            sec[key] = (p, lab)
            lines[i] = f"{'#' * lvl} {rest} {{#{lab}}}"
            count("headings")
        texts[p] = "\n".join(lines)

    # --------------------------------------------- 2. equations and boxes
    BOX_TITLE = re.compile(r"\*\*(?:Verified|Unverified) equations? ([^*\n]+)\*\*")
    TOKEN = re.compile(r"(?<![\d.])(\d{1,2}\.\d{1,2})([a-z])?(?![\d]|\.\d)")
    eqdefs: dict[
        str, list[tuple[Path, int, str, str]]
    ] = {}  # old number -> [(page, offset, label, tag|box)]
    tag_edits: dict[Path, list[tuple[int, int, str]]] = {}
    box_defs: dict[Path, list[tuple[int, int, str]]] = {}
    for p, kind, pnum, ci in pages:
        if ci is None:
            continue
        text = texts[p]
        scrub = bb.blank_spans(text)
        tags = [
            (m.start(), m.end(), m.group(1))
            for m in re.finditer(r"\\tag\{([^}]*)\}", scrub)
        ]
        tagnums = {n for _, _, n in tags}
        boxes = []
        for m in BOX_TITLE.finditer(scrub):
            c = bb.BOX_RE.search(text, m.end())
            bid = bb.parse_fields(c.group(2)).get("id", "") if c else ""
            toks = list(TOKEN.finditer(m.group(1)))
            boxes.append((m, bid, toks))
        # Which tag does each box certify (same page only)?
        tag_box_label: dict[str, str] = {}
        for m, bid, toks in boxes:
            for t in toks:
                if t.group(1) in tagnums and t.group(1) not in tag_box_label and bid:
                    tag_box_label[t.group(1)] = box_label(bid)
        k = 0
        for s, e, n in tags:
            k += 1
            lab = labels.take(
                tag_box_label.pop(n, None) or f"{page_label[p]}-{k}",
                f"{page_label[p]}-{k}",
            )
            eqdefs.setdefault(n, []).append((p, s, lab, "tag"))
            tag_edits.setdefault(p, []).append((s, e, f"\\label{{eq:{lab}}}"))
            count("tags")
        for m, bid, toks in boxes:
            if not toks:
                report["unresolved"].append(
                    f"{p.relative_to(src)}: box title {m.group(0)!r}"
                )
                continue
            t0 = toks[0]
            if t0.group(1) not in tagnums and not t0.group(2):
                lab = labels.take(
                    box_label(bid) if bid else f"{page_label[p]}-box",
                    f"{page_label[p]}-box",
                )
                eqdefs.setdefault(t0.group(1), []).append((p, m.start(), lab, "box"))
                s = m.start(1) + t0.start()
                box_defs.setdefault(p, []).append(
                    (s, s + len(t0.group(0)), f"{{#eq:{lab}}}")
                )
                report["box"].append(
                    f"{p.relative_to(src)}: box {t0.group(1)} defines eq:{lab}"
                )
                count("box definitions")

    # A formula numbered by hand in a table cell, "$q_A = ...$ (27.5)", whose number
    # nothing else defines: it becomes a numbered item {#eq:...}.
    for p, kind, pnum, ci in pages:
        if ci is None:
            continue
        scrub = bb.blank_spans(texts[p])
        for m in re.finditer(r"\$[ \t]*\((\d+\.\d+)\)", scrub):
            n = m.group(1)
            if n in eqdefs or not n.startswith(f"{ci}."):
                continue
            line = texts[p][
                texts[p].rfind("\n", 0, m.start()) + 1 : texts[p].find("\n", m.start())
            ]
            first = (
                re.split(r"(?<!\\)\|", line)[1] if line.lstrip().startswith("|") else ""
            )
            lab = labels.take(f"{page_label[p]}-{short(first, 2)}")
            s = m.start(1)
            eqdefs.setdefault(n, []).append((p, s, lab, "box"))
            box_defs.setdefault(p, []).append((s, m.end(1), f"{{#eq:{lab}}}"))
            report["inline-def"].append(
                f"{p.relative_to(src)}: ({n}) after inline math defines eq:{lab}"
            )
            count("inline definitions")

    def resolve_eq(num: str, p: Path, off: int, prefer: str = "tag") -> str | None:
        """A number with several definitions: "Verified equation N" and "box N"
        prefer a box that states its own equation, every other form a tagged
        display equation; then the same page, then the nearest page."""
        allc = eqdefs.get(num, [])
        if not allc:
            return None
        if len(allc) == 1:
            return allc[0][2]
        c = [d for d in allc if d[3] == prefer] or allc
        same = [d for d in c if d[0] == p]
        pick = (
            min(same, key=lambda d: abs(d[1] - off))
            if same
            else min(c, key=lambda d: (abs(page_idx[d[0]] - page_idx[p]), d[1]))
        )
        report["ambiguous"].append(
            f"{p.relative_to(src)}:{texts[p].count(chr(10), 0, off) + 1}: {num} ({prefer} preferred) has {len(allc)} definitions "
            f"[{', '.join(str(d[0].relative_to(src)) + ':eq:' + d[2] + '/' + d[3] for d in allc)}] -> eq:{pick[2]}"
        )
        return pick[2]

    # Apply tag and box-definition edits before references (offsets are per text).
    for p in pages:
        p = p[0]
        ed = sorted(tag_edits.get(p, []) + box_defs.get(p, []), reverse=True)
        t = texts[p]
        # References are found on the text with these edits applied; record the
        # mapping from new offsets to old ones is unnecessary because resolve_eq
        # only compares offsets within one page, and the edits keep their order.
        for s, e, new in ed:
            t = t[:s] + new + t[e:]
        texts[p] = t
        # Recompute definition offsets on the edited text for nearest-match.
    for n, lst in eqdefs.items():
        for i, (p, _, lab, ty) in enumerate(lst):
            t = texts[p]
            j = t.find(f"eq:{lab}" + "}")
            lst[i] = (p, j, lab, ty)

    # --------------------------------------------------------- 3. references
    def ref(kind: str, lab: str, nxt: str) -> str:
        r = f"@{kind}:{lab}"
        return "{" + r + "}" if re.match(r"[A-Za-z0-9-]", nxt or "") else r

    def sec_ref(num: str, notation: bool) -> str | None:
        """'chapter 0' is the notation page; 'notation, section 8' its section 8."""
        if num == "0":
            return sec["notation"][1]
        if notation and f"n:{num}" in sec:
            return sec[f"n:{num}"][1]
        return sec.get(num, (None, None))[1]

    for p, kind, pnum, ci in pages:
        text = texts[p]
        nocomment = re.sub(
            r"<!--.*?-->", lambda m: re.sub(r"[^\n]", " ", m.group()), text, flags=re.S
        )
        scrub = bb.blank_spans(text)
        scrub = bb.DISPLAY_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), scrub)
        scrub = re.sub(
            r"(?<![\\$])\$(?=\S)[^\n$]+?(?<=\S)\$(?![\d$])",
            lambda m: " " * len(m.group()),
            scrub,
        )
        lines = nocomment.split("\n")
        starts = [0]
        for ln in lines:
            starts.append(starts[-1] + len(ln) + 1)
        edits: list[tuple[int, int, str]] = []
        rp = p.relative_to(src)

        def line_of(off: int) -> int:
            lo, hi = 0, len(starts) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if starts[mid] <= off:
                    lo = mid
                else:
                    hi = mid
            return lo

        def convert(
            s: int,
            e: int,
            kindhint: str,
            notation: bool,
            what: str,
            prefer: str = "tag",
        ) -> None:
            """Replace the number text[s:e] (may carry parentheses or a suffix)."""
            tok = text[s:e]
            m = re.fullmatch(r"(\(?)(\d{1,2}(?:\.\d{1,2}){0,2})([a-z]?)(\)?)", tok)
            if not m:
                return
            lp, num, suf, rpar = m.groups()
            if num.startswith("0.") and kind != "front":
                return  # a value such as "(0.1)": chapter 0 numbers only its example box
            lab, k = None, kindhint
            if kindhint == "eq":
                lab = resolve_eq(num, p, s, prefer)
            elif kindhint == "sec":
                lab = sec_ref(num, notation)
            else:  # parenthesized two-level: an equation if one has the number
                key3 = (str(rp), line_of(s) + 1, num)
                forced_sec = key3 in FORCE_SECTION or (
                    kind == "chapter" and key3 not in FORCE_EQUATION
                )
                lab = (
                    resolve_eq(num, p, s)
                    if num.count(".") == 1 and not forced_sec
                    else None
                )
                k = "eq"
                if lab is None:
                    lab, k = sec_ref(num, False), "sec"
                elif sec_ref(num, False):
                    ln0 = line_of(s)
                    report["paren-both"].append(
                        f"{rp}:{ln0 + 1}: ({num}) taken as an equation || {lines[ln0].strip()[:100]}"
                    )
            if lab is None:
                report["unresolved"].append(f"{rp}:{line_of(s) + 1}: {what} {num}{suf}")
                return
            new = (
                lp
                + ref(k, lab, suf or (text[e : e + 1] if not rpar else ""))
                + suf
                + rpar
            )
            edits.append((s, e, new))
            count(f"refs {what} ({k})")

        for m in PAT.finditer(scrub):
            ln = line_of(m.start())
            ctx = bb.literature_context(nocomment, m.start(), m.end())
            nums = set(re.findall(r"\d+(?:\.\d+)*", m.group()))
            forced = any((str(rp), ln + 1, n) in FORCE_CONVERT for n in nums)
            if any((str(rp), ln + 1, n) in FORCE_SKIP for n in nums):
                report["literature-skip"].append(
                    f"{rp}:{ln + 1}: {' '.join(m.group().split())} (by hand)"
                )
                continue
            if bb.LITERATURE_CUE.search(ctx) and not forced:
                report["literature-skip"].append(
                    f"{rp}:{ln + 1}: {' '.join(m.group().split())}"
                )
                continue
            # The paragraph names a source although the clause does not: list for review.
            pa = nocomment.rfind("\n\n", 0, m.start())
            pb = nocomment.find("\n\n", m.end())
            para = nocomment[pa + 1 : pb if pb > 0 else len(nocomment)]
            if bb.LITERATURE_CUE.search(para) and not forced and not m.group("box"):
                report["review"].append(
                    f"{rp}:{ln + 1}: {' '.join(m.group().split())}  || {' '.join(ctx.split())[:110]}"
                )
            if m.group("box"):
                for t in TOKEN.finditer(m.group("boxnums")):
                    s = m.start("boxnums") + t.start()
                    if text[s - 2 : s] == "{#" or text[s:].startswith("{#eq"):
                        continue
                    convert(s, s + len(t.group(0)), "eq", False, "box title", "tag")
                continue
            if m.group("kw"):
                kw = m.group("kw").lower()
                k = "sec" if kw[0] in "sc§" else "eq"
                before = nocomment[max(0, m.start() - 30) : m.end()].lower()
                notation = k == "sec" and (
                    "notation" in before or "chapter 0" in before or kind == "notation"
                )
                pre = nocomment[max(0, m.start() - 20) : m.start()].lower()
                prefer = (
                    "box"
                    if (kw.startswith("box") or pre.rstrip().endswith("verified"))
                    else "tag"
                )
                for t in re.finditer(rf"\(?{NUM}\)?", m.group("list")):
                    s = m.start("list") + t.start()
                    tok = t.group()
                    if k == "eq" and "." not in tok:
                        continue
                    convert(
                        s, s + len(tok), k, notation, f"after '{m.group('kw')}'", prefer
                    )
                continue
            if m.group("plist"):
                pre = scrub[max(0, m.start() - 20) : m.start()]
                if VERSION_CUE.search(pre):
                    report["version-skip"].append(f"{rp}:{ln + 1}: {m.group()}")
                    continue
                for t in re.finditer(NUMD, m.group("plist")):
                    s = m.start("plist") + t.start()
                    convert(
                        s,
                        s + len(t.group()),
                        "sec" if t.group().count(".") == 2 else "paren",
                        False,
                        "in parentheses",
                    )
                continue
            if m.group("lnum"):
                convert(m.start("lnum"), m.end("lnum"), "sec", False, "link text")
                continue
            if m.group("bnum"):
                convert(m.start("bnum"), m.end("bnum"), "sec", False, "list item")
                continue
            if m.group("b3"):
                pre = scrub[max(0, m.start() - 20) : m.start()]
                if VERSION_CUE.search(pre) or m.group("b3") not in sec:
                    report["version-skip"].append(
                        f"{rp}:{ln + 1}: {pre.strip()[-12:]} {m.group('b3')}"
                    )
                    continue
                convert(m.start("b3"), m.end("b3"), "sec", False, "bare")
        for s, e, new in sorted(edits, reverse=True):
            text = text[:s] + new + text[e:]
        # The header comment: drop the section number.
        text = re.sub(r"(<!--\s*)section:\s*[^;]*;\s*", r"\1", text, count=1)
        texts[p] = text

    # The reading guide of the front matter names chapters by bare number:
    # "7 → 10 → 11", "screening (5)", "chapters 7 and 8". Every standalone integer
    # in that one section is a chapter number.
    for p, kind, *_ in pages:
        if kind != "front":
            continue
        text = texts[p]
        s0 = text.find("\n## How the chapters depend on each other")
        if s0 < 0:
            continue
        s1 = text.find("\n## ", s0 + 1)
        body = text[s0:s1]
        scrub = bb.blank_spans(body)

        def chap(m: re.Match) -> str:
            n = m.group(1)
            if n in sec and "." not in n and scrub[m.start()] != " ":
                count("refs reading guide (sec)")
                return "@sec:" + sec[n][1]
            return n

        body = re.sub(r"(?<![\w.@:#-])(\d{1,2})(?![\w.]\d|\d|\w)", chap, body)
        texts[p] = text[:s0] + body + text[s1:]

    # ------------------------------------------------------------- 4. PLAN
    plan = root / "book" / "PLAN.md"
    ptext = plan.read_text(encoding="utf-8")
    key = {num: lab for num, (_, lab) in sec.items() if not num.startswith("n:")}
    key["0.2"] = sec["notation"][1]
    key = {k: v for k, v in key.items() if k != "0"}
    for letter in "ABCD":
        key[letter] = f"appendix-{letter.lower()}"

    def plan_row(m: re.Match) -> str:
        if m.group(1) in key:
            count("PLAN rows")
            return f"| `{key[m.group(1)]}` |"
        report["unresolved"].append(f"PLAN.md row {m.group(1)}")
        return m.group()

    in3 = False
    out = []
    for line in ptext.split("\n"):
        if line.startswith("## "):
            in3 = line.startswith("## 3.")
        if in3:
            line = re.sub(r"^\|\s*(\d+\.\d+|[A-Z])\s*\|", plan_row, line)
            line = line.replace("| § | Section |", "| Label | Section |")
        out.append(line)
    ptext = "\n".join(out)

    for k in sorted(counts):
        print(f"count {k}: {counts[k]}")
    for k, v in report.items():
        print(f"\n== {k} ({len(v)})")
        for line in v:
            print("  " + line)
    if a.write:
        for p, *_ in pages:
            p.write_text(texts[p], encoding="utf-8")
        plan.write_text(ptext, encoding="utf-8")
        print("\nwrote book/src and book/PLAN.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
