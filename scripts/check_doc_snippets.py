"""Run the Python examples in the published docs and check their printed output.

A page opts a ```python block in by putting a marker on the line before it:

    <!-- doctest -->
    <!-- doctest: atol=1e-4 -->     (looser numeric tolerance for this block)

Marked blocks on one page run IN ORDER IN ONE PROCESS, like a notebook, so a
later block may use names an earlier one defined. Each page gets a fresh
interpreter and a fresh temporary working directory.

A block's expected output is a ```text block that either follows it directly
(only blank lines between) or carries its own marker further down the page,
before the next marked block:

    <!-- doctest-output -->

Output lines are compared one to one. Numbers are compared with an absolute
tolerance (default atol=1e-8; rtol defaults to 0 so that a 75 Ha energy is held
to the same 1e-8 as a 0.01 Ha one), everything else exactly, after collapsing
runs of whitespace. A block with no expected output only has to run without
raising.

Usage:
    python scripts/check_doc_snippets.py [--root REPO] [PAGE.md ...]

With no pages given, every Markdown file under site/src plus README.md is
scanned. Exit status is 0 when every marked block ran and matched, 1 otherwise.
The script uses only the standard library, so it runs in a bare venv that has
just the ferric wheel installed.
"""

from __future__ import annotations

import argparse
import math
import os
import re

# subprocess runs the current interpreter on generated code; no shell.
import subprocess  # nosec B404
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MARKER = re.compile(r"^\s*<!--\s*doctest(?::\s*(?P<opts>.*?))?\s*-->\s*$")
OUTPUT_MARKER = re.compile(r"^\s*<!--\s*doctest-output\s*-->\s*$")
NUMBER = re.compile(r"[-+−]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
# Plain ASCII on purpose: str.splitlines() splits on \x1e and other
# control separators, which would silently hide every block boundary.
SENTINEL = "@@ferric-doctest-block-"
DEFAULT_ATOL = 1e-8
DEFAULT_RTOL = 0.0
PAGE_TIMEOUT_S = 900


@dataclass
class Block:
    line: int
    code: str
    expected: str | None
    atol: float = DEFAULT_ATOL
    rtol: float = DEFAULT_RTOL


@dataclass
class PageResult:
    page: Path
    blocks: int
    failures: list[str] = field(default_factory=list)


def parse_options(opts: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in (opts or "").split():
        key, _, value = item.partition("=")
        if key not in ("atol", "rtol") or not value:
            raise ValueError(f"unknown doctest option {item!r}")
        out[key] = float(value)
    return out


def fenced(lines: list[str], start: int) -> tuple[str, str, int]:
    """Return (info string, body, index after the closing fence) of the fence at start."""
    info = lines[start].strip()[3:].strip()
    body: list[str] = []
    i = start + 1
    while i < len(lines) and not lines[i].strip().startswith("```"):
        body.append(lines[i])
        i += 1
    return info, "\n".join(body) + "\n", i + 1


def marked_blocks(text: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        if OUTPUT_MARKER.match(lines[i]):
            j = i + 1
            if (
                not blocks
                or j >= len(lines)
                or not lines[j].strip().startswith("```text")
            ):
                raise ValueError(
                    f"line {i + 1}: doctest-output marker must sit directly above a "
                    "```text block and after a <!-- doctest --> block"
                )
            if blocks[-1].expected is not None:
                raise ValueError(
                    f"line {i + 1}: the block at line {blocks[-1].line} already has output"
                )
            _, blocks[-1].expected, i = fenced(lines, j)
            continue
        m = MARKER.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        if j >= len(lines) or not lines[j].strip().startswith("```python"):
            raise ValueError(
                f"line {i + 1}: doctest marker is not directly above a ```python block"
            )
        opts = parse_options(m.group("opts"))
        _, code, after = fenced(lines, j)
        expected = None
        k = after
        while k < len(lines) and not lines[k].strip():
            k += 1
        if k < len(lines) and lines[k].strip().startswith("```text"):
            _, expected, after = fenced(lines, k)
        blocks.append(Block(line=j + 1, code=code, expected=expected, **opts))
        i = after
    return blocks


def compare(expected: str, actual: str, atol: float, rtol: float) -> str | None:
    """Return None if actual matches expected, else a description of the first mismatch."""

    def norm(s: str) -> list[str]:
        return [
            " ".join(line.split()) for line in s.strip().splitlines() if line.strip()
        ]

    exp_lines, act_lines = norm(expected), norm(actual)
    if len(exp_lines) != len(act_lines):
        return f"expected {len(exp_lines)} output lines, got {len(act_lines)}:\n{actual.rstrip()}"
    for n, (e, a) in enumerate(zip(exp_lines, act_lines), start=1):
        if NUMBER.sub("#", e) != NUMBER.sub("#", a):
            return f"output line {n} differs:\n  expected: {e}\n  actual:   {a}"
        for x, y in zip(NUMBER.findall(e), NUMBER.findall(a)):
            fx, fy = float(x.replace("−", "-")), float(y.replace("−", "-"))
            if not math.isclose(fx, fy, rel_tol=rtol, abs_tol=atol):
                return (
                    f"output line {n}: {y} differs from documented {x} "
                    f"(atol={atol:g}, rtol={rtol:g})\n  expected: {e}\n  actual:   {a}"
                )
    return None


def run_page(page: Path, root: Path) -> PageResult:
    blocks = marked_blocks(page.read_text(encoding="utf-8"))
    result = PageResult(page=page, blocks=len(blocks))
    if not blocks:
        return result

    script = ["import sys"]
    for n, b in enumerate(blocks):
        script.append(f"print({SENTINEL + str(n)!r}, flush=True)")
        script.append(b.code)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    rel = page.relative_to(root) if page.is_relative_to(root) else page
    with tempfile.TemporaryDirectory(prefix="ferric-doctest-") as tmp:
        path = Path(tmp) / "page.py"
        path.write_text("\n".join(script), encoding="utf-8")
        try:
            proc = subprocess.run(  # nosec B603
                [sys.executable, str(path)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=PAGE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            result.failures.append(f"{rel}: timed out after {PAGE_TIMEOUT_S} s")
            return result

    outputs: dict[int, list[str]] = {}
    current = None
    for line in proc.stdout.splitlines():
        if line.startswith(SENTINEL) and line[len(SENTINEL) :].isdigit():
            current = int(line[len(SENTINEL) :])
            outputs[current] = []
        elif current is not None:
            outputs[current].append(line)

    if proc.returncode != 0:
        failed = max(outputs, default=0)
        tail = "\n".join(proc.stderr.rstrip().splitlines()[-15:])
        result.failures.append(f"{rel}:{blocks[failed].line}: block raised\n{tail}")
    for n, b in enumerate(blocks):
        if b.expected is None:
            continue
        if proc.returncode != 0 and n >= max(outputs, default=0):
            continue
        if n not in outputs:
            # A block that ran but whose output was never captured would
            # otherwise pass without anything being compared.
            result.failures.append(f"{rel}:{b.line}: no output captured for this block")
            continue
        problem = compare(b.expected, "\n".join(outputs[n]), b.atol, b.rtol)
        if problem:
            result.failures.append(f"{rel}:{b.line}: {problem}")
    return result


def default_pages(root: Path) -> list[Path]:
    pages = sorted((root / "site" / "src").rglob("*.md"))
    readme = root / "README.md"
    return [readme, *pages] if readme.is_file() else pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("pages", nargs="*", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    pages = [p.resolve() for p in args.pages] or default_pages(root)

    total_blocks = 0
    failures: list[str] = []
    for page in pages:
        res = run_page(page, root)
        if res.blocks:
            status = "FAIL" if res.failures else "ok"
            print(f"{status:4} {res.blocks:2} block(s)  {page.relative_to(root)}")
        total_blocks += res.blocks
        failures.extend(res.failures)

    if total_blocks == 0:
        print("no <!-- doctest --> blocks found; is --root right?")
        return 1
    for f in failures:
        print(f"\n{f}")
    print(f"\n{total_blocks} block(s) checked, {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
