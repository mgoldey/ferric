"""Append-as-you-go JSONL for long runs, so partial results survive.

Every long-running script in `experiments/` builds a list in memory and calls
`json.dump` once at the end. That has three costs, all of which were paid
during the danuglipron pocket work:

1. **A crash loses everything.** An eight-pose SCF sweep is ~12 minutes; a
   failure in pose 7 discards poses 0-6, which were correct and expensive.
2. **A running job is opaque.** There is no file to look at, so "is it working
   or is it stuck?" can only be answered by reading CPU counters.
3. **You cannot start analysing until it finishes**, which discourages looking
   at partial results early — exactly when a mistake is cheapest to catch.

JSONL fixes all three: one JSON object per line, flushed as it is written, so
the file is readable and complete-up-to-the-last-line at every instant.

    with JsonlWriter(path, meta={"basis": "sto-3g"}) as w:
        for pose in poses:
            w.append({"pose": pose.i, "energy": run(pose)})

    rows = read_jsonl(path)                     # skips the meta header

**Why not `json.dump` at the end as well?** Because then there are two files
that can disagree. `to_json` converts a finished JSONL to a single JSON array
when something downstream needs one.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

__all__ = ["JsonlWriter", "read_jsonl", "iter_jsonl", "to_json"]

#: Marks the header line that `JsonlWriter` writes first. Readers skip it, so
#: run metadata never has to be mistaken for a data row.
META_KEY = "__meta__"


class JsonlWriter:
    """Append rows to a JSONL file, flushed after every row.

    `flush=True` (the default) calls `flush()` and `os.fsync()` after each row
    so the data is on disk, not sitting in a buffer that a `kill -9` discards.
    That costs a syscall per row, which is nothing next to an SCF, and is the
    entire point for a long run.

    Set `flush=False` only for a tight loop where rows arrive faster than the
    work between them.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        meta: dict[str, Any] | None = None,
        flush: bool = True,
        append: bool = False,
    ) -> None:
        self.path = Path(path)
        self._flush = flush
        self.n_rows = 0
        self._t0 = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists() and self.path.stat().st_size > 0
        if append and existed:
            self._repair_unterminated_tail()
        self._fh = self.path.open("a" if append else "w", encoding="utf-8")
        if meta is not None and not (append and existed):
            self._write({META_KEY: {**meta, "started": time.time()}})

    def _repair_unterminated_tail(self) -> None:
        """Make an interrupted file safe to append to.

        A process killed mid-row leaves a file with no trailing newline.
        Opening that with "a" writes the next object onto the partial row and
        fuses them into one invalid line, destroying BOTH.

        A complete-but-unterminated final object just needs its newline. An
        incomplete fragment is truncated -- it was never a row.
        """
        # Modify only the TAIL, in binary mode. `write_text` truncates the
        # whole file before rewriting it, so a live reader can observe an
        # empty file and a second interruption mid-rewrite destroys rows that
        # were already durable. Appending a newline, or truncating back to the
        # last complete one, touches nothing that is already correct.
        with self.path.open("r+b") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size == 0:
                return
            fh.seek(size - 1)
            if fh.read(1) == b"\n":
                return  # already well-formed

            window = min(size, 1 << 20)
            fh.seek(size - window)
            chunk = fh.read(window)
            nl = chunk.rfind(b"\n")
            if nl < 0 and window < size:
                # Final row exceeds the window; leave the file untouched
                # rather than guessing where it starts.
                return
            tail = chunk[nl + 1 :]

            try:
                json.loads(tail.decode("utf-8"))
                complete = True
            except (json.JSONDecodeError, UnicodeDecodeError):
                complete = False

            if complete:
                fh.seek(0, os.SEEK_END)
                fh.write(b"\n")  # whole row, just unterminated
            else:
                fh.truncate(size - len(tail))  # drop the fragment only
            fh.flush()
            os.fsync(fh.fileno())

    def _write(self, obj: dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj, default=str) + "\n")
        if self._flush:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def append(self, row: dict[str, Any]) -> None:
        """Write one row. Raises on a non-dict so a bad row fails here."""
        if not isinstance(row, dict):
            raise TypeError(
                f"JSONL rows must be dicts, got {type(row).__name__} -- a bare "
                "value cannot carry the field names a reader needs"
            )
        self._write(row)
        self.n_rows += 1

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def iter_jsonl(
    path: str | Path, *, include_meta: bool = False, strict: bool = False
) -> Iterator[dict[str, Any]]:
    """Yield rows from a JSONL file, tolerating a truncated final line.

    A file being written right now usually ends mid-line. That is normal and
    is not corruption, so the partial tail is skipped rather than raised --
    which is what makes it safe to read a run in progress. `strict=True`
    raises instead, for the case where the run is known to be finished.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no JSONL at {p}")
    # Only an UNTERMINATED final line may be skipped. A malformed line that
    # ends in a newline is a completed, corrupt row: dropping it silently
    # while returning the rows after it would hand back a short result that
    # looks whole.
    # Stream one line at a time: `read_text()` would materialise the whole
    # file before yielding the first row, so memory would scale with the
    # experiment despite the streaming contract.
    with p.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            terminated = raw_line.endswith("\n")
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if not strict and not terminated:
                    return  # the writer is mid-row; stop cleanly
                raise
            # The writer refuses non-dict rows, so one here was not written by
            # it. Yielding it would break the declared row type downstream.
            if not isinstance(obj, dict):
                raise TypeError(
                    f"{p}:{lineno}: JSONL row must be a JSON object, "
                    f"got {type(obj).__name__}"
                )
            if META_KEY in obj and not include_meta:
                continue
            yield obj


def read_jsonl(
    path: str | Path, *, include_meta: bool = False, strict: bool = False
) -> list[dict[str, Any]]:
    """`iter_jsonl` as a list."""
    return list(iter_jsonl(path, include_meta=include_meta, strict=strict))


def to_json(src: str | Path, dst: str | Path | None = None) -> Path:
    """Convert a finished JSONL to a single JSON array.

    Written atomically via a temporary file plus `os.replace`, so a reader
    never sees a half-written array.
    """
    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".json")
    rows = read_jsonl(src, strict=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    os.replace(tmp, dst)
    return dst
