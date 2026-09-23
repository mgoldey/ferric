"""A long run must leave readable, complete-up-to-the-last-line output.

The failure this prevents: an eight-pose SCF sweep is ~12 minutes, and a
`json.dump` at the end means a crash in pose 7 discards poses 0-6 — which
were correct and expensive.
"""

from __future__ import annotations

import json

import pytest

from tools.campaign.jsonl import (
    META_KEY,
    JsonlWriter,
    iter_jsonl,
    read_jsonl,
    to_json,
)


def test_rows_are_readable_BEFORE_the_writer_closes(tmp_path):
    """The whole point: a running job is inspectable."""
    p = tmp_path / "run.jsonl"
    w = JsonlWriter(p, meta={"basis": "sto-3g"})
    w.append({"pose": 0, "energy": -1.0})
    w.append({"pose": 1, "energy": -2.0})

    rows = read_jsonl(p)  # writer still OPEN
    assert [r["pose"] for r in rows] == [0, 1]
    w.append({"pose": 2, "energy": -3.0})
    assert len(read_jsonl(p)) == 3
    w.close()


def test_a_crash_keeps_every_row_written_so_far(tmp_path):
    p = tmp_path / "run.jsonl"
    with pytest.raises(RuntimeError):
        with JsonlWriter(p) as w:
            w.append({"pose": 0})
            w.append({"pose": 1})
            raise RuntimeError("SCF blew up on pose 2")
    assert [r["pose"] for r in read_jsonl(p)] == [0, 1]


def test_a_truncated_final_line_is_skipped_not_raised(tmp_path):
    """A file mid-write ends mid-line; that is normal, not corruption."""
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p) as w:
        w.append({"pose": 0})
        w.append({"pose": 1})
    with p.open("a") as fh:
        fh.write('{"pose": 2, "ener')  # killed mid-row

    assert [r["pose"] for r in read_jsonl(p)] == [0, 1]
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(p, strict=True)


def test_meta_is_not_mistaken_for_a_data_row(tmp_path):
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p, meta={"basis": "sto-3g", "n_charges": 6458}) as w:
        w.append({"pose": 0})

    assert read_jsonl(p) == [{"pose": 0}]
    full = read_jsonl(p, include_meta=True)
    assert full[0][META_KEY]["basis"] == "sto-3g"
    assert "started" in full[0][META_KEY]


def test_a_non_dict_row_is_REFUSED(tmp_path):
    """A bare value has no field names, so it cannot be read back usefully."""
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p) as w:
        with pytest.raises(TypeError, match="must be dicts"):
            w.append([1, 2, 3])


def test_a_non_dict_row_is_REFUSED_on_read(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"a": 1}\n["pose", 1]\n{"a": 2}\n', encoding="utf-8")
    with pytest.raises(TypeError, match=r":2: .*got list"):
        read_jsonl(p)


def test_append_mode_does_not_duplicate_the_meta_header(tmp_path):
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p, meta={"run": 1}) as w:
        w.append({"pose": 0})
    with JsonlWriter(p, meta={"run": 2}, append=True) as w:
        w.append({"pose": 1})

    everything = read_jsonl(p, include_meta=True)
    assert sum(1 for r in everything if META_KEY in r) == 1
    assert [r["pose"] for r in read_jsonl(p)] == [0, 1]


def test_append_mode_on_a_MISSING_file_still_writes_meta(tmp_path):
    p = tmp_path / "fresh.jsonl"
    with JsonlWriter(p, meta={"run": 1}, append=True) as w:
        w.append({"pose": 0})
    assert sum(1 for r in read_jsonl(p, include_meta=True) if META_KEY in r) == 1


def test_to_json_produces_the_same_rows(tmp_path):
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p, meta={"basis": "sto-3g"}) as w:
        for i in range(4):
            w.append({"pose": i, "energy": -float(i)})

    out = to_json(p)
    assert out.suffix == ".json"
    assert json.loads(out.read_text()) == read_jsonl(p)
    assert not out.with_suffix(".json.tmp").exists()  # no temp left behind


def test_iter_does_not_load_the_whole_file(tmp_path):
    """Streaming out, as well as in."""
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p, flush=False) as w:
        for i in range(500):
            w.append({"i": i})
    it = iter_jsonl(p)
    assert next(it)["i"] == 0  # first row without reading 500
    assert next(it)["i"] == 1


def test_a_missing_file_raises_rather_than_reading_as_empty(tmp_path):
    """Empty results and a missing run are different, and must look different."""
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "never_ran.jsonl")


def test_resume_after_a_kill_mid_row_does_not_fuse_two_rows(tmp_path):
    """A process killed mid-write leaves no trailing newline.

    Appending onto that fuses the partial row and the next object into one
    invalid line, destroying BOTH.
    """
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p) as w:
        w.append({"pose": 0})
    with p.open("a") as fh:
        fh.write('{"pose": 1, "ener')  # killed here

    with JsonlWriter(p, append=True) as w:
        w.append({"pose": 2})

    poses = [r["pose"] for r in read_jsonl(p)]
    assert poses == [0, 2], f"the fragment must be dropped, not fused: {poses}"


def test_resume_keeps_a_COMPLETE_row_that_merely_lacks_its_newline(tmp_path):
    """Whole object, missing newline: it is data and must survive."""
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p) as w:
        w.append({"pose": 0})
    with p.open("a") as fh:
        fh.write('{"pose": 1}')  # complete, unterminated

    with JsonlWriter(p, append=True) as w:
        w.append({"pose": 2})
    assert [r["pose"] for r in read_jsonl(p)] == [0, 1, 2]


def test_a_malformed_row_in_the_MIDDLE_is_not_silently_dropped(tmp_path):
    """Tolerance is for the live tail only.

    A corrupt newline-terminated row is a completed row. Skipping it while
    returning the rows after it hands back a short result that looks whole.
    """
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p) as w:
        w.append({"pose": 0})
        w.append({"pose": 1})
    text = p.read_text().split("\n")
    text[1] = '{"pose": 0, "brok'  # corrupt a row that IS newline-terminated
    p.write_text("\n".join(text))

    with pytest.raises(json.JSONDecodeError):
        read_jsonl(p)


def test_tail_repair_does_not_rewrite_durable_rows(tmp_path):
    """The repair must touch only the tail.

    `write_text` truncates the whole file first, so a live reader sees an
    empty file and a crash mid-rewrite destroys rows that were already on
    disk. Verified by inode + prefix: the bytes before the tail must be
    untouched.
    """
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p) as w:
        for i in range(50):
            w.append({"pose": i})
    good_prefix = p.read_bytes()
    with p.open("a") as fh:
        fh.write('{"pose": 50, "par')  # killed mid-row

    with JsonlWriter(p, append=True) as w:
        w.append({"pose": 51})

    after = p.read_bytes()
    assert after.startswith(good_prefix), "the durable prefix was rewritten"
    assert [r["pose"] for r in read_jsonl(p)] == list(range(50)) + [51]


def test_tail_repair_never_opens_the_file_for_TRUNCATION(tmp_path, monkeypatch):
    """The hazard is the window, not the end state.

    A truncate-and-rewrite leaves identical bytes when it succeeds, so
    comparing the final file cannot see it. What matters is that the file is
    never opened in a mode that empties it: in that window a reader sees
    nothing and a crash loses every durable row.
    """
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p) as w:
        w.append({"pose": 0})
    with p.open("a") as fh:
        fh.write('{"pose": 1, "par')

    from pathlib import Path as _P

    real_open, real_write_text = _P.open, _P.write_text

    def guarded_open(self, mode="r", *a, **k):
        if self == p and ("w" in mode or "+" in mode and "r" not in mode):
            raise AssertionError(f"repair opened {p.name} with mode {mode!r}")
        return real_open(self, mode, *a, **k)

    def guarded_write_text(self, *a, **k):
        if self == p:
            raise AssertionError("repair used write_text(): that truncates")
        return real_write_text(self, *a, **k)

    monkeypatch.setattr(_P, "open", guarded_open)
    monkeypatch.setattr(_P, "write_text", guarded_write_text)
    with JsonlWriter(p, append=True) as w:
        w.append({"pose": 2})


def test_reader_streams_rather_than_loading_the_file(tmp_path, monkeypatch):
    """Memory must not scale with the experiment.

    Asserted by forbidding the whole-file read outright: if `read_text` is
    called, the streaming contract is not being honoured.
    """
    p = tmp_path / "run.jsonl"
    with JsonlWriter(p, flush=False) as w:
        for i in range(200):
            w.append({"i": i})

    from pathlib import Path as _P

    def forbidden(*a, **k):
        raise AssertionError("iter_jsonl called read_text(); it must stream")

    monkeypatch.setattr(_P, "read_text", forbidden)
    it = iter_jsonl(p)
    assert next(it)["i"] == 0
    assert next(it)["i"] == 1
