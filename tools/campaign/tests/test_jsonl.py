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
