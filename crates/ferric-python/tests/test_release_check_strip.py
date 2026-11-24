"""release-check rejects a Linux wheel whose extension still has a .symtab.

The PyPI wheel is close to PyPI's 100 MiB limit, and the unstripped extension
carries 13 MB of symbol table (about 3 MB zipped). ``scripts/release-check.py``
reads the ELF section names of ``ferric/ferric.*.so`` straight from the wheel.

The extensions here are hand-built ELF64 files (header, section names, section
header table), so the test needs no compiler. One test also runs the system
``strip`` on a real shared object when both exist, so the byte layout the
reader assumes is checked against an independent writer.
"""

import importlib.util
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "release-check.py"
EXT = "ferric/ferric.cpython-311-x86_64-linux-gnu.so"


def _load():
    assert SCRIPT.is_file(), f"{SCRIPT} is missing -- this test selects on it"
    spec = importlib.util.spec_from_file_location("_under_test_release_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_elf(section_names):
    """A minimal little-endian ELF64 with the named sections plus .shstrtab."""
    names = [""] + list(section_names) + [".shstrtab"]
    strtab = b"\0"
    offsets = {"": 0}
    for n in names[1:]:
        offsets[n] = len(strtab)
        strtab += n.encode() + b"\0"
    shentsize = 64
    strtab_off = 64
    shoff = strtab_off + len(strtab)
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4] = 2  # ELFCLASS64
    header[5] = 1  # little endian
    struct.pack_into("<Q", header, 0x28, shoff)
    struct.pack_into("<HHH", header, 0x3A, shentsize, len(names), len(names) - 1)
    table = bytearray()
    for n in names:
        entry = bytearray(shentsize)
        struct.pack_into("<I", entry, 0, offsets[n])
        if n == ".shstrtab":
            struct.pack_into("<QQ", entry, 24, strtab_off, len(strtab))
        table += entry
    return bytes(header) + strtab + bytes(table)


def make_wheel(tmp_path, payload):
    path = tmp_path / "ferric-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(EXT, payload)
    return path


def run_check(tmp_path, payload):
    rc = _load()
    rep = rc.Report()
    rc.check_extension_stripped(rep, make_wheel(tmp_path, payload))
    return rep.errors


def test_section_names_round_trip():
    rc = _load()
    import io

    names = [".text", ".symtab", ".strtab"]
    assert rc.elf_section_names(io.BytesIO(make_elf(names))) == [
        "",
        *names,
        ".shstrtab",
    ]


def test_wheel_with_symtab_is_an_error(tmp_path):
    errors = run_check(tmp_path, make_elf([".text", ".symtab", ".strtab"]))
    assert len(errors) == 1
    assert ".symtab" in errors[0]
    assert "--strip" in errors[0]


def test_stripped_wheel_passes(tmp_path):
    assert run_check(tmp_path, make_elf([".text", ".dynsym", ".dynstr"])) == []


def test_non_elf_extension_is_an_error(tmp_path):
    errors = run_check(tmp_path, b"MZ" + b"\0" * 200)
    assert len(errors) == 1
    assert "not an ELF64" in errors[0]


def test_missing_extension_is_an_error(tmp_path):
    rc = _load()
    rep = rc.Report()
    path = tmp_path / "ferric-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ferric/__init__.py", "")
    rc.check_extension_stripped(rep, path)
    assert len(rep.errors) == 1
    assert "expected one" in rep.errors[0]


@pytest.mark.skipif(
    sys.platform != "linux"
    or shutil.which("cc") is None
    or shutil.which("strip") is None,
    reason="needs cc and strip producing ELF (Linux)",
)
def test_reader_agrees_with_system_strip(tmp_path):
    src = tmp_path / "x.c"
    src.write_text("int ferric_probe(int a) { return a + 1; }\n")
    so = tmp_path / "x.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    stripped = tmp_path / "x-stripped.so"
    subprocess.run(["strip", "-o", str(stripped), str(so)], check=True)
    assert ".symtab" in _load().elf_section_names(open(so, "rb"))
    assert ".symtab" not in _load().elf_section_names(open(stripped, "rb"))
    assert run_check(tmp_path, so.read_bytes()) != []
    assert run_check(tmp_path, stripped.read_bytes()) == []
