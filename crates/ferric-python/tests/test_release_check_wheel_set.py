"""release-check's expected wheel set: ONE cp310-abi3 wheel per platform and variant.

The release workflows build the extension once with ``--features abi3``
(pyo3 ``abi3-py310``) and smoke-test that single wheel on CPython 3.10-3.13.
These tests run ``scripts/release-check.py check`` end to end on fake release
directories (stripped ELF extension, METADATA, WHEEL, sdist, licence texts,
SHA256SUMS) so the file count, names and SHA256SUMS expectations are pinned:
the abi3 set passes, and the old one-wheel-per-interpreter set is refused.
"""

import hashlib
import importlib.util
import io
import struct
import tarfile
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "release-check.py"
VERSION = "0.1.0rc99"
TAG = "1largeam"
LINUX = "manylinux_2_28_x86_64"
OTHER = ("macosx_11_0_arm64", "win_amd64")


def _load():
    spec = importlib.util.spec_from_file_location("_under_test_release_check2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stripped_elf():
    """A minimal ELF64 with only .text and .shstrtab (no .symtab)."""
    names = ["", ".text", ".shstrtab"]
    strtab = b"\0.text\0.shstrtab\0"
    offs = {"": 0, ".text": 1, ".shstrtab": 7}
    shoff = 64 + len(strtab)
    head = bytearray(64)
    head[:4] = b"\x7fELF"
    head[4], head[5] = 2, 1
    struct.pack_into("<Q", head, 0x28, shoff)
    struct.pack_into("<HHH", head, 0x3A, 64, len(names), len(names) - 1)
    table = bytearray()
    for n in names:
        e = bytearray(64)
        struct.pack_into("<I", e, 0, offs[n])
        if n == ".shstrtab":
            struct.pack_into("<QQ", e, 24, 64, len(strtab))
        table += e
    return bytes(head) + strtab + bytes(table)


def _wheel(d, py, abi, plat, build=""):
    parts = ["ferric", VERSION] + ([build] if build else []) + [py, abi, plat]
    path = d / ("-".join(parts) + ".whl")
    di = f"ferric-{VERSION}.dist-info"
    wheel = f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\nTag: {py}-{abi}-{plat}\n"
    if build:
        wheel += f"Build: {build}\n"
    with zipfile.ZipFile(path, "w") as zf:
        ext = "ferric/ferric.abi3.so" if plat == LINUX else "ferric/ferric.pyd"
        zf.writestr(ext, _stripped_elf())
        zf.writestr(
            f"{di}/METADATA",
            f"Metadata-Version: 2.4\nName: ferric\nVersion: {VERSION}\n",
        )
        zf.writestr(f"{di}/WHEEL", wheel)
        if build:
            zf.writestr("ferric.libs/libint2-0123abcd.so", b"x")
    return path


def _sdist(d):
    path = d / f"ferric-{VERSION}.tar.gz"
    data = f"Metadata-Version: 2.4\nName: ferric\nVersion: {VERSION}\n".encode()
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(f"ferric-{VERSION}/PKG-INFO")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return path


def _release(tmp_path, tags):
    dirs = {k: tmp_path / k for k in ("pypi", "largeam", "licenses", *OTHER)}
    for d in dirs.values():
        d.mkdir()
    files = [_sdist(dirs["pypi"])]
    for py, abi in tags:
        files.append(_wheel(dirs["pypi"], py, abi, LINUX))
        files.append(_wheel(dirs["largeam"], py, abi, LINUX, TAG))
        for plat in OTHER:
            files.append(_wheel(dirs[plat], py, abi, plat))
    for name, text in {
        "libint2-LICENSE.txt": "Libint GNU Lesser General Public License",
        "libint2-COPYING.LESSER.txt": "GNU LESSER GENERAL PUBLIC LICENSE Version 3",
    }.items():
        p = dirs["licenses"] / name
        p.write_text(text)
        files.append(p)
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files
        )
    )
    argv = [
        "check", "--version", VERSION,
        "--pypi-dir", str(dirs["pypi"]), "--largeam-dir", str(dirs["largeam"]),
        "--licenses-dir", str(dirs["licenses"]), "--build-tag", TAG,
        "--sha256sums", str(sums),
    ]  # fmt: skip
    for plat in OTHER:
        argv += ["--platform-dir", f"{plat}={dirs[plat]}"]
    return argv, files


def test_abi3_release_passes_with_seven_files(tmp_path, capsys):
    argv, files = _release(tmp_path, [("cp310", "abi3")])
    # 3 PyPI wheels (Linux, macOS arm64, Windows) + sdist + 1 largeam wheel +
    # 2 licence texts: what github-release counts before adding SHA256SUMS.
    assert len(files) == 7
    assert _load().main(argv) == 0, capsys.readouterr().out


def test_per_interpreter_wheels_are_refused(tmp_path, capsys):
    tags = [(t, t) for t in ("cp310", "cp311", "cp312", "cp313")]
    argv, _ = _release(tmp_path, tags)
    assert _load().main(argv) == 1
    out = capsys.readouterr().out
    assert "unexpected python/abi tags cp311-cp311" in out
    assert "expected one wheel each for cp310-abi3" in out
