#!/usr/bin/env python3
"""Check a ferric release's files before they are uploaded anywhere.

Standard library only. Run by the ``release-check`` job of
.github/workflows/wheels.yml on every run: with the tag's version on a tag
(``publish`` and ``github-release`` need that job), and with the
pyproject.toml placeholder version on nightly/dispatch runs, so the checks
pass on real files before a tag ever depends on them.

Usage:

    release-check.py check --version 0.1.0rc8 --pypi-dir dist/pypi \\
        [--largeam-dir dist/largeam --licenses-dir dist/licenses \\
         --build-tag 1largeam] [--sha256sums SHA256SUMS] [--summary FILE]
    release-check.py notes --version 0.1.0rc8 --build-tag 1largeam
    release-check.py pyproject-version pyproject.toml

``check`` exits 1 with one ``::error::`` line per problem, after checking
everything (so one run reports every problem, not the first). ``--summary``
appends a size table in Markdown (pass ``$GITHUB_STEP_SUMMARY``).

What ``check`` asserts, for every file:

- PyPI set: exactly one wheel per CPython 3.10-3.13, named
  ``ferric-<version>-cp3NN-cp3NN-<platform>.whl`` with NO build tag and a
  ``manylinux_2_28_x86_64`` platform; its METADATA says ``Name: ferric`` and
  ``Version: <version>``; its WHEEL file has no ``Build:`` line; it bundles no
  libint2 shared library (the small libint2 is static); it is under PyPI's
  100 MiB per-file limit. Exactly one sdist, ``ferric-<version>.tar.gz``,
  whose PKG-INFO says the same version, also under 100 MiB. No file name
  contains ``mpi``.
- Large-libint set (when ``--largeam-dir`` is given): the same four CPython
  tags, each named ``ferric-<version>-<build-tag>-cp3NN-...``, whose WHEEL
  says ``Build: <build-tag>``, whose METADATA has the same name and version,
  which bundles exactly one ``ferric.libs/libint2*`` shared library, and which
  is under GitHub's 2 GiB per-asset limit.
- Licence texts (``--licenses-dir``, required with ``--largeam-dir``):
  exactly ``libint2-LICENSE.txt`` and ``libint2-COPYING.LESSER.txt`` (libint2
  is LGPL-3.0-only), each containing its expected phrases.
- Every directory holds only those files: wheels and the sdist (pypi), wheels
  (largeam), ``.txt`` (licences). A subdirectory or any other file is an error.
- ``--sha256sums``: the file lists exactly the release files, each once, with
  its current sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tarfile
import zipfile
from pathlib import Path

PYPI_LIMIT = 100 * 1024 * 1024  # warehouse MAX_FILESIZE without an increase
GITHUB_ASSET_LIMIT = 2 * 1024 * 1024 * 1024  # GitHub release asset limit
PYTHON_TAGS = ("cp310", "cp311", "cp312", "cp313")
PLATFORM = "manylinux_2_28_x86_64"
NAME = "ferric"
LIBINT_SOURCE_TARBALL = "https://github.com/loriab/libint/releases/tag/v0.1"
LIBINT_FEEDSTOCK = "https://github.com/conda-forge/libint-feedstock"
# From the conda package's info/recipe/meta.yaml; re-hashed from the download.
LIBINT_SOURCE_SHA256 = "324b578f795d1f406f37a1d936f04ea51a85eae1b2ea37b5cf773afb4e25ab14"  # pragma: allowlist secret
# The two licence texts, each with phrases it must contain.
LICENSE_FILES = {
    "libint2-LICENSE.txt": ("Libint", "GNU Lesser General Public License"),
    "libint2-COPYING.LESSER.txt": ("GNU LESSER GENERAL PUBLIC LICENSE", "Version 3"),
}
LIBINT_CONDA_PACKAGE = (
    "https://conda.anaconda.org/conda-forge/linux-64/libint-2.13.1-h83f5b4b_0.conda"
)


class Report:
    """Collects problems so one run lists all of them."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.rows: list[tuple[str, str, int, int]] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def size(self, path: Path, variant: str, limit: int) -> None:
        size = path.stat().st_size
        self.rows.append((path.name, variant, size, limit))
        if size >= limit:
            self.error(f"{path.name} is {size} bytes, not under the {limit}-byte limit")


def parse_wheel_name(name: str) -> dict[str, str] | None:
    """Split a wheel file name (PEP 427) into its fields, or None."""
    if not name.endswith(".whl"):
        return None
    parts = name[: -len(".whl")].split("-")
    if len(parts) == 5:
        dist, version, py, abi, plat = parts
        build = ""
    elif len(parts) == 6:
        dist, version, build, py, abi, plat = parts
    else:
        return None
    return {
        "dist": dist,
        "version": version,
        "build": build,
        "py": py,
        "abi": abi,
        "plat": plat,
    }


def dist_info_file(zf: zipfile.ZipFile, version: str, leaf: str) -> str | None:
    """Text of ``ferric-<version>.dist-info/<leaf>`` inside a wheel, or None."""
    member = f"{NAME}-{version}.dist-info/{leaf}"
    try:
        return zf.read(member).decode("utf-8")
    except KeyError:
        return None


def header_values(text: str, key: str) -> list[str]:
    """Values of an RFC 822 style ``Key: value`` header (METADATA, WHEEL)."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            break  # headers end at the first blank line (METADATA body follows)
        k, sep, v = line.partition(":")
        if sep and k.strip() == key:
            out.append(v.strip())
    return out


def check_wheel(
    rep: Report, path: Path, version: str, variant: str, build_tag: str
) -> str | None:
    """Check one wheel; return its CPython tag when the name parses."""
    fields = parse_wheel_name(path.name)
    if fields is None:
        rep.error(f"{path.name}: not a wheel file name")
        return None
    if fields["dist"] != NAME:
        rep.error(f"{path.name}: distribution {fields['dist']!r} is not {NAME!r}")
    if fields["version"] != version:
        rep.error(f"{path.name}: version {fields['version']!r} is not {version!r}")
    if fields["py"] not in PYTHON_TAGS or fields["abi"] != fields["py"]:
        rep.error(
            f"{path.name}: unexpected python/abi tags {fields['py']}-{fields['abi']}"
        )
    if PLATFORM not in fields["plat"].split("."):
        rep.error(
            f"{path.name}: platform {fields['plat']!r} does not include {PLATFORM}"
        )
    want_build = build_tag if variant == "largeam" else ""
    if fields["build"] != want_build:
        rep.error(
            f"{path.name}: build tag {fields['build']!r}, expected "
            f"{want_build!r} for the {variant} set"
        )

    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            metadata = dist_info_file(zf, version, "METADATA")
            wheel = dist_info_file(zf, version, "WHEEL")
    except zipfile.BadZipFile as exc:
        rep.error(f"{path.name}: not a zip archive ({exc})")
        return fields["py"]

    if metadata is None:
        rep.error(f"{path.name}: no {NAME}-{version}.dist-info/METADATA")
    else:
        if header_values(metadata, "Name") != [NAME]:
            rep.error(
                f"{path.name}: METADATA Name is {header_values(metadata, 'Name')}"
            )
        if header_values(metadata, "Version") != [version]:
            rep.error(
                f"{path.name}: METADATA Version is "
                f"{header_values(metadata, 'Version')}, not [{version!r}]"
            )
    if wheel is None:
        rep.error(f"{path.name}: no {NAME}-{version}.dist-info/WHEEL")
    else:
        build_lines = header_values(wheel, "Build")
        want = [build_tag] if variant == "largeam" else []
        if build_lines != want:
            rep.error(f"{path.name}: WHEEL Build lines {build_lines}, expected {want}")

    libint = [
        n for n in names if re.fullmatch(r"ferric\.libs/libint2[^/]*\.so[^/]*", n)
    ]
    if variant == "largeam" and len(libint) != 1:
        rep.error(
            f"{path.name}: expected one bundled ferric.libs/libint2*.so, found {libint}"
        )
    if variant == "pypi" and libint:
        rep.error(
            f"{path.name}: bundles {libint}; the PyPI wheel links the small "
            "libint2 statically"
        )

    rep.size(path, variant, GITHUB_ASSET_LIMIT if variant == "largeam" else PYPI_LIMIT)
    return fields["py"]


def check_wheel_set(
    rep: Report, wheels: list[Path], version: str, variant: str, build_tag: str
) -> None:
    tags = [check_wheel(rep, w, version, variant, build_tag) for w in wheels]
    if sorted(t for t in tags if t) != sorted(PYTHON_TAGS) or len(wheels) != len(
        PYTHON_TAGS
    ):
        rep.error(
            f"{variant} set: expected one wheel each for {', '.join(PYTHON_TAGS)}; "
            f"found {sorted(w.name for w in wheels)}"
        )


def check_sdist(rep: Report, sdists: list[Path], version: str) -> None:
    want = f"{NAME}-{version}.tar.gz"
    if [p.name for p in sdists] != [want]:
        rep.error(
            f"expected exactly one sdist {want}, found {[p.name for p in sdists]}"
        )
        return
    path = sdists[0]
    try:
        with tarfile.open(path, "r:gz") as tf:
            member = tf.extractfile(f"{NAME}-{version}/PKG-INFO")
            pkg_info = member.read().decode("utf-8") if member else ""
    except (tarfile.TarError, KeyError, OSError) as exc:
        rep.error(f"{path.name}: cannot read {NAME}-{version}/PKG-INFO ({exc})")
        pkg_info = None
    if pkg_info is not None and header_values(pkg_info, "Version") != [version]:
        rep.error(
            f"{path.name}: PKG-INFO Version is {header_values(pkg_info, 'Version')}"
        )
    rep.size(path, "sdist", PYPI_LIMIT)


def write_summary(rep: Report, summary: Path) -> None:
    lines = [
        "### Release files",
        "",
        "| File | Set | Bytes | MiB | % of limit | Limit |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, variant, size, limit in rep.rows:
        label = "GitHub 2 GiB" if limit == GITHUB_ASSET_LIMIT else "PyPI 100 MiB"
        lines.append(
            f"| `{name}` | {variant} | {size:,} | {size / 2**20:.1f} | "
            f"{100 * size / limit:.1f}% | {label} |"
        )
    lines.append("")
    if rep.errors:
        lines.append(f"**{len(rep.errors)} problem(s)** -- see the job log.")
    else:
        lines.append("All release checks passed.")
    with summary.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def list_dir(rep: Report, d: Path, allowed: tuple[str, ...]) -> list[Path]:
    """Files directly in ``d``; anything else, or a name ending in none of
    ``allowed``, is an error (a release directory holds only release files)."""
    if not d.is_dir():
        rep.error(f"{d}: not a directory")
        return []
    out = []
    for p in sorted(d.iterdir()):
        if not p.is_file():
            rep.error(
                f"{d}/{p.name}: not a regular file (directories are not released)"
            )
        elif not p.name.endswith(allowed):
            rep.error(f"{d}/{p.name}: unexpected file; allowed: {', '.join(allowed)}")
        else:
            out.append(p)
    return out


def check_licenses(rep: Report, d: Path) -> list[Path]:
    """The libint2 licence texts the large-libint wheels must ship with."""
    files = list_dir(rep, d, (".txt",))
    names = sorted(p.name for p in files)
    if names != sorted(LICENSE_FILES):
        rep.error(f"{d}: expected exactly {sorted(LICENSE_FILES)}, found {names}")
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        for needle in LICENSE_FILES.get(p.name, ()):
            if needle not in text:
                rep.error(f"{p.name}: does not contain {needle!r}")
    return files


def check_sha256sums(rep: Report, sums: Path, files: list[Path]) -> None:
    """``sums`` (sha256sum format) lists exactly ``files``, by name, with
    their current hashes."""
    listed: dict[str, str] = {}
    for n, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        m = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if not m:
            rep.error(f"{sums.name}:{n}: not a `<sha256>  <file name>` line: {line!r}")
            continue
        if m.group(2) in listed:
            rep.error(f"{sums.name}: {m.group(2)} listed twice")
        listed[m.group(2)] = m.group(1)
    actual = {p.name: p for p in files}
    if sorted(listed) != sorted(actual):
        rep.error(
            f"{sums.name} lists {sorted(listed)}, but the release files are "
            f"{sorted(actual)}"
        )
    for name, digest in listed.items():
        if name in actual and sha256(actual[name]) != digest:
            rep.error(f"{sums.name}: hash of {name} does not match the file")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_check(args: argparse.Namespace) -> int:
    rep = Report()
    version = args.version
    if args.build_tag and not re.fullmatch(r"[0-9][A-Za-z0-9_]*", args.build_tag):
        rep.error(f"build tag {args.build_tag!r} must start with a digit (PEP 427)")
    pypi_dir = Path(args.pypi_dir)
    pypi_files = list_dir(rep, pypi_dir, (".whl", ".tar.gz"))
    large_files: list[Path] = []
    if args.largeam_dir:
        large_files = list_dir(rep, Path(args.largeam_dir), (".whl",))
    license_files: list[Path] = []
    if args.licenses_dir:
        if not args.largeam_dir:
            rep.error("--licenses-dir belongs with --largeam-dir (they ship together)")
        license_files = check_licenses(rep, Path(args.licenses_dir))
    elif args.largeam_dir:
        rep.error("--largeam-dir needs --licenses-dir: the wheel bundles LGPL libint2")
    for p in pypi_files + large_files:
        if "mpi" in p.name.lower():
            rep.error(
                f"{p.name}: an MPI-variant file is never released from this workflow"
            )

    check_wheel_set(
        rep,
        [p for p in pypi_files if p.name.endswith(".whl")],
        version,
        "pypi",
        args.build_tag,
    )
    check_sdist(rep, [p for p in pypi_files if p.name.endswith(".tar.gz")], version)
    if args.largeam_dir:
        if not args.build_tag:
            rep.error("--largeam-dir needs --build-tag")
        else:
            check_wheel_set(rep, large_files, version, "largeam", args.build_tag)
    if args.sha256sums:
        check_sha256sums(
            rep, Path(args.sha256sums), pypi_files + large_files + license_files
        )

    if args.summary:
        write_summary(rep, Path(args.summary))
    for name, variant, size, limit in rep.rows:
        print(f"{variant:8s} {size:>12,d} bytes {100 * size / limit:5.1f}%  {name}")
    for msg in rep.errors:
        print(f"::error::{msg}")
    if rep.errors:
        return 1
    print(f"release check OK: version {version}, {len(rep.rows)} files")
    return 0


def cmd_notes(args: argparse.Namespace) -> int:
    v = args.version
    tag = args.build_tag
    print(
        f"""ferric {v}.

## Files

- **PyPI wheels and sdist** (`ferric-{v}-cp3NN-...whl`, `ferric-{v}.tar.gz`):
  the same files `pip install ferric=={v}` downloads. Their libint2 is
  ferric's small 2.7.2 export, linked statically.
- **Large-libint wheels** (`ferric-{v}-{tag}-cp3NN-...whl`): the same version
  and commit, built against conda-forge's libint2 2.13.1, which is bundled in
  the wheel as a shared library. It adds l = 7 shells, analytic 3- and
  2-centre second derivatives, multipoles above order 2 and G12 (F12)
  integrals. It is not a strict superset: its one-electron integrals go to
  l = 7 / 6 / 3 (energy / 1st / 2nd derivative) where the PyPI wheel's go to
  6 / 6 / 6. Analytic 4-centre Hessians stay limited to f functions in both
  wheels, and RI Hessians already work with the PyPI wheel. Install it with
  `pip install --force-reinstall --no-deps <asset URL>`; `pip install -U
  ferric` later replaces it with the PyPI wheel. Check which one is loaded
  with `python -c "import ferric; print(ferric.build_info()['libint_max_am'])"`
  (`eri3` is `[7, 7, 4]` for this wheel, `[6, 6]` for the PyPI one).
- `libint2-LICENSE.txt`, `libint2-COPYING.LESSER.txt`: the licence texts of
  the libint2 the large-libint wheels bundle, from its conda-forge package.
- `SHA256SUMS`: `sha256sum -c SHA256SUMS` checks every file above.

## libint2 source (LGPL-3.0-only)

The large-libint wheels contain a libint2 binary built by conda-forge. Its
source and build recipe:

- generated library source: {LIBINT_SOURCE_TARBALL}
  (`libint-2.13.1-7-6-3-7-7-4_mm10f12ob2_0.tgz`, sha256
  `{LIBINT_SOURCE_SHA256}`)
- conda-forge recipe: {LIBINT_FEEDSTOCK}
- the exact package bundled: {LIBINT_CONDA_PACKAGE}
  (sha256 `d5219aae3bdd9629c6cff471f129f3aebbaddb058d0d19fbbab12c28924a0535`)

The PyPI wheels' libint2 export is regenerated by
`scripts/generate-libint-small.sh`; its source tarball is the
`libint-2.7.2-ferric-small-1` release of this repository.
"""
    )
    return 0


def read_pyproject_version(path: Path) -> str:
    """The first ``version = "..."`` line, as the workflow's stamp edits it."""
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r'version = "([^"]+)"', line)
        if m:
            return m.group(1)
    raise SystemExit(f'::error::no `version = "..."` line in {path}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="check release files")
    p.add_argument("--version", required=True)
    p.add_argument("--pypi-dir", required=True)
    p.add_argument("--largeam-dir")
    p.add_argument("--build-tag", default="")
    p.add_argument("--licenses-dir", help="the libint2 licence texts")
    p.add_argument("--sha256sums", help="a SHA256SUMS file that must cover every file")
    p.add_argument("--summary")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("notes", help="print the GitHub release notes")
    p.add_argument("--version", required=True)
    p.add_argument("--build-tag", required=True)
    p.set_defaults(func=cmd_notes)

    p = sub.add_parser("pyproject-version", help="print pyproject.toml's version")
    p.add_argument("path")
    p.set_defaults(func=lambda a: print(read_pyproject_version(Path(a.path))) or 0)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
