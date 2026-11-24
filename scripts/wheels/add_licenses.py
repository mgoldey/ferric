#!/usr/bin/env python3
"""Add the bundled libraries' licence texts to a repaired wheel, in place.

    python add_licenses.py WHEEL FILE [FILE ...]

Each FILE is stored as ``<name>-<version>.dist-info/licenses/third-party/<basename>``
and RECORD gains a line for it (sha256, size). Every other member is copied
byte for byte with its original compression. Standard library only.

Used by the macOS/Windows wheel jobs (wheels-macwin.yml): their wheels carry
libint2 (LGPL-3.0-only, statically linked), libxc (MPL-2.0), OpenBLAS
(BSD-3-Clause), Eigen (MPL-2.0, headers), libecpint and Faddeeva (MIT).
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
import zipfile
from pathlib import Path


def record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    wheel = Path(argv[0])
    files = [Path(f) for f in argv[1:]]
    names = [f.name for f in files]
    if len(set(names)) != len(names):
        raise SystemExit(f"::error::duplicate licence file names: {sorted(names)}")

    with zipfile.ZipFile(wheel) as zin:
        infos = zin.infolist()
        records = [
            i.filename for i in infos if i.filename.endswith(".dist-info/RECORD")
        ]
        if len(records) != 1:
            raise SystemExit(f"::error::{wheel}: expected one RECORD, found {records}")
        record = records[0]
        dist_info = record[: -len("/RECORD")]
        dest = f"{dist_info}/licenses/third-party"
        clash = [i.filename for i in infos if i.filename.startswith(dest + "/")]
        if clash:
            raise SystemExit(f"::error::{wheel} already has {clash}")

        tmp = wheel.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w") as zout:
            record_text = zin.read(record).decode("utf-8")
            for info in infos:
                if info.filename != record:
                    zout.writestr(
                        info, zin.read(info), compress_type=info.compress_type
                    )
            added = []
            for f in files:
                data = f.read_bytes()
                arc = f"{dest}/{f.name}"
                zout.writestr(arc, data, compress_type=zipfile.ZIP_DEFLATED)
                added.append(f"{arc},{record_hash(data)},{len(data)}")
            lines = [ln for ln in record_text.splitlines() if ln]
            # RECORD lists itself last, with no hash.
            body = [ln for ln in lines if not ln.startswith(record + ",")]
            new_record = "\n".join(body + added + [f"{record},,"]) + "\n"
            zout.writestr(record, new_record, compress_type=zipfile.ZIP_DEFLATED)
    os.replace(tmp, wheel)
    for a in added:
        print("added", a.split(",")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
