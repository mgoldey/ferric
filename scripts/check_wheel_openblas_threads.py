#!/usr/bin/env python3
"""Fail if the OpenBLAS bundled in an INSTALLED ferric wheel caps threads below N.

    python check_wheel_openblas_threads.py 128

OpenBLAS fixes its maximum thread count at BUILD time (``NUM_THREADS``);
left unset it is the build machine's core count. The 0.1.0rc11 Linux wheel
was built on a 4-core runner and reports ``MAX_THREADS=4``:
``openblas_set_num_threads(64)`` leaves it at 4 on any machine, so the
serial BLAS sections ferric runs at the rayon pool size (``[parallel]
blas_threads``) silently used at most 4 threads. wheels.yml now builds with
``NUM_THREADS=$OPENBLAS_MAX_THREADS``; this check runs in the smoke test so
the cap cannot come back unnoticed.

Three independent checks, all against the library inside the wheel (found
next to the extension: ``ferric.libs/`` from auditwheel/delvewheel,
``ferric/.dylibs/`` from delocate), after ``import ferric`` has loaded it:

1. ``openblas_get_config()`` reports ``MAX_THREADS=<n>`` with n >= N;
2. ``openblas_set_num_threads(N)`` then ``openblas_get_num_threads()``
   returns N. ``set_num_threads`` clamps to the compiled maximum, not to the
   machine's cores, so this holds on a 4-core CI runner;
3. after ``ferric.set_blas_threads(100000)``, ``ferric.blas_threads()`` equals
   the count OpenBLAS runs (ferric reports the clamped count, not the
   request).

Standard library only. Run it outside any ferric source tree.
"""

from __future__ import annotations

import ctypes
import glob
import os
import re
import sys


def bundled_openblas() -> str:
    import ferric  # loads the bundled OpenBLAS (as a dependency of the extension)

    pkg = os.path.dirname(os.path.abspath(ferric.__file__))
    site = os.path.dirname(pkg)
    patterns = [
        os.path.join(site, "ferric.libs", "*openblas*"),  # auditwheel, delvewheel
        os.path.join(pkg, ".dylibs", "*openblas*"),  # delocate
    ]
    found = sorted(p for pat in patterns for p in glob.glob(pat))
    if len(found) != 1:
        raise SystemExit(f"FAIL: expected one bundled OpenBLAS, found {found}")
    return found[0]


def main(argv: list[str]) -> int:
    if len(argv) != 1 or not argv[0].isdigit():
        print(__doc__, file=sys.stderr)
        return 2
    want = int(argv[0])
    path = bundled_openblas()
    lib = ctypes.CDLL(path)
    lib.openblas_get_config.restype = ctypes.c_char_p
    lib.openblas_get_num_threads.restype = ctypes.c_int
    lib.openblas_set_num_threads.argtypes = [ctypes.c_int]
    config = lib.openblas_get_config().decode()
    print(f"{os.path.basename(path)}: {config}")

    failures = []
    m = re.search(r"MAX_THREADS=(\d+)", config)
    if not m:
        failures.append(f"openblas_get_config() has no MAX_THREADS: {config!r}")
    elif int(m.group(1)) < want:
        failures.append(
            f"MAX_THREADS={m.group(1)} < {want} (built without NUM_THREADS={want}?)"
        )
    before = lib.openblas_get_num_threads()
    lib.openblas_set_num_threads(want)
    got = lib.openblas_get_num_threads()
    lib.openblas_set_num_threads(before)
    print(f"openblas_set_num_threads({want}) -> openblas_get_num_threads() = {got}")
    if got != want:
        failures.append(f"openblas_set_num_threads({want}) gave {got} threads")
    # 3. ferric's own report is what OpenBLAS runs: a request far above any
    #    cap comes back clamped (the rc11 wheel reported the request).
    import ferric

    huge = 100_000
    try:
        ferric.set_blas_threads(huge)
        reported = ferric.blas_threads()
    finally:
        ferric.set_blas_threads(None)
    lib.openblas_set_num_threads(reported)
    runs = lib.openblas_get_num_threads()
    lib.openblas_set_num_threads(before)
    print(
        f"ferric.set_blas_threads({huge}) -> ferric.blas_threads() = {reported}, OpenBLAS runs {runs}"
    )
    if reported != runs:
        failures.append(
            f"ferric.blas_threads() reports {reported}, OpenBLAS runs {runs}"
        )
    for f in failures:
        print(f"FAIL: {f}")
    if failures:
        return 1
    print(f"OK: the bundled OpenBLAS allows {want} threads")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
