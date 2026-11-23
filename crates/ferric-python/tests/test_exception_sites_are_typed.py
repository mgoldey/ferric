"""No new bare `PyValueError::new_err` / `PyRuntimeError::new_err` in the bindings.

Since roadmap 10 a failure the bindings raise as a `ValueError` goes through
`errors::input_err` (`ferric.InputError`, `ferric.BasisError`) and one raised as
a `RuntimeError` through `make_err` / `IntoPyResult::py` (`ferric.ComputationError`
and the typed subclasses), so callers can catch `ferric.FerricError`. A new
bare `new_err` call silently opts out. This scan fails on one, unless the
enclosing function is in `PLAIN` below, where plain builtins are deliberate:
the GPU / precision settings (their traceback text is pinned by subprocess tests
and they are configuration, not computation), the libint test hook and
`set_blas_threads`. `TypeError`, `IndexError`, `OSError`, `KeyError` and
`NotImplementedError` sites are not covered (they are not in the hierarchy).
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

PLAIN = {
    ("integrals.rs", "non_negative_gpu_key"),
    ("integrals.rs", "parse_kernel_list"),
    ("integrals.rs", "configure_precision"),
    ("integrals.rs", "configure_gpu"),
    ("integrals.rs", "_set_libint_max_am_for_test"),
    ("integrals.rs", "set_blas_threads"),
}
BARE = re.compile(r"\bPy(?:Value|Runtime)Error::new_err\b")
FN = re.compile(r"\bfn\s+(\w+)")


def _sites():
    for path in sorted(SRC.rglob("*.rs")):
        if path.name == "errors.rs":
            continue
        fn = None
        for line in path.read_text().splitlines():
            if m := FN.search(line):
                fn = m.group(1)
            if BARE.search(line) and not line.lstrip().startswith("//"):
                yield path.relative_to(SRC).as_posix(), fn, line.strip()


def test_the_scan_finds_the_known_plain_sites():
    """Reachability: a scan that matched nothing would pass the check below."""
    found = {(Path(f).name, fn) for f, fn, _ in _sites()}
    assert PLAIN <= found, PLAIN - found


def test_no_new_bare_value_or_runtime_error_sites():
    bad = [
        f"{f}: in fn {fn}: {line}"
        for f, fn, line in _sites()
        if (Path(f).name, fn) not in PLAIN
    ]
    assert not bad, (
        "raise ValueError through `crate::errors::input_err(..)` (ferric.InputError)\n"
        "and RuntimeError through `crate::make_err(..)` (ferric.ComputationError), or\n"
        "add the function to PLAIN with a reason:\n" + "\n".join(bad)
    )
