"""Command-line toxicity assessment: `python -m tools.tox`.

`tools/tox` has had a provider architecture, structural-alert screening and
two web predictors for a while, and no way to run any of it from a shell. A
library with no entry point gets used by the code that already imports it and
by nobody else.

    python -m tools.tox "CC(=O)Oc1ccccc1C(=O)O"
    python -m tools.tox --offline aspirin.smi caffeine.smi
    python -m tools.tox --json --offline "CCO" > tox.json

Exit status (highest-precedence first; one number per run):

    1  usage or input error: bad flags, a duplicate label, no molecules, or a
       SMILES the local screen could not parse.
    2  a REQUIRED check did not run: the local screen itself failed (or a
       provider raised -- a contract violation), or `--require-online` was
       given and an online provider was unavailable.
    4  structural alerts found (`alert_total_count` > 0 for any molecule) --
       only with `--fail-on-alerts`; without it, alerts are reported, not
       signalled, exactly as before.
    3  online checks unavailable: the local screen ran and is reported in
       full, but at least one online provider could not answer (outage, HTTP
       error, rate limit, timeout). Each one is named, with its reason.
    0  clean: every molecule assessed by every provider that was asked to.

A provider outage is NOT folded into success ("no alerts found" and "the check
did not run" look alike and mean opposite things), but it is also not a usage
error or a screen verdict: it has its own status, 3. A caller that needs the
online predictions passes `--require-online` and gets a hard 2 instead.
`protox3` never contributes endpoints (no JSON API; see `web.py`), so its
state is reported as `unsupported` and never changes the exit status.
Before 2026-09-23 the online mode exited 2 on every run: ADMETlab's documented
path 404s and the ProTox stub's by-design note was counted as a failure.

Every web call has a wall-clock deadline (`--timeout`, default
20 s), and a provider that does not answer at all is not
retried for the rest of the batch, so an outage costs at most one timeout per
provider per run.

`--offline` skips the web providers. It is the right default for a batch:
the local RDKit screen needs no network, and a firewalled run that silently
returns fewer endpoints would read as a cleaner molecule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.tox.assess import assess_many, default_providers
from tools.tox.model import (
    STATUS_OK,
    STATUS_UNAVAILABLE,
    STATUS_UNSUPPORTED,
    ToxAssessment,
)

EXIT_CLEAN = 0
EXIT_INPUT_ERROR = 1
EXIT_REQUIRED_CHECK_FAILED = 2
EXIT_ONLINE_UNAVAILABLE = 3
EXIT_ALERTS = 4
DEFAULT_TIMEOUT = 20.0

_EPILOG = (
    "exit status: 0 = clean (every requested provider answered); "
    "1 = usage/input error (bad flag, unparseable SMILES, duplicate label); "
    "2 = a required check did not run (local screen failed, or "
    "--require-online and an online provider was unavailable); "
    "3 = online checks unavailable (local screen complete and reported; "
    "the unavailable provider is named); "
    "4 = structural alerts found (only with --fail-on-alerts). "
    "Precedence when several apply: 1 > 2 > 4 > 3 > 0."
)


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error, which is this CLI's "a required
    check did not run". A typo'd flag is a usage error: exit 1."""

    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_INPUT_ERROR, f"{self.prog}: error: {message}\n")


def _read_inputs(items: list[str]) -> dict[str, str]:
    """Map label -> SMILES from literal SMILES or `.smi` files.

    A `.smi` line is `<smiles>[ <label>]`. A path that exists is read as a
    file; anything else is treated as a literal SMILES, so a mistyped filename
    becomes an unparseable-SMILES error rather than being silently skipped.

    A duplicate label is a hard error rather than a silent overwrite: `dict`
    assignment would keep only the LAST SMILES for a repeated label, so
    `assess_many` would never see the earlier molecule at all, and the CLI
    could exit 0 having assessed fewer molecules than were requested.
    """
    out: dict[str, str] = {}
    for n, item in enumerate(items, 1):
        p = Path(item)
        if p.exists() and p.suffix in {".smi", ".smiles", ".txt"}:
            for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                smi = parts[0]
                label = parts[1].strip() if len(parts) > 1 else f"{p.stem}:{ln}"
                if label in out and out[label] != smi:
                    raise ValueError(
                        f"duplicate label {label!r} ({p}:{ln}): already mapped "
                        f"to {out[label]!r}, now given {smi!r}"
                    )
                out[label] = smi
        else:
            label = f"input{n}" if len(items) > 1 else "molecule"
            if label in out and out[label] != item:
                raise ValueError(
                    f"duplicate label {label!r}: already mapped to "
                    f"{out[label]!r}, now given {item!r}"
                )
            out[label] = item
    return out


def _as_dict(a: ToxAssessment) -> dict:
    return {
        "label": a.label,
        "smiles": a.smiles,
        "endpoints": [
            {
                "name": e.name,
                "value": e.value,
                "units": e.units,
                "higher_is_worse": e.higher_is_worse,
                "source": e.source,
                "note": e.note,
            }
            for e in a.endpoints
        ],
        "provider_errors": a.provider_errors,
        "provider_status": a.provider_status,
    }


def _print_human(a: ToxAssessment) -> None:
    head = f"{a.label}  {a.smiles}" if a.label else a.smiles
    print(f"\n{head}")
    print("-" * min(len(head), 78))

    known = [e for e in a.endpoints if e.known]
    if not known:
        print("  no endpoint returned a value")
    for e in sorted(known, key=lambda e: (e.source, e.name)):
        # Flag the direction explicitly. A bare number invites the reader to
        # assume higher is worse, which is false for half of these.
        arrow = "higher=worse" if e.higher_is_worse else "higher=better"
        val = f"{e.value:.4g}" if isinstance(e.value, float) else str(e.value)
        line = f"  {e.name:<34} {val:>10} {e.units:<12} [{arrow}] {e.source}"
        print(line)
        if e.note:
            print(f"      {e.note}")

    unknown = [e for e in a.endpoints if not e.known]
    if unknown:
        names = ", ".join(sorted({e.name for e in unknown}))
        print(f"  unknown ({len(unknown)}): {names}")

    for prov, err in a.provider_errors.items():
        status = a.provider_status.get(prov, "error")
        print(f"  !! {prov} [{status}]: {err}")


def _has_alerts(a: ToxAssessment) -> bool:
    e = a.endpoint("alert_total_count")
    return e is not None and e.value is not None and e.value > 0


def main(argv: list[str] | None = None) -> int:
    ap = _Parser(
        prog="python -m tools.tox",
        description="Structural-alert and predicted-toxicity readouts for one "
        "or more molecules.",
        epilog=_EPILOG,
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        metavar="SMILES|FILE",
        help="literal SMILES, or a .smi/.smiles/.txt file of '<smiles> [label]' lines",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--offline",
        action="store_true",
        help="local RDKit screen only; skip the web predictors (no network)",
    )
    mode.add_argument(
        "--require-online",
        action="store_true",
        help="treat an unavailable online provider as a hard failure (exit 2) "
        "instead of a degraded run (exit 3)",
    )
    ap.add_argument(
        "--fail-on-alerts",
        action="store_true",
        help="exit 4 when any molecule has a structural alert",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help="wall-clock limit per web request (default %(default)g); a "
        "provider that does not answer is not retried for the rest of the run",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    # Return, rather than raise, argparse's exit so `main()` has one exit
    # channel for programmatic callers (and --help still returns 0).
    try:
        args = ap.parse_args(argv)
        if not args.timeout > 0:
            ap.error(f"--timeout must be positive, got {args.timeout:g}")
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else EXIT_INPUT_ERROR

    try:
        inputs = _read_inputs(args.inputs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not inputs:
        print("no molecules to assess", file=sys.stderr)
        return EXIT_INPUT_ERROR

    providers = default_providers(include_web=not args.offline, timeout=args.timeout)
    online = {p.name for p in providers if getattr(p, "online", False)}
    results = assess_many(inputs, providers=providers)

    if args.json:
        json.dump([_as_dict(r) for r in results], sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        for r in results:
            _print_human(r)

    # Input error: the LOCAL screen produced nothing for a molecule. Testing
    # "no endpoints at all" is not enough once web providers run -- a service
    # may return value=None rows for a SMILES RDKit rejected.
    unparseable = [
        r
        for r in results
        if not any(e.known for e in r.endpoints if e.source not in online)
    ]
    if unparseable:
        for r in unparseable:
            print(
                f"could not assess {r.label or r.smiles!r}: the local screen "
                "returned no endpoint (usually an unparseable SMILES)",
                file=sys.stderr,
            )
        return EXIT_INPUT_ERROR

    broken: dict[str, str] = {}  # offline provider or contract violation
    unavailable: dict[str, str] = {}  # online provider outage
    for r in results:
        for prov, status in r.provider_status.items():
            if status in (STATUS_OK, STATUS_UNSUPPORTED):
                continue
            reason = r.provider_errors.get(prov, status)
            if prov in online and status == STATUS_UNAVAILABLE:
                unavailable.setdefault(prov, reason)
            else:
                broken.setdefault(prov, f"[{status}] {reason}")

    for prov, reason in broken.items():
        print(f"error: required provider {prov} failed: {reason}", file=sys.stderr)
    for prov, reason in unavailable.items():
        print(
            f"{'error' if args.require_online else 'warning'}: online provider "
            f"{prov} unavailable: {reason}",
            file=sys.stderr,
        )
    if unavailable and not args.require_online:
        print(
            "warning: online checks incomplete; the local screen results above "
            "are complete (exit 3). Pass --require-online to make this fatal.",
            file=sys.stderr,
        )

    if broken or (unavailable and args.require_online):
        return EXIT_REQUIRED_CHECK_FAILED
    if args.fail_on_alerts and any(_has_alerts(r) for r in results):
        return EXIT_ALERTS
    if unavailable:
        return EXIT_ONLINE_UNAVAILABLE
    return EXIT_CLEAN


if __name__ == "__main__":
    raise SystemExit(main())
