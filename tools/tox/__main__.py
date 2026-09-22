"""Command-line toxicity assessment: `python -m tools.tox`.

`tools/tox` has had a provider architecture, structural-alert screening and
two web predictors for a while, and no way to run any of it from a shell. A
library with no entry point gets used by the code that already imports it and
by nobody else.

    python -m tools.tox "CC(=O)Oc1ccccc1C(=O)O"
    python -m tools.tox --offline aspirin.smi caffeine.smi
    python -m tools.tox --json --offline "CCO" > tox.json

Exit status is 0 when every requested molecule was assessed, 1 when a SMILES
could not be parsed, and 2 when a provider failed. A provider failure is NOT
folded into success: "no alerts found" and "the alert screen did not run" look
identical in the output otherwise, and they mean opposite things.

`--offline` skips the web providers. It is the right default for a batch:
the local RDKit screen needs no network, and a firewalled run that silently
returns fewer endpoints would read as a cleaner molecule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.tox.assess import assess_many
from tools.tox.model import ToxAssessment


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
        print(f"  !! {prov}: {err}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tools.tox",
        description="Structural-alert and predicted-toxicity readouts for one "
        "or more molecules.",
        epilog="Exit 0 = all assessed, 1 = a SMILES failed to parse, "
        "2 = a provider failed.",
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        metavar="SMILES|FILE",
        help="literal SMILES, or a .smi/.smiles/.txt file of '<smiles> [label]' lines",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="local RDKit screen only; skip the web predictors (no network)",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    try:
        inputs = _read_inputs(args.inputs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not inputs:
        print("no molecules to assess", file=sys.stderr)
        return 1

    results = assess_many(inputs, include_web=not args.offline)

    if args.json:
        json.dump([_as_dict(r) for r in results], sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        for r in results:
            _print_human(r)

    # An unparseable SMILES yields NO endpoints. Note the providers do report
    # an error for it ("returned no endpoints and reported no error"), so
    # testing for an empty `provider_errors` never fires -- the first version
    # of this check was dead code that happened to return the right status.
    unparseable = [r for r in results if not r.endpoints]
    if unparseable:
        for r in unparseable:
            print(
                f"could not assess {r.label or r.smiles!r}: no provider "
                "returned an endpoint (usually an unparseable SMILES)",
                file=sys.stderr,
            )
        return 1
    if any(r.provider_errors for r in results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
