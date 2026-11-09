"""avogadro-ferric: the ferric quantum-chemistry engine inside Avogadro 2.

Avogadro runs ``avogadro-ferric <identifier> [flags] --lang <locale>`` from the
package's pixi environment and writes the input JSON to stdin (never as an
argument, whatever the plugin docs say; avogadrolibs 2.1.0
qtgui/pythonscript.cpp). The identifiers are the ``identifier`` keys of the
``[tool.avogadro]`` tables in pyproject.toml.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

__version__ = "0.1.0"

CHARGE_MODELS = {
    "ferric_lowdin": "lowdin",
    "ferric_hirshfeld": "hirshfeld",
    "ferric_chelpg": "chelpg",
}

MENU_COMMANDS = ["sp", "opt", "freq", "orbitals", "config"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avogadro-ferric")
    sub = parser.add_subparsers(dest="feature", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lang", nargs="?", default="en")
    common.add_argument("--debug", action="store_true")

    energy = sub.add_parser("ferric", parents=[common])
    energy.add_argument("--protocol", default="text")
    for ident in CHARGE_MODELS:
        p = sub.add_parser(ident, parents=[common])
        p.add_argument("--charges", action="store_true")
        p.add_argument("--potentials", action="store_true")
    sub.add_parser("ferric_input", parents=[common])
    for ident in MENU_COMMANDS:
        p = sub.add_parser(ident, parents=[common])
        p.add_argument("--user-options", action="store_true")
    return parser


def _read_json(stream) -> dict:
    text = stream.read()
    return json.loads(text) if text.strip() else {}


def run(args, stdin=None) -> dict | None:
    """Dispatch one invocation. Returns the JSON reply (None for the energy server)."""
    from . import avo, config

    stdin = sys.stdin if stdin is None else stdin
    config.apply_runtime_env(config.load())

    if args.feature == "ferric":
        if args.protocol != "text":
            raise ValueError("avogadro-ferric speaks the text energy protocol only")
        from . import energy_model

        bootstrap = json.loads(stdin.readline())
        energy_model.serve(bootstrap, stdin)
        return None

    if args.feature in MENU_COMMANDS and getattr(args, "user_options", False):
        from . import commands

        if args.feature == "config":
            return commands.configure({}, want_options=True)
        # Do not read stdin here: Avogadro may not send (or close) it for an
        # options request.
        return {"userOptions": config.job_options(config.load())}

    avo_input = _read_json(stdin)
    if args.feature in CHARGE_MODELS:
        from . import commands

        return commands.electrostatics(
            avo_input, CHARGE_MODELS[args.feature], want_potentials=args.potentials
        )
    if args.feature == "ferric_input":
        from . import inputgen

        return inputgen.generate(avo_input)

    from . import commands

    if args.feature == "sp":
        return commands.single_point(avo_input)
    if args.feature == "opt":
        return commands.optimize(avo_input)
    if args.feature == "freq":
        return commands.frequencies(avo_input)
    if args.feature == "orbitals":
        return commands.molecular_orbitals(avo_input)
    if args.feature == "config":
        return commands.configure(avo_input, want_options=False)
    avo.reply({"error": f"unknown feature {args.feature!r}"})
    return None


def main(argv: list[str] | None = None) -> int:
    from . import avo

    args = build_parser().parse_args(argv)
    avo.claim_stdout()
    try:
        out = run(args)
    except Exception as exc:
        limit = None if args.debug else 4
        message = "".join(traceback.format_exception(exc, limit=limit))
        print(message, file=sys.stderr)
        if args.feature == "ferric":
            return 1
        avo.reply({"error": f"ferric: {exc}" if not args.debug else message})
        return 0
    if out is not None:
        avo.reply(out)
    return 0
