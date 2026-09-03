#!/usr/bin/env python3
"""
Cyclomatic complexity (CC) / maintainability index (MI) regression gate.

Uses `rust-code-analysis-cli` (Mozilla's radon-equivalent for Rust; install
via `cargo install rust-code-analysis-cli`) to compute real, per-function
complexity metrics and compares them against a checked-in baseline
(scripts/complexity_baseline.json).

WHY REGRESSION-TRACKING, NOT AN ABSOLUTE THRESHOLD:
This repo's own Cargo.toml deliberately allow-lists `clippy::too_many_arguments`
specifically for numerical kernels whose argument lists mirror the physics
(e.g. mol/basis/operator/bounds/config in ff_polar.rs) -- the same reasoning
applies to complexity metrics. Several SCF/RPA kernels (solve_rhf CC=134,
solve_uhf_fockmod CC=111, solve_rohf CC=111, pdep_polarizability_becke_dynamic
CC=106) are legitimately complex iterative numerical code, already covered by
regression tests, and are explicitly NOT a target for mechanical splitting
(see docs/performance.md and the 2026-07-19 cyclomatic-complexity sweep that
split crates/ferric-cli/src/main.rs's CC=289 dispatch function but left these
untouched). An absolute CC/MI threshold would either have to be set so high
it catches nothing, or would immediately fail the gate on already-accepted,
validated code.

Instead: this gate fails only when a function's CC or MI gets WORSE than the
baseline snapshot by more than a small tolerance (a few points of noise from
metric-computation nondeterminism/tool-version drift is expected and
tolerated), OR when a genuinely NEW function appears with complexity far
above the codebase's own historical worst-case (a generous ceiling, not a
strict one -- see NEW_FUNCTION_CC_CEILING below). This catches organic
accretion (a new giant dispatch function, or an existing one growing worse
over time) without re-litigating decisions already made about existing code.

Usage:
  python3 scripts/complexity_gate.py                  # check against baseline
  python3 scripts/complexity_gate.py --update-baseline # regenerate the baseline
                                                        # (only after a deliberate,
                                                        # reviewed complexity change --
                                                        # e.g. this session's main.rs split)

Exit code: 0 = no regression, 1 = regression found, 2 = tool not installed
(soft-skip; see ci-gate.sh's handling).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE_PATH = os.path.join(ROOT, "scripts", "complexity_baseline.json")
CRATES_DIR = os.path.join(ROOT, "crates")

# Regression tolerance: baseline metrics are re-measured per-run (not
# byte-identical across tool versions/platforms), so a delta this small is
# noise, not a real regression.
CC_TOLERANCE = 3
MI_TOLERANCE = 5.0  # mi_original units are large-magnitude and coarse

# A brand-new function (not in baseline) is only flagged if its CC exceeds
# this ceiling -- deliberately set ABOVE the current worst-case numerical
# kernel (solve_rhf, CC=134) so it doesn't re-litigate already-accepted
# code, but still catches genuinely new dispatch-sprawl before it grows to
# main()'s old CC=289.
NEW_FUNCTION_CC_CEILING = 150

# Paths excluded from the scan -- tests/examples/build scripts are not
# production code paths this gate cares about.
EXCLUDE_GLOBS = ["**/target/**", "**/tests/**", "**/examples/**", "**/build.rs"]


def check_tool_available():
    if shutil.which("rust-code-analysis-cli") is None:
        print(
            "complexity_gate.py: rust-code-analysis-cli not installed -- "
            "skipping (install via `cargo install rust-code-analysis-cli` "
            "to enable this check locally). This is a soft-skip, not a "
            "gate failure.",
            file=sys.stderr,
        )
        sys.exit(2)


def scan() -> dict:
    """Run rust-code-analysis-cli over crates/, return {qualified_name: {cc, mi}}."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "rust-code-analysis-cli",
            "-p", CRATES_DIR,
            "-m",
            "-O", "json",
            "-I", "**/*.rs",
            "-o", tmp,
        ]
        for g in EXCLUDE_GLOBS:
            cmd += ["-X", g]
        subprocess.run(cmd, check=True, capture_output=True)

        results = {}
        for dirpath, _dirs, files in os.walk(tmp):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath) as fh:
                        data = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    continue
                # Recover the real source path, RELATIVE TO THE REPO ROOT.
                #
                # rust-code-analysis-cli mirrors its input's ABSOLUTE path
                # under -o, so relpath(fpath, tmp) yields the absolute source
                # path minus its leading "/" (e.g.
                # "home/you/qc/ferric/crates/ferric-cc/src/ccd.rs"). Baking
                # that into the key made every key machine- and
                # checkout-specific: running the gate from a git worktree, a
                # clone at another path, or CI produced keys that matched
                # NOTHING in the baseline. Every function then read as
                # brand-new, and brand-new functions only fail above
                # NEW_FUNCTION_CC_CEILING -- so the gate PASSED vacuously
                # while comparing against nothing at all. It reported
                # "6102 functions checked, no regressions" from a worktree
                # whose real regression count was 16.
                #
                # Keying on the repo-relative path makes the baseline portable
                # and the comparison meaningful from any checkout. See
                # assert_baseline_is_comparable() for the guard that makes a
                # future recurrence of the vacuous-pass class loud.
                abs_src = "/" + os.path.relpath(fpath, tmp).removesuffix(".json")
                rel = os.path.relpath(abs_src, ROOT)

                def walk(node):
                    if node.get("kind") == "function":
                        metrics = node.get("metrics", {})
                        cc = metrics.get("cyclomatic", {}).get("sum")
                        mi = metrics.get("mi", {}).get("mi_original")
                        name = node.get("name") or "<anonymous>"
                        start = node.get("start_line")
                        qualified = f"{rel}::{name}@{start}"
                        if cc is not None:
                            results[qualified] = {"cc": cc, "mi": mi, "path": rel, "name": name, "line": start}
                    for child in node.get("spaces", []):
                        walk(child)

                walk(data)
        return results


def assert_baseline_is_comparable(baseline: dict, current: dict) -> None:
    """Refuse to report PASS when the baseline cannot actually be compared.

    A gate that silently compares against nothing is worse than no gate: it
    reports PASS with real regressions in the tree. That is not hypothetical
    -- it is exactly what absolute-path keys did from a worktree (see scan()).

    Both failure modes below are LOUD (exit 1) rather than a soft-skip,
    because both mean "this run measured nothing" and a soft-skip would
    reintroduce the silent pass under a different name.
    """
    if not baseline:
        return  # a genuinely empty baseline is handled by the caller

    overlap = len(set(baseline) & set(current))
    if overlap > 0:
        return

    # Zero overlap. Distinguish the known legacy cause (absolute-path keys
    # from before this fix) from the generic case, so the message is
    # actionable rather than merely alarming.
    legacy = sum(1 for k in baseline if k.startswith(("/", "home/", "Users/")))
    if legacy:
        print(
            f"complexity_gate.py: FAIL -- the baseline at {BASELINE_PATH} uses "
            f"ABSOLUTE-path keys ({legacy} of {len(baseline)}), which this "
            "version cannot compare against.\n"
            "  Those keys were machine- and checkout-specific: the gate could "
            "not match them from a worktree or another clone, matched nothing, "
            "and PASSED while comparing against nothing.\n"
            "  Regenerate with repo-relative keys:\n"
            "    python3 scripts/complexity_gate.py --update-baseline\n"
            "  and commit the result so the change is visible in review.",
            file=sys.stderr,
        )
    else:
        print(
            f"complexity_gate.py: FAIL -- the baseline at {BASELINE_PATH} has "
            f"{len(baseline)} entries and the scan found {len(current)}, but "
            "NONE of them match.\n"
            "  Refusing to report PASS from a comparison against nothing. "
            "Either the baseline is for a different tree, or the key format "
            "changed; regenerate it after review:\n"
            "    python3 scripts/complexity_gate.py --update-baseline",
            file=sys.stderr,
        )
    sys.exit(1)


def main():
    update = "--update-baseline" in sys.argv
    check_tool_available()

    current = scan()

    if update:
        # Store a stable, sorted, minimal snapshot -- diffable in review.
        snapshot = {
            k: {"cc": v["cc"], "mi": v["mi"]}
            for k, v in sorted(current.items())
        }
        with open(BASELINE_PATH, "w") as fh:
            json.dump(snapshot, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"complexity_gate.py: baseline updated -> {BASELINE_PATH} ({len(snapshot)} functions)")
        return 0

    if not os.path.exists(BASELINE_PATH):
        print(
            f"complexity_gate.py: no baseline at {BASELINE_PATH} -- "
            "run with --update-baseline first (after review). Soft-skip.",
            file=sys.stderr,
        )
        return 2

    with open(BASELINE_PATH) as fh:
        baseline = json.load(fh)

    assert_baseline_is_comparable(baseline, current)

    regressions = []
    new_functions = []

    for qualified, cur in current.items():
        if qualified in baseline:
            base = baseline[qualified]
            cc_delta = cur["cc"] - base["cc"]
            if cc_delta > CC_TOLERANCE:
                regressions.append(
                    f"  CC regression: {cur['path']}:{cur['line']} {cur['name']} "
                    f"-- CC {base['cc']:.0f} -> {cur['cc']:.0f} (+{cc_delta:.0f})"
                )
            if base.get("mi") is not None and cur.get("mi") is not None:
                mi_delta = base["mi"] - cur["mi"]  # MI dropping = worse
                if mi_delta > MI_TOLERANCE:
                    regressions.append(
                        f"  MI regression: {cur['path']}:{cur['line']} {cur['name']} "
                        f"-- MI {base['mi']:.1f} -> {cur['mi']:.1f} (-{mi_delta:.1f})"
                    )
        else:
            if cur["cc"] > NEW_FUNCTION_CC_CEILING:
                new_functions.append(
                    f"  NEW high-complexity function: {cur['path']}:{cur['line']} "
                    f"{cur['name']} -- CC={cur['cc']:.0f} (ceiling {NEW_FUNCTION_CC_CEILING})"
                )

    if regressions or new_functions:
        print("complexity_gate.py: FAIL -- complexity regression(s) found")
        for line in regressions:
            print(line)
        for line in new_functions:
            print(line)
        print()
        print(
            "If this is a deliberate, reviewed change (e.g. a function "
            "genuinely needed to grow, or you just did a real refactor that "
            "changes the shape), regenerate the baseline: "
            "python3 scripts/complexity_gate.py --update-baseline -- and "
            "commit the updated scripts/complexity_baseline.json alongside "
            "your change so the regression is visible in review, not silent."
        )
        return 1

    print(f"complexity_gate.py: PASS -- {len(current)} functions checked, no regressions vs baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
