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
import re
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
            "-p",
            CRATES_DIR,
            "-m",
            "-O",
            "json",
            "-I",
            "**/*.rs",
            "-o",
            tmp,
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
                # Recover the real source path relative to the REPO ROOT.
                #
                # rust-code-analysis-cli mirrors each input's ABSOLUTE path
                # into the output tree, so relpath(fpath, tmp) yields
                # "home/matt/qc/ferric/crates/..." -- the absolute path minus
                # its leading slash, not a repo-relative one. Baselining that
                # pins the keys to ONE checkout directory: every git worktree
                # then reports every function as NEW, and the gate fails on
                # changes that touch no Rust at all (a .config/nextest.toml
                # edit hit exactly this). Strip the repo root so the keys are
                # portable across worktrees and machines.
                rel = os.path.relpath(fpath, tmp).removesuffix(".json")
                root_key = os.path.relpath(ROOT, "/")
                if rel.startswith(root_key + os.sep):
                    rel = rel[len(root_key) + 1 :]

                # Key by STABLE identity, never by line number: the file, the
                # chain of enclosing named scopes (mod/impl/trait/fn), the
                # name, and an ordinal among same-named siblings in source
                # order (closures are `<anonymous>` and usually need it).
                #
                # Keys used to be `name@start_line`. Any edit that shifted
                # lines renamed every key below it in that file, so every PR
                # rewrote large parts of the baseline, any two PRs touching a
                # shared file (ferric-python/src/lib.rs, rhf.rs) conflicted on
                # it, and a closure landing on another closure's old line was
                # compared against the WRONG function (spurious "MI
                # regression: <anonymous>" reports). With stable keys a PR's
                # baseline diff is exactly the functions it changed.
                def walk(node, scope):
                    counts = {}
                    for child in node.get("spaces", []):
                        kind = child.get("kind")
                        name = child.get("name") or "<anonymous>"
                        if kind == "function":
                            k = counts.get(name, 0)
                            counts[name] = k + 1
                            label = name if k == 0 else f"{name}#{k}"
                            metrics = child.get("metrics", {})
                            cc = metrics.get("cyclomatic", {}).get("sum")
                            mi = metrics.get("mi", {}).get("mi_original")
                            qualified = "::".join([rel, *scope, label])
                            if cc is not None:
                                results[qualified] = {
                                    "cc": cc,
                                    "mi": mi,
                                    "path": rel,
                                    "name": name,
                                    "line": child.get("start_line"),
                                }
                            walk(child, [*scope, label])
                        elif kind in ("unit",):
                            walk(child, scope)
                        else:
                            # mod / impl / trait / struct scopes: named
                            # containers contribute to the path so two
                            # `fn new` in different impls stay distinct.
                            k = counts.get(("scope", kind, name), 0)
                            counts[("scope", kind, name)] = k + 1
                            label = f"{kind}:{name}" if k == 0 else f"{kind}:{name}#{k}"
                            walk(child, [*scope, label])

                walk(data, [])
        return results


def _base_label(label: str) -> str:
    return re.sub(r"#\d+$", "", label)


def _sibling_groups(keys) -> dict:
    """(parent path, base name) -> set of ordinal labels, at every key level."""
    groups: dict = {}
    for key in keys:
        segs = key.split("::")
        for i in range(1, len(segs)):
            g = (tuple(segs[:i]), _base_label(segs[i]))
            groups.setdefault(g, set()).add(segs[i])
    return groups


def reshaped_keys(current_keys, baseline_keys) -> set:
    """Current keys whose ordinal-based identity is not trustworthy.

    Same-named siblings (closures, repeated names) are told apart by source
    order (`name`, `name#1`, ...). If a group gains or loses a member, every
    later member's ordinal shifts, so matching by key would compare a function
    with a DIFFERENT sibling's baseline. A key is reshaped when any of its
    levels belongs to a sibling group whose size differs from the baseline's;
    such keys are held to the new-function ceiling instead of a 1:1 match.
    """
    cur = _sibling_groups(current_keys)
    base = _sibling_groups(baseline_keys)
    out = set()
    for key in current_keys:
        segs = key.split("::")
        for i in range(1, len(segs)):
            g = (tuple(segs[:i]), _base_label(segs[i]))
            n_cur, n_base = len(cur[g]), len(base.get(g, ()))
            # 0<->1 is a plain new/removed function, not a reshape.
            if n_cur != n_base and max(n_cur, n_base) > 1:
                out.add(key)
                break
    return out


def main():
    update = "--update-baseline" in sys.argv
    check_tool_available()

    current = scan()

    if update:
        # Store a stable, sorted, minimal snapshot -- diffable in review.
        snapshot = {
            k: {"cc": v["cc"], "mi": v["mi"]} for k, v in sorted(current.items())
        }
        with open(BASELINE_PATH, "w") as fh:
            json.dump(snapshot, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(
            f"complexity_gate.py: baseline updated -> {BASELINE_PATH} ({len(snapshot)} functions)"
        )
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

    regressions = []
    new_functions = []

    reshaped = reshaped_keys(current, baseline)
    for qualified, cur in current.items():
        if qualified in baseline and qualified not in reshaped:
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

    note = (
        f" ({len(reshaped)} in reshaped sibling groups held to the new-function ceiling)"
        if reshaped
        else ""
    )
    print(
        f"complexity_gate.py: PASS -- {len(current)} functions checked, "
        f"no regressions vs baseline{note}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
