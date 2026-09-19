#!/usr/bin/env python3
"""Generate `src/tables.rs` from the upstream s-dftd3 Fortran sources.

The D3 reference data is large (262,444 C6 floats, plus r4/r2, covalent radii
and vdW radii tables). Hand-transcribing it would be unauditable and almost
certainly wrong somewhere, so this script PARSES the authoritative upstream
Fortran and emits Rust. Re-running it against the same source must be
byte-reproducible.

Upstream: https://github.com/dftd3/simple-dftd3 (LGPL-3.0-or-later).
The numerical reference data it contains originates with Grimme et al.,
JCP 132, 154104 (2010) and JCC 32, 1456 (2011).

Usage:
    python3 generate_tables.py /path/to/s-dftd3-src /path/to/mctc-lib-src

The default paths point at the copies vendored into the local xtb build tree.
"""

import re
import sys
from pathlib import Path

MAX_ELEM = 103
MAX_REF = 7
# Upstream stores the element-pair axis as a lower-triangular packed index of
# size (max_elem+1)*max_elem/2.
N_PAIRS = (MAX_ELEM + 1) * MAX_ELEM // 2


def floats_in(text):
    """Every Fortran `<number>_wp` literal in `text`, in source order."""
    return [
        float(m) for m in re.findall(r"([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)_wp", text)
    ]


def slice_between(path, start_pat, end_pat):
    """The text of `path` between the first line matching `start_pat` and the
    next line matching `end_pat` (exclusive of the start line's prefix)."""
    text = path.read_text()
    m = re.search(start_pat, text)
    if not m:
        raise SystemExit(f"FATAL: start pattern {start_pat!r} not found in {path}")
    rest = text[m.end() :]
    e = re.search(end_pat, rest)
    if not e:
        raise SystemExit(f"FATAL: end pattern {end_pat!r} not found in {path}")
    return rest[: e.start()]


def parse_number_of_references(ref_f90):
    body = slice_between(
        ref_f90,
        r"integer, parameter :: number_of_references\(max_elem\) = \[ &",
        r"\]\s*!\s*Ac-Lr",
    )
    # This block is integers, not `_wp` floats.
    body = re.sub(r"!.*", "", body)  # strip trailing element-name comments
    nums = [int(x) for x in re.findall(r"\b(\d+)\b", body)]
    # The closing line (matched as the end pattern) holds the last 15 entries,
    # so re-read including it.
    text = ref_f90.read_text()
    m = re.search(
        r"integer, parameter :: number_of_references\(max_elem\) = \[(.*?)\]\s*!\s*Ac-Lr",
        text,
        re.S,
    )
    body = re.sub(r"!.*", "", m.group(1))
    nums = [int(x) for x in re.findall(r"\b(\d+)\b", body)]
    assert len(nums) == MAX_ELEM, (
        f"number_of_references: got {len(nums)}, want {MAX_ELEM}"
    )
    return nums


def parse_reference_cn(ref_f90):
    text = ref_f90.read_text()
    m = re.search(
        r"real\(wp\), parameter :: reference_cn\(max_ref, max_elem\) = reshape\(\[(.*?)\[max_ref, max_elem\]\)",
        text,
        re.S,
    )
    if not m:
        raise SystemExit("FATAL: reference_cn block not found")
    vals = floats_in(m.group(1))
    want = MAX_REF * MAX_ELEM
    assert len(vals) == want, f"reference_cn: got {len(vals)}, want {want}"
    return vals  # column-major (max_ref, max_elem): index = iref + MAX_REF*izp


def parse_reference_c6(ref_f90):
    """The C6 table is emitted as many `c6ab_view(a:b) = [ ... ]` chunks."""
    text = ref_f90.read_text()
    chunks = re.findall(
        r"c6ab_view\((\d+):(\d+)\)\s*=\s*\[(.*?)\]\s*$",
        text,
        re.S | re.M,
    )
    if not chunks:
        raise SystemExit("FATAL: no c6ab_view chunks found")
    total = MAX_REF * MAX_REF * N_PAIRS
    out = [0.0] * total
    covered = 0
    for lo, hi, body in chunks:
        lo, hi = int(lo), int(hi)
        vals = floats_in(body)
        n = hi - lo + 1
        assert len(vals) == n, f"chunk {lo}:{hi} has {len(vals)} values, want {n}"
        out[lo - 1 : hi] = vals
        covered += n
    assert covered == total, f"C6 coverage {covered} != {total}"
    return out  # flat, Fortran order (iref, jref, ipair)


def parse_simple_table(path, decl_pat, expect, scale_name=None):
    text = path.read_text()
    m = re.search(decl_pat + r"(.*?)\]", text, re.S)
    if not m:
        raise SystemExit(f"FATAL: pattern {decl_pat!r} not found in {path}")
    vals = floats_in(m.group(1))
    assert len(vals) == expect, f"{path.name}: got {len(vals)}, want {expect}"
    return vals


def main():
    src = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("/home/matt/qc/xtb-build/xtb-6.7.1/build-cmake/_deps/s-dftd3-src")
    )
    mctc = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("/home/matt/qc/xtb-build/xtb-6.7.1/build-cmake/_deps/mctc-lib-src")
    )
    d3 = src / "src" / "dftd3"

    nref = parse_number_of_references(d3 / "reference.f90")
    refcn = parse_reference_cn(d3 / "reference.f90")
    c6 = parse_reference_c6(d3 / "reference.f90")

    # r4/r2 expectation values: `sqrt(0.5*r4r2(i)*sqrt(i))` is applied upstream,
    # so store the RAW table and do the transform in Rust (keeps this file a
    # pure transcription).
    r4r2 = parse_simple_table(
        d3 / "data" / "r4r2.f90",
        r"real\(wp\), parameter :: r4_over_r2\(max_elem\) = \[",
        118,
    )

    # vdW radii for the ATM term, packed lower-triangular, in Angstrom
    # (upstream multiplies by aatoau at declaration).
    vdw = parse_simple_table(
        d3 / "data" / "vdwrad.f90",
        r"real\(wp\), parameter :: vdwrad\(max_elem\*\(1\+max_elem\)/2\) = aatoau \* \[",
        N_PAIRS,
    )

    # Covalent radii (Pyykko/Atsumi, scaled by 4/3 upstream), in Angstrom.
    covrad = parse_simple_table(
        mctc / "src" / "mctc" / "data" / "covrad.f90",
        r"real\(wp\), parameter :: covalent_rad_2009\(max_elem\) = aatoau \* \[",
        118,
    )

    def fmt(vals, per_line=None):
        """Emit values packed to rustfmt's 100-column width.

        rustfmt reflows array literals to fill the line width, so emitting a
        fixed number of values per line makes `cargo fmt --check` fail on this
        generated file forever. Packing greedily to the same width rustfmt
        targets makes the generator's output a FIXED POINT of rustfmt, so the
        file can be regenerated and formatted interchangeably.
        """
        lines, cur = [], "   "
        for v in vals:
            tok = f" {v!r},"
            if len(cur) + len(tok) > 100:
                lines.append(cur)
                cur = "   " + tok
            else:
                cur += tok
        if cur.strip():
            lines.append(cur)
        return "\n".join(lines)

    out = f"""//! D3 reference data, GENERATED -- do not edit by hand.
//!
//! Produced by `generate_tables.py` from the upstream s-dftd3 Fortran sources
//! (<https://github.com/dftd3/simple-dftd3>, LGPL-3.0-or-later). The numerical
//! data originates with Grimme et al., JCP 132, 154104 (2010) and
//! JCC 32, 1456 (2011).
//!
//! Re-run the generator to regenerate; it is a pure transcription, so the
//! output is byte-reproducible against a given upstream revision.

// These are transcribed physical constants, not computed expressions:
//   - `approx_constant`: the vdW radius table contains 3.1416 (an Angstrom
//     radius for one element pair) which clippy mistakes for an approximation
//     of PI. Substituting `std::f64::consts::PI` would silently change a
//     tabulated datum into a different number.
//   - `excessive_precision`: the repo's own convention for verbatim reference
//     constants (see the workspace `[lints.clippy]` rationale) -- truncating
//     digits would change the constant.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

/// Highest atomic number with D3 reference data.
pub const MAX_ELEM: usize = {MAX_ELEM};
/// Maximum number of reference systems per element.
pub const MAX_REF: usize = {MAX_REF};
/// Packed lower-triangular element-pair count: (MAX_ELEM+1)*MAX_ELEM/2.
pub const N_PAIRS: usize = {N_PAIRS};

/// Number of reference systems actually present for each element (index Z-1).
pub static NUMBER_OF_REFERENCES: [u8; MAX_ELEM] = [
{fmt([int(x) for x in nref])}
];

/// Reference coordination numbers, column-major `(MAX_REF, MAX_ELEM)`:
/// `REFERENCE_CN[iref + MAX_REF * (z - 1)]`. Entries beyond an element's
/// reference count are -1.0 and must never be read.
pub static REFERENCE_CN: [f64; MAX_REF * MAX_ELEM] = [
{fmt(refcn)}
];

/// Reference C6 coefficients, Fortran order `(MAX_REF, MAX_REF, N_PAIRS)`:
/// `REFERENCE_C6[iref + MAX_REF * (jref + MAX_REF * ipair)]`.
pub static REFERENCE_C6: [f64; MAX_REF * MAX_REF * N_PAIRS] = [
{fmt(c6)}
];

/// Raw r4/r2 expectation values (PBE0/def2-QZVP, Grimme 2010), index Z-1,
/// in atomic units. The D3 `sqrt(0.5 * r4r2 * sqrt(Z))` transform is applied
/// in `model.rs`, not here.
pub static R4R2: [f64; 118] = [
{fmt(r4r2)}
];

/// Pairwise vdW radii for the ATM three-body term, packed lower-triangular
/// (same index scheme as `REFERENCE_C6`'s pair axis), in ANGSTROM. Upstream
/// applies the Angstrom->Bohr factor at declaration; `model.rs` does it here.
pub static VDW_RAD: [f64; N_PAIRS] = [
{fmt(vdw)}
];

/// Covalent radii (Pyykko/Atsumi 2009), index Z-1, in ANGSTROM. Upstream scales
/// these by 4/3 for the coordination-number counting function; `model.rs`
/// applies that factor.
pub static COVALENT_RAD: [f64; 118] = [
{fmt(covrad)}
];
"""
    dest = Path(__file__).resolve().parent / "src" / "tables.rs"
    dest.write_text(out)

    # Normalise through rustfmt so the generated file is a FIXED POINT of the
    # formatter. Without this, `cargo fmt --check` fails on this file forever:
    # rustfmt reflows array literals to its own width, and no hand-chosen
    # values-per-line matches that exactly. Running it here means regenerating
    # and formatting are interchangeable, and the committed file is the same
    # either way.
    import shutil
    import subprocess

    if shutil.which("rustfmt"):
        subprocess.run(["rustfmt", "--edition", "2021", str(dest)], check=True)
        out = dest.read_text()
    else:
        print("WARNING: rustfmt not found; output may fail `cargo fmt --check`")

    print(f"wrote {dest} ({len(out)} bytes)")
    print(f"  number_of_references: {len(nref)}")
    print(f"  reference_cn:         {len(refcn)}")
    print(
        f"  reference_c6:         {len(c6)} ({sum(1 for v in c6 if v != 0.0)} nonzero)"
    )
    print(f"  r4r2:                 {len(r4r2)}")
    print(f"  vdwrad:               {len(vdw)}")
    print(f"  covrad:               {len(covrad)}")


if __name__ == "__main__":
    main()
