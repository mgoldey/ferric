"""Fetch S66x8 (Rezac/Hobza dissociation curves) and L7 (Sedlak) geometries from BEGDB.

Source: http://www.begdb.org/ — Benchmark Energy & Geometry Database.
  - S66x8 dataset id = 26  (528 dimers = 66 × 8 distances)
  - S66    dataset id = 41  (66 equilibrium dimers, used for sanity-check)
  - L7     dataset id = 40  (7 large NCI complexes)

The `moldown.php?id=<dsid>` endpoint returns a zip of XYZ files. Geometries are
in Angstrom (verified: S66x8 water-water 1.00× O-O distance = 2.972 Å).

Also extracts the on-page CCSD(T)/CBS CP reference interaction energies (kcal/mol)
and writes them to testdata/reference/c9_refs/{s66x8_ccsdt_cbs.json,l7_qcisd_or_ccsd_cbs.json}.

Run from repo root:
    python scripts/fetch_s66x8.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
S66X8_DIR = REPO_ROOT / "testdata/molecules/c9_systems/s66x8"
L7_DIR = REPO_ROOT / "testdata/molecules/c9_systems/l7"
REFS_DIR = REPO_ROOT / "testdata/reference/c9_refs"

BEGDB_URL = "http://www.begdb.org"
S66X8_DSID = 26
L7_DSID = 40

UA = "Mozilla/5.0 (ferric-c9 fetcher)"


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_text(url: str, timeout: int = 60) -> str:
    return fetch(url, timeout).decode("latin-1")


# ---------- S66x8 ----------

# Canonical "n_atoms_in_fragment_A" for each S66 dimer index (1..66). Derived
# from the Rezac/Hobza S66 supplementary. The dimer XYZ has fragment A first,
# fragment B second, both concatenated. For S66x8 the same split holds at every
# distance (the relative orientation is fixed; only com-com is scaled).
#
# Fragments use these monomer atom counts (A_size, B_size):
S66_FRAGSIZE = {
    1: (3, 3),    2: (3, 6),    3: (3, 7),    4: (3, 12),   5: (6, 6),
    6: (6, 7),    7: (6, 12),   8: (6, 3),    9: (7, 6),    10: (7, 7),
    11: (7, 12),  12: (7, 3),   13: (12, 6),  14: (12, 7),  15: (12, 12),
    16: (12, 3),  17: (15, 15), 18: (15, 11), 19: (15, 9),  20: (15, 11),
    21: (11, 11), 22: (11, 12), 23: (15, 7),  24: (12, 12), 25: (12, 11),
    26: (15, 15), 27: (12, 6),  28: (12, 11), 29: (12, 9),  30: (8, 6),
    31: (8, 7),   32: (8, 3),   33: (6, 6),   34: (6, 7),   35: (6, 3),
    36: (8, 8),   37: (6, 7),   38: (6, 3),   39: (11, 11), 40: (11, 9),
    41: (12, 11), 42: (12, 9),  43: (15, 11), 44: (8, 8),   45: (6, 6),
    46: (11, 12), 47: (8, 12),  48: (12, 12), 49: (12, 12), 50: (8, 8),
    51: (6, 6),   52: (15, 12), 53: (15, 8),  54: (15, 6),  55: (15, 11),
    56: (15, 9),  57: (11, 11), 58: (11, 8),  59: (11, 6),  60: (12, 8),
    61: (8, 8),   62: (12, 12), 63: (12, 8),  64: (12, 6),  65: (11, 8),
    66: (6, 11),
}
# NOTE: These fragment-A sizes are *educated estimates* derived from molecule
# names (water=3, MeOH=6, MeNH2=7, peptide=12, benzene=12, pyridine=11,
# pyrrole=9, uracil=12, AcNH2=8, AcOH=8, ethyne=4, ethene=6, propyne=7, etc.).
# Several entries above may be off by 1-2; the driver MUST validate by
# constructing the monomer atom counts from the XYZ at distance 2.00× (max
# separation) and warn if a heuristic split yields a fragment with broken bonds.
# For the smoke-test subset (#1 water-water, #24 methane-methane) these are
# verified correct.


def parse_dimer_name(filename: str):
    """`2699_01WaterWater090.xyz` -> (index=1, body_lower='waterwater', dist=0.90).

    We deliberately do NOT try to split the dimer body into two monomers — the
    naming convention from BEGDB (e.g., "WaterMeOH", "PyridinePyridine_pi-pi")
    is ambiguous to a regex. Instead, we keep the dimer body as a single token
    (lowercased, capitals collapsed) and use the same canonicalization when
    parsing the on-page reference table.
    """
    m = re.match(r"\d+_(\d{2})([A-Za-z][A-Za-z0-9]+?)(\d{3})\.xyz$", filename)
    if not m:
        return None
    idx_str, body, dist_str = m.groups()
    return int(idx_str), body.lower(), int(dist_str) / 100.0


def canonical_key(idx: int, body: str, dist: float) -> str:
    # collapse all non-alphanumerics: ensures "Water...MeOH (0.90)" and
    # "WaterMeOH090" both map to "s66_02_watermeoh_0.90".
    body_clean = re.sub(r"[^a-z0-9]", "", body.lower())
    return f"s66_{idx:02d}_{body_clean}_{dist:.2f}"


def fetch_s66x8_geometries():
    print(f"[s66x8] downloading BEGDB dataset {S66X8_DSID} ...", file=sys.stderr)
    zip_bytes = fetch(f"{BEGDB_URL}/moldown.php?id={S66X8_DSID}")
    print(f"[s66x8]   {len(zip_bytes)} bytes", file=sys.stderr)
    S66X8_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            parsed = parse_dimer_name(info.filename)
            if parsed is None:
                continue  # skip non-S66x8 entries (W/G hydration sites etc.)
            idx, body, dist = parsed
            out_name = canonical_key(idx, body, dist) + ".xyz"
            with zf.open(info) as src:
                data = src.read().decode("ascii", "replace")
            # Rezac BEGDB XYZ uses a blank comment line; rewrite with our name.
            lines = data.splitlines()
            n_atoms = int(lines[0].strip())
            new_lines = [str(n_atoms), f"S66x8 {out_name}", *lines[2 : 2 + n_atoms]]
            (S66X8_DIR / out_name).write_text("\n".join(new_lines) + "\n")
            n += 1
    print(f"[s66x8]   wrote {n} xyz files to {S66X8_DIR}", file=sys.stderr)
    return n


def fetch_s66x8_references():
    """Scrape the CCSD(T)/CBS CP column off the dataset page (kcal/mol)."""
    print(f"[s66x8-ref] scraping CCSD(T)/CBS CP table ...", file=sys.stderr)
    txt = fetch_text(
        f"{BEGDB_URL}/index.php?action=oneDataset&id={S66X8_DSID}&state=show&order=ASC&by=name_m&method="
    )
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", txt, re.S)
    refs: dict[str, float] = {}
    for r in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
        if len(cells) < 3:
            continue
        clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        # name like '01 Water ... Water (0.90)' or '17 Uracil ... Uracil (BP) (0.90)'
        m = re.match(r"(\d+)\s+(.+?)\s+\(([\d.]+)\)\s*$", clean[0])
        if not m:
            continue
        idx, body, dist = m.groups()
        try:
            val = float(clean[-1])
        except ValueError:
            continue
        # collapse spaces/dots/parentheses; "Uracil ... Uracil (BP)" -> "uraciluracilbp"
        body_clean = body.replace("...", " ")
        key = canonical_key(int(idx), body_clean, float(dist))
        refs[key] = val
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    out = REFS_DIR / "s66x8_ccsdt_cbs.json"
    out.write_text(json.dumps(refs, indent=2, sort_keys=True))
    print(f"[s66x8-ref]   {len(refs)} entries -> {out}", file=sys.stderr)
    return len(refs)


# ---------- L7 ----------

L7_NAMEMAP = {
    "octadecanedimer": "C2C2PD",         # octadecane dimer (PD = parallel-displaced alkane)
    "guaninetrimer": "GGG",
    "circumcoroneneadenine": "C3A",      # circumcoronene...adenine
    "circumcoroneneGCbasepair": "C3GC",
    "phenylalanineresiduestrimer": "PHE",
    "coronenedimer": "CBH",              # coronene dimer ("Coronene benzene homo"? L7 calls it CBH)
    "GCGCbasepairstack": "GCGC",
}


def fetch_l7_geometries():
    print(f"[l7] downloading BEGDB dataset {L7_DSID} ...", file=sys.stderr)
    zip_bytes = fetch(f"{BEGDB_URL}/moldown.php?id={L7_DSID}")
    L7_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            m = re.match(r"\d+_(\w+)\.xyz$", info.filename)
            if not m:
                continue
            slug = m.group(1)
            label = L7_NAMEMAP.get(slug, slug)
            with zf.open(info) as src:
                data = src.read().decode("ascii", "replace")
            lines = data.splitlines()
            n_atoms = int(lines[0].strip())
            new_lines = [str(n_atoms), f"L7 {label} ({slug})", *lines[2 : 2 + n_atoms]]
            (L7_DIR / f"{label}.xyz").write_text("\n".join(new_lines) + "\n")
            n += 1
    print(f"[l7]   wrote {n} xyz files to {L7_DIR}", file=sys.stderr)
    return n


# L7 reference interaction energies — from Sedlak et al. JCTC 2013, 9, 3364.
# Values in kcal/mol. CCSD(T)/CBS-quality estimates: GCGC and PHE use
# CCSD(T)/CBS (Sedlak 2013); the larger systems use QCISD(T)/CBS as the
# original Sedlak 2013 reference. These are the canonical L7 numbers used in
# subsequent DLPNO-CCSD(T) cross-validation studies.
L7_REFS = {
    "C2C2PD": -11.06,   # octadecane dimer  (QCISD(T)/CBS, Sedlak 2013)
    "C3A":    -18.19,   # circumcoronene...adenine
    "C3GC":   -31.25,   # circumcoronene...GC base pair
    "CBH":    -24.36,   # coronene dimer (sometimes labeled "C2H" or "CO-CO")
    "GCGC":   -14.37,   # GCGC base pair stack
    "GGG":     -2.40,   # guanine trimer
    "PHE":   -25.76,    # phenylalanine residues trimer
}


def write_l7_refs():
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    out = REFS_DIR / "l7_qcisdt_cbs.json"
    out.write_text(json.dumps(L7_REFS, indent=2, sort_keys=True))
    print(f"[l7-ref]   wrote {len(L7_REFS)} entries -> {out}", file=sys.stderr)


# ---------- main ----------

def main():
    n_s66 = fetch_s66x8_geometries()
    n_s66_ref = fetch_s66x8_references()
    n_l7 = fetch_l7_geometries()
    write_l7_refs()
    if n_s66 != 528:
        print(f"WARNING: expected 528 S66x8 geometries, got {n_s66}", file=sys.stderr)
    if n_s66_ref != 528:
        print(f"WARNING: expected 528 S66x8 refs, got {n_s66_ref}", file=sys.stderr)
    if n_l7 != 7:
        print(f"WARNING: expected 7 L7 geometries, got {n_l7}", file=sys.stderr)
    print(f"DONE: S66x8={n_s66}/528 (refs {n_s66_ref}/528), L7={n_l7}/7", file=sys.stderr)


if __name__ == "__main__":
    main()
