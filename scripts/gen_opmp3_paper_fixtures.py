"""Committed subset of the OP-MP3 paper's data for crates/ferric-mp2/tests/opmp3_paper.rs.

Source: K. Bystrom, T. C. Berkelbach, D. Hait, arXiv:2609.25330, Zenodo
10.5281/zenodo.23482348 (`SI_pade_paper.tar.gz`, extracted). Inputs:

* `FINAL_RESULTS/aug-cc-pCVTZ/{HF,PBE0,KOOMP2}/W4-17_<s>.xyz/`: the eight
  W matrices (`<s>.W{2,3}{alpha,beta}_{ss,os}.npy`) and `<s>.final.out`
  (E_HF of the reference density and E_NBS) of every structure below;
* `results/{HF,PBE0,kOOMP2}/TAE_W4-17nonMR.default.csv`: the paper's
  per-reaction reference TAE and model errors (kcal/mol);
* `results/RXN_ALL_plus_RE.yaml`: the reactions' structures and counts.

Outputs:

* `testdata/reference/opmp3/paper_w4-17.json`: W matrices, E_HF, E_NBS per
  structure and reference; the 12 TAE reactions below with the paper's
  errors; the sha256 of every input file;
* `testdata/molecules/w4-17/<s>.xyz`: the ACCDB W4-17 geometries the paper
  used (charge and multiplicity on line 2), copied from `--geom`;
* `testdata/reference/opmp3/aug-cc-pcvtz_hbcnof.json`: aug-cc-pCVTZ (BSE)
  for B, C, N, O, F and aug-cc-pVTZ for H, the paper's basis, cut from the
  merged BSE JSON given as `--basis-json`;
* `testdata/reference/opmp3/qchem_cc-pvdz.json`: the Q-Chem 6.4 values the
  paper's own `wmatrix_code` tests compare with
  (`wmatrix_code/tests/data/references/<mol>_<hf|pbe0>_cc-pvdz_<tag>.json`:
  E_HF of the reference density, E_NBS, Tr W2 SS / OS, Tr W3), for the
  restricted open-shell CH2, CH3, CN, HCO and the closed-shell H2O, He and
  one-electron H, with each value's printed decimals and the geometry.

Reactions: the W4-17 nonMR TAEs of H2, H2O, HCN, HF, benzene, N2, NH3, BH3,
C2H2, C2H4, CH4 and CO (atoms H, B, C, N, O, F).

Run (light; no SCF):
    uv run --no-sync python scripts/gen_opmp3_paper_fixtures.py \\
        --zenodo <SI_pade_paper> --geom <ACCDB W4-17 xyz dir> --basis-json <merged json>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "reference" / "opmp3"
GEOM_OUT = ROOT / "testdata" / "molecules" / "w4-17"
MOLECULES = [
    "h2",
    "h2o",
    "hcn",
    "hf",
    "benzene",
    "n2",
    "nh3",
    "bh3",
    "c2h2",
    "c2h4",
    "ch4",
    "co",
]
ATOMS = ["h", "b", "c", "n", "o", "f"]
REFS = {"HF": "HF", "PBE0": "PBE0", "KOOMP2": "kOOMP2"}  # FINAL_RESULTS -> results
MODELS = ["MP2", "MP3", "OP-MP3", "SOP-MP3", "SSD-OPMP3", "P-MP3"]
W_LABELS = [
    f"W{n}{s}_{c}" for n in (2, 3) for s in ("alpha", "beta") for c in ("ss", "os")
]
BASIS_Z = ["1", "5", "6", "7", "8", "9"]
QCHEM = {
    "ch2": "restricted",
    "ch3": "restricted",
    "cn": "restricted",
    "hco": "restricted",
    "h2o": "base",
    "he": "base",
    "h": "base",
}
QCHEM_KEYS = ["e_hf", "e_nbs", "tr_w2_ss", "tr_w2_os", "tr_w3"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_out(path: Path) -> dict:
    out = {}
    for ln in path.read_text().splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def read_rxn(path: Path) -> dict:
    """The flat reaction YAML (name -> Dataset, counts, energy, structs)."""
    out, cur, key = {}, None, None
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        if not ln.startswith(" "):
            cur = out.setdefault(ln.rstrip(":").strip(), {"counts": [], "structs": []})
            continue
        body = ln.strip()
        if body.startswith("- "):
            val = body[2:].strip()
            cur[key].append(float(val) if key == "counts" else val)
        else:
            k, _, v = body.partition(":")
            key = k.strip()
            if v.strip():
                cur[key] = v.strip()
    return out


def structure(zen: Path, ref: str, name: str, hashes: dict) -> dict:
    d = zen / "FINAL_RESULTS" / "aug-cc-pCVTZ" / ref / f"W4-17_{name}.xyz"
    stem = f"W4-17_{name}"
    w = {}
    for lbl in W_LABELS:
        p = d / f"{stem}.{lbl}.npy"
        hashes[str(p.relative_to(zen))] = sha256(p)
        m = np.load(p)
        if not np.array_equal(m, m.T):
            raise SystemExit(f"{p}: W is not exactly symmetric")
        # Upper triangle, row-major (i <= j): halves the fixture, lossless.
        w[lbl] = {"n": m.shape[0], "upper": m[np.triu_indices(m.shape[0])].tolist()}
    fo_path = d / f"{stem}.final.out"
    hashes[str(fo_path.relative_to(zen))] = sha256(fo_path)
    fo = final_out(fo_path)
    e_hf = next(v for k, v in fo.items() if k.startswith("Reference energy"))
    return {"e_hf": e_hf, "e_nbs": fo["Non-Brillouin singles correction"], "w": w}


def reactions(zen: Path, ref: str, rxn: dict, hashes: dict) -> dict:
    csv_path = zen / "results" / REFS[ref] / "TAE_W4-17nonMR.default.csv"
    hashes[str(csv_path.relative_to(zen))] = sha256(csv_path)
    with open(csv_path) as fh:
        rows = {r["reaction"]: r for r in csv.DictReader(fh)}
    want = {f"W4-17_{m}" for m in MOLECULES}
    out = {}
    for name, r in rxn.items():
        if r.get("Dataset") != "TAE_W4-17nonMR" or r["structs"][0] not in want:
            continue
        row = rows[name]
        errors = {}
        for col, v in row.items():
            label = col.replace(" error (kcal/mol)", "")
            if label.startswith("SSD-OPMP3("):
                errors["SSD-OPMP3"] = float(v)
                errors["x_ss_label"] = label
            elif label in MODELS:
                errors[label] = float(v)
        out[name] = {
            "structs": [s.removeprefix("W4-17_") for s in r["structs"]],
            "counts": r["counts"],
            "reference_kcal": float(row["reference (kcal/mol)"]),
            "errors_kcal": errors,
        }
    return out


def decimals(text: str, key: str) -> int:
    """Digits after the point of `key`'s value as printed in a JSON text."""
    tok = text.split(f'"{key}":', 1)[1].split(",", 1)[0].split("}", 1)[0].strip()
    return len(tok.split(".", 1)[1].lower().split("e", 1)[0]) if "." in tok else 0


def qchem_refs(zen: Path, hashes: dict) -> dict:
    tests = zen / "wmatrix_code" / "tests" / "data"
    out = {}
    for mol, tag in QCHEM.items():
        xyz = tests / f"{mol}.xyz"
        hashes[str(xyz.relative_to(zen))] = sha256(xyz)
        entry = {"xyz": xyz.read_text(), "tag": tag}
        for fn in ("hf", "pbe0"):
            p = tests / "references" / f"{mol}_{fn}_cc-pvdz_{tag}.json"
            hashes[str(p.relative_to(zen))] = sha256(p)
            text = p.read_text()
            vals = json.loads(text)
            entry[fn] = {k: vals[k] for k in QCHEM_KEYS}
            entry[fn]["decimals"] = {k: decimals(text, k) for k in QCHEM_KEYS}
        out[mol] = entry
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zenodo", required=True, type=Path)
    p.add_argument("--geom", required=True, type=Path)
    p.add_argument("--basis-json", required=True, type=Path)
    args = p.parse_args()
    zen = args.zenodo
    hashes: dict = {}
    rxn_path = zen / "results" / "RXN_ALL_plus_RE.yaml"
    hashes[str(rxn_path.relative_to(zen))] = sha256(rxn_path)
    rxn = read_rxn(rxn_path)
    data = {
        "provenance": {
            "paper": "K. Bystrom, T. C. Berkelbach, D. Hait, arXiv:2609.25330v2",
            "zenodo": "10.5281/zenodo.23482348 (SI_pade_paper.tar.gz)",
            "basis": "aug-cc-pCVTZ (H: aug-cc-pVTZ), all electrons, exact integrals",
            "generator": "scripts/gen_opmp3_paper_fixtures.py",
            "kcal_per_hartree": 627.5094740630558,
            "sha256": hashes,
        },
        "structures": {},
        "reactions": {},
    }
    for ref in REFS:
        data["structures"][ref] = {
            s: structure(zen, ref, s, hashes) for s in MOLECULES + ATOMS
        }
        data["reactions"][ref] = reactions(zen, ref, rxn, hashes)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paper_w4-17.json").write_text(
        json.dumps(data, separators=(",", ":")) + "\n"
    )
    GEOM_OUT.mkdir(parents=True, exist_ok=True)
    for s in MOLECULES + ATOMS:
        src = args.geom / f"W4-17_{s}.xyz"
        if src.exists():
            shutil.copyfile(src, GEOM_OUT / f"{s}.xyz")
    bs = json.loads(args.basis_json.read_text())
    bs["elements"] = {z: bs["elements"][z] for z in BASIS_Z}
    bs["description"] = (
        "aug-cc-pCVTZ (BSE) for B, C, N, O, F and aug-cc-pVTZ for H, the basis of "
        "arXiv:2609.25330; cut by scripts/gen_opmp3_paper_fixtures.py"
    )
    (OUT / "aug-cc-pcvtz_hbcnof.json").write_text(json.dumps(bs, indent=0) + "\n")
    qhashes: dict = {}
    qc = {
        "provenance": {
            "source": "Zenodo 10.5281/zenodo.23482348, SI_pade_paper/wmatrix_code/tests/data "
            "(Q-Chem 6.4; HF and PBE0 with the SG-1 grid; scf_convergence 9; cc-pVDZ, all "
            "electrons, exact integrals; tag restricted = RO reference, base = unrestricted)",
            "generator": "scripts/gen_opmp3_paper_fixtures.py",
            "sha256": qhashes,
        },
        "molecules": qchem_refs(zen, qhashes),
    }
    (OUT / "qchem_cc-pvdz.json").write_text(json.dumps(qc, indent=1) + "\n")
    n_rxn = sum(len(v) for v in data["reactions"].values())
    print(
        f"wrote {OUT / 'paper_w4-17.json'} ({n_rxn} reaction rows), geometries, basis"
    )


if __name__ == "__main__":
    main()
