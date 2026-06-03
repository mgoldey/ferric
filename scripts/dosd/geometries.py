#!/usr/bin/env python3
"""Idempotently create the 10 DOSD geometries not already in testdata/molecules.
Geometries are standard experimental/B3LYP-relaxed structures in Angstrom."""
import os
from pathlib import Path

MOL_DIR = Path(__file__).resolve().parents[2] / "testdata" / "molecules"

# name -> (comment, [(symbol, x, y, z), ...])
GEOMS = {
    "nh3": ("ammonia C3v (exp r=1.012, angle=106.7)", [
        ("N",  0.000000,  0.000000,  0.116489),
        ("H",  0.000000,  0.939731, -0.271808),
        ("H",  0.813831, -0.469865, -0.271808),
        ("H", -0.813831, -0.469865, -0.271808)]),
    "co2": ("carbon dioxide linear (r=1.162)", [
        ("C", 0.0, 0.0,  0.000000),
        ("O", 0.0, 0.0,  1.162000),
        ("O", 0.0, 0.0, -1.162000)]),
    "c2h2": ("acetylene linear (rCC=1.203, rCH=1.063)", [
        ("C", 0.0, 0.0,  0.601500),
        ("C", 0.0, 0.0, -0.601500),
        ("H", 0.0, 0.0,  1.664500),
        ("H", 0.0, 0.0, -1.664500)]),
    "c2h4": ("ethylene D2h (rCC=1.339, rCH=1.086, HCH=117.4)", [
        ("C",  0.000000, 0.000000,  0.669500),
        ("C",  0.000000, 0.000000, -0.669500),
        ("H",  0.000000, 0.922832,  1.237695),
        ("H",  0.000000,-0.922832,  1.237695),
        ("H",  0.000000, 0.922832, -1.237695),
        ("H",  0.000000,-0.922832, -1.237695)]),
    "c2h6": ("ethane D3d staggered (rCC=1.536, rCH=1.091)", [
        ("C",  0.000000,  0.000000,  0.768000),
        ("C",  0.000000,  0.000000, -0.768000),
        ("H",  0.000000,  1.013302,  1.164532),
        ("H",  0.877488, -0.506651,  1.164532),
        ("H", -0.877488, -0.506651,  1.164532),
        ("H",  0.000000, -1.013302, -1.164532),
        ("H", -0.877488,  0.506651, -1.164532),
        ("H",  0.877488,  0.506651, -1.164532)]),
    "hf": ("hydrogen fluoride (r=0.917)", [
        ("F", 0.0, 0.0, 0.000000),
        ("H", 0.0, 0.0, 0.917000)]),
    "hcl": ("hydrogen chloride (r=1.275)", [
        ("Cl", 0.0, 0.0, 0.000000),
        ("H",  0.0, 0.0, 1.275000)]),
    "h2s": ("hydrogen sulfide C2v (r=1.336, angle=92.1)", [
        ("S", 0.000000, 0.000000,  0.103729),
        ("H", 0.000000, 0.961700, -0.829834),
        ("H", 0.000000,-0.961700, -0.829834)]),
    "o2": ("oxygen triplet (r=1.208); multiplicity=3 set in TOML", [
        ("O", 0.0, 0.0, 0.000000),
        ("O", 0.0, 0.0, 1.208000)]),
}


def xyz_text(name, comment, atoms):
    lines = [str(len(atoms)), comment]
    for s, x, y, z in atoms:
        lines.append(f"{s:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f}")
    return "\n".join(lines) + "\n"


def main():
    MOL_DIR.mkdir(parents=True, exist_ok=True)
    for name, (comment, atoms) in GEOMS.items():
        path = MOL_DIR / f"{name}.xyz"
        text = xyz_text(name, comment, atoms)
        if path.exists() and path.read_text() == text:
            print(f"  ok (unchanged) {path.name}")
            continue
        path.write_text(text)
        print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
