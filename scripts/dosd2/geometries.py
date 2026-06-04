#!/usr/bin/env python3
"""Idempotently create TS-stress benchmark geometries (Angstrom).
Standard experimental/B3LYP-relaxed structures."""
from pathlib import Path

MOL_DIR = Path(__file__).resolve().parents[2] / "testdata" / "molecules"

GEOMS = {
    # PROBES — anisotropic / multiply-bonded / heavy-diffuse
    "so2": ("sulfur dioxide C2v (r=1.431, angle=119.3)", [
        ("S", 0.000000, 0.000000,  0.370268),
        ("O", 0.000000, 1.237997, -0.370268),
        ("O", 0.000000,-1.237997, -0.370268)]),
    "cs2": ("carbon disulfide linear (rCS=1.553)", [
        ("C", 0.0, 0.0, 0.000000),
        ("S", 0.0, 0.0, 1.553000),
        ("S", 0.0, 0.0,-1.553000)]),
    "cos": ("carbonyl sulfide linear (rCO=1.157, rCS=1.561)", [
        ("C", 0.0, 0.0, 0.000000),
        ("O", 0.0, 0.0, 1.157000),
        ("S", 0.0, 0.0,-1.561000)]),
    "n2o": ("nitrous oxide linear N-N-O (rNN=1.128, rNO=1.184)", [
        ("N", 0.0, 0.0, 0.000000),
        ("N", 0.0, 0.0, 1.128000),
        ("O", 0.0, 0.0,-1.184000)]),
    "cl2": ("chlorine (r=1.988)", [
        ("Cl", 0.0, 0.0, 0.000000),
        ("Cl", 0.0, 0.0, 1.988000)]),
    "hbr": ("hydrogen bromide (r=1.414)", [
        ("Br", 0.0, 0.0, 0.000000),
        ("H",  0.0, 0.0, 1.414000)]),
    # CONTROLS — saturated / isotropic
    "sih4": ("silane Td (rSiH=1.480)", [
        ("Si", 0.000000, 0.000000, 0.000000),
        ("H",  0.854419, 0.854419, 0.854419),
        ("H", -0.854419,-0.854419, 0.854419),
        ("H", -0.854419, 0.854419,-0.854419),
        ("H",  0.854419,-0.854419,-0.854419)]),
    "ccl4": ("carbon tetrachloride Td (rCCl=1.766)", [
        ("C",  0.000000, 0.000000, 0.000000),
        ("Cl", 1.019600, 1.019600, 1.019600),
        ("Cl",-1.019600,-1.019600, 1.019600),
        ("Cl",-1.019600, 1.019600,-1.019600),
        ("Cl", 1.019600,-1.019600,-1.019600)]),
    "ch3oh": ("methanol Cs", [
        ("C",  -0.046800, 0.663000, 0.000000),
        ("O",  -0.046800,-0.758000, 0.000000),
        ("H",  -1.092600, 0.969000, 0.000000),
        ("H",   0.438000, 1.080000, 0.890000),
        ("H",   0.438000, 1.080000,-0.890000),
        ("H",   0.861000,-1.075000, 0.000000)]),
    "ch3och3": ("dimethyl ether C2v (rCO=1.410, COC=111.7)", [
        ("O",  0.000000, 0.000000, 0.585000),
        ("C",  0.000000, 1.165800,-0.215000),
        ("C",  0.000000,-1.165800,-0.215000),
        ("H",  0.000000, 1.992000, 0.490000),
        ("H",  0.892000, 1.205000,-0.840000),
        ("H", -0.892000, 1.205000,-0.840000),
        ("H",  0.000000,-1.992000, 0.490000),
        ("H",  0.892000,-1.205000,-0.840000),
        ("H", -0.892000,-1.205000,-0.840000)]),
}


def xyz_text(comment, atoms):
    lines = [str(len(atoms)), comment]
    for s, x, y, z in atoms:
        lines.append(f"{s:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f}")
    return "\n".join(lines) + "\n"


def main():
    for name, (comment, atoms) in GEOMS.items():
        path = MOL_DIR / f"{name}.xyz"
        text = xyz_text(comment, atoms)
        if path.exists() and path.read_text() == text:
            print(f"  ok {path.name}")
            continue
        path.write_text(text)
        print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
