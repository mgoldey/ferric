"""Danuglipron (PF-06882961): the designed analogue set and its pharmacophore.

THE EXPERIMENT'S HYPOTHESES. The generic machinery -- `Analogue`,
`PharmacophoreSpec`, embedding -- lives in `tools/morph`. What is here is
specific to this campaign and belongs with it: the parent structure, the GLP-1R
pharmacophore, and the nine designed analogues plus two negative controls.

## Why these analogues

Measured starting point (`RESULTS.md` M1): danuglipron trips **zero** structural
alerts across Brenk, PAINS, NIH, Glaxo, Dundee and BMS. Its only developability
flag is **MW 555.6 > 500**, with **cLogP 4.89** at the rule-of-5 limit.

That reframes the problem, and contradicts the obvious first guess:

- The clinical failure was **not** a structural toxicophore. Pfizer's April 2025
  discontinuation followed a **single asymptomatic, reversible** DILI case, with
  liver-enzyme elevation rates "in line with approved agents in the class" across
  >1400 participants. The dose-limiting, program-defining problem was
  **dose-dependent GI intolerability**, which is an **exposure** problem.
- So the tractable lever is **reducing the dose needed for efficacy**, not
  deleting a toxicophore that is not there.

Three exposure-side hypotheses follow:

  **H1 (size/lipophilicity).** Trim MW below 500 and cLogP below 5 while keeping
  every pharmacophore contact. Lower dose for equal free-fraction exposure.
  **H2 (metabolic soft spots).** Block the benzylic ether and oxetane positions
  most likely to drive oxidative clearance and reactive-metabolite formation --
  the mechanistic route to idiosyncratic DILI, the one liability that IS
  structural even though no catalog flags it.
  **H3 (acid bioisosteres).** Replace the carboxylic acid, which drives potency
  but also acyl-glucuronide formation, with bioisosteres that keep the anionic
  contact but cannot form an acyl glucuronide.

## The constraint

The cryo-EM structure (PDB 7LCJ / 7S15) shows the pocket requires the
primate-specific **Trp33**, and the carboxylic acid is the potency-critical
anchor. Every analogue declares what must survive.

## READ THIS BEFORE REUSING ANY FIT NUMBER

`RESULTS.md` M7: candidate **pose generation failed** for this set. No generated
conformer reaches the bound pose within the conventional 2.0 A bar (best 2.23 A).
The analogue DESIGNS below are unaffected -- they are 2D hypotheses -- but every
3D fit number computed from them measures pose error rather than chemistry.
"""
from __future__ import annotations

from tools.morph.design import Analogue, PharmacophoreSpec

# Parent. Matches scripts/fetch_danuglipron.py (PubChem CID 134611040), so the
# 3D ensemble in testdata/molecules/c9_systems/danuglipron corresponds to this
# exact connectivity. `test_design.py` asserts the two stay in sync.
DANUGLIPRON_SMILES = (
    "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)"
    "C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
)


# The features the GLP-1R pocket actually needs, per the cryo-EM structures:
#  - an anionic/acid-like H-bond acceptor cluster (the potency anchor). Written
#    as a recursive-SMARTS union so acid BIOISOSTERES also satisfy it -- an
#    exact [CX3](=O)[OX2H1] would fail every H3 analogue by construction, which
#    would make the whole H3 arm untestable (a pass condition that cannot be
#    reached is arithmetic, not measurement).
#  - the benzimidazole (or an aza-equivalent) core.
#  - the basic piperidine nitrogen linker.
#  - an aromatic nitrile or equivalent electron-poor aryl terminus that packs
#    against the Trp33 region.
GLP1R_PHARMACOPHORE = PharmacophoreSpec(
    features=(
        (
            "acid_or_bioisostere",
            # Carboxylic acid, tetrazole, acylsulfonamide, oxadiazolone,
            # hydroxamic acid, sulfonic acid -- each in BOTH its protonated and
            # its DEPROTONATED form.
            #
            # The anionic alternatives are not optional decoration: the species
            # actually scored is the anion (pKa ~4, so anionic at pH 7.4). A
            # protonated-only pattern reported every scored molecule as
            # pharmacophore-breaking, which
            # test_ionized_forms_still_satisfy_the_pharmacophore caught.
            "["
            # --- protonated ---
            "$([CX3](=O)[OX2H1]),$(c1nn[nH]n1),$(c1nnn[nH]1),"
            "$([CX3](=O)[NX3H1][SX4](=O)=O),$([SX4](=O)(=O)[OX2H1]),"
            "$([CX3](=O)[NX3H1][OX2H1]),$(c1[nH]oc(=O)n1),$(c1onc(=O)[nH]1),"
            # --- deprotonated ---
            "$([CX3](=O)[OX1H0-]),$([CX3]([OX1-])=O),"
            "$(c1nn[n-]n1),$(c1nnn[n-]1),$([nX2-]1nnnc1),$([nX2-]1nncn1),"
            "$([CX3](=O)[NX2-][SX4](=O)=O),$([SX4](=O)(=O)[OX1-]),"
            "$([CX3](=O)[NX2-][OX2H1]),$(c1[n-]oc(=O)n1),$(c1onc(=O)[n-]1),"
            "$([nX2-]1oc(=O)nc1),$([nX2-]1nc(=O)oc1)"
            "]",
            1,
        ),
        ("fused_diazole_core", "c1nc2ccccc2n1", 1),
        ("basic_amine_linker", "[NX3;H0;!$(N[C,S]=[O,S,N]);!$(N=*);R]", 1),
        ("electron_poor_aryl_terminus", "[$(c[CX2]#[NX1]),$(c[F,Cl]),$(c[CX4](F)(F)F)]", 1),
    )
)


def danuglipron_analogues() -> list[Analogue]:
    """The designed set. Ordered parent, H1, H2, H3, then negative controls."""
    return [
        Analogue(
            label="parent",
            smiles=DANUGLIPRON_SMILES,
            hypothesis="H0-reference",
            rationale=(
                "Danuglipron itself. Included so every metric in the campaign has "
                "an in-set reference and so the 'zero modification' case can be "
                "shown to score identically to the parent -- the plumbing anchor."
            ),
            smiles_ionized=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)[O-])N=C2CN4CCC(CC4)C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
            ),
            net_charge=-1,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
        # ── H1: size / lipophilicity, to lower the efficacious dose ──
        Analogue(
            label="H1a-defluoro",
            smiles=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)"
                "C5=NC(=CC=C5)OCC6=CC=C(C=C6)C#N"
            ),
            hypothesis="H1-size-lipophilicity",
            rationale=(
                "Remove the benzylic ortho-fluorine. -18 Da and lowers cLogP. The "
                "F is a metabolic blocking group, so this trades H2 for H1 -- "
                "included precisely because that trade should be visible as "
                "opposite movement in the two liability axes."
            ),
            smiles_ionized=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)[O-])N=C2CN4CCC(CC4)C5=NC(=CC=C5)OCC6=CC=C(C=C6)C#N"
            ),
            net_charge=-1,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
        Analogue(
            label="H1b-des-oxetane-methyl",
            smiles=(
                "CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)"
                "C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
            ),
            hypothesis="H1-size-lipophilicity",
            rationale=(
                "Replace the (oxetan-2-yl)methyl N-substituent with a plain methyl. "
                "-56 Da, taking MW well under 500. The oxetane is a solubility/"
                "metabolism handle, so this is the aggressive size cut and is "
                "expected to cost fit if the oxetane oxygen makes a real contact."
            ),
            smiles_ionized=(
                "CN2C3=C(C=CC(=C3)C(=O)[O-])N=C2CN4CCC(CC4)C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
            ),
            net_charge=-1,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
        Analogue(
            label="H1c-azetidine-for-piperidine",
            smiles=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CC(C4)"
                "C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
            ),
            hypothesis="H1-size-lipophilicity",
            rationale=(
                "Contract the piperidine linker to an azetidine. -28 Da and one "
                "fewer rotatable bond, but it shortens the distance between the "
                "benzimidazole and pyridine vectors -- a direct test of whether "
                "the pocket tolerates a shorter linker."
            ),
            smiles_ionized=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)[O-])N=C2CN4CC(C4)C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
            ),
            net_charge=-1,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
        # ── H2: block oxidative soft spots / reactive-metabolite routes ──
        Analogue(
            label="H2a-difluoro-benzylic",
            smiles=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)"
                "C5=NC(=CC=C5)OC(F)(F)C6=C(C=C(C=C6)C#N)F"
            ),
            hypothesis="H2-metabolic-soft-spot",
            rationale=(
                "Geminal difluorination of the benzylic ether carbon -- the most "
                "likely CYP oxidation site, and the route to a quinone-methide-"
                "type reactive intermediate. Blocks it sterically and "
                "electronically at +36 Da (works against H1)."
            ),
            smiles_ionized=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)[O-])N=C2CN4CCC(CC4)C5=NC(=CC=C5)OC(F)(F)C6=C(C=C(C=C6)C#N)F"
            ),
            net_charge=-1,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
        Analogue(
            label="H2b-oxetane-to-gem-dimethyl-oxetane",
            smiles=(
                "CC1(C)CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)"
                "C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
            ),
            hypothesis="H2-metabolic-soft-spot",
            rationale=(
                "Gem-dimethyl on the oxetane guards against ring-opening "
                "hydrolysis/oxidation while keeping the oxygen. Tests whether the "
                "oxetane's liability is the ring strain rather than the O contact."
            ),
            smiles_ionized=(
                "CC1(C)CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)[O-])N=C2CN4CCC(CC4)C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
            ),
            net_charge=-1,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
        # ── H3: acid bioisosteres, to remove the acyl-glucuronide DILI route ──
        Analogue(
            label="H3a-tetrazole",
            smiles=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C4=NN=NN4)N=C2CN5CCC(CC5)"
                "C6=NC(=CC=C6)OCC7=C(C=C(C=C7)C#N)F"
            ),
            hypothesis="H3-acid-bioisostere",
            rationale=(
                "Tetrazole for CO2H: matched pKa and anionic geometry, cannot form "
                "an acyl glucuronide. The canonical carboxylate bioisostere; +25 Da "
                "and more lipophilic, so it works against H1."
            ),
            smiles_ionized=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C4=NN=N[N-]4)N=C2CN5CCC(CC5)C6=NC(=CC=C6)OCC7=C(C=C(C=C7)C#N)F"
            ),
            net_charge=-1,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
        Analogue(
            label="H3b-acylsulfonamide",
            smiles=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)NS(=O)(=O)C)N=C2CN4CCC(CC4)"
                "C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
            ),
            hypothesis="H3-acid-bioisostere",
            rationale=(
                "Methyl acylsulfonamide: acidic NH, similar pKa, an extended "
                "anionic surface. Retains a carbonyl so it is a weaker test of the "
                "glucuronide hypothesis than the tetrazole -- included as the "
                "intermediate case."
            ),
            smiles_ionized=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)[N-]S(=O)(=O)C)N=C2CN4CCC(CC4)C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
            ),
            net_charge=-1,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
        Analogue(
            label="H3c-oxadiazolone",
            smiles=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C4=NC(=O)ON4)N=C2CN5CCC(CC5)"
                "C6=NC(=CC=C6)OCC7=C(C=C(C=C7)C#N)F"
            ),
            hypothesis="H3-acid-bioisostere",
            rationale=(
                "1,2,4-Oxadiazol-5(4H)-one: acidic, planar, slightly larger than "
                "CO2H but less lipophilic than tetrazole. The compromise between "
                "H1 and H3."
            ),
            smiles_ionized=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C4=NC(=O)O[N-]4)N=C2CN5CCC(CC5)C6=NC(=CC=C6)OCC7=C(C=C(C=C7)C#N)F"
            ),
            net_charge=-1,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
        # ── negative controls: deliberately break the pharmacophore ──
        Analogue(
            label="NC1-methyl-ester",
            smiles=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)OC)N=C2CN4CCC(CC4)"
                "C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
            ),
            hypothesis="negative-control",
            rationale=(
                "Methyl ester of the potency-critical acid: removes the anionic "
                "anchor entirely while changing MW by only +14 Da. It CANNOT "
                "ionize, so it stays neutral at pH 7.4 while every real "
                "candidate carries -1 -- that charge difference IS the "
                "hypothesis under test, and it is why this control is the "
                "sharpest one in the set."
            ),
            is_negative_control=True,
            # Deliberately NO smiles_ionized: an ester has no ionizable proton.
            # net_charge stays 0, which is the entire point of this control.
            net_charge=0,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
        Analogue(
            label="NC2-decyano",
            smiles=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)"
                "C5=NC(=CC=C5)OCC6=CC=CC=C6"
            ),
            hypothesis="negative-control",
            rationale=(
                "Deletes both the nitrile and the fluorine from the distal aryl, "
                "removing the electron-poor terminus that packs near Trp33 while "
                "REDUCING MW (-43 Da). A fit metric that rewards this because it "
                "is smaller is inverted."
            ),
            is_negative_control=True,
            smiles_ionized=(
                "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)[O-])N=C2CN4CCC(CC4)C5=NC(=CC=C5)OCC6=CC=CC=C6"
            ),
            net_charge=-1,
            pharmacophore=GLP1R_PHARMACOPHORE,
        ),
    ]
