# Danuglipron toxicity-reduction experiment plan

**Date:** 2026-08-29  **Status:** executing

## Question

Danuglipron (PF-06882961, Pfizer oral GLP-1R agonist) was discontinued in April 2025
after a Phase-2b/3 program showed dose-dependent GI adverse events (nausea/vomiting
up to ~73% at high dose) and, in the twice-daily program, an idiosyncratic
drug-induced liver injury (DILI) signal that ended the once-daily program too.

Two separable questions, and only one of them is a conformer question:

1. **Conformer question.** Does the bound (bioactive) conformer differ enough in
   energy from the free-solution minimum that a *strain penalty* is being paid? A
   large strain penalty means potency is bought with concentration, i.e. higher dose,
   i.e. more exposure-driven GI tox. Reducing strain is a tox lever that does not
   change the molecule at all.
2. **Morphology question.** Which structural modifications reduce predicted
   toxicity liability while retaining active-site fit? DILI/GI liability here is
   *structural* (the aryl-nitrile + benzimidazole-carboxylic-acid core), so this
   needs analogues, not conformers.

## Design

### Toxicity: external, not ours
ferric computes electronic structure; it has no business predicting DILI. Toxicity
comes from **external** sources, in descending order of preference:
- **ADMETlab 3.0** (`admetlab3.scbdd.com`) — REST endpoint, ~90 endpoints incl.
  DILI, hERG, Ames, H-HT, and GI-relevant absorption. Primary.
- **ProTox-3.0** (`tox.charite.de/protox3`) — organ-specific tox, LD50. Secondary.
- **RDKit FilterCatalog** (Brenk / PAINS / NIH structural alerts) — offline,
  deterministic, always available. Used as the *always-runs* baseline so the
  experiment produces a ranking even if both web services are down or rate-limited.

We do NOT train a tox model. We do NOT claim a tox number is ours.

### Fit: ours
Active-site fit is where ferric earns its place, via the existing
`tools/active_site` pipeline on PDB 7LCJ (2.82 Å cryo-EM, ligand UK4):
- pocket -> PDB2PQR AMBER point charges (once)
- classical prescreen (free) -> rank ensemble
- QM-in-field single points on survivors
- in-field geometry relaxation for the strain number

### Arms

| Arm | What | Cost tier |
|-----|------|-----------|
| **A** | Conformer strain: 20-member existing ensemble, GFN2-xTB in-vacuo + in-field relax, bound-vs-free strain | xTB (seconds/conf) |
| **B** | Morphology: ~12-20 designed analogues addressing specific alerts, RDKit-embedded, prescreened, xTB-refined | xTB |
| **C** | External toxicity on parent + all analogues | network |
| **D** | Ranking: Pareto front over (predicted tox liability, active-site fit loss, strain) | free |

### Exactness anchors (write BEFORE measuring — repo protocol)

Per CLAUDE.md's Experimental Protocol, each new component gets a trivial-limit test
committed before any sweep runs:

- `Molecule.coords()` round-trips `from_xyz` byte-for-byte (new binding).
- Relaxing an already-relaxed pose moves it < tolerance (idempotence).
- An empty pocket (zero charges) makes in-field energy == vacuum energy exactly.
- A zero-modification "analogue" (parent SMILES) scores identically to the parent
  in the tox layer AND the fit layer — catches the whole analogue plumbing.
- Tox layer with an offline/failed service must return `None`, never 0.0
  (a fabricated 0 would rank a molecule as maximally safe).

### Artifact hypothesis (stated before measuring)

- *If real:* strain penalties scatter across conformers (a few kcal/mol spread with
  outliers); analogue fit-loss correlates with which pocket contact was disturbed,
  not with molecular size.
- *If broken:* every conformer shows near-identical strain (means the field isn't
  being applied, or the relax isn't moving), or fit-loss tracks atom count
  monotonically (means we're measuring total energy difference, not interaction).
  These are distinguishable, so the experiment is admissible.

## Known gaps to build

1. `Molecule.coords()` / `.coords_bohr()` / `.symbols()` pyo3 getters — **blocks
   Arm A entirely** (relaxed geometry currently unretrievable; `RelaxedPose.
   coords_angstrom` is hardcoded `None`).
2. xTB not exposed to Python at all (`ferric-xtb` is Rust-only, feature-gated OFF).
   Needed for the cheap tier; without it every conformer costs a DFT SCF.
3. No toxicity module anywhere in the repo.
4. No analogue enumeration / morphology module.
