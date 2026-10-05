# Pose-handoff salvage: hypotheses (2026-10-05, written before measuring)

Porting the tests of the unmerged `fix/funnel-pose-handoff` branch onto main's
implementation (`funnel._harvest_geometry`, `tiers._embedded`,
`docking.united_atom.restore_hydrogens`, `context["point_charges"]`).

## H1. The unit of xtb's `pcharge` file

The branch says Angstrom; main's `xtb_engine._run_xtb` comment says Bohr. The
branch's evidence (tier 3 converted == xtb fed Angstrom) is a tautology: it
shows the conversion was applied, not which unit xtb reads.

Independent construction: Coulomb's law. NH4+ (charge +1) with a +1 point
charge at d = 20 file units from N. The leading term is q1 q2 / R in Hartree
with R in Bohr; polarisation is O(R^-4), well under 1e-3 Ha at this range.

* If xtb reads BOHR: dE ~ +1/20 = +0.0500 Ha.
* If xtb reads ANGSTROM: dE ~ +1/(20 x 1.8897) = +0.0265 Ha.
* Artifact (pcharge file ignored): dE = 0.

All three are distinct, so the probe can decide.

## H2. An MMFF stage between dock and the quantum tiers

`_harvest_geometry` writes every successful payload's geometry, and
`tier2_forcefield` re-embeds from SMILES and returns a geometry. In the
production stage list (dock -> mmff -> gfn2 -> dft):

* If harvesting keeps the docked pose: tier 3/4 see the docked coordinates.
* If it is last-writer-wins: tier 3/4 see tier 2's free-solution MMFF
  geometry (the branch's D1 defect, reintroduced).

Prediction from reading the code: last-writer-wins, so the ported test FAILS
on main.

## H3. Element check on an explicit heavy-atom map

`restore_hydrogens` checks the map is a permutation of the heavy-atom indices,
not that each docked atom's element matches its target. A reversed map on a
molecule whose heavy atoms are not all one element is a permutation.

* If guarded: raises.
* If not: returns a molecule whose heavy atoms sit on the wrong elements'
  positions, unless the stereo guard fires by luck (achiral molecule: it
  cannot).

Prediction: not guarded; the ported test FAILS on main.

## H4. The production Meeko path on a molecule whose orders differ

Through `tier1_dock` with only Vina's search faked (it returns the input
PDBQT, which Vina's real output preserves the order of), the restored pose
must have the molecule's own connectivity, perceived from the coordinates
alone (rdDetermineBonds), and the docked heavy atoms must not move.

Prediction: passes on main. Artifact check: the killer molecule's Meeko order
must actually differ from RDKit's, or positional assignment would pass too.
