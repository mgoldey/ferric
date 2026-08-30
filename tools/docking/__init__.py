"""Pose generation by search — the tier this pipeline was missing.

`tools/campaign` scores poses. It never PRODUCED them: conformers came from
free-solution ETKDG, ignoring the receptor, and were rigidly superimposed onto a
reference scaffold. Nothing ever proposed a pose and asked whether the pocket
liked it. Measured consequence (experiments/danuglipron/RESULTS.md M7): the best
of 20 generated conformers sat 2.23 A from the experimentally determined bound
pose, against a 2.0 A docking-success bar, and neither geometry optimization
(+0.04-0.16 A) nor 4 ps of GFN2 MD (+0.41 A) closed the gap.

This package supplies the search.
"""

from .vina_dock import DockResult, DockedPose, dock_ligand, prepare_receptor

__all__ = ["DockResult", "DockedPose", "dock_ligand", "prepare_receptor"]
