# Pre-registered hypotheses — `optimize_qmmm` across boundary schemes, references and free sets

**Registered 2026-10-05, BEFORE any number in `validation_optimize_qmmm.rs` was
measured.** Issue #277. The harness extends `qmmm_gradient_at_minimum.rs`
(RHF × RCD × `MoveMm::All`, capped ethane, STO-3G) to Z1 / RC / Keep, UHF,
RKS/UKS (PBE, DF-J/K `def2-universal-jkfit`), `MoveMm::WithinRadius` and
`MoveMm::Residues`.

No external code defines this surface (partition, link scale and boundary
scheme are not standardized), so every check is INTERNAL: the analytic
`full_gradient_with_mm` against a central finite difference of the energy the
optimizer minimizes, with the partition rebuilt at every displaced geometry.
What the two paths share: the SCF, the integrals, the DF fit and the XC grid.
What they do not share: the chain rule (link `(1−g)/g`, midpoint ½/½), the
`mm_forces` electric-field contraction and the analytic SCF gradient. A
construction error in the PARTITION ITSELF (wrong charge on a midpoint, host
charge not deleted) is invisible to FD-vs-analytic, because both sides see the
same wrong partition; those are guarded by structural assertions (total
embedding charge, host charge zero) and by the scheme-swap negative controls.

## Anchor (runs first)

A0. With no MM atoms and `MoveMm::None`, `optimize_qmmm` for each of
Rhf / Uhf / Rks("PBE") / Uks("PBE") reproduces `optimize_geometry` /
`optimize_geometry_uhf` (with `xc` and the same DF aux set by hand) **bit for
bit** in energy and step count.
- If real: identical bits. If the closure's config differs from the plain
  driver (aux basis, screening, gradient routing): a difference at 1e-10..1e-6.

## H1 — HF gradient is correct under every boundary scheme

- If real: off-minimum FD-vs-analytic relative residual ≤ 1e-6 on every probed
  row, growing ~h² over h ∈ {5e-5, 1e-4, 2e-4} (FD truncation), as measured
  for RCD (4.1e-7).
- If a scheme-specific fold is broken: ≥ 1e-3 relative on the host / M2 rows,
  flat in h. The two predictions are separated by ≥ 3 decades, so the
  experiment distinguishes them.

## H2 — Keep is physically pathological, not a gradient bug

The Keep link H sits 0.44 Å from the full host charge.
- Physics prediction: the gradient is still CORRECT at fixed geometries
  (FD agrees as in H1), while the optimization may fail to converge or end at a
  geometry distorted by the near-contact charge. Record
  `min_link_to_charge_distance()` and the step count; do not loosen anything to
  make it converge.
- Artifact prediction: if FD disagrees for Keep only, that is a code defect in
  the Keep path, not physics. These are distinguishable.

## H3 — UHF open-shell boundary (ethyl radical, CH3• capped)

- If real: same FD floor as H1; ⟨S²⟩ constant (≤ 1e-4 drift) over every
  displaced geometry; stability check Stable at the start and the end.
- Artifact (state flip during FD): one probe row with a residual orders of
  magnitude above the others, coinciding with an ⟨S²⟩ jump.

## H4 — KS (PBE, DF-J/K, Becke grid) floor

The KS gradient carries grid response and differentiates the DF energy the
SCF computed, so FD and analytic should share one surface.
- If real: an h-independent absolute floor set by SCF convergence and grid
  noise, expected 1e-7..1e-6 Ha/Bohr, i.e. looser than HF's 2e-7 at the
  minimum but still ≥ 3 decades below a fold error off the minimum.
- If the gradient ignored DF (exact-J derivative of a DF energy) or omitted
  grid response: an h-independent floor of 1e-5..1e-4 and a nonzero FD
  derivative at the optimizer's "minimum". Distinguishable from the above.

## H5 — `MoveMm::WithinRadius` freezes what it says, once

- If real: atoms outside r at the START keep bit-identical coordinates; their
  full-gradient rows are nonzero at the returned geometry; free rows are
  stationary.
- A per-step re-evaluation of the free set is an identity unless membership
  changes during the run, so the case is built so that it WOULD change (an MM
  atom starts outside r and the relaxed QM region moves toward it). If the
  measured trajectory does not cross r, that mutation is recorded as UNREACHED,
  not as caught.

## H6 — negative controls must fail

- Analytic gradient of a DIFFERENT scheme compared against the FD of this
  scheme's energy misses by ≥ 10× the bar (e.g. RC fold fed to a Z1 system).
- A broken MM-row fold (midpoint 1.0/0.0, or M2 own-charge force dropped)
  misses on the MM rows but PASSES a QM-only probe set — which is what makes
  the MM-row probes load-bearing.

## Results

(Filled in after measurement; see the trailing tables of
`validation_optimize_qmmm.rs`.)
