# Pharma use-case coverage

Every named use case, what runs it, what it costs, and the plot that answers
it. Costs are MEASURED on this project; where a figure depends on basis or on
warm-vs-cold, both are given, because quoting one hides a 3-30x spread.

## Coverage

| use case | entry point | cost (measured) | plot |
|---|---|---|---|
| docking | `docking.vina_dock`, `tiers.tier1_dock` | **31 s** @ 57 atoms, 5.7 @ 21, 1.9 @ 9 (ex=4, 7LCJ); ~N^1.5 | `pose_ensemble`, `funnel_survival` |
| docking geom opt | `active_site.pose_relaxation` | **77.8 s/step** @ 71 atoms in a 6458-charge pocket | `optimization_trace` |
| minima with FF | `tiers.tier2_forcefield` | **9 ms** @ 21 atoms (2.2 ms @ 9 atoms, 8.2 @ 19, 21.6 @ 34) | `tier_comparison` |
| minima with xtb | `tiers.tier3_gfn2` | **39 ms** @ 21 atoms (0.152 s @ 9, 0.050 @ 19) | `tier_comparison` |
| score with DFT | `tiers.tier4_dft` | **2.6 s** @ 9 atoms at the def2-svp DEFAULT (0.75 s at STO-3G) | `tier_comparison` |
| transition state | `ferric.run_saddle` | `2*(6N+1) + (n_steps+1)` gradients | `imaginary_mode` |
| reaction path (IRC) | `ferric.run_irc` | ~70 gradients/branch | `reaction_path` |
| common substitutions | `pipeline.substitution` | **7.6 ms** warm / 7 proposals (248 ms first call) | `site_substituent_heatmap` |
| toxicology | `tox.alerts`, `tox.assess` | **3.7 ms** screen; **54 ms** offline assess, **1.6 s** with the default `include_web=True` | `liability_profile` |
| binding energy in site | `active_site.binding_energy` | **137 s** @ 71 atoms / 6458 charges (TWO SCFs + pdb2pqr) | `pocket_polarization` |
| QM/MM setup | `ferric.QmmmSystem` | free | `qmmm_partition` |
| dispersion D3(BJ) | `run_dft(dispersion="d3bj")` | microseconds, energy and gradient | folded into the DFT energy |

## Where the campaign time actually goes

The whole funnel RUN end to end — 10 substitution candidates of benzoic acid
through dock → FF → xtb → DFT against the 7LCJ pocket, keeping 6/4/2/1, zero
failures, same survivor both times:

| basis | total | dock | FF | xtb | DFT |
|---|---:|---:|---:|---:|---:|
| STO-3G | 45.0 s | **72.7%** | 0.1% | 0.3% | 27.0% |
| **def2-svp (the `tier4_dft` DEFAULT)** | 82.1 s | **39.8%** | 0.1% | 0.1% | **60.0%** |

**"Docking dominates, not DFT" holds only at STO-3G.** At the default basis the
ranking inverts and DFT is the majority of the run. The absolute docking cost
is identical between the rows (32.7 s); it is DFT that moves, because
`tier4_dft` defaults to def2-svp and that is 3.5x STO-3G.

So: quote the share WITH the basis, and decide where to optimize from the row
that matches the basis you actually run.

## What "has a plot" does and does not mean

The plot has to answer *that* use case's question. A transition-state search
produces an imaginary MODE — a 3N vector — so `imaginary_mode` shows whether it
displaces the reacting atoms, which is the second and non-optional half of
verifying a saddle. One imaginary frequency is necessary, not sufficient: a
methyl rotor gives one too.

**A plot and a cost do not license a RANKING.** The binding-energy row has
both and still cannot order two analogues: every pose protocol tried is closed
(RESULTS.md M4-M13), and the best available ddE noise is ~4.07 kcal/mol against
substituent effects of 1-2. `site_substituent_heatmap(noise_floor=...)` greys
out every cell inside that limit so a figure cannot imply otherwise.

## Known gaps

- **AMBER `prmtop`** — no reader; go through OpenMM.
- **Periodic boundary conditions** — absent. `solvate()` gives a finite
  droplet with a vacuum boundary.
- **QM/MM dispersion** — D3/D4/XDM/VV10 are QM-atom-pairwise, so dispersion
  between the QM region and MM charges is absent.
- **Pose-ensemble ranking** — see above; this is a noise floor, not a missing
  feature.
