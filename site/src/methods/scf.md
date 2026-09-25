# SCF and DFT

Ground-state self-consistent field methods, their nuclear gradients, and the
things built on them: geometry optimization, harmonic frequencies,
transition-state search, implicit solvation and dispersion correction. The
`[scf]` and `[dft]` keys are in [Input file](../reference/input.md#scf);
grades are on [What is validated](../reference/validation.md).

## Hartree–Fock

**What it is.** RHF (closed shell), UHF and ROHF (open shell), with DIIS,
Schwarz screening, and a choice of direct, LinK, density-fitted (RI-J / RI-K)
or seminumerical (COSX) Fock builds (see *Choosing how exchange is built*
below). The open-shell solvers add per-spin DIIS, a virtual-block level shift,
augmented-Hessian Newton, and **Maximum-Overlap-Method (MOM)** orbital tracking
for near-degenerate cases (Gilbert, Besley & Gill 2008). These exist because
specific systems failed without them.

**Run it.** `method.kind = "rhf"` / `"uhf"` / `"rohf"` (`examples/water-rhf.toml`,
`examples/h_uhf.toml`); Python `ferric.run_rhf`, `run_uhf`, `run_rohf`. Spin
and charge are set on the molecule (`Molecule.from_xyz(path, charge,
multiplicity)`), not on the solver call.

**Defaults.** Exact four-index J and K (no density fitting) unless
`[scf] df_j_aux` / `df_k_aux` are set. Convergence gates on the density:
`density_conv = 1e-6` by default; `energy_conv` (default 1e-3) is a sanity
bound, not the convergence target.

**Accuracy.** Proven (`rhf`, `uhf`, `rohf`).

## Kohn–Sham DFT

**What it is.** Kohn–Sham DFT through **libxc**: LDA, GGA, hybrid and
range-separated hybrid functionals (for example PBE, B3LYP, ωB97X-V), **VV10**
nonlocal correlation, and the **SCAN / r2SCAN** meta-GGAs (τ-dependent; no
density Laplacian). The solvers cover RKS, UKS and ROKS.

**Run it.** `method.kind = "ksdft"` with `[dft] functional = "PBE"`
(`examples/water-wb97xv.toml`, `examples/benzene-dfb3lyp.toml`); Python
`ferric.run_dft` or `run_ksdft`. Both of these entry points are **closed shell
(RKS) only**. Open-shell KS is reachable as the reference inside `pdep-rpa`
and `gw` (`[rpa] xc` with multiplicity > 1), through
`run_frequencies(reference="uhf", xc=...)`, and through QM/MM
(`run_qmmm(method="uks")`); see [Capabilities](../reference/capabilities.md).

**Defaults.**

- Functional: LDA if `[dft] functional` is omitted.
- Grid: Becke partitioning of atom-centred Treutler–Ahlrichs radial ×
  Lebedev angular grids, **75 × 110** points per atom, **unpruned**.
- Pruning is **opt-in**: `[dft] grid_prune = "nwchem"`
  (`examples/water-pbe-pruned-grid.toml`) removes about 23% of points at
  75 × 110. It is accepted for `task = "energy"` only; optimization and
  frequency runs refuse a pruned grid because the gradient's grid-response
  term is built on the unpruned grid.
- Density fitting: **RI-J and RI-K are on by default** for `ksdft` and
  `run_dft` (`def2-universal-jkfit`), while `rhf` is exact by default. In
  Python, `df_j_aux=""` selects exact Coulomb. The fitting error grows with
  size: measured at PBE/STO-3G against exact J it was 0.28 kcal/mol for water,
  1.16 for benzene and 9.5 for a 71-atom drug molecule. Compare with an
  exact-Coulomb code only after turning fitting off, or at matched fitting.
- Meta-GGAs get an automatic 0.5 Ha virtual-block level shift (ramped to zero
  at convergence) when you set none, because τ amplifies grid noise and plain
  DIIS limit-cycles.

**Accuracy.** `ksdft` is Proven. SCAN/r2SCAN energies are pinned against PySCF
(`ferric-scf/tests/dft_scan.rs`); their closed-shell gradients against PySCF
with grid response (`dft_gradient_mgga.rs`). Other meta-GGA variants
(deorbitalized, +VV10, hybrid meta-GGA) are refused, not approximated.

## Dispersion correction: D3(BJ)

**What it is.** Grimme's D3 with Becke–Johnson damping, a post-SCF pairwise
correction, implemented natively in Rust. The two-body term only; the
three-body (ATM) term is not implemented.

**Run it.** `[dft] dispersion = "d3bj"` on `method.kind = "ksdft"`
(`examples/water-pbe-d3bj.toml`); `"d3bj(b3lyp)"` borrows another functional's
fitted damping parameters. Python: `run_dft(..., dispersion="d3bj")`, or
`ferric.d3bj_energy(mol, functional)` for the correction alone.

**Scope.** The damping parameters are fitted per functional, so the key is
refused on any method other than `ksdft`. `task = "optimize"` works (the D3
gradient is implemented; `ferric-d3/tests/gradient_vs_fd.rs`);
`task = "frequencies"` with dispersion is refused until the finite-difference
Hessian built from it is validated.

**Accuracy.** Agrees with simple-dftd3 1.6.0 to below 1e-12 Ha on water,
methane, benzene and an argon dimer with the three-body term off on both
sides (`ferric-d3/tests/vs_reference_dftd3.rs`). The omitted three-body term
is 0.1% of the two-body energy at benzene and grows with size.

## Implicit solvation

Two independent implementations. Both are threaded uniformly through RHF, UHF,
ROHF and the KS variants, and both leave the energy bit-identical to vacuum
when unset.

| Model | What it is | How to run it | Anchor |
|---|---|---|---|
| **IEF-PCM** (`ferric-pcm`) | Integral-equation PCM; modified Bondi radii (H 1.10 Å), atom-centred Lebedev spheres | CLI `[pcm] solvent = "water"` (or `epsilon`), energy runs of `rhf`, `uhf`, `rohf`, `ksdft` and `pdep-rpa` (`examples/water-pcm.toml`). Python `run_rhf(..., solvent=78.4)` or a solvent name; also `run_pdep_rpa`. | water / STO-3G, ε = 78.4: −3.813 vs PySCF IEF-PCM −3.8228 kcal/mol |
| **COSMO** (`ferric_scf::cosmo`) | Conductor-like screening | CLI `[cosmo] epsilon = 78.39` | water / cc-pVDZ, ε = 78.39: −5.955 vs PySCF COSMO −5.94 kcal/mol (`ferric-scf/tests/cosmo_water.rs`) |

Neither has cavitation, dispersion or repulsion terms, and neither has an
analytic gradient. An unknown solvent name is an error, not a silent vacuum.

## Gradients, optimization, frequencies, transition states

**Gradients.** Analytic nuclear gradients for RHF, UHF, ROHF and KS-DFT
(including meta-GGA, with grid response), checked against finite differences
and, for DFT, against PySCF. RI-MP2 also has an analytic gradient.

**Optimization.** `method.task = "optimize"` for `rhf`, `uhf`, `rohf`,
`ksdft`, `rimp2` and `pdep-rpa` (`examples/h2_opt.toml`,
`examples/h2-lda-opt.toml`). Python `ferric.run_optimize` is RHF.

**Harmonic frequencies.** `method.task = "frequencies"` for `rhf`, `uhf`,
`rohf` and `ksdft` (`examples/water-frequencies.toml`); Python
`ferric.run_frequencies(mol, basis_name, reference="rhf", xc=...)`. Built by
**central finite differences of the analytic gradient**: 6N gradient
evaluations, mass-weighted, translations and rotations projected out. The
displacement `[frequencies] delta` (default 5e-4 Bohr) is a real accuracy
knob; the printed Hessian asymmetry, zero in exact arithmetic, is the check
that it and the SCF thresholds suit the system.

**Analytic Hessians are not implemented.** The mpqc4 libint2 export has no
second-derivative integrals.

**Transition states and IRC (Python only, closed shell).**
`ferric.run_saddle(mol, basis_name, xc=...)` searches for a first-order saddle
by P-RFO, using two finite-difference Hessians with Bofill updates between
them; it costs `2(6N + 1) + (n_steps + 1)` gradient evaluations and refuses to
start from a geometry with no negative mode. `result.is_transition_state()`
requires both convergence and exactly one imaginary frequency.
`ferric.run_irc(mol, basis_name, result.imaginary_mode)` then follows the
reaction path both ways to show which minima the saddle connects (measured
about 71 gradients per branch on NH3 inversion). Both refuse
multiplicity ≠ 1.

## Choosing how exchange is built

Four ways to build K, and the choice is a real one, measured on this code.
Butane, one thread; TZ is def2-TZVP (184 functions), QZ is def2-QZVP (528).

| `[scf]` setting | What it is | Exact? | Scope |
|---|---|---|---|
| *(default)* | Schwarz-screened direct four-centre J + K | yes | all SCF types |
| `k_builder = "link"` | LinK: pair-list-screened direct K | yes (== direct to 9e-12 Ha, butane/def2-SVP) | RHF, UHF, ROHF |
| `df_j_aux` / `df_k_aux` | density-fitted J and K (RI-JK) | fitting error, grows with size (see *Kohn–Sham DFT* above) | all SCF types |
| `k_builder = "cosx"` | seminumerical (COSX) K on a grid | grid-dependent error, see below | RHF/UHF/ROHF, Coulomb operator only; analytic gradient for RHF, RKS, UHF (see below) |

**Time for one exchange build** (seconds, butane, one thread):

| Builder | What the time covers | def2-TZVP | def2-QZVP |
|---|---|---:|---:|
| direct | J and K together (one integral sweep) | not measured | 400 |
| LinK | K | being re-measured | being re-measured |
| RI-JK | K | 0.05 | 0.43 |
| COSX | K | not measured | 90 |

**Time for a full SCF** (seconds, butane, one thread):

| Builder | def2-TZVP | def2-QZVP |
|---|---:|---:|
| direct | 98 | not measured |
| LinK | being re-measured | being re-measured |
| RI-JK | not measured | not measured |
| COSX | 358 | not measured |

**Energy error against exact exchange:**

| Builder | System / basis | Error |
|---|---|---:|
| LinK | butane / def2-SVP | 9e-12 Ha |
| COSX, default grid | water / cc-pVDZ | 5e-6 Ha |
| COSX, default grid | butane / def2-SVP | 1.7e-4 Ha |
| COSX, default grid | butane / def2-TZVP | 1.2e-4 Ha |
| RI-JK | — | not measured on these systems |

**Start with density fitting** whenever its three-index tensor fits in memory
(`n_aux × n_bf² × 8` bytes: 1 GB for butane/QZVP, 7 GB for octane/QZVP). It
spills to disk when it does not. It is two to three orders of magnitude faster
per build than anything else here. Its error is a fitting error, not zero, and
it grows with system size.

**Need exact exchange?** Direct or LinK; both are exact to the screening
threshold as *K builders* (butane/def2-SVP: `link` == direct to 9e-12 Ha).
LinK's cost is being re-measured, so no LinK timing is quoted here.
`k_builder` is honoured by UHF and ROHF as well as RHF: the open-shell solvers
build K_α and K_β from one builder instance, refreshing its density-dependent
state per spin. Whether LinK is the faster choice for a given system is a
separate question from whether it is honoured — see the cost note above, which
is being re-measured. It is skipped with a warning, never silently, whenever
density-fitted J/K is active, the functional uses no exact exchange, or the
functional is range-separated (exchange then comes from the SR/LR fitters).

**COSX is for large basis sets on systems too big for RI-JK.** Its cost per
grid point barely moves with angular momentum while analytic exchange grows
roughly tenfold from SVP to QZVP, so it wins at high angular momentum, not at
large system size. Measured on one thread at the default grid:

- On butane, against exact direct exchange: at def2-TZVP the full COSX SCF is
  3.7× slower (358 s vs 98 s); at def2-QZVP the COSX K build takes 90 s,
  against about 400 s for the direct build's single J+K sweep, at a relative
  K error of 5.4e-5.
- On n-alkanes at def2-SVP, against LinK: slower at every size measured,
  1.59× (C20), 1.09× (C32) and 1.22× (C48), with no trend toward parity. At
  def2-TZVP on C20 it is faster (0.67×).

Below quadruple zeta it is the wrong tool unless RI-JK's three-index tensor
does not fit in memory.

A density-driven pair screen (on the product of the integral bound and the
local half-transformed density) keeps the K error below 2e-6 Ha at the default
threshold, and the half-transforms `D·X` and `X·Gᵀ` run over per-batch sparse
AO lists (`cosx_half_transform = "sparse"`, the default). At def2-SVP the K
build grows as N^1.29–N^1.32 between C20 and C48; that is one family of
molecules in one basis.

COSX's energy error at the default grid is about 5e-6 Ha on water/cc-pVDZ
(4.9e-6 in `cosx_k_anchors.rs`, 5.1e-6 in the ORCA comparison run),
1.7e-4 Ha on butane/def2-SVP and 1.2e-4 Ha on butane/def2-TZVP (against exact
exchange). ORCA 6.1.1's COSX at its own default grid gives 4.9e-6, 3.8e-5 and
1.3e-5 Ha on the same systems and bases, although ferric's default grid has
about twice as many points: ORCA evaluates its final energy once on a finer
grid, and ferric does not. Refining ferric's grid converges water to 3.4e-8 Ha,
but butane/def2-SVP stops at 3.4e-5 Ha at Lebedev-302, the finest angular grid
ferric supports; that residual is not yet explained. Reaction energies cancel
most of the error (0.02 kcal/mol on an isodesmic alkane reaction at the default
grid); absolute energies do not. Four knobs, all optional:

- `cosx_grid = { radial = 50, angular = 110 }` is the default and the coarsest
  grid that meets a 0.1 kcal/mol reaction-energy bar. Coarser grids fail it.
  The angular order matters most: on butane/def2-TZVP the error falls from
  1.2e-4 Ha at 110 points per shell to 4.0e-6 Ha at 302, while going from 50
  to 100 radial shells changes it by 1e-6 or less. Cost grows with the number
  of points.
- `cosx_overlap_fit = true` (default) applies the Izsák–Neese overlap
  correction. At the default grid it helps; on coarser grids it makes things
  *worse*, and its benefit is strongly molecule-dependent — large on water,
  nil to negative on ethane — so do not expect the factor quoted in the
  literature.
- `cosx_backend = "md3c1e"` (default) is the batched McMurchie–Davidson
  integral kernel. `"cosx-a"` is the per-point libint2 path, about three times
  slower and kept only as the cross-check the kernel is anchored against.
- `cosx_screen_thresh = 1e-7` (default) is the density-driven pair-screening
  threshold. `0.0` disables screening bit-identically; `1e-6` already fails a
  1e-6 Ha K-error bar on butane. Not available with the `"cosx-a"` backend.

Setting any `cosx_*` key without `k_builder = "cosx"`, or `k_builder` together
with `df_k_aux`, is refused or warned about rather than silently ignored.

**COSX gradients.** `task = "optimize"` and `task = "frequencies"` with
`k_builder = "cosx"` differentiate the COSX energy itself (grid-function,
ESP-integral and Becke-weight derivatives). With the default overlap fit the
fitted exchange is not variational in the orbitals, so the gradient adds an
orbital-response (Z-vector) term. The gradient is exact for RHF, RKS and UHF
with `cosx_overlap_fit = false`, and for RHF and UHF with the default fit;
measured against finite differences of the COSX energy it agrees to
2e-9–4e-9 Ha/Bohr. Fitted COSX with a Kohn–Sham functional, UKS, ROHF/ROKS
and pruned COSX grids are refused for gradient tasks before the SCF runs.

## Convergence

A few things worth knowing before debugging a stubborn SCF:

**Check `converged`.** These routines return a result whether or not they
converged; a non-converged SCF is a result with `converged = false`, not an
error. Downstream code that ignores the flag will happily consume a
half-converged density.

**Multiple solutions are real.** For systems like alkane chains, different
initial guesses converge to genuinely different SCF solutions — not a
convergence failure but a different basin. The guess picks the basin.

**Density-fitting has a noise floor.** DF-JK introduces an error floor that
makes energy-based convergence criteria below roughly 1e-9 meaningless; SCF
gates on the density RMS change instead.

**Near-linear-dependence.** Diffuse (aug-) basis sets on close-packed systems
can drive the overlap matrix near-singular; the canonical-orthogonalization
threshold is tunable via `FERRIC_LINDEP_THRESH`.

## Screening

- **Schwarz** bounds on every 4-centre path, built so they can never
  underestimate (zero-valued table entries are floored; left unfloored, one
  such entry was measured to cost 1.5e-4 Ha)
- **LinK** (Ochsenfeld, White & Head-Gordon 1998) — exchange via
  significant-pair and density-pair lists; its scaling is being re-measured
- **QQR** (Maurer, Lambrecht & Ochsenfeld 2012) is implemented and
  validated as a bound but is *not* used in production: on LinK it screened
  only 0.009% more quartets than Schwarz at alkane_16 (measured against a
  LinK pair-list implementation that skipped quartets; not yet repeated on the
  current lists)
- **COSX** shell-pair screening uses a primitive-level Hölder bound that
  provably never underestimates; an overlap-based bound can underestimate,
  which silently corrupts K

## QM/MM embedding

QM/MM (electrostatic, polarizable and Gaussian-smeared embedding, link atoms,
boundary-charge schemes, and a `[qmmm]` CLI section that reads a PQR) has its
own page: [QM/MM](../using/qmmm.md), including [its CLI section](../using/qmmm.md#from-the-cli).

## Determinism

The Fock build's reduction folds partial matrices in a **strict ascending group
order**, independent of thread count and of the memory band width. Results are
bit-identical across `RAYON_NUM_THREADS` — a property pinned by tests, not just
intended.

This matters more than it might seem: a tree-fold reduction would be equally
*deterministic* but would produce *different* bits, since floating-point
addition is not associative. The ascending order is load-bearing.

## Cite

DIIS: Pulay 1980. MOM: Gilbert, Besley & Gill 2008. LinK: Ochsenfeld, White &
Head-Gordon 1998. COSX: Neese et al. 2009; overlap fit: Izsák & Neese 2011.
CSB/CSAM screening: Thompson & Ochsenfeld 2017. DFT grids: Becke 1988,
Treutler & Ahlrichs 1995, Lebedev & Laikov 1999. Functionals through libxc
(Lehtola et al. 2018): PBE, B3LYP, ωB97X-V, SCAN, r2SCAN, VV10. D3(BJ): Grimme
et al. 2010, Grimme, Ehrlich & Goerigk 2011. IEF-PCM: Cancès, Mennucci &
Tomasi 1997; COSMO: Klamt & Schüürmann 1993. P-RFO: Banerjee et al. 1985;
Bofill 1994. Full entries in [References](../reference/references.md).
