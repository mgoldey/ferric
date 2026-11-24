# SCF and DFT

Ground-state self-consistent field methods, their nuclear gradients, and the
things built on them: geometry optimization, harmonic frequencies,
transition-state search, implicit solvation and dispersion correction. The
`[scf]` and `[dft]` keys are in [Input file](../reference/input.md#scf);
grades are on [Capabilities and validation](../reference/validation.md).

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
bound, not the convergence target; a value below the estimated noise floor of the
energy change draws a warning (see [sharp bits](../using/sharp-bits.md)).

**Accuracy.** Proven (`rhf`, `uhf`, `rohf`).

## Kohn–Sham DFT

**What it is.** Kohn–Sham DFT through **libxc**: LDA, GGA, hybrid and
range-separated hybrid functionals (for example PBE, B3LYP, ωB97X-V), **VV10**
nonlocal correlation, the **SCAN / r2SCAN / TPSS** meta-GGAs (τ-dependent;
no density Laplacian), and the hybrid meta-GGAs **TPSSh, M06-2X, SCAN0,
r2SCAN0** and **ωB97M-V** (range-separated, with VV10).
Any libxc meta-GGA identifier (`MGGA_*`, `HYB_MGGA_*`) is accepted unless libxc
flags it as needing the density Laplacian (`XC_FLAGS_NEEDS_LAPLACIAN`, e.g.
SCAN-L, BR89), which is refused with a message naming that component. A hybrid meta-GGA
takes its exact-exchange fraction, and range separation if any, from libxc,
the same way the hybrid GGAs do. The solvers cover RKS, UKS and ROKS.

**Run it.** `method.kind = "ksdft"` with `[dft] functional = "PBE"`
(`examples/water-wb97xv.toml`, `examples/benzene-dfb3lyp.toml`); Python
`ferric.run_dft` or `run_ksdft`. The two Python entry points are **closed shell
(RKS) only**. In the CLI, `ksdft` on a molecule with multiplicity > 1 runs UKS,
and `kind = "uhf"`/`"rohf"` with `[dft] functional` run UKS/ROKS. Open-shell
KS is also the reference inside `pdep-rpa` and `gw` (`[rpa] xc` with
multiplicity > 1), and in Python through
`run_frequencies(reference="uhf", xc=...)` and QM/MM
(`run_qmmm(method="uks")`); see [Capabilities and validation](../reference/validation.md).

**Defaults.**

- Functional: LDA if `[dft] functional` is omitted.
- Grid: Becke partitioning of atom-centred Treutler–Ahlrichs radial ×
  Lebedev angular grids, up to **75 × 110** points per atom, **pruned** by
  default (`grid_prune = "nwchem-radial"`: NWChem angular regions, and 50
  radial shells for H and He, 66 for Li–Ne, 75 for heavier atoms) with a
  **1e-14 weight cutoff** (`grid_weight_cut`, ORCA's `WeightCut`). That is
  42-47% fewer points than the flat 75 × 110 grid on organic molecules, with
  energies within 2e-7 to 4e-6 Ha of it on PBE/def2-SVP (the flat grid's own
  error against a 99 × 590 grid is 4.5e-4 to 3.3e-3 Ha on C10–C32 alkanes).
- The flat 75 × 110 grid is `grid_prune = "none"` with `grid_weight_cut = 0`.
  `"nwchem"` prunes the angular grid only (about 23% fewer points,
  `examples/water-pbe-pruned-grid.toml`); `"defgrid1"`–`"defgrid3"` are
  ORCA 6.1's DEFGRIDs, with their own sizes: fewer radial shells than the
  default (H 30/35/44, Li–Ne 38/44/53, Na–Ar 46/54/63 against 50/66/75) but
  more angular points (peak orders 194/302/590 for heavier atoms and 110/194/434
  for H and He, against 110). `defgrid2`, ORCA's default, moved PBE/def2-SVP
  energies by 1e-3 to 4e-3 Ha from the flat grid. Setting only `grid_radial` /
  `grid_angular` keeps the default scheme at that size. Optimization,
  frequency (including the analytic moving-grid Hessian) and TDDFT runs use
  the grid the SCF ran on.
- SCAN is the exception to "pruning costs nothing". On water/6-31G its
  gradient error against a 99 × 590 grid is 2.1e-4 Ha/Bohr on the flat grid
  and 5.6e-4 on the default, because the hydrogens lose radial shells.
  r2SCAN (3.7e-4 flat, 3.6e-4 default) and PBE (unchanged) are not
  affected. For SCAN, use `grid_prune = "nwchem"` or a larger grid.
- Density fitting: **RI-J and RI-K are on by default** for `ksdft` and
  `run_dft` (`def2-universal-jkfit`), while `rhf` is exact by default. In
  Python, `df_j_aux=""` selects exact Coulomb. The fitting error grows with
  size: measured at PBE/STO-3G against exact J it was 0.28 kcal/mol for water,
  1.16 for benzene and 9.5 for a 71-atom drug molecule. Compare with an
  exact-Coulomb code only after turning fitting off, or at matched fitting.
- Meta-GGAs get an automatic 0.5 Ha virtual-block level shift (ramped to zero
  at convergence) when you set none, because τ amplifies grid noise and plain
  DIIS limit-cycles.

**Accuracy.** `ksdft` is Proven. SCAN/r2SCAN/TPSS energies are pinned against PySCF
(`ferric-scf/tests/dft_scan.rs`); the SCAN/r2SCAN closed-shell gradients against
PySCF with grid response (`dft_gradient_mgga.rs`). TPSS gradients have no
PySCF pin. Hybrid meta-GGA energies match PySCF to 1e-13..2e-13 Ha with
exact J/K on the flat 75 × 110 grid (TPSSh, M06-2X, SCAN0; H2O RKS, NH2 UKS,
TPSSh NH2 ROKS, cc-pVDZ) and wB97M-V to 6e-8 Ha with RI-J/K
(`dft_hybrid_mgga.rs`); their RKS and UKS gradients match PySCF with grid
response to 4e-8 Ha/Bohr and finite differences of ferric's energy to 1.4e-7
(TPSSh, M06-2X, 6-31G; `dft_gradient_hybrid_mgga.rs`). The response consumers
take the τ kernel plus the libxc fraction of exact exchange: the TPSSh internal
stability eigenvalue matches PySCF's dense orbital Hessian to 6e-9 (RKS H2O)
and 7e-10 Ha (UKS NH2), Newton lands on the DIIS energy, the MBD@rsSCS gradient's Z-vector
(TPSSh, RKS H2O/STO-3G) matches finite differences of the full SCF pipeline to
3.5e-9 Ha/Bohr, and TDA/Casida
excitations of water/6-31G match PySCF to 3e-7 eV (TPSSh) and 1e-6 eV
(M06-2X). Laplacian-dependent
meta-GGAs are refused, not approximated. Periodic (PBC) routes refuse hybrid
meta-GGAs. The
\\( V_{xc} \\), \\( f_{xc} \\), VV10, Becke-weight and gradient formulas are
derived in [DFT derivations](./dft-derivations.md).

### RI-J tensor precision

The RI-J three-index tensor is stored as **f32 by default for single
points** and **f64 by default for geometry and frequency tasks**
(`[scf] ri_j_precision = "f32"|"f64"`; Python `ri_j_precision=`, or
`scf_ri_j_precision=` on drivers with more than one iterative stage;
library `RhfConfig::ri_j_precision`). Each integral is rounded to the
nearest f32 once; both RI-J contractions multiply the stored values by f64
vectors and accumulate in f64, and the fitted density, the metric solve and
J itself stay f64. The tensor takes half the memory. `"f64"` stores the
tensor in f64 and reproduces the former default bit for bit (measured:
identical energies, gradients and iteration counts for water and
sulfamethoxazole, PBE and B3LYP, and RHF with RI-J). An explicit value
always wins. Omitted, the precision is resolved in one place
(`ferric_scf::df_j::resolve_ri_j_precision`, from `RhfConfig::ri_j_task`):

| task | omitted `ri_j_precision` resolves to |
|---|---|
| CLI `task = "optimize"` / `"frequencies"`; Python `run_optimize`, `run_optimize_uhf`, `run_optimize_qmmm`, `run_frequencies`, `run_saddle`, `run_irc`; library `optimize_geometry*`, `harmonic_frequencies*`, `optimize_qmmm` | f64 (geometry-task default) |
| everything else (energies, SCF, gradients at one geometry, TDDFT, GW, MP2, ...) | f32, or f64 under a `[gpu]` mode that may build J on the device (it needs the f64 tensor) |

The geometry default is f64 because the f32 surface is jagged (below):
optimizers and finite-difference Hessians compare energies and gradients at
nearby geometries, where the f32 jitter is as large as the convergence
criteria. `ri_j_precision = "f32"` in an optimisation is still honoured.

An f32 request falls back
to f64 (with a notice) when the f32 tensor does not fit in memory or
`jk_storage` is `disk`/`direct`; Split-RI-J has no tensor. The free-atom
SCFs of the MINAO guess always use f64.

The SCF converges on the operator built from the rounded tensor, so the
energy carries a first-order error. Gradients, optimisation, Hessians and
frequencies **follow the SCF's precision**: they form the fitted density
from the same rounded integrals (derivative integrals stay f64), so they
are the gradient of the rounded-tensor energy with each integral's rounding
held fixed. The rounding itself is not a smooth function of the geometry:
moving an atom re-rounds every integral, so the f32 energy surface carries a
small jitter that no analytic gradient can follow. Measured by central finite
differences of the PBE/def2-SVP energy (SCF converged to 1e-10 Ha): at a
2e-3 Bohr step the f32 FD-minus-analytic gradient was up to 4.9e-5 Ha/Bohr
on water and 2.1e-4 Ha/Bohr on sulfamethoxazole, against 5e-7 and 9.5e-7 with
f64, i.e. an energy jitter of about 2e-7 Ha (water) and 9e-7 Ha
(sulfamethoxazole) between nearby geometries; at a 5e-3 Bohr step the f32
residual fell to 3.6e-6 and 2.2e-5 Ha/Bohr. That jitter is the size of the
optimiser's default energy-change criterion (1e-6 Ha), which is why geometry
and frequency tasks default to f64; for your own finite-difference work on
single-point energies, set `ri_j_precision = "f64"`. The precision used and
why is recorded in the certificate (`ri-j` entry: "aux …, f64 tensor storage
(requested f64: geometry-task default)"; the reason is `explicit`,
`geometry-task default`, `gpu default` or `default`), the run log (the
`run_start` header's `ri_j_precision_resolved` / `ri_j_precision_source` /
`ri_j_task`, each `ri_j_tensor` record's `precision_source`, plus a
`dfj-f32` entry in the precision record) and `result.ri_j_precision` in
Python.

**The default f32 tensor and the memory budget.** The f32 tensor is held in
memory only. When it does not fit the memory budget, an f32 request (the
default included) keeps the f64 tensor and says so on stderr ("…
`ri_j_precision = "f32"` (default) but the tensor stays f64 …") and in the
certificate. That choice reads the budget alone, never what else the shared
memory pool happens to hold, so one input under one budget always gets one
precision. Because the default switches precision with the
budget, an energy (and everything correlated on that reference) can move by
up to ~1e-5 Ha between a budget that fits the f32 tensor and one that does
not: the f32 and f64 tensors of benzene/aug-cc-pVDZ (def2-universal-jkfit)
give RHF references 5.3e-6 Ha apart and rs-mp2-rpa totals 7.8e-6 Ha apart.
Set `ri_j_precision = "f64"` for energies that do not depend on the budget
(with it, the same benzene run agrees to 7e-13 Ha from 0.12 GiB to an
unset budget). The f32 default is kept: it halves the tensor and its error
is the measured one below.

Measured, def2-SVP, `def2-universal-jkfit`, default grid and thresholds,
f32 minus f64 (2026-11-07):

| system | atoms | ΔE (Ha) | ΔE / atom | max \|Δg\| (Ha/Bohr) | rms Δg | SCF iterations f64 / f32 |
|---|---|---|---|---|---|---|
| water, PBE | 3 | −1.00e-6 | −3.3e-7 | 6.7e-8 | 3.4e-8 | 9 / 9 |
| water, B3LYP | 3 | −1.00e-6 | −3.3e-7 | 6.7e-8 | 3.4e-8 | 8 / 8 |
| H2O⁺ doublet, UKS PBE | 3 | −1.09e-6 | −3.6e-7 | 8.5e-8 | 4.9e-8 | 8 / 8 |
| sulfamethoxazole, PBE | 28 | +2.57e-6 | +9.2e-8 | 3.0e-6 | 5.3e-7 | 15 / 15 |
| sulfamethoxazole, B3LYP | 28 | +2.51e-6 | +9.0e-8 | 3.1e-6 | 5.6e-7 | 13 / 13 |
| sulfamethoxazole⁺ doublet, UKS PBE | 28 | +2.50e-6 | +8.9e-8 | 2.6e-6 | 5.1e-7 | 20 / 20 |
| danuglipron, PBE | 71 | +5.37e-6 | +7.6e-8 | 2.2e-6 | 5.4e-7 | 16 / 16 |
| danuglipron, B3LYP (RIJCOSX, energy only) | 71 | +5.09e-6 | +7.2e-8 | — | — | 15 / 18 |

Δg is f32 minus f64 of the analytic gradient (each at its own converged
density). Cost, danuglipron PBE/def2-SVP, CLI, 6 threads, 3 runs each on a
quiet box: the SCF peak RSS fell from 6.44 to 4.50 GB and its wall time from
24.1 to 23.7 s (CPU time 110 to 102 s); a one-step optimisation (SCF +
gradient) peaked at 4.68 against 6.55 GB, with the same wall time (122 s). For danuglipron B3LYP the RI-JK tensors do not fit a 14 GB memory
cap, so it was run as RIJCOSX (`k_builder = "cosx"`), energy only.

The error is µEh-scale and roughly size-extensive (≈1e-7 Ha per atom on the
drug-sized molecules), two or more orders of magnitude below the RI-J
fitting error against exact Coulomb quoted above (4.5e-4 Ha for water,
1.5e-2 Ha for a 71-atom drug, PBE/STO-3G). At a fixed geometry it is
deterministic (reproduced bit for bit across runs). The SCF iteration count
was unchanged in seven of the eight runs; danuglipron B3LYP/RIJCOSX took 18
instead of 15. f32 rounding also breaks exact point-group degeneracy at the
1e-8 level (benzene/def2-SVP HF with RI-J: the e1g orbital pair split by
9.5e-9 Ha). Compare f32 and f64 results with each other only at this level,
and use `"f64"` when validating against an f64 reference code at tighter
bars, for finite differences of energies, and for symmetry-blocked work that
needs exact degeneracies.

### RI-K tensor storage

The RI-K (DF-K) builder keeps the metric-dressed three-index tensor
`B = M (P|μν)`, `MᵀM = V⁻¹`, for the whole SCF. `M` is the Cholesky `L⁻¹`
when the factor certifies that no metric eigenvalue lies near the `1e-10`
linear-dependence cut, and otherwise the truncated `V^{-1/2}` from an
eigendecomposition (which drops those modes); the two give the same K up to
rounding (about `1e-13` relative), and the Cholesky route is several times
cheaper to set up. `FERRIC_DFK_METRIC=eigh` forces the eigendecomposition; it
is a debug escape hatch for comparing the two routes, not a tuning option
(`auto`, the default, is the only other value; anything else is an error). That tensor is symmetric in its two orbital indices, and it is
now held as the **packed symmetric half** (`n_aux × n_bf(n_bf+1)/2`
elements) rather than the full square: half the memory at f64. Both K
contractions read only one triangle (the occupied path unpacks each
auxiliary row into the triangle its symmetric matrix product reads; the
density path unpacks a few rows at a time), so K is bit-for-bit the square
tensor's. `[scf] ri_k_precision` (Python `ri_k_precision=`, or
`scf_ri_k_precision=` on drivers with more than one iterative stage;
library `RhfConfig::ri_k_precision`) chooses the stored precision:

- `"f64"` (default): the packed f64 tensor. Energies, iteration counts and
  gradients are those of the former default bit for bit
  (`ferric-scf/tests/dfk_packed_storage.rs`).
- `"f32"`: each stored element rounded once to the nearest f32, a quarter of
  the former memory; the K contractions still run in f64. The SCF converges
  on the rounded operator, so the energy carries a first-order error,
  bracketed by the variational sandwich the tests check.

When J and K share one auxiliary basis, the raw tensor RI-J keeps is held
packed instead of unpacked whenever the unpacked one would not leave room
for the dressed band beside it. When even the packed raw tensor and the
dressed band do not fit `[memory] budget_gb` together but the band does
("dress first", `jk_storage = "auto"` or `"split"`, f64 RI-K, one rank,
Coulomb operator), the raw tensor is not held at all: its integrals are
computed once in 128-row blocks and dressed straight into the packed band
(the same matrix products in the same order, so K is bitwise unchanged).
J then gets its f32 tensor by a second pass over the integrals when
`ri_j_precision = "f32"` and it fits beside the band (bitwise the J the
shared tensor would have given), and Split-RI-J otherwise; a notice names
the route. A `[gpu]` mode that may build K on the device keeps the square
f64 tensor (the device upload reads it). A band that does not fit in memory
is still spilled (packed f64 on disk), with a notice when `"f32"` was asked
for.

Measured, def2-SVP, `def2-universal-jkfit`, `density_conv = 1e-8`, CLI,
f32 minus f64 RI-K storage (2026-11-08; RI-J f64 in both):

| system | ΔE (Ha) | SCF iterations f64 / f32 |
|---|---|---|
| water, B3LYP | −7.7e-9 | 10 / 10 |
| water, PBE0 | −9.6e-9 | 10 / 10 |
| water, CAM-B3LYP | +2.6e-9 | 11 / 11 |
| H2O⁺ doublet, UKS B3LYP | −6.5e-9 | 10 / 10 |
| sulfamethoxazole, B3LYP | +8.6e-8 | 19 / 19 |
| sulfamethoxazole, PBE0 | +1.08e-7 | 19 / 19 |
| sulfamethoxazole, CAM-B3LYP | +1.0e-8 | 19 / 19 |
| sulfamethoxazole triplet, UKS B3LYP | +8.6e-8 | 45 / 45 |
| water, HF / cc-pVDZ (RI-JK) | 2.1e-7 | — |

The error is an order of magnitude below the RI-J f32 error above, but it
has the same character: the rounding is re-drawn at every geometry, so the
f32 energy is not smooth at that level. The analytic gradient recomputes the
f64 integrals, so it is the gradient of the f64 functional at the f32 SCF's
density: 7.5e-9 Ha/Bohr from the f64 SCF's gradient on HF water/cc-pVDZ,
while a central finite difference of the f32 energy at h = 1e-3 Bohr missed it
by 2.0e-4 Ha/Bohr (f64: 1.3e-7). Removing that jitter needs an f64 energy
exit (the f64 functional evaluated once at the f32 density), which is not
implemented; this is why `"f32"` is opt-in. Use it for single points and
analytic-gradient work where the memory matters; keep `"f64"` for finite
differences of energies, energy-based line searches and tight scans. The
precision used is recorded in the certificate (`ri-k` entry: "aux …, f32
tensor storage"), the precision record (`dfk-f32`), the checkpoint route
and `result.ri_k_precision` in Python.

Memory and wall time, B3LYP, `[memory] budget_gb = 12` (12.88 GB), RI-J
f64 in every run, CLI, 6 threads, each run started with ≤ 1.5 busy cores
(2026-11-08; "before" is the square tensor, from the 2026-11-07 scaling
runs). "spills" means the dressed tensor was written to disk and the run was
stopped there (each iteration then re-reads it, about a minute of IO per
iteration for danuglipron):

| system (n_bf) | before: peak RSS, wall | `"f64"`: peak RSS, wall | `"f32"`: peak RSS, wall |
|---|---|---|---|
| C32H66 / 6-31G (420) | 11.3 GB, 61 s | 9.3 GB, 49-55 s | 8.1 GB, 48-75 s |
| C40H82 / 6-31G (524) | spills | 11.9 GB, 106-111 s | 12.9 GB, 116-129 s |
| C48H98 / 6-31G (628) | spills | spills | 13.4 GB, 224 s |
| C20H42 / def2-SVP (490) | 9.8 GB, 31 s | 8.2 GB, 30 s | 7.1 GB, 30 s |
| C24H50 / def2-SVP (586) | spills | 13.0 GB, 74 s | 11.2 GB, 56-65 s |
| C32H66 / def2-SVP (778) | spills | spills | spills |
| danuglipron / def2-SVP (724) | spills | spills | 12.1 GB, 174-177 s |

With `"f32"` the raw tensor may stay unpacked where `"f64"` packs it (the
smaller dressed band leaves room), which is why C40 / 6-31G peaks higher at
f32. Two runs peaked 0.1-0.5 GB above the 12.88 GB budget (C24 / def2-SVP
f64, C48 / 6-31G f32): the budget bounds the three-index tensors and their
scratch, not the whole process.

## Dispersion correction: D3(BJ)

**What it is.** Grimme's D3 with Becke–Johnson damping, a post-SCF pairwise
correction, implemented natively in Rust: the two-body term, plus the
Axilrod–Teller–Muto three-body term when its scale `s9` is nonzero. `s9`
defaults to 0 (off), the reference default for a named functional
(simple-dftd3 without `--atm`).

**Run it.** `[dft] dispersion = "d3bj"` on `method.kind = "ksdft"`
(`examples/water-pbe-d3bj.toml`); `"d3bj(b3lyp)"` borrows another functional's
fitted damping parameters. Python: `run_dft(..., dispersion="d3bj")`, or
`ferric.d3bj_energy(mol, functional, s9=0.0)` for the correction alone;
`s9=1.0` adds the three-body term. The CLI and `run_dft(dispersion=...)` use
`s9 = 0`.

**Scope.** The damping parameters are fitted per functional, so the key is
refused on any method other than `ksdft`. Tabulated functionals, each equal
to simple-dftd3 1.6.0's rational-damping entry: BLYP, BP86, B97-D, PBE, PBE0,
revPBE, B3LYP, TPSS, TPSS0, B2PLYP, BHLYP, SCAN, r2SCAN, ωB97X. M06-2X has no
D3(BJ) fit (only zero-damping D3(0)) and is an error. The `wb97x` row is
ωB97X-D3(BJ) (Najibi and Goerigk, JCTC 14, 5725 (2018)): D3(BJ) fitted to
replace the VV10 term of ωB97X-V, so it belongs to ωB97X-V's functional
without VV10, not to the 2008 ωB97X (libxc `HYB_GGA_XC_WB97X`). It is not
ωB97X-D3 (Lin, Li, Mao and Chai, JCTC 9, 263 (2013)), a separate functional
fitted with zero damping, nor ωB97X-D (Chai and Head-Gordon, 2008).
ferric cannot run ωB97X-D3(BJ) as published: `functional = "wB97X-V"` with
`dispersion = "d3bj"` finds no `wb97x` row (the lookup uses the functional's
own name), and `dispersion = "d3bj(wb97x)"` adds D3(BJ) on top of the VV10
term instead of replacing it. ferric has no ωB97X-V-without-VV10 functional,
so the `wb97x` row is tabulated for comparison with simple-dftd3 only.
`task = "optimize"` works (the D3
gradient is implemented; `ferric-d3/tests/gradient_vs_fd.rs`), and so does
`task = "frequencies"` on RKS, UKS and ROKS: see
[Frequencies with dispersion](#frequencies-with-dispersion).

**Accuracy.** Energy and analytic gradient agree with simple-dftd3 1.6.0
with the three-body term off and on, on nine molecules from water to benzene
and CHBr3: 1.2e-13 Ha and 5.5e-13 Ha/Bohr on the totals, 2.6e-15 Ha on the
three-body part (`ferric-d3/tests/vs_reference_dftd3_atm.rs`). The three-body
term is 0.1% of the two-body energy at benzene and grows with size. Unlike
simple-dftd3, no real-space cutoff is applied (it drops triples beyond
40 Bohr), and the triple sum costs O(N³).

## Dispersion correction: MBD@rsSCS

**What it is.** The many-body dispersion energy of Ambrosetti et al. 2014
(range-separated self-consistent screening), added post-SCF. Its per-atom
inputs are Tkatchenko–Scheffler free-atom α, C6 and R_vdW scaled by Hirshfeld
volume ratios v_A / v_A^free of the converged SCF density; the free-atom
volumes come from live free-atom SCFs in the same basis and SCF settings,
solved for the isolated atom: point charges, external fields, implicit
solvent, polarizable sites and cDFT constraints of the molecular run are not
applied to it. The
model itself is described under
[Polarizabilities and dispersion coefficients](./rpa-gw.md#polarizabilities-and-dispersion-coefficients).

**Run it.** `[dft] dispersion = "mbd"` on a Kohn–Sham SCF
(`examples/water-pbe-mbd.toml`) uses the β published for `[dft] functional`
(PBE 0.83, PBE0 0.85, HSE06 0.85); `"mbd(pbe0)"` uses another functional's β.
Any other functional is an error, not a default β. The printout gives
`E(KS-DFT)`, `E(MBD@rsSCS)` with β and the functional, and the corrected
total; the JSON run log's `run_end` record carries the corrected total as
`energy`, with `scf_energy` and a `dispersion` object (`model`, `params`,
`energy`, `beta`, `volume_ratios`) in `extra` (D3(BJ) runs log the same
object without `beta` and `volume_ratios`). UKS/ROKS energy runs, which write
no `run_end` record, write a `dispersion` record with the same fields. Python:
`run_dft(..., dispersion="mbd")`, which also reports `DftResult.volume_ratios`.

**Gradient.** `task = "optimize"` on an RKS, UKS or ROKS reference and
`run_dft(with_gradient=True)` add the exact analytic MBD@rsSCS gradient: the
explicit dependence on the nuclear positions, and the dependence through the
Hirshfeld volumes, including how the SCF density itself responds to the
displacement. That response is the orbital relaxation, obtained from one
coupled-perturbed Kohn–Sham (Z-vector) solve per gradient; it costs about as
much as a few SCF iterations. On a UKS reference the solve is coupled across
the α and β orbital rotations (Coulomb couples the spins; exchange and the XC
kernel are spin-resolved), and the orthonormality of each spin's occupied
orbitals enters separately. On a ROKS reference the solve runs over the three
rotation blocks of the shared orbitals (closed→virtual, open→virtual,
closed→open) with the exact restricted-open-shell orbital Hessian, and the
closed and open orbitals are kept orthonormal as one set, which adds a
closed–open cross term to the orthonormality contribution. The volumes are
integrated on a lattice anchored to the molecular centroid, and its motion is
part of the gradient. Against finite differences of the full SCF + MBD
calculation at 6-31G the gradient agrees to 6e-9 Hartree/Bohr for H2O with
PBE, PBE0, HSE06 and PBE with RI-J, and to 9e-10 for NH3; the orbital
relaxation alone is 1.0e-5 for H2O (11.5% of the largest MBD component). For
UKS doublets and a triplet (NH2, OH, O2 at 6-31G with PBE, PBE0 and HSE06) it
agrees to ≤ 1.9e-9, and to 6e-9 for OH with PBE + RI-J and PBE0 + RI-JK,
against an orbital relaxation of 3e-6–1e-5. For ROKS doublets and triplets
(HCO, NH2, CH2, O2 at 6-31G) it agrees to ≤ 2.1e-11 with PBE and PBE0, ≤ 4.5e-10
with HSE06 and 7.5e-9 for HCO with PBE + RI-J, against an orbital relaxation
of 2.7e-6–9.8e-6. For a linear radical whose state breaks the cylindrical
symmetry (²Π: OH, NO, CH), a rotation of the electrons about the molecular
axis is an orbital rotation that leaves the energy unchanged, so the orbital
Hessian has a null mode; only the anisotropy of the XC grid gives it a small
eigenvalue of either sign. The UKS and ROKS Z-vectors detect it on linear
molecules and solve on its complement, which is exact for a property that is
symmetric about the axis. On OH, NO and CH (STO-3G and 6-31G, PBE and PBE0)
the gradient agrees with finite differences of the full pipeline to
7e-12–5e-11 Hartree/Bohr. A state with any other negative or near-null
orbital-Hessian direction is still refused (the UKS-PBE CH state is a saddle
point and is refused). The relaxation term needs an LDA, GGA or hybrid-GGA
functional without VV10, no implicit solvation, polarizable embedding or cDFT
constraints, and integer aufbau occupation (no MOM). With COSX exchange the
overlap fit must be off (`cosx_overlap_fit = false`); the UKS-PBE0 + COSX
gradient on NH2/STO-3G agrees with finite differences of the full pipeline to
2.3e-11. Other setups are refused rather than given an
approximate gradient. Python's `run_dft` is closed-shell only; open-shell
frequencies with MBD@rsSCS run through the CLI or `run_frequencies`.

## Frequencies with dispersion

**What it is.** Harmonic frequencies on the dispersion-corrected surface
E(KS) + E(disp), on an RKS, UKS or ROKS reference.

D3(BJ) depends on the geometry only. Its Hessian is analytic
(`ferric_d3::d3bj_hessian`, derivation in `book/src/25-dispersion/01-d3bj.md#d3bj-hessian`). The run takes the Hessian route it would take without
dispersion and adds the D3(BJ) Hessian: under `hessian = "auto"`, the analytic
moving-grid KS Hessian (RKS or UKS; LDA, GGA, pure or global hybrid; ECPs
allowed) plus the D3(BJ) Hessian, with no gradient evaluations. Where no
analytic KS Hessian applies (ROKS, meta-GGA, VV10, range-separated hybrids),
`auto` differences the KS + D3(BJ) gradient. An explicit
`hessian = "analytic-moving-grid"` on one of those references is refused
before the SCF. The ATM three-body term (`s9 != 0`, Rust API only) has no
Hessian and is refused.

MBD@rsSCS follows the density. Its Hessian is the central difference of the
corrected analytic gradient: at each of the 6N displaced geometries the SCF
of that reference is converged, the KS gradient and the MBD gradient are
evaluated there, and their sum is differenced. The MBD gradient is the exact
one above, with the Z-vector of the reference solved (RKS, UKS or ROKS), so
the response of the Hirshfeld volumes to the displacement, through the SCF
density, is in the Hessian. With MBD, `[frequencies] hessian = "analytic"` and
`"analytic-moving-grid"` are errors and `"auto"` runs finite differences.

**Run it.** `[dft] dispersion` with `method.task = "frequencies"` on a KS
SCF: `ksdft`, or `rhf`/`uhf`/`rohf` with `[dft] functional`. The printout gives the corrected `energy`, `E(KS-DFT)`
and the dispersion energy at the input geometry, and the JSON run log gets a
`dispersion` record with the same fields as an energy run. Python:
`run_frequencies(mol, basis, xc="PBE", dispersion="d3bj")`, which reports
`.e_dispersion` and the corrected `.energy`; `reference="uhf"` or `"rohf"`
selects UKS or ROKS.

**Accuracy.** For PBE/STO-3G water, the D3(BJ) part of the Hessian agrees
with 4-point second differences of the D3(BJ) energy to 5.3e-9 Hartree/Bohr²
(largest element 1.9e-5), and the resulting frequencies with those of the KS
Hessian plus that independent D3 Hessian to 1e-5 cm⁻¹. The MBD@rsSCS part
agrees along three fixed directions with second differences of the full
SCF + MBD energy to 1.5e-6 Hartree/Bohr² (0.8% of the largest curvature, 2.0e-4).
That figure is set by noise in the energy differences, not by the Hessian
construction. A gradient without the orbital-relaxation term misses by 3.2e-5.
Dispersion leaves the Hessian's asymmetry (7.6e-6 Hartree/Bohr² for
PBE/6-31G water) and the projected translation/rotation modes (below 1e-4
cm⁻¹) where they are without it.

On open shells (PBE/STO-3G, bent NH2 doublet on UKS, HCO doublet on ROKS), the
D3(BJ) part agrees with 4-point second differences of the D3(BJ) energy to
3.7e-9 and 6.5e-9 Hartree/Bohr² and the frequencies to 9.4e-6 and 3.0e-5 cm⁻¹;
the D3 Hessian obeys the translational sum rule to 1e-11 and the rotational one
(with the D3 gradient) to 3e-9. The MBD@rsSCS part agrees along three fixed
directions with 5-point second differences of the full SCF + MBD energy to
≤ 4.8e-8 Hartree/Bohr² for UKS NH2 and CH2 (triplet) and ROKS HCO and CH2,
against curvatures of 6e-5–2e-4; without the orbital-relaxation term the
misses are 8.3e-7–1.6e-5. A correction that does nothing reproduces the plain
UKS or ROKS finite-difference Hessian bit for bit.

The shifts are small for a single molecule. For PBE/6-31G water at its
uncorrected PBE minimum (1594.35, 3500.27 and 3669.29 cm⁻¹), D3(BJ) shifts
the three modes by +0.006, −0.077 and −0.095 cm⁻¹, and MBD@rsSCS by +0.107,
−0.406 and −0.516 cm⁻¹.

## Scalar relativity: X2C-1e

`[scf] relativistic = "x2c1e"` (Python `relativistic="x2c1e"` on `run_rhf`,
`run_uhf`, `run_rohf`, `run_dft`/`run_ksdft`, `run_uks`, `run_roks`) replaces
`T + V` by the spin-free X2C-1e core Hamiltonian before the SCF: the
one-electron Dirac equation of the nuclear potential is decoupled in the
uncontracted basis (one shell per unique exponent and angular momentum) and
contracted back, with c = 137.03599967994 a.u. These are the settings of
PySCF `scf.RHF(mol).sfx2c1e()`. The two-electron interaction and property
operators stay nonrelativistic (no picture change), and the guesses (SAD,
MINAO, `basis2`) are nonrelativistic starting points only. The certificate
lists `x2c1e` among its approximations and grades the run Ungraded; post-SCF
certificates inherit it.

| Check | System | Result |
|---|---|---|
| Driver vs block (constant Fock hook `h_X2C − h_NR`, `E = Tr[D dH]`) | HBr / cc-pVDZ RHF; HBr⁺ UHF, ROHF, UKS/PBE | 4.6e-13 to 9.1e-13 Ha (bar 2e-10) |
| RHF vs PySCF `sfx2c1e()` | HBr, H2Se at cc-pVDZ and uncontracted cc-pVDZ | 6.7e-11 to 1.6e-10 Ha (bars 2.2e-10 to 2.3e-10); X2C shifts −21.9 to −32.0 Ha |
| RI-MP2 on the X2C reference vs PySCF DF-MP2 | HBr / cc-pVDZ, `cc-pvdz-ri` | E_corr within 1.3e-11 Ha; X2C moves E_corr by 1.3e-3 Ha |
| `None` | RHF, UKS, ROHF | energy bit-identical to the build without the option |

What runs on an X2C reference: the SCF energies above, and post-SCF energies
that read only the orbitals and orbital energies (CLI kinds `rimp2`,
`scs-mp2`, `ccsd`, `ccsd(t)`). What is refused, with a message naming
`relativistic = "x2c1e"`: analytic and finite-difference gradients, Hessians
and frequencies, CLI `task = "optimize"`/`"frequencies"`, Python
`with_gradient=True`, `run_optimize`/`run_frequencies(scf_relativistic=)`,
saddle and IRC searches, Z-vector relaxed gradients, orbital-optimized MP2
(it rebuilds the nonrelativistic core Hamiltonian), periodic cells, and CLI
kinds outside the list above (`casscf` and `nevpt2` rebuild `h`). An atom
carrying an ECP is an error: the ECP already holds the core's scalar
relativity.

## Implicit solvation

Two independent implementations. Both are threaded uniformly through RHF, UHF,
ROHF and the KS variants, and both leave the energy bit-identical to vacuum
when unset.

| Model | What it is | How to run it | Anchor |
|---|---|---|---|
| **IEF-PCM** (`ferric-pcm`) | Integral-equation PCM; modified Bondi radii (H 1.10 Å), atom-centred Lebedev spheres | CLI `[pcm] solvent = "water"` (or `epsilon`), energy runs of `rhf`, `uhf`, `rohf`, `ksdft` and `pdep-rpa` (`examples/water-pcm.toml`), and `task = "optimize"` for `rhf`, `uhf`, `rohf` and `ksdft`. Python `run_rhf(..., solvent=78.4)` or a solvent name; also `run_pdep_rpa`. | water / STO-3G, ε = 78.4: −3.813 vs PySCF IEF-PCM −3.8228 kcal/mol |
| **COSMO** (`ferric_scf::cosmo`) | Conductor-like screening | CLI `[cosmo] epsilon = 78.39` | water / cc-pVDZ, ε = 78.39: −5.955 vs PySCF COSMO −5.94 kcal/mol (`ferric-scf/tests/cosmo_water.rs`) |

Neither has cavitation, dispersion or repulsion terms. IEF-PCM has an
analytic nuclear gradient (`ferric_pcm::pcm_gradient`: the tessera-charge
potential term plus the S/D/area response term, switching-function
derivatives included; equations in `book/src/22-implicit-solvation/04-gradient.md#gradient-derivative`), added
to the RHF, UHF, ROHF and KS gradients whenever `RhfConfig.pcm` is set. It
matches PySCF IEF-PCM `nuc_grad_method()` to 7e-11 Ha/Bohr on PySCF's
302-point cavity and finite differences of the solvated SCF energy to
3e-7 Ha/Bohr. It needs the cavity built from the molecule (an injected
`PcmConfig::cavity` is refused). There is no PCM Hessian: frequencies with
`[pcm]` are refused. COSMO has no gradient term and no Hessian: every
analytic-gradient entry (the SCF gradient dispatchers, `optimize_geometry*`,
`optimize_qmmm`, finite-difference frequencies, and CLI `task = "optimize"` or
`"frequencies"` with `[cosmo]`) refuses a COSMO SCF with "COSMO has no analytic
gradient; use [pcm] ... or finite differences". COSMO is energy-only; use
`[pcm]` for geometry work. An unknown solvent name is an error, not a silent
vacuum.

## Gradients, optimization, frequencies, transition states

**Gradients.** Analytic nuclear gradients for RHF, UHF, ROHF and KS-DFT
(including meta-GGA, with grid response), checked against finite differences
and, for DFT, against PySCF. RI-MP2 and SCS-MP2 also have analytic
gradients, closed shell only (there is no unrestricted MP2 nuclear gradient).

**Optimization.** `method.task = "optimize"` for `rhf`, `uhf`, `rohf`,
`ksdft`, `rimp2`, `scs-mp2`, `oo-rimp2` and `pdep-rpa`
(`examples/h2_opt.toml`, `examples/h2-lda-opt.toml`); on an open-shell
molecule only `uhf`, `rohf` and `ksdft`. Python `ferric.run_optimize` covers
the same SCF and MP2 routes (`reference=`, `xc=`, `dispersion=`, `solvent=`,
`method="rimp2"`, `"scs-mp2"` (with `c_os=` / `c_ss=`) or `"oo-rimp2"`)
through the same library drivers. The SCS-MP2 gradient is the relaxed RI-MP2
gradient with the pair weight `(c_os + c_ss) t − c_ss t̄` in place of
`2t − t̄` (derivation in `book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-scs-mp2`), checked against
finite differences of the SCS-MP2 energy.

**Harmonic frequencies.** `method.task = "frequencies"` for `rhf`, `uhf`,
`rohf` and `ksdft` (`examples/water-frequencies.toml`,
`examples/nh2-rohf-frequencies.toml`); Python
`ferric.run_frequencies(mol, basis_name, reference="rhf", xc=...)`. Mass-weighted,
translations and rotations projected out. The Hessian is:

- **Analytic** for closed-shell RHF and for UHF of any multiplicity, with
  exact four-centre J/K or with density-fitted J and/or K (`df_j_aux` /
  `df_k_aux`, Coulomb operator, no external potential on the fitted
  routes, a well-conditioned fitting metric), no solvent or polarizable
  embedding, and a basis up to f functions (RHF and UHF also with an ECP, on
  the fitted routes too, and with exact J/K also with fixed external point or
  smeared charges, or a uniform field),
  and for ROHF with exact four-centre J/K or RI-J / RI-K, an ECP allowed, no
  external potential and a basis up to f functions:
  one SCF plus a coupled-perturbed HF solve per nuclear coordinate (for UHF,
  one solve coupling the α and β orbital rotations; for ROHF, one solve over
  the closed-virtual, open-virtual and closed-open rotations). This
  needs a libint2 with second derivatives (the build
  `scripts/install-libint.sh` installs).
- **Analytic on the moving grid** for closed-shell Kohn-Sham (`ksdft`, or
  `rhf` with a functional) and UKS: LDA and GGA functionals, pure or global
  hybrid, exact or density-fitted J and K, Becke weight response included, so
  it is the Hessian of the surface the KS gradient differentiates. One SCF plus
  a coupled-perturbed KS solve (all nuclear coordinates solved together, one
  J/K integral pass per iteration).
- **Central finite differences of the analytic gradient** everywhere else
  (ROKS, meta-GGA, VV10, range-separated hybrids, KS with an ECP or an
  external potential, pruned grids, `[dft] dispersion`, RI J/K with an
  external potential or an ill-conditioned fitting metric, solvent, polarizable
  embedding, g functions, ROHF with an external potential): 6N gradient
  evaluations. The displacement `[frequencies] delta` (default 5e-3 Bohr) is a
  real accuracy knob; the printed Hessian asymmetry, zero in exact arithmetic,
  is the check that it and the SCF thresholds suit the system.

`[frequencies] hessian` (Python `hessian=`) selects it: `"auto"` (default,
analytic where it applies, the moving-grid Hessian for closed-shell KS
and UKS),
`"analytic"` (the Hartree-Fock analytic Hessian; an error where it does not
apply, including every KS reference), `"analytic-moving-grid"` (the
KS one, RKS or UKS; an error where it does not apply) or `"fd"`. The output
names the one that ran (`Hessian = analytic`, `analytic-moving-grid` or
`finite-difference`; Python `.hessian_source`).

The moving-grid KS Hessian agrees with `"fd"` up to the finite-difference
error: ≤ 5e-7 cm⁻¹ against Richardson-extrapolated FD at a tight SCF
(`density_conv` 1e-11) on water / STO-3G, for one SCF plus a
coupled-perturbed KS solve instead of 6N gradients. Against the default
`"fd"` (step 5e-3 Bohr, default SCF), which `"auto"` ran for KS before, the
wavenumbers move by ≤ 0.071 cm⁻¹ (water / STO-3G, LDA, PBE, B3LYP). Under
`"auto"`, ROKS, meta-GGA, VV10, range-separated hybrids, ECPs,
external potentials, pruned grids, `[dft] dispersion`, COSX, implicit solvent,
fractional occupations, MOM, and systems whose response AO table (basis
functions × grid points × 4 for a GGA, 8 bytes each) exceeds 2 GiB (on the
default 75×110 grid, a GGA above about 8,100 basis functions × atoms, e.g.
about 30 atoms at def2-SVP) run finite differences; `"analytic-moving-grid"`
refuses them before the SCF.

For UKS, `"auto"` takes the analytic Hessian at every size: measured
(def2-SVP, default grid, exact J/K, 6 cores, wall seconds including the SCF,
median of 3 where shown) it is 2.4 to 4.4 times faster than `"fd"`:

| radical | atoms | basis functions | PBE analytic / fd | B3LYP analytic / fd |
|---|---|---|---|---|
| NH2 | 3 | 24 | 0.96 / 3.38 | 1.09 / 4.42 |
| CH3 | 4 | 29 | 1.56 / 6.16 | 1.77 / 7.69 |
| C2H5 | 7 | 53 | 10.1 / 27.0 | 12.6 / 35.2 |
| C6H5 | 11 | 109 | 103 / 330 (1 run) | 156 / 407 (1 run) |
| C4H9 | 13 | 101 | 96.9 / 266 (1 run) | 137 / 331 (1 run) |
| C9H11 | 20 | 181 | 763 / 2628 (1 run each) | — |

There is no crossover to place: the finite-difference route never won.

The coupled-perturbed KS solve applies the XC kernel to the occupied-virtual
rotations directly and contracts every density of a block in one pass over the
RI-J tensor; for closed-shell RHF and RKS the fitted K of the CPHF/CPKS blocks
is formed from a factored occupied half-transform, so one tensor pass serves a
whole block. Measured on butane with PBE, interleaved A B B A A B, three
repetitions each: the Hessian took 43.1 s on def2-SVP and 106.0 s on def2-TZVP
before the XC kernel change and 19.4 s and 40.3 s after it; the frequencies
agree to the printed 0.01 cm⁻¹ in all six cases.

The choice is made before the SCF. An error the analytic path raises after the
SCF (for ROHF: the stored spin Fock matrices differing from those rebuilt from
the final densities by more than 1e-4 Ha, which a loose `density_conv` can
cause; a response solve that does not converge; non-positive orbital curvature;
a first-order rotation above 10 rad/Bohr) is returned as an error under
`"auto"` too, never retried with finite differences; the same holds for the
moving-grid KS Hessian (a non-converged SCF included), with one exception: a
UKS coupled-perturbed solve that fails (non-positive orbital curvature or no
convergence) makes `"auto"` run `"fd"` instead, with a warning. The measured
case is OH (²Π): its β π hole rotates about the bond at no energy cost, an
exact zero mode of the orbital Hessian, and the solve meets a curvature of
−1.9e-10; `"analytic-moving-grid"` still reports that error.

### Thermochemistry and IR intensities

Every frequency run also reports ideal-gas thermochemistry and, for the SCF
kinds, IR intensities.

**Thermochemistry.** The rigid-rotor / harmonic-oscillator model, with the
conventions of PySCF `hessian.thermo`: translational, rotational,
vibrational and electronic E, H, S and G, and the heat capacities, at
`[frequencies] temperature` (K, default 298.15) and `pressure` (Pa, default
101325, one atmosphere). Linear molecules use one rotational constant. The
electronic entropy is `R ln(2S+1)`. Imaginary modes are left out of the
vibrational terms with a warning. The masses are those of the frequency
calculation (isotope-averaged standard atomic weights); the constants are
CODATA 2018.

The rotational symmetry number `[frequencies] symmetry_number` defaults to 1.
ferric does not detect it, so set it: 2 for water or CO2, 3 for NH3, 12 for
CH4. With the default, a molecule whose true number is σ gets an entropy too
large by `R ln σ` (1.38 cal/(mol K) for σ = 2) and a G too low by `RT ln σ`.

Python: `fr = ferric.run_frequencies(...)`, then
`fr.thermo(temperature=298.15, pressure=101325.0, symmetry_number=2)`
returns a dict of the parts (Hartree and Hartree/K per molecule; the
`.pyi` lists the keys). The CLI prints a table and writes the same numbers
to the `thermochemistry` field of the run log's `frequencies` record.

Checked against PySCF's `thermo` on the same Hessian, geometry and masses
(RHF/STO-3G water at 298.15 K and at 500 K, 2e5 Pa; linear CO2; UHF OH
doublet): every E, H, TS and G term agrees to 8e-13 Hartree once PySCF's
constants are set to CODATA 2018. With PySCF's own CODATA 2014 constants the
difference is 8e-9 to 1.5e-8 Hartree, from k_B.

**IR intensities.** The harmonic intensity of each mode in km/mol, from the
dipole derivative along the normal mode. The dipole derivative comes from
the analytic nuclear gradient of SCFs solved in a uniform field of ±1e-3
a.u. (`dμ/dR = −∂²E/∂F∂R`): six extra SCF + gradient evaluations at any
molecule size, with the SCF density threshold capped at 1e-9. The CLI adds
an `IR (km/mol)` column and writes `ir_intensities_km_mol` and the dipole
derivatives to the run log (`[frequencies] ir_intensities = false` skips
them). Python: `fr.ir_intensities()`, computed on the first call.

This covers `rhf`, `uhf`, `rohf` and `ksdft` (RKS, UKS, ROKS). The correlated
kinds (`rimp2`, `scs-mp2`, `oo-rimp2`, `pdep-rpa`) have no dipole derivative
in ferric: the CLI leaves the column out, and `ir_intensities = true` is an
error. With `[dft] dispersion` the modes come from the corrected Hessian and
the dipole derivative from the SCF alone.

Checked against a second route, central differences of the SCF dipole along
every Cartesian displacement, with a bar built from each route's
Richardson truncation estimate and measured SCF-noise bounds. On water/6-31G
(RHF, PBE) and OH/6-31G (UHF, ROHF, UKS-PBE) the two routes give dipole
derivatives within 7.5e-6 to 1.5e-5 e of each other, against bars of 2.2e-5
to 3.3e-5 e; the intensities agree to 1e-3 km/mol on modes of 1.7 to 98
km/mol. The field route meets the translational sum rule to 3e-9 e.

**Transition states and IRC (Python only, closed shell).**
`ferric.run_saddle(mol, basis_name, xc=...)` searches for a first-order saddle
by P-RFO, using two finite-difference Hessians with Bofill updates between
them; it costs `2(6N + 1) + (n_steps + 1)` gradient evaluations and refuses to
start from a geometry with no negative mode. `hessian="analytic"` (or `"auto"`,
which falls back to finite differences where the analytic Hessian does not
apply) swaps in the analytic RHF Hessian (with `xc`, `"auto"` takes the
moving-grid KS Hessian), which needs no displaced gradients;
`"analytic"` raises for KS `xc`, point charges / an external field, ECPs or a
basis above the libint second-derivative limit, `delta=` is only valid with
`"fd"`, and `result.hessian_source` says which ran. The default is `"fd"`. `result.is_transition_state()`
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
| `k_builder = "cosx"` | seminumerical (COSX) K on a grid; with RI-J active this is RIJCOSX | grid-dependent error, see below | RHF/UHF/ROHF and their Kohn–Sham variants, Coulomb operator only; analytic gradient for all six (see below) |
| `jk = "rijk"` | RI-J + RI-K for the reference SCF of an MP2-family kind, on `jk_aux` | fitting error, below the RI-MP2 fit error on the matched set (see *MP2*) | MP2-family kinds; the default with `jk` omitted for a closed-shell energy and for `rimp2` / `scs-mp2` / `oo-rimp2` optimizations, whose gradient checks its peak memory against `[memory] budget_gb` first |

**Which route runs.** RHF, UHF and ROHF and their Kohn–Sham variants take
J and K from one decision, made from `df_j_aux`, `df_k_aux`, `k_builder`, the
functional's exact-exchange mix and the spin case. The run's `SCF J/K` line,
the JSON log and the certificate state the route that ran, and the certificate
lists LinK or COSX only when they actually run. For J, `name` is a fitting
basis such as `def2-universal-jkfit`:

| Functional | Spin | `df_j_aux` omitted | `df_j_aux = ""` | `df_j_aux = name` |
|---|---|---|---|---|
| none (Hartree–Fock) | any | four-centre J | four-centre J | RI-J on `name` |
| any functional | closed shell | RI-J on `def2-universal-jkfit` | four-centre J | RI-J on `name` |
| any functional | open shell | four-centre J (the CLI and the Python `run_uks` / `run_roks` / `run_dft` entry points name `def2-universal-jkfit`) | four-centre J | RI-J on `name` |

J does not depend on the range-separation parameter ω, so a range-separated
hybrid fits J exactly as a global hybrid does, for UKS and ROKS as for RKS.
For K:

| Functional | Spin | `df_k_aux` omitted | `df_k_aux = ""` | `df_k_aux = name` |
|---|---|---|---|---|
| none (Hartree–Fock) | any | exact K | exact K | RI-K on `name` |
| global hybrid | closed shell | RI-K on `def2-universal-jkfit` | exact K | RI-K on `name` |
| global hybrid | open shell | exact K (as for J, the entry points above name the fit) | exact K | RI-K on `name` |
| pure (no exact exchange) | any | no K | no K | no K |
| range-separated (ω > 0) | any | short-range / long-range pair on `def2-universal-jkfit` | error | pair on `name` |

"Exact K" is the four-centre builder, or the one `k_builder` names. In Python,
`run_optimize` and `run_frequencies` with `xc=` and no `df_*_aux` fit J and K on
`def2-universal-jkfit` for RKS, UKS and ROKS alike, so a geometry run matches
`run_uks`, `run_roks`, `run_dft` and the CLI for the same input. Under a
range-separated hybrid in the CLI, UKS and ROKS therefore run RI-JK (NH2 /
6-31G CAM-B3LYP moved from −55.8219504993 to −55.8219773531 Ha for UKS when
J became fitted). `omega` tuning accepts a named `df_j_aux`.

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

**Threads.** COSX splits the grid into 1024-point blocks and each block
into four 256-point kernel tasks. It runs the tasks of several blocks in one
parallel region (about eight tasks per thread) and adds the blocks into K
in grid order. K is bitwise the same for every thread count. Sulfamethoxazole /
def2-TZVP (599 basis functions), B3LYP energy, 6 threads, default grids, 14
iterations: RIJCOSX takes 365 s wall (median of 3; 5.5 cores busy, 2.5 GB
peak RSS), and RI-JK takes 31 s (7.7 GB). The two energies differ by 1.9e-4 Ha.
The COSX kernel is 93% of the exchange build.

**Energy error against exact exchange:**

| Builder | System / basis | Error |
|---|---|---:|
| LinK | butane / def2-SVP | 9e-12 Ha |
| COSX, default (sgx (35,194) + final pass on sgx (50,302)) | water / aug-cc-pVDZ | −8.1e-9 Ha |
| COSX, default | butane / def2-SVP | −3.4e-5 Ha |
| COSX, default | butane / def2-TZVP | −4.3e-6 Ha |
| COSX, flat (50,110) | water / cc-pVDZ | 5e-6 Ha |
| COSX, flat (50,110) | butane / def2-SVP | 1.7e-4 Ha |
| COSX, flat (50,110) | butane / def2-TZVP | 1.2e-4 Ha |
| RI-JK | — | not measured on these systems |

**Start with density fitting** whenever its three-index tensor fits in memory
(`n_aux × n_bf² × 8` bytes: 1 GB for butane/QZVP, 7 GB for octane/QZVP; half
that when stored as the packed symmetric triangle). When it does not fit, J is
built by Split-RI-J, which stores no tensor and on danuglipron PBE/def2-SVP
took 81 s for the whole SCF against 46 s with the tensor in memory and about
590 s spilling or recomputing it. A tensor K still needs is spilled to disk or
recomputed each iteration, whichever a measurement on the machine says is
cheaper (`[scf] jk_storage`). It is two to three orders of magnitude faster
per build than anything else here. Its error is a fitting error, not zero, and
it grows with system size.

**Direct J/K against PySCF.** The direct four-centre builder digests integrals
in shell-block tiles and streams its reduction. Measured with the same density,
a Schwarz threshold of 1e-12 and 12 threads (paired, mean of 3, ferric's
segmented shells in both codes), one J+K build took 5.20 s for octane /
cc-pVDZ against 9.47 s for PySCF `get_jk`, and 47.8 s for octane / cc-pVTZ
against 49.7 s. PySCF integrates cc-pVXZ sets as general contractions, which
libint2 cannot, so its gap to ferric's segmented-shell timing at those sets
is a property of the integral library, not of the digestion.

**Need exact exchange?** Direct or LinK; both are exact to the screening
threshold as *K builders* (butane/def2-SVP: `link` == direct to 9e-12 Ha).
LinK's cost is being re-measured, so no LinK timing is quoted here.
`k_builder` is honoured by UHF and ROHF as well as RHF: the open-shell solvers
build K_α and K_β from one builder instance, refreshing its density-dependent
state per spin. Whether LinK is the faster choice for a given system is a
separate question from whether it is honoured; see the cost note above, which
is being re-measured. `link` is skipped with a warning, never silently,
whenever density-fitted J/K is active, and so is an explicit `direct` whenever
exchange is not built by the direct four-centre builder (RI-K active, a pure
functional, or a range-separated one); `direct` with RI-J alone is honoured.
Both `link` and `cosx` are skipped with
a warning when the functional uses no exact exchange or is range-separated
(exchange then comes from the SR/LR fitters). For a Kohn–Sham hybrid, whose
RI-J and RI-K are on by default, `direct` needs `df_k_aux = ""`; `link` needs
`df_j_aux = ""` as well: the resolver keeps `link` only when J is not density
fitted (COSX is the one K builder it pairs with RI-J), and with RI-J off J
comes from the direct four-centre J builder.

**RIJCOSX.** `k_builder = "cosx"` combines with density-fitted Coulomb: when
RI-J is active (`df_j_aux` named, or the Kohn–Sham default), J comes from RI-J
and K from COSX, as in ORCA's RIJCOSX and Psi4's DFDIRJ+COSX. COSX then
replaces RI-K: the Kohn–Sham RI-K default is not applied, and an explicitly
named `df_k_aux` next to `k_builder = "cosx"` is an error. `df_j_aux = ""`
keeps exact J with COSX K. The two approximations add: on water/6-31G (HF) the
COSX error is 7.140e-6 Ha with exact J and 7.141e-6 Ha with RI-J, and the
cross term is second order (2.8–5.8 × the product of the two errors; 7.8e-10
Ha at the (50,110) grid, 2.3e-11 Ha at (75,302)). Gradients take J from the
RI-J derivative and K from the COSX derivative (FD agreement 1.7e-9 Ha/Bohr
RHF, 3.6e-9 UHF).

**COSX is for large basis sets on systems too big for RI-JK.** Its cost per
grid point barely moves with angular momentum while analytic exchange grows
roughly tenfold from SVP to QZVP, so it wins at high angular momentum, not at
large system size. Measured on one thread at the flat (50,110) grid:

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

On the flat (50,110) grid COSX's energy error is about 5e-6 Ha on water/cc-pVDZ
(4.9e-6 in `cosx_k_anchors.rs`, 5.1e-6 in the ORCA comparison run),
1.7e-4 Ha on butane/def2-SVP and 1.2e-4 Ha on butane/def2-TZVP (against exact
exchange). ORCA 6.1.1's COSX at its own default grid gives 4.9e-6, 3.8e-5 and
1.3e-5 Ha on the same systems and bases with about half as many points: ORCA's
grids are pruned around a 194-point valence shell and it evaluates its final
energy once on a finer grid. ferric's default does the same (`cosx_grid`
pruning and `cosx_final_pass`, below). Refining the flat grid converges water to 3.4e-8 Ha,
but butane/def2-SVP stays at 3.4e-5 Ha at Lebedev-302 for every radial grid.
That residual is angular: an independent COSX (PySCF SGX, 75 radial shells) on
the same system goes from 5.5e-5 Ha at 302 points per shell to -8.5e-6 at 434
and 2.7e-6 at 590. ferric's COSX follows the same path at 75 radial shells:
-3.4e-5 at 302, -5.6e-6 at 434 and +1.8e-6 at 590, for 1.8x and 2.2x the
302-point wall time. Reaction energies cancel
most of the error (0.02 kcal/mol on an isodesmic alkane reaction at the flat
(50,110) grid); absolute energies do not. The knobs, all optional:

- `cosx_grid = { radial = 35, angular = 194, prune = "sgx" }` is the default;
  `{ radial = 50, angular = 110 }` is the flat grid. The angular order
  matters most: on butane/def2-TZVP the error falls from
  1.2e-4 Ha at 110 points per shell to 4.0e-6 Ha at 302, while going from 50
  to 100 radial shells changes it by 1e-6 or less. Cost grows with the number
  of points. `angular` takes any Lebedev order `grid_angular` does except the
  negative-weight rules 74, 230 and 266. `prune = "sgx"` prunes the grid the way ORCA's GridX and PySCF's SGX
  do: five radial regions per atom (NWChem boundaries on Bragg radii) with
  Lebedev orders from one row, picked by the peak `angular` — at 194 the
  regions get 26/50/110/194/110 points. ferric's pruned COSX energy equals
  PySCF SGX on the same grid to 1.8e-12 Ha. A table without `prune` is flat.
- `cosx_final_pass = true` (the default; `false` turns it off) re-evaluates exchange once on a
  larger grid at the converged density and reports that energy; the SCF-grid
  energy is printed and logged next to it. `cosx_final_grid` picks the grid
  (default `{ radial = 50, angular = 302, prune = "sgx" }`). The pass is not
  self-consistent, but it lands within 1.5e-7 Ha of an SCF converged on the
  final grid (table below). It is energy-only: gradients and geometry tasks
  run without it.
- `cosx_grid_schedule = false` (default). `true` runs the early SCF
  iterations on a coarse pruned `{ radial = 25, angular = 110, prune = "sgx" }`
  grid and switches to `cosx_grid` once the largest density-matrix change
  between iterations drops below 1e-3, restarting DIIS at the switch.
  Convergence is only accepted after a full iteration on `cosx_grid`, so the
  converged energy is the `cosx_grid` energy; the final pass is unchanged.
  On water/cc-pVDZ the coarse grid has 4062 points against 10866, and the
  scheduled SCF reaches the unscheduled energy to 1e-13 Ha in 15 iterations
  (8 coarse, 7 on `cosx_grid`) against 12. Its wall time has not been
  measured. RHF/UHF and their Kohn–Sham variants; ROHF refuses it.
- `cosx_overlap_fit = true` (default) applies the Izsák–Neese overlap
  correction. At the default grid it helps; on coarser grids it makes things
  worse, and its benefit depends strongly on the molecule (large on water,
  nil to negative on ethane), so the factor quoted in the literature does
  not carry over.
- `cosx_backend = "md3c1e"` (default) is the batched McMurchie–Davidson
  integral kernel. `"cosx-a"` is the per-point libint2 path, about three times
  slower and kept only as the cross-check the kernel is anchored against.
- `cosx_screen_thresh = 1e-7` (default) is the density-driven pair-screening
  threshold. `0.0` disables screening bit-identically; `1e-6` already fails a
  1e-6 Ha K-error bar on butane. Not available with the `"cosx-a"` backend.

Setting any `cosx_*` key without `k_builder = "cosx"`, or `k_builder = "cosx"`
together with a named `df_k_aux`, is an error.

**Which COSX grid.** Measured against exact exchange (exact J on both sides,
RHF, overlap fit on unless noted), one run each, six threads; times are whole
SCFs, indicative only. Errors, E − E_exact in Ha:

| System | atoms | flat (50,110) | sgx (35,194) | sgx (35,194) + final sgx (50,302), default | sgx (50,302) | sgx (50,194) | flat (50,194) | sgx (35,194), no fit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| water / aug-cc-pVDZ | 3 | +6.1e-6 | +1.9e-6 | −8.1e-9 | −8.1e-9 | +9.8e-7 | +1.6e-7 | +1.6e-6 |
| butane / def2-SVP | 14 | +1.7e-4 | +4.6e-5 | −3.4e-5 | −3.4e-5 | +4.1e-5 | +5.0e-5 | −2.7e-4 |
| butane / def2-TZVP | 14 | −1.2e-4 | −4.1e-5 | −4.3e-6 | −4.3e-6 | −3.2e-5 | −1.6e-5 | −1.2e-4 |
| benzene / def2-SVP | 12 | +8.2e-5 | −4.8e-5 | +5.6e-6 | +5.6e-6 | −4.8e-5 | −4.0e-5 | −8.8e-5 |
| sulfamethoxazole / def2-SVP | 28 | +2.5e-4 | −1.9e-5 | +2.2e-5 | +2.1e-5 | −2.3e-5 | −2.3e-5 | −1.9e-4 |
| methane / cc-pVDZ | 5 | +8.0e-6 | +9.2e-6 | +6.5e-7 | +6.5e-7 | +1.0e-5 | +5.1e-6 | −1.9e-4 |
| ethane / cc-pVDZ | 8 | +1.2e-4 | −4.3e-6 | −2.6e-6 | −2.6e-6 | −4.5e-6 | +1.7e-6 | −9.1e-5 |
| propane / cc-pVDZ | 11 | +1.2e-4 | +1.9e-5 | −1.0e-5 | −1.0e-5 | +1.8e-5 | +1.8e-5 | −7.0e-5 |
| C3H8 + CH4 → 2 C2H6, kcal/mol | | +0.074 | −0.023 | +0.003 | +0.003 | −0.024 | −0.012 | +0.052 |
| points per atom | | 5500 | 3545–3629 | 3545–3629 SCF, 8109–8214 final | 8109–8214 | 4996–5068 | 9700 | 3545–3629 |

Whole-SCF wall time relative to exact-K RHF on the same system (COSX is
slower than exact exchange at all of these sizes and bases; see above for
where it wins):

| System | exact K | flat (50,110) | sgx (35,194) | sgx (35,194) + final | sgx (50,302) |
|---|---:|---:|---:|---:|---:|
| water / aug-cc-pVDZ | 0.3 s | 15.7× | 10.1× | 12.8× | 21.7× |
| butane / def2-SVP | 2.6 s | 17.7× | 11.8× | 14.3× | 25.5× |
| butane / def2-TZVP | 19.0 s | 5.4× | 3.9× | 4.4× | 7.5× |
| benzene / def2-SVP | 3.2 s | 13.2× | 8.8× | 10.6× | 18.9× |
| sulfamethoxazole / def2-SVP | 78.7 s | 7.2× | 5.9× | 6.5× | 14.4× |
| methane / cc-pVDZ | 0.2 s | 27.0× | 20.5× | 22.5× | 40.2× |
| ethane / cc-pVDZ | 1.0 s | 18.7× | 12.7× | 15.8× | 28.7× |
| propane / cc-pVDZ | 3.6 s | 13.8× | 9.7× | 11.0× | 20.8× |

What the table shows:

- The pruned `sgx (35,194)` grid uses 0.65× the points of flat (50,110)
  and is more accurate on seven of the eight molecules, by 1.7× (benzene) to
  29× (ethane). On methane it is 1.15× worse (9.2e-6 against 8.0e-6 Ha).
  Geometry tasks (optimize, frequencies) use this grid, since they
  run without the final pass.
- The final pass on `sgx (50,302)` reproduces an SCF converged on that grid to
  4e-8 Ha on seven molecules and 1.5e-7 Ha on sulfamethoxazole, at 0.45–0.59×
  of that SCF's cost. Together with the pruned SCF grid it is more accurate
  than flat (50,110) on all eight molecules and on the reaction (0.003
  against 0.074 kcal/mol), and faster on all eight (0.80–0.91× flat
  (50,110)'s wall time). This pair is the default for energies; ROHF/ROKS has
  no final pass and skips it with a note.
- More radial shells (35 → 50) at a 194 peak buy little; removing the
  pruning (flat 194) costs 2.7× the points for errors of the same size. The
  overlap fit helps on every molecule but water (1.8× to 21×; on water it is
  1.2× worse).

**COSX gradients.**

`task = "optimize"` and `task = "frequencies"` with
`k_builder = "cosx"` differentiate the COSX energy itself (grid-function,
ESP-integral and Becke-weight derivatives). With the default overlap fit the
fitted exchange is not variational in the orbitals, so the gradient adds an
orbital-response (Z-vector) term. The gradient is exact for RHF, RKS, UHF,
UKS, ROHF and ROKS with `cosx_overlap_fit = false`, and for RHF and UHF with
the default fit; measured against finite differences of the COSX energy it
agrees to 2e-9–4e-9 Ha/Bohr, on flat and pruned (`sgx`) grids alike, and with
RI-J (RIJCOSX). The open-shell exchange term is differentiated per spin; an
exact-exchange gradient paired with the same COSX energy misses those finite
differences by 4e-5–2e-4 Ha/Bohr (HO2 and NH2, B3LYP and ROHF, STO-3G). The
MBD@rsSCS gradient's orbital-response term on a UKS or ROKS reference uses the
same COSX derivative. The gradient differentiates the SCF-grid energy: with
`cosx_final_pass` the reported energy is the final-grid one, so geometry tasks
run without the pass and a gradient of a final-pass result is refused (an
ORCA-style gradient evaluated on the final grid misses finite differences of
the final-grid energy by 4.6e-7–7.3e-7 Ha/Bohr on water/6-31G, so it is not
used). Refused for gradient tasks before the SCF runs: fitted COSX with a
Kohn–Sham functional (RKS, UKS, ROKS; the fitted energy's response needs the
nuclear derivative of the XC Fock matrix) and fitted COSX for ROHF (no
restricted-open-shell response).

### In-core integrals (`[scf] eri`)

Below about 200 basis functions the whole set of AO integrals fits in a few
GB. With `[scf] eri = "incore"` (Python `eri=` / `scf_eri=`) the integrals of
the geometry are computed once, every shell quartet with no screening, and
stored as the packed lower triangle of the pair matrix (`P(P+1)/2` doubles
with `P = n(n+1)/2`, about `n⁴/8`: 0.17 GB at 114 functions, 0.57 GB at 154, and
1.68 GB for octane / cc-pVDZ, which `"auto"` takes at a 10 GB budget), charged to the `[memory]` budget. J is then one
matrix product per density and K a gathered matrix product, for any number
of densities, symmetric or not. The SCF, and the CPHF of the analytic
Hessian, the RI-MP2 Z-vector and the OO-MP2 loop that follow it on the same
geometry, all read the store. The store belongs to one geometry: each SCF
builds its own, and every use checks it against the atoms and basis it
serves.

`"auto"` (the default) takes the store when it needs at most a quarter of the
budget and J and K are both exact four-centre (no RI, COSX, LinK or range
separation); otherwise, and under MPI, the build is direct. `"direct"` is the
screened direct build and gives bit-for-bit the same output as before the key
existed. `"incore"` on a fitted or seminumerical route is an error. In-core
and direct J/K agree to a rounding bound derived from the summation length
(the largest measured ratio to that bound is 9e-3), and the in-core result
does not depend on the thread count.

On benzene / cc-pVDZ (12 threads) one direct J/K build takes 2.3 s; building
the store takes 1.1-2.2 s, and J + K from it 0.13 s. End to end, paired runs
(direct, in-core, in-core, direct, direct, in-core), wall and CPU (user +
system) seconds, mean ± sd of 3, 12 rayon threads, box load 5-13 from other
jobs:

| Run | direct wall | in-core wall | direct CPU | in-core CPU | wall ratio |
|---|---:|---:|---:|---:|---:|
| benzene / cc-pVDZ RHF energy | 9.43 ± 0.02 | 2.55 ± 0.01 | 103.5 ± 0.1 | 26.7 ± 0.2 | 3.7 |
| butane / cc-pVDZ RHF analytic frequencies | 58.3 ± 0.3 | 17.8 ± 0.2 | 624 ± 1 | 187 ± 1 | 3.3 |
| hexane / cc-pVDZ OO-RI-MP2 energy | 80.1 ± 0.5 | 25.2 ± 0.1 | 879 ± 2 | 258 ± 3 | 3.2 |

In the same window PySCF 2.13.1 (in-core by default at these sizes) took
0.52 ± 0.02 s wall (5.6 s CPU) for the benzene RHF and 39.9 ± 0.5 s wall
(188 s CPU) for the butane RHF and Hessian.

## Convergence

A few things worth knowing before debugging a stubborn SCF:

**Check `converged`.** These routines return a result whether or not they
converged; a non-converged SCF is a result with `converged = false`, not an
error. Downstream code that ignores the flag will happily consume a
half-converged density.

**Multiple solutions.** For systems like alkane chains, different
initial guesses converge to different SCF solutions, each in its own basin.
The guess picks the basin.

**Density-fitting has a noise floor.** DF-JK introduces an error floor that
makes energy-based convergence criteria below roughly 1e-9 meaningless; SCF
gates on the density RMS change instead.

**Near-linear-dependence.** Diffuse (aug-) basis sets on close-packed systems
can drive the overlap matrix near-singular; the canonical-orthogonalization
threshold is tunable via `FERRIC_LINDEP_THRESH`.

### Secondary-basis guess (Q-Chem `BASIS2`)

`[scf] guess_basis = "6-31g"` (Python `guess_basis="6-31g"` on `run_rhf`,
`run_uhf`, `run_rohf`, `run_dft`, `run_ksdft`) converges the same SCF
(method, functional, charge, multiplicity and settings) in the named basis
first, projects the result into the target basis, and starts the target SCF
from it. `guess_projection` picks how the projection is done:

- `"fock"` (default; Q-Chem `FOPPROJECTION`): the small-basis spin densities
  are projected, `D = P D_small Pᵀ` with `P = S_bb⁻¹ S_bs`, and the target
  Fock built at that density is diagonalised. Closed shell, that Fock build
  is the target SCF's first iteration. Open shell, the guess Fock is
  `h + J − K_σ` with no XC term, which is the form ferric's MINAO
  open-shell guess also uses.
- `"overlap"` (Q-Chem `OVPROJECTION`): the occupied MOs are mapped,
  `C = S_bb⁻¹ S_bs C_small`, and re-orthonormalised in the target metric. For
  ROHF, the doubly occupied block is orthonormalised first and the singly
  occupied block is made orthogonal to it.

`S_bb⁻¹` is the lindep-filtered pseudo-inverse from the canonical
orthogonalizer. A secondary basis missing an element, a mismatched ECP core
count, or a projected occupied space that is near-singular in the target
basis are errors. A secondary basis larger than the target is allowed, with
a warning. The guess is refused with `symmetry = true` and on periodic
systems. An explicit starting density or MOs take precedence, as they
do over `guess = "minao"`. With `guess_basis` set, `guess = "minao" | "sad" |
"hcore"` selects the starting point of the *secondary* SCF only; the target
starts from the projection. The dual-basis energy correction (Q-Chem
`DUAL_BASIS_ENERGY`) is not implemented.

**A secondary SCF that does not converge never stops the run.** The target
SCF then starts from its default guess, a warning is printed, and the report
says `converged = false`, why, and `used_as_guess = false`. Projecting the
unconverged last iterate was tried and rejected: on CN UHF/aug-cc-pVDZ with
an STO-3G secondary (which oscillates for 200 iterations), the target
started from it failed to converge as well, for both projections, while the
default guess converges in 22 iterations. The result reports the secondary
SCF's iterations, energy and convergence (Python: `result.guess_basis`). The
report is kept when a later stage supplies the final result: a stability
descent, a later ladder rung, or the exact stage after a DF pre-stage.

**Cost on re-solves.** Every solve without an explicit starting point runs
the secondary SCF again: each geometry of an optimization and each displaced
geometry of a finite-difference Hessian. A ladder's later rungs, the
stability descent's re-solves and the cDFT λ loop start from a carried
density or MOs, so they do not repeat it (cDFT builds the guess once, before
the λ loop).

**The guess can select the state.** The converged energy is *not*
guaranteed to equal the default-guess energy. Like any guess, it decides
which stationary state the SCF falls into, in either direction:

- NO UKS-PBE/aug-cc-pVDZ: MINAO −129.7833808133, BASIS2 −129.7833798480
  Ha. These are two states 0.97 µHa apart, and BASIS2 lands on the higher.
- He₂ at 0.3 Å RHF/aug-cc-pVTZ: BASIS2 and the core guess give
  −3.2458864766 Ha, while MINAO gives −3.2388590827 Ha. Here BASIS2 lands 7
  mHa *lower*.

Use `check_stability` (and `stability_descent`) when the state matters.

What it buys, measured in ferric (build 352eabab9, `density_conv` 1e-8;
target-SCF iterations, with the secondary SCF's own iterations listed
separately; converged energies agree to ≤ 3e-11 Ha in every row; wall times
were taken on a loaded machine and are not quoted):

| system | MINAO | hcore | BASIS2 fock | BASIS2 overlap | secondary SCF |
|---|---|---|---|---|---|
| NH₂ UHF 6-31G→aDZ | 14 | 18 (other state, +0.081 Ha) | 13 | 14 | 13 |
| NH₂ ROHF 6-31G→aDZ | 12 | – | 11 | 12 | 11 |
| CN UHF 6-31G→aDZ | 22 | no convergence in 200 | 15 | 16 | 24 |
| CN ROHF 6-31G→aDZ | 18 | – | 13 | 14 | 18 |
| O₂ UHF triplet 6-31G→aDZ | 12 | 14 | 10 | 11 | 11 |
| water dimer RHF 6-31G→aDZ | 14 | 17 | 13 | 13 | 13 |
| water dimer RHF 6-31G→aTZ | 14 | 17 | 14 | 14 | 13 |
| water dimer RHF cc-pVDZ→aTZ | 14 | 17 | 14 | 14 | 14 |

On closed shells the saving over MINAO is 0 to 1 target iterations, so there
is no wall-time win. On open-shell radicals it saves up to 7 target
iterations (CN), but the secondary SCF costs about as many iterations as it
saves, or more; they are cheap iterations in a small basis. The value is a
better starting point on hard open-shell cases, not a faster closed-shell
SCF. Fock projection beats overlap projection by 0 to 1 iterations.

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
bit-identical across `RAYON_NUM_THREADS`, a property pinned by tests.

A tree-fold reduction would be deterministic but would produce different
bits, since floating-point addition is not associative, so the sum must run
in ascending order.

## Cite

DIIS: Pulay 1980. MOM: Gilbert, Besley & Gill 2008. LinK: Ochsenfeld, White &
Head-Gordon 1998. COSX: Neese et al. 2009; overlap fit: Izsák & Neese 2011.
CSB/CSAM screening: Thompson & Ochsenfeld 2017. DFT grids: Becke 1988,
Treutler & Ahlrichs 1995, Lebedev & Laikov 1999. Functionals through libxc
(Lehtola et al. 2018): PBE, B3LYP, ωB97X-V, SCAN, r2SCAN, VV10. D3(BJ): Grimme
et al. 2010, Grimme, Ehrlich & Goerigk 2011. IEF-PCM: Cancès, Mennucci &
Tomasi 1997; COSMO: Klamt & Schüürmann 1993. P-RFO: Banerjee et al. 1985;
Bofill 1994. Full entries in [References](../reference/references.md).
