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
bound, not the convergence target.

**Accuracy.** Proven (`rhf`, `uhf`, `rohf`).

## Kohn–Sham DFT

**What it is.** Kohn–Sham DFT through **libxc**: LDA, GGA, hybrid and
range-separated hybrid functionals (for example PBE, B3LYP, ωB97X-V), **VV10**
nonlocal correlation, and the **SCAN / r2SCAN** meta-GGAs (τ-dependent; no
density Laplacian). The solvers cover RKS, UKS and ROKS.

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
gradient is implemented; `ferric-d3/tests/gradient_vs_fd.rs`), and so does
closed-shell `task = "frequencies"`: see
[Frequencies with dispersion](#frequencies-with-dispersion).

**Accuracy.** Agrees with simple-dftd3 1.6.0 to below 1e-12 Ha on water,
methane, benzene and an argon dimer with the three-body term off on both
sides (`ferric-d3/tests/vs_reference_dftd3.rs`). The omitted three-body term
is 0.1% of the two-body energy at benzene and grows with size.

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
of 2.7e-6–9.8e-6. The relaxation term needs an LDA, GGA or hybrid-GGA
functional without VV10, no implicit solvation, polarizable embedding or cDFT
constraints, and integer aufbau occupation (no MOM); UKS and ROKS additionally
need exchange that is not COSX. Other setups are refused rather than given an
approximate gradient. Open-shell frequencies with dispersion are refused. Python's `run_dft` is closed-shell
only.

## Frequencies with dispersion

**What it is.** Harmonic frequencies on the dispersion-corrected surface
E(KS) + E(disp), closed-shell Kohn–Sham only. The Hessian is the central
difference of the corrected analytic gradient: at each of the 6N displaced
geometries the SCF is converged, the KS gradient and the dispersion gradient
are evaluated there, and their sum is differenced. For MBD@rsSCS the
dispersion gradient is the exact one above, so the response of the Hirshfeld
volumes to the displacement, through the SCF density, is in the Hessian.
There is no analytic Hessian on this path: `[frequencies] hessian = "analytic"`
is an error and `"auto"` runs finite differences.

**Run it.** `[dft] dispersion` with `method.task = "frequencies"` on a
closed-shell KS SCF. The printout gives the corrected `energy`, `E(KS-DFT)`
and the dispersion energy at the input geometry, and the JSON run log gets a
`dispersion` record with the same fields as an energy run. Python:
`run_frequencies(mol, basis, xc="PBE", dispersion="d3bj")`, which reports
`.e_dispersion` and the corrected `.energy`. Open-shell (UKS/ROKS)
frequencies with dispersion are refused.

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

The shifts are small for a single molecule. For PBE/6-31G water at its
uncorrected PBE minimum (1594.35, 3500.27 and 3669.29 cm⁻¹), D3(BJ) shifts
the three modes by +0.006, −0.077 and −0.095 cm⁻¹, and MBD@rsSCS by +0.107,
−0.406 and −0.516 cm⁻¹.

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
and, for DFT, against PySCF. RI-MP2 also has an analytic gradient, closed
shell only (there is no unrestricted MP2 nuclear gradient).

**Optimization.** `method.task = "optimize"` for `rhf`, `uhf`, `rohf`,
`ksdft`, `rimp2` and `pdep-rpa` (`examples/h2_opt.toml`,
`examples/h2-lda-opt.toml`); on an open-shell molecule only `uhf`, `rohf` and
`ksdft`. Python `ferric.run_optimize` is RHF.

**Harmonic frequencies.** `method.task = "frequencies"` for `rhf`, `uhf`,
`rohf` and `ksdft` (`examples/water-frequencies.toml`); Python
`ferric.run_frequencies(mol, basis_name, reference="rhf", xc=...)`. Mass-weighted,
translations and rotations projected out. The Hessian is:

- **Analytic** for closed-shell RHF, and for UHF of any multiplicity, with
  exact four-centre J/K, no ECP, no external potential or solvent, and a basis
  up to f functions: one SCF plus a coupled-perturbed HF solve per nuclear
  coordinate (for UHF, one solve coupling the α and β orbital rotations). This
  needs a libint2 with second derivatives (the build
  `scripts/install-libint.sh` installs).
- **Central finite differences of the analytic gradient** everywhere else
  (ROHF, KS-DFT, RI J/K, ECPs, embedding, g functions): 6N gradient
  evaluations. The displacement `[frequencies] delta` (default 5e-3 Bohr) is a
  real accuracy knob; the printed Hessian asymmetry, zero in exact arithmetic,
  is the check that it and the SCF thresholds suit the system.

`[frequencies] hessian` (Python `hessian=`) selects it: `"auto"` (default,
analytic where it applies), `"analytic"` (an error where it does not) or
`"fd"`. The output names the one that ran (`Hessian = analytic`; Python
`.hessian_source`).

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
| `k_builder = "cosx"` | seminumerical (COSX) K on a grid; with RI-J active this is RIJCOSX | grid-dependent error, see below | RHF/UHF/ROHF and their Kohn–Sham variants, Coulomb operator only; analytic gradient for RHF, RKS, UHF (see below) |

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
| COSX, default (sgx (35,194) + final pass on sgx (50,302)) | water / aug-cc-pVDZ | −8.1e-9 Ha |
| COSX, default | butane / def2-SVP | −3.4e-5 Ha |
| COSX, default | butane / def2-TZVP | −4.3e-6 Ha |
| COSX, flat (50,110) | water / cc-pVDZ | 5e-6 Ha |
| COSX, flat (50,110) | butane / def2-SVP | 1.7e-4 Ha |
| COSX, flat (50,110) | butane / def2-TZVP | 1.2e-4 Ha |
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
is being re-measured. `link` is skipped with a warning, never silently,
whenever density-fitted J/K is active; both `link` and `cosx` are skipped with
a warning when the functional uses no exact exchange or is range-separated
(exchange then comes from the SR/LR fitters).

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
(50,110) grid); absolute energies do not. Four knobs, all optional:

- `cosx_grid = { radial = 35, angular = 194, prune = "sgx" }` is the default;
  `{ radial = 50, angular = 110 }` is the flat grid. The angular order
  matters most: on butane/def2-TZVP the error falls from
  1.2e-4 Ha at 110 points per shell to 4.0e-6 Ha at 302, while going from 50
  to 100 radial shells changes it by 1e-6 or less. Cost grows with the number
  of points. `angular` must be one of 6, 14, 26, 50, 110, 194, 302, 434 or
  590. `prune = "sgx"` prunes the grid the way ORCA's GridX and PySCF's SGX
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
  This is the grid geometry tasks (optimize, frequencies) use, since they
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
orbital-response (Z-vector) term. The gradient is exact for RHF, RKS and UHF
with `cosx_overlap_fit = false`, and for RHF and UHF with the default fit;
measured against finite differences of the COSX energy it agrees to
2e-9–4e-9 Ha/Bohr, on flat and pruned (`sgx`) grids alike, and with RI-J
(RIJCOSX). The gradient differentiates the SCF-grid energy: with
`cosx_final_pass` the reported energy is the final-grid one, so geometry tasks
run without the pass and a gradient of a final-pass result is refused (an
ORCA-style gradient evaluated on the final grid misses finite differences of
the final-grid energy by 4.6e-7–7.3e-7 Ha/Bohr on water/6-31G, so it is not
used). Fitted COSX with a Kohn–Sham functional, UKS and ROHF/ROKS are refused
for gradient tasks before the SCF runs.

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
