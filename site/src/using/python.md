# Python bindings

`import ferric` gives you the same engine as the `ferric` CLI, as plain
function calls that return result objects holding floats and numpy arrays. To
install it, see [Installation](./installation.md). The wheel is enough; you do
not need a clone of the repository to run anything on this page.

If you already know PySCF, [For PySCF users](./pyscf-users.md) maps the calls
you know onto ferric's and lists where the two behave differently.

Every snippet below uses an inline geometry. Run them in order, since later
ones reuse `water`, `o2`, `bs`, `bs_dz` and `aux` from earlier ones. The
snippets through "Properties and charges" were run when this page was written
(2026-09-23), and any output shown is what they printed. The one-line calls in
the reference tables were not run.

## Molecules and basis sets

<!-- doctest -->
```python
import ferric

water = ferric.Molecule.from_xyz_string("""3
water
O   0.000000   0.000000   0.117790
H   0.000000   0.755453  -0.471161
H   0.000000  -0.755453  -0.471161
""")
bs = ferric.BasisSet.bundled("sto-3g")
```

The string is standard XYZ: an atom count on the **first** line, a comment
line, then one `symbol x y z` line per atom. The count must be the first line,
so start the string with `"""3`. Starting it with `"""` and a newline gives an
empty first line, and parsing fails with `bad atom count`. `Molecule.from_xyz(path)`
reads the same format from a file. A leading `@` on a symbol (`@H`) makes a
ghost atom, which carries basis functions but no nucleus or electrons.

**Charge and spin belong to the molecule**, not to the SCF call:

<!-- doctest -->
```python
o2 = ferric.Molecule.from_xyz_string("""2
O2 triplet
O 0.0 0.0 0.0
O 0.0 0.0 1.208
""", charge=0, multiplicity=3)
```

`multiplicity` is 2S+1, so a triplet is 3. (PySCF's `spin` is 2S, so the same
triplet there is `spin=2`.) Both constructors default to `charge=0,
multiplicity=1`. `run_uhf` and `run_rohf` take no spin argument; they read it
from the molecule. The geometry-changing drivers (`run_frequencies`,
`run_saddle`, `run_irc`) also accept a `multiplicity=` keyword.

### Units

| Quantity | Unit |
|---|---|
| XYZ input to `from_xyz` / `from_xyz_string` | Ångström |
| `Molecule.coords()` | Ångström |
| `Molecule.coords_bohr()` | Bohr (ferric stores Bohr internally) |
| `point_charges=` / `smeared_charges=` on `run_rhf`, `run_uhf`, `run_rohf`, `run_dft`, … | Bohr, charges in e |
| `external_field=` | Hartree atomic units |
| `esp_at_points(..., points)` | points in Bohr; potential in atomic units |
| `omega` on `run_attenuated_rimp2`, `run_rs_mp2_rpa` | Å⁻¹ (default 0.420) |
| `r0` on the terfc drivers | Å |
| `omega` / `r0` on `compute_eri3_mo`, `compute_metric_2c` | Bohr⁻¹ / Bohr (raw, unlike the `run_*` drivers) |
| `QmmmSystem(..., coords_angstrom)` | Ångström |
| `QmmmSystem.point_charges()` | Bohr |
| Energies | Hartree |
| Gradients | Hartree/Bohr |

<!-- doctest -->
```python
print(water.coords()[0])        # (0.0, 0.0, 0.11779)            Ångström
print(water.coords_bohr()[0])   # (0.0, 0.0, 0.22259084021251865) Bohr
```

The [Sharp bits](./sharp-bits.md) page has the full units table, including the
QM/MM accessors.

### Bundled basis sets

`BasisSet.bundled(name)` loads a basis compiled into the library. Names are
case-insensitive. An unknown name raises `ValueError`. These 25 sets are
available (`cc-pvdz-rifit` is also accepted, as an alias of `cc-pvdz-ri`):

| Kind | Names |
|---|---|
| Orbital | `sto-3g`, `6-31g`, `cc-pvdz`, `cc-pvtz`, `aug-cc-pvdz`, `aug-cc-pvtz`, `aug-cc-pvqz`, `def2-svp`, `def2-tzvp`, `def2-qzvp` |
| Orbital with ECP (heavy elements) | `aug-cc-pvdz-pp`, `aug-cc-pvtz-pp` |
| Explicitly correlated (F12) | `cc-pvdz-f12` (orbital), `cc-pvdz-f12-optri` (OptRI auxiliary) |
| RI (MP2/RPA/CC fitting) | `cc-pvdz-ri`, `cc-pvtz-rifit`, `aug-cc-pvdz-rifit`, `aug-cc-pvtz-rifit`, `aug-cc-pvqz-rifit`, `def2-svp-rifit`, `def2-tzvp-rifit`, `def2-tzvpp-rifit`, `def2-qzvp-rifit`, `def2-qzvpp-rifit` |
| JK (SCF fitting) | `def2-universal-jkfit` |

The list comes from the `bundled()` match in `crates/ferric-core/src/basis.rs`.
The source records two coverage gaps: `cc-pvtz` lacks K, and `cc-pvtz-rifit`
lacks K and Ca. An RI run on a missing element errors; it is not silently
patched. The Python API has no loader for basis files on disk; the Rust
library parses BSE-JSON and Gaussian-94 files (see the
[Rust API](../reference/api.md)).

Most drivers take a `BasisSet` object. The drivers that move atoms
(`run_optimize`, `run_frequencies`, `run_saddle`, `run_irc`, `run_qmmm`,
`run_optimize_qmmm`) take the basis **name** as a string instead, because they
rebuild the basis at every geometry.

## Ground state

<!-- doctest -->
```python
rhf = ferric.run_rhf(water, bs)
print(rhf)
print(f"RHF: {rhf.energy:.10f} Ha, converged={rhf.converged}")
```

```text
RHF Energy: -74.9631468000 Ha (converged: true, 8 iterations)
RHF: -74.9631468000 Ha, converged=True
```

Open shell, using the triplet `o2` built above:

<!-- doctest -->
```python
bs_dz = ferric.BasisSet.bundled("cc-pvdz")
uhf  = ferric.run_uhf(o2, bs_dz)
rohf = ferric.run_rohf(o2, bs_dz)
print(f"UHF  {uhf.energy:.10f}  converged={uhf.converged}")
print(f"ROHF {rohf.energy:.10f}  converged={rohf.converged}")
```

```text
UHF  -149.6276689907  converged=True
ROHF -149.6079865946  converged=True
```

PySCF 2.12 on the same geometry and basis gives −149.6276689907 (UHF) and
−149.6079865946 (ROHF). `run_rohf` returns a `UhfResult`: it has α and β
densities and orbital energies, which coincide for the spatial orbitals.

Kohn–Sham DFT:

<!-- doctest -->
```python
dft = ferric.run_dft(water, bs_dz, functional="b3lyp")
print(f"B3LYP {dft.total_energy:.10f}")
```

`run_dft` is closed-shell only, and its default functional is LDA. It uses
density fitting for Coulomb by default (`def2-universal-jkfit`); `run_rhf`
does not. Pass `df_j_aux="exact"` for conventional four-centre Coulomb when you
compare against an exact-Coulomb code. `dispersion="d3bj"` adds Grimme D3(BJ);
the result then carries `e_scf`, `e_dispersion` and their sum in
`total_energy`. `with_gradient=True` also returns the analytic nuclear gradient
from `dft.gradient()`. There is no open-shell `run_dft`. Unrestricted
Kohn–Sham is reachable from Python only inside other drivers:
`run_frequencies(reference="uhf", xc=...)`, `run_u_gw(xc=...)` and
`run_qmmm(method="uks")`.

`run_rhf` also takes implicit solvent (`solvent=78.4` or `solvent="water"`,
IEF-PCM), point charges and a uniform field. See its docstring
(`help(ferric.run_rhf)`) for the full SCF knob set, which matches the CLI
`[scf]` section.

### Check `.converged`, and know what it means

`run_rhf`, `run_uhf`, `run_rohf` and `run_qmmm` **return a result whether or
not the SCF converged**. A non-converged result is not an error; it is a result
with `converged = False`, and its energy is a plausible, wrong number.
`run_dft`, `run_ksdft` and the correlated drivers (MP2, CC, RPA, GW, TDDFT)
behave differently: they **raise** if their reference SCF does not converge.

`converged = True` means the SCF reached a stationary point. It does not mean
the lowest one. The O2 triplet above with `sto-3g` instead of `cc-pvdz` shows
this. `run_uhf` with the default MINAO guess converges to −147.63397 Ha, which
is a saddle of the orbital Hessian 1.33 mHa above the UHF minimum at
−147.635296 Ha (PySCF's default guess lands on the same saddle). Without
`stability_descent` nothing in the result flags it.
`run_uhf(o2, bs, stability_descent=True)` follows the downhill eigenvector and
reaches the minimum. `guess="hcore"` converges (`converged=True`) to a much
higher stationary point, −147.3789 Ha, which lies above ferric's own ROHF
(−147.63219 Ha). A UHF energy above the ROHF energy for the same molecule
cannot be a ground state, so comparing the two is a cheap check for open-shell
work.

## Correlation

Correlated drivers run their own reference SCF internally, so they take the
molecule and basis rather than an SCF result. They also take an explicit
auxiliary (RI) basis. There is no automatic choice.

<!-- doctest -->
```python
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

rhf = ferric.run_rhf(water, bs_dz)
mp2 = ferric.run_rimp2(water, bs_dz, aux)
cc  = ferric.run_ccsd_t(water, bs_dz, aux)

print(f"RHF      {rhf.energy:.10f}")
print(f"RI-MP2   {mp2.total_energy:.10f}  (corr {mp2.mp2_corr:.10f})")
print(f"CCSD(T)  {rhf.energy + cc.correlation_energy + cc.t_correction:.10f}  total")
print(f"         {cc.correlation_energy:.10f}  CCSD correlation")
print(f"         {cc.t_correction:.10f}  (T)")
```

```text
closed-shell CCSD converged in 10 iterations. E_corr = -0.2135061893
RHF      -76.0267679974
RI-MP2   -76.2308014541  (corr -0.2040334567)
CCSD(T)  -76.2433412449  total
         -0.2135061893  CCSD correlation
         -0.0030670582  (T)
```

The first line is progress output that the CCSD solver prints to stdout.

`CcResult` holds only `correlation_energy` and `t_correction` (which is `None`
for `run_ccd` and `run_ccsd`). It carries no reference energy, so the total
above adds `run_rhf(...).energy` by hand. The MP2-family results all carry a
`total_energy`, and most also carry the reference energy.

Other members of the family use the same call shape:

<!-- doctest -->
```python
att   = ferric.run_attenuated_rimp2(water, bs_dz, aux, omega=0.420)  # Å⁻¹
scs   = ferric.run_scs_mp2(water, bs_dz, aux)
sos   = ferric.run_laplace_sos_mp2(water, bs_dz, aux)
oo    = ferric.run_oo_rimp2(water, bs_dz, aux)
mp3   = ferric.run_mp3(water, bs_dz, aux)
```

`frozen_core=` is accepted by every correlated driver. `run_ccd`,
`run_ccsd_t`, `run_terfc_rimp2` and the amplitude-threshold drivers
`run_drpa`, `run_drpa_scan` and `run_linlccd_amplitude` have no CLI
`method.kind`; `run_lmp2` and `run_lmp2_direct` do (`lmp2`, `lmp2-direct`).
The reference table below marks which drivers have one.

## Response and excited states

<!-- doctest -->
```python
rpa   = ferric.run_pdep_rpa(water, bs_dz, aux)            # RPA correlation
gw    = ferric.run_gw(water, bs_dz, aux)                  # G0W0@HF by default
tddft = ferric.run_tddft(water, bs_dz, aux, n_roots=3, method="tda")

print(rpa.total_energy, rpa.eigensolver_converged)
print(gw.mo_indices, gw.eps_qp)          # quasiparticle energies, Hartree
print(tddft.excitation_energies)         # Hartree
```

`run_gw` runs `method="g0w0"` on an HF reference unless you pass `xc=` (for
example `xc="pbe"`); by default it corrects HOMO−2 through LUMO+2. Check
`gw.outer_converged` and `gw.qp_converged` before using the numbers.
`run_u_gw` is the open-shell version.

`run_tddft` is closed-shell only. With `functional=...` it includes the
`(ia|f_xc|jb)` exchange-correlation kernel; meta-GGA, VV10 and range-separated
functionals are refused because their kernel is not built. With no
`functional`, it is CIS (`method="tda"`) or TDHF (`method="casida"`). Both
methods match PySCF `TDA`/`TDDFT` to 1e-3 eV on the systems listed in
[What is validated](../reference/validation.md#anchors).

## Properties and charges

The property functions take the molecule, the basis and a converged
`RhfResult` or `DftResult`. They work on closed-shell results.

<!-- doctest -->
```python
import numpy as np

rhf = ferric.run_rhf(water, bs)                      # water / STO-3G

q_lowdin = ferric.lowdin_charges(water, bs, rhf)
q_resp   = ferric.resp_charges(water, bs, rhf)
print(np.round(q_lowdin, 4), np.round(q_resp, 4))

# ESP 3 Bohr above each atom. Points are an (N, 3) array in Bohr.
pts = np.array(water.coords_bohr()) + np.array([0.0, 0.0, 3.0])
print(np.round(ferric.esp_at_points(water, bs, rhf, pts), 6))
```

```text
[-0.2525  0.1263  0.1263] [-0.6176  0.3088  0.3088]
[-0.064684 -0.067154 -0.067154]
```

The charge family is `mulliken_charges`, `lowdin_charges`,
`hirshfeld_charges`, `chelpg_charges` and `resp_charges`, all returning one
charge per atom in units of e. `resp_charges` is a single-stage restrained fit,
not the multi-stage, multi-conformer RESP procedure. `esp_at_atoms` gives the
potential at each nucleus. `hirshfeld_polarizability` returns per-atom 3×3
polarizability tensors (Bohr³) and needs an RI basis.

For NPZ export of ML-ready features (MO coefficients, PDEP eigenvectors, ESP,
charges, polarizabilities, C6 coefficients) in one run, use the CLI's
`[rpa] export_npz` section; see [Input file (TOML)](../reference/input.md).

## What comes back

Results are Python objects with plain attributes for scalars and methods for
arrays:

<!-- doctest -->
```python
D = rhf.density()             # numpy.ndarray, (n_bf, n_bf), AO basis
e = rhf.orbital_energies()    # numpy.ndarray, ascending, Hartree
C = rhf.mo_coefficients()     # numpy.ndarray, (n_bf, n_mo)
```

Matrices and tensors come back as `numpy.ndarray`. Per-atom lists (charges,
ESP values) come back as Python lists; wrap them in `np.asarray` if you need
arrays. `run_lmp2`, `run_lmp2_direct`, `run_drpa`, `run_linlccd_amplitude` and
`tune_omega` return plain `dict`s, and `run_drpa_scan` returns a list of them.

AO-basis matrices follow libint2's basis-function conventions, which are not
PySCF's. MEASURED on CO/cc-pVDZ: total and orbital energies agree with
PySCF to 3e-12 and 8e-9 Ha, and the two AO density matrices differ element by
element by up to 1.5. Compare invariant quantities, not raw AO matrices.

## Memory, threads and MPI

Most drivers accept `memory_budget_gb` (GiB). It sets the same per-allocation
limits as the CLI's `[memory] budget_gb`: an allocation that does not fit is
spilled to disk, recomputed on demand (the DFT grid AO cache) or refused with
an error naming it (for example `run_rimp2` and `run_ccsd`). Unlike the CLI,
Python installs no shared ledger, so each check compares its own allocation
with the whole budget rather than with what other live allocations have left.
Two checks instead subtract the process's current resident memory (RSS) first
and allow 90% of the remainder: the KS-DFT decision to store or recompute the
grid AO cache, and the UKS Newton/TRAH f<sub>xc</sub> kernel's second grid
cache. Because RSS includes everything already resident, those two see less
than the full budget.
It is **not** a cap on total process memory; see
[Sharp bits](./sharp-bits.md#memory-budget_gb-does-not-cap-the-whole-process).

<!-- doctest -->
```python
mp2 = ferric.run_rimp2(water, bs_dz, aux, memory_budget_gb=8.0)
```

`import ferric` pins OpenBLAS to one thread unless `OPENBLAS_NUM_THREADS` is
already set. ferric parallelizes with rayon instead. `run_rhf`, `run_uhf`,
`run_rohf`, `run_dft`, `run_rimp2`, `run_ccsd`, `run_ccsd_t`, `run_gw` and
`run_tddft` release the GIL, so independent jobs submitted from a
`ThreadPoolExecutor` run in parallel. The other drivers hold it. For throughput
across many molecules, prefer many single-threaded processes.

Do not run a Python script under `mpirun`. The bindings expose no rank or
world-size accessor, so every rank runs the whole script. Distributed-memory
runs go through the CLI built from source with MPI; see
[Installation](./installation.md#mpi).

## Full reference

The module registers **61 public functions** and **39 classes**. That count
excludes `_cli_main`, the entry point behind the `ferric` console command. It
also exports two constants: `DEFAULT_TEMPERATURE_K` (298.15) and
`BOLTZMANN_HARTREE_PER_K`. The list below was taken from the registration
block of `#[pymodule] fn ferric` in `crates/ferric-python/src/lib.rs`, and each
purpose line is condensed from that item's doc comment, or from its code where
it has none. `help(ferric.<name>)` shows the full docstring and signature.

"CLI" gives the matching `method.kind`, task or TOML section, or "—" when the
capability is Python-only. How well each one is validated is in the
[capability matrix](../reference/capabilities.md).

### Molecules and basis sets

| Name | Purpose | CLI |
|---|---|---|
| `Molecule` | Geometry, charge and multiplicity. `from_xyz`, `from_xyz_string`, `coords`, `coords_bohr`, `symbols`, `atomic_numbers`, `is_ghost`, `natoms`, `nelec`, `nuclear_repulsion`, `to_xyz_string`. | `[molecule]` |
| `BasisSet` | A Gaussian basis set, orbital or auxiliary. `BasisSet.bundled(name)`. | `[basis]` |

### SCF and DFT

| Name | Purpose | CLI |
|---|---|---|
| `run_rhf` | Closed-shell RHF with the full SCF knob set, point charges, field and IEF-PCM solvent. | `rhf` |
| `run_uhf` | Unrestricted HF; α/β counts come from the molecule's charge and multiplicity. | `uhf` |
| `run_rohf` | Restricted open-shell HF (Guest–Saunders coupling); returns a `UhfResult`. | `rohf` |
| `run_dft` | Closed-shell Kohn–Sham DFT (LDA/GGA/hybrid/RSH/meta-GGA by name), optional D3(BJ) and analytic gradient. | `ksdft` |
| `run_ksdft` | Alias of `run_dft`. | `ksdft` |
| `d3bj_energy` | Grimme D3(BJ) dispersion energy for a molecule and functional, in Hartree. | `[dft] dispersion` |
| `tune_omega` | IP-based (Baer/Kronik) tuning of an RSH functional's ω (Bohr⁻¹); closed-shell neutral plus doublet cation. | — |
| `dft_grid_point_count` | Number of points in the main KS grid `run_dft` would build for the molecule with the same `grid_*` kwargs, without running an SCF (shows what pruning saves). | — |
| `RhfResult` | Result of `run_rhf`: `energy`, `converged`, `iterations`, `density()`, `orbital_energies()`, `mo_coefficients()`. | |
| `UhfResult` | Result of `run_uhf`/`run_rohf`: α and β densities and orbital energies. | |
| `DftResult` | Result of `run_dft`: `total_energy`, `e_scf`, `e_dispersion`, `converged`, `exit_reason()`, `density()`, `gradient()`. | |

### Constrained DFT

| Name | Purpose | CLI |
|---|---|---|
| `run_cdft` | Constrained UHF, or UKS when `functional` names a libxc functional other than `"HF"` (`None` and `"HF"`, any case, give UHF): minimize the energy subject to fragment population constraints (Wu–Van Voorhis nested λ loop). Raises if the λ loop does not converge. | — |
| `CdftConstraint` | One fragment constraint: `atoms` (0-based), `target` (a Becke electron population, not a net charge), `kind` = `"charge"` (Nα + Nβ) or `"spin"` (Nα − Nβ). | — |
| `cdft_coupling` | Wu–Van Voorhis coupling H_ab between two converged single-`"charge"`-constraint states solved with the same geometry, basis, occupations and Hamiltonian. | — |
| `CdftResult` | `energy` (without the constraint term), `converged`, `scf_converged`, `lambdas`, `populations`, `targets`, `density_alpha()`, `density_beta()`, `weight_matrix(i)`. | |
| `CdftCouplingResult` | `h_ab` (sign is a phase convention), `s_ab`, `e_a`, `e_b`. | |

See [Constrained DFT](../methods/cdft.md) for a worked example.

### Geometry, vibrations and reaction paths

| Name | Purpose | CLI |
|---|---|---|
| `run_optimize` | RHF geometry optimization (basis by name). | `task = "optimize"` |
| `run_frequencies` | Harmonic frequencies by finite differences of the analytic gradient; RHF/UHF/ROHF or their KS variants. | `task = "frequencies"` |
| `run_saddle` | First-order saddle-point (transition-state) search by P-RFO; closed-shell. | — |
| `run_irc` | Intrinsic reaction coordinate in both directions from a saddle; closed-shell. | — |
| `OptimizeResult` | `energy`, `converged`, `steps`, `energy_trace`, `mol()`. | |
| `FrequencyResult` | Frequencies in cm⁻¹ (negative = imaginary), normal modes, `asymmetry` diagnostic. | |
| `SaddleResult` | Outcome of a P-RFO search: geometry (Å), `n_imaginary`, `imaginary_mode`, `is_transition_state()`. | |
| `IrcResult` | Both directions of an IRC, plus the saddle they came from. | |
| `IrcBranch` | One direction of an IRC walk. | |

### QM/MM

| Name | Purpose | CLI |
|---|---|---|
| `QmmmSystem` | A QM/MM partition with link atoms and boundary-charge schemes (coordinates in Å). | `[qmmm]` |
| `MmTopology` | Explicit-parameter AMBER-form MM force field; assigns no parameters itself. | — |
| `run_qmmm` | Embedded SCF energy plus QM gradient, MM forces and full gradient. | `[qmmm]` (energy) |
| `run_optimize_qmmm` | Optimize a `QmmmSystem`; QM atoms always move, MM atoms per `move_mm`. | — |
| `QmmmResult` | `energy`, `qm_gradient()`, `mm_forces()` (forces, not gradients), `full_gradient()`. | |
| `QmmmOptimizeResult` | The relaxed partition and its energy trajectory. | |

See [QM/MM](./qmmm.md) for a worked example.

### MP2 family

| Name | Purpose | CLI |
|---|---|---|
| `run_rimp2` | RI-MP2 on a closed-shell RHF reference. | `rimp2` |
| `run_oo_rimp2` | Orbital-optimized RI-MP2 (level-shifted Newton + DIIS + Cayley rotation). | `oo-rimp2` |
| `run_mp3` | MP3 on an RHF reference, with RI integrals. | `mp3` |
| `run_attenuated_rimp2` | RI-MP2 with the erfc-attenuated operator; ω in Å⁻¹, default 0.420. | `att-rimp2` |
| `run_terfc_rimp2` | RI-MP2 with the exact tempered-erfc operator at one cutoff `r0` (Å); needs the terfc tables. | — |
| `run_scs_mp2` | Spin-component-scaled MP2 (defaults c_OS = 6/5, c_SS = 1/3). | `scs-mp2` |
| `run_scs_mp2_2terfc` | Dual-attenuated SCS-MP2(2terfc); needs the terfc tables. | `scs-mp2-2terfc` |
| `run_mp2_v` | MP2-V: attenuated MP2 plus damped VV10 nonlocal correlation. | `mp2-v` |
| `run_double_hybrid` | B2PLYP or DSD-PBEP86 double hybrid. | `b2plyp`, `dsd-pbep86` |
| `run_laplace_mp2` | Laplace-transform RI-MP2 (default 7 quadrature points). | `laplace-mp2` |
| `run_laplace_sos_mp2` | Laplace-transform SOS-MP2, `E = c_os · E_OS`; MO, AO or AO-sparse formulations. | `laplace-sos-mp2` |
| `RiMp2Result` | Result of `run_rimp2` and `run_terfc_rimp2`: `total_energy`, `rhf_energy`, `mp2_corr`. | |
| `OoRiMp2Result` | Result of `run_oo_rimp2`, with `converged` and `grad_norm`. | |
| `Mp3Result` | `e_hf`, `e_mp2`, `e_mp3`, `e_corr`, `e_total`. | |
| `AttenuatedMp2Result` | Attenuated MP2 total, correlation and spin components. | |
| `ScsMp2Result` | Result of `run_scs_mp2`/`run_scs_mp2_2terfc`, with `e_os`/`e_ss`. | |
| `Mp2VResult` | MP2-V total, attenuated MP2 part and VV10 part. | |
| `LaplaceMp2Result` | Laplace MP2 total, correlation and spin components. | |
| `SosMp2Result` | Scaled and unscaled OS energy, `c_os`, `n_quad` and `formulation` echoed back. | |
| `DoubleHybridResult` | Result of `run_double_hybrid`: `total_energy`, `e_ks`, `e_corr_scaled`, `e_os`, `e_ss`, `c_os`, `c_ss`. | |

### Amplitude-threshold local correlation

All closed-shell. `eps = 0` reproduces the canonical method; a finite `eps`
carries a one-sided truncation error. Each returns a `dict`.

The canonical reference is opt-in. `run_lmp2`, `run_lmp2_direct`, `run_drpa`
and `run_drpa_scan` take `compute_reference` (default `False`); only with
`compute_reference=True` do they compute it, and otherwise the dict's
reference key (`e_corr_canonical_ri` for LMP2, `e_corr_plasmon_canonical` for
dRPA) is present and `None`. The reference is a full canonical calculation
over global tensors, so switching it on removes any cost saving.
`run_linlccd_amplitude` computes no canonical reference.

| Name | Purpose | CLI |
|---|---|---|
| `run_lmp2` | Amplitude-threshold local MP2. | `lmp2` |
| `run_lmp2_direct` | Integral-direct local MP2 that never forms the global 3-index tensor. | `lmp2-direct` |
| `run_drpa` | Amplitude-threshold direct RPA (drCCD Riccati). | — |
| `run_drpa_scan` | `run_drpa` over a list of `eps` values, sharing one SCF and localization. | — |
| `run_linlccd_amplitude` | Amplitude-threshold LinLCCD (`variant` = `"drivers"`, `"hh"`, `"full"`). | — (CLI `linlccd` is canonical LinLCCD(hh)) |

### Coupled cluster

All use RI integrals from the `auxbasis` argument and a closed-shell RHF
reference.

| Name | Purpose | CLI |
|---|---|---|
| `run_ccd` | CCD correlation energy. | — |
| `run_ccsd` | Spin-adapted closed-shell CCSD. | `ccsd` |
| `run_ccsd_t` | CCSD plus the spin-adapted (T) correction. | — |
| `CcResult` | `correlation_energy` and `t_correction` (`None` without triples). No reference energy. | |

### RPA, GW and excited states

| Name | Purpose | CLI |
|---|---|---|
| `run_pdep_rpa` | Direct RPA correlation energy by PDEP (projective dielectric eigenpotentials); accepts point charges, field and solvent. | `pdep-rpa` |
| `run_rs_mp2_rpa` | Range-separated SR-MP2 + LR-RPA (`formulation` = `"delta-lr"` or `"coupled-rings"`; ω in Å⁻¹). | `rs-mp2-rpa` |
| `run_gw` | Closed-shell G0W0 / COHSEX / evGW0 / evGW on an RHF or RKS reference. | `gw` |
| `run_u_gw` | Open-shell GW variants on a UHF/UKS or ROHF reference. | `gw` with multiplicity > 1 |
| `run_bse_tda` | BSE-TDA singlet excitation energies on a closed-shell RHF reference. | `bse-tda` |
| `run_tdhf_static_polarizability` | RPAx@KS static (ω = 0) polarizability on a closed-shell KS reference. | `tdhf-static-polarizability` |
| `run_tddft` | TDA or Casida excitations on a closed-shell HF (CIS/TDHF) or KS reference, with the f_xc kernel. | `tda`, `tddft` |
| `PdepRpaResult` | `total_energy`, `e_rpa`, `eigensolver_converged`, eigenvalues and quadrature grid. | |
| `RsMp2RpaResult` | SR-MP2, LR-MP2 and dRPA pieces; which fields are set depends on `formulation`. | |
| `GwResult` | `eps_qp`, `eps_mf`, `sigma_x`, `sigma_c`, `z_factor`, `outer_converged`, `qp_converged`. | |
| `UGwResult` | α and β versions of the `GwResult` fields. | |
| `BseResult` | Excitation energies and oscillator strengths. | |
| `TdhfStaticPolarizabilityResult` | Polarizability `tensor` and its isotropic value `iso`. | |
| `TddftResult` | `excitation_energies`, `oscillator_strengths`, `method`. | |

### Properties and charges

Each takes `(mol, basis_set, result)` with a converged closed-shell
`RhfResult` or `DftResult`; `esp_at_points` also takes the points and
`hirshfeld_polarizability` an RI basis.

| Name | Purpose | CLI |
|---|---|---|
| `esp_at_atoms` | Electrostatic potential at each nucleus, in atomic units. | `[rpa] compute_esp` |
| `esp_at_points` | Electrostatic potential at arbitrary points given in Bohr. | `[rpa] compute_esp_surface` (vdW-surface points only) |
| `mulliken_charges` | Mulliken population charges. | `[rpa] compute_mulliken_charges` |
| `lowdin_charges` | Löwdin (symmetric-orthogonalization) charges. | `[rpa] compute_lowdin_charges` |
| `hirshfeld_charges` | Hirshfeld charges with a single-exponential Slater proatom. | `[rpa] compute_hirshfeld_charges` |
| `chelpg_charges` | CHELPG charges fitted to the ESP on a grid. | `[rpa] compute_chelpg_charges` |
| `resp_charges` | Single-stage RESP (restrained ESP-fit) charges. | `[rpa] compute_resp_charges` |
| `hirshfeld_polarizability` | Per-atom Hirshfeld-partitioned static polarizability tensors (Bohr³) from PDEP-RPA. | — |
| `orbital_moments` | Per-orbital centroids and spatial spreads (Bohr) of the restricted MOs. | — |
| `density_second_moment` | 3×3 second-moment tensor of the electron density (Bohr²). | — |

The `[rpa] compute_*` keys write into the CLI's NPZ export.

### Integrals and orbitals (prototyping)

These take `omega`/`r0` in raw Bohr units, unlike the `run_*` drivers.

| Name | Purpose | CLI |
|---|---|---|
| `compute_eri3` | Raw 3-centre Coulomb integrals (P\|μν), shape (naux, n_bf, n_bf). | — |
| `compute_eri3_mo` | MO-basis 3-centre integrals (P\|pq) for any two coefficient matrices, built blockwise under a memory budget. | — |
| `compute_metric_2c` | 2-centre metric (P\|w\|Q) over the auxiliary basis, Coulomb by default. | — |
| `shell_info` | Shell centres (Bohr), first-function offsets and sizes, for building fitting domains. | — |
| `boys_localize` | Foster–Boys localization of given orbitals; returns a `BoysResult`. | — |
| `BoysResult` | Result of `boys_localize`: `c_loc()`, `centers()`, `converged`, `iterations`. | |

### Conformer ensembles

| Name | Purpose | CLI |
|---|---|---|
| `ConformerEnsemble` | Conformers of one species sharing atom order, composition, charge and multiplicity. | — |
| `BoltzmannWeights` | Boltzmann populations of an ensemble at one temperature. | — |
| `EnsembleDiagnostics` | Population-structure readout: effective number of conformers, dominance verdict. | — |
| `WeightedStats` | A weighted mean with its spread (`mean`, `std_dev`, `min`, `max`). | — |
| `boltzmann_weights` | Boltzmann weights from a list of energies (Hartree) at a temperature (default 298.15 K). | — |
| `weighted_stats` | Weighted mean and standard deviation of a scalar property. | — |
| `weighted_stats_vector` | The same, component-wise, for a vector property. | — |
| `weighted_stats_tensor` | The same, element-wise, for a rank-2 tensor property. | — |
