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

The string is standard XYZ: an atom count on the first line, a comment
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
case-insensitive. An unknown name raises `ValueError` listing the known ones;
`BasisSet.bundled_names()` returns them, and `bs.elements()` gives the atomic
numbers a set covers. These 64 sets are available:

| Kind | Names |
|---|---|
| Pople | `sto-3g`, `6-31g`, `6-31g*`, `6-31g**`, `6-31+g*`, `6-31+g**`, `6-31++g**`, `6-311g**`, `6-311++g**` |
| Dunning | `cc-pvdz`, `cc-pvtz`, `cc-pvqz`, `aug-cc-pvdz`, `aug-cc-pvtz`, `aug-cc-pvqz` |
| Dunning with ECP (Z >= 29; lighter elements are the all-electron set) | `cc-pvdz-pp`, `cc-pvtz-pp`, `cc-pvqz-pp`, `aug-cc-pvdz-pp`, `aug-cc-pvtz-pp`, `aug-cc-pvqz-pp` |
| Karlsruhe (ECP inline for Z >= 37) | `def2-svp`, `def2-svpd`, `def2-tzvp`, `def2-tzvpp`, `def2-tzvpd`, `def2-tzvppd`, `def2-qzvp`, `def2-qzvpp`, `def2-qzvppd` |
| Jensen | `pcseg-0`, `pcseg-1`, `pcseg-2`, `aug-pcseg-1`, `aug-pcseg-2` |
| Explicitly correlated (F12) | `cc-pvdz-f12` (orbital), `cc-pvdz-f12-optri` (OptRI auxiliary) |
| RI (MP2/RPA/CC fitting) | `cc-pvdz-rifit`, `cc-pvtz-rifit`, `cc-pvqz-rifit`, `aug-cc-pvdz-rifit`, `aug-cc-pvtz-rifit`, `aug-cc-pvqz-rifit`, the six `<set>-pp-rifit` partners of the ECP sets, and `def2-<x>-rifit` for each of the nine def2 sets |
| JK (SCF fitting) | `def2-universal-jkfit`, `cc-pvtz-jkfit`, `cc-pvqz-jkfit`, `aug-cc-pvdz-jkfit`, `aug-cc-pvtz-jkfit`, `aug-cc-pvqz-jkfit` |

Aliases: the Pople stars also load as `6-31g(d)`, `6-31g(d,p)`, `6-31+g(d)`,
and so on; every RIFIT set also loads as `<orbital>-ri` and ORCA's
`<orbital>/c` (so `cc-pvdz-ri` is `cc-pvdz-rifit`); the JKFIT sets as
`<orbital>/jk`; and `def2-universal-jkfit` as `def2/jk` or PySCF's
`def2-<x>-jkfit`.

The `aug-cc-pv{d,t,q}z-jkfit` sets (Weigend, PCCP 4, 4285 (2002)) are the one
exception to "from the Basis Set Exchange": BSE does not carry them. They come
from Molpro's `weigend_jkfit.libmol` via Psi4's library (commit pinned in
`provenance.json`), checked bit-identical to PySCF's copy. Two caveats from
that source: their Ne, Ar and Kr functions are extrapolated, not published,
and `aug-cc-pvdz-jkfit` was never published at all; it is
`aug-cc-pvtz-jkfit` with the highest angular momentum dropped. For the
S66-type noncovalent work they exist for, `aug-cc-pvtz-jkfit` is the set to
use with aug-cc-pVTZ.

Each set covers what its source covers, and an element it lacks is an error
at basis setup that names the element and the covered range (for example
`cc-pvtz` has H-Ar and Ca-Kr, with no K; `cc-pvtz-jkfit` and `cc-pvqz-jkfit`
have only H, B-F, Al-Cl and Ga-Br; the `-pp-rifit` sets cover H-Ar (the aug ones
without Li and Na), Sc-Ni at TZ and QZ only, Y-Pd and Hf-Pt). It is never patched silently. The Pople sets follow
Gaussian's d convention: the 6-31G family has Cartesian d (6D), and the
6-311G family has spherical d (5D) on every element, including Na-Ar, which
the Basis Set Exchange marks Cartesian. Everything else is spherical, as the
Basis Set Exchange types it. See
[For PySCF users](./pyscf-users.md) for matching PySCF.
`crates/ferric-core/src/basis/provenance.json` records the source, version and
element list of every set. `BasisSet.from_bse_json(path)` loads any other
Basis Set Exchange JSON file.

Validation: every set is checked against PySCF fed the same basis
(`crates/ferric-mp2/tests/basis_coverage_vs_pyscf.rs`, at most 1e-8 Ha). Orbital
sets are checked by an RHF energy, RI sets by an RI-MP2 correlation energy, and
JK sets also by a density-fitted (RI-J + RI-K) RHF energy. For the BSE sets,
PySCF fetches the basis from BSE by name, so the comparison covers both the
shipped digits and ferric's parsing and normalisation. For the three aug-JKFIT
sets, PySCF uses its own copy of the data. The digits are covered instead by
the import's Psi4-vs-PySCF comparison, and the energy comparison covers parsing
and normalisation.

Most drivers take a `BasisSet` object. The drivers that move atoms
(`run_optimize`, `run_frequencies`, `run_saddle`, `run_irc`, `run_qmmm`,
`run_optimize_qmmm`) take the basis name as a string instead, because they
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
compare against an exact-Coulomb code. The RI-J tensor is stored as f32 by
default (half the memory; about 1e-7 Ha per atom of energy shift, see
[RI-J tensor precision](../methods/scf.md#ri-j-tensor-precision));
`ri_j_precision="f64"` restores the f64 tensor bit for bit, any other value
than `"f32"`/`"f64"` raises `ValueError`, and `dft.ri_j_precision` says which
was used. Drivers with more than one iterative stage (`run_frequencies`,
`run_cdft`, the MP2 family, GW, TDDFT, ...) spell it `scf_ri_j_precision`.
The geometry and frequency drivers (`run_optimize`, `run_optimize_uhf`,
`run_optimize_qmmm`, `run_frequencies`, `run_saddle`, `run_irc`) default to
the f64 tensor instead, because the f32 energy surface jitters between nearby
geometries; `scf_ri_j_precision="f32"` there is still honoured, and the
certificate's `ri-j` entry says which was requested and why.
The RI-K tensor (hybrids, HF with `df_k_aux`) is held as its packed
symmetric half; `ri_k_precision="f32"` (opt-in; `scf_ri_k_precision` on
`run_cdft` and `run_frequencies`) stores it in f32 for another factor of
two, at a cost described in
[RI-K tensor storage](../methods/scf.md#ri-k-tensor-storage);
`dft.ri_k_precision` says which was used.
`dispersion="d3bj"` adds Grimme D3(BJ);
the result then carries `e_scf`, `e_dispersion` and their sum in
`total_energy`. `with_gradient=True` also returns the analytic nuclear gradient
from `dft.gradient()`. There is no open-shell `run_dft`. Unrestricted
Kohn–Sham is reachable from Python only inside other drivers:
`run_frequencies(reference="uhf", xc=...)`, `run_u_gw(xc=...)`,
`run_qmmm(method="uks")` and `run_cdft(functional=...)`.

`run_rhf` also takes implicit solvent (`solvent=78.4` or `solvent="water"`,
IEF-PCM), point charges and a uniform field. `run_optimize` and
`run_optimize_uhf` take the same `solvent` and `pcm_lebedev_order` and follow
the analytic IEF-PCM gradient; there is no Python COSMO option, and COSMO has
no gradient. See its docstring
(`help(ferric.run_rhf)`) for the full SCF knob set, which matches the CLI
`[scf]` section.

### Naming the convergence keywords

A bare `max_iter`, `energy_conv` or `density_conv` belongs to a driver whose
method is the SCF itself (`run_rhf`, `run_uhf`, `run_rohf`, `run_dft`,
`run_ksdft`, `run_qmmm`). Every other driver solves a reference SCF as one stage
among others (MP2, CC, RPA, GW, TDDFT, orbital optimization, constrained DFT,
frequencies, geometry optimizers, saddle and IRC searches), so its SCF knobs are
`scf_max_iter`, `scf_energy_conv` and `scf_density_conv`, with the meaning and
defaults of `run_rhf`'s. Another stage's iteration knob is stage-named
(`oo_max_iter`, `max_outer`, `max_steps`, `eigensolver_*`). The bare spellings
that older releases accepted on `run_cdft`, `run_frequencies` and
`run_oo_rimp2` (`max_iter` there) still work, warn with a
`DeprecationWarning`, and raise if combined with the prefixed one. The CLI keeps
each key in its own section (`[scf]`, `[optimize]`, `[gw]`, `[rpa]`, `[mp2] oo_*`).

### Check `.converged`, and know what it means

`run_rhf`, `run_uhf`, `run_rohf` and `run_qmmm` return a result whether or
not the SCF converged. A non-converged result has `converged = False`, and its
energy is a plausible, wrong number.
`run_dft`, `run_ksdft` and the correlated drivers (MP2, CC, RPA, GW, TDDFT)
behave differently: they raise if their reference SCF does not converge.

`converged = True` means the SCF reached a stationary point. It does not mean
the lowest one. The O2 triplet above with `sto-3g` instead of `cc-pvdz` shows
this. `run_uhf` with the default MINAO guess converges to −147.63397 Ha, which
is a saddle of the orbital Hessian 1.33 mHa above the UHF minimum at
−147.635296 Ha (PySCF's default guess lands on the same saddle). Without
`stability_descent` nothing in the result flags it.
`run_uhf(o2, bs, stability_descent=True)` (CLI: `[scf] stability_descent =
true` on `kind = "uhf"`) follows the downhill eigenvector and reaches the
minimum. `guess="hcore"` converges (`converged=True`) to a much
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

`frozen_core=` is accepted by every correlated driver. `run_terfc_rimp2` has
no CLI `method.kind` of its own (the CLI reaches it through `att-rimp2` with
`att_operator = "terfc"`); the reference table below marks the CLI kind of
every driver. `run_rimp2`, `run_drpa` and `run_linlccd` compute their method
exactly unless `local="amplitude-threshold"` and `eps=` are passed (see
[Exact and local correlation](#exact-and-local-correlation)).

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
`(ia|f_xc|jb)` exchange-correlation kernel; VV10 and range-separated
functionals are refused because their kernel is not built. A τ-only meta-GGA
(SCAN, r2SCAN, TPSS) defaults to a 150 × 434 grid; `result.grid` reports the
grid the reference SCF and the kernel shared, and `result.certificate` lists it
as `response-grid`. With no
`functional`, it is CIS (`method="tda"`) or TDHF (`method="casida"`). Both
methods match PySCF `TDA`/`TDDFT` to 1e-3 eV on the systems listed in
[Capabilities and validation](../reference/validation.md#anchors).

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
charge per atom in units of e. `hirshfeld_charges` builds its proatoms from
free-atom SCF densities in the molecule's basis, as the CLI does;
`proatom="slater"` selects the single-exponential Slater proatom instead, which
is 0.23–0.72 e away on H2O, CO and CH3OH. `resp_charges` is a single-stage
restrained fit, not the multi-stage, multi-conformer RESP procedure. `esp_at_atoms` gives the
potential at each nucleus. `hirshfeld_polarizability` returns per-atom 3×3
polarizability tensors (Bohr³) and needs an RI basis.

### Orbitals, density and cube files

`eval_ao_on_points`, `eval_density_on_points` and `eval_mo_on_points` evaluate
basis functions, the total electron density and restricted MOs at an (N, 3)
array of points. **Points and cube grids are in Bohr**, like `esp_at_points`.
`eval_ao_on_points` returns an (N, nbasis) array (PySCF `eval_gto` layout);
`eval_mo_on_points` takes 0-based `mo_indices` into `mo_coefficients` and
returns (N, len(mo_indices)). Orbital signs are the SCF's arbitrary phase.
`write_cube` writes the density (e/Bohr³) or one MO (`what="mo"`, `mo_index=`)
on an axis-aligned grid, from the atoms' bounding box plus `padding` unless
`origin` and `shape` are given, and returns the `(origin, shape)` used.

<!-- doctest -->
```python
import numpy as np

rhf = ferric.run_rhf(water, bs)
pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # Bohr
print(np.round(ferric.eval_density_on_points(water, bs, rhf, pts), 4))

origin, shape = ferric.write_cube("homo.cube", water, bs, rhf,
                                  what="mo", mo_index=4, spacing=0.3)
print(shape)
```

```text
[10.2392  0.8103]
[28, 38, 32]
```

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
arrays. `run_drpa`, `run_linlccd` and `tune_omega` return plain `dict`s, and
`run_drpa_scan` returns a list of them.

AO-basis matrices follow libint2's basis-function conventions, which are not
PySCF's. Measured on CO/cc-pVDZ: total and orbital energies agree with
PySCF to 3e-12 and 8e-9 Ha, and the two AO density matrices differ element by
element by up to 1.5. Compare invariant quantities, not raw AO matrices.

## Building blocks

`ferric.AoIntegrals(mol, basis_set)` exposes the integrals and J/K builds the
drivers use, as numpy arrays, so a method can be written in numpy the way it
is on top of PySCF's `get_hcore` / `get_jk` / `ao2mo`. Bohr and Hartree
throughout; AOs are in ferric's shell order (`shell_offsets()`).

| Block | Returns |
|---|---|
| `overlap()`, `kinetic()`, `nuclear()`, `ecp()`, `hcore()`, `dipole(origin)` | (nao, nao); `hcore()` includes the ECP; `dipole` is (3, nao, nao) |
| `jk(D)` | (J, K) of the TOTAL density, direct and Schwarz-screened: the RHF Fock matrix is `hcore() + J - 0.5 * K` |
| `df_jk(aux)` | a `DfJk` holding the RI-J/RI-K fit, reused for every `.jk(D)` |
| `eri()` | dense (μν\|λσ), chemists' order, refused above `max_gb` (default 2 GiB) |
| `eri3(aux)`, `eri2(aux)` | raw (P\|μν) as (naux, nao, nao) and the metric (P\|Q) |
| `df_mo(aux, C1, C2)` | fitted B^P_pq, (naux, k1, k2), with (pq\|rs) ≈ Σ_P B^P_pq B^P_rs (the RI-MP2 fit) |
| `ao2mo(C1, C2, C3, C4)` | exact (pq\|rs), integral-direct, never holding the nao⁴ AO tensor |
| `grad_*`, `hess_*`, `cphf`, `xc_functional` | nuclear-derivative, response and XC pieces |
| `ferric.einsum(spec, a, b)` | the contraction engine of ferric's correlation codes ([grammar](../reference/einsum.md)) |

Every two-electron method takes `operator="coulomb" | "erf" | "erfc"` with
`omega` in Bohr⁻¹. Orbitals come from the result objects
(`mo_coefficients()`, `orbital_energies()`, `density()`) or from a numpy SCF on
the blocks.

<!-- doctest -->
```python
ints = ferric.AoIntegrals(water, bs)
J, K = ints.jk(D)
F = ints.hcore() + J - 0.5 * K
E = 0.5 * np.sum(D * (ints.hcore() + F)) + ints.nuclear_repulsion()
print(abs(E - rhf.energy) < 1e-8)
B = ints.df_mo(aux, C[:, :5], C[:, 5:])    # (naux, nocc, nvir)
print(B.shape)
```

```text
True
(84, 5, 2)
```

Three methods written this way are in `examples/python/` and are tested
against the Rust drivers: `rhf_from_blocks.py` (DIIS RHF from `hcore()` and
`jk()`, equal to `run_rhf` and PySCF to 1e-8 Ha), `rimp2_from_blocks.py`
(RI-MP2 from `df_mo`, equal to `run_rimp2` and PySCF `DFMP2` to 1e-8 Ha) and
`cis_from_blocks.py` (singlet CIS from `df_mo` or `ao2mo`, equal to
`run_tddft(method="tda")` and PySCF `TDA`).

Not exposed: the Rust-side structs (`PreparedBasis`, `Engine`, screening
bounds), method intermediates (CC amplitudes, GW self-energies, PDEP
eigenpairs) and periodic (PBC) integrals. Those stay behind the drivers.

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
It does not cap total process memory; see
[Sharp bits](./sharp-bits.md#memory-budget_gb-does-not-cap-the-whole-process).

<!-- doctest -->
```python
mp2 = ferric.run_rimp2(water, bs_dz, aux, memory_budget_gb=8.0)
```

`import ferric` runs OpenBLAS at one thread inside ferric's parallel regions
unless `OPENBLAS_NUM_THREADS` is already set; ferric parallelizes with rayon
(`RAYON_NUM_THREADS`). Serial sections such as the per-iteration Fock
diagonalization use `ferric.blas_threads()` OpenBLAS threads, by default the
rayon thread count; `ferric.set_blas_threads(1)` makes results bit-identical
across thread counts. The OpenBLAS thread count is process-wide, so when you
run several jobs at once from Python threads, call `ferric.set_blas_threads(1)`
first: otherwise one job's serial section raises the count under the other
jobs' parallel regions, which oversubscribes the machine and makes their
results vary at the last bit from run to run. `run_rhf`, `run_uhf`,
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
capability is Python-only. How well each one is validated is on
[Capabilities and validation](../reference/validation.md).

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
| `run_dft` | Closed-shell Kohn–Sham DFT (LDA/GGA/hybrid/RSH/meta-GGA by name), optional dispersion (`dispersion="d3bj"` or `"mbd"`, added to the energy and, with `with_gradient=True`, to the gradient) and analytic gradient. | `ksdft` |
| `run_ksdft` | Alias of `run_dft`. | `ksdft` |
| `d3bj_energy` | Grimme D3(BJ) dispersion energy for a molecule and functional, in Hartree; `s9=1.0` adds the ATM three-body term (default 0, off). | `[dft] dispersion` |
| `mbd_rsscs_energy` | MBD@rsSCS dispersion energy (Hartree) from per-atom Hirshfeld volume ratios and β (or a functional with a published β); returns `MbdRsscsResult` with the screened α₀, C6, R_vdW and ω. | — |
| `tune_omega` | IP-based (Baer/Kronik) tuning of an RSH functional's ω (Bohr⁻¹); closed-shell neutral plus doublet cation. `continuation=True` seeds each evaluation from the nearest already-evaluated ω, which keeps the cation on one branch across an out-of-order golden-section search; `branch_tol` flags an evaluation whose cation spin-population asymmetry jumps. `check_cation_stability` (default `True`) runs the UKS internal-stability analysis on every cation: each eval dict carries `cation_lambda_min` and `cation_stability`, and a tuned ω whose cation is an internal saddle raises `RuntimeError` naming ω and λ_min. Functionals without an orbital-Hessian kernel (VV10 such as ωB97X-V, meta-GGA) are reported `not_analysed` with a `stability_warning`, never as stable. | — |
| `dft_grid_point_count` | Number of points in the main KS grid `run_dft` would build for the molecule with the same `grid_*` kwargs, without running an SCF (shows what pruning saves). | — |
| `SymmetricRhfResult` | Result of `run_rhf(symmetry=True)` (point-group-blocked RHF): `energy` (of the symmetrized geometry), `point_group`, `orbital_irreps`, `max_symmetrization_shift`, `irrep_occupations()`, `orbital_energies()`, `symmetrized_coords()`, `frame_transform()`. `density()`, `mo_coefficients()` and `gradient()` raise: the AO matrices live in the rotated frame. | |
| `RhfResult` | Result of `run_rhf`: `energy`, `converged`, `iterations`, `density()`, `orbital_energies()`, `mo_coefficients()`, `one_step_energy_drop` and `stationarity_tol` (the converged exit is stationary when the drop is above `-stationarity_tol`; both `None` for a non-aufbau or unconverged exit). | |
| `UhfResult` | Result of `run_uhf`/`run_rohf`: α and β densities and orbital energies, `one_step_energy_drop`, `stationarity_tol`. | |
| `DftResult` | Result of `run_dft`: `total_energy` (= `e_scf + e_dispersion`), `e_scf`, `e_dispersion` (`None` when not requested), `dispersion_model` (`"D3(BJ)"`, `"MBD@rsSCS"` or `None`), `volume_ratios` (MBD@rsSCS only), `converged`, `exit_reason()`, `one_step_energy_drop`, `stationarity_tol`, `density()`, `gradient()`. | |

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
| `run_optimize` | RHF geometry optimization (basis by name); `solvent` / `pcm_lebedev_order` optimize on the IEF-PCM surface (`run_optimize_uhf` likewise); `scf_warm_start=False` re-guesses every geometry's SCF instead of starting from the previous geometry's orbitals (`[optimize] scf_warm_start`). | `task = "optimize"` |
| `run_frequencies` | Harmonic frequencies from the analytic Hessian (RHF, UHF, ROHF with exact J/K, or RI-J / RI-K) or finite differences of the analytic gradient; RHF/UHF/ROHF or their KS variants. Closed-shell KS and UKS (`reference="uhf"`) (LDA/GGA, pure or global hybrid) use the analytic moving-grid Hessian under the default `hessian="auto"` (explicitly: `hessian="analytic-moving-grid"`); other KS references use finite differences. `scf_warm_start=True` (opt-in) starts every displaced SCF of a finite-difference Hessian from the reference geometry's orbitals (`[frequencies] scf_warm_start`); pair it with a tighter `scf_density_conv`. | `task = "frequencies"` |
| `run_xtb_energy_gradient` | GFN1/GFN2/GFN-FF energy and gradient (Hartree, Hartree/Bohr). Present in every build; raises `RuntimeError` unless the extension was built with the `xtb` cargo feature (libxtb on the loader path). | no CLI kind |
| `run_xtb_optimize` | Geometry optimization on a GFN-xTB surface with the `run_optimize` BFGS driver; `max_steps` bounds geometry steps, `scc_max_iter` the SCC at each geometry, and an unconverged SCC raises. Certificate is `not graded`. | no CLI kind |
| `run_xtb_frequencies` | Harmonic frequencies on a GFN-xTB surface: central differences of the xTB gradient (`6N` evaluations), shared mass-weighting and translation/rotation projection. | no CLI kind |
| `run_saddle` | First-order saddle-point (transition-state) search by P-RFO; closed-shell. `hessian="fd"` (default) / `"analytic"` / `"auto"`; `.hessian_source` reports which ran. `scf_warm_start=False` re-guesses every step's SCF instead of warm-starting it (the FD Hessians are always cold). | — |
| `run_irc` | Intrinsic reaction coordinate in both directions from a saddle; closed-shell. `scf_warm_start=False` re-guesses every point's SCF. | — |
| `OptimizeResult` | `energy`, `converged`, `steps`, `energy_trace`, `mol()`. | |
| `FrequencyResult` | Frequencies in cm⁻¹ (negative = imaginary), normal modes, `cartesian_hessian`, `asymmetry` diagnostic, `hessian_source`. | |
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
| `run_rimp2` | RI-MP2 on an RHF reference; UHF + unrestricted RI-MP2 when `multiplicity > 1` (`reference` says which). Exact by default; `local=`/`eps=` for the local approximation. | `rimp2` |
| `run_oo_rimp2` | Orbital-optimized RI-MP2 (level-shifted Newton + DIIS + Cayley rotation). Closed shell only; open-shell OO-RI-MP2 is CLI-only. | `oo-rimp2` |
| `run_mp3` | MP3 on an RHF reference, with RI integrals. | `mp3` |
| `run_attenuated_rimp2` | RI-MP2 with the erfc-attenuated operator; ω in Å⁻¹, default 0.420. | `att-rimp2` |
| `run_terfc_rimp2` | RI-MP2 with the exact tempered-erfc operator at one cutoff `r0` (Å); needs the terfc tables. | — |
| `run_scs_mp2` | Spin-component-scaled MP2 (defaults c_OS = 6/5, c_SS = 1/3). | `scs-mp2` |
| `run_scs_mp2_2terfc` | Dual-attenuated SCS-MP2(2terfc); needs the terfc tables. | `scs-mp2-2terfc` |
| `run_mp2_v` | MP2-V: attenuated MP2 plus damped VV10 nonlocal correlation. | `mp2-v` |
| `run_double_hybrid` | B2PLYP or DSD-PBEP86 double hybrid; the result carries a `certificate` (Spike). | `b2plyp`, `dsd-pbep86` |
| `run_laplace_mp2` | Laplace-transform RI-MP2 (default 7 quadrature points). | `laplace-mp2` |
| `run_laplace_sos_mp2` | Laplace-transform SOS-MP2, `E = c_os · E_OS`; MO, AO or AO-sparse formulations. | `laplace-sos-mp2` |
| `RiMp2Result` | Result of `run_rimp2` and `run_terfc_rimp2`: `total_energy`, `rhf_energy` (the reference SCF energy, RHF or UHF), `mp2_corr`, `reference`, and `local` (`None` for the exact method, else the local model dict). | |
| `OoRiMp2Result` | Result of `run_oo_rimp2`, with `converged` and `grad_norm`. | |
| `Mp3Result` | `e_hf`, `e_mp2`, `e_mp3`, `e_corr`, `e_total`. | |
| `AttenuatedMp2Result` | Attenuated MP2 total, correlation and spin components. | |
| `ScsMp2Result` | Result of `run_scs_mp2`/`run_scs_mp2_2terfc`, with `e_os`/`e_ss`. | |
| `Mp2VResult` | MP2-V total, attenuated MP2 part and VV10 part. | |
| `LaplaceMp2Result` | Laplace MP2 total, correlation and spin components. | |
| `SosMp2Result` | Scaled and unscaled OS energy, `c_os`, `n_quad` and `formulation` echoed back. | |
| `DoubleHybridResult` | Result of `run_double_hybrid`: `total_energy`, `e_ks`, `e_corr_scaled`, `e_os`, `e_ss`, `c_os`, `c_ss`, `certificate`. | |

`run_rimp2`, `run_oo_rimp2`, `run_attenuated_rimp2`, `run_terfc_rimp2`,
`run_scs_mp2`, `run_scs_mp2_2terfc`, `run_mp2_v`, `run_laplace_mp2`,
`run_laplace_sos_mp2` and `run_rs_mp2_rpa` take `scf_jk="exact"|"rijk"|"rijcosx"`
and `jk_auxbasis=` to choose how the reference SCF builds J and K, and their
results echo `scf_jk` and `jk_auxbasis`. The default is `"exact"`, except for
`run_rs_mp2_rpa`, whose default is `"rijk"`. `run_oo_rimp2` accepts only
`"exact"` and raises on the fitted modes, because it rebuilds exact J/K at
every orbital step. See
[The MP2 family](../methods/mp2.md).

### Exact and local correlation

`run_rimp2`, `run_drpa` and `run_linlccd` name a method and compute it
exactly by default. The local approximation is asked for with keywords
that mirror the CLI's [`[local]`](../reference/input.md#local) section, under
the same rules, raised as `ValueError` before any SCF:

- `local=` is `None` (exact; `"none"` is the same) or `"amplitude-threshold"`;
  any other value is an error.
- `eps=` is required with `local="amplitude-threshold"`: the threshold is
  part of the model and has no default. `eps=0` reproduces the exact method.
  `eps=` without `local=` is an error, not ignored.
- `compute_reference=True` (local only) also computes the exact method and
  reports it: `local["e_corr_canonical_ri"]` (MP2),
  `e_corr_plasmon_canonical` (dRPA), `local["e_corr_exact"]` (LinLCCD).
  Off, the key is present and `None`. It is a full exact calculation, so it
  removes any cost saving.
- `run_rimp2(integral_direct=True, ...)` selects the integral-direct local
  MP2, with its locality maps `aux_radius`, `virt_radius` (Bohr), `ao_tail`,
  `schwarz_skip`, `batch_merge`, `gate_cal` and `virt_schwarz_kappa` (the CLI
  names and defaults). They are errors without `integral_direct=True`.
  `kappa` is an error on the local MP2.

Every result says which model it is: `RiMp2Result.local` and the dicts'
`"local"` key are `None` for the exact method, else a dict with `scheme`,
`eps`, `keep_fraction` and `integral_direct` plus solver counters. All local
paths are closed shell.

| Name | Purpose | CLI |
|---|---|---|
| `run_drpa` | dRPA@HF by the drCCD Riccati solve. Exact by default (equals the plasmon formula); `MemoryError` before the SCF when the exact solve cannot fit (use `run_pdep_rpa(..., trunc_thresh=0)`). `diis=` (default 8); `eps_rtol_factor=` (local only). | `drpa` |
| `run_drpa_scan` | The local dRPA over a list of `eps` values, sharing one SCF and localization; each dict carries `"local"`. | `drpa` + `[local] eps_sweep` |
| `run_linlccd` | Linearized ladder CCD, `variant` = `"hh"` (default), `"drivers-only"`, `"full"`. Exact by default. | `linlccd` |

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
| `run_pdep_rpa` | Direct RPA correlation energy by PDEP (projective dielectric eigenpotentials); accepts point charges, field and solvent. Closed shell only; open-shell U-PDEP-RPA is CLI-only. | `pdep-rpa` |
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
| `TddftResult` | `excitation_energies`, `oscillator_strengths`, `method`, `grid`, `certificate`. | |

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
| `hirshfeld_charges` | Hirshfeld charges. The default `proatom="scf"` uses free-atom SCF densities in the molecule's basis, solved with the result's own SCF settings, as the CLI does; `proatom="slater"` uses a single-exponential Slater proatom. | `[rpa] compute_hirshfeld_charges` |
| `chelpg_charges` | CHELPG charges fitted to the ESP on a grid. | `[rpa] compute_chelpg_charges` |
| `resp_charges` | Single-stage RESP (restrained ESP-fit) charges. | `[rpa] compute_resp_charges` |
| `hirshfeld_polarizability` | Per-atom Hirshfeld-partitioned static polarizability tensors (Bohr³) from PDEP-RPA. | — |
| `orbital_moments` | Per-orbital centroids and spatial spreads (Bohr) of the restricted MOs. | — |
| `density_second_moment` | 3×3 second-moment tensor of the electron density (Bohr²). | — |
| `eval_ao_on_points` | AO values at (N, 3) points in Bohr, shape (N, nbasis). | — |
| `eval_density_on_points` | Total electron density (e/Bohr³) at points in Bohr. | — |
| `eval_mo_on_points` | Restricted MO values at points in Bohr, for chosen 0-based MO indices. | — |
| `write_cube` | Gaussian cube file of the density or one restricted MO; all lengths in Bohr. | — |

The `[rpa] compute_*` keys write into the CLI's NPZ export.

### Integrals and orbitals (prototyping)

These take `omega`/`r0` in raw Bohr units, unlike the `run_*` drivers.

| Name | Purpose | CLI |
|---|---|---|
| `AoIntegrals` | One-electron matrices, ERIs (dense, 3- and 2-index, MO-transformed), J/K, gradient/Hessian/response/XC pieces for one molecule and basis; see [Building blocks](#building-blocks). | — |
| `DfJk` | From `AoIntegrals.df_jk(aux)`: RI-J/RI-K `j(D)`, `k(D)`, `jk(D)` over a fit built once. | — |
| `XcFunctional` | From `AoIntegrals.xc_functional(name)`: E_xc, V_xc, f_xc response and XC gradient/Hessian on `run_dft`'s grid. | — |
| `Cphf`, `Ucphf` | From `AoIntegrals.cphf(...)` / `ucphf(...)`: orbital-Hessian `matvec`, CPHF/CPKS `solve`, response `density`. | — |
| `einsum` | Two-operand contraction under ferric's einsum spec grammar. | — |
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
