# For PySCF users

ferric's Python API will look familiar if you use PySCF, but it is built
differently. This page maps the PySCF calls you already know onto ferric's,
then lists the differences that change numbers or cause errors.

The table only lists pairs where both sides exist. If a PySCF feature has no
row, assume ferric does not have it; check the
[full function reference](./python.md#full-reference) to be sure. The PySCF
names were checked against PySCF 2.12.1.

## Rosetta table

Assume `import ferric` and, on the PySCF side, the matching `from pyscf import ...`
(`gto`, `scf`, `dft`, `mp`, `cc`, `tdscf`, `gw`, `lo`, `df`, `qmmm`, `solvent`,
`geomopt`, `hessian`).

### Molecule and basis

| PySCF | ferric |
|---|---|
| `mol = gto.M(atom=..., basis="cc-pvdz", charge=0, spin=2)` | `mol = ferric.Molecule.from_xyz_string(xyz, charge=0, multiplicity=3)` and `bs = ferric.BasisSet.bundled("cc-pvdz")` |
| `mol.energy_nuc()` | `mol.nuclear_repulsion()` |
| `mol.nelectron` | `mol.nelec()` |
| `mol.natm` | `mol.natoms()` |
| `mol.atom_coords()` (Bohr) | `mol.coords_bohr()` |
| `mol.atom_coords(unit="Angstrom")` | `mol.coords()` |
| `mol.elements` | `mol.symbols()` |
| `mol.atom_charges()` | `mol.atomic_numbers()` (true Z, not reduced by an ECP) |
| ghost atom `"ghost-H"` or `"X-H"` | `@H` in the XYZ text |

### SCF and DFT

| PySCF | ferric |
|---|---|
| `mf = scf.RHF(mol).run()` | `r = ferric.run_rhf(mol, bs)` |
| `scf.UHF(mol).run()` | `ferric.run_uhf(mol, bs)` (spin from `mol`) |
| `scf.ROHF(mol).run()` | `ferric.run_rohf(mol, bs)` |
| `scf.RHF(mol).density_fit(auxbasis="def2-universal-jkfit")` | `ferric.run_rhf(mol, bs, df_j_aux="def2-universal-jkfit", df_k_aux="def2-universal-jkfit")` |
| `mf = dft.RKS(mol); mf.xc = "b3lyp"; mf.run()` | `ferric.run_dft(mol, bs, functional="b3lyp")` (closed-shell only) |
| `mf.disp = "d3bj"` | `ferric.run_dft(..., dispersion="d3bj")`, or `ferric.d3bj_energy(mol, "b3lyp")` alone |
| `mf.e_tot` | `r.energy` (RHF/UHF/ROHF) or `r.total_energy` (DFT) |
| `mf.converged` | `r.converged` |
| `mf.mo_energy` | `r.orbital_energies()` (RHF), `r.orbital_energies_alpha()` / `_beta()` (UHF/ROHF) |
| `mf.mo_coeff` | `r.mo_coefficients()` (RHF only) |
| `mf.make_rdm1()` | `r.density()` (RHF, DFT), `r.density_alpha()` / `r.density_beta()` (UHF/ROHF) |
| `mf.level_shift = 0.2` | `run_rhf(..., level_shift=0.2)` |
| `mf = solvent.PCM(scf.RHF(mol)); mf.with_solvent.eps = 78.4` | `ferric.run_rhf(mol, bs, solvent=78.4)` or `solvent="water"` |
| `qmmm.mm_charge(mf, coords, charges)` | `run_rhf(..., point_charges=[(q, x, y, z), ...])`, coordinates in **Bohr** |

### Geometry and vibrations

| PySCF | ferric |
|---|---|
| `geomopt.geometric_solver.optimize(mf)` | `ferric.run_optimize(mol, "cc-pvdz")` (RHF; basis by name) |
| `mf.Hessian().kernel()` then `thermo.harmonic_analysis(mol, h)` | `ferric.run_frequencies(mol, "cc-pvdz")` |

### Correlation, response and properties

| PySCF | ferric |
|---|---|
| `mp.MP2(mf).density_fit(auxbasis="cc-pvdz-ri").run()` | `ferric.run_rimp2(mol, bs, ferric.BasisSet.bundled("cc-pvdz-ri"))` |
| `cc.CCSD(mf).density_fit(auxbasis=...).run()` | `ferric.run_ccsd(mol, bs, aux)` |
| `mycc.ccsd_t()` | `ferric.run_ccsd_t(mol, bs, aux).t_correction` |
| `tdscf.TDA(mf).run()` | `ferric.run_tddft(mol, bs, aux, method="tda")` |
| `tdscf.TDHF(mf).run()` / `tdscf.TDDFT(mf).run()` | `ferric.run_tddft(mol, bs, aux, method="casida")` |
| `gw.GW(mf).kernel()` | `ferric.run_gw(mol, bs, aux)` |
| `lo.Boys(mol, mo).kernel()` | `ferric.boys_localize(mol, bs, mo).c_loc()` |
| `mf.mulliken_pop()` | `ferric.mulliken_charges(mol, bs, r)` |
| `df.incore.aux_e2(mol, auxbasis)` | `ferric.compute_eri3(mol, bs, aux)` |
| `df.incore.fill_2c2e(mol, auxbasis)` | `ferric.compute_metric_2c(mol, bs, aux)` |

## How ferric differs

**Stateless calls, not mutable method objects.** PySCF builds an `mf`
object, lets you set attributes on it, then runs it. ferric has one function
per calculation. Settings are keyword arguments, and the call returns a result
object. There is nothing to reconfigure and rerun; call the function again.
Correlated drivers run their own reference SCF, so you pass them the molecule
and basis, not a converged `mf`.

**Spin is `multiplicity`, and it lives on the molecule.** PySCF's `spin` is
2S; ferric's `multiplicity` is 2S+1. It is set when the molecule is built
(`from_xyz_string(..., multiplicity=3)`). `run_uhf` and `run_rohf` have no spin
keyword.

**The auxiliary basis is always explicit.** PySCF picks an auxiliary basis for
you when you call `.density_fit()` without one. ferric's MP2, CC, RPA, GW and
TDDFT drivers take the RI basis as a required argument. Only SCF-level
fitting has a default: `run_dft` (and the reference SCF inside
`run_rs_mp2_rpa`) fit with `def2-universal-jkfit` unless told otherwise.

**Density fitting is on in `run_dft` and off in `run_rhf`.** `run_dft` uses
RI-J by default, and RI-K for hybrids. `dft.RKS` in PySCF uses exact Coulomb
unless you call `.density_fit()`. `run_rhf` builds exact four-centre J and K
unless you pass `df_j_aux`/`df_k_aux`. The difference between a fitted and an
exact energy is the fitting error, not a bug; the ferric source records it at
PBE/STO-3G against conventional J as 0.28 (water), 1.16 (benzene) and
9.5 kcal/mol (a 71-atom drug molecule). Pass `df_j_aux="exact"` to `run_dft`
when you compare against exact-Coulomb PySCF.

**The DFT grid is smaller and not pruned.** ferric's default is 75 radial ×
110 Lebedev angular points on every atom, with no pruning. PySCF's default
(`grids.level = 3`) uses 302 angular points for H through Ne and 434 from Na
on, with 50 radial points for H and 75 for C–Ne, and prunes with
`nwchem_prune`. Expect grid-level differences in DFT energies.

**Correlated methods use RI integrals throughout.** ferric's CCSD and CCSD(T)
build their integrals from the auxiliary basis you pass. The fair PySCF
comparison is `cc.CCSD(mf).density_fit(...)`, not plain `cc.CCSD(mf)`, which
uses exact four-index integrals.

**Range-separation ω is in Å⁻¹ in the `run_*` drivers.** `omega` on
`run_attenuated_rimp2` and `run_rs_mp2_rpa` is in Å⁻¹ (default 0.420). PySCF,
and ferric's low-level `compute_eri3_mo` and `compute_metric_2c`, take Bohr⁻¹.

**Point charges are in Bohr; `QmmmSystem` coordinates are in Å.**
`point_charges=` on `run_rhf` and the other SCF drivers takes Bohr. PySCF's
`qmmm.mm_charge` defaults to `mol.unit`, which is Ångström unless you changed
it. `ferric.QmmmSystem` takes Ångström coordinates, and its `point_charges()`
accessor returns Bohr, ready to pass to `run_rhf`.

**An unconverged SCF can raise.** PySCF's `kernel()` returns the energy
either way. ferric's `run_rhf`, `run_uhf`, `run_rohf` and `run_qmmm` do the
same, so check `.converged`. `run_dft` and every correlated driver raise
instead when their SCF does not converge.

**TDDFT has no exchange-correlation kernel.** PySCF's `TDDFT` on a DFT
reference includes the f_xc response. ferric's `run_tddft` does not implement
it, so on a DFT reference its excitation energies are approximate (it warns on
stderr). On a Hartree–Fock reference (CIS/TDHF) it is exact within the method.

**GW defaults to a Hartree–Fock reference.** PySCF's `gw.GW(mf)` runs on
whatever `mf` you pass, usually a DFT one. `ferric.run_gw` runs its own HF
reference unless you pass `xc="pbe"` or another functional.

**The PCM defaults differ.** ferric's `solvent=` is IEF-PCM with 110
tesserae per atomic sphere (`pcm_lebedev_order=110`). PySCF's `PCM` defaults
to C-PCM with Lebedev order 29 (302 points per sphere) and SWIG
discretization. The named solvent `"water"` is ε = 78.4 in ferric;
PySCF's default ε is 78.3553.

**AO matrices do not match element by element.** ferric's AO basis-function
conventions come from libint2 and are not the same as PySCF's (libcint).
MEASURED on CO/cc-pVDZ: RHF total energies agree to 3e-12 Ha and orbital
energies to 8e-9 Ha, while the two AO density matrices differ element by
element by up to 1.5. Compare invariant quantities (energies, orbital
energies, charges), not raw AO matrices. Axis order differs too:
`compute_eri3` returns `(naux, n_bf, n_bf)`, with the auxiliary index first.

**Basis sets come from a fixed bundled list.** `BasisSet.bundled(name)` knows
25 names (see [Bundled basis sets](./python.md#bundled-basis-sets)). There is
no Python loader for a basis file or for a per-element basis dictionary.

**Results come back as numpy arrays or lists, not live objects.** Matrices are
`numpy.ndarray`. Per-atom quantities such as charges are Python lists.
