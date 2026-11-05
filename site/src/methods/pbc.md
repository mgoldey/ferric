# Periodic systems

ferric runs 3-D periodic Gaussian-basis calculations through the
`ferric-pbc` crate. Conventions follow PySCF `pyscf.pbc`: Ewald nuclear
repulsion with a neutralising background, the dropped-G = 0 Coulomb
convention, and the Madelung exchange correction (`exxdiv = "ewald"`).
Energies are Hartree per cell. The equations, with code and test
cross-references, are in `crates/ferric-pbc/theory.md`; the identities are
checked by `scripts/pbc_ewald_pairft_stress_sympy.py`.

## What runs

| | Gamma point | k-point mesh (`kmesh`) |
|---|---|---|
| SCF | RHF, UHF, ROHF | RHF, UHF, ROHF |
| Kohn-Sham | RKS, UKS, ROKS: LDA, GGA, global hybrids; range-separated (CAM) GGA hybrids; SCAN / r2SCAN / TPSS on the routes listed in [the input reference](../reference/input.md#cell) | RKS, UKS, ROKS with the same families (per-route functional limits in the input reference) |
| Correlation | MP2, dRPA on the RHF reference (open-shell UMP2 / URPA library-only) | MP2, dRPA on the RHF reference |
| Nuclear forces | analytic, every SCF route; CLI `task = "optimize"` (atoms move, lattice fixed) | analytic; CLI `task = "optimize"` |
| Stress | analytic, Python `with_stress=True` | analytic, Python `with_stress=True` and CLI `[cell] stress = true` |
| ECPs | yes | yes |

Meshes are Gamma-centred (`centring = "gamma"`) or shifted Monkhorst-Pack
(`"mp"`). The exchange divergence uses the Gamma Madelung constant, or for a
mesh that of the `diag(N)` supercell.

## Two-electron builders

- `jk = "dense"` (default): a dense pure-AFT `nao⁴` ERI tensor. It is a
  correctness oracle for toy cells, hard-capped by `max_eri_gb` (default
  0.5 GiB; range-separated hybrids need 40 `nao⁴` bytes).
- `jk = "rsgdf"`: range-separated Gaussian density fitting (PySCF RS-GDF
  conventions). Requires an auxiliary basis (`auxbasis`, e.g.
  `"cc-pvdz-ri"`) and is bounded by the unified memory budget
  (`[memory] budget_gb` / `memory_budget_gb`). Options: `gdf_omega` (the fit's
  Ewald split), `range_split` (energy only), `sr_column_rotation`.

The nuclear attraction is split at its own `omega` (default
`min(2.5 √π/Ω^{1/3}, ω_cap)`); any value gives the same energy up to
truncation.

## Entry points

**CLI.** Add a `[cell]` section (`lattice`, `unit`, optional `kmesh`,
`exxdiv`, `jk`, …) to an input file; `method.kind` is `rhf`, `uhf`, `rohf`,
`ksdft`, `rimp2` (periodic MP2) or `pdep-rpa` (periodic dRPA). Full key list:
[Input file, `[cell]`](../reference/input.md#cell). Examples:
`examples/h2-cell-rhf.toml`, `examples/h2-cell-kpts-rhf.toml`,
`examples/h2-cell-kpts-rhf-stress.toml`, `examples/h2-cell-rks-opt.toml`.

**Python.** Coordinates and lattice rows in Å, `omega`/`gdf_omega` in Å⁻¹:

- Gamma point: `run_rhf_gamma`, `run_uhf_gamma`, `run_rohf_gamma`,
  `run_rks_gamma`, `run_uks_gamma`, `run_roks_gamma`, `run_mp2_gamma`,
  `run_drpa_gamma`.
- k-point mesh: `run_rhf_kpts`, `run_uhf_kpts`, `run_rohf_kpts`,
  `run_rks_kpts`, `run_uks_kpts`, `run_roks_kpts`, `run_mp2_kpts`,
  `run_drpa_kpts`.
- `auto_gdf_omega` chooses an RS-GDF split.

SCF drivers take `with_gradient` / `with_stress`; results expose
`energy`, `e_nuc` (Ewald), `madelung`, `gradient()` (Hartree/Bohr per cell)
and `stress()` (`σ = (1/Ω) dE/dε`, Hartree/Bohr³, not symmetrised; pressure
`−tr σ/3`). Signatures and refusals are in `ferric.pyi`.

```python
import ferric
mol = ferric.Molecule.from_xyz("testdata/molecules/h2.xyz")
a = 4.0 * 0.529177210903  # 4 Bohr cube, in Angstrom
res = ferric.run_rhf_gamma(
    mol, [[a, 0, 0], [0, a, 0], [0, 0, a]],
    ferric.BasisSet.bundled("sto-3g"), exxdiv="ewald", with_stress=True,
)
print(res.energy, res.stress())
```

## Limits

- 3-D periodicity only; no slabs, wires or 2-D Coulomb truncation.
- Charged cells are an error.
- No lattice-vector relaxation: `task = "optimize"` moves atoms at a fixed
  lattice. `task = "frequencies"` is an error.
- Gamma-point stress is Python-only; the CLI `stress = true` requires
  `kmesh`.
- The dense AFT builder is toy-scale by construction (memory `8 nao⁴` bytes,
  G count growing as `Ω p_max^{3/2}`); production-size cells need
  `jk = "rsgdf"`.
- VV10 is refused on every periodic derivative path; per-functional and
  per-route support for forces and stress is listed in
  [the input reference](../reference/input.md#cell) and
  [Capabilities and validation](../reference/validation.md).
- Gamma-point local MP2 is library-only and its default pair gate keeps
  every pair below a volume onset.
