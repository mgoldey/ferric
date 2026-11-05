# ferric-cli

The `ferric` command-line driver. A calculation is one TOML file:

```
cargo run --release -- examples/water-rhf.toml
```

Every section and key is specified in the book's
[Input reference](../../site/src/reference/input.md)
(`site/src/reference/input.md`), which a test
(`config_doc_tests.rs`) keeps in step with the config structs in
`crates/ferric-cli/src/config.rs`. This README does not repeat those tables.

Unknown keys are a hard error. Every config struct is
`#[serde(deny_unknown_fields)]`, so a misspelled key or section aborts the run
before anything is computed. Every ```` ```toml ```` block in this README is parsed
against the real `Config` by the same test.

A minimal file:

```toml
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rimp2"
[mp2]
auxbasis = "cc-pvdz-ri"
```

---

## `[method]`

`kind` is required; `task` defaults to `"energy"`.

| Key | Allowed values |
| :--- | :--- |
| `kind` | `rhf` `uhf` `rohf` `ksdft` `rimp2` `mp3` `oo-rimp2` `att-rimp2` `mp2-v` `scs-mp2` `scs-mp2-2terfc` `laplace-mp2` `laplace-sos-mp2` `pdep-rpa` `rs-mp2-rpa` `gw` `bse-tda` `tdhf-static-polarizability` `ccsd` `ccd` `ccsd(t)` `linlccd` `drpa` `wb97x-l-v` `b2plyp` `dsd-pbep86` `tda` `tddft` |
| `task` | `energy` `optimize` `frequencies` |

* Density-functional calculations use `kind = "ksdft"` (or `rhf`/`uhf`/`rohf`
  with `[dft] functional` set); there is no `kind = "dft"`.
* There is no `task = "gradient"`. `optimize` runs for `rhf` `ksdft` `uhf`
  `rohf` `rimp2` `pdep-rpa`; `frequencies` for `rhf` `ksdft` `uhf` `rohf`.
* `rimp2`, `drpa` and `linlccd` are exact by default; a local approximation is
  the `[local]` section, never a separate kind.
* Any other `kind` or `task` is an error that lists the accepted values.

## Sections

All keys, types, defaults and allowed values for each table are in the
[Input reference](../../site/src/reference/input.md).

| Section | Purpose |
| :--- | :--- |
| `[molecule]` `[basis]` `[method]` | Required: geometry, basis, method and task. |
| `[scf]` `[[scf.ladder]]` | SCF convergence, exchange builder (`k_builder`), density fitting, convergence ladder. |
| `[dft]` | Functional, XC grid (`grid_radial`, `grid_angular`, `grid_prune`, `grid_weight_cut`), dispersion. |
| `[mp2]` `[cc]` `[local]` | MP2 family, coupled cluster, local-correlation approximation. |
| `[rpa]` `[gw]` `[tddft]` | PDEP-RPA and property export, GW/BSE, TDA/TDDFT. |
| `[optimize]` `[frequencies]` | Geometry optimizer, harmonic frequencies. |
| `[memory]` `[gpu]` `[precision]` `[output]` | Resources, devices, mixed precision, JSON run log. |
| `[qmmm]` `[pcm]` `[cosmo]` `[external_potential]` `[cell]` | Embedding, implicit solvent, point charges, periodic cell. |

### XC grid

A Kohn-Sham run with no grid key set uses the production grid: 75 radial x 110
angular points per atom, pruned with `grid_prune = "nwchem-radial"`, with a
`1e-14` weight cutoff. Setting only `grid_radial`/`grid_angular` keeps that
scheme and cutoff at the new size. The flat grid is `grid_prune = "none"` with
`grid_weight_cut = 0`:

```toml
[dft]
functional = "PBE"
grid_radial = 99
grid_angular = 302
grid_prune = "none"
grid_weight_cut = 0.0
```

---

## Frozen core

`frozen_core` appears in `[mp2]` (shared by the whole MP2 family and by the CC
and double-hybrid methods), in `[rpa]`, and in `[gw]`. It takes either an
explicit orbital count or `"auto"`:

```toml
[mp2]
frozen_core = "auto"   # standard small-core count for this molecule
# frozen_core = 3      # exactly three orbitals, whatever the molecule is
# frozen_core = "none" # correlate everything (the default, same as 0)
```

`true` and `false` are accepted as synonyms of `"auto"` and `"none"`.
Anything else (`"fc"`, `"core"`, a negative number) is a parse error, never
a silent fall-through to 0.

The default is 0 (all-electron), not `"auto"`. Freezing the core changes
the correlation energy, so it stays opt-in: an input file that omits
`frozen_core` correlates all electrons.

What `"auto"` counts: one orbital per core shell below the element's
valence shell:

| Elements | Core orbitals | Shells frozen |
|----------|---------------|---------------|
| H–He     | 0             | — |
| Li–Ne    | 1             | 1s |
| Na–Ar    | 5             | +2s2p |
| K–Zn     | 9             | +3s3p |
| Ga–Kr    | 14            | +3d |
| Rb–Cd    | 18            | +4s4p |
| In–Xe    | 23            | +4d |

Psi4, ORCA and Q-Chem apply the same small-core convention. The (n−1)d
shell stays correlated across its own transition series and becomes core
only once the following p-block starts (3d is valence for Sc–Zn, core from Ga
on). The d electrons of a transition metal are chemically active, and freezing
them is a large, uncontrolled error.

Two adjustments apply:

* **Ghost atoms count 0.** A basis-only center has no electrons, so it has no
  core to freeze.
* **ECP atoms are counted on what is left.** An effective core potential has
  already removed core orbitals from the MO space, so only the remainder is
  frozen. For iodine under a def2-ECP (`n_core = 28`, i.e. 14 orbitals),
  23 − 14 = 9 orbitals (4s4p4d).

A run that resolves an `"auto"` prints the number it picked:

```
[ferric] frozen core: 3 orbital(s) frozen from [mp2] frozen_core = "auto"
```

A `frozen_core` that would freeze every occupied orbital of the minority spin
(leaving nothing to correlate) is an error before the SCF starts, naming the
section that set it.

`[gw] frozen_core` may be left unset, in which case it follows
`[rpa] frozen_core`; W and Σ must use the same frozen core for
self-consistency. Setting it to `"none"` is not the same as omitting it.
