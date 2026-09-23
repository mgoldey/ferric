# Your first calculation

This page takes you from an installed `ferric` to one checked number, then shows
where to go next. It assumes you have run `pip install ferric`
([Installation](./installation.md)). No git clone is needed until the last
section.

We compute the Hartree–Fock energy of water in the minimal STO-3G basis. It
takes well under a second and has a known answer, so you can tell immediately
whether your installation is right.

## 1. From Python

Save this as `water.py`:

```python
import ferric

# XYZ format: atom count, a comment line, then symbol x y z in Ångström.
water = ferric.Molecule.from_xyz_string("""3
water
O   0.000000   0.000000   0.117790
H   0.000000   0.755453  -0.471161
H   0.000000  -0.755453  -0.471161
""", 0, 1)                       # charge 0, spin multiplicity 1 (singlet)

basis = ferric.BasisSet.bundled("sto-3g")
rhf = ferric.run_rhf(water, basis)

print(rhf.converged, f"{rhf.energy:.10f}")
```

Run it:

```bash
OPENBLAS_NUM_THREADS=1 python water.py
```

You should see:

```text
True -74.9631468000
```

Two things to notice:

- **Always read `converged`.** When `run_rhf` runs out of iterations it still
  returns an energy; it sets the flag and does not raise. A number with
  `converged == False` is not a result. See [Sharp bits](./sharp-bits.md).
- **Energies are in Hartree.** Geometries go in as Ångström; see
  [Sharp bits](./sharp-bits.md) for which accessors return Bohr.

## 2. The same thing from the command line

The wheel also installs a `ferric` command that reads a TOML input file. Save
the three atom lines above, with their two header lines, as `water.xyz`, and
write `water-rhf.toml` next to it:

```toml
[molecule]
xyz = "water.xyz"      # resolved relative to the directory you run ferric from

[basis]
name = "sto-3g"

[method]
kind = "rhf"
```

```bash
OPENBLAS_NUM_THREADS=1 ferric water-rhf.toml
```

The output ends with:

```text
RHF/sto-3g on water.xyz
  nbasis     = 7
  iterations = 8
  converged  = true
  energy     = -74.9631468000 Hartree
```

Same molecule, same number. The CLI rejects any key it does not recognise, so
a typo is an error rather than a silently ignored setting. Every key is listed
in the [input reference](../reference/input.md).

## 3. Add correlation

Change one line in the TOML, or one call in Python, to go beyond Hartree–Fock.
RI-MP2 needs an auxiliary (fitting) basis alongside the orbital basis:

```python
bs  = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")
mp2 = ferric.run_rimp2(water, bs, aux)
print(f"RI-MP2 total energy: {mp2.total_energy:.10f} Ha")
```

```text
RI-MP2 total energy: -76.2308014550 Ha
```

In TOML the same calculation is `kind = "rimp2"` with `auxbasis` in the `[mp2]`
section; see [`examples/water-rimp2.toml`](https://github.com/mgoldey/ferric/blob/main/examples/water-rimp2.toml).

## 4. Where to next

| You want to | Go to |
|---|---|
| Pick a method for a chemistry question | [Choosing a method](./choosing-a-method.md) |
| See what every method supports (open shell? gradients? CLI?) | [Capabilities](../reference/capabilities.md) |
| Charged or open-shell molecules, geometry optimization, SMILES input | [Recipes](./recipes.md) |
| The whole Python surface | [Python bindings](./python.md) |
| Coming from PySCF | [For PySCF users](./pyscf-users.md) |
| Know what to trust | [What is validated](../reference/validation.md) |

### Running the bundled examples

The repository has one input file per method under `examples/`, indexed in
[Examples](../reference/examples.md). They refer to molecules under
`testdata/`, which the wheel does not include, so run them from a clone:

```bash
git clone https://github.com/mgoldey/ferric && cd ferric
OPENBLAS_NUM_THREADS=1 ferric examples/water-rimp2.toml
OPENBLAS_NUM_THREADS=1 ferric examples/water-attmp2.toml   # attenuated MP2, ω = 0.420 Å⁻¹
```

If you built from source rather than installing the wheel, replace `ferric` with
`cargo run --release --bin ferric --`.
