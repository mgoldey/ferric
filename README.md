# ferric

A quantum chemistry engine written in Rust, with Python bindings and a
TOML-driven command line. libint2 supplies the integrals and libxc the
functionals.

**Documentation: <https://matthew.thegoldeys.com/ferric/>**

## Install

```bash
pip install ferric      # Linux x86_64, CPython 3.10–3.13
```

The wheel needs no compiler. To change ferric itself or to use MPI, build from
source; see [Installation](https://matthew.thegoldeys.com/ferric/using/installation.html).

## A first calculation

<!-- doctest -->
```python
import ferric

water = ferric.Molecule.from_xyz_string(
    """3
water
O   0.000000   0.000000   0.117790
H   0.000000   0.755453  -0.471161
H   0.000000  -0.755453  -0.471161
""",
    0,
    1,
)  # charge, multiplicity

rhf = ferric.run_rhf(water, ferric.BasisSet.bundled("sto-3g"))
print(rhf.converged, f"{rhf.energy:.10f}")  # True -74.9631468000
```

The same calculation from the command line (`ferric` is installed with the
wheel):

```toml
# water-rhf.toml
[molecule]
xyz = "water.xyz"

[basis]
name = "sto-3g"

[method]
kind = "rhf"
```

```bash
OPENBLAS_NUM_THREADS=1 ferric water-rhf.toml
```

Continue with [Your first calculation](https://matthew.thegoldeys.com/ferric/using/quickstart.html)
and the [Sharp bits](https://matthew.thegoldeys.com/ferric/using/sharp-bits.html).

## What it does

| Family | Examples | Page |
|---|---|---|
| SCF and DFT | RHF/UHF/ROHF, RKS/UKS/ROKS via libxc (LDA to meta-GGA, range-separated hybrids, VV10, D3(BJ)), analytic gradients, optimization, finite-difference frequencies, PCM/COSMO solvation | [SCF and DFT](https://matthew.thegoldeys.com/ferric/methods/scf.html) |
| MP2 family | RI-MP2, attenuated MP2, SCS-MP2 and SCS-MP2(2terfc), OO-MP2, Laplace and SOS-MP2, MP3, local MP2, MP2-V | [The MP2 family](https://matthew.thegoldeys.com/ferric/methods/mp2.html) |
| Coupled cluster | CCD, CCSD, CCSD(T), LinLCCD, double hybrids | [Coupled cluster](https://matthew.thegoldeys.com/ferric/methods/cc.html) |
| Response | PDEP-RPA, GW (G0W0, COHSEX, evGW), BSE-TDA, TDDFT/TDA, polarizabilities and C6 | [RPA and GW](https://matthew.thegoldeys.com/ferric/methods/rpa-gw.html) |
| Constrained DFT | charge/spin constraints, electron-transfer couplings | [Constrained DFT](https://matthew.thegoldeys.com/ferric/methods/cdft.html) |
| Embedding | QM/MM with link atoms, smeared and polarizable charges | [QM/MM](https://matthew.thegoldeys.com/ferric/using/qmmm.html) |

Which of these run from the CLI, which are Python-only, and which support open
shells or gradients is tabulated in
[Capabilities](https://matthew.thegoldeys.com/ferric/reference/capabilities.html).

> **Implemented ≠ validated.** Working code is not a checked number. Each
> capability is graded (Proven / Smoke / Spike) against independent references
> in [What is validated](https://matthew.thegoldeys.com/ferric/reference/validation.html).
> Read it before relying on a result.

## Why it exists

ferric is organized around **electronic response**: how the density reacts to a
perturbation, the object behind polarizabilities, dielectric screening and
dispersion. Attenuated MP2, PDEP-RPA/GW and constrained DFT are three ways of
getting that object right where standard methods get it wrong. See
[Electronic response](https://matthew.thegoldeys.com/ferric/idea/response.html).

## Building from source and contributing

```bash
git clone https://github.com/mgoldey/ferric && cd ferric
# libint2 first: see the Installation page (about 30 minutes)
cargo build --release
OPENBLAS_NUM_THREADS=1 ./target/release/ferric examples/water-rhf.toml
OPENBLAS_NUM_THREADS=1 cargo test --workspace
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development loop, the Python
binding tests and the push gate.

## Citing

If you use ferric, cite the software (see [CITATION.cff](CITATION.cff)) and the
papers for the methods you ran, listed in
[References and citing](https://matthew.thegoldeys.com/ferric/reference/references.html).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Third-party code

`crates/ferric-integrals/shim/libecpint/` vendors [libecpint](https://github.com/robashaw/libecpint)
(MIT, Copyright (c) 2021 Robert A. Shaw); its license is retained at
`crates/ferric-integrals/shim/libecpint/LICENSE`.

ferric links against external libraries with their own licenses, notably
libint2 (LGPL-3.0-or-later), libxc (MPL-2.0), OpenBLAS/LAPACK (BSD), and
optionally xtb (LGPL-3.0), which govern redistribution of binaries built
against them. libint2 is a required dependency, not an optional one:
distributing a compiled ferric binary means distributing LGPL-linked code, which
carries obligations (relinking, source availability) beyond ferric's own MIT/
Apache-2.0 terms. Building from source for your own use is unaffected.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
