# ferric

A quantum chemistry engine written in Rust, with Python bindings and a
TOML-driven command line. It computes Hartree–Fock and DFT energies and
gradients, the MP2 family, coupled cluster, RPA and GW, and constrained DFT,
with libint2 for the integrals.

```bash
pip install ferric      # Linux x86_64, Python 3.10–3.13
```

## Start here

<div class="routes">

**[Get started →](./using/quickstart.md)**
Install the wheel and compute a checked number in under a minute. Then read
the [sharp bits](./using/sharp-bits.md) that catch new users.

**[How-to guides →](./using/recipes.md)**
Task recipes: charged and open-shell molecules, optimization, SMILES input,
QM/MM, ligand pipelines. [Choosing a method](./using/choosing-a-method.md)
maps a chemistry question to a method.

**[Methods →](./methods/index.md)**
What each method family is, how to run it, how accurate it is, what it costs
and what to cite.

**[Reference →](./reference/validation.md)**
[Capabilities and validation](./reference/validation.md) ·
[Input file](./reference/input.md) ·
[Python API](./using/python.md) ·
[Examples](./reference/examples.md) ·
[Rust API](./reference/api.md)

</div>

| If you are | Start with |
|---|---|
| A chemist who wants numbers | [Your first calculation](./using/quickstart.md), then [Choosing a method](./using/choosing-a-method.md) |
| Coming from PySCF | [For PySCF users](./using/pyscf-users.md) |
| Working on drug-discovery workflows | [End-to-end applications](./using/applications.md) and [QM/MM](./using/qmmm.md) |
| A method developer | [Electronic response](./idea/response.md), [Architecture](./reference/architecture.md), [Rust API](./reference/api.md) |
| An automated agent | [For automated agents](./using/agents.md) |

## Implemented ≠ validated

Working code is not a checked number. These pages describe what exists; how
far each capability's numbers have been checked against an independent
reference differs a lot between methods. Each one is graded individually in
**[Capabilities and validation](./reference/validation.md)**. Read it before relying on a
result.

## The idea

`ferric` is organized around **electronic response**: how the density reacts to
a perturbation. That object appears as the polarizability \\( \alpha \\), the
dielectric function \\( \varepsilon \\) and the susceptibility \\( \chi \\), and
three of the method families here are different ways of getting it right where
standard methods get it wrong:

- **[Attenuated MP2](./methods/mp2.md)**: MP2 builds dispersion from an
  uncoupled polarizability, which overbinds some systems (π-stacking is the
  classic case). Attenuating the correlation operator removes the long-range
  part where that error lives.
- **[PDEP-RPA / GW](./methods/rpa-gw.md)**: the dielectric matrix is the
  density–density response. PDEP keeps only its dominant eigenmodes, a low-rank
  representation of the screening used by RPA correlation and the GW
  interaction.
- **[Constrained DFT](./methods/cdft.md)**: a constraint couples to the density
  and reads its response, building charge-localized diabatic states and their
  electron-transfer couplings.

[Electronic response](./idea/response.md) develops the argument and says which
parts of it are demonstrated and which remain a design premise.

## Source, license, citation

[github.com/mgoldey/ferric](https://github.com/mgoldey/ferric), dual-licensed
MIT / Apache-2.0. To cite ferric and the methods you used, see
[References and citing](./reference/references.md#citing-ferric).
