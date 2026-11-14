# ferric

A quantum chemistry engine written in Rust, with Python bindings and a
TOML-driven command line, using libint2 for the integrals. It computes:

- energies and analytic gradients with Hartree–Fock and Kohn–Sham DFT, for
  closed and open shells (RHF, UHF, ROHF, RKS, UKS, ROKS);
- geometry optimization, harmonic frequencies, transition-state search
  (P-RFO) and reaction paths (IRC);
- implicit solvation (IEF-PCM, COSMO) and QM/MM embedding;
- correlated energies: RI-MP2 and its variants, CCSD, CCSD(T);
- excited states: TDA and TDDFT, BSE-TDA on G0W0;
- multireference energies: CASSCF and NEVPT2;
- periodic systems at the Gamma point and on k-point meshes;
- RPA and GW, and constrained DFT.

It plugs into other tools through the
[ASE calculator](./using/ase.md), [QCSchema](./using/qcschema.md) input and
output, and the [Avogadro plugin](./using/avogadro.md). How far each
capability is checked is graded in
[Capabilities and validation](./reference/validation.md).

```bash
pip install ferric      # Linux x86_64, Python 3.10–3.13
```

## Start here

<div class="routes">

**[Get started →](./using/quickstart.md)**
Install the wheel and compute a checked number in under a minute. Then read
the [sharp bits](./using/sharp-bits.md) that catch new users.

**[How-to guides →](./using/recipes.md)**
Task recipes: charged and open-shell molecules, optimization,
QM/MM. [Choosing a method](./using/choosing-a-method.md)
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
| Driving ferric from ASE, QCFractal or geomeTRIC | [ASE calculator](./using/ase.md), [QCSchema](./using/qcschema.md) |
| A method developer | [Architecture](./reference/architecture.md), [Rust API](./reference/api.md), [Research focus: electronic response](./idea/response.md) |
| An automated agent | [For automated agents](./using/agents.md) |

## Implemented ≠ validated

Working code is not a checked number. These pages describe what exists; how
far each capability's numbers have been checked against an independent
reference differs a lot between methods. Each one is graded individually in
**[Capabilities and validation](./reference/validation.md)**. Read it before relying on a
result.

## Maintainer's focus

ferric is maintained by Matthew Goldey. His research interests are RI-MP2,
attenuated MP2, MP2-V, constrained DFT and PDEP-RPA. Those areas get the
deepest validation and the most active development. The other areas are
validated per the grades in
[Capabilities and validation](./reference/validation.md), mostly against
PySCF, and are less exercised in research use. The
[electronic-response essay](./idea/response.md) states the research
perspective behind the focus areas.

## Source, license, citation

[github.com/mgoldey/ferric](https://github.com/mgoldey/ferric), dual-licensed
MIT / Apache-2.0. To cite ferric and the methods you used, see
[References and citing](./reference/references.md#citing-ferric).
