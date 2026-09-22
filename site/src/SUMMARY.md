# Summary

[ferric](./index.md)

# The idea

- [Electronic response](./idea/response.md)
- [Where the methods come from](./idea/methods.md)

# Using it

- [Installation](./using/installation.md)
- [Quick start](./using/quickstart.md)
- [Python bindings](./using/python.md)
- [Toxicity screening](./using/toxicity.md)
- [Recipes](./using/recipes.md)
- [Applications](./using/applications.md)
- [QM/MM](./using/qmmm.md)
- [For agents](./using/agents.md)
- [Run logs](./using/run-logs.md)

# Methods

- [Overview](./methods/index.md)
- [SCF and DFT](./methods/scf.md)
- [The MP2 family](./methods/mp2.md)
- [Coupled cluster](./methods/cc.md)
- [RPA and GW](./methods/rpa-gw.md)
- [Constrained DFT](./methods/cdft.md)

# Reference

- [Architecture](./reference/architecture.md)
- [What is validated](./reference/validation.md)
- [API documentation](./reference/api.md)
- [References](./reference/references.md)

# Measured pipeline notes

These are working notes, not a tutorial: every claim is labelled MEASURED (with
its source) or ESTIMATED (with its reasoning), and several record a RETRACTION
where a first measurement turned out to be an artifact. They are committed
because the numbers in them are expensive to reproduce.

- [Golden path: formats to docking to xtb to DFT](./reference/pipeline-golden-path.md)
- [Proposing substitutions at an active site](./reference/substitution-pipeline.md)
- [Substitution pipeline: resolved shape](./reference/substitution-pipeline-resolved.md)
- [CI timing and test sharding](./reference/ci-timing-analysis.md)
- [Pharma use-case coverage](./reference/pharma-use-case-coverage.md)
